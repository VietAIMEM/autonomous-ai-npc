"""PA29-005 / PA29-009: the audit runner must be traceable and secret-free.

The post-Phase-29 audit produced a pytest log with no command line and no
record of which LLM provider/model the run used. These tests pin the harness
that fixes that: every run records the exact command, exit code, timestamps and
summary, plus the effective runtime configuration read from the environment
(never hard-coded), and no secret value can reach the evidence file.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = ROOT / "AgentScripts" / "production_audit" / "tools" / "audit_test_run.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "production_audit_test_run", TOOL_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


tool = _load_tool()


class RedactionTests(unittest.TestCase):
    def test_secret_looking_keys_are_replaced_recursively(self) -> None:
        payload = {
            "provider": "nvidia",
            "api_key_path": "api_key_nvidia.txt",
            "nested": [
                {"credentials": "abc", "model": "m"},
                {"chat_twitch_token": "t"},
            ],
        }
        cleaned = tool.redact_structure(payload)
        self.assertEqual(cleaned["provider"], "nvidia")
        self.assertEqual(cleaned["api_key_path"], "[redacted]")
        self.assertEqual(cleaned["nested"][0]["credentials"], "[redacted]")
        self.assertEqual(cleaned["nested"][0]["model"], "m")
        self.assertEqual(cleaned["nested"][1]["chat_twitch_token"], "[redacted]")

    def test_environment_provenance_names_secrets_without_reading_them(self) -> None:
        environ = {
            "LLM_PROVIDER": "nvidia",
            "LLM_MODEL": "example/model-under-test",
            "LLM_API_KEY_PATH": "C:/secret/keys/nvidia.txt",
            "SOME_SERVICE_TOKEN": "sup3r-secret-value",
        }
        record = tool.environment_provenance(environ)
        serialized = json.dumps(record, indent=2)
        self.assertEqual(record["values"]["LLM_MODEL"], "example/model-under-test")
        self.assertNotIn("sup3r-secret-value", serialized)
        self.assertNotIn("C:/secret/keys/nvidia.txt", serialized)
        self.assertIn("SOME_SERVICE_TOKEN", record["secret_names_present"])
        self.assertFalse(record["secrets_recorded"])


class SummaryTests(unittest.TestCase):
    def test_pytest_summary_lines_are_parsed(self) -> None:
        summary = tool.parse_summary("....\n2 failed, 988 passed, 1 warning in 61.20s\n")
        self.assertEqual(summary["counts"]["failed"], 2)
        self.assertEqual(summary["counts"]["passed"], 988)
        self.assertEqual(summary["counts"]["warning"], 1)
        self.assertIn("988 passed", summary["tail"])

    def test_no_tests_ran_is_not_a_silent_pass(self) -> None:
        summary = tool.parse_summary("no tests ran in 0.01s\n")
        self.assertEqual(summary["counts"], {"no_tests_ran": 1})


class EffectiveConfigurationTests(unittest.TestCase):
    def test_record_reads_the_effective_model_instead_of_hard_coding_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            key_path = Path(tmp) / "api_key_nvidia.txt"
            key_path.write_text("fake-key-value-for-test\n", encoding="utf-8")
            environ = {
                "LLM_PROVIDER": "nvidia",
                "LLM_MODEL": "example/under-test-model",
                "LLM_API_KEY_PATH": str(key_path),
            }
            record = tool.build_record(
                label="unit",
                command=["python", "-m", "pytest", "-q"],
                exit_code=0,
                started=datetime(2026, 10, 16, tzinfo=timezone.utc),
                ended=datetime(2026, 10, 16, tzinfo=timezone.utc) + timedelta(seconds=4),
                output_log=Path(tmp) / "run.log",
                stdout="1 passed in 4.00s\n",
                stderr="",
                environ=environ,
                live={"reachable": False, "error": "URLError"},
            )
        llm = record["runtime"]["process_env_llm"]
        self.assertEqual(llm["provider"], "nvidia")
        self.assertEqual(llm["model"], "example/under-test-model")
        self.assertTrue(llm["configured"])
        # Even the key *file name* is scrubbed in evidence: the record proves
        # which provider/model was effective, not where its key lives.
        self.assertEqual(llm["api_key_path"], "[redacted]")
        serialized = json.dumps(record, indent=2)
        self.assertNotIn("fake-key-value-for-test", serialized)
        self.assertEqual(record["run"]["summary"]["counts"]["passed"], 1)
        self.assertEqual(record["run"]["duration_seconds"], 4.0)
        self.assertEqual(record["run"]["exit_code"], 0)
        self.assertEqual(
            record["config_provenance"]["secret_names_present"].count(
                "LLM_API_KEY_PATH"
            ),
            1,
        )

    def test_config_sources_are_hashed_not_copied(self) -> None:
        records = {item["path"]: item for item in tool.config_sources()}
        launcher = records["AgentScripts/phase23/start_runtime_phase23.ps1"]
        self.assertTrue(launcher["exists"])
        self.assertEqual(len(launcher["sha256"]), 64)
        self.assertNotIn("LLM_MODEL", json.dumps(records))


class CliTests(unittest.TestCase):
    def test_end_to_end_run_writes_metadata_and_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "audit_run_unit.json"
            script = "print('3 passed in 0.05s')"
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("ROOM_AGENT_ENV", None)
                exit_code = tool.main(
                    [
                        "--label",
                        "unit",
                        "--out",
                        str(out),
                        "--no-live",
                        "--",
                        sys.executable,
                        "-c",
                        script,
                    ]
                )
            self.assertEqual(exit_code, 0)
            record = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(record["run"]["exit_code"], 0)
            self.assertEqual(record["run"]["summary"]["counts"]["passed"], 3)
            self.assertTrue(record["run"]["started_utc"].endswith("Z"))
            self.assertTrue(record["run"]["ended_utc"].endswith("Z"))
            self.assertEqual(record["run"]["flags"], ["-c", script])
            log_path = Path(tmp) / "audit_run_unit.log"
            self.assertTrue(log_path.is_file())
            log = log_path.read_text(encoding="utf-8")
            self.assertIn("3 passed in 0.05s", log)
            self.assertIn("# command:", log)

    def test_failing_command_exit_code_is_propagated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "audit_run_fail.json"
            exit_code = tool.main(
                [
                    "--label",
                    "unit-fail",
                    "--out",
                    str(out),
                    "--no-live",
                    "--",
                    sys.executable,
                    "-c",
                    "import sys; print('1 failed, 2 passed in 0.10s'); sys.exit(1)",
                ]
            )
            self.assertEqual(exit_code, 1)
            record = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(record["run"]["exit_code"], 1)
            self.assertEqual(record["run"]["summary"]["counts"]["failed"], 1)

    def test_production_environment_is_refused_unless_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "audit_run_prod.json"
            with mock.patch.dict(os.environ, {"ROOM_AGENT_ENV": "production"}):
                exit_code = tool.main(
                    [
                        "--label",
                        "unit-prod",
                        "--out",
                        str(out),
                        "--no-live",
                        "--",
                        sys.executable,
                        "-c",
                        "print('0 passed in 0.01s')",
                    ]
                )
            self.assertEqual(exit_code, 2)
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
