"""Focused tests for the backend process supervisor and exit-code capture.

Everything here is offline. The supervisor is exercised against dedicated
dummy children only; it never imports, launches, or signals
``backend.api.brain`` or any production process.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from backend.runtime.config import RuntimeConfig
from backend.runtime.lifecycle import (
    CLASSIFICATION_ABRUPT_OR_UNKNOWN,
    CLASSIFICATION_CLEAN_EXIT,
    CLASSIFICATION_NONZERO_EXIT,
    CLASSIFICATION_SUPERVISOR_ERROR,
    SUPERVISOR_VERSION,
    DuplicateInstanceError,
    RoomAgentController,
    classify_child_exit,
    default_backend_command,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = (
    REPO_ROOT
    / "AgentScripts"
    / "production_audit"
    / "evidence"
    / "start_runtime_ollama_qwen35_4b_numgpu.ps1"
)
#: A port nothing listens on, so the readiness probe never touches the real
#: production brain port while a dummy child runs.
IDLE_PROBE_PORT = 18799


def _python_child(code: str) -> list[str]:
    return [sys.executable, "-c", code]


def _read_events(path: Path) -> list[dict[str, Any]]:
    """Parse every JSONL line; a malformed line fails the test."""
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


class SupervisorHarness(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory(prefix="roomagent_supervisor_test_")
        self.tmp = Path(self._tmp.name)
        self.config = RuntimeConfig(
            environment="test",
            brain_port=IDLE_PROBE_PORT,
            chat_port=IDLE_PROBE_PORT + 1,
            viewer_port=IDLE_PROBE_PORT + 2,
            admin_port=IDLE_PROBE_PORT + 3,
            data_dir=self.tmp / "data",
            log_dir=self.tmp / "logs",
            state_dir=self.tmp / "state",
        )
        self.config.log_dir.mkdir(parents=True, exist_ok=True)
        self.config.state_dir.mkdir(parents=True, exist_ok=True)
        self.events = self.tmp / "supervisor.jsonl"
        self.controller = RoomAgentController(self.config)

    def tearDown(self) -> None:
        self._tmp.cleanup()


class ClassificationTests(SupervisorHarness):
    def test_classification_never_invents_an_exit_code(self) -> None:
        self.assertEqual(classify_child_exit(None, True), CLASSIFICATION_ABRUPT_OR_UNKNOWN)
        self.assertEqual(classify_child_exit(0, True), CLASSIFICATION_CLEAN_EXIT)
        self.assertEqual(classify_child_exit(0, False), CLASSIFICATION_ABRUPT_OR_UNKNOWN)
        self.assertEqual(classify_child_exit(9, True), CLASSIFICATION_NONZERO_EXIT)

    def test_default_command_is_the_unchanged_backend_entrypoint(self) -> None:
        self.assertEqual(
            default_backend_command(), [sys.executable, "-m", "backend.api.brain"]
        )


class ExitCaptureTests(SupervisorHarness):
    def test_dummy_child_exit_zero_with_shutdown_evidence_is_clean(self) -> None:
        status_path = self.config.state_dir / "runtime_status.json"
        code = (
            "import json, sys\n"
            f"open({str(status_path)!r}, 'w', encoding='utf-8').write("
            "json.dumps({'state': 'STOPPED'}))\n"
            "sys.exit(0)\n"
        )
        result = self.controller.supervise(
            wait_seconds=1.0,
            run_id="run-clean",
            events_path=self.events,
            command=_python_child(code),
        )
        self.assertEqual(result["exit_code"], 0)
        self.assertTrue(result["shutdown_evidence"])
        self.assertEqual(result["classification"], CLASSIFICATION_CLEAN_EXIT)

    def test_dummy_child_exit_zero_without_shutdown_evidence_is_not_clean(self) -> None:
        result = self.controller.supervise(
            wait_seconds=1.0,
            run_id="run-no-evidence",
            events_path=self.events,
            command=_python_child("import sys\nsys.exit(0)\n"),
        )
        self.assertEqual(result["exit_code"], 0)
        self.assertFalse(result["shutdown_evidence"])
        self.assertEqual(result["classification"], CLASSIFICATION_ABRUPT_OR_UNKNOWN)

    def test_dummy_child_nonzero_exit_code_is_captured(self) -> None:
        result = self.controller.supervise(
            wait_seconds=1.0,
            run_id="run-nonzero",
            events_path=self.events,
            command=_python_child("import sys\nsys.exit(3)\n"),
        )
        self.assertEqual(result["exit_code"], 3)
        self.assertEqual(result["classification"], CLASSIFICATION_NONZERO_EXIT)
        self.assertFalse(result["shutdown_evidence"])

    def test_real_child_code_is_recorded_not_the_supervisor_result(self) -> None:
        """The wrapper itself always "succeeds"; only the child code is real."""
        result = self.controller.supervise(
            wait_seconds=1.0,
            run_id="run-code7",
            events_path=self.events,
            command=_python_child("import sys\nsys.exit(7)\n"),
        )
        self.assertEqual(result["exit_code"], 7)
        self.assertEqual(result["classification"], CLASSIFICATION_NONZERO_EXIT)

    def test_start_end_timestamps_and_child_pid_are_recorded(self) -> None:
        result = self.controller.supervise(
            wait_seconds=1.0,
            run_id="run-identity",
            events_path=self.events,
            command=_python_child("import sys\nsys.exit(0)\n"),
        )
        self.assertGreater(int(result["backend_pid"]), 0)
        self.assertNotEqual(int(result["backend_pid"]), os.getpid())
        self.assertLessEqual(result["started_at"], result["ended_at"])
        self.assertGreaterEqual(float(result["elapsed_seconds"]), 0.0)

        records = _read_events(self.events)
        start = next(r for r in records if r["event"] == "supervisor.start")
        spawned = next(r for r in records if r["event"] == "supervisor.spawned")
        exit_record = next(r for r in records if r["event"] == "supervisor.exit")
        self.assertEqual(int(start["parent_pid"]), os.getpid())
        self.assertEqual(int(spawned["backend_pid"]), int(result["backend_pid"]))
        self.assertEqual(int(exit_record["backend_pid"]), int(result["backend_pid"]))
        self.assertEqual(exit_record["exit_code"], result["exit_code"])
        self.assertEqual(start["command"], [sys.executable, "-c", "import sys\nsys.exit(0)\n"])
        self.assertEqual(start["automatic_restart"], False)

    def test_interrupted_test_child_is_never_reported_clean(self) -> None:
        pid_file = self.tmp / "child_pid.txt"
        code = (
            "import os, time\n"
            f"open({str(pid_file)!r}, 'w', encoding='utf-8').write(str(os.getpid()))\n"
            "time.sleep(30)\n"
        )
        run_id = "run-interrupted"
        holder: dict[str, Any] = {}

        def run() -> None:
            holder["result"] = self.controller.supervise(
                wait_seconds=0.5,
                run_id=run_id,
                events_path=self.events,
                command=_python_child(code),
            )

        thread = threading.Thread(target=run, name="supervise-interrupted", daemon=True)
        thread.start()

        pid = self._await_own_child_pid(run_id, pid_file, timeout=30.0)
        self.assertIsNotNone(pid, "dummy child never reported its pid")
        try:
            os.kill(int(pid), signal.SIGTERM)
        except OSError as exc:  # pragma: no cover - only on a test-host failure
            self.fail(f"could not interrupt the test child: {exc}")

        thread.join(timeout=30.0)
        self.assertFalse(thread.is_alive(), "supervisor did not return after the child died")
        result = holder["result"]
        self.assertNotEqual(result["classification"], CLASSIFICATION_CLEAN_EXIT)
        self.assertIn(
            result["classification"],
            {CLASSIFICATION_NONZERO_EXIT, CLASSIFICATION_ABRUPT_OR_UNKNOWN},
        )
        self.assertNotEqual(result["exit_code"], 0)

    def test_supervisor_cannot_record_is_reported_and_child_not_started(self) -> None:
        blocker = self.tmp / "blocker"
        blocker.write_text("not a directory", encoding="utf-8")
        marker = self.tmp / "child_ran.txt"
        code = f"open({str(marker)!r}, 'w', encoding='utf-8').write('ran')\n"
        result = self.controller.supervise(
            wait_seconds=0.5,
            run_id="run-sink-failure",
            events_path=blocker / "nested" / "supervisor.jsonl",
            command=_python_child(code),
        )
        self.assertEqual(result["classification"], CLASSIFICATION_SUPERVISOR_ERROR)
        self.assertIsNone(result["exit_code"])
        self.assertFalse(marker.exists(), "child must not run without a supervisor sink")

    def test_capture_failure_is_recorded_as_supervisor_error(self) -> None:
        child = _python_child("import time\ntime.sleep(30)\n")
        original = RoomAgentController._capture_exit
        spawned: dict[str, Any] = {}

        def boom(self: RoomAgentController, process: Any) -> int:  # noqa: ARG001
            spawned["process"] = process
            raise RuntimeError("capture exploded")

        RoomAgentController._capture_exit = boom  # type: ignore[assignment]
        try:
            result = self.controller.supervise(
                wait_seconds=1.0,
                run_id="run-capture-failure",
                events_path=self.events,
                command=child,
            )
        finally:
            RoomAgentController._capture_exit = original  # type: ignore[assignment]
            process = spawned.get("process")
            if process is not None:
                # Test-owned dummy only; never a production process.
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:  # pragma: no cover - defensive
                    process.kill()
                    process.wait(timeout=10)

        self.assertEqual(result["classification"], CLASSIFICATION_SUPERVISOR_ERROR)
        self.assertIsNone(result["exit_code"])
        records = _read_events(self.events)
        errors = [r for r in records if r["event"] == "supervisor.error"]
        self.assertTrue(any(r.get("stage") == "capture_exit" for r in errors))
        self.assertFalse(
            any(r["event"] == "supervisor.exit" for r in records),
            "a capture failure must not write a clean exit record",
        )

    def test_structured_log_stays_parseable_across_multiple_runs(self) -> None:
        child = _python_child("import sys\nsys.exit(0)\n")
        for run_id in ("run-a", "run-b"):
            self.controller.supervise(
                wait_seconds=0.5,
                run_id=run_id,
                events_path=self.events,
                command=child,
            )
        records = _read_events(self.events)
        self.assertEqual({r["run_id"] for r in records}, {"run-a", "run-b"})
        for record in records:
            self.assertIn("event", record)
            self.assertIn("at", record)
            self.assertEqual(record["supervisor_version"], SUPERVISOR_VERSION)

    # --------------------------------------------------------------- helpers

    def _await_own_child_pid(
        self, run_id: str, pid_file: Path, *, timeout: float
    ) -> int | None:
        """Return the dummy child's pid, proven by its own pid file."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.events.exists():
                for record in _read_events(self.events):
                    if (
                        record.get("event") == "supervisor.spawned"
                        and record.get("run_id") == run_id
                        and record.get("backend_pid")
                    ):
                        pid = int(record["backend_pid"])
                        if pid_file.exists():
                            try:
                                written = int(pid_file.read_text(encoding="utf-8").strip())
                            except (OSError, ValueError):
                                written = None
                            if written == pid:
                                return pid
            time.sleep(0.1)
        return None


class ProductionCompatibilityTests(SupervisorHarness):
    """Read-only proofs that supervision cannot disturb the live backend."""

    def test_supervisor_writes_only_its_own_event_log(self) -> None:
        result = self.controller.supervise(
            wait_seconds=0.5,
            run_id="run-artifacts",
            command=_python_child("import sys\nsys.exit(0)\n"),
        )
        self.assertEqual(
            Path(result["events_path"]),
            self.config.log_dir / "room_agent.supervisor.jsonl",
        )
        # log_dir holds the backend stdout sink plus the supervisor stream.
        self.assertEqual(
            sorted(p.name for p in self.config.log_dir.iterdir()),
            ["room_agent.out.log", "room_agent.supervisor.jsonl"],
        )
        # The process supervisor never publishes a health/observation status.
        self.assertEqual(sorted(p.name for p in self.config.state_dir.iterdir()), [])

    def test_stale_running_status_does_not_block_or_report_clean(self) -> None:
        stale = self.config.state_dir / "runtime_status.json"
        stale.write_text(
            json.dumps({"state": "RUNNING", "pid": 29400, "instance_id": "29400-x"}),
            encoding="utf-8",
        )
        result = self.controller.supervise(
            wait_seconds=0.5,
            run_id="run-stale-status",
            events_path=self.events,
            command=_python_child("import sys\nsys.exit(0)\n"),
        )
        # A stale status file is not liveness evidence, so the child still ran.
        self.assertGreater(int(result["backend_pid"]), 0)
        self.assertFalse(result["shutdown_evidence"])
        self.assertEqual(result["classification"], CLASSIFICATION_ABRUPT_OR_UNKNOWN)

    def test_live_lock_still_blocks_a_second_launch(self) -> None:
        self.config.instance_lock_path.write_text(
            json.dumps(
                {
                    "role": "backend",
                    "pid": os.getpid(),
                    "started_at": "2026-10-24T00:00:00Z",
                }
            ),
            encoding="utf-8",
        )
        marker = self.tmp / "second_child.txt"
        with self.assertRaises(DuplicateInstanceError):
            self.controller.supervise(
                wait_seconds=0.5,
                run_id="run-duplicate",
                events_path=self.events,
                command=_python_child(
                    f"open({str(marker)!r}, 'w', encoding='utf-8').write('ran')\n"
                ),
            )
        self.assertFalse(marker.exists(), "duplicate launch must not spawn a child")

    def test_readiness_failure_never_terminates_the_child(self) -> None:
        result = self.controller.supervise(
            wait_seconds=0.5,
            run_id="run-not-ready",
            events_path=self.events,
            command=_python_child("import time, sys\ntime.sleep(1.0)\nsys.exit(5)\n"),
        )
        records = _read_events(self.events)
        ready = next(r for r in records if r["event"] == "supervisor.ready")
        self.assertFalse(ready["ready"], "idle probe port must report not-ready")
        # The child still ran to its own exit code: no kill on probe failure.
        self.assertEqual(result["exit_code"], 5)
        self.assertEqual(
            len([r for r in records if r["event"] == "supervisor.spawned"]), 1
        )
        self.assertEqual(
            len([r for r in records if r["event"] == "supervisor.exit"]), 1
        )


class LauncherCompatibilityTests(unittest.TestCase):
    def test_documented_launcher_keeps_original_arguments_and_config(self) -> None:
        text = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("[int]$NumGpu = 24", text)
        self.assertIn("python -m backend.runtime.lifecycle start --wait 25", text)
        self.assertIn("python -m backend.runtime.lifecycle supervise --wait 25", text)
        self.assertIn("$env:LLM_MODEL = 'qwen3.5:4b'", text)
        self.assertIn('$env:LLM_NUM_GPU = "$NumGpu"', text)
        self.assertIn("$env:LLM_NUM_CTX = '8192'", text)
        # Default (no -Supervise) must stay the original fire-and-forget start.
        self.assertIn("if ($Supervise) {", text)
        self.assertLess(
            text.index("if ($Supervise) {"),
            text.index("python -m backend.runtime.lifecycle supervise --wait 25"),
        )

    def test_cli_exposes_supervise_without_launching_anything(self) -> None:
        proc = subprocess.run(
            [sys.executable, "-m", "backend.runtime.lifecycle", "--help"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            timeout=60,
        )
        self.assertEqual(proc.returncode, 0)
        self.assertIn("supervise", proc.stdout)


if __name__ == "__main__":  # pragma: no cover - manual entry point
    unittest.main()
