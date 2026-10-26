"""PA29-001 closure: supervisor wrappers are diagnostic-only."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INSTALL_SCRIPT = ROOT / "AgentScripts" / "production_audit" / "install_backend_supervisor.ps1"
WRAPPER_SCRIPT = ROOT / "AgentScripts" / "production_audit" / "run_backend_supervisor_once.ps1"
HIDDEN_WRAPPER_SCRIPT = ROOT / "AgentScripts" / "production_audit" / "run_backend_supervisor_once.pyw"


class InstallerPolicyTests(unittest.TestCase):
    def test_registration_path_is_absent(self) -> None:
        text = INSTALL_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("registration allowed     : no", text)
        self.assertNotIn("Register-ScheduledTask", text)
        self.assertNotIn("New-ScheduledTaskAction", text)
        self.assertNotIn("New-ScheduledTaskTrigger", text)
        self.assertIn("use -Uninstall to remove the legacy task", text)

    def test_uninstall_is_ownership_checked(self) -> None:
        text = INSTALL_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("run_backend_supervisor_once.pyw", text)
        self.assertIn("run_backend_supervisor_once.ps1", text)
        self.assertIn("refusing to remove an unknown task name", text)
        self.assertIn("does not reference the RoomAgent supervisor wrapper", text)

    def test_what_if_only_reports_manual_policy(self) -> None:
        completed = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(INSTALL_SCRIPT),
                "-WhatIfOnly",
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("manual startup; no automatic recovery", completed.stdout)
        self.assertIn("registration allowed     : no", completed.stdout)


class WrapperPolicyTests(unittest.TestCase):
    def test_console_wrapper_is_observe_only(self) -> None:
        text = WRAPPER_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("'observe'", text)
        self.assertNotIn("'tick'", text)
        self.assertNotIn("start_runtime_phase23", text)
        self.assertIn("never resolves or executes it", text)

    def test_hidden_wrapper_is_observe_only(self) -> None:
        text = HIDDEN_WRAPPER_SCRIPT.read_text(encoding="utf-8")
        self.assertIn('"observe", "--json"', text)
        self.assertNotIn('"tick"', text)
        self.assertIn("automatic backend launch is disabled", text)


class WrapperRuntimeTests(unittest.TestCase):
    def test_console_wrapper_reports_stopped_backend_in_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env = dict(os.environ)
            env["ROOM_AGENT_STATE_DIR"] = directory
            completed = subprocess.run(
                [
                    sys.executable,
                    str(HIDDEN_WRAPPER_SCRIPT),
                    "--observe",
                ],
                cwd=str(ROOT),
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(completed.stdout.splitlines()[-1])
            self.assertEqual(payload["action"], "none")
            self.assertFalse(payload["running"])
            self.assertFalse(payload["auto_recovery_enabled"])
            self.assertTrue((Path(directory) / "supervisor_status.json").is_file())


if __name__ == "__main__":
    unittest.main()
