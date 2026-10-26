"""PA29-001 closure: RoomAgent is operator-controlled, not self-recovering."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from backend.runtime.supervisor import (
    ACTION_NONE,
    AUTO_RECOVERY_ENABLED,
    DEFAULT_STATE_NAME,
    observe,
    run_lock,
)


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "AgentScripts" / "phase23" / "start_runtime_phase23.ps1"
INSTALLER = ROOT / "AgentScripts" / "production_audit" / "install_backend_supervisor.ps1"
WRAPPER = ROOT / "AgentScripts" / "production_audit" / "run_backend_supervisor_once.ps1"
HIDDEN_WRAPPER = ROOT / "AgentScripts" / "production_audit" / "run_backend_supervisor_once.pyw"
SUPERVISOR = ROOT / "backend" / "runtime" / "supervisor.py"


def _config(directory: Path):
    from backend.runtime.config import RuntimeConfig

    return RuntimeConfig(
        environment="test",
        data_dir=directory / "data",
        log_dir=directory / "logs",
        temp_dir=directory / "temp",
        backup_dir=directory / "backups",
        state_dir=directory / "state",
    )


def _probe(reachable: bool):
    return lambda _config: (
        {"reachable": True, "status": "READY", "ready": True}
        if reachable
        else {"reachable": False, "error": "connection refused"}
    )


def _controller(running: bool, *, pid: int | None = None):
    return SimpleNamespace(
        status=lambda *, probe: {
            "running": running,
            "pid": pid,
            "lock": None,
            "status_file": {},
            "stop_requested": False,
        }
    )


class ManualModeSupervisorTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.directory = Path(self._tmp.name)
        self.config = _config(self.directory)
        self.config.state_dir.mkdir(parents=True, exist_ok=True)
        self.addCleanup(self._tmp.cleanup)

    def _state_path(self) -> Path:
        return self.config.state_dir / DEFAULT_STATE_NAME

    def test_stopped_backend_is_not_started(self) -> None:
        payload = observe(
            self.config,
            controller=_controller(False),
            probe=_probe(False),
        )

        self.assertFalse(payload["running"])
        self.assertFalse(payload["health_reachable"])
        self.assertEqual(payload["action"], ACTION_NONE)
        self.assertEqual(
            payload["reason"], "backend_stopped_until_operator_starts_it"
        )
        self.assertFalse(payload["auto_recovery_enabled"])

    def test_unhealthy_backend_is_not_restarted(self) -> None:
        payload = observe(
            self.config,
            controller=_controller(True, pid=os.getpid()),
            probe=_probe(False),
        )

        self.assertTrue(payload["running"])
        self.assertFalse(payload["health_reachable"])
        self.assertEqual(payload["action"], ACTION_NONE)
        self.assertEqual(
            payload["reason"],
            "unhealthy_backend_left_running_by_operator_policy",
        )

    def test_healthy_backend_is_reported_without_lifecycle_change(self) -> None:
        payload = observe(
            self.config,
            controller=_controller(True, pid=os.getpid()),
            probe=_probe(True),
        )

        self.assertTrue(payload["running"])
        self.assertTrue(payload["health_reachable"])
        self.assertEqual(payload["action"], ACTION_NONE)
        self.assertEqual(payload["reason"], "healthy")

    def test_observation_preserves_historical_recovery_ledger(self) -> None:
        state_path = self._state_path()
        historical = {
            "last_action": "start",
            "recoveries": [{"pid": 14692, "at": "2026-10-15T23:46:22Z"}],
        }
        state_path.write_text(json.dumps(historical), encoding="utf-8")

        observe(self.config, controller=_controller(False), probe=_probe(False))

        self.assertEqual(json.loads(state_path.read_text(encoding="utf-8")), historical)

    def test_run_lock_is_exclusive_and_released(self) -> None:
        path = self.config.state_dir / "supervisor.lock"
        with run_lock(path) as first:
            self.assertTrue(first)
            with run_lock(path) as second:
                self.assertFalse(second)
        self.assertFalse(path.exists())


class ManualModePolicyTests(unittest.TestCase):
    def test_supervisor_has_no_launch_runner_or_recovery_actions(self) -> None:
        text = SUPERVISOR.read_text(encoding="utf-8")
        self.assertIn("AUTO_RECOVERY_ENABLED = False", text)
        self.assertNotIn("ACTION_START", text)
        self.assertNotIn("ACTION_RESTART", text)
        self.assertNotIn("subprocess", text)
        self.assertNotIn("_run_launcher", text)
        self.assertNotIn("Register-ScheduledTask", text)

    def test_manual_launcher_does_not_implicitly_launch_a_supervisor(self) -> None:
        text = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("python -m backend.runtime.lifecycle $Action", text)
        for forbidden in ("supervisor", "install_backend_supervisor"):
            self.assertNotIn(forbidden, text.casefold())

    def test_task_registration_is_no_longer_supported(self) -> None:
        text = INSTALLER.read_text(encoding="utf-8")
        self.assertIn("registration allowed     : no", text)
        self.assertNotIn("Register-ScheduledTask", text)
        self.assertNotIn("New-ScheduledTaskAction", text)
        self.assertNotIn("New-ScheduledTaskTrigger", text)
        self.assertIn("Unregister-ScheduledTask", text)

    def test_what_if_only_never_registers_a_task(self) -> None:
        completed = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(INSTALLER),
                "-WhatIfOnly",
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("registration allowed     : no", completed.stdout)
        self.assertNotIn("registered:", completed.stdout)

    def test_roomagent_supervisor_task_is_absent(self) -> None:
        completed = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "if (Get-ScheduledTask -TaskName 'RoomAgentBackendSupervisor' "
                "-ErrorAction SilentlyContinue) { 'present' } else { 'absent' }",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "absent")


class ManualModeWrapperTests(unittest.TestCase):
    def test_console_wrapper_defaults_to_observation(self) -> None:
        text = WRAPPER.read_text(encoding="utf-8")
        self.assertIn("'observe'", text)
        self.assertNotIn("else { 'tick' }", text)
        self.assertIn("never resolves or executes it", text)

    def test_console_wrapper_run_is_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env = dict(os.environ)
            env["ROOM_AGENT_STATE_DIR"] = directory
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "backend.runtime.supervisor",
                    "observe",
                    "--json",
                ],
                cwd=str(ROOT),
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(completed.stdout.splitlines()[-1])
            self.assertEqual(payload["action"], ACTION_NONE)
            self.assertFalse(payload["auto_recovery_enabled"])
            self.assertTrue((Path(directory) / "supervisor_status.json").is_file())

    def test_hidden_wrapper_defaults_to_observation(self) -> None:
        text = HIDDEN_WRAPPER.read_text(encoding="utf-8")
        self.assertIn('"observe", "--json"', text)
        self.assertNotIn('"tick"', text)
        self.assertIn("automatic backend launch is disabled", text)


if __name__ == "__main__":
    unittest.main()
