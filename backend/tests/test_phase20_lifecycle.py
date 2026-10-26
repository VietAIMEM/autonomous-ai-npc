"""PHASE 20 process lifecycle tests.

Single-instance locking, cooperative stop and idempotent shutdown are the
foundation of "start -> run -> stop -> restart -> continue". These tests keep
every child process bounded and clean up after themselves.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from backend.runtime.config import RuntimeConfig
from backend.runtime.lifecycle import (
    DuplicateInstanceError,
    InstanceLock,
    ProcessIdentity,
    RoomAgentController,
    RuntimeLifecycle,
    TERMINATION_CLEAN_SHUTDOWN,
    TERMINATION_STARTUP_RECOVERY,
    TERMINATION_UNEXPECTED_EXIT,
    TerminationLedger,
    TerminationRecord,
    rotate_stdout_sink,
    STATE_RUNNING,
    STATE_STOPPED,
    pid_is_alive,
    read_status,
)


def make_config(directory: Path, environment: str = "test") -> RuntimeConfig:
    return RuntimeConfig(
        environment=environment,
        data_dir=directory / "data",
        log_dir=directory / "logs",
        temp_dir=directory / "temp",
        backup_dir=directory / "backups",
        state_dir=directory / "state",
    )


class PidLivenessTests(unittest.TestCase):
    """Regression: probing a pid must never signal it (Windows ``CTRL_C``)."""

    def _sleeper(self, seconds: float = 30.0) -> subprocess.Popen:
        process = subprocess.Popen(
            [sys.executable, "-c", f"import time; time.sleep({seconds})"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.addCleanup(self._terminate, process)
        return process

    def _terminate(self, process: subprocess.Popen) -> None:
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:  # pragma: no cover - defensive
            process.kill()
            process.wait(timeout=10.0)

    def test_own_process_is_alive(self) -> None:
        self.assertTrue(pid_is_alive(os.getpid()))

    def test_live_child_is_alive_then_dead_child_is_not(self) -> None:
        process = self._sleeper()
        self.assertTrue(pid_is_alive(process.pid))

        process.terminate()
        process.wait(timeout=10.0)

        self.assertFalse(pid_is_alive(process.pid))

    def test_invalid_pids_are_not_alive(self) -> None:
        self.assertFalse(pid_is_alive(0))
        self.assertFalse(pid_is_alive(-4))


class InstanceLockTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.directory = Path(self._directory.name)
        self.path = self.directory / "room_agent.lock"
        self.addCleanup(self._directory.cleanup)

    def _sleeper(self) -> subprocess.Popen:
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        def cleanup() -> None:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:  # pragma: no cover - defensive
                process.kill()
                process.wait(timeout=10.0)

        self.addCleanup(cleanup)
        return process

    def test_duplicate_acquire_is_refused_then_release_allows_reacquire(self) -> None:
        lock = InstanceLock(self.path, role="backend")
        payload = lock.acquire()
        self.assertTrue(lock.held)
        self.assertEqual(payload["pid"], os.getpid())
        self.assertTrue(self.path.is_file())

        with self.assertRaises(DuplicateInstanceError):
            InstanceLock(self.path, role="backend").acquire()

        lock.release()
        self.assertFalse(lock.held)
        self.assertFalse(self.path.exists())

        replacement = InstanceLock(self.path, role="backend")
        replacement.acquire()
        self.assertTrue(replacement.held)
        replacement.release()

    def test_dead_owner_lock_is_reclaimed_after_a_crash(self) -> None:
        process = self._sleeper()
        holder_pid = process.pid
        process.terminate()
        process.wait(timeout=10.0)
        self.path.write_text(
            '{"pid": %d, "role": "backend", "started_at": "2026-09-28T00:00:00Z"}'
            % holder_pid,
            encoding="utf-8",
        )

        lock = InstanceLock(self.path, role="backend")
        payload = lock.acquire()

        self.assertEqual(payload["pid"], os.getpid())
        self.assertEqual(lock.read_existing()["pid"], os.getpid())
        lock.release()

    def test_release_never_deletes_another_owners_file(self) -> None:
        lock = InstanceLock(self.path, role="backend")
        lock.acquire()

        process = self._sleeper()
        self.path.write_text(
            '{"pid": %d, "role": "backend"}' % process.pid, encoding="utf-8"
        )

        lock.release()

        self.assertTrue(self.path.is_file())
        self.assertEqual(lock.read_existing()["pid"], process.pid)

    def test_release_without_acquire_leaves_foreign_lock_untouched(self) -> None:
        process = self._sleeper()
        self.path.write_text(
            '{"pid": %d, "role": "backend"}' % process.pid, encoding="utf-8"
        )

        InstanceLock(self.path, role="backend").release()

        self.assertTrue(self.path.is_file())
        self.assertEqual(InstanceLock(self.path).read_existing()["pid"], process.pid)

    def test_stale_grace_period_reclaims_a_hung_owner(self) -> None:
        now = datetime(2026, 9, 29, tzinfo=timezone.utc)
        self.path.write_text(
            (
                '{"pid": %d, "role": "backend", "started_at": "%s"}'
                % (
                    os.getpid(),
                    (now - timedelta(hours=2)).isoformat().replace("+00:00", "Z"),
                )
            ),
            encoding="utf-8",
        )

        fresh = InstanceLock(self.path, role="backend", clock=lambda: now)
        with self.assertRaises(DuplicateInstanceError):
            fresh.acquire(stale_grace_seconds=60.0 * 60.0 * 24.0)
        payload = fresh.acquire(stale_grace_seconds=60.0)

        self.assertEqual(payload["pid"], os.getpid())
        fresh.release()


class LockOwnershipIdentityTests(unittest.TestCase):
    """PA29-001: reclaiming a lock must prove the recorded owner is gone.

    A PID alone is not an identity: Windows recycles PIDs, so a crash plus a
    later unrelated process landing on the same PID must not be mistaken for a
    healthy backend (it would both block restarts and, worse, let a supervisor
    signal the wrong process).
    """

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.directory = Path(self._directory.name)
        self.path = self.directory / "room_agent.lock"
        self.addCleanup(self._directory.cleanup)

    def _write_lock(self, payload: dict) -> None:
        self.path.write_text(json.dumps(payload), encoding="utf-8")

    def test_live_pid_with_a_different_creation_stamp_is_reclaimed(self) -> None:
        """PID reuse: same pid, different process start time -> stale lock."""
        self._write_lock(
            {
                "pid": os.getpid(),
                "role": "backend",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "process": {"filetime": 111_000_000_000},
            }
        )
        lock = InstanceLock(
            self.path,
            role="backend",
            identity_reader=lambda pid: ProcessIdentity(
                pid=pid, creation_filetime=222_000_000_000, image="python.exe"
            ),
        )

        payload = lock.acquire()

        self.assertEqual(payload["pid"], os.getpid())
        self.assertEqual(lock.read_existing()["process"]["filetime"], payload["process"]["filetime"])
        lock.release()
        self.assertFalse(self.path.exists())

    def test_live_pid_with_a_different_image_is_reclaimed(self) -> None:
        """Fallback signal when creation stamps are unavailable on both sides."""
        self._write_lock(
            {
                "pid": os.getpid(),
                "role": "backend",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "process": {"image": "notepad.exe"},
            }
        )
        lock = InstanceLock(
            self.path,
            role="backend",
            identity_reader=lambda pid: ProcessIdentity(pid=pid, image="python.exe"),
        )

        payload = lock.acquire()

        self.assertEqual(payload["pid"], os.getpid())
        lock.release()

    def test_live_pid_with_matching_identity_keeps_its_lock(self) -> None:
        """The central guarantee: a healthy owner's lock is never removed."""
        identity = ProcessIdentity(
            pid=os.getpid(), creation_filetime=333_000_000_000, image="python.exe"
        )
        lock = InstanceLock(
            self.path, role="backend", identity_reader=lambda _pid: identity
        )
        first = lock.acquire()
        self.assertEqual(first["process"]["filetime"], 333_000_000_000)

        contender = InstanceLock(
            self.path, role="backend", identity_reader=lambda _pid: identity
        )
        with self.assertRaises(DuplicateInstanceError):
            contender.acquire()

        self.assertTrue(self.path.is_file())
        self.assertEqual(lock.read_existing()["process"]["filetime"], 333_000_000_000)
        lock.release()
        self.assertFalse(self.path.exists())

    def test_legacy_lock_without_identity_is_conservative(self) -> None:
        """Old-format locks (no process block) must not be reclaimed while live."""
        self._write_lock(
            {
                "pid": os.getpid(),
                "role": "backend",
                "started_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        with self.assertRaises(DuplicateInstanceError):
            InstanceLock(self.path, role="backend").acquire()

        self.assertTrue(self.path.is_file())

    def test_dead_owner_is_reclaimed_even_with_a_recorded_identity(self) -> None:
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        process.terminate()
        process.wait(timeout=10.0)
        self._write_lock(
            {
                "pid": process.pid,
                "role": "backend",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "process": {"filetime": 444_000_000_000, "image": "python.exe"},
            }
        )

        lock = InstanceLock(self.path, role="backend")
        payload = lock.acquire()

        self.assertEqual(payload["pid"], os.getpid())
        lock.release()

    def test_controller_status_does_not_report_a_recycled_pid_as_running(self) -> None:
        config = make_config(self.directory)
        controller = RoomAgentController(config)
        self._write_lock(
            {
                "pid": os.getpid(),
                "role": "backend",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "process": {"filetime": 555_000_000_000, "image": "python.exe"},
            }
        )
        self.addCleanup(lambda: self.path.unlink(missing_ok=True))

        lock = InstanceLock(
            self.path,
            role="probe",
            identity_reader=lambda pid: ProcessIdentity(
                pid=pid, creation_filetime=1, image="python.exe"
            ),
        )
        live, holder = lock.holder_is_live()

        self.assertFalse(live)
        self.assertEqual(holder, os.getpid())


class RuntimeLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.directory = Path(self._directory.name)
        self.config = make_config(self.directory)
        self.addCleanup(self._directory.cleanup)

    def _lifecycle(self, **kwargs) -> RuntimeLifecycle:
        lifecycle = RuntimeLifecycle(self.config, **kwargs)
        self.addCleanup(lifecycle.shutdown)
        return lifecycle

    def test_start_stop_is_idempotent_and_releases_the_lock(self) -> None:
        lifecycle = self._lifecycle()
        lifecycle.start()
        lifecycle.mark_ready(tick_count=1)

        status = lifecycle.status()
        self.assertEqual(status["state"], STATE_RUNNING)
        self.assertTrue(status["lock_held"])
        self.assertEqual(read_status(lifecycle.status_path)["state"], STATE_RUNNING)

        first = lifecycle.shutdown(reason="test")
        self.assertFalse(first["already_stopped"])
        self.assertTrue(first["ok"])
        self.assertEqual(lifecycle.state, STATE_STOPPED)
        self.assertFalse(lifecycle.lock.held)
        self.assertFalse(self.config.instance_lock_path.exists())

        second = lifecycle.shutdown(reason="test-again")
        self.assertTrue(second["already_stopped"])
        self.assertEqual(second["steps"], first["steps"])

    def test_running_instance_refreshes_the_on_disk_snapshot(self) -> None:
        """Regression: the status file must not freeze at its startup values.

        Phase 29 found ``runtime_status.json`` still carrying the ``STARTING``
        projection from process launch hours into a healthy run, so anything
        reading the file instead of probing ``/health`` saw "STARTING" and a
        frozen uptime. The running instance now re-publishes it on a slow
        cadence through the existing stop-watcher thread.
        """
        from backend.runtime import lifecycle as lifecycle_module

        lifecycle = self._lifecycle()
        lifecycle.start()
        lifecycle.mark_ready(tick_count=3)

        first = read_status(lifecycle.status_path)
        self.assertEqual(first["state"], STATE_RUNNING)
        self.assertEqual(first["on_disk_state"], "STARTING")

        with mock.patch.object(
            lifecycle_module, "STATUS_REFRESH_INTERVAL_SECONDS", 0.05
        ), mock.patch.object(lifecycle_module, "STOP_WATCH_INTERVAL_SECONDS", 0.01):
            deadline = time.monotonic() + 5.0
            refreshed = None
            while time.monotonic() < deadline:
                candidate = read_status(lifecycle.status_path)
                if candidate and candidate.get("on_disk_state") == STATE_RUNNING:
                    refreshed = candidate
                    break
                time.sleep(0.02)

        self.assertIsNotNone(refreshed, "status file was never refreshed while running")
        self.assertEqual(refreshed["state"], STATE_RUNNING)
        self.assertGreaterEqual(refreshed["uptime_seconds"], first["uptime_seconds"])
        self.assertTrue(refreshed["lock_held"])
        self.assertEqual(refreshed["pid"], os.getpid())

    def test_shutdown_steps_run_in_reverse_order(self) -> None:
        lifecycle = self._lifecycle()
        lifecycle.start()
        order: list[str] = []
        lifecycle.register_shutdown("first", lambda: order.append("first"))
        lifecycle.register_shutdown("second", lambda: order.append("second"))
        lifecycle.register_shutdown("third", lambda: order.append("third"))

        report = lifecycle.shutdown()

        self.assertEqual(order, ["third", "second", "first"])
        self.assertEqual([step["step"] for step in report["steps"]], order)
        self.assertTrue(report["ok"])
        for step in report["steps"]:
            self.assertTrue(step["ok"])

    def test_failing_shutdown_step_is_isolated(self) -> None:
        lifecycle = self._lifecycle()
        lifecycle.start()
        order: list[str] = []

        def explode() -> None:
            order.append("explode")
            raise RuntimeError("provider refused to stop")

        lifecycle.register_shutdown("closing", lambda: order.append("closing"))
        lifecycle.register_shutdown("provider", explode)

        report = lifecycle.shutdown()

        self.assertEqual(order, ["explode", "closing"])
        self.assertFalse(report["ok"])
        self.assertEqual(report["steps"][0]["step"], "provider")
        self.assertFalse(report["steps"][0]["ok"])
        self.assertIn("RuntimeError", report["steps"][0]["error"])
        self.assertEqual(report["steps"][1]["step"], "closing")
        self.assertTrue(report["steps"][1]["ok"])
        self.assertEqual(lifecycle.state, STATE_STOPPED)
        self.assertFalse(self.config.instance_lock_path.exists())

    def test_duplicate_start_is_refused_not_silently_duplicated(self) -> None:
        lifecycle = self._lifecycle()
        lifecycle.start()

        with self.assertRaises(DuplicateInstanceError):
            RuntimeLifecycle(self.config, role="backend").start()

        lifecycle.shutdown()

    def test_stop_file_watcher_triggers_the_cooperative_callback(self) -> None:
        lifecycle = self._lifecycle()
        fired = threading.Event()
        lifecycle.on_stop_request(fired.set)
        lifecycle.start()

        lifecycle.request_stop()

        self.assertTrue(fired.wait(timeout=8.0))
        self.assertTrue(lifecycle.stop_request_path.is_file())
        lifecycle.clear_stop_request()
        lifecycle.shutdown()
        self.assertFalse(lifecycle.stop_request_path.exists())


class TerminationLedgerTests(unittest.TestCase):
    """PA29-002: the next run must be able to judge how the last one ended."""

    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.directory = Path(self._directory.name)
        self.config = make_config(self.directory)
        self.config.state_dir.mkdir(parents=True, exist_ok=True)
        self.addCleanup(self._directory.cleanup)
        self.status_path = self.config.state_dir / "runtime_status.json"
        self.ledger = TerminationLedger(
            self.config.state_dir / "runtime_termination.json",
            status_path=self.status_path,
        )

    def test_clean_shutdown_is_recorded_and_reconciled(self) -> None:
        lifecycle = RuntimeLifecycle(self.config)
        self.addCleanup(lifecycle.shutdown)
        lifecycle.start()

        report = lifecycle.shutdown(reason="test-clean")

        self.assertTrue(report["ok"])
        record = self.ledger.read()
        self.assertEqual(record["classification"], TERMINATION_CLEAN_SHUTDOWN)
        self.assertTrue(record["conclusive"])
        self.assertEqual(record["reason"], "test-clean")
        self.assertEqual(record["timestamp"][-1], "Z")
        on_disk = read_status(self.status_path)
        self.assertTrue(on_disk["stopped_cleanly"])
        self.assertEqual(on_disk["state"], STATE_STOPPED)

        second = RuntimeLifecycle(self.config)
        self.addCleanup(second.shutdown)
        second.start()
        self.assertEqual(
            second.last_termination["classification"], TERMINATION_CLEAN_SHUTDOWN
        )
        self.assertTrue(second.last_termination["conclusive"])
        # Reconciliation rewrites the ledger at startup; the stop reason from
        # the status snapshot must survive that rewrite.
        self.assertEqual(second.last_termination["reason"], "test-clean")
        self.assertEqual(self.ledger.read()["reason"], "test-clean")

    def test_reconciled_clean_shutdown_keeps_top_level_reason(self) -> None:
        """A legacy snapshot stores the reason at the top level, not in shutdown.

        Live evidence 2026-10-16: the pre-remediation build wrote
        ``{"state": "STOPPED", "reason": "stop_request", "shutdown": {...}}``
        with no ``stopped_cleanly`` key; reconciliation must still classify it
        as a clean shutdown *and* carry the recorded reason forward.
        """
        self.status_path.write_text(
            json.dumps(
                {
                    "state": STATE_STOPPED,
                    "reason": "stop_request",
                    "pid": 500,
                    "instance_id": "500-legacy",
                    "shutdown": {
                        "ok": True,
                        "started_at": "2026-10-15T20:13:29Z",
                        "finished_at": "2026-10-15T20:13:31Z",
                    },
                }
            ),
            encoding="utf-8",
        )

        record = self.ledger.reconcile(instance_id="16512-new")

        self.assertEqual(record["classification"], TERMINATION_CLEAN_SHUTDOWN)
        self.assertTrue(record["conclusive"])
        self.assertFalse(record["inferred"])
        self.assertEqual(record["reason"], "stop_request")
        self.assertEqual(record["previous_pid"], 500)

    def test_unclean_prior_run_is_detected_at_startup(self) -> None:
        """A leftover RUNNING snapshot with no clean record is an unclean exit."""
        self.status_path.write_text(
            json.dumps(
                {
                    "state": STATE_RUNNING,
                    "pid": 4321,
                    "instance_id": "4321-old",
                    "started_at": "2026-10-15T00:00:00Z",
                }
            ),
            encoding="utf-8",
        )

        lifecycle = RuntimeLifecycle(self.config)
        self.addCleanup(lifecycle.shutdown)
        lifecycle.start()

        record = lifecycle.last_termination
        self.assertEqual(record["classification"], TERMINATION_UNEXPECTED_EXIT)
        self.assertTrue(record["conclusive"])
        self.assertTrue(record["inferred"])
        self.assertEqual(record["previous_pid"], 4321)
        self.assertEqual(self.ledger.read()["classification"], TERMINATION_UNEXPECTED_EXIT)

    def test_missing_prior_status_is_an_explicit_unknown(self) -> None:
        lifecycle = RuntimeLifecycle(self.config)
        self.addCleanup(lifecycle.shutdown)
        lifecycle.start()

        record = lifecycle.last_termination
        self.assertEqual(record["classification"], "unknown")
        self.assertFalse(record["conclusive"])
        self.assertTrue(record["inferred"])
        self.assertEqual(record["detail"], "no_previous_status")

    def test_legacy_status_without_new_fields_still_reconciles(self) -> None:
        """Legacy snapshots have no ``stopped_cleanly``/``shutdown`` keys."""
        self.status_path.write_text(
            json.dumps(
                {
                    "state": STATE_STOPPED,
                    "pid": 999,
                    "instance_id": "999-legacy",
                }
            ),
            encoding="utf-8",
        )

        record = self.ledger.reconcile(instance_id="new")

        self.assertEqual(record["classification"], TERMINATION_STARTUP_RECOVERY)
        self.assertFalse(record["conclusive"])
        self.assertIsNone(record["reason"])

    def test_malformed_ledger_file_is_tolerated(self) -> None:
        self.ledger.path.write_text("{not json", encoding="utf-8")
        self.assertIsNone(self.ledger.read())

        lifecycle = RuntimeLifecycle(self.config)
        self.addCleanup(lifecycle.shutdown)
        lifecycle.start()
        self.assertEqual(lifecycle.last_termination["classification"], "unknown")

    def test_record_round_trip_keeps_missing_fields_null(self) -> None:
        recorded = self.ledger.write(
            TerminationRecord(
                timestamp="2026-10-15T00:00:00Z",
                classification=TERMINATION_UNEXPECTED_EXIT,
                conclusive=False,
                previous_pid=17,
            )
        )

        self.assertIsNone(recorded["reason"])
        self.assertIsNone(recorded["exit_code"])
        self.assertEqual(self.ledger.read()["previous_pid"], 17)


class RoomAgentControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.directory = Path(self._directory.name)
        self.config = make_config(self.directory)
        self.controller = RoomAgentController(self.config)
        self.addCleanup(self._directory.cleanup)

    def test_status_without_a_lock_reports_stopped(self) -> None:
        status = self.controller.status(probe=False)

        self.assertFalse(status["running"])
        self.assertIsNone(status["pid"])
        self.assertFalse(status["stop_requested"])

    def test_status_reads_the_instance_lock_without_probing_http(self) -> None:
        lock = InstanceLock(self.config.instance_lock_path, role="backend")
        lock.acquire()
        self.addCleanup(lock.release)

        status = self.controller.status(probe=False)

        self.assertTrue(status["running"])
        self.assertEqual(status["pid"], os.getpid())
        self.assertEqual(status["lock"]["role"], "backend")

    def test_stop_without_a_running_instance_is_already_stopped(self) -> None:
        result = self.controller.stop(timeout=1.0)

        self.assertTrue(result["stopped"])
        self.assertTrue(result["already_stopped"])

    def test_start_refuses_when_a_live_instance_owns_the_lock(self) -> None:
        lock = InstanceLock(self.config.instance_lock_path, role="backend")
        lock.acquire()
        self.addCleanup(lock.release)

        with self.assertRaises(DuplicateInstanceError):
            self.controller.start(wait_seconds=0.5)

        # The refused start must not have disturbed the live owner's lock.
        self.assertTrue(self.config.instance_lock_path.is_file())
        self.assertEqual(
            InstanceLock(self.config.instance_lock_path).read_existing()["pid"],
            os.getpid(),
        )

    def test_stdout_sink_rotates_once_it_exceeds_the_bound(self) -> None:
        """Phase 29: the append-only child sink must not grow without bound.

        Retention cannot reclaim the sink while the backend holds it open, so
        a long-running instance accumulated output forever. The sink is now
        rotated at start into a single previous file.
        """
        sink = self.config.log_dir / "room_agent.out.log"
        sink.parent.mkdir(parents=True, exist_ok=True)
        sink.write_bytes(b"x" * 128)

        self.assertIsNone(rotate_stdout_sink(sink, max_bytes=256))
        self.assertTrue(sink.is_file())

        rotated = rotate_stdout_sink(sink, max_bytes=64)
        self.assertEqual(rotated, "room_agent.out.log.1")
        self.assertFalse(sink.exists())
        self.assertEqual((self.config.log_dir / "room_agent.out.log.1").stat().st_size, 128)

    def test_start_rotates_an_oversized_sink_before_spawning(self) -> None:
        sink = self.config.log_dir / "room_agent.out.log"
        sink.parent.mkdir(parents=True, exist_ok=True)
        sink.write_bytes(b"y" * 64)

        class _FinishedProcess:
            pid = 4321

            def poll(self) -> int:
                return 1

        recorded: dict[str, object] = {}

        def fake_popen(*args, **kwargs):
            recorded["stdout"] = kwargs.get("stdout")
            recorded["cmd"] = args[0]
            return _FinishedProcess()

        with mock.patch("backend.runtime.lifecycle.subprocess.Popen", fake_popen), mock.patch(
            "backend.runtime.lifecycle.STDOUT_SINK_MAX_BYTES", 32
        ):
            result = self.controller.start(wait_seconds=0.1)

        self.assertEqual(result["log_rotated"], "room_agent.out.log.1")
        self.assertEqual(result["log"], "room_agent.out.log")
        self.assertEqual(result["pid"], 4321)
        self.assertEqual((self.config.log_dir / "room_agent.out.log.1").stat().st_size, 64)
        self.assertTrue(recorded.get("cmd"))


if __name__ == "__main__":  # pragma: no cover - manual entry point
    unittest.main()
