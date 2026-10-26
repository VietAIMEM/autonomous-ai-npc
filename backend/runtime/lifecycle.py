"""Process lifecycle for the single production instance.

One backend owns one Brain loop, one chat ingest runner and one streaming
transport. This module enforces that with a file lock, records the instance
status on disk, and runs shutdown callbacks in a fixed, idempotent order.

Stopping is cooperative: ``stop`` writes a request file that the running
process watches, so a shutdown is always graceful on Windows (where
``SIGTERM`` is a hard terminate). A hard signal is only a bounded last resort.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from .config import PROJECT_ROOT, RuntimeConfig
from .obs import log_event
from .redaction import safe_error


LOGGER = logging.getLogger("room_agent.lifecycle")

STATE_STARTING = "STARTING"
STATE_RUNNING = "RUNNING"
STATE_STOPPING = "STOPPING"
STATE_STOPPED = "STOPPED"

#: Termination classifications persisted by :class:`TerminationLedger`.
TERMINATION_CLEAN_SHUTDOWN = "clean_shutdown"
TERMINATION_UNEXPECTED_EXIT = "unexpected_process_exit"
TERMINATION_STARTUP_RECOVERY = "startup_recovery_after_previous_termination"
TERMINATION_UNKNOWN = "unknown"

STOP_WATCH_INTERVAL_SECONDS = 0.5
#: How often the running instance re-publishes ``runtime_status.json``.
#: Without this the on-disk file keeps the values written at startup and an
#: operator reading it sees a frozen ``uptime_seconds`` and a stale
#: ``on_disk_state`` long after ``/health`` has moved on.
STATUS_REFRESH_INTERVAL_SECONDS = 30.0
DEFAULT_STOP_TIMEOUT_SECONDS = 20.0

#: The child process' stdout/stderr sink is an append-only file, so retention
#: can never reclaim it while the backend runs (an open sink cannot be
#: unlinked on Windows). Phase 29 measured steady growth at ~35 KB/h with no
#: upper bound, so the sink is rotated once at start instead: at most two
#: files of this size exist, and the older one is a regular ``*.log`` that the
#: existing retention sweep already owns.
STDOUT_SINK_MAX_BYTES = 4 * 1024 * 1024

#: Bumped when the supervisor event schema changes incompatibly.
SUPERVISOR_VERSION = "1"

#: Supervisor outcome for the monitored child. ``CLEAN_EXIT`` additionally
#: requires evidence that the documented shutdown path ran; a bare ``exit 0``
#: from a child that never unwound is reported as ``ABRUPT_OR_UNKNOWN`` so an
#: unexplained disappearance is never recorded as a clean stop.
CLASSIFICATION_CLEAN_EXIT = "CLEAN_EXIT"
CLASSIFICATION_NONZERO_EXIT = "NONZERO_EXIT"
CLASSIFICATION_ABRUPT_OR_UNKNOWN = "ABRUPT_OR_UNKNOWN"
CLASSIFICATION_SUPERVISOR_ERROR = "SUPERVISOR_ERROR"

#: Filename of the append-only supervisor event stream inside ``log_dir``.
SUPERVISOR_LOG_NAME = "room_agent.supervisor.jsonl"


def default_backend_command() -> list[str]:
    """The exact argv the documented launcher starts the backend with."""
    return [sys.executable, "-m", "backend.api.brain"]


def classify_child_exit(
    exit_code: Optional[int], shutdown_evidence: bool
) -> str:
    """Map an observed child exit to a supervisor classification.

    ``exit_code`` is ``None`` only when the operating system never reported
    one; the classification then stays ``ABRUPT_OR_UNKNOWN`` rather than
    inventing a code or actor.
    """
    if exit_code is None:
        return CLASSIFICATION_ABRUPT_OR_UNKNOWN
    try:
        code = int(exit_code)
    except (TypeError, ValueError):
        return CLASSIFICATION_ABRUPT_OR_UNKNOWN
    if code != 0:
        return CLASSIFICATION_NONZERO_EXIT
    if shutdown_evidence:
        return CLASSIFICATION_CLEAN_EXIT
    return CLASSIFICATION_ABRUPT_OR_UNKNOWN


def _read_json_document(path: Path) -> Optional[dict[str, Any]]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _new_supervisor_run_id() -> str:
    stamp = _utc_now().strftime("%Y%m%dT%H%M%S%fZ")
    return f"sup_{stamp}_{os.getpid()}"


class SupervisorEventLog:
    """Append-only JSONL sink for one supervised run.

    Every event is flushed and fsynced before :meth:`emit` returns, so even a
    hard termination of the supervisor leaves the most recent record readable
    on disk. Construction raises ``OSError`` when the sink cannot be opened;
    the caller must surface that as ``SUPERVISOR_ERROR`` instead of assuming a
    clean child exit.
    """

    def __init__(self, path: Path, run_id: str) -> None:
        self.path = Path(path)
        self.run_id = run_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = open(self.path, "a", encoding="utf-8")

    def emit(self, event: str, **fields: Any) -> dict[str, Any]:
        record: dict[str, Any] = {
            "event": event,
            "run_id": self.run_id,
            "supervisor_version": SUPERVISOR_VERSION,
            "at": _iso(_utc_now()),
        }
        record.update(fields)
        self._stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._stream.flush()
        try:
            os.fsync(self._stream.fileno())
        except OSError:  # pragma: no cover - fsync is best effort
            pass
        return record

    def close(self) -> None:
        try:
            self._stream.close()
        except OSError:  # pragma: no cover - defensive
            pass

#: ``OpenProcess`` rights needed to read an exit code, nothing more.
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
#: Windows reports this exit code while a process is still running.
_STILL_ACTIVE = 259
_ERROR_ACCESS_DENIED = 5


class DuplicateInstanceError(RuntimeError):
    """Raised when a live instance already owns the lock."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def rotate_stdout_sink(
    log_path: Path, *, max_bytes: Optional[int] = None
) -> Optional[str]:
    """Keep the append-only child stdout sink bounded to one previous file.

    Returns the rotated file name, or ``None`` when nothing needed to move.
    A rotation failure is never fatal: the caller still opens the sink and the
    operator keeps a growing-but-working log instead of no log at all.
    """
    limit = STDOUT_SINK_MAX_BYTES if max_bytes is None else int(max_bytes)
    path = Path(log_path)
    try:
        if not path.is_file() or path.stat().st_size < limit:
            return None
    except OSError:
        return None
    rotated = path.with_name(f"{path.name}.1")
    try:
        os.replace(path, rotated)
    except OSError as exc:
        LOGGER.warning("stdout sink rotation skipped: %s", safe_error(exc, limit=120))
        return None
    return rotated.name


def _iso(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def pid_is_alive(pid: int) -> bool:
    """Best-effort liveness check that never signals the target.

    On Windows ``os.kill(pid, 0)`` is *not* a probe: ``0`` is
    ``signal.CTRL_C_EVENT``, so it generates a console control event instead of
    checking liveness, which can interrupt every process sharing the console -
    including the running backend the caller is only trying to inspect. The
    Windows path therefore opens the process with
    ``PROCESS_QUERY_LIMITED_INFORMATION`` and reads its exit code.
    """
    if pid <= 0:
        return False
    if os.name == "nt":
        return _windows_pid_is_alive(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    except Exception:  # pragma: no cover - defensive
        return False
    return True


def _windows_pid_is_alive(pid: int) -> bool:
    """Open the process and read its exit code; never sends a signal."""
    try:
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.OpenProcess.argtypes = (
            ctypes.c_uint32,
            ctypes.c_int,
            ctypes.c_uint32,
        )
        kernel32.GetExitCodeProcess.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32),
        )
        kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
        handle = kernel32.OpenProcess(
            _PROCESS_QUERY_LIMITED_INFORMATION, 0, int(pid)
        )
        if not handle:
            # Access denied means the process exists but is not ours; anything
            # else (invalid parameter) means no such process.
            return ctypes.get_last_error() == _ERROR_ACCESS_DENIED
        try:
            code = ctypes.c_uint32(0)
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                # Cannot tell: assume alive so a live lock is never reclaimed.
                return True
            return int(code.value) == _STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    except Exception:  # pragma: no cover - defensive
        return True


def _windows_process_creation_filetime(pid: int) -> Optional[int]:
    """Return the Win32 100ns FILETIME creation stamp, or ``None``."""
    try:
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.OpenProcess.argtypes = (
            ctypes.c_uint32,
            ctypes.c_int,
            ctypes.c_uint32,
        )
        kernel32.GetProcessTimes.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32 * 2),
            ctypes.POINTER(ctypes.c_uint32 * 2),
            ctypes.POINTER(ctypes.c_uint32 * 2),
            ctypes.POINTER(ctypes.c_uint32 * 2),
        )
        kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
        handle = kernel32.OpenProcess(
            _PROCESS_QUERY_LIMITED_INFORMATION, 0, int(pid)
        )
        if not handle:
            return None
        try:
            creation = (ctypes.c_uint32 * 2)()
            exit_time = (ctypes.c_uint32 * 2)()
            kernel = (ctypes.c_uint32 * 2)()
            user = (ctypes.c_uint32 * 2)()
            ok = kernel32.GetProcessTimes(
                handle,
                ctypes.byref(creation),
                ctypes.byref(exit_time),
                ctypes.byref(kernel),
                ctypes.byref(user),
            )
            if not ok:
                return None
            return (int(creation[1]) << 32) | int(creation[0])
        finally:
            kernel32.CloseHandle(handle)
    except Exception:  # pragma: no cover - defensive
        return None


@dataclass(frozen=True)
class ProcessIdentity:
    """Cheap, non-intrusive identity for one running process.

    ``creation_filetime`` is the Win32 process creation stamp (100 ns ticks
    since 1601) and never changes for the life of a process, so a recycled PID
    compares unequal. ``image`` (the executable's base name) is the portable
    fallback where creation time is unavailable.
    """

    pid: int
    creation_filetime: Optional[int] = None
    image: Optional[str] = None
    executable: Optional[str] = None

    def as_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"filetime": self.creation_filetime}
        if self.image:
            payload["image"] = self.image
        if self.executable:
            payload["executable"] = self.executable
        return payload


def read_process_identity(pid: int) -> Optional[ProcessIdentity]:
    """Best-effort identity for ``pid``; never signals or blocks on it.

    Uses ``psutil`` when available and falls back to the Win32 API. An
    unreadable identity returns ``None``, and every caller treats that as
    "cannot prove this is a different process" - i.e. conservative.
    """
    if pid <= 0:
        return None
    if os.name != "nt":  # pragma: no cover - exercised only on Windows
        try:
            import psutil

            process = psutil.Process(pid)
            return ProcessIdentity(
                pid=pid,
                creation_filetime=None,
                image=process.name() or None,
                executable=None,
            )
        except Exception:
            return None
    creation: Optional[int] = None
    image: Optional[str] = None
    executable: Optional[str] = None
    try:
        import psutil

        process = psutil.Process(pid)
        created = process.create_time()
        if created:
            creation = int(created * 10_000_000) + 116_444_736_000_000_000
        image = process.name() or None
        try:
            executable = process.exe() or None
        except Exception:  # pragma: no cover - access denied
            executable = None
    except Exception:
        pass
    if creation is None:
        creation = _windows_process_creation_filetime(pid)
    return ProcessIdentity(
        pid=pid,
        creation_filetime=creation,
        image=image,
        executable=executable,
    )


def _identity_differs(
    current: Optional[ProcessIdentity], recorded: Any
) -> bool:
    """True when ``recorded`` provably belongs to a *different* process.

    Only strong evidence counts: a different creation stamp, or (when the
    creation stamp is unavailable on both sides) a different executable image.
    Missing data returns ``False`` so an unreadable or legacy lock is never
    reclaimed while its recorded PID is still alive.
    """
    if current is None or not isinstance(recorded, dict):
        return False
    recorded_creation = recorded.get("filetime")
    if current.creation_filetime and isinstance(recorded_creation, int):
        return int(recorded_creation) != int(current.creation_filetime)
    recorded_image = recorded.get("image")
    if current.image and isinstance(recorded_image, str) and recorded_image:
        return current.image.lower() != recorded_image.lower()
    return False


@dataclass
class InstanceLock:
    """An exclusive lock file holding ``{pid, started_at, role, process}``.

    The ``process`` block records a process-identity stamp so a stale lock is
    only reclaimed after proving the recorded PID is gone *or* that the PID
    now belongs to a different process. A live PID that cannot be proven
    different is always treated as the rightful owner.
    """

    path: Path
    role: str = "backend"
    clock: Callable[[], datetime] = _utc_now
    identity_reader: Callable[[int], Optional[ProcessIdentity]] = (
        read_process_identity
    )

    _payload: Optional[dict[str, Any]] = field(default=None, init=False, repr=False)

    @property
    def held(self) -> bool:
        return self._payload is not None

    def read_existing(self) -> Optional[dict[str, Any]]:
        try:
            raw = self.path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    def acquire(self, *, stale_grace_seconds: float = 0.0) -> dict[str, Any]:
        """Take the lock, reclaiming it when the recorded process is gone."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = self._new_payload()
        try:
            handle = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            existing = self.read_existing()
            holder = int(existing.get("pid", 0)) if existing else 0
            if self._current_holder_is_live(existing, holder) and not (
                stale_grace_seconds and self._age_seconds(existing) > stale_grace_seconds
            ):
                raise DuplicateInstanceError(
                    f"instance_already_running:{holder}"
                ) from None
            # Stale lock: the previous owner is gone (crash), the recorded PID
            # now belongs to a different process, or the file is junk.
            log_event(
                LOGGER,
                "lifecycle.stale_lock_reclaimed",
                lock=str(self.path.name),
                previous_pid=holder or None,
                previous_process=(existing or {}).get("process"),
                pid_reused=self._pid_reused(existing, holder),
            )
            try:
                self.path.unlink()
            except OSError:
                pass
            try:
                handle = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError as exc:
                raise DuplicateInstanceError("instance_lock_race") from exc
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, sort_keys=True)
            stream.flush()
        self._payload = payload
        return payload

    def holder_is_live(
        self, payload: Optional[dict[str, Any]] = None
    ) -> tuple[bool, int]:
        """Whether the lock's recorded owner is still that same live process."""
        existing = self.read_existing() if payload is None else payload
        holder = int(existing.get("pid", 0)) if existing else 0
        return self._current_holder_is_live(existing, holder), holder

    def _current_holder_is_live(
        self, existing: Optional[dict[str, Any]], holder: int
    ) -> bool:
        if existing is None or holder <= 0 or not pid_is_alive(holder):
            return False
        if _identity_differs(self.identity_reader(holder), existing.get("process")):
            return False
        return True

    def _pid_reused(
        self, existing: Optional[dict[str, Any]], holder: int
    ) -> bool:
        if existing is None or holder <= 0:
            return False
        if not pid_is_alive(holder):
            return False
        return _identity_differs(self.identity_reader(holder), existing.get("process"))

    def release(self) -> None:
        payload = self._payload
        self._payload = None
        if payload is None:
            # This object never owned the lock. Never delete a file another
            # (possibly live) instance may own - a failed start must not
            # disturb the instance it collided with.
            return
        existing = self.read_existing()
        if existing is not None:
            if int(existing.get("pid", 0)) != int(payload.get("pid", -1)):
                return  # someone else took the lock; never delete their file
            recorded = existing.get("process")
            mine = payload.get("process")
            if isinstance(recorded, dict) and isinstance(mine, dict):
                if _identity_differs(
                    self.identity_reader(int(payload.get("pid", 0))), recorded
                ):
                    return  # the file belongs to a different process now
        try:
            self.path.unlink()
        except OSError:
            pass

    def _new_payload(self) -> dict[str, Any]:
        payload = {
            "pid": os.getpid(),
            "role": self.role,
            "started_at": _iso(self.clock()),
            "host": os.environ.get("COMPUTERNAME") or os.environ.get("HOSTNAME") or "",
        }
        identity = self.identity_reader(os.getpid())
        if identity is not None:
            payload["process"] = identity.as_payload()
        return payload

    def _age_seconds(self, payload: dict[str, Any]) -> float:
        started = payload.get("started_at")
        if not isinstance(started, str):
            return 0.0
        try:
            moment = datetime.fromisoformat(started.replace("Z", "+00:00"))
        except ValueError:
            return 0.0
        return max(0.0, (self.clock() - moment).total_seconds())


@dataclass
class ShutdownReport:
    started_at: str
    finished_at: str
    duration_ms: float
    already_stopped: bool
    steps: list[dict[str, Any]]

    @property
    def ok(self) -> bool:
        return all(step.get("ok", False) for step in self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "already_stopped": self.already_stopped,
            "ok": self.ok,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_ms": round(self.duration_ms, 3),
            "steps": self.steps,
        }


class RuntimeLifecycle:
    """Owns the instance lock, the on-disk status file and shutdown ordering."""

    def __init__(
        self,
        config: RuntimeConfig,
        *,
        role: str = "backend",
        clock: Callable[[], datetime] = _utc_now,
        instance_id: Optional[str] = None,
    ) -> None:
        self.config = config
        self.role = role
        self._clock = clock
        self.instance_id = instance_id or f"{os.getpid()}-{int(time.time())}"
        self.lock = InstanceLock(config.instance_lock_path, role=role, clock=clock)
        self.status_path = config.state_dir / "runtime_status.json"
        self.stop_request_path = config.state_dir / "room_agent.stop"
        self.termination_path = config.state_dir / "runtime_termination.json"
        self.termination = TerminationLedger(
            self.termination_path, clock=clock, status_path=self.status_path
        )
        self.last_termination: Optional[dict[str, Any]] = None
        self.started_at = clock()
        self.state = STATE_STOPPED
        self._steps: list[tuple[str, Callable[[], Any], float]] = []
        self._extra: dict[str, Any] = {}
        self._shutdown_report: Optional[ShutdownReport] = None
        self._lock_guard = threading.RLock()
        self._watcher: Optional[threading.Thread] = None
        self._watcher_stop = threading.Event()
        self._shutdown_started = threading.Event()
        self._shutdown_finished = threading.Event()
        self._on_stop_request: Optional[Callable[[], None]] = None

    # ------------------------------------------------------------ lifecycle

    def start(self, *, watch_stop_file: bool = True) -> dict[str, Any]:
        """Acquire the instance lock and publish ``STARTING`` status."""
        with self._lock_guard:
            if self.state == STATE_RUNNING:
                return self.status()
            payload = self.lock.acquire()
            self.last_termination = self.termination.reconcile(
                instance_id=self.instance_id
            )
            self.state = STATE_STARTING
            self.started_at = self._clock()
            self._write_status()
            if watch_stop_file:
                self._start_stop_watcher()
            log_event(
                LOGGER,
                "lifecycle.started",
                role=self.role,
                pid=payload.get("pid"),
                instance_id=self.instance_id,
                environment=self.config.environment,
            )
            return self.status()

    def mark_ready(self, **extra: Any) -> dict[str, Any]:
        with self._lock_guard:
            self.state = STATE_RUNNING
            self._extra.update(extra)
            self._write_status()
            log_event(LOGGER, "lifecycle.ready", role=self.role, **self._safe_extra())
            return self.status()

    def update(self, **extra: Any) -> None:
        """Record extra status fields without touching the state."""
        with self._lock_guard:
            self._extra.update(extra)

    def register_shutdown(
        self,
        name: str,
        callback: Callable[[], Any],
        *,
        timeout_seconds: float = 5.0,
    ) -> None:
        """Register a shutdown step; steps run in reverse registration order."""
        self._steps.append((name, callback, max(0.1, float(timeout_seconds))))

    def on_stop_request(self, callback: Callable[[], None]) -> None:
        """Handler invoked when a cooperative stop file appears."""
        self._on_stop_request = callback

    def shutdown(
        self, reason: str = "shutdown", *, wait_seconds: float = 25.0
    ) -> dict[str, Any]:
        """Run every shutdown step once, in reverse order, and release the lock.

        Calling this again while a shutdown is already running never starts a
        second pass: the caller waits (bounded) for the first one and reuses its
        report, so repeated ``stop``/``restart`` requests stay idempotent.
        """
        with self._lock_guard:
            if self._shutdown_report is not None:
                report = self._shutdown_report.to_dict()
                report["already_stopped"] = True
                return report
            already_running = self._shutdown_started.is_set()
            if not already_running:
                self.state = STATE_STOPPING
                self._write_status()
                self._shutdown_started.set()

        if already_running:
            self._shutdown_finished.wait(timeout=max(0.1, float(wait_seconds)))
            with self._lock_guard:
                finished = self._shutdown_report
            if finished is None:
                # Still inside the bounded budget: report the in-progress state
                # instead of pretending the process shut down cleanly.
                return {
                    "already_stopped": False,
                    "in_progress": True,
                    "ok": False,
                    "steps": [],
                    "reason": reason,
                }
            payload = finished.to_dict()
            payload["already_stopped"] = True
            return payload

        started = time.monotonic()
        start_iso = _iso(self._clock())
        steps: list[dict[str, Any]] = []
        try:
            for name, callback, timeout in reversed(self._steps):
                step_started = time.monotonic()
                ok = True
                detail: Optional[str] = None
                try:
                    callback()
                except Exception as exc:  # every step is isolated on purpose
                    ok = False
                    detail = safe_error(exc, limit=200)
                    log_event(
                        LOGGER,
                        "lifecycle.shutdown_step_failed",
                        role=self.role,
                        step=name,
                        error=detail,
                    )
                elapsed_ms = (time.monotonic() - step_started) * 1000.0
                steps.append(
                    {
                        "step": name,
                        "ok": ok,
                        "budget_seconds": timeout,
                        "duration_ms": round(elapsed_ms, 3),
                        "error": detail,
                    }
                )
        finally:
            self._stop_watcher()
            with self._lock_guard:
                self.state = STATE_STOPPED
                report = ShutdownReport(
                    started_at=start_iso or "",
                    finished_at=_iso(self._clock()) or "",
                    duration_ms=(time.monotonic() - started) * 1000.0,
                    already_stopped=False,
                    steps=steps,
                )
                self._shutdown_report = report
                # Finish the status snapshot and the termination record *before*
                # the lock is released: the next instance reads both, and a
                # half-written record would look like an unclean exit.
                self._write_status(
                    reason=reason,
                    shutdown=report.to_dict(),
                    stopped_cleanly=True,
                )
                self.last_termination = self.termination.record_clean_shutdown(
                    reason=reason, instance_id=self.instance_id
                )
                self.lock.release()
            self._shutdown_finished.set()
        log_event(
            LOGGER,
            "lifecycle.stopped",
            role=self.role,
            reason=reason,
            ok=report.ok,
            duration_ms=round(report.duration_ms, 3),
        )
        return report.to_dict()

    @property
    def stopping(self) -> bool:
        return self._shutdown_started.is_set()

    # --------------------------------------------------------------- status

    def status(self) -> dict[str, Any]:
        on_disk = read_status(self.status_path) or {}
        with self._lock_guard:
            payload = {
                "role": self.role,
                "state": self.state,
                "pid": os.getpid(),
                "instance_id": self.instance_id,
                "environment": self.config.environment,
                "host": self.config.host,
                "brain_port": self.config.brain_port,
                "chat_port": self.config.chat_port,
                "started_at": _iso(self.started_at),
                "uptime_seconds": round(
                    max(0.0, (self._clock() - self.started_at).total_seconds()), 3
                ),
                "lock_held": self.lock.held,
                "stop_requested": self.stop_request_path.exists(),
                "shutdown": self._shutdown_report.to_dict() if self._shutdown_report else None,
                "last_termination": self.last_termination,
            }
            payload.update(self._safe_extra())
        payload["on_disk_state"] = on_disk.get("state")
        return payload

    def _write_status(self, **extra: Any) -> None:
        payload = self.status()
        payload.update(extra)
        write_status(self.status_path, payload)

    def _safe_extra(self) -> dict[str, Any]:
        return dict(self._extra)

    # --------------------------------------------------------------- stop file

    def request_stop(self) -> Path:
        """Ask the running instance to shut down (cooperative, cross-platform)."""
        self.stop_request_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "requested_at": _iso(self._clock()),
            "requested_by_pid": os.getpid(),
        }
        _atomic_write(self.stop_request_path, json.dumps(payload, sort_keys=True))
        return self.stop_request_path

    def clear_stop_request(self) -> None:
        try:
            self.stop_request_path.unlink()
        except OSError:
            pass

    def _start_stop_watcher(self) -> None:
        self.clear_stop_request()

        def watch() -> None:
            last_refresh = time.monotonic()
            while not self._watcher_stop.wait(STOP_WATCH_INTERVAL_SECONDS):
                if self._shutdown_started.is_set():
                    return
                # Keep the on-disk snapshot honest while the instance runs.
                now = time.monotonic()
                if (
                    now - last_refresh >= STATUS_REFRESH_INTERVAL_SECONDS
                    and self.state in (STATE_STARTING, STATE_RUNNING)
                ):
                    last_refresh = now
                    try:
                        self._write_status()
                    except Exception as exc:  # pragma: no cover - defensive
                        log_event(
                            LOGGER,
                            "lifecycle.status_refresh_failed",
                            error=safe_error(exc, limit=120),
                        )
                if self.stop_request_path.exists():
                    log_event(LOGGER, "lifecycle.stop_requested", role=self.role)
                    on_request = self._on_stop_request
                    if callable(on_request):
                        try:
                            on_request()
                        except Exception as exc:  # pragma: no cover - defensive
                            log_event(
                                LOGGER,
                                "lifecycle.stop_request_handler_failed",
                                error=safe_error(exc, limit=120),
                            )
                    return

        self._watcher_stop.clear()
        self._watcher = threading.Thread(target=watch, name="lifecycle-watch", daemon=True)
        self._watcher.start()

    def _stop_watcher(self) -> None:
        self._watcher_stop.set()
        watcher = self._watcher
        self._watcher = None
        if watcher is not None and watcher.is_alive():
            watcher.join(timeout=1.0)

    # --------------------------------------------------------------- signals

    def install_signal_handlers(self, on_signal: Callable[[str], None]) -> bool:
        """Install SIGINT/SIGTERM handlers; returns False off the main thread."""
        if threading.current_thread() is not threading.main_thread():
            return False

        def handler(signum, _frame):  # pragma: no cover - exercised by hand
            name = signal.Signals(signum).name
            log_event(LOGGER, "lifecycle.signal", signal=name)
            on_signal(name)

        for name in ("SIGINT", "SIGTERM"):
            value = getattr(signal, name, None)
            if value is None:
                continue
            try:
                signal.signal(value, handler)
            except (ValueError, OSError):  # pragma: no cover - platform dependent
                continue
        return True


def write_status(path: Path, payload: dict[str, Any]) -> None:
    """Atomically publish the status file; never raises, never blocks."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(path, json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str))
    except OSError:
        pass


def read_status(path: Path) -> Optional[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _atomic_write(path: Path, text: str) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    with open(temporary, "w", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


@dataclass
class TerminationRecord:
    """One persisted termination observation for the current instance.

    ``conclusive`` distinguishes what the process *observed* (a shutdown it
    ran itself, or a live PID it started alongside) from what it merely
    *inferred* from stale on-disk state. Python cannot write anything after an
    abrupt kill or a power loss, so an inference is always labelled as such
    rather than presented as a recorded crash.
    """

    timestamp: str
    classification: str
    reason: Optional[str] = None
    conclusive: bool = False
    exit_code: Optional[int] = None
    previous_pid: Optional[int] = None
    instance_id: Optional[str] = None
    source: str = "runtime"
    inferred: bool = False
    detail: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "classification": self.classification,
            "reason": self.reason,
            "conclusive": bool(self.conclusive),
            "exit_code": self.exit_code,
            "previous_pid": self.previous_pid,
            "instance_id": self.instance_id,
            "source": self.source,
            "inferred": bool(self.inferred),
            "detail": self.detail,
        }


class TerminationLedger:
    """Persist the previous run's termination so the next run can judge it."""

    def __init__(
        self,
        path: Path,
        *,
        clock: Callable[[], datetime] = _utc_now,
        status_path: Optional[Path] = None,
        stale_after_seconds: float = 90.0,
    ) -> None:
        self.path = path
        self.status_path = status_path
        self.stale_after_seconds = float(stale_after_seconds)
        self._clock = clock

    def read(self) -> Optional[dict[str, Any]]:
        """Return the last record, or ``None`` when there is none/malformed.

        A malformed or legacy file is never an error: consumers must keep
        working with missing or null termination data.
        """
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    def write(self, record: TerminationRecord) -> dict[str, Any]:
        payload = record.to_dict()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write(
                self.path,
                json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str),
            )
        except OSError:
            # Losing the diagnostic never blocks a start or a stop.
            pass
        return payload

    def record_clean_shutdown(
        self, *, reason: str, instance_id: Optional[str] = None
    ) -> dict[str, Any]:
        return self.write(
            TerminationRecord(
                timestamp=_iso(self._clock()) or "",
                classification=TERMINATION_CLEAN_SHUTDOWN,
                reason=reason,
                conclusive=True,
                instance_id=instance_id,
                source="shutdown",
            )
        )

    def reconcile(self, *, instance_id: Optional[str] = None) -> dict[str, Any]:
        """Classify how the previous run ended and persist the observation.

        A previous run that wrote a clean-shutdown record is authoritative.
        Otherwise an on-disk ``RUNNING``/``STARTING`` state whose PID is still
        alive proves the process died without unwinding; anything else is an
        inferred recovery from stale state.
        """
        previous = read_status(self.status_path) if self.status_path else None
        if previous is None:
            return self.write(
                TerminationRecord(
                    timestamp=_iso(self._clock()) or "",
                    classification=TERMINATION_UNKNOWN,
                    conclusive=False,
                    instance_id=instance_id,
                    source="startup",
                    inferred=True,
                    detail="no_previous_status",
                )
            )
        prior_state = str(previous.get("state") or "")
        prior_pid = int(previous.get("pid") or 0)
        prior_instance = previous.get("instance_id")
        clean = bool(previous.get("stopped_cleanly")) or (
            prior_state == STATE_STOPPED and previous.get("shutdown") is not None
        )
        if clean:
            shutdown = previous.get("shutdown") or {}
            # The status snapshot keeps the stop reason at the top level; a
            # legacy snapshot may instead carry it inside the shutdown block.
            return self.write(
                TerminationRecord(
                    timestamp=_iso(self._clock()) or "",
                    classification=TERMINATION_CLEAN_SHUTDOWN,
                    reason=str(
                        shutdown.get("reason") or previous.get("reason") or ""
                    ),
                    conclusive=True,
                    previous_pid=prior_pid or None,
                    instance_id=instance_id,
                    source="startup",
                    detail="previous_run_recorded_a_clean_shutdown",
                )
            )
        if prior_state in (STATE_RUNNING, STATE_STARTING, STATE_STOPPING):
            return self.write(
                TerminationRecord(
                    timestamp=_iso(self._clock()) or "",
                    classification=TERMINATION_UNEXPECTED_EXIT,
                    conclusive=True,
                    previous_pid=prior_pid or None,
                    instance_id=instance_id,
                    source="startup",
                    inferred=True,
                    detail=f"previous_state={prior_state}",
                )
            )
        return self.write(
            TerminationRecord(
                timestamp=_iso(self._clock()) or "",
                classification=TERMINATION_STARTUP_RECOVERY,
                conclusive=False,
                previous_pid=prior_pid or None,
                instance_id=instance_id or prior_instance,
                source="startup",
                inferred=True,
                detail=f"previous_state={prior_state or 'missing'}",
            )
        )


# ------------------------------------------------------------------ controller


class RoomAgentController:
    """start/stop/restart/status/health for the single production instance."""

    def __init__(self, config: Optional[RuntimeConfig] = None) -> None:
        self.config = config or RuntimeConfig.from_environment()
        self.status_path = self.config.state_dir / "runtime_status.json"
        self.stop_request_path = self.config.state_dir / "room_agent.stop"
        self.lock_path = self.config.instance_lock_path

    def status(self, *, probe: bool = True) -> dict[str, Any]:
        on_disk = read_status(self.status_path)
        lock = InstanceLock(self.lock_path, role="probe")
        existing = lock.read_existing()
        holder = int(existing.get("pid", 0)) if existing else 0
        running, holder = lock.holder_is_live(existing)
        payload: dict[str, Any] = {
            "running": running,
            "pid": (holder or None) if running else None,
            "lock": existing,
            "status_file": on_disk,
            "stop_requested": self.stop_request_path.exists(),
        }
        if probe and running:
            payload["http_status"] = _http_get(
                f"http://{self.config.host}:{self.config.brain_port}/status"
            )
        return payload

    def health(self) -> dict[str, Any]:
        return _http_get(f"http://{self.config.host}:{self.config.brain_port}/health")

    def stop(self, *, timeout: float = DEFAULT_STOP_TIMEOUT_SECONDS) -> dict[str, Any]:
        """Cooperative stop with a bounded wait, then a hard fallback."""
        lock = InstanceLock(self.lock_path, role="probe")
        existing = lock.read_existing()
        live, holder = lock.holder_is_live(existing)
        if not live:
            self.clear_stop_request()
            return {"stopped": True, "already_stopped": True, "pid": None}
        self.stop_request_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(
            self.stop_request_path,
            json.dumps({"requested_at": _iso(_utc_now()), "requested_by_pid": os.getpid()}),
        )
        deadline = time.monotonic() + max(1.0, float(timeout))
        while time.monotonic() < deadline:
            if not pid_is_alive(holder):
                break
            time.sleep(0.2)
        stopped = not pid_is_alive(holder)
        forced = False
        if not stopped:
            forced = True
            try:
                os.kill(holder, signal.SIGTERM)
            except OSError:
                pass
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline and pid_is_alive(holder):
                time.sleep(0.2)
            stopped = not pid_is_alive(holder)
        if stopped:
            self.clear_stop_request()
        return {
            "stopped": stopped,
            "already_stopped": False,
            "forced": forced,
            "pid": holder,
        }

    def clear_stop_request(self) -> None:
        try:
            self.stop_request_path.unlink()
        except OSError:
            pass

    # ------------------------------------------------------------ spawning

    def _child_log_path(self) -> Path:
        return self.config.log_dir / "room_agent.out.log"

    def _guard_single_instance(self) -> None:
        """The duplicate-launch + stop-file gate shared by start/supervise."""
        current = self.status(probe=False)
        if current["running"]:
            raise DuplicateInstanceError(f"instance_already_running:{current['pid']}")
        self.clear_stop_request()

    def _spawn_child(
        self,
        *,
        command: Optional[list[str]] = None,
        cwd: Optional[Path] = None,
        env: Optional[dict[str, str]] = None,
    ) -> tuple[subprocess.Popen, Path, Optional[str]]:
        """Spawn one backend child. The argv/cwd/flags are unchanged."""
        log_path = self._child_log_path()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        rotated_sink = rotate_stdout_sink(log_path)
        creationflags = 0
        if os.name == "nt":  # pragma: no cover - Windows only
            # The scheduled entry point is a GUI process, so this child has no
            # inherited console. CREATE_NO_WINDOW keeps the recovered backend
            # from allocating one while CREATE_NEW_PROCESS_GROUP preserves the
            # existing Windows process-group semantics.
            creationflags = (
                getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                | getattr(subprocess, "CREATE_NO_WINDOW", 0)
            )
        argv = list(command) if command is not None else default_backend_command()
        workdir = str(cwd) if cwd is not None else str(PROJECT_ROOT)
        with open(log_path, "a", encoding="utf-8") as sink:
            process = subprocess.Popen(
                argv,
                cwd=workdir,
                stdout=sink,
                stderr=sink,
                creationflags=creationflags,
                close_fds=True,
                env=env,
            )
        return process, log_path, rotated_sink

    def _await_ready(self, process: subprocess.Popen, wait_seconds: float) -> bool:
        deadline = time.monotonic() + max(0.5, float(wait_seconds))
        while time.monotonic() < deadline:
            if process.poll() is not None:
                return False
            probe = _http_get(
                f"http://{self.config.host}:{self.config.brain_port}/status"
            )
            if probe.get("reachable"):
                return True
            time.sleep(0.25)
        return False

    def start(self, *, wait_seconds: float = 10.0) -> dict[str, Any]:
        """Spawn the backend as the single instance, then wait for readiness."""
        self._guard_single_instance()
        process, log_path, rotated_sink = self._spawn_child()
        started = self._await_ready(process, wait_seconds)
        return {
            "started": started,
            "pid": process.pid,
            "exit_code": process.poll(),
            "log": log_path.name,
            "log_rotated": rotated_sink,
        }

    # ---------------------------------------------------------- supervision

    def _capture_exit(self, process: subprocess.Popen) -> Optional[int]:
        """Block until the child exits and return the OS-reported exit code."""
        return process.wait()

    def _shutdown_evidence(
        self, started_at: datetime
    ) -> tuple[bool, dict[str, Any]]:
        """Whether the documented shutdown path left evidence for this run."""
        detail: dict[str, Any] = {
            "status_state": None,
            "termination_classification": None,
            "termination_after_start": False,
        }
        status = read_status(self.status_path)
        if status is not None:
            detail["status_state"] = status.get("state")
        if str(detail["status_state"]).upper() == STATE_STOPPED:
            return True, detail
        termination = _read_json_document(
            self.config.state_dir / "runtime_termination.json"
        )
        if termination is not None:
            detail["termination_classification"] = termination.get("classification")
            stamp = termination.get("timestamp")
            if isinstance(stamp, str):
                try:
                    moment = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
                    detail["termination_after_start"] = moment >= started_at
                except ValueError:
                    detail["termination_after_start"] = False
            if (
                termination.get("classification") == TERMINATION_CLEAN_SHUTDOWN
                and detail["termination_after_start"]
            ):
                return True, detail
        return False, detail

    def supervise(
        self,
        *,
        wait_seconds: float = 10.0,
        run_id: Optional[str] = None,
        events_path: Optional[Path] = None,
        command: Optional[list[str]] = None,
        cwd: Optional[Path] = None,
        env: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        """Run the single backend as a supervised child and record its exit.

        This is the documented ``start`` path plus a parent that stays alive
        for the child's whole lifetime, so the operating system hands the real
        child exit code to a process still able to record it. There is no
        automatic restart: the supervisor only observes and classifies, and a
        failed readiness probe never terminates the child.
        """
        self._guard_single_instance()
        run_id = run_id or _new_supervisor_run_id()
        sink_path = events_path or (self.config.log_dir / SUPERVISOR_LOG_NAME)
        try:
            events = SupervisorEventLog(sink_path, run_id)
        except OSError as exc:
            # A supervisor that cannot record must not start a child it could
            # never classify, and must never report a clean result.
            return {
                "run_id": run_id,
                "classification": CLASSIFICATION_SUPERVISOR_ERROR,
                "error": safe_error(exc, limit=200),
                "backend_pid": None,
                "exit_code": None,
                "started_at": None,
                "ended_at": _iso(_utc_now()),
                "events_path": str(sink_path),
            }
        argv = list(command) if command is not None else default_backend_command()
        workdir = str(cwd) if cwd is not None else str(PROJECT_ROOT)
        started_at = _utc_now()
        started_monotonic = time.monotonic()
        events.emit(
            "supervisor.start",
            parent_pid=os.getpid(),
            command=argv,
            cwd=workdir,
            log=str(self._child_log_path()),
            started_at=_iso(started_at),
            automatic_restart=False,
        )
        try:
            process, log_path, rotated_sink = self._spawn_child(
                command=command, cwd=cwd, env=env
            )
        except Exception as exc:  # noqa: BLE001 - a spawn failure is a result
            events.emit("supervisor.error", stage="spawn", error=safe_error(exc, limit=200))
            events.close()
            return {
                "run_id": run_id,
                "classification": CLASSIFICATION_SUPERVISOR_ERROR,
                "error": safe_error(exc, limit=200),
                "backend_pid": None,
                "exit_code": None,
                "started_at": _iso(started_at),
                "ended_at": _iso(_utc_now()),
                "events_path": str(events.path),
            }

        backend_pid = process.pid
        events.emit(
            "supervisor.spawned", backend_pid=backend_pid, log_rotated=rotated_sink
        )
        ready = self._await_ready(process, wait_seconds)
        events.emit("supervisor.ready", backend_pid=backend_pid, ready=ready)

        try:
            exit_code = self._capture_exit(process)
        except Exception as exc:  # noqa: BLE001 - capture failure is a result
            events.emit(
                "supervisor.error",
                stage="capture_exit",
                backend_pid=backend_pid,
                error=safe_error(exc, limit=200),
            )
            events.close()
            return {
                "run_id": run_id,
                "classification": CLASSIFICATION_SUPERVISOR_ERROR,
                "error": safe_error(exc, limit=200),
                "backend_pid": backend_pid,
                "exit_code": None,
                "started_at": _iso(started_at),
                "ended_at": _iso(_utc_now()),
                "events_path": str(events.path),
            }

        ended_at = _utc_now()
        shutdown_evidence, evidence_detail = self._shutdown_evidence(started_at)
        classification = classify_child_exit(exit_code, shutdown_evidence)
        events.emit(
            "supervisor.exit",
            backend_pid=backend_pid,
            exit_code=exit_code,
            classification=classification,
            shutdown_evidence=shutdown_evidence,
            shutdown_evidence_detail=evidence_detail,
            ended_at=_iso(ended_at),
            elapsed_seconds=round(time.monotonic() - started_monotonic, 3),
        )
        events.close()
        return {
            "run_id": run_id,
            "classification": classification,
            "backend_pid": backend_pid,
            "exit_code": exit_code,
            "shutdown_evidence": shutdown_evidence,
            "started_at": _iso(started_at),
            "ended_at": _iso(ended_at),
            "elapsed_seconds": round(time.monotonic() - started_monotonic, 3),
            "log": log_path.name,
            "events_path": str(events.path),
        }

    def restart(self, *, timeout: float = DEFAULT_STOP_TIMEOUT_SECONDS) -> dict[str, Any]:
        stopped = self.stop(timeout=timeout)
        started = self.start()
        return {"stop": stopped, "start": started}


def _http_get(url: str, timeout: float = 2.0) -> dict[str, Any]:
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if isinstance(payload, dict):
            payload["reachable"] = True
            return payload
        return {"reachable": True, "payload": payload}
    except urllib.error.HTTPError as exc:
        return {"reachable": True, "http_status": exc.code}
    except Exception as exc:
        return {"reachable": False, "error": safe_error(exc, limit=120)}


# ------------------------------------------------------------------------ cli


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m backend.runtime.lifecycle",
        description="Start/stop/status/health for the single Room Agent instance.",
    )
    parser.add_argument(
        "command",
        choices=("start", "stop", "restart", "status", "health", "supervise"),
    )
    parser.add_argument("--timeout", type=float, default=DEFAULT_STOP_TIMEOUT_SECONDS)
    parser.add_argument("--wait", type=float, default=10.0, help="startup wait seconds")
    parser.add_argument("--run-id", default=None, help="supervisor run identifier")
    parser.add_argument(
        "--events", default=None, help="supervisor JSONL path (default: log dir)"
    )
    args = parser.parse_args(argv)

    controller = RoomAgentController()
    exit_status = 0
    try:
        if args.command == "status":
            payload = controller.status()
        elif args.command == "health":
            payload = controller.health()
        elif args.command == "stop":
            payload = controller.stop(timeout=args.timeout)
        elif args.command == "start":
            payload = controller.start(wait_seconds=args.wait)
        elif args.command == "supervise":
            payload = controller.supervise(
                wait_seconds=args.wait,
                run_id=args.run_id,
                events_path=Path(args.events) if args.events else None,
            )
            if payload.get("classification") == CLASSIFICATION_SUPERVISOR_ERROR:
                exit_status = 1
        else:
            payload = controller.restart(timeout=args.timeout)
    except DuplicateInstanceError as exc:
        print(json.dumps({"error": str(exc)}, indent=2))
        return 3
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=str))
    return exit_status


if __name__ == "__main__":  # pragma: no cover - manual entry point
    raise SystemExit(main())


__all__ = [
    "CLASSIFICATION_ABRUPT_OR_UNKNOWN",
    "CLASSIFICATION_CLEAN_EXIT",
    "CLASSIFICATION_NONZERO_EXIT",
    "CLASSIFICATION_SUPERVISOR_ERROR",
    "DEFAULT_STOP_TIMEOUT_SECONDS",
    "DuplicateInstanceError",
    "InstanceLock",
    "ProcessIdentity",
    "RoomAgentController",
    "RuntimeLifecycle",
    "SUPERVISOR_VERSION",
    "ShutdownReport",
    "STATE_RUNNING",
    "STATE_STARTING",
    "STATE_STOPPED",
    "STATE_STOPPING",
    "SupervisorEventLog",
    "TERMINATION_CLEAN_SHUTDOWN",
    "TERMINATION_STARTUP_RECOVERY",
    "TERMINATION_UNEXPECTED_EXIT",
    "TERMINATION_UNKNOWN",
    "TerminationLedger",
    "TerminationRecord",
    "classify_child_exit",
    "default_backend_command",
    "pid_is_alive",
    "read_process_identity",
    "read_status",
    "write_status",
]
