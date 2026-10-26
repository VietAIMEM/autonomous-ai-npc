"""Read-only backend observation for the operator-controlled runtime.

RoomAgent deliberately has no automatic startup, automatic restart, or
automatic crash recovery. An operator starts the backend through the existing
Phase 23 launcher, and a crashed backend remains stopped until that operator
starts it again.

This module retains only the useful operator diagnostic:

* it asks the existing ``RoomAgentController`` whether a live lock exists;
* it probes ``/health`` without changing process or lifecycle state;
* it publishes an observation status without spawning a backend;
* historical supervisor state remains readable, but is never advanced into a
  restart decision.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from .config import RuntimeConfig
from .lifecycle import (
    RoomAgentController,
    _atomic_write,
    _http_get,
    _utc_now,
    write_status,
)
from .redaction import safe_error


LOGGER = logging.getLogger("room_agent.supervisor")

DEFAULT_STATE_NAME = "supervisor_state.json"
DEFAULT_RUN_LOCK_NAME = "supervisor.lock"

ACTION_NONE = "none"
AUTO_RECOVERY_ENABLED = False
AUTO_RECOVERY_DISABLED_REASON = "auto_recovery_disabled_by_policy"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _parse_iso(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value:
        return None
    try:
        moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment


def _age_seconds(value: Any, *, now: Optional[datetime] = None) -> Optional[float]:
    moment = _parse_iso(value)
    if moment is None:
        return None
    reference = now or _utc_now()
    return max(0.0, (reference - moment).total_seconds())


class SupervisorState:
    """Read-only access to the historical supervisor decision ledger."""

    def __init__(self, path: Path, *, clock: Callable[[], datetime] = _utc_now) -> None:
        self.path = path
        self._clock = clock
        self.payload: dict[str, Any] = {}
        self.load()

    def load(self) -> dict[str, Any]:
        self.payload = {}
        try:
            parsed = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            parsed = None
        if isinstance(parsed, dict):
            self.payload = parsed
        return self.payload

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write(
                self.path,
                json.dumps(self.payload, ensure_ascii=False, sort_keys=True, default=str),
            )
        except OSError:
            pass

@contextmanager
def run_lock(path: Path) -> Iterator[bool]:
    """Yield False when another supervisor run already owns ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        yield False
        return
    try:
        os.write(handle, str(os.getpid()).encode("ascii"))
        yield True
    finally:
        try:
            os.close(handle)
        except OSError:
            pass
        try:
            path.unlink()
        except OSError:
            pass


def _default_health_probe(config: RuntimeConfig) -> dict[str, Any]:
    return _http_get(
        f"http://{config.host}:{config.brain_port}/health", timeout=3.0
    )


def observe(
    config: Optional[RuntimeConfig] = None,
    *,
    controller: Optional[RoomAgentController] = None,
    state: Optional[SupervisorState] = None,
    probe: Optional[Callable[[RuntimeConfig], dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Read-only view of backend liveness, health, and historical state."""
    cfg = config or RuntimeConfig.from_environment()
    ctl = controller or RoomAgentController(cfg)
    store = state or SupervisorState(cfg.state_dir / DEFAULT_STATE_NAME)
    health_probe = probe or _default_health_probe

    record: dict[str, Any] = {
        "checked_at": _utc_now_iso(),
        "environment": cfg.environment,
        "running": False,
        "health_reachable": False,
        "health_status": None,
        "action": ACTION_NONE,
        "reason": None,
        "stopped_by_operator": False,
        "cooldown_seconds": 0.0,
        "auto_recovery_enabled": AUTO_RECOVERY_ENABLED,
        "auto_recovery_reason": AUTO_RECOVERY_DISABLED_REASON,
        "result": None,
    }

    status = ctl.status(probe=False)
    record["running"] = bool(status.get("running"))
    record["pid"] = status.get("pid")

    health: dict[str, Any] = {}
    if status.get("running"):
        try:
            health = health_probe(cfg)
        except Exception as exc:  # pragma: no cover - defensive
            health = {"reachable": False, "error": safe_error(exc, limit=120)}
        record["health_reachable"] = bool(health.get("reachable"))
        record["health_status"] = health.get("status")

    if status.get("running") and record["health_reachable"]:
        record["reason"] = "healthy"
    elif status.get("running"):
        record["reason"] = "unhealthy_backend_left_running_by_operator_policy"
    else:
        record["reason"] = "backend_stopped_until_operator_starts_it"
    return record


def publish_observation(payload: dict[str, Any], path: Optional[Path] = None) -> None:
    """Publish the supervisor view where operators (and the admin UI) can read it."""
    target = path
    if target is None:
        config = RuntimeConfig.from_environment()
        target = config.state_dir / "supervisor_status.json"
    payload = dict(payload)
    payload["updated_at"] = _utc_now_iso()
    write_status(target, payload)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m backend.runtime.supervisor",
        description=(
            "Observe the RoomAgent backend without starting, stopping, or "
            "restarting it. Recovery is operator-controlled by policy."
        ),
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="observe",
        choices=("observe",),
        help="read-only observation; automatic recovery is disabled by policy",
    )
    parser.add_argument(
        "--launcher",
        type=Path,
        help="accepted for compatibility; observe mode never launches it",
    )
    parser.add_argument("--json", action="store_true", help="print the record only")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    config = RuntimeConfig.from_environment()
    lock_path = config.state_dir / DEFAULT_RUN_LOCK_NAME
    record: dict[str, Any]
    with run_lock(lock_path) as acquired:
        if not acquired:
            record = {
                "checked_at": _utc_now_iso(),
                "action": ACTION_NONE,
                "reason": "supervisor_run_already_in_progress",
            }
        else:
            record = observe(config)
            publish_observation(record)

    if args.json:
        print(json.dumps(record, ensure_ascii=False, sort_keys=True, default=str))
    else:
        print(
            json.dumps(
                record, ensure_ascii=False, sort_keys=True, indent=2, default=str
            )
        )
    # Observation never signals the backend and never changes its lifecycle.
    return 0


if __name__ == "__main__":  # pragma: no cover - manual entry point
    raise SystemExit(main())


__all__ = [
    "ACTION_NONE",
    "AUTO_RECOVERY_DISABLED_REASON",
    "AUTO_RECOVERY_ENABLED",
    "DEFAULT_STATE_NAME",
    "SupervisorState",
    "main",
    "observe",
    "publish_observation",
    "run_lock",
]
