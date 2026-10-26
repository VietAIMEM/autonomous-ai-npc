"""Durable record of the last deliberate ``STOP_STREAM``.

PHASE 13 keeps a small in-memory latch inside
:class:`~backend.social.context.SocialContext` so an unrelated Brain action can
never restart a stream a viewer or operator explicitly stopped. That latch is
enough inside one process; after a restart the Brain would forget the stop and
:class:`~backend.brain.goals.GoalEvaluator` could propose ``MAINTAIN_STREAM``
again.

This module persists exactly that one bit in the existing
``RuntimeConfig.state_dir`` using an atomic replace. The file never holds
provider data: one boolean and one timestamp. A missing file means "no
deliberate stop on record"; an unreadable or corrupt file fails closed because
restarting a deliberately stopped stream is the failure this latch exists to
prevent. PHASE 24 tightened the clearing rule: only the control plane
(``/stream/start`` from the Admin Dashboard or an operator start) clears it,
while the Brain refuses to dispatch ``START_STREAM`` at all while a deliberate
stop is on record.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from . import instrument
from .obs import log_event

LATCH_FILE_NAME = "stream_stop_latch.json"
LATCH_VERSION = 1

log = logging.getLogger("room_agent.runtime.stream_latch")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class StreamStopLatch:
    """One durable boolean: was the stream deliberately stopped?

    ``path=None`` keeps the latch in memory only, which is what automated tests
    and transport-less runs use. Any storage fault is reported through
    :meth:`snapshot` and never raises into a Brain tick.
    """

    def __init__(
        self,
        path: Optional[Path] = None,
        *,
        clock: Callable[[], datetime] = _utc_now,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.path = Path(path) if path is not None else None
        self._clock = clock
        self._log = logger or log
        self._stopped = False
        self._loaded = False
        self._source = ""
        self._updated_at: Optional[datetime] = None
        self._load_error = ""
        self._write_failures = 0

    @property
    def stopped(self) -> bool:
        return self._stopped

    @property
    def persistent(self) -> bool:
        return self.path is not None

    def load(self) -> bool:
        """Read the persisted value and make it current.

        A missing file leaves the latch clear. A corrupt or unreadable file
        fails closed (``stopped``) so a damaged latch cannot lead to an
        autonomous restart; the next explicit start clears it.
        """
        self._load_error = ""
        if self.path is None:
            self._loaded = True
            return self._stopped
        try:
            raw = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            self._stopped = False
            self._loaded = True
            return self._stopped
        except OSError as exc:
            self._fail_closed("read_failed", exc)
            return self._stopped
        try:
            payload = json.loads(raw)
        except ValueError as exc:
            self._fail_closed("corrupt_payload", exc)
            return self._stopped
        if not isinstance(payload, dict):
            self._fail_closed("unexpected_payload", "object_required")
            return self._stopped
        self._stopped = bool(payload.get("stopped", False))
        self._source = str(payload.get("source") or "")
        self._updated_at = _parse_moment(payload.get("updated_at"))
        self._loaded = True
        return self._stopped

    def store(self, stopped: bool, *, source: str = "") -> bool:
        """Apply and, when a path is configured, persist the value.

        Returns ``True`` when the value was written to disk (or when no path is
        configured). A failed write keeps the in-memory value and reports the
        fault so the caller can keep running degraded instead of crashing.
        """
        stopped = bool(stopped)
        moment = self._clock()
        changed = stopped != self._stopped
        self._stopped = stopped
        self._source = str(source or "")[:120]
        self._updated_at = moment
        if self.path is None:
            return True
        payload: dict[str, Any] = {
            "version": LATCH_VERSION,
            "stopped": stopped,
            "source": self._source,
            "updated_at": moment.astimezone(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z"),
        }
        try:
            _atomic_write_json(self.path, payload)
        except OSError as exc:
            self._write_failures += 1
            instrument.record_error("stream", f"stop_latch_write_failed:{type(exc).__name__}")
            log_event(
                self._log,
                "stream.stop_latch_write_failed",
                path=str(self.path),
                stopped=stopped,
                error=f"{type(exc).__name__}",
            )
            return False
        if changed:
            instrument.count(
                "stream.stop_latch_set" if stopped else "stream.stop_latch_cleared"
            )
            log_event(
                self._log,
                "stream.stop_latch_recorded",
                stopped=stopped,
                source=self._source,
            )
        return True

    def record_stop(self, *, source: str = "") -> bool:
        return self.store(True, source=source)

    def clear(self, *, source: str = "") -> bool:
        return self.store(False, source=source)

    def snapshot(self) -> dict[str, Any]:
        """Credential-free view safe for ``/status`` and ``/diagnostics``."""
        return {
            "stopped": self._stopped,
            "persistent": self.persistent,
            "loaded": self._loaded,
            "source": self._source or None,
            "updated_at": self._updated_at.astimezone(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z")
            if self._updated_at
            else None,
            "load_error": self._load_error or None,
            "write_failures": self._write_failures,
        }

    def _fail_closed(self, reason: str, detail: object) -> None:
        self._stopped = True
        self._loaded = False
        self._load_error = reason
        instrument.record_error("stream", f"stop_latch_{reason}")
        log_event(
            self._log,
            "stream.stop_latch_load_failed",
            path=str(self.path),
            reason=reason,
            error=f"{type(detail).__name__}",
            failed_closed=True,
        )


def _parse_moment(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            if temporary.exists():
                temporary.unlink()
        except OSError:  # pragma: no cover - best-effort cleanup
            pass


__all__ = ["LATCH_FILE_NAME", "LATCH_VERSION", "StreamStopLatch"]
