"""Ingest loop that feeds a :class:`ChatProvider` into the existing ChatService.

This is the only place real platform messages enter the pipeline, and it is
deliberately thin: identity namespacing, duplicate-id guarding, bounded
per-poll intake, and counters. Normalization, moderation, dedup-by-similarity,
rate limiting, priority, intent, and queueing all stay in ChatService.

The Brain never touches this module; it keeps reading ``ChatService``'s queue.
A provider that fails degrades chat input, never the character.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Deque, Optional

from ...runtime import instrument
from ...runtime.redaction import redact_text
from ..models import utc_now
from ..service import ChatService, SubmissionResult
from .base import ChatProvider, ChatProviderError, ChatSourceEvent, ProviderState


MAX_ERROR_DETAIL = 200
RATE_WINDOW_SECONDS = 60.0
ACTIVE_VIEWER_WINDOW_SECONDS = 300.0
MAX_ACTIVE_VIEWERS = 200


@dataclass(frozen=True)
class IngestReport:
    """Result of one :meth:`ChatSourceRunner.run_once` pass."""

    events: int = 0
    accepted: int = 0
    filtered: int = 0
    rejected: int = 0
    merged: int = 0
    ignored: int = 0
    invalid: int = 0
    errors: int = 0
    detail: Optional[str] = None
    submissions: tuple[SubmissionResult, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "events": self.events,
            "accepted": self.accepted,
            "filtered": self.filtered,
            "rejected": self.rejected,
            "merged": self.merged,
            "ignored": self.ignored,
            "invalid": self.invalid,
            "errors": self.errors,
            "detail": self.detail,
        }


class ChatSourceRunner:
    def __init__(
        self,
        provider: ChatProvider,
        chat_service: ChatService,
        clock: Callable[[], datetime] = utc_now,
        poll_interval_seconds: float = 1.0,
        thread_name: str = "chat-source",
    ) -> None:
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        self.provider = provider
        self.chat_service = chat_service
        self.clock = clock
        self.poll_interval_seconds = poll_interval_seconds
        self._thread_name = thread_name
        self._ingest_capacity = max(1, int(getattr(provider, "ingest_capacity", 25)))
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._seen_ids: Deque[str] = deque(maxlen=2048)
        self._seen_id_set: set[str] = set()
        self._recent_stamps: Deque[datetime] = deque(maxlen=512)
        self._viewer_stamps: Deque[tuple[datetime, str]] = deque(maxlen=MAX_ACTIVE_VIEWERS * 4)
        self._counters: dict[str, int] = {
            "polls": 0,
            "events": 0,
            "messages": 0,
            "accepted": 0,
            "filtered": 0,
            "rejected": 0,
            "merged": 0,
            "ignored": 0,
            "invalid": 0,
            "duplicate_ids": 0,
            "dropped_overflow": 0,
            "provider_errors": 0,
            "submit_errors": 0,
        }
        self._last_error: Optional[str] = None
        self._last_message: Optional[dict[str, Any]] = None
        self._last_poll_at: Optional[datetime] = None
        self._started_at: Optional[datetime] = None
        self._degraded = False

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Start background ingest. A provider outage must not block startup."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._started_at = self.clock()
            self._thread = threading.Thread(
                target=self._serve, name=self._thread_name, daemon=True
            )
            self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        with self._lock:
            thread = self._thread
            self._thread = None
        self._stop_event.set()
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        try:
            self.provider.disconnect()
        except Exception:
            self._record_error("disconnect_failed")

    def _serve(self) -> None:
        try:
            self.provider.connect()
        except ChatProviderError as exc:
            self._record_error(exc.reason)
        except Exception as exc:  # pragma: no cover - defensive
            self._record_error(type(exc).__name__)
        while not self._stop_event.is_set():
            try:
                self.run_once()
            except Exception as exc:  # pragma: no cover - never escape a thread
                self._record_error(type(exc).__name__)
            # Interruptible wait: stopping never has to outlast a full interval.
            self._stop_event.wait(self.poll_interval_seconds)
        try:
            self.provider.disconnect()
        except Exception:
            self._record_error("disconnect_failed")

    # -- one pass ----------------------------------------------------------

    def run_once(self) -> IngestReport:
        """Drain the provider once and push each message through ChatService."""
        submissions: list[SubmissionResult] = []
        try:
            events = list(self.provider.poll())
        except ChatProviderError as exc:
            self._record_error(exc.reason or "provider_error")
            return IngestReport(errors=1, detail=self._last_error)
        except Exception as exc:  # pragma: no cover - defensive
            self._record_error(type(exc).__name__)
            return IngestReport(errors=1, detail=self._last_error)
        finally:
            with self._lock:
                self._counters["polls"] += 1
                self._last_poll_at = self.clock()

        if not events:
            return IngestReport()

        # Bounded intake: a chat flood can never grow an unbounded in-memory list.
        intake = events[: self._ingest_capacity]
        overflow = len(events) - len(intake)
        accepted = 0
        filtered = 0
        rejected = 0
        merged = 0
        ignored = 0
        invalid = 0

        with self._lock:
            self._counters["events"] += len(intake)
            if overflow:
                self._counters["dropped_overflow"] += overflow
        instrument.count("chat_source.events", len(intake))
        if overflow:
            instrument.count("chat_source.overflow_dropped", overflow)

        for event in intake:
            if not event.is_message:
                ignored += 1
                if event.kind.value in {"VIEWER_JOINED", "VIEWER_LEFT"}:
                    self._note_viewer(event)
                continue
            if not event.viewer_id or not event.text.strip():
                invalid += 1
                continue
            if self._already_seen(event.message_id):
                ignored += 1
                with self._lock:
                    self._counters["duplicate_ids"] += 1
                instrument.count("chat_source.duplicate_ids")
                continue
            with self._lock:
                self._counters["messages"] += 1
            try:
                result = self.chat_service.submit(
                    event.viewer_id,
                    event.text,
                    message_id=event.message_id,
                    timestamp=event.timestamp,
                    platform=event.platform,
                    viewer_name=event.viewer_name,
                    channel=event.channel,
                )
            except ValueError:
                invalid += 1
                with self._lock:
                    self._counters["submit_errors"] += 1
                continue
            except Exception as exc:  # pragma: no cover - defensive
                with self._lock:
                    self._counters["submit_errors"] += 1
                self._record_error(type(exc).__name__)
                continue

            submissions.append(result)
            if result.accepted:
                accepted += 1
            elif result.merged_message is not None:
                merged += 1
                filtered += 1
            elif result.message.status.value == "REJECTED":
                rejected += 1
            else:
                filtered += 1
            self._note_message(event, result)

        with self._lock:
            self._counters["accepted"] += accepted
            self._counters["filtered"] += filtered
            self._counters["rejected"] += rejected
            self._counters["merged"] += merged
            self._counters["ignored"] += ignored
            self._counters["invalid"] += invalid
        instrument.count("chat_source.accepted", accepted)
        instrument.count("chat_source.filtered", filtered)
        instrument.count("chat_source.rejected", rejected)
        instrument.gauge("queue.chat.depth", float(len(self.chat_service.queue)))

        return IngestReport(
            events=len(intake),
            accepted=accepted,
            filtered=filtered,
            rejected=rejected,
            merged=merged,
            ignored=ignored,
            invalid=invalid,
            detail=f"overflow={overflow}" if overflow else None,
            submissions=tuple(submissions),
        )

    # -- bookkeeping -------------------------------------------------------

    def _already_seen(self, message_id: Optional[str]) -> bool:
        if not message_id:
            return False
        with self._lock:
            if message_id in self._seen_id_set:
                return True
            if len(self._seen_ids) == self._seen_ids.maxlen:
                self._seen_id_set.discard(self._seen_ids[0])
            self._seen_ids.append(message_id)
            self._seen_id_set.add(message_id)
            return False

    def _note_message(self, event: ChatSourceEvent, result: SubmissionResult) -> None:
        now = event.timestamp or self.clock()
        with self._lock:
            self._recent_stamps.append(now)
            self._viewer_stamps.append((now, event.viewer_id))
            self._last_message = {
                "message_id": event.message_id,
                "viewer_id": event.viewer_id,
                "viewer_name": event.viewer_name,
                "text": event.text[:160],
                "timestamp": now.isoformat(timespec="seconds").replace("+00:00", "Z"),
                "channel": event.channel,
                "status": result.message.status.value,
                "intent": result.message.intent.value,
                "accepted": result.accepted,
            }

    def _note_viewer(self, event: ChatSourceEvent) -> None:
        now = event.timestamp or self.clock()
        with self._lock:
            self._viewer_stamps.append((now, event.viewer_id))

    def _record_error(self, reason: str) -> None:
        instrument.count("chat_source.errors")
        instrument.record_error("chat_source", reason)
        with self._lock:
            self._counters["provider_errors"] += 1
            # Provider errors can echo the endpoint or a credential; this
            # string is served by ``/chat/source/health`` and by diagnostics.
            self._last_error = redact_text(reason or "unknown", limit=MAX_ERROR_DETAIL)
            self._degraded = True

    def _rate_per_minute(self) -> float:
        cutoff = self.clock() - timedelta(seconds=RATE_WINDOW_SECONDS)
        with self._lock:
            stamps = [stamp for stamp in self._recent_stamps if stamp >= cutoff]
        return float(len(stamps))

    def _viewer_snapshot(self) -> tuple[list[str], int]:
        """Viewers seen inside the activity window, plus every viewer ever seen."""
        cutoff = self.clock() - timedelta(seconds=ACTIVE_VIEWER_WINDOW_SECONDS)
        with self._lock:
            pairs = [(stamp, viewer) for stamp, viewer in self._viewer_stamps if stamp >= cutoff]
            known = {viewer for _, viewer in self._viewer_stamps}
        seen: list[str] = []
        for _, viewer in sorted(pairs, key=lambda item: item[0]):
            if viewer not in seen:
                seen.append(viewer)
        return seen[-MAX_ACTIVE_VIEWERS:], len(known)

    @property
    def state(self) -> ProviderState:
        return self.provider.state

    def get_state(self) -> dict[str, Any]:
        provider_state = self.provider.state
        with self._lock:
            counters = dict(self._counters)
            last_error = self._last_error
            last_message = dict(self._last_message) if self._last_message else None
            last_poll_at = self._last_poll_at
            started_at = self._started_at
            running = bool(self._thread and self._thread.is_alive())
        try:
            chat_state = self.chat_service.get_state()
        except Exception:  # pragma: no cover - defensive
            chat_state = {}
        active_viewers, viewers_seen = self._viewer_snapshot()
        try:
            provider_diagnostics = dict(self.provider.diagnostics())
        except Exception:  # pragma: no cover - defensive
            provider_diagnostics = {}
        return {
            "provider": getattr(self.provider, "name", "unknown"),
            "platform": getattr(self.provider, "platform", "unknown"),
            "state": provider_state.value,
            "connected": provider_state is ProviderState.CONNECTED,
            "running": running,
            "degraded": bool(last_error) and provider_state is not ProviderState.CONNECTED,
            "channels": provider_diagnostics.get("channels", []),
            "poll_interval_seconds": self.poll_interval_seconds,
            "ingest_capacity": self._ingest_capacity,
            "counters": counters,
            "message_rate_per_minute": self._rate_per_minute(),
            "active_viewers": active_viewers,
            "viewers_seen": viewers_seen,
            "reconnect_count": provider_diagnostics.get("reconnect_count", 0),
            "connect_attempts": provider_diagnostics.get("connect_attempts", 0),
            "last_error": last_error,
            "last_message": last_message,
            "last_poll_at": (
                last_poll_at.isoformat(timespec="seconds").replace("+00:00", "Z")
                if last_poll_at
                else None
            ),
            "started_at": (
                started_at.isoformat(timespec="seconds").replace("+00:00", "Z")
                if started_at
                else None
            ),
            "queue": {
                "queued_count": chat_state.get("queued_count", 0),
                "queue_capacity": chat_state.get("queue_capacity", 0),
            },
            "provider_diagnostics": provider_diagnostics,
        }

    def health(self) -> dict[str, Any]:
        state = self.get_state()
        provider_state = state["state"]
        healthy = provider_state in {
            ProviderState.CONNECTED.value,
            ProviderState.CONNECTING.value,
            ProviderState.RETRYING.value,
        }
        return {
            "ok": healthy,
            "provider": state["provider"],
            "platform": state["platform"],
            "state": provider_state,
            "channels": state["channels"],
            "degraded": state["degraded"],
            "reconnect_count": state["reconnect_count"],
            "message_rate_per_minute": state["message_rate_per_minute"],
            "queued_count": state["queue"]["queued_count"],
            "last_error": state["last_error"],
        }
