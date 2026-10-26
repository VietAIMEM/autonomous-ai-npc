"""Bounded, deduplicating speech queue that sits between the Brain and TTS.

The Brain only ever calls :meth:`SpeechService.request`; it never blocks on a
provider, never sees credentials, and never owns audio bytes. A worker thread
drains the queue, so a slow or failing provider can only ever delay speech,
never stall the autonomous loop.
"""

from __future__ import annotations

import hashlib
import queue
import re
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Optional

from ..chat.models import utc_now
from ..runtime import instrument
from ..tts.base import TTSProvider, TTSProviderError, TTSRequest, TTSAudio
from ..tts.config import TTSConfig, build_provider
from .models import IN_FLIGHT_STATES, TERMINAL_STATES, SpeechMetrics, SpeechRequest, SpeechState


TEXT_PATTERN = re.compile(r"\s+")
MAX_RETAINED_REQUESTS = 64
RECENT_WINDOW = 8
#: Failure reasons that mean "come back later", not "this utterance is broken".
RATE_LIMIT_REASONS = ("http_429", "rate_limited", "too_many_requests")
MAX_ERROR_DETAIL = 300
# Reported by the service-level diagnostics when no utterance is in flight.
IDLE_STATE = "IDLE"


@dataclass(frozen=True)
class SpeechConfig:
    queue_capacity: int = 8
    max_text_length: int = 500
    result_ttl_seconds: float = 120.0
    max_cached_audio: int = 8
    worker_poll_interval_seconds: float = 0.05
    stale_after_seconds: float = 45.0
    request_timeout_seconds: Optional[float] = None
    # A rate-limited provider is not retried per Brain tick: the service goes
    # quiet for the provider's own cool-down (or this default) instead.
    rate_limit_backoff_seconds: float = 300.0
    max_rate_limit_backoff_seconds: float = 3600.0


class SpeechService:
    def __init__(
        self,
        provider: Optional[TTSProvider] = None,
        config: SpeechConfig | None = None,
        tts_config: TTSConfig | None = None,
        clock: Callable[[], datetime] = utc_now,
        start_worker: bool = True,
    ) -> None:
        self.tts_config = tts_config or getattr(provider, "config", None) or TTSConfig()
        self.provider = provider if provider is not None else build_provider(self.tts_config)
        self.config = config or SpeechConfig()
        self._clock = clock
        self._lock = threading.RLock()
        self._queue: queue.Queue = queue.Queue(maxsize=max(1, int(self.config.queue_capacity)))
        self._requests: "OrderedDict[str, SpeechRequest]" = OrderedDict()
        self._by_dedup_key: dict[str, str] = {}
        self._by_text_key: dict[str, str] = {}
        self._audio_order: list[str] = []
        self._playing_id: Optional[str] = None
        self._last_request_id: Optional[str] = None
        self._last_result_id: Optional[str] = None
        self._consecutive_failures = 0
        self._paused_until: Optional[datetime] = None
        self._last_error_detail = ""
        self._counters = {
            "requested": 0,
            "synthesized": 0,
            "reused": 0,
            "rejected": 0,
            "failed": 0,
            "cancelled": 0,
            "completed": 0,
        }
        self._stopping = threading.Event()
        self._worker: Optional[threading.Thread] = None
        if start_worker:
            self.start()

    # ------------------------------------------------------------------ state

    @property
    def enabled(self) -> bool:
        return bool(self.tts_config.enabled)

    @property
    def provider_name(self) -> str:
        return getattr(self.provider, "name", "unknown")

    def start(self) -> None:
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                return
            self._stopping.clear()
            self._worker = threading.Thread(
                target=self._worker_loop,
                name="speech-worker",
                daemon=True,
            )
            self._worker.start()
        self._warm_provider_async()

    def _warm_provider_async(self) -> None:
        """Let a provider that owns a local model load it off the speech path.

        Only providers that expose ``warmup`` do anything here. The call is
        best-effort: the first real utterance warms the provider inline anyway.
        """
        warmup = getattr(self.provider, "warmup", None)
        if not callable(warmup):
            return

        def _run() -> None:
            try:
                warmup()
            except Exception:  # warmup is an optimisation, never a failure
                pass

        threading.Thread(target=_run, name="speech-warmup", daemon=True).start()

    def shutdown(self, join_timeout: float = 1.0) -> None:
        with self._lock:
            worker = self._worker
            self._worker = None
        self._stopping.set()
        if worker is not None and worker.is_alive():
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass
            worker.join(timeout=join_timeout)
        close = getattr(self.provider, "close", None)
        if callable(close):
            # Providers that own a process or a socket release it here so a
            # runtime stop cannot leave an orphan behind.
            try:
                close()
            except Exception:  # a provider teardown must not break shutdown
                pass

    # ------------------------------------------------------------- public API

    def request(
        self,
        text: str,
        *,
        source: str = "brain",
        dedup_key: Optional[str] = None,
        plan_id: Optional[str] = None,
        step_index: Optional[int] = None,
        priority: float = 0.5,
        interruptible: bool = True,
        voice: Optional[str] = None,
        model: Optional[str] = None,
        output_format: Optional[str] = None,
    ) -> SpeechRequest:
        """Queue an utterance and return immediately; never blocks the caller."""
        with self._lock:
            self._evict()
            instrument.count("tts.requests")
            body = (text or "").strip()
            if not body:
                return self._reject("empty_text", text)
            if len(body) > self.config.max_text_length:
                return self._reject("text_too_long", body)
            if not self.enabled:
                return self._reject("speech_disabled", body)
            if self._paused_seconds_locked() > 0:
                # The provider already said "not now". Queueing more work only
                # turns one rate limit into a backlog of guaranteed failures.
                instrument.count("tts.rate_limited")
                return self._reject("speech_rate_limited", body)

            text_key = _text_key(body)
            normalized_dedup = dedup_key.strip() if isinstance(dedup_key, str) and dedup_key.strip() else None

            if normalized_dedup:
                existing = self._requests.get(self._by_dedup_key.get(normalized_dedup, ""))
                if existing is not None and existing.state in _IN_FLIGHT:
                    self._counters["reused"] += 1
                    instrument.count("tts.reused")
                    return existing

            cached_id = self._by_text_key.get(text_key)
            cached = self._requests.get(cached_id or "")
            if cached is not None and cached.audio is not None:
                return self._reuse(cached, body, text_key, normalized_dedup, source, plan_id, step_index)

            attempt = 1
            if normalized_dedup:
                prior = self._requests.get(self._by_dedup_key.get(normalized_dedup, ""))
                if prior is not None:
                    attempt = prior.attempt + 1

            request = SpeechRequest(
                request_id=_new_id(),
                text=body,
                text_key=text_key,
                dedup_key=normalized_dedup or "",
                source=source,
                plan_id=plan_id,
                step_index=step_index,
                voice=voice or self.tts_config.voice,
                model=model or self.tts_config.model,
                output_format=output_format or self.tts_config.output_format,
                priority=priority,
                interruptible=interruptible,
                created_at=self._clock(),
                state=SpeechState.SPEECH_REQUESTED,
                attempt=attempt,
                provider=self.provider_name,
                metrics=SpeechMetrics(requested_at=self._clock()),
            )
            self._requests[request.request_id] = request
            if normalized_dedup:
                self._by_dedup_key[normalized_dedup] = request.request_id
            self._last_request_id = request.request_id
            self._counters["requested"] += 1

            try:
                self._queue.put_nowait(request.request_id)
            except queue.Full:
                request.state = SpeechState.FAILED
                request.error = "speech_queue_full"
                self._last_result_id = request.request_id
                self._counters["rejected"] += 1
                self._counters["failed"] += 1
                instrument.count("tts.rejected")
                instrument.count("tts.failed")
                instrument.record_error("tts", "speech_queue_full")
            instrument.gauge("queue.speech.depth", float(self._queue.qsize()))
            return request

    def get(self, request_id: str) -> Optional[SpeechRequest]:
        with self._lock:
            return self._requests.get(request_id)

    def state(self, request_id: str) -> Optional[SpeechState]:
        with self._lock:
            request = self._requests.get(request_id)
            return request.state if request else None

    def audio(self, request_id: str) -> Optional[tuple[bytes, str, str]]:
        """Payload, MIME type and download name for a synthesised utterance."""
        with self._lock:
            request = self._requests.get(request_id)
            if request is None or request.audio is None:
                return None
            probe = request.audio.probe
            return (
                request.audio.payload,
                probe.mime_type,
                f"{request.request_id}.{probe.format}",
            )

    def mark_playing(self, request_id: str) -> bool:
        with self._lock:
            request = self._requests.get(request_id)
            if request is None:
                return False
            if request.state == SpeechState.PLAYING:
                return True
            if request.state != SpeechState.READY or request.audio is None:
                return False
            request.state = SpeechState.PLAYING
            request.metrics.playback_started_at = self._clock()
            self._playing_id = request_id
            return True

    def mark_completed(self, request_id: str) -> bool:
        with self._lock:
            request = self._requests.get(request_id)
            if request is None:
                return False
            if request.state == SpeechState.COMPLETED:
                return True
            if request.state not in {SpeechState.PLAYING, SpeechState.READY}:
                return False
            request.state = SpeechState.COMPLETED
            if request.metrics.playback_started_at is None:
                request.metrics.playback_started_at = self._clock()
            request.metrics.playback_completed_at = self._clock()
            if self._playing_id == request_id:
                self._playing_id = None
            self._last_result_id = request_id
            self._counters["completed"] += 1
            instrument.count("tts.completed")
            return True

    def cancel(self, request_id: str, reason: str = "cancelled") -> bool:
        with self._lock:
            request = self._requests.get(request_id)
            if request is None or request.state in _TERMINAL:
                return False
            request.state = SpeechState.CANCELLED
            request.error = f"cancelled:{reason}"
            if self._playing_id == request_id:
                self._playing_id = None
            self._last_result_id = request_id
            self._counters["cancelled"] += 1
            instrument.count("tts.cancelled")
            return True

    def interrupt_current(self, reason: str = "interrupted") -> Optional[str]:
        with self._lock:
            target = self._playing_id or self._active_request_id()
            if target is None:
                return None
            request = self._requests.get(target)
            if request is None or not request.interruptible:
                return None
            return target if self.cancel(target, reason) else None

    def run_pending_once(self, timeout: float = 0.2) -> bool:
        """Process a single queued item inline; used by deterministic tests."""
        return self._process_next(timeout)

    def pending_count(self) -> int:
        return self._queue.qsize()

    # ------------------------------------------------------------ diagnostics

    def get_state(self) -> dict[str, object]:
        with self._lock:
            active_id = self._active_request_id()
            active = self._requests.get(active_id or "")
            last_request = self._requests.get(self._last_request_id or "")
            last_result = self._requests.get(self._last_result_id or "")
            recent = [
                request.to_dict()
                for request in list(self._requests.values())[-RECENT_WINDOW:]
            ]
            if last_result is not None:
                latency = last_result.metrics.to_dict()
            elif active is not None:
                latency = active.metrics.to_dict()
            else:
                latency = {}
            return {
                "provider": self.provider_name,
                "model": self.tts_config.model,
                "voice": self.tts_config.voice,
                "format": self.tts_config.output_format,
                "enabled": self.enabled,
                "streaming_supported": bool(getattr(self.provider, "streaming_supported", False)),
                # Idle reads as idle: claiming SPEECH_REQUESTED with nothing in
                # flight looks like a stuck utterance to anyone watching.
                "state": active.state.value if active else IDLE_STATE,
                "queue_length": self._queue.qsize(),
                "queue_capacity": self._queue.maxsize,
                "active_request_id": active_id,
                "playing_request_id": self._playing_id,
                "counters": dict(self._counters),
                "consecutive_failures": self._consecutive_failures,
                "cached_audio_requests": len(self._audio_order),
                "last_request": last_request.to_dict() if last_request else None,
                "last_result": last_result.to_dict() if last_result else None,
                "latency": latency,
                "recent": recent,
                **self._rate_limit_view_locked(),
            }

    def health(self) -> dict[str, object]:
        with self._lock:
            state = self.get_state()
            status = "disabled"
            if self.enabled:
                status = "degraded" if self._consecutive_failures else "ok"
            described: dict[str, object] = {}
            describe = getattr(self.provider, "describe", None)
            if callable(describe):
                try:
                    described = dict(describe())
                except Exception:  # a provider must never break diagnostics
                    described = {"provider": self.provider_name}
            last_result = self._requests.get(self._last_result_id or "")
            return {
                "status": status,
                "provider": described,
                "queue_length": state["queue_length"],
                "queue_capacity": state["queue_capacity"],
                "consecutive_failures": self._consecutive_failures,
                "last_error": last_result.error if last_result else None,
                "last_error_detail": self._last_error_detail or None,
                "latency": state["latency"],
            }

    def summary(self) -> dict[str, object]:
        """Compact view for the Brain state endpoint: no text, no audio, no secrets."""
        with self._lock:
            active_id = self._active_request_id()
            active = self._requests.get(active_id or "")
            return {
                "provider": self.provider_name,
                "model": self.tts_config.model,
                "voice": self.tts_config.voice,
                "format": self.tts_config.output_format,
                "enabled": self.enabled,
                "state": active.state.value if active else IDLE_STATE,
                "queue_length": self._queue.qsize(),
                "queue_capacity": self._queue.maxsize,
                "consecutive_failures": self._consecutive_failures,
            }

    # ---------------------------------------------------------------- internal

    def _worker_loop(self) -> None:
        while not self._stopping.is_set():
            if not self._process_next(self.config.worker_poll_interval_seconds):
                return

    def _process_next(self, timeout: float) -> bool:
        try:
            item = self._queue.get(timeout=timeout)
        except queue.Empty:
            return True
        if item is None:
            return False
        try:
            self._synthesize(str(item))
        finally:
            self._queue.task_done()
        return True

    def _synthesize(self, request_id: str) -> None:
        started = time.monotonic()
        with self._lock:
            request = self._requests.get(request_id)
            if request is None or request.state != SpeechState.SPEECH_REQUESTED:
                return
            age = (self._clock() - request.created_at).total_seconds()
            if age > self.config.stale_after_seconds:
                self._fail(request, "speech_stale")
                return
            if self._paused_seconds_locked() > 0:
                self._fail(request, "speech_rate_limited")
                return
            request.state = SpeechState.SYNTHESIZING
            request.metrics.synthesis_started_at = self._clock()
            tts_request = TTSRequest(
                text=request.text,
                model=request.model,
                voice=request.voice,
                output_format=request.output_format,
                language=self.tts_config.language,
                speed=self.tts_config.speed,
                request_id=request.request_id,
                timeout_seconds=self.request_timeout_seconds,
                metadata={"source": request.source, "plan_id": request.plan_id},
            )

        audio: Optional[TTSAudio] = None
        failure: Optional[str] = None
        failure_detail = ""
        retry_after: Optional[float] = None
        try:
            audio = self.provider.synthesize(tts_request)
        except TTSProviderError as exc:
            failure = exc.reason or "provider_error"
            failure_detail = str(exc.detail or "")
            retry_after = getattr(exc, "retry_after_seconds", None)
        except Exception as exc:  # a provider bug must not kill the worker
            failure = f"provider_crash:{type(exc).__name__}"

        with self._lock:
            request = self._requests.get(request_id)
            if request is None:
                return
            if request.state != SpeechState.SYNTHESIZING:
                # Cancelled while in flight: never resurrect a dead utterance.
                return
            if audio is None:
                self._fail(
                    request,
                    failure or "provider_error",
                    detail=failure_detail,
                    retry_after=retry_after,
                )
                instrument.observe_ms(
                    "tts.synthesis_ms", (time.monotonic() - started) * 1000.0
                )
                return
            request.audio = audio
            request.provider = audio.provider
            request.generation_id = audio.generation_id
            request.state = SpeechState.READY
            request.metrics.audio_ready_at = self._clock()
            request.error = None
            self._consecutive_failures = 0
            self._paused_until = None
            self._counters["synthesized"] += 1
            self._last_result_id = request_id
            self._cache_audio(request)
            instrument.count("tts.synthesized")
            instrument.observe_ms("tts.synthesis_ms", (time.monotonic() - started) * 1000.0)
            instrument.gauge("tts.cache.size", float(len(self._audio_order)))

    @property
    def request_timeout_seconds(self) -> Optional[float]:
        return self.config.request_timeout_seconds or self.tts_config.timeout_seconds

    def _reuse(
        self,
        cached: SpeechRequest,
        body: str,
        text_key: str,
        dedup_key: Optional[str],
        source: str,
        plan_id: Optional[str],
        step_index: Optional[int],
    ) -> SpeechRequest:
        """Hand a fresh request the audio that already exists: one text, one generation."""
        assert cached.audio is not None
        request = SpeechRequest(
            request_id=_new_id(),
            text=body,
            text_key=text_key,
            dedup_key=dedup_key or "",
            source=source,
            plan_id=plan_id,
            step_index=step_index,
            voice=cached.voice,
            model=cached.model,
            output_format=cached.audio.format,
            created_at=self._clock(),
            state=SpeechState.READY,
            attempt=cached.attempt,
            provider=cached.audio.provider,
            generation_id=cached.generation_id,
            reused_from=cached.request_id,
            audio=cached.audio,
            metrics=SpeechMetrics(
                requested_at=self._clock(),
                audio_ready_at=self._clock(),
            ),
        )
        self._requests[request.request_id] = request
        if dedup_key:
            self._by_dedup_key[dedup_key] = request.request_id
        self._by_text_key[text_key] = request.request_id
        self._last_request_id = request.request_id
        self._last_result_id = request.request_id
        self._counters["requested"] += 1
        self._counters["reused"] += 1
        instrument.count("tts.reused")
        return request

    def _reject(self, reason: str, text: str) -> SpeechRequest:
        request = SpeechRequest(
            request_id=_new_id(),
            text=(text or "").strip()[:200],
            text_key=_text_key(text or ""),
            dedup_key="",
            source="invalid",
            created_at=self._clock(),
            state=SpeechState.FAILED,
            provider=self.provider_name,
            error=reason,
            metrics=SpeechMetrics(requested_at=self._clock()),
        )
        self._requests[request.request_id] = request
        self._last_request_id = request.request_id
        self._last_result_id = request.request_id
        self._counters["requested"] += 1
        self._counters["rejected"] += 1
        self._counters["failed"] += 1
        instrument.count("tts.rejected")
        instrument.count("tts.failed")
        return request

    def _fail(
        self,
        request: SpeechRequest,
        reason: str,
        detail: str = "",
        retry_after: Optional[float] = None,
    ) -> None:
        request.state = SpeechState.FAILED
        request.error = reason
        if self._playing_id == request.request_id:
            self._playing_id = None
        self._last_result_id = request.request_id
        self._consecutive_failures += 1
        self._counters["failed"] += 1
        instrument.count("tts.failed")
        instrument.record_error("tts", reason)
        if detail:
            self._last_error_detail = detail[:MAX_ERROR_DETAIL]
        if _is_rate_limited(reason):
            delay = retry_after if retry_after and retry_after > 1.0 else self.config.rate_limit_backoff_seconds
            delay = max(1.0, min(float(delay), float(self.config.max_rate_limit_backoff_seconds)))
            self._paused_until = self._clock() + timedelta(seconds=delay)
            instrument.count("tts.rate_limited")

    def _paused_seconds_locked(self) -> float:
        """Cool-down remaining; callers must already hold the lock."""
        if self._paused_until is None:
            return 0.0
        remaining = (self._paused_until - self._clock()).total_seconds()
        if remaining <= 0:
            self._paused_until = None
            return 0.0
        return remaining

    def _rate_limit_view_locked(self) -> dict[str, object]:
        remaining = self._paused_seconds_locked()
        return {
            "paused": remaining > 0,
            "paused_seconds_remaining": round(remaining, 1),
            "paused_until": (
                self._paused_until.isoformat() if self._paused_until is not None else None
            ),
            "last_error_detail": self._last_error_detail or None,
        }

    def _cache_audio(self, request: SpeechRequest) -> None:
        self._by_text_key[request.text_key] = request.request_id
        if request.request_id in self._audio_order:
            self._audio_order.remove(request.request_id)
        self._audio_order.append(request.request_id)
        # Trim here as well: eviction only runs on new requests, and a burst of
        # completions must not push the cache past its configured ceiling.
        while len(self._audio_order) > max(1, int(self.config.max_cached_audio)):
            self._drop(self._audio_order.pop(0))

    def _active_request_id(self) -> Optional[str]:
        playing = self._requests.get(self._playing_id or "")
        if playing is not None and playing.state == SpeechState.PLAYING:
            return self._playing_id
        for request_id, request in reversed(list(self._requests.items())):
            if request.state in _IN_FLIGHT:
                return request_id
        return None

    def _evict(self) -> None:
        now = self._clock()
        ttl = self.config.result_ttl_seconds
        stale = [
            request_id
            for request_id, request in self._requests.items()
            if request.state in _TERMINAL
            and (now - request.created_at).total_seconds() > ttl
        ]
        for request_id in stale:
            self._drop(request_id)
        while len(self._audio_order) > max(1, int(self.config.max_cached_audio)):
            self._drop(self._audio_order.pop(0))
        while len(self._requests) > MAX_RETAINED_REQUESTS:
            self._drop(next(iter(self._requests)))
        instrument.gauge("tts.retained_requests", float(len(self._requests)))
        instrument.gauge("tts.cache.size", float(len(self._audio_order)))

    def _drop(self, request_id: str) -> None:
        request = self._requests.pop(request_id, None)
        if request is None:
            return
        if request_id in self._audio_order:
            self._audio_order.remove(request_id)
        if self._by_text_key.get(request.text_key) == request_id:
            del self._by_text_key[request.text_key]
        if request.dedup_key and self._by_dedup_key.get(request.dedup_key) == request_id:
            del self._by_dedup_key[request.dedup_key]
        if self._last_request_id == request_id:
            self._last_request_id = None
        if self._last_result_id == request_id:
            self._last_result_id = None
        if self._playing_id == request_id:
            self._playing_id = None


_IN_FLIGHT = IN_FLIGHT_STATES
_TERMINAL = TERMINAL_STATES


def _text_key(text: str) -> str:
    normalized = TEXT_PATTERN.sub(" ", (text or "").strip()).casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]


def _is_rate_limited(reason: str) -> bool:
    return any((reason or "").startswith(prefix) for prefix in RATE_LIMIT_REASONS)


def _new_id() -> str:
    return f"speech_{uuid.uuid4().hex[:16]}"
