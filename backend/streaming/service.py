"""Queue-shaped access to a streaming provider, owned by one worker thread.

Unity owns *logical* stream state (PHASE 04) and the Brain keeps issuing
``START_STREAM`` / ``STOP_STREAM``. This service owns the *transport*: it
serialises every provider call onto a single worker thread, buffers a bounded
amount of captured media, and answers "is it really live?" rather than "did we
try?".

Why a worker thread: an encoder can block on a full pipe, and a caller in the
middle of a Brain tick must never inherit that stall. Why bounded queues: a
stream that falls behind should drop pictures, not grow memory without limit.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Deque, Optional

from ..runtime import instrument
from .base import (
    ACTIVE_STATES,
    AudioChunk,
    StreamHealth,
    StreamingProvider,
    StreamingProviderError,
    StreamState,
    VideoFrame,
)
from .config import PROVIDER_SRT, StreamingConfig, build_provider, looks_like_endpoint


DEFAULT_POLL_INTERVAL_SECONDS = 0.02
#: Media older than this means the capture side stopped, not the platform.
INPUT_FRESHNESS_SECONDS = 2.0
#: A live muxer needs the picture track to keep the configured cadence: when
#: capture runs a little under the declared frame rate, the video timestamps
#: fall behind the real-time audio track until ffmpeg stops reading its inputs
#: and the write watchdog kills an encoder that was healthy. The newest
#: accepted picture is therefore re-sent to fill every cadence slot the
#: capture missed, which keeps the tracks interleaved. Repeats never chase a
#: backlog older than this many frame intervals, so a long hitch cannot burst
#: a pile of duplicate frames into the muxer at once.
REPEAT_BURST_LIMIT_INTERVALS = 4.0
MAX_WAIT_SECONDS = 30.0
WAIT_TICK_SECONDS = 0.01
#: How long to refuse a fresh start after one failed, so a bad key or an offline
#: platform cannot turn Brain retries into an encoder respawn storm.
START_BACKOFF_SECONDS = 5.0
STARTABLE_STATES = frozenset(
    {StreamState.CONNECTING, StreamState.LIVE, StreamState.RECONNECTING}
)
STOPPED_STATES = frozenset({StreamState.STOPPED, StreamState.IDLE, StreamState.ERROR})


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class IngestResult:
    """Whether one picture or audio block was taken for the encoder."""

    accepted: bool
    reason: str = ""
    queued: int = 0
    dropped: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "queued": self.queued,
            "dropped": self.dropped,
        }


@dataclass(frozen=True)
class StreamCommandResult:
    """Outcome of ``START_STREAM`` / ``STOP_STREAM`` at the transport boundary.

    ``ok`` on a start means the platform has a live stream carrying media, never
    that a connection attempt merely began.
    """

    ok: bool
    state: StreamState
    reason: str = ""
    waited_ms: int = 0
    transport_enabled: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "state": self.state.value,
            "reason": self.reason,
            "waited_ms": self.waited_ms,
            "transport_enabled": self.transport_enabled,
        }


class StreamingService:
    def __init__(
        self,
        provider: StreamingProvider,
        config: StreamingConfig | None = None,
        clock: Callable[[], datetime] = utc_now,
        poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
        start_worker: bool = True,
    ) -> None:
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        self.provider = provider
        self.config = config or getattr(provider, "config", None) or StreamingConfig()
        self._clock = clock
        self.poll_interval_seconds = poll_interval_seconds
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._stopping = threading.Event()
        self._video: Deque[VideoFrame] = deque(maxlen=max(1, int(self.config.video_queue_capacity)))
        self._audio: Deque[AudioChunk] = deque(maxlen=max(1, int(self.config.audio_queue_capacity)))
        self._thread: Optional[threading.Thread] = None
        self._pending_start = False
        self._pending_stop = False
        self._generation = 0
        # Ticket handshake: a caller can tell when the worker consumed *its*
        # start request rather than an earlier one.
        self._start_ticket = 0
        self._start_applied = 0
        self._last_video_sequence = -1
        self._last_audio_sequence = -1
        self._last_video_at: Optional[datetime] = None
        self._last_audio_at: Optional[datetime] = None
        self._last_video_frame: Optional[VideoFrame] = None
        self._next_paced_frame_at: Optional[datetime] = None
        self._last_camera = ""
        self._started_at: Optional[datetime] = None
        self._counters = {
            "video_accepted": 0,
            "video_repeated": 0,
            "audio_accepted": 0,
            "video_rejected": 0,
            "audio_rejected": 0,
            "video_dropped": 0,
            "audio_dropped": 0,
            "duplicates": 0,
            "starts": 0,
            "stops": 0,
            "connect_errors": 0,
            "polls": 0,
        }
        self._last_reason = ""
        self._start_latency_ms = 0
        self._failed_start_at: Optional[datetime] = None
        if start_worker:
            self.start()

    # ------------------------------------------------------------------ state

    @property
    def transport_enabled(self) -> bool:
        return bool(getattr(self.provider, "transport_enabled", False))

    @property
    def state(self) -> StreamState:
        return self.provider.state

    @property
    def start_backoff_seconds(self) -> float:
        """Cool-down after a failed start, so a bad key cannot respawn an encoder loop."""
        return max(0.0, float(getattr(self.config, "start_backoff_seconds", START_BACKOFF_SECONDS)))

    def start(self) -> None:
        """Start the transport worker. A broken provider must not block startup."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stopping.clear()
            self._started_at = self._clock()
            self._thread = threading.Thread(
                target=self._serve, name="stream-transport", daemon=True
            )
            self._thread.start()

    def stop(self, timeout: float = 5.0) -> StreamCommandResult:
        """Ask for a full teardown, then guarantee it happened."""
        result = self.request_stop(wait_seconds=timeout)
        self._stopping.set()
        self._wake.set()
        with self._lock:
            thread = self._thread
            self._thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(0.1, float(timeout)))
        if self.provider.state is not StreamState.STOPPED:
            self._disconnect()
        with self._lock:
            self._video.clear()
            self._audio.clear()
        return result

    # --------------------------------------------------------------- commands

    def request_start(
        self,
        camera: str = "",
        wait_seconds: Optional[float] = None,
    ) -> StreamCommandResult:
        """Bring the transport up and wait until the stream is genuinely live.

        ``wait_seconds`` is clamped so a caller on the Unity side can bound its
        own response time. The Brain is never blocked by it: the action bridge
        passes a short budget and treats ``TIMEOUT`` as a failed action.
        """
        budget = (
            self.config.live_timeout_seconds if wait_seconds is None else float(wait_seconds)
        )
        budget = max(0.0, min(budget, MAX_WAIT_SECONDS))
        started = self._clock()
        with self._lock:
            if self.provider.state is StreamState.LIVE:
                self._counters["starts"] += 1
                self._failed_start_at = None
                instrument.count("stream.starts")
                return self._command(True, StreamState.LIVE, "ALREADY_LIVE", started)
            if self._stopping.is_set():
                instrument.count("stream.starts_refused")
                return self._command(False, self.provider.state, "SERVICE_STOPPING", started)
            cooling = (
                self._failed_start_at is not None
                and (self._clock() - self._failed_start_at).total_seconds()
                < self.start_backoff_seconds
            )
            if cooling and self.provider.state in {StreamState.IDLE, StreamState.STOPPED, StreamState.ERROR}:
                instrument.count("stream.starts_refused")
                return self._command(False, self.provider.state, "BACKOFF", started)
            self._start_ticket += 1
            ticket = self._start_ticket
            self._pending_start = True
            self._pending_stop = False
            self._last_reason = ""
            self._counters["starts"] += 1
        instrument.count("stream.starts")
        self._wake.set()

        # Wait for the worker to actually consume this request. Without this a
        # provider that has not been touched yet still reads as IDLE, and IDLE
        # counts as "settled", so a start would be reported as STOPPED instantly.
        applied = self._wait(
            lambda: self._start_applied >= ticket
            or self.provider.state is StreamState.LIVE,
            min(budget, 2.0),
        )
        if not applied:
            with self._lock:
                self._failed_start_at = self._clock()
            instrument.count("stream.start_failures")
            instrument.record_error("stream", "WORKER_STALLED")
            return self._command(False, self.provider.state, "WORKER_STALLED", started)
        settled = self._wait(
            lambda: self._start_settled(self.provider.state), budget
        )
        state = self.provider.state
        with self._lock:
            self._start_latency_ms = int((self._clock() - started).total_seconds() * 1000)
            reason = self._last_reason
            frames = self._counters["video_accepted"]
        if state is StreamState.LIVE:
            with self._lock:
                self._failed_start_at = None
            instrument.count("stream.start_success")
            return self._command(True, state, "", started)
        if not reason:
            reason = "TIMEOUT" if settled else self._classify_wait(state)
        if reason == "TIMEOUT" and frames <= 0:
            # Nothing to encode: the capture side never started, so say exactly that.
            reason = "NO_MEDIA"
        with self._lock:
            self._failed_start_at = self._clock()
        instrument.count("stream.start_failures")
        instrument.record_error("stream", reason)
        return self._command(False, state, reason, started)

    def request_stop(self, wait_seconds: Optional[float] = None) -> StreamCommandResult:
        """Tear the transport down. The wait is deliberately best effort.

        A stop must never hang a Brain tick: the caller learns the logical stop
        was accepted even if the encoder needs a moment to die, and
        :meth:`stop` makes the teardown final.
        """
        budget = (
            5.0 if wait_seconds is None else max(0.0, min(float(wait_seconds), MAX_WAIT_SECONDS))
        )
        started = self._clock()
        with self._lock:
            generation = self._generation
            was_idle = self.provider.state in {StreamState.IDLE, StreamState.STOPPED}
            self._pending_stop = True
            self._pending_start = False
            self._counters["stops"] += 1
            # A completed stop is a clean slate: the next start is a new decision.
            self._failed_start_at = None
        instrument.count("stream.stops")
        self._wake.set()
        if not was_idle:
            self._wait(lambda: self._generation > generation, min(budget, 2.0))
        self._wait(lambda: self.provider.state in STOPPED_STATES, budget)
        with self._lock:
            self._video.clear()
            self._audio.clear()
        state = self.provider.state
        ok = state in STOPPED_STATES
        if not ok:
            instrument.count("stream.stop_pending")
            instrument.record_error("stream", "STOP_PENDING")
        return self._command(ok, state, "" if ok else "STOP_PENDING", started)

    @staticmethod
    def _start_settled(state: StreamState) -> bool:
        """Live, or a state that cannot become live without a fresh request."""
        return state in {StreamState.LIVE} or state in STOPPED_STATES

    def _classify_wait(self, state: StreamState) -> str:
        if state is StreamState.ERROR:
            return "PROVIDER_FAILURE"
        if state is StreamState.RECONNECTING:
            return "RECONNECTING"
        if state in {StreamState.STOPPING, StreamState.STOPPED, StreamState.IDLE}:
            return "STOPPED"
        return "TIMEOUT"

    def _command(
        self,
        ok: bool,
        state: StreamState,
        reason: str,
        started: datetime,
    ) -> StreamCommandResult:
        waited = int((self._clock() - started).total_seconds() * 1000)
        with self._lock:
            self._last_reason = reason
        return StreamCommandResult(
            ok=ok,
            state=state,
            reason=reason,
            waited_ms=waited,
            transport_enabled=self.transport_enabled,
        )

    # ------------------------------------------------------------------ media

    def submit_video(
        self,
        payload: bytes,
        sequence: int = 0,
        camera: str = "",
    ) -> IngestResult:
        """Take one encoded picture. Never blocks: it queues or refuses."""
        return self._submit("video", payload, sequence, camera=camera)

    def submit_audio(
        self,
        payload: bytes,
        sequence: int = 0,
        sample_rate: Optional[int] = None,
        channels: Optional[int] = None,
    ) -> IngestResult:
        """Take one raw PCM block for the audio track."""
        return self._submit(
            "audio", payload, sequence, sample_rate=sample_rate, channels=channels
        )

    def _submit(self, kind: str, payload: bytes, sequence: int, **options: Any) -> IngestResult:
        if not isinstance(payload, (bytes, bytearray)) or not payload:
            return self._refuse(kind, "EMPTY_PAYLOAD")
        try:
            number = int(sequence)
        except (TypeError, ValueError):
            return self._refuse(kind, "INVALID_SEQUENCE")
        if number < 0:
            return self._refuse(kind, "INVALID_SEQUENCE")
        if self.provider.state not in STARTABLE_STATES:
            return self._refuse(kind, "NOT_STREAMING")

        video = kind == "video"
        try:
            if video:
                item: Any = VideoFrame(
                    payload=bytes(payload),
                    sequence=number,
                    camera=str(options.get("camera") or "")[:80],
                )
            else:
                item = AudioChunk(
                    payload=bytes(payload),
                    sequence=number,
                    sample_rate=int(options.get("sample_rate") or self.config.sample_rate),
                    channels=int(options.get("channels") or self.config.channels),
                )
        except ValueError:
            return self._refuse(kind, "INVALID_AUDIO" if not video else "INVALID_FRAME")

        limit = self.provider.max_audio_bytes if not video else self.provider.max_frame_bytes
        if item.byte_length > limit:
            return self._refuse(kind, "TOO_LARGE")

        with self._lock:
            queue = self._video if video else self._audio
            last = self._last_video_sequence if video else self._last_audio_sequence
            if number <= last:
                self._counters["duplicates"] += 1
                self._counters[f"{kind}_rejected"] += 1
                return IngestResult(False, "DUPLICATE", len(queue), self._counters[f"{kind}_dropped"])
            if video:
                self._last_video_sequence = number
                self._last_video_at = self._clock()
                self._last_video_frame = item
                if item.camera:
                    self._last_camera = item.camera
            else:
                self._last_audio_sequence = number
                self._last_audio_at = self._clock()
            dropped = 1 if len(queue) == queue.maxlen else 0
            queue.append(item)
            self._counters[f"{kind}_accepted"] += 1
            self._counters[f"{kind}_dropped"] += dropped
            queued = len(queue)
            total_dropped = self._counters[f"{kind}_dropped"]
        instrument.gauge(f"queue.stream.{kind}", float(queued))
        if dropped:
            instrument.count(f"stream.{kind}_dropped")
        self._wake.set()
        return IngestResult(True, "", queued, total_dropped)

    def _refuse(self, kind: str, reason: str) -> IngestResult:
        with self._lock:
            self._counters[f"{kind}_rejected"] += 1
            queue = self._video if kind == "video" else self._audio
            depth = len(queue)
            dropped = self._counters[f"{kind}_dropped"]
        instrument.count(f"stream.{kind}_refused")
        return IngestResult(False, reason, depth, dropped)

    # ------------------------------------------------------------ diagnostics

    def _wait(
        self,
        predicate: Callable[[], bool],
        timeout: float,
    ) -> bool:
        """Poll until ``predicate`` holds or the budget runs out."""
        deadline = self._clock() + timedelta(seconds=max(0.0, min(float(timeout), MAX_WAIT_SECONDS)))
        while True:
            try:
                if predicate():
                    return True
            except Exception:  # pragma: no cover - defensive
                return False
            if self._clock() >= deadline:
                return False
            time.sleep(WAIT_TICK_SECONDS)

    def wait_for_state(
        self,
        predicate: Callable[[StreamState], bool],
        timeout: float,
    ) -> StreamState:
        """Public helper for tests and callers: wait on a provider state."""
        self._wait(lambda: bool(predicate(self.provider.state)), timeout)
        return self.provider.state

    def get_state(self) -> dict[str, Any]:
        health = self.health()
        with self._lock:
            counters = dict(self._counters)
            last_reason = self._last_reason
            start_latency = self._start_latency_ms
            last_camera = self._last_camera
            last_video_at = _iso(self._last_video_at)
            last_audio_at = _iso(self._last_audio_at)
            started_at = _iso(self._started_at)
            video_queue = len(self._video)
            audio_queue = len(self._audio)
        return {
            "provider": self.provider.name,
            "platform": getattr(self.provider, "platform", "unknown"),
            "transport_enabled": self.transport_enabled,
            "real_transport": self.config.is_real,
            "enabled": True,
            "state": health["state"],
            "health": health,
            "config": self.config.to_dict(),
            "ingest": self.ingest_profile(),
            "queues": {"video": video_queue, "audio": audio_queue},
            "counters": counters,
            "last_reason": last_reason,
            "last_camera": last_camera,
            "last_video_at": last_video_at,
            "last_audio_at": last_audio_at,
            "start_latency_ms": start_latency,
            "started_at": started_at,
        }

    def ingest_profile(self) -> dict[str, Any]:
        """What the capture side must produce. No credentials, ever."""
        return {
            "width": self.config.width,
            "height": self.config.height,
            "fps": self.config.fps,
            "jpeg_quality": self.config.jpeg_quality,
            "audio_sample_rate": self.config.sample_rate,
            "audio_channels": self.config.channels,
            "video_queue_capacity": self._video.maxlen,
            "audio_queue_capacity": self._audio.maxlen,
            "max_frame_bytes": self.provider.max_frame_bytes,
            "max_audio_bytes": self.provider.max_audio_bytes,
        }

    def health(self) -> dict[str, Any]:
        snapshot = self._provider_health()
        now = self._clock()
        state = self.provider.state
        with self._lock:
            video_age = _age(now, self._last_video_at)
            audio_age = _age(now, self._last_audio_at)
            dropped = self._counters["video_dropped"] + self._counters["audio_dropped"]
            frames_ingested = self._counters["video_accepted"]
            last_reason = self._last_reason
        want_media = state in ACTIVE_STATES or state is StreamState.CONNECTING
        payload = replace(
            snapshot,
            state=state,
            frames_ingested=frames_ingested,
            dropped_frames=max(snapshot.dropped_frames + dropped, 0),
            video_state=_freshness(want_media, video_age),
            audio_state=_freshness(want_media, audio_age),
            last_error=snapshot.last_error or (last_reason or None),
            started_at=snapshot.started_at or self._started_at,
        ).to_dict()
        payload["queue_video"] = self._depth("video")
        payload["queue_audio"] = self._depth("audio")
        with self._lock:
            payload["frames_repeated"] = self._counters["video_repeated"]
        payload["ok"] = state is StreamState.LIVE
        instrument.gauge("stream.live", 1.0 if state is StreamState.LIVE else 0.0)
        instrument.gauge("stream.uptime_seconds", self._worker_uptime_seconds(now))
        reconnect_count = payload.get("reconnect_count")
        if isinstance(reconnect_count, (int, float)) and not isinstance(reconnect_count, bool):
            instrument.gauge("stream.reconnect_count", float(reconnect_count))
        instrument.gauge("queue.stream.video", float(payload["queue_video"]["length"]))
        instrument.gauge("queue.stream.audio", float(payload["queue_audio"]["length"]))
        return payload

    def _worker_uptime_seconds(self, now: Optional[datetime] = None) -> float:
        """Seconds since the transport worker started; 0.0 before it exists."""
        with self._lock:
            started_at = self._started_at
        if started_at is None:
            return 0.0
        return max(0.0, ((now or self._clock()) - started_at).total_seconds())

    def _provider_health(self) -> StreamHealth:
        try:
            return self.provider.health()
        except Exception:  # pragma: no cover - defensive
            return StreamHealth(
                state=self.provider.state, transport_enabled=self.transport_enabled
            )

    def _depth(self, kind: str) -> dict[str, Any]:
        with self._lock:
            queue = self._video if kind == "video" else self._audio
            return {
                "length": len(queue),
                "capacity": queue.maxlen,
                "dropped": self._counters[f"{kind}_dropped"],
            }

    # ----------------------------------------------------------------- worker

    def _serve(self) -> None:
        while not self._stopping.is_set():
            try:
                self.run_once(self.poll_interval_seconds)
            except Exception as exc:  # pragma: no cover - never escape a thread
                with self._lock:
                    self._last_reason = type(exc).__name__
                self._generation += 1
        self._disconnect()

    def run_once(self, timeout: float = DEFAULT_POLL_INTERVAL_SECONDS) -> bool:
        """One worker pass: honour intent, push media, refresh state. Testable."""
        work = False
        with self._lock:
            pending_start = self._pending_start
            pending_stop = self._pending_stop

        if pending_stop:
            with self._lock:
                self._pending_stop = False
                self._pending_start = False
                # A stop supersedes any queued start ticket, so a waiting caller
                # sees the settled state instead of timing out.
                self._start_applied = self._start_ticket
            self._disconnect()
            work = True
        elif pending_start:
            with self._lock:
                self._pending_start = False
                self._start_applied = self._start_ticket
                # Media sequence numbers are per capture session, not per process:
                # a fresh Unity/encoder session restarts at zero, so the old
                # high-water marks would reject every frame as DUPLICATE forever.
                self._last_video_sequence = -1
                self._last_audio_sequence = -1
                self._last_video_at = None
                self._last_audio_at = None
                self._last_video_frame = None
                self._next_paced_frame_at = None
            work = self._connect() or work

        if self.provider.state in STARTABLE_STATES and not self._stopping.is_set():
            # The pacer runs before the drain: a repeat is paired with an audio
            # chunk, and draining first leaves the audio queue empty, so the
            # repeat owed by a capture that runs under the declared rate could
            # never be sent.
            work = self._repeat_last_video() or work
            work = self._drain() or work

        try:
            self.provider.poll()
        except StreamingProviderError as exc:
            with self._lock:
                self._counters["connect_errors"] += 1
                self._last_reason = exc.reason
            instrument.count("stream.poll_errors")
            instrument.record_error("stream", exc.reason)
        except Exception as exc:  # pragma: no cover - defensive
            with self._lock:
                self._counters["connect_errors"] += 1
                self._last_reason = type(exc).__name__
            instrument.count("stream.poll_errors")
            instrument.record_error("stream", exc)

        with self._lock:
            self._counters["polls"] += 1
            self._generation += 1
        if not work and not self._stopping.is_set():
            self._wake.wait(max(0.001, float(timeout)))
            self._wake.clear()
        return work

    def _drain(self) -> bool:
        """Push queued media to the provider, alternating video and audio.

        Draining every queued picture before the first audio block lets the
        encoder's video input run ahead of its audio input. Once ffmpeg stops
        reading video to let audio catch up, the worker can block in
        ``write_video`` before it reaches the audio queue. Alternating keeps
        both muxer tracks fed while still returning once neither queue has
        work.
        """
        moved = False
        while True:
            frame = self._take("video")
            chunk = self._take("audio")
            if frame is None and chunk is None:
                break
            moved = True
            video_ok = True
            audio_ok = True
            if frame is not None:
                video_ok = self.provider.write_video(frame)
                if video_ok:
                    self._note_video_emitted()
            if chunk is not None:
                audio_ok = self.provider.write_audio(chunk)
            if not video_ok or not audio_ok:
                # At most one of each was consumed. Leave the remainder queued
                # so a provider that stopped accepting media cannot drain the
                # buffers or hide a write failure.
                break
        return moved

    def _connect(self) -> bool:
        try:
            self.provider.connect()
        except StreamingProviderError as exc:
            with self._lock:
                self._counters["connect_errors"] += 1
                self._last_reason = exc.reason
            instrument.count("stream.connect_errors")
            instrument.record_error("stream", exc.reason)
            return False
        except Exception as exc:  # pragma: no cover - defensive
            with self._lock:
                self._counters["connect_errors"] += 1
                self._last_reason = type(exc).__name__
            instrument.count("stream.connect_errors")
            instrument.record_error("stream", exc)
            return False
        return True

    def _disconnect(self) -> bool:
        try:
            self.provider.disconnect()
        except Exception:  # pragma: no cover - disconnect never raises
            return False
        return True

    def _take(self, kind: str) -> Optional[Any]:
        with self._lock:
            queue = self._video if kind == "video" else self._audio
            if not queue:
                return None
            return queue.popleft()

    def _note_video_emitted(self) -> None:
        """Advance the cadence clock by one slot for a written picture.

        Real captures and repeats each consume exactly one slot, so the
        picture timeline advances at the declared frame rate even when the
        capture itself runs under it.
        """
        interval = timedelta(seconds=1.0 / max(1.0, float(self.config.fps)))
        now = self._clock()
        with self._lock:
            scheduled = self._next_paced_frame_at
            if scheduled is None:
                scheduled = now
            self._next_paced_frame_at = scheduled + interval

    def _repeat_last_video(self) -> bool:
        """Hold the picture track at the configured cadence.

        Unity captures on its main thread, so real capture runs a little under
        the configured frame rate with jittery gaps. The audio track, however,
        arrives at exact real time. Without pacing, the video timeline falls
        further behind the audio timeline, the muxer buffers the leading audio
        until ffmpeg stops reading stdin, and the write watchdog kills an
        encoder that was never broken. Re-sending the newest accepted picture
        fills the cadence slots the capture missed, exactly like a CFR pacer.
        Nothing is invented: with no accepted frame yet, this is a no-op.

        The cadence is a frame-count clock that every written picture consumes
        (see ``_note_video_emitted``), not a deadline re-anchored on each new
        capture. Re-anchoring meant a capture slower than the declared rate --
        exactly Unity's main-thread capture -- never let the clock fall behind,
        so no repeat was ever owed: the video timeline advanced at the capture
        rate while audio stayed at real time, and the muxer eventually parked
        the worker on a blocked audio socket with the queues still looking
        healthy.
        """
        interval = timedelta(seconds=1.0 / max(1.0, float(self.config.fps)))
        now = self._clock()
        with self._lock:
            frame = self._last_video_frame
            scheduled = self._next_paced_frame_at
            if frame is None or scheduled is None:
                return False
            if now < scheduled:
                return False
            # A repeat picture cannot be sent on its own. If the encoder is
            # waiting for audio, the video write parks this worker on stdin
            # before the audio queue can be drained. Feed an audio chunk first
            # when one is available; otherwise defer the repeat until the next
            # capture/audio wake-up.
            if not self._audio:
                return False
            chunk = self._audio.popleft()
            # After a long hitch, resync instead of replaying the backlog.
            burst_limit = interval * REPEAT_BURST_LIMIT_INTERVALS
            if now - scheduled > burst_limit:
                scheduled = now
            self._next_paced_frame_at = scheduled + interval
            self._counters["video_repeated"] += 1
        if not self.provider.write_audio(chunk):
            return False
        return self.provider.write_video(frame)


def build_streaming_service(
    environ: Optional[Any] = None,
    config: StreamingConfig | None = None,
    provider: StreamingProvider | None = None,
    start_worker: bool = True,
) -> Optional[StreamingService]:
    """Create the configured transport service, or ``None`` when streaming is off.

    Mirrors ``build_chat_source``: PHASE 01-12 behaviour is unchanged unless
    ``STREAM_PROVIDER`` (or a configured endpoint) opts into a real transport.
    """
    resolved = config or StreamingConfig.from_environment(environ)
    if provider is None:
        if not resolved.enabled:
            return None
        if resolved.is_real and not looks_like_endpoint(resolved.server_url):
            return None
        provider = build_provider(resolved)
    if provider is None:
        return None
    return StreamingService(provider, resolved, start_worker=start_worker)


def _freshness(want_media: bool, age: Optional[float]) -> str:
    if not want_media or age is None:
        return "NO_INPUT"
    return "OK" if age <= INPUT_FRESHNESS_SECONDS else "STALE"


def _iso(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _age(now: datetime, then: Optional[datetime]) -> Optional[float]:
    if then is None:
        return None
    return max(0.0, (now - then).total_seconds())
