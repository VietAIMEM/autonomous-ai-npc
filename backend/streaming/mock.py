"""Deterministic in-process streaming provider.

The mock keeps the whole PHASE 13 lifecycle reachable without ffmpeg, without a
platform account, and without a network: it counts what it is handed, advances
through the same states as the real transport, and can be told to fail in the
same places the real one fails. Every number it reports comes from media it
actually accepted, so health stays honest even when simulated.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from .base import (
    AudioChunk,
    StreamHealth,
    StreamingProvider,
    StreamingProviderError,
    StreamState,
    VideoFrame,
)
from .config import StreamingConfig


class MockStreamingProvider(StreamingProvider):
    """A provider that behaves like a platform without becoming one."""

    name = "mock"
    platform = "simulated"
    transport_enabled = True

    def __init__(
        self,
        config: StreamingConfig | None = None,
        connect_failure: Optional[str] = None,
        fail_after_seconds: Optional[float] = None,
        reconnect_failure: Optional[str] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config or StreamingConfig(provider="mock")
        #: Reason code raised out of :meth:`connect`, when configured.
        self.connect_failure = connect_failure
        #: Drop the connection this many seconds after it went live, once.
        self.fail_after_seconds = fail_after_seconds
        #: Reason code used for the first reconnect attempt, when configured.
        self.reconnect_failure = reconnect_failure
        self._clock = clock
        self._state = StreamState.IDLE
        self._lock_holder = None
        self._live_since: Optional[float] = None
        self._started_at: Optional[float] = None
        self._live_wall: Optional[datetime] = None
        self._start_wall: Optional[datetime] = None
        self._frames = 0
        self._frame_bytes = 0
        self._audio_seconds = 0.0
        self._audio_bytes = 0
        self._reconnect_count = 0
        self._reconnect_attempt = 0
        self._next_attempt_at = 0.0
        self._ever_live = False
        self._connect_attempts = 0
        self._dropped = 0
        self._last_error: Optional[str] = None
        self._dropped_once = False

    # -- lifecycle ---------------------------------------------------------

    def connect(self) -> None:
        if self._state in {StreamState.CONNECTING, StreamState.LIVE}:
            return
        self._connect_attempts += 1
        failure = self.connect_failure
        if self._reconnect_attempt and self.reconnect_failure is not None:
            failure, self.reconnect_failure = self.reconnect_failure, None
        if failure:
            self._last_error = failure
            # A failed *reconnect* attempt leaves the state alone so the bounded
            # backoff in :meth:`poll` can keep trying; a failed first start is terminal.
            if not self._reconnect_attempt:
                self._state = StreamState.ERROR
            raise StreamingProviderError(failure, "mock provider refused the connection")
        self._last_error = None
        self._state = StreamState.CONNECTING
        if self._started_at is None:
            self._started_at = self._clock()
            self._start_wall = datetime.now(timezone.utc)

    def disconnect(self) -> None:
        self._state = StreamState.STOPPING
        self._live_since = None
        self._live_wall = None
        self._state = StreamState.STOPPED
        self._reconnect_attempt = 0

    # -- media -------------------------------------------------------------

    def write_video(self, frame: VideoFrame) -> bool:
        if self._state not in {StreamState.CONNECTING, StreamState.LIVE}:
            self._dropped += 1
            return False
        self._frames += 1
        self._frame_bytes += frame.byte_length
        return True

    def write_audio(self, chunk: AudioChunk) -> bool:
        if self._state not in {StreamState.CONNECTING, StreamState.LIVE}:
            self._dropped += 1
            return False
        self._audio_seconds += chunk.duration_seconds
        self._audio_bytes += chunk.byte_length
        return True

    # -- state -------------------------------------------------------------

    def poll(self) -> StreamState:
        now = self._clock()
        if self._state is StreamState.CONNECTING:
            # Like the real transport, "live" needs media actually flowing.
            if self._frames > 0 and self._audio_seconds > 0:
                self._state = StreamState.LIVE
                self._live_since = now
                self._live_wall = datetime.now(timezone.utc)
                self._ever_live = True
                # Count a reconnect only when it actually restored the stream,
                # matching FFmpegStreamingProvider._promote_live.
                if self._reconnect_attempt:
                    self._reconnect_count += 1
                    self._reconnect_attempt = 0
            return self._state
        if self._state is StreamState.LIVE:
            if (
                self.fail_after_seconds is not None
                and not self._dropped_once
                and self._live_since is not None
                and now - self._live_since >= self.fail_after_seconds
            ):
                self._dropped_once = True
                self._fail("CONNECTION_FAILED", now)
            elif self._frames == 0:
                self._state = StreamState.ERROR
                self._last_error = "NO_MEDIA"
            return self._state
        if self._state is StreamState.RECONNECTING:
            if now < self._next_attempt_at:
                return self._state
            if not self.config.reconnect.allows_attempt(self._reconnect_attempt + 1):
                self._state = StreamState.ERROR
                self._last_error = self._last_error or "RECONNECT_EXHAUSTED"
                return self._state
            try:
                self.connect()
            except StreamingProviderError as exc:
                # poll() must never raise. ``connect`` already armed the next
                # attempt, or moved us to ERROR when the budget ran out.
                self._last_error = exc.reason
                self._reconnect_attempt += 1
                self._next_attempt_at = now + self.config.reconnect.delay_for_attempt(
                    self._reconnect_attempt
                )
            return self._state
        return self._state

    def _fail(self, reason: str, now: Optional[float] = None) -> None:
        """Arm a bounded reconnect, matching :class:`FFmpegStreamingProvider`."""
        moment = self._clock() if now is None else now
        policy = self.config.reconnect
        self._last_error = reason
        self._live_since = None
        self._live_wall = None
        if self._ever_live and policy.allows_attempt(self._reconnect_attempt + 1):
            self._reconnect_attempt += 1
            self._next_attempt_at = moment + policy.delay_for_attempt(self._reconnect_attempt)
            self._state = StreamState.RECONNECTING
        else:
            self._state = StreamState.ERROR

    @property
    def state(self) -> StreamState:
        return self._state

    def uptime_seconds(self) -> float:
        if self._started_at is None or self._state not in {
            StreamState.LIVE,
            StreamState.RECONNECTING,
            StreamState.CONNECTING,
        }:
            return 0.0
        return round(self._clock() - self._started_at, 2)

    def health(self) -> StreamHealth:
        elapsed = self.uptime_seconds()
        return StreamHealth(
            state=self._state,
            provider=self.name,
            platform=self.platform,
            resolution=self.config.resolution,
            target_fps=self.config.fps,
            encoder_fps=round(self._frames / elapsed, 2) if elapsed else 0.0,
            bitrate_kbps=round(self._frame_bytes * 8 / 1000 / elapsed, 1) if elapsed else 0.0,
            uptime_seconds=elapsed,
            frames_encoded=self._frames,
            dropped_frames=self._dropped,
            audio_seconds_ingested=round(self._audio_seconds, 2),
            video_state="OK" if self._frames else "NO_INPUT",
            audio_state="OK" if self._audio_seconds else "NO_INPUT",
            reconnect_count=self._reconnect_count,
            connect_attempts=self._connect_attempts,
            last_error=self._last_error,
            live_since=self._live_wall,
            started_at=self._start_wall,
            transport_enabled=True,
            attributes={"simulated": True, "audio_bytes_ingested": self._audio_bytes},
        )

    def diagnostics(self) -> dict[str, Any]:
        payload = dict(super().diagnostics())
        payload.update(
            {
                "simulated": True,
                "resolution": self.config.resolution,
                "fps": self.config.fps,
                "reconnect": self.config.reconnect.to_dict(),
            }
        )
        return payload
