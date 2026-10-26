"""Streaming transport abstraction shared by the mock and real providers.

PHASE 13 splits *logical stream state* (Unity's LivestreamSystem, PHASE 04) from
*real streaming transport* (this package). A provider owns connection,
encoding, ingest of video/audio, health, and reconnect. It never reasons about
goals or actions and never talks to the Brain, so a dead platform can degrade
the stream without touching the character.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


MAX_ERROR_DETAIL = 200
REDACTED = "[redacted]"


class StreamState(str, Enum):
    """Transport lifecycle. ``IDLE``/``STOPPED`` are both "not streaming"."""

    IDLE = "IDLE"
    CONNECTING = "CONNECTING"
    LIVE = "LIVE"
    RECONNECTING = "RECONNECTING"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


#: States where a stream is usable end to end.
ACTIVE_STATES = frozenset({StreamState.LIVE, StreamState.RECONNECTING})


class StreamingProviderError(RuntimeError):
    """Raised for connection, authentication, encoder, and transport failures.

    ``detail`` must already be credential-free: providers scrub before raising
    so a key can never reach a log line or an API response by accident.
    """

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}:{detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class ReconnectPolicy:
    """Bounded exponential backoff, so a platform outage cannot spin the CPU.

    Mirrors the PHASE 12 chat policy. Deliberately redeclared rather than
    imported: the streaming layer must not depend on the chat package.
    """

    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 20.0
    max_attempts: int = 4

    def __post_init__(self) -> None:
        if self.base_delay_seconds <= 0:
            raise ValueError("base_delay_seconds must be positive")
        if self.max_delay_seconds < self.base_delay_seconds:
            raise ValueError("max_delay_seconds must be >= base_delay_seconds")
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be positive")

    def delay_for_attempt(self, attempt: int) -> float:
        """Delay before ``attempt`` (1-based), without randomness."""
        exponent = max(0, int(attempt) - 1)
        delay = self.base_delay_seconds * (2 ** min(exponent, 16))
        return round(min(self.max_delay_seconds, delay), 3)

    def allows_attempt(self, attempt: int) -> bool:
        return int(attempt) <= self.max_attempts

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_delay_seconds": self.base_delay_seconds,
            "max_delay_seconds": self.max_delay_seconds,
            "max_attempts": self.max_attempts,
        }


def to_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def iso_seconds(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    return to_utc(value).isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class VideoFrame:
    """One encoded picture handed over by the capture side."""

    payload: bytes
    sequence: int = 0
    camera: str = ""
    source: str = "unity"

    def __post_init__(self) -> None:
        if not isinstance(self.payload, (bytes, bytearray)):
            raise ValueError("payload must be bytes")
        if self.sequence < 0:
            raise ValueError("sequence must not be negative")

    @property
    def byte_length(self) -> int:
        return len(self.payload)

    def to_dict(self) -> dict[str, Any]:
        """Diagnostics view. Never carries the picture itself."""
        return {
            "sequence": self.sequence,
            "byte_length": self.byte_length,
            "camera": self.camera,
            "source": self.source,
        }


@dataclass(frozen=True)
class AudioChunk:
    """One block of raw PCM (signed 16-bit little endian) for the audio track."""

    payload: bytes
    sample_rate: int
    channels: int = 1
    sequence: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.payload, (bytes, bytearray)):
            raise ValueError("payload must be bytes")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.channels not in (1, 2):
            raise ValueError("channels must be 1 or 2")
        if len(self.payload) % (2 * self.channels):
            raise ValueError("payload must hold whole 16-bit samples")

    @property
    def byte_length(self) -> int:
        return len(self.payload)

    @property
    def duration_seconds(self) -> float:
        frames = self.byte_length / (2 * self.channels)
        return round(frames / self.sample_rate, 4)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "byte_length": self.byte_length,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class StreamHealth:
    """Lightweight stream health, assembled from provider and ingest facts."""

    state: StreamState = StreamState.IDLE
    provider: str = "unknown"
    platform: str = "unknown"
    resolution: str = ""
    target_fps: float = 0.0
    encoder_fps: float = 0.0
    bitrate_kbps: float = 0.0
    uptime_seconds: float = 0.0
    frames_ingested: int = 0
    frames_encoded: int = 0
    dropped_frames: int = 0
    audio_seconds_ingested: float = 0.0
    video_state: str = "NO_INPUT"
    audio_state: str = "NO_INPUT"
    reconnect_count: int = 0
    connect_attempts: int = 0
    last_error: Optional[str] = None
    live_since: Optional[datetime] = None
    started_at: Optional[datetime] = None
    transport_enabled: bool = False
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def is_live(self) -> bool:
        return self.state is StreamState.LIVE

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "state": self.state.value,
            "live": self.state is StreamState.LIVE,
            "provider": self.provider,
            "platform": self.platform,
            "transport_enabled": self.transport_enabled,
            "resolution": self.resolution,
            "target_fps": round(self.target_fps, 2),
            "encoder_fps": round(self.encoder_fps, 2),
            "bitrate_kbps": round(self.bitrate_kbps, 1),
            "uptime_seconds": round(self.uptime_seconds, 2),
            "frames_ingested": self.frames_ingested,
            "frames_encoded": self.frames_encoded,
            "dropped_frames": self.dropped_frames,
            "audio_seconds_ingested": round(self.audio_seconds_ingested, 2),
            "video_state": self.video_state,
            "audio_state": self.audio_state,
            "reconnect_count": self.reconnect_count,
            "connect_attempts": self.connect_attempts,
            "last_error": self.last_error,
            "live_since": iso_seconds(self.live_since),
            "started_at": iso_seconds(self.started_at),
        }
        payload.update(self.attributes)
        return payload


class StreamingProvider(ABC):
    """Transport boundary for one streaming platform.

    Providers may raise :class:`StreamingProviderError` out of
    :meth:`connect`; :meth:`write_video`, :meth:`write_audio`, and
    :meth:`poll` must not raise for ordinary network trouble. They move to
    :attr:`StreamState.RECONNECTING` or :attr:`StreamState.ERROR` and report
    through :meth:`health` instead, so the streaming layer stays alive and the
    character keeps running.
    """

    name: str = "base"
    platform: str = "unknown"
    #: True only when this provider actually moves encoded media to a platform.
    transport_enabled: bool = False
    #: Maximum accepted size of one encoded picture.
    max_frame_bytes: int = 768 * 1024
    #: Maximum accepted size of one raw audio block.
    max_audio_bytes: int = 512 * 1024

    @abstractmethod
    def connect(self) -> None:
        """Bring the transport up. Raises :class:`StreamingProviderError`.

        Returning means the output side is established; providers that need
        media before the platform counts the stream as delivered stay in
        :attr:`StreamState.CONNECTING` until :meth:`poll` confirms it.
        """

    @abstractmethod
    def disconnect(self) -> None:
        """Tear the transport down and release every resource. Never raises."""

    @abstractmethod
    def write_video(self, frame: VideoFrame) -> bool:
        """Hand one picture to the encoder. False when it was not accepted."""

    @abstractmethod
    def write_audio(self, chunk: AudioChunk) -> bool:
        """Hand one PCM block to the encoder. False when it was not accepted."""

    @abstractmethod
    def poll(self) -> StreamState:
        """Refresh state without blocking: liveness, failures, reconnects."""

    @abstractmethod
    def health(self) -> StreamHealth:
        """Current health snapshot. Structurally incapable of carrying secrets."""

    @property
    @abstractmethod
    def state(self) -> StreamState:
        ...

    def diagnostics(self) -> dict[str, Any]:
        """Connection facts only. Never credentials, URLs with keys, or media."""
        return {
            "provider": self.name,
            "platform": self.platform,
            "state": self.state.value,
            "transport_enabled": self.transport_enabled,
        }
