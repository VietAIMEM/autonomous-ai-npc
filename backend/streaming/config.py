"""Stream transport selection and provider construction.

Read like ``backend/tts/config.py`` and ``backend/chat/providers/config.py``:
the stream key is only ever read from disk inside the real provider. Nothing in
this module returns key material through a dataclass, ``repr``, or diagnostics
payload, and the full ingest URL (which contains the key once combined) is never
stored on a public attribute.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Optional
from urllib.parse import urlsplit

from .base import ReconnectPolicy, StreamingProvider


PROVIDER_DISABLED = "disabled"
PROVIDER_MOCK = "mock"
PROVIDER_RTMP = "rtmp"
PROVIDER_SRT = "srt"
VALID_PROVIDERS = (PROVIDER_DISABLED, PROVIDER_MOCK, PROVIDER_RTMP, PROVIDER_SRT)

DEFAULT_PLATFORM = "twitch"
#: Twitch application ingest. Closest documented edge for this project's region.
DEFAULT_SERVER_URL = "rtmp://live-sin04.twitch.tv/app"
DEFAULT_KEY_PATH = Path(__file__).resolve().parents[1] / "stream_key.txt"
DEFAULT_URL_PATH = Path(__file__).resolve().parents[1] / "stream_url.txt"

DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 360
DEFAULT_FPS = 15.0
DEFAULT_VIDEO_BITRATE_KBPS = 1500
DEFAULT_AUDIO_BITRATE_KBPS = 96
DEFAULT_SAMPLE_RATE = 48000
DEFAULT_AUDIO_CHANNELS = 1
DEFAULT_JPEG_QUALITY = 70
DEFAULT_CONNECT_TIMEOUT_SECONDS = 10.0
DEFAULT_LIVE_TIMEOUT_SECONDS = 25.0
DEFAULT_WRITE_TIMEOUT_SECONDS = 3.0
DEFAULT_VIDEO_QUEUE_CAPACITY = 6
DEFAULT_AUDIO_QUEUE_CAPACITY = 12
#: A failed start is retried no sooner than this, so a bad key cannot make the
#: Brain spawn encoders on every tick.
DEFAULT_START_BACKOFF_SECONDS = 5.0


def endpoint_host(value: str) -> str:
    """Scheme + host only. Safe for logs; the path may carry a key."""
    text = (value or "").strip()
    if not text:
        return ""
    parts = urlsplit(text)
    if parts.scheme and parts.netloc:
        host = parts.netloc.rsplit("@", 1)[-1]
        return f"{parts.scheme}://{host}"
    return ""


def looks_like_endpoint(value: str) -> bool:
    text = (value or "").strip()
    if not text:
        return False
    return "://" in text or bool(Path(text).suffix)


@dataclass(frozen=True)
class StreamingConfig:
    provider: str = PROVIDER_DISABLED
    platform: str = DEFAULT_PLATFORM
    server_url: str = DEFAULT_SERVER_URL
    stream_key_path: Path | str = DEFAULT_KEY_PATH
    ffmpeg_path: str = ""
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    fps: float = DEFAULT_FPS
    video_bitrate_kbps: int = DEFAULT_VIDEO_BITRATE_KBPS
    audio_bitrate_kbps: int = DEFAULT_AUDIO_BITRATE_KBPS
    sample_rate: int = DEFAULT_SAMPLE_RATE
    channels: int = DEFAULT_AUDIO_CHANNELS
    keyframe_interval: int = 0
    jpeg_quality: int = DEFAULT_JPEG_QUALITY
    connect_timeout_seconds: float = DEFAULT_CONNECT_TIMEOUT_SECONDS
    live_timeout_seconds: float = DEFAULT_LIVE_TIMEOUT_SECONDS
    write_timeout_seconds: float = DEFAULT_WRITE_TIMEOUT_SECONDS
    start_backoff_seconds: float = DEFAULT_START_BACKOFF_SECONDS
    video_queue_capacity: int = DEFAULT_VIDEO_QUEUE_CAPACITY
    audio_queue_capacity: int = DEFAULT_AUDIO_QUEUE_CAPACITY
    reconnect: ReconnectPolicy = field(default_factory=ReconnectPolicy)

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        if not 1 <= self.fps <= 60:
            raise ValueError("fps must be between 1 and 60")
        if self.channels not in (1, 2):
            raise ValueError("channels must be 1 or 2")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.video_queue_capacity <= 0 or self.audio_queue_capacity <= 0:
            raise ValueError("queue capacities must be positive")

    @property
    def enabled(self) -> bool:
        return self.provider != PROVIDER_DISABLED

    @property
    def is_real(self) -> bool:
        return self.provider in {PROVIDER_RTMP, PROVIDER_SRT}

    @property
    def resolution(self) -> str:
        return f"{self.width}x{self.height}"

    @property
    def container(self) -> str:
        """Muxer chosen by protocol: FLV for RTMP, MPEG-TS for SRT."""
        return "mpegts" if self.provider == PROVIDER_SRT else "flv"

    @property
    def effective_keyframe_interval(self) -> int:
        return self.keyframe_interval or max(1, int(round(self.fps * 2)))

    def sanitized(self) -> "StreamingConfig":
        """A copy that is safe to log: no key material, no key directory."""
        return replace(
            self,
            server_url=endpoint_host(self.server_url) or "(configured)",
            stream_key_path=Path(self.stream_key_path).name,
            ffmpeg_path=Path(self.ffmpeg_path).name if self.ffmpeg_path else "",
        )

    def __repr__(self) -> str:
        safe = self.sanitized()
        return (
            "StreamingConfig(provider={!r}, platform={!r}, server_url={!r}, "
            "stream_key_path={!r}, resolution={!r}, fps={!r})"
        ).format(
            safe.provider,
            safe.platform,
            safe.server_url,
            str(safe.stream_key_path),
            safe.resolution,
            safe.fps,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "platform": self.platform,
            "enabled": self.enabled,
            "real_transport": self.is_real,
            "endpoint": endpoint_host(self.server_url),
            "key_file": Path(self.stream_key_path).name,
            "container": self.container,
            "resolution": self.resolution,
            "fps": self.fps,
            "video_bitrate_kbps": self.video_bitrate_kbps,
            "audio_bitrate_kbps": self.audio_bitrate_kbps,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "keyframe_interval": self.effective_keyframe_interval,
            "jpeg_quality": self.jpeg_quality,
            "live_timeout_seconds": self.live_timeout_seconds,
            "start_backoff_seconds": self.start_backoff_seconds,
            "video_queue_capacity": self.video_queue_capacity,
            "audio_queue_capacity": self.audio_queue_capacity,
            "reconnect": self.reconnect.to_dict(),
        }

    @classmethod
    def from_environment(
        cls, environ: Mapping[str, str] | None = None
    ) -> "StreamingConfig":
        env = os.environ if environ is None else environ
        requested = (env.get("STREAM_PROVIDER") or "").strip().lower()
        configured_url = (env.get("STREAM_SERVER_URL") or "").strip()
        if not configured_url:
            configured_url = _read_text(
                Path(env.get("STREAM_SERVER_URL_PATH") or DEFAULT_URL_PATH)
            )
        if not requested:
            # Unset means "no real transport", so PHASE 01-12 runs are unchanged.
            requested = (
                PROVIDER_RTMP if looks_like_endpoint(configured_url) else PROVIDER_DISABLED
            )
        if requested not in VALID_PROVIDERS:
            requested = PROVIDER_DISABLED

        server_url = configured_url or DEFAULT_SERVER_URL
        return cls(
            provider=requested,
            platform=(env.get("STREAM_PLATFORM") or DEFAULT_PLATFORM).strip().lower(),
            server_url=server_url,
            stream_key_path=Path(env.get("STREAM_KEY_PATH") or DEFAULT_KEY_PATH),
            ffmpeg_path=(env.get("STREAM_FFMPEG_PATH") or "").strip(),
            width=int(_as_float(env.get("STREAM_WIDTH"), DEFAULT_WIDTH)),
            height=int(_as_float(env.get("STREAM_HEIGHT"), DEFAULT_HEIGHT)),
            fps=_as_float(env.get("STREAM_FPS"), DEFAULT_FPS),
            video_bitrate_kbps=int(
                _as_float(
                    env.get("STREAM_VIDEO_BITRATE_KBPS"), DEFAULT_VIDEO_BITRATE_KBPS
                )
            ),
            audio_bitrate_kbps=int(
                _as_float(
                    env.get("STREAM_AUDIO_BITRATE_KBPS"), DEFAULT_AUDIO_BITRATE_KBPS
                )
            ),
            sample_rate=int(_as_float(env.get("STREAM_SAMPLE_RATE"), DEFAULT_SAMPLE_RATE)),
            channels=int(_as_float(env.get("STREAM_AUDIO_CHANNELS"), DEFAULT_AUDIO_CHANNELS)),
            keyframe_interval=_as_positive_int(env.get("STREAM_KEYFRAME_INTERVAL"), 0),
            jpeg_quality=int(_as_float(env.get("STREAM_JPEG_QUALITY"), DEFAULT_JPEG_QUALITY)),
            connect_timeout_seconds=_as_float(
                env.get("STREAM_CONNECT_TIMEOUT_SECONDS"), DEFAULT_CONNECT_TIMEOUT_SECONDS
            ),
            live_timeout_seconds=_as_float(
                env.get("STREAM_LIVE_TIMEOUT_SECONDS"), DEFAULT_LIVE_TIMEOUT_SECONDS
            ),
            write_timeout_seconds=_as_float(
                env.get("STREAM_WRITE_TIMEOUT_SECONDS"), DEFAULT_WRITE_TIMEOUT_SECONDS
            ),
            start_backoff_seconds=_as_float(
                env.get("STREAM_START_BACKOFF_SECONDS"), DEFAULT_START_BACKOFF_SECONDS
            ),
            video_queue_capacity=int(
                _as_float(
                    env.get("STREAM_VIDEO_QUEUE_CAPACITY"), DEFAULT_VIDEO_QUEUE_CAPACITY
                )
            ),
            audio_queue_capacity=int(
                _as_float(
                    env.get("STREAM_AUDIO_QUEUE_CAPACITY"), DEFAULT_AUDIO_QUEUE_CAPACITY
                )
            ),
            reconnect=ReconnectPolicy(
                base_delay_seconds=_as_float(env.get("STREAM_RECONNECT_BASE_SECONDS"), 1.0),
                max_delay_seconds=_as_float(env.get("STREAM_RECONNECT_MAX_SECONDS"), 20.0),
                max_attempts=int(_as_float(env.get("STREAM_RECONNECT_MAX_ATTEMPTS"), 4)),
            ),
        )


def read_stream_key(path: Path | str) -> str:
    """Return the configured stream key, or an empty string when absent.

    Treat the result as a secret: it is never stored on a public attribute and
    never included in diagnostics or log output.
    """
    return _read_text(Path(path))


def build_provider(
    config: StreamingConfig | None = None,
    spawner: Any = None,
) -> Optional[StreamingProvider]:
    """Create the configured provider, or ``None`` when transport is off.

    ``spawner`` exists for tests: it replaces the process launcher so the real
    provider can be exercised without ffmpeg or a network.
    """
    resolved = config or StreamingConfig()
    if resolved.provider in {PROVIDER_RTMP, PROVIDER_SRT}:
        from .ffmpeg_provider import FFmpegStreamingProvider

        return FFmpegStreamingProvider(resolved, spawner=spawner)
    if resolved.provider == PROVIDER_MOCK:
        from .mock import MockStreamingProvider

        return MockStreamingProvider(resolved)
    return None


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _as_float(value: Optional[str], default: float) -> float:
    try:
        return float(value) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _as_positive_int(value: Optional[str], default: int) -> int:
    try:
        parsed = int(float(value)) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default
