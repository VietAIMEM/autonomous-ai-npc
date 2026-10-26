"""Minimal TTS abstraction shared by the speech service and every provider.

Audio validation lives here so that no caller has to trust an HTTP 200 on its
own: a response only counts as speech once its bytes are a real container.
"""

from __future__ import annotations

import wave
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Mapping, Optional


MIN_AUDIO_BYTES = 256


class TTSProviderError(RuntimeError):
    """Raised for provider transport, protocol, or audio-payload failures."""

    def __init__(
        self,
        reason: str,
        detail: str = "",
        retry_after_seconds: Optional[float] = None,
    ) -> None:
        super().__init__(f"{reason}:{detail}" if detail else reason)
        self.reason = reason
        self.detail = detail
        #: Provider-declared cool-down, when the failure carries one. A rate
        #: limit that is retried immediately is not a recovery, it is a storm.
        self.retry_after_seconds = retry_after_seconds


@dataclass(frozen=True)
class TTSRequest:
    text: str
    model: str
    voice: str
    output_format: str = "mp3"
    language: Optional[str] = None
    speed: float = 1.0
    request_id: Optional[str] = None
    timeout_seconds: Optional[float] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AudioProbe:
    """What the bytes actually are, independent of what the provider claimed."""

    format: str
    mime_type: str
    byte_length: int
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    bits_per_sample: Optional[int] = None
    duration_seconds: Optional[float] = None
    exact_duration: bool = False


@dataclass(frozen=True)
class TTSAudio:
    """A synthesised utterance ready for playback."""

    payload: bytes
    probe: AudioProbe
    provider: str
    generation_id: Optional[str] = None
    streamed: bool = False

    @property
    def byte_length(self) -> int:
        return len(self.payload)

    @property
    def format(self) -> str:
        return self.probe.format

    @property
    def duration_seconds(self) -> Optional[float]:
        return self.probe.duration_seconds

    def to_dict(self) -> dict[str, object]:
        """Diagnostics view; never contains the audio payload itself."""
        return {
            "provider": self.provider,
            "format": self.probe.format,
            "mime_type": self.probe.mime_type,
            "byte_length": self.probe.byte_length,
            "sample_rate": self.probe.sample_rate,
            "channels": self.probe.channels,
            "duration_seconds": (
                round(self.probe.duration_seconds, 3)
                if self.probe.duration_seconds is not None
                else None
            ),
            "duration_exact": self.probe.exact_duration,
            "generation_id": self.generation_id,
            "streamed": self.streamed,
        }


class TTSProvider(ABC):
    """A provider turns speech text into validated audio bytes."""

    name: str = "unknown"
    streaming_supported: bool = False

    @abstractmethod
    def synthesize(self, request: TTSRequest) -> TTSAudio:
        raise NotImplementedError

    def describe(self) -> dict[str, object]:
        return {
            "provider": self.name,
            "streaming_supported": self.streaming_supported,
        }


# Layer III frame headers, indexed by the version bits that follow the sync word.
_MPEG1_LAYER3_BITRATES = (0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448)
_MPEG2_LAYER3_BITRATES = (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160)
_SAMPLE_RATES = {
    3: (44100, 48000, 32000),  # MPEG Version 1
    2: (22050, 24000, 16000),  # MPEG Version 2
    0: (8000, 11025, 12000),   # MPEG Version 2.5
}
_LAYER_III = 1


def parse_content_type(value: Optional[str]) -> tuple[Optional[str], dict[str, str]]:
    if not value:
        return None, {}
    parts = [part.strip() for part in value.split(";")]
    base = parts[0].lower() if parts else None
    params: dict[str, str] = {}
    for part in parts[1:]:
        if "=" not in part:
            continue
        key, _, raw = part.partition("=")
        params[key.strip().lower()] = raw.strip().strip('"').lower()
    return base or None, params


def _syncsafe_size(payload: bytes) -> int:
    return (
        (payload[0] & 0x7F) << 21
        | (payload[1] & 0x7F) << 14
        | (payload[2] & 0x7F) << 7
        | (payload[3] & 0x7F)
    )


def _skip_id3(payload: bytes) -> int:
    offset = 0
    while payload[offset : offset + 3] == b"ID3" and len(payload) >= offset + 10:
        offset += 10 + _syncsafe_size(payload[offset + 6 : offset + 10])
    return offset


def _probe_mp3(payload: bytes) -> Optional[AudioProbe]:
    start = _skip_id3(payload)
    limit = min(len(payload) - 4, start + 4096)
    index = start
    while index < limit:
        if payload[index] == 0xFF and (payload[index + 1] & 0xE0) == 0xE0:
            header = int.from_bytes(payload[index : index + 4], "big")
            version_bits = (header >> 19) & 0x03
            layer_bits = (header >> 17) & 0x03
            bitrate_index = (header >> 12) & 0x0F
            rate_index = (header >> 10) & 0x03
            channel_bits = (header >> 6) & 0x03
            # Only Layer III is treated as MP3; version 1 uses its own bitrate row.
            rates = _SAMPLE_RATES.get(version_bits)
            table = None
            if layer_bits == _LAYER_III and rates is not None:
                table = _MPEG1_LAYER3_BITRATES if version_bits == 3 else _MPEG2_LAYER3_BITRATES
            if table is not None and 0 < bitrate_index < 15 and rate_index != 3:
                bitrate = table[bitrate_index]
                sample_rate = rates[rate_index]
                body = max(0, len(payload) - index)
                duration = (body * 8) / (bitrate * 1000) if bitrate else None
                return AudioProbe(
                    format="mp3",
                    mime_type="audio/mpeg",
                    byte_length=len(payload),
                    sample_rate=sample_rate,
                    channels=1 if channel_bits == 3 else 2,
                    duration_seconds=duration,
                    exact_duration=False,
                )
        index += 1
    return None


def _probe_wav(payload: bytes) -> Optional[AudioProbe]:
    if payload[:4] != b"RIFF" or payload[8:12] != b"WAVE":
        return None
    try:
        with wave.open(BytesIO(payload), "rb") as handle:
            frames = handle.getnframes()
            rate = handle.getframerate()
            return AudioProbe(
                format="wav",
                mime_type="audio/wav",
                byte_length=len(payload),
                sample_rate=rate or None,
                channels=handle.getnchannels() or None,
                bits_per_sample=handle.getsampwidth() * 8 or None,
                duration_seconds=(frames / rate) if frames and rate else None,
                exact_duration=bool(frames and rate),
            )
    except (EOFError, wave.Error, ValueError):
        # A RIFF header is enough for Unity's WAV decoder to try; report it as
        # a WAV container without inventing duration we cannot read.
        return AudioProbe(format="wav", mime_type="audio/wav", byte_length=len(payload))


def probe_audio(payload: bytes, content_type: str | None = None) -> Optional[AudioProbe]:
    """Identify the real container of ``payload``; ``None`` means not audio."""
    if not payload:
        return None

    wav = _probe_wav(payload)
    if wav is not None:
        return wav

    _, params = parse_content_type(content_type)
    declared, _, subtype = (content_type or "").partition("/")
    if payload[:3] == b"ID3" or (payload[0] == 0xFF and (payload[1] & 0xE0) == 0xE0):
        mp3 = _probe_mp3(payload)
        if mp3 is not None:
            return mp3
    if payload[:4] == b"OggS":
        return AudioProbe(format="ogg", mime_type="audio/ogg", byte_length=len(payload))
    if payload[:4] == b"fLaC":
        return AudioProbe(format="flac", mime_type="audio/flac", byte_length=len(payload))

    base, _ = parse_content_type(content_type)
    pcm_declared = base in {"audio/pcm", "audio/raw", "audio/wave"} or subtype in {"pcm", "raw"}
    if pcm_declared:
        rate = _as_int(params.get("rate") or params.get("samplerate"))
        channels = _as_int(params.get("channels") or params.get("num_channels")) or 1
        bits = _as_int(params.get("bits_per_sample") or params.get("bitrate")) or 16
        frame_bytes = max(1, channels * bits // 8)
        duration = len(payload) / (rate * frame_bytes) if rate else None
        return AudioProbe(
            format="pcm",
            mime_type="audio/pcm",
            byte_length=len(payload),
            sample_rate=rate,
            channels=channels,
            bits_per_sample=bits,
            duration_seconds=duration,
            exact_duration=bool(rate),
        )
    return None


def _as_int(value: Optional[str]) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def validate_audio_payload(
    payload: bytes,
    content_type: str | None = None,
    *,
    min_bytes: int = MIN_AUDIO_BYTES,
) -> AudioProbe:
    """Turn a provider response into a probe, or fail with a structured reason."""
    if payload is None or len(payload) == 0:
        raise TTSProviderError("empty_audio", "no bytes returned")
    if len(payload) < min_bytes:
        raise TTSProviderError("empty_audio", f"{len(payload)} bytes is below {min_bytes}")
    probe = probe_audio(payload, content_type)
    if probe is None:
        raise TTSProviderError("malformed_audio", "payload is not a recognised audio container")
    return probe


def wav_container_from_pcm(payload: bytes, sample_rate: int, channels: int = 1, bits: int = 16) -> bytes:
    """Wrap raw PCM in a WAV header so Unity can decode it without an audio library."""
    buffer = BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(max(1, channels))
        handle.setsampwidth(max(1, bits // 8))
        handle.setframerate(max(1, sample_rate))
        handle.writeframes(payload)
    return buffer.getvalue()
