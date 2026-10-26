"""Deterministic offline provider used by automated tests and mock mode.

The bytes it emits are a real, decodable RIFF/WAVE container, so the whole
speech pipeline (validation, caching, transport to Unity) is exercised without
ever reaching the network. The mock cannot encode MP3, so it always returns
WAV and the pipeline adapts to whatever container it is handed.
"""

from __future__ import annotations

import hashlib
from array import array
from io import BytesIO
from wave import open as open_wave

from .base import AudioProbe, TTSProvider, TTSProviderError, TTSAudio, TTSRequest
from .config import TTSConfig


SAMPLE_RATE = 24_000
CHANNELS = 1
SAMPLE_WIDTH = 2
AMPLITUDE = 6_000
PERIOD = 48
SECONDS_PER_CHARACTER = 0.035
MIN_DURATION = 0.25
MAX_DURATION = 12.0


class MockTTSProvider(TTSProvider):
    name = "mock"
    streaming_supported = False

    def __init__(self, config: TTSConfig | None = None, fail_with: str | None = None) -> None:
        self.config = config or TTSConfig(provider="mock")
        self.fail_with = fail_with
        self.calls: list[TTSRequest] = []

    def synthesize(self, request: TTSRequest) -> TTSAudio:
        self.calls.append(request)
        if self.fail_with:
            raise TTSProviderError(self.fail_with)
        if not request.text.strip():
            raise TTSProviderError("empty_text")

        payload = _wav_bytes(request.text)
        probe = _probe(payload)
        return TTSAudio(
            payload=payload,
            probe=probe,
            provider=self.name,
            generation_id=_generation_id(request.text),
            streamed=False,
        )


def _wav_bytes(text: str) -> bytes:
    duration = min(MAX_DURATION, max(MIN_DURATION, len(text.strip()) * SECONDS_PER_CHARACTER))
    total_samples = int(SAMPLE_RATE * duration)
    # Integer-only triangle wave: identical bytes for identical input, with no
    # dependence on floating-point transcendentals.
    samples = array("h")
    span = PERIOD - 1
    for index in range(total_samples):
        step = index % PERIOD
        rise = step if step < PERIOD // 2 else PERIOD - 1 - step
        samples.append((rise * 2 * AMPLITUDE) // span - AMPLITUDE)

    buffer = BytesIO()
    with open_wave(buffer, "wb") as handle:
        handle.setnchannels(CHANNELS)
        handle.setsampwidth(SAMPLE_WIDTH)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(samples.tobytes())
    return buffer.getvalue()


def _probe(payload: bytes) -> AudioProbe:
    with open_wave(BytesIO(payload), "rb") as handle:
        frames = handle.getnframes()
        rate = handle.getframerate()
        return AudioProbe(
            format="wav",
            mime_type="audio/wav",
            byte_length=len(payload),
            sample_rate=rate,
            channels=handle.getnchannels(),
            bits_per_sample=handle.getsampwidth() * 8,
            duration_seconds=frames / rate if rate else None,
            exact_duration=True,
        )


def _generation_id(text: str) -> str:
    return "mock-" + hashlib.sha256(text.strip().casefold().encode("utf-8")).hexdigest()[:12]
