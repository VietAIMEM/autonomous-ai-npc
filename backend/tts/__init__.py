"""Provider-agnostic text-to-speech boundary for the backend.

The Brain and SpeechService talk to :class:`TTSProvider`; only the concrete
providers in this package know how to reach an external service.
"""

from .base import (
    AudioProbe,
    TTSProvider,
    TTSProviderError,
    TTSAudio,
    TTSRequest,
    probe_audio,
    validate_audio_payload,
    wav_container_from_pcm,
)
from .config import TTSConfig, build_provider
from .kokoro import KokoroTTSProvider
from .mock import MockTTSProvider
from .openrouter import OpenRouterTTSProvider

__all__ = [
    "AudioProbe",
    "KokoroTTSProvider",
    "MockTTSProvider",
    "OpenRouterTTSProvider",
    "TTSConfig",
    "TTSProvider",
    "TTSProviderError",
    "TTSAudio",
    "TTSRequest",
    "build_provider",
    "probe_audio",
    "validate_audio_payload",
    "wav_container_from_pcm",
]
