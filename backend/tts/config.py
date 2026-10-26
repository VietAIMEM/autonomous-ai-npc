"""TTS selection and provider construction.

The API key is only ever read from disk here and inside the OpenRouter
provider. It is never returned by any dataclass, ``repr``, or diagnostics
payload in this package.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

from .base import TTSProvider


DEFAULT_MODEL = "deepgram/flux-tts:free"
DEFAULT_VOICE = "flux-alexis-en"
DEFAULT_FORMAT = "mp3"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_API_KEY_PATH = Path(__file__).resolve().parents[1] / "api_key_voice.txt"
DEFAULT_TIMEOUT_SECONDS = 25.0
#: Kokoro runs on the local CPU: the 82M checkpoint answers in seconds, but a
#: long utterance can outlast an HTTP provider's budget. The model load is a
#: one-time cost and gets its own, much longer allowance.
DEFAULT_KOKORO_PYTHON = Path(r"D:\kokoro_tts\.venv\Scripts\python.exe")
DEFAULT_KOKORO_VOICE = "af_heart"
DEFAULT_KOKORO_FORMAT = "wav"
DEFAULT_KOKORO_MODEL = "hexgrad/Kokoro-82M"
DEFAULT_KOKORO_TIMEOUT_SECONDS = 120.0
DEFAULT_KOKORO_WARMUP_TIMEOUT_SECONDS = 300.0

PROVIDER_MOCK = "mock"
PROVIDER_OPENROUTER = "openrouter"
PROVIDER_KOKORO = "kokoro"
PROVIDER_DISABLED = "disabled"

VALID_FORMATS = ("wav", "mp3", "pcm")


@dataclass(frozen=True)
class TTSConfig:
    provider: str = PROVIDER_MOCK
    model: str = DEFAULT_MODEL
    voice: str = DEFAULT_VOICE
    output_format: str = DEFAULT_FORMAT
    language: Optional[str] = None
    speed: float = 1.0
    base_url: str = DEFAULT_BASE_URL
    api_key_path: Path | str = DEFAULT_API_KEY_PATH
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_text_length: int = 500
    extra_headers: Mapping[str, str] = field(default_factory=dict)
    kokoro_python: Path | str = DEFAULT_KOKORO_PYTHON
    kokoro_warmup_timeout_seconds: float = DEFAULT_KOKORO_WARMUP_TIMEOUT_SECONDS

    @property
    def enabled(self) -> bool:
        return self.provider != PROVIDER_DISABLED

    def sanitized(self) -> "TTSConfig":
        """A copy that is safe to log: no key material, no headers."""
        return TTSConfig(
            provider=self.provider,
            model=self.model,
            voice=self.voice,
            output_format=self.output_format,
            language=self.language,
            speed=self.speed,
            base_url=self.base_url,
            api_key_path=Path(self.api_key_path).name,
            timeout_seconds=self.timeout_seconds,
            max_text_length=self.max_text_length,
            extra_headers={},
            kokoro_python=self.kokoro_python,
            kokoro_warmup_timeout_seconds=self.kokoro_warmup_timeout_seconds,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "voice": self.voice,
            "format": self.output_format,
            "language": self.language,
            "speed": self.speed,
            "base_url": self.base_url,
            "timeout_seconds": self.timeout_seconds,
            "enabled": self.enabled,
            "kokoro_warmup_timeout_seconds": self.kokoro_warmup_timeout_seconds,
        }

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> "TTSConfig":
        env = os.environ if environ is None else environ
        api_key_path = Path(env.get("TTS_API_KEY_PATH") or DEFAULT_API_KEY_PATH)
        requested = (env.get("TTS_PROVIDER") or "").strip().lower()
        if not requested:
            requested = (
                PROVIDER_OPENROUTER if _read_api_key(api_key_path) else PROVIDER_MOCK
            )
        if requested not in {
            PROVIDER_MOCK,
            PROVIDER_OPENROUTER,
            PROVIDER_KOKORO,
            PROVIDER_DISABLED,
        }:
            requested = PROVIDER_MOCK
        if requested == PROVIDER_KOKORO:
            default_voice = DEFAULT_KOKORO_VOICE
            default_format = DEFAULT_KOKORO_FORMAT
            default_model = DEFAULT_KOKORO_MODEL
            default_timeout = DEFAULT_KOKORO_TIMEOUT_SECONDS
        else:
            default_voice = DEFAULT_VOICE
            default_format = DEFAULT_FORMAT
            default_model = DEFAULT_MODEL
            default_timeout = DEFAULT_TIMEOUT_SECONDS
        output_format = (env.get("TTS_FORMAT") or default_format).strip().lower()
        if output_format not in VALID_FORMATS:
            output_format = default_format
        return cls(
            provider=requested,
            model=env.get("TTS_MODEL") or default_model,
            voice=env.get("TTS_VOICE") or default_voice,
            output_format=output_format,
            language=(env.get("TTS_LANGUAGE") or None),
            speed=_as_float(env.get("TTS_SPEED"), 1.0),
            base_url=env.get("TTS_BASE_URL") or DEFAULT_BASE_URL,
            api_key_path=api_key_path,
            timeout_seconds=_as_float(env.get("TTS_TIMEOUT_SECONDS"), default_timeout),
            max_text_length=int(_as_float(env.get("TTS_MAX_TEXT_LENGTH"), 500)),
            kokoro_python=Path(
                env.get("TTS_KOKORO_PYTHON") or DEFAULT_KOKORO_PYTHON
            ),
            kokoro_warmup_timeout_seconds=_as_float(
                env.get("TTS_KOKORO_WARMUP_TIMEOUT_SECONDS"),
                DEFAULT_KOKORO_WARMUP_TIMEOUT_SECONDS,
            ),
        )


def build_provider(config: TTSConfig | None = None) -> TTSProvider:
    """Create the configured provider. Unknown names degrade to the mock."""
    resolved = config or TTSConfig()
    if resolved.provider == PROVIDER_OPENROUTER:
        from .openrouter import OpenRouterTTSProvider

        return OpenRouterTTSProvider(resolved)
    if resolved.provider == PROVIDER_KOKORO:
        from .kokoro import KokoroTTSProvider

        return KokoroTTSProvider(resolved)
    from .mock import MockTTSProvider

    return MockTTSProvider(resolved)


def _read_api_key(path: Path | str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _as_float(value: Optional[str], default: float) -> float:
    try:
        return float(value) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default
