"""OpenRouter audio-speech transport.

Only this module talks to OpenRouter, and only the backend runs it. Unity and
the LLM never see the endpoint, the key, or the request body.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .base import (
    TTSProvider,
    TTSProviderError,
    TTSAudio,
    TTSRequest,
    validate_audio_payload,
    wav_container_from_pcm,
)
from .config import TTSConfig


SPEECH_PATH = "/audio/speech"
MAX_ERROR_DETAIL = 300


class OpenRouterTTSProvider(TTSProvider):
    name = "openrouter"
    streaming_supported = False

    def __init__(
        self,
        config: TTSConfig | None = None,
        api_key: str | None = None,
        urlopen: Any = urllib.request.urlopen,
    ) -> None:
        self.config = config or TTSConfig(provider="openrouter")
        self._api_key = api_key if api_key is not None else _read_api_key(self.config.api_key_path)
        self._urlopen = urlopen

    @property
    def has_api_key(self) -> bool:
        return bool(self._api_key)

    def __repr__(self) -> str:  # never surface credentials
        return f"OpenRouterTTSProvider(model={self.config.model!r}, voice={self.config.voice!r})"

    def describe(self) -> dict[str, object]:
        return {
            "provider": self.name,
            "model": self.config.model,
            "voice": self.config.voice,
            "format": self.config.output_format,
            "streaming_supported": self.streaming_supported,
            "api_key_configured": self.has_api_key,
        }

    def build_payload(self, request: TTSRequest) -> dict[str, object]:
        """The provider body. Split out so tests can assert it without network."""
        payload: dict[str, object] = {
            "model": request.model or self.config.model,
            "input": request.text,
            "voice": request.voice or self.config.voice,
            # The endpoint's default is raw PCM; asking explicitly keeps the
            # response format stable.
            "response_format": request.output_format or self.config.output_format,
        }
        if request.language:
            payload["language"] = request.language
        if request.speed and abs(request.speed - 1.0) > 1e-6:
            payload["speed"] = request.speed
        return payload

    def build_headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        headers.update(self.config.extra_headers)
        return headers

    def synthesize(self, request: TTSRequest) -> TTSAudio:
        if not self.has_api_key:
            raise TTSProviderError("api_key_unavailable")
        if not request.text.strip():
            raise TTSProviderError("empty_text")

        body = json.dumps(self.build_payload(request), ensure_ascii=False).encode("utf-8")
        http_request = urllib.request.Request(
            f"{self.config.base_url.rstrip('/')}{SPEECH_PATH}",
            data=body,
            headers=self.build_headers(),
            method="POST",
        )
        timeout = request.timeout_seconds or self.config.timeout_seconds

        try:
            with self._urlopen(http_request, timeout=timeout) as response:
                payload = response.read()
                content_type = response.headers.get("Content-Type")
                generation_id = response.headers.get("X-Generation-Id")
                status = getattr(response, "status", 200)
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            detail = _safe_text(raw, self._api_key)[:MAX_ERROR_DETAIL]
            raise TTSProviderError(
                f"http_{exc.code}",
                detail,
                retry_after_seconds=_retry_after_seconds(exc, raw),
            ) from None
        except (TimeoutError, urllib.error.URLError, OSError) as exc:
            reason = "provider_timeout" if isinstance(exc, TimeoutError) else "transport_error"
            raise TTSProviderError(reason, type(exc).__name__) from None

        if status != 200:
            raise TTSProviderError(f"http_{status}")

        probe = validate_audio_payload(payload, content_type)
        if probe.format == "pcm":
            # Minimal conversion, no audio framework: raw PCM gets a WAV header
            # so the Unity decoder has something self-describing to work with.
            rate = probe.sample_rate or 24_000
            payload = wav_container_from_pcm(payload, rate, probe.channels or 1)
            probe = validate_audio_payload(payload, "audio/wav")

        return TTSAudio(
            payload=payload,
            probe=probe,
            provider=self.name,
            generation_id=generation_id or None,
            streamed=False,
        )


def _read_api_key(path: Path | str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _safe_text(raw: bytes, secret: str) -> str:
    text = ""
    if raw:
        try:
            text = raw.decode("utf-8", errors="replace")
        except (LookupError, ValueError):
            text = ""
    if secret and secret in text:
        text = text.replace(secret, "[redacted]")
    return text.replace("\n", " ").strip()


def _retry_after_seconds(exc: urllib.error.HTTPError, body: bytes) -> float | None:
    """How long the provider says to wait, when it says so at all.

    ``Retry-After`` wins; OpenRouter's rate-limit body carries the window reset
    as an epoch-millisecond stamp, which is the difference between a handful of
    attempts a day and one attempt every Brain tick.
    """
    try:
        header = exc.headers.get("Retry-After") if exc.headers else None
    except (LookupError, AttributeError):
        header = None
    if header:
        try:
            return max(0.0, float(str(header).strip()))
        except ValueError:
            pass

    try:
        payload = json.loads(body.decode("utf-8", errors="replace"))
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    metadata = error.get("metadata") if isinstance(error, dict) else None
    headers = metadata.get("headers") if isinstance(metadata, dict) else None
    reset = headers.get("X-RateLimit-Reset") if isinstance(headers, dict) else None
    try:
        reset_seconds = float(str(reset)) / 1000.0
    except (TypeError, ValueError):
        return None
    return max(0.0, reset_seconds - time.time())
