"""OpenRouter-compatible chat completion transport."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..runtime.redaction import redact_text
from .base import LLMProvider, LLMProviderError, LLMRequest, TransportFailureInfo
from .failure_diagnostics import rate_limit_headers


DEFAULT_API_KEY_PATH = Path(__file__).resolve().parent / "api_key.txt"

#: Statuses a bounded retry is allowed to absorb. The gateway in front of
#: NVIDIA NIM throttles a burst with a bare 429 and no ``Retry-After`` header,
#: and the same burst clears in well under a second (measured PHASE 24).
_RETRYABLE_STATUS_CODES = frozenset({429, 502, 503, 504})


@dataclass(frozen=True)
class OpenRouterConfig:
    model: str = "openai/gpt-5-nano"
    base_url: str = "https://openrouter.ai/api/v1"
    timeout_seconds: float = 30.0
    max_tokens: int = 2048
    system_prompt: str = (
        "You are a planner for an embodied virtual character. "
        "Return strict JSON only and never execute anything."
    )
    extra_headers: Mapping[str, str] | None = None
    #: Optional sampling knobs. Left unset for OpenRouter so its payload stays
    #: exactly as it was; providers that need them (NVIDIA NIM) set them.
    temperature: float | None = None
    top_p: float | None = None
    #: Optional provider-specific chat-template switches. Also left unset for
    #: OpenRouter, so its payload is byte-for-byte what it was before.
    chat_template_kwargs: Mapping[str, Any] | None = None
    #: Optional raw payload additions for gateways that speak the same shape
    #: but accept extra top-level keys (see ``cline.py``). Left unset for
    #: OpenRouter so its payload is byte-for-byte what it was before.
    extra_payload: Mapping[str, Any] | None = None
    #: Optional extra top-level keys merged after ``extra_payload``. Used by
    #: local OpenAI-compatible servers that need a non-standard switch (for
    #: example Ollama's ``options`` block); left unset, the payload stays
    #: byte-for-byte what it was before.
    request_options: Mapping[str, Any] | None = None
    #: Bounded retry for transient provider throttling. Left at zero for
    #: OpenRouter so its transport behaves exactly as it did before; providers
    #: that observe real 429 bursts (see ``nvidia.py``) opt in.
    retry_attempts: int = 0
    retry_backoff_seconds: float = 0.5
    retry_max_backoff_seconds: float = 4.0


class OpenRouterProvider(LLMProvider):
    """Small, testable transport for OpenRouter chat completions."""

    def __init__(
        self,
        config: OpenRouterConfig | None = None,
        api_key: str | None = None,
        api_key_path: Path | str = DEFAULT_API_KEY_PATH,
        urlopen: Any = urllib.request.urlopen,
    ) -> None:
        self.config = config or OpenRouterConfig()
        self._api_key = api_key or _read_api_key(api_key_path)
        self._urlopen = urlopen

    @property
    def has_api_key(self) -> bool:
        return bool(self._api_key)

    def generate(self, request: LLMRequest) -> str:
        if not self.has_api_key:
            raise LLMProviderError("api_key_unavailable")

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": request.prompt},
        ]
        assistant_message = request.context.get("assistant_message")
        follow_up = request.context.get("follow_up")
        if isinstance(assistant_message, Mapping) and isinstance(follow_up, str) and follow_up:
            messages.append(_safe_assistant_message(assistant_message))
            messages.append({"role": "user", "content": follow_up})

        payload: dict[str, Any] = {"model": self.config.model, "messages": messages}
        if self.config.max_tokens > 0:
            # Without an explicit ceiling the provider reserves the model's
            # full output budget and rejects the call when the account cannot
            # cover it, so real replies would never leave the transport.
            payload["max_tokens"] = self.config.max_tokens
        if self.config.temperature is not None:
            payload["temperature"] = self.config.temperature
        if self.config.top_p is not None:
            payload["top_p"] = self.config.top_p
        if self.config.chat_template_kwargs:
            payload["chat_template_kwargs"] = dict(self.config.chat_template_kwargs)
        if self.config.extra_payload:
            payload.update(self.config.extra_payload)
        if self.config.request_options:
            payload.update(self.config.request_options)
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        if self.config.extra_headers:
            headers.update(self.config.extra_headers)
        raw = self._post(
            f"{self.config.base_url.rstrip('/')}/chat/completions", body, headers
        )
        return self._extract(raw)

    def _post(
        self,
        url: str,
        body: bytes,
        headers: Mapping[str, str],
    ) -> str:
        """Send ``body`` to ``url`` with the shared bounded retry loop.

        Split out of :meth:`generate` so an OpenAI-compatible subclass can keep
        the audited retry, backoff, ``Retry-After`` clamp, transport-error
        annotation and redaction while supplying its own endpoint, payload
        shape and response extractor. The request bytes are reused across
        attempts, exactly as before.
        """
        http_request = urllib.request.Request(
            url,
            data=body,
            headers=headers,
            method="POST",
        )

        attempts = max(0, int(self.config.retry_attempts))
        configured_max_attempts = attempts + 1
        for attempt in range(configured_max_attempts):
            try:
                with self._urlopen(
                    http_request, timeout=self.config.timeout_seconds
                ) as response:
                    raw = response.read().decode("utf-8")
                break
            except urllib.error.HTTPError as exc:
                # The body can echo request headers back; it is redacted before
                # it can reach an exception message, a log line or a
                # diagnostics file.
                detail = redact_text(
                    exc.read().decode("utf-8", errors="replace"), limit=300
                )
                retryable_status = exc.code in _RETRYABLE_STATUS_CODES
                if attempt < attempts and retryable_status:
                    time.sleep(self._retry_delay(exc, attempt))
                    continue
                # PHASE 39: observability only. The validated allowlisted
                # rate-limit headers and the number of attempts this call
                # actually made travel with the exception, so the failure
                # diagnostic reports real history instead of inferring it from
                # elapsed time. Nothing here changes whether a request is
                # retried, the retryable status set, the backoff, the
                # ``Retry-After`` clamp or the timeout.
                rl_headers, rl_unparsed = rate_limit_headers(
                    getattr(exc, "headers", None)
                )
                raise self._annotate_transport_error(
                    LLMProviderError(f"http_{exc.code}:{detail}"),
                    attempt_count=attempt + 1,
                    configured_max_attempts=configured_max_attempts,
                    # True only when at least one retry was configured and the
                    # final attempt still returned a retryable status.
                    retries_exhausted=(
                        attempts > 0 and retryable_status and attempt >= attempts
                    ),
                    headers=rl_headers,
                    unparsed=rl_unparsed,
                ) from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                # A transport failure is never retried (the branch above is the
                # only retry path), so exactly one attempt happened and no
                # retry budget was spent.
                raise self._annotate_transport_error(
                    LLMProviderError(f"transport_error:{type(exc).__name__}"),
                    attempt_count=attempt + 1,
                    configured_max_attempts=configured_max_attempts,
                    retries_exhausted=False,
                    headers={},
                    unparsed=(),
                ) from exc

        return raw

    @staticmethod
    def _annotate_transport_error(
        error: LLMProviderError,
        *,
        attempt_count: int,
        configured_max_attempts: int,
        retries_exhausted: bool,
        headers: Mapping[str, str],
        unparsed: tuple[str, ...],
    ) -> LLMProviderError:
        """Attach bounded transport facts to the error, then return it.

        The attribute is additive: the exception type, message and ``__cause__``
        are unchanged, so existing callers and the public provider contract are
        unaffected. The diagnostics module re-validates every value it reads.
        """
        error.transport_info = TransportFailureInfo(
            attempt_count=attempt_count,
            configured_max_attempts=configured_max_attempts,
            retries_exhausted=retries_exhausted,
            rate_limit_headers=dict(headers),
            rate_limit_headers_unparsed=tuple(unparsed),
        )
        return error

    def _retry_delay(self, exc: urllib.error.HTTPError, attempt: int) -> float:
        """Seconds to wait before retrying a throttled request.

        ``Retry-After`` wins when the provider sends one; otherwise the delay
        doubles per attempt. Both are clamped to ``retry_max_backoff_seconds``
        so a hostile or absent header can never stall the runtime.
        """
        ceiling = max(0.0, float(self.config.retry_max_backoff_seconds))
        base = max(0.0, float(self.config.retry_backoff_seconds))
        delay = base * (2**attempt)
        header = None
        headers = getattr(exc, "headers", None)
        if headers is not None:
            try:
                header = headers.get("Retry-After")
            except AttributeError:
                header = None
        if header:
            try:
                delay = float(str(header).strip())
            except ValueError:
                delay = base * (2**attempt)
        return max(0.0, min(delay, ceiling))

    def _extract(self, raw_response: str) -> str:
        """Turn a raw provider body into assistant content.

        Overridden by gateways that wrap the OpenAI payload (see ``cline.py``);
        OpenRouter's own response shape is used unchanged.
        """
        return _extract_content(raw_response)


def _read_api_key(path: Path | str) -> str:
    try:
        value = Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return ""
    return value


def _safe_assistant_message(value: Mapping[str, Any]) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant"}
    content = value.get("content")
    if isinstance(content, str):
        message["content"] = content
    if isinstance(value.get("reasoning_details"), list):
        message["reasoning_details"] = value["reasoning_details"]
    return message


def _extract_content(raw_response: str) -> str:
    try:
        payload = json.loads(raw_response)
    except (json.JSONDecodeError, TypeError) as exc:
        raise LLMProviderError("invalid_provider_json") from exc
    if not isinstance(payload, dict):
        raise LLMProviderError("invalid_provider_payload")

    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise LLMProviderError("missing_provider_choice")

    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise LLMProviderError("missing_provider_message")

    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        if "reasoning_details" in message:
            raise LLMProviderError("empty_content_with_reasoning_details")
        raise LLMProviderError("empty_provider_content")
    return content
