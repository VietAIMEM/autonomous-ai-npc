"""Local Ollama chat transport.

Ollama runs the model on this machine and speaks HTTP on loopback. This module
is deliberately a thin subclass of the already-audited
:class:`~backend.llm.openrouter.OpenRouterProvider`: it reuses that transport's
bounded retry, backoff, ``Retry-After`` clamp, redaction of HTTP error bodies,
latency/timeout handling and the ``TransportFailureInfo`` annotation that feeds
``brain.llm_request_failed``. Only the pieces that must differ for a local
server are overridden:

* the endpoint (loopback only), and
* the payload shape and response shape of Ollama's native ``/api/chat``.

Native ``/api/chat`` is used instead of the OpenAI-compatible shim because it
is the only Ollama surface that accepts ``options.num_ctx`` directly. That is
the supported mechanism for the requested 8192-token context target; the OpenAI
shim does not forward ``num_ctx``.

Security posture: the endpoint is validated to be loopback, there is no API key
or ``Authorization`` header, and a local failure is raised as an ordinary
``LLMProviderError`` (category ``connection``/``provider_server_error``) so the
runtime never falls back to a remote provider.
"""

from __future__ import annotations

import ipaddress
import json
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Mapping

from .base import LLMProviderError, LLMRequest
from .openrouter import OpenRouterConfig, OpenRouterProvider

DEFAULT_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen3.5:4b"
#: Local generation can be slower than a hosted endpoint while the model loads
#: from disk or spills past VRAM. This is a transport ceiling only; the
#: environment variable ``LLM_TIMEOUT_SECONDS`` still overrides it.
DEFAULT_TIMEOUT_SECONDS = 120.0
#: Requested starting context window. Configurable through ``options.num_ctx``.
DEFAULT_NUM_CTX = 8192
#: qwen3.5 renders a hidden reasoning block. Left enabled, the model can spend
#: the entire generation budget inside ``message.thinking`` and return an empty
#: ``message.content`` (measured on 9b: 0 content characters in 3/3 planner
#: cases at a 512-token cap). Ollama's native API supports an explicit
#: ``think`` switch; ``think=False`` asks the model to answer directly so the
#: planner receives JSON in ``message.content``. If a model ignores it, the
#: transport still fails closed rather than parsing reasoning text as a plan.
DEFAULT_THINK = False
#: Bounded generation ceiling. Large enough for a validated multi-step plan
#: even when a model spends some budget on reasoning, small enough that a
#: confused model cannot generate an unbounded local reply. This is a
#: generation setting, reported separately from ``num_ctx``.
DEFAULT_NUM_PREDICT = 1024
#: Sampling defaults; kept explicit so the payload is self-describing.
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_P = 0.9
#: Native Ollama structured-output switch. ``"json"`` asks the server to
#: constrain generation to a single valid JSON object. This is applied to
#: planner (Brain) requests only -- the shared chat path must keep returning
#: free prose, so scoping is deliberate rather than global.
DEFAULT_JSON_MODE = "json"


def _is_loopback(base_url: str) -> bool:
    """True when ``base_url`` resolves to a loopback host.

    Accepts ``localhost`` and any loopback IP literal (127.0.0.0/8, ::1). A
    hostname that merely *could* resolve to loopback is rejected, because the
    check must not depend on the current DNS answer.
    """
    try:
        parsed = urllib.parse.urlsplit(base_url)
    except ValueError:
        return False
    if parsed.scheme not in ("http", "https"):
        return False
    host = (parsed.hostname or "").strip().lower()
    if not host:
        return False
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@dataclass(frozen=True)
class OllamaConfig(OpenRouterConfig):
    """Local Ollama defaults layered over the shared OpenAI-compatible config."""

    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    temperature: float | None = DEFAULT_TEMPERATURE
    top_p: float | None = DEFAULT_TOP_P
    #: No template switches or extra top-level keys are needed for Ollama.
    chat_template_kwargs: Mapping[str, Any] | None = None
    #: Ollama uses a bounded local retry only to absorb a brief server restart;
    #: it is the same knob the shared transport already exposes.
    retry_attempts: int = 1
    retry_backoff_seconds: float = 0.5
    retry_max_backoff_seconds: float = 2.0
    #: Context window requested from Ollama's native API (``options.num_ctx``).
    num_ctx: int = DEFAULT_NUM_CTX
    #: Optional number of model layers to place on the GPU (native
    #: ``options.num_gpu``). ``None`` leaves placement to Ollama's automatic
    #: default (which currently offloads all layers). A lower value moves the
    #: remaining layers to CPU/system RAM, trading VRAM for latency.
    num_gpu: int | None = None
    #: Generation ceiling requested from Ollama (``options.num_predict``).
    num_predict: int = DEFAULT_NUM_PREDICT
    #: Ask the model not to emit a hidden reasoning block (native ``think``).
    think: bool = DEFAULT_THINK
    #: Native ``format`` constraint for planner requests, or ``None`` to leave
    #: the response unconstrained. Ollama's ``/api/chat`` accepts the string
    #: ``"json"`` or a JSON Schema object; only the planner path uses it.
    json_mode: str | None = DEFAULT_JSON_MODE


class OllamaProvider(OpenRouterProvider):
    """Local Ollama chat provider built on the shared chat-completions client."""

    name = "ollama"

    def __init__(
        self,
        config: OpenRouterConfig | None = None,
        api_key: str | None = None,
        api_key_path: Any = None,
        urlopen: Any = None,
    ) -> None:
        # A local server needs no credential. ``api_key``/``api_key_path`` are
        # accepted only so the provider keeps the same constructor shape as the
        # rest of the family; they are ignored.
        resolved = config if config is not None else OllamaConfig()
        if not _is_loopback(getattr(resolved, "base_url", "")):
            raise LLMProviderError("ollama_non_loopback_endpoint")
        if urlopen is None:
            super().__init__(resolved, api_key="local")
        else:
            super().__init__(resolved, api_key="local", urlopen=urlopen)

    def generate(self, request: LLMRequest) -> str:
        """POST a native ``/api/chat`` request through the shared retry loop."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": request.prompt},
        ]
        assistant_message = request.context.get("assistant_message")
        follow_up = request.context.get("follow_up")
        if (
            isinstance(assistant_message, Mapping)
            and isinstance(follow_up, str)
            and follow_up
        ):
            content = assistant_message.get("content")
            if isinstance(content, str):
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": follow_up})

        options: dict[str, Any] = {"num_ctx": int(getattr(self.config, "num_ctx", DEFAULT_NUM_CTX))}
        # ``num_predict`` bounds generation; ``max_tokens`` is the shared
        # provider-wide token budget, so the tighter of the two wins.
        num_predict = int(getattr(self.config, "num_predict", DEFAULT_NUM_PREDICT))
        if self.config.max_tokens > 0:
            num_predict = min(num_predict, int(self.config.max_tokens))
        if num_predict > 0:
            options["num_predict"] = num_predict
        if self.config.temperature is not None:
            options["temperature"] = float(self.config.temperature)
        if self.config.top_p is not None:
            options["top_p"] = float(self.config.top_p)
        # Optional GPU-layer split. Absent (the default) preserves Ollama's
        # automatic placement exactly as before; only a configured value is sent.
        num_gpu = getattr(self.config, "num_gpu", None)
        if isinstance(num_gpu, int) and not isinstance(num_gpu, bool) and num_gpu >= 0:
            options["num_gpu"] = num_gpu

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "options": options,
        }
        if hasattr(self.config, "think"):
            payload["think"] = bool(getattr(self.config, "think"))
        # Structured output for the planner only. ``request.context["brain"]``
        # is set by the reasoning path; the shared chat path does not set it and
        # must keep returning free-form prose, so it is never constrained.
        json_mode = getattr(self.config, "json_mode", None)
        if json_mode and request.context.get("brain"):
            payload["format"] = json_mode
        if self.config.request_options:
            payload.update(self.config.request_options)
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        raw = self._post(f"{self.config.base_url.rstrip('/')}/api/chat", body, headers)
        return self._extract(raw)

    def _extract(self, raw_response: str) -> str:
        """Pull assistant content out of a native Ollama chat response.

        Shape: ``{"message": {"role": "assistant", "content": "..."},
        "done": true, ...}``. Errors arrive as ``{"error": "..."}``; the message
        is never surfaced verbatim -- only a coarse, fixed-reason
        ``LLMProviderError`` is raised so nothing provider-supplied reaches a
        caller message or a log line.
        """
        try:
            payload = json.loads(raw_response)
        except (json.JSONDecodeError, TypeError) as exc:
            raise LLMProviderError("invalid_provider_json") from exc
        if not isinstance(payload, dict):
            raise LLMProviderError("invalid_provider_payload")
        if "error" in payload:
            raise LLMProviderError("ollama_error")
        message = payload.get("message")
        if not isinstance(message, dict):
            raise LLMProviderError("missing_provider_message")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            # A reasoning-only reply (``thinking`` with empty ``content``) is a
            # distinct, actionable failure rather than an empty success.
            if isinstance(message.get("thinking"), str) and message["thinking"].strip():
                raise LLMProviderError("empty_content_with_reasoning_details")
            raise LLMProviderError("empty_provider_content")
        return content


__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_JSON_MODE",
    "DEFAULT_MODEL",
    "DEFAULT_NUM_CTX",
    "DEFAULT_NUM_PREDICT",
    "DEFAULT_THINK",
    "DEFAULT_TIMEOUT_SECONDS",
    "OllamaConfig",
    "OllamaProvider",
]
