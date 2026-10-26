"""Cline chat completion transport.

Cline exposes an OpenAI-compatible ``/api/v1/chat/completions`` endpoint, so
this provider deliberately reuses the already-audited transport in
``openrouter.py``: same request shape, same key-file loading, same redaction
of HTTP error bodies and the same strict content extraction. Only the
endpoint and model differ.

The API key is read from ``backend/api_key_cline.txt`` and never returned
through ``repr``, diagnostics or logs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .openrouter import OpenRouterConfig, OpenRouterProvider, _extract_content


DEFAULT_API_KEY_PATH = Path(__file__).resolve().parent.parent / "api_key_cline.txt"
DEFAULT_BASE_URL = "https://api.cline.bot/api/v1"
DEFAULT_MODEL = "nvidia/nemotron-3.5-lightning:free"
#: Deliberately generous transport ceiling for the first measurement pass.
#: The value only bounds waiting; ``LLM_TIMEOUT_SECONDS`` still wins.
DEFAULT_TIMEOUT_SECONDS = 120.0
#: PHASE 21 / PART 2: this gateway reports
#: ``usage.completion_tokens_details.reasoning_tokens`` for this model - it
#: spends the completion budget on hidden reasoning before the JSON plan and
#: stops only at that ceiling (measured: 62 of 64 tokens on a two-word
#: answer, ``finish_reason=length``). The gateway exposes the standard
#: OpenRouter-style reasoning switch, so the planner pays for the answer
#: instead of the reasoning. The prompt, parser and validation are untouched.
DEFAULT_REASONING: Mapping[str, Any] = {"enabled": False}


@dataclass(frozen=True)
class ClineConfig(OpenRouterConfig):
    """Cline defaults layered over the shared OpenAI-compatible config."""

    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    extra_payload: Mapping[str, Any] | None = field(
        default_factory=lambda: {"reasoning": dict(DEFAULT_REASONING)}
    )


class ClineProvider(OpenRouterProvider):
    """Real Cline provider built on the shared chat-completions client."""

    name = "cline"

    def _extract(self, raw_response: str) -> str:
        """Unwrap Cline's ``{"success":..,"data":{..}}`` gateway envelope.

        Everything below ``data`` is the ordinary OpenAI chat-completion
        payload, so the shared strict extractor still owns content validation.
        A body without the envelope is passed through untouched.
        """
        return _extract_content(_unwrap_envelope(raw_response))

    def __init__(
        self,
        config: OpenRouterConfig | None = None,
        api_key: str | None = None,
        api_key_path: Path | str = DEFAULT_API_KEY_PATH,
        urlopen: Any = None,
    ) -> None:
        resolved = config if config is not None else ClineConfig()
        if urlopen is None:
            super().__init__(resolved, api_key=api_key, api_key_path=api_key_path)
        else:
            super().__init__(
                resolved, api_key=api_key, api_key_path=api_key_path, urlopen=urlopen
            )


def _unwrap_envelope(raw_response: str) -> str:
    try:
        payload = json.loads(raw_response)
    except (json.JSONDecodeError, TypeError):
        return raw_response
    if not isinstance(payload, dict):
        return raw_response
    data = payload.get("data")
    if not isinstance(data, dict):
        return raw_response
    if "choices" not in data:
        return raw_response
    return json.dumps(data)


__all__ = [
    "DEFAULT_API_KEY_PATH",
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "DEFAULT_REASONING",
    "DEFAULT_TIMEOUT_SECONDS",
    "ClineConfig",
    "ClineProvider",
]
