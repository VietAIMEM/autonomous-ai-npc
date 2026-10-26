"""NVIDIA NIM chat completion transport.

NVIDIA's ``/v1/chat/completions`` endpoint is OpenAI-compatible, so this
provider deliberately reuses the already-audited transport in
``openrouter.py``: same request shape, same key-file loading, same redaction
of HTTP error bodies and the same strict content extraction. Only the
endpoint, model and sampling parameters differ.

The API key is read from ``backend/api_key_nvidia.txt`` and never returned
through ``repr``, diagnostics or logs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .openrouter import OpenRouterConfig, OpenRouterProvider


DEFAULT_API_KEY_PATH = Path(__file__).resolve().parent.parent / "api_key_nvidia.txt"
DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "deepseek-ai/deepseek-v4.1-flash"
DEFAULT_TIMEOUT_SECONDS = 120.0

#: Sampling values documented by NVIDIA for this model.
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.95

#: PHASE 21 / PART 2: this reasoning model bills (and waits on) hidden
#: reasoning tokens before it emits the answer. Measured on the real
#: production planner prompt: 57.4s wall time with 1350 of 1421 completion
#: tokens spent on hidden reasoning. NIM accepts this template switch and the
#: assistant content still satisfies the same strict JSON contract, so the
#: planner prompt, parser and validation are untouched.
DEFAULT_CHAT_TEMPLATE_KWARGS: Mapping[str, Any] = {"thinking": False}


@dataclass(frozen=True)
class NvidiaConfig(OpenRouterConfig):
    """NVIDIA NIM defaults layered over the shared OpenAI-compatible config."""

    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    temperature: float | None = DEFAULT_TEMPERATURE
    top_p: float | None = DEFAULT_TOP_P
    chat_template_kwargs: Mapping[str, Any] | None = field(
        default_factory=lambda: dict(DEFAULT_CHAT_TEMPLATE_KWARGS)
    )
    #: PHASE 24: the live NIM endpoint throttles a burst with a bare HTTP 429
    #: that carries no ``Retry-After`` and no rate-limit headers, and the same
    #: burst clears in under a second (measured against the real endpoint).
    #: Two short retries absorb the throttle instead of having the runtime
    #: speak the provider-failure line to a real viewer.
    retry_attempts: int = 2
    retry_backoff_seconds: float = 0.5
    retry_max_backoff_seconds: float = 2.0


class NvidiaProvider(OpenRouterProvider):
    """Real NVIDIA NIM provider built on the shared chat-completions client."""

    name = "nvidia"

    def __init__(
        self,
        config: OpenRouterConfig | None = None,
        api_key: str | None = None,
        api_key_path: Path | str = DEFAULT_API_KEY_PATH,
        urlopen: Any = None,
    ) -> None:
        resolved = config if config is not None else NvidiaConfig()
        if urlopen is None:
            super().__init__(resolved, api_key=api_key, api_key_path=api_key_path)
        else:
            super().__init__(
                resolved, api_key=api_key, api_key_path=api_key_path, urlopen=urlopen
            )


__all__ = [
    "DEFAULT_API_KEY_PATH",
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "DEFAULT_TIMEOUT_SECONDS",
    "NvidiaConfig",
    "NvidiaProvider",
]
