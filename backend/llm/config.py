"""LLM provider selection for the reasoning and chat paths.

Read like ``backend/tts/config.py``: the API key is only ever read from disk
inside the real provider. Nothing here returns key material through a
dataclass, ``repr`` or diagnostics payload.

The selected provider is made explicit instead of implied: an unset
``LLM_PROVIDER`` prefers the real transport when a key file exists, and an
unknown provider name is rejected rather than silently degraded to the mock.
``backend.runtime.config.RuntimeConfig.enforce_provider`` is what keeps that
mock out of production.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from .base import LLMProvider


PROVIDER_MOCK = "mock"
PROVIDER_OPENROUTER = "openrouter"
PROVIDER_NVIDIA = "nvidia"
PROVIDER_CLINE = "cline"
PROVIDER_OLLAMA = "ollama"
VALID_PROVIDERS = (
    PROVIDER_MOCK,
    PROVIDER_OPENROUTER,
    PROVIDER_NVIDIA,
    PROVIDER_CLINE,
    PROVIDER_OLLAMA,
)

DEFAULT_MODEL = "openai/gpt-5-nano"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_TOKENS = 2048
DEFAULT_API_KEY_PATH = Path(__file__).resolve().parent / "api_key.txt"

#: NVIDIA NIM speaks the OpenAI chat-completions shape, so it reuses the
#: existing transport and only needs its own endpoint, model and secret path.
DEFAULT_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_NVIDIA_MODEL = "deepseek-ai/deepseek-v4.1-flash"
DEFAULT_NVIDIA_API_KEY_PATH = Path(__file__).resolve().parent.parent / "api_key_nvidia.txt"
#: The reasoning model needs longer than the shared 30s budget to answer the
#: full planner prompt (4k+ characters) before it emits the JSON plan. The
#: value only widens the transport ceiling; ``LLM_TIMEOUT_SECONDS`` still wins.
DEFAULT_NVIDIA_TIMEOUT_SECONDS = 120.0

#: Cline also speaks the OpenAI chat-completions shape; see ``cline.py``.
DEFAULT_CLINE_BASE_URL = "https://api.cline.bot/api/v1"
DEFAULT_CLINE_MODEL = "nvidia/nemotron-3.5-lightning:free"
DEFAULT_CLINE_API_KEY_PATH = Path(__file__).resolve().parent.parent / "api_key_cline.txt"
DEFAULT_CLINE_TIMEOUT_SECONDS = 120.0

#: Ollama is a local, loopback-only server with no credential. The native
#: ``/api/chat`` endpoint accepts ``options.num_ctx``, which is how the
#: requested 8192-token context target is configured (see ``ollama.py``). The
#: default context window is a starting point, not a verified limit.
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "qwen3.5:4b"
DEFAULT_OLLAMA_TIMEOUT_SECONDS = 120.0
DEFAULT_OLLAMA_NUM_CTX = 8192


class LLMConfigError(ValueError):
    """Raised when the environment names a provider that does not exist."""


@dataclass(frozen=True)
class LLMConfig:
    provider: str = PROVIDER_MOCK
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_tokens: int = DEFAULT_MAX_TOKENS
    api_key_path: Path | str = DEFAULT_API_KEY_PATH
    system_prompt: Optional[str] = None
    #: Context window requested from a local provider that supports it
    #: (currently only Ollama's native ``options.num_ctx``). Ignored by every
    #: remote provider, which negotiate context server-side.
    num_ctx: int = DEFAULT_OLLAMA_NUM_CTX
    #: Optional GPU-layer count for a local provider that supports it (only
    #: Ollama's native ``options.num_gpu``). ``None`` leaves placement to the
    #: provider's automatic default. Ignored by every remote provider.
    num_gpu: int | None = None

    @property
    def mock(self) -> bool:
        return self.provider == PROVIDER_MOCK

    @property
    def configured(self) -> bool:
        """Whether a real provider has everything it needs to attempt a call."""
        if self.provider == PROVIDER_MOCK:
            return True
        if self.provider == PROVIDER_OPENROUTER:
            return bool(_read_api_key(self.api_key_path))
        if self.provider == PROVIDER_NVIDIA:
            return bool(_read_api_key(self.api_key_path))
        if self.provider == PROVIDER_CLINE:
            return bool(_read_api_key(self.api_key_path))
        if self.provider == PROVIDER_OLLAMA:
            # A local server needs no credential; reachability is a runtime fact
            # the health surface reports, not a config-time one.
            return True
        return False

    def to_dict(self) -> dict[str, Any]:
        """Diagnostics-safe description: provider facts only, never the key."""
        return {
            "provider": self.provider,
            "mock": self.mock,
            "model": self.model,
            "base_url": self.base_url,
            "timeout_seconds": self.timeout_seconds,
            "max_tokens": self.max_tokens,
            "api_key_path": Path(self.api_key_path).name,
            "configured": self.configured,
        }

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> "LLMConfig":
        env = os.environ if environ is None else environ
        api_key_path = Path(env.get("LLM_API_KEY_PATH") or DEFAULT_API_KEY_PATH)
        requested = (env.get("LLM_PROVIDER") or "").strip().lower()
        if not requested:
            requested = (
                PROVIDER_OPENROUTER if _read_api_key(api_key_path) else PROVIDER_MOCK
            )
        if requested not in VALID_PROVIDERS:
            raise LLMConfigError(f"unknown_llm_provider:{requested}")
        if requested == PROVIDER_NVIDIA:
            default_model = DEFAULT_NVIDIA_MODEL
            default_base_url = DEFAULT_NVIDIA_BASE_URL
            default_api_key_path = DEFAULT_NVIDIA_API_KEY_PATH
            default_timeout = DEFAULT_NVIDIA_TIMEOUT_SECONDS
            default_max_tokens = DEFAULT_MAX_TOKENS
        elif requested == PROVIDER_CLINE:
            default_model = DEFAULT_CLINE_MODEL
            default_base_url = DEFAULT_CLINE_BASE_URL
            default_api_key_path = DEFAULT_CLINE_API_KEY_PATH
            default_timeout = DEFAULT_CLINE_TIMEOUT_SECONDS
            # The shared Brain token budget is kept as-is; ``cline.py``
            # disables the gateway's hidden reasoning instead of paying for it.
            default_max_tokens = DEFAULT_MAX_TOKENS
        elif requested == PROVIDER_OLLAMA:
            default_model = DEFAULT_OLLAMA_MODEL
            default_base_url = DEFAULT_OLLAMA_BASE_URL
            default_api_key_path = DEFAULT_API_KEY_PATH
            default_timeout = DEFAULT_OLLAMA_TIMEOUT_SECONDS
            default_max_tokens = DEFAULT_MAX_TOKENS
        else:
            default_model = DEFAULT_MODEL
            default_base_url = DEFAULT_BASE_URL
            default_api_key_path = DEFAULT_API_KEY_PATH
            default_timeout = DEFAULT_TIMEOUT_SECONDS
            default_max_tokens = DEFAULT_MAX_TOKENS
        if not env.get("LLM_API_KEY_PATH") and requested in (
            PROVIDER_NVIDIA,
            PROVIDER_CLINE,
        ):
            api_key_path = default_api_key_path
        return cls(
            provider=requested,
            model=env.get("LLM_MODEL") or default_model,
            base_url=env.get("LLM_BASE_URL") or default_base_url,
            timeout_seconds=_as_float(
                env.get("LLM_TIMEOUT_SECONDS"), default_timeout
            ),
            max_tokens=int(_as_float(env.get("LLM_MAX_TOKENS"), default_max_tokens)),
            api_key_path=api_key_path,
            system_prompt=(env.get("LLM_SYSTEM_PROMPT") or None),
            num_ctx=int(_as_float(env.get("LLM_NUM_CTX"), DEFAULT_OLLAMA_NUM_CTX)),
            num_gpu=_as_optional_int(env.get("LLM_NUM_GPU")),
        )


def build_provider(config: Optional[LLMConfig] = None) -> LLMProvider:
    """Create the configured provider. Never falls back to the mock silently."""
    resolved = config or LLMConfig()
    if resolved.provider == PROVIDER_OPENROUTER:
        from .openrouter import OpenRouterConfig, OpenRouterProvider

        transport_config = OpenRouterConfig(
            model=resolved.model,
            base_url=resolved.base_url,
            timeout_seconds=resolved.timeout_seconds,
            max_tokens=resolved.max_tokens,
            **(
                {"system_prompt": resolved.system_prompt}
                if resolved.system_prompt
                else {}
            ),
        )
        return OpenRouterProvider(transport_config, api_key_path=resolved.api_key_path)
    if resolved.provider == PROVIDER_NVIDIA:
        from .nvidia import NvidiaConfig, NvidiaProvider

        transport_config = NvidiaConfig(
            model=resolved.model,
            base_url=resolved.base_url,
            timeout_seconds=resolved.timeout_seconds,
            max_tokens=resolved.max_tokens,
            **(
                {"system_prompt": resolved.system_prompt}
                if resolved.system_prompt
                else {}
            ),
        )
        return NvidiaProvider(transport_config, api_key_path=resolved.api_key_path)
    if resolved.provider == PROVIDER_CLINE:
        from .cline import ClineConfig, ClineProvider

        transport_config = ClineConfig(
            model=resolved.model,
            base_url=resolved.base_url,
            timeout_seconds=resolved.timeout_seconds,
            max_tokens=resolved.max_tokens,
            **(
                {"system_prompt": resolved.system_prompt}
                if resolved.system_prompt
                else {}
            ),
        )
        return ClineProvider(transport_config, api_key_path=resolved.api_key_path)
    if resolved.provider == PROVIDER_OLLAMA:
        from .ollama import OllamaConfig, OllamaProvider

        # Ollama-specific knobs (``num_ctx``, temperature, top_p, bounded retry)
        # keep their ``OllamaConfig`` defaults; the shared fields come from the
        # environment so ``LLM_MODEL`` / ``LLM_BASE_URL`` / timeout still apply.
        transport_config = OllamaConfig(
            model=resolved.model,
            base_url=resolved.base_url,
            timeout_seconds=resolved.timeout_seconds,
            max_tokens=resolved.max_tokens,
            num_ctx=resolved.num_ctx,
            num_gpu=resolved.num_gpu,
            **(
                {"system_prompt": resolved.system_prompt}
                if resolved.system_prompt
                else {}
            ),
        )
        return OllamaProvider(transport_config)
    from .mock import MockLLMProvider

    return MockLLMProvider()


def describe_provider(provider: Any) -> dict[str, Any]:
    """Redacted description of a live provider instance for diagnostics."""
    if provider is None:
        return {
            "name": "none",
            "kind": "disabled",
            "mock": False,
            "configured": False,
            "model": None,
        }
    class_name = type(provider).__name__
    if class_name.startswith("Mock"):
        return {
            "name": getattr(provider, "name", "mock"),
            "kind": "mock",
            "mock": True,
            "configured": True,
            "model": None,
        }
    if hasattr(provider, "has_api_key"):
        config = getattr(provider, "config", None)
        name = getattr(provider, "name", "openrouter")
        model = getattr(config, "model", None)
        base_url = getattr(config, "base_url", None)
        return {
            "name": name,
            "kind": name,
            "mock": False,
            "configured": bool(getattr(provider, "has_api_key", False)),
            "model": model,
            "base_url": base_url,
        }
    return {
        "name": class_name,
        "kind": "custom",
        "mock": False,
        "configured": True,
        "model": getattr(getattr(provider, "config", None), "model", None),
    }


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


def _as_optional_int(value: Optional[str]) -> int | None:
    """A non-negative integer from an environment string, or ``None``.

    A blank/absent/unparseable/negative value means "not set", so a typo can
    never silently pin an unexpected GPU-layer count.
    """
    if value in (None, ""):
        return None
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


__all__ = [
    "DEFAULT_CLINE_API_KEY_PATH",
    "DEFAULT_CLINE_BASE_URL",
    "DEFAULT_CLINE_MODEL",
    "DEFAULT_CLINE_TIMEOUT_SECONDS",
    "DEFAULT_API_KEY_PATH",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_NVIDIA_API_KEY_PATH",
    "DEFAULT_NVIDIA_BASE_URL",
    "DEFAULT_NVIDIA_MODEL",
    "DEFAULT_NVIDIA_TIMEOUT_SECONDS",
    "DEFAULT_OLLAMA_BASE_URL",
    "DEFAULT_OLLAMA_MODEL",
    "DEFAULT_OLLAMA_NUM_CTX",
    "DEFAULT_OLLAMA_TIMEOUT_SECONDS",
    "LLMConfig",
    "LLMConfigError",
    "PROVIDER_CLINE",
    "PROVIDER_MOCK",
    "PROVIDER_NVIDIA",
    "PROVIDER_OLLAMA",
    "PROVIDER_OPENROUTER",
    "VALID_PROVIDERS",
    "build_provider",
    "describe_provider",
]
