"""Minimal LLM abstraction shared by chat and future reasoning systems."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class LLMRequest:
    prompt: str
    context: Mapping[str, Any] = field(default_factory=dict)


class LLMProviderError(RuntimeError):
    """Raised for provider transport or execution failures."""


@dataclass(frozen=True)
class TransportFailureInfo:
    """Bounded transport-side facts about the call that failed.

    Observability only: nothing in the runtime reads this to make a decision.
    It exists so the failure diagnostic can report what the HTTP transport
    actually did -- how many attempts it made, whether a retryable status spent
    the whole configured budget -- without the diagnostics module having to
    infer either from elapsed time or from an exception message.

    ``rate_limit_headers`` holds only values already filtered against the
    validated allowlist; ``rate_limit_headers_unparsed`` names allowlisted
    headers that were present but whose values were not safely representable.
    The transport attaches this to the ``LLMProviderError`` it raises, so the
    information travels with the existing exception instead of through a
    global or cross-request store.
    """

    attempt_count: int
    configured_max_attempts: int
    retries_exhausted: bool
    rate_limit_headers: Mapping[str, str] = field(default_factory=dict)
    rate_limit_headers_unparsed: tuple[str, ...] = ()


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, request: LLMRequest) -> str:
        raise NotImplementedError

