"""Chat source abstraction shared by every provider and the ingest runner.

PHASE 12 separates *where messages come from* from *how they are processed*.
A provider only delivers platform-neutral :class:`ChatSourceEvent` values; the
existing ChatService keeps ownership of normalization, filtering, dedup,
rate limiting, priority, and queueing. Nothing in this module talks to the
Brain, Memory, or Unity.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Optional, Sequence


MAX_EVENT_TEXT_LENGTH = 2000


class ProviderState(str, Enum):
    """Connection lifecycle for a chat source."""

    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    RETRYING = "RETRYING"
    DEGRADED = "DEGRADED"
    STOPPED = "STOPPED"


class SourceEventKind(str, Enum):
    MESSAGE = "MESSAGE"
    VIEWER_JOINED = "VIEWER_JOINED"
    VIEWER_LEFT = "VIEWER_LEFT"


class ChatProviderError(RuntimeError):
    """Raised for transport, protocol, or authentication failures.

    ``detail`` is expected to be already credential-free; providers must
    scrub before raising.
    """

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}:{detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


def to_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class ChatSourceEvent:
    """One platform-neutral thing that happened in a chat room."""

    kind: SourceEventKind
    viewer_id: str = ""
    viewer_name: str = ""
    text: str = ""
    message_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    platform: str = "unknown"
    channel: Optional[str] = None
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.kind, SourceEventKind):
            raise ValueError("kind must be a SourceEventKind")
        if len(self.text) > MAX_EVENT_TEXT_LENGTH:
            raise ValueError("text exceeds the ingest limit")

    @property
    def is_message(self) -> bool:
        return self.kind is SourceEventKind.MESSAGE

    def to_dict(self) -> dict[str, Any]:
        """Diagnostics view. Structurally incapable of carrying secrets."""
        return {
            "kind": self.kind.value,
            "viewer_id": self.viewer_id,
            "viewer_name": self.viewer_name,
            "text": self.text if self.is_message else "",
            "message_id": self.message_id,
            "timestamp": (
                to_utc(self.timestamp).isoformat(timespec="seconds").replace("+00:00", "Z")
                if self.timestamp is not None
                else None
            ),
            "platform": self.platform,
            "channel": self.channel,
        }


@dataclass(frozen=True)
class ReconnectPolicy:
    """Bounded exponential backoff, so a platform outage cannot spin the CPU."""

    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 30.0
    max_attempts: int = 5

    def __post_init__(self) -> None:
        if self.base_delay_seconds <= 0:
            raise ValueError("base_delay_seconds must be positive")
        if self.max_delay_seconds < self.base_delay_seconds:
            raise ValueError("max_delay_seconds must be >= base_delay_seconds")
        if self.max_attempts <= 0:
            raise ValueError("max_attempts must be positive")

    def delay_for_attempt(self, attempt: int) -> float:
        """Delay before ``attempt`` (1-based) without any randomness."""
        exponent = max(0, int(attempt) - 1)
        delay = self.base_delay_seconds * (2 ** min(exponent, 16))
        return round(min(self.max_delay_seconds, delay), 3)

    def allows_attempt(self, attempt: int) -> bool:
        return int(attempt) <= self.max_attempts

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_delay_seconds": self.base_delay_seconds,
            "max_delay_seconds": self.max_delay_seconds,
            "max_attempts": self.max_attempts,
        }


class ChatProvider(ABC):
    """Delivery boundary for one chat platform.

    Implementations must never raise out of :meth:`poll` for ordinary network
    trouble; they move to :attr:`ProviderState.DEGRADED` and report through
    :meth:`diagnostics` instead, so the ingest loop stays alive.
    """

    name: str = "base"
    platform: str = "unknown"
    #: Whether this platform reliably reports join/leave. Providers that cannot
    #: say so must leave this False rather than inventing presence events.
    supports_presence_events: bool = False
    #: Hard cap on how many events one poll may hand over.
    ingest_capacity: int = 25

    @abstractmethod
    def connect(self) -> None:
        """Open the source. Raises :class:`ChatProviderError` on failure."""

    @abstractmethod
    def disconnect(self) -> None:
        """Close the source. Idempotent; never raises."""

    @abstractmethod
    def poll(self) -> Sequence[ChatSourceEvent]:
        """Drain what is currently available without blocking indefinitely."""

    @property
    @abstractmethod
    def state(self) -> ProviderState:
        ...

    def channels(self) -> tuple[str, ...]:
        return ()

    def diagnostics(self) -> dict[str, Any]:
        """Connection facts only. No credentials, no raw transport buffers."""
        return {
            "provider": self.name,
            "platform": self.platform,
            "state": self.state.value,
            "channels": list(self.channels()),
            "presence_events": self.supports_presence_events,
        }
