"""Deterministic chat provider used by automated tests and offline development.

The PHASE 05 simulated input path (Unity ``BrainChatSubmitter`` posting to
``/chat/message``) is untouched; this provider is the same idea expressed at the
provider boundary so reconnect, dedup, and failure behaviour can be tested
without ever opening a socket.
"""

from __future__ import annotations

from collections import deque
from typing import Callable, Deque, Iterable, Optional

from ..models import utc_now
from .base import (
    ChatProvider,
    ChatProviderError,
    ChatSourceEvent,
    ProviderState,
    SourceEventKind,
)
from .config import ChatProviderConfig, PROVIDER_MOCK


class MockChatProvider(ChatProvider):
    name = PROVIDER_MOCK
    platform = "mock"
    supports_presence_events = True
    ingest_capacity = 25

    def __init__(
        self,
        config: ChatProviderConfig | None = None,
        clock: Callable[[], object] = utc_now,
        script: Iterable[ChatSourceEvent] = (),
    ) -> None:
        self.config = config or ChatProviderConfig(provider=PROVIDER_MOCK)
        self.ingest_capacity = max(1, int(self.config.ingest_capacity))
        self.clock = clock
        self._events: Deque[ChatSourceEvent] = deque(script)
        self._state = ProviderState.DISCONNECTED
        self._poll_count = 0
        self._connect_attempts = 0
        self._reconnects = 0
        self._last_error: Optional[str] = None
        # Scripted fault injection, all deterministic and all off by default.
        self.fail_next_connect = 0
        self.fail_next_poll = 0
        self.require_auth = False
        self.drop_connection_after: Optional[int] = None
        #: When False the source stays down until ``connect()`` is called again,
        #: which models a persistent platform outage instead of a blip.
        self.auto_reconnect = True

    @property
    def state(self) -> ProviderState:
        return self._state

    def push(self, event: ChatSourceEvent) -> None:
        self._events.append(event)

    def push_message(
        self,
        viewer_id: str,
        text: str,
        message_id: Optional[str] = None,
        viewer_name: str = "",
        kind: SourceEventKind = SourceEventKind.MESSAGE,
    ) -> ChatSourceEvent:
        event = ChatSourceEvent(
            kind=kind,
            viewer_id=viewer_id,
            viewer_name=viewer_name or viewer_id,
            text=text,
            message_id=message_id,
            timestamp=self.clock(),  # type: ignore[arg-type]
            platform=self.platform,
            channel=self.config.channel or "mock-channel",
        )
        self.push(event)
        return event

    def connect(self) -> None:
        if self._state is ProviderState.STOPPED:
            raise ChatProviderError("stopped")
        if not self.auto_reconnect and self._state in {
            ProviderState.DISCONNECTED,
            ProviderState.RETRYING,
        }:
            raise ChatProviderError("outage", "mock source is holding its outage")
        self._connect_attempts += 1
        if self.require_auth:
            self._state = ProviderState.DEGRADED
            self._last_error = "authentication_failed"
            raise ChatProviderError("authentication_failed", "mock credentials rejected")
        if self.fail_next_connect > 0:
            self.fail_next_connect -= 1
            self._state = ProviderState.RETRYING
            self._last_error = "connection_refused"
            raise ChatProviderError("connection_refused", "mock transport down")
        if self._state is ProviderState.RETRYING or self._state is ProviderState.DEGRADED:
            self._reconnects += 1
        self._state = ProviderState.CONNECTED
        self._last_error = None

    def disconnect(self) -> None:
        self._state = ProviderState.STOPPED
        self._events.clear()

    def lose_connection(self, reason: str = "connection_reset") -> None:
        """Simulate a temporary platform outage between polls."""
        self.auto_reconnect = False
        self._state = ProviderState.DISCONNECTED
        self._last_error = reason

    def poll(self) -> list[ChatSourceEvent]:
        if self._state is ProviderState.STOPPED:
            return []
        if self.fail_next_poll > 0:
            self.fail_next_poll -= 1
            self._state = ProviderState.DISCONNECTED
            self._last_error = "transport_error"
            raise ChatProviderError("transport_error", "mock read failure")
        if self._state in {ProviderState.DISCONNECTED, ProviderState.RETRYING}:
            # A blip heals on the next pump, exactly like the real provider's
            # bounded reconnect inside ``poll``.
            try:
                self.connect()
            except ChatProviderError:
                return []
        if self._state is not ProviderState.CONNECTED:
            return []
        self._poll_count += 1
        capacity = max(1, self.config.ingest_capacity)
        drained = [self._events.popleft() for _ in range(min(capacity, len(self._events)))]
        if (
            self.drop_connection_after is not None
            and self._poll_count >= self.drop_connection_after
        ):
            self.drop_connection_after = None
            self.auto_reconnect = False
            self._state = ProviderState.DISCONNECTED
            self._last_error = "connection_reset"
        return drained

    def channels(self) -> tuple[str, ...]:
        return tuple(self.config.channels) or ("mock-channel",)

    def diagnostics(self) -> dict[str, object]:
        payload = super().diagnostics()
        payload.update(
            {
                "connect_attempts": self._connect_attempts,
                "reconnect_count": self._reconnects,
                "pending_events": len(self._events),
                "last_error": self._last_error,
            }
        )
        return payload
