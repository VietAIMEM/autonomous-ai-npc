"""Bounded social context for the existing autonomous brain."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from itertools import count
from typing import Any, Optional

from ..chat.models import ChatMessage, format_utc
from ..runtime.stream_latch import StreamStopLatch
from .models import (
    ConversationState,
    PendingInteraction,
    SocialConfig,
    SocialDecision,
    TopicState,
)


@dataclass(frozen=True)
class SocialReaction:
    viewer_id: str
    kind: str
    timestamp: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "viewer_id": self.viewer_id,
            "kind": self.kind,
            "timestamp": format_utc(self.timestamp),
        }


class SocialContext:
    """Small event-driven state; never replaces chat or memory storage."""

    def __init__(
        self,
        config: SocialConfig | None = None,
        *,
        stop_latch: StreamStopLatch | None = None,
    ) -> None:
        self.config = config or SocialConfig()
        self._stop_latch = stop_latch
        self.active_viewer_id: Optional[str] = None
        self.active_topic: Optional[str] = None
        self.active_message_id: Optional[str] = None
        self.conversation_state = ConversationState.IDLE
        self.conversation_started_at: Optional[datetime] = None
        self.last_activity_at: Optional[datetime] = None
        self.last_speech_at: Optional[datetime] = None
        self.last_attention_at: Optional[datetime] = None
        self.last_social_action: Optional[dict[str, Any]] = None
        # A deliberate stop must survive a backend restart and must stay
        # visible to the evaluator when the control plane writes the durable
        # latch, so a wired latch is the source of truth for this bit and the
        # private field only backs the latch-less in-memory case.
        self._last_intentional_stream_stop: bool = (
            bool(stop_latch.load()) if stop_latch is not None else False
        )
        self.recent_messages: list[ChatMessage] = []
        self.recent_reactions: list[SocialReaction] = []
        self.pending_requests: list[PendingInteraction] = []
        self.viewer_topics: dict[str, list[str]] = {}
        self.viewer_last_seen: dict[str, datetime] = {}
        self.social_engagement: float = 0.0
        self._pending_sequence = count(1)

    def record_message(self, message: ChatMessage, now: datetime) -> None:
        self.recent_messages.append(message)
        if len(self.recent_messages) > self.config.max_recent_messages:
            del self.recent_messages[:-self.config.max_recent_messages]
        self.viewer_last_seen[message.viewer_id] = now
        self.last_activity_at = now
        self._refresh_engagement(now)

    def note_reaction(self, viewer_id: str, kind: str, now: datetime) -> None:
        if kind not in {"positive", "negative", "neutral", "request_followup"}:
            return
        self.recent_reactions.append(SocialReaction(viewer_id, kind, now))
        if len(self.recent_reactions) > self.config.max_recent_reactions:
            del self.recent_reactions[:-self.config.max_recent_reactions]
        self._refresh_engagement(now)

    def record_topic(self, message: ChatMessage, topic: str, now: datetime) -> None:
        topics = self.viewer_topics.setdefault(message.viewer_id, [])
        if topic in topics:
            topics.remove(topic)
        topics.append(topic)
        if len(topics) > self.config.max_viewer_recent_topics:
            del topics[:-self.config.max_viewer_recent_topics]

    def activate(
        self,
        message: ChatMessage,
        topic: str | None,
        now: datetime,
        state: ConversationState,
    ) -> None:
        previous_topic = self.active_topic
        self.active_viewer_id = message.viewer_id
        self.active_message_id = message.message_id
        if topic and previous_topic and topic != previous_topic:
            self.active_topic = topic
            self.conversation_state = ConversationState.CHANGING_TOPIC
        else:
            self.active_topic = topic or self.active_topic
            self.conversation_state = state
        if self.active_viewer_id != message.viewer_id or self.conversation_started_at is None:
            self.conversation_started_at = now
        if topic:
            self.record_topic(message, topic, now)
        self.last_activity_at = now

    def record_speech(self, viewer_id: str, speech: str, now: datetime) -> None:
        self.last_speech_at = now
        self.last_activity_at = now
        if self.conversation_state in {
            ConversationState.GREETING,
            ConversationState.CHANGING_TOPIC,
        }:
            self.conversation_state = ConversationState.CONVERSING
        self.social_engagement = min(1.0, self.social_engagement + 0.08)

    def defer(self, message: ChatMessage, topic: str | None, reason: str, score: float, now: datetime) -> PendingInteraction:
        existing = self.find_pending(message.message_id)
        if existing is not None:
            existing.attempts += 1
            existing.reason = reason
            return existing
        pending = PendingInteraction(
            message=message,
            received_at=now,
            topic=topic,
            reason=reason,
            attention_score=score,
        )
        self.pending_requests.append(pending)
        if len(self.pending_requests) > self.config.max_pending_requests:
            del self.pending_requests[:-self.config.max_pending_requests]
        self.conversation_state = ConversationState.DEFERRED
        self.last_activity_at = now
        return pending

    def release_pending(self, message_id: str | None = None) -> list[PendingInteraction]:
        if message_id is None:
            released = list(self.pending_requests)
            self.pending_requests.clear()
        else:
            released = [item for item in self.pending_requests if item.message.message_id == message_id]
            self.pending_requests = [
                item for item in self.pending_requests if item.message.message_id != message_id
            ]
        if not self.pending_requests and self.conversation_state == ConversationState.DEFERRED:
            self.conversation_state = ConversationState.WAITING
        return released

    def find_pending(self, message_id: str) -> PendingInteraction | None:
        return next(
            (item for item in self.pending_requests if item.message.message_id == message_id),
            None,
        )

    def update_on_decision(self, decision: SocialDecision, now: datetime) -> None:
        self.last_attention_at = now

    def update_on_action_result(
        self,
        action_name: str,
        success: bool,
        now: datetime,
        viewer_id: str | None = None,
    ) -> None:
        self.last_social_action = {
            "action": action_name,
            "success": success,
            "viewer_id": viewer_id,
            "at": format_utc(now),
        }
        if action_name == "STOP_STREAM" and success:
            self.set_stream_stop_latch(True, source="brain:STOP_STREAM")
        elif action_name == "START_STREAM" and success:
            self.set_stream_stop_latch(False, source="brain:START_STREAM")
        if self.conversation_state == ConversationState.DEFERRED and success:
            self.conversation_state = ConversationState.WAITING
        self._refresh_engagement(now)

    def set_stream_stop_latch(self, stopped: bool, *, source: str = "") -> bool:
        """Record an explicit stop/start and persist it when one is wired in.

        Returns ``True`` when the value reached durable storage (always true
        for an in-memory latch). A storage fault never raises: the in-memory
        flag still protects this process, and ``snapshot()`` reports the fault.
        """
        self.last_intentional_stream_stop = bool(stopped)
        if self._stop_latch is None:
            return True
        return self._stop_latch.store(bool(stopped), source=source)

    @property
    def last_intentional_stream_stop(self) -> bool:
        """Whether the stream was deliberately stopped.

        The durable latch is shared with the admin/API control plane, so
        reading through it keeps one source of truth: an operator stop gates
        the very next Brain tick instead of only surviving a restart.
        """
        if self._stop_latch is not None:
            return self._stop_latch.stopped
        return self._last_intentional_stream_stop

    @last_intentional_stream_stop.setter
    def last_intentional_stream_stop(self, value: bool) -> None:
        self._last_intentional_stream_stop = bool(value)

    def expire_if_needed(self, now: datetime) -> bool:
        if self.conversation_state == ConversationState.IDLE:
            return False
        reference = self.last_activity_at or self.conversation_started_at
        if reference is None:
            return False
        if (now - reference).total_seconds() < self.config.conversation_timeout_seconds:
            return False
        self.conversation_state = ConversationState.IDLE
        self.active_viewer_id = None
        self.active_topic = None
        self.active_message_id = None
        self.social_engagement = 0.0
        return True

    def is_recent_duplicate(self, message: ChatMessage) -> bool:
        normalized = message.normalized_text.casefold()
        for recent in reversed(self.recent_messages):
            elapsed = (message.timestamp - recent.timestamp).total_seconds()
            if elapsed > self.config.merge_window_seconds:
                break
            if recent.viewer_id == message.viewer_id and recent.normalized_text.casefold() == normalized:
                return True
        return False

    def snapshot(self) -> dict[str, Any]:
        return {
            "active_viewer_id": self.active_viewer_id,
            "active_topic": self.active_topic,
            "active_message_id": self.active_message_id,
            "conversation_state": self.conversation_state.value,
            "conversation_started_at": format_utc(self.conversation_started_at) if self.conversation_started_at else None,
            "last_activity_at": format_utc(self.last_activity_at) if self.last_activity_at else None,
            "last_speech_at": format_utc(self.last_speech_at) if self.last_speech_at else None,
            "last_attention_at": format_utc(self.last_attention_at) if self.last_attention_at else None,
            "recent_messages": [message.to_dict() for message in self.recent_messages],
            "recent_reactions": [reaction.to_dict() for reaction in self.recent_reactions],
            "pending_requests": [item.to_dict() for item in self.pending_requests],
            "viewer_recent_topics": {
                viewer_id: list(topics)
                for viewer_id, topics in self.viewer_topics.items()
            },
            "social_engagement": round(self.social_engagement, 3),
            "last_social_action": self.last_social_action,
            "last_intentional_stream_stop": self.last_intentional_stream_stop,
            "stream_stop_latch": (
                self._stop_latch.snapshot() if self._stop_latch is not None else None
            ),
        }

    def _refresh_engagement(self, now: datetime) -> None:
        recent_count = len(
            [
                message
                for message in self.recent_messages
                if (now - message.timestamp).total_seconds() <= self.config.conversation_timeout_seconds
            ]
        )
        viewer_count = len(
            [
                viewer_id
                for viewer_id, stamp in self.viewer_last_seen.items()
                if (now - stamp).total_seconds() <= self.config.conversation_timeout_seconds
            ]
        )
        message_signal = min(1.0, recent_count / max(1, self.config.max_recent_messages))
        viewer_signal = min(1.0, viewer_count / 3.0)
        pending_signal = 0.1 if self.pending_requests else 0.0
        reaction_signal = 0.08 if self.recent_reactions else 0.0
        self.social_engagement = max(0.0, min(1.0, message_signal * 0.55 + viewer_signal * 0.25 + pending_signal + reaction_signal))
