"""Typed Phase 08 social autonomy models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from ..chat.models import ChatMessage, format_utc


@dataclass(frozen=True)
class SocialConfig:
    attend_now_threshold: float = 0.62
    attention_threshold: float = 0.35
    busy_defer_threshold: float = 0.45
    speech_cooldown_seconds: float = 12.0
    attention_cooldown_seconds: float = 5.0
    conversation_timeout_seconds: float = 180.0
    merge_window_seconds: float = 30.0
    max_recent_messages: int = 8
    max_recent_reactions: int = 6
    max_pending_requests: int = 10
    max_viewer_recent_topics: int = 5

    def __post_init__(self) -> None:
        values = (
            self.attend_now_threshold,
            self.attention_threshold,
            self.busy_defer_threshold,
            self.speech_cooldown_seconds,
            self.attention_cooldown_seconds,
            self.conversation_timeout_seconds,
            self.merge_window_seconds,
        )
        if any(value < 0 for value in values):
            raise ValueError("social thresholds and windows cannot be negative")
        if self.attention_threshold > self.attend_now_threshold:
            raise ValueError("attention_threshold cannot exceed attend_now_threshold")
        if min(
            self.max_recent_messages,
            self.max_recent_reactions,
            self.max_pending_requests,
            self.max_viewer_recent_topics,
        ) <= 0:
            raise ValueError("social buffer sizes must be greater than zero")


class ConversationState(str, Enum):
    IDLE = "IDLE"
    GREETING = "GREETING"
    CONVERSING = "CONVERSING"
    WAITING = "WAITING"
    DEFERRED = "DEFERRED"
    CHANGING_TOPIC = "CHANGING_TOPIC"
    ENDING = "ENDING"


class AttentionDecision(str, Enum):
    ATTEND_NOW = "ATTEND_NOW"
    ATTEND_SOON = "ATTEND_SOON"
    DEFER = "DEFER"
    IGNORE = "IGNORE"
    MERGE = "MERGE"


class SocialSpeakingPolicy(str, Enum):
    RESPOND = "RESPOND"
    CONTINUE_CONVERSATION = "CONTINUE_CONVERSATION"
    CHANGE_TOPIC = "CHANGE_TOPIC"
    DEFER = "DEFER"
    IGNORE = "IGNORE"


@dataclass(frozen=True)
class SocialDecision:
    decision: AttentionDecision
    speaking_policy: SocialSpeakingPolicy
    viewer_id: str
    topic: Optional[str]
    reason: str
    priority: float
    attention_score: float
    message: Optional[ChatMessage] = None
    speech: Optional[str] = None
    requested_action: Optional[str] = None

    def to_dict(self, include_message: bool = False) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "speaking_policy": self.speaking_policy.value,
            "viewer_id": self.viewer_id,
            "topic": self.topic,
            "reason": self.reason,
            "priority": round(self.priority, 3),
            "attention_score": round(self.attention_score, 3),
            "message": self.message.to_dict() if include_message and self.message else None,
            "speech": self.speech,
            "requested_action": self.requested_action,
        }


@dataclass
class TopicState:
    viewer_id: str
    topic: str
    started_at: datetime
    last_message_at: datetime
    last_message_id: Optional[str] = None
    message_count: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "viewer_id": self.viewer_id,
            "topic": self.topic,
            "started_at": format_utc(self.started_at),
            "last_message_at": format_utc(self.last_message_at),
            "last_message_id": self.last_message_id,
            "message_count": self.message_count,
        }


@dataclass
class PendingInteraction:
    message: ChatMessage
    received_at: datetime
    topic: Optional[str]
    reason: str
    attention_score: float
    attempts: int = 0

    @property
    def message_id(self) -> str:
        return self.message.message_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message.message_id,
            "viewer_id": self.message.viewer_id,
            "text": self.message.raw_text,
            "received_at": format_utc(self.received_at),
            "topic": self.topic,
            "reason": self.reason,
            "attention_score": round(self.attention_score, 3),
            "attempts": self.attempts,
        }


@dataclass(frozen=True)
class ViewerSocialContext:
    viewer_id: str
    relationship_score: float = 0.2
    preferences: tuple[str, ...] = ()
    recent_topics: tuple[str, ...] = ()
    interaction_count: int = 0
    summary: str = ""

    @classmethod
    def from_memory(cls, viewer_id: str, memory_viewer: dict[str, Any] | None) -> "ViewerSocialContext":
        if not memory_viewer:
            return cls(viewer_id)
        return cls(
            viewer_id=viewer_id,
            relationship_score=float(memory_viewer.get("relationship_score", 0.2)),
            preferences=tuple(memory_viewer.get("preferences", ())),
            recent_topics=tuple(memory_viewer.get("recent_topics", ())),
            interaction_count=int(memory_viewer.get("interaction_count", 0)),
            summary="viewer memory available",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "viewer_id": self.viewer_id,
            "relationship_score": round(self.relationship_score, 3),
            "preferences": list(self.preferences),
            "recent_topics": list(self.recent_topics),
            "interaction_count": self.interaction_count,
            "summary": self.summary,
        }
