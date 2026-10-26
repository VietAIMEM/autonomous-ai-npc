"""Structured Phase 05 chat and dialogue models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def format_utc(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


class ChatStatus(str, Enum):
    RECEIVED = "RECEIVED"
    FILTERED = "FILTERED"
    QUEUED = "QUEUED"
    PROCESSED = "PROCESSED"
    REJECTED = "REJECTED"


class ChatIntent(str, Enum):
    GREETING = "GREETING"
    QUESTION = "QUESTION"
    REQUEST_ACTION = "REQUEST_ACTION"
    REQUEST_MOVEMENT = "REQUEST_MOVEMENT"
    COMMENT = "COMMENT"
    FAREWELL = "FAREWELL"
    # PHASE 26A: minimal continuity semantics resolved against the bounded
    # conversation context (never a wider intent taxonomy). Extraction still
    # returns the Phase 05 intents; the continuity resolver only refines them
    # when an active conversation already exists.
    FOLLOW_UP = "FOLLOW_UP"
    TOPIC_CONTINUATION = "TOPIC_CONTINUATION"
    ACKNOWLEDGEMENT = "ACKNOWLEDGEMENT"
    CONVERSATION_END = "CONVERSATION_END"
    UNKNOWN = "UNKNOWN"


@dataclass
class ChatMessage:
    message_id: str
    viewer_id: str
    raw_text: str
    normalized_text: str
    timestamp: datetime
    status: ChatStatus = ChatStatus.RECEIVED
    priority: float = 0.0
    intent: ChatIntent = ChatIntent.UNKNOWN
    target: Optional[str] = None
    repeat_count: int = 1
    filtered_reason: Optional[str] = None
    # PHASE 12: provenance for real platform messages. Optional so every
    # existing simulated/constructed message keeps working unchanged.
    platform: Optional[str] = None
    viewer_name: Optional[str] = None
    channel: Optional[str] = None
    # PHASE 25: desired spatial state carried alongside the target so the Brain
    # can plan a physical action without re-parsing the viewer's words.
    desired_state: Optional[str] = None
    # PA29-006: when the local attention queue admitted this message. Kept
    # apart from ``timestamp`` (when the viewer actually sent it) so an
    # upstream-age reading can never be confused with local queue wait.
    enqueued_at: Optional[datetime] = None

    def to_dict(self) -> dict[str, object]:
        return {
            "message_id": self.message_id,
            "viewer_id": self.viewer_id,
            "text": self.raw_text,
            "normalized_text": self.normalized_text,
            "timestamp": format_utc(self.timestamp),
            "status": self.status.value,
            "priority": round(self.priority, 3),
            "intent": self.intent.value,
            "target": self.target,
            "repeat_count": self.repeat_count,
            "filtered_reason": self.filtered_reason,
            "platform": self.platform,
            "viewer_name": self.viewer_name,
            "channel": self.channel,
            "desired_state": self.desired_state,
        }


@dataclass(frozen=True)
class DialogueResponse:
    speech: str
    intent: ChatIntent
    requested_action: Optional[str]
    priority: float
    message_id: str
    viewer_id: str

    def to_dict(self) -> dict[str, object]:
        return {
            "speech": self.speech,
            "intent": self.intent.value,
            "requested_action": self.requested_action,
            "priority": round(self.priority, 3),
            "message_id": self.message_id,
            "viewer_id": self.viewer_id,
        }


@dataclass(frozen=True)
class WorldContext:
    character_state: str = "standing"
    character_location: str = "near center"
    phone_state: str = "on table"
    stream_state: str = "stopped"

    def to_prompt_lines(self) -> list[str]:
        return [
            f"Lacrimosa: {self.character_state}, {self.character_location}",
            f"Phone: {self.phone_state}",
            f"Stream: {self.stream_state}",
        ]
