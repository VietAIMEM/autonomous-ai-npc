"""Typed models for the Phase 07 memory subsystem."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from ..chat.models import format_utc


DEFAULT_MEMORY_DB_PATH = Path(__file__).resolve().parents[1] / "data" / "memory.db"
SCHEMA_VERSION = 2


class RelationshipLevel(str, Enum):
    """Small interpretable set of viewer relationship levels (PHASE 19).

    Progression is evidence based and gradual; see ``MemoryService`` for the
    bounded thresholds. The levels never replace viewer memory, they only
    summarise it for diagnostics and bounded personalization.
    """

    NEW = "NEW"
    FAMILIAR = "FAMILIAR"
    REGULAR = "REGULAR"
    TRUSTED = "TRUSTED"

    @property
    def rank(self) -> int:
        return _RELATIONSHIP_ORDER[self]


_RELATIONSHIP_ORDER = {
    RelationshipLevel.NEW: 0,
    RelationshipLevel.FAMILIAR: 1,
    RelationshipLevel.REGULAR: 2,
    RelationshipLevel.TRUSTED: 3,
}


def coerce_relationship_level(value: Any) -> RelationshipLevel:
    """Tolerate missing/corrupt stored levels without raising."""
    if isinstance(value, RelationshipLevel):
        return value
    if isinstance(value, str):
        try:
            return RelationshipLevel(value.strip().upper())
        except ValueError:
            return RelationshipLevel.NEW
    return RelationshipLevel.NEW


def _bounded(value: Any, fallback: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return fallback
    number = float(value)
    if number != number:  # NaN
        return fallback
    return max(0.0, min(1.0, number))


def _count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return max(0, int(value))


def _string_tuple(value: Any, limit: int = 5) -> tuple[str, ...]:
    if isinstance(value, str):
        value = (value,)
    if not isinstance(value, (list, tuple, set, frozenset)):
        return ()
    cleaned = [item.strip() for item in value if isinstance(item, str) and item.strip()]
    return tuple(dict.fromkeys(cleaned))[:limit]


class MemoryType(str, Enum):
    EPISODE = "episode"
    SEMANTIC = "semantic"
    VIEWER = "viewer"
    INTERACTION = "interaction"


@dataclass(frozen=True)
class MemoryConfig:
    database_path: Path = DEFAULT_MEMORY_DB_PATH
    short_term_message_limit: int = 20
    short_term_event_limit: int = 30
    short_term_action_limit: int = 20
    importance_threshold: float = 0.55
    retrieval_top_k: int = 5
    semantic_confidence_threshold: float = 0.65
    semantic_evidence_threshold: int = 2
    recency_window_seconds: float = 900.0
    max_summary_length: int = 180
    max_tag_count: int = 12


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class ShortTermItem:
    item_id: str
    kind: str
    text: str
    timestamp: datetime
    importance: float = 0.0
    viewer_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "kind": self.kind,
            "text": self.text[:self.max_text_length()],
            "timestamp": format_utc(self.timestamp),
            "importance": round(self.importance, 3),
            "viewer_id": self.viewer_id,
            "metadata": self.metadata,
        }

    @staticmethod
    def max_text_length() -> int:
        return 500


@dataclass(frozen=True)
class Episode:
    episode_id: str
    timestamp: datetime
    event: str
    context: dict[str, Any]
    action: Optional[str]
    result: str
    importance: float
    viewer_id: Optional[str] = None
    goal: Optional[str] = None
    location: Optional[str] = None
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "timestamp": format_utc(self.timestamp),
            "event": self.event,
            "context": self.context,
            "action": self.action,
            "result": self.result,
            "importance": round(self.importance, 3),
            "viewer_id": self.viewer_id,
            "goal": self.goal,
            "location": self.location,
            "tags": list(self.tags[:MemoryConfig().max_tag_count]),
        }


@dataclass(frozen=True)
class SemanticMemory:
    memory_id: str
    subject: str
    predicate: str
    value: str
    confidence: float
    importance: float
    source: str
    evidence_count: int = 1
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @property
    def statement(self) -> str:
        return f"{self.subject} {self.predicate} {self.value}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "subject": self.subject,
            "predicate": self.predicate,
            "value": self.value,
            "confidence": round(self.confidence, 3),
            "importance": round(self.importance, 3),
            "source": self.source,
            "evidence_count": self.evidence_count,
            "statement": self.statement,
            "created_at": format_utc(self.created_at) if self.created_at else None,
            "updated_at": format_utc(self.updated_at) if self.updated_at else None,
        }


@dataclass
class ViewerMemory:
    viewer_id: str
    interaction_count: int = 0
    successful_interactions: int = 0
    failed_interactions: int = 0
    relationship_score: float = 0.2
    last_interaction: Optional[datetime] = None
    preferences: tuple[str, ...] = ()
    recent_topics: tuple[str, ...] = ()
    # PHASE 19 fields are appended last so every existing positional
    # construction (Phase 07-18 callers and tests) keeps its meaning.
    relationship_level: str = RelationshipLevel.NEW.value
    engagement_score: float = 0.0
    familiarity: float = 0.0
    preferred_skills: tuple[str, ...] = ()
    preferred_topics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Normalize stored values so corrupted rows degrade instead of raising."""
        self.viewer_id = str(self.viewer_id)
        self.interaction_count = _count(self.interaction_count)
        self.successful_interactions = _count(self.successful_interactions)
        self.failed_interactions = _count(self.failed_interactions)
        self.relationship_score = _bounded(self.relationship_score, 0.2)
        self.relationship_level = coerce_relationship_level(self.relationship_level).value
        self.engagement_score = _bounded(self.engagement_score)
        self.familiarity = _bounded(self.familiarity)
        self.preferences = _string_tuple(self.preferences)
        self.recent_topics = _string_tuple(self.recent_topics)
        self.preferred_skills = _string_tuple(self.preferred_skills)
        self.preferred_topics = _string_tuple(self.preferred_topics)

    def to_dict(self) -> dict[str, Any]:
        return {
            "viewer_id": self.viewer_id,
            "interaction_count": self.interaction_count,
            "successful_interactions": self.successful_interactions,
            "failed_interactions": self.failed_interactions,
            "relationship_score": round(self.relationship_score, 3),
            "last_interaction": format_utc(self.last_interaction) if self.last_interaction else None,
            "preferences": list(self.preferences),
            "recent_topics": list(self.recent_topics),
            "relationship_level": self.relationship_level,
            "engagement_score": round(self.engagement_score, 3),
            "familiarity": round(self.familiarity, 3),
            "preferred_skills": list(self.preferred_skills),
            "preferred_topics": list(self.preferred_topics),
        }


@dataclass(frozen=True)
class InteractionRecord:
    interaction_id: str
    timestamp: datetime
    viewer_id: Optional[str]
    message_id: Optional[str]
    intent: Optional[str]
    requested_action: Optional[str]
    action: Optional[str]
    response: Optional[str]
    result: str
    importance: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "interaction_id": self.interaction_id,
            "timestamp": format_utc(self.timestamp),
            "viewer_id": self.viewer_id,
            "message_id": self.message_id,
            "intent": self.intent,
            "requested_action": self.requested_action,
            "action": self.action,
            "response": self.response[:500] if self.response else None,
            "result": self.result,
            "importance": round(self.importance, 3),
        }


@dataclass(frozen=True)
class MemoryQuery:
    viewer_id: Optional[str] = None
    topics: tuple[str, ...] = ()
    goal: Optional[str] = None
    intent: Optional[str] = None
    action: Optional[str] = None
    text: Optional[str] = None

    def topic_terms(self) -> tuple[str, ...]:
        values = {item.lower() for item in self.topics if item}
        for value in (self.goal, self.intent, self.action, self.text):
            if isinstance(value, str) and value:
                values.update(part for part in value.lower().replace("_", " ").split() if len(part) > 2)
        return tuple(sorted(values))


@dataclass(frozen=True)
class RetrievedMemory:
    memory_type: MemoryType
    memory_id: str
    summary: str
    score: float
    importance: float
    timestamp: datetime
    viewer_id: Optional[str] = None
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.memory_type.value,
            "id": self.memory_id,
            "summary": self.summary,
            "score": round(self.score, 3),
            "importance": round(self.importance, 3),
            "timestamp": format_utc(self.timestamp),
            "viewer_id": self.viewer_id,
        }

    def to_prompt_dict(self) -> dict[str, str]:
        return {"type": self.memory_type.value, "summary": self.summary}
