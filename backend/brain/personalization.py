"""PHASE 19 bounded viewer personalization for the existing Brain loop.

The builder turns one viewer's existing memory state into a small, inspectable
preference block. It never retrieves whole histories, never fabricates
evidence, never stores anything, and never executes an action: it only shapes
context that the existing planner, chat prompt and skill ranking may use.

Every failure path returns a neutral generic context so chat, Brain, TTS and
Unity stay operational when personalization data is missing or corrupt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .models import PersonalityProfile
from .personality import personality_speech_style


#: Relationship vocabulary shared with ``memory.models.RelationshipLevel``.
RELATIONSHIP_LEVELS = ("NEW", "FAMILIAR", "REGULAR", "TRUSTED")

_MAX_TOPICS = 3
_MAX_SKILLS = 3
_MAX_REASON_CHARS = 200
#: Personalization may only nudge an advisory ranking inside this bound; the
#: learned history itself is never rewritten.
MAX_SKILL_PREFERENCE_BIAS = 0.12
_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    {
        "about", "again", "also", "and", "are", "can", "could", "did", "does",
        "for", "from", "have", "her", "him", "how", "into", "its", "just",
        "lacrimosa", "like", "more", "much", "not", "now", "our", "out",
        "please", "she", "some", "than", "that", "the", "their", "them",
        "then", "there", "these", "they", "this", "those", "too", "very",
        "was", "were", "what", "when", "where", "which", "who", "why",
        "will", "with", "would", "you", "your",
    }
)


@dataclass(frozen=True)
class PersonalizedContext:
    """Bounded viewer-specific preference context for one interaction."""

    viewer_id: str = ""
    known: bool = False
    relationship_level: str = "NEW"
    familiarity: float = 0.0
    interaction_count: int = 0
    engagement_score: float = 0.0
    preferred_topics: tuple[str, ...] = ()
    preferred_skills: tuple[str, ...] = ()
    recent_topic: str | None = None
    last_interaction: str | None = None
    topic_continuity: str | None = None
    greeting: str | None = None
    speech_style: str | None = None
    reason: str = "generic context: personalization unavailable"
    degraded: bool = False

    @property
    def returning(self) -> bool:
        return self.known and self.interaction_count >= 1

    def to_prompt_block(self) -> dict[str, Any]:
        """Compact advisory block; no transcript or stored history is copied."""
        return {
            "known_viewer": self.known,
            "returning": self.returning,
            "relationship_level": self.relationship_level,
            "familiarity": round(self.familiarity, 3),
            "interaction_count": self.interaction_count,
            "preferred_topics": list(self.preferred_topics),
            "preferred_skills": list(self.preferred_skills),
            "topic_continuity": self.topic_continuity,
            "greeting": self.greeting,
            "speech_style": self.speech_style,
            "reason": self.reason,
            "rule": (
                "advisory personalization only; never bypasses skill conditions, "
                "cooldowns, allowlists, social attention, or SkillExecutor"
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        block = self.to_prompt_block()
        block["viewer_id"] = self.viewer_id
        block["engagement_score"] = round(self.engagement_score, 3)
        block["recent_topic"] = self.recent_topic
        block["last_interaction"] = self.last_interaction
        block["degraded"] = self.degraded
        return block


def _safe_level(value: Any) -> str:
    if isinstance(value, str) and value.strip().upper() in RELATIONSHIP_LEVELS:
        return value.strip().upper()
    return "NEW"


def _safe_unit(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    number = float(value)
    if number != number:
        return 0.0
    return max(0.0, min(1.0, number))


def _safe_count(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return max(0, int(value))


def _safe_items(value: Any, limit: int) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    cleaned = [item.strip().lower() for item in value if isinstance(item, str) and item.strip()]
    return tuple(dict.fromkeys(cleaned))[:limit]


def _keywords(text: str | None) -> set[str]:
    if not isinstance(text, str) or not text:
        return set()
    return {
        word
        for word in _WORD.findall(text.lower())
        if len(word) > 2 and word not in _STOPWORDS
    }


def _blank_context(
    viewer_id: str | None,
    reason: str,
    *,
    degraded: bool = False,
    speech_style: str | None = None,
) -> PersonalizedContext:
    return PersonalizedContext(
        viewer_id=(viewer_id or "").strip()[:120],
        known=False,
        speech_style=speech_style,
        reason=reason[:_MAX_REASON_CHARS],
        degraded=degraded,
    )


def build_personalized_context(
    viewer_id: str | None,
    viewer_state: Mapping[str, Any] | None,
    *,
    message_text: str | None = None,
    intent: str | None = None,
    greeting_allowed: bool = True,
    personality: PersonalityProfile | None = None,
) -> PersonalizedContext:
    """Build one bounded personalization block; never raises.

    ``viewer_state`` is the ``viewer`` payload returned by the existing memory
    service. Missing, partial or corrupt payloads fall back to a generic
    context so the Brain keeps working without inventing history.
    """
    speech_style = personality_speech_style(personality) if personality else None
    try:
        if not viewer_id or not isinstance(viewer_state, Mapping):
            return _blank_context(
                viewer_id,
                "no viewer memory recorded; generic context",
                speech_style=speech_style,
            )

        interaction_count = _safe_count(viewer_state.get("interaction_count"))
        level = _safe_level(viewer_state.get("relationship_level"))
        familiarity = _safe_unit(viewer_state.get("familiarity"))
        engagement = _safe_unit(viewer_state.get("engagement_score"))
        preferred_topics = _safe_items(
            viewer_state.get("preferred_topics") or viewer_state.get("recent_topics"),
            _MAX_TOPICS,
        )
        preferred_skills = _safe_items(viewer_state.get("preferred_skills"), _MAX_SKILLS)
        recent_topics = _safe_items(viewer_state.get("recent_topics"), 5)
        recent_topic = recent_topics[-1] if recent_topics else None
        last_interaction = viewer_state.get("last_interaction")
        if not isinstance(last_interaction, str):
            last_interaction = None

        if interaction_count <= 0:
            return _blank_context(
                viewer_id,
                "viewer memory empty; generic context",
                speech_style=speech_style,
            )

        degraded = bool(viewer_state.get("degraded"))
        continuity = _topic_continuity(message_text, intent, preferred_topics)
        greeting = _returning_greeting(level, interaction_count, intent, greeting_allowed)

        reasons = [f"{level} viewer with {interaction_count} recorded interactions"]
        if preferred_topics:
            reasons.append(f"prefers {', '.join(preferred_topics)}")
        if continuity:
            reasons.append(f"topic continuity with '{continuity}'")
        if greeting:
            reasons.append("returning-viewer greeting within social cooldown")
        if degraded:
            reasons.append("partial viewer memory; bounded fallback values used")

        return PersonalizedContext(
            viewer_id=viewer_id.strip()[:120],
            known=True,
            relationship_level=level,
            familiarity=familiarity,
            interaction_count=interaction_count,
            engagement_score=engagement,
            preferred_topics=preferred_topics,
            preferred_skills=preferred_skills,
            recent_topic=recent_topic,
            last_interaction=last_interaction,
            topic_continuity=continuity,
            greeting=greeting,
            speech_style=speech_style,
            reason="; ".join(reasons)[:_MAX_REASON_CHARS],
            degraded=degraded,
        )
    except Exception:
        return _blank_context(
            viewer_id,
            "personalization unavailable; generic context",
            degraded=True,
            speech_style=speech_style,
        )


def skill_preference_bias(
    skill_id: str,
    *,
    skill_category: str | None = None,
    preferred_skills: Iterable[str] | None = None,
    personality: PersonalityProfile | None = None,
) -> tuple[float, str | None]:
    """Bounded advisory nudge for one skill candidate; never a hard filter.

    Two independent signals are combined and clamped:

    * repeated viewer evidence (the skill id itself, or a skill id the viewer
      has preferred before appearing inside this candidate's name), and
    * the stable personality preferring a skill category.

    The result only reorders advisory scores. Skill conditions, cooldowns,
    allowlists and the SkillExecutor stay authoritative.
    """
    normalized = (skill_id or "").strip().lower()
    if not normalized:
        return 0.0, None

    bias = 0.0
    reasons: list[str] = []

    preferred = {
        item.strip().lower()
        for item in (preferred_skills or ())
        if isinstance(item, str) and item.strip()
    }
    if preferred:
        if normalized in preferred:
            bias += 0.10
            reasons.append("viewer preference evidence")
        elif any(item in normalized or normalized in item for item in preferred):
            bias += 0.06
            reasons.append("related viewer preference evidence")

    if personality is not None and isinstance(skill_category, str):
        category = skill_category.strip().upper()
        if category == "MOVEMENT" and max(personality.playfulness, personality.energy) >= 0.6:
            bias += 0.03
            reasons.append("playful/energetic personality")
        elif category == "SOCIAL" and max(personality.warmth, personality.talkativeness) >= 0.6:
            bias += 0.03
            reasons.append("warm/sociable personality")
        elif category == "INTERACTION" and personality.curiosity >= 0.6:
            bias += 0.03
            reasons.append("curious personality")

    bounded = max(-MAX_SKILL_PREFERENCE_BIAS, min(MAX_SKILL_PREFERENCE_BIAS, bias))
    return round(bounded, 4), ("; ".join(reasons) or None)


def _topic_continuity(
    message_text: str | None,
    intent: str | None,
    preferred_topics: tuple[str, ...],
) -> str | None:
    """Only continue a topic with both current-message overlap and evidence.

    An unrelated message, a low-confidence topic (one-off mention) or a
    greeting/farewell never triggers continuity.
    """
    if not preferred_topics:
        return None
    if isinstance(intent, str) and intent.strip().upper() in {"GREETING", "FAREWELL"}:
        return None
    terms = _keywords(message_text)
    if not terms:
        return None
    for topic in preferred_topics:
        topic_terms = _keywords(topic)
        if topic_terms and topic_terms.intersection(terms):
            return topic
    return None


def _returning_greeting(
    level: str,
    interaction_count: int,
    intent: str | None,
    greeting_allowed: bool,
) -> str | None:
    """Bounded welcoming line for an established viewer, never per message."""
    if not greeting_allowed or interaction_count < 1:
        return None
    if not isinstance(intent, str) or intent.strip().upper() != "GREETING":
        return None
    if level in {"REGULAR", "TRUSTED"}:
        return "Welcome back! It's good to see you again."
    return "Hello again! Good to see you."
