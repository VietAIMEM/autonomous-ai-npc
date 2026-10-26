"""Deterministic topic, continuity, and reaction helpers."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from ..chat.models import ChatIntent, ChatMessage
from .models import SocialConfig


_ACTION_TOPICS = {
    "dance": "dance",
    "wave": "waving",
    "look": "looking",
    "camera": "camera",
    "sit": "posture",
    "stand": "posture",
    "run": "running",
}
_LOCATION_TOPICS = {
    "table": "table",
    "bench": "bench",
    "center": "room_navigation",
    "floor": "room_navigation",
    "wall": "room_navigation",
    "phone": "phone",
}
_POSITIVE_WORDS = {"nice", "cool", "great", "cute", "amazing", "love", "thanks", "thank"}
_NEGATIVE_WORDS = {"bad", "awful", "terrible", "hate", "wrong"}
# PHASE 26A: the same tiny everyday-topic anchors the chat layer uses, so the
# wired resolver keeps readable topics ("dance", "camera", ...) for ordinary
# chat instead of collapsing them to "general_conversation". "room" is left
# out on purpose: here it already resolves to the richer "room_exploration".
_TOPIC_KEYWORDS = (
    ("camera", "camera"),
    ("webcam", "camera"),
    ("livestream", "stream"),
    ("stream", "stream"),
    ("danc", "dance"),
    ("music", "music"),
    ("song", "music"),
    ("sing", "music"),
)
# PHASE 26A: topics that only mark the social frame (a greeting, a sign-off, a
# reaction) instead of a subject. "What are you doing?" after a greeting starts
# a real activity topic instead of reporting the greeting back as the subject.
_META_TOPICS = frozenset(
    {"greeting", "farewell", "social_reaction", "general_conversation"}
)


def _inherit_topic(previous_topic: str | None, fallback: str) -> str:
    """Keep a substantive active topic; drop meta frames like a greeting."""
    if previous_topic and previous_topic not in _META_TOPICS:
        return previous_topic
    return fallback


_TOKEN_EDGES = ".,!?;:\"'()[]{}*~-"


def _reaction_words(text: str) -> set[str]:
    """Everyday punctuation must not hide a word ("great!" is still praise)."""
    return {
        token
        for raw in text.casefold().split()
        if (token := raw.strip(_TOKEN_EDGES))
    }


class ConversationManager:
    def __init__(self, config: SocialConfig | None = None) -> None:
        self.config = config or SocialConfig()

    def infer_topic(
        self,
        message: ChatMessage,
        previous_topic: str | None = None,
    ) -> Optional[str]:
        intent = message.intent
        target = (message.target or "").casefold()
        text = message.normalized_text.casefold()

        if intent == ChatIntent.GREETING:
            return "greeting"
        # PHASE 26A: an explicit end of conversation closes the topic exactly
        # like a farewell, so the wired social resolver and the chat layer's
        # own bounded fallback report the same active topic.
        if intent in {ChatIntent.FAREWELL, ChatIntent.CONVERSATION_END}:
            return "farewell"
        if target in _ACTION_TOPICS:
            return _ACTION_TOPICS[target]
        if target in _LOCATION_TOPICS:
            return _LOCATION_TOPICS[target]

        if target == "current_activity":
            return _inherit_topic(previous_topic, "current_activity")
        if target == "destination":
            return _inherit_topic(previous_topic, "destination")
        if "room" in text or "around" in text or "explore" in text:
            return "room_exploration"

        for keyword, topic in _TOPIC_KEYWORDS:
            if keyword in text:
                return topic

        # A short follow-up with no new topic anchors belongs to the active topic.
        if intent == ChatIntent.QUESTION and previous_topic and len(text.split()) <= 8:
            return previous_topic
        if intent == ChatIntent.COMMENT:
            return "social_reaction"
        return previous_topic or target or "general_conversation"

    def is_topic_change(self, new_topic: str | None, previous_topic: str | None) -> bool:
        return bool(new_topic and previous_topic and new_topic != previous_topic)

    def is_topic_continuation(
        self,
        new_topic: str | None,
        previous_topic: str | None,
    ) -> bool:
        """Whether the turn stays inside the active conversational thread.

        Literal equality is the ordinary continuation. A move out of a meta
        frame ("greeting", "social_reaction") is the same conversation gaining
        a real subject, so attention must not score it like a cold start.
        """
        if not new_topic or not previous_topic:
            return False
        if new_topic == previous_topic:
            return True
        return previous_topic in _META_TOPICS

    def classify_reaction(self, message: ChatMessage) -> str:
        text = _reaction_words(message.normalized_text)
        if text & _NEGATIVE_WORDS:
            return "negative"
        if text & _POSITIVE_WORDS:
            return "positive"
        if message.intent in {ChatIntent.QUESTION, ChatIntent.REQUEST_ACTION, ChatIntent.REQUEST_MOVEMENT}:
            return "request_followup"
        return "neutral"

    def can_speak(
        self,
        now: datetime,
        last_speech_at: datetime | None,
    ) -> bool:
        if last_speech_at is None:
            return True
        return (now - last_speech_at).total_seconds() >= self.config.speech_cooldown_seconds
