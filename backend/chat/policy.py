"""PHASE 26A bounded response policy.

The LLM keeps writing the words. This module only decides *how much* and *what
kind* of reply the turn needs, from the existing signals (intent, bounded
conversation context, personality, viewer relationship). It never renders a
response and never talks to a provider by itself.

Everything here is bounded on purpose: six response classes, a handful of
acknowledgement hints, and a small recent-hint memory so the same
acknowledgement is not repeated every turn.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping, Optional

from .context import ConversationContext
from .models import ChatIntent, ChatMessage

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle
    from .dialogue import CharacterContext


RESPONSE_CLASS_SHORT_ACK = "SHORT_ACK"
RESPONSE_CLASS_SHORT_REPLY = "SHORT_REPLY"
RESPONSE_CLASS_NORMAL_REPLY = "NORMAL_REPLY"
RESPONSE_CLASS_ACTION_ACK = "ACTION_ACK"
RESPONSE_CLASS_FOLLOW_UP = "FOLLOW_UP_REPLY"
RESPONSE_CLASS_FAREWELL = "FAREWELL"

#: Word budgets per class. Small numbers on purpose: a livestream reply is
#: spoken, so a 60-word answer to "hi" is a failure, not a feature.
_BASE_BUDGETS = {
    RESPONSE_CLASS_SHORT_ACK: 6,
    RESPONSE_CLASS_SHORT_REPLY: 20,
    RESPONSE_CLASS_NORMAL_REPLY: 45,
    RESPONSE_CLASS_ACTION_ACK: 25,
    RESPONSE_CLASS_FOLLOW_UP: 24,
    RESPONSE_CLASS_FAREWELL: 14,
}
_MIN_WORDS = 4
_MAX_WORDS = 60

_CLASS_STYLE = {
    RESPONSE_CLASS_SHORT_ACK: "One short acknowledgement; add nothing else.",
    RESPONSE_CLASS_SHORT_REPLY: "One or two short sentences.",
    RESPONSE_CLASS_NORMAL_REPLY: "A concise answer; stay on the current topic.",
    RESPONSE_CLASS_ACTION_ACK: (
        "Acknowledge the request briefly; the physical action is handled by "
        "another system and must not be re-described in detail."
    ),
    RESPONSE_CLASS_FOLLOW_UP: (
        "Answer the follow-up directly and keep the previous topic; do not "
        "start a new subject."
    ),
    RESPONSE_CLASS_FAREWELL: "A brief, warm sign-off.",
}

#: Bounded acknowledgement hints. The LLM may phrase these naturally; the
#: rotation only stops the same word from being reused turn after turn.
_ACK_HINTS = ("Sure.", "Okay.", "Got it.", "Alright.", "Yeah.", "One moment.")
_ACK_HISTORY = 4

_COMPLEX_QUESTION_WORDS = 12
_WARM_RELATIONSHIPS = {"FRIEND", "CLOSE", "FAMILIAR", "TRUSTED"}
_SHORT_REPLY_CLASSES = {
    RESPONSE_CLASS_SHORT_REPLY,
    RESPONSE_CLASS_NORMAL_REPLY,
    RESPONSE_CLASS_FOLLOW_UP,
}


@dataclass(frozen=True)
class ResponsePolicy:
    """One bounded instruction set for the current turn."""

    response_class: str
    max_words: int
    constraint: str
    acknowledgement_hint: Optional[str] = None

    def to_dict(self) -> dict[str, object]:
        return {
            "response_class": self.response_class,
            "max_words": self.max_words,
            "acknowledgement_hint": self.acknowledgement_hint,
        }


class ResponsePolicyEngine:
    """Deterministic, bounded policy selection with a tiny ack history."""

    def __init__(self, history_size: int = _ACK_HISTORY) -> None:
        self.history_size = max(1, int(history_size))
        self._recent_acks: list[str] = []
        self._ack_cursor = -1

    def select(
        self,
        message: ChatMessage,
        context: ConversationContext | None = None,
        character: Optional["CharacterContext"] = None,
        personalization: "Mapping[str, object] | None" = None,
    ) -> ResponsePolicy:
        intent = message.intent
        response_class = self._classify(message, intent)
        budget = _BASE_BUDGETS[response_class]
        budget = self._adjust_budget(budget, response_class, character, personalization)

        hint: Optional[str] = None
        if response_class in {RESPONSE_CLASS_SHORT_ACK, RESPONSE_CLASS_ACTION_ACK}:
            hint = self._next_ack_hint()

        parts = [f"Reply in at most {budget} words.", _CLASS_STYLE[response_class]]
        if response_class == RESPONSE_CLASS_FOLLOW_UP and context is not None:
            topic = context.current_topic
            if topic:
                parts.append(f"Active topic: {topic}.")
            if context.previous_response:
                parts.append(
                    "Your previous line was: "
                    + _quote(context.previous_response)
                    + "."
                )
        if hint:
            parts.append(f"You may open with something like: {hint}")
        return ResponsePolicy(
            response_class=response_class,
            max_words=budget,
            constraint=" ".join(parts),
            acknowledgement_hint=hint,
        )

    def recent_hints(self) -> tuple[str, ...]:
        return tuple(self._recent_acks)

    # ------------------------------------------------------------- internals

    @staticmethod
    def _classify(message: ChatMessage, intent: ChatIntent) -> str:
        if intent in {ChatIntent.FAREWELL, ChatIntent.CONVERSATION_END}:
            return RESPONSE_CLASS_FAREWELL
        if intent == ChatIntent.ACKNOWLEDGEMENT:
            return RESPONSE_CLASS_SHORT_ACK
        if intent in {ChatIntent.REQUEST_ACTION, ChatIntent.REQUEST_MOVEMENT}:
            return RESPONSE_CLASS_ACTION_ACK
        if intent in {ChatIntent.FOLLOW_UP, ChatIntent.TOPIC_CONTINUATION}:
            return RESPONSE_CLASS_FOLLOW_UP
        if intent == ChatIntent.GREETING:
            return RESPONSE_CLASS_SHORT_REPLY
        if intent == ChatIntent.QUESTION:
            words = len(message.normalized_text.split())
            return (
                RESPONSE_CLASS_NORMAL_REPLY
                if words > _COMPLEX_QUESTION_WORDS
                else RESPONSE_CLASS_SHORT_REPLY
            )
        if intent == ChatIntent.COMMENT:
            return (
                RESPONSE_CLASS_SHORT_ACK
                if message.target == "noise"
                else RESPONSE_CLASS_SHORT_REPLY
            )
        return RESPONSE_CLASS_SHORT_REPLY

    def _adjust_budget(
        self,
        budget: int,
        response_class: str,
        character: Optional["CharacterContext"],
        personalization: "Mapping[str, object] | None",
    ) -> int:
        traits = {
            trait.casefold()
            for trait in (getattr(character, "personality", ()) or ())
            if isinstance(trait, str)
        }
        if (
            response_class in _SHORT_REPLY_CLASSES
            and "playful" in traits
        ):
            budget += 4
        if response_class == RESPONSE_CLASS_SHORT_ACK and (
            "calm" in traits or "gentle" in traits
        ):
            budget -= 1
        if response_class in _SHORT_REPLY_CLASSES and isinstance(
            personalization, Mapping
        ):
            level = str(personalization.get("relationship_level") or "").upper()
            if personalization.get("returning") and level in _WARM_RELATIONSHIPS:
                budget += 4
        return max(_MIN_WORDS, min(_MAX_WORDS, budget))

    def _next_ack_hint(self) -> str:
        # Round-robin rotation that skips anything inside the bounded recent
        # window, so the whole tiny set stays reachable instead of the first
        # few cycling forever. When the window already covers every hint the
        # rotation simply advances one step and restarts the cycle.
        recent = set(self._recent_acks[-self.history_size :])
        for _ in range(len(_ACK_HINTS)):
            self._ack_cursor = (self._ack_cursor + 1) % len(_ACK_HINTS)
            choice = _ACK_HINTS[self._ack_cursor]
            if choice not in recent:
                break
        else:
            self._ack_cursor = (self._ack_cursor + 1) % len(_ACK_HINTS)
            choice = _ACK_HINTS[self._ack_cursor]
        self._recent_acks.append(choice)
        if len(self._recent_acks) > self.history_size:
            del self._recent_acks[:-self.history_size]
        return choice


def _quote(value: str) -> str:
    text = " ".join(value.split())
    return f'"{text[:120]}"'
