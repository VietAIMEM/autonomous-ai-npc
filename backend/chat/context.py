"""Bounded short-term conversation context and PHASE 26A continuity.

Two bounded things live here and nothing else:

* the Phase 05 turn window used to build the dialogue prompt, and
* PHASE 26A continuity scalars for the *current* conversation (active viewer,
  topic, previous intent/response, pending question/action, turn count and a
  bounded conversation status).

Nothing in this module is persisted. Long-lived viewer information keeps living
in the existing memory subsystem, and the Phase 08 social context stays the
authoritative conversation state machine. This is a small read-model the chat
pipeline and diagnostics share so several viewer messages read as one
conversation instead of isolated requests.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from .models import ChatIntent, ChatMessage, DialogueResponse, format_utc


#: PHASE 26A: bounded conversation status. Values mirror the existing Phase 08
#: ``social.models.ConversationState`` vocabulary on purpose: the social
#: context remains the single state machine, and this field only reports the
#: turn-taking phase the chat pipeline is in.
PHASE_IDLE = "IDLE"
PHASE_GREETING = "GREETING"
PHASE_CONVERSING = "CONVERSING"
PHASE_WAITING = "WAITING"
PHASE_CHANGING_TOPIC = "CHANGING_TOPIC"
PHASE_ENDING = "ENDING"

#: Response phase for the in-flight turn.
RESPONSE_IDLE = "IDLE"
RESPONSE_RESPONDING = "RESPONDING"
RESPONSE_COMPLETED = "COMPLETED"
RESPONSE_FALLBACK = "FALLBACK"
RESPONSE_DISCARDED = "DISCARDED"

_MAX_RESPONSE_CHARS = 240
_MAX_QUESTION_CHARS = 160
_MAX_ACTION_CHARS = 80

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
    "center": "center",
    "floor": "floor",
    "wall": "wall",
    "phone": "phone",
}
#: Small keyword fallback for turns with no structured target ("do you like
#: dancing?"). Deliberately tiny: a topic is a hint, never a taxonomy.
_TOPIC_KEYWORDS = (
    ("camera", "camera"),
    ("webcam", "camera"),
    ("livestream", "stream"),
    ("stream", "stream"),
    ("danc", "dance"),
    ("music", "music"),
    ("song", "music"),
    ("sing", "music"),
    ("room", "room"),
)
#: PHASE 26A: topics that only mark the social frame (a greeting, a sign-off, a
#: reaction) instead of a subject. A question that explicitly names the current
#: activity or destination starts its own topic rather than inheriting one of
#: these; the intent layer already refuses to anchor continuations on them.
_META_TOPICS = frozenset(
    {"greeting", "farewell", "social_reaction", "general_conversation"}
)
_SHORT_FOLLOW_UP_WORDS = 8

_QUESTION_SENTENCE = re.compile(r"[^.!?]*\?")


def _inherit_topic(previous_topic: str | None, fallback: str) -> str:
    """Keep a substantive active topic; drop meta frames like a greeting."""
    if previous_topic and previous_topic not in _META_TOPICS:
        return previous_topic
    return fallback


def infer_topic(message: ChatMessage, previous_topic: str | None = None) -> str | None:
    """Lightweight current-topic hint for one viewer turn.

    Bounded and deterministic: structured target first, then a handful of
    keyword anchors, then the active topic for short continuations. The social
    layer owns its own (larger) topic vocabulary; this one only has to keep the
    chat pipeline's continuity readable.
    """
    intent = message.intent
    target = (message.target or "").casefold()
    text = message.normalized_text.casefold()

    if intent == ChatIntent.GREETING:
        return "greeting"
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
    for keyword, topic in _TOPIC_KEYWORDS:
        if keyword in text:
            return topic
    if intent in {
        ChatIntent.QUESTION,
        ChatIntent.FOLLOW_UP,
        ChatIntent.TOPIC_CONTINUATION,
    } and previous_topic and len(text.split()) <= _SHORT_FOLLOW_UP_WORDS:
        return previous_topic
    if intent in {ChatIntent.COMMENT, ChatIntent.ACKNOWLEDGEMENT}:
        return previous_topic or "social_reaction"
    return previous_topic or target or "general_conversation"


@dataclass(frozen=True)
class ConversationTurn:
    speaker: str
    viewer_id: str
    text: str


class ConversationContext:
    def __init__(self, max_turns: int = 8, *, timeout_seconds: float = 180.0) -> None:
        if max_turns <= 0:
            raise ValueError("max_turns must be greater than zero")
        self.max_turns = max_turns
        #: A conversation that goes quiet for this long expires back to a
        #: coherent listening state. Bounded so nothing stays active forever.
        self.timeout_seconds = max(0.0, float(timeout_seconds))
        self._turns: list[ConversationTurn] = []
        self._reset_continuity()

    def add_exchange(self, message: ChatMessage, response: DialogueResponse) -> None:
        self._turns.append(ConversationTurn(message.viewer_id, message.viewer_id, message.raw_text))
        self._turns.append(ConversationTurn("Lacrimosa", message.viewer_id, response.speech))
        if len(self._turns) > self.max_turns:
            del self._turns[:-self.max_turns]

    def snapshot(self) -> list[ConversationTurn]:
        return list(self._turns)

    def clear(self, now: datetime | None = None) -> None:
        self._turns.clear()
        self._reset_continuity()

    # ------------------------------------------------------------ continuity

    def observe_message(
        self,
        message: ChatMessage,
        *,
        topic: str | None = None,
        now: datetime | None = None,
    ) -> None:
        """Record an attended viewer turn before the response is generated."""
        moment = self._moment(message.timestamp, now)
        self.expire_if_needed(moment)
        previous_topic = self.current_topic
        intent = message.intent
        resolved_topic = topic or infer_topic(message, previous_topic)
        switching = bool(
            resolved_topic and previous_topic and resolved_topic != previous_topic
        )

        self.active_viewer = message.viewer_id
        self.current_topic = resolved_topic
        self.last_interaction_time = moment
        self.response_status = RESPONSE_RESPONDING
        if intent == ChatIntent.CONVERSATION_END:
            self.conversation_status = PHASE_ENDING
        elif intent == ChatIntent.GREETING:
            self.conversation_status = PHASE_GREETING
        elif switching and self.conversation_status in {
            PHASE_GREETING,
            PHASE_CONVERSING,
            PHASE_WAITING,
        }:
            self.conversation_status = PHASE_CHANGING_TOPIC
        elif self.conversation_status in {PHASE_IDLE, PHASE_ENDING}:
            self.conversation_status = PHASE_CONVERSING

        if intent in {
            ChatIntent.QUESTION,
            ChatIntent.FOLLOW_UP,
            ChatIntent.TOPIC_CONTINUATION,
        } and message.desired_state is None:
            self.pending_question = _clip(message.raw_text, _MAX_QUESTION_CHARS)

        # PHASE 26A: the structured request arms the pending action before its
        # response exists, so a bare continuation that arrives while this turn
        # is still being answered ("Stand in front of the camera." -> "Closer.")
        # already has the PHASE 25 spatial context to attach to. The response
        # below can still refine it with the provider's own requested_action.
        self._arm_pending_action(message)

    def begin_response(
        self,
        message: ChatMessage,
        *,
        now: datetime | None = None,
    ) -> None:
        """Mark the turn as actively being answered (turn-taking handoff)."""
        moment = self._moment(message.timestamp, now)
        self.expire_if_needed(moment)
        self.response_status = RESPONSE_RESPONDING
        if self.conversation_status == PHASE_GREETING:
            self.conversation_status = PHASE_CONVERSING

    def observe_response(
        self,
        message: ChatMessage,
        response: DialogueResponse,
        *,
        topic: str | None = None,
        fallback_reason: str | None = None,
        now: datetime | None = None,
    ) -> None:
        """Close the turn: store the exchange and update continuity."""
        moment = self._moment(message.timestamp, now)
        self.add_exchange(message, response)
        self.turn_count += 1
        self.active_viewer = message.viewer_id
        self.current_topic = topic or infer_topic(message, self.current_topic)
        self.last_interaction_time = moment
        self.previous_intent = message.intent
        self.previous_response = _clip(response.speech, _MAX_RESPONSE_CHARS)
        self.response_status = RESPONSE_FALLBACK if fallback_reason else RESPONSE_COMPLETED
        self.pending_question = _question_in(response.speech)
        self._set_pending_action(message, response)

        if message.intent == ChatIntent.CONVERSATION_END:
            self.conversation_status = PHASE_ENDING
        elif self.conversation_status in {
            PHASE_IDLE,
            PHASE_GREETING,
            PHASE_CHANGING_TOPIC,
        }:
            self.conversation_status = PHASE_CONVERSING

    def mark_discarded(
        self,
        reason: str = "discarded",
        *,
        now: datetime | None = None,
    ) -> None:
        """A queued turn was dropped without a response; never stay waiting."""
        self.response_status = RESPONSE_DISCARDED
        if self.conversation_status not in {PHASE_IDLE, PHASE_ENDING}:
            self.conversation_status = PHASE_WAITING

    def recover(
        self,
        reason: str,
        *,
        now: datetime | None = None,
    ) -> None:
        """Reconcile continuity after success/failure/timeout/cancel/replan."""
        token = (reason or "").strip().casefold()
        if token in {"timeout", "timed_out", "expired"}:
            self.clear()
            return
        if token in {
            "cancel",
            "cancelled",
            "canceled",
            "replan",
            "replanning",
            "failure",
            "failed",
            "error",
            "action_failed",
            "action_rejected",
            "skill_timeout",
            "interrupted_by_viewer",
        }:
            if self.response_status == RESPONSE_RESPONDING:
                self.response_status = RESPONSE_FALLBACK
            # A cancelled/failed physical request never became a fact, so a
            # later bare continuation cannot silently re-target it.
            self.pending_action = None
            self._pending_action_target = None
            self._pending_action_desired = None
            self._pending_action_movement = False
            if self.conversation_status in {PHASE_GREETING, PHASE_CHANGING_TOPIC}:
                self.conversation_status = PHASE_CONVERSING
            elif self.conversation_status == PHASE_IDLE and self._turns:
                self.conversation_status = PHASE_WAITING
            return
        # success / completed / anything else: never leave a half-open turn.
        if self.response_status == RESPONSE_RESPONDING:
            self.response_status = RESPONSE_COMPLETED
        if self.conversation_status in {PHASE_GREETING, PHASE_CHANGING_TOPIC}:
            self.conversation_status = PHASE_CONVERSING

    def expire_if_needed(self, now: datetime | None = None) -> bool:
        """Drop a conversation that has been quiet longer than the bound."""
        if self.conversation_status == PHASE_IDLE:
            return False
        last = self.last_interaction_time
        if last is None:
            return False
        moment = now or _now()
        try:
            elapsed = (moment - last).total_seconds()
        except TypeError:
            # Mixed naive/aware timestamps: never guess, never crash.
            return False
        if elapsed < self.timeout_seconds:
            return False
        self.clear()
        return True

    @property
    def pending_action_target(self) -> Optional[str]:
        return self._pending_action_target

    @property
    def pending_action_desired(self) -> Optional[str]:
        return self._pending_action_desired

    @property
    def pending_action_is_movement(self) -> bool:
        return self._pending_action_movement

    def has_active_conversation(self) -> bool:
        return self.conversation_status not in {PHASE_IDLE, PHASE_ENDING}

    def to_state(self, now: datetime | None = None) -> dict[str, object]:
        self.expire_if_needed(now)
        return {
            "conversation_status": self.conversation_status,
            "active_viewer": self.active_viewer,
            "current_topic": self.current_topic,
            "previous_intent": self.previous_intent.value if self.previous_intent else None,
            "previous_response": self.previous_response,
            "pending_question": self.pending_question,
            "pending_action": self.pending_action,
            "last_interaction_at": (
                format_utc(self.last_interaction_time)
                if self.last_interaction_time is not None
                else None
            ),
            "turn_count": self.turn_count,
            "response_status": self.response_status,
            "conversation_turns": len(self._turns),
        }

    # ------------------------------------------------------------- internals

    def _reset_continuity(self) -> None:
        self.active_viewer: Optional[str] = None
        self.current_topic: Optional[str] = None
        self.previous_intent: Optional[ChatIntent] = None
        self.previous_response: Optional[str] = None
        self.pending_question: Optional[str] = None
        self.pending_action: Optional[str] = None
        self.last_interaction_time: Optional[datetime] = None
        self.turn_count = 0
        self.conversation_status = PHASE_IDLE
        self.response_status = RESPONSE_IDLE
        self._pending_action_target: Optional[str] = None
        self._pending_action_desired: Optional[str] = None
        self._pending_action_movement = False

    def _set_pending_action(
        self,
        message: ChatMessage,
        response: DialogueResponse,
    ) -> None:
        requested = (response.requested_action or "").strip()
        # The viewer's own structured request counts as evidence too: if the
        # provider echoed no action for a movement turn, the conversation is
        # still *about* that movement and a bare "Closer." must resolve to it.
        viewer_actionable = message.intent in {
            ChatIntent.REQUEST_ACTION,
            ChatIntent.REQUEST_MOVEMENT,
        } and bool(message.target)
        if not requested and not viewer_actionable:
            self.pending_action = None
            self._pending_action_target = None
            self._pending_action_desired = None
            self._pending_action_movement = False
            return
        target = message.target
        movement = message.intent == ChatIntent.REQUEST_MOVEMENT
        if requested.startswith("move_to:"):
            target = requested.split(":", 1)[1] or target
            movement = True
        self.pending_action = _clip(
            requested or f"{message.intent.value.casefold()}:{message.target}",
            _MAX_ACTION_CHARS,
        )
        self._pending_action_target = target
        self._pending_action_desired = message.desired_state
        self._pending_action_movement = movement

    def _arm_pending_action(self, message: ChatMessage) -> None:
        """Pending-action evidence from the viewer's own structured request.

        Bounded to the same two intents and the same rules ``_set_pending_action``
        already applies once a response exists; this only makes the evidence
        available one step earlier, while the turn is queued/being answered.
        """
        if message.intent not in {
            ChatIntent.REQUEST_ACTION,
            ChatIntent.REQUEST_MOVEMENT,
        }:
            return
        if not message.target:
            return
        self.pending_action = _clip(
            f"{message.intent.value.casefold()}:{message.target}",
            _MAX_ACTION_CHARS,
        )
        self._pending_action_target = message.target
        self._pending_action_desired = message.desired_state
        self._pending_action_movement = message.intent == ChatIntent.REQUEST_MOVEMENT

    @staticmethod
    def _moment(
        timestamp: Optional[datetime],
        now: Optional[datetime],
    ) -> datetime:
        if now is not None:
            return now
        if timestamp is not None:
            return timestamp
        return _now()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _clip(value: object, limit: int) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = " ".join(value.split())
    if not text:
        return None
    return text[:limit]


def _question_in(speech: object) -> Optional[str]:
    """The last question Lacrimosa left open, if any (bounded)."""
    if not isinstance(speech, str):
        return None
    matches = _QUESTION_SENTENCE.findall(speech)
    if not matches:
        return None
    return _clip(matches[-1], _MAX_QUESTION_CHARS)
