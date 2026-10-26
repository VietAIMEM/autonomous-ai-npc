"""PHASE 26B: bounded deterministic expression coordination for one character.

One reasoning result already exists per turn (the chat response and the plan).
This module only maps that result plus already-derived context -- intent,
behavior state, personality, viewer relationship, current activity -- onto a
small, inspectable expression decision:

    tone + reaction intensity + optional existing gesture

It never calls a provider, never runs a second emotion model and never plans
execution. Gesture availability, cooldowns and world conditions are still owned
by the existing SkillLibrary / SkillExecutor; the policy only decides whether
asking for a gesture is appropriate at all.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional


class ReactionIntensity(str, Enum):
    """Bounded reaction strength; deliberately only three levels."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


_INTENSITY_ORDER = {
    ReactionIntensity.LOW: 0,
    ReactionIntensity.MEDIUM: 1,
    ReactionIntensity.HIGH: 2,
}

#: Existing expressive actions that count as gestures. No new motion is
#: introduced by this phase; these are the already-registered ones.
GESTURE_WAVE = "wave"
GESTURE_DANCE = "dance"
GESTURE_ACTIONS = (GESTURE_WAVE, GESTURE_DANCE)

_WARM_INTENTS = frozenset({"GREETING", "COMMENT", "ACKNOWLEDGEMENT"})
#: Intent/state pairs that may reuse an already-registered expressive motion.
#: Greeting and farewell are the two bounded social moments where a wave is the
#: existing, documented mapping; everything else stays speech-only.
_GREETING_INTENT = "GREETING"
_FAREWELL_INTENTS = frozenset({"FAREWELL", "CONVERSATION_END"})
_WAVE_STATES = frozenset({"HAPPY", "ENGAGED", "CURIOUS", "CALM"})
_LOW_BASE_INTENTS = frozenset(
    {
        "QUESTION",
        "FOLLOW_UP",
        "TOPIC_CONTINUATION",
        "ACKNOWLEDGEMENT",
        "FAREWELL",
        "CONVERSATION_END",
        "REQUEST_ACTION",
        "REQUEST_MOVEMENT",
    }
)
_WARM_RELATIONSHIPS = frozenset({"FAMILIAR", "REGULAR", "TRUSTED"})


@dataclass(frozen=True)
class ExpressionConfig:
    """Bounded knobs. Defaults match a calm, warm livestream character."""

    #: A repeated identical gesture inside this window is refused so the
    #: character cannot spam the same animation on consecutive interactions.
    #: This is longer than the builtin skill cooldowns on purpose: those guard
    #: execution, this guards visible repetition. It also has to outlast the
    #: real viewer-turn cadence: a live turn takes tens of seconds end to end,
    #: so a shorter window would let two consecutive greetings wave twice.
    gesture_suppression_seconds: float = 120.0
    max_recent_gestures: int = 8
    speech_hint_limit: int = 160

    def __post_init__(self) -> None:
        if self.gesture_suppression_seconds < 0:
            raise ValueError("gesture_suppression_seconds cannot be negative")
        if self.max_recent_gestures <= 0:
            raise ValueError("max_recent_gestures must be greater than zero")
        if self.speech_hint_limit <= 0:
            raise ValueError("speech_hint_limit must be greater than zero")


@dataclass(frozen=True)
class ExpressionDecision:
    """One bounded, explainable expression decision."""

    tone: str
    reaction_intensity: str
    gesture: Optional[str]
    gesture_reason: str
    intent: Optional[str]
    behavior_state: str
    attention_required: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "tone": self.tone,
            "reaction_intensity": self.reaction_intensity,
            "gesture": self.gesture,
            "gesture_reason": self.gesture_reason,
            "intent": self.intent,
            "behavior_state": self.behavior_state,
            "attention_required": self.attention_required,
            "reason": self.reason,
        }

    def speech_hint(self, speech_style: str | None = None, *, limit: int = 160) -> str:
        """Compact behavioral hint for the single existing response prompt.

        Only bounded, already-derived text is forwarded; the raw viewer message
        and any transcript content never appear here.
        """
        parts = [f"tone {self.tone}", f"reaction {self.reaction_intensity.lower()}"]
        detail = self.reason or self.behavior_state.lower()
        parts.append(f"because {detail}")
        if speech_style:
            parts.append(f"style {speech_style}")
        return "; ".join(parts)[: max(1, limit)]


class ExpressionPolicy:
    """Deterministic intent/behavior/personality -> expression mapping."""

    def __init__(
        self,
        config: ExpressionConfig | None = None,
        clock: Any = None,
    ) -> None:
        self.config = config or ExpressionConfig()
        self._clock = clock
        self._recent: deque[tuple[str, datetime]] = deque(
            maxlen=self.config.max_recent_gestures
        )

    # ------------------------------------------------------------ inspection
    def recent_gestures(self) -> list[dict[str, Any]]:
        return [
            {"gesture": gesture, "timestamp": timestamp.isoformat()}
            for gesture, timestamp in self._recent
        ]

    @property
    def last_gesture(self) -> Optional[str]:
        return self._recent[-1][0] if self._recent else None

    def snapshot(self) -> dict[str, Any]:
        return {
            "recent_gestures": [
                {"gesture": gesture, "timestamp": timestamp.isoformat()}
                for gesture, timestamp in self._recent
            ]
        }

    def restore(self, payload: dict[str, Any] | None) -> None:
        if not isinstance(payload, dict):
            return
        self._recent.clear()
        for item in payload.get("recent_gestures") or ():
            if not isinstance(item, dict):
                continue
            gesture = item.get("gesture")
            timestamp = _parse_datetime(item.get("timestamp"))
            if isinstance(gesture, str) and gesture and timestamp is not None:
                self._recent.append((gesture, timestamp))

    def note_gesture(self, gesture: str | None, now: datetime | None = None) -> None:
        """Record that a gesture was accepted into a dispatched plan."""
        if not isinstance(gesture, str) or not gesture.strip():
            return
        self._recent.append((gesture.strip().lower(), now or self._now()))

    def suppressed(self, gesture: str, now: datetime | None = None) -> bool:
        """Whether this gesture repeats inside the bounded suppression window."""
        if not isinstance(gesture, str) or not gesture:
            return False
        current = now or self._now()
        identity = gesture.strip().lower()
        for recent, timestamp in reversed(self._recent):
            if recent != identity:
                continue
            elapsed = (current - timestamp).total_seconds()
            if 0 <= elapsed < self.config.gesture_suppression_seconds:
                return True
        return False

    # -------------------------------------------------------------- decision
    def decide(
        self,
        *,
        intent: str | None,
        behavior_state: str | None,
        behavior_reason: str | None = None,
        personality: Any | None = None,
        personalization: Any | None = None,
        current_action: str | None = None,
        attention_allowed: bool = False,
        availability: Callable[[str], bool] | None = None,
        world: Any | None = None,
        now: datetime | None = None,
    ) -> ExpressionDecision:
        current = now or self._now()
        intent_value = _upper(intent)
        state = _upper(behavior_state) or "CALM"
        tone = self._tone(state, intent_value, personalization)
        intensity = self._intensity(
            state,
            intent_value,
            personality,
            personalization,
            behavior_reason=_upper(behavior_reason),
        )
        gesture, gesture_reason = self._gesture(
            intent_value,
            state,
            current_action=current_action,
            availability=availability,
            world=world,
            now=current,
        )
        return ExpressionDecision(
            tone=tone,
            reaction_intensity=intensity.value,
            gesture=gesture,
            gesture_reason=gesture_reason,
            intent=intent_value,
            behavior_state=state,
            attention_required=self._attention_required(
                intent_value, state, attention_allowed
            ),
            reason=self._reason(intent_value, state),
        )

    # ------------------------------------------------------------- internals
    @staticmethod
    def _tone(state: str, intent: str | None, personalization: Any | None) -> str:
        if intent in {"FAREWELL", "CONVERSATION_END"}:
            return "warm"
        base = {
            "HAPPY": "warm",
            "ENGAGED": "warm",
            "CURIOUS": "interested",
            "SURPRISED": "reactive",
            "FOCUSED": "focused",
            "TIRED": "quiet",
        }.get(state, "neutral")
        if base == "neutral" and _relationship_known(personalization):
            # A known viewer still gets a neutral-but-warm register, never a
            # second emotional state machine.
            return "warm"
        return base

    @staticmethod
    def _intensity(
        state: str,
        intent: str | None,
        personality: Any | None,
        personalization: Any | None,
        *,
        behavior_reason: str | None = None,
    ) -> ReactionIntensity:
        # A live viewer reaction is a discrete event owned by the existing
        # behavior manager, not background mood: a fresh positive/negative
        # reaction keeps at least a MEDIUM reaction instead of being flattened
        # by the calmness damper that bounds everyday states.
        discrete_reaction = behavior_reason in {
            "POSITIVE_VIEWER_REACTION",
            "NEGATIVE_VIEWER_REACTION",
        }
        base = 1 if intent in {"GREETING", "COMMENT"} else 0
        if state == "HAPPY":
            base += 1 if base == 0 else 0
        elif state == "SURPRISED":
            base += 2 if base == 0 else 1
        elif state == "ENGAGED":
            base += 1 if base == 0 else 0
        elif state in {"FOCUSED", "TIRED"}:
            base -= 1

        playfulness = _trait(personality, "playfulness")
        energy = _trait(personality, "energy")
        calmness = _trait(personality, "calmness")
        if intent in _WARM_INTENTS and playfulness >= 0.6:
            base += 1
        if intent in {"GREETING", "COMMENT"} and energy >= 0.75:
            base += 1
        # A calm character keeps everyday reactions bounded; an unexpected
        # event may still startle her.
        if calmness >= 0.85 and state != "SURPRISED" and not discrete_reaction:
            base -= 1
        if discrete_reaction and state in {"HAPPY", "SURPRISED"}:
            base = max(base, 1)
        if (
            base <= 0
            and intent in {"GREETING", "COMMENT"}
            and _returning(personalization)
            and _relationship_level(personalization) in _WARM_RELATIONSHIPS
        ):
            base += 1

        return _clamp(base)

    def _gesture(
        self,
        intent: str | None,
        state: str,
        *,
        current_action: str | None,
        availability: Callable[[str], bool] | None,
        world: Any | None,
        now: datetime,
    ) -> tuple[Optional[str], str]:
        candidate: Optional[str] = None
        reason = "no_compatible_gesture"
        if state == "HAPPY" and intent in {
            "COMMENT",
            "QUESTION",
            "FOLLOW_UP",
            "ACKNOWLEDGEMENT",
            "TOPIC_CONTINUATION",
            "GREETING",
            "UNKNOWN",
            None,
        }:
            candidate, reason = GESTURE_WAVE, "positive_reaction"
        elif intent in _FAREWELL_INTENTS and state in _WAVE_STATES:
            candidate, reason = GESTURE_WAVE, "farewell"
        elif intent == _GREETING_INTENT and state in _WAVE_STATES:
            # PHASE 26B: greeting -> LOOK_AT -> WAVE -> SPEAK is the documented
            # sequence. The behavior state stays CURIOUS (Phase 18 owns that
            # transition); this layer only decides whether the greeting may
            # reuse the registered wave.
            candidate, reason = GESTURE_WAVE, "greeting"
        elif state == "ENGAGED" and intent is None:
            candidate, reason = GESTURE_WAVE, "engagement"

        if candidate is None:
            return None, reason
        if _busy_action(current_action):
            # Never stack a gesture onto an action Unity is still running.
            return None, "character_busy"
        if self.suppressed(candidate, now):
            return None, "recent_gesture_suppressed"
        if availability is not None and not _available(availability, candidate, world):
            return None, "gesture_unavailable"
        return candidate, reason

    @staticmethod
    def _attention_required(
        intent: str | None, state: str, attention_allowed: bool
    ) -> bool:
        """Whether this turn should be visually engaged with the camera.

        The window itself is owned by the existing behavior manager; this only
        reports what the current turn would like, so Phase 25 spatial planning
        stays authoritative for actually turning/looking.
        """
        if not attention_allowed:
            return False
        return state == "CURIOUS" or intent in {
            "GREETING",
            "QUESTION",
            "FOLLOW_UP",
            "COMMENT",
            "FAREWELL",
        }

    @staticmethod
    def _reason(intent: str | None, state: str) -> str:
        if intent is None:
            return f"{state.lower()}_idle"
        return f"{intent.lower()}_while_{state.lower()}"

    def _now(self) -> datetime:
        if self._clock is None:
            from ..chat.models import utc_now

            return utc_now()
        value = self._clock()
        if not isinstance(value, datetime):
            raise TypeError("expression_clock_must_return_datetime")
        return value


def _clamp(value: int) -> ReactionIntensity:
    bounded = max(0, min(2, value))
    for intensity, order in _INTENSITY_ORDER.items():
        if order == bounded:
            return intensity
    return ReactionIntensity.LOW  # pragma: no cover - exhaustive


def _available(
    availability: Callable[[str], bool], gesture: str, world: Any | None
) -> bool:
    try:
        return bool(availability(gesture))
    except Exception:
        # Availability is advisory here; the plan validator and SkillExecutor
        # still own the real gate, so an unknown answer simply skips the
        # optional gesture instead of failing the turn.
        return False


def _busy_action(current_action: str | None) -> bool:
    value = (current_action or "").strip().casefold()
    return value not in {"", "idle", "standing", "none", "stopped"}


def _upper(value: Any) -> Optional[str]:
    if value is None:
        return None
    if hasattr(value, "value"):
        value = value.value
    text = str(value).strip()
    return text.upper() if text else None


def _field(source: Any, name: str) -> Any:
    if source is None:
        return None
    if isinstance(source, dict):
        return source.get(name)
    return getattr(source, name, None)


def _trait(personality: Any, name: str) -> float:
    value = _field(personality, name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)


def _returning(personalization: Any) -> bool:
    return bool(_field(personalization, "returning"))


def _relationship_level(personalization: Any) -> Optional[str]:
    return _upper(_field(personalization, "relationship_level"))


def _relationship_known(personalization: Any) -> bool:
    level = _relationship_level(personalization)
    return bool(level) and level != "NEW"


def _parse_datetime(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


__all__ = [
    "ExpressionConfig",
    "ExpressionDecision",
    "ExpressionPolicy",
    "GESTURE_ACTIONS",
    "GESTURE_DANCE",
    "GESTURE_WAVE",
    "ReactionIntensity",
]
