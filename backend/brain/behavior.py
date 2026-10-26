"""Bounded advisory behavior state for the authoritative Brain loop."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Iterable, Optional

from ..chat.models import format_utc
from .models import ActionResult, ExecutionStatus, Goal, WorldState


class BehaviorState(str, Enum):
    CALM = "CALM"
    HAPPY = "HAPPY"
    CURIOUS = "CURIOUS"
    FOCUSED = "FOCUSED"
    SURPRISED = "SURPRISED"
    TIRED = "TIRED"
    ENGAGED = "ENGAGED"


@dataclass(frozen=True)
class BehaviorConfig:
    minimum_dwell_seconds: float = 10.0
    reaction_window_seconds: float = 30.0
    outcome_window_seconds: float = 30.0
    memory_score_threshold: float = 0.65
    engagement_threshold: float = 0.55
    tired_stream_seconds: float = 7200.0
    tired_engagement_threshold: float = 0.20
    proactive_cooldown_seconds: float = 30.0
    attention_cooldown_seconds: float = 15.0
    max_recent_outcomes: int = 8

    def __post_init__(self) -> None:
        positive = (
            self.minimum_dwell_seconds,
            self.reaction_window_seconds,
            self.outcome_window_seconds,
            self.tired_stream_seconds,
            self.proactive_cooldown_seconds,
            self.attention_cooldown_seconds,
        )
        if any(value < 0 for value in positive):
            raise ValueError("behavior windows and cooldowns cannot be negative")
        if not 0.0 <= self.memory_score_threshold <= 1.0:
            raise ValueError("memory_score_threshold must be between 0 and 1")
        if not 0.0 <= self.engagement_threshold <= 1.0:
            raise ValueError("engagement_threshold must be between 0 and 1")
        if not 0.0 <= self.tired_engagement_threshold <= 1.0:
            raise ValueError("tired_engagement_threshold must be between 0 and 1")
        if self.max_recent_outcomes <= 0:
            raise ValueError("max_recent_outcomes must be greater than zero")


@dataclass(frozen=True)
class BehaviorOutcome:
    action: str
    status: str
    success: bool
    reason: str
    timestamp: datetime
    viewer_response: Optional[str] = None
    skill_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "status": self.status,
            "success": self.success,
            "reason": self.reason,
            "timestamp": format_utc(self.timestamp),
            "viewer_response": self.viewer_response,
            "skill_id": self.skill_id,
        }


class BehaviorManager:
    """Derive one inspectable state from existing Brain context.

    The manager never plans or executes. It only records bounded outcomes and
    supplies an advisory context to the existing reasoner and SkillExecutor.
    """

    def __init__(
        self,
        config: BehaviorConfig | None = None,
        clock: Any = None,
    ) -> None:
        self.config = config or BehaviorConfig()
        self._clock = clock
        self.state = BehaviorState.CALM
        self.previous_state: Optional[BehaviorState] = None
        self.transition_reason = "initialized"
        self.transitioned_at: Optional[datetime] = None
        self.transition_tick = 0
        self.logical_tick = 0
        self.last_proactive_at: Optional[datetime] = None
        self.last_attention_at: Optional[datetime] = None
        self.proactive_reason: Optional[str] = None
        self.expression_count = 0
        self._expression_log: deque[str] = deque(maxlen=4)
        self.recent_outcomes: deque[BehaviorOutcome] = deque(
            maxlen=self.config.max_recent_outcomes
        )
        self._previous_viewer_count = 0
        self._last_memory_signal_id: Optional[str] = None

    @property
    def last_outcome(self) -> Optional[BehaviorOutcome]:
        return self.recent_outcomes[-1] if self.recent_outcomes else None

    @property
    def expression_sequence(self) -> list[str]:
        """Bounded log of the expression windows consumed so far."""
        return list(self._expression_log)

    def record_outcome(
        self,
        result: ActionResult,
        *,
        skill_id: str | None = None,
        now: datetime | None = None,
    ) -> None:
        current = now or self._now()
        self.recent_outcomes.append(
            BehaviorOutcome(
                action=result.action.value,
                status=result.status.value,
                success=result.status == ExecutionStatus.SUCCESS,
                reason=result.reason,
                timestamp=current,
                viewer_response=result.viewer_response,
                skill_id=skill_id,
            )
        )

    def update(
        self,
        world: WorldState | None,
        goal: Goal | None,
        *,
        social_context: Any | None = None,
        memory_context: Iterable[Any] | None = None,
        chat: Any | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        current = now or self._now()
        self.logical_tick += 1
        memories = list(memory_context or ())

        candidate, reason, signal_id, urgent = self._candidate_state(
            world,
            goal,
            social_context,
            memories,
            chat,
            current,
        )
        if candidate != self.state and self._dwell_satisfied(current, urgent=urgent):
            self.previous_state = self.state
            self.state = candidate
            self.transition_reason = reason
            self.transitioned_at = current
            self.transition_tick = self.logical_tick
            if signal_id is not None:
                # Remembering the last memory that caused curiosity keeps one
                # retrieved memory from re-triggering on every later tick; a
                # different memory still counts as a new meaningful signal.
                self._last_memory_signal_id = signal_id

        expression = self._consume_expression_window(world, social_context, current)
        return self.advisory_context(
            goal,
            proactive_allowed=expression == "positive_viewer_reaction",
            reactive_allowed=expression == "recent_setback",
            attention_allowed=self._consume_attention_window(world, current),
        )

    def advisory_context(
        self,
        goal: Goal | None,
        *,
        proactive_allowed: bool = False,
        reactive_allowed: bool = False,
        attention_allowed: bool = False,
    ) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "previous_state": self.previous_state.value if self.previous_state else None,
            "transition_reason": self.transition_reason,
            "transition_tick": self.transition_tick,
            "speech_style": self._speech_style(),
            "preferred_skills": self._preferred_skills(goal),
            "proactive_allowed": proactive_allowed,
            "proactive_reason": self.proactive_reason if proactive_allowed else None,
            "reactive_allowed": reactive_allowed,
            "reactive_reason": self.proactive_reason if reactive_allowed else None,
            "attention_allowed": attention_allowed,
            "expression_sequence": list(self._expression_log),
            "rule": "advisory only; never bypass skill validation or SkillExecutor",
        }

    def snapshot(self) -> dict[str, Any]:
        """Return only the state needed to reconstruct coherent continuity."""
        return {
            "current_state": self.state.value,
            "previous_state": self.previous_state.value if self.previous_state else None,
            "transition_reason": self.transition_reason,
            "transition_time": format_utc(self.transitioned_at) if self.transitioned_at else None,
            "transition_tick": self.transition_tick,
            "logical_tick": self.logical_tick,
            "last_proactive_at": format_utc(self.last_proactive_at)
            if self.last_proactive_at
            else None,
            "last_attention_at": format_utc(self.last_attention_at)
            if self.last_attention_at
            else None,
        }

    def restore(self, payload: dict[str, Any] | None) -> None:
        if not payload:
            return
        current = _optional_state(payload.get("current_state"), BehaviorState.CALM)
        previous = _optional_state(payload.get("previous_state"))
        if current is None:
            raise ValueError("invalid_behavior_state")
        self.state = current
        self.previous_state = previous
        self.transition_reason = str(payload.get("transition_reason") or "restored")
        self.transitioned_at = _parse_datetime(payload.get("transition_time"))
        self.transition_tick = int(payload.get("transition_tick") or 0)
        self.logical_tick = int(payload.get("logical_tick") or self.transition_tick)
        self.last_proactive_at = _parse_datetime(payload.get("last_proactive_at"))
        self.last_attention_at = _parse_datetime(payload.get("last_attention_at"))

    def to_dict(
        self,
        *,
        current_goal: Goal | dict[str, Any] | None = None,
        current_skill: str | None = None,
        current_action: str | None = None,
    ) -> dict[str, Any]:
        goal_value = current_goal
        if hasattr(goal_value, "to_dict"):
            goal_value = goal_value.to_dict()
        return {
            **self.snapshot(),
            "current_goal": goal_value,
            "current_skill": current_skill,
            "current_action": current_action,
            "recent_outcome": self.recent_outcomes[-1].to_dict() if self.recent_outcomes else None,
            "recent_outcomes": [outcome.to_dict() for outcome in self.recent_outcomes],
            "config": {
                "minimum_dwell_seconds": self.config.minimum_dwell_seconds,
                "reaction_window_seconds": self.config.reaction_window_seconds,
                "outcome_window_seconds": self.config.outcome_window_seconds,
                "proactive_cooldown_seconds": self.config.proactive_cooldown_seconds,
                "attention_cooldown_seconds": self.config.attention_cooldown_seconds,
            },
        }

    def _candidate_state(
        self,
        world: WorldState | None,
        goal: Goal | None,
        social_context: Any | None,
        memory_context: Iterable[Any] | None,
        chat: Any | None,
        now: datetime,
    ) -> tuple[BehaviorState, str, Optional[str], bool]:
        # The viewer baseline is refreshed on every update so a viewer that
        # arrived while another state was active never re-triggers curiosity
        # later as a phantom "new viewer" event.
        new_viewer = self._consume_new_viewer(world)
        if self._recent_outcome_failed(now):
            return BehaviorState.SURPRISED, "recent_action_failure", None, True
        if self._recent_reaction(social_context, "negative", now):
            return BehaviorState.SURPRISED, "negative_viewer_reaction", None, True
        if self._recent_reaction(social_context, "positive", now):
            return BehaviorState.HAPPY, "positive_viewer_reaction", None, False
        if self._active_task(world, goal):
            return BehaviorState.FOCUSED, "active_task_requires_attention", None, False
        # PHASE 26B: a goodbye is a discrete closing moment, not an idle tick.
        # It is answered while still socially engaged, and it may preempt the
        # dwell so a short exchange that ends right after a greeting does not
        # stay stuck in a transient state. Later idle ticks recover through the
        # existing engagement/calm rules, so no emotional state lingers.
        if self._farewell_context(chat):
            return BehaviorState.ENGAGED, "farewell_exchange", None, True
        if self._new_social_context(social_context, chat, new_viewer):
            return BehaviorState.CURIOUS, "new_viewer_or_topic", None, False
        memory_id = self._relevant_memory(memory_context)
        if memory_id is not None and memory_id != self._last_memory_signal_id:
            return BehaviorState.CURIOUS, "relevant_memory", memory_id, False
        if self._meaningfully_engaged(social_context, now):
            return BehaviorState.ENGAGED, "sustained_social_engagement", None, False
        if self._long_low_engagement_stream(world, social_context):
            return BehaviorState.TIRED, "long_stream_with_low_engagement", None, False
        return BehaviorState.CALM, "no_meaningful_behavior_event", None, False

    def _consume_new_viewer(self, world: WorldState | None) -> bool:
        current_count = world.viewer_count if world is not None else 0
        arrived = current_count > self._previous_viewer_count
        self._previous_viewer_count = current_count
        return arrived

    def _dwell_satisfied(self, now: datetime, *, urgent: bool = False) -> bool:
        """Minimum dwell between transitions, except for hard setbacks.

        A rejected action or a negative viewer reaction is a discrete event
        that may preempt the dwell; everything else waits so the state cannot
        flap on consecutive ticks.
        """
        if urgent or self.transitioned_at is None:
            return True
        elapsed = (now - self.transitioned_at).total_seconds()
        return elapsed >= self.config.minimum_dwell_seconds

    def _consume_expression_window(
        self,
        world: WorldState | None,
        social_context: Any | None,
        now: datetime,
    ) -> Optional[str]:
        """Allow at most one bounded expression per cooldown window.

        Returns the reason an expression was consumed, or ``None`` when the
        character should stay quiet. A live audience is required so an idle
        room never produces unprompted acting.
        """
        if world is None or not world.stream_active or world.viewer_count <= 0:
            return None
        if self.state == BehaviorState.HAPPY:
            reason = "positive_viewer_reaction"
            ready = self._recent_reaction(social_context, "positive", now)
        elif self.state == BehaviorState.SURPRISED:
            reason = "recent_setback"
            ready = self._recent_outcome_failed(now) or self._recent_reaction(
                social_context, "negative", now
            )
        else:
            return None
        if not ready:
            return None
        if self.last_proactive_at is not None:
            elapsed = (now - self.last_proactive_at).total_seconds()
            if elapsed < self.config.proactive_cooldown_seconds:
                return None
        self.last_proactive_at = now
        self.proactive_reason = reason
        self.expression_count += 1
        self._expression_log.append(f"{self.expression_count}:{reason}")
        return reason

    def _consume_attention_window(self, world: WorldState | None, now: datetime) -> bool:
        """Allow one bounded look toward a lookable camera on a live stream."""
        if self.state != BehaviorState.CURIOUS:
            return False
        if world is None or not world.stream_active or world.viewer_count <= 0:
            return False
        camera = world.target("CAMERA")
        if camera is None or not camera.look_at_target:
            return False
        if self.last_attention_at is not None:
            elapsed = (now - self.last_attention_at).total_seconds()
            if elapsed < self.config.attention_cooldown_seconds:
                return False
        self.last_attention_at = now
        return True

    def _recent_outcome_failed(self, now: datetime) -> bool:
        for outcome in reversed(self.recent_outcomes):
            elapsed = (now - outcome.timestamp).total_seconds()
            if elapsed > self.config.outcome_window_seconds:
                break
            if not outcome.success:
                return True
        return False

    def _recent_reaction(
        self,
        social_context: Any | None,
        kind: str,
        now: datetime,
    ) -> bool:
        reactions = getattr(social_context, "recent_reactions", ()) or ()
        for reaction in reversed(reactions):
            elapsed = (now - reaction.timestamp).total_seconds()
            if elapsed < 0:
                continue
            if elapsed > self.config.reaction_window_seconds:
                break
            if getattr(reaction, "kind", None) == kind:
                return True
        return False

    def _new_social_context(
        self,
        social_context: Any | None,
        chat: Any | None,
        new_viewer: bool,
    ) -> bool:
        conversation_state = _enum_value(getattr(social_context, "conversation_state", None))
        if conversation_state in {"GREETING", "CHANGING_TOPIC"}:
            return True
        chat_intent = _enum_value(getattr(chat, "intent", None))
        if chat_intent == "GREETING":
            return True
        if new_viewer:
            return True

        active_viewer = getattr(social_context, "active_viewer_id", None)
        messages = getattr(social_context, "recent_messages", ()) or ()
        if active_viewer:
            viewer_messages = [
                message
                for message in messages
                if getattr(message, "viewer_id", None) == active_viewer
            ]
            if len(viewer_messages) == 1:
                return True
        return False

    @staticmethod
    def _farewell_context(chat: Any | None) -> bool:
        """A goodbye turn is a discrete closing moment, not an idle tick."""
        if chat is None:
            return False
        intent = _enum_value(getattr(chat, "intent", None))
        return intent in {"FAREWELL", "CONVERSATION_END"}

    def _relevant_memory(self, memory_context: Iterable[Any] | None) -> Optional[str]:
        """Return the strongest relevant memory id, once per distinct memory."""
        for memory in memory_context or ():
            memory_type = _enum_value(getattr(memory, "memory_type", None))
            score = getattr(memory, "score", 0.0)
            if (
                str(memory_type).upper() in {"EPISODE", "SEMANTIC"}
                and isinstance(score, (int, float))
                and not isinstance(score, bool)
                and score >= self.config.memory_score_threshold
            ):
                memory_id = getattr(memory, "memory_id", None)
                return str(memory_id) if memory_id else "retrieved_memory"
        return None

    @staticmethod
    def _active_task(world: WorldState | None, goal: Goal | None) -> bool:
        current_action = world.current_action.strip().casefold() if world else ""
        if current_action and current_action not in {"idle", "standing", "none", "stopped"}:
            return True
        return goal is not None and goal.goal_type.value == "PERFORM_ACTIVITY"

    def _meaningfully_engaged(self, social_context: Any | None, now: datetime) -> bool:
        engagement = getattr(social_context, "social_engagement", 0.0)
        last_activity = getattr(social_context, "last_activity_at", None)
        if not isinstance(engagement, (int, float)) or isinstance(engagement, bool):
            return False
        if engagement < self.config.engagement_threshold or last_activity is None:
            return False
        return (now - last_activity).total_seconds() <= 180.0

    def _long_low_engagement_stream(
        self,
        world: WorldState | None,
        social_context: Any | None,
    ) -> bool:
        if world is None or not world.stream_active:
            return False
        if world.stream_duration_seconds < self.config.tired_stream_seconds:
            return False
        engagement = getattr(social_context, "social_engagement", 0.0)
        if not isinstance(engagement, (int, float)) or isinstance(engagement, bool):
            return False
        return (
            engagement <= self.config.tired_engagement_threshold
            and world.viewer_count <= 1
        )

    def _speech_style(self) -> str:
        return {
            BehaviorState.CALM: "neutral and concise",
            BehaviorState.HAPPY: "positive and energetic, but concise",
            BehaviorState.CURIOUS: "interested and attentive",
            BehaviorState.FOCUSED: "minimal unnecessary speech; continue the task",
            BehaviorState.SURPRISED: "short reactive wording",
            BehaviorState.TIRED: "quiet and low-energy, but clear",
            BehaviorState.ENGAGED: "warm and conversational",
        }[self.state]

    def _preferred_skills(self, goal: Goal | None) -> list[str]:
        goal_type = goal.goal_type.value if goal else None
        if goal_type in {"RESPOND_TO_VIEWER", "INTERACT_WITH_VIEWER"}:
            preferred = ["look_at", "wave", "speak"]
        elif goal_type == "EXPLORE_ROOM":
            preferred = ["look_at"]
        else:
            preferred = []
        for skill_id in {
            BehaviorState.HAPPY: ("wave", "dance", "speak"),
            BehaviorState.CURIOUS: ("look_at", "speak"),
            BehaviorState.SURPRISED: ("speak",),
            BehaviorState.ENGAGED: ("speak", "look_at"),
        }.get(self.state, ()):
            if skill_id not in preferred:
                preferred.append(skill_id)
        return preferred[:4]

    def _now(self) -> datetime:
        if self._clock is None:
            from ..chat.models import utc_now

            return utc_now()
        value = self._clock()
        if not isinstance(value, datetime):
            raise TypeError("behavior_clock_must_return_datetime")
        return value


def _enum_value(value: Any) -> Any:
    return value.value if hasattr(value, "value") else value


def _optional_state(value: Any, default: Optional[BehaviorState] = None) -> Optional[BehaviorState]:
    if value is None:
        return default
    try:
        return BehaviorState(str(value).upper())
    except ValueError:
        return None


def _parse_datetime(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
