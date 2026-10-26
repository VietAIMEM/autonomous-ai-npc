"""The controlled autonomous decision loop."""

from __future__ import annotations

import hashlib
import logging
import re
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime
from itertools import count
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, Optional

from ..chat.models import ChatIntent, ChatMessage, utc_now
from ..chat.service import ChatService
from ..memory.models import ViewerMemory
from ..memory.models import MemoryQuery
from ..runtime.obs import log_event
from ..skills.executor import SkillExecutor
from ..skills.models import SkillSelection
from .behavior import BehaviorManager
from .expression import (
    GESTURE_DANCE,
    GESTURE_WAVE,
    ExpressionDecision,
    ExpressionPolicy,
)
from .goals import (
    INITIATIVE_IDLE_ACTIONS,
    INITIATIVE_WAIT,
    MAX_RECENT_INITIATIVES,
    STREAM_LIFECYCLE_UNKNOWN,
    GoalEvaluator,
    InitiativeDecision,
    InitiativeContext,
    InitiativeKnowledge,
    camera_available_from_world,
    initiative_goal_context,
    initiative_signature,
    normalize_stream_lifecycle,
    resolve_initiative_context,
)
from .models import (
    ActionRequest,
    ActionResult,
    ActionType,
    BrainStatus,
    Decision,
    ExecutionStatus,
    Goal,
    GoalStatus,
    GoalType,
    PersonalityProfile,
    Plan,
    PlanStep,
    WorldState,
)
from .personality import load_personality
from .personalization import PersonalizedContext
from .planner import MAX_TEXT_LENGTH, BrainPlanner
from .reasoning import BrainReasoner
from .spatial import (
    DESIRED_CLOSER,
    DESIRED_IN_FRAME,
    DESIRED_NEAR,
    FRAMING_PLACEMENT_ONLY,
    ORIENTATION_TOLERANCE_DEGREES,
    POSITION_ARRIVAL_TOLERANCE_METERS,
    TARGET_CAMERA,
    distance,
    is_spatial_plan,
    plan_anchor,
    spatial_observation,
    spatial_relationship,
)

if TYPE_CHECKING:
    from ..learning.service import SkillLearningService

if TYPE_CHECKING:
    from ..memory.service import MemoryService
    from ..social.service import SocialAutonomyService
    from ..speech.service import SpeechService


#: PHASE 25: actions whose reported success is a claim about a physical
#: condition, and therefore has to be confirmed against reported transforms.
_SPATIAL_VERIFIABLE_ACTIONS = frozenset(
    {ActionType.MOVE_TO, ActionType.TURN_TO, ActionType.LOOK_AT}
)

#: PHASE 26A: a spatial step against an inactive camera has a measurable
#: placement half (arrival, body orientation) and an unmeasurable visibility
#: half. This result advances the plan only when the placement half passed, and
#: keeps the unmeasured framing explicit in diagnostics instead of claiming
#: that the character is inside a frame nobody can currently observe.
SPATIAL_PLACEMENT_VERIFIED = "PLACEMENT_VERIFIED"

#: PHASE 26B: the already-registered expressive motions this phase may place
#: into an existing plan. No new animation or motion source is introduced.
_GESTURE_ACTION_TYPES: dict[str, ActionType] = {
    GESTURE_WAVE: ActionType.WAVE,
    GESTURE_DANCE: ActionType.DANCE,
}

#: A bare greeting is a social turn, not permission to explore or manipulate
#: the room. Gestures stay allowed because they can accompany the spoken reply.
_GREETING_SOCIAL_ACTIONS = frozenset(
    {
        ActionType.SPEAK,
        ActionType.WAVE,
        ActionType.DANCE,
        ActionType.STOP,
        ActionType.LOOK_AT,
    }
)

#: PHASE 27B: where the goal/plan the runtime is currently carrying came from.
#: ``brain:initiative`` is an autonomous activity the initiative evaluator
#: chose, ``brain:evaluator`` is the legacy decision path (viewer work, stream
#: maintenance, exploration), and ``external`` marks a goal or plan this loop
#: never built -- so an injected or hand-loaded plan can never be reported as
#: an actual autonomous Brain decision.
DECISION_SOURCE_INITIATIVE = "brain:initiative"
DECISION_SOURCE_EVALUATOR = "brain:evaluator"
DECISION_SOURCE_EXTERNAL = "external"

#: PHASE 27B: the bounded tokens ``next_decision_eligibility`` may report.
NEXT_DECISION_ACTION_IN_FLIGHT = "action_in_flight"
NEXT_DECISION_STEP_READY = "step_ready"
NEXT_DECISION_COOLDOWN = "decision_cooldown"
NEXT_DECISION_INITIATIVE_INTERVAL = "initiative_interval"
NEXT_DECISION_ELIGIBLE = "eligible"

#: PHASE 27B: how many eligible/rejected candidates the diagnostics keep.
MAX_INITIATIVE_CANDIDATES_IN_STATE = 6

#: PHASE 32: the target-bearing actions whose dispatch is instrumented. Other
#: actions carry no spatial target, so logging them would add volume without
#: answering "which canonical target was sent".
_DISPATCH_INSTRUMENTED_ACTIONS = frozenset(
    {ActionType.LOOK_AT, ActionType.TURN_TO, ActionType.MOVE_TO}
)

#: PHASE 32: bounded shape a logged target/goal/plan token must have. Anything
#: else (an over-long string, punctuation, embedded spacing) is dropped rather
#: than truncated, so an unexpected value can never leak into a log line.
_SAFE_TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9._:\-/]{1,64}$")

_LOGGER = logging.getLogger(__name__)


def _bounded_token(value: object) -> Optional[str]:
    """A short, identifier-shaped token, or ``None``."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or len(text) > 64:
        return None
    return text if _SAFE_TOKEN_PATTERN.match(text) else None


def _new_dispatch_id() -> str:
    """PHASE 34: the single authoritative id for one action dispatch.

    Compact and bounded (well under ``_SAFE_TOKEN_PATTERN``'s 64-character
    limit) so it is safe to carry on the wire and into structured logs. Random
    rather than sequential so an id is never reused across process restarts.
    """
    return f"dsp_{uuid.uuid4().hex}"


#: PHASE 27B: refusals that mean the *world* no longer offers the context the
#: recorded initiative evidence describes (no audience, no capability, no
#: configuration). Those release the decision-time evidence, so diagnostics
#: never present a stopped stream's "live" context as the current verdict.
#: Transient refusals -- a goal or action already in flight, a pending viewer
#: message, active speech -- keep it: the evidence still explains the decision
#: the runtime is carrying.
INITIATIVE_OUT_OF_SCOPE_REASONS = frozenset(
    {"autonomy_disabled", "skills_unavailable", "no_world", "no_audience"}
)


def _first_gesture_index(plan: Plan) -> int:
    """Index of the first registered expressive motion in a plan, else -1."""
    for index, step in enumerate(plan.steps):
        if step.action in _GESTURE_ACTION_TYPES.values():
            return index
    return -1


def _bounded_count(value: Any) -> int:
    """A non-negative integer read from an untyped record, defaulting to 0."""
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


@dataclass(frozen=True)
class BrainConfig:
    decision_interval_seconds: float = 5.0
    max_plan_steps: int = 8
    max_retries: int = 1
    interruption_priority: float = 0.7
    #: PHASE 27A: the single Brain loop may fill an idle tick with a registered
    #: autonomous activity. The interval paces how often a *new* activity may
    #: start; it reuses the existing decision cadence and never schedules work
    #: of its own.
    autonomy_enabled: bool = True
    initiative_interval_seconds: float = 25.0


@dataclass
class BrainRuntimeState:
    status: BrainStatus = BrainStatus.IDLE
    current_goal: Optional[Goal] = None
    current_plan: Optional[Plan] = None
    step_index: int = 0
    pending_action: Optional[ActionRequest] = None
    retries: int = 0
    plan_version: int = 0
    last_result: Optional[ActionResult] = None
    decision_id_sequence: count = field(default_factory=count)
    active_message: Optional[ChatMessage] = None
    #: Why the last tick could not produce an action. Without this the runtime
    #: silently returned ``pending_action=null`` and the cause was invisible.
    last_failure: Optional[str] = None
    #: PHASE 25: the structured spatial state behind the active plan
    #: (FACING/IN_FRAME/CLOSER/NEAR), captured while the chat message is still
    #: attached so the post-action verifier checks the right condition.
    spatial_desired_state: Optional[str] = None
    #: PHASE 25: bounded evidence of the last spatial verification. Populated
    #: only for plans the deterministic spatial branch produced, so legacy
    #: plans keep the exact runtime shape they had before.
    spatial_verification: Optional[dict[str, Any]] = None
    #: PHASE 26A: why the last plan was cancelled for a viewer interruption.
    #: Cleared when a plan completes or a new plan is built, so diagnostics
    #: never report a stale reason.
    interruption_reason: Optional[str] = None
    #: PHASE 26B: the bounded expression decision behind the current or last
    #: interaction (tone + reaction intensity + optional gesture). Advisory
    #: only: it is derived from the same single reasoning result and never
    #: plans on its own.
    expression_decision: Optional[ExpressionDecision] = None
    #: PHASE 26B: the gesture accepted into the active plan. Cleared with the
    #: plan so an interruption can never leave a stale gesture in diagnostics.
    active_gesture: Optional[str] = None
    #: PHASE 27A: why the last autonomy decision happened (or why it waited).
    #: Kept as the last initiative reason; the current goal carries its own.
    initiative_reason: Optional[str] = None
    #: PHASE 27A: the last activity the initiative selector chose (WAIT when it
    #: decided to wait), exposed verbatim through the autonomy diagnostics.
    selected_activity: Optional[str] = None
    #: PHASE 27A: bookkeeping for the explore cadence. ``legacy_explore_goal_id``
    #: names the legacy explore decision currently in flight (so its completion
    #: can be recognised), and ``explore_completed_since_initiative`` records
    #: that such a cycle finished. The initiative gate claims exactly one
    #: explore-due decision after that completion instead of letting the explore
    #: cadence starve autonomy for the lifetime of the process.
    legacy_explore_goal_id: Optional[str] = None
    explore_completed_since_initiative: bool = False
    #: PHASE 27A: bounded outcome of the last completed autonomous activity,
    #: e.g. ``"dance:completed"``. WAIT decisions are not outcomes.
    last_outcome: Optional[str] = None
    #: PHASE 27A: bounded journal of recently finished autonomous activities as
    #: ``(activity, world signature)`` pairs. Repetition protection compares
    #: against exactly this journal and nothing else.
    recent_autonomy_activities: deque[tuple[str, str]] = field(
        default_factory=lambda: deque(maxlen=MAX_RECENT_INITIATIVES)
    )
    #: PHASE 27A: when the last autonomous activity was started, so a new one
    #: is paced by ``BrainConfig.initiative_interval_seconds``.
    last_initiative_at: Optional[datetime] = None
    #: PHASE 27B: the top-ranked candidate the initiative evaluator offered
    #: this tick (``None`` when no candidate passed the filters) and the
    #: bounded per-candidate rejection reasons behind the decision.
    candidate_activity: Optional[str] = None
    eligible_candidates: tuple[str, ...] = ()
    candidate_rejections: tuple[tuple[str, str], ...] = ()
    #: PHASE 27B: the deterministic score the ranking used for each eligible
    #: candidate, in the same ranked order. An empty tuple means "nothing was
    #: ranked this tick", never a fabricated zero.
    candidate_scores: tuple[tuple[str, float], ...] = ()
    #: PHASE 27B: the part of each of those scores that came from the existing
    #: verified ``SkillKnowledge`` record, in the same ranked order. Reported
    #: separately so diagnostics can show what the persisted evidence actually
    #: contributed instead of only the total it was folded into.
    knowledge_contributions: tuple[tuple[str, float], ...] = ()
    #: PHASE 27B: the bounded context the last initiative decision actually
    #: read (stream lifecycle, camera verdict, phone state, behaviour state,
    #: behaviour-preferred skills, skills with verified knowledge). Captured
    #: at decision time so diagnostics never describe a later world.
    initiative_context: Optional[dict[str, Any]] = None
    #: PHASE 27B: why the last cancellation sent the loop back into REPLANNING
    #: (cleared once a new plan is dispatched or the goal completes).
    replanning_reason: Optional[str] = None
    #: PHASE 27B: bounded token describing when the single Brain loop may make
    #: its next decision, so operators can see pacing without reading timers.
    next_decision_eligibility: Optional[str] = None
    #: PHASE 27B: which decision path produced the goal/plan in flight (see
    #: ``DECISION_SOURCE_*``). ``None`` while idle.
    decision_source: Optional[str] = None


@dataclass(frozen=True)
class BrainTickResult:
    status: BrainStatus
    goal: Optional[Goal]
    plan: Optional[Plan]
    action: Optional[ActionRequest]
    message: str = "ok"

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "goal": self.goal.to_dict() if self.goal else None,
            "plan": self.plan.to_dict() if self.plan else None,
            "action": self.action.to_dict() if self.action else None,
            "message": self.message,
        }


class AutonomousBrain:
    def __init__(
        self,
        provider,
        chat_service: ChatService,
        config: BrainConfig | None = None,
        personality: PersonalityProfile | None = None,
        clock: Callable[[], object] = utc_now,
        memory_service: Optional["MemoryService"] = None,
        social_service: Optional["SocialAutonomyService"] = None,
        skill_library=None,
        learning_service: Optional[SkillLearningService] = None,
        speech_service: Optional["SpeechService"] = None,
        behavior: Optional[BehaviorManager] = None,
        expression: Optional[ExpressionPolicy] = None,
        streaming=None,
    ) -> None:
        self.provider = provider
        self.chat = chat_service
        self.config = config or BrainConfig()
        self.personality = personality or self._load_persisted_personality(memory_service)
        self.goal_evaluator = GoalEvaluator(self.personality)
        self.planner = BrainPlanner()
        self.skill_executor = (
            SkillExecutor(skill_library, self.planner, clock)
            if skill_library is not None
            else None
        )
        self.reasoner = BrainReasoner(provider, self.planner, self.skill_executor)
        self.state = BrainRuntimeState()
        self.world: WorldState | None = None
        self.memory = memory_service
        self.social = social_service
        if self.social is not None:
            # PHASE 26A: the social layer owns the wider topic vocabulary, so
            # the chat pipeline asks it for the active topic instead of
            # maintaining a second, divergent one.
            resolver = getattr(self.social, "conversation", None)
            setter = getattr(self.chat, "set_topic_resolver", None)
            if resolver is not None and callable(setter):
                setter(resolver.infer_topic)
        self.learning = learning_service
        self.speech = speech_service
        #: PHASE 27B: read-only handle on the existing streaming service. The
        #: initiative evaluator reads its lifecycle so a start/stop in progress
        #: is never mistaken for a settled state; nothing here drives it.
        self.streaming = streaming
        self.behavior = behavior or BehaviorManager(clock=clock)
        #: PHASE 26B: bounded expression coordination for the same single
        #: reasoning result. It never plans, never executes and never calls a
        #: provider; it maps intent + behavior + personality + relationship
        #: onto one inspectable tone/intensity/gesture decision.
        self.expression = expression or ExpressionPolicy(clock=clock)
        self.last_behavior_context: dict[str, object] | None = None
        self.last_expression: ExpressionDecision | None = None
        self.last_personalization: PersonalizedContext | None = None
        self._active_context_key: str | None = None
        #: Skill that produced the current plan, captured while it is active.
        self._plan_skill_id: str | None = None
        #: PHASE 26B: the plan step the expression policy filled with a
        #: decorative gesture, recorded as ``(plan_id, step_index)``. Only that
        #: exact step may be skipped when Unity rejects it; a gesture the plan
        #: itself asked for keeps the legacy reject-and-replan path.
        self._coordinated_gesture_step: tuple[str, int] | None = None
        self._clock = clock
        self._last_decision_at = None
        #: PHASE 27B: provenance of the goal/plan this loop actually built, so
        #: diagnostics can tell an autonomous or evaluator decision apart from
        #: a plan a harness (or another process) loaded into the runtime state.
        self._decision_source: Optional[str] = None
        self._decision_goal_id: Optional[str] = None
        self._decision_plan_id: Optional[str] = None
        #: PHASE 26A: chat intake runs on its own thread (the viewer/Twitch
        #: handler) while Unity owns the tick cadence. This lock serialises
        #: the two, and intake takes it non-blocking so a running tick can
        #: never be made to wait on a viewer.
        self._state_lock = threading.RLock()
        listener = getattr(self.chat, "set_message_listener", None)
        if callable(listener):
            # The same bounded priority check the tick runs, evaluated when
            # the message actually arrives. Unity does not tick while an
            # action is executing, so without this an important request
            # queued during a long utterance could only be answered after it.
            listener(self.note_viewer_message)

    @staticmethod
    def _load_persisted_personality(
        memory_service: Optional["MemoryService"],
    ) -> PersonalityProfile:
        """Load the stable profile from the existing store, defaulting safely.

        Nothing here evolves: the profile is read once at construction and only
        an explicit configuration write can change it later.
        """
        if memory_service is not None:
            try:
                return memory_service.load_personality_state()
            except Exception:
                pass
        return load_personality()

    def tick(
        self,
        world: WorldState,
        result: ActionResult | None = None,
    ) -> BrainTickResult:
        with self._state_lock:
            outcome = self._tick_locked(world, result)
            self._finalize_tick_diagnostics()
            return outcome

    def _tick_locked(
        self,
        world: WorldState,
        result: ActionResult | None = None,
    ) -> BrainTickResult:
        self.world = world
        if self.social is not None:
            self.social.update_world(world)
        if result is not None and not self._is_awaited_result(result):
            # Unity retries a result when its tick request fails. Once another
            # action is pending, an old result must not advance that action, and
            # (PHASE 34) a result that does not echo the pending dispatch id can
            # never complete it either. Drop it without disturbing the pending
            # action or the result already recorded for the action it belongs
            # to.
            self._note_uncorrelated_result(result)
            result = None
        else:
            self.state.last_result = result
        if self.skill_executor is not None and self.skill_executor.timeout_active():
            # PHASE 27B: a timed-out execution is its own terminal outcome, so
            # diagnostics can separate it from an ordinary failed/cancelled
            # activity. The existing cancellation/replanning path is unchanged.
            self._cancel_plan("SKILL_TIMEOUT", outcome="timed_out")

        if result is not None:
            self._observe_result(result)
            if (
                result.status == ExecutionStatus.SUCCESS
                and self.state.pending_action is None
                and self.state.last_failure is None
                and not self._has_replannable_step()
                and self.state.current_plan is None
            ):
                return BrainTickResult(BrainStatus.IDLE, None, None, None, "plan_complete")

        pending = self.state.pending_action
        if pending is not None:
            # PHASE 26A: an important viewer message outranks in-flight work.
            # This reuses the existing priority/cancellation machinery -- the
            # plan is cancelled, the speech request is cancelled with it, and
            # the queued message is handled by the very same tick so no second
            # response and no second interruption system exist.
            interruption = self._viewer_interruption()
            if interruption is not None:
                self._interrupt_for_viewer(interruption)
                pending = None
        if pending is not None:
            blocked = self._deliberate_stop_conflict((pending.action,))
            if blocked is not None:
                # A stop latched while the action was in flight outranks the
                # retry; cancelling also drops the stale pending start.
                self._cancel_plan(blocked)
                self.state.last_failure = f"validation_failed:{blocked}"
                self.state.status = BrainStatus.ERROR
                return BrainTickResult(
                    BrainStatus.ERROR, None, None, None, self.state.last_failure
                )
            return BrainTickResult(
                status=BrainStatus.EXECUTING,
                goal=self.state.current_goal,
                plan=self.state.current_plan,
                action=pending,
                message="awaiting_action_result",
            )

        if self._reconcile_stale_initiative_goal():
            # PHASE 27B: the active autonomous activity was started on a
            # premise the world no longer supports (the stream moved to a
            # state the activity declared it needs, the phone went away, the
            # camera was switched off, or the activity's whole reason is gone).
            # The existing cancellation machinery already released the plan and
            # cleared the decision cooldown, so this same tick re-decides
            # instead of resuming a stale goal.
            pass
        elif self._has_replannable_step():
            return self._begin_step()

        if self._new_decision_too_soon():
            return BrainTickResult(
                status=BrainStatus.WAITING,
                goal=self.state.current_goal,
                plan=self.state.current_plan,
                action=None,
                message="decision_cooldown",
            )

        return self._decide()

    def _is_awaited_result(self, result: ActionResult) -> bool:
        pending = self.state.pending_action
        if pending is None or pending.action != result.action:
            return False
        pending_id = pending.dispatch_id
        if pending_id is None:
            # A legacy or test-created pending action carries no id, so the
            # historical action-type-only match is preserved unchanged.
            return True
        # PHASE 34: the pending dispatch carries an id, so the result must echo
        # exactly that id. A missing or different id is a stale, duplicate, or
        # uncorrelated result and must never complete this action.
        return result.dispatch_id == pending_id

    def _note_uncorrelated_result(self, result: ActionResult) -> None:
        """PHASE 34: bounded diagnostics for a result that matched no dispatch.

        Only the genuinely new condition -- the action type matches the pending
        action but the echoed dispatch id does not -- is logged, and only once
        per distinct ``(action, dispatch_id)`` so a result Unity replays after a
        failed POST cannot flood the log. Never logs payload bodies.
        """
        pending = self.state.pending_action
        if pending is None or pending.action != result.action:
            return
        signature = (result.action.value, result.dispatch_id)
        if signature == getattr(self, "_last_uncorrelated_signature", None):
            return
        self._last_uncorrelated_signature = signature
        try:
            fields: dict[str, Any] = {"action": result.action.value}
            echoed = _bounded_token(result.dispatch_id)
            if echoed is not None:
                fields["dispatch_id"] = echoed
            expected = _bounded_token(pending.dispatch_id)
            if expected is not None:
                fields["expected_dispatch_id"] = expected
            log_event(_LOGGER, "brain.action_result_uncorrelated", **fields)
        except Exception:  # pragma: no cover - diagnostics never block the loop
            pass

    def _viewer_interruption(self) -> Optional[ChatMessage]:
        """PHASE 26A: the queued viewer message that may preempt the plan.

        Bounded and conservative: the message must clear the configured
        interruption priority *and* outrank the goal that is currently in
        flight. Everything else keeps waiting its turn.
        """
        return self._interruption_candidate(self.chat.queue.peek())

    def _interruption_candidate(
        self, message: Optional[ChatMessage]
    ) -> Optional[ChatMessage]:
        """The bounded priority gate shared by arrival and tick evaluation."""
        if self.world is None or message is None:
            return None
        if message.priority < self.config.interruption_priority:
            return None
        current = self.state.current_goal
        if current is not None and current.priority >= message.priority:
            return None
        return message

    def _interrupt_for_viewer(self, message: ChatMessage) -> bool:
        """Cancel in-flight work for an important viewer message.

        The existing cancellation/replanning machinery does the work; the
        caller only keeps the consequences visible in diagnostics and decides
        again immediately instead of waiting out the decision cooldown.
        """
        self._cancel_plan("interrupted_by_viewer")
        # The cancel already consumed the decision cooldown's purpose: the new
        # request must be decided now, not in N seconds.
        self._last_decision_at = None
        return True

    def note_viewer_message(self, message: ChatMessage) -> bool:
        """PHASE 26A: evaluate an arriving message against in-flight work.

        Unity only ticks between actions, so a request that arrives while an
        utterance or a movement is executing would otherwise wait for it to
        finish. This entry point runs the exact tick-time priority gate as
        soon as the message is queued. The state lock is taken without
        blocking: a tick that already owns the Brain evaluates the queue
        itself, and a message that never raises the bounded flag stays queued
        for the normal turn order.
        """
        if self.state.pending_action is None:
            return False
        if not self._state_lock.acquire(blocking=False):
            return False
        try:
            if self.state.pending_action is None:
                return False
            candidate = self._interruption_candidate(message)
            if candidate is None:
                return False
            return self._interrupt_for_viewer(candidate)
        finally:
            self._state_lock.release()

    def _decide(self) -> BrainTickResult:
        # PHASE 26A: when this decision replaces work that was just cancelled
        # (an important viewer message, a skill timeout), the cancel reason
        # stays visible for the replacement response it caused so diagnostics
        # can attribute the new plan. A fresh, unrelated decision clears it.
        replacing_cancelled_work = self.state.status == BrainStatus.REPLANNING
        self.state.status = BrainStatus.THINKING
        processed = None
        chat: ChatMessage | None = None
        social_decision = None
        personalization = None
        # PHASE 24: a dead stream must not be starved by continuous chat. Real
        # traffic kept the attention queue non-empty on almost every tick, so
        # the maintenance goal only surfaced after the backlog drained
        # (measured ~10.4 min from an operator START to LIVE). Probing the
        # evaluator first lets recovery run; the queued message is left in
        # place and expires under its existing bound. A deliberate stop keeps
        # the probe false, so chat handling is untouched in that case.
        goal = self._preempted_maintenance_goal()
        if goal is None:
            head = self.chat.queue.peek()
            if self.social is not None and head is not None:
                self.social.context.expire_if_needed(self._clock())
                try:
                    social_decision = self.social.process_message(head, self.world, self.memory)
                except Exception:
                    # A social-layer fault must not disable the authoritative brain.
                    social_decision = None

            if social_decision is not None:
                decision_name = social_decision.decision.value
                if decision_name == "IGNORE":
                    self.chat.discard_next("social_attention_ignore")
                    head = None
                elif decision_name == "DEFER":
                    head = None

            personalization = self._personalized_context(head)
            self.last_personalization = personalization
            if head is not None:
                # PHASE 26B: the same decision that will answer this message
                # already carries the bounded expression hint, so the single
                # response prompt gets the tone without a second model call.
                expression = self._expression_decision(head.intent, personalization)
                processed = self.chat.process_next(
                    personalization=self._chat_personalization(
                        personalization, expression
                    )
                )
                chat = processed.message if processed else None
                if self.social is not None and chat is not None:
                    self.social.notify_processed(
                        chat,
                        processed.response.speech if processed else None,
                        processed.response.requested_action if processed else None,
                        self._clock(),
                    )

            goal = self.goal_evaluator.evaluate(
                self.world,
                chat,
                social_context=self.social.context if self.social else None,
            )
        # PHASE 27A: remember the explore decision the legacy path just made.
        # ``evaluate`` consumes its own explore cadence, so once a real
        # exploration plan has finished the next decision is explore-due again
        # (planning regularly takes longer than the 25 s explore interval).
        # Only a *completed* exploration opens the initiative gate below; the
        # id alone must never, or the very first explore plan would be skipped.
        self.state.legacy_explore_goal_id = (
            goal.goal_id
            if (
                goal.goal_type is GoalType.EXPLORE_ROOM
                and initiative_goal_context(goal) is None
            )
            else None
        )
        if goal is None:
            self.state.status = BrainStatus.WAITING
            message = (
                "social_attention_deferred"
                if social_decision is not None and social_decision.decision.value == "DEFER"
                else "no_goal"
            )
            return BrainTickResult(BrainStatus.WAITING, None, None, None, message)

        context_key = None
        if self.learning is not None:
            try:
                context_key = self.learning.observe_context(
                    goal,
                    chat,
                    self.world,
                    self._viewer_for_context(chat),
                )
            except Exception:
                context_key = None
        self._active_context_key = context_key

        if social_decision is not None and social_decision.decision.value == "DEFER":
            # Do not start an incidental idle/explore plan while Unity reports
            # that a socially interrupting action is already in progress.
            self.state.status = BrainStatus.WAITING
            return BrainTickResult(BrainStatus.WAITING, None, None, None, "social_attention_deferred")

        if self.memory is not None:
            if chat is not None:
                self.memory.remember_chat(chat, processed.response if processed else None)
            self.memory.observe_context(
                goal=goal.goal_type.value,
                action=self.state.pending_action.action.value if self.state.pending_action else None,
                world_context=self._memory_world_context(),
            )

        memories = self._retrieve_memories(goal, chat)
        behavior_context = self._update_behavior(goal, chat, memories)
        # PHASE 26B: the behavior state may have just changed (a greeting made
        # her curious, a setback surprised her). The expression decision is
        # confirmed against that state before the plan is built, so the plan
        # prompt and diagnostics describe the same interaction.
        expression = self._expression_decision(
            chat.intent if chat is not None else None,
            personalization,
        )
        behavior_context = self._expression_advisory(behavior_context, expression)
        self.state.expression_decision = expression
        self.last_expression = expression

        initiative_offered = False
        if goal.goal_type is GoalType.IDLE:
            # PHASE 27A: the evaluator found no viewer work and no exploration
            # due. This is exactly the idle gap the single Brain loop may fill
            # with a registered autonomous activity; every other decision the
            # evaluator can make keeps the legacy path untouched.
            initiative_offered = True
            initiative = self._initiative_tick(chat, replacing_cancelled_work)
            if initiative is not None:
                return initiative
        elif (
            self.state.legacy_explore_goal_id
            and self._legacy_explore_cycle_ready()
        ):
            # PHASE 27A: a completed legacy exploration doesn't starve
            # autonomy. The first explore-due decision after it hands the tick
            # to the same bounded initiative gate; when the gate is
            # ineligible, still waiting out its interval, or decides to WAIT,
            # the legacy exploration plan continues exactly as before.
            eligible, _reason = self._initiative_eligibility(chat)
            if eligible:
                initiative_offered = True
                initiative = self._initiative_tick(chat, replacing_cancelled_work)
                if initiative is not None:
                    return initiative
        if initiative_offered:
            # The completed exploration has now released its one initiative
            # slot, whatever the gate decided.
            self.state.explore_completed_since_initiative = False
            self.state.legacy_explore_goal_id = None

        self.state.status = BrainStatus.PLANNING
        # The legacy evaluator produced a real goal, so any ACTIVE WAIT goal
        # from a previous initiative tick has been superseded and must not
        # linger as the current goal.
        self._retire_wait_goal()
        self.state.current_goal = goal
        self.state.active_message = chat
        # PHASE 25: the structured spatial state, if any, travels with the
        # decision so the deterministic planner and the post-action verifier
        # read the same request. Free text still never reaches the spatial
        # branch on its own.
        self.state.spatial_desired_state = chat.desired_state if chat else None
        plan, reason = self.reasoner.plan(
            self.world,
            goal,
            self.personality,
            chat.normalized_text if chat else None,
            memories,
            self._ranking_context(context_key, personalization),
            behavior_context,
            personalization,
            desired_state=chat.desired_state if chat else None,
            spatial_target=chat.target if chat else None,
        )
        if plan is None:
            self._plan_skill_id = None
            self.state.status = BrainStatus.ERROR
            self.state.current_plan = None
            self.state.pending_action = None
            self.state.last_failure = f"planning_failed:{reason}"
            return BrainTickResult(BrainStatus.ERROR, goal, None, None, self.state.last_failure)

        # PHASE 19: capture the skill behind this plan now. Appended dialogue
        # steps and SkillExecutor.observe_success both clear the active skill
        # before the plan's final step completes, so the completed exchange
        # would otherwise reach viewer memory without its skill evidence.
        self._plan_skill_id = self.current_skill_id()
        plan = self._strip_unrelated_greeting_actions(plan, chat)
        plan = self._normalize_gesture_dialogue(plan)
        plan = self._append_dialogue_step(plan, goal, processed)
        uncoordinated = plan
        plan = self._coordinate_gesture(plan, goal, expression)
        self._publish_authoritative_response(plan, processed)
        self._coordinated_gesture_step = (
            (plan.plan_id, _first_gesture_index(plan))
            if plan is not uncoordinated
            else None
        )

        invalid = [self.planner.validate_step(step, self.world) for step in plan.steps]
        failed = next((item.reason for item in invalid if not item.valid), None)
        if failed is None:
            failed = self._deliberate_stop_conflict(step.action for step in plan.steps)
        if failed:
            self.state.status = BrainStatus.ERROR
            self.state.current_plan = None
            self.state.pending_action = None
            self.state.last_failure = f"validation_failed:{failed}"
            return BrainTickResult(BrainStatus.ERROR, goal, None, None, self.state.last_failure)

        self.state.current_plan = plan
        self.state.plan_version = plan.version
        self.state.step_index = 0
        self.state.retries = 0
        self.state.last_failure = None
        self.state.active_gesture = None
        gesture = self._plan_gesture(plan)
        if gesture is not None:
            # Only a gesture that actually entered a validated, dispatched plan
            # counts toward suppression, so a refused plan never silences the
            # next real greeting.
            self.state.active_gesture = gesture
            try:
                self.expression.note_gesture(gesture, self._clock())
            except Exception:
                pass
        if not replacing_cancelled_work:
            self.state.interruption_reason = None
        decision = Decision(
            decision_id=f"decision_{next(self.state.decision_id_sequence):06d}",
            goal=goal,
            plan=plan,
        )
        self.state.replanning_reason = None
        self._note_decision_source(DECISION_SOURCE_EVALUATOR, goal, plan)
        self._last_decision_at = self._clock()
        return self._begin_step()

    # ---------------------------------------------------- initiative (Phase 27A)
    def _initiative_tick(
        self,
        chat: ChatMessage | None,
        replacing_cancelled_work: bool,
    ) -> BrainTickResult | None:
        """Fill one idle Brain tick with an autonomous activity (PHASE 27A).

        Returns ``None`` when autonomy is out of scope for this tick, in which
        case the legacy decision path continues exactly as before. Otherwise
        the single Brain loop either dispatches a registered skill through the
        existing SkillExecutor -> plan -> action path, or records an explicit
        WAIT. Either way the decision lives in ``state.current_goal`` and is
        continued across ticks instead of being rebuilt every tick.
        """
        now = self._clock()
        eligible, reason = self._initiative_eligibility(chat)
        if not eligible:
            # The world (or the configuration) took autonomy out of scope: a
            # WAIT that is still current has been superseded, and the legacy
            # evaluator owns this tick.
            self._retire_wait_goal()
            if reason in INITIATIVE_OUT_OF_SCOPE_REASONS:
                self._release_initiative_evidence()
            return None

        if not self._initiative_interval_elapsed(now):
            return self._initiative_wait("initiative_interval", now)

        # PHASE 27B: resolve the caller-read context against the snapshot once,
        # so the diagnostics record exactly the verdicts the evaluator judged
        # (including the world fallback) instead of a pre-merge guess.
        context = resolve_initiative_context(self.world, self._initiative_context())
        self._record_initiative_context(context)
        decision = self.goal_evaluator.evaluate_initiative(
            self.world,
            availability=self._initiative_availability,
            recent_activities=tuple(self.state.recent_autonomy_activities),
            context=context,
        )
        self._record_initiative_evidence(decision)
        if decision.is_wait:
            return self._initiative_wait(decision.reason, now)
        return self._initiative_execute(decision, now, replacing_cancelled_work)

    def _record_initiative_context(self, context: InitiativeContext) -> None:
        """Keep the bounded decision-time context for diagnostics (PHASE 27B).

        A bounded summary only: counts and tokens, never transcripts, tokens or
        credentials. It is what the Brain *read* for this decision, so the
        evidence behind an autonomous choice stays explainable afterwards.
        """
        with_knowledge = sorted(context.knowledge)[:MAX_INITIATIVE_CANDIDATES_IN_STATE]
        # The verified evidence behind those skills, exactly as the existing
        # ``SkillKnowledge`` record measured it. Carried so an operator can see
        # *why* a contribution came out the size it did -- counts and the
        # ranking score, never transcripts.
        evidence: list[dict[str, Any]] = []
        for skill_id in with_knowledge:
            record = context.knowledge[skill_id]
            evidence.append(
                {
                    "skill_id": str(skill_id)[:64],
                    "trials": int(record.trials),
                    "successes": int(record.successes),
                    "failures": int(record.failures),
                    "ranking_score": round(float(record.score), 4),
                    "reliability": round(record.reliability, 4),
                }
            )
        self.state.initiative_context = {
            "stream_lifecycle": str(context.stream_lifecycle),
            "camera_available": context.camera_available,
            "phone_state": str(context.phone_state)[:32],
            "behavior_state": str(context.behavior_state)[:32],
            "preferred_skills": list(context.preferred_skills[:4]),
            "knowledge_skills": with_knowledge,
            "knowledge_evidence": evidence,
            # The exact context key the existing SkillLearningService was read
            # with for this decision; it is the key the outcome is journaled
            # under, so the evidence behind a choice stays traceable.
            "context_key": str(self._active_context_key or "")[:160],
        }

    def _record_initiative_evidence(self, decision: InitiativeDecision) -> None:
        """Keep the bounded candidate evidence behind one decision (PHASE 27B)."""
        eligible = tuple(str(item) for item in decision.eligible)[
            :MAX_INITIATIVE_CANDIDATES_IN_STATE
        ]
        self.state.eligible_candidates = eligible
        self.state.candidate_activity = eligible[0] if eligible else None
        self.state.candidate_rejections = tuple(
            (str(activity), str(code)) for activity, code in decision.rejections
        )[:MAX_INITIATIVE_CANDIDATES_IN_STATE]
        self.state.candidate_scores = tuple(
            (str(activity), float(score)) for activity, score in decision.scores
        )[:MAX_INITIATIVE_CANDIDATES_IN_STATE]
        self.state.knowledge_contributions = tuple(
            (str(activity), float(contribution))
            for activity, contribution in decision.knowledge_contributions
        )[:MAX_INITIATIVE_CANDIDATES_IN_STATE]

    def _release_initiative_evidence(self) -> None:
        """Drop decision-time evidence once the world leaves autonomy's scope.

        ``selected_activity``/``initiative_reason``/``last_outcome`` stay: they
        name the last decision and its verified end. Only the evidence that
        claims to describe the *current* decision round is released, so a
        stopped stream is never explained by a stale ``live`` context.
        """
        self.state.initiative_context = None
        self.state.eligible_candidates = ()
        self.state.candidate_activity = None
        self.state.candidate_rejections = ()
        self.state.candidate_scores = ()
        self.state.knowledge_contributions = ()

    def _initiative_eligibility(self, chat: ChatMessage | None) -> tuple[bool, str]:
        """Whether this tick may be spent on an autonomous activity.

        Bounded and conservative, checked in the order the spec asks for:
        configuration, capability, audience, viewer priority, physical
        availability, active speech, and the goal already in flight. A refusal
        never cancels anything; it only keeps the legacy decision path
        authoritative for this tick.
        """
        if not getattr(self.config, "autonomy_enabled", True):
            return False, "autonomy_disabled"
        if self.skill_executor is None:
            return False, "skills_unavailable"
        world = self.world
        if world is None:
            return False, "no_world"
        if not world.stream_active or world.viewer_count <= 0:
            # Nobody is watching: an unattended character has no initiative to
            # take, and stream maintenance/exploration keep owning the tick.
            return False, "no_audience"
        if chat is not None:
            return False, "viewer_message_pending"
        queue = getattr(self.chat, "queue", None)
        peek = getattr(queue, "peek", None)
        if callable(peek):
            try:
                queued = peek()
            except Exception:
                return False, "queue_unavailable"
            if queued is not None:
                # An unrecognised viewer message still owes an answer through
                # the legacy chat path; autonomy must never answer it.
                return False, "viewer_message_pending"
        action = str(world.current_action).strip().casefold()
        if action not in INITIATIVE_IDLE_ACTIONS:
            return False, f"action_in_progress:{action}"
        if self._speech_active():
            return False, "speech_active"
        goal = self.state.current_goal
        if goal is not None and goal.status is not GoalStatus.ACTIVE:
            # A terminal goal is not "in flight"; treat it as absent so a
            # stale reference can never permanently block autonomy.
            goal = None
        if goal is not None:
            if not self._is_initiative_goal(goal):
                return False, "goal_in_progress"
            if not self._is_wait_goal(goal):
                # A live initiative activity already owns the tick; never
                # start a competing one on top of it.
                return False, "initiative_in_flight"
        return True, "eligible"

    def _initiative_availability(self, skill_id: str, parameters: Mapping[str, Any]) -> Any:
        """The narrowest availability view the selector may read.

        This is the existing ``SkillExecutor.availability`` -- the same gate
        ``create_plan`` re-validates against -- so the selector can never
        choose something the executor would then refuse. The probe is a filter
        only; ``create_plan`` remains the single execution authority.
        """
        executor = self.skill_executor
        world = self.world
        if executor is None or world is None:
            return "SKILL_UNAVAILABLE"
        validation = executor.availability(
            skill_id,
            world,
            dict(parameters),
            now=self._clock(),
        )
        if getattr(validation, "valid", False):
            return None
        return str(getattr(validation, "reason", "SKILL_UNAVAILABLE"))

    # ------------------------------------------- initiative context (Phase 27B)
    def _initiative_context(self) -> InitiativeContext:
        """The bounded context one initiative decision reads (PHASE 27B).

        Every value is read from an existing authoritative source at decision
        time -- the world snapshot, the streaming service read-model, the
        behaviour manager's advisory context and the existing
        ``SkillLearningService`` ranking -- and nothing is cached between
        ticks. A source that cannot answer leaves its field unknown instead of
        inventing a verdict.
        """
        world = self.world
        behavior_state = ""
        preferred_skills: tuple[str, ...] = ()
        advisory = self.last_behavior_context
        if isinstance(advisory, Mapping):
            reported = advisory.get("state")
            if isinstance(reported, str):
                behavior_state = reported.strip()
            skills = advisory.get("preferred_skills")
            if isinstance(skills, (list, tuple, set, frozenset)):
                preferred_skills = tuple(
                    str(item).strip().casefold()
                    for item in skills
                    if isinstance(item, str) and item.strip()
                )
        if not behavior_state:
            behavior_state = self._behavior_state_name()
        return InitiativeContext(
            stream_lifecycle=self._streaming_lifecycle(),
            camera_available=camera_available_from_world(world),
            phone_state=str(getattr(world, "phone_state", "") or ""),
            behavior_state=behavior_state,
            preferred_skills=preferred_skills,
            knowledge=self._initiative_knowledge(world),
        )

    def _behavior_state_name(self) -> str:
        """The existing behaviour manager's current state, or an empty string."""
        manager = self.behavior
        state = getattr(manager, "state", None)
        if state is None:
            return ""
        value = getattr(state, "value", state)
        return str(value) if isinstance(value, str) else ""

    def _streaming_lifecycle(self) -> str:
        """The real streaming service's lifecycle, or ``unknown``.

        Read-only: the streaming service already owns and exposes this state
        for diagnostics, and the Brain only reads the same read-model. Any
        fault or absent service degrades to ``unknown``, so missing data is
        never read as "the stream is off".
        """
        streaming = self.streaming
        if streaming is None:
            return STREAM_LIFECYCLE_UNKNOWN
        raw: Any = None
        try:
            reader = getattr(streaming, "get_state", None)
            payload = reader() if callable(reader) else None
            if isinstance(payload, Mapping):
                raw = payload.get("state")
            if raw is None:
                raw = getattr(streaming, "state", None)
        except Exception:
            return STREAM_LIFECYCLE_UNKNOWN
        if isinstance(raw, Mapping):
            raw = raw.get("state")
        return normalize_stream_lifecycle(getattr(raw, "value", raw))

    def _initiative_knowledge(
        self,
        world: WorldState | None,
    ) -> dict[str, InitiativeKnowledge]:
        """Verified SkillKnowledge for the skills this tick could choose.

        Reuses the existing ``SkillLearningService.rank_candidates`` against
        the context key this decision will itself be recorded under, with
        ``choose=False`` so ranking never mutates selection bookkeeping. A
        missing, disabled or faulty service yields no evidence at all: absence
        of a record is never turned into a success or a failure.
        """
        learning = self.learning
        # The legacy ``_active_context_key`` still describes the *previous*
        # goal when this gate runs (the initiative branch precedes the ranking
        # context update), so reading it here would look up evidence for an
        # unrelated context. Compute the key this autonomous decision will be
        # recorded under instead, and keep the existing field in sync so the
        # completion path journals the outcome to exactly the same key.
        context_key = self._initiative_context_key(learning, world)
        if learning is None or not context_key or world is None:
            return {}
        skill_ids = self.goal_evaluator.initiative_skill_ids(world)
        if not skill_ids:
            return {}
        ranker = getattr(learning, "rank_candidates", None)
        if not callable(ranker):
            return {}
        try:
            ranked = ranker(context_key, list(skill_ids), choose=False)
        except Exception:
            return {}
        knowledge: dict[str, InitiativeKnowledge] = {}
        for item in ranked or ():
            if not isinstance(item, Mapping):
                continue
            skill_id = str(item.get("skill_id") or "")
            record = item.get("knowledge")
            if not skill_id or not isinstance(record, Mapping):
                # No stored record: there is no evidence to reason from.
                continue
            try:
                score = float(item.get("score"))
            except (TypeError, ValueError):
                continue
            knowledge[skill_id] = InitiativeKnowledge(
                score=score,
                trials=_bounded_count(record.get("trials")),
                successes=_bounded_count(record.get("successes")),
                failures=_bounded_count(record.get("failures")),
            )
        return knowledge

    def _initiative_context_key(self, learning, world: WorldState | None) -> str | None:
        """The existing learning context key for this autonomous decision.

        The key is built through the same ``SkillLearningService`` context
        machinery the rest of the Brain uses, with the same ``EXPLORE_ROOM``
        goal type ``initiative_goal`` assigns, so the evaluator reads and the
        outcome recorder writes the very same context. It is read-only apart
        from the service's own last-context bookkeeping.
        """
        if learning is None or world is None:
            return None
        probe = Goal(
            goal_id="initiative_context_probe",
            goal_type=GoalType.EXPLORE_ROOM,
            priority=0.0,
            reason="context probe",
        )
        try:
            key = learning.observe_context(probe, None, world, None)
        except Exception:
            return None
        key = str(key or "").strip()
        if not key:
            return None
        self._active_context_key = key
        return key

    def _speech_active(self) -> bool:
        """Whether an utterance is in flight or queued right now."""
        speech = self.speech
        if speech is None:
            return False
        try:
            state = speech.get_state()
        except Exception:
            # An unreadable speech state is treated as busy: autonomy must not
            # start a physical activity on top of a possible utterance.
            return True
        if not isinstance(state, Mapping):
            return True
        active = str(state.get("state") or "IDLE").strip().upper()
        if active not in {"", "IDLE"}:
            return True
        try:
            return int(state.get("queue_length") or 0) > 0
        except (TypeError, ValueError):
            return True

    def _initiative_interval_elapsed(self, now: datetime) -> bool:
        """Reuse the existing pacing idea: at most one new activity per window."""
        last = self.state.last_initiative_at
        if last is None:
            return True
        try:
            elapsed = (now - last).total_seconds()
        except Exception:
            return True
        return elapsed >= float(self.config.initiative_interval_seconds)

    def _legacy_explore_cycle_ready(self) -> bool:
        """Whether a completed legacy exploration may hand over one tick.

        True only while the completion flag is set and no legacy explore plan
        is still in flight. ``_complete_goal`` sets the flag when the goal the
        marker named finishes; a cancellation clears both, and the handle is
        cleared once the initiative gate has had its single attempt. Each
        completed exploration therefore releases at most one initiative
        attempt even when the world signature is unchanged.
        """
        if not self.state.explore_completed_since_initiative:
            return False
        goal = self.state.current_goal
        marker = self.state.legacy_explore_goal_id
        if marker and goal is not None and goal.goal_id == marker:
            return False
        return True

    def _initiative_execute(
        self,
        decision: InitiativeDecision,
        now: datetime,
        replacing_cancelled_work: bool,
    ) -> BrainTickResult:
        """Dispatch one selected activity through the normal production path."""
        self._retire_wait_goal()
        goal = self.goal_evaluator.initiative_goal(decision, now=now)
        self.state.last_initiative_at = now
        self.state.selected_activity = decision.activity
        self.state.initiative_reason = decision.reason
        self.state.current_goal = goal
        self.state.active_message = None
        self.state.spatial_desired_state = None
        result = None
        failure = "SKILL_UNAVAILABLE"
        executor = self.skill_executor
        if executor is not None and self.world is not None:
            selection = SkillSelection(
                skill_id=str(decision.skill_id),
                parameters=dict(decision.parameters),
                reason=decision.reason,
            )
            try:
                # No provider call: the activity was chosen deterministically,
                # and the executor is the single place that expands it into the
                # same validated plan a viewer request would produce.
                result = executor.create_plan(selection, goal, self.world, now=now)
            except Exception:
                result = None
                failure = "EXECUTOR_ERROR"
        plan = getattr(result, "plan", None)
        if plan is None:
            failure = str(getattr(result, "reason", failure) or failure)
            reason = f"initiative_unavailable:{failure}"
            goal.status = GoalStatus.FAILED
            self._record_initiative_outcome(goal, "failed")
            # A failed initiative is over: keep it out of ``current_goal`` so
            # diagnostics report no live goal and the next idle tick may
            # re-evaluate (with the failure journalled for repetition).
            self.state.current_goal = None
            self.state.current_plan = None
            self.state.pending_action = None
            self.state.step_index = 0
            self.state.retries = 0
            self.state.status = BrainStatus.WAITING
            self.state.last_failure = reason
            self.state.initiative_reason = reason
            self._clear_decision_source()
            self._last_decision_at = now
            return BrainTickResult(BrainStatus.WAITING, goal, None, None, "autonomy_wait")

        self.state.status = BrainStatus.PLANNING
        self.state.current_plan = plan
        self.state.plan_version = plan.version
        self.state.step_index = 0
        self.state.retries = 0
        self.state.last_failure = None
        self.state.active_gesture = None
        self._coordinated_gesture_step = None
        self._plan_skill_id = str(decision.skill_id)
        if not replacing_cancelled_work:
            self.state.interruption_reason = None
        self.state.replanning_reason = None
        self._note_decision_source(DECISION_SOURCE_INITIATIVE, goal, plan)
        self._last_decision_at = now
        return self._begin_step()

    def _initiative_wait(self, reason: str, now: datetime) -> BrainTickResult:
        """Record an explicit WAIT and keep the same goal until it changes."""
        goal = self._current_wait_goal()
        if goal is None:
            decision = InitiativeDecision(
                activity=INITIATIVE_WAIT,
                reason=reason,
                signature=initiative_signature(self.world),
            )
            goal = self.goal_evaluator.initiative_goal(decision, now=now)
        else:
            # The goal persists; only the reason for the wait moves with the
            # world, so diagnostics never report a stale explanation.
            goal.reason = reason
        self.state.current_goal = goal
        self.state.current_plan = None
        self.state.pending_action = None
        self.state.step_index = 0
        self.state.retries = 0
        self.state.status = BrainStatus.WAITING
        self.state.active_message = None
        self.state.spatial_desired_state = None
        self.state.selected_activity = INITIATIVE_WAIT
        self.state.initiative_reason = reason
        # A WAIT is still this loop's own initiative decision, so the
        # provenance must not make an autonomous wait look like injected state.
        self._note_decision_source(DECISION_SOURCE_INITIATIVE, goal, None)
        self._last_decision_at = now
        return BrainTickResult(BrainStatus.WAITING, goal, None, None, "autonomy_wait")

    def _reconcile_stale_initiative_goal(self) -> bool:
        """PHASE 27B: release an autonomous goal the world no longer supports.

        The check reuses the evaluator's own candidate definitions and hard
        context rules, so an activity can never be started under one verdict
        and resumed under another (the stream moved to a state the activity
        declared it needs, the phone went away, the camera was switched off,
        or the premise the activity was chosen for is gone). Recognised goals
        only: a legacy plan keeps its Phase 25/26 lifecycle untouched, and an
        explicit WAIT has nothing to invalidate.
        """
        goal = self.state.current_goal
        if goal is None or goal.status is not GoalStatus.ACTIVE:
            return False
        if not self._is_initiative_goal(goal) or self._is_wait_goal(goal):
            return False
        if self.world is None:
            return False
        reason = self.goal_evaluator.initiative_invalidation(
            self.world,
            goal,
            context=self._initiative_context(),
        )
        if reason is None:
            return False
        # The existing cancellation machinery releases the plan, journals the
        # invalidation through the initiative outcome path and returns the loop
        # to REPLANNING; dropping the decision clock lets this very tick
        # re-decide against the world that invalidated the goal.
        self._cancel_plan(reason, outcome="invalidated")
        self.state.last_failure = f"initiative_invalidated:{reason}"
        self._last_decision_at = None
        return True

    @staticmethod
    def _is_initiative_goal(goal: Goal | None) -> bool:
        return initiative_goal_context(goal) is not None

    def _is_wait_goal(self, goal: Goal | None) -> bool:
        context = initiative_goal_context(goal)
        return context is not None and context.get("activity") == INITIATIVE_WAIT

    def _current_wait_goal(self) -> Goal | None:
        goal = self.state.current_goal
        if goal is None or goal.status is not GoalStatus.ACTIVE:
            return None
        if not self._is_wait_goal(goal):
            return None
        return goal

    def _retire_wait_goal(self) -> None:
        """Retire a current WAIT goal whose reason no longer holds.

        A wait is not an activity outcome, so it is never journalled for
        repetition protection; it only stops being the current goal.
        """
        goal = self._current_wait_goal()
        if goal is None:
            return
        goal.status = GoalStatus.CANCELLED
        if self.state.current_goal is goal:
            self.state.current_goal = None
        self.state.legacy_explore_goal_id = None
        self.state.explore_completed_since_initiative = False

    def _record_initiative_outcome(self, goal: Goal | None, outcome: str) -> None:
        """Journal the terminal outcome of a real autonomous activity."""
        context = initiative_goal_context(goal)
        if context is None:
            return
        activity = str(context.get("activity") or "").strip()
        if not activity or activity == INITIATIVE_WAIT:
            # WAIT is a live decision, not a performed activity.
            return
        signature = str(context.get("signature") or "")
        self.state.recent_autonomy_activities.append((activity, signature))
        self.state.last_outcome = f"{activity}:{outcome}"

    def _preempted_maintenance_goal(self) -> Optional[Goal]:
        """The stream-maintenance goal when it must bypass chat (PHASE 24).

        The evaluator already owns the decision and the deliberate-stop latch
        through ``social_context``; this only runs that condition before the
        tick's social/chat work so a saturated attention queue cannot starve
        stream recovery. Returns ``None`` when the stream is up, the queue is
        empty, or a deliberate stop is on record.
        """
        if self.world is None or self.world.stream_active:
            return None
        social_context = self.social.context if self.social else None
        if not self.goal_evaluator.stream_maintenance_required(self.world, social_context):
            return None
        if self.chat.queue.peek() is None:
            return None
        candidate = self.goal_evaluator.evaluate(
            self.world,
            None,
            social_context=social_context,
        )
        if candidate is None or candidate.goal_type is not GoalType.MAINTAIN_STREAM:
            return None
        return candidate

    def _retrieve_memories(self, goal: Goal, chat: ChatMessage | None) -> list | None:
        if self.memory is None:
            return None
        query = MemoryQuery(
            viewer_id=chat.viewer_id if chat else None,
            goal=goal.goal_type.value,
            intent=chat.intent.value if chat else None,
            action=chat.target if chat else None,
            text=chat.normalized_text if chat else None,
        )
        return self.memory.retrieve(query)

    def _personalized_context(
        self,
        chat: ChatMessage | None,
    ) -> PersonalizedContext | None:
        """One bounded personalization block per real decision.

        The greeting suggestion reuses the existing social speech cooldown so
        relationship logic can never bypass the social attention system. Any
        memory fault degrades to generic context; the Brain keeps working.
        """
        if self.memory is None:
            return None
        greeting_allowed = True
        if self.social is not None:
            try:
                greeting_allowed = self.social.conversation.can_speak(
                    self._clock(),
                    self.social.context.last_speech_at,
                )
            except Exception:
                greeting_allowed = True
        try:
            return self.memory.personalize(
                chat.viewer_id if chat else None,
                message_text=chat.raw_text if chat else None,
                intent=chat.intent.value if chat else None,
                greeting_allowed=greeting_allowed,
                personality=self.personality,
            )
        except Exception:
            # Personalization is advisory; a fault must never stop the Brain.
            return None

    def _ranking_context(
        self,
        context_key: str | None,
        personalization: PersonalizedContext | None,
    ) -> list | None:
        if self.learning is None or not context_key:
            return None
        try:
            return self.learning.ranking_prompt_context(
                context_key,
                personalization=personalization,
                personality=self.personality,
            )
        except Exception:
            try:
                return self.learning.ranking_prompt_context(context_key)
            except Exception:
                return None

    def _update_behavior(
        self,
        goal: Goal,
        chat: ChatMessage | None,
        memories: list | None,
    ) -> dict[str, object]:
        """Derive the bounded behavior state once per real decision.

        The state is advisory context for existing planning. It never plans,
        selects, or executes anything on its own.
        """
        try:
            context = self.behavior.update(
                self.world,
                goal,
                social_context=self.social.context if self.social else None,
                memory_context=memories,
                chat=chat,
                now=self._clock(),
            )
        except Exception:
            # Behavior is advisory; a fault must never stop the Brain loop.
            context = self.behavior.advisory_context(goal)
        self.last_behavior_context = context
        return context

    # ------------------------------------------------- expression (Phase 26B)
    def _expression_decision(
        self,
        intent,
        personalization: PersonalizedContext | None = None,
    ) -> ExpressionDecision | None:
        """One bounded expression decision for the current interaction.

        Deterministic and side-effect free: it reads the intent the existing
        chat pipeline already extracted, the behavior state, the stable
        personality profile, the Phase 19 relationship context and the current
        reported activity. It never calls a provider, never plans and never
        executes. Any fault degrades to "no expression decision" so the Brain
        loop keeps its previous behaviour.
        """
        behavior_context = self.last_behavior_context or {}
        try:
            return self.expression.decide(
                intent=intent,
                behavior_state=self.behavior.state,
                behavior_reason=self._behavior_reason(behavior_context),
                personality=self.personality,
                personalization=personalization,
                current_action=(
                    self.world.current_action if self.world is not None else None
                ),
                attention_allowed=bool(behavior_context.get("attention_allowed")),
                availability=self._gesture_available,
            )
        except Exception:
            return None

    def _behavior_reason(self, behavior_context: dict[str, Any]) -> str | None:
        """The behavior transition reason, bounded to its own validity window.

        A discrete viewer reaction only counts while the existing BehaviorManager
        reaction window still considers it live, so a stale reason cannot keep
        lifting a later, unrelated turn.
        """
        reason = behavior_context.get("transition_reason")
        if not isinstance(reason, str):
            return None
        reason = reason.strip()
        if not reason:
            return None
        if reason.upper() not in {
            "POSITIVE_VIEWER_REACTION",
            "NEGATIVE_VIEWER_REACTION",
        }:
            return reason
        transitioned_at = getattr(self.behavior, "transitioned_at", None)
        if transitioned_at is None:
            return None
        window = getattr(
            getattr(self.behavior, "config", None),
            "reaction_window_seconds",
            0.0,
        )
        elapsed = (self._clock() - transitioned_at).total_seconds()
        if elapsed < 0 or elapsed > float(window):
            return None
        return reason

    def _gesture_available(self, gesture: str) -> bool:
        """Reuse the existing SkillLibrary/SkillExecutor gate for a gesture.

        A registered expressive skill that is disabled or cooling down refuses
        the gesture, so learned motion and built-in motion share one cooldown
        authority. An unregistered name is not a refusal: WAVE/DANCE are
        built-in actions Unity plays through its own animation bridge, and the
        existing plan validator plus SkillExecutor remain the execution gates.
        """
        executor = self.skill_executor
        checker = getattr(executor, "availability", None)
        world = self.world
        if not callable(checker) or world is None:
            return True
        try:
            validation = checker(gesture, world)
        except Exception:
            return False
        if getattr(validation, "valid", False):
            return True
        return getattr(validation, "reason", None) == "SKILL_NOT_FOUND"

    @staticmethod
    def _expression_advisory(
        behavior_context: Mapping[str, object] | object,
        expression: ExpressionDecision | None,
    ) -> dict[str, object]:
        """Attach the expression decision to the existing advisory block."""
        context = (
            dict(behavior_context)
            if isinstance(behavior_context, Mapping)
            else {}
        )
        if expression is not None:
            context["expression"] = expression.to_dict()
        return context

    def _expression_speech_hint(
        self,
        expression: ExpressionDecision | None,
        personalization: PersonalizedContext | None,
    ) -> Optional[str]:
        if expression is None:
            return None
        style = getattr(personalization, "speech_style", None)
        if not isinstance(style, str) or not style.strip():
            style = None
        try:
            hint = expression.speech_hint(
                style, limit=self.expression.config.speech_hint_limit
            )
        except Exception:
            return None
        hint = hint.strip()
        return hint or None

    def _chat_personalization(
        self,
        personalization: PersonalizedContext | None,
        expression: ExpressionDecision | None,
    ) -> Optional[dict[str, object]]:
        """The single existing personalization block plus the expression hint.

        The Phase 19 block is copied, never mutated, and the Phase 26B hint is
        added as one bounded derived string. With no expression decision the
        payload is exactly what Phase 19 produced.
        """
        block: Optional[dict[str, object]] = None
        if personalization is not None:
            try:
                payload = personalization.to_prompt_block()
            except Exception:
                payload = None
            if isinstance(payload, Mapping):
                block = dict(payload)
        hint = self._expression_speech_hint(expression, personalization)
        if hint is None or expression is None:
            return block
        if block is None:
            block = {}
        block["expression"] = {
            "tone": expression.tone,
            "reaction_intensity": expression.reaction_intensity,
            "hint": hint,
        }
        return block

    def _coordinate_gesture(
        self,
        plan: Plan,
        goal: Goal,
        expression: ExpressionDecision | None,
    ) -> Plan:
        """Place the already-selected gesture into the existing plan.

        PHASE 26B: the expression policy -- not a scheduler -- decided that a
        bounded gesture fits this interaction. This only positions the
        already-registered motion immediately before the utterance the same
        single reasoning pass produced. It never runs when the plan already
        contains a gesture, when the plan is a task/deliberate-stop plan, or
        when there is no speech for the gesture to accompany.
        """
        if expression is None or not expression.gesture:
            return plan
        if goal.goal_type not in {
            GoalType.RESPOND_TO_VIEWER,
            GoalType.INTERACT_WITH_VIEWER,
        }:
            return plan
        action = _GESTURE_ACTION_TYPES.get(str(expression.gesture).strip().lower())
        if action is None or len(plan.steps) >= self.config.max_plan_steps:
            return plan
        actions = [step.action for step in plan.steps]
        if any(item in _GESTURE_ACTION_TYPES.values() for item in actions):
            return plan
        if any(
            item in {ActionType.STOP, ActionType.START_STREAM, ActionType.STOP_STREAM}
            for item in actions
        ):
            return plan
        speech_index = next(
            (index for index, item in enumerate(actions) if item == ActionType.SPEAK),
            None,
        )
        if speech_index is None:
            return plan
        steps = (
            plan.steps[:speech_index]
            + (PlanStep(action=action),)
            + plan.steps[speech_index:]
        )
        return replace(plan, steps=steps)

    @staticmethod
    def _plan_gesture(plan: Plan | None) -> Optional[str]:
        """The first registered expressive motion in a plan, if any."""
        if plan is None:
            return None
        for step in plan.steps:
            for name, action in _GESTURE_ACTION_TYPES.items():
                if step.action == action:
                    return name
        return None

    @staticmethod
    def _strip_unrelated_greeting_actions(plan: Plan, chat: ChatMessage | None) -> Plan:
        """Keep a bare greeting conversational instead of exploring the room.

        The deterministic classifier is exact, so a message that also contains an
        explicit supported request is not classified as a bare greeting and keeps
        its requested action.
        """
        if chat is None or chat.intent != ChatIntent.GREETING:
            return plan
        steps = tuple(
            step for step in plan.steps if step.action in _GREETING_SOCIAL_ACTIONS
        )
        if steps == plan.steps:
            return plan
        return replace(plan, steps=steps)

    @staticmethod
    def _normalize_gesture_dialogue(plan: Plan) -> Plan:
        """Give gesture-carried dialogue the existing SPEAK path.

        Unity treats WAVE/DANCE as motion only. When the planner attaches text to
        one, the Brain--not Unity--turns that text into an explicit SPEAK step.
        """
        if not plan.steps:
            return plan

        if any(step.action == ActionType.SPEAK for step in plan.steps):
            return plan

        # The gesture owns text only when no chat reply exists. Keep the gesture
        # and dispatch its text through the existing SPEAK action after it.
        normalized = []
        inserted = False
        for step in plan.steps:
            normalized.append(step)
            gesture_text = step.text.strip() if isinstance(step.text, str) else ""
            if (
                not inserted
                and step.action in {ActionType.WAVE, ActionType.DANCE}
                and gesture_text
                and len(gesture_text) <= MAX_TEXT_LENGTH
            ):
                normalized.append(PlanStep(action=ActionType.SPEAK, text=gesture_text))
                inserted = True
        return replace(plan, steps=tuple(normalized)) if inserted else plan

    def _publish_authoritative_response(self, plan: Plan, processed) -> None:
        """Make the visible turn agree with the one line the plan will speak."""
        if processed is None:
            return
        spoken = next(
            (step.text for step in plan.steps if step.action == ActionType.SPEAK),
            None,
        )
        if not isinstance(spoken, str) or not spoken.strip():
            return
        try:
            self.chat.set_last_response(spoken)
        except Exception:
            # Response publication must never block dispatch or planning.
            pass

    def current_skill_id(self) -> str | None:
        executor = self.skill_executor
        active = executor.active if executor is not None else None
        return active.skill_id if active is not None else None

    def _memory_world_context(self) -> dict[str, object]:
        if self.world is None:
            return {}
        return {
            "posture": self.world.character_posture,
            "current_action": self.world.current_action,
            "phone_state": self.world.phone_state,
            "stream_active": self.world.stream_active,
        }

    def _begin_step(self) -> BrainTickResult:
        plan = self.state.current_plan
        if plan is None or self.state.step_index >= len(plan.steps):
            self._complete_goal()
            return BrainTickResult(BrainStatus.IDLE, None, None, None, "plan_complete")

        step = plan.steps[self.state.step_index]
        blocked = self._deliberate_stop_conflict((step.action,))
        if blocked is not None:
            # A stop latched after the plan was validated (an operator stop while
            # the step was retried) must not be dispatched again.
            self._cancel_plan(blocked)
            self.state.last_failure = f"validation_failed:{blocked}"
            self.state.status = BrainStatus.ERROR
            return BrainTickResult(BrainStatus.ERROR, None, None, None, self.state.last_failure)
        request = ActionRequest(
            action=step.action,
            target=step.target,
            text=step.text,
            position=step.position,
            speech_request_id=self._request_speech(plan, step),
            # PHASE 34: exactly one id per accepted dispatch. Re-reading the
            # same pending request (the "awaiting_action_result" branch) reuses
            # this object, so the id is stable until a new step is dispatched.
            dispatch_id=_new_dispatch_id(),
        )
        self.state.pending_action = request
        self.state.status = BrainStatus.EXECUTING
        result = BrainTickResult(
            status=BrainStatus.EXECUTING,
            goal=self.state.current_goal,
            plan=plan,
            action=request,
            message="action_started",
        )
        # PHASE 32: emit the dispatch event only once the dispatch state and the
        # result are fully built, so diagnostics can never affect the hand-off.
        self._log_action_dispatch(request, plan)
        return result

    def _log_action_dispatch(self, request: ActionRequest, plan: Optional[Plan]) -> None:
        """One privacy-safe event per accepted target-bearing action dispatch.

        Semantics: the action was accepted by the plan validator and is being
        handed to the execution bridge as this tick's response payload
        (``BrainTickResult.action``, which becomes ``tick.action`` on the wire
        and is consumed by Unity's ``BrainActionBridge.Execute``). It does **not**
        mean Unity executed or completed the action.

        Only the three spatial actions are instrumented, and every field is a
        bounded token (or an integer index); targets that are absent or not
        identifier-shaped are omitted rather than truncated. A logging failure
        is swallowed so it cannot block or replace the dispatch.
        """
        if request.action not in _DISPATCH_INSTRUMENTED_ACTIONS:
            return
        try:
            goal = self.state.current_goal
            fields: dict[str, Any] = {
                "action": request.action.value,
                "dispatch": "tick_response",
            }
            target = _bounded_token(request.target)
            if target is not None:
                fields["target"] = target
            dispatch_id = _bounded_token(request.dispatch_id)
            if dispatch_id is not None:
                fields["dispatch_id"] = dispatch_id
            goal_id = _bounded_token(getattr(goal, "goal_id", None))
            if goal_id is not None:
                fields["goal_id"] = goal_id
            plan_id = _bounded_token(getattr(plan, "plan_id", None))
            if plan_id is not None:
                fields["plan_id"] = plan_id
            decision_source = _bounded_token(self._decision_source)
            if decision_source is not None:
                fields["decision_source"] = decision_source
            if isinstance(self.state.step_index, int) and not isinstance(
                self.state.step_index, bool
            ):
                fields["step_index"] = self.state.step_index
            log_event(_LOGGER, "brain.action_dispatch_requested", **fields)
        except Exception:  # pragma: no cover - diagnostics never block dispatch
            pass

    def _append_dialogue_step(self, plan: Plan, goal: Goal, processed) -> Plan:
        """Give a generated chat reply a voice through the existing SPEAK action.

        Dialogue and text-to-speech stay separate concerns: the chat service
        decides the words, the plan turns them into one SPEAK step. A plan that
        already speaks -- a dance sign-off, for example -- is left untouched so a
        single exchange can never queue two utterances.

        Voice only ever adds to a plan: with no usable speech service the plan
        stays exactly as PHASE 06-10 built it.
        """
        if goal.goal_type not in {GoalType.RESPOND_TO_VIEWER, GoalType.INTERACT_WITH_VIEWER}:
            return plan
        if not (self.speech is not None and getattr(self.speech, "enabled", False)):
            return plan
        if processed is None or len(plan.steps) >= self.config.max_plan_steps:
            return plan
        if any(step.action == ActionType.SPEAK for step in plan.steps):
            return plan
        speech = getattr(processed.response, "speech", None)
        text = speech.strip() if isinstance(speech, str) else ""
        if not text or len(text) > MAX_TEXT_LENGTH:
            return plan
        return replace(
            plan,
            steps=plan.steps + (PlanStep(action=ActionType.SPEAK, text=text),),
        )

    def _request_speech(self, plan: Plan, step: PlanStep) -> Optional[str]:
        """Give a SPEAK step to the speech service without ever blocking the Brain.

        Returns the speech request id for Unity to play, or None when speech is
        unavailable, in which case the action still runs and the loop continues.
        """
        if self.speech is None or step.action != ActionType.SPEAK:
            return None
        text = (step.text or "").strip()
        if not text or not getattr(self.speech, "enabled", False):
            return None
        try:
            # One utterance at a time: an older line never survives a new one.
            self.speech.interrupt_current("new_speech")
            digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
            speech = self.speech.request(
                text,
                source="brain",
                dedup_key=f"{plan.plan_id}:{self.state.step_index}:{digest}",
                plan_id=plan.plan_id,
                step_index=self.state.step_index,
            )
            return speech.request_id
        except Exception:
            # Voice is additive; a speech fault must never stop the autonomous loop.
            return None

    def _cancel_speech(self, reason: str) -> None:
        if self.speech is None:
            return
        pending = self.state.pending_action
        try:
            if pending is not None and pending.action == ActionType.SPEAK and pending.speech_request_id:
                self.speech.cancel(pending.speech_request_id, reason)
            else:
                self.speech.interrupt_current(reason)
        except Exception:
            pass

    def _observe_result(self, result: ActionResult) -> None:
        self.state.last_result = result
        self._record_behavior_outcome(result)
        if self.social is not None:
            try:
                self.social.update_on_result(
                    result,
                    self._clock(),
                    self.state.active_message.viewer_id if self.state.active_message else None,
                )
            except Exception:
                pass
        if result.status in {ExecutionStatus.SUCCESS}:
            # PHASE 25: a reported success is a claim, not an observation. For
            # a plan the deterministic spatial branch produced, the physical
            # condition is measured against the Unity state reported after the
            # action; a failed or unmeasurable check never advances the plan.
            verification_state, verification_reason = self._verify_spatial_result(result)
            if verification_state in {"FAILED", "UNVERIFIED"}:
                self.state.pending_action = None
                self.state.retries += 1
                self.state.last_failure = (
                    f"spatial_verification_failed:{verification_reason}"
                )
                if self.state.retries <= self.config.max_retries:
                    self.state.status = BrainStatus.REPLANNING
                else:
                    # PHASE 27B: exhausting the retry budget is a real failure,
                    # not a plain cancellation, and the journal says so.
                    self._cancel_plan("spatial_verification_failed", outcome="failed")
                return
            completed_final_step = (
                self.state.current_plan is not None
                and self.state.step_index + 1 >= len(self.state.current_plan.steps)
            )
            self.state.pending_action = None
            self.state.retries = 0
            self.state.step_index += 1
            self.state.status = BrainStatus.THINKING
            self._record_successful_action(result, completed_final_step)
            if self.skill_executor is not None:
                evaluation = self.skill_executor.observe_success(result)
                if self.learning is not None and evaluation is not None:
                    self._record_learning(evaluation, context_key=self._active_context_key)
                if not self._has_replannable_step() and self.state.current_plan is not None:
                    self._complete_goal()
                    return BrainTickResult(BrainStatus.IDLE, None, None, None, "plan_complete")
            return

        if result.status == ExecutionStatus.REJECTED:
            if self.skill_executor is not None:
                self.skill_executor.observe_failure(result, final=True)
                if self.learning is not None and self.skill_executor.evaluations:
                    self._record_learning(
                        self.skill_executor.evaluations[-1],
                        context_key=self._active_context_key,
                    )
            if self._rejected_gesture_keeps_the_utterance(result):
                # PHASE 26B: the coordinated gesture is decoration for the
                # reply the same reasoning pass produced. When Unity rejects
                # it, the answer is still owed to the viewer: the failed step
                # is skipped (the rejected outcome was already recorded above,
                # so behavior reacts to the setback) and the remaining steps
                # run. A gesture the plan itself asked for keeps the legacy
                # reject-and-replan path below.
                self.state.pending_action = None
                self.state.retries = 0
                self.state.active_gesture = None
                self.state.step_index += 1
                self.state.status = BrainStatus.THINKING
                return
            self._cancel_plan("action_rejected")
            return

        self.state.retries += 1
        self._remember_failed_action(result)
        if self.state.retries <= self.config.max_retries:
            self.state.status = BrainStatus.REPLANNING
            return

        # PHASE 27B: the retry budget is gone; the activity failed. A terminal
        # failure is a verified outcome, so it flows through the same existing
        # observation/learning path a rejection already uses: the executor
        # closes the execution with a FAILED evaluation and the existing
        # SkillLearningService records it under the context this goal was
        # chosen in. Without this, verified failures never became knowledge and
        # the evaluator's RECENT_FAILURE rule could never see them.
        if self.skill_executor is not None:
            self.skill_executor.observe_failure(result, final=True)
            if self.learning is not None and self.skill_executor.evaluations:
                self._record_learning(
                    self.skill_executor.evaluations[-1],
                    context_key=self._active_context_key,
                )
        self._cancel_plan("action_failed", outcome="failed")

    def _rejected_gesture_keeps_the_utterance(self, result: ActionResult) -> bool:
        """PHASE 26B: a rejected decoration must not cost the reply.

        Only the exact step the expression policy inserted is allowed to be
        skipped, and only while the utterance it decorates is still ahead in
        the same plan.
        """
        plan = self.state.current_plan
        if plan is None or result.action not in _GESTURE_ACTION_TYPES.values():
            return False
        if self._coordinated_gesture_step != (plan.plan_id, self.state.step_index):
            return False
        if not any(
            step.action == ActionType.SPEAK
            for step in plan.steps[self.state.step_index + 1 :]
        ):
            return False
        return self.speech is not None and getattr(self.speech, "enabled", False)

    def _verify_spatial_result(self, result: ActionResult) -> tuple[str, Optional[str]]:
        """PHASE 25: confirm a marked spatial step against real reported state.

        A spatial plan is only ever built from transforms Unity already
        reported, so the same transforms must confirm the outcome. The action
        result alone never proves the physical condition: MOVE_TO can report
        success while Lacrimosa stands beside the anchor, and LOOK_AT returns
        the moment the request is issued.

        Returns ``("SKIPPED", None)`` for every plan the deterministic spatial
        branch did not produce, so legacy planning keeps its exact behaviour.
        """
        plan = self.state.current_plan
        if plan is None or not is_spatial_plan(plan):
            return "SKIPPED", None

        anchor = plan_anchor(plan)
        step = (
            plan.steps[self.state.step_index]
            if 0 <= self.state.step_index < len(plan.steps)
            else None
        )
        if step is None or step.action not in _SPATIAL_VERIFIABLE_ACTIONS:
            # An appended dialogue step (SPEAK) has no physical condition to
            # measure. The last real spatial measurement stays in diagnostics
            # instead of being overwritten with a meaningless UNVERIFIED
            # entry that would read as if the spatial outcome were unknown.
            return "SKIPPED", None
        world = self.world
        relationship = (
            spatial_relationship(world, target=anchor) if world is not None else None
        )
        desired = (self.state.spatial_desired_state or "").upper()
        expected: dict[str, Any] = {}
        if desired == DESIRED_IN_FRAME:
            expected["inside_frame"] = True
        elif desired in {DESIRED_CLOSER, DESIRED_NEAR}:
            expected["position_change_required"] = False
        else:
            expected["max_orientation_error_degrees"] = ORIENTATION_TOLERANCE_DEGREES
        record: dict[str, Any] = {
            "action": result.action.value,
            "target": step.target if step is not None else None,
            "anchor": anchor,
            "desired_state": self.state.spatial_desired_state,
            "expected": expected,
            "observed": (
                spatial_observation(relationship)
                if relationship is not None
                else {"transform_based": False}
            ),
            "result": "UNVERIFIED",
            "reason": "no_spatial_state",
        }
        self.state.spatial_verification = record
        observed = record["observed"]

        def finish(state: str, reason: Optional[str]) -> tuple[str, Optional[str]]:
            record["result"] = state
            record["reason"] = reason
            return state, reason

        if world is None or relationship is None or relationship.camera_position is None:
            return finish("UNVERIFIED", "no_anchor_transform")
        if anchor == TARGET_CAMERA and not relationship.transform_based:
            return finish("UNVERIFIED", "no_camera_transform")

        if result.action == ActionType.MOVE_TO:
            definition = (
                world.target(step.target)
                if step is not None and step.target
                else None
            )
            destination = definition.position if definition is not None else None
            arrival = distance(world.character_position, destination)
            expected["max_arrival_distance_meters"] = POSITION_ARRIVAL_TOLERANCE_METERS
            observed["distance_to_target"] = (
                round(arrival, 3) if arrival is not None else None
            )
            if arrival is None:
                return finish("UNVERIFIED", "no_position_evidence")
            if arrival > POSITION_ARRIVAL_TOLERANCE_METERS:
                return finish("FAILED", f"position_not_reached:{arrival:.2f}m")
            return finish(*self._verify_spatial_framing(relationship))

        if result.action in {ActionType.TURN_TO, ActionType.LOOK_AT}:
            error = relationship.orientation_error_degrees
            if error is None:
                return finish("UNVERIFIED", "no_orientation_evidence")
            if error > ORIENTATION_TOLERANCE_DEGREES:
                return finish("FAILED", f"orientation_error:{error:.1f}deg")
            return finish(*self._verify_spatial_framing(relationship))

        return "SKIPPED", None

    def _verify_spatial_framing(self, relationship) -> tuple[str, Optional[str]]:
        """The framing half of a spatial verification.

        FACING is satisfied by the measured orientation error alone; an
        IN_FRAME request additionally has to land inside the camera frustum,
        and a CLOSER / NEAR request must no longer be classified as needing a
        position change. A framing state that cannot be measured stays
        unverified rather than reading as success.
        """
        desired = (self.state.spatial_desired_state or "").upper()
        if desired == DESIRED_IN_FRAME:
            if relationship.inside_frame is None:
                if (
                    relationship.transform_based
                    and relationship.framing in FRAMING_PLACEMENT_ONLY
                ):
                    # The transform is real but the camera is not producing
                    # frames. Arrival and orientation were already measured by
                    # the caller, so the placement half can advance while the
                    # visibility half stays explicitly unverified.
                    return (
                        SPATIAL_PLACEMENT_VERIFIED,
                        f"framing_not_measured:{relationship.framing.lower()}",
                    )
                # No usable camera transform / direction / field of view:
                # framing cannot be measured, so nothing here reads as success.
                return "UNVERIFIED", f"framing_unknown:{relationship.framing.lower()}"
            if relationship.inside_frame is not True:
                return "FAILED", f"framing_not_achieved:{relationship.framing.lower()}"
        elif desired in {DESIRED_CLOSER, DESIRED_NEAR} and relationship.needs_position_change:
            return "FAILED", f"distance_not_improved:{relationship.framing.lower()}"
        return "VERIFIED", None

    def _record_behavior_outcome(self, result: ActionResult) -> None:
        # The active skill id is captured before SkillExecutor.observe_success
        # clears it, so diagnostics keep the skill that actually produced this
        # outcome instead of the next one.
        try:
            self.behavior.record_outcome(
                result,
                skill_id=self.current_skill_id(),
                now=self._clock(),
            )
        except Exception:
            # Behavior is advisory; a fault must never stop the Brain loop.
            pass

    def _record_learning(self, evaluation, context_key=None) -> None:
        try:
            self.learning.observe_evaluation(evaluation, context_key=context_key)
        except Exception:
            # Learning is advisory; an update fault must not disturb execution.
            pass

    def _viewer_for_context(self, chat: ChatMessage | None):
        if self.memory is None or chat is None:
            return None
        state = self.memory.get_viewer(chat.viewer_id)
        viewer = state.get("viewer") if isinstance(state, dict) else None
        if isinstance(viewer, dict):
            return ViewerMemory(
                viewer_id=viewer.get("viewer_id", chat.viewer_id),
                interaction_count=viewer.get("interaction_count", 0),
                successful_interactions=viewer.get("successful_interactions", 0),
                failed_interactions=viewer.get("failed_interactions", 0),
                relationship_score=viewer.get("relationship_score", 0.2),
                preferences=tuple(viewer.get("preferences", ())),
                recent_topics=tuple(viewer.get("recent_topics", ())),
            )
        # A first interaction must use the same relationship baseline as the
        # viewer memory created later by the outcome; otherwise the same
        # exchange would be split across two learning contexts.
        return ViewerMemory(viewer_id=chat.viewer_id)

    def _record_successful_action(self, result: ActionResult, final_step: bool) -> None:
        if self.memory is None or result.action not in {
            ActionType.SPEAK,
            ActionType.WAVE,
            ActionType.DANCE,
            ActionType.RUN,
            ActionType.PICK_UP_PHONE,
            ActionType.PLACE_PHONE,
            ActionType.START_STREAM,
            ActionType.STOP_STREAM,
        }:
            return
        if not final_step:
            return
        chat = self.state.active_message
        self.memory.record_outcome(
            viewer_id=chat.viewer_id if chat else None,
            chat=chat,
            goal=self.state.current_goal,
            action=result.action,
            success=True,
            response=None,
            context=self._memory_world_context(),
            # The plan's skill was captured when the plan was built, before
            # appended dialogue steps or SkillExecutor.observe_success cleared
            # the active execution, so viewer skill preferences come from real
            # evidence.
            skill_id=self._plan_skill_id or self.current_skill_id(),
        )

    def _remember_failed_action(self, result: ActionResult) -> None:
        if self.memory is None:
            return
        chat = self.state.active_message
        self.memory.remember_event(
            f"{result.action.value} {result.status.value}: {result.reason}",
            importance=0.4,
            viewer_id=chat.viewer_id if chat else None,
            metadata={"action": result.action.value, "status": result.status.value},
        )

    def _complete_goal(self) -> None:
        if self.state.current_goal is not None:
            # PHASE 27A: a completed autonomous activity is journalled against
            # the world signature it was chosen for, so the same activity is
            # not re-selected on the next tick without a new reason.
            self._record_initiative_outcome(self.state.current_goal, "completed")
            self.state.current_goal.status = GoalStatus.COMPLETED
            if self.state.current_goal.goal_id == self.state.legacy_explore_goal_id:
                # The exploration finished; the next explore-due decision may
                # offer one initiative before the cadence resumes exploring.
                self.state.explore_completed_since_initiative = True
        self.state.current_goal = None
        self.state.current_plan = None
        self.state.pending_action = None
        self.state.step_index = 0
        self.state.status = BrainStatus.IDLE
        self.state.active_message = None
        self.state.spatial_desired_state = None
        self.state.interruption_reason = None
        self.state.active_gesture = None
        self._coordinated_gesture_step = None
        self._plan_skill_id = None
        self.state.replanning_reason = None
        self._clear_decision_source()
        self._reconcile_conversation("success")

    def _has_replannable_step(self) -> bool:
        plan = self.state.current_plan
        return (
            plan is not None
            and self.state.step_index < len(plan.steps)
            and (
                self.state.step_index > 0
                or self.state.status == BrainStatus.REPLANNING
            )
        )

    def _deliberate_stop_conflict(self, actions: Iterable[ActionType]) -> str | None:
        """A deliberate STOP_STREAM outranks every autonomous restart.

        The durable latch records an explicit stop (operator control plane or
        the Brain's own ``STOP_STREAM``) and is only cleared by the control
        plane. Without this gate the planner could select the ``start_stream``
        skill for an ambiguous viewer message and silently undo that stop.
        """
        if self.social is None:
            return None
        if not self.social.context.last_intentional_stream_stop:
            return None
        if ActionType.START_STREAM not in actions:
            return None
        return "STREAM_INTENTIONALLY_STOPPED"

    def _cancel_plan(self, reason: str, outcome: str | None = None) -> None:
        """Release the active plan and return the loop to REPLANNING.

        PHASE 27B: ``outcome`` names the journal entry the cancelled activity
        gets when it was an autonomous one, so a timeout or an invalidation is
        no longer recorded as a plain cancellation. Existing callers keep the
        Phase 27A vocabulary: a viewer interruption is ``interrupted`` and
        everything else defaults to ``cancelled``.
        """
        if self.skill_executor is not None:
            self.skill_executor.cancel_active(reason)
        self._cancel_speech(reason)
        self.state.legacy_explore_goal_id = None
        self.state.explore_completed_since_initiative = False
        if self.state.current_goal is not None:
            # PHASE 27A: a cancelled autonomous activity is journalled too --
            # a viewer interruption is an explicit outcome for diagnostics.
            self._record_initiative_outcome(
                self.state.current_goal,
                outcome
                or (
                    "interrupted"
                    if reason == "interrupted_by_viewer"
                    else "cancelled"
                ),
            )
            self.state.current_goal.status = GoalStatus.CANCELLED
        self.state.current_goal = None
        self.state.current_plan = None
        self.state.pending_action = None
        self.state.step_index = 0
        self.state.status = BrainStatus.REPLANNING
        self.state.replanning_reason = str(reason)
        self.state.active_message = None
        self.state.spatial_desired_state = None
        self.state.interruption_reason = reason
        self.state.active_gesture = None
        self._coordinated_gesture_step = None
        self._plan_skill_id = None
        self._clear_decision_source()
        self._reconcile_conversation(reason)

    def _reconcile_conversation(self, reason: str) -> None:
        """PHASE 26A: never leave the chat turn stuck in RESPONDING.

        Success, failure, timeout, cancel and replan all funnel through here so
        the bounded conversation context returns to a coherent status. A
        missing/older chat service is ignored: this is reconciliation, never a
        new dependency.
        """
        conversation = getattr(self.chat, "conversation", None)
        recover = getattr(conversation, "recover", None)
        if not callable(recover):
            return
        try:
            recover(reason, now=self._clock())
        except Exception:
            # Continuity bookkeeping must never stop the autonomous loop.
            pass

    def _new_decision_too_soon(self) -> bool:
        if self._last_decision_at is None:
            return False
        elapsed = (self._clock() - self._last_decision_at).total_seconds()
        return elapsed < self.config.decision_interval_seconds

    # ------------------------------------------------- diagnostics (Phase 27B)
    def _note_decision_source(
        self,
        source: str,
        goal: Goal | None,
        plan: Plan | None,
    ) -> None:
        """Remember which path built the goal/plan the runtime now carries."""
        self._decision_source = source
        self._decision_goal_id = goal.goal_id if goal is not None else None
        self._decision_plan_id = plan.plan_id if plan is not None else None

    def _clear_decision_source(self) -> None:
        self._decision_source = None
        self._decision_goal_id = None
        self._decision_plan_id = None

    def _decision_source_for_runtime(self) -> Optional[str]:
        """Where the goal/plan in flight came from, or ``None`` when idle.

        A runtime goal or plan this loop never recorded is reported as
        ``external``: a harness-injected action can therefore never be read
        back as an actual autonomous Brain decision.
        """
        runtime = self.state
        plan = runtime.current_plan
        goal = runtime.current_goal
        if plan is None and goal is None and runtime.pending_action is None:
            return None
        if (
            plan is not None
            and self._decision_plan_id is not None
            and plan.plan_id == self._decision_plan_id
        ):
            return self._decision_source
        if (
            goal is not None
            and self._decision_goal_id is not None
            and goal.goal_id == self._decision_goal_id
        ):
            return self._decision_source
        return DECISION_SOURCE_EXTERNAL

    def _next_decision_eligibility(self) -> str:
        """The bounded token describing when the next decision may happen.

        Mirrors the exact gate order ``_tick_locked`` uses: an action in
        flight, a replannable step, the reuse of the existing decision cooldown
        (naming the initiative interval when that is what paces autonomy), or
        an immediately eligible decision.
        """
        runtime = self.state
        if runtime.pending_action is not None:
            return NEXT_DECISION_ACTION_IN_FLIGHT
        if self._has_replannable_step():
            return NEXT_DECISION_STEP_READY
        if self._new_decision_too_soon():
            if (
                runtime.status is BrainStatus.WAITING
                and runtime.initiative_reason == "initiative_interval"
            ):
                return NEXT_DECISION_INITIATIVE_INTERVAL
            return NEXT_DECISION_COOLDOWN
        return NEXT_DECISION_ELIGIBLE

    def _finalize_tick_diagnostics(self) -> None:
        """Refresh the per-tick diagnostics after one Brain tick (PHASE 27B).

        Both fields are derived from the state the tick actually left behind,
        so they describe the runtime rather than the intent of any code path.
        """
        self.state.next_decision_eligibility = self._next_decision_eligibility()
        self.state.decision_source = self._decision_source_for_runtime()
