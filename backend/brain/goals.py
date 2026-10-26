"""Dynamic goal evaluation for the autonomous loop."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from itertools import count
from typing import Any, Callable, Iterable, Mapping, Optional

from ..chat.models import ChatIntent, ChatMessage, utc_now
from ..skills.conditions import PHONE_EXISTS_STATES
from .models import Goal, GoalStatus, GoalType, PersonalityProfile, WorldState
from .spatial import (
    POSITION_ARRIVAL_TOLERANCE_METERS,
    TARGET_CAMERA,
    TARGET_CAMERA_STAND,
    TARGET_CENTER,
    distance,
)
from ..social.attention import assess_camera_framing
from ..social.context import SocialContext

#: PHASE 27A: the activity label used when no eligible initiative exists. WAIT
#: is a real, recorded decision -- never a silent fallback.
INITIATIVE_WAIT = "WAIT"

#: PHASE 27A: bounded recent-initiative journal the selector may skip against.
MAX_RECENT_INITIATIVES = 4

#: PHASE 27A: the bounded priority band an autonomous initiative is allowed to
#: use. The top of the band stays below ``BrainConfig.interruption_priority``
#: (0.7), so a viewer request always outranks an initiative through the existing
#: priority/cancellation machinery instead of through a second preemption path.
INITIATIVE_PRIORITY_BASE = 0.25
INITIATIVE_PRIORITY_CURIOSITY_WEIGHT = 0.10

#: PHASE 27A: the physical states a world snapshot must report before the
#: character counts as free. Anything else is work someone else started, and
#: starting a second activity on top of it would be a competing action.
INITIATIVE_IDLE_ACTIONS = frozenset({"", "idle", "standing", "none", "stopped"})

#: PHASE 27B: the bounded stream-lifecycle vocabulary the initiative evaluator
#: understands. ``unknown`` means the runtime did not report a lifecycle at
#: all, so missing data is never read as "the stream is off". ``starting``
#: covers the provider's CONNECTING/RECONNECTING states: a start in progress is
#: not a reason to abstain, because autonomy never issues a lifecycle operation
#: of its own -- only a stop in progress makes new stream-facing work point at
#: a teardown that has already begun.
STREAM_LIFECYCLE_UNKNOWN = "unknown"
STREAM_LIFECYCLE_INACTIVE = "inactive"
STREAM_LIFECYCLE_STARTING = "starting"
STREAM_LIFECYCLE_LIVE = "live"
STREAM_LIFECYCLE_STOPPING = "stopping"
STREAM_LIFECYCLE_ERROR = "error"

_STREAM_LIFECYCLE_ALIASES = {
    "idle": STREAM_LIFECYCLE_INACTIVE,
    "stopped": STREAM_LIFECYCLE_INACTIVE,
    "inactive": STREAM_LIFECYCLE_INACTIVE,
    "offline": STREAM_LIFECYCLE_INACTIVE,
    "connecting": STREAM_LIFECYCLE_STARTING,
    "starting": STREAM_LIFECYCLE_STARTING,
    "reconnecting": STREAM_LIFECYCLE_STARTING,
    "live": STREAM_LIFECYCLE_LIVE,
    "streaming": STREAM_LIFECYCLE_LIVE,
    "stopping": STREAM_LIFECYCLE_STOPPING,
    "error": STREAM_LIFECYCLE_ERROR,
    "failed": STREAM_LIFECYCLE_ERROR,
}

#: PHASE 27B: the stream-facing prerequisites a candidate may declare.
INITIATIVE_REQUIRES_CAMERA = "camera"
INITIATIVE_REQUIRES_PHONE = "phone"
INITIATIVE_REQUIRES_AUDIENCE = "audience"

#: PHASE 27B: bounded secondary weights for the deterministic ranking, read
#: only after every hard eligibility rule has already passed. The signed
#: knowledge band (``2 * INITIATIVE_KNOWLEDGE_WEIGHT = 0.56``) is calibrated
#: against the narrowest gap the ranking can actually present between
#: candidates backed by *different* skills: ``0.46`` -- ``look_at`` at 1.0 plus
#: the bounded behaviour preference against ``walk_to`` at 0.6, which is what
#: the live CURIOUS state and the documented table both produce. That measured
#: gap requires ``0.46 / 2 = 0.23``; the constant sits at ``0.28`` so the bound
#: holds with margin. Verified evidence may therefore decide an ordering the
#: static weights left genuinely comparable, while no record -- however strong
#: -- can leapfrog a materially more relevant activity: the narrowest
#: *material* gap (``1.0``: e.g. ``dance`` at 3.0 against ``wave`` at 2.0)
#: stays wider than the whole band plus the behaviour term (``2 * W + B =
#: 0.62``). A skill with no knowledge record scores exactly its static weight:
#: missing outcome evidence is never read as a judgement.
#:
#: The band is only usable if the normalization maps the learning model's
#: *whole* achievable score range onto it (see
#: :func:`initiative_knowledge_contribution`). Normalizing against ``0.0 ..
#: 1.0`` instead would leave most of the band unreachable -- a plain perfect
#: production row scores ``0.88`` and would move less than a quarter of the way
#: -- so two records could differ as much as the model allows and still not
#: decide a comparable pair. The two envelope constants below are the extremes
#: the existing ``SkillLearningService._ranking_score`` can actually emit for a
#: verified record; ``test_phase27b_autonomy`` pins them against that service
#: and against the live production rows.
INITIATIVE_KNOWLEDGE_WEIGHT = 0.28
#: The existing behaviour preference stays a tie-breaker even next to the
#: knowledge band: worst case the two bounded terms move a candidate pair
#: ``2 * W + B = 0.62``, still well inside the narrowest material gap.
INITIATIVE_BEHAVIOR_WEIGHT = 0.06
#: The baseline ``SkillLearningService`` uses for a skill with no knowledge.
INITIATIVE_KNOWLEDGE_BASELINE = 0.45
#: Lowest ranking score a saturated verified record can report. The service
#: blends success rate (0.45), average observed quality (0.40) and trial
#: confidence (0.15, capped at ten trials), so the floor is the confidence term
#: alone: nothing ever succeeded and every observed outcome scored zero.
INITIATIVE_KNOWLEDGE_SCORE_FLOOR = 0.15
#: Highest ranking score a verified record can report. A perfect success rate
#: plus saturated confidence cap two terms, and the existing outcome model tops
#: observed quality at 0.85 (success 0.70 plus the bounded positive viewer
#: reaction of 0.15), so the ceiling is 0.45 + 0.40 * 0.85 + 0.15 = 0.94.
INITIATIVE_KNOWLEDGE_SCORE_CEILING = 0.94

#: PHASE 27B: how many verified trials in one context make the existing
#: ``SkillKnowledge`` record *saturated*. The Brain scales the bounded
#: knowledge contribution by ``min(1, trials / this)``, which is exactly the
#: confidence term ``SkillLearningService._ranking_score`` already applies to
#: the same records. Reading the learning service's scores was never enough on
#: its own: a one-trial record can report a perfect rate and would otherwise
#: collect almost the full bounded bonus, i.e. sparse evidence would speak
#: nearly as loudly as dozens of verified trials. The Brain uses the learning
#: model's own saturation point instead of inventing a second reliability
#: model, and a missing record still contributes exactly nothing.
INITIATIVE_KNOWLEDGE_SATURATION_TRIALS = 10

#: PHASE 27B: how many verified failures in the same context key count as a
#: *repeated* failure. One failure is a data point, not a verdict; the rule
#: only fires once the existing ``SkillKnowledge`` record shows that every
#: verified trial in this exact context ended in failure.
INITIATIVE_KNOWLEDGE_FAILURE_TRIALS = 2

#: PHASE 27B: how many per-candidate rejection records stay in diagnostics.
MAX_CANDIDATE_REJECTIONS = 6


@dataclass(frozen=True)
class InitiativeKnowledge:
    """One bounded, already-verified ``SkillKnowledge`` summary (PHASE 27B).

    ``score`` is the existing ``SkillLearningService.rank_candidates`` score
    for this skill in this context, carried verbatim so the Brain never
    recomputes the learning model's own arithmetic. It stays conservative about
    what it does *not* know: a skill with no record is simply absent, and only
    ``trials``/``successes`` -- the counts the record actually measured -- may
    reject a candidate.
    """

    score: float = INITIATIVE_KNOWLEDGE_BASELINE
    trials: int = 0
    successes: int = 0
    failures: int = 0

    @property
    def reliability(self) -> float:
        """How much this record's verified trial count supports the score.

        ``0.0`` for an empty record, ``1.0`` once the record reaches the same
        saturation point the existing ``SkillLearningService`` ranking uses.
        It only ever *scales down* the bounded contribution; it never turns a
        record the learning service ranked highly into a negative verdict.
        """
        trials = max(0, int(self.trials))
        return min(1.0, trials / INITIATIVE_KNOWLEDGE_SATURATION_TRIALS)

    @property
    def repeatedly_failed(self) -> bool:
        """Whether every verified trial in this context failed, at least twice."""
        return (
            self.trials >= INITIATIVE_KNOWLEDGE_FAILURE_TRIALS
            and self.successes <= 0
        )


@dataclass(frozen=True)
class InitiativeContext:
    """PHASE 27B: the bounded, already-derived context of one initiative call.

    Every value is read from an existing authoritative source by the caller
    (the world snapshot, the streaming service read-model, the behaviour
    manager's advisory context and the existing ``SkillLearningService``
    ranking). Nothing here is a second copy of live state: the loop assembles
    it per decision and the evaluator keeps no memory of its own.
    """

    #: ``live`` / ``inactive`` / ``starting`` / ``stopping`` / ``error`` /
    #: ``unknown`` -- see ``normalize_stream_lifecycle``.
    stream_lifecycle: str = STREAM_LIFECYCLE_UNKNOWN
    #: Tri-state camera evidence. ``None`` means the runtime did not report a
    #: verdict, and an unreported camera never blocks a candidate.
    camera_available: Optional[bool] = None
    #: The reported phone state (``World`` / ``Held`` / ``Placed``, or a
    #: terminal state such as ``Missing``). Blank means "not reported".
    phone_state: str = ""
    #: The existing behaviour state name, for diagnostics only.
    behavior_state: str = ""
    #: The existing behaviour manager's ``preferred_skills`` advisory list.
    preferred_skills: tuple[str, ...] = ()
    #: Real SkillKnowledge evidence keyed by skill id. A skill absent from this
    #: mapping has no verified historical evidence at all.
    knowledge: Mapping[str, InitiativeKnowledge] = field(default_factory=dict)


def normalize_stream_lifecycle(value: Any) -> str:
    """Map a provider/world lifecycle string onto the bounded vocabulary."""
    token = str(value or "").strip().casefold()
    return _STREAM_LIFECYCLE_ALIASES.get(token, STREAM_LIFECYCLE_UNKNOWN)


def stream_lifecycle_from_world(world: WorldState | None) -> str:
    """The lifecycle the world snapshot itself reports, or ``unknown``.

    ``WorldState.stream_state_known`` records whether the snapshot actually
    carried the streaming flag. A snapshot that never said must not be read as
    "the stream is off"; the caller may still have a better answer from the
    streaming service, and the evaluator treats ``unknown`` as "do not judge
    the candidate on stream state".
    """
    if world is None:
        return STREAM_LIFECYCLE_UNKNOWN
    reported = getattr(world, "stream_state_known", None)
    if reported is None:
        # The snapshot did not report the flag at all: missing data is not
        # evidence that the stream is inactive.
        return STREAM_LIFECYCLE_UNKNOWN
    return STREAM_LIFECYCLE_LIVE if reported else STREAM_LIFECYCLE_INACTIVE


def camera_available_from_world(world: WorldState | None) -> Optional[bool]:
    """The camera verdict the world snapshot itself reported, or ``None``.

    ``WorldState.stream_camera_available`` defaults to ``False`` when Unity
    omitted the field, so the companion ``camera_state_known`` flag is what
    separates "the camera is unavailable" from "nobody said". An unreported
    camera never blocks a candidate.
    """
    if world is None:
        return None
    reported = getattr(world, "camera_state_known", None)
    if reported is None:
        return None
    return bool(reported)


def resolve_initiative_context(
    world: WorldState | None,
    context: InitiativeContext | None = None,
) -> InitiativeContext:
    """Merge a caller-provided context with the world-derived fallbacks.

    Explicit evidence wins; anything the caller left unspecified falls back to
    the snapshot, and a snapshot that never reported a value leaves the field
    "unknown" instead of inventing an inactive verdict.
    """
    lifecycle = (
        normalize_stream_lifecycle(context.stream_lifecycle)
        if context is not None
        else STREAM_LIFECYCLE_UNKNOWN
    )
    if lifecycle == STREAM_LIFECYCLE_UNKNOWN:
        lifecycle = stream_lifecycle_from_world(world)
    camera = context.camera_available if context is not None else None
    if camera is None:
        camera = camera_available_from_world(world)
    if context is None:
        return InitiativeContext(stream_lifecycle=lifecycle, camera_available=camera)
    return replace(context, stream_lifecycle=lifecycle, camera_available=camera)


def _phone_usable(state: str) -> bool:
    """Whether a reported phone state means the phone can be manipulated.

    A blank state is "not reported", not "no phone": only an explicit terminal
    state rejects a phone-dependent candidate.
    """
    token = str(state or "").strip()
    if not token:
        return True
    return token.casefold() in {value.casefold() for value in PHONE_EXISTS_STATES}


@dataclass(frozen=True)
class InitiativeDecision:
    """One deterministic autonomous activity choice (PHASE 27A).

    A ``WAIT`` decision carries no ``skill_id``: either the situation offers no
    registered activity or every candidate was suppressed, and the honest
    outcome is to keep waiting rather than to fabricate an action.

    PHASE 27B adds the bounded evidence behind the choice without changing any
    PHASE 27A semantics: ``eligible`` names the candidates that passed every
    filter in ranked order, and ``rejections`` records the per-candidate reason
    code for the ones that did not. ``scores`` carries the deterministic score
    behind each ranked candidate exactly as the ranking computed it, so
    diagnostics can show that verified SkillKnowledge was read into the
    decision instead of merely claiming it was. All three stay bounded and are
    diagnostics only.

    ``knowledge_contributions`` decomposes those scores: the part of each
    ranked candidate's score that came from the existing verified
    ``SkillKnowledge`` record, bounded to ``+/-INITIATIVE_KNOWLEDGE_WEIGHT``.
    Reporting the actual term (not just the total) is what lets an operator
    tell "the evidence was read and it was uniform" apart from "the evidence
    was read and it decided the ranking".
    """

    activity: str
    reason: str
    signature: str = ""
    skill_id: Optional[str] = None
    parameters: Mapping[str, Any] = field(default_factory=dict)
    eligible: tuple[str, ...] = ()
    rejections: tuple[tuple[str, str], ...] = ()
    scores: tuple[tuple[str, float], ...] = ()
    knowledge_contributions: tuple[tuple[str, float], ...] = ()

    @property
    def is_wait(self) -> bool:
        return self.activity == INITIATIVE_WAIT


@dataclass(frozen=True)
class _InitiativeCandidate:
    """One activity the selector may choose, mapped onto a real skill.

    PHASE 27B adds the two bounded inputs the context-aware filter and ranking
    read: the static preference ``weight`` (documented so the bounded secondary
    terms above can only decide *comparable* candidates, never a materially
    more relevant one) and the ``requires`` prerequisite tags evaluated against
    :class:`InitiativeContext`.
    """

    activity: str
    skill_id: str
    parameters: Mapping[str, Any]
    reason: str
    weight: float = 0.0
    requires: frozenset[str] = frozenset()


def initiative_context_rejection(
    context: InitiativeContext,
    candidate: _InitiativeCandidate,
) -> Optional[str]:
    """PHASE 27B: the bounded context reason a candidate is ineligible, or None.

    Hard eligibility only. Personality, behaviour and SkillKnowledge never
    appear here: they may only reorder candidates that already passed every
    one of these checks.
    """
    requires = candidate.requires
    if INITIATIVE_REQUIRES_PHONE in requires and not _phone_usable(context.phone_state):
        return "PHONE_UNAVAILABLE"
    if INITIATIVE_REQUIRES_CAMERA in requires and context.camera_available is False:
        # An explicitly unavailable camera means the framing work this
        # candidate claims cannot be delivered; autonomy must not claim it.
        return "CAMERA_INACTIVE"
    if INITIATIVE_REQUIRES_AUDIENCE in requires:
        lifecycle = normalize_stream_lifecycle(context.stream_lifecycle)
        if lifecycle in {
            STREAM_LIFECYCLE_INACTIVE,
            STREAM_LIFECYCLE_STOPPING,
            STREAM_LIFECYCLE_ERROR,
        }:
            return "STREAM_STATE_INCOMPATIBLE"
    record = context.knowledge.get(candidate.skill_id)
    if record is not None and record.repeatedly_failed:
        # Repeated verified failure in this exact context key: retrying the
        # same activity is the one thing the evidence already ruled out. The
        # rule is per context, so a real change of situation releases it again.
        return "RECENT_FAILURE"
    return None


def initiative_candidate_score(
    candidate: _InitiativeCandidate,
    context: InitiativeContext,
) -> float:
    """PHASE 27B: the deterministic bounded score behind the ranking.

    ``static weight + verified-knowledge bonus + behaviour preference bonus``.
    The knowledge bonus is signed around the learning service's own baseline
    for an unknown skill, so repeated verified failure pushes a skill down and
    verified success lifts it -- both bounded, both from real records only,
    and both scaled by how much verified evidence the record actually carries
    (see :func:`initiative_knowledge_contribution`).
    """
    score = float(candidate.weight)
    score += initiative_knowledge_contribution(candidate, context)
    if candidate.skill_id in context.preferred_skills:
        score += INITIATIVE_BEHAVIOR_WEIGHT
    return round(score, 4)


def initiative_knowledge_contribution(
    candidate: _InitiativeCandidate,
    context: InitiativeContext,
) -> float:
    """PHASE 27B: the bounded score change one skill's verified evidence earns.

    Three inputs, all already verified and all traceable:

    * the existing ``SkillLearningService`` ranking score for this exact
      context key, carried verbatim (it already blends the record's verified
      success rate with its average observed quality);
    * the same saturation the learning service itself uses, read from the
      record's verified trial count, so a one-trial record cannot speak as
      loudly as dozens of verified trials;
    * nothing else -- a skill with no record contributes exactly ``0.0``, so
      missing evidence can never be read as a judgement.

    The reported score is normalized against the range the learning model can
    actually emit for a verified record (``INITIATIVE_KNOWLEDGE_SCORE_FLOOR``
    to ``INITIATIVE_KNOWLEDGE_SCORE_CEILING``), with the learning service's own
    unknown-skill baseline (``INITIATIVE_KNOWLEDGE_BASELINE``) as zero. That is
    what makes the configured band genuinely usable: the model's best record
    earns ``+INITIATIVE_KNOWLEDGE_WEIGHT``, its worst earns the full negative
    term, and the baseline earns nothing. Normalizing against ``0.0 .. 1.0``
    would leave most of the band unreachable -- a plain perfect production row
    scores ``0.88`` -- so even maximally divergent legitimate records could not
    decide the narrowest comparable pair the candidate table presents
    (``0.46``), which is the entire purpose of the term.

    The result is bounded to ``+/-INITIATIVE_KNOWLEDGE_WEIGHT`` and is the only
    knowledge term the ranking consumes. It is deliberately *uniform* when the
    evidence is uniform: identical records earn identical contributions, so a
    uniform additive term cannot reorder anything at any weight -- only
    *differing* verified evidence can change an ordering. The band is
    calibrated (see ``INITIATIVE_KNOWLEDGE_WEIGHT``) so that such differing
    evidence can decide the narrowest genuinely comparable pair as the ranking
    actually presents it -- static weight plus the bounded behaviour
    preference -- while no record can leapfrog a materially more relevant
    activity.
    """
    record = context.knowledge.get(candidate.skill_id)
    if record is None:
        return 0.0
    try:
        reported = min(1.0, max(0.0, float(record.score)))
    except (TypeError, ValueError):
        reported = INITIATIVE_KNOWLEDGE_BASELINE
    if reported >= INITIATIVE_KNOWLEDGE_BASELINE:
        span = INITIATIVE_KNOWLEDGE_SCORE_CEILING - INITIATIVE_KNOWLEDGE_BASELINE
    else:
        span = INITIATIVE_KNOWLEDGE_BASELINE - INITIATIVE_KNOWLEDGE_SCORE_FLOOR
    span = max(1e-6, span)
    relative = (reported - INITIATIVE_KNOWLEDGE_BASELINE) / span
    bounded = min(1.0, max(-1.0, relative))
    return round(INITIATIVE_KNOWLEDGE_WEIGHT * bounded * record.reliability, 4)


def initiative_signature(world: WorldState | None) -> str:
    """The bounded world signature an initiative decision is recorded against.

    Repetition protection compares ``(activity, signature)`` pairs, so the same
    activity is only suppressed while the situation that justified it is
    unchanged. Any movement in stream state, audience, posture or reported
    action is a new reason and releases the activity again.
    """
    if world is None:
        return "stream=False;viewers=0;posture=;action="
    return (
        f"stream={bool(world.stream_active)};"
        f"viewers={int(world.viewer_count)};"
        f"posture={str(world.character_posture).strip().casefold()};"
        f"action={str(world.current_action).strip().casefold()}"
    )


def initiative_availability_reason(
    availability: Callable[[str, Mapping[str, Any]], Any] | None,
    candidate: _InitiativeCandidate,
) -> Optional[str]:
    """``None`` when the candidate is available, else a bounded reason.

    The probe is the caller's existing ``SkillExecutor.availability`` view (the
    loop owns the wrapper). It may answer with ``None``/``True`` for "available"
    or with a reason string, so a refusal stays explainable in diagnostics
    without a second availability implementation living here.
    """
    if availability is None:
        return "availability_unavailable"
    try:
        verdict = availability(candidate.skill_id, dict(candidate.parameters))
    except Exception:
        return "availability_error"
    if verdict is True or verdict is None or verdict == "":
        return None
    if verdict is False:
        return "SKILL_UNAVAILABLE"
    return str(verdict)


def initiative_goal_context(goal: Any) -> Optional[Mapping[str, Any]]:
    """The initiative context a goal carries, or ``None`` for every other goal.

    PHASE 27A goals mark themselves with ``context["initiative"] is True`` so
    the loop, the repetition journal and the diagnostics read-model all
    recognise the same bounded set of goals without a second goal registry.
    """
    context = getattr(goal, "context", None)
    if not isinstance(context, Mapping) or context.get("initiative") is not True:
        return None
    return context


@dataclass(frozen=True)
class GoalEvaluatorConfig:
    explore_interval_seconds: float = 25.0


class GoalEvaluator:
    """Small deterministic evaluator with context-sensitive priorities."""

    def __init__(
        self,
        personality: PersonalityProfile,
        config: GoalEvaluatorConfig | None = None,
        id_factory: Callable[[], str] | None = None,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self.personality = personality
        self.config = config or GoalEvaluatorConfig()
        self._id_sequence = count(1)
        self._id_factory = id_factory
        self._clock = clock
        self._last_explore_at: Optional[datetime] = None

    def evaluate(
        self,
        world: WorldState,
        chat: ChatMessage | None = None,
        social_context: SocialContext | None = None,
    ) -> Optional[Goal]:
        now = self._clock()

        if chat is not None and chat.intent in {
            ChatIntent.REQUEST_ACTION,
            ChatIntent.REQUEST_MOVEMENT,
            ChatIntent.QUESTION,
            # PHASE 26A: a follow-up or a topic continuation is still an
            # answer owed to the same viewer, so it keeps the response goal.
            ChatIntent.FOLLOW_UP,
            ChatIntent.TOPIC_CONTINUATION,
        }:
            priority = max(0.82, min(1.0, chat.priority + 0.08))
            if social_context is not None and chat.viewer_id == social_context.active_viewer_id:
                priority += 0.03
            return self._goal(
                GoalType.RESPOND_TO_VIEWER,
                priority=priority,
                reason=f"viewer chat requested {chat.intent.value.lower().replace('_', ' ')}",
                context=self._chat_attribution(chat),
                now=now,
            )

        if chat is not None and chat.intent in {
            ChatIntent.GREETING,
            ChatIntent.COMMENT,
            ChatIntent.ACKNOWLEDGEMENT,
        }:
            return self._goal(
                GoalType.INTERACT_WITH_VIEWER,
                priority=max(0.55, min(0.8, chat.priority)),
                reason="viewer opened a low-urgency social exchange",
                context=self._chat_attribution(chat),
                now=now,
            )

        camera_awareness = assess_camera_framing(world)
        if camera_awareness == "OUTSIDE_FRAME":
            priority = 0.66 + 0.06 * (
                social_context.social_engagement if social_context else 0.0
            )
            return self._goal(
                GoalType.IMPROVE_CAMERA_POSITION,
                priority=min(1.0, priority),
                reason="stream is active but the known transform places Lacrimosa outside frame",
                context={"camera_awareness": camera_awareness},
                now=now,
            )

        if self.stream_maintenance_required(world, social_context):
            return self._goal(
                GoalType.MAINTAIN_STREAM,
                priority=0.58,
                reason="camera is available but the stream is inactive",
                now=now,
            )

        # PHASE 26A: an explicit end of conversation is answered without a new
        # autonomous activity, and without consuming the explore cadence the
        # next quiet tick is entitled to.
        # PHASE 26B: a plain farewell ("bye Lacrimosa") is the same closing
        # moment, so it is answered here too instead of falling through to
        # exploration; the expression policy decides whether the goodbye also
        # reuses the registered wave.
        if chat is not None and chat.intent in {
            ChatIntent.CONVERSATION_END,
            ChatIntent.FAREWELL,
        }:
            return self._goal(
                GoalType.INTERACT_WITH_VIEWER,
                priority=max(0.30, min(0.6, chat.priority)),
                reason="viewer closed the conversation",
                context=self._chat_attribution(chat),
                now=now,
            )

        explore_due = (
            self._last_explore_at is None
            or (now - self._last_explore_at).total_seconds() >= self.config.explore_interval_seconds
        )
        if explore_due:
            self._last_explore_at = now
            priority = 0.20 + 0.15 * self.personality.curiosity
            return self._goal(
                GoalType.EXPLORE_ROOM,
                priority=priority,
                reason="no urgent chat; light autonomous exploration",
                # PHASE 26B: a chat message that the intent classifier did not
                # recognise (UNKNOWN) still reaches the planner with its chat
                # context, so the resulting plan is that viewer's answer. The
                # goal carries the same message attribution as every other
                # chat-derived goal instead of hiding it behind an empty dict.
                context=self._chat_attribution(chat),
                now=now,
            )

        return self._goal(
            GoalType.IDLE,
            priority=0.18,
            reason="recent activity is complete; controlled idle",
            context=self._chat_attribution(chat),
            now=now,
        )

    @staticmethod
    def _chat_attribution(chat: ChatMessage | None) -> dict[str, object]:
        """The bounded per-message attribution every chat-derived goal carries.

        PHASE 26A documented ``goal.context["message_id"]`` as *the* surface a
        plan is attributed to its viewer message through. PHASE 26B keeps that
        promise for the fallback branches too: any goal evaluated while a chat
        message is pending names that message, so behaviour/expression evidence
        can never be orphaned from the turn that produced it.
        """
        if chat is None:
            return {}
        return {"message_id": chat.message_id, "viewer_id": chat.viewer_id}

    def stream_maintenance_required(
        self,
        world: WorldState,
        social_context: SocialContext | None = None,
    ) -> bool:
        """Whether stream maintenance outranks other work right now.

        PHASE 24: this is the single condition behind the ``MAINTAIN_STREAM``
        branch, exposed read-only so the loop can probe it before doing any
        social/chat work. Unlike :meth:`evaluate` it consumes nothing (no goal
        ids, no explore cadence) and it keeps the deliberate-stop latch
        authoritative.
        """
        return (
            not world.stream_active
            and world.stream_camera_available
            and not _stream_intentionally_stopped(social_context)
        )

    # ----------------------------------------------------- initiative (Phase 27A)
    def evaluate_initiative(
        self,
        world: WorldState,
        availability: Callable[[str, Mapping[str, Any]], Any] | None = None,
        recent_activities: Iterable[tuple[str, str]] = (),
        *,
        context: InitiativeContext | None = None,
    ) -> InitiativeDecision:
        """One deterministic autonomous activity choice, or an explicit WAIT.

        PHASE 27A: this is decision support for the single existing Brain loop,
        not a second planner. It only ever selects from registered skills the
        caller's existing ``SkillExecutor`` reports as available, only ever
        returns a candidate whose ``(activity, signature)`` pair was not
        recently completed against the same world state, and records a concise
        reason for every branch it takes -- including the honest WAIT.

        The caller owns the wider eligibility gates (audience, physical state,
        viewer priority, pacing); this evaluator stays a pure function of the
        world, the availability probe and the recent-outcome journal so the same
        inputs always produce the same decision.

        PHASE 27B: ``context`` carries the already-derived stream/phone/camera/
        behaviour/SkillKnowledge evidence the caller read from the existing
        services. Hard context rules filter candidates first; verified
        SkillKnowledge and the existing behaviour preference may then reorder
        the survivors, but they can never override a hard rule or widen the
        candidate set. Without a context the evaluator falls back to what the
        world snapshot itself reported and treats unreported values as unknown.
        """
        signature = initiative_signature(world)
        effective = resolve_initiative_context(world, context)
        suppressed = {
            (str(activity), str(item))
            for activity, item in recent_activities
        }
        unavailable: list[str] = []
        blocked_notes: list[str] = []
        rejected: list[tuple[str, str]] = []
        survivors: list[tuple[float, int, _InitiativeCandidate]] = []
        for index, candidate in enumerate(self._initiative_candidates(world)):
            if (candidate.activity, signature) in suppressed:
                # Same activity, same situation, nothing new to justify it.
                rejected.append((candidate.activity, "COOLDOWN_ACTIVE"))
                continue
            blocked = initiative_context_rejection(effective, candidate)
            if blocked is not None:
                # A hard context rule refused the candidate: no amount of
                # preference or knowledge evidence may bring it back.
                rejected.append((candidate.activity, blocked))
                blocked_notes.append(f"{candidate.activity}:{blocked}")
                continue
            reason = initiative_availability_reason(availability, candidate)
            if reason is not None:
                rejected.append((candidate.activity, reason))
                unavailable.append(f"{candidate.activity}:{reason}")
                continue
            survivors.append((initiative_candidate_score(candidate, effective), index, candidate))

        if survivors:
            # Deterministic ranking: bounded score first, documented static
            # order second. Nothing outside these inputs can influence it.
            survivors.sort(key=lambda item: (-item[0], item[1]))
            ranked = tuple(item[2] for item in survivors)
            best = ranked[0]
            return InitiativeDecision(
                activity=best.activity,
                reason=best.reason,
                signature=signature,
                skill_id=best.skill_id,
                parameters=dict(best.parameters),
                eligible=tuple(candidate.activity for candidate in ranked),
                rejections=tuple(rejected[:MAX_CANDIDATE_REJECTIONS]),
                scores=tuple(
                    (candidate.activity, round(score, 4))
                    for score, _, candidate in survivors
                )[:MAX_CANDIDATE_REJECTIONS],
                knowledge_contributions=tuple(
                    (
                        candidate.activity,
                        initiative_knowledge_contribution(candidate, effective),
                    )
                    for _, _, candidate in survivors
                )[:MAX_CANDIDATE_REJECTIONS],
            )

        notes: list[str] = []
        if any(code == "COOLDOWN_ACTIVE" for _activity, code in rejected):
            notes.append("repetition_suppressed")
        if unavailable:
            notes.append("unavailable:" + ",".join(unavailable))
        if blocked_notes:
            notes.append("rejected:" + ",".join(blocked_notes[:MAX_CANDIDATE_REJECTIONS]))
        if not notes:
            notes.append("no_candidate_available")
        return InitiativeDecision(
            activity=INITIATIVE_WAIT,
            reason=";".join(notes),
            signature=signature,
            rejections=tuple(rejected[:MAX_CANDIDATE_REJECTIONS]),
        )

    def initiative_invalidation(
        self,
        world: WorldState,
        goal: Goal | None,
        context: InitiativeContext | None = None,
    ) -> Optional[str]:
        """PHASE 27B: why an in-flight initiative goal no longer holds, or None.

        The check reuses exactly the candidate definitions and the hard context
        rules the selector used, so an activity can never be started under one
        verdict and continued under another. It is read-only: the caller owns
        cancellation through the existing plan/cancellation machinery.
        """
        goal_context = initiative_goal_context(goal)
        if goal_context is None:
            return None
        activity = str(goal_context.get("activity") or "").strip()
        if not activity or activity == INITIATIVE_WAIT:
            return None
        skill_id = str(goal_context.get("skill_id") or "")
        raw_parameters = goal_context.get("parameters")
        parameters = dict(raw_parameters) if isinstance(raw_parameters, Mapping) else {}
        effective = resolve_initiative_context(world, context)
        for candidate in self._initiative_candidates(world):
            if candidate.activity != activity or candidate.skill_id != skill_id:
                continue
            if dict(candidate.parameters) != parameters:
                continue
            return initiative_context_rejection(effective, candidate)
        # The premise the activity was chosen for is gone: a movement anchor
        # that was reached (or withdrawn) leaves the activity nothing to
        # achieve, so it must not be resumed on the stale premise.
        return "GOAL_PREMISE_GONE"

    def initiative_goal(
        self,
        decision: InitiativeDecision,
        now: datetime | None = None,
    ) -> Goal:
        """The bounded goal that carries one initiative decision across ticks.

        The priority band stays below the configured viewer-interruption
        priority, so an important viewer request always wins through the
        existing priority/cancellation machinery instead of through a second
        preemption path. ``EXPLORE_ROOM`` is the existing goal type that
        already means "no viewer work is owed right now".
        """
        priority = (
            INITIATIVE_PRIORITY_BASE
            + INITIATIVE_PRIORITY_CURIOSITY_WEIGHT * self.personality.curiosity
        )
        return self._goal(
            GoalType.EXPLORE_ROOM,
            priority=priority,
            reason=decision.reason,
            now=now or self._clock(),
            context={
                "initiative": True,
                "activity": decision.activity,
                "skill_id": decision.skill_id,
                "parameters": dict(decision.parameters),
                "signature": decision.signature,
            },
        )

    def initiative_skill_ids(self, world: WorldState | None) -> tuple[str, ...]:
        """The registered skill ids this tick's candidates could select.

        PHASE 27B: exposed read-only so the loop can ask the existing
        ``SkillLearningService`` for *these* skills' verified evidence without
        duplicating the candidate table or widening the query to the whole
        library. Order follows the candidate order and duplicates collapse.
        """
        if world is None:
            return ()
        seen: list[str] = []
        for candidate in self._initiative_candidates(world):
            if candidate.skill_id not in seen:
                seen.append(candidate.skill_id)
        return tuple(seen)

    def _initiative_candidates(self, world: WorldState) -> tuple[_InitiativeCandidate, ...]:
        """The registered activities an idle, observed character may start.

        The order is the deterministic preference order and every entry is a
        genuinely registered skill: the composite camera routine, the
        individual expressive motions, gaze, and finally returning into the
        reported framing anchor. A movement candidate whose destination is
        already reached is not eligible -- walking nowhere is not an activity.

        PHASE 27B: each entry also declares the static ``weight`` and the
        stream-facing ``requires`` tags the context filter reads. The gaps
        between *materially* different activities stay wider than the bounded
        knowledge/behaviour bonuses, so verified evidence can decide a
        genuinely comparable pair but never silently reorder the policy. (The
        two ``walk_to`` destinations share one skill and therefore one
        knowledge record and one contribution, so their 0.2 separation is
        deliberately outside that rule: no evidence can tell them apart.) No
        entry ever asks for a stream lifecycle operation: autonomy reacts to
        the stream, it never starts or stops it.
        """
        candidates: list[_InitiativeCandidate] = [
            _InitiativeCandidate(
                activity="PERFORM_REGISTERED_SKILL:setup_camera_and_dance",
                skill_id="setup_camera_and_dance",
                parameters={},
                reason="camera composition routine is registered and available",
                weight=4.0,
                requires=frozenset(
                    {INITIATIVE_REQUIRES_CAMERA, INITIATIVE_REQUIRES_AUDIENCE}
                ),
            ),
            _InitiativeCandidate(
                activity="PERFORM_REGISTERED_SKILL:dance",
                skill_id="dance",
                parameters={},
                reason="registered dance is available while the audience watches",
                weight=3.0,
                requires=frozenset({INITIATIVE_REQUIRES_AUDIENCE}),
            ),
            _InitiativeCandidate(
                activity="PERFORM_REGISTERED_SKILL:wave",
                skill_id="wave",
                parameters={},
                reason="registered wave is available while the audience watches",
                weight=2.0,
                requires=frozenset({INITIATIVE_REQUIRES_AUDIENCE}),
            ),
            _InitiativeCandidate(
                activity="LOOK_AROUND",
                skill_id="look_at",
                parameters={"target": TARGET_CAMERA},
                reason="camera target is known, so she can look toward the audience",
                weight=1.0,
            ),
        ]
        if not self._already_there(world, TARGET_CENTER):
            candidates.append(
                _InitiativeCandidate(
                    activity="MOVE_TO_SUITABLE_LOCATION",
                    skill_id="walk_to",
                    parameters={"target": TARGET_CENTER},
                    reason="character stands away from the framing anchor",
                    weight=0.6,
                )
            )
        if not self._already_there(world, TARGET_CAMERA_STAND):
            candidates.append(
                _InitiativeCandidate(
                    activity="RETURN_TO_CAMERA",
                    skill_id="walk_to",
                    parameters={"target": TARGET_CAMERA_STAND},
                    reason="camera stand is reachable and closer framing is possible",
                    weight=0.4,
                )
            )
        return tuple(candidates)

    @staticmethod
    def _already_there(world: WorldState, target_name: str) -> bool:
        """Whether a movement candidate's destination is already reached."""
        target = world.target(target_name)
        if target is None:
            return True
        gap = distance(world.character_position, target.position)
        if gap is None:
            return False
        return gap <= POSITION_ARRIVAL_TOLERANCE_METERS

    def _goal(
        self,
        goal_type: GoalType,
        priority: float,
        reason: str,
        now: datetime,
        context: dict[str, object] | None = None,
    ) -> Goal:
        goal_id = self._id_factory() if self._id_factory else f"goal_{next(self._id_sequence):06d}"
        return Goal(
            goal_id=goal_id,
            goal_type=goal_type,
            priority=max(0.0, min(1.0, priority)),
            reason=reason,
            created_at=now,
            status=GoalStatus.ACTIVE,
            context=context or {},
        )


def _stream_intentionally_stopped(social_context: SocialContext | None) -> bool:
    if social_context is None:
        return False
    return social_context.last_intentional_stream_stop
