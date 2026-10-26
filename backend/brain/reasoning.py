"""Compact reasoning and deterministic mock planning for Phase 06."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Iterable, Mapping

from ..llm.base import LLMProvider, LLMProviderError, LLMRequest
from ..llm.failure_diagnostics import log_planning_failure, safe_token
from ..runtime.obs import log_event
from .models import ActionType, Goal, PersonalityProfile, PlanStep, WorldState
from .planner import BrainPlanner
from .targets import TargetResolver
from .spatial import (
    DESIRED_CLOSER,
    DESIRED_FACING,
    DESIRED_IN_FRAME,
    DESIRED_NEAR,
    TARGET_CAMERA,
    TARGET_PHONE,
    anchor_position,
    has_camera_transform,
    plan_spatial_steps,
    request_anchor,
    spatial_relationship,
)

_LOGGER = logging.getLogger(__name__)

# PHASE 21 / PART 2: the plan parser refuses these two reasons for payloads a
# real provider produced when it deviated from the advertised schema. Both are
# eligible for the registered-skill salvage below; no other reason is.
_SALVAGEABLE_PLAN_REASONS = frozenset({"UNKNOWN_ACTION", "EMPTY_PLAN"})
_REJECTED_OUTPUT_LIMIT = 1200

#: Structured desired states the deterministic spatial branch understands.
#: Only a request that the chat layer already normalized into one of these may
#: take the spatial path; free text never does.
_SPATIAL_DESIRED_STATES = frozenset(
    {DESIRED_FACING, DESIRED_IN_FRAME, DESIRED_CLOSER, DESIRED_NEAR}
)

# The chat layer uses these labels when the viewer did not name a concrete
# scene object. A concrete noun that fails TargetResolver is a real contract
# refusal, not a prompt for an unrelated movement fallback.
_GENERIC_REQUESTED_TARGETS = frozenset(
    {
        "location",
        "target",
        "room",
        "you",
        "me",
        "here",
        "dance",
        "dancing",
        "wave",
        "waving",
        "look",
        "looking",
        "sit",
        "sitting",
        "stand",
        "standing",
        "run",
        "running",
        "current_activity",
        "destination",
        "phone",
        "name",
        "stream",
        "noise",
        "social_reaction",
        "general_conversation",
        "greeting",
        "farewell",
    }
)


def build_reasoning_prompt(
    world: WorldState,
    goal: Goal,
    personality: PersonalityProfile,
    chat_text: str | None,
    planner: BrainPlanner,
    memory_context: Iterable[Any] | None = None,
    skill_descriptions: Iterable[Any] | None = None,
    learned_ranking: Iterable[Any] | None = None,
    behavior_context: dict[str, Any] | None = None,
    personalization_context: Any | None = None,
) -> str:
    return planner.build_prompt(
        world,
        goal,
        personality,
        chat_text,
        memory_context,
        skill_descriptions,
        learned_ranking,
        behavior_context,
        personalization_context,
    )


class BrainReasoner:
    def __init__(
        self,
        provider: LLMProvider,
        planner: BrainPlanner,
        skill_executor=None,
    ) -> None:
        self.provider = provider
        self.planner = planner
        self.skill_executor = skill_executor

    def plan(
        self,
        world: WorldState,
        goal: Goal,
        personality: PersonalityProfile,
        chat_text: str | None = None,
        memory_context: Iterable[Any] | None = None,
        learned_ranking: Iterable[Any] | None = None,
        behavior_context: dict[str, Any] | None = None,
        personalization_context: Any | None = None,
        *,
        desired_state: str | None = None,
        spatial_target: str | None = None,
    ) -> tuple[object | None, str]:
        # PHASE 25: a request the chat layer already normalized into a spatial
        # intent is answered from the reported transforms, deterministically
        # and without a provider call. The result is a normal plan: it is
        # created by the existing planner and every step passes the existing
        # step validator, so skills, conditions and cooldowns still own
        # execution. Free text never reaches this branch.
        spatial_plan, spatial_failure = self._spatial_plan(
            world,
            goal,
            desired_state,
            spatial_target,
        )
        if spatial_plan is not None:
            return spatial_plan, "OK"
        if spatial_failure is not None:
            return None, spatial_failure

        clarification = self._unsupported_target_clarification(
            world,
            goal,
            spatial_target,
        )
        if clarification is not None:
            return clarification

        ranking = list(learned_ranking or ())
        prompt = build_reasoning_prompt(
            world,
            goal,
            personality,
            chat_text,
            self.planner,
            memory_context,
            self.skill_executor.library.descriptions() if self.skill_executor else None,
            ranking,
            behavior_context,
            personalization_context,
        )
        request = LLMRequest(
            prompt=prompt,
            context={
                "brain": True,
                "goal": goal.goal_type.value,
                "chat": chat_text,
                "learned_ranking": ranking,
                "behavior": dict(behavior_context) if behavior_context else None,
                "personalization": (
                    personalization_context.to_prompt_block()
                    if hasattr(personalization_context, "to_prompt_block")
                    else (
                        dict(personalization_context)
                        if isinstance(personalization_context, dict)
                        else None
                    )
                ),
                "lookable_targets": sorted(
                    name for name, target in world.targets.items() if target.look_at_target
                ),
                "behavior_skill_ids": self._behavior_skill_ids(world, ranking),
            },
        )
        try:
            # PHASE 34: the only caller-side measurement of the provider
            # request, so a timeout can be reported with the duration that
            # actually elapsed. ``monotonic`` never goes backwards.
            request_started_at = time.monotonic()
            raw = self.provider.generate(request)
        except LLMProviderError as exc:
            # PHASE 30: one privacy-safe diagnostic per failed planning request.
            # It reads only structured exception metadata (class, HTTP status,
            # validated code/request id, chained cause class, transport phase,
            # effective timeout, elapsed duration) and never the provider
            # message, prompt or response body. It cannot raise, so the
            # caller-facing outcome below is unchanged: still ``None`` plus the
            # exception class name, which is what the loop records as
            # ``planning_failed:LLMProviderError``.
            log_planning_failure(
                exc,
                provider=self.provider,
                elapsed_ms=(time.monotonic() - request_started_at) * 1000.0,
            )
            return None, type(exc).__name__

        if self.skill_executor is not None:
            selection, selection_reason = self.skill_executor.parse_selection(raw)
            if selection is not None:
                return self._execute_selection(selection, goal, world)
            if selection_reason != "NOT_A_SKILL_SELECTION":
                return None, selection_reason

        plan, reason = self.planner.parse_plan(raw, goal, world)
        if plan is not None:
            return plan, reason
        if reason not in _SALVAGEABLE_PLAN_REASONS or self.skill_executor is None:
            _log_rejected_output(raw, reason, self.planner.last_rejection_context)
            return None, reason

        # PHASE 21 / PART 2: real providers answer the planner prompt with the
        # skill invocation nested inside ``steps`` (UNKNOWN_ACTION) or wrapped
        # in a payload without steps at all (EMPTY_PLAN), which the plan parser
        # correctly refuses. Re-wrap only a candidate that names a *registered*
        # skill id and hand it to the same selection validation; anything else
        # still fails with the original, unmodified reason.
        salvaged = _salvage_skill_selection(self.skill_executor, raw, reason)
        if salvaged is None:
            _log_rejected_output(raw, reason, self.planner.last_rejection_context)
            return None, reason
        return self._execute_selection(salvaged, goal, world)

    def _spatial_plan(
        self,
        world: WorldState,
        goal: Goal,
        desired_state: str | None,
        spatial_target: str | None,
    ) -> tuple[object | None, str | None]:
        """Deterministic plan for a structured spatial request.

        Returns ``(plan, None)`` when a real plan was built, ``(None, reason)``
        when the request is a spatial one that cannot be satisfied right now
        (never a fabricated success), and ``(None, None)`` when this is not a
        spatial request or no transform was reported, so the ordinary
        provider/legacy path stays in charge.
        """
        desired = (desired_state or "").strip().upper()
        if desired not in _SPATIAL_DESIRED_STATES:
            return None, None

        anchor = request_anchor(spatial_target)
        if anchor == TARGET_PHONE:
            if str(getattr(world, "phone_state", "")).casefold() == "held":
                # PHASE 25 step 13: a held phone travels with Lacrimosa, so no
                # fixed standing point in front of it can exist. The request is
                # evaluated against the camera the phone carries instead; the
                # phone state machine itself is untouched.
                anchor = TARGET_CAMERA
            elif anchor_position(world, TARGET_PHONE) is None:
                return None, None
        if anchor == TARGET_CAMERA and not has_camera_transform(world):
            # Legacy snapshots carry no camera transform; keep the historical
            # "Look at the camera" -> LOOK_AT CAMERA behaviour there.
            return None, None

        relationship = spatial_relationship(world, target=anchor)
        spatial = plan_spatial_steps(
            world,
            relationship,
            desired_state=desired,
            target=anchor,
        )
        if spatial.empty:
            if spatial.strategy == "NO_REACHABLE_POSITION":
                # Moving is the only fix and no reachable standing point
                # exists; reporting a plan here is the historical "stand
                # beside the phone" defect.
                return None, spatial.reason
            return None, None

        steps = tuple(
            PlanStep(action=ActionType(action), target=target)
            for action, target in spatial.steps
        )
        for step in steps:
            validation = self.planner.validate_step(step, world)
            if not validation.valid:
                return None, validation.reason
        return self.planner.create_plan(goal, spatial.reason, steps), None

    def _unsupported_target_clarification(
        self,
        world: WorldState,
        goal: Goal,
        requested_target: str | None,
    ) -> tuple[object | None, str] | None:
        """Answer a named, resolver-unsupported target without dispatching it.

        The model can still answer generic movement language with a supported
        action. This guard only fires when the chat layer extracted a concrete
        target noun and the existing target resolver cannot identify it. The
        single SPEAK step uses the normal plan validation and action bridge.
        """
        target = (requested_target or "").strip().casefold()
        if not target or target in _GENERIC_REQUESTED_TARGETS:
            return None
        resolved, _reason = TargetResolver(world).resolve(target)
        if resolved is not None:
            return None
        step = PlanStep(
            action=ActionType.SPEAK,
            text=f"I can't find the {target} here.",
        )
        validation = self.planner.validate_step(step, world)
        if not validation.valid:
            return None, validation.reason
        return (
            self.planner.create_plan(
                goal,
                "requested target is not available",
                (step,),
            ),
            "OK",
        )

    def _execute_selection(self, selection, goal: Goal, world: WorldState):
        """Build the plan through ``SkillExecutor``; never bypass validation."""
        skill_result = self.skill_executor.create_plan(selection, goal, world)
        if skill_result.plan is not None:
            return skill_result.plan, skill_result.reason
        return None, skill_result.reason

    def _behavior_skill_ids(
        self,
        world: WorldState,
        ranking: list[Any],
    ) -> list[str] | None:
        """Bounded list of learned skills that are valid right now.

        The advisory behavior preference is only allowed to choose from this
        pre-filtered candidate set. Availability is still re-checked by
        ``SkillExecutor.create_plan`` when the plan is built, so this narrows
        the candidates instead of bypassing any condition or cooldown.
        """
        executor = self.skill_executor
        if executor is None:
            return None
        candidates: list[str] = []
        for item in ranking:
            if not isinstance(item, dict):
                continue
            skill_id = item.get("skill_id")
            if not isinstance(skill_id, str) or not skill_id.strip():
                continue
            candidate = skill_id.strip().lower()
            if candidate in candidates:
                continue
            try:
                available = executor.availability(candidate, world).valid
            except Exception:
                available = False
            if available:
                candidates.append(candidate)
            if len(candidates) >= 8:
                break
        return candidates


def _log_rejected_output(
    raw_output: str,
    reason: str,
    context: Mapping[str, Any] | None = None,
) -> None:
    """Bounded, redacted capture of the payload the plan parser refused.

    PHASE 32: ``context`` is the bounded record of the exact step the validator
    refused (see ``BrainPlanner.last_rejection_context``). Only short,
    identifier-shaped tokens are logged; a missing or oddly shaped value is
    omitted rather than truncated, and the raw plan is neither duplicated nor
    re-serialized for these fields. Providing no context keeps the previous
    event shape exactly.
    """
    text = raw_output if isinstance(raw_output, str) else str(raw_output)
    fields: dict[str, Any] = {
        "reason": reason,
        "output": text[:_REJECTED_OUTPUT_LIMIT],
    }
    if isinstance(context, Mapping):
        action = safe_token(context.get("action"))
        if action is not None:
            fields["rejected_action"] = action
        target = safe_token(context.get("target"))
        if target is not None:
            fields["rejected_target"] = target
        index = context.get("step_index")
        if isinstance(index, int) and not isinstance(index, bool) and 0 <= index < 64:
            fields["rejected_step_index"] = index
    log_event(_LOGGER, "brain.planner_output_rejected", **fields)


def _iter_skill_candidates(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Bounded candidate objects that may name a skill invocation.

    Ordered from the most explicit shape to the least: the top level itself,
    a single wrapper object, ``steps`` entries, and the same shapes nested one
    level deeper inside ``plan``/``action``/``skill`` wrappers.
    """
    candidates: list[dict[str, Any]] = []

    def add(item: Any) -> None:
        if isinstance(item, dict) and not any(item is existing for existing in candidates):
            candidates.append(item)

    def add_steps(steps: Any) -> None:
        if isinstance(steps, list):
            for item in steps:
                add(item)

    add(payload)
    add_steps(payload.get("steps"))
    for key in (
        "plan",
        "step",
        "selected_skill",
        "action",
        "skill",
        "skill_output_schema",
    ):
        container = payload.get(key)
        if not isinstance(container, dict):
            continue
        add(container)
        add(container.get("step"))
        add(container.get("selected_skill"))
        add_steps(container.get("steps"))
    return candidates


def _skill_name_from(
    candidate: dict[str, Any], payload: dict[str, Any]
) -> tuple[Any, bool]:
    """Name plus whether it came from an explicit ``skill_id`` key.

    The distinction matters: an explicit ``skill_id`` is a skill selection
    (the same shape ``SkillExecutor.parse_selection`` accepts at top level),
    while a name under an ``action``/``skill`` alias may be a bare built-in
    action that must never be substituted by a similarly named skill.
    """
    sources = (candidate,) if candidate is payload else (candidate, payload)
    for source in sources:
        for key in ("skill_id", "skillId", "skill", "action"):
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                return value, key in ("skill_id", "skillId")
    return None, False


def _salvage_skill_selection(executor, raw_output: str, fallback_reason: str):
    """Extract a skill invocation the plan parser refused, if one is valid.

    Only a candidate that resolves to a *registered* skill is accepted, so
    the normal plan path keeps owning ordinary actions: a name under an
    ``action``/``skill`` alias that is a known :class:`ActionType` is never
    substituted, while an explicit ``skill_id`` naming a registered skill is
    honored exactly like a top-level skill selection. The re-wrapped payload
    goes back through ``SkillExecutor.parse_selection``/``create_plan``, which
    re-check parameters, cooldowns and world conditions.
    """
    try:
        payload = json.loads(raw_output)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None

    for candidate in _iter_skill_candidates(payload):
        name, explicit_skill_id = _skill_name_from(candidate, payload)
        if not isinstance(name, str) or not name.strip():
            continue
        if not explicit_skill_id:
            try:
                ActionType(name.strip().upper())
                # A bare built-in action name is never substituted by a skill.
                continue
            except (ValueError, TypeError):
                pass
        skill_id = name.strip().lower()
        if executor.library.get(skill_id) is None:
            continue
        parameters = candidate.get("parameters")
        if not isinstance(parameters, dict):
            parameters = payload.get("parameters")
        reason = candidate.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            reason = payload.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            reason = fallback_reason
        selection, _ = executor.parse_selection(
            json.dumps(
                {
                    "skill_id": skill_id,
                    "parameters": dict(parameters) if isinstance(parameters, dict) else {},
                    "reason": reason.strip(),
                }
            )
        )
        if selection is not None:
            return selection
    return None


class MockBrainLLMProvider(LLMProvider):
    """Deterministic provider used by tests and local API mode."""

    fail_with: str | None = None

    def generate(self, request: LLMRequest) -> str:
        if self.fail_with == "timeout":
            raise LLMProviderError("provider timeout")
        if self.fail_with == "provider":
            raise LLMProviderError("provider unavailable")
        if self.fail_with == "invalid_json":
            return "not-json"
        if self.fail_with == "unknown_action":
            return json.dumps({"reason": "bad output", "steps": [{"action": "FLY_TO", "target": "MOON"}]})

        chat = request.context.get("chat") or ""
        normalized = str(chat).lower()
        if _matches_stream_control(normalized, "start"):
            return json.dumps(
                {
                    "skill_id": "start_stream",
                    "parameters": {},
                    "reason": "viewer requested starting the stream",
                }
            )
        if _matches_stream_control(normalized, "stop"):
            return json.dumps(
                {
                    "skill_id": "stop_stream",
                    "parameters": {},
                    "reason": "viewer requested stopping the stream",
                }
            )
        if _PHONE_PICKUP_PATTERN.search(normalized):
            return json.dumps(
                {
                    "skill_id": "pick_up_phone",
                    "parameters": {},
                    "reason": "viewer requested picking up the phone",
                }
            )
        if _PHONE_PLACEMENT_PATTERN.search(normalized):
            return json.dumps(
                {
                    "skill_id": "place_phone",
                    "parameters": {"target": _phone_placement_target(normalized)},
                    "reason": "viewer requested placing the phone",
                }
            )
        if "dance" in normalized:
            selected = (
                _highest_ranked_learned_composite(request)
                if "camera" in normalized
                else _highest_ranked_learned_skill(request)
            )
            if selected is not None:
                selected_composite = "camera" in normalized
                return json.dumps(
                    {
                        "skill_id": selected,
                        "parameters": {},
                        "reason": (
                            "selected the highest-ranked learned dance composition"
                            if selected_composite
                            else "selected the highest-ranked learned dance"
                        ),
                    }
                )
            steps = [
                {"action": "MOVE_TO", "target": "CENTER"},
                {"action": "LOOK_AT", "target": "CAMERA"},
                {"action": "DANCE"},
                {"action": "SPEAK", "text": "Hope you enjoyed that!"},
            ]
        elif "table" in normalized:
            steps = [
                {"action": "MOVE_TO", "target": "TABLE"},
                {"action": "LOOK_AT", "target": "TABLE"},
                {"action": "SPEAK", "text": "This is the table."},
            ]
        elif "wave" in normalized:
            steps = [{"action": "WAVE"}]
        elif "look" in normalized and "camera" in normalized:
            steps = [{"action": "LOOK_AT", "target": "CAMERA"}]
        elif request.context.get("goal") == "EXPLORE_ROOM":
            steps = [{"action": "MOVE_TO", "target": "CENTER"}]
        elif request.context.get("goal") == "MAINTAIN_STREAM":
            steps = [{"action": "START_STREAM"}]
        else:
            preferred = _behavior_preference(request)
            if preferred is not None:
                return preferred
            steps = [{"action": "STOP"}]
        return json.dumps({"reason": "structured plan", "steps": steps})


def _behavior_preference(request: LLMRequest) -> str | None:
    """Express the advisory behavior state only when nothing else applied.

    This is an ordering preference inside the existing mock provider, not a
    planner: every branch below still goes through ``parse_selection``/skill
    availability or ``parse_plan`` validation before anything can run.
    """
    context = request.context.get("behavior")
    if not isinstance(context, dict):
        return None
    state = str(context.get("state") or "").upper()

    if state == "HAPPY" and context.get("proactive_allowed"):
        learned = _highest_ranked_learned_wave(request)
        if learned is not None:
            return json.dumps(
                {
                    "skill_id": learned,
                    "parameters": {},
                    "reason": "behavior preference: positive viewer reaction",
                }
            )
        return json.dumps(
            {
                "reason": "behavior preference: positive viewer reaction",
                "steps": [{"action": "WAVE"}],
            }
        )
    if state == "SURPRISED" and context.get("reactive_allowed"):
        return json.dumps(
            {
                "reason": "behavior preference: short reaction",
                "steps": [{"action": "SPEAK", "text": "Oh, that caught me off guard."}],
            }
        )
    if state == "CURIOUS" and context.get("attention_allowed") and "CAMERA" in _lookable(request):
        return json.dumps(
            {
                "reason": "behavior preference: attention toward the viewer",
                "steps": [{"action": "LOOK_AT", "target": "CAMERA"}],
            }
        )
    return None


def _lookable(request: LLMRequest) -> tuple[str, ...]:
    targets = request.context.get("lookable_targets")
    if not isinstance(targets, (list, tuple)):
        return ()
    return tuple(str(target).upper() for target in targets)


def _highest_ranked_learned_wave(request: LLMRequest) -> str | None:
    """Highest-ranked learned skill whose description reads as a wave.

    Registered motion skills keep their own motion id, so the wave is matched
    on the human-readable description instead of a naming convention.
    """
    ranking = request.context.get("learned_ranking")
    if not isinstance(ranking, list):
        return None

    allowed = request.context.get("behavior_skill_ids")
    if isinstance(allowed, list) and allowed:
        permitted = {str(skill).strip().lower() for skill in allowed}
    elif isinstance(allowed, list):
        # An empty candidate list means nothing learned is valid right now.
        return None
    else:
        permitted = None

    candidates: list[tuple[float, str]] = []
    for item in ranking:
        if not isinstance(item, dict):
            continue
        skill_id = item.get("skill_id")
        description = item.get("description")
        score = item.get("advisory_score", item.get("score"))
        if not isinstance(skill_id, str) or not skill_id.strip():
            continue
        if not isinstance(description, str) or "wave" not in description.casefold():
            continue
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            continue
        if item.get("learned") is not True:
            continue
        normalized_id = skill_id.strip().lower()
        if permitted is not None and normalized_id not in permitted:
            continue
        candidates.append((float(score), normalized_id))
    return max(candidates)[1] if candidates else None


_START_STREAM_PATTERN = re.compile(
    r"\b(?:start|begin|turn\s+on|go\s+live)\b.*\b(?:livestream|stream)\b",
    re.IGNORECASE,
)
_STOP_STREAM_PATTERN = re.compile(
    r"\b(?:stop|end|turn\s+off|shut\s+down|finish)\b.*\b(?:livestream|stream)\b",
    re.IGNORECASE,
)
_PHONE_PICKUP_PATTERN = re.compile(
    r"\b(?:pick\s+up|grab|take)\s+(?:the\s+)?phone\b",
    re.IGNORECASE,
)
_PHONE_PLACEMENT_PATTERN = re.compile(
    r"\b(?:place|put|set)\s+(?:the\s+)?phone\b|\bput\s+(?:the\s+)?phone\s+down\b",
    re.IGNORECASE,
)


def _matches_stream_control(text: str, operation: str) -> bool:
    pattern = _START_STREAM_PATTERN if operation == "start" else _STOP_STREAM_PATTERN
    return pattern.search(text) is not None


def _phone_placement_target(text: str) -> str:
    if re.search(r"\bbench\b", text):
        return "BENCH"
    if re.search(r"\bfloor\b", text):
        return "FLOOR"
    return "TABLE"


def _highest_ranked_learned_skill(request: LLMRequest) -> str | None:
    return _highest_ranked_learned_skill_internal(request, include_composites=False)


def _highest_ranked_learned_composite(request: LLMRequest) -> str | None:
    ranking = request.context.get("learned_ranking")
    if not isinstance(ranking, list):
        return None

    candidates: list[tuple[float, str]] = []
    for item in ranking:
        if not isinstance(item, dict):
            continue
        skill_id = item.get("skill_id")
        score = item.get("advisory_score", item.get("score"))
        if not isinstance(skill_id, str) or not skill_id.strip():
            continue
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            continue
        if item.get("learned") is not True or item.get("category") != "COMPOSITE":
            continue
        candidates.append((float(score), skill_id.strip().lower()))
    return max(candidates)[1] if candidates else None


def _highest_ranked_learned_skill_internal(
    request: LLMRequest,
    *,
    include_composites: bool,
) -> str | None:
    ranking = request.context.get("learned_ranking")
    if not isinstance(ranking, list):
        return None

    candidates: list[tuple[float, str]] = []
    for item in ranking:
        if not isinstance(item, dict):
            continue
        skill_id = item.get("skill_id")
        score = item.get("advisory_score", item.get("score"))
        if not isinstance(skill_id, str) or not skill_id.strip():
            continue
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            continue
        if item.get("learned") is not True:
            continue
        if item.get("category") == "COMPOSITE" and not include_composites:
            continue
        skill_id = skill_id.strip().lower()
        if "dance" not in skill_id:
            continue
        candidates.append((float(score), skill_id))
    return max(candidates)[1] if candidates else None
