"""Provider-independent reasoning prompt, parser, and plan validation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from itertools import count
from typing import Any, Callable, Iterable

from .models import (
    ActionType,
    Goal,
    PersonalityProfile,
    Plan,
    PlanStep,
    PlanValidation,
    WorldState,
    _target_dict,
)
from .targets import TargetResolver


MAX_TEXT_LENGTH = 500
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")

#: Canonical placement surfaces the Unity bridge can actually resolve.
PHONE_PLACEMENT_TARGETS = frozenset({"TABLE", "BENCH", "FLOOR"})


def _is_motion_reference(value: str | None) -> bool:
    """True when a gesture target names an animation asset, not a world target.

    PHASE 28: the live provider emitted ``DANCE:BENCH`` for a viewer asking to
    dance near the bench and ``WAVE:CAMERA`` for a wave at the camera. Both are
    spatial names, so the Unity bridge could only resolve them as animation
    references and answered ``generated_motion_unresolved``. Bare WAVE/DANCE
    means the built-in state; a target is only meaningful for those actions
    when it is a motion asset reference, which is exactly what the learning
    registry stores in ``animation_reference``.
    """
    if not isinstance(value, str):
        return False
    normalized = value.strip().replace("\\", "/")
    return normalized.lower().startswith("assets/") and len(normalized) > len("assets/")


def _phone_placement_destinations(steps: list[PlanStep]) -> set[str]:
    """Distinct canonical placement targets selected by earlier moves."""
    return {
        step.target
        for step in steps
        if step.action == ActionType.MOVE_TO
        and step.target in PHONE_PLACEMENT_TARGETS
    }


def _resolve_phone_placements(steps: list[PlanStep]) -> list[PlanStep] | None:
    """Fill an omitted ``PLACE_PHONE`` target only from an earlier move.

    A destination is inferred only when the steps before that placement contain
    exactly one distinct ``MOVE_TO`` target among the canonical Unity placement
    surfaces. Explicit targets, no candidates, conflicting candidates, and
    placement targets that appear only later all keep the safe rejection.
    """
    resolved_steps: list[PlanStep] = []
    for index, step in enumerate(steps):
        if step.action == ActionType.PLACE_PHONE and step.target is None:
            destinations = _phone_placement_destinations(steps[:index])
            inferred_target = (
                next(iter(destinations), None) if len(destinations) == 1 else None
            )
            if inferred_target is None:
                return None
            step = PlanStep(
                action=step.action,
                target=inferred_target,
                text=step.text,
                position=step.position,
            )
        resolved_steps.append(step)
    return resolved_steps


def _behavior_prompt_block(behavior_context: dict[str, Any] | None) -> dict[str, Any] | None:
    """Compact, bounded advisory block for the planner prompt.

    Only derived runtime context is forwarded; no transcript or viewer
    identity is copied into the prompt from here.
    """
    if not isinstance(behavior_context, dict):
        return None
    block = {
        "state": behavior_context.get("state"),
        "previous_state": behavior_context.get("previous_state"),
        "transition_reason": behavior_context.get("transition_reason"),
        "speech_style": behavior_context.get("speech_style"),
        "preferred_skills": list(behavior_context.get("preferred_skills") or ())[:4],
        "proactive_allowed": bool(behavior_context.get("proactive_allowed")),
        "reactive_allowed": bool(behavior_context.get("reactive_allowed")),
        "attention_allowed": bool(behavior_context.get("attention_allowed")),
    }
    expression = behavior_context.get("expression")
    if isinstance(expression, dict):
        # PHASE 26B: the bounded expression decision travels with the same
        # single reasoning prompt so tone/intensity stay advisory input, never
        # a second model call.
        block["expression"] = {
            "tone": expression.get("tone"),
            "reaction_intensity": expression.get("reaction_intensity"),
            "gesture": expression.get("gesture"),
            "gesture_reason": expression.get("gesture_reason"),
        }
    block["rule"] = "advisory preference only; never bypasses skill validation"
    return block


def _personalization_prompt_block(
    personalization_context: Any | None,
) -> dict[str, Any] | None:
    """Compact viewer-specific advisory block; never raw history.

    Accepts the Phase 19 :class:`PersonalizedContext` or its prompt dict. Only
    bounded, already-derived fields are forwarded; an unknown viewer or a
    broken payload simply yields ``None``.
    """
    if personalization_context is None:
        return None
    if hasattr(personalization_context, "to_prompt_block"):
        try:
            payload = personalization_context.to_prompt_block()
        except Exception:
            return None
    else:
        payload = personalization_context
    if not isinstance(payload, dict):
        return None
    known = bool(payload.get("known_viewer"))
    returning = bool(payload.get("returning"))
    topics = payload.get("preferred_topics")
    skills = payload.get("preferred_skills")
    return {
        "known_viewer": known,
        "returning": returning,
        "relationship_level": payload.get("relationship_level"),
        "familiarity": payload.get("familiarity"),
        "interaction_count": payload.get("interaction_count"),
        "preferred_topics": [str(item)[:40] for item in topics][:3]
        if isinstance(topics, (list, tuple))
        else [],
        "preferred_skills": [str(item)[:40] for item in skills][:3]
        if isinstance(skills, (list, tuple))
        else [],
        "topic_continuity": payload.get("topic_continuity"),
        "greeting": payload.get("greeting"),
        "speech_style": payload.get("speech_style"),
        "reason": str(payload.get("reason") or "")[:200] or None,
        "rule": (
            "advisory personalization only; never bypasses skill conditions, "
            "cooldowns, allowlists, social attention, or SkillExecutor"
        ),
    }


@dataclass(frozen=True)
class BrainPlannerConfig:
    max_plan_steps: int = 8


# Compact prompt guard. The decision inputs and the output contract are ordered
# first (see ``build_prompt``), so only trailing advisory context can ever be
# clipped. The real Phase 21 context (17 registered skills, learned ranking,
# memories, personalization) measures ~7.8k characters, so the guard keeps all
# decision-relevant blocks while still bounding the request.
MAX_PLANNER_PROMPT_CHARS = 8000


class BrainPlanner:
    def __init__(
        self,
        config: BrainPlannerConfig | None = None,
        id_factory: Callable[[], str] | None = None,
        clock: Callable[[], object] | None = None,
        version_factory: Callable[[], int] | None = None,
    ) -> None:
        self.config = config or BrainPlannerConfig()
        self._id_sequence = count(1)
        self._id_factory = id_factory
        self._clock = clock
        self._version_sequence = count(1)
        self._version_factory = version_factory
        #: PHASE 32: instrumentation-only record of the step the validator
        #: refused during the most recent ``parse_plan`` call. It lets the
        #: rejection event name the exact action/target/index without changing
        #: ``validate_step``, its failure codes, or ``parse_plan``'s return
        #: value. Ticks run one at a time under the Brain's lock, so this is not
        #: shared across concurrent decisions.
        self.last_rejection_context: dict[str, Any] | None = None

    def build_prompt(
        self,
        world: WorldState,
        goal: Goal,
        personality: PersonalityProfile,
        chat_text: str | None = None,
        memory_context: Iterable[Any] | None = None,
        skill_descriptions: Iterable[Any] | None = None,
        learned_ranking: Iterable[Any] | None = None,
        behavior_context: dict[str, Any] | None = None,
        personalization_context: Any | None = None,
    ) -> str:
        memories = [
            item.to_prompt_dict() if hasattr(item, "to_prompt_dict") else {"summary": str(item)}
            for item in (memory_context or ())
        ]
        skills = [
            item.to_dict()
            if hasattr(item, "to_dict")
            else dict(item)
            if isinstance(item, dict)
            else {"skill_id": str(item)}
            for item in (skill_descriptions or ())
        ]
        ranking = [
            item if isinstance(item, dict) else {"skill_id": str(item)}
            for item in (learned_ranking or ())
        ]
        behavior = _behavior_prompt_block(behavior_context)
        personalization = _personalization_prompt_block(personalization_context)

        # PHASE 31: action-specific target rules, derived from the authoritative
        # target flags at prompt-building time. LOOK_AT/TURN_TO and MOVE_TO do
        # not accept the same entities: ``CAMERA_STAND``/``PHONE_STAND`` are
        # navmesh standing points (navigable, not lookable) while ``CAMERA``/
        # ``PHONE`` are their lookable, non-navigable counterparts. A live model
        # repeatedly emitted LOOK_AT against the standing points even though the
        # nested ``world.targets`` flags already said they were not lookable
        # (see Target_Not_Lookable_Audit_20261018T204826Z), so the distinction is
        # now stated explicitly and the valid sets are enumerated.
        #
        # Only targets that ``TargetResolver`` can execute may enter the prompt.
        # Unity is authoritative about what exists in the scene, but the
        # resolver remains the executable allowlist; this keeps a stale or
        # test-only target such as WINDOW from being advertised by mistake.
        resolver = TargetResolver(world)
        supported_targets = {
            resolved
            for name in world.targets
            if (resolved := resolver.resolve(name)[0]) is not None
        }

        # The rules and lists are computed here from the same TargetDefinition
        # objects the validator reads, so they can never drift from
        # ``validate_step`` and never need hand-editing when an entity is added.
        # They sit immediately after ``world`` so the compact prompt guard below
        # cannot clip them.
        look_targets = sorted(
            name
            for name, target in world.targets.items()
            if target.look_at_target and name in supported_targets
        )
        navigation_targets = sorted(
            name
            for name, target in world.targets.items()
            if target.navigation_target and name in supported_targets
        )
        navigation_only_targets = sorted(
            name
            for name, target in world.targets.items()
            if target.navigation_target
            and not target.look_at_target
            and name in supported_targets
        )
        look_only_targets = sorted(
            name
            for name, target in world.targets.items()
            if target.look_at_target
            and not target.navigation_target
            and name in supported_targets
        )
        world_payload = world.to_dict()
        world_payload["targets"] = {
            name: _target_dict(target)
            for name, target in world.targets.items()
            if name in supported_targets
        }
        world_payload["valid_look_targets"] = look_targets
        world_payload["valid_navigation_targets"] = navigation_targets
        world_payload["navigation_only_targets"] = navigation_only_targets
        world_payload["look_only_targets"] = look_only_targets
        action_target_rules = [
            "LOOK_AT and TURN_TO accept only targets listed in "
            "world.valid_look_targets (targets whose look_at_target is true)",
            "MOVE_TO accepts only targets listed in "
            "world.valid_navigation_targets (targets whose navigation_target is true)",
            "never use a navigation-only standing point "
            "(world.navigation_only_targets) as a LOOK_AT or TURN_TO target",
            "a valid look target is not automatically a valid navigation "
            "destination, and vice versa",
        ]

        prompt = {
            "role": "planner",
            "character": personality.to_dict(),
            "world": world_payload,
            "action_target_rules": action_target_rules,
            "goal": goal.to_dict(),
            "behavior": behavior,
            "chat": chat_text[:300] if chat_text else None,
            "available_actions": [action.value for action in ActionType],
            # PHASE 21: the decision inputs and output contract must survive the
            # compact prompt guard below, so constraints/output_schema precede
            # every variable-length context block. Those blocks may be
            # truncated; the schema the parser validates against may not.
            "constraints": [
                "return strict JSON only",
                "maximum 8 steps",
                "use only available actions and valid_targets",
                # PHASE 37: the prompt carries two vocabularies -- canonical
                # uppercase action names and lowercase skill ids -- and a live
                # provider copied the skill spelling into the ``action`` field.
                # The distinction is stated here, next to the other action rules
                # and inside the truncation-protected constraint block, so a
                # clipped prompt still carries it. These lines name no actions:
                # the vocabulary itself stays ``ActionType`` further down, so
                # nothing here can go stale.
                "action must be one uppercase name from available_actions "
                "(the action vocabulary)",
                "available_skills ids are skill ids, not actions; never copy "
                "one into action",
                "invoke a skill only via skill_output_schema, not as an "
                "action step",
                "action names take no aliases, suffixes, target names or "
                "punctuation",
                "if a request is not expressible with these actions, do not "
                "invent one",
                # PHASE 31: the action/target pairing rule, stated up front so a
                # clipped prompt still carries it. The enumerated sets live in
                # the world block right above.
                "for LOOK_AT/TURN_TO use world.valid_look_targets only, and for "
                "MOVE_TO use world.valid_navigation_targets only",
                # PHASE 28: a live provider answered "dance over by the bench"
                # with DANCE:BENCH. A gesture target is an animation asset
                # reference, never a spatial name, so state that explicitly.
                "WAVE/DANCE targets are motion asset references only; use a bare "
                "WAVE/DANCE for the built-in motion and MOVE_TO plus the gesture "
                "for a location request",
                "prefer available_skills when one fits the goal",
                # PHASE 21 / PART 2: real viewers name a registered motion
                # explicitly; the live DeepSeek planner answered such requests
                # with the built-in DANCE action or with a payload that had no
                # steps at all (planning_failed:EMPTY_PLAN), which silently
                # dropped the viewer turn. The skill form is already the
                # documented contract; this only makes it unambiguous.
                "when chat names or clearly requests a listed skill_id, "
                "return skill_output_schema with that exact skill_id and no steps; "
                "a requested listed skill takes precedence over a built-in action",
                "never return an empty response: always include at least one step "
                "or a skill_output_schema object",
                "learned_skill_ranking is advisory ordering only",
                "treat behavior as advisory preference only; it never bypasses "
                "skill conditions, cooldowns, allowlists, or SkillExecutor",
                "use behavior.speech_style for tone; keep speech factual",
                "treat personalization as advisory preference only; it never "
                "bypasses skill conditions, cooldowns, allowlists, or SkillExecutor",
                "never claim viewer details that are absent from personalization",
                "to use a listed skill return skill_output_schema and no steps",
                "never invent a skill_id not listed in available_skills",
                "do not output code",
                "include SPEAK when responding to a viewer",
            ],
            "output_schema": {
                "reason": "short string",
                "steps": [{"action": "MOVE_TO", "target": "CENTER", "text": None}],
            },
            # PHASE 21 / PART 2: real providers answered the old, ambiguous
            # "return skill_id and parameters" line by nesting a skill inside
            # ``steps``, which the strict plan parser correctly rejected with
            # UNKNOWN_ACTION. The skill form is advertised explicitly, at the
            # top level, exactly where ``SkillExecutor.parse_selection`` reads
            # it; validation is unchanged.
            "skill_output_schema": {
                "skill_id": "<id from available_skills>",
                "parameters": {},
                "reason": "short string",
            },
            "valid_targets": sorted(supported_targets),
            "available_skills": skills,
            "learned_skill_ranking": ranking,
            "relevant_memories": memories,
            # PHASE 19 stays last: advisory personalization must never displace
            # skill/goal/target information if the compact prompt is clipped.
            "personalization": personalization,
        }
        encoded = json.dumps(prompt, ensure_ascii=False, separators=(",", ":"))
        return encoded[:MAX_PLANNER_PROMPT_CHARS]

    def parse_plan(
        self,
        raw_output: str,
        goal: Goal,
        world: WorldState,
    ) -> tuple[Plan | None, str]:
        # PHASE 38: every rejection starts with no attribution. A refusal that
        # returns before it can name a step (bad JSON, an empty plan, an
        # unresolved action token) therefore emits the previous, context-free
        # event shape instead of inheriting a stale context from an earlier
        # call.
        self.last_rejection_context = None
        try:
            payload = json.loads(raw_output)
        except (json.JSONDecodeError, TypeError):
            return None, "INVALID_JSON"
        if not isinstance(payload, dict):
            return None, "INVALID_SCHEMA"

        steps_value = payload.get("steps")
        reason = payload.get("reason", goal.reason)
        if not isinstance(steps_value, list) or not steps_value:
            return None, "EMPTY_PLAN"
        if len(steps_value) > self.config.max_plan_steps:
            return None, "PLAN_TOO_LONG"
        if not isinstance(reason, str) or not reason.strip():
            return None, "INVALID_REASON"

        steps: list[PlanStep] = []
        for index, item in enumerate(steps_value):
            if not isinstance(item, dict):
                return None, "INVALID_STEP_SCHEMA"
            action_name = item.get("action")
            try:
                # PHASE 36: the action token is case-insensitive. Live providers
                # have answered with the lowercase snake_case spelling of a
                # supported action ("look_at", "speak") because the prompt
                # carries lowercase skill ids next to the uppercase action
                # vocabulary. Only the *spelling* is normalised here; the
                # ActionType enum stays the authoritative vocabulary, so an
                # unsupported name still fails below. Targets are normalised the
                # same way further down, and the skill-salvage helper already
                # upper-cases its action name.
                action = ActionType(str(action_name).strip().upper())
            except (ValueError, TypeError):
                # PHASE 38: name the refused token, not just the category. The
                # resolution above stringifies its input for the enum lookup, so
                # the *original* value is attached only when it is already a
                # string: ``str(None)``/``str(17)`` would otherwise log "None" or
                # "17" as if the model had emitted that action name. The
                # rejection logger re-validates this field through the existing
                # ``safe_token`` bound (identifier characters, 64 max), so an
                # oversized, malformed or non-string token is dropped rather than
                # truncated, and no ``str()``/``repr()`` of an arbitrary object
                # ever reaches the event. Casing is preserved so a lowercase or
                # mixed-case supported name stays distinguishable from a
                # genuinely unsupported one.
                self.last_rejection_context = {
                    "step_index": index,
                    "action": action_name if isinstance(action_name, str) else None,
                    "target": None,
                    "reason": "UNKNOWN_ACTION",
                }
                return None, "UNKNOWN_ACTION"

            target_value = item.get("target")
            text_value = item.get("text")
            position_value = item.get("position")
            target = (
                target_value.strip()
                if isinstance(target_value, str) and target_value.strip()
                else None
            )
            # PHASE 28: spatial targets are matched case-insensitively by the
            # resolver, but an animation reference is a case-sensitive asset
            # path (GeneratedMotionLibrary.ResolveClip compares it verbatim).
            # A gesture target that is an ``Assets/...`` reference is therefore
            # preserved exactly; everything else keeps the historical fold.
            if (
                target is not None
                and not (
                    action in {ActionType.WAVE, ActionType.DANCE}
                    and _is_motion_reference(target)
                )
            ):
                target = target.upper()
            text = text_value if isinstance(text_value, str) else None
            if text and len(text) > MAX_TEXT_LENGTH:
                return None, "TEXT_TOO_LONG"

            position: tuple[float, float, float] | None = None
            if isinstance(position_value, (list, tuple)) and len(position_value) == 3:
                try:
                    position = tuple(float(value) for value in position_value)
                except (TypeError, ValueError):
                    return None, "INVALID_POSITION"

            steps.append(PlanStep(action=action, target=target, text=text, position=position))

        resolved_steps = _resolve_phone_placements(steps)
        if resolved_steps is None:
            placement_step_index = next(
                (
                    index
                    for index, step in enumerate(steps)
                    if step.action == ActionType.PLACE_PHONE and step.target is None
                ),
                None,
            )
            self.last_rejection_context = {
                "step_index": placement_step_index,
                "action": ActionType.PLACE_PHONE.value,
                "target": None,
                "reason": "PLACEMENT_TARGET_INVALID",
            }
            return None, "PLACEMENT_TARGET_INVALID"
        steps = resolved_steps

        plan = self.create_plan(goal, reason.strip(), steps)
        for index, step in enumerate(plan.steps):
            validation = self.validate_step(step, world)
            if not validation.valid:
                # PHASE 32: bounded attribution of the refused step. Validation
                # itself is untouched -- the same reason string is returned.
                self.last_rejection_context = {
                    "step_index": index,
                    "action": step.action.value,
                    "target": step.target,
                    "reason": validation.reason,
                }
                return None, validation.reason
        return plan, "OK"

    def create_plan(self, goal: Goal, reason: str, steps: Iterable[PlanStep]) -> Plan:
        version = self._version_factory() if self._version_factory else next(self._version_sequence)
        plan_id = self._id_factory() if self._id_factory else f"plan_{next(self._id_sequence):06d}"
        return Plan(
            plan_id=plan_id,
            goal_id=goal.goal_id,
            reason=reason,
            steps=tuple(steps),
            version=version,
        )

    def validate_step(self, step: PlanStep, world: WorldState) -> PlanValidation:
        resolver = TargetResolver(world)
        if step.action in {ActionType.MOVE_TO, ActionType.TURN_TO, ActionType.LOOK_AT}:
            target, reason = resolver.resolve(step.target)
            if target is None:
                return PlanValidation(False, reason)
            definition = world.target(target)
            if step.action == ActionType.MOVE_TO and not definition.navigation_target:
                return PlanValidation(False, "TARGET_NOT_NAVIGABLE")
            if step.action != ActionType.MOVE_TO and not definition.look_at_target:
                return PlanValidation(False, "TARGET_NOT_LOOKABLE")
            return PlanValidation(True)

        if step.action == ActionType.SPEAK:
            if not step.text or not step.text.strip():
                return PlanValidation(False, "SPEECH_REQUIRED")
            return PlanValidation(True)

        if step.action in {ActionType.WAVE, ActionType.DANCE}:
            # PHASE 28: a bare gesture runs the built-in state; the only valid
            # target is a registered motion reference. A spatial name here is
            # rejected before dispatch so it can never reach the bridge as an
            # unresolvable animation lookup.
            if step.target is None or _is_motion_reference(step.target):
                return PlanValidation(True)
            return PlanValidation(False, "MOTION_REFERENCE_INVALID")

        if step.action in {ActionType.RUN, ActionType.STOP}:
            return PlanValidation(True)

        if step.action == ActionType.PICK_UP_PHONE:
            if world.phone_state != "World" and world.phone_state != "Placed":
                return PlanValidation(False, "PHONE_ALREADY_HELD")
            return PlanValidation(True)

        if step.action == ActionType.PLACE_PHONE:
            if world.phone_state != "Held":
                return PlanValidation(False, "PHONE_NOT_HELD")
            if step.target not in {"TABLE", "BENCH", "FLOOR"}:
                return PlanValidation(False, "PLACEMENT_TARGET_INVALID")
            return PlanValidation(True)

        if step.action == ActionType.START_STREAM:
            if world.stream_active:
                return PlanValidation(False, "STREAM_ALREADY_ACTIVE")
            if not world.stream_camera_available:
                return PlanValidation(False, "CAMERA_UNAVAILABLE")
            return PlanValidation(True)

        if step.action == ActionType.STOP_STREAM:
            if not world.stream_active:
                return PlanValidation(False, "STREAM_NOT_ACTIVE")
            return PlanValidation(True)

        if step.action == ActionType.SET_CAMERA:
            target, reason = resolver.resolve(step.target)
            if target is None:
                return PlanValidation(False, reason)
            return PlanValidation(True)

        return PlanValidation(False, "UNKNOWN_ACTION")
