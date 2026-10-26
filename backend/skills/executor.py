"""Validate, expand, and evaluate skills without becoming a second executor."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from itertools import count
from typing import Any, Callable, Optional

from ..brain.models import (
    ActionType,
    ActionResult,
    ExecutionStatus,
    Goal,
    Plan,
    PlanStep,
    WorldState,
)
from ..brain.planner import BrainPlanner
from ..runtime import instrument
from .conditions import SkillConditionEvaluator, validate_parameters
from .library import SkillLibrary
from .models import (
    Skill,
    SkillEvaluation,
    SkillExecution,
    SkillSelection,
    SkillStatus,
    SkillValidation,
)


_MAX_EVALUATIONS = 50
_ACTION_WORLD_EFFECTS = {
    "MOVE_TO": "character_position_changed",
    "STOP": "character_stopped",
    "TURN_TO": "character_rotation_changed",
    "LOOK_AT": "character_gaze_changed",
    "WAVE": "social_animation_played",
    "DANCE": "social_animation_played",
    "RUN": "character_position_changed",
    "PICK_UP_PHONE": "phone_state_changed",
    "PLACE_PHONE": "phone_state_changed",
    "START_STREAM": "stream_started",
    "STOP_STREAM": "stream_stopped",
    "SET_CAMERA": "camera_position_changed",
    "SPEAK": "speech_delivered",
}


@dataclass(frozen=True)
class SkillPlanResult:
    plan: Optional[Plan] = None
    reason: str = "OK"
    execution: Optional[SkillExecution] = None


class SkillExecutor:
    """Expand an allowlisted skill into the Brain's existing action plan."""

    def __init__(
        self,
        library: SkillLibrary,
        planner: BrainPlanner,
        clock: Callable[[], datetime],
        id_factory: Callable[[], str] | None = None,
    ) -> None:
        self.library = library
        self.planner = planner
        self._clock = clock
        self._id_factory = id_factory
        self._id_sequence = count(1)
        self._condition_evaluator = SkillConditionEvaluator()
        self._active: Optional[SkillExecution] = None
        self._last_success_at: dict[str, datetime] = {}
        self._evaluations: deque[SkillEvaluation] = deque(maxlen=_MAX_EVALUATIONS)

    @property
    def active(self) -> Optional[SkillExecution]:
        return self._active

    @property
    def evaluations(self) -> tuple[SkillEvaluation, ...]:
        return tuple(self._evaluations)

    def parse_selection(self, raw_output: str) -> tuple[SkillSelection | None, str]:
        try:
            payload = json.loads(raw_output)
        except (json.JSONDecodeError, TypeError):
            return None, "NOT_A_SKILL_SELECTION"
        if not isinstance(payload, dict) or "skill_id" not in payload:
            return None, "NOT_A_SKILL_SELECTION"

        skill_id = payload.get("skill_id")
        parameters = payload.get("parameters", {})
        reason = payload.get("reason", "skill selected")
        if not isinstance(skill_id, str) or not skill_id.strip():
            return None, "INVALID_SKILL_SELECTION"
        if not isinstance(parameters, dict):
            return None, "INVALID_SKILL_PARAMETERS"
        if not isinstance(reason, str) or not reason.strip():
            return None, "INVALID_REASON"
        return SkillSelection(skill_id.strip().lower(), dict(parameters), reason.strip()), "OK"

    def availability(
        self,
        skill_id: str,
        world: WorldState,
        parameters: dict[str, Any] | None = None,
        now: datetime | None = None,
        override_cooldown: bool = False,
    ) -> SkillValidation:
        skill = self.library.get(skill_id)
        if skill is None:
            return SkillValidation(False, "SKILL_NOT_FOUND")
        if not skill.enabled:
            return SkillValidation(False, "SKILL_DISABLED")

        resolved, reason = validate_parameters(skill, parameters)
        if resolved is None:
            return SkillValidation(False, reason)
        if not override_cooldown and self._is_on_cooldown(skill, now):
            return SkillValidation(False, "SKILL_ON_COOLDOWN")
        return self._condition_evaluator.evaluate(skill, world, resolved)

    def can_execute(
        self,
        skill_id: str,
        world: WorldState,
        parameters: dict[str, Any] | None = None,
        now: datetime | None = None,
    ) -> bool:
        return self.availability(skill_id, world, parameters, now).valid

    def create_plan(
        self,
        selection: SkillSelection,
        goal: Goal,
        world: WorldState,
        now: datetime | None = None,
    ) -> SkillPlanResult:
        now = now or self._clock()
        validation = self.availability(selection.skill_id, world, selection.parameters, now)
        if not validation.valid or validation.skill is None:
            instrument.count("skill.plan_rejected")
            return SkillPlanResult(reason=validation.reason)

        steps: list[PlanStep] = []
        expansion = self._expand_skill(validation.skill, validation.parameters, world, steps)
        if expansion is not None:
            instrument.count("skill.plan_rejected")
            return SkillPlanResult(reason=expansion)
        if not steps:
            instrument.count("skill.plan_rejected")
            return SkillPlanResult(reason="EMPTY_SKILL")

        for step in steps:
            step_validation = self.planner.validate_step(step, world)
            if not step_validation.valid:
                instrument.count("skill.plan_rejected")
                return SkillPlanResult(reason=step_validation.reason)

        plan = self.planner.create_plan(goal, validation.skill.name, steps)
        execution = self._begin_execution(validation.skill, plan, now)
        return SkillPlanResult(plan, "OK", execution)

    def observe_success(self, result: ActionResult) -> Optional[SkillEvaluation]:
        execution = self._active
        if execution is None:
            return None
        execution.current_step += 1
        if execution.total_steps <= 0 or execution.current_step < execution.total_steps:
            return None

        skill = self.library.get(execution.skill_id)
        completed_at = self._clock()
        effects = list(skill.effects if skill else ())
        action_effect = _ACTION_WORLD_EFFECTS.get(result.action.value)
        if action_effect and action_effect not in effects:
            effects.append(action_effect)
        evaluation = SkillEvaluation(
            skill_id=execution.skill_id,
            execution_id=execution.execution_id,
            status=SkillStatus.SUCCESS,
            duration_seconds=max(0.0, (completed_at - execution.started_at).total_seconds()),
            steps_completed=execution.current_step,
            steps_failed=0,
            world_effects=tuple(effects),
            viewer_response=result.viewer_response or "none",
            notes="completed through the existing action executor",
        )
        self._finish(execution, completed_at, evaluation)
        return evaluation

    def observe_failure(
        self,
        result: ActionResult,
        final: bool,
        interrupted: bool = False,
        viewer_response: str | None = None,
    ) -> None:
        execution = self._active
        if execution is None or not final:
            return
        completed_at = self._clock()
        status = SkillStatus.INTERRUPTED if interrupted else SkillStatus.FAILED
        evaluation = SkillEvaluation(
            skill_id=execution.skill_id,
            execution_id=execution.execution_id,
            status=status,
            duration_seconds=max(0.0, (completed_at - execution.started_at).total_seconds()),
            steps_completed=execution.current_step,
            steps_failed=1,
            interruption_reason=result.reason,
            viewer_response=viewer_response,
            world_effects=tuple(),
            notes="failure propagated from the existing action result",
        )
        self._finish(execution, completed_at, evaluation)

    def cancel_active(self, reason: str = "cancelled") -> SkillEvaluation | None:
        execution = self._active
        if execution is None:
            return None
        completed_at = self._clock()
        evaluation = SkillEvaluation(
            skill_id=execution.skill_id,
            execution_id=execution.execution_id,
            status=SkillStatus.CANCELLED,
            duration_seconds=max(0.0, (completed_at - execution.started_at).total_seconds()),
            steps_completed=execution.current_step,
            steps_failed=0,
            interruption_reason=reason,
            viewer_response="none",
            world_effects=tuple(),
            notes="skill cancellation handed back to the authoritative Brain",
        )
        self._finish(execution, completed_at, evaluation)
        return evaluation

    def timeout_active(self, now: datetime | None = None) -> bool:
        execution = self._active
        if execution is None:
            return False
        skill = self.library.get(execution.skill_id)
        timeout = skill.timeout_seconds if skill else 120.0
        current = now or self._clock()
        if (current - execution.started_at).total_seconds() <= timeout:
            return False
        self.observe_failure(
            ActionResult(ActionType.STOP, ExecutionStatus.FAILED, "SKILL_TIMEOUT"),
            final=True,
            interrupted=True,
            viewer_response="none",
        )
        return True

    def runtime_state(self) -> dict[str, Any]:
        active = self._active.to_dict() if self._active else None
        return {
            "active": active,
            "recent_evaluations": [evaluation.to_dict() for evaluation in reversed(self._evaluations)],
        }

    def _expand_skill(
        self,
        skill: Skill,
        parameters: dict[str, Any],
        world: WorldState,
        output: list[PlanStep],
        trail: tuple[str, ...] = (),
) -> str | None:
        if skill.skill_id in trail:
            return "COMPOSITION_CYCLE"
        if skill.composition:
            for composition_step in skill.composition:
                child = self.library.get(composition_step.skill_id)
                if child is None:
                    return "COMPOSITION_DEPENDENCY_MISSING"
                child_validation = self.availability(
                    child.skill_id,
                    world,
                    composition_step.parameters,
                    override_cooldown=True,
                )
                if not child_validation.valid:
                    if composition_step.required:
                        return child_validation.reason
                    continue
                assert child_validation.parameters is not None
                failure = self._expand_skill(
                    child,
                    child_validation.parameters,
                    world,
                    output,
                    trail + (skill.skill_id,),
                )
                if failure is not None and composition_step.required:
                    return failure
            return None

        for action_step in skill.steps:
            target = None
            if action_step.target_parameter:
                target = parameters.get(action_step.target_parameter)
            elif action_step.constant_target:
                target = action_step.constant_target
            text = None
            if action_step.text_parameter:
                text = parameters.get(action_step.text_parameter)
            elif action_step.constant_text:
                text = action_step.constant_text
            output.append(
                PlanStep(
                    action=action_step.action,
                    target=target,
                    text=text,
                    position=action_step.position,
                )
            )
        return None

    def _begin_execution(self, skill: Skill, plan: Plan, now: datetime) -> SkillExecution:
        if self._active is not None and self._active.status == SkillStatus.RUNNING:
            self.cancel_active("replaced_by_new_skill")
        execution_id = (
            self._id_factory()
            if self._id_factory
            else f"skill_execution_{next(self._id_sequence):06d}"
        )
        self._active = SkillExecution(
            execution_id=execution_id,
            skill_id=skill.skill_id,
            plan_id=plan.plan_id,
            started_at=now,
            total_steps=len(plan.steps),
        )
        instrument.count("skill.started")
        return self._active

    def _finish(self, execution: SkillExecution, completed_at: datetime, evaluation: SkillEvaluation) -> None:
        execution.status = evaluation.status
        execution.completed_at = completed_at
        execution.result = evaluation
        if evaluation.status == SkillStatus.SUCCESS:
            self._last_success_at[execution.skill_id] = completed_at
            instrument.count("skill.success")
        elif evaluation.status == SkillStatus.CANCELLED:
            instrument.count("skill.cancelled")
        else:
            instrument.count("skill.failed")
            instrument.record_error("skill", evaluation.interruption_reason or evaluation.status)
        instrument.observe_ms(
            "skill.execution_ms", max(0.0, evaluation.duration_seconds) * 1000.0
        )
        instrument.gauge("skill.evaluations_retained", float(len(self._evaluations) + 1))
        self._evaluations.append(evaluation)
        if self._active is execution:
            self._active = None

    def _is_on_cooldown(self, skill: Skill, now: datetime | None) -> bool:
        if skill.cooldown_seconds <= 0:
            return False
        completed_at = self._last_success_at.get(skill.skill_id)
        if completed_at is None:
            return False
        current = now or self._clock()
        elapsed = (current - completed_at).total_seconds()
        return elapsed < skill.cooldown_seconds
