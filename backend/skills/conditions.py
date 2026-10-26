"""Central condition and parameter validation for skills."""

from __future__ import annotations

from typing import Any, Iterable

from ..brain.targets import TargetResolver
from .models import (
    Skill,
    SkillCondition,
    SkillConditionOperator,
    SkillParameter,
    SkillValidation,
)


PHONE_EXISTS_STATES = {"World", "Held", "Placed"}
BUSY_CHARACTER_STATES = {"dance", "dancing", "wave", "waving", "run", "running"}


class SkillConditionEvaluator:
    """Evaluate lightweight world conditions without touching Unity."""

    def evaluate(
        self,
        skill: Skill,
        world,
        parameters: dict[str, Any],
    ) -> SkillValidation:
        for condition in skill.preconditions:
            failure = self.evaluate_condition(condition, world, parameters)
            if failure is not None:
                return SkillValidation(False, failure)
        return SkillValidation(True, "OK", skill, dict(parameters))

    def evaluate_condition(
        self,
        condition: SkillCondition,
        world,
        parameters: dict[str, Any],
    ) -> str | None:
        if condition.condition_type.value == "CHARACTER_STATE":
            actual = str(world.character_posture).casefold()
            return _compare_state(actual, condition)
        if condition.condition_type.value == "SPACE_AVAILABLE":
            actual = str(world.character_posture).casefold() not in BUSY_CHARACTER_STATES
            return _compare_bool(actual, condition)
        if condition.condition_type.value == "PHONE_STATE":
            if condition.operator == SkillConditionOperator.PRESENT:
                passed = world.phone_state in PHONE_EXISTS_STATES
            else:
                actual = str(world.phone_state).casefold()
                expected = _normalize_scalar(condition.value)
                if condition.operator == SkillConditionOperator.IN:
                    passed = actual in _normalize_collection(condition.value)
                else:
                    passed = _compare_scalar(actual, expected, condition.operator)
            return None if passed else condition.failure_reason
        if condition.condition_type.value == "STREAM_STATE":
            if condition.operator == SkillConditionOperator.IS_TRUE:
                passed = bool(world.stream_active)
            elif condition.operator == SkillConditionOperator.IS_FALSE:
                passed = not bool(world.stream_active)
            else:
                passed = _compare_scalar(bool(world.stream_active), bool(condition.value), condition.operator)
            return None if passed else condition.failure_reason
        if condition.condition_type.value in {"TARGET_EXISTS", "TARGET_REACHABLE"}:
            target_name = parameters.get("target")
            if target_name is None:
                return condition.failure_reason
            resolved, _ = TargetResolver(world).resolve(str(target_name))
            definition = world.target(resolved) if resolved is not None else None
            if definition is None:
                return condition.failure_reason
            if condition.condition_type.value == "TARGET_REACHABLE" and condition.operator == SkillConditionOperator.IS_TRUE:
                # PHASE 25: Unity's navmesh verdict wins when it reported one;
                # otherwise the legacy navigable flag remains the authority.
                reachable = definition.reachable
                if reachable is False or (reachable is None and not definition.navigation_target):
                    return condition.failure_reason
            return None
        return condition.failure_reason


def validate_parameters(skill: Skill, supplied: dict[str, Any] | None) -> tuple[dict[str, Any] | None, str]:
    provided = dict(supplied or {})
    resolved: dict[str, Any] = {}
    for parameter in skill.parameters:
        if parameter.name not in provided:
            if parameter.required and parameter.default is None:
                return None, f"MISSING_PARAMETER:{parameter.name}"
            if parameter.default is not None:
                resolved[parameter.name] = parameter.default
            continue

        value = provided[parameter.name]
        validated, reason = _validate_value(parameter, value)
        if reason is not None:
            return None, reason
        resolved[parameter.name] = validated

    unknown = set(provided) - set(skill.parameter_names)
    if unknown:
        return None, "UNKNOWN_PARAMETER:" + ",".join(sorted(unknown))
    return resolved, "OK"


def _validate_value(parameter: SkillParameter, value: Any) -> tuple[Any, str | None]:
    try:
        if parameter.parameter_type.value == "STRING":
            validated = str(value)
            if not validated.strip():
                return None, f"INVALID_PARAMETER:{parameter.name}"
        elif parameter.parameter_type.value == "NUMBER":
            validated = float(value)
        elif parameter.parameter_type.value == "BOOLEAN":
            validated = bool(value)
        else:
            validated = str(value)
    except (TypeError, ValueError):
        return None, f"INVALID_PARAMETER:{parameter.name}"

    if parameter.choices is not None and validated not in parameter.choices:
        return None, f"INVALID_PARAMETER:{parameter.name}"
    if parameter.minimum is not None and float(validated) < parameter.minimum:
        return None, f"INVALID_PARAMETER:{parameter.name}"
    if parameter.maximum is not None and float(validated) > parameter.maximum:
        return None, f"INVALID_PARAMETER:{parameter.name}"
    return validated, None


def _compare_state(actual: str, condition: SkillCondition) -> str | None:
    expected = _normalize_scalar(condition.value)
    if condition.operator == SkillConditionOperator.IN:
        passed = actual in _normalize_collection(condition.value)
    else:
        passed = _compare_scalar(actual, expected, condition.operator)
    return None if passed else condition.failure_reason


def _compare_bool(actual: bool, condition: SkillCondition) -> str | None:
    if condition.operator == SkillConditionOperator.IS_TRUE:
        passed = bool(actual)
    elif condition.operator == SkillConditionOperator.IS_FALSE:
        passed = not actual
    else:
        passed = _compare_scalar(actual, bool(condition.value), condition.operator)
    return None if passed else condition.failure_reason


def _compare_scalar(actual: Any, expected: Any, operator: SkillConditionOperator) -> bool:
    if operator == SkillConditionOperator.EQUALS:
        return actual == expected
    if operator == SkillConditionOperator.NOT_EQUALS:
        return actual != expected
    if operator == SkillConditionOperator.IN:
        return actual in expected
    if operator == SkillConditionOperator.PRESENT:
        return actual is not None
    return False


def _normalize_scalar(value: Any) -> Any:
    return value.casefold() if isinstance(value, str) else value


def _normalize_collection(value: Any) -> set[Any]:
    if isinstance(value, str):
        return {value.casefold()}
    if isinstance(value, Iterable):
        return {_normalize_scalar(item) for item in value}
    return {value}
