"""Typed models for registered skills and their runtime state."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from ..brain.models import ActionType, ExecutionStatus, format_utc


class SkillCategory(str, Enum):
    MOVEMENT = "MOVEMENT"
    INTERACTION = "INTERACTION"
    SOCIAL = "SOCIAL"
    PHONE = "PHONE"
    LIVESTREAM = "LIVESTREAM"
    COMPOSITE = "COMPOSITE"


class SkillConditionType(str, Enum):
    CHARACTER_STATE = "CHARACTER_STATE"
    PHONE_STATE = "PHONE_STATE"
    STREAM_STATE = "STREAM_STATE"
    TARGET_EXISTS = "TARGET_EXISTS"
    TARGET_REACHABLE = "TARGET_REACHABLE"
    SPACE_AVAILABLE = "SPACE_AVAILABLE"


class SkillConditionOperator(str, Enum):
    EQUALS = "EQUALS"
    NOT_EQUALS = "NOT_EQUALS"
    IS_TRUE = "IS_TRUE"
    IS_FALSE = "IS_FALSE"
    IN = "IN"
    PRESENT = "PRESENT"


class SkillParameterType(str, Enum):
    STRING = "STRING"
    NUMBER = "NUMBER"
    BOOLEAN = "BOOLEAN"
    TARGET = "TARGET"


class SkillStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    INTERRUPTED = "INTERRUPTED"
    CANCELLED = "CANCELLED"


@dataclass(frozen=True)
class SkillCondition:
    condition_type: SkillConditionType
    operator: SkillConditionOperator
    value: Any = None
    failure_reason: str = "CONDITION_FAILED"

    def to_dict(self) -> dict[str, Any]:
        value = self.value
        if isinstance(value, tuple):
            value = list(value)
        return {
            "type": self.condition_type.value,
            "operator": self.operator.value,
            "value": value,
            "failure_reason": self.failure_reason,
        }


@dataclass(frozen=True)
class SkillParameter:
    name: str
    parameter_type: SkillParameterType
    required: bool = False
    default: Any = None
    choices: tuple[Any, ...] | None = None
    minimum: float | None = None
    maximum: float | None = None
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.parameter_type.value,
            "required": self.required,
            "default": self.default,
            "choices": list(self.choices) if self.choices is not None else None,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "description": self.description,
        }


@dataclass(frozen=True)
class SkillActionStep:
    action: ActionType
    target_parameter: Optional[str] = None
    text_parameter: Optional[str] = None
    constant_target: Optional[str] = None
    constant_text: Optional[str] = None
    position: Optional[tuple[float, float, float]] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "target_parameter": self.target_parameter,
            "text_parameter": self.text_parameter,
            "constant_target": self.constant_target,
            "constant_text": self.constant_text,
            "position": list(self.position) if self.position is not None else None,
        }


@dataclass(frozen=True)
class SkillCompositionStep:
    skill_id: str
    parameters: dict[str, Any] = field(default_factory=dict)
    required: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "parameters": dict(self.parameters),
            "required": self.required,
        }


@dataclass(frozen=True)
class Skill:
    skill_id: str
    name: str
    description: str
    category: SkillCategory
    preconditions: tuple[SkillCondition, ...] = ()
    parameters: tuple[SkillParameter, ...] = ()
    steps: tuple[SkillActionStep, ...] = ()
    composition: tuple[SkillCompositionStep, ...] = ()
    animation_reference: Optional[str] = None
    estimated_duration_seconds: float = 1.0
    effects: tuple[str, ...] = ()
    cooldown_seconds: float = 0.0
    timeout_seconds: float = 120.0
    tags: tuple[str, ...] = ()
    version: int = 1
    enabled: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.skill_id, str) or not self.skill_id.strip():
            raise ValueError("skill_id is required")
        if self.skill_id != self.skill_id.strip().lower().replace(" ", "_"):
            raise ValueError("skill_id must be a stable lowercase identifier")
        if not self.name.strip():
            raise ValueError("skill name is required")
        if not self.description.strip():
            raise ValueError("skill description is required")
        if not self.steps and not self.composition:
            raise ValueError("skill requires executable steps or composition")
        if self.steps and self.composition:
            raise ValueError("skill cannot mix direct steps and composition")
        if self.estimated_duration_seconds < 0 or self.cooldown_seconds < 0:
            raise ValueError("durations and cooldowns cannot be negative")
        if self.timeout_seconds <= 0:
            raise ValueError("skill timeout must be positive")

    @property
    def parameter_names(self) -> frozenset[str]:
        return frozenset(parameter.name for parameter in self.parameters)

    def to_dict(self, include_steps: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "skill_id": self.skill_id,
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "parameters": [parameter.to_dict() for parameter in self.parameters],
            "animation_reference": self.animation_reference,
            "estimated_duration_seconds": self.estimated_duration_seconds,
            "effects": list(self.effects),
            "cooldown_seconds": self.cooldown_seconds,
            "timeout_seconds": self.timeout_seconds,
            "tags": list(self.tags),
            "version": self.version,
            "enabled": self.enabled,
            "preconditions": [condition.to_dict() for condition in self.preconditions],
        }
        if include_steps:
            payload["steps"] = [step.to_dict() for step in self.steps]
            payload["composition"] = [step.to_dict() for step in self.composition]
        return payload


@dataclass(frozen=True)
class SkillSelection:
    skill_id: str
    parameters: dict[str, Any] = field(default_factory=dict)
    reason: str = "selected"


@dataclass
class SkillExecution:
    execution_id: str
    skill_id: str
    plan_id: str
    started_at: datetime
    status: SkillStatus = SkillStatus.RUNNING
    current_step: int = 0
    total_steps: int = 0
    completed_at: Optional[datetime] = None
    result: Optional["SkillEvaluation"] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "skill_id": self.skill_id,
            "plan_id": self.plan_id,
            "status": self.status.value,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "started_at": format_utc(self.started_at),
            "completed_at": format_utc(self.completed_at) if self.completed_at else None,
            "result": self.result.to_dict() if self.result else None,
        }


@dataclass(frozen=True)
class SkillEvaluation:
    skill_id: str
    execution_id: str
    status: SkillStatus
    duration_seconds: float
    steps_completed: int
    steps_failed: int
    interruption_reason: Optional[str] = None
    viewer_response: Optional[str] = None
    world_effects: tuple[str, ...] = ()
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "execution_id": self.execution_id,
            "status": self.status.value,
            "duration_seconds": round(self.duration_seconds, 3),
            "steps_completed": self.steps_completed,
            "steps_failed": self.steps_failed,
            "interruption_reason": self.interruption_reason,
            "viewer_response": self.viewer_response,
            "world_effects": list(self.world_effects),
            "notes": self.notes,
        }


@dataclass(frozen=True)
class SkillValidation:
    valid: bool
    reason: str = "OK"
    skill: Optional[Skill] = None
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "reason": self.reason,
            "skill_id": self.skill.skill_id if self.skill else None,
            "parameters": self.parameters,
        }


__all__ = [
    "ActionType",
    "ExecutionStatus",
    "Skill",
    "SkillActionStep",
    "SkillCategory",
    "SkillCompositionStep",
    "SkillCondition",
    "SkillConditionOperator",
    "SkillConditionType",
    "SkillEvaluation",
    "SkillExecution",
    "SkillParameter",
    "SkillParameterType",
    "SkillSelection",
    "SkillStatus",
    "SkillValidation",
]
