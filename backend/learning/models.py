"""Typed Phase 10 learning models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from ..chat.models import format_utc
from ..memory.models import DEFAULT_MEMORY_DB_PATH
from ..brain.models import ActionType


@dataclass(frozen=True)
class LearningConfig:
    enabled: bool = True
    database_path: Any = DEFAULT_MEMORY_DB_PATH
    exploration_epsilon: float = 0.15
    exploration_confidence_bonus: float = 0.25
    confidence_growth: float = 0.05
    min_explore_confidence: float = 0.55
    min_composite_trials: int = 3
    min_composite_score: float = 0.65
    max_contexts_per_skill: int = 256
    max_child_skills: int = 8

    def __post_init__(self) -> None:
        if not 0.0 <= self.exploration_epsilon <= 1.0:
            raise ValueError("exploration epsilon must be between 0 and 1")
        if self.min_composite_trials < 1:
            raise ValueError("composite trial threshold must be positive")


class MotionSource(str, Enum):
    IMPORTED = "IMPORTED"
    RECORDED = "RECORDED"
    PROCEDURAL = "PROCEDURAL"
    GENERATED = "GENERATED"
    EXTERNAL = "EXTERNAL"


class MotionValidationStatus(str, Enum):
    PENDING = "PENDING"
    VALID = "VALID"
    INVALID = "INVALID"


class SkeletonCompatibility(str, Enum):
    COMPATIBLE = "COMPATIBLE"
    RETARGETABLE = "RETARGETABLE"
    INCOMPATIBLE = "INCOMPATIBLE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class MotionDefinition:
    motion_id: str
    name: str
    description: str
    animation_reference: str
    duration_seconds: float
    tags: tuple[str, ...] = ()
    source: MotionSource = MotionSource.IMPORTED
    parameters: tuple[str, ...] = ()
    execution_action: ActionType = ActionType.DANCE
    enabled: bool = True
    version: int = 1
    source_path: Optional[str] = None
    source_format: Optional[str] = None
    clip_name: Optional[str] = None
    frame_count: Optional[int] = None
    sample_rate: Optional[float] = None
    bone_count: Optional[int] = None
    root_bone_name: Optional[str] = None
    skeleton_status: SkeletonCompatibility = SkeletonCompatibility.UNKNOWN
    validation_status: MotionValidationStatus = MotionValidationStatus.PENDING
    source_motion_id: Optional[str] = None
    source_version: Optional[int] = None
    processing_version: Optional[int] = None
    quality_score: Optional[float] = None
    quality_flags: tuple[str, ...] = ()
    cleanup_status: str = "NOT_REQUIRED"
    processing_timestamp: Optional[str] = None
    last_error: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.motion_id or self.motion_id != self.motion_id.strip().lower().replace(" ", "_"):
            raise ValueError("motion_id must be a stable lowercase identifier")
        if not self.name.strip() or not self.description.strip() or not self.animation_reference.strip():
            raise ValueError("motion name, description, and animation_reference are required")
        if self.duration_seconds <= 0:
            raise ValueError("motion duration must be positive")
        if self.version <= 0:
            raise ValueError("motion version must be positive")
        if not isinstance(self.execution_action, ActionType):
            raise ValueError("execution_action must be an allowlisted brain action")
        if not isinstance(self.skeleton_status, SkeletonCompatibility):
            raise ValueError("skeleton_status must be a known compatibility value")
        if not isinstance(self.validation_status, MotionValidationStatus):
            raise ValueError("validation_status must be a known validation value")
        if self.source_format is not None and not self.source_format.strip():
            raise ValueError("motion source_format cannot be blank")
        if self.clip_name is not None and not self.clip_name.strip():
            raise ValueError("motion clip_name cannot be blank")
        if self.root_bone_name is not None and not self.root_bone_name.strip():
            raise ValueError("motion root_bone_name cannot be blank")
        if self.frame_count is not None and self.frame_count <= 0:
            raise ValueError("motion frame_count must be positive")
        if self.sample_rate is not None and self.sample_rate <= 0:
            raise ValueError("motion sample_rate must be positive")
        if self.bone_count is not None and self.bone_count <= 0:
            raise ValueError("motion bone_count must be positive")
        if self.source_motion_id is not None and not self.source_motion_id.strip():
            raise ValueError("motion source_motion_id cannot be blank")
        if self.source_version is not None and self.source_version <= 0:
            raise ValueError("motion source_version must be positive")
        if self.processing_version is not None and self.processing_version <= 0:
            raise ValueError("motion processing_version must be positive")
        if self.quality_score is not None and not 0.0 <= self.quality_score <= 1.0:
            raise ValueError("motion quality_score must be between 0 and 1")
        if not self.cleanup_status.strip():
            raise ValueError("motion cleanup_status cannot be blank")

    def to_dict(self) -> dict[str, Any]:
        return {
            "motion_id": self.motion_id,
            "name": self.name,
            "description": self.description,
            "animation_reference": self.animation_reference,
            "duration_seconds": self.duration_seconds,
            "tags": list(self.tags),
            "source": self.source.value,
            "parameters": list(self.parameters),
            "execution_action": self.execution_action.value,
            "enabled": self.enabled,
            "version": self.version,
            "source_path": self.source_path,
            "source_format": self.source_format,
            "clip_name": self.clip_name,
            "frame_count": self.frame_count,
            "sample_rate": self.sample_rate,
            "bone_count": self.bone_count,
            "root_bone_name": self.root_bone_name,
            "skeleton_status": self.skeleton_status.value,
            "validation_status": self.validation_status.value,
            "source_motion_id": self.source_motion_id,
            "source_version": self.source_version,
            "processing_version": self.processing_version,
            "quality_score": self.quality_score,
            "quality_flags": list(self.quality_flags),
            "cleanup_status": self.cleanup_status,
            "processing_timestamp": self.processing_timestamp,
            "last_error": self.last_error,
        }


@dataclass
class SkillKnowledge:
    skill_id: str
    context_key: str
    trials: int = 0
    successes: int = 0
    failures: int = 0
    interruptions: int = 0
    positive_reactions: int = 0
    negative_reactions: int = 0
    total_score: float = 0.0
    total_duration_seconds: float = 0.0
    confidence: float = 0.35
    exploration_count: int = 0
    exploitation_count: int = 0
    last_used: Optional[datetime] = None
    version: int = 1

    @property
    def average_score(self) -> float:
        return self.total_score / self.trials if self.trials else 0.0

    @property
    def success_rate(self) -> float:
        return self.successes / self.trials if self.trials else 0.0

    @property
    def average_duration_seconds(self) -> float:
        return self.total_duration_seconds / self.trials if self.trials else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "context_key": self.context_key,
            "trials": self.trials,
            "successes": self.successes,
            "failures": self.failures,
            "interruptions": self.interruptions,
            "positive_reactions": self.positive_reactions,
            "negative_reactions": self.negative_reactions,
            "success_rate": round(self.success_rate, 4),
            "average_score": round(self.average_score, 4),
            "average_duration_seconds": round(self.average_duration_seconds, 3),
            "confidence": round(self.confidence, 4),
            "exploration_count": self.exploration_count,
            "exploitation_count": self.exploitation_count,
            "last_used": format_utc(self.last_used) if self.last_used else None,
            "version": self.version,
        }


@dataclass(frozen=True)
class ExperienceRecord:
    skill_id: str
    context_key: str
    execution_id: str
    status: str
    outcome_score: float
    duration_seconds: float
    viewer_response: Optional[str]
    interruption_reason: Optional[str]
    timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "context_key": self.context_key,
            "execution_id": self.execution_id,
            "status": self.status,
            "outcome_score": round(self.outcome_score, 4),
            "duration_seconds": round(self.duration_seconds, 3),
            "viewer_response": self.viewer_response,
            "interruption_reason": self.interruption_reason,
            "timestamp": format_utc(self.timestamp),
            "metadata": dict(self.metadata),
        }


@dataclass
class CompositeKnowledge:
    composite_skill_id: str
    child_skill_ids: tuple[str, ...]
    context_key: str
    trials: int = 0
    successes: int = 0
    total_score: float = 0.0
    confidence: float = 0.35
    created_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    enabled: bool = False
    child_parameters: tuple[dict[str, Any], ...] = ()

    @property
    def average_score(self) -> float:
        return self.total_score / self.trials if self.trials else 0.0

    @property
    def success_rate(self) -> float:
        return self.successes / self.trials if self.trials else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "composite_skill_id": self.composite_skill_id,
            "child_skill_ids": list(self.child_skill_ids),
            "child_parameters": [dict(parameters) for parameters in self.child_parameters],
            "context_key": self.context_key,
            "trials": self.trials,
            "successes": self.successes,
            "success_rate": round(self.success_rate, 4),
            "average_score": round(self.average_score, 4),
            "confidence": round(self.confidence, 4),
            "created_at": format_utc(self.created_at) if self.created_at else None,
            "last_used": format_utc(self.last_used) if self.last_used else None,
            "enabled": self.enabled,
        }
