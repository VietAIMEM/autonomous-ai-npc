"""Typed Phase 17 motion-processing models."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_UNITY_EXECUTABLE = Path(
    "C:/Program Files/Unity/Hub/Editor/6000.6.0f1/Editor/Unity.exe"
)
DEFAULT_UNITY_PROJECT = PROJECT_ROOT / "unity" / "3d_unity"
DEFAULT_MOTION_PROCESSING_DB_PATH = PROJECT_ROOT / "backend" / "data" / "motion_processing.db"


class MotionProcessingState(str, Enum):
    INPUT = "INPUT"
    ANALYZING = "ANALYZING"
    RETARGETING = "RETARGETING"
    CLEANING = "CLEANING"
    VALIDATING = "VALIDATING"
    READY = "READY"
    APPROVED = "APPROVED"
    REGISTERED = "REGISTERED"
    FAILED = "FAILED"
    REJECTED = "REJECTED"


class QualityStatus(str, Enum):
    VALID = "VALID"
    FLAGGED = "FLAGGED"
    REJECTED = "REJECTED"


class RetargetStatus(str, Enum):
    MAPPED = "MAPPED"
    NOT_REQUIRED = "NOT_REQUIRED"
    FAILED = "FAILED"


class CleanupStatus(str, Enum):
    NOT_REQUIRED = "NOT_REQUIRED"
    APPLIED = "APPLIED"
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class MotionProcessingConfig:
    unity_executable: Path = DEFAULT_UNITY_EXECUTABLE
    unity_project_path: Path = DEFAULT_UNITY_PROJECT
    database_path: Path = DEFAULT_MOTION_PROCESSING_DB_PATH
    timeout_seconds: float = 420.0
    playback_timeout_seconds: float = 420.0
    max_queue_depth: int = 2
    max_concurrent_processes: int = 1
    max_retained_jobs: int = 64
    worker_poll_interval_seconds: float = 0.05

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0 or self.playback_timeout_seconds <= 0:
            raise ValueError("processing timeouts must be positive")
        if self.max_queue_depth <= 0:
            raise ValueError("max_queue_depth must be positive")
        if self.max_retained_jobs <= 0:
            raise ValueError("max_retained_jobs must be positive")
        if self.max_concurrent_processes != 1:
            raise ValueError("max_concurrent_processes must be 1")
        if self.worker_poll_interval_seconds <= 0:
            raise ValueError("worker_poll_interval_seconds must be positive")


@dataclass(frozen=True)
class MotionProcessingRequest:
    source_motion_id: str
    source_version: int
    processed_motion_id: str
    processing_version: int
    name: str
    description: str
    tags: tuple[str, ...] = ()
    skill_category: str = "MOVEMENT"

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_motion_id": self.source_motion_id,
            "source_version": self.source_version,
            "processed_motion_id": self.processed_motion_id,
            "processing_version": self.processing_version,
            "name": self.name,
            "description": self.description,
            "tags": list(self.tags),
            "skill_category": self.skill_category,
        }


@dataclass(frozen=True)
class SkeletonMappingEntry:
    human_bone: str
    source_bone: str
    target_bone: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SkeletonMappingEntry":
        return cls(
            human_bone=str(payload.get("humanBone") or payload.get("human_bone") or ""),
            source_bone=str(payload.get("sourceBone") or payload.get("source_bone") or ""),
            target_bone=str(payload.get("targetBone") or payload.get("target_bone") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "human_bone": self.human_bone,
            "source_bone": self.source_bone,
            "target_bone": self.target_bone,
        }


@dataclass(frozen=True)
class MotionProcessingReport:
    success: bool
    source_motion_id: str
    source_version: int
    processed_motion_id: str
    processing_version: int
    state: str
    source_animation_reference: str
    animation_reference: str
    processed_clip_path: str
    library_path: str
    duration_seconds: float
    fps: float
    frame_count: int
    curve_count: int
    key_count: int
    bone_count: int
    root_bone_name: str
    quality_status: str
    quality_score: float
    quality_flags: tuple[str, ...] = ()
    hard_failures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    finite_values: bool = False
    quaternion_valid: bool = False
    quaternion_curves_found: bool = False
    key_times_monotonic: bool = False
    missing_sample_count: int = 0
    missing_sample_ratio: float = 0.0
    max_curve_jump: float = 0.0
    root_status: str = "UNKNOWN"
    root_horizontal_distance: float = 0.0
    root_vertical_distance: float = 0.0
    root_authoritative_navigation: bool = False
    apply_root_motion: bool = True
    foot_status: str = "UNKNOWN"
    foot_confidence: float = 0.0
    foot_sliding_distance: float = 0.0
    foot_penetration_depth: float = 0.0
    foot_flags: tuple[str, ...] = ()
    joint_status: str = "UNKNOWN"
    joint_flags: tuple[str, ...] = ()
    source_avatar_valid: bool = False
    source_avatar_human: bool = False
    target_avatar_valid: bool = False
    target_avatar_human: bool = False
    source_humanoid_mappings: tuple[str, ...] = ()
    target_humanoid_mappings: tuple[str, ...] = ()
    mapping_entries: tuple[SkeletonMappingEntry, ...] = ()
    missing_mappings: tuple[str, ...] = ()
    ambiguous_mappings: tuple[str, ...] = ()
    skeleton_status: str = "UNKNOWN"
    retarget_status: str = "FAILED"
    cleanup_status: str = "REJECTED"
    cleanup_magnitude: float = 1.0
    cleanup_key_count: int = 0
    processing_timestamp: Optional[str] = None
    error: Optional[str] = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MotionProcessingReport":
        payload = _camel_case_payload(payload)
        mapping_entries = payload.get("mappingEntries") or []
        return cls(
            success=bool(payload.get("success")),
            source_motion_id=str(payload.get("sourceMotionId") or ""),
            source_version=int(payload.get("sourceVersion") or 0),
            processed_motion_id=str(payload.get("processedMotionId") or ""),
            processing_version=int(payload.get("processingVersion") or 0),
            state=str(payload.get("state") or "FAILED").upper(),
            source_animation_reference=str(payload.get("sourceAnimationReference") or ""),
            animation_reference=str(payload.get("animationReference") or ""),
            processed_clip_path=str(payload.get("processedClipPath") or ""),
            library_path=str(payload.get("libraryPath") or ""),
            duration_seconds=float(payload.get("durationSeconds") or 0.0),
            fps=float(payload.get("fps") or 0.0),
            frame_count=int(payload.get("frameCount") or 0),
            curve_count=int(payload.get("curveCount") or 0),
            key_count=int(payload.get("keyCount") or 0),
            bone_count=int(payload.get("boneCount") or 0),
            root_bone_name=str(payload.get("rootBoneName") or ""),
            quality_status=str(payload.get("qualityStatus") or "REJECTED").upper(),
            quality_score=float(payload.get("qualityScore") or 0.0),
            quality_flags=_text_tuple(payload.get("qualityFlags")),
            hard_failures=_text_tuple(payload.get("hardFailures")),
            warnings=_text_tuple(payload.get("warnings")),
            finite_values=bool(payload.get("finiteValues")),
            quaternion_valid=bool(payload.get("quaternionValid")),
            quaternion_curves_found=bool(payload.get("quaternionCurvesFound")),
            key_times_monotonic=bool(payload.get("keyTimesMonotonic")),
            missing_sample_count=int(payload.get("missingSampleCount") or 0),
            missing_sample_ratio=float(payload.get("missingSampleRatio") or 0.0),
            max_curve_jump=float(payload.get("maxCurveJump") or 0.0),
            root_status=str(payload.get("rootStatus") or "UNKNOWN").upper(),
            root_horizontal_distance=float(payload.get("rootHorizontalDistance") or 0.0),
            root_vertical_distance=float(payload.get("rootVerticalDistance") or 0.0),
            root_authoritative_navigation=bool(payload.get("rootAuthoritativeNavigation")),
            apply_root_motion=bool(payload.get("applyRootMotion", True)),
            foot_status=str(payload.get("footStatus") or "UNKNOWN").upper(),
            foot_confidence=float(payload.get("footConfidence") or 0.0),
            foot_sliding_distance=float(payload.get("footSlidingDistance") or 0.0),
            foot_penetration_depth=float(payload.get("footPenetrationDepth") or 0.0),
            foot_flags=_text_tuple(payload.get("footFlags")),
            joint_status=str(payload.get("jointStatus") or "UNKNOWN").upper(),
            joint_flags=_text_tuple(payload.get("jointFlags")),
            source_avatar_valid=bool(payload.get("sourceAvatarValid")),
            source_avatar_human=bool(payload.get("sourceAvatarHuman")),
            target_avatar_valid=bool(payload.get("targetAvatarValid")),
            target_avatar_human=bool(payload.get("targetAvatarHuman")),
            source_humanoid_mappings=_text_tuple(payload.get("sourceHumanoidMappings")),
            target_humanoid_mappings=_text_tuple(payload.get("targetHumanoidMappings")),
            mapping_entries=tuple(
                SkeletonMappingEntry.from_dict(item)
                for item in mapping_entries
                if isinstance(item, dict)
            ),
            missing_mappings=_text_tuple(payload.get("missingMappings")),
            ambiguous_mappings=_text_tuple(payload.get("ambiguousMappings")),
            skeleton_status=str(payload.get("skeletonStatus") or "UNKNOWN").upper(),
            retarget_status=str(payload.get("retargetStatus") or "FAILED").upper(),
            cleanup_status=str(payload.get("cleanupStatus") or "REJECTED").upper(),
            cleanup_magnitude=float(payload.get("cleanupMagnitude") or 0.0),
            cleanup_key_count=int(payload.get("cleanupKeyCount") or 0),
            processing_timestamp=payload.get("processingTimestamp"),
            error=payload.get("error"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "source_motion_id": self.source_motion_id,
            "source_version": self.source_version,
            "processed_motion_id": self.processed_motion_id,
            "processing_version": self.processing_version,
            "state": self.state,
            "source_animation_reference": self.source_animation_reference,
            "animation_reference": self.animation_reference,
            "processed_clip_path": self.processed_clip_path,
            "library_path": self.library_path,
            "duration_seconds": self.duration_seconds,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "curve_count": self.curve_count,
            "key_count": self.key_count,
            "bone_count": self.bone_count,
            "root_bone_name": self.root_bone_name,
            "quality_status": self.quality_status,
            "quality_score": self.quality_score,
            "quality_flags": list(self.quality_flags),
            "hard_failures": list(self.hard_failures),
            "warnings": list(self.warnings),
            "finite_values": self.finite_values,
            "quaternion_valid": self.quaternion_valid,
            "quaternion_curves_found": self.quaternion_curves_found,
            "key_times_monotonic": self.key_times_monotonic,
            "missing_sample_count": self.missing_sample_count,
            "missing_sample_ratio": self.missing_sample_ratio,
            "max_curve_jump": self.max_curve_jump,
            "root_status": self.root_status,
            "root_horizontal_distance": self.root_horizontal_distance,
            "root_vertical_distance": self.root_vertical_distance,
            "root_authoritative_navigation": self.root_authoritative_navigation,
            "apply_root_motion": self.apply_root_motion,
            "foot_status": self.foot_status,
            "foot_confidence": self.foot_confidence,
            "foot_sliding_distance": self.foot_sliding_distance,
            "foot_penetration_depth": self.foot_penetration_depth,
            "foot_flags": list(self.foot_flags),
            "joint_status": self.joint_status,
            "joint_flags": list(self.joint_flags),
            "source_avatar_valid": self.source_avatar_valid,
            "source_avatar_human": self.source_avatar_human,
            "target_avatar_valid": self.target_avatar_valid,
            "target_avatar_human": self.target_avatar_human,
            "source_humanoid_mappings": list(self.source_humanoid_mappings),
            "target_humanoid_mappings": list(self.target_humanoid_mappings),
            "mapping_entries": [item.to_dict() for item in self.mapping_entries],
            "missing_mappings": list(self.missing_mappings),
            "ambiguous_mappings": list(self.ambiguous_mappings),
            "skeleton_status": self.skeleton_status,
            "retarget_status": self.retarget_status,
            "cleanup_status": self.cleanup_status,
            "cleanup_magnitude": self.cleanup_magnitude,
            "cleanup_key_count": self.cleanup_key_count,
            "processing_timestamp": self.processing_timestamp,
            "error": self.error,
        }


@dataclass(frozen=True)
class MotionPlaybackReport:
    success: bool
    source_motion_id: str
    processed_motion_id: str
    processing_version: int
    animation_reference: str
    processed_clip_path: str
    avatar_valid: bool = False
    avatar_human: bool = False
    clip_persistent: bool = False
    clip_duration_seconds: float = 0.0
    clip_frame_rate: float = 0.0
    curve_binding_count: int = 0
    library_resolved: bool = False
    playback_success: bool = False
    processed_clip_active_at_all_samples: bool = False
    dance_state_active_at_all_samples: bool = False
    finite_samples: bool = False
    nontrivial_movement: bool = False
    max_bone_displacement: float = 0.0
    generated_clip_cleared: bool = False
    original_controller_restored: bool = False
    idle_before: bool = False
    idle_after: bool = False
    navigation_success: bool = False
    navigation_displacement: float = 0.0
    apply_root_motion_disabled: bool = False
    built_in_animation_substituted: bool = True
    required_controller_states_present: bool = False
    controller_states: tuple[str, ...] = ()
    error: Optional[str] = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MotionPlaybackReport":
        payload = _camel_case_payload(payload)
        return cls(
            success=bool(payload.get("success")),
            source_motion_id=str(payload.get("sourceMotionId") or ""),
            processed_motion_id=str(payload.get("processedMotionId") or ""),
            processing_version=int(payload.get("processingVersion") or 0),
            animation_reference=str(payload.get("animationReference") or ""),
            processed_clip_path=str(payload.get("processedClipPath") or ""),
            avatar_valid=bool(payload.get("avatarValid")),
            avatar_human=bool(payload.get("avatarHuman")),
            clip_persistent=bool(payload.get("clipPersistent")),
            clip_duration_seconds=float(payload.get("clipDurationSeconds") or 0.0),
            clip_frame_rate=float(payload.get("clipFrameRate") or 0.0),
            curve_binding_count=int(payload.get("curveBindingCount") or 0),
            library_resolved=bool(payload.get("libraryResolved")),
            playback_success=bool(payload.get("playbackSuccess")),
            processed_clip_active_at_all_samples=bool(payload.get("processedClipActiveAtAllSamples")),
            dance_state_active_at_all_samples=bool(payload.get("danceStateActiveAtAllSamples")),
            finite_samples=bool(payload.get("finiteSamples")),
            nontrivial_movement=bool(payload.get("nontrivialMovement")),
            max_bone_displacement=float(payload.get("maxBoneDisplacement") or 0.0),
            generated_clip_cleared=bool(payload.get("generatedClipCleared")),
            original_controller_restored=bool(payload.get("originalControllerRestored")),
            idle_before=bool(payload.get("idleBefore")),
            idle_after=bool(payload.get("idleAfter")),
            navigation_success=bool(payload.get("navigationSuccess")),
            navigation_displacement=float(payload.get("navigationDisplacement") or 0.0),
            apply_root_motion_disabled=bool(payload.get("applyRootMotionDisabled")),
            built_in_animation_substituted=bool(payload.get("builtInAnimationSubstituted", True)),
            required_controller_states_present=bool(payload.get("requiredControllerStatesPresent")),
            controller_states=_text_tuple(payload.get("controllerStates")),
            error=payload.get("error"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "source_motion_id": self.source_motion_id,
            "processed_motion_id": self.processed_motion_id,
            "processing_version": self.processing_version,
            "animation_reference": self.animation_reference,
            "processed_clip_path": self.processed_clip_path,
            "avatar_valid": self.avatar_valid,
            "avatar_human": self.avatar_human,
            "clip_persistent": self.clip_persistent,
            "clip_duration_seconds": self.clip_duration_seconds,
            "clip_frame_rate": self.clip_frame_rate,
            "curve_binding_count": self.curve_binding_count,
            "library_resolved": self.library_resolved,
            "playback_success": self.playback_success,
            "processed_clip_active_at_all_samples": self.processed_clip_active_at_all_samples,
            "dance_state_active_at_all_samples": self.dance_state_active_at_all_samples,
            "finite_samples": self.finite_samples,
            "nontrivial_movement": self.nontrivial_movement,
            "max_bone_displacement": self.max_bone_displacement,
            "generated_clip_cleared": self.generated_clip_cleared,
            "original_controller_restored": self.original_controller_restored,
            "idle_before": self.idle_before,
            "idle_after": self.idle_after,
            "navigation_success": self.navigation_success,
            "navigation_displacement": self.navigation_displacement,
            "apply_root_motion_disabled": self.apply_root_motion_disabled,
            "built_in_animation_substituted": self.built_in_animation_substituted,
            "required_controller_states_present": self.required_controller_states_present,
            "controller_states": list(self.controller_states),
            "error": self.error,
        }


@dataclass
class MotionProcessingResult:
    request_id: str
    request: MotionProcessingRequest
    state: MotionProcessingState
    quality_status: str = "REJECTED"
    quality_score: float = 0.0
    skeleton_status: str = "UNKNOWN"
    validation_status: str = "INVALID"
    retarget_status: str = "FAILED"
    cleanup_status: str = "REJECTED"
    cleanup_magnitude: float = 1.0
    duration_seconds: float = 0.0
    fps: float = 0.0
    frame_count: int = 0
    animation_reference: str = ""
    processed_clip_path: str = ""
    error: Optional[str] = None
    motion: Optional[dict[str, Any]] = None
    skill: Optional[dict[str, Any]] = None
    processing_report: dict[str, Any] = field(default_factory=dict)
    playback_report: dict[str, Any] = field(default_factory=dict)
    processing_timestamp: Optional[str] = None
    registered_at: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "request": self.request.to_dict(),
            "state": self.state.value,
            "quality_status": self.quality_status,
            "quality_score": self.quality_score,
            "skeleton_status": self.skeleton_status,
            "validation_status": self.validation_status,
            "retarget_status": self.retarget_status,
            "cleanup_status": self.cleanup_status,
            "cleanup_magnitude": self.cleanup_magnitude,
            "duration_seconds": self.duration_seconds,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "animation_reference": self.animation_reference,
            "processed_clip_path": self.processed_clip_path,
            "error": self.error,
            "motion": dict(self.motion) if self.motion else None,
            "skill": dict(self.skill) if self.skill else None,
            "processing_report": dict(self.processing_report),
            "playback_report": dict(self.playback_report),
            "processing_timestamp": self.processing_timestamp,
            "registered_at": self.registered_at,
        }


class MotionProcessingError(RuntimeError):
    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}:{detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


def _text_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    try:
        return tuple(str(item) for item in value)
    except TypeError:
        return ()


def _camel_case_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Accept both Unity camelCase reports and internal snake_case metadata."""
    result: dict[str, Any] = {}
    for key, value in payload.items():
        parts = str(key).split("_")
        normalized = parts[0] + "".join(part.capitalize() for part in parts[1:])
        result[normalized] = value
    return result
