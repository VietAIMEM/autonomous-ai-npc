"""Fail-closed validation for Unity processing and production playback reports."""

from __future__ import annotations

import math
from pathlib import Path

from ..learning.models import MotionDefinition, SkeletonCompatibility
from .models import MotionPlaybackReport, MotionProcessingError, MotionProcessingReport, MotionProcessingRequest


REQUIRED_HUMANOID_MAPPINGS = frozenset(
    {
        "Hips", "Spine", "Chest", "Neck", "Head",
        "LeftShoulder", "RightShoulder",
        "LeftUpperArm", "RightUpperArm",
        "LeftLowerArm", "RightLowerArm",
        "LeftHand", "RightHand",
        "LeftUpperLeg", "RightUpperLeg",
        "LeftLowerLeg", "RightLowerLeg",
        "LeftFoot", "RightFoot",
    }
)
_PROCESSING_ROOT = "Assets/Resources/GeneratedMotions/Phase17"


def validate_processing_report(
    report: MotionProcessingReport,
    request: MotionProcessingRequest,
    source_motion: MotionDefinition,
    *,
    unity_project_path: Path,
) -> MotionProcessingReport:
    _require_identity(report, request, source_motion)
    if not report.success or report.error:
        raise MotionProcessingError("MOTION_PROCESSING_FAILED", report.error or "unknown_error")
    if report.state != "READY":
        raise MotionProcessingError("MOTION_NOT_READY", report.state)
    if report.hard_failures:
        raise MotionProcessingError("MOTION_HARD_VALIDATION_FAILED", ",".join(report.hard_failures))
    if report.quality_status == "REJECTED":
        raise MotionProcessingError("MOTION_QUALITY_REJECTED")
    if report.quality_status not in {"VALID", "FLAGGED"}:
        raise MotionProcessingError("INVALID_QUALITY_STATUS", report.quality_status)
    if not math.isfinite(report.quality_score) or not 0.0 <= report.quality_score <= 1.0:
        raise MotionProcessingError("INVALID_QUALITY_SCORE")
    if not report.finite_values:
        raise MotionProcessingError("NON_FINITE_MOTION_VALUES")
    if report.quaternion_curves_found and not report.quaternion_valid:
        raise MotionProcessingError("INVALID_QUATERNION_DATA")
    if not report.key_times_monotonic:
        raise MotionProcessingError("NON_MONOTONIC_KEY_TIMES")
    if report.missing_sample_ratio > 0.20:
        raise MotionProcessingError("EXCESSIVE_MISSING_SAMPLES")

    _require_positive(report.duration_seconds, "INVALID_MOTION_DURATION")
    _require_positive(report.fps, "INVALID_SAMPLE_RATE")
    _require_positive_int(report.frame_count, "INVALID_MOTION_FRAMES")
    _require_positive_int(report.curve_count, "MOTION_CURVES_MISSING")
    _require_positive_int(report.key_count, "MOTION_KEYS_MISSING")
    _require_positive_int(report.bone_count, "MISSING_SKELETON_INFORMATION")
    if not report.root_bone_name.strip():
        raise MotionProcessingError("MISSING_SKELETON_INFORMATION")

    if not report.source_avatar_valid or not report.source_avatar_human:
        raise MotionProcessingError("INVALID_SOURCE_AVATAR")
    if not report.target_avatar_valid or not report.target_avatar_human:
        raise MotionProcessingError("INVALID_PRODUCTION_RIG")
    if not REQUIRED_HUMANOID_MAPPINGS.issubset(report.source_humanoid_mappings):
        raise MotionProcessingError("SOURCE_MAPPING_INCOMPLETE")
    if not REQUIRED_HUMANOID_MAPPINGS.issubset(report.target_humanoid_mappings):
        raise MotionProcessingError("TARGET_MAPPING_INCOMPLETE")
    if report.missing_mappings or report.ambiguous_mappings:
        raise MotionProcessingError("AMBIGUOUS_OR_MISSING_SKELETON_MAPPING")
    try:
        final_skeleton = SkeletonCompatibility(report.skeleton_status)
    except ValueError as exc:
        raise MotionProcessingError("INVALID_SKELETON_STATUS", report.skeleton_status) from exc
    if final_skeleton != SkeletonCompatibility.COMPATIBLE:
        raise MotionProcessingError("MOTION_NOT_EXECUTABLE_WITHOUT_RETARGETING")
    if report.retarget_status != "MAPPED":
        raise MotionProcessingError("RETARGETING_FAILED", report.retarget_status)

    if report.cleanup_status not in {"NOT_REQUIRED", "APPLIED"}:
        raise MotionProcessingError("CLEANUP_REJECTED", report.cleanup_status)
    if not math.isfinite(report.cleanup_magnitude) or not 0.0 <= report.cleanup_magnitude <= 0.05:
        raise MotionProcessingError("EXCESSIVE_CLEANUP_MAGNITUDE")
    if report.cleanup_status == "APPLIED" and (report.cleanup_magnitude <= 0.0 or report.cleanup_key_count <= 0):
        raise MotionProcessingError("CLEANUP_METADATA_INVALID")
    if report.cleanup_status == "NOT_REQUIRED" and (report.cleanup_magnitude != 0.0 or report.cleanup_key_count != 0):
        raise MotionProcessingError("UNJUSTIFIED_CLEANUP_METADATA")

    if report.foot_status not in {"VALID", "FLAGGED", "REJECTED"}:
        raise MotionProcessingError("FOOT_VALIDATION_UNAVAILABLE")
    if not math.isfinite(report.foot_confidence) or not 0.0 <= report.foot_confidence <= 1.0:
        raise MotionProcessingError("INVALID_FOOT_CONFIDENCE")
    if not math.isfinite(report.foot_penetration_depth) or report.foot_penetration_depth < 0.0:
        raise MotionProcessingError("INVALID_FOOT_PENETRATION")
    if not math.isfinite(report.foot_sliding_distance) or report.foot_sliding_distance < 0.0:
        raise MotionProcessingError("INVALID_FOOT_SLIDING")
    if report.foot_penetration_depth > 0.05 and report.foot_status != "REJECTED":
        raise MotionProcessingError("FOOT_STATUS_DISAGREES_WITH_METRICS")
    if report.foot_penetration_depth > 0.02 and report.foot_status == "VALID":
        raise MotionProcessingError("FOOT_STATUS_DISAGREES_WITH_METRICS")
    if report.foot_status == "REJECTED":
        raise MotionProcessingError("FOOT_VALIDATION_REJECTED")
    if report.joint_status not in {"VALID", "FLAGGED", "REJECTED"}:
        raise MotionProcessingError("JOINT_VALIDATION_UNAVAILABLE")
    if report.joint_status == "REJECTED":
        raise MotionProcessingError("JOINT_VALIDATION_REJECTED")

    if not report.root_authoritative_navigation or report.apply_root_motion:
        raise MotionProcessingError("ROOT_MOTION_AUTHORITY_INVALID")
    if report.root_status not in {"STABLE", "NAVIGATION_EXEMPT"}:
        raise MotionProcessingError("ROOT_VALIDATION_UNAVAILABLE")
    if not math.isfinite(report.root_horizontal_distance) or report.root_horizontal_distance < 0.0:
        raise MotionProcessingError("INVALID_ROOT_TRAJECTORY")
    if not math.isfinite(report.root_vertical_distance) or report.root_vertical_distance < 0.0:
        raise MotionProcessingError("INVALID_ROOT_TRAJECTORY")
        raise MotionProcessingError("ROOT_TRAJECTORY_REJECTED")

    _require_processed_clip(report, unity_project_path)
    if report.animation_reference != report.processed_clip_path:
        raise MotionProcessingError("ANIMATION_REFERENCE_MISMATCH")
    if "__preview__scene" in report.animation_reference.casefold():
        raise MotionProcessingError("PREVIEW_ANIMATION_FORBIDDEN")
    if report.library_path != "Assets/Resources/GeneratedMotions/GeneratedMotionLibrary.asset":
        raise MotionProcessingError("INVALID_GENERATED_MOTION_LIBRARY")
    return report


def validate_playback_report(
    report: MotionPlaybackReport,
    request: MotionProcessingRequest,
    processing: MotionProcessingReport,
) -> MotionPlaybackReport:
    if report.source_motion_id != request.source_motion_id:
        raise MotionProcessingError("PLAYBACK_SOURCE_IDENTITY_MISMATCH")
    if report.processed_motion_id != request.processed_motion_id:
        raise MotionProcessingError("PLAYBACK_IDENTITY_MISMATCH")
    if report.processing_version != request.processing_version:
        raise MotionProcessingError("PLAYBACK_VERSION_MISMATCH")
    if report.animation_reference != processing.animation_reference:
        raise MotionProcessingError("PLAYBACK_REFERENCE_MISMATCH")
    if report.processed_clip_path != processing.processed_clip_path:
        raise MotionProcessingError("PLAYBACK_CLIP_MISMATCH")
    if not report.success or report.error:
        raise MotionProcessingError("PRODUCTION_PLAYBACK_FAILED", report.error or "unknown_error")
    if not report.avatar_valid or not report.avatar_human:
        raise MotionProcessingError("INVALID_PRODUCTION_RIG")
    if not report.clip_persistent or not report.library_resolved:
        raise MotionProcessingError("PROCESSED_CLIP_RESOLUTION_FAILED")
    _require_positive(report.clip_duration_seconds, "INVALID_MOTION_DURATION")
    _require_positive(report.clip_frame_rate, "INVALID_SAMPLE_RATE")
    _require_positive_int(report.curve_binding_count, "MOTION_CURVES_MISSING")
    if not report.playback_success:
        raise MotionProcessingError("PROCESSED_CLIP_PLAYBACK_FAILED")
    if not report.processed_clip_active_at_all_samples or not report.dance_state_active_at_all_samples:
        raise MotionProcessingError("PROCESSED_CLIP_NOT_ACTIVE")
    if not report.finite_samples:
        raise MotionProcessingError("NON_FINITE_PLAYBACK_SAMPLES")
    if not report.nontrivial_movement or report.max_bone_displacement <= 0.01:
        raise MotionProcessingError("NO_VISIBLE_MOVEMENT")
    if report.built_in_animation_substituted:
        raise MotionProcessingError("BUILT_IN_ANIMATION_SUBSTITUTED")
    if not report.generated_clip_cleared or not report.original_controller_restored:
        raise MotionProcessingError("ANIMATOR_CONTROLLER_NOT_RESTORED")
    if not report.idle_before or not report.idle_after:
        raise MotionProcessingError("IDLE_TRANSITION_FAILED")
    if not report.navigation_success or report.navigation_displacement <= 0.05:
        raise MotionProcessingError("NAVIGATION_PROBE_FAILED")
    if not report.apply_root_motion_disabled:
        raise MotionProcessingError("ROOT_MOTION_AUTHORITY_INVALID")
    if not report.required_controller_states_present:
        raise MotionProcessingError("EXISTING_CONTROLLER_STATES_MISSING")
    return report


def _require_identity(
    report: MotionProcessingReport,
    request: MotionProcessingRequest,
    source_motion: MotionDefinition,
) -> None:
    if report.source_motion_id != request.source_motion_id:
        raise MotionProcessingError("SOURCE_IDENTITY_MISMATCH")
    if report.source_version != request.source_version:
        raise MotionProcessingError("SOURCE_VERSION_MISMATCH")
    if report.processed_motion_id != request.processed_motion_id:
        raise MotionProcessingError("PROCESSED_IDENTITY_MISMATCH")
    if report.processing_version != request.processing_version:
        raise MotionProcessingError("PROCESSING_VERSION_MISMATCH")
    if report.source_animation_reference != source_motion.animation_reference:
        raise MotionProcessingError("SOURCE_REFERENCE_MISMATCH")


def _require_processed_clip(report: MotionProcessingReport, unity_project_path: Path) -> None:
    path = _resolve_asset(report.processed_clip_path, unity_project_path)
    try:
        relative = path.relative_to((unity_project_path / _PROCESSING_ROOT).resolve())
    except ValueError as exc:
        raise MotionProcessingError("INVALID_PROCESSED_CLIP_REFERENCE") from exc
    if not relative.as_posix().endswith(".anim"):
        raise MotionProcessingError("INVALID_PROCESSED_CLIP_REFERENCE")
    if not path.is_file() or path.stat().st_size == 0:
        raise MotionProcessingError("PROCESSED_CLIP_MISSING")


def _resolve_asset(value: str, unity_project_path: Path) -> Path:
    reference = str(value).strip().replace("\\", "/").rstrip("/")
    if not reference.startswith("Assets/") or ".." in reference.split("/"):
        raise MotionProcessingError("INVALID_ANIMATION_REFERENCE")
    return (unity_project_path / reference).resolve()


def _require_positive(value: float, reason: str) -> None:
    if not math.isfinite(float(value)) or float(value) <= 0:
        raise MotionProcessingError(reason)


def _require_positive_int(value: int, reason: str) -> None:
    if value <= 0:
        raise MotionProcessingError(reason)


__all__ = [
    "REQUIRED_HUMANOID_MAPPINGS",
    "validate_playback_report",
    "validate_processing_report",
]
