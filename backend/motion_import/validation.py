"""Validation for Unity import reports and original source files."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

from ..learning.models import SkeletonCompatibility
from .models import MotionImportError, MotionImportRequest, UnityMotionImportReport


REQUIRED_HUMANOID_MAPPINGS = frozenset(
    {
        "Hips",
        "LeftUpperLeg",
        "RightUpperLeg",
        "LeftLowerLeg",
        "RightLowerLeg",
        "LeftFoot",
        "RightFoot",
        "Spine",
        "Chest",
        "Neck",
        "Head",
        "LeftShoulder",
        "RightShoulder",
        "LeftUpperArm",
        "RightUpperArm",
        "LeftLowerArm",
        "RightLowerArm",
        "LeftHand",
        "RightHand",
    }
)

_STRICTNESS = {
    SkeletonCompatibility.INCOMPATIBLE: 0,
    SkeletonCompatibility.UNKNOWN: 1,
    SkeletonCompatibility.RETARGETABLE: 2,
    SkeletonCompatibility.COMPATIBLE: 3,
}


def hash_source_file(path: Path | str) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_source_file(path: Path | str, supported_formats: frozenset[str]) -> tuple[Path, str, str]:
    source = Path(path).expanduser()
    if not source.is_file():
        raise MotionImportError("MOTION_SOURCE_MISSING")
    if source.stat().st_size == 0:
        raise MotionImportError("MOTION_SOURCE_EMPTY")
    source_format = source.suffix.removeprefix(".").upper()
    if source_format not in supported_formats:
        raise MotionImportError("UNSUPPORTED_MOTION_FORMAT")
    try:
        source_hash = hash_source_file(source)
    except OSError as exc:
        raise MotionImportError("MOTION_SOURCE_UNREADABLE", type(exc).__name__) from exc
    return source.resolve(), source_format, source_hash


def classify_skeleton(
    *,
    source_avatar_valid: bool,
    source_avatar_human: bool,
    source_mappings: tuple[str, ...],
    target_mappings: tuple[str, ...],
) -> SkeletonCompatibility:
    source_names = frozenset(source_mappings)
    target_names = frozenset(target_mappings)
    if not source_avatar_valid or not source_avatar_human:
        return SkeletonCompatibility.INCOMPATIBLE
    if not REQUIRED_HUMANOID_MAPPINGS.issubset(target_names):
        return SkeletonCompatibility.INCOMPATIBLE
    if REQUIRED_HUMANOID_MAPPINGS.issubset(source_names):
        return SkeletonCompatibility.COMPATIBLE
    if "Hips" in source_names and len(source_names) >= 8:
        return SkeletonCompatibility.RETARGETABLE
    return SkeletonCompatibility.UNKNOWN


def stricter_skeleton(
    first: SkeletonCompatibility, second: SkeletonCompatibility
) -> SkeletonCompatibility:
    return min(first, second, key=lambda status: _STRICTNESS[status])


def validate_import_report(
    report: UnityMotionImportReport,
    request: MotionImportRequest,
    *,
    unity_project_path: Path,
) -> SkeletonCompatibility:
    if not report.success:
        raise MotionImportError("UNITY_IMPORT_FAILED", report.error or "unknown_error")
    if report.error:
        raise MotionImportError("UNITY_IMPORT_FAILED", report.error)
    if report.motion_id != request.actual_motion_id or report.version != request.version:
        raise MotionImportError("UNITY_IMPORT_IDENTITY_MISMATCH")
    if report.source_sha256 != request.source_sha256:
        raise MotionImportError("SOURCE_HASH_MISMATCH")
    if Path(report.original_source_file).resolve() != Path(request.source_file).resolve():
        raise MotionImportError("ORIGINAL_SOURCE_MISMATCH")
    if report.source_format != request.source_format:
        raise MotionImportError("MOTION_FORMAT_MISMATCH")
    if report.clip_name != (request.clip_name or report.clip_name):
        raise MotionImportError("MOTION_CLIP_MISMATCH")

    _require_positive_float(report.duration_seconds, "INVALID_MOTION_DURATION")
    _require_positive_float(report.fps, "INVALID_SAMPLE_RATE")
    _require_positive_int(report.frame_count, "INVALID_MOTION_FRAMES")
    _require_positive_int(report.curve_count, "MOTION_CURVES_MISSING")
    _require_positive_int(report.key_count, "MOTION_KEYS_MISSING")
    _require_positive_int(report.bone_count, "MISSING_SKELETON_INFORMATION")
    if not report.root_bone_name.strip():
        raise MotionImportError("MISSING_SKELETON_INFORMATION")
    if not report.finite_values:
        raise MotionImportError("NON_FINITE_MOTION_VALUES")
    if not report.target_avatar_valid or not report.target_avatar_human:
        raise MotionImportError("INVALID_PRODUCTION_RIG")
    if not REQUIRED_HUMANOID_MAPPINGS.issubset(frozenset(report.target_humanoid_mappings)):
        raise MotionImportError("INVALID_PRODUCTION_RIG")

    try:
        reported_status = SkeletonCompatibility(report.skeleton_status)
    except ValueError as exc:
        raise MotionImportError("INVALID_SKELETON_STATUS", report.skeleton_status) from exc
    calculated_status = classify_skeleton(
        source_avatar_valid=report.source_avatar_valid,
        source_avatar_human=report.source_avatar_human,
        source_mappings=report.source_humanoid_mappings,
        target_mappings=report.target_humanoid_mappings,
    )
    status = stricter_skeleton(reported_status, calculated_status)

    staged_source = _resolve_project_asset(report.staged_source_file, unity_project_path)
    if not staged_source.is_file() or staged_source.stat().st_size == 0:
        raise MotionImportError("STAGED_SOURCE_MISSING")
    if not _is_staged_motion_path(staged_source, unity_project_path):
        raise MotionImportError("INVALID_STAGED_SOURCE_REFERENCE")

    if status in {SkeletonCompatibility.COMPATIBLE, SkeletonCompatibility.RETARGETABLE}:
        if not report.source_avatar_valid or not report.source_avatar_human:
            raise MotionImportError("INVALID_SOURCE_AVATAR")
        _require_stable_clip(report.stable_clip_path, unity_project_path)
        _require_animation_reference(report.animation_reference, staged_source, unity_project_path)

    return status


def _require_positive_float(value: object, reason: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MotionImportError(reason, "not_a_number")
    if not math.isfinite(float(value)) or float(value) <= 0:
        raise MotionImportError(reason)


def _require_positive_int(value: object, reason: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise MotionImportError(reason)


def _resolve_project_asset(value: str, unity_project_path: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (unity_project_path / path).resolve()


def _is_staged_motion_path(path: Path, unity_project_path: Path) -> bool:
    try:
        relative = path.relative_to((unity_project_path / "Assets").resolve())
    except ValueError:
        return False
    return relative.as_posix().startswith("MotionImports/Phase16/")


def _require_stable_clip(value: str, unity_project_path: Path) -> None:
    if not value.strip():
        raise MotionImportError("STABLE_CLIP_MISSING")
    path = _resolve_project_asset(value, unity_project_path)
    try:
        relative = path.relative_to(
            (unity_project_path / "Assets" / "Resources" / "GeneratedMotions" / "Phase16").resolve()
        )
    except ValueError as exc:
        raise MotionImportError("INVALID_STABLE_CLIP_REFERENCE") from exc
    if not path.is_file() or path.stat().st_size == 0:
        raise MotionImportError("STABLE_CLIP_MISSING")
    if not relative.as_posix().endswith(".anim"):
        raise MotionImportError("INVALID_STABLE_CLIP_REFERENCE")


def _require_animation_reference(
    value: str, staged_source: Path, unity_project_path: Path
) -> None:
    reference = value.strip().replace("\\", "/").rstrip("/")
    if not reference.startswith("Assets/") or ".." in reference.split("/"):
        raise MotionImportError("INVALID_ANIMATION_REFERENCE")
    try:
        relative_source = staged_source.relative_to((unity_project_path / "Assets").resolve())
    except ValueError as exc:
        raise MotionImportError("INVALID_STAGED_SOURCE_REFERENCE") from exc
    if reference != "Assets/" + relative_source.as_posix():
        raise MotionImportError("INVALID_ANIMATION_REFERENCE")


__all__ = [
    "REQUIRED_HUMANOID_MAPPINGS",
    "classify_skeleton",
    "hash_source_file",
    "stricter_skeleton",
    "validate_import_report",
    "validate_source_file",
]
