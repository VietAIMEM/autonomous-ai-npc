"""Typed models for the motion-file import boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_UNITY_EXECUTABLE = Path(
    "C:/Program Files/Unity/Hub/Editor/6000.6.0f1/Editor/Unity.exe"
)
DEFAULT_UNITY_PROJECT = PROJECT_ROOT / "unity" / "3d_unity"
DEFAULT_MOTION_IMPORT_DB_PATH = PROJECT_ROOT / "backend" / "data" / "motion_import.db"


class MotionImportStatus(str, Enum):
    QUEUED = "QUEUED"
    IMPORTING = "IMPORTING"
    VALIDATED = "VALIDATED"
    REGISTERED = "REGISTERED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class MotionImportConfig:
    supported_formats: frozenset[str] = frozenset({"FBX"})
    unity_executable: Path = DEFAULT_UNITY_EXECUTABLE
    unity_project_path: Path = DEFAULT_UNITY_PROJECT
    database_path: Path = DEFAULT_MOTION_IMPORT_DB_PATH
    timeout_seconds: float = 300.0
    max_queue_depth: int = 2
    max_concurrent_imports: int = 1
    max_retained_jobs: int = 64
    worker_poll_interval_seconds: float = 0.05

    def __post_init__(self) -> None:
        if not self.supported_formats:
            raise ValueError("supported_formats cannot be empty")
        if any(format_name != format_name.upper() for format_name in self.supported_formats):
            raise ValueError("supported_formats must be uppercase")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_queue_depth <= 0:
            raise ValueError("max_queue_depth must be positive")
        if self.max_retained_jobs <= 0:
            raise ValueError("max_retained_jobs must be positive")
        if self.max_concurrent_imports != 1:
            raise ValueError("max_concurrent_imports must be 1")
        if self.worker_poll_interval_seconds <= 0:
            raise ValueError("worker_poll_interval_seconds must be positive")


@dataclass(frozen=True)
class MotionImportRequest:
    source_file: Path
    source_sha256: str
    source_format: str
    motion_id: str
    actual_motion_id: str
    version: int
    name: str
    description: str
    clip_name: Optional[str] = None
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_file": str(self.source_file),
            "source_sha256": self.source_sha256,
            "source_format": self.source_format,
            "motion_id": self.motion_id,
            "actual_motion_id": self.actual_motion_id,
            "version": self.version,
            "name": self.name,
            "description": self.description,
            "clip_name": self.clip_name,
            "tags": list(self.tags),
        }


@dataclass(frozen=True)
class UnityMotionImportReport:
    success: bool
    motion_id: str
    version: int
    original_source_file: str
    staged_source_file: str
    source_sha256: str
    source_format: str
    clip_name: str
    duration_seconds: float
    fps: float
    frame_count: int
    curve_count: int
    key_count: int
    finite_values: bool
    bone_count: int
    root_bone_name: str
    source_avatar_valid: bool
    source_avatar_human: bool
    source_humanoid_mappings: tuple[str, ...] = ()
    target_avatar_name: str = ""
    target_avatar_valid: bool = False
    target_avatar_human: bool = False
    target_humanoid_mappings: tuple[str, ...] = ()
    skeleton_status: str = "UNKNOWN"
    import_status: str = "VALIDATED"
    animation_reference: str = ""
    stable_clip_path: str = ""
    library_path: str = ""
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    imported_at: Optional[str] = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "UnityMotionImportReport":
        mappings = payload.get("sourceHumanoidMappings")
        target_mappings = payload.get("targetHumanoidMappings")
        metadata = payload.get("metadata")
        return cls(
            success=bool(payload.get("success")),
            motion_id=str(payload.get("motionId") or ""),
            version=int(payload.get("version") or 1),
            original_source_file=str(payload.get("originalSourceFile") or ""),
            staged_source_file=str(payload.get("stagedSourceFile") or ""),
            source_sha256=str(payload.get("sourceSha256") or ""),
            source_format=str(payload.get("sourceFormat") or "").upper(),
            clip_name=str(payload.get("clipName") or ""),
            duration_seconds=float(payload.get("durationSeconds") or 0.0),
            fps=float(payload.get("fps") or 0.0),
            frame_count=int(payload.get("frameCount") or 0),
            curve_count=int(payload.get("curveCount") or 0),
            key_count=int(payload.get("keyCount") or 0),
            finite_values=bool(payload.get("finiteValues")),
            bone_count=int(payload.get("boneCount") or 0),
            root_bone_name=str(payload.get("rootBoneName") or ""),
            source_avatar_valid=bool(payload.get("sourceAvatarValid")),
            source_avatar_human=bool(payload.get("sourceAvatarHuman")),
            source_humanoid_mappings=tuple(str(item) for item in (mappings or ())),
            target_avatar_name=str(payload.get("targetAvatarName") or ""),
            target_avatar_valid=bool(payload.get("targetAvatarValid")),
            target_avatar_human=bool(payload.get("targetAvatarHuman")),
            target_humanoid_mappings=tuple(str(item) for item in (target_mappings or ())),
            skeleton_status=str(payload.get("skeletonStatus") or "UNKNOWN").upper(),
            import_status=str(payload.get("importStatus") or "VALIDATED").upper(),
            animation_reference=str(payload.get("animationReference") or ""),
            stable_clip_path=str(payload.get("stableClipPath") or ""),
            library_path=str(payload.get("libraryPath") or ""),
            error=_optional_text(payload.get("error")),
            metadata=dict(metadata) if isinstance(metadata, dict) else {},
            imported_at=_optional_text(payload.get("importedAt")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "motion_id": self.motion_id,
            "version": self.version,
            "original_source_file": self.original_source_file,
            "staged_source_file": self.staged_source_file,
            "source_sha256": self.source_sha256,
            "source_format": self.source_format,
            "clip_name": self.clip_name,
            "duration_seconds": self.duration_seconds,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "curve_count": self.curve_count,
            "key_count": self.key_count,
            "finite_values": self.finite_values,
            "bone_count": self.bone_count,
            "root_bone_name": self.root_bone_name,
            "source_avatar_valid": self.source_avatar_valid,
            "source_avatar_human": self.source_avatar_human,
            "source_humanoid_mappings": list(self.source_humanoid_mappings),
            "target_avatar_name": self.target_avatar_name,
            "target_avatar_valid": self.target_avatar_valid,
            "target_avatar_human": self.target_avatar_human,
            "target_humanoid_mappings": list(self.target_humanoid_mappings),
            "skeleton_status": self.skeleton_status,
            "import_status": self.import_status,
            "animation_reference": self.animation_reference,
            "stable_clip_path": self.stable_clip_path,
            "library_path": self.library_path,
            "error": self.error,
            "metadata": dict(self.metadata),
            "imported_at": self.imported_at,
        }


@dataclass(frozen=True)
class MotionImportResult:
    request_id: str
    requested_base_motion_id: str
    actual_motion_id: str
    version: int
    original_source_file: str
    staged_source_file: str
    source_sha256: str
    source_format: str
    import_status: MotionImportStatus
    validation_status: str
    skeleton_status: str
    duration_seconds: float
    fps: float
    frame_count: int
    bone_count: int
    root_bone_name: str
    animation_reference: str
    stable_clip_path: str
    error: Optional[str] = None
    motion: Optional[dict[str, Any]] = None
    skill: Optional[dict[str, Any]] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    imported_at: Optional[str] = None
    registered_at: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "requested_base_motion_id": self.requested_base_motion_id,
            "actual_motion_id": self.actual_motion_id,
            "version": self.version,
            "original_source_file": self.original_source_file,
            "staged_source_file": self.staged_source_file,
            "source_sha256": self.source_sha256,
            "source_format": self.source_format,
            "import_status": self.import_status.value,
            "validation_status": self.validation_status,
            "skeleton_status": self.skeleton_status,
            "duration_seconds": self.duration_seconds,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "bone_count": self.bone_count,
            "root_bone_name": self.root_bone_name,
            "animation_reference": self.animation_reference,
            "stable_clip_path": self.stable_clip_path,
            "error": self.error,
            "motion": dict(self.motion) if self.motion else None,
            "skill": dict(self.skill) if self.skill else None,
            "metadata": dict(self.metadata),
            "imported_at": self.imported_at,
            "registered_at": self.registered_at,
        }


class MotionImportError(RuntimeError):
    """Raised for structured request, report, adapter, or import failures."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}:{detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


def _optional_text(value: object) -> Optional[str]:
    if value is None:
        return None
    return str(value)
