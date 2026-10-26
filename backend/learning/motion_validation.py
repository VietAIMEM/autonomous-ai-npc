"""Deterministic metadata extraction and validation for imported motions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Optional

from .models import SkeletonCompatibility


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_UNITY_ASSET_ROOT = _PROJECT_ROOT / "unity" / "3d_unity" / "Assets"
_SUPPORTED_FORMATS = {"FBX"}
_MAX_SAMPLE_RATE = 240.0

_REQUIRED_HUMANOID_MAPPINGS = frozenset(
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


class MotionValidationError(ValueError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class MotionSourceMetadata:
    source_path: str
    source_format: str
    clip_name: str
    frame_count: int
    sample_rate: float
    duration_seconds: float
    bone_count: int
    root_bone_name: str
    skeleton_status: SkeletonCompatibility

    def to_dict(self) -> dict[str, object]:
        return {
            "source_path": self.source_path,
            "source_format": self.source_format,
            "clip_name": self.clip_name,
            "frame_count": self.frame_count,
            "sample_rate": self.sample_rate,
            "duration_seconds": self.duration_seconds,
            "bone_count": self.bone_count,
            "root_bone_name": self.root_bone_name,
            "skeleton_status": self.skeleton_status.value,
        }


def validate_motion_source(
    source_path: Path | str,
    animation_reference: str,
    duration_seconds: float,
    *,
    source_format: str | None = None,
    clip_name: str | None = None,
    sample_rate: float | None = None,
) -> MotionSourceMetadata:
    """Validate a Unity-imported motion and extract lightweight acquisition metadata."""
    source = _resolve_source_path(source_path)
    format_name = (source_format or source.suffix[1:]).strip().upper()
    if format_name not in _SUPPORTED_FORMATS:
        raise MotionValidationError("UNSUPPORTED_MOTION_FORMAT")

    metadata_path = Path(f"{source}.meta")
    if not source.is_file() or source.stat().st_size == 0:
        raise MotionValidationError("MOTION_SOURCE_MISSING")
    if not metadata_path.is_file() or metadata_path.stat().st_size == 0:
        raise MotionValidationError("MOTION_METADATA_MISSING")
    if not _is_valid_animation_reference(source, animation_reference):
        raise MotionValidationError("INVALID_ANIMATION_REFERENCE")

    try:
        metadata_text = metadata_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise MotionValidationError("MOTION_METADATA_UNREADABLE") from exc

    clip = _select_clip(metadata_text, clip_name)
    frame_span = clip["last_frame"] - clip["first_frame"]
    frame_count = frame_span + 1
    if frame_count <= 0:
        raise MotionValidationError("INVALID_MOTION_FRAMES")
    if duration_seconds <= 0:
        raise MotionValidationError("INVALID_MOTION_DURATION")

    if sample_rate is None:
        sample_rate = frame_span / duration_seconds
    if not 0.0 < sample_rate <= _MAX_SAMPLE_RATE:
        raise MotionValidationError("INVALID_SAMPLE_RATE")
    expected_duration = frame_span / sample_rate
    duration_tolerance = max(0.01, expected_duration * 0.001)
    if abs(duration_seconds - expected_duration) > duration_tolerance:
        raise MotionValidationError("DURATION_SAMPLE_RATE_MISMATCH")

    skeleton = _extract_skeleton(metadata_text)
    if skeleton.root_bone_name is None or not skeleton.humanoid_mappings:
        raise MotionValidationError("MISSING_SKELETON_INFORMATION")
    if skeleton.bone_count <= 0:
        raise MotionValidationError("MISSING_SKELETON_INFORMATION")

    return MotionSourceMetadata(
        source_path=_stable_source_path(source),
        source_format=format_name,
        clip_name=clip["name"],
        frame_count=frame_count,
        sample_rate=float(sample_rate),
        duration_seconds=float(duration_seconds),
        bone_count=skeleton.bone_count,
        root_bone_name=skeleton.root_bone_name,
        skeleton_status=skeleton.compatibility,
    )


def _resolve_source_path(source_path: Path | str) -> Path:
    path = Path(source_path).expanduser()
    if path.is_absolute():
        return path.resolve()

    project_candidate = (_PROJECT_ROOT / path).resolve()
    if project_candidate.is_file():
        return project_candidate

    unity_candidate = (_UNITY_ASSET_ROOT / path).resolve()
    if unity_candidate.is_file():
        return unity_candidate

    if path.parts and path.parts[0].casefold() == "assets":
        return (_UNITY_ASSET_ROOT / Path(*path.parts[1:])).resolve()
    return project_candidate


def _stable_source_path(source: Path) -> str:
    try:
        return source.resolve().relative_to(_PROJECT_ROOT).as_posix()
    except ValueError:
        return source.resolve().as_posix()


def _is_valid_animation_reference(source: Path, animation_reference: str) -> bool:
    reference = str(animation_reference).strip().replace("\\", "/").rstrip("/")
    if not reference.startswith("Assets/") or ".." in reference.split("/"):
        return False
    try:
        relative = "Assets/" + source.resolve().relative_to(_UNITY_ASSET_ROOT).as_posix()
    except ValueError:
        return False
    parent = str(Path(relative).parent).replace("\\", "/")
    return relative == reference or parent == reference or parent.startswith(reference + "/")


def _select_clip(metadata_text: str, requested_clip_name: str | None) -> dict[str, object]:
    clip_block = _indented_block(metadata_text, "clipAnimations:")
    pattern = re.compile(
        r"(?ms)^    - serializedVersion:.*?(?=^    - serializedVersion:|^    isReadable:|\Z)"
    )
    clips: list[dict[str, object]] = []
    for entry in pattern.findall(clip_block):
        name = _scalar_value(entry, "name")
        first_frame = _integer_value(entry, "firstFrame")
        last_frame = _integer_value(entry, "lastFrame")
        if name is None or first_frame is None or last_frame is None:
            raise MotionValidationError("MALFORMED_MOTION_CLIP")
        clips.append({"name": name, "first_frame": first_frame, "last_frame": last_frame})

    if not clips:
        raise MotionValidationError("MOTION_CLIP_MISSING")
    if requested_clip_name is not None:
        matches = [clip for clip in clips if clip["name"] == requested_clip_name.strip()]
        if not matches:
            raise MotionValidationError("MOTION_CLIP_MISSING")
        if len(matches) > 1:
            raise MotionValidationError("AMBIGUOUS_MOTION_CLIP")
        return matches[0]
    if len(clips) > 1:
        raise MotionValidationError("AMBIGUOUS_MOTION_CLIP")
    return clips[0]


@dataclass(frozen=True)
class _SkeletonMetadata:
    bone_count: int
    humanoid_mappings: frozenset[str]
    root_bone_name: Optional[str]
    compatibility: SkeletonCompatibility


def _extract_skeleton(metadata_text: str) -> _SkeletonMetadata:
    animation_type = _scalar_value(metadata_text, "animationType")
    if animation_type is None:
        raise MotionValidationError("MISSING_SKELETON_INFORMATION")

    human_block = _indented_block(metadata_text, "human:")
    mappings: dict[str, str] = {}
    current_bone: Optional[str] = None
    for line in human_block.splitlines():
        bone_match = re.match(r"^    - boneName: (.+)$", line)
        human_match = re.match(r"^      humanName: (.+)$", line)
        if bone_match:
            current_bone = bone_match.group(1).strip()
        if human_match and current_bone is not None:
            mappings[human_match.group(1).strip()] = current_bone
            current_bone = None

    skeleton_block = _indented_block(metadata_text, "skeleton:")
    bone_count = len(re.findall(r"^    - name: .+$", skeleton_block, flags=re.MULTILINE))
    human_names = frozenset(mappings)
    root_bone = mappings.get("Hips")

    if animation_type != "3":
        compatibility = SkeletonCompatibility.INCOMPATIBLE
    elif _REQUIRED_HUMANOID_MAPPINGS.issubset(human_names):
        compatibility = SkeletonCompatibility.COMPATIBLE
    elif "Hips" in human_names and len(human_names) >= 8:
        compatibility = SkeletonCompatibility.RETARGETABLE
    else:
        compatibility = SkeletonCompatibility.UNKNOWN

    return _SkeletonMetadata(bone_count, human_names, root_bone, compatibility)


def _indented_block(text: str, key: str) -> str:
    match = re.search(
        rf"(?ms)^(?P<indent>[ ]*){re.escape(key)}\n(?P<content>.*?)(?=^(?P=indent)[^ \-]|\Z)",
        text,
    )
    return match.group("content") if match else ""


def _scalar_value(text: str, key: str) -> Optional[str]:
    match = re.search(rf"(?m)^\s*{re.escape(key)}: (.+)$", text)
    return match.group(1).strip() if match else None


def _integer_value(text: str, key: str) -> Optional[int]:
    value = _scalar_value(text, key)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


__all__ = [
    "MotionSourceMetadata",
    "MotionValidationError",
    "validate_motion_source",
]
