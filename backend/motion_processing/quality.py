"""Deterministic quality helpers shared by report validation and tests."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping, Sequence


class QualityStatus(str, Enum):
    VALID = "VALID"
    FLAGGED = "FLAGGED"
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class MotionFrame:
    timestamp: float
    positions: Mapping[str, tuple[float, float, float]]
    rotations: Mapping[str, tuple[float, float, float, float]]


@dataclass(frozen=True)
class QualityAnalysis:
    status: QualityStatus
    score: float
    flags: tuple[str, ...]
    hard_failures: tuple[str, ...]
    warnings: tuple[str, ...]
    missing_sample_ratio: float
    max_acceleration: float


def build_skeleton_mapping(
    source: Mapping[str, str] | Sequence[tuple[str, str]],
    target: Mapping[str, str] | Sequence[tuple[str, str]],
) -> tuple[dict[str, tuple[str, str]], tuple[str, ...], tuple[str, ...]]:
    """Return explicit mappings, missing semantic joints, and ambiguous joints."""
    semantic = (
        "Hips", "Spine", "Chest", "Neck", "Head",
        "LeftShoulder", "RightShoulder",
        "LeftUpperArm", "RightUpperArm",
        "LeftLowerArm", "RightLowerArm",
        "LeftHand", "RightHand",
        "LeftUpperLeg", "RightUpperLeg",
        "LeftLowerLeg", "RightLowerLeg",
        "LeftFoot", "RightFoot",
    )
    source_groups = _group_mappings(source)
    target_groups = _group_mappings(target)

    mappings: dict[str, tuple[str, str]] = {}
    missing: list[str] = []
    ambiguous: list[str] = []
    for human in semantic:
        source_bones = source_groups.get(human, [])
        target_bones = target_groups.get(human, [])
        if not source_bones or not target_bones:
            missing.append(human)
        elif len(source_bones) > 1 or len(target_bones) > 1:
            ambiguous.append(human)
        else:
            mappings[human] = (source_bones[0], target_bones[0])
    return mappings, tuple(missing), tuple(ambiguous)


def classify_skeleton_mapping(
    source: Mapping[str, str] | Sequence[tuple[str, str]],
    target: Mapping[str, str] | Sequence[tuple[str, str]],
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    _mappings, missing, ambiguous = build_skeleton_mapping(source, target)
    if ambiguous:
        return "AMBIGUOUS", missing, ambiguous
    if missing:
        return "INCOMPLETE", missing, ambiguous
    return "COMPLETE", missing, ambiguous


def _group_mappings(
    mappings: Mapping[str, str] | Sequence[tuple[str, str]]
) -> dict[str, list[str]]:
    if isinstance(mappings, Mapping):
        return {
            str(human): [str(bone)]
            for human, bone in mappings.items()
        }
    grouped: dict[str, list[str]] = {}
    for human, bone in mappings:
        grouped.setdefault(str(human), []).append(str(bone))
    return grouped


def compute_quality_score(
    *,
    hard_failures: int = 0,
    warnings: int = 0,
    missing_sample_ratio: float = 0.0,
    cleanup_magnitude: float = 0.0,
    root_stable: bool = True,
    foot_stable: bool = True,
    joint_valid: bool = True,
) -> float:
    if hard_failures:
        return 0.0
    penalty = min(0.25, warnings * 0.04)
    penalty += min(0.25, max(0.0, missing_sample_ratio) * 2.0)
    penalty += min(0.15, max(0.0, cleanup_magnitude) * 1.5)
    penalty += 0.08 if not root_stable else 0.0
    penalty += 0.08 if not foot_stable else 0.0
    penalty += 0.12 if not joint_valid else 0.0
    return round(max(0.0, min(1.0, 1.0 - penalty)), 4)


def analyze_motion_samples(
    frames: Sequence[MotionFrame], expected_fps: float
) -> QualityAnalysis:
    """Analyze a compact sampled pose stream without inventing missing motion."""
    hard: list[str] = []
    warnings: list[str] = []
    if not frames:
        return QualityAnalysis(QualityStatus.REJECTED, 0.0, ("EMPTY_MOTION",), ("EMPTY_MOTION",), (), 1.0, 0.0)
    if expected_fps <= 0 or not math.isfinite(expected_fps):
        hard.append("INVALID_SAMPLE_RATE")

    timestamps = [frame.timestamp for frame in frames]
    if any(not math.isfinite(value) for value in timestamps):
        hard.append("NON_FINITE_SAMPLE_TIMES")
    elif any(right <= left for left, right in zip(timestamps, timestamps[1:])):
        hard.append("NON_MONOTONIC_SAMPLE_TIMES")

    expected_interval = 1.0 / expected_fps if expected_fps > 0 else 0.0
    missing_count = 0
    if expected_interval > 0:
        for left, right in zip(timestamps, timestamps[1:]):
            interval = right - left
            if interval > expected_interval * 1.75:
                missing_count += max(1, int(round(interval / expected_interval)) - 1)
    missing_ratio = min(1.0, missing_count / max(1, len(frames)))
    if missing_count:
        warnings.append("MISSING_SAMPLES")
    if missing_ratio > 0.20:
        hard.append("EXCESSIVE_MISSING_SAMPLES")

    for frame in frames:
        for joint, position in frame.positions.items():
            if not all(math.isfinite(value) for value in position):
                hard.append("NON_FINITE_POSITION")
                break
        for joint, rotation in frame.rotations.items():
            if not all(math.isfinite(value) for value in rotation):
                hard.append("NON_FINITE_ROTATION")
                break
            norm = math.sqrt(sum(value * value for value in rotation))
            if not 0.9 <= norm <= 1.1:
                hard.append("INVALID_QUATERNION")
                break

    left_foot = _series(frames, "LeftFoot")
    right_foot = _series(frames, "RightFoot")
    foot_stable = True
    if left_foot and right_foot:
        minimum_y = min(point[1] for point in left_foot + right_foot)
        if minimum_y < -0.20:
            hard.append("FOOT_PENETRATION")
            foot_stable = False
        elif minimum_y < -0.05:
            warnings.append("FOOT_PENETRATION_WARNING")
            foot_stable = False
        sliding = max(
            _horizontal_distance(left_foot),
            _horizontal_distance(right_foot),
        )
        if sliding > 1.0:
            warnings.append("FOOT_SLIDING")
            foot_stable = False
    else:
        warnings.append("FOOT_DATA_UNAVAILABLE")

    root = _series(frames, "Hips")
    root_stable = True
    if not root:
        hard.append("ROOT_DATA_MISSING")
        root_stable = False
    else:
        vertical = max(point[1] for point in root) - min(point[1] for point in root)
        if vertical > 1.5:
            hard.append("EXTREME_ROOT_TRAJECTORY")
            root_stable = False
        elif vertical > 0.8:
            warnings.append("ROOT_VERTICAL_MOVEMENT")
            root_stable = False

    accelerations = _accelerations(frames)
    max_acceleration = max(accelerations, default=0.0)
    if max_acceleration > 150.0:
        hard.append("EXCESSIVE_ACCELERATION")
    elif max_acceleration > 50.0:
        warnings.append("HIGH_ACCELERATION")

    joint_valid = not any(
        item in {"NON_FINITE_POSITION", "NON_FINITE_ROTATION", "INVALID_QUATERNION"}
        for item in hard
    )
    hard = list(dict.fromkeys(hard))
    warnings = list(dict.fromkeys(warnings))
    status = QualityStatus.REJECTED if hard else (QualityStatus.FLAGGED if warnings else QualityStatus.VALID)
    score = compute_quality_score(
        hard_failures=len(hard),
        warnings=len(warnings),
        missing_sample_ratio=missing_ratio,
        root_stable=root_stable,
        foot_stable=foot_stable,
        joint_valid=joint_valid,
    )
    return QualityAnalysis(
        status,
        score,
        tuple(dict.fromkeys((*hard, *warnings))),
        tuple(hard),
        tuple(warnings),
        missing_ratio,
        max_acceleration,
    )


def _series(frames: Iterable[MotionFrame], joint: str) -> list[tuple[float, float, float]]:
    return [frame.positions[joint] for frame in frames if joint in frame.positions]


def _horizontal_distance(points: Sequence[tuple[float, float, float]]) -> float:
    if not points:
        return 0.0
    first = points[0]
    last = points[-1]
    return math.hypot(last[0] - first[0], last[2] - first[2])


def _accelerations(frames: Sequence[MotionFrame]) -> list[float]:
    result: list[float] = []
    for index in range(1, len(frames) - 1):
        dt1 = frames[index].timestamp - frames[index - 1].timestamp
        dt2 = frames[index + 1].timestamp - frames[index].timestamp
        if dt1 <= 0 or dt2 <= 0:
            continue
        for joint in frames[index].positions:
            if joint not in frames[index - 1].positions or joint not in frames[index + 1].positions:
                continue
            before = frames[index - 1].positions[joint]
            current = frames[index].positions[joint]
            after = frames[index + 1].positions[joint]
            velocity1 = _difference(current, before, dt1)
            velocity2 = _difference(after, current, dt2)
            acceleration = _difference(velocity2, velocity1, (dt1 + dt2) * 0.5)
            result.append(math.sqrt(sum(value * value for value in acceleration)))
    return result


def _difference(
    left: tuple[float, ...], right: tuple[float, ...], divisor: float
) -> tuple[float, ...]:
    return tuple((left[index] - right[index]) / divisor for index in range(len(left)))
