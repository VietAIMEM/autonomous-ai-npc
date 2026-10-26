"""Deterministic HumanML3D-263 output validation."""

from __future__ import annotations

import math
from typing import Any, Optional

from .models import MotionGenerationError


DEFAULT_MOTION_RATE = 20.0
DEFAULT_EXPECTED_DIMENSIONS = 263
DEFAULT_MAX_FRAMES = 100


def validate_motion_payload(
    payload: Any,
    *,
    expected_dimensions: int = DEFAULT_EXPECTED_DIMENSIONS,
    frame_rate: float = DEFAULT_MOTION_RATE,
    max_frames: int = DEFAULT_MAX_FRAMES,
    expected_duration_seconds: Optional[float] = None,
) -> dict[str, Any]:
    """Validate shape, finite values, and duration without importing numpy."""
    if not isinstance(payload, (list, tuple)):
        raise MotionGenerationError("INVALID_MOTION_OUTPUT", "payload is not a sequence")
    if not payload:
        raise MotionGenerationError("EMPTY_MOTION_OUTPUT")
    if expected_dimensions <= 0 or frame_rate <= 0 or max_frames <= 0:
        raise MotionGenerationError("INVALID_MOTION_VALIDATION", "invalid validation limits")
    if len(payload) > max_frames:
        raise MotionGenerationError("INVALID_MOTION_LENGTH", f"{len(payload)} frames exceeds {max_frames}")

    for row_index, row in enumerate(payload):
        if not isinstance(row, (list, tuple)):
            raise MotionGenerationError("INVALID_MOTION_OUTPUT", f"row {row_index} is not a sequence")
        if len(row) != expected_dimensions:
            raise MotionGenerationError(
                "INVALID_MOTION_DIMENSIONS",
                f"row {row_index} has {len(row)} values, expected {expected_dimensions}",
            )
        for column_index, value in enumerate(row):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise MotionGenerationError(
                    "INVALID_MOTION_VALUE",
                    f"row {row_index}, column {column_index} is not finite",
                )

    duration_seconds = len(payload) / frame_rate
    if expected_duration_seconds is not None:
        tolerance = max(0.05, expected_duration_seconds * 0.02)
        if abs(duration_seconds - expected_duration_seconds) > tolerance:
            raise MotionGenerationError("INVALID_MOTION_DURATION")

    return {
        "format": "humanml3d-263",
        "motion_rate": float(frame_rate),
        "frame_count": len(payload),
        "duration_seconds": float(duration_seconds),
        "dimensions": expected_dimensions,
    }
