"""Deterministic offline motion generator used by automated tests."""

from __future__ import annotations

import hashlib
import re
import uuid

from .base import MotionGenerator
from .models import (
    MotionGenerationConfig,
    MotionGenerationError,
    MotionGenerationRequest,
    MotionGenerationResult,
    MotionGenerationStatus,
)
from .validation import DEFAULT_EXPECTED_DIMENSIONS, validate_motion_payload


class MockMotionGenerator(MotionGenerator):
    name = "mock"

    def __init__(self, config: MotionGenerationConfig | None = None) -> None:
        self.config = config or MotionGenerationConfig()

    def generate(self, request: MotionGenerationRequest) -> MotionGenerationResult:
        prompt = request.prompt.strip()
        if not prompt:
            raise MotionGenerationError("EMPTY_PROMPT")
        if len(prompt) > self.config.max_prompt_length:
            raise MotionGenerationError("PROMPT_TOO_LONG")

        frames = round(request.duration_seconds * self.config.motion_rate)
        if frames <= 0 or frames > 100:
            raise MotionGenerationError("INVALID_MOTION_LENGTH")

        payload = tuple(
            tuple(0.0 for _ in range(DEFAULT_EXPECTED_DIMENSIONS))
            for _ in range(frames)
        )
        validate_motion_payload(
            payload,
            frame_rate=self.config.motion_rate,
            expected_duration_seconds=request.duration_seconds,
        )
        return MotionGenerationResult(
            request=request,
            motion_id=_motion_id(request),
            duration_seconds=frames / self.config.motion_rate,
            format=self.config.representation,
            generation_status=MotionGenerationStatus.GENERATED,
            motion=payload,
        )


def _motion_id(request: MotionGenerationRequest) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", request.prompt.strip().casefold()).strip("_")[:40]
    digest = hashlib.sha256(request.prompt.strip().casefold().encode("utf-8")).hexdigest()[:10]
    return f"text_{slug or 'motion'}_{digest}_{uuid.uuid4().hex[:8]}_v1"
