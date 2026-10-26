"""Typed Phase 15 motion generation models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


PROVIDER_MOCK = "mock"
PROVIDER_MOTIUS = "motius"
PROVIDER_DISABLED = "disabled"

DEFAULT_MODEL = "Motius T2M-GPT"
DEFAULT_CHECKPOINT = "D:/motius/model-cache/Motius-T2M-GPT-HumanML3D"
DEFAULT_PIPELINE_CLASS = "motius.pipelines.t2mgpt.T2MGPTPipeline"
DEFAULT_REPRESENTATION = "humanml3d-263"
DEFAULT_MOTION_RATE = 20.0
DEFAULT_MOTIUS_PYTHON = "D:/motius/Motius/.venv_t2m/Scripts/python.exe"
DEFAULT_MOTIUS_REPO = "D:/motius/Motius"


class MotionGenerationStatus(str, Enum):
    QUEUED = "QUEUED"
    GENERATING = "GENERATING"
    GENERATED = "GENERATED"
    VALIDATED = "VALIDATED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class MotionGenerationConfig:
    provider: str = PROVIDER_MOCK
    model: str = DEFAULT_MODEL
    checkpoint: str = DEFAULT_CHECKPOINT
    pipeline_class: str = DEFAULT_PIPELINE_CLASS
    representation: str = DEFAULT_REPRESENTATION
    motion_rate: float = DEFAULT_MOTION_RATE
    device: str = "cpu"
    cache_dir: Optional[str] = None
    python_executable: str = DEFAULT_MOTIUS_PYTHON
    repo_path: str = DEFAULT_MOTIUS_REPO
    timeout_seconds: float = 120.0
    max_prompt_length: int = 500
    max_duration_seconds: float = 5.0
    min_duration_seconds: float = 0.5
    max_queue_depth: int = 2
    max_concurrent_generations: int = 1
    max_retained_jobs: int = 64
    worker_poll_interval_seconds: float = 0.05

    def __post_init__(self) -> None:
        if self.motion_rate <= 0:
            raise ValueError("motion_rate must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_prompt_length <= 0:
            raise ValueError("max_prompt_length must be positive")
        if self.max_duration_seconds <= 0:
            raise ValueError("max_duration_seconds must be positive")
        if self.min_duration_seconds <= 0:
            raise ValueError("min_duration_seconds must be positive")
        if self.max_queue_depth <= 0:
            raise ValueError("max_queue_depth must be positive")
        if self.max_retained_jobs <= 0:
            raise ValueError("max_retained_jobs must be positive")
        if self.max_concurrent_generations != 1:
            raise ValueError("max_concurrent_generations must be 1")


@dataclass(frozen=True)
class MotionGenerationRequest:
    prompt: str
    duration_seconds: float = 2.5
    motion_type: str = "general"
    style: str = ""
    speed: float = 1.0
    seed: Optional[int] = None
    dedup_key: str = ""


@dataclass(frozen=True)
class MotionGenerationResult:
    request: MotionGenerationRequest
    motion_id: str
    duration_seconds: float
    format: str
    animation_reference: str = ""
    generation_status: MotionGenerationStatus = MotionGenerationStatus.GENERATED
    metadata: dict[str, Any] = field(default_factory=dict)
    motion: tuple[tuple[float, ...], ...] = ()
    validation_status: str = "PENDING"
    error: Optional[str] = None
    requested_at: datetime = datetime.min
    generated_at: Optional[datetime] = None
    validated_at: Optional[datetime] = None
    generation_ms: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        """Diagnostics and persistence view; never contains the raw motion."""
        return {
            "motion_id": self.motion_id,
            "source": "GENERATED",
            "prompt": self.request.prompt,
            "duration_seconds": self.duration_seconds,
            "format": self.format,
            "animation_reference": self.animation_reference,
            "generation_status": self.generation_status.value,
            "metadata": dict(self.metadata),
            "validation_status": self.validation_status,
            "error": self.error,
            "requested_at": self.requested_at.isoformat(),
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
            "validated_at": self.validated_at.isoformat() if self.validated_at else None,
            "generation_ms": self.generation_ms,
        }


class MotionGenerationError(RuntimeError):
    """Raised for provider transport, protocol, or output-payload failures."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}:{detail}" if detail else reason)
        self.reason = reason
        self.detail = detail
