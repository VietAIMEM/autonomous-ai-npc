"""Minimal motion generation abstraction shared by Phase 15 providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .models import MotionGenerationRequest, MotionGenerationResult


class MotionGenerator(ABC):
    """A provider turns a controlled prompt into validated motion output."""

    name: str = "unknown"

    @abstractmethod
    def generate(self, request: MotionGenerationRequest) -> MotionGenerationResult:
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        return {"provider": self.name}
