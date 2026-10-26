"""Structured action execution result used by the Unity bridge."""

from __future__ import annotations

from .models import ActionResult, ActionType, ExecutionStatus


class ActionExecutor:
    """A safe executor receives only validated allowlisted actions."""

    def execute(self, action: ActionType, target: str | None, text: str | None) -> ActionResult:
        raise NotImplementedError
