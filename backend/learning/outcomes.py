"""Interpretable skill outcome scoring."""

from __future__ import annotations

from ..skills.models import SkillEvaluation, SkillStatus
from .context import build_context_key
from .models import ExperienceRecord


class OutcomeEvaluator:
    """Convert a completed SkillEvaluation into a bounded 0..1 score.

    Execution success dominates social quality: SUCCESS starts at 0.70,
    FAILED at 0.20, and CANCELLED/INTERRUPTED at 0.30. Positive viewer
    responses add 0.15, negative responses subtract 0.15, failed steps
    subtract 0.05 each, and interruption/cancellation subtracts 0.10.
    """

    _POSITIVE = {"positive", "like", "liked", "great", "good", "clap", "happy"}
    _NEGATIVE = {"negative", "dislike", "disliked", "bad", "boring", "angry"}

    def evaluate(
        self,
        evaluation: SkillEvaluation,
        context_key: str | None = None,
        timestamp=None,
        metadata: dict | None = None,
    ) -> ExperienceRecord:
        if evaluation.status == SkillStatus.SUCCESS:
            score, success = 0.70, True
        elif evaluation.status == SkillStatus.FAILED:
            score, success = 0.20, False
        else:
            score, success = 0.30, False

        response = (evaluation.viewer_response or "").strip().casefold()
        positive = response in self._POSITIVE
        negative = response in self._NEGATIVE
        score += 0.15 if positive else 0.0
        score -= 0.15 if negative else 0.0
        score -= 0.05 * max(0, evaluation.steps_failed)
        if evaluation.status in {SkillStatus.INTERRUPTED, SkillStatus.CANCELLED}:
            score -= 0.10

        return ExperienceRecord(
            skill_id=evaluation.skill_id,
            context_key=context_key or build_context_key(),
            execution_id=evaluation.execution_id,
            status=evaluation.status.value,
            outcome_score=round(max(0.0, min(1.0, score)), 6),
            duration_seconds=evaluation.duration_seconds,
            viewer_response=evaluation.viewer_response,
            interruption_reason=evaluation.interruption_reason,
            timestamp=timestamp,
            metadata={
                **(metadata or {}),
                "success": success,
                "positive_reaction": positive,
                "negative_reaction": negative,
            },
        )
