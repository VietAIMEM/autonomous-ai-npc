"""Advisory skill-learning service attached to the existing Brain loop."""

from __future__ import annotations

import random
import threading
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from ..chat.models import utc_now
from ..brain.models import ActionType, PersonalityProfile
from ..brain.personalization import skill_preference_bias
from ..runtime import instrument
from ..skills.models import Skill, SkillActionStep, SkillCategory
from ..skills.models import SkillCompositionStep
from .context import build_context_key, context_from_world
from .models import (
    CompositeKnowledge,
    LearningConfig,
    MotionDefinition,
    MotionSource,
    MotionValidationStatus,
    SkeletonCompatibility,
    SkillKnowledge,
)
from .motion_validation import MotionValidationError, validate_motion_source
from .outcomes import OutcomeEvaluator
from .repository import SQLiteSkillLearningRepository


_BASE_SCORE = 0.45
_MAX_PROMPT_CANDIDATES = 8


class SkillLearningService:
    def __init__(
        self,
        library,
        config: LearningConfig | None = None,
        repository: SQLiteSkillLearningRepository | None = None,
        clock: Callable[[], datetime] = utc_now,
        random_source: random.Random | None = None,
    ) -> None:
        self.library = library
        self.config = config or LearningConfig()
        self.repository = repository
        self._clock = clock
        self._random = random_source or random.Random(1005)
        self._evaluator = OutcomeEvaluator()
        self._lock = threading.RLock()
        self._enabled = self.config.enabled
        self._motions: dict[str, MotionDefinition] = {}
        self._knowledge: dict[tuple[str, str], SkillKnowledge] = {}
        self._composites: dict[tuple[str, str], CompositeKnowledge] = {}
        self._last_context_key: Optional[str] = None
        self._last_selections: dict[str, str] = {}
        self._selected_sequence: list[str] = []
        if self.repository is not None:
            self._motions = self.repository.load_motions()
            self._knowledge = self.repository.load_knowledge()
            self._composites = self.repository.load_composites()
            for motion in self._motions.values():
                self._register_motion_skill(motion)
            for composite in self._composites.values():
                self._ensure_composite_skill(composite)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)

    def register_motion(
        self,
        motion_id: str,
        name: str,
        description: str,
        animation_reference: str,
        duration_seconds: float,
        *,
        tags: Iterable[str] = (),
        source: MotionSource | str = MotionSource.IMPORTED,
        source_path: Path | str | None = None,
        source_format: str | None = None,
        clip_name: str | None = None,
        sample_rate: float | None = None,
        skill_category: SkillCategory = SkillCategory.SOCIAL,
        execution_action: ActionType = ActionType.DANCE,
        enabled: bool = True,
        version: int = 1,
    ) -> tuple[MotionDefinition, Skill]:
        if not self._enabled:
            raise RuntimeError("LEARNING_DISABLED")
        if source_path is None:
            raise MotionValidationError("MOTION_SOURCE_REQUIRED")
        metadata = validate_motion_source(
            source_path,
            animation_reference,
            duration_seconds,
            source_format=source_format,
            clip_name=clip_name,
            sample_rate=sample_rate,
        )
        if metadata.skeleton_status != SkeletonCompatibility.COMPATIBLE:
            raise MotionValidationError("MOTION_NOT_EXECUTABLE_WITHOUT_RETARGETING")
        source = source if isinstance(source, MotionSource) else MotionSource(str(source).upper())
        with self._lock:
            motion = MotionDefinition(
                motion_id=motion_id,
                name=name,
                description=description,
                animation_reference=animation_reference,
                duration_seconds=metadata.duration_seconds,
                tags=tuple(dict.fromkeys(str(tag).casefold() for tag in tags)),
                source=source,
                parameters=(),
                execution_action=execution_action,
                enabled=enabled,
                version=version,
                source_path=metadata.source_path,
                source_format=metadata.source_format,
                clip_name=metadata.clip_name,
                frame_count=metadata.frame_count,
                sample_rate=metadata.sample_rate,
                bone_count=metadata.bone_count,
                root_bone_name=metadata.root_bone_name,
                skeleton_status=metadata.skeleton_status,
                validation_status=MotionValidationStatus.VALID,
            )
            if motion.motion_id in self._motions or self.library.exists(motion.motion_id):
                raise ValueError("DUPLICATE_MOTION_OR_SKILL_ID")

            skill = self._register_motion_skill(motion, skill_category)
            self._motions[motion.motion_id] = motion
            if self.repository is not None:
                self.repository.save_motion(motion)
            return motion, skill

    def validate_motion(
        self,
        source_path: Path | str,
        animation_reference: str,
        duration_seconds: float,
        *,
        source_format: str | None = None,
        clip_name: str | None = None,
        sample_rate: float | None = None,
    ) -> dict[str, Any]:
        metadata = validate_motion_source(
            source_path,
            animation_reference,
            duration_seconds,
            source_format=source_format,
            clip_name=clip_name,
            sample_rate=sample_rate,
        )
        return {
            "valid": metadata.skeleton_status == SkeletonCompatibility.COMPATIBLE,
            **metadata.to_dict(),
        }

    def register_validated_motion(
        self,
        motion: MotionDefinition,
        *,
        skill_category=None,
    ) -> tuple[MotionDefinition, Skill]:
        """Register an externally validated persistent motion through the shared gate."""
        if not self._enabled:
            raise RuntimeError("LEARNING_DISABLED")
        if not isinstance(motion, MotionDefinition):
            raise ValueError("INVALID_MOTION_DEFINITION")
        if not motion.enabled:
            raise MotionValidationError("MOTION_DISABLED")
        if motion.validation_status != MotionValidationStatus.VALID:
            raise MotionValidationError("MOTION_NOT_VALID")
        if motion.skeleton_status != SkeletonCompatibility.COMPATIBLE:
            raise MotionValidationError("MOTION_NOT_EXECUTABLE_WITHOUT_RETARGETING")
        if motion.source_motion_id is not None:
            source = self.get_motion(motion.source_motion_id)
            if source is None or source.version != motion.source_version:
                raise MotionValidationError("MOTION_LINEAGE_MISSING")
        if "__preview__scene" in motion.animation_reference.casefold():
            raise MotionValidationError("PREVIEW_ANIMATION_FORBIDDEN")

        with self._lock:
            if motion.motion_id in self._motions or self.library.exists(motion.motion_id):
                raise ValueError("DUPLICATE_MOTION_OR_SKILL_ID")
            skill = self._register_motion_skill(motion, skill_category)
            if skill is None:
                raise MotionValidationError("MOTION_REGISTRATION_GATE_FAILED")
            self._motions[motion.motion_id] = motion
            if self.repository is not None:
                self.repository.save_motion(motion)
            return motion, skill

    def _register_motion_skill(self, motion: MotionDefinition, skill_category=None):
        if (
            motion.validation_status != MotionValidationStatus.VALID
            or motion.skeleton_status != SkeletonCompatibility.COMPATIBLE
        ):
            if self.library.exists(motion.motion_id):
                self.library.disable(motion.motion_id)
            return None
        if self.library.exists(motion.motion_id):
            return self.library.get(motion.motion_id)
        return self.library.register(
            Skill(
                skill_id=motion.motion_id,
                name=motion.name,
                description=motion.description,
                category=skill_category or SkillCategory.SOCIAL,
                animation_reference=motion.animation_reference,
                estimated_duration_seconds=motion.duration_seconds,
                tags=tuple(dict.fromkeys(("learned", *motion.tags))),
                version=motion.version,
                steps=(
                    SkillActionStep(
                        motion.execution_action,
                        constant_target=motion.animation_reference,
                    ),
                ),
                enabled=motion.enabled,
            )
        )

    def motions(self) -> list[dict[str, Any]]:
        with self._lock:
            return [motion.to_dict() for motion in self._motions.values()]

    def get_motion(self, motion_id: str) -> MotionDefinition | None:
        with self._lock:
            return self._motions.get(motion_id.strip().lower())

    def get_motion_state(self, motion_id: str) -> dict[str, Any]:
        motion = self.get_motion(motion_id)
        if motion is None:
            raise KeyError("motion_not_found")
        return motion.to_dict()

    def disable_motion(self, motion_id: str) -> MotionDefinition:
        return self._set_motion_enabled(motion_id, False)

    def enable_motion(self, motion_id: str) -> MotionDefinition:
        return self._set_motion_enabled(motion_id, True)

    def _set_motion_enabled(self, motion_id: str, enabled: bool) -> MotionDefinition:
        with self._lock:
            current = self._motions.get(motion_id.strip().lower())
            if current is None:
                raise KeyError("motion_not_found")
            motion = replace(current, enabled=enabled)
            if self.library.exists(motion.motion_id):
                if enabled:
                    self.library.enable(motion.motion_id)
                else:
                    self.library.disable(motion.motion_id)
            self._motions[motion.motion_id] = motion
            if self.repository is not None:
                self.repository.save_motion(motion)
            return motion

    def observe_context(self, goal, chat, world, viewer_memory=None) -> str:
        key = context_from_world(goal, chat, world, viewer_memory)
        with self._lock:
            self._last_context_key = key
        return key

    def rank_candidates(
        self,
        context_key: str,
        skill_ids: Iterable[str] | None = None,
        *,
        choose: bool = False,
    ) -> list[dict[str, Any]]:
        with self._lock:
            ids = list(
                dict.fromkeys(
                    str(item).lower()
                    for item in (skill_ids or [skill.skill_id for skill in self.library.list(enabled_only=True)])
                )
            )
            candidates = []
            for skill_id in ids:
                skill = self.library.get(skill_id)
                if skill is None or not skill.enabled:
                    continue
                knowledge = self._knowledge.get((skill_id, context_key))
                exploration = choose and self._should_explore(knowledge)
                if exploration:
                    knowledge = self._mark_exploration(knowledge)
                candidates.append(
                    {
                        "skill_id": skill_id,
                        "score": round(self._ranking_score(knowledge, exploration), 4),
                        "exploration": exploration,
                        "knowledge": knowledge.to_dict() if knowledge else None,
                    }
                )
            candidates.sort(key=lambda item: (-item["score"], item["skill_id"]))
            if choose and candidates:
                selected = candidates[0]["skill_id"]
                knowledge = self._knowledge.get((selected, context_key))
                if knowledge is not None:
                    knowledge.exploitation_count += 1
                    if self.repository is not None:
                        self.repository.save_knowledge(knowledge)
                self._last_selections[selected] = context_key
                self._selected_sequence.append(selected)
                self._selected_sequence = self._selected_sequence[-self.config.max_child_skills :]
            return candidates[:_MAX_PROMPT_CANDIDATES]

    def should_explore(self, skill_id: str, context_key: str) -> bool:
        with self._lock:
            return self._should_explore(self._knowledge.get((skill_id.lower(), context_key)))

    def _should_explore(self, knowledge: SkillKnowledge | None) -> bool:
        if not self._enabled:
            return False
        if knowledge is None:
            return self._random.random() < self.config.exploration_epsilon
        if knowledge.confidence >= self.config.min_explore_confidence:
            return False
        return self._random.random() < self.config.exploration_epsilon

    def _mark_exploration(self, knowledge: SkillKnowledge | None) -> SkillKnowledge | None:
        if knowledge is None:
            return None
        knowledge.exploration_count += 1
        instrument.count("learning.exploration")
        if self.repository is not None:
            self.repository.save_knowledge(knowledge)
        return knowledge

    def _ranking_score(self, knowledge: SkillKnowledge | None, exploration: bool) -> float:
        if knowledge is None:
            return _BASE_SCORE + (self.config.exploration_confidence_bonus if exploration else 0.0)
        confidence = min(1.0, knowledge.trials / 10.0)
        score = (
            0.45 * knowledge.success_rate
            + 0.40 * knowledge.average_score
            + 0.15 * confidence
        )
        if exploration:
            score += self.config.exploration_confidence_bonus
        return min(1.0, score)

    def observe_evaluation(
        self,
        evaluation: SkillEvaluation,
        *,
        context_key: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if not self._enabled or evaluation is None:
            return None
        key = context_key or self._last_context_key or build_context_key()
        record = self._evaluator.evaluate(evaluation, key, self._clock(), metadata)
        with self._lock:
            knowledge = self._knowledge.get((record.skill_id, key))
            if knowledge is None:
                knowledge = SkillKnowledge(record.skill_id, key)
                existing = [item for (skill_id, _), item in self._knowledge.items() if skill_id == record.skill_id]
                if len(existing) >= self.config.max_contexts_per_skill:
                    oldest = min(existing, key=lambda item: item.last_used or self._clock())
                    self._knowledge.pop((record.skill_id, oldest.context_key), None)
            knowledge.trials += 1
            if record.metadata["success"]:
                knowledge.successes += 1
            else:
                knowledge.failures += 1
            if record.status == "INTERRUPTED":
                knowledge.interruptions += 1
            if record.metadata["positive_reaction"]:
                knowledge.positive_reactions += 1
            if record.metadata["negative_reaction"]:
                knowledge.negative_reactions += 1
            knowledge.total_score += record.outcome_score
            knowledge.total_duration_seconds += record.duration_seconds
            knowledge.last_used = record.timestamp
            knowledge.confidence = min(
                0.98,
                knowledge.confidence + self.config.confidence_growth,
            )
            self._knowledge[(record.skill_id, key)] = knowledge
            self._record_composite_candidate(key)
            if self.repository is not None:
                self.repository.save_knowledge(knowledge)
            instrument.count("learning.updates")
            if record.metadata["success"]:
                instrument.count("learning.success")
            else:
                instrument.count("learning.failure")
            instrument.gauge("learning.knowledge_entries", float(len(self._knowledge)))
            instrument.gauge("learning.composites", float(len(self._composites)))
            return {"experience": record.to_dict(), "knowledge": knowledge.to_dict()}

    def _record_composite_candidate(self, context_key: str) -> None:
        sequence = tuple(self._selected_sequence)
        if len(sequence) < 2:
            return
        composite_id = "_and_".join(sequence)[:120]
        existing = self._composites.get((composite_id, context_key))
        if existing is None:
            existing = CompositeKnowledge(composite_id, sequence, context_key, created_at=self._clock())
        existing.trials += 1
        last = self._knowledge.get((sequence[-1], context_key))
        existing.total_score += last.average_score if last else 0.0
        if all(
            (self._knowledge.get((skill_id, context_key)) or SkillKnowledge(skill_id, context_key)).successes
            for skill_id in sequence
        ):
            existing.successes += 1
        existing.last_used = self._clock()
        existing.confidence = min(0.98, 0.35 + existing.trials * self.config.confidence_growth)
        existing.enabled = (
            existing.successes >= self.config.min_composite_trials
            and existing.success_rate >= self.config.min_composite_score
        )
        self._composites[(composite_id, context_key)] = existing
        if self.repository is not None:
            self.repository.save_composite(existing)

    def get_state(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": self._enabled,
                "motion_count": len(self._motions),
                "knowledge_entries": len(self._knowledge),
                "composite_count": len(self._composites),
                "last_context_key": self._last_context_key,
                "exploration_epsilon": self.config.exploration_epsilon,
                "motions": self.motions(),
            }

    def get_skill_state(self, skill_id: str) -> dict[str, Any]:
        skill_id = skill_id.lower()
        with self._lock:
            entries = sorted(
                (knowledge for (candidate, _), knowledge in self._knowledge.items() if candidate == skill_id),
                key=lambda item: item.context_key,
            )
            return {"skill_id": skill_id, "contexts": [knowledge.to_dict() for knowledge in entries]}

    def get_context_state(self, context_key: str) -> dict[str, Any]:
        with self._lock:
            entries = sorted(
                (knowledge for (_, key), knowledge in self._knowledge.items() if key == context_key),
                key=lambda item: item.skill_id,
            )
            return {
                "context_key": context_key,
                "skills": [knowledge.to_dict() for knowledge in entries],
                "composites": [
                    composite.to_dict()
                    for composite in self._composites.values()
                    if composite.context_key == context_key
                ],
            }

    def get_compositions(self) -> list[dict[str, Any]]:
        with self._lock:
            return [composite.to_dict() for composite in self._composites.values()]

    def register_learned_composite(
        self,
        composite_skill_id: str,
        child_skill_ids: Iterable[Any],
        context_key: str,
        *,
        name: str,
        description: str,
        enabled: bool = True,
    ) -> Skill:
        normalized: list[tuple[str, dict[str, Any]]] = []
        seen: set[str] = set()
        for child in child_skill_ids:
            if isinstance(child, SkillCompositionStep):
                skill_id, parameters = child.skill_id, child.parameters
            elif isinstance(child, str):
                skill_id, parameters = child, {}
            elif isinstance(child, (tuple, list)) and len(child) == 2:
                skill_id, parameters = child
            else:
                raise ValueError("INVALID_COMPOSITION_CHILD")
            if not isinstance(skill_id, str) or not isinstance(parameters, dict):
                raise ValueError("INVALID_COMPOSITION_CHILD")
            skill_id = skill_id.strip().lower()
            if not skill_id or skill_id in seen:
                continue
            normalized.append((skill_id, dict(parameters)))
            seen.add(skill_id)
        children = tuple(normalized)
        if not children or len(children) > self.config.max_child_skills:
            raise ValueError("INVALID_COMPOSITION_SIZE")
        missing = [skill_id for skill_id, _ in children if self.library.get(skill_id) is None]
        if missing:
            raise ValueError("COMPOSITION_DEPENDENCY_MISSING:" + ",".join(missing))
        if self.library.exists(composite_skill_id):
            raise ValueError("DUPLICATE_MOTION_OR_SKILL_ID")
        existing = self._composites.get((composite_skill_id, context_key))
        if existing is None:
            existing = CompositeKnowledge(
                composite_skill_id,
                tuple(skill_id for skill_id, _ in children),
                context_key,
                trials=1,
                successes=1,
                total_score=1.0,
                confidence=0.98,
                created_at=self._clock(),
                last_used=self._clock(),
                enabled=enabled,
                child_parameters=tuple(parameters for _, parameters in children),
            )
        self._composites[(composite_skill_id, context_key)] = existing
        skill = Skill(
            skill_id=composite_skill_id,
            name=name,
            description=description,
            category=SkillCategory.COMPOSITE,
            composition=tuple(
                SkillCompositionStep(skill_id, parameters) for skill_id, parameters in children
            ),
            estimated_duration_seconds=sum(
                self.library.require(skill_id).estimated_duration_seconds
                for skill_id, _ in children
            ),
            tags=("learned", "composite"),
            enabled=enabled,
        )
        self.library.register(skill)
        if self.repository is not None:
            self.repository.save_composite(existing)
        return skill

    def _ensure_composite_skill(self, composite: CompositeKnowledge) -> None:
        if self.library.exists(composite.composite_skill_id):
            return
        try:
            self.library.register(
                Skill(
                    skill_id=composite.composite_skill_id,
                    name="Learned Sequence",
                    description="Learned successful skill sequence for this context.",
                    category=SkillCategory.COMPOSITE,
                    composition=tuple(
                        SkillCompositionStep(
                            skill_id,
                            composite.child_parameters[index]
                            if index < len(composite.child_parameters)
                            else {},
                        )
                        for index, skill_id in enumerate(composite.child_skill_ids)
                    ),
                    estimated_duration_seconds=sum(
                        self.library.require(child).estimated_duration_seconds
                        for child in composite.child_skill_ids
                    ),
                    tags=("learned", "composite"),
                    enabled=True,
                )
            )
        except Exception:
            # The learned record remains inspectable, but only a valid
            # library composition can execute.
            pass

    def ranking_prompt_context(
        self,
        context_key: str,
        *,
        personalization: Any | None = None,
        personality: PersonalityProfile | None = None,
    ) -> list[dict[str, Any]]:
        """Advisory ranking for the planner prompt.

        PHASE 19 adds two bounded *preference* signals on top of the learned
        history: repeated viewer evidence and the stable personality. They only
        reorder ``advisory_score``; the stored :class:`SkillKnowledge` is never
        rewritten and every hard constraint stays authoritative.
        """
        candidates = self.rank_candidates(context_key)
        preferred_skills = self._viewer_preferred_skills(personalization)
        prompt: list[dict[str, Any]] = []
        for candidate in candidates:
            skill = self.library.get(candidate["skill_id"])
            if skill is None:
                continue
            knowledge = candidate.get("knowledge") or {}
            bias, bias_reason = skill_preference_bias(
                skill.skill_id,
                skill_category=skill.category.value,
                preferred_skills=preferred_skills,
                personality=personality,
            )
            advisory_score = round(min(1.0, max(0.0, candidate["score"] + bias)), 4)
            prompt.append(
                {
                    "skill_id": skill.skill_id,
                    "description": skill.description,
                    "category": skill.category.value,
                    "learned": "learned" in skill.tags,
                    "learned_success_rate": knowledge.get("success_rate"),
                    "learned_average_score": knowledge.get("average_score"),
                    "confidence": knowledge.get("confidence"),
                    "exploration_candidate": candidate["exploration"],
                    "advisory_score": advisory_score,
                    "personalization_bias": bias or None,
                    "personalization_reason": bias_reason,
                }
            )
        return prompt

    @staticmethod
    def _viewer_preferred_skills(personalization: Any | None) -> tuple[str, ...]:
        """Bounded preferred-skill ids from the Phase 19 context; never raises."""
        if personalization is None:
            return ()
        try:
            if hasattr(personalization, "preferred_skills"):
                values = personalization.preferred_skills
            elif isinstance(personalization, dict):
                values = personalization.get("preferred_skills")
            else:
                return ()
        except Exception:
            return ()
        if not isinstance(values, (list, tuple)):
            return ()
        cleaned = [
            str(item).strip().lower()[:60]
            for item in values
            if isinstance(item, str) and item.strip()
        ]
        return tuple(dict.fromkeys(cleaned))[:5]

    def reset(self) -> None:
        with self._lock:
            self._knowledge.clear()
            self._composites.clear()
            self._last_selections.clear()
            self._selected_sequence.clear()
        if self.repository is not None:
            self.repository.reset()

    def close(self) -> None:
        if self.repository is not None:
            self.repository.close()
