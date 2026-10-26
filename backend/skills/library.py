"""In-memory registry for source-defined, allowlisted skills."""

from __future__ import annotations

import threading
from typing import Callable, Iterable, Optional

from .models import Skill, SkillCategory, SkillCompositionStep


class SkillRegistryError(ValueError):
    """Raised when a skill definition violates the controlled schema."""


class SkillLibrary:
    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}
        self._lock = threading.RLock()

    def register(self, skill: Skill) -> Skill:
        with self._lock:
            if skill.skill_id in self._skills:
                raise SkillRegistryError("DUPLICATE_SKILL_ID")
            try:
                self._validate_composition(skill)
            except SkillRegistryError:
                raise
            self._skills[skill.skill_id] = skill
            return skill

    def get(self, skill_id: str) -> Optional[Skill]:
        with self._lock:
            return self._skills.get(str(skill_id).strip().lower())

    def require(self, skill_id: str) -> Skill:
        skill = self.get(skill_id)
        if skill is None:
            raise SkillRegistryError("SKILL_NOT_FOUND")
        return skill

    def list(self, enabled_only: bool = False) -> tuple[Skill, ...]:
        with self._lock:
            skills = tuple(self._skills.values())
        return tuple(skill for skill in skills if skill.enabled or not enabled_only)

    def exists(self, skill_id: str) -> bool:
        return self.get(skill_id) is not None

    def remove(self, skill_id: str) -> bool:
        with self._lock:
            return self._skills.pop(str(skill_id).strip().lower(), None) is not None

    def disable(self, skill_id: str) -> Skill:
        return self._set_enabled(skill_id, False)

    def enable(self, skill_id: str) -> Skill:
        return self._set_enabled(skill_id, True)

    def search(
        self,
        tags: Iterable[str] | None = None,
        category: SkillCategory | str | None = None,
        query: str | None = None,
    ) -> tuple[Skill, ...]:
        wanted_tags = {str(tag).casefold() for tag in (tags or ())}
        wanted_category = category.value if isinstance(category, SkillCategory) else str(category).upper() if category else None
        lowered_query = query.casefold() if query else None
        results: list[Skill] = []
        for skill in self.list():
            if wanted_tags and not wanted_tags.issubset({tag.casefold() for tag in skill.tags}):
                continue
            if wanted_category and skill.category.value != wanted_category:
                continue
            haystack = f"{skill.skill_id} {skill.name} {skill.description}".casefold()
            if lowered_query and lowered_query not in haystack:
                continue
            results.append(skill)
        return tuple(results)

    def validate_all(self) -> dict[str, str]:
        """Disable invalid definitions without preventing startup."""
        failures: dict[str, str] = {}
        for skill in self.list():
            try:
                self._validate_composition(skill, require_registered=False)
            except SkillRegistryError as exc:
                self.disable(skill.skill_id)
                failures[skill.skill_id] = str(exc)
        return failures

    def descriptions(self) -> list[dict[str, object]]:
        return [
            {
                "skill_id": skill.skill_id,
                "description": skill.description,
                "category": skill.category.value,
            }
            for skill in self.list(enabled_only=True)
        ]

    def _validate_composition(self, skill: Skill, require_registered: bool = True) -> None:
        if not skill.composition:
            return
        if skill.skill_id in {step.skill_id for step in skill.composition}:
            raise SkillRegistryError("COMPOSITION_CYCLE")
        seen: set[str] = set()

        def visit(current: Skill, trail: tuple[str, ...]) -> None:
            if current.skill_id in trail:
                raise SkillRegistryError("COMPOSITION_CYCLE")
            for step in current.composition:
                child = self.get(step.skill_id)
                if child is None:
                    if require_registered:
                        raise SkillRegistryError("COMPOSITION_DEPENDENCY_MISSING")
                    continue
                if child.skill_id in seen and child.skill_id in trail:
                    raise SkillRegistryError("COMPOSITION_CYCLE")
                visit(child, trail + (current.skill_id,))
                seen.add(child.skill_id)

        visit(skill, ())

    def _set_enabled(self, skill_id: str, enabled: bool) -> Skill:
        skill = self.require(skill_id)
        replacement = _replace_skill(skill, enabled=enabled)
        with self._lock:
            self._skills[replacement.skill_id] = replacement
        return replacement


def _replace_skill(skill: Skill, **changes: object) -> Skill:
    values = {
        field_name: getattr(skill, field_name)
        for field_name in skill.__dataclass_fields__  # type: ignore[attr-defined]
    }
    values.update(changes)
    return Skill(**values)


def create_default_skill_library() -> SkillLibrary:
    """Build the stable, source-controlled Phase 09 library."""

    from .builtins import build_builtin_skills

    library = SkillLibrary()
    for skill in build_builtin_skills():
        library.register(skill)
    failures = library.validate_all()
    if failures:
        raise SkillRegistryError(",".join(sorted(failures)))
    return library
