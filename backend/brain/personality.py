"""Static Phase 06 personality configuration, extended by PHASE 19.

The profile is loaded from configuration (defaults plus explicit overrides)
and persisted through the existing memory store so it is identical after a
restart. Nothing in the runtime is allowed to drift the traits on its own.
"""

from __future__ import annotations

from dataclasses import fields, replace
from typing import Any, Mapping

from .models import PERSONALITY_TRAITS, PersonalityProfile, _unit_value


DEFAULT_PERSONALITY = PersonalityProfile()

#: Metadata key inside the existing memory database; no new storage is created.
PERSONALITY_METADATA_KEY = "personality_profile"


def load_personality(overrides: dict[str, float] | None = None) -> PersonalityProfile:
    if not overrides:
        return DEFAULT_PERSONALITY

    defaults = {item.name: item.default for item in fields(PersonalityProfile)}
    clean: dict[str, float] = {}
    for key, value in overrides.items():
        if key not in PERSONALITY_TRAITS:
            continue
        clean[key] = _unit_value(value, float(defaults.get(key, 0.5)))
    return replace(DEFAULT_PERSONALITY, **clean)


def personality_traits(profile: PersonalityProfile) -> dict[str, float]:
    """Bounded trait payload suitable for persistence or diagnostics."""
    return {trait: round(float(getattr(profile, trait)), 4) for trait in PERSONALITY_TRAITS}


def personality_from_payload(payload: Any) -> PersonalityProfile | None:
    """Rebuild a profile from stored data; ``None`` when nothing usable exists.

    Corrupt or partial payloads never raise: unknown keys are ignored and the
    remaining traits fall back to the configured defaults.
    """
    if not isinstance(payload, Mapping):
        return None
    traits = payload.get("traits") if isinstance(payload.get("traits"), Mapping) else payload
    if not isinstance(traits, Mapping):
        return None
    overrides = {
        key: value
        for key, value in traits.items()
        if key in PERSONALITY_TRAITS and isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    if not overrides:
        return None
    return load_personality(overrides)


def personality_descriptor(profile: PersonalityProfile) -> dict[str, float]:
    """Interpretable speech/behavior hints derived from the stable traits.

    The values are advisory style weights only; they never gate or execute an
    action, and they are recomputed from the profile instead of stored.
    """
    return {
        "warmth": round(profile.warmth, 4),
        "playfulness": round(profile.playfulness, 4),
        "formality": round(profile.formality, 4),
        "talkativeness": round(profile.talkativeness, 4),
        "energy": round(profile.energy, 4),
    }


def personality_speech_style(profile: PersonalityProfile) -> str:
    """One-line style hint appended to the generated speech prompt."""
    parts: list[str] = []
    if profile.warmth >= 0.6:
        parts.append("friendly and warm wording")
    if profile.playfulness >= 0.6:
        parts.append("a light playful touch")
    if profile.formality >= 0.5:
        parts.append("restrained, formal wording")
    if profile.talkativeness < 0.4:
        parts.append("keep replies short")
    elif profile.talkativeness >= 0.75:
        parts.append("a slightly longer, conversational reply")
    if profile.energy >= 0.7:
        parts.append("upbeat energy")
    return "; ".join(parts) or "neutral, concise wording"
