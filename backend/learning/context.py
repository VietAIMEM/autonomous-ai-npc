"""Deterministic, bounded context keys for skill associations."""

from __future__ import annotations

import re
from typing import Any


CONTEXT_SEPARATOR = "|"
_TOKEN = re.compile(r"[^A-Z0-9]+")


def _token(value: Any) -> str:
    if not value:
        return ""
    return _TOKEN.sub("_", str(value).strip().upper()).strip("_")[:60]


def build_context_key(
    *,
    intent: str | None = None,
    target: str | None = None,
    topic: str | None = None,
    goal: str | None = None,
    character_state: str | None = None,
    stream_state: str | None = None,
    location: str | None = None,
    viewer_relationship: str | None = None,
) -> str:
    segments = [
        _token(intent),
        _token(target),
        _token(goal),
        _token(topic),
        _token(character_state),
        _token(stream_state),
        _token(location),
        _token(viewer_relationship),
    ]
    return CONTEXT_SEPARATOR.join(value for value in segments if value)[:720]


def context_from_world(goal: Any, chat: Any, world: Any, viewer_memory: Any = None) -> str:
    intent = getattr(chat, "intent", None)
    target = getattr(chat, "target", None)
    relationship = None
    if viewer_memory is not None:
        relationship = f"{int(viewer_memory.relationship_score * 2) / 2:.1f}"
    return build_context_key(
        intent=intent.value if intent is not None else None,
        target=target,
        goal=goal.goal_type.value if goal is not None else None,
        character_state=getattr(world, "character_posture", None),
        stream_state="ACTIVE" if getattr(world, "stream_active", False) else "INACTIVE",
        location=getattr(world, "placement_surface", None),
        viewer_relationship=relationship,
    )
