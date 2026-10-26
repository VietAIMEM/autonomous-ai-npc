"""Normalized world state construction from safe Unity snapshots."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

from .models import TargetDefinition, WorldState


def _position(value: Any, fallback: tuple[float, float, float]) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return fallback
    try:
        numbers = tuple(float(item) for item in value[:3])
    except (TypeError, ValueError):
        return fallback
    return (numbers[0], numbers[1], numbers[2] if len(numbers) > 2 else fallback[2])


def _optional_position(value: Any) -> tuple[float, float, float] | None:
    """PHASE 25: a reported transform, or ``None`` when Unity did not send one."""
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    try:
        numbers = tuple(float(item) for item in value[:3])
    except (TypeError, ValueError):
        return None
    if any(number != number or number in (float("inf"), float("-inf")) for number in numbers):
        return None
    return (numbers[0], numbers[1], numbers[2])


def _string(value: Any, fallback: str) -> str:
    return value if isinstance(value, str) and value else fallback


def _boolean(value: Any, fallback: bool = False) -> bool:
    return value if isinstance(value, bool) else fallback


def _optional_boolean(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _number(value: Any, fallback: float = 0.0) -> float:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else fallback


def _optional_number(value: Any) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _optional_string(value: Any) -> str | None:
    """PHASE 26B: an optional reported string; blanks mean "not reported"."""
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def build_world_state(
    snapshot: Mapping[str, Any],
    timestamp: datetime | None = None,
    *,
    viewer_count_floor: int = 0,
) -> WorldState:
    character = snapshot.get("character", {})
    phone = snapshot.get("phone", {})
    stream = snapshot.get("stream", {})
    raw_targets = snapshot.get("targets", {})

    targets: dict[str, TargetDefinition] = {}
    if isinstance(raw_targets, dict):
        for name, value in raw_targets.items():
            if not isinstance(name, str) or not isinstance(value, Mapping):
                continue
            position = _position(value.get("position"), (0.0, 0.0, 0.0))
            targets[name.upper()] = TargetDefinition(
                name=name.upper(),
                position=position,
                navigation_target=_boolean(value.get("navigation_target"), True),
                look_at_target=_boolean(value.get("look_at_target"), True),
                reachable=_optional_boolean(value.get("reachable")),
            )

    camera_fov = _optional_number(stream.get("camera_field_of_view"))
    if camera_fov is None:
        camera_fov = _optional_number(stream.get("field_of_view"))
    # PHASE 25: Unity serializes every float field, so a missing camera reports
    # a zero field of view. Only a positive value is a real reported FOV.
    if camera_fov is not None and camera_fov <= 0:
        camera_fov = None

    return WorldState(
        timestamp=timestamp or datetime.now(timezone.utc),
        character_position=_position(character.get("position"), (0.0, 0.0, 0.0)),
        character_rotation_y=_number(character.get("rotation_y")),
        character_posture=_string(character.get("posture"), "standing"),
        current_action=_string(character.get("current_action"), "idle"),
        phone_state=_string(phone.get("state"), "World"),
        phone_position=_position(phone.get("position"), (0.0, 0.0, 0.0)),
        phone_holder=_string(phone.get("holder"), None) if phone.get("holder") else None,
        placement_surface=_string(phone.get("placement_surface"), None) if phone.get("placement_surface") else None,
        stream_active=_boolean(stream.get("is_streaming")),
        stream_camera_available=_boolean(stream.get("camera_available")),
        stream_camera_active=_boolean(stream.get("camera_active")),
        stream_duration_seconds=_number(stream.get("duration_seconds")),
        viewer_count=max(
            int(_number(stream.get("viewer_count"))),
            max(0, int(viewer_count_floor)),
        ),
        targets=targets,
        character_forward=_optional_position(character.get("forward")),
        camera_position=_optional_position(stream.get("camera_position")),
        camera_rotation=_optional_position(stream.get("camera_rotation")),
        camera_forward=_optional_position(stream.get("camera_forward")),
        camera_fov=camera_fov,
        # PHASE 26B: Unity gesture observation. A legacy snapshot that never
        # reported these stays "unknown" rather than inventing evidence.
        last_gesture=_optional_string(character.get("last_gesture")),
        last_gesture_at=_optional_string(character.get("last_gesture_at")),
        gesture_active=_optional_boolean(character.get("gesture_active")),
        # PHASE 27B: keep "Unity did not report it" separate from an explicit
        # inactive verdict, so autonomy never reads missing data as an answer.
        stream_state_known=_optional_boolean(stream.get("is_streaming")),
        camera_state_known=_optional_boolean(stream.get("camera_available")),
    )
