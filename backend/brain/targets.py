"""Controlled mapping of semantic target names to normalized world targets."""

from __future__ import annotations

from .models import WorldState


class TargetResolver:
    def __init__(self, world: WorldState) -> None:
        self.world = world

    def resolve(self, requested: str | None) -> tuple[str | None, str]:
        if requested is None:
            return None, "target_not_required"

        key = requested.strip().upper().replace(" ", "_")
        aliases = {
            "CENTER": "CENTER",
            "CENTER_OF_ROOM": "CENTER",
            "ROOM_CENTER": "CENTER",
            "TABLE": "TABLE",
            "BENCH": "BENCH",
            "FLOOR": "FLOOR",
            "PHONE": "PHONE",
            "CAMERA": "CAMERA",
            # PHASE 25: viewers say "the camera" / "the stream camera"; both
            # normalize onto the one allowlisted CAMERA target.
            "STREAM_CAMERA": "CAMERA",
            "CAMERA_TARGET": "CAMERA",
            # Only valid when Unity itself computed a navmesh-sampled standing
            # point; an arbitrary text name can never become a transform here.
            "CAMERA_STAND": "CAMERA_STAND",
            "PHONE_STAND": "PHONE_STAND",
            "LACRIMOSA": "LACRIMOSA",
        }
        resolved = aliases.get(key)
        if resolved is None:
            return None, "TARGET_NOT_FOUND"
        if self.world.target(resolved) is None:
            return None, "TARGET_NOT_FOUND"
        return resolved, "OK"
