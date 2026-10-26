"""Whitelist-built public state for the viewer page.

Nothing here passes a runtime payload through unchanged. Every field is
selected and renamed on purpose so the browser can never receive credentials,
prompts, memory contents, relationship internals, skill knowledge, database
state, or internal errors. Absent/optional pieces degrade to an explicit
public status instead of leaking a reason string.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Optional

from ..chat.models import utc_now

#: Public activity vocabulary. Derived from real speech/skill/world state.
ACTIVITY_SPEAKING = "Speaking"
ACTIVITY_MOVING = "Moving"
ACTIVITY_DANCING = "Dancing"
ACTIVITY_LOOKING = "Looking"
ACTIVITY_WAVING = "Waving"
ACTIVITY_ACTIVE = "Active"
ACTIVITY_LIVE = "Live"
ACTIVITY_IDLE = "Idle"

_SPEECH_ACTIVE = {"SPEECH_REQUESTED", "SYNTHESIZING", "READY", "PLAYING"}

_ACTION_LABELS = {
    "SPEAK": ACTIVITY_SPEAKING,
    "MOVE_TO": ACTIVITY_MOVING,
    "RUN": ACTIVITY_MOVING,
    "WALK_TO": ACTIVITY_MOVING,
    "TURN_TO": ACTIVITY_LOOKING,
    "LOOK_AT": ACTIVITY_LOOKING,
    "SET_CAMERA": ACTIVITY_LOOKING,
    "SHOW_TABLE": ACTIVITY_LOOKING,
    "DANCE": ACTIVITY_DANCING,
    "SETUP_CAMERA_AND_DANCE": ACTIVITY_DANCING,
    "WAVE": ACTIVITY_WAVING,
    "PICK_UP_PHONE": ACTIVITY_ACTIVE,
    "PLACE_PHONE": ACTIVITY_ACTIVE,
    "START_STREAM": ACTIVITY_ACTIVE,
    "STOP": ACTIVITY_IDLE,
}

_WORLD_ACTION_LABELS = {
    "turning": ACTIVITY_MOVING,
    "moving": ACTIVITY_MOVING,
    "walking": ACTIVITY_MOVING,
    "running": ACTIVITY_MOVING,
}


def _iso(value: object) -> Optional[str]:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    if isinstance(value, str) and value:
        return value
    return None


def activity_for_action(action: object) -> Optional[str]:
    if not isinstance(action, str):
        return None
    return _ACTION_LABELS.get(action.strip().upper()) or _ACTION_LABELS.get(action.strip().lower())


def derive_activity(
    brain_service: Any,
    world: Any,
    *,
    stream_live: bool,
) -> dict[str, object]:
    """One public activity label from the real, already-existing state."""
    speech_state = ""
    try:
        summary = brain_service.get_speech_summary()
        if isinstance(summary, Mapping):
            speech_state = str(summary.get("state") or "").upper()
    except Exception:
        speech_state = ""
    if speech_state in _SPEECH_ACTIVE:
        return {"label": ACTIVITY_SPEAKING, "at": _iso(utc_now())}

    try:
        runtime = brain_service.get_skills_runtime()
        active = runtime.get("runtime", {}).get("active") if isinstance(runtime, Mapping) else None
        if isinstance(active, Mapping):
            label = activity_for_action(active.get("skill_id"))
            if label:
                return {"label": label, "at": _iso(active.get("started_at") or utc_now())}
    except Exception:
        pass

    try:
        pending = brain_service.brain.state.pending_action
    except Exception:
        pending = None
    if pending is not None:
        label = activity_for_action(getattr(getattr(pending, "action", None), "value", None))
        if label:
            return {"label": label, "at": _iso(utc_now())}

    current = getattr(world, "current_action", None)
    if isinstance(current, str):
        label = _WORLD_ACTION_LABELS.get(current.strip().lower())
        if label:
            return {"label": label, "at": _iso(utc_now())}

    if stream_live:
        return {"label": ACTIVITY_LIVE, "at": _iso(utc_now())}
    return {"label": ACTIVITY_IDLE, "at": _iso(utc_now())}


def public_viewer_count(social: Any, present_viewers: set[str], now: datetime) -> int:
    """Distinct viewers the real runtime has seen recently, plus live page sessions."""
    viewers = set(present_viewers)
    try:
        context = social.context
        timeout = float(getattr(social.config, "conversation_timeout_seconds", 180.0))
        for viewer_id, stamp in getattr(context, "viewer_last_seen", {}).items():
            if isinstance(stamp, datetime) and (now - stamp).total_seconds() <= timeout:
                viewers.add(str(viewer_id))
    except Exception:
        pass
    return len(viewers)


def _stream_public(
    streaming: Any,
    relay_snapshot: Mapping[str, object],
) -> dict[str, object]:
    transport = "STOPPED"
    resolution: Optional[str] = None
    fps: Optional[float] = None
    since: Optional[str] = None
    if streaming is not None:
        try:
            state = streaming.get_state()
        except Exception:
            state = {}
        if isinstance(state, Mapping):
            transport = str(state.get("state") or "STOPPED").upper()
            config = state.get("config")
            if isinstance(config, Mapping):
                resolution = config.get("resolution") if isinstance(config.get("resolution"), str) else None
                fps_value = config.get("fps")
                fps = float(fps_value) if isinstance(fps_value, (int, float)) else None
            since = _iso(state.get("started_at"))

    relay_state = str(relay_snapshot.get("state") or "UNAVAILABLE").upper()
    playback_ready = bool(relay_snapshot.get("playlist_ready"))
    live = transport == "LIVE"

    if live and playback_ready:
        connection = "LIVE"
    elif live:
        connection = "DEGRADED"
    elif transport == "CONNECTING":
        connection = "CONNECTING"
    elif transport == "RECONNECTING":
        connection = "RECONNECTING"
    else:
        connection = "OFFLINE"

    return {
        "connection": connection,
        "live": live,
        "transport": transport,
        "playback": {
            "type": "hls",
            "url": relay_snapshot.get("playlist_url"),
            "ready": playback_ready,
            "relay": relay_state,
        },
        "resolution": resolution,
        "fps": fps,
        "since": since,
    }


def build_viewer_state(
    brain_service: Any,
    *,
    streaming: Any,
    relay_snapshot: Mapping[str, object],
    feed_snapshot: Mapping[str, object],
    viewer_count: int,
    viewer_id: str,
    chat_busy: bool,
    queued_count: int,
) -> dict[str, object]:
    """Assemble the single public state document the page polls."""
    try:
        world = brain_service.brain.world
    except Exception:
        world = None
    stream = _stream_public(streaming, relay_snapshot)
    activity = derive_activity(brain_service, world, stream_live=bool(stream["live"]))
    return {
        "ok": True,
        "server_time": _iso(utc_now()),
        "stream": stream,
        "activity": activity,
        "viewers": {"count": max(0, int(viewer_count))},
        "chat": {
            "available": True,
            "busy": bool(chat_busy),
            "queued": max(0, int(queued_count)),
            "version": int(feed_snapshot.get("version") or 0),
            "messages": list(feed_snapshot.get("messages") or []),
        },
        "viewer": {"id": viewer_id},
    }
