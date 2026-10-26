"""Typed models for the Phase 06 autonomous brain."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from ..chat.models import format_utc, utc_now


class BrainStatus(str, Enum):
    IDLE = "IDLE"
    THINKING = "THINKING"
    PLANNING = "PLANNING"
    EXECUTING = "EXECUTING"
    WAITING = "WAITING"
    REPLANNING = "REPLANNING"
    ERROR = "ERROR"


class GoalType(str, Enum):
    IDLE = "IDLE"
    EXPLORE_ROOM = "EXPLORE_ROOM"
    INTERACT_WITH_VIEWER = "INTERACT_WITH_VIEWER"
    RESPOND_TO_VIEWER = "RESPOND_TO_VIEWER"
    MAINTAIN_STREAM = "MAINTAIN_STREAM"
    IMPROVE_CAMERA_POSITION = "IMPROVE_CAMERA_POSITION"
    PERFORM_ACTIVITY = "PERFORM_ACTIVITY"
    REST = "REST"


class GoalStatus(str, Enum):
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"
    FAILED = "FAILED"


class ActionType(str, Enum):
    MOVE_TO = "MOVE_TO"
    STOP = "STOP"
    TURN_TO = "TURN_TO"
    LOOK_AT = "LOOK_AT"
    SPEAK = "SPEAK"
    WAVE = "WAVE"
    DANCE = "DANCE"
    RUN = "RUN"
    PICK_UP_PHONE = "PICK_UP_PHONE"
    PLACE_PHONE = "PLACE_PHONE"
    START_STREAM = "START_STREAM"
    STOP_STREAM = "STOP_STREAM"
    SET_CAMERA = "SET_CAMERA"


class ExecutionStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    REJECTED = "REJECTED"


PERSONALITY_TRAITS: tuple[str, ...] = (
    "warmth",
    "playfulness",
    "curiosity",
    "energy",
    "talkativeness",
    "formality",
    "calmness",
    "sociability",
)


def _unit_value(value: Any, fallback: float) -> float:
    """Clamp one personality trait into the bounded 0..1 range."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = fallback
    if number != number:  # NaN never becomes a stored trait.
        number = fallback
    return max(0.0, min(1.0, number))


@dataclass(frozen=True)
class PersonalityProfile:
    """Stable, bounded personality configuration.

    PHASE 19: the profile is configuration, not runtime drift. Every trait is
    clamped to 0..1 on construction and the values never change while the
    process runs; only an explicit reload from persisted configuration does.
    """

    name: str = "Lacrimosa"
    identity: str = "autonomous embodied virtual character"
    calmness: float = 0.9
    curiosity: float = 0.8
    playfulness: float = 0.6
    sociability: float = 0.7
    # PHASE 19 traits are appended last so every existing positional
    # construction (Phase 06-18 tests and callers) keeps its meaning.
    warmth: float = 0.8
    energy: float = 0.5
    talkativeness: float = 0.6
    formality: float = 0.25

    def __post_init__(self) -> None:
        defaults = {item.name: item.default for item in fields(self)}
        for trait in PERSONALITY_TRAITS:
            fallback = float(defaults.get(trait, 0.5))
            object.__setattr__(self, trait, _unit_value(getattr(self, trait), fallback))

    @property
    def traits(self) -> dict[str, float]:
        return {trait: getattr(self, trait) for trait in PERSONALITY_TRAITS}

    @property
    def summary(self) -> str:
        """Short human-readable descriptor derived from the bounded traits."""
        words = []
        if self.warmth >= 0.6:
            words.append("warm")
        if self.calmness >= 0.7:
            words.append("calm")
        if self.playfulness >= 0.6:
            words.append("playful")
        if self.curiosity >= 0.7:
            words.append("curious")
        if self.talkativeness >= 0.6:
            words.append("talkative")
        if self.energy >= 0.7:
            words.append("energetic")
        if self.formality >= 0.5:
            words.append("formal")
        if self.sociability >= 0.6:
            words.append("social")
        return ", ".join(words) or "reserved"

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "identity": self.identity,
            "traits": self.traits,
            "summary": self.summary,
        }


@dataclass(frozen=True)
class TargetDefinition:
    name: str
    position: tuple[float, float, float]
    navigation_target: bool = True
    look_at_target: bool = True
    # PHASE 25: tri-state navmesh verdict reported by Unity. ``None`` means the
    # snapshot did not say, so existing callers keep the navigable-flag rule.
    reachable: Optional[bool] = None


@dataclass(frozen=True)
class WorldState:
    timestamp: datetime
    character_position: tuple[float, float, float]
    character_rotation_y: float
    character_posture: str
    current_action: str
    phone_state: str
    phone_position: tuple[float, float, float]
    phone_holder: Optional[str]
    placement_surface: Optional[str]
    stream_active: bool
    stream_camera_available: bool
    stream_camera_active: bool
    stream_duration_seconds: float
    viewer_count: int
    targets: dict[str, TargetDefinition] = field(default_factory=dict)
    # PHASE 25 known transforms. Appended after ``targets`` so every existing
    # positional construction keeps its meaning; all default to "not reported".
    character_forward: Optional[tuple[float, float, float]] = None
    camera_position: Optional[tuple[float, float, float]] = None
    camera_rotation: Optional[tuple[float, float, float]] = None
    camera_forward: Optional[tuple[float, float, float]] = None
    camera_fov: Optional[float] = None
    # PHASE 26B observed gesture evidence reported by Unity. ``None`` means the
    # snapshot did not say (a legacy or mocked world keeps its prior shape).
    last_gesture: Optional[str] = None
    last_gesture_at: Optional[str] = None
    gesture_active: Optional[bool] = None
    # PHASE 27B: whether the snapshot actually reported the streaming flag and
    # the camera verdict. ``None`` means "not reported", so missing data is
    # never read as "the stream/camera is off"; the boolean fields above keep
    # their historical defaulting for every existing caller.
    stream_state_known: Optional[bool] = None
    camera_state_known: Optional[bool] = None

    def target(self, name: str) -> Optional[TargetDefinition]:
        return self.targets.get(name.upper())

    def to_dict(self) -> dict[str, object]:
        character: dict[str, object] = {
            "position": list(self.character_position),
            "rotation_y": round(self.character_rotation_y, 2),
            "posture": self.character_posture,
            "current_action": self.current_action,
        }
        if self.character_forward is not None:
            character["forward"] = [round(float(value), 3) for value in self.character_forward]
        # PHASE 26B: only reported gesture evidence is echoed, so every legacy
        # snapshot and prompt keeps the exact shape it had before.
        if self.last_gesture is not None:
            character["last_gesture"] = self.last_gesture
        if self.last_gesture_at is not None:
            character["last_gesture_at"] = self.last_gesture_at
        if self.gesture_active is not None:
            character["gesture_active"] = self.gesture_active
        stream: dict[str, object] = {
            "is_streaming": self.stream_active,
            "camera_available": self.stream_camera_available,
            "camera_active": self.stream_camera_active,
            "duration_seconds": round(self.stream_duration_seconds, 3),
            "viewer_count": self.viewer_count,
        }
        # Only reported transforms are echoed, so a legacy snapshot keeps the
        # exact prompt/diagnostic shape it had before PHASE 25.
        if self.camera_position is not None:
            stream["camera_position"] = [round(float(value), 3) for value in self.camera_position]
        if self.camera_rotation is not None:
            stream["camera_rotation"] = [round(float(value), 3) for value in self.camera_rotation]
        if self.camera_forward is not None:
            stream["camera_forward"] = [round(float(value), 3) for value in self.camera_forward]
        if self.camera_fov is not None:
            stream["camera_field_of_view"] = round(float(self.camera_fov), 3)
        # PHASE 27B: only reported lifecycle evidence is echoed, so a legacy
        # snapshot keeps the exact shape it had before.
        if self.stream_state_known is not None:
            stream["stream_state_known"] = self.stream_state_known
        if self.camera_state_known is not None:
            stream["camera_state_known"] = self.camera_state_known
        return {
            "timestamp": format_utc(self.timestamp),
            "character": character,
            "phone": {
                "state": self.phone_state,
                "position": list(self.phone_position),
                "holder": self.phone_holder,
                "placement_surface": self.placement_surface,
            },
            "stream": stream,
            "targets": {
                name: _target_dict(target)
                for name, target in self.targets.items()
            },
        }


def _target_dict(target: TargetDefinition) -> dict[str, object]:
    payload: dict[str, object] = {
        "position": list(target.position),
        "navigation_target": target.navigation_target,
        "look_at_target": target.look_at_target,
    }
    if target.reachable is not None:
        payload["reachable"] = target.reachable
    return payload


@dataclass
class Goal:
    goal_id: str
    goal_type: GoalType
    priority: float
    reason: str
    created_at: datetime = field(default_factory=utc_now)
    status: GoalStatus = GoalStatus.PENDING
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "goal_id": self.goal_id,
            "goal_type": self.goal_type.value,
            "priority": round(self.priority, 3),
            "status": self.status.value,
            "reason": self.reason,
            "created_at": format_utc(self.created_at),
            "context": self.context,
        }


@dataclass(frozen=True)
class PlanStep:
    action: ActionType
    target: Optional[str] = None
    text: Optional[str] = None
    position: Optional[tuple[float, float, float]] = None

    def to_dict(self) -> dict[str, object]:
        return {
            "action": self.action.value,
            "target": self.target,
            "text": self.text,
            "position": list(self.position) if self.position is not None else None,
        }


@dataclass(frozen=True)
class Plan:
    plan_id: str
    goal_id: str
    reason: str
    steps: tuple[PlanStep, ...]
    created_at: datetime = field(default_factory=utc_now)
    version: int = 1

    def to_dict(self) -> dict[str, object]:
        return {
            "plan_id": self.plan_id,
            "goal_id": self.goal_id,
            "reason": self.reason,
            "version": self.version,
            "steps": [step.to_dict() for step in self.steps],
            "created_at": format_utc(self.created_at),
        }


@dataclass(frozen=True)
class PlanValidation:
    valid: bool
    reason: str = "valid"

    def to_dict(self) -> dict[str, object]:
        return {"valid": self.valid, "reason": self.reason}


@dataclass(frozen=True)
class ActionRequest:
    action: ActionType
    target: Optional[str] = None
    text: Optional[str] = None
    position: Optional[tuple[float, float, float]] = None
    speech_request_id: Optional[str] = None
    #: PHASE 34 correlation id for exactly one action dispatch. Unity echoes it
    #: with the action result so a stale or duplicate result can never complete
    #: a later action of the same type. ``None`` means the request predates the
    #: field (legacy), in which case result matching stays action-type only.
    dispatch_id: Optional[str] = None

    def to_dict(self) -> dict[str, object]:
        return {
            "action": self.action.value,
            "target": self.target,
            "text": self.text,
            "position": list(self.position) if self.position is not None else None,
            "speech_request_id": self.speech_request_id,
            "dispatch_id": self.dispatch_id,
        }


@dataclass(frozen=True)
class ActionResult:
    action: ActionType
    status: ExecutionStatus
    reason: str = "accepted"
    reached: Optional[bool] = None
    viewer_response: Optional[str] = None
    #: PHASE 34 the dispatch id Unity echoed for the action that produced this
    #: result. ``None`` when the originating request carried no id or the value
    #: was missing/malformed, in which case it can only complete a pending
    #: action that itself has no id.
    dispatch_id: Optional[str] = None

    def to_dict(self) -> dict[str, object]:
        return {
            "action": self.action.value,
            "status": self.status.value,
            "reason": self.reason,
            "reached": self.reached,
            "viewer_response": self.viewer_response,
            "dispatch_id": self.dispatch_id,
        }


@dataclass(frozen=True)
class Decision:
    decision_id: str
    goal: Goal
    plan: Plan
    created_at: datetime = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, object]:
        return {
            "decision_id": self.decision_id,
            "goal": self.goal.to_dict(),
            "plan": self.plan.to_dict(),
            "created_at": format_utc(self.created_at),
        }
