"""PHASE 25: deterministic spatial relationships derived from known transforms.

The Brain never guesses about space. Unity already knows where the character,
the phone and the stream camera are; this module turns those known transforms
into a small, bounded, serializable relationship that the goal evaluator, the
existing planner and the post-action verifier can all share.

Nothing here touches Unity, and nothing here is computer vision. Every value is
either a straight geometric derivation of a reported transform or a bounded
classification of one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

from .models import WorldState

# --------------------------------------------------------------------- states

#: Camera framing vocabulary shared by social attention, the goal evaluator,
#: the spatial planner and the post-action verifier. The first four values are
#: the ones PHASE 25 names explicitly; the rest describe why no measurement is
#: possible, so a missing transform can never read as a success.
FRAME_INSIDE = "INSIDE_FRAME"
FRAME_OUTSIDE = "OUTSIDE_FRAME"
FRAME_TOO_CLOSE = "TOO_CLOSE"
FRAME_TOO_FAR = "TOO_FAR"
FRAME_CAMERA_INACTIVE = "CAMERA_INACTIVE"
FRAME_STREAM_INACTIVE = "STREAM_INACTIVE"
FRAME_CAMERA_MOVES_WITH_PHONE = "CAMERA_MOVES_WITH_PHONE"
FRAME_UNKNOWN = "UNKNOWN"

#: Framing states that require moving, not only rotating. These are exactly the
#: measured "outside the camera frame" states: every other non-inside value
#: (inactive camera, missing transform) is unmeasurable.
FRAMING_NEEDS_POSITION = frozenset({FRAME_OUTSIDE, FRAME_TOO_CLOSE, FRAME_TOO_FAR})

#: Framing states that mean "the camera cannot be used right now".
FRAMING_UNAVAILABLE = frozenset({FRAME_CAMERA_INACTIVE, FRAME_STREAM_INACTIVE, FRAME_UNKNOWN})

#: PHASE 26A: the reported camera transform is real but no frames are being
#: produced. The placement half of a request (position and orientation against
#: that transform) stays plan-able and measurable; visibility itself does not,
#: so these states never read as "inside frame".
FRAMING_PLACEMENT_ONLY = frozenset({FRAME_CAMERA_INACTIVE, FRAME_STREAM_INACTIVE})

# ---------------------------------------------------------------- thresholds

#: Legacy Phase 08 distance rule, kept as the single source of truth.
DEFAULT_CAMERA_FRAMING_DISTANCE = 5.5
DEFAULT_CAMERA_TOO_CLOSE_DISTANCE = 1.2
#: Unity's ``fieldOfView`` is vertical; the horizontal half-angle needs an
#: aspect ratio and the project renders 16:9.
DEFAULT_CAMERA_ASPECT = 16.0 / 9.0
#: The character must land inside a slightly tighter cone than the raw frustum
#: edge, so a "verified" framing cannot sit on the exact border.
DEFAULT_FRAME_MARGIN = 0.9
ORIENTATION_TOLERANCE_DEGREES = 20.0
POSITION_TOLERANCE_METERS = 0.8
#: How far a reported character position may sit from the planned standing
#: point and still count as "arrived" after a MOVE_TO.
POSITION_ARRIVAL_TOLERANCE_METERS = 1.0
#: Comfortable standing band in front of the phone (the classic streaming
#: framing) used when a viewer asks to stand in front of the phone.
MIN_PHONE_STAND_DISTANCE = 1.2
MAX_PHONE_STAND_DISTANCE = 3.5

#: Desired states of a structured spatial request (PHASE 25 step 5).
DESIRED_FACING = "FACING"
DESIRED_IN_FRAME = "IN_FRAME"
DESIRED_CLOSER = "CLOSER"
DESIRED_NEAR = "NEAR"

#: Target names the Unity bridge computes; the allowlist here is the only way
#: a spatial plan can name them.
TARGET_CAMERA = "CAMERA"
TARGET_CAMERA_STAND = "CAMERA_STAND"
TARGET_PHONE = "PHONE"
TARGET_PHONE_STAND = "PHONE_STAND"
TARGET_CENTER = "CENTER"

#: Plans produced by :func:`plan_spatial_steps` carry this reason prefix so the
#: loop can verify them against real Unity state instead of trusting the
#: action result. Reading it is the only thing that marks a plan as spatial.
SPATIAL_PLAN_PREFIX = "spatial:"


def spatial_reason(detail: str) -> str:
    return f"{SPATIAL_PLAN_PREFIX}{detail}"


def is_spatial_plan(plan: Any) -> bool:
    reason = getattr(plan, "reason", None)
    return isinstance(reason, str) and reason.startswith(SPATIAL_PLAN_PREFIX)


def _vec3(value: Any) -> Optional[tuple[float, float, float]]:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    try:
        numbers = tuple(float(item) for item in value[:3])
    except (TypeError, ValueError):
        return None
    if any(number != number or math.isinf(number) for number in numbers):
        return None
    return (numbers[0], numbers[1], numbers[2])


def _sub(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _norm(vector: tuple[float, float, float]) -> Optional[tuple[float, float, float]]:
    length = math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)
    if length <= 1e-6:
        return None
    return (vector[0] / length, vector[1] / length, vector[2] / length)


def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _clamp_unit(value: float) -> float:
    return max(-1.0, min(1.0, value))


def _horizontal(vector: tuple[float, float, float]) -> Optional[tuple[float, float]]:
    projected = _norm((vector[0], 0.0, vector[2]))
    if projected is None:
        return None
    return (projected[0], projected[2])


def horizontal_distance(a: Any, b: Any) -> Optional[float]:
    first, second = _vec3(a), _vec3(b)
    if first is None or second is None:
        return None
    return math.sqrt((first[0] - second[0]) ** 2 + (first[2] - second[2]) ** 2)


def distance(a: Any, b: Any) -> Optional[float]:
    first, second = _vec3(a), _vec3(b)
    if first is None or second is None:
        return None
    delta = _sub(first, second)
    return math.sqrt(delta[0] ** 2 + delta[1] ** 2 + delta[2] ** 2)


def forward_from_yaw(yaw_degrees: float) -> tuple[float, float, float]:
    """Unity yaw (rotation about +Y) to a flattened forward vector."""
    radians = math.radians(float(yaw_degrees))
    return (math.sin(radians), 0.0, math.cos(radians))


def forward_from_euler(euler_degrees: Any) -> Optional[tuple[float, float, float]]:
    """Unity euler angles (ZXY order) to a forward vector.

    Roll cannot affect the forward axis, so only pitch and yaw are needed.
    """
    angles = _vec3(euler_degrees)
    if angles is None:
        return None
    pitch = math.radians(angles[0])
    yaw = math.radians(angles[1])
    return (
        math.sin(yaw) * math.cos(pitch),
        -math.sin(pitch),
        math.cos(yaw) * math.cos(pitch),
    )


def character_forward(world: WorldState) -> Optional[tuple[float, float, float]]:
    """Reported body forward, derived from yaw when Unity did not send it."""
    reported = _vec3(getattr(world, "character_forward", None))
    if reported is not None:
        return _norm(reported)
    if world.character_rotation_y is None:
        return None
    return forward_from_yaw(world.character_rotation_y)


def camera_position(world: WorldState) -> Optional[tuple[float, float, float]]:
    """The active stream camera position, with the CAMERA target as fallback."""
    reported = _vec3(getattr(world, "camera_position", None))
    if reported is not None:
        return reported
    target = world.target(TARGET_CAMERA)
    return target.position if target is not None else None


def camera_forward(world: WorldState) -> Optional[tuple[float, float, float]]:
    reported = _vec3(getattr(world, "camera_forward", None))
    if reported is not None:
        return _norm(reported)
    return forward_from_euler(getattr(world, "camera_rotation", None))


def has_camera_transform(world: WorldState | None) -> bool:
    """Whether Unity actually reported a camera transform in this snapshot.

    Legacy snapshots (and tests) carry no ``stream.*`` transform fields at all.
    A spatial plan may only be derived from real reported transforms, so this
    is the gate every deterministic spatial branch checks first.
    """
    if world is None:
        return False
    return (
        _vec3(getattr(world, "camera_position", None)) is not None
        or _vec3(getattr(world, "camera_forward", None)) is not None
        or _vec3(getattr(world, "camera_rotation", None)) is not None
        or getattr(world, "camera_fov", None) is not None
    )


def anchor_position(
    world: WorldState,
    target: str = TARGET_CAMERA,
) -> Optional[tuple[float, float, float]]:
    """The world position a spatial request is anchored to.

    ``CAMERA`` resolves to the reported stream-camera transform (with the
    allowlisted CAMERA target as fallback); ``PHONE`` resolves to the reported
    phone position, which exists even while the stream is off.
    """
    if _normalize_anchor(target) == TARGET_PHONE:
        reported = _vec3(world.phone_position)
        if reported is not None:
            return reported
        definition = world.target(TARGET_PHONE)
        return definition.position if definition is not None else None
    return camera_position(world)


def _normalize_anchor(target: Optional[str]) -> str:
    return TARGET_PHONE if (target or "").strip().upper() == TARGET_PHONE else TARGET_CAMERA


def request_anchor(target: Optional[str]) -> str:
    """The anchor a structured chat target refers to.

    Chat only ever carries the small intent vocabulary ("camera", "phone");
    anything that is not the phone is the stream camera, and both names still
    have to exist in the world for a plan to validate.
    """
    return TARGET_PHONE if (target or "").strip().casefold() == "phone" else TARGET_CAMERA


def _split_angles(
    forward: tuple[float, float, float],
    direction: tuple[float, float, float],
) -> tuple[Optional[float], Optional[float]]:
    """Horizontal (yaw) and signed vertical (pitch) deviation in degrees."""
    horizontal: Optional[float] = None
    forward_xz = _horizontal(forward)
    direction_xz = _horizontal(direction)
    if forward_xz is not None and direction_xz is not None:
        cosine = _clamp_unit(forward_xz[0] * direction_xz[0] + forward_xz[1] * direction_xz[1])
        cross = forward_xz[0] * direction_xz[1] - forward_xz[1] * direction_xz[0]
        horizontal = math.degrees(math.atan2(cross, cosine))
    vertical = math.degrees(
        math.asin(_clamp_unit(direction[1])) - math.asin(_clamp_unit(forward[1]))
    )
    return horizontal, vertical


@dataclass(frozen=True)
class SpatialRelationship:
    """Bounded derived relationship between Lacrimosa and the stream camera."""

    framing: str
    transform_based: bool
    character_position: Optional[tuple[float, float, float]]
    character_forward: Optional[tuple[float, float, float]]
    camera_position: Optional[tuple[float, float, float]]
    camera_forward: Optional[tuple[float, float, float]]
    camera_fov: Optional[float]
    distance_to_camera: Optional[float]
    horizontal_distance: Optional[float]
    direction_to_camera: Optional[tuple[float, float, float]]
    relative_angle_degrees: Optional[float]
    horizontal_angle_degrees: Optional[float]
    vertical_angle_degrees: Optional[float]
    orientation_error_degrees: Optional[float]
    inside_frame: Optional[bool]
    target: Optional[str] = TARGET_CAMERA
    reachable: Optional[bool] = None
    orientation_tolerance_degrees: float = ORIENTATION_TOLERANCE_DEGREES
    position_tolerance_meters: float = POSITION_TOLERANCE_METERS

    @property
    def needs_position_change(self) -> bool:
        return self.framing in FRAMING_NEEDS_POSITION

    @property
    def needs_orientation_change(self) -> bool:
        error = self.orientation_error_degrees
        return error is not None and error > self.orientation_tolerance_degrees

    @property
    def facing_camera(self) -> bool:
        error = self.orientation_error_degrees
        return error is not None and error <= self.orientation_tolerance_degrees

    @property
    def spatial_state(self) -> str:
        """One word for diagnostics: what physically remains to be done."""
        if self.needs_position_change and self.needs_orientation_change:
            return "MOVE_AND_TURN"
        if self.needs_position_change:
            return "MOVE"
        if self.needs_orientation_change:
            return "TURN"
        if self.framing == FRAME_INSIDE:
            return "ALREADY_SUITABLE"
        return self.framing

    def to_dict(self) -> dict[str, Any]:
        return {
            "spatial_state": self.spatial_state,
            "framing": self.framing,
            "inside_frame": self.inside_frame,
            "transform_based": self.transform_based,
            "target": self.target,
            "reachable": self.reachable,
            "character_position": _rounded(self.character_position),
            "character_forward": _rounded(self.character_forward),
            "camera_position": _rounded(self.camera_position),
            "camera_forward": _rounded(self.camera_forward),
            "camera_field_of_view": _round(self.camera_fov),
            "distance_to_camera": _round(self.distance_to_camera),
            "horizontal_distance": _round(self.horizontal_distance),
            "direction_to_camera": _rounded(self.direction_to_camera),
            "relative_angle_degrees": _round(self.relative_angle_degrees),
            "horizontal_angle_degrees": _round(self.horizontal_angle_degrees),
            "vertical_angle_degrees": _round(self.vertical_angle_degrees),
            "orientation_error_degrees": _round(self.orientation_error_degrees),
            "position_tolerance_meters": _round(self.position_tolerance_meters),
            "orientation_tolerance_degrees": _round(self.orientation_tolerance_degrees),
        }


def _round(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or math.isinf(number):
        return None
    return round(number, 3)


def _inside_frame_state(framing: str) -> Optional[bool]:
    """Tri-state framing: measured inside, measured outside, or unmeasurable."""
    if framing == FRAME_INSIDE:
        return True
    if framing in FRAMING_NEEDS_POSITION:
        return False
    return None


def _rounded(value: Optional[tuple[float, float, float]]) -> Optional[list[float]]:
    if value is None:
        return None
    return [round(float(item), 3) for item in value]


def spatial_observation(relationship: SpatialRelationship) -> dict[str, Any]:
    """Bounded observed-state block shared by verification and diagnostics.

    Only derived, already-rounded values are exposed; nothing here is a
    prompt, a credential, or a raw Unity payload.
    """
    return {
        "spatial_state": relationship.spatial_state,
        "framing": relationship.framing,
        "transform_based": relationship.transform_based,
        "target": relationship.target,
        "character_position": _rounded(relationship.character_position),
        "character_forward": _rounded(relationship.character_forward),
        "camera_position": _rounded(relationship.camera_position),
        "camera_forward": _rounded(relationship.camera_forward),
        "camera_field_of_view": _round(relationship.camera_fov),
        "distance": _round(relationship.distance_to_camera),
        "horizontal_distance": _round(relationship.horizontal_distance),
        "relative_angle": _round(relationship.relative_angle_degrees),
        "horizontal_angle": _round(relationship.horizontal_angle_degrees),
        "vertical_angle": _round(relationship.vertical_angle_degrees),
        "orientation_error": _round(relationship.orientation_error_degrees),
        "inside_frame": relationship.inside_frame,
        "reachable": relationship.reachable,
    }


def spatial_relationship(
    world: WorldState | None,
    *,
    target: str = TARGET_CAMERA,
    require_stream: bool | None = None,
    max_distance: float = DEFAULT_CAMERA_FRAMING_DISTANCE,
    too_close_distance: float = DEFAULT_CAMERA_TOO_CLOSE_DISTANCE,
    aspect: float = DEFAULT_CAMERA_ASPECT,
    frame_margin: float = DEFAULT_FRAME_MARGIN,
) -> SpatialRelationship:
    """Derive the full relationship from the reported transforms.

    Every branch is deterministic and bounded: a missing transform yields a
    state that says why, never a fabricated "inside frame".

    ``target=PHONE`` anchors the geometry on the phone instead of the stream
    camera and does not require an active stream: ``stand in front of the
    phone`` must stay plan-able while the stream is off, because Unity reports
    the phone transform and the navmesh-sampled PHONE_STAND either way.
    """
    if world is None:
        return _empty(FRAME_UNKNOWN, target=_normalize_anchor(target))

    anchor = _normalize_anchor(target)
    if require_stream is None:
        require_stream = anchor == TARGET_CAMERA
    character = _vec3(world.character_position)
    body_forward = character_forward(world)
    reported_camera = _vec3(getattr(world, "camera_position", None))
    reported_forward = _vec3(getattr(world, "camera_forward", None))
    reported_fov = getattr(world, "camera_fov", None)
    camera_fov = float(reported_fov) if isinstance(reported_fov, (int, float)) and not isinstance(reported_fov, bool) and reported_fov > 0 else None
    transform_based = (
        reported_camera is not None
        or reported_forward is not None
        or camera_fov is not None
        or getattr(world, "camera_rotation", None) is not None
    )
    position = anchor_position(world, anchor)
    forward = camera_forward(world)

    # PHASE 26A: an inactive stream or camera still reports a real transform.
    # The framing label below keeps saying why visibility cannot be judged, but
    # the geometry is still derived from the reported transform so a placement
    # request ("stand in front of the camera.") stays plan-able and measurable
    # while the stream is off.
    inactive_framing: Optional[str] = None
    if require_stream and not world.stream_active:
        inactive_framing = FRAME_STREAM_INACTIVE
    elif require_stream and (
        not world.stream_camera_available or not world.stream_camera_active
    ):
        inactive_framing = FRAME_CAMERA_INACTIVE

    if character is None or position is None:
        return _empty(
            inactive_framing or FRAME_UNKNOWN,
            character=character,
            body_forward=body_forward,
            camera=position,
            camera_forward=forward,
            camera_fov=camera_fov,
            transform_based=transform_based,
            target=anchor,
        )

    full_distance = distance(character, position)
    flat_distance = horizontal_distance(character, position)
    to_camera = _norm(_sub(position, character))
    to_character = _norm(_sub(character, position))

    relative_angle: Optional[float] = None
    horizontal_angle: Optional[float] = None
    vertical_angle: Optional[float] = None
    if forward is not None and to_character is not None:
        relative_angle = math.degrees(math.acos(_clamp_unit(_dot(forward, to_character))))
        horizontal_angle, vertical_angle = _split_angles(forward, to_character)

    orientation_error: Optional[float] = None
    body_xz = _horizontal(body_forward) if body_forward is not None else None
    camera_xz = _horizontal(to_camera) if to_camera is not None else None
    if body_xz is not None and camera_xz is not None:
        cosine = _clamp_unit(body_xz[0] * camera_xz[0] + body_xz[1] * camera_xz[1])
        orientation_error = math.degrees(math.acos(cosine))

    if anchor == TARGET_PHONE:
        framing = _phone_framing(world, character, flat_distance)
    elif not transform_based:
        # Legacy Phase 08 behaviour: a CAMERA target without a live camera
        # transform can only support the bounded distance rule.
        framing = FRAME_OUTSIDE if flat_distance is not None and flat_distance > max_distance else FRAME_INSIDE
    elif flat_distance is not None and flat_distance > max_distance:
        framing = FRAME_TOO_FAR
    elif flat_distance is not None and flat_distance < too_close_distance:
        framing = FRAME_TOO_CLOSE
    elif forward is not None and camera_fov is not None and horizontal_angle is not None and vertical_angle is not None:
        half_vertical = camera_fov / 2.0
        half_horizontal = math.degrees(math.atan(math.tan(math.radians(half_vertical)) * max(0.1, aspect)))
        inside = (
            abs(horizontal_angle) <= half_horizontal * frame_margin
            and abs(vertical_angle) <= half_vertical * frame_margin
        )
        framing = FRAME_INSIDE if inside else FRAME_OUTSIDE
    else:
        # A camera position without a usable direction or field of view cannot
        # prove framing either way; claiming INSIDE here would fabricate it.
        framing = FRAME_UNKNOWN

    if inactive_framing is not None:
        # The placement above is measured; the visibility claim is not made.
        framing = inactive_framing

    stand_name = TARGET_PHONE_STAND if anchor == TARGET_PHONE else TARGET_CAMERA_STAND
    stand = world.target(stand_name)
    return SpatialRelationship(
        framing=framing,
        transform_based=transform_based,
        character_position=character,
        character_forward=body_forward,
        camera_position=position,
        camera_forward=forward,
        camera_fov=camera_fov,
        distance_to_camera=full_distance,
        horizontal_distance=flat_distance,
        direction_to_camera=to_camera,
        relative_angle_degrees=relative_angle,
        horizontal_angle_degrees=horizontal_angle,
        vertical_angle_degrees=vertical_angle,
        orientation_error_degrees=orientation_error,
        inside_frame=_inside_frame_state(framing),
        target=anchor,
        reachable=stand.reachable if stand is not None else None,
    )


def _phone_framing(
    world: WorldState,
    character: tuple[float, float, float],
    flat_distance: Optional[float],
) -> str:
    """Phone-relative framing without a reported phone rotation.

    The comfortable band is measured against the phone itself, and the lateral
    error against the navmesh-sampled PHONE_STAND Unity already computed. A
    missing PHONE_STAND falls back to the bounded distance band, never to a
    fabricated "inside frame".
    """
    if flat_distance is None:
        return FRAME_UNKNOWN
    if flat_distance > MAX_PHONE_STAND_DISTANCE:
        return FRAME_TOO_FAR
    if flat_distance < MIN_PHONE_STAND_DISTANCE:
        return FRAME_TOO_CLOSE
    stand = world.target(TARGET_PHONE_STAND)
    if stand is None:
        return FRAME_UNKNOWN
    offset = horizontal_distance(character, stand.position)
    if offset is None:
        return FRAME_UNKNOWN
    return FRAME_INSIDE if offset <= POSITION_TOLERANCE_METERS else FRAME_OUTSIDE


def _empty(
    framing: str,
    *,
    character: Optional[tuple[float, float, float]] = None,
    body_forward: Optional[tuple[float, float, float]] = None,
    camera: Optional[tuple[float, float, float]] = None,
    camera_forward: Optional[tuple[float, float, float]] = None,
    camera_fov: Optional[float] = None,
    transform_based: bool = False,
    target: str = TARGET_CAMERA,
) -> SpatialRelationship:
    return SpatialRelationship(
        framing=framing,
        transform_based=transform_based,
        character_position=character,
        character_forward=body_forward,
        camera_position=camera,
        camera_forward=camera_forward,
        camera_fov=camera_fov,
        distance_to_camera=None,
        horizontal_distance=None,
        direction_to_camera=None,
        relative_angle_degrees=None,
        horizontal_angle_degrees=None,
        vertical_angle_degrees=None,
        orientation_error_degrees=None,
        inside_frame=None,
        target=target,
    )


# ------------------------------------------------------------------ planning


@dataclass(frozen=True)
class SpatialPlan:
    """Small deterministic action sketch produced from a spatial relationship."""

    steps: tuple[tuple[str, Optional[str]], ...]
    reason: str
    strategy: str
    target: str

    @property
    def empty(self) -> bool:
        return not self.steps


def _position_target(world: WorldState, anchor: str = TARGET_CAMERA) -> Optional[str]:
    """The allowlisted standing point used for framing.

    Only points Unity itself computed and reported as navigable are usable: an
    anchor-relative standing point first (``CAMERA_STAND``/``PHONE_STAND``),
    then the pre-existing room ``CENTER`` fallback for camera framing. A phone
    request never falls back to an unrelated point, so a missing PHONE_STAND
    stays a visible failure instead of a move to the wrong place.
    """
    candidates = (TARGET_PHONE_STAND,) if anchor == TARGET_PHONE else (
        TARGET_CAMERA_STAND,
        TARGET_CENTER,
    )
    for name in candidates:
        definition = world.target(name)
        if definition is None or not definition.navigation_target:
            continue
        if definition.reachable is False:
            continue
        return name
    return None


def plan_anchor(plan: Any) -> str:
    """Which allowlisted anchor an existing spatial plan works against.

    Derived from the plan's own step targets, so the post-action verifier and
    the diagnostics read the same allowlisted name the planner chose. Anything
    that is not a phone target verifies against the stream camera.
    """
    for step in getattr(plan, "steps", ()) or ():
        target = str(getattr(step, "target", "") or "").strip().upper()
        if target == TARGET_PHONE or target.startswith(f"{TARGET_PHONE}_"):
            return TARGET_PHONE
    return TARGET_CAMERA


def plan_spatial_steps(
    world: WorldState,
    relationship: SpatialRelationship,
    *,
    desired_state: str = DESIRED_FACING,
    target: str = TARGET_CAMERA,
) -> SpatialPlan:
    """Choose the minimum set of existing actions for a spatial request.

    The result is a sketch of existing action names; it is turned into a plan
    by the existing planner, validated by the existing step validator, and
    executed by the existing SkillExecutor. No Unity call is made here.
    """
    desired = (desired_state or DESIRED_FACING).upper()
    anchor = _normalize_anchor(target)

    framing = relationship.framing
    # PHASE 26A: an inactive camera still has a real reported transform and a
    # Unity-computed standing point, so the placement half of the request is
    # planned and measured here. Visibility itself is never claimed from it.
    placement_only = (
        framing in FRAMING_PLACEMENT_ONLY
        and relationship.transform_based
        and relationship.camera_position is not None
    )
    if framing in FRAMING_UNAVAILABLE and not placement_only:
        return SpatialPlan(
            (),
            spatial_reason(f"unavailable:{framing.lower()}"),
            "NO_DATA",
            anchor,
        )
    if relationship.camera_position is None:
        return SpatialPlan((), spatial_reason("unavailable:no_anchor_transform"), "NO_DATA", anchor)

    wants_position = framing in FRAMING_NEEDS_POSITION or (
        placement_only and desired in {DESIRED_IN_FRAME, DESIRED_CLOSER, DESIRED_NEAR}
    )
    stand = _position_target(world, anchor) if wants_position else None
    if wants_position and stand is None:
        # Moving is the only fix for a position request; without a reachable
        # standing point the request is a visible failure. Reporting success
        # from here is exactly the "stand beside the phone" defect.
        return SpatialPlan(
            (),
            spatial_reason(f"unreachable:{framing.lower()}"),
            "NO_REACHABLE_POSITION",
            anchor,
        )

    needs_position = framing in FRAMING_NEEDS_POSITION
    if placement_only and stand is not None:
        # Standing on Unity's own navmesh-sampled point is the measured position
        # fix, so an already-placed character is not asked to walk again.
        definition = world.target(stand)
        offset = (
            distance(relationship.character_position, definition.position)
            if definition is not None
            else None
        )
        needs_position = offset is None or offset > POSITION_TOLERANCE_METERS

    # The anchor target is the thing a request refers to: the stream camera for
    # camera requests, the phone itself for phone requests. Both are allowlisted
    # names Unity published; a request can never invent one.
    look_target = TARGET_PHONE if anchor == TARGET_PHONE else TARGET_CAMERA
    lookable = world.target(look_target) is not None

    steps: list[tuple[str, Optional[str]]] = []
    if needs_position and stand is not None:
        steps.append(("MOVE_TO", stand))

    # Turn when the measured orientation is actually wrong, and always after a
    # planned move: Unity's walker points the body along the path, so whatever
    # the pre-move yaw measured is invalidated by the walk itself. This is the
    # measured live defect -- a phone request whose pre-move error read 0.05deg
    # arrived 135deg away and the following LOOK_AT could not fix the body.
    orientation_needed = (
        needs_position
        or relationship.needs_orientation_change
        or relationship.orientation_error_degrees is None
    )
    turn = orientation_needed and lookable
    if turn:
        steps.append(("TURN_TO", look_target))

    wants_look = desired in {DESIRED_FACING, DESIRED_IN_FRAME}
    if desired in {DESIRED_CLOSER, DESIRED_NEAR} and not steps and not turn:
        # Position and orientation already satisfy the request. A look at the
        # anchor is the only observable action left, so the request still ends
        # on a real spatial measurement instead of a fabricated no-op.
        wants_look = True
    look = wants_look and lookable
    if look:
        steps.append(("LOOK_AT", look_target))

    if not steps:
        return SpatialPlan(
            (),
            spatial_reason(f"{anchor.lower()} framing {framing.lower()}; already suitable"),
            "ALREADY_SUITABLE",
            anchor,
        )

    if needs_position and turn:
        strategy = "MOVE_AND_TURN"
    elif needs_position:
        strategy = "MOVE"
    elif turn and look:
        strategy = "TURN_THEN_LOOK"
    elif turn:
        strategy = "TURN"
    else:
        strategy = "LOOK"
    reason = spatial_reason(
        f"{anchor.lower()} framing {framing.lower()}; {strategy.lower().replace('_', ' ')}"
    )
    return SpatialPlan(tuple(steps), reason, strategy, anchor)


__all__ = [
    "DESIRED_CLOSER",
    "DESIRED_FACING",
    "DESIRED_IN_FRAME",
    "DESIRED_NEAR",
    "DEFAULT_CAMERA_ASPECT",
    "DEFAULT_CAMERA_FRAMING_DISTANCE",
    "DEFAULT_CAMERA_TOO_CLOSE_DISTANCE",
    "DEFAULT_FRAME_MARGIN",
    "FRAME_CAMERA_INACTIVE",
    "FRAME_CAMERA_MOVES_WITH_PHONE",
    "FRAME_INSIDE",
    "FRAME_OUTSIDE",
    "FRAME_STREAM_INACTIVE",
    "FRAME_TOO_CLOSE",
    "FRAME_TOO_FAR",
    "FRAME_UNKNOWN",
    "FRAMING_NEEDS_POSITION",
    "FRAMING_PLACEMENT_ONLY",
    "FRAMING_UNAVAILABLE",
    "MAX_PHONE_STAND_DISTANCE",
    "MIN_PHONE_STAND_DISTANCE",
    "ORIENTATION_TOLERANCE_DEGREES",
    "POSITION_ARRIVAL_TOLERANCE_METERS",
    "POSITION_TOLERANCE_METERS",
    "SPATIAL_PLAN_PREFIX",
    "SpatialPlan",
    "SpatialRelationship",
    "TARGET_CAMERA",
    "TARGET_CAMERA_STAND",
    "TARGET_CENTER",
    "TARGET_PHONE",
    "TARGET_PHONE_STAND",
    "anchor_position",
    "camera_forward",
    "camera_position",
    "character_forward",
    "distance",
    "forward_from_euler",
    "forward_from_yaw",
    "has_camera_transform",
    "horizontal_distance",
    "is_spatial_plan",
    "plan_anchor",
    "plan_spatial_steps",
    "request_anchor",
    "spatial_observation",
    "spatial_reason",
    "spatial_relationship",
]
