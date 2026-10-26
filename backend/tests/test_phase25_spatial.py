"""PHASE 25: character intelligence / spatial awareness regression.

Every value under test is derived from transforms Unity already reports; there
is no computer vision, no second planner, and no direct transform access. The
historical failures this suite pins are "standing beside the phone instead of
in frame" and "turning away from the camera".
"""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace
from typing import Any

from backend.brain.loop import (
    SPATIAL_PLACEMENT_VERIFIED,
    AutonomousBrain,
    BrainConfig,
    BrainStatus,
)
from backend.brain.models import ActionResult, ActionType, ExecutionStatus
from backend.brain.reasoning import MockBrainLLMProvider
from backend.brain.service import BrainService
from backend.brain.spatial import (
    DESIRED_CLOSER,
    DESIRED_FACING,
    DESIRED_IN_FRAME,
    FRAME_CAMERA_INACTIVE,
    FRAME_INSIDE,
    FRAME_OUTSIDE,
    FRAME_STREAM_INACTIVE,
    FRAME_TOO_CLOSE,
    FRAME_TOO_FAR,
    ORIENTATION_TOLERANCE_DEGREES,
    POSITION_ARRIVAL_TOLERANCE_METERS,
    TARGET_CAMERA,
    TARGET_CAMERA_STAND,
    TARGET_PHONE,
    forward_from_euler,
    has_camera_transform,
    is_spatial_plan,
    plan_anchor,
    plan_spatial_steps,
    request_anchor,
    spatial_relationship,
)
from backend.brain.targets import TargetResolver
from backend.brain.world_state import build_world_state
from backend.chat.intent import extract_intent
from backend.chat.normalization import normalize_text
from backend.chat.service import ChatService, ChatServiceConfig
from backend.llm.mock import MockLLMProvider
from backend.skills.conditions import SkillConditionEvaluator
from backend.skills.library import create_default_skill_library
from backend.skills.models import (
    SkillCondition,
    SkillConditionOperator,
    SkillConditionType,
)
from backend.social.attention import assess_camera_framing
from backend.social.service import SocialAutonomyService
from backend.tests.dispatch_support import confirm

#: Live-room fixture (measured from the production Unity scene): the stream
#: camera sits where the phone was placed and Lacrimosa stands far away,
#: turned away from it.
CAMERA_POSITION = (-2.990, 1.226, 0.080)
CAMERA_EULER = (8.0, 55.0, 0.0)
CAMERA_FOV = 60.0
FAR_CHARACTER = (2.005, 0.056, 3.590)
#: NavMesh-sampled standing point Unity computes 3.5m in front of the camera.
CAMERA_STAND = (-0.151, 0.117, 2.068)


def yaw_toward(origin: tuple[float, float, float], target: tuple[float, float, float]) -> float:
    """Unity yaw whose flattened forward points from ``origin`` to ``target``."""
    return math.degrees(math.atan2(target[0] - origin[0], target[2] - origin[2]))


def snapshot(**overrides: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "character": {
            "position": list(FAR_CHARACTER),
            "rotation_y": 0.0,
            "posture": "standing",
            "current_action": "idle",
        },
        "phone": {
            "state": "World",
            "position": list(CAMERA_POSITION),
            "holder": None,
            "placement_surface": None,
        },
        "stream": {
            "is_streaming": True,
            "camera_available": True,
            "camera_active": True,
            "camera_position": list(CAMERA_POSITION),
            "camera_rotation": list(CAMERA_EULER),
            "camera_field_of_view": CAMERA_FOV,
        },
        "targets": {
            "CENTER": {
                "position": [0.0, 0.0, 0.0],
                "navigation_target": True,
                "look_at_target": True,
            },
            "CAMERA": {
                "position": list(CAMERA_POSITION),
                "navigation_target": False,
                "look_at_target": True,
            },
            "CAMERA_STAND": {
                "position": list(CAMERA_STAND),
                "navigation_target": True,
                "look_at_target": False,
                "reachable": True,
            },
            "PHONE": {
                "position": list(CAMERA_POSITION),
                "navigation_target": False,
                "look_at_target": True,
            },
            "PHONE_STAND": {
                "position": list(CAMERA_STAND),
                "navigation_target": True,
                "look_at_target": False,
                "reachable": True,
            },
        },
    }
    value.update(overrides)
    return value


def world(**overrides: Any):
    return build_world_state(snapshot(**overrides))


def moved_character(yaw: float = 0.0) -> dict[str, Any]:
    return {
        "position": list(CAMERA_STAND),
        "rotation_y": yaw,
        "posture": "standing",
        "current_action": "idle",
    }


def facing_yaw() -> float:
    return yaw_toward(CAMERA_STAND, CAMERA_POSITION)


class StubSpeech:
    """Minimal enabled speech service: enough for the appended SPEAK step."""

    enabled = True

    def __init__(self) -> None:
        self.requests: list[Any] = []

    def interrupt_current(self, reason: str) -> None:
        return None

    def request(self, text: str, **_: Any) -> Any:
        record = SimpleNamespace(request_id=f"speech_{len(self.requests) + 1}", text=text)
        self.requests.append(record)
        return record

    def cancel(self, request_id: str, reason: str) -> None:
        return None


def build_brain(
    speech: Any = None,
    social: SocialAutonomyService | None = None,
) -> tuple[AutonomousBrain, ChatService]:
    chat = ChatService(
        MockLLMProvider(),
        ChatServiceConfig(max_messages_per_rate_window=20),
    )
    brain = AutonomousBrain(
        MockBrainLLMProvider(),
        chat,
        BrainConfig(decision_interval_seconds=0.0),
        skill_library=create_default_skill_library(),
        speech_service=speech,
        social_service=social,
    )
    return brain, chat


def success(action: ActionType) -> ActionResult:
    return ActionResult(
        action=action,
        status=ExecutionStatus.SUCCESS,
        reason="reached destination",
    )


class WorldTransformTests(unittest.TestCase):
    def test_reported_transforms_are_parsed_and_echoed(self) -> None:
        value = world(
            character={
                "position": list(FAR_CHARACTER),
                "rotation_y": 0.0,
                "forward": [0.0, 0.0, 1.0],
                "current_action": "idle",
            }
        )
        self.assertEqual(value.character_forward, (0.0, 0.0, 1.0))
        self.assertEqual(value.camera_position, CAMERA_POSITION)
        self.assertEqual(value.camera_rotation, CAMERA_EULER)
        self.assertEqual(value.camera_fov, CAMERA_FOV)
        self.assertTrue(has_camera_transform(value))
        payload = value.to_dict()
        self.assertEqual(payload["character"]["forward"], [0.0, 0.0, 1.0])
        self.assertEqual(payload["stream"]["camera_position"], list(CAMERA_POSITION))
        self.assertEqual(payload["stream"]["camera_field_of_view"], CAMERA_FOV)

    def test_zero_field_of_view_is_treated_as_missing(self) -> None:
        # Unity's JsonUtility serializes an absent float as 0; only a positive
        # FOV is a real measurement.
        stream = dict(snapshot()["stream"])
        stream["camera_field_of_view"] = 0
        value = build_world_state(snapshot(stream=stream))
        self.assertIsNone(value.camera_fov)

    def test_legacy_snapshot_keeps_the_original_shape(self) -> None:
        value = build_world_state(
            {
                "character": {"position": [0, 0, 0], "rotation_y": 90},
                "stream": {"is_streaming": False, "camera_available": True},
                "targets": {
                    "CAMERA": {
                        "position": [1, 0.8, 0],
                        "navigation_target": False,
                        "look_at_target": True,
                    }
                },
            }
        )
        self.assertFalse(has_camera_transform(value))
        payload = value.to_dict()
        self.assertNotIn("forward", payload["character"])
        self.assertNotIn("camera_position", payload["stream"])
        self.assertNotIn("reachable", payload["targets"]["CAMERA"])

    def test_reachability_is_tri_state(self) -> None:
        value = build_world_state(
            snapshot(
                targets={
                    "CAMERA_STAND": {
                        "position": list(CAMERA_STAND),
                        "navigation_target": True,
                        "look_at_target": False,
                        "reachable": False,
                    },
                    "CENTER": {
                        "position": [0, 0, 0],
                        "navigation_target": True,
                        "look_at_target": True,
                    },
                }
            )
        )
        self.assertIs(value.target("CAMERA_STAND").reachable, False)
        self.assertIsNone(value.target("CENTER").reachable)
        self.assertIs(value.to_dict()["targets"]["CAMERA_STAND"]["reachable"], False)


class GeometryTests(unittest.TestCase):
    def test_forward_from_euler_matches_unity_yaw(self) -> None:
        forward = forward_from_euler(CAMERA_EULER)
        pitch = math.radians(CAMERA_EULER[0])
        yaw = math.radians(CAMERA_EULER[1])
        self.assertAlmostEqual(forward[0], math.sin(yaw) * math.cos(pitch), places=6)
        self.assertAlmostEqual(forward[1], -math.sin(pitch), places=6)
        self.assertAlmostEqual(forward[2], math.cos(yaw) * math.cos(pitch), places=6)

    def test_inside_frame_requires_the_character_in_the_frustum(self) -> None:
        value = build_world_state(
            snapshot(
                character={
                    "position": [0.0, 0.0, 3.0],
                    "rotation_y": 180.0,
                    "current_action": "idle",
                },
                phone={"state": "Placed", "position": [0, 0, 0]},
                stream={
                    "is_streaming": True,
                    "camera_available": True,
                    "camera_active": True,
                    "camera_position": [0.0, 0.0, 0.0],
                    "camera_rotation": [0.0, 0.0, 0.0],
                    "camera_field_of_view": CAMERA_FOV,
                },
                targets={
                    "CAMERA": {
                        "position": [0.0, 0.0, 0.0],
                        "navigation_target": False,
                        "look_at_target": True,
                    }
                },
            )
        )
        relationship = spatial_relationship(value)
        self.assertEqual(relationship.framing, FRAME_INSIDE)
        self.assertTrue(relationship.inside_frame)
        self.assertAlmostEqual(relationship.distance_to_camera, 3.0, places=3)
        self.assertAlmostEqual(relationship.relative_angle_degrees or 0.0, 0.0, places=3)
        # Facing the camera means the body forward points back at it.
        self.assertAlmostEqual(relationship.orientation_error_degrees, 0.0, places=3)

    def test_outside_frame_is_detected_instead_of_assuming_visibility(self) -> None:
        relationship = spatial_relationship(
            world(
                character={
                    "position": [50.0, 0.0, 50.0],
                    "rotation_y": 0.0,
                    "current_action": "idle",
                }
            )
        )
        # 50m away is outside the 5.5m framing rule before any frustum maths.
        self.assertEqual(relationship.framing, FRAME_TOO_FAR)
        self.assertFalse(relationship.inside_frame)
        self.assertGreater(relationship.orientation_error_degrees or 0.0, 20.0)

    def test_too_close_and_too_far_bands(self) -> None:
        near = spatial_relationship(
            world(
                character={
                    "position": [CAMERA_POSITION[0], 0.05, CAMERA_POSITION[2] + 0.5],
                    "rotation_y": facing_yaw(),
                    "current_action": "idle",
                }
            )
        )
        self.assertEqual(near.framing, FRAME_TOO_CLOSE)
        far = spatial_relationship(world())
        self.assertEqual(far.framing, FRAME_TOO_FAR)

    def test_inactive_states_never_read_as_inside_frame(self) -> None:
        stream_off = spatial_relationship(
            world(stream={"is_streaming": False, "camera_available": True, "camera_active": False})
        )
        self.assertEqual(stream_off.framing, FRAME_STREAM_INACTIVE)
        self.assertIsNone(stream_off.inside_frame)
        camera_off = spatial_relationship(
            world(
                stream={
                    "is_streaming": True,
                    "camera_available": True,
                    "camera_active": False,
                    "camera_position": list(CAMERA_POSITION),
                    "camera_rotation": list(CAMERA_EULER),
                    "camera_field_of_view": CAMERA_FOV,
                }
            )
        )
        self.assertEqual(camera_off.framing, FRAME_CAMERA_INACTIVE)
        self.assertIsNone(camera_off.inside_frame)

    def test_inactive_stream_with_reported_transform_keeps_placement_geometry(self) -> None:
        stream = dict(snapshot()["stream"])
        stream.update({"is_streaming": False, "camera_active": False})
        relationship = spatial_relationship(build_world_state(snapshot(stream=stream)))
        # Visibility is still unmeasurable, but the reported transform keeps
        # the placement half (distance and body orientation) observable.
        self.assertEqual(relationship.framing, FRAME_STREAM_INACTIVE)
        self.assertIsNone(relationship.inside_frame)
        self.assertTrue(relationship.transform_based)
        self.assertIsNotNone(relationship.distance_to_camera)
        self.assertIsNotNone(relationship.orientation_error_degrees)

    def test_inactive_camera_with_reported_transform_keeps_placement_geometry(self) -> None:
        stream = dict(snapshot()["stream"])
        stream.update({"is_streaming": True, "camera_active": False})
        relationship = spatial_relationship(build_world_state(snapshot(stream=stream)))
        self.assertEqual(relationship.framing, FRAME_CAMERA_INACTIVE)
        self.assertIsNone(relationship.inside_frame)
        self.assertTrue(relationship.transform_based)
        self.assertIsNotNone(relationship.distance_to_camera)
        self.assertIsNotNone(relationship.orientation_error_degrees)

    def test_camera_without_direction_cannot_prove_framing(self) -> None:
        relationship = spatial_relationship(
            world(
                stream={
                    "is_streaming": True,
                    "camera_available": True,
                    "camera_active": True,
                    "camera_position": list(CAMERA_POSITION),
                },
                character={
                    "position": [CAMERA_POSITION[0] + 3.0, 0.05, CAMERA_POSITION[2]],
                    "rotation_y": -90.0,
                    "current_action": "idle",
                },
            )
        )
        self.assertEqual(relationship.framing, "UNKNOWN")
        self.assertIsNone(relationship.inside_frame)

    def test_orientation_error_uses_world_transforms_not_the_head_bone(self) -> None:
        relationship = spatial_relationship(world())
        # Camera yaw 55 degrees, Lacrimosa yaw 0: the body faces elsewhere.
        expected = 125.0
        self.assertAlmostEqual(
            relationship.orientation_error_degrees or 0.0, expected, delta=1.0
        )
        self.assertGreater(
            relationship.orientation_error_degrees or 0.0,
            ORIENTATION_TOLERANCE_DEGREES,
        )


class TargetResolutionTests(unittest.TestCase):
    def test_camera_aliases_resolve_to_one_allowlisted_target(self) -> None:
        value = world()
        for name in ("camera", "CAMERA_TARGET", "stream_camera", "STREAM CAMERA"):
            resolved, reason = TargetResolver(value).resolve(name)
            self.assertEqual(resolved, TARGET_CAMERA, msg=name)
            self.assertEqual(reason, "OK")

    def test_stand_points_only_resolve_when_unity_published_them(self) -> None:
        value = world()
        resolved, _ = TargetResolver(value).resolve("camera_stand")
        self.assertEqual(resolved, TARGET_CAMERA_STAND)
        legacy = build_world_state(snapshot(targets={"CAMERA": snapshot()["targets"]["CAMERA"]}))
        resolved, reason = TargetResolver(legacy).resolve("CAMERA_STAND")
        self.assertIsNone(resolved)
        self.assertEqual(reason, "TARGET_NOT_FOUND")

    def test_arbitrary_text_never_becomes_a_transform(self) -> None:
        resolved, reason = TargetResolver(world()).resolve("UnityEngine.Object/../Secret")
        self.assertIsNone(resolved)
        self.assertEqual(reason, "TARGET_NOT_FOUND")

    def test_request_anchor_normalizes_chat_targets(self) -> None:
        self.assertEqual(request_anchor("phone"), TARGET_PHONE)
        self.assertEqual(request_anchor(" Phone "), TARGET_PHONE)
        self.assertEqual(request_anchor("camera"), TARGET_CAMERA)
        self.assertEqual(request_anchor(None), TARGET_CAMERA)

    def test_reachability_gates_the_existing_condition(self) -> None:
        evaluator = SkillConditionEvaluator()
        condition = SkillCondition(
            condition_type=SkillConditionType.TARGET_REACHABLE,
            operator=SkillConditionOperator.IS_TRUE,
            failure_reason="TARGET_NOT_REACHABLE",
        )
        reachable = world()
        self.assertIsNone(
            evaluator.evaluate_condition(condition, reachable, {"target": "CAMERA_STAND"})
        )
        unreachable = build_world_state(
            snapshot(
                targets={
                    "CAMERA_STAND": {
                        "position": list(CAMERA_STAND),
                        "navigation_target": True,
                        "look_at_target": False,
                        "reachable": False,
                    }
                }
            )
        )
        self.assertEqual(
            evaluator.evaluate_condition(condition, unreachable, {"target": "CAMERA_STAND"}),
            "TARGET_NOT_REACHABLE",
        )


class IntentTests(unittest.TestCase):
    def test_camera_request_vocabulary_normalizes_to_structured_intent(self) -> None:
        cases = {
            "look at the camera": ("camera", DESIRED_FACING),
            "face the camera": ("camera", DESIRED_FACING),
            "turn toward the camera": ("camera", DESIRED_FACING),
            "turn toward me": ("camera", DESIRED_FACING),
            "come closer to the camera": ("camera", DESIRED_CLOSER),
            "move near the camera": ("camera", DESIRED_CLOSER),
            "stand in front of the phone": ("phone", DESIRED_IN_FRAME),
            "stand in front of the camera": ("camera", DESIRED_IN_FRAME),
        }
        for text, (target, desired) in cases.items():
            result = extract_intent(normalize_text(text))
            self.assertEqual(result.target, target, msg=text)
            self.assertEqual(result.desired_state, desired, msg=text)

    def test_free_text_has_no_spatial_desired_state(self) -> None:
        result = extract_intent(normalize_text("what are you doing?"))
        self.assertIsNone(result.desired_state)

    def test_chat_submission_carries_the_desired_state(self) -> None:
        chat = ChatService(
            MockLLMProvider(),
            ChatServiceConfig(max_messages_per_rate_window=20),
        )
        submission = chat.submit("viewer_01", "Look at the camera!")
        self.assertTrue(submission.accepted)
        self.assertEqual(submission.message.desired_state, DESIRED_FACING)
        self.assertEqual(submission.message.target, "camera")


class SpatialPlanTests(unittest.TestCase):
    def test_far_and_turned_away_plans_move_then_turn_then_look(self) -> None:
        value = world()
        relationship = spatial_relationship(value)
        spatial = plan_spatial_steps(
            value, relationship, desired_state=DESIRED_FACING, target=TARGET_CAMERA
        )
        self.assertEqual(
            spatial.steps,
            (("MOVE_TO", TARGET_CAMERA_STAND), ("TURN_TO", TARGET_CAMERA), ("LOOK_AT", TARGET_CAMERA)),
        )
        self.assertEqual(spatial.strategy, "MOVE_AND_TURN")

    def test_already_suitable_position_only_looks(self) -> None:
        value = build_world_state(
            snapshot(character=moved_character(facing_yaw()))
        )
        relationship = spatial_relationship(value)
        self.assertEqual(relationship.framing, FRAME_INSIDE)
        self.assertFalse(relationship.needs_orientation_change)
        spatial = plan_spatial_steps(
            value, relationship, desired_state=DESIRED_FACING, target=TARGET_CAMERA
        )
        self.assertEqual(spatial.steps, (("LOOK_AT", TARGET_CAMERA),))
        self.assertEqual(spatial.strategy, "LOOK")

    def test_closer_request_still_acts_when_already_inside_frame(self) -> None:
        value = build_world_state(snapshot(character=moved_character(facing_yaw())))
        spatial = plan_spatial_steps(
            value,
            spatial_relationship(value),
            desired_state=DESIRED_CLOSER,
            target=TARGET_CAMERA,
        )
        self.assertEqual(spatial.steps, (("LOOK_AT", TARGET_CAMERA),))

    def test_outside_frame_without_a_reachable_stand_point_fails_visibly(self) -> None:
        value = build_world_state(
            snapshot(
                targets={
                    "CAMERA": snapshot()["targets"]["CAMERA"],
                    "CENTER": {
                        "position": [0, 0, 0],
                        "navigation_target": True,
                        "look_at_target": True,
                        "reachable": False,
                    },
                    "CAMERA_STAND": {
                        "position": list(CAMERA_STAND),
                        "navigation_target": True,
                        "look_at_target": False,
                        "reachable": False,
                    },
                }
            )
        )
        spatial = plan_spatial_steps(
            value,
            spatial_relationship(value),
            desired_state=DESIRED_IN_FRAME,
            target=TARGET_CAMERA,
        )
        self.assertTrue(spatial.empty)
        self.assertEqual(spatial.strategy, "NO_REACHABLE_POSITION")
        self.assertIn("unreachable", spatial.reason)

    def test_plan_anchor_reads_the_allowlisted_step_targets(self) -> None:
        class Step:
            def __init__(self, target: str) -> None:
                self.target = target

        class Plan:
            def __init__(self, *targets: str, reason: str = "spatial:test") -> None:
                self.steps = tuple(Step(target) for target in targets)
                self.reason = reason

        self.assertTrue(is_spatial_plan(Plan("CAMERA_STAND")))
        self.assertEqual(plan_anchor(Plan("CAMERA_STAND", "CAMERA")), TARGET_CAMERA)
        self.assertEqual(plan_anchor(Plan("PHONE_STAND", "PHONE")), TARGET_PHONE)
        # Provenance is the spatial reason the deterministic branch writes; a
        # provider plan that only looks at the camera stays legacy.
        self.assertFalse(is_spatial_plan(Plan("CAMERA", reason="structured plan")))
        self.assertFalse(is_spatial_plan(Plan("CAMERA_STAND", reason="structured plan")))

    def test_phone_anchor_does_not_require_an_active_stream(self) -> None:
        value = build_world_state(
            snapshot(
                stream={"is_streaming": False, "camera_available": True, "camera_active": False}
            )
        )
        relationship = spatial_relationship(value, target=TARGET_PHONE)
        self.assertNotEqual(relationship.framing, FRAME_STREAM_INACTIVE)
        spatial = plan_spatial_steps(
            value, relationship, desired_state=DESIRED_IN_FRAME, target=TARGET_PHONE
        )
        self.assertIn(("MOVE_TO", "PHONE_STAND"), spatial.steps)

    def test_inactive_stream_still_plans_the_camera_placement_half(self) -> None:
        stream = dict(snapshot()["stream"])
        stream.update({"is_streaming": False, "camera_active": False})
        value = build_world_state(snapshot(stream=stream))
        relationship = spatial_relationship(value)
        self.assertEqual(relationship.framing, FRAME_STREAM_INACTIVE)
        spatial = plan_spatial_steps(
            value,
            relationship,
            desired_state=DESIRED_IN_FRAME,
            target=TARGET_CAMERA,
        )
        self.assertNotEqual(spatial.strategy, "NO_DATA")
        self.assertEqual(
            spatial.steps,
            (
                ("MOVE_TO", TARGET_CAMERA_STAND),
                ("TURN_TO", TARGET_CAMERA),
                ("LOOK_AT", TARGET_CAMERA),
            ),
        )
        self.assertEqual(spatial.strategy, "MOVE_AND_TURN")

    def test_inactive_stream_closer_at_the_stand_point_plans_a_look(self) -> None:
        stream = dict(snapshot()["stream"])
        stream.update({"is_streaming": False, "camera_active": False})
        value = build_world_state(
            snapshot(character=moved_character(facing_yaw()), stream=stream)
        )
        spatial = plan_spatial_steps(
            value,
            spatial_relationship(value),
            desired_state=DESIRED_CLOSER,
            target=TARGET_CAMERA,
        )
        self.assertEqual(spatial.steps, (("LOOK_AT", TARGET_CAMERA),))
        self.assertEqual(spatial.strategy, "LOOK")

    def test_planned_move_always_re_turns_before_looking(self) -> None:
        # The measured live defect: the pre-move yaw was already inside the
        # orientation tolerance, so no TURN_TO was planned -- but Unity's
        # walker points the body along the path, the walk landed 135deg away,
        # and LOOK_AT cannot fix a body yaw. A planned move invalidates the
        # pre-move orientation measurement, so the turn is mandatory.
        value = build_world_state(
            snapshot(
                character={
                    "position": list(FAR_CHARACTER),
                    "rotation_y": yaw_toward(FAR_CHARACTER, CAMERA_POSITION),
                    "posture": "standing",
                    "current_action": "idle",
                }
            )
        )
        for target in (TARGET_CAMERA, TARGET_PHONE):
            with self.subTest(target=target):
                relationship = spatial_relationship(value, target=target)
                self.assertIn(relationship.framing, {FRAME_OUTSIDE, FRAME_TOO_FAR})
                self.assertFalse(relationship.needs_orientation_change)
                spatial = plan_spatial_steps(
                    value, relationship, desired_state=DESIRED_IN_FRAME, target=target
                )
                self.assertEqual(
                    spatial.steps,
                    (
                        ("MOVE_TO", f"{target}_STAND"),
                        ("TURN_TO", target),
                        ("LOOK_AT", target),
                    ),
                )
                self.assertEqual(spatial.strategy, "MOVE_AND_TURN")


class LoopVerificationTests(unittest.TestCase):
    def _start(self, brain: AutonomousBrain, chat: ChatService, text: str, value) -> Any:
        submission = chat.submit("viewer_01", text)
        self.assertTrue(submission.accepted)
        return brain.tick(value, None)

    def test_move_success_without_arrival_is_not_reported_as_success(self) -> None:
        brain, chat = build_brain()
        result = self._start(brain, chat, "Look at the camera.", world())
        self.assertEqual(result.action.action, ActionType.MOVE_TO)
        self.assertEqual(result.action.target, TARGET_CAMERA_STAND)
        self.assertTrue(is_spatial_plan(result.plan))

        # Unity reports success but the character never moved.
        retry = brain.tick(world(), confirm(brain, success(ActionType.MOVE_TO)))
        self.assertEqual(retry.action.action, ActionType.MOVE_TO)
        self.assertEqual(retry.action.target, TARGET_CAMERA_STAND)
        # The failed step is re-issued (the existing bounded retry), and the
        # reported "success" is recorded as a verification failure.
        self.assertEqual(brain.state.status, BrainStatus.EXECUTING)
        self.assertIn("spatial_verification_failed:position_not_reached", brain.state.last_failure)
        verification = brain.state.spatial_verification
        self.assertEqual(verification["result"], "FAILED")
        self.assertGreater(
            verification["observed"]["distance_to_target"],
            POSITION_ARRIVAL_TOLERANCE_METERS,
        )

    def test_arrived_move_advances_to_the_turn_step(self) -> None:
        brain, chat = build_brain()
        self._start(brain, chat, "Look at the camera.", world())
        after_move = brain.tick(
            build_world_state(snapshot(character=moved_character(0.0))),
            confirm(brain, success(ActionType.MOVE_TO)),
        )
        self.assertEqual(after_move.action.action, ActionType.TURN_TO)
        self.assertEqual(after_move.action.target, TARGET_CAMERA)
        self.assertEqual(brain.state.spatial_verification["result"], "VERIFIED")

    def test_turn_that_leaves_the_body_facing_away_is_rejected(self) -> None:
        brain, chat = build_brain()
        self._start(brain, chat, "Look at the camera.", world())
        brain.tick(
            build_world_state(snapshot(character=moved_character(0.0))),
            confirm(brain, success(ActionType.MOVE_TO)),
        )
        # TURN_TO reported success, but the body still faces elsewhere.
        retry = brain.tick(
            build_world_state(snapshot(character=moved_character(90.0))),
            confirm(brain, success(ActionType.TURN_TO)),
        )
        self.assertEqual(retry.action.action, ActionType.TURN_TO)
        self.assertTrue(
            brain.state.last_failure.startswith("spatial_verification_failed:orientation_error")
        )
        verification = brain.state.spatial_verification
        self.assertEqual(verification["result"], "FAILED")
        self.assertGreater(
            verification["observed"]["orientation_error"], ORIENTATION_TOLERANCE_DEGREES
        )

    def test_facing_camera_then_looking_completes_the_plan(self) -> None:
        brain, chat = build_brain()
        self._start(brain, chat, "Look at the camera.", world())
        facing = build_world_state(snapshot(character=moved_character(facing_yaw())))
        after_move = brain.tick(facing, confirm(brain, success(ActionType.MOVE_TO)))
        self.assertEqual(after_move.action.action, ActionType.TURN_TO)
        after_turn = brain.tick(facing, confirm(brain, success(ActionType.TURN_TO)))
        self.assertEqual(after_turn.action.action, ActionType.LOOK_AT)
        self.assertEqual(after_turn.action.target, TARGET_CAMERA)
        done = brain.tick(facing, confirm(brain, success(ActionType.LOOK_AT)))
        self.assertEqual(done.status, BrainStatus.IDLE)
        self.assertEqual(done.message, "plan_complete")
        self.assertEqual(brain.state.spatial_verification["result"], "VERIFIED")
        self.assertLessEqual(
            brain.state.spatial_verification["observed"]["orientation_error"],
            ORIENTATION_TOLERANCE_DEGREES,
        )

    def test_outside_frame_request_verifies_the_framing_after_the_move(self) -> None:
        brain, chat = build_brain()
        self._start(brain, chat, "Stand in front of the camera.", world())
        # The move completed somewhere else entirely: framing is still wrong.
        failed = brain.tick(world(), confirm(brain, success(ActionType.MOVE_TO)))
        self.assertEqual(failed.action.action, ActionType.MOVE_TO)
        self.assertTrue(
            brain.state.last_failure.startswith("spatial_verification_failed:position_not_reached")
        )
        # Retry lands on the navmesh-sampled stand point: now inside frame.
        moved = brain.tick(
            build_world_state(snapshot(character=moved_character(0.0))),
            confirm(brain, success(ActionType.MOVE_TO)),
        )
        self.assertEqual(moved.action.action, ActionType.TURN_TO)
        verification = brain.state.spatial_verification
        self.assertEqual(verification["result"], "VERIFIED")
        self.assertTrue(verification["observed"]["inside_frame"])
        self.assertEqual(verification["observed"]["framing"], FRAME_INSIDE)

    def test_move_that_arrives_outside_frame_is_rejected(self) -> None:
        brain, chat = build_brain()
        # The published standing point sits right where the character already
        # is, so the arrival half of the check passes while the reported
        # position is still far outside the camera frame. The framing half of
        # the verification is what has to reject the move.
        nearby_stand = (FAR_CHARACTER[0] + 0.4, FAR_CHARACTER[1], FAR_CHARACTER[2])
        targets = dict(snapshot()["targets"])
        targets["CAMERA_STAND"] = {
            "position": list(nearby_stand),
            "navigation_target": True,
            "look_at_target": False,
            "reachable": True,
        }
        value = build_world_state(snapshot(targets=targets))
        self._start(brain, chat, "Stand in front of the camera.", value)
        rejected = brain.tick(value, confirm(brain, success(ActionType.MOVE_TO)))
        self.assertEqual(rejected.action.action, ActionType.MOVE_TO)
        self.assertTrue(
            brain.state.last_failure.startswith(
                "spatial_verification_failed:framing_not_achieved"
            )
        )
        verification = brain.state.spatial_verification
        self.assertEqual(verification["result"], "FAILED")
        self.assertLessEqual(
            verification["observed"]["distance_to_target"],
            POSITION_ARRIVAL_TOLERANCE_METERS,
        )
        self.assertIs(verification["observed"]["inside_frame"], False)

    def test_inactive_stream_verifies_placement_without_claiming_framing(self) -> None:
        social = SocialAutonomyService()
        # Mirror the live Phase 24 state: the stream was deliberately stopped,
        # so stream maintenance does not preempt the viewer's placement
        # request.
        social.context.last_intentional_stream_stop = True
        brain, chat = build_brain(social=social)
        stream = dict(snapshot()["stream"])
        stream.update({"is_streaming": False, "camera_active": False})
        world_off = build_world_state(snapshot(stream=stream))
        started = self._start(brain, chat, "Stand in front of the camera.", world_off)
        self.assertEqual(started.action.action, ActionType.MOVE_TO)
        self.assertEqual(started.action.target, TARGET_CAMERA_STAND)
        self.assertTrue(is_spatial_plan(started.plan))

        moved_off = build_world_state(snapshot(character=moved_character(0.0), stream=stream))
        after_move = brain.tick(moved_off, confirm(brain, success(ActionType.MOVE_TO)))
        self.assertEqual(after_move.action.action, ActionType.TURN_TO)
        placement = brain.state.spatial_verification
        self.assertEqual(placement["result"], SPATIAL_PLACEMENT_VERIFIED)
        self.assertEqual(placement["reason"], "framing_not_measured:stream_inactive")
        self.assertIsNone(placement["observed"]["inside_frame"])
        self.assertEqual(placement["observed"]["framing"], FRAME_STREAM_INACTIVE)

        facing_off = build_world_state(
            snapshot(character=moved_character(facing_yaw()), stream=stream)
        )
        after_turn = brain.tick(facing_off, confirm(brain, success(ActionType.TURN_TO)))
        self.assertEqual(after_turn.action.action, ActionType.LOOK_AT)
        done = brain.tick(facing_off, confirm(brain, success(ActionType.LOOK_AT)))
        self.assertEqual(done.status, BrainStatus.IDLE)
        self.assertEqual(done.message, "plan_complete")
        # The visibility claim remains absent through completion.
        self.assertEqual(
            brain.state.spatial_verification["result"], SPATIAL_PLACEMENT_VERIFIED
        )
        self.assertIsNone(brain.state.spatial_verification["observed"]["inside_frame"])

    def test_verification_exhausts_retries_instead_of_looping(self) -> None:
        brain, chat = build_brain()
        started = self._start(brain, chat, "Look at the camera.", world())
        # Each reported success fails verification against real state: the
        # bounded retry re-issues the step once, then the plan is abandoned
        # instead of re-running the same failing action forever.
        brain.tick(world(), confirm(brain, success(ActionType.MOVE_TO)))
        outcome = brain.tick(world(), confirm(brain, success(ActionType.MOVE_TO)))
        self.assertNotEqual(outcome.message, "plan_complete")
        current = brain.state.current_plan
        self.assertTrue(
            current is None or current.plan_id != started.plan.plan_id,
            "the failed spatial plan must not stay active",
        )
        verification = brain.state.spatial_verification
        self.assertEqual(verification["result"], "FAILED")
        self.assertTrue(verification["reason"].startswith("position_not_reached"))

    def test_look_at_success_while_the_body_turned_away_is_rejected(self) -> None:
        # The already-in-frame case plans only the look. Unity reports LOOK_AT
        # as done, but the body has turned away since; the request must never
        # complete on the command alone, and the boundary is bounded: one
        # retry, then the plan is abandoned for a fresh decision (replan).
        brain, chat = build_brain()
        started = self._start(
            brain,
            chat,
            "Look at the camera.",
            build_world_state(snapshot(character=moved_character(facing_yaw()))),
        )
        self.assertEqual(started.action.action, ActionType.LOOK_AT)
        turned_away = build_world_state(snapshot(character=moved_character(0.0)))
        retry = brain.tick(turned_away, confirm(brain, success(ActionType.LOOK_AT)))
        self.assertEqual(retry.action.action, ActionType.LOOK_AT)
        self.assertTrue(
            brain.state.last_failure.startswith("spatial_verification_failed:orientation_error")
        )
        self.assertEqual(brain.state.spatial_verification["result"], "FAILED")
        outcome = brain.tick(turned_away, confirm(brain, success(ActionType.LOOK_AT)))
        self.assertNotEqual(outcome.message, "plan_complete")
        current = brain.state.current_plan
        self.assertTrue(
            current is None or current.plan_id != started.plan.plan_id,
            "the failed spatial plan must not stay active",
        )
        self.assertEqual(brain.state.spatial_verification["result"], "FAILED")
        self.assertGreater(
            brain.state.spatial_verification["observed"]["orientation_error"],
            ORIENTATION_TOLERANCE_DEGREES,
        )

    def test_legacy_plan_is_never_spatially_verified(self) -> None:
        brain, chat = build_brain()
        # Unity reported no camera transform (a pre-PHASE-25 snapshot): the
        # structured request must fall back to the historical provider plan,
        # and that plan keeps advancing on SUCCESS exactly as it did before.
        legacy = build_world_state(
            {
                "character": {
                    "position": list(FAR_CHARACTER),
                    "rotation_y": 0.0,
                    "posture": "standing",
                    "current_action": "idle",
                },
                "phone": {
                    "state": "Placed",
                    "position": list(CAMERA_POSITION),
                    "placement_surface": "Table",
                },
                "stream": {
                    "is_streaming": True,
                    "camera_available": True,
                    "camera_active": True,
                },
                "targets": {
                    "CENTER": {
                        "position": [0.0, 0.0, 0.0],
                        "navigation_target": True,
                        "look_at_target": True,
                    },
                    "CAMERA": {
                        "position": list(CAMERA_POSITION),
                        "navigation_target": False,
                        "look_at_target": True,
                    },
                },
            }
        )
        self.assertFalse(has_camera_transform(legacy))
        started = self._start(brain, chat, "Look at the camera.", legacy)
        self.assertEqual(started.action.action, ActionType.LOOK_AT)
        self.assertEqual(started.action.target, TARGET_CAMERA)
        self.assertFalse(is_spatial_plan(started.plan))
        done = brain.tick(legacy, confirm(brain, success(ActionType.LOOK_AT)))
        self.assertEqual(done.status, BrainStatus.IDLE)
        self.assertEqual(done.message, "plan_complete")
        self.assertIsNone(brain.state.spatial_verification)

    def test_dialogue_step_keeps_the_verified_spatial_record(self) -> None:
        # The measured live defect: a completed spatial plan ended with an
        # appended SPEAK step, and that step overwrote the VERIFIED LOOK_AT
        # record with UNVERIFIED / no_spatial_state -- diagnostics denied a
        # success the transforms had actually confirmed. Only the spatial
        # steps own that record.
        brain, chat = build_brain(speech=StubSpeech())
        started = self._start(
            brain,
            chat,
            "Look at the camera.",
            build_world_state(snapshot(character=moved_character(facing_yaw()))),
        )
        self.assertEqual(
            [step.action for step in started.plan.steps],
            [ActionType.LOOK_AT, ActionType.SPEAK],
        )
        after_look = brain.tick(
            build_world_state(snapshot(character=moved_character(facing_yaw()))),
            confirm(brain, success(ActionType.LOOK_AT)),
        )
        self.assertEqual(after_look.action.action, ActionType.SPEAK)
        verified = brain.state.spatial_verification
        self.assertEqual(verified["result"], "VERIFIED")
        self.assertEqual(verified["action"], "LOOK_AT")
        done = brain.tick(
            build_world_state(snapshot(character=moved_character(facing_yaw()))),
            confirm(brain, success(ActionType.SPEAK)),
        )
        self.assertEqual(done.message, "plan_complete")
        # The dialogue step must not have replaced the measured record.
        self.assertEqual(brain.state.spatial_verification["result"], "VERIFIED")
        self.assertEqual(brain.state.spatial_verification["action"], "LOOK_AT")
        self.assertLessEqual(
            brain.state.spatial_verification["observed"]["orientation_error"],
            ORIENTATION_TOLERANCE_DEGREES,
        )

    def test_phone_request_moves_to_the_phone_stand_point(self) -> None:
        brain, chat = build_brain()
        result = self._start(brain, chat, "Stand in front of the phone.", world())
        self.assertEqual(result.action.action, ActionType.MOVE_TO)
        self.assertEqual(result.action.target, "PHONE_STAND")
        self.assertEqual(plan_anchor(result.plan), TARGET_PHONE)
        verification_world = build_world_state(snapshot(character=moved_character(facing_yaw())))
        brain.tick(verification_world, confirm(brain, success(ActionType.MOVE_TO)))
        # The phone stand point is the same navmesh point in this fixture, so
        # the move verifies and the plan continues.
        self.assertEqual(brain.state.spatial_verification["result"], "VERIFIED")

    def test_held_phone_request_is_evaluated_against_the_camera(self) -> None:
        brain, chat = build_brain()
        value = world(
            phone={
                "state": "Held",
                "position": list(FAR_CHARACTER),
                "holder": "Lacrimosa",
                "placement_surface": None,
            }
        )
        result = self._start(brain, chat, "Stand in front of the phone.", value)
        self.assertEqual(plan_anchor(result.plan), TARGET_CAMERA)

    def test_standing_beside_the_phone_without_a_transform_fails_loudly(self) -> None:
        brain, chat = build_brain()
        value = build_world_state(
            snapshot(
                targets={
                    "CAMERA": snapshot()["targets"]["CAMERA"],
                    "CENTER": {
                        "position": [0, 0, 0],
                        "navigation_target": True,
                        "look_at_target": True,
                        "reachable": False,
                    },
                    "CAMERA_STAND": {
                        "position": list(CAMERA_STAND),
                        "navigation_target": True,
                        "look_at_target": False,
                        "reachable": False,
                    },
                }
            )
        )
        outcome = self._start(brain, chat, "Stand in front of the camera.", value)
        self.assertIsNone(outcome.action)
        self.assertEqual(outcome.status, BrainStatus.ERROR)
        self.assertTrue(outcome.message.startswith("planning_failed:spatial:unreachable"))


class DiagnosticsTests(unittest.TestCase):
    def test_service_exposes_bounded_spatial_diagnostics(self) -> None:
        brain, chat = build_brain()
        service = BrainService(brain, chat)
        chat.submit("viewer_01", "Look at the camera.")
        brain.tick(world(), None)
        payload = service.get_state()["spatial"]
        for key in (
            "character_position",
            "character_forward",
            "camera_position",
            "camera_forward",
            "distance",
            "relative_angle",
            "orientation_error",
            "inside_frame",
            "target",
            "reachable",
            "spatial_state",
            "verification_result",
        ):
            self.assertIn(key, payload)
        self.assertEqual(payload["target"], TARGET_CAMERA)
        self.assertEqual(payload["spatial_state"], "MOVE_AND_TURN")
        self.assertTrue(payload["spatial_plan_active"])

    def test_legacy_social_framing_semantics_are_unchanged(self) -> None:
        # Phase 08 semantics are pinned exactly: the camera is a fixed framing
        # reference only while the phone sits on a surface, and the rule is the
        # plain 5.5m horizontal distance to the CAMERA target.
        placed = world(
            character=moved_character(facing_yaw()),
            phone={
                "state": "Placed",
                "position": list(CAMERA_POSITION),
                "placement_surface": "Table",
            },
        )
        self.assertEqual(assess_camera_framing(placed), "INSIDE_FRAME")
        self.assertEqual(assess_camera_framing(world()), "CAMERA_MOVES_WITH_PHONE")
        # The same standing point with the phone only in the world is still the
        # "camera moves with the phone" case for the legacy assessment.
        self.assertEqual(
            assess_camera_framing(world(character=moved_character(facing_yaw()))),
            "CAMERA_MOVES_WITH_PHONE",
        )
        self.assertEqual(
            assess_camera_framing(
                world(stream={"is_streaming": False, "camera_available": True})
            ),
            "STREAM_INACTIVE",
        )
        self.assertEqual(assess_camera_framing(None), "UNKNOWN")
        held = world(phone={"state": "Held", "position": [0, 0, 0]})
        self.assertEqual(assess_camera_framing(held), "CAMERA_MOVES_WITH_PHONE")

    def test_phone_and_camera_relationships_recalculate_from_reported_state(self) -> None:
        placed = world()
        placed_relationship = spatial_relationship(placed, target=TARGET_PHONE)
        moved_phone = world(
            phone={
                "state": "World",
                "position": [10.0, 0.75, 10.0],
                "holder": None,
                "placement_surface": None,
            }
        )
        moved_relationship = spatial_relationship(moved_phone, target=TARGET_PHONE)
        self.assertNotAlmostEqual(
            placed_relationship.distance_to_camera,
            moved_relationship.distance_to_camera,
        )


if __name__ == "__main__":
    unittest.main()
