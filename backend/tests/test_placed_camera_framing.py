from __future__ import annotations

import unittest
from datetime import datetime, timezone

from backend.brain.models import ActionResult, ActionType, ExecutionStatus, Goal, GoalType
from backend.brain.planner import BrainPlanner
from backend.brain.world_state import build_world_state
from backend.skills.executor import SkillExecutor
from backend.skills.library import create_default_skill_library
from backend.skills.models import SkillSelection, SkillStatus


NOW = datetime(2026, 10, 17, 0, 0, tzinfo=timezone.utc)


def snapshot(
    *,
    phone_state: str = "Placed",
    camera_stand_reachable: bool = True,
) -> dict:
    return {
        "character": {
            "position": [2.0, 0.0, 3.6],
            "rotation_y": 0,
            "posture": "standing",
            "current_action": "idle",
        },
        "phone": {
            "state": phone_state,
            "position": [-3.1, 0.763, 0.0],
            "placement_surface": "Table" if phone_state == "Placed" else None,
        },
        "stream": {
            "is_streaming": True,
            "camera_available": True,
            "camera_active": True,
        },
        "targets": {
            "CENTER": {
                "position": [0.0, 0.05, 0.0],
                "navigation_target": True,
                "look_at_target": True,
            },
            "CAMERA": {
                "position": [-3.1, 0.766, 0.0],
                "navigation_target": False,
                "look_at_target": True,
            },
            "CAMERA_STAND": {
                "position": [-0.233, 0.117, 2.008],
                "navigation_target": True,
                "look_at_target": False,
                "reachable": camera_stand_reachable,
            },
        },
    }


def world(
    *,
    phone_state: str = "Placed",
    camera_stand_reachable: bool = True,
):
    return build_world_state(
        snapshot(
            phone_state=phone_state,
            camera_stand_reachable=camera_stand_reachable,
        )
    )


class PlacedCameraFramingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.library = create_default_skill_library()
        self.executor = SkillExecutor(
            self.library,
            BrainPlanner(),
            clock=lambda: NOW,
        )

    def _start(self, value):
        return self.executor.create_plan(
            SkillSelection("setup_camera_and_dance", {}, "framing regression test"),
            Goal("goal_framing", GoalType.RESPOND_TO_VIEWER, 0.9, "framing test"),
            value,
            now=NOW,
        )

    def test_placed_camera_workflow_uses_existing_camera_stand(self) -> None:
        skill = self.library.get("setup_camera_and_dance")
        self.assertIsNotNone(skill)
        self.assertEqual(skill.composition[0].parameters["target"], "CAMERA_STAND")
        self.assertEqual(skill.preconditions[0].value, "Placed")

        result = self._start(world())
        self.assertEqual(result.reason, "OK")
        self.assertIsNotNone(result.plan)
        self.assertEqual(
            [step.action for step in result.plan.steps],
            [ActionType.MOVE_TO, ActionType.LOOK_AT, ActionType.DANCE],
        )
        self.assertEqual(result.plan.steps[0].target, "CAMERA_STAND")
        self.assertEqual(result.plan.steps[1].target, "CAMERA")

    def test_dispatch_is_not_completion_and_intermediate_steps_stay_running(self) -> None:
        result = self._start(world())
        self.assertEqual(result.execution.status, SkillStatus.RUNNING)
        self.assertEqual(result.execution.current_step, 0)

        self.assertIsNone(
            self.executor.observe_success(
                ActionResult(
                    ActionType.MOVE_TO,
                    ExecutionStatus.SUCCESS,
                    "reported",
                    "CAMERA_STAND",
                )
            )
        )
        self.assertEqual(result.execution.status, SkillStatus.RUNNING)
        self.assertEqual(result.execution.current_step, 1)

        self.executor.observe_success(
            ActionResult(
                ActionType.LOOK_AT,
                ExecutionStatus.SUCCESS,
                "reported",
                "CAMERA",
            )
        )
        self.assertEqual(result.execution.status, SkillStatus.RUNNING)
        self.assertEqual(result.execution.current_step, 2)

        completed = self.executor.observe_success(
            ActionResult(ActionType.DANCE, ExecutionStatus.SUCCESS, "reported")
        )
        self.assertIsNotNone(completed)
        self.assertEqual(completed.status, SkillStatus.SUCCESS)
        self.assertEqual(completed.steps_completed, 3)

    def test_unreachable_camera_stand_fails_without_a_plan(self) -> None:
        result = self._start(world(camera_stand_reachable=False))
        self.assertEqual(result.reason, "TARGET_UNREACHABLE")
        self.assertIsNone(result.plan)
        self.assertIsNone(result.execution)
        self.assertIsNone(self.executor.active)

    def test_interrupted_movement_never_reports_success(self) -> None:
        result = self._start(world())
        self.executor.observe_failure(
            ActionResult(
                ActionType.MOVE_TO,
                ExecutionStatus.FAILED,
                "interrupted_by_viewer",
                "CAMERA_STAND",
            ),
            final=True,
            interrupted=True,
        )
        self.assertEqual(result.execution.status, SkillStatus.INTERRUPTED)
        evaluation = self.executor.evaluations[-1]
        self.assertEqual(evaluation.status, SkillStatus.INTERRUPTED)
        self.assertEqual(evaluation.interruption_reason, "interrupted_by_viewer")
        self.assertIsNone(self.executor.active)

    def test_held_phone_does_not_enter_the_placed_camera_workflow(self) -> None:
        held = world(phone_state="Held")
        availability = self.executor.availability(
            "setup_camera_and_dance",
            held,
            now=NOW,
        )
        self.assertFalse(availability.valid)
        self.assertEqual(availability.reason, "PHONE_NOT_PLACED")
        self.assertEqual(self._start(held).reason, "PHONE_NOT_PLACED")


if __name__ == "__main__":
    unittest.main()
