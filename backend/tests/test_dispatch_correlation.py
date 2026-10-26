"""Focused tests for PHASE 34 end-to-end action dispatch correlation.

Background (see ``Brain_Action_Execution_Lifecycle_Audit_20261019T001054Z``):
``_is_awaited_result`` matched a pending result by action *type* only, so a
stale or duplicate result of the same type could complete a newer action.
Every accepted dispatch now carries a unique ``dispatch_id`` that Unity echoes
with the result, and a result may only complete the exact dispatch that
produced it.

Everything here is offline: no provider call, no live action, and no C# compile
(the Unity half is pinned by source-contract checks against the shipped file).
"""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path
from unittest import mock

from backend.brain.loop import AutonomousBrain, BrainConfig, _bounded_token
from backend.brain.models import (
    ActionResult,
    ActionRequest,
    ActionType,
    BrainStatus,
    ExecutionStatus,
    Goal,
    GoalStatus,
    GoalType,
    PersonalityProfile,
    PlanStep,
)
from backend.brain.planner import BrainPlanner
from backend.brain.reasoning import BrainReasoner
from backend.brain.service import _parse_result
from backend.brain.world_state import build_world_state
from backend.chat.service import ChatService, ChatServiceConfig
from backend.llm.base import LLMProviderError
from backend.llm.mock import MockLLMProvider


REPO_ROOT = Path(__file__).resolve().parents[2]
LOOP_CONTROLLER = (
    REPO_ROOT / "unity/3d_unity/Assets/Scripts/Brain/BrainLoopController.cs"
)
ACTION_BRIDGE = (
    REPO_ROOT / "unity/3d_unity/Assets/Scripts/Brain/BrainActionBridge.cs"
)

#: The identifier grammar the backend forwards on the wire.
TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9._:\-/]{1,64}$")


def world():
    return build_world_state(
        {
            "character": {"position": [0, 0, 0], "current_action": "idle"},
            "phone": {"state": "Placed", "position": [1, 0.75, 0]},
            "stream": {"is_streaming": False, "camera_available": True},
            "targets": {
                "CENTER": {"position": [0.0, 0.0, 0.0], "navigation_target": True, "look_at_target": True},
                "TABLE": {"position": [1.0, 0.0, 0.0], "navigation_target": True, "look_at_target": True},
                "CAMERA": {"position": [1.0, 0.8, 0.0], "navigation_target": False, "look_at_target": True},
                "CAMERA_STAND": {"position": [1.0, 0.0, 1.0], "navigation_target": True, "look_at_target": False},
            },
        }
    )


def make_brain() -> AutonomousBrain:
    chat = ChatService(
        MockLLMProvider(), ChatServiceConfig(max_messages_per_rate_window=20)
    )
    return AutonomousBrain(
        MockLLMProvider(), chat, BrainConfig(decision_interval_seconds=0.0)
    )


def goal() -> Goal:
    return Goal("goal_x", GoalType.EXPLORE_ROOM, 0.5, "test", status=GoalStatus.ACTIVE)


def dispatch_step(
    brain: AutonomousBrain,
    action: ActionType,
    target: str | None,
    index: int = 0,
    *,
    steps: list[PlanStep] | None = None,
):
    """Drive the real dispatch point (``AutonomousBrain._begin_step``)."""
    plan_steps = steps if steps is not None else [PlanStep(action, target)]
    plan = BrainPlanner().create_plan(goal(), "test", plan_steps)
    brain.state.current_goal = goal()
    brain.state.current_plan = plan
    brain.state.step_index = index
    return brain._begin_step()


class DispatchIdGenerationTests(unittest.TestCase):
    """The backend mints exactly one id per accepted dispatch."""

    def test_new_dispatch_id_is_a_bounded_token(self):
        brain = make_brain()
        result = dispatch_step(brain, ActionType.MOVE_TO, "CAMERA_STAND")
        dispatch_id = result.action.dispatch_id
        self.assertTrue(dispatch_id)
        self.assertRegex(dispatch_id, TOKEN_PATTERN)
        self.assertEqual(_bounded_token(dispatch_id), dispatch_id)

    def test_id_is_in_the_serialized_payload_and_pending_state(self):
        brain = make_brain()
        result = dispatch_step(brain, ActionType.MOVE_TO, "CAMERA_STAND")
        payload = result.to_dict()["action"]
        self.assertEqual(payload["dispatch_id"], result.action.dispatch_id)
        # The pending action is what a later tick re-offers, so it must carry
        # the same id.
        self.assertEqual(
            brain.state.pending_action.dispatch_id, result.action.dispatch_id
        )
        # Round-trips exactly as the JSON tick response serializes it.
        self.assertEqual(
            json.loads(json.dumps(payload))["dispatch_id"],
            result.action.dispatch_id,
        )

    def test_reading_the_same_pending_request_keeps_the_id(self):
        brain = make_brain()
        first = dispatch_step(brain, ActionType.LOOK_AT, "CAMERA")
        again = brain.tick(world(), None)
        # The repeated read re-offers the same request object, so it also
        # re-offers the same dispatch id and never mints a second one.
        self.assertIs(again.action, first.action)
        self.assertEqual(again.action.dispatch_id, first.action.dispatch_id)
        self.assertEqual(again.message, "awaiting_action_result")

    def test_a_genuinely_new_dispatch_gets_a_different_id(self):
        brain = make_brain()
        first = dispatch_step(brain, ActionType.LOOK_AT, "CAMERA")
        second = dispatch_step(brain, ActionType.LOOK_AT, "CAMERA")
        self.assertNotEqual(first.action.dispatch_id, second.action.dispatch_id)

    def test_dispatch_event_carries_the_exact_payload_id(self):
        brain = make_brain()
        with mock.patch("backend.brain.loop.log_event") as spy:
            result = dispatch_step(brain, ActionType.LOOK_AT, "CAMERA")
        self.assertEqual(spy.call_count, 1)
        args, kwargs = spy.call_args
        self.assertEqual(args[1], "brain.action_dispatch_requested")
        self.assertTrue(kwargs["dispatch_id"])
        self.assertEqual(
            kwargs["dispatch_id"], result.action.to_dict()["dispatch_id"]
        )

    def test_repeat_reads_do_not_emit_a_second_dispatch_event(self):
        brain = make_brain()
        dispatch_step(brain, ActionType.LOOK_AT, "CAMERA")
        with mock.patch("backend.brain.loop.log_event") as spy:
            brain.tick(world(), None)
        self.assertEqual(spy.call_count, 0)


class ResultParsingTests(unittest.TestCase):
    """The wire parser captures only a well-formed echoed id."""

    def test_parse_preserves_a_well_formed_id(self):
        parsed = _parse_result(
            {
                "action": "LOOK_AT",
                "status": "FAILED",
                "reason": "target_unresolved",
                "dispatch_id": "dsp_abc123",
            }
        )
        self.assertEqual(parsed.dispatch_id, "dsp_abc123")
        self.assertEqual(parsed.reason, "target_unresolved")

    def test_parse_drops_a_missing_or_malformed_id(self):
        for bad in (None, "", "   ", 17, ["x"], {"a": 1}, "has space", "a" * 65, "semi;colon"):
            with self.subTest(value=bad):
                parsed = _parse_result(
                    {"action": "LOOK_AT", "status": "SUCCESS", "dispatch_id": bad}
                )
                self.assertIsNone(parsed.dispatch_id)

    def test_legacy_payload_without_the_field_still_parses(self):
        parsed = _parse_result({"action": "LOOK_AT", "status": "SUCCESS"})
        self.assertIsNone(parsed.dispatch_id)
        self.assertEqual(parsed.status, ExecutionStatus.SUCCESS)


class MatchingTests(unittest.TestCase):
    """Strict matching: id *and* action type must both match."""

    def _brain(self, dispatch_id):
        brain = make_brain()
        brain.state.pending_action = ActionRequest(
            action=ActionType.LOOK_AT, target="CAMERA", dispatch_id=dispatch_id
        )
        return brain

    def test_matching_type_and_id_is_accepted(self):
        brain = self._brain("dsp_a")
        self.assertTrue(
            brain._is_awaited_result(
                ActionResult(
                    ActionType.LOOK_AT, ExecutionStatus.SUCCESS, dispatch_id="dsp_a"
                )
            )
        )

    def test_same_type_different_id_is_rejected(self):
        brain = self._brain("dsp_new")
        self.assertFalse(
            brain._is_awaited_result(
                ActionResult(
                    ActionType.LOOK_AT, ExecutionStatus.SUCCESS, dispatch_id="dsp_old"
                )
            )
        )

    def test_correct_id_wrong_type_is_rejected(self):
        brain = self._brain("dsp_a")
        self.assertFalse(
            brain._is_awaited_result(
                ActionResult(
                    ActionType.MOVE_TO, ExecutionStatus.SUCCESS, dispatch_id="dsp_a"
                )
            )
        )

    def test_missing_id_is_rejected_for_an_id_bearing_pending_action(self):
        brain = self._brain("dsp_a")
        for missing in (None, "", "   "):
            with self.subTest(missing=missing):
                self.assertFalse(
                    brain._is_awaited_result(
                        ActionResult(
                            ActionType.LOOK_AT,
                            ExecutionStatus.SUCCESS,
                            dispatch_id=missing,
                        )
                    )
                )

    def test_legacy_pending_without_id_keeps_action_type_matching(self):
        brain = make_brain()
        brain.state.pending_action = ActionRequest(
            action=ActionType.LOOK_AT, target="CAMERA"
        )
        self.assertIsNone(brain.state.pending_action.dispatch_id)
        self.assertTrue(
            brain._is_awaited_result(
                ActionResult(ActionType.LOOK_AT, ExecutionStatus.SUCCESS)
            )
        )


class StaleResultHandlingTests(unittest.TestCase):
    """An uncorrelated result must never disturb the live dispatch."""

    def _armed(self):
        brain = make_brain()
        first = dispatch_step(brain, ActionType.LOOK_AT, "CAMERA")
        return brain, first

    def test_stale_same_type_result_does_not_clear_pending_or_advance(self):
        brain, first = self._armed()
        sentinel = ActionResult(
            ActionType.LOOK_AT,
            ExecutionStatus.SUCCESS,
            "bridge_success",
            True,
            dispatch_id=first.action.dispatch_id,
        )
        brain.state.last_result = sentinel
        stale = ActionResult(
            ActionType.LOOK_AT,
            ExecutionStatus.SUCCESS,
            "bridge_success",
            True,
            dispatch_id="dsp_older",
        )
        with mock.patch("backend.brain.loop.log_event") as spy:
            outcome = brain.tick(world(), stale)
        self.assertIs(brain.state.pending_action, first.action)
        self.assertEqual(brain.state.step_index, 0)
        self.assertEqual(brain.state.retries, 0)
        self.assertIsNone(brain.state.last_failure)
        # The previously recorded result is not overwritten by the stale one.
        self.assertIs(brain.state.last_result, sentinel)
        # The loop keeps offering the still-pending dispatch.
        self.assertIs(outcome.action, first.action)
        # The mismatch is reported once, safely, and never with a payload body.
        self.assertEqual(spy.call_count, 1)
        self.assertEqual(spy.call_args.args[1], "brain.action_result_uncorrelated")
        self.assertEqual(spy.call_args.kwargs["action"], "LOOK_AT")

    def test_replayed_mismatch_is_logged_only_once(self):
        brain, first = self._armed()
        stale = ActionResult(
            ActionType.LOOK_AT,
            ExecutionStatus.SUCCESS,
            "bridge_success",
            dispatch_id="dsp_older",
        )
        with mock.patch("backend.brain.loop.log_event") as spy:
            brain.tick(world(), stale)
            brain.tick(world(), stale)
        self.assertEqual(spy.call_count, 1)

    def test_duplicate_after_completion_cannot_complete_the_next_dispatch(self):
        brain = make_brain()
        steps = [
            PlanStep(ActionType.LOOK_AT, "CAMERA"),
            PlanStep(ActionType.LOOK_AT, "TABLE"),
        ]
        first = dispatch_step(brain, ActionType.LOOK_AT, "CAMERA", 0, steps=steps)
        completed = brain.tick(
            world(),
            ActionResult(
                ActionType.LOOK_AT,
                ExecutionStatus.SUCCESS,
                "bridge_success",
                True,
                dispatch_id=first.action.dispatch_id,
            ),
        )
        second = completed.action
        self.assertIsNotNone(second)
        self.assertNotEqual(second.dispatch_id, first.action.dispatch_id)
        self.assertEqual(brain.state.step_index, 1)

        # Unity re-sends the first result: same action type as the new pending
        # dispatch, but the id belongs to the dispatch already completed.
        replay = ActionResult(
            ActionType.LOOK_AT,
            ExecutionStatus.SUCCESS,
            "bridge_success",
            True,
            dispatch_id=first.action.dispatch_id,
        )
        outcome = brain.tick(world(), replay)
        self.assertIs(brain.state.pending_action, second)
        self.assertEqual(brain.state.step_index, 1)
        self.assertIs(outcome.action, second)


class RegressionTests(unittest.TestCase):
    """Behaviour that must not change under the new protocol."""

    def test_matched_failure_still_retries_without_advancing(self):
        brain = make_brain()
        first = dispatch_step(brain, ActionType.LOOK_AT, "CAMERA")
        brain.tick(
            world(),
            ActionResult(
                ActionType.LOOK_AT,
                ExecutionStatus.FAILED,
                "target_unresolved",
                dispatch_id=first.action.dispatch_id,
            ),
        )
        self.assertEqual(brain.state.retries, 1)
        self.assertEqual(brain.state.status, BrainStatus.REPLANNING)
        self.assertIs(brain.state.pending_action, first.action)
        self.assertEqual(brain.state.step_index, 0)

    def test_specific_failure_reason_is_still_preserved(self):
        parsed = _parse_result(
            {
                "action": "MOVE_TO",
                "status": "FAILED",
                "reason": "move_unreachable",
                "dispatch_id": "dsp_a",
            }
        )
        self.assertEqual(parsed.reason, "move_unreachable")
        self.assertNotEqual(parsed.reason, "bridge_failed")

    def test_planner_target_validation_is_unchanged(self):
        planner = BrainPlanner()
        w = world()
        for action, name in (
            (ActionType.LOOK_AT, "CAMERA_STAND"),
            (ActionType.TURN_TO, "CAMERA_STAND"),
        ):
            with self.subTest(action=action):
                validation = planner.validate_step(PlanStep(action, name), w)
                self.assertFalse(validation.valid)
                self.assertEqual(validation.reason, "TARGET_NOT_LOOKABLE")
        self.assertTrue(
            planner.validate_step(PlanStep(ActionType.LOOK_AT, "CAMERA"), w).valid
        )
        self.assertTrue(
            planner.validate_step(PlanStep(ActionType.MOVE_TO, "CAMERA_STAND"), w).valid
        )

    def test_provider_failure_diagnostics_are_unchanged(self):
        class _Raising(MockLLMProvider):
            def generate(self, request):  # pragma: no cover - mock
                raise LLMProviderError("boom")

        reasoner = BrainReasoner(_Raising(), BrainPlanner(), None)
        with mock.patch("backend.llm.failure_diagnostics.log_event") as spy:
            plan, reason = reasoner.plan(world(), goal(), PersonalityProfile())
        self.assertIsNone(plan)
        self.assertEqual(reason, "LLMProviderError")
        self.assertEqual(spy.call_count, 1)
        self.assertEqual(spy.call_args.args[1], "brain.llm_request_failed")


class UnityContractSourceTests(unittest.TestCase):
    """The shipped C# carries the id through parse -> result."""

    def setUp(self) -> None:
        self.controller = LOOP_CONTROLLER.read_text(encoding="utf-8")
        self.bridge = ACTION_BRIDGE.read_text(encoding="utf-8")

    def test_request_and_result_types_carry_the_id(self):
        # One field on the parsed request and one on the echoed result.
        self.assertEqual(self.controller.count("public string dispatch_id;"), 2)

    def test_result_echoes_the_id_of_the_action_that_just_finished(self):
        self.assertIn("dispatch_id = action.dispatch_id", self.controller)

    def test_result_payload_is_built_by_the_single_serializer(self):
        self.assertIn(": BuildResultJson(lastResult);", self.controller)
        self.assertNotIn("JsonUtility.ToJson(lastResult)", self.controller)
        self.assertIn(
            "private static string BuildResultJson(ActionResultSnapshot snapshot)",
            self.controller,
        )
        # The id is appended only when present, so a legacy request keeps the
        # historical result shape and no id is ever invented.
        self.assertIn(
            'if (!string.IsNullOrEmpty(snapshot.dispatch_id))', self.controller
        )
        self.assertIn(
            'builder.Append(",\\"dispatch_id\\":").Append(BrainActionBridge.JsonUtilityQuote(snapshot.dispatch_id))',
            self.controller,
        )

    def test_status_and_reason_semantics_are_unchanged(self):
        self.assertIn('status = success ? "SUCCESS" : "FAILED"', self.controller)
        self.assertIn('"bridge_success"', self.controller)
        self.assertIn("ResolveFailureReason(bridge)", self.controller)
        self.assertIn("reached = success", self.controller)

    def test_bridge_source_is_untouched_by_the_protocol(self):
        # Correlation lives entirely in the loop controller: the execution
        # bridge neither sees nor forwards the id.
        self.assertNotIn("dispatch_id", self.bridge)


if __name__ == "__main__":
    unittest.main()
