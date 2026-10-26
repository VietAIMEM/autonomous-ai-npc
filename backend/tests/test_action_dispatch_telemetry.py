"""Focused tests for target-aware action dispatch telemetry.

Background (see ``Brain_Target_Constraints_Runtime_Verification_20261018T224314Z``):
an accepted ``LOOK_AT`` was dispatched during the verification window but its
target was never recorded, so the prompt fix could not be behaviourally
confirmed. These tests pin the two additions that close that gap:

* ``brain.action_dispatch_requested`` — emitted at the single point where an
  accepted action becomes the tick-response payload handed to the execution
  bridge, carrying the canonical target.
* ``brain.planner_output_rejected`` now also names the exact step the validator
  refused.

Everything here is mocked and offline; no provider call is made.
"""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest import mock

from backend.brain.loop import AutonomousBrain, BrainConfig
from backend.brain.models import (
    ActionRequest,
    ActionType,
    Goal,
    GoalStatus,
    GoalType,
    PersonalityProfile,
    PlanStep,
)
from backend.brain.planner import BrainPlanner
from backend.brain.reasoning import BrainReasoner
from backend.brain.world_state import build_world_state
from backend.chat.service import ChatService, ChatServiceConfig
from backend.llm.base import LLMProvider, LLMProviderError, LLMRequest
from backend.llm.mock import MockLLMProvider


def target(name: str, navigable: bool, lookable: bool) -> dict:
    return {
        "position": [0.0, 0.0, 0.0],
        "navigation_target": navigable,
        "look_at_target": lookable,
    }


#: Production-shaped flags: standing points navigate, device anchors look.
PRODUCTION_TARGETS = {
    "CENTER": target("CENTER", True, True),
    "TABLE": target("TABLE", True, True),
    "BENCH": target("BENCH", True, True),
    "CAMERA": target("CAMERA", False, True),
    "PHONE": target("PHONE", False, True),
    "CAMERA_STAND": target("CAMERA_STAND", True, False),
    "PHONE_STAND": target("PHONE_STAND", True, False),
}


def world():
    return build_world_state(
        {
            "character": {"position": [0, 0, 0], "current_action": "idle"},
            "phone": {"state": "Placed", "position": [1, 0.75, 0]},
            "stream": {"is_streaming": False, "camera_available": True},
            "targets": PRODUCTION_TARGETS,
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


def begin_step(
    brain: AutonomousBrain,
    action: ActionType,
    target_name: str | None,
    index: int = 0,
    *,
    steps: list[PlanStep] | None = None,
):
    """Drive the real dispatch point for one accepted step."""
    selected = goal()
    plan_steps = steps if steps is not None else [PlanStep(action, target_name)]
    plan = BrainPlanner().create_plan(selected, "test", plan_steps)
    brain.state.current_goal = selected
    brain.state.current_plan = plan
    brain.state.step_index = index
    return brain._begin_step()


class _ScriptedProvider(LLMProvider):
    """Returns a fixed raw plan; never touches the network."""

    name = "scripted"

    def __init__(self, payload: str) -> None:
        self._payload = payload
        self.config = SimpleNamespace(model="test/model-1")

    def generate(self, request: LLMRequest) -> str:  # pragma: no cover - mock
        return self._payload


class DispatchEventTests(unittest.TestCase):
    def setUp(self) -> None:
        self.brain = make_brain()

    def _dispatch(
        self,
        action: ActionType,
        target_name: str | None,
        index: int = 0,
        *,
        steps: list[PlanStep] | None = None,
    ):
        with mock.patch("backend.brain.loop.log_event") as spy:
            result = begin_step(self.brain, action, target_name, index, steps=steps)
        self.assertEqual(spy.call_count, 1)
        args, kwargs = spy.call_args
        self.assertEqual(args[1], "brain.action_dispatch_requested")
        return result, kwargs

    def test_look_at_dispatch_logs_canonical_target_matching_payload(self):
        result, kwargs = self._dispatch(ActionType.LOOK_AT, "CAMERA")
        # The event target is proven identical to the dispatched bridge payload
        # (``ActionRequest.to_dict()`` is what the tick response serializes).
        self.assertEqual(
            kwargs["target"], result.action.to_dict()["target"]
        )
        self.assertEqual(kwargs["target"], "CAMERA")
        self.assertEqual(kwargs["action"], "LOOK_AT")
        self.assertEqual(kwargs["dispatch"], "tick_response")
        self.assertEqual(kwargs["goal_id"], "goal_x")
        self.assertEqual(kwargs["step_index"], 0)
        self.assertIsInstance(kwargs["plan_id"], str)

    def test_turn_to_dispatch_records_its_target(self):
        result, kwargs = self._dispatch(ActionType.TURN_TO, "BENCH")
        self.assertEqual(result.action.action, ActionType.TURN_TO)
        self.assertEqual(kwargs["target"], "BENCH")
        self.assertEqual(kwargs["target"], result.action.to_dict()["target"])

    def test_move_to_dispatch_uses_the_navigation_target(self):
        # CAMERA_STAND is navigable but *not* lookable; the logged target must be
        # the navigation destination actually sent, never the lookable CAMERA.
        steps = [
            PlanStep(ActionType.LOOK_AT, "TABLE"),
            PlanStep(ActionType.LOOK_AT, "BENCH"),
            PlanStep(ActionType.MOVE_TO, "CAMERA_STAND"),
        ]
        result, kwargs = self._dispatch(
            ActionType.MOVE_TO, "CAMERA_STAND", index=2, steps=steps
        )
        self.assertEqual(kwargs["action"], "MOVE_TO")
        self.assertEqual(kwargs["target"], "CAMERA_STAND")
        self.assertNotEqual(kwargs["target"], "CAMERA")
        self.assertEqual(kwargs["target"], result.action.to_dict()["target"])
        self.assertEqual(kwargs["step_index"], 2)

    def test_non_target_action_is_not_instrumented(self):
        with mock.patch("backend.brain.loop.log_event") as spy:
            begin_step(self.brain, ActionType.STOP, None)
        self.assertEqual(spy.call_count, 0)

    def test_dispatch_state_is_set_even_when_logging_fails(self):
        with mock.patch(
            "backend.brain.loop.log_event", side_effect=RuntimeError("logger exploded")
        ):
            result = begin_step(self.brain, ActionType.LOOK_AT, "TABLE")
        self.assertEqual(result.message, "action_started")
        self.assertEqual(result.action.target, "TABLE")
        self.assertIsNotNone(self.brain.state.pending_action)
        self.assertEqual(self.brain.state.pending_action.target, "TABLE")

    def test_missing_correlation_fields_are_omitted(self):
        self.brain.state.current_goal = None
        self.brain._decision_source = None
        request = ActionRequest(action=ActionType.LOOK_AT, target="CAMERA")
        with mock.patch("backend.brain.loop.log_event") as spy:
            self.brain._log_action_dispatch(request, None)
        kwargs = spy.call_args.kwargs
        self.assertEqual(kwargs["target"], "CAMERA")
        for omitted in ("goal_id", "plan_id", "decision_source"):
            self.assertNotIn(omitted, kwargs)

    def test_unsafe_target_values_never_reach_the_log(self):
        secret = "Bearer sk-live-SECRETVALUE0123456789 with spaces"
        for bad in (secret, "x" * 200, "", "   ", None, 17, {"a": 1}):
            with self.subTest(value=bad):
                request = ActionRequest(action=ActionType.LOOK_AT, target=bad)
                with mock.patch("backend.brain.loop.log_event") as spy:
                    self.brain._log_action_dispatch(request, None)
                kwargs = spy.call_args.kwargs
                self.assertNotIn("target", kwargs)
                self.assertNotIn("sk-live", json.dumps(kwargs))

    def test_bounded_token_helper_rules(self):
        from backend.brain.loop import _bounded_token

        self.assertEqual(_bounded_token("CAMERA_STAND"), "CAMERA_STAND")
        self.assertEqual(_bounded_token("Assets/a/b.fbx"), "Assets/a/b.fbx")
        for bad in (None, 17, "", "   ", "has space", "a" * 65, "semi;colon"):
            with self.subTest(value=bad):
                self.assertIsNone(_bounded_token(bad))


class RejectionAttributionTests(unittest.TestCase):
    def _plan_from(self, payload: dict):
        reasoner = BrainReasoner(_ScriptedProvider(json.dumps(payload)), BrainPlanner(), None)
        with mock.patch("backend.brain.reasoning.log_event") as log_spy:
            plan, reason = reasoner.plan(world(), goal(), PersonalityProfile())
        return plan, reason, log_spy

    def test_rejected_plan_emits_no_dispatch_event(self):
        payload = {
            "reason": "look at the camera stand",
            "steps": [
                {"action": "MOVE_TO", "target": "CENTER"},
                {"action": "LOOK_AT", "target": "CAMERA_STAND"},
            ],
        }
        reasoner = BrainReasoner(_ScriptedProvider(json.dumps(payload)), BrainPlanner(), None)
        with mock.patch("backend.brain.loop.log_event") as dispatch_spy:
            plan, reason = reasoner.plan(world(), goal(), PersonalityProfile())
        self.assertIsNone(plan)
        self.assertEqual(reason, "TARGET_NOT_LOOKABLE")
        # A rejected plan never reaches the dispatch boundary.
        self.assertEqual(dispatch_spy.call_count, 0)

    def test_rejection_event_names_the_failing_action_and_target(self):
        payload = {
            "reason": "look at the camera stand",
            "steps": [
                {"action": "MOVE_TO", "target": "CENTER"},
                {"action": "LOOK_AT", "target": "CAMERA_STAND"},
            ],
        }
        plan, reason, log_spy = self._plan_from(payload)
        self.assertIsNone(plan)
        self.assertEqual(reason, "TARGET_NOT_LOOKABLE")
        self.assertEqual(log_spy.call_count, 1)
        args, kwargs = log_spy.call_args
        self.assertEqual(args[1], "brain.planner_output_rejected")
        self.assertEqual(kwargs["reason"], "TARGET_NOT_LOOKABLE")
        self.assertEqual(kwargs["rejected_action"], "LOOK_AT")
        self.assertEqual(kwargs["rejected_target"], "CAMERA_STAND")
        self.assertEqual(kwargs["rejected_step_index"], 1)

    def test_parsed_but_invalid_plan_leaves_no_rejection_context(self):
        planner = BrainPlanner()
        raw = json.dumps(
            {"reason": "ok", "steps": [{"action": "LOOK_AT", "target": "CAMERA"}]}
        )
        plan, reason = planner.parse_plan(raw, goal(), world())
        self.assertIsNotNone(plan)
        self.assertEqual(reason, "OK")
        self.assertIsNone(planner.last_rejection_context)

    def test_odd_target_in_rejection_context_is_dropped(self):
        from backend.brain.reasoning import _log_rejected_output

        secret = "Bearer sk-live-SECRET"
        with mock.patch("backend.brain.reasoning.log_event") as spy:
            _log_rejected_output(
                "{}",
                "TARGET_NOT_LOOKABLE",
                {"action": "LOOK_AT", "target": secret, "step_index": 3},
            )
        kwargs = spy.call_args.kwargs
        self.assertEqual(kwargs["rejected_action"], "LOOK_AT")
        self.assertEqual(kwargs["rejected_step_index"], 3)
        self.assertNotIn("rejected_target", kwargs)
        self.assertNotIn("sk-live", json.dumps(kwargs))

    def test_missing_context_keeps_the_previous_event_shape(self):
        from backend.brain.reasoning import _log_rejected_output

        with mock.patch("backend.brain.reasoning.log_event") as spy:
            _log_rejected_output("{}", "EMPTY_PLAN")
        kwargs = spy.call_args.kwargs
        self.assertEqual(set(kwargs), {"reason", "output"})


class UnknownActionAttributionTests(unittest.TestCase):
    """Enum-resolution refusals name the exact token the model emitted.

    PHASE 38: ``brain.planner_output_rejected`` previously carried the bounded
    step attribution only for *validator* refusals, so an unsupported action
    token had to be recovered from the bounded ``output`` preview during an
    audit. These tests pin the attribution now supplied by the enum-resolution
    failure path itself.
    """

    def _plan_from(self, payload: dict):
        reasoner = BrainReasoner(_ScriptedProvider(json.dumps(payload)), BrainPlanner(), None)
        with mock.patch("backend.brain.reasoning.log_event") as log_spy:
            plan, reason = reasoner.plan(world(), goal(), PersonalityProfile())
        return plan, reason, log_spy

    def test_unsupported_action_names_the_token(self):
        plan, reason, log_spy = self._plan_from(
            {"reason": "nothing to do", "steps": [{"action": "WAIT", "text": None}]}
        )
        self.assertIsNone(plan)
        self.assertEqual(reason, "UNKNOWN_ACTION")
        kwargs = log_spy.call_args.kwargs
        self.assertEqual(log_spy.call_args.args[1], "brain.planner_output_rejected")
        self.assertEqual(kwargs["reason"], "UNKNOWN_ACTION")
        self.assertEqual(kwargs["rejected_action"], "WAIT")
        self.assertEqual(kwargs["rejected_step_index"], 0)
        self.assertNotIn("rejected_target", kwargs)
        self.assertEqual(
            set(kwargs), {"reason", "output", "rejected_action", "rejected_step_index"}
        )

    def test_original_casing_is_preserved_in_the_diagnostics(self):
        for token in ("wait", "Wait", "wAiT"):
            with self.subTest(token=token):
                _, reason, log_spy = self._plan_from(
                    {"reason": "idle", "steps": [{"action": token}]}
                )
                self.assertEqual(reason, "UNKNOWN_ACTION")
                self.assertEqual(log_spy.call_args.kwargs["rejected_action"], token)

    def test_failing_step_index_points_at_the_offending_step(self):
        _, reason, log_spy = self._plan_from(
            {
                "reason": "two steps",
                "steps": [
                    {"action": "MOVE_TO", "target": "CENTER"},
                    {"action": "WAIT"},
                ],
            }
        )
        self.assertEqual(reason, "UNKNOWN_ACTION")
        kwargs = log_spy.call_args.kwargs
        self.assertEqual(kwargs["rejected_step_index"], 1)
        self.assertEqual(kwargs["rejected_action"], "WAIT")

    def test_malformed_action_values_are_dropped_not_stringified(self):
        # ``str(None)`` is "None" and ``str(17)`` is "17", both of which would
        # satisfy the bounded-token pattern. Neither may be reported as a token
        # the model named.
        for bad in (None, 17, 3.5, True, ["WAIT"], {"action": "WAIT"}, ""):
            with self.subTest(value=bad):
                _, reason, log_spy = self._plan_from(
                    {"reason": "bad", "steps": [{"action": bad}]}
                )
                self.assertEqual(reason, "UNKNOWN_ACTION")
                self.assertNotIn("rejected_action", log_spy.call_args.kwargs)

    def test_missing_action_field_is_dropped(self):
        _, reason, log_spy = self._plan_from(
            {"reason": "bad", "steps": [{"target": "CENTER"}]}
        )
        self.assertEqual(reason, "UNKNOWN_ACTION")
        self.assertNotIn("rejected_action", log_spy.call_args.kwargs)

    def test_oversized_and_ill_formed_tokens_are_dropped_not_truncated(self):
        for bad in ("W" * 65, "a" * 200, "WAIT NOW", "wait;drop-table", "look at the camera"):
            with self.subTest(value=bad):
                _, reason, log_spy = self._plan_from(
                    {"reason": "bad", "steps": [{"action": bad}]}
                )
                self.assertEqual(reason, "UNKNOWN_ACTION")
                kwargs = log_spy.call_args.kwargs
                self.assertNotIn("rejected_action", kwargs)
                # Dropped, never truncated to a prefix of the original token.
                self.assertLessEqual(len(kwargs["output"]), 1200)

    def test_supported_lowercase_action_still_normalizes_and_is_accepted(self):
        planner = BrainPlanner()
        plan, reason = planner.parse_plan(
            json.dumps({"reason": "ok", "steps": [{"action": "look_at", "target": "CAMERA"}]}),
            goal(),
            world(),
        )
        self.assertIsNotNone(plan)
        self.assertEqual(reason, "OK")
        self.assertIsNone(planner.last_rejection_context)

    def test_unsupported_action_never_reaches_the_dispatch_boundary(self):
        reasoner = BrainReasoner(
            _ScriptedProvider(json.dumps({"reason": "idle", "steps": [{"action": "WAIT"}]})),
            BrainPlanner(),
            None,
        )
        with mock.patch("backend.brain.loop.log_event") as dispatch_spy:
            plan, reason = reasoner.plan(world(), goal(), PersonalityProfile())
        self.assertIsNone(plan)
        self.assertEqual(reason, "UNKNOWN_ACTION")
        self.assertEqual(dispatch_spy.call_count, 0)

    def test_context_never_leaks_into_a_later_context_free_rejection(self):
        planner = BrainPlanner()
        planner.parse_plan(
            json.dumps({"reason": "x", "steps": [{"action": "WAIT"}]}), goal(), world()
        )
        self.assertIsNotNone(planner.last_rejection_context)
        plan, reason = planner.parse_plan("not json at all", goal(), world())
        self.assertIsNone(plan)
        self.assertEqual(reason, "INVALID_JSON")
        self.assertIsNone(planner.last_rejection_context)

    def test_unsupported_action_still_reports_the_unchanged_reason_code(self):
        planner = BrainPlanner()
        plan, reason = planner.parse_plan(
            json.dumps({"reason": "x", "steps": [{"action": "WAIT"}]}), goal(), world()
        )
        self.assertIsNone(plan)
        self.assertEqual(reason, "UNKNOWN_ACTION")
        self.assertEqual(planner.last_rejection_context["reason"], "UNKNOWN_ACTION")


class UnchangedBehaviourTests(unittest.TestCase):
    def test_validator_still_rejects_standing_points(self):
        planner = BrainPlanner()
        w = world()
        for action, name in (
            (ActionType.LOOK_AT, "CAMERA_STAND"),
            (ActionType.TURN_TO, "PHONE_STAND"),
        ):
            with self.subTest(action=action, target=name):
                validation = planner.validate_step(PlanStep(action, name), w)
                self.assertFalse(validation.valid)
                self.assertEqual(validation.reason, "TARGET_NOT_LOOKABLE")
        # ...and still accepts the lookable counterparts and navigable anchors.
        self.assertTrue(planner.validate_step(PlanStep(ActionType.LOOK_AT, "CAMERA"), w).valid)
        self.assertTrue(
            planner.validate_step(PlanStep(ActionType.MOVE_TO, "PHONE_STAND"), w).valid
        )

    def test_provider_diagnostics_and_last_failure_are_unchanged(self):
        class _Raising(LLMProvider):
            name = "test-provider"

            def __init__(self) -> None:
                self.config = SimpleNamespace(model="vendor/model-1")

            def generate(self, request: LLMRequest) -> str:  # pragma: no cover - mock
                raise LLMProviderError("boom")

        reasoner = BrainReasoner(_Raising(), BrainPlanner(), None)
        with mock.patch("backend.llm.failure_diagnostics.log_event") as spy:
            plan, reason = reasoner.plan(world(), goal(), PersonalityProfile())
        self.assertIsNone(plan)
        self.assertEqual(reason, "LLMProviderError")
        self.assertEqual(f"planning_failed:{reason}", "planning_failed:LLMProviderError")
        self.assertEqual(spy.call_count, 1)
        self.assertEqual(spy.call_args.args[1], "brain.llm_request_failed")


if __name__ == "__main__":
    unittest.main()
