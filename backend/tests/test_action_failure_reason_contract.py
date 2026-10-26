"""Focused tests for the Unity action-failure reason contract.

Background (see ``Brain_Action_Execution_Lifecycle_Audit_20261019T001054Z``):
the Unity bridge already computes bounded failure labels such as
``target_unresolved``, ``move_unreachable`` and ``turn_unreachable``, but the
result payload collapsed every failure into the constant ``bridge_failed``.
``BrainLoopController`` now forwards ``BrainActionBridge.LastObservedReason``
through the existing ``reason`` field, guarded by a small token check so that
only a short identifier-shaped value can reach the backend.

These tests pin both halves of the contract offline:

* the backend parser / failure-memory path preserves a specific reason
  verbatim and falls back to ``reported`` for a missing or non-string value;
* the shipped C# source forwards the bridge reason on failure only, keeps the
  ``bridge_success`` success branch, and bounds every reason token.

No provider call, no live action, and no C# compile are involved.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path
from unittest import mock

from backend.brain.loop import AutonomousBrain, BrainConfig
from backend.brain.models import (
    ActionResult,
    ActionRequest,
    ActionType,
    ExecutionStatus,
    Goal,
    GoalStatus,
    GoalType,
    PlanStep,
)
from backend.brain.planner import BrainPlanner
from backend.brain.service import _parse_result
from backend.chat.service import ChatService, ChatServiceConfig
from backend.llm.mock import MockLLMProvider


#: Repo root: ``<root>/backend/tests/this_file.py``.
REPO_ROOT = Path(__file__).resolve().parents[2]
LOOP_CONTROLLER = (
    REPO_ROOT / "unity/3d_unity/Assets/Scripts/Brain/BrainLoopController.cs"
)
ACTION_BRIDGE = (
    REPO_ROOT / "unity/3d_unity/Assets/Scripts/Brain/BrainActionBridge.cs"
)

#: The identifiers Unity's ``ResolveFailureReason`` is allowed to forward.
TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")

#: A representative interruption reason supplied by a caller, i.e. not one of
#: the bridge's own literals; it must survive the same bounded path.
INTERRUPTION_REASON = "interrupted"


def make_brain(memory=None) -> AutonomousBrain:
    chat = ChatService(
        MockLLMProvider(), ChatServiceConfig(max_messages_per_rate_window=20)
    )
    return AutonomousBrain(
        MockLLMProvider(),
        chat,
        BrainConfig(decision_interval_seconds=0.0),
        memory_service=memory,
    )


def goal() -> Goal:
    return Goal("goal_x", GoalType.EXPLORE_ROOM, 0.5, "test", status=GoalStatus.ACTIVE)


def unity_resolve_failure_reason(reason: object, limit: int = 64) -> str:
    """Python mirror of ``BrainLoopController.ResolveFailureReason``.

    Kept in lock-step with the C# by ``UnityFailureReasonSourceTests`` below,
    which assert the shipped source still carries the same length bound,
    character allow-list and ``bridge_failed`` fallback.
    """
    if not isinstance(reason, str) or not reason.strip():
        return "bridge_failed"
    trimmed = reason.strip()
    if len(trimmed) > limit:
        return "bridge_failed"
    for character in trimmed:
        if not (
            "a" <= character <= "z"
            or "A" <= character <= "Z"
            or "0" <= character <= "9"
            or character in "_.-"
        ):
            return "bridge_failed"
    return trimmed


class BackendParsePreservationTests(unittest.TestCase):
    """A specific Unity reason reaches the parser unchanged."""

    def test_named_failure_reasons_survive_the_parser(self):
        cases = (
            (ActionType.LOOK_AT, "target_unresolved"),
            (ActionType.MOVE_TO, "move_unreachable"),
            (ActionType.TURN_TO, "turn_unreachable"),
            (ActionType.LOOK_AT, INTERRUPTION_REASON),
        )
        for action, reason in cases:
            with self.subTest(action=action, reason=reason):
                parsed = _parse_result(
                    {"action": action.value, "status": "FAILED", "reason": reason}
                )
                self.assertIsNotNone(parsed)
                self.assertEqual(parsed.action, action)
                self.assertEqual(parsed.status, ExecutionStatus.FAILED)
                self.assertEqual(parsed.reason, reason)
                self.assertNotEqual(parsed.reason, "bridge_failed")

    def test_missing_reason_uses_the_existing_fallback(self):
        parsed = _parse_result({"action": "LOOK_AT", "status": "FAILED"})
        self.assertEqual(parsed.reason, "reported")

    def test_non_string_reason_falls_back_safely(self):
        for bad in (123, None, ["x"], {"reason": "y"}, True):
            with self.subTest(value=bad):
                parsed = _parse_result(
                    {"action": "MOVE_TO", "status": "FAILED", "reason": bad}
                )
                self.assertEqual(parsed.reason, "reported")

    def test_success_result_is_unchanged(self):
        parsed = _parse_result(
            {"action": "LOOK_AT", "status": "SUCCESS", "reason": "bridge_success"}
        )
        self.assertEqual(parsed.status, ExecutionStatus.SUCCESS)
        self.assertEqual(parsed.reason, "bridge_success")

    def test_serialised_result_round_trips_the_reason(self):
        # The parser reads the same ``reason`` key that ActionResult.to_dict()
        # emits, so a reason can never be dropped between the two.
        original = ActionResult(
            action=ActionType.MOVE_TO,
            status=ExecutionStatus.FAILED,
            reason="move_unreachable",
        )
        parsed = _parse_result(original.to_dict())
        self.assertEqual(parsed.reason, "move_unreachable")


class BackendFailureMemoryTests(unittest.TestCase):
    """The specific reason is what gets remembered, not ``bridge_failed``."""

    def test_reason_is_preserved_in_the_persisted_event(self):
        memory = mock.MagicMock()
        brain = make_brain(memory=memory)
        result = ActionResult(
            action=ActionType.MOVE_TO,
            status=ExecutionStatus.FAILED,
            reason="move_unreachable",
        )
        brain._remember_failed_action(result)
        memory.remember_event.assert_called_once()
        text = memory.remember_event.call_args.args[0]
        self.assertEqual(text, "MOVE_TO FAILED: move_unreachable")
        self.assertNotIn("bridge_failed", text)

    def test_each_named_reason_reaches_the_memory_text(self):
        for action, reason in (
            (ActionType.LOOK_AT, "target_unresolved"),
            (ActionType.TURN_TO, "turn_unreachable"),
            (ActionType.LOOK_AT, INTERRUPTION_REASON),
        ):
            with self.subTest(reason=reason):
                memory = mock.MagicMock()
                brain = make_brain(memory=memory)
                brain._remember_failed_action(
                    ActionResult(
                        action=action,
                        status=ExecutionStatus.FAILED,
                        reason=reason,
                    )
                )
                text = memory.remember_event.call_args.args[0]
                self.assertTrue(text.endswith(reason))

    def test_no_memory_service_is_a_no_op(self):
        brain = make_brain(memory=None)
        # Must not raise and must not attempt any persistence.
        brain._remember_failed_action(
            ActionResult(
                action=ActionType.MOVE_TO,
                status=ExecutionStatus.FAILED,
                reason="move_unreachable",
            )
        )


class MatchingAndObserveTests(unittest.TestCase):
    """Result matching, pending-action release and success are unchanged."""

    def test_matching_is_still_by_action_type(self):
        brain = make_brain()
        brain.state.pending_action = ActionRequest(
            action=ActionType.LOOK_AT, target="CAMERA"
        )
        self.assertTrue(
            brain._is_awaited_result(
                ActionResult(
                    action=ActionType.LOOK_AT,
                    status=ExecutionStatus.FAILED,
                    reason="target_unresolved",
                )
            )
        )
        self.assertFalse(
            brain._is_awaited_result(
                ActionResult(
                    action=ActionType.MOVE_TO,
                    status=ExecutionStatus.FAILED,
                    reason="move_unreachable",
                )
            )
        )

    def _arm_plain_plan(self, brain: AutonomousBrain, action: ActionType, target: str):
        plan = BrainPlanner().create_plan(goal(), "test", [PlanStep(action, target)])
        brain.state.current_goal = goal()
        brain.state.current_plan = plan
        brain.state.step_index = 0
        brain.state.pending_action = ActionRequest(action=action, target=target)

    def test_success_still_clears_pending_and_advances(self):
        brain = make_brain()
        self._arm_plain_plan(brain, ActionType.LOOK_AT, "CAMERA")
        brain._observe_result(
            ActionResult(
                action=ActionType.LOOK_AT,
                status=ExecutionStatus.SUCCESS,
                reason="bridge_success",
                reached=True,
            )
        )
        self.assertIsNone(brain.state.pending_action)
        self.assertEqual(brain.state.step_index, 1)

    def test_failure_reason_reaches_last_result(self):
        brain = make_brain()
        self._arm_plain_plan(brain, ActionType.MOVE_TO, "CAMERA_STAND")
        brain._observe_result(
            ActionResult(
                action=ActionType.MOVE_TO,
                status=ExecutionStatus.FAILED,
                reason="move_unreachable",
            )
        )
        self.assertEqual(brain.state.last_result.reason, "move_unreachable")


class UnityTokenGuardTests(unittest.TestCase):
    """The bounded-token guard admits labels and rejects anything else."""

    def test_bridge_labels_are_admitted(self):
        for token in (
            "target_unresolved",
            "move_unreachable",
            "turn_unreachable",
            "generated_motion_interrupted",
            INTERRUPTION_REASON,
            "none",
        ):
            with self.subTest(token=token):
                self.assertEqual(unity_resolve_failure_reason(token), token)

    def test_unsafe_values_collapse_to_the_fallback(self):
        secret = "Bearer sk-live-SECRETVALUE0123456789"
        for bad in (
            secret,
            "has spaces",
            "semi;colon",
            "new\nline",
            "a" * 65,
            "",
            "   ",
            None,
            17,
            ["x"],
        ):
            with self.subTest(value=bad):
                resolved = unity_resolve_failure_reason(bad)
                self.assertEqual(resolved, "bridge_failed")
                self.assertNotIn("sk-live", resolved)


class UnityFailureReasonSourceTests(unittest.TestCase):
    """The shipped C# actually wires the guard into the result payload."""

    def setUp(self) -> None:
        self.controller = LOOP_CONTROLLER.read_text(encoding="utf-8")
        self.bridge = ACTION_BRIDGE.read_text(encoding="utf-8")

    def test_failure_branch_forwards_the_bridge_reason(self):
        self.assertIn(
            'reason = success ? "bridge_success" : ResolveFailureReason(bridge)',
            self.controller,
        )

    def test_success_branch_and_status_fields_are_unchanged(self):
        self.assertIn('status = success ? "SUCCESS" : "FAILED"', self.controller)
        self.assertIn('"bridge_success"', self.controller)
        # ``reached`` still tracks success exactly as before.
        self.assertIn("reached = success", self.controller)

    def test_guard_is_a_bounded_identifier_check(self):
        self.assertIn("MaxFailureReasonLength = 64", self.controller)
        self.assertIn("private static string ResolveFailureReason(", self.controller)
        self.assertIn("source.LastObservedReason", self.controller)
        self.assertIn('return "bridge_failed";', self.controller)
        for fragment in (
            "character >= 'a' && character <= 'z'",
            "character >= 'A' && character <= 'Z'",
            "character >= '0' && character <= '9'",
            "character == '_'",
            "character == '-'",
            "character == '.'",
        ):
            self.assertIn(fragment, self.controller)

    def test_bridge_exposes_last_observed_reason_before_the_callback(self):
        self.assertIn("public string LastObservedReason => lastObservedReason;", self.bridge)
        # The property is written by Complete() before the callback that reads
        # it fires, so it always describes the action that just finished.
        written_at = self.bridge.index('lastObservedReason = operation ?? "none";')
        callback_at = self.bridge.index("callback?.Invoke(success);", written_at)
        self.assertGreater(callback_at, written_at)

    def test_every_bridge_label_is_a_canonical_token(self):
        labels = re.findall(
            r'Complete\(\s*\w+\s*,\s*(?:true|false)\s*,\s*"([^"]*)"\s*\)',
            self.bridge,
        )
        self.assertGreater(len(labels), 30)
        for label in labels:
            with self.subTest(label=label):
                self.assertRegex(label, TOKEN_PATTERN)

    def test_interruption_token_flows_through_the_same_guard(self):
        # A caller-supplied interruption reason is completed with the same
        # pipeline; the guard in the controller bounds it.
        self.assertIn(
            'interruptReason = string.IsNullOrWhiteSpace(reason) ? "interrupted" : reason;',
            self.bridge,
        )
        self.assertIn("Complete(callback, false, interruptReason);", self.bridge)
        self.assertEqual(unity_resolve_failure_reason("interrupted"), "interrupted")

    def test_wrist_rotation_edits_are_still_present(self):
        # Cheap sentinels that the unrelated uncommitted wrist work was not
        # clobbered by the reason-propagation edit.
        self.assertIn("heldSelfieHandTarget", self.bridge)
        self.assertIn("HeldSelfiePoseActive", self.bridge)


if __name__ == "__main__":
    unittest.main()
