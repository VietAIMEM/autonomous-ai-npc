"""Focused tests for case-insensitive action-name resolution in the planner.

Background (see ``Nemotron_Parse_Rejection_Audit_20261019T201623Z``): four of
the five planner rejections during the Nemotron Super production trial were
``UNKNOWN_ACTION`` on payloads that were otherwise valid JSON and used
*supported* action names - spelled in the lowercase snake_case style the
prompt's ``available_skills`` list uses (``look_at``, ``speak``, ``move_to``,
``turn_to``). ``parse_plan`` resolved actions with an exact, case-sensitive
``ActionType(str(action_name))`` lookup, so only the spelling failed.

The parser now normalises the token before the lookup. These tests pin that the
normalisation is *spelling only*: the ``ActionType`` enum stays the
authoritative vocabulary, ``validate_step`` and every failure code are
unchanged, and no alias or fallback action was introduced.

All tests are deterministic and offline; no provider call is made.
"""

from __future__ import annotations

import json
import types
import unittest
from pathlib import Path
from unittest import mock

from backend.brain.models import (
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
from backend.llm.base import LLMProvider, LLMProviderError, LLMRequest


REPO_ROOT = Path(__file__).resolve().parents[2]
PLANNER_SOURCE = REPO_ROOT / "backend/brain/planner.py"


def target(navigable: bool, lookable: bool) -> dict:
    return {
        "position": [0.0, 0.0, 0.0],
        "navigation_target": navigable,
        "look_at_target": lookable,
    }


#: Production-shaped flags: standing points navigate, device anchors look.
PRODUCTION_TARGETS = {
    "CENTER": target(True, True),
    "TABLE": target(True, True),
    "BENCH": target(True, True),
    "FLOOR": target(True, True),
    "PHONE": target(False, True),
    "CAMERA": target(False, True),
    "CAMERA_STAND": target(True, False),
    "PHONE_STAND": target(True, False),
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


def goal() -> Goal:
    return Goal("goal", GoalType.EXPLORE_ROOM, 0.5, "test", status=GoalStatus.ACTIVE)


def raw_plan(steps: list, reason: str = "test plan") -> str:
    return json.dumps({"reason": reason, "steps": steps})


def parse(steps: list):
    return BrainPlanner().parse_plan(raw_plan(steps), goal(), world())


class ActionCaseResolutionTests(unittest.TestCase):
    def test_supported_uppercase_action_is_still_accepted(self):
        plan, reason = parse([{"action": "LOOK_AT", "target": "CAMERA"}])
        self.assertIsNotNone(plan)
        self.assertEqual(reason, "OK")
        self.assertEqual(plan.steps[0].action, ActionType.LOOK_AT)

    def test_supported_lowercase_action_resolves_to_the_same_enum(self):
        for spelling in ("look_at", "look_AT", "Look_At", "LOOK_at", "lOoK_aT"):
            with self.subTest(action=spelling):
                plan, reason = parse([{"action": spelling, "target": "CAMERA"}])
                self.assertIsNotNone(plan, reason)
                self.assertEqual(reason, "OK")
                self.assertEqual(plan.steps[0].action, ActionType.LOOK_AT)
                # Identical downstream value, not a new/aliased member.
                self.assertIs(plan.steps[0].action, ActionType.LOOK_AT)

    def test_surrounding_whitespace_is_handled(self):
        cases = (
            ("  MOVE_TO  ", "MOVE_TO", {"target": "CENTER"}),
            ("\tmove_to\n", "MOVE_TO", {"target": "CENTER"}),
            ("  speak  ", "SPEAK", {"text": "Hello there."}),
            (" look_at ", "LOOK_AT", {"target": "CAMERA"}),
            ("  turn_to  ", "TURN_TO", {"target": "TABLE"}),
        )
        for spelling, expected, extra in cases:
            with self.subTest(action=spelling):
                payload: dict = {"action": spelling, **extra}
                plan, reason = parse([payload])
                self.assertIsNotNone(plan, reason)
                self.assertEqual(plan.steps[0].action.value, expected)

    def test_every_supported_action_resolves_from_its_lowercase_spelling(self):
        self.assertEqual(len(list(ActionType)), 13)
        for member in ActionType:
            with self.subTest(action=member.value):
                payload = {"action": member.value.lower()}
                if member == ActionType.MOVE_TO:
                    payload["target"] = "CENTER"
                elif member in (ActionType.LOOK_AT, ActionType.TURN_TO):
                    payload["target"] = "CAMERA"
                plan, reason = parse([payload])
                # A missing required target may still be refused later, but the
                # action itself must always resolve to the right enum member.
                if plan is not None:
                    self.assertEqual(plan.steps[0].action, member)
                else:
                    self.assertNotEqual(reason, "UNKNOWN_ACTION")

    def test_the_enum_remains_the_authoritative_vocabulary(self):
        # No alias table, no mapping to a fallback action: the supported set is
        # exactly the enum, so an unsupported name still fails.
        parsed = []
        for member in ActionType:
            plan, reason = parse([{"action": member.value.lower()}])
            if plan is not None:
                parsed.append(plan.steps[0].action)
        self.assertTrue(set(parsed).issubset(set(ActionType)))


class RejectionBehaviourTests(unittest.TestCase):
    def test_unsupported_action_names_still_return_unknown_action(self):
        for bad in (
            "fly_to",
            "look",  # abbreviated
            "lookat",  # missing separator
            "look_at_table",
            "move",  # the movement enum is MOVE_TO, not MOVE
            "walk_to",  # that is a skill id, not an action
            "speek",
            "stop_streaming",
            "dance!",  # punctuation is not stripped by normalisation
        ):
            with self.subTest(action=bad):
                plan, reason = parse([{"action": bad, "target": "CENTER"}])
                self.assertIsNone(plan)
                self.assertEqual(reason, "UNKNOWN_ACTION")

    def test_malformed_values_follow_the_existing_error_path(self):
        for bad in ("", "   ", None, 17, True, 3.5, ["LOOK_AT"], {"a": "LOOK_AT"}):
            with self.subTest(value=bad):
                plan, reason = parse([{"action": bad, "target": "CENTER"}])
                self.assertIsNone(plan)
                self.assertEqual(reason, "UNKNOWN_ACTION")

    def test_missing_action_key_still_follows_the_same_path(self):
        plan, reason = parse([{"target": "CENTER"}])
        self.assertIsNone(plan)
        self.assertEqual(reason, "UNKNOWN_ACTION")

    def test_earlier_parse_stages_are_untouched(self):
        # JSON decode and the other structural failure codes keep their order.
        planner = BrainPlanner()
        plan, reason = planner.parse_plan("not-json", goal(), world())
        self.assertIsNone(plan)
        self.assertEqual(reason, "INVALID_JSON")

        plan, reason = planner.parse_plan(json.dumps({"reason": "r", "steps": []}), goal(), world())
        self.assertIsNone(plan)
        self.assertEqual(reason, "EMPTY_PLAN")

        plan, reason = planner.parse_plan(
            json.dumps({"reason": "", "steps": [{"action": "LOOK_AT", "target": "CAMERA"}]}),
            goal(),
            world(),
        )
        self.assertIsNone(plan)
        self.assertEqual(reason, "INVALID_REASON")


class ValidationUnchangedTests(unittest.TestCase):
    def test_normalised_action_still_goes_through_validate_step(self):
        # parse_plan validates every step before returning; a look at the
        # non-lookable standing point must still be refused, whether the action
        # name was upper- or lowercase.
        for spelling in ("LOOK_AT", "look_at", "Look_At"):
            with self.subTest(action=spelling):
                plan, reason = parse([{"action": spelling, "target": "CAMERA_STAND"}])
                self.assertIsNone(plan)
                self.assertEqual(reason, "TARGET_NOT_LOOKABLE")

    def test_target_validation_is_unchanged(self):
        planner = BrainPlanner()
        w = world()
        for action, name, expected in (
            (ActionType.LOOK_AT, "CAMERA_STAND", "TARGET_NOT_LOOKABLE"),
            (ActionType.TURN_TO, "PHONE_STAND", "TARGET_NOT_LOOKABLE"),
            (ActionType.MOVE_TO, "CAMERA", "TARGET_NOT_NAVIGABLE"),
            (ActionType.LOOK_AT, "MISSING_OBJECT", "TARGET_NOT_FOUND"),
        ):
            with self.subTest(action=action, target=name):
                validation = planner.validate_step(PlanStep(action, name), w)
                self.assertFalse(validation.valid)
                self.assertEqual(validation.reason, expected)
        # ...and the valid counterparts are still accepted.
        self.assertTrue(planner.validate_step(PlanStep(ActionType.LOOK_AT, "CAMERA"), w).valid)
        self.assertTrue(planner.validate_step(PlanStep(ActionType.TURN_TO, "TABLE"), w).valid)
        self.assertTrue(planner.validate_step(PlanStep(ActionType.MOVE_TO, "PHONE_STAND"), w).valid)

    def test_mixed_spelling_plan_validates_end_to_end(self):
        plan, reason = parse(
            [
                {"action": "move_to", "target": "center"},
                {"action": "look_at", "target": "camera"},
                {"action": "TURN_TO", "target": "table"},
                {"action": "speak", "text": "Looking around."},
            ]
        )
        self.assertIsNotNone(plan, reason)
        self.assertEqual(reason, "OK")
        self.assertEqual(
            [step.action for step in plan.steps],
            [ActionType.MOVE_TO, ActionType.LOOK_AT, ActionType.TURN_TO, ActionType.SPEAK],
        )
        # Targets keep their existing case folding too.
        self.assertEqual(plan.steps[0].target, "CENTER")
        self.assertEqual(plan.steps[1].target, "CAMERA")


class SourceContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = PLANNER_SOURCE.read_text(encoding="utf-8")

    def test_action_lookup_normalises_before_the_enum(self):
        self.assertIn(
            "ActionType(str(action_name).strip().upper())", self.source
        )
        self.assertNotIn("action = ActionType(str(action_name))", self.source)

    def test_no_alias_table_or_fallback_was_added(self):
        for forbidden in ("ACTION_ALIASES", "FALLBACK_ACTION", "_ALIASES"):
            self.assertNotIn(forbidden, self.source)


class _ScriptedProvider(LLMProvider):
    """Returns a fixed raw plan; never touches the network."""

    name = "scripted"

    def __init__(self, payload: str) -> None:
        self._payload = payload
        self.config = types.SimpleNamespace(model="test/model-1")

    def generate(self, request: LLMRequest) -> str:  # pragma: no cover - mock
        return self._payload


class DownstreamUnchangedTests(unittest.TestCase):
    """Requirement 10: neighbouring behaviour must not move."""

    def test_lowercase_plan_now_survives_the_reasoner_boundary(self):
        payload = raw_plan(
            [
                {"action": "look_at", "target": "CAMERA"},
                {"action": "speak", "text": "Looking at the camera."},
            ]
        )
        reasoner = BrainReasoner(_ScriptedProvider(payload), BrainPlanner(), None)
        plan, reason = reasoner.plan(world(), goal(), PersonalityProfile())
        self.assertIsNotNone(plan, reason)
        self.assertEqual(reason, "OK")
        self.assertEqual(plan.steps[0].action, ActionType.LOOK_AT)

    def test_provider_failure_diagnostics_are_unchanged(self):
        class _Raising(LLMProvider):
            name = "test-provider"

            def __init__(self) -> None:
                self.config = types.SimpleNamespace(model="vendor/model-1")

            def generate(self, request: LLMRequest) -> str:  # pragma: no cover - mock
                raise LLMProviderError("boom")

        reasoner = BrainReasoner(_Raising(), BrainPlanner(), None)
        with mock.patch("backend.llm.failure_diagnostics.log_event") as spy:
            plan, reason = reasoner.plan(world(), goal(), PersonalityProfile())
        self.assertIsNone(plan)
        self.assertEqual(reason, "LLMProviderError")
        self.assertEqual(spy.call_args.args[1], "brain.llm_request_failed")

    def test_dispatch_id_and_result_reason_paths_are_untouched(self):
        loop_source = (REPO_ROOT / "backend/brain/loop.py").read_text(encoding="utf-8")
        models_source = (REPO_ROOT / "backend/brain/models.py").read_text(encoding="utf-8")
        self.assertIn("dispatch_id=_new_dispatch_id()", loop_source)
        self.assertIn("def _is_awaited_result", loop_source)
        self.assertIn("dispatch_id: Optional[str] = None", models_source)
        # The action enum itself is unchanged by this task.
        self.assertEqual(
            [m.value for m in ActionType],
            [
                "MOVE_TO", "STOP", "TURN_TO", "LOOK_AT", "SPEAK", "WAVE", "DANCE",
                "RUN", "PICK_UP_PHONE", "PLACE_PHONE", "START_STREAM",
                "STOP_STREAM", "SET_CAMERA",
            ],
        )


if __name__ == "__main__":
    unittest.main()
