"""Focused tests for action-specific target constraints in the planner prompt.

Background (see ``Target_Not_Lookable_Audit_20261018T204826Z``): every historical
``TARGET_NOT_LOOKABLE`` rejection involved ``CAMERA_STAND`` (13) or
``PHONE_STAND`` (2) — targets that are navigable by design but explicitly not
lookable. The flags were already present in the nested ``world.targets`` block,
so these tests pin the stronger, action-specific contract that is now rendered
directly into the transmitted prompt, while asserting the validator itself is
unchanged.

All tests are deterministic and offline; no provider call is made.
"""

from __future__ import annotations

import json
import unittest
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


def target(name: str, navigable: bool, lookable: bool) -> dict:
    return {
        "position": [0.0, 0.0, 0.0],
        "navigation_target": navigable,
        "look_at_target": lookable,
    }


#: The production-shaped table: standing points are navigable-only, their
#: device counterparts are lookable-only.
PRODUCTION_TARGETS = {
    "CENTER": target("CENTER", True, True),
    "TABLE": target("TABLE", True, True),
    "BENCH": target("BENCH", True, True),
    "FLOOR": target("FLOOR", True, True),
    "PHONE": target("PHONE", False, True),
    "CAMERA": target("CAMERA", False, True),
    "CAMERA_STAND": target("CAMERA_STAND", True, False),
    "PHONE_STAND": target("PHONE_STAND", True, False),
}


def world_with(targets: dict) -> object:
    return build_world_state(
        {
            "character": {"position": [0, 0, 0], "current_action": "idle"},
            "phone": {"state": "Placed", "position": [1, 0.75, 0]},
            "stream": {"is_streaming": False, "camera_available": True},
            "targets": targets,
        }
    )


def goal(goal_type: GoalType = GoalType.EXPLORE_ROOM) -> Goal:
    return Goal("goal", goal_type, 0.5, "test", status=GoalStatus.ACTIVE)


def prompt_payload(targets: dict | None = None) -> dict:
    planner = BrainPlanner()
    prompt = planner.build_prompt(
        world_with(targets if targets is not None else PRODUCTION_TARGETS),
        goal(),
        PersonalityProfile(),
    )
    return json.loads(prompt)


class PromptContentTests(unittest.TestCase):
    def test_prompt_contains_action_specific_target_constraints(self):
        payload = prompt_payload()
        rules = payload["action_target_rules"]
        self.assertIsInstance(rules, list)
        self.assertTrue(rules)
        joined = " ".join(rules)
        self.assertIn("LOOK_AT", joined)
        self.assertIn("TURN_TO", joined)
        self.assertIn("MOVE_TO", joined)
        self.assertIn("valid_look_targets", joined)
        self.assertIn("valid_navigation_targets", joined)
        # The same rule is also carried by the prominent constraints block.
        constraints = " ".join(payload["constraints"])
        self.assertIn("LOOK_AT/TURN_TO use world.valid_look_targets only", constraints)
        self.assertIn("MOVE_TO use world.valid_navigation_targets only", constraints)

    def test_look_targets_derive_from_look_at_target_only(self):
        payload = prompt_payload()
        world_block = payload["world"]
        self.assertEqual(
            world_block["valid_look_targets"],
            sorted(
                name
                for name, spec in PRODUCTION_TARGETS.items()
                if spec["look_at_target"]
            ),
        )
        # Not simply "every entity".
        self.assertNotEqual(
            world_block["valid_look_targets"], sorted(PRODUCTION_TARGETS)
        )

    def test_navigation_targets_derive_independently_from_navigation_flag(self):
        payload = prompt_payload()
        world_block = payload["world"]
        self.assertEqual(
            world_block["valid_navigation_targets"],
            sorted(
                name
                for name, spec in PRODUCTION_TARGETS.items()
                if spec["navigation_target"]
            ),
        )
        self.assertNotEqual(
            world_block["valid_navigation_targets"],
            world_block["valid_look_targets"],
        )

    def test_standing_points_are_excluded_from_look_targets(self):
        payload = prompt_payload()
        world_block = payload["world"]
        self.assertNotIn("CAMERA_STAND", world_block["valid_look_targets"])
        self.assertNotIn("PHONE_STAND", world_block["valid_look_targets"])
        # ...and they are named as the navigation-only set to avoid.
        self.assertEqual(
            world_block["navigation_only_targets"], ["CAMERA_STAND", "PHONE_STAND"]
        )
        self.assertEqual(world_block["look_only_targets"], ["CAMERA", "PHONE"])

    def test_look_only_objects_are_not_navigation_destinations(self):
        payload = prompt_payload()
        world_block = payload["world"]
        self.assertNotIn("CAMERA", world_block["valid_navigation_targets"])
        self.assertNotIn("PHONE", world_block["valid_navigation_targets"])
        # They are still legitimate look targets.
        self.assertIn("CAMERA", world_block["valid_look_targets"])
        self.assertIn("PHONE", world_block["valid_look_targets"])

    def test_target_lists_track_a_changed_target_table(self):
        changed = dict(PRODUCTION_TARGETS)
        # SHELF and LAMP model the old failure mode: a world target that is
        # present but not executable because TargetResolver has no alias.
        changed["SHELF"] = target("SHELF", True, False)
        changed["LAMP"] = target("LAMP", False, True)
        changed["CAMERA_STAND"] = target("CAMERA_STAND", True, True)  # flag flipped
        payload = prompt_payload(changed)
        world_block = payload["world"]
        self.assertNotIn("SHELF", world_block["valid_navigation_targets"])
        self.assertNotIn("SHELF", world_block["valid_look_targets"])
        self.assertNotIn("LAMP", world_block["valid_look_targets"])
        self.assertNotIn("LAMP", world_block["valid_navigation_targets"])
        self.assertNotIn("SHELF", world_block["navigation_only_targets"])
        self.assertNotIn("LAMP", world_block["look_only_targets"])
        self.assertNotIn("SHELF", world_block["targets"])
        self.assertNotIn("LAMP", world_block["targets"])
        # Flipping the authoritative flag moves the entity between the sets: the
        # lists are computed, never hard-coded.
        self.assertIn("CAMERA_STAND", world_block["valid_look_targets"])
        self.assertNotIn("CAMERA_STAND", world_block["navigation_only_targets"])

    def test_rules_and_lists_stay_present_under_the_prompt_limit(self):
        import backend.brain.planner as planner_module

        planner = BrainPlanner()
        world = world_with(PRODUCTION_TARGETS)
        full = planner.build_prompt(world, goal(), PersonalityProfile())
        self.assertLessEqual(len(full), planner_module.MAX_PLANNER_PROMPT_CHARS)

        # Truncate deliberately to a cut that keeps role/character/world/rules
        # but drops the later variable-length blocks, then prove the contract
        # still survived the compact prompt guard.
        boundary = full.index('"goal"')
        cap = boundary + 50
        self.assertLess(cap, len(full))
        with mock.patch.object(planner_module, "MAX_PLANNER_PROMPT_CHARS", cap):
            truncated = planner.build_prompt(world, goal(), PersonalityProfile())
        self.assertEqual(len(truncated), cap)
        self.assertNotEqual(truncated, full)
        self.assertIn('"action_target_rules"', truncated)
        self.assertIn('"valid_look_targets"', truncated)
        self.assertIn('"valid_navigation_targets"', truncated)
        self.assertIn('"navigation_only_targets"', truncated)

    def test_existing_prompt_keys_are_preserved(self):
        payload = prompt_payload()
        for key in (
            "role",
            "character",
            "world",
            "goal",
            "behavior",
            "chat",
            "available_actions",
            "constraints",
            "output_schema",
            "skill_output_schema",
            "valid_targets",
            "available_skills",
            "learned_skill_ranking",
            "relevant_memories",
            "personalization",
        ):
            self.assertIn(key, payload)
        # The flat allow-list is unchanged and still names every entity.
        self.assertEqual(payload["valid_targets"], sorted(PRODUCTION_TARGETS))
        # The per-entity flags are still present too.
        self.assertFalse(payload["world"]["targets"]["CAMERA_STAND"]["look_at_target"])


class ValidationUnchangedTests(unittest.TestCase):
    """The validator must keep rejecting what it rejected before."""

    def setUp(self) -> None:
        self.planner = BrainPlanner()
        self.world = world_with(PRODUCTION_TARGETS)

    def test_look_at_standing_points_still_rejected(self):
        cases = [
            (ActionType.LOOK_AT, "CAMERA_STAND"),
            (ActionType.TURN_TO, "PHONE_STAND"),
        ]
        for action, name in cases:
            with self.subTest(action=action, target=name):
                validation = self.planner.validate_step(
                    PlanStep(action, name), self.world
                )
                self.assertFalse(validation.valid)
                self.assertEqual(validation.reason, "TARGET_NOT_LOOKABLE")

    def test_navigable_only_targets_still_rejected_for_movement(self):
        validation = self.planner.validate_step(
            PlanStep(ActionType.MOVE_TO, "CAMERA"), self.world
        )
        self.assertFalse(validation.valid)
        self.assertEqual(validation.reason, "TARGET_NOT_NAVIGABLE")

    def test_valid_look_and_navigation_targets_still_accepted(self):
        accepted = [
            (ActionType.LOOK_AT, "CAMERA"),
            (ActionType.LOOK_AT, "TABLE"),
            (ActionType.TURN_TO, "BENCH"),
            (ActionType.MOVE_TO, "CENTER"),
            (ActionType.MOVE_TO, "CAMERA_STAND"),
        ]
        for action, name in accepted:
            with self.subTest(action=action, target=name):
                validation = self.planner.validate_step(
                    PlanStep(action, name), self.world
                )
                self.assertTrue(validation.valid)

    def test_validation_matches_the_advertised_sets(self):
        payload = prompt_payload()
        world_block = payload["world"]
        for name in world_block["valid_look_targets"]:
            self.assertTrue(
                self.planner.validate_step(
                    PlanStep(ActionType.LOOK_AT, name), self.world
                ).valid,
                name,
            )
        for name in world_block["valid_navigation_targets"]:
            self.assertTrue(
                self.planner.validate_step(
                    PlanStep(ActionType.MOVE_TO, name), self.world
                ).valid,
                name,
            )
        for name in world_block["navigation_only_targets"]:
            self.assertEqual(
                self.planner.validate_step(
                    PlanStep(ActionType.LOOK_AT, name), self.world
                ).reason,
                "TARGET_NOT_LOOKABLE",
            )


class _RaisingProvider(LLMProvider):
    name = "test-provider"

    def __init__(self, error: BaseException) -> None:
        self._error = error
        from types import SimpleNamespace

        self.config = SimpleNamespace(model="vendor/model-1")

    def generate(self, request: LLMRequest) -> str:  # pragma: no cover - stub
        raise self._error


class DecisionBehaviourUnchangedTests(unittest.TestCase):
    def test_provider_failure_outcome_and_diagnostic_are_unchanged(self):
        reasoner = BrainReasoner(
            _RaisingProvider(LLMProviderError("boom")), BrainPlanner(), None
        )
        with mock.patch("backend.llm.failure_diagnostics.log_event") as spy:
            plan, reason = reasoner.plan(
                world_with(PRODUCTION_TARGETS), goal(), PersonalityProfile()
            )
        self.assertIsNone(plan)
        # The loop still composes the same "planning_failed:" string.
        self.assertEqual(reason, "LLMProviderError")
        self.assertEqual(f"planning_failed:{reason}", "planning_failed:LLMProviderError")
        self.assertEqual(spy.call_count, 1)
        self.assertEqual(spy.call_args.args[1], "brain.llm_request_failed")

    def test_parse_plan_validation_path_is_unchanged(self):
        planner = BrainPlanner()
        world = world_with(PRODUCTION_TARGETS)
        raw = json.dumps(
            {
                "reason": "look at a standing point",
                "steps": [{"action": "LOOK_AT", "target": "CAMERA_STAND"}],
            }
        )
        plan, reason = planner.parse_plan(raw, goal(), world)
        self.assertIsNone(plan)
        self.assertEqual(reason, "TARGET_NOT_LOOKABLE")


if __name__ == "__main__":
    unittest.main()
