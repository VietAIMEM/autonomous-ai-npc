"""Focused tests for the two planner contract gaps found in live validation.

The tests stay offline and exercise the real prompt builder, target resolver,
and plan validator. They do not mutate production state and do not dispatch
Unity actions.
"""

from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace

from backend.brain.models import (
    ActionType,
    Goal,
    GoalStatus,
    GoalType,
    PersonalityProfile,
)
from backend.brain.loop import AutonomousBrain, BrainConfig
from backend.brain.planner import BrainPlanner
from backend.brain.reasoning import BrainReasoner
from backend.brain.targets import TargetResolver
from backend.brain.world_state import build_world_state
from backend.chat.service import ChatService, ChatServiceConfig
from backend.llm.base import LLMProvider, LLMRequest
from backend.llm.mock import MockLLMProvider


def target(navigable: bool = True, lookable: bool = True) -> dict:
    return {
        "position": [0.0, 0.0, 0.0],
        "navigation_target": navigable,
        "look_at_target": lookable,
    }


def world_with_phone(phone_state: str = "Held"):
    return build_world_state(
        {
            "character": {"position": [0, 0, 0], "current_action": "idle"},
            "phone": {"state": phone_state, "position": [1, 0.75, 0]},
            "stream": {"is_streaming": False, "camera_available": True},
            "targets": {
                "CENTER": target(),
                "TABLE": target(),
                "BENCH": target(),
                "FLOOR": target(),
                "CAMERA": target(navigable=False, lookable=True),
            },
        },
        timestamp=datetime(2026, 10, 22, tzinfo=timezone.utc),
    )


def world_with_window():
    world = build_world_state(
        {
            "character": {"position": [0, 0, 0], "current_action": "idle"},
            "phone": {"state": "World", "position": [1, 0.75, 0]},
            "stream": {"is_streaming": False, "camera_available": True},
            "targets": {
                "CENTER": target(),
                # The validation fixture had WINDOW, but Unity does not publish
                # it. Keeping it here proves the prompt can no longer advertise
                # a target the resolver cannot execute.
                "WINDOW": target(navigable=False, lookable=True),
            },
        },
        timestamp=datetime(2026, 10, 22, tzinfo=timezone.utc),
    )
    return world


def goal(goal_type: GoalType = GoalType.EXPLORE_ROOM) -> Goal:
    return Goal("goal", goal_type, 0.5, "test", status=GoalStatus.ACTIVE)


def prompt_payload(world) -> dict:
    return json.loads(
        BrainPlanner().build_prompt(world, goal(), PersonalityProfile())
    )


class WindowTargetContractTests(unittest.TestCase):
    def test_unsupported_world_target_is_not_advertised(self) -> None:
        world = world_with_window()
        payload = prompt_payload(world)

        self.assertNotIn("WINDOW", payload["world"]["targets"])
        self.assertNotIn("WINDOW", payload["valid_targets"])
        self.assertNotIn("WINDOW", payload["world"]["valid_look_targets"])
        self.assertNotIn("WINDOW", payload["world"]["valid_navigation_targets"])
        self.assertNotIn("WINDOW", payload["world"]["look_only_targets"])
        self.assertNotIn("WINDOW", payload["world"]["navigation_only_targets"])

    def test_every_advertised_target_is_resolvable(self) -> None:
        payload = prompt_payload(world_with_window())
        resolver = TargetResolver(world_with_window())

        advertised = set(payload["valid_targets"])
        advertised.update(payload["world"]["valid_look_targets"])
        advertised.update(payload["world"]["valid_navigation_targets"])
        for name in sorted(advertised):
            resolved, reason = resolver.resolve(name)
            self.assertIsNotNone(resolved, name)
            self.assertEqual(reason, "OK")

    def test_window_step_is_still_rejected(self) -> None:
        planner = BrainPlanner()
        world = world_with_window()
        raw = json.dumps(
            {
                "reason": "look at the window",
                "steps": [{"action": "LOOK_AT", "target": "WINDOW"}],
            }
        )
        plan, reason = planner.parse_plan(raw, goal(), world)

        self.assertIsNone(plan)
        self.assertEqual(reason, "TARGET_NOT_FOUND")


class PhonePlacementContractTests(unittest.TestCase):
    def test_explicit_valid_destination_is_accepted(self) -> None:
        planner = BrainPlanner()
        world = world_with_phone("Held")
        raw = json.dumps(
            {
                "reason": "place the phone",
                "steps": [{"action": "PLACE_PHONE", "target": "TABLE"}],
            }
        )
        plan, reason = planner.parse_plan(raw, goal(), world)

        self.assertIsNotNone(plan, reason)
        self.assertEqual(reason, "OK")
        self.assertEqual(plan.steps[0].action, ActionType.PLACE_PHONE)
        self.assertEqual(plan.steps[0].target, "TABLE")

    def test_destination_is_inferred_from_one_prior_move_to(self) -> None:
        planner = BrainPlanner()
        world = world_with_phone("Held")
        raw = json.dumps(
            {
                "reason": "place the phone at the table",
                "steps": [
                    {"action": "MOVE_TO", "target": "TABLE"},
                    {"action": "LOOK_AT", "target": "TABLE"},
                    {"action": "PLACE_PHONE", "target": None},
                ],
            }
        )
        plan, reason = planner.parse_plan(raw, goal(), world)

        self.assertIsNotNone(plan, reason)
        self.assertEqual(reason, "OK")
        self.assertEqual(plan.steps[-1].action, ActionType.PLACE_PHONE)
        self.assertEqual(plan.steps[-1].target, "TABLE")

    def test_explicit_destination_wins_over_inference(self) -> None:
        planner = BrainPlanner()
        world = world_with_phone("Held")
        raw = json.dumps(
            {
                "reason": "use the bench explicitly",
                "steps": [
                    {"action": "MOVE_TO", "target": "TABLE"},
                    {"action": "PLACE_PHONE", "target": "BENCH"},
                ],
            }
        )
        plan, reason = planner.parse_plan(raw, goal(), world)

        self.assertIsNotNone(plan, reason)
        self.assertEqual(reason, "OK")
        self.assertEqual(plan.steps[-1].target, "BENCH")

    def test_missing_destination_without_placement_move_is_rejected(self) -> None:
        planner = BrainPlanner()
        world = world_with_phone("Held")
        raw = json.dumps(
            {
                "reason": "place the phone",
                "steps": [{"action": "PLACE_PHONE", "target": None}],
            }
        )
        plan, reason = planner.parse_plan(raw, goal(), world)

        self.assertIsNone(plan)
        self.assertEqual(reason, "PLACEMENT_TARGET_INVALID")


    def test_non_placement_destination_is_not_inferred(self) -> None:
        planner = BrainPlanner()
        world = world_with_phone("Held")
        raw = json.dumps(
            {
                "reason": "go to center and place the phone",
                "steps": [
                    {"action": "MOVE_TO", "target": "CENTER"},
                    {"action": "PLACE_PHONE", "target": None},
                ],
            }
        )
        plan, reason = planner.parse_plan(raw, goal(), world)

        self.assertIsNone(plan)
        self.assertEqual(reason, "PLACEMENT_TARGET_INVALID")

    def test_later_placement_move_is_not_used_for_inference(self) -> None:
        planner = BrainPlanner()
        world = world_with_phone("Held")
        raw = json.dumps(
            {
                "reason": "placement before the move",
                "steps": [
                    {"action": "PLACE_PHONE", "target": None},
                    {"action": "MOVE_TO", "target": "TABLE"},
                ],
            }
        )
        plan, reason = planner.parse_plan(raw, goal(), world)

        self.assertIsNone(plan)
        self.assertEqual(reason, "PLACEMENT_TARGET_INVALID")

    def test_conflicting_destinations_are_rejected(self) -> None:
        planner = BrainPlanner()
        world = world_with_phone("Held")
        raw = json.dumps(
            {
                "reason": "ambiguous placement",
                "steps": [
                    {"action": "MOVE_TO", "target": "TABLE"},
                    {"action": "MOVE_TO", "target": "BENCH"},
                    {"action": "PLACE_PHONE", "target": None},
                ],
            }
        )
        plan, reason = planner.parse_plan(raw, goal(), world)

        self.assertIsNone(plan)
        self.assertEqual(reason, "PLACEMENT_TARGET_INVALID")

    def test_unknown_destination_is_rejected(self) -> None:
        planner = BrainPlanner()
        world = world_with_phone("Held")
        raw = json.dumps(
            {
                "reason": "place on an unknown surface",
                "steps": [{"action": "PLACE_PHONE", "target": "SHELF"}],
            }
        )
        plan, reason = planner.parse_plan(raw, goal(), world)

        self.assertIsNone(plan)
        self.assertEqual(reason, "PLACEMENT_TARGET_INVALID")

    def test_rejected_plan_is_not_executed(self) -> None:
        class ScriptedProvider(LLMProvider):
            name = "scripted"

            def __init__(self) -> None:
                self.config = SimpleNamespace(model="test/model")

            def generate(self, request: LLMRequest) -> str:  # pragma: no cover
                return json.dumps(
                    {
                        "reason": "ambiguous",
                        "steps": [{"action": "PLACE_PHONE", "target": None}],
                    }
                )

        provider = ScriptedProvider()
        plan, reason = BrainReasoner(provider, BrainPlanner(), None).plan(
            world_with_phone("Held"),
            goal(GoalType.INTERACT_WITH_VIEWER),
            PersonalityProfile(),
            "put the phone down",
        )

        self.assertIsNone(plan)
        self.assertEqual(reason, "PLACEMENT_TARGET_INVALID")


class _StaticPlanProvider(LLMProvider):
    name = "static_plan"

    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls = 0
        self.config = SimpleNamespace(model="test/model")

    def generate(self, request: LLMRequest) -> str:  # pragma: no cover - stub
        self.calls += 1
        return json.dumps(self.payload)


class UnsupportedTargetFallbackContractTests(unittest.TestCase):
    def _brain(self, provider: _StaticPlanProvider) -> AutonomousBrain:
        chat = ChatService(
            MockLLMProvider(),
            ChatServiceConfig(max_messages_per_rate_window=20),
        )
        return AutonomousBrain(
            provider,
            chat,
            BrainConfig(decision_interval_seconds=0.0),
        )

    def test_unsupported_shelf_clarifies_instead_of_moving_to_table(self) -> None:
        provider = _StaticPlanProvider(
            {
                "reason": "fallback movement while responding",
                "steps": [
                    {"action": "MOVE_TO", "target": "TABLE"},
                    {"action": "TURN_TO", "target": "CENTER"},
                    {"action": "SPEAK", "text": "I'm heading to the table now."},
                ],
            }
        )
        loop = self._brain(provider)
        loop.chat.submit("viewer_01", "Please go to the shelf.")
        world = build_world_state(
            {
                "character": {"position": [0, 0, 0], "current_action": "idle"},
                "phone": {"state": "Placed", "position": [1, 0.75, 0]},
                "stream": {"is_streaming": True, "camera_available": True},
                "targets": {
                    "CENTER": target(),
                    "TABLE": target(),
                    "BENCH": target(),
                    "FLOOR": target(),
                    "CAMERA": target(navigable=False, lookable=True),
                },
            }
        )

        tick = loop.tick(world)

        self.assertIsNotNone(tick.action)
        self.assertEqual(tick.action.action, ActionType.SPEAK)
        self.assertIsNone(tick.action.target)
        self.assertIn("shelf", tick.action.text.casefold())
        self.assertNotIn("table", tick.action.text.casefold())
        self.assertEqual(provider.calls, 0)

    def test_supported_table_request_keeps_its_direct_action(self) -> None:
        provider = _StaticPlanProvider(
            {
                "reason": "viewer requested the table",
                "steps": [{"action": "MOVE_TO", "target": "TABLE"}],
            }
        )
        loop = self._brain(provider)
        loop.chat.submit("viewer_01", "Please go to the table.")
        world = build_world_state(
            {
                "character": {"position": [0, 0, 0], "current_action": "idle"},
                "phone": {"state": "Placed", "position": [1, 0.75, 0]},
                "stream": {"is_streaming": True, "camera_available": True},
                "targets": {
                    "CENTER": target(),
                    "TABLE": target(),
                },
            }
        )

        tick = loop.tick(world)

        self.assertIsNotNone(tick.action)
        self.assertEqual(tick.action.action, ActionType.MOVE_TO)
        self.assertEqual(tick.action.target, "TABLE")
        self.assertEqual(provider.calls, 1)

    def test_generic_location_fallback_is_not_treated_as_unsupported_target(self) -> None:
        provider = _StaticPlanProvider(
            {
                "reason": "move toward the room center",
                "steps": [{"action": "MOVE_TO", "target": "CENTER"}],
            }
        )
        loop = self._brain(provider)
        loop.chat.submit("viewer_01", "Please go to the room.")
        world = build_world_state(
            {
                "character": {"position": [0, 0, 0], "current_action": "idle"},
                "phone": {"state": "Placed", "position": [1, 0.75, 0]},
                "stream": {"is_streaming": True, "camera_available": True},
                "targets": {"CENTER": target(), "TABLE": target()},
            }
        )

        tick = loop.tick(world)

        self.assertIsNotNone(tick.action)
        self.assertEqual(tick.action.action, ActionType.MOVE_TO)
        self.assertEqual(tick.action.target, "CENTER")
        self.assertEqual(provider.calls, 1)


if __name__ == "__main__":
    unittest.main()
