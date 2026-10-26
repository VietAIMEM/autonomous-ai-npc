from __future__ import annotations

import json
import unittest
from datetime import datetime, timedelta, timezone
from typing import Any

from backend.brain.loop import AutonomousBrain, BrainConfig, BrainStatus
from backend.brain.models import ActionResult, ActionType, ExecutionStatus, Goal, GoalType
from backend.brain.reasoning import BrainReasoner
from backend.brain.world_state import build_world_state
from backend.chat.models import ChatMessage
from backend.chat.service import ChatService, ChatServiceConfig
from backend.llm.base import LLMProvider, LLMRequest
from backend.skills.executor import SkillExecutor
from backend.skills.library import SkillLibrary, SkillRegistryError, create_default_skill_library
from backend.skills.models import Skill, SkillCategory, SkillStatus
from backend.tests.dispatch_support import confirm


BASE_TIME = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)


class FixedClock:
    def __init__(self) -> None:
        self.value = BASE_TIME

    def __call__(self) -> datetime:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += timedelta(seconds=seconds)


class SkillLLM(LLMProvider):
    """Select only from the supplied skill allowlist."""

    def __init__(self, skill_id: str = "show_table") -> None:
        self.skill_id = skill_id

    def generate(self, request: LLMRequest) -> str:
        if "show_table" not in request.prompt:
            raise AssertionError("skill allowlist missing")
        return json.dumps({"skill_id": self.skill_id, "parameters": {}})


def snapshot(**overrides: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "character": {
            "position": [0, 0, 0],
            "rotation_y": 90,
            "posture": "standing",
            "current_action": "idle",
        },
        "phone": {"state": "Placed", "position": [1, 0.75, 0], "placement_surface": "TABLE"},
        "stream": {"is_streaming": True, "camera_available": True, "camera_active": True},
        "targets": {
            "CENTER": {"position": [0, 0, 0], "navigation_target": True, "look_at_target": True},
            "TABLE": {"position": [1, 0.75, 0], "navigation_target": True, "look_at_target": True},
            "CAMERA": {"position": [2, 0.8, 0], "navigation_target": False, "look_at_target": True},
        },
    }
    value.update(overrides)
    return value


def world(**overrides: Any):
    return build_world_state(snapshot(**overrides))


def chat_message(text: str) -> ChatMessage:
    return ChatMessage("msg_skill", "viewer_01", text, text, BASE_TIME)


def goal() -> Goal:
    return Goal("goal_skill", GoalType.RESPOND_TO_VIEWER, 0.9, "skill test")


def executor(clock: FixedClock | None = None, library: SkillLibrary | None = None) -> tuple[SkillExecutor, FixedClock, SkillLibrary]:
    from backend.brain.planner import BrainPlanner

    clock = clock or FixedClock()
    library = library or create_default_skill_library()
    return SkillExecutor(library, BrainPlanner(), clock), clock, library


class SkillModelTests(unittest.TestCase):
    def test_schema_and_registration(self) -> None:
        skill_executor, _, library = executor()
        self.assertTrue(library.exists("dance"))
        self.assertFalse(library.exists("fly_to_moon"))
        self.assertEqual(len(library.list()), 14)
        self.assertEqual(library.get("dance").category, SkillCategory.SOCIAL)
        self.assertIn("navigation", library.search(tags=["navigation"])[0].tags)

        with self.assertRaises(SkillRegistryError):
            library.register(library.get("dance"))

    def test_disabled_and_lookup(self) -> None:
        skill_executor, _, library = executor()
        library.disable("dance")
        availability = skill_executor.availability("dance", world(), now=BASE_TIME)
        self.assertFalse(availability.valid)
        self.assertEqual(availability.reason, "SKILL_DISABLED")
        self.assertTrue(library.disable("dance").enabled is False)


class SkillValidationTests(unittest.TestCase):
    def test_parameter_validation(self) -> None:
        skill_executor, _, _ = executor()
        self.assertFalse(skill_executor.can_execute("walk_to", world(), now=BASE_TIME))
        self.assertFalse(skill_executor.can_execute("walk_to", world(), {"target": "MOON"}, now=BASE_TIME))
        self.assertFalse(skill_executor.can_execute("dance", world(), {"target": "TABLE"}, now=BASE_TIME))
        self.assertTrue(skill_executor.can_execute("walk_to", world(), {"target": "TABLE"}, now=BASE_TIME))

    def test_conditions(self) -> None:
        skill_executor, _, _ = executor()
        self.assertTrue(skill_executor.can_execute("dance", world(), now=BASE_TIME))
        bad_posture = snapshot(character={"posture": "dancing", "current_action": "dancing"})
        self.assertEqual(
            skill_executor.availability("dance", build_world_state(bad_posture), now=BASE_TIME).reason,
            "CHARACTER_STATE_INCOMPATIBLE",
        )
        missing_phone = snapshot(phone={"state": "Missing", "position": [1, 0, 0]})
        self.assertEqual(
            skill_executor.availability("pick_up_phone", build_world_state(missing_phone), now=BASE_TIME).reason,
            "PHONE_NOT_PICKABLE",
        )

    def test_cooldown(self) -> None:
        skill_executor, clock, _ = executor()
        plan_result = skill_executor.create_plan(
            _selection("dance", {}),
            _goal(),
            world(),
            now=clock.value,
        )
        self.assertEqual(plan_result.reason, "OK")
        for _ in plan_result.plan.steps:
            skill_executor.observe_success(ActionResult(ActionType.DANCE, ExecutionStatus.SUCCESS))
        self.assertEqual(plan_result.execution.status, SkillStatus.SUCCESS)
        self.assertEqual(
            skill_executor.availability("dance", world(), now=clock.value).reason,
            "SKILL_ON_COOLDOWN",
        )
        clock.advance(5.1)
        self.assertTrue(skill_executor.can_execute("dance", world(), now=clock.value))


class SkillExecutionTests(unittest.TestCase):
    def test_execution_and_evaluation(self) -> None:
        skill_executor, _, _ = executor()
        result = skill_executor.create_plan(_selection("show_table", {}), _goal(), world(), now=BASE_TIME)
        self.assertEqual(result.reason, "OK")
        self.assertEqual([step.action for step in result.plan.steps], [ActionType.MOVE_TO, ActionType.LOOK_AT, ActionType.SPEAK])
        self.assertEqual(result.execution.total_steps, 3)

        skill_executor.observe_success(ActionResult(ActionType.MOVE_TO, ExecutionStatus.SUCCESS))
        skill_executor.observe_success(ActionResult(ActionType.LOOK_AT, ExecutionStatus.SUCCESS))
        skill_executor.observe_success(ActionResult(ActionType.SPEAK, ExecutionStatus.SUCCESS))
        evaluation = result.execution.result
        self.assertEqual(evaluation.status, SkillStatus.SUCCESS)
        self.assertEqual(evaluation.steps_completed, 3)
        self.assertIn("character_position_changed", evaluation.world_effects)

    def test_failure_propagation_and_cancellation(self) -> None:
        skill_executor, _, _ = executor()
        result = skill_executor.create_plan(_selection("show_table", {}), _goal(), world(), now=BASE_TIME)
        skill_executor.observe_failure(ActionResult(ActionType.MOVE_TO, ExecutionStatus.FAILED, "TARGET_UNREACHABLE"), final=True)
        self.assertEqual(result.execution.status, SkillStatus.FAILED)
        self.assertEqual(result.execution.result.interruption_reason, "TARGET_UNREACHABLE")

        second = skill_executor.create_plan(_selection("show_table", {}), _goal(), world(), now=BASE_TIME)
        skill_executor.cancel_active("social_priority")
        self.assertEqual(second.execution.status, SkillStatus.CANCELLED)

    def test_timeout(self) -> None:
        skill_executor, clock, _ = executor()
        result = skill_executor.create_plan(_selection("show_table", {}), _goal(), world(), now=clock.value)
        clock.advance(121)
        self.assertTrue(skill_executor.timeout_active(clock.value))
        self.assertEqual(result.execution.status, SkillStatus.INTERRUPTED)
        self.assertEqual(result.execution.result.interruption_reason, "SKILL_TIMEOUT")

    def test_invalid_composition_registration(self) -> None:
        library = SkillLibrary()
        orphan = library.get("dance") or create_default_skill_library().get("dance")
        invalid = _composition_skill("broken", ["missing_skill"])
        with self.assertRaises(SkillRegistryError):
            library.register(invalid)


class BrainSkillIntegrationTests(unittest.TestCase):
    def _brain(self, provider: SkillLLM, clock: FixedClock) -> AutonomousBrain:
        from backend.brain.personality import load_personality

        chat = ChatService(_MockChatLLM(), ChatServiceConfig(max_messages_per_rate_window=20), clock=clock)
        library = create_default_skill_library()
        return AutonomousBrain(
            provider,
            chat,
            BrainConfig(decision_interval_seconds=0.0),
            personality=load_personality(),
            clock=clock,
            skill_library=library,
        )

    def test_skill_selection_compiles_to_existing_actions(self) -> None:
        clock = FixedClock()
        brain = self._brain(SkillLLM("show_table"), clock)
        brain.chat.submit("viewer_01", "Lacrimosa, show me the table")
        tick = brain.tick(world())
        self.assertEqual(tick.status, BrainStatus.EXECUTING)
        self.assertEqual(tick.action.action, ActionType.MOVE_TO)
        self.assertEqual(tick.action.target, "TABLE")
        self.assertIsNotNone(brain.skill_executor.active)

        second = brain.tick(world(), confirm(brain, ActionResult(ActionType.MOVE_TO, ExecutionStatus.SUCCESS)))
        self.assertEqual(second.action.action, ActionType.LOOK_AT)
        third = brain.tick(world(), confirm(brain, ActionResult(ActionType.LOOK_AT, ExecutionStatus.SUCCESS)))
        self.assertEqual(third.action.action, ActionType.SPEAK)
        final = brain.tick(world(), confirm(brain, ActionResult(ActionType.SPEAK, ExecutionStatus.SUCCESS)))
        self.assertEqual(final.status, BrainStatus.IDLE)
        self.assertEqual(len(brain.skill_executor.evaluations), 1)

    def test_unknown_skill_is_rejected(self) -> None:
        clock = FixedClock()
        brain = self._brain(SkillLLM("fly_to_moon"), clock)
        brain.chat.submit("viewer_01", "please fly")
        tick = brain.tick(world())
        self.assertEqual(tick.status, BrainStatus.ERROR)
        self.assertEqual(tick.message, "planning_failed:SKILL_NOT_FOUND")

    def test_legacy_action_plan_still_works(self) -> None:
        from backend.brain.reasoning import MockBrainLLMProvider

        clock = FixedClock()
        chat = ChatService(_MockChatLLM(), ChatServiceConfig(max_messages_per_rate_window=20), clock=clock)
        brain = AutonomousBrain(
            MockBrainLLMProvider(),
            chat,
            BrainConfig(decision_interval_seconds=0.0),
            clock=clock,
            skill_library=create_default_skill_library(),
        )
        brain.chat.submit("viewer_01", "Lacrimosa can you dance?")
        first = brain.tick(world())
        self.assertEqual(first.action.action, ActionType.MOVE_TO)
        self.assertEqual(first.action.target, "CENTER")

    def test_natural_stream_and_phone_skills_execute_and_stop_stays_stopped(self) -> None:
        from backend.brain.reasoning import MockBrainLLMProvider
        from backend.social.service import SocialAutonomyService

        clock = FixedClock()
        chat = ChatService(_MockChatLLM(), ChatServiceConfig(max_messages_per_rate_window=20), clock=clock)
        social = SocialAutonomyService(clock=clock)
        brain = AutonomousBrain(
            MockBrainLLMProvider(),
            chat,
            BrainConfig(decision_interval_seconds=0.0),
            clock=clock,
            social_service=social,
            skill_library=create_default_skill_library(),
        )

        brain.chat.submit("viewer_01", "Lacrimosa, pick up the phone.")
        pickup = brain.tick(world())
        self.assertEqual(pickup.action.action, ActionType.PICK_UP_PHONE)
        held_world = world(phone={"state": "Held", "placement_surface": None})
        placed = brain.tick(
            held_world,
            confirm(brain, ActionResult(ActionType.PICK_UP_PHONE, ExecutionStatus.SUCCESS)),
        )
        self.assertEqual(placed.status, BrainStatus.IDLE)

        brain.chat.submit("viewer_01", "Lacrimosa, place the phone on the table.")
        place = brain.tick(held_world)
        self.assertEqual(place.action.action, ActionType.PLACE_PHONE)
        self.assertEqual(place.action.target, "TABLE")
        resting_world = world()
        placement_complete = brain.tick(
            resting_world,
            confirm(brain, ActionResult(ActionType.PLACE_PHONE, ExecutionStatus.SUCCESS)),
        )
        self.assertEqual(placement_complete.status, BrainStatus.IDLE)
        brain.chat.submit("viewer_01", "Lacrimosa, can you start the stream?")
        started = brain.tick(
            world(stream={"is_streaming": False, "camera_available": True}),
        )
        self.assertIsNotNone(started.action, started.to_dict())
        self.assertEqual(started.action.action, ActionType.START_STREAM)
        streaming_world = world()
        start_complete = brain.tick(
            streaming_world,
            confirm(brain, ActionResult(ActionType.START_STREAM, ExecutionStatus.SUCCESS)),
        )
        self.assertEqual(start_complete.status, BrainStatus.IDLE)
        # The maintenance tick intentionally leaves the "start the stream"
        # request in the attention queue, so give the follow-up a realistic
        # later timestamp instead of letting both submissions tie.
        clock.advance(1.0)
        brain.chat.submit("viewer_01", "Lacrimosa, can you stop the stream?")
        stopped = brain.tick(streaming_world)
        self.assertEqual(stopped.action.action, ActionType.STOP_STREAM)
        stopped_world = world(stream={"is_streaming": False})

        after_stop = brain.tick(
            stopped_world,
            confirm(brain, ActionResult(ActionType.STOP_STREAM, ExecutionStatus.SUCCESS)),
        )
        self.assertEqual(after_stop.status, BrainStatus.IDLE)
        self.assertIsNone(after_stop.action)

        clock.advance(6.0)
        subsequent = brain.tick(stopped_world)
        self.assertNotEqual(subsequent.goal.goal_type, GoalType.MAINTAIN_STREAM)
        self.assertNotEqual(subsequent.action.action if subsequent.action else None, ActionType.START_STREAM)


class _MockChatLLM(LLMProvider):
    def generate(self, request: LLMRequest) -> str:
        return json.dumps({"speech": "Okay.", "intent": "REQUEST_ACTION", "requested_action": None, "priority": 0.8})


def _selection(skill_id: str, parameters: dict[str, Any]):
    from backend.skills.models import SkillSelection

    return SkillSelection(skill_id, parameters, "test")


def _goal() -> Goal:
    return Goal("goal_test", GoalType.RESPOND_TO_VIEWER, 0.9, "test")


def _composition_skill(skill_id: str, children: list[str]) -> Skill:
    from backend.skills.models import SkillCompositionStep, SkillCondition, SkillConditionOperator, SkillConditionType

    return Skill(
        skill_id=skill_id,
        name="Broken",
        description="invalid registration fixture",
        category=SkillCategory.COMPOSITE,
        preconditions=(SkillCondition(SkillConditionType.TARGET_EXISTS, SkillConditionOperator.PRESENT),),
        composition=tuple(SkillCompositionStep(child, {}) for child in children),
    )


if __name__ == "__main__":
    unittest.main()
