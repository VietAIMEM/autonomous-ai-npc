"""Focused regression coverage for conversation quality contracts."""

from __future__ import annotations

import json
import unittest

from backend.brain.loop import AutonomousBrain, BrainConfig
from backend.brain.models import ActionType, Plan, PlanStep
from backend.brain.planner import BrainPlanner
from backend.brain.world_state import build_world_state
from backend.chat.intent import extract_intent
from backend.chat.models import ChatIntent, ChatMessage, utc_now
from backend.chat.normalization import normalize_text
from backend.chat.service import ChatService, ChatServiceConfig
from backend.llm.base import LLMProvider, LLMRequest
from backend.llm.mock import MockLLMProvider
from backend.tests.dispatch_support import confirm


class ScriptedBrainProvider(LLMProvider):
    def __init__(self, plans: list[dict[str, object]]) -> None:
        self.plans = list(plans)
        self.calls: list[LLMRequest] = []

    def generate(self, request: LLMRequest) -> str:
        self.calls.append(request)
        plan = self.plans.pop(0) if self.plans else {"steps": []}
        return json.dumps(plan)


def snapshot() -> dict[str, object]:
    return {
        "character": {"position": [0, 0, 0], "rotation_y": 90, "current_action": "idle"},
        "phone": {"state": "Placed", "position": [1, 0.75, 0], "placement_surface": "TABLE"},
        "stream": {"is_streaming": True, "camera_available": True, "camera_active": True},
        "targets": {
            "CENTER": {"position": [0, 0, 0], "navigation_target": True, "look_at_target": True},
            "TABLE": {"position": [1, 0.75, 0], "navigation_target": True, "look_at_target": True},
        },
    }


def world():
    return build_world_state(snapshot())


def make_brain(plans: list[dict[str, object]]):
    chat = ChatService(MockLLMProvider(), ChatServiceConfig(max_messages_per_rate_window=20))
    brain = AutonomousBrain(
        ScriptedBrainProvider(plans),
        chat,
        BrainConfig(decision_interval_seconds=0.0),
    )
    return brain, chat


class GreetingIntentTests(unittest.TestCase):
    def test_common_greetings_are_social_turns(self) -> None:
        cases = {
            "Hello there!": ChatIntent.GREETING,
            "Hi!": ChatIntent.GREETING,
            "Good morning!": ChatIntent.GREETING,
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(extract_intent(normalize_text(text)).intent, expected)

    def test_greeting_alone_does_not_plan_unrelated_physical_actions(self) -> None:
        brain, chat = make_brain(
            [
                {
                    "steps": [
                        {"action": "SPEAK", "text": "Hello! It's nice to meet you."},
                        {"action": "LOOK_AT", "target": "CENTER"},
                        {"action": "MOVE_TO", "target": "TABLE"},
                    ]
                }
            ]
        )
        submission = brain.chat.submit("viewer_quality", "Hello there!")
        self.assertTrue(submission.accepted)
        brain.tick(world(), confirm(brain, None))
        plan = brain.state.current_plan
        self.assertIsNotNone(plan)
        actions = [step.action for step in plan.steps]
        self.assertNotIn(ActionType.MOVE_TO, actions)
        self.assertNotIn(ActionType.PICK_UP_PHONE, actions)
        self.assertNotIn(ActionType.PLACE_PHONE, actions)
        self.assertNotIn(ActionType.START_STREAM, actions)
        self.assertNotIn(ActionType.STOP_STREAM, actions)
        self.assertTrue(actions)

    def test_explicit_action_after_a_greeting_is_not_suppressed(self) -> None:
        message = ChatMessage(
            message_id="greeting_action",
            viewer_id="viewer_quality",
            raw_text="Hello there, please move to the table.",
            normalized_text=normalize_text("Hello there, please move to the table."),
            timestamp=utc_now(),
            intent=ChatIntent.REQUEST_MOVEMENT,
            target="table",
        )
        plan = Plan(
            plan_id="plan_action",
            goal_id="goal_action",
            reason="explicit supported request",
            steps=(PlanStep(action=ActionType.MOVE_TO, target="TABLE"),),
        )
        normalized = AutonomousBrain._strip_unrelated_greeting_actions(plan, message)
        self.assertEqual(normalized.steps, plan.steps)


class GestureDialogueTests(unittest.TestCase):
    def test_positive_comment_reaches_speak_without_silent_text_loss(self) -> None:
        brain, chat = make_brain(
            [
                {
                    "steps": [
                        {
                            "action": "WAVE",
                            "text": "Hello there! You look great today!",
                        }
                    ]
                }
            ]
        )
        submission = brain.chat.submit("viewer_quality", "You look great today.")
        self.assertTrue(submission.accepted)
        brain.tick(world(), confirm(brain, None))
        plan = brain.state.current_plan
        self.assertIsNotNone(plan)
        actions = [step.action for step in plan.steps]
        self.assertIn(ActionType.WAVE, actions)
        self.assertIn(ActionType.SPEAK, actions)
        spoken = next(step.text for step in plan.steps if step.action == ActionType.SPEAK)
        visible = chat.get_state()["last_response"]["speech"]
        self.assertEqual(visible, spoken)

    def test_wave_without_dialogue_does_not_invent_speech(self) -> None:
        brain, _chat = make_brain([])
        plan = Plan(
            plan_id="plan_wave",
            goal_id="goal_wave",
            reason="gesture only",
            steps=(PlanStep(action=ActionType.WAVE),),
        )
        normalized = brain._normalize_gesture_dialogue(plan)
        self.assertEqual([step.action for step in normalized.steps], [ActionType.WAVE])

    def test_explicit_speak_without_gesture_is_unchanged(self) -> None:
        brain, _chat = make_brain([])
        plan = Plan(
            plan_id="plan_speak",
            goal_id="goal_speak",
            reason="speech only",
            steps=(PlanStep(action=ActionType.SPEAK, text="Hello there."),),
        )
        normalized = brain._normalize_gesture_dialogue(plan)
        self.assertEqual(
            [(step.action, step.text) for step in normalized.steps],
            [(ActionType.SPEAK, "Hello there.")],
        )

    def test_wave_plus_speak_does_not_gain_a_duplicate_speak(self) -> None:
        brain, _chat = make_brain([])
        plan = Plan(
            plan_id="plan_wave_speak",
            goal_id="goal_wave_speak",
            reason="gesture and reply",
            steps=(
                PlanStep(action=ActionType.WAVE, text="Hello there! You look great today!"),
                PlanStep(action=ActionType.SPEAK, text="Thank you for saying that."),
            ),
        )
        normalized = brain._normalize_gesture_dialogue(plan)
        self.assertEqual(
            [step.action for step in normalized.steps],
            [ActionType.WAVE, ActionType.SPEAK],
        )
        self.assertEqual(normalized.steps[-1].text, "Thank you for saying that.")


class ExistingBehaviorRegressionTests(unittest.TestCase):
    def test_visible_reply_and_speak_text_use_one_authoritative_line(self) -> None:
        brain, chat = make_brain(
            [
                {
                    "steps": [
                        {"action": "SPEAK", "text": "A different planner line."},
                        {"action": "LOOK_AT", "target": "CENTER"},
                    ]
                }
            ]
        )
        brain.chat.submit("viewer_quality", "Hello there!")
        brain.tick(world(), confirm(brain, None))
        plan = brain.state.current_plan
        self.assertIsNotNone(plan)
        spoken = [step.text for step in plan.steps if step.action == ActionType.SPEAK]
        self.assertEqual(spoken, [chat.get_state()["last_response"]["speech"]])

    def test_unsupported_shelf_target_is_still_rejected(self) -> None:
        world = build_world_state(snapshot())
        validation = BrainPlanner().validate_step(
            PlanStep(action=ActionType.MOVE_TO, target="SHELF"), world
        )
        self.assertFalse(validation.valid)
        self.assertEqual(validation.reason, "TARGET_NOT_FOUND")

    def test_duplicate_messages_are_still_merged(self) -> None:
        service = ChatService(MockLLMProvider(), ChatServiceConfig(max_messages_per_rate_window=10))
        first = service.submit("viewer_quality", "dance please", "msg_quality_1")
        second = service.submit("viewer_quality", "DANCE PLEASE!", "msg_quality_2")
        self.assertTrue(first.accepted)
        self.assertFalse(second.accepted)
        self.assertEqual(second.message.filtered_reason, "duplicate")
        self.assertIsNotNone(second.merged_message)


if __name__ == "__main__":
    unittest.main()

