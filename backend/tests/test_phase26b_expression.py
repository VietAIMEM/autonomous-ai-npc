"""PHASE 26B focused tests: expression / gesture / voice coordination.

These pin the contract this phase added on top of the existing systems:

* one deterministic expression decision per turn (tone, bounded reaction
  intensity, optional already-registered gesture) derived from the same single
  reasoning result -- never a second emotion/gesture/rewrite model call;
* a selected gesture is placed into the plan the same reasoning pass already
  built, immediately before its utterance, and a repeated gesture is refused
  inside a bounded suppression window;
* the bounded expression hint rides the existing Phase 19 personalization
  block into the one response prompt, and is truncated before it can grow;
* speech keeps its existing bounded queue, cancellation and IDLE recovery;
* diagnostics expose the coordination triple plus the observed gesture state.
"""

from __future__ import annotations

import json
import socket
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from backend.brain.expression import (
    GESTURE_DANCE,
    GESTURE_WAVE,
    ExpressionConfig,
    ExpressionDecision,
    ExpressionPolicy,
    ReactionIntensity,
)
from backend.brain.goals import GoalEvaluator
from backend.brain.loop import AutonomousBrain, BrainConfig, BrainStatus
from backend.brain.models import (
    ActionResult,
    ActionType,
    ExecutionStatus,
    GoalType,
    Plan,
    PlanStep,
)
from backend.brain.personalization import PersonalizedContext
from backend.brain.personality import load_personality
from backend.brain.planner import BrainPlanner
from backend.brain.reasoning import MockBrainLLMProvider
from backend.brain.world_state import build_world_state
from backend.chat.service import ChatService, ChatServiceConfig
from backend.chat.models import ChatIntent, ChatMessage
from backend.llm.mock import MockLLMProvider
from backend.social.conversation import ConversationManager
from backend.social.service import SocialAutonomyService
from backend.speech.models import SpeechState
from backend.speech.service import IDLE_STATE, SpeechConfig, SpeechService
from backend.tts.base import TTSProviderError
from backend.tts.config import PROVIDER_MOCK, TTSConfig
from backend.tts.mock import MockTTSProvider
from backend.admin_web.console import AdminConsole
from backend.api.brain import create_default_service
from backend.runtime.config import RuntimeConfig

from backend.tests import test_behavior as behavior
from backend.tests.dispatch_support import confirm


BASE_TIME = datetime(2026, 10, 8, 9, 0, tzinfo=timezone.utc)


def make_plan(*actions: ActionType) -> Plan:
    steps = tuple(PlanStep(action=action) for action in actions)
    return Plan(
        plan_id="plan_26b",
        goal_id="goal_26b",
        reason="phase26b test",
        steps=steps,
        created_at=behavior.BASE_TIME,
    )


def expression_decision(
    gesture: str | None = GESTURE_WAVE,
    *,
    gesture_reason: str = "greeting",
    intent: str | None = "GREETING",
    behavior_state: str = "CURIOUS",
) -> ExpressionDecision:
    return ExpressionDecision(
        tone="interested",
        reaction_intensity=ReactionIntensity.MEDIUM.value,
        gesture=gesture,
        gesture_reason=gesture_reason,
        intent=intent,
        behavior_state=behavior_state,
        attention_required=True,
        reason="greeting_while_curious",
    )


def snapshot(**overrides: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "character": {"position": [0, 0, 0], "rotation_y": 90, "current_action": "idle"},
        "phone": {"state": "Placed", "position": [1, 0.75, 0], "placement_surface": "TABLE"},
        "stream": {
            "is_streaming": True,
            "camera_available": True,
            "camera_active": True,
            "viewer_count": 1,
        },
        "targets": {
            "CENTER": {"position": [0, 0, 0], "navigation_target": True, "look_at_target": True},
            "TABLE": {"position": [1, 0.75, 0], "navigation_target": True, "look_at_target": True},
            "CAMERA": {"position": [1, 0.8, 0], "navigation_target": False, "look_at_target": True},
        },
    }
    value.update(overrides)
    return value


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


class ExpressionConfigTests(unittest.TestCase):
    def test_defaults_are_bounded(self) -> None:
        config = ExpressionConfig()
        self.assertEqual(config.gesture_suppression_seconds, 120.0)
        self.assertEqual(config.max_recent_gestures, 8)
        self.assertEqual(config.speech_hint_limit, 160)

    def test_negative_suppression_window_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ExpressionConfig(gesture_suppression_seconds=-0.1)

    def test_non_positive_recent_gesture_bound_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ExpressionConfig(max_recent_gestures=0)

    def test_non_positive_hint_limit_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ExpressionConfig(speech_hint_limit=0)


class ExpressionDecisionTests(unittest.TestCase):
    """Tone / intensity / hint stay deterministic and explainable."""

    def setUp(self) -> None:
        self.clock = behavior.FakeClock(BASE_TIME)
        self.policy = ExpressionPolicy(clock=self.clock)

    def decide(self, **overrides: Any) -> ExpressionDecision:
        arguments: dict[str, Any] = {
            "intent": "GREETING",
            "behavior_state": "CURIOUS",
            "personality": load_personality(),
        }
        arguments.update(overrides)
        return self.policy.decide(**arguments)

    def test_default_personality_greeting_is_medium_and_waves(self) -> None:
        decision = self.decide(attention_allowed=True)
        self.assertEqual(decision.tone, "interested")
        self.assertEqual(decision.reaction_intensity, ReactionIntensity.MEDIUM.value)
        self.assertEqual(decision.gesture, GESTURE_WAVE)
        self.assertEqual(decision.gesture_reason, "greeting")
        self.assertTrue(decision.attention_required)
        self.assertEqual(decision.reason, "greeting_while_curious")
        self.assertEqual(decision.intent, "GREETING")
        self.assertEqual(set(decision.to_dict()), {
            "tone",
            "reaction_intensity",
            "gesture",
            "gesture_reason",
            "intent",
            "behavior_state",
            "attention_required",
            "reason",
        })

    def test_positive_reaction_is_warm_and_waves(self) -> None:
        decision = self.decide(intent="COMMENT", behavior_state="HAPPY")
        self.assertEqual(decision.tone, "warm")
        self.assertEqual(decision.reaction_intensity, ReactionIntensity.MEDIUM.value)
        self.assertEqual(decision.gesture, GESTURE_WAVE)
        self.assertEqual(decision.gesture_reason, "positive_reaction")
        self.assertEqual(decision.reason, "comment_while_happy")

    def test_unexpected_event_reaches_high_without_a_gesture(self) -> None:
        decision = self.decide(intent=None, behavior_state="SURPRISED")
        self.assertEqual(decision.tone, "reactive")
        self.assertEqual(decision.reaction_intensity, ReactionIntensity.HIGH.value)
        self.assertIsNone(decision.gesture)
        self.assertEqual(decision.gesture_reason, "no_compatible_gesture")
        self.assertEqual(decision.reason, "surprised_idle")
        self.assertFalse(decision.attention_required)

    def test_task_and_tired_states_stay_low(self) -> None:
        focused = self.decide(intent="REQUEST_MOVEMENT", behavior_state="FOCUSED")
        self.assertEqual(focused.tone, "focused")
        self.assertEqual(focused.reaction_intensity, ReactionIntensity.LOW.value)
        tired = self.decide(intent=None, behavior_state="TIRED")
        self.assertEqual(tired.tone, "quiet")
        self.assertEqual(tired.reaction_intensity, ReactionIntensity.LOW.value)

    def test_calm_personality_bounds_everyday_reactions(self) -> None:
        calm = {"playfulness": 0.2, "energy": 0.4, "calmness": 0.95}
        decision = self.decide(personality=calm)
        self.assertEqual(decision.reaction_intensity, ReactionIntensity.LOW.value)
        self.assertEqual(decision.gesture, GESTURE_WAVE)

    def test_discrete_viewer_reaction_keeps_medium_read_under_a_calm_personality(
        self,
    ) -> None:
        calm = {"playfulness": 0.2, "energy": 0.4, "calmness": 0.95}
        positive = self.decide(
            intent=None,
            behavior_state="HAPPY",
            behavior_reason="positive_viewer_reaction",
            personality=calm,
        )
        self.assertEqual(positive.reaction_intensity, ReactionIntensity.MEDIUM.value)
        self.assertEqual(positive.tone, "warm")
        negative = self.decide(
            intent=None,
            behavior_state="SURPRISED",
            behavior_reason="negative_viewer_reaction",
            personality=calm,
        )
        self.assertEqual(negative.reaction_intensity, ReactionIntensity.HIGH.value)
        self.assertEqual(negative.tone, "reactive")

    def test_non_reaction_reason_still_respects_the_calm_damper(self) -> None:
        calm = {"playfulness": 0.2, "energy": 0.4, "calmness": 0.95}
        decision = self.decide(
            intent=None,
            behavior_state="HAPPY",
            behavior_reason="relevant_memory",
            personality=calm,
        )
        self.assertEqual(decision.reaction_intensity, ReactionIntensity.LOW.value)

    def test_returning_viewer_lifts_the_low_bound(self) -> None:
        calm = {"playfulness": 0.2, "energy": 0.4, "calmness": 0.95}
        returning = PersonalizedContext(
            known=True,
            interaction_count=3,
            relationship_level="FAMILIAR",
        )
        decision = self.decide(personality=calm, personalization=returning)
        self.assertEqual(decision.reaction_intensity, ReactionIntensity.MEDIUM.value)

    def test_playful_energetic_personality_clamps_at_high(self) -> None:
        lively = {"playfulness": 0.9, "energy": 0.9, "calmness": 0.5}
        decision = self.decide(personality=lively)
        self.assertEqual(decision.reaction_intensity, ReactionIntensity.HIGH.value)
        question = self.decide(intent="QUESTION", personality=lively)
        self.assertEqual(question.reaction_intensity, ReactionIntensity.LOW.value)

    def test_known_viewer_warms_a_neutral_state(self) -> None:
        unknown = self.decide(intent=None, behavior_state="CALM")
        self.assertEqual(unknown.tone, "neutral")
        known = self.decide(
            intent=None,
            behavior_state="CALM",
            personalization=PersonalizedContext(
                known=True,
                interaction_count=6,
                relationship_level="REGULAR",
            ),
        )
        self.assertEqual(known.tone, "warm")

    def test_attention_flag_follows_the_behavior_window(self) -> None:
        question_without_window = self.decide(
            intent="QUESTION", behavior_state="CALM", attention_allowed=False
        )
        self.assertFalse(question_without_window.attention_required)
        question_with_window = self.decide(
            intent="QUESTION", behavior_state="CALM", attention_allowed=True
        )
        self.assertTrue(question_with_window.attention_required)
        curious_idle = self.decide(
            intent=None, behavior_state="CURIOUS", attention_allowed=True
        )
        self.assertTrue(curious_idle.attention_required)

    def test_speech_hint_is_bounded_and_carries_style(self) -> None:
        decision = self.decide()
        hint = decision.speech_hint("warm and concise")
        self.assertIn("tone interested", hint)
        self.assertIn("reaction medium", hint)
        self.assertIn("because greeting_while_curious", hint)
        self.assertIn("style warm and concise", hint)
        self.assertLessEqual(len(hint), 160)

    def test_speech_hint_without_style_and_with_custom_limit(self) -> None:
        decision = self.decide()
        self.assertNotIn("style", decision.speech_hint())
        self.assertLessEqual(len(decision.speech_hint(limit=12)), 12)
        self.assertEqual(len(decision.speech_hint(limit=0)), 1)


class GestureSelectionTests(unittest.TestCase):
    """Selection is deterministic and refuses spam without silencing her."""

    def setUp(self) -> None:
        self.clock = behavior.FakeClock(BASE_TIME)
        self.policy = ExpressionPolicy(clock=self.clock)

    def decide(self, **overrides: Any) -> ExpressionDecision:
        arguments: dict[str, Any] = {
            "intent": "GREETING",
            "behavior_state": "CURIOUS",
            "personality": load_personality(),
        }
        arguments.update(overrides)
        return self.policy.decide(**arguments)

    def test_greeting_and_farewell_reuse_the_registered_wave(self) -> None:
        greeting = self.decide()
        self.assertEqual(greeting.gesture, GESTURE_WAVE)
        self.assertEqual(greeting.gesture_reason, "greeting")
        farewell = self.decide(intent="FAREWELL", behavior_state="ENGAGED")
        self.assertEqual(farewell.gesture, GESTURE_WAVE)
        self.assertEqual(farewell.gesture_reason, "farewell")
        ending = self.decide(intent="CONVERSATION_END", behavior_state="CALM")
        self.assertEqual(ending.gesture_reason, "farewell")

    def test_engaged_idle_tick_may_wave_once(self) -> None:
        decision = self.decide(intent=None, behavior_state="ENGAGED")
        self.assertEqual(decision.gesture, GESTURE_WAVE)
        self.assertEqual(decision.gesture_reason, "engagement")

    def test_no_candidate_keeps_the_turn_speech_only(self) -> None:
        decision = self.decide(intent="REQUEST_MOVEMENT", behavior_state="FOCUSED")
        self.assertIsNone(decision.gesture)
        self.assertEqual(decision.gesture_reason, "no_compatible_gesture")
        self.assertEqual(decision.tone, "focused")

    def test_busy_character_never_stacks_a_gesture(self) -> None:
        busy = self.decide(current_action="walking")
        self.assertIsNone(busy.gesture)
        self.assertEqual(busy.gesture_reason, "character_busy")
        idle = self.decide(current_action="idle")
        self.assertEqual(idle.gesture, GESTURE_WAVE)

    def test_unavailable_gesture_skips_the_gesture_only(self) -> None:
        refused = self.decide(availability=lambda name: False)
        self.assertIsNone(refused.gesture)
        self.assertEqual(refused.gesture_reason, "gesture_unavailable")
        self.assertEqual(refused.reaction_intensity, ReactionIntensity.MEDIUM.value)
        allowed = self.decide(availability=lambda name: name == GESTURE_WAVE)
        self.assertEqual(allowed.gesture, GESTURE_WAVE)

    def test_availability_fault_skips_the_gesture_without_raising(self) -> None:
        def explode(name: str) -> bool:
            raise RuntimeError("availability fault")

        decision = self.decide(availability=explode)
        self.assertIsNone(decision.gesture)
        self.assertEqual(decision.gesture_reason, "gesture_unavailable")

    def test_repeated_gesture_is_suppressed_then_allowed_again(self) -> None:
        self.policy.note_gesture(GESTURE_WAVE, self.clock())
        self.assertTrue(self.policy.suppressed(GESTURE_WAVE, self.clock()))
        self.assertEqual(self.policy.last_gesture, GESTURE_WAVE)
        repeat = self.decide()
        self.assertIsNone(repeat.gesture)
        self.assertEqual(repeat.gesture_reason, "recent_gesture_suppressed")
        self.clock.advance(120)
        self.assertFalse(self.policy.suppressed(GESTURE_WAVE, self.clock()))
        self.assertEqual(self.decide().gesture, GESTURE_WAVE)

    def test_zero_window_disables_suppression(self) -> None:
        policy = ExpressionPolicy(
            ExpressionConfig(gesture_suppression_seconds=0.0), clock=self.clock
        )
        policy.note_gesture(GESTURE_WAVE, self.clock())
        self.assertFalse(policy.suppressed(GESTURE_WAVE, self.clock()))

    def test_blank_gesture_notes_are_ignored(self) -> None:
        self.policy.note_gesture(None)
        self.policy.note_gesture("   ")
        self.assertEqual(self.policy.recent_gestures(), [])
        self.assertIsNone(self.policy.last_gesture)

    def test_snapshot_round_trip_restores_suppression(self) -> None:
        self.policy.note_gesture(GESTURE_WAVE, self.clock())
        payload = json.loads(json.dumps(self.policy.snapshot()))
        restored = ExpressionPolicy(clock=self.clock)
        restored.restore(payload)
        self.assertEqual(restored.recent_gestures(), self.policy.recent_gestures())
        self.assertTrue(restored.suppressed(GESTURE_WAVE, self.clock()))

    def test_restore_ignores_malformed_payloads(self) -> None:
        policy = ExpressionPolicy(clock=self.clock)
        policy.restore(None)
        policy.restore(["not", "a", "mapping"])  # type: ignore[arg-type]
        policy.restore(
            {
                "recent_gestures": [
                    "wave",
                    {"gesture": "", "timestamp": "2026-10-08T09:00:00+00:00"},
                    {"gesture": GESTURE_WAVE, "timestamp": "not-a-date"},
                ]
            }
        )
        self.assertEqual(policy.recent_gestures(), [])
        self.assertIsNone(policy.last_gesture)

    def test_recent_gesture_history_is_bounded(self) -> None:
        policy = ExpressionPolicy(
            ExpressionConfig(max_recent_gestures=3), clock=self.clock
        )
        for name in (GESTURE_WAVE, GESTURE_DANCE, GESTURE_WAVE, GESTURE_DANCE, GESTURE_WAVE):
            policy.note_gesture(name, self.clock())
        self.assertEqual(len(policy.recent_gestures()), 3)
        self.assertEqual(policy.last_gesture, GESTURE_WAVE)
        self.assertEqual(
            [item["gesture"] for item in policy.recent_gestures()],
            [GESTURE_WAVE, GESTURE_DANCE, GESTURE_WAVE],
        )


class GesturePlanCoordinationTests(behavior.BehaviorTestBase):
    """The policy decides; the existing plan is only repositioned."""

    def setUp(self) -> None:
        super().setUp()
        self.instance, self.chat, self.social = self.brain()

    def coordinate(self, plan: Plan, decision: ExpressionDecision | None = None, *, goal_type=GoalType.RESPOND_TO_VIEWER):
        return self.instance._coordinate_gesture(
            plan, behavior.goal(goal_type), decision
        )

    def test_wave_is_placed_immediately_before_the_utterance(self) -> None:
        plan = make_plan(ActionType.LOOK_AT, ActionType.SPEAK)
        coordinated = self.coordinate(plan, expression_decision())
        self.assertEqual(
            [step.action for step in coordinated.steps],
            [ActionType.LOOK_AT, ActionType.WAVE, ActionType.SPEAK],
        )
        self.assertIs(coordinated.steps[0], plan.steps[0])

    def test_dance_decision_uses_the_existing_dance_action(self) -> None:
        plan = make_plan(ActionType.SPEAK)
        coordinated = self.coordinate(
            plan,
            expression_decision(GESTURE_DANCE, gesture_reason="positive_reaction"),
        )
        self.assertEqual(
            [step.action for step in coordinated.steps],
            [ActionType.DANCE, ActionType.SPEAK],
        )

    def test_gesture_spelling_is_normalized(self) -> None:
        plan = make_plan(ActionType.SPEAK)
        coordinated = self.coordinate(
            plan, expression_decision(" WAVE ", gesture_reason="greeting")
        )
        self.assertEqual(
            [step.action for step in coordinated.steps],
            [ActionType.WAVE, ActionType.SPEAK],
        )

    def test_skips_when_the_decision_has_nothing_usable(self) -> None:
        plan = make_plan(ActionType.SPEAK)
        self.assertIs(self.coordinate(plan, None), plan)
        self.assertIs(
            self.coordinate(
                plan, expression_decision(None, gesture_reason="no_compatible_gesture")
            ),
            plan,
        )
        self.assertIs(
            self.coordinate(
                plan, expression_decision("kick", gesture_reason="no_compatible_gesture")
            ),
            plan,
        )

    def test_skips_plans_that_already_move(self) -> None:
        for action in (ActionType.WAVE, ActionType.DANCE):
            plan = make_plan(ActionType.LOOK_AT, action, ActionType.SPEAK)
            self.assertIs(self.coordinate(plan, expression_decision()), plan)

    def test_skips_task_and_stream_control_plans(self) -> None:
        for action in (ActionType.STOP, ActionType.START_STREAM, ActionType.STOP_STREAM):
            plan = make_plan(action, ActionType.SPEAK)
            self.assertIs(self.coordinate(plan, expression_decision()), plan)

    def test_skips_when_there_is_no_utterance_or_viewer_goal(self) -> None:
        silent = make_plan(ActionType.LOOK_AT)
        self.assertIs(self.coordinate(silent, expression_decision()), silent)
        task = make_plan(ActionType.SPEAK)
        self.assertIs(
            self.coordinate(
                task, expression_decision(), goal_type=GoalType.PERFORM_ACTIVITY
            ),
            task,
        )

    def test_respects_the_existing_plan_step_bound(self) -> None:
        full = make_plan(*(list([ActionType.MOVE_TO] * 7) + [ActionType.SPEAK]))
        self.assertIs(self.coordinate(full, expression_decision()), full)
        roomy = make_plan(*(list([ActionType.MOVE_TO] * 6) + [ActionType.SPEAK]))
        coordinated = self.coordinate(roomy, expression_decision())
        self.assertEqual(len(coordinated.steps), self.instance.config.max_plan_steps)
        self.assertEqual(coordinated.steps[-2].action, ActionType.WAVE)
        self.assertEqual(coordinated.steps[-1].action, ActionType.SPEAK)


class ExpressionTurnFlowTests(behavior.BehaviorTestBase):
    """End-to-end Brain turns keep gesture, speech and state reconciled."""

    def greeting_world(self):
        return behavior.world(streaming=True, viewer_count=1)

    def test_repeated_greeting_suppresses_the_wave_but_keeps_the_voice(self) -> None:
        speech = SpeechService(
            tts_config=TTSConfig(provider=PROVIDER_MOCK), start_worker=False
        )
        self.addCleanup(speech.shutdown)
        instance, chat, _social = self.brain(speech=speech)
        chat.submit("viewer_01", "hello lacrimosa")
        self.assertEqual(instance.tick(self.greeting_world()).action.action, ActionType.LOOK_AT)
        self.assertEqual(
            instance.tick(self.greeting_world(), confirm(instance, behavior.success(ActionType.LOOK_AT))).action.action,
            ActionType.WAVE,
        )
        self.assertEqual(instance.state.active_gesture, GESTURE_WAVE)
        self.assertEqual(
            instance.tick(self.greeting_world(), confirm(instance, behavior.success(ActionType.WAVE))).action.action,
            ActionType.SPEAK,
        )
        done = instance.tick(self.greeting_world(), confirm(instance, behavior.success(ActionType.SPEAK)))
        self.assertEqual(done.status, BrainStatus.IDLE)
        self.assertIsNone(instance.state.active_gesture)

        # A second greeting inside the suppression window must not repeat the
        # wave; the turn still speaks.
        self.clock.advance(16)
        chat.submit("viewer_01", "hey lacrimosa")
        repeat = instance.tick(self.greeting_world())
        self.assertEqual(repeat.status, BrainStatus.EXECUTING)
        self.assertIsNotNone(instance.state.current_plan)
        actions = [step.action for step in instance.state.current_plan.steps]
        self.assertNotIn(ActionType.WAVE, actions)
        self.assertIn(ActionType.SPEAK, actions)
        decision = instance.state.expression_decision
        self.assertIsNotNone(decision)
        self.assertIsNone(decision.gesture)
        self.assertEqual(decision.gesture_reason, "recent_gesture_suppressed")
        self.assertIsNone(instance.state.active_gesture)

    def test_expression_policy_fault_keeps_the_turn_alive(self) -> None:
        speech = SpeechService(
            tts_config=TTSConfig(provider=PROVIDER_MOCK), start_worker=False
        )
        self.addCleanup(speech.shutdown)
        instance, chat, _social = self.brain(speech=speech)

        def explode(*args: Any, **kwargs: Any) -> ExpressionDecision:
            raise RuntimeError("expression fault")

        instance.expression.decide = explode  # type: ignore[assignment]
        chat.submit("viewer_01", "hello lacrimosa")
        tick = instance.tick(self.greeting_world())
        self.assertEqual(tick.status, BrainStatus.EXECUTING)
        self.assertIsNone(instance.state.expression_decision)
        self.assertNotIn("expression", instance.last_behavior_context)
        actions = [step.action for step in instance.state.current_plan.steps]
        self.assertNotIn(ActionType.WAVE, actions)
        self.assertIn(ActionType.SPEAK, actions)

    def test_rejected_gesture_does_not_lose_the_utterance(self) -> None:
        speech = SpeechService(
            tts_config=TTSConfig(provider=PROVIDER_MOCK), start_worker=False
        )
        self.addCleanup(speech.shutdown)
        instance, chat, _social = self.brain(speech=speech)
        chat.submit("viewer_01", "hello lacrimosa")
        self.assertEqual(instance.tick(self.greeting_world()).action.action, ActionType.LOOK_AT)
        self.assertEqual(
            instance.tick(self.greeting_world(), confirm(instance, behavior.success(ActionType.LOOK_AT))).action.action,
            ActionType.WAVE,
        )
        # The gesture is decoration for the utterance: when Unity rejects it,
        # the greeting must still be spoken instead of being cancelled.
        after = instance.tick(self.greeting_world(), confirm(instance, behavior.failure(ActionType.WAVE)))
        self.assertEqual(after.status, BrainStatus.EXECUTING)
        self.assertEqual(after.action.action, ActionType.SPEAK)
        self.assertIsNotNone(after.action.speech_request_id)
        self.assertIsNone(instance.state.active_gesture)
        done = instance.tick(self.greeting_world(), confirm(instance, behavior.success(ActionType.SPEAK)))
        self.assertEqual(done.status, BrainStatus.IDLE)
        self.assertIsNone(instance.state.active_gesture)

    def test_interruption_reconciles_gesture_and_playing_speech(self) -> None:
        speech = SpeechService(
            tts_config=TTSConfig(provider=PROVIDER_MOCK), start_worker=False
        )
        self.addCleanup(speech.shutdown)
        instance, chat, _social = self.brain(speech=speech)
        chat.submit("viewer_01", "hello lacrimosa")
        instance.tick(self.greeting_world())
        instance.tick(self.greeting_world(), confirm(instance, behavior.success(ActionType.LOOK_AT)))
        speak = instance.tick(self.greeting_world(), confirm(instance, behavior.success(ActionType.WAVE)))
        self.assertEqual(speak.action.action, ActionType.SPEAK)
        request_id = speak.action.speech_request_id
        self.assertIsNotNone(request_id)
        self.assertTrue(speech.run_pending_once())
        self.assertEqual(speech.state(request_id), SpeechState.READY)
        self.assertEqual(instance.state.active_gesture, GESTURE_WAVE)

        chat.submit("viewer_01", "Lacrimosa, can you dance?")
        turn = instance.tick(self.greeting_world(), None)
        self.assertEqual(instance.state.interruption_reason, "interrupted_by_viewer")
        self.assertEqual(speech.state(request_id), SpeechState.CANCELLED)
        # The replacement turn owns the new gesture; the old wave is gone.
        self.assertEqual(instance.state.active_gesture, GESTURE_DANCE)
        state = speech.get_state()
        self.assertEqual(state["state"], IDLE_STATE)
        self.assertIsNone(state["playing_request_id"])
        self.assertEqual(turn.status, BrainStatus.EXECUTING)


class ViewerSentimentWiringTests(behavior.BehaviorTestBase):
    """Viewer wording is the honest sentiment source on the live chat path."""

    def stream_world(self):
        return behavior.world(streaming=True, viewer_count=1)

    def test_punctuated_praise_and_complaint_still_classify(self) -> None:
        conversation = ConversationManager()
        self.assertEqual(
            conversation.classify_reaction(behavior.chat_message("That looks great!")),
            "positive",
        )
        self.assertEqual(
            conversation.classify_reaction(behavior.chat_message("That was terrible.")),
            "negative",
        )
        self.assertEqual(
            conversation.classify_reaction(behavior.chat_message("what are you doing?")),
            "request_followup",
        )

    def test_positive_viewer_wording_reaches_happy_behavior(self) -> None:
        instance, chat, social = self.brain()
        chat.submit("viewer_01", "That looks great!")
        instance.tick(self.stream_world())

        self.assertEqual(instance.behavior.state, behavior.BehaviorState.HAPPY)
        self.assertEqual(
            instance.behavior.transition_reason, "positive_viewer_reaction"
        )
        kinds = [reaction.kind for reaction in social.context.recent_reactions]
        self.assertEqual(kinds.count("positive"), 1)
        decision = instance.state.expression_decision
        self.assertIsNotNone(decision)
        self.assertEqual(decision.tone, "warm")
        self.assertEqual(decision.reaction_intensity, "MEDIUM")

    def test_negative_viewer_wording_reaches_surprised_behavior(self) -> None:
        instance, chat, social = self.brain()
        chat.submit("viewer_01", "That was terrible.")
        instance.tick(self.stream_world())

        self.assertEqual(instance.behavior.state, behavior.BehaviorState.SURPRISED)
        self.assertEqual(
            instance.behavior.transition_reason, "negative_viewer_reaction"
        )
        kinds = [reaction.kind for reaction in social.context.recent_reactions]
        self.assertEqual(kinds.count("negative"), 1)
        decision = instance.state.expression_decision
        self.assertIsNotNone(decision)
        self.assertEqual(decision.tone, "reactive")
        self.assertEqual(decision.reaction_intensity, "HIGH")

    def test_reaction_reason_is_bounded_to_its_own_window(self) -> None:
        instance, _chat, _social = self.brain()
        instance.behavior.transition_reason = "positive_viewer_reaction"
        instance.behavior.transitioned_at = self.clock()
        context = {"transition_reason": "positive_viewer_reaction"}
        self.assertEqual(instance._behavior_reason(context), "positive_viewer_reaction")

        self.clock.advance(31)
        self.assertIsNone(instance._behavior_reason(context))

        self.assertEqual(
            instance._behavior_reason({"transition_reason": "new_viewer_or_topic"}),
            "new_viewer_or_topic",
        )
        self.assertIsNone(instance._behavior_reason({}))
        self.assertIsNone(instance._behavior_reason({"transition_reason": "  "}))

    def test_neutral_and_request_wording_fabricate_no_reaction(self) -> None:
        instance, chat, social = self.brain()
        chat.submit("viewer_01", "hello lacrimosa")
        instance.tick(self.stream_world())
        self.clock.advance(11)
        chat.submit("viewer_01", "Lacrimosa, can you dance?")
        instance.tick(self.stream_world())

        kinds = [reaction.kind for reaction in social.context.recent_reactions]
        self.assertNotIn("positive", kinds)
        self.assertNotIn("negative", kinds)


class ExpressionPromptTests(behavior.BehaviorTestBase):
    """One reasoning pass carries the hint; no second model call happens."""

    def test_chat_personalization_adds_one_bounded_hint(self) -> None:
        instance, _chat, _social = self.brain()
        personalization = PersonalizedContext(
            known=True,
            interaction_count=4,
            relationship_level="FAMILIAR",
            speech_style="warm and concise",
        )
        decision = instance.expression.decide(
            intent="GREETING",
            behavior_state="CURIOUS",
            personality=load_personality(),
            personalization=personalization,
            attention_allowed=True,
        )
        block = instance._chat_personalization(personalization, decision)
        self.assertIsInstance(block, dict)
        self.assertEqual(block["relationship_level"], "FAMILIAR")
        self.assertEqual(
            set(block["expression"]), {"tone", "reaction_intensity", "hint"}
        )
        self.assertIn("tone interested", block["expression"]["hint"])
        self.assertIn("style warm and concise", block["expression"]["hint"])
        # The Phase 19 prompt block is copied, never extended in place.
        self.assertNotIn("expression", personalization.to_prompt_block())

    def test_without_a_decision_the_phase19_block_is_unchanged(self) -> None:
        instance, _chat, _social = self.brain()
        personalization = PersonalizedContext(
            known=True,
            interaction_count=2,
            relationship_level="FAMILIAR",
            speech_style="warm and concise",
        )
        self.assertEqual(
            instance._chat_personalization(personalization, None),
            personalization.to_prompt_block(),
        )
        self.assertIsNone(instance._chat_personalization(None, None))

    def test_expression_hint_reaches_the_single_response_prompt(self) -> None:
        recorded: list[Any] = []

        class RecordingChatProvider:
            def __init__(self) -> None:
                self.inner = MockLLMProvider()

            def generate(self, request: Any) -> str:
                recorded.append(request)
                return self.inner.generate(request)

        chat = ChatService(
            RecordingChatProvider(),
            ChatServiceConfig(max_messages_per_rate_window=20),
            clock=self.clock,
        )
        planner_provider = behavior.CapturingProvider()
        instance = AutonomousBrain(
            planner_provider,
            chat,
            BrainConfig(decision_interval_seconds=0.0),
            personality=load_personality(),
            clock=self.clock,
            memory_service=self.memory,
            social_service=SocialAutonomyService(clock=self.clock),
            skill_library=self.library,
            learning_service=self.learning,
        )
        chat.submit("viewer_01", "hello lacrimosa")
        tick = instance.tick(behavior.world(streaming=True, viewer_count=1))
        self.assertEqual(tick.status, BrainStatus.EXECUTING)
        # Exactly the pre-existing two provider passes: one chat response and
        # one planning prompt. Phase 26B adds no model call of its own.
        self.assertEqual(len(recorded), 1)
        self.assertEqual(len(planner_provider.requests), 1)
        prompt = recorded[0].prompt
        self.assertIn("Speech style:", prompt)
        self.assertIn("reaction medium", prompt)
        self.assertIn("because greeting_while_", prompt)

        planner_prompt = planner_provider.requests[0].prompt
        self.assertIn('"expression"', planner_prompt)
        self.assertIn('"tone":"interested"', planner_prompt)
        self.assertIn('"gesture":"wave"', planner_prompt)


class PlannerExpressionPromptTests(behavior.BehaviorTestBase):
    def test_expression_block_rides_the_single_planner_prompt(self) -> None:
        planner = BrainPlanner()
        prompt = planner.build_prompt(
            behavior.world(),
            behavior.goal(GoalType.INTERACT_WITH_VIEWER),
            load_personality(),
            "hello lacrimosa",
            None,
            None,
            None,
            {
                "state": "CURIOUS",
                "previous_state": "CALM",
                "transition_reason": "new_viewer_or_topic",
                "speech_style": "interested and attentive",
                "preferred_skills": ["look_at", "wave", "speak"],
                "proactive_allowed": False,
                "reactive_allowed": False,
                "attention_allowed": True,
                "expression": {
                    "tone": "interested",
                    "reaction_intensity": "MEDIUM",
                    "gesture": "wave",
                    "gesture_reason": "greeting",
                },
                "rule": "advisory only",
            },
        )
        payload = json.loads(prompt)
        self.assertEqual(
            payload["behavior"]["expression"],
            {
                "tone": "interested",
                "reaction_intensity": "MEDIUM",
                "gesture": "wave",
                "gesture_reason": "greeting",
            },
        )

    def test_behavior_block_without_expression_keeps_prior_shape(self) -> None:
        planner = BrainPlanner()
        prompt = planner.build_prompt(
            behavior.world(),
            behavior.goal(GoalType.RESPOND_TO_VIEWER),
            load_personality(),
            "hello lacrimosa",
            None,
            None,
            None,
            {
                "state": "CALM",
                "previous_state": None,
                "transition_reason": "no_meaningful_behavior_event",
                "speech_style": "neutral, concise wording",
                "preferred_skills": ["speak"],
                "proactive_allowed": False,
                "reactive_allowed": False,
                "attention_allowed": False,
                "rule": "advisory only",
            },
        )
        payload = json.loads(prompt)
        self.assertNotIn("expression", payload["behavior"])


class WorldGestureEvidenceTests(unittest.TestCase):
    def test_reported_gesture_evidence_round_trips(self) -> None:
        raw = behavior.world_snapshot()
        raw["character"]["last_gesture"] = "Waving"
        raw["character"]["last_gesture_at"] = "2026-10-08T09:00:05+00:00"
        raw["character"]["gesture_active"] = True
        state = build_world_state(raw)
        self.assertEqual(state.last_gesture, "Waving")
        self.assertEqual(state.last_gesture_at, "2026-10-08T09:00:05+00:00")
        self.assertIs(state.gesture_active, True)
        character = state.to_dict()["character"]
        self.assertEqual(character["last_gesture"], "Waving")
        self.assertEqual(character["last_gesture_at"], "2026-10-08T09:00:05+00:00")
        self.assertIs(character["gesture_active"], True)

    def test_legacy_snapshot_keeps_its_prior_shape(self) -> None:
        state = build_world_state(behavior.world_snapshot())
        self.assertIsNone(state.last_gesture)
        self.assertIsNone(state.gesture_active)
        character = state.to_dict()["character"]
        for key in ("last_gesture", "last_gesture_at", "gesture_active"):
            self.assertNotIn(key, character)

    def test_blank_gesture_evidence_stays_unknown(self) -> None:
        raw = behavior.world_snapshot()
        raw["character"]["last_gesture"] = "   "
        state = build_world_state(raw)
        self.assertIsNone(state.last_gesture)


class ChatExpressionContextTests(unittest.TestCase):
    """The Phase 26B hint reuses the existing single style prompt field."""

    def setUp(self) -> None:
        self.chat = ChatService(
            MockLLMProvider(), ChatServiceConfig(max_messages_per_rate_window=20)
        )

    def test_expression_hint_overrides_the_style_for_one_turn(self) -> None:
        hint = "tone warm; reaction medium; because greeting_while_happy"
        context = self.chat._character_context(
            {
                "speech_style": "positive and energetic",
                "expression": {
                    "tone": "warm",
                    "reaction_intensity": "MEDIUM",
                    "hint": hint,
                },
            }
        )
        self.assertEqual(context.speech_style, hint)
        self.assertNotEqual(self.chat.get_character_context().speech_style, hint)

    def test_expression_hint_is_truncated_and_falls_back(self) -> None:
        context = self.chat._character_context(
            {"expression": {"hint": "x" * 400}}
        )
        self.assertEqual(context.speech_style, "x" * 160)
        fallback = self.chat._character_context(
            {
                "speech_style": "warm and concise",
                "expression": {"hint": "   "},
            }
        )
        self.assertEqual(fallback.speech_style, "warm and concise")
        self.assertIs(self.chat._character_context(None), self.chat.get_character_context())


class GoalAttributionTests(unittest.TestCase):
    """Every chat-derived goal names the message it was evaluated for.

    PHASE 26A documented ``goal.context["message_id"]`` as the surface a plan
    is attributed to its viewer message through, and the acceptance probe reads
    exactly that. Real DeepSeek turns routed through the intent fallback (an
    unrecognised comment such as "That looks great!") previously produced an
    explore/idle goal with an empty context, which orphaned the plan -- and the
    speech request it created -- from the turn that caused it.
    """

    def _evaluator(self, **kwargs: Any) -> GoalEvaluator:
        return GoalEvaluator(load_personality(), **kwargs)

    def _message(self, intent: ChatIntent, message_id: str = "msg_000123") -> ChatMessage:
        return ChatMessage(
            message_id,
            "viewer_26b",
            "That looks great!",
            "that looks great!",
            BASE_TIME,
            priority=0.4,
            intent=intent,
        )

    def test_intent_fallback_goals_carry_message_attribution(self) -> None:
        message = self._message(ChatIntent.UNKNOWN)
        evaluator = self._evaluator(clock=lambda: BASE_TIME)
        world = build_world_state(snapshot())
        explore = evaluator.evaluate(world, message)
        self.assertEqual(explore.goal_type, GoalType.EXPLORE_ROOM)
        self.assertEqual(explore.context.get("message_id"), message.message_id)
        self.assertEqual(explore.context.get("viewer_id"), message.viewer_id)
        idle = evaluator.evaluate(world, message)
        self.assertEqual(idle.goal_type, GoalType.IDLE)
        self.assertEqual(idle.context.get("message_id"), message.message_id)

    def test_every_chat_intent_keeps_the_same_attribution_surface(self) -> None:
        world = build_world_state(snapshot())
        intents = (
            ChatIntent.GREETING,
            ChatIntent.COMMENT,
            ChatIntent.ACKNOWLEDGEMENT,
            ChatIntent.QUESTION,
            ChatIntent.REQUEST_ACTION,
            ChatIntent.FAREWELL,
            ChatIntent.CONVERSATION_END,
            ChatIntent.UNKNOWN,
        )
        for intent in intents:
            with self.subTest(intent=intent.value):
                message = self._message(intent)
                goal = self._evaluator().evaluate(world, message)
                self.assertIsNotNone(goal)
                self.assertEqual(goal.context.get("message_id"), message.message_id)
                self.assertEqual(goal.context.get("viewer_id"), message.viewer_id)

    def test_autonomous_goals_stay_unattributed(self) -> None:
        world = build_world_state(snapshot())
        goal = self._evaluator(clock=lambda: BASE_TIME).evaluate(world)
        self.assertEqual(goal.goal_type, GoalType.EXPLORE_ROOM)
        self.assertNotIn("message_id", goal.context)


class SpeechQueueTests(unittest.TestCase):
    """The existing bounded queue/cancel lifecycle stays intact."""

    def setUp(self) -> None:
        self.speech = SpeechService(
            tts_config=TTSConfig(provider=PROVIDER_MOCK), start_worker=False
        )
        self.addCleanup(self.speech.shutdown)

    def test_queue_capacity_is_bounded(self) -> None:
        speech = SpeechService(
            tts_config=TTSConfig(provider=PROVIDER_MOCK),
            config=SpeechConfig(queue_capacity=1),
            start_worker=False,
        )
        self.addCleanup(speech.shutdown)
        first = speech.request("first utterance")
        self.assertEqual(first.state, SpeechState.SPEECH_REQUESTED)
        second = speech.request("second utterance")
        self.assertEqual(second.state, SpeechState.FAILED)
        self.assertEqual(second.error, "speech_queue_full")
        self.assertEqual(speech.pending_count(), 1)
        state = speech.get_state()
        self.assertEqual(state["queue_length"], 1)
        self.assertEqual(state["queue_capacity"], 1)
        self.assertGreaterEqual(state["counters"]["failed"], 1)

    def test_in_flight_dedup_reuses_one_request(self) -> None:
        first = self.speech.request("same utterance", dedup_key="plan:0:abc")
        second = self.speech.request("same utterance", dedup_key="plan:0:abc")
        self.assertIs(first, second)
        self.assertEqual(self.speech.pending_count(), 1)
        self.assertEqual(self.speech.get_state()["counters"]["reused"], 1)

    def test_ready_to_completed_cycle_recovers_to_idle(self) -> None:
        request = self.speech.request("hello viewer")
        self.assertTrue(self.speech.run_pending_once())
        self.assertEqual(self.speech.state(request.request_id), SpeechState.READY)
        self.assertTrue(self.speech.mark_playing(request.request_id))
        self.assertEqual(
            self.speech.get_state()["state"], SpeechState.PLAYING.value
        )
        self.assertEqual(
            self.speech.get_state()["playing_request_id"], request.request_id
        )
        self.assertTrue(self.speech.mark_completed(request.request_id))
        state = self.speech.get_state()
        self.assertEqual(state["state"], IDLE_STATE)
        self.assertIsNone(state["playing_request_id"])
        self.assertGreaterEqual(state["counters"]["completed"], 1)

    def test_cancelled_utterance_is_never_synthesized(self) -> None:
        first = self.speech.request("first utterance")
        second = self.speech.request("second utterance")
        self.assertTrue(self.speech.cancel(first.request_id, "interrupted_by_viewer"))
        self.assertEqual(self.speech.state(first.request_id), SpeechState.CANCELLED)
        self.assertFalse(self.speech.cancel(first.request_id, "again"))
        self.assertTrue(self.speech.run_pending_once())
        self.assertEqual(self.speech.state(first.request_id), SpeechState.CANCELLED)
        self.assertTrue(self.speech.run_pending_once())
        self.assertEqual(self.speech.state(second.request_id), SpeechState.READY)
        self.assertEqual(self.speech.get_state()["counters"]["cancelled"], 1)

    def test_interrupt_current_respects_the_interruptible_flag(self) -> None:
        protected = self.speech.request("protected line", interruptible=False)
        self.speech.run_pending_once()
        self.speech.mark_playing(protected.request_id)
        self.assertIsNone(self.speech.interrupt_current("new_speech"))
        self.assertEqual(self.speech.state(protected.request_id), SpeechState.PLAYING)
        self.speech.mark_completed(protected.request_id)

        interruptible = self.speech.request("interruptible line")
        self.speech.run_pending_once()
        self.speech.mark_playing(interruptible.request_id)
        self.assertEqual(
            self.speech.interrupt_current("new_speech"), interruptible.request_id
        )
        self.assertEqual(
            self.speech.state(interruptible.request_id), SpeechState.CANCELLED
        )
        self.assertEqual(self.speech.get_state()["state"], IDLE_STATE)

    def test_provider_failure_recovers_to_idle(self) -> None:
        class FailingProvider(MockTTSProvider):
            def synthesize(self, request: Any) -> Any:
                raise TTSProviderError("provider_error", "boom")

        speech = SpeechService(
            provider=FailingProvider(),
            tts_config=TTSConfig(provider=PROVIDER_MOCK),
            start_worker=False,
        )
        self.addCleanup(speech.shutdown)
        request = speech.request("this will fail")
        self.assertTrue(speech.run_pending_once())
        self.assertEqual(speech.state(request.request_id), SpeechState.FAILED)
        self.assertEqual(request.error, "provider_error")
        state = speech.get_state()
        self.assertEqual(state["state"], IDLE_STATE)
        self.assertEqual(state["consecutive_failures"], 1)


class ExpressionDiagnosticsTests(unittest.TestCase):
    """The bounded read-models expose the coordination triple."""

    def setUp(self) -> None:
        self._temp = TemporaryDirectory()
        temp = Path(self._temp.name)
        self.config = RuntimeConfig(
            environment="test",
            data_dir=temp / "data",
            log_dir=temp / "logs",
            temp_dir=temp / "temp",
            state_dir=temp / "state",
            brain_port=_free_port(),
            admin_port=_free_port(),
        )
        self.service = create_default_service(
            database_path=temp / "learning.db",
            runtime_config=self.config,
        )
        self.console = AdminConsole(
            self.service, config=self.config, project_root=temp
        )

    def tearDown(self) -> None:
        self.service.speech_service.shutdown()
        self.service.brain.learning.close()
        self._temp.cleanup()

    def test_conversation_state_exposes_the_expression_triple(self) -> None:
        self.service.chat_service.submit("viewer_01", "Hi Lacrimosa!")
        self.service.tick(snapshot(), None)
        payload = self.service.get_conversation_state()
        self.assertEqual(len(payload), 20)
        self.assertEqual(payload["current_intent"], "GREETING")
        self.assertEqual(payload["behavior_state"], "CURIOUS")
        self.assertEqual(payload["response_tone"], "interested")
        self.assertEqual(payload["reaction_intensity"], "MEDIUM")
        self.assertEqual(payload["active_gesture"], GESTURE_WAVE)
        self.assertEqual(payload["gesture_reason"], "greeting")
        self.assertEqual(payload["speech_state"], IDLE_STATE)
        self.assertIsNone(payload["interruption_reason"])

    def test_active_gesture_clears_when_the_plan_completes(self) -> None:
        self.service.chat_service.submit("viewer_01", "Hi Lacrimosa!")
        self.service.tick(snapshot(), None)
        self.assertEqual(
            self.service.get_conversation_state()["active_gesture"], GESTURE_WAVE
        )
        for action in (ActionType.LOOK_AT, ActionType.WAVE, ActionType.SPEAK):
            pending = self.service.brain.state.pending_action
            self.service.tick(
                snapshot(),
                {
                    "action": action.value,
                    "status": ExecutionStatus.SUCCESS.value,
                    "reason": "reported",
                    "dispatch_id": (
                        pending.dispatch_id
                        if pending is not None and pending.action == action
                        else None
                    ),
                },
            )
        payload = self.service.get_conversation_state()
        self.assertIsNone(payload["active_gesture"])
        self.assertIsNone(payload["pending_action"])
        # The turn is over and nothing is left "speaking". The single
        # utterance stays in the existing pipeline until the Unity client
        # closes it through the same playing/completed callbacks the real
        # runtime uses; that callback is replayed here explicitly.
        self.assertNotEqual(payload["speech_state"], SpeechState.PLAYING.value)
        speech_state = self.service.get_speech_state()
        request_id = speech_state["active_request_id"]
        self.assertIsNotNone(request_id)
        self.assertIsNone(speech_state["playing_request_id"])
        deadline = time.monotonic() + 2.0
        while (
            self.service.get_speech_state()["state"] != SpeechState.READY.value
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        self.assertTrue(self.service.mark_speech_playing(request_id))
        self.assertTrue(self.service.mark_speech_completed(request_id))
        self.assertEqual(self.service.get_speech_state()["state"], IDLE_STATE)

    def test_admin_overview_exposes_the_same_bounded_block(self) -> None:
        self.service.chat_service.submit("viewer_01", "Hi Lacrimosa!")
        self.service.tick(snapshot(), None)
        block = self.console.overview()["conversation"]
        self.assertEqual(len(block), 15)
        self.assertEqual(block["response_tone"], "interested")
        self.assertEqual(block["reaction_intensity"], "MEDIUM")
        self.assertEqual(block["behavior_state"], "CURIOUS")
        self.assertEqual(block["active_gesture"], GESTURE_WAVE)
        self.assertEqual(block["gesture_reason"], "greeting")


if __name__ == "__main__":  # pragma: no cover - manual entry point
    unittest.main()
