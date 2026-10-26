"""PHASE 26A: natural conversation / context / turn-taking regression.

Pins the behaviours this phase added on top of the existing Brain, chat queue,
social attention and speech systems:

* several viewer messages read as one bounded conversation (follow-ups, topic
  continuity, spatial continuations);
* the response policy keeps livestream replies short and rotates
  acknowledgements instead of repeating one hardcoded line;
* an important viewer message cancels in-flight work through the existing
  priority/cancellation machinery -- never a second interruption engine;
* every bounded conversation status recovers from success/failure/timeout/
  cancel/replan and returns to a coherent listening state.
"""

from __future__ import annotations

import itertools
import json
import socket
import threading
import unittest
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any

from backend.brain.loop import AutonomousBrain, BrainConfig, BrainStatus
from backend.brain.models import ActionType, ExecutionStatus, GoalStatus
from backend.brain.reasoning import MockBrainLLMProvider
from backend.brain.world_state import build_world_state
from backend.skills.library import create_default_skill_library
from backend.admin_web.console import AdminConsole
from backend.api.brain import create_brain_server, create_default_service
from backend.chat.context import ConversationContext
from backend.chat.dialogue import CharacterContext
from backend.chat.models import ChatIntent, ChatMessage, DialogueResponse
from backend.chat.normalization import normalize_text
from backend.chat.policy import (
    RESPONSE_CLASS_ACTION_ACK,
    RESPONSE_CLASS_FAREWELL,
    RESPONSE_CLASS_FOLLOW_UP,
    RESPONSE_CLASS_NORMAL_REPLY,
    RESPONSE_CLASS_SHORT_ACK,
    RESPONSE_CLASS_SHORT_REPLY,
    ResponsePolicyEngine,
)
from backend.chat.service import ChatService, ChatServiceConfig
from backend.llm.mock import MockLLMProvider
from backend.runtime.config import RuntimeConfig
from backend.social.service import SocialAutonomyService

BASE_TIME = datetime(2026, 10, 7, 12, 0, tzinfo=timezone.utc)

_MESSAGE_IDS = itertools.count(1)


def chat_message(
    text: str,
    intent: ChatIntent = ChatIntent.UNKNOWN,
    *,
    target: str | None = None,
    desired_state: str | None = None,
    viewer_id: str = "viewer_01",
    timestamp: datetime | None = None,
) -> ChatMessage:
    return ChatMessage(
        message_id=f"msg_{next(_MESSAGE_IDS):05d}",
        viewer_id=viewer_id,
        raw_text=text,
        normalized_text=normalize_text(text),
        timestamp=timestamp or BASE_TIME,
        intent=intent,
        target=target,
        desired_state=desired_state,
    )


def dialogue_response(
    message: ChatMessage,
    speech: str,
    requested_action: str | None = None,
) -> DialogueResponse:
    return DialogueResponse(
        speech=speech,
        intent=message.intent,
        requested_action=requested_action,
        priority=message.priority,
        message_id=message.message_id,
        viewer_id=message.viewer_id,
    )


def snapshot(**overrides: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "character": {"position": [0, 0, 0], "rotation_y": 90, "current_action": "idle"},
        "phone": {"state": "Placed", "position": [1, 0.75, 0], "placement_surface": "TABLE"},
        "stream": {"is_streaming": True, "camera_available": True},
        "targets": {
            "CENTER": {"position": [0, 0, 0], "navigation_target": True, "look_at_target": True},
            "TABLE": {"position": [1, 0.75, 0], "navigation_target": True, "look_at_target": True},
            "CAMERA": {"position": [1, 0.8, 0], "navigation_target": False, "look_at_target": True},
        },
    }
    value.update(overrides)
    return value


def world(**overrides: Any):
    return build_world_state(snapshot(**overrides))


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def build_chat(
    provider: Any = None,
    *,
    clock: Any = None,
    topic_resolver: Any = None,
) -> ChatService:
    config = ChatServiceConfig(max_messages_per_rate_window=20)
    if clock is None:
        return ChatService(provider or MockLLMProvider(), config, topic_resolver=topic_resolver)
    return ChatService(
        provider or MockLLMProvider(),
        config,
        clock=clock,
        topic_resolver=topic_resolver,
    )


class StubSpeech:
    """Minimal enabled speech service that records cancels and utterances."""

    enabled = True

    def __init__(self) -> None:
        self.requests: list[Any] = []
        self.interrupts: list[str] = []
        self.cancels: list[tuple[str, str]] = []

    def interrupt_current(self, reason: str) -> None:
        self.interrupts.append(reason)

    def request(self, text: str, **_: Any) -> Any:
        record = SimpleNamespace(request_id=f"speech_{len(self.requests) + 1}", text=text)
        self.requests.append(record)
        return record

    def cancel(self, request_id: str, reason: str) -> None:
        self.cancels.append((request_id, reason))


def build_brain(
    chat: ChatService,
    speech: Any = None,
    *,
    social: bool = True,
) -> AutonomousBrain:
    return AutonomousBrain(
        MockBrainLLMProvider(),
        chat,
        BrainConfig(decision_interval_seconds=0.0),
        skill_library=create_default_skill_library(),
        speech_service=speech,
        social_service=SocialAutonomyService() if social else None,
    )


class ConversationContextTests(unittest.TestCase):
    """The bounded context behind every continuity decision."""

    def test_snapshot_stays_a_bounded_list_of_turns(self) -> None:
        context = ConversationContext(max_turns=2)
        for index in range(4):
            message = chat_message(f"question {index}", ChatIntent.QUESTION)
            context.observe_message(message, now=BASE_TIME)
            context.observe_response(
                message,
                dialogue_response(message, f"answer {index}"),
                now=BASE_TIME,
            )
        turns = context.snapshot()
        self.assertIsInstance(turns, list)
        self.assertEqual(len(turns), 2)
        self.assertEqual(turns[0].text, "question 3")
        self.assertEqual(turns[1].speaker, "Lacrimosa")
        self.assertEqual(turns[1].text, "answer 3")

    def test_quiet_conversation_expires_back_to_idle(self) -> None:
        context = ConversationContext(timeout_seconds=180.0)
        message = chat_message("Hi Lacrimosa!", ChatIntent.GREETING)
        context.observe_message(message, now=BASE_TIME)
        context.observe_response(message, dialogue_response(message, "Hey!"), now=BASE_TIME)
        self.assertTrue(context.has_active_conversation())

        self.assertFalse(context.expire_if_needed(BASE_TIME + timedelta(seconds=179)))
        self.assertTrue(context.has_active_conversation())
        self.assertTrue(context.expire_if_needed(BASE_TIME + timedelta(seconds=180)))

        state = context.to_state()
        self.assertEqual(state["conversation_status"], "IDLE")
        self.assertEqual(state["response_status"], "IDLE")
        self.assertEqual(state["turn_count"], 0)
        self.assertIsNone(state["current_topic"])
        self.assertIsNone(state["active_viewer"])
        self.assertEqual(context.snapshot(), [])

    def test_idle_context_never_expires_again(self) -> None:
        context = ConversationContext(timeout_seconds=1.0)
        self.assertFalse(context.expire_if_needed(BASE_TIME + timedelta(days=1)))

    def test_recover_success_completes_the_turn(self) -> None:
        context = ConversationContext()
        message = chat_message("what are you doing?", ChatIntent.QUESTION)
        context.observe_message(message, now=BASE_TIME)
        self.assertEqual(context.response_status, "RESPONDING")
        context.recover("success", now=BASE_TIME + timedelta(seconds=1))
        self.assertEqual(context.response_status, "COMPLETED")

    def test_recover_failure_and_cancel_never_stay_responding(self) -> None:
        for reason in ("failure", "action_failed", "action_rejected", "cancel", "replan"):
            with self.subTest(reason=reason):
                context = ConversationContext()
                message = chat_message(
                    "Stand in front of the camera.",
                    ChatIntent.REQUEST_MOVEMENT,
                    target="camera",
                    desired_state="IN_FRAME",
                )
                context.observe_message(message, now=BASE_TIME)
                self.assertIsNotNone(context.pending_action)
                context.recover(reason, now=BASE_TIME + timedelta(seconds=1))
                self.assertEqual(context.response_status, "FALLBACK")
                # A cancelled/failed physical request never became a fact, so
                # the armed continuation context is dropped immediately.
                self.assertIsNone(context.pending_action)
                context.begin_response(message, now=BASE_TIME + timedelta(seconds=1))
                context.recover(reason, now=BASE_TIME + timedelta(seconds=1))
                self.assertEqual(context.response_status, "FALLBACK")
                self.assertIsNone(context.pending_action)
                self.assertIsNone(context.pending_action_target)
                self.assertFalse(context.pending_action_is_movement)
                self.assertEqual(context.conversation_status, "CONVERSING")

    def test_recover_timeout_clears_the_conversation(self) -> None:
        context = ConversationContext()
        message = chat_message("hello", ChatIntent.GREETING)
        context.observe_message(message, now=BASE_TIME)
        context.recover("timeout", now=BASE_TIME + timedelta(seconds=1))
        self.assertEqual(context.conversation_status, "IDLE")
        self.assertEqual(context.response_status, "IDLE")
        self.assertEqual(context.snapshot(), [])

    def test_recover_interruption_returns_to_waiting_with_history_kept(self) -> None:
        context = ConversationContext()
        first = chat_message("Hi Lacrimosa!", ChatIntent.GREETING)
        context.observe_message(first, now=BASE_TIME)
        context.observe_response(first, dialogue_response(first, "Hey!"), now=BASE_TIME)
        second = chat_message("Stand in front of the camera.", ChatIntent.REQUEST_MOVEMENT, target="camera")
        context.observe_message(second, now=BASE_TIME)
        context.recover("interrupted_by_viewer", now=BASE_TIME)
        self.assertEqual(context.response_status, "FALLBACK")
        self.assertEqual(context.conversation_status, "CONVERSING")
        self.assertNotEqual(context.response_status, "RESPONDING")
        # The viewer's history survives the interruption; only the unfulfilled
        # request is dropped.
        self.assertEqual(len(context.snapshot()), 2)
        self.assertEqual(context.previous_response, "Hey!")
        self.assertIsNone(context.pending_action)
        self.assertIsNone(context.pending_action_target)

    def test_discard_returns_a_waiting_conversation(self) -> None:
        context = ConversationContext()
        first = chat_message("Hi Lacrimosa!", ChatIntent.GREETING)
        context.observe_message(first, now=BASE_TIME)
        context.observe_response(first, dialogue_response(first, "Hey!"), now=BASE_TIME)
        second = chat_message("noise!!", ChatIntent.COMMENT, target="noise")
        context.observe_message(second, now=BASE_TIME)
        context.mark_discarded("social_attention_ignore", now=BASE_TIME)
        state = context.to_state(now=BASE_TIME + timedelta(seconds=1))
        self.assertEqual(state["response_status"], "DISCARDED")
        self.assertEqual(state["conversation_status"], "WAITING")

    def test_recover_success_is_reported_as_completed(self) -> None:
        context = ConversationContext()
        message = chat_message("go to the table", ChatIntent.REQUEST_MOVEMENT, target="table")
        context.observe_message(message, now=BASE_TIME)
        context.recover("success", now=BASE_TIME)
        self.assertEqual(context.response_status, "COMPLETED")


class IntentContinuityTests(unittest.TestCase):
    """Two viewer messages must read as one conversation."""

    def test_question_then_why_is_a_follow_up_in_the_same_topic(self) -> None:
        chat = build_chat()
        first = chat.submit("viewer_01", "What are you doing?")
        self.assertTrue(first.accepted)
        self.assertEqual(first.message.intent, ChatIntent.QUESTION)
        chat.process_next()
        topic = chat.get_conversation_state()["current_topic"]
        self.assertEqual(topic, "current_activity")

        second = chat.submit("viewer_01", "Why?")
        self.assertEqual(second.message.intent, ChatIntent.FOLLOW_UP)
        self.assertEqual(second.message.target, topic)
        state = chat.get_conversation_state()
        self.assertEqual(state["current_topic"], topic)
        self.assertEqual(state["active_viewer"], "viewer_01")

    def test_question_then_short_question_continues_the_topic(self) -> None:
        chat = build_chat()
        chat.submit("viewer_01", "Do you like dancing?")
        chat.process_next()
        self.assertEqual(chat.get_conversation_state()["current_topic"], "dance")

        second = chat.submit("viewer_01", "Is dancing fun?")
        self.assertEqual(second.message.intent, ChatIntent.TOPIC_CONTINUATION)
        self.assertEqual(second.message.target, "dance")
        self.assertEqual(chat.get_conversation_state()["current_topic"], "dance")

    def test_question_then_what_kind_is_a_follow_up(self) -> None:
        chat = build_chat()
        chat.submit("viewer_01", "Do you like dancing?")
        chat.process_next()
        second = chat.submit("viewer_01", "What kind?")
        self.assertEqual(second.message.intent, ChatIntent.FOLLOW_UP)
        self.assertEqual(chat.get_conversation_state()["current_topic"], "dance")

    def test_activity_question_after_a_greeting_starts_its_own_topic(self) -> None:
        # PHASE 26A: a greeting only frames the conversation. The wired social
        # resolver must not report "greeting" back as the subject that "Why?"
        # attaches to once the viewer asks about the current activity.
        resolver = SocialAutonomyService().conversation.infer_topic
        chat = build_chat(topic_resolver=resolver)
        chat.submit("viewer_01", "Hi Lacrimosa!")
        chat.process_next()
        self.assertEqual(chat.get_conversation_state()["current_topic"], "greeting")

        question = chat.submit("viewer_01", "What are you doing?")
        self.assertEqual(question.message.intent, ChatIntent.QUESTION)
        self.assertEqual(question.message.target, "current_activity")
        chat.process_next()
        self.assertEqual(
            chat.get_conversation_state()["current_topic"], "current_activity"
        )

        follow_up = chat.submit("viewer_01", "Why?")
        self.assertEqual(follow_up.message.intent, ChatIntent.FOLLOW_UP)
        self.assertEqual(follow_up.message.target, "current_activity")
        chat.process_next()
        self.assertEqual(
            chat.get_conversation_state()["current_topic"], "current_activity"
        )

        # The chat layer's own bounded fallback (no wired resolver) agrees.
        fallback = build_chat()
        fallback.submit("viewer_01", "Hi Lacrimosa!")
        fallback.process_next()
        fallback.submit("viewer_01", "What are you doing?")
        fallback.process_next()
        self.assertEqual(
            fallback.get_conversation_state()["current_topic"], "current_activity"
        )

    def test_activity_question_keeps_a_substantive_topic(self) -> None:
        # The same question inside a real subject ("Do you like dancing?") keeps
        # continuing that subject instead of resetting it.
        resolver = SocialAutonomyService().conversation.infer_topic
        chat = build_chat(topic_resolver=resolver)
        chat.submit("viewer_01", "Do you like dancing?")
        chat.process_next()
        self.assertEqual(chat.get_conversation_state()["current_topic"], "dance")

        chat.submit("viewer_01", "What are you doing?")
        self.assertEqual(chat.get_conversation_state()["current_topic"], "dance")

    def test_new_subject_is_visible_as_a_topic_change(self) -> None:
        chat = build_chat()
        chat.submit("viewer_01", "Do you like dancing?")
        chat.process_next()
        switch = chat.submit("viewer_01", "Can you stand in front of the camera?")
        self.assertEqual(switch.message.intent, ChatIntent.REQUEST_MOVEMENT)
        state = chat.get_conversation_state()
        self.assertEqual(state["conversation_status"], "CHANGING_TOPIC")
        self.assertEqual(state["current_topic"], "camera")

    def test_cold_conversation_never_invents_context(self) -> None:
        # PHASE 05 reading stays authoritative outside an active conversation.
        # Each phrase starts a brand new chat, so no earlier turn can make the
        # continuity resolver believe a conversation already exists.
        for text, expected in (
            ("Why?", ChatIntent.QUESTION),
            ("Closer.", ChatIntent.UNKNOWN),
            ("Okay.", ChatIntent.COMMENT),
            ("got it", ChatIntent.UNKNOWN),
        ):
            with self.subTest(text=text):
                cold = build_chat()
                self.assertEqual(
                    cold.submit("viewer_01", text).message.intent,
                    expected,
                )
        # No previous turn, so a cold "Why?" can only resolve to the generic
        # bucket -- never the invented topic of an imaginary conversation.
        cold = build_chat()
        cold.submit("viewer_01", "Why?")
        self.assertEqual(
            cold.get_conversation_state()["current_topic"],
            "general_conversation",
        )
        self.assertIsNone(cold.get_conversation_state()["previous_intent"])
        self.assertEqual(cold.get_conversation_state()["turn_count"], 0)

    def test_acknowledgement_only_upgrades_inside_an_active_conversation(self) -> None:
        chat = build_chat()
        chat.submit("viewer_01", "Hi Lacrimosa!")
        chat.process_next()
        for text in ("okay", "got it", "yeah", "sure"):
            with self.subTest(text=text):
                result = chat.submit("viewer_01", text)
                self.assertEqual(result.message.intent, ChatIntent.ACKNOWLEDGEMENT)

    def test_explicit_end_closes_the_conversation(self) -> None:
        chat = build_chat()
        chat.submit("viewer_01", "Hi Lacrimosa!")
        chat.process_next()
        end = chat.submit("viewer_01", "that's all")
        self.assertEqual(end.message.intent, ChatIntent.CONVERSATION_END)
        state = chat.get_conversation_state()
        self.assertEqual(state["conversation_status"], "ENDING")
        self.assertEqual(state["current_topic"], "farewell")

    def test_spatial_continuation_reuses_the_previous_camera_turn(self) -> None:
        chat = build_chat()
        first = chat.submit("viewer_01", "Stand in front of the camera.")
        self.assertEqual(first.message.intent, ChatIntent.REQUEST_MOVEMENT)
        self.assertEqual(first.message.target, "camera")
        chat.process_next()

        second = chat.submit("viewer_01", "Closer.")
        self.assertEqual(second.message.intent, ChatIntent.REQUEST_MOVEMENT)
        self.assertEqual(second.message.target, "camera")
        self.assertEqual(second.message.desired_state, "CLOSER")
        self.assertEqual(chat.get_conversation_state()["current_topic"], "camera")

    def test_spatial_continuation_works_before_the_first_turn_is_answered(self) -> None:
        # "Stand in front of the camera." -> "Closer." may arrive inside one
        # decision interval: the queued request already arms the spatial
        # context, so the continuation never falls through to UNKNOWN.
        chat = build_chat()
        chat.submit("viewer_01", "Stand in front of the camera.")
        second = chat.submit("viewer_01", "Closer.")
        self.assertEqual(second.message.intent, ChatIntent.REQUEST_MOVEMENT)
        self.assertEqual(second.message.target, "camera")
        self.assertEqual(second.message.desired_state, "CLOSER")
        self.assertGreater(second.message.priority, 0.7)

    def test_phone_anchor_is_inherited_by_the_continuation(self) -> None:
        chat = build_chat()
        chat.submit("viewer_01", "Move closer to the phone.")
        chat.process_next()
        second = chat.submit("viewer_01", "A little more.")
        self.assertEqual(second.message.intent, ChatIntent.REQUEST_MOVEMENT)
        self.assertEqual(second.message.target, "phone")
        self.assertEqual(second.message.desired_state, "CLOSER")


class ResponsePolicyTests(unittest.TestCase):
    """Bounded response class / length selection with a rotating ack."""

    def test_response_class_per_intent(self) -> None:
        engine = ResponsePolicyEngine()
        cases = (
            (chat_message("Hi Lacrimosa!", ChatIntent.GREETING), RESPONSE_CLASS_SHORT_REPLY, 20),
            (chat_message("yeah", ChatIntent.ACKNOWLEDGEMENT), RESPONSE_CLASS_SHORT_ACK, 6),
            (chat_message("Do you like dancing?", ChatIntent.QUESTION), RESPONSE_CLASS_SHORT_REPLY, 20),
            (chat_message("Why?", ChatIntent.FOLLOW_UP), RESPONSE_CLASS_FOLLOW_UP, 24),
            (
                chat_message("Can you dance?", ChatIntent.REQUEST_ACTION, target="dance"),
                RESPONSE_CLASS_ACTION_ACK,
                25,
            ),
            (chat_message("bye", ChatIntent.FAREWELL), RESPONSE_CLASS_FAREWELL, 14),
        )
        for message, expected_class, expected_budget in cases:
            with self.subTest(intent=message.intent.value):
                policy = engine.select(message, ConversationContext())
                self.assertEqual(policy.response_class, expected_class)
                self.assertEqual(policy.max_words, expected_budget)

    def test_long_question_gets_the_moderate_budget(self) -> None:
        engine = ResponsePolicyEngine()
        long_question = (
            "Could you explain in some detail what you think about dancing "
            "and music during a livestream tonight?"
        )
        policy = engine.select(chat_message(long_question, ChatIntent.QUESTION), None)
        self.assertEqual(policy.response_class, RESPONSE_CLASS_NORMAL_REPLY)
        self.assertEqual(policy.max_words, 45)

    def test_noise_reaction_is_a_one_word_turn(self) -> None:
        engine = ResponsePolicyEngine()
        policy = engine.select(
            chat_message("lol", ChatIntent.COMMENT, target="noise"),
            None,
        )
        self.assertEqual(policy.response_class, RESPONSE_CLASS_SHORT_ACK)
        self.assertEqual(policy.max_words, 6)

    def test_personality_and_relationship_only_adjust_bounded_budgets(self) -> None:
        engine = ResponsePolicyEngine()
        greeting = chat_message("Hi Lacrimosa!", ChatIntent.GREETING)
        playful = CharacterContext(personality=("playful", "curious"))
        self.assertEqual(engine.select(greeting, None, playful).max_words, 24)

        ack = chat_message("yeah", ChatIntent.ACKNOWLEDGEMENT)
        gentle = CharacterContext(personality=("calm", "gentle"))
        self.assertEqual(engine.select(ack, None, gentle).max_words, 5)

        warm = {"returning": True, "relationship_level": "FRIEND", "interaction_count": 9}
        self.assertEqual(engine.select(greeting, None, None, warm).max_words, 24)
        cold = {"returning": True, "relationship_level": "NEW"}
        self.assertEqual(engine.select(greeting, None, None, cold).max_words, 20)

    def test_budgets_stay_clamped(self) -> None:
        engine = ResponsePolicyEngine()
        ack = chat_message("yeah", ChatIntent.ACKNOWLEDGEMENT)
        quiet = CharacterContext(personality=("calm", "gentle"))
        policy = engine.select(ack, None, quiet)
        self.assertGreaterEqual(policy.max_words, 4)
        self.assertLessEqual(policy.max_words, 60)

    def test_follow_up_policy_carries_the_active_topic(self) -> None:
        engine = ResponsePolicyEngine()
        context = ConversationContext()
        first = chat_message(
            "What are you doing?",
            ChatIntent.QUESTION,
            target="current_activity",
        )
        context.observe_message(first, now=BASE_TIME)
        context.observe_response(
            first,
            dialogue_response(first, "I'm getting the camera ready."),
            now=BASE_TIME,
        )
        follow_up = chat_message("Why?", ChatIntent.FOLLOW_UP, target="current_activity")
        policy = engine.select(follow_up, context)
        self.assertEqual(policy.response_class, RESPONSE_CLASS_FOLLOW_UP)
        self.assertIn("Active topic: current_activity.", policy.constraint)
        self.assertIn("I'm getting the camera ready.", policy.constraint)

    def test_acknowledgement_hints_rotate_through_the_whole_set(self) -> None:
        engine = ResponsePolicyEngine(history_size=4)
        hints = [
            engine.select(
                chat_message(f"ok {index}", ChatIntent.ACKNOWLEDGEMENT),
                None,
            ).acknowledgement_hint
            for index in range(12)
        ]
        self.assertEqual(
            hints[:6],
            ["Sure.", "Okay.", "Got it.", "Alright.", "Yeah.", "One moment."],
        )
        for index in range(4, len(hints)):
            self.assertNotIn(hints[index], hints[index - 4 : index])
        self.assertEqual(engine.recent_hints(), tuple(hints[-4:]))

    def test_action_acknowledgements_get_a_hint_and_farewells_do_not(self) -> None:
        engine = ResponsePolicyEngine()
        action = engine.select(
            chat_message("Can you dance?", ChatIntent.REQUEST_ACTION, target="dance"),
            None,
        )
        self.assertIn(action.acknowledgement_hint, {"Sure.", "Okay.", "Got it.", "Alright.", "Yeah.", "One moment."})
        farewell = engine.select(chat_message("bye", ChatIntent.FAREWELL), None)
        self.assertIsNone(farewell.acknowledgement_hint)


class ChatServiceContinuityTests(unittest.TestCase):
    """Service-level reads diagnostics and the Brain rely on."""

    def test_state_publishes_conversation_and_response_policy(self) -> None:
        chat = build_chat()
        chat.submit("viewer_01", "Hi Lacrimosa!")
        self.assertIsNotNone(chat.process_next())
        state = chat.get_state()
        self.assertIn("conversation", state)
        self.assertIn("last_response_policy", state)
        conversation = state["conversation"]
        self.assertEqual(conversation["conversation_status"], "CONVERSING")
        self.assertEqual(conversation["previous_intent"], "GREETING")
        self.assertEqual(conversation["turn_count"], 1)
        self.assertLessEqual(conversation["conversation_turns"], chat.config.max_context_turns)
        policy = state["last_response_policy"]
        self.assertEqual(policy["response_class"], RESPONSE_CLASS_SHORT_REPLY)
        self.assertEqual(policy["max_words"], 24)
        self.assertIsNone(policy["acknowledgement_hint"])

    def test_discard_next_returns_a_waiting_conversation(self) -> None:
        chat = build_chat()
        chat.submit("viewer_01", "Hi Lacrimosa!")
        chat.process_next()
        chat.submit("viewer_01", "lol")
        dropped = chat.discard_next("social_attention_ignore")
        self.assertIsNotNone(dropped)
        self.assertEqual(dropped.filtered_reason, "social_attention_ignore")
        state = chat.get_conversation_state()
        self.assertEqual(state["response_status"], "DISCARDED")
        self.assertEqual(state["conversation_status"], "WAITING")
        self.assertEqual(chat.get_queue(), [])

    def test_wired_social_resolver_keeps_everyday_topics(self) -> None:
        # The production wiring points the chat pipeline at the social topic
        # vocabulary; everyday anchors must survive that override.
        resolver = SocialAutonomyService().conversation.infer_topic
        chat = build_chat(topic_resolver=resolver)
        chat.submit("viewer_01", "Do you like dancing?")
        self.assertEqual(chat.get_conversation_state()["current_topic"], "dance")
        chat.process_next()
        chat.submit("viewer_01", "What kind?")
        self.assertEqual(chat.get_conversation_state()["current_topic"], "dance")
        chat.process_next()
        chat.submit("viewer_01", "Lacrimosa, start the stream.")
        self.assertEqual(chat.get_conversation_state()["current_topic"], "stream")

    def test_wired_social_resolver_treats_end_of_conversation_like_a_farewell(self) -> None:
        resolver = SocialAutonomyService().conversation.infer_topic
        chat = build_chat(topic_resolver=resolver)
        chat.submit("viewer_01", "Hi Lacrimosa!")
        chat.process_next()
        chat.submit("viewer_01", "that's all")
        self.assertEqual(chat.get_conversation_state()["current_topic"], "farewell")

    def test_personalization_shapes_the_turn_without_persisting(self) -> None:
        recorded: list[Any] = []

        class RecordingProvider:
            def __init__(self) -> None:
                self.inner = MockLLMProvider()

            def generate(self, request: Any) -> str:
                recorded.append(request)
                return self.inner.generate(request)

        chat = build_chat(RecordingProvider())
        chat.submit("viewer_01", "Hi Lacrimosa!")
        chat.process_next(
            personalization={
                "returning": True,
                "relationship_level": "FRIEND",
                "interaction_count": 12,
                "preferred_topics": ["dance"],
                "topic_continuity": "dance",
                "greeting": "Welcome back!",
            }
        )
        self.assertIn("returning viewer, relationship FRIEND", recorded[0].prompt)
        self.assertIn("topics with repeat interest: dance", recorded[0].prompt)
        self.assertEqual(chat.get_state()["last_response_policy"]["max_words"], 28)

        # The base character context is never mutated by a personalized turn:
        # viewer history stays owned by the memory/relationship systems.
        self.assertEqual(chat.get_character_context().viewer_context, ())
        chat.submit("viewer_01", "hello again")
        chat.process_next()
        self.assertNotIn("returning viewer", recorded[1].prompt)
        self.assertEqual(chat.get_state()["last_response_policy"]["max_words"], 24)

    def test_provider_request_carries_the_policy_constraint(self) -> None:
        recorded: list[Any] = []

        class RecordingProvider:
            def __init__(self) -> None:
                self.inner = MockLLMProvider()

            def generate(self, request: Any) -> str:
                recorded.append(request)
                return self.inner.generate(request)

        chat = build_chat(RecordingProvider())
        chat.submit("viewer_01", "Hi Lacrimosa!")
        chat.process_next()
        self.assertIn("Reply in at most 24 words.", recorded[0].prompt)

    def test_conversation_recovers_after_the_quiet_timeout(self) -> None:
        now = {"value": BASE_TIME}
        chat = build_chat(clock=lambda: now["value"])
        chat.submit("viewer_01", "Hi Lacrimosa!")
        chat.process_next()
        self.assertEqual(chat.get_conversation_state()["turn_count"], 1)

        now["value"] = now["value"] + timedelta(seconds=181)
        expired = chat.get_conversation_state()
        self.assertEqual(expired["conversation_status"], "IDLE")
        self.assertEqual(expired["turn_count"], 0)

        restarted = chat.submit("viewer_01", "hello lacrimosa")
        self.assertEqual(restarted.message.intent, ChatIntent.GREETING)
        chat.process_next()
        self.assertEqual(chat.get_conversation_state()["turn_count"], 1)


class InterruptionTests(unittest.TestCase):
    """One interruption path: existing priority + cancellation + replanning."""

    def test_important_viewer_message_cancels_in_flight_work_once(self) -> None:
        chat = build_chat()
        speech = StubSpeech()
        brain = build_brain(chat, speech)

        chat.submit("viewer_01", "Hi Lacrimosa!")
        first = brain.tick(world(), None)
        self.assertEqual(first.status, BrainStatus.EXECUTING)
        old_goal = brain.state.current_goal
        self.assertIsNotNone(old_goal)
        first_plan = brain.state.current_plan
        self.assertIsNotNone(first_plan)
        first_actions = [step.action.value for step in first_plan.steps]
        self.assertIn("SPEAK", first_actions)

        request = chat.submit("viewer_01", "Lacrimosa, can you dance?")
        self.assertTrue(request.accepted)
        self.assertGreaterEqual(request.message.priority, brain.config.interruption_priority)

        second = brain.tick(world(), None)
        self.assertEqual(old_goal.status, GoalStatus.CANCELLED)
        self.assertEqual(brain.state.interruption_reason, "interrupted_by_viewer")
        self.assertIn("interrupted_by_viewer", speech.interrupts)
        self.assertIsNotNone(second.plan)
        self.assertEqual(
            [step.action.value for step in second.plan.steps],
            ["MOVE_TO", "LOOK_AT", "DANCE", "SPEAK"],
        )
        self.assertEqual(second.status, BrainStatus.EXECUTING)
        self.assertEqual(brain.state.pending_action.action, ActionType.MOVE_TO)
        # No duplicate work: the queue is drained and the dance turn was
        # answered exactly once.
        self.assertEqual(chat.get_queue(), [])
        conversation = chat.get_state()["conversation"]
        self.assertEqual(conversation["turn_count"], 2)
        self.assertEqual(conversation["current_topic"], "dance")
        self.assertNotEqual(conversation["response_status"], "RESPONDING")
        # No duplicate response: the greeting's SPEAK step never reached the
        # speech queue because the plan was cancelled while it was still on
        # its first (physical) step.
        self.assertEqual(speech.requests, [])

    def test_low_priority_message_waits_for_the_in_flight_plan(self) -> None:
        chat = build_chat()
        speech = StubSpeech()
        brain = build_brain(chat, speech)

        chat.submit("viewer_01", "Hi Lacrimosa!")
        brain.tick(world(), None)
        pending_before = brain.state.pending_action
        self.assertIsNotNone(pending_before)

        ack = chat.submit("viewer_01", "yeah")
        self.assertEqual(ack.message.intent, ChatIntent.ACKNOWLEDGEMENT)
        self.assertLess(ack.message.priority, brain.config.interruption_priority)

        tick = brain.tick(world(), None)
        self.assertEqual(tick.status, BrainStatus.EXECUTING)
        self.assertEqual(tick.message, "awaiting_action_result")
        self.assertEqual(brain.state.pending_action, pending_before)
        self.assertNotIn("interrupted_by_viewer", speech.interrupts)
        self.assertIsNone(brain.state.interruption_reason)
        self.assertEqual(len(chat.get_queue()), 1)

    def test_cancelled_plan_clears_the_conversation_response_state(self) -> None:
        chat = build_chat()
        speech = StubSpeech()
        brain = build_brain(chat, speech)
        chat.submit("viewer_01", "Hi Lacrimosa!")
        brain.tick(world(), None)
        chat.submit("viewer_01", "Lacrimosa, can you dance?")
        brain.tick(world(), None)

        self.assertEqual(brain.state.interruption_reason, "interrupted_by_viewer")
        # The interrupted turn was reconciled, not abandoned: the replacement
        # turn completed instead of leaving a half-open RESPONDING state.
        conversation = chat.get_conversation_state()
        self.assertEqual(conversation["response_status"], "COMPLETED")
        self.assertEqual(conversation["previous_intent"], "REQUEST_ACTION")
        self.assertEqual(conversation["turn_count"], 2)
        self.assertEqual(conversation["current_topic"], "dance")
        self.assertEqual(conversation["conversation_status"], "CONVERSING")
        self.assertEqual(chat.get_queue(), [])

    def test_arriving_message_interrupts_before_any_next_tick(self) -> None:
        """Unity only ticks between actions, so arrival itself must decide.

        On the real runtime no tick arrives while an utterance or movement is
        executing, so an important request queued during one would otherwise
        be answered only after it finished. The arrival path runs the same
        bounded priority gate and the same cancellation machinery.
        """
        chat = build_chat()
        speech = StubSpeech()
        brain = build_brain(chat, speech)

        chat.submit("viewer_01", "Hi Lacrimosa!")
        first = brain.tick(world(), None)
        self.assertEqual(first.status, BrainStatus.EXECUTING)
        pending = brain.state.pending_action
        self.assertIsNotNone(pending)

        request = chat.submit("viewer_01", "Lacrimosa, can you dance?")
        self.assertTrue(request.accepted)
        # No tick ran between the two lines: the arrival alone cancelled the
        # in-flight plan through the existing machinery.
        self.assertIsNone(brain.state.pending_action)
        self.assertEqual(brain.state.interruption_reason, "interrupted_by_viewer")
        self.assertEqual(brain.state.status, BrainStatus.REPLANNING)
        self.assertIn("interrupted_by_viewer", speech.interrupts)

        # The very next tick decides the replacement turn exactly once.
        second = brain.tick(world(), None)
        self.assertEqual(
            [step.action.value for step in second.plan.steps],
            ["MOVE_TO", "LOOK_AT", "DANCE", "SPEAK"],
        )
        self.assertEqual(brain.state.interruption_reason, "interrupted_by_viewer")
        self.assertEqual(chat.get_queue(), [])
        conversation = chat.get_state()["conversation"]
        self.assertEqual(conversation["turn_count"], 2)
        self.assertEqual(conversation["current_topic"], "dance")

    def test_arriving_low_priority_message_still_waits_its_turn(self) -> None:
        chat = build_chat()
        speech = StubSpeech()
        brain = build_brain(chat, speech)

        chat.submit("viewer_01", "Hi Lacrimosa!")
        brain.tick(world(), None)
        pending_before = brain.state.pending_action
        self.assertIsNotNone(pending_before)

        ack = chat.submit("viewer_01", "yeah")
        self.assertLess(ack.message.priority, brain.config.interruption_priority)
        self.assertEqual(brain.state.pending_action, pending_before)
        self.assertIsNone(brain.state.interruption_reason)
        self.assertEqual(speech.interrupts, [])
        self.assertEqual(len(chat.get_queue()), 1)

    def test_arrival_before_the_first_tick_is_left_to_the_tick(self) -> None:
        chat = build_chat()
        speech = StubSpeech()
        brain = build_brain(chat, speech)

        # No tick has ever reported a world, so there is nothing to interrupt.
        request = chat.submit("viewer_01", "Lacrimosa, can you dance?")
        self.assertTrue(request.accepted)
        self.assertEqual(speech.interrupts, [])

        first = brain.tick(world(), None)
        self.assertEqual(first.status, BrainStatus.EXECUTING)
        self.assertEqual(brain.state.interruption_reason, None)

    def test_a_listener_fault_never_breaks_intake(self) -> None:
        chat = build_chat()

        def explode(_: ChatMessage) -> None:
            raise RuntimeError("listener fault")

        chat.set_message_listener(explode)
        result = chat.submit("viewer_01", "Hi Lacrimosa!")
        self.assertTrue(result.accepted)
        self.assertEqual(len(chat.get_queue()), 1)


class DiagnosticsTests(unittest.TestCase):
    """The bounded read-model on the real HTTP surface and the admin console."""

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
        self.server = create_brain_server(self.service, "127.0.0.1", 0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address
        self.base = f"http://{host}:{port}"
        self.console = AdminConsole(self.service, config=self.config, project_root=temp)

    def tearDown(self) -> None:
        self.server.shutdown()
        self.thread.join(timeout=2)
        self.server.server_close()
        self.service.speech_service.shutdown()
        self.service.brain.learning.close()
        self._temp.cleanup()

    def _get(self, path: str) -> tuple[int, dict[str, Any]]:
        request = urllib.request.Request(f"{self.base}{path}", method="GET")
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read().decode("utf-8"))

    def test_conversation_state_route_exposes_bounded_fields(self) -> None:
        self.service.chat_service.submit("viewer_01", "Hi Lacrimosa!")
        self.service.tick(snapshot(), None)
        status, payload = self._get("/conversation/state")
        self.assertEqual(status, 200)
        self.assertEqual(
            set(payload),
            {
                "conversation_state",
                "current_intent",
                "current_topic",
                "response_status",
                "active_viewer",
                "turn_count",
                "pending_question",
                "pending_action",
                "pending_action_target",
                "interruption_reason",
                "speech_state",
                # PHASE 26B: bounded expression coordination fields.
                "response_tone",
                "reaction_intensity",
                "active_gesture",
                "gesture_reason",
                "behavior_state",
                "previous_intent",
                "previous_response",
                "last_interaction_at",
                "response_policy",
            },
        )
        self.assertEqual(payload["conversation_state"], "CONVERSING")
        self.assertEqual(payload["current_intent"], "GREETING")
        self.assertEqual(payload["active_viewer"], "viewer_01")
        self.assertEqual(payload["turn_count"], 1)
        self.assertIsNone(payload["interruption_reason"])
        self.assertEqual(payload["response_policy"]["response_class"], RESPONSE_CLASS_SHORT_REPLY)
        # Continuity stays bounded: no transcript is exposed beyond the
        # clipped previous response and pending question.
        self.assertNotIn("conversation_turns", payload)

    def test_interruption_reason_is_visible_then_cleared(self) -> None:
        self.service.chat_service.submit("viewer_01", "Hi Lacrimosa!")
        self.service.tick(snapshot(), None)
        self.service.chat_service.submit("viewer_01", "Lacrimosa, can you dance?")
        self.service.tick(snapshot(), None)
        status, payload = self._get("/conversation/state")
        self.assertEqual(status, 200)
        self.assertEqual(payload["interruption_reason"], "interrupted_by_viewer")
        self.assertEqual(payload["current_topic"], "dance")
        self.assertEqual(payload["current_intent"], "REQUEST_ACTION")
        self.assertEqual(payload["turn_count"], 2)

    def test_admin_overview_exposes_the_same_bounded_block(self) -> None:
        self.service.chat_service.submit("viewer_01", "Hi Lacrimosa!")
        self.service.tick(snapshot(), None)
        block = self.console.overview()["conversation"]
        self.assertEqual(
            set(block),
            {
                "conversation_state",
                "current_intent",
                "current_topic",
                "response_status",
                "active_viewer",
                "turn_count",
                "pending_question",
                "pending_action",
                "interruption_reason",
                "speech_state",
                "response_tone",
                "reaction_intensity",
                "active_gesture",
                "gesture_reason",
                "behavior_state",
            },
        )
        self.assertEqual(block["conversation_state"], "CONVERSING")
        self.assertEqual(block["active_viewer"], "viewer_01")
        self.assertEqual(block["turn_count"], 1)

    def test_conversation_diagnostics_stay_secret_free(self) -> None:
        self.service.chat_service.submit("viewer_01", "Hi Lacrimosa!")
        self.service.tick(snapshot(), None)
        _, payload = self._get("/conversation/state")
        encoded = json.dumps(payload)
        for marker in ("api_key", "API_KEY", "token", "password", "secret"):
            self.assertNotIn(marker, encoded)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
