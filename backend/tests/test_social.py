from __future__ import annotations

import json
import threading
import unittest
import urllib.error
import urllib.request
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from backend.api.brain import create_brain_server, create_default_service
from backend.brain.goals import GoalEvaluator
from backend.brain.loop import AutonomousBrain, BrainConfig, BrainStatus
from backend.brain.models import ActionResult, ActionType, ExecutionStatus
from backend.brain.personality import load_personality
from backend.brain.reasoning import MockBrainLLMProvider
from backend.brain.service import BrainService
from backend.brain.world_state import build_world_state
from backend.chat.intent import extract_intent
from backend.chat.models import ChatMessage
from backend.chat.normalization import normalize_text
from backend.chat.priority import score_priority
from backend.chat.service import ChatService, ChatServiceConfig
from backend.llm.mock import MockLLMProvider
from backend.memory.models import MemoryConfig
from backend.memory.service import MemoryService
from backend.memory.sqlite_repository import SQLiteMemoryRepository
from backend.social.attention import SocialAttention, assess_camera_framing
from backend.social.conversation import ConversationManager
from backend.social.context import SocialContext
from backend.social.models import (
    AttentionDecision,
    ConversationState,
    SocialConfig,
    SocialSpeakingPolicy,
    ViewerSocialContext,
)
from backend.social.service import SocialAutonomyService


BASE_TIME = datetime(2026, 9, 9, 10, 0, tzinfo=timezone.utc)


class FakeClock:
    def __init__(self, value: datetime = BASE_TIME) -> None:
        self.value = value

    def __call__(self) -> datetime:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += timedelta(seconds=seconds)


def chat_message(
    text: str,
    viewer_id: str = "viewer_01",
    message_id: str = "msg_001",
    timestamp: datetime | None = None,
    repeat_count: int = 1,
    priority: float | None = None,
) -> ChatMessage:
    normalized = normalize_text(text)
    value = ChatMessage(
        message_id=message_id,
        viewer_id=viewer_id,
        raw_text=text,
        normalized_text=normalized,
        timestamp=timestamp or BASE_TIME,
        repeat_count=repeat_count,
    )
    intent = extract_intent(normalized)
    value.intent = intent.intent
    value.target = intent.target
    value.priority = priority if priority is not None else score_priority(value)
    return value


def world_snapshot(**overrides: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "character": {
            "position": [0, 0, 0],
            "rotation_y": 90,
            "posture": "standing",
            "current_action": "idle",
        },
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
            "CAMERA": {"position": [2, 0.8, 0], "navigation_target": False, "look_at_target": True},
        },
    }
    value.update(overrides)
    return value


def social_brain(clock: FakeClock) -> tuple[AutonomousBrain, ChatService]:
    chat = ChatService(
        MockLLMProvider(),
        ChatServiceConfig(max_messages_per_rate_window=20),
        clock=clock,
    )
    social = SocialAutonomyService(clock=clock)
    brain = AutonomousBrain(
        MockBrainLLMProvider(),
        chat,
        BrainConfig(decision_interval_seconds=0.0),
        personality=load_personality(),
        clock=clock,
        social_service=social,
    )
    return brain, chat


@contextmanager
def temporary_memory_service():
    with TemporaryDirectory() as root:
        config = MemoryConfig(database_path=Path(root) / "memory.db")
        service = MemoryService(SQLiteMemoryRepository(config.database_path), config)
        try:
            yield service
        finally:
            service.close()


class SocialConfigTests(unittest.TestCase):
    def test_invalid_config_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SocialConfig(attention_threshold=0.8, attend_now_threshold=0.5)
        with self.assertRaises(ValueError):
            SocialConfig(max_recent_messages=0)


class SocialContextTests(unittest.TestCase):
    def test_buffers_and_engagement_remain_bounded(self) -> None:
        context = SocialContext(
            SocialConfig(
                max_recent_messages=2,
                max_recent_reactions=1,
                max_viewer_recent_topics=2,
            )
        )
        for index in range(3):
            context.record_message(
                chat_message("dance", message_id=f"m{index}"),
                BASE_TIME,
            )
        context.note_reaction("viewer_01", "positive", BASE_TIME)
        context.note_reaction("viewer_01", "negative", BASE_TIME)
        context.record_topic(chat_message("wave"), "waving", BASE_TIME)
        context.record_topic(chat_message("dance"), "dance", BASE_TIME)

        self.assertEqual(len(context.recent_messages), 2)
        self.assertEqual(len(context.recent_reactions), 1)
        self.assertEqual(context.viewer_topics["viewer_01"], ["waving", "dance"])
        self.assertGreaterEqual(context.social_engagement, 0.0)
        self.assertLessEqual(context.social_engagement, 1.0)

    def test_active_conversation_expires_to_idle(self) -> None:
        context = SocialContext(SocialConfig(conversation_timeout_seconds=30))
        context.activate(
            chat_message("Lacrimosa, can you dance?"),
            "dance",
            BASE_TIME,
            ConversationState.CHANGING_TOPIC,
        )
        context.record_message(chat_message("Lacrimosa, can you dance?"), BASE_TIME)
        self.assertFalse(context.expire_if_needed(BASE_TIME + timedelta(seconds=29)))
        self.assertTrue(context.expire_if_needed(BASE_TIME + timedelta(seconds=31)))
        self.assertEqual(context.conversation_state, ConversationState.IDLE)
        self.assertIsNone(context.active_viewer_id)

    def test_farewell_ends_then_expires_to_idle(self) -> None:
        service = SocialAutonomyService(clock=FakeClock(BASE_TIME))
        message = chat_message("bye Lacrimosa")
        service.process_message(message, None)
        service.notify_processed(message, "Goodbye for now.", now=BASE_TIME)
        self.assertEqual(service.context.conversation_state, ConversationState.ENDING)
        self.assertTrue(
            service.context.expire_if_needed(BASE_TIME + timedelta(minutes=4))
        )
        self.assertEqual(service.context.conversation_state, ConversationState.IDLE)


class ConversationTests(unittest.TestCase):
    def test_topics_and_reactions_are_deterministic(self) -> None:
        conversation = ConversationManager()
        self.assertEqual(conversation.infer_topic(chat_message("dance")), "dance")
        self.assertEqual(conversation.infer_topic(chat_message("go to the table")), "table")
        self.assertEqual(conversation.infer_topic(chat_message("wave")), "waving")
        self.assertEqual(conversation.infer_topic(chat_message("bye")), "farewell")
        self.assertEqual(conversation.classify_reaction(chat_message("that was cute")), "positive")
        self.assertEqual(conversation.classify_reaction(chat_message("what are you doing?")), "request_followup")

    def test_activity_question_does_not_inherit_a_meta_topic(self) -> None:
        # PHASE 26A: "greeting" frames the conversation, it is not the subject
        # "What are you doing?" is about; a real subject still continues.
        conversation = ConversationManager()
        question = chat_message("what are you doing?")
        self.assertEqual(
            conversation.infer_topic(question, previous_topic="greeting"),
            "current_activity",
        )
        self.assertEqual(
            conversation.infer_topic(question, previous_topic="dance"), "dance"
        )

    def test_topic_continuation_distinguishes_meta_frames(self) -> None:
        # PHASE 26A: continuity is what attention scores, not literal equality.
        # Leaving a meta frame for a real subject is still the same thread; a
        # substantive topic switch is not.
        conversation = ConversationManager()
        self.assertTrue(conversation.is_topic_continuation("dance", "dance"))
        self.assertTrue(
            conversation.is_topic_continuation("current_activity", "greeting")
        )
        self.assertFalse(conversation.is_topic_continuation("camera", "dance"))
        self.assertFalse(conversation.is_topic_continuation(None, "dance"))
        self.assertFalse(conversation.is_topic_continuation("dance", None))


class AttentionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.attention = SocialAttention()
        self.context = SocialContext()

    def test_all_five_decisions(self) -> None:
        busy_world = build_world_state(world_snapshot(character={"current_action": "dancing"}))
        cases = [
            ("attend_now", chat_message("Lacrimosa can you dance?"), None, AttentionDecision.ATTEND_NOW),
            ("attend_soon", chat_message("what are you doing?", priority=0.45), None, AttentionDecision.ATTEND_SOON),
            ("ignore", chat_message("lol", message_id="low", priority=0.05), None, AttentionDecision.IGNORE),
            ("defer", chat_message("Lacrimosa can you dance?", message_id="busy"), busy_world, AttentionDecision.DEFER),
            ("merge", chat_message("dance again", repeat_count=2), None, AttentionDecision.MERGE),
        ]
        for name, message, world, expected in cases:
            with self.subTest(name=name):
                assessment = self.attention.evaluate(message, self.context, world, now=BASE_TIME)
                self.assertEqual(assessment.decision, expected)

    def test_repeated_low_value_message_is_not_promoted(self) -> None:
        message = chat_message("lol", message_id="low_repeat", priority=0.05, repeat_count=4)
        assessment = self.attention.evaluate(message, self.context, now=BASE_TIME)
        self.assertEqual(assessment.decision, AttentionDecision.IGNORE)

    def test_speech_cooldown_defers_rapid_high_value_reply(self) -> None:
        self.context.record_speech("viewer_01", "hello", BASE_TIME - timedelta(seconds=2))
        message = chat_message("Lacrimosa can you dance?", priority=0.75)
        first = self.attention.evaluate(message, self.context, now=BASE_TIME)
        self.assertEqual(first.decision, AttentionDecision.DEFER)
        self.assertEqual(first.speaking_policy, SocialSpeakingPolicy.DEFER)

        later = BASE_TIME + timedelta(seconds=13)
        second = self.attention.evaluate(message, self.context, now=later)
        self.assertEqual(second.decision, AttentionDecision.ATTEND_NOW)

    def test_follow_up_after_greeting_keeps_conversation_attention(self) -> None:
        # PHASE 26A: "What are you doing?" right after a greeting is the same
        # conversation gaining a subject; the speech cooldown must not defer it
        # like a cold topic switch.
        service = SocialAutonomyService(clock=FakeClock())
        service.process_message(
            chat_message("hello lacrimosa", message_id="msg_greeting"),
            None,
        )
        service.notify_processed(
            chat_message("hello lacrimosa", message_id="msg_greeting"),
            "Hello!",
            now=BASE_TIME,
        )
        service.clock.advance(11)
        decision = service.process_message(
            chat_message("what are you doing?", message_id="msg_follow_up"),
            None,
        )
        self.assertEqual(decision.topic, "current_activity")
        self.assertEqual(decision.decision, AttentionDecision.ATTEND_NOW)
        self.assertGreaterEqual(decision.attention_score, 0.8)

    def test_camera_awareness_uses_known_transforms(self) -> None:
        self.assertEqual(assess_camera_framing(None), "UNKNOWN")
        self.assertEqual(assess_camera_framing(build_world_state(world_snapshot(stream={"is_streaming": False}))), "STREAM_INACTIVE")
        far_camera = world_snapshot(
            targets={
                "CAMERA": {"position": [10, 0.8, 0], "navigation_target": False, "look_at_target": True}
            }
        )
        self.assertEqual(assess_camera_framing(build_world_state(far_camera)), "OUTSIDE_FRAME")


class SocialServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.service = SocialAutonomyService(clock=self.clock)

    def test_multi_viewer_context_does_not_corrupt_ids(self) -> None:
        first = chat_message("Lacrimosa can you dance?", "viewer_01", "m1")
        second = chat_message("Lacrimosa can you wave?", "viewer_02", "m2", BASE_TIME + timedelta(seconds=1))
        third = chat_message("Lacrimosa can you run?", "viewer_01", "m3", BASE_TIME + timedelta(seconds=2))

        self.assertEqual(self.service.process_message(first, None).viewer_id, "viewer_01")
        self.assertEqual(self.service.context.active_viewer_id, "viewer_01")
        self.clock.advance(1)
        self.assertEqual(self.service.process_message(second, None).viewer_id, "viewer_02")
        self.assertEqual(self.service.context.active_viewer_id, "viewer_02")
        self.clock.advance(1)
        self.assertEqual(self.service.process_message(third, None).viewer_id, "viewer_01")
        self.assertEqual(self.service.context.active_viewer_id, "viewer_01")
        self.assertEqual(self.service.context.viewer_topics["viewer_01"], ["dance", "running"])
        self.assertEqual(self.service.context.viewer_topics["viewer_02"], ["waving"])

    def test_viewer_memory_informs_but_does_not_control(self) -> None:
        with temporary_memory_service() as memory:
            memory.record_outcome(
                viewer_id="viewer_42",
                chat=chat_message("Lacrimosa, can you dance?", "viewer_42"),
                goal=None,
                action=ActionType.DANCE,
                success=True,
            )
            viewer_context = self.service.load_viewer_context("viewer_42", memory)
            self.assertEqual(viewer_context.viewer_id, "viewer_42")
            self.assertIn("dance", viewer_context.recent_topics)
            self.assertGreater(viewer_context.interaction_count, 0)

            decision = self.service.process_message(
                chat_message("Lacrimosa can you dance again?", "viewer_42", "m42"),
                None,
                memory,
            )
            self.assertEqual(decision.decision, AttentionDecision.ATTEND_NOW)

    def test_viewer_memory_failure_falls_back_safely(self) -> None:
        class BrokenMemory:
            def get_viewer(self, viewer_id: str) -> dict[str, Any]:
                raise RuntimeError("database unavailable")

        context = self.service.load_viewer_context("viewer_42", BrokenMemory())
        self.assertEqual(context, ViewerSocialContext("viewer_42"))

    def test_deferred_message_is_released_when_processed(self) -> None:
        message = chat_message("Lacrimosa can you wave?", message_id="pending")
        busy_world = build_world_state(world_snapshot(character={"current_action": "dancing"}))
        self.assertEqual(self.service.process_message(message, busy_world).decision, AttentionDecision.DEFER)
        self.assertEqual(len(self.service.context.pending_requests), 1)
        self.assertEqual(self.service.context.conversation_state, ConversationState.DEFERRED)

        assessment = SocialAttention().evaluate(message, self.service.context, None, now=BASE_TIME)
        self.assertEqual(assessment.decision, AttentionDecision.MERGE)
        self.service.context.release_pending(message.message_id)
        self.assertEqual(self.service.context.pending_requests, [])

    def test_social_state_contains_camera_and_context(self) -> None:
        self.service.update_world(build_world_state(world_snapshot()))
        state = self.service.get_state()
        self.assertEqual(state["camera_awareness"], "INSIDE_FRAME")
        self.assertIn("active_viewer_id", state["context"])
        self.assertIn("social_engagement", state["context"])

    def test_recent_viewer_count_tracks_window_and_expiry(self) -> None:
        self.service.process_message(chat_message("hello", "alice", "m1"), None)
        self.clock.advance(30)
        self.service.process_message(chat_message("hi", "bob", "m2"), None)
        self.assertEqual(self.service.recent_viewer_count(), 2)

        # Default conversation window is 180s: alice (190s) expired, bob (160s) still recent.
        self.clock.advance(160)
        self.assertEqual(self.service.recent_viewer_count(), 1)

        self.clock.advance(30)
        self.assertEqual(self.service.recent_viewer_count(), 0)

    def test_recent_viewer_count_refreshes_on_new_message(self) -> None:
        self.service.process_message(chat_message("hello", "alice", "m1"), None)
        self.clock.advance(100)
        self.service.process_message(chat_message("still here", "alice", "m2"), None)
        self.clock.advance(100)
        # Alice spoke again at +100, so she is 100s stale, not 200s.
        self.assertEqual(self.service.recent_viewer_count(), 1)

    def test_explicit_now_is_respected(self) -> None:
        self.service.process_message(chat_message("hello", "alice", "m1"), None)
        later = BASE_TIME + timedelta(seconds=181)
        self.assertEqual(self.service.recent_viewer_count(now=later), 0)
        sooner = BASE_TIME + timedelta(seconds=179)
        self.assertEqual(self.service.recent_viewer_count(now=sooner), 1)


class BrainViewerFloorTests(unittest.TestCase):
    def test_tick_floors_viewer_count_from_recent_chat_audience(self) -> None:
        clock = FakeClock()
        brain, chat = social_brain(clock)
        service = BrainService(brain, chat)

        brain.social.process_message(chat_message("hello", "alice", "m1"), None)
        clock.advance(5)
        brain.social.process_message(chat_message("hi", "bob", "m2"), None)

        service.tick(world_snapshot(stream={"is_streaming": True, "viewer_count": 0}), None)
        world = service.get_state()["world"]
        self.assertEqual(world["stream"]["viewer_count"], 2)

    def test_tick_keeps_higher_snapshot_viewer_count(self) -> None:
        clock = FakeClock()
        brain, chat = social_brain(clock)
        service = BrainService(brain, chat)

        brain.social.process_message(chat_message("hello", "alice", "m1"), None)
        service.tick(world_snapshot(stream={"is_streaming": True, "viewer_count": 40}), None)
        world = service.get_state()["world"]
        self.assertEqual(world["stream"]["viewer_count"], 40)


class BrainSocialIntegrationTests(unittest.TestCase):
    def test_busy_social_attention_defers_then_resumes(self) -> None:
        clock = FakeClock()
        brain, chat = social_brain(clock)
        # Keep the evaluator deterministic so the only available decision is
        # the deferred social request while Unity reports a busy action.
        brain.goal_evaluator._last_explore_at = clock.value
        chat.submit("viewer_01", "Lacrimosa can you wave?")
        deferred = brain.tick(
            build_world_state(world_snapshot(character={"current_action": "dancing"}))
        )
        self.assertEqual(deferred.status, BrainStatus.WAITING)
        self.assertEqual(deferred.message, "social_attention_deferred")
        self.assertEqual(len(chat.queue), 1)
        self.assertEqual(len(brain.social.context.pending_requests), 1)
        self.assertEqual(brain.social.context.conversation_state, ConversationState.DEFERRED)

        resumed = brain.tick(build_world_state(world_snapshot()))
        self.assertEqual(resumed.status, BrainStatus.EXECUTING)
        self.assertEqual(len(chat.queue), 0)
        self.assertEqual(brain.social.context.pending_requests, [])
        self.assertIsNotNone(resumed.action)


class CameraGoalTests(unittest.TestCase):
    def test_outside_frame_creates_camera_goal(self) -> None:
        far_camera = world_snapshot(
            targets={
                "CAMERA": {"position": [12, 0.8, 0], "navigation_target": False, "look_at_target": True}
            }
        )
        evaluator = GoalEvaluator(load_personality())
        selected = evaluator.evaluate(build_world_state(far_camera))
        self.assertEqual(selected.goal_type.value, "IMPROVE_CAMERA_POSITION")


class SocialApiTests(unittest.TestCase):
    def test_social_state_route(self) -> None:
        service = create_default_service()
        server = create_brain_server(service, "127.0.0.1", 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        host, port = server.server_address
        try:
            with urllib.request.urlopen(f"http://{host}:{port}/social/state", timeout=2) as response:
                payload = json.loads(response.read().decode("utf-8"))
            self.assertEqual(response.status, 200)
            self.assertIn("config", payload)
            self.assertIn("context", payload)
            self.assertEqual(payload["camera_awareness"], "UNKNOWN")
        finally:
            server.shutdown()
            thread.join(timeout=2)
            server.server_close()


if __name__ == "__main__":
    unittest.main()
