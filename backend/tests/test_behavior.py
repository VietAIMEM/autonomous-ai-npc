"""PHASE 18 focused tests for the bounded advisory behavior state."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from backend.brain.behavior import BehaviorConfig, BehaviorManager, BehaviorState
from backend.brain.loop import AutonomousBrain, BrainConfig, BrainStatus
from backend.brain.models import (
    ActionResult,
    ActionType,
    ExecutionStatus,
    Goal,
    GoalStatus,
    GoalType,
    WorldState,
)
from backend.brain.personality import load_personality
from backend.brain.planner import BrainPlanner
from backend.brain.reasoning import BrainReasoner, MockBrainLLMProvider
from backend.brain.world_state import build_world_state
from backend.chat.intent import extract_intent
from backend.chat.models import ChatMessage
from backend.chat.normalization import normalize_text
from backend.chat.priority import score_priority
from backend.chat.service import ChatService, ChatServiceConfig
from backend.learning.models import LearningConfig
from backend.learning.repository import SQLiteSkillLearningRepository
from backend.learning.service import SkillLearningService
from backend.llm.base import LLMRequest
from backend.llm.mock import MockLLMProvider
from backend.memory.models import MemoryConfig, MemoryQuery, MemoryType, RetrievedMemory
from backend.memory.service import MemoryService
from backend.memory.sqlite_repository import SQLiteMemoryRepository
from backend.skills.executor import SkillExecutor
from backend.skills.library import create_default_skill_library
from backend.skills.models import SkillSelection
from backend.social.models import ConversationState
from backend.social.service import SocialAutonomyService
from backend.speech.service import SpeechService
from backend.tts.config import PROVIDER_MOCK, TTSConfig
from backend.tests.dispatch_support import confirm


BASE_TIME = datetime(2026, 9, 26, 9, 0, tzinfo=timezone.utc)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MOTIUS_SOURCE = (
    PROJECT_ROOT
    / "unity/3d_unity/Assets/animation/phase15_motius/lacrimosa_motius_smpl22_animated.fbx"
)
MOTIUS_REFERENCE = "Assets/animation/phase15_motius/lacrimosa_motius_smpl22_animated.fbx"
MOTIUS_DURATION = 1.0
MOTIUS_SAMPLE_RATE = 20.0
MOTIUS_CLIP = "MotiusPhase15Wave"
MOTIUS_SKILL_ID = "phase15_motius_wave"


class FakeClock:
    def __init__(self, value: datetime = BASE_TIME) -> None:
        self.value = value

    def __call__(self) -> datetime:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += timedelta(seconds=seconds)


def world_snapshot(
    *,
    streaming: bool = True,
    viewer_count: int = 1,
    duration: float = 0.0,
    current_action: str = "idle",
    camera_available: bool = True,
    camera_active: bool = True,
) -> dict[str, Any]:
    return {
        "character": {
            "position": [0, 0, 0],
            "rotation_y": 90,
            "posture": "standing",
            "current_action": current_action,
        },
        "phone": {"state": "Placed", "position": [1, 0.75, 0], "placement_surface": "TABLE"},
        "stream": {
            "is_streaming": streaming,
            "camera_available": camera_available,
            "camera_active": camera_active,
            "viewer_count": viewer_count,
            "duration_seconds": duration,
        },
        "targets": {
            "CENTER": {"position": [0, 0, 0], "navigation_target": True, "look_at_target": True},
            "TABLE": {"position": [1, 0.75, 0], "navigation_target": True, "look_at_target": True},
            "CAMERA": {"position": [2, 0.8, 0], "navigation_target": False, "look_at_target": True},
        },
    }


def world(**overrides: Any) -> WorldState:
    return build_world_state(world_snapshot(**overrides))


def chat_message(
    text: str,
    *,
    viewer_id: str = "viewer_01",
    message_id: str = "msg_001",
    timestamp: datetime = BASE_TIME,
) -> ChatMessage:
    normalized = normalize_text(text)
    message = ChatMessage(
        message_id=message_id,
        viewer_id=viewer_id,
        raw_text=text,
        normalized_text=normalized,
        timestamp=timestamp,
    )
    intent = extract_intent(normalized)
    message.intent = intent.intent
    message.target = intent.target
    message.priority = score_priority(message)
    return message


def goal(goal_type: GoalType = GoalType.RESPOND_TO_VIEWER) -> Goal:
    return Goal("goal_1", goal_type, 0.9, "test", status=GoalStatus.ACTIVE)


def success(action: ActionType = ActionType.WAVE) -> ActionResult:
    return ActionResult(action, ExecutionStatus.SUCCESS, "reported")


def failure(action: ActionType = ActionType.WAVE) -> ActionResult:
    return ActionResult(action, ExecutionStatus.REJECTED, "rejected_by_unity")


def episode(
    memory_id: str,
    *,
    score: float = 0.9,
    timestamp: datetime = BASE_TIME,
) -> RetrievedMemory:
    return RetrievedMemory(
        memory_type=MemoryType.EPISODE,
        memory_id=memory_id,
        summary=f"episode {memory_id}",
        score=score,
        importance=0.7,
        timestamp=timestamp,
        viewer_id="viewer_01",
    )


class CapturingProvider(MockBrainLLMProvider):
    """Deterministic mock that keeps the request it was asked to answer."""

    def __init__(self) -> None:
        self.requests: list[LLMRequest] = []

    def generate(self, request: LLMRequest) -> str:
        self.requests.append(request)
        return super().generate(request)


class BehaviorTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.temp_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_directory.cleanup)
        memory_path = Path(self.temp_directory.name) / "memory.db"
        self.memory = MemoryService(
            SQLiteMemoryRepository(memory_path),
            MemoryConfig(database_path=memory_path),
        )
        self.addCleanup(self.memory.close)
        self.library = create_default_skill_library()
        self.repository = SQLiteSkillLearningRepository(
            Path(self.temp_directory.name) / "learning.db"
        )
        self.addCleanup(self.repository.close)
        self.learning = SkillLearningService(
            self.library,
            LearningConfig(),
            self.repository,
            clock=self.clock,
        )

    def manager(self, config: BehaviorConfig | None = None) -> tuple[BehaviorManager, SocialAutonomyService]:
        social = SocialAutonomyService(clock=self.clock)
        return BehaviorManager(config=config, clock=self.clock), social

    def register_wave(self, *, cooldown: float = 0.0):
        _motion, skill = self.learning.register_motion(
            MOTIUS_SKILL_ID,
            "Generated one hand wave",
            "Motius generated motion: a person raises one hand and waves",
            MOTIUS_REFERENCE,
            MOTIUS_DURATION,
            tags=("wave",),
            source_path=MOTIUS_SOURCE,
            source_format="FBX",
            clip_name=MOTIUS_CLIP,
            sample_rate=MOTIUS_SAMPLE_RATE,
        )
        if cooldown:
            self.library.remove(skill.skill_id)
            skill = self.library.register(replace(skill, cooldown_seconds=cooldown))
        return skill

    def executor(self) -> SkillExecutor:
        return SkillExecutor(self.library, BrainPlanner(), self.clock)

    def brain(
        self,
        *,
        speech: SpeechService | None = None,
        behavior: BehaviorManager | None = None,
        provider: MockBrainLLMProvider | None = None,
    ):
        chat = ChatService(
            MockLLMProvider(),
            ChatServiceConfig(max_messages_per_rate_window=20),
            clock=self.clock,
        )
        social = SocialAutonomyService(clock=self.clock)
        instance = AutonomousBrain(
            provider or MockBrainLLMProvider(),
            chat,
            BrainConfig(decision_interval_seconds=0.0),
            personality=load_personality(),
            clock=self.clock,
            memory_service=self.memory,
            social_service=social,
            skill_library=self.library,
            learning_service=self.learning,
            speech_service=speech,
            behavior=behavior,
        )
        return instance, chat, social


class InitializationTests(BehaviorTestBase):
    def test_default_state_is_calm_and_inspectable(self) -> None:
        manager, _social = self.manager()
        self.assertEqual(manager.state, BehaviorState.CALM)
        self.assertIsNone(manager.previous_state)
        self.assertEqual(manager.transition_reason, "initialized")
        self.assertIsNone(manager.transitioned_at)
        context = manager.advisory_context(None)
        self.assertEqual(context["state"], "CALM")
        self.assertEqual(context["preferred_skills"], [])
        self.assertFalse(context["proactive_allowed"])
        self.assertIn("advisory only", context["rule"])

    def test_every_initial_state_is_available(self) -> None:
        self.assertEqual(
            {state.value for state in BehaviorState},
            {"CALM", "HAPPY", "CURIOUS", "FOCUSED", "SURPRISED", "TIRED", "ENGAGED"},
        )

    def test_config_rejects_unbounded_values(self) -> None:
        with self.assertRaises(ValueError):
            BehaviorConfig(minimum_dwell_seconds=-1.0)
        with self.assertRaises(ValueError):
            BehaviorConfig(memory_score_threshold=1.5)
        with self.assertRaises(ValueError):
            BehaviorConfig(max_recent_outcomes=0)

    def test_snapshot_round_trip_restores_continuity(self) -> None:
        manager, social = self.manager()
        manager.update(
            world(),
            None,
            social_context=social.context,
            chat=chat_message("hello lacrimosa"),
            now=self.clock(),
        )
        self.clock.advance(11)
        social.record_viewer_reaction("positive", "viewer_01", self.clock())
        manager.update(world(), None, social_context=social.context, now=self.clock())
        self.assertEqual(manager.state, BehaviorState.HAPPY)
        payload = manager.snapshot()
        self.assertEqual(payload["current_state"], "HAPPY")
        self.assertEqual(payload["previous_state"], "CURIOUS")
        self.assertEqual(payload["transition_reason"], "positive_viewer_reaction")
        self.assertIsNotNone(payload["transition_time"])

        restored = BehaviorManager(clock=self.clock)
        restored.restore(payload)
        self.assertEqual(restored.state, BehaviorState.HAPPY)
        self.assertEqual(restored.previous_state, BehaviorState.CURIOUS)
        self.assertEqual(restored.transition_reason, "positive_viewer_reaction")
        self.assertEqual(restored.transition_tick, 2)
        self.assertEqual(
            restored.to_dict()["transition_time"],
            manager.to_dict()["transition_time"],
        )

    def test_restore_rejects_unknown_state(self) -> None:
        manager = BehaviorManager(clock=self.clock)
        with self.assertRaises(ValueError):
            manager.restore({"current_state": "ECSTATIC"})
        manager.restore(None)
        self.assertEqual(manager.state, BehaviorState.CALM)


class TransitionTests(BehaviorTestBase):
    def test_greeting_transitions_to_curious_once(self) -> None:
        manager, social = self.manager()
        context = manager.update(
            world(),
            None,
            social_context=social.context,
            chat=chat_message("hello lacrimosa"),
            now=self.clock(),
        )
        self.assertEqual(context["state"], "CURIOUS")
        self.assertEqual(context["transition_reason"], "new_viewer_or_topic")
        self.assertEqual(manager.transition_tick, 1)

    def test_minimum_dwell_suppresses_rapid_flapping(self) -> None:
        manager, social = self.manager()
        manager.update(
            world(),
            None,
            social_context=social.context,
            chat=chat_message("hello lacrimosa"),
            now=self.clock(),
        )
        self.assertEqual(manager.state, BehaviorState.CURIOUS)
        transitions = 0
        for step in range(3):
            self.clock.advance(1)
            social.record_viewer_reaction("positive", "viewer_01", self.clock())
            before = manager.state
            manager.update(world(), None, social_context=social.context, now=self.clock())
            if manager.state != before:
                transitions += 1
        self.assertEqual(transitions, 0)
        self.assertEqual(manager.state, BehaviorState.CURIOUS)

        self.clock.advance(7)
        manager.update(world(), None, social_context=social.context, now=self.clock())
        self.assertEqual(manager.state, BehaviorState.HAPPY)
        self.assertEqual(manager.transition_reason, "positive_viewer_reaction")

    def test_positive_reaction_transitions_to_happy(self) -> None:
        manager, social = self.manager()
        social.record_viewer_reaction("positive", "viewer_01", self.clock())
        context = manager.update(world(), None, social_context=social.context, now=self.clock())
        self.assertEqual(context["state"], "HAPPY")
        self.assertEqual(context["transition_reason"], "positive_viewer_reaction")
        self.assertIn("positive", context["speech_style"])
        self.assertIn("wave", context["preferred_skills"])

    def test_negative_reaction_transitions_to_surprised(self) -> None:
        manager, social = self.manager()
        social.record_viewer_reaction("negative", "viewer_01", self.clock())
        context = manager.update(world(), None, social_context=social.context, now=self.clock())
        self.assertEqual(context["state"], "SURPRISED")
        self.assertEqual(context["transition_reason"], "negative_viewer_reaction")
        self.assertEqual(context["preferred_skills"], ["speak"])
        self.assertIn("short", context["speech_style"])

    def test_neutral_reaction_is_context_only(self) -> None:
        manager, social = self.manager()
        self.assertTrue(social.record_viewer_reaction("neutral", "viewer_01", self.clock()))
        context = manager.update(
            world(streaming=False, viewer_count=0),
            None,
            social_context=social.context,
            now=self.clock(),
        )
        self.assertEqual(context["state"], "CALM")
        self.assertFalse(context["proactive_allowed"])
        self.assertFalse(context["reactive_allowed"])

    def test_reaction_window_expiry_returns_to_baseline(self) -> None:
        manager, social = self.manager()
        self.clock.advance(11)
        social.record_viewer_reaction("positive", "viewer_01", self.clock())
        manager.update(world(), None, social_context=social.context, now=self.clock())
        self.assertEqual(manager.state, BehaviorState.HAPPY)

        self.clock.advance(31)
        manager.update(world(), None, social_context=social.context, now=self.clock())
        self.assertEqual(manager.state, BehaviorState.CALM)
        self.assertEqual(manager.transition_reason, "no_meaningful_behavior_event")
        self.assertFalse(manager.advisory_context(None)["proactive_allowed"])

    def test_new_viewer_count_transitions_to_curious(self) -> None:
        manager, social = self.manager()
        baseline = manager.update(
            world(streaming=True, viewer_count=0),
            None,
            social_context=social.context,
            now=self.clock(),
        )
        self.assertEqual(baseline["state"], "CALM")
        self.assertIsNone(manager.transitioned_at)

        self.clock.advance(11)
        context = manager.update(
            world(streaming=True, viewer_count=1),
            None,
            social_context=social.context,
            now=self.clock(),
        )
        self.assertEqual(context["state"], "CURIOUS")
        self.assertEqual(context["transition_reason"], "new_viewer_or_topic")
        self.assertEqual(manager.previous_state, BehaviorState.CALM)

        # A stable audience must not re-trigger curiosity on later updates.
        self.clock.advance(11)
        steady = manager.update(
            world(streaming=True, viewer_count=1),
            None,
            social_context=social.context,
            now=self.clock(),
        )
        self.assertEqual(steady["state"], "CALM")

    def test_active_task_transitions_to_focused(self) -> None:
        manager, _social = self.manager()
        context = manager.update(world(current_action="dance"), None, now=self.clock())
        self.assertEqual(context["state"], "FOCUSED")
        self.assertEqual(context["transition_reason"], "active_task_requires_attention")
        self.assertIn("minimal", context["speech_style"])
        self.assertEqual(context["preferred_skills"], [])

    def test_activity_goal_transitions_to_focused(self) -> None:
        manager, _social = self.manager()
        context = manager.update(
            world(streaming=False, viewer_count=0),
            goal(GoalType.PERFORM_ACTIVITY),
            now=self.clock(),
        )
        self.assertEqual(context["state"], "FOCUSED")

    def test_failed_outcome_transitions_to_surprised(self) -> None:
        manager, _social = self.manager()
        manager.record_outcome(failure(ActionType.DANCE), skill_id="dance", now=self.clock())
        context = manager.update(world(), None, now=self.clock())
        self.assertEqual(context["state"], "SURPRISED")
        self.assertEqual(context["transition_reason"], "recent_action_failure")
        diagnostics = manager.to_dict()
        self.assertEqual(diagnostics["recent_outcome"]["status"], "REJECTED")
        self.assertFalse(diagnostics["recent_outcome"]["success"])
        self.assertEqual(diagnostics["recent_outcome"]["skill_id"], "dance")

    def test_long_low_engagement_stream_transitions_to_tired(self) -> None:
        manager, social = self.manager()
        manager.update(
            world(streaming=True, viewer_count=1),
            None,
            social_context=social.context,
            now=self.clock(),
        )
        self.assertEqual(manager.state, BehaviorState.CURIOUS)
        self.clock.advance(11)
        social.context.social_engagement = 0.05
        context = manager.update(
            world(streaming=True, viewer_count=1, duration=7300.0),
            None,
            social_context=social.context,
            now=self.clock(),
        )
        self.assertEqual(context["state"], "TIRED")
        self.assertEqual(context["transition_reason"], "long_stream_with_low_engagement")
        self.assertIn("quiet", context["speech_style"])

    def test_sustained_engagement_transitions_to_engaged(self) -> None:
        manager, social = self.manager()
        manager.update(
            world(streaming=True, viewer_count=1),
            None,
            social_context=social.context,
            now=self.clock(),
        )
        self.clock.advance(11)
        social.context.social_engagement = 0.8
        social.context.last_activity_at = self.clock()
        context = manager.update(
            world(streaming=True, viewer_count=1),
            None,
            social_context=social.context,
            now=self.clock(),
        )
        self.assertEqual(context["state"], "ENGAGED")
        self.assertEqual(context["transition_reason"], "sustained_social_engagement")
        self.assertIn("speak", context["preferred_skills"])

    def test_relevant_memory_is_curious_once_per_distinct_memory(self) -> None:
        manager, _social = self.manager()
        first = manager.update(
            world(streaming=False, viewer_count=0),
            None,
            memory_context=[episode("ep_000001")],
            now=self.clock(),
        )
        self.assertEqual(first["state"], "CURIOUS")
        self.assertEqual(first["transition_reason"], "relevant_memory")

        self.clock.advance(11)
        baseline = manager.update(
            world(streaming=False, viewer_count=0),
            None,
            now=self.clock(),
        )
        self.assertEqual(baseline["state"], "CALM")

        self.clock.advance(11)
        repeated = manager.update(
            world(streaming=False, viewer_count=0),
            None,
            memory_context=[episode("ep_000001")],
            now=self.clock(),
        )
        self.assertEqual(repeated["state"], "CALM")

        self.clock.advance(11)
        distinct = manager.update(
            world(streaming=False, viewer_count=0),
            None,
            memory_context=[episode("ep_000002")],
            now=self.clock(),
        )
        self.assertEqual(distinct["state"], "CURIOUS")
        self.assertEqual(distinct["transition_reason"], "relevant_memory")

    def test_weak_memory_does_not_transition(self) -> None:
        manager, _social = self.manager()
        context = manager.update(
            world(streaming=False, viewer_count=0),
            None,
            memory_context=[episode("ep_000003", score=0.2)],
            now=self.clock(),
        )
        self.assertEqual(context["state"], "CALM")

    def test_retrieved_memory_from_memory_service_drives_curious(self) -> None:
        manager, _social = self.manager()
        self.memory.remember_chat(chat_message("hello lacrimosa"))
        self.memory.record_outcome(
            viewer_id="viewer_01",
            chat=chat_message("hello lacrimosa"),
            goal=None,
            action=ActionType.WAVE,
            success=True,
            response=None,
            context={"stream_active": False},
        )
        memories = self.memory.retrieve(
            MemoryQuery(viewer_id="viewer_01", text="hello lacrimosa")
        )
        self.assertTrue(memories)
        context = manager.update(
            world(streaming=False, viewer_count=0),
            None,
            memory_context=memories,
            now=self.clock(),
        )
        self.assertEqual(context["state"], "CURIOUS")
        self.assertEqual(context["transition_reason"], "relevant_memory")

    def test_no_meaningful_context_stays_calm_and_quiet(self) -> None:
        manager, social = self.manager()
        offline = world(streaming=False, viewer_count=0)
        for _ in range(4):
            context = manager.update(
                offline,
                None,
                social_context=social.context,
                now=self.clock(),
            )
            self.clock.advance(5)
        self.assertEqual(context["state"], "CALM")
        self.assertIsNone(manager.transitioned_at)
        self.assertEqual(manager.expression_count, 0)
        self.assertFalse(context["proactive_allowed"])
        self.assertFalse(context["attention_allowed"])


class ExpressionWindowTests(BehaviorTestBase):
    def test_proactive_expression_requires_live_audience(self) -> None:
        manager, social = self.manager()
        social.record_viewer_reaction("positive", "viewer_01", self.clock())
        offline = manager.update(
            world(streaming=False, viewer_count=0),
            None,
            social_context=social.context,
            now=self.clock(),
        )
        self.assertEqual(offline["state"], "HAPPY")
        self.assertFalse(offline["proactive_allowed"])
        self.assertIsNone(manager.last_proactive_at)

        online = manager.update(world(), None, social_context=social.context, now=self.clock())
        self.assertTrue(online["proactive_allowed"])
        self.assertEqual(online["proactive_reason"], "positive_viewer_reaction")
        self.assertEqual(manager.expression_count, 1)
        self.assertEqual(manager.expression_sequence, ["1:positive_viewer_reaction"])

    def test_proactive_cooldown_bounds_expression_frequency(self) -> None:
        manager, social = self.manager()
        social.record_viewer_reaction("positive", "viewer_01", self.clock())
        manager.update(world(), None, social_context=social.context, now=self.clock())
        self.assertEqual(manager.expression_count, 1)

        self.clock.advance(5)
        social.record_viewer_reaction("positive", "viewer_01", self.clock())
        throttled = manager.update(world(), None, social_context=social.context, now=self.clock())
        self.assertFalse(throttled["proactive_allowed"])
        self.assertEqual(manager.expression_count, 1)

        self.clock.advance(26)
        social.record_viewer_reaction("positive", "viewer_01", self.clock())
        allowed = manager.update(world(), None, social_context=social.context, now=self.clock())
        self.assertTrue(allowed["proactive_allowed"])
        self.assertEqual(manager.expression_count, 2)
        self.assertEqual(len(manager.expression_sequence), 2)

    def test_reactive_expression_follows_a_setback(self) -> None:
        manager, _social = self.manager()
        manager.record_outcome(failure(ActionType.DANCE), now=self.clock())
        context = manager.update(world(), None, now=self.clock())
        self.assertEqual(context["state"], "SURPRISED")
        self.assertTrue(context["reactive_allowed"])
        self.assertEqual(context["reactive_reason"], "recent_setback")

    def test_attention_window_requires_camera_and_cooldown(self) -> None:
        manager, social = self.manager()
        context = manager.update(
            world(),
            None,
            social_context=social.context,
            chat=chat_message("hello lacrimosa"),
            now=self.clock(),
        )
        self.assertEqual(context["state"], "CURIOUS")
        self.assertTrue(context["attention_allowed"])
        self.assertIsNotNone(manager.last_attention_at)

        self.clock.advance(5)
        throttled = manager.update(world(), None, social_context=social.context, now=self.clock())
        self.assertFalse(throttled["attention_allowed"])

        self.clock.advance(11)
        offline = manager.update(
            world(streaming=False, viewer_count=0),
            None,
            social_context=social.context,
            now=self.clock(),
        )
        self.assertFalse(offline["attention_allowed"])

    def test_attention_window_needs_a_lookable_camera(self) -> None:
        manager, social = self.manager()
        snapshot = world_snapshot()
        snapshot["targets"]["CAMERA"]["look_at_target"] = False
        manager.update(
            build_world_state(snapshot),
            None,
            social_context=social.context,
            chat=chat_message("hello lacrimosa"),
            now=self.clock(),
        )
        context = manager.update(
            build_world_state(snapshot),
            None,
            social_context=social.context,
            now=self.clock(),
        )
        self.assertEqual(context["state"], "CURIOUS")
        self.assertFalse(context["attention_allowed"])


class SkillSelectionTests(BehaviorTestBase):
    def behavior_context(self, **overrides: Any) -> dict[str, Any]:
        context = {
            "state": "HAPPY",
            "previous_state": "CURIOUS",
            "transition_reason": "positive_viewer_reaction",
            "speech_style": "positive and energetic, but concise",
            "preferred_skills": ["look_at", "wave", "speak"],
            "proactive_allowed": True,
            "reactive_allowed": False,
            "attention_allowed": False,
        }
        context.update(overrides)
        return context

    def ranking(self, skill_id: str = MOTIUS_SKILL_ID, score: float = 0.72) -> list[dict[str, Any]]:
        return [
            {
                "skill_id": skill_id,
                "description": "Motius generated motion: a person raises one hand and waves",
                "category": "SOCIAL",
                "learned": True,
                "advisory_score": score,
                "confidence": 0.35,
            },
            {
                "skill_id": "dance",
                "description": "Play the registered dance motion.",
                "category": "SOCIAL",
                "learned": False,
                "advisory_score": 0.5,
                "confidence": 0.5,
            },
        ]

    def test_happy_proactive_selects_learned_wave_through_executor(self) -> None:
        self.register_wave()
        executor = self.executor()
        provider = CapturingProvider()
        reasoner = BrainReasoner(provider, BrainPlanner(), executor)
        plan, reason = reasoner.plan(
            world(),
            goal(GoalType.INTERACT_WITH_VIEWER),
            load_personality(),
            "hello lacrimosa",
            None,
            self.ranking(),
            self.behavior_context(),
        )
        self.assertEqual(reason, "OK")
        self.assertIsNotNone(plan)
        self.assertEqual(plan.steps[0].action, ActionType.DANCE)
        self.assertEqual(plan.steps[0].target, MOTIUS_REFERENCE)
        self.assertEqual(executor.active.skill_id, MOTIUS_SKILL_ID)
        request = provider.requests[0]
        self.assertEqual(request.context["behavior"]["state"], "HAPPY")
        self.assertIn(MOTIUS_SKILL_ID, request.context["behavior_skill_ids"])

    def test_highest_ranked_wave_wins_between_learned_motions(self) -> None:
        self.register_wave()
        _motion, other = self.learning.register_motion(
            "phase16_walking_wave",
            "Imported walking wave",
            "Imported motion that reads as a hello wave",
            MOTIUS_REFERENCE,
            1.0,
            tags=("wave",),
            source_path=MOTIUS_SOURCE,
            source_format="FBX",
            clip_name=MOTIUS_CLIP,
            sample_rate=MOTIUS_SAMPLE_RATE,
        )
        executor = self.executor()
        reasoner = BrainReasoner(MockBrainLLMProvider(), BrainPlanner(), executor)
        ranking = [
            {
                "skill_id": other.skill_id,
                "description": other.description,
                "category": "SOCIAL",
                "learned": True,
                "advisory_score": 0.9,
            },
            *self.ranking(score=0.3),
        ]
        plan, reason = reasoner.plan(
            world(),
            goal(GoalType.INTERACT_WITH_VIEWER),
            load_personality(),
            "hello lacrimosa",
            None,
            ranking,
            self.behavior_context(),
        )
        self.assertEqual(reason, "OK")
        self.assertEqual(executor.active.skill_id, other.skill_id)
        self.assertNotEqual(executor.active.skill_id, MOTIUS_SKILL_ID)

    def test_cooldown_excludes_learned_wave_and_falls_back(self) -> None:
        self.register_wave(cooldown=60.0)
        executor = self.executor()
        reasoner = BrainReasoner(MockBrainLLMProvider(), BrainPlanner(), executor)
        first, first_reason = reasoner.plan(
            world(),
            goal(GoalType.INTERACT_WITH_VIEWER),
            load_personality(),
            "hello lacrimosa",
            None,
            self.ranking(),
            self.behavior_context(),
        )
        self.assertEqual(first_reason, "OK")
        self.assertEqual(executor.active.skill_id, MOTIUS_SKILL_ID)
        executor.observe_success(success(ActionType.DANCE))

        forced = executor.create_plan(
            SkillSelection(MOTIUS_SKILL_ID, {}, "behavior preference"),
            goal(GoalType.INTERACT_WITH_VIEWER),
            world(),
        )
        self.assertIsNone(forced.plan)
        self.assertEqual(forced.reason, "SKILL_ON_COOLDOWN")

        second, second_reason = reasoner.plan(
            world(),
            goal(GoalType.INTERACT_WITH_VIEWER),
            load_personality(),
            "hello lacrimosa",
            None,
            self.ranking(),
            self.behavior_context(),
        )
        self.assertEqual(second_reason, "OK")
        self.assertEqual([step.action for step in second.steps], [ActionType.WAVE])
        self.assertIsNone(executor.active)

    def test_conditions_still_block_unavailable_learned_skill(self) -> None:
        skill = self.register_wave()
        executor = self.executor()
        self.library.disable(skill.skill_id)
        provider = CapturingProvider()
        reasoner = BrainReasoner(provider, BrainPlanner(), executor)
        plan, reason = reasoner.plan(
            world(),
            goal(GoalType.INTERACT_WITH_VIEWER),
            load_personality(),
            "hello lacrimosa",
            None,
            self.ranking(),
            self.behavior_context(),
        )
        self.assertEqual(reason, "OK")
        self.assertNotIn(
            MOTIUS_SKILL_ID, provider.requests[0].context["behavior_skill_ids"]
        )
        self.assertEqual([step.action for step in plan.steps], [ActionType.WAVE])

    def test_surprised_reactive_state_uses_short_reaction(self) -> None:
        executor = self.executor()
        reasoner = BrainReasoner(MockBrainLLMProvider(), BrainPlanner(), executor)
        plan, reason = reasoner.plan(
            world(),
            goal(GoalType.INTERACT_WITH_VIEWER),
            load_personality(),
            "hello lacrimosa",
            None,
            [],
            self.behavior_context(
                state="SURPRISED",
                transition_reason="recent_action_failure",
                speech_style="short reactive wording",
                preferred_skills=["speak"],
                proactive_allowed=False,
                reactive_allowed=True,
            ),
        )
        self.assertEqual(reason, "OK")
        self.assertEqual([step.action for step in plan.steps], [ActionType.SPEAK])
        self.assertEqual(plan.steps[0].text, "Oh, that caught me off guard.")

    def test_curious_attention_looks_at_the_camera(self) -> None:
        executor = self.executor()
        reasoner = BrainReasoner(MockBrainLLMProvider(), BrainPlanner(), executor)
        plan, reason = reasoner.plan(
            world(),
            goal(GoalType.INTERACT_WITH_VIEWER),
            load_personality(),
            "hello lacrimosa",
            None,
            [],
            self.behavior_context(
                state="CURIOUS",
                transition_reason="new_viewer_or_topic",
                speech_style="interested and attentive",
                preferred_skills=["look_at", "speak"],
                proactive_allowed=False,
                attention_allowed=True,
            ),
        )
        self.assertEqual(reason, "OK")
        self.assertEqual([step.action for step in plan.steps], [ActionType.LOOK_AT])
        self.assertEqual(plan.steps[0].target, "CAMERA")

    def test_focused_state_never_forces_an_expression(self) -> None:
        executor = self.executor()
        reasoner = BrainReasoner(MockBrainLLMProvider(), BrainPlanner(), executor)
        plan, reason = reasoner.plan(
            world(),
            goal(GoalType.INTERACT_WITH_VIEWER),
            load_personality(),
            "hello lacrimosa",
            None,
            self.ranking(),
            self.behavior_context(
                state="FOCUSED",
                transition_reason="active_task_requires_attention",
                speech_style="minimal unnecessary speech; continue the task",
                preferred_skills=[],
                proactive_allowed=False,
            ),
        )
        self.assertEqual(reason, "OK")
        self.assertEqual([step.action for step in plan.steps], [ActionType.STOP])
        self.assertIsNone(executor.active)


class PlannerPromptTests(BehaviorTestBase):
    def test_prompt_contains_bounded_behavior_block(self) -> None:
        planner = BrainPlanner()
        prompt = planner.build_prompt(
            world(),
            goal(GoalType.INTERACT_WITH_VIEWER),
            load_personality(),
            "hello lacrimosa",
            None,
            None,
            None,
            {
                "state": "HAPPY",
                "previous_state": "CURIOUS",
                "transition_reason": "positive_viewer_reaction",
                "speech_style": "positive and energetic, but concise",
                "preferred_skills": ["look_at", "wave", "speak"],
                "proactive_allowed": True,
                "reactive_allowed": False,
                "attention_allowed": False,
                "rule": "advisory only",
            },
        )
        payload = json.loads(prompt)
        self.assertEqual(payload["behavior"]["state"], "HAPPY")
        self.assertEqual(payload["behavior"]["transition_reason"], "positive_viewer_reaction")
        self.assertTrue(payload["behavior"]["proactive_allowed"])
        self.assertEqual(payload["behavior"]["preferred_skills"], ["look_at", "wave", "speak"])
        self.assertEqual(payload["behavior"]["rule"], "advisory preference only; never bypasses skill validation")
        constraints = " ".join(payload["constraints"])
        self.assertIn("never bypasses skill conditions, cooldowns, allowlists, or SkillExecutor", constraints)
        self.assertIn("speech_style", constraints)

    def test_prompt_without_behavior_keeps_previous_shape(self) -> None:
        planner = BrainPlanner()
        payload = json.loads(
            planner.build_prompt(
                world(),
                goal(GoalType.RESPOND_TO_VIEWER),
                load_personality(),
                "hello lacrimosa",
            )
        )
        self.assertIsNone(payload["behavior"])
        self.assertEqual(payload["goal"]["goal_type"], "RESPOND_TO_VIEWER")


class BrainIntegrationTests(BehaviorTestBase):
    def greeting_world(self):
        return world(streaming=True, viewer_count=1)

    def test_curious_greeting_runs_look_at_wave_then_speaks_through_tts(self) -> None:
        speech = SpeechService(tts_config=TTSConfig(provider=PROVIDER_MOCK), start_worker=False)
        self.addCleanup(speech.shutdown)
        brain, chat, _social = self.brain(speech=speech)
        chat.submit("viewer_01", "hello lacrimosa")

        first = brain.tick(self.greeting_world())
        self.assertEqual(first.status, BrainStatus.EXECUTING)
        self.assertEqual(first.action.action, ActionType.LOOK_AT)
        self.assertEqual(first.action.target, "CAMERA")
        self.assertEqual(brain.behavior.state, BehaviorState.CURIOUS)
        self.assertEqual(brain.behavior.transition_reason, "new_viewer_or_topic")
        self.assertTrue(brain.last_behavior_context["attention_allowed"])

        # PHASE 26B: the greeting expression policy inserts the already
        # registered wave between orientation and speech so the gesture,
        # behavior state, and voice belong to one coordinated moment.
        second = brain.tick(self.greeting_world(), confirm(brain, success(ActionType.LOOK_AT)))
        self.assertEqual(second.action.action, ActionType.WAVE)
        self.assertEqual(brain.state.active_gesture, "wave")

        third = brain.tick(self.greeting_world(), confirm(brain, success(ActionType.WAVE)))
        self.assertEqual(third.action.action, ActionType.SPEAK)
        self.assertIsNotNone(third.action.speech_request_id)
        request = speech.get(third.action.speech_request_id)
        self.assertIsNotNone(request)
        self.assertEqual(request.source, "brain")
        # The gesture stays attributable to the exchange it belongs to while
        # her voice is still the active expression.
        self.assertEqual(brain.state.active_gesture, "wave")

        done = brain.tick(self.greeting_world(), confirm(brain, success(ActionType.SPEAK)))
        self.assertEqual(done.status, BrainStatus.IDLE)
        self.assertIsNone(brain.state.active_gesture)

    def test_positive_reaction_selects_learned_motion_and_records_learning(self) -> None:
        self.register_wave()
        brain, chat, social = self.brain()
        chat.submit("viewer_01", "hello lacrimosa")

        first = brain.tick(world(streaming=True, viewer_count=1))
        self.assertEqual(first.action.action, ActionType.LOOK_AT)
        brain.tick(world(streaming=True, viewer_count=1), confirm(brain, success(ActionType.LOOK_AT)))
        self.assertEqual(brain.behavior.state, BehaviorState.CURIOUS)

        self.clock.advance(11)
        self.assertTrue(
            social.record_viewer_reaction("positive", "viewer_01", self.clock())
        )
        chat.submit("viewer_01", "what are you doing?")
        decision = brain.tick(world(streaming=True, viewer_count=1), None)

        self.assertEqual(brain.behavior.state, BehaviorState.HAPPY)
        self.assertEqual(brain.behavior.transition_reason, "positive_viewer_reaction")
        self.assertTrue(brain.last_behavior_context["proactive_allowed"])
        self.assertEqual(decision.status, BrainStatus.EXECUTING)
        self.assertEqual(decision.action.action, ActionType.DANCE)
        self.assertEqual(decision.action.target, MOTIUS_REFERENCE)
        self.assertEqual(brain.current_skill_id(), MOTIUS_SKILL_ID)

        outcome = brain.tick(
            world(streaming=True, viewer_count=1),
            confirm(brain, success(ActionType.DANCE)),
        )
        self.assertEqual(outcome.status, BrainStatus.IDLE)
        self.assertIsNone(brain.current_skill_id())
        knowledge = self.learning.get_skill_state(MOTIUS_SKILL_ID)["contexts"]
        self.assertEqual(len(knowledge), 1)
        self.assertEqual(knowledge[0]["trials"], 1)
        self.assertEqual(knowledge[0]["successes"], 1)
        behavior_state = brain.behavior.to_dict()
        self.assertEqual(behavior_state["recent_outcome"]["skill_id"], MOTIUS_SKILL_ID)
        self.assertTrue(behavior_state["recent_outcome"]["success"])

    def test_cooldown_prevents_behavior_from_bypassing_skill_gates(self) -> None:
        self.register_wave(cooldown=120.0)
        brain, chat, social = self.brain()
        chat.submit("viewer_01", "hello lacrimosa")
        brain.tick(world(streaming=True, viewer_count=1))
        brain.tick(world(streaming=True, viewer_count=1), confirm(brain, success(ActionType.LOOK_AT)))

        self.clock.advance(11)
        social.record_viewer_reaction("positive", "viewer_01", self.clock())
        chat.submit("viewer_01", "what are you doing?")
        first = brain.tick(world(streaming=True, viewer_count=1))
        self.assertEqual(first.action.action, ActionType.DANCE)
        brain.tick(world(streaming=True, viewer_count=1), confirm(brain, success(ActionType.DANCE)))

        self.clock.advance(31)
        social.record_viewer_reaction("positive", "viewer_01", self.clock())
        chat.submit("viewer_01", "how are you today?")
        second = brain.tick(world(streaming=True, viewer_count=1))

        self.assertEqual(brain.behavior.state, BehaviorState.HAPPY)
        self.assertEqual(second.action.action, ActionType.WAVE)
        self.assertIsNone(brain.current_skill_id())
        validation = brain.skill_executor.availability(MOTIUS_SKILL_ID, world())
        self.assertFalse(validation.valid)
        self.assertEqual(validation.reason, "SKILL_ON_COOLDOWN")

    def test_rejected_skill_cancels_and_transitions_to_surprised(self) -> None:
        self.register_wave()
        brain, chat, social = self.brain()
        chat.submit("viewer_01", "hello lacrimosa")
        brain.tick(world(streaming=True, viewer_count=1))
        brain.tick(world(streaming=True, viewer_count=1), confirm(brain, success(ActionType.LOOK_AT)))

        self.clock.advance(11)
        social.record_viewer_reaction("positive", "viewer_01", self.clock())
        chat.submit("viewer_01", "what are you doing?")
        started = brain.tick(world(streaming=True, viewer_count=1))
        self.assertEqual(started.action.action, ActionType.DANCE)
        cancelled_goal = brain.state.current_goal
        self.assertEqual(cancelled_goal.status, GoalStatus.ACTIVE)

        brain.tick(world(streaming=True, viewer_count=1), confirm(brain, failure(ActionType.DANCE)))
        self.assertEqual(brain.behavior.state, BehaviorState.SURPRISED)
        self.assertEqual(brain.behavior.transition_reason, "recent_action_failure")
        self.assertIsNone(brain.skill_executor.active)
        # Rejection still cancels the failed goal before the loop replans.
        self.assertEqual(cancelled_goal.status, GoalStatus.CANCELLED)
        self.assertNotEqual(brain.state.current_plan, started.plan)
        self.assertIsNotNone(brain.state.pending_action)
        evaluations = brain.skill_executor.evaluations
        self.assertEqual(evaluations[-1].skill_id, MOTIUS_SKILL_ID)
        self.assertEqual(evaluations[-1].status.value, "FAILED")
        knowledge = self.learning.get_skill_state(MOTIUS_SKILL_ID)["contexts"]
        self.assertEqual(knowledge[0]["failures"], 1)

    def test_behavior_updates_once_per_real_decision_only(self) -> None:
        provider = CapturingProvider()
        brain, chat, _social = self.brain(provider=provider)
        chat.submit("viewer_01", "hello lacrimosa")
        brain.tick(world(streaming=True, viewer_count=1))
        ticks_after_decision = brain.behavior.logical_tick
        self.assertEqual(ticks_after_decision, 1)

        # Reporting the awaited result must not count as another behavior tick.
        brain.tick(world(streaming=True, viewer_count=1), confirm(brain, success(ActionType.LOOK_AT)))
        self.assertEqual(brain.behavior.logical_tick, 1)

        # The decision cooldown path also never re-derives behavior.
        brain.config = BrainConfig(decision_interval_seconds=60.0)
        waiting = brain.tick(world(streaming=True, viewer_count=1))
        self.assertEqual(waiting.status, BrainStatus.WAITING)
        self.assertEqual(brain.behavior.logical_tick, 1)

    def test_idle_world_produces_no_proactive_behavior(self) -> None:
        brain, _chat, _social = self.brain()
        for _ in range(3):
            brain.tick(world(streaming=False, viewer_count=0))
            self.clock.advance(5)
        self.assertEqual(brain.behavior.state, BehaviorState.CALM)
        self.assertEqual(brain.behavior.expression_count, 0)
        self.assertFalse(brain.last_behavior_context["proactive_allowed"])


class ServiceDiagnosticsTests(BehaviorTestBase):
    def test_brain_service_reports_behavior_diagnostics(self) -> None:
        from backend.api.brain import create_default_service

        service = create_default_service(
            memory_service=self.memory,
            database_path=Path(self.temp_directory.name) / "api-learning.db",
        )
        self.addCleanup(service.speech_service.shutdown)
        self.addCleanup(service.brain.learning.repository.close)
        service.brain.config = BrainConfig(decision_interval_seconds=0.0)
        service.tick(world_snapshot(), None)
        state = service.get_state()
        behavior = state["behavior"]
        self.assertEqual(behavior["current_state"], "CURIOUS")
        self.assertEqual(behavior["transition_reason"], "new_viewer_or_topic")
        self.assertEqual(behavior["previous_state"], "CALM")
        self.assertIsNone(behavior["current_skill"])
        self.assertIn("recent_outcomes", behavior)
        self.assertEqual(behavior["config"]["proactive_cooldown_seconds"], 30.0)
        self.assertEqual(service.get_behavior_state()["current_state"], "CURIOUS")


if __name__ == "__main__":
    unittest.main()
