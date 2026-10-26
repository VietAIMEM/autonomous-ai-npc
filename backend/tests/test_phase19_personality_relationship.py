"""PHASE 19 focused tests: stable personality and bounded viewer relationships.

Every assertion here works against the existing production path: the Phase 07
memory facade, the Phase 08 social cooldown, the Phase 10 learning ranking, the
Phase 11 speech service and the Phase 06/18 Brain loop. No parallel system is
introduced by the tests.
"""

from __future__ import annotations

import dataclasses
import json
import threading
import unittest
import urllib.error
import urllib.request
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any

from backend.api.brain import create_brain_server, create_default_service
from backend.brain.loop import AutonomousBrain, BrainConfig, BrainStatus
from backend.brain.models import (
    PERSONALITY_TRAITS,
    ActionResult,
    ActionType,
    ExecutionStatus,
    PersonalityProfile,
)
from backend.brain.personality import (
    DEFAULT_PERSONALITY,
    PERSONALITY_METADATA_KEY,
    load_personality,
    personality_descriptor,
    personality_from_payload,
    personality_speech_style,
    personality_traits,
)
from backend.brain.personalization import (
    MAX_SKILL_PREFERENCE_BIAS,
    build_personalized_context,
    skill_preference_bias,
)
from backend.brain.reasoning import MockBrainLLMProvider
from backend.brain.world_state import build_world_state
from backend.chat.dialogue import CharacterContext, build_prompt
from backend.chat.intent import extract_intent
from backend.chat.models import ChatMessage
from backend.chat.normalization import normalize_text
from backend.chat.priority import score_priority
from backend.chat.service import ChatService, ChatServiceConfig
from backend.learning.context import build_context_key
from backend.learning.models import LearningConfig
from backend.learning.service import SkillLearningService
from backend.llm.base import LLMProvider, LLMRequest
from backend.memory.models import MemoryConfig, RelationshipLevel, ViewerMemory
from backend.memory.service import MemoryService
from backend.memory.sqlite_repository import SQLiteMemoryRepository
from backend.skills.library import create_default_skill_library
from backend.skills.models import (
    SkillCategory,
    SkillEvaluation,
    SkillStatus,
)
from backend.social.models import SocialConfig
from backend.social.service import SocialAutonomyService
from backend.speech.service import SpeechService
from backend.tts.config import PROVIDER_MOCK, TTSConfig
from backend.tests.dispatch_support import confirm


BASE_TIME = datetime(2026, 9, 27, 12, 0, tzinfo=timezone.utc)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DANCE_SOURCE = PROJECT_ROOT / "unity/3d_unity/Assets/animation/dance/dance.fbx"
DANCE_REFERENCE = "Assets/animation/dance"
DANCE_DURATION = 18.466667
DANCE_SAMPLE_RATE = 30.0
VIEWER_A = "phase19_viewer_01"
VIEWER_B = "phase19_viewer_02"


class FixedClock:
    def __init__(self) -> None:
        self.value = BASE_TIME

    def __call__(self) -> datetime:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += timedelta(seconds=seconds)


def message(
    text: str,
    viewer_id: str = VIEWER_A,
    message_id: str = "msg_001",
) -> ChatMessage:
    normalized = normalize_text(text)
    value = ChatMessage(message_id, viewer_id, text, normalized, BASE_TIME)
    intent = extract_intent(normalized)
    value.intent = intent.intent
    value.target = intent.target
    value.priority = score_priority(value)
    return value


def memory_service(root: str | Path, **overrides: Any) -> MemoryService:
    config = MemoryConfig(database_path=Path(root) / "memory.db", **overrides)
    return MemoryService(SQLiteMemoryRepository(config.database_path), config)


@contextmanager
def temporary_memory_service(**config_overrides: Any):
    with TemporaryDirectory() as root:
        service = memory_service(root, **config_overrides)
        try:
            yield service
        finally:
            service.close()


def world_snapshot(**overrides: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "character": {
            "position": [0, 0, 0],
            "rotation_y": 90,
            "posture": "standing",
            "current_action": "idle",
        },
        "phone": {"state": "World", "position": [1, 0.75, 0], "placement_surface": None},
        # PHASE 24: viewer chat is handled while the stream is live.
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
    return build_world_state(world_snapshot(**overrides))


def record_dance(
    service: MemoryService,
    viewer_id: str = VIEWER_A,
    *,
    message_id: str = "msg_dance",
    skill_id: str | None = "dance_02",
    success: bool = True,
) -> None:
    service.record_outcome(
        viewer_id=viewer_id,
        chat=message("Lacrimosa can you dance?", viewer_id, message_id),
        goal=None,
        action=ActionType.DANCE,
        success=success,
        skill_id=skill_id,
    )


class FailingMemoryRepository:
    """Every repository call fails; personalization must degrade, not raise."""

    def __getattr__(self, name: str):
        def raise_memory_error(*args: Any, **kwargs: Any):
            raise RuntimeError("database unavailable")

        return raise_memory_error


class CapturingChatLLM(LLMProvider):
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, request: LLMRequest) -> str:
        self.prompts.append(request.prompt)
        intent_value = str(request.context.get("intent"))
        target = request.context.get("target")
        requested = (
            target
            if intent_value in {"REQUEST_ACTION", "REQUEST_MOVEMENT"} and target
            else None
        )
        return json.dumps(
            {
                "speech": "Of course, I can do that.",
                "intent": intent_value,
                "requested_action": requested,
                "priority": float(request.context.get("priority", 0.0)),
            }
        )


class PersonalityProfileTests(unittest.TestCase):
    def test_default_profile_is_bounded_serializable_and_immutable(self) -> None:
        traits = personality_traits(DEFAULT_PERSONALITY)
        self.assertEqual(set(traits), set(PERSONALITY_TRAITS))
        for value in traits.values():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

        payload = json.loads(json.dumps(DEFAULT_PERSONALITY.to_dict()))
        self.assertEqual(payload["name"], "Lacrimosa")
        self.assertAlmostEqual(payload["traits"]["warmth"], round(DEFAULT_PERSONALITY.warmth, 4))

        with self.assertRaises(dataclasses.FrozenInstanceError):
            DEFAULT_PERSONALITY.warmth = 0.1  # type: ignore[misc]

    def test_overrides_are_clamped_and_unknown_keys_rejected(self) -> None:
        profile = load_personality(
            {
                "warmth": 5.0,
                "energy": -2.0,
                "playfulness": float("nan"),
                "formality": 0.75,
                "mood": 0.9,
            }
        )
        self.assertEqual(profile.warmth, 1.0)
        self.assertEqual(profile.energy, 0.0)
        self.assertAlmostEqual(profile.playfulness, DEFAULT_PERSONALITY.playfulness)
        self.assertAlmostEqual(profile.formality, 0.75)
        self.assertFalse(hasattr(profile, "mood"))
        self.assertIsInstance(profile, PersonalityProfile)

    def test_payload_round_trip_and_corruption_tolerance(self) -> None:
        original = load_personality({"warmth": 0.31, "playfulness": 0.92})
        restored = personality_from_payload({"traits": personality_traits(original)})
        self.assertIsNotNone(restored)
        self.assertAlmostEqual(restored.warmth, 0.31, places=3)
        self.assertAlmostEqual(restored.playfulness, 0.92, places=3)

        self.assertIsNone(personality_from_payload("not-a-mapping"))
        self.assertIsNone(personality_from_payload(None))
        self.assertIsNone(personality_from_payload({"traits": {"bogus": 1.0}}))

        partial = personality_from_payload({"traits": {"warmth": 0.4, "bogus": "loud"}})
        self.assertIsNotNone(partial)
        self.assertAlmostEqual(partial.warmth, 0.4, places=3)
        self.assertAlmostEqual(partial.playfulness, DEFAULT_PERSONALITY.playfulness)

    def test_speech_style_and_descriptor_reflect_traits(self) -> None:
        warm = load_personality(
            {"warmth": 0.9, "playfulness": 0.9, "formality": 0.0, "talkativeness": 0.6}
        )
        style = personality_speech_style(warm)
        self.assertIn("warm", style)
        self.assertIn("playful", style)

        formal = load_personality(
            {"warmth": 0.2, "playfulness": 0.1, "formality": 0.9, "talkativeness": 0.3}
        )
        formal_style = personality_speech_style(formal)
        self.assertIn("formal", formal_style)
        self.assertIn("short", formal_style)

        descriptor = personality_descriptor(DEFAULT_PERSONALITY)
        self.assertEqual(
            set(descriptor),
            {"warmth", "playfulness", "formality", "talkativeness", "energy"},
        )

    def test_personality_persists_through_existing_metadata_table(self) -> None:
        with TemporaryDirectory() as root:
            service = memory_service(root)
            try:
                configured = load_personality({"warmth": 0.42, "playfulness": 0.33})
                service.set_personality(configured)
                self.assertTrue(
                    service.repository.get_metadata(PERSONALITY_METADATA_KEY)
                )
                self.assertEqual(service.personality_diagnostics()["source"], "persisted")
            finally:
                service.close()

            reopened = memory_service(root)
            try:
                restored = reopened.load_personality_state()
                self.assertAlmostEqual(restored.warmth, 0.42, places=3)
                self.assertAlmostEqual(restored.playfulness, 0.33, places=3)
                self.assertEqual(reopened.personality_diagnostics()["source"], "persisted")
            finally:
                reopened.close()

    def test_corrupt_personality_payload_falls_back_to_configuration(self) -> None:
        with temporary_memory_service() as service:
            for corrupt in ("not-json", json.dumps({"traits": {"bogus": 3}}), json.dumps([])):
                service.repository.set_metadata(PERSONALITY_METADATA_KEY, corrupt)
                profile = service.load_personality_state()
                self.assertEqual(personality_traits(profile), personality_traits(DEFAULT_PERSONALITY))
            self.assertEqual(service.personality_diagnostics()["source"], "configuration")

    def test_personality_never_drifts_from_recorded_interactions(self) -> None:
        with temporary_memory_service() as service:
            configured = load_personality({"warmth": 0.71, "playfulness": 0.64, "energy": 0.58})
            service.set_personality(configured)
            for index in range(10):
                record_dance(
                    service,
                    VIEWER_A if index % 2 else VIEWER_B,
                    message_id=f"drift_{index}",
                    success=index % 3 != 0,
                )
            self.assertEqual(
                personality_traits(service.load_personality_state()),
                personality_traits(configured),
            )
            self.assertEqual(
                service.personality_diagnostics()["profile"],
                personality_traits(configured),
            )


class PersonalizedContextTests(unittest.TestCase):
    def test_missing_or_corrupt_state_degrades_to_generic(self) -> None:
        blank = build_personalized_context(VIEWER_A, None)
        self.assertFalse(blank.known)
        self.assertFalse(blank.returning)
        self.assertEqual(blank.relationship_level, "NEW")
        self.assertIsNone(blank.greeting)

        corrupt = build_personalized_context(
            VIEWER_A,
            {
                "interaction_count": "many",
                "relationship_level": 42,
                "preferred_topics": {"dance": 1},
                "preferred_skills": 7,
                "familiarity": "high",
            },
        )
        self.assertFalse(corrupt.known)
        self.assertEqual(corrupt.preferred_topics, ())
        self.assertEqual(corrupt.preferred_skills, ())

        empty = build_personalized_context(VIEWER_A, {"interaction_count": 0})
        self.assertFalse(empty.known)
        self.assertFalse(empty.returning)

        partial = build_personalized_context(
            VIEWER_A,
            {
                "interaction_count": 5,
                "relationship_level": "FAMILIAR",
                "familiarity": 0.2,
                "degraded": True,
            },
        )
        self.assertTrue(partial.known)
        self.assertTrue(partial.degraded)
        self.assertIn("partial viewer memory", partial.reason)

    def test_returning_context_is_bounded(self) -> None:
        state = {
            "interaction_count": 12,
            "relationship_level": "regular",
            "familiarity": 0.42,
            "engagement_score": "high",
            "preferred_topics": ["dance", "music", "coding", "extra", "more"],
            "preferred_skills": ["dance_02", "wave", "dance_01", "x"],
            "recent_topics": ["dance", "music"],
            "last_interaction": "2026-09-27T12:00:00Z",
        }
        context = build_personalized_context(
            VIEWER_A,
            state,
            message_text="can we dance again?",
            intent="REQUEST_ACTION",
            personality=DEFAULT_PERSONALITY,
        )
        self.assertTrue(context.known)
        self.assertTrue(context.returning)
        self.assertEqual(context.relationship_level, "REGULAR")
        self.assertEqual(len(context.preferred_topics), 3)
        self.assertEqual(len(context.preferred_skills), 3)
        self.assertEqual(context.topic_continuity, "dance")
        self.assertEqual(context.engagement_score, 0.0)
        self.assertEqual(context.last_interaction, "2026-09-27T12:00:00Z")
        self.assertIn("REGULAR", context.reason)
        self.assertEqual(context.speech_style, personality_speech_style(DEFAULT_PERSONALITY))

        block = context.to_prompt_block()
        self.assertIn("advisory personalization only", block["rule"])
        self.assertNotIn("recent_topics", block)
        self.assertNotIn("last_interaction", block)

    def test_greeting_requires_intent_and_cooldown(self) -> None:
        state = {"interaction_count": 4, "relationship_level": "FAMILIAR", "familiarity": 0.2}
        greeting = build_personalized_context(
            VIEWER_A, state, message_text="hello", intent="GREETING", greeting_allowed=True
        )
        self.assertEqual(greeting.greeting, "Hello again! Good to see you.")

        blocked = build_personalized_context(
            VIEWER_A, state, message_text="hello", intent="GREETING", greeting_allowed=False
        )
        self.assertIsNone(blocked.greeting)

        other_intent = build_personalized_context(
            VIEWER_A, state, message_text="can you dance?", intent="REQUEST_ACTION"
        )
        self.assertIsNone(other_intent.greeting)

        trusted = build_personalized_context(
            VIEWER_A,
            {**state, "relationship_level": "TRUSTED"},
            message_text="hello",
            intent="GREETING",
            greeting_allowed=True,
        )
        self.assertEqual(trusted.greeting, "Welcome back! It's good to see you again.")

        weak = build_personalized_context(
            VIEWER_A, {"interaction_count": 0}, message_text="hello", intent="GREETING"
        )
        self.assertIsNone(weak.greeting)

    def test_topic_continuity_is_conditional(self) -> None:
        state = {
            "interaction_count": 5,
            "relationship_level": "FAMILIAR",
            "familiarity": 0.2,
            "preferred_topics": ["dance"],
        }
        self.assertEqual(
            build_personalized_context(
                VIEWER_A, state, message_text="can we dance again?", intent="REQUEST_ACTION"
            ).topic_continuity,
            "dance",
        )
        self.assertIsNone(
            build_personalized_context(
                VIEWER_A, state, message_text="what is the weather?", intent="QUESTION"
            ).topic_continuity
        )
        self.assertIsNone(
            build_personalized_context(
                VIEWER_A, state, message_text="hello there", intent="GREETING"
            ).topic_continuity
        )
        self.assertIsNone(
            build_personalized_context(
                VIEWER_A,
                {"interaction_count": 1, "preferred_topics": []},
                message_text="dance?",
                intent="REQUEST_ACTION",
            ).topic_continuity
        )

    def test_skill_preference_bias_is_bounded_and_advisory(self) -> None:
        exact, exact_reason = skill_preference_bias("dance_02", preferred_skills=("dance_02",))
        self.assertAlmostEqual(exact, 0.10, places=3)
        self.assertIn("viewer preference", exact_reason or "")

        related, _ = skill_preference_bias("dance_02", preferred_skills=("dance",))
        self.assertAlmostEqual(related, 0.06, places=3)

        combined, combined_reason = skill_preference_bias(
            "dance_02",
            skill_category="MOVEMENT",
            preferred_skills=("dance_02",),
            personality=load_personality({"playfulness": 0.95, "energy": 0.9}),
        )
        self.assertLessEqual(combined, MAX_SKILL_PREFERENCE_BIAS)
        self.assertGreater(combined, exact)
        self.assertIn("playful", combined_reason or "")

        unrelated, unrelated_reason = skill_preference_bias(
            "wave",
            skill_category="SOCIAL",
            preferred_skills=("dance_02",),
            personality=load_personality({"warmth": 0.1, "talkativeness": 0.1}),
        )
        self.assertEqual(unrelated, 0.0)
        self.assertIsNone(unrelated_reason)
        self.assertEqual(skill_preference_bias(""), (0.0, None))


class ViewerRelationshipTests(unittest.TestCase):
    def test_viewer_memory_normalizes_corrupt_values(self) -> None:
        memory = ViewerMemory(
            viewer_id="v",
            interaction_count="3",
            successful_interactions=-2,
            relationship_score=float("nan"),
            relationship_level="legendary",
            preferred_topics=tuple(f"topic{index}" for index in range(10)),
            preferred_skills=("DANCE_02", "dance_02"),
            familiarity=7.5,
            engagement_score=-1.0,
        )
        self.assertEqual(memory.interaction_count, 0)
        self.assertEqual(memory.successful_interactions, 0)
        self.assertEqual(memory.relationship_score, 0.2)
        self.assertEqual(memory.relationship_level, RelationshipLevel.NEW.value)
        self.assertEqual(memory.familiarity, 1.0)
        self.assertEqual(memory.engagement_score, 0.0)
        self.assertEqual(len(memory.preferred_topics), 5)
        self.assertEqual(memory.preferred_skills, ("DANCE_02", "dance_02"))

    def test_relationship_creation_and_gradual_progression(self) -> None:
        with temporary_memory_service() as service:
            self.assertIsNone(service.get_viewer(VIEWER_A)["viewer"])

            observed: list[str] = []
            for index in range(20):
                record_dance(service, VIEWER_A, message_id=f"progress_{index}")
                payload = service.get_viewer(VIEWER_A)["viewer"]
                self.assertIsNotNone(payload)
                observed.append(payload["relationship_level"])

            self.assertEqual(observed[0], "NEW")
            self.assertEqual(observed[2], "FAMILIAR")
            self.assertEqual(observed[7], "REGULAR")
            self.assertEqual(observed[19], "TRUSTED")
            self.assertTrue(all(level == "NEW" for level in observed[:2]))
            self.assertTrue(all(level == "FAMILIAR" for level in observed[2:7]))
            self.assertTrue(all(level == "REGULAR" for level in observed[7:19]))
            self.assertNotIn("TRUSTED", observed[:19])

            payload = service.get_viewer(VIEWER_A)["viewer"]
            self.assertEqual(payload["interaction_count"], 20)
            self.assertEqual(payload["successful_interactions"], 20)
            self.assertGreaterEqual(payload["familiarity"], 0.75)
            self.assertGreaterEqual(payload["relationship_score"], 0.70)
            self.assertGreaterEqual(payload["engagement_score"], 0.75)

    def test_single_interaction_cannot_jump_multiple_levels(self) -> None:
        with temporary_memory_service() as service:
            service.repository.save_viewer_memory(
                ViewerMemory(
                    viewer_id=VIEWER_A,
                    interaction_count=19,
                    successful_interactions=17,
                    failed_interactions=2,
                    relationship_score=0.8,
                    familiarity=0.9,
                    engagement_score=0.9,
                    relationship_level="NEW",
                    preferred_topics=("dance",),
                )
            )
            record_dance(service, VIEWER_A, message_id="one_more")
            payload = service.get_viewer(VIEWER_A)["viewer"]
            self.assertEqual(payload["relationship_level"], "FAMILIAR")

            record_dance(service, VIEWER_A, message_id="one_more_again")
            payload = service.get_viewer(VIEWER_A)["viewer"]
            self.assertEqual(payload["relationship_level"], "REGULAR")

    def test_failures_do_not_promote_relationship(self) -> None:
        with temporary_memory_service() as service:
            record_dance(service, VIEWER_B, message_id="fail_0")
            for index in range(1, 5):
                record_dance(service, VIEWER_B, message_id=f"fail_{index}", success=False)
            payload = service.get_viewer(VIEWER_B)["viewer"]
            self.assertEqual(payload["relationship_level"], "NEW")
            self.assertEqual(payload["interaction_count"], 5)
            self.assertEqual(payload["successful_interactions"], 1)
            self.assertEqual(payload["failed_interactions"], 4)
            self.assertLess(payload["relationship_score"], 0.2)

    def test_preference_learning_requires_repeat_evidence(self) -> None:
        with temporary_memory_service() as service:
            record_dance(service, VIEWER_A, message_id="weak_0")
            first = service.get_viewer(VIEWER_A)["viewer"]
            self.assertEqual(first["interaction_count"], 1)
            self.assertEqual(first["preferred_topics"], [])
            self.assertEqual(first["preferred_skills"], [])

            record_dance(service, VIEWER_A, message_id="weak_1")
            second = service.get_viewer(VIEWER_A)["viewer"]
            self.assertIn("dance", second["preferred_topics"])
            self.assertIn("dance_02", second["preferred_skills"])
            self.assertEqual(second["successful_interactions"], 2)

    def test_preferred_skills_need_successful_evidence(self) -> None:
        with temporary_memory_service() as service:
            for index in range(2):
                record_dance(service, VIEWER_B, message_id=f"failed_{index}", success=False)
            payload = service.get_viewer(VIEWER_B)["viewer"]
            self.assertIn("dance", payload["preferred_topics"])
            self.assertNotIn("dance_02", payload["preferred_skills"])
            self.assertEqual(payload["successful_interactions"], 0)

    def test_create_default_service_writes_stable_personality(self) -> None:
        with temporary_memory_service() as service:
            create_default_service(memory_service=service)
            self.assertEqual(service.personality_diagnostics()["source"], "persisted")


class ViewerIsolationTests(unittest.TestCase):
    def test_viewers_keep_isolated_relationships_and_preferences(self) -> None:
        with temporary_memory_service() as service:
            for index in range(3):
                record_dance(service, VIEWER_A, message_id=f"a_{index}")
            service.record_outcome(
                viewer_id=VIEWER_B,
                chat=message("Lacrimosa can you wave?", VIEWER_B, "b_0"),
                goal=None,
                action=ActionType.WAVE,
                success=True,
                skill_id="wave",
            )

            first = service.get_viewer(VIEWER_A)["viewer"]
            second = service.get_viewer(VIEWER_B)["viewer"]
            self.assertEqual(first["relationship_level"], "FAMILIAR")
            self.assertEqual(second["relationship_level"], "NEW")
            self.assertEqual(first["interaction_count"], 3)
            self.assertEqual(second["interaction_count"], 1)
            self.assertGreater(first["familiarity"], second["familiarity"])
            self.assertIn("dance", first["preferred_topics"])
            self.assertNotIn("dance", second["preferred_topics"])
            self.assertNotIn("dance_02", second["preferred_skills"])
            self.assertFalse(set(first["preferred_topics"]) & set(second["preferred_topics"]))

            context_a = service.personalize(
                VIEWER_A, message_text="can we dance?", intent="REQUEST_ACTION"
            )
            context_b = service.personalize(
                VIEWER_B, message_text="can we dance?", intent="REQUEST_ACTION"
            )
            self.assertIn("dance", context_a.preferred_topics)
            self.assertNotIn("dance", context_b.preferred_topics)
            self.assertEqual(context_b.relationship_level, "NEW")
            self.assertIsNone(context_b.topic_continuity)

            unknown = service.personalize("phase19_viewer_99")
            self.assertFalse(unknown.known)
            self.assertEqual(unknown.preferred_topics, ())
            self.assertEqual(unknown.preferred_skills, ())

            self.assertNotIn("viewer_01", json.dumps(service.get_viewer(VIEWER_B)))


class PersistenceTests(unittest.TestCase):
    def test_relationship_preferences_and_personality_survive_restart(self) -> None:
        with TemporaryDirectory() as root:
            service = memory_service(root)
            try:
                service.set_personality(
                    load_personality({"warmth": 0.42, "playfulness": 0.33, "formality": 0.66})
                )
                for index in range(3):
                    record_dance(service, VIEWER_A, message_id=f"restart_{index}")
                before = service.get_viewer(VIEWER_A)["viewer"]
            finally:
                service.close()

            reopened = memory_service(root)
            try:
                after = reopened.get_viewer(VIEWER_A)["viewer"]
                self.assertEqual(after["interaction_count"], before["interaction_count"])
                self.assertEqual(after["relationship_level"], before["relationship_level"])
                self.assertEqual(after["preferred_topics"], before["preferred_topics"])
                self.assertEqual(after["preferred_skills"], before["preferred_skills"])
                self.assertAlmostEqual(after["familiarity"], before["familiarity"], places=3)
                self.assertEqual(after["last_interaction"], before["last_interaction"])

                personality = reopened.load_personality_state()
                self.assertAlmostEqual(personality.warmth, 0.42, places=3)
                self.assertAlmostEqual(personality.playfulness, 0.33, places=3)
                self.assertAlmostEqual(personality.formality, 0.66, places=3)
                self.assertEqual(reopened.personality_diagnostics()["source"], "persisted")

                self.assertIsNone(reopened.get_viewer(VIEWER_B)["viewer"])
            finally:
                reopened.close()


class BrainPersonalizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FixedClock()
        self.temp_directory = TemporaryDirectory()
        self.memory = memory_service(self.temp_directory.name)
        self.provider = CapturingChatLLM()
        self.chat = ChatService(
            self.provider,
            ChatServiceConfig(max_messages_per_rate_window=50),
            clock=self.clock,
        )

    def tearDown(self) -> None:
        self.memory.close()
        self.temp_directory.cleanup()

    def _brain(self, **overrides: Any) -> AutonomousBrain:
        return AutonomousBrain(
            MockBrainLLMProvider(),
            self.chat,
            BrainConfig(decision_interval_seconds=0.0),
            clock=self.clock,
            memory_service=self.memory,
            **overrides,
        )

    def _seed_returning_viewer(self, level: str = "FAMILIAR") -> None:
        self.memory.repository.save_viewer_memory(
            ViewerMemory(
                viewer_id=VIEWER_A,
                interaction_count=4,
                successful_interactions=3,
                failed_interactions=1,
                relationship_score=0.35,
                familiarity=0.2,
                engagement_score=0.6,
                relationship_level=level,
                preferred_topics=("dance",),
                preferred_skills=("dance_02",),
                recent_topics=("dance",),
            )
        )

    def test_returning_viewer_context_reaches_chat_prompt(self) -> None:
        self._seed_returning_viewer()
        brain = self._brain()
        self.chat.submit(VIEWER_A, "hello Lacrimosa")
        tick = brain.tick(world())
        self.assertIn(tick.status, {BrainStatus.EXECUTING, BrainStatus.WAITING})

        prompt = self.provider.prompts[-1]
        self.assertIn("Speech style:", prompt)
        self.assertIn("Viewer context (advisory only):", prompt)
        self.assertIn("returning viewer, relationship FAMILIAR", prompt)
        self.assertIn("dance", prompt)

        personalization = brain.last_personalization
        self.assertIsNotNone(personalization)
        self.assertTrue(personalization.known)
        self.assertTrue(personalization.returning)
        self.assertEqual(personalization.relationship_level, "FAMILIAR")
        self.assertIsNotNone(personalization.greeting)

    def test_new_viewer_gets_no_personalized_context(self) -> None:
        self._seed_returning_viewer()
        brain = self._brain()
        self.chat.submit(VIEWER_B, "hello Lacrimosa")
        brain.tick(world())

        prompt = self.provider.prompts[-1]
        self.assertIn("Speech style:", prompt)
        self.assertNotIn("Viewer context (advisory only):", prompt)
        self.assertNotIn("returning viewer", prompt)
        self.assertNotIn("dance_02", prompt)

        personalization = brain.last_personalization
        self.assertIsNotNone(personalization)
        self.assertFalse(personalization.known)
        self.assertIsNone(personalization.greeting)
        self.assertEqual(personalization.preferred_topics, ())

    def test_returning_greeting_respects_social_speech_cooldown(self) -> None:
        self._seed_returning_viewer()
        social = SocialAutonomyService(
            SocialConfig(speech_cooldown_seconds=12.0),
            clock=self.clock,
        )
        brain = self._brain(social_service=social)

        first = brain._personalized_context(message("hello Lacrimosa", VIEWER_A, "greet_1"))
        self.assertIsNotNone(first)
        self.assertIsNotNone(first.greeting)

        social.notify_processed(message("hello Lacrimosa", VIEWER_A, "greet_1"), "hi", None, self.clock())
        self.assertFalse(social.conversation.can_speak(self.clock(), social.context.last_speech_at))

        self.clock.advance(1.0)
        blocked = brain._personalized_context(message("good morning Lacrimosa", VIEWER_A, "greet_2"))
        self.assertIsNone(blocked.greeting)

        self.clock.advance(13.0)
        allowed = brain._personalized_context(message("hey Lacrimosa", VIEWER_A, "greet_3"))
        self.assertIsNotNone(allowed.greeting)

    def test_personalization_failure_keeps_brain_operational(self) -> None:
        broken = MemoryService(
            FailingMemoryRepository(),
            MemoryConfig(database_path=Path(self.temp_directory.name) / "broken.db"),
        )
        brain = AutonomousBrain(
            MockBrainLLMProvider(),
            self.chat,
            BrainConfig(decision_interval_seconds=0.0),
            clock=self.clock,
            memory_service=broken,
        )
        self.chat.submit(VIEWER_A, "hello Lacrimosa")
        tick = brain.tick(world())
        self.assertIn(tick.status, {BrainStatus.EXECUTING, BrainStatus.WAITING})
        self.assertIsNotNone(brain.last_personalization)
        self.assertFalse(brain.last_personalization.known)
        self.assertEqual(brain.last_personalization.preferred_topics, ())
        self.assertFalse(broken.load_personality_state().warmth != DEFAULT_PERSONALITY.warmth)
        self.assertFalse(broken.save_personality_state(DEFAULT_PERSONALITY))
        self.assertEqual(broken._update_viewer_memory.__name__, "_update_viewer_memory")

    def test_personalization_does_not_change_tts_provider_configuration(self) -> None:
        self._seed_returning_viewer()
        speech = SpeechService(
            tts_config=TTSConfig(provider=PROVIDER_MOCK, voice="lacrimosa_v1"),
            clock=self.clock,
            start_worker=False,
        )
        brain = self._brain(speech_service=speech)
        self.chat.submit(VIEWER_A, "hello Lacrimosa")
        brain.tick(world())

        self.assertEqual(speech.tts_config.voice, "lacrimosa_v1")
        self.assertEqual(speech.tts_config.provider, PROVIDER_MOCK)
        self.assertIn("Speech style:", self.provider.prompts[-1])
        voice_config = speech.tts_config.to_dict()
        self.assertNotIn("api_key", json.dumps(voice_config))
        self.assertEqual(voice_config["voice"], "lacrimosa_v1")

    def test_character_context_defaults_stay_backwards_compatible(self) -> None:
        character = CharacterContext()
        self.assertIsNone(character.speech_style)
        self.assertEqual(character.viewer_context, ())
        prompt = build_prompt(
            message("hello Lacrimosa"),
            self.chat.conversation,
            self.chat.world_context,
            character,
        )
        self.assertNotIn("Speech style:", prompt)
        self.assertNotIn("Viewer context", prompt)

    def test_plan_skill_evidence_survives_the_appended_dialogue_step(self) -> None:
        """A chat-driven dance plan ends on an appended SPEAK step.

        SkillExecutor clears its active skill after the DANCE step, so the final
        SPEAK step is what records the exchange in viewer memory. The captured
        plan skill must therefore reach the recorded evidence instead of None,
        otherwise viewer skill preferences can never accumulate.
        """
        library = create_default_skill_library()
        learning = SkillLearningService(library, LearningConfig(), clock=self.clock)
        learning.register_motion(
            "dance_02",
            "Dance 02",
            "Second dance motion",
            DANCE_REFERENCE,
            DANCE_DURATION,
            tags=("dance",),
            source_path=DANCE_SOURCE,
            source_format="FBX",
            clip_name="Dance",
            sample_rate=DANCE_SAMPLE_RATE,
            skill_category=SkillCategory.MOVEMENT,
        )
        speech = SpeechService(
            tts_config=TTSConfig(provider=PROVIDER_MOCK),
            clock=self.clock,
            start_worker=False,
        )
        self.addCleanup(speech.shutdown)
        brain = self._brain(
            skill_library=library,
            learning_service=learning,
            speech_service=speech,
        )
        stage_world = world(
            phone={"state": "Placed", "position": [1, 0.75, 0], "placement_surface": "Table"},
        )
        self.chat.submit(VIEWER_A, "Lacrimosa can you dance?")

        started = brain.tick(stage_world)
        self.assertEqual(started.status, BrainStatus.EXECUTING)
        self.assertEqual(started.action.action, ActionType.DANCE)
        self.assertEqual(brain.current_skill_id(), "dance_02")
        self.assertEqual(
            [step.action for step in started.plan.steps],
            [ActionType.DANCE, ActionType.SPEAK],
        )

        speaking = brain.tick(
            stage_world,
            confirm(brain, ActionResult(ActionType.DANCE, ExecutionStatus.SUCCESS)),
        )
        self.assertEqual(speaking.action.action, ActionType.SPEAK)
        self.assertIsNone(brain.current_skill_id())

        finished = brain.tick(
            stage_world,
            confirm(brain, ActionResult(ActionType.SPEAK, ExecutionStatus.SUCCESS)),
        )
        self.assertEqual(finished.status, BrainStatus.IDLE)
        self.assertEqual(finished.message, "plan_complete")

        episodes = self.memory.repository.search_episodes(viewer_id=VIEWER_A, limit=20)
        tags = {
            tag
            for episode in episodes
            if episode.result == "SUCCESS"
            for tag in episode.tags
        }
        self.assertIn("dance_02", tags)
        viewer = self.memory.get_viewer(VIEWER_A)
        self.assertEqual(viewer["viewer"]["interaction_count"], 1)


class SkillRankingPersonalizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FixedClock()
        self.library = create_default_skill_library()
        self.learning = SkillLearningService(self.library, LearningConfig(), clock=self.clock)
        self.learning.register_motion(
            "dance_02",
            "Dance 02",
            "Second dance motion",
            DANCE_REFERENCE,
            DANCE_DURATION,
            tags=("dance",),
            source_path=DANCE_SOURCE,
            source_format="FBX",
            clip_name="Dance",
            sample_rate=DANCE_SAMPLE_RATE,
            skill_category=SkillCategory.MOVEMENT,
        )
        self.context = build_context_key(intent="REQUEST_ACTION", target="dance")

    def _observe_success(self) -> None:
        self.learning.observe_evaluation(
            SkillEvaluation(
                skill_id="dance_02",
                execution_id="dance_02_execution",
                status=SkillStatus.SUCCESS,
                duration_seconds=1.0,
                steps_completed=1,
                steps_failed=0,
            ),
            context_key=self.context,
        )

    def test_viewer_preference_and_personality_bias_advisory_ranking(self) -> None:
        self._observe_success()
        baseline = next(
            item
            for item in self.learning.ranking_prompt_context(self.context)
            if item["skill_id"] == "dance_02"
        )

        personalized = self.learning.ranking_prompt_context(
            self.context,
            personalization=SimpleNamespace(preferred_skills=("dance_02",)),
            personality=load_personality({"playfulness": 0.95, "energy": 0.9}),
        )
        biased = next(item for item in personalized if item["skill_id"] == "dance_02")
        self.assertGreater(biased["advisory_score"], baseline["advisory_score"])
        self.assertGreater(biased["personalization_bias"], 0.0)
        self.assertLessEqual(biased["personalization_bias"], MAX_SKILL_PREFERENCE_BIAS)
        self.assertIn("viewer preference evidence", biased["personalization_reason"])
        self.assertIn("playful", biased["personalization_reason"])
        self.assertEqual(baseline["learned_success_rate"], biased["learned_success_rate"])
        self.assertEqual(
            self.learning.get_skill_state("dance_02")["contexts"][0]["trials"],
            1,
        )

    def test_preference_signal_never_fabricates_knowledge(self) -> None:
        entries = self.learning.ranking_prompt_context(
            self.context,
            personalization=SimpleNamespace(preferred_skills=("dance_02",)),
            personality=load_personality({"playfulness": 1.0}),
        )
        self.assertTrue(entries)
        self.assertEqual(self.learning.get_skill_state("dance_02")["contexts"], [])
        for item in entries:
            if item["skill_id"] != "dance_02":
                continue
            self.assertEqual(item["learned_success_rate"], None)
            self.assertLessEqual(item["advisory_score"], 1.0)

    def test_malformed_personalization_is_ignored(self) -> None:
        self._observe_success()
        baseline = next(
            item
            for item in self.learning.ranking_prompt_context(self.context)
            if item["skill_id"] == "dance_02"
        )
        malformed = next(
            item
            for item in self.learning.ranking_prompt_context(
                self.context,
                personalization=SimpleNamespace(preferred_skills={"dance_02": True}),
                personality=None,
            )
            if item["skill_id"] == "dance_02"
        )
        self.assertEqual(malformed["advisory_score"], baseline["advisory_score"])
        self.assertIsNone(malformed["personalization_bias"])


class PersonalizationApiTests(unittest.TestCase):
    def test_personality_and_viewer_diagnostics_routes(self) -> None:
        with TemporaryDirectory() as root:
            memory = memory_service(root)
            service = create_default_service(memory_service=memory)
            server = create_brain_server(service, "127.0.0.1", 0)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            host, port = server.server_address
            base = f"http://{host}:{port}"
            try:
                status, personality = self._get(f"{base}/personality/state")
                self.assertEqual(status, 200)
                self.assertEqual(set(personality["profile"]), set(PERSONALITY_TRAITS))
                self.assertEqual(personality["source"], "persisted")
                self.assertIn("rule", personality)
                serialized = json.dumps(personality).lower()
                for secret in ("token", "api_key", "password", "credential"):
                    self.assertNotIn(secret, serialized)

                status, unknown = self._get(f"{base}/memory/viewer/{VIEWER_B}")
                self.assertEqual(status, 200)
                self.assertIsNone(unknown["viewer"])
                self.assertFalse(unknown["personalization"]["known_viewer"])
                self.assertEqual(unknown["recent_interactions"], [])

                for index in range(3):
                    record_dance(memory, VIEWER_A, message_id=f"api_{index}")

                status, viewer = self._get(f"{base}/memory/viewer/{VIEWER_A}")
                self.assertEqual(status, 200)
                self.assertEqual(viewer["viewer"]["viewer_id"], VIEWER_A)
                self.assertEqual(viewer["viewer"]["relationship_level"], "FAMILIAR")
                personalization = viewer["personalization"]
                self.assertTrue(personalization["known_viewer"])
                self.assertIn("dance", personalization["preferred_topics"])
                self.assertIn("dance_02", personalization["preferred_skills"])
                self.assertIn("reason", personalization)
                self.assertIn("last_interaction", personalization)
                self.assertEqual(len(viewer["recent_interactions"]), 3)

                status, other = self._get(f"{base}/memory/viewer/{VIEWER_B}")
                self.assertEqual(status, 200)
                self.assertIsNone(other["viewer"])

                status, invalid = self._get(f"{base}/memory/viewer/{'x' * 130}")
                self.assertEqual(status, 400)
                self.assertEqual(invalid["error"], "invalid_viewer_id")
            finally:
                server.shutdown()
                thread.join(timeout=2)
                server.server_close()
                memory.close()

    def _get(self, url: str) -> tuple[int, dict[str, Any]]:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
