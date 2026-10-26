from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from backend.brain.loop import AutonomousBrain, BrainConfig, BrainStatus
from backend.brain.models import ActionResult, ActionType, ExecutionStatus, Goal, GoalType
from backend.brain.reasoning import BrainReasoner, MockBrainLLMProvider
from backend.brain.world_state import build_world_state
from backend.chat.models import ChatIntent, ChatMessage
from backend.chat.service import ChatService, ChatServiceConfig
from backend.learning.context import build_context_key, context_from_world
from backend.learning.models import LearningConfig
from backend.learning.outcomes import OutcomeEvaluator
from backend.learning.repository import SQLiteSkillLearningRepository
from backend.learning.service import SkillLearningService
from backend.llm.base import LLMProvider, LLMRequest
from backend.memory import MemoryConfig, MemoryService, SQLiteMemoryRepository
from backend.memory.models import ViewerMemory
from backend.skills.library import SkillRegistryError, create_default_skill_library
from backend.skills.models import Skill, SkillCategory, SkillCompositionStep, SkillStatus
from backend.tests.dispatch_support import confirm


BASE_TIME = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DANCE_SOURCE = PROJECT_ROOT / "unity/3d_unity/Assets/animation/dance/dance.fbx"
DANCE_REFERENCE = "Assets/animation/dance"
DANCE_DURATION = 18.466667
DANCE_SAMPLE_RATE = 30.0


class FixedClock:
    def __init__(self) -> None:
        self.value = BASE_TIME

    def __call__(self) -> datetime:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += timedelta(seconds=seconds)


def snapshot(**overrides: Any) -> dict[str, Any]:
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
    return build_world_state(snapshot(**overrides))


def register_dance(
    service: SkillLearningService,
    motion_id: str = "dance_02",
    name: str = "Dance 02",
    description: str = "Second dance motion",
) -> tuple[Any, Any]:
    return service.register_motion(
        motion_id,
        name,
        description,
        DANCE_REFERENCE,
        DANCE_DURATION,
        tags=("dance",),
        source_path=DANCE_SOURCE,
        source_format="FBX",
        clip_name="Dance",
        sample_rate=DANCE_SAMPLE_RATE,
    )


class LearningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FixedClock()
        self.temp_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_directory.name) / "learning.db"
        self.library = create_default_skill_library()
        self.repository = SQLiteSkillLearningRepository(self.database_path)
        self.service = SkillLearningService(
            self.library,
            LearningConfig(),
            self.repository,
            clock=self.clock,
            random_source=__import__("random").Random(10),
        )

    def _memory(self) -> MemoryService:
        database = Path(self.temp_directory.name) / "memory.db"
        service = MemoryService(
            SQLiteMemoryRepository(database),
            MemoryConfig(database_path=database),
        )
        self.memory_service = service
        return service

    def tearDown(self) -> None:
        if getattr(self, "memory_service", None):
            self.memory_service.close()
        self.repository.close()
        self.temp_directory.cleanup()

    def evaluation(
        self,
        skill_id: str,
        status: SkillStatus = SkillStatus.SUCCESS,
        viewer_response: str | None = None,
        duration: float = 1.0,
    ):
        from backend.skills.models import SkillEvaluation

        return SkillEvaluation(
            skill_id=skill_id,
            execution_id=f"{skill_id}_execution",
            status=status,
            duration_seconds=duration,
            steps_completed=1 if status == SkillStatus.SUCCESS else 0,
            steps_failed=0 if status == SkillStatus.SUCCESS else 1,
            viewer_response=viewer_response,
        )

    def test_motion_registration_and_conversion(self) -> None:
        motion, skill = self.service.register_motion(
            "dance_02",
            "Dance 02",
            "Plays an already imported dance motion.",
            DANCE_REFERENCE,
            DANCE_DURATION,
            tags=("dance", "entertainment"),
            source_path=DANCE_SOURCE,
            source_format="FBX",
            clip_name="Dance",
            sample_rate=DANCE_SAMPLE_RATE,
        )
        self.assertEqual(motion.motion_id, "dance_02")
        self.assertEqual(skill.skill_id, "dance_02")
        self.assertTrue(self.library.exists("dance_02"))
        self.assertIn("learned", skill.tags)
        self.assertEqual(skill.animation_reference, "Assets/animation/dance")

        with self.assertRaises(ValueError):
            register_dance(self.service, "dance_02", "Duplicate", "Duplicate motion")

        with self.assertRaises(ValueError):
            register_dance(self.service, "Bad ID", "Invalid", "Invalid motion id")

    def test_context_keys_are_deterministic_and_isolated(self) -> None:
        first = build_context_key(
            intent="REQUEST_ACTION",
            target="dance",
            goal="RESPOND_TO_VIEWER",
            character_state="standing",
            stream_state="ACTIVE",
        )
        second = build_context_key(
            intent=" request_action ",
            target="Dance",
            goal="respond_to_viewer",
            character_state=" Standing ",
            stream_state="active",
        )
        self.assertEqual(first, second)
        other = build_context_key(intent="REQUEST_ACTION", target="wave")
        self.assertNotEqual(first, other)
        self.assertLessEqual(len(first), 720)

    def test_world_context_matches_persisted_learning_key(self) -> None:
        chat = ChatMessage(
            message_id="message",
            viewer_id="viewer_01",
            raw_text="Lacrimosa, dance!",
            normalized_text="lacrimosa, dance!",
            timestamp=BASE_TIME,
            intent=ChatIntent.REQUEST_ACTION,
            target="dance",
        )
        goal = Goal("goal", GoalType.RESPOND_TO_VIEWER, 0.9, "dance request")
        viewer = ViewerMemory("viewer_01", relationship_score=0.2)

        context = context_from_world(
            goal,
            chat,
            world(
                stream={"is_streaming": True, "camera_available": True, "camera_active": True},
                phone={"state": "Placed", "position": [1, 0.75, 0], "placement_surface": "Table"},
            ),
            viewer,
        )

        self.assertEqual(
            context,
            "REQUEST_ACTION|DANCE|RESPOND_TO_VIEWER|STANDING|ACTIVE|TABLE|0_0",
        )

    def test_outcome_scores_are_bounded_and_interpretable(self) -> None:
        success = OutcomeEvaluator().evaluate(
            self.evaluation("dance", SkillStatus.SUCCESS, "positive", 2.0),
            "context",
            BASE_TIME,
        )
        failure = OutcomeEvaluator().evaluate(
            self.evaluation("dance", SkillStatus.FAILED, "negative", 1.0),
            "context",
            BASE_TIME,
        )
        interrupted = OutcomeEvaluator().evaluate(
            self.evaluation("dance", SkillStatus.INTERRUPTED, None, 1.0),
            "context",
            BASE_TIME,
        )
        self.assertEqual(success.outcome_score, 0.85)
        self.assertEqual(failure.outcome_score, 0.0)
        self.assertEqual(interrupted.outcome_score, 0.15)
        self.assertTrue(success.metadata["positive_reaction"])

    def test_successful_and_failed_updates(self) -> None:
        context = build_context_key(intent="REQUEST_ACTION", target="dance")
        self.service.observe_evaluation(self.evaluation("dance"), context_key=context)
        failed = self.service.observe_evaluation(
            self.evaluation("dance", SkillStatus.FAILED),
            context_key=context,
        )
        knowledge = self.service.get_skill_state("dance")["contexts"][0]
        self.assertEqual(knowledge["trials"], 2)
        self.assertEqual(knowledge["successes"], 1)
        self.assertEqual(knowledge["failures"], 1)
        self.assertEqual(knowledge["confidence"], 0.45)
        self.assertIn("experience", failed)

    def test_ranking_prefers_high_performer_but_keeps_eligibility(self) -> None:
        context = build_context_key(intent="REQUEST_ACTION", target="dance")
        register_dance(self.service, "dance_01", "Dance 01", "First dance motion")
        register_dance(self.service, "dance_02", "Dance 02", "Second dance motion")
        for _ in range(4):
            self.service.observe_evaluation(
                self.evaluation("dance_01", SkillStatus.SUCCESS, "negative"), context_key=context
            )
        for _ in range(4):
            self.service.observe_evaluation(
                self.evaluation("dance_02", SkillStatus.SUCCESS, "positive"), context_key=context
            )
        ranking = self.service.rank_candidates(context, ["dance_01", "dance_02"])
        self.assertEqual(ranking[0]["skill_id"], "dance_02")
        self.assertGreater(ranking[0]["score"], ranking[1]["score"])
        self.assertIn("dance_01", [item["skill_id"] for item in ranking])

        self.library.disable("dance_02")
        ranking = self.service.rank_candidates(context, ["dance_01", "dance_02"])
        self.assertEqual([item["skill_id"] for item in ranking], ["dance_01"])

    def test_exploration_is_bounded_and_configurable(self) -> None:
        context = build_context_key(intent="REQUEST_ACTION", target="dance")
        forced = SkillLearningService(
            self.library,
            LearningConfig(exploration_epsilon=1.0),
            clock=self.clock,
        )
        candidates = forced.rank_candidates(context, ["dance"], choose=True)
        self.assertTrue(candidates[0]["exploration"])

        disabled = SkillLearningService(
            self.library,
            LearningConfig(exploration_epsilon=0.0),
            clock=self.clock,
        )
        candidates = disabled.rank_candidates(context, ["dance"], choose=True)
        self.assertFalse(candidates[0]["exploration"])

    def test_persistence_survives_restart(self) -> None:
        context = build_context_key(intent="REQUEST_ACTION", target="dance")
        register_dance(self.service)
        self.service.observe_evaluation(
            self.evaluation("dance_02", SkillStatus.SUCCESS, "positive"),
            context_key=context,
        )
        self.service.register_learned_composite(
            "learned_dance_pair",
            [("dance_02", {}), ("look_at", {"target": "CAMERA"})],
            context,
            name="Dance Then Look",
            description="Dance and look at the camera.",
        )

        repository = SQLiteSkillLearningRepository(self.database_path)
        restarted = SkillLearningService(
            create_default_skill_library(),
            LearningConfig(),
            repository,
            clock=self.clock,
        )
        knowledge = restarted.get_skill_state("dance_02")["contexts"][0]
        compositions = restarted.get_compositions()
        repository.close()
        self.assertEqual(knowledge["trials"], 1)
        self.assertEqual(knowledge["successes"], 1)

    def test_learned_outcome_uses_viewer_context_key(self) -> None:
        register_dance(self.service)
        provider = CapturingSkillLLM("dance_02")
        chat = ChatService(_ChatLLM(), ChatServiceConfig(max_messages_per_rate_window=20), clock=self.clock)
        brain = AutonomousBrain(
            provider,
            chat,
            BrainConfig(decision_interval_seconds=0.0),
            clock=self.clock,
            skill_library=self.library,
            learning_service=self.service,
            memory_service=self._memory(),
        )
        brain.chat.submit("viewer_01", "Lacrimosa can you dance?")
        dance_world = world(
            phone={"state": "Placed", "position": [1, 0.75, 0], "placement_surface": "Table"},
        )
        brain.tick(dance_world)
        brain.tick(
            dance_world,
            confirm(brain, ActionResult(ActionType.DANCE, ExecutionStatus.SUCCESS)),
        )

        context_key = "REQUEST_ACTION|DANCE|RESPOND_TO_VIEWER|STANDING|ACTIVE|TABLE|0_0"
        contexts = self.service.get_skill_state("dance_02")["contexts"]
        self.assertEqual([item["context_key"] for item in contexts], [context_key])

    def test_learned_composite_registration_and_reuse(self) -> None:
        from backend.skills.executor import SkillExecutor
        from backend.brain.planner import BrainPlanner

        context = build_context_key(intent="REQUEST_ACTION", target="dance")
        register_dance(self.service)
        skill = self.service.register_learned_composite(
            "phone_camera_dance",
            [("dance_02", {}), ("wave", {})],
            context,
            name="Learned Dance And Wave",
            description="Dance, then wave to the viewer.",
        )
        self.assertTrue(self.library.exists("phone_camera_dance"))
        executor = SkillExecutor(self.library, BrainPlanner(), self.clock)
        from backend.skills.models import SkillSelection

        result = executor.create_plan(
            SkillSelection("phone_camera_dance", {}, "test"),
            Goal("goal", GoalType.RESPOND_TO_VIEWER, 0.9, "test"),
            world(),
            now=BASE_TIME,
        )
        self.assertEqual(result.reason, "OK")
        self.assertEqual(
            [step.action for step in result.plan.steps],
            [ActionType.DANCE, ActionType.WAVE],
        )
        self.assertEqual(skill.composition[0].skill_id, "dance_02")
        self.assertEqual(skill.composition[0].parameters, {})

        with self.assertRaises(ValueError):
            self.service.register_learned_composite(
                "broken_composite",
                ["does_not_exist"],
                context,
                name="Broken",
                description="Missing dependency",
            )

        cyclic = Skill(
            skill_id="cyclic_skill",
            name="Cyclic",
            description="self cycle fixture",
            category=SkillCategory.COMPOSITE,
            composition=(SkillCompositionStep("cyclic_skill", {}),),
        )
        with self.assertRaises(SkillRegistryError):
            self.library.register(cyclic)

    def test_reset_preserves_skill_definitions(self) -> None:
        context = build_context_key(intent="REQUEST_ACTION", target="dance")
        register_dance(self.service)
        self.service.observe_evaluation(
            self.evaluation("dance_02"), context_key=context
        )
        self.service.reset()
        self.assertTrue(self.library.exists("dance_02"))
        self.assertEqual(self.service.get_skill_state("dance_02")["contexts"], [])
        self.assertEqual(self.service.get_compositions(), [])


class CapturingSkillLLM(LLMProvider):
    def __init__(self, skill_id: str = "dance_02") -> None:
        self.skill_id = skill_id
        self.prompts: list[str] = []

    def generate(self, request: LLMRequest) -> str:
        self.prompts.append(request.prompt)
        return json.dumps({"skill_id": self.skill_id, "parameters": {}, "reason": "learned selection"})


class BrainLearningIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FixedClock()
        self.temp_directory = tempfile.TemporaryDirectory()
        self.library = create_default_skill_library()
        self.learning = SkillLearningService(
            self.library,
            LearningConfig(),
            clock=self.clock,
        )
        register_dance(self.learning)

    def tearDown(self) -> None:
        if getattr(self, "memory_service", None):
            self.memory_service.close()
        self.temp_directory.cleanup()

    def _memory(self) -> MemoryService:
        database = Path(self.temp_directory.name) / "memory.db"
        service = MemoryService(
            SQLiteMemoryRepository(database),
            MemoryConfig(database_path=database),
        )
        self.memory_service = service
        return service

    def _brain(self, provider: CapturingSkillLLM) -> AutonomousBrain:
        chat = ChatService(_ChatLLM(), ChatServiceConfig(max_messages_per_rate_window=20), clock=self.clock)
        memory = self._memory()
        return AutonomousBrain(
            provider,
            chat,
            BrainConfig(decision_interval_seconds=0.0),
            clock=self.clock,
            skill_library=self.library,
            learning_service=self.learning,
            memory_service=memory,
        )

    def test_learned_ranking_enters_prompt_and_outcome_updates(self) -> None:
        provider = CapturingSkillLLM("dance_02")
        brain = self._brain(provider)
        brain.chat.submit("viewer_01", "Lacrimosa can you dance?")
        tick = brain.tick(world())
        self.assertEqual(tick.status, BrainStatus.EXECUTING)
        self.assertEqual(tick.action.action, ActionType.DANCE)
        self.assertIn("learned_skill_ranking", provider.prompts[-1])
        self.assertIn("dance_02", provider.prompts[-1])

        final = brain.tick(world(), confirm(brain, ActionResult(ActionType.DANCE, ExecutionStatus.SUCCESS)))
        self.assertEqual(final.status, BrainStatus.IDLE)
        contexts = self.learning.get_skill_state("dance_02")["contexts"]
        self.assertEqual(len(contexts), 1)
        self.assertEqual(contexts[0]["trials"], 1)
        self.assertEqual(contexts[0]["successes"], 1)

    def test_viewer_reaction_updates_learned_outcome(self) -> None:
        provider = CapturingSkillLLM("dance_02")
        brain = self._brain(provider)
        brain.chat.submit("viewer_01", "Lacrimosa can you dance?")
        first = brain.tick(world())
        self.assertEqual(first.action.action, ActionType.DANCE)
        final = brain.tick(
            world(),
            confirm(brain, ActionResult(
                ActionType.DANCE,
                ExecutionStatus.SUCCESS,
                viewer_response="positive",
            )),
        )
        self.assertEqual(final.status, BrainStatus.IDLE)
        evaluation = brain.skill_executor.evaluations[-1]
        self.assertEqual(evaluation.viewer_response, "positive")
        knowledge = self.learning.get_skill_state("dance_02")["contexts"][0]
        self.assertEqual(knowledge["trials"], 1)
        self.assertEqual(knowledge["positive_reactions"], 1)
        self.assertGreater(knowledge["average_score"], 0.70)

    def test_duplicate_late_success_is_not_observed_twice(self) -> None:
        provider = CapturingSkillLLM("dance_02")
        brain = self._brain(provider)
        brain.chat.submit("viewer_01", "Lacrimosa can you dance?")
        brain.tick(world())
        final = brain.tick(
            world(),
            confirm(brain, ActionResult(ActionType.DANCE, ExecutionStatus.SUCCESS)),
        )
        duplicate = brain.tick(
            world(),
            confirm(brain, ActionResult(ActionType.DANCE, ExecutionStatus.SUCCESS)),
        )

        self.assertEqual(final.status, BrainStatus.IDLE)
        self.assertEqual(len(brain.skill_executor.evaluations), 1)
        self.assertEqual(self.learning.get_skill_state("dance_02")["contexts"][0]["trials"], 1)

    def test_unrelated_success_does_not_reobserve_completed_skill(self) -> None:
        provider = CapturingSkillLLM("dance_02")
        brain = self._brain(provider)
        brain.chat.submit("viewer_01", "Lacrimosa can you dance?")
        brain.tick(world())
        final = brain.tick(
            world(),
            confirm(brain, ActionResult(ActionType.DANCE, ExecutionStatus.SUCCESS)),
        )
        idle_world = world(
            stream={"is_streaming": True, "camera_available": True, "camera_active": True}
        )
        brain.reasoner.provider = MockBrainLLMProvider()
        idle_decision = brain.tick(idle_world)
        unrelated = brain.tick(
            idle_world,
            confirm(brain, ActionResult(idle_decision.action.action, ExecutionStatus.SUCCESS)),
        )

        self.assertEqual(final.status, BrainStatus.IDLE)
        self.assertNotEqual(idle_decision.action.action, ActionType.DANCE)
        self.assertEqual(unrelated.status, BrainStatus.IDLE)
        self.assertEqual(len(brain.skill_executor.evaluations), 1)
        knowledge = self.learning.get_skill_state("dance_02")["contexts"][0]
        self.assertEqual(knowledge["trials"], 1)

    def test_invalid_skill_still_rejected_with_learning_enabled(self) -> None:
        brain = self._brain(CapturingSkillLLM("fly_to_moon"))
        brain.chat.submit("viewer_01", "please fly")
        tick = brain.tick(world())
        self.assertEqual(tick.status, BrainStatus.ERROR)
        self.assertEqual(tick.message, "planning_failed:SKILL_NOT_FOUND")

    def test_legacy_action_plan_still_works(self) -> None:
        from backend.brain.reasoning import MockBrainLLMProvider

        brain = self._brain(MockBrainLLMProvider())
        brain.chat.submit("viewer_01", "Lacrimosa, go to the table.")
        tick = brain.tick(world())
        self.assertEqual(tick.status, BrainStatus.EXECUTING)
        self.assertEqual(tick.action.action, ActionType.MOVE_TO)

    def test_legacy_action_result_does_not_crash_learning(self) -> None:
        from backend.brain.reasoning import MockBrainLLMProvider

        brain = self._brain(MockBrainLLMProvider())
        brain.chat.submit("viewer_01", "Lacrimosa, go to the table.")
        first = brain.tick(world())
        self.assertEqual(first.status, BrainStatus.EXECUTING)
        self.assertEqual(first.action.action, ActionType.MOVE_TO)
        second = brain.tick(world(), confirm(brain, ActionResult(ActionType.MOVE_TO, ExecutionStatus.SUCCESS)))
        self.assertEqual(second.status, BrainStatus.EXECUTING)
        self.assertEqual(second.action.action, ActionType.LOOK_AT)

    def test_mock_provider_selects_wave_and_camera_skills(self) -> None:
        from backend.brain.reasoning import MockBrainLLMProvider

        brain = self._brain(MockBrainLLMProvider())
        brain.chat.submit("viewer_01", "Can you wave?")
        wave = brain.tick(world())
        self.assertEqual(wave.action.action, ActionType.WAVE)
        brain.tick(world(), confirm(brain, ActionResult(ActionType.WAVE, ExecutionStatus.SUCCESS)))

        brain.chat.submit("viewer_01", "Look at the camera")
        look = brain.tick(world())
        self.assertEqual(look.action.action, ActionType.LOOK_AT)
        self.assertEqual(look.action.target, "CAMERA")

    def test_rejected_legacy_action_does_not_crash_learning(self) -> None:
        from backend.brain.reasoning import MockBrainLLMProvider

        brain = self._brain(MockBrainLLMProvider())
        brain.chat.submit("viewer_01", "Lacrimosa can you dance?")
        first = brain.tick(world())
        self.assertEqual(first.status, BrainStatus.EXECUTING)
        final = brain.tick(world(), confirm(brain, ActionResult(ActionType.MOVE_TO, ExecutionStatus.REJECTED)))
        self.assertEqual(final.status, BrainStatus.EXECUTING)


class _ChatLLM(LLMProvider):
    def generate(self, request: LLMRequest) -> str:
        return json.dumps({"speech": "Okay.", "intent": "REQUEST_ACTION", "requested_action": None, "priority": 0.8})


if __name__ == "__main__":
    unittest.main()
