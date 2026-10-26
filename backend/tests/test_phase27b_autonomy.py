"""PHASE 27B focused tests: context-aware autonomy and adaptation.

These pin the contract PHASE 27B added on top of the PHASE 27A foundation:

* initiative decisions read the *actual* context -- stream lifecycle, camera
  verdict, phone state, behaviour preference and existing ``SkillKnowledge`` --
  from the authoritative services instead of a private copy of live state;
* hard context rules (phone/camera/stream/audience) filter candidates first,
  and verified knowledge or personality/behaviour may only reorder the ones
  that already passed;
* an active autonomous goal is invalidated, not resumed, when the world stops
  supporting its premise, and every terminal path is classified honestly
  (completed / failed / interrupted / invalidated / timed_out / cancelled);
* retries stay bounded by the existing ``BrainConfig.max_retries`` and the
  existing cooldown/repetition journal;
* diagnostics explain the decision (candidates, rejections, context,
  provenance) without ever presenting an injected plan as a Brain decision.

Every test here is deterministic and runs against the real collaborating
services (``SkillExecutor``, ``SkillLearningService`` on a temporary SQLite
repository, the real chat/social stack). No fabricated learning records exist
outside the isolated tests that seed them explicitly.
"""

from __future__ import annotations

import random
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any

from backend.brain.goals import (
    INITIATIVE_BEHAVIOR_WEIGHT,
    INITIATIVE_KNOWLEDGE_BASELINE,
    INITIATIVE_KNOWLEDGE_SCORE_CEILING,
    INITIATIVE_KNOWLEDGE_SCORE_FLOOR,
    INITIATIVE_KNOWLEDGE_WEIGHT,
    INITIATIVE_PRIORITY_BASE,
    INITIATIVE_PRIORITY_CURIOSITY_WEIGHT,
    INITIATIVE_REQUIRES_AUDIENCE,
    INITIATIVE_REQUIRES_CAMERA,
    INITIATIVE_REQUIRES_PHONE,
    INITIATIVE_WAIT,
    STREAM_LIFECYCLE_INACTIVE,
    STREAM_LIFECYCLE_LIVE,
    STREAM_LIFECYCLE_STARTING,
    STREAM_LIFECYCLE_STOPPING,
    STREAM_LIFECYCLE_UNKNOWN,
    GoalEvaluator,
    GoalEvaluatorConfig,
    InitiativeContext,
    InitiativeDecision,
    InitiativeKnowledge,
    _InitiativeCandidate,
    initiative_context_rejection,
    initiative_candidate_score,
    initiative_knowledge_contribution,
    normalize_stream_lifecycle,
    stream_lifecycle_from_world,
)
from backend.brain.loop import (
    DECISION_SOURCE_INITIATIVE,
    MAX_INITIATIVE_CANDIDATES_IN_STATE,
    AutonomousBrain,
    BrainConfig,
    BrainStatus,
)
from backend.brain.models import (
    ActionResult,
    ActionRequest,
    ActionType,
    ExecutionStatus,
    Goal,
    GoalStatus,
    GoalType,
    Plan,
    PlanStep,
)
from backend.brain.personality import load_personality
from backend.brain.service import BrainService
from backend.brain.world_state import build_world_state
from backend.chat.service import ChatService, ChatServiceConfig
from backend.learning.models import LearningConfig
from backend.learning.repository import SQLiteSkillLearningRepository
from backend.learning.service import SkillLearningService
from backend.llm.mock import MockLLMProvider
from backend.skills.builtins import build_builtin_skills
from backend.skills.library import SkillLibrary, create_default_skill_library
from backend.skills.models import SkillEvaluation, SkillStatus
from backend.social.service import SocialAutonomyService

from backend.tests import test_behavior as behavior
from backend.tests.dispatch_support import confirm


class StreamingStub:
    """The narrow read-model the Brain is allowed to read from streaming."""

    def __init__(self, state: Any) -> None:
        self._state = state

    def get_state(self) -> dict[str, Any]:
        return {"state": self._state}


def snapshot(**overrides: Any) -> dict[str, Any]:
    """An idle, observed, streaming room with one viewer at CENTER."""
    value: dict[str, Any] = {
        "character": {
            "position": [0, 0, 0],
            "rotation_y": 90,
            "posture": "standing",
            "current_action": "idle",
        },
        "phone": {
            "state": "Placed",
            "position": [1, 0.75, 0],
            "placement_surface": "TABLE",
        },
        "stream": {
            "is_streaming": True,
            "camera_available": True,
            "camera_active": True,
            "viewer_count": 3,
        },
        "targets": {
            "CENTER": {
                "position": [0, 0, 0],
                "navigation_target": True,
                "look_at_target": True,
            },
            "CAMERA": {
                "position": [2, 0.8, 0],
                "navigation_target": False,
                "look_at_target": True,
            },
            "CAMERA_STAND": {
                "position": [1.5, 0, 0],
                "navigation_target": True,
                "look_at_target": True,
            },
            "TABLE": {
                "position": [1, 0.75, 0],
                "navigation_target": True,
                "look_at_target": True,
            },
        },
    }
    for key, override in overrides.items():
        current = value.get(key)
        if key != "targets" and isinstance(current, dict) and isinstance(override, dict):
            merged = dict(current)
            merged.update(override)
            value[key] = merged
        else:
            value[key] = override
    return value


def world(**overrides: Any):
    return build_world_state(snapshot(**overrides))


def success(action: ActionType, target: str | None = None) -> ActionResult:
    return ActionResult(action, ExecutionStatus.SUCCESS, "reported", target)


def failure(action: ActionType, reason: str = "rejected_by_unity") -> ActionResult:
    return ActionResult(action, ExecutionStatus.FAILED, reason)


def rejected(action: ActionType, reason: str = "rejected_by_unity") -> ActionResult:
    return ActionResult(action, ExecutionStatus.REJECTED, reason)


def library_with(*skill_ids: str) -> SkillLibrary:
    """A real library containing exactly the named registered skills."""
    wanted = {str(skill_id).strip().lower() for skill_id in skill_ids}
    library = SkillLibrary()
    for skill in build_builtin_skills():
        if skill.skill_id in wanted:
            library.register(skill)
    return library


def evaluator() -> GoalEvaluator:
    return GoalEvaluator(
        load_personality(),
        config=GoalEvaluatorConfig(explore_interval_seconds=3600.0),
    )


def rejection_reasons(decision: InitiativeDecision) -> dict[str, str]:
    return {activity: reason for activity, reason in decision.rejections}


class Autonomy27BTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = behavior.FakeClock()
        self._repositories: list[SQLiteSkillLearningRepository] = []
        self._temp_dirs: list[tempfile.TemporaryDirectory] = []

    def tearDown(self) -> None:
        for repository in self._repositories:
            repository.close()
        for directory in self._temp_dirs:
            directory.cleanup()

    def learning_service(
        self, library: SkillLibrary
    ) -> SkillLearningService:
        directory = tempfile.TemporaryDirectory()
        self._temp_dirs.append(directory)
        repository = SQLiteSkillLearningRepository(Path(directory.name) / "learning.db")
        self._repositories.append(repository)
        return SkillLearningService(
            library,
            LearningConfig(),
            repository,
            clock=self.clock,
            random_source=random.Random(10),
        )

    def brain(
        self,
        *,
        library: SkillLibrary | None = None,
        config: BrainConfig | None = None,
        streaming: Any = None,
        with_learning: bool = True,
    ) -> AutonomousBrain:
        selected_library = library if library is not None else create_default_skill_library()
        chat = ChatService(
            MockLLMProvider(),
            ChatServiceConfig(max_messages_per_rate_window=20),
            clock=self.clock,
        )
        learning = self.learning_service(selected_library) if with_learning else None
        instance = AutonomousBrain(
            behavior.CapturingProvider(),
            chat,
            config
            or BrainConfig(decision_interval_seconds=0.0, initiative_interval_seconds=5.0),
            personality=load_personality(),
            clock=self.clock,
            skill_library=selected_library,
            learning_service=learning,
            speech_service=None,
            social_service=SocialAutonomyService(clock=self.clock),
            streaming=streaming,
        )
        instance.goal_evaluator.config = GoalEvaluatorConfig(explore_interval_seconds=3600.0)
        instance._chat = chat
        return instance

    def settle(self, brain: AutonomousBrain, world_value) -> None:
        """Consume the one due legacy explore plan, returning to IDLE."""
        first = brain.tick(world_value, None)
        self.assertIsNotNone(first.action, "the legacy explore tick must plan")
        action = first.action.action
        follow = brain.tick(world_value, confirm(brain, ActionResult(action, ExecutionStatus.SUCCESS, "reported")))
        self.assertIn(follow.status, {BrainStatus.IDLE, BrainStatus.EXECUTING})


# ------------------------------------------------------------ stream lifecycle


class StreamLifecyclePolicyTests(Autonomy27BTestBase):
    def test_live_stream_prefers_the_stream_facing_activity(self) -> None:
        decision = evaluator().evaluate_initiative(
            world(),
            availability=lambda *_: None,
            context=InitiativeContext(stream_lifecycle=STREAM_LIFECYCLE_LIVE),
        )
        self.assertEqual(
            decision.activity, "PERFORM_REGISTERED_SKILL:setup_camera_and_dance"
        )
        self.assertEqual(decision.rejections, ())

    def test_inactive_stream_rejects_audience_candidates(self) -> None:
        decision = evaluator().evaluate_initiative(
            world(),
            availability=lambda *_: None,
            context=InitiativeContext(stream_lifecycle=STREAM_LIFECYCLE_INACTIVE),
        )
        reasons = rejection_reasons(decision)
        for activity in (
            "PERFORM_REGISTERED_SKILL:setup_camera_and_dance",
            "PERFORM_REGISTERED_SKILL:dance",
            "PERFORM_REGISTERED_SKILL:wave",
        ):
            self.assertEqual(reasons.get(activity), "STREAM_STATE_INCOMPATIBLE")
        # A stream-independent activity is still a valid choice, so the
        # character is never forced into waiting when something honest exists.
        self.assertEqual(decision.activity, "LOOK_AROUND")

    def test_stopping_stream_rejects_audience_candidates(self) -> None:
        decision = evaluator().evaluate_initiative(
            world(),
            availability=lambda *_: None,
            context=InitiativeContext(stream_lifecycle=STREAM_LIFECYCLE_STOPPING),
        )
        self.assertEqual(
            rejection_reasons(decision)["PERFORM_REGISTERED_SKILL:dance"],
            "STREAM_STATE_INCOMPATIBLE",
        )
        self.assertNotIn("PERFORM_REGISTERED_SKILL:dance", decision.eligible)

    def test_error_stream_rejects_audience_candidates(self) -> None:
        decision = evaluator().evaluate_initiative(
            world(),
            availability=lambda *_: None,
            context=InitiativeContext(stream_lifecycle="error"),
        )
        self.assertEqual(
            rejection_reasons(decision)["PERFORM_REGISTERED_SKILL:wave"],
            "STREAM_STATE_INCOMPATIBLE",
        )

    def test_starting_stream_is_not_treated_as_inactive(self) -> None:
        decision = evaluator().evaluate_initiative(
            world(),
            availability=lambda *_: None,
            context=InitiativeContext(stream_lifecycle="CONNECTING"),
        )
        self.assertEqual(decision.rejections, ())
        self.assertEqual(
            decision.activity, "PERFORM_REGISTERED_SKILL:setup_camera_and_dance"
        )

    def test_unknown_stream_is_not_treated_as_inactive(self) -> None:
        for token in (STREAM_LIFECYCLE_UNKNOWN, "", None, "not-a-real-state"):
            self.assertEqual(normalize_stream_lifecycle(token), STREAM_LIFECYCLE_UNKNOWN)
        decision = evaluator().evaluate_initiative(
            world(),
            availability=lambda *_: None,
            context=InitiativeContext(),
        )
        self.assertEqual(
            decision.activity, "PERFORM_REGISTERED_SKILL:setup_camera_and_dance"
        )

    def test_missing_stream_flag_is_not_read_as_inactive(self) -> None:
        unreported = replace(world(), stream_state_known=None)
        self.assertEqual(stream_lifecycle_from_world(unreported), STREAM_LIFECYCLE_UNKNOWN)
        decision = evaluator().evaluate_initiative(
            unreported, availability=lambda *_: None
        )
        self.assertEqual(
            decision.activity, "PERFORM_REGISTERED_SKILL:setup_camera_and_dance"
        )

    def test_world_snapshot_supplies_the_lifecycle_when_context_omits_it(self) -> None:
        stopped = world(stream={"is_streaming": False})
        self.assertEqual(stream_lifecycle_from_world(stopped), STREAM_LIFECYCLE_INACTIVE)
        decision = evaluator().evaluate_initiative(stopped, availability=lambda *_: None)
        self.assertEqual(
            rejection_reasons(decision)["PERFORM_REGISTERED_SKILL:dance"],
            "STREAM_STATE_INCOMPATIBLE",
        )

    def test_autonomy_never_offers_a_stream_lifecycle_operation(self) -> None:
        candidates = evaluator()._initiative_candidates(world())
        skill_ids = {candidate.skill_id for candidate in candidates}
        self.assertTrue(skill_ids)
        self.assertTrue(
            skill_ids.isdisjoint(
                {"start_stream", "stop_stream", "restart_stream", "setup_camera_and_dance_start"}
            )
        )

    def test_stopping_lifecycle_from_the_streaming_service_reaches_selection(self) -> None:
        brain = self.brain(streaming=StreamingStub("STOPPING"))
        self.settle(brain, world())
        outcome = brain.tick(world(), None)
        self.assertEqual(outcome.status, BrainStatus.EXECUTING)
        self.assertEqual(brain.state.selected_activity, "LOOK_AROUND")
        self.assertIn(
            ("PERFORM_REGISTERED_SKILL:dance", "STREAM_STATE_INCOMPATIBLE"),
            brain.state.candidate_rejections,
        )
        self.assertEqual(brain.state.initiative_context["stream_lifecycle"], "stopping")

    def test_starting_lifecycle_from_the_streaming_service_is_viable(self) -> None:
        brain = self.brain(streaming=StreamingStub("CONNECTING"))
        self.settle(brain, world())
        brain.tick(world(), None)
        self.assertEqual(
            brain.state.selected_activity,
            "PERFORM_REGISTERED_SKILL:setup_camera_and_dance",
        )
        self.assertEqual(brain.state.initiative_context["stream_lifecycle"], "starting")

    def test_missing_streaming_service_uses_the_world_snapshot(self) -> None:
        """An absent service never overrides a world snapshot that did report.

        The streaming service is the authoritative lifecycle source when it is
        wired in; without it the Brain falls back to the ``is_streaming`` flag
        the snapshot actually carried, and only an unreported flag is unknown.
        """
        brain = self.brain(streaming=None)
        self.settle(brain, world())
        brain.tick(world(), None)
        self.assertEqual(brain.state.initiative_context["stream_lifecycle"], "live")
        self.assertEqual(
            brain.state.selected_activity,
            "PERFORM_REGISTERED_SKILL:setup_camera_and_dance",
        )

    def test_unreported_stream_state_defers_to_the_legacy_path(self) -> None:
        # With neither the streaming service nor the world flag reporting, the
        # lifecycle stays ``unknown``. The initiative gate only runs on a
        # positively live audience, so the tick is not an autonomous decision:
        # the legacy maintenance/evaluation path keeps owning it.
        raw = snapshot()
        raw["stream"].pop("is_streaming")
        unreported = build_world_state(raw, viewer_count_floor=3)
        self.assertIsNone(unreported.stream_state_known)
        self.assertEqual(stream_lifecycle_from_world(unreported), STREAM_LIFECYCLE_UNKNOWN)
        brain = self.brain(streaming=None)
        self.settle(brain, unreported)
        brain.tick(unreported, None)
        self.assertIsNone(brain.state.initiative_context)
        self.assertNotEqual(brain.state.decision_source, DECISION_SOURCE_INITIATIVE)


# ------------------------------------------------- prerequisites and hard rules


class PrerequisiteTests(Autonomy27BTestBase):
    def test_camera_unavailable_rejects_the_camera_candidate(self) -> None:
        decision = evaluator().evaluate_initiative(
            world(),
            availability=lambda *_: None,
            context=InitiativeContext(
                stream_lifecycle=STREAM_LIFECYCLE_LIVE, camera_available=False
            ),
        )
        self.assertEqual(
            rejection_reasons(decision)["PERFORM_REGISTERED_SKILL:setup_camera_and_dance"],
            "CAMERA_INACTIVE",
        )
        # The dance does not claim camera framing, so it stays honestly usable.
        self.assertEqual(decision.activity, "PERFORM_REGISTERED_SKILL:dance")

    def test_camera_unknown_does_not_block_the_camera_candidate(self) -> None:
        decision = evaluator().evaluate_initiative(
            replace(world(), camera_state_known=None),
            availability=lambda *_: None,
        )
        self.assertEqual(
            decision.activity, "PERFORM_REGISTERED_SKILL:setup_camera_and_dance"
        )

    def test_reported_camera_verdict_reaches_the_loop_decision(self) -> None:
        brain = self.brain()
        self.settle(brain, world(stream={"camera_available": False}))
        brain.tick(world(stream={"camera_available": False}), None)
        self.assertEqual(brain.state.selected_activity, "PERFORM_REGISTERED_SKILL:dance")
        self.assertEqual(brain.state.initiative_context["camera_available"], False)
        self.assertIn(
            ("PERFORM_REGISTERED_SKILL:setup_camera_and_dance", "CAMERA_INACTIVE"),
            brain.state.candidate_rejections,
        )

    def test_phone_unavailable_rejects_a_phone_dependent_candidate(self) -> None:
        candidate = _InitiativeCandidate(
            activity="PHONE_TASK",
            skill_id="place_phone",
            parameters={},
            reason="synthetic phone candidate",
            weight=1.0,
            requires=frozenset({INITIATIVE_REQUIRES_PHONE}),
        )
        for state in ("World", "Held", "Placed", ""):
            self.assertIsNone(
                initiative_context_rejection(
                    InitiativeContext(phone_state=state), candidate
                ),
                state,
            )
        for state in ("Missing", "Destroyed", "missing"):
            self.assertEqual(
                initiative_context_rejection(
                    InitiativeContext(phone_state=state), candidate
                ),
                "PHONE_UNAVAILABLE",
                state,
            )

    def test_every_candidate_rejected_by_context_records_an_explicit_wait(self) -> None:
        goal_evaluator = evaluator()
        phone_candidate = _InitiativeCandidate(
            activity="PHONE_TASK",
            skill_id="place_phone",
            parameters={},
            reason="synthetic phone candidate",
            weight=1.0,
            requires=frozenset({INITIATIVE_REQUIRES_PHONE}),
        )
        goal_evaluator._initiative_candidates = (  # type: ignore[method-assign]
            lambda _world: (phone_candidate,)
        )
        decision = goal_evaluator.evaluate_initiative(
            world(),
            availability=lambda *_: None,
            context=InitiativeContext(phone_state="Missing"),
        )
        self.assertTrue(decision.is_wait)
        self.assertEqual(
            rejection_reasons(decision)["PHONE_TASK"], "PHONE_UNAVAILABLE"
        )
        self.assertIn("PHONE_UNAVAILABLE", decision.reason)

    def test_no_registered_activity_requires_the_phone(self) -> None:
        candidates = evaluator()._initiative_candidates(world())
        self.assertTrue(candidates)
        for candidate in candidates:
            self.assertNotIn(INITIATIVE_REQUIRES_PHONE, candidate.requires)

    def test_hard_rules_cannot_be_overridden_by_knowledge_or_behaviour(self) -> None:
        candidate = _InitiativeCandidate(
            activity="PERFORM_REGISTERED_SKILL:setup_camera_and_dance",
            skill_id="setup_camera_and_dance",
            parameters={},
            reason="composite",
            weight=4.0,
            requires=frozenset(
                {INITIATIVE_REQUIRES_CAMERA, INITIATIVE_REQUIRES_AUDIENCE}
            ),
        )
        from backend.brain.goals import InitiativeKnowledge

        context = InitiativeContext(
            stream_lifecycle=STREAM_LIFECYCLE_LIVE,
            camera_available=False,
            preferred_skills=("setup_camera_and_dance",),
            knowledge={
                "setup_camera_and_dance": InitiativeKnowledge(
                    score=1.0, trials=9, successes=9
                )
            },
        )
        self.assertEqual(
            initiative_context_rejection(context, candidate), "CAMERA_INACTIVE"
        )

    def test_audience_requirement_follows_the_stream_lifecycle(self) -> None:
        candidate = _InitiativeCandidate(
            activity="WAVE",
            skill_id="wave",
            parameters={},
            reason="synthetic",
            weight=1.0,
            requires=frozenset({INITIATIVE_REQUIRES_AUDIENCE}),
        )
        for lifecycle, expected in (
            (STREAM_LIFECYCLE_LIVE, None),
            (STREAM_LIFECYCLE_STARTING, None),
            (STREAM_LIFECYCLE_UNKNOWN, None),
            (STREAM_LIFECYCLE_INACTIVE, "STREAM_STATE_INCOMPATIBLE"),
            (STREAM_LIFECYCLE_STOPPING, "STREAM_STATE_INCOMPATIBLE"),
            ("ERROR", "STREAM_STATE_INCOMPATIBLE"),
        ):
            self.assertEqual(
                initiative_context_rejection(
                    InitiativeContext(stream_lifecycle=lifecycle), candidate
                ),
                expected,
                lifecycle,
            )


# -------------------------------------------- personality / behaviour influence


class BehaviorInfluenceTests(Autonomy27BTestBase):
    """PHASE 27B: the existing behaviour preference informs, and never rules.

    The existing ``BehaviorManager`` has always produced an advisory
    ``preferred_skills`` list; PHASE 27B is where an initiative decision reads
    it. These tests pin both halves of the contract: the bounded bonus is
    actually consumed by the real ``evaluate_initiative`` ranking, and it
    cannot outrank the documented static policy or a hard context rule.
    """

    def test_behaviour_preference_decides_a_near_tie_between_ranked_skills(self) -> None:
        """The real ranking path gives a preferred skill its bounded bonus.

        The registered candidate set deliberately keeps its documented static
        weights further apart than the bounded bonus, so this near-tie pair is
        isolated synthetic data that exercises the real
        ``GoalEvaluator.evaluate_initiative`` ordering; it is not a stand-in
        for a production candidate.
        """
        candidates = (
            _InitiativeCandidate(
                activity="PERFORM_REGISTERED_SKILL:dance",
                skill_id="dance",
                parameters={},
                reason="static lead",
                weight=2.04,
            ),
            _InitiativeCandidate(
                activity="PERFORM_REGISTERED_SKILL:wave",
                skill_id="wave",
                parameters={},
                reason="preferred candidate",
                weight=2.0,
            ),
        )

        class FixedCandidates(GoalEvaluator):
            def _initiative_candidates(self, world):  # type: ignore[override]
                return candidates

        fixed = FixedCandidates(load_personality())
        plain = fixed.evaluate_initiative(
            world(),
            availability=lambda *_: None,
            context=InitiativeContext(stream_lifecycle=STREAM_LIFECYCLE_LIVE),
        )
        self.assertEqual(plain.activity, "PERFORM_REGISTERED_SKILL:dance")

        preferred = fixed.evaluate_initiative(
            world(),
            availability=lambda *_: None,
            context=InitiativeContext(
                stream_lifecycle=STREAM_LIFECYCLE_LIVE,
                preferred_skills=("wave",),
            ),
        )
        self.assertEqual(preferred.activity, "PERFORM_REGISTERED_SKILL:wave")
        self.assertEqual(
            preferred.eligible,
            ("PERFORM_REGISTERED_SKILL:wave", "PERFORM_REGISTERED_SKILL:dance"),
        )

    def test_real_happy_state_reaches_the_decision_and_stays_bounded(self) -> None:
        """A real reaction drives the real behaviour state into the decision.

        The preference is read from the existing ``BehaviorManager`` advisory
        context inside a real Brain tick; the bounded bonus is positive, and
        the documented static order still wins because preference informs the
        ranking instead of rewriting the policy.
        """
        brain = self.brain()
        self.settle(brain, world())
        self.assertTrue(
            brain.social.record_viewer_reaction("positive", "viewer_01", self.clock())
        )
        self.clock.advance(12)
        brain.tick(world(stream={"viewer_count": 4}), None)

        advisory = brain.last_behavior_context or {}
        self.assertEqual(advisory.get("state"), "HAPPY")
        for skill_id in ("wave", "dance"):
            self.assertIn(skill_id, advisory.get("preferred_skills", ()))

        recorded = brain.state.initiative_context or {}
        self.assertEqual(recorded.get("behavior_state"), "HAPPY")
        self.assertIn("wave", recorded.get("preferred_skills", ()))
        self.assertEqual(
            brain.state.selected_activity,
            "PERFORM_REGISTERED_SKILL:setup_camera_and_dance",
        )

        candidates = {
            candidate.skill_id: candidate
            for candidate in brain.goal_evaluator._initiative_candidates(world())
        }
        preferred_context = InitiativeContext(
            stream_lifecycle=STREAM_LIFECYCLE_LIVE,
            preferred_skills=("wave",),
        )
        neutral_context = InitiativeContext(stream_lifecycle=STREAM_LIFECYCLE_LIVE)
        wave = candidates["wave"]
        self.assertEqual(
            round(
                initiative_candidate_score(wave, preferred_context)
                - initiative_candidate_score(wave, neutral_context),
                4,
            ),
            INITIATIVE_BEHAVIOR_WEIGHT,
        )
        self.assertLess(
            initiative_candidate_score(wave, preferred_context),
            initiative_candidate_score(candidates["dance"], neutral_context),
        )

    def test_personality_shapes_the_initiative_priority_below_viewer_priority(self) -> None:
        """The stable personality profile sets the bounded goal priority.

        Personality reaches the decision through the goal priority only: an
        initiative goal always stays below the configured viewer-interruption
        priority, so a real viewer request keeps winning through the existing
        Phase 26A machinery.
        """
        brain = self.brain()
        self.settle(brain, world())
        self.clock.advance(6)
        brain.tick(world(stream={"viewer_count": 4}), None)

        goal = brain.state.current_goal
        self.assertIsNotNone(goal)
        self.assertTrue(brain._is_initiative_goal(goal))
        expected = (
            INITIATIVE_PRIORITY_BASE
            + INITIATIVE_PRIORITY_CURIOSITY_WEIGHT * brain.personality.curiosity
        )
        self.assertAlmostEqual(goal.priority, expected, places=6)
        self.assertLess(goal.priority, brain.config.interruption_priority)


# --------------------------------------------------- SkillKnowledge integration


class SkillKnowledgeSelectionTests(Autonomy27BTestBase):
    def test_verified_repeated_failure_changes_the_real_selection(self) -> None:
        """Two real rejections written by the production path suppress the skill.

        This is the strongest available proof that existing ``SkillKnowledge``
        influences a decision: the records are not seeded by hand -- Unity's
        rejected result travels the normal observation/learning path twice, and
        the third decision refuses to retry the same invalid activity.
        """
        brain = self.brain()
        self.settle(brain, world())
        brain.tick(world(), None)
        self.assertEqual(
            brain.state.selected_activity,
            "PERFORM_REGISTERED_SKILL:setup_camera_and_dance",
        )

        self.clock.advance(1)
        brain.tick(world(), confirm(brain, rejected(ActionType.MOVE_TO)))
        self.clock.advance(6)
        brain.tick(world(stream={"viewer_count": 4}), None)
        self.assertEqual(
            brain.state.selected_activity,
            "PERFORM_REGISTERED_SKILL:setup_camera_and_dance",
        )
        self.clock.advance(1)
        brain.tick(world(stream={"viewer_count": 4}), confirm(brain, rejected(ActionType.MOVE_TO)))

        knowledge = brain.learning.get_skill_state("setup_camera_and_dance")
        self.assertEqual(len(knowledge["contexts"]), 1)
        record = knowledge["contexts"][0]
        self.assertEqual(record["trials"], 2)
        self.assertEqual(record["successes"], 0)

        self.clock.advance(6)
        final = brain.tick(world(stream={"viewer_count": 5}), None)
        self.assertEqual(final.status, BrainStatus.EXECUTING)
        self.assertEqual(brain.state.selected_activity, "PERFORM_REGISTERED_SKILL:dance")
        self.assertIn(
            ("PERFORM_REGISTERED_SKILL:setup_camera_and_dance", "RECENT_FAILURE"),
            brain.state.candidate_rejections,
        )

    def test_a_single_verified_failure_is_not_a_verdict(self) -> None:
        brain = self.brain(library=library_with("dance", "wave", "look_at"))
        self.settle(brain, world())
        context_key = brain._initiative_context_key(brain.learning, world())
        self.assertTrue(context_key)
        brain.learning.observe_evaluation(
            SkillEvaluation(
                skill_id="dance",
                execution_id="seed-1",
                status=SkillStatus.FAILED,
                duration_seconds=1.0,
                steps_completed=0,
                steps_failed=1,
            ),
            context_key=context_key,
        )
        brain.tick(world(), None)
        self.assertEqual(brain.state.selected_activity, "PERFORM_REGISTERED_SKILL:dance")
        self.assertNotIn(
            "RECENT_FAILURE",
            {reason for _activity, reason in brain.state.candidate_rejections},
        )

    def test_missing_knowledge_is_never_inferred_as_success_or_failure(self) -> None:
        brain = self.brain(library=library_with("dance", "wave", "look_at"))
        self.settle(brain, world())
        self.assertEqual(brain._initiative_knowledge(world()), {})
        brain.tick(world(), None)
        self.assertEqual(brain.state.selected_activity, "PERFORM_REGISTERED_SKILL:dance")
        self.assertNotIn(
            "RECENT_FAILURE",
            {reason for _activity, reason in brain.state.candidate_rejections},
        )

    def test_verified_success_is_recorded_under_the_decision_context(self) -> None:
        brain = self.brain(library=library_with("look_at"))
        self.settle(brain, world())
        decision_key = brain._initiative_context_key(brain.learning, world())
        self.assertTrue(decision_key)

        brain.tick(world(), None)
        self.assertEqual(brain.state.selected_activity, "LOOK_AROUND")
        self.clock.advance(1)
        done = brain.tick(world(), confirm(brain, success(ActionType.LOOK_AT, "CAMERA")))
        self.assertEqual(done.status, BrainStatus.IDLE)

        records = brain.learning.get_skill_state("look_at")["contexts"]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["context_key"], decision_key)
        self.assertEqual(records[0]["trials"], 1)
        self.assertEqual(records[0]["successes"], 1)

    def test_verified_success_bonus_cannot_invert_the_documented_policy(self) -> None:
        brain = self.brain(library=library_with("dance", "wave", "look_at"))
        self.settle(brain, world())
        context_key = brain._initiative_context_key(brain.learning, world())
        for index in range(4):
            brain.learning.observe_evaluation(
                SkillEvaluation(
                    skill_id="wave",
                    execution_id=f"wave-{index}",
                    status=SkillStatus.SUCCESS,
                    duration_seconds=1.0,
                    steps_completed=1,
                    steps_failed=0,
                ),
                context_key=context_key,
            )
        brain.tick(world(), None)
        # The bounded knowledge weight may lift `wave`, but the documented
        # static preference for `dance` still wins: evidence informs, it does
        # not silently rewrite the policy.
        self.assertEqual(brain.state.selected_activity, "PERFORM_REGISTERED_SKILL:dance")

    def test_decisions_never_mutate_the_learning_record(self) -> None:
        brain = self.brain(library=library_with("dance", "wave", "look_at"))
        self.settle(brain, world())
        context_key = brain._initiative_context_key(brain.learning, world())
        brain.learning.observe_evaluation(
            SkillEvaluation(
                skill_id="dance",
                execution_id="seed",
                status=SkillStatus.SUCCESS,
                duration_seconds=1.0,
                steps_completed=1,
                steps_failed=0,
            ),
            context_key=context_key,
        )
        brain.tick(world(), None)
        self.assertEqual(brain.state.selected_activity, "PERFORM_REGISTERED_SKILL:dance")
        record = brain.learning.get_skill_state("dance")["contexts"][0]
        self.assertEqual(record["trials"], 1)
        self.assertEqual(record["exploration_count"], 0)
        self.assertEqual(record["exploitation_count"], 0)

    def test_knowledge_read_is_bounded_to_the_candidate_skills(self) -> None:
        brain = self.brain(library=library_with("dance", "wave", "look_at"))
        self.settle(brain, world())
        context_key = brain._initiative_context_key(brain.learning, world())
        brain.learning.observe_evaluation(
            SkillEvaluation(
                skill_id="dance",
                execution_id="seed",
                status=SkillStatus.FAILED,
                duration_seconds=1.0,
                steps_completed=0,
                steps_failed=1,
            ),
            context_key=context_key,
        )
        knowledge = brain._initiative_knowledge(world())
        self.assertEqual(set(knowledge), {"dance"})

    def test_sparse_evidence_earns_a_smaller_bonus_than_saturated_evidence(
        self,
    ) -> None:
        """The bounded bonus is scaled by the evidence actually behind it.

        The existing ``SkillLearningService`` ranking score alone cannot tell
        these two records apart: one verified success reports the same perfect
        rate and the same observed quality as a dozen, because the service
        caps its own confidence term at ten trials and both records clear the
        blend's other two inputs. Reading that score verbatim would therefore
        let a single lucky trial speak almost as loudly as dozens of verified
        ones. The Brain scales the contribution by that same learning-model
        saturation point instead of inventing a second reliability model, and
        a missing record still contributes exactly nothing.
        """
        candidate = _InitiativeCandidate(
            activity="PERFORM_REGISTERED_SKILL:wave",
            skill_id="wave",
            parameters={},
            reason="synthetic",
            weight=2.0,
        )
        saturated = initiative_knowledge_contribution(
            candidate,
            InitiativeContext(
                stream_lifecycle=STREAM_LIFECYCLE_LIVE,
                knowledge={
                    "wave": InitiativeKnowledge(score=0.88, trials=12, successes=12)
                },
            ),
        )
        sparse = initiative_knowledge_contribution(
            candidate,
            InitiativeContext(
                stream_lifecycle=STREAM_LIFECYCLE_LIVE,
                knowledge={
                    "wave": InitiativeKnowledge(score=0.88, trials=1, successes=1)
                },
            ),
        )
        missing_context = InitiativeContext(stream_lifecycle=STREAM_LIFECYCLE_LIVE)
        self.assertEqual(initiative_knowledge_contribution(candidate, missing_context), 0.0)
        self.assertEqual(
            initiative_candidate_score(candidate, missing_context), candidate.weight
        )
        self.assertAlmostEqual(saturated, 0.2457, places=4)
        self.assertAlmostEqual(sparse, 0.0246, places=4)
        self.assertLess(sparse, saturated)
        self.assertLessEqual(saturated, INITIATIVE_KNOWLEDGE_WEIGHT)

    def test_knowledge_envelope_matches_the_learning_model(self) -> None:
        """The normalization envelope is the range the real model can emit.

        The two constants are not a tuned guess: they are the extremes the
        existing ``OutcomeEvaluator`` plus ``SkillLearningService`` pipeline can
        actually report for a verified record. Driving real evaluations through
        the production observation path and reading the ranking score back pins
        the envelope to that model, so a future change to either side fails
        here instead of silently shrinking the usable knowledge band -- a plain
        perfect production row already scores 0.88, near the top of a 0.0..1.0
        scale the model can never fully use.
        """
        service = self.learning_service(library_with("wave"))
        ceiling_key = "EXPLORE_ROOM|STANDING|ACTIVE"
        floor_key = "EXPLORE_ROOM|STANDING|INACTIVE"

        def ranking_score(context_key: str) -> float:
            rows = service.rank_candidates(context_key, ["wave"])
            self.assertEqual(len(rows), 1)
            return float(rows[0]["score"])

        for index in range(10):
            service.observe_evaluation(
                SkillEvaluation(
                    skill_id="wave",
                    execution_id=f"best-{index}",
                    status=SkillStatus.SUCCESS,
                    duration_seconds=1.0,
                    steps_completed=1,
                    steps_failed=0,
                    viewer_response="positive",
                ),
                context_key=ceiling_key,
            )
        for index in range(10):
            service.observe_evaluation(
                SkillEvaluation(
                    skill_id="wave",
                    execution_id=f"worst-{index}",
                    status=SkillStatus.FAILED,
                    duration_seconds=1.0,
                    steps_completed=0,
                    steps_failed=1,
                    viewer_response="negative",
                ),
                context_key=floor_key,
            )

        ceiling = ranking_score(ceiling_key)
        floor = ranking_score(floor_key)
        self.assertAlmostEqual(ceiling, INITIATIVE_KNOWLEDGE_SCORE_CEILING, places=4)
        self.assertAlmostEqual(floor, INITIATIVE_KNOWLEDGE_SCORE_FLOOR, places=4)

        # The envelope maps onto the whole bounded band: the model's best and
        # worst saturated records earn exactly +/-INITIATIVE_KNOWLEDGE_WEIGHT.
        candidate = _InitiativeCandidate(
            activity="PERFORM_REGISTERED_SKILL:wave",
            skill_id="wave",
            parameters={},
            reason="envelope probe",
            weight=2.0,
        )
        for score, expected in (
            (ceiling, INITIATIVE_KNOWLEDGE_WEIGHT),
            (floor, -INITIATIVE_KNOWLEDGE_WEIGHT),
        ):
            contribution = initiative_knowledge_contribution(
                candidate,
                InitiativeContext(
                    knowledge={
                        "wave": InitiativeKnowledge(
                            score=score, trials=10, successes=10
                        )
                    }
                ),
            )
            self.assertAlmostEqual(contribution, expected, places=4)

    def test_candidate_specific_knowledge_decides_a_near_tie(self) -> None:
        """Isolated near-tie data: verified evidence can change the ordering.

        The registered candidate table keeps its documented static weights
        further apart than the bounded bonus and its only near-tie pair maps
        onto a single skill (``walk_to``), so this pair is deliberately
        synthetic. It exercises the real
        ``GoalEvaluator.evaluate_initiative`` ranking to pin the rule the
        production table would need in order to reorder *without* weakening
        static relevance, and the decision now reports the term that decided
        it.
        """
        candidates = (
            _InitiativeCandidate(
                activity="PERFORM_REGISTERED_SKILL:dance",
                skill_id="dance",
                parameters={},
                reason="static lead, no evidence yet",
                weight=2.0,
            ),
            _InitiativeCandidate(
                activity="PERFORM_REGISTERED_SKILL:wave",
                skill_id="wave",
                parameters={},
                reason="slightly behind, verified success",
                weight=1.96,
            ),
        )

        class FixedCandidates(GoalEvaluator):
            def _initiative_candidates(self, world):  # type: ignore[override]
                return candidates

        fixed = FixedCandidates(load_personality())
        plain = fixed.evaluate_initiative(
            world(),
            availability=lambda *_: None,
            context=InitiativeContext(stream_lifecycle=STREAM_LIFECYCLE_LIVE),
        )
        self.assertEqual(plain.activity, "PERFORM_REGISTERED_SKILL:dance")
        self.assertEqual(
            dict(plain.knowledge_contributions)[
                "PERFORM_REGISTERED_SKILL:wave"
            ],
            0.0,
        )

        informed = fixed.evaluate_initiative(
            world(),
            availability=lambda *_: None,
            context=InitiativeContext(
                stream_lifecycle=STREAM_LIFECYCLE_LIVE,
                knowledge={
                    "wave": InitiativeKnowledge(score=0.88, trials=12, successes=12)
                },
            ),
        )
        self.assertEqual(informed.activity, "PERFORM_REGISTERED_SKILL:wave")
        self.assertEqual(
            informed.eligible,
            (
                "PERFORM_REGISTERED_SKILL:wave",
                "PERFORM_REGISTERED_SKILL:dance",
            ),
        )
        contributions = dict(informed.knowledge_contributions)
        self.assertEqual(contributions["PERFORM_REGISTERED_SKILL:wave"], 0.2457)
        self.assertEqual(contributions["PERFORM_REGISTERED_SKILL:dance"], 0.0)

    def test_uniform_saturated_evidence_cannot_reorder_the_documented_policy(
        self,
    ) -> None:
        """Identical records earn identical contributions, so order is kept.

        This is the honest reading of the live production table: every
        autonomy candidate that has evidence under
        ``EXPLORE_ROOM|STANDING|ACTIVE`` has saturated, identical evidence, so
        no weight -- however large -- could reorder them. A uniform additive
        term cannot change an ordering; only *differing* evidence can, and the
        live rows do not differ. The band is calibrated for the narrowest
        *differing* pair the candidate table can produce (see
        ``test_knowledge_band_is_calibrated_for_comparable_static_gaps``); it
        is not, and cannot be, a way for uniform evidence to change anything.
        """
        real = evaluator()
        candidates = real._initiative_candidates(world())
        self.assertTrue(candidates)

        knowledge = {
            candidate.skill_id: InitiativeKnowledge(
                score=1.0, trials=50, successes=50
            )
            for candidate in candidates
        }
        static = real.evaluate_initiative(
            world(),
            availability=lambda *_: None,
            context=InitiativeContext(stream_lifecycle=STREAM_LIFECYCLE_LIVE),
        )
        informed = real.evaluate_initiative(
            world(),
            availability=lambda *_: None,
            context=InitiativeContext(
                stream_lifecycle=STREAM_LIFECYCLE_LIVE, knowledge=knowledge
            ),
        )
        self.assertEqual(informed.activity, static.activity)
        self.assertEqual(informed.eligible, static.eligible)
        contributions = [value for _activity, value in informed.knowledge_contributions]
        self.assertEqual(len(contributions), len(static.eligible))
        self.assertTrue(
            all(0.0 < value <= INITIATIVE_KNOWLEDGE_WEIGHT for value in contributions)
        )
        # Uniform evidence earns one uniform contribution: nothing to reorder.
        self.assertEqual(len(set(contributions)), 1)

    def test_knowledge_band_is_calibrated_for_comparable_static_gaps(self) -> None:
        """The bounded band can decide a comparable pair -- and only that.

        Measured against the real candidate table: with the character away
        from CENTER both ``walk_to`` destinations are eligible, so the
        narrowest gap between candidates backed by *different* skills is 0.4
        (``look_at`` at 1.0 against ``walk_to`` at 0.6). The live CURIOUS
        behaviour state adds its bounded 0.06 preference to ``look_at``, so the
        narrowest gap *as the ranking actually presents the pair* is 0.46.
        That measured gap is the requirement behind the constant:
        ``0.46 / 2 = 0.23``, set to 0.28 so the bound holds with margin. The
        knowledge term normalizes the learning model's entire achievable score
        range onto ``+/-INITIATIVE_KNOWLEDGE_WEIGHT`` (see
        ``test_knowledge_envelope_matches_the_learning_model``), so the widest
        spread two saturated records can produce is the whole signed band,
        ``2 * W = 0.56`` -- wide enough for genuinely divergent verified
        evidence to flip exactly such a comparable pair. The narrowest
        *material* gap is 1.0 and stays wider than the whole band plus the
        behaviour term (0.62), so no record can leapfrog a more relevant
        activity. The 0.2 separation between the two ``walk_to`` destinations
        is deliberately not part of the comparison: they share one skill, one
        knowledge record and one contribution.
        """
        real = evaluator()
        away = world(character={"position": [3.0, 0.0, 0.0]})
        candidates = real._initiative_candidates(away)
        self.assertEqual(
            [candidate.weight for candidate in candidates],
            [4.0, 3.0, 2.0, 1.0, 0.6, 0.4],
        )
        weight_by_skill: dict[str, float] = {}
        for candidate in candidates:
            weight_by_skill[candidate.skill_id] = max(
                weight_by_skill.get(candidate.skill_id, 0.0), candidate.weight
            )
        weights = sorted(weight_by_skill.values())
        gaps = [round(high - low, 4) for low, high in zip(weights, weights[1:])]
        self.assertEqual(gaps, [0.4, 1.0, 1.0, 1.0])
        comparable_gap, material_gaps = gaps[0], gaps[1:]
        reachable_spread = round(
            2.0 * INITIATIVE_KNOWLEDGE_WEIGHT,
            4,
        )
        self.assertEqual(reachable_spread, 0.56)
        self.assertGreater(reachable_spread, comparable_gap)
        # The pair as the ranking actually presents it: the behaviour
        # preference widens its leader by the bounded behaviour term.
        self.assertGreater(
            reachable_spread, round(comparable_gap + INITIATIVE_BEHAVIOR_WEIGHT, 4)
        )
        self.assertLess(
            round(reachable_spread + INITIATIVE_BEHAVIOR_WEIGHT, 4),
            min(material_gaps),
        )
        measured_requirement = round(
            (comparable_gap + INITIATIVE_BEHAVIOR_WEIGHT) / 2.0,
            4,
        )
        self.assertEqual(measured_requirement, 0.23)
        self.assertGreaterEqual(INITIATIVE_KNOWLEDGE_WEIGHT, measured_requirement)

    def test_real_candidate_table_reorders_on_divergent_knowledge(self) -> None:
        """Persisted-style divergent evidence flips a real comparable pair.

        Uses the production candidate table rather than synthetic candidates:
        the character stands away from CENTER, so both movement destinations
        are eligible and the table offers ``look_at`` (1.0) against ``walk_to``
        (0.6) -- the only pair the calibrated band is allowed to decide. Only
        those two skills carry knowledge; the camera routine, ``dance`` and
        ``wave`` are not registered, so the availability gate keeps the
        comparison on exactly that pair. Strongly negative verified evidence
        for ``look_at`` plus strong positive evidence for ``walk_to`` changes
        the selection from LOOK_AROUND to MOVE_TO_SUITABLE_LOCATION.

        No behaviour state is supplied, so the pair is decided by the verified
        evidence alone; the live CURIOUS preference is covered by the
        calibration test above.
        """
        real = evaluator()
        away = world(character={"position": [3.0, 0.0, 0.0]})
        registered = library_with("look_at", "walk_to")

        def availability(skill_id: str, _parameters: Any) -> Any:
            return None if registered.get(skill_id) is not None else "SKILL_UNAVAILABLE"

        static = real.evaluate_initiative(
            away,
            availability=availability,
            context=InitiativeContext(stream_lifecycle=STREAM_LIFECYCLE_LIVE),
        )
        self.assertEqual(static.activity, "LOOK_AROUND")

        informed = real.evaluate_initiative(
            away,
            availability=availability,
            context=InitiativeContext(
                stream_lifecycle=STREAM_LIFECYCLE_LIVE,
                knowledge={
                    "look_at": InitiativeKnowledge(
                        score=0.10, trials=20, successes=4, failures=16
                    ),
                    "walk_to": InitiativeKnowledge(
                        score=1.0, trials=20, successes=20
                    ),
                },
            ),
        )
        self.assertEqual(informed.activity, "MOVE_TO_SUITABLE_LOCATION")
        self.assertEqual(informed.eligible[0], "MOVE_TO_SUITABLE_LOCATION")
        scores = dict(informed.scores)
        contributions = dict(informed.knowledge_contributions)
        # The failing record scores below the learning model's own floor, so it
        # earns the full negative term; the perfect record earns the full
        # positive one.
        self.assertAlmostEqual(contributions["LOOK_AROUND"], -0.28, places=4)
        self.assertAlmostEqual(
            contributions["MOVE_TO_SUITABLE_LOCATION"], 0.28, places=4
        )
        self.assertAlmostEqual(scores["LOOK_AROUND"], 0.72, places=4)
        self.assertAlmostEqual(
            scores["MOVE_TO_SUITABLE_LOCATION"], 0.88, places=4
        )

    def test_material_static_gap_survives_maximal_knowledge_contrast(self) -> None:
        """Even the widest possible evidence spread cannot cross a 1.0 gap.

        ``dance`` (3.0) and ``wave`` (2.0) are materially different activities.
        First the coded bound itself: even with one candidate at the extreme
        low end of the knowledge term, the other at the extreme high end and
        the behaviour preference on the trailing candidate, the material order
        still holds. Then the same rule on a reachable record pair: a badly
        failing (but not ``RECENT_FAILURE``) ``dance`` record against a perfect
        ``wave`` record keeps ``dance`` ahead with a wide margin. Verified
        evidence informs the ranking; it does not rewrite the policy.
        """
        real = evaluator()
        registered = library_with("dance", "wave")

        def availability(skill_id: str, _parameters: Any) -> Any:
            return None if registered.get(skill_id) is not None else "SKILL_UNAVAILABLE"

        # Coded bound: the whole knowledge band plus behaviour cannot close a
        # material static gap.
        self.assertGreater(
            round(3.0 - INITIATIVE_KNOWLEDGE_WEIGHT, 4),
            round(
                2.0 + INITIATIVE_KNOWLEDGE_WEIGHT + INITIATIVE_BEHAVIOR_WEIGHT,
                4,
            ),
        )
        decision = real.evaluate_initiative(
            world(),
            availability=availability,
            context=InitiativeContext(
                stream_lifecycle=STREAM_LIFECYCLE_LIVE,
                knowledge={
                    # 20 verified trials, 4 successes, average observed
                    # quality 0.1: the existing learning formula scores that
                    # record 0.28, well below the 0.45 unknown-skill baseline.
                    # (A zero-success record would instead trip the
                    # RECENT_FAILURE hard rule, which is covered elsewhere.)
                    "dance": InitiativeKnowledge(
                        score=0.28, trials=20, successes=4, failures=16
                    ),
                    "wave": InitiativeKnowledge(
                        score=1.0, trials=20, successes=20
                    ),
                },
            ),
        )
        self.assertEqual(decision.activity, "PERFORM_REGISTERED_SKILL:dance")
        scores = dict(decision.scores)
        self.assertAlmostEqual(
            scores["PERFORM_REGISTERED_SKILL:dance"], 2.8413, places=4
        )
        self.assertAlmostEqual(scores["PERFORM_REGISTERED_SKILL:wave"], 2.28, places=4)
        contributions = dict(decision.knowledge_contributions)
        self.assertAlmostEqual(
            contributions["PERFORM_REGISTERED_SKILL:dance"], -0.1587, places=4
        )
        self.assertAlmostEqual(
            contributions["PERFORM_REGISTERED_SKILL:wave"], 0.28, places=4
        )

    def test_knowledge_never_makes_an_ineligible_candidate_eligible(self) -> None:
        """A hard prerequisite is decided before any evidence is read.

        A perfect record cannot buy a candidate the phone it does not have:
        the context filter runs first, and the only honest end state is WAIT.
        """
        candidate = _InitiativeCandidate(
            activity="PHONE_TASK",
            skill_id="wave",
            parameters={},
            reason="synthetic phone-bound activity",
            weight=1.0,
            requires=frozenset({INITIATIVE_REQUIRES_PHONE}),
        )
        context = InitiativeContext(
            phone_state="Missing",
            knowledge={
                "wave": InitiativeKnowledge(score=1.0, trials=40, successes=40)
            },
        )
        self.assertEqual(
            initiative_context_rejection(context, candidate), "PHONE_UNAVAILABLE"
        )

        class FixedCandidates(GoalEvaluator):
            def _initiative_candidates(self, world):  # type: ignore[override]
                return (candidate,)

        decision = FixedCandidates(load_personality()).evaluate_initiative(
            world(), availability=lambda *_: None, context=context
        )
        self.assertTrue(decision.is_wait)
        self.assertEqual(rejection_reasons(decision)["PHONE_TASK"], "PHONE_UNAVAILABLE")
        self.assertEqual(decision.eligible, ())

    def test_repeated_failure_threshold_needs_verified_unsuccessful_trials(
        self,
    ) -> None:
        """The existing failure rule fires on repeated, not single, evidence."""
        candidate = _InitiativeCandidate(
            activity="PERFORM_REGISTERED_SKILL:wave",
            skill_id="wave",
            parameters={},
            reason="synthetic",
            weight=2.0,
        )

        def rejection(trials: int, successes: int) -> str | None:
            return initiative_context_rejection(
                InitiativeContext(
                    knowledge={
                        "wave": InitiativeKnowledge(
                            score=0.2, trials=trials, successes=successes
                        )
                    }
                ),
                candidate,
            )

        # One verified failure is a data point, not a verdict.
        self.assertIsNone(rejection(1, 0))
        # Two unsuccessful trials in this exact context do rule the retry out.
        self.assertEqual(rejection(2, 0), "RECENT_FAILURE")
        self.assertEqual(rejection(5, 0), "RECENT_FAILURE")
        # A record with any verified success is not a repeated failure.
        self.assertIsNone(rejection(3, 1))
        # And positive evidence never rejects a candidate.
        self.assertIsNone(rejection(50, 50))

    def test_persisted_knowledge_orders_the_real_selection_and_outcome_round_trips(
        self,
    ) -> None:
        """Controlled end-to-end proof that persisted evidence decides the work.

        The Scenario B acceptance question is causal, not statistical: when two
        genuinely eligible candidates differ *only* in their persisted verified
        evidence, does the ordinary Brain decision change -- and does the
        resulting outcome travel the ordinary executor/learning path?

        The fixture is deliberately isolated: a temporary SQLite repository
        seeded through ``SkillLearningService.observe_evaluation`` -- the very
        call the Brain itself makes after an observed outcome -- never the
        production database. World state, candidate eligibility, static weights
        and the behaviour preference are held fixed between the two runs; the
        only input that changes is the persisted ``SkillKnowledge``.

        ``look_at`` (LOOK_AROUND, static 1.0 plus the live CURIOUS preference
        ``+0.06``) is the unevidenced winner, 0.46 ahead of ``walk_to``
        (MOVE_TO_SUITABLE_LOCATION, static 0.6). A heavily failing ``look_at``
        record against a perfect ``walk_to`` record moves the bounded knowledge
        contributions to ``-0.2459`` / ``+0.28``, which flips the real
        decision. The selected skill is then dispatched by the existing
        ``SkillExecutor`` under a real execution id, the observed arrival is
        journaled under the decision's own context key, and a freshly reopened
        repository reads that outcome back for the next decision.
        """
        library = library_with("look_at", "walk_to")
        away = world(character={"position": [3.0, 0.0, 0.0]})

        # ---- control: identical world, no persisted evidence ----------------
        control = self.brain(library=library)
        self.settle(control, away)
        control.tick(away, None)
        self.assertEqual(control.state.selected_activity, "LOOK_AROUND")
        control_contributions = dict(control.state.knowledge_contributions)
        self.assertEqual(control_contributions["LOOK_AROUND"], 0.0)
        self.assertEqual(control_contributions["MOVE_TO_SUITABLE_LOCATION"], 0.0)
        control_scores = dict(control.state.candidate_scores)
        self.assertAlmostEqual(
            round(
                control_scores["LOOK_AROUND"]
                - control_scores["MOVE_TO_SUITABLE_LOCATION"],
                4,
            ),
            0.46,
            places=4,
        )

        # ---- informed: the same world, genuinely different evidence ---------
        brain = self.brain(library=library)
        self.settle(brain, away)
        context_key = brain._initiative_context_key(brain.learning, away)
        self.assertEqual(context_key, "EXPLORE_ROOM|STANDING|ACTIVE|TABLE")
        # 19 verified failures with a negative viewer reaction plus one verified
        # success: a badly failing -- but not repeatedly-failed, so it stays
        # eligible and is scored instead of rejected -- record.
        for index in range(19):
            brain.learning.observe_evaluation(
                SkillEvaluation(
                    skill_id="look_at",
                    execution_id=f"seed-look-{index:02d}",
                    status=SkillStatus.FAILED,
                    duration_seconds=1.0,
                    steps_completed=0,
                    steps_failed=1,
                    viewer_response="negative",
                ),
                context_key=context_key,
            )
        brain.learning.observe_evaluation(
            SkillEvaluation(
                skill_id="look_at",
                execution_id="seed-look-success",
                status=SkillStatus.SUCCESS,
                duration_seconds=1.0,
                steps_completed=1,
                steps_failed=0,
            ),
            context_key=context_key,
        )
        for index in range(10):
            brain.learning.observe_evaluation(
                SkillEvaluation(
                    skill_id="walk_to",
                    execution_id=f"seed-walk-{index:02d}",
                    status=SkillStatus.SUCCESS,
                    duration_seconds=1.0,
                    steps_completed=1,
                    steps_failed=0,
                    viewer_response="positive",
                ),
                context_key=context_key,
            )
        look_record = brain.learning.get_skill_state("look_at")["contexts"][0]
        walk_record = brain.learning.get_skill_state("walk_to")["contexts"][0]
        self.assertEqual(
            (look_record["trials"], look_record["successes"], look_record["failures"]),
            (20, 1, 19),
        )
        self.assertEqual((walk_record["trials"], walk_record["successes"]), (10, 10))
        ranked = brain.learning.rank_candidates(
            context_key, ["look_at", "walk_to"], choose=False
        )
        self.assertEqual([item["skill_id"] for item in ranked], ["walk_to", "look_at"])
        self.assertEqual(ranked[0]["score"], 0.94)
        self.assertEqual(ranked[1]["score"], 0.1865)

        # ---- the ordinary Brain decision flips on that evidence -------------
        brain.tick(away, None)
        self.assertEqual(brain.state.selected_activity, "MOVE_TO_SUITABLE_LOCATION")
        pending = brain.state.pending_action
        self.assertIsNotNone(pending)
        self.assertEqual(pending.action, ActionType.MOVE_TO)
        self.assertEqual(pending.target, "CENTER")
        contributions = dict(brain.state.knowledge_contributions)
        self.assertAlmostEqual(contributions["LOOK_AROUND"], -0.2459, places=4)
        self.assertAlmostEqual(
            contributions["MOVE_TO_SUITABLE_LOCATION"], 0.28, places=4
        )
        scores = dict(brain.state.candidate_scores)
        self.assertAlmostEqual(scores["LOOK_AROUND"], 0.8141, places=4)
        self.assertAlmostEqual(scores["MOVE_TO_SUITABLE_LOCATION"], 0.88, places=4)
        self.assertGreater(scores["MOVE_TO_SUITABLE_LOCATION"], scores["LOOK_AROUND"])

        # The read-model publishes exactly the evidence and scores the ranking
        # used, so the decision stays explainable after the fact.
        read_model = brain.state.initiative_context
        self.assertEqual(read_model["context_key"], context_key)
        evidence = {
            item["skill_id"]: item for item in read_model["knowledge_evidence"]
        }
        self.assertEqual(evidence["look_at"]["ranking_score"], 0.1865)
        self.assertEqual(evidence["walk_to"]["ranking_score"], 0.94)

        # ---- the existing executor owns the real dispatch -------------------
        execution = brain.skill_executor.active
        self.assertIsNotNone(execution)
        self.assertEqual(execution.skill_id, "walk_to")
        self.assertEqual(execution.execution_id, "skill_execution_000001")
        self.assertEqual(execution.status.value, "RUNNING")

        # ---- the observed outcome is journaled under the decision context ---
        arrived = world(character={"position": [0.0, 0.0, 0.0]})
        done = brain.tick(arrived, confirm(brain, success(ActionType.MOVE_TO, "CAMERA_STAND")))
        self.assertEqual(done.status, BrainStatus.IDLE)
        self.assertIsNone(brain.state.last_failure)
        updated = brain.learning.get_skill_state("walk_to")["contexts"][0]
        self.assertEqual(updated["context_key"], context_key)
        self.assertEqual((updated["trials"], updated["successes"]), (11, 11))
        evaluations = [
            evaluation
            for evaluation in brain.skill_executor.evaluations
            if evaluation.execution_id == execution.execution_id
        ]
        self.assertEqual(len(evaluations), 1)
        self.assertEqual(evaluations[0].skill_id, "walk_to")
        self.assertEqual(evaluations[0].status, SkillStatus.SUCCESS)
        self.assertEqual(evaluations[0].steps_completed, 1)
        self.assertIn("character_position_changed", evaluations[0].world_effects)

        # ---- a freshly reopened service reads the outcome back --------------
        reopened = SQLiteSkillLearningRepository(
            brain.learning.repository.database_path
        )
        self._repositories.append(reopened)
        later = SkillLearningService(
            library,
            LearningConfig(),
            reopened,
            clock=self.clock,
            random_source=random.Random(10),
        )
        reloaded = later.get_skill_state("walk_to")["contexts"][0]
        self.assertEqual(reloaded["context_key"], context_key)
        self.assertEqual((reloaded["trials"], reloaded["successes"]), (11, 11))
        later_ranked = later.rank_candidates(
            context_key, ["look_at", "walk_to"], choose=False
        )
        self.assertEqual(later_ranked[0]["skill_id"], "walk_to")
        self.assertEqual(later_ranked[0]["score"], 0.9345)


# ------------------------------------------------------- failure and adaptation


class FailureRecoveryTests(Autonomy27BTestBase):
    def test_failed_results_are_bounded_by_max_retries(self) -> None:
        brain = self.brain()
        self.settle(brain, world())
        brain.tick(world(), None)
        goal = brain.state.current_goal

        attempts = 0
        while brain.state.current_goal is goal and attempts < 5:
            attempts += 1
            self.clock.advance(1)
            brain.tick(world(), confirm(brain, failure(ActionType.MOVE_TO)))
        self.assertEqual(attempts, brain.config.max_retries + 1)
        self.assertEqual(goal.status, GoalStatus.CANCELLED)
        self.assertEqual(
            brain.state.last_outcome,
            "PERFORM_REGISTERED_SKILL:setup_camera_and_dance:failed",
        )
        self.assertEqual(brain.state.replanning_reason, "action_failed")
        self.assertIsNone(brain.state.pending_action)
        self.assertIsNone(brain.skill_executor.active)

    def test_zero_retry_budget_fails_on_the_first_result(self) -> None:
        brain = self.brain(config=BrainConfig(decision_interval_seconds=0.0, max_retries=0))
        self.settle(brain, world())
        brain.tick(world(), None)
        goal = brain.state.current_goal
        self.clock.advance(1)
        brain.tick(world(), confirm(brain, failure(ActionType.MOVE_TO)))
        self.assertEqual(goal.status, GoalStatus.CANCELLED)
        self.assertEqual(
            brain.state.last_outcome,
            "PERFORM_REGISTERED_SKILL:setup_camera_and_dance:failed",
        )

    def test_a_failed_activity_does_not_starve_the_next_decision(self) -> None:
        brain = self.brain()
        self.settle(brain, world())
        brain.tick(world(), None)
        self.clock.advance(1)
        brain.tick(world(), confirm(brain, failure(ActionType.MOVE_TO)))
        self.clock.advance(1)
        brain.tick(world(), confirm(brain, failure(ActionType.MOVE_TO)))
        self.clock.advance(6)
        outcome = brain.tick(world(), None)
        self.assertEqual(outcome.status, BrainStatus.EXECUTING)
        self.assertEqual(brain.state.selected_activity, "PERFORM_REGISTERED_SKILL:dance")

    def test_terminal_failure_is_recorded_as_verified_knowledge(self) -> None:
        """A Unity-observed FAILED result reaches the existing learning service.

        Before the Phase 27B gap closure the generic ``FAILED`` branch released
        the plan without ever handing the executor's failure evaluation to
        ``SkillLearningService``, so a verified failure could never be read
        back through the evaluator's ``RECENT_FAILURE`` rule. The retry that is
        still owed must not write a record; only the terminal failure does.
        """
        brain = self.brain()
        self.settle(brain, world())
        brain.tick(world(), None)
        decision_key = brain._initiative_context_key(brain.learning, world())
        self.assertTrue(decision_key)
        self.assertEqual(
            brain.state.selected_activity,
            "PERFORM_REGISTERED_SKILL:setup_camera_and_dance",
        )
        self.assertEqual(
            brain.learning.get_skill_state("setup_camera_and_dance")["contexts"], []
        )

        # First failure: the bounded retry is still owed, so no record yet.
        self.clock.advance(1)
        brain.tick(world(), confirm(brain, failure(ActionType.MOVE_TO, "unity_move_to_returned_false")))
        self.assertEqual(
            brain.learning.get_skill_state("setup_camera_and_dance")["contexts"], []
        )

        # Second failure: terminal, so the real evaluation is persisted.
        self.clock.advance(1)
        brain.tick(world(), confirm(brain, failure(ActionType.MOVE_TO, "unity_move_to_returned_false")))
        records = brain.learning.get_skill_state("setup_camera_and_dance")["contexts"]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["context_key"], decision_key)
        self.assertEqual(records[0]["trials"], 1)
        self.assertEqual(records[0]["successes"], 0)
        self.assertEqual(records[0]["failures"], 1)
        self.assertIsNone(brain.skill_executor.active)

    def test_skill_timeout_is_classified_as_timed_out(self) -> None:
        brain = self.brain()
        self.settle(brain, world())
        brain.tick(world(), None)
        goal = brain.state.current_goal
        self.assertIsNotNone(brain.state.pending_action)

        self.clock.advance(121)
        brain.tick(world(), None)
        self.assertEqual(goal.status, GoalStatus.CANCELLED)
        self.assertEqual(
            brain.state.last_outcome,
            "PERFORM_REGISTERED_SKILL:setup_camera_and_dance:timed_out",
        )
        self.assertEqual(brain.state.interruption_reason, "SKILL_TIMEOUT")
        # The timed-out execution is cancelled; a replanned successor may
        # already own the executor in this same tick.
        active = brain.skill_executor.active
        if active is not None:
            self.assertNotEqual(
                active.execution_id, "skill_execution_000001"
            )
            self.assertEqual(active.plan_id, brain.state.current_plan.plan_id)

    def test_lost_camera_invalidates_the_active_goal_and_replans(self) -> None:
        brain = self.brain()
        self.settle(brain, world())
        brain.tick(world(), None)
        goal = brain.state.current_goal

        self.clock.advance(1)
        out = brain.tick(
            world(stream={"camera_available": False}),
            confirm(brain, success(ActionType.MOVE_TO, "CAMERA_STAND")),
        )
        self.assertEqual(goal.status, GoalStatus.CANCELLED)
        self.assertEqual(
            brain.state.last_outcome,
            "PERFORM_REGISTERED_SKILL:setup_camera_and_dance:invalidated",
        )
        self.assertEqual(brain.state.interruption_reason, "CAMERA_INACTIVE")
        self.assertIsNone(brain.state.pending_action)
        # The stale premise is never resumed: this same tick re-decides.
        self.assertIn(out.status, {BrainStatus.WAITING, BrainStatus.EXECUTING})
        self.assertNotEqual(brain.state.current_goal, goal)

    def test_stream_stop_invalidates_and_hands_the_tick_to_maintenance(self) -> None:
        brain = self.brain()
        self.settle(brain, world())
        brain.tick(world(), None)
        goal = brain.state.current_goal

        self.clock.advance(1)
        outcome = brain.tick(
            world(stream={"is_streaming": False}),
            confirm(brain, success(ActionType.MOVE_TO, "CAMERA_STAND")),
        )
        self.assertEqual(goal.status, GoalStatus.CANCELLED)
        self.assertEqual(
            brain.state.last_outcome,
            "PERFORM_REGISTERED_SKILL:setup_camera_and_dance:invalidated",
        )
        self.assertEqual(brain.state.interruption_reason, "STREAM_STATE_INCOMPATIBLE")
        self.assertIn(outcome.status, {BrainStatus.EXECUTING, BrainStatus.ERROR})
        # Autonomy released the goal; the legacy maintenance path owns the
        # tick and nothing autonomous resumes on the dead stream.
        self.assertFalse(brain._is_initiative_goal(brain.state.current_goal))

    def test_viewer_interruption_is_observed_as_interrupted(self) -> None:
        brain = self.brain()
        self.settle(brain, world())
        brain.tick(world(), None)
        goal = brain.state.current_goal

        request = brain._chat.submit("viewer_01", "Lacrimosa, can you dance?")
        self.assertTrue(request.accepted)
        self.assertGreaterEqual(request.message.priority, brain.config.interruption_priority)
        self.assertEqual(goal.status, GoalStatus.CANCELLED)
        self.assertEqual(
            brain.state.last_outcome,
            "PERFORM_REGISTERED_SKILL:setup_camera_and_dance:interrupted",
        )
        self.assertEqual(brain.state.replanning_reason, "interrupted_by_viewer")
        self.assertIsNone(brain.state.pending_action)
        self.assertIsNone(brain.skill_executor.active)

        self.clock.advance(1)
        answer = brain.tick(world(), None)
        self.assertEqual(answer.status, BrainStatus.EXECUTING)
        self.assertEqual(brain.state.current_goal.goal_type, GoalType.RESPOND_TO_VIEWER)
        self.assertEqual(brain.state.decision_source, "brain:evaluator")
        self.assertFalse(brain._is_initiative_goal(brain.state.current_goal))

    def test_a_goal_that_cannot_be_planned_fails_without_wedging(self) -> None:
        brain = self.brain()
        self.settle(brain, world())
        brain.skill_executor.library.disable("dance")
        brain.skill_executor.library.disable("wave")
        brain.skill_executor.library.disable("look_at")
        brain.skill_executor.library.disable("walk_to")
        outcome = brain.tick(world(), None)
        self.assertEqual(outcome.status, BrainStatus.WAITING)
        self.assertEqual(outcome.message, "autonomy_wait")
        self.assertEqual(outcome.goal.status, GoalStatus.FAILED)
        self.assertEqual(
            brain.state.last_outcome,
            "PERFORM_REGISTERED_SKILL:setup_camera_and_dance:failed",
        )
        self.assertIsNone(brain.state.current_goal)
        self.assertIsNone(brain.state.pending_action)


# ------------------------------------------------------ continuity and waiting


class GoalContinuityTests(Autonomy27BTestBase):
    def test_an_active_goal_is_not_superseded_by_a_new_candidate(self) -> None:
        brain = self.brain(library=library_with("look_at"))
        self.settle(brain, world())
        brain.tick(world(), None)
        goal = brain.state.current_goal
        self.assertEqual(brain.state.selected_activity, "LOOK_AROUND")

        # A higher-weighted activity appears mid-flight. The valid goal keeps
        # running: nothing here may rebuild the activity on every tick.
        brain.skill_executor.library.register(
            next(skill for skill in build_builtin_skills() if skill.skill_id == "dance")
        )
        self.clock.advance(1)
        continuing = brain.tick(world(), None)
        self.assertEqual(continuing.status, BrainStatus.EXECUTING)
        self.assertIs(brain.state.current_goal, goal)
        self.assertEqual(brain.state.selected_activity, "LOOK_AROUND")
        self.assertEqual(brain.state.pending_action.action, ActionType.LOOK_AT)

    def test_terminal_goal_is_released_and_never_resumed(self) -> None:
        brain = self.brain(library=library_with("look_at"))
        self.settle(brain, world())
        brain.tick(world(), None)
        goal = brain.state.current_goal
        self.clock.advance(1)
        done = brain.tick(world(), confirm(brain, success(ActionType.LOOK_AT, "CAMERA")))
        self.assertEqual(done.status, BrainStatus.IDLE)
        self.assertEqual(goal.status, GoalStatus.COMPLETED)
        self.assertIsNone(brain.state.current_goal)

        self.clock.advance(6)
        suppressed = brain.tick(world(), None)
        self.assertEqual(suppressed.status, BrainStatus.WAITING)
        self.assertEqual(brain.state.selected_activity, INITIATIVE_WAIT)
        self.assertIn("repetition_suppressed", brain.state.initiative_reason)
        self.assertNotEqual(brain.state.current_goal.goal_id, goal.goal_id)

    def test_wait_goal_is_explicit_and_paced_by_the_initiative_interval(self) -> None:
        brain = self.brain(library=SkillLibrary())
        self.settle(brain, world())
        outcome = brain.tick(world(), None)
        self.assertEqual(outcome.status, BrainStatus.WAITING)
        self.assertEqual(outcome.message, "autonomy_wait")
        self.assertEqual(brain.state.selected_activity, INITIATIVE_WAIT)
        self.assertEqual(brain.state.decision_source, "brain:initiative")
        self.assertTrue(brain.state.current_goal.context["initiative"])

    def test_viewer_work_keeps_priority_over_a_new_initiative(self) -> None:
        brain = self.brain()
        self.settle(brain, world())
        brain._chat.submit("viewer_01", "Lacrimosa, can you dance?")
        outcome = brain.tick(world(), None)
        self.assertEqual(outcome.status, BrainStatus.EXECUTING)
        self.assertEqual(brain.state.current_goal.goal_type, GoalType.RESPOND_TO_VIEWER)
        self.assertFalse(brain._is_initiative_goal(brain.state.current_goal))
        self.assertIsNone(brain.state.selected_activity)


# ------------------------------------------------------------- diagnostics


class DiagnosticsTests(Autonomy27BTestBase):
    def test_autonomy_state_exposes_bounded_decision_evidence(self) -> None:
        brain = self.brain(streaming=StreamingStub("LIVE"))
        service = BrainService(brain, brain._chat, clock=lambda: 0.0)
        self.settle(brain, world())
        brain.tick(world(), None)
        payload = service.get_autonomy_state()
        self.assertEqual(payload["decision_source"], "brain:initiative")
        self.assertEqual(payload["next_decision_eligibility"], "action_in_flight")
        self.assertEqual(
            payload["candidate_activity"],
            "PERFORM_REGISTERED_SKILL:setup_camera_and_dance",
        )
        self.assertLessEqual(
            len(payload["eligible_candidates"]), MAX_INITIATIVE_CANDIDATES_IN_STATE
        )
        self.assertLessEqual(
            len(payload["candidate_rejections"]), MAX_INITIATIVE_CANDIDATES_IN_STATE
        )
        context = payload["initiative_context"]
        self.assertEqual(context["stream_lifecycle"], "live")
        self.assertTrue(context["camera_available"])
        self.assertEqual(context["phone_state"], "Placed")
        self.assertTrue(context["behavior_state"])
        self.assertEqual(context["knowledge_skills"], [])
        self.assertIsNone(payload["replanning_reason"])

    def test_diagnostics_report_verified_knowledge_and_rejections(self) -> None:
        brain = self.brain(library=library_with("dance", "wave", "look_at"))
        service = BrainService(brain, brain._chat, clock=lambda: 0.0)
        self.settle(brain, world())
        context_key = brain._initiative_context_key(brain.learning, world())
        for index in range(2):
            brain.learning.observe_evaluation(
                SkillEvaluation(
                    skill_id="dance",
                    execution_id=f"seed-{index}",
                    status=SkillStatus.FAILED,
                    duration_seconds=1.0,
                    steps_completed=0,
                    steps_failed=1,
                ),
                context_key=context_key,
            )
        brain.tick(world(), None)
        payload = service.get_autonomy_state()
        self.assertEqual(payload["candidate_activity"], "PERFORM_REGISTERED_SKILL:wave")
        self.assertEqual(payload["initiative_context"]["knowledge_skills"], ["dance"])
        self.assertIn(
            {"activity": "PERFORM_REGISTERED_SKILL:dance", "reason": "RECENT_FAILURE"},
            payload["candidate_rejections"],
        )

    def test_candidate_scores_expose_the_knowledge_bonus_the_ranking_used(self) -> None:
        """The live decision reports the score each candidate was ranked by.

        A candidate whose existing ``SkillKnowledge`` record is verified must
        score strictly above its documented static weight, while a candidate
        with no record scores exactly its weight. That is the diagnostics-level
        proof that the persisted learning evidence was read into the real
        ranking instead of merely being claimed by the report.
        """
        brain = self.brain(library=library_with("dance", "look_at"))
        service = BrainService(brain, brain._chat, clock=lambda: 0.0)
        self.settle(brain, world())
        context_key = brain._initiative_context_key(brain.learning, world())
        self.assertTrue(context_key)
        for index in range(3):
            brain.learning.observe_evaluation(
                SkillEvaluation(
                    skill_id="look_at",
                    execution_id=f"look-{index}",
                    status=SkillStatus.SUCCESS,
                    duration_seconds=1.0,
                    steps_completed=1,
                    steps_failed=0,
                ),
                context_key=context_key,
            )
        brain.tick(world(), None)

        payload = service.get_autonomy_state()
        self.assertEqual(payload["decision_source"], DECISION_SOURCE_INITIATIVE)
        self.assertEqual(payload["initiative_context"]["context_key"], context_key)
        self.assertEqual(payload["initiative_context"]["knowledge_skills"], ["look_at"])
        scores = {
            item["activity"]: item["score"] for item in payload["candidate_scores"]
        }
        # The ranking order and the reported scores describe the same list.
        self.assertEqual(
            [item["activity"] for item in payload["candidate_scores"]],
            payload["eligible_candidates"],
        )
        self.assertGreater(scores["LOOK_AROUND"], 1.0)
        self.assertEqual(
            scores["PERFORM_REGISTERED_SKILL:dance"], 3.0
        )
        self.assertEqual(payload["candidate_activity"], "PERFORM_REGISTERED_SKILL:dance")

    def test_injected_runtime_state_is_reported_as_external(self) -> None:
        brain = self.brain(with_learning=False)
        service = BrainService(brain, brain._chat, clock=lambda: 0.0)
        goal = Goal(
            goal_id="injected-goal",
            goal_type=GoalType.EXPLORE_ROOM,
            priority=0.3,
            reason="harness injected",
        )
        plan = Plan(
            plan_id="injected-plan",
            goal_id=goal.goal_id,
            reason="harness injected",
            steps=(PlanStep(action=ActionType.WAVE),),
        )
        brain.state.current_goal = goal
        brain.state.current_plan = plan
        brain.state.pending_action = ActionRequest(action=ActionType.WAVE)
        brain.state.selected_activity = "PERFORM_REGISTERED_SKILL:wave"
        brain._finalize_tick_diagnostics()

        payload = service.get_autonomy_state()
        self.assertEqual(payload["decision_source"], "external")
        self.assertEqual(payload["next_decision_eligibility"], "action_in_flight")
        # An injected goal is never reported as an autonomous one.
        self.assertIsNone(payload["current_goal"])

    def test_candidate_evidence_records_stay_bounded(self) -> None:
        brain = self.brain(with_learning=False)
        decision = InitiativeDecision(
            activity="A",
            reason="synthetic",
            eligible=tuple(f"activity_{index}" for index in range(10)),
            rejections=tuple(
                (f"activity_{index}", "SKILL_UNAVAILABLE") for index in range(10)
            ),
            scores=tuple(
                (f"activity_{index}", float(index)) for index in range(10)
            ),
        )
        brain._record_initiative_evidence(decision)
        self.assertEqual(
            len(brain.state.eligible_candidates), MAX_INITIATIVE_CANDIDATES_IN_STATE
        )
        self.assertEqual(
            len(brain.state.candidate_rejections), MAX_INITIATIVE_CANDIDATES_IN_STATE
        )
        self.assertEqual(
            len(brain.state.candidate_scores), MAX_INITIATIVE_CANDIDATES_IN_STATE
        )

    def test_last_activity_tracks_the_bounded_journal(self) -> None:
        brain = self.brain(library=library_with("look_at"))
        service = BrainService(brain, brain._chat, clock=lambda: 0.0)
        self.settle(brain, world())
        brain.tick(world(), None)
        self.clock.advance(1)
        brain.tick(world(), confirm(brain, success(ActionType.LOOK_AT, "CAMERA")))
        payload = service.get_autonomy_state()
        self.assertEqual(payload["last_activity"], "LOOK_AROUND")
        self.assertEqual(payload["last_outcome"], "LOOK_AROUND:completed")
        self.assertIsNone(payload["decision_source"])
        self.assertLessEqual(len(brain.state.recent_autonomy_activities), 4)

    def test_evidence_is_released_once_the_world_leaves_autonomy_scope(self) -> None:
        """A stopped stream stops claiming the last decision-time context.

        The recorded evidence describes the context of the *live* decision it
        came from. Once the operator deliberately stops the stream, autonomy is
        out of scope for that tick, and the stale ``live`` context must not keep
        being reported as if this tick had judged anything.
        """
        brain = self.brain(
            library=library_with("look_at"), streaming=StreamingStub("LIVE")
        )
        service = BrainService(brain, brain._chat, clock=lambda: 0.0)
        self.settle(brain, world())
        brain.tick(world(), None)
        self.assertEqual(brain.state.initiative_context["stream_lifecycle"], "live")
        self.clock.advance(1)
        brain.tick(world(), confirm(brain, success(ActionType.LOOK_AT, "CAMERA")))
        # A verified completion keeps the decision explainable.
        self.assertIsNotNone(brain.state.initiative_context)
        self.assertIsNone(brain.state.decision_source)

        # An operator stop, written through the same latch the control plane
        # uses: the next idle tick reaches the gate and refuses it as out of
        # scope instead of repeating the stale context.
        brain.social.context.set_stream_stop_latch(True, source="test:operator_stop")
        brain.streaming._state = "STOPPED"
        self.clock.advance(1)
        brain.tick(world(stream={"is_streaming": False, "camera_active": False}), None)

        payload = service.get_autonomy_state()
        self.assertIsNone(brain.state.initiative_context)
        self.assertIsNone(payload["initiative_context"])
        self.assertEqual(brain.state.eligible_candidates, ())
        self.assertIsNone(brain.state.candidate_activity)
        self.assertEqual(brain.state.candidate_rejections, ())
        self.assertNotEqual(payload["decision_source"], DECISION_SOURCE_INITIATIVE)
        # The last real decision and its verified end survive as history.
        self.assertEqual(payload["selected_activity"], "LOOK_AROUND")
        self.assertEqual(payload["last_activity"], "LOOK_AROUND")
        self.assertEqual(payload["last_outcome"], "LOOK_AROUND:completed")

    def test_transient_refusal_keeps_the_decision_evidence(self) -> None:
        """A refusal about *this* tick keeps the last decision explainable.

        Losing the audience is a scope change; a world action still in progress
        is not. The world still supports autonomous activity, so the decision
        evidence stays for the next eligible decision instead of being dropped.
        """
        brain = self.brain(
            library=library_with("look_at"), streaming=StreamingStub("LIVE")
        )
        self.settle(brain, world())
        brain.tick(world(), None)
        context = dict(brain.state.initiative_context)
        candidate = brain.state.candidate_activity
        self.assertEqual(candidate, "LOOK_AROUND")
        self.clock.advance(1)
        brain.tick(world(), confirm(brain, success(ActionType.LOOK_AT, "CAMERA")))
        # Nothing is in flight and no decision is paced away, so the next tick
        # definitely reaches the initiative gate.
        self.assertIsNone(brain.state.pending_action)
        self.assertIsNone(brain.state.current_plan)
        self.assertIsNone(brain.state.decision_source)

        self.clock.advance(1)
        brain.tick(world(character={"current_action": "walking"}), None)
        eligible, reason = brain._initiative_eligibility(None)
        self.assertFalse(eligible)
        self.assertEqual(reason, "action_in_progress:walking")
        # The legacy evaluator owned this tick, which proves the gate ran.
        self.assertIs(brain.state.current_goal.goal_type, GoalType.IDLE)
        self.assertEqual(brain.state.initiative_context, context)
        self.assertEqual(brain.state.candidate_activity, candidate)
        self.assertNotEqual(brain.state.decision_source, DECISION_SOURCE_INITIATIVE)


# --------------------------------------------------------- unity/recovery edges


class UnityRecoveryTests(Autonomy27BTestBase):
    def test_unexpected_result_is_ignored_and_the_expected_one_advances(self) -> None:
        brain = self.brain()
        self.settle(brain, world())
        brain.tick(world(), None)
        goal = brain.state.current_goal
        plan_id = brain.state.current_plan.plan_id

        self.clock.advance(1)
        brain.tick(world(), confirm(brain, success(ActionType.WAVE)))
        self.assertIs(brain.state.current_goal, goal)
        self.assertEqual(brain.state.current_plan.plan_id, plan_id)
        self.assertEqual(brain.state.step_index, 0)
        self.assertEqual(brain.state.pending_action.action, ActionType.MOVE_TO)

        self.clock.advance(1)
        brain.tick(world(), confirm(brain, success(ActionType.MOVE_TO, "CAMERA_STAND")))
        self.assertIs(brain.state.current_goal, goal)
        self.assertEqual(brain.state.step_index, 1)
        self.assertEqual(brain.state.pending_action.action, ActionType.LOOK_AT)

    def test_result_after_interruption_does_not_advance_the_viewer_plan(self) -> None:
        brain = self.brain()
        self.settle(brain, world())
        brain.tick(world(), None)
        brain._chat.submit("viewer_01", "Lacrimosa, can you dance?")
        self.clock.advance(1)
        brain.tick(world(), None)
        viewer_goal = brain.state.current_goal
        self.assertEqual(viewer_goal.goal_type, GoalType.RESPOND_TO_VIEWER)
        plan_id = brain.state.current_plan.plan_id

        # A leftover result from the cancelled autonomous plan is not the
        # action the new plan awaits.
        self.clock.advance(1)
        brain.tick(world(), confirm(brain, success(ActionType.WAVE)))
        self.assertIs(brain.state.current_goal, viewer_goal)
        self.assertEqual(brain.state.current_plan.plan_id, plan_id)
        self.assertEqual(brain.state.step_index, 0)

    def test_state_is_stable_when_the_executor_is_gone(self) -> None:
        brain = self.brain()
        self.settle(brain, world())
        brain.skill_executor = None
        outcome = brain.tick(world(), None)
        self.assertNotEqual(outcome.message, "autonomy_wait")
        self.assertIsNone(brain.state.selected_activity)
        self.assertIsNone(brain.state.initiative_reason)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
