"""PHASE 27A focused tests: autonomy foundation / goal continuity.

These pin the contract this phase added on top of the existing Brain:

* the single Brain loop may fill an *idle* tick (the legacy evaluator had
  nothing to do) with one deterministic autonomous activity, chosen from
  registered skills the existing ``SkillExecutor`` reports as available;
* an autonomous goal persists across ticks -- it is continued, observed and
  completed/interrupted exactly like any other goal, never rebuilt every tick;
* the existing viewer priority/cancellation machinery still outranks every
  initiative, and no action is ever dispatched outside the production path
  (Brain -> ActionRequest -> SkillExecutor -> pending action -> observation);
* basic repetition protection stops the same finished activity from being
  re-selected against an unchanged world, while a changed world releases it;
* diagnostics explain the current goal, its status and why it was chosen.
"""

from __future__ import annotations

import unittest
from typing import Any

from backend.brain.goals import (
    INITIATIVE_WAIT,
    GoalEvaluator,
    GoalEvaluatorConfig,
    initiative_signature,
)
from backend.brain.loop import AutonomousBrain, BrainConfig, BrainStatus
from backend.brain.models import (
    ActionResult,
    ActionType,
    ExecutionStatus,
    GoalStatus,
    GoalType,
)
from backend.brain.personality import load_personality
from backend.brain.service import BrainService
from backend.brain.world_state import build_world_state
from backend.chat.service import ChatService, ChatServiceConfig
from backend.llm.mock import MockLLMProvider
from backend.skills.builtins import build_builtin_skills
from backend.skills.executor import SkillPlanResult
from backend.skills.library import SkillLibrary, create_default_skill_library
from backend.social.service import SocialAutonomyService

from backend.tests import test_behavior as behavior
from backend.tests.dispatch_support import confirm


def snapshot(**overrides: Any) -> dict[str, Any]:
    """An idle, observed room: streaming, one viewer, standing at CENTER."""
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
            # Nested overrides merge, so a test only names the fields the
            # scenario actually moves. ``targets`` is the exception: it always
            # replaces the registry, so a scenario can remove a target.
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


def library_with(*skill_ids: str) -> SkillLibrary:
    """A real library containing exactly the named registered skills."""
    wanted = {str(skill_id).strip().lower() for skill_id in skill_ids}
    library = SkillLibrary()
    for skill in build_builtin_skills():
        if skill.skill_id in wanted:
            library.register(skill)
    return library


class AutonomyTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = behavior.FakeClock()

    def brain(
        self,
        *,
        library: SkillLibrary | None = None,
        provider=None,
        config: BrainConfig | None = None,
        speech=None,
    ) -> AutonomousBrain:
        chat = ChatService(
            MockLLMProvider(),
            ChatServiceConfig(max_messages_per_rate_window=20),
            clock=self.clock,
        )
        instance = AutonomousBrain(
            provider or behavior.CapturingProvider(),
            chat,
            config
            or BrainConfig(decision_interval_seconds=0.0, initiative_interval_seconds=5.0),
            personality=load_personality(),
            clock=self.clock,
            skill_library=library if library is not None else create_default_skill_library(),
            speech_service=speech,
            social_service=SocialAutonomyService(clock=self.clock),
        )
        # The legacy explore cadence stays out of the way so the tests observe
        # the idle gap autonomy exists for; it is still consumed for real by
        # one legacy explore tick in ``settle``.
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


class EligibilityTests(AutonomyTestBase):
    def test_eligibility_matrix_reports_concrete_reasons(self) -> None:
        brain = self.brain()
        self.settle(brain, world())
        self.assertEqual(brain._initiative_eligibility(None), (True, "eligible"))

        brain.tick(world(stream={"is_streaming": False}), None)
        self.assertEqual(
            brain._initiative_eligibility(None), (False, "no_audience")
        )

        brain = self.brain()
        brain.tick(world(stream={"viewer_count": 0}), None)
        self.assertEqual(
            brain._initiative_eligibility(None), (False, "no_audience")
        )

        brain = self.brain()
        brain.tick(world(character={"current_action": "walking"}), None)
        self.assertEqual(
            brain._initiative_eligibility(None),
            (False, "action_in_progress:walking"),
        )

    def test_disabled_autonomy_leaves_the_legacy_path_authoritative(self) -> None:
        brain = self.brain(config=BrainConfig(decision_interval_seconds=0.0, autonomy_enabled=False))
        brain.tick(world(), None)
        brain.tick(world(), confirm(brain, ActionResult(ActionType.MOVE_TO, ExecutionStatus.SUCCESS, "reported")))
        outcome = brain.tick(world(), None)
        self.assertNotEqual(outcome.message, "autonomy_wait")
        self.assertIsNone(brain.state.selected_activity)
        self.assertIsNone(brain.state.initiative_reason)

    def test_missing_skill_library_never_enters_autonomy(self) -> None:
        brain = self.brain(library=SkillLibrary())
        self.assertIsNotNone(brain.skill_executor)
        brain.skill_executor = None
        brain.tick(world(), None)
        brain.tick(world(), confirm(brain, ActionResult(ActionType.MOVE_TO, ExecutionStatus.SUCCESS, "reported")))
        outcome = brain.tick(world(), None)
        self.assertNotEqual(outcome.message, "autonomy_wait")
        self.assertIsNone(brain.state.initiative_reason)

    def test_active_speech_blocks_initiative(self) -> None:
        class BusySpeech:
            enabled = True

            def get_state(self) -> dict[str, Any]:
                return {"state": "SPEAKING", "queue_length": 0}

        brain = self.brain(speech=BusySpeech())
        brain.tick(world(), None)
        brain.tick(world(), confirm(brain, ActionResult(ActionType.MOVE_TO, ExecutionStatus.SUCCESS, "reported")))
        outcome = brain.tick(world(), None)
        self.assertNotEqual(outcome.message, "autonomy_wait")
        self.assertIsNone(brain.state.selected_activity)

    def test_pending_viewer_message_keeps_viewer_priority(self) -> None:
        brain = self.brain()
        brain._chat.submit("viewer_01", "Hi Lacrimosa!")
        outcome = brain.tick(world(), None)
        self.assertIsNotNone(outcome.goal)
        self.assertIsNone(brain.state.selected_activity)
        self.assertFalse(brain._is_initiative_goal(brain.state.current_goal))


class WaitTests(AutonomyTestBase):
    def test_no_available_candidate_records_an_explicit_wait(self) -> None:
        brain = self.brain(library=SkillLibrary())
        self.settle(brain, world())

        outcome = brain.tick(world(), None)
        self.assertEqual(outcome.status, BrainStatus.WAITING)
        self.assertEqual(outcome.message, "autonomy_wait")
        goal = brain.state.current_goal
        self.assertIsNotNone(goal)
        self.assertEqual(goal.context["activity"], INITIATIVE_WAIT)
        self.assertEqual(goal.status, GoalStatus.ACTIVE)
        self.assertIn("unavailable:", goal.reason)
        self.assertEqual(brain.state.selected_activity, INITIATIVE_WAIT)
        self.assertEqual(brain.state.last_failure, None)

    def test_wait_goal_persists_until_the_world_really_changes(self) -> None:
        brain = self.brain(library=SkillLibrary())
        self.settle(brain, world())
        brain.tick(world(), None)
        first = brain.state.current_goal
        self.assertIsNotNone(first)

        self.clock.advance(3)
        second = brain.tick(world(), None)
        self.assertEqual(second.status, BrainStatus.WAITING)
        self.assertIs(brain.state.current_goal, first)
        self.assertEqual(brain.state.current_goal.goal_id, first.goal_id)

        # A world that now offers a registered activity supersedes the wait.
        brain.skill_executor.library.register(
            next(skill for skill in build_builtin_skills() if skill.skill_id == "look_at")
        )
        third = brain.tick(world(), None)
        self.assertEqual(third.status, BrainStatus.EXECUTING)
        self.assertEqual(first.status, GoalStatus.CANCELLED)
        self.assertEqual(brain.state.selected_activity, "LOOK_AROUND")
        self.assertNotEqual(brain.state.current_goal.goal_id, first.goal_id)


class SelectionTests(AutonomyTestBase):
    def test_default_library_selects_the_composite_through_the_production_path(self) -> None:
        brain = self.brain()
        self.settle(brain, world())

        outcome = brain.tick(world(), None)
        self.assertEqual(outcome.status, BrainStatus.EXECUTING)
        self.assertEqual(
            brain.state.selected_activity, "PERFORM_REGISTERED_SKILL:setup_camera_and_dance"
        )
        self.assertEqual(brain.state.current_goal.context["skill_id"], "setup_camera_and_dance")
        self.assertEqual(brain.state.pending_action.action, ActionType.MOVE_TO)
        self.assertEqual(brain.state.pending_action.target, "CAMERA_STAND")
        self.assertEqual(brain._plan_skill_id, "setup_camera_and_dance")
        self.assertIsNotNone(brain.skill_executor.active)
        self.assertEqual(brain.skill_executor.active.skill_id, "setup_camera_and_dance")

    def test_disabled_candidate_is_filtered_and_the_next_one_wins(self) -> None:
        brain = self.brain()
        brain.skill_executor.library.disable("setup_camera_and_dance")
        self.settle(brain, world())

        brain.tick(world(), None)
        self.assertEqual(brain.state.selected_activity, "PERFORM_REGISTERED_SKILL:dance")
        self.assertEqual(brain.state.pending_action.action, ActionType.DANCE)

    def test_gaze_candidate_is_selected_when_motion_is_unavailable(self) -> None:
        brain = self.brain()
        for skill_id in ("setup_camera_and_dance", "dance", "wave"):
            brain.skill_executor.library.disable(skill_id)
        self.settle(brain, world())

        brain.tick(world(), None)
        self.assertEqual(brain.state.selected_activity, "LOOK_AROUND")
        self.assertEqual(brain.state.pending_action.action, ActionType.LOOK_AT)
        self.assertEqual(brain.state.pending_action.target, "CAMERA")

    def test_movement_candidates_appear_only_when_the_destination_is_away(self) -> None:
        brain = self.brain()
        for skill_id in ("setup_camera_and_dance", "dance", "wave", "look_at"):
            brain.skill_executor.library.disable(skill_id)
        self.settle(brain, world())

        # Inside the camera frame (3 m from CAMERA) but 5 m from the framing
        # anchor: the legacy evaluator has nothing to say about this world once
        # its explore cadence is spent, so only initiative movement is left.
        far = world(
            character={
                "position": [5, 0, 0],
            }
        )

        outcome = brain.tick(far, None)
        self.assertEqual(outcome.status, BrainStatus.EXECUTING)
        self.assertEqual(brain.state.selected_activity, "MOVE_TO_SUITABLE_LOCATION")
        self.assertEqual(brain.state.pending_action.target, "CENTER")

    def test_unreachable_candidate_is_reported_and_never_fabricated(self) -> None:
        brain = self.brain()
        for skill_id in ("setup_camera_and_dance", "dance", "wave", "look_at"):
            brain.skill_executor.library.disable(skill_id)
        self.settle(brain, world())

        unreachable = world(
            character={
                "position": [5, 0, 0],
            },
            targets={
                "CENTER": {
                    "position": [0, 0, 0],
                    "navigation_target": False,
                    "look_at_target": True,
                },
                "CAMERA": {
                    "position": [2, 0.8, 0],
                    "navigation_target": False,
                    "look_at_target": True,
                },
            },
        )

        outcome = brain.tick(unreachable, None)
        self.assertEqual(outcome.status, BrainStatus.WAITING)
        self.assertIn("TARGET_UNREACHABLE", brain.state.initiative_reason)
        self.assertEqual(brain.state.selected_activity, INITIATIVE_WAIT)

    def test_selector_is_deterministic_for_the_same_inputs(self) -> None:
        evaluator = GoalEvaluator(load_personality(), config=GoalEvaluatorConfig(explore_interval_seconds=3600.0))
        world_value = world()
        first = evaluator.evaluate_initiative(world_value, availability=lambda *_: None)
        second = evaluator.evaluate_initiative(world_value, availability=lambda *_: None)
        self.assertEqual(first, second)
        self.assertEqual(first.activity, "PERFORM_REGISTERED_SKILL:setup_camera_and_dance")
        self.assertEqual(first.signature, initiative_signature(world_value))

    def test_cooldown_filters_the_candidate_before_the_executor_sees_it(self) -> None:
        evaluator = GoalEvaluator(load_personality(), config=GoalEvaluatorConfig(explore_interval_seconds=3600.0))

        def availability(skill_id: str, _parameters) -> Any:
            if skill_id == "setup_camera_and_dance":
                return "SKILL_ON_COOLDOWN"
            return None

        decision = evaluator.evaluate_initiative(world(), availability=availability)
        self.assertEqual(decision.skill_id, "dance")
        self.assertEqual(decision.activity, "PERFORM_REGISTERED_SKILL:dance")

    def test_missing_skill_never_reaches_the_executor(self) -> None:
        evaluator = GoalEvaluator(load_personality(), config=GoalEvaluatorConfig(explore_interval_seconds=3600.0))
        decision = evaluator.evaluate_initiative(
            world(), availability=lambda *_: "SKILL_NOT_FOUND"
        )
        self.assertTrue(decision.is_wait)
        self.assertIn("SKILL_NOT_FOUND", decision.reason)


class GoalContinuityTests(AutonomyTestBase):
    def test_goal_persists_across_ticks_and_completes_once(self) -> None:
        brain = self.brain()
        self.settle(brain, world())
        started = brain.tick(world(), None)
        goal = brain.state.current_goal
        plan_id = brain.state.current_plan.plan_id
        self.assertEqual(started.status, BrainStatus.EXECUTING)

        self.clock.advance(1)
        middle = brain.tick(world(), confirm(brain, success(ActionType.MOVE_TO, "CAMERA_STAND")))
        self.assertEqual(middle.status, BrainStatus.EXECUTING)
        # Same goal, same plan: the next step of the *same* activity runs.
        self.assertIs(brain.state.current_goal, goal)
        self.assertEqual(brain.state.current_plan.plan_id, plan_id)
        self.assertEqual(brain.state.pending_action.action, ActionType.LOOK_AT)
        self.assertEqual(
            brain.state.selected_activity, "PERFORM_REGISTERED_SKILL:setup_camera_and_dance"
        )

        self.clock.advance(1)
        dance = brain.tick(world(), confirm(brain, success(ActionType.LOOK_AT, "CAMERA")))
        self.assertEqual(dance.status, BrainStatus.EXECUTING)
        self.assertIs(brain.state.current_goal, goal)
        self.assertEqual(brain.state.pending_action.action, ActionType.DANCE)

        self.clock.advance(1)
        done = brain.tick(world(), confirm(brain, success(ActionType.DANCE)))
        self.assertEqual(done.status, BrainStatus.IDLE)
        self.assertEqual(done.message, "plan_complete")
        self.assertEqual(goal.status, GoalStatus.COMPLETED)
        self.assertIsNone(brain.state.current_goal)
        self.assertEqual(
            brain.state.last_outcome, "PERFORM_REGISTERED_SKILL:setup_camera_and_dance:completed"
        )
        self.assertIsNone(brain.skill_executor.active)
        self.assertEqual(brain.skill_executor.evaluations[-1].status.value, "SUCCESS")

    def test_rejected_action_cancels_the_goal_and_returns_to_a_decision_state(self) -> None:
        brain = self.brain()
        self.settle(brain, world())
        brain.tick(world(), None)
        goal = brain.state.current_goal

        self.clock.advance(1)
        rejected = brain.tick(
            world(), confirm(brain, ActionResult(ActionType.MOVE_TO, ExecutionStatus.REJECTED, "rejected_by_unity"))
        )
        # The cancelled activity hands the tick straight back to the single
        # decision loop, which either plans again or records an explicit wait.
        self.assertIn(rejected.status, {BrainStatus.REPLANNING, BrainStatus.WAITING})
        self.assertEqual(goal.status, GoalStatus.CANCELLED)
        self.assertEqual(
            brain.state.last_outcome, "PERFORM_REGISTERED_SKILL:setup_camera_and_dance:cancelled"
        )
        self.assertIsNone(brain.state.pending_action)

        self.clock.advance(1)
        outcome = brain.tick(world(), None)
        self.assertIn(outcome.status, {BrainStatus.EXECUTING, BrainStatus.WAITING})
        self.assertNotEqual(brain.state.current_goal, goal)

    def test_plan_time_failure_marks_failed_and_never_errors(self) -> None:
        brain = self.brain()
        self.settle(brain, world())

        def revoked(*_args, **_kwargs):
            return SkillPlanResult(reason="SKILL_NOT_FOUND")

        brain.skill_executor.create_plan = revoked  # type: ignore[method-assign]
        outcome = brain.tick(world(), None)
        self.assertEqual(outcome.status, BrainStatus.WAITING)
        self.assertEqual(outcome.message, "autonomy_wait")
        self.assertEqual(outcome.goal.status, GoalStatus.FAILED)
        self.assertIsNone(brain.state.current_goal)
        self.assertIsNone(brain.state.pending_action)
        self.assertEqual(brain.state.last_failure, "initiative_unavailable:SKILL_NOT_FOUND")
        self.assertEqual(
            brain.state.last_outcome, "PERFORM_REGISTERED_SKILL:setup_camera_and_dance:failed"
        )

        # The failed activity is journalled, the tick is not wedged, and the
        # next idle tick may select something else.
        del brain.skill_executor.create_plan
        self.clock.advance(6)
        follow = brain.tick(world(), None)
        self.assertEqual(follow.status, BrainStatus.EXECUTING)
        self.assertEqual(brain.state.selected_activity, "PERFORM_REGISTERED_SKILL:dance")

    def test_interval_paces_new_activities_without_resetting_the_goal(self) -> None:
        brain = self.brain()
        self.settle(brain, world())
        started = brain.tick(world(), None)
        self.assertEqual(started.status, BrainStatus.EXECUTING)
        first_goal = brain.state.current_goal
        brain.tick(world(), confirm(brain, success(ActionType.MOVE_TO, "CAMERA_STAND")))
        brain.tick(world(), confirm(brain, success(ActionType.LOOK_AT, "CAMERA")))
        brain.tick(world(), confirm(brain, success(ActionType.DANCE)))

        self.clock.advance(1)
        waiting = brain.tick(world(), None)
        self.assertEqual(waiting.status, BrainStatus.WAITING)
        self.assertEqual(brain.state.selected_activity, INITIATIVE_WAIT)
        self.assertEqual(brain.state.initiative_reason, "initiative_interval")
        self.assertEqual(first_goal.status, GoalStatus.COMPLETED)

        self.clock.advance(6)
        next_activity = brain.tick(world(), None)
        self.assertEqual(next_activity.status, BrainStatus.EXECUTING)
        self.assertEqual(brain.state.selected_activity, "PERFORM_REGISTERED_SKILL:dance")
        self.assertNotEqual(brain.state.current_goal.goal_id, first_goal.goal_id)

    def test_explore_cadence_does_not_starve_the_initiative(self) -> None:
        """A completed exploration hands one explore-due decision to autonomy.

        Production runs the default 25 s explore interval, and one legacy
        planning cycle regularly takes longer than that, so ``evaluate``
        returns a fresh ``EXPLORE_ROOM`` goal on every quiet decision and the
        ``GoalType.IDLE`` branch never runs. The gate under test claims the
        first explore-due decision *after* a completed exploration instead.
        """
        brain = self.brain()
        explore_interval = GoalEvaluatorConfig().explore_interval_seconds
        brain.goal_evaluator.config = GoalEvaluatorConfig(
            explore_interval_seconds=explore_interval
        )

        first = brain.tick(world(), None)
        self.assertEqual(first.goal.goal_type, GoalType.EXPLORE_ROOM)
        self.assertIsNone(first.goal.context.get("initiative"))
        follow = brain.tick(
            world(), confirm(brain, ActionResult(first.action.action, ExecutionStatus.SUCCESS, "reported"))
        )
        self.assertEqual(follow.message, "plan_complete")

        # The next decision is explore-due again (the planning cycle outlives
        # the interval); autonomy claims it through the production path once.
        self.clock.advance(explore_interval)
        started = brain.tick(world(), None)
        self.assertEqual(started.status, BrainStatus.EXECUTING)
        self.assertEqual(
            brain.state.selected_activity, "PERFORM_REGISTERED_SKILL:setup_camera_and_dance"
        )
        self.assertTrue(brain.state.current_goal.context.get("initiative"))
        self.assertIsNone(brain.state.legacy_explore_goal_id)


class RepetitionProtectionTests(AutonomyTestBase):
    def test_completed_activity_is_not_reselected_against_the_same_world(self) -> None:
        brain = self.brain(library=library_with("look_at"))
        self.settle(brain, world())

        started = brain.tick(world(), None)
        self.assertEqual(started.status, BrainStatus.EXECUTING)
        self.assertEqual(brain.state.selected_activity, "LOOK_AROUND")
        self.clock.advance(1)
        done = brain.tick(world(), confirm(brain, success(ActionType.LOOK_AT, "CAMERA")))
        self.assertEqual(done.status, BrainStatus.IDLE)
        self.assertEqual(brain.state.last_outcome, "LOOK_AROUND:completed")

        self.clock.advance(6)
        suppressed = brain.tick(world(), None)
        self.assertEqual(suppressed.status, BrainStatus.WAITING)
        self.assertEqual(brain.state.selected_activity, INITIATIVE_WAIT)
        self.assertIn("repetition_suppressed", brain.state.initiative_reason)

        # A real change in the world is a new reason and releases the activity.
        changed = world(stream={"viewer_count": 5})
        self.clock.advance(1)
        released = brain.tick(changed, None)
        self.assertEqual(released.status, BrainStatus.EXECUTING)
        self.assertEqual(brain.state.selected_activity, "LOOK_AROUND")

    def test_journal_is_bounded(self) -> None:
        brain = self.brain()
        for index in range(12):
            brain.state.recent_autonomy_activities.append(("LOOK_AROUND", f"s{index}"))
        self.assertLessEqual(len(brain.state.recent_autonomy_activities), 4)


class ViewerPriorityTests(AutonomyTestBase):
    def test_important_viewer_request_interrupts_the_initiative_without_stale_actions(self) -> None:
        brain = self.brain()
        self.settle(brain, world())
        brain.tick(world(), None)
        goal = brain.state.current_goal
        stale_plan_id = brain.state.current_plan.plan_id
        self.assertEqual(brain.state.pending_action.action, ActionType.MOVE_TO)

        request = brain._chat.submit("viewer_01", "Lacrimosa, can you dance?")
        self.assertTrue(request.accepted)
        self.assertGreaterEqual(request.message.priority, brain.config.interruption_priority)

        self.assertEqual(goal.status, GoalStatus.CANCELLED)
        self.assertIsNone(brain.state.current_goal)
        self.assertEqual(brain.state.interruption_reason, "interrupted_by_viewer")
        self.assertEqual(
            brain.state.last_outcome, "PERFORM_REGISTERED_SKILL:setup_camera_and_dance:interrupted"
        )
        self.assertIsNone(brain.state.pending_action)
        self.assertIsNone(brain.skill_executor.active)
        self.assertEqual(brain.skill_executor.evaluations[-1].status.value, "CANCELLED")

        self.clock.advance(1)
        answer = brain.tick(world(), None)
        self.assertEqual(answer.status, BrainStatus.EXECUTING)
        self.assertEqual(brain.state.current_goal.goal_type, GoalType.RESPOND_TO_VIEWER)
        self.assertIsNotNone(brain.state.current_plan)
        self.assertNotEqual(brain.state.current_plan.plan_id, stale_plan_id)
        self.assertFalse(brain._is_initiative_goal(brain.state.current_goal))

    def test_low_priority_message_does_not_preempt_in_flight_initiative(self) -> None:
        brain = self.brain()
        self.settle(brain, world())
        brain.tick(world(), None)
        goal = brain.state.current_goal
        plan_id = brain.state.current_plan.plan_id

        ack = brain._chat.submit("viewer_01", "yeah")
        self.assertLess(ack.message.priority, brain.config.interruption_priority)
        self.assertEqual(goal.status, GoalStatus.ACTIVE)
        self.assertIs(brain.state.current_goal, goal)
        self.assertEqual(brain.state.current_plan.plan_id, plan_id)

        self.clock.advance(1)
        continuing = brain.tick(world(), None)
        self.assertEqual(continuing.status, BrainStatus.EXECUTING)
        self.assertIs(brain.state.current_goal, goal)


class ObservationTests(AutonomyTestBase):
    def test_real_result_is_observed_and_advances_the_same_goal(self) -> None:
        brain = self.brain()
        self.settle(brain, world())
        brain.tick(world(), None)
        goal = brain.state.current_goal

        self.clock.advance(1)
        result = confirm(brain, success(ActionType.MOVE_TO, "CAMERA_STAND"))
        brain.tick(world(), result)
        self.assertIs(brain.state.last_result, result)
        self.assertIs(brain.state.current_goal, goal)
        self.assertEqual(brain.state.step_index, 1)
        self.assertEqual(brain.skill_executor.active.current_step, 1)

    def test_unexpected_result_is_not_observed_as_the_pending_action(self) -> None:
        brain = self.brain()
        self.settle(brain, world())
        brain.tick(world(), None)
        goal = brain.state.current_goal
        plan_id = brain.state.current_plan.plan_id

        self.clock.advance(1)
        # A result for a different action belongs to no awaited step.
        brain.tick(world(), confirm(brain, success(ActionType.WAVE)))
        self.assertIs(brain.state.current_goal, goal)
        self.assertEqual(brain.state.current_plan.plan_id, plan_id)
        self.assertEqual(brain.state.pending_action.action, ActionType.MOVE_TO)
        self.assertEqual(brain.state.step_index, 0)

    def test_autonomy_ticks_never_call_the_provider(self) -> None:
        provider = behavior.CapturingProvider()
        brain = self.brain(provider=provider)
        self.settle(brain, world())
        before = len(provider.requests)

        outcome = brain.tick(world(), None)
        self.assertEqual(outcome.status, BrainStatus.EXECUTING)
        self.assertEqual(len(provider.requests), before)

        # The whole initiative lifecycle -- start, observation, completion --
        # stays inside the Brain: no autonomous tick consults the provider.
        self.clock.advance(1)
        brain.tick(world(), confirm(brain, success(ActionType.MOVE_TO, "CAMERA_STAND")))
        self.clock.advance(1)
        brain.tick(world(), confirm(brain, success(ActionType.LOOK_AT, "CAMERA")))
        self.clock.advance(1)
        done = brain.tick(world(), confirm(brain, success(ActionType.DANCE)))
        self.assertEqual(done.status, BrainStatus.IDLE)
        self.assertEqual(done.message, "plan_complete")
        self.assertEqual(len(provider.requests), before)

        # And leaving the audience drops autonomy out of scope entirely: the
        # tick is handed back to the legacy path instead of a new activity.
        self.clock.advance(6)
        silent = brain.tick(world(stream={"viewer_count": 0}), None)
        self.assertNotEqual(silent.message, "autonomy_wait")
        self.assertFalse(brain._is_initiative_goal(brain.state.current_goal))


class DiagnosticsTests(AutonomyTestBase):
    def test_autonomy_state_exposes_the_documented_fields(self) -> None:
        brain = self.brain()
        service = BrainService(brain, brain._chat, clock=lambda: 0.0)
        payload = service.get_state()["autonomy"]
        # PHASE 27B: the spec requires the autonomy read-model to expose the
        # decision evidence additively, so the Phase 27A keys are pinned as a
        # required subset and the documented PHASE 27B fields are pinned too.
        # No Phase 27A field may disappear or change meaning.
        self.assertLessEqual(
            {
                "autonomy_enabled",
                "current_goal",
                "goal_status",
                "goal_reason",
                "initiative_reason",
                "selected_activity",
                "last_outcome",
            },
            set(payload),
        )
        self.assertLessEqual(
            {
                "candidate_activity",
                "eligible_candidates",
                "candidate_rejections",
                "last_activity",
                "replanning_reason",
                "next_decision_eligibility",
                "decision_source",
                "initiative_context",
            },
            set(payload),
        )
        self.assertTrue(payload["autonomy_enabled"])
        self.assertIsNone(payload["current_goal"])
        self.assertIsNone(payload["last_outcome"])

    def test_diagnostics_follow_wait_then_activity_then_completion(self) -> None:
        brain = self.brain(library=library_with("look_at"))
        service = BrainService(brain, brain._chat, clock=lambda: 0.0)
        self.settle(brain, world())

        brain.tick(world(), None)
        started = service.get_autonomy_state()
        self.assertEqual(started["selected_activity"], "LOOK_AROUND")
        self.assertEqual(started["goal_status"], "ACTIVE")
        self.assertIsNotNone(started["current_goal"])
        self.assertEqual(started["current_goal"]["context"]["activity"], "LOOK_AROUND")
        self.assertEqual(started["goal_reason"], brain.state.current_goal.reason)
        self.assertEqual(started["initiative_reason"], brain.state.current_goal.reason)
        self.assertIsNone(started["last_outcome"])

        self.clock.advance(1)
        brain.tick(world(), confirm(brain, success(ActionType.LOOK_AT, "CAMERA")))
        completed = service.get_autonomy_state()
        self.assertIsNone(completed["current_goal"])
        self.assertIsNone(completed["goal_status"])
        self.assertIsNone(completed["goal_reason"])
        self.assertEqual(completed["selected_activity"], "LOOK_AROUND")
        self.assertEqual(completed["last_outcome"], "LOOK_AROUND:completed")

    def test_legacy_goal_is_not_reported_as_an_autonomy_goal(self) -> None:
        brain = self.brain()
        service = BrainService(brain, brain._chat, clock=lambda: 0.0)
        brain._chat.submit("viewer_01", "Hi Lacrimosa!")
        brain.tick(world(), None)
        self.assertIsNotNone(brain.state.current_goal)
        payload = service.get_autonomy_state()
        self.assertIsNone(payload["current_goal"])
        self.assertIsNone(payload["goal_status"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
