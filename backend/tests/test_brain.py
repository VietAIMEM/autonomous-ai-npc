from __future__ import annotations

import json
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from backend.brain.goals import GoalEvaluator, GoalEvaluatorConfig
from backend.brain.loop import AutonomousBrain, BrainConfig, BrainStatus
from backend.brain.models import (
    ActionResult,
    ActionType,
    ExecutionStatus,
    Goal,
    GoalStatus,
    GoalType,
    PlanStep,
)
from backend.brain.personality import load_personality
from backend.brain.planner import BrainPlanner, BrainPlannerConfig
from backend.brain.reasoning import BrainReasoner, MockBrainLLMProvider
from backend.brain.targets import TargetResolver
from backend.brain.world_state import build_world_state
from backend.social.context import SocialContext
from backend.social.service import SocialAutonomyService
from backend.chat.intent import extract_intent
from backend.chat.models import ChatMessage
from backend.chat.normalization import normalize_text
from backend.chat.priority import score_priority
from backend.chat.service import ChatService, ChatServiceConfig
from backend.llm.base import LLMProvider, LLMRequest
from backend.llm.mock import MockLLMProvider
from backend.llm.openrouter import OpenRouterProvider


def snapshot(**overrides: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "character": {"position": [0, 0, 0], "rotation_y": 90, "current_action": "idle"},
        "phone": {"state": "Placed", "position": [1, 0.75, 0], "placement_surface": "TABLE"},
        "stream": {"is_streaming": False, "camera_available": True},
        "targets": {
            "CENTER": {"position": [0, 0, 0], "navigation_target": True, "look_at_target": True},
            "TABLE": {"position": [1, 0.75, 0], "navigation_target": True, "look_at_target": True},
            "CAMERA": {"position": [1, 0.8, 0], "navigation_target": False, "look_at_target": True},
        },
    }
    value.update(overrides)
    return value


def chat_message(text: str) -> ChatMessage:
    normalized = normalize_text(text)
    message = ChatMessage("msg", "viewer", text, normalized, datetime.now(timezone.utc))
    result = extract_intent(normalized)
    message.intent = result.intent
    message.target = result.target
    message.priority = score_priority(message)
    return message


def goal(goal_type: GoalType = GoalType.RESPOND_TO_VIEWER) -> Goal:
    return Goal("goal", goal_type, 0.9, "test", status=GoalStatus.ACTIVE)


def brain(
    provider: MockBrainLLMProvider | None = None,
    decision_interval: float = 0.0,
) -> AutonomousBrain:
    chat = ChatService(
        MockLLMProvider(),
        ChatServiceConfig(max_messages_per_rate_window=20),
    )
    return AutonomousBrain(
        provider or MockBrainLLMProvider(),
        chat,
        BrainConfig(decision_interval_seconds=decision_interval),
    )


class WorldStateTests(unittest.TestCase):
    def test_snapshot_normalization(self) -> None:
        world = build_world_state(snapshot())
        self.assertEqual(world.character_position, (0.0, 0.0, 0.0))
        self.assertEqual(world.character_rotation_y, 90.0)
        self.assertEqual(world.phone_state, "Placed")
        self.assertEqual(world.placement_surface, "TABLE")
        self.assertFalse(world.stream_active)
        self.assertTrue(world.stream_camera_available)
        self.assertEqual(world.viewer_count, 0)
        self.assertIn("CENTER", world.targets)

    def test_bad_snapshot_uses_safe_defaults(self) -> None:
        world = build_world_state({"character": {"position": "bad"}, "targets": "bad"})
        self.assertEqual(world.character_position, (0.0, 0.0, 0.0))
        self.assertEqual(world.targets, {})

    def test_viewer_count_floor_raises_snapshot_value(self) -> None:
        world = build_world_state(
            snapshot(stream={"is_streaming": True, "viewer_count": 2}),
            viewer_count_floor=7,
        )
        self.assertEqual(world.viewer_count, 7)

    def test_viewer_count_floor_never_lowers_snapshot_value(self) -> None:
        world = build_world_state(
            snapshot(stream={"is_streaming": True, "viewer_count": 9}),
            viewer_count_floor=3,
        )
        self.assertEqual(world.viewer_count, 9)

    def test_viewer_count_floor_clamps_negative_values(self) -> None:
        world = build_world_state(snapshot(), viewer_count_floor=-5)
        self.assertEqual(world.viewer_count, 0)


class GoalTests(unittest.TestCase):
    def test_action_chat_creates_viewer_response_goal(self) -> None:
        evaluator = GoalEvaluator(load_personality())
        selected = evaluator.evaluate(build_world_state(snapshot()), chat_message("Lacrimosa, can you dance?"))
        self.assertEqual(selected.goal_type, GoalType.RESPOND_TO_VIEWER)
        self.assertGreaterEqual(selected.priority, 0.82)

    def test_available_camera_creates_stream_maintenance_goal(self) -> None:
        evaluator = GoalEvaluator(load_personality(), GoalEvaluatorConfig(explore_interval_seconds=0))
        selected = evaluator.evaluate(build_world_state(snapshot()))
        self.assertEqual(selected.goal_type, GoalType.MAINTAIN_STREAM)

    def test_explore_cooldown_then_idle(self) -> None:
        start = datetime(2026, 9, 9, tzinfo=timezone.utc)
        evaluator = GoalEvaluator(
            load_personality(),
            GoalEvaluatorConfig(explore_interval_seconds=10),
            clock=lambda: start,
        )
        active_snapshot = snapshot(stream={"is_streaming": True})
        first = evaluator.evaluate(build_world_state(active_snapshot))
        second = evaluator.evaluate(build_world_state(active_snapshot))
        self.assertEqual(first.goal_type, GoalType.EXPLORE_ROOM)
        self.assertEqual(second.goal_type, GoalType.IDLE)

    def test_successful_stream_stop_is_not_immediately_restarted(self) -> None:
        start = datetime(2026, 9, 9, tzinfo=timezone.utc)
        social = SocialContext()
        social.update_on_action_result("STOP_STREAM", True, start, "viewer_01")
        social.update_on_action_result("STOP", True, start, "viewer_01")
        self.assertTrue(social.last_intentional_stream_stop)
        evaluator = GoalEvaluator(
            load_personality(),
            GoalEvaluatorConfig(explore_interval_seconds=0),
            clock=lambda: start,
        )
        stopped_world = build_world_state(
            snapshot(stream={"is_streaming": False, "camera_available": True})
        )
        first = evaluator.evaluate(stopped_world, social_context=social)
        second = evaluator.evaluate(stopped_world, social_context=social)
        self.assertNotEqual(first.goal_type, GoalType.MAINTAIN_STREAM)
        self.assertNotEqual(second.goal_type, GoalType.MAINTAIN_STREAM)
        social.update_on_action_result("START_STREAM", True, start, "viewer_01")
        self.assertFalse(social.last_intentional_stream_stop)
        restarted = GoalEvaluator(
            load_personality(),
            GoalEvaluatorConfig(explore_interval_seconds=100),
            clock=lambda: start,
        ).evaluate(stopped_world, social_context=social)
        self.assertEqual(restarted.goal_type, GoalType.MAINTAIN_STREAM)


class TargetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.resolver = TargetResolver(build_world_state(snapshot()))

    def test_valid_alias_resolution(self) -> None:
        self.assertEqual(self.resolver.resolve("center_of_room"), ("CENTER", "OK"))
        self.assertEqual(self.resolver.resolve("stream camera"), ("CAMERA", "OK"))

    def test_invalid_resolution(self) -> None:
        self.assertEqual(self.resolver.resolve("moon"), (None, "TARGET_NOT_FOUND"))


class PlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.world = build_world_state(snapshot())
        self.planner = BrainPlanner(BrainPlannerConfig(max_plan_steps=3))

    def test_prompt_is_compact_and_structured(self) -> None:
        prompt = self.planner.build_prompt(self.world, goal(), load_personality(), "dance")
        self.assertLessEqual(len(prompt), 4000)
        self.assertIn("DANCE", prompt)
        self.assertIn("CENTER", prompt)

    def test_parse_valid_plan(self) -> None:
        raw = json.dumps({"reason": "dance", "steps": [{"action": "DANCE"}]})
        plan, reason = self.planner.parse_plan(raw, goal(), self.world)
        self.assertEqual(reason, "OK")
        self.assertEqual(len(plan.steps), 1)

    def test_parse_failure_modes(self) -> None:
        selected = goal()
        cases = {
            "not-json": "INVALID_JSON",
            json.dumps([]): "INVALID_SCHEMA",
            json.dumps({"steps": []}): "EMPTY_PLAN",
            json.dumps({"steps": [{"action": "FLY_TO"}]}): "UNKNOWN_ACTION",
            json.dumps({"steps": [{"action": "DANCE"}] * 4}): "PLAN_TOO_LONG",
            json.dumps({"steps": [{"action": "MOVE_TO", "target": "MOON"}]}): "TARGET_NOT_FOUND",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                plan, reason = self.planner.parse_plan(raw, selected, self.world)
                self.assertIsNone(plan)
                self.assertEqual(reason, expected)

    def test_step_validation_safety(self) -> None:
        self.assertTrue(self.planner.validate_step(PlanStep(ActionType.MOVE_TO, "CENTER"), self.world).valid)
        self.assertFalse(self.planner.validate_step(PlanStep(ActionType.MOVE_TO, "MOON"), self.world).valid)
        self.assertTrue(self.planner.validate_step(PlanStep(ActionType.LOOK_AT, "CAMERA"), self.world).valid)
        held_world = build_world_state(snapshot(phone={"state": "Held"}))
        self.assertFalse(self.planner.validate_step(PlanStep(ActionType.PICK_UP_PHONE), held_world).valid)
        self.assertFalse(self.planner.validate_step(PlanStep(ActionType.PLACE_PHONE, "TABLE"), self.world).valid)

    def test_dance_with_motion_reference_is_valid(self) -> None:
        reference = "Assets/animation/dance"
        raw = json.dumps(
            {"reason": "dance", "steps": [{"action": "DANCE", "target": reference}]}
        )
        plan, reason = self.planner.parse_plan(raw, goal(), self.world)
        self.assertEqual(reason, "OK")
        self.assertIsNotNone(plan)
        self.assertEqual(plan.steps[0].target, reference)
        self.assertTrue(
            self.planner.validate_step(
                PlanStep(ActionType.DANCE, reference), self.world
            ).valid
        )

    def test_wave_with_motion_reference_is_valid(self) -> None:
        reference = "Assets/Resources/GeneratedMotions/Phase17/wave_demo.anim"
        plan, reason = self.planner.parse_plan(
            json.dumps(
                {"reason": "wave", "steps": [{"action": "WAVE", "target": reference}]}
            ),
            goal(),
            self.world,
        )
        self.assertEqual(reason, "OK")
        self.assertIsNotNone(plan)
        self.assertEqual(plan.steps[0].target, reference)

    def test_motion_reference_backslashes_are_accepted_unchanged(self) -> None:
        reference = "Assets\\Resources\\GeneratedMotions\\Phase17\\walk.anim"
        plan, reason = self.planner.parse_plan(
            json.dumps(
                {"reason": "walk", "steps": [{"action": "DANCE", "target": reference}]}
            ),
            goal(),
            self.world,
        )
        self.assertEqual(reason, "OK")
        self.assertIsNotNone(plan)
        self.assertEqual(plan.steps[0].target, reference)

    def test_dance_with_spatial_target_is_rejected(self) -> None:
        # The live provider returned DANCE:BENCH for a viewer asking to dance
        # near the bench. BENCH is a spatial target, not a motion reference;
        # accepting it here sent it to the Unity bridge, which could only
        # answer generated_motion_unresolved:FAILED. The honest rejection
        # keeps the plan and the Animator state consistent.
        plan, reason = self.planner.parse_plan(
            json.dumps(
                {"reason": "dance near the bench", "steps": [{"action": "DANCE", "target": "BENCH"}]}
            ),
            goal(),
            self.world,
        )
        self.assertIsNone(plan)
        self.assertEqual(reason, "MOTION_REFERENCE_INVALID")
        validation = self.planner.validate_step(
            PlanStep(ActionType.DANCE, "BENCH"), self.world
        )
        self.assertFalse(validation.valid)
        self.assertEqual(validation.reason, "MOTION_REFERENCE_INVALID")

    def test_wave_with_spatial_target_is_rejected(self) -> None:
        plan, reason = self.planner.parse_plan(
            json.dumps(
                {"reason": "wave at the camera", "steps": [{"action": "WAVE", "target": "CAMERA"}]}
            ),
            goal(),
            self.world,
        )
        self.assertIsNone(plan)
        self.assertEqual(reason, "MOTION_REFERENCE_INVALID")

    def test_bare_gesture_actions_stay_targetless(self) -> None:
        # The built-in WAVE/DANCE states are the intended targetless form and
        # must keep validating exactly as before.
        self.assertTrue(self.planner.validate_step(PlanStep(ActionType.WAVE), self.world).valid)
        self.assertTrue(self.planner.validate_step(PlanStep(ActionType.DANCE), self.world).valid)

    def test_spatial_targets_keep_working_for_movement_actions(self) -> None:
        self.assertTrue(
            self.planner.validate_step(
                PlanStep(ActionType.MOVE_TO, "BENCH"), build_world_state(
                    snapshot(targets={"BENCH": {"position": [0, 0, 0], "navigation_target": True, "look_at_target": True}})
                )
            ).valid
        )
        lowered = self.planner.validate_step(
            PlanStep(ActionType.MOVE_TO, "center"), self.world
        )
        self.assertTrue(lowered.valid)


class ReasoningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.world = build_world_state(snapshot())
        self.planner = BrainPlanner()

    def test_mock_dance_plan(self) -> None:
        plan, reason = BrainReasoner(MockBrainLLMProvider(), self.planner).plan(
            self.world,
            goal(),
            load_personality(),
            "dance",
        )
        self.assertEqual(reason, "OK")
        self.assertEqual(plan.steps[0].action, ActionType.MOVE_TO)
        self.assertEqual(plan.steps[-1].action, ActionType.SPEAK)

    def test_mock_maintain_stream_plan_starts_stream(self) -> None:
        plan, reason = BrainReasoner(MockBrainLLMProvider(), self.planner).plan(
            self.world,
            goal(GoalType.MAINTAIN_STREAM),
            load_personality(),
        )
        self.assertEqual(reason, "OK")
        self.assertEqual(plan.steps[0].action, ActionType.START_STREAM)

    def test_mock_selects_existing_stream_and_phone_skills(self) -> None:
        cases = {
            "Lacrimosa, can you start the stream?": (
                "start_stream",
                {},
            ),
            "Lacrimosa, can you stop the stream?": (
                "stop_stream",
                {},
            ),
            "Lacrimosa, pick up the phone.": (
                "pick_up_phone",
                {},
            ),
            "Lacrimosa, place the phone on the table.": (
                "place_phone",
                {"target": "TABLE"},
            ),
        }
        for chat, (skill_id, parameters) in cases.items():
            with self.subTest(chat=chat):
                raw = MockBrainLLMProvider().generate(
                    LLMRequest(prompt=chat, context={"chat": chat})
                )
                payload = json.loads(raw)
                self.assertEqual(payload["skill_id"], skill_id)
                self.assertEqual(payload["parameters"], parameters)
                self.assertTrue(payload["reason"].startswith("viewer requested"))

    def test_mock_dance_uses_highest_ranked_learned_skill(self) -> None:
        ranking = [
            {"skill_id": "dance_02", "advisory_score": 0.85, "learned": True},
            {"skill_id": "dance_01", "advisory_score": 0.55, "learned": True},
            {"skill_id": "phone_camera_dance", "advisory_score": 1.0, "learned": True, "category": "COMPOSITE"},
            {"skill_id": "dance", "advisory_score": 1.0, "learned": False},
        ]
        raw = MockBrainLLMProvider().generate(
            LLMRequest(prompt="dance", context={"chat": "dance", "learned_ranking": ranking})
        )
        self.assertEqual(json.loads(raw), {
            "skill_id": "dance_02",
            "parameters": {},
            "reason": "selected the highest-ranked learned dance",
        })

    def test_camera_dance_uses_highest_ranked_learned_composite(self) -> None:
        ranking = [
            {"skill_id": "dance_02", "advisory_score": 0.85, "learned": True},
            {"skill_id": "dance_01", "advisory_score": 0.55, "learned": True},
            {
                "skill_id": "phone_camera_dance",
                "advisory_score": 1.0,
                "learned": True,
                "category": "COMPOSITE",
            },
            {
                "skill_id": "setup_camera_and_dance",
                "advisory_score": 1.2,
                "learned": False,
                "category": "COMPOSITE",
            },
        ]
        raw = MockBrainLLMProvider().generate(
            LLMRequest(
                prompt="dance for the camera",
                context={
                    "chat": "dance for the camera",
                    "learned_ranking": ranking,
                },
            )
        )
        self.assertEqual(
            json.loads(raw),
            {
                "skill_id": "phone_camera_dance",
                "parameters": {},
                "reason": "selected the highest-ranked learned dance composition",
            },
        )

    def test_camera_dance_prefers_available_learned_composite_over_simple_dance(self) -> None:
        ranking = [
            {"skill_id": "dance_02", "advisory_score": 0.85, "learned": True},
            {
                "skill_id": "phone_camera_dance",
                "advisory_score": 0.55,
                "learned": True,
                "category": "COMPOSITE",
            },
        ]
        raw = MockBrainLLMProvider().generate(
            LLMRequest(
                prompt="dance for the camera",
                context={
                    "chat": "dance for the camera",
                    "learned_ranking": ranking,
                },
            )
        )
        self.assertEqual(
            json.loads(raw),
            {
                "skill_id": "phone_camera_dance",
                "parameters": {},
                "reason": "selected the highest-ranked learned dance composition",
            },
        )

    def test_provider_failures_are_safe(self) -> None:
        for failure in ("timeout", "provider", "invalid_json", "unknown_action"):
            with self.subTest(failure=failure):
                provider = MockBrainLLMProvider()
                provider.fail_with = failure
                plan, reason = BrainReasoner(provider, self.planner).plan(
                    self.world,
                    goal(),
                    load_personality(),
                    "dance",
                )
                self.assertIsNone(plan)
                self.assertTrue(reason)


class LoopTests(unittest.TestCase):
    def test_chat_plan_advances_step_by_step(self) -> None:
        loop = brain(decision_interval=1.0)
        loop.chat.submit("viewer_01", "Lacrimosa can you dance?")
        first = loop.tick(build_world_state(snapshot(stream={"is_streaming": True})))
        self.assertEqual(first.status, BrainStatus.EXECUTING)
        self.assertEqual(first.action.action, ActionType.MOVE_TO)
        self.assertEqual(first.action.target, "CENTER")

        second = loop.tick(
            build_world_state(snapshot(stream={"is_streaming": True})),
            ActionResult(
                first.action.action,
                ExecutionStatus.SUCCESS,
                dispatch_id=first.action.dispatch_id,
            ),
        )
        self.assertEqual(second.action.action, ActionType.LOOK_AT)
        third = loop.tick(
            build_world_state(snapshot(stream={"is_streaming": True})),
            ActionResult(
                second.action.action,
                ExecutionStatus.SUCCESS,
                dispatch_id=second.action.dispatch_id,
            ),
        )
        self.assertEqual(third.action.action, ActionType.DANCE)
        fourth = loop.tick(
            build_world_state(snapshot(stream={"is_streaming": True})),
            ActionResult(
                third.action.action,
                ExecutionStatus.SUCCESS,
                dispatch_id=third.action.dispatch_id,
            ),
        )
        self.assertEqual(fourth.action.action, ActionType.SPEAK)

    def test_failed_action_retries_then_cancels(self) -> None:
        loop = brain()
        loop.chat.submit("viewer_01", "Lacrimosa can you dance?")
        first = loop.tick(build_world_state(snapshot(stream={"is_streaming": True})))
        retry = loop.tick(
            build_world_state(snapshot(stream={"is_streaming": True})),
            ActionResult(
                first.action.action,
                ExecutionStatus.FAILED,
                dispatch_id=first.action.dispatch_id,
            ),
        )
        self.assertEqual(retry.action, first.action)
        cancelled = loop.tick(
            build_world_state(snapshot(stream={"is_streaming": True})),
            ActionResult(
                first.action.action,
                ExecutionStatus.FAILED,
                dispatch_id=first.action.dispatch_id,
            ),
        )
        self.assertEqual(cancelled.status, BrainStatus.EXECUTING)
        self.assertNotEqual(cancelled.plan, first.plan)
        self.assertIsNotNone(cancelled.action)

    def test_rejected_action_cancels_plan(self) -> None:
        loop = brain()
        loop.chat.submit("viewer_01", "table")
        first = loop.tick(build_world_state(snapshot(stream={"is_streaming": True})))
        second = loop.tick(
            build_world_state(snapshot(stream={"is_streaming": True})),
            ActionResult(
                first.action.action,
                ExecutionStatus.REJECTED,
                dispatch_id=first.action.dispatch_id,
            ),
        )
        self.assertEqual(second.status, BrainStatus.EXECUTING)
        self.assertNotEqual(second.plan, first.plan)
        self.assertIsNotNone(second.action)

    def test_decision_cooldown(self) -> None:
        loop = brain(decision_interval=10.0)
        loop.chat.submit("viewer_01", "Lacrimosa can you dance?")
        first = loop.tick(build_world_state(snapshot(stream={"is_streaming": True})))
        second = loop.tick(
            build_world_state(snapshot(stream={"is_streaming": True})),
            ActionResult(
                first.action.action,
                ExecutionStatus.SUCCESS,
                dispatch_id=first.action.dispatch_id,
            ),
        )
        self.assertTrue(second.action or second.message == "decision_cooldown")

    def test_stream_maintenance_preempts_a_saturated_chat_queue(self) -> None:
        # PHASE 24: real Twitch traffic kept the queue non-empty on nearly every
        # tick, so chat handling ran first and stream recovery waited minutes.
        # While the stream is down and no deliberate stop is on record, the
        # maintenance goal must win the tick without consuming the message.
        loop = brain(decision_interval=1.0)
        loop.chat.submit("viewer_01", "Lacrimosa can you dance?")

        first = loop.tick(build_world_state(snapshot()))

        self.assertEqual(first.status, BrainStatus.EXECUTING)
        self.assertIsNotNone(first.goal)
        self.assertEqual(first.goal.goal_type, GoalType.MAINTAIN_STREAM)
        self.assertIsNotNone(first.action)
        self.assertEqual(first.action.action, ActionType.START_STREAM)
        self.assertEqual(len(loop.chat.queue), 1)
        queued = loop.chat.queue.peek()
        self.assertIsNotNone(queued)
        self.assertEqual(queued.viewer_id, "viewer_01")

    def test_latched_stop_keeps_chat_ahead_of_stream_maintenance(self) -> None:
        # The preemption must respect the deliberate-stop latch: after an
        # operator stop the viewer's request is still attended to (and the
        # planner's existing latch gate rejects any stream restart).
        social = SocialAutonomyService(clock=datetime.now(timezone.utc))
        social.context.set_stream_stop_latch(True, source="test:operator_stop")
        loop = AutonomousBrain(
            MockBrainLLMProvider(),
            ChatService(
                MockLLMProvider(),
                ChatServiceConfig(max_messages_per_rate_window=20),
            ),
            BrainConfig(decision_interval_seconds=0.0),
            social_service=social,
        )
        loop.chat.submit("viewer_01", "Lacrimosa can you dance?")

        tick = loop.tick(build_world_state(snapshot()))

        self.assertIsNotNone(tick.goal)
        self.assertEqual(tick.goal.goal_type, GoalType.RESPOND_TO_VIEWER)
        self.assertNotEqual(
            tick.action.action if tick.action else None, ActionType.START_STREAM
        )
        self.assertEqual(len(loop.chat.queue), 0)


class FakeResponse:
    def __init__(self, raw: bytes) -> None:
        self.raw = raw

    def read(self) -> bytes:
        return self.raw

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None


class OpenRouterTests(unittest.TestCase):
    def test_generation_uses_configured_key(self) -> None:
        payload = {"choices": [{"message": {"content": '{"steps":[]}'}}]}
        captured: dict[str, Any] = {}

        def fake_urlopen(request: Any, timeout: float) -> FakeResponse:
            captured["authorization"] = request.headers.get("Authorization")
            return FakeResponse(json.dumps(payload).encode("utf-8"))

        provider = OpenRouterProvider(api_key="test-key", urlopen=fake_urlopen)
        output = provider.generate(LLMRequest(prompt="plan"))
        self.assertEqual(output, '{"steps":[]}')
        self.assertIn("Bearer test-key", str(captured["authorization"]))

    def test_reasoning_details_are_preserved_for_follow_up(self) -> None:
        captured: dict[str, Any] = {}

        def fake_urlopen(request: Any, timeout: float) -> FakeResponse:
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return FakeResponse(b'{"choices":[{"message":{"content":"ok"}}]}')

        provider = OpenRouterProvider(api_key="test-key", urlopen=fake_urlopen)
        request = LLMRequest(
            prompt="continue",
            context={
                "assistant_message": {
                    "content": "",
                    "reasoning_details": [{"type": "text", "summary": "checking"}],
                },
                "follow_up": "return JSON",
            },
        )
        self.assertEqual(provider.generate(request), "ok")
        assistant = captured["body"]["messages"][2]
        self.assertEqual(assistant["role"], "assistant")
        self.assertEqual(assistant["reasoning_details"], [{"type": "text", "summary": "checking"}])

    def test_malformed_and_empty_responses_fail_safely(self) -> None:
        cases = [
            "not-json",
            "{}",
            json.dumps({"choices": []}),
            json.dumps({"choices": [{"message": {}}]}),
        ]
        for raw in cases:
            with self.subTest(raw=raw):
                provider = OpenRouterProvider(
                    api_key="test-key",
                    urlopen=lambda request, timeout: FakeResponse(raw.encode("utf-8")),
                )
                with self.assertRaises(Exception):
                    provider.generate(LLMRequest(prompt="plan"))

    def test_local_key_file_is_ignored_and_available(self) -> None:
        from backend.llm.openrouter import DEFAULT_API_KEY_PATH

        provider = OpenRouterProvider()
        self.assertTrue(DEFAULT_API_KEY_PATH.exists())
        self.assertTrue(provider.has_api_key)
        gitignore = DEFAULT_API_KEY_PATH.parents[2].joinpath(".gitignore").read_text(encoding="utf-8")
        self.assertIn("backend/llm/api_key.txt", gitignore)


class NvidiaTransportTests(unittest.TestCase):
    """PHASE 21 / PART 2: the real NIM transport must not pay for hidden
    reasoning tokens, and OpenRouter's payload must stay exactly as it was.
    """

    def test_nvidia_payload_disables_hidden_thinking(self) -> None:
        from backend.llm.nvidia import NvidiaConfig, NvidiaProvider

        captured: dict[str, Any] = {}

        def fake_urlopen(request: Any, timeout: float) -> FakeResponse:
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return FakeResponse(b'{"choices":[{"message":{"content":"{}"}}]}')

        provider = NvidiaProvider(NvidiaConfig(), api_key="test-key", urlopen=fake_urlopen)
        provider.generate(LLMRequest(prompt="plan"))
        body = captured["body"]
        self.assertEqual(body["model"], "deepseek-ai/deepseek-v4.1-flash")
        self.assertEqual(body["chat_template_kwargs"], {"thinking": False})
        self.assertEqual(body["temperature"], 1.0)
        self.assertEqual(body["top_p"], 0.95)
        self.assertEqual(body["max_tokens"], 2048)

    def test_openrouter_payload_has_no_chat_template_kwargs(self) -> None:
        captured: dict[str, Any] = {}

        def fake_urlopen(request: Any, timeout: float) -> FakeResponse:
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return FakeResponse(b'{"choices":[{"message":{"content":"ok"}}]}')

        OpenRouterProvider(api_key="test-key", urlopen=fake_urlopen).generate(
            LLMRequest(prompt="plan")
        )
        self.assertNotIn("chat_template_kwargs", captured["body"])
        self.assertNotIn("reasoning", captured["body"])


class ClineTransportTests(unittest.TestCase):
    """PHASE 21 / PART 2: Cline reuses the shared transport unchanged.

    Only the endpoint, model and secret path may differ from OpenRouter; the
    sampling payload stays empty so the provider's request shape is exactly
    what the shared (already-audited) client produces.
    """

    def test_cline_payload_targets_cline_endpoint_and_model(self) -> None:
        from backend.llm.cline import ClineConfig, ClineProvider

        captured: dict[str, Any] = {}

        def fake_urlopen(request: Any, timeout: float) -> FakeResponse:
            captured["url"] = request.full_url
            captured["auth"] = request.get_header("Authorization")
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return FakeResponse(b'{"choices":[{"message":{"content":"{}"}}]}')

        provider = ClineProvider(
            ClineConfig(max_tokens=512), api_key="test-key", urlopen=fake_urlopen
        )
        provider.generate(LLMRequest(prompt="plan"))

        self.assertEqual(
            captured["url"], "https://api.cline.bot/api/v1/chat/completions"
        )
        self.assertEqual(captured["auth"], "Bearer test-key")
        body = captured["body"]
        self.assertEqual(body["model"], "nvidia/nemotron-3.5-lightning:free")
        self.assertEqual(body["max_tokens"], 512)
        # The gateway spends the completion budget on hidden reasoning, so the
        # planner turns it off rather than raising the token ceiling.
        self.assertEqual(body["reasoning"], {"enabled": False})
        self.assertNotIn("temperature", body)
        self.assertNotIn("top_p", body)
        self.assertNotIn("chat_template_kwargs", body)

    def test_cline_provider_is_registered_and_secret_is_file_backed(self) -> None:
        from backend.llm import config as llm_config
        from backend.llm.cline import DEFAULT_API_KEY_PATH

        self.assertIn(llm_config.PROVIDER_CLINE, llm_config.VALID_PROVIDERS)
        resolved = llm_config.LLMConfig.from_environment({"LLM_PROVIDER": "cline"})
        self.assertEqual(resolved.provider, "cline")
        self.assertEqual(resolved.model, "nvidia/nemotron-3.5-lightning:free")
        self.assertEqual(resolved.base_url, "https://api.cline.bot/api/v1")
        # The existing Brain token budget is preserved for this provider.
        self.assertEqual(resolved.max_tokens, llm_config.DEFAULT_MAX_TOKENS)
        self.assertEqual(Path(resolved.api_key_path), DEFAULT_API_KEY_PATH)
        provider = llm_config.build_provider(resolved)
        self.assertEqual(provider.name, "cline")
        expected_configured = bool(
            DEFAULT_API_KEY_PATH.read_text(encoding="utf-8").strip()
        )
        self.assertEqual(provider.has_api_key, expected_configured)
        self.assertEqual(resolved.configured, expected_configured)
        self.assertEqual(
            llm_config.describe_provider(provider)["kind"],
            "cline",
        )
        self.assertEqual(llm_config.describe_provider(provider)["mock"], False)

    def test_unknown_provider_still_rejected(self) -> None:
        from backend.llm import config as llm_config

        with self.assertRaises(llm_config.LLMConfigError):
            llm_config.LLMConfig.from_environment({"LLM_PROVIDER": "cline-typo"})

    def test_cline_gateway_envelope_is_unwrapped(self) -> None:
        """Cline returns ``{"success":..,"data":<openai payload>}``."""
        from backend.llm.cline import ClineConfig, ClineProvider

        body = json.dumps(
            {
                "success": True,
                "data": {
                    "object": "chat.completion",
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": '{"reason":"dance","steps":[]}',
                                "reasoning": "hidden",
                            },
                        }
                    ],
                },
            }
        ).encode("utf-8")
        provider = ClineProvider(
            ClineConfig(),
            api_key="test-key",
            urlopen=lambda request, timeout: FakeResponse(body),
        )
        self.assertEqual(
            provider.generate(LLMRequest(prompt="plan")),
            '{"reason":"dance","steps":[]}',
        )

    def test_openrouter_extraction_is_unchanged_for_flat_payload(self) -> None:
        """The shared extractor still owns unwrapped provider bodies."""
        from backend.llm.cline import ClineConfig, ClineProvider

        raw = b'{"choices":[{"message":{"content":"ok"}}]}'
        provider = ClineProvider(
            ClineConfig(),
            api_key="test-key",
            urlopen=lambda request, timeout: FakeResponse(raw),
        )
        self.assertEqual(provider.generate(LLMRequest(prompt="plan")), "ok")


class StaticProvider(LLMProvider):
    """Returns one canned payload, standing in for a real provider reply."""

    def __init__(self, payload: str) -> None:
        self.payload = payload

    def generate(self, request: LLMRequest) -> str:
        return self.payload


class NestedSkillRoutingTests(unittest.TestCase):
    """A skill invocation nested in ``steps`` routes through SkillExecutor."""

    ANIMATION_REFERENCE = (
        "Assets/Resources/GeneratedMotions/Phase16/learned_walk_demo.anim"
    )

    def setUp(self) -> None:
        from backend.skills.executor import SkillExecutor
        from backend.skills.library import SkillLibrary
        from backend.skills.models import Skill, SkillActionStep, SkillCategory

        library = SkillLibrary()
        library.register(
            Skill(
                skill_id="learned_walk_demo",
                name="Learned walk demo",
                description="imported Phase 16 walking motion",
                category=SkillCategory.SOCIAL,
                animation_reference=self.ANIMATION_REFERENCE,
                estimated_duration_seconds=4.0,
                tags=("learned",),
                steps=(
                    SkillActionStep(
                        ActionType.DANCE, constant_target=self.ANIMATION_REFERENCE
                    ),
                ),
            )
        )
        self.world = build_world_state(snapshot())
        self.executor = SkillExecutor(
            library, BrainPlanner(), lambda: datetime.now(timezone.utc)
        )

    def _plan(self, payload: dict[str, Any]):
        return BrainReasoner(
            StaticProvider(json.dumps(payload)), BrainPlanner(), self.executor
        ).plan(
            self.world,
            goal(),
            load_personality(),
            "show me the walking motion you learned",
        )

    def test_nested_registered_skill_routes_to_skill_executor(self) -> None:
        plan, reason = self._plan(
            {
                "reason": "viewer asked for the learned walking motion",
                "steps": [{"action": "learned_walk_demo", "parameters": {}}],
            }
        )
        self.assertEqual(reason, "OK")
        self.assertIsNotNone(plan)
        self.assertEqual(plan.steps[0].action, ActionType.DANCE)
        self.assertEqual(plan.steps[0].target, self.ANIMATION_REFERENCE)

    def test_nested_unregistered_skill_still_rejected(self) -> None:
        plan, reason = self._plan(
            {"reason": "bad output", "steps": [{"action": "fly_to", "parameters": {}}]}
        )
        self.assertIsNone(plan)
        self.assertEqual(reason, "UNKNOWN_ACTION")

    def test_case_variant_of_a_known_action_is_not_a_skill_id(self) -> None:
        # PHASE 36: action names are resolved case-insensitively, so the
        # lowercase "dance" is now the built-in DANCE action rather than an
        # unknown token. The original intent still holds: it is NOT routed
        # through the skill executor as a skill invocation, which would have
        # replaced the step with the skill's constant target.
        plan, reason = self._plan(
            {"reason": "bad output", "steps": [{"action": "dance", "parameters": {}}]}
        )
        self.assertIsNotNone(plan, reason)
        self.assertEqual(reason, "OK")
        self.assertEqual(plan.steps[0].action, ActionType.DANCE)
        self.assertIsNone(plan.steps[0].target)

    def test_top_level_action_alias_routes_to_skill_executor(self) -> None:
        # A real provider answered without ``steps`` and named the registered
        # skill under ``action``; the salvage re-wraps it through the normal
        # SkillExecutor selection validation instead of dropping the turn.
        plan, reason = self._plan(
            {
                "reason": "viewer asked for the learned walking motion",
                "action": "learned_walk_demo",
                "parameters": {},
            }
        )
        self.assertEqual(reason, "OK")
        self.assertIsNotNone(plan)
        self.assertEqual(plan.steps[0].action, ActionType.DANCE)
        self.assertEqual(plan.steps[0].target, self.ANIMATION_REFERENCE)

    def test_wrapped_plan_step_routes_to_skill_executor(self) -> None:
        plan, reason = self._plan(
            {
                "plan": {
                    "steps": [
                        {"skill_id": "learned_walk_demo", "parameters": {}}
                    ]
                }
            }
        )
        self.assertEqual(reason, "OK")
        self.assertIsNotNone(plan)
        self.assertEqual(plan.steps[0].target, self.ANIMATION_REFERENCE)

    def test_top_level_skill_alias_routes_to_skill_executor(self) -> None:
        plan, reason = self._plan(
            {"skill": "learned_walk_demo", "reason": "viewer request"}
        )
        self.assertEqual(reason, "OK")
        self.assertIsNotNone(plan)
        self.assertEqual(plan.steps[0].target, self.ANIMATION_REFERENCE)

    def test_skill_output_schema_wrapper_routes_to_skill_executor(self) -> None:
        # Captured from the real runtime: the provider echoed a
        # ``skill_output_schema`` wrapper instead of emitting ``steps``.
        plan, reason = self._plan(
            {
                "reason": (
                    "Viewer directly requests the walking motion; use the "
                    "listed learned_walk_demo skill as requested."
                ),
                "skill_output_schema": {
                    "skill_id": "learned_walk_demo",
                    "parameters": {},
                    "reason": "Viewer chat explicitly asked for the motion now.",
                },
            }
        )
        self.assertEqual(reason, "OK")
        self.assertIsNotNone(plan)
        self.assertEqual(plan.steps[0].action, ActionType.DANCE)
        self.assertEqual(plan.steps[0].target, self.ANIMATION_REFERENCE)

    def test_explicit_skill_id_matching_builtin_action_routes_to_executor(self) -> None:
        # Captured from the real runtime: ``start_stream`` is both a built-in
        # action and a registered skill. An explicit ``skill_id`` is a skill
        # selection (the same shape ``parse_selection`` accepts at top level),
        # so the built-in-name guard must not veto it.
        from backend.skills.models import Skill, SkillActionStep, SkillCategory

        self.executor.library.register(
            Skill(
                skill_id="start_stream",
                name="Start Stream",
                description="starts the livestream through the stream system",
                category=SkillCategory.LIVESTREAM,
                steps=(SkillActionStep(ActionType.START_STREAM),),
            )
        )
        plan, reason = self._plan(
            {
                "reason": (
                    "Viewer directly requested starting the stream, which "
                    "matches the listed start_stream skill."
                ),
                "skill_output_schema": {
                    "skill_id": "start_stream",
                    "parameters": {},
                    "reason": "Viewer asked to start the stream now.",
                },
            }
        )
        self.assertEqual(reason, "OK")
        self.assertIsNotNone(plan)
        self.assertEqual(plan.steps[0].action, ActionType.START_STREAM)

    def test_explicit_unregistered_skill_id_is_not_salvaged(self) -> None:
        plan, reason = self._plan(
            {
                "reason": "bad output",
                "skill_output_schema": {
                    "skill_id": "stop_stream",
                    "parameters": {},
                },
            }
        )
        self.assertIsNone(plan)
        self.assertEqual(reason, "EMPTY_PLAN")

    def test_action_alias_matching_builtin_action_is_still_refused(self) -> None:
        # Same collision, but reached through the ``action`` alias: a bare
        # built-in action is still owned by the plan path, never a skill.
        from backend.skills.models import Skill, SkillActionStep, SkillCategory

        self.executor.library.register(
            Skill(
                skill_id="start_stream",
                name="Start Stream",
                description="starts the livestream through the stream system",
                category=SkillCategory.LIVESTREAM,
                steps=(SkillActionStep(ActionType.START_STREAM),),
            )
        )
        plan, reason = self._plan(
            {"reason": "bad output", "action": "START_STREAM"}
        )
        self.assertIsNone(plan)
        self.assertEqual(reason, "EMPTY_PLAN")

    def test_top_level_unregistered_action_still_rejected(self) -> None:
        plan, reason = self._plan(
            {"reason": "bad output", "action": "fly_to", "parameters": {}}
        )
        self.assertIsNone(plan)
        self.assertEqual(reason, "EMPTY_PLAN")

    def test_bare_builtin_action_is_not_salvaged(self) -> None:
        # No silent substitution: a bare built-in action without steps keeps
        # failing instead of being re-routed to a similarly named skill.
        plan, reason = self._plan({"reason": "bad output", "action": "DANCE"})
        self.assertIsNone(plan)
        self.assertEqual(reason, "EMPTY_PLAN")


if __name__ == "__main__":
    unittest.main()
