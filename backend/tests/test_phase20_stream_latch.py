"""PHASE 20: a deliberate STOP_STREAM survives a backend restart.

Everything here is offline. The transport is the existing
``MockStreamingProvider``; no encoder, platform or Unity connection is needed.
"""

from __future__ import annotations

import json
import threading
import time
import unittest
from datetime import datetime, timedelta, timezone
from http import HTTPStatus
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any

from backend.api.diagnostics import build_diagnostics, build_status
from backend.api.stream import handle_stream_request
from backend.brain.goals import GoalEvaluator, GoalEvaluatorConfig
from backend.brain.loop import AutonomousBrain, BrainConfig, BrainStatus
from backend.brain.models import ActionResult, ActionType, ExecutionStatus, GoalType
from backend.brain.personality import load_personality
from backend.brain.world_state import build_world_state
from backend.chat.service import ChatService, ChatServiceConfig
from backend.llm.base import LLMProvider, LLMRequest
from backend.runtime.config import RuntimeConfig
from backend.runtime.health import STATUS_DEGRADED
from backend.runtime.stream_latch import LATCH_FILE_NAME, StreamStopLatch
from backend.skills.library import create_default_skill_library
from backend.social.context import SocialContext
from backend.social.service import SocialAutonomyService
from backend.streaming import (
    MockStreamingProvider,
    StreamState,
    StreamingConfig,
    StreamingService,
)
from backend.tests.dispatch_support import confirm

SECRET = "sk-live-phase20latchsecret000000"
NOW = datetime(2026, 9, 29, tzinfo=timezone.utc)


def make_config(directory: Path) -> RuntimeConfig:
    return RuntimeConfig(
        environment="test",
        data_dir=directory / "data",
        log_dir=directory / "logs",
        temp_dir=directory / "temp",
        backup_dir=directory / "backups",
        state_dir=directory / "state",
    )


def stopped_world():
    return build_world_state(
        {
            "stream": {"is_streaming": False, "camera_available": True},
            "character": {"position": [0, 0, 0], "rotation_y": 90},
        }
    )


def evaluate(world, social: SocialContext):
    evaluator = GoalEvaluator(
        load_personality(),
        GoalEvaluatorConfig(explore_interval_seconds=0),
        clock=lambda: NOW,
    )
    return [evaluator.evaluate(world, social_context=social) for _ in range(3)]


class _Feeder:
    """Keep captured media flowing while a start request is waiting."""

    def __init__(self, service: StreamingService) -> None:
        self._service = service
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="phase20-feeder", daemon=True)

    def __enter__(self) -> "_Feeder":
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        sequence = 1
        while not self._stop.is_set():
            self._service.submit_video(b"jpg-frame", sequence=sequence)
            self._service.submit_audio(bytes(1920), sequence=sequence)
            sequence += 1
            time.sleep(0.01)


class _RecordingProvider(MockStreamingProvider):
    """Rejects every connect attempt so a start cannot become live."""

    def __init__(self, config: StreamingConfig) -> None:
        super().__init__(config, connect_failure="AUTH_FAILED")


class StreamStopLatchStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._workdir = TemporaryDirectory()
        self.addCleanup(self._workdir.cleanup)
        self.path = Path(self._workdir.name) / LATCH_FILE_NAME

    def test_missing_file_is_clear_and_not_an_error(self) -> None:
        latch = StreamStopLatch(self.path)
        self.assertFalse(latch.load())
        snapshot = latch.snapshot()
        self.assertFalse(snapshot["stopped"])
        self.assertTrue(snapshot["persistent"])
        self.assertIsNone(snapshot["load_error"])
        self.assertEqual(snapshot["write_failures"], 0)

    def test_latch_round_trips_through_disk(self) -> None:
        self.assertTrue(StreamStopLatch(self.path).record_stop(source="test"))
        restored = StreamStopLatch(self.path)
        self.assertTrue(restored.load())
        self.assertTrue(restored.stopped)
        self.assertEqual(restored.snapshot()["source"], "test")
        self.assertTrue(restored.clear(source="test"))
        self.assertFalse(StreamStopLatch(self.path).load())

    def test_latch_file_holds_one_bounded_flag_record(self) -> None:
        latch = StreamStopLatch(self.path)
        latch.record_stop(source="x" * 500)
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(
            set(payload),
            {"version", "stopped", "source", "updated_at"},
        )
        self.assertTrue(payload["stopped"])
        self.assertLessEqual(len(payload["source"]), 120)

    def test_corrupt_latch_fails_closed_and_can_be_cleared(self) -> None:
        self.path.write_text("{not json", encoding="utf-8")
        latch = StreamStopLatch(self.path)
        self.assertTrue(latch.load())
        self.assertTrue(latch.stopped)
        self.assertEqual(latch.snapshot()["load_error"], "corrupt_payload")
        latch.clear(source="operator")
        self.assertFalse(StreamStopLatch(self.path).load())

    def test_write_failure_keeps_the_in_memory_flag(self) -> None:
        blocker = Path(self._workdir.name) / "not-a-directory"
        blocker.write_text("x", encoding="utf-8")
        latch = StreamStopLatch(blocker / LATCH_FILE_NAME)
        self.assertFalse(latch.record_stop(source="test"))
        self.assertTrue(latch.stopped)
        self.assertEqual(latch.snapshot()["write_failures"], 1)

    def test_memory_only_latch_never_touches_disk(self) -> None:
        latch = StreamStopLatch()
        self.assertTrue(latch.record_stop(source="test"))
        self.assertTrue(latch.stopped)
        self.assertFalse(latch.snapshot()["persistent"])
        self.assertEqual(list(Path(self._workdir.name).iterdir()), [])


class StreamStopLatchBrainTests(unittest.TestCase):
    def setUp(self) -> None:
        self._workdir = TemporaryDirectory()
        self.addCleanup(self._workdir.cleanup)
        self.path = Path(self._workdir.name) / LATCH_FILE_NAME

    def test_deliberate_stop_survives_a_restart_and_blocks_maintain_stream(self) -> None:
        context = SocialContext(stop_latch=StreamStopLatch(self.path))
        context.update_on_action_result("STOP_STREAM", True, NOW, "viewer_01")
        self.assertTrue(context.last_intentional_stream_stop)
        self.assertTrue(json.loads(self.path.read_text(encoding="utf-8"))["stopped"])

        # "Restart": a brand new latch and context read the same state directory.
        fresh = SocialContext(stop_latch=StreamStopLatch(self.path))
        self.assertTrue(fresh.last_intentional_stream_stop)
        goals = evaluate(stopped_world(), fresh)
        self.assertTrue(all(goal.goal_type is not GoalType.MAINTAIN_STREAM for goal in goals))

    def test_explicit_start_clears_the_persisted_latch(self) -> None:
        context = SocialContext(stop_latch=StreamStopLatch(self.path))
        context.update_on_action_result("STOP_STREAM", True, NOW, "viewer_01")
        context.update_on_action_result("START_STREAM", True, NOW, "viewer_01")
        self.assertFalse(context.last_intentional_stream_stop)

        fresh = SocialContext(stop_latch=StreamStopLatch(self.path))
        self.assertFalse(fresh.last_intentional_stream_stop)
        restarted = GoalEvaluator(
            load_personality(),
            GoalEvaluatorConfig(explore_interval_seconds=100),
            clock=lambda: NOW,
        ).evaluate(stopped_world(), social_context=fresh)
        self.assertEqual(restarted.goal_type, GoalType.MAINTAIN_STREAM)

    def test_unrelated_and_failed_actions_never_change_the_latch(self) -> None:
        context = SocialContext(stop_latch=StreamStopLatch(self.path))
        context.update_on_action_result("STOP_STREAM", True, NOW, "viewer_01")
        context.update_on_action_result("WAVE", True, NOW, "viewer_01")
        context.update_on_action_result("START_STREAM", False, NOW, "viewer_01")
        context.update_on_action_result("STOP_STREAM", False, NOW, "viewer_01")
        self.assertTrue(context.last_intentional_stream_stop)

        fresh = SocialContext(stop_latch=StreamStopLatch(self.path))
        self.assertTrue(fresh.last_intentional_stream_stop)

    def test_context_without_a_latch_keeps_the_old_in_memory_behaviour(self) -> None:
        context = SocialContext()
        context.update_on_action_result("STOP_STREAM", True, NOW, "viewer_01")
        self.assertTrue(context.last_intentional_stream_stop)
        self.assertIsNone(context.snapshot()["stream_stop_latch"])

    def test_shared_durable_latch_is_the_live_source_of_truth(self) -> None:
        # PHASE 24: the admin/API control plane writes the same latch instance
        # the Brain holds, so an operator stop must gate the evaluator on the
        # next tick instead of only after a process restart.
        latch = StreamStopLatch(self.path)
        context = SocialContext(stop_latch=latch)
        self.assertFalse(context.last_intentional_stream_stop)

        latch.record_stop(source="api:/stream/stop")
        self.assertTrue(context.last_intentional_stream_stop)
        goals = evaluate(stopped_world(), context)
        self.assertTrue(all(goal.goal_type is not GoalType.MAINTAIN_STREAM for goal in goals))

        latch.clear(source="api:/stream/start")
        self.assertFalse(context.last_intentional_stream_stop)
        restarted = evaluate(stopped_world(), context)
        self.assertIn(GoalType.MAINTAIN_STREAM, [goal.goal_type for goal in restarted])

    def test_snapshot_reports_the_latch_without_secrets(self) -> None:
        context = SocialContext(stop_latch=StreamStopLatch(self.path))
        payload = context.snapshot()
        self.assertFalse(payload["stream_stop_latch"]["stopped"])
        self.assertNotIn(SECRET, json.dumps(payload))


class _FixedClock:
    def __init__(self, value: datetime = NOW) -> None:
        self.value = value

    def __call__(self) -> datetime:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += timedelta(seconds=seconds)


class _CapturedStartSelection(LLMProvider):
    """Failure fixture taken from the real runtime, not a behaviour model.

    Episode ``ep_002271`` (2026-10-05T04:02:32Z): viewer "twitch:588268196"
    asked "Can we watch again? Brows" -- intent QUESTION, ``requested_action``
    null -- and the real LLM still returned the ``start_stream`` skill, which
    went live and cleared the operator's stop latch.
    """

    payload = json.dumps(
        {
            "skill_id": "start_stream",
            "parameters": {},
            "reason": "the viewer wants to watch the stream again",
        }
    )

    def generate(self, request: LLMRequest) -> str:  # noqa: ARG002 - fixture
        return self.payload


class _DialogueStub(LLMProvider):
    """Deterministic chat answer; the plan fixture is what this test measures."""

    def generate(self, request: LLMRequest) -> str:  # noqa: ARG002 - fixture
        return json.dumps(
            {
                "speech": "Let us see.",
                "intent": "QUESTION",
                "requested_action": None,
                "priority": 0.6,
            }
        )


class DeliberateStopBrainGateTests(unittest.TestCase):
    """PHASE 24: a latched stop survives a chat/skill-driven restart.

    The evaluator already refused to propose ``MAINTAIN_STREAM`` after a
    deliberate stop, but a chat-driven plan could still reach the stream.
    """

    def setUp(self) -> None:
        self._workdir = TemporaryDirectory()
        self.addCleanup(self._workdir.cleanup)
        self.path = Path(self._workdir.name) / LATCH_FILE_NAME
        self.latch = StreamStopLatch(self.path)
        self.clock = _FixedClock()
        self.brain = AutonomousBrain(
            _CapturedStartSelection(),
            ChatService(
                _DialogueStub(),
                ChatServiceConfig(max_messages_per_rate_window=20),
                clock=self.clock,
            ),
            BrainConfig(decision_interval_seconds=0.0),
            clock=self.clock,
            social_service=SocialAutonomyService(clock=self.clock, stream_latch=self.latch),
            skill_library=create_default_skill_library(),
        )

    def test_latched_stop_blocks_a_chat_driven_start(self) -> None:
        self.assertTrue(self.latch.record_stop(source="api:/stream/stop"))
        self.brain.chat.submit("viewer_01", "Can we watch again? Brows")

        blocked = self.brain.tick(stopped_world())

        self.assertIsNone(blocked.action)
        self.assertEqual(blocked.status, BrainStatus.ERROR)
        self.assertEqual(blocked.message, "validation_failed:STREAM_INTENTIONALLY_STOPPED")
        self.assertTrue(self.latch.stopped)
        self.assertTrue(json.loads(self.path.read_text(encoding="utf-8"))["stopped"])

    def test_chat_driven_start_is_untouched_without_a_deliberate_stop(self) -> None:
        self.assertFalse(self.latch.stopped)
        self.brain.chat.submit("viewer_01", "Can we watch again? Brows")

        started = self.brain.tick(stopped_world())

        self.assertIsNotNone(started.action, started.to_dict())
        self.assertEqual(started.action.action, ActionType.START_STREAM)

    def test_operator_start_clears_the_latch_and_maintenance_resumes(self) -> None:
        self.assertTrue(self.latch.record_stop(source="api:/stream/stop"))
        self.brain.chat.submit("viewer_01", "Can we watch again? Brows")
        self.brain.tick(stopped_world())

        # Only the control plane clears the latch; the Brain cannot talk itself
        # back on air while a deliberate stop is on record.
        self.assertTrue(self.latch.clear(source="api:/stream/start"))
        resumed = self.brain.tick(stopped_world())

        self.assertIsNotNone(resumed.action, resumed.to_dict())
        self.assertEqual(resumed.action.action, ActionType.START_STREAM)
        self.assertEqual(resumed.goal.goal_type, GoalType.MAINTAIN_STREAM)

    def test_a_stop_latched_mid_flight_stops_a_retried_start(self) -> None:
        self.brain.chat.submit("viewer_01", "Can we watch again? Brows")
        first = self.brain.tick(stopped_world())
        self.assertEqual(first.action.action, ActionType.START_STREAM)

        # The operator stops the stream while Unity's start is still in flight.
        self.assertTrue(self.latch.record_stop(source="api:/stream/stop"))
        retry = self.brain.tick(
            stopped_world(),
            confirm(
                self.brain,
                ActionResult(ActionType.START_STREAM, ExecutionStatus.FAILED, "transport_error"),
            ),
        )

        self.assertIsNone(retry.action)
        self.assertEqual(retry.status, BrainStatus.ERROR)
        self.assertIn("STREAM_INTENTIONALLY_STOPPED", retry.message)
        self.assertTrue(self.latch.stopped)


class StreamRouteLatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self._workdir = TemporaryDirectory()
        self.addCleanup(self._workdir.cleanup)
        self.path = Path(self._workdir.name) / LATCH_FILE_NAME
        self.latch = StreamStopLatch(self.path)
        self.service = StreamingService(
            MockStreamingProvider(self._stream_config()),
            poll_interval_seconds=0.01,
        )
        self.addCleanup(self.service.stop, 2.0)

    @staticmethod
    def _stream_config() -> StreamingConfig:
        return StreamingConfig(
            provider="mock",
            video_queue_capacity=4,
            audio_queue_capacity=8,
            live_timeout_seconds=0.4,
            start_backoff_seconds=0.0,
            connect_timeout_seconds=0.2,
            write_timeout_seconds=0.2,
        )

    def _go_live(self) -> None:
        with _Feeder(self.service):
            status, payload = handle_stream_request(
                self.service,
                path="/stream/start",
                method="POST",
                query={"wait": ["1.5"]},
                latch=self.latch,
            )
        self.assertEqual(status, HTTPStatus.OK)
        self.assertTrue(payload["ok"], payload)
        self.assertIs(self.service.state, StreamState.LIVE)

    def test_explicit_stop_latches_and_survives_restart(self) -> None:
        self._go_live()
        status, payload = handle_stream_request(
            self.service,
            path="/stream/stop",
            method="POST",
            query={"wait": ["1.0"]},
            latch=self.latch,
        )
        self.assertEqual(status, HTTPStatus.OK)
        self.assertEqual(payload["state"], "STOPPED")
        self.assertTrue(self.latch.stopped)
        self.assertTrue(StreamStopLatch(self.path).load())
        fresh = SocialContext(stop_latch=StreamStopLatch(self.path))
        self.assertTrue(fresh.last_intentional_stream_stop)

    def test_route_stop_gates_the_live_brain_context(self) -> None:
        # PHASE 24 reproduction: the operator stops through the control plane
        # while the Brain process keeps running; without the shared latch the
        # evaluator would propose MAINTAIN_STREAM on the next quiet tick.
        context = SocialContext(stop_latch=self.latch)
        self._go_live()
        status, payload = handle_stream_request(
            self.service,
            path="/stream/stop",
            method="POST",
            query={"wait": ["1.0"]},
            latch=self.latch,
        )
        self.assertEqual(status, HTTPStatus.OK)
        self.assertEqual(payload["state"], "STOPPED")
        self.assertTrue(context.last_intentional_stream_stop)
        goals = evaluate(stopped_world(), context)
        self.assertTrue(all(goal.goal_type is not GoalType.MAINTAIN_STREAM for goal in goals))

        self._go_live()
        self.assertFalse(context.last_intentional_stream_stop)

    def test_live_start_clears_a_persisted_latch(self) -> None:
        self.latch.record_stop(source="previous_run")
        self._go_live()
        self.assertIs(self.service.state, StreamState.LIVE)
        self.assertFalse(self.latch.stopped)
        self.assertFalse(StreamStopLatch(self.path).load())

    def test_explicit_start_clears_the_latch_even_when_the_transport_times_out(self) -> None:
        # PHASE 24 reproduction: with a stopped capture pump only the control
        # plane can release the deliberate stop, and the transport start it
        # issues cannot reach LIVE until the Brain's maintenance goal re-arms
        # Unity. Keeping the latch engaged after the operator's start request
        # wedged the stream permanently (CONNECTING -> TIMEOUT -> ERROR, latch
        # still engaged, Brain gate still closed), so operator intent alone
        # clears it.
        failing = StreamingService(
            _RecordingProvider(self._stream_config()),
            poll_interval_seconds=0.01,
        )
        self.addCleanup(failing.stop, 2.0)
        self.latch.record_stop(source="previous_run")
        status, payload = handle_stream_request(
            failing,
            path="/stream/start",
            method="POST",
            query={"wait": ["0.2"]},
            latch=self.latch,
        )
        self.assertEqual(status, HTTPStatus.OK)
        self.assertFalse(payload["ok"])
        self.assertFalse(self.latch.stopped)
        self.assertEqual(self.latch.snapshot()["source"], "api:/stream/start")
        self.assertFalse(StreamStopLatch(self.path).load())

    def test_start_without_a_latch_keeps_the_old_call_shape(self) -> None:
        with _Feeder(self.service):
            status, payload = handle_stream_request(
                self.service,
                path="/stream/start",
                method="POST",
                query={"wait": ["1.5"]},
            )
        self.assertEqual(status, HTTPStatus.OK)
        self.assertTrue(payload["ok"], payload)
        self.assertIs(self.service.state, StreamState.LIVE)

    def test_shutdown_stop_never_latches(self) -> None:
        self._go_live()
        self.service.stop(2.0)
        self.assertFalse(self.latch.stopped)
        self.assertFalse(json.loads(self.path.read_text(encoding="utf-8"))["stopped"])

    def test_route_without_a_latch_keeps_the_old_call_shape(self) -> None:
        status, payload = handle_stream_request(
            self.service, path="/stream/state", method="GET"
        )
        self.assertEqual(status, HTTPStatus.OK)
        self.assertIn("state", payload)


class StreamLatchDiagnosticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._workdir = TemporaryDirectory()
        self.addCleanup(self._workdir.cleanup)
        self.config = make_config(Path(self._workdir.name))
        self.latch = StreamStopLatch(self.config.state_dir / LATCH_FILE_NAME)

    def _service(self) -> SimpleNamespace:
        return SimpleNamespace(
            brain=SimpleNamespace(provider=None),
            memory_service=None,
            runtime_metrics=lambda: {"enabled": True, "tick_in_progress": False},
        )

    def test_status_surfaces_the_latch_without_secrets(self) -> None:
        self.latch.record_stop(source="/stream/stop")
        payload = build_status(self._service(), config=self.config, stream_latch=self.latch)
        self.assertTrue(payload["stream_stop_latch"]["stopped"])
        self.assertNotIn(SECRET, json.dumps(payload))

    def test_status_without_a_latch_stays_compatible(self) -> None:
        payload = build_status(self._service(), config=self.config)
        self.assertIsNone(payload["stream_stop_latch"])

    def test_latch_persistence_fault_degrades_the_stream_component(self) -> None:
        latch = StreamStopLatch(
            Path(self._workdir.name) / "not-a-directory" / LATCH_FILE_NAME
        )
        (Path(self._workdir.name) / "not-a-directory").write_text("x", encoding="utf-8")
        latch.record_stop(source="test")
        streaming = SimpleNamespace(
            health=lambda: {"state": "STOPPED", "transport_enabled": True, "ok": True},
            get_state=lambda: {
                "state": "STOPPED",
                "transport_enabled": True,
                "queues": {},
            },
        )
        payload = build_diagnostics(
            self._service(), config=self.config, streaming=streaming, stream_latch=latch
        )
        stream = payload["health"]["components"]["stream"]
        self.assertEqual(stream["status"], STATUS_DEGRADED)
        self.assertEqual(stream["reason"], "stop_latch_persistence_fault")
        self.assertEqual(stream["detail"]["stop_latch"]["write_failures"], 1)


if __name__ == "__main__":  # pragma: no cover - unittest entry point
    unittest.main()
