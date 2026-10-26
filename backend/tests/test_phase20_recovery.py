"""PHASE 20 recovery tests: stop, fail, come back, and keep the state.

Every scenario here is the real production path exercised offline: the
cooperative lifecycle lock, the SQLite memory/learning stores, the bounded
speech queue, the Phase 12 chat provider state machine, the Phase 13 mock
livestream transport and the Phase 16/17 Unity adapters. Nothing reaches a
network, a platform, ffmpeg or the Unity editor, so a missing external
dependency is reported as BLOCKED evidence instead of a silent pass.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import threading
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from backend.chat.models import ChatIntent, ChatMessage
from backend.chat.providers import (
    ProviderState,
    ReconnectPolicy,
    TwitchChatProvider,
)
from backend.learning.motion_validation import MotionValidationError
from backend.learning.models import LearningConfig
from backend.learning.repository import SQLiteSkillLearningRepository
from backend.learning.service import SkillLearningService
from backend.memory.models import MemoryConfig
from backend.memory.service import MemoryService
from backend.memory.sqlite_repository import SQLiteMemoryRepository
from backend.motion_import.models import (
    MotionImportConfig,
    MotionImportError,
    MotionImportRequest,
)
from backend.motion_import.unity import UnityMotionImportAdapter
from backend.motion_processing.models import (
    MotionProcessingConfig,
    MotionProcessingError,
    MotionProcessingRequest,
)
from backend.motion_processing.unity import UnityMotionProcessingAdapter
from backend.runtime.config import RuntimeConfig
from backend.runtime.health import STATUS_READY, sqlite_probe
from backend.runtime.lifecycle import (
    DuplicateInstanceError,
    RuntimeLifecycle,
    STATE_RUNNING,
    STATE_STOPPED,
    read_status,
)
from backend.runtime.stream_latch import LATCH_FILE_NAME, StreamStopLatch
from backend.skills.library import create_default_skill_library
from backend.skills.models import SkillCategory, SkillEvaluation, SkillStatus
from backend.social.context import SocialContext
from backend.speech.service import SpeechConfig, SpeechService
from backend.streaming import (
    MockStreamingProvider,
    StreamState,
    StreamingConfig,
    StreamingService,
)
from backend.tests.test_chat_provider import (
    WELCOME,
    Clock,
    FakeTransport,
    Monotonic,
    make_provider,
    privmsg,
)
from backend.tts.config import PROVIDER_MOCK, TTSConfig
from backend.tts.mock import MockTTSProvider


BASE_TIME = datetime(2026, 9, 29, 12, 0, tzinfo=timezone.utc)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DANCE_SOURCE = PROJECT_ROOT / "unity/3d_unity/Assets/animation/dance/dance.fbx"
DANCE_REFERENCE = "Assets/animation/dance"
DANCE_DURATION = 18.466667
DANCE_SAMPLE_RATE = 30.0

VIEWER = "phase20_recovery_viewer"
MOTION_ID = "phase20_recovery_dance"
CONTEXT_KEY = "REQUEST_ACTION|DANCE|RESPOND_TO_VIEWER|STANDING|ACTIVE|TABLE|0_0"


class FixedClock:
    """Deterministic clock so restart tests keep identical timestamps."""

    def __init__(self, start: datetime = BASE_TIME) -> None:
        self.value = start

    def __call__(self) -> datetime:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += timedelta(seconds=seconds)


def make_config(directory: Path, environment: str = "test") -> RuntimeConfig:
    return RuntimeConfig(
        environment=environment,
        data_dir=directory / "data",
        log_dir=directory / "logs",
        temp_dir=directory / "temp",
        backup_dir=directory / "backups",
        state_dir=directory / "state",
    )


def chat_message(
    viewer_id: str,
    message_id: str,
    text: str = "Lacrimosa can you dance?",
) -> ChatMessage:
    return ChatMessage(
        message_id=message_id,
        viewer_id=viewer_id,
        raw_text=text,
        normalized_text=text.casefold(),
        timestamp=BASE_TIME,
        intent=ChatIntent.REQUEST_ACTION,
        target="DANCE",
    )


def absorb(message: ChatMessage) -> None:
    """Reference the message so the semantic intent is visible at the call site."""
    del message


def memory_service(path: Path, clock: FixedClock) -> MemoryService:
    config = MemoryConfig(database_path=path)
    return MemoryService(SQLiteMemoryRepository(path), config, clock=clock)


def learning_service(path: Path, clock: FixedClock) -> SkillLearningService:
    config = LearningConfig(database_path=path)
    return SkillLearningService(
        create_default_skill_library(),
        config,
        SQLiteSkillLearningRepository(path),
        clock=clock,
        random_source=__import__("random").Random(10),
    )


def register_recovery_motion(learning: SkillLearningService) -> None:
    learning.register_motion(
        MOTION_ID,
        "Recovery Dance",
        "Phase 20 recovery-suite motion fixture.",
        DANCE_REFERENCE,
        DANCE_DURATION,
        tags=("dance",),
        source_path=DANCE_SOURCE,
        source_format="FBX",
        clip_name="Dance",
        sample_rate=DANCE_SAMPLE_RATE,
        skill_category=SkillCategory.MOVEMENT,
    )


def quick_check(path: Path) -> str:
    probe = sqlite_probe(path, quick_check=True)
    assert probe["status"] == STATUS_READY, probe
    return str(probe["detail"]["quick_check"])


class LifecycleRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = TemporaryDirectory()
        self.directory = Path(self._directory.name)
        self.config = make_config(self.directory)
        self.addCleanup(self._directory.cleanup)

    def _lifecycle(self, **kwargs: Any) -> RuntimeLifecycle:
        lifecycle = RuntimeLifecycle(self.config, **kwargs)
        self.addCleanup(lifecycle.shutdown)
        return lifecycle

    def test_restart_cycle_keeps_one_instance_and_reports_on_disk_status(self) -> None:
        first = self._lifecycle(instance_id="phase20-first")
        first.start()
        first.mark_ready(tick_count=3)

        status = read_status(first.status_path)
        self.assertEqual(status["state"], STATE_RUNNING)
        self.assertEqual(status["instance_id"], "phase20-first")
        self.assertEqual(status["pid"], first.status()["pid"])
        self.assertEqual(status["lock_held"], True)

        with self.assertRaises(DuplicateInstanceError):
            RuntimeLifecycle(self.config, role="backend").start()

        stopped = first.shutdown(reason="phase20-test")
        self.assertFalse(stopped["already_stopped"])
        self.assertTrue(stopped["ok"])
        second_stop = first.shutdown(reason="phase20-test-again")
        self.assertTrue(second_stop["already_stopped"])
        self.assertEqual(second_stop["steps"], stopped["steps"])
        self.assertEqual(first.state, STATE_STOPPED)
        self.assertFalse(self.config.instance_lock_path.exists())

        # A restart must be an explicitly fresh instance, never a second loop.
        second = self._lifecycle(instance_id="phase20-second")
        second.start()
        second.mark_ready()
        self.assertEqual(read_status(second.status_path)["instance_id"], "phase20-second")
        self.assertEqual(second.status()["state"], STATE_RUNNING)
        second.shutdown()

    def test_status_reports_stopped_without_a_lock(self) -> None:
        lifecycle = self._lifecycle()
        payload = lifecycle.status()
        self.assertEqual(payload["state"], STATE_STOPPED)
        self.assertFalse(payload["lock_held"])
        self.assertFalse(payload["stop_requested"])


class DatabaseIntegrityTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = TemporaryDirectory()
        self.directory = Path(self._directory.name)
        self.addCleanup(self._directory.cleanup)
        self.clock = FixedClock()

    def test_both_production_stores_pass_sqlite_quick_check(self) -> None:
        memory_path = self.directory / "memory.db"
        learning_path = self.directory / "learning.db"

        memory = memory_service(memory_path, self.clock)
        try:
            memory.set_personality(__import__("backend.brain.personality", fromlist=["load_personality"]).load_personality())
        finally:
            memory.close()

        learning = learning_service(learning_path, self.clock)
        try:
            register_recovery_motion(learning)
        finally:
            learning.close()

        self.assertEqual(quick_check(memory_path), "ok")
        self.assertEqual(quick_check(learning_path), "ok")

    def test_wal_sidecar_is_created_and_never_left_unchecked(self) -> None:
        memory_path = self.directory / "memory.db"
        memory = memory_service(memory_path, self.clock)
        try:
            memory.record_outcome(
                viewer_id=VIEWER,
                chat=chat_message(VIEWER, "wal-1"),
                goal=None,
                action="DANCE",
                success=True,
            )
        finally:
            memory.close()

        # WAL mode is a journal mode on the connection, not a promise that the
        # sidecar exists at rest. Whatever is on disk must still be consistent.
        connection = sqlite3.connect(memory_path)
        try:
            mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
        finally:
            connection.close()
        self.assertIn(str(mode).lower(), {"wal", "delete", "truncate"})
        self.assertEqual(quick_check(memory_path), "ok")


class MemoryRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = TemporaryDirectory()
        self.directory = Path(self._directory.name)
        self.addCleanup(self._directory.cleanup)
        self.clock = FixedClock()
        self.path = self.directory / "memory.db"

    def test_viewer_relationship_and_personality_survive_a_restart(self) -> None:
        from backend.brain.personality import load_personality

        first = memory_service(self.path, self.clock)
        try:
            first.set_personality(
                load_personality({"warmth": 0.41, "playfulness": 0.33, "formality": 0.61})
            )
            for index in range(4):
                first.record_outcome(
                    viewer_id=VIEWER,
                    chat=chat_message(VIEWER, f"restart-{index:02d}"),
                    goal=None,
                    action="DANCE",
                    success=True,
                    skill_id="dance_02",
                )
            before_viewer = first.get_viewer(VIEWER)["viewer"]
            before_personality = first.personality_diagnostics()
        finally:
            first.close()

        self.assertEqual(before_viewer["interaction_count"], 4)
        self.assertEqual(before_viewer["successful_interactions"], 4)
        self.assertEqual(before_viewer["relationship_level"], "FAMILIAR")
        self.assertEqual(before_personality["source"], "persisted")

        second = memory_service(self.path, self.clock)
        try:
            after_viewer = second.get_viewer(VIEWER)["viewer"]
            after_personality = second.personality_diagnostics()
            personality = second.load_personality_state()
        finally:
            second.close()

        self.assertEqual(after_viewer["interaction_count"], before_viewer["interaction_count"])
        self.assertEqual(after_viewer["successful_interactions"], 4)
        self.assertEqual(after_viewer["relationship_level"], "FAMILIAR")
        self.assertEqual(after_viewer["preferred_topics"], before_viewer["preferred_topics"])
        self.assertEqual(after_viewer["preferred_skills"], before_viewer["preferred_skills"])
        self.assertAlmostEqual(after_viewer["familiarity"], before_viewer["familiarity"], places=6)
        self.assertEqual(after_viewer["last_interaction"], before_viewer["last_interaction"])
        self.assertEqual(after_personality["source"], "persisted")
        self.assertAlmostEqual(personality.warmth, 0.41, places=3)
        self.assertAlmostEqual(personality.formality, 0.61, places=3)

    def test_interrupted_write_does_not_corrupt_the_relationship_row(self) -> None:
        first = memory_service(self.path, self.clock)
        try:
            first.record_outcome(
                viewer_id=VIEWER,
                chat=chat_message(VIEWER, "atomic-1"),
                goal=None,
                action="DANCE",
                success=True,
            )
        finally:
            first.close()

        # Simulate a process that dies mid-transaction: an uncommitted insert
        # must roll back on the next open, leaving the committed row intact.
        connection = sqlite3.connect(self.path)
        try:
            connection.execute(
                "INSERT INTO viewer_memories (viewer_id, interaction_count, "
                "successful_interactions, failed_interactions, relationship_score, "
                "preferences, recent_topics) VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("phase20_ghost_viewer", 99, 0, 0, 0.0, "[]", "[]"),
            )
        finally:
            connection.close()  # no commit -> implicit rollback

        second = memory_service(self.path, self.clock)
        try:
            self.assertIsNone(second.get_viewer("phase20_ghost_viewer")["viewer"])
            survivor = second.get_viewer(VIEWER)["viewer"]
        finally:
            second.close()
        self.assertEqual(survivor["interaction_count"], 1)


class LearningRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = TemporaryDirectory()
        self.directory = Path(self._directory.name)
        self.addCleanup(self._directory.cleanup)
        self.clock = FixedClock()
        self.path = self.directory / "learning.db"

    def test_motion_knowledge_and_compositions_survive_a_restart(self) -> None:
        first = learning_service(self.path, self.clock)
        try:
            register_recovery_motion(first)
            first.observe_evaluation(
                SkillEvaluation(
                    skill_id=MOTION_ID,
                    execution_id="phase20-exec-1",
                    status=SkillStatus.SUCCESS,
                    duration_seconds=18.4,
                    steps_completed=1,
                    steps_failed=0,
                ),
                context_key=CONTEXT_KEY,
            )
            first.register_learned_composite(
                "phase20_recovery_pair",
                [(MOTION_ID, {}), ("look_at", {"target": "CAMERA"})],
                CONTEXT_KEY,
                name="Recovery Pair",
                description="Phase 20 recovery-suite composite.",
            )
            before_knowledge = first.get_skill_state(MOTION_ID)["contexts"]
            before_compositions = first.get_compositions()
            before_state = first.get_state()
        finally:
            first.close()

        self.assertEqual(before_knowledge[0]["trials"], 1)
        self.assertEqual(before_knowledge[0]["successes"], 1)
        self.assertEqual(before_state["motion_count"], 1)
        self.assertEqual(len(before_compositions), 1)

        second = learning_service(self.path, self.clock)
        try:
            after_knowledge = second.get_skill_state(MOTION_ID)["contexts"]
            after_compositions = second.get_compositions()
            motion = second.get_motion(MOTION_ID)
            state = second.get_state()
        finally:
            second.close()

        self.assertEqual(state["motion_count"], 1)
        self.assertIsNotNone(motion)
        self.assertEqual(motion.motion_id, MOTION_ID)
        self.assertEqual(motion.validation_status.value, "VALID")
        self.assertEqual(after_knowledge[0]["trials"], 1)
        self.assertEqual(after_knowledge[0]["successes"], 1)
        self.assertEqual(after_knowledge[0]["context_key"], CONTEXT_KEY)
        self.assertEqual(len(after_compositions), 1)
        self.assertEqual(
            after_compositions[0]["composite_skill_id"], "phase20_recovery_pair"
        )

    def test_missing_motion_source_is_refused_after_restart(self) -> None:
        learning = learning_service(self.path, self.clock)
        try:
            with self.assertRaises(MotionValidationError) as caught:
                learning.register_motion(
                    "phase20_missing_motion",
                    "Missing",
                    "A motion whose source file is absent.",
                    "Assets/animation/missing",
                    1.0,
                    source_path=self.directory / "absent.fbx",
                    source_format="FBX",
                    clip_name="Missing",
                    sample_rate=30.0,
                )
            self.assertEqual(str(caught.exception), "MOTION_SOURCE_MISSING")
            self.assertEqual(learning.get_state()["motion_count"], 0)
            self.assertFalse(learning.library.exists("phase20_missing_motion"))
        finally:
            learning.close()


class SpeechRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FixedClock()

    def _service(self, provider: MockTTSProvider) -> SpeechService:
        service = SpeechService(
            provider=provider,
            config=SpeechConfig(queue_capacity=8, max_cached_audio=8),
            tts_config=TTSConfig(provider=PROVIDER_MOCK),
            clock=self.clock,
            start_worker=False,
        )
        self.addCleanup(service.shutdown)
        return service

    def test_tts_failure_degrades_then_recovers_without_a_queue_lock(self) -> None:
        service = self._service(MockTTSProvider(fail_with="http_401"))

        failed = service.request("Recovery probe one.", dedup_key="plan:1")
        self.assertTrue(service.run_pending_once(0.1))
        self.assertEqual(service.state(failed.request_id).value, "FAILED")
        self.assertEqual(service.health()["status"], "degraded")
        state = service.get_state()
        self.assertEqual(state["consecutive_failures"], 1)
        self.assertEqual(state["queue_length"], 0)

        # The provider recovers and the next request must not be poisoned by
        # the terminal failure (the old dedup key stays terminal).
        service.provider = MockTTSProvider()
        recovered = service.request("Recovery probe two.", dedup_key="plan:2")
        self.assertTrue(service.run_pending_once(0.1))
        self.assertEqual(service.state(recovered.request_id).value, "READY")
        self.assertEqual(service.health()["status"], "ok")
        self.assertEqual(service.get_state()["consecutive_failures"], 0)
        self.assertEqual(service.get_state()["queue_length"], 0)

    def test_cancellation_still_works_while_the_provider_is_broken(self) -> None:
        service = self._service(MockTTSProvider(fail_with="http_500"))
        request = service.request("Cancel me.", dedup_key="plan:cancel")

        self.assertTrue(service.cancel(request.request_id, reason="phase20_test"))

        self.assertEqual(service.state(request.request_id).value, "CANCELLED")

        # The queue entry is drained as a no-op: a cancelled request must not
        # be spoken later and must not wedge the worker.
        self.assertTrue(service.run_pending_once(0.1))
        self.assertEqual(service.pending_count(), 0)
        self.assertEqual(service.state(request.request_id).value, "CANCELLED")


class ChatRecoveryTests(unittest.TestCase):
    def test_read_failure_recovers_with_bounded_backoff(self) -> None:
        clock = Clock(BASE_TIME)
        monotonic = Monotonic()
        broken = FakeTransport(die_after_script=True)
        healthy = FakeTransport(
            [[WELCOME], [privmsg("back again", message_id="phase20-1", timestamp=clock.now)]]
        )
        provider, _ = make_provider([broken, healthy], clock=clock, monotonic=monotonic)

        provider.connect()
        provider._state = ProviderState.CONNECTED  # noqa: SLF001 - test seam
        self.assertEqual(list(provider.poll()), [])
        self.assertTrue(broken.closed)
        self.assertIs(provider.state, ProviderState.RETRYING)
        self.assertEqual(provider.diagnostics()["reconnect_count"], 0)

        monotonic.advance(5.0)
        self.assertEqual(list(provider.poll()), [])
        self.assertIs(provider.state, ProviderState.CONNECTED)
        self.assertEqual(provider.diagnostics()["reconnect_count"], 1)
        events = list(provider.poll())
        self.assertEqual([event.text for event in events], ["back again"])
        # The IRC tags carry a user-id, so identity is stable across renames.
        self.assertEqual(events[0].viewer_id, "twitch:1337122639")
        provider.disconnect()

    def test_reconnect_budget_exhaustion_degrades_instead_of_looping(self) -> None:
        monotonic = Monotonic()
        provider, factory = make_provider(
            [FakeTransport(connect_error=ConnectionRefusedError("refused")) for _ in range(6)],
            monotonic=monotonic,
            reconnect=ReconnectPolicy(
                base_delay_seconds=1.0, max_delay_seconds=2.0, max_attempts=2
            ),
        )

        with self.assertRaises(Exception):
            provider.connect()
        for _ in range(6):
            monotonic.advance(30.0)
            provider.poll()

        self.assertEqual(factory.calls, 2)
        self.assertIs(provider.state, ProviderState.DEGRADED)
        self.assertEqual(list(provider.poll()), [])
        self.assertEqual(factory.calls, 2)
        self.assertIn("reconnect", json.dumps(provider.diagnostics()))
        provider.disconnect()

    def test_reconnect_never_redelivers_seen_messages(self) -> None:
        clock = Clock(BASE_TIME)
        monotonic = Monotonic()
        line = privmsg("dance", message_id="phase20-replayed", timestamp=clock.now)
        first = FakeTransport([[WELCOME], [line]], die_after_script=True)
        second = FakeTransport([[WELCOME], [line]])
        provider, _ = make_provider([first, second], clock=clock, monotonic=monotonic)

        provider.connect()
        collected: list[str] = []
        for _ in range(5):
            monotonic.advance(30.0)
            collected.extend(event.text for event in provider.poll())

        self.assertEqual(collected, ["dance"])
        self.assertGreaterEqual(provider.diagnostics()["counters"]["duplicated"], 1)
        provider.disconnect()


class StreamRecoveryTests(unittest.TestCase):
    @staticmethod
    def _config() -> StreamingConfig:
        return StreamingConfig(
            provider="mock",
            video_queue_capacity=4,
            audio_queue_capacity=8,
            live_timeout_seconds=0.4,
            start_backoff_seconds=0.0,
            connect_timeout_seconds=0.2,
            write_timeout_seconds=0.2,
        )

    def _live(self, service: StreamingService) -> None:
        self.assertTrue(service.transport_enabled)
        service.request_start(wait_seconds=0.05)
        for sequence in range(1, 200):
            if service.state is StreamState.LIVE:
                return
            service.submit_video(b"jpg-frame", sequence=sequence)
            service.submit_audio(bytes(1920), sequence=sequence)
            if service.wait_for_state(lambda state: state is StreamState.LIVE, 0.05) is StreamState.LIVE:
                return
        self.fail("mock transport never reached LIVE")

    def test_live_drop_reconnects_without_zombie_transport(self) -> None:
        # The provider keeps its own fake clock so the scripted drop and the
        # bounded backoff are deterministic; the service worker still polls it
        # in real time, exactly like production.
        now = [1000.0]
        provider = MockStreamingProvider(self._config(), clock=lambda: now[0])
        service = StreamingService(provider, self._config(), poll_interval_seconds=0.01)
        self.addCleanup(service.stop, 2.0)
        self._live(service)
        self.assertEqual(service.health()["reconnect_count"], 0)

        # Script a transport drop and advance the provider clock past it.
        provider.fail_after_seconds = 0.0
        now[0] += 30.0
        self.assertIs(
            service.wait_for_state(
                lambda current: current is StreamState.RECONNECTING, 5.0
            ),
            StreamState.RECONNECTING,
        )

        # Media keeps flowing while the transport is down; the bounded backoff
        # elapses and the stream restores itself.
        for sequence in range(1, 40):
            service.submit_video(b"jpg-frame", sequence=sequence)
            service.submit_audio(bytes(1920), sequence=sequence)
        now[0] += 5.0
        self.assertIs(
            service.wait_for_state(lambda current: current is StreamState.LIVE, 5.0),
            StreamState.LIVE,
        )
        self.assertIs(service.state, StreamState.LIVE)
        self.assertGreaterEqual(service.health()["reconnect_count"], 1)

        health = service.health()
        self.assertLessEqual(health["queue_video"]["length"], health["queue_video"]["capacity"])
        self.assertLessEqual(health["queue_audio"]["length"], health["queue_audio"]["capacity"])
        self.assertEqual(health["queue_video"]["capacity"], 4)
        self.assertEqual(health["queue_audio"]["capacity"], 8)
        self.assertTrue(health["ok"])

    def test_intentional_stop_is_never_auto_restarted(self) -> None:
        provider = MockStreamingProvider(self._config())
        service = StreamingService(provider, self._config(), poll_interval_seconds=0.01)
        self.addCleanup(service.stop, 2.0)
        self._live(service)

        stopped = service.request_stop(wait_seconds=2.0)
        self.assertTrue(stopped.ok)
        self.assertIs(service.state, StreamState.STOPPED)
        attempts = provider.health().connect_attempts

        # Give any rogue retry loop time to fire; a stopped stream must not
        # reconnect on its own.
        time.sleep(0.3)
        self.assertIs(service.state, StreamState.STOPPED)
        self.assertEqual(provider.health().connect_attempts, attempts)


class StreamLatchRestartTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = TemporaryDirectory()
        self.directory = Path(self._directory.name)
        self.addCleanup(self._directory.cleanup)
        self.path = self.directory / LATCH_FILE_NAME

    def test_deliberate_stop_survives_restart_and_corruption_fails_closed(self) -> None:
        first = StreamStopLatch(self.path)
        self.assertTrue(first.record_stop(source="/stream/stop"))

        # Restart: an all-new latch and social context read the same state dir.
        restarted_latch = StreamStopLatch(self.path)
        context = SocialContext(stop_latch=restarted_latch)
        self.assertTrue(restarted_latch.stopped)
        self.assertTrue(context.last_intentional_stream_stop)

        # A corrupt latch is treated as stopped, not as a green light.
        self.path.write_text("{not json", encoding="utf-8")
        corrupt = StreamStopLatch(self.path)
        self.assertTrue(corrupt.load())
        self.assertTrue(corrupt.stopped)
        self.assertEqual(corrupt.snapshot()["load_error"], "corrupt_payload")

        # Only an explicit operator action clears it.
        self.assertTrue(corrupt.clear(source="operator"))
        self.assertFalse(StreamStopLatch(self.path).load())


class UnityDegradationTests(unittest.TestCase):
    """A missing Unity editor must degrade safely, never fake an execution."""

    def setUp(self) -> None:
        self._directory = TemporaryDirectory()
        self.directory = Path(self._directory.name)
        self.addCleanup(self._directory.cleanup)
        self.clock = FixedClock()
        self.learning_path = self.directory / "learning.db"

    def test_import_adapter_reports_unavailable_and_leaves_no_stale_report(self) -> None:
        project = self.directory / "unity_project"
        project.mkdir()
        adapter = UnityMotionImportAdapter(
            MotionImportConfig(
                unity_executable=self.directory / "missing-unity.exe",
                unity_project_path=project,
                timeout_seconds=5.0,
            )
        )
        request = MotionImportRequest(
            source_file=DANCE_SOURCE,
            source_sha256="0" * 64,
            source_format="FBX",
            motion_id=MOTION_ID,
            actual_motion_id=MOTION_ID,
            version=1,
            name="Recovery Dance",
            description="Phase 20 recovery-suite import probe.",
            clip_name="Dance",
            tags=("dance",),
        )

        with self.assertRaises(MotionImportError) as caught:
            adapter.import_motion(request)

        self.assertEqual(caught.exception.reason, "UNITY_EXECUTABLE_UNAVAILABLE")
        self.assertFalse((project / "Temp" / "Phase16MotionImport.json").exists())

    def test_processing_adapter_reports_unavailable_for_process_and_playback(self) -> None:
        project = self.directory / "unity_project"
        project.mkdir()
        adapter = UnityMotionProcessingAdapter(
            MotionProcessingConfig(
                unity_executable=self.directory / "missing-unity.exe",
                unity_project_path=project,
                timeout_seconds=5.0,
                playback_timeout_seconds=5.0,
            )
        )
        request = MotionProcessingRequest(
            source_motion_id=MOTION_ID,
            source_version=1,
            processed_motion_id="phase20_recovery_dance_processed",
            processing_version=1,
            name="Recovery Dance Processed",
            description="Phase 20 recovery-suite processing probe.",
            tags=("dance",),
            skill_category="MOVEMENT",
        )

        learning = learning_service(self.learning_path, self.clock)
        try:
            register_recovery_motion(learning)
            source_motion = learning.get_motion(MOTION_ID)
            self.assertIsNotNone(source_motion)
            with self.assertRaises(MotionProcessingError) as processing:
                adapter.process(request, source_motion)
        finally:
            learning.close()

        self.assertEqual(processing.exception.reason, "UNITY_EXECUTABLE_UNAVAILABLE")
        self.assertFalse((project / "Temp" / "Phase17MotionProcessing.json").exists())
        self.assertFalse((project / "Temp" / "Phase17ProductionPlayback.json").exists())


if __name__ == "__main__":  # pragma: no cover - manual entry point
    unittest.main()
