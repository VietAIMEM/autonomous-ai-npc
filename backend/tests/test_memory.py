from __future__ import annotations

import json
import threading
import unittest
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from contextlib import contextmanager
from tempfile import TemporaryDirectory
from typing import Any

from backend.api.brain import create_brain_server, create_default_service
from backend.brain.goals import GoalEvaluator
from backend.brain.loop import AutonomousBrain, BrainConfig, BrainStatus
from backend.brain.models import (
    ActionResult,
    ActionType,
    ExecutionStatus,
    GoalType,
    PersonalityProfile,
)
from backend.brain.personality import load_personality
from backend.brain.reasoning import MockBrainLLMProvider
from backend.brain.world_state import build_world_state
from backend.chat.intent import extract_intent
from backend.chat.models import ChatMessage
from backend.chat.normalization import normalize_text
from backend.chat.priority import score_priority
from backend.chat.service import ChatService, ChatServiceConfig
from backend.llm.mock import MockLLMProvider
from backend.memory.models import (
    Episode,
    MemoryConfig,
    MemoryQuery,
    MemoryType,
)
from backend.memory.service import MemoryService
from backend.memory.sqlite_repository import SQLiteMemoryRepository
from backend.tests.dispatch_support import confirm


def message(text: str, viewer_id: str = "viewer_42", message_id: str = "msg_001") -> ChatMessage:
    normalized = normalize_text(text)
    value = ChatMessage(message_id, viewer_id, text, normalized, datetime.now(timezone.utc))
    intent = extract_intent(normalized)
    value.intent = intent.intent
    value.target = intent.target
    value.priority = score_priority(value)
    return value


def memory_service(
    root: str | Path,
    **overrides: Any,
) -> MemoryService:
    config = MemoryConfig(
        database_path=Path(root) / "memory.db",
        **overrides,
    )
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
    value = {
        "character": {"position": [0, 0, 0], "rotation_y": 90, "current_action": "idle"},
        "phone": {"state": "Placed", "position": [1, 0.75, 0], "placement_surface": "TABLE"},
        # PHASE 24: viewer chat is handled while the stream is live.
        "stream": {"is_streaming": True, "camera_available": True},
        "targets": {
            "CENTER": {"position": [0, 0, 0], "navigation_target": True, "look_at_target": True},
            "TABLE": {"position": [1, 0.75, 0], "navigation_target": True, "look_at_target": True},
            "CAMERA": {"position": [1, 0.8, 0], "navigation_target": False, "look_at_target": True},
        },
    }
    value.update(overrides)
    return value


class CapturingBrainProvider(MockBrainLLMProvider):
    def __init__(self) -> None:
        super().__init__()
        self.captured_prompt = ""

    def generate(self, request: Any) -> str:
        self.captured_prompt = request.prompt
        return super().generate(request)


class FailingMemoryRepository:
    def __getattr__(self, name: str):
        def raise_memory_error(*args: Any, **kwargs: Any):
            raise RuntimeError("database unavailable")

        return raise_memory_error


class ShortTermMemoryTests(unittest.TestCase):
    def test_buffers_remain_bounded_and_keep_context(self) -> None:
        with temporary_memory_service() as service:
            for index in range(25):
                service.remember_chat(message(f"Lacrimosa can you dance {index}?", "v", f"m{index}"))
            for index in range(35):
                service.remember_event(f"event {index}", importance=0.2)
            for index in range(25):
                service.remember_event(f"action {index}", importance=0.4)

            state = service.get_state()["short_term"]
            self.assertEqual(state["sizes"]["messages"], 20)
            self.assertEqual(state["sizes"]["events"], 30)
            self.assertNotIn("Lacrimosa can you dance 0?", state["messages"][0]["text"])
            self.assertIn("Lacrimosa can you dance 24?", state["messages"][-1]["text"])


class MemoryConsolidationTests(unittest.TestCase):
    def test_low_importance_outcome_stays_short_term(self) -> None:
        with temporary_memory_service() as service:
            service.record_outcome(
                viewer_id="viewer_42",
                chat=message("move to the table"),
                goal=None,
                action=ActionType.MOVE_TO,
                success=True,
                importance=0.3,
            )
            state = service.get_state()
            self.assertEqual(state["counts"]["episodes"], 0)
            self.assertEqual(state["counts"]["interactions"], 0)
            self.assertEqual(state["counts"]["viewer_memories"], 0)
            self.assertEqual(state["short_term"]["sizes"]["actions"], 1)

    def test_episode_and_viewer_memory_are_stored(self) -> None:
        with temporary_memory_service() as service:
            service.record_outcome(
                viewer_id="viewer_42",
                chat=message("Lacrimosa, can you dance?"),
                goal=None,
                action=ActionType.DANCE,
                success=True,
            )
            state = service.get_state()
            self.assertEqual(state["counts"]["episodes"], 1)
            self.assertEqual(state["counts"]["interactions"], 1)
            self.assertEqual(state["counts"]["viewer_memories"], 1)
            self.assertEqual(state["last_episode"]["viewer_id"], "viewer_42")

            viewer = service.get_viewer("viewer_42")["viewer"]
            self.assertEqual(viewer["interaction_count"], 1)
            self.assertEqual(viewer["successful_interactions"], 1)
            self.assertIn("dance", viewer["recent_topics"])
            self.assertGreater(viewer["relationship_score"], 0.2)

    def test_repeated_evidence_updates_semantic_memory_without_duplicates(self) -> None:
        with temporary_memory_service() as service:
            for index in range(2):
                service.record_outcome(
                    viewer_id="viewer_42",
                    chat=message("Lacrimosa, can you dance?", "viewer_42", f"msg_{index}"),
                    goal=None,
                    action=ActionType.DANCE,
                    success=True,
                )
            state = service.get_state()
            self.assertEqual(state["counts"]["episodes"], 2)
            self.assertEqual(state["counts"]["semantic_memories"], 1)
            found = service.repository.find_semantic_memory("viewer_42", "likes", "dance")
            self.assertIsNotNone(found)
            self.assertEqual(found.evidence_count, 2)

    def test_conflicting_outcome_does_not_overwrite_higher_evidence(self) -> None:
        with temporary_memory_service() as service:
            for index in range(2):
                service.record_outcome(
                    viewer_id="viewer_42",
                    chat=message("Lacrimosa, can you dance?", "viewer_42", f"msg_{index}"),
                    goal=None,
                    action=ActionType.DANCE,
                    success=True,
                )
            service.record_outcome(
                viewer_id="viewer_42",
                chat=message("Lacrimosa, can you dance?", "viewer_42", "msg_fail"),
                goal=None,
                action=ActionType.DANCE,
                success=False,
                importance=0.9,
            )
            state = service.get_state()
            self.assertEqual(state["counts"]["semantic_memories"], 1)
            found = service.repository.find_semantic_memory("viewer_42", "likes", "dance")
            self.assertIsNotNone(found)
            self.assertEqual(found.value, "dance")

    def test_interaction_history_is_retrievable(self) -> None:
        with temporary_memory_service() as service:
            service.record_outcome(
                viewer_id="viewer_42",
                chat=message("Lacrimosa, can you dance?"),
                goal=None,
                action=ActionType.DANCE,
                success=True,
            )
            records = service.repository.search_interactions(viewer_id="viewer_42")
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].requested_action, "dance")
            self.assertEqual(records[0].result, "SUCCESS")


class MemoryRetrievalTests(unittest.TestCase):
    def test_relevant_viewer_memory_ranks_first_within_top_k(self) -> None:
        with temporary_memory_service() as service:
            start = datetime(2026, 9, 9, tzinfo=timezone.utc)
            for index in range(10):
                service.repository.save_episode(
                    Episode(
                        f"ep_irrelevant_{index:02d}",
                        start + timedelta(minutes=index),
                        f"user_{index} explored the room",
                        {"location": "center"},
                        "MOVE_TO",
                        "SUCCESS",
                        0.8,
                        viewer_id=f"viewer_{index}",
                        goal="EXPLORE_ROOM",
                        location="center",
                        tags=("move",),
                    )
                )
            service.repository.save_episode(
                Episode(
                    "ep_irrelevant_same_viewer",
                    start + timedelta(minutes=10),
                    "viewer_42 explored the room",
                    {"location": "center"},
                    "MOVE_TO",
                    "SUCCESS",
                    0.8,
                    viewer_id="viewer_42",
                    goal="EXPLORE_ROOM",
                    location="center",
                    tags=("move",),
                )
            )
            service.record_outcome(
                viewer_id="viewer_42",
                chat=message("Lacrimosa, can you dance?"),
                goal=None,
                action=ActionType.DANCE,
                success=True,
            )

            results = service.retrieve(
                MemoryQuery(
                    viewer_id="viewer_42",
                    goal="RESPOND_TO_VIEWER",
                    intent="REQUEST_ACTION",
                    action="dance",
                    text="Lacrimosa, can you dance?",
                )
            )
            self.assertLessEqual(len(results), service.config.retrieval_top_k)
            self.assertTrue(results)
            self.assertEqual(results[0].viewer_id, "viewer_42")
            self.assertIn("dance", results[0].summary.lower())
            self.assertGreater(results[0].score, results[-1].score)


class MemoryPersistenceTests(unittest.TestCase):
    def test_memories_survive_repository_restart(self) -> None:
        with TemporaryDirectory() as root:
            config = MemoryConfig(database_path=Path(root) / "memory.db")
            service = MemoryService(SQLiteMemoryRepository(config.database_path), config)
            for index in range(2):
                service.record_outcome(
                    viewer_id="viewer_42",
                    chat=message("Lacrimosa, can you dance?", "viewer_42", f"msg_{index}"),
                    goal=None,
                    action=ActionType.DANCE,
                    success=True,
                )
            service.close()

            restarted = MemoryService(SQLiteMemoryRepository(config.database_path), config)
            state = restarted.get_state()
            self.assertEqual(state["counts"]["episodes"], 2)
            self.assertEqual(state["counts"]["semantic_memories"], 1)
            viewer = restarted.get_viewer("viewer_42")["viewer"]
            self.assertEqual(viewer["interaction_count"], 2)
            self.assertEqual(viewer["successful_interactions"], 2)
            self.assertIn("dance", viewer["recent_topics"])
            self.assertIn("dance", viewer["preferences"])
            restarted.close()


class MemoryFailureTests(unittest.TestCase):
    def test_repository_failure_keeps_service_and_brain_alive(self) -> None:
        service = MemoryService(FailingMemoryRepository(), MemoryConfig(database_path=Path("unused.db")))
        service.record_outcome(
            viewer_id="viewer_42",
            chat=message("Lacrimosa, can you dance?"),
            goal=None,
            action=ActionType.DANCE,
            success=True,
        )
        results = service.retrieve(MemoryQuery(viewer_id="viewer_42", text="dance"))
        state = service.get_state()
        self.assertEqual(results, [])
        self.assertTrue(state["degraded"])
        self.assertTrue(state["last_error"])

        chat = ChatService(
            MockLLMProvider(),
            ChatServiceConfig(max_messages_per_rate_window=20),
        )
        brain = AutonomousBrain(
            MockBrainLLMProvider(),
            chat,
            BrainConfig(),
            memory_service=service,
        )
        chat.submit("viewer_42", "Lacrimosa can you dance?")
        tick = brain.tick(build_world_state(world_snapshot()))
        self.assertEqual(tick.status, BrainStatus.EXECUTING)
        self.assertEqual(tick.action.action, ActionType.MOVE_TO)


class BrainMemoryIntegrationTests(unittest.TestCase):
    def test_session2_prompt_contains_recovered_memory(self) -> None:
        with TemporaryDirectory() as root:
            config = MemoryConfig(database_path=Path(root) / "memory.db")
            repository = SQLiteMemoryRepository(config.database_path)
            session1_memory = MemoryService(repository, config)
            session1_provider = CapturingBrainProvider()
            brain = self._brain(session1_provider, session1_memory)
            brain.chat.submit("viewer_42", "Lacrimosa can you dance?")

            tick = brain.tick(build_world_state(world_snapshot()))
            self.assertEqual(tick.action.action, ActionType.MOVE_TO)
            for expected in (ActionType.LOOK_AT, ActionType.DANCE, ActionType.SPEAK):
                tick = brain.tick(
                    build_world_state(world_snapshot(stream={"is_streaming": True})),
                    confirm(brain, ActionResult(tick.action.action, ExecutionStatus.SUCCESS)),
                )
                self.assertEqual(tick.action.action, expected)
            brain.tick(
                build_world_state(world_snapshot(stream={"is_streaming": True})),
                confirm(brain, ActionResult(ActionType.SPEAK, ExecutionStatus.SUCCESS)),
            )

            state = session1_memory.get_state()
            self.assertEqual(state["counts"]["episodes"], 1)
            self.assertEqual(state["counts"]["viewer_memories"], 1)

            # Two prior successful dance interactions satisfy the conservative
            # semantic-memory evidence threshold without inferring from one request.
            for index in range(2):
                session1_memory.record_outcome(
                    viewer_id="viewer_42",
                    chat=message("Lacrimosa, can you dance?", "viewer_42", f"past_msg_{index}"),
                    goal=None,
                    action=ActionType.DANCE,
                    success=True,
                )
            session1_memory.close()

            session2_memory = MemoryService(SQLiteMemoryRepository(config.database_path), config)
            session2_provider = CapturingBrainProvider()
            session2_brain = self._brain(session2_provider, session2_memory)
            session2_brain.chat.submit("viewer_42", "Can you dance again?")
            tick = session2_brain.tick(build_world_state(world_snapshot(stream={"is_streaming": True})))
            self.assertEqual(tick.status, BrainStatus.EXECUTING)
            self.assertEqual(tick.action.action, ActionType.MOVE_TO)
            prompt = session2_provider.captured_prompt
            self.assertIn("relevant_memories", prompt)
            self.assertIn("viewer_42", prompt)
            self.assertIn("dance", prompt)
            self.assertIn("likes", prompt)
            session2_memory.close()

    def _brain(self, provider: CapturingBrainProvider, memory: MemoryService) -> AutonomousBrain:
        chat = ChatService(
            MockLLMProvider(),
            ChatServiceConfig(max_messages_per_rate_window=20),
        )
        return AutonomousBrain(
            provider,
            chat,
            BrainConfig(decision_interval_seconds=100.0),
            personality=load_personality(),
            memory_service=memory,
        )


class MemoryApiTests(unittest.TestCase):
    def test_memory_state_and_viewer_routes(self) -> None:
        with TemporaryDirectory() as root:
            memory = memory_service(root)
            memory.record_outcome(
                viewer_id="viewer_42",
                chat=message("Lacrimosa, can you dance?"),
                goal=None,
                action=ActionType.DANCE,
                success=True,
            )
            service = create_default_service(memory_service=memory)
            server = create_brain_server(service, "127.0.0.1", 0)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            host, port = server.server_address
            base = f"http://{host}:{port}"
            try:
                status, state = self._get(f"{base}/memory/state")
                self.assertEqual(status, 200)
                self.assertFalse(state["degraded"])
                self.assertEqual(state["counts"]["episodes"], 1)

                status, viewer = self._get(f"{base}/memory/viewer/viewer_42")
                self.assertEqual(status, 200)
                self.assertEqual(viewer["viewer"]["viewer_id"], "viewer_42")
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
