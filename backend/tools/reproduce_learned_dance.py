"""Reproduce learned dance selection against a copied database."""

from __future__ import annotations

import json
import shutil
import sqlite3
import tempfile
from pathlib import Path

from backend.api.brain import create_default_service
from backend.brain.loop import BrainConfig
from backend.brain.reasoning import MockBrainLLMProvider
from backend.llm.base import LLMProvider, LLMRequest
from backend.memory import MemoryConfig, MemoryService, SQLiteMemoryRepository


def _snapshot() -> dict:
    return {
        "character": {
            "position": [0, 0, 0],
            "rotation_y": 209,
            "posture": "standing",
            "current_action": "idle",
        },
        "phone": {
            "state": "Placed",
            "position": [-3.1, 0.763, 0],
            "holder": None,
            "placement_surface": "Table",
        },
        "stream": {
            "is_streaming": True,
            "camera_available": True,
            "camera_active": True,
            "duration_seconds": 1720,
            "viewer_count": 1,
        },
        "targets": {
            "CENTER": {"position": [0, 0.05, 0], "navigation_target": True, "look_at_target": True},
            "TABLE": {"position": [-3.1, 0.735, 0], "navigation_target": True, "look_at_target": True},
            "CAMERA": {"position": [-3.1, 0.766, 0], "navigation_target": False, "look_at_target": True},
        },
    }


def _result(action: str, viewer_response: str = "positive") -> dict:
    return {
        "action": action,
        "status": "SUCCESS",
        "reason": "probe_success",
        "reached": True,
        "viewer_response": viewer_response,
    }


class CapturingProvider(LLMProvider):
    def __init__(self) -> None:
        self.inner = MockBrainLLMProvider()
        self.requests: list[dict] = []

    def generate(self, request: LLMRequest) -> str:
        self.requests.append({"context": request.context, "prompt": request.prompt})
        return self.inner.generate(request)


def main() -> None:
    source = Path("backend/data/memory.db")
    context_key = "REQUEST_ACTION|DANCE|RESPOND_TO_VIEWER|STANDING|ACTIVE|TABLE|0_0"
    with tempfile.TemporaryDirectory() as directory:
        database = Path(directory) / "memory.db"
        shutil.copy2(source, database)
        memory = MemoryService(
            SQLiteMemoryRepository(database),
            MemoryConfig(database_path=database),
        )
        service = create_default_service(
            memory_service=memory,
            database_path=database,
        )
        # The API default interval is intended for the Unity loop; this probe
        # advances synchronously after each completed action.
        service.brain.config = BrainConfig(decision_interval_seconds=0.0)
        provider = CapturingProvider()
        service.brain.provider = provider
        service.brain.reasoner.provider = provider
        ranking = service.brain.learning.ranking_prompt_context(context_key)

        before = service.get_learning_skill("dance_02")
        service.chat_service.submit("viewer_probe", "Lacrimosa, dance!")
        decision = service.tick(_snapshot(), None)
        assert decision.action is not None
        dance = decision.action.to_dict()

        completion = service.tick(
            _snapshot(),
            _result(dance["action"], "positive"),
        )
        after = service.get_learning_skill("dance_02")
        current_context = service.get_learning_context(context_key)
        connection = sqlite3.connect(database)
        connection.row_factory = sqlite3.Row
        knowledge_row = dict(
            connection.execute(
                "SELECT * FROM skill_knowledge WHERE skill_id = ? AND context_key = ?",
                ("dance_02", context_key),
            ).fetchone()
        )
        connection.close()

        print(
            json.dumps(
                {
                    "decision": dance,
                    "decision_reason": decision.plan.reason,
                    "ranking_passed_to_reasoner": ranking,
                    "provider_request_context": provider.requests[-1]["context"] if provider.requests else None,
                    "completion_status": completion.status.value,
                    "dance_02_before": before,
                    "dance_02_after": after,
                    "learning_context": current_context,
                    "persisted_knowledge": knowledge_row,
                    "active_skill": service.brain.skill_executor.active.to_dict()
                    if service.brain.skill_executor.active
                    else None,
                    "recent_evaluations": service.brain.skill_executor.evaluations[-1].to_dict()
                    if service.brain.skill_executor.evaluations
                    else None,
                },
                indent=2,
            )
        )
        service.brain.learning.close()
        memory.repository.close()


if __name__ == "__main__":
    main()
