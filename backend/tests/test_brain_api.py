from __future__ import annotations

import json
import threading
import unittest
import tempfile
import urllib.error
import urllib.request
from typing import Any

from backend.api.brain import create_brain_server, create_default_service
from backend.runtime import instrument
from backend.runtime.metrics import MetricsRegistry


class BrainApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self._database_dir = tempfile.TemporaryDirectory()
        database_path = f"{self._database_dir.name}/learning.db"
        self.service = create_default_service(database_path=database_path)
        self.server = create_brain_server(self.service, "127.0.0.1", 0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.host, self.port = self.server.server_address
        self.base = f"http://{self.host}:{self.port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.thread.join(timeout=2)
        self.server.server_close()
        self.service.brain.learning.close()
        self._database_dir.cleanup()

    def test_tick_returns_allowlisted_action(self) -> None:
        self.service.chat_service.submit("viewer_01", "Lacrimosa can you dance?")
        status, payload = self._post("/brain/tick", {"world": self.snapshot()})
        self.assertEqual(status, 200)
        self.assertEqual(payload["tick"]["status"], "EXECUTING")
        self.assertEqual(payload["tick"]["action"]["action"], "MOVE_TO")
        self.assertEqual(payload["tick"]["action"]["target"], "CENTER")

    def test_state_reflects_runtime(self) -> None:
        self.service.chat_service.submit("viewer_01", "dance")
        self.service.tick(self.snapshot(), None)
        status, payload = self._get("/brain/state")
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "EXECUTING")
        self.assertIsNotNone(payload["pending_action"])
        self.assertIsNotNone(payload["goal"])
        self.assertIsNotNone(payload["plan"])

    def test_invalid_result_is_rejected(self) -> None:
        status, payload = self._post(
            "/brain/tick",
            {"world": self.snapshot(), "result": {"action": "INVALID"}},
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"], "invalid_action_result")

    def test_skill_diagnostics_routes(self) -> None:
        status, payload = self._get("/skills")
        self.assertEqual(status, 200)
        self.assertTrue(payload["enabled"])
        self.assertEqual(len(payload["skills"]), 14)

        status, payload = self._get("/skills/show_table")
        self.assertEqual(status, 200)
        self.assertEqual(payload["skill"]["skill_id"], "show_table")

        status, payload = self._get("/skills/runtime")
        self.assertEqual(status, 200)
        self.assertTrue(payload["enabled"])
        self.assertIsNone(payload["runtime"]["active"])

    def test_learning_diagnostics_routes(self) -> None:
        context = "REQUEST_ACTION|DANCE"
        status, payload = self._get(f"/skills/learning/context/{urllib.parse.quote(context)}")
        self.assertEqual(status, 200)
        self.assertEqual(payload["context_key"], context)

        status, payload = self._get("/skills/learning/skill/dance")
        self.assertEqual(status, 200)
        self.assertEqual(payload["skill_id"], "dance")
        self.assertEqual(payload["contexts"], [])

    def snapshot(self) -> dict[str, Any]:
        return {
            "character": {"position": [0, 0, 0], "rotation_y": 90, "current_action": "idle"},
            "phone": {"state": "Placed", "position": [1, 0.75, 0], "placement_surface": "TABLE"},
            # PHASE 24: viewer chat is handled while the stream is live; an
            # inactive stream now preempts a saturated queue with recovery.
            "stream": {"is_streaming": True, "camera_available": True},
            "targets": {
                "CENTER": {"position": [0, 0, 0], "navigation_target": True, "look_at_target": True},
                "TABLE": {"position": [1, 0.75, 0], "navigation_target": True, "look_at_target": True},
                "CAMERA": {"position": [1, 0.8, 0], "navigation_target": False, "look_at_target": True},
            },
        }

    def _post(self, path: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        encoded = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base}{path}",
            data=encoded,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return self._request(request)

    def _get(self, path: str) -> tuple[int, dict[str, Any]]:
        return self._request(urllib.request.Request(f"{self.base}{path}", method="GET"))

    def _request(self, request: urllib.request.Request) -> tuple[int, dict[str, Any]]:
        try:
            with urllib.request.urlopen(request, timeout=2) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode("utf-8"))


class BrainTickActionTelemetryTests(unittest.TestCase):
    """PA29-003: ``brain.tick.actions`` counts ticks that *selected* an action.

    The counter used to read a non-existent ``outcome.actions`` attribute, so
    it was structurally pinned at zero even while actions were dispatched.
    """

    def setUp(self) -> None:
        self._database_dir = tempfile.TemporaryDirectory()
        database_path = f"{self._database_dir.name}/learning.db"
        self.service = create_default_service(database_path=database_path)
        self.registry = MetricsRegistry()
        instrument.set_registry(self.registry)
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        instrument.set_registry(None)
        self.service.brain.learning.close()
        self._database_dir.cleanup()

    def snapshot(self) -> dict[str, Any]:
        return {
            "character": {"position": [0, 0, 0], "rotation_y": 90, "current_action": "idle"},
            "phone": {"state": "Placed", "position": [1, 0.75, 0], "placement_surface": "TABLE"},
            "stream": {"is_streaming": True, "camera_available": True},
            "targets": {
                "CENTER": {"position": [0, 0, 0], "navigation_target": True, "look_at_target": True},
                "TABLE": {"position": [1, 0.75, 0], "navigation_target": True, "look_at_target": True},
                "CAMERA": {"position": [1, 0.8, 0], "navigation_target": False, "look_at_target": True},
            },
        }

    def test_tick_with_action_counts_exactly_one_selected_action(self) -> None:
        self.service.chat_service.submit("viewer_01", "Lacrimosa can you dance?")

        outcome = self.service.tick(self.snapshot(), None)

        self.assertIsNotNone(outcome.action)
        self.assertEqual(self.registry.counter_value("brain.ticks"), 1)
        self.assertEqual(self.registry.counter_value("brain.tick.actions"), 1)
        self.assertEqual(self.registry.counter_value("brain.tick.errors"), 0)

    def test_ticks_without_an_action_do_not_count_as_actions(self) -> None:
        first = self.service.tick(self.snapshot(), None)
        self.assertIsNotNone(first.action)
        completed = {
            "action": first.action.action.value,
            "status": "SUCCESS",
            "reason": "reported",
            "dispatch_id": first.action.dispatch_id,
        }

        # Tick 2 observes the reported success and closes the plan; tick 3
        # lands in the decision cooldown. Neither selects an action.
        second = self.service.tick(self.snapshot(), completed)
        third = self.service.tick(self.snapshot(), completed)

        self.assertIsNone(second.action)
        self.assertIsNone(third.action)
        self.assertEqual(self.registry.counter_value("brain.ticks"), 3)
        self.assertEqual(self.registry.counter_value("brain.tick.actions"), 1)


if __name__ == "__main__":
    unittest.main()
