from __future__ import annotations

import json
import threading
import unittest
import urllib.error
import urllib.request

from backend.api.chat import create_chat_server
from backend.chat.service import ChatService
from backend.llm.mock import MockLLMProvider


class ChatApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = ChatService(MockLLMProvider())
        self.server = create_chat_server(self.service, "127.0.0.1", 0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.host, self.port = self.server.server_address
        self.base = f"http://{self.host}:{self.port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.thread.join(timeout=2)
        self.server.server_close()

    def test_submit_queue_and_state(self) -> None:
        status, payload = self._post(
            "/chat/message",
            {"viewer_id": "viewer_01", "text": "Lacrimosa, can you dance?"},
        )
        self.assertEqual(status, 202)
        self.assertTrue(payload["accepted"])
        self.assertEqual(payload["message"]["intent"], "REQUEST_ACTION")
        self.assertEqual(payload["message"]["target"], "dance")

        status, payload = self._get("/chat/queue")
        self.assertEqual(status, 200)
        self.assertEqual(len(payload["messages"]), 1)
        self.assertEqual(payload["messages"][0]["status"], "QUEUED")

        status, payload = self._get("/chat/state")
        self.assertEqual(status, 200)
        self.assertEqual(payload["queued_count"], 1)
        self.assertIsNone(payload["last_response"])

        processed = self.service.process_next()
        status, payload = self._get("/chat/state")
        self.assertEqual(status, 200)
        self.assertEqual(payload["queued_count"], 0)
        self.assertEqual(payload["last_response"]["requested_action"], "dance")
        self.assertEqual(processed.message.viewer_id, "viewer_01")

    def test_invalid_api_payload(self) -> None:
        status, payload = self._post("/chat/message", {"text": "missing viewer"})
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"], "viewer_id_and_text_required")

    def test_unknown_endpoint(self) -> None:
        status, payload = self._get("/chat/unknown")
        self.assertEqual(status, 404)
        self.assertEqual(payload["error"], "not_found")

    def _post(self, path: str, body: dict[str, object]) -> tuple[int, dict[str, object]]:
        encoded = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base}{path}",
            data=encoded,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return self._request(request)

    def _get(self, path: str) -> tuple[int, dict[str, object]]:
        return self._request(urllib.request.Request(f"{self.base}{path}", method="GET"))

    def _request(self, request: urllib.request.Request) -> tuple[int, dict[str, object]]:
        try:
            with urllib.request.urlopen(request, timeout=2) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode("utf-8"))
