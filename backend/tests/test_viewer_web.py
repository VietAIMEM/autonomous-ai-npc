"""PHASE 22 viewer web: real intake boundary, whitelist projection, relay states."""

from __future__ import annotations

import http.client
import json
import os
import threading
import unittest
import urllib.error
import urllib.parse
import urllib.request
from unittest import mock
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory

from backend.api.brain import create_default_service
from backend.runtime.config import ConfigurationError, RuntimeConfig
from backend.viewer_web.feed import ChatFeed
from backend.viewer_web.projection import (
    build_viewer_state,
    derive_activity,
    public_viewer_count,
)
from backend.viewer_web.relay import (
    STATE_LIVE,
    STATE_RETRYING,
    STATE_STARTING,
    STATE_STOPPED,
    STATE_UNAVAILABLE,
    HlsRelay,
    parse_srt_listen_endpoint,
)
from backend.viewer_web import relay as relay_module
from backend.viewer_web.server import ViewerWebServer

POISON = "phase22-secret-sentinel-4a1f"


class _ConfiguredRelay(HlsRelay):
    """Relay with a resolvable ffmpeg so state classification can be tested."""

    @staticmethod
    def _resolve_ffmpeg(path: str):  # noqa: ARG004 - test double
        return "ffmpeg"


class _ExplodingChatService:
    def submit(self, *args, **kwargs):  # noqa: ANN002 - test double
        raise RuntimeError("internal detail that must never reach the browser")


def _poisoned_brain():
    brain = SimpleNamespace(
        state=SimpleNamespace(pending_action=None),
        world=SimpleNamespace(current_action="idle"),
    )
    brain.get_speech_summary = lambda: {
        "state": "IDLE",
        "api_key": POISON,
        "prompt": POISON,
    }
    brain.get_skills_runtime = lambda: {
        "runtime": {"active": {"skill_id": "dance", "memory": POISON}},
        "skill_knowledge": POISON,
        "provider_token": POISON,
    }
    return brain


class ViewerWebHttpTests(unittest.TestCase):
    """The real ChatService/BrainService boundary over the viewer HTTP port."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = TemporaryDirectory()
        temp = Path(cls._tmp.name)
        cls.service = create_default_service(
            database_path=temp / "learning.db",
            runtime_config=RuntimeConfig(
                environment="test",
                data_dir=temp / "data",
                temp_dir=temp / "temp",
                state_dir=temp / "state",
            ),
        )
        cls.server = ViewerWebServer(cls.service, host="127.0.0.1", port=0)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.host, cls.port = cls.server.server_address
        cls.base = f"http://{cls.host}:{cls.port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.thread.join(timeout=3)
        cls.server.server_close()
        cls.service.speech_service.shutdown()
        cls.service.brain.learning.close()
        cls._tmp.cleanup()

    # ------------------------------------------------------------- helpers

    def _post(self, path: str, body: dict) -> tuple[int, dict]:
        encoded = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base}{path}",
            data=encoded,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return self._request(request)

    def _get(self, path: str) -> tuple[int, dict]:
        request = urllib.request.Request(f"{self.base}{path}", method="GET")
        return self._request(request)

    def _request(self, request: urllib.request.Request) -> tuple[int, dict]:
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                raw = response.read().decode("utf-8")
                content_type = response.headers.get("Content-Type", "")
                if "application/json" in content_type:
                    return response.status, json.loads(raw)
                return response.status, {"raw": raw}
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8")
            try:
                return exc.code, json.loads(raw)
            except json.JSONDecodeError:
                return exc.code, {"raw": raw}

    def _state(self, viewer: str, session: str = "sess_test00000000") -> dict:
        status, payload = self._get(
            f"/viewer/api/state?viewer={viewer}&session={session}"
        )
        self.assertEqual(status, 200)
        return payload

    # -------------------------------------------------------------- viewer

    def test_state_requires_valid_viewer_id(self) -> None:
        status, payload = self._get("/viewer/api/state")
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"], "viewer_required")

        for bad in ("bad id", "../etc", "x" * 65, "!nope"):
            status, payload = self._get(
                f"/viewer/api/state?viewer={urllib.parse.quote(bad)}"
            )
            self.assertEqual(status, 400, bad)
            self.assertEqual(payload["error"], "viewer_required")

    def test_public_state_is_whitelisted(self) -> None:
        payload = self._state("viewer_01")
        self.assertEqual(
            set(payload.keys()),
            {"ok", "server_time", "stream", "activity", "viewers", "chat", "viewer"},
        )
        self.assertEqual(payload["viewer"]["id"], "viewer_01")
        self.assertIn(
            payload["activity"]["label"],
            {"Speaking", "Moving", "Dancing", "Looking", "Waving", "Active", "Live", "Idle"},
        )
        self.assertEqual(
            set(payload["stream"].keys()),
            {"connection", "live", "transport", "playback", "resolution", "fps", "since"},
        )
        self.assertFalse(payload["stream"]["live"])
        self.assertEqual(payload["stream"]["connection"], "OFFLINE")
        self.assertIsNone(payload["stream"]["playback"]["url"])
        rendered = json.dumps(payload)
        for forbidden in (
            "api_key",
            "prompt",
            "memory",
            "skill_knowledge",
            "provider",
            "database",
            "traceback",
        ):
            self.assertNotIn(forbidden, rendered.lower(), forbidden)

    def test_projection_drops_poisoned_runtime_fields(self) -> None:
        payload = build_viewer_state(
            _poisoned_brain(),
            streaming=None,
            relay_snapshot={"state": "UNAVAILABLE", "playlist_ready": False, "playlist_url": None},
            feed_snapshot={"version": 3, "messages": []},
            viewer_count=1,
            viewer_id="viewer_01",
            chat_busy=False,
            queued_count=0,
        )
        rendered = json.dumps(payload)
        self.assertNotIn(POISON, rendered)
        self.assertEqual(payload["activity"]["label"], "Dancing")

    def test_activity_derivation_uses_real_states(self) -> None:
        speaking = SimpleNamespace(
            get_speech_summary=lambda: {"state": "PLAYING"},
            get_skills_runtime=lambda: {},
            brain=SimpleNamespace(state=SimpleNamespace(pending_action=None)),
        )
        world = SimpleNamespace(current_action="idle")
        self.assertEqual(derive_activity(speaking, world, stream_live=True)["label"], "Speaking")

        dancing = SimpleNamespace(
            get_speech_summary=lambda: {"state": "IDLE"},
            get_skills_runtime=lambda: {"runtime": {"active": {"skill_id": "dance"}}},
            brain=SimpleNamespace(state=SimpleNamespace(pending_action=None)),
        )
        self.assertEqual(derive_activity(dancing, world, stream_live=True)["label"], "Dancing")

        idle = SimpleNamespace(
            get_speech_summary=lambda: {"state": "IDLE"},
            get_skills_runtime=lambda: {"runtime": {"active": None}},
            brain=SimpleNamespace(state=SimpleNamespace(pending_action=None)),
        )
        self.assertEqual(derive_activity(idle, world, stream_live=False)["label"], "Idle")
        self.assertEqual(derive_activity(idle, world, stream_live=True)["label"], "Live")

    def test_viewer_count_uses_recency_window(self) -> None:
        now = datetime.now(timezone.utc)
        social = SimpleNamespace(
            config=SimpleNamespace(conversation_timeout_seconds=180.0),
            context=SimpleNamespace(
                viewer_last_seen={
                    "twitch:recent": now - timedelta(seconds=30),
                    "twitch:stale": now - timedelta(seconds=600),
                }
            ),
        )
        self.assertEqual(public_viewer_count(social, set(), now), 1)
        self.assertEqual(public_viewer_count(social, {"viewer_01"}, now), 2)

    # ---------------------------------------------------------------- chat

    def test_chat_message_reaches_existing_intake_and_reply_is_visible(self) -> None:
        status, payload = self._post(
            "/viewer/api/chat",
            {"viewer_id": "viewer_01", "text": "Lacrimosa, wave to chat!", "client_message_id": "c_req000000000001"},
        )
        self.assertEqual(status, 202)
        self.assertTrue(payload["accepted"])
        self.assertEqual(payload["delivery"], "accepted")
        message_id = payload["message"]["id"]
        queued = {message.message_id: message for message in self.service.chat_service.get_queue()}
        self.assertIn(message_id, queued)
        self.assertEqual(queued[message_id].viewer_id, "viewer_01")

        state = self._state("viewer_01")
        self.assertEqual(state["chat"]["version"] > 0, True)
        observed = {item["id"]: item for item in state["chat"]["messages"]}
        self.assertEqual(observed[message_id]["viewer_id"], "viewer_01")
        self.assertEqual(observed[message_id]["client_id"], "c_req000000000001")

        # The real chat pipeline processes the queued message (mock LLM inside
        # the unit-test service), and its reply becomes visible to the viewer.
        processed = self.service.chat_service.process_next()
        self.assertIsNotNone(processed)
        state = self._state("viewer_01")
        replies = [item for item in state["chat"]["messages"] if item["role"] == "lacrimosa"]
        self.assertTrue(replies)
        self.assertEqual(replies[-1]["reply_to"], message_id)
        self.assertTrue(replies[-1]["text"])

    def test_duplicate_message_is_not_reported_as_success(self) -> None:
        status, first = self._post(
            "/viewer/api/chat",
            {"viewer_id": "viewer_09", "text": "hello Lacrimosa"},
        )
        self.assertEqual(status, 202)
        self.assertTrue(first["accepted"])
        status, second = self._post(
            "/viewer/api/chat",
            {"viewer_id": "viewer_09", "text": "hello Lacrimosa"},
        )
        self.assertEqual(status, 200)
        self.assertFalse(second["accepted"])
        self.assertEqual(second["delivery"], "delivered")
        self.assertEqual(second["merged_message_id"], first["message"]["id"])
        self.assertIsNone(second["message"]["reason"])
        self.assertEqual(second["message"]["delivery"], "delivered")
        self.assertIsNotNone(second["reason"])

    def test_filtered_message_reports_failure_reason(self) -> None:
        for index in range(5):
            status, payload = self._post(
                "/viewer/api/chat",
                {"viewer_id": "viewer_10", "text": f"rate probe {index} alpha{index}"},
            )
            self.assertEqual(status, 202, payload)
        status, payload = self._post(
            "/viewer/api/chat", {"viewer_id": "viewer_10", "text": "one too many"}
        )
        self.assertEqual(status, 200)
        self.assertFalse(payload["accepted"])
        self.assertEqual(payload["delivery"], "failed")
        self.assertEqual(payload["reason"], "sending too fast, try again in a moment")

    def test_client_id_replay_is_idempotent(self) -> None:
        body = {
            "viewer_id": "viewer_11",
            "text": "unique replay probe",
            "client_message_id": "c_replay0000000001",
        }
        status, first = self._post("/viewer/api/chat", body)
        self.assertEqual(status, 202)
        status, second = self._post("/viewer/api/chat", body)
        self.assertEqual(status, 202)
        self.assertTrue(second.get("duplicate_request"))
        self.assertEqual(second["message"]["id"], first["message"]["id"])
        matches = [
            message
            for message in self.service.chat_service.get_queue()
            if message.viewer_id == "viewer_11"
        ]
        self.assertEqual(len(matches), 1)

    def test_viewer_identity_isolation(self) -> None:
        status, a = self._post(
            "/viewer/api/chat", {"viewer_id": "viewer_21", "text": "alpha question"}
        )
        self.assertEqual(status, 202)
        status, b = self._post(
            "/viewer/api/chat", {"viewer_id": "viewer_22", "text": "bravo question"}
        )
        self.assertEqual(status, 202)

        state_a = self._state("viewer_21", session="sess_alice0000000")
        state_b = self._state("viewer_22", session="sess_bob00000000")
        self.assertEqual(state_a["viewer"]["id"], "viewer_21")
        self.assertEqual(state_b["viewer"]["id"], "viewer_22")
        by_id = {item["id"]: item for item in state_a["chat"]["messages"]}
        self.assertEqual(by_id[a["message"]["id"]]["viewer_id"], "viewer_21")
        self.assertEqual(by_id[b["message"]["id"]]["viewer_id"], "viewer_22")
        self.assertGreaterEqual(state_a["viewers"]["count"], 2)

    def test_generic_error_body_never_leaks_details(self) -> None:
        original = self.server.chat_service
        self.server.chat_service = _ExplodingChatService()
        try:
            status, payload = self._post(
                "/viewer/api/chat",
                {"viewer_id": "viewer_30", "text": "trigger"},
            )
        finally:
            self.server.chat_service = original
        self.assertEqual(status, 500)
        self.assertEqual(payload, {"error": "unavailable"})

    # -------------------------------------------------------------- assets

    def test_static_assets_and_traversal_rejected(self) -> None:
        status, page = self._get("/")
        self.assertEqual(status, 200)
        self.assertIn("Lacrimosa", page["raw"])

        status, state_js = self._get("/app-state.js")
        self.assertEqual(status, 200)
        self.assertIn("ViewerAppState", state_js["raw"])

        status, hls = self._get("/vendor/hls.min.js")
        self.assertEqual(status, 200)
        self.assertGreater(len(hls["raw"]), 100_000)

        connection = http.client.HTTPConnection(self.host, self.port, timeout=5)
        try:
            for target in (
                "/hls/../server.py",
                "/hls/..%2Fserver.py",
                "/static/../server.py",
                "/backend/api/brain.py",
            ):
                connection.request("GET", target)
                response = connection.getresponse()
                response.read()
                self.assertEqual(response.status, 404, target)
        finally:
            connection.close()

    def test_hls_route_unavailable_without_relay(self) -> None:
        status, payload = self._get("/hls/live.m3u8")
        self.assertEqual(status, 404)
        self.assertEqual(payload["error"], "not_found")


class HlsRouteWithoutPlaylistTests(unittest.TestCase):
    """A configured relay with no media yet reports stream_unavailable."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = TemporaryDirectory()
        temp = Path(cls._tmp.name)
        cls.service = create_default_service(
            database_path=temp / "learning.db",
            runtime_config=RuntimeConfig(
                environment="test",
                data_dir=temp / "data",
                temp_dir=temp / "temp",
                state_dir=temp / "state",
            ),
        )
        relay = _ConfiguredRelay(
            listen_url="srt://127.0.0.1:9000",
            directory=temp / "hls",
        )
        cls.server = ViewerWebServer(cls.service, host="127.0.0.1", port=0, relay=relay)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.host, cls.port = cls.server.server_address

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.thread.join(timeout=3)
        cls.server.server_close()
        cls.service.speech_service.shutdown()
        cls.service.brain.learning.close()
        cls._tmp.cleanup()

    def test_missing_playlist_is_stream_unavailable(self) -> None:
        request = urllib.request.Request(
            f"http://{self.host}:{self.port}/hls/live.m3u8", method="GET"
        )
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                self.fail(f"expected 404, got {response.status}")
        except urllib.error.HTTPError as exc:
            self.assertEqual(exc.code, 404)
            payload = json.loads(exc.read().decode("utf-8"))
        self.assertEqual(payload["error"], "stream_unavailable")


class RelayClassificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.relay = _ConfiguredRelay(
            listen_url="srt://127.0.0.1:9000",
            directory=Path(self._tmp.name) / "hls",
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_endpoint_parsing(self) -> None:
        self.assertEqual(parse_srt_listen_endpoint("srt://127.0.0.1:9000"), ("127.0.0.1", 9000))
        self.assertEqual(parse_srt_listen_endpoint("srt://example.test"), ("example.test", 9000))
        self.assertIsNone(parse_srt_listen_endpoint("rtmp://127.0.0.1/live"))
        self.assertIsNone(parse_srt_listen_endpoint(""))
        self.assertIsNone(parse_srt_listen_endpoint("srt://host:99999"))

    def test_state_machine(self) -> None:
        self.assertTrue(self.relay.configured)
        self.assertEqual(
            self.relay._classify(alive=True, age_seconds=1.0, manifest_age_seconds=None, segment_count=0),
            (STATE_STARTING, "waiting_for_stream"),
        )
        self.assertEqual(
            self.relay._classify(alive=True, age_seconds=30.0, manifest_age_seconds=None, segment_count=0),
            (STATE_RETRYING, "no_media"),
        )
        self.assertEqual(
            self.relay._classify(alive=True, age_seconds=30.0, manifest_age_seconds=0.5, segment_count=4),
            (STATE_LIVE, "receiving"),
        )
        self.assertEqual(
            self.relay._classify(alive=False, age_seconds=None, manifest_age_seconds=None, segment_count=0),
            (STATE_STOPPED, "relay_not_running"),
        )

    def test_unconfigured_relay_is_honest(self) -> None:
        relay = HlsRelay(ffmpeg_path="missing-ffmpeg-binary-xyz", listen_url="srt://127.0.0.1:9000")
        snapshot = relay.snapshot()
        self.assertEqual(snapshot["state"], STATE_UNAVAILABLE)
        self.assertFalse(snapshot["configured"])
        self.assertIsNone(snapshot["playlist_url"])

        wrong_transport = HlsRelay(listen_url="rtmp://127.0.0.1/live")
        self.assertFalse(wrong_transport.configured)
        self.assertEqual(wrong_transport.snapshot()["state"], STATE_UNAVAILABLE)


class RelayOrphanCleanupTests(unittest.TestCase):
    """PA29-015: only the viewer HLS relay may be reclaimed after a crash."""

    class _Process:
        def __init__(
            self,
            pid: int,
            *,
            name: str = "ffmpeg.exe",
            command: str = "",
            ppid: int = 1,
        ) -> None:
            self.pid = pid
            self._name = name
            self._command = command
            self._ppid = ppid
            self.terminated = False
            self.killed = False
            self.waited = False

        def name(self) -> str:
            return self._name

        def cmdline(self) -> list[str]:
            return ["C:\\tools\\ffmpeg.exe", "-i", "srt://127.0.0.1:9000", self._command]

        def ppid(self) -> int:
            return self._ppid

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            self.killed = True

        def wait(self, timeout: float) -> int:
            self.waited = True
            return 0

    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.directory = Path(self._tmp.name) / "viewer_web" / "hls_8770"
        self.directory.mkdir(parents=True)
        self.addCleanup(self._tmp.cleanup)

    def _owned_process(self, pid: int = 1234, *, ppid: int = 1) -> "RelayOrphanCleanupTests._Process":
        return self._Process(pid, command=str(self.directory), ppid=ppid)

    def test_owned_orphan_is_terminated(self) -> None:
        process = self._owned_process()
        reaped = relay_module.reap_orphan_relays(
            self.directory,
            exclude_pid=os.getpid(),
            enumerate_processes=lambda: [process],
            process_factory=lambda _pid: process,
        )

        self.assertEqual(reaped, [1234])
        self.assertTrue(process.terminated)
        self.assertTrue(process.waited)
        self.assertFalse(process.killed)

    def test_unrelated_ffmpeg_is_untouched(self) -> None:
        unrelated = self._Process(2345, command="C:\\stream\\encoder\\live")
        vendor = self._Process(3456, command="C:\\vendor\\relay\\hls_9000")
        reaped = relay_module.reap_orphan_relays(
            self.directory,
            exclude_pid=os.getpid(),
            enumerate_processes=lambda: [unrelated, vendor],
            process_factory=lambda _pid: (_ for _ in ()).throw(AssertionError("must not recheck")),
        )

        self.assertEqual(reaped, [])
        self.assertFalse(unrelated.terminated)
        self.assertFalse(vendor.terminated)

    def test_current_backend_relay_is_protected(self) -> None:
        process = self._owned_process(ppid=os.getpid())
        reaped = relay_module.reap_orphan_relays(
            self.directory,
            exclude_pid=os.getpid(),
            enumerate_processes=lambda: [process],
            process_factory=lambda _pid: process,
        )

        self.assertEqual(reaped, [])
        self.assertFalse(process.terminated)

    def test_pid_reuse_is_rejected_by_command_line(self) -> None:
        candidate = self._owned_process(4321)
        replacement = self._Process(4321, name="ffmpeg.exe", command="C:\\not-the-relay")
        reaped = relay_module.reap_orphan_relays(
            self.directory,
            exclude_pid=os.getpid(),
            enumerate_processes=lambda: [candidate],
            process_factory=lambda _pid: replacement,
        )

        self.assertEqual(reaped, [])
        self.assertFalse(replacement.terminated)

    def test_default_pid_recheck_acquires_current_process(self) -> None:
        candidate = self._owned_process(6543)
        replacement = self._Process(6543, name="ffmpeg.exe", command="C:\\not-the-relay")
        with mock.patch.object(
            relay_module, "_psutil_process_for_pid", return_value=replacement
        ) as factory:
            reaped = relay_module.reap_orphan_relays(
                self.directory,
                exclude_pid=os.getpid(),
                enumerate_processes=lambda: [candidate],
            )

        factory.assert_called_once_with(6543)
        self.assertEqual(reaped, [])
        self.assertFalse(replacement.terminated)

    def test_missing_process_is_tolerated(self) -> None:
        import psutil

        candidate = self._owned_process(5432)
        reaped = relay_module.reap_orphan_relays(
            self.directory,
            exclude_pid=os.getpid(),
            enumerate_processes=lambda: [candidate],
            process_factory=lambda _pid: (_ for _ in ()).throw(psutil.NoSuchProcess(5432)),
        )

        self.assertEqual(reaped, [])
        self.assertFalse(candidate.terminated)

    def test_repeated_cleanup_is_a_no_op(self) -> None:
        first = relay_module.reap_orphan_relays(
            self.directory,
            exclude_pid=os.getpid(),
            enumerate_processes=lambda: [],
            process_factory=lambda _pid: self._owned_process(),
        )
        second = relay_module.reap_orphan_relays(
            self.directory,
            exclude_pid=os.getpid(),
            enumerate_processes=lambda: [],
            process_factory=lambda _pid: self._owned_process(),
        )

        self.assertEqual(first, [])
        self.assertEqual(second, [])

    def test_missing_ffmpeg_still_reaps_before_disabled_start(self) -> None:
        relay = HlsRelay(
            ffmpeg_path="missing-ffmpeg-binary-xyz",
            listen_url="srt://127.0.0.1:9000",
            directory=self.directory,
        )
        with mock.patch.object(
            relay_module, "reap_orphan_relays", return_value=[1234]
        ) as reaper:
            self.assertFalse(relay.start())

        reaper.assert_called_once_with(self.directory, exclude_pid=os.getpid())
        self.assertEqual(relay.snapshot()["state"], STATE_UNAVAILABLE)

    def test_disabled_relay_still_reaps(self) -> None:
        relay = HlsRelay(
            listen_url="srt://127.0.0.1:9000",
            directory=self.directory,
            enabled=False,
        )
        with mock.patch.object(
            relay_module, "reap_orphan_relays", return_value=[]
        ) as reaper:
            self.assertFalse(relay.start())

        reaper.assert_called_once_with(self.directory, exclude_pid=os.getpid())
        self.assertEqual(relay.snapshot()["state"], STATE_UNAVAILABLE)


class ChatFeedTests(unittest.TestCase):
    def _message(self, message_id: str, viewer: str, text: str, status: str = "QUEUED"):
        return {
            "message_id": message_id,
            "viewer_id": viewer,
            "text": text,
            "status": status,
            "timestamp": datetime.now(timezone.utc),
            "filtered_reason": None,
        }

    def test_dedupe_version_and_capacity(self) -> None:
        feed = ChatFeed(capacity=10)
        first = [self._message(f"m{index}", "viewer_01", f"text {index}") for index in range(5)]
        self.assertTrue(feed.observe(first))
        self.assertEqual(feed.version, 1)
        self.assertFalse(feed.observe(first))
        self.assertEqual(feed.version, 1)
        messages = [self._message(f"n{index}", "viewer_01", f"other {index}") for index in range(15)]
        self.assertTrue(feed.observe(messages))
        snapshot = feed.snapshot()
        self.assertEqual(len(snapshot["messages"]), 10)
        self.assertEqual(snapshot["messages"][0]["id"], "n5")

    def test_reply_projection_minimal(self) -> None:
        feed = ChatFeed()
        feed.observe(
            [self._message("m1", "viewer_01", "hi")],
            {
                "speech": "Hello!",
                "intent": "CHAT",
                "requested_action": None,
                "priority": 1,
                "message_id": "m1",
                "viewer_id": "viewer_01",
            },
        )
        items = {item["id"]: item for item in feed.snapshot()["messages"]}
        reply = items["resp:m1"]
        self.assertEqual(reply["role"], "lacrimosa")
        self.assertEqual(reply["text"], "Hello!")
        self.assertEqual(reply["reply_to"], "m1")
        self.assertEqual(reply["delivery"], "delivered")

    def test_snapshot_sorted_by_timestamp(self) -> None:
        feed = ChatFeed(capacity=10)
        base = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        def stamped(message_id: str, seconds: int) -> dict[str, object]:
            item = self._message(message_id, "viewer_01", message_id)
            item["timestamp"] = base + timedelta(seconds=seconds)
            return item

        # Real discovery order: a message processed straight through the queue
        # can be recorded via SocialContext before an older queue row is first
        # observed. The transcript must still render chronologically.
        self.assertTrue(feed.observe([stamped("newer", 40)]))
        self.assertTrue(feed.observe([stamped("late_older", 5)]))
        ids = [item["id"] for item in feed.snapshot()["messages"]]
        self.assertEqual(ids, ["late_older", "newer"])

        # Same timestamps keep discovery order, and rows without a usable
        # stamp stay attached to the row discovered before them.
        self.assertTrue(feed.observe([stamped("tie_second", 40)]))
        self.assertTrue(feed.observe([{"message_id": "no_stamp", "viewer_id": "viewer_01", "text": "x", "status": "QUEUED"}]))
        ids = [item["id"] for item in feed.snapshot()["messages"]]
        self.assertEqual(ids, ["late_older", "newer", "tie_second", "no_stamp"])


class ViewerRuntimeConfigTests(unittest.TestCase):
    def test_viewer_port_environment(self) -> None:
        config = RuntimeConfig.from_environment({"VIEWER_PORT": "8899"})
        self.assertEqual(config.viewer_port, 8899)
        self.assertEqual(config.to_dict()["viewer_port"], 8899)
        self.assertEqual(RuntimeConfig().viewer_port, 8770)

    def test_invalid_viewer_port_rejected(self) -> None:
        with self.assertRaises(ConfigurationError):
            RuntimeConfig(viewer_port=0)
        with self.assertRaises(ConfigurationError):
            RuntimeConfig(viewer_port=70000)


if __name__ == "__main__":
    unittest.main()
