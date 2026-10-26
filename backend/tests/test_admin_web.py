"""PHASE 23 admin dashboard: authorization boundary, projections, honest commands.

Everything here runs against the real HTTP surface on an ephemeral port: an
authenticated admin session is required for every ``/admin/api/*`` route, the
viewer port must not expose the admin namespace, settings validation is
backend-enforced and persisted, and no response may carry a secret.
"""

from __future__ import annotations

import json
import os
import socket
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from tempfile import TemporaryDirectory

from backend.admin_web import server as admin_server_module
from backend.admin_web.auth import (
    SESSION_COOKIE_NAME,
    AdminAuthenticator,
    AdminSessionStore,
)
from backend.admin_web.console import AdminConsole
from backend.admin_web.server import (
    PERMISSION_READ,
    PERMISSION_WRITE,
    ROLE_PERMISSIONS,
    AdminWebServer,
)
from backend.api.brain import create_default_service
from backend.runtime.config import RuntimeConfig
from backend.viewer_web.server import ViewerWebServer

ADMIN_TOKEN = "phase23-admin-token-5c91d0b4f7a2"
POISON = "phase23-secret-sentinel-7b21e0"

GET_ROUTES = (
    "/admin/api/status",
    "/admin/api/health",
    "/admin/api/overview",
    "/admin/api/runtime",
    "/admin/api/runtime/health",
    "/admin/api/providers",
    "/admin/api/tts",
    "/admin/api/chat",
    "/admin/api/stream",
    "/admin/api/motion",
    "/admin/api/motion/job?kind=processing&request_id=job_1",
    "/admin/api/skills",
    "/admin/api/learning",
    "/admin/api/memory",
    "/admin/api/viewers",
    "/admin/api/personality",
    "/admin/api/diagnostics",
    "/admin/api/logs",
    "/admin/api/settings",
)

POST_ROUTES = (
    "/admin/api/runtime/start",
    "/admin/api/runtime/stop",
    "/admin/api/runtime/restart",
    "/admin/api/tts/test",
    "/admin/api/chat/reconnect",
    "/admin/api/stream/start",
    "/admin/api/stream/stop",
    "/admin/api/stream/reconnect",
    "/admin/api/motion/import",
    "/admin/api/motion/generate",
    "/admin/api/motion/process",
    "/admin/api/motion/approve",
    "/admin/api/motion/register",
    "/admin/api/skills/enable",
    "/admin/api/skills/disable",
    "/admin/api/learning/reset",
    "/admin/api/settings",
)


# --------------------------------------------------------------------- helpers


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _request(
    base: str,
    method: str,
    path: str,
    *,
    payload: object = None,
    raw_body: bytes | None = None,
    content_type: str = "application/json",
    cookie: str | None = None,
    extra_headers: dict[str, str] | None = None,
) -> tuple[int, object, "urllib.request.headers.Message"]:
    headers = dict(extra_headers or {})
    if cookie:
        headers["Cookie"] = f"{SESSION_COOKIE_NAME}={cookie}"
    data: bytes | None = raw_body
    if data is None and payload is not None:
        data = json.dumps(payload).encode("utf-8")
    if data is not None:
        headers["Content-Type"] = content_type
    request = urllib.request.Request(
        f"{base}{path}", data=data, headers=headers, method=method
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            body = response.read()
            return response.status, _decode(body, response.headers), response.headers
    except urllib.error.HTTPError as exc:
        body = exc.read()
        return exc.code, _decode(body, exc.headers), exc.headers


def _decode(body: bytes, headers: object) -> object:
    content_type = ""
    if headers is not None:
        content_type = headers.get("Content-Type", "")  # type: ignore[union-attr]
    text = body.decode("utf-8", errors="replace")
    if "application/json" in content_type:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"raw": text}
    return {"raw": text}


def _multipart(
    fields: dict[str, str],
    files: dict[str, tuple[str, bytes]],
    boundary: str = "----phase23boundary4d1f",
) -> bytes:
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.append(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                f"{value}\r\n"
            ).encode("utf-8")
        )
    for name, (filename, content) in files.items():
        chunks.append(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"; '
                f'filename="{filename}"\r\n'
                "Content-Type: application/octet-stream\r\n\r\n"
            ).encode("utf-8")
            + content
            + b"\r\n"
        )
    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(chunks)


def _cookie_value(headers: object, name: str = SESSION_COOKIE_NAME) -> str | None:
    raw = None
    if headers is not None:
        raw = headers.get("Set-Cookie")  # type: ignore[union-attr]
    if not raw:
        return None
    for part in raw.split(";"):
        key, sep, value = part.strip().partition("=")
        if sep and key == name:
            return value
    return None


class _FakeProcess:
    """Stand-in for a lifecycle ``Popen`` handle; never spawns anything."""

    def __init__(self, running: bool = True) -> None:
        self._running = running

    def poll(self) -> int | None:
        return None if self._running else 0


class _FakeJob:
    def __init__(self, state: str = "PENDING", **fields: object) -> None:
        self.state = type("State", (), {"value": state})()
        self._fields = {
            "job_id": "job_phase23_test",
            "state": state,
            **fields,
        }

    def to_dict(self) -> dict:
        return dict(self._fields)


class _FakeImportService:
    """Records the exact handle the console hands to the real import boundary."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def request_import(self, source_file, motion_id, **kwargs):  # noqa: ANN001
        self.calls.append(
            {"source_file": Path(source_file), "motion_id": motion_id, **kwargs}
        )
        return _FakeJob(state="PENDING", motion_id=motion_id)

    def get_state(self) -> dict:
        return {"enabled": True, "pending": 0, "active_request": None}

    def get(self, request_id):  # noqa: ANN001
        return None


class _FakeProcessingService:
    def __init__(self, jobs: dict | None = None) -> None:
        self.jobs = dict(jobs or {})

    def get(self, request_id):  # noqa: ANN001
        return self.jobs.get(request_id)

    def get_state(self) -> dict:
        return {"enabled": True}


# ------------------------------------------------------------ primary surface


class AdminSurfaceTests(unittest.TestCase):
    """The real admin HTTP boundary plus the untouched viewer surface."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = TemporaryDirectory()
        temp = Path(cls._tmp.name)
        cls.temp = temp
        cls.config = RuntimeConfig(
            environment="test",
            data_dir=temp / "data",
            log_dir=temp / "logs",
            temp_dir=temp / "temp",
            state_dir=temp / "state",
            brain_port=_free_port(),
            admin_port=8771,
        )
        cls.service = create_default_service(
            database_path=temp / "learning.db",
            runtime_config=cls.config,
        )
        cls.console = AdminConsole(
            cls.service,
            config=cls.config,
            project_root=temp,
        )
        cls.server = AdminWebServer(
            cls.console,
            host="127.0.0.1",
            port=0,
            token=ADMIN_TOKEN,
            upload_dir=temp / "admin_uploads",
            project_root=temp,
        )
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.bound_port}"

        cls.viewer = ViewerWebServer(cls.service, host="127.0.0.1", port=0)
        cls.viewer_thread = threading.Thread(
            target=cls.viewer.serve_forever, daemon=True
        )
        cls.viewer_thread.start()
        cls.viewer_base = f"http://127.0.0.1:{cls.viewer.server_address[1]}"

    @classmethod
    def tearDownClass(cls) -> None:
        for server, thread in (
            (cls.viewer, cls.viewer_thread),
            (cls.server, cls.thread),
        ):
            server.shutdown()
            thread.join(timeout=3)
            server.server_close()
        cls.service.speech_service.shutdown()
        cls.service.brain.learning.close()
        cls._tmp.cleanup()

    # ------------------------------------------------------------- helpers

    def _login(self, token: str = ADMIN_TOKEN) -> str:
        status, payload, headers = _request(
            self.base, "POST", "/admin/api/login", payload={"token": token}
        )
        self.assertEqual(status, 200, payload)
        cookie = _cookie_value(headers)
        self.assertTrue(cookie, "login must set the session cookie")
        return cookie or ""

    def _get(self, path: str, cookie: str | None = None):
        return _request(self.base, "GET", path, cookie=cookie)

    def _post(self, path: str, payload: object = None, cookie: str | None = None):
        return _request(self.base, "POST", path, payload=payload, cookie=cookie)

    # ---------------------------------------------------------------- authz

    def test_every_admin_get_route_requires_a_session(self) -> None:
        for route in GET_ROUTES:
            status, payload, headers = self._get(route)
            self.assertEqual(status, 401, route)
            self.assertIsInstance(payload, dict)
            self.assertEqual(payload["error"], "authentication_required", route)
            self.assertTrue(payload["login_required"], route)
            self.assertEqual(
                headers.get("X-Content-Type-Options"), "nosniff", route
            )

    def test_every_admin_post_route_requires_a_session(self) -> None:
        for route in POST_ROUTES:
            status, payload, _ = self._post(route)
            self.assertEqual(status, 401, route)
            self.assertIsInstance(payload, dict)
            self.assertEqual(payload["error"], "authentication_required", route)

    def test_the_raw_token_is_not_accepted_as_a_query_parameter(self) -> None:
        status, payload, _ = self._get(
            f"/admin/api/status?token={ADMIN_TOKEN}"
        )
        self.assertEqual(status, 401)
        self.assertEqual(payload["error"], "authentication_required")

    def test_hidden_buttons_are_not_the_boundary(self) -> None:
        """A write route without a session must fail even with a valid body."""
        status, payload, _ = self._post(
            "/admin/api/settings",
            {"values": {"log_level": "DEBUG"}},
        )
        self.assertEqual(status, 401)
        self.assertEqual(payload["error"], "authentication_required")
        stored = self.console.settings.values()
        self.assertNotEqual(stored.get("log_level"), "DEBUG")

    # --------------------------------------------------------------- login

    def test_login_rejects_missing_and_wrong_tokens(self) -> None:
        status, payload, _ = self._post("/admin/api/login", {})
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"], "token_required")

        status, payload, _ = self._post(
            "/admin/api/login", {"token": "not-the-admin-token"}
        )
        self.assertEqual(status, 401)
        self.assertEqual(payload["error"], "invalid_credentials")
        self.assertNotIn(ADMIN_TOKEN, json.dumps(payload))

    def test_login_issues_httponly_strict_cookie_and_session_probe(self) -> None:
        status, payload, headers = self._post(
            "/admin/api/login", {"token": ADMIN_TOKEN}
        )
        self.assertEqual(status, 200)
        self.assertTrue(payload["authenticated"])
        self.assertEqual(payload["role"], "admin")
        self.assertIn(PERMISSION_READ, payload["permissions"])
        self.assertIn(PERMISSION_WRITE, payload["permissions"])
        self.assertNotIn(ADMIN_TOKEN, json.dumps(payload))

        raw_cookie = headers.get("Set-Cookie")
        self.assertIn("HttpOnly", raw_cookie)
        self.assertIn("SameSite=Strict", raw_cookie)
        self.assertIn("Path=/", raw_cookie)
        cookie = _cookie_value(headers)

        for probe_cookie, expected in ((None, False), (cookie, True)):
            status, payload, _ = self._get(
                "/admin/api/session", cookie=probe_cookie
            )
            self.assertEqual(status, 200)
            self.assertEqual(payload["authenticated"], expected)
            self.assertTrue(payload["login_required"])

        status, payload, _ = self._get("/admin/api/status", cookie=cookie)
        self.assertEqual(status, 200, payload)
        self.assertEqual(payload["environment"], "test")

    def test_logout_revokes_the_session(self) -> None:
        cookie = self._login()
        status, payload, headers = self._post(
            "/admin/api/logout", cookie=cookie
        )
        self.assertEqual(status, 200)
        self.assertFalse(payload["authenticated"])
        self.assertIn("Max-Age=0", headers.get("Set-Cookie"))

        status, payload, _ = self._get("/admin/api/status", cookie=cookie)
        self.assertEqual(status, 401)
        status, payload, _ = self._get("/admin/api/session", cookie=cookie)
        self.assertFalse(payload["authenticated"])

    def test_cross_origin_admin_post_is_rejected(self) -> None:
        cookie = self._login()
        status, payload, _ = _request(
            self.base,
            "POST",
            "/admin/api/settings",
            payload={"values": {"log_level": "INFO"}},
            cookie=cookie,
            extra_headers={"Origin": "https://evil.example"},
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"], "cross_origin_rejected")

    def test_secure_static_page_has_no_inline_script_and_no_token(self) -> None:
        status, page, headers = self._get("/")
        self.assertEqual(status, 200)
        html = page["raw"]
        self.assertIn("Admin Dashboard", html)
        self.assertNotIn("<script>", html.lower())
        self.assertNotIn(ADMIN_TOKEN, html)
        self.assertEqual(headers.get("X-Frame-Options"), "DENY")
        self.assertIn("script-src 'self'", headers.get("Content-Security-Policy"))
        self.assertIn("no-cache", headers.get("Cache-Control"))

        for route, marker in (
            ("/app.js", "AdminAppState"),
            ("/app-state.js", "AdminAppState"),
            ("/styles.css", ".nav"),
        ):
            status, body, _ = self._get(route)
            self.assertEqual(status, 200, route)
            self.assertIn(marker, body["raw"], route)
            self.assertNotIn(ADMIN_TOKEN, body["raw"], route)

    def test_favicon_and_unknown_paths(self) -> None:
        status, _, headers = self._get("/favicon.ico")
        self.assertEqual(status, 204)
        self.assertEqual(headers.get("Content-Length"), "0")

        for route in ("/nope", "/static/app.js", "/admin/api"):
            status, payload, _ = self._get(route)
            self.assertEqual(status, 404, route)
            self.assertEqual(payload["error"], "not_found", route)

        cookie = self._login()
        status, payload, _ = self._get("/admin/api/not-a-route", cookie=cookie)
        self.assertEqual(status, 404)
        self.assertEqual(payload["error"], "not_found")

    def test_unsupported_methods_are_rejected(self) -> None:
        cookie = self._login()
        for method in ("PUT", "PATCH", "DELETE"):
            status, payload, _ = _request(
                self.base, method, "/admin/api/settings", cookie=cookie
            )
            self.assertEqual(status, 405, method)
            self.assertEqual(payload["error"], "method_not_allowed")

    # ------------------------------------------------------ viewer separation

    def test_viewer_port_never_serves_the_admin_surface(self) -> None:
        status, payload, _ = _request(
            self.viewer_base, "GET", "/admin/api/status"
        )
        self.assertEqual(status, 404)
        self.assertEqual(payload["error"], "not_found")

        status, payload, _ = _request(
            self.viewer_base,
            "POST",
            "/admin/api/settings",
            payload={"values": {"log_level": "DEBUG"}},
        )
        self.assertEqual(status, 404)
        self.assertEqual(payload["error"], "not_found")
        self.assertNotEqual(
            self.console.settings.values().get("log_level"), "DEBUG"
        )

        # The viewer page itself is unchanged and exposes no admin namespace.
        status, page, _ = _request(self.viewer_base, "GET", "/")
        self.assertEqual(status, 200)
        self.assertNotIn("/admin/api", page["raw"])

    # ------------------------------------------------------------ projections

    def test_read_projections_are_real_and_secret_free(self) -> None:
        cookie = self._login()
        routes = (
            "/admin/api/status",
            "/admin/api/health",
            "/admin/api/overview",
            "/admin/api/runtime",
            "/admin/api/providers",
            "/admin/api/tts",
            "/admin/api/chat",
            "/admin/api/stream",
            "/admin/api/motion",
            "/admin/api/skills",
            "/admin/api/learning",
            "/admin/api/memory",
            "/admin/api/viewers",
            "/admin/api/personality",
            "/admin/api/diagnostics",
            "/admin/api/settings",
        )
        for route in routes:
            status, payload, _ = self._get(route, cookie=cookie)
            self.assertEqual(status, 200, (route, payload))
            rendered = json.dumps(payload)
            self.assertNotIn(ADMIN_TOKEN, rendered, route)
            self.assertNotIn(POISON, rendered, route)

        _, providers, _ = self._get("/admin/api/providers", cookie=cookie)
        self.assertFalse(providers["secrets_exposed"])
        _, settings, _ = self._get("/admin/api/settings", cookie=cookie)
        self.assertFalse(settings["secret_values_exposed"])
        keys = {spec["key"] for spec in settings["specs"]}
        self.assertEqual(
            keys, {"decision_interval_seconds", "learning_enabled", "log_level"}
        )
        _, overview, _ = self._get("/admin/api/overview", cookie=cookie)
        self.assertIn("components", overview)
        self.assertIn("brain", overview)
        self.assertIsInstance(overview["queues"], dict)
        _, motion, _ = self._get("/admin/api/motion", cookie=cookie)
        self.assertIn("VALID + COMPATIBLE", motion["gate"])

    def test_status_and_health_reflect_the_real_service(self) -> None:
        cookie = self._login()
        status, payload, _ = self._get("/admin/api/status", cookie=cookie)
        self.assertEqual(status, 200)
        self.assertEqual(payload["environment"], "test")
        self.assertEqual(payload["process"]["pid"], os.getpid())
        self.assertIn("health", payload)
        self.assertEqual(payload["health"]["backend"], "READY")
        self.assertTrue(payload["brain"]["enabled"])

        status, payload, _ = self._get("/admin/api/health", cookie=cookie)
        self.assertEqual(status, 200)
        self.assertIn("components", payload)
        self.assertIn("status", payload)

    def test_logs_are_bounded_filtered_and_redacted(self) -> None:
        log_dir = Path(self.config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        lines = [
            f"2026-10-03T10:00:{index:02d} INFO room_agent.brain tick {index}"
            for index in range(30)
        ]
        lines.extend(
            (
                f"2026-10-03T10:01:00 ERROR room_agent.chat password={POISON}",
                f"2026-10-03T10:01:01 ERROR room_agent.chat Authorization: Bearer {POISON}",
                f"2026-10-03T10:01:02 WARNING room_agent.tts api_key={POISON}",
            )
        )
        (log_dir / "room_agent.log").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

        cookie = self._login()
        status, payload, _ = self._get(
            "/admin/api/logs?limit=10", cookie=cookie
        )
        self.assertEqual(status, 200)
        self.assertEqual(payload["limit"], 10)
        self.assertLessEqual(payload["count"], 10)
        rendered = json.dumps(payload)
        self.assertNotIn(POISON, rendered)
        self.assertNotIn("Bearer", rendered)

        status, payload, _ = self._get(
            "/admin/api/logs?limit=99999", cookie=cookie
        )
        self.assertEqual(status, 200)
        self.assertLessEqual(payload["limit"], 500)
        self.assertLessEqual(payload["count"], 500)

        status, payload, _ = self._get(
            "/admin/api/logs?severity=ERROR&component=chat", cookie=cookie
        )
        self.assertEqual(status, 200)
        self.assertTrue(payload["entries"])
        for entry in payload["entries"]:
            self.assertEqual(entry["level"], "ERROR")
            self.assertIn("chat", entry["component"])
        self.assertNotIn(POISON, json.dumps(payload))

        status, payload, _ = self._get(
            "/admin/api/logs?operation=tick&limit=5", cookie=cookie
        )
        self.assertEqual(status, 200)
        self.assertLessEqual(payload["count"], 5)

    # ------------------------------------------------------------- settings

    def test_settings_validation_is_backend_enforced(self) -> None:
        cookie = self._login()
        cases = (
            ({"nope": 1}, "invalid_settings"),
            ({"log_level": "LOUD"}, "invalid_settings"),
            ({"learning_enabled": "maybe"}, "invalid_settings"),
            ({"decision_interval_seconds": 99999}, "invalid_settings"),
            ({"decision_interval_seconds": "abc"}, "invalid_settings"),
        )
        for values, error in cases:
            status, payload, _ = self._post(
                "/admin/api/settings", {"values": values}, cookie=cookie
            )
            self.assertEqual(status, 400, values)
            self.assertEqual(payload["error"], error, values)
            self.assertIn(next(iter(values)), payload["settings"])

        status, payload, _ = self._post(
            "/admin/api/settings", {"values": {}}, cookie=cookie
        )
        self.assertEqual(status, 400)

        stored = self.console.settings.values()
        self.assertNotIn("nope", stored)

    def test_settings_persist_and_report_honest_status(self) -> None:
        cookie = self._login()
        status, payload, _ = self._post(
            "/admin/api/settings",
            {"values": {"learning_enabled": True, "log_level": "WARNING"}},
            cookie=cookie,
        )
        self.assertEqual(status, 200, payload)
        self.assertEqual(payload["results"]["log_level"]["status"], "applied")
        self.assertTrue(payload["results"]["log_level"]["applied"])
        self.assertFalse(
            payload["results"]["learning_enabled"]["requires_restart"]
        )
        self.assertTrue(payload["values"]["learning_enabled"])

        status, payload, _ = self._post(
            "/admin/api/settings",
            {"values": {"decision_interval_seconds": 12.5}},
            cookie=cookie,
        )
        self.assertEqual(status, 200, payload)
        entry = payload["results"]["decision_interval_seconds"]
        self.assertTrue(entry["requires_restart"])
        self.assertEqual(entry["status"], "requires_restart")
        self.assertFalse(entry["applied"])

        # A brand-new console must read the persisted values from disk.
        reopened = AdminConsole(self.service, config=self.config)
        view = reopened.settings_view()
        self.assertEqual(view["values"]["decision_interval_seconds"], 12.5)
        self.assertEqual(view["values"]["log_level"], "WARNING")
        self.assertEqual(view["values"]["learning_enabled"], True)
        self.assertEqual(view["applied"]["log_level"], "WARNING")
        self.assertTrue(view["applied"]["learning_enabled"])
        self.assertTrue(
            (Path(self.config.state_dir) / "admin_settings.json").is_file()
        )

        # Restore the shared process log level for the remaining tests.
        status, payload, _ = self._post(
            "/admin/api/settings",
            {"values": {"log_level": "INFO"}},
            cookie=cookie,
        )
        self.assertEqual(status, 200, payload)
        self.assertEqual(payload["results"]["log_level"]["status"], "applied")

    def test_learning_reset_requires_write_permission(self) -> None:
        cookie = self._login()
        with _read_only_permissions():
            status, payload, _ = self._post(
                "/admin/api/learning/reset", cookie=cookie
            )
        self.assertEqual(status, 403)
        self.assertEqual(payload["error"], "permission_denied")
        self.assertEqual(payload["permission"], PERMISSION_WRITE)

    # ------------------------------------------------------- motion plumbing

    def test_motion_source_path_traversal_is_rejected(self) -> None:
        cookie = self._login()
        for source in (
            "../outside.fbx",
            "nested/../../outside.fbx",
            str(Path(self.temp).resolve()),
            "C:/Windows/win.ini",
        ):
            status, payload, _ = self._post(
                "/admin/api/motion/import",
                {"source_path": source, "motion_id": "walk_probe"},
                cookie=cookie,
            )
            self.assertEqual(status, 400, source)
            self.assertEqual(payload["error"], "invalid_source_path", source)

        status, payload, _ = self._post(
            "/admin/api/motion/import",
            {"source_path": "missing/walk.fbx", "motion_id": "walk_probe"},
            cookie=cookie,
        )
        self.assertEqual(status, 404)
        self.assertEqual(payload["error"], "source_file_not_found")

        asset = Path(self.temp) / "assets" / "probe.fbx"
        asset.parent.mkdir(parents=True, exist_ok=True)
        asset.write_bytes(b"FAKE-FBX")
        status, payload, _ = self._post(
            "/admin/api/motion/import",
            {"source_path": "assets/probe.fbx", "motion_id": "BAD ID!"},
            cookie=cookie,
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"], "invalid_motion_id")

    def test_motion_import_unavailable_is_reported_honestly(self) -> None:
        """Without MOTION_IMPORT_ENABLED there is no service, so no success."""
        cookie = self._login()
        asset = Path(self.temp) / "assets"
        asset.mkdir(parents=True, exist_ok=True)
        (asset / "walk.fbx").write_bytes(b"FAKE-FBX")
        status, payload, _ = self._post(
            "/admin/api/motion/import",
            {"source_path": "assets/walk.fbx", "motion_id": "walk_probe"},
            cookie=cookie,
        )
        self.assertEqual(status, 503)
        self.assertEqual(payload["error"], "motion_import_unavailable")


class _read_only_permissions:
    """Temporarily drop ``admin.write``; the backend must still enforce it."""

    def __enter__(self):
        self._original = dict(ROLE_PERMISSIONS)
        ROLE_PERMISSIONS.clear()
        ROLE_PERMISSIONS["admin"] = frozenset({PERMISSION_READ})
        return self

    def __exit__(self, *exc) -> None:
        ROLE_PERMISSIONS.clear()
        ROLE_PERMISSIONS.update(self._original)


# ------------------------------------------------------------ dedicated servers


class _ServedAdminCase(unittest.TestCase):
    """Shared fixture: one real service, an ephemeral admin server per class."""

    project_root: Path
    upload_dir: Path

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = TemporaryDirectory()
        temp = Path(cls._tmp.name)
        cls.temp = temp
        cls.config = RuntimeConfig(
            environment="test",
            data_dir=temp / "data",
            log_dir=temp / "logs",
            temp_dir=temp / "temp",
            state_dir=temp / "state",
            brain_port=_free_port(),
        )
        cls.service = create_default_service(
            database_path=temp / "learning.db",
            runtime_config=cls.config,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.service.speech_service.shutdown()
        cls.service.brain.learning.close()
        cls._tmp.cleanup()

    def _serve(self, console: AdminConsole) -> tuple[AdminWebServer, str]:
        server = AdminWebServer(
            console,
            host="127.0.0.1",
            port=0,
            token=ADMIN_TOKEN,
            upload_dir=self.upload_dir,
            project_root=self.project_root,
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(self._stop, server, thread)
        return server, f"http://127.0.0.1:{server.bound_port}"

    @staticmethod
    def _stop(server: AdminWebServer, thread: threading.Thread) -> None:
        server.shutdown()
        thread.join(timeout=3)
        server.server_close()

    def _login(self, base: str) -> str:
        status, payload, headers = _request(
            base, "POST", "/admin/api/login", payload={"token": ADMIN_TOKEN}
        )
        self.assertEqual(status, 200, payload)
        return _cookie_value(headers) or ""


class AdminLoginThrottleTests(_ServedAdminCase):
    def test_login_throttle_blocks_after_five_failures(self) -> None:
        self.project_root = self.temp
        self.upload_dir = self.temp / "uploads"
        console = AdminConsole(
            self.service, config=self.config, project_root=self.temp
        )
        server, base = self._serve(console)
        server.authenticator = AdminAuthenticator(
            ADMIN_TOKEN,
            sessions=AdminSessionStore(ttl_seconds=600.0),
        )

        for attempt in range(5):
            status, payload, _ = _request(
                base, "POST", "/admin/api/login", payload={"token": "wrong-token"}
            )
            self.assertEqual(status, 401, attempt)
            self.assertEqual(payload["error"], "invalid_credentials")

        status, payload, headers = _request(
            base, "POST", "/admin/api/login", payload={"token": "wrong-token"}
        )
        self.assertEqual(status, 429)
        self.assertEqual(payload["error"], "login_throttled")
        self.assertGreater(payload["retry_after_seconds"], 0)
        self.assertGreaterEqual(int(headers.get("Retry-After")), 1)

        # Even the correct token waits out the window: the throttle is real.
        status, payload, _ = _request(
            base, "POST", "/admin/api/login", payload={"token": ADMIN_TOKEN}
        )
        self.assertEqual(status, 429)


class AdminRuntimeCommandTests(_ServedAdminCase):
    def setUp(self) -> None:
        self.project_root = self.temp
        self.upload_dir = self.temp / "uploads"
        self.console = AdminConsole(
            self.service, config=self.config, project_root=self.temp
        )
        self.server, self.base = self._serve(self.console)
        self.cookie = self._login(self.base)
        self.lock_path = Path(self.config.state_dir) / "room_agent.lock"
        # Each test starts from an honest stopped state, and the lifecycle CLI
        # is stubbed: a test must never spawn a real backend process.
        self.lock_path.unlink(missing_ok=True)
        self.addCleanup(lambda: self.lock_path.unlink(missing_ok=True))
        self.dispatches: list[str] = []

        def _fake_dispatch(action: str) -> dict:
            self.dispatches.append(action)
            return {
                "action": action,
                "pid": 424242,
                "log": "admin_lifecycle.out.log",
                "process": _FakeProcess(running=True),
            }

        self.console._dispatch_lifecycle = _fake_dispatch

    def test_stop_reports_already_stopped_without_dispatching(self) -> None:
        status, payload, _ = _request(
            self.base,
            "POST",
            "/admin/api/runtime/stop",
            payload={},
            cookie=self.cookie,
        )
        self.assertEqual(status, 200, payload)
        self.assertTrue(payload["ok"])
        self.assertFalse(payload["requested"])
        self.assertTrue(payload["already_stopped"])
        self.assertEqual(self.console._last_dispatch, {})
        self.assertEqual(self.dispatches, [])

    def test_start_dispatches_the_existing_lifecycle_cli_once(self) -> None:
        status, payload, _ = _request(
            self.base,
            "POST",
            "/admin/api/runtime/start",
            payload={},
            cookie=self.cookie,
        )
        self.assertEqual(status, 200, payload)
        self.assertTrue(payload["requested"])
        self.assertEqual(payload["dispatched_pid"], 424242)
        self.assertEqual(self.dispatches, ["start"])

        # A second command while the first is still in flight never dispatches.
        status, payload, _ = _request(
            self.base,
            "POST",
            "/admin/api/runtime/start",
            payload={},
            cookie=self.cookie,
        )
        self.assertEqual(status, 409, payload)
        self.assertEqual(payload["error"], "runtime_command_in_progress")
        self.assertEqual(self.dispatches, ["start"])

    def test_start_refuses_when_an_instance_is_alive(self) -> None:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path.write_text(
            json.dumps({"pid": os.getpid(), "started_at": None, "role": "backend"}),
            encoding="utf-8",
        )
        status, payload, _ = _request(
            self.base,
            "POST",
            "/admin/api/runtime/start",
            payload={},
            cookie=self.cookie,
        )
        self.assertEqual(status, 409, payload)
        self.assertEqual(payload["error"], "instance_already_running")
        self.assertEqual(payload["detail"]["pid"], os.getpid())
        self.assertEqual(self.console._last_dispatch, {})

    def test_second_command_is_refused_while_one_is_in_flight(self) -> None:
        self.lock_path.unlink(missing_ok=True)
        self.console._last_dispatch = {
            "action": "start",
            "pid": 4242,
            "log": "admin_lifecycle.out.log",
            "process": _FakeProcess(running=True),
        }
        status, payload, _ = _request(
            self.base,
            "POST",
            "/admin/api/runtime/start",
            payload={},
            cookie=self.cookie,
        )
        self.assertEqual(status, 409, payload)
        self.assertEqual(payload["error"], "runtime_command_in_progress")

    def test_runtime_status_projects_the_controller_honestly(self) -> None:
        status, payload, _ = _request(
            self.base, "GET", "/admin/api/runtime", cookie=self.cookie
        )
        self.assertEqual(status, 200, payload)
        self.assertIn("running", payload["runtime"])
        self.assertEqual(payload["admin"]["port"], self.config.admin_port)
        self.assertEqual(payload["admin"]["pid"], os.getpid())

    def test_unknown_runtime_action_is_rejected(self) -> None:
        status, payload, _ = _request(
            self.base,
            "POST",
            "/admin/api/runtime/pause",
            payload={},
            cookie=self.cookie,
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"], "unknown_runtime_action")


class AdminMotionImportTests(_ServedAdminCase):
    def setUp(self) -> None:
        self.project_root = self.temp
        self.upload_dir = self.temp / "admin_uploads"
        self.import_service = _FakeImportService()
        self.console = AdminConsole(
            self.service,
            config=self.config,
            motion_import=self.import_service,
            project_root=self.temp,
        )
        self.server, self.base = self._serve(self.console)
        self.cookie = self._login(self.base)

    def test_multipart_upload_preserves_the_original_file(self) -> None:
        body = _multipart(
            {"motion_id": "walk_admin_01", "name": "Walk Admin", "tags": "walk,test"},
            {"file": ("walk.fbx", b"FAKE-FBX-ORIGINAL")},
        )
        status, payload, _ = _request(
            self.base,
            "POST",
            "/admin/api/motion/import",
            raw_body=body,
            content_type="multipart/form-data; boundary=----phase23boundary4d1f",
            cookie=self.cookie,
        )
        self.assertEqual(status, 200, payload)
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["job"]["motion_id"], "walk_admin_01")

        self.assertEqual(len(self.import_service.calls), 1)
        call = self.import_service.calls[0]
        stored = call["source_file"]
        self.assertTrue(stored.is_file())
        self.assertEqual(stored.read_bytes(), b"FAKE-FBX-ORIGINAL")
        self.assertTrue(stored.name.endswith("_walk.fbx"), stored.name)
        self.assertEqual(stored.parent, self.upload_dir)
        self.assertEqual(call["motion_id"], "walk_admin_01")
        self.assertEqual(call["tags"], ["walk", "test"])
        self.assertEqual(call["name"], "Walk Admin")

        second = _multipart(
            {"motion_id": "walk_admin_02"},
            {"file": ("walk.fbx", b"FAKE-FBX-ORIGINAL")},
        )
        status, payload, _ = _request(
            self.base,
            "POST",
            "/admin/api/motion/import",
            raw_body=second,
            content_type="multipart/form-data; boundary=----phase23boundary4d1f",
            cookie=self.cookie,
        )
        self.assertEqual(status, 200, payload)
        self.assertNotEqual(
            self.import_service.calls[1]["source_file"],
            self.import_service.calls[0]["source_file"],
        )
        self.assertTrue(self.import_service.calls[0]["source_file"].is_file())

    def test_multipart_without_the_file_part_is_rejected(self) -> None:
        body = _multipart({"motion_id": "walk_admin_03"}, {})
        status, payload, _ = _request(
            self.base,
            "POST",
            "/admin/api/motion/import",
            raw_body=body,
            content_type="multipart/form-data; boundary=----phase23boundary4d1f",
            cookie=self.cookie,
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"], "motion_file_required")
        self.assertEqual(self.import_service.calls, [])

    def test_json_source_path_hands_over_the_untouched_original(self) -> None:
        assets = self.temp / "assets"
        assets.mkdir(parents=True, exist_ok=True)
        original = assets / "walk.fbx"
        original.write_bytes(b"REAL-PROJECT-ASSET")
        status, payload, _ = _request(
            self.base,
            "POST",
            "/admin/api/motion/import",
            payload={
                "source_path": "assets/walk.fbx",
                "motion_id": "walk_admin_04",
                "skill_category": "MOVEMENT",
            },
            cookie=self.cookie,
        )
        self.assertEqual(status, 200, payload)
        call = self.import_service.calls[-1]
        self.assertEqual(call["source_file"], original.resolve())
        self.assertEqual(original.read_bytes(), b"REAL-PROJECT-ASSET")

    def test_import_failure_is_structured_not_a_stack_trace(self) -> None:
        class _Exploding:
            def request_import(self, *args, **kwargs):  # noqa: ANN002
                raise RuntimeError(f"internal {POISON}")

            def get_state(self):
                return {"enabled": True}

        console = AdminConsole(
            self.service,
            config=self.config,
            motion_import=_Exploding(),
            project_root=self.temp,
        )
        _, base = self._serve(console)
        cookie = self._login(base)
        body = _multipart(
            {"motion_id": "walk_admin_05"},
            {"file": ("walk.fbx", b"FAKE-FBX")},
        )
        status, payload, _ = _request(
            base,
            "POST",
            "/admin/api/motion/import",
            raw_body=body,
            content_type="multipart/form-data; boundary=----phase23boundary4d1f",
            cookie=cookie,
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"], "motion_import_rejected")
        self.assertNotIn(POISON, json.dumps(payload))
        self.assertNotIn("Traceback", json.dumps(payload))


class AdminMotionJobTests(_ServedAdminCase):
    def setUp(self) -> None:
        self.project_root = self.temp
        self.upload_dir = self.temp / "uploads"
        job = _FakeJob(state="VALIDATED", motion_id="phase16_walking")
        self.processing = _FakeProcessingService({"req_phase23": job})
        self.console = AdminConsole(
            self.service,
            config=self.config,
            motion_processing=self.processing,
            project_root=self.temp,
        )
        self.server, self.base = self._serve(self.console)
        self.cookie = self._login(self.base)

    def test_motion_job_requires_a_request_id(self) -> None:
        status, payload, _ = _request(
            self.base,
            "GET",
            "/admin/api/motion/job?kind=processing",
            cookie=self.cookie,
        )
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"], "request_id_required")

    def test_motion_job_returns_the_service_record(self) -> None:
        status, payload, _ = _request(
            self.base,
            "GET",
            "/admin/api/motion/job?kind=processing&request_id=req_phase23",
            cookie=self.cookie,
        )
        self.assertEqual(status, 200, payload)
        self.assertEqual(payload["job"]["state"], "VALIDATED")

    def test_unknown_motion_job_is_not_found(self) -> None:
        status, payload, _ = _request(
            self.base,
            "GET",
            "/admin/api/motion/job?kind=processing&request_id=missing",
            cookie=self.cookie,
        )
        self.assertEqual(status, 404)
        self.assertEqual(payload["error"], "motion_request_not_found")

    def test_absent_motion_service_is_unavailable(self) -> None:
        status, payload, _ = _request(
            self.base,
            "GET",
            "/admin/api/motion/job?kind=import&request_id=req_phase23",
            cookie=self.cookie,
        )
        self.assertEqual(status, 503)
        self.assertEqual(payload["error"], "motion_service_unavailable")


class AdminSessionStoreTests(unittest.TestCase):
    """The session/throttle primitives behind the HTTP boundary."""

    def test_sessions_are_hashed_bounded_and_revocable(self) -> None:
        store = AdminSessionStore(ttl_seconds=600.0, capacity=4)
        issued = [store.issue()[0] for _ in range(6)]
        self.assertLessEqual(store.active_sessions, 4)
        self.assertFalse(store.validate(issued[0]))
        newest = issued[-1]
        self.assertTrue(store.validate(newest))
        self.assertTrue(store.revoke(newest))
        self.assertFalse(store.validate(newest))
        self.assertFalse(store.validate(None))

    def test_throttle_window(self) -> None:
        authenticator = AdminAuthenticator(ADMIN_TOKEN)
        for _ in range(5):
            self.assertIsNone(authenticator.login("wrong-token"))
        self.assertFalse(authenticator.throttle.allowed())
        self.assertGreater(authenticator.throttle.retry_after_seconds(), 0)
        self.assertIsNone(authenticator.login(ADMIN_TOKEN))

    def test_token_resolution_prefers_the_environment(self) -> None:
        from backend.admin_web.auth import load_admin_token

        with TemporaryDirectory() as tmp:
            token = load_admin_token(
                tmp, environ={"ROOM_AGENT_ADMIN_TOKEN": ADMIN_TOKEN}
            )
            self.assertEqual(token, ADMIN_TOKEN)
            self.assertFalse(
                (Path(tmp) / "admin_token.txt").exists(),
                "an environment token must not be written to disk",
            )

            minted = load_admin_token(tmp, environ={})
            self.assertGreaterEqual(len(minted), 16)
            self.assertEqual(
                (Path(tmp) / "admin_token.txt").read_text(encoding="utf-8").strip(),
                minted,
            )

    def test_module_permission_table_defaults_to_full_admin(self) -> None:
        self.assertIs(admin_server_module.ROLE_PERMISSIONS, ROLE_PERMISSIONS)
        self.assertEqual(
            set(ROLE_PERMISSIONS["admin"]), {PERMISSION_READ, PERMISSION_WRITE}
        )


if __name__ == "__main__":
    unittest.main()
