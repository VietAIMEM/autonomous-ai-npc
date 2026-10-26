"""PHASE 23 admin HTTP surface: authenticated control plane on its own port.

The dashboard is a *separate privileged surface* from the public viewer web:
it listens on ``ADMIN_PORT`` (default 8771), serves its own whitelisted static
page, and speaks only ``/admin/api/*``. Every mutating route is authorized by
the backend here before the request ever reaches an existing service. The
browser never sees a provider credential, a token, a SQLite handle, a shell or
Unity: it sees the structured projections and command results that
:mod:`backend.admin_web.console` already produced.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping, Optional
from urllib.parse import parse_qs, urlsplit

from ..runtime.obs import log_event
from ..runtime.redaction import safe_error
from .auth import SESSION_COOKIE_NAME, AdminAuthenticator, load_admin_token
from .console import AdminActionError, AdminConsole
from .logs import LogQuery
from .settings import SettingValidationError, SettingsBatchError

log = logging.getLogger("room_agent.admin_web.server")

STATIC_FILES: dict[str, tuple[str, str]] = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/app-state.js": ("app-state.js", "application/javascript; charset=utf-8"),
    "/app.js": ("app.js", "application/javascript; charset=utf-8"),
}

MAX_JSON_BODY_BYTES = 64_000
MAX_UPLOAD_BYTES = 64 * 1024 * 1024
MAX_IMPORT_FILE_BYTES = 48 * 1024 * 1024
MAX_FORM_FIELD_BYTES = 4_000
MAX_MULTIPART_PARTS = 24

PERMISSION_READ = "admin.read"
PERMISSION_WRITE = "admin.write"
ROLE_PERMISSIONS: dict[str, frozenset[str]] = {
    "admin": frozenset({PERMISSION_READ, PERMISSION_WRITE}),
}

_FILENAME_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")
_DISPOSITION_PARAM = re.compile(r';\s*([A-Za-z0-9_-]+)="([^"]*)"')
_MOTION_ID_PATTERN = re.compile(r"^[a-z0-9_]{1,64}$")

_CONFLICT_CODES = frozenset(
    {
        "instance_already_running",
        "runtime_command_in_progress",
        "motion_import_in_progress",
        "motion_processing_in_progress",
    }
)


class _BadRequest(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class _PayloadTooLarge(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class _Unauthorized(Exception):
    def __init__(self, code: str = "authentication_required") -> None:
        super().__init__(code)
        self.code = code


class _Responded(Exception):
    """Sentinel: a response body was already written for this request."""


class AdminWebServer(ThreadingHTTPServer):
    """Dedicated admin port. Authenticated; never a viewer-facing surface."""

    daemon_threads = True

    def __init__(
        self,
        console: AdminConsole,
        *,
        host: str = "127.0.0.1",
        port: int = 8771,
        token: Optional[str] = None,
        authenticator: Optional[AdminAuthenticator] = None,
        static_dir: Optional[Path] = None,
        upload_dir: Optional[Path] = None,
        project_root: Optional[Path] = None,
        max_upload_bytes: int = MAX_UPLOAD_BYTES,
    ) -> None:
        self.console = console
        if authenticator is None:
            secret = token or load_admin_token(console.config.state_dir)
            authenticator = AdminAuthenticator(secret)
        self.authenticator = authenticator
        self._static_dir = Path(static_dir) if static_dir else Path(__file__).with_name("static")
        self._static_cache: dict[str, tuple[bytes, str]] = {}
        self._static_lock = threading.Lock()
        self._upload_dir = (
            Path(upload_dir)
            if upload_dir is not None
            else Path(console.config.temp_dir) / "admin_uploads"
        )
        self._project_root = Path(project_root) if project_root else getattr(
            console, "project_root", None
        )
        self._max_upload_bytes = max(64_000, int(max_upload_bytes))
        super().__init__((host, port), self._build_handler())

    @property
    def bound_port(self) -> int:
        return int(self.server_address[1])

    # ---------------------------------------------------------------- static

    def _load_static(self, relative: str) -> Optional[tuple[bytes, str]]:
        with self._static_lock:
            cached = self._static_cache.get(relative)
            if cached is not None:
                return cached
        path = self._static_dir / relative
        try:
            if not path.is_file() or self._static_dir.resolve() not in path.resolve().parents:
                return None
            payload = path.read_bytes()
        except OSError:
            return None
        content_type = STATIC_FILES.get("/" + relative, (relative, "application/octet-stream"))[1]
        with self._static_lock:
            self._static_cache[relative] = (payload, content_type)
        return payload, content_type

    # --------------------------------------------------------------- uploads

    def _resolve_source_path(self, relative: str) -> Path:
        """Resolve an operator-supplied project-relative motion file.

        Only existing files inside the project root are accepted; the original
        is handed to the import service untouched.
        """
        if self._project_root is None:
            raise AdminActionError("motion_source_path_unavailable")
        candidate = Path(str(relative).strip())
        if candidate.is_absolute() or ".." in candidate.parts:
            raise AdminActionError("invalid_source_path")
        resolved = (self._project_root / candidate).resolve()
        root = self._project_root.resolve()
        if resolved != root and root not in resolved.parents:
            raise AdminActionError("invalid_source_path")
        if not resolved.is_file():
            raise AdminActionError("source_file_not_found")
        return resolved

    def _store_upload(self, filename: str, payload: bytes) -> Path:
        if not payload:
            raise AdminActionError("empty_motion_file")
        if len(payload) > MAX_IMPORT_FILE_BYTES:
            raise AdminActionError("motion_file_too_large")
        safe_name = _FILENAME_PATTERN.sub("_", Path(filename).name).strip("._")
        if not safe_name:
            raise AdminActionError("invalid_file_name")
        safe_name = safe_name[-120:]
        if not Path(safe_name).suffix:
            raise AdminActionError("missing_file_extension")
        self._upload_dir.mkdir(parents=True, exist_ok=True)
        target = self._upload_dir / f"{uuid.uuid4().hex[:12]}_{safe_name}"
        if target.exists():  # pragma: no cover - uuid collision guard
            raise AdminActionError("upload_collision")
        with open(target, "xb") as stream:
            stream.write(payload)
        return target

    # -------------------------------------------------------------- handler

    def _build_handler(self) -> type[BaseHTTPRequestHandler]:
        server = self

        class AdminHandler(BaseHTTPRequestHandler):
            server_version = "room_agent_admin"
            sys_version = ""

            # -------------------------------------------------------- verbs
            def do_GET(self) -> None:  # noqa: N802 - http.server API
                try:
                    self._do_get()
                except _Responded:
                    return
                except _BadRequest as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": exc.code})
                except _Unauthorized as exc:
                    self._send_json(HTTPStatus.UNAUTHORIZED, self._unauthorized_body(exc.code))
                except AdminActionError as exc:
                    self._send_action_error(exc)
                except SettingValidationError as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, **exc.to_dict()})
                except Exception as exc:  # pragma: no cover - defensive
                    self._fail(exc)

            def do_POST(self) -> None:  # noqa: N802 - http.server API
                try:
                    self._do_post()
                except _Responded:
                    return
                except _BadRequest as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": exc.code})
                except _PayloadTooLarge as exc:
                    self._send_json(
                        HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                        {"ok": False, "error": exc.code},
                    )
                except _Unauthorized as exc:
                    self._send_json(HTTPStatus.UNAUTHORIZED, self._unauthorized_body(exc.code))
                except AdminActionError as exc:
                    self._send_action_error(exc)
                except SettingValidationError as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, **exc.to_dict()})
                except Exception as exc:
                    self._fail(exc)

            def do_PUT(self) -> None:  # noqa: N802 - http.server API
                self._method_not_allowed()

            def do_PATCH(self) -> None:  # noqa: N802 - http.server API
                self._method_not_allowed()

            def do_DELETE(self) -> None:  # noqa: N802 - http.server API
                self._method_not_allowed()

            # ----------------------------------------------------------- GET
            def _do_get(self) -> None:
                parts = urlsplit(self.path)
                path = parts.path or "/"
                query = parse_qs(parts.query)
                if path == "/favicon.ico":
                    self.send_response(HTTPStatus.NO_CONTENT)
                    self._security_headers()
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if path in STATIC_FILES:
                    relative, content_type = STATIC_FILES[path]
                    cached = server._load_static(relative)
                    if cached is None:
                        self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not_found"})
                        return
                    payload, cached_type = cached
                    self._send_bytes(
                        HTTPStatus.OK,
                        payload,
                        cached_type or content_type,
                        cache_control="no-cache",
                    )
                    return
                if path.startswith("/admin/api/"):
                    self._api_get(path, query)
                    return
                self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not_found"})

            def _api_get(self, path: str, query: Mapping[str, list[str]]) -> None:
                console = server.console
                if path == "/admin/api/session":
                    authenticated = server.authenticator.authenticate(self._session_token())
                    body: dict[str, Any] = {
                        "ok": True,
                        "authenticated": authenticated,
                        "login_required": True,
                        "session_ttl_seconds": server.authenticator.sessions.ttl_seconds,
                    }
                    if authenticated:
                        body["role"] = "admin"
                        body["permissions"] = sorted(ROLE_PERMISSIONS["admin"])
                    self._send_json(HTTPStatus.OK, body)
                    return

                self._authorize(PERMISSION_READ)

                if path == "/admin/api/status":
                    payload: Any = console.status()
                elif path == "/admin/api/health":
                    payload = console.health()
                elif path == "/admin/api/overview":
                    payload = console.overview()
                elif path == "/admin/api/runtime":
                    payload = console.runtime_status()
                elif path == "/admin/api/runtime/health":
                    payload = console.runtime_health()
                elif path == "/admin/api/providers":
                    payload = console.providers()
                elif path == "/admin/api/tts":
                    payload = console.tts(_first(query, "request_id"))
                elif path == "/admin/api/chat":
                    payload = console.chat()
                elif path == "/admin/api/stream":
                    payload = console.stream()
                elif path == "/admin/api/motion":
                    payload = console.motion()
                elif path == "/admin/api/motion/job":
                    request_id = _first(query, "request_id")
                    if not request_id:
                        raise _BadRequest("request_id_required")
                    payload = console.motion_job(_first(query, "kind") or "processing", request_id)
                elif path == "/admin/api/skills":
                    payload = console.skills()
                elif path == "/admin/api/learning":
                    payload = console.learning()
                elif path == "/admin/api/memory":
                    payload = console.memory()
                elif path == "/admin/api/viewers":
                    payload = console.viewers(_int_param(query, "limit", 50))
                elif path == "/admin/api/personality":
                    payload = console.personality()
                elif path == "/admin/api/diagnostics":
                    payload = console.diagnostics()
                elif path == "/admin/api/logs":
                    payload = console.logs(self._log_query(query))
                elif path == "/admin/api/settings":
                    payload = console.settings_view()
                else:
                    self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not_found"})
                    return
                self._send_result(payload)

            # ---------------------------------------------------------- POST
            def _do_post(self) -> None:
                path = urlsplit(self.path).path
                if not path.startswith("/admin/api/"):
                    self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not_found"})
                    return
                if path == "/admin/api/login":
                    self._handle_login()
                    return

                self._check_same_origin()
                if path == "/admin/api/logout":
                    server.authenticator.logout(self._session_token())
                    self._send_json(
                        HTTPStatus.OK,
                        {"ok": True, "authenticated": False},
                        extra_headers=[self._clear_cookie()],
                    )
                    return

                self._authorize(PERMISSION_WRITE)
                console = server.console

                if path.startswith("/admin/api/runtime/"):
                    action = path.rsplit("/", 1)[-1]
                    payload: Any = console.runtime_command(action)
                elif path == "/admin/api/tts/test":
                    body = self._json_body(required=True)
                    payload = console.tts_test(str(body.get("text") or ""))
                elif path == "/admin/api/chat/reconnect":
                    payload = console.chat_reconnect()
                elif path.startswith("/admin/api/stream/"):
                    action = path.rsplit("/", 1)[-1]
                    body = self._json_body()
                    payload = console.stream_command(
                        action,
                        camera=str(body.get("camera") or ""),
                        wait_seconds=_optional_float(body.get("wait")),
                    )
                elif path == "/admin/api/motion/import":
                    payload = self._handle_motion_import()
                elif path == "/admin/api/motion/generate":
                    body = self._json_body(required=True)
                    prompt = str(body.get("prompt") or "").strip()
                    if not prompt:
                        raise _BadRequest("prompt_required")
                    payload = console.motion_generate(
                        prompt,
                        duration_seconds=_float_or(body.get("duration_seconds"), 2.5),
                        motion_type=str(body.get("motion_type") or "general"),
                        style=str(body.get("style") or ""),
                        speed=_float_or(body.get("speed"), 1.0),
                    )
                elif path == "/admin/api/motion/process":
                    body = self._json_body(required=True)
                    source_motion_id = str(body.get("source_motion_id") or "").strip()
                    if not source_motion_id:
                        raise _BadRequest("source_motion_id_required")
                    payload = console.motion_process(
                        source_motion_id,
                        processed_motion_id=_optional_str(body.get("processed_motion_id")),
                        processing_version=int(_float_or(body.get("processing_version"), 1)),
                        name=str(body.get("name") or ""),
                        description=str(body.get("description") or ""),
                        skill_category=str(body.get("skill_category") or "MOVEMENT"),
                    )
                elif path in {"/admin/api/motion/approve", "/admin/api/motion/register"}:
                    body = self._json_body(required=True)
                    request_id = str(body.get("request_id") or "").strip()
                    if not request_id:
                        raise _BadRequest("request_id_required")
                    payload = (
                        console.motion_approve(request_id)
                        if path.endswith("/approve")
                        else console.motion_register(request_id)
                    )
                elif path in {"/admin/api/skills/enable", "/admin/api/skills/disable"}:
                    body = self._json_body(required=True)
                    skill_id = str(body.get("skill_id") or "").strip()
                    if not skill_id:
                        raise _BadRequest("skill_id_required")
                    payload = console.skill_set_enabled(skill_id, path.endswith("/enable"))
                elif path == "/admin/api/learning/reset":
                    payload = console.learning_reset()
                elif path == "/admin/api/settings":
                    body = self._json_body(required=True)
                    values = body.get("values")
                    if not isinstance(values, Mapping):
                        raise _BadRequest("settings_required")
                    payload = console.save_settings(values)
                else:
                    self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not_found"})
                    return
                self._send_result(payload)

            # --------------------------------------------------------- login
            def _handle_login(self) -> None:
                self._check_same_origin()
                if not server.authenticator.throttle.allowed():
                    retry_after = server.authenticator.throttle.retry_after_seconds()
                    self._send_json(
                        HTTPStatus.TOO_MANY_REQUESTS,
                        {
                            "ok": False,
                            "error": "login_throttled",
                            "retry_after_seconds": round(retry_after, 1),
                        },
                        extra_headers=[("Retry-After", str(max(1, int(retry_after + 0.5))))],
                    )
                    return
                body = self._json_body(required=True)
                candidate = body.get("token")
                if not isinstance(candidate, str) or not candidate.strip():
                    raise _BadRequest("token_required")
                session = server.authenticator.login(
                    candidate.strip(), source=self.client_address[0] if self.client_address else "local"
                )
                if session is None:
                    self._send_json(
                        HTTPStatus.UNAUTHORIZED,
                        {"ok": False, "error": "invalid_credentials"},
                    )
                    return
                ttl = int(server.authenticator.sessions.ttl_seconds)
                self._send_json(
                    HTTPStatus.OK,
                    {
                        "ok": True,
                        "authenticated": True,
                        "role": "admin",
                        "permissions": sorted(ROLE_PERMISSIONS["admin"]),
                        "expires_in_seconds": ttl,
                    },
                    extra_headers=[self._session_cookie(session, ttl)],
                )

            # ---------------------------------------------------- motion upload
            def _handle_motion_import(self) -> dict[str, Any]:
                content_type = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
                if content_type == "multipart/form-data":
                    fields, files = self._multipart_body()
                    filename, payload = files.get("file") or ("", b"")
                    if not filename:
                        raise _BadRequest("motion_file_required")
                    stored = server._store_upload(filename, payload)
                elif content_type == "application/json":
                    body = self._json_body(required=True)
                    source_path = str(body.get("source_path") or "").strip()
                    if not source_path:
                        raise _BadRequest("motion_file_required")
                    stored = server._resolve_source_path(source_path)
                    fields = {
                        key: body.get(key)
                        for key in ("motion_id", "name", "description", "clip_name", "tags", "skill_category")
                    }
                else:
                    raise _BadRequest("content_type_required")

                motion_id = str(fields.get("motion_id") or "").strip().lower()
                if not _MOTION_ID_PATTERN.match(motion_id):
                    raise _BadRequest("invalid_motion_id")
                tags = fields.get("tags") or ()
                if isinstance(tags, str):
                    tags = tuple(part.strip() for part in tags.split(",") if part.strip())
                elif isinstance(tags, (list, tuple)):
                    tags = tuple(str(part).strip() for part in tags if str(part).strip())
                else:
                    tags = ()

                return server.console.motion_import_file(
                    stored,
                    motion_id,
                    name=str(fields.get("name") or ""),
                    description=str(fields.get("description") or ""),
                    clip_name=_optional_str(fields.get("clip_name")),
                    tags=tags,
                    skill_category=str(fields.get("skill_category") or "MOVEMENT"),
                )

            # ------------------------------------------------------- helpers
            def _authorize(self, permission: str) -> None:
                if not server.authenticator.authenticate(self._session_token()):
                    raise _Unauthorized()
                role = "admin"
                if permission not in ROLE_PERMISSIONS.get(role, frozenset()):
                    log_event(log, "admin.auth.permission_denied", permission=permission, role=role)
                    self._send_json(
                        HTTPStatus.FORBIDDEN,
                        {"ok": False, "error": "permission_denied", "permission": permission},
                    )
                    raise _Responded()

            def _session_token(self) -> Optional[str]:
                header = self.headers.get("Cookie") or ""
                for part in header.split(";"):
                    name, sep, value = part.strip().partition("=")
                    if sep and name.strip() == SESSION_COOKIE_NAME:
                        return value.strip() or None
                return None

            def _session_cookie(self, token: str, ttl: int) -> tuple[str, str]:
                return (
                    "Set-Cookie",
                    f"{SESSION_COOKIE_NAME}={token}; Path=/; HttpOnly; SameSite=Strict; Max-Age={ttl}",
                )

            def _clear_cookie(self) -> tuple[str, str]:
                return (
                    "Set-Cookie",
                    f"{SESSION_COOKIE_NAME}=; Path=/; HttpOnly; SameSite=Strict; Max-Age=0",
                )

            def _check_same_origin(self) -> None:
                """Cookie is SameSite=Strict; this rejects forged same-site calls."""
                origin = (self.headers.get("Origin") or "").strip()
                if origin:
                    host = (self.headers.get("Host") or "").strip()
                    if not host or origin.rstrip("/") not in {
                        f"http://{host}",
                        f"https://{host}",
                    }:
                        raise _BadRequest("cross_origin_rejected")
                site = (self.headers.get("Sec-Fetch-Site") or "").strip().lower()
                if site and site not in {"same-origin", "none"}:
                    raise _BadRequest("cross_site_rejected")

            def _read_body(self, limit: int) -> bytes:
                raw_length = self.headers.get("Content-Length")
                try:
                    length = int(raw_length or "0")
                except ValueError as exc:
                    raise _BadRequest("invalid_content_length") from exc
                if length < 0:
                    raise _BadRequest("invalid_content_length")
                if length > limit:
                    raise _PayloadTooLarge("request_too_large")
                if length == 0:
                    return b""
                return self.rfile.read(length)

            def _json_body(self, *, required: bool = False) -> dict[str, Any]:
                content_type = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
                raw = self._read_body(MAX_JSON_BODY_BYTES)
                if not raw:
                    if required:
                        raise _BadRequest("json_body_required")
                    return {}
                if content_type != "application/json":
                    raise _BadRequest("content_type_required")
                try:
                    value = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise _BadRequest("invalid_json") from exc
                if not isinstance(value, dict):
                    raise _BadRequest("json_object_required")
                return value

            def _multipart_body(self) -> tuple[dict[str, Any], dict[str, tuple[str, bytes]]]:
                content_type = self.headers.get("Content-Type") or ""
                match = re.search(r'boundary="?([^";]+)"?', content_type)
                if match is None:
                    raise _BadRequest("invalid_multipart_boundary")
                boundary = match.group(1).strip()
                if not boundary or len(boundary) > 200:
                    raise _BadRequest("invalid_multipart_boundary")
                raw = self._read_body(server._max_upload_bytes)
                return _parse_multipart(raw, boundary.encode("utf-8"))

            def _log_query(self, query: Mapping[str, list[str]]) -> LogQuery:
                return LogQuery(
                    limit=_int_param(query, "limit", 200),
                    severity=_first(query, "severity"),
                    component=_first(query, "component"),
                    operation=_first(query, "operation"),
                    since_minutes=_optional_float(_first(query, "since_minutes")),
                )

            def _send_result(self, payload: Any) -> None:
                status = HTTPStatus.OK
                if isinstance(payload, Mapping):
                    http_status = payload.get("http_status")
                    if isinstance(http_status, int) and 100 <= http_status <= 599:
                        status = HTTPStatus(http_status)
                    elif payload.get("ok") is False:
                        status = HTTPStatus.BAD_REQUEST
                self._send_json(status, payload)

            def _send_action_error(self, exc: AdminActionError) -> None:
                self._send_json(_action_status(exc.code), exc.to_dict())

            def _unauthorized_body(self, code: str) -> dict[str, Any]:
                return {"ok": False, "error": code, "login_required": True}

            def _method_not_allowed(self) -> None:
                self._send_json(
                    HTTPStatus.METHOD_NOT_ALLOWED,
                    {"ok": False, "error": "method_not_allowed"},
                )

            def _security_headers(self) -> None:
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Referrer-Policy", "no-referrer")
                self.send_header("X-Frame-Options", "DENY")
                self.send_header("Cross-Origin-Opener-Policy", "same-origin")
                self.send_header("Cross-Origin-Resource-Policy", "same-origin")
                self.send_header(
                    "Content-Security-Policy",
                    "default-src 'self'; script-src 'self'; style-src 'self'; "
                    "img-src 'self' data:; connect-src 'self'; form-action 'self'; "
                    "frame-ancestors 'none'; base-uri 'none'",
                )

            def _send_bytes(
                self,
                status: HTTPStatus,
                payload: bytes,
                content_type: str,
                *,
                cache_control: str = "no-store",
                extra_headers: Optional[list[tuple[str, str]]] = None,
            ) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", cache_control)
                self._security_headers()
                for name, value in extra_headers or ():
                    self.send_header(name, value)
                self.end_headers()
                self.wfile.write(payload)

            def _send_json(
                self,
                status: HTTPStatus,
                body: Mapping[str, Any],
                *,
                cache_control: str = "no-store",
                extra_headers: Optional[list[tuple[str, str]]] = None,
            ) -> None:
                encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
                self._send_bytes(
                    status,
                    encoded,
                    "application/json; charset=utf-8",
                    cache_control=cache_control,
                    extra_headers=extra_headers,
                )

            def _fail(self, exc: Exception) -> None:
                log_event(log, "admin.web.request_failed", error=safe_error(exc, limit=200))
                try:
                    self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": "unavailable"})
                except Exception:
                    pass

            def log_message(self, format: str, *args: object) -> None:
                return

        return AdminHandler


def _action_status(code: str) -> HTTPStatus:
    if code in _CONFLICT_CODES:
        return HTTPStatus.CONFLICT
    if code.endswith("_not_found"):
        return HTTPStatus.NOT_FOUND
    if code.endswith("_unavailable") or code.endswith("_disabled"):
        return HTTPStatus.SERVICE_UNAVAILABLE
    return HTTPStatus.BAD_REQUEST


def _parse_multipart(
    body: bytes,
    boundary: bytes,
) -> tuple[dict[str, Any], dict[str, tuple[str, bytes]]]:
    """Bounded ``multipart/form-data`` parser: fields plus one file per name."""
    delimiter = b"--" + boundary
    segments = body.split(delimiter)
    if len(segments) < 3 or len(segments) > MAX_MULTIPART_PARTS + 2:
        raise _BadRequest("invalid_multipart_body")
    fields: dict[str, Any] = {}
    files: dict[str, tuple[str, bytes]] = {}
    for segment in segments[1:]:
        if segment.startswith(b"--"):
            break
        if segment.startswith(b"\r\n"):
            segment = segment[2:]
        if segment.endswith(b"\r\n"):
            segment = segment[:-2]
        header_blob, separator, content = segment.partition(b"\r\n\r\n")
        if not separator:
            continue
        params = _disposition_params(header_blob)
        name = params.get("name")
        if not name:
            continue
        filename = params.get("filename")
        if filename is not None:
            if len(content) > MAX_IMPORT_FILE_BYTES:
                raise _PayloadTooLarge("motion_file_too_large")
            files[name] = (filename, content)
        else:
            if len(content) > MAX_FORM_FIELD_BYTES:
                raise _PayloadTooLarge("form_field_too_large")
            fields[name] = content.decode("utf-8", errors="replace")
    return fields, files


def _disposition_params(header_blob: bytes) -> dict[str, str]:
    text = header_blob.decode("iso-8859-1", errors="replace")
    for line in text.split("\r\n"):
        if line.lower().startswith("content-disposition:"):
            return {key.lower(): value for key, value in _DISPOSITION_PARAM.findall(line)}
    return {}


def _first(query: Mapping[str, list[str]], key: str) -> Optional[str]:
    values = query.get(key)
    if not values or not isinstance(values[0], str):
        return None
    return values[0].strip() or None


def _int_param(query: Mapping[str, list[str]], key: str, default: int) -> int:
    raw = _first(query, key)
    if raw is None:
        return default
    try:
        return int(float(raw))
    except ValueError:
        return default


def _optional_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _float_or(value: Any, default: float) -> float:
    parsed = _optional_float(value)
    return default if parsed is None else parsed


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


__all__ = ["AdminWebServer", "PERMISSION_READ", "PERMISSION_WRITE", "ROLE_PERMISSIONS", "STATIC_FILES"]
