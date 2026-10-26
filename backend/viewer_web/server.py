"""The viewer-facing web server: static page, public state, chat intake.

Same-origin by design. The page, its data, and its chat POSTs all come from
this one server so the browser never needs CORS and never talks to the
sensitive Brain/Chat ports (``/diagnostics``, ``/memory/*``, ``/skills`` stay
where they were). Messages are forwarded into the one existing
``ChatService.submit`` entry point, which is what makes viewer chat reach the
real Brain.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping, Optional
from urllib.parse import parse_qs, urlsplit

from ..chat.models import utc_now
from ..runtime.obs import log_event
from ..runtime.redaction import safe_error
from .feed import (
    DELIVERY_ACCEPTED,
    DELIVERY_DELIVERED,
    DELIVERY_FAILED,
    ChatFeed,
    project_message,
    public_reason,
)
from .projection import build_viewer_state, public_viewer_count

log = logging.getLogger("room_agent.viewer_web.server")

VIEWER_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{6,64}$")
CLIENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{6,64}$")
SEGMENT_PATTERN = re.compile(r"^seg_\d{5}\.ts$")

MAX_CHAT_BODY_BYTES = 20_000
MAX_TEXT_LENGTH = 500

STATIC_FILES: dict[str, tuple[str, str]] = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/app-state.js": ("app-state.js", "application/javascript; charset=utf-8"),
    "/app.js": ("app.js", "application/javascript; charset=utf-8"),
    "/vendor/hls.min.js": ("vendor/hls.min.js", "application/javascript; charset=utf-8"),
}


class PresenceTracker:
    """Bounded heartbeat registry behind the public viewer count."""

    def __init__(self, ttl_seconds: float = 30.0, capacity: int = 500, clock=utc_now) -> None:
        self._ttl = max(5.0, float(ttl_seconds))
        self._capacity = max(10, int(capacity))
        self._clock = clock
        self._lock = threading.Lock()
        self._sessions: dict[str, tuple[str, datetime]] = {}

    def touch(self, session_id: str, viewer_id: str) -> None:
        now = self._clock()
        with self._lock:
            self._sessions[session_id] = (viewer_id, now)
            self._prune_locked(now)

    def viewers(self) -> set[str]:
        now = self._clock()
        with self._lock:
            self._prune_locked(now)
            return {viewer_id for viewer_id, _ in self._sessions.values()}

    def _prune_locked(self, now: datetime) -> None:
        stale = [
            session
            for session, (_, stamp) in self._sessions.items()
            if (now - stamp).total_seconds() > self._ttl
        ]
        for session in stale:
            self._sessions.pop(session, None)
        overflow = len(self._sessions) - self._capacity
        if overflow > 0:
            oldest = sorted(
                self._sessions.items(), key=lambda item: item[1][1]
            )[:overflow]
            for session, _ in oldest:
                self._sessions.pop(session, None)


class IdempotencyStore:
    """Bounded replay cache so a reconnect retry cannot post twice."""

    def __init__(self, ttl_seconds: float = 600.0, capacity: int = 500, clock=time.monotonic) -> None:
        self._ttl = max(30.0, float(ttl_seconds))
        self._capacity = max(10, int(capacity))
        self._clock = clock
        self._lock = threading.Lock()
        self._entries: dict[tuple[str, str], tuple[dict[str, Any], float]] = {}

    def get(self, viewer_id: str, client_id: str) -> Optional[dict[str, Any]]:
        now = self._clock()
        with self._lock:
            entry = self._entries.get((viewer_id, client_id))
            if entry is None:
                return None
            body, stored_at = entry
            if now - stored_at > self._ttl:
                self._entries.pop((viewer_id, client_id), None)
                return None
            return dict(body)

    def put(self, viewer_id: str, client_id: str, body: dict[str, Any]) -> None:
        now = self._clock()
        with self._lock:
            self._entries[(viewer_id, client_id)] = (dict(body), now)
            if len(self._entries) > self._capacity:
                oldest = sorted(self._entries.items(), key=lambda item: item[1][1])
                for key, _ in oldest[: len(self._entries) - self._capacity]:
                    self._entries.pop(key, None)


class ViewerWebServer(ThreadingHTTPServer):
    """Dedicated viewer port server. Never exposes the ops surfaces."""

    daemon_threads = True

    def __init__(
        self,
        brain_service: Any,
        *,
        host: str = "127.0.0.1",
        port: int = 8770,
        relay: Any = None,
        streaming: Any = None,
        static_dir: Optional[Path] = None,
        presence_ttl_seconds: float = 30.0,
    ) -> None:
        self.brain_service = brain_service
        self.chat_service = brain_service.chat_service
        self.streaming = streaming
        self.relay = relay
        self._static_dir = Path(static_dir) if static_dir else Path(__file__).with_name("static")
        self._static_cache: dict[str, tuple[bytes, str]] = {}
        self._static_lock = threading.Lock()
        self.presence = PresenceTracker(ttl_seconds=presence_ttl_seconds)
        self.idempotency = IdempotencyStore()
        self.feed = ChatFeed()
        self._client_ids: dict[str, tuple[str, float]] = {}
        self._client_ids_lock = threading.Lock()
        super().__init__((host, port), self._build_handler())

    # -------------------------------------------------------------- public API

    @property
    def bound_port(self) -> int:
        return int(self.server_address[1])

    def public_state(self, viewer_id: str) -> dict[str, object]:
        """Build the public state document; also used by tests directly."""
        chat_state = self.chat_service.get_state()
        social = getattr(self.brain_service.brain, "social", None)

        observed: list[Mapping[str, object]] = []
        try:
            observed.extend(
                message.to_dict() for message in self.chat_service.get_queue()
            )
        except Exception:
            pass
        if social is not None:
            try:
                observed.extend(social.context.snapshot().get("recent_messages", []))
            except Exception:
                pass

        last_response = chat_state.get("last_response")
        if not isinstance(last_response, Mapping):
            last_response = None

        with self._client_ids_lock:
            client_ids = {
                message_id: client_id
                for message_id, (client_id, _) in self._client_ids.items()
            }
        self.feed.observe(observed, last_response, client_ids=client_ids)
        present = self.presence.viewers()
        relay_snapshot = self.relay.snapshot() if self.relay is not None else {
            "state": "UNAVAILABLE",
            "detail": "relay_not_configured",
            "configured": False,
            "playlist_url": None,
            "playlist_ready": False,
            "segments": 0,
            "restart_count": 0,
            "reason": "relay_not_configured",
        }
        queued = int(chat_state.get("queued_count") or 0)
        return build_viewer_state(
            self.brain_service,
            streaming=self.streaming,
            relay_snapshot=relay_snapshot,
            feed_snapshot=self.feed.snapshot(),
            viewer_count=public_viewer_count(social, present, utc_now()),
            viewer_id=viewer_id,
            chat_busy=queued > 0,
            queued_count=queued,
        )

    def submit_message(
        self,
        viewer_id: str,
        text: str,
        client_id: Optional[str],
    ) -> tuple[int, dict[str, Any]]:
        """Forward one viewer message into the existing chat intake."""
        if client_id:
            cached = self.idempotency.get(viewer_id, client_id)
            if cached is not None:
                cached["duplicate_request"] = True
                return int(cached.get("http_status", HTTPStatus.OK)), cached

        try:
            result = self.chat_service.submit(viewer_id, text)
        except ValueError as exc:
            return HTTPStatus.BAD_REQUEST, {"error": safe_error(exc, limit=120)}

        item = project_message(result.message.to_dict(), client_message_id=client_id)
        assert item is not None
        if result.accepted:
            delivery = DELIVERY_ACCEPTED
        elif result.merged_message is not None:
            delivery = DELIVERY_DELIVERED
            item["delivery"] = DELIVERY_DELIVERED
            item["reason"] = None
        else:
            delivery = DELIVERY_FAILED
        body: dict[str, Any] = {
            "accepted": bool(result.accepted),
            "delivery": delivery,
            "reason": public_reason(result.message.filtered_reason),
            "message": item,
        }
        if result.merged_message is not None:
            body["merged_message_id"] = result.merged_message.message_id
            body["merged_repeat_count"] = result.merged_message.repeat_count
        status = HTTPStatus.ACCEPTED if result.accepted else HTTPStatus.OK
        body["http_status"] = int(status)
        if client_id and isinstance(item.get("id"), str):
            self._remember_client_id(str(item["id"]), client_id)
            self.idempotency.put(viewer_id, client_id, body)
        return status, body

    # -------------------------------------------------------------- internals

    def _remember_client_id(self, message_id: str, client_id: str) -> None:
        now = time.monotonic()
        with self._client_ids_lock:
            self._client_ids[message_id] = (client_id, now)
            if len(self._client_ids) > 500:
                oldest = sorted(self._client_ids.items(), key=lambda item: item[1][1])
                for key, _ in oldest[: len(self._client_ids) - 500]:
                    self._client_ids.pop(key, None)

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

    def _hls_file(self, name: str) -> Optional[tuple[Path, str]]:
        if self.relay is None or self.relay.directory is None:
            return None
        if name == "live.m3u8":
            return self.relay.directory / "live.m3u8", "application/vnd.apple.mpegurl"
        if SEGMENT_PATTERN.match(name):
            return self.relay.directory / name, "video/mp2t"
        return None

    def _build_handler(self) -> type[BaseHTTPRequestHandler]:
        server = self

        class ViewerHandler(BaseHTTPRequestHandler):
            server_version = "room_agent_viewer"
            sys_version = ""

            # -------------------------------------------------------- GET
            def do_GET(self) -> None:  # noqa: N802 - http.server API
                try:
                    self._do_get()
                except Exception as exc:
                    self._fail(exc)

            def _do_get(self) -> None:
                parts = urlsplit(self.path)
                path = parts.path or "/"
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
                        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                        return
                    payload, cached_type = cached
                    self._send_bytes(
                        HTTPStatus.OK,
                        payload,
                        cached_type or content_type,
                        cache_control="no-cache",
                    )
                    return
                if path.startswith("/hls/"):
                    self._serve_hls(path[len("/hls/"):])
                    return
                if path == "/viewer/api/state":
                    self._serve_state(parse_qs(parts.query))
                    return
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

            def _serve_state(self, query: Mapping[str, list[str]]) -> None:
                viewer_id = _first(query, "viewer")
                session_id = _first(query, "session")
                if not viewer_id or not VIEWER_ID_PATTERN.match(viewer_id):
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "viewer_required"})
                    return
                if session_id and SESSION_ID_PATTERN.match(session_id):
                    server.presence.touch(session_id, viewer_id)
                payload = server.public_state(viewer_id)
                self._send_json(HTTPStatus.OK, payload, cache_control="no-store")

            def _serve_hls(self, name: str) -> None:
                resolved = server._hls_file(name)
                if resolved is None:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                    return
                path, content_type = resolved
                try:
                    payload = path.read_bytes()
                except OSError:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "stream_unavailable"})
                    return
                cache = "no-store" if name.endswith(".m3u8") else "public, max-age=30"
                self._send_bytes(HTTPStatus.OK, payload, content_type, cache_control=cache)

            # ------------------------------------------------------- POST
            def do_POST(self) -> None:  # noqa: N802 - http.server API
                try:
                    self._do_post()
                except Exception as exc:
                    self._fail(exc)

            def _do_post(self) -> None:
                path = urlsplit(self.path).path
                if path != "/viewer/api/chat":
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                    return
                try:
                    payload = self._read_json()
                except (ValueError, json.JSONDecodeError):
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_json"})
                    return

                viewer_id = payload.get("viewer_id")
                text = payload.get("text")
                client_id = payload.get("client_message_id")
                if (
                    not isinstance(viewer_id, str)
                    or not VIEWER_ID_PATTERN.match(viewer_id.strip())
                ):
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_viewer_id"})
                    return
                if not isinstance(text, str) or not text.strip():
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "text_required"})
                    return
                if len(text.strip()) > MAX_TEXT_LENGTH:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "text_too_long"})
                    return
                if client_id is not None and (
                    not isinstance(client_id, str) or not CLIENT_ID_PATTERN.match(client_id)
                ):
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_client_message_id"})
                    return

                status, body = server.submit_message(
                    viewer_id.strip(), text.strip(), client_id
                )
                self._send_json(status, body)

            # ------------------------------------------------------ helpers
            def _read_json(self) -> dict[str, Any]:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > MAX_CHAT_BODY_BYTES:
                    raise ValueError("invalid_content_length")
                value = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(value, dict):
                    raise ValueError("json_object_required")
                return value

            def _security_headers(self) -> None:
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Referrer-Policy", "no-referrer")
                self.send_header("X-Frame-Options", "SAMEORIGIN")

            def _send_bytes(
                self,
                status: HTTPStatus,
                payload: bytes,
                content_type: str,
                *,
                cache_control: str = "no-store",
            ) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", cache_control)
                self._security_headers()
                self.end_headers()
                self.wfile.write(payload)

            def _send_json(
                self,
                status: HTTPStatus,
                body: Mapping[str, Any],
                *,
                cache_control: str = "no-store",
            ) -> None:
                encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
                self._send_bytes(
                    status,
                    encoded,
                    "application/json; charset=utf-8",
                    cache_control=cache_control,
                )

            def _fail(self, exc: Exception) -> None:
                log_event(
                    log,
                    "viewer.web.request_failed",
                    error=safe_error(exc, limit=200),
                )
                try:
                    self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "unavailable"})
                except Exception:
                    pass

            def log_message(self, format: str, *args: object) -> None:
                return

        return ViewerHandler


def _first(query: Mapping[str, list[str]], key: str) -> Optional[str]:
    values = query.get(key)
    if not values or not isinstance(values[0], str):
        return None
    return values[0].strip() or None
