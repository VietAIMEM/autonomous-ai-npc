"""Minimal JSON HTTP API for the Phase 05 chat queue.

Two ways to run it exist, and both share one implementation:

* the production backend (``backend.api.brain``) mounts this handler on
  ``CHAT_PORT`` next to the Brain HTTP surface, and
* ``python -m backend.api.chat`` starts the same handler standalone for
  development, which is still what the PHASE 05 README documents.

Either way there is exactly one chat intake: the standalone entry point takes
the same instance lock as the backend, so a second copy is refused instead of
racing for the port or duplicating the ingest worker.
"""

from __future__ import annotations

import json
import logging
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional
from urllib.parse import urlsplit

from ..chat.providers.runner import ChatSourceRunner
from ..chat.service import ChatService, ChatServiceConfig
from ..runtime import health as health_module
from ..runtime.redaction import safe_error


log = logging.getLogger("room_agent.api.chat")


def create_chat_server(
    service: ChatService,
    host: str = "127.0.0.1",
    port: int = 8765,
    source: Optional[ChatSourceRunner] = None,
) -> ThreadingHTTPServer:
    handler = _build_handler(service, source)
    return ThreadingHTTPServer((host, port), handler)


def chat_health(service: ChatService, source: Optional[ChatSourceRunner]) -> dict[str, Any]:
    """Cheap standalone health probe in the shared status vocabulary.

    Mirrors ``backend.api.diagnostics._chat_component`` for the standalone
    entry point, without importing the Brain package. A disabled chat provider
    is ``UNAVAILABLE`` *and* intentional, never a silent success.
    """
    queue: dict[str, Any] = {}
    try:
        state = service.get_state()
        queue = {
            "queued_count": state.get("queued_count", 0),
            "queue_capacity": state.get("queue_capacity", 0),
        }
    except Exception:  # pragma: no cover - defensive
        queue = {}
    if source is None:
        entry = health_module.component(
            "chat",
            configured=False,
            status=health_module.STATUS_UNAVAILABLE,
            required=False,
            intentional=True,
            reason="chat_provider_disabled",
            detail=queue,
        )
        return {"status": entry["status"], "component": entry}
    try:
        probe = dict(source.health())
    except Exception as exc:
        entry = health_module.component(
            "chat",
            configured=True,
            reachable=False,
            status=health_module.STATUS_UNAVAILABLE,
            required=False,
            reason=safe_error(exc, limit=120),
            detail=queue,
        )
        return {"status": entry["status"], "component": entry}
    provider_state = str(probe.get("state") or "").upper()
    connected = provider_state == "CONNECTED"
    retrying = provider_state in {"CONNECTING", "RETRYING"}
    entry = health_module.component(
        "chat",
        configured=True,
        reachable=connected or retrying,
        healthy=connected and not bool(probe.get("degraded")),
        required=False,
        reason=None if connected else (probe.get("last_error") or None),
        detail={
            **queue,
            "provider": probe.get("provider"),
            "platform": probe.get("platform"),
            "state": provider_state,
            "reconnect_count": probe.get("reconnect_count"),
        },
    )
    return {"status": entry["status"], "component": entry}


def _build_handler(
    service: ChatService,
    source: Optional[ChatSourceRunner] = None,
) -> type[BaseHTTPRequestHandler]:
    class ChatAPIHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - http.server API
            path = urlsplit(self.path).path
            if path == "/chat/queue":
                self._send_json(HTTPStatus.OK, {"messages": [m.to_dict() for m in service.get_queue()]})
            elif path == "/chat/state":
                self._send_json(HTTPStatus.OK, service.get_state())
            elif path == "/chat/health":
                # Unified vocabulary for the standalone entry point; the
                # production surface is ``/health`` on the backend.
                self._send_json(HTTPStatus.OK, chat_health(service, source))
            elif path in {"/chat/source/state", "/chat/source/health"}:
                # PHASE 12 diagnostics. Never includes credentials.
                if source is None:
                    self._send_json(
                        HTTPStatus.OK,
                        {"provider": "disabled", "state": "STOPPED", "enabled": False},
                    )
                    return
                probe = source.health() if path.endswith("/health") else source.get_state()
                self._send_json(HTTPStatus.OK, probe)
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

        def do_POST(self) -> None:  # noqa: N802 - http.server API
            path = urlsplit(self.path).path
            if path != "/chat/message":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return

            try:
                payload = self._read_json()
            except (ValueError, json.JSONDecodeError):
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_json"})
                return

            viewer_id = payload.get("viewer_id")
            raw_text = payload.get("text")
            if not isinstance(viewer_id, str) or not isinstance(raw_text, str):
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "viewer_id_and_text_required"})
                return

            try:
                result = service.submit(viewer_id, raw_text)
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return

            status = HTTPStatus.ACCEPTED if result.accepted else HTTPStatus.OK
            body: dict[str, Any] = {
                "message": result.message.to_dict(),
                "accepted": result.accepted,
            }
            if result.merged_message is not None:
                body["merged_message"] = result.merged_message.to_dict()
            self._send_json(status, body)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 20_000:
                raise ValueError("invalid_content_length")
            value = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(value, dict):
                raise ValueError("json_object_required")
            return value

        def _send_json(self, status: HTTPStatus, body: dict[str, Any]) -> None:
            encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format: str, *args: object) -> None:
            return

    return ChatAPIHandler


def main() -> int:
    """Standalone chat intake with the same lifecycle as the backend.

    The LLM is resolved from the environment exactly like ``backend.api.brain``
    does: ``LLM_PROVIDER`` selects the transport, an unset variable keeps the
    offline mock in development (as this entry point always documented), and
    ``ROOM_AGENT_ENV=production`` refuses to start with that mock instead of
    quietly answering viewers with canned text.
    """
    import os

    from ..llm.config import LLMConfig, build_provider as build_llm_provider
    from ..runtime.config import ConfigurationError, RuntimeConfig
    from ..runtime.lifecycle import DuplicateInstanceError, RuntimeLifecycle
    from ..runtime.obs import configure_logging, log_event
    from ..runtime.redaction import safe_error
    from ..llm.config import PROVIDER_MOCK as LLM_PROVIDER_MOCK
    from .brain import enforce_provider_environment, resolve_llm_provider

    runtime_config = RuntimeConfig.from_environment()
    configure_logging(runtime_config)
    try:
        enforce_provider_environment(runtime_config)
        if (os.environ.get("LLM_PROVIDER") or "").strip():
            provider = resolve_llm_provider(runtime_config)
        else:
            runtime_config.enforce_provider("llm", LLM_PROVIDER_MOCK)
            provider = build_llm_provider(LLMConfig(provider=LLM_PROVIDER_MOCK))
    except ConfigurationError as exc:
        log.error("startup refused: %s", safe_error(exc, limit=200))
        print(json.dumps({"error": safe_error(exc, limit=200)}, indent=2))
        return 2

    lifecycle = RuntimeLifecycle(runtime_config, role="chat")
    try:
        lifecycle.start()
    except DuplicateInstanceError as exc:
        log.error("duplicate instance refused: %s", safe_error(exc, limit=200))
        print(json.dumps({"error": str(exc)}, indent=2))
        return 3

    service = ChatService(provider, ChatServiceConfig.from_environment())
    source = None
    server = None
    thread = None
    shutdown_done = threading.Event()
    serving = threading.Event()
    try:
        source = build_chat_source(service)
        if source is not None:
            source.start()
        server = create_chat_server(
            service,
            host=runtime_config.host,
            port=runtime_config.chat_port,
            source=source,
        )
    except Exception as exc:
        log_event(log, "chat.startup_failed", error=safe_error(exc, limit=200))
        if source is not None:
            try:
                source.stop()
            except Exception:
                pass
        if server is not None:
            server.server_close()
        lifecycle.shutdown(reason="startup_failed")
        return 4

    def _stop_http() -> None:
        """Stop external input, then the HTTP loop, then the socket."""
        if source is not None:
            source.stop()
        if thread is not None:
            if serving.wait(timeout=5.0):
                server.shutdown()
            thread.join(timeout=3.0)
        server.server_close()

    stop_triggered = threading.Event()
    stop_lock = threading.Lock()

    def _stop(reason: str = "shutdown") -> None:
        """Idempotent, off-thread stop so the watcher thread never joins itself."""
        if stop_triggered.is_set():
            return
        with stop_lock:
            if stop_triggered.is_set():
                return
            stop_triggered.set()

        def run() -> None:
            try:
                lifecycle.register_shutdown("chat.stop", _stop_http, timeout_seconds=10.0)
                lifecycle.shutdown(reason=reason)
            finally:
                shutdown_done.set()

        threading.Thread(target=run, name="chat-shutdown", daemon=True).start()

    lifecycle.on_stop_request(lambda: _stop("stop_request"))
    lifecycle.install_signal_handlers(lambda name: _stop(f"signal:{name}"))
    lifecycle.mark_ready(
        chat_port=runtime_config.chat_port,
        http_ready=True,
        chat_provider=getattr(source, "provider", None).__class__.__name__
        if source is not None and getattr(source, "provider", None) is not None
        else "disabled",
    )
    log_event(
        log,
        "chat.started",
        environment=runtime_config.environment,
        mocks_allowed=runtime_config.mocks_allowed,
        source_enabled=source is not None,
    )
    thread = threading.Thread(
        target=_serve, args=(server, serving, log), name="chat-http-server", daemon=True
    )
    thread.start()
    try:
        while not shutdown_done.wait(0.5):
            if not thread.is_alive():
                _stop("http_server_exited")
                break
    except KeyboardInterrupt:  # pragma: no cover - interactive runs only
        _stop("keyboard_interrupt")
    _stop("serve_forever_returned")
    shutdown_done.wait(timeout=20.0)
    log_event(log, "chat.stopped", stopped=lifecycle.state == "STOPPED")
    return 0


def _serve(server: ThreadingHTTPServer, serving: threading.Event, logger: logging.Logger) -> None:
    """Run ``serve_forever`` in a worker so the main thread owns shutdown."""
    try:
        serving.set()
        server.serve_forever(poll_interval=0.5)
    except OSError as exc:  # pragma: no cover - socket closed during startup
        from ..runtime.redaction import safe_error

        logger.error("chat http failed: %s", safe_error(exc, limit=200))


def build_chat_source(service: ChatService):
    """Create the configured ingest runner, or ``None`` when chat input is off.

    ``CHAT_PROVIDER`` is opt-in; the default keeps PHASE 05-11 behaviour, where
    the only chat input is the local simulated ``POST /chat/message`` endpoint.
    """
    from ..chat.providers.config import ChatProviderConfig, build_provider
    from ..chat.providers.runner import ChatSourceRunner

    config = ChatProviderConfig.from_environment()
    provider = build_provider(config)
    if provider is None:
        return None
    return ChatSourceRunner(
        provider,
        service,
        poll_interval_seconds=config.poll_interval_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
