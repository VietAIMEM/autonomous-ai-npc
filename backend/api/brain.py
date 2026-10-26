"""Minimal HTTP API for controlled brain ticks and state inspection."""

from __future__ import annotations

import json
import logging
import math
import os
import signal
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, unquote, urlsplit

from ..brain.loop import AutonomousBrain
from ..brain.service import BrainService
from .chat import create_chat_server
from ..chat.service import ChatService, ChatServiceConfig
from ..learning.models import LearningConfig
from ..learning.repository import SQLiteSkillLearningRepository
from ..learning.service import SkillLearningService
from ..llm.base import LLMProvider
from ..llm.config import (
    LLMConfig,
    PROVIDER_MOCK as LLM_PROVIDER_MOCK,
    build_provider as build_llm_provider,
)
from ..motion_generation.service import MotionGenerationService
from ..runtime import instrument
from ..runtime.config import ConfigurationError, RuntimeConfig
from ..runtime.lifecycle import DuplicateInstanceError, RuntimeLifecycle
from ..runtime.obs import configure_logging, log_event
from ..runtime.redaction import safe_error
from ..runtime.stream_latch import LATCH_FILE_NAME, StreamStopLatch
from ..social.service import SocialAutonomyService
from ..skills.library import create_default_skill_library
from ..streaming.service import StreamingService
from ..viewer_web import HlsRelay, ViewerWebServer
from .stream import handle_stream_request, is_stream_path

#: Ceiling for one captured media POST; provider limits are stricter.
MAX_STREAM_BODY_BYTES = 2_000_000

log = logging.getLogger("room_agent.api.brain")

#: Provider kinds whose mock/real selection is enforced against the runtime
#: environment before the process accepts traffic.
ENFORCED_PROVIDER_ENV = {
    "llm": ("LLM_PROVIDER", ("mock", "openrouter", "nvidia", "cline", "ollama")),
    "tts": ("TTS_PROVIDER", ("mock", "openrouter", "kokoro", "disabled")),
    "chat": ("CHAT_PROVIDER", ("disabled", "mock", "twitch")),
    "stream": ("STREAM_PROVIDER", ("disabled", "mock", "rtmp", "srt")),
}


def enforce_provider_environment(config: RuntimeConfig, environ=None) -> dict[str, str]:
    """Reject an unknown or mock provider name before the process starts.

    ``RuntimeConfig.enforce_provider`` already knows whether mocks are allowed
    in this environment; this adds the "unknown name" half so a typo can never
    be quietly downgraded to a mock or to "disabled" in production.
    """
    env = os.environ if environ is None else environ
    resolved: dict[str, str] = {}
    for kind, (variable, valid) in ENFORCED_PROVIDER_ENV.items():
        raw = (env.get(variable) or "").strip().lower()
        if raw and raw not in valid:
            raise ConfigurationError(f"unknown_{kind}_provider:{raw}")
        config.enforce_provider(kind, raw)
        resolved[kind] = raw
    return resolved


def resolve_llm_provider(
    runtime_config: RuntimeConfig,
    environ=None,
) -> LLMProvider:
    """Build the explicitly configured LLM provider, never a silent mock."""
    llm_config = LLMConfig.from_environment(environ)
    runtime_config.enforce_provider("llm", llm_config.provider)
    return build_llm_provider(llm_config)


def create_brain_server(
    service: BrainService,
    host: str = "127.0.0.1",
    port: int = 8766,
    streaming: Optional[StreamingService] = None,
    motion_generation: Optional[MotionGenerationService] = None,
    *,
    motion_processing=None,
    chat_source=None,
    lifecycle: Optional[RuntimeLifecycle] = None,
    runtime_config: Optional[RuntimeConfig] = None,
    stream_latch: Optional[StreamStopLatch] = None,
) -> ThreadingHTTPServer:
    """Brain API plus the PHASE 13 ``/stream/*`` transport routes.

    ``streaming`` is optional on purpose: with no real provider configured the
    routes answer "disabled", so Unity stops capturing instead of retrying.
    The extra handles are only used by the read-only ``/health``, ``/status``
    and ``/diagnostics`` surfaces; keeping them optional preserves the
    existing test construction (`create_brain_server(service, host, port)`).
    """
    if streaming is not None and service is not None:
        # PHASE 27B: the initiative evaluator reads the real stream lifecycle
        # through this read-only handle, so a candidate that needs the audience
        # sees a start/stop in progress instead of guessing. The streaming
        # service remains the sole owner of every stream command.
        service.brain.streaming = streaming
    return ThreadingHTTPServer(
        (host, port),
        _build_handler(
            service,
            streaming,
            motion_generation,
            motion_processing=motion_processing,
            chat_source=chat_source,
            lifecycle=lifecycle,
            runtime_config=runtime_config,
            stream_latch=stream_latch,
        ),
    )


def create_default_service(
    brain: AutonomousBrain | None = None,
    chat_service: ChatService | None = None,
    memory_service=None,
    database_path=None,
    speech_service=None,
    tts_from_environment: bool = False,
    llm_from_environment: bool = False,
    decision_interval_seconds: float | None = None,
    runtime_config: RuntimeConfig | None = None,
    llm_provider: LLMProvider | None = None,
    stream_latch: StreamStopLatch | None = None,
) -> BrainService:
    """Create the single Brain service for this process.

    Development keeps the deterministic, offline defaults. The environment
    decides whether the LLM may be a mock: with ``ROOM_AGENT_ENV=production``
    an unset/absent real provider raises instead of silently running the mock.

    Both provider families are opt-in: automated runs and tests keep the
    deterministic mock unless the caller passes ``llm_from_environment`` /
    ``tts_from_environment`` (production passes both from ``main``).

    The two offline mocks stay distinct on purpose: the chat pipeline parses
    ``MockLLMProvider`` replies while the Brain plans with
    ``MockBrainLLMProvider``. A real provider serves both prompts.
    """

    from ..brain.loop import BrainConfig
    from ..brain.reasoning import MockBrainLLMProvider
    from ..speech.service import SpeechService
    from ..tts.config import PROVIDER_MOCK, TTSConfig

    runtime = runtime_config or RuntimeConfig.from_environment()
    if llm_provider is not None:
        chat_provider = brain_provider = llm_provider
    elif llm_from_environment:
        brain_provider = resolve_llm_provider(runtime)
        chat_provider = brain_provider
    else:
        runtime.enforce_provider("llm", LLM_PROVIDER_MOCK)
        brain_provider = MockBrainLLMProvider()
        chat_provider = build_llm_provider(LLMConfig(provider=LLM_PROVIDER_MOCK))
    chat = chat_service or ChatService(
        chat_provider, ChatServiceConfig.from_environment()
    )
    social = SocialAutonomyService(stream_latch=stream_latch)
    skill_library = create_default_skill_library()
    learning_config = LearningConfig(database_path=database_path) if database_path else LearningConfig()
    learning = SkillLearningService(
        skill_library,
        learning_config,
        SQLiteSkillLearningRepository(learning_config.database_path),
    )
    if speech_service is None:
        # Automated runs stay offline unless the process opts into the real
        # provider through the environment (see ``main``).
        if tts_from_environment:
            tts_config = TTSConfig.from_environment()
            runtime.enforce_provider("tts", tts_config.provider)
        else:
            runtime.enforce_provider("tts", PROVIDER_MOCK)
            tts_config = TTSConfig(provider=PROVIDER_MOCK)
        speech_service = SpeechService(tts_config=tts_config)
    if brain is None:
        brain_config = (
            BrainConfig(decision_interval_seconds=decision_interval_seconds)
            if decision_interval_seconds is not None
            else BrainConfig()
        )
        brain = AutonomousBrain(
            brain_provider,
            chat,
            brain_config,
            memory_service=memory_service,
            social_service=social,
            skill_library=skill_library,
            learning_service=learning,
            speech_service=speech_service,
        )
    elif getattr(brain, "speech", None) is None:
        brain.speech = speech_service
    if memory_service is not None:
        # PHASE 19: persist the stable profile once through the existing
        # metadata table so it is identical after a restart.
        try:
            memory_service.set_personality(brain.personality)
        except Exception:
            pass
    return BrainService(brain, chat)


def _learning_database_path() -> Optional[Path]:
    """PHASE 18: learned motions may live in their own database file.

    Defaults stay untouched: with ``LEARNING_DB_PATH`` unset the learning
    service keeps using ``LearningConfig``'s existing database.
    """
    value = os.environ.get("LEARNING_DB_PATH", "").strip()
    return Path(value) if value else None


def _memory_database_path() -> Optional[Path]:
    """PHASE 19: viewer relationships may live in their own database file.

    Defaults stay untouched: with ``MEMORY_DB_PATH`` unset the production
    server keeps using ``MemoryConfig``'s existing database.
    """
    value = os.environ.get("MEMORY_DB_PATH", "").strip()
    return Path(value) if value else None


def _decision_interval_seconds(default: float = 5.0) -> float:
    """PHASE 18: production tuning hook for the single existing Brain loop."""
    value = os.environ.get("BRAIN_DECISION_INTERVAL_SECONDS", "").strip()
    if not value:
        return default
    try:
        interval = float(value)
    except ValueError as exc:
        raise ValueError("BRAIN_DECISION_INTERVAL_SECONDS must be a number") from exc
    if not math.isfinite(interval) or interval < 0:
        raise ValueError("BRAIN_DECISION_INTERVAL_SECONDS must be finite and non-negative")
    return interval


def _build_handler(
    service: BrainService,
    streaming: Optional[StreamingService] = None,
    motion_generation: Optional[MotionGenerationService] = None,
    *,
    motion_processing=None,
    chat_source=None,
    lifecycle: Optional[RuntimeLifecycle] = None,
    runtime_config: Optional[RuntimeConfig] = None,
    stream_latch: Optional[StreamStopLatch] = None,
) -> type[BaseHTTPRequestHandler]:
    resolved_config = runtime_config or RuntimeConfig.from_environment()

    def _diagnostics_context() -> dict[str, Any]:
        return {
            "config": resolved_config,
            "lifecycle": lifecycle,
            "streaming": streaming,
            "chat_source": chat_source,
            "motion_generation": motion_generation,
            "motion_processing": motion_processing,
            "stream_latch": stream_latch,
        }

    class BrainAPIHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - http.server API
            path = urlsplit(self.path).path
            if path in {"/health", "/status", "/diagnostics"}:
                self._ops_route(path)
                return
            if is_stream_path(path):
                self._stream_route(path)
                return
            if path.startswith("/memory/viewer/"):
                self._memory_viewer_route(path)
                return
            if path == "/skills/runtime":
                self._send_json(HTTPStatus.OK, service.get_skills_runtime())
                return
            if path == "/personality/state":
                self._send_json(HTTPStatus.OK, service.get_personality_state())
                return
            if path == "/tts/state":
                self._send_json(HTTPStatus.OK, service.get_speech_state())
                return
            if path == "/tts/health":
                self._send_json(HTTPStatus.OK, service.get_speech_health())
                return
            if path in {"/motion-generation/state", "/motion-generation/health"}:
                if motion_generation is None:
                    self._send_json(HTTPStatus.OK, {"enabled": False})
                    return
                probe = (
                    motion_generation.health()
                    if path.endswith("/health")
                    else motion_generation.get_state()
                )
                self._send_json(HTTPStatus.OK, probe)
                return
            if path.startswith("/speech/audio/"):
                self._speech_audio_route(path)
                return
            if path.startswith("/speech/state/"):
                self._speech_state_route(path)
                return
            if path == "/skills/learning":
                self._send_json(HTTPStatus.OK, service.get_learning_state())
                return
            if path == "/skills/learning/compositions":
                self._send_json(HTTPStatus.OK, {"compositions": service.get_learning_compositions()})
                return
            if path.startswith("/skills/learning/"):
                self._learning_route(path)
                return
            if path.startswith("/skills/") and not path.startswith("/skills/learning/"):
                self._skill_route(path)
                return
            routes = {
                "/brain/state": service.get_state,
                "/brain/goal": service.get_goal,
                "/brain/plan": service.get_plan,
                "/brain/status": service.get_status,
                "/brain/autonomy": service.get_autonomy_state,
                "/memory/state": service.get_memory_state,
                "/social/state": service.get_social_state,
                "/conversation/state": service.get_conversation_state,
                "/skills": service.get_skills,
            }
            route = routes.get(path)
            if route is None:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            else:
                self._send_json(HTTPStatus.OK, route())

        def _ops_route(self, path: str) -> None:
            """Unified read-only runtime surface (never includes credentials)."""
            from . import diagnostics as diagnostics_module

            context = _diagnostics_context()
            try:
                if path == "/health":
                    payload = diagnostics_module.build_health(service, **context)
                elif path == "/status":
                    payload = diagnostics_module.build_status(service, **context)
                else:
                    payload = diagnostics_module.build_diagnostics(service, **context)
            except Exception as exc:  # diagnostics must never take the process down
                log_event(
                    log,
                    "diagnostics.failed",
                    route=path,
                    error=safe_error(exc, limit=200),
                )
                self._send_json(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    {"error": "diagnostics_unavailable", "reason": safe_error(exc, limit=200)},
                )
                return
            self._send_json(HTTPStatus.OK, payload)

        def _memory_viewer_route(self, path: str) -> None:
            parts = path.rstrip("/").split("/")
            if len(parts) != 4 or not parts[3]:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            viewer_id = unquote(parts[3]).strip()
            if not viewer_id or len(viewer_id) > 120:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_viewer_id"})
                return
            self._send_json(HTTPStatus.OK, service.get_viewer_memory(viewer_id))

        def _skill_route(self, path: str) -> None:
            parts = path.rstrip("/").split("/")
            if len(parts) != 3 or not parts[2]:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            skill_id = unquote(parts[2]).strip()
            if not skill_id or len(skill_id) > 120:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_skill_id"})
                return
            try:
                self._send_json(HTTPStatus.OK, service.get_skill(skill_id))
            except KeyError:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "skill_not_found"})

        def _learning_route(self, path: str) -> None:
            parts = path.rstrip("/").split("/")
            if len(parts) == 5 and parts[3] == "context" and parts[4]:
                self._send_json(HTTPStatus.OK, service.get_learning_context(unquote(parts[4])))
                return
            if len(parts) == 5 and parts[3] == "skill" and parts[4]:
                self._send_json(HTTPStatus.OK, service.get_learning_skill(unquote(parts[4])))
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

        def _speech_id(self, path: str, prefix: str) -> Optional[str]:
            parts = path.rstrip("/").split("/")
            # Request paths always split with a leading empty segment.
            if len(parts) != 4 or parts[1] != "speech" or parts[2] != prefix or not parts[3]:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return None
            request_id = unquote(parts[3]).strip()
            if not request_id or len(request_id) > 120:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_speech_request_id"})
                return None
            return request_id

        def _speech_state_route(self, path: str) -> None:
            request_id = self._speech_id(path, "state")
            if request_id is None:
                return
            try:
                self._send_json(HTTPStatus.OK, service.get_speech_request(request_id))
            except KeyError:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "speech_request_not_found"})

        def _speech_audio_route(self, path: str) -> None:
            request_id = self._speech_id(path, "audio")
            if request_id is None:
                return
            payload = service.get_speech_audio(request_id)
            if payload is None:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "audio_not_ready"})
                return
            audio_bytes, mime_type, filename = payload
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", mime_type)
            self.send_header("Content-Length", str(len(audio_bytes)))
            self.send_header("Content-Disposition", f'inline; filename="{filename}"')
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(audio_bytes)

        def do_POST(self) -> None:  # noqa: N802 - http.server API
            path = urlsplit(self.path).path
            if is_stream_path(path):
                self._stream_route(path)
                return
            if path == "/skills/learning/reset":
                service.reset_learning()
                self._send_json(HTTPStatus.OK, {"learning": service.get_learning_state()})
                return
            if path.startswith("/speech/"):
                self._speech_post_route(path)
                return
            if path != "/brain/tick":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return

            try:
                payload = self._read_json()
            except (ValueError, json.JSONDecodeError):
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_json"})
                return

            snapshot = payload.get("world")
            if not isinstance(snapshot, dict):
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "world_required"})
                return
            result = payload.get("result")
            if result is not None and not isinstance(result, dict):
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_result"})
                return
            viewer_response = payload.get("viewer_response")
            if viewer_response is not None and (
                not isinstance(viewer_response, str)
                or len(viewer_response.strip()) > 40
            ):
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_viewer_response"})
                return
            if result is not None and viewer_response is not None:
                result = {**result, "viewer_response": viewer_response}

            try:
                tick_result = service.tick(snapshot, result)
            except ValueError as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                return
            self._send_json(HTTPStatus.OK, {"tick": tick_result.to_dict()})

        def _speech_post_route(self, path: str) -> None:
            actions = {
                "playing": service.mark_speech_playing,
                "completed": service.mark_speech_completed,
            }
            parts = path.rstrip("/").split("/")
            if len(parts) != 4 or parts[1] != "speech" or not parts[3]:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            request_id = unquote(parts[3]).strip()
            if not request_id or len(request_id) > 120:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_speech_request_id"})
                return
            if parts[2] == "cancel":
                self._send_json(
                    HTTPStatus.OK,
                    {"cancelled": service.cancel_speech(request_id, "unity_cancelled")},
                )
                return
            action = actions.get(parts[2])
            if action is None:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            self._send_json(HTTPStatus.OK, {"updated": bool(action(request_id))})

        def _stream_route(self, path: str) -> None:
            """PHASE 13 transport routes. Captured bytes stay in memory only."""
            parts = urlsplit(self.path)
            method = self.command or "GET"
            body = b""
            if method == "POST":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except (TypeError, ValueError):
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_content_length"})
                    return
                if length < 0 or length > MAX_STREAM_BODY_BYTES:
                    # Without the whole body the connection cannot be reused.
                    self.close_connection = True
                    self._send_json(
                        HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "body_too_large"}
                    )
                    return
                if length:
                    body = self.rfile.read(length)
            status, payload = handle_stream_request(
                streaming,
                path=path,
                method=method,
                query=parse_qs(parts.query),
                body=body,
                latch=stream_latch,
            )
            self._send_json(status, payload)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 100_000:
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

    return BrainAPIHandler


def _env_flag(name: str, default: bool = False) -> bool:
    value = (os.environ.get(name) or "").strip().lower()
    if not value:
        return default
    return value not in {"0", "false", "no", "off", "disabled"}


def _build_motion_services(
    learning,
    runtime_config: RuntimeConfig,
) -> tuple[Any, Any, Any]:
    """Optional motion workers. Off unless the operator opts in explicitly."""
    generation = None
    processing = None
    import_service = None
    if _env_flag("MOTION_GENERATION_ENABLED"):
        from ..motion_generation.models import (
            PROVIDER_MOCK as MOTION_PROVIDER_MOCK,
            MotionGenerationConfig,
        )

        provider = (os.environ.get("MOTION_PROVIDER") or MOTION_PROVIDER_MOCK).strip().lower()
        if provider not in {MOTION_PROVIDER_MOCK, "motius"}:
            raise ConfigurationError(f"unknown_motion_provider:{provider}")
        runtime_config.enforce_provider("motion_generation", provider)
        generation = MotionGenerationService(
            config=MotionGenerationConfig(provider=provider)
        )
    if _env_flag("MOTION_PROCESSING_ENABLED"):
        from ..motion_processing.service import MotionProcessingService
        from ..motion_bridge import build_motion_processing_adapter

        unity_mode = (os.environ.get("MOTION_UNITY_MODE") or "batch").strip().lower()
        try:
            processing_adapter = build_motion_processing_adapter(unity_mode)
        except ValueError as exc:
            raise ConfigurationError(str(exc)) from exc
        processing = MotionProcessingService(learning, adapter=processing_adapter)
        log_event(
            log,
            "motion.processing_adapter",
            adapter=getattr(processing_adapter, "name", type(processing_adapter).__name__),
        )
    # PHASE 23: the admin dashboard drives the existing Phase 16 import
    # pipeline, which stays off unless the operator opts in like the others.
    if _env_flag("MOTION_IMPORT_ENABLED"):
        from ..motion_import import MotionFileImportService
        from ..motion_bridge import build_motion_import_adapter

        unity_mode = (os.environ.get("MOTION_UNITY_MODE") or "batch").strip().lower()
        try:
            import_adapter = build_motion_import_adapter(unity_mode)
        except ValueError as exc:
            raise ConfigurationError(str(exc)) from exc
        import_service = MotionFileImportService(learning, adapter=import_adapter)
        log_event(
            log,
            "motion.import_adapter",
            adapter=getattr(import_adapter, "name", type(import_adapter).__name__),
        )
    return generation, processing, import_service


def main() -> int:
    from ..admin_web import AdminConsole, AdminSettingsStore, AdminWebServer
    from ..memory import MemoryConfig, MemoryService, SQLiteMemoryRepository
    from ..memory.models import DEFAULT_MEMORY_DB_PATH
    from ..tts.config import TTSConfig

    runtime_config = RuntimeConfig.from_environment()
    project_root = Path(__file__).resolve().parents[2]
    configure_logging(runtime_config)
    try:
        enforce_provider_environment(runtime_config)
    except ConfigurationError as exc:
        log.error("startup refused: %s", safe_error(exc, limit=200))
        print(json.dumps({"error": safe_error(exc, limit=200)}, indent=2))
        return 2

    lifecycle = RuntimeLifecycle(runtime_config, role="backend")
    try:
        lifecycle.start()
    except DuplicateInstanceError as exc:
        log.error("duplicate instance refused: %s", safe_error(exc, limit=200))
        print(json.dumps({"error": str(exc)}, indent=2))
        return 3

    # PHASE 19: production runs may point at dedicated viewer-memory and
    # relationship databases; unset variables keep the existing default.
    config = MemoryConfig(
        database_path=_memory_database_path() or DEFAULT_MEMORY_DB_PATH
    )
    memory = MemoryService(SQLiteMemoryRepository(config.database_path), config)
    # PHASE 20: a deliberate STOP_STREAM must survive a backend restart, so the
    # one latch bit lives in the runtime state directory instead of only in RAM.
    stream_latch = StreamStopLatch(runtime_config.state_dir / LATCH_FILE_NAME)
    # PHASE 23: allow-listed admin settings are persisted next to the token so
    # a "requires restart" value survives one. No secret ever lives here.
    admin_settings = AdminSettingsStore(runtime_config.state_dir)
    stored_settings = admin_settings.values()
    streaming = None
    chat_source = None
    chat_server = None
    chat_thread = None
    relay = None
    viewer_server = None
    viewer_thread = None
    admin_server = None
    admin_thread = None
    motion_generation = None
    motion_processing = None
    motion_import = None
    brain_server = None
    service = None
    shutdown_done = threading.Event()
    serving = threading.Event()
    chat_provider_name = "disabled"

    def _safe(name: str, action) -> None:
        try:
            action()
        except Exception as exc:
            instrument.record_error("shutdown", f"{name}:{safe_error(exc, limit=120)}")
            log_event(
                log,
                "shutdown.step_failed",
                step=name,
                error=safe_error(exc, limit=160),
            )

    try:
        # PHASE 18/19: behavior-aware skill selection reads the learned motion
        # library, which production keeps in its own database file. Both
        # overrides are opt-in; unset variables keep the PHASE 01-17 defaults.
        # A value saved through the dashboard is the operator's explicit choice
        # and outranks the environment default; the environment still applies
        # until the first save.
        stored_interval = admin_settings.effective("decision_interval_seconds")
        decision_interval_seconds = (
            float(stored_interval)
            if stored_interval is not None
            else _decision_interval_seconds()
        )
        service = create_default_service(
            memory_service=memory,
            database_path=_learning_database_path(),
            llm_from_environment=True,
            tts_from_environment=True,
            decision_interval_seconds=decision_interval_seconds,
            runtime_config=runtime_config,
            stream_latch=stream_latch,
        )
        # Live values are re-applied on every start so a restart keeps them.
        if stored_settings.get("learning_enabled") is not None:
            try:
                service.brain.learning.set_enabled(bool(stored_settings["learning_enabled"]))
            except Exception as exc:
                log_event(
                    log,
                    "admin.settings.startup_failed",
                    key="learning_enabled",
                    error=safe_error(exc, limit=160),
                )
        stored_level = stored_settings.get("log_level")
        if stored_level:
            level = getattr(logging, str(stored_level).upper(), None)
            if isinstance(level, int):
                logging.getLogger().setLevel(level)
                logging.getLogger("room_agent").setLevel(level)
        tts_config = TTSConfig.from_environment()
        runtime_config.enforce_provider("tts", tts_config.provider)

        motion_generation, motion_processing, motion_import = _build_motion_services(
            service.brain.learning, runtime_config
        )

        # PHASE 13: real streaming transport is opt-in through STREAM_PROVIDER,
        # so PHASE 01-12 runs (simulation only) are untouched when it is unset.
        from .stream import build_streaming_source
        from ..streaming.config import StreamingConfig

        stream_config = StreamingConfig.from_environment()
        runtime_config.enforce_provider("stream", stream_config.provider)

        # PHASE 22: browsers cannot play SRT, so the viewer web re-publishes the
        # same production transport as HLS. The relay listens before the SRT
        # encoder connects; with any other transport it stays unconfigured and
        # the page degrades to an honest OFFLINE state instead of faking LIVE.
        relay = HlsRelay(
            ffmpeg_path=stream_config.ffmpeg_path,
            listen_url=stream_config.server_url,
            directory=runtime_config.temp_dir
            / "viewer_web"
            / f"hls_{runtime_config.viewer_port}",
            enabled=stream_config.provider == "srt",
        )
        relay.start()

        streaming = build_streaming_source()
        if streaming is not None:
            streaming.start()

        # PHASE 12: real viewer chat is opt-in through CHAT_PROVIDER. The Brain
        # keeps reading the same ChatService queue either way.
        from .chat import build_chat_source
        from ..chat.providers.config import ChatProviderConfig

        chat_config = ChatProviderConfig.from_environment()
        runtime_config.enforce_provider("chat", chat_config.provider)
        chat_source = build_chat_source(service.chat_service)
        if chat_source is not None:
            chat_source.start()

        brain_server = create_brain_server(
            service,
            host=runtime_config.host,
            port=runtime_config.brain_port,
            streaming=streaming,
            motion_generation=motion_generation,
            motion_processing=motion_processing,
            chat_source=chat_source,
            runtime_config=runtime_config,
            lifecycle=lifecycle,
            stream_latch=stream_latch,
        )
        chat_server = create_chat_server(
            service.chat_service,
            host=runtime_config.host,
            port=runtime_config.chat_port,
            source=chat_source,
        )
        viewer_server = ViewerWebServer(
            service,
            host=runtime_config.host,
            port=runtime_config.viewer_port,
            relay=relay,
            streaming=streaming,
        )
        # PHASE 23: the privileged surface is a separate authenticated server on
        # its own port; it shares the same services and never widens the viewer
        # API. The admin token resolves from the environment or the state dir.
        admin_server = AdminWebServer(
            AdminConsole(
                service,
                config=runtime_config,
                lifecycle=lifecycle,
                streaming=streaming,
                chat_source=chat_source,
                motion_generation=motion_generation,
                motion_processing=motion_processing,
                motion_import=motion_import,
                stream_latch=stream_latch,
                settings=admin_settings,
                project_root=project_root,
            ),
            host=runtime_config.host,
            port=runtime_config.admin_port,
            upload_dir=runtime_config.temp_dir / "admin_uploads",
            project_root=project_root,
        )
        chat_provider_name = chat_config.provider
    except Exception as exc:
        log_event(
            log,
            "startup.failed",
            error=safe_error(exc, limit=200),
        )
        _safe(
            "viewer.web.close",
            lambda: viewer_server.server_close() if viewer_server is not None else None,
        )
        _safe(
            "admin.web.close",
            lambda: admin_server.server_close() if admin_server is not None else None,
        )
        _safe("chat.stop", lambda: chat_source.stop() if chat_source is not None else None)
        _safe(
            "chat.server.close",
            lambda: chat_server.server_close() if chat_server is not None else None,
        )
        _safe("stream.stop", lambda: streaming.stop() if streaming is not None else None)
        _safe("viewer.relay.stop", lambda: relay.stop() if relay is not None else None)
        _safe(
            "motion.shutdown",
            lambda: [
                handle.shutdown()
                for handle in (motion_generation, motion_processing)
                if handle is not None
            ],
        )
        _safe(
            "speech.shutdown",
            lambda: service.speech_service.shutdown()
            if service is not None and service.speech_service is not None
            else None,
        )
        _safe("learning.close", lambda: service.brain.learning.close() if service else None)
        _safe("memory.close", memory.close)
        lifecycle.shutdown(reason="startup_failed")
        return 4

    chat_thread = threading.Thread(
        target=chat_server.serve_forever,
        name="chat-ingest-server",
        daemon=True,
    )
    chat_thread.start()
    viewer_thread = threading.Thread(
        target=viewer_server.serve_forever,
        name="viewer-web-server",
        daemon=True,
    )
    viewer_thread.start()
    admin_thread = threading.Thread(
        target=admin_server.serve_forever,
        name="admin-web-server",
        daemon=True,
    )
    admin_thread.start()

    def _stop_chat_ingest() -> None:
        """Stop external input, then the chat HTTP intake, then close it."""
        if chat_source is not None:
            chat_source.stop()
        if chat_server is None:
            return
        if chat_thread is not None:
            if chat_thread.is_alive():
                chat_server.shutdown()
            chat_thread.join(timeout=3.0)
        chat_server.server_close()

    def _stop_brain_http() -> None:
        """Never wait on ``shutdown()`` unless ``serve_forever`` is running."""
        if serving.wait(timeout=5.0):
            brain_server.shutdown()
        brain_server.server_close()

    def _stop_viewer_web() -> None:
        """Close the viewer HTTP surface; the relay is stopped with the pipeline."""
        if viewer_server is None:
            return
        if viewer_thread is not None and viewer_thread.is_alive():
            viewer_server.shutdown()
        if viewer_thread is not None:
            viewer_thread.join(timeout=3.0)
        viewer_server.server_close()

    def _stop_admin_web() -> None:
        """Close the privileged HTTP surface; sessions live only in memory."""
        if admin_server is None:
            return
        if admin_thread is not None and admin_thread.is_alive():
            admin_server.shutdown()
        if admin_thread is not None:
            admin_thread.join(timeout=3.0)
        admin_server.server_close()

    def _stop_viewer_relay() -> None:
        if relay is not None:
            relay.stop()

    # Steps run in reverse registration order, so register the reverse of the
    # intended shutdown order: HTTP intake first, storage last.
    desired_steps = [
        ("brain.http.stop", _stop_brain_http),
        ("admin.web.stop", _stop_admin_web),
        ("viewer.web.stop", _stop_viewer_web),
        ("chat.stop", _stop_chat_ingest),
        ("stream.stop", lambda: streaming.stop() if streaming is not None else None),
        ("viewer.relay.stop", _stop_viewer_relay),
        (
            "motion.shutdown",
            lambda: [
                handle.shutdown()
                for handle in (motion_generation, motion_processing)
                if handle is not None
            ],
        ),
        (
            "speech.shutdown",
            lambda: service.speech_service.shutdown()
            if service is not None and service.speech_service is not None
            else None,
        ),
        (
            "learning.close",
            lambda: service.brain.learning.close() if service is not None else None,
        ),
        ("memory.close", memory.close),
        (
            "retention.cleanup",
            lambda: _run_retention(runtime_config),
        ),
    ]
    for name, action in reversed(desired_steps):
        lifecycle.register_shutdown(name, lambda action=action: action(), timeout_seconds=10.0)

    stop_triggered = threading.Event()
    stop_lock = threading.Lock()

    def _begin_shutdown(reason: str) -> None:
        if stop_triggered.is_set():
            return
        with stop_lock:
            if stop_triggered.is_set():
                return
            stop_triggered.set()

        def run() -> None:
            try:
                lifecycle.shutdown(reason=reason)
            finally:
                shutdown_done.set()

        threading.Thread(target=run, name="shutdown-worker", daemon=True).start()

    lifecycle.on_stop_request(lambda: _begin_shutdown("stop_request"))
    lifecycle.install_signal_handlers(lambda name: _begin_shutdown(f"signal:{name}"))
    lifecycle.mark_ready(
        brain_port=runtime_config.brain_port,
        chat_port=runtime_config.chat_port,
        viewer_port=runtime_config.viewer_port,
        viewer_web=viewer_server is not None,
        admin_port=runtime_config.admin_port,
        admin_web=admin_server is not None,
        http_ready=True,
    )
    log_event(
        log,
        "runtime.started",
        environment=runtime_config.environment,
        mocks_allowed=runtime_config.mocks_allowed,
        streaming=bool(streaming),
        chat_provider=chat_provider_name,
        viewer_port=runtime_config.viewer_port,
        admin_port=runtime_config.admin_port,
        admin_import=motion_import is not None,
    )
    try:
        serving.set()
        brain_server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:  # pragma: no cover - interactive runs only
        _begin_shutdown("keyboard_interrupt")
    except OSError as exc:  # pragma: no cover - socket closed during startup
        log_event(log, "brain.http.failed", error=safe_error(exc, limit=200))
        _begin_shutdown("brain_http_failed")
    finally:
        _begin_shutdown("serve_forever_returned")
        shutdown_done.wait(timeout=30.0)
        # Idempotent: returns the recorded report when already stopped.
        lifecycle.shutdown(reason="final")
    report = lifecycle.status().get("shutdown")
    if isinstance(report, dict) and not report.get("ok", True):
        log_event(log, "runtime.shutdown_incomplete", steps=report.get("steps"))
    log_event(log, "runtime.stopped", stopped=lifecycle.state == "STOPPED")
    return 0


def _run_retention(config: RuntimeConfig) -> None:
    """Bounded retention sweep; never touches databases or learned state."""
    from ..runtime.retention import apply_retention

    apply_retention(config)


if __name__ == "__main__":
    raise SystemExit(main())
