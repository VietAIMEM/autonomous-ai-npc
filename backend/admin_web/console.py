"""Projections and commands for the PHASE 23 admin dashboard.

The console owns no business logic. Every read is a bounded projection of an
existing service (Brain, memory, learning, skills, speech, chat, stream,
motion, diagnostics); every write forwards to the existing service method that
already enforces its own gates (``VALID + COMPATIBLE``, skill conditions,
cooldowns, allowlists, the ``SkillExecutor``, the stream stop latch).

Two rules hold everywhere in this module:

* the browser never talks to SQLite, Unity, FFmpeg, a provider or a shell; it
  only sees the structured results below, and
* a command reports the truth. Nothing says "started" until the lifecycle or
  service confirms it.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from ..api import diagnostics as diagnostics_module
from ..api.stream import handle_stream_request
from ..runtime import instrument
from ..runtime.lifecycle import RoomAgentController
from ..runtime.obs import log_event
from ..runtime.redaction import safe_error
from .logs import AdminLogReader, LogQuery
from .settings import AdminSettingsStore, SettingValidationError

log = logging.getLogger("room_agent.admin_web.console")

RUNTIME_ACTIONS = ("start", "stop", "restart")
MAX_TTS_TEST_SECONDS = 12.0
MAX_MOTION_RECORDS = 50
MAX_LEARNING_ROWS = 200
MAX_MEMORY_EPISODES = 50
MAX_MEMORY_SEMANTIC = 50
MAX_VIEWERS = 100


class AdminActionError(RuntimeError):
    """A structured, browser-safe command failure."""

    def __init__(self, code: str, *, detail: Any = None) -> None:
        super().__init__(code)
        self.code = code
        self.detail = detail

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"ok": False, "error": self.code}
        if self.detail is not None:
            payload["detail"] = self.detail
        return payload


def _now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _rejected(
    exc: Exception, code: str, *, event: str, **fields: Any
) -> AdminActionError:
    """Record the cause server-side; never echo it across the admin boundary.

    Downstream exception text can embed file contents, operator input or
    credential-shaped values, so browser-facing failures carry the machine
    code only. The redacted cause stays in the runtime log for operators.
    """
    log_event(log, event, error=safe_error(exc, limit=200), **fields)
    return AdminActionError(code)


def _safe(call: Callable[[], Any], fallback: Any = None) -> Any:
    """Run a read-only probe; a broken subsystem must not break the page."""
    try:
        return call()
    except Exception as exc:  # diagnostics must never raise
        return {"error": safe_error(exc, limit=160)} if fallback is None else fallback


class AdminConsole:
    """The whole admin surface, expressed as methods over existing services."""

    def __init__(
        self,
        service: Any,
        *,
        config: Any,
        lifecycle: Any = None,
        streaming: Any = None,
        chat_source: Any = None,
        motion_generation: Any = None,
        motion_processing: Any = None,
        motion_import: Any = None,
        stream_latch: Any = None,
        settings: Optional[AdminSettingsStore] = None,
        logs: Optional[AdminLogReader] = None,
        project_root: Optional[Path] = None,
        python_executable: Optional[str] = None,
    ) -> None:
        self.service = service
        self.config = config
        self.lifecycle = lifecycle
        self.streaming = streaming
        self.chat_source = chat_source
        self.motion_generation = motion_generation
        self.motion_processing = motion_processing
        self.motion_import = motion_import
        self.stream_latch = stream_latch
        self.settings = settings or AdminSettingsStore(config.state_dir)
        self.log_reader = logs or AdminLogReader(config.log_dir)
        self.project_root = Path(project_root) if project_root else None
        self._python = python_executable or sys.executable
        self._dispatch_lock = threading.Lock()
        self._last_dispatch: dict[str, Any] = {}

    # ------------------------------------------------------------ diagnostics

    def _context(self) -> dict[str, Any]:
        return {
            "config": self.config,
            "lifecycle": self.lifecycle,
            "streaming": self.streaming,
            "chat_source": self.chat_source,
            "motion_generation": self.motion_generation,
            "motion_processing": self.motion_processing,
            "stream_latch": self.stream_latch,
        }

    def health(self) -> dict[str, Any]:
        return diagnostics_module.build_health(self.service, **self._context())

    def status(self) -> dict[str, Any]:
        return diagnostics_module.build_status(self.service, **self._context())

    def diagnostics(self) -> dict[str, Any]:
        return diagnostics_module.build_diagnostics(self.service, **self._context())

    def overview(self) -> dict[str, Any]:
        """One honest snapshot for the landing page."""
        status = self.status()
        health = self.health()
        behavior = _safe(self.service.get_behavior_state, {})
        goal = _safe(self.service.get_goal, {})
        social = _safe(self.service.get_social_state, {})
        chat_source_state = (
            _safe(self.chat_source.get_state, {}) if self.chat_source else {}
        )
        streaming_state = (
            _safe(self.streaming.get_state, {}) if self.streaming else {}
        )
        speech = _safe(self.service.get_speech_summary, {})
        runtime = _safe(self.service.runtime_metrics, {})
        # PHASE 26A: bounded turn-taking read-model for the operator view.
        conversation = _safe(getattr(self.service, "get_conversation_state", None), {})
        latch = (
            _safe(self.stream_latch.snapshot, {}) if self.stream_latch else None
        )
        viewers_active = chat_source_state.get("active_viewers")
        if not isinstance(viewers_active, list):
            viewers_active = social.get("active_viewers") if isinstance(social, Mapping) else None
        return {
            "generated_at": _now_iso(),
            "status": status.get("status"),
            "environment": status.get("environment"),
            "process": status.get("process"),
            "components": health.get("components"),
            "health_summary": status.get("health"),
            "active_skill": status.get("active_skill"),
            "active_goal": goal.get("goal") if isinstance(goal, Mapping) else None,
            "behavior": behavior,
            "viewer_count": len(viewers_active) if isinstance(viewers_active, list) else None,
            "active_viewers": viewers_active if isinstance(viewers_active, list) else [],
            "queues": status.get("queues"),
            "stream": {
                "state": streaming_state.get("state") if isinstance(streaming_state, Mapping) else None,
                "live": streaming_state.get("live") if isinstance(streaming_state, Mapping) else None,
                "provider": streaming_state.get("provider") if isinstance(streaming_state, Mapping) else None,
                "stop_latch": latch,
            },
            "speech": {
                "provider": speech.get("provider") if isinstance(speech, Mapping) else None,
                "state": speech.get("state") if isinstance(speech, Mapping) else None,
                "enabled": speech.get("enabled") if isinstance(speech, Mapping) else None,
                "queue_length": speech.get("queue_length") if isinstance(speech, Mapping) else None,
            },
            "brain": {
                "tick_count": runtime.get("tick_count") if isinstance(runtime, Mapping) else None,
                "tick_in_progress": runtime.get("tick_in_progress") if isinstance(runtime, Mapping) else None,
                "last_tick_age_seconds": runtime.get("last_tick_age_seconds") if isinstance(runtime, Mapping) else None,
                "last_tick_at": runtime.get("last_tick_at") if isinstance(runtime, Mapping) else None,
                "uptime_seconds": runtime.get("uptime_seconds") if isinstance(runtime, Mapping) else None,
            },
            "conversation": (
                {
                    key: conversation.get(key)
                    for key in (
                        "conversation_state",
                        "current_intent",
                        "current_topic",
                        "response_status",
                        "active_viewer",
                        "turn_count",
                        "pending_question",
                        "pending_action",
                        "interruption_reason",
                        "speech_state",
                        # PHASE 26B: bounded expression coordination fields.
                        "response_tone",
                        "reaction_intensity",
                        "active_gesture",
                        "gesture_reason",
                        "behavior_state",
                    )
                }
                if isinstance(conversation, Mapping)
                else None
            ),
            "recent_errors": status.get("recent_errors"),
            "stream_stop_latch": status.get("stream_stop_latch"),
        }

    # -------------------------------------------------------------- providers

    def providers(self) -> dict[str, Any]:
        diagnostics = self.diagnostics()
        providers = diagnostics.get("providers") or {}
        llm = dict(providers.get("llm") or {})
        tts = dict(providers.get("tts") or {})
        chat = dict(providers.get("chat") or {})
        stream = dict(providers.get("stream") or {})
        speech_health = _safe(self.service.get_speech_health, {})
        latency: dict[str, Any] = {}
        if isinstance(speech_health, Mapping):
            latency["tts"] = speech_health.get("latency")
        return {
            "llm": {
                "provider": llm.get("provider"),
                "model": llm.get("model"),
                "endpoint": llm.get("endpoint"),
                "configured": llm.get("configured"),
                "mock": llm.get("mock"),
                "reachable": llm.get("reachable"),
                "credentials": llm.get("credential_source") or llm.get("credentials"),
                "detail": llm,
            },
            "tts": {
                "provider": tts.get("provider"),
                "model": tts.get("model"),
                "voice": tts.get("voice"),
                "configured": tts.get("configured"),
                "reachable": tts.get("reachable"),
                "latency": latency.get("tts"),
                "detail": tts,
            },
            "chat": {
                "provider": chat.get("provider"),
                "configured": chat.get("configured"),
                "channels": chat.get("channels"),
                "detail": chat,
            },
            "stream": {
                "provider": stream.get("provider"),
                "platform": stream.get("platform"),
                "configured": stream.get("configured"),
                "real_transport": stream.get("real_transport"),
                "detail": stream,
            },
            "motion": {
                "generation_enabled": self.motion_generation is not None,
                "processing_enabled": self.motion_processing is not None,
                "import_enabled": self.motion_import is not None,
            },
            "secrets_exposed": False,
        }

    # -------------------------------------------------------------------- TTS

    def tts(self, request_id: Optional[str] = None) -> dict[str, Any]:
        state = _safe(self.service.get_speech_state, {"enabled": False})
        health = _safe(self.service.get_speech_health, {"enabled": False})
        selected = None
        if request_id:
            selected = _safe(
                lambda: self.service.get_speech_request(request_id), None
            )
            if selected is None:
                raise AdminActionError("speech_request_not_found")
        recent_errors = [
            entry
            for entry in (state.get("recent") or [])
            if isinstance(entry, Mapping) and entry.get("error")
        ][-10:]
        return {
            "state": state,
            "health": health,
            "request": selected,
            "recent_errors": recent_errors,
            "secrets_exposed": False,
        }

    def tts_test(self, text: str, *, wait_seconds: float = MAX_TTS_TEST_SECONDS) -> dict[str, Any]:
        """Enqueue one real utterance through the existing speech service."""
        speech = self.service.speech_service
        if speech is None:
            raise AdminActionError("speech_unavailable")
        body = (text or "").strip()
        if not body:
            raise AdminActionError("text_required")
        if len(body) > 200:
            raise AdminActionError("text_too_long")
        request = speech.request(body, source="admin", dedup_key=None)
        state = request.state.value if hasattr(request.state, "value") else str(request.state)
        if state in {"FAILED", "CANCELLED"}:
            # ``SpeechService._reject`` reports a refused utterance as FAILED
            # with a machine-readable reason; a rejection is never a success.
            return {
                "ok": False,
                "error": request.error or "speech_rejected",
                "state": state,
                "request": request.to_dict(),
            }
        deadline = time.monotonic() + max(0.0, min(float(wait_seconds), MAX_TTS_TEST_SECONDS))
        while time.monotonic() < deadline:
            if request.state.value in {"COMPLETED", "FAILED", "CANCELLED"}:
                break
            time.sleep(0.25)
            refreshed = speech.get(request.request_id)
            if refreshed is not None:
                request = refreshed
        state = request.state.value
        result = request.to_dict()
        if state == "COMPLETED":
            return {"ok": True, "state": state, "request": result}
        if state in {"FAILED", "CANCELLED"}:
            return {
                "ok": False,
                "error": request.error or f"speech_{state.lower()}",
                "state": state,
                "request": result,
            }
        return {"ok": True, "state": state, "pending": True, "request": result}

    # ------------------------------------------------------------------- chat

    def chat(self) -> dict[str, Any]:
        service_state = _safe(self.service.chat_service.get_state, {})
        source_state = (
            _safe(self.chat_source.get_state, {"enabled": False})
            if self.chat_source
            else {"enabled": False, "reason": "no_real_chat_input"}
        )
        source_health = (
            _safe(self.chat_source.health, {}) if self.chat_source else {}
        )
        return {
            "queue": service_state,
            "source": source_state,
            "health": source_health,
            "recent_errors": [
                error
                for error in (
                    source_state.get("last_error") if isinstance(source_state, Mapping) else None,
                    source_health.get("last_error") if isinstance(source_health, Mapping) else None,
                )
                if error
            ][:5],
        }

    def chat_reconnect(self) -> dict[str, Any]:
        """Restart the existing chat ingest runner; never a second pipeline."""
        if self.chat_source is None:
            raise AdminActionError("chat_source_unavailable")
        try:
            self.chat_source.stop(timeout=3.0)
            self.chat_source.start()
        except Exception as exc:
            raise _rejected(exc, "chat_reconnect_failed", event="admin.chat.reconnect_failed") from exc
        log_event(log, "admin.chat.reconnect")
        state = _safe(self.chat_source.get_state, {})
        return {"ok": True, "state": state}

    # ----------------------------------------------------------------- stream

    def stream(self) -> dict[str, Any]:
        state = (
            _safe(self.streaming.get_state, {"state": "UNAVAILABLE"})
            if self.streaming
            else {"state": "UNAVAILABLE", "reason": "transport_disabled"}
        )
        health = _safe(self.streaming.health, {}) if self.streaming else {}
        latch = _safe(self.stream_latch.snapshot, None) if self.stream_latch else None
        return {
            "state": state,
            "health": health,
            "stop_latch": latch,
            "provider": getattr(getattr(self.streaming, "provider", None), "name", "disabled"),
            "commands": ["start", "stop", "reconnect"],
        }

    def stream_command(self, action: str, *, camera: str = "", wait_seconds: Optional[float] = None) -> dict[str, Any]:
        """Route through the exact PHASE 13 handler so latch rules hold."""
        if self.streaming is None:
            raise AdminActionError("stream_unavailable")
        normalized = (action or "").strip().lower()
        if normalized == "start":
            status, payload = handle_stream_request(
                self.streaming,
                path="/stream/start",
                method="POST",
                query={"camera": camera, "wait": wait_seconds},
                latch=self.stream_latch,
            )
        elif normalized == "stop":
            status, payload = handle_stream_request(
                self.streaming,
                path="/stream/stop",
                method="POST",
                query={"wait": wait_seconds},
                latch=self.stream_latch,
            )
        elif normalized == "reconnect":
            _status, stopped = handle_stream_request(
                self.streaming,
                path="/stream/stop",
                method="POST",
                query={"wait": 5.0},
                latch=self.stream_latch,
            )
            status, payload = handle_stream_request(
                self.streaming,
                path="/stream/start",
                method="POST",
                query={"camera": camera, "wait": wait_seconds},
                latch=self.stream_latch,
            )
            payload = {**payload, "previous_stop": stopped}
        else:
            raise AdminActionError("unknown_stream_action")
        payload = dict(payload)
        payload["ok"] = bool(payload.get("ok")) and int(status) < 400
        payload["http_status"] = int(status)
        payload["stop_latch"] = _safe(self.stream_latch.snapshot, None) if self.stream_latch else None
        log_event(log, "admin.stream.command", action=normalized, ok=payload["ok"])
        return payload

    # ----------------------------------------------------------------- motion

    def motion(self) -> dict[str, Any]:
        learning = getattr(getattr(self.service, "brain", None), "learning", None)
        motions: list[dict[str, Any]] = []
        if learning is not None:
            raw = _safe(learning.motions, [])
            if isinstance(raw, list):
                motions = [_operator_motion(item) for item in raw if isinstance(item, Mapping)]
        return {
            "motions": motions[:MAX_MOTION_RECORDS],
            "motion_count": len(motions),
            "import": (
                _safe(self.motion_import.get_state, {"enabled": False})
                if self.motion_import
                else {"enabled": False, "reason": "MOTION_IMPORT_ENABLED not set"}
            ),
            "processing": (
                _safe(self.motion_processing.get_state, {"enabled": False})
                if self.motion_processing
                else {"enabled": False, "reason": "MOTION_PROCESSING_ENABLED not set"}
            ),
            "generation": (
                _safe(self.motion_generation.get_state, {"enabled": False})
                if self.motion_generation
                else {"enabled": False, "reason": "MOTION_GENERATION_ENABLED not set"}
            ),
            "gate": "VALID + COMPATIBLE enforced by MotionProcessingService.approve",
        }

    def motion_job(self, kind: str, request_id: str) -> dict[str, Any]:
        handle = self._motion_handle(kind)
        job = _safe(lambda: handle.get(request_id), None)
        if job is None:
            raise AdminActionError("motion_request_not_found")
        return {"ok": True, "job": job.to_dict()}

    def motion_import_file(
        self,
        source_file: Path,
        motion_id: str,
        *,
        name: str = "",
        description: str = "",
        clip_name: Optional[str] = None,
        tags: tuple[str, ...] | list[str] = (),
        skill_category: str = "MOVEMENT",
    ) -> dict[str, Any]:
        if self.motion_import is None:
            raise AdminActionError("motion_import_unavailable")
        from ..skills.models import SkillCategory

        try:
            category = SkillCategory(str(skill_category).strip().upper())
        except ValueError as exc:
            raise AdminActionError("invalid_skill_category") from exc
        try:
            job = self.motion_import.request_import(
                source_file,
                motion_id,
                name=name or motion_id.replace("_", " ").title(),
                description=description or f"Admin import of {Path(source_file).name}",
                clip_name=clip_name or None,
                tags=list(tags),
                skill_category=category,
            )
        except Exception as exc:
            raise _rejected(
                exc,
                "motion_import_rejected",
                event="admin.motion.import_rejected",
                motion_id=motion_id,
            ) from exc
        log_event(log, "admin.motion.import_requested", motion_id=motion_id, state=job.state.value)
        return {"ok": True, "job": job.to_dict()}

    def motion_generate(
        self,
        prompt: str,
        *,
        duration_seconds: float = 2.5,
        motion_type: str = "general",
        style: str = "",
        speed: float = 1.0,
    ) -> dict[str, Any]:
        if self.motion_generation is None:
            raise AdminActionError("motion_generation_unavailable")
        try:
            job = self.motion_generation.request(
                prompt,
                duration_seconds=float(duration_seconds),
                motion_type=motion_type,
                style=style,
                speed=float(speed),
            )
        except Exception as exc:
            raise _rejected(
                exc, "motion_generation_rejected", event="admin.motion.generate_rejected"
            ) from exc
        log_event(log, "admin.motion.generate_requested", state=job.state.value)
        return {"ok": True, "job": job.to_dict()}

    def motion_process(
        self,
        source_motion_id: str,
        *,
        processed_motion_id: Optional[str] = None,
        processing_version: int = 1,
        name: str = "",
        description: str = "",
        skill_category: str = "MOVEMENT",
    ) -> dict[str, Any]:
        if self.motion_processing is None:
            raise AdminActionError("motion_processing_unavailable")
        from ..skills.models import SkillCategory

        try:
            category = SkillCategory(str(skill_category).strip().upper())
        except ValueError as exc:
            raise AdminActionError("invalid_skill_category") from exc
        try:
            job = self.motion_processing.request_processing(
                source_motion_id,
                processed_motion_id=processed_motion_id or None,
                processing_version=int(processing_version),
                name=name or f"{source_motion_id} processed",
                description=description or f"Admin retarget/cleanup of {source_motion_id}",
                skill_category=category,
            )
        except Exception as exc:
            raise _rejected(
                exc,
                "motion_processing_rejected",
                event="admin.motion.process_rejected",
                source=source_motion_id,
            ) from exc
        log_event(log, "admin.motion.process_requested", source=source_motion_id, state=job.state.value)
        return {"ok": True, "job": job.to_dict()}

    def motion_approve(self, request_id: str) -> dict[str, Any]:
        if self.motion_processing is None:
            raise AdminActionError("motion_processing_unavailable")
        try:
            job = self.motion_processing.approve(request_id)
        except KeyError as exc:
            raise AdminActionError("motion_request_not_found") from exc
        except Exception as exc:
            raise _rejected(
                exc,
                "motion_approve_rejected",
                event="admin.motion.approve_rejected",
                request_id=request_id,
            ) from exc
        log_event(log, "admin.motion.approved", request_id=request_id)
        return {"ok": True, "job": job.to_dict()}

    def motion_register(self, request_id: str) -> dict[str, Any]:
        if self.motion_processing is None:
            raise AdminActionError("motion_processing_unavailable")
        try:
            job = self.motion_processing.register(request_id)
        except KeyError as exc:
            raise AdminActionError("motion_request_not_found") from exc
        except Exception as exc:
            raise _rejected(
                exc,
                "motion_register_rejected",
                event="admin.motion.register_rejected",
                request_id=request_id,
            ) from exc
        payload = job.to_dict()
        ok = job.state.value == "REGISTERED"
        log_event(log, "admin.motion.registered", request_id=request_id, ok=ok)
        return {"ok": ok, "job": payload, "error": None if ok else job.error}

    def _motion_handle(self, kind: str) -> Any:
        handle = {
            "import": self.motion_import,
            "processing": self.motion_processing,
            "generation": self.motion_generation,
        }.get((kind or "").strip().lower())
        if handle is None:
            raise AdminActionError("motion_service_unavailable")
        return handle

    # ------------------------------------------------------- skills / learning

    def _knowledge_rows(
        self, learning: Any, skill_ids: "list[Any]"
    ) -> list[dict[str, Any]]:
        """Bounded ``SkillKnowledge`` rows through the public read API.

        ``LearningService.get_state`` reports counts, not rows; its supported
        per-skill read is ``get_skill_state``. Candidates stay bounded to the
        ids the caller already has (registered skills, learning motions), so
        the dashboard never scans or edits the learning database directly.
        """
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw in skill_ids:
            skill_id = str(raw or "").strip().lower()
            if not skill_id or skill_id in seen:
                continue
            seen.add(skill_id)
            state = _safe(
                lambda candidate=skill_id: learning.get_skill_state(candidate), {}
            )
            contexts = state.get("contexts") if isinstance(state, Mapping) else None
            for row in contexts or []:
                if isinstance(row, Mapping):
                    rows.append(dict(row))
        rows.sort(
            key=lambda row: (
                -float(row.get("average_score") or 0.0),
                str(row.get("skill_id")),
            )
        )
        return rows[:MAX_LEARNING_ROWS]

    def skills(self) -> dict[str, Any]:
        payload = _safe(self.service.get_skills, {"enabled": False, "skills": []})
        runtime = _safe(self.service.get_skills_runtime, {"enabled": False})
        learning = getattr(getattr(self.service, "brain", None), "learning", None)
        knowledge: list[dict[str, Any]] = []
        if learning is not None:
            knowledge = self._knowledge_rows(
                learning,
                [
                    skill.get("skill_id")
                    for skill in payload.get("skills") or []
                    if isinstance(skill, Mapping)
                ],
            )
        by_skill: dict[str, dict[str, Any]] = {}
        for row in knowledge:
            if isinstance(row, Mapping) and row.get("skill_id"):
                by_skill.setdefault(str(row["skill_id"]), row)
        skills = []
        for skill in payload.get("skills") or []:
            if not isinstance(skill, Mapping):
                continue
            item = dict(skill)
            item["learning"] = by_skill.get(str(skill.get("skill_id")))
            skills.append(item)
        return {
            "enabled": payload.get("enabled"),
            "skills": skills,
            "runtime": runtime,
            "count": len(skills),
        }

    def skill_set_enabled(self, skill_id: str, enabled: bool) -> dict[str, Any]:
        executor = getattr(getattr(self.service, "brain", None), "skill_executor", None)
        library = getattr(executor, "library", None)
        if library is None:
            raise AdminActionError("skills_unavailable")
        try:
            skill = library.enable(skill_id) if enabled else library.disable(skill_id)
        except KeyError as exc:
            raise AdminActionError("skill_not_found") from exc
        except Exception as exc:
            raise _rejected(
                exc,
                "skill_update_rejected",
                event="admin.skill.update_rejected",
                skill_id=skill_id,
            ) from exc
        log_event(log, "admin.skill.toggled", skill_id=skill_id, enabled=bool(enabled))
        return {"ok": True, "skill": skill.to_dict(include_steps=True)}

    # --------------------------------------------------------------- learning

    def learning(self) -> dict[str, Any]:
        learning = getattr(getattr(self.service, "brain", None), "learning", None)
        if learning is None:
            return {"enabled": False, "knowledge": [], "compositions": []}
        state = _safe(learning.get_state, {"enabled": False})
        compositions = _safe(learning.get_compositions, [])
        registered = _safe(self.service.get_skills, {"skills": []})
        candidates = [
            skill.get("skill_id")
            for skill in registered.get("skills") or []
            if isinstance(skill, Mapping)
        ]
        if isinstance(state, Mapping):
            candidates.extend(
                motion.get("motion_id")
                for motion in state.get("motions") or []
                if isinstance(motion, Mapping)
            )
        knowledge = self._knowledge_rows(learning, candidates)
        return {
            "enabled": state.get("enabled") if isinstance(state, Mapping) else None,
            "state": state,
            "knowledge": knowledge,
            "compositions": compositions,
            "motions": state.get("motions") if isinstance(state, Mapping) else None,
            "destructive_reset_available": True,
        }

    def learning_reset(self) -> dict[str, Any]:
        learning = getattr(getattr(self.service, "brain", None), "learning", None)
        if learning is None:
            raise AdminActionError("learning_unavailable")
        try:
            learning.reset()
        except Exception as exc:
            raise _rejected(exc, "learning_reset_failed", event="admin.learning.reset_failed") from exc
        instrument.count("admin.learning.reset")
        log_event(log, "admin.learning.reset")
        return {"ok": True, "state": _safe(learning.get_state, {})}

    # ----------------------------------------------------------------- memory

    def memory(self) -> dict[str, Any]:
        memory = getattr(self.service, "memory_service", None)
        if memory is None:
            return {"enabled": False}
        state = _safe(memory.get_state, {"enabled": False})
        repository = getattr(memory, "repository", None)
        episodes: list[Any] = []
        semantic: list[Any] = []
        interactions: list[Any] = []
        if repository is not None:
            raw_episodes = _safe(lambda: repository.search_episodes(limit=MAX_MEMORY_EPISODES), [])
            raw_semantic = _safe(
                lambda: repository.search_semantic_memory(limit=MAX_MEMORY_SEMANTIC), []
            )
            raw_interactions = _safe(lambda: repository.search_interactions(limit=MAX_MEMORY_EPISODES), [])
            episodes = [_project(item) for item in raw_episodes][:MAX_MEMORY_EPISODES]
            semantic = [_project(item) for item in raw_semantic][:MAX_MEMORY_SEMANTIC]
            interactions = [_project(item) for item in raw_interactions][:MAX_MEMORY_EPISODES]
        return {
            "enabled": state.get("enabled") if isinstance(state, Mapping) else None,
            "state": state,
            "episodic": episodes,
            "semantic": semantic,
            "interactions": interactions,
            "short_term": _safe(memory.short_term.snapshot, {}) if getattr(memory, "short_term", None) else {},
        }

    # ---------------------------------------------------------------- viewers

    def viewers(self, limit: int = 50) -> dict[str, Any]:
        bounded = max(1, min(int(limit), MAX_VIEWERS))
        payload = _safe(lambda: self.service.get_viewer_list(bounded), {"enabled": False, "viewers": []})
        viewers = payload.get("viewers") if isinstance(payload, Mapping) else []
        return {
            "enabled": payload.get("enabled") if isinstance(payload, Mapping) else None,
            "viewers": viewers or [],
            "count": payload.get("count") if isinstance(payload, Mapping) else 0,
            "limit": bounded,
        }

    # ----------------------------------------------------------- personality

    def personality(self) -> dict[str, Any]:
        personality = _safe(self.service.get_personality_state, {"enabled": False})
        behavior = _safe(self.service.get_behavior_state, {"enabled": False})
        personalization = _safe(self.service.get_personalization_state, {})
        goal = _safe(self.service.get_goal, {})
        return {
            "personality": personality,
            "behavior": behavior,
            "personalization": personalization,
            "goal": goal.get("goal") if isinstance(goal, Mapping) else None,
            "active_skill": _safe(self.service.brain.current_skill_id, None),
            "prompts_exposed": False,
        }

    # ------------------------------------------------------------------ logs

    def logs(self, query: Optional[LogQuery] = None) -> dict[str, Any]:
        return self.log_reader.tail(query or LogQuery())

    # -------------------------------------------------------------- settings

    def settings_view(self) -> dict[str, Any]:
        payload = self.settings.public()
        payload["applied"] = self._effective_view()
        return payload

    def save_settings(self, mapping: Mapping[str, Any]) -> dict[str, Any]:
        result = self.settings.save(mapping)
        for key, entry in result["results"].items():
            if entry.get("requires_restart"):
                continue
            applied, detail = self._apply_live(key, entry.get("value"))
            entry["applied"] = bool(applied)
            entry["status"] = "applied" if applied else "failed"
            if not applied:
                entry["error"] = detail
        log_event(
            log,
            "admin.settings.applied",
            keys=sorted(result["results"]),
            applied=sorted(
                key for key, entry in result["results"].items() if entry.get("applied")
            ),
        )
        return result

    def _apply_live(self, key: str, value: Any) -> tuple[bool, Optional[str]]:
        if key == "learning_enabled":
            learning = getattr(getattr(self.service, "brain", None), "learning", None)
            if learning is None:
                return False, "learning_unavailable"
            try:
                learning.set_enabled(bool(value))
            except Exception as exc:
                return False, safe_error(exc, limit=160)
            return True, None
        if key == "log_level":
            try:
                level = getattr(logging, str(value).upper())
                root = logging.getLogger()
                root.setLevel(level)
                logging.getLogger("room_agent").setLevel(level)
            except Exception as exc:
                return False, safe_error(exc, limit=160)
            return True, None
        return False, "setting_not_live"

    def _effective_view(self) -> dict[str, Any]:
        learning = getattr(getattr(self.service, "brain", None), "learning", None)
        return {
            "learning_enabled": bool(learning.enabled) if learning is not None else None,
            "log_level": logging.getLevelName(logging.getLogger().level),
        }

    # --------------------------------------------------------------- runtime

    def runtime_status(self) -> dict[str, Any]:
        controller = RoomAgentController(self.config)
        status = _safe(controller.status, {"error": "status_unavailable"})
        return {
            "runtime": status,
            "dispatched": self._dispatch_view(),
            "admin": {
                "pid": os.getpid(),
                "port": self.config.admin_port,
                "state": getattr(self.lifecycle, "state", None) or "RUNNING",
            },
        }

    def _dispatch_view(self) -> Optional[dict[str, Any]]:
        """Serialisable view of the last lifecycle dispatch; never a ``Popen``."""
        if not self._last_dispatch:
            return None
        process = self._last_dispatch.get("process")
        return {
            "action": self._last_dispatch.get("action"),
            "pid": self._last_dispatch.get("pid"),
            "log": self._last_dispatch.get("log"),
            "running": bool(process is not None and process.poll() is None),
        }

    def runtime_health(self) -> dict[str, Any]:
        controller = RoomAgentController(self.config)
        return {
            "health": _safe(controller.health, {"reachable": False}),
            "runtime": _safe(controller.status, {}),
        }

    def runtime_command(self, action: str) -> dict[str, Any]:
        """Dispatch the existing lifecycle CLI; never stop this process inline."""
        normalized = (action or "").strip().lower()
        if normalized not in RUNTIME_ACTIONS:
            raise AdminActionError("unknown_runtime_action")
        controller = RoomAgentController(self.config)
        current = _safe(controller.status, {})
        running = bool(current.get("running"))
        if normalized == "start" and running:
            raise AdminActionError("instance_already_running", detail={"pid": current.get("pid")})
        if normalized in {"stop", "restart"} and not running:
            return {
                "ok": True,
                "action": normalized,
                "requested": False,
                "already_stopped": True,
                "runtime": current,
            }
        if self.project_root is None:
            raise AdminActionError("runtime_dispatch_unavailable")
        with self._dispatch_lock:
            if (
                self._last_dispatch.get("process") is not None
                and self._last_dispatch["process"].poll() is None
            ):
                raise AdminActionError("runtime_command_in_progress")
            dispatch = self._dispatch_lifecycle(normalized)
            self._last_dispatch = dispatch
        return {
            "ok": True,
            "action": normalized,
            "requested": True,
            "dispatched_pid": dispatch["pid"],
            "log": dispatch["log"],
            "note": "confirmation arrives through /admin/api/runtime; poll until the state settles",
            "runtime": current,
        }

    def _dispatch_lifecycle(self, action: str) -> dict[str, Any]:
        log_path = Path(self.config.log_dir) / "admin_lifecycle.out.log"
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            sink = open(log_path, "a", encoding="utf-8")
        except OSError as exc:
            raise _rejected(
                exc, "runtime_dispatch_failed", event="admin.runtime.dispatch_failed", action=action
            ) from exc
        creationflags = 0
        if os.name == "nt":  # pragma: no cover - Windows only
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(
                subprocess, "CREATE_NO_WINDOW", 0
            )
        try:
            process = subprocess.Popen(
                [
                    self._python,
                    "-m",
                    "backend.runtime.lifecycle",
                    action,
                    "--wait",
                    "25",
                ],
                cwd=str(self.project_root),
                stdout=sink,
                stderr=sink,
                close_fds=True,
                creationflags=creationflags,
            )
        except Exception as exc:
            raise _rejected(
                exc, "runtime_dispatch_failed", event="admin.runtime.dispatch_failed", action=action
            ) from exc
        finally:
            sink.close()
        log_event(log, "admin.runtime.dispatched", action=action, pid=process.pid)
        return {"action": action, "pid": process.pid, "log": log_path.name, "process": process}


def _operator_motion(item: Mapping[str, Any]) -> dict[str, Any]:
    """The operator view of a MotionDefinition; no new fields are invented."""
    fps = item.get("sample_rate")
    return {
        "motion_id": item.get("motion_id"),
        "name": item.get("name"),
        "source": item.get("source"),
        "version": item.get("version"),
        "duration_seconds": item.get("duration_seconds"),
        "fps": fps,
        "frame_count": item.get("frame_count"),
        "quality_score": item.get("quality_score"),
        "quality_flags": item.get("quality_flags"),
        "skeleton_status": item.get("skeleton_status"),
        "validation_status": item.get("validation_status"),
        "animation_reference": item.get("animation_reference"),
        "clip_name": item.get("clip_name"),
        "enabled": item.get("enabled"),
        "execution_action": item.get("execution_action"),
        "processing_version": item.get("processing_version"),
        "cleanup_status": item.get("cleanup_status"),
        "source_motion_id": item.get("source_motion_id"),
        "processing_timestamp": item.get("processing_timestamp"),
        "last_error": item.get("last_error"),
        "tags": item.get("tags"),
    }


def _project(item: Any) -> Any:
    to_dict = getattr(item, "to_dict", None)
    if callable(to_dict):
        try:
            return to_dict()
        except Exception:  # pragma: no cover - defensive
            return {"error": "projection_failed"}
    if isinstance(item, Mapping):
        return dict(item)
    return item


__all__ = ["AdminActionError", "AdminConsole"]
