"""Bounded, single-concurrency text-to-motion generation service."""

from __future__ import annotations

import hashlib
import queue
import re
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Callable, Optional

from ..chat.models import utc_now
from .base import MotionGenerator
from .models import (
    PROVIDER_MOCK,
    MotionGenerationConfig,
    MotionGenerationError,
    MotionGenerationRequest,
    MotionGenerationStatus,
    MotionGenerationResult,
)
from .mock import MockMotionGenerator
from .motius import MotiusT2MGPTGenerator
from .validation import validate_motion_payload


_IN_FLIGHT = frozenset({MotionGenerationStatus.QUEUED, MotionGenerationStatus.GENERATING})
_UNSAFE_PROMPT = re.compile(
    r"(?i)\b("
    r"import\s+|subprocess|terminal|shell|Unity|Animator|GameObject|C#"
    r"|delete\s+file|write\s+file|open\s+file|file\s+operations|Unity\s+commands"
    r")\b"
)
_UNSAFE_CHARACTERS = re.compile(r"[{}<>]|&&|\|\|")


@dataclass
class MotionGenerationJob:
    request_id: str
    request: MotionGenerationRequest
    state: MotionGenerationStatus = MotionGenerationStatus.QUEUED
    result: Optional[MotionGenerationResult] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.min)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "prompt": self.request.prompt,
            "duration_seconds": self.request.duration_seconds,
            "state": self.state.value,
            "error": self.error,
            "result": self.result.to_dict() if self.result else None,
        }


class MotionGenerationService:
    def __init__(
        self,
        generator: MotionGenerator | None = None,
        config: MotionGenerationConfig | None = None,
        *,
        metadata_store=None,
        clock: Callable[[], datetime] = utc_now,
        start_worker: bool = True,
    ) -> None:
        self.config = config or MotionGenerationConfig()
        self.generator = generator if generator is not None else build_generator(self.config)
        self.metadata_store = metadata_store
        self._clock = clock
        self._lock = threading.RLock()
        self._queue: queue.Queue = queue.Queue(maxsize=self.config.max_queue_depth)
        self._jobs: "OrderedDict[str, MotionGenerationJob]" = OrderedDict()
        self._by_dedup_key: dict[str, str] = {}
        self._last_request_id: Optional[str] = None
        self._last_result_id: Optional[str] = None
        self._counters = {
            "requested": 0,
            "reused": 0,
            "rejected": 0,
            "generated": 0,
            "validated": 0,
            "failed": 0,
        }
        self._last_generation_ms: Optional[float] = None
        self._last_motion_length: Optional[int] = None
        self._stopping = threading.Event()
        self._worker: Optional[threading.Thread] = None
        if start_worker:
            self.start()

    def start(self) -> None:
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                return
            self._stopping.clear()
            self._worker = threading.Thread(
                target=self._worker_loop,
                name="motion-generation-worker",
                daemon=True,
            )
            self._worker.start()

    def shutdown(self, join_timeout: float = 1.0) -> None:
        with self._lock:
            worker = self._worker
            self._worker = None
        self._stopping.set()
        if worker is not None and worker.is_alive():
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                pass
            worker.join(timeout=join_timeout)

    def request(
        self,
        prompt: str,
        *,
        duration_seconds: float = 2.5,
        motion_type: str = "general",
        style: str = "",
        speed: float = 1.0,
        seed: Optional[int] = None,
        dedup_key: Optional[str] = None,
    ) -> MotionGenerationJob:
        """Queue a generation and return immediately; never blocks the caller."""
        with self._lock:
            body = (prompt or "").strip()
            if not body:
                return self._reject("empty_prompt", body, duration_seconds)
            if len(body) > self.config.max_prompt_length:
                return self._reject("prompt_too_long", body, duration_seconds)
            if not _is_safe_prompt(body):
                return self._reject("unsafe_prompt", body, duration_seconds)
            if duration_seconds < self.config.min_duration_seconds or duration_seconds > self.config.max_duration_seconds:
                return self._reject("duration_out_of_bounds", body, duration_seconds)
            if speed <= 0 or speed > 4.0:
                return self._reject("speed_out_of_bounds", body, duration_seconds)
            if len(motion_type) > 100 or len(style) > 200:
                return self._reject("invalid_request_metadata", body, duration_seconds)

            normalized_dedup = (
                dedup_key.strip()
                if isinstance(dedup_key, str) and dedup_key.strip()
                else _default_dedup_key(body, duration_seconds, seed, style)
            )
            existing = self._jobs.get(self._by_dedup_key.get(normalized_dedup, ""))
            if existing is not None and existing.state in _IN_FLIGHT:
                self._counters["reused"] += 1
                return existing

            request = MotionGenerationRequest(
                prompt=body,
                duration_seconds=duration_seconds,
                motion_type=motion_type,
                style=style,
                speed=speed,
                seed=seed,
                dedup_key=normalized_dedup,
            )
            request_id = _new_id()
            job = MotionGenerationJob(
                request_id=request_id,
                request=request,
                created_at=self._clock(),
            )
            self._jobs[request_id] = job
            self._by_dedup_key[normalized_dedup] = request_id
            self._last_request_id = request_id
            self._counters["requested"] += 1
            try:
                self._queue.put_nowait(request_id)
            except queue.Full:
                job.state = MotionGenerationStatus.FAILED
                job.error = "generation_queue_full"
                self._last_result_id = request_id
                self._counters["rejected"] += 1
                self._counters["failed"] += 1
            self._evict_jobs()
            return job

    def get(self, request_id: str) -> Optional[MotionGenerationJob]:
        with self._lock:
            return self._jobs.get(request_id)

    def state(self, request_id: str) -> Optional[MotionGenerationStatus]:
        with self._lock:
            job = self._jobs.get(request_id)
            return job.state if job else None

    def run_pending_once(self, timeout: float = 0.2) -> bool:
        """Process one queued item inline; used by deterministic tests."""
        return self._process_next(timeout)

    def pending_count(self) -> int:
        return self._queue.qsize()

    def get_state(self) -> dict[str, Any]:
        with self._lock:
            active_id = self._active_request_id()
            active = self._jobs.get(active_id or "")
            last_request = self._jobs.get(self._last_request_id or "")
            last_result = self._jobs.get(self._last_result_id or "")
            return {
                "generator": self.generator.name,
                "model": self.config.model,
                "checkpoint": self.config.checkpoint,
                "device": self.config.device,
                "loaded": bool(getattr(self.generator, "loaded", False)),
                "queue_depth": self._queue.qsize(),
                "queue_capacity": self._queue.maxsize,
                "max_concurrent_generations": self.config.max_concurrent_generations,
                "active_request_id": active.request_id if active else None,
                "state": active.state.value if active else "IDLE",
                "counters": dict(self._counters),
                "last_request": last_request.to_dict() if last_request else None,
                "last_result": last_result.to_dict() if last_result else None,
                "last_generation_ms": self._last_generation_ms,
                "last_motion_length": self._last_motion_length,
                "last_error": last_result.error if last_result else None,
            }

    def diagnostics(self) -> dict[str, Any]:
        return self.get_state()

    def health(self) -> dict[str, Any]:
        state = self.get_state()
        failures = state["counters"]["failed"]
        return {
            "status": "degraded" if failures else "ok",
            "generator": state["generator"],
            "model": state["model"],
            "queue_depth": state["queue_depth"],
            "queue_capacity": state["queue_capacity"],
            "last_error": state["last_error"],
        }

    def _worker_loop(self) -> None:
        while not self._stopping.is_set():
            if not self._process_next(self.config.worker_poll_interval_seconds):
                return

    def _process_next(self, timeout: float) -> bool:
        try:
            item = self._queue.get(timeout=timeout)
        except queue.Empty:
            return True
        if item is None:
            return False
        try:
            self._process(str(item))
        finally:
            self._queue.task_done()
        return True

    def _process(self, request_id: str) -> None:
        with self._lock:
            job = self._jobs.get(request_id)
            if job is None or job.state != MotionGenerationStatus.QUEUED:
                return
            job.state = MotionGenerationStatus.GENERATING
            started_at = self._clock()

        request = job.request
        try:
            generated = self.generator.generate(request)
            probe = validate_motion_payload(
                generated.motion,
                frame_rate=self.config.motion_rate,
                expected_duration_seconds=request.duration_seconds,
            )
            generation_ms = (self._clock() - started_at).total_seconds() * 1000.0
            with self._lock:
                job = self._jobs.get(request_id)
                if job is None or job.state != MotionGenerationStatus.GENERATING:
                    return
                motion_id = generated.motion_id or _new_id()
                result = replace(
                    generated,
                    motion_id=motion_id,
                    generated_at=self._clock(),
                    validated_at=self._clock(),
                    generation_ms=generation_ms,
                    metadata={
                        **generated.metadata,
                        **probe,
                        "generator": self.generator.name,
                        "model": self.config.model,
                        "device": self.config.device,
                    },
                    validation_status="VALID",
                )
                job.result = result
                job.state = MotionGenerationStatus.VALIDATED
                job.error = None
                self._last_result_id = request_id
                self._counters["generated"] += 1
                self._counters["validated"] += 1
                self._last_generation_ms = generation_ms
                self._last_motion_length = probe["frame_count"]
                if self.metadata_store is not None:
                    self.metadata_store.save(result)
                self._evict_jobs()
        except Exception as exc:
            self._fail(request_id, exc)

    def _fail(self, request_id: str, exc: BaseException) -> None:
        with self._lock:
            job = self._jobs.get(request_id)
            if job is None or job.state not in {MotionGenerationStatus.GENERATING, MotionGenerationStatus.QUEUED}:
                return
            job.state = MotionGenerationStatus.FAILED
            job.error = _safe_error(exc)
            self._last_result_id = request_id
            self._counters["failed"] += 1
            self._evict_jobs()

    def _reject(
        self,
        reason: str,
        prompt: str,
        duration_seconds: float,
    ) -> MotionGenerationJob:
        with self._lock:
            request = MotionGenerationRequest(
                prompt=(prompt or "").strip()[:200],
                duration_seconds=duration_seconds,
            )
            request_id = _new_id()
            job = MotionGenerationJob(
                request_id=request_id,
                request=request,
                state=MotionGenerationStatus.FAILED,
                error=reason,
                created_at=self._clock(),
            )
            self._jobs[request_id] = job
            self._last_request_id = request_id
            self._last_result_id = request_id
            self._counters["requested"] += 1
            self._counters["rejected"] += 1
            self._counters["failed"] += 1
            self._evict_jobs()
            return job

    def _evict_jobs(self) -> None:
        """Keep the in-memory job table bounded (oldest terminal job first)."""
        limit = self.config.max_retained_jobs
        if len(self._jobs) <= limit:
            return
        protected = {self._last_request_id, self._last_result_id}
        for request_id, job in list(self._jobs.items()):
            if len(self._jobs) <= limit:
                return
            if job.state in _IN_FLIGHT or request_id in protected:
                continue
            del self._jobs[request_id]
            dedup_key = getattr(job.request, "dedup_key", "")
            if dedup_key and self._by_dedup_key.get(dedup_key) == request_id:
                self._by_dedup_key.pop(dedup_key, None)

    def _active_request_id(self) -> Optional[str]:
        for request_id, job in reversed(list(self._jobs.items())):
            if job.state in _IN_FLIGHT:
                return request_id
        return None


def build_generator(config: MotionGenerationConfig) -> MotionGenerator:
    if config.provider == PROVIDER_MOCK:
        return MockMotionGenerator(config)
    if config.provider == "motius":
        return MotiusT2MGPTGenerator(config)
    raise ValueError("UNSUPPORTED_MOTION_GENERATOR")


def _is_safe_prompt(prompt: str) -> bool:
    return not _UNSAFE_PROMPT.search(prompt) and not _UNSAFE_CHARACTERS.search(prompt)


def _default_dedup_key(prompt: str, duration: float, seed: Optional[int], style: str) -> str:
    normalized = f"{prompt.casefold()}|{duration}|{seed}|{style}".encode("utf-8")
    digest = hashlib.sha256(normalized).hexdigest()[:24]
    return f"motion_{digest}"


def _safe_error(exc: BaseException) -> str:
    if isinstance(exc, MotionGenerationError):
        return exc.reason
    return type(exc).__name__


def _new_id() -> str:
    return f"motion_{uuid.uuid4().hex[:16]}"
