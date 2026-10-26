"""Bounded Phase 17 processing service for existing MotionDefinitions."""

from __future__ import annotations

import queue
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from ..chat.models import format_utc, utc_now
from ..learning.models import (
    MotionDefinition,
    MotionSource,
    MotionValidationStatus,
    SkeletonCompatibility,
)
from ..learning.service import SkillLearningService
from ..skills.models import SkillCategory
from .models import (
    MotionPlaybackReport,
    MotionProcessingConfig,
    MotionProcessingError,
    MotionProcessingReport,
    MotionProcessingRequest,
    MotionProcessingResult,
    MotionProcessingState,
)
from .persistence import SQLiteMotionProcessingMetadataStore
from .unity import MotionProcessingAdapter, UnityMotionProcessingAdapter
from .validation import validate_playback_report, validate_processing_report


_IN_FLIGHT = frozenset(
    {
        MotionProcessingState.INPUT,
        MotionProcessingState.ANALYZING,
        MotionProcessingState.RETARGETING,
        MotionProcessingState.CLEANING,
        MotionProcessingState.VALIDATING,
    }
)


@dataclass
class MotionProcessingJob:
    request_id: str
    request: MotionProcessingRequest
    state: MotionProcessingState = MotionProcessingState.INPUT
    result: Optional[MotionProcessingResult] = None
    error: Optional[str] = None
    created_at: datetime = datetime.min

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "state": self.state.value,
            "error": self.error,
            "request": self.request.to_dict(),
            "result": self.result.to_dict() if self.result else None,
        }


class MotionProcessingService:
    def __init__(
        self,
        learning_service: SkillLearningService,
        adapter: MotionProcessingAdapter | None = None,
        config: MotionProcessingConfig | None = None,
        metadata_store: SQLiteMotionProcessingMetadataStore | None = None,
        *,
        clock: Callable[[], datetime] = utc_now,
        start_worker: bool = True,
    ) -> None:
        self.learning_service = learning_service
        self.config = config or MotionProcessingConfig()
        self.adapter = adapter if adapter is not None else UnityMotionProcessingAdapter(self.config)
        self.metadata_store = metadata_store or SQLiteMotionProcessingMetadataStore(self.config.database_path)
        self._clock = clock
        self._lock = threading.RLock()
        self._queue: queue.Queue[str] = queue.Queue(maxsize=self.config.max_queue_depth)
        self._jobs: "OrderedDict[str, MotionProcessingJob]" = OrderedDict()
        self._last_request_id: Optional[str] = None
        self._last_result_id: Optional[str] = None
        self._counters = {"requested": 0, "ready": 0, "approved": 0, "registered": 0, "rejected": 0, "failed": 0}
        self._stopping = threading.Event()
        self._worker: Optional[threading.Thread] = None
        if start_worker:
            self.start()

    def request_processing(
        self,
        source_motion_id: str,
        *,
        processed_motion_id: str | None = None,
        processing_version: int = 1,
        name: str = "",
        description: str = "",
        tags: tuple[str, ...] | list[str] = (),
        skill_category: SkillCategory | str = SkillCategory.MOVEMENT,
    ) -> MotionProcessingJob:
        source_id = str(source_motion_id).strip().lower()
        source = self.learning_service.get_motion(source_id)
        if source is None:
            return self._invalid_request(source_id, "SOURCE_MOTION_NOT_FOUND", processing_version)
        if source.validation_status != MotionValidationStatus.VALID:
            return self._invalid_request(source_id, "SOURCE_MOTION_NOT_VALID", processing_version)
        output_id = str(processed_motion_id or f"{source_id}_processed").strip().lower()
        if not _is_valid_motion_id(output_id):
            return self._invalid_request(output_id, "INVALID_PROCESSED_MOTION_ID", processing_version)
        if processing_version <= 0:
            return self._invalid_request(output_id, "INVALID_PROCESSING_VERSION", processing_version)
        category = skill_category if isinstance(skill_category, SkillCategory) else SkillCategory(str(skill_category).upper())

        request = MotionProcessingRequest(
            source_motion_id=source.motion_id,
            source_version=source.version,
            processed_motion_id=output_id,
            processing_version=processing_version,
            name=name.strip() or f"{source.name} (Processed)",
            description=description.strip() or f"Quality-processed version of {source.motion_id} v{source.version}.",
            tags=tuple(dict.fromkeys((*source.tags, "processed", *(str(tag).casefold() for tag in tags)))),
            skill_category=category.value,
        )
        with self._lock:
            if (
                self.learning_service.get_motion(output_id) is not None
                or self.learning_service.library.exists(output_id)
                or self.metadata_store.has_output(output_id, processing_version)
            ):
                return self._invalid_request(output_id, "DUPLICATE_PROCESSED_MOTION", processing_version)
            request_id = f"motion_processing_{uuid.uuid4().hex[:16]}"
            job = MotionProcessingJob(request_id, request, created_at=self._clock())
            self._jobs[request_id] = job
            self._last_request_id = request_id
            self._counters["requested"] += 1
            self.metadata_store.save_request(
                request_id,
                request,
                MotionProcessingState.INPUT,
                None,
                format_utc(job.created_at),
            )
            try:
                self._queue.put_nowait(request_id)
            except queue.Full:
                job.state = MotionProcessingState.FAILED
                job.error = "PROCESSING_QUEUE_FULL"
                self._last_result_id = request_id
                self._counters["failed"] += 1
                self._save_failed(job)
            self._evict_jobs()
            return job

    def get(self, request_id: str) -> Optional[MotionProcessingJob]:
        with self._lock:
            return self._jobs.get(request_id)

    def get_state(self) -> dict[str, Any]:
        with self._lock:
            last_request = self._jobs.get(self._last_request_id or "")
            last_result = self._jobs.get(self._last_result_id or "")
            return {
                "queue_depth": self._queue.qsize(),
                "queue_capacity": self._queue.maxsize,
                "max_concurrent_processes": self.config.max_concurrent_processes,
                "active_request_id": self._active_request_id(),
                "state": self._active_state(),
                "counters": dict(self._counters),
                "records": self.metadata_store.recent(50),
                "last_request": last_request.to_dict() if last_request else None,
                "last_result": last_result.to_dict() if last_result else None,
                "last_error": last_result.error if last_result else None,
            }

    def health(self) -> dict[str, Any]:
        state = self.get_state()
        return {
            "status": "degraded" if state["counters"]["failed"] else "ok",
            "queue_depth": state["queue_depth"],
            "queue_capacity": state["queue_capacity"],
            "max_concurrent_processes": state["max_concurrent_processes"],
            "last_error": state["last_error"],
        }

    def pending_count(self) -> int:
        return self._queue.qsize()

    def run_pending_once(self, timeout: float = 0.2) -> bool:
        try:
            request_id = self._queue.get(timeout=timeout)
        except queue.Empty:
            return True
        if request_id is None:
            return False
        try:
            self._process(request_id)
        finally:
            self._queue.task_done()
        return True

    def start(self) -> None:
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                return
            self._stopping.clear()
            self._worker = threading.Thread(
                target=self._worker_loop,
                name="motion-processing-worker",
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

    def approve(self, request_id: str) -> MotionProcessingJob:
        with self._lock:
            job = self._jobs.get(request_id)
            if job is None:
                raise KeyError("processing_request_not_found")
            if job.state != MotionProcessingState.READY:
                raise MotionProcessingError("MOTION_NOT_READY", job.state.value)
            if job.result is None or not job.result.processing_report or not job.result.playback_report:
                raise MotionProcessingError("MOTION_VALIDATION_INCOMPLETE")
            if job.result.validation_status != MotionValidationStatus.VALID.value:
                raise MotionProcessingError("MOTION_NOT_VALID", job.result.validation_status)
            if job.result.skeleton_status != SkeletonCompatibility.COMPATIBLE.value:
                raise MotionProcessingError("MOTION_NOT_EXECUTABLE_WITHOUT_RETARGETING")
            job.state = MotionProcessingState.APPROVED
            job.result.state = MotionProcessingState.APPROVED
            self._counters["approved"] += 1
            self._save_result(job)
            return job

    def register(self, request_id: str) -> MotionProcessingJob:
        with self._lock:
            job = self._jobs.get(request_id)
            if job is None:
                raise KeyError("processing_request_not_found")
            if job.state != MotionProcessingState.APPROVED:
                raise MotionProcessingError("MOTION_NOT_APPROVED", job.state.value)
            assert job.result is not None
            result = job.result
            request = job.request
            source = self.learning_service.get_motion(request.source_motion_id)
            if source is None or source.version != request.source_version:
                raise MotionProcessingError("MOTION_LINEAGE_MISSING")

        try:
            motion = _build_motion(result, source)
            category = SkillCategory(request.skill_category)
            registered_motion, skill = self.learning_service.register_validated_motion(
                motion,
                skill_category=category,
            )
            with self._lock:
                job.state = MotionProcessingState.REGISTERED
                result.state = MotionProcessingState.REGISTERED
                result.motion = registered_motion.to_dict()
                result.skill = skill.to_dict(include_steps=True)
                result.registered_at = format_utc(self._clock())
                self._counters["registered"] += 1
                self._save_result(job)
                return job
        except Exception as exc:
            self._fail(request_id, exc)
            return self._jobs[request_id]

    def _worker_loop(self) -> None:
        while not self._stopping.is_set():
            if not self.run_pending_once(self.config.worker_poll_interval_seconds):
                return

    def _process(self, request_id: str) -> None:
        with self._lock:
            job = self._jobs.get(request_id)
            if job is None or job.state != MotionProcessingState.INPUT:
                return
            job.state = MotionProcessingState.ANALYZING
            self._save_job(job)
            request = job.request
            source = self.learning_service.get_motion(request.source_motion_id)
            if source is None:
                raise MotionProcessingError("SOURCE_MOTION_NOT_FOUND")

        try:
            processing = self.adapter.process(request, source)
            processing = validate_processing_report(
                processing,
                request,
                source,
                unity_project_path=self.config.unity_project_path,
            )
            with self._lock:
                job = self._jobs[request_id]
                job.state = MotionProcessingState.VALIDATING
                partial = self._build_partial_result(job, processing=processing)
                job.result = partial
                self._save_result(job)

            playback = self.adapter.validate_playback(request, source, processing)
            playback = validate_playback_report(playback, request, processing)
            result = self._build_partial_result(
                job,
                processing=processing,
                playback=playback,
                state=MotionProcessingState.READY,
                error=None,
            )
            with self._lock:
                current = self._jobs.get(request_id)
                if current is None or current.state not in _IN_FLIGHT:
                    return
                current.state = MotionProcessingState.READY
                current.result = result
                current.error = None
                self._last_result_id = request_id
                self._counters["ready"] += 1
                self._save_result(current)
                self._evict_jobs()
        except Exception as exc:
            self._fail(request_id, exc)

    def _build_partial_result(
        self,
        job: MotionProcessingJob,
        *,
        processing: MotionProcessingReport,
        playback: MotionPlaybackReport | None = None,
        state: MotionProcessingState = MotionProcessingState.VALIDATING,
        error: Optional[str] = None,
    ) -> MotionProcessingResult:
        return MotionProcessingResult(
            request_id=job.request_id,
            request=job.request,
            state=state,
            quality_status=processing.quality_status,
            quality_score=processing.quality_score,
            skeleton_status=processing.skeleton_status,
            validation_status=(
                MotionValidationStatus.VALID.value
                if state == MotionProcessingState.READY and playback is not None
                else MotionValidationStatus.PENDING.value
            ),
            retarget_status=processing.retarget_status,
            cleanup_status=processing.cleanup_status,
            cleanup_magnitude=processing.cleanup_magnitude,
            duration_seconds=processing.duration_seconds,
            fps=processing.fps,
            frame_count=processing.frame_count,
            animation_reference=processing.animation_reference,
            processed_clip_path=processing.processed_clip_path,
            error=error,
            processing_report=processing.to_dict(),
            playback_report=playback.to_dict() if playback else {},
            processing_timestamp=processing.processing_timestamp,
        )

    def _invalid_request(
        self,
        motion_id: str,
        reason: str,
        processing_version: int,
    ) -> MotionProcessingJob:
        request = MotionProcessingRequest(
            source_motion_id=motion_id,
            source_version=0,
            processed_motion_id=motion_id,
            processing_version=max(0, processing_version),
            name="Rejected motion processing request",
            description="Rejected before processing.",
        )
        request_id = f"motion_processing_{uuid.uuid4().hex[:16]}"
        job = MotionProcessingJob(
            request_id,
            request,
            state=MotionProcessingState.REJECTED,
            error=reason,
            created_at=self._clock(),
        )
        with self._lock:
            self._jobs[request_id] = job
            self._last_request_id = request_id
            self._last_result_id = request_id
            self._counters["requested"] += 1
            self._counters["rejected"] += 1
            self._save_failed(job)
            self._evict_jobs()
        return job

    def _fail(self, request_id: str, exc: BaseException) -> None:
        with self._lock:
            job = self._jobs.get(request_id)
            if job is None or job.state not in _IN_FLIGHT | {MotionProcessingState.APPROVED}:
                return
            job.state = MotionProcessingState.FAILED
            job.error = _safe_error(exc)
            self._last_result_id = request_id
            self._counters["failed"] += 1
            self._save_failed(job)
            self._evict_jobs()

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

    def _save_job(self, job: MotionProcessingJob) -> None:
        if job.result is None:
            self.metadata_store.save_request(
                job.request_id,
                job.request,
                job.state,
                job.error,
                format_utc(self._clock()),
            )
        else:
            self._save_result(job)

    def _save_result(self, job: MotionProcessingJob) -> None:
        if job.result is not None:
            job.result.state = job.state
            self.metadata_store.save_result(job.result, format_utc(self._clock()))
        else:
            self._save_failed(job)

    def _save_failed(self, job: MotionProcessingJob) -> None:
        if job.result is None:
            result = MotionProcessingResult(
                request_id=job.request_id,
                request=job.request,
                state=job.state,
                error=job.error,
                processing_timestamp=format_utc(job.created_at),
            )
        else:
            result = job.result
            result.state = job.state
            result.error = job.error
        job.result = result
        self.metadata_store.save_result(result, format_utc(self._clock()))

    def _active_request_id(self) -> Optional[str]:
        for request_id, job in reversed(list(self._jobs.items())):
            if job.state in _IN_FLIGHT:
                return request_id
        return None

    def _active_state(self) -> str:
        request_id = self._active_request_id()
        if request_id is None:
            return "IDLE"
        return self._jobs[request_id].state.value


def _build_motion(result: MotionProcessingResult, source: MotionDefinition) -> MotionDefinition:
    report = MotionProcessingReport.from_dict(result.processing_report)
    return MotionDefinition(
        motion_id=result.request.processed_motion_id,
        name=result.request.name,
        description=result.request.description,
        animation_reference=result.animation_reference,
        duration_seconds=result.duration_seconds,
        tags=result.request.tags,
        source=source.source,
        parameters=source.parameters,
        execution_action=source.execution_action,
        enabled=True,
        version=result.request.processing_version,
        source_path=result.processed_clip_path,
        source_format="ANIM",
        clip_name=result.request.processed_motion_id,
        frame_count=result.frame_count,
        sample_rate=result.fps,
        bone_count=report.bone_count,
        root_bone_name=report.root_bone_name,
        skeleton_status=SkeletonCompatibility.COMPATIBLE,
        validation_status=MotionValidationStatus.VALID,
        source_motion_id=result.request.source_motion_id,
        source_version=result.request.source_version,
        processing_version=result.request.processing_version,
        quality_score=result.quality_score,
        quality_flags=report.quality_flags,
        cleanup_status=report.cleanup_status,
        processing_timestamp=report.processing_timestamp,
        last_error=None,
    )


def _is_valid_motion_id(motion_id: str) -> bool:
    return bool(motion_id) and len(motion_id) <= 120 and all(
        character == "_" or character.isdigit() or character.islower()
        for character in motion_id
    )


def _safe_error(exc: BaseException) -> str:
    if isinstance(exc, MotionProcessingError):
        return exc.reason
    detail = str(exc).strip()
    return f"{type(exc).__name__}:{detail}" if detail else type(exc).__name__
