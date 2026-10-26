"""Bounded, single-concurrency motion-file import service."""

from __future__ import annotations

import queue
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from ..chat.models import format_utc, utc_now
from ..learning.models import MotionSource, MotionValidationStatus, SkeletonCompatibility
from ..learning.motion_validation import validate_motion_source
from ..learning.service import SkillLearningService
from ..skills.models import SkillCategory
from .models import (
    MotionImportConfig,
    MotionImportError,
    MotionImportRequest,
    MotionImportResult,
    MotionImportStatus,
    UnityMotionImportReport,
)
from .persistence import SQLiteMotionImportMetadataStore
from .unity import MotionImportAdapter, UnityMotionImportAdapter
from .validation import (
    hash_source_file,
    stricter_skeleton,
    validate_import_report,
    validate_source_file,
)


_IN_FLIGHT = frozenset({MotionImportStatus.QUEUED, MotionImportStatus.IMPORTING})


@dataclass
class MotionImportJob:
    request_id: str
    request: MotionImportRequest
    skill_category: SkillCategory
    state: MotionImportStatus = MotionImportStatus.QUEUED
    result: Optional[MotionImportResult] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.min)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "state": self.state.value,
            "error": self.error,
            "request": self.request.to_dict(),
            "skill_category": self.skill_category.value,
            "result": self.result.to_dict() if self.result else None,
        }


class MotionFileImportService:
    def __init__(
        self,
        learning_service: SkillLearningService,
        adapter: MotionImportAdapter | None = None,
        config: MotionImportConfig | None = None,
        metadata_store: SQLiteMotionImportMetadataStore | None = None,
        *,
        clock: Callable[[], datetime] = utc_now,
        start_worker: bool = True,
    ) -> None:
        self.learning_service = learning_service
        self.config = config or MotionImportConfig()
        self.adapter = adapter if adapter is not None else UnityMotionImportAdapter(self.config)
        self.metadata_store = metadata_store or SQLiteMotionImportMetadataStore(self.config.database_path)
        self._clock = clock
        self._lock = threading.RLock()
        self._queue: queue.Queue[str] = queue.Queue(maxsize=self.config.max_queue_depth)
        self._jobs: "OrderedDict[str, MotionImportJob]" = OrderedDict()
        self._last_request_id: Optional[str] = None
        self._last_result_id: Optional[str] = None
        self._counters = {
            "requested": 0,
            "validated": 0,
            "registered": 0,
            "rejected": 0,
            "failed": 0,
        }
        self._stopping = threading.Event()
        self._worker: Optional[threading.Thread] = None
        if start_worker:
            self.start()

    def request_import(
        self,
        source_file: str | Path,
        motion_id: str,
        *,
        name: str = "",
        description: str = "",
        clip_name: str | None = None,
        tags: tuple[str, ...] | list[str] = (),
        skill_category: SkillCategory = SkillCategory.MOVEMENT,
    ) -> MotionImportJob:
        """Queue one import and return immediately."""
        requested_motion_id = str(motion_id).strip().lower()
        if requested_motion_id != requested_motion_id.replace(" ", "_"):
            requested_motion_id = requested_motion_id.replace(" ", "_")
        try:
            if not requested_motion_id or any(
                character not in "abcdefghijklmnopqrstuvwxyz0123456789_"
                for character in requested_motion_id
            ):
                raise MotionImportError("INVALID_MOTION_ID")
            source, source_format, source_hash = validate_source_file(
                source_file, self.config.supported_formats
            )
            if clip_name is not None and not clip_name.strip():
                raise MotionImportError("INVALID_CLIP_NAME")
        except MotionImportError as exc:
            return self._rejected(exc, source_file, requested_motion_id)

        with self._lock:
            version = self._next_version(requested_motion_id)
            actual_motion_id = (
                requested_motion_id
                if version == 1
                else f"{requested_motion_id}_v{version}"
            )
            request = MotionImportRequest(
                source_file=source,
                source_sha256=source_hash,
                source_format=source_format,
                motion_id=requested_motion_id,
                actual_motion_id=actual_motion_id,
                version=version,
                name=name.strip() or requested_motion_id.replace("_", " ").title(),
                description=description.strip()
                or f"Imported {source_format} motion skill {actual_motion_id}.",
                clip_name=clip_name.strip() if clip_name else None,
                tags=tuple(dict.fromkeys(str(tag).casefold() for tag in tags)),
            )
            request_id = _new_id()
            job = MotionImportJob(
                request_id,
                request,
                skill_category,
                created_at=self._clock(),
            )
            self._jobs[request_id] = job
            self._last_request_id = request_id
            self._counters["requested"] += 1
            self.metadata_store.save_request(
                request_id,
                request,
                MotionImportStatus.QUEUED,
                None,
                format_utc(job.created_at),
            )
            try:
                self._queue.put_nowait(request_id)
            except queue.Full:
                job.state = MotionImportStatus.FAILED
                job.error = "IMPORT_QUEUE_FULL"
                self._last_result_id = request_id
                self._counters["rejected"] += 1
                self._counters["failed"] += 1
                self._save_failed(job)
            self._evict_jobs()
            return job

    def get(self, request_id: str) -> Optional[MotionImportJob]:
        with self._lock:
            return self._jobs.get(request_id)

    def get_state(self) -> dict[str, Any]:
        with self._lock:
            last_request = self._jobs.get(self._last_request_id or "")
            last_result = self._jobs.get(self._last_result_id or "")
            return {
                "supported_formats": sorted(self.config.supported_formats),
                "queue_depth": self._queue.qsize(),
                "queue_capacity": self._queue.maxsize,
                "max_concurrent_imports": self.config.max_concurrent_imports,
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
            "supported_formats": state["supported_formats"],
            "queue_depth": state["queue_depth"],
            "queue_capacity": state["queue_capacity"],
            "max_concurrent_imports": state["max_concurrent_imports"],
            "last_error": state["last_error"],
        }

    def pending_count(self) -> int:
        return self._queue.qsize()

    def run_pending_once(self, timeout: float = 0.2) -> bool:
        return self._process_next(timeout)

    def start(self) -> None:
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                return
            self._stopping.clear()
            self._worker = threading.Thread(
                target=self._worker_loop,
                name="motion-file-import-worker",
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

    def _worker_loop(self) -> None:
        while not self._stopping.is_set():
            if not self._process_next(self.config.worker_poll_interval_seconds):
                return

    def _process_next(self, timeout: float) -> bool:
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

    def _process(self, request_id: str) -> None:
        with self._lock:
            job = self._jobs.get(request_id)
            if job is None or job.state != MotionImportStatus.QUEUED:
                return
            job.state = MotionImportStatus.IMPORTING
            self._save_job(job)
            request = job.request
            skill_category = job.skill_category

        try:
            source, _, source_hash = validate_source_file(
                request.source_file, self.config.supported_formats
            )
            if source.resolve() != request.source_file.resolve() or source_hash != request.source_sha256:
                raise MotionImportError("SOURCE_HASH_MISMATCH")
            report = self.adapter.import_motion(request)
            status = validate_import_report(
                report,
                request,
                unity_project_path=self.config.unity_project_path,
            )
            metadata = validate_motion_source(
                report.staged_source_file,
                report.animation_reference,
                report.duration_seconds,
                source_format=report.source_format,
                clip_name=report.clip_name,
                sample_rate=report.fps,
            )
            status = stricter_skeleton(status, metadata.skeleton_status)
            result = self._build_result(
                request_id,
                request,
                report,
                status.value,
                MotionImportStatus.VALIDATED,
                metadata.to_dict(),
            )
            if status == SkeletonCompatibility.COMPATIBLE:
                motion, skill = self.learning_service.register_motion(
                    request.actual_motion_id,
                    request.name,
                    request.description,
                    report.animation_reference,
                    report.duration_seconds,
                    tags=request.tags,
                    source=MotionSource.IMPORTED,
                    source_path=report.staged_source_file,
                    source_format=report.source_format,
                    clip_name=report.clip_name,
                    sample_rate=report.fps,
                    skill_category=skill_category,
                    version=request.version,
                )
                result = MotionImportResult(
                    **{
                        **result.to_dict(),
                        "import_status": MotionImportStatus.REGISTERED,
                        "motion": motion.to_dict(),
                        "skill": skill.to_dict(include_steps=True),
                        "registered_at": format_utc(self._clock()),
                    }
                )
                self._counters["registered"] += 1
            self._counters["validated"] += 1
            with self._lock:
                current = self._jobs.get(request_id)
                if current is None or current.state != MotionImportStatus.IMPORTING:
                    return
                current.state = result.import_status
                current.result = result
                current.error = None
                self._last_result_id = request_id
                self.metadata_store.save_result(result, format_utc(self._clock()))
                self._evict_jobs()
        except Exception as exc:
            self._fail(request_id, exc)

    def _build_result(
        self,
        request_id: str,
        request: MotionImportRequest,
        report: UnityMotionImportReport,
        skeleton_status: str,
        status: MotionImportStatus,
        metadata: dict[str, Any],
    ) -> MotionImportResult:
        return MotionImportResult(
            request_id=request_id,
            requested_base_motion_id=request.motion_id,
            actual_motion_id=request.actual_motion_id,
            version=request.version,
            original_source_file=str(request.source_file),
            staged_source_file=report.staged_source_file,
            source_sha256=request.source_sha256,
            source_format=report.source_format,
            import_status=status,
            validation_status=MotionValidationStatus.VALID.value,
            skeleton_status=skeleton_status,
            duration_seconds=report.duration_seconds,
            fps=report.fps,
            frame_count=report.frame_count,
            bone_count=report.bone_count,
            root_bone_name=report.root_bone_name,
            animation_reference=report.animation_reference,
            stable_clip_path=report.stable_clip_path,
            metadata={
                **report.metadata,
                **metadata,
                "source_avatar_valid": report.source_avatar_valid,
                "source_avatar_human": report.source_avatar_human,
                "source_humanoid_mappings": list(report.source_humanoid_mappings),
                "target_avatar_name": report.target_avatar_name,
                "target_avatar_valid": report.target_avatar_valid,
                "target_avatar_human": report.target_avatar_human,
                "target_humanoid_mappings": list(report.target_humanoid_mappings),
                "curve_count": report.curve_count,
                "key_count": report.key_count,
                "finite_values": report.finite_values,
            },
            imported_at=report.imported_at,
        )

    def _next_version(self, requested_motion_id: str) -> int:
        version = self.metadata_store.next_version(requested_motion_id)
        while True:
            actual = requested_motion_id if version == 1 else f"{requested_motion_id}_v{version}"
            if (
                self.learning_service.get_motion(actual) is None
                and not self.learning_service.library.exists(actual)
            ):
                return version
            version += 1

    def _rejected(
        self,
        error: MotionImportError,
        source_file: str | Path,
        motion_id: str,
    ) -> MotionImportJob:
        source = Path(source_file)
        request = MotionImportRequest(
            source_file=source,
            source_sha256="",
            source_format=source.suffix.removeprefix(".").upper(),
            motion_id=motion_id,
            actual_motion_id=motion_id,
            version=0,
            name=motion_id.replace("_", " ").title(),
            description="Rejected motion-file import request.",
        )
        request_id = _new_id()
        job = MotionImportJob(
            request_id=request_id,
            request=request,
            skill_category=SkillCategory.MOVEMENT,
            state=MotionImportStatus.FAILED,
            error=error.reason,
            created_at=self._clock(),
        )
        with self._lock:
            self._jobs[request_id] = job
            self._last_request_id = request_id
            self._last_result_id = request_id
            self._counters["requested"] += 1
            self._counters["rejected"] += 1
            self._counters["failed"] += 1
            self._save_job(job)
            self._evict_jobs()
        return job

    def _fail(self, request_id: str, exc: BaseException) -> None:
        with self._lock:
            job = self._jobs.get(request_id)
            if job is None or job.state not in _IN_FLIGHT:
                return
            job.state = MotionImportStatus.FAILED
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

    def _save_job(self, job: MotionImportJob) -> None:
        if job.state == MotionImportStatus.QUEUED or job.state == MotionImportStatus.IMPORTING:
            self.metadata_store.save_request(
                job.request_id,
                job.request,
                job.state,
                job.error,
                format_utc(self._clock()),
            )
        elif job.result is not None:
            self.metadata_store.save_result(job.result, format_utc(self._clock()))
        else:
            self._save_failed(job)

    def _save_failed(self, job: MotionImportJob) -> None:
        result = MotionImportResult(
            request_id=job.request_id,
            requested_base_motion_id=job.request.motion_id,
            actual_motion_id=job.request.actual_motion_id,
            version=job.request.version,
            original_source_file=str(job.request.source_file),
            staged_source_file="",
            source_sha256=job.request.source_sha256,
            source_format=job.request.source_format,
            import_status=MotionImportStatus.FAILED,
            validation_status=MotionValidationStatus.INVALID.value,
            skeleton_status=SkeletonCompatibility.UNKNOWN.value,
            duration_seconds=0.0,
            fps=0.0,
            frame_count=0,
            bone_count=0,
            root_bone_name="",
            animation_reference="",
            stable_clip_path="",
            error=job.error,
            imported_at=format_utc(job.created_at),
        )
        job.result = result
        self.metadata_store.save_result(result, format_utc(self._clock()))

    def _active_request_id(self) -> Optional[str]:
        for request_id, job in reversed(list(self._jobs.items())):
            if job.state in _IN_FLIGHT:
                return request_id
        return None

    def _active_state(self) -> str:
        active_id = self._active_request_id()
        if active_id is None:
            return "IDLE"
        return self._jobs[active_id].state.value


def _safe_error(exc: BaseException) -> str:
    if isinstance(exc, MotionImportError):
        return exc.reason
    return type(exc).__name__


def _new_id() -> str:
    return f"motion_import_{uuid.uuid4().hex[:16]}"
