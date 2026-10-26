"""Metadata-only SQLite persistence for motion processing."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Optional

from .models import MotionProcessingRequest, MotionProcessingResult, MotionProcessingState


class SQLiteMotionProcessingMetadataStore:
    def __init__(self, database_path: Path | str) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            str(self.database_path), check_same_thread=False, timeout=5.0
        )
        self._connection.row_factory = sqlite3.Row
        self.initialize()

    def initialize(self) -> None:
        with self._lock, self._connection:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute(
                "CREATE TABLE IF NOT EXISTS motion_processing_records ("
                "request_id TEXT PRIMARY KEY, source_motion_id TEXT NOT NULL, source_version INTEGER NOT NULL,"
                "processed_motion_id TEXT NOT NULL, processing_version INTEGER NOT NULL,"
                "state TEXT NOT NULL, quality_status TEXT, quality_score REAL, skeleton_status TEXT,"
                "validation_status TEXT, retarget_status TEXT, cleanup_status TEXT, cleanup_magnitude REAL,"
                "duration_seconds REAL, fps REAL, frame_count INTEGER, animation_reference TEXT,"
                "processed_clip_path TEXT, error TEXT, metadata TEXT NOT NULL,"
                "created_at TEXT NOT NULL, updated_at TEXT NOT NULL)"
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_motion_processing_output "
                "ON motion_processing_records(processed_motion_id, processing_version)"
            )

    def save_request(
        self,
        request_id: str,
        request: MotionProcessingRequest,
        state: MotionProcessingState,
        error: Optional[str],
        timestamp: str,
    ) -> None:
        self._save(
            request_id,
            request,
            state,
            error=error,
            metadata={},
            created_at=timestamp,
            updated_at=timestamp,
        )

    def save_result(
        self,
        result: MotionProcessingResult,
        timestamp: str,
    ) -> None:
        self._save(
            result.request_id,
            result.request,
            result.state,
            quality_status=result.quality_status,
            quality_score=result.quality_score,
            skeleton_status=result.skeleton_status,
            validation_status=result.validation_status,
            retarget_status=result.retarget_status,
            cleanup_status=result.cleanup_status,
            cleanup_magnitude=result.cleanup_magnitude,
            duration_seconds=result.duration_seconds,
            fps=result.fps,
            frame_count=result.frame_count,
            animation_reference=result.animation_reference,
            processed_clip_path=result.processed_clip_path,
            error=result.error,
            metadata={
                "processing_report": result.processing_report,
                "playback_report": result.playback_report,
                "motion": result.motion,
                "skill": result.skill,
                "processing_timestamp": result.processing_timestamp,
                "registered_at": result.registered_at,
            },
            created_at=timestamp,
            updated_at=timestamp,
        )

    def get(self, request_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM motion_processing_records WHERE request_id = ?", (request_id,)
            ).fetchone()
        return _row_to_dict(row) if row is not None else None

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM motion_processing_records ORDER BY created_at, request_id"
            ).fetchall()
        return [_row_to_dict(row) for row in rows]

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        """Newest records only; used by diagnostics so status stays bounded."""
        bounded = max(1, min(int(limit), 500))
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM motion_processing_records "
                "ORDER BY created_at DESC, request_id DESC LIMIT ?",
                (bounded,),
            ).fetchall()
        return [_row_to_dict(row) for row in rows]

    def has_output(self, processed_motion_id: str, processing_version: int) -> bool:
        with self._lock:
            row = self._connection.execute(
                "SELECT 1 FROM motion_processing_records "
                "WHERE processed_motion_id = ? AND processing_version = ? LIMIT 1",
                (processed_motion_id, processing_version),
            ).fetchone()
        return row is not None

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def _save(
        self,
        request_id: str,
        request: MotionProcessingRequest,
        state: MotionProcessingState,
        *,
        quality_status: Optional[str] = None,
        quality_score: Optional[float] = None,
        skeleton_status: Optional[str] = None,
        validation_status: Optional[str] = None,
        retarget_status: Optional[str] = None,
        cleanup_status: Optional[str] = None,
        cleanup_magnitude: Optional[float] = None,
        duration_seconds: Optional[float] = None,
        fps: Optional[float] = None,
        frame_count: Optional[int] = None,
        animation_reference: Optional[str] = None,
        processed_clip_path: Optional[str] = None,
        error: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        created_at: str = "",
        updated_at: str = "",
    ) -> None:
        with self._lock, self._connection:
            existing = self._connection.execute(
                "SELECT created_at FROM motion_processing_records WHERE request_id = ?",
                (request_id,),
            ).fetchone()
            original_created_at = existing["created_at"] if existing else created_at
            self._connection.execute(
                "INSERT OR REPLACE INTO motion_processing_records "
                "(request_id, source_motion_id, source_version, processed_motion_id, processing_version,"
                " state, quality_status, quality_score, skeleton_status, validation_status,"
                " retarget_status, cleanup_status, cleanup_magnitude, duration_seconds, fps,"
                " frame_count, animation_reference, processed_clip_path, error, metadata, created_at, updated_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    request_id,
                    request.source_motion_id,
                    request.source_version,
                    request.processed_motion_id,
                    request.processing_version,
                    state.value,
                    quality_status,
                    quality_score,
                    skeleton_status,
                    validation_status,
                    retarget_status,
                    cleanup_status,
                    cleanup_magnitude,
                    duration_seconds,
                    fps,
                    frame_count,
                    animation_reference,
                    processed_clip_path,
                    error,
                    json.dumps(metadata or {}),
                    original_created_at or created_at,
                    updated_at or created_at,
                ),
            )


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    result = {name: row[name] for name in row.keys()}
    try:
        result["metadata"] = json.loads(result.get("metadata") or "{}")
    except (TypeError, json.JSONDecodeError):
        result["metadata"] = {}
    return result


__all__ = ["SQLiteMotionProcessingMetadataStore"]
