"""Metadata-only SQLite persistence for motion-file imports."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Optional

from ..chat.models import format_utc
from .models import MotionImportRequest, MotionImportResult, MotionImportStatus


class SQLiteMotionImportMetadataStore:
    """Persist import provenance without storing motion binaries."""

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
                "CREATE TABLE IF NOT EXISTS motion_import_records ("
                "request_id TEXT PRIMARY KEY, requested_base_motion_id TEXT NOT NULL,"
                "actual_versioned_motion_id TEXT NOT NULL, original_source_file TEXT NOT NULL,"
                "staged_source_file TEXT NOT NULL, source_sha256 TEXT NOT NULL,"
                "source_format TEXT NOT NULL, version INTEGER NOT NULL,"
                "duration_seconds REAL, fps REAL, frame_count INTEGER, bone_count INTEGER,"
                "root_bone_name TEXT, skeleton_status TEXT, validation_status TEXT,"
                "animation_reference TEXT, stable_clip_path TEXT, import_status TEXT NOT NULL,"
                "error TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)"
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_motion_import_base_id "
                "ON motion_import_records(requested_base_motion_id, version)"
            )

    def save_request(
        self,
        request_id: str,
        request: MotionImportRequest,
        status: MotionImportStatus,
        error: Optional[str],
        timestamp: str,
    ) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT OR REPLACE INTO motion_import_records "
                "(request_id, requested_base_motion_id, actual_versioned_motion_id,"
                " original_source_file, staged_source_file, source_sha256, source_format, version,"
                " duration_seconds, fps, frame_count, bone_count, root_bone_name, skeleton_status,"
                " validation_status, animation_reference, stable_clip_path, import_status, error,"
                " created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    request_id,
                    request.motion_id,
                    request.actual_motion_id,
                    str(request.source_file),
                    "",
                    request.source_sha256,
                    request.source_format,
                    request.version,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    status.value,
                    error,
                    timestamp,
                    timestamp,
                ),
            )

    def save_result(
        self,
        result: MotionImportResult,
        timestamp: str,
    ) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT OR REPLACE INTO motion_import_records "
                "(request_id, requested_base_motion_id, actual_versioned_motion_id,"
                " original_source_file, staged_source_file, source_sha256, source_format, version,"
                " duration_seconds, fps, frame_count, bone_count, root_bone_name, skeleton_status,"
                " validation_status, animation_reference, stable_clip_path, import_status, error,"
                " created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    result.request_id,
                    result.requested_base_motion_id,
                    result.actual_motion_id,
                    result.original_source_file,
                    result.staged_source_file,
                    result.source_sha256,
                    result.source_format,
                    result.version,
                    result.duration_seconds,
                    result.fps,
                    result.frame_count,
                    result.bone_count,
                    result.root_bone_name,
                    result.skeleton_status,
                    result.validation_status,
                    result.animation_reference,
                    result.stable_clip_path,
                    result.import_status.value,
                    result.error,
                    result.imported_at or timestamp,
                    result.registered_at or timestamp,
                ),
            )

    def get(self, request_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM motion_import_records WHERE request_id = ?", (request_id,)
            ).fetchone()
        return _row_to_dict(row) if row is not None else None

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM motion_import_records ORDER BY created_at, request_id"
            ).fetchall()
        return [_row_to_dict(row) for row in rows]

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        """Newest records only; used by diagnostics so status stays bounded."""
        bounded = max(1, min(int(limit), 500))
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM motion_import_records "
                "ORDER BY created_at DESC, request_id DESC LIMIT ?",
                (bounded,),
            ).fetchall()
        return [_row_to_dict(row) for row in rows]

    def next_version(self, requested_base_motion_id: str) -> int:
        with self._lock:
            row = self._connection.execute(
                "SELECT MAX(version) AS version FROM motion_import_records "
                "WHERE requested_base_motion_id = ?",
                (requested_base_motion_id,),
            ).fetchone()
        version = row["version"] if row is not None else None
        return int(version) + 1 if version is not None else 1

    def close(self) -> None:
        with self._lock:
            self._connection.close()


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {name: row[name] for name in row.keys()}


__all__ = ["SQLiteMotionImportMetadataStore"]
