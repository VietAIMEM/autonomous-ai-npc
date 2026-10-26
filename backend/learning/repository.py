"""SQLite persistence for Phase 10 knowledge; existing memory tables are untouched."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from ..chat.models import format_utc
from ..memory.repository import parse_datetime
from ..brain.models import ActionType
from .models import (
    CompositeKnowledge,
    MotionDefinition,
    MotionSource,
    MotionValidationStatus,
    SkeletonCompatibility,
    SkillKnowledge,
)


class SQLiteSkillLearningRepository:
    def __init__(self, database_path: Path | str) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(str(self.database_path), check_same_thread=False, timeout=5.0)
        self._connection.row_factory = sqlite3.Row
        self.initialize()

    def initialize(self) -> None:
        with self._lock, self._connection:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute(
                "CREATE TABLE IF NOT EXISTS motion_definitions ("
                "motion_id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT NOT NULL,"
                "animation_reference TEXT NOT NULL, duration_seconds REAL NOT NULL, tags TEXT NOT NULL,"
                "source TEXT NOT NULL, parameters TEXT NOT NULL, enabled INTEGER NOT NULL, version INTEGER NOT NULL,"
                "execution_action TEXT NOT NULL DEFAULT 'DANCE')"
            )
            existing_motion_columns = {
                row["name"] for row in self._connection.execute("PRAGMA table_info(motion_definitions)")
            }
            if "execution_action" not in existing_motion_columns:
                self._connection.execute(
                    "ALTER TABLE motion_definitions ADD COLUMN execution_action TEXT NOT NULL DEFAULT 'DANCE'"
                )
            motion_column_migrations = {
                "source_path": "TEXT",
                "source_format": "TEXT",
                "clip_name": "TEXT",
                "frame_count": "INTEGER",
                "sample_rate": "REAL",
                "bone_count": "INTEGER",
                "root_bone_name": "TEXT",
                "skeleton_status": "TEXT NOT NULL DEFAULT 'UNKNOWN'",
                "validation_status": "TEXT NOT NULL DEFAULT 'PENDING'",
                "source_motion_id": "TEXT",
                "source_version": "INTEGER",
                "processing_version": "INTEGER",
                "quality_score": "REAL",
                "quality_flags": "TEXT NOT NULL DEFAULT '[]'",
                "cleanup_status": "TEXT NOT NULL DEFAULT 'NOT_REQUIRED'",
                "processing_timestamp": "TEXT",
                "last_error": "TEXT",
            }
            for column_name, column_type in motion_column_migrations.items():
                if column_name not in existing_motion_columns:
                    self._connection.execute(
                        f"ALTER TABLE motion_definitions ADD COLUMN {column_name} {column_type}"
                    )
            self._connection.execute(
                "CREATE TABLE IF NOT EXISTS skill_knowledge ("
                "skill_id TEXT NOT NULL, context_key TEXT NOT NULL, trials INTEGER NOT NULL,"
                "successes INTEGER NOT NULL, failures INTEGER NOT NULL, interruptions INTEGER NOT NULL,"
                "positive_reactions INTEGER NOT NULL, negative_reactions INTEGER NOT NULL,"
                "total_score REAL NOT NULL, total_duration_seconds REAL NOT NULL, confidence REAL NOT NULL,"
                "exploration_count INTEGER NOT NULL, exploitation_count INTEGER NOT NULL,"
                "last_used TEXT, version INTEGER NOT NULL, PRIMARY KEY(skill_id, context_key))"
            )
            self._connection.execute(
                "CREATE TABLE IF NOT EXISTS learned_compositions ("
                "composite_skill_id TEXT NOT NULL, child_skill_ids TEXT NOT NULL, context_key TEXT NOT NULL,"
                "trials INTEGER NOT NULL, successes INTEGER NOT NULL, total_score REAL NOT NULL,"
                "confidence REAL NOT NULL, created_at TEXT, last_used TEXT, enabled INTEGER NOT NULL,"
                "PRIMARY KEY(composite_skill_id, context_key))"
            )
            existing_composite_columns = {
                row["name"] for row in self._connection.execute("PRAGMA table_info(learned_compositions)")
            }
            if "child_parameters" not in existing_composite_columns:
                self._connection.execute(
                    "ALTER TABLE learned_compositions ADD COLUMN child_parameters TEXT NOT NULL DEFAULT '[]'"
                )

    def save_motion(self, motion: MotionDefinition) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT OR REPLACE INTO motion_definitions "
                "(motion_id, name, description, animation_reference, duration_seconds, tags, source, "
                "parameters, enabled, version, execution_action, source_path, source_format, clip_name, "
                "frame_count, sample_rate, bone_count, root_bone_name, skeleton_status, validation_status, "
                "source_motion_id, source_version, processing_version, quality_score, quality_flags, "
                "cleanup_status, processing_timestamp, last_error) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    motion.motion_id,
                    motion.name,
                    motion.description,
                    motion.animation_reference,
                    motion.duration_seconds,
                    json.dumps(list(motion.tags)),
                    motion.source.value,
                    json.dumps(list(motion.parameters)),
                    int(motion.enabled),
                    motion.version,
                    motion.execution_action.value,
                    motion.source_path,
                    motion.source_format,
                    motion.clip_name,
                    motion.frame_count,
                    motion.sample_rate,
                    motion.bone_count,
                    motion.root_bone_name,
                    motion.skeleton_status.value,
                    motion.validation_status.value,
                    motion.source_motion_id,
                    motion.source_version,
                    motion.processing_version,
                    motion.quality_score,
                    json.dumps(list(motion.quality_flags)),
                    motion.cleanup_status,
                    motion.processing_timestamp,
                    motion.last_error,
                ),
            )

    def load_motions(self) -> dict[str, MotionDefinition]:
        with self._lock:
            rows = self._connection.execute("SELECT * FROM motion_definitions").fetchall()
        motions: dict[str, MotionDefinition] = {}
        for row in rows:
            try:
                motions[row["motion_id"]] = MotionDefinition(
                    row["motion_id"], row["name"], row["description"], row["animation_reference"],
                    row["duration_seconds"], tuple(json.loads(row["tags"])), MotionSource(row["source"]),
                    tuple(json.loads(row["parameters"])), ActionType(row["execution_action"]),
                    bool(row["enabled"]), row["version"],
                    _row_value(row, "source_path"),
                    _row_value(row, "source_format"),
                    _row_value(row, "clip_name"),
                    _optional_int(_row_value(row, "frame_count")),
                    _optional_float(_row_value(row, "sample_rate")),
                    _optional_int(_row_value(row, "bone_count")),
                    _row_value(row, "root_bone_name"),
                    _enum_value(
                        _row_value(row, "skeleton_status"),
                        SkeletonCompatibility,
                        SkeletonCompatibility.UNKNOWN,
                    ),
                    _enum_value(
                        _row_value(row, "validation_status"),
                        MotionValidationStatus,
                        MotionValidationStatus.PENDING,
                    ),
                    _row_value(row, "source_motion_id"),
                    _optional_int(_row_value(row, "source_version")),
                    _optional_int(_row_value(row, "processing_version")),
                    _optional_float(_row_value(row, "quality_score")),
                    _quality_flags(_row_value(row, "quality_flags")),
                    str(_row_value(row, "cleanup_status") or "NOT_REQUIRED"),
                    _row_value(row, "processing_timestamp"),
                    _row_value(row, "last_error"),
                )
            except (ValueError, TypeError, json.JSONDecodeError):
                continue
        return motions

    def save_knowledge(self, knowledge: SkillKnowledge) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT OR REPLACE INTO skill_knowledge VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    knowledge.skill_id, knowledge.context_key, knowledge.trials, knowledge.successes,
                    knowledge.failures, knowledge.interruptions, knowledge.positive_reactions,
                    knowledge.negative_reactions, knowledge.total_score, knowledge.total_duration_seconds,
                    knowledge.confidence, knowledge.exploration_count, knowledge.exploitation_count,
                    format_utc(knowledge.last_used) if knowledge.last_used else None, knowledge.version,
                ),
            )

    def load_knowledge(self) -> dict[tuple[str, str], SkillKnowledge]:
        with self._lock:
            rows = self._connection.execute("SELECT * FROM skill_knowledge").fetchall()
        result: dict[tuple[str, str], SkillKnowledge] = {}
        for row in rows:
            result[(row["skill_id"], row["context_key"])] = SkillKnowledge(
                row["skill_id"], row["context_key"], row["trials"], row["successes"], row["failures"],
                row["interruptions"], row["positive_reactions"], row["negative_reactions"],
                row["total_score"], row["total_duration_seconds"], row["confidence"],
                row["exploration_count"], row["exploitation_count"], parse_datetime(row["last_used"]), row["version"],
            )
        return result

    def save_composite(self, composite: CompositeKnowledge) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT OR REPLACE INTO learned_compositions "
                "(composite_skill_id, child_skill_ids, context_key, trials, successes, total_score, "
                "confidence, created_at, last_used, enabled, child_parameters) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    composite.composite_skill_id, json.dumps(list(composite.child_skill_ids)),
                    composite.context_key, composite.trials, composite.successes, composite.total_score,
                    composite.confidence,
                    format_utc(composite.created_at) if composite.created_at else None,
                    format_utc(composite.last_used) if composite.last_used else None,
                    int(composite.enabled),
                    json.dumps([dict(parameters) for parameters in composite.child_parameters]),
                ),
            )

    def load_composites(self) -> dict[tuple[str, str], CompositeKnowledge]:
        with self._lock:
            rows = self._connection.execute("SELECT * FROM learned_compositions").fetchall()
        result: dict[tuple[str, str], CompositeKnowledge] = {}
        for row in rows:
            try:
                child_parameters = tuple(json.loads(row["child_parameters"]))
            except (KeyError, TypeError, json.JSONDecodeError):
                child_parameters = ()
            result[(row["composite_skill_id"], row["context_key"])] = CompositeKnowledge(
                row["composite_skill_id"], tuple(json.loads(row["child_skill_ids"])), row["context_key"],
                row["trials"], row["successes"], row["total_score"], row["confidence"],
                parse_datetime(row["created_at"]), parse_datetime(row["last_used"]), bool(row["enabled"]),
                child_parameters,
            )
        return result

    def reset(self) -> None:
        with self._lock, self._connection:
            self._connection.execute("DELETE FROM skill_knowledge")
            self._connection.execute("DELETE FROM learned_compositions")

    def close(self) -> None:
        with self._lock:
            self._connection.close()


def _row_value(row: sqlite3.Row, name: str) -> object | None:
    return row[name] if name in row.keys() else None


def _optional_int(value: object | None) -> int | None:
    return int(value) if value is not None else None


def _optional_float(value: object | None) -> float | None:
    return float(value) if value is not None else None


def _enum_value(value: object | None, enum_type, default):
    if value is None:
        return default
    try:
        return enum_type(str(value))
    except ValueError:
        return default


def _quality_flags(value: object | None) -> tuple[str, ...]:
    if value is None:
        return ()
    try:
        parsed = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return ()
    if not isinstance(parsed, list):
        return ()
    return tuple(str(item) for item in parsed)
