"""SQLite-backed implementation of the memory repository."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from ..chat.models import format_utc
from .models import (
    SCHEMA_VERSION,
    Episode,
    InteractionRecord,
    SemanticMemory,
    ViewerMemory,
    utc_now,
)
from .repository import parse_datetime


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _tags(value: Any) -> tuple[str, ...]:
    if not value:
        return ()
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return ()
    return tuple(item for item in parsed if isinstance(item, str))


class SQLiteMemoryRepository:
    def __init__(self, database_path: Path | str) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            str(self.database_path),
            check_same_thread=False,
            timeout=5.0,
        )
        self._connection.row_factory = sqlite3.Row
        self.initialize()

    def initialize(self) -> None:
        with self._lock:
            with self._connection:
                self._connection.execute("PRAGMA journal_mode=WAL")
                self._connection.execute(
                    "CREATE TABLE IF NOT EXISTS memory_metadata ("
                    "key TEXT PRIMARY KEY, value TEXT NOT NULL)"
                )
                self._connection.execute(
                    "INSERT INTO memory_metadata(key, value) VALUES ('schema_version', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (str(SCHEMA_VERSION),),
                )
                self._connection.execute(
                    "CREATE TABLE IF NOT EXISTS episodes ("
                    "episode_id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, event TEXT NOT NULL,"
                    "context TEXT NOT NULL, action TEXT, result TEXT NOT NULL, importance REAL NOT NULL,"
                    "viewer_id TEXT, goal TEXT, location TEXT, tags TEXT NOT NULL)"
                )
                self._connection.execute(
                    "CREATE TABLE IF NOT EXISTS semantic_memories ("
                    "memory_id TEXT PRIMARY KEY, subject TEXT NOT NULL, predicate TEXT NOT NULL,"
                    "value TEXT NOT NULL, confidence REAL NOT NULL, importance REAL NOT NULL,"
                    "source TEXT NOT NULL, evidence_count INTEGER NOT NULL,"
                    "created_at TEXT NOT NULL, updated_at TEXT NOT NULL,"
                    "UNIQUE(subject, predicate, value))"
                )
                self._connection.execute(
                    "CREATE TABLE IF NOT EXISTS viewer_memories ("
                    "viewer_id TEXT PRIMARY KEY, interaction_count INTEGER NOT NULL,"
                    "successful_interactions INTEGER NOT NULL, failed_interactions INTEGER NOT NULL,"
                    "relationship_score REAL NOT NULL, last_interaction TEXT,"
                    "preferences TEXT NOT NULL, recent_topics TEXT NOT NULL,"
                    "relationship_level TEXT NOT NULL DEFAULT 'NEW',"
                    "engagement_score REAL NOT NULL DEFAULT 0,"
                    "familiarity REAL NOT NULL DEFAULT 0,"
                    "preferred_skills TEXT NOT NULL DEFAULT '[]',"
                    "preferred_topics TEXT NOT NULL DEFAULT '[]')"
                )
                self._migrate_viewer_memories()
                self._connection.execute(
                    "CREATE TABLE IF NOT EXISTS interactions ("
                    "interaction_id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, viewer_id TEXT,"
                    "message_id TEXT, intent TEXT, requested_action TEXT, action TEXT,"
                    "response TEXT, result TEXT NOT NULL, importance REAL NOT NULL)"
                )
                self._connection.execute(
                    "CREATE INDEX IF NOT EXISTS idx_episodes_viewer ON episodes(viewer_id)"
                )
                self._connection.execute(
                    "CREATE INDEX IF NOT EXISTS idx_interactions_viewer ON interactions(viewer_id)"
                )

    def _migrate_viewer_memories(self) -> None:
        """PHASE 19: append-only column migration inside the existing database."""
        columns = {
            row["name"]
            for row in self._connection.execute("PRAGMA table_info(viewer_memories)").fetchall()
        }
        additions = (
            ("relationship_level", "TEXT NOT NULL DEFAULT 'NEW'"),
            ("engagement_score", "REAL NOT NULL DEFAULT 0"),
            ("familiarity", "REAL NOT NULL DEFAULT 0"),
            ("preferred_skills", "TEXT NOT NULL DEFAULT '[]'"),
            ("preferred_topics", "TEXT NOT NULL DEFAULT '[]'"),
        )
        for name, definition in additions:
            if name not in columns:
                self._connection.execute(
                    f"ALTER TABLE viewer_memories ADD COLUMN {name} {definition}"
                )

    def get_metadata(self, key: str) -> Optional[str]:
        rows = self._query("SELECT value FROM memory_metadata WHERE key = ?", (key,))
        return rows[0]["value"] if rows else None

    def set_metadata(self, key: str, value: str) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT INTO memory_metadata(key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    def next_id(self, kind: str) -> str:
        queries = {
            "episode": ("episodes", "episode_id", "ep_"),
            "semantic": ("semantic_memories", "memory_id", "sem_"),
            "interaction": ("interactions", "interaction_id", "int_"),
        }
        if kind not in queries:
            raise ValueError("invalid_memory_id_kind")
        table, column, prefix = queries[kind]
        digits = len(prefix)
        with self._lock:
            row = self._connection.execute(
                f"SELECT {column} FROM {table} "
                f"WHERE {column} LIKE ? "
                f"ORDER BY CAST(SUBSTR({column}, {digits + 1}) AS INTEGER) DESC LIMIT 1",
                (f"{prefix}%",),
            ).fetchone()
        number = int(row[0][digits:]) + 1 if row and row[0][digits:].isdigit() else 1
        return f"{prefix}{number:06d}"

    def save_episode(self, episode: Episode) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT OR REPLACE INTO episodes VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    episode.episode_id,
                    format_utc(episode.timestamp),
                    episode.event,
                    _json(episode.context),
                    episode.action,
                    episode.result,
                    episode.importance,
                    episode.viewer_id,
                    episode.goal,
                    episode.location,
                    _json(list(episode.tags)),
                ),
            )

    def get_episode(self, episode_id: str) -> Optional[Episode]:
        rows = self._query("SELECT * FROM episodes WHERE episode_id = ?", (episode_id,))
        return self._episode_from_row(rows[0]) if rows else None

    def search_episodes(
        self,
        viewer_id: Optional[str] = None,
        topic: Optional[str] = None,
        limit: int = 100,
    ) -> list[Episode]:
        clauses: list[str] = []
        params: list[Any] = []
        if viewer_id:
            clauses.append("viewer_id = ?")
            params.append(viewer_id)
        if topic:
            clauses.append("(event LIKE ? OR goal LIKE ? OR tags LIKE ? OR context LIKE ?)")
            like = f"%{topic.lower()}%"
            params.extend([like, like, like, like])
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        rows = self._query(
            f"SELECT * FROM episodes {where} ORDER BY timestamp DESC LIMIT ?",
            params,
        )
        return [self._episode_from_row(row) for row in rows]

    def save_semantic_memory(self, memory: SemanticMemory) -> None:
        now = format_utc(memory.updated_at or utc_now())
        created = format_utc(memory.created_at or memory.updated_at or utc_now())
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT OR REPLACE INTO semantic_memories VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    memory.memory_id,
                    memory.subject,
                    memory.predicate,
                    memory.value,
                    memory.confidence,
                    memory.importance,
                    memory.source,
                    memory.evidence_count,
                    created,
                    now,
                ),
            )

    def find_semantic_memory(
        self,
        subject: str,
        predicate: str,
        value: Optional[str] = None,
    ) -> Optional[SemanticMemory]:
        if value is None:
            rows = self._query(
                "SELECT * FROM semantic_memories WHERE subject = ? AND predicate = ?",
                (subject, predicate),
            )
        else:
            rows = self._query(
                "SELECT * FROM semantic_memories WHERE subject = ? AND predicate = ? AND value = ?",
                (subject, predicate, value),
            )
        return self._semantic_from_row(rows[0]) if rows else None

    def search_semantic_memory(
        self,
        subject: Optional[str] = None,
        topic: Optional[str] = None,
        limit: int = 100,
    ) -> list[SemanticMemory]:
        clauses: list[str] = []
        params: list[Any] = []
        if subject:
            clauses.append("subject = ?")
            params.append(subject)
        if topic:
            clauses.append("(value LIKE ? OR predicate LIKE ? OR subject LIKE ?)")
            like = f"%{topic.lower()}%"
            params.extend([like, like, like])
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        rows = self._query(
            f"SELECT * FROM semantic_memories {where} ORDER BY updated_at DESC LIMIT ?",
            params,
        )
        return [self._semantic_from_row(row) for row in rows]

    def get_viewer_memory(self, viewer_id: str) -> Optional[ViewerMemory]:
        rows = self._query("SELECT * FROM viewer_memories WHERE viewer_id = ?", (viewer_id,))
        if not rows:
            return None
        return self._viewer_from_row(rows[0])

    def list_viewer_memories(self, limit: int = 50) -> list[ViewerMemory]:
        """Bounded, read-only listing for operator surfaces (PHASE 23)."""
        bounded = max(1, min(int(limit), 200))
        rows = self._query(
            "SELECT * FROM viewer_memories ORDER BY last_interaction DESC LIMIT ?",
            (bounded,),
        )
        return [self._viewer_from_row(row) for row in rows]

    def _viewer_from_row(self, row: sqlite3.Row) -> ViewerMemory:
        return ViewerMemory(
            viewer_id=row["viewer_id"],
            interaction_count=row["interaction_count"],
            successful_interactions=row["successful_interactions"],
            failed_interactions=row["failed_interactions"],
            relationship_score=row["relationship_score"],
            last_interaction=parse_datetime(row["last_interaction"]),
            preferences=_tags(row["preferences"]),
            recent_topics=_tags(row["recent_topics"]),
            relationship_level=row["relationship_level"],
            engagement_score=row["engagement_score"],
            familiarity=row["familiarity"],
            preferred_skills=_tags(row["preferred_skills"]),
            preferred_topics=_tags(row["preferred_topics"]),
        )

    def save_viewer_memory(self, memory: ViewerMemory) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT OR REPLACE INTO viewer_memories ("
                "viewer_id, interaction_count, successful_interactions, failed_interactions,"
                "relationship_score, last_interaction, preferences, recent_topics,"
                "relationship_level, engagement_score, familiarity, preferred_skills,"
                "preferred_topics) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    memory.viewer_id,
                    memory.interaction_count,
                    memory.successful_interactions,
                    memory.failed_interactions,
                    memory.relationship_score,
                    format_utc(memory.last_interaction) if memory.last_interaction else None,
                    _json(list(memory.preferences)),
                    _json(list(memory.recent_topics)),
                    memory.relationship_level,
                    memory.engagement_score,
                    memory.familiarity,
                    _json(list(memory.preferred_skills)),
                    _json(list(memory.preferred_topics)),
                ),
            )

    def save_interaction(self, record: InteractionRecord) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT OR REPLACE INTO interactions VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    record.interaction_id,
                    format_utc(record.timestamp),
                    record.viewer_id,
                    record.message_id,
                    record.intent,
                    record.requested_action,
                    record.action,
                    record.response,
                    record.result,
                    record.importance,
                ),
            )

    def search_interactions(
        self,
        viewer_id: Optional[str] = None,
        topic: Optional[str] = None,
        limit: int = 100,
    ) -> list[InteractionRecord]:
        clauses: list[str] = []
        params: list[Any] = []
        if viewer_id:
            clauses.append("viewer_id = ?")
            params.append(viewer_id)
        if topic:
            clauses.append(
                "(requested_action LIKE ? OR action LIKE ? OR intent LIKE ? OR response LIKE ?)"
            )
            like = f"%{topic.lower()}%"
            params.extend([like, like, like, like])
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        rows = self._query(
            f"SELECT * FROM interactions {where} ORDER BY timestamp DESC LIMIT ?",
            params,
        )
        return [
            InteractionRecord(
                row["interaction_id"],
                parse_datetime(row["timestamp"]) or datetime.now().astimezone(),
                row["viewer_id"],
                row["message_id"],
                row["intent"],
                row["requested_action"],
                row["action"],
                row["response"],
                row["result"],
                row["importance"],
            )
            for row in rows
        ]

    def counts(self) -> dict[str, int]:
        return {
            "episodes": self._scalar("SELECT COUNT(*) FROM episodes"),
            "semantic_memories": self._scalar("SELECT COUNT(*) FROM semantic_memories"),
            "viewer_memories": self._scalar("SELECT COUNT(*) FROM viewer_memories"),
            "interactions": self._scalar("SELECT COUNT(*) FROM interactions"),
        }

    def last_episode(self) -> Optional[Episode]:
        rows = self._query("SELECT * FROM episodes ORDER BY timestamp DESC LIMIT 1")
        return self._episode_from_row(rows[0]) if rows else None

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def _query(self, sql: str, params: list[Any] | tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return list(self._connection.execute(sql, params).fetchall())

    def _scalar(self, sql: str) -> int:
        with self._lock:
            return int(self._connection.execute(sql).fetchone()[0])

    def _episode_from_row(self, row: sqlite3.Row) -> Episode:
        return Episode(
            row["episode_id"],
            parse_datetime(row["timestamp"]) or datetime.now().astimezone(),
            row["event"],
            json.loads(row["context"]),
            row["action"],
            row["result"],
            row["importance"],
            row["viewer_id"],
            row["goal"],
            row["location"],
            _tags(row["tags"]),
        )

    def _semantic_from_row(self, row: sqlite3.Row) -> SemanticMemory:
        return SemanticMemory(
            row["memory_id"],
            row["subject"],
            row["predicate"],
            row["value"],
            row["confidence"],
            row["importance"],
            row["source"],
            row["evidence_count"],
            parse_datetime(row["created_at"]),
            parse_datetime(row["updated_at"]),
        )
