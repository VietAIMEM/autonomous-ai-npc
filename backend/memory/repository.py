"""Persistent memory repository boundary."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, Protocol

from .models import (
    Episode,
    InteractionRecord,
    SemanticMemory,
    ViewerMemory,
)


class MemoryRepository(Protocol):
    def initialize(self) -> None:
        ...

    def next_id(self, kind: str) -> str:
        ...

    def save_episode(self, episode: Episode) -> None:
        ...

    def get_episode(self, episode_id: str) -> Optional[Episode]:
        ...

    def search_episodes(
        self,
        viewer_id: Optional[str] = None,
        topic: Optional[str] = None,
        limit: int = 100,
    ) -> list[Episode]:
        ...

    def save_semantic_memory(self, memory: SemanticMemory) -> None:
        ...

    def find_semantic_memory(
        self,
        subject: str,
        predicate: str,
        value: Optional[str] = None,
    ) -> Optional[SemanticMemory]:
        ...

    def search_semantic_memory(
        self,
        subject: Optional[str] = None,
        topic: Optional[str] = None,
        limit: int = 100,
    ) -> list[SemanticMemory]:
        ...

    def get_viewer_memory(self, viewer_id: str) -> Optional[ViewerMemory]:
        ...

    def list_viewer_memories(self, limit: int = 50) -> list[ViewerMemory]:
        ...

    def save_viewer_memory(self, memory: ViewerMemory) -> None:
        ...

    def get_metadata(self, key: str) -> Optional[str]:
        ...

    def set_metadata(self, key: str, value: str) -> None:
        ...

    def save_interaction(self, record: InteractionRecord) -> None:
        ...

    def search_interactions(
        self,
        viewer_id: Optional[str] = None,
        topic: Optional[str] = None,
        limit: int = 100,
    ) -> list[InteractionRecord]:
        ...

    def counts(self) -> dict[str, int]:
        ...

    def last_episode(self) -> Optional[Episode]:
        ...

    def close(self) -> None:
        ...


def parse_datetime(value: str | None) -> Optional[datetime]:
    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
