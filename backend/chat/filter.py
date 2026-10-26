"""Lightweight validation, noise and rate filtering."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from difflib import SequenceMatcher
from typing import Optional

from .models import ChatMessage


@dataclass(frozen=True)
class FilterConfig:
    max_text_length: int = 500
    duplicate_window_seconds: int = 30
    duplicate_similarity: float = 0.94
    rate_window_seconds: int = 10
    max_messages_per_rate_window: int = 5


@dataclass(frozen=True)
class FilterDecision:
    accepted: bool
    reason: Optional[str] = None


@dataclass(frozen=True)
class _SeenMessage:
    message_id: str
    normalized_text: str
    timestamp: object


class ChatFilter:
    def __init__(self, config: FilterConfig | None = None) -> None:
        self.config = config or FilterConfig()
        self._seen: dict[str, list[_SeenMessage]] = {}
        self._timestamps: dict[str, list[object]] = {}

    def evaluate(self, message: ChatMessage) -> FilterDecision:
        if not message.normalized_text:
            return FilterDecision(False, "empty")
        if len(message.normalized_text) > self.config.max_text_length:
            return FilterDecision(False, "too_long")

        viewer = message.viewer_id.casefold()
        window = timedelta(seconds=self.config.rate_window_seconds)
        recent = [
            stamp
            for stamp in self._timestamps.get(viewer, [])
            if message.timestamp - stamp <= window
        ]
        if len(recent) >= self.config.max_messages_per_rate_window:
            return FilterDecision(False, "rate_limited")
        return FilterDecision(True)

    def find_duplicate(self, message: ChatMessage) -> Optional[_SeenMessage]:
        viewer = message.viewer_id.casefold()
        window = timedelta(seconds=self.config.duplicate_window_seconds)
        self._prune(message.timestamp)
        for seen in reversed(self._seen.get(viewer, [])):
            if message.timestamp - seen.timestamp > window:
                break
            similarity = SequenceMatcher(
                None,
                message.normalized_text.casefold(),
                seen.normalized_text,
            ).ratio()
            if similarity >= self.config.duplicate_similarity:
                return seen
        return None

    def record(self, message: ChatMessage, accepted: bool) -> None:
        viewer = message.viewer_id.casefold()
        self._timestamps.setdefault(viewer, []).append(message.timestamp)
        if accepted:
            self._seen.setdefault(viewer, []).append(
                _SeenMessage(
                    message.message_id,
                    message.normalized_text.casefold(),
                    message.timestamp,
                )
            )
        self._prune(message.timestamp)

    def _prune(self, now: object) -> None:
        duplicate_window = timedelta(seconds=self.config.duplicate_window_seconds)
        rate_window = timedelta(seconds=self.config.rate_window_seconds)
        self._seen = {
            viewer: [
                item
                for item in items
                if now - item.timestamp <= duplicate_window
            ]
            for viewer, items in self._seen.items()
        }
        self._timestamps = {
            viewer: [
                stamp
                for stamp in stamps
                if now - stamp <= rate_window
            ]
            for viewer, stamps in self._timestamps.items()
        }
