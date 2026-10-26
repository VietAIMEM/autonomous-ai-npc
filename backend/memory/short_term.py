"""Bounded short-term memory for current context and recent experience."""

from __future__ import annotations

from itertools import count
from typing import Any, Optional

from ..chat.models import ChatMessage, DialogueResponse
from .models import MemoryConfig, ShortTermItem, utc_now


class ShortTermMemory:
    """Small, bounded buffers that never evict the active context pointers."""

    def __init__(self, config: MemoryConfig) -> None:
        self.config = config
        self.messages: list[ShortTermItem] = []
        self.events: list[ShortTermItem] = []
        self.actions: list[ShortTermItem] = []
        self.current_goal: Optional[str] = None
        self.current_action: Optional[str] = None
        self.world_context: dict[str, Any] = {}
        self._id_sequence = count(1)

    def add_message(self, message: ChatMessage, response: DialogueResponse | None = None) -> None:
        self._add(
            self.messages,
            self.config.short_term_message_limit,
            text=f"{message.viewer_id}: {message.raw_text}",
            importance=max(0.2, message.priority),
            viewer_id=message.viewer_id,
            metadata={
                "message_id": message.message_id,
                "intent": message.intent.value,
                "target": message.target,
            },
        )
        if response is not None:
            self.add_event(
                f"Lacrimosa: {response.speech}",
                importance=max(0.2, response.priority),
                viewer_id=message.viewer_id,
                metadata={"message_id": message.message_id},
            )

    def add_event(
        self,
        text: str,
        *,
        importance: float = 0.2,
        viewer_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._add(
            self.events,
            self.config.short_term_event_limit,
            text=text,
            importance=importance,
            viewer_id=viewer_id,
            metadata=metadata or {},
        )

    def add_action(
        self,
        text: str,
        *,
        importance: float = 0.2,
        viewer_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._add(
            self.actions,
            self.config.short_term_action_limit,
            text=text,
            importance=importance,
            viewer_id=viewer_id,
            metadata=metadata or {},
        )

    def set_current_goal(self, goal: str | None) -> None:
        self.current_goal = goal

    def set_current_action(self, action: str | None) -> None:
        self.current_action = action

    def set_world_context(self, context: dict[str, Any]) -> None:
        self.world_context = dict(context)

    def snapshot(self) -> dict[str, Any]:
        return {
            "messages": [item.to_dict() for item in self.messages],
            "events": [item.to_dict() for item in self.events],
            "actions": [item.to_dict() for item in self.actions],
            "sizes": {
                "messages": len(self.messages),
                "events": len(self.events),
                "actions": len(self.actions),
            },
            "current_goal": self.current_goal,
            "current_action": self.current_action,
            "world_context": self.world_context,
        }

    def _add(
        self,
        items: list[ShortTermItem],
        limit: int,
        *,
        text: str,
        importance: float,
        viewer_id: str | None,
        metadata: dict[str, Any],
    ) -> None:
        item = ShortTermItem(
            item_id=f"short_{next(self._id_sequence):06d}",
            kind="message" if items is self.messages else "event" if items is self.events else "action",
            text=text,
            timestamp=utc_now(),
            importance=max(0.0, min(1.0, importance)),
            viewer_id=viewer_id,
            metadata=metadata,
        )
        if len(items) >= limit:
            self._evict_one(items)
        items.append(item)

    @staticmethod
    def _evict_one(items: list[ShortTermItem]) -> None:
        if not items:
            return
        evict_index = min(
            range(len(items)),
            key=lambda index: (items[index].importance, index),
        )
        del items[evict_index]
