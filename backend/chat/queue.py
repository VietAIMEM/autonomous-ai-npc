"""Bounded attention queue for selected chat messages."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional

from .models import ChatMessage


@dataclass(frozen=True)
class QueueEntry:
    message: ChatMessage


class AttentionQueue:
    def __init__(
        self,
        capacity: int = 100,
        *,
        max_age_seconds: float | None = None,
        clock: Callable[[], datetime] | None = None,
        on_evict: Callable[[int], None] | None = None,
        on_expire: Callable[[int], None] | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("queue capacity must be greater than zero")
        self.capacity = capacity
        #: Optional staleness bound. A busy real chat channel can hold more
        #: low-value messages than the Brain can attend to; replaying them
        #: minutes later produces stale reactions. When set, entries older
        #: than this are dropped on the next peek/dequeue instead of being
        #: answered late. ``None`` keeps the historical behaviour for tests
        #: and callers that manage their own lifetime.
        self.max_age_seconds = (
            None if max_age_seconds is None else max(0.0, float(max_age_seconds))
        )
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._items: list[ChatMessage] = []
        #: Number of low-value messages dropped to admit a more valuable one.
        #: Exposed so the runtime can prove backlog pressure stayed bounded
        #: without the reservoir growing.
        self.evicted_count = 0
        #: Number of messages dropped because they were already stale.
        self.expired_count = 0
        #: PA29-004: optional observers so the owning service can publish the
        #: same terminal dispositions into the telemetry registry. They are
        #: notified exactly once per dropped message, which keeps the metric
        #: and ``evicted_count``/``expired_count`` reconcilable.
        self._on_evict = on_evict
        self._on_expire = on_expire

    def __len__(self) -> int:
        return len(self._items)

    def expire(self) -> int:
        """Drop already-stale entries and return how many were removed."""
        before = len(self._items)
        self._expire()
        return before - len(self._items)

    def _expire(self) -> None:
        if (
            self.max_age_seconds is None
            or self.max_age_seconds <= 0.0
            or not self._items
        ):
            return
        now = self._clock()
        kept: list[ChatMessage] = []
        for item in self._items:
            timestamp = item.timestamp
            if timestamp is None:
                kept.append(item)
                continue
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            if (now - timestamp).total_seconds() > self.max_age_seconds:
                self.expired_count += 1
                self._notify(self._on_expire)
                continue
            kept.append(item)
        self._items = kept

    def enqueue(self, message: ChatMessage) -> bool:
        self._expire()
        if len(self._items) >= self.capacity:
            # A full queue used to reject the incoming message outright, so a
            # burst of low-value chat could lock a genuine viewer request out
            # of the attention budget. The bound stays; the queue now drops
            # its own lowest-ranked entry when the incoming message outranks
            # it. Priority and freshness come from the same ordering the
            # dequeuer uses, so nothing else about selection changes.
            lowest = self._items[-1]
            if self._rank(message) >= self._rank(lowest):
                return False
            self._items.pop()
            self.evicted_count += 1
            self._notify(self._on_evict)
        self._items.append(message)
        self._items.sort(key=self._rank)
        return True

    @staticmethod
    def _notify(callback: Optional[Callable[[int], None]]) -> None:
        """Notify one observer of a single terminal disposition.

        A broken observer must never break message intake; the same rule the
        metrics helpers already follow.
        """
        if callback is None:
            return
        try:
            callback(1)
        except Exception:  # pragma: no cover - defensive
            pass

    @staticmethod
    def _rank(message: ChatMessage) -> tuple[float, float]:
        """Selection order: highest priority first, freshest first in a class.

        PHASE 24: a saturated real channel (measured ~2-3 msg/s) kept the
        queue at capacity, so oldest-first tie-breaking answered the oldest
        survivor every tick - real chat replies landed 60-110s late while
        fresh messages were dropped. Priority still protects genuine viewer
        requests; within one priority class the queue now answers what is
        live and lets the rest expire instead of replaying stale backlog.
        """
        timestamp = message.timestamp
        fresh = timestamp.timestamp() if timestamp is not None else 0.0
        return (-message.priority, -fresh)

    def dequeue(self) -> Optional[ChatMessage]:
        self._expire()
        if not self._items:
            return None
        return self._items.pop(0)

    def peek(self) -> Optional[ChatMessage]:
        self._expire()
        if not self._items:
            return None
        return self._items[0]

    def remove(self, message_id: str) -> Optional[ChatMessage]:
        for index, item in enumerate(self._items):
            if item.message_id == message_id:
                return self._items.pop(index)
        return None

    def find(self, message_id: str) -> Optional[ChatMessage]:
        for item in self._items:
            if item.message_id == message_id:
                return item
        return None

    def clear(self) -> None:
        self._items.clear()

    def snapshot(self) -> list[ChatMessage]:
        return list(self._items)
