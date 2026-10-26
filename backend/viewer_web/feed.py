"""Bounded public transcript assembled from the real chat pipeline state.

This is a presentation buffer, not a chat backend: it only observes messages
that the existing ``ChatService`` queue / ``SocialContext`` already hold and
the one ``last_response`` the Brain produced. It never advances the pipeline,
never invents a message, and never stores anything about relationships.
"""

from __future__ import annotations

import threading
from collections import OrderedDict, deque
from datetime import datetime, timezone
from typing import Iterable, Mapping, Optional

from ..chat.models import ChatStatus, utc_now

#: Public delivery vocabulary the viewer page renders.
DELIVERY_SENT = "sent"
DELIVERY_ACCEPTED = "accepted"
DELIVERY_DELIVERED = "delivered"
DELIVERY_FAILED = "failed"

_DELIVERY_BY_STATUS = {
    ChatStatus.RECEIVED.value: DELIVERY_SENT,
    ChatStatus.QUEUED.value: DELIVERY_ACCEPTED,
    ChatStatus.PROCESSED.value: DELIVERY_DELIVERED,
    ChatStatus.FILTERED.value: DELIVERY_FAILED,
    ChatStatus.REJECTED.value: DELIVERY_FAILED,
}

#: Public, non-diagnostic explanations for a refused message.
_REASON_LABELS = {
    "empty": "the message was empty",
    "too_long": "the message is too long",
    "rate_limited": "sending too fast, try again in a moment",
    "duplicate": "that looked like a repeat of your recent message",
    "queue_full": "chat is busy right now, try again in a moment",
}


def delivery_for_status(status: object) -> str:
    value = status.value if isinstance(status, ChatStatus) else str(status or "")
    return _DELIVERY_BY_STATUS.get(value, DELIVERY_ACCEPTED)


def public_reason(reason: object) -> Optional[str]:
    if not isinstance(reason, str) or not reason:
        return None
    return _REASON_LABELS.get(reason, "the message could not be accepted")


def _format_utc(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def project_message(
    message: Mapping[str, object],
    *,
    client_message_id: Optional[str] = None,
) -> Optional[dict[str, object]]:
    """Whitelist projection of one ``ChatMessage.to_dict()`` payload."""
    message_id = message.get("message_id")
    viewer_id = message.get("viewer_id")
    text = message.get("text")
    if not isinstance(message_id, str) or not isinstance(viewer_id, str):
        return None
    if not isinstance(text, str):
        return None
    status = message.get("status")
    item: dict[str, object] = {
        "id": message_id,
        "role": "viewer",
        "viewer_id": viewer_id,
        "text": text,
        "at": message.get("timestamp"),
        "status": status,
        "delivery": delivery_for_status(status),
        "reason": public_reason(message.get("filtered_reason")),
    }
    if client_message_id:
        item["client_id"] = client_message_id
    return item


class ChatFeed:
    """Ordered, bounded, deduplicated view of real chat activity."""

    def __init__(self, capacity: int = 120) -> None:
        self._capacity = max(10, int(capacity))
        self._lock = threading.Lock()
        self._items: "OrderedDict[str, dict[str, object]]" = OrderedDict()
        self._order: deque[str] = deque()
        self._version = 0

    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    def observe(
        self,
        messages: Iterable[Mapping[str, object]],
        response: Optional[Mapping[str, object]] = None,
        *,
        client_ids: Optional[Mapping[str, str]] = None,
        now: Optional[datetime] = None,
    ) -> bool:
        """Upsert viewer messages and the latest Lacrimosa reply.

        ``client_ids`` maps a server message id to the browser-side request id
        so the page can reconcile its optimistic bubble without a second
        identity mechanism.
        """
        changed = False
        with self._lock:
            for message in messages:
                if not isinstance(message, Mapping):
                    continue
                message_id = message.get("message_id")
                client_id: Optional[str] = None
                if isinstance(client_ids, Mapping) and isinstance(message_id, str):
                    candidate = client_ids.get(message_id)
                    client_id = candidate if isinstance(candidate, str) else None
                projected = project_message(message, client_message_id=client_id)
                if projected is None:
                    continue
                if self._upsert_locked(projected["id"], projected):
                    changed = True
            if isinstance(response, Mapping):
                speech = response.get("speech")
                reply_to = response.get("message_id")
                if isinstance(speech, str) and speech.strip() and isinstance(reply_to, str):
                    reply = {
                        "id": f"resp:{reply_to}",
                        "role": "lacrimosa",
                        "viewer_id": None,
                        "text": speech.strip(),
                        "at": _format_utc(now or utc_now()),
                        "status": ChatStatus.PROCESSED.value,
                        "delivery": DELIVERY_DELIVERED,
                        "reply_to": reply_to,
                        "reason": None,
                    }
                    if self._upsert_locked(reply["id"], reply):
                        changed = True
            if changed:
                self._version += 1
        return changed

    def _upsert_locked(self, item_id: object, item: dict[str, object]) -> bool:
        assert isinstance(item_id, str)
        existing = self._items.get(item_id)
        if existing is not None:
            comparable = {key: value for key, value in item.items() if value is not None}
            updated = {**existing, **comparable}
            if updated == existing:
                return False
            self._items[item_id] = updated
            return True
        self._items[item_id] = item
        self._order.append(item_id)
        while len(self._order) > self._capacity:
            oldest = self._order.popleft()
            self._items.pop(oldest, None)
        return True

    def snapshot(self, limit: int = 60) -> dict[str, object]:
        with self._lock:
            ids = list(self._order)[-max(1, int(limit)):]
            # Discovery order is not transcript order: a message can be first
            # observed in the queue after a newer one was already recorded via
            # ``SocialContext`` (processed before the next poll), which would
            # leave the older row appended at the tail. Sort by the server
            # timestamp instead; the insertion index breaks same-second ties
            # and keeps rows without a usable stamp attached to the row before
            # them.
            ordered: list[tuple[str, int, str]] = []
            carry = ""
            for index, item_id in enumerate(ids):
                stamp = self._items[item_id].get("at")
                if isinstance(stamp, datetime):
                    stamp = _format_utc(stamp)
                if isinstance(stamp, str) and stamp:
                    carry = stamp
                ordered.append((carry, index, item_id))
            ordered.sort()
            return {
                "version": self._version,
                "messages": [dict(self._items[item_id]) for _, _, item_id in ordered],
            }
