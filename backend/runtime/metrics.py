"""Bounded in-process metrics and a small recent-error ledger.

Deliberately not Prometheus: every series is capped, latency history keeps a
fixed number of samples per name, and the error ledger is a fixed-size deque.
Nothing here can grow with traffic.
"""

from __future__ import annotations

import threading
from collections import deque
from datetime import datetime, timezone
from typing import Any, Deque, Optional

from .redaction import redact_text


MAX_SERIES = 128
MAX_LATENCY_SAMPLES = 64
MAX_RECENT_ERRORS = 20


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


class MetricsRegistry:
    """Thread-safe counters, gauges and bounded latency series."""

    def __init__(
        self,
        max_series: int = MAX_SERIES,
        max_samples: int = MAX_LATENCY_SAMPLES,
        max_errors: int = MAX_RECENT_ERRORS,
        clock=_utc_now,
    ) -> None:
        self._lock = threading.RLock()
        self._clock = clock
        self._max_series = max(1, int(max_series))
        self._max_samples = max(2, int(max_samples))
        self._counters: dict[str, int] = {}
        self._gauges: dict[str, float] = {}
        self._latencies: dict[str, Deque[float]] = {}
        self._errors: Deque[dict[str, Any]] = deque(maxlen=max(1, int(max_errors)))
        self._error_counts: dict[str, int] = {}
        self._dropped_series = 0
        self._started_at = self._clock()

    # ------------------------------------------------------------ recording

    def counter(self, name: str, delta: int = 1) -> int:
        with self._lock:
            if name not in self._counters and not self._has_room():
                self._dropped_series += 1
                return 0
            self._counters[name] = self._counters.get(name, 0) + int(delta)
            return self._counters[name]

    def gauge(self, name: str, value: float) -> None:
        with self._lock:
            if name not in self._gauges and not self._has_room():
                self._dropped_series += 1
                return
            try:
                self._gauges[name] = float(value)
            except (TypeError, ValueError):
                return

    def observe(self, name: str, value_ms: float) -> None:
        """Record one latency sample in milliseconds (fixed-size history)."""
        try:
            sample = float(value_ms)
        except (TypeError, ValueError):
            return
        with self._lock:
            series = self._latencies.get(name)
            if series is None:
                if not self._has_room():
                    self._dropped_series += 1
                    return
                series = deque(maxlen=self._max_samples)
                self._latencies[name] = series
            series.append(sample)

    def record_error(self, component: str, reason: Any) -> None:
        """Count and remember a redacted error; the ledger is a fixed deque."""
        message = redact_text(reason, limit=200) or "unknown"
        with self._lock:
            self._error_counts[component] = self._error_counts.get(component, 0) + 1
            self._errors.append(
                {
                    "at": _iso(self._clock()),
                    "component": str(component)[:60],
                    "error": message,
                }
            )

    # ----------------------------------------------------------- inspection

    def _has_room(self) -> bool:
        total = len(self._counters) + len(self._gauges) + len(self._latencies)
        return total < self._max_series

    def counter_value(self, name: str) -> int:
        with self._lock:
            return int(self._counters.get(name, 0))

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            counters = dict(self._counters)
            gauges = dict(self._gauges)
            latencies = {name: list(samples) for name, samples in self._latencies.items()}
            errors = list(self._errors)
            error_counts = dict(self._error_counts)
            dropped = self._dropped_series
            started_at = self._started_at
            clock = self._clock()
        return {
            "counter_count": len(counters),
            "series_limit": self._max_series,
            "dropped_series": dropped,
            "uptime_seconds": max(0.0, (clock - started_at).total_seconds()),
            "counters": counters,
            "gauges": gauges,
            "latency_ms": {
                name: _summarize(samples) for name, samples in latencies.items()
            },
            "errors_by_component": error_counts,
            "recent_errors": errors,
        }

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._latencies.clear()
            self._errors.clear()
            self._error_counts.clear()
            self._dropped_series = 0
            self._started_at = self._clock()


def _summarize(samples: list[float]) -> dict[str, float]:
    if not samples:
        return {"count": 0, "last": 0.0, "avg": 0.0, "min": 0.0, "max": 0.0}
    ordered = sorted(samples)
    # Nearest-rank p95 without unbounded history.
    index = min(len(ordered) - 1, max(0, int(round(0.95 * len(ordered))) - 1))
    return {
        "count": float(len(ordered)),
        "last": float(samples[-1]),
        "avg": float(sum(ordered) / len(ordered)),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
        "p95": float(ordered[index]),
    }


#: Process-wide registry. Subsystems record into it defensively (a metrics
#: failure must never break a tick, a speech request or a stream), and the
#: diagnostics endpoint renders it.
DEFAULT_METRICS = MetricsRegistry()
