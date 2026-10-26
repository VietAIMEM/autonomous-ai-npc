"""Best-effort metrics helpers used by the running subsystems.

Every call here is defensive: metrics, and especially a metrics bug, must never
break a Brain tick, a speech request or a stream. The registry is bounded (see
``metrics.MetricsRegistry``), so an unexpected label cannot grow memory.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator, Optional

from .metrics import DEFAULT_METRICS, MetricsRegistry


def set_registry(registry: Optional[MetricsRegistry]) -> None:
    """Swap the process-wide registry (tests); ``None`` restores the default."""
    global _REGISTRY
    _REGISTRY = registry or DEFAULT_METRICS


def get_registry() -> MetricsRegistry:
    return _REGISTRY


def count(name: str, delta: int = 1) -> None:
    try:
        _REGISTRY.counter(name, delta)
    except Exception:  # pragma: no cover - defensive
        pass


def gauge(name: str, value: float) -> None:
    try:
        _REGISTRY.gauge(name, value)
    except Exception:  # pragma: no cover - defensive
        pass


def observe_ms(name: str, value_ms: float) -> None:
    try:
        _REGISTRY.observe(name, value_ms)
    except Exception:  # pragma: no cover - defensive
        pass


def record_error(component: str, reason: object) -> None:
    try:
        _REGISTRY.record_error(component, reason)
    except Exception:  # pragma: no cover - defensive
        pass


@contextmanager
def timer(name: str) -> Iterator[None]:
    """Record one latency sample in milliseconds, success or failure."""
    started = time.monotonic()
    try:
        yield
    finally:
        observe_ms(name, (time.monotonic() - started) * 1000.0)


_REGISTRY: MetricsRegistry = DEFAULT_METRICS


__all__ = [
    "count",
    "gauge",
    "get_registry",
    "observe_ms",
    "record_error",
    "set_registry",
    "timer",
]
