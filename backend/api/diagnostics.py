"""Unified, read-only runtime status surface.

One place assembles what an operator (or Unity) needs during a production run:
per-subsystem health in the shared ``READY``/``DEGRADED``/``UNAVAILABLE``
vocabulary, queue depths, the active skill, recent errors and the bounded
metrics snapshot.

Everything here is derived from existing subsystem probes. Nothing performs a
paid provider call, nothing writes state, and every payload passes through
``redact_structure`` before it leaves the process.
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Mapping, Optional

from ..llm.config import describe_provider
from ..runtime.health import (
    STATUS_DEGRADED,
    STATUS_READY,
    STATUS_UNAVAILABLE,
    build_report,
    component,
    from_legacy,
    overall,
    sqlite_probe,
    worst_of,
)
from ..runtime import instrument
from ..runtime.metrics import DEFAULT_METRICS
from ..runtime.redaction import redact_structure, safe_error

#: A single Brain tick may legitimately spend tens of seconds inside a real
#: LLM call (observed p95 ~24s, max ~30s on the production provider). Only an
#: in-flight tick that outlives every plausible provider call is reported as
#: stuck; routine planning must not read as degraded.
_BRAIN_TICK_STUCK_SECONDS = 90.0


def collect_components(
    service: Any,
    *,
    config: Any,
    lifecycle: Any = None,
    streaming: Any = None,
    chat_source: Any = None,
    motion_generation: Any = None,
    motion_processing: Any = None,
    stream_latch: Any = None,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, dict[str, Any]]:
    """Build the per-subsystem health entries. Never raises."""
    components: dict[str, dict[str, Any]] = {}
    uptime = _uptime_seconds(service, lifecycle, clock)

    components["backend"] = component(
        "backend",
        configured=True,
        reachable=True,
        healthy=True,
        required=True,
        detail={
            "pid": os.getpid(),
            "environment": config.environment,
            "host": config.host,
            "brain_port": config.brain_port,
            "chat_port": config.chat_port,
            "state": getattr(lifecycle, "state", None) or "RUNNING",
            "uptime_seconds": round(uptime, 3),
        },
    )

    components["database"] = _database_component(service, config)
    components["llm"] = _llm_component(service, config)
    components["tts"] = _tts_component(service)
    components["chat"] = _chat_component(service, chat_source)
    components["stream"] = _stream_component(streaming, stream_latch)
    components["unity"] = _unity_component(service, config, clock)
    components["brain"] = _brain_component(service, lifecycle)
    components["memory"] = _memory_component(service)
    components["learning"] = _learning_component(service)
    if motion_generation is not None or motion_processing is not None:
        components["motion"] = _motion_component(motion_generation, motion_processing)
    return components


def build_health(
    service: Any,
    *,
    config: Any,
    lifecycle: Any = None,
    streaming: Any = None,
    chat_source: Any = None,
    motion_generation: Any = None,
    motion_processing: Any = None,
    stream_latch: Any = None,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """The ``/health`` payload: cheap probes only."""
    started = clock()
    components = collect_components(
        service,
        config=config,
        lifecycle=lifecycle,
        streaming=streaming,
        chat_source=chat_source,
        motion_generation=motion_generation,
        motion_processing=motion_processing,
        stream_latch=stream_latch,
        clock=clock,
    )
    report = build_report(
        components,
        environment=config.environment,
        uptime_seconds=_uptime_seconds(service, lifecycle, clock),
        extra={"probe_ms": round((clock() - started) * 1000.0, 3)},
    )
    return redact_structure(report)


def build_status(
    service: Any,
    *,
    config: Any,
    lifecycle: Any = None,
    streaming: Any = None,
    chat_source: Any = None,
    motion_generation: Any = None,
    motion_processing: Any = None,
    stream_latch: Any = None,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """The ``/status`` payload: one compact, operator-readable snapshot."""
    components = collect_components(
        service,
        config=config,
        lifecycle=lifecycle,
        streaming=streaming,
        chat_source=chat_source,
        motion_generation=motion_generation,
        motion_processing=motion_processing,
        stream_latch=stream_latch,
        clock=clock,
    )
    runtime = _runtime_metrics(service)
    payload: dict[str, Any] = {
        # Match ``/health``: components that are optional-by-configuration (an
        # intentionally stopped stream) stay visible in the per-subsystem
        # ``health`` map without dragging the headline verdict down.
        "status": overall(components),
        "environment": config.environment,
        "process": {
            "pid": os.getpid(),
            "host": config.host,
            "brain_port": config.brain_port,
            "chat_port": config.chat_port,
            "uptime_seconds": round(_uptime_seconds(service, lifecycle, clock), 3),
            "state": getattr(lifecycle, "state", None) or "RUNNING",
            "instance_id": getattr(lifecycle, "instance_id", None),
        },
        "health": {
            name: entry.get("status") for name, entry in sorted(components.items())
        },
        "brain": runtime,
        "queues": queue_depths(
            service,
            streaming=streaming,
            chat_source=chat_source,
            motion_generation=motion_generation,
            motion_processing=motion_processing,
        ),
        "active_skill": _active_skill(service),
        "stream_stop_latch": _latch_snapshot(stream_latch),
        "recent_errors": DEFAULT_METRICS.snapshot()["recent_errors"],
    }
    if lifecycle is not None:
        try:
            payload["lifecycle"] = {
                "lock_held": bool(getattr(lifecycle.lock, "held", False)),
                "stop_requested": lifecycle.stop_request_path.exists(),
            }
        except Exception:  # pragma: no cover - defensive
            pass
    return redact_structure(payload)


def build_diagnostics(
    service: Any,
    *,
    config: Any,
    lifecycle: Any = None,
    streaming: Any = None,
    chat_source: Any = None,
    motion_generation: Any = None,
    motion_processing: Any = None,
    stream_latch: Any = None,
    backup: Optional[Mapping[str, Any]] = None,
    retention: Optional[Mapping[str, Any]] = None,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """The ``/diagnostics`` payload: health, sanitized config, metrics, storage."""
    health = build_health(
        service,
        config=config,
        lifecycle=lifecycle,
        streaming=streaming,
        chat_source=chat_source,
        motion_generation=motion_generation,
        motion_processing=motion_processing,
        stream_latch=stream_latch,
        clock=clock,
    )
    status = build_status(
        service,
        config=config,
        lifecycle=lifecycle,
        streaming=streaming,
        chat_source=chat_source,
        motion_generation=motion_generation,
        motion_processing=motion_processing,
        stream_latch=stream_latch,
        clock=clock,
    )
    payload: dict[str, Any] = {
        "status": status["status"],
        "environment": config.environment,
        "health": health,
        "runtime": status,
        "config": config.to_dict(),
        "providers": {
            "llm": describe_provider(getattr(getattr(service, "brain", None), "provider", None)),
            "tts": _tts_provider_description(service),
            "chat": _chat_provider_description(chat_source),
            "stream": _stream_provider_description(streaming),
        },
        "metrics": DEFAULT_METRICS.snapshot(),
        "storage": {
            "backup": dict(backup) if backup else None,
            "retention": dict(retention) if retention else None,
        },
        "persistence": _persistence_summary(service),
    }
    return redact_structure(payload)


def queue_depths(
    service: Any,
    *,
    streaming: Any = None,
    chat_source: Any = None,
    motion_generation: Any = None,
    motion_processing: Any = None,
) -> dict[str, Any]:
    """Bounded queue depths for the unified status surface.

    Reading a queue depth is also the cheapest moment to refresh the matching
    metric gauge, so the bounded metrics snapshot tracks load without a timer
    thread of its own.
    """
    queues: dict[str, Any] = {}
    chat = getattr(service, "chat_service", None)
    if chat is not None:
        queues["chat"] = _safe_dict(chat.get_state, keys=("queued_count", "queue_capacity"))
    speech = getattr(service, "speech_service", None)
    if speech is not None:
        queues["speech"] = _safe_dict(
            speech.summary, keys=("queue_length", "queue_capacity")
        )
    if chat_source is not None:
        state = _safe_dict(chat_source.get_state, keys=("state", "reconnect_count"))
        queues["chat_source"] = state
    if streaming is not None:
        state = _safe_dict(streaming.get_state, keys=("state", "queues", "transport_enabled"))
        queues["stream"] = state
    for name, handle in (
        ("motion_generation", motion_generation),
        ("motion_processing", motion_processing),
    ):
        if handle is None:
            continue
        queues[name] = _safe_dict(
            handle.get_state, keys=("state", "queue_depth", "queue_capacity")
        )
    _publish_queue_gauges(queues)
    return queues


def _publish_queue_gauges(queues: Mapping[str, Any]) -> None:
    for name, payload in queues.items():
        if not isinstance(payload, Mapping):
            continue
        depth = payload.get("queue_length", payload.get("queued_count"))
        if depth is None:
            depth = payload.get("queue_depth")
        if isinstance(depth, (int, float)) and not isinstance(depth, bool):
            instrument.gauge(f"queue.{name}.depth", float(depth))
        capacity = payload.get("queue_capacity")
        if isinstance(capacity, (int, float)) and not isinstance(capacity, bool):
            instrument.gauge(f"queue.{name}.capacity", float(capacity))
        sample = payload.get("queues")
        if isinstance(sample, Mapping):
            for kind, value in sample.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    instrument.gauge(f"queue.{name}.{kind}", float(value))


def _safe_dict(call: Callable[[], Any], *, keys: tuple[str, ...]) -> dict[str, Any]:
    try:
        payload = call()
    except Exception as exc:
        return {"error": safe_error(exc, limit=120)}
    if not isinstance(payload, Mapping):
        return {"error": "unexpected_probe_payload"}
    return {key: payload.get(key) for key in keys if key in payload}


def _runtime_metrics(service: Any) -> dict[str, Any]:
    getter = getattr(service, "runtime_metrics", None)
    if callable(getter):
        try:
            payload = getter()
            if isinstance(payload, Mapping):
                return dict(payload)
        except Exception as exc:
            return {"error": safe_error(exc, limit=120)}
    return {"enabled": False}


def _active_skill(service: Any) -> Optional[str]:
    brain = getattr(service, "brain", None)
    getter = getattr(brain, "current_skill_id", None)
    if not callable(getter):
        return None
    try:
        return getter()
    except Exception:  # pragma: no cover - defensive
        return None


def _latch_snapshot(latch: Any) -> Optional[dict[str, Any]]:
    """Read the deliberate-stop latch without ever raising into a probe."""
    if latch is None:
        return None
    if isinstance(latch, Mapping):
        return dict(latch)
    getter = getattr(latch, "snapshot", None)
    if callable(getter):
        try:
            payload = getter()
            if isinstance(payload, Mapping):
                return dict(payload)
        except Exception as exc:  # pragma: no cover - defensive
            return {"error": safe_error(exc, limit=120)}
    stopped = getattr(latch, "stopped", None)
    return {"stopped": bool(stopped)} if isinstance(stopped, bool) else None


def _uptime_seconds(service: Any, lifecycle: Any, clock: Callable[[], float]) -> float:
    getter = getattr(service, "uptime_seconds", None)
    if callable(getter):
        try:
            return float(getter())
        except Exception:  # pragma: no cover - defensive
            pass
    started = getattr(lifecycle, "started_at", None)
    if started is not None:
        try:
            now = clock()
            origin = started.timestamp() if hasattr(started, "timestamp") else started
            if origin > 1_000_000_000:  # wall clock, not a monotonic counter
                return max(0.0, time.time() - float(origin))
            return max(0.0, now - float(origin))
        except Exception:  # pragma: no cover - defensive
            pass
    return 0.0


def _database_component(service: Any, config: Any) -> dict[str, Any]:
    probes: dict[str, Any] = {}
    statuses: list[str] = []
    required_any = False
    for label, path in _database_paths(service, config):
        probe = sqlite_probe(path)
        probes[label] = probe
        statuses.append(str(probe.get("status")))
        required_any = True
    if not probes:
        return component(
            "database",
            configured=False,
            status=STATUS_UNAVAILABLE,
            required=False,
            intentional=True,
            reason="not_configured",
        )
    return component(
        "database",
        configured=True,
        status=worst_of(statuses),
        required=required_any,
        detail=probes,
    )


def _database_paths(service: Any, config: Any) -> list[tuple[str, Any]]:
    paths: list[tuple[str, Any]] = []
    memory = getattr(service, "memory_service", None)
    repository = getattr(memory, "repository", None)
    memory_path = getattr(repository, "database_path", None)
    if memory_path:
        paths.append(("memory", memory_path))
    learning = getattr(getattr(service, "brain", None), "learning", None)
    learning_repo = getattr(learning, "repository", None)
    learning_path = getattr(learning_repo, "database_path", None)
    if learning_path and all(str(learning_path) != str(item) for _name, item in paths):
        paths.append(("learning", learning_path))
    if not paths:
        fallback = getattr(config, "data_dir", None)
        if fallback is not None:
            for name in ("memory.db", "learning.db"):
                candidate = fallback / name
                if candidate.exists():
                    paths.append((name.replace(".db", ""), candidate))
    return paths


def _llm_component(service: Any, config: Any) -> dict[str, Any]:
    provider = getattr(getattr(service, "brain", None), "provider", None)
    described = describe_provider(provider)
    if described["kind"] == "disabled":
        return component(
            "llm",
            configured=False,
            status=STATUS_UNAVAILABLE,
            required=False,
            intentional=True,
            reason="no_provider",
            detail=described,
        )
    if described["mock"]:
        allowed = bool(getattr(config, "mocks_allowed", True))
        status = STATUS_DEGRADED if allowed else STATUS_UNAVAILABLE
        return component(
            "llm",
            configured=True,
            reachable=True,
            healthy=False,
            status=status,
            required=not allowed,
            reason="mock_provider",
            detail=described,
        )
    if not described["configured"]:
        return component(
            "llm",
            configured=False,
            reachable=False,
            status=STATUS_UNAVAILABLE,
            required=True,
            reason="api_key_unavailable",
            detail=described,
        )
    return component(
        "llm",
        configured=True,
        reachable=None,
        healthy=None,
        status=STATUS_READY,
        required=True,
        reason="configured_skip_paid_probe",
        detail=described,
    )


def _tts_component(service: Any) -> dict[str, Any]:
    health = None
    getter = getattr(service, "get_speech_health", None)
    if callable(getter):
        try:
            health = getter()
        except Exception as exc:
            return component(
                "tts",
                configured=True,
                reachable=False,
                status=STATUS_UNAVAILABLE,
                required=False,
                reason=safe_error(exc, limit=120),
            )
    if not isinstance(health, Mapping):
        return component(
            "tts",
            configured=False,
            status=STATUS_UNAVAILABLE,
            required=False,
            intentional=True,
            reason="not_configured",
        )
    enabled = bool(health.get("enabled", health.get("status") != "disabled"))
    if not enabled:
        return component(
            "tts",
            configured=False,
            status=STATUS_UNAVAILABLE,
            required=False,
            intentional=True,
            reason="disabled_by_configuration",
            detail=health,
        )
    mapped = from_legacy("tts", health, enabled_key="enabled", required=False)
    return mapped


def _chat_component(service: Any, chat_source: Any) -> dict[str, Any]:
    if chat_source is None:
        return component(
            "chat",
            configured=False,
            status=STATUS_UNAVAILABLE,
            required=False,
            intentional=True,
            reason="no_real_chat_input",
            detail=_safe_dict(
                getattr(service, "chat_service").get_state, keys=("queued_count",)
            )
            if getattr(service, "chat_service", None) is not None
            else {},
        )
    payload = None
    try:
        payload = chat_source.health()
    except Exception as exc:
        return component(
            "chat",
            configured=True,
            reachable=False,
            status=STATUS_UNAVAILABLE,
            reason=safe_error(exc, limit=120),
        )
    return from_legacy("chat", payload, enabled_key="enabled", required=True)


def _stream_component(streaming: Any, stream_latch: Any = None) -> dict[str, Any]:
    latch = _latch_snapshot(stream_latch)
    if streaming is None:
        return component(
            "stream",
            configured=False,
            status=STATUS_UNAVAILABLE,
            required=False,
            intentional=True,
            reason="transport_disabled",
            detail={"stop_latch": latch} if latch is not None else None,
        )
    try:
        payload = streaming.health()
    except Exception as exc:
        return component(
            "stream",
            configured=True,
            reachable=False,
            status=STATUS_UNAVAILABLE,
            reason=safe_error(exc, limit=120),
        )
    if not isinstance(payload, Mapping):
        return component(
            "stream",
            configured=True,
            status=STATUS_UNAVAILABLE,
            reason="unexpected_probe_payload",
        )
    state = str(payload.get("state", "")).upper()
    # A durable operator stop is intent even after a restart, when the
    # transport sits IDLE until something explicitly starts it again; the
    # stop-latch, not the transient transport word, is the source of truth
    # for "the stream is off on purpose".
    latch_stopped = bool(isinstance(latch, Mapping) and latch.get("stopped"))
    intentional = state.startswith("STOPPED") or latch_stopped
    if latch is not None and (latch.get("load_error") or latch.get("write_failures")):
        # The stream itself may be fine, but the deliberate-stop latch is no
        # longer durable; an autonomous restart after the next boot would be
        # unsafe, so the subsystem reports degraded instead of ready.
        return component(
            "stream",
            configured=True,
            reachable=True,
            healthy=False,
            required=not intentional,
            reason="stop_latch_persistence_fault",
            detail={"probe": dict(payload), "stop_latch": latch},
        )
    mapped_entry = from_legacy(
        "stream",
        payload,
        enabled_key="transport_enabled",
        required=not intentional,
    )
    if intentional:
        # The operator's intent must be visible on the entry itself, exactly
        # as it is for a fresh stop request. A stale media error from an
        # earlier start attempt (e.g. the last NO_MEDIA/TIMEOUT) must not read
        # as the current reason for a stream that is off on purpose.
        mapped_entry["intentional"] = True
        prior_reason = mapped_entry.pop("reason", None)
        if isinstance(prior_reason, str) and prior_reason:
            detail = dict(mapped_entry.get("detail") or {})
            detail.setdefault("last_error", prior_reason)
            mapped_entry["detail"] = detail
        mapped_entry["reason"] = "intentional_stop"
    if latch_stopped and state in {"", "IDLE", "STOPPED"}:
        # A durable stop reads the same whether the transport word is STOPPED
        # or the IDLE it boots into after a restart; DEGRADED would tell the
        # Admin Dashboard that a deliberate shutdown is an impairment.
        mapped_entry["status"] = STATUS_UNAVAILABLE
        mapped_entry["reachable"] = False
        mapped_entry["healthy"] = False
    if latch is not None:
        detail = dict(mapped_entry.get("detail") or {})
        detail["stop_latch"] = latch
        mapped_entry["detail"] = detail
    return mapped_entry


def _unity_component(service: Any, config: Any, clock: Callable[[], float]) -> dict[str, Any]:
    getter = getattr(service, "last_tick_age_seconds", None)
    age: Optional[float] = None
    if callable(getter):
        try:
            age = getter()
        except Exception:  # pragma: no cover - defensive
            age = None
    # Unity's tick loop is synchronous: it only sends the next snapshot after
    # the previous planner tick returns. Real production latency was measured
    # at p95 ~26-31 s plus the 5 s decision interval (max observed 34.8 s), so
    # the old 30 s threshold flapped a genuinely ticking Unity to DEGRADED.
    stale_after = float(getattr(config, "unity_tick_stale_seconds", 60.0))
    required = bool(getattr(config, "is_production", False))
    if age is None:
        return component(
            "unity",
            configured=False,
            reachable=False,
            status=STATUS_UNAVAILABLE,
            required=required,
            intentional=not required,
            reason="no_tick_received",
            detail={"stale_after_seconds": stale_after},
        )
    healthy = age <= stale_after
    return component(
        "unity",
        configured=True,
        reachable=True,
        healthy=healthy,
        required=required,
        reason=None if healthy else "tick_stale",
        detail={
            "last_tick_age_seconds": round(age, 3),
            "stale_after_seconds": stale_after,
        },
    )


def _brain_component(service: Any, lifecycle: Any) -> dict[str, Any]:
    runtime = _runtime_metrics(service)
    state = str(getattr(lifecycle, "state", "") or "").upper()
    if state == "STOPPING":
        return component(
            "brain",
            configured=True,
            reachable=True,
            healthy=False,
            reason="shutting_down",
            detail=runtime,
        )
    tick_age = runtime.get("last_tick_age_seconds")
    if (
        runtime.get("tick_in_progress")
        and isinstance(tick_age, (int, float))
        and not isinstance(tick_age, bool)
        and float(tick_age) > _BRAIN_TICK_STUCK_SECONDS
    ):
        # A tick in flight is the normal planning state (the single loop
        # rejects a second concurrent tick), so only a tick that has outlived
        # every plausible provider call is reported as stuck.
        return component(
            "brain",
            configured=True,
            reachable=True,
            healthy=False,
            reason="tick_stuck",
            detail={**runtime, "stuck_after_seconds": _BRAIN_TICK_STUCK_SECONDS},
        )
    return component(
        "brain",
        configured=True,
        reachable=True,
        healthy=True,
        detail=runtime,
    )


def _memory_component(service: Any) -> dict[str, Any]:
    state = None
    getter = getattr(service, "get_memory_state", None)
    if callable(getter):
        try:
            state = getter()
        except Exception as exc:
            return component(
                "memory",
                configured=True,
                reachable=False,
                status=STATUS_UNAVAILABLE,
                reason=safe_error(exc, limit=120),
            )
    if not isinstance(state, Mapping):
        # Fall back to the memory service itself so degraded writes still
        # surface when the aggregate accessor is not part of this assembly.
        memory = getattr(service, "memory_service", None)
        fallback = getattr(memory, "get_state", None)
        if callable(fallback):
            try:
                state = fallback()
            except Exception:  # pragma: no cover - defensive
                state = None
    if not isinstance(state, Mapping) or state.get("enabled") is False:
        return component(
            "memory",
            configured=False,
            status=STATUS_UNAVAILABLE,
            required=False,
            intentional=True,
            reason="disabled",
        )
    degraded = bool(state.get("degraded"))
    return component(
        "memory",
        configured=True,
        reachable=True,
        healthy=not degraded,
        required=not degraded,
        reason="degraded_writes" if degraded else None,
        detail={
            "counts": state.get("counts"),
            "last_error": state.get("last_error"),
            "short_term": state.get("short_term"),
        },
    )


def _learning_component(service: Any) -> dict[str, Any]:
    state = None
    getter = getattr(service, "get_learning_state", None)
    if callable(getter):
        try:
            state = getter()
        except Exception as exc:
            return component(
                "learning",
                configured=True,
                reachable=False,
                status=STATUS_UNAVAILABLE,
                required=False,
                reason=safe_error(exc, limit=120),
            )
    if not isinstance(state, Mapping) or state.get("enabled") is False:
        return component(
            "learning",
            configured=False,
            status=STATUS_UNAVAILABLE,
            required=False,
            intentional=True,
            reason="disabled",
        )
    return component(
        "learning",
        configured=True,
        reachable=True,
        healthy=True,
        required=False,
        detail={
            "contexts": state.get("contexts"),
            "knowledge": state.get("knowledge"),
            "compositions": state.get("compositions"),
            "motions": state.get("motions"),
        },
    )


def _motion_component(
    motion_generation: Any, motion_processing: Any
) -> dict[str, Any]:
    detail: dict[str, Any] = {}
    statuses: list[str] = []
    for label, handle in (
        ("generation", motion_generation),
        ("processing", motion_processing),
    ):
        if handle is None:
            continue
        try:
            state = handle.get_state()
        except Exception as exc:
            detail[label] = {"error": safe_error(exc, limit=120)}
            statuses.append(STATUS_UNAVAILABLE)
            continue
        if not isinstance(state, Mapping):
            detail[label] = {"error": "unexpected_probe_payload"}
            statuses.append(STATUS_UNAVAILABLE)
            continue
        failures = 0
        counters = state.get("counters")
        if isinstance(counters, Mapping):
            failures = int(counters.get("failed", 0) or 0)
        detail[label] = {
            "state": state.get("state"),
            "queue_depth": state.get("queue_depth"),
            "queue_capacity": state.get("queue_capacity"),
            "counters": counters,
            "last_error": state.get("last_error"),
        }
        statuses.append(STATUS_DEGRADED if failures else STATUS_READY)
    if not statuses:
        return component(
            "motion",
            configured=False,
            status=STATUS_UNAVAILABLE,
            required=False,
            intentional=True,
            reason="disabled",
        )
    return component(
        "motion",
        configured=True,
        status=worst_of(statuses),
        required=False,
        detail=detail,
    )


def _persistence_summary(service: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    memory = getattr(service, "memory_service", None)
    if memory is not None:
        try:
            state = memory.get_state()
            summary["memory"] = {
                "counts": state.get("counts"),
                "personality": bool(memory.personality_diagnostics()),
            }
        except Exception as exc:
            summary["memory"] = {"error": safe_error(exc, limit=120)}
    learning = getattr(getattr(service, "brain", None), "learning", None)
    if learning is not None:
        try:
            summary["learning"] = learning.get_state()
        except Exception as exc:
            summary["learning"] = {"error": safe_error(exc, limit=120)}
    return summary


def _tts_provider_description(service: Any) -> dict[str, Any]:
    speech = getattr(service, "speech_service", None)
    if speech is None:
        return {"provider": "disabled", "configured": False}
    described: dict[str, Any] = {"provider": getattr(speech, "provider_name", "unknown")}
    config = getattr(speech, "tts_config", None)
    to_dict = getattr(config, "to_dict", None)
    if callable(to_dict):
        try:
            described.update(to_dict())
        except Exception:  # pragma: no cover - defensive
            pass
    return described


def _chat_provider_description(chat_source: Any) -> dict[str, Any]:
    if chat_source is None:
        return {"provider": "disabled", "configured": False}
    state = _safe_dict(chat_source.get_state, keys=("provider", "platform", "channels"))
    state["configured"] = True
    return state


def _stream_provider_description(streaming: Any) -> dict[str, Any]:
    if streaming is None:
        return {"provider": "disabled", "configured": False}
    state = _safe_dict(streaming.get_state, keys=("provider", "platform", "real_transport"))
    state["configured"] = True
    return state


__all__ = [
    "build_diagnostics",
    "build_health",
    "build_status",
    "collect_components",
    "queue_depths",
]
