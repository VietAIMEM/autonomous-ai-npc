"""Unified, cheap health/status assembly.

The subsystems keep their own ``health()`` probes. This module only maps them
onto one vocabulary so Unity, an operator script and a test all read the same
thing:

``READY``       configured, reachable and healthy
``DEGRADED``    reachable but impaired, or a deliberately disabled optional part
``UNAVAILABLE`` not configured, not reachable, or actively failing

Three different questions are kept apart on purpose: *configured* (did the
operator set this up), *reachable* (can we talk to it now) and *healthy* (is it
doing useful work). A provider with no key is ``configured=False`` and
``UNAVAILABLE``; it is never reported as a success.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from .redaction import safe_error


STATUS_READY = "READY"
STATUS_DEGRADED = "DEGRADED"
STATUS_UNAVAILABLE = "UNAVAILABLE"

VALID_STATUSES = (STATUS_READY, STATUS_DEGRADED, STATUS_UNAVAILABLE)
_SEVERITY = {STATUS_READY: 0, STATUS_DEGRADED: 1, STATUS_UNAVAILABLE: 2}

#: Legacy words mapped onto the unified vocabulary. ``ok``/``healthy`` mean
#: ready, ``disabled`` means unavailable-by-configuration (never "fine").
_LEGACY_STATUS = {
    "ok": STATUS_READY,
    "ready": STATUS_READY,
    "healthy": STATUS_READY,
    "up": STATUS_READY,
    "live": STATUS_READY,
    "running": STATUS_READY,
    "connected": STATUS_READY,
    "degraded": STATUS_DEGRADED,
    "warning": STATUS_DEGRADED,
    "connecting": STATUS_DEGRADED,
    "reconnecting": STATUS_DEGRADED,
    "starting": STATUS_DEGRADED,
    "stopping": STATUS_DEGRADED,
    "disabled": STATUS_UNAVAILABLE,
    "disconnected": STATUS_UNAVAILABLE,
    "unavailable": STATUS_UNAVAILABLE,
    "error": STATUS_UNAVAILABLE,
    "failed": STATUS_UNAVAILABLE,
    "down": STATUS_UNAVAILABLE,
    "stopped": STATUS_UNAVAILABLE,
}


def normalize_status(raw: Any) -> Optional[str]:
    """Map a subsystem word onto the unified vocabulary (or ``None``)."""
    if not isinstance(raw, str):
        return None
    return _LEGACY_STATUS.get(raw.strip().lower())


def _normalize_status_token(raw: Any) -> Optional[str]:
    """Normalize one producer word, including ``STREAM_*`` vocabulary."""
    mapped = normalize_status(raw)
    if mapped is None and isinstance(raw, str):
        mapped = normalize_status(raw.replace("STREAM", "").lower())
    return mapped


def classify(
    *,
    configured: bool,
    reachable: Optional[bool] = None,
    healthy: Optional[bool] = None,
    intentional: bool = False,
    required: bool = True,
) -> str:
    """Derive one status word from the three separate questions."""
    if not configured:
        # An explicitly disabled optional dependency is still unavailable, but
        # callers can see ``intentional``/``required`` and keep it out of the
        # overall verdict.
        return STATUS_UNAVAILABLE
    if reachable is False:
        return STATUS_UNAVAILABLE
    if healthy is False:
        return STATUS_DEGRADED
    if reachable is None and healthy is None:
        return STATUS_DEGRADED
    return STATUS_READY


def component(
    name: str,
    *,
    configured: bool,
    reachable: Optional[bool] = None,
    healthy: Optional[bool] = None,
    status: Optional[str] = None,
    required: bool = True,
    intentional: bool = False,
    detail: Mapping[str, Any] | None = None,
    reason: Optional[str] = None,
) -> dict[str, Any]:
    """Build one subsystem entry for the unified report."""
    resolved = status or classify(
        configured=configured,
        reachable=reachable,
        healthy=healthy,
        intentional=intentional,
        required=required,
    )
    payload: dict[str, Any] = {
        "status": resolved,
        "configured": bool(configured),
        "reachable": reachable,
        "healthy": healthy,
        "required": bool(required),
        "intentional": bool(intentional),
    }
    if reason:
        payload["reason"] = reason
    if detail:
        payload["detail"] = dict(detail)
    return payload


def from_legacy(
    name: str,
    payload: Mapping[str, Any] | None,
    *,
    enabled_key: str = "enabled",
    status_key: str = "status",
    required: bool = True,
    default_configured: Optional[bool] = None,
) -> dict[str, Any]:
    """Adapt an existing subsystem ``health()`` dict without rewriting it."""
    if payload is None:
        configured = bool(default_configured)
        return component(
            name,
            configured=configured,
            status=STATUS_UNAVAILABLE if not configured else None,
            required=required,
            reason="no_health_payload",
        )
    data = dict(payload)
    enabled = data.get(enabled_key)
    if default_configured is None:
        configured = True if enabled is None else bool(enabled)
    else:
        configured = bool(default_configured if enabled is None else enabled)
    raw_status = data.get(status_key)
    mapped = _normalize_status_token(raw_status)
    if mapped is None:
        # Several real probes report a lifecycle word instead of ``status``:
        # the Twitch chat source sends ``CONNECTED``/``RETRYING`` and the
        # stream transport sends ``LIVE``/``IDLE``/``STOPPED``.
        mapped = _normalize_status_token(data.get("state"))
    if mapped is None and isinstance(data.get("ok"), bool):
        # Producers that only publish a boolean verdict (``ok``): true means
        # the subsystem is doing its job, false means reachable-but-impaired.
        mapped = STATUS_READY if data["ok"] else STATUS_DEGRADED
    if mapped == STATUS_READY and data.get("degraded") is True:
        # A connected-but-impaired source must not read as ready.
        mapped = STATUS_DEGRADED
    if not configured:
        # An explicit ``enabled: false`` is unavailable by configuration; it
        # is never reported as a success, whatever other fields claim.
        mapped = STATUS_UNAVAILABLE
    reachable = data.get("reachable")
    if not isinstance(reachable, bool):
        reachable = None if mapped is None else mapped != STATUS_UNAVAILABLE
    healthy = data.get("healthy")
    if not isinstance(healthy, bool):
        healthy = None if mapped is None else mapped == STATUS_READY
    last_error = data.get("last_error")
    reason = data.get("reason") or last_error
    if mapped == STATUS_READY:
        # A stale historical error (e.g. the stream's old ``NO_MEDIA``) must
        # not be reported as the current reason for a healthy subsystem.
        reason = None
    detail = {
        key: value
        for key, value in data.items()
        if key not in {"status", "enabled", "reachable", "healthy", "reason", "last_error"}
    }
    if isinstance(last_error, str) and last_error.strip() and reason != last_error:
        # Keep the retained error visible without letting it masquerade as the
        # current status reason.
        detail["last_error"] = last_error
    return component(
        name,
        configured=configured,
        reachable=reachable,
        healthy=healthy,
        status=mapped,
        required=required,
        intentional=bool(data.get("intentional")),
        detail=detail,
        reason=reason if isinstance(reason, str) else None,
    )


def overall(components: Mapping[str, Mapping[str, Any]]) -> str:
    """Worst status among the components that must work."""
    verdict = STATUS_READY
    for entry in components.values():
        if not entry.get("required", True):
            continue
        status = str(entry.get("status") or STATUS_UNAVAILABLE)
        if _SEVERITY.get(status, 2) > _SEVERITY[verdict]:
            verdict = status
    return verdict


def build_report(
    components: Mapping[str, Mapping[str, Any]],
    *,
    environment: str,
    uptime_seconds: float,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the unified health payload."""
    verdict = overall(components)
    counts = {STATUS_READY: 0, STATUS_DEGRADED: 0, STATUS_UNAVAILABLE: 0}
    issues: list[dict[str, str]] = []
    for name, entry in components.items():
        status = str(entry.get("status") or STATUS_UNAVAILABLE)
        counts[status] = counts.get(status, 0) + 1
        if status != STATUS_READY:
            issue = {"component": name, "status": status}
            reason = entry.get("reason")
            if isinstance(reason, str) and reason:
                issue["reason"] = reason
            issues.append(issue)
    payload: dict[str, Any] = {
        "status": verdict,
        "ready": verdict == STATUS_READY,
        "environment": environment,
        "uptime_seconds": round(max(0.0, float(uptime_seconds)), 3),
        "counts": counts,
        "components": {name: dict(entry) for name, entry in components.items()},
        "issues": issues,
    }
    if extra:
        payload.update(dict(extra))
    return payload


def sqlite_probe(path: Path | str, *, quick_check: bool = False) -> dict[str, Any]:
    """Cheap read-only probe of one SQLite file.

    ``quick_check`` is opt-in because it reads the whole file; the default
    probe only opens the database and reads ``schema_version``.
    """
    resolved = Path(path)
    detail: dict[str, Any] = {"path": resolved.name}
    if not resolved.exists():
        return component(
            resolved.name,
            configured=False,
            status=STATUS_UNAVAILABLE,
            reason="database_missing",
            detail=detail,
        )
    try:
        detail["size_bytes"] = resolved.stat().st_size
        detail["wal_bytes"] = _size_or_zero(f"{resolved}-wal")
        detail["shm_bytes"] = _size_or_zero(f"{resolved}-shm")
    except OSError as exc:
        return component(
            resolved.name,
            configured=True,
            reachable=False,
            status=STATUS_UNAVAILABLE,
            reason=safe_error(exc, limit=120),
            detail=detail,
        )
    try:
        connection = sqlite3.connect(
            f"file:{resolved.as_posix()}?mode=ro",
            uri=True,
            timeout=2.0,
        )
        try:
            version = connection.execute("PRAGMA schema_version").fetchone()
            detail["schema_version"] = int(version[0]) if version else 0
            if quick_check:
                result = connection.execute("PRAGMA quick_check").fetchone()
                verdict = str(result[0]).lower() if result else "unknown"
                detail["quick_check"] = verdict
                if verdict != "ok":
                    return component(
                        resolved.name,
                        configured=True,
                        reachable=True,
                        healthy=False,
                        reason="integrity_check_failed",
                        detail=detail,
                    )
        finally:
            connection.close()
    except sqlite3.Error as exc:
        return component(
            resolved.name,
            configured=True,
            reachable=False,
            status=STATUS_UNAVAILABLE,
            reason=safe_error(exc, limit=120),
            detail=detail,
        )
    return component(
        resolved.name,
        configured=True,
        reachable=True,
        healthy=True,
        detail=detail,
    )


def worst_of(statuses: Iterable[str]) -> str:
    verdict = STATUS_READY
    for status in statuses:
        if _SEVERITY.get(status, 2) > _SEVERITY[verdict]:
            verdict = status
    return verdict


def _size_or_zero(path: str) -> int:
    try:
        return Path(path).stat().st_size
    except OSError:
        return 0


__all__ = [
    "STATUS_DEGRADED",
    "STATUS_READY",
    "STATUS_UNAVAILABLE",
    "VALID_STATUSES",
    "build_report",
    "classify",
    "component",
    "from_legacy",
    "normalize_status",
    "overall",
    "sqlite_probe",
    "worst_of",
]
