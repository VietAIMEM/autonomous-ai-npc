"""PHASE 20 production-hardening probe.

This is a read-only production validation driver. It exercises the same
health, diagnostics, metrics, backup, restore, retention, lifecycle and
configuration surfaces used by the running backend. Destructive operations are
performed only against isolated temporary state.

The probe never starts, stops or restarts the production backend. If a live
instance is already running, it reads its existing HTTP status endpoints. If
there is no live instance, the API checks run against a real in-process object
graph backed by temporary databases and report the live probe as BLOCKED.

Usage:
    python -m backend.tools.phase20_production_probe

The primary output is JSON. A copy is written to
``Temp/Phase20Production.json`` unless ``--output`` selects another path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from backend.api.brain import create_default_service
from backend.api.diagnostics import build_diagnostics, build_health, build_status
from backend.memory.models import MemoryConfig
from backend.memory.service import MemoryService
from backend.memory.sqlite_repository import SQLiteMemoryRepository
from backend.runtime.backup import (
    BackupSource,
    backup_status,
    create_backup,
    integrity_check,
    restore_backup,
    verify_backup,
)
from backend.runtime.config import ConfigurationError, RuntimeConfig
from backend.runtime.health import sqlite_probe
from backend.runtime.lifecycle import RoomAgentController, RuntimeLifecycle
from backend.runtime.metrics import DEFAULT_METRICS
from backend.runtime.redaction import redact_structure, safe_error
from backend.runtime.retention import (
    RetentionError,
    apply_retention,
    guard_directory,
    log_usage,
)
from backend.speech.service import SpeechService
from backend.tts.config import PROVIDER_MOCK, TTSConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = PROJECT_ROOT / "Temp" / "Phase20Production.json"

VALID_STATUSES = {"PASS", "DEGRADED", "BLOCKED", "FAIL"}
HEALTH_STATUSES = {"READY", "DEGRADED", "UNAVAILABLE"}
DATABASE_NAMES = (
    "memory.db",
    "learning.db",
    "motion_import.db",
    "motion_processing.db",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _create_sqlite_fixture(path: Path, marker: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    try:
        connection.execute("CREATE TABLE probe_marker (value TEXT NOT NULL)")
        connection.execute("INSERT INTO probe_marker (value) VALUES (?)", (marker,))
        connection.commit()
    finally:
        connection.close()


def _read_sqlite_marker(path: Path) -> str | None:
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        row = connection.execute(
            "SELECT value FROM probe_marker LIMIT 1"
        ).fetchone()
    finally:
        connection.close()
    return str(row[0]) if row else None


def _configured_secrets() -> tuple[str, ...]:
    markers = ("API_KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
    values: list[str] = []
    for name, value in os.environ.items():
        if not value or len(value) < 8:
            continue
        if any(marker in name.upper() for marker in markers):
            values.append(value)
    return tuple(sorted(set(values)))


def _scrub_text(value: str, secrets: tuple[str, ...]) -> str:
    result = value
    for secret in secrets:
        if secret:
            result = result.replace(secret, "[REDACTED]")
    return result


def _contains_secret(value: Any, secrets: tuple[str, ...]) -> bool:
    if not secrets:
        return False
    encoded = json.dumps(value, ensure_ascii=False, default=str)
    return any(secret in encoded for secret in secrets)


def _isolated_config(root: Path, name: str, **overrides: Any) -> RuntimeConfig:
    base = root / name
    values: dict[str, Any] = {
        "environment": "test",
        "host": "127.0.0.1",
        "brain_port": 18421,
        "chat_port": 18422,
        "data_dir": base / "data",
        "log_dir": base / "logs",
        "temp_dir": base / "temp",
        "backup_dir": base / "backups",
        "state_dir": base / "state",
        "allow_mock_providers": True,
    }
    values.update(overrides)
    config = RuntimeConfig(**values)
    for path in (
        config.data_dir,
        config.log_dir,
        config.temp_dir,
        config.backup_dir,
        config.state_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)
    return config


def _database_fingerprint(config: RuntimeConfig) -> dict[str, dict[str, Any]]:
    fingerprint: dict[str, dict[str, Any]] = {}
    for name in DATABASE_NAMES:
        path = Path(config.data_dir) / name
        if not path.is_file():
            continue
        try:
            stat = path.stat()
            probe = sqlite_probe(path, quick_check=False)
            fingerprint[name] = {
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "status": probe.get("status"),
                "schema_version": (probe.get("detail") or {}).get("schema_version"),
            }
        except OSError as exc:
            fingerprint[name] = {"error": safe_error(exc, limit=120)}
    return fingerprint


def _http_get_json(url: str, timeout: float = 3.0) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if isinstance(payload, dict):
            payload["reachable"] = True
            return payload
        return {"reachable": True, "payload_type": type(payload).__name__}
    except urllib.error.HTTPError as exc:
        return {"reachable": True, "http_status": exc.code}
    except Exception as exc:
        return {"reachable": False, "error": safe_error(exc, limit=160)}


@dataclass(frozen=True)
class IsolatedSurfaces:
    config: RuntimeConfig
    health: dict[str, Any]
    status: dict[str, Any]
    diagnostics: dict[str, Any]
    usage: dict[str, Any]
    backup: dict[str, Any]


@contextmanager
def _isolated_service(
    config: RuntimeConfig,
) -> Iterator[tuple[Any, RuntimeLifecycle, MemoryService, SpeechService]]:
    memory_path = Path(config.data_dir) / "memory.db"
    memory = MemoryService(
        SQLiteMemoryRepository(memory_path),
        MemoryConfig(database_path=memory_path),
    )
    speech = SpeechService(
        tts_config=TTSConfig(provider=PROVIDER_MOCK),
        start_worker=False,
    )
    service = create_default_service(
        memory_service=memory,
        database_path=Path(config.data_dir) / "learning.db",
        speech_service=speech,
        runtime_config=config,
    )
    lifecycle = RuntimeLifecycle(config, role="phase20-probe")
    try:
        yield service, lifecycle, memory, speech
    finally:
        for close in (
            getattr(getattr(service, "brain", None), "learning", None),
            memory,
            speech,
        ):
            callback = getattr(close, "close", None) or getattr(
                close, "shutdown", None
            )
            if callback is None:
                continue
            try:
                callback()
            except Exception:
                pass


def _collect_isolated_surfaces(ctx: "ProbeContext") -> IsolatedSurfaces:
    config = _isolated_config(ctx.temp_root, "runtime_api")
    with _isolated_service(config) as (service, lifecycle, _memory, _speech):
        health = build_health(service, config=config, lifecycle=lifecycle)
        status = build_status(service, config=config, lifecycle=lifecycle)
        backup = backup_status(config)
        usage = log_usage(config)
        diagnostics = build_diagnostics(
            service,
            config=config,
            lifecycle=lifecycle,
            backup=backup,
            retention=usage,
        )
    return IsolatedSurfaces(
        config=config,
        health=health,
        status=status,
        diagnostics=diagnostics,
        usage=usage,
        backup=backup,
    )


@dataclass
class ProbeContext:
    temp_root: Path
    live_config: RuntimeConfig
    secrets: tuple[str, ...]
    isolated: IsolatedSurfaces | None = None
    isolated_error: str | None = None


class Report:
    def __init__(self) -> None:
        self.components: list[dict[str, Any]] = []

    def run(
        self,
        name: str,
        callback: Callable[[], tuple[str, dict[str, Any]]],
    ) -> None:
        started = time.perf_counter()
        status = "FAIL"
        details: dict[str, Any] = {}
        error: str | None = None
        try:
            status, details = callback()
            if status not in VALID_STATUSES:
                raise ValueError(f"invalid_probe_status:{status}")
        except Exception as exc:
            error = safe_error(exc, limit=300)
            details = {"exception_type": type(exc).__name__}
        self.components.append(
            {
                "component": name,
                "status": status,
                "details": details,
                "duration": round(time.perf_counter() - started, 6),
                "error": error,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        counts = {
            status: sum(1 for item in self.components if item["status"] == status)
            for status in ("PASS", "DEGRADED", "BLOCKED", "FAIL")
        }
        return {
            "phase": "20",
            "probe": "backend.tools.phase20_production_probe",
            "generated_at": _utc_now(),
            "summary": {
                "components": len(self.components),
                "status_counts": counts,
                "failed_components": [
                    item["component"]
                    for item in self.components
                    if item["status"] == "FAIL"
                ],
            },
            "components": self.components,
        }


def _config_security(ctx: ProbeContext) -> tuple[str, dict[str, Any]]:
    live = ctx.live_config
    payload = live.to_dict()
    configured = {
        name: bool(os.environ.get(name))
        for name in (
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "AZURE_OPENAI_API_KEY",
            "ELEVENLABS_API_KEY",
            "ROOM_AGENT_LLM_API_KEY",
            "ROOM_AGENT_TTS_API_KEY",
        )
    }

    production_mock_rejected = False
    try:
        RuntimeConfig(
            environment="production",
            allow_mock_providers=None,
        ).enforce_provider("llm", "mock")
    except ConfigurationError:
        production_mock_rejected = True

    sentinel = "phase20-secret-sentinel-7f3c9d"
    redacted = redact_structure(
        {
            "api_key": sentinel,
            "nested": {"authorization": f"Bearer {sentinel}"},
            "safe": True,
        }
    )
    redaction_works = sentinel not in json.dumps(redacted, sort_keys=True)
    diagnostics_safe = not _contains_secret(payload, ctx.secrets)

    details = {
        "environment": live.environment,
        "production_mode": live.is_production,
        "mocks_allowed": live.mocks_allowed,
        "production_mock_rejected": production_mock_rejected,
        "secret_env_configured": configured,
        "secret_values_reported": False,
        "diagnostics_config_redacted": diagnostics_safe,
        "redaction_probe": redaction_works,
        "redacted": True,
    }
    if production_mock_rejected and redaction_works and diagnostics_safe:
        return "PASS", details
    return "FAIL", details


def _database_integrity(ctx: ProbeContext) -> tuple[str, dict[str, Any]]:
    live_results: dict[str, Any] = {}
    for name in DATABASE_NAMES:
        path = Path(ctx.live_config.data_dir) / name
        if not path.is_file():
            continue
        probe = sqlite_probe(path, quick_check=True)
        live_results[name] = {
            "status": probe.get("status"),
            "reachable": probe.get("reachable"),
            "quick_check": (probe.get("detail") or {}).get("quick_check"),
            "schema_version": (probe.get("detail") or {}).get("schema_version"),
        }

    fixture = ctx.temp_root / "database_integrity" / "isolated.db"
    _create_sqlite_fixture(fixture, "phase20-database-integrity")
    fixture_probe = sqlite_probe(fixture, quick_check=True)
    fixture_integrity = integrity_check(fixture)

    live_ok = bool(live_results) and all(
        item["status"] == "READY" and item.get("quick_check") == "ok"
        for item in live_results.values()
    )
    fixture_ok = (
        fixture_probe.get("status") == "READY"
        and (fixture_probe.get("detail") or {}).get("quick_check") == "ok"
        and fixture_integrity.get("ok") is True
    )
    details = {
        "live_databases": live_results,
        "live_databases_checked": len(live_results),
        "isolated_fixture_integrity": fixture_integrity,
        "read_only": True,
    }
    if not live_results:
        details["reason"] = "no_live_databases_present"
        return ("PASS" if fixture_ok else "FAIL"), details
    return ("PASS" if live_ok and fixture_ok else "FAIL"), details


def _backup_restore(ctx: ProbeContext) -> tuple[str, dict[str, Any]]:
    before = _database_fingerprint(ctx.live_config)
    root = ctx.temp_root / "backup_restore"
    config = _isolated_config(root, "fixture")
    source_db = Path(config.data_dir) / "memory.db"
    marker = "phase20-backup-restore-marker"
    _create_sqlite_fixture(source_db, marker)

    result = create_backup(
        config,
        sources=[BackupSource(source_db, "sqlite")],
        destination=Path(config.backup_dir),
        retention=2,
    )
    verification = verify_backup(result.directory)
    restore_target = root / "isolated_restore"
    restored = restore_backup(
        config,
        result.directory,
        target_dir=restore_target,
        overwrite=False,
        safety_backup=False,
    )
    restored_db = restore_target / source_db.name
    restored_integrity = (
        integrity_check(restored_db) if restored_db.is_file() else {"ok": False}
    )
    restored_marker = (
        _read_sqlite_marker(restored_db) if restored_db.is_file() else None
    )
    after = _database_fingerprint(ctx.live_config)

    checks = {
        "backup_created": result.directory.is_dir(),
        "backup_verified": result.verified is True,
        "backup_readable": verification.get("ok") is True,
        "restore_ok": restored.get("ok") is True,
        "restore_target_isolated": restored.get("live_target") is False,
        "live_overwrite": restored.get("overwrote_live_state") is True,
        "restored_database_integrity": restored_integrity.get("ok") is True,
        "restored_marker_matches": restored_marker == marker,
        "live_database_unchanged": before == after,
    }
    details = {
        "backup_file_count": len(result.files),
        "backup_verified": verification.get("ok"),
        "restore_target_isolated": restored.get("live_target") is False,
        "restored_file_count": len(restored.get("restored", [])),
        "restored_database_integrity": restored_integrity.get("ok"),
        "restored_marker_matches": restored_marker == marker,
        "live_database_fingerprint_unchanged": before == after,
        "live_database_fingerprint_count": len(before),
        "checks": checks,
    }
    required = (
        checks["backup_created"]
        and checks["backup_verified"]
        and checks["backup_readable"]
        and checks["restore_ok"]
        and checks["restore_target_isolated"]
        and not checks["live_overwrite"]
        and checks["restored_database_integrity"]
        and checks["restored_marker_matches"]
        and checks["live_database_unchanged"]
    )
    return ("PASS" if required else "FAIL"), details


def _retention(ctx: ProbeContext) -> tuple[str, dict[str, Any]]:
    config = _isolated_config(
        ctx.temp_root,
        "retention",
        log_max_age_days=0.00001,
        log_max_files=1,
        log_max_total_mb=0.0001,
        temp_max_age_hours=0.00001,
        temp_max_files=1,
        temp_max_total_mb=0.0001,
    )
    old_temp = Path(config.temp_dir) / "old_probe.tmp"
    old_log = Path(config.log_dir) / "old_probe.log"
    persistent = Path(config.data_dir) / "important-persistent.json"
    old_temp.write_text("temporary", encoding="utf-8")
    old_log.write_text("log", encoding="utf-8")
    persistent.write_text('{"retained": true}', encoding="utf-8")
    old_time = time.time() - 86400.0
    os.utime(old_temp, (old_time, old_time))
    os.utime(old_log, (old_time, old_time))

    guard_rejected = False
    try:
        guard_directory(PROJECT_ROOT, label="probe_project_root")
    except RetentionError:
        guard_rejected = True

    result = apply_retention(config)
    details = {
        "cleanup_result": result,
        "temporary_artifact_removed": not old_temp.exists(),
        "old_log_removed": not old_log.exists(),
        "persistent_data_retained": persistent.is_file(),
        "project_root_rejected_by_guard": guard_rejected,
        "usage_after": log_usage(config),
    }
    ok = (
        details["temporary_artifact_removed"]
        and details["old_log_removed"]
        and details["persistent_data_retained"]
        and guard_rejected
        and not result.get("errors")
    )
    return ("PASS" if ok else "FAIL"), details


def _health_component(ctx: ProbeContext) -> tuple[str, dict[str, Any]]:
    controller = RoomAgentController(ctx.live_config)
    runtime = controller.status(probe=False)
    if runtime.get("running"):
        health = controller.health()
        component_status = health.get("components")
        valid = (
            health.get("reachable") is True
            and health.get("status") in HEALTH_STATUSES
            and isinstance(component_status, dict)
            and {"backend", "database", "brain", "memory", "learning"}.issubset(
                component_status
            )
            and not _contains_secret(health, ctx.secrets)
        )
        details = {
            "scope": "live_http",
            "overall_status": health.get("status"),
            "components": {
                name: value.get("status")
                for name, value in (component_status or {}).items()
                if isinstance(value, dict)
            },
            "reachable": health.get("reachable"),
            "read_only": True,
        }
        return ("PASS" if valid else "FAIL"), details

    if ctx.isolated is None:
        return "FAIL", {"reason": ctx.isolated_error or "isolated_surface_unavailable"}
    health = ctx.isolated.health
    component_status = health.get("components")
    valid = (
        health.get("status") in HEALTH_STATUSES
        and isinstance(component_status, dict)
        and {"backend", "database", "brain", "memory", "learning"}.issubset(
            component_status
        )
        and not _contains_secret(health, ctx.secrets)
    )
    details = {
        "scope": "isolated_in_process",
        "overall_status": health.get("status"),
        "components": {
            name: value.get("status")
            for name, value in (component_status or {}).items()
            if isinstance(value, dict)
        },
        "live_runtime": "BLOCKED:no_running_instance",
        "read_only": True,
    }
    return ("PASS" if valid else "FAIL"), details


def _metrics_component(ctx: ProbeContext) -> tuple[str, dict[str, Any]]:
    snapshot = DEFAULT_METRICS.snapshot()
    required = {
        "counter_count",
        "series_limit",
        "dropped_series",
        "uptime_seconds",
        "counters",
        "gauges",
        "latency_ms",
        "errors_by_component",
        "recent_errors",
    }
    bounded = (
        isinstance(snapshot.get("counter_count"), int)
        and isinstance(snapshot.get("series_limit"), int)
        and int(snapshot["series_limit"]) > 0
        and int(snapshot["counter_count"]) <= int(snapshot["series_limit"])
        and isinstance(snapshot.get("recent_errors"), list)
    )
    valid = (
        required.issubset(snapshot)
        and bounded
        and not _contains_secret(snapshot, ctx.secrets)
    )
    details = {
        "snapshot_keys": sorted(snapshot),
        "counter_count": snapshot.get("counter_count"),
        "series_limit": snapshot.get("series_limit"),
        "dropped_series": snapshot.get("dropped_series"),
        "recent_error_count": len(snapshot.get("recent_errors", [])),
        "bounded": bounded,
        "redacted": not _contains_secret(snapshot, ctx.secrets),
    }
    return ("PASS" if valid else "FAIL"), details


def _diagnostics_component(ctx: ProbeContext) -> tuple[str, dict[str, Any]]:
    controller = RoomAgentController(ctx.live_config)
    runtime = controller.status(probe=False)
    if runtime.get("running"):
        diagnostics = _http_get_json(
            f"http://{ctx.live_config.host}:{ctx.live_config.brain_port}/diagnostics"
        )
        providers = diagnostics.get("providers")
        runtime_payload = diagnostics.get("runtime")
        valid = (
            diagnostics.get("reachable") is True
            and diagnostics.get("status") in HEALTH_STATUSES
            and isinstance(providers, dict)
            and isinstance(runtime_payload, dict)
            and not _contains_secret(diagnostics, ctx.secrets)
        )
        provider_summary = {
            name: {
                key: value.get(key)
                for key in ("name", "kind", "mock", "configured", "model")
                if key in value
            }
            for name, value in (providers or {}).items()
            if isinstance(value, dict)
        }
        details = {
            "scope": "live_http",
            "status": diagnostics.get("status"),
            "reachable": diagnostics.get("reachable"),
            "providers": provider_summary,
            "brain": (runtime_payload or {}).get("brain"),
            "queues": (runtime_payload or {}).get("queues"),
            "active_skill": (runtime_payload or {}).get("active_skill"),
            "storage_keys": sorted((diagnostics.get("storage") or {}).keys()),
            "redacted": not _contains_secret(diagnostics, ctx.secrets),
        }
        return ("PASS" if valid else "FAIL"), details

    if ctx.isolated is None:
        return "FAIL", {"reason": ctx.isolated_error or "isolated_surface_unavailable"}
    diagnostics = ctx.isolated.diagnostics
    required = {
        "status",
        "environment",
        "health",
        "runtime",
        "config",
        "providers",
        "metrics",
        "storage",
        "persistence",
    }
    valid = (
        required.issubset(diagnostics)
        and diagnostics.get("status") in HEALTH_STATUSES
        and isinstance(diagnostics.get("providers"), dict)
        and not _contains_secret(diagnostics, ctx.secrets)
    )
    details = {
        "scope": "isolated_in_process",
        "status": diagnostics.get("status"),
        "providers": sorted((diagnostics.get("providers") or {}).keys()),
        "storage_keys": sorted((diagnostics.get("storage") or {}).keys()),
        "redacted": not _contains_secret(diagnostics, ctx.secrets),
        "live_runtime": "BLOCKED:no_running_instance",
    }
    return ("PASS" if valid else "FAIL"), details


def _runtime_status(ctx: ProbeContext) -> tuple[str, dict[str, Any]]:
    controller = RoomAgentController(ctx.live_config)
    status = controller.status(probe=True)
    running = bool(status.get("running"))
    lock = status.get("lock") if isinstance(status.get("lock"), dict) else {}
    lock_holder = int(lock.get("pid", 0) or 0)
    state_file = (
        status.get("status_file")
        if isinstance(status.get("status_file"), dict)
        else {}
    )
    state_pid = int(state_file.get("pid", 0) or 0)
    lock_consistent = not running or (
        lock_holder > 0 and (state_pid == 0 or state_pid == lock_holder)
    )
    health: dict[str, Any] | None = None
    if running:
        health = controller.health()

    details = {
        "running": running,
        "pid": status.get("pid"),
        "lock_role": lock.get("role"),
        "state": state_file.get("state"),
        "stop_requested": bool(status.get("stop_requested")),
        "lock_consistent": lock_consistent,
        "http_status_reachable": (
            status.get("http_status", {}).get("reachable")
            if isinstance(status.get("http_status"), dict)
            else None
        ),
        "health_reachable": health.get("reachable") if health else None,
        "health_status": health.get("status") if health else None,
        "read_only": True,
    }
    if not lock_consistent:
        return "FAIL", details
    return "PASS", details


def _queue_bounds(status_payload: dict[str, Any]) -> dict[str, Any]:
    queues = status_payload.get("queues")
    checks: dict[str, Any] = {}
    if not isinstance(queues, dict):
        return checks
    for name, value in queues.items():
        if not isinstance(value, dict):
            continue
        depth = value.get("queued_count")
        if depth is None:
            depth = value.get("queue_length")
        capacity = value.get("queue_capacity")
        if depth is None or capacity is None:
            checks[name] = {"depth": depth, "capacity": capacity, "bounded": None}
            continue
        try:
            numeric_depth = int(depth)
            numeric_capacity = int(capacity)
            checks[name] = {
                "depth": numeric_depth,
                "capacity": numeric_capacity,
                "bounded": 0 <= numeric_depth <= numeric_capacity,
            }
        except (TypeError, ValueError):
            checks[name] = {"depth": depth, "capacity": capacity, "bounded": False}
    return checks


def _resource_process(ctx: ProbeContext) -> tuple[str, dict[str, Any]]:
    if ctx.isolated is None:
        return "FAIL", {"reason": ctx.isolated_error or "isolated_surface_unavailable"}

    controller = RoomAgentController(ctx.live_config)
    live = controller.status(probe=False)
    running = bool(live.get("running"))
    lock = live.get("lock") if isinstance(live.get("lock"), dict) else {}
    lock_holders = 1 if running and lock.get("pid") else 0
    duplicate_loops = max(0, lock_holders - 1)

    queue_checks = _queue_bounds(ctx.isolated.status)
    bounded_queues = all(
        item.get("bounded") is True
        for item in queue_checks.values()
        if item.get("bounded") is not None
    )
    usage = ctx.isolated.usage
    temp_usage = usage.get("temp", {}) if isinstance(usage, dict) else {}
    temp_files = int(temp_usage.get("files", 0) or 0)
    temp_bytes = int(temp_usage.get("bytes", 0) or 0)
    temp_bounded = temp_files <= ctx.isolated.config.temp_max_files

    supported_ok = (
        duplicate_loops == 0
        and bounded_queues
        and temp_bounded
    )
    details = {
        "live_instance_running": running,
        "observed_brain_locks": lock_holders,
        "duplicate_brain_loops": duplicate_loops,
        "queue_checks": queue_checks,
        "queues_bounded": bounded_queues,
        "temporary_files": temp_files,
        "temporary_bytes": temp_bytes,
        "temporary_files_bounded": temp_bounded,
        "unexpected_child_processes": "BLOCKED:no_portable_runtime_scanner",
    }
    if not supported_ok:
        return "FAIL", details
    if not queue_checks or any(
        item.get("bounded") is None for item in queue_checks.values()
    ):
        return "DEGRADED", details
    return "PASS", details


def _recovery_checks(
    ctx: ProbeContext,
    timeout_seconds: float,
    skip: bool,
) -> tuple[str, dict[str, Any]]:
    if skip:
        return "BLOCKED", {"reason": "skipped_by_request"}

    command = [
        sys.executable,
        "-m",
        "unittest",
        "backend.tests.test_phase20_recovery",
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        output = _scrub_text(
            (exc.stderr or "") if isinstance(exc.stderr, str) else "",
            ctx.secrets,
        )
        return (
            "FAIL",
            {
                "command": "python -m unittest backend.tests.test_phase20_recovery",
                "timeout_seconds": timeout_seconds,
                "output_tail": output[-4000:],
            },
        )

    combined = _scrub_text(
        f"{completed.stdout}\n{completed.stderr}".strip(),
        ctx.secrets,
    )
    details = {
        "command": "python -m unittest backend.tests.test_phase20_recovery",
        "returncode": completed.returncode,
        "output_tail": combined[-4000:],
        "isolated": True,
        "started_production_runtime": False,
    }
    return ("PASS" if completed.returncode == 0 else "FAIL"), details


def run(*, recovery_timeout: float = 180.0, skip_recovery: bool = False) -> dict[str, Any]:
    live_config = RuntimeConfig.from_environment()
    secrets = _configured_secrets()
    report = Report()

    with tempfile.TemporaryDirectory(prefix="phase20_production_probe_") as raw:
        ctx = ProbeContext(
            temp_root=Path(raw),
            live_config=live_config,
            secrets=secrets,
        )
        try:
            ctx.isolated = _collect_isolated_surfaces(ctx)
        except Exception as exc:
            ctx.isolated_error = safe_error(exc, limit=240)

        report.run("configuration_security", lambda: _config_security(ctx))
        report.run("database_integrity", lambda: _database_integrity(ctx))
        report.run("backup_restore", lambda: _backup_restore(ctx))
        report.run("retention", lambda: _retention(ctx))
        report.run("health", lambda: _health_component(ctx))
        report.run("diagnostics", lambda: _diagnostics_component(ctx))
        report.run("metrics", lambda: _metrics_component(ctx))
        report.run("runtime_status", lambda: _runtime_status(ctx))
        report.run("resource_process", lambda: _resource_process(ctx))
        report.run(
            "recovery_checks",
            lambda: _recovery_checks(
                ctx,
                timeout_seconds=recovery_timeout,
                skip=skip_recovery,
            ),
        )

    return report.to_dict()


def _write_report(payload: dict[str, Any], output: Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary, output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m backend.tools.phase20_production_probe",
        description="Run the Phase 20 production-hardening probe.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPORT_PATH,
        help="JSON report path (default: Temp/Phase20Production.json).",
    )
    parser.add_argument(
        "--recovery-timeout",
        type=float,
        default=180.0,
        help="Maximum seconds for the isolated recovery-test subprocess.",
    )
    parser.add_argument(
        "--skip-recovery-tests",
        action="store_true",
        help="Skip the recovery test subprocess and report it BLOCKED.",
    )
    args = parser.parse_args(argv)

    payload = run(
        recovery_timeout=max(1.0, float(args.recovery_timeout)),
        skip_recovery=bool(args.skip_recovery_tests),
    )
    _write_report(payload, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    return 1 if payload["summary"]["status_counts"]["FAIL"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
