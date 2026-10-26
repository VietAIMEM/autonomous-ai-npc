"""Timestamped, verified backups of the persistent production state.

Uses the SQLite online-backup API so a backup is a consistent snapshot even
while the Brain is writing. Every copy is checksummed and integrity-checked at
creation time; a restore refuses to touch live state unless the caller says so
explicitly, and takes a safety backup first when it does.

Only state is copied: databases, learned motion metadata and the sanitized
runtime configuration. Temporary audio/video is deliberately excluded.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from .config import RuntimeConfig
from .redaction import safe_error


BACKUP_PREFIX = "backup_"
MANIFEST_NAME = "manifest.json"
RESTORE_DIR_NAME = "restore"
SNAPSHOT_NAME = "runtime_config.json"
CHUNK_BYTES = 1 << 20


class BackupError(RuntimeError):
    """Raised when a backup cannot be created or verified."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _stamp(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


@dataclass(frozen=True)
class BackupSource:
    path: Path
    kind: str

    @property
    def name(self) -> str:
        return self.path.name


def default_sources(config: RuntimeConfig) -> list[BackupSource]:
    """State that must survive a restart, in restore order."""
    candidates = (
        (config.data_dir / "memory.db", "sqlite"),
        (config.data_dir / "learning.db", "sqlite"),
        (config.data_dir / "motion_import.db", "sqlite"),
        (config.data_dir / "motion_processing.db", "sqlite"),
        (config.data_dir / "generated_motion_metadata.json", "json"),
    )
    return [BackupSource(path, kind) for path, kind in candidates if path.exists()]


@dataclass
class BackupResult:
    directory: Path
    created_at: str
    files: list[dict[str, Any]] = field(default_factory=list)
    verified: bool = False
    pruned: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "directory": self.directory.name,
            "created_at": self.created_at,
            "verified": self.verified,
            "file_count": len(self.files),
            "total_bytes": sum(int(item.get("size_bytes", 0)) for item in self.files),
            "files": self.files,
            "pruned": self.pruned,
            "errors": self.errors,
        }


def create_backup(
    config: RuntimeConfig,
    *,
    sources: Optional[Iterable[BackupSource]] = None,
    destination: Optional[Path] = None,
    retention: Optional[int] = None,
    clock=_utc_now,
) -> BackupResult:
    """Create and verify a timestamped backup directory."""
    moment = clock()
    root = Path(destination) if destination is not None else config.backup_dir
    directory = root / f"{BACKUP_PREFIX}{_stamp(moment)}"
    suffix = 1
    while directory.exists():
        suffix += 1
        directory = root / f"{BACKUP_PREFIX}{_stamp(moment)}_{suffix}"
    directory.mkdir(parents=True, exist_ok=False)

    result = BackupResult(directory=directory, created_at=_iso(moment))
    selected = list(sources) if sources is not None else default_sources(config)

    for source in selected:
        if not source.path.exists():
            continue
        target = directory / source.name
        try:
            _copy_state(source, target)
        except (OSError, sqlite3.Error, BackupError) as exc:
            result.errors.append(f"{source.name}:{safe_error(exc, limit=160)}")
            continue
        entry = _describe(target, source.kind)
        if source.kind == "sqlite":
            integrity = integrity_check(target)
            entry["integrity"] = integrity
            if not integrity["ok"]:
                result.errors.append(f"{source.name}:integrity_check_failed")
        result.files.append(entry)

    try:
        _write_json(
            directory / SNAPSHOT_NAME,
            {
                "created_at": result.created_at,
                "environment": config.environment,
                "config": config.to_dict(),
                "sources": [source.name for source in selected],
                "note": "sanitized configuration snapshot; contains no credentials",
            },
        )
        result.files.append(_describe(directory / SNAPSHOT_NAME, "config"))
        _write_json(
            directory / MANIFEST_NAME,
            {
                "format": 1,
                "created_at": result.created_at,
                "environment": config.environment,
                "files": result.files,
                "errors": result.errors,
            },
        )
    except OSError as exc:
        result.errors.append(f"manifest:{safe_error(exc, limit=160)}")

    result.verified = not result.errors and all(
        item.get("integrity", {}).get("ok", True) for item in result.files
    )
    result.pruned = prune_backups(
        root, keep=config.backup_retention if retention is None else retention
    )
    return result


def integrity_check(path: Path) -> dict[str, Any]:
    """Full ``PRAGMA integrity_check`` on a SQLite file (backup copies only)."""
    detail: dict[str, Any] = {"ok": False, "errors": []}
    try:
        connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, timeout=5.0)
        try:
            rows = connection.execute("PRAGMA integrity_check").fetchall()
        finally:
            connection.close()
    except sqlite3.Error as exc:
        detail["errors"].append(safe_error(exc, limit=160))
        return detail
    messages = [str(row[0]) for row in rows if row]
    detail["messages"] = messages[:5]
    detail["ok"] = len(messages) == 1 and messages[0].lower() == "ok"
    if not detail["ok"]:
        detail["errors"] = messages[:5]
    return detail


def verify_backup(directory: Path) -> dict[str, Any]:
    """Re-verify a backup directory: manifest, checksums and DB integrity."""
    resolved = Path(directory)
    report: dict[str, Any] = {
        "directory": resolved.name,
        "exists": resolved.is_dir(),
        "manifest": False,
        "files": [],
        "ok": False,
        "errors": [],
    }
    if not resolved.is_dir():
        report["errors"].append("backup_directory_missing")
        return report
    manifest_path = resolved / MANIFEST_NAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        report["errors"].append(f"manifest:{safe_error(exc, limit=160)}")
        return report
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), list):
        report["errors"].append("manifest_invalid")
        return report
    report["manifest"] = True
    report["created_at"] = manifest.get("created_at")

    for item in manifest["files"]:
        if not isinstance(item, dict):
            report["errors"].append("manifest_entry_invalid")
            continue
        name = str(item.get("name", ""))
        entry: dict[str, Any] = {"name": name}
        target = resolved / name
        entry["exists"] = target.is_file()
        if not entry["exists"]:
            report["errors"].append(f"{name}:missing")
            report["files"].append(entry)
            continue
        digest = _sha256(target)
        entry["sha256_matches"] = digest == item.get("sha256")
        if not entry["sha256_matches"]:
            report["errors"].append(f"{name}:checksum_mismatch")
        if item.get("kind") == "sqlite":
            integrity = integrity_check(target)
            entry["integrity"] = integrity["ok"]
            if not integrity["ok"]:
                report["errors"].append(f"{name}:integrity_check_failed")
        report["files"].append(entry)

    if not report["files"]:
        report["errors"].append("backup_empty")
    report["ok"] = not report["errors"]
    return report


def restore_backup(
    config: RuntimeConfig,
    backup_directory: Path,
    *,
    target_dir: Optional[Path] = None,
    overwrite: bool = False,
    safety_backup: bool = True,
    clock=_utc_now,
) -> dict[str, Any]:
    """Restore a verified backup into a controlled location.

    ``target_dir`` defaults to ``<state>/restore/<stamp>``. Restoring over the
    live data directory requires ``overwrite=True`` and takes a safety backup
    first, so live state is never silently replaced.
    """
    source = Path(backup_directory)
    verification = verify_backup(source)
    if not verification["ok"]:
        raise BackupError("backup_verification_failed")

    live_dir = Path(config.data_dir)
    if target_dir is None:
        destination = config.state_dir / RESTORE_DIR_NAME / _stamp(clock())
        live_target = False
    else:
        destination = Path(target_dir)
        live_target = _same_location(destination, live_dir)

    if live_target and not overwrite:
        raise BackupError("refusing_to_overwrite_live_state")

    manifest = json.loads((source / MANIFEST_NAME).read_text(encoding="utf-8"))
    names = [
        str(item.get("name"))
        for item in manifest.get("files", [])
        if isinstance(item, dict) and item.get("name")
    ]
    if live_target and safety_backup:
        create_backup(config, clock=clock)

    destination.mkdir(parents=True, exist_ok=True)
    restored: list[dict[str, Any]] = []
    for name in names:
        if name in {MANIFEST_NAME, SNAPSHOT_NAME}:
            continue
        origin = source / name
        if not origin.is_file():
            continue
        target = destination / name
        if target.exists() and not overwrite:
            raise BackupError(f"target_exists:{name}")
        temporary = target.with_name(f"{target.name}.restore-tmp")
        shutil.copy2(origin, temporary)
        os.replace(temporary, target)
        restored.append(
            {
                "name": name,
                "size_bytes": target.stat().st_size,
                "sha256": _sha256(target),
            }
        )

    for name in names:
        if name == SNAPSHOT_NAME:
            shutil.copy2(source / name, destination / name)

    report = {
        "ok": True,
        "source": source.name,
        "target": str(destination),
        "live_target": live_target,
        "overwrote_live_state": bool(live_target and overwrite),
        "safety_backup": live_target and safety_backup,
        "restored": restored,
        "verification": verification,
    }
    return report


def prune_backups(root: Path, *, keep: int = 10) -> list[str]:
    """Delete only our own timestamped backup directories beyond ``keep``."""
    resolved = Path(root)
    if not resolved.is_dir():
        return []
    candidates = sorted(
        (
            entry
            for entry in resolved.iterdir()
            if entry.is_dir()
            and entry.name.startswith(BACKUP_PREFIX)
            and (entry / MANIFEST_NAME).is_file()
        ),
        key=lambda entry: entry.name,
        reverse=True,
    )
    removed: list[str] = []
    for stale in candidates[max(1, int(keep)) :]:
        try:
            shutil.rmtree(stale)
        except OSError:
            continue
        removed.append(stale.name)
    return removed


def backup_status(config: RuntimeConfig) -> dict[str, Any]:
    """Cheap listing for diagnostics: newest backup, count and size."""
    root = Path(config.backup_dir)
    if not root.is_dir():
        return {"directory": root.name, "count": 0, "newest": None}
    entries = sorted(
        (
            entry
            for entry in root.iterdir()
            if entry.is_dir() and entry.name.startswith(BACKUP_PREFIX)
        ),
        key=lambda entry: entry.name,
        reverse=True,
    )
    total = 0
    for entry in entries:
        for item in entry.rglob("*"):
            if item.is_file():
                try:
                    total += item.stat().st_size
                except OSError:
                    continue
    newest: Optional[dict[str, Any]] = None
    if entries:
        newest = {
            "name": entries[0].name,
            "created_at": (read_manifest(entries[0]) or {}).get("created_at"),
        }
    return {
        "directory": root.name,
        "count": len(entries),
        "total_bytes": total,
        "newest": newest,
        "retention": config.backup_retention,
    }


def read_manifest(directory: Path) -> Optional[dict[str, Any]]:
    try:
        payload = json.loads((Path(directory) / MANIFEST_NAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


# -------------------------------------------------------------------- helpers


def _copy_state(source: BackupSource, target: Path) -> None:
    if source.kind == "sqlite":
        _copy_sqlite(source.path, target)
    else:
        shutil.copy2(source.path, target)


def _copy_sqlite(source: Path, target: Path) -> None:
    """Consistent snapshot via the SQLite online backup API."""
    origin = sqlite3.connect(f"file:{Path(source).as_posix()}?mode=ro", uri=True, timeout=5.0)
    try:
        destination = sqlite3.connect(str(target), timeout=5.0)
        try:
            origin.backup(destination)
            destination.commit()
        finally:
            destination.close()
    finally:
        origin.close()


def _describe(path: Path, kind: str) -> dict[str, Any]:
    return {
        "name": path.name,
        "kind": kind,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while True:
            chunk = stream.read(CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _same_location(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return Path(left).absolute() == Path(right).absolute()


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


__all__ = [
    "BACKUP_PREFIX",
    "MANIFEST_NAME",
    "BackupError",
    "BackupResult",
    "BackupSource",
    "backup_status",
    "create_backup",
    "default_sources",
    "integrity_check",
    "prune_backups",
    "restore_backup",
    "verify_backup",
]
