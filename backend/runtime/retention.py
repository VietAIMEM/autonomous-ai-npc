"""Bounded retention for logs and temporary artifacts.

Only two kinds of data are touched: log files in the configured log directory
and temporary files in the configured temp directory. Persistent state
(Memory, SkillKnowledge, learned compositions, viewer relationships, databases,
backups) is never deleted here - the caller has to opt in per directory.

Every pass is age-first and then size-first, and a directory guard refuses
obviously dangerous roots so a bad environment variable cannot wipe a disk.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from .config import BACKEND_ROOT, PROJECT_ROOT, RuntimeConfig
from .redaction import safe_error


#: Files whose name contains one of these markers are treated as logs.
LOG_MARKERS = (".log",)


class RetentionError(RuntimeError):
    """Raised when a cleanup target is unsafe to touch."""


@dataclass
class RetentionResult:
    removed: list[str] = field(default_factory=list)
    bytes_freed: int = 0
    kept: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "removed_count": len(self.removed),
            "bytes_freed": self.bytes_freed,
            "kept": self.kept,
            "removed": self.removed[:50],
            "errors": self.errors,
        }


def guard_directory(path: Path, *, label: str) -> Path:
    """Refuse filesystem roots and the project/backend roots themselves."""
    resolved = Path(path).expanduser()
    try:
        resolved = resolved.resolve()
    except OSError as exc:
        raise RetentionError(f"{label}_unresolvable") from exc
    if resolved.parent == resolved:
        raise RetentionError(f"{label}_is_filesystem_root")
    for forbidden in (PROJECT_ROOT.resolve(), BACKEND_ROOT.resolve()):
        if resolved == forbidden:
            raise RetentionError(f"{label}_is_project_root")
    return resolved


def apply_retention(
    config: RuntimeConfig,
    *,
    include_temp: bool = True,
    include_logs: bool = True,
    now: Optional[float] = None,
) -> dict[str, Any]:
    """Run one bounded cleanup pass and report exactly what was removed."""
    moment = time.time() if now is None else float(now)
    report = RetentionResult()

    if include_logs:
        try:
            _clean_logs(config, moment, report)
        except RetentionError as exc:
            report.errors.append(safe_error(exc, limit=160))
    if include_temp:
        try:
            _clean_temp(config, moment, report)
        except RetentionError as exc:
            report.errors.append(safe_error(exc, limit=160))

    return report.to_dict()


def _clean_logs(config: RuntimeConfig, now: float, report: RetentionResult) -> None:
    directory = guard_directory(config.log_dir, label="log_dir")
    if not directory.is_dir():
        return
    files = _list_files(directory, markers=LOG_MARKERS)
    if config.log_max_age_days > 0:
        cutoff = now - config.log_max_age_days * 86400.0
        for path, stat in files:
            if stat.st_mtime < cutoff:
                _remove_file(path, stat, report)
    remaining = [(path, stat) for path, stat in files if path.exists()]
    if len(remaining) > config.log_max_files:
        for path, stat in sorted(remaining, key=lambda item: item[1].st_mtime)[
            : len(remaining) - config.log_max_files
        ]:
            _remove_file(path, stat, report)
    _enforce_size_cap(remaining, config.log_max_total_mb * (1 << 20), report)


def _clean_temp(config: RuntimeConfig, now: float, report: RetentionResult) -> None:
    directory = guard_directory(config.temp_dir, label="temp_dir")
    if not directory.is_dir():
        return
    files = _list_files(directory, markers=None)
    if config.temp_max_age_hours > 0:
        cutoff = now - config.temp_max_age_hours * 3600.0
        for path, stat in files:
            if stat.st_mtime < cutoff:
                _remove_file(path, stat, report)
    remaining = [(path, stat) for path, stat in files if path.exists()]
    if len(remaining) > config.temp_max_files:
        for path, stat in sorted(remaining, key=lambda item: item[1].st_mtime)[
            : len(remaining) - config.temp_max_files
        ]:
            _remove_file(path, stat, report)
    remaining = [(path, stat) for path, stat in remaining if path.exists()]
    _enforce_size_cap(remaining, config.temp_max_total_mb * (1 << 20), report)
    _remove_empty_dirs(directory, report)


def _enforce_size_cap(
    files: Iterable[tuple[Path, os.stat_result]],
    cap_bytes: float,
    report: RetentionResult,
) -> None:
    entries = [(path, stat) for path, stat in files if path.exists()]
    total = sum(stat.st_size for _path, stat in entries)
    if total <= cap_bytes:
        report.kept += len(entries)
        return
    for path, stat in sorted(entries, key=lambda item: item[1].st_mtime):
        if total <= cap_bytes:
            break
        total -= stat.st_size
        _remove_file(path, stat, report)
    report.kept += len([1 for path, _stat in entries if path.exists()])


def _list_files(
    directory: Path, *, markers: Optional[tuple[str, ...]]
) -> list[tuple[Path, os.stat_result]]:
    entries: list[tuple[Path, os.stat_result]] = []
    for path in _walk_files(directory):
        if markers is not None and not any(
            marker in path.name.lower() for marker in markers
        ):
            continue
        try:
            entries.append((path, path.stat()))
        except OSError:
            continue
    return entries


def _walk_files(directory: Path) -> list[Path]:
    found: list[Path] = []
    for root, _dirs, names in os.walk(directory):
        for name in names:
            found.append(Path(root) / name)
    return found


def _remove_file(path: Path, stat: os.stat_result, report: RetentionResult) -> None:
    try:
        path.unlink()
    except OSError as exc:
        report.errors.append(f"{path.name}:{safe_error(exc, limit=120)}")
        return
    report.removed.append(path.name)
    report.bytes_freed += int(stat.st_size)


def _remove_empty_dirs(directory: Path, report: RetentionResult) -> None:
    for root, _dirs, _files in os.walk(directory, topdown=False):
        current = Path(root)
        if current == directory:
            continue
        try:
            # Check the live directory rather than the walk's stale listing so
            # nested empty directories collapse in a single pass.
            if not any(current.iterdir()):
                current.rmdir()
        except OSError:
            continue


def log_usage(config: RuntimeConfig) -> dict[str, Any]:
    """Cheap size probe used by diagnostics."""
    return {
        "log": _directory_usage(config.log_dir),
        "temp": _directory_usage(config.temp_dir),
    }


def _directory_usage(directory: Path) -> dict[str, Any]:
    resolved = Path(directory)
    if not resolved.is_dir():
        return {"files": 0, "bytes": 0}
    count = 0
    total = 0
    for root, _dirs, names in os.walk(resolved):
        for name in names:
            try:
                total += (Path(root) / name).stat().st_size
                count += 1
            except OSError:
                continue
    return {"files": count, "bytes": total}


__all__ = [
    "RetentionError",
    "RetentionResult",
    "apply_retention",
    "guard_directory",
    "log_usage",
]
