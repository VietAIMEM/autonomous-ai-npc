"""Bounded, redacted operational log reader (PHASE 23).

The dashboard never streams whole log files. This reader seeks to a bounded
window at the end of the active ``room_agent.log``, parses the existing log
line format, applies filters, redacts every message and returns at most the
requested number of lines.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from ..runtime.redaction import redact_text

DEFAULT_LOG_NAME = "room_agent.log"
MAX_SCAN_BYTES = 1_000_000
MAX_LINES = 500
DEFAULT_LIMIT = 200

_LINE_PATTERN = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\s+"
    r"(?P<level>[A-Z]+)\s+"
    r"(?P<component>[^\s]+)\s+"
    r"(?P<message>.*)$"
)

_LEVELS = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50,
}


@dataclass(frozen=True)
class LogQuery:
    limit: int = DEFAULT_LIMIT
    severity: Optional[str] = None
    component: Optional[str] = None
    operation: Optional[str] = None
    since_minutes: Optional[float] = None

    def describe(self) -> dict[str, object]:
        return {
            "limit": self.limit,
            "severity": self.severity,
            "component": self.component,
            "operation": self.operation,
            "since_minutes": self.since_minutes,
        }


@dataclass
class LogEntry:
    timestamp: Optional[str]
    level: str
    component: str
    message: str

    def to_dict(self) -> dict[str, object]:
        return {
            "timestamp": self.timestamp,
            "level": self.level,
            "component": self.component,
            "message": self.message,
        }


class AdminLogReader:
    """Read-only tail reader; never returns more than :data:`MAX_LINES`."""

    def __init__(
        self,
        log_dir: Path | str,
        *,
        log_name: str = DEFAULT_LOG_NAME,
        max_scan_bytes: int = MAX_SCAN_BYTES,
        clock=time.time,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_name = log_name
        self.path = self.log_dir / log_name
        self.max_scan_bytes = max(64_000, int(max_scan_bytes))
        self._clock = clock

    def available(self) -> bool:
        try:
            return self.path.is_file()
        except OSError:
            return False

    def tail(self, query: LogQuery) -> dict[str, object]:
        limit = max(1, min(int(query.limit), MAX_LINES))
        payload: dict[str, object] = {
            "entries": [],
            "count": 0,
            "limit": limit,
            "filters": query.describe(),
            "log_file": self.log_name,
            "available": self.available(),
            "truncated": False,
            "scanned_bytes": 0,
        }
        if not payload["available"]:
            payload["reason"] = "log_file_missing"
            return payload

        raw_lines, scanned = self._read_tail()
        payload["scanned_bytes"] = scanned
        payload["truncated"] = scanned >= self.max_scan_bytes

        threshold = _level_threshold(query.severity)
        cutoff = None
        if query.since_minutes is not None:
            minutes = max(0.0, float(query.since_minutes))
            cutoff = datetime.fromtimestamp(self._clock()) - timedelta(minutes=minutes)
        component = (query.component or "").strip().lower()
        operation = (query.operation or "").strip().lower()

        entries: list[LogEntry] = []
        for line in raw_lines:
            entry = _parse_line(line)
            if entry is None:
                continue
            if threshold is not None and _LEVELS.get(entry.level, 20) < threshold:
                continue
            if component and component not in entry.component.lower():
                continue
            if operation and operation not in entry.message.lower():
                continue
            if cutoff is not None:
                stamp = _parse_time(entry.timestamp)
                if stamp is not None and stamp < cutoff:
                    continue
            entries.append(entry)

        selected = entries[-limit:]
        payload["entries"] = [entry.to_dict() for entry in selected]
        payload["count"] = len(selected)
        payload["matched"] = len(entries)
        if len(entries) > limit:
            payload["truncated"] = True
        return payload

    def _read_tail(self) -> tuple[list[str], int]:
        try:
            size = self.path.stat().st_size
        except OSError:
            return [], 0
        start = max(0, size - self.max_scan_bytes)
        scanned = size - start
        try:
            with open(self.path, "rb") as stream:
                stream.seek(start)
                data = stream.read(self.max_scan_bytes)
        except OSError:
            return [], 0
        text = data.decode("utf-8", errors="replace")
        lines = text.splitlines()
        if start > 0 and lines:
            lines = lines[1:]
        return lines, scanned


def _parse_line(line: str) -> Optional[LogEntry]:
    text = line.strip()
    if not text:
        return None
    match = _LINE_PATTERN.match(text)
    if match is None:
        return LogEntry(None, "RAW", "room_agent", redact_text(text, limit=600))
    return LogEntry(
        match.group("timestamp"),
        match.group("level"),
        match.group("component"),
        redact_text(match.group("message"), limit=600),
    )


def _level_threshold(severity: Optional[str]) -> Optional[int]:
    if not severity:
        return None
    return _LEVELS.get(str(severity).strip().upper())


def _parse_time(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return None


__all__ = [
    "AdminLogReader",
    "DEFAULT_LIMIT",
    "DEFAULT_LOG_NAME",
    "LogEntry",
    "LogQuery",
    "MAX_LINES",
    "MAX_SCAN_BYTES",
]
