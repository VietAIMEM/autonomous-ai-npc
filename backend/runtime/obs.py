"""Structured, redacted logging helpers.

Existing modules keep their ``logging.getLogger`` calls. New and recovered
operations use :func:`log_event`, which emits one redacted JSON object per line
so a run can be traced by identifier.
"""

from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Optional

from .config import RuntimeConfig
from .redaction import redact_structure


LOG_FILE_NAME = "room_agent.log"
MAX_LOG_BYTES = 2_000_000


def configure_logging(config: RuntimeConfig, log_name: str = LOG_FILE_NAME) -> Optional[Path]:
    """Attach a bounded rotating file handler plus console output, idempotently."""
    root = logging.getLogger()
    if getattr(root, "_room_agent_configured", False):
        return None
    root.setLevel(getattr(logging, config.log_level, logging.INFO))
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    path: Optional[Path] = None
    try:
        config.log_dir.mkdir(parents=True, exist_ok=True)
        path = config.log_dir / log_name
        handler = RotatingFileHandler(
            path,
            maxBytes=MAX_LOG_BYTES,
            backupCount=max(1, min(int(config.log_max_files), 20)),
            encoding="utf-8",
        )
        handler.setFormatter(formatter)
        root.addHandler(handler)
    except OSError:
        path = None
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)
    root._room_agent_configured = True  # type: ignore[attr-defined]
    return path


def log_event(logger: logging.Logger, event: str, /, **fields: Any) -> None:
    """One structured, redacted log line. Never raises."""
    payload: dict[str, Any] = {"event": event}
    for key, value in fields.items():
        if value is None:
            continue
        payload[key] = redact_structure(value, depth=3)
    try:
        logger.info(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str))
    except Exception:  # pragma: no cover - logging must never break a caller
        try:
            logger.info("event=%s", event)
        except Exception:
            pass
