"""Metadata-only persistence for validated generated motions.

The file is rewritten atomically (temp file + ``os.replace``) so a crash during
a save can never truncate the existing metadata, and it keeps only the newest
``max_records`` entries so continuous operation cannot grow it without bound.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any


class JSONMotionGenerationMetadataStore:
    """Stores result metadata only; never stores raw motion tensors."""

    def __init__(self, path: Path | str, *, max_records: int = 500) -> None:
        if max_records <= 0:
            raise ValueError("max_records must be positive")
        self.path = Path(path)
        self.max_records = int(max_records)
        self._lock = threading.Lock()

    def save(self, result: Any) -> None:
        with self._lock:
            records = self.list()
            records.append(result.to_dict())
            if len(records) > self.max_records:
                records = records[-self.max_records :]
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(records, ensure_ascii=False, indent=2)
            temporary = self.path.with_name(f"{self.path.name}.tmp")
            with open(temporary, "w", encoding="utf-8") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)

    def count(self) -> int:
        return len(self.list())

    def list(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        return value if isinstance(value, list) else []
