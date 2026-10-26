"""Secret redaction for logs, diagnostics and error text.

The project already keeps credentials out of dataclasses and diagnostics. This
module is the last line of defence: anything derived from provider output,
exception text or environment values passes through here before it can reach a
log line, an HTTP response or a report file.
"""

from __future__ import annotations

import re
from typing import Any, Mapping


REDACTED = "[redacted]"
MAX_TEXT_LENGTH = 300

#: Mapping keys whose values are always replaced, whatever they contain.
SECRET_KEY_MARKERS = (
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "apikey",
    "api-key",
    "authorization",
    "credential",
    "stream_key",
    "streamkey",
    "access_key",
    "private_key",
)

_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # scheme://user:password@host
    (
        re.compile(r"(?i)([a-z][a-z0-9+.\-]*://)([^/\s:@]+):([^@/\s]+)@"),
        r"\1" + REDACTED + r"@",
    ),
    # Authorization: Bearer xyz / oauth:xyz / PASS oauth:xyz
    (
        re.compile(r"(?i)\b(bearer|oauth)\s*[:=]?\s*[A-Za-z0-9._\-]{6,}"),
        REDACTED,
    ),
    (
        re.compile(r"(?i)\b(pass|password)\s+[^\s]+"),
        r"\1 " + REDACTED,
    ),
    # key=value / key: value / "key": "value" for credential-shaped keys
    (
        re.compile(
            r"(?i)([\"']?\b(?:api[_-]?key|access[_-]?key|auth[_-]?token|token|secret|"
            r"stream[_-]?key|client[_-]?secret|password|authorization|credential|"
            r"private[_-]?key)\b[\"']?\s*[:=]\s*[\"']?)"
            r"([^\s,;\"'&}]+)"
        ),
        r"\1" + REDACTED,
    ),
)


def redact_text(value: Any, limit: int = MAX_TEXT_LENGTH) -> str:
    """Return ``value`` as text with credential shapes replaced and truncated."""
    if value is None:
        return ""
    text = value if isinstance(value, str) else str(value)
    for pattern, replacement in _PATTERNS:
        text = pattern.sub(replacement, text)
    text = text.replace("\r", " ").replace("\n", " ").strip()
    if limit > 0 and len(text) > limit:
        text = text[:limit]
    return text


def safe_error(exc: BaseException, limit: int = MAX_TEXT_LENGTH) -> str:
    """A one-line, redacted ``TypeName: message`` description of an exception."""
    message = redact_text(str(exc), limit=limit)
    if not message:
        return type(exc).__name__
    return f"{type(exc).__name__}: {message}"


def _is_secret_key(key: Any) -> bool:
    if not isinstance(key, str):
        return False
    lowered = key.strip().lower()
    return any(marker in lowered for marker in SECRET_KEY_MARKERS)


def redact_structure(value: Any, depth: int = 6) -> Any:
    """Recursively copy ``value`` with secret-looking mapping entries replaced."""
    if depth <= 0:
        return REDACTED if isinstance(value, (dict, list, tuple, set)) else value
    if isinstance(value, Mapping):
        return {
            str(key): (
                REDACTED if _is_secret_key(key) else redact_structure(item, depth - 1)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_structure(item, depth - 1) for item in value]
    if isinstance(value, set):
        return [redact_structure(item, depth - 1) for item in sorted(value, key=str)]
    if isinstance(value, str):
        return redact_text(value, limit=1000)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return redact_text(value, limit=300)


def contains_secret(text: Any) -> bool:
    """Whether :func:`redact_text` would change ``text`` (used by tests/audits)."""
    if not isinstance(text, str):
        return False
    return any(pattern.search(text) for pattern, _ in _PATTERNS)
