"""Text normalization used before filtering and intent extraction."""

from __future__ import annotations

import re
import unicodedata


_SPACES = re.compile(r"\s+")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def normalize_text(raw_text: str) -> str:
    """Keep readable text while producing a stable matching form."""
    if not isinstance(raw_text, str):
        return ""
    value = unicodedata.normalize("NFKC", raw_text)
    value = _CONTROL.sub(" ", value)
    value = _SPACES.sub(" ", value).strip()
    return value

