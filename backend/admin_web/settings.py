"""Validated, non-secret runtime settings (PHASE 23).

Only a small allowlist of settings can be changed from the dashboard, and only
through this store. Secrets are not settings here: provider credentials stay in
the environment and are never readable or writable from the browser.

Each key declares how it takes effect:

``live``             the running process applies it immediately (and the admin
                     API reports whether that succeeded).
``requires_restart`` the value is persisted and read by the next process start.

Nothing is ever reported as applied without the backend actually applying it.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from ..runtime.obs import log_event

log = logging.getLogger("room_agent.admin_web.settings")

SETTINGS_FILE_NAME = "admin_settings.json"
SETTINGS_VERSION = 1


class SettingValidationError(ValueError):
    """Raised with a stable machine-readable code, never a stack trace."""

    def __init__(self, code: str, key: str = "") -> None:
        super().__init__(code)
        self.code = code
        self.key = key

    def to_dict(self) -> dict[str, str]:
        payload = {"error": self.code}
        if self.key:
            payload["setting"] = self.key
        return payload


@dataclass(frozen=True)
class SettingSpec:
    key: str
    kind: str  # "float" | "bool" | "enum"
    default: Any
    requires_restart: bool
    description: str
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    choices: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "key": self.key,
            "kind": self.kind,
            "default": self.default,
            "requires_restart": self.requires_restart,
            "description": self.description,
        }
        if self.minimum is not None:
            payload["minimum"] = self.minimum
        if self.maximum is not None:
            payload["maximum"] = self.maximum
        if self.choices:
            payload["choices"] = list(self.choices)
        return payload


SETTING_SPECS: tuple[SettingSpec, ...] = (
    SettingSpec(
        key="decision_interval_seconds",
        kind="float",
        default=None,
        requires_restart=True,
        minimum=0.0,
        maximum=600.0,
        description=(
            "Brain decision interval used by the next backend start; "
            "blank keeps the environment/default value."
        ),
    ),
    SettingSpec(
        key="learning_enabled",
        kind="bool",
        default=True,
        requires_restart=False,
        description="Whether skill learning records trials and rankings.",
    ),
    SettingSpec(
        key="log_level",
        kind="enum",
        default="INFO",
        requires_restart=False,
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        description="Root logger level for this process.",
    ),
)

_SPEC_BY_KEY: dict[str, SettingSpec] = {spec.key: spec for spec in SETTING_SPECS}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class AdminSettingsStore:
    """Persisted, allowlisted settings with validation and honest status."""

    def __init__(
        self,
        state_dir: Path | str,
        *,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self.state_dir = Path(state_dir)
        self.path = self.state_dir / SETTINGS_FILE_NAME
        self._clock = clock
        self._lock = threading.Lock()
        self._cache: Optional[dict[str, Any]] = None

    # -------------------------------------------------------------- reading

    @property
    def specs(self) -> tuple[SettingSpec, ...]:
        return SETTING_SPECS

    def load(self) -> dict[str, Any]:
        with self._lock:
            if self._cache is None:
                self._cache = self._read_file()
            return dict(self._cache)

    def values(self) -> dict[str, Any]:
        return dict(self.load().get("values") or {})

    def updated_at(self) -> Optional[str]:
        value = self.load().get("updated_at")
        return value if isinstance(value, str) else None

    def effective(self, key: str, fallback: Any = None) -> Any:
        """The value the runtime should use for ``key`` (stored value wins)."""
        values = self.values()
        if key in values and values[key] is not None:
            return values[key]
        return fallback

    def public(self) -> dict[str, Any]:
        """Settings view for the browser: values, specs and restart flags."""
        return {
            "values": self.values(),
            "updated_at": self.updated_at(),
            "specs": [spec.to_dict() for spec in SETTING_SPECS],
            "secret_values_exposed": False,
        }

    # -------------------------------------------------------------- writing

    def validate(self, key: str, raw: Any) -> Any:
        spec = _SPEC_BY_KEY.get(key)
        if spec is None:
            raise SettingValidationError("unknown_setting", key)
        if spec.kind == "bool":
            if isinstance(raw, bool):
                return raw
            if isinstance(raw, str) and raw.strip().lower() in {"true", "false"}:
                return raw.strip().lower() == "true"
            raise SettingValidationError("invalid_setting_value", key)
        if spec.kind == "enum":
            if not isinstance(raw, str):
                raise SettingValidationError("invalid_setting_value", key)
            value = raw.strip().upper()
            if value not in spec.choices:
                raise SettingValidationError("invalid_setting_choice", key)
            return value
        if spec.kind == "float":
            if raw is None or (isinstance(raw, str) and not raw.strip()):
                if spec.default is None:
                    return None
                raw = spec.default
            if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
                raise SettingValidationError("invalid_setting_value", key)
            try:
                value = float(raw)
            except (TypeError, ValueError) as exc:
                raise SettingValidationError("invalid_setting_value", key) from exc
            if spec.minimum is not None and value < spec.minimum:
                raise SettingValidationError("setting_below_minimum", key)
            if spec.maximum is not None and value > spec.maximum:
                raise SettingValidationError("setting_above_maximum", key)
            return value
        raise SettingValidationError("unsupported_setting_kind", key)

    def save(self, mapping: Mapping[str, Any]) -> dict[str, Any]:
        """Validate and persist a batch of settings.

        Returns ``{"values", "results", "changed", "updated_at"}`` where every
        entry of ``results`` starts as ``saved`` (with ``requires_restart``) or
        ``requires_restart``; the console upgrades ``saved`` to ``applied`` only
        after the running process accepts the value.
        """
        if not isinstance(mapping, Mapping) or not mapping:
            raise SettingValidationError("settings_required")
        validated: dict[str, Any] = {}
        errors: dict[str, str] = {}
        for key, raw in mapping.items():
            try:
                validated[str(key)] = self.validate(str(key), raw)
            except SettingValidationError as exc:
                errors[str(key)] = exc.code
        if errors:
            raise SettingsBatchError(errors)

        with self._lock:
            current = dict(self._cache or self._read_file())
            values = dict(current.get("values") or {})
            changed = False
            for key, value in validated.items():
                if values.get(key) != value:
                    changed = True
                values[key] = value
            payload = {
                "version": SETTINGS_VERSION,
                "updated_at": self._clock().isoformat(),
                "values": values,
            }
            self._write_file(payload)
            self._cache = payload

        results: dict[str, dict[str, Any]] = {}
        for key, value in validated.items():
            spec = _SPEC_BY_KEY[key]
            results[key] = {
                "status": "requires_restart" if spec.requires_restart else "saved",
                "requires_restart": spec.requires_restart,
                "value": value,
                "applied": False,
            }
        log_event(
            log,
            "admin.settings.saved",
            keys=sorted(validated),
            changed=changed,
        )
        return {
            "values": values,
            "results": results,
            "changed": changed,
            "updated_at": payload["updated_at"],
        }

    # ------------------------------------------------------------- internals

    def _read_file(self) -> dict[str, Any]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            return {"version": SETTINGS_VERSION, "updated_at": None, "values": {}}
        if not isinstance(payload, dict):
            return {"version": SETTINGS_VERSION, "updated_at": None, "values": {}}
        values = payload.get("values")
        if not isinstance(values, dict):
            values = {}
        clean: dict[str, Any] = {}
        for key, value in values.items():
            spec = _SPEC_BY_KEY.get(str(key))
            if spec is None:
                continue
            try:
                clean[spec.key] = self.validate(spec.key, value)
            except SettingValidationError:
                continue
        return {
            "version": SETTINGS_VERSION,
            "updated_at": payload.get("updated_at"),
            "values": clean,
        }

    def _write_file(self, payload: Mapping[str, Any]) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f"{self.path.name}.tmp")
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        with open(temporary, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.path)


class SettingsBatchError(SettingValidationError):
    """A rejected batch; carries one code per offending key."""

    def __init__(self, errors: Mapping[str, str]) -> None:
        super().__init__("invalid_settings")
        self.errors = {str(key): str(code) for key, code in errors.items()}

    def to_dict(self) -> dict[str, Any]:
        return {"error": self.code, "settings": dict(self.errors)}


__all__ = [
    "AdminSettingsStore",
    "SETTINGS_FILE_NAME",
    "SETTING_SPECS",
    "SettingSpec",
    "SettingValidationError",
    "SettingsBatchError",
]
