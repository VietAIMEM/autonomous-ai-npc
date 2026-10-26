"""Single read-only facade over the environment the runtime actually uses.

It does not replace the per-subsystem config dataclasses (TTS, chat, stream,
memory, learning). It answers the questions the whole process must agree on:
which environment this is, where state lives, which ports are used, and whether
mock providers are acceptable right now.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional


ENV_DEVELOPMENT = "development"
ENV_TEST = "test"
ENV_PRODUCTION = "production"
VALID_ENVIRONMENTS = (ENV_DEVELOPMENT, ENV_TEST, ENV_PRODUCTION)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = BACKEND_ROOT / "data"
DEFAULT_LOG_DIR = PROJECT_ROOT / "Logs"
DEFAULT_TEMP_DIR = PROJECT_ROOT / "Temp"
DEFAULT_BACKUP_DIR = PROJECT_ROOT / "Backups"
DEFAULT_STATE_DIR = BACKEND_ROOT / "state"

DEFAULT_BRAIN_PORT = 8766
DEFAULT_CHAT_PORT = 8765
DEFAULT_VIEWER_PORT = 8770
DEFAULT_ADMIN_PORT = 8771

#: Provider names that are safe in development and never implicit in production.
MOCK_PROVIDERS = frozenset({"mock"})
TEST_PROVIDERS = frozenset({"mock", "disabled", "fake"})


class ConfigurationError(RuntimeError):
    """Raised when the environment would silently pick an unsafe configuration."""


def _as_bool(value: Optional[str], default: bool) -> bool:
    if value in (None, ""):
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", "disabled"}


def _as_int(value: Optional[str], default: int) -> int:
    try:
        return int(float(value)) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _as_float(value: Optional[str], default: float) -> float:
    try:
        return float(value) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _as_path(value: Optional[str], default: Path) -> Path:
    text = (value or "").strip()
    return Path(text) if text else default


def normalize_environment(value: Optional[str]) -> str:
    text = (value or "").strip().lower()
    if text in {"prod", "production", "live"}:
        return ENV_PRODUCTION
    if text in {"test", "testing", "ci"}:
        return ENV_TEST
    return ENV_DEVELOPMENT


@dataclass(frozen=True)
class RuntimeConfig:
    """Environment facade. ``allow_mock_providers`` None means "decide by env"."""

    environment: str = ENV_DEVELOPMENT
    host: str = "127.0.0.1"
    brain_port: int = DEFAULT_BRAIN_PORT
    chat_port: int = DEFAULT_CHAT_PORT
    viewer_port: int = DEFAULT_VIEWER_PORT
    admin_port: int = DEFAULT_ADMIN_PORT
    data_dir: Path = DEFAULT_DATA_DIR
    log_dir: Path = DEFAULT_LOG_DIR
    temp_dir: Path = DEFAULT_TEMP_DIR
    backup_dir: Path = DEFAULT_BACKUP_DIR
    state_dir: Path = DEFAULT_STATE_DIR
    log_level: str = "INFO"
    log_max_files: int = 200
    log_max_age_days: float = 7.0
    log_max_total_mb: float = 64.0
    temp_max_age_hours: float = 24.0
    temp_max_files: int = 200
    temp_max_total_mb: float = 256.0
    metrics_enabled: bool = True
    # Unity's snapshot loop is synchronous: the next tick is only sent after
    # the previous planner tick returns (real p95 ~26-31 s + a 5 s decision
    # interval). A 30 s threshold reported a genuinely ticking Unity as
    # DEGRADED; 60 s keeps death detection while tolerating that cadence.
    unity_tick_stale_seconds: float = 60.0
    backup_retention: int = 10
    allow_mock_providers: Optional[bool] = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.environment not in VALID_ENVIRONMENTS:
            raise ConfigurationError(f"invalid_environment:{self.environment}")
        if not 1 <= int(self.brain_port) <= 65535:
            raise ConfigurationError("invalid_brain_port")
        if not 1 <= int(self.chat_port) <= 65535:
            raise ConfigurationError("invalid_chat_port")
        if not 1 <= int(self.viewer_port) <= 65535:
            raise ConfigurationError("invalid_viewer_port")
        if not 1 <= int(self.admin_port) <= 65535:
            raise ConfigurationError("invalid_admin_port")

    # ------------------------------------------------------------ decisions

    @property
    def is_production(self) -> bool:
        return self.environment == ENV_PRODUCTION

    @property
    def is_test(self) -> bool:
        return self.environment == ENV_TEST

    @property
    def is_development(self) -> bool:
        return self.environment == ENV_DEVELOPMENT

    @property
    def mocks_allowed(self) -> bool:
        if self.allow_mock_providers is not None:
            return bool(self.allow_mock_providers)
        return not self.is_production

    @property
    def instance_lock_path(self) -> Path:
        return self.state_dir / "room_agent.lock"

    def enforce_provider(self, kind: str, provider: str) -> None:
        """Reject a silently selected mock provider outside development/test."""
        name = (provider or "").strip().lower()
        if name not in MOCK_PROVIDERS:
            return
        if self.mocks_allowed:
            return
        raise ConfigurationError(
            f"{kind}_mock_provider_not_allowed_in_{self.environment}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Diagnostics-safe description: no credentials, no absolute secrets."""
        return {
            "environment": self.environment,
            "host": self.host,
            "brain_port": self.brain_port,
            "chat_port": self.chat_port,
            "viewer_port": self.viewer_port,
            "admin_port": self.admin_port,
            "data_dir": _relative_name(self.data_dir),
            "log_dir": _relative_name(self.log_dir),
            "temp_dir": _relative_name(self.temp_dir),
            "backup_dir": _relative_name(self.backup_dir),
            "state_dir": _relative_name(self.state_dir),
            "log_level": self.log_level,
            "log_max_files": self.log_max_files,
            "log_max_age_days": self.log_max_age_days,
            "log_max_total_mb": self.log_max_total_mb,
            "temp_max_age_hours": self.temp_max_age_hours,
            "metrics_enabled": self.metrics_enabled,
            "unity_tick_stale_seconds": self.unity_tick_stale_seconds,
            "backup_retention": self.backup_retention,
            "mocks_allowed": self.mocks_allowed,
            "notes": list(self.notes),
        }

    # ---------------------------------------------------------- construction

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> "RuntimeConfig":
        env = os.environ if environ is None else environ
        notes: list[str] = []
        raw_environment = (
            env.get("ROOM_AGENT_ENV") or env.get("APP_ENV") or env.get("ENVIRONMENT")
        )
        environment = normalize_environment(raw_environment)
        if raw_environment and environment != raw_environment.strip().lower():
            notes.append("unrecognized_environment_value_treated_as_development")

        allow_mock_raw = env.get("ROOM_AGENT_ALLOW_MOCK_PROVIDERS")
        allow_mock: Optional[bool] = None
        if allow_mock_raw not in (None, ""):
            allow_mock = _as_bool(allow_mock_raw, False)
        if environment == ENV_PRODUCTION and allow_mock is None:
            allow_mock = False

        return cls(
            environment=environment,
            host=(env.get("ROOM_AGENT_HOST") or "127.0.0.1").strip(),
            brain_port=_as_int(env.get("BRAIN_PORT"), DEFAULT_BRAIN_PORT),
            chat_port=_as_int(env.get("CHAT_PORT"), DEFAULT_CHAT_PORT),
            viewer_port=_as_int(env.get("VIEWER_PORT"), DEFAULT_VIEWER_PORT),
            admin_port=_as_int(env.get("ADMIN_PORT"), DEFAULT_ADMIN_PORT),
            data_dir=_as_path(env.get("ROOM_AGENT_DATA_DIR"), DEFAULT_DATA_DIR),
            log_dir=_as_path(env.get("ROOM_AGENT_LOG_DIR"), DEFAULT_LOG_DIR),
            temp_dir=_as_path(env.get("ROOM_AGENT_TEMP_DIR"), DEFAULT_TEMP_DIR),
            backup_dir=_as_path(env.get("ROOM_AGENT_BACKUP_DIR"), DEFAULT_BACKUP_DIR),
            state_dir=_as_path(env.get("ROOM_AGENT_STATE_DIR"), DEFAULT_STATE_DIR),
            log_level=_log_level(env.get("ROOM_AGENT_LOG_LEVEL")),
            log_max_files=max(1, _as_int(env.get("ROOM_AGENT_LOG_MAX_FILES"), 200)),
            log_max_age_days=max(0.0, _as_float(env.get("ROOM_AGENT_LOG_MAX_AGE_DAYS"), 7.0)),
            log_max_total_mb=max(1.0, _as_float(env.get("ROOM_AGENT_LOG_MAX_TOTAL_MB"), 64.0)),
            temp_max_age_hours=max(
                0.0, _as_float(env.get("ROOM_AGENT_TEMP_MAX_AGE_HOURS"), 24.0)
            ),
            temp_max_files=max(1, _as_int(env.get("ROOM_AGENT_TEMP_MAX_FILES"), 200)),
            temp_max_total_mb=max(
                1.0, _as_float(env.get("ROOM_AGENT_TEMP_MAX_TOTAL_MB"), 256.0)
            ),
            metrics_enabled=_as_bool(env.get("ROOM_AGENT_METRICS"), True),
            unity_tick_stale_seconds=max(
                1.0, _as_float(env.get("ROOM_AGENT_UNITY_STALE_SECONDS"), 60.0)
            ),
            backup_retention=max(1, _as_int(env.get("ROOM_AGENT_BACKUP_RETENTION"), 10)),
            allow_mock_providers=allow_mock,
            notes=tuple(notes),
        )


def _log_level(value: Optional[str]) -> str:
    text = (value or "INFO").strip().upper()
    if text in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}:
        return text
    return "INFO"


def _relative_name(path: Path | str) -> str:
    resolved = Path(path)
    try:
        return str(resolved.resolve().relative_to(PROJECT_ROOT))
    except (ValueError, OSError):
        return resolved.name
