"""PHASE 20 production hardening primitives.

Small, dependency-free helpers that sit beside the existing runtime instead of
replacing any part of it: configuration facade, secret redaction, bounded
metrics, health/diagnostics assembly, lifecycle control, backup/restore and
retention.
"""

from .config import (
    ENV_DEVELOPMENT,
    ENV_PRODUCTION,
    ENV_TEST,
    ConfigurationError,
    RuntimeConfig,
)
from .health import (
    STATUS_DEGRADED,
    STATUS_READY,
    STATUS_UNAVAILABLE,
    build_report,
    component,
    from_legacy,
    sqlite_probe,
)
from .lifecycle import (
    DuplicateInstanceError,
    InstanceLock,
    RoomAgentController,
    RuntimeLifecycle,
)
from .metrics import DEFAULT_METRICS, MetricsRegistry
from .redaction import REDACTED, redact_structure, redact_text, safe_error

__all__ = [
    "DEFAULT_METRICS",
    "ENV_DEVELOPMENT",
    "ENV_PRODUCTION",
    "ENV_TEST",
    "ConfigurationError",
    "DuplicateInstanceError",
    "InstanceLock",
    "MetricsRegistry",
    "REDACTED",
    "RoomAgentController",
    "RuntimeConfig",
    "RuntimeLifecycle",
    "STATUS_DEGRADED",
    "STATUS_READY",
    "STATUS_UNAVAILABLE",
    "build_report",
    "component",
    "from_legacy",
    "redact_structure",
    "redact_text",
    "safe_error",
    "sqlite_probe",
]
