"""Backend-enforced admin authentication (PHASE 23).

Two layers, both server-side:

* a long-lived **admin token** that only exists in the environment
  (``ROOM_AGENT_ADMIN_TOKEN``) or in a restricted file under the runtime state
  directory. It is never bundled into client JavaScript and never logged.
* a short-lived **session cookie** handed out by ``POST /admin/api/login``.
  Every ``/admin/api/*`` route except the login/session probes requires it.

The dashboard hiding a control is a convenience, never the authorization. The
authorization decision always happens here, inside the backend process.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets
import threading
import time
from pathlib import Path
from typing import Callable, Mapping, Optional

from ..runtime.obs import log_event

log = logging.getLogger("room_agent.admin_web.auth")

TOKEN_ENV_VAR = "ROOM_AGENT_ADMIN_TOKEN"
TOKEN_FILE_NAME = "admin_token.txt"
SESSION_COOKIE_NAME = "room_agent_admin"
MIN_TOKEN_LENGTH = 16
DEFAULT_SESSION_TTL_SECONDS = 8 * 3600.0
MAX_SESSIONS = 64


class AdminTokenError(RuntimeError):
    """Raised when no usable admin token can be resolved."""


def load_admin_token(
    state_dir: Path | str,
    *,
    environ: Mapping[str, str] | None = None,
    create: bool = True,
) -> str:
    """Resolve the admin token: environment first, then a local secret file.

    With ``create`` the first run mints a 32-byte token into
    ``<state_dir>/admin_token.txt`` so a local deployment has an explicit
    secret without shipping one in the repository. The file is written with
    owner-only permissions where the platform supports it.
    """
    env = os.environ if environ is None else environ
    value = (env.get(TOKEN_ENV_VAR) or "").strip()
    if value:
        if len(value) < MIN_TOKEN_LENGTH:
            raise AdminTokenError("admin_token_too_short")
        return value

    path = Path(state_dir) / TOKEN_FILE_NAME
    existing = _read_token_file(path)
    if existing:
        return existing
    if not create:
        raise AdminTokenError("admin_token_unavailable")
    token = secrets.token_urlsafe(32)
    try:
        _write_token_file(path, token)
    except OSError as exc:
        raise AdminTokenError(f"admin_token_write_failed:{type(exc).__name__}") from exc
    return token


def _read_token_file(path: Path) -> Optional[str]:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    except OSError:
        return None
    if len(value) < MIN_TOKEN_LENGTH:
        return None
    return value


def _write_token_file(path: Path, token: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with open(temporary, "w", encoding="utf-8") as stream:
        stream.write(token)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.chmod(temporary, 0o600)
    except OSError:  # pragma: no cover - Windows/ACL platforms
        pass
    os.replace(temporary, path)


class AdminSessionStore:
    """Bounded, in-memory session registry keyed by hashed session id."""

    def __init__(
        self,
        *,
        ttl_seconds: float = DEFAULT_SESSION_TTL_SECONDS,
        capacity: int = MAX_SESSIONS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ttl = max(300.0, float(ttl_seconds))
        self._capacity = max(4, int(capacity))
        self._clock = clock
        self._lock = threading.Lock()
        self._sessions: dict[str, float] = {}

    @property
    def ttl_seconds(self) -> float:
        return self._ttl

    def issue(self) -> tuple[str, float]:
        token = secrets.token_urlsafe(32)
        expires_at = self._clock() + self._ttl
        with self._lock:
            self._prune_locked()
            self._sessions[_hash(token)] = expires_at
            if len(self._sessions) > self._capacity:
                oldest = sorted(self._sessions.items(), key=lambda item: item[1])
                for key, _ in oldest[: len(self._sessions) - self._capacity]:
                    self._sessions.pop(key, None)
        return token, expires_at

    def validate(self, token: Optional[str]) -> bool:
        if not token:
            return False
        key = _hash(token)
        with self._lock:
            self._prune_locked()
            expires_at = self._sessions.get(key)
            if expires_at is None:
                return False
            if expires_at <= self._clock():
                self._sessions.pop(key, None)
                return False
            return True

    def revoke(self, token: Optional[str]) -> bool:
        if not token:
            return False
        key = _hash(token)
        with self._lock:
            return self._sessions.pop(key, None) is not None

    def revoke_all(self) -> int:
        with self._lock:
            count = len(self._sessions)
            self._sessions.clear()
            return count

    @property
    def active_sessions(self) -> int:
        with self._lock:
            self._prune_locked()
            return len(self._sessions)

    def _prune_locked(self) -> None:
        now = self._clock()
        expired = [key for key, expires in self._sessions.items() if expires <= now]
        for key in expired:
            self._sessions.pop(key, None)


class AdminLoginThrottle:
    """Small fixed-window limiter so token guessing is not free."""

    def __init__(
        self,
        *,
        max_attempts: int = 5,
        window_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max_attempts = max(1, int(max_attempts))
        self._window = max(5.0, float(window_seconds))
        self._clock = clock
        self._lock = threading.Lock()
        self._failures: list[float] = []

    def allowed(self) -> bool:
        now = self._clock()
        with self._lock:
            self._prune_locked(now)
            return len(self._failures) < self._max_attempts

    def retry_after_seconds(self) -> float:
        now = self._clock()
        with self._lock:
            self._prune_locked(now)
            if len(self._failures) < self._max_attempts:
                return 0.0
            return max(0.0, self._failures[0] + self._window - now)

    def record_failure(self) -> None:
        with self._lock:
            self._failures.append(self._clock())

    def record_success(self) -> None:
        with self._lock:
            self._failures.clear()

    def _prune_locked(self, now: float) -> None:
        self._failures = [stamp for stamp in self._failures if now - stamp < self._window]


class AdminAuthenticator:
    """Token + session boundary shared by every admin route."""

    def __init__(
        self,
        token: str,
        *,
        sessions: Optional[AdminSessionStore] = None,
        throttle: Optional[AdminLoginThrottle] = None,
    ) -> None:
        if not token or len(token) < MIN_TOKEN_LENGTH:
            raise AdminTokenError("admin_token_too_short")
        self._token = token
        self.sessions = sessions or AdminSessionStore()
        self.throttle = throttle or AdminLoginThrottle()

    def verify_token(self, candidate: Optional[str]) -> bool:
        if not candidate:
            return False
        return hmac.compare_digest(candidate.encode("utf-8"), self._token.encode("utf-8"))

    def login(self, candidate: Optional[str], *, source: str = "local") -> Optional[str]:
        """Exchange a valid token for a session id; ``None`` means rejected."""
        if not self.throttle.allowed():
            log_event(
                log,
                "admin.auth.throttled",
                source=source,
                retry_after=round(self.throttle.retry_after_seconds(), 3),
            )
            return None
        if not self.verify_token(candidate):
            self.throttle.record_failure()
            log_event(log, "admin.auth.rejected", source=source)
            return None
        self.throttle.record_success()
        token, _expires = self.sessions.issue()
        log_event(log, "admin.auth.login", source=source)
        return token

    def authenticate(self, session_token: Optional[str]) -> bool:
        return self.sessions.validate(session_token)

    def logout(self, session_token: Optional[str]) -> bool:
        revoked = self.sessions.revoke(session_token)
        if revoked:
            log_event(log, "admin.auth.logout")
        return revoked


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


__all__ = [
    "AdminAuthenticator",
    "AdminLoginThrottle",
    "AdminSessionStore",
    "AdminTokenError",
    "DEFAULT_SESSION_TTL_SECONDS",
    "MAX_SESSIONS",
    "MIN_TOKEN_LENGTH",
    "SESSION_COOKIE_NAME",
    "TOKEN_ENV_VAR",
    "TOKEN_FILE_NAME",
    "load_admin_token",
]
