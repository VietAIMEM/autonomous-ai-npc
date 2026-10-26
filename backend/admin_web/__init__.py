"""PHASE 23 privileged admin surface.

A dedicated, authenticated control plane that sits beside the public viewer
web. It owns no business logic of its own: every route projects or commands an
existing backend service (lifecycle controller, Brain, memory, learning,
skills, speech, chat, stream, motion). The browser never touches SQLite,
providers, Unity, FFmpeg or shell commands directly.
"""

from .auth import (
    AdminAuthenticator,
    AdminSessionStore,
    AdminLoginThrottle,
    AdminTokenError,
    load_admin_token,
)
from .console import AdminConsole
from .server import AdminWebServer
from .settings import AdminSettingsStore

__all__ = [
    "AdminAuthenticator",
    "AdminConsole",
    "AdminLoginThrottle",
    "AdminSessionStore",
    "AdminSettingsStore",
    "AdminTokenError",
    "AdminWebServer",
    "load_admin_token",
]
