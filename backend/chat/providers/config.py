"""Chat source selection and provider construction.

Read like ``backend/tts/config.py``: the token, when one is configured, is only
ever read from disk inside the real provider. Nothing here returns key material
through a dataclass, ``repr``, or diagnostics payload.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Optional

from .base import ChatProvider, ReconnectPolicy


PROVIDER_DISABLED = "disabled"
PROVIDER_MOCK = "mock"
PROVIDER_TWITCH = "twitch"
VALID_PROVIDERS = (PROVIDER_DISABLED, PROVIDER_MOCK, PROVIDER_TWITCH)

DEFAULT_PLATFORM = "twitch"
DEFAULT_IRC_HOST = "irc.chat.twitch.tv"
DEFAULT_IRC_PORT = 443
DEFAULT_CHANNEL = "ironmouse"
DEFAULT_TIMEOUT_SECONDS = 8.0
DEFAULT_POLL_INTERVAL_SECONDS = 1.0
DEFAULT_SOCKET_READ_SECONDS = 0.25
DEFAULT_MAX_MESSAGE_AGE_SECONDS = 60.0
DEFAULT_INGEST_CAPACITY = 25
DEFAULT_TOKEN_PATH = Path(__file__).resolve().parents[2] / "chat_twitch_token.txt"


def anonymous_nickname(seed: Optional[int] = None) -> str:
    """A read-only anonymous IRC nickname; Twitch requires no password for one."""
    import random

    rng = random.Random(seed)
    return f"justinfan{rng.randint(100000, 999999)}"


@dataclass(frozen=True)
class ChatProviderConfig:
    provider: str = PROVIDER_DISABLED
    platform: str = DEFAULT_PLATFORM
    channels: tuple[str, ...] = ()
    nickname: str = ""
    token_path: Path | str = DEFAULT_TOKEN_PATH
    server_host: str = DEFAULT_IRC_HOST
    server_port: int = DEFAULT_IRC_PORT
    use_tls: bool = True
    connect_timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    socket_read_seconds: float = DEFAULT_SOCKET_READ_SECONDS
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS
    max_message_age_seconds: float = DEFAULT_MAX_MESSAGE_AGE_SECONDS
    ingest_capacity: int = DEFAULT_INGEST_CAPACITY
    reconnect: ReconnectPolicy = field(default_factory=ReconnectPolicy)

    @property
    def enabled(self) -> bool:
        return self.provider != PROVIDER_DISABLED

    @property
    def channel(self) -> str:
        return self.channels[0] if self.channels else ""

    def sanitized(self) -> "ChatProviderConfig":
        """A copy that is safe to log: no token material, no token directory."""
        return replace(self, token_path=Path(self.token_path).name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "platform": self.platform,
            "channels": list(self.channels),
            "nickname": self.nickname,
            "server_host": self.server_host,
            "server_port": self.server_port,
            "use_tls": self.use_tls,
            "connect_timeout_seconds": self.connect_timeout_seconds,
            "poll_interval_seconds": self.poll_interval_seconds,
            "max_message_age_seconds": self.max_message_age_seconds,
            "ingest_capacity": self.ingest_capacity,
            "enabled": self.enabled,
            "reconnect": self.reconnect.to_dict(),
        }

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> "ChatProviderConfig":
        env = os.environ if environ is None else environ
        channel = (env.get("CHAT_TWITCH_CHANNEL") or "").strip().lstrip("#").lower()
        requested = (env.get("CHAT_PROVIDER") or "").strip().lower()
        if not requested:
            # Unset means "do nothing" unless an operator configured a channel,
            # so PHASE 01-11 default runs are byte-for-byte unchanged.
            requested = PROVIDER_TWITCH if channel else PROVIDER_DISABLED
        if requested not in VALID_PROVIDERS:
            requested = PROVIDER_DISABLED

        channels = tuple(
            dict.fromkeys(
                part.strip().lstrip("#").lower()
                for part in (channel or DEFAULT_CHANNEL).split(",")
                if part.strip().lstrip("#").lower()
            )
        )
        if requested == PROVIDER_TWITCH and not channels:
            channels = (DEFAULT_CHANNEL,)

        return cls(
            provider=requested,
            platform=(env.get("CHAT_PLATFORM") or DEFAULT_PLATFORM).strip().lower(),
            channels=channels,
            nickname=(env.get("CHAT_TWITCH_NICKNAME") or "").strip(),
            token_path=Path(env.get("CHAT_TWITCH_TOKEN_PATH") or DEFAULT_TOKEN_PATH),
            server_host=(env.get("CHAT_TWITCH_HOST") or DEFAULT_IRC_HOST).strip(),
            server_port=int(_as_float(env.get("CHAT_TWITCH_PORT"), DEFAULT_IRC_PORT)),
            use_tls=_as_bool(env.get("CHAT_TWITCH_TLS"), True),
            connect_timeout_seconds=_as_float(
                env.get("CHAT_CONNECT_TIMEOUT_SECONDS"), DEFAULT_TIMEOUT_SECONDS
            ),
            socket_read_seconds=_as_float(
                env.get("CHAT_SOCKET_READ_SECONDS"), DEFAULT_SOCKET_READ_SECONDS
            ),
            poll_interval_seconds=_as_float(
                env.get("CHAT_POLL_INTERVAL_SECONDS"), DEFAULT_POLL_INTERVAL_SECONDS
            ),
            max_message_age_seconds=_as_float(
                env.get("CHAT_MAX_MESSAGE_AGE_SECONDS"), DEFAULT_MAX_MESSAGE_AGE_SECONDS
            ),
            ingest_capacity=int(
                _as_float(env.get("CHAT_INGEST_CAPACITY"), DEFAULT_INGEST_CAPACITY)
            ),
            reconnect=ReconnectPolicy(
                base_delay_seconds=_as_float(
                    env.get("CHAT_RECONNECT_BASE_SECONDS"), 1.0
                ),
                max_delay_seconds=_as_float(
                    env.get("CHAT_RECONNECT_MAX_SECONDS"), 30.0
                ),
                max_attempts=int(_as_float(env.get("CHAT_RECONNECT_MAX_ATTEMPTS"), 5)),
            ),
        )


def build_provider(
    config: ChatProviderConfig | None = None,
    transport: Any = None,
) -> Optional[ChatProvider]:
    """Create the configured provider, or ``None`` when chat input is disabled."""
    resolved = config or ChatProviderConfig()
    if resolved.provider == PROVIDER_TWITCH:
        from .twitch import TwitchChatProvider

        return TwitchChatProvider(resolved, transport=transport)
    if resolved.provider == PROVIDER_MOCK:
        from .mock import MockChatProvider

        return MockChatProvider(resolved)
    return None


def read_token(path: Path | str) -> str:
    """Return the configured OAuth token, or an empty string when absent.

    Callers must treat the result as a secret: it is never stored on the
    provider's public attributes and never included in diagnostics.
    """
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _as_float(value: Optional[str], default: float) -> float:
    try:
        return float(value) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _as_bool(value: Optional[str], default: bool) -> bool:
    if value in (None, ""):
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}
