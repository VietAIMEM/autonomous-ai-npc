"""PHASE 12 chat sources: real platform ingest kept behind one boundary.

``ChatProvider`` delivers platform-neutral events; ``ChatSourceRunner`` feeds
them into the existing ChatService. Nothing in this package reaches into the
Brain, Memory, Social, Skill, or Unity layers.
"""

from .base import (
    ChatProvider,
    ChatProviderError,
    ChatSourceEvent,
    ProviderState,
    ReconnectPolicy,
    SourceEventKind,
)
from .config import (
    PROVIDER_DISABLED,
    PROVIDER_MOCK,
    PROVIDER_TWITCH,
    ChatProviderConfig,
    build_provider,
)
from .mock import MockChatProvider
from .runner import ChatSourceRunner, IngestReport
from .twitch import TwitchChatProvider, parse_irc_line

__all__ = [
    "ChatProvider",
    "ChatProviderConfig",
    "ChatProviderError",
    "ChatSourceEvent",
    "ChatSourceRunner",
    "IngestReport",
    "MockChatProvider",
    "PROVIDER_DISABLED",
    "PROVIDER_MOCK",
    "PROVIDER_TWITCH",
    "ProviderState",
    "ReconnectPolicy",
    "SourceEventKind",
    "TwitchChatProvider",
    "build_provider",
    "parse_irc_line",
]
