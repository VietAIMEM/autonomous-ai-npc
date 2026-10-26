"""Twitch chat provider: IRCv3 tag parsing plus a bounded reconnect loop.

Only this module knows the platform exists, and only the backend runs it. The
connection is **read-only**: the command whitelist below makes it impossible to
post into a channel. Credentials are optional (anonymous read works) and are
scrubbed from every error string that can reach logs or diagnostics.
"""

from __future__ import annotations

import socket
import ssl
import re
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Deque, Mapping, Optional, Sequence

from ..models import utc_now
from .base import (
    ChatProvider,
    ChatProviderError,
    ChatSourceEvent,
    ProviderState,
    ReconnectPolicy,
    SourceEventKind,
)
from .config import ChatProviderConfig, PROVIDER_TWITCH, anonymous_nickname, read_token


PLATFORM = "twitch"
MAX_DEDUP_IDS = 2048
MAX_ERROR_DETAIL = 200
OAUTH_RE = re.compile(r"oauth:[^\s]*")
SEND_PREFIX_WHITELIST = (
    "CAP REQ",
    "PASS ",
    "NICK ",
    "JOIN ",
    "PONG ",
    "QUIT",
)


@dataclass(frozen=True)
class IrcMessage:
    tags: Mapping[str, str] = field(default_factory=dict)
    prefix: str = ""
    command: str = ""
    params: tuple[str, ...] = ()

    @property
    def nick(self) -> str:
        if not self.prefix:
            return ""
        head = self.prefix.split("!", 1)[0]
        return head.split("@", 1)[0]

    @property
    def trailing(self) -> str:
        return self.params[-1] if self.params else ""


def unescape_tag_value(value: str) -> str:
    if "\\" not in value:
        return value
    out: list[str] = []
    index = 0
    length = len(value)
    while index < length:
        char = value[index]
        if char == "\\" and index + 1 < length:
            nxt = value[index + 1]
            mapping = {":": ";", "s": " ", "\\": "\\", "r": "\r", "n": "\n"}
            out.append(mapping.get(nxt, nxt))
            index += 2
            continue
        out.append(char)
        index += 1
    return "".join(out)


def parse_tags(segment: str) -> dict[str, str]:
    tags: dict[str, str] = {}
    for entry in segment.split(";"):
        if not entry:
            continue
        key, _, raw = entry.partition("=")
        tags[key] = unescape_tag_value(raw)
    return tags


def parse_irc_line(line: str) -> Optional[IrcMessage]:
    """Parse one raw IRCv3 line. Returns ``None`` for anything malformed."""
    text = line.strip("\r\n")
    if not text:
        return None
    tags: dict[str, str] = {}
    if text.startswith("@"):
        tag_segment, _, text = text[1:].partition(" ")
        tags = parse_tags(tag_segment)
        text = text.lstrip(" ")
    prefix = ""
    if text.startswith(":"):
        prefix, _, text = text[1:].partition(" ")
    if not text:
        return None
    params: list[str] = []
    remainder = text
    while remainder:
        if remainder.startswith(":"):
            params.append(remainder[1:])
            break
        token, _, remainder = remainder.partition(" ")
        if token:
            params.append(token)
        else:
            remainder = remainder.lstrip(" ")
    if not params:
        return None
    return IrcMessage(tags=tags, prefix=prefix, command=params[0].upper(), params=tuple(params[1:]))


def epoch_ms_to_utc(value: str) -> Optional[datetime]:
    try:
        millis = int(value)
    except (TypeError, ValueError):
        return None
    if millis <= 0:
        return None
    return datetime.fromtimestamp(millis / 1000.0, tz=timezone.utc)


class ChatTransport:
    """Line-oriented connection seam so unit tests need no network."""

    def connect(self) -> None:
        raise NotImplementedError

    def send(self, line: str) -> None:
        raise NotImplementedError

    def read_lines(self) -> list[str]:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class SocketChatTransport(ChatTransport):
    #: Upper bound on bytes consumed by a single read_lines() call. The
    #: provider polls about once per second; draining everything already
    #: buffered keeps the server from backpressuring into a stale FIFO,
    #: while the bound stops a burst from monopolizing the runner.
    MAX_DRAIN_BYTES = 256 * 1024

    def __init__(
        self,
        host: str,
        port: int,
        connect_timeout_seconds: float,
        read_timeout_seconds: float,
        use_tls: bool = True,
    ) -> None:
        self.host = host
        self.port = port
        self.connect_timeout_seconds = connect_timeout_seconds
        self.read_timeout_seconds = read_timeout_seconds
        self.use_tls = use_tls
        self._sock: Optional[socket.socket] = None
        self._buffer = b""

    def connect(self) -> None:
        raw = socket.create_connection(
            (self.host, self.port), timeout=self.connect_timeout_seconds
        )
        if self.use_tls:
            context = ssl.create_default_context()
            raw = context.wrap_socket(raw, server_hostname=self.host)
        raw.settimeout(self.read_timeout_seconds)
        self._sock = raw
        self._buffer = b""

    def send(self, line: str) -> None:
        if self._sock is None:
            raise ChatProviderError("not_connected")
        self._sock.sendall(f"{line}\r\n".encode("utf-8"))

    def read_lines(self) -> list[str]:
        if self._sock is None:
            raise ChatProviderError("not_connected")
        try:
            chunk = self._sock.recv(8192)
        except (socket.timeout, ssl.SSLWantReadError):
            # A quiet socket is not a closed one: returning empty keeps the
            # connection alive between messages instead of triggering a
            # reconnect on every read-timeout tick.
            return []
        except OSError as exc:
            raise ChatProviderError("transport_error", type(exc).__name__) from None
        if chunk == b"":
            if self._buffer:
                lines = self._drain_buffer()
                if lines:
                    return lines
            raise ChatProviderError("connection_closed")
        self._buffer += chunk
        self._drain_socket()
        return self._drain_buffer()

    def _drain_socket(self) -> None:
        """Non-blocking, bounded drain of bytes already available.

        One blocking ``recv`` per poll leaves the rest of the kernel buffer
        queued; Twitch then backpressures that reader onto an older FIFO
        whose messages age past the freshness guard. Reading everything that
        is immediately available, up to ``MAX_DRAIN_BYTES``, keeps the
        connection live without touching the staleness policy itself.
        """
        if self._sock is None:
            return
        total = len(self._buffer)
        self._sock.settimeout(0)
        try:
            while total < self.MAX_DRAIN_BYTES:
                try:
                    chunk = self._sock.recv(8192)
                except (socket.timeout, ssl.SSLWantReadError, BlockingIOError):
                    break
                except OSError as exc:
                    raise ChatProviderError("transport_error", type(exc).__name__) from None
                if chunk == b"":
                    # A close found mid-drain is surfaced by the next poll's
                    # first recv, exactly like the pre-drain behavior.
                    break
                self._buffer += chunk
                total += len(chunk)
        finally:
            self._sock.settimeout(self.read_timeout_seconds)

    def close(self) -> None:
        sock, self._sock = self._sock, None
        self._buffer = b""
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    def _drain_buffer(self) -> list[str]:
        if b"\n" not in self._buffer:
            return []
        *complete, remainder = self._buffer.split(b"\n")
        self._buffer = remainder
        return [entry.decode("utf-8", errors="replace").rstrip("\r") for entry in complete]


class TwitchChatProvider(ChatProvider):
    name = PROVIDER_TWITCH
    platform = PLATFORM
    #: Anonymous IRC gives no reliable join/leave stream, so presence is
    #: inferred from first message by the existing social layer instead.
    supports_presence_events = False

    def __init__(
        self,
        config: ChatProviderConfig | None = None,
        transport: Any = None,
        clock: Callable[[], datetime] = utc_now,
        monotonic: Callable[[], float] = time.monotonic,
        token_reader: Callable[[Any], str] = read_token,
    ) -> None:
        self.config = config or ChatProviderConfig(provider=PROVIDER_TWITCH)
        self.ingest_capacity = max(1, int(self.config.ingest_capacity))
        self.clock = clock
        self.monotonic = monotonic
        self._token_reader = token_reader
        self._transport_factory = transport or self._default_transport
        self._transport: Optional[Any] = None
        self._state = ProviderState.DISCONNECTED
        self._nickname = self.config.nickname or anonymous_nickname()
        self._token = ""
        self.supports_presence_events = TwitchChatProvider.supports_presence_events
        self._authenticated = False
        self._channels_joined: tuple[str, ...] = ()
        self._pending_join_ack = 0
        self._seen_ids: Deque[str] = deque(maxlen=MAX_DEDUP_IDS)
        self._seen_id_set: set[str] = set()
        self._connect_attempts = 0
        #: Consecutive attempts inside the *current* outage. The reconnect
        # budget bounds one outage, not the lifetime of the process, so a
        # long-running stream never degrades permanently after honest
        # reconnects spread over hours.
        self._outage_attempts = 0
        self._reconnect_count = 0
        self._next_attempt_at: Optional[float] = None
        self._connected_at: Optional[datetime] = None
        self._last_error: Optional[str] = None
        self._last_message_at: Optional[datetime] = None
        self._last_stale_at: Optional[datetime] = None
        self._last_stale_age_seconds: Optional[float] = None
        self._last_stale_message_id: Optional[str] = None
        self._last_notice: Optional[str] = None
        self._auth_failed = False
        self._counters = {
            "lines_read": 0,
            "messages": 0,
            "duplicated": 0,
            "stale": 0,
            "dropped_overflow": 0,
            "notices": 0,
            "malformed": 0,
            "pings": 0,
        }

    @property
    def state(self) -> ProviderState:
        return self._state

    @property
    def nickname(self) -> str:
        return self._nickname

    @property
    def reconnect_policy(self) -> ReconnectPolicy:
        return self.config.reconnect

    def __repr__(self) -> str:  # never surface credentials
        return (
            f"TwitchChatProvider(channel={self.config.channel!r}, "
            f"nickname={self._nickname!r}, state={self._state.value!r})"
        )

    def channels(self) -> tuple[str, ...]:
        return tuple(self._channels_joined or self.config.channels)

    def _default_transport(self) -> ChatTransport:
        return SocketChatTransport(
            host=self.config.server_host,
            port=self.config.server_port,
            connect_timeout_seconds=self.config.connect_timeout_seconds,
            read_timeout_seconds=self.config.socket_read_seconds,
            use_tls=self.config.use_tls,
        )

    # -- lifecycle ---------------------------------------------------------

    def connect(self) -> None:
        if self._state in {ProviderState.CONNECTED, ProviderState.CONNECTING}:
            return
        policy = self.reconnect_policy
        if self._auth_failed:
            self._state = ProviderState.DEGRADED
            raise ChatProviderError("authentication_failed", self._last_error or "")
        if not policy.allows_attempt(self._outage_attempts + 1):
            self._state = ProviderState.DEGRADED
            raise ChatProviderError("reconnect_budget_exhausted")
        self._outage_attempts += 1
        self._connect_attempts += 1
        self._state = (
            ProviderState.CONNECTING
            if self._connect_attempts == 1
            else ProviderState.RETRYING
        )
        transport = self._transport_factory()
        try:
            transport.connect()
        except ChatProviderError:
            self._fail_transport(transport, "connection_failed")
            raise ChatProviderError("connection_failed", self._last_error or "") from None
        except (TimeoutError, OSError) as exc:
            self._fail_transport(transport, type(exc).__name__)
            raise ChatProviderError("connection_failed", type(exc).__name__) from None

        self._transport = transport
        try:
            self._send("CAP REQ :twitch.tv/tags twitch.tv/commands")
            token = self._token = self._read_token()
            if token:
                self._send(f"PASS oauth:{token}")
            self._send(f"NICK {self._nickname}")
        except ChatProviderError:
            transport.close()
            self._transport = None
            self._fail_transport(None, "handshake_failed")
            raise ChatProviderError("handshake_failed", self._last_error or "") from None

    def disconnect(self) -> None:
        transport, self._transport = self._transport, None
        if transport is not None:
            try:
                self._send_with(transport, "QUIT")
            except Exception:
                pass
            try:
                transport.close()
            except Exception:
                pass
        self._state = ProviderState.STOPPED
        self._authenticated = False
        self._channels_joined = ()

    # -- ingest ------------------------------------------------------------

    def poll(self) -> list[ChatSourceEvent]:
        if self._state is ProviderState.STOPPED or self._auth_failed:
            return []
        if self._transport is None:
            self._reconnect_if_due()
            if self._transport is None:
                return []
        events: list[ChatSourceEvent] = []
        try:
            lines = self._transport.read_lines()
        except ChatProviderError as exc:
            self._handle_drop(exc.reason)
            return self._bound(events)
        except Exception as exc:  # pragma: no cover - defensive
            self._handle_drop(type(exc).__name__)
            return self._bound(events)

        for line in lines:
            self._counters["lines_read"] += 1
            parsed = parse_irc_line(line)
            if parsed is None:
                self._counters["malformed"] += 1
                continue
            try:
                event = self._handle(parsed)
            except ChatProviderError as exc:
                self._handle_drop(exc.reason)
                break
            if event is not None:
                events.append(event)
        return self._bound(events)

    def _handle(self, message: IrcMessage) -> Optional[ChatSourceEvent]:
        command = message.command
        if command == "PING":
            self._counters["pings"] += 1
            self._send(f"PONG :{message.trailing or 'tmi.twitch.tv'}")
            return None
        if command == "001":
            self._on_registered()
            return None
        if command in {"NOTICE", "WARN"}:
            self._counters["notices"] += 1
            self._last_notice = self._scrub(message.trailing)[:MAX_ERROR_DETAIL]
            self._inspect_notice(message.trailing)
            return None
        if command in {"JOIN", "PART"}:
            # Twitch does not deliver these on an anonymous read connection; if
            # a token'd connection does, map them without inventing anything.
            self.supports_presence_events = True  # observed, not assumed
            channel = message.params[0] if message.params else None
            viewer_login = message.nick
            if not viewer_login:
                return None
            return ChatSourceEvent(
                kind=(
                    SourceEventKind.VIEWER_JOINED
                    if command == "JOIN"
                    else SourceEventKind.VIEWER_LEFT
                ),
                viewer_id=f"{PLATFORM}:{viewer_login}",
                viewer_name=viewer_login,
                platform=PLATFORM,
                channel=(channel or "").lstrip("#") or None,
                timestamp=self.clock(),
            )
        if command != "PRIVMSG":
            return None

        tags = message.tags
        login = message.nick or tags.get("login", "")
        user_id = tags.get("user-id") or login
        text = message.trailing
        if not user_id or not text.strip():
            self._counters["malformed"] += 1
            return None
        if login and login.casefold() == self._nickname.casefold():
            return None

        message_id = tags.get("id") or None
        if message_id and self._already_seen(message_id):
            self._counters["duplicated"] += 1
            return None

        timestamp = epoch_ms_to_utc(tags.get("tmi-sent-ts", "")) or self.clock()
        if self._is_stale(timestamp):
            self._counters["stale"] += 1
            self._last_stale_at = timestamp
            self._last_stale_message_id = message_id
            self._last_stale_age_seconds = round(
                (self.clock() - timestamp).total_seconds(), 1
            )
            return None

        self._counters["messages"] += 1
        self._last_message_at = timestamp
        channel = message.params[0].lstrip("#") if message.params else None
        return ChatSourceEvent(
            kind=SourceEventKind.MESSAGE,
            viewer_id=f"{PLATFORM}:{user_id}",
            viewer_name=tags.get("display-name") or login,
            text=text[:MAX_ERROR_DETAIL * 5],
            message_id=message_id,
            timestamp=timestamp,
            platform=PLATFORM,
            channel=channel or self.config.channel,
            attributes={
                "login": login,
                "color": tags.get("color", ""),
                "first_msg": tags.get("first-msg", "") == "1",
                "mod": tags.get("mod", "") == "1",
                "subscriber": tags.get("subscriber", "") == "1",
                "emotes": tags.get("emotes", ""),
                "room_id": tags.get("room-id", ""),
            },
        )

    def _bound(self, events: Sequence[ChatSourceEvent]) -> list[ChatSourceEvent]:
        capacity = self.ingest_capacity
        if len(events) <= capacity:
            return list(events)
        self._counters["dropped_overflow"] += len(events) - capacity
        return list(events[:capacity])

    def _on_registered(self) -> None:
        self._authenticated = True
        self._state = ProviderState.CONNECTED
        self._connected_at = self.clock()
        self._last_error = None
        if self._connect_attempts > 1:
            self._reconnect_count += 1
        # A healthy registration pays back the reconnect budget.
        self._outage_attempts = 0
        wanted = tuple(self.config.channels)
        self._pending_join_ack = len(wanted)
        for channel in wanted:
            self._send(f"JOIN #{channel}")
        self._channels_joined = wanted
        self._next_attempt_at = None

    def _inspect_notice(self, text: str) -> None:
        lowered = (text or "").lower()
        if "authentication failed" in lowered or "improperly identified" in lowered:
            self._auth_failed = True
            self._last_error = self._scrub("authentication_failed: " + lowered)
            self._state = ProviderState.DEGRADED
        elif "rate limit" in lowered or "slow mode" in lowered:
            self._last_error = self._scrub(lowered)[:MAX_ERROR_DETAIL]

    def _already_seen(self, message_id: str) -> bool:
        if message_id in self._seen_id_set:
            return True
        if len(self._seen_ids) == self._seen_ids.maxlen:
            self._seen_id_set.discard(self._seen_ids[0])
        self._seen_ids.append(message_id)
        self._seen_id_set.add(message_id)
        return False

    def _is_stale(self, timestamp: datetime) -> bool:
        limit = self.config.max_message_age_seconds
        if limit <= 0:
            return False
        return (self.clock() - timestamp) > timedelta(seconds=limit)

    # -- reconnection ------------------------------------------------------

    def _handle_drop(self, reason: str) -> None:
        transport, self._transport = self._transport, None
        if transport is not None:
            try:
                transport.close()
            except Exception:
                pass
        self._authenticated = False
        self._channels_joined = ()
        self._last_error = self._scrub(reason)[:MAX_ERROR_DETAIL]
        if self._state is ProviderState.STOPPED:
            return
        self._state = ProviderState.DISCONNECTED
        self._schedule_retry()

    def _fail_transport(self, transport: Any, reason: str) -> None:
        if transport is not None:
            try:
                transport.close()
            except Exception:
                pass
        self._last_error = self._scrub(reason)[:MAX_ERROR_DETAIL]
        self._schedule_retry()

    def _schedule_retry(self) -> None:
        policy = self.reconnect_policy
        if not policy.allows_attempt(self._outage_attempts + 1):
            self._state = ProviderState.DEGRADED
            self._next_attempt_at = None
            return
        delay = policy.delay_for_attempt(self._outage_attempts + 1)
        self._next_attempt_at = self.monotonic() + delay
        self._state = ProviderState.RETRYING

    def _reconnect_if_due(self) -> None:
        if self._next_attempt_at is None:
            return
        if self.monotonic() < self._next_attempt_at:
            return
        try:
            self.connect()
        except ChatProviderError as exc:
            if exc.reason == "authentication_failed":
                self._auth_failed = True
            self._last_error = self._scrub(exc.reason)[:MAX_ERROR_DETAIL]

    # -- helpers -----------------------------------------------------------

    def _send(self, line: str) -> None:
        if self._transport is None:
            raise ChatProviderError("not_connected")
        self._send_with(self._transport, line)

    @staticmethod
    def _send_with(transport: Any, line: str) -> None:
        # Read-only guarantee: nothing on this connection can address a channel.
        if not line.startswith(SEND_PREFIX_WHITELIST):
            raise ChatProviderError("command_not_allowed")
        transport.send(line)

    def _read_token(self) -> str:
        try:
            return self._token_reader(self.config.token_path) or ""
        except Exception:
            return ""

    def _scrub(self, text: str) -> str:
        """Remove any token material that could have echoed back from the server."""
        lowered = text or ""
        if self._token and self._token in lowered:
            lowered = lowered.replace(self._token, "[redacted]")
        return OAUTH_RE.sub("oauth:[redacted]", lowered).replace("\n", " ").strip()

    def diagnostics(self) -> dict[str, Any]:
        payload = super().diagnostics()
        payload.update(
            {
                "server": f"{self.config.server_host}:{self.config.server_port}",
                "tls": self.config.use_tls,
                "nickname": self._nickname,
                "authenticated": self._authenticated,
                "anonymous": not bool(self._token),
                "connect_attempts": self._connect_attempts,
                "reconnect_count": self._reconnect_count,
                "connected_at": (
                    self._connected_at.isoformat(timespec="seconds").replace("+00:00", "Z")
                    if self._connected_at
                    else None
                ),
                "last_message_at": (
                    self._last_message_at.isoformat(timespec="seconds").replace("+00:00", "Z")
                    if self._last_message_at
                    else None
                ),
                "last_stale_message_at": (
                    self._last_stale_at.isoformat(timespec="seconds").replace("+00:00", "Z")
                    if self._last_stale_at
                    else None
                ),
                "last_stale_age_seconds": self._last_stale_age_seconds,
                "last_stale_message_id": self._last_stale_message_id,
                "last_notice": self._last_notice,
                "next_retry_in_seconds": (
                    round(max(0.0, self._next_attempt_at - self.monotonic()), 2)
                    if self._next_attempt_at is not None
                    else None
                ),
                "last_error": self._last_error,
                "counters": dict(self._counters),
            }
        )
        return payload

    def _token_present(self) -> bool:
        return bool(self._token)
