"""PHASE 12 deterministic chat provider tests.

Nothing here touches the network: the real platform provider is exercised
through a scripted transport that replays captured IRC lines.
"""

from __future__ import annotations

import json
import socket
import tempfile
import threading
import time
import unittest
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from backend.brain.loop import AutonomousBrain, BrainConfig
from backend.brain.models import ActionType, ActionResult, ExecutionStatus
from backend.brain.reasoning import MockBrainLLMProvider
from backend.brain.world_state import build_world_state
from backend.chat.models import ChatIntent, ChatStatus
from backend.chat.providers import (
    PROVIDER_DISABLED,
    PROVIDER_MOCK,
    PROVIDER_TWITCH,
    ChatProvider,
    ChatProviderConfig,
    ChatProviderError,
    ChatSourceEvent,
    ChatSourceRunner,
    MockChatProvider,
    ProviderState,
    ReconnectPolicy,
    SourceEventKind,
    TwitchChatProvider,
    build_provider,
)
from backend.chat.providers.base import MAX_EVENT_TEXT_LENGTH
from backend.chat.providers.config import read_token
from backend.chat.providers.twitch import (
    SocketChatTransport,
    parse_irc_line,
    unescape_tag_value,
)
from backend.chat.service import ChatService, ChatServiceConfig
from backend.llm.mock import MockLLMProvider
from backend.runtime import instrument
from backend.runtime.metrics import MetricsRegistry
from backend.speech.service import SpeechService
from backend.tts.mock import MockTTSProvider
from backend.tests.dispatch_support import confirm


BASE_TIME = datetime(2026, 9, 16, 9, 0, tzinfo=timezone.utc)
SECRET_TOKEN = "sup3rsecrettokenvalue"
WELCOME = ":tmi.twitch.tv 001 justinfan123456 :Welcome, GLHF!"


class Clock:
    def __init__(self, start: datetime = BASE_TIME) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


class Monotonic:
    def __init__(self) -> None:
        self.value = 1000.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class FakeTransport:
    """Stands in for the TLS socket; replays scripted IRC batches.

    Once the script runs out a transport either goes quiet (the default) or,
    with ``die_after_script``, raises - which is how a dropped connection and
    the resulting reconnect get tested without a network.
    """

    def __init__(
        self,
        batches: Iterable[Iterable[str]] = (),
        *,
        connect_error: Optional[BaseException] = None,
        read_error: Optional[BaseException] = None,
        die_after_script: bool = False,
    ) -> None:
        self.batches: deque[list[str]] = deque([list(b) for b in batches])
        self.sent: list[str] = []
        self.connect_error = connect_error
        self.read_error = read_error
        self.die_after_script = die_after_script
        self.connected = connect_error is None
        self.closed = False
        self.connect_calls = 0

    def connect(self) -> None:
        self.connect_calls += 1
        if self.connect_error is not None:
            raise self.connect_error
        self.connected = True

    def send(self, line: str) -> None:
        self.sent.append(line)

    def read_lines(self) -> list[str]:
        if not self.connected:
            raise ChatProviderError("not_connected")
        if self.batches:
            return self.batches.popleft()
        if self.die_after_script:
            raise self.read_error or ConnectionResetError("reset")
        return []

    def close(self) -> None:
        self.connected = False
        self.closed = True


class TransportFactory:
    def __init__(self, transports: list[FakeTransport]) -> None:
        self.transports = list(transports)
        self.created: list[FakeTransport] = []
        self.calls = 0

    def __call__(self) -> FakeTransport:
        self.calls += 1
        if not self.transports:
            raise ConnectionRefusedError("no transport left")
        transport = self.transports.pop(0)
        self.created.append(transport)
        return transport


def privmsg(
    text: str,
    *,
    login: str = "quezi_x",
    user_id: str = "1337122639",
    channel: str = "xqc",
    message_id: str = "dcf0357e-ff4f-49eb-ba61-e916c1adf628",
    timestamp: datetime = BASE_TIME,
    display_name: Optional[str] = None,
) -> str:
    """A line in the exact shape observed from the live Twitch IRC endpoint."""
    tags = ";".join(
        [
            "badge-info=subscriber/9",
            "badges=subscriber/6,premium",
            "color=#E0CCFF",
            f"display-name={display_name or login}",
            "emotes=;first-msg=0;flags=",
            f"id={message_id}",
            "mod=0;returning-chatter=0",
            "room-id=71092938",
            "subscriber=1",
            f"tmi-sent-ts={int(timestamp.timestamp() * 1000)}",
            "turbo=0",
            f"user-id={user_id}",
            "user-type=",
        ]
    )
    return f"@{tags} :{login}!{login}@{login}.tmi.twitch.tv PRIVMSG #{channel} :{text}"


def make_config(**overrides: Any) -> ChatProviderConfig:
    values: dict[str, Any] = dict(
        provider=PROVIDER_TWITCH,
        channels=("ironmouse",),
        nickname="justinfan123456",
    )
    values.update(overrides)
    return ChatProviderConfig(**values)


def make_provider(
    transports: list[FakeTransport],
    clock: Optional[Clock] = None,
    monotonic: Optional[Monotonic] = None,
    token: str = "",
    **config_overrides: Any,
) -> tuple[TwitchChatProvider, TransportFactory]:
    config = make_config(**config_overrides)
    factory = TransportFactory(transports)
    provider = TwitchChatProvider(
        config,
        transport=factory,
        clock=clock or Clock(),
        monotonic=monotonic or Monotonic(),
        token_reader=lambda path: token,
    )
    return provider, factory


def registered_provider(
    extra_batches: Iterable[Iterable[str]] = (),
    **kwargs: Any,
) -> tuple[TwitchChatProvider, FakeTransport, Clock]:
    """A provider that completed its handshake, ready to deliver messages."""
    clock = kwargs.pop("clock", None) or Clock()
    transport = FakeTransport([[WELCOME], *extra_batches])
    provider, _ = make_provider([transport], clock=clock, **kwargs)
    provider.connect()
    provider.poll()
    assert provider.state is ProviderState.CONNECTED
    return provider, transport, clock


def make_service(clock: Optional[Clock] = None, **config: Any) -> ChatService:
    config.setdefault("max_messages_per_rate_window", 5)
    return ChatService(MockLLMProvider(), ChatServiceConfig(**config), clock=clock or Clock())


def make_mock_runner(
    provider: MockChatProvider,
    service: ChatService,
    clock: Clock,
) -> ChatSourceRunner:
    return ChatSourceRunner(provider, service, clock=clock, poll_interval_seconds=0.01)


class ProviderAbstractionTests(unittest.TestCase):
    def test_chat_provider_is_an_abstract_boundary(self) -> None:
        with self.assertRaises(TypeError):
            ChatProvider()  # type: ignore[abstract]

    def test_partial_implementations_are_rejected(self) -> None:
        class Incomplete(ChatProvider):
            def connect(self) -> None:
                return None

        with self.assertRaises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_twitch_and_mock_share_one_contract(self) -> None:
        twitch = build_provider(make_config())
        mock = build_provider(make_config(provider=PROVIDER_MOCK))
        assert twitch is not None and mock is not None
        for provider in (twitch, mock):
            self.assertIsInstance(provider, ChatProvider)
            self.assertIn(provider.state, set(ProviderState))
            self.assertIsInstance(provider.diagnostics(), dict)
            self.assertIsInstance(provider.poll(), list)

    def test_event_rejects_oversized_text(self) -> None:
        with self.assertRaises(ValueError):
            ChatSourceEvent(
                kind=SourceEventKind.MESSAGE,
                viewer_id="twitch:1",
                text="x" * (MAX_EVENT_TEXT_LENGTH + 1),
            )

    def test_event_requires_a_known_kind(self) -> None:
        with self.assertRaises(ValueError):
            ChatSourceEvent(kind="MESSAGE")  # type: ignore[arg-type]

    def test_event_dict_carries_provenance_not_secrets(self) -> None:
        event = ChatSourceEvent(
            kind=SourceEventKind.MESSAGE,
            viewer_id="twitch:1337122639",
            viewer_name="quezi_x",
            text="hello",
            message_id="abc",
            timestamp=BASE_TIME,
            platform="twitch",
            channel="ironmouse",
            attributes={"token": SECRET_TOKEN},
        )
        payload = event.to_dict()
        self.assertEqual(payload["viewer_name"], "quezi_x")
        self.assertEqual(payload["channel"], "ironmouse")
        self.assertNotIn("attributes", payload)
        self.assertNotIn(SECRET_TOKEN, json.dumps(payload))

    def test_twitch_does_not_fabricate_presence_events(self) -> None:
        self.assertFalse(TwitchChatProvider.supports_presence_events)
        self.assertTrue(MockChatProvider.supports_presence_events)


class ReconnectPolicyTests(unittest.TestCase):
    def test_backoff_is_exponential_and_bounded(self) -> None:
        policy = ReconnectPolicy(base_delay_seconds=1.0, max_delay_seconds=8.0, max_attempts=9)
        self.assertEqual(
            [policy.delay_for_attempt(attempt) for attempt in range(1, 7)],
            [1.0, 2.0, 4.0, 8.0, 8.0, 8.0],
        )

    def test_attempts_are_bounded(self) -> None:
        policy = ReconnectPolicy(max_attempts=3)
        self.assertTrue(policy.allows_attempt(3))
        self.assertFalse(policy.allows_attempt(4))

    def test_invalid_policy_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ReconnectPolicy(base_delay_seconds=0)
        with self.assertRaises(ValueError):
            ReconnectPolicy(max_attempts=0)


class ConfigTests(unittest.TestCase):
    def test_default_environment_disables_real_chat(self) -> None:
        config = ChatProviderConfig.from_environment({})
        self.assertEqual(config.provider, PROVIDER_DISABLED)
        self.assertFalse(config.enabled)
        self.assertIsNone(build_provider(config))

    def test_unknown_provider_name_degrades_to_disabled(self) -> None:
        config = ChatProviderConfig.from_environment({"CHAT_PROVIDER": "kick"})
        self.assertEqual(config.provider, PROVIDER_DISABLED)

    def test_mock_provider_is_buildable(self) -> None:
        self.assertIsInstance(
            build_provider(ChatProviderConfig(provider=PROVIDER_MOCK)), MockChatProvider
        )

    def test_sanitized_config_never_exposes_the_credential_file(self) -> None:
        config = ChatProviderConfig(
            provider=PROVIDER_TWITCH,
            token_path=Path("C:/secret/place/chat_twitch_token.txt"),
        )
        self.assertNotIn("secret/place", json.dumps(config.sanitized().to_dict()))
        self.assertEqual(str(config.sanitized().token_path), "chat_twitch_token.txt")
        self.assertNotIn(SECRET_TOKEN, json.dumps(config.to_dict()))

    def test_token_file_reading_is_optional_and_fail_soft(self) -> None:
        self.assertEqual(read_token(Path("C:/definitely/not/here.txt")), "")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "token.txt"
            path.write_text(SECRET_TOKEN + "\n", encoding="utf-8")
            self.assertEqual(read_token(path), SECRET_TOKEN)


class IrcParsingTests(unittest.TestCase):
    def test_parses_a_real_privmsg(self) -> None:
        message = parse_irc_line(privmsg("Hello Lacrimosa"))
        self.assertIsNotNone(message)
        assert message is not None
        self.assertEqual(message.command, "PRIVMSG")
        self.assertEqual(message.nick, "quezi_x")
        self.assertEqual(message.params[0], "#xqc")
        self.assertEqual(message.trailing, "Hello Lacrimosa")
        self.assertEqual(message.tags["user-id"], "1337122639")
        self.assertEqual(message.tags["display-name"], "quezi_x")

    def test_trailing_is_never_split_on_colons(self) -> None:
        message = parse_irc_line(privmsg("dance: now :please:"))
        assert message is not None
        self.assertEqual(message.trailing, "dance: now :please:")

    def test_unescapes_tag_values(self) -> None:
        # IRCv3: \: -> ';'  \s -> ' '  \\ -> '\'
        self.assertEqual(unescape_tag_value("a\\:b\\s\\\\c"), "a;b \\c")

    def test_malformed_input_returns_none(self) -> None:
        for line in ("", "   ", "@tags-only-no-space", ":prefix-only "):
            self.assertIsNone(parse_irc_line(line), line)

    def test_bare_command_parses_without_params(self) -> None:
        message = parse_irc_line("QUIT")
        assert message is not None
        self.assertEqual(message.command, "QUIT")
        self.assertEqual(message.params, ())
        self.assertEqual(message.trailing, "")


class SocketTransportTests(unittest.TestCase):
    """The real socket wrapper, driven with a fake socket (still no network)."""

    class FakeSocket:
        def __init__(self, results: Iterable[Any]) -> None:
            self.results = list(results)
            self.closed = False
            self.timeout: Optional[float] = None

        def recv(self, _size: int) -> bytes:
            outcome = self.results.pop(0) if self.results else b""
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

        def settimeout(self, value: Optional[float]) -> None:
            self.timeout = value

        def close(self) -> None:
            self.closed = True

    def transport(self, results: Iterable[Any]) -> SocketChatTransport:
        transport = SocketChatTransport("host", 443, 1.0, 0.25)
        transport._sock = self.FakeSocket(results)
        return transport

    def test_a_quiet_socket_does_not_look_like_a_dropped_connection(self) -> None:
        transport = self.transport([socket.timeout(), socket.timeout(), socket.timeout()])
        for _ in range(3):
            self.assertEqual(transport.read_lines(), [])

    def test_real_lines_still_arrive_between_timeouts(self) -> None:
        transport = self.transport(
            [socket.timeout(), b":a!a@a.tmi.twitch.tv PRIVMSG #c :hi\r\n", b""]
        )
        self.assertEqual(transport.read_lines(), [])
        self.assertEqual(len(transport.read_lines()), 1)
        with self.assertRaises(ChatProviderError) as caught:
            transport.read_lines()
        self.assertEqual(caught.exception.reason, "connection_closed")

    def test_truncated_line_is_not_delivered_as_if_complete(self) -> None:
        transport = self.transport([b":tmi.twitch.tv 001 nick :Wel", b""])
        self.assertEqual(transport.read_lines(), [])
        with self.assertRaises(ChatProviderError) as caught:
            transport.read_lines()
        self.assertEqual(caught.exception.reason, "connection_closed")

    def test_os_error_becomes_a_structured_transport_error(self) -> None:
        transport = self.transport([ConnectionResetError("reset")])
        with self.assertRaises(ChatProviderError) as caught:
            transport.read_lines()
        self.assertEqual(caught.exception.reason, "transport_error")

    def test_all_buffered_chunks_are_drained_in_one_call(self) -> None:
        # Twitch backpressures a single-recv reader onto a stale FIFO, so one
        # poll must empty everything already buffered on the socket.
        transport = self.transport(
            [
                b":a!a@a PRIVMSG #c :one\r\n",
                b":b!b@b PRIVMSG #c :two\r\n",
                b":c!c@c PRIVMSG #c :three\r\n",
                socket.timeout(),
                socket.timeout(),
            ]
        )
        lines = transport.read_lines()
        self.assertEqual(len(lines), 3)
        self.assertIn(":three", lines[-1])
        # The read timeout is restored so the next quiet poll still blocks
        # normally instead of spinning on a non-blocking socket.
        self.assertEqual(transport._sock.timeout, 0.25)
        self.assertEqual(transport.read_lines(), [])

    def test_drain_is_bounded_per_call(self) -> None:
        chunk = b"a" * 8190 + b"\r\n"
        transport = self.transport([chunk] * 40)
        per_call = SocketChatTransport.MAX_DRAIN_BYTES // len(chunk)
        lines = transport.read_lines()
        self.assertEqual(len(lines), per_call)
        # The rest is still readable on the following poll rather than lost.
        self.assertEqual(len(transport.read_lines()), 40 - per_call)


class TwitchIngestTests(unittest.TestCase):
    def test_connect_handshake_is_read_only_and_anonymous(self) -> None:
        provider, factory = make_provider([FakeTransport()])
        provider.connect()
        sent = factory.created[0].sent
        self.assertEqual(sent[0], "CAP REQ :twitch.tv/tags twitch.tv/commands")
        self.assertEqual(sent[1], "NICK justinfan123456")
        self.assertNotIn("PASS", " ".join(sent))
        self.assertEqual(provider.state, ProviderState.CONNECTING)
        self.assertFalse(any(line.startswith("PRIVMSG") for line in sent))

    def test_registration_joins_the_channel(self) -> None:
        transport = FakeTransport([[WELCOME]])
        provider, _ = make_provider([transport])
        provider.connect()
        self.assertEqual(list(provider.poll()), [])
        self.assertEqual(provider.state, ProviderState.CONNECTED)
        self.assertIn("JOIN #ironmouse", transport.sent)
        self.assertEqual(provider.channels(), ("ironmouse",))

    def test_token_is_sent_once_and_never_repeated(self) -> None:
        provider, factory = make_provider([FakeTransport()], token=SECRET_TOKEN)
        provider.connect()
        sent = factory.created[0].sent
        self.assertEqual([line for line in sent if line.startswith("PASS")], [f"PASS oauth:{SECRET_TOKEN}"])
        blob = json.dumps(provider.diagnostics()) + repr(provider) + str(provider)
        self.assertNotIn(SECRET_TOKEN, blob)
        self.assertNotIn("oauth:", blob)
        self.assertFalse(provider.diagnostics()["anonymous"])

    def test_real_message_maps_to_stable_viewer_identity(self) -> None:
        provider, _, clock = registered_provider([[privmsg("Hello Lacrimosa")]])
        events = list(provider.poll())
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event.viewer_id, "twitch:1337122639")
        self.assertEqual(event.viewer_name, "quezi_x")
        self.assertEqual(event.text, "Hello Lacrimosa")
        self.assertEqual(event.message_id, "dcf0357e-ff4f-49eb-ba61-e916c1adf628")
        self.assertEqual(event.platform, "twitch")
        self.assertEqual(event.channel, "xqc")
        self.assertEqual(event.timestamp, clock.now)
        self.assertTrue(event.is_message)

    def test_ping_is_answered_without_becoming_an_event(self) -> None:
        provider, transport, _ = registered_provider([["PING :tmi.twitch.tv"]])
        self.assertEqual(list(provider.poll()), [])
        self.assertEqual(transport.sent[-1], "PONG :tmi.twitch.tv")

    def test_duplicate_message_ids_are_dropped(self) -> None:
        line = privmsg("dance", message_id="same-id")
        provider, _, _ = registered_provider([[line], [line], [line]])
        counts = [len(provider.poll()) for _ in range(3)]
        self.assertEqual(counts, [1, 0, 0])
        self.assertEqual(provider.diagnostics()["counters"]["duplicated"], 2)

    def test_replayed_history_is_dropped_as_stale(self) -> None:
        old = privmsg("old news", message_id="old-1", timestamp=BASE_TIME - timedelta(minutes=10))
        provider, _, _ = registered_provider([[old]])
        self.assertEqual(list(provider.poll()), [])
        diag = provider.diagnostics()
        self.assertEqual(diag["counters"]["stale"], 1)
        # Phase 24: stalled feeds must be diagnosable from state, not logs.
        self.assertEqual(diag["last_stale_age_seconds"], 600.0)
        self.assertEqual(diag["last_stale_message_at"], "2026-09-16T08:50:00Z")
        self.assertEqual(diag["last_stale_message_id"], "old-1")

    def test_last_notice_is_exposed_scrubbed_and_bounded(self) -> None:
        notice = f"NOTICE * :bad credentials oauth:{SECRET_TOKEN} {'x' * 400}"
        provider, _, _ = registered_provider([[notice]])
        self.assertEqual(list(provider.poll()), [])
        diag = provider.diagnostics()
        self.assertEqual(diag["counters"]["notices"], 1)
        self.assertNotIn(SECRET_TOKEN, json.dumps(diag))
        self.assertIn("[redacted]", diag["last_notice"])
        self.assertLessEqual(len(diag["last_notice"]), 200)

    def test_malformed_and_empty_messages_are_ignored(self) -> None:
        provider, _, _ = registered_provider(
            [
                [
                    "@id=1;user-id=;tmi-sent-ts=1 :ghost!ghost@ghost.tmi.twitch.tv PRIVMSG #a :   ",
                    "@id=2;tmi-sent-ts=1 PRIVMSG #a :no prefix and no user-id",
                ]
            ]
        )
        self.assertEqual(list(provider.poll()), [])
        self.assertGreaterEqual(provider.diagnostics()["counters"]["malformed"], 2)

    def test_own_messages_are_never_ingested(self) -> None:
        provider, _, _ = registered_provider(
            [[privmsg("self", login="justinfan123456", message_id="self-1")]]
        )
        self.assertEqual(list(provider.poll()), [])
        self.assertEqual(provider.diagnostics()["counters"]["messages"], 0)

    def test_intake_per_poll_is_bounded(self) -> None:
        batch = [
            privmsg(f"message {index}", message_id=f"id-{index}") for index in range(10)
        ]
        provider, _, _ = registered_provider([batch], ingest_capacity=4)
        events = list(provider.poll())
        self.assertEqual(len(events), 4)
        self.assertEqual(provider.diagnostics()["counters"]["dropped_overflow"], 6)

    def test_presence_lines_are_mapped_when_the_platform_sends_them(self) -> None:
        provider, _, _ = registered_provider(
            [[":alice!alice@alice.tmi.twitch.tv JOIN #ironmouse"]]
        )
        events = list(provider.poll())
        self.assertEqual(events[0].kind, SourceEventKind.VIEWER_JOINED)
        self.assertEqual(events[0].viewer_id, "twitch:alice")
        self.assertTrue(provider.supports_presence_events)


class TwitchFailureTests(unittest.TestCase):
    def test_connection_failure_raises_and_schedules_a_bounded_retry(self) -> None:
        monotonic = Monotonic()
        provider, factory = make_provider(
            [
                FakeTransport(connect_error=ConnectionRefusedError("refused")),
                FakeTransport(),
            ],
            monotonic=monotonic,
        )
        with self.assertRaises(ChatProviderError) as ctx:
            provider.connect()
        self.assertEqual(ctx.exception.reason, "connection_failed")
        self.assertEqual(provider.state, ProviderState.RETRYING)
        # Still inside the backoff window: no new socket is opened.
        self.assertEqual(list(provider.poll()), [])
        self.assertEqual(factory.calls, 1)
        monotonic.advance(2.0)
        self.assertEqual(list(provider.poll()), [])
        self.assertEqual(factory.calls, 2)

    def test_reconnect_budget_is_exhausted_rather_than_looping(self) -> None:
        monotonic = Monotonic()
        provider, factory = make_provider(
            [
                FakeTransport(connect_error=ConnectionRefusedError("refused"))
                for _ in range(6)
            ],
            monotonic=monotonic,
            reconnect=ReconnectPolicy(base_delay_seconds=1.0, max_delay_seconds=2.0, max_attempts=2),
        )
        with self.assertRaises(ChatProviderError):
            provider.connect()
        for _ in range(6):
            monotonic.advance(30.0)
            provider.poll()
        self.assertEqual(factory.calls, 2)
        self.assertEqual(provider.state, ProviderState.DEGRADED)
        self.assertEqual(list(provider.poll()), [])
        self.assertEqual(factory.calls, 2)
        self.assertIn("reconnect", json.dumps(provider.diagnostics()))

    def test_authentication_failure_degrades_without_retrying(self) -> None:
        transport = FakeTransport(
            [[":tmi.twitch.tv NOTICE * :Login authentication failed"]]
        )
        provider, factory = make_provider([transport], token=SECRET_TOKEN)
        provider.connect()
        self.assertEqual(list(provider.poll()), [])
        self.assertEqual(provider.state, ProviderState.DEGRADED)
        self.assertEqual(list(provider.poll()), [])
        self.assertEqual(factory.calls, 1)
        diagnostics = provider.diagnostics()
        self.assertIn("authentication_failed", diagnostics["last_error"])
        self.assertNotIn(SECRET_TOKEN, json.dumps(diagnostics))

    def test_read_failure_then_clean_reconnect(self) -> None:
        clock = Clock()
        monotonic = Monotonic()
        broken = FakeTransport(die_after_script=True)
        healthy = FakeTransport(
            [[WELCOME], [privmsg("back again", message_id="fresh-1", timestamp=clock.now)]]
        )
        provider, _ = make_provider([broken, healthy], clock=clock, monotonic=monotonic)
        provider.connect()
        provider._state = ProviderState.CONNECTED
        self.assertEqual(list(provider.poll()), [])
        self.assertTrue(broken.closed)
        self.assertEqual(provider.state, ProviderState.RETRYING)
        monotonic.advance(5.0)
        self.assertEqual(list(provider.poll()), [])
        self.assertEqual(provider.state, ProviderState.CONNECTED)
        self.assertEqual(provider.diagnostics()["reconnect_count"], 1)
        events = list(provider.poll())
        self.assertEqual([event.text for event in events], ["back again"])

    def test_reconnect_does_not_redeliver_seen_messages(self) -> None:
        clock = Clock()
        monotonic = Monotonic()
        line = privmsg("dance", message_id="replayed", timestamp=clock.now)
        first = FakeTransport([[WELCOME], [line]], die_after_script=True)
        second = FakeTransport([[WELCOME], [line]])
        provider, _ = make_provider(
            [first, second], clock=clock, monotonic=monotonic
        )
        provider.connect()
        collected: list[str] = []
        for _ in range(5):
            monotonic.advance(30.0)
            collected.extend(event.text for event in provider.poll())
        self.assertEqual(collected, ["dance"])
        self.assertGreaterEqual(provider.diagnostics()["counters"]["duplicated"], 1)

    def test_healthy_sessions_repay_the_reconnect_budget(self) -> None:
        """Drops spread over hours must not burn a lifetime retry budget."""
        clock = Clock()
        monotonic = Monotonic()
        transports = [
            FakeTransport([[WELCOME]], die_after_script=True) for _ in range(5)
        ]
        provider, _ = make_provider(
            transports,
            clock=clock,
            monotonic=monotonic,
            reconnect=ReconnectPolicy(
                base_delay_seconds=1.0, max_delay_seconds=2.0, max_attempts=2
            ),
        )
        provider.connect()
        self.assertEqual(list(provider.poll()), [])
        self.assertEqual(provider.state, ProviderState.CONNECTED)
        for _ in range(4):
            # The platform drops the live connection.
            self.assertEqual(list(provider.poll()), [])
            self.assertEqual(provider.state, ProviderState.RETRYING)
            # ...and the next socket registers cleanly, paying the budget back.
            monotonic.advance(5.0)
            self.assertEqual(list(provider.poll()), [])
            self.assertEqual(provider.state, ProviderState.CONNECTED)
        diagnostics = provider.diagnostics()
        self.assertEqual(diagnostics["state"], "CONNECTED")
        self.assertGreaterEqual(diagnostics["reconnect_count"], 3)
        self.assertGreaterEqual(diagnostics["connect_attempts"], 4)

    def test_send_whitelist_blocks_platform_writes(self) -> None:
        provider, _ = make_provider([FakeTransport()])
        with self.assertRaises(ChatProviderError) as ctx:
            provider._send_with(FakeTransport(), "PRIVMSG #xqc :hello")
        self.assertEqual(ctx.exception.reason, "command_not_allowed")
        provider._send_with(FakeTransport(), "PONG :tmi.twitch.tv")
        provider._send_with(FakeTransport(), "JOIN #ironmouse")

    def test_disconnect_is_idempotent_and_stops_ingest(self) -> None:
        provider, factory = make_provider([FakeTransport()])
        provider.connect()
        provider.disconnect()
        provider.disconnect()
        self.assertEqual(provider.state, ProviderState.STOPPED)
        self.assertEqual(list(provider.poll()), [])
        self.assertTrue(factory.created[0].closed)


class MockProviderTests(unittest.TestCase):
    def test_mock_is_deterministic_and_scriptable(self) -> None:
        clock = Clock()
        provider = MockChatProvider(ChatProviderConfig(provider=PROVIDER_MOCK), clock=clock)
        provider.push_message("twitch:1", "hello", "m1")
        provider.push_message("twitch:2", "dance!", "m2")
        provider.connect()
        events = list(provider.poll())
        self.assertEqual([event.text for event in events], ["hello", "dance!"])
        self.assertEqual(list(provider.poll()), [])

    def test_mock_connect_failure_is_scriptable(self) -> None:
        provider = MockChatProvider()
        provider.fail_next_connect = 1
        with self.assertRaises(ChatProviderError) as ctx:
            provider.connect()
        self.assertEqual(ctx.exception.reason, "connection_refused")
        provider.connect()
        self.assertEqual(provider.state, ProviderState.CONNECTED)

    def test_mock_auth_failure_is_scriptable(self) -> None:
        provider = MockChatProvider()
        provider.require_auth = True
        with self.assertRaises(ChatProviderError) as ctx:
            provider.connect()
        self.assertEqual(ctx.exception.reason, "authentication_failed")
        self.assertEqual(provider.state, ProviderState.DEGRADED)

    def test_mock_holds_a_persistent_outage(self) -> None:
        provider = MockChatProvider()
        provider.connect()
        provider.lose_connection()
        self.assertEqual(list(provider.poll()), [])
        self.assertEqual(provider.state, ProviderState.DISCONNECTED)


class RunnerIngestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = Clock()
        self.service = make_service(self.clock)

    def make_provider(self) -> MockChatProvider:
        provider = MockChatProvider(clock=self.clock)
        provider.connect()
        return provider

    def test_message_lands_in_the_existing_bounded_queue(self) -> None:
        provider = self.make_provider()
        provider.push_message("twitch:42", "Lacrimosa can you dance?", "real-1", "quezi_x")
        runner = make_mock_runner(provider, self.service, self.clock)
        report = runner.run_once()
        self.assertEqual(report.accepted, 1)
        queued = self.service.get_queue()
        self.assertEqual(len(queued), 1)
        self.assertEqual(queued[0].viewer_id, "twitch:42")
        self.assertEqual(queued[0].intent, ChatIntent.REQUEST_ACTION)
        self.assertEqual(queued[0].target, "dance")
        self.assertEqual(queued[0].platform, "mock")
        self.assertEqual(queued[0].viewer_name, "quezi_x")
        self.assertEqual(queued[0].message_id, "real-1")
        self.assertEqual(queued[0].status, ChatStatus.QUEUED)

    def test_duplicate_ids_from_the_platform_are_ingested_once(self) -> None:
        provider = self.make_provider()
        provider.push_message("twitch:1", "hello there", "dup-1")
        runner = make_mock_runner(provider, self.service, self.clock)
        self.assertEqual(runner.run_once().accepted, 1)
        provider.push_message("twitch:1", "hello there", "dup-1")
        self.assertEqual(runner.run_once().accepted, 0)
        self.assertEqual(len(self.service.get_queue()), 1)
        self.assertEqual(runner.get_state()["counters"]["duplicate_ids"], 1)

    def test_rate_limiting_still_comes_from_chat_service(self) -> None:
        provider = self.make_provider()
        for index in range(9):
            provider.push_message("twitch:7", f"dance please {index}", f"rl-{index}")
        runner = make_mock_runner(provider, self.service, self.clock)
        report = runner.run_once()
        self.assertEqual(report.accepted, 5)
        self.assertEqual(report.filtered, 4)
        reasons = {
            result.message.filtered_reason
            for result in report.submissions
            if not result.accepted
        }
        self.assertEqual(reasons, {"rate_limited"})

    def test_queue_overflow_is_a_safe_rejection(self) -> None:
        service = make_service(self.clock, queue_capacity=2)
        provider = self.make_provider()
        for index in range(5):
            provider.push_message(f"twitch:{index}", f"hello there {index}", f"of-{index}")
        runner = make_mock_runner(provider, service, self.clock)
        report = runner.run_once()
        self.assertEqual(report.accepted, 2)
        self.assertEqual(report.rejected, 3)
        self.assertEqual(service.get_state()["queued_count"], 2)
        self.assertEqual(runner.get_state()["counters"]["rejected"], 3)
        self.assertEqual(
            [
                result.message.filtered_reason
                for result in report.submissions
                if not result.accepted
            ],
            ["queue_full"] * 3,
        )

    def test_intake_per_pass_is_bounded(self) -> None:
        provider = self.make_provider()
        provider.ingest_capacity = 3
        for index in range(8):
            provider.push_message(f"twitch:{index}", f"hello there {index}", f"cap-{index}")
        runner = make_mock_runner(provider, self.service, self.clock)
        report = runner.run_once()
        self.assertEqual(report.events, 3)
        self.assertEqual(runner.get_state()["counters"]["dropped_overflow"], 5)

    def test_two_viewers_stay_isolated(self) -> None:
        provider = self.make_provider()
        provider.push_message("twitch:11", "Lacrimosa can you dance?", "a1", "alice")
        provider.push_message("twitch:22", "hello", "b1", "bob")
        runner = make_mock_runner(provider, self.service, self.clock)
        runner.run_once()
        queue = self.service.get_queue()
        self.assertEqual({m.viewer_id for m in queue}, {"twitch:11", "twitch:22"})
        processed = self.service.process_next()
        assert processed is not None
        self.assertEqual(processed.message.viewer_id, "twitch:11")
        turns = self.service.conversation.snapshot()
        self.assertEqual({turn.viewer_id for turn in turns}, {"twitch:11"})
        # alice and bob are remembered by priority order, not mixed together.
        self.assertEqual(queue[0].viewer_id, "twitch:11")

    def test_same_text_from_two_viewers_is_not_a_duplicate(self) -> None:
        provider = self.make_provider()
        provider.push_message("twitch:11", "hello there", "x1", "alice")
        provider.push_message("twitch:22", "hello there", "x2", "bob")
        runner = make_mock_runner(provider, self.service, self.clock)
        self.assertEqual(runner.run_once().accepted, 2)

    def test_non_message_events_do_not_create_chat(self) -> None:
        provider = self.make_provider()
        provider.push_message("twitch:5", "", "p1", "carol", kind=SourceEventKind.VIEWER_JOINED)
        runner = make_mock_runner(provider, self.service, self.clock)
        report = runner.run_once()
        self.assertEqual(report.ignored, 1)
        self.assertEqual(report.accepted, 0)
        self.assertEqual(self.service.get_queue(), [])
        self.assertIn("twitch:5", runner.get_state()["active_viewers"])

    def test_blank_and_unnamed_events_are_invalid_not_fatal(self) -> None:
        provider = self.make_provider()
        provider.push(ChatSourceEvent(kind=SourceEventKind.MESSAGE, viewer_id="", text="hi"))
        provider.push(
            ChatSourceEvent(kind=SourceEventKind.MESSAGE, viewer_id="twitch:1", text="   ")
        )
        runner = make_mock_runner(provider, self.service, self.clock)
        report = runner.run_once()
        self.assertEqual(report.invalid, 2)
        self.assertEqual(report.accepted, 0)

    def test_provider_failure_degrades_chat_but_never_raises(self) -> None:
        provider = self.make_provider()
        provider.fail_next_poll = 1
        runner = make_mock_runner(provider, self.service, self.clock)
        report = runner.run_once()
        self.assertEqual(report.errors, 1)
        state = runner.get_state()
        self.assertEqual(state["counters"]["provider_errors"], 1)
        self.assertIsNotNone(state["last_error"])
        # Chat input recovers, and local simulated chat keeps working throughout.
        provider.push_message("twitch:1", "Lacrimosa can you dance?", "after-failure")
        self.assertEqual(runner.run_once().accepted, 1)
        self.assertTrue(self.service.submit("viewer_local", "hello").accepted)

    def test_persistent_outage_stays_visible_in_diagnostics(self) -> None:
        provider = self.make_provider()
        runner = make_mock_runner(provider, self.service, self.clock)
        provider.lose_connection()
        report = runner.run_once()
        self.assertEqual(report.errors, 0)
        state = runner.get_state()
        self.assertEqual(state["state"], "DISCONNECTED")
        self.assertEqual(state["connected"], False)
        self.assertFalse(runner.health()["ok"])

    def test_background_loop_survives_a_failed_connect(self) -> None:
        provider = MockChatProvider(clock=self.clock)
        provider.fail_next_connect = 1
        service = self.service
        runner = ChatSourceRunner(provider, service, clock=self.clock, poll_interval_seconds=0.01)
        self.addCleanup(runner.stop)
        runner.start()
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            state = runner.get_state()
            if state["counters"]["provider_errors"] and state["connected"]:
                break
            time.sleep(0.01)
        state = runner.get_state()
        self.assertEqual(state["counters"]["provider_errors"], 1)
        self.assertTrue(runner._thread.is_alive())
        self.assertTrue(state["connected"])
        # The Brain's own input path is unaffected by the source being down.
        service.submit("viewer_local", "Lacrimosa can you dance?")
        self.assertEqual(service.get_state()["queued_count"], 1)

    def test_background_loop_ingests_and_stops_cleanly(self) -> None:
        provider = MockChatProvider(clock=self.clock)
        provider.push_message("twitch:1", "hello", "bg-1")
        runner = ChatSourceRunner(provider, self.service, clock=self.clock, poll_interval_seconds=0.01)
        runner.start()
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not self.service.get_queue():
            time.sleep(0.01)
        self.assertEqual(len(self.service.get_queue()), 1)
        runner.stop()
        self.assertFalse(runner.get_state()["running"])
        self.assertIsNone(runner._thread)
        self.assertEqual(provider.state, ProviderState.STOPPED)

    def test_diagnostics_report_every_required_field_without_secrets(self) -> None:
        provider = self.make_provider()
        provider.push_message("twitch:3", "hello", "d1", "dave")
        runner = make_mock_runner(provider, self.service, self.clock)
        runner.run_once()
        state = runner.get_state()
        for key in (
            "state",
            "platform",
            "channels",
            "message_rate_per_minute",
            "queue",
            "last_message",
            "active_viewers",
            "reconnect_count",
            "last_error",
        ):
            self.assertIn(key, state)
        self.assertEqual(state["platform"], "mock")
        self.assertEqual(state["last_message"]["viewer_name"], "dave")
        self.assertEqual(state["queue"]["queued_count"], 1)
        self.assertEqual(state["message_rate_per_minute"], 1.0)
        self.assertNotIn(SECRET_TOKEN, json.dumps(state))
        self.assertNotIn("token", json.dumps(state))
        health = runner.health()
        self.assertTrue(health["ok"])
        self.assertEqual(health["provider"], "mock")

    def test_stop_is_safe_before_start(self) -> None:
        provider = self.make_provider()
        runner = make_mock_runner(provider, self.service, self.clock)
        runner.stop()
        self.assertEqual(runner.get_state()["running"], False)


class RunnerThroughTwitchTests(unittest.TestCase):
    def test_captured_platform_lines_reach_the_chat_queue(self) -> None:
        clock = Clock()
        provider, _, _ = registered_provider(
            [
                [
                    privmsg("Lacrimosa can you dance?", message_id="t1", timestamp=clock.now),
                    privmsg("hello", login="bob", user_id="99", message_id="t2", timestamp=clock.now),
                ]
            ],
            clock=clock,
        )
        service = make_service(clock)
        runner = ChatSourceRunner(provider, service, clock=clock, poll_interval_seconds=0.01)
        self.addCleanup(runner.stop)
        report = runner.run_once()
        self.assertEqual(report.accepted, 2)
        self.assertEqual(
            {message.viewer_id for message in service.get_queue()},
            {"twitch:1337122639", "twitch:99"},
        )
        self.assertEqual(service.get_state()["queued_count"], 2)
        state = runner.get_state()
        self.assertEqual(state["platform"], "twitch")
        self.assertEqual(state["connected"], True)
        self.assertEqual(state["last_message"]["viewer_name"], "bob")

    def test_reconnect_replay_neither_loses_nor_backdates_messages(self) -> None:
        """PA29-006: a reconnect may replay lines; the queue must stay honest.

        The already-seen line is dropped as a duplicate, the new line lands
        once with its original send time, and the upstream intake age is
        measured against that send time rather than the reconnect moment.
        """
        registry = MetricsRegistry()
        instrument.set_registry(registry)
        try:
            clock = Clock()
            monotonic = Monotonic()
            seen = privmsg("dance", message_id="replayed", timestamp=clock.now)
            first = FakeTransport([[WELCOME], [seen]], die_after_script=True)

            # Still inside the provider's own 60s freshness window, so the
            # duplicate guard (not the staleness guard) is what suppresses the
            # replayed line below.
            clock.advance(30)
            fresh_sent_at = clock.now
            second = FakeTransport(
                [
                    [WELCOME],
                    # The platform replays the old line, then delivers a
                    # genuinely new one from the same viewer.
                    [seen],
                    [privmsg("still here?", message_id="fresh-2", timestamp=clock.now)],
                ]
            )
            provider, _ = make_provider([first, second], clock=clock, monotonic=monotonic)
            service = make_service(clock)
            runner = ChatSourceRunner(
                provider, service, clock=clock, poll_interval_seconds=0.01
            )
            self.addCleanup(runner.stop)

            provider.connect()
            self.assertEqual(provider.poll(), [])  # registration only
            self.assertIs(provider.state, ProviderState.CONNECTED)

            first_report = runner.run_once()
            self.assertEqual(first_report.accepted, 1)
            self.assertEqual(
                [message.message_id for message in service.get_queue()], ["replayed"]
            )

            # The socket drops; the provider schedules a bounded retry instead
            # of raising out of poll.
            self.assertEqual(provider.poll(), [])
            self.assertIs(provider.state, ProviderState.RETRYING)
            monotonic.advance(5.0)

            # Reconnect: a fresh handshake registers again, then the platform
            # replays history.
            self.assertEqual(runner.run_once().accepted, 0)
            self.assertIs(provider.state, ProviderState.CONNECTED)
            self.assertEqual(provider.diagnostics()["reconnect_count"], 1)
            self.assertEqual(runner.run_once().accepted, 0)  # replayed duplicate

            fresh_report = runner.run_once()
            self.assertEqual(fresh_report.accepted, 1)
            queued = {message.message_id: message for message in service.get_queue()}
            self.assertEqual(set(queued), {"replayed", "fresh-2"})
            self.assertEqual(queued["fresh-2"].timestamp, fresh_sent_at)
            self.assertEqual(queued["fresh-2"].enqueued_at, fresh_sent_at)
            # No silent loss and no backdating: both distinct viewer messages
            # are queued, and only the replayed duplicate was suppressed.
            self.assertEqual(registry.counter_value("chat.accepted"), 2)
            self.assertEqual(provider.diagnostics()["counters"]["duplicated"], 1)
            self.assertEqual(registry.counter_value("chat_source.duplicate_ids"), 0)
        finally:
            instrument.set_registry(None)


def world_snapshot() -> dict[str, Any]:
    return {
        "character": {"position": [0, 0, 0], "rotation_y": 90, "current_action": "idle"},
        "phone": {"state": "Placed", "position": [1, 0.75, 0], "placement_surface": "TABLE"},
        # PHASE 24: chat flows to the Brain while the stream is live.
        "stream": {"is_streaming": True, "camera_available": True},
        "targets": {
            "CENTER": {"position": [0, 0, 0], "navigation_target": True, "look_at_target": True},
            "TABLE": {"position": [1, 0.75, 0], "navigation_target": True, "look_at_target": True},
            "CAMERA": {"position": [1, 0.8, 0], "navigation_target": False, "look_at_target": True},
        },
    }


class BrainCompatibilityTests(unittest.TestCase):
    def tick(self, brain: AutonomousBrain) -> Any:
        return brain.tick(build_world_state(world_snapshot()), None)

    def make_brain(self, service: ChatService) -> AutonomousBrain:
        """A brain wired exactly like production: chat + the PHASE 11 voice path."""
        speech = SpeechService(provider=MockTTSProvider(), clock=Clock(), start_worker=False)
        self.addCleanup(speech.shutdown)
        return AutonomousBrain(
            MockBrainLLMProvider(),
            service,
            BrainConfig(decision_interval_seconds=0.0),
            speech_service=speech,
        )

    def test_real_message_flows_through_the_brain(self) -> None:
        clock = Clock()
        service = make_service(clock, max_messages_per_rate_window=20)
        provider = MockChatProvider(clock=clock)
        provider.connect()
        provider.push_message("twitch:501", "Lacrimosa can you dance?", "brain-1", "erin")
        runner = make_mock_runner(provider, service, clock)
        runner.run_once()
        self.assertEqual(service.get_state()["queued_count"], 1)

        brain = self.make_brain(service)
        world = build_world_state(world_snapshot())
        actions: list[ActionType] = []
        result = brain.tick(world, None)
        for _ in range(12):
            if result.action is None:
                break
            actions.append(result.action.action)
            if result.action.action is ActionType.SPEAK:
                break
            result = brain.tick(world, confirm(brain, ActionResult(result.action.action, ExecutionStatus.SUCCESS)))
        # The dance request still drives the physical skill (PHASE 09/10 intact).
        self.assertIn(ActionType.DANCE, actions)
        # PHASE 11 behaviour: the same turn still reached the speech pipeline.
        self.assertIn(ActionType.SPEAK, actions)

    def test_brain_keeps_ticking_while_the_chat_source_is_down(self) -> None:
        clock = Clock()
        provider = MockChatProvider(clock=clock)
        provider.connect()
        service = make_service(clock)
        runner = make_mock_runner(provider, service, clock)
        brain = AutonomousBrain(
            MockBrainLLMProvider(), service, BrainConfig(decision_interval_seconds=0.0)
        )
        provider.fail_next_poll = 2
        for _ in range(3):
            runner.run_once()
            result = self.tick(brain)
            self.assertIsNotNone(result.status)
        self.assertEqual(runner.get_state()["counters"]["provider_errors"], 2)
        self.assertEqual(service.get_state()["queued_count"], 0)
        provider.push_message("twitch:9", "Lacrimosa can you dance?", "recovered")
        self.assertEqual(runner.run_once().accepted, 1)
        self.assertEqual(service.get_state()["queued_count"], 1)


class ApiDiagnosticsTests(unittest.TestCase):
    def start_server(self, service: ChatService, source: Optional[ChatSourceRunner] = None):
        from backend.api.chat import create_chat_server

        server = create_chat_server(service, port=0, source=source)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(
            lambda: (server.shutdown(), server.server_close(), thread.join(timeout=2))
        )
        return f"http://127.0.0.1:{server.server_address[1]}"

    def get(self, url: str) -> tuple[int, dict[str, Any]]:
        import urllib.error
        import urllib.request

        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def test_source_routes_report_disabled_without_a_runner(self) -> None:
        base = self.start_server(make_service())
        status, payload = self.get(f"{base}/chat/source/state")
        self.assertEqual(status, 200)
        self.assertEqual(payload["provider"], "disabled")
        status, queue = self.get(f"{base}/chat/queue")
        self.assertEqual((status, queue["messages"]), (200, []))
        status, _ = self.get(f"{base}/chat/nope")
        self.assertEqual(status, 404)

    def test_source_routes_expose_runner_state_without_credentials(self) -> None:
        clock = Clock()
        service = make_service(clock)
        provider = MockChatProvider(clock=clock)
        provider.connect()
        provider.push_message("twitch:1", "hello", "api-1", "frank")
        runner = make_mock_runner(provider, service, clock)
        runner.run_once()
        base = self.start_server(service, runner)
        status, payload = self.get(f"{base}/chat/source/state")
        self.assertEqual(status, 200)
        self.assertEqual(payload["platform"], "mock")
        self.assertEqual(payload["last_message"]["viewer_name"], "frank")
        self.assertNotIn("token", json.dumps(payload))
        status, health = self.get(f"{base}/chat/source/health")
        self.assertEqual(status, 200)
        self.assertTrue(health["ok"])


if __name__ == "__main__":
    unittest.main()
