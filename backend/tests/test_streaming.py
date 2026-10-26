"""PHASE 13 livestream transport tests. Everything here runs offline.

No test reaches a streaming platform and none needs ffmpeg installed: the real
provider is exercised through a fake process spawner, and the key file is
always a temp file holding a sentinel string, so credential leakage is
provable rather than assumed.
"""

from __future__ import annotations

import json
import re
import socket
import sys
import threading
import time
import unittest
from collections import deque
from datetime import timedelta
from http import HTTPStatus
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Optional

from backend.api.stream import (
    MAX_MEDIA_BODY_BYTES,
    disabled_state,
    handle_stream_request,
    is_stream_path,
)
from backend.streaming import (
    AudioChunk,
    MockStreamingProvider,
    StreamState,
    StreamingConfig,
    StreamingProvider,
    StreamingProviderError,
    VideoFrame,
)
from backend.streaming.base import ReconnectPolicy
from backend.streaming.config import (
    PROVIDER_DISABLED,
    PROVIDER_MOCK,
    PROVIDER_RTMP,
    PROVIDER_SRT,
    build_provider,
    endpoint_host,
    looks_like_endpoint,
    read_stream_key,
)
from backend.streaming.encoder import (
    audio_listener_url,
    build_encoder_arguments,
    classify_failure,
    free_port,
    parse_progress,
    progress_bitrate_kbps,
    progress_output_seconds,
    resolve_ffmpeg,
    scrub,
)
from backend.streaming.ffmpeg_provider import FFmpegStreamingProvider
from backend.streaming.service import StreamingService, build_streaming_service


SECRET_KEY = "live1234567890abcdefghijklmnopqrstuvwx"
SENTINEL_ABSENT_PATH = "no/such/ffmpeg-here.exe"


def wait_until(predicate, timeout: float = 2.0, interval: float = 0.01) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return bool(predicate())


class StreamingTestCase(unittest.TestCase):
    """Shared scaffolding: a throwaway key file and quick configs."""

    def setUp(self) -> None:
        self._workdir = TemporaryDirectory()
        self.addCleanup(self._workdir.cleanup)
        self.key_path = Path(self._workdir.name) / "stream_key.txt"
        self.key_path.write_text(f"{SECRET_KEY}\n", encoding="utf-8")

    def config(self, provider: str = PROVIDER_MOCK, **overrides: Any) -> StreamingConfig:
        values: dict[str, Any] = {
            "provider": provider,
            "stream_key_path": self.key_path,
            "video_queue_capacity": 4,
            "audio_queue_capacity": 8,
            "live_timeout_seconds": 0.4,
            "start_backoff_seconds": 0.0,
            "connect_timeout_seconds": 0.2,
            "write_timeout_seconds": 0.2,
        }
        values.update(overrides)
        return StreamingConfig(**values)

    def assertNoSecret(self, value: Any, label: str = "") -> None:
        text = repr(value)
        self.assertNotIn(SECRET_KEY, text, f"secret leaked into {label or type(value)}")


class ProviderAbstractionTests(StreamingTestCase):
    def test_base_class_is_abstract(self) -> None:
        with self.assertRaises(TypeError):
            StreamingProvider()  # type: ignore[abstract]

    def test_mock_satisfies_the_interface(self) -> None:
        provider = MockStreamingProvider(self.config())
        for method in ("connect", "disconnect", "write_video", "write_audio", "poll", "health"):
            self.assertTrue(callable(getattr(provider, method)), method)
        self.assertIs(provider.state, StreamState.IDLE)
        self.assertTrue(provider.transport_enabled)

    def test_diagnostics_are_counters_only(self) -> None:
        payload = MockStreamingProvider(self.config()).diagnostics()
        self.assertEqual(payload["provider"], "mock")
        self.assertEqual(payload["state"], "IDLE")
        self.assertNoSecret(payload, "mock diagnostics")


class ConfigurationTests(StreamingTestCase):
    def test_disabled_is_the_default(self) -> None:
        resolved = StreamingConfig.from_environment({})
        self.assertEqual(resolved.provider, PROVIDER_DISABLED)
        self.assertFalse(resolved.enabled)
        self.assertFalse(resolved.is_real)
        self.assertIsNone(build_streaming_service(config=resolved))

    def test_real_provider_without_an_endpoint_stays_off(self) -> None:
        real_without_endpoint = StreamingConfig(
            provider=PROVIDER_RTMP, server_url="not-an-endpoint"
        )
        self.assertIsNone(build_streaming_service(config=real_without_endpoint))

    def test_explicit_mock_selection(self) -> None:
        resolved = StreamingConfig.from_environment({"STREAM_PROVIDER": "Mock"})
        self.assertEqual(resolved.provider, PROVIDER_MOCK)
        self.assertTrue(resolved.enabled)
        self.assertFalse(resolved.is_real)

    def test_unknown_provider_falls_back_to_disabled(self) -> None:
        resolved = StreamingConfig.from_environment({"STREAM_PROVIDER": "youtube"})
        self.assertEqual(resolved.provider, PROVIDER_DISABLED)

    def test_endpoint_file_selects_real_transport(self) -> None:
        url_path = Path(self._workdir.name) / "stream_url.txt"
        url_path.write_text("rtmp://live.example.com/app\n", encoding="utf-8")
        resolved = StreamingConfig.from_environment(
            {"STREAM_SERVER_URL_PATH": str(url_path)}
        )
        self.assertEqual(resolved.provider, PROVIDER_RTMP)
        self.assertTrue(resolved.is_real)
        self.assertEqual(endpoint_host(resolved.server_url), "rtmp://live.example.com")

    def test_environment_overrides_are_parsed(self) -> None:
        resolved = StreamingConfig.from_environment(
            {
                "STREAM_PROVIDER": "srt",
                "STREAM_WIDTH": "854",
                "STREAM_HEIGHT": "480",
                "STREAM_FPS": "24",
                "STREAM_SAMPLE_RATE": "44100",
                "STREAM_AUDIO_CHANNELS": "2",
                "STREAM_START_BACKOFF_SECONDS": "0",
                "STREAM_RECONNECT_MAX_ATTEMPTS": "2",
            }
        )
        self.assertEqual(resolved.provider, PROVIDER_SRT)
        self.assertEqual(resolved.resolution, "854x480")
        self.assertEqual(resolved.sample_rate, 44100)
        self.assertEqual(resolved.channels, 2)
        self.assertEqual(resolved.start_backoff_seconds, 0.0)
        self.assertEqual(resolved.reconnect.max_attempts, 2)
        self.assertEqual(resolved.container, "mpegts")

    def test_container_follows_protocol(self) -> None:
        self.assertEqual(self.config(PROVIDER_RTMP).container, "flv")
        self.assertEqual(self.config(PROVIDER_SRT).container, "mpegts")

    def test_invalid_geometry_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            StreamingConfig(width=0)
        with self.assertRaises(ValueError):
            StreamingConfig(fps=0)
        with self.assertRaises(ValueError):
            StreamingConfig(channels=3)

    def test_looks_like_endpoint(self) -> None:
        self.assertTrue(looks_like_endpoint("rtmp://live.example.com/app"))
        self.assertTrue(looks_like_endpoint("srt://127.0.0.1:9000"))
        self.assertTrue(looks_like_endpoint("C:/tmp/out.flv"))
        self.assertFalse(looks_like_endpoint(""))
        self.assertFalse(looks_like_endpoint("not-an-endpoint"))

    def test_build_provider_choice(self) -> None:
        self.assertIsInstance(build_provider(self.config(PROVIDER_MOCK)), MockStreamingProvider)
        self.assertIsInstance(
            build_provider(self.config(PROVIDER_RTMP, ffmpeg_path=sys.executable)),
            FFmpegStreamingProvider,
        )
        self.assertIsNone(build_provider(self.config(PROVIDER_DISABLED)))

    def test_read_stream_key(self) -> None:
        self.assertEqual(read_stream_key(self.key_path), SECRET_KEY)
        self.assertEqual(read_stream_key(Path(self._workdir.name) / "missing.txt"), "")

    def test_no_public_surface_carries_the_key(self) -> None:
        real = self.config(
            PROVIDER_RTMP,
            server_url=f"rtmp://live.example.com/app/{SECRET_KEY}",
            ffmpeg_path=sys.executable,
        )
        self.assertNoSecret(repr(real), "config repr")
        self.assertNoSecret(real.to_dict(), "config to_dict")
        self.assertNoSecret(repr(real.sanitized()), "sanitized config")
        self.assertNoSecret(endpoint_host(real.server_url), "endpoint host")

        provider = FFmpegStreamingProvider(real)
        self.assertNoSecret(repr(provider), "provider repr")
        self.assertNoSecret(provider.diagnostics(), "provider diagnostics")
        self.assertNoSecret(provider.health().to_dict(), "provider health")

        service = StreamingService(provider, real, start_worker=False)
        self.assertNoSecret(service.get_state(), "service state")
        self.assertNoSecret(service.health(), "service health")
        self.assertNoSecret(service.ingest_profile(), "ingest profile")


class ReconnectPolicyTests(unittest.TestCase):
    def test_backoff_grows_and_is_capped(self) -> None:
        policy = ReconnectPolicy(base_delay_seconds=1.0, max_delay_seconds=20.0, max_attempts=4)
        self.assertEqual(policy.delay_for_attempt(1), 1.0)
        self.assertEqual(policy.delay_for_attempt(3), 4.0)
        self.assertEqual(policy.delay_for_attempt(9), 20.0)
        self.assertTrue(policy.allows_attempt(4))
        self.assertFalse(policy.allows_attempt(5))

    def test_policy_validates_itself(self) -> None:
        with self.assertRaises(ValueError):
            ReconnectPolicy(base_delay_seconds=0)
        with self.assertRaises(ValueError):
            ReconnectPolicy(base_delay_seconds=5, max_delay_seconds=1)
        with self.assertRaises(ValueError):
            ReconnectPolicy(max_attempts=0)


class MediaModelTests(unittest.TestCase):
    def test_video_frame_validation(self) -> None:
        with self.assertRaises(ValueError):
            VideoFrame(payload="not bytes")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            VideoFrame(payload=b"x", sequence=-1)
        frame = VideoFrame(payload=b"abc", sequence=3, camera="phone")
        self.assertEqual(frame.byte_length, 3)
        payload = frame.to_dict()
        self.assertNotIn("payload", payload)
        self.assertEqual(payload["camera"], "phone")

    def test_audio_chunk_validation_and_duration(self) -> None:
        with self.assertRaises(ValueError):
            AudioChunk(payload=b"abc", sample_rate=48000)  # odd byte count
        with self.assertRaises(ValueError):
            AudioChunk(payload=b"abcd", sample_rate=0)
        with self.assertRaises(ValueError):
            AudioChunk(payload=b"abcd", sample_rate=48000, channels=3)
        chunk = AudioChunk(payload=bytes(960), sample_rate=48000, channels=1)
        self.assertAlmostEqual(chunk.duration_seconds, 0.01, places=4)


class EncoderHelperTests(unittest.TestCase):
    def test_parse_progress_folds_latest_values(self) -> None:
        parsed = parse_progress(
            "frame=1\nfps=10.0\n"
            "frame=3\nfps=14.9\ndrop_frames=2\ndup_frames=1\n"
            "bitrate= 1523.4kbits/s\nspeed=0.98x\nout_time_us=1234567\n"
            "total_size=98765\nprogress=end\n"
        )
        self.assertEqual(parsed["frame"], 3)
        self.assertEqual(parsed["fps"], 14.9)
        self.assertEqual(parsed["drop_frames"], 2)
        self.assertEqual(parsed["dup_frames"], 1)
        self.assertEqual(parsed["total_size"], 98765)
        self.assertEqual(parsed["progress"], "end")
        self.assertEqual(progress_bitrate_kbps(parsed), 1523.4)
        self.assertAlmostEqual(progress_output_seconds(parsed), 1.234567, places=6)

    def test_progress_helpers_tolerate_missing_values(self) -> None:
        self.assertEqual(progress_bitrate_kbps({}), 0.0)
        self.assertEqual(progress_bitrate_kbps({"bitrate": "N/A"}), 0.0)
        self.assertEqual(progress_output_seconds({"out_time_us": "bad"}), 0.0)
        self.assertEqual(parse_progress(""), {})

    def test_scrub_removes_secrets_and_credentials(self) -> None:
        text = f"Connection failed for key {SECRET_KEY} at rtmp://user:pw@live.example.com/app"
        cleaned = scrub(text, [SECRET_KEY])
        self.assertNotIn(SECRET_KEY, cleaned)
        self.assertIn("[redacted]", cleaned)
        self.assertNotIn("user:pw@", cleaned)

    def test_scrub_ignores_short_values_and_truncates(self) -> None:
        self.assertEqual(scrub("abcdef", ["abc"]), "abcdef")
        self.assertEqual(len(scrub("x" * 900, [])), 200)

    def test_classify_failure_vocabulary(self) -> None:
        self.assertEqual(classify_failure("HTTP 403 Forbidden"), "AUTH_FAILED")
        self.assertEqual(classify_failure("invalid stream key"), "AUTH_FAILED")
        self.assertEqual(classify_failure("Error while opening encoder"), "ENCODER_FAILED")
        self.assertEqual(classify_failure("Operation timed out"), "TIMEOUT")
        self.assertEqual(classify_failure("Service unavailable"), "PROVIDER_OUTAGE")
        self.assertEqual(classify_failure("Connection refused"), "CONNECTION_FAILED")
        self.assertEqual(classify_failure("mystery", 1), "TRANSPORT_FAILED")
        self.assertEqual(classify_failure(""), "PROVIDER_FAILURE")

    def test_resolve_ffmpeg(self) -> None:
        self.assertEqual(resolve_ffmpeg(SENTINEL_ABSENT_PATH), (None, "FFMPEG_PATH_INVALID"))
        found, reason = resolve_ffmpeg("")
        if found is None:
            self.assertEqual(reason, "FFMPEG_NOT_FOUND")
        else:
            self.assertTrue(Path(found).is_file())

    def test_audio_listener_url_and_free_port(self) -> None:
        self.assertEqual(
            audio_listener_url(5000), "tcp://127.0.0.1:5000?listen=1"
        )
        port = free_port()
        self.assertGreater(port, 0)
        # The hint is only a hint: the port must still be bindable.
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", port))

    def test_encoder_arguments_shape(self) -> None:
        arguments = build_encoder_arguments(
            StreamingConfig(provider=PROVIDER_RTMP),
            "rtmp://live.example.com/app/key",
            "tcp://127.0.0.1:5000?listen=1",
            "ffmpeg",
        )
        self.assertEqual(arguments[0], "ffmpeg")
        self.assertEqual(arguments[-1], "rtmp://live.example.com/app/key")
        self.assertIn("-progress", arguments)
        self.assertEqual(arguments[arguments.index("-progress") + 1], "pipe:2")
        self.assertIn("image2pipe", arguments)
        self.assertIn("mjpeg", arguments)
        self.assertIn("s16le", arguments)
        self.assertIn("libx264", arguments)
        self.assertIn("aac", arguments)
        self.assertIn("flv", arguments)
        self.assertIn("no_duration_filesize", arguments)
        self.assertNotIn("-vf", arguments)
        self.assertEqual(arguments.count("tcp://127.0.0.1:5000?listen=1"), 1)
        # ffmpeg opens inputs in order, so the listening PCM input has to come
        # first: a video-first command line blocks reading stdin and never binds
        # the audio port, which deadlocks the encoder against its own sender.
        self.assertLess(
            arguments.index("tcp://127.0.0.1:5000?listen=1"),
            arguments.index("pipe:0"),
        )
        maps = [
            arguments[index + 1]
            for index, value in enumerate(arguments)
            if value == "-map"
        ]
        self.assertEqual(maps, ["0:a:0", "1:v:0"])
        # A live pipe has no end, so both inputs must be probed cheaply or
        # ffmpeg waits for megabytes of data before it starts encoding.
        self.assertEqual(arguments.count("-probesize"), 2)
        self.assertEqual(arguments.count("-analyzeduration"), 2)

    def test_encoder_arguments_fix_odd_dimensions(self) -> None:
        arguments = build_encoder_arguments(
            StreamingConfig(provider=PROVIDER_SRT, width=641, height=361),
            "srt://127.0.0.1:9000",
            "tcp://127.0.0.1:5001?listen=1",
            "ffmpeg",
        )
        self.assertIn("-vf", arguments)
        self.assertIn("mpegts", arguments)

    def test_encoder_arguments_require_target_and_binary(self) -> None:
        with self.assertRaises(ValueError):
            build_encoder_arguments(StreamingConfig(), "", "tcp://x", "ffmpeg")
        with self.assertRaises(ValueError):
            build_encoder_arguments(StreamingConfig(), "rtmp://x/y", "tcp://x", "")


class MockProviderLifecycleTests(StreamingTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.now = 100.0
        self.provider = MockStreamingProvider(self.config(), clock=lambda: self.now)

    def feed(self, frames: int = 3) -> None:
        for index in range(1, frames + 1):
            self.assertTrue(self.provider.write_video(VideoFrame(payload=b"jpg", sequence=index)))
        self.assertTrue(
            self.provider.write_audio(AudioChunk(payload=bytes(1920), sample_rate=48000))
        )

    def test_happy_lifecycle(self) -> None:
        self.assertIs(self.provider.state, StreamState.IDLE)
        self.provider.connect()
        self.assertIs(self.provider.state, StreamState.CONNECTING)
        # Still connecting: nothing has been encoded yet.
        self.assertIs(self.provider.poll(), StreamState.CONNECTING)
        self.feed()
        self.assertIs(self.provider.poll(), StreamState.LIVE)
        health = self.provider.health()
        self.assertEqual(health.frames_encoded, 3)
        self.assertGreater(health.audio_seconds_ingested, 0.0)
        self.assertEqual(health.video_state, "OK")
        self.assertEqual(health.audio_state, "OK")
        self.provider.disconnect()
        self.assertIs(self.provider.state, StreamState.STOPPED)

    def test_media_is_refused_before_connect(self) -> None:
        self.assertFalse(self.provider.write_video(VideoFrame(payload=b"jpg")))
        self.assertFalse(self.provider.write_audio(AudioChunk(payload=b"ab", sample_rate=48000)))
        self.assertEqual(self.provider.health().dropped_frames, 2)

    def test_connect_failure_is_structured(self) -> None:
        provider = MockStreamingProvider(self.config(), connect_failure="AUTH_FAILED")
        with self.assertRaises(StreamingProviderError) as caught:
            provider.connect()
        self.assertEqual(caught.exception.reason, "AUTH_FAILED")
        self.assertIs(provider.state, StreamState.ERROR)
        self.assertEqual(provider.health().last_error, "AUTH_FAILED")
        # A later, well-behaved start is still possible.
        provider.connect_failure = None
        provider.connect()
        self.assertIs(provider.state, StreamState.CONNECTING)

    def drive(self, seconds: float, steps: int = 6) -> StreamState:
        """Advance the fake clock in steps, polling like the transport worker would."""
        state = self.provider.state
        for _ in range(steps):
            self.now += seconds / steps
            state = self.provider.poll()
        return state

    def test_drop_then_reconnect_returns_to_live(self) -> None:
        self.provider.connect()
        self.feed()
        self.assertIs(self.provider.poll(), StreamState.LIVE)
        self.now += 30.0
        self.provider.fail_after_seconds = 1.0
        self.assertIs(self.provider.poll(), StreamState.RECONNECTING)
        # Not restored yet, so a reconnect has not been counted.
        self.assertEqual(self.provider.health().reconnect_count, 0)
        self.assertIs(self.drive(5.0), StreamState.LIVE)
        self.assertEqual(self.provider.health().reconnect_count, 1)

    def test_reconnect_failure_is_consumed_once(self) -> None:
        self.provider.reconnect_failure = "CONNECTION_FAILED"
        self.provider.fail_after_seconds = 0.0
        self.provider.connect()
        self.feed()
        self.assertIs(self.provider.poll(), StreamState.LIVE)
        self.now += 1.0
        self.assertIs(self.provider.poll(), StreamState.RECONNECTING)
        self.now += 2.0
        # The scripted failure burns the first retry without raising out of poll.
        self.assertIs(self.provider.poll(), StreamState.RECONNECTING)
        self.assertEqual(self.provider.health().last_error, "CONNECTION_FAILED")
        self.assertIs(self.drive(30.0), StreamState.LIVE)
        self.assertEqual(self.provider.health().reconnect_count, 1)

    def test_reconnect_budget_is_bounded(self) -> None:
        class AlwaysRefuses(MockStreamingProvider):
            def connect(self) -> None:
                if self._reconnect_attempt:
                    raise StreamingProviderError("CONNECTION_FAILED", "still down")
                super().connect()

        policy = ReconnectPolicy(base_delay_seconds=0.5, max_delay_seconds=1.0, max_attempts=2)
        provider = AlwaysRefuses(
            self.config(reconnect=policy), clock=lambda: self.now
        )
        self.provider = provider
        provider.connect()
        self.feed()
        self.assertIs(provider.poll(), StreamState.LIVE)
        provider.fail_after_seconds = 0.0
        self.now += 1.0
        self.assertIs(provider.poll(), StreamState.RECONNECTING)
        state = self.drive(600.0)
        self.assertIs(state, StreamState.ERROR)
        self.assertEqual(provider.health().reconnect_count, 0)
        self.assertLessEqual(provider._reconnect_attempt, policy.max_attempts)

    def test_health_never_carries_the_key(self) -> None:
        provider = MockStreamingProvider(self.config(PROVIDER_RTMP))
        self.assertNoSecret(provider.health().to_dict(), "mock health")


class ServiceCommandTests(StreamingTestCase):
    def build(self, provider: StreamingProvider) -> StreamingService:
        service = StreamingService(provider, provider.config, poll_interval_seconds=0.01)
        self.addCleanup(service.stop, 2.0)
        return service

    def live(self, service: StreamingService) -> None:
        """Start the transport and keep feeding it until it is genuinely live."""
        self.assertTrue(service.transport_enabled)
        service.request_start(wait_seconds=0.05)
        for sequence in range(1, 200):
            if service.state is StreamState.LIVE:
                return
            service.submit_video(b"jpg-frame", sequence=sequence)
            service.submit_audio(bytes(1920), sequence=sequence)
            if service.wait_for_state(lambda state: state is StreamState.LIVE, 0.05) is StreamState.LIVE:
                return
        self.fail("mock transport never reached LIVE")

    def test_start_without_media_reports_no_media(self) -> None:
        service = self.build(MockStreamingProvider(self.config()))
        result = service.request_start(wait_seconds=0.5)
        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "NO_MEDIA")
        self.assertNotEqual(result.state, StreamState.LIVE)
        self.assertFalse(service.get_state()["health"]["ok"])

    def test_start_becomes_live_once_media_flows(self) -> None:
        service = self.build(MockStreamingProvider(self.config()))
        self.live(service)
        self.assertIs(service.state, StreamState.LIVE)
        health = service.health()
        self.assertTrue(health["ok"])
        self.assertEqual(health["video_state"], "OK")
        self.assertEqual(health["audio_state"], "OK")

    def test_start_is_idempotent_while_live(self) -> None:
        provider = MockStreamingProvider(self.config())
        service = self.build(provider)
        self.live(service)
        attempts = provider.health().connect_attempts
        again = service.request_start(wait_seconds=0.2)
        self.assertTrue(again.ok)
        self.assertEqual(again.reason, "ALREADY_LIVE")
        self.assertEqual(provider.health().connect_attempts, attempts)

    def test_stop_then_start_works_again(self) -> None:
        provider = MockStreamingProvider(self.config())
        service = self.build(provider)
        self.live(service)
        stopped = service.request_stop(wait_seconds=2.0)
        self.assertTrue(stopped.ok)
        self.assertIs(service.state, StreamState.STOPPED)
        after = service.health()
        self.assertEqual(after["video_state"], "NO_INPUT")
        self.assertFalse(after["ok"])
        # Media submitted while stopped is refused, not queued forever.
        self.assertFalse(service.submit_video(b"late", sequence=999).accepted)
        self.live(service)
        self.assertIs(service.state, StreamState.LIVE)

    def test_failed_start_cools_down_then_retries(self) -> None:
        provider = MockStreamingProvider(self.config(), connect_failure="AUTH_FAILED")
        service = StreamingService(provider, provider.config, poll_interval_seconds=0.01)
        self.addCleanup(service.stop, 2.0)
        first = service.request_start(wait_seconds=0.5)
        self.assertFalse(first.ok)
        self.assertEqual(first.reason, "AUTH_FAILED")
        self.assertIs(service.state, StreamState.ERROR)
        provider.connect_failure = None
        # The configured cool-down has already expired in this config.
        self.live(service)
        self.assertIs(service.state, StreamState.LIVE)

    def test_start_backoff_refuses_immediate_retries(self) -> None:
        provider = MockStreamingProvider(self.config(start_backoff_seconds=60.0))
        provider.connect_failure = "AUTH_FAILED"
        service = self.build(provider)
        self.assertFalse(service.request_start(wait_seconds=0.5).ok)
        provider.connect_failure = None
        refused = service.request_start(wait_seconds=0.5)
        self.assertFalse(refused.ok)
        self.assertEqual(refused.reason, "BACKOFF")
        # An explicit stop clears the cool-down: the operator decided, not the retry loop.
        service.request_stop(wait_seconds=1.0)
        self.live(service)

    def test_service_survives_a_provider_that_raises_everywhere(self) -> None:
        class Hostile(MockStreamingProvider):
            def poll(self) -> StreamState:
                raise RuntimeError("boom")

            def connect(self) -> None:
                raise RuntimeError("boom")

        service = self.build(Hostile(self.config()))
        result = service.request_start(wait_seconds=0.3)
        self.assertFalse(result.ok)
        self.assertEqual(service.get_state()["counters"]["connect_errors"] > 0, True)
        service.stop(1.0)
        self.assertIs(service.state, StreamState.STOPPED)

    def test_stop_never_hangs_on_a_stuck_provider(self) -> None:
        class Stuck(MockStreamingProvider):
            def disconnect(self) -> None:
                self._state = StreamState.STOPPED

        service = self.build(Stuck(self.config()))
        self.assertTrue(service.request_stop(wait_seconds=1.0).ok)


class QueueTests(StreamingTestCase):
    """Queue maths driven by hand, with no worker thread in the way."""

    def build(self, **overrides: Any) -> StreamingService:
        provider = MockStreamingProvider(self.config(**overrides))
        provider.connect()
        service = StreamingService(
            provider, provider.config, poll_interval_seconds=0.01, start_worker=False
        )
        return service

    def test_empty_and_invalid_media_is_refused(self) -> None:
        service = self.build()
        self.assertEqual(service.submit_video(b"").reason, "EMPTY_PAYLOAD")
        self.assertEqual(service.submit_audio(b"").reason, "EMPTY_PAYLOAD")
        self.assertEqual(service.submit_video(b"jpeg", sequence=-3).reason, "INVALID_SEQUENCE")
        self.assertEqual(service.submit_video(b"jpeg", sequence="x").reason, "INVALID_SEQUENCE")
        self.assertEqual(
            service.submit_audio(bytes(7), sequence=1).reason, "INVALID_AUDIO"
        )

    def test_oversize_media_is_refused(self) -> None:
        service = self.build()
        big_video = bytes(service.provider.max_frame_bytes + 1)
        big_audio = bytes(service.provider.max_audio_bytes + 2)
        self.assertEqual(service.submit_video(big_video, sequence=1).reason, "TOO_LARGE")
        self.assertEqual(service.submit_audio(big_audio, sequence=1).reason, "TOO_LARGE")

    def test_duplicate_and_out_of_order_sequences_are_dropped(self) -> None:
        service = self.build()
        self.assertTrue(service.submit_video(b"a", sequence=5).accepted)
        self.assertEqual(service.submit_video(b"b", sequence=5).reason, "DUPLICATE")
        self.assertEqual(service.submit_video(b"c", sequence=4).reason, "DUPLICATE")
        state = service.get_state()
        self.assertEqual(state["counters"]["duplicates"], 2)
        self.assertEqual(state["counters"]["video_accepted"], 1)

    def test_a_new_capture_session_restarts_media_sequences(self) -> None:
        """Sequences belong to a capture session, not to the backend process.

        A Unity client that reconnects counts from one again, so marks left over
        from an older session must not mute its audio and video forever.
        """
        provider = MockStreamingProvider(self.config())
        provider.connect()
        service = StreamingService(
            provider, provider.config, poll_interval_seconds=0.01, start_worker=False
        )
        self.addCleanup(service.stop, 2.0)
        # First session runs to a high-water mark.
        self.assertTrue(service.submit_audio(bytes(1920), sequence=900).accepted)
        self.assertTrue(service.submit_video(b"frame", sequence=900).accepted)
        service.run_once(0.001)
        service.request_stop(wait_seconds=0.05)
        service.run_once(0.001)
        # A brand new client starts counting again at one.
        service.request_start(wait_seconds=0.05)
        service.run_once(0.001)
        self.assertTrue(
            service.submit_audio(bytes(1920), sequence=1).accepted,
            "a new session must not be muted by the previous session's sequences",
        )
        self.assertTrue(service.submit_video(b"frame", sequence=1).accepted)
        # Deduplication still holds inside the new session.
        self.assertEqual(service.submit_audio(bytes(1920), sequence=1).reason, "DUPLICATE")

    def test_queue_is_bounded_and_drops_the_oldest(self) -> None:
        service = self.build(video_queue_capacity=2, audio_queue_capacity=2)
        for sequence in range(1, 6):
            self.assertTrue(service.submit_video(b"frame-%d" % sequence, sequence=sequence).accepted)
        result = service.submit_video(b"frame-6", sequence=6)
        self.assertTrue(result.accepted)
        self.assertEqual(result.queued, 2)
        depth = service.get_state()["queues"]["video"]
        self.assertLessEqual(depth, 2)
        self.assertEqual(service.health()["queue_video"]["dropped"], 4)
        self.assertEqual(service.health()["queue_video"]["capacity"], 2)

    def test_overflow_protects_memory(self) -> None:
        service = self.build(video_queue_capacity=3)
        for sequence in range(1, 500):
            service.submit_video(b"f", sequence=sequence)
        self.assertLessEqual(len(service._video), 3)
        self.assertEqual(service.get_state()["queues"]["video"], 3)

    def test_media_is_refused_when_not_streaming(self) -> None:
        provider = MockStreamingProvider(self.config())
        service = StreamingService(provider, provider.config, start_worker=False)
        self.assertEqual(service.submit_video(b"a", sequence=1).reason, "NOT_STREAMING")
        self.assertEqual(service.submit_audio(b"ab", sequence=1).reason, "NOT_STREAMING")

    def test_run_once_drains_video_then_audio(self) -> None:
        provider = MockStreamingProvider(self.config())
        provider.connect()
        service = StreamingService(provider, provider.config, start_worker=False)
        service.submit_video(b"one", sequence=1)
        service.submit_audio(bytes(1920), sequence=1)
        self.assertTrue(service.run_once(0.001))
        health = provider.health()
        self.assertEqual(health.frames_encoded, 1)
        self.assertGreater(health.audio_seconds_ingested, 0.0)
        self.assertIs(provider.state, StreamState.LIVE)
        # Nothing left to do, so the next pass reports no work.
        self.assertFalse(service.run_once(0.001))

    def test_input_goes_stale_when_capture_stops(self) -> None:
        service = self.build()
        service.submit_video(b"a", sequence=1)
        service.submit_audio(bytes(1920), sequence=1)
        self.assertEqual(service.health()["video_state"], "OK")
        service._last_video_at = service._clock() - timedelta(seconds=30)
        service._last_audio_at = service._clock() - timedelta(seconds=30)
        health = service.health()
        self.assertEqual(health["video_state"], "STALE")
        self.assertEqual(health["audio_state"], "STALE")

    def test_state_schema_is_stable_and_secret_free(self) -> None:
        service = self.build()
        state = service.get_state()
        for key in (
            "provider",
            "platform",
            "transport_enabled",
            "state",
            "health",
            "config",
            "ingest",
            "queues",
            "counters",
        ):
            self.assertIn(key, state)
        self.assertNoSecret(json.dumps(state), "service state json")
        # The payload names the key *file* (the PHASE 11/12 convention) but must
        # never carry a URL that could hold a key, an absolute path, or key bytes.
        blob = json.dumps(state)
        self.assertNotIn("server_url", blob)
        self.assertNotIn(str(self.key_path.resolve().parent), blob)
        self.assertNotIn(SECRET_KEY[:8], blob)

    def test_stop_clears_the_queues(self) -> None:
        service = self.build()
        service.submit_video(b"a", sequence=1)
        service.submit_audio(bytes(1920), sequence=1)
        service.stop(1.0)
        self.assertEqual(service.get_state()["queues"], {"video": 0, "audio": 0})
        self.assertEqual(service.health()["video_state"], "NO_INPUT")


class FakeStdin:
    def __init__(self) -> None:
        self.payload = bytearray()
        self.closed = False
        self.fail_on_write = False

    def write(self, payload: bytes) -> None:
        if self.fail_on_write:
            raise OSError("broken pipe")
        self.payload.extend(payload)

    def flush(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


class FakeStderr:
    """A stderr that blocks until a test hands it a line, like the real pipe."""

    def __init__(self) -> None:
        self._lines: deque = deque()
        self._condition = threading.Condition()
        self._ending = False

    def push(self, text: str) -> None:
        with self._condition:
            self._lines.append(text.encode())
            self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            self._ending = True
            self._condition.notify_all()

    def read1(self, size: int) -> bytes:
        with self._condition:
            deadline = time.monotonic() + 5.0
            while not self._lines and not self._ending and time.monotonic() < deadline:
                self._condition.wait(0.02)
            if self._lines:
                return self._lines.popleft()
            self._ending = True
            return b""


class FakeProcess:
    """The handful of ``Popen`` members the provider actually touches."""

    def __init__(self, stderr: FakeStderr) -> None:
        self.stdin = FakeStdin()
        self.stderr = stderr
        self.pid = 424242
        self.exit_code: Optional[int] = None
        self.terminations = 0
        self.kills = 0

    def poll(self) -> Optional[int]:
        return self.exit_code

    def wait(self, timeout: Optional[float] = None) -> int:
        if self.exit_code is None:
            raise TimeoutError("encoder is still running")
        return self.exit_code

    def terminate(self) -> None:
        self.terminations += 1
        self.exit_code = -15

    def kill(self) -> None:
        self.kills += 1
        self.exit_code = -9


class FFmpegTransportTests(StreamingTestCase):
    """The real provider, driven by a fake encoder: no ffmpeg, no network."""

    def setUp(self) -> None:
        super().setUp()
        self.processes: list[FakeProcess] = []
        self.spawned: list[list[str]] = []

    def stderr_of(self, provider: FFmpegStreamingProvider) -> FakeStderr:
        """The stderr pipe of whichever encoder generation is currently live."""
        process = provider._process
        assert process is not None, "the provider has no encoder process"
        return process.stderr

    def stdin(self, provider: FFmpegStreamingProvider) -> FakeStdin:
        process = provider._process
        assert process is not None, "the provider has no encoder process"
        return process.stdin

    def live(self, provider: FFmpegStreamingProvider) -> FakeProcess:
        """Connect and make the encoder's audio input reachable."""
        provider.connect()
        self.accept_audio(provider)
        return provider._process

    def emit(self, provider: FFmpegStreamingProvider, text: str) -> None:
        """Push encoder progress lines and let the reader thread fold them in."""
        stream = self.stderr_of(provider)
        stream.push(text)
        fields = parse_progress(text)
        plain = [line.strip() for line in text.split("\n") if line.strip() and "=" not in line]

        def absorbed() -> bool:
            with provider._lock:
                if plain and not set(plain).issubset(set(provider._log_tail)):
                    return False
                return all(provider._progress.get(name) == value for name, value in fields.items())

        assert wait_until(absorbed, timeout=2.0), f"encoder output never absorbed: {text!r}"

    def provider(
        self,
        provider_name: str = PROVIDER_RTMP,
        **overrides: Any,
    ) -> FFmpegStreamingProvider:
        values: dict[str, Any] = {
            "server_url": "rtmp://live.example.com/app",
            "ffmpeg_path": sys.executable,
            "connect_timeout_seconds": 0.05,
            "live_timeout_seconds": 0.3,
            "write_timeout_seconds": 0.2,
        }
        values.update(overrides)
        config = self.config(provider_name, **values)

        def spawner(arguments: list[str]) -> FakeProcess:
            # Each generation gets its own stderr pipe, so a reader thread left over
            # from a killed encoder cannot steal the next generation's progress lines.
            process = FakeProcess(FakeStderr())
            self.addCleanup(process.stderr.close)
            self.spawned.append(arguments)
            self.processes.append(process)
            return process

        provider = FFmpegStreamingProvider(config, spawner=spawner)
        self.addCleanup(provider.disconnect)
        return provider

    def accept_audio(self, provider: FFmpegStreamingProvider) -> None:
        """Bind the PCM listener the provider dials into."""
        listener = socket.socket()
        listener.bind(("127.0.0.1", provider._audio_port))
        listener.listen(1)
        self.addCleanup(listener.close)
        assert wait_until(lambda: provider._open_audio_socket(0.05), timeout=2.0)
        connection, _ = listener.accept()
        self.addCleanup(connection.close)
        assert wait_until(lambda: provider._audio_socket is not None, timeout=2.0)

    def test_connect_builds_the_encoder_and_waits_for_media(self) -> None:
        provider = self.provider()
        provider.connect()
        self.assertIs(provider.state, StreamState.CONNECTING)
        self.assertEqual(len(self.spawned), 1)
        arguments = self.spawned[0]
        self.assertEqual(Path(arguments[0]), Path(sys.executable))
        self.assertIn("rtmp://live.example.com/app/" + SECRET_KEY, arguments[-1])
        self.assertEqual(provider.poll(), StreamState.CONNECTING)
        self.assertEqual(provider.health().frames_encoded, 0)

    def test_srt_target_is_used_verbatim_and_needs_no_key(self) -> None:
        provider = self.provider(
            PROVIDER_SRT, server_url="srt://127.0.0.1:9500?mode=caller"
        )
        provider.connect()
        self.assertEqual(self.spawned[0][-1], "srt://127.0.0.1:9500?mode=caller")

    def test_key_placeholder_target(self) -> None:
        provider = self.provider(
            PROVIDER_RTMP, server_url="rtmp://live.example.com/app/{key}"
        )
        provider.connect()
        self.assertEqual(
            self.spawned[0][-1], f"rtmp://live.example.com/app/{SECRET_KEY}"
        )

    def test_encoder_progress_promotes_live_and_reports_health(self) -> None:
        provider = self.provider()
        self.live(provider)
        self.assertTrue(provider.write_video(VideoFrame(payload=b"jpeg-bytes", sequence=1)))
        self.assertTrue(
            provider.write_audio(AudioChunk(payload=bytes(9600), sample_rate=48000))
        )
        self.assertEqual(bytes(self.stdin(provider).payload), b"jpeg-bytes")
        self.emit(
            provider,
            "frame=42\nfps=14.8\ndrop_frames=3\nbitrate= 1523.4kbits/s\n"
            "out_time_us=2800000\ntotal_size=534000\n",
        )
        assert wait_until(lambda: provider.poll() is StreamState.LIVE)
        health = provider.health()
        self.assertEqual(health.frames_encoded, 42)
        self.assertEqual(health.encoder_fps, 14.8)
        self.assertEqual(health.dropped_frames, 3)
        # ffmpeg already reports kbit/s: no division happens here.
        self.assertEqual(health.bitrate_kbps, 1523.4)
        self.assertAlmostEqual(health.attributes["muxed_seconds"], 2.8, places=2)
        self.assertEqual(health.attributes["encoder_pid"], 424242)
        self.assertEqual(health.video_state, "OK")
        self.assertEqual(health.audio_state, "OK")

    def test_live_audio_bytes_reach_the_encoder(self) -> None:
        provider = self.provider()
        self.live(provider)
        chunk = AudioChunk(payload=bytes(9600), sample_rate=48000)
        self.assertTrue(provider.write_audio(chunk))
        # Health reports seconds at two decimals, which is what the dashboards read.
        self.assertAlmostEqual(provider.health().audio_seconds_ingested, 0.1, places=2)
        self.assertEqual(provider._audio_bytes, chunk.byte_length)

    def test_media_is_refused_when_not_connected(self) -> None:
        provider = self.provider()
        self.assertFalse(provider.write_video(VideoFrame(payload=b"x")))
        self.assertFalse(provider.write_audio(AudioChunk(payload=b"ab", sample_rate=48000)))
        self.assertEqual(provider.write_failures, 0)

    def test_write_failure_is_reported_not_raised(self) -> None:
        provider = self.provider()
        self.live(provider)
        self.stdin(provider).fail_on_write = True
        self.assertFalse(provider.write_video(VideoFrame(payload=b"x", sequence=1)))
        self.assertEqual(provider.write_failures, 1)
        self.assertIs(provider.state, StreamState.CONNECTING)

    def test_transient_write_failure_does_not_tear_down_a_recovered_stream(self) -> None:
        """One refused write must not poison every later live window.

        A lifetime failure counter used to fail the stream exactly
        ``live_timeout_seconds`` after it came back, so a single retry churned
        the transport forever instead of recovering.
        """
        provider = self.provider()
        process = self.live(provider)
        provider.write_video(VideoFrame(payload=b"x", sequence=1))
        self.emit(provider, "frame=5\nfps=15.0\n")
        assert wait_until(lambda: provider.poll() is StreamState.LIVE)
        self.stdin(provider).fail_on_write = True
        self.assertFalse(provider.write_video(VideoFrame(payload=b"y", sequence=2)))
        self.stdin(provider).fail_on_write = False
        # Media keeps flowing well past the live timeout on the same encoder.
        deadline = time.monotonic() + 0.5
        sequence = 3
        while time.monotonic() < deadline:
            self.assertTrue(
                provider.write_video(VideoFrame(payload=b"z", sequence=sequence))
            )
            sequence += 1
            time.sleep(0.02)
        self.assertIs(provider.poll(), StreamState.LIVE)
        self.assertIs(provider._process, process)
        self.assertEqual(provider.write_failures, 1)

    def test_a_stalled_encoder_is_still_declared_transport_lost(self) -> None:
        provider = self.provider()
        self.live(provider)
        provider.write_video(VideoFrame(payload=b"x", sequence=1))
        self.emit(provider, "frame=5\nfps=15.0\n")
        assert wait_until(lambda: provider.poll() is StreamState.LIVE)
        self.stdin(provider).fail_on_write = True
        self.assertFalse(provider.write_video(VideoFrame(payload=b"y", sequence=2)))
        # No picture is accepted again, so the encoder really is not draining.
        time.sleep(0.4)
        self.assertIs(provider.poll(), StreamState.RECONNECTING)

    def test_a_never_live_encoder_exit_is_terminal(self) -> None:
        provider = self.provider()
        provider.connect()
        self.processes[0].exit_code = 1
        provider.poll()
        self.assertIs(provider.state, StreamState.ERROR)
        # No silent respawn behind the caller's back.
        self.assertEqual(len(self.spawned), 1)
        # An exit with no explanatory output still gets a stable reason code.
        self.assertTrue(str(provider.health().last_error).startswith("TRANSPORT_FAILED"))

    def test_a_live_encoder_exit_reconnects_with_a_new_process(self) -> None:
        provider = self.provider()
        process = self.live(provider)
        provider.write_video(VideoFrame(payload=b"x", sequence=1))
        self.emit(provider, "frame=5\nfps=15.0\n")
        assert wait_until(lambda: provider.poll() is StreamState.LIVE)
        self.assertEqual(provider.health().reconnect_count, 0)
        process.exit_code = 1
        provider.poll()
        self.assertIs(provider.state, StreamState.RECONNECTING)
        # The reconnect is counted when the picture comes back, not on the drop.
        self.assertEqual(provider.health().reconnect_count, 0)
        provider._next_attempt_at = 0.0
        provider.poll()
        self.assertIs(provider.state, StreamState.CONNECTING)
        self.assertEqual(len(self.spawned), 2)
        self.assertIsNot(provider._process, process)
        provider.write_video(VideoFrame(payload=b"y", sequence=2))
        self.emit(provider, "frame=1\nfps=12.0\n")
        assert wait_until(lambda: provider.poll() is StreamState.LIVE)
        self.assertEqual(provider.health().reconnect_count, 1)

    def test_connection_failure_text_is_classified_and_scrubbed(self) -> None:
        provider = self.provider()
        process = self.live(provider)
        self.emit(provider, f"HTTP 403 Forbidden - invalid stream key {SECRET_KEY}\n")
        assert wait_until(lambda: "403" in "\n".join(provider._log_tail))
        process.exit_code = 1
        provider.poll()
        self.assertIs(provider.state, StreamState.ERROR)
        health = provider.health()
        self.assertTrue(str(health.last_error).startswith("AUTH_FAILED"))
        self.assertNoSecret(health.to_dict(), "provider health after failure")
        self.assertNoSecret(repr(provider), "provider repr after failure")

    def test_no_frames_before_the_deadline_is_a_timeout(self) -> None:
        provider = self.provider()
        clock = {"now": 1000.0}
        provider._clock = lambda: clock["now"]
        provider.connect()
        clock["now"] += 5.0
        provider.poll()
        self.assertIs(provider.state, StreamState.ERROR)
        self.assertTrue(str(provider.health().last_error).startswith("TIMEOUT"))

    def test_disconnect_terminates_and_releases(self) -> None:
        provider = self.provider()
        process = self.live(provider)
        provider.disconnect()
        self.assertIs(provider.state, StreamState.STOPPED)
        self.assertTrue(process.stdin.closed)
        self.assertIsNotNone(process.exit_code)
        self.assertIsNone(provider._process)
        self.assertEqual(provider.poll(), StreamState.STOPPED)
        self.assertEqual(provider.uptime_seconds(), 0.0)

    def test_health_and_repr_carry_no_key(self) -> None:
        provider = self.provider()
        provider.connect()
        self.assertNoSecret(repr(provider), "ffmpeg repr")
        self.assertNoSecret(provider.health().to_dict(), "ffmpeg health")
        self.assertNoSecret(provider.diagnostics(), "ffmpeg diagnostics")
        self.assertNoSecret(str(provider._log_tail), "ffmpeg log tail")

    def test_a_missing_binary_degrades_without_raising_out_of_poll(self) -> None:
        config = self.config(
            PROVIDER_RTMP,
            server_url="rtmp://live.example.com/app",
            ffmpeg_path=SENTINEL_ABSENT_PATH,
        )
        provider = FFmpegStreamingProvider(config, spawner=lambda args: FakeProcess())
        with self.assertRaises(StreamingProviderError) as caught:
            provider.connect()
        self.assertEqual(caught.exception.reason, "FFMPEG_PATH_INVALID")
        self.assertIs(provider.state, StreamState.ERROR)
        service = StreamingService(provider, config, poll_interval_seconds=0.01)
        self.addCleanup(service.stop, 2.0)
        result = service.request_start(wait_seconds=0.3)
        self.assertFalse(result.ok)
        self.assertIn(result.reason, {"FFMPEG_PATH_INVALID", "NO_MEDIA", "TIMEOUT"})
        # The Brain keeps running: a start failure is data, not an exception.
        self.assertEqual(service.state, StreamState.ERROR)
