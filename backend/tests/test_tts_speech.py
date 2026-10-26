"""PHASE 11 TTS and speech tests. Everything here runs offline and deterministic.

The OpenRouter transport is always faked: no test in this file reaches the
network, and no test reads the real key file.
"""

from __future__ import annotations

import io
import json
import threading
import unittest
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Optional
from types import SimpleNamespace
from wave import open as open_wave

from backend.api.brain import create_brain_server, create_default_service
from backend.brain.loop import AutonomousBrain, BrainConfig, BrainStatus
from backend.brain.models import ActionType, ExecutionStatus
from backend.brain.models import ActionResult
from backend.brain.models import Goal, GoalType, Plan, PlanStep
from backend.brain.planner import MAX_TEXT_LENGTH
from backend.brain.reasoning import MockBrainLLMProvider
from backend.brain.world_state import build_world_state
from backend.chat.service import ChatService, ChatServiceConfig
from backend.llm.mock import MockLLMProvider
from backend.speech.service import IDLE_STATE, SpeechConfig, SpeechService
from backend.speech.models import SpeechState
from backend.tts.base import (
    TTSProvider,
    TTSProviderError,
    TTSRequest,
    probe_audio,
    validate_audio_payload,
    wav_container_from_pcm,
)
from backend.tts.config import (
    PROVIDER_DISABLED,
    PROVIDER_MOCK,
    PROVIDER_OPENROUTER,
    TTSConfig,
    build_provider,
)
from backend.tts.mock import MockTTSProvider
from backend.tts.openrouter import OpenRouterTTSProvider
from backend.tests.dispatch_support import confirm


SECRET = "sk-or-VIVE-TEST-SECRET-DO-NOT-LEAK"


def mp3_bytes(duration_frames: int = 40) -> bytes:
    """A frame-sync shaped MP3 body, enough for the container sniff to pass."""
    header = bytes([0xFF, 0xFB, 0x90, 0x44])
    return header + bytes(duration_frames * 41)


def pcm_bytes(samples: int = 2000) -> bytes:
    return bytes((index * 37) % 256 for index in range(samples * 2))


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 9, 15, 12, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


class FakeResponse:
    def __init__(
        self,
        payload: bytes,
        content_type: str = "audio/mpeg",
        status: int = 200,
        generation_id: Optional[str] = "gen-tts-fake-1",
    ) -> None:
        self._payload = payload
        self.headers = {"Content-Type": content_type}
        if generation_id:
            self.headers["X-Generation-Id"] = generation_id
        self.status = status

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None


class FakeUrlopen:
    """Stands in for urllib.request.urlopen and records what was sent."""

    def __init__(
        self,
        response: Any = None,
        error: Optional[BaseException] = None,
    ) -> None:
        self.response = response if response is not None else FakeResponse(mp3_bytes())
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def __call__(self, request: urllib.request.Request, timeout: Any = None) -> Any:
        self.calls.append({"request": request, "timeout": timeout})
        if self.error is not None:
            raise self.error
        return self.response


def http_error(code: int, body: bytes = b"upstream failure") -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "https://openrouter.ai/api/v1/audio/speech",
        code,
        "error",
        {},  # type: ignore[arg-type]
        io.BytesIO(body),
    )


def openrouter_config(key_path: Path, output_format: str = "mp3") -> TTSConfig:
    return TTSConfig(
        provider=PROVIDER_OPENROUTER,
        output_format=output_format,
        api_key_path=key_path,
    )


def key_file(case: unittest.TestCase, secret: str = SECRET) -> Path:
    """A throwaway key file in a temp dir; the real key file is never read."""
    directory = TemporaryDirectory()
    case.addCleanup(directory.cleanup)
    path = Path(directory.name) / "api_key_voice.txt"
    path.write_text(secret + "\n", encoding="utf-8")
    return path


class ProviderAbstractionTests(unittest.TestCase):
    def test_build_provider_selects_by_name(self) -> None:
        self.assertIsInstance(build_provider(TTSConfig(provider=PROVIDER_MOCK)), MockTTSProvider)
        config = openrouter_config(key_file(self))
        self.assertIsInstance(build_provider(config), OpenRouterTTSProvider)

    def test_unknown_provider_degrades_to_mock(self) -> None:
        provider = build_provider(TTSConfig(provider="something-else"))
        self.assertIsInstance(provider, MockTTSProvider)
        self.assertEqual(provider.name, "mock")

    def test_environment_selects_provider(self) -> None:
        mock = TTSConfig.from_environment({"TTS_PROVIDER": "mock"})
        self.assertEqual(mock.provider, PROVIDER_MOCK)
        live = TTSConfig.from_environment(
            {"TTS_PROVIDER": "openrouter", "TTS_MODEL": "m", "TTS_VOICE": "v", "TTS_FORMAT": "pcm"}
        )
        self.assertEqual(live.provider, PROVIDER_OPENROUTER)
        self.assertEqual((live.model, live.voice, live.output_format), ("m", "v", "pcm"))

    def test_defaults_match_the_specified_provider(self) -> None:
        config = TTSConfig()
        self.assertEqual(config.model, "deepgram/flux-tts:free")
        self.assertEqual(config.voice, "flux-alexis-en")
        self.assertEqual(config.output_format, "mp3")
        self.assertEqual(config.base_url.rstrip("/"), "https://openrouter.ai/api/v1")
        self.assertTrue(Path(config.api_key_path).name.endswith("api_key_voice.txt"))

    def test_disabled_config_disables_speech(self) -> None:
        service = SpeechService(
            provider=MockTTSProvider(),
            tts_config=TTSConfig(provider=PROVIDER_DISABLED),
            start_worker=False,
        )
        self.addCleanup(service.shutdown)
        result = service.request("hello")
        self.assertEqual(result.state, SpeechState.FAILED)
        self.assertEqual(result.error, "speech_disabled")

    def test_api_key_is_read_from_the_file_at_runtime(self) -> None:
        config = openrouter_config(key_file(self, "sk-or-FROM-DISK-9f2c"))
        provider = OpenRouterTTSProvider(config, urlopen=FakeUrlopen())
        self.assertTrue(provider.has_api_key)
        self.assertEqual(
            provider.build_headers()["Authorization"], "Bearer sk-or-FROM-DISK-9f2c"
        )

    def test_environment_picks_openrouter_only_when_a_key_exists(self) -> None:
        present = key_file(self, "sk-or-ENV-PROBE-1")
        detected = TTSConfig.from_environment({"TTS_API_KEY_PATH": str(present)})
        self.assertEqual(detected.provider, PROVIDER_OPENROUTER)
        missing = TTSConfig.from_environment(
            {"TTS_API_KEY_PATH": str(present.parent / "not-here.txt")}
        )
        self.assertEqual(missing.provider, PROVIDER_MOCK)
        forced = TTSConfig.from_environment({"TTS_PROVIDER": "mock", "TTS_API_KEY_PATH": str(present)})
        self.assertEqual(forced.provider, PROVIDER_MOCK)

    def test_real_key_file_is_never_referenced_by_diagnostics(self) -> None:
        # The shipped default must be the only place the production key lives.
        self.assertEqual(Path(TTSConfig().api_key_path).name, "api_key_voice.txt")
        service = SpeechService(provider=MockTTSProvider(), start_worker=False)
        self.addCleanup(service.shutdown)
        blob = json.dumps([service.get_state(), service.health(), service.summary()])
        self.assertNotIn("api_key_voice", blob)
        self.assertNotIn("Bearer", blob)


class AudioValidationTests(unittest.TestCase):
    def test_probe_recognises_each_container(self) -> None:
        self.assertEqual(probe_audio(mp3_bytes(), "audio/mpeg").format, "mp3")
        wav = wav_container_from_pcm(pcm_bytes(), 24_000)
        self.assertEqual(probe_audio(wav, "audio/wav").format, "wav")
        pcm = probe_audio(pcm_bytes(), "audio/pcm;rate=24000;channels=1")
        self.assertEqual((pcm.format, pcm.sample_rate, pcm.channels), ("pcm", 24_000, 1))

    def test_wav_wrapper_is_decodable(self) -> None:
        payload = wav_container_from_pcm(pcm_bytes(500), 24_000)
        with open_wave(io.BytesIO(payload), "rb") as handle:
            self.assertEqual(handle.getnframes(), 500)
            self.assertEqual(handle.getframerate(), 24_000)

    def test_non_audio_is_rejected(self) -> None:
        self.assertIsNone(probe_audio(b'{"error":"nope"}', "application/json"))
        with self.assertRaises(TTSProviderError) as raised:
            validate_audio_payload(b"not audio at all " * 40, "application/json")
        self.assertEqual(raised.exception.reason, "malformed_audio")

    def test_short_payload_is_rejected(self) -> None:
        with self.assertRaises(TTSProviderError) as raised:
            validate_audio_payload(mp3_bytes(1), "audio/mpeg")
        self.assertEqual(raised.exception.reason, "empty_audio")


class MockProviderTests(unittest.TestCase):
    def test_output_is_deterministic_and_decodable(self) -> None:
        first = MockTTSProvider().synthesize(TTSRequest(text="hello there", model="m", voice="v"))
        second = MockTTSProvider().synthesize(TTSRequest(text="hello there", model="m", voice="v"))
        self.assertEqual(first.payload, second.payload)
        self.assertEqual(first.generation_id, second.generation_id)
        self.assertEqual(first.format, "wav")
        self.assertGreater(first.duration_seconds, 0.0)

    def test_failure_reason_is_preserved(self) -> None:
        provider = MockTTSProvider(fail_with="provider_timeout")
        with self.assertRaises(TTSProviderError) as raised:
            provider.synthesize(TTSRequest(text="hello", model="m", voice="v"))
        self.assertEqual(raised.exception.reason, "provider_timeout")


class OpenRouterTransportTests(unittest.TestCase):
    def test_request_shape(self) -> None:
        config = openrouter_config(key_file(self))
        opener = FakeUrlopen()
        provider = OpenRouterTTSProvider(config, urlopen=opener)
        audio = provider.synthesize(TTSRequest(text="Hello Lacrimosa", model=config.model,
                                              voice=config.voice, output_format="mp3"))

        self.assertEqual(len(opener.calls), 1)
        request = opener.calls[0]["request"]
        self.assertEqual(request.full_url, "https://openrouter.ai/api/v1/audio/speech")
        self.assertEqual(request.method, "POST")
        self.assertEqual(
            json.loads(request.data.decode("utf-8")),
            {
                "model": "deepgram/flux-tts:free",
                "input": "Hello Lacrimosa",
                "voice": "flux-alexis-en",
                "response_format": "mp3",
            },
        )
        self.assertEqual(request.get_header("Authorization"), f"Bearer {SECRET}")
        self.assertEqual(audio.format, "mp3")
        self.assertEqual(audio.generation_id, "gen-tts-fake-1")
        self.assertFalse(audio.streamed)

    def test_timeout_is_forwarded(self) -> None:
        config = openrouter_config(key_file(self))
        opener = FakeUrlopen()
        provider = OpenRouterTTSProvider(config, urlopen=opener)
        provider.synthesize(
            TTSRequest(text="hi", model="m", voice="v", timeout_seconds=3.5),
        )
        self.assertEqual(opener.calls[0]["timeout"], 3.5)

    def test_pcm_is_wrapped_into_a_wav_container(self) -> None:
        config = openrouter_config(key_file(self))
        opener = FakeUrlopen(
            FakeResponse(pcm_bytes(), content_type="audio/pcm;rate=24000;channels=1")
        )
        audio = OpenRouterTTSProvider(config, urlopen=opener).synthesize(
            TTSRequest(text="hi", model="m", voice="v", output_format="pcm")
        )
        self.assertEqual(audio.format, "wav")
        self.assertEqual(probe_audio(audio.payload, "audio/wav").sample_rate, 24_000)

    def test_missing_key_never_reaches_the_network(self) -> None:
        config = openrouter_config(key_file(self).parent / "absent.txt")
        opener = FakeUrlopen()
        provider = OpenRouterTTSProvider(config, urlopen=opener)
        with self.assertRaises(TTSProviderError) as raised:
            provider.synthesize(TTSRequest(text="hi", model="m", voice="v"))
        self.assertEqual(raised.exception.reason, "api_key_unavailable")
        self.assertEqual(opener.calls, [])

    def test_blank_text_is_rejected_before_the_request(self) -> None:
        config = openrouter_config(key_file(self))
        opener = FakeUrlopen()
        with self.assertRaises(TTSProviderError) as raised:
            OpenRouterTTSProvider(config, urlopen=opener).synthesize(
                TTSRequest(text="   ", model="m", voice="v")
            )
        self.assertEqual(raised.exception.reason, "empty_text")
        self.assertEqual(opener.calls, [])

    def test_failure_matrix(self) -> None:
        cases = {
            "http_401": http_error(401),
            "http_500": http_error(500),
            "provider_timeout": TimeoutError(),
            "transport_error": urllib.error.URLError("down"),
            "empty_audio": FakeResponse(b"", content_type="audio/mpeg"),
            "malformed_audio": FakeResponse(b"not audio " * 60, content_type="application/json"),
        }
        for expected, failure in cases.items():
            with self.subTest(reason=expected):
                config = openrouter_config(key_file(self))
                opener = (
                    FakeUrlopen(error=failure)
                    if isinstance(failure, BaseException)
                    else FakeUrlopen(failure)
                )
                provider = OpenRouterTTSProvider(config, urlopen=opener)
                with self.assertRaises(TTSProviderError) as raised:
                    provider.synthesize(TTSRequest(text="hi", model="m", voice="v"))
                self.assertEqual(raised.exception.reason, expected)

    def test_error_body_never_contains_the_key(self) -> None:
        config = openrouter_config(key_file(self))
        opener = FakeUrlopen(error=http_error(429, f"quota for {SECRET}".encode("utf-8")))
        provider = OpenRouterTTSProvider(config, urlopen=opener)
        with self.assertRaises(TTSProviderError) as raised:
            provider.synthesize(TTSRequest(text="hi", model="m", voice="v"))
        self.assertNotIn(SECRET, str(raised.exception))
        self.assertIn("[redacted]", raised.exception.detail)

    def test_credentials_stay_out_of_repr_and_diagnostics(self) -> None:
        config = openrouter_config(key_file(self))
        provider = OpenRouterTTSProvider(config, urlopen=FakeUrlopen())
        self.assertNotIn(SECRET, repr(provider))
        self.assertNotIn(SECRET, json.dumps(provider.describe()))
        self.assertTrue(provider.describe()["api_key_configured"])
        self.assertNotIn(SECRET, json.dumps(config.to_dict()))
        self.assertNotIn(SECRET, json.dumps(config.sanitized().extra_headers))

    def test_optional_identity_headers_are_passthrough_only(self) -> None:
        config = TTSConfig(
            provider=PROVIDER_OPENROUTER,
            api_key_path=key_file(self),
            extra_headers={},
        )
        headers = OpenRouterTTSProvider(config, urlopen=FakeUrlopen()).build_headers()
        self.assertEqual(set(headers), {"Authorization", "Content-Type"})


class SpeechServiceTests(unittest.TestCase):
    def make(self, provider: Optional[TTSProvider] = None, **overrides: Any) -> SpeechService:
        config = SpeechConfig(**overrides)
        service = SpeechService(
            provider=provider or MockTTSProvider(),
            config=config,
            clock=self.clock,
            start_worker=False,
        )
        self.addCleanup(service.shutdown)
        return service

    def setUp(self) -> None:
        self.clock = Clock()

    def test_queue_then_synthesize_then_complete(self) -> None:
        service = self.make()
        request = service.request("Hello there")
        self.assertEqual(request.state, SpeechState.SPEECH_REQUESTED)
        self.assertEqual(service.pending_count(), 1)

        self.assertTrue(service.run_pending_once(0.1))
        self.assertEqual(service.state(request.request_id), SpeechState.READY)
        self.assertTrue(service.mark_playing(request.request_id))
        self.assertEqual(service.state(request.request_id), SpeechState.PLAYING)
        self.assertTrue(service.mark_completed(request.request_id))
        self.assertEqual(service.state(request.request_id), SpeechState.COMPLETED)

        payload, mime, filename = service.audio(request.request_id)
        self.assertEqual(mime, "audio/wav")
        self.assertTrue(filename.startswith(request.request_id))
        self.assertGreater(len(payload), 256)
        metrics = request.metrics.to_dict()
        self.assertIsNotNone(metrics["time_to_first_audio_ms"])
        self.assertIsNotNone(metrics["playback_ms"])
        self.assertIsNotNone(metrics["total_ms"])

    def test_request_validation(self) -> None:
        service = self.make()
        for text, reason in (("", "empty_text"), ("   ", "empty_text"), ("a" * 501, "text_too_long")):
            with self.subTest(text=len(text)):
                request = service.request(text)
                self.assertEqual(request.state, SpeechState.FAILED)
                self.assertEqual(request.error, reason)
        self.assertEqual(service.pending_count(), 0)

    def test_queue_is_bounded_and_rejects_safely(self) -> None:
        service = self.make(queue_capacity=2)
        accepted = [service.request(f"sentence number {index}") for index in range(2)]
        overflow = service.request("one sentence too many")
        self.assertTrue(all(item.state == SpeechState.SPEECH_REQUESTED for item in accepted))
        self.assertEqual(overflow.state, SpeechState.FAILED)
        self.assertEqual(overflow.error, "speech_queue_full")
        self.assertLessEqual(service.pending_count(), 2)
        self.assertEqual(service.get_state()["counters"]["rejected"], 1)

    def test_one_speak_one_generation_one_playback(self) -> None:
        provider = MockTTSProvider()
        service = self.make(provider)
        first = service.request("Hope you enjoyed that!", dedup_key="plan-1:3")
        again = service.request("Hope you enjoyed that!", dedup_key="plan-1:3")
        self.assertEqual(first.request_id, again.request_id)
        self.assertEqual(service.pending_count(), 1)

        service.run_pending_once(0.1)
        self.assertEqual(service.state(first.request_id), SpeechState.READY)
        self.assertEqual(len(provider.calls), 1)

        # A later plan repeating the line reuses the audio instead of generating.
        reused = service.request("Hope you enjoyed that!", dedup_key="plan-2:3")
        self.assertNotEqual(reused.request_id, first.request_id)
        self.assertEqual(reused.state, SpeechState.READY)
        self.assertEqual(reused.reused_from, first.request_id)
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(reused.audio.payload, first.audio.payload)

    def test_mark_playing_is_single_owner(self) -> None:
        service = self.make()
        request = service.request("only one voice")
        self.assertFalse(service.mark_playing(request.request_id))
        service.run_pending_once(0.1)
        self.assertTrue(service.mark_playing(request.request_id))
        self.assertFalse(service.mark_playing("speech_missing"))

    def test_cancel_stops_speech_and_leaves_service_usable(self) -> None:
        service = self.make()
        request = service.request("long sentence that nobody wants to hear")
        self.assertTrue(service.cancel(request.request_id, "viewer_cancelled"))
        self.assertEqual(service.state(request.request_id), SpeechState.CANCELLED)
        service.run_pending_once(0.1)
        self.assertEqual(service.state(request.request_id), SpeechState.CANCELLED)
        self.assertFalse(service.mark_playing(request.request_id))

        next_request = service.request("next line")
        service.run_pending_once(0.1)
        self.assertEqual(service.state(next_request.request_id), SpeechState.READY)

    def test_cancel_while_synthesizing_is_not_resurrected(self) -> None:
        holder: dict[str, SpeechService] = {}
        provider = MockTTSProvider()
        original = provider.synthesize

        def cancelling(request: TTSRequest) -> Any:
            audio = original(request)
            holder["service"].cancel(holder["request_id"], "unity_stopped")
            return audio

        provider.synthesize = cancelling  # type: ignore[method-assign]
        service = self.make(provider)
        request = service.request("interrupt me")
        holder["service"] = service
        holder["request_id"] = request.request_id
        service.run_pending_once(0.1)
        self.assertEqual(service.state(request.request_id), SpeechState.CANCELLED)
        self.assertIsNone(service.audio(request.request_id))

    def test_interrupt_current_cancels_playback(self) -> None:
        service = self.make()
        playing = service.request("current sentence")
        service.run_pending_once(0.1)
        service.mark_playing(playing.request_id)
        self.assertEqual(service.interrupt_current("new_speech"), playing.request_id)
        self.assertEqual(service.state(playing.request_id), SpeechState.CANCELLED)
        self.assertIsNone(service.get_state()["playing_request_id"])

    def test_provider_failure_is_structured_and_recoverable(self) -> None:
        broken = MockTTSProvider(fail_with="http_401")
        service = self.make(broken)
        failed = service.request("this will not play", dedup_key="plan:1")
        service.run_pending_once(0.1)
        self.assertEqual(service.state(failed.request_id), SpeechState.FAILED)
        self.assertEqual(failed.error, "http_401")
        self.assertIsNone(failed.audio)
        self.assertEqual(service.get_state()["consecutive_failures"], 1)
        self.assertEqual(service.health()["status"], "degraded")

        service.provider = MockTTSProvider()
        retry = service.request("this will not play", dedup_key="plan:1")
        self.assertEqual(retry.attempt, 2)
        service.run_pending_once(0.1)
        self.assertEqual(service.state(retry.request_id), SpeechState.READY)
        self.assertEqual(service.get_state()["consecutive_failures"], 0)
        self.assertEqual(service.health()["status"], "ok")

    def test_provider_crash_does_not_kill_the_worker(self) -> None:
        class Crashing(TTSProvider):
            name = "crashing"

            def synthesize(self, request: TTSRequest) -> Any:
                raise RuntimeError("boom")

        service = self.make(Crashing())
        request = service.request("hello")
        service.run_pending_once(0.1)
        self.assertEqual(service.state(request.request_id), SpeechState.FAILED)
        self.assertIn("provider_crash", request.error)

    def test_stale_request_is_dropped(self) -> None:
        service = self.make(stale_after_seconds=1.0)
        request = service.request("too late now")
        self.clock.advance(5.0)
        service.run_pending_once(0.1)
        self.assertEqual(service.state(request.request_id), SpeechState.FAILED)
        self.assertEqual(request.error, "speech_stale")

    def test_cached_audio_is_bounded(self) -> None:
        service = self.make(max_cached_audio=1)
        for index in range(3):
            service.request(f"distinct line {index}")
            service.run_pending_once(0.1)
        self.assertLessEqual(service.get_state()["cached_audio_requests"], 1)

    def test_terminal_requests_are_evicted_after_ttl(self) -> None:
        service = self.make(result_ttl_seconds=1.0)
        finished = service.request("short lived")
        service.run_pending_once(0.1)
        service.mark_playing(finished.request_id)
        service.mark_completed(finished.request_id)
        self.clock.advance(10.0)
        service.request("trigger eviction")
        recent = service.get_state()["recent"]
        self.assertTrue(all(item["text"] != "short lived" for item in recent))

    def test_state_and_summary_expose_no_secrets(self) -> None:
        config = openrouter_config(key_file(self))
        provider = OpenRouterTTSProvider(config, urlopen=FakeUrlopen())
        service = SpeechService(provider=provider, clock=self.clock, start_worker=False)
        self.addCleanup(service.shutdown)
        request = service.request("openrouter line")
        provider_ref = service.provider
        service.run_pending_once(0.1)
        self.assertEqual(service.state(request.request_id), SpeechState.READY)
        blob = json.dumps([service.get_state(), service.health(), service.summary(), request.to_dict()])
        self.assertNotIn(SECRET, blob)
        self.assertNotIn("Authorization", blob)
        self.assertIsInstance(provider_ref, OpenRouterTTSProvider)

    def test_diagnostics_describe_the_active_provider(self) -> None:
        service = self.make()
        state = service.get_state()
        self.assertEqual(state["provider"], "mock")
        self.assertEqual(state["model"], "deepgram/flux-tts:free")
        self.assertEqual(state["voice"], "flux-alexis-en")
        # Nothing in flight has to read as idle; reporting SPEECH_REQUESTED
        # looks like a stuck utterance to anyone watching diagnostics.
        self.assertEqual(state["state"], IDLE_STATE)
        self.assertEqual(service.summary()["state"], IDLE_STATE)
        self.assertFalse(state["streaming_supported"])
        self.assertEqual(state["queue_length"], 0)
        self.assertIsNone(service.get_state()["last_request"])

        # A live utterance reports the lifecycle state it is actually in.
        request = service.request("diagnostic line")
        self.assertEqual(service.get_state()["state"], SpeechState.SPEECH_REQUESTED.value)
        self.assertEqual(service.get_state()["active_request_id"], request.request_id)
        service.run_pending_once(0.1)
        self.assertEqual(service.get_state()["state"], SpeechState.READY.value)
        service.mark_playing(request.request_id)
        self.assertEqual(service.get_state()["state"], SpeechState.PLAYING.value)
        service.mark_completed(request.request_id)
        self.assertEqual(service.get_state()["state"], IDLE_STATE)


class AlwaysFailingProvider(TTSProvider):
    """A provider that never produces audio; the Brain must survive it."""

    name = "crashing"

    def __init__(self) -> None:
        self.calls = 0

    def synthesize(self, request: TTSRequest) -> Any:
        self.calls += 1
        raise TTSProviderError("provider_timeout")


class RateLimitedThenHealthyProvider(TTSProvider):
    """Returns a real rate limit once, then audio; models a provider window reset."""

    name = "rate_limited"

    def __init__(self, retry_after_seconds: Optional[float] = None) -> None:
        self.calls = 0
        self.retry_after_seconds = retry_after_seconds

    def synthesize(self, request: TTSRequest) -> Any:
        self.calls += 1
        if self.calls == 1:
            raise TTSProviderError(
                "http_429",
                "Rate limit exceeded: free-models-per-day",
                retry_after_seconds=self.retry_after_seconds,
            )
        return MockTTSProvider().synthesize(request)


class RateLimitBackoffTests(unittest.TestCase):
    """A rate-limited provider must not be retried on every Brain tick."""

    def setUp(self) -> None:
        self.clock = Clock()

    def make(self, provider: TTSProvider, **overrides: Any) -> SpeechService:
        service = SpeechService(
            provider=provider,
            config=SpeechConfig(**overrides),
            clock=self.clock,
            start_worker=False,
        )
        self.addCleanup(service.shutdown)
        return service

    def test_rate_limit_pauses_provider_and_recovers_after_cooldown(self) -> None:
        provider = RateLimitedThenHealthyProvider()
        service = self.make(provider, rate_limit_backoff_seconds=120.0)

        first = service.request("Can you dance for me?")
        service.run_pending_once(0.1)
        self.assertEqual(first.state, SpeechState.FAILED)
        self.assertEqual(first.error, "http_429")
        self.assertEqual(provider.calls, 1)

        health = service.health()
        self.assertTrue(health["last_error_detail"].startswith("Rate limit exceeded"))

        # While paused the service answers immediately: no queueing, no attempt.
        second = service.request("Second line while rate limited")
        self.assertEqual(second.state, SpeechState.FAILED)
        self.assertEqual(second.error, "speech_rate_limited")
        self.assertEqual(provider.calls, 1)
        self.assertEqual(service.pending_count(), 0)
        self.assertTrue(service.get_state()["paused"])

        self.clock.advance(121)
        third = service.request("Speech after the provider window resets")
        self.assertTrue(service.run_pending_once(0.1))
        self.assertEqual(third.state, SpeechState.READY)
        self.assertEqual(provider.calls, 2)
        self.assertFalse(service.get_state()["paused"])

    def test_provider_declared_cooldown_is_honoured_within_bounds(self) -> None:
        provider = RateLimitedThenHealthyProvider(retry_after_seconds=15.0)
        service = self.make(
            provider,
            rate_limit_backoff_seconds=300.0,
            max_rate_limit_backoff_seconds=3600.0,
        )
        service.request("First")
        service.run_pending_once(0.1)
        remaining = service.get_state()["paused_seconds_remaining"]
        self.assertLessEqual(remaining, 15.0)
        self.assertGreater(remaining, 0.0)

        # An absurd reset stamp cannot park the service longer than the ceiling.
        provider_calls = RateLimitedThenHealthyProvider(retry_after_seconds=99_999.0)
        capped = self.make(provider_calls, max_rate_limit_backoff_seconds=60.0)
        capped.request("First")
        capped.run_pending_once(0.1)
        self.assertLessEqual(capped.get_state()["paused_seconds_remaining"], 60.0)


class BrainSpeechIntegrationTests(unittest.TestCase):
    def snapshot(self, streaming: bool = True) -> dict[str, Any]:
        return {
            "character": {"position": [0, 0, 0], "rotation_y": 90, "current_action": "idle"},
            "phone": {"state": "Placed", "position": [1, 0.75, 0], "placement_surface": "TABLE"},
            "stream": {"is_streaming": streaming, "camera_available": True},
            "targets": {
                "CENTER": {"position": [0, 0, 0], "navigation_target": True, "look_at_target": True},
                "TABLE": {"position": [1, 0.75, 0], "navigation_target": True, "look_at_target": True},
                "CAMERA": {"position": [1, 0.8, 0], "navigation_target": False, "look_at_target": True},
            },
        }

    def make_brain(self, provider: Optional[TTSProvider] = None) -> tuple[AutonomousBrain, SpeechService]:
        speech = SpeechService(
            provider=provider or MockTTSProvider(),
            clock=Clock(),
            start_worker=False,
        )
        self.addCleanup(speech.shutdown)
        chat = ChatService(MockLLMProvider(), ChatServiceConfig(max_messages_per_rate_window=20))
        brain = AutonomousBrain(
            MockBrainLLMProvider(),
            chat,
            BrainConfig(decision_interval_seconds=0.0),
            speech_service=speech,
        )
        return brain, speech

    def world(self, streaming: bool = True) -> Any:
        return build_world_state(self.snapshot(streaming=streaming))

    def dispatch(self, brain: AutonomousBrain, result: Any = None, streaming: bool = True) -> Any:
        """One tick, reported as the action it dispatched (or the pending one)."""
        return brain.tick(self.world(streaming), confirm(brain, result)).action

    def run_to_speak(self, brain: AutonomousBrain, message: str = "Lacrimosa can you dance?") -> Any:
        """Drive the real loop step by step until it emits a SPEAK action."""
        brain.chat.submit("viewer_phase11", message)
        action = self.dispatch(brain, streaming=True)
        for _ in range(8):
            if action is None or action.action == ActionType.SPEAK:
                break
            action = self.dispatch(brain, ActionResult(action.action, ExecutionStatus.SUCCESS))
        return action

    def test_speak_step_carries_a_speech_request(self) -> None:
        brain, speech = self.make_brain()
        provider = speech.provider
        action = self.run_to_speak(brain)
        self.assertIsNotNone(action)
        self.assertEqual(action.action, ActionType.SPEAK)
        self.assertIsNotNone(action.speech_request_id)
        self.assertEqual(speech.state(action.speech_request_id), SpeechState.SPEECH_REQUESTED)

        speech.run_pending_once(0.1)
        self.assertEqual(speech.state(action.speech_request_id), SpeechState.READY)
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(speech.audio(action.speech_request_id)[1], "audio/wav")
        self.assertIn("speech_request_id", action.to_dict())
        self.assertEqual(action.to_dict()["speech_request_id"], action.speech_request_id)

    def test_repeated_ticks_do_not_regenerate_speech(self) -> None:
        brain, speech = self.make_brain()
        provider = speech.provider
        action = self.run_to_speak(brain)
        speech.run_pending_once(0.1)

        # Unity has not reported back yet: further ticks re-offer the same action
        # and must not touch the provider again.
        pending = self.dispatch(brain)
        self.assertEqual(pending.speech_request_id, action.speech_request_id)
        self.assertEqual(pending.action, ActionType.SPEAK)
        self.assertEqual(len(provider.calls), 1)

        # A later plan that repeats the same line reuses the cached audio.
        self.dispatch(brain, ActionResult(ActionType.SPEAK, ExecutionStatus.SUCCESS))
        replayed = self.run_to_speak(brain, "Lacrimosa dance again please")
        self.assertEqual(replayed.action, ActionType.SPEAK)
        self.assertNotEqual(replayed.speech_request_id, action.speech_request_id)
        self.assertEqual(speech.state(replayed.speech_request_id), SpeechState.READY)
        self.assertEqual(len(provider.calls), 1)

    def test_tts_failure_leaves_the_brain_operational(self) -> None:
        provider = AlwaysFailingProvider()
        brain, speech = self.make_brain(provider)
        action = self.run_to_speak(brain)
        self.assertIsNotNone(action.speech_request_id)
        speech.run_pending_once(0.1)
        self.assertEqual(speech.state(action.speech_request_id), SpeechState.FAILED)

        # The failed utterance is reported back like any other action result, and
        # the loop keeps planning instead of wedging.
        result = brain.tick(
            self.world(),
            confirm(brain, ActionResult(ActionType.SPEAK, ExecutionStatus.FAILED)),
        )
        # The loop retries the step under the same action identity rather than
        # regenerating speech, so a provider outage cannot spam OpenRouter.
        self.assertEqual(result.status, BrainStatus.EXECUTING)
        self.assertEqual(result.action.speech_request_id, action.speech_request_id)
        self.assertEqual(speech.get_state()["consecutive_failures"], 1)
        self.assertEqual(provider.calls, 1)

        # Recovery: the loop keeps acting, and a later utterance still reaches
        # the provider instead of being locked out by the earlier failure.
        after = self.dispatch(brain, ActionResult(ActionType.SPEAK, ExecutionStatus.SUCCESS))
        self.assertIsNot(after, action)
        speech.provider = MockTTSProvider()
        follow_up = speech.request("Recovery line", dedup_key="plan:2")
        speech.run_pending_once(0.1)
        self.assertEqual(speech.state(follow_up.request_id), SpeechState.READY)
        self.assertEqual(provider.calls, 1)
        self.assertEqual(speech.get_state()["consecutive_failures"], 0)

    def test_speech_without_service_is_inert(self) -> None:
        chat = ChatService(MockLLMProvider(), ChatServiceConfig(max_messages_per_rate_window=20))
        brain = AutonomousBrain(
            MockBrainLLMProvider(),
            chat,
            BrainConfig(decision_interval_seconds=0.0),
        )
        chat.submit("viewer_phase11", "Lacrimosa can you dance?")
        action = brain.tick(self.world()).action
        self.assertIsNotNone(action)
        # Genuine coverage: the tick actually served the viewer turn (the
        # chat-driven dance plan), it is not a stream-maintenance plan.
        self.assertEqual(action.action, ActionType.MOVE_TO)
        self.assertIsNone(action.speech_request_id)

    # ------------------------------------------------- dialogue voicing (PHASE 11)

    def brain_for(self, speech: Optional[SpeechService]) -> AutonomousBrain:
        chat = ChatService(MockLLMProvider(), ChatServiceConfig(max_messages_per_rate_window=20))
        return AutonomousBrain(
            MockBrainLLMProvider(),
            chat,
            BrainConfig(decision_interval_seconds=0.0),
            speech_service=speech,
        )

    def live_speech(self) -> SpeechService:
        service = SpeechService(provider=MockTTSProvider(), clock=Clock(), start_worker=False)
        self.addCleanup(service.shutdown)
        return service

    def planned_steps(self, brain: AutonomousBrain, message: Optional[str]) -> tuple:
        """The plan the Brain commits to for the first decision it takes."""
        if message is not None:
            brain.chat.submit("viewer_phase11", message)
        brain.tick(self.world())
        plan = brain.state.current_plan
        return () if plan is None else plan.steps

    def test_chat_reply_is_voiced_through_the_existing_speak_action(self) -> None:
        speech = self.live_speech()
        brain = self.brain_for(speech)
        steps = self.planned_steps(brain, "Hello Lacrimosa")
        spoken = [step for step in steps if step.action == ActionType.SPEAK]
        self.assertEqual(len(spoken), 1)
        self.assertEqual(spoken[0].text, "Hello! It's good to see you.")

        # The plan still runs STOP first; the utterance is only handed to the
        # speech service once its own step dispatches.
        self.assertEqual(speech.get_state()["queue_length"], 0)
        world = self.world()
        brain.tick(world, confirm(brain, ActionResult(ActionType.STOP, ExecutionStatus.SUCCESS)))
        action = brain.tick(world).action
        self.assertEqual(action.action, ActionType.SPEAK)
        self.assertEqual(action.text, "Hello! It's good to see you.")
        self.assertIsNotNone(action.speech_request_id)
        self.assertEqual(speech.state(action.speech_request_id), SpeechState.SPEECH_REQUESTED)
        self.assertEqual(speech.get_state()["queue_length"], 1)

    def test_disabled_speech_leaves_the_plan_untouched(self) -> None:
        speech = SpeechService(
            provider=MockTTSProvider(),
            tts_config=TTSConfig(provider=PROVIDER_DISABLED),
            clock=Clock(),
            start_worker=False,
        )
        self.addCleanup(speech.shutdown)
        brain = self.brain_for(speech)
        steps = self.planned_steps(brain, "Hello Lacrimosa")
        self.assertFalse([step for step in steps if step.action == ActionType.SPEAK])
        self.assertEqual(steps[0].action, ActionType.STOP)
        # The SPEAK-free plan still dispatches without a speech id, so Unity's
        # legacy path runs and the loop is never waiting on voice.
        self.assertIsNone(brain.tick(self.world()).action.speech_request_id)

    def test_missing_speech_service_leaves_the_plan_untouched(self) -> None:
        steps = self.planned_steps(self.brain_for(None), "Hello Lacrimosa")
        self.assertFalse([step for step in steps if step.action == ActionType.SPEAK])

    def test_dialogue_never_double_voices_a_plan_that_already_speaks(self) -> None:
        steps = self.planned_steps(self.brain_for(self.live_speech()), "Lacrimosa can you dance?")
        spoken = [step.text for step in steps if step.action == ActionType.SPEAK]
        self.assertEqual(spoken, ["Hope you enjoyed that!"])
        self.assertEqual([step.action for step in steps][-1], ActionType.SPEAK)

    def test_non_viewer_goals_are_never_voiced(self) -> None:
        speech = self.live_speech()
        steps = self.planned_steps(self.brain_for(speech), None)
        self.assertFalse([step for step in steps if step.action == ActionType.SPEAK])
        self.assertEqual(speech.get_state()["queue_length"], 0)

    def test_dialogue_voicing_rules(self) -> None:
        """The guard conditions live on one method; each edge is checked directly."""
        brain = self.brain_for(self.live_speech())
        plan = Plan(plan_id="plan_1", goal_id="goal_1", reason="test",
                    steps=(PlanStep(action=ActionType.STOP),))
        goal = Goal(goal_id="goal_1", goal_type=GoalType.RESPOND_TO_VIEWER, priority=1.0,
                    reason="a viewer spoke")

        def processed(speech_text: Any) -> Any:
            return SimpleNamespace(response=SimpleNamespace(speech=speech_text))

        self.assertIs(
            brain._append_dialogue_step(plan, goal, processed("x" * (MAX_TEXT_LENGTH + 1))),
            plan,
        )
        self.assertIs(brain._append_dialogue_step(plan, goal, processed("   ")), plan)
        self.assertIs(brain._append_dialogue_step(plan, goal, processed(None)), plan)
        self.assertIs(brain._append_dialogue_step(plan, goal, None), plan)
        self.assertIs(
            brain._append_dialogue_step(
                plan, Goal(goal_id="g", goal_type=GoalType.REST, priority=0.1, reason="rest"),
                processed("Hi there"),
            ),
            plan,
        )

        extended = brain._append_dialogue_step(plan, goal, processed("Hi there"))
        self.assertEqual([step.action for step in extended.steps],
                         [ActionType.STOP, ActionType.SPEAK])
        self.assertEqual(extended.steps[-1].text, "Hi there")
        # Voice only ever adds: the original plan object is never mutated.
        self.assertEqual(len(plan.steps), 1)


class SpeechApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.speech = SpeechService(
            provider=MockTTSProvider(),
            clock=Clock(),
            start_worker=False,
        )
        self.addCleanup(self.speech.shutdown)
        self.service = create_default_service(
            database_path=f"{self.directory.name}/learning.db",
            speech_service=self.speech,
        )
        self.server = create_brain_server(self.service, "127.0.0.1", 0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.teardown_server)
        host, port = self.server.server_address
        self.base = f"http://{host}:{port}"
        self.request = self.speech.request("routing check")

    def teardown_server(self) -> None:
        self.server.shutdown()
        self.thread.join(timeout=2)
        self.server.server_close()
        self.service.brain.learning.close()

    def get(self, path: str) -> tuple[int, bytes, str]:
        try:
            with urllib.request.urlopen(urllib.request.Request(f"{self.base}{path}"), timeout=5) as response:
                return response.status, response.read(), response.headers.get("Content-Type", "")
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read(), exc.headers.get("Content-Type", "")

    def post(self, path: str) -> tuple[int, dict[str, Any]]:
        request = urllib.request.Request(
            f"{self.base}{path}",
            data=b"",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode("utf-8"))

    def test_tts_diagnostics_routes(self) -> None:
        status, payload, mime = self.get("/tts/state")
        self.assertEqual((status, mime.split(";")[0]), (200, "application/json"))
        state = json.loads(payload.decode("utf-8"))
        self.assertEqual(state["provider"], "mock")
        self.assertEqual(state["voice"], "flux-alexis-en")
        self.assertEqual(state["queue_length"], 1)

        status, payload, _ = self.get("/tts/health")
        health = json.loads(payload.decode("utf-8"))
        self.assertEqual((status, health["status"]), (200, "ok"))
        self.assertNotIn("api_key", json.dumps(health))

    def test_request_state_and_audio_routes(self) -> None:
        request_id = self.request.request_id
        status, payload, _ = self.get(f"/speech/state/{request_id}")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(payload.decode("utf-8"))["state"], "SPEECH_REQUESTED")

        status, payload, _ = self.get(f"/speech/audio/{request_id}")
        self.assertEqual(status, 404)
        self.assertEqual(json.loads(payload.decode("utf-8"))["error"], "audio_not_ready")

        self.speech.run_pending_once(0.1)
        status, payload, mime = self.get(f"/speech/audio/{request_id}")
        self.assertEqual((status, mime), (200, "audio/wav"))
        self.assertEqual(payload[:4], b"RIFF")
        self.assertGreater(len(payload), 256)

        status, payload, _ = self.get("/speech/state/speech_missing")
        self.assertEqual((status, json.loads(payload.decode("utf-8"))["error"]), (404, "speech_request_not_found"))

    def test_lifecycle_routes(self) -> None:
        request_id = self.request.request_id
        self.speech.run_pending_once(0.1)
        self.assertEqual(self.post(f"/speech/playing/{request_id}")[1], {"updated": True})
        self.assertEqual(self.post(f"/speech/completed/{request_id}")[1], {"updated": True})
        self.assertEqual(self.speech.state(request_id), SpeechState.COMPLETED)
        self.assertEqual(self.post(f"/speech/cancel/{request_id}")[1], {"cancelled": False})
        self.assertEqual(self.post("/speech/unknown/x")[0], 404)

    def test_playback_flag_is_rejected_before_audio_exists(self) -> None:
        status, payload = self.post(f"/speech/playing/{self.request.request_id}")
        self.assertEqual((status, payload), (200, {"updated": False}))


if __name__ == "__main__":
    unittest.main()
