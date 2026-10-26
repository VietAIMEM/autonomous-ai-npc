"""PHASE 21 PART 2: Kokoro local TTS provider tests (deterministic, offline).

The real 82M model is never loaded here; a stub worker speaks the exact same
JSON protocol so the provider's process handling, validation, restart and
shutdown behaviour stay testable without torch. The isolated real synthesis
probe lives in ``Temp/phase21/kokoro_probe.py``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import threading
import time
import unittest
import wave
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

from backend.speech.service import SpeechService
from backend.speech.models import SpeechState
from backend.tts.base import TTSProviderError, TTSRequest
from backend.tts.config import (
    PROVIDER_KOKORO,
    PROVIDER_MOCK,
    PROVIDER_OPENROUTER,
    TTSConfig,
    build_provider,
)
from backend.tts.kokoro import KokoroTTSProvider
from backend.tts.mock import MockTTSProvider
from backend.tts.openrouter import OpenRouterTTSProvider


STUB_WORKER = textwrap.dedent(
    '''
    import json
    import os
    import sys
    import wave
    from array import array
    from pathlib import Path

    FAIL = os.environ.get("STUB_FAIL") == "1"
    NOISE = os.environ.get("STUB_NOISE") == "1"
    EXIT_AFTER = os.environ.get("STUB_EXIT_AFTER") == "1"

    def write_wav(path, seconds=0.2, rate=24000):
        path.parent.mkdir(parents=True, exist_ok=True)
        samples = array("h", [1200 if i % 4 < 2 else -1200 for i in range(int(rate * seconds))])
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(rate)
            handle.writeframes(samples.tobytes())

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        message = json.loads(line)
        op = message.get("op")
        if NOISE:
            print("library noise on stdout")
        if op == "ping":
            reply = {"id": message["id"], "ok": True, "loaded": False}
        elif op == "warmup":
            reply = {
                "id": message["id"],
                "ok": True,
                "model": "stub-kokoro",
                "sample_rate": 24000,
                "model_load_ms": 1.5,
                "voices": [message.get("voice")],
            }
        elif FAIL:
            reply = {
                "id": message["id"],
                "ok": False,
                "error": "stub_failure",
                "detail": "synthetic worker failure",
            }
        else:
            path = Path(message["output_path"])
            write_wav(path)
            reply = {
                "id": message["id"],
                "ok": True,
                "output_path": str(path),
                "sample_rate": 24000,
                "channels": 1,
                "duration_seconds": 0.2,
                "chunks": 1,
                "synthesis_ms": 5.0,
                "generation_id": "stub-generation",
            }
        sys.stdout.write(json.dumps(reply) + "\\n")
        sys.stdout.flush()
        if op == "synthesize" and EXIT_AFTER:
            break
    '''
).lstrip()


class KokoroHarness:
    def __init__(self, directory: Path, environ: dict[str, str] | None = None) -> None:
        self.worker = directory / "stub_kokoro_worker.py"
        self.worker.write_text(STUB_WORKER, encoding="utf-8")
        env = dict(os.environ)
        env.update(environ or {})
        self.config = TTSConfig(
            provider=PROVIDER_KOKORO,
            voice="af_heart",
            output_format="wav",
            kokoro_python=sys.executable,
            timeout_seconds=10.0,
            kokoro_warmup_timeout_seconds=10.0,
        )
        self.provider = KokoroTTSProvider(
            self.config,
            python_executable=sys.executable,
            worker_script=self.worker,
            environ=env,
        )

    def request(self, text: str = "Hello from the stub worker.") -> TTSRequest:
        return TTSRequest(
            text=text,
            model=self.config.model,
            voice=self.config.voice,
            output_format="wav",
        )


class KokoroConfigTests(unittest.TestCase):
    def test_kokoro_selection_uses_wav_english_defaults(self) -> None:
        config = TTSConfig.from_environment({"TTS_PROVIDER": "kokoro"})
        self.assertEqual(config.provider, PROVIDER_KOKORO)
        self.assertEqual(config.voice, "af_heart")
        self.assertEqual(config.output_format, "wav")
        self.assertEqual(config.timeout_seconds, 120.0)
        self.assertTrue(str(config.kokoro_python).endswith("python.exe"))
        self.assertTrue(config.enabled)

    def test_openrouter_defaults_are_untouched(self) -> None:
        config = TTSConfig.from_environment({"TTS_PROVIDER": "openrouter"})
        self.assertEqual(config.voice, "flux-alexis-en")
        self.assertEqual(config.output_format, "mp3")
        self.assertEqual(config.timeout_seconds, 25.0)

    def test_build_provider_keeps_every_provider_available(self) -> None:
        self.assertIsInstance(
            build_provider(TTSConfig(provider=PROVIDER_KOKORO)), KokoroTTSProvider
        )
        self.assertIsInstance(
            build_provider(TTSConfig(provider=PROVIDER_OPENROUTER)),
            OpenRouterTTSProvider,
        )
        self.assertIsInstance(
            build_provider(TTSConfig(provider=PROVIDER_MOCK)), MockTTSProvider
        )

    def test_unknown_provider_still_degrades_to_mock(self) -> None:
        self.assertIsInstance(
            build_provider(TTSConfig(provider="something-else")), MockTTSProvider
        )

    def test_sanitized_config_has_no_secret_and_keeps_paths(self) -> None:
        config = TTSConfig.from_environment(
            {"TTS_PROVIDER": "kokoro", "TTS_KOKORO_PYTHON": r"D:\kokoro_tts\.venv\Scripts\python.exe"}
        )
        sanitized = config.sanitized()
        self.assertEqual(sanitized.provider, PROVIDER_KOKORO)
        self.assertEqual(Path(sanitized.api_key_path).name, "api_key_voice.txt")
        self.assertIn("kokoro", json.dumps(sanitized.to_dict()))


class KokoroProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.directory = Path(self._tmp.name)

    def test_synthesize_returns_validated_wav_from_worker(self) -> None:
        harness = KokoroHarness(self.directory)
        self.addCleanup(harness.provider.close)
        audio = harness.provider.synthesize(harness.request())

        self.assertEqual(audio.provider, "kokoro")
        self.assertEqual(audio.format, "wav")
        self.assertEqual(audio.probe.sample_rate, 24_000)
        self.assertEqual(audio.probe.channels, 1)
        self.assertGreater(audio.byte_length, 256)
        self.assertAlmostEqual(audio.duration_seconds or 0.0, 0.2, places=2)
        self.assertEqual(audio.generation_id, "stub-generation")
        self.assertTrue(harness.provider.worker_running)

        with wave.open(BytesIO(audio.payload), "rb") as handle:
            self.assertEqual(handle.getframerate(), 24_000)

    def test_warmup_runs_once_per_voice(self) -> None:
        harness = KokoroHarness(self.directory)
        self.addCleanup(harness.provider.close)
        harness.provider.synthesize(harness.request("First sentence."))
        harness.provider.synthesize(harness.request("Second sentence."))
        self.assertEqual(harness.provider.describe()["warm"], True)
        self.assertEqual(
            harness.provider.describe()["model_load_ms"], 1.5
        )

    def test_stdout_noise_does_not_break_the_protocol(self) -> None:
        harness = KokoroHarness(self.directory, {"STUB_NOISE": "1"})
        self.addCleanup(harness.provider.close)
        audio = harness.provider.synthesize(harness.request())
        self.assertEqual(audio.format, "wav")

    def test_worker_error_is_reported_as_provider_failure(self) -> None:
        harness = KokoroHarness(self.directory, {"STUB_FAIL": "1"})
        self.addCleanup(harness.provider.close)
        with self.assertRaises(TTSProviderError) as caught:
            harness.provider.synthesize(harness.request())
        self.assertEqual(caught.exception.reason, "kokoro_synthesis_failed")
        self.assertIn("synthetic worker failure", caught.exception.detail)
        self.assertFalse(harness.provider.worker_running)

    def test_provider_restarts_a_dead_worker(self) -> None:
        harness = KokoroHarness(self.directory, {"STUB_EXIT_AFTER": "1"})
        self.addCleanup(harness.provider.close)
        first = harness.provider.synthesize(harness.request("Dies after this."))
        self.assertEqual(first.provider, "kokoro")
        second = harness.provider.synthesize(harness.request("Fresh worker."))
        self.assertEqual(second.provider, "kokoro")

    def test_missing_python_is_a_clear_failure(self) -> None:
        provider = KokoroTTSProvider(
            TTSConfig(provider=PROVIDER_KOKORO),
            python_executable=self.directory / "not-a-python.exe",
            worker_script=self.directory / "stub_kokoro_worker.py",
        )
        self.addCleanup(provider.close)
        with self.assertRaises(TTSProviderError) as caught:
            provider.synthesize(
                TTSRequest(text="Hello", model="m", voice="af_heart", output_format="wav")
            )
        self.assertEqual(caught.exception.reason, "kokoro_python_unavailable")

    def test_empty_text_is_rejected_before_the_worker_starts(self) -> None:
        harness = KokoroHarness(self.directory)
        self.addCleanup(harness.provider.close)
        with self.assertRaises(TTSProviderError) as caught:
            harness.provider.synthesize(harness.request("   "))
        self.assertEqual(caught.exception.reason, "empty_text")
        self.assertFalse(harness.provider.worker_running)

    def test_warmup_loads_the_model_without_an_utterance(self) -> None:
        harness = KokoroHarness(self.directory)
        self.addCleanup(harness.provider.close)
        self.assertFalse(harness.provider.worker_running)

        self.assertTrue(harness.provider.warmup())
        self.assertTrue(harness.provider.worker_running)
        described = harness.provider.describe()
        self.assertEqual(described["warm"], True)
        self.assertEqual(described["model_load_ms"], 1.5)

    def test_warmup_without_a_worker_is_a_reported_failure(self) -> None:
        provider = KokoroTTSProvider(
            TTSConfig(provider=PROVIDER_KOKORO),
            python_executable=self.directory / "not-a-python.exe",
            worker_script=self.directory / "stub_kokoro_worker.py",
        )
        self.addCleanup(provider.close)
        self.assertFalse(provider.warmup())
        self.assertTrue(
            str(provider.describe()["last_error"]).endswith("not-a-python.exe")
        )

    def test_close_does_not_block_on_an_in_flight_synthesis(self) -> None:
        harness = KokoroHarness(self.directory)
        provider = harness.provider
        self.assertTrue(provider.warmup())
        self.assertTrue(provider.worker_running)

        release = threading.Event()
        holding = threading.Event()

        def hold_provider_lock() -> None:
            with provider._lock:  # the "synthesis in flight" stand-in
                holding.set()
                release.wait(15.0)

        holder = threading.Thread(target=hold_provider_lock, daemon=True)
        holder.start()
        self.assertTrue(holding.wait(5.0))
        started = time.monotonic()
        provider.close()
        elapsed = time.monotonic() - started
        release.set()
        holder.join(5.0)

        # Shutdown stops the worker instead of waiting out the provider timeout.
        self.assertLess(elapsed, 4.0)
        self.assertFalse(provider.worker_running)


class KokoroSpeechServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.directory = Path(self._tmp.name)

    def test_speech_service_delivers_kokoro_wav_and_closes_worker(self) -> None:
        harness = KokoroHarness(self.directory)
        service = SpeechService(
            provider=harness.provider,
            tts_config=harness.config,
            start_worker=False,
        )
        request = service.request("Brain speech through the local provider.")
        self.assertTrue(service.run_pending_once(timeout=5.0))
        self.assertEqual(service.state(request.request_id), SpeechState.READY)

        payload, mime_type, name = service.audio(request.request_id) or (b"", "", "")
        self.assertTrue(payload.startswith(b"RIFF"))
        self.assertEqual(mime_type, "audio/wav")
        self.assertTrue(name.endswith(".wav"))
        self.assertEqual(service.provider_name, "kokoro")

        service.shutdown()
        self.assertFalse(harness.provider.worker_running)
        summary = service.summary()
        self.assertEqual(summary["provider"], "kokoro")
        self.assertEqual(summary["format"], "wav")

    def test_speech_service_start_warms_the_local_provider(self) -> None:
        harness = KokoroHarness(self.directory)
        service = SpeechService(provider=harness.provider, tts_config=harness.config)
        deadline = time.monotonic() + 10.0
        while not harness.provider.describe()["warm"] and time.monotonic() < deadline:
            time.sleep(0.05)
        try:
            # A warm worker means the first real utterance skips the model load.
            self.assertTrue(harness.provider.worker_running)
            self.assertEqual(harness.provider.describe()["warm"], True)
        finally:
            service.shutdown()
        self.assertFalse(harness.provider.worker_running)


class KokoroWorkerScriptTests(unittest.TestCase):
    """The shipped worker must run under a plain interpreter for its ping op."""

    def test_worker_ping_responds_without_loading_the_model(self) -> None:
        worker = Path(__file__).resolve().parents[1] / "tts" / "kokoro_worker.py"
        process = subprocess.run(
            [sys.executable, "-u", str(worker)],
            input=json.dumps({"id": "1", "op": "ping"}) + "\n",
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        reply = json.loads(process.stdout.strip().splitlines()[-1])
        self.assertTrue(reply["ok"])
        self.assertEqual(reply["id"], "1")
        self.assertFalse(reply["loaded"])


if __name__ == "__main__":
    unittest.main()
