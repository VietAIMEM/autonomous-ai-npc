"""Local Kokoro-82M synthesis through a persistent worker process.

The checkpoint lives in its own virtual environment (``D:\\kokoro_tts``);
this provider talks to it over newline-delimited JSON so the backend keeps
zero heavyweight dependencies. The worker is started on demand and kept warm
because reloading an 82M checkpoint per utterance would cost tens of seconds.
"""

from __future__ import annotations

import atexit
import json
import os
import queue
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Optional

from .base import (
    TTSProvider,
    TTSProviderError,
    TTSAudio,
    TTSRequest,
    validate_audio_payload,
)
from .config import TTSConfig


WORKER_SCRIPT = Path(__file__).with_name("kokoro_worker.py")
DEFAULT_LANG_CODE = "a"
MAX_LOG_TAIL = 2_000
MAX_ERROR_DETAIL = 300
MAX_NOISE_LINES = 5
#: How long shutdown waits for an in-flight synthesis before it stops the
#: worker directly. A runtime stop must never sit behind a provider timeout.
CLOSE_LOCK_TIMEOUT_SECONDS = 2.0


class KokoroTTSProvider(TTSProvider):
    """Turns speech text into WAV bytes using the local Kokoro-82M model."""

    name = "kokoro"
    streaming_supported = False

    def __init__(
        self,
        config: TTSConfig | None = None,
        *,
        popen: Any = subprocess.Popen,
        python_executable: Path | str | None = None,
        worker_script: Path | str | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self.config = config or TTSConfig(provider="kokoro")
        self._popen = popen
        self._python = str(python_executable or self.config.kokoro_python or "")
        self._worker_script = Path(worker_script or WORKER_SCRIPT)
        self._environ = dict(os.environ if environ is None else environ)
        self._lock = threading.RLock()
        self._process: Optional[Any] = None
        self._responses: "queue.Queue[Any]" = queue.Queue()
        self._reader: Optional[threading.Thread] = None
        self._log_handle = None
        self._request_counter = 0
        self._warm_voices: set[str] = set()
        self._model_load_ms: Optional[float] = None
        self._last_synthesis_ms: Optional[float] = None
        self._last_error = ""
        self._closed = False
        atexit.register(self.close)

    # ------------------------------------------------------------------ state

    @property
    def worker_running(self) -> bool:
        process = self._process
        return bool(process is not None and process.poll() is None)

    @property
    def python_available(self) -> bool:
        try:
            return bool(self._python) and Path(self._python).is_file()
        except OSError:
            return False

    def __repr__(self) -> str:  # never surface more than paths
        return f"KokoroTTSProvider(python={self._python!r}, voice={self.config.voice!r})"

    def describe(self) -> dict[str, object]:
        return {
            "provider": self.name,
            "model": self.config.model,
            "voice": self.config.voice,
            "format": self.config.output_format,
            "streaming_supported": self.streaming_supported,
            "python": self._python,
            "python_available": self.python_available,
            "worker_running": self.worker_running,
            "warm": bool(self._warm_voices),
            "model_load_ms": self._model_load_ms,
            "last_synthesis_ms": self._last_synthesis_ms,
            "last_error": self._last_error or None,
        }

    # ------------------------------------------------------------- public API

    def warmup(self) -> bool:
        """Load the checkpoint and voice into a resident worker.

        The first local synthesis otherwise pays interpreter start plus model
        load, which can outlast a playback client's patience. ``SpeechService``
        calls this in the background at startup; a failed warmup is not an
        error because the next ``synthesize`` retries it inline.
        """
        with self._lock:
            if self._closed:
                return False
            voice = (self.config.voice or "").strip()
            if not voice:
                self._last_error = "voice_unavailable"
                return False
            try:
                self._warm_locked(voice)
            except TTSProviderError as exc:
                self._last_error = str(exc.detail or exc.reason)[:MAX_ERROR_DETAIL]
                return False
            return True

    def synthesize(self, request: TTSRequest) -> TTSAudio:
        body = (request.text or "").strip()
        if not body:
            raise TTSProviderError("empty_text")
        voice = (request.voice or self.config.voice or "").strip()
        if not voice:
            raise TTSProviderError("voice_unavailable", "no Kokoro voice configured")
        speed = request.speed or self.config.speed or 1.0
        timeout = request.timeout_seconds or self.config.timeout_seconds

        handle, output_path = tempfile.mkstemp(prefix="kokoro_", suffix=".wav")
        os.close(handle)
        try:
            with self._lock:
                if self._closed:
                    raise TTSProviderError("kokoro_closed")
                self._warm_locked(voice)
                message = {
                    "op": "synthesize",
                    "text": body,
                    "voice": voice,
                    "speed": speed,
                    "lang_code": DEFAULT_LANG_CODE,
                    "output_path": output_path,
                }
                # A worker that died between utterances is respawned once, not
                # reported as a failed utterance: that is the difference
                # between a local provider and a fragile one.
                for attempt in (1, 2):
                    try:
                        response = self._exchange_locked(
                            message, float(timeout), failure_reason="kokoro_synthesis_failed"
                        )
                        break
                    except TTSProviderError as exc:
                        if exc.reason != "kokoro_worker_crashed" or attempt == 2:
                            raise
                        self._warm_locked(voice)
                self._last_synthesis_ms = _as_float(response.get("synthesis_ms"))
            try:
                payload = Path(output_path).read_bytes()
            except OSError as exc:
                raise TTSProviderError("kokoro_audio_missing", type(exc).__name__) from None
        finally:
            try:
                os.unlink(output_path)
            except OSError:
                pass

        probe = validate_audio_payload(payload, "audio/wav")
        generation = response.get("generation_id") or (
            f"kokoro-{int(time.time() * 1000)}-{self._request_counter}"
        )
        return TTSAudio(
            payload=payload,
            probe=probe,
            provider=self.name,
            generation_id=str(generation),
            streamed=False,
        )

    def close(self) -> None:
        """Stop the worker; called by SpeechService at runtime shutdown."""
        self._closed = True
        if not self._lock.acquire(timeout=CLOSE_LOCK_TIMEOUT_SECONDS):
            # A synthesis is still in flight. Waiting for it could stall the
            # runtime stop for the provider's whole timeout, so the worker is
            # stopped directly: the blocked caller sees EOF and reports it.
            _terminate_process(self._process)
            return
        try:
            self._kill_locked()
        finally:
            self._lock.release()

    # ---------------------------------------------------------------- internal

    def _warm_locked(self, voice: str) -> None:
        if voice in self._warm_voices and self.worker_running:
            return
        if not self.python_available:
            raise TTSProviderError("kokoro_python_unavailable", self._python or "unset")
        if not self._worker_script.is_file():
            raise TTSProviderError("kokoro_worker_missing", self._worker_script.name)
        self._ensure_process_locked()
        response = self._exchange_locked(
            {"op": "warmup", "voice": voice, "lang_code": DEFAULT_LANG_CODE},
            float(self.config.kokoro_warmup_timeout_seconds),
            failure_reason="kokoro_unavailable",
        )
        self._model_load_ms = _as_float(response.get("model_load_ms"))
        self._warm_voices.add(voice)

    def _ensure_process_locked(self) -> None:
        if self.worker_running:
            return
        self._kill_locked()
        log_dir = Path(tempfile.gettempdir()) / "room_agent_kokoro"
        log_dir.mkdir(parents=True, exist_ok=True)
        self._log_handle = open(  # noqa: SIM115 - kept open for the worker's life
            log_dir / f"worker-{os.getpid()}.log", "ab", buffering=0
        )
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            self._process = self._popen(
                [self._python, "-u", str(self._worker_script)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._log_handle,
                cwd=str(self._worker_script.parent),
                text=True,
                encoding="utf-8",
                bufsize=1,
                creationflags=creationflags,
                env=self._environ,
            )
        except OSError as exc:
            raise TTSProviderError("kokoro_start_failed", type(exc).__name__) from None
        self._responses = queue.Queue()
        self._reader = threading.Thread(
            target=self._reader_loop,
            args=(self._process, self._responses),
            name="kokoro-worker-reader",
            daemon=True,
        )
        self._reader.start()

    def _reader_loop(self, process: Any, sink: "queue.Queue[Any]") -> None:
        stdout = getattr(process, "stdout", None)
        if stdout is None:
            sink.put(None)
            return
        try:
            for line in stdout:
                sink.put(line)
        except (OSError, ValueError):
            pass
        sink.put(None)

    def _exchange_locked(
        self,
        message: dict[str, Any],
        timeout: float,
        failure_reason: str,
    ) -> dict[str, Any]:
        self._request_counter += 1
        payload = dict(message, id=str(self._request_counter))
        self._noise: list[str] = []
        try:
            stdin = self._process.stdin
            stdin.write(json.dumps(payload, ensure_ascii=True) + "\n")
            stdin.flush()
        except (BrokenPipeError, OSError, ValueError, AttributeError) as exc:
            detail = self._log_tail() or type(exc).__name__
            self._kill_locked()
            raise TTSProviderError("kokoro_worker_crashed", detail) from None

        response = None
        deadline = time.monotonic() + max(1.0, timeout)
        # Model libraries occasionally print to stdout; only a JSON object with
        # our request id counts as the answer. Everything else is bounded noise.
        for _ in range(MAX_NOISE_LINES + 2):
            try:
                line = self._responses.get(timeout=max(0.1, deadline - time.monotonic()))
            except queue.Empty:
                self._kill_locked()
                raise TTSProviderError("kokoro_timeout", f"{timeout:.0f}s") from None
            if line is None:
                detail = self._log_tail() or "worker exited"
                self._kill_locked()
                raise TTSProviderError("kokoro_worker_crashed", detail) from None
            try:
                candidate = json.loads(line)
            except ValueError:
                self._note_noise(line)
                continue
            if not isinstance(candidate, dict):
                self._note_noise(line)
                continue
            if str(candidate.get("id") or "") != payload["id"]:
                self._note_noise(line)
                continue
            response = candidate
            break
        if response is None:
            self._kill_locked()
            raise TTSProviderError(
                "kokoro_protocol_error",
                " | ".join(self._noise[-3:])[:MAX_ERROR_DETAIL] or "no response for request",
            )
        if not response.get("ok"):
            detail = str(response.get("detail") or response.get("error") or "worker error")
            self._last_error = detail[:MAX_ERROR_DETAIL]
            self._kill_locked()
            raise TTSProviderError(failure_reason, self._last_error)
        self._last_error = ""
        return response

    def _note_noise(self, line: Any) -> None:
        text = str(line).strip().replace("\n", " ")
        if text:
            self._noise.append(text[:MAX_ERROR_DETAIL])

    def _log_tail(self) -> str:
        handle = self._log_handle
        if handle is None:
            return ""
        try:
            handle.flush()
            with open(handle.name, "rb") as reader:
                reader.seek(0, os.SEEK_END)
                size = reader.tell()
                reader.seek(max(0, size - MAX_LOG_TAIL))
                return reader.read().decode("utf-8", errors="replace").replace("\n", " ").strip()
        except OSError:
            return ""

    def _kill_locked(self) -> None:
        process = self._process
        self._process = None
        _terminate_process(process)
        if self._log_handle is not None:
            try:
                self._log_handle.close()
            except OSError:
                pass
            self._log_handle = None
        self._warm_voices.clear()
        self._responses = queue.Queue()


def _terminate_process(process: Any) -> None:
    """Close a worker's pipes and make sure the process itself is gone."""
    if process is None:
        return
    try:
        if getattr(process, "stdin", None) is not None:
            process.stdin.close()
    except (OSError, ValueError):
        pass
    try:
        if getattr(process, "stdout", None) is not None:
            process.stdout.close()
    except (OSError, ValueError):
        pass
    try:
        process.wait(timeout=3.0)
    except Exception:
        try:
            process.kill()
        except Exception:
            pass
        try:
            process.wait(timeout=3.0)
        except Exception:
            pass


def _as_float(value: object) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
