"""Real streaming transport: one ffmpeg encoder process per connection.

Unity renders, PHASE 11 TTS speaks, and this provider turns those two live
signals into a platform stream. Video enters the encoder on stdin as the JPEG
pictures Unity can produce cheaply; audio enters over a loopback socket so the
two tracks stay independent instead of being interleaved by hand.

Credential rule for this whole module: the ingest target for RTMP *contains* the
stream key, so the argument list is never logged, printed, returned, or stored on
a public attribute, and anything lifted out of the encoder's own output is passed
through :func:`scrub` before it is kept.

Failure rule: only :meth:`connect` raises. Everything else reports through
:attr:`state` and :meth:`health`, so a dead platform degrades the stream and the
character keeps running.
"""

from __future__ import annotations

import socket
import subprocess
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Callable, Deque, Optional

from .base import (
    MAX_ERROR_DETAIL,
    AudioChunk,
    StreamHealth,
    StreamingProvider,
    StreamingProviderError,
    StreamState,
    VideoFrame,
)
from .config import PROVIDER_SRT, StreamingConfig, read_stream_key
from .encoder import (
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


#: How long a stalled encoder is given to finish flushing before it is killed.
GRACEFUL_EXIT_SECONDS = 1.2
PROCESS_EXIT_SECONDS = 1.5
READER_JOIN_SECONDS = 1.0
LOG_TAIL_LINES = 48
STALL_POLL_SECONDS = 0.25
#: A live encoder is already listening; these are retries, not patience.
AUDIO_DIAL_SECONDS = 0.2
AUDIO_DIAL_RETRY_SECONDS = 0.05
#: Silence pushed at the head of the PCM track so the encoder starts reading video.
AUDIO_PRIME_SECONDS = 0.25


class FFmpegStreamingProvider(StreamingProvider):
    """Push a live RTMP (or SRT) stream through an external ffmpeg process.

    ``spawner`` replaces the process launcher and exists for tests: a fake
    process object with the handful of ``Popen`` members used here is enough to
    exercise connect, write, failure, reconnect, and teardown offline.
    """

    platform = "twitch"
    transport_enabled = True

    def __init__(
        self,
        config: StreamingConfig | None = None,
        spawner: Optional[Callable[[list[str]], Any]] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config or StreamingConfig(provider=PROVIDER_SRT)
        self.name = self.config.provider
        self.platform = self.config.platform
        self._spawn = spawner or _default_spawner
        self._clock = clock
        self._key_path = self.config.stream_key_path

        self._lock = threading.RLock()
        self._io_lock = threading.Lock()
        self._state = StreamState.IDLE
        self._process: Optional[Any] = None
        self._reader: Optional[threading.Thread] = None
        self._watchdog: Optional[threading.Thread] = None
        self._stop_requested = threading.Event()
        self._closing = False

        # Progress records folded from the encoder's own machine-readable output.
        self._progress: dict[str, Any] = {}
        self._log_tail: Deque[str] = deque(maxlen=LOG_TAIL_LINES)
        self._pending = ""

        self._connect_started: Optional[float] = None
        self._live_since: Optional[float] = None
        self._live_wall: Optional[datetime] = None
        self._start_wall: Optional[datetime] = None
        self._started_at: Optional[float] = None
        self._ever_live = False
        self._connecting_for_reconnect = False
        self._reconnect_attempt = 0
        self._reconnect_count = 0
        self._connect_attempts = 0
        self._next_attempt_at = 0.0
        self._last_error: Optional[str] = None
        self._secrets: list[str] = []
        self._write_since: Optional[float] = None

        self._frames_written = 0
        self._frame_bytes = 0
        self._audio_written = 0
        self._audio_bytes = 0
        self._audio_seconds = 0.0
        self._write_failures = 0
        # ``_write_failures`` is a lifetime counter for diagnostics; these two
        # timestamps decide whether the *current* live window has media moving.
        self._last_write_failure_at: Optional[float] = None
        self._last_video_write_at: Optional[float] = None
        self._audio_port = 0
        self._audio_socket: Optional[socket.socket] = None
        self._audio_primed = False

    # -- lifecycle ---------------------------------------------------------

    def connect(self) -> None:
        """Start the encoder. Raises if the transport cannot be established.

        Returning means the process is up and its input sockets are open. The
        state stays ``CONNECTING`` until :meth:`poll` sees the encoder actually
        produce pictures, because a platform only counts a stream as delivered
        once media is moving through it.
        """
        with self._lock:
            # ``CONNECTING`` and ``LIVE`` already own a process, so starting again
            # would spawn a second encoder. ``RECONNECTING`` deliberately has no
            # process: :meth:`poll` calls back into ``connect`` to replace it, so
            # refusing here would strand the stream in RECONNECTING forever.
            resumable = (
                self._state is StreamState.RECONNECTING and self._process is None
            )
            if not resumable and self._state not in {
                StreamState.IDLE,
                StreamState.STOPPED,
                StreamState.ERROR,
            }:
                return
            resetting = not self._connecting_for_reconnect
            self._connecting_for_reconnect = False
            if resetting:
                self._reconnect_attempt = 0
            self._connect_attempts += 1
            self._closing = False
            self._stop_requested.clear()
            self._state = StreamState.CONNECTING
            self._progress = {}
            self._log_tail.clear()
            self._pending = ""
            self._connect_started = self._clock()
            # A replacement encoder has a fresh stdin and a fresh PCM listener, so
            # it needs the priming silence all over again.
            self._audio_primed = False
            self._last_write_failure_at = None
            self._last_video_write_at = None
            if self._started_at is None:
                self._started_at = self._connect_started
                self._start_wall = datetime.now(timezone.utc)

        ffmpeg_path, reason = resolve_ffmpeg(self.config.ffmpeg_path)
        if not ffmpeg_path:
            self._abort_connect(reason, "no encoder available")
            raise StreamingProviderError(reason, "ffmpeg could not be located")

        key = read_stream_key(self._key_path) if self.config.provider != PROVIDER_SRT else ""
        target = self._build_target(key)
        self._secrets = [value for value in (key, target) if value]

        self._audio_port = free_port()
        arguments = build_encoder_arguments(
            self.config, target, audio_listener_url(self._audio_port), ffmpeg_path
        )
        try:
            process = self._spawn(arguments)
        except OSError as exc:
            self._abort_connect("ENCODER_FAILED", type(exc).__name__)
            raise StreamingProviderError(
                "ENCODER_FAILED", "the encoder process could not be started"
            ) from exc

        with self._lock:
            self._process = process
            self._reader = threading.Thread(
                target=self._read_encoder_output,
                args=(process,),
                name="stream-encoder-output",
                daemon=True,
            )
            self._reader.start()
            self._watchdog = threading.Thread(
                target=self._watch_stalled_writes,
                args=(process,),
                name="stream-write-watchdog",
                daemon=True,
            )
            self._watchdog.start()

        # The encoder opens its audio input by waiting for this connection, so
        # dialing it here is what lets the output (the platform) be opened at all.
        self._open_audio_socket(min(1.0, self.config.connect_timeout_seconds))
        # And dialing is not enough: the track has to carry bytes before the
        # encoder will look at video at all.
        self._prime_audio_input()

    def _build_target(self, key: str) -> str:
        base = (self.config.server_url or "").strip()
        if "{key}" in base:
            return base.replace("{key}", key)
        if self.config.provider == PROVIDER_SRT or not key:
            return base
        return base.rstrip("/") + "/" + key

    def _abort_connect(self, reason: str, detail: str) -> None:
        with self._lock:
            self._state = StreamState.ERROR
            self._last_error = scrub(f"{reason}:{detail}", self._secrets)

    def disconnect(self) -> None:
        """Close the transport and release every resource. Never raises."""
        with self._lock:
            self._closing = True
            self._stop_requested.set()
            self._state = StreamState.STOPPING
            process = self._process
            self._live_since = None
            self._live_wall = None
        # A killed encoder is also the only reliable way to unblock a writer that
        # is parked on a full stdin pipe, so arm the watchdog before waiting.
        # The audio socket has to go first. It is a live input, so while it is
        # open the encoder never sees end of stream and will not finalise its
        # output; closing stdin alone leaves ffmpeg parked on the PCM track,
        # and the graceful window then expires into a kill that discards the
        # buffered tail. With both inputs ended the encoder shuts itself down
        # cleanly, which is also what a real platform expects to see.
        self._close_audio_socket()
        self._kill_quietly(process, GRACEFUL_EXIT_SECONDS)
        reader, watchdog = self._reader, self._watchdog
        if reader is not None and reader is not threading.current_thread():
            reader.join(timeout=READER_JOIN_SECONDS)
        if watchdog is not None and watchdog is not threading.current_thread():
            watchdog.join(timeout=READER_JOIN_SECONDS)
        with self._lock:
            self._process = None
            self._reader = None
            self._watchdog = None
            self._state = StreamState.STOPPED
            self._reconnect_attempt = 0
            self._connecting_for_reconnect = False

    # -- media -------------------------------------------------------------

    def write_video(self, frame: VideoFrame) -> bool:
        with self._lock:
            process = self._process
            if process is None or self._closing or self._state not in {
                StreamState.CONNECTING,
                StreamState.LIVE,
            }:
                return False
        if not frame.payload:
            return False
        # An encoder writes no pictures until it has seen its first audio block,
        # so the listener is dialed here too in case connect() could not.
        if self._open_audio_socket(AUDIO_DIAL_RETRY_SECONDS):
            self._prime_audio_input()
        else:
            # Writing a picture now would park this thread on a stdin pipe the
            # encoder has no intention of reading, so refuse and let the caller
            # retry once audio is flowing.
            self._note_write_failure()
            return False
        payload = bytes(frame.payload)
        try:
            with self._io_lock:
                self._write_since = self._clock()
                stdin = getattr(process, "stdin", None)
                if stdin is None:
                    raise OSError("encoder input is closed")
                stdin.write(payload)
                stdin.flush()
                self._write_since = None
        except (OSError, ValueError):
            self._note_write_failure()
            return False
        with self._lock:
            self._frames_written += 1
            self._frame_bytes += len(payload)
            self._last_video_write_at = self._clock()
        return True

    def write_audio(self, chunk: AudioChunk) -> bool:
        with self._lock:
            if self._closing or self._state not in {StreamState.CONNECTING, StreamState.LIVE}:
                return False
        if not chunk.payload:
            return False
        if not self._open_audio_socket(AUDIO_DIAL_SECONDS):
            self._note_write_failure()
            return False
        payload = bytes(chunk.payload)
        try:
            with self._io_lock:
                self._write_since = self._clock()
                sock = self._audio_socket
                if sock is None:
                    raise OSError("encoder audio input is closed")
                sock.sendall(payload)
                self._write_since = None
        except OSError:
            self._close_audio_socket()
            self._note_write_failure()
            return False
        with self._lock:
            self._audio_written += 1
            self._audio_bytes += len(payload)
            self._audio_seconds += chunk.duration_seconds
            # Real content arrived, so the handshake silence is no longer needed.
            self._audio_primed = True
        return True

    def _note_write_failure(self) -> None:
        with self._lock:
            self._write_since = None
            self._write_failures += 1
            self._last_write_failure_at = self._clock()

    def _prime_audio_input(self) -> None:
        """Put a little silence at the head of the PCM track, once per encoder.

        ffmpeg opens its inputs in order and analyses the audio track before it
        pulls a single picture from stdin. A socket that is merely *connected*
        leaves it parked there, so the video writer fills the OS pipe, stops
        making progress, and the write watchdog kills an encoder that was
        perfectly healthy. That looks exactly like a transport failure.

        A short run of silence is enough to finish the analysis, and the audio
        filter chain starts the track at zero, so it only delays the picture by
        less than a frame interval's worth of buffer. It is deliberately not
        counted as ingested media: it is a transport handshake, not content.
        """
        with self._lock:
            if self._audio_primed:
                return
            sock = self._audio_socket
            rate = int(self.config.sample_rate) or 48000
            channels = int(self.config.channels) or 1
        if sock is None:
            return
        payload = b"\x00" * (int(rate * AUDIO_PRIME_SECONDS) * channels * 2)
        try:
            with self._io_lock:
                sock.sendall(payload)
        except OSError:
            return
        with self._lock:
            self._audio_primed = True

    # -- audio socket ------------------------------------------------------

    def _open_audio_socket(self, budget: float = AUDIO_DIAL_SECONDS) -> bool:
        """Connect to the encoder's PCM listener, retrying briefly while it binds."""
        with self._lock:
            if self._audio_socket is not None:
                return True
            port = self._audio_port
            process = self._process
        if not port or process is None:
            return False
        deadline = self._clock() + max(0.0, float(budget))
        # The attempt cap mirrors the deadline so a caller-supplied clock that does
        # not advance cannot turn this retry loop into an unbounded spin.
        for _ in range(int(max(0.0, float(budget)) / AUDIO_DIAL_RETRY_SECONDS) + 2):
            try:
                candidate = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                candidate.settimeout(max(0.05, self.config.write_timeout_seconds))
                candidate.connect(("127.0.0.1", port))
            except OSError:
                candidate.close()
                if self._clock() >= deadline or self._stop_requested.is_set():
                    return False
                time.sleep(0.02)
                continue
            with self._lock:
                if self._audio_socket is None:
                    self._audio_socket = candidate
                    return True
            candidate.close()
            return True

    def _close_audio_socket(self) -> None:
        with self._io_lock:
            with self._lock:
                socket_handle = self._audio_socket
                self._audio_socket = None
            if socket_handle is not None:
                try:
                    socket_handle.shutdown(2)
                except OSError:
                    pass
                try:
                    socket_handle.close()
                except OSError:
                    pass

    # -- state -------------------------------------------------------------

    @property
    def state(self) -> StreamState:
        with self._lock:
            return self._state

    def poll(self) -> StreamState:
        with self._lock:
            state = self._state
            process = self._process
            now = self._clock()

        if state in {StreamState.STOPPING, StreamState.STOPPED, StreamState.IDLE, StreamState.ERROR}:
            return state

        exit_code = process.poll() if process is not None else None
        with self._lock:
            frames_encoded = int(self._progress.get("frame") or 0)
            live_for = (now - self._live_since) if self._live_since else 0.0

        if state is StreamState.CONNECTING:
            if process is None:
                self._handle_failure("TRANSPORT_LOST", "encoder is not running")
                return self.state
            if exit_code is not None:
                self._handle_failure(
                    classify_failure(self._scrubbed_log(), exit_code),
                    f"encoder exited with code {exit_code}",
                )
                return self.state
            if frames_encoded > 0:
                self._promote_live()
                return self.state
            if now - (self._connect_started or now) >= self.config.live_timeout_seconds:
                self._handle_failure("TIMEOUT", "no encoded frames before the deadline")
                return self.state
            return self.state

        if state is StreamState.LIVE:
            if exit_code is not None:
                self._handle_failure(
                    classify_failure(self._scrubbed_log(), exit_code),
                    f"encoder exited with code {exit_code}",
                )
                return self.state
            if self._media_stalled(now):
                self._handle_failure("TRANSPORT_LOST", "the encoder stopped accepting media")
                return self.state
            return self.state

        if state is StreamState.RECONNECTING:
            if now < self._next_attempt_at:
                return self.state
            try:
                self.connect()
            except StreamingProviderError as exc:
                self._handle_failure(exc.reason, exc.detail)
            return self.state

        return self.state

    def _promote_live(self) -> None:
        now = self._clock()
        with self._lock:
            restored = self._ever_live and self._reconnect_attempt > 0
            self._state = StreamState.LIVE
            self._live_since = now
            self._live_wall = datetime.now(timezone.utc)
            self._ever_live = True
            self._last_error = None
            if restored:
                self._reconnect_count += 1
            self._reconnect_attempt = 0

    def _media_stalled(self, now: float) -> bool:
        """True only when *this* live window stopped accepting pictures.

        A lifetime failure counter cannot answer that question: one transient
        write timeout during a reconnect used to tear the stream down again
        ``live_timeout_seconds`` after it came back, which reads on the wire as
        a stream that churns forever every retry. A stalled encoder is one that
        refused a write after its last accepted picture and never took another.
        """
        with self._lock:
            live_since = self._live_since
            failed_at = self._last_write_failure_at
            last_ok = self._last_video_write_at
        if live_since is None:
            return False
        if last_ok is None or last_ok < live_since:
            last_ok = live_since
        if failed_at is None or failed_at < last_ok:
            return False
        return (now - last_ok) >= self.config.live_timeout_seconds

    def _handle_failure(self, reason: str, detail: str = "") -> None:
        """Move to RECONNECTING (bounded) after an outage, or ERROR when spent."""
        policy = self.config.reconnect
        now = self._clock()
        with self._lock:
            self._last_error = scrub(f"{reason}:{detail}", self._secrets) or reason
            self._live_since = None
            self._live_wall = None
            self._last_write_failure_at = None
            self._last_video_write_at = None
            # A stream that was never live should not respawn encoders behind its
            # caller's back; the next explicit start clears ERROR instead.
            eligible = self._ever_live and policy.allows_attempt(self._reconnect_attempt + 1)
            if eligible:
                self._reconnect_attempt += 1
                delay = policy.delay_for_attempt(self._reconnect_attempt)
                self._state = StreamState.RECONNECTING
                self._next_attempt_at = now + delay
                self._connecting_for_reconnect = True
            else:
                self._state = StreamState.ERROR
                self._connecting_for_reconnect = False
        self._kill_quietly(self._process, 0.0)
        self._close_audio_socket()
        with self._lock:
            self._process = None
            self._reader = None
            self._watchdog = None

    def _kill_quietly(self, process: Optional[Any], grace_seconds: float) -> None:
        if process is None:
            return
        try:
            stdin = getattr(process, "stdin", None)
            if stdin is not None and self._io_lock.acquire(timeout=0.25):
                try:
                    stdin.close()
                except OSError:
                    pass
                finally:
                    self._io_lock.release()
        except Exception:
            pass
        try:
            if process.poll() is None:
                if grace_seconds > 0:
                    try:
                        process.wait(timeout=grace_seconds)
                    except Exception:
                        pass
                if process.poll() is None:
                    process.terminate()
                try:
                    process.wait(timeout=PROCESS_EXIT_SECONDS)
                except Exception:
                    process.kill()
                    try:
                        process.wait(timeout=0.5)
                    except Exception:
                        pass
        except Exception:
            pass

    def _watch_stalled_writes(self, process: Any) -> None:
        """Kill an encoder that stops draining its input, so no writer is stuck."""
        limit = max(0.5, self.config.write_timeout_seconds)
        while not self._stop_requested.wait(STALL_POLL_SECONDS):
            with self._lock:
                started = self._write_since
                closing = self._closing
            if started is None or closing:
                continue
            if self._clock() - started > limit:
                self._kill_quietly(process, 0.0)
                return

    def _read_encoder_output(self, process: Any) -> None:
        stream = getattr(process, "stderr", None)
        if stream is None:
            return
        read = getattr(stream, "read1", None) or stream.read
        while not self._stop_requested.is_set():
            try:
                block = read(4096)
            except (OSError, ValueError):
                break
            if not block:
                self._absorb_text("")
                break
            if isinstance(block, str):
                text = block
            else:
                text = block.decode("utf-8", errors="replace")
            self._absorb_text(text)
        # This thread is the only reader, so it is also the only place the pipe
        # can be released. Leaving it open holds the child's file descriptor.
        try:
            stream.close()
        except OSError:
            pass

    def _absorb_text(self, text: str) -> None:
        """Split encoder output into progress records and human log lines."""
        buffer = self._pending + text
        lines = buffer.split("\n")
        self._pending = lines.pop() if lines else ""
        if not lines:
            return
        additions = parse_progress("\n".join(lines))
        with self._lock:
            if additions:
                self._progress.update(additions)
            for line in lines:
                stripped = line.strip("\r").strip()
                if stripped and "=" not in stripped:
                    self._log_tail.append(stripped)

    def _scrubbed_log(self) -> str:
        with self._lock:
            text = "\n".join(self._log_tail)
            ended = str(self._progress.get("progress") or "")
        if ended == "end":
            text = text + "\nend of file"
        return scrub(text, self._secrets)[:MAX_ERROR_DETAIL]

    def uptime_seconds(self) -> float:
        with self._lock:
            if self._started_at is None or self._state not in {
                StreamState.CONNECTING,
                StreamState.LIVE,
                StreamState.RECONNECTING,
            }:
                return 0.0
            return round(self._clock() - self._started_at, 2)

    @property
    def write_failures(self) -> int:
        """How many media handoffs the encoder refused. Diagnostics only."""
        with self._lock:
            return self._write_failures

    def health(self) -> StreamHealth:
        with self._lock:
            progress = dict(self._progress)
            frames_encoded = int(progress.get("frame") or 0)
            encoder_fps = float(progress.get("fps") or 0.0)
            bitrate = progress_bitrate_kbps(progress)
            muxed_seconds = progress_output_seconds(progress)
            health = StreamHealth(
                state=self._state,
                provider=self.name,
                platform=self.platform,
                resolution=self.config.resolution,
                target_fps=self.config.fps,
                encoder_fps=round(encoder_fps, 2),
                # ffmpeg reports ``bitrate=`` in kbit/s already, so no unit conversion.
                bitrate_kbps=round(bitrate, 1),
                uptime_seconds=self.uptime_seconds(),
                frames_encoded=frames_encoded,
                dropped_frames=int(progress.get("drop_frames") or 0),
                audio_seconds_ingested=round(self._audio_seconds, 2),
                video_state="OK" if self._frames_written else "NO_INPUT",
                audio_state="OK" if self._audio_seconds else "NO_INPUT",
                reconnect_count=self._reconnect_count,
                connect_attempts=self._connect_attempts,
                last_error=self._last_error,
                live_since=self._live_wall,
                started_at=self._start_wall,
                transport_enabled=True,
                attributes={
                    "muxed_seconds": round(muxed_seconds, 2),
                    "bytes_output": int(progress.get("total_size") or 0),
                    "encoder_pid": getattr(self._process, "pid", 0) or 0,
                    "write_failures": self._write_failures,
                    # A scrubbed, capped encoder log tail: enough to diagnose a
                    # failure without ever publishing the stream key.
                    "error_log": (
                        self._scrubbed_log()
                        if self._state in {StreamState.ERROR, StreamState.RECONNECTING}
                        else ""
                    ),
                },
            )
        return health

    def diagnostics(self) -> dict[str, Any]:
        payload = dict(super().diagnostics())
        payload.update(
            {
                "resolution": self.config.resolution,
                "fps": self.config.fps,
                "container": self.config.container,
                "reconnect": self.config.reconnect.to_dict(),
            }
        )
        return payload

    def __repr__(self) -> str:  # pragma: no cover - logging guard
        return (
            "FFmpegStreamingProvider(provider={!r}, platform={!r}, state={!r}, "
            "resolution={!r})"
        ).format(self.name, self.platform, self._state.value, self.config.resolution)


def _default_spawner(arguments: list[str]) -> Any:
    """Launch the encoder. The argument list carries the key; never print it."""
    return subprocess.Popen(
        arguments,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        shell=False,
    )
