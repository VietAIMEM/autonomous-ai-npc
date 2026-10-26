"""Pure ffmpeg helpers: binary discovery, command building, and progress parsing.

Nothing in this module starts a process, so every rule is testable offline. The
provider owns the subprocess and its pipes; this module owns the vocabulary.

The command line built here ends with the real ingest target, which for RTMP
contains the stream key. Callers must therefore never print, log, or return the
argument list. Anything lifted out of the encoder's own output has to pass
through :func:`scrub` first.
"""

from __future__ import annotations

import os
import re
import shutil
import socket
from typing import Any, Mapping, Optional, Sequence

from .base import MAX_ERROR_DETAIL, REDACTED
from .config import StreamingConfig


#: Encoder output is read from stderr as ``key=value`` progress records.
PROGRESS_URL = "pipe:2"
MIN_FFMPEG_BYTES = 1_000_000

_PROGRESS_PATTERN = re.compile(r"([a-zA-Z_0-9]+)=([^\r\n]*)")
_BITRATE_PATTERN = re.compile(r"(-?\d+(?:\.\d+)?)")


def resolve_ffmpeg(configured: str = "") -> tuple[Optional[str], str]:
    """Find an ffmpeg binary. Returns ``(path, reason)``; path is None on failure.

    Order: explicit configuration, ``PATH``, then the optional ``imageio-ffmpeg``
    runtime helper. ffmpeg is never vendored into this repository, so a machine
    without one simply cannot run the real transport, and the provider says so.
    """
    candidate = (configured or "").strip()
    if candidate:
        if os.path.isfile(candidate):
            return candidate, ""
        return None, "FFMPEG_PATH_INVALID"
    found = shutil.which("ffmpeg")
    if found:
        return found, ""
    try:  # optional runtime helper, not a declared project dependency
        import imageio_ffmpeg  # type: ignore

        bundled = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        bundled = ""
    if bundled and os.path.isfile(bundled):
        return bundled, ""
    return None, "FFMPEG_NOT_FOUND"


def audio_listener_url(port: int) -> str:
    """Where the encoder listens for the live PCM track fed by this process."""
    return f"tcp://127.0.0.1:{int(port)}?listen=1"


def free_port() -> int:
    """Ask the OS for an unused loopback port for the encoder's audio listener."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def build_encoder_arguments(
    config: StreamingConfig,
    target: str,
    audio_url: str,
    ffmpeg_path: str,
) -> list[str]:
    """Build the encoder command line.

    Video enters on stdin as a stream of JPEG pictures, which is what Unity can
    produce cheaply; audio enters over a loopback socket, which keeps the two
    live tracks independent. Output is H.264 + AAC in the container the
    protocol expects.

    The audio listener has to be the first input. ffmpeg opens inputs one at a
    time, and reading ``pipe:0`` blocks until pictures arrive, so a
    video-first command would never get as far as binding the PCM socket: the
    encoder would wait on Unity while Unity waited on the encoder. Audio first
    breaks that cycle because the socket is listening before anything reads
    stdin.
    """
    if not target:
        raise ValueError("target is required")
    if not ffmpeg_path:
        raise ValueError("ffmpeg_path is required")

    arguments: list[str] = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-nostats",
        "-progress",
        PROGRESS_URL,
        # Audio first: raw PCM, delivered by the ingest thread as it is
        # produced. Listening before reading stdin avoids the startup cycle
        # described above.
        #
        # The probe limits are what make a live raw track usable. With its
        # defaults ffmpeg reads PCM until it has collected ``probesize`` bytes
        # (5 MB, about 52 seconds of 48 kHz mono) before it opens the video
        # input at all, so a real-time sender deadlocks against it. Raw PCM
        # carries no metadata, so a small probe is enough: the sample rate and
        # channel count are both stated outright.
        "-probesize",
        "4096",
        "-analyzeduration",
        "0",
        "-thread_queue_size",
        "512",
        "-f",
        "s16le",
        "-ar",
        str(int(config.sample_rate)),
        "-ac",
        str(int(config.channels)),
        "-i",
        audio_url,
        # Video: pictures encoded by Unity, paced by the capture clock. A live
        # pipe never reaches EOF, so this input needs the same probe limits;
        # without them ffmpeg waits for several seconds' worth of pictures
        # before it starts encoding.
        "-probesize",
        "4096",
        "-analyzeduration",
        "0",
        "-f",
        "image2pipe",
        "-framerate",
        _number(config.fps),
        "-vcodec",
        "mjpeg",
        "-thread_queue_size",
        "512",
        "-i",
        "pipe:0",
        "-map",
        "0:a:0",
        "-map",
        "1:v:0",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-tune",
        "zerolatency",
        "-profile:v",
        "main",
        "-pix_fmt",
        "yuv420p",
        "-r",
        _number(config.fps),
        "-g",
        str(int(config.effective_keyframe_interval)),
        "-b:v",
        f"{int(config.video_bitrate_kbps)}k",
        "-maxrate",
        f"{int(round(config.video_bitrate_kbps * 1.45))}k",
        "-bufsize",
        f"{int(config.video_bitrate_kbps * 2)}k",
        "-c:a",
        "aac",
        "-b:a",
        f"{int(config.audio_bitrate_kbps)}k",
        "-ar",
        str(int(config.sample_rate)),
        "-ac",
        str(int(config.channels)),
        # Keeps the audio track against the video clock when the two drift.
        "-af",
        "aresample=async=1:first_pts=0",
        "-max_muxing_queue_size",
        "512",
        "-f",
        config.container,
    ]
    if config.container == "flv":
        # A live stream has no known duration up front; do not let the muxer
        # stall waiting for one.
        arguments += ["-flvflags", "no_duration_filesize"]
    if config.width % 2 or config.height % 2:
        arguments += ["-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2"]
    arguments.append(target)
    return arguments


def parse_progress(text: str) -> dict[str, Any]:
    """Fold every ``key=value`` progress record in ``text`` into one snapshot.

    Later records win, which is what the caller wants: the newest encoder
    counters. Values are converted where a number is expected, otherwise kept
    as trimmed strings.
    """
    snapshot: dict[str, Any] = {}
    for key, raw in _PROGRESS_PATTERN.findall(text or ""):
        value = raw.strip()
        name = key.lower()
        if name in {"frame", "dup_frames", "drop_frames", "total_size"}:
            snapshot[name] = _as_int(value)
        elif name in {"fps", "bitrate", "speed", "out_time_us", "out_time_ms"}:
            snapshot[name] = _as_float_or_string(value)
        else:
            snapshot[name] = value
    return snapshot


def progress_bitrate_kbps(progress: Mapping[str, Any]) -> float:
    """``bitrate= 1523.4kbits/s`` -> ``1523.4``."""
    raw = progress.get("bitrate")
    if isinstance(raw, (int, float)):
        return float(raw)
    match = _BITRATE_PATTERN.search(str(raw or ""))
    return float(match.group(1)) if match else 0.0


def progress_output_seconds(progress: Mapping[str, Any]) -> float:
    """Muxed media time in seconds, taken from ``out_time_us`` when present."""
    raw = progress.get("out_time_us")
    try:
        return max(0.0, int(raw) / 1_000_000)
    except (TypeError, ValueError):
        return 0.0


def scrub(text: str, secrets: Sequence[str] = ()) -> str:
    """Remove every configured secret from encoder output before it is stored."""
    cleaned = text or ""
    for secret in secrets:
        value = str(secret or "")
        if len(value) >= 4:
            cleaned = cleaned.replace(value, REDACTED)
    # A leftover scheme://user:password@host form is never useful to keep.
    cleaned = re.sub(r"://[^/\s]*:[^/\s]*@", "://" + REDACTED + "@", cleaned)
    return cleaned[-MAX_ERROR_DETAIL:]


#: Failure vocabulary shared with the provider. Ordered most specific first.
FAILURE_SIGNATURES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "AUTH_FAILED",
        (
            "authentication failed",
            "invalid stream key",
            "token audience is invalid",
            "reason=invalid",
            "not allowed",
            "handshake failed",
            "403",
        ),
    ),
    (
        "ENCODER_FAILED",
        (
            "error while opening encoder",
            "encoder not found",
            "unknown encoder",
            "invalid pixel format",
            "unsupported codec",
            "codec not found",
            "conversion failed",
            "no such filter",
        ),
    ),
    (
        "TIMEOUT",
        (
            "timed out",
            "timeout",
            "operation timed out",
        ),
    ),
    (
        "PROVIDER_OUTAGE",
        (
            "internal server error",
            "service unavailable",
            "temporarily unavailable",
            "close stream",
            "502",
            "503",
            "504",
        ),
    ),
    (
        "CONNECTION_FAILED",
        (
            "cannot open connection",
            "connection refused",
            "connection reset",
            "connection aborted",
            "network is unreachable",
            "no route to host",
            "name or service not known",
            "getaddrinfo",
            "could not resolve host",
            "broken pipe",
            "input/output error",
            "end of file",
            "operation not permitted",
        ),
    ),
)


def classify_failure(text: str, returncode: Optional[int] = None) -> str:
    """Map encoder stderr onto a stable, credential-free reason code."""
    haystack = (text or "").lower()
    for reason, signatures in FAILURE_SIGNATURES:
        if any(needle in haystack for needle in signatures):
            return reason
    if returncode is not None and returncode != 0:
        return "TRANSPORT_FAILED"
    return "PROVIDER_FAILURE"


def _number(value: float) -> str:
    rendered = f"{float(value):.3f}".rstrip("0").rstrip(".")
    return rendered or "0"


def _as_int(value: str) -> int:
    match = _BITRATE_PATTERN.search(value or "")
    return int(float(match.group(1))) if match else 0


def _as_float_or_string(value: str) -> Any:
    match = _BITRATE_PATTERN.search(value or "")
    if not match:
        return value
    try:
        return float(match.group(1))
    except ValueError:  # pragma: no cover - regex already narrowed this
        return value
