"""Browser presentation relay: the existing SRT livestream, re-published as HLS.

PHASE 13/21 production pushes MPEG-TS over SRT (``STREAM_SERVER_URL``) from the
ffmpeg encoder that consumes Unity frames. Browsers cannot play SRT, so this
module owns the smallest browser-compatible presentation path: a bounded
ffmpeg process that *listens* on the configured SRT endpoint exactly like the
Phase 21 evidence receiver did, stream-copies the transport stream and writes a
rolling HLS playlist into a temporary directory the viewer web server serves.

It never replaces or duplicates the production stream pipeline: the encoder,
the ingest routes, and the streaming service are untouched. If the relay is
unavailable the viewer page degrades honestly instead of faking LIVE.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlsplit

from ..runtime.obs import log_event
from ..runtime.redaction import safe_error

log = logging.getLogger("room_agent.viewer_web.relay")

STATE_UNAVAILABLE = "UNAVAILABLE"
STATE_STOPPED = "STOPPED"
STATE_STARTING = "STARTING"
STATE_LIVE = "LIVE"
STATE_RETRYING = "RETRYING"

PLAYLIST_NAME = "live.m3u8"
SEGMENT_PREFIX = "seg_"

RelayProcess = object


def _directory_marker(directory: Path | str) -> str:
    return str(Path(directory).resolve()).casefold()


def _owned_relay_process(
    process: RelayProcess,
    *,
    directory_marker: str,
    exclude_pid: int,
) -> bool:
    """Identify only this application's viewer HLS FFmpeg process.

    The marker is the exact resolved HLS directory, not an executable name or
    UDP port. The parent check also protects the current backend's own live
    relay; an orphan's parent necessarily differs after the backend exits.
    """
    try:
        pid = int(process.pid)
        if pid <= 0 or pid == exclude_pid:
            return False
        name = (process.name() or "").casefold()
        if not name.startswith("ffmpeg"):
            return False
        command = process.cmdline() or []
        command_text = " ".join(str(part) for part in command).casefold()
        if directory_marker not in command_text:
            return False
        return int(process.ppid() or 0) != exclude_pid
    except Exception:
        return False


def _psutil_process_for_pid(pid: int) -> RelayProcess:
    import psutil

    return psutil.Process(pid)


def _terminate_owned_relay(process: RelayProcess) -> bool:
    import psutil

    try:
        process.terminate()
    except (psutil.Error, OSError):
        return False
    try:
        process.wait(timeout=2.0)
    except psutil.TimeoutExpired:
        try:
            process.kill()
            process.wait(timeout=2.0)
        except (psutil.Error, OSError):
            return False
    except (OSError, ValueError):
        return False
    return True


def reap_orphan_relays(
    directory: Path | str,
    *,
    exclude_pid: int,
    enumerate_processes: Optional[Callable[[], list[RelayProcess]]] = None,
    process_factory: Optional[Callable[[int], RelayProcess]] = None,
) -> list[int]:
    """Stop viewer HLS relays that survived a prior backend process.

    Returns the PIDs that were terminated. The operation is idempotent and
    never raises outward: a missing, inaccessible, or racing process is simply
    not owned at that instant.
    """
    try:
        marker = _directory_marker(directory)
    except OSError:
        return []
    try:
        candidates = list(
            enumerate_processes() if enumerate_processes is not None else _psutil_processes()
        )
    except Exception as exc:
        log_event(
            log,
            "viewer.relay.orphan_scan_failed",
            error=safe_error(exc, limit=120),
        )
        return []
    reaped: list[int] = []
    for candidate in candidates:
        try:
            pid = int(candidate.pid)
        except Exception:
            continue
        if not _owned_relay_process(
            candidate, directory_marker=marker, exclude_pid=exclude_pid
        ):
            continue
        try:
            # Reacquire the process so PID reuse after the snapshot is rejected
            # by command-line ownership rather than trusted from a stale PID.
            recheck = (
                process_factory(pid)
                if process_factory is not None
                else _psutil_process_for_pid(pid)
            )
        except Exception as exc:
            log_event(
                log,
                "viewer.relay.orphan_recheck_failed",
                pid=pid,
                error=safe_error(exc, limit=120),
            )
            continue
        if not _owned_relay_process(
            recheck, directory_marker=marker, exclude_pid=exclude_pid
        ):
            continue
        try:
            if _terminate_owned_relay(recheck):
                reaped.append(pid)
        except Exception as exc:
            log_event(
                log,
                "viewer.relay.orphan_terminate_failed",
                pid=pid,
                error=safe_error(exc, limit=120),
            )
    if reaped:
        log_event(
            log,
            "viewer.relay.orphans_reaped",
            count=len(reaped),
            pids=reaped,
        )
    return reaped


def _psutil_processes() -> list[RelayProcess]:
    import psutil

    return list(psutil.process_iter())


def parse_srt_listen_endpoint(url: str) -> Optional[tuple[str, int]]:
    """Return ``(host, port)`` for an ``srt://`` endpoint, else ``None``.

    The relay only engages when the configured transport really is SRT; an
    RTMP or disabled stream must not silently produce a fake local listener.
    """
    try:
        parts = urlsplit((url or "").strip())
    except ValueError:
        return None
    if parts.scheme.lower() != "srt" or not parts.hostname:
        return None
    try:
        port = parts.port or 9000
    except ValueError:
        return None
    if not 1 <= port <= 65535:
        return None
    return parts.hostname, port


class HlsRelay:
    """Bounded ffmpeg SRT-listener -> HLS re-muxer with restart backoff."""

    def __init__(
        self,
        *,
        ffmpeg_path: str = "",
        listen_url: str = "",
        directory: Path | str | None = None,
        enabled: bool = True,
        segment_seconds: float = 1.0,
        playlist_window: int = 6,
        start_timeout_seconds: float = 12.0,
        stale_after_seconds: float = 6.0,
        backoff_base_seconds: float = 2.0,
        backoff_max_seconds: float = 15.0,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self._ffmpeg = self._resolve_ffmpeg(ffmpeg_path)
        self._listen = parse_srt_listen_endpoint(listen_url) if enabled else None
        self._directory = Path(directory) if directory else None
        self._segment_seconds = max(0.2, float(segment_seconds))
        self._playlist_window = max(2, int(playlist_window))
        self._start_timeout = max(1.0, float(start_timeout_seconds))
        self._stale_after = max(2.0, float(stale_after_seconds))
        self._backoff_base = max(0.2, float(backoff_base_seconds))
        self._backoff_max = max(self._backoff_base, float(backoff_max_seconds))
        self._clock = clock
        self._wall_clock = wall_clock

        self._lock = threading.Lock()
        self._process: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._started_at: Optional[float] = None
        self._restart_count = 0
        self._reason = self._configuration_reason(enabled)

    # ------------------------------------------------------------- lifecycle

    @property
    def configured(self) -> bool:
        return self._reason is None

    @property
    def directory(self) -> Optional[Path]:
        return self._directory

    @property
    def playlist_path(self) -> Optional[Path]:
        if self._directory is None:
            return None
        return self._directory / PLAYLIST_NAME

    def start(self) -> bool:
        """Start the relay thread. Returns False when not configured."""
        if self._directory is not None:
            reap_orphan_relays(self._directory, exclude_pid=os.getpid())
        if not self.configured:
            log_event(log, "viewer.relay.disabled", reason=self._reason)
            return False
        if self._directory is None:
            self._reason = "relay_directory_missing"
            return False
        try:
            self._directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self._reason = "relay_directory_unavailable"
            log_event(log, "viewer.relay.directory_failed", error=safe_error(exc, limit=160))
            return False
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return True
            self._stop.clear()
            thread = threading.Thread(
                target=self._serve_loop, name="viewer-hls-relay", daemon=True
            )
            self._thread = thread
        thread.start()
        log_event(
            log,
            "viewer.relay.started",
            host=self._listen[0],
            port=self._listen[1],
        )
        return True

    def stop(self, timeout: float = 8.0) -> None:
        self._stop.set()
        with self._lock:
            process = self._process
            thread = self._thread
        if process is not None and process.poll() is None:
            try:
                process.terminate()
            except OSError:
                pass
            try:
                process.wait(timeout=max(1.0, timeout / 2.0))
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                except OSError:
                    pass
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(1.0, timeout / 2.0))
        self._cleanup_media()

    # --------------------------------------------------------------- state

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            process = self._process
            started_at = self._started_at
            restarts = self._restart_count
        alive = process is not None and process.poll() is None
        age = self._clock() - started_at if (alive and started_at is not None) else None
        manifest_age = self._manifest_age_seconds()
        segments = self._segment_count()
        state, detail = self._classify(
            alive=alive,
            age_seconds=age,
            manifest_age_seconds=manifest_age,
            segment_count=segments,
        )
        return {
            "state": state,
            "detail": detail,
            "configured": self.configured,
            "reason": self._reason,
            "playlist_url": "/hls/live.m3u8" if self.configured else None,
            "playlist_ready": state == STATE_LIVE,
            "segments": segments,
            "restart_count": restarts,
        }

    def _classify(
        self,
        *,
        alive: bool,
        age_seconds: Optional[float],
        manifest_age_seconds: Optional[float],
        segment_count: int,
    ) -> tuple[str, str]:
        """Pure state classification so tests can cover every presentation state."""
        if self._reason is not None:
            return STATE_UNAVAILABLE, self._reason
        if not alive:
            return STATE_STOPPED, "relay_not_running"
        fresh = (
            manifest_age_seconds is not None
            and manifest_age_seconds <= self._stale_after
            and segment_count > 0
        )
        if fresh:
            return STATE_LIVE, "receiving"
        if age_seconds is not None and age_seconds < self._start_timeout:
            return STATE_STARTING, "waiting_for_stream"
        return STATE_RETRYING, "no_media"

    # -------------------------------------------------------------- internals

    def _configuration_reason(self, enabled: bool) -> Optional[str]:
        if not enabled:
            return "relay_disabled"
        if self._ffmpeg is None:
            return "ffmpeg_missing"
        if self._listen is None:
            return "stream_endpoint_not_srt"
        return None

    @staticmethod
    def _resolve_ffmpeg(path: str) -> Optional[str]:
        candidate = (path or "").strip()
        if candidate:
            # An explicitly configured encoder must resolve to something real.
            # Falling back to a different ``ffmpeg`` on PATH here would hide a
            # misconfiguration and report a relay as configured that the
            # operator never asked for; the streaming encoder already keeps the
            # same contract (see ``backend/streaming/encoder.py``: an invalid
            # explicit path is ``FFMPEG_PATH_INVALID``, never a PATH retry).
            return shutil.which(candidate) or (
                candidate if Path(candidate).is_file() else None
            )
        return shutil.which("ffmpeg")

    def _spawn(self) -> subprocess.Popen:
        assert self._listen is not None and self._directory is not None and self._ffmpeg
        host, port = self._listen
        source = f"srt://{host}:{port}?mode=listener"
        command = [
            self._ffmpeg,
            "-hide_banner",
            "-nostdin",
            "-loglevel",
            "error",
            "-fflags",
            "+genpts",
            "-i",
            source,
            "-c",
            "copy",
            "-f",
            "hls",
            "-hls_time",
            f"{self._segment_seconds:.2f}",
            "-hls_list_size",
            str(self._playlist_window),
            "-hls_flags",
            "delete_segments+omit_endlist",
            "-hls_segment_type",
            "mpegts",
            "-hls_segment_filename",
            str(self._directory / f"{SEGMENT_PREFIX}%05d.ts"),
            str(self._directory / PLAYLIST_NAME),
        ]
        creationflags = 0
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        return subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )

    def _serve_loop(self) -> None:
        failures = 0
        while not self._stop.is_set():
            try:
                self._cleanup_media()
                process = self._spawn()
            except Exception as exc:
                log_event(
                    log,
                    "viewer.relay.spawn_failed",
                    error=safe_error(exc, limit=160),
                )
                failures += 1
                self._stop.wait(self._backoff(failures))
                continue
            with self._lock:
                self._process = process
                self._started_at = self._clock()
                if failures:
                    self._restart_count += 1
            process.wait()
            with self._lock:
                self._process = None
            if self._stop.is_set():
                break
            failures += 1
            delay = self._backoff(failures)
            log_event(
                log,
                "viewer.relay.retry",
                failures=failures,
                delay_seconds=delay,
            )
            self._stop.wait(delay)

    def _backoff(self, failures: int) -> float:
        return min(self._backoff_max, self._backoff_base * (2 ** max(0, failures - 1)))

    def _manifest_age_seconds(self) -> Optional[float]:
        path = self.playlist_path
        if path is None:
            return None
        try:
            return max(0.0, self._wall_clock() - path.stat().st_mtime)
        except OSError:
            return None

    def _segment_count(self) -> int:
        if self._directory is None:
            return 0
        try:
            return sum(
                1
                for entry in self._directory.iterdir()
                if entry.name.startswith(SEGMENT_PREFIX) and entry.suffix == ".ts"
            )
        except OSError:
            return 0

    def _cleanup_media(self) -> None:
        if self._directory is None:
            return
        try:
            for entry in self._directory.iterdir():
                if entry.name == PLAYLIST_NAME or (
                    entry.name.startswith(SEGMENT_PREFIX) and entry.suffix == ".ts"
                ):
                    try:
                        entry.unlink()
                    except OSError:
                        pass
        except OSError:
            pass
