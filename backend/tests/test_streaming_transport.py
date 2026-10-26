"""PHASE 13 integration test against a real encoder. This file is opt-in.

It runs the genuine ``FFmpegStreamingProvider`` (real subprocess, real H.264 +
AAC encoding) but sends the output to a local ``.ts`` file instead of a
platform, so it proves the transport without needing a streaming account. Every
artifact is written to a temp directory and removed, and the whole module skips
cleanly when no ffmpeg binary exists.

Run directly while developing:

    python -m unittest backend.tests.test_streaming_transport -v
"""

from __future__ import annotations

import math
import struct
import subprocess
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from backend.streaming.base import AudioChunk, StreamState, VideoFrame
from backend.streaming.config import PROVIDER_SRT, StreamingConfig
from backend.streaming.ffmpeg_provider import FFmpegStreamingProvider
from backend.streaming.encoder import resolve_ffmpeg


WIDTH = 64
HEIGHT = 64
FPS = 10.0
SAMPLE_RATE = 48000
FRAME_COUNT = 12


def _run(arguments: list[str], timeout: float = 30.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        arguments,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
        timeout=timeout,
    )


def _sine_seconds(seconds: float, frequency: float = 440.0) -> bytes:
    """A real tone, so the audio track is verifiable rather than empty."""
    count = int(SAMPLE_RATE * seconds)
    amplitude = 12000
    return b"".join(
        struct.pack("<h", int(amplitude * math.sin(2 * math.pi * frequency * n / SAMPLE_RATE)))
        for n in range(count)
    )


@unittest.skipUnless(resolve_ffmpeg()[0], "no ffmpeg binary available")
class RealEncoderTests(unittest.TestCase):
    """One real encode, start to finish, into a file rather than a platform."""

    def setUp(self) -> None:
        self.workdir = TemporaryDirectory()
        self.addCleanup(self.workdir.cleanup)
        self.root = Path(self.workdir.name)
        self.ffmpeg, reason = resolve_ffmpeg()
        self.assertIsNotNone(self.ffmpeg, reason)
        self.output = self.root / "stream.ts"

    def make_frames(self) -> list[bytes]:
        """Ask ffmpeg for a short test pattern as individual JPEG pictures."""
        pattern = self.root / "frame_%03d.jpg"
        result = _run(
            [
                self.ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                f"testsrc=size={WIDTH}x{HEIGHT}:rate={int(FPS)}",
                "-frames:v",
                str(FRAME_COUNT),
                "-q:v",
                "6",
                str(pattern),
            ]
        )
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", "replace"))
        frames = sorted(self.root.glob("frame_*.jpg"))
        self.assertEqual(len(frames), FRAME_COUNT)
        payloads = [path.read_bytes() for path in frames]
        for payload in payloads:
            # SOI marker: these really are JPEG pictures, which is the contract
            # between Unity's capture pump and the encoder's stdin.
            self.assertEqual(payload[:2], b"\xff\xd8")
        return payloads

    def config(self) -> StreamingConfig:
        return StreamingConfig(
            provider=PROVIDER_SRT,
            server_url=str(self.output),
            ffmpeg_path=self.ffmpeg,
            width=WIDTH,
            height=HEIGHT,
            fps=FPS,
            sample_rate=SAMPLE_RATE,
            channels=1,
            video_bitrate_kbps=400,
            audio_bitrate_kbps=64,
            connect_timeout_seconds=5.0,
            live_timeout_seconds=25.0,
            write_timeout_seconds=5.0,
        )

    def test_a_real_encoder_turns_frames_and_pcm_into_a_playable_stream(self) -> None:
        frames = self.make_frames()
        provider = FFmpegStreamingProvider(self.config())
        self.addCleanup(provider.disconnect)

        provider.connect()
        self.assertIs(provider.state, StreamState.CONNECTING)

        audio = _sine_seconds(FRAME_COUNT / FPS)
        deadline = time.monotonic() + 30.0
        index = 0
        pumped = 0
        while time.monotonic() < deadline:
            if index < len(audio):
                block = audio[index : index + 4800]
                index += len(block)
                self.assertTrue(provider.write_audio(AudioChunk(payload=block, sample_rate=SAMPLE_RATE)))
            if pumped < len(frames):
                self.assertTrue(
                    provider.write_video(VideoFrame(payload=frames[pumped], sequence=pumped + 1))
                )
                pumped += 1
            if provider.poll() is StreamState.LIVE and pumped >= len(frames) and index >= len(audio):
                break
            time.sleep(0.05)

        # The encoder reported real pictures, so the stream is genuinely live.
        self.assertIs(provider.state, StreamState.LIVE)
        live = provider.health()
        self.assertGreater(live.frames_encoded, 0)
        self.assertGreater(live.bitrate_kbps, 0.0)
        self.assertAlmostEqual(live.audio_seconds_ingested, FRAME_COUNT / FPS, places=1)
        self.assertEqual(live.video_state, "OK")
        self.assertEqual(live.audio_state, "OK")
        self.assertEqual(live.reconnect_count, 0)

        provider.disconnect()
        self.assertIs(provider.state, StreamState.STOPPED)

        # A clean shutdown must leave a readable file, not a truncated pipe.
        self.assertTrue(self.output.is_file(), "the encoder wrote no output")
        self.assertGreater(self.output.stat().st_size, 10_000)

        decoded = _run([self.ffmpeg, "-hide_banner", "-i", str(self.output), "-f", "null", "-"])
        summary = decoded.stderr.decode("utf-8", "replace")
        # Decode to nothing: this checks the container really holds one track of
        # each kind, not merely that bytes were written.
        self.assertIn("Video: h264", summary, summary)
        self.assertIn("Audio: aac", summary, summary)
        self.assertEqual(summary.count("Video: h264"), 1)
        self.assertEqual(summary.count("Audio: aac"), 1)
        self.assertEqual(decoded.returncode, 0, summary)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
