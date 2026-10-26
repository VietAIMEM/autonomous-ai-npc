"""Kokoro-82M synthesis worker.

This script is executed by the Kokoro virtual environment interpreter (the
one that owns ``torch``/``kokoro``), never by the room_agent backend
interpreter. It speaks newline-delimited JSON on stdin/stdout so the backend
never imports a heavyweight model runtime:

    request : {"id": "1", "op": "warmup"|"synthesize", "text": "...",
               "voice": "af_heart", "speed": 1.0, "output_path": "..."}
    response: {"id": "1", "ok": true, ...}
              {"id": "1", "ok": false, "error": "...", "detail": "..."}

The worker keeps the 82M checkpoint resident across requests and exits on
stdin EOF, so a dead backend cannot leave an orphaned process behind.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path


# This worker lives next to the backend's own ``kokoro.py`` provider shim.
# Drop its script directory from ``sys.path`` so ``import kokoro`` resolves to
# the installed Kokoro-82M package and never to the room_agent shim.
_SCRIPT_DIR = str(Path(__file__).resolve().parent)
sys.path[:] = [
    entry
    for entry in sys.path
    if os.path.abspath(entry or os.getcwd()) != _SCRIPT_DIR
]

REPO_ID = os.environ.get("KOKORO_REPO_ID") or "hexgrad/Kokoro-82M"
DEFAULT_LANG_CODE = os.environ.get("KOKORO_LANG_CODE") or "a"
SAMPLE_RATE = 24_000
CHUNK_GAP_SECONDS = 0.12
MAX_TEXT_LENGTH = 2_000
MAX_DETAIL = 500


def _detail(error: BaseException) -> str:
    text = f"{type(error).__name__}: {error}".strip()
    return text[:MAX_DETAIL].replace("\n", " ")


class Worker:
    """Lazy model owner: the checkpoint loads once, on first warmup."""

    def __init__(self) -> None:
        self._pipeline = None
        self._voices: set[str] = set()
        self._model_load_ms: float | None = None

    def _ensure_pipeline(self, voice: str, lang_code: str):
        if self._pipeline is None:
            from kokoro import KPipeline

            started = time.monotonic()
            self._pipeline = KPipeline(lang_code=lang_code, repo_id=REPO_ID)
            self._model_load_ms = (time.monotonic() - started) * 1000.0
        if voice not in self._voices:
            self._pipeline.load_voice(voice)
            self._voices.add(voice)
        return self._pipeline

    def ready_info(self) -> dict[str, object]:
        return {
            "model": REPO_ID,
            "sample_rate": SAMPLE_RATE,
            "format": "wav",
            "model_load_ms": (
                round(self._model_load_ms, 1)
                if self._model_load_ms is not None
                else None
            ),
            "voices": sorted(self._voices),
        }

    def warmup(self, voice: str, lang_code: str) -> dict[str, object]:
        self._ensure_pipeline(voice, lang_code)
        return self.ready_info()

    def synthesize(
        self,
        text: str,
        voice: str,
        speed: float,
        lang_code: str,
        output_path: Path,
    ) -> dict[str, object]:
        import numpy as np
        import soundfile as sf

        body = (text or "").strip()
        if not body:
            raise ValueError("empty_text")
        if len(body) > MAX_TEXT_LENGTH:
            raise ValueError(f"text_too_long:{len(body)}")
        speed_value = float(speed) if speed else 1.0
        if not 0.1 <= speed_value <= 3.0:
            raise ValueError(f"speed_out_of_range:{speed_value}")

        pipeline = self._ensure_pipeline(voice, lang_code)
        started = time.monotonic()
        chunks: list["np.ndarray"] = []
        for result in pipeline(body, voice=voice, speed=speed_value):
            audio = getattr(result, "audio", None)
            if audio is None:
                continue
            chunks.append(audio.detach().to("cpu").numpy().astype("float32"))
        if not chunks:
            raise RuntimeError("no_audio_generated")

        gap = np.zeros(int(SAMPLE_RATE * CHUNK_GAP_SECONDS), dtype="float32")
        pieces: list["np.ndarray"] = []
        for index, chunk in enumerate(chunks):
            if index:
                pieces.append(gap)
            pieces.append(chunk.reshape(-1))
        waveform = np.concatenate(pieces) if len(pieces) > 1 else pieces[0]
        if not np.isfinite(waveform).all():
            raise RuntimeError("non_finite_audio")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), waveform, SAMPLE_RATE, subtype="PCM_16")
        return {
            "output_path": str(output_path),
            "sample_rate": SAMPLE_RATE,
            "channels": 1,
            "bits_per_sample": 16,
            "duration_seconds": round(len(waveform) / SAMPLE_RATE, 4),
            "chunks": len(chunks),
            "synthesis_ms": round((time.monotonic() - started) * 1000.0, 1),
            **self.ready_info(),
        }


def _respond(payload: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=True) + "\n")
    sys.stdout.flush()


def _handle(worker: Worker, message: dict[str, object]) -> dict[str, object]:
    request_id = str(message.get("id") or "")
    op = str(message.get("op") or "synthesize")
    voice = str(message.get("voice") or "af_heart")
    lang_code = str(message.get("lang_code") or DEFAULT_LANG_CODE)

    if op == "ping":
        return {"id": request_id, "ok": True, "op": op, "loaded": worker._pipeline is not None}
    if op == "warmup":
        return {"id": request_id, "ok": True, "op": op, **worker.warmup(voice, lang_code)}
    if op != "synthesize":
        return {"id": request_id, "ok": False, "error": "unknown_op", "detail": op}

    output_path = message.get("output_path")
    if not output_path:
        return {"id": request_id, "ok": False, "error": "missing_output_path", "detail": ""}
    try:
        info = worker.synthesize(
            str(message.get("text") or ""),
            voice,
            float(message.get("speed") or 1.0),
            lang_code,
            Path(str(output_path)),
        )
    except Exception as exc:  # the protocol must survive any model failure
        return {"id": request_id, "ok": False, "error": _detail(exc), "detail": _detail(exc)}
    return {"id": request_id, "ok": True, "op": op, **info}


def main() -> int:
    worker = Worker()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except ValueError:
            _respond({"id": "", "ok": False, "error": "invalid_json", "detail": ""})
            continue
        if not isinstance(message, dict):
            _respond({"id": "", "ok": False, "error": "invalid_request", "detail": ""})
            continue
        try:
            response = _handle(worker, message)
        except Exception as exc:
            print(traceback.format_exc(), file=sys.stderr, flush=True)
            response = {"id": str(message.get("id") or ""), "ok": False, "error": _detail(exc), "detail": _detail(exc)}
        _respond(response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
