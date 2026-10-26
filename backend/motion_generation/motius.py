"""Isolated Motius T2M-GPT provider boundary.

The provider never imports PyTorch, Motius, or a checkpoint. Generation runs in
the dedicated Motius Python environment and returns structured motion data only.
"""

from __future__ import annotations

import json
import os
import subprocess
import uuid
import threading
from pathlib import Path

from .base import MotionGenerator
from .models import (
    MotionGenerationConfig,
    MotionGenerationError,
    MotionGenerationRequest,
    MotionGenerationResult,
    MotionGenerationStatus,
)
from .validation import validate_motion_payload


class MotiusT2MGPTGenerator(MotionGenerator):
    name = "motius-t2m-gpt"

    def __init__(self, config: MotionGenerationConfig | None = None) -> None:
        self.config = config or MotionGenerationConfig()
        self._loaded = False
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._loaded

    def generate(self, request: MotionGenerationRequest) -> MotionGenerationResult:
        if not request.prompt.strip():
            raise MotionGenerationError("EMPTY_PROMPT")
        frame_count = round(request.duration_seconds * self.config.motion_rate)
        try:
            response = self._run_worker(request.prompt, frame_count)
        except MotionGenerationError:
            raise
        except subprocess.TimeoutExpired as exc:
            raise MotionGenerationError("timeout") from exc
        except Exception as exc:
            raise MotionGenerationError("generation_failed", type(exc).__name__) from exc

        if response.get("status") != "OK":
            raise MotionGenerationError(
                "generation_failed",
                str(response.get("reason") or response.get("status")),
            )
        motion = response.get("motion")

        probe = validate_motion_payload(
            motion,
            frame_rate=self.config.motion_rate,
            expected_duration_seconds=request.duration_seconds,
        )
        return MotionGenerationResult(
            request=request,
            motion_id=f"motius_{uuid.uuid4().hex[:16]}",
            duration_seconds=probe["duration_seconds"],
            format=probe["format"],
            generation_status=MotionGenerationStatus.GENERATED,
        metadata={
            **probe,
            "resource": dict(response.get("resource") or {}),
        },
            motion=tuple(tuple(float(value) for value in row) for row in motion),
        )

    def _build_command(self) -> list[str]:
        # -P prevents the worker directory from shadowing the Motius package.
        return [
            str(self.config.python_executable),
            "-P",
            str(Path(__file__).with_name("motius_worker.py")),
            str(self.config.repo_path),
        ]

    def _build_input(self, prompt: str, frame_count: int) -> str:
        return json.dumps(
            {
                "prompt": prompt,
                "frame_count": frame_count,
                "device": self.config.device,
            }
        )

    def _run_worker(self, prompt: str, frame_count: int) -> dict[str, object]:
        try:
            process = subprocess.Popen(
                self._build_command(),
                cwd=str(self.config.repo_path),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env={
                    **os.environ,
                    "PYTHONUNBUFFERED": "1",
                    "PYTHONPATH": os.pathsep.join(
                        (str(self.config.repo_path), os.environ.get("PYTHONPATH", ""))
                    ).rstrip(os.pathsep),
                },
            )
        except OSError as exc:
            raise MotionGenerationError("model_unavailable", type(exc).__name__) from exc

        try:
            stdout, stderr = process.communicate(
                input=self._build_input(prompt, frame_count),
                timeout=self.config.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            process.kill()
            process.communicate()
            raise

        result_line = next(
            (
                line
                for line in (stdout or "").splitlines()
                if line.startswith("MOTIUS_RESULT ")
            ),
            None,
        )
        if result_line is None:
            if process.returncode != 0:
                raise MotionGenerationError(
                    "model_unavailable",
                    f"exit_{process.returncode}",
                )
            raise MotionGenerationError("generation_failed", "INVALID_RESPONSE")
        try:
            response = json.loads(result_line.removeprefix("MOTIUS_RESULT "))
            if not isinstance(response, dict):
                raise ValueError("response is not an object")
        except (json.JSONDecodeError, ValueError) as exc:
            raise MotionGenerationError("generation_failed", "INVALID_RESPONSE") from exc
        return response

    def describe(self) -> dict[str, object]:
        return {
            "provider": self.name,
            "model": self.config.model,
            "checkpoint": self.config.checkpoint,
            "device": self.config.device,
            "loaded": self._loaded,
            "representation": self.config.representation,
            "motion_rate": self.config.motion_rate,
            "isolated_process": True,
            "python_executable": self.config.python_executable,
            "repo_path": self.config.repo_path,
        }
