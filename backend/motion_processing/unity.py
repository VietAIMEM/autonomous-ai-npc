"""Unity batch adapters for Phase 17 processing and production playback."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ..learning.models import MotionDefinition
from .models import (
    MotionPlaybackReport,
    MotionProcessingConfig,
    MotionProcessingError,
    MotionProcessingReport,
    MotionProcessingRequest,
)


class MotionProcessingAdapter:
    def process(self, request: MotionProcessingRequest, source_motion: MotionDefinition) -> MotionProcessingReport:
        raise NotImplementedError

    def validate_playback(
        self,
        request: MotionProcessingRequest,
        source_motion: MotionDefinition,
        processing: MotionProcessingReport,
    ) -> MotionPlaybackReport:
        raise NotImplementedError


class UnityMotionProcessingAdapter(MotionProcessingAdapter):
    name = "unity-motion-processor"

    def __init__(self, config: MotionProcessingConfig | None = None) -> None:
        self.config = config or MotionProcessingConfig()

    def process(self, request: MotionProcessingRequest, source_motion: MotionDefinition) -> MotionProcessingReport:
        report_path = self.config.unity_project_path / "Temp" / "Phase17MotionProcessing.json"
        return self._run(
            report_path,
            [
                "-batchmode", "-quit",
                "-projectPath", str(self.config.unity_project_path),
                "-executeMethod", "RoomAgent.Editor.Phase17MotionProcessor.Run",
                "-phase17SourceMotionId", request.source_motion_id,
                "-phase17SourceVersion", str(request.source_version),
                "-phase17SourceReference", source_motion.animation_reference,
                "-phase17ProcessedMotionId", request.processed_motion_id,
                "-phase17ProcessingVersion", str(request.processing_version),
                "-logFile", str(self.config.unity_project_path / "Logs" / "phase17_motion_processing.log"),
            ],
            self.config.timeout_seconds,
            MotionProcessingReport.from_dict,
            "UNITY_PROCESSING_REPORT_MISSING",
            "UNITY_PROCESSING_REPORT_INVALID",
            "UNITY_PROCESSING_EXIT_FAILED",
        )

    def validate_playback(
        self,
        request: MotionProcessingRequest,
        source_motion: MotionDefinition,
        processing: MotionProcessingReport,
    ) -> MotionPlaybackReport:
        report_path = self.config.unity_project_path / "Temp" / "Phase17ProductionPlayback.json"
        return self._run(
            report_path,
            [
                "-batchmode",
                "-projectPath", str(self.config.unity_project_path),
                "-executeMethod", "RoomAgent.Editor.Phase17ProductionPlaybackValidator.Run",
                "-phase17SourceMotionId", request.source_motion_id,
                "-phase17SourceVersion", str(request.source_version),
                "-phase17ProcessedMotionId", request.processed_motion_id,
                "-phase17ProcessingVersion", str(request.processing_version),
                "-phase17SourceReference", source_motion.animation_reference,
                "-phase17ProcessedClipPath", processing.processed_clip_path,
                "-logFile", str(self.config.unity_project_path / "Logs" / "phase17_production_playback.log"),
            ],
            self.config.playback_timeout_seconds,
            MotionPlaybackReport.from_dict,
            "UNITY_PLAYBACK_REPORT_MISSING",
            "UNITY_PLAYBACK_REPORT_INVALID",
            "UNITY_PLAYBACK_EXIT_FAILED",
        )

    def _run(
        self,
        report_path: Path,
        command_tail: list[str],
        timeout: float,
        parser,
        missing_reason: str,
        invalid_reason: str,
        exit_reason: str,
    ):
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.unlink(missing_ok=True)
        command = [str(self.config.unity_executable), *command_tail]
        try:
            completed = subprocess.run(
                command,
                cwd=str(self.config.unity_project_path),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise MotionProcessingError("UNITY_PROCESSING_TIMEOUT") from exc
        except OSError as exc:
            raise MotionProcessingError("UNITY_EXECUTABLE_UNAVAILABLE", type(exc).__name__) from exc
        if not report_path.is_file():
            raise MotionProcessingError(missing_reason, f"exit_{completed.returncode}")
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MotionProcessingError(invalid_reason, type(exc).__name__) from exc
        if not isinstance(payload, dict):
            raise MotionProcessingError(invalid_reason, "not_an_object")
        try:
            report = parser(payload)
        except (TypeError, ValueError) as exc:
            raise MotionProcessingError(invalid_reason, str(exc)) from exc
        if completed.returncode != 0:
            raise MotionProcessingError(exit_reason, f"exit_{completed.returncode}")
        if not report.success:
            raise MotionProcessingError("UNITY_PROCESSING_FAILED", report.error or "unknown_error")
        return report


class UnityEditorReportMotionProcessingAdapter(MotionProcessingAdapter):
    """Consume editor-produced reports for deterministic tests and local reruns."""

    name = "unity-editor-motion-processing-report"

    def __init__(
        self,
        processing_report_path: Path | str,
        playback_report_path: Path | str,
    ) -> None:
        self.processing_report_path = Path(processing_report_path)
        self.playback_report_path = Path(playback_report_path)

    def process(self, request: MotionProcessingRequest, source_motion: MotionDefinition) -> MotionProcessingReport:
        return _read_report(self.processing_report_path, MotionProcessingReport.from_dict)

    def validate_playback(
        self,
        request: MotionProcessingRequest,
        source_motion: MotionDefinition,
        processing: MotionProcessingReport,
    ) -> MotionPlaybackReport:
        return _read_report(self.playback_report_path, MotionPlaybackReport.from_dict)


def _read_report(path: Path, parser):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MotionProcessingError("UNITY_PROCESSING_REPORT_INVALID", type(exc).__name__) from exc
    if not isinstance(payload, dict):
        raise MotionProcessingError("UNITY_PROCESSING_REPORT_INVALID", "not_an_object")
    try:
        return parser(payload)
    except (TypeError, ValueError) as exc:
        raise MotionProcessingError("UNITY_PROCESSING_REPORT_INVALID", str(exc)) from exc


__all__ = [
    "MotionProcessingAdapter",
    "UnityEditorReportMotionProcessingAdapter",
    "UnityMotionProcessingAdapter",
]
