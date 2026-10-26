"""Unity batch-mode adapter for motion-file import."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .models import MotionImportConfig, MotionImportError, MotionImportRequest, UnityMotionImportReport
from .validation import hash_source_file


class MotionImportAdapter:
    def import_motion(self, request: MotionImportRequest) -> UnityMotionImportReport:
        raise NotImplementedError


class UnityMotionImportAdapter(MotionImportAdapter):
    name = "unity-motion-importer"

    def __init__(self, config: MotionImportConfig | None = None) -> None:
        self.config = config or MotionImportConfig()

    def import_motion(self, request: MotionImportRequest) -> UnityMotionImportReport:
        report_path = self.config.unity_project_path / "Temp" / "Phase16MotionImport.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.unlink(missing_ok=True)
        command = [
            str(self.config.unity_executable),
            "-batchmode",
            "-quit",
            "-projectPath",
            str(self.config.unity_project_path),
            "-executeMethod",
            "RoomAgent.Editor.Phase16MotionFileImporter.Run",
            "-phase16SourceFile",
            str(request.source_file),
            "-phase16ClipName",
            request.clip_name or "",
            "-phase16MotionId",
            request.actual_motion_id,
            "-phase16Version",
            str(request.version),
            "-logFile",
            str(self.config.unity_project_path / "Logs" / "phase16_motion_import.log"),
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=str(self.config.unity_project_path),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=self.config.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise MotionImportError("UNITY_IMPORT_TIMEOUT") from exc
        except OSError as exc:
            raise MotionImportError("UNITY_EXECUTABLE_UNAVAILABLE", type(exc).__name__) from exc

        if not report_path.is_file():
            raise MotionImportError(
                "UNITY_IMPORT_REPORT_MISSING", f"exit_{completed.returncode}"
            )
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MotionImportError("UNITY_IMPORT_REPORT_INVALID", type(exc).__name__) from exc
        if not isinstance(payload, dict):
            raise MotionImportError("UNITY_IMPORT_REPORT_INVALID", "not_an_object")
        try:
            report = UnityMotionImportReport.from_dict(payload)
        except (TypeError, ValueError) as exc:
            raise MotionImportError("UNITY_IMPORT_REPORT_INVALID", str(exc)) from exc
        if completed.returncode != 0:
            raise MotionImportError("UNITY_IMPORT_EXIT_FAILED", f"exit_{completed.returncode}")
        if not report.success:
            raise MotionImportError("UNITY_IMPORT_FAILED", report.error or "unknown_error")
        return report


class UnityEditorReportMotionImportAdapter(MotionImportAdapter):
    """Consume a report produced by a Unity editor-hosted production import."""

    name = "unity-editor-motion-import-report"

    def __init__(self, report_path: Path | str, unity_project_path: Path | str) -> None:
        self.report_path = Path(report_path)
        self.unity_project_path = Path(unity_project_path)

    def import_motion(self, request: MotionImportRequest) -> UnityMotionImportReport:
        try:
            payload = json.loads(self.report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MotionImportError("UNITY_IMPORT_REPORT_INVALID", type(exc).__name__) from exc
        if not isinstance(payload, dict):
            raise MotionImportError("UNITY_IMPORT_REPORT_INVALID", "not_an_object")
        try:
            report = UnityMotionImportReport.from_dict(payload)
        except (TypeError, ValueError) as exc:
            raise MotionImportError("UNITY_IMPORT_REPORT_INVALID", str(exc)) from exc
        if not report.success:
            raise MotionImportError("UNITY_IMPORT_FAILED", report.error or "unknown_error")
        return report


__all__ = [
    "MotionImportAdapter",
    "UnityMotionImportAdapter",
    "UnityEditorReportMotionImportAdapter",
]
