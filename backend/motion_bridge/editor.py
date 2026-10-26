"""Bridge adapters that hand Unity work to an already-running editor.

Batch mode remains the default. ``MOTION_UNITY_MODE=editor`` selects a
file-based request/response contract implemented by
``unity/3d_unity/Assets/Editor/Phase23AdminMotionBridge.cs``. The bridge only
dispatches to the existing Phase 16/17 editor entry points; reports are parsed
by the same editor-report adapters used elsewhere, so validation is unchanged.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, ClassVar

from ..learning.models import MotionDefinition
from ..motion_import.models import (
    MotionImportConfig,
    MotionImportError,
    MotionImportRequest,
    UnityMotionImportReport,
)
from ..motion_import.unity import MotionImportAdapter, UnityEditorReportMotionImportAdapter
from ..motion_processing.models import (
    MotionPlaybackReport,
    MotionProcessingConfig,
    MotionProcessingError,
    MotionProcessingReport,
    MotionProcessingRequest,
)
from ..motion_processing.unity import (
    MotionProcessingAdapter,
    UnityEditorReportMotionProcessingAdapter,
)


MOTION_UNITY_MODE_BATCH = "batch"
MOTION_UNITY_MODE_EDITOR = "editor"

_BRIDGE_DIR = Path("Temp") / "Phase23AdminMotionBridge"
_REQUEST_SUFFIX = ".request.json"
_RESPONSE_SUFFIX = ".response.json"
_POLL_SECONDS = 0.05


class _EditorBridgeClient:
    """Write one request file and wait for its bounded response file."""

    def __init__(self, unity_project_path: Path | str, timeout_seconds: float) -> None:
        self.unity_project_path = Path(unity_project_path)
        self.timeout_seconds = float(timeout_seconds)
        self.bridge_dir = self.unity_project_path / _BRIDGE_DIR

    def send(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.bridge_dir.mkdir(parents=True, exist_ok=True)
        request_id = f"{operation}_{uuid.uuid4().hex[:16]}"
        request_path = self.bridge_dir / f"{request_id}{_REQUEST_SUFFIX}"
        response_path = self.bridge_dir / f"{request_id}{_RESPONSE_SUFFIX}"
        response_path.unlink(missing_ok=True)
        body = {
            "id": request_id,
            "operation": operation,
            "responsePath": str(response_path),
            **payload,
        }
        # A .pending name keeps the watcher from seeing a partial request.
        pending = self.bridge_dir / f"{request_id}.pending"
        pending.write_text(json.dumps(body, ensure_ascii=True), encoding="utf-8")
        pending.replace(request_path)

        deadline = time.monotonic() + self.timeout_seconds
        while time.monotonic() < deadline:
            try:
                raw = response_path.read_text(encoding="utf-8")
            except OSError:
                time.sleep(_POLL_SECONDS)
                continue
            try:
                response = json.loads(raw)
            except json.JSONDecodeError:
                time.sleep(_POLL_SECONDS)
                continue
            if not isinstance(response, dict):
                raise MotionProcessingError("UNITY_BRIDGE_RESPONSE_INVALID", "not_an_object")
            response_path.unlink(missing_ok=True)
            request_path.unlink(missing_ok=True)
            return response
        request_path.unlink(missing_ok=True)
        raise TimeoutError(
            f"unity_editor_bridge_timeout:{operation}:{self.timeout_seconds:g}s"
        )

    @staticmethod
    def report_paths(project_path: Path | str) -> dict[str, Path]:
        root = Path(project_path)
        return {
            "import": root / "Temp" / "Phase16MotionImport.json",
            "processing": root / "Temp" / "Phase17MotionProcessing.json",
            "playback": root / "Temp" / "Phase17ProductionPlayback.json",
        }


class _EditorReportWaiter:
    """Delete a stale report, then wait for a fresh one written by the editor."""

    _required_keys: ClassVar[dict[str, tuple[str, ...]]] = {
        "import": ("success", "motionId"),
        "processing": ("success", "processedMotionId"),
        "playback": ("success", "processedMotionId"),
    }

    def __init__(self, report_paths: dict[str, Path]) -> None:
        self.report_paths = report_paths

    def clear(self, kind: str) -> None:
        try:
            self.report_paths[kind].unlink()
        except FileNotFoundError:
            pass

    def wait(self, kind: str, timeout_seconds: float) -> None:
        path = self.report_paths[kind]
        required = self._required_keys[kind]
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                time.sleep(_POLL_SECONDS)
                continue
            if isinstance(payload, dict) and all(key in payload for key in required):
                return
            time.sleep(_POLL_SECONDS)
        raise TimeoutError(f"unity_editor_report_timeout:{kind}")


class EditorBridgeMotionImportAdapter(MotionImportAdapter):
    """Send Phase 16 import work to the open editor, then read its report."""

    name: ClassVar[str] = "unity-editor-bridge-motion-import"

    def __init__(self, config: MotionImportConfig | None = None) -> None:
        self.config = config or MotionImportConfig()
        self._client = _EditorBridgeClient(
            self.config.unity_project_path, self.config.timeout_seconds
        )
        self._waiter = _EditorReportWaiter(
            _EditorBridgeClient.report_paths(self.config.unity_project_path)
        )
        self._reports = UnityEditorReportMotionImportAdapter(
            self._waiter.report_paths["import"], self.config.unity_project_path
        )

    def import_motion(self, request: MotionImportRequest) -> UnityMotionImportReport:
        self._waiter.clear("import")
        try:
            response = self._client.send(
                "import",
                {
                    "sourceFile": str(request.source_file),
                    "clipName": request.clip_name or "",
                    "motionId": request.actual_motion_id,
                    "version": request.version,
                },
            )
        except TimeoutError as exc:
            raise MotionImportError("UNITY_EDITOR_BRIDGE_TIMEOUT", str(exc)) from exc
        _raise_import_error(response)
        try:
            self._waiter.wait("import", self.config.timeout_seconds)
        except TimeoutError as exc:
            raise MotionImportError("UNITY_IMPORT_REPORT_MISSING", str(exc)) from exc
        return self._reports.import_motion(request)


class EditorBridgeMotionProcessingAdapter(MotionProcessingAdapter):
    """Send Phase 17 process/playback work to the open editor."""

    name: ClassVar[str] = "unity-editor-bridge-motion-processing"

    def __init__(self, config: MotionProcessingConfig | None = None) -> None:
        self.config = config or MotionProcessingConfig()
        self._client = _EditorBridgeClient(
            self.config.unity_project_path, self.config.timeout_seconds
        )
        self._waiter = _EditorReportWaiter(
            _EditorBridgeClient.report_paths(self.config.unity_project_path)
        )
        self._reports = UnityEditorReportMotionProcessingAdapter(
            self._waiter.report_paths["processing"],
            self._waiter.report_paths["playback"],
        )

    def process(
        self, request: MotionProcessingRequest, source_motion: MotionDefinition
    ) -> MotionProcessingReport:
        self._waiter.clear("processing")
        try:
            response = self._client.send(
                "process",
                {
                    "sourceMotionId": request.source_motion_id,
                    "sourceVersion": request.source_version,
                    "sourceReference": source_motion.animation_reference,
                    "processedMotionId": request.processed_motion_id,
                    "processingVersion": request.processing_version,
                },
            )
        except TimeoutError as exc:
            raise MotionProcessingError("UNITY_EDITOR_BRIDGE_TIMEOUT", str(exc)) from exc
        _raise_processing_error(response)
        try:
            self._waiter.wait("processing", self.config.timeout_seconds)
        except TimeoutError as exc:
            raise MotionProcessingError("UNITY_PROCESSING_REPORT_MISSING", str(exc)) from exc
        return self._reports.process(request, source_motion)

    def validate_playback(
        self,
        request: MotionProcessingRequest,
        source_motion: MotionDefinition,
        processing: MotionProcessingReport,
    ) -> MotionPlaybackReport:
        self._waiter.clear("playback")
        try:
            response = self._client.send(
                "playback",
                {
                    "sourceMotionId": request.source_motion_id,
                    "sourceVersion": request.source_version,
                    "processedMotionId": request.processed_motion_id,
                    "processingVersion": request.processing_version,
                    "sourceReference": source_motion.animation_reference,
                    "processedClipPath": processing.processed_clip_path,
                },
            )
        except TimeoutError as exc:
            raise MotionProcessingError("UNITY_EDITOR_BRIDGE_TIMEOUT", str(exc)) from exc
        _raise_processing_error(response)
        try:
            self._waiter.wait("playback", self.config.playback_timeout_seconds)
        except TimeoutError as exc:
            raise MotionProcessingError("UNITY_PLAYBACK_REPORT_MISSING", str(exc)) from exc
        return self._reports.validate_playback(request, source_motion, processing)


def _raise_import_error(response: dict[str, Any]) -> None:
    if response.get("status") != "OK" or response.get("success") is not True:
        raise MotionImportError(
            str(response.get("error") or "UNITY_EDITOR_BRIDGE_FAILED"),
            str(response.get("message") or ""),
        )


def _raise_processing_error(response: dict[str, Any]) -> None:
    if response.get("status") != "OK" or response.get("success") is not True:
        raise MotionProcessingError(
            str(response.get("error") or "UNITY_EDITOR_BRIDGE_FAILED"),
            str(response.get("message") or ""),
        )


def build_motion_import_adapter(
    mode: str, config: MotionImportConfig | None = None
) -> MotionImportAdapter:
    normalized = (mode or MOTION_UNITY_MODE_BATCH).strip().lower()
    if normalized == MOTION_UNITY_MODE_EDITOR:
        return EditorBridgeMotionImportAdapter(config)
    if normalized == MOTION_UNITY_MODE_BATCH:
        from ..motion_import.unity import UnityMotionImportAdapter

        return UnityMotionImportAdapter(config)
    raise ValueError(f"unknown_motion_unity_mode:{mode}")


def build_motion_processing_adapter(
    mode: str, config: MotionProcessingConfig | None = None
) -> MotionProcessingAdapter:
    normalized = (mode or MOTION_UNITY_MODE_BATCH).strip().lower()
    if normalized == MOTION_UNITY_MODE_EDITOR:
        return EditorBridgeMotionProcessingAdapter(config)
    if normalized == MOTION_UNITY_MODE_BATCH:
        from ..motion_processing.unity import UnityMotionProcessingAdapter

        return UnityMotionProcessingAdapter(config)
    raise ValueError(f"unknown_motion_unity_mode:{mode}")


__all__ = [
    "MOTION_UNITY_MODE_BATCH",
    "MOTION_UNITY_MODE_EDITOR",
    "EditorBridgeMotionImportAdapter",
    "EditorBridgeMotionProcessingAdapter",
    "build_motion_import_adapter",
    "build_motion_processing_adapter",
]
