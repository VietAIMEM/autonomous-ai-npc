"""Focused tests for the Phase 23 live-Unity-editor motion bridge.

Batch mode stays the default; ``MOTION_UNITY_MODE=editor`` routes Phase 16/17
work through the open editor's request/response contract. These tests fake the
editor side of that contract and reuse the canonical report adapters, so a
report that reaches the service layer here is parsed exactly as in production.
"""

from __future__ import annotations

import json
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable

from backend.learning.models import MotionDefinition
from backend.motion_bridge import (
    MOTION_UNITY_MODE_BATCH,
    MOTION_UNITY_MODE_EDITOR,
    EditorBridgeMotionImportAdapter,
    EditorBridgeMotionProcessingAdapter,
    build_motion_import_adapter,
    build_motion_processing_adapter,
)
from backend.motion_bridge.editor import _EditorBridgeClient, _EditorReportWaiter
from backend.motion_import.models import (
    MotionImportConfig,
    MotionImportError,
    MotionImportRequest,
)
from backend.motion_import.unity import UnityMotionImportAdapter
from backend.motion_processing.models import (
    MotionProcessingConfig,
    MotionProcessingError,
    MotionProcessingRequest,
)
from backend.motion_processing.unity import UnityMotionProcessingAdapter


BRIDGE_DIRECTORY = Path("Temp") / "Phase23AdminMotionBridge"


def _import_report_payload(motion_id: str, version: int = 1) -> dict[str, Any]:
    return {
        "success": True,
        "motionId": motion_id,
        "version": version,
        "originalSourceFile": "D:/source/Walking.fbx",
        "stagedSourceFile": "D:/staging/Walking.fbx",
        "sourceSha256": "a" * 64,
        "sourceFormat": "FBX",
        "clipName": "Walking",
        "durationSeconds": 2.5,
        "fps": 30.0,
        "frameCount": 75,
        "curveCount": 1,
        "keyCount": 2,
        "finiteValues": True,
        "boneCount": 19,
        "rootBoneName": "Hips",
        "sourceAvatarValid": True,
        "sourceAvatarHuman": True,
        "targetAvatarValid": True,
        "targetAvatarHuman": True,
        "skeletonStatus": "COMPATIBLE",
        "importStatus": "VALIDATED",
        "animationReference": "Assets/MotionImports/Phase16/Walking.fbx",
    }


def _processing_report_payload(motion_id: str, source_motion_id: str) -> dict[str, Any]:
    return {
        "success": True,
        "sourceMotionId": source_motion_id,
        "sourceVersion": 1,
        "processedMotionId": motion_id,
        "processingVersion": 1,
        "state": "READY",
        "qualityStatus": "VALID",
        "qualityScore": 0.95,
        "processedClipPath": "Assets/Resources/GeneratedMotions/Phase17/Walking.anim",
        "animationReference": "Assets/Resources/GeneratedMotions/Phase17/Walking.anim",
        "libraryPath": "Assets/Resources/GeneratedMotions/Phase17/Walking.anim",
        "durationSeconds": 2.5,
        "fps": 30.0,
        "frameCount": 75,
        "curveCount": 1,
        "keyCount": 2,
        "boneCount": 19,
        "rootBoneName": "Hips",
    }


def _playback_report_payload(motion_id: str, source_motion_id: str) -> dict[str, Any]:
    return {
        "success": True,
        "sourceMotionId": source_motion_id,
        "processedMotionId": motion_id,
        "processingVersion": 1,
        "animationReference": "Assets/Resources/GeneratedMotions/Phase17/Walking.anim",
        "processedClipPath": "Assets/Resources/GeneratedMotions/Phase17/Walking.anim",
        "avatarValid": True,
        "avatarHuman": True,
        "clipPersistent": True,
        "clipDurationSeconds": 2.5,
        "clipFrameRate": 30.0,
        "playbackSuccess": True,
        "libraryResolved": True,
    }


class _SimulatedEditor(threading.Thread):
    """Answer bridge requests the way the Unity editor watcher does."""

    def __init__(
        self,
        project_root: Path,
        handler: Callable[[Path, dict[str, Any]], None],
    ) -> None:
        super().__init__(daemon=True)
        self.project_root = Path(project_root)
        self.handler = handler
        self.stop_event = threading.Event()
        self.handled_operations: list[str] = []
        self.errors: list[str] = []

    def run(self) -> None:
        directory = self.project_root / BRIDGE_DIRECTORY
        while not self.stop_event.is_set():
            try:
                requests = sorted(directory.glob("*.request.json"))
            except OSError:
                requests = []
            for path in requests:
                try:
                    body = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                try:
                    path.unlink(missing_ok=True)
                    self.handler(self.project_root, body)
                    self.handled_operations.append(str(body.get("operation")))
                except Exception as exc:  # pragma: no cover - surfaces in assertEqual
                    self.errors.append(f"{type(exc).__name__}:{exc}")
            time.sleep(0.01)

    def stop(self) -> None:
        self.stop_event.set()
        self.join(timeout=5.0)

    def wait_for_operations(self, count: int, timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if len(self.handled_operations) >= count:
                return True
            time.sleep(0.01)
        return False


def _write_response(
    request: dict[str, Any],
    *,
    success: bool = True,
    error: str = "",
    message: str = "",
) -> None:
    response_path = Path(request["responsePath"])
    response_path.parent.mkdir(parents=True, exist_ok=True)
    response_path.write_text(
        json.dumps(
            {
                "id": request.get("id", ""),
                "status": "OK" if success else "ERROR",
                "success": success,
                "error": error,
                "message": message,
            }
        ),
        encoding="utf-8",
    )


def _write_report(project_root: Path, relative_path: str, payload: dict[str, Any]) -> None:
    path = project_root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _import_request(motion_id: str, source_file: Path) -> MotionImportRequest:
    return MotionImportRequest(
        source_file=source_file,
        source_sha256="a" * 64,
        source_format="FBX",
        motion_id=motion_id,
        actual_motion_id=motion_id,
        version=1,
        name="Walking",
        description="phase23 bridge test",
    )


def _processing_request(processed_id: str, source_id: str) -> MotionProcessingRequest:
    return MotionProcessingRequest(
        source_motion_id=source_id,
        source_version=1,
        processed_motion_id=processed_id,
        processing_version=1,
        name="Walking",
        description="phase23 bridge test",
    )


def _source_motion(motion_id: str) -> MotionDefinition:
    return MotionDefinition(
        motion_id=motion_id,
        name="Walking",
        description="phase23 bridge test",
        animation_reference="Assets/MotionImports/Phase16/Walking.fbx",
        duration_seconds=2.5,
    )


class MotionBridgeFactoryTests(unittest.TestCase):
    def test_batch_mode_is_default_for_both_adapters(self) -> None:
        self.assertIsInstance(build_motion_import_adapter(""), UnityMotionImportAdapter)
        self.assertIsInstance(
            build_motion_import_adapter(MOTION_UNITY_MODE_BATCH), UnityMotionImportAdapter
        )
        self.assertIsInstance(
            build_motion_processing_adapter(""), UnityMotionProcessingAdapter
        )
        self.assertIsInstance(
            build_motion_processing_adapter(MOTION_UNITY_MODE_BATCH),
            UnityMotionProcessingAdapter,
        )

    def test_editor_mode_selects_bridge_adapters(self) -> None:
        with TemporaryDirectory() as directory:
            project = Path(directory)
            import_adapter = build_motion_import_adapter(
                MOTION_UNITY_MODE_EDITOR, MotionImportConfig(unity_project_path=project)
            )
            processing_adapter = build_motion_processing_adapter(
                MOTION_UNITY_MODE_EDITOR,
                MotionProcessingConfig(unity_project_path=project),
            )
        self.assertIsInstance(import_adapter, EditorBridgeMotionImportAdapter)
        self.assertIsInstance(processing_adapter, EditorBridgeMotionProcessingAdapter)
        self.assertEqual(
            import_adapter.name, "unity-editor-bridge-motion-import"
        )
        self.assertEqual(
            processing_adapter.name, "unity-editor-bridge-motion-processing"
        )

    def test_unknown_mode_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_motion_import_adapter("carrier-pigeon")
        with self.assertRaises(ValueError):
            build_motion_processing_adapter("carrier-pigeon")

    def test_report_paths_match_phase16_and_phase17_conventions(self) -> None:
        paths = _EditorBridgeClient.report_paths("unity/3d_unity")
        self.assertEqual(
            paths["import"], Path("unity/3d_unity") / "Temp" / "Phase16MotionImport.json"
        )
        self.assertEqual(
            paths["processing"],
            Path("unity/3d_unity") / "Temp" / "Phase17MotionProcessing.json",
        )
        self.assertEqual(
            paths["playback"],
            Path("unity/3d_unity") / "Temp" / "Phase17ProductionPlayback.json",
        )


class EditorBridgeClientTests(unittest.TestCase):
    def test_round_trip_returns_response_and_cleans_request(self) -> None:
        with TemporaryDirectory() as directory:
            project = Path(directory)

            def handler(_project: Path, request: dict[str, Any]) -> None:
                self.assertEqual(request["operation"], "import")
                _write_response(request)

            editor = _SimulatedEditor(project, handler)
            editor.start()
            try:
                client = _EditorBridgeClient(project, timeout_seconds=5.0)
                response = client.send("import", {"sourceFile": "D:/src.fbx"})
            finally:
                editor.stop()
            self.assertEqual(editor.errors, [])
            self.assertEqual(response["status"], "OK")
            self.assertTrue(response["success"])
            self.assertEqual(
                list((project / BRIDGE_DIRECTORY).glob("*.request.json")), []
            )

    def test_missing_editor_times_out_without_mock_success(self) -> None:
        with TemporaryDirectory() as directory:
            client = _EditorBridgeClient(Path(directory), timeout_seconds=0.3)
            with self.assertRaises(TimeoutError):
                client.send("import", {"sourceFile": "D:/src.fbx"})

    def test_stale_report_is_cleared_and_fresh_report_is_waited_for(self) -> None:
        with TemporaryDirectory() as directory:
            project = Path(directory)
            report_path = project / "Temp" / "Phase16MotionImport.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps({"success": True, "motionId": "stale"}), encoding="utf-8"
            )
            waiter = _EditorReportWaiter(_EditorBridgeClient.report_paths(project))
            waiter.clear("import")
            self.assertFalse(report_path.exists())
            with self.assertRaises(TimeoutError):
                waiter.wait("import", 0.3)
            report_path.write_text(
                json.dumps({"success": True, "motionId": "fresh"}), encoding="utf-8"
            )
            waiter.wait("import", 2.0)


class EditorBridgeImportAdapterTests(unittest.TestCase):
    def test_import_returns_canonical_report_from_editor_output(self) -> None:
        with TemporaryDirectory() as directory:
            project = Path(directory)
            motion_id = "phase23_bridge_walking"

            def handler(_project: Path, request: dict[str, Any]) -> None:
                self.assertEqual(request["operation"], "import")
                self.assertEqual(request["motionId"], motion_id)
                _write_report(
                    project,
                    "Temp/Phase16MotionImport.json",
                    _import_report_payload(motion_id, int(request["version"])),
                )
                _write_response(request)

            editor = _SimulatedEditor(project, handler)
            editor.start()
            try:
                adapter = build_motion_import_adapter(
                    MOTION_UNITY_MODE_EDITOR,
                    MotionImportConfig(
                        unity_project_path=project, timeout_seconds=10.0
                    ),
                )
                report = adapter.import_motion(
                    _import_request(motion_id, project / "Walking.fbx")
                )
            finally:
                editor.stop()
            self.assertEqual(editor.errors, [])
            self.assertTrue(report.success)
            self.assertEqual(report.motion_id, motion_id)
            self.assertEqual(report.skeleton_status, "COMPATIBLE")
            self.assertEqual(report.fps, 30.0)

    def test_ignores_stale_report_and_requires_a_fresh_one(self) -> None:
        with TemporaryDirectory() as directory:
            project = Path(directory)
            motion_id = "phase23_bridge_fresh"
            _write_report(
                project,
                "Temp/Phase16MotionImport.json",
                _import_report_payload("phase23_bridge_stale"),
            )

            def handler(_project: Path, request: dict[str, Any]) -> None:
                _write_report(
                    project,
                    "Temp/Phase16MotionImport.json",
                    _import_report_payload(motion_id, int(request["version"])),
                )
                _write_response(request)

            editor = _SimulatedEditor(project, handler)
            editor.start()
            try:
                adapter = EditorBridgeMotionImportAdapter(
                    MotionImportConfig(unity_project_path=project, timeout_seconds=10.0)
                )
                report = adapter.import_motion(
                    _import_request(motion_id, project / "Walking.fbx")
                )
            finally:
                editor.stop()
            self.assertEqual(report.motion_id, motion_id)

    def test_editor_error_response_is_structured_failure(self) -> None:
        with TemporaryDirectory() as directory:
            project = Path(directory)

            def handler(_project: Path, request: dict[str, Any]) -> None:
                _write_response(
                    request,
                    success=False,
                    error="UNITY_BRIDGE_DISPATCH_FAILED",
                    message="ImportAvatar failed",
                )

            editor = _SimulatedEditor(project, handler)
            editor.start()
            try:
                adapter = EditorBridgeMotionImportAdapter(
                    MotionImportConfig(unity_project_path=project, timeout_seconds=10.0)
                )
                with self.assertRaises(MotionImportError) as raised:
                    adapter.import_motion(
                        _import_request("phase23_bridge_error", project / "Walking.fbx")
                    )
            finally:
                editor.stop()
            self.assertEqual(raised.exception.reason, "UNITY_BRIDGE_DISPATCH_FAILED")

    def test_play_mode_error_response_surfaces_verbatim(self) -> None:
        with TemporaryDirectory() as directory:
            project = Path(directory)

            def handler(_project: Path, request: dict[str, Any]) -> None:
                _write_response(
                    request,
                    success=False,
                    error="UNITY_EDITOR_PLAY_MODE_ACTIVE",
                    message="Unity editor is in Play mode; stop Play mode before running motion import.",
                )

            editor = _SimulatedEditor(project, handler)
            editor.start()
            try:
                adapter = EditorBridgeMotionImportAdapter(
                    MotionImportConfig(unity_project_path=project, timeout_seconds=10.0)
                )
                with self.assertRaises(MotionImportError) as raised:
                    adapter.import_motion(
                        _import_request("phase23_bridge_play_mode", project / "Walking.fbx")
                    )
            finally:
                editor.stop()
            self.assertEqual(raised.exception.reason, "UNITY_EDITOR_PLAY_MODE_ACTIVE")
            self.assertIn("Play mode", str(raised.exception))

    def test_missing_report_after_dispatch_is_not_treated_as_success(self) -> None:
        with TemporaryDirectory() as directory:
            project = Path(directory)

            def handler(_project: Path, request: dict[str, Any]) -> None:
                _write_response(request)

            editor = _SimulatedEditor(project, handler)
            editor.start()
            try:
                adapter = EditorBridgeMotionImportAdapter(
                    MotionImportConfig(unity_project_path=project, timeout_seconds=0.5)
                )
                with self.assertRaises(MotionImportError) as raised:
                    adapter.import_motion(
                        _import_request("phase23_bridge_missing", project / "Walking.fbx")
                    )
            finally:
                editor.stop()
            self.assertEqual(raised.exception.reason, "UNITY_IMPORT_REPORT_MISSING")


class EditorBridgeProcessingAdapterTests(unittest.TestCase):
    def test_process_and_playback_use_editor_reports(self) -> None:
        with TemporaryDirectory() as directory:
            project = Path(directory)
            source_id = "phase23_bridge_source"
            processed_id = "phase23_bridge_source_processed_v1"

            def handler(_project: Path, request: dict[str, Any]) -> None:
                if request["operation"] == "process":
                    _write_report(
                        project,
                        "Temp/Phase17MotionProcessing.json",
                        _processing_report_payload(processed_id, source_id),
                    )
                elif request["operation"] == "playback":
                    _write_report(
                        project,
                        "Temp/Phase17ProductionPlayback.json",
                        _playback_report_payload(processed_id, source_id),
                    )
                else:  # pragma: no cover - guards the contract
                    raise AssertionError(f"unexpected operation {request['operation']}")
                _write_response(request)

            editor = _SimulatedEditor(project, handler)
            editor.start()
            try:
                adapter = EditorBridgeMotionProcessingAdapter(
                    MotionProcessingConfig(
                        unity_project_path=project,
                        timeout_seconds=10.0,
                        playback_timeout_seconds=10.0,
                    )
                )
                request = _processing_request(processed_id, source_id)
                source_motion = _source_motion(source_id)
                processing = adapter.process(request, source_motion)
                playback = adapter.validate_playback(request, source_motion, processing)
            finally:
                editor.stop()
            self.assertEqual(editor.errors, [])
            self.assertTrue(processing.success)
            self.assertEqual(processing.processed_motion_id, processed_id)
            self.assertEqual(processing.quality_status, "VALID")
            self.assertTrue(playback.success)
            self.assertEqual(playback.processed_motion_id, processed_id)
            self.assertTrue(playback.playback_success)
            self.assertTrue(editor.wait_for_operations(2))

    def test_playback_report_missing_is_structured_failure(self) -> None:
        with TemporaryDirectory() as directory:
            project = Path(directory)
            source_id = "phase23_bridge_source"
            processed_id = "phase23_bridge_source_processed_v1"

            def handler(_project: Path, request: dict[str, Any]) -> None:
                if request["operation"] == "process":
                    _write_report(
                        project,
                        "Temp/Phase17MotionProcessing.json",
                        _processing_report_payload(processed_id, source_id),
                    )
                _write_response(request)

            editor = _SimulatedEditor(project, handler)
            editor.start()
            try:
                adapter = EditorBridgeMotionProcessingAdapter(
                    MotionProcessingConfig(
                        unity_project_path=project,
                        timeout_seconds=10.0,
                        playback_timeout_seconds=0.5,
                    )
                )
                request = _processing_request(processed_id, source_id)
                source_motion = _source_motion(source_id)
                processing = adapter.process(request, source_motion)
                with self.assertRaises(MotionProcessingError) as raised:
                    adapter.validate_playback(request, source_motion, processing)
            finally:
                editor.stop()
            self.assertEqual(raised.exception.reason, "UNITY_PLAYBACK_REPORT_MISSING")


if __name__ == "__main__":
    unittest.main()
