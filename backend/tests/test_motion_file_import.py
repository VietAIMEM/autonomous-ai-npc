from __future__ import annotations

import shutil
import sqlite3
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from backend.brain.models import ActionType, Goal, GoalType
from backend.brain.planner import BrainPlanner
from backend.brain.world_state import build_world_state
from backend.learning.models import LearningConfig, MotionSource, MotionValidationStatus, SkeletonCompatibility
from backend.learning.repository import SQLiteSkillLearningRepository
from backend.learning.service import SkillLearningService
from backend.motion_import import (
    MotionFileImportService,
    MotionImportConfig,
    MotionImportStatus,
    SQLiteMotionImportMetadataStore,
    UnityMotionImportReport,
)
from backend.motion_import.unity import MotionImportAdapter
from backend.skills.executor import SkillExecutor
from backend.skills.library import create_default_skill_library
from backend.skills.models import SkillCategory, SkillEvaluation, SkillSelection, SkillStatus


PROJECT_ROOT = Path(__file__).resolve().parents[2]
UNITY_ASSET_ROOT = PROJECT_ROOT / "unity" / "3d_unity" / "Assets"
PHASE16_SOURCE_ROOT = UNITY_ASSET_ROOT / "MotionImports" / "Phase16"
PHASE16_CLIP_ROOT = UNITY_ASSET_ROOT / "Resources" / "GeneratedMotions" / "Phase16"
FIXED_NOW = datetime(2026, 1, 1)
FPS = 30.0
DURATION = 29.0 / FPS

REQUIRED_HUMANOID_MAPPINGS = (
    "Hips",
    "LeftUpperLeg",
    "RightUpperLeg",
    "LeftLowerLeg",
    "RightLowerLeg",
    "LeftFoot",
    "RightFoot",
    "Spine",
    "Chest",
    "Neck",
    "Head",
    "LeftShoulder",
    "RightShoulder",
    "LeftUpperArm",
    "RightUpperArm",
    "LeftLowerArm",
    "RightLowerArm",
    "LeftHand",
    "RightHand",
)


class FakeMotionImportAdapter(MotionImportAdapter):
    def __init__(
        self,
        source_root: Path,
        clip_root: Path,
        *,
        skeleton_status: str = "COMPATIBLE",
        write_stable_clip: bool = True,
        report_overrides: dict | None = None,
    ) -> None:
        self.source_root = source_root
        self.clip_root = clip_root
        self.skeleton_status = skeleton_status
        self.write_stable_clip = write_stable_clip
        self.report_overrides = report_overrides or {}

    def import_motion(self, request) -> UnityMotionImportReport:
        stage_directory = self.source_root / request.actual_motion_id
        stage_directory.mkdir(parents=True, exist_ok=True)
        staged_source = stage_directory / f"{request.source_sha256}.fbx"
        shutil.copyfile(request.source_file, staged_source)
        mappings, animation_type = _fixture_mappings(self.skeleton_status)
        _write_motion_meta(staged_source, mappings, animation_type)

        stable_clip_path = ""
        if self.write_stable_clip and self.skeleton_status in {"COMPATIBLE", "RETARGETABLE"}:
            clip_directory = self.clip_root / request.actual_motion_id
            clip_directory.mkdir(parents=True, exist_ok=True)
            stable_clip_path = clip_directory / f"{request.source_sha256}.anim"
            stable_clip_path.write_bytes(b"persistent clip fixture")

        payload = {
            "success": True,
            "motionId": request.actual_motion_id,
            "version": request.version,
            "originalSourceFile": str(request.source_file),
            "stagedSourceFile": staged_source.as_posix(),
            "sourceSha256": request.source_sha256,
            "sourceFormat": request.source_format,
            "clipName": request.clip_name or "Imported",
            "durationSeconds": DURATION,
            "fps": FPS,
            "frameCount": 30,
            "curveCount": 1,
            "keyCount": 2,
            "finiteValues": True,
            "boneCount": max(1, len(mappings)),
            "rootBoneName": "Hips",
            "sourceAvatarValid": True,
            "sourceAvatarHuman": True,
            "sourceHumanoidMappings": list(mappings),
            "targetAvatarName": "LacrimosaAvatar",
            "targetAvatarValid": True,
            "targetAvatarHuman": True,
            "targetHumanoidMappings": list(REQUIRED_HUMANOID_MAPPINGS),
            "skeletonStatus": self.skeleton_status,
            "importStatus": "VALIDATED",
            "animationReference": "Assets/" + staged_source.relative_to(UNITY_ASSET_ROOT).as_posix(),
            "stableClipPath": (
                "Assets/" + stable_clip_path.relative_to(UNITY_ASSET_ROOT).as_posix()
                if stable_clip_path
                else ""
            ),
            "libraryPath": "Assets/Resources/GeneratedMotions/GeneratedMotionLibrary.asset",
            "metadata": {"import_adapter": "fake"},
            "importedAt": "2026-01-01T00:00:00Z",
        }
        payload.update(self.report_overrides)
        report = UnityMotionImportReport.from_dict(payload)
        return report


class MotionFileImportTests(unittest.TestCase):
    def setUp(self) -> None:
        PHASE16_SOURCE_ROOT.mkdir(parents=True, exist_ok=True)
        PHASE16_CLIP_ROOT.mkdir(parents=True, exist_ok=True)
        self.source_directory = tempfile.TemporaryDirectory()
        self.stage_directory = tempfile.TemporaryDirectory(prefix="test_", dir=PHASE16_SOURCE_ROOT)
        self.clip_directory = tempfile.TemporaryDirectory(prefix="test_", dir=PHASE16_CLIP_ROOT)
        self.source = Path(self.source_directory.name) / "motion.fbx"
        self.source.write_bytes(b"FBX fixture")

        self.temporary_directory = tempfile.TemporaryDirectory()
        root = Path(self.temporary_directory.name)
        self.learning_database = root / "learning.db"
        self.import_database = root / "motion_import.db"
        self.repository = SQLiteSkillLearningRepository(self.learning_database)
        self.library = create_default_skill_library()
        self.learning = SkillLearningService(
            self.library,
            LearningConfig(),
            self.repository,
            clock=lambda: FIXED_NOW,
        )
        self.metadata_store = SQLiteMotionImportMetadataStore(self.import_database)
        self.config = MotionImportConfig(database_path=self.import_database)
        self.service = self._service()

    def tearDown(self) -> None:
        self.metadata_store.close()
        self.repository.close()
        self.stage_directory.cleanup()
        self.clip_directory.cleanup()
        self.source_directory.cleanup()
        self.temporary_directory.cleanup()

    def _service(self, **adapter_kwargs) -> MotionFileImportService:
        adapter = FakeMotionImportAdapter(
            Path(self.stage_directory.name),
            Path(self.clip_directory.name),
            **adapter_kwargs,
        )
        return MotionFileImportService(
            self.learning,
            adapter,
            self.config,
            self.metadata_store,
            clock=lambda: FIXED_NOW,
            start_worker=False,
        )

    def _import(self, motion_id: str = "imported_motion", **kwargs):
        job = self.service.request_import(self.source, motion_id, **kwargs)
        self.assertEqual(job.state, MotionImportStatus.QUEUED)
        self.assertTrue(self.service.run_pending_once())
        return self.service.get(job.request_id)

    def test_supported_fbx_reaches_motion_skill_executor_and_learning(self) -> None:
        job = self._import(
            "imported_motion",
            name="Imported Motion",
            description="Valid imported motion.",
            clip_name="Imported",
            tags=("walk", "test"),
        )

        self.assertEqual(job.state, MotionImportStatus.REGISTERED)
        self.assertIsNone(job.error)
        result = job.result
        self.assertEqual(result.validation_status, MotionValidationStatus.VALID.value)
        self.assertEqual(result.skeleton_status, SkeletonCompatibility.COMPATIBLE.value)
        self.assertEqual(result.version, 1)
        self.assertEqual(result.frame_count, 30)
        self.assertEqual(result.fps, FPS)
        self.assertTrue(Path(result.staged_source_file).is_file())
        stable_clip = UNITY_ASSET_ROOT / result.stable_clip_path.removeprefix("Assets/")
        self.assertTrue(stable_clip.is_file())
        self.assertTrue(result.animation_reference.startswith("Assets/MotionImports/Phase16/"))

        motion = self.learning.get_motion("imported_motion")
        self.assertEqual(motion.source, MotionSource.IMPORTED)
        self.assertEqual(motion.source_format, "FBX")
        self.assertEqual(motion.clip_name, "Imported")
        self.assertEqual(motion.animation_reference, result.animation_reference)
        self.assertEqual(motion.skeleton_status, SkeletonCompatibility.COMPATIBLE)
        self.assertEqual(motion.validation_status, MotionValidationStatus.VALID)

        skill = self.library.get("imported_motion")
        self.assertEqual(skill.category, SkillCategory.MOVEMENT)
        self.assertEqual(skill.version, 1)
        self.assertEqual(skill.animation_reference, result.animation_reference)
        self.assertEqual(skill.steps[0].action, ActionType.DANCE)
        self.assertEqual(skill.steps[0].constant_target, result.animation_reference)

        executor = SkillExecutor(self.library, BrainPlanner(), lambda: FIXED_NOW)
        plan = executor.create_plan(
            SkillSelection("imported_motion", {}, "phase 16 test"),
            Goal("goal", GoalType.RESPOND_TO_VIEWER, 0.9, "test"),
            build_world_state({}),
            now=FIXED_NOW,
        )
        self.assertEqual(plan.reason, "OK")
        self.assertEqual(plan.plan.steps[0].action, ActionType.DANCE)
        self.assertEqual(plan.plan.steps[0].target, result.animation_reference)

        context = "REQUEST_ACTION|IMPORTED|RESPOND_TO_VIEWER|STANDING|ACTIVE|WORLD|0_0"
        self.learning.observe_evaluation(
            SkillEvaluation(
                skill_id="imported_motion",
                execution_id="imported_execution",
                status=SkillStatus.SUCCESS,
                duration_seconds=DURATION,
                steps_completed=1,
                steps_failed=0,
                viewer_response="positive",
            ),
            context_key=context,
        )
        knowledge = self.learning.get_skill_state("imported_motion")["contexts"][0]
        self.assertEqual(knowledge["trials"], 1)
        self.assertEqual(knowledge["successes"], 1)
        self.assertGreater(knowledge["average_score"], 0.7)
        self.assertIn(
            "imported_motion",
            [candidate["skill_id"] for candidate in self.learning.rank_candidates(context)],
        )

    def test_invalid_requests_are_rejected_without_skills(self) -> None:
        missing = Path(self.source_directory.name) / "missing.fbx"
        self.assertEqual(
            self.service.request_import(missing, "missing_motion").error,
            "MOTION_SOURCE_MISSING",
        )
        for extension in ("bvh", "glb", "gltf"):
            unsupported = Path(self.source_directory.name) / f"motion.{extension}"
            unsupported.write_bytes(b"motion fixture")
            self.assertEqual(
                self.service.request_import(unsupported, f"unsupported_{extension}").error,
                "UNSUPPORTED_MOTION_FORMAT",
            )
        empty = Path(self.source_directory.name) / "empty.fbx"
        empty.write_bytes(b"")
        self.assertEqual(
            self.service.request_import(empty, "empty_motion").error,
            "MOTION_SOURCE_EMPTY",
        )
        self.assertEqual(
            self.service.request_import(self.source, "invalid-id").error,
            "INVALID_MOTION_ID",
        )
        self.assertFalse(self.library.exists("missing_motion"))
        self.assertFalse(self.learning.get_motion("missing_motion"))

    def test_invalid_unity_reports_fail_closed(self) -> None:
        for overrides, expected in (
            ({"durationSeconds": 0.0}, "INVALID_MOTION_DURATION"),
            ({"fps": 0.0}, "INVALID_SAMPLE_RATE"),
            ({"frameCount": 0}, "INVALID_MOTION_FRAMES"),
            ({"curveCount": 0}, "MOTION_CURVES_MISSING"),
            ({"keyCount": 0}, "MOTION_KEYS_MISSING"),
            ({"finiteValues": False}, "NON_FINITE_MOTION_VALUES"),
            ({"sourceSha256": "0" * 64}, "SOURCE_HASH_MISMATCH"),
            ({"targetAvatarValid": False}, "INVALID_PRODUCTION_RIG"),
        ):
            with self.subTest(expected=expected):
                service = self._service(report_overrides=overrides)
                job = service.request_import(self.source, f"invalid_{expected.lower()}")
                service.run_pending_once()
                completed = service.get(job.request_id)
                self.assertEqual(completed.state, MotionImportStatus.FAILED)
                self.assertEqual(completed.error, expected)
                self.assertFalse(self.learning.get_motion(f"invalid_{expected.lower()}"))

    def test_non_executable_classifications_are_validated_but_not_registered(self) -> None:
        for status in ("RETARGETABLE", "UNKNOWN", "INCOMPATIBLE"):
            with self.subTest(status=status):
                service = self._service(skeleton_status=status)
                job = service.request_import(self.source, f"non_executable_{status.lower()}")
                service.run_pending_once()
                completed = service.get(job.request_id)
                self.assertEqual(completed.state, MotionImportStatus.VALIDATED)
                self.assertEqual(completed.result.skeleton_status, status)
                self.assertEqual(completed.result.validation_status, "VALID")
                self.assertFalse(self.learning.get_motion(f"non_executable_{status.lower()}"))
                self.assertFalse(self.library.exists(f"non_executable_{status.lower()}"))

    def test_missing_stable_clip_fails_compatible_import(self) -> None:
        service = self._service(write_stable_clip=False)
        job = service.request_import(self.source, "missing_clip")
        service.run_pending_once()
        completed = service.get(job.request_id)
        self.assertEqual(completed.state, MotionImportStatus.FAILED)
        self.assertEqual(completed.error, "STABLE_CLIP_MISSING")
        self.assertFalse(self.library.exists("missing_clip"))

    def test_repeated_imports_create_preserved_versions(self) -> None:
        first = self._import("versioned_motion")
        second = self._import("versioned_motion")

        self.assertEqual(first.result.version, 1)
        self.assertEqual(first.result.actual_motion_id, "versioned_motion")
        self.assertEqual(second.result.version, 2)
        self.assertEqual(second.result.actual_motion_id, "versioned_motion_v2")
        self.assertTrue(self.library.exists("versioned_motion"))
        self.assertTrue(self.library.exists("versioned_motion_v2"))
        self.assertIsNotNone(self.learning.get_motion("versioned_motion"))
        self.assertIsNotNone(self.learning.get_motion("versioned_motion_v2"))

        context = "REQUEST_ACTION|VERSIONED|RESPOND_TO_VIEWER|STANDING|ACTIVE|WORLD|0_0"
        for motion_id in ("versioned_motion", "versioned_motion_v2"):
            self.learning.observe_evaluation(
                SkillEvaluation(
                    skill_id=motion_id,
                    execution_id=f"{motion_id}_execution",
                    status=SkillStatus.SUCCESS,
                    duration_seconds=DURATION,
                    steps_completed=1,
                    steps_failed=0,
                ),
                context_key=context,
            )
        self.assertEqual(
            self.learning.get_skill_state("versioned_motion")["contexts"][0]["trials"], 1
        )
        self.assertEqual(
            self.learning.get_skill_state("versioned_motion_v2")["contexts"][0]["trials"], 1
        )

    def test_metadata_persistence_is_metadata_only(self) -> None:
        self._import("persisted_motion")
        connection = sqlite3.connect(self.import_database)
        try:
            connection.row_factory = sqlite3.Row
            columns = {
                row[1]: row[2].upper()
                for row in connection.execute("PRAGMA table_info(motion_import_records)")
            }
            self.assertNotIn("BLOB", set(columns.values()))
            self.assertIn("source_sha256", columns)
            self.assertIn("animation_reference", columns)
            self.assertIn("stable_clip_path", columns)
            record = connection.execute(
                "SELECT * FROM motion_import_records WHERE actual_versioned_motion_id = ?",
                ("persisted_motion",),
            ).fetchone()
            row = {
                name: record[name]
                for name in record.keys()
                if name in {
                    "import_status",
                    "validation_status",
                    "skeleton_status",
                    "version",
                }
            }
        finally:
            connection.close()
        self.assertEqual(row["import_status"], "REGISTERED")
        self.assertEqual(row["validation_status"], "VALID")
        self.assertEqual(row["skeleton_status"], "COMPATIBLE")
        self.assertEqual(row["version"], 1)

    def test_queue_is_bounded_and_single_concurrency(self) -> None:
        first = self.service.request_import(self.source, "queued_one")
        second = self.service.request_import(self.source, "queued_two")
        third = self.service.request_import(self.source, "queued_three")
        self.assertEqual(first.state, MotionImportStatus.QUEUED)
        self.assertEqual(second.state, MotionImportStatus.QUEUED)
        self.assertEqual(third.state, MotionImportStatus.FAILED)
        self.assertEqual(third.error, "IMPORT_QUEUE_FULL")
        self.assertEqual(self.service.config.max_concurrent_imports, 1)
        self.assertEqual(self.service.pending_count(), 2)
        self.service.run_pending_once()
        self.assertEqual(self.service.pending_count(), 1)
        self.assertEqual(self.service.health()["max_concurrent_imports"], 1)


def _fixture_mappings(status: str) -> tuple[tuple[str, ...], int]:
    if status == "COMPATIBLE":
        return REQUIRED_HUMANOID_MAPPINGS, 3
    if status == "RETARGETABLE":
        return REQUIRED_HUMANOID_MAPPINGS[:8], 3
    if status == "UNKNOWN":
        return ("Hips", "Spine", "Head"), 3
    return ("Hips", "Spine"), 2


def _write_motion_meta(source: Path, mappings: tuple[str, ...], animation_type: int) -> None:
    lines = [
        "ModelImporter:",
        "  animations:",
        "    clipAnimations:",
        "    - serializedVersion: 16",
        "      name: Imported",
        "      firstFrame: 0",
        "      lastFrame: 29",
        "    isReadable: 0",
        "  humanDescription:",
        "    human:",
    ]
    for mapping in mappings:
        lines.extend([f"    - boneName: {mapping}", f"      humanName: {mapping}"])
    lines.extend(["    skeleton:", "    - name: Hips", "      parentName: "])
    lines.append(f"  animationType: {animation_type}")
    Path(f"{source}.meta").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
