from __future__ import annotations

import sqlite3
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from backend.brain.models import ActionType, Goal, GoalType
from backend.brain.planner import BrainPlanner
from backend.brain.world_state import build_world_state
from backend.learning import MotionValidationError, validate_motion_source
from backend.learning.models import (
    LearningConfig,
    MotionDefinition,
    MotionSource,
    MotionValidationStatus,
    SkeletonCompatibility,
)
from backend.learning.repository import SQLiteSkillLearningRepository
from backend.learning.service import SkillLearningService
from backend.skills.executor import SkillExecutor
from backend.skills.library import create_default_skill_library
from backend.skills.models import SkillEvaluation, SkillSelection, SkillStatus


PROJECT_ROOT = Path(__file__).resolve().parents[2]
UNITY_ASSET_ROOT = PROJECT_ROOT / "unity" / "3d_unity" / "Assets"
DANCE_SOURCE = UNITY_ASSET_ROOT / "animation" / "dance" / "dance.fbx"
DANCE_REFERENCE = "Assets/animation/dance"
DANCE_DURATION = 18.466667
DANCE_SAMPLE_RATE = 30.0
FIXED_NOW = datetime(2026, 1, 1)

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


class MotionValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._fixture_directories: list[tempfile.TemporaryDirectory] = []

    def test_real_imported_dance_metadata_is_extracted(self) -> None:
        metadata = validate_motion_source(
            DANCE_SOURCE,
            DANCE_REFERENCE,
            DANCE_DURATION,
            source_format="FBX",
            clip_name="Dance",
            sample_rate=DANCE_SAMPLE_RATE,
        )

        self.assertEqual(metadata.source_format, "FBX")
        self.assertEqual(metadata.clip_name, "Dance")
        self.assertEqual(metadata.frame_count, 555)
        self.assertEqual(metadata.sample_rate, DANCE_SAMPLE_RATE)
        self.assertAlmostEqual(metadata.duration_seconds, 18.466667, places=5)
        self.assertEqual(metadata.bone_count, 79)
        self.assertEqual(metadata.root_bone_name, "Hips")
        self.assertEqual(metadata.skeleton_status, SkeletonCompatibility.COMPATIBLE)

    def test_missing_source_is_rejected(self) -> None:
        with self.assertRaises(MotionValidationError) as context:
            validate_motion_source(
                UNITY_ASSET_ROOT / "does_not_exist.fbx",
                "Assets/does_not_exist",
                1.0,
            )
        self.assertEqual(context.exception.reason, "MOTION_SOURCE_MISSING")

    def test_unsupported_format_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir=UNITY_ASSET_ROOT) as directory:
            source = Path(directory) / "motion.bvh"
            source.write_bytes(b"BVH fixture")
            with self.assertRaises(MotionValidationError) as context:
                validate_motion_source(source, f"Assets/{Path(directory).name}", 1.0)
            self.assertEqual(context.exception.reason, "UNSUPPORTED_MOTION_FORMAT")

    def test_missing_clip_is_rejected(self) -> None:
        source, reference, _ = self._write_fixture(include_clip=False)
        with self.assertRaises(MotionValidationError) as context:
            validate_motion_source(source, reference, 29.0 / DANCE_SAMPLE_RATE)
        self.assertEqual(context.exception.reason, "MOTION_CLIP_MISSING")

    def test_missing_skeleton_is_rejected(self) -> None:
        source, reference, _ = self._write_fixture(include_skeleton=False)
        with self.assertRaises(MotionValidationError) as context:
            validate_motion_source(source, reference, 29.0 / DANCE_SAMPLE_RATE)
        self.assertEqual(context.exception.reason, "MISSING_SKELETON_INFORMATION")

    def test_invalid_reference_is_rejected(self) -> None:
        source, _, _ = self._write_fixture()
        with self.assertRaises(MotionValidationError) as context:
            validate_motion_source(source, "Assets/other-motion", 29.0 / DANCE_SAMPLE_RATE)
        self.assertEqual(context.exception.reason, "INVALID_ANIMATION_REFERENCE")

    def test_duration_sample_rate_mismatch_is_rejected(self) -> None:
        source, reference, _ = self._write_fixture()
        with self.assertRaises(MotionValidationError) as context:
            validate_motion_source(
                source,
                reference,
                8.0,
                clip_name="Dance",
                sample_rate=DANCE_SAMPLE_RATE,
            )
        self.assertEqual(context.exception.reason, "DURATION_SAMPLE_RATE_MISMATCH")

    def test_partial_humanoid_mapping_is_retargetable(self) -> None:
        source, reference, _ = self._write_fixture(
            humanoid_mappings=REQUIRED_HUMANOID_MAPPINGS[:8]
        )
        metadata = validate_motion_source(
            source,
            reference,
            29.0 / DANCE_SAMPLE_RATE,
            sample_rate=DANCE_SAMPLE_RATE,
        )
        self.assertEqual(metadata.skeleton_status, SkeletonCompatibility.RETARGETABLE)

    def test_non_humanoid_mapping_is_incompatible(self) -> None:
        source, reference, _ = self._write_fixture(animation_type=2)
        metadata = validate_motion_source(
            source,
            reference,
            29.0 / DANCE_SAMPLE_RATE,
            sample_rate=DANCE_SAMPLE_RATE,
        )
        self.assertEqual(metadata.skeleton_status, SkeletonCompatibility.INCOMPATIBLE)

    def test_sparse_humanoid_mapping_is_unknown(self) -> None:
        source, reference, _ = self._write_fixture(humanoid_mappings=("Hips",))
        metadata = validate_motion_source(
            source,
            reference,
            29.0 / DANCE_SAMPLE_RATE,
            sample_rate=DANCE_SAMPLE_RATE,
        )
        self.assertEqual(metadata.skeleton_status, SkeletonCompatibility.UNKNOWN)

    def _write_fixture(
        self,
        *,
        include_clip: bool = True,
        include_skeleton: bool = True,
        animation_type: int = 3,
        humanoid_mappings: tuple[str, ...] = REQUIRED_HUMANOID_MAPPINGS,
    ) -> tuple[Path, str, tempfile.TemporaryDirectory]:
        directory = tempfile.TemporaryDirectory(dir=UNITY_ASSET_ROOT)
        self._fixture_directories.append(directory)
        source = Path(directory.name) / "motion.fbx"
        source.write_bytes(b"FBX fixture")
        relative_source = source.relative_to(UNITY_ASSET_ROOT).as_posix()
        reference = "Assets/" + str(Path(relative_source).parent)

        lines = ["ModelImporter:"]
        if include_clip:
            lines.extend(
                [
                    "  animations:",
                    "    clipAnimations:",
                    "    - serializedVersion: 16",
                    "      name: Dance",
                    "      firstFrame: 0",
                    "      lastFrame: 29",
                    "    isReadable: 0",
                ]
            )
        if include_skeleton:
            lines.extend(["  humanDescription:", "    human:"])
            for mapping in humanoid_mappings:
                lines.extend(
                    [
                        f"    - boneName: {mapping}",
                        f"      humanName: {mapping}",
                    ]
                )
            lines.extend(["    skeleton:", "    - name: Hips", "      parentName: "])
        lines.extend(["  animationType: " + str(animation_type)])
        Path(f"{source}.meta").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return source, reference, directory

    def tearDown(self) -> None:
        for directory in self._fixture_directories:
            directory.cleanup()


class MotionAcquisitionServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_directory.name) / "learning.db"
        self.repository = SQLiteSkillLearningRepository(self.database_path)
        self.library = create_default_skill_library()
        self.service = SkillLearningService(
            self.library,
            LearningConfig(),
            self.repository,
        )

    def tearDown(self) -> None:
        self.repository.close()
        self.temp_directory.cleanup()

    def _register_acquired_dance(self):
        return self.service.register_motion(
            "acquired_dance",
            "Acquired Dance",
            "Validated imported humanoid dance motion.",
            DANCE_REFERENCE,
            DANCE_DURATION,
            tags=("dance", "social", "imported"),
            source_path=DANCE_SOURCE,
            source_format="FBX",
            clip_name="Dance",
            sample_rate=DANCE_SAMPLE_RATE,
        )

    def test_valid_motion_registers_as_executable_skill(self) -> None:
        motion, skill = self._register_acquired_dance()
        self.assertEqual(motion.validation_status, MotionValidationStatus.VALID)
        self.assertEqual(motion.skeleton_status, SkeletonCompatibility.COMPATIBLE)
        self.assertTrue(self.library.exists("acquired_dance"))
        self.assertEqual(skill.animation_reference, DANCE_REFERENCE)
        self.assertEqual(skill.estimated_duration_seconds, DANCE_DURATION)
        self.assertEqual(skill.steps[0].action, ActionType.DANCE)
        self.assertEqual(skill.steps[0].constant_target, DANCE_REFERENCE)

        executor = SkillExecutor(self.library, BrainPlanner(), lambda: FIXED_NOW)
        plan = executor.create_plan(
            SkillSelection("acquired_dance", {}, "motion acquisition test"),
            Goal("goal", GoalType.RESPOND_TO_VIEWER, 0.9, "validate motion"),
            build_world_state({}),
            now=FIXED_NOW,
        )
        self.assertEqual(plan.reason, "OK")
        self.assertEqual([step.action for step in plan.plan.steps], [ActionType.DANCE])
        self.assertEqual(plan.plan.steps[0].target, DANCE_REFERENCE)

    def test_duplicate_motion_is_rejected(self) -> None:
        self._register_acquired_dance()
        with self.assertRaises(ValueError) as context:
            self._register_acquired_dance()
        self.assertEqual(str(context.exception), "DUPLICATE_MOTION_OR_SKILL_ID")

    def test_retargetable_motion_is_not_registered_for_direct_execution(self) -> None:
        directory = tempfile.TemporaryDirectory(dir=UNITY_ASSET_ROOT)
        self.addCleanup(directory.cleanup)
        source = Path(directory.name) / "motion.fbx"
        source.write_bytes(b"FBX fixture")
        reference = "Assets/" + Path(directory.name).name
        lines = [
            "ModelImporter:",
            "  animations:",
            "    clipAnimations:",
            "    - serializedVersion: 16",
            "      name: Partial",
            "      firstFrame: 0",
            "      lastFrame: 29",
            "    isReadable: 0",
            "  humanDescription:",
            "    human:",
        ]
        for mapping in REQUIRED_HUMANOID_MAPPINGS[:8]:
            lines.extend([f"    - boneName: {mapping}", f"      humanName: {mapping}"])
        lines.extend(["    skeleton:", "    - name: Hips", "      parentName: ", "  animationType: 3"])
        Path(f"{source}.meta").write_text("\n".join(lines) + "\n", encoding="utf-8")

        with self.assertRaises(MotionValidationError) as context:
            self.service.register_motion(
                "retargetable_motion",
                "Retargetable Motion",
                "Partial humanoid mapping.",
                reference,
                29.0 / DANCE_SAMPLE_RATE,
                source_path=source,
                clip_name="Partial",
                sample_rate=DANCE_SAMPLE_RATE,
            )
        self.assertEqual(context.exception.reason, "MOTION_NOT_EXECUTABLE_WITHOUT_RETARGETING")
        self.assertFalse(self.library.exists("retargetable_motion"))

    def test_motion_metadata_persists_across_restart(self) -> None:
        motion, _ = self._register_acquired_dance()
        self.service.observe_evaluation(
            SkillEvaluation(
                skill_id=motion.motion_id,
                execution_id="acquired_execution",
                status=SkillStatus.SUCCESS,
                duration_seconds=DANCE_DURATION,
                steps_completed=1,
                steps_failed=0,
            ),
            context_key="REQUEST_ACTION|DANCE|RESPOND_TO_VIEWER|STANDING|INACTIVE|WORLD|0_0",
        )

        restarted_repository = SQLiteSkillLearningRepository(self.database_path)
        restarted = SkillLearningService(
            create_default_skill_library(),
            LearningConfig(),
            restarted_repository,
        )
        try:
            persisted = restarted.get_motion_state("acquired_dance")
            self.assertEqual(persisted["clip_name"], "Dance")
            self.assertEqual(persisted["frame_count"], 555)
            self.assertEqual(persisted["sample_rate"], DANCE_SAMPLE_RATE)
            self.assertEqual(persisted["bone_count"], 79)
            self.assertEqual(persisted["root_bone_name"], "Hips")
            self.assertEqual(persisted["skeleton_status"], "COMPATIBLE")
            self.assertEqual(persisted["validation_status"], "VALID")
            self.assertEqual(restarted.get_skill_state("acquired_dance")["contexts"][0]["trials"], 1)
        finally:
            restarted_repository.close()

    def test_disable_and_enable_preserves_learned_knowledge(self) -> None:
        motion, _ = self._register_acquired_dance()
        self.service.observe_evaluation(
            SkillEvaluation(
                skill_id=motion.motion_id,
                execution_id="acquired_execution",
                status=SkillStatus.SUCCESS,
                duration_seconds=DANCE_DURATION,
                steps_completed=1,
                steps_failed=0,
            ),
            context_key="motion-context",
        )

        disabled = self.service.disable_motion("acquired_dance")
        self.assertFalse(disabled.enabled)
        self.assertFalse(self.library.get("acquired_dance").enabled)
        self.assertEqual(self.service.get_skill_state("acquired_dance")["contexts"][0]["trials"], 1)

        enabled = self.service.enable_motion("acquired_dance")
        self.assertTrue(enabled.enabled)
        self.assertTrue(self.library.get("acquired_dance").enabled)
        self.assertEqual(self.service.get_skill_state("acquired_dance")["contexts"][0]["trials"], 1)

    def test_motion_diagnostics_include_acquisition_metadata(self) -> None:
        self._register_acquired_dance()
        state = self.service.get_state()
        motion = state["motions"][0]
        for key in (
            "source",
            "source_path",
            "source_format",
            "clip_name",
            "frame_count",
            "sample_rate",
            "bone_count",
            "root_bone_name",
            "skeleton_status",
            "validation_status",
            "enabled",
            "version",
        ):
            self.assertIn(key, motion)
        self.assertEqual(motion["validation_status"], "VALID")

    def test_legacy_motion_rows_load_as_pending_and_unknown(self) -> None:
        connection = sqlite3.connect(self.database_path)
        try:
            with connection:
                connection.execute("DROP TABLE motion_definitions")
                connection.execute(
                    "CREATE TABLE motion_definitions ("
                    "motion_id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT NOT NULL,"
                    "animation_reference TEXT NOT NULL, duration_seconds REAL NOT NULL, tags TEXT NOT NULL,"
                    "source TEXT NOT NULL, parameters TEXT NOT NULL, enabled INTEGER NOT NULL,"
                    "version INTEGER NOT NULL, execution_action TEXT NOT NULL DEFAULT 'DANCE')"
                )
                connection.execute(
                    "INSERT INTO motion_definitions VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        "legacy_dance",
                        "Legacy Dance",
                        "Phase 10 motion",
                        "Assets/animation/dance",
                        18.466667,
                        "[\"dance\"]",
                        "IMPORTED",
                        "[]",
                        1,
                        1,
                        "DANCE",
                    ),
                )
        finally:
            connection.close()

        repository = SQLiteSkillLearningRepository(self.database_path)
        try:
            motions = repository.load_motions()
            motion = motions["legacy_dance"]
            self.assertIsNone(motion.source_path)
            self.assertEqual(motion.skeleton_status, SkeletonCompatibility.UNKNOWN)
            self.assertEqual(motion.validation_status, MotionValidationStatus.PENDING)
        finally:
            repository.close()

    def test_non_executable_persisted_motion_loads_without_skill_registration(self) -> None:
        definition = MotionDefinition(
            motion_id="legacy_dance",
            name="Legacy Dance",
            description="Phase 10 motion",
            animation_reference="Assets/animation/dance",
            duration_seconds=DANCE_DURATION,
            tags=("dance",),
            source=MotionSource.IMPORTED,
            execution_action=ActionType.DANCE,
            enabled=False,
            skeleton_status=SkeletonCompatibility.UNKNOWN,
            validation_status=MotionValidationStatus.PENDING,
        )
        self.repository.save_motion(definition)

        restarted = SkillLearningService(
            create_default_skill_library(),
            LearningConfig(),
            SQLiteSkillLearningRepository(self.database_path),
        )
        try:
            self.assertFalse(self.library.exists("legacy_dance"))
            self.assertEqual(restarted.get_motion_state("legacy_dance")["validation_status"], "PENDING")
        finally:
            restarted.close()


if __name__ == "__main__":
    unittest.main()
