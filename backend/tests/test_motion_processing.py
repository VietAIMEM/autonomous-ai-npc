from __future__ import annotations

import tempfile
import threading
import unittest
from datetime import datetime
from dataclasses import replace
from pathlib import Path

from backend.brain.models import ActionType
from backend.learning.models import (
    LearningConfig,
    MotionDefinition,
    MotionSource,
    MotionValidationStatus,
    SkeletonCompatibility,
)
from backend.learning.repository import SQLiteSkillLearningRepository
from backend.learning.service import SkillLearningService
from backend.motion_processing.models import (
    MotionPlaybackReport,
    MotionProcessingConfig,
    MotionProcessingError,
    MotionProcessingReport,
    MotionProcessingState,
    SkeletonMappingEntry,
)
from backend.motion_processing.persistence import SQLiteMotionProcessingMetadataStore
from backend.motion_processing.quality import (
    MotionFrame,
    QualityStatus,
    analyze_motion_samples,
    build_skeleton_mapping,
    compute_quality_score,
)
from backend.motion_processing.service import MotionProcessingService
from backend.motion_processing.unity import MotionProcessingAdapter
from backend.motion_processing.validation import (
    REQUIRED_HUMANOID_MAPPINGS,
    validate_processing_report,
)
from backend.skills.library import create_default_skill_library


FIXED_NOW = datetime(2026, 9, 26, 12, 0)
SOURCE_ID = "phase17_fixture_source"
PROCESSED_ID = "phase17_fixture_processed"
DURATION = 1.0
FPS = 60.0
LIBRARY_PATH = "Assets/Resources/GeneratedMotions/GeneratedMotionLibrary.asset"


class BlockingPlaybackAdapter(MotionProcessingAdapter):
    def __init__(
        self,
        clip_path: Path,
        *,
        release: threading.Event,
        playback_started: threading.Event,
    ) -> None:
        self.clip_path = clip_path
        self.release = release
        self.playback_started = playback_started

    def process(self, request, source_motion) -> MotionProcessingReport:
        return processing_report(
            request.source_motion_id,
            request.source_version,
            request.processed_motion_id,
            request.processing_version,
            source_motion.animation_reference,
            processed_clip_path(self.clip_path, request.processed_motion_id),
        )

    def validate_playback(self, request, source_motion, processing) -> MotionPlaybackReport:
        self.playback_started.set()
        self.release.wait(2.0)
        return playback_report(
            request.source_motion_id,
            request.processed_motion_id,
            request.processing_version,
            processing.animation_reference,
            processing.processed_clip_path,
        )


class ValidAdapter(MotionProcessingAdapter):
    def __init__(self, clip_path: Path) -> None:
        self.clip_path = clip_path

    def process(self, request, source_motion) -> MotionProcessingReport:
        return processing_report(
            request.source_motion_id,
            request.source_version,
            request.processed_motion_id,
            request.processing_version,
            source_motion.animation_reference,
            processed_clip_path(self.clip_path, request.processed_motion_id),
        )

    def validate_playback(self, request, source_motion, processing) -> MotionPlaybackReport:
        return playback_report(
            request.source_motion_id,
            request.processed_motion_id,
            request.processing_version,
            processing.animation_reference,
            processing.processed_clip_path,
        )


class MotionProcessingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        root = Path(self.temporary_directory.name)
        self.unity_project = root / "unity"
        self.clip_directory = (
            self.unity_project / "Assets" / "Resources" / "GeneratedMotions" / "Phase17"
        )
        self.clip_directory.mkdir(parents=True, exist_ok=True)

        self.learning_database = root / "learning.db"
        self.processing_database = root / "motion_processing.db"
        self.repository = SQLiteSkillLearningRepository(self.learning_database)
        self.library = create_default_skill_library()
        self.learning = SkillLearningService(
            self.library,
            LearningConfig(),
            self.repository,
            clock=lambda: FIXED_NOW,
        )
        self.metadata_store = SQLiteMotionProcessingMetadataStore(self.processing_database)
        self.config = MotionProcessingConfig(
            unity_project_path=self.unity_project,
            database_path=self.processing_database,
        )
        self.source = self._register_source()
        self.service = self._service()

    def tearDown(self) -> None:
        self.metadata_store.close()
        self.repository.close()
        self.temporary_directory.cleanup()

    def _register_source(self) -> MotionDefinition:
        motion = MotionDefinition(
            motion_id=SOURCE_ID,
            name="Phase 17 source fixture",
            description="Existing Phase 16-compatible motion used for processing tests.",
            animation_reference="Assets/MotionImports/Phase16/source/source.fbx",
            duration_seconds=DURATION,
            tags=("walking",),
            source=MotionSource.IMPORTED,
            execution_action=ActionType.DANCE,
            enabled=True,
            version=1,
            source_path="Assets/MotionImports/Phase16/source/source.fbx",
            source_format="FBX",
            clip_name="Walk",
            frame_count=62,
            sample_rate=FPS,
            bone_count=19,
            root_bone_name="mixamorig:Hips",
            skeleton_status=SkeletonCompatibility.COMPATIBLE,
            validation_status=MotionValidationStatus.VALID,
        )
        self.learning.register_validated_motion(motion)
        return motion

    def _service(self, adapter=None) -> MotionProcessingService:
        return MotionProcessingService(
            self.learning,
            adapter or ValidAdapter(self.clip_directory),
            self.config,
            self.metadata_store,
            clock=lambda: FIXED_NOW,
            start_worker=False,
        )

    def _request(self) -> None:
        self.job = self.service.request_processing(
            SOURCE_ID,
            processed_motion_id=PROCESSED_ID,
            name="Processed walking fixture",
            description="Processed version of the Phase 17 fixture.",
            tags=("walking",),
        )
        self.assertEqual(self.job.state, MotionProcessingState.INPUT)
        self.assertTrue(self.service.run_pending_once())

    def test_quality_analysis_separates_warnings_from_hard_failures(self) -> None:
        frames = [
            MotionFrame(
                index / FPS,
                {"Hips": (0, 0.9, index * 0.01), "LeftFoot": (0, 0, 0), "RightFoot": (0, 0.1, 0)},
                {"Hips": (0, 0, 0, 1)},
            )
            for index in range(30)
        ]
        valid = analyze_motion_samples(frames, FPS)
        self.assertEqual(valid.status, QualityStatus.VALID)
        self.assertEqual(valid.hard_failures, ())

        rejected = analyze_motion_samples(
            [MotionFrame(0, {"Hips": (0, 0, 0)}, {"Hips": (0, 0, 0, 2)})],
            FPS,
        )
        self.assertEqual(rejected.status, QualityStatus.REJECTED)
        self.assertIn("INVALID_QUATERNION", rejected.hard_failures)
        self.assertGreaterEqual(compute_quality_score(), 0.0)
        self.assertLessEqual(compute_quality_score(), 1.0)

    def test_skeleton_mapping_reports_missing_and_ambiguous_joints(self) -> None:
        mappings, missing, ambiguous = build_skeleton_mapping(
            {"Hips": "SourceHips", "Spine": "SourceSpine"},
            [("Hips", "TargetHips"), ("Hips", "TargetHipsAlternate"), ("Spine", "TargetSpine")],
        )
        self.assertIn("Hips", ambiguous)
        self.assertIn("Head", missing)
        self.assertNotIn("Hips", mappings)
        self.assertIn("Spine", mappings)

        complete, no_missing, no_ambiguous = build_skeleton_mapping(
            {human: "Source" + human for human in REQUIRED_HUMANOID_MAPPINGS},
            {human: "Target" + human for human in REQUIRED_HUMANOID_MAPPINGS},
        )
        self.assertEqual(set(complete), set(REQUIRED_HUMANOID_MAPPINGS))
        self.assertEqual(no_missing, ())
        self.assertEqual(no_ambiguous, ())

    def test_processed_motion_requires_approval_and_preserves_source(self) -> None:
        self._request()
        completed = self.service.get(self.job.request_id)
        self.assertEqual(completed.state, MotionProcessingState.READY)
        self.assertIsNone(completed.error)
        self.assertFalse(self.library.exists(PROCESSED_ID))
        self.assertEqual(
            self.learning.get_motion(SOURCE_ID).animation_reference,
            self.source.animation_reference,
        )

        with self.assertRaises(MotionProcessingError):
            self.service.register(self.job.request_id)

        completed.result.validation_status = "PENDING"
        with self.assertRaises(MotionProcessingError) as raised:
            self.service.approve(self.job.request_id)
        self.assertEqual(raised.exception.reason, "MOTION_NOT_VALID")
        self.assertEqual(self.service.get(self.job.request_id).state, MotionProcessingState.READY)
        completed.result.validation_status = "VALID"

        approved = self.service.approve(self.job.request_id)
        self.assertEqual(approved.state, MotionProcessingState.APPROVED)
        self.assertFalse(self.library.exists(PROCESSED_ID))

        registered = self.service.register(self.job.request_id)
        self.assertEqual(
            registered.state,
            MotionProcessingState.REGISTERED,
            registered.error,
        )
        processed = self.learning.get_motion(PROCESSED_ID)
        self.assertEqual(processed.source_motion_id, SOURCE_ID)
        self.assertEqual(processed.source_version, 1)
        self.assertEqual(processed.processing_version, 1)
        self.assertEqual(processed.validation_status, MotionValidationStatus.VALID)
        self.assertEqual(processed.skeleton_status, SkeletonCompatibility.COMPATIBLE)
        self.assertTrue(processed.enabled)
        self.assertEqual(self.library.get(PROCESSED_ID).steps[0].action, ActionType.DANCE)

        self.assertNotEqual(
            self.learning.get_motion(SOURCE_ID).animation_reference,
            processed.animation_reference,
        )
        duplicate = self.service.request_processing(SOURCE_ID, processed_motion_id=PROCESSED_ID)
        self.assertEqual(duplicate.state, MotionProcessingState.REJECTED)
        self.assertEqual(duplicate.error, "DUPLICATE_PROCESSED_MOTION")

    def test_intermediate_processing_result_is_persisted_before_playback(self) -> None:
        release = threading.Event()
        playback_started = threading.Event()
        service = self._service(
            BlockingPlaybackAdapter(self.clip_directory, release=release, playback_started=playback_started)
        )
        job = service.request_processing(
            SOURCE_ID,
            processed_motion_id=PROCESSED_ID + "_blocked",
        )
        worker = threading.Thread(target=service.run_pending_once, daemon=True)
        worker.start()
        self.assertTrue(playback_started.wait(2.0))
        record = self.metadata_store.get(job.request_id)
        self.assertIsNotNone(record)
        self.assertEqual(record["state"], MotionProcessingState.VALIDATING.value)
        self.assertEqual(record["validation_status"], "PENDING")
        self.assertTrue(record["metadata"]["processing_report"])
        self.assertFalse(self.library.exists(PROCESSED_ID + "_blocked"))

        release.set()
        worker.join(2.0)
        completed = service.get(job.request_id)
        self.assertEqual(completed.state, MotionProcessingState.READY)
        self.assertEqual(
            self.metadata_store.get(job.request_id)["state"],
            MotionProcessingState.READY.value,
        )

    def test_high_quality_score_cannot_override_incompatible_skeleton(self) -> None:
        request = self.service.request_processing(
            SOURCE_ID,
            processed_motion_id=PROCESSED_ID + "_incompatible",
        )
        report = processing_report(
            request.request.source_motion_id,
            request.request.source_version,
            request.request.processed_motion_id,
            request.request.processing_version,
            self.source.animation_reference,
            processed_clip_path(self.clip_directory, request.request.processed_motion_id),
            skeleton_status="INCOMPATIBLE",
            quality_score=0.95,
        )
        with self.assertRaises(MotionProcessingError) as raised:
            validate_processing_report(
                report,
                request.request,
                self.source,
                unity_project_path=self.unity_project,
            )
        self.assertEqual(raised.exception.reason, "MOTION_NOT_EXECUTABLE_WITHOUT_RETARGETING")
        self.assertFalse(self.library.exists(PROCESSED_ID + "_incompatible"))

    def test_validation_rejects_unknown_production_sample_statuses(self) -> None:
        request = self.service.request_processing(
            SOURCE_ID,
            processed_motion_id=PROCESSED_ID + "_statuses",
        )
        report = processing_report(
            request.request.source_motion_id,
            request.request.source_version,
            request.request.processed_motion_id,
            request.request.processing_version,
            self.source.animation_reference,
            processed_clip_path(self.clip_directory, request.request.processed_motion_id),
        )

        for field, value, reason in (
            ("foot_status", "UNKNOWN", "FOOT_VALIDATION_UNAVAILABLE"),
            ("joint_status", "UNKNOWN", "JOINT_VALIDATION_UNAVAILABLE"),
            ("root_status", "UNKNOWN", "ROOT_VALIDATION_UNAVAILABLE"),
        ):
            with self.subTest(field=field):
                with self.assertRaises(MotionProcessingError) as raised:
                    validate_processing_report(
                        replace(report, **{field: value}),
                        request.request,
                        self.source,
                        unity_project_path=self.unity_project,
                    )
                self.assertEqual(raised.exception.reason, reason)

    def test_validation_rejects_foot_status_metric_mismatch(self) -> None:
        request = self.service.request_processing(
            SOURCE_ID,
            processed_motion_id=PROCESSED_ID + "_foot_metrics",
        )
        report = processing_report(
            request.request.source_motion_id,
            request.request.source_version,
            request.request.processed_motion_id,
            request.request.processing_version,
            self.source.animation_reference,
            processed_clip_path(self.clip_directory, request.request.processed_motion_id),
        )

        with self.assertRaises(MotionProcessingError) as raised:
            validate_processing_report(
                replace(report, foot_status="VALID", foot_penetration_depth=0.10),
                request.request,
                self.source,
                unity_project_path=self.unity_project,
            )
        self.assertEqual(raised.exception.reason, "FOOT_STATUS_DISAGREES_WITH_METRICS")

    def test_metadata_persistence_and_bounded_concurrency(self) -> None:
        self._request()
        record = self.metadata_store.get(self.job.request_id)
        self.assertEqual(record["processed_motion_id"], PROCESSED_ID)
        self.assertEqual(record["processing_version"], 1)
        self.assertEqual(record["skeleton_status"], SkeletonCompatibility.COMPATIBLE.value)
        self.assertEqual(record["validation_status"], "VALID")
        self.assertIn("processing_report", record["metadata"])
        self.assertIn("playback_report", record["metadata"])
        self.assertEqual(self.config.max_concurrent_processes, 1)
        self.assertEqual(self.config.max_queue_depth, 2)
        self.assertEqual(self.service.health()["status"], "ok")


def processed_clip_path(directory: Path, motion_id: str) -> str:
    path = directory / f"{motion_id}.anim"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"persistent processed clip fixture")
    return "Assets/Resources/GeneratedMotions/Phase17/" + path.name


def processing_report(
    source_id: str,
    source_version: int,
    processed_id: str,
    processing_version: int,
    source_reference: str,
    clip_path: str,
    *,
    skeleton_status: str = "COMPATIBLE",
    quality_score: float = 0.96,
) -> MotionProcessingReport:
    return MotionProcessingReport(
        success=True,
        source_motion_id=source_id,
        source_version=source_version,
        processed_motion_id=processed_id,
        processing_version=processing_version,
        state="READY",
        source_animation_reference=source_reference,
        animation_reference=clip_path,
        processed_clip_path=clip_path,
        library_path=LIBRARY_PATH,
        duration_seconds=DURATION,
        fps=FPS,
        frame_count=62,
        curve_count=69,
        key_count=4278,
        bone_count=19,
        root_bone_name="mixamorig:Hips",
        quality_status="VALID",
        quality_score=quality_score,
        quality_flags=(),
        hard_failures=(),
        warnings=(),
        finite_values=True,
        quaternion_valid=True,
        quaternion_curves_found=True,
        key_times_monotonic=True,
        missing_sample_count=0,
        missing_sample_ratio=0.0,
        max_curve_jump=0.08,
        root_status="STABLE",
        root_horizontal_distance=0.2,
        root_vertical_distance=0.04,
        root_authoritative_navigation=True,
        apply_root_motion=False,
        foot_status="VALID",
        foot_confidence=0.82,
        foot_sliding_distance=0.04,
        foot_penetration_depth=0.01,
        foot_flags=(),
        joint_status="VALID",
        joint_flags=(),
        source_avatar_valid=True,
        source_avatar_human=True,
        target_avatar_valid=True,
        target_avatar_human=True,
        source_humanoid_mappings=tuple(REQUIRED_HUMANOID_MAPPINGS),
        target_humanoid_mappings=tuple(REQUIRED_HUMANOID_MAPPINGS),
        mapping_entries=tuple(
            SkeletonMappingEntry(human, "Source" + human, "Target" + human)
            for human in REQUIRED_HUMANOID_MAPPINGS
        ),
        missing_mappings=(),
        ambiguous_mappings=(),
        skeleton_status=skeleton_status,
        retarget_status="MAPPED",
        cleanup_status="NOT_REQUIRED",
        cleanup_magnitude=0.0,
        cleanup_key_count=0,
        processing_timestamp="2026-09-26T12:00:00Z",
    )


def playback_report(
    source_id: str,
    processed_id: str,
    processing_version: int,
    animation_reference: str,
    processed_clip_path: str,
) -> MotionPlaybackReport:
    return MotionPlaybackReport(
        success=True,
        source_motion_id=source_id,
        processed_motion_id=processed_id,
        processing_version=processing_version,
        animation_reference=animation_reference,
        processed_clip_path=processed_clip_path,
        avatar_valid=True,
        avatar_human=True,
        clip_persistent=True,
        clip_duration_seconds=DURATION,
        clip_frame_rate=FPS,
        curve_binding_count=69,
        library_resolved=True,
        playback_success=True,
        processed_clip_active_at_all_samples=True,
        dance_state_active_at_all_samples=True,
        finite_samples=True,
        nontrivial_movement=True,
        max_bone_displacement=0.12,
        generated_clip_cleared=True,
        original_controller_restored=True,
        idle_before=True,
        idle_after=True,
        navigation_success=True,
        navigation_displacement=0.2,
        apply_root_motion_disabled=True,
        built_in_animation_substituted=False,
        required_controller_states_present=True,
        controller_states=("Idle", "Walk", "Run", "Wave", "Dance", "LeftTurn", "RightTurn", "LookingBehind"),
    )


if __name__ == "__main__":
    unittest.main()
