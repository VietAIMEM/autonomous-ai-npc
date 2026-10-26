"""Command-line entry point for one controlled motion-processing run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..chat.models import utc_now
from ..learning.models import LearningConfig
from ..learning.repository import SQLiteSkillLearningRepository
from ..learning.service import SkillLearningService
from ..skills.library import create_default_skill_library
from .models import MotionProcessingConfig
from .persistence import SQLiteMotionProcessingMetadataStore
from .service import MotionProcessingService
from .unity import (
    MotionProcessingAdapter,
    UnityEditorReportMotionProcessingAdapter,
    UnityMotionProcessingAdapter,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Process one existing motion for Lacrimosa")
    parser.add_argument("source_motion_id")
    parser.add_argument("--processed-motion-id")
    parser.add_argument("--processing-version", type=int, default=1)
    parser.add_argument("--learning-db", default="backend/data/memory.db")
    parser.add_argument("--processing-db", default="backend/data/motion_processing.db")
    parser.add_argument(
        "--processing-report",
        help="Reuse a successful Unity processing report instead of running Unity again",
    )
    parser.add_argument(
        "--playback-report",
        help="Reuse a successful Unity playback report instead of running Unity again",
    )
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--json")
    args = parser.parse_args()
    if bool(args.processing_report) != bool(args.playback_report):
        parser.error("--processing-report and --playback-report must be provided together")

    learning_path = Path(args.learning_db)
    processing_path = Path(args.processing_db)
    repository = SQLiteSkillLearningRepository(learning_path)
    metadata_store = SQLiteMotionProcessingMetadataStore(processing_path)
    try:
        learning = SkillLearningService(
            create_default_skill_library(),
            LearningConfig(database_path=learning_path),
            repository,
            clock=utc_now,
        )
        config = MotionProcessingConfig(database_path=processing_path)
        adapter: MotionProcessingAdapter
        if args.processing_report and args.playback_report:
            adapter = UnityEditorReportMotionProcessingAdapter(
                args.processing_report,
                args.playback_report,
            )
        else:
            adapter = UnityMotionProcessingAdapter(config)
        service = MotionProcessingService(
            learning,
            adapter,
            config=config,
            metadata_store=metadata_store,
            clock=utc_now,
            start_worker=False,
        )
        job = service.request_processing(
            args.source_motion_id,
            processed_motion_id=args.processed_motion_id,
            processing_version=args.processing_version,
        )
        if job.state.value == "INPUT":
            service.run_pending_once(0.1)
        payload = {"request_id": job.request_id, "state": job.state.value, "error": job.error}
        if args.register and job.state.value == "READY":
            job = service.approve(job.request_id)
            job = service.register(job.request_id)
            payload["state"] = job.state.value
            payload["error"] = job.error
        if job.result is not None:
            payload["result"] = job.result.to_dict()
        output = json.dumps(payload, indent=2)
        if args.json:
            Path(args.json).write_text(output + "\n", encoding="utf-8")
        print(output)
        return 0 if job.state.value in {"READY", "REGISTERED"} else 1
    finally:
        metadata_store.close()
        repository.close()


if __name__ == "__main__":
    raise SystemExit(main())
