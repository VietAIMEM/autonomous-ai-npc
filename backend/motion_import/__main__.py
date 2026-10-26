"""Synchronous production entry point for motion-file imports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..learning.models import LearningConfig
from ..learning.repository import SQLiteSkillLearningRepository
from ..learning.service import SkillLearningService
from ..skills.library import create_default_skill_library
from ..skills.models import SkillCategory
from . import MotionFileImportService, MotionImportConfig, SQLiteMotionImportMetadataStore
from .unity import UnityEditorReportMotionImportAdapter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m backend.motion_import",
        description="Import one motion file through the Phase 16 pipeline.",
    )
    parser.add_argument("--source-file", required=True)
    parser.add_argument("--motion-id", required=True)
    parser.add_argument("--clip-name")
    parser.add_argument("--name")
    parser.add_argument("--description")
    parser.add_argument("--tags", default="", help="Comma-separated tags.")
    parser.add_argument(
        "--skill-category",
        choices=[category.value for category in SkillCategory],
        default=SkillCategory.MOVEMENT.value,
    )
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--database-path", default=None)
    parser.add_argument("--learning-database-path", default=None)
    parser.add_argument("--unity-executable", default=None)
    parser.add_argument("--unity-project-path", default=None)
    parser.add_argument(
        "--unity-report",
        default=None,
        help="Consume a Unity editor import report instead of launching batch-mode Unity.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    learning_database = Path(args.learning_database_path or LearningConfig().database_path)
    import_database = Path(args.database_path or MotionImportConfig().database_path)

    config_kwargs = {
        "database_path": import_database,
        "timeout_seconds": args.timeout_seconds,
    }
    if args.unity_executable:
        config_kwargs["unity_executable"] = Path(args.unity_executable)
    if args.unity_project_path:
        config_kwargs["unity_project_path"] = Path(args.unity_project_path)
    config = MotionImportConfig(**config_kwargs)

    repository: SQLiteSkillLearningRepository | None = None
    metadata_store: SQLiteMotionImportMetadataStore | None = None
    try:
        repository = SQLiteSkillLearningRepository(learning_database)
        learning = SkillLearningService(
            create_default_skill_library(),
            LearningConfig(database_path=learning_database),
            repository,
        )
        metadata_store = SQLiteMotionImportMetadataStore(import_database)
        adapter = (
            UnityEditorReportMotionImportAdapter(
                Path(args.unity_report),
                config.unity_project_path,
            )
            if args.unity_report
            else None
        )
        service = MotionFileImportService(
            learning,
            adapter,
            config=config,
            metadata_store=metadata_store,
            start_worker=False,
        )
        job = service.request_import(
            args.source_file,
            args.motion_id,
            name=args.name or "",
            description=args.description or "",
            clip_name=args.clip_name,
            tags=tuple(tag.strip() for tag in args.tags.split(",") if tag.strip()),
            skill_category=SkillCategory(args.skill_category),
        )
        service.run_pending_once()
        completed = service.get(job.request_id)
        payload = {
            "job": completed.to_dict() if completed else None,
            "health": service.health(),
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0 if completed is not None and completed.state.value != "FAILED" else 1
    finally:
        if metadata_store is not None:
            metadata_store.close()
        if repository is not None:
            repository.close()


if __name__ == "__main__":
    raise SystemExit(main())
