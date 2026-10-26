"""PHASE 20 data retention tests.

Retention is deliberately narrow: it bounds logs and temporary artifacts, and
never touches Memory, SkillKnowledge, learned compositions, viewer
relationships, databases or backups. These tests pin that boundary and the
age-first/count/bytes ordering that keeps a disk from filling up.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from backend.runtime.config import BACKEND_ROOT, PROJECT_ROOT, RuntimeConfig
from backend.runtime.retention import (
    RetentionError,
    apply_retention,
    guard_directory,
    log_usage,
)


NOW = 1_800_000_000.0


def make_config(directory: Path, **overrides) -> RuntimeConfig:
    values = {
        "environment": "test",
        "data_dir": directory / "data",
        "log_dir": directory / "logs",
        "temp_dir": directory / "temp",
        "backup_dir": directory / "backups",
        "state_dir": directory / "state",
    }
    values.update(overrides)
    return RuntimeConfig(**values)


def write_file(path: Path, size: int, mtime: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    os.utime(path, (mtime, mtime))
    return path


class GuardDirectoryTests(unittest.TestCase):
    def test_guard_refuses_the_filesystem_root(self) -> None:
        with self.assertRaises(RetentionError):
            guard_directory(Path(Path.cwd().anchor), label="temp_dir")

    def test_guard_refuses_the_project_and_backend_roots(self) -> None:
        for root, label in ((PROJECT_ROOT, "log_dir"), (BACKEND_ROOT, "temp_dir")):
            with self.subTest(root=str(root)):
                with self.assertRaises(RetentionError):
                    guard_directory(root, label=label)

    def test_guard_accepts_a_normal_directory(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            resolved = guard_directory(Path(workdir), label="temp_dir")
            self.assertTrue(resolved.is_absolute())
            self.assertEqual(resolved.name, Path(workdir).name)


class LogRetentionTests(unittest.TestCase):
    def test_old_logs_are_removed_by_age_and_fresh_logs_survive(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            config = make_config(Path(workdir), log_max_age_days=1.0)
            old = write_file(config.log_dir / "old.log", 16, NOW - 3 * 86400)
            fresh = write_file(config.log_dir / "fresh.log", 16, NOW - 60)

            report = apply_retention(config, now=NOW)

            self.assertEqual(report["removed"], ["old.log"])
            self.assertEqual(report["bytes_freed"], 16)
            self.assertFalse(old.exists())
            self.assertTrue(fresh.exists())

    def test_log_count_cap_keeps_the_newest_files(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            config = make_config(
                Path(workdir), log_max_age_days=0.0, log_max_files=3
            )
            files = [
                write_file(config.log_dir / f"run-{index}.log", 8, NOW - 100 + index)
                for index in range(6)
            ]

            report = apply_retention(config, now=NOW)

            self.assertEqual(report["removed_count"], 3)
            remaining = sorted(path.name for path in config.log_dir.iterdir())
            self.assertEqual(remaining, ["run-3.log", "run-4.log", "run-5.log"])
            self.assertFalse(files[0].exists())
            self.assertTrue(files[5].exists())

    def test_log_byte_cap_frees_the_oldest_files_first(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            config = make_config(
                Path(workdir),
                log_max_age_days=0.0,
                log_max_files=100,
                log_max_total_mb=0.001,  # 1048 bytes
            )
            files = [
                write_file(config.log_dir / f"chunk-{index}.log", 1000, NOW - 10 + index)
                for index in range(4)
            ]

            report = apply_retention(config, now=NOW)

            self.assertEqual(report["removed_count"], 3)
            self.assertEqual(report["bytes_freed"], 3000)
            self.assertFalse(files[0].exists())
            self.assertTrue(files[3].exists())
            self.assertLessEqual(
                sum(path.stat().st_size for path in config.log_dir.iterdir()), 1048
            )

    def test_only_log_marked_files_are_removed(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            config = make_config(Path(workdir), log_max_age_days=1.0)
            old_log = write_file(config.log_dir / "app.LOG", 32, NOW - 5 * 86400)
            events = write_file(config.log_dir / "events.jsonl", 32, NOW - 5 * 86400)
            marker = write_file(config.log_dir / "runtime.pid", 32, NOW - 5 * 86400)

            report = apply_retention(config, now=NOW)

            self.assertEqual(report["removed"], ["app.LOG"])
            self.assertFalse(old_log.exists())
            self.assertTrue(events.exists())
            self.assertTrue(marker.exists())

    def test_age_cap_of_zero_disables_age_based_deletion(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            config = make_config(
                Path(workdir), log_max_age_days=0.0, log_max_files=100
            )
            ancient = write_file(
                config.log_dir / "ancient.log", 16, NOW - 900 * 86400
            )

            report = apply_retention(config, now=NOW)

            self.assertEqual(report["removed_count"], 0)
            self.assertTrue(ancient.exists())

    def test_log_cleanup_can_be_skipped_entirely(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            config = make_config(Path(workdir), log_max_age_days=1.0)
            old = write_file(config.log_dir / "old.log", 16, NOW - 10 * 86400)

            report = apply_retention(config, include_logs=False, now=NOW)

            self.assertEqual(report["removed_count"], 0)
            self.assertTrue(old.exists())


class TempRetentionTests(unittest.TestCase):
    def test_temp_files_are_cleaned_recursively_by_age(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            config = make_config(Path(workdir), temp_max_age_hours=24.0)
            stale = write_file(
                config.temp_dir / "speech" / "clip-1.wav", 64, NOW - 4 * 86400
            )
            fresh = write_file(config.temp_dir / "clip-2.wav", 64, NOW - 60)

            report = apply_retention(config, now=NOW)

            self.assertEqual(report["removed"], ["clip-1.wav"])
            self.assertFalse(stale.exists())
            self.assertTrue(fresh.exists())

    def test_temp_count_cap_keeps_the_newest_files(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            config = make_config(
                Path(workdir), temp_max_age_hours=0.0, temp_max_files=2
            )
            files = [
                write_file(config.temp_dir / f"frame-{index}.bin", 10, NOW - 50 + index)
                for index in range(5)
            ]

            report = apply_retention(config, now=NOW)

            self.assertEqual(report["removed_count"], 3)
            self.assertFalse(files[0].exists())
            self.assertTrue(files[3].exists())
            self.assertTrue(files[4].exists())

    def test_temp_byte_cap_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            config = make_config(
                Path(workdir),
                temp_max_age_hours=0.0,
                temp_max_files=100,
                temp_max_total_mb=0.01,  # 10485 bytes
            )
            files = [
                write_file(config.temp_dir / f"audio-{index}.raw", 8000, NOW - 10 + index)
                for index in range(3)
            ]

            report = apply_retention(config, now=NOW)

            self.assertEqual(report["removed_count"], 2)
            self.assertFalse(files[0].exists())
            self.assertFalse(files[1].exists())
            self.assertTrue(files[2].exists())

    def test_empty_temp_subdirectories_are_pruned(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            config = make_config(Path(workdir), temp_max_age_hours=1.0)
            write_file(
                config.temp_dir / "a" / "b" / "c" / "artifact.bin",
                8,
                NOW - 10 * 3600,
            )
            keep_dir = config.temp_dir / "keep"
            write_file(keep_dir / "fresh.bin", 8, NOW - 60)

            apply_retention(config, now=NOW)

            self.assertFalse((config.temp_dir / "a").exists())
            self.assertTrue(keep_dir.is_dir())

    def test_temp_cleanup_can_be_skipped_entirely(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            config = make_config(Path(workdir), temp_max_age_hours=1.0)
            stale = write_file(config.temp_dir / "old.bin", 8, NOW - 48 * 3600)

            report = apply_retention(config, include_temp=False, now=NOW)

            self.assertEqual(report["removed_count"], 0)
            self.assertTrue(stale.exists())


class ScopeBoundaryTests(unittest.TestCase):
    def test_persistent_state_and_backups_are_never_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            config = make_config(
                Path(workdir),
                log_max_age_days=1.0,
                temp_max_age_hours=1.0,
            )
            old = NOW - 60 * 86400
            protected = [
                write_file(config.data_dir / "memory.db", 32, old),
                write_file(config.data_dir / "learning.db", 32, old),
                write_file(config.data_dir / "voices" / "narrator.wav", 32, old),
                write_file(config.state_dir / "room_agent.lock", 32, old),
                write_file(
                    config.backup_dir / "backup_20260101T000000Z" / "manifest.json",
                    32,
                    old,
                ),
            ]
            write_file(config.log_dir / "old.log", 16, old)
            write_file(config.temp_dir / "old.bin", 16, old)

            report = apply_retention(config, now=NOW)

            self.assertEqual(report["removed_count"], 2)
            for path in protected:
                self.assertTrue(path.exists(), path)

    def test_unsafe_log_directory_is_reported_not_touched(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            config = make_config(Path(workdir), log_dir=PROJECT_ROOT)

            report = apply_retention(config, now=NOW)

            self.assertEqual(report["removed_count"], 0)
            self.assertTrue(
                any("log_dir_is_project_root" in item for item in report["errors"]),
                report["errors"],
            )

    def test_missing_directories_are_a_noop(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            config = make_config(Path(workdir))
            report = apply_retention(config, now=NOW)
            self.assertEqual(report["removed_count"], 0)
            self.assertEqual(report["errors"], [])

    def test_result_payload_is_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            config = make_config(
                Path(workdir),
                log_max_age_days=0.0,
                log_max_files=1,
            )
            for index in range(60):
                write_file(
                    config.log_dir / f"burst-{index}.log",
                    8,
                    NOW - 5 * 86400 + index,
                )

            report = apply_retention(config, now=NOW)

            self.assertEqual(report["removed_count"], 59)
            self.assertEqual(len(report["removed"]), 50)

    def test_log_usage_probe_reports_sizes(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            config = make_config(Path(workdir))
            write_file(config.log_dir / "brain.log", 100, NOW)
            write_file(config.log_dir / "chat.log", 50, NOW)
            write_file(config.temp_dir / "nested" / "clip.wav", 25, NOW)

            usage = log_usage(config)

            self.assertEqual(usage["log"], {"files": 2, "bytes": 150})
            self.assertEqual(usage["temp"], {"files": 1, "bytes": 25})

    def test_log_usage_probe_tolerates_missing_directories(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            config = make_config(Path(workdir))
            usage = log_usage(config)
            self.assertEqual(usage["log"], {"files": 0, "bytes": 0})
            self.assertEqual(usage["temp"], {"files": 0, "bytes": 0})


if __name__ == "__main__":  # pragma: no cover - unittest entry point
    unittest.main()
