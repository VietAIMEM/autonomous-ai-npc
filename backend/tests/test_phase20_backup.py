"""PHASE 20 backup and restore tests.

The production contract is: create -> verify -> restore into a controlled
location -> re-validate. Backup must never silently overwrite live state, must
exclude temporary media, must survive a writer holding the database open, and
must never carry credentials.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from backend.runtime.backup import (
    BACKUP_PREFIX,
    MANIFEST_NAME,
    SNAPSHOT_NAME,
    BackupError,
    BackupSource,
    backup_status,
    create_backup,
    default_sources,
    integrity_check,
    prune_backups,
    read_manifest,
    restore_backup,
    verify_backup,
)
from backend.runtime.config import RuntimeConfig


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


def make_sqlite(path: Path, rows: int = 3, table: str = "episodes") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            f"CREATE TABLE IF NOT EXISTS {table} (id INTEGER PRIMARY KEY, note TEXT)"
        )
        connection.executemany(
            f"INSERT INTO {table} (note) VALUES (?)",
            [(f"row-{index}",) for index in range(rows)],
        )
        connection.commit()
    finally:
        connection.close()


def row_count(path: Path, table: str = "episodes") -> int:
    connection = sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True)
    try:
        return int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    finally:
        connection.close()


class BackupSourceTests(unittest.TestCase):
    def test_default_sources_covers_persistent_state_only(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            root = Path(workdir)
            config = make_config(root)
            make_sqlite(config.data_dir / "memory.db")
            make_sqlite(config.data_dir / "learning.db")
            make_sqlite(config.data_dir / "motion_import.db")
            make_sqlite(config.data_dir / "motion_processing.db")
            (config.data_dir / "generated_motion_metadata.json").write_text(
                "[]", encoding="utf-8"
            )
            # Temporary media must never be picked up as production state.
            config.temp_dir.mkdir(parents=True, exist_ok=True)
            (config.temp_dir / "speech-0001.wav").write_bytes(b"RIFFfake")

            names = [source.name for source in default_sources(config)]

            self.assertEqual(
                names,
                [
                    "memory.db",
                    "learning.db",
                    "motion_import.db",
                    "motion_processing.db",
                    "generated_motion_metadata.json",
                ],
            )

    def test_default_sources_skips_missing_files(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            config = make_config(Path(workdir))
            self.assertEqual(default_sources(config), [])


class BackupCreationTests(unittest.TestCase):
    def test_create_backup_is_verified_and_checksummed(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            root = Path(workdir)
            config = make_config(root)
            make_sqlite(config.data_dir / "memory.db", rows=4)

            result = create_backup(config)

            self.assertTrue(result.verified, result.errors)
            self.assertTrue(result.directory.is_dir())
            self.assertTrue(result.directory.name.startswith(BACKUP_PREFIX))
            manifest = read_manifest(result.directory)
            self.assertIsNotNone(manifest)
            names = {item["name"] for item in manifest["files"]}
            self.assertIn("memory.db", names)
            self.assertIn(SNAPSHOT_NAME, names)
            entry = next(
                item for item in manifest["files"] if item["name"] == "memory.db"
            )
            self.assertTrue(entry["integrity"]["ok"])
            self.assertEqual(len(entry["sha256"]), 64)
            payload = result.to_dict()
            self.assertEqual(payload["file_count"], len(result.files))
            self.assertGreater(payload["total_bytes"], 0)
            json.dumps(payload)  # diagnostics-safe: must be JSON encodable

    def test_create_backup_suffixes_colliding_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            root = Path(workdir)
            config = make_config(root)
            make_sqlite(config.data_dir / "memory.db")
            moment = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)

            first = create_backup(config, clock=lambda: moment)
            second = create_backup(config, clock=lambda: moment)

            self.assertNotEqual(first.directory, second.directory)
            self.assertTrue(second.directory.is_dir())
            self.assertIn("_2", second.directory.name)

    def test_create_backup_skips_missing_sources_without_failing(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            root = Path(workdir)
            config = make_config(root)
            sources = [BackupSource(config.data_dir / "memory.db", "sqlite")]

            result = create_backup(config, sources=sources)

            self.assertTrue(result.verified, result.errors)
            self.assertEqual(
                [item["name"] for item in result.files], [SNAPSHOT_NAME]
            )

    def test_create_backup_excludes_temp_media_and_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            root = Path(workdir)
            config = RuntimeConfig.from_environment(
                {
                    "ROOM_AGENT_ENV": "production",
                    "ROOM_AGENT_DATA_DIR": str(root / "data"),
                    "ROOM_AGENT_TEMP_DIR": str(root / "temp"),
                    "ROOM_AGENT_BACKUP_DIR": str(root / "backups"),
                    "ROOM_AGENT_STATE_DIR": str(root / "state"),
                    "TWITCH_OAUTH_TOKEN": "oauth:supersecretvalue",
                    "OPENROUTER_API_KEY": "sk-live-must-not-leak",
                }
            )
            make_sqlite(config.data_dir / "memory.db")
            config.temp_dir.mkdir(parents=True, exist_ok=True)
            (config.temp_dir / "motion-raw.mp4").write_bytes(b"\x00" * 32)

            result = create_backup(config)

            blob = ""
            for item in result.directory.iterdir():
                if item.is_file():
                    blob += item.read_text(encoding="utf-8", errors="ignore")
            self.assertNotIn("supersecretvalue", blob)
            self.assertNotIn("sk-live-must-not-leak", blob)
            self.assertNotIn("motion-raw.mp4", blob)
            self.assertFalse((result.directory / "motion-raw.mp4").exists())
            snapshot = json.loads(
                (result.directory / SNAPSHOT_NAME).read_text(encoding="utf-8")
            )
            self.assertEqual(snapshot["environment"], "production")
            self.assertIn("config", snapshot)
            self.assertNotIn("token", json.dumps(snapshot).lower())

    def test_create_backup_prunes_beyond_retention(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            root = Path(workdir)
            config = make_config(root)
            make_sqlite(config.data_dir / "memory.db")
            stamps = iter(
                [
                    datetime(2026, 1, 1, 0, 0, seconds, tzinfo=timezone.utc)
                    for seconds in range(30)
                ]
            )

            first = create_backup(config, clock=lambda: next(stamps), retention=2)
            second = create_backup(config, clock=lambda: next(stamps), retention=2)
            third = create_backup(config, clock=lambda: next(stamps), retention=2)

            self.assertEqual(third.pruned, [first.directory.name])
            self.assertFalse(first.directory.exists())
            self.assertTrue(second.directory.is_dir())
            self.assertTrue(third.directory.is_dir())


class IntegrityTests(unittest.TestCase):
    def test_integrity_check_accepts_a_healthy_database(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            database = Path(workdir) / "memory.db"
            make_sqlite(database, rows=10)
            report = integrity_check(database)
            self.assertTrue(report["ok"], report)

    def test_integrity_check_rejects_a_corrupt_file(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            database = Path(workdir) / "memory.db"
            database.write_bytes(b"this is not a sqlite database")
            report = integrity_check(database)
            self.assertFalse(report["ok"])
            self.assertTrue(report["errors"])

    def test_backup_of_a_live_write_ahead_log_database_is_consistent(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            root = Path(workdir)
            config = make_config(root)
            database = config.data_dir / "memory.db"
            database.parent.mkdir(parents=True, exist_ok=True)
            writer = sqlite3.connect(database)
            try:
                writer.execute("PRAGMA journal_mode=WAL")
                writer.execute(
                    "CREATE TABLE episodes (id INTEGER PRIMARY KEY, body TEXT)"
                )
                for index in range(25):
                    writer.execute(
                        "INSERT INTO episodes (body) VALUES (?)", (f"episode-{index}",)
                    )
                    writer.commit()

                result = create_backup(config)  # writer still open

                self.assertTrue(result.verified, result.errors)
                self.assertTrue((config.data_dir / "memory.db-wal").exists())
                restored = restore_backup(
                    config, result.directory, target_dir=root / "restored"
                )
                self.assertTrue(restored["ok"])
                self.assertEqual(row_count(root / "restored" / "memory.db"), 25)
            finally:
                writer.close()


class VerifyTests(unittest.TestCase):
    def _backup(self, root: Path) -> tuple[RuntimeConfig, Path]:
        config = make_config(root)
        make_sqlite(config.data_dir / "memory.db", rows=5)
        result = create_backup(config)
        self.assertTrue(result.verified, result.errors)
        return config, result.directory

    def test_verify_backup_passes_for_a_clean_backup(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            _config, directory = self._backup(Path(workdir))
            report = verify_backup(directory)
            self.assertTrue(report["ok"], report["errors"])
            self.assertTrue(report["manifest"])
            self.assertTrue(all(item["exists"] for item in report["files"]))

    def test_verify_backup_detects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            _config, directory = self._backup(Path(workdir))
            target = directory / "memory.db"
            with open(target, "ab") as stream:
                stream.write(b"tampered")

            report = verify_backup(directory)

            self.assertFalse(report["ok"])
            self.assertIn("memory.db:checksum_mismatch", report["errors"])

    def test_verify_backup_reports_a_missing_directory(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            report = verify_backup(Path(workdir) / "backup_missing")
            self.assertFalse(report["ok"])
            self.assertIn("backup_directory_missing", report["errors"])

    def test_verify_backup_reports_a_broken_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            _config, directory = self._backup(Path(workdir))
            (directory / MANIFEST_NAME).write_text("{not json", encoding="utf-8")
            report = verify_backup(directory)
            self.assertFalse(report["ok"])
            self.assertTrue(any(item.startswith("manifest") for item in report["errors"]))


class RestoreTests(unittest.TestCase):
    def _backup(self, root: Path, rows: int = 5) -> tuple[RuntimeConfig, Path]:
        config = make_config(root)
        make_sqlite(config.data_dir / "memory.db", rows=rows)
        (config.data_dir / "generated_motion_metadata.json").write_text(
            json.dumps([{"name": "wave"}]), encoding="utf-8"
        )
        result = create_backup(config)
        self.assertTrue(result.verified, result.errors)
        return config, result.directory

    def test_restore_into_a_fresh_directory_matches_checksums(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            root = Path(workdir)
            config, directory = self._backup(root)
            target = root / "recovered"

            report = restore_backup(config, directory, target_dir=target)

            self.assertTrue(report["ok"])
            self.assertFalse(report["live_target"])
            self.assertFalse(report["overwrote_live_state"])
            manifest = read_manifest(directory)
            expected = {
                item["name"]: item["sha256"]
                for item in manifest["files"]
                if item["name"] not in {MANIFEST_NAME, SNAPSHOT_NAME}
            }
            self.assertEqual(
                {item["name"]: item["sha256"] for item in report["restored"]}, expected
            )
            self.assertEqual(row_count(target / "memory.db"), 5)
            self.assertTrue((target / SNAPSHOT_NAME).is_file())
            recheck = verify_backup(directory)
            self.assertTrue(recheck["ok"])

    def test_restore_defaults_to_the_controlled_state_directory(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            root = Path(workdir)
            config, directory = self._backup(root)

            report = restore_backup(config, directory)

            target = Path(report["target"])
            self.assertTrue(target.is_dir())
            self.assertTrue(target.is_relative_to(config.state_dir))
            self.assertFalse(report["live_target"])

    def test_restore_refuses_to_overwrite_live_state_silently(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            root = Path(workdir)
            config, directory = self._backup(root)
            make_sqlite(config.data_dir / "memory.db", rows=9, table="episodes")

            with self.assertRaises(BackupError) as context:
                restore_backup(config, directory, target_dir=config.data_dir)

            self.assertEqual(str(context.exception), "refusing_to_overwrite_live_state")
            self.assertNotEqual(row_count(config.data_dir / "memory.db"), 5)

    def test_restore_over_live_state_takes_a_safety_backup(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            root = Path(workdir)
            config, directory = self._backup(root, rows=5)
            live = config.data_dir / "memory.db"
            make_sqlite(live, rows=8, table="episodes")
            self.assertEqual(row_count(live), 13)
            before = backup_status(config)["count"]

            report = restore_backup(
                config, directory, target_dir=config.data_dir, overwrite=True
            )

            self.assertTrue(report["overwrote_live_state"])
            self.assertTrue(report["safety_backup"])
            self.assertEqual(row_count(live), 5)
            status = backup_status(config)
            self.assertEqual(status["count"], before + 1)

    def test_restore_refuses_an_existing_target_file(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            root = Path(workdir)
            config, directory = self._backup(root)
            target = root / "occupied"
            target.mkdir(parents=True)
            (target / "memory.db").write_bytes(b"existing")

            with self.assertRaises(BackupError) as context:
                restore_backup(config, directory, target_dir=target)

            self.assertEqual(str(context.exception), "target_exists:memory.db")
            self.assertEqual((target / "memory.db").read_bytes(), b"existing")
            self.assertFalse((target / "memory.db.restore-tmp").exists())

    def test_restore_rejects_a_tampered_backup(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            root = Path(workdir)
            config, directory = self._backup(root)
            with open(directory / "memory.db", "ab") as stream:
                stream.write(b"tampered")

            with self.assertRaises(BackupError) as context:
                restore_backup(config, directory, target_dir=root / "recovered")

            self.assertEqual(str(context.exception), "backup_verification_failed")
            self.assertFalse((root / "recovered").exists())

    def test_restore_is_repeatable_into_distinct_locations(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            root = Path(workdir)
            config, directory = self._backup(root)
            first = restore_backup(config, directory, target_dir=root / "one")
            second = restore_backup(config, directory, target_dir=root / "two")
            self.assertEqual(
                {item["name"]: item["sha256"] for item in first["restored"]},
                {item["name"]: item["sha256"] for item in second["restored"]},
            )


class PruneAndStatusTests(unittest.TestCase):
    def test_prune_keeps_newest_and_only_removes_own_backups(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            root = Path(workdir)
            config = make_config(root)
            make_sqlite(config.data_dir / "memory.db")
            stamps = iter(
                [
                    datetime(2026, 2, 1, 0, 0, seconds, tzinfo=timezone.utc)
                    for seconds in range(40)
                ]
            )
            created = [
                create_backup(config, clock=lambda: next(stamps)).directory
                for _ in range(5)
            ]
            # Foreign directories must be left alone.
            (config.backup_dir / "backup_manual_notes").mkdir()
            (config.backup_dir / (BACKUP_PREFIX + "no-manifest")).mkdir()
            (config.backup_dir / "unrelated").mkdir()

            removed = prune_backups(config.backup_dir, keep=2)

            self.assertEqual(
                removed, [created[2].name, created[1].name, created[0].name]
            )
            self.assertTrue(created[3].is_dir())
            self.assertTrue(created[4].is_dir())
            self.assertTrue((config.backup_dir / "backup_manual_notes").is_dir())
            self.assertTrue(
                (config.backup_dir / (BACKUP_PREFIX + "no-manifest")).is_dir()
            )
            self.assertTrue((config.backup_dir / "unrelated").is_dir())

    def test_backup_status_reports_the_newest_backup(self) -> None:
        with tempfile.TemporaryDirectory() as workdir:
            root = Path(workdir)
            config = make_config(root, backup_retention=4)
            self.assertEqual(
                backup_status(config),
                {"directory": "backups", "count": 0, "newest": None},
            )
            make_sqlite(config.data_dir / "memory.db")
            first = create_backup(config)
            second = create_backup(config)

            status = backup_status(config)

            self.assertEqual(status["count"], 2)
            self.assertEqual(status["retention"], 4)
            self.assertGreater(status["total_bytes"], 0)
            self.assertIn(status["newest"]["name"], {first.directory.name, second.directory.name})
            self.assertTrue(status["newest"]["created_at"])


if __name__ == "__main__":  # pragma: no cover - unittest entry point
    unittest.main()
