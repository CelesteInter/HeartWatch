"""Tests that the app's database path is read when a function runs, not
when the module is imported.

Plain-language overview: data/db.py reads HEARTWATCH_DB_PATH once, at
import, into db.DB_PATH. Functions used to take `path=DB_PATH` as a default
argument, which Python fixes when the function is defined -- so patching
db.DB_PATH in a test didn't reach them, and building the main window in a
test would quietly read the developer's real database (the Settings
developer panel lists its sessions on construction). Now every default is
None, meaning "db.DB_PATH, read at call time". These tests check each one
is redirected by a patch, and that building the whole main window on a
patched path leaves the real database's files byte-for-byte unchanged.

The real database is never opened: the guard only hashes its files.

    python -m unittest heartwatchapp.tests.test_db_path
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import contextlib  # noqa: E402
import hashlib  # noqa: E402
import tempfile  # noqa: E402
import unittest  # noqa: E402
from pathlib import Path  # noqa: E402
from unittest import mock  # noqa: E402

from PySide6.QtWidgets import QApplication  # noqa: E402

from ..data import db, stats  # noqa: E402
from ..data.writer import DatabaseWriter  # noqa: E402
from ..ui.main_window import HeartWatchApp  # noqa: E402
from ..ui.theme import Theme  # noqa: E402

REAL_DB_FILES = tuple(
    db.DEFAULT_DB_PATH.with_name(db.DEFAULT_DB_PATH.name + suffix) for suffix in ("", "-wal", "-shm")
)


def real_db_hashes() -> dict[str, str | None]:
    """{file name: sha256, or None if it doesn't exist} for the real
    database and its -wal / -shm files. Reads bytes; never opens SQLite."""
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        for path in REAL_DB_FILES
    }


class TempDbCase(unittest.TestCase):
    """A fresh database in a temporary folder, with db.DB_PATH pointed at it
    for the whole test."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "patched.db"
        patcher = mock.patch.object(db, "DB_PATH", self.path)
        patcher.start()
        self.addCleanup(patcher.stop)


class PathDefaultTests(TempDbCase):
    def add_session(self) -> int:
        with contextlib.closing(db._connect(self.path)) as conn, conn:
            sid = db.start_session(conn, "fixture", "Walking", 1.0, 1.0, started_at=1_000)
            db.insert_hr_batch(conn, sid, [(1_000 + 1_000 * i, 90, 0.9) for i in range(5)])
            db.end_session(conn, sid, 6_000)
        return sid

    def test_init_db_and_connect_use_the_patched_path(self) -> None:
        db.init_db()
        self.assertTrue(self.path.exists())
        with contextlib.closing(db._connect()) as conn:
            file = conn.execute("PRAGMA database_list").fetchone()[2]
        self.assertEqual(Path(file).resolve(), self.path.resolve())

    def test_get_db_info_uses_the_patched_path(self) -> None:
        db.init_db(self.path)
        self.add_session()
        info = db.get_db_info()
        self.assertEqual(info["path"], str(self.path))
        self.assertEqual(info["row_counts"]["sessions"], 1)
        self.assertEqual(info["row_counts"]["hr_samples"], 5)

    def test_get_session_timeline_uses_the_patched_path(self) -> None:
        db.init_db(self.path)
        sid = self.add_session()
        timeline = db.get_session_timeline(sid)
        self.assertEqual(timeline["session"]["label"], "Walking")
        self.assertEqual(len(timeline["hr"]), 5)

    def test_compute_session_stats_uses_the_patched_path(self) -> None:
        db.init_db(self.path)
        sid = self.add_session()
        facts = stats.compute_session_stats(sid)
        self.assertEqual(facts["hr_sample_count"], 5)
        self.assertEqual(facts["duration_s"], 5.0)

    def test_database_writer_uses_the_patched_path(self) -> None:
        db.init_db(self.path)
        writer = DatabaseWriter()
        writer.start()
        try:
            sid = writer.start_session("fixture", None, 1.0, 1.0)
            writer.end_session(sid)
        finally:
            writer.stop()
        self.assertEqual(db.get_db_info(self.path)["row_counts"]["sessions"], 1)


class RealDatabaseGuardTests(TempDbCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_main_window_on_a_patched_path_leaves_the_real_db_untouched(self) -> None:
        before = real_db_hashes()
        db.init_db()
        writer = DatabaseWriter()
        writer.start()
        try:
            window = HeartWatchApp(Theme(mode="dark"), writer)
            self.app.processEvents()
            window.shutdown()
            window.deleteLater()
            self.app.processEvents()
        finally:
            writer.stop()
        self.assertEqual(real_db_hashes(), before)


if __name__ == "__main__":
    unittest.main()
