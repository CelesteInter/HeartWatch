"""Tests that scripts/demo_session.py never touches the real database.

Plain-language overview: the demo script builds a database to try the
summaries on. These tests check, byte for byte, that the real database's
files (the .db and its -wal / -shm companions) come out of a demo run
exactly as they went in -- in the default mode (a brand-new demo database)
and in the opt-in --from-real mode (a file-level copy).

Every test builds its own FAKE "real" database in a temporary directory and
points HEARTWATCH_DB_PATH at it, so no test ever opens the developer's
actual database. Stdlib unittest, like the other test files:

    python -m unittest heartwatchapp.tests.test_demo_db_fresh
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ..config import DB_PATH_ENV_VAR
from ..data import db, stats
from ..scripts import demo_db, demo_session, eval_summaries

# Where this came from: captured by running the pre-change demo_db.py
# (build_demo_db(..., copy_real=False, data_seed=0)) on October 6, 2026,
# before this handoff's edits. Timestamps are stored relative to the
# session's start, since the start time is "now". Hashes the session row
# (device, label, scales, duration), every heart-rate sample and every
# prediction.
DEMO_SEED0_HR_ROWS = 340
DEMO_SEED0_PREDICTION_ROWS = 68
DEMO_SEED0_FINGERPRINT = "4b3adafb4e7f510ac6e698a98e0a6f0da9dfaf247060125a0ff112031c51706a"

SIDECARS = ("-wal", "-shm", "-journal")


def snapshot(directory: Path) -> dict[str, tuple[str, int, int]]:
    """{file name: (sha256, st_mtime_ns, st_size)} for every file in
    `directory`, so a newly created sidecar file changes the result too."""
    result = {}
    for path in sorted(directory.iterdir()):
        if path.is_file():
            stat = path.stat()
            result[path.name] = (
                hashlib.sha256(path.read_bytes()).hexdigest(),
                stat.st_mtime_ns,
                stat.st_size,
            )
    return result


def fingerprint(path: Path, session_id: int) -> tuple[int, int, str]:
    """(hr rows, prediction rows, sha256) of one demo session's content."""
    with contextlib.closing(sqlite3.connect(str(path))) as conn:
        session = conn.execute(
            "SELECT device_id, label, accel_scale, gyro_scale, ended_at - started_at, "
            "started_at FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        t0 = session[5]
        hr = conn.execute(
            "SELECT ts - ?, bpm, confidence FROM hr_samples WHERE session_id = ? ORDER BY ts",
            (t0, session_id),
        ).fetchall()
        preds = conn.execute(
            "SELECT ts - ?, predicted, confidence, model_version FROM predictions "
            "WHERE session_id = ? ORDER BY ts",
            (t0, session_id),
        ).fetchall()
    payload = json.dumps([list(session[:5]), [list(r) for r in hr], [list(r) for r in preds]])
    return len(hr), len(preds), hashlib.sha256(payload.encode()).hexdigest()


class FakeRealDbCase(unittest.TestCase):
    """Builds a fake "real" database with a live WAL writer, plus a
    separate directory for demo output, and runs demo_session.main()."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.real_dir = self.root / "real"
        self.out_dir = self.root / "demo"
        self.real_dir.mkdir()
        self.out_dir.mkdir()
        self.real = self.real_dir / "heartwatch.db"

        db.init_db(self.real)  # opens and closes: everything checkpointed
        # Live-WAL fixture: keep a writer open with auto-checkpointing off,
        # so -wal and -shm exist and the rows below live only in the WAL --
        # like a real database the app has recently used.
        self.writer = db._connect(self.real)
        self.addCleanup(self.writer.close)
        self.writer.execute("PRAGMA wal_autocheckpoint = 0")
        session_id = db.start_session(self.writer, "real-device", "Walking", 1.0, 1.0, 1_000)
        db.insert_hr_batch(self.writer, session_id, [(1_000, 70, 0.9), (2_000, 71, 0.9)])
        db.end_session(self.writer, session_id, 3_000)
        self.writer.commit()
        self.assertTrue((self.real_dir / "heartwatch.db-wal").stat().st_size > 0)
        self.assertTrue((self.real_dir / "heartwatch.db-shm").exists())

    def run_main(self, *args: str, dest: Path | str | None = None) -> tuple[int, str, str]:
        """demo_session.main(--db dest ... --no-launch) against the fake
        real database. Returns (exit code, stdout, stderr). main() points
        HEARTWATCH_DB_PATH at the destination; patch.dict undoes that."""
        dest = self.out_dir / "demo.db" if dest is None else dest
        argv = ["--db", str(dest), "--no-launch", *args]
        resolved = Path(dest).expanduser().resolve()
        stdout, stderr = io.StringIO(), io.StringIO()
        env = {DB_PATH_ENV_VAR: str(self.real)}
        with (
            mock.patch.dict(os.environ, env),
            # This process imported data.db long ago, so main()'s "did the
            # app pick up the new path" check needs this stand-in.
            mock.patch.object(db, "DB_PATH", resolved),
            # The pre-change copy step read this constant, not the env var.
            mock.patch.object(demo_db, "REAL_DB_PATH", self.real),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            code = demo_session.main(argv)
        return code, stdout.getvalue(), stderr.getvalue()

    def session_count(self, path: Path) -> int:
        with contextlib.closing(sqlite3.connect(str(path))) as conn:
            return conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]

    def integrity(self, path: Path) -> str:
        with contextlib.closing(sqlite3.connect(str(path))) as conn:
            return conn.execute("PRAGMA integrity_check").fetchone()[0]


class DefaultModeTests(FakeRealDbCase):
    def test_a_default_mode_leaves_the_real_db_untouched(self) -> None:
        before = snapshot(self.real_dir)
        code, _, err = self.run_main()
        self.assertEqual(code, 0, err)
        self.assertEqual(snapshot(self.real_dir), before)

    def test_default_mode_makes_a_database_with_only_the_demo_session(self) -> None:
        code, out, err = self.run_main()
        self.assertEqual(code, 0, err)
        self.assertEqual(self.session_count(self.out_dir / "demo.db"), 1)
        self.assertIn("Demo session #1", out)

    def test_e_stale_sidecar_files_are_cleaned(self) -> None:
        dest = self.out_dir / "demo.db"
        (self.out_dir / "demo.db-wal").write_bytes(b"junk" * 1000)
        (self.out_dir / "demo.db-shm").write_bytes(b"junk" * 1000)
        code, _, err = self.run_main()
        self.assertEqual(code, 0, err)
        self.assertEqual(self.integrity(dest), "ok")
        self.assertEqual(self.session_count(dest), 1)

    def test_an_existing_db_path_is_refused_without_overwrite(self) -> None:
        dest = self.out_dir / "demo.db"
        dest.write_bytes(b"someone's file, not a demo database")
        before = snapshot(self.out_dir)
        code, _, err = self.run_main(dest=dest)
        self.assertEqual(code, 2)
        self.assertIn("already exists", err)
        self.assertIn("--overwrite", err)
        self.assertEqual(snapshot(self.out_dir), before)

    def test_from_real_also_refuses_an_existing_db_path_without_overwrite(self) -> None:
        dest = self.out_dir / "demo.db"
        dest.write_bytes(b"keep me")
        before = snapshot(self.out_dir)
        code, _, _ = self.run_main("--from-real", dest=dest)
        self.assertEqual(code, 2)
        self.assertEqual(snapshot(self.out_dir), before)

    def test_overwrite_replaces_an_existing_demo_database(self) -> None:
        dest = self.out_dir / "demo.db"
        self.assertEqual(self.run_main()[0], 0)
        code, _, err = self.run_main("--overwrite")
        self.assertEqual(code, 0, err)
        self.assertEqual(self.integrity(dest), "ok")
        self.assertEqual(self.session_count(dest), 1)  # replaced, not appended to

    def test_overwrite_replaces_a_file_that_is_not_a_database(self) -> None:
        dest = self.out_dir / "demo.db"
        dest.write_bytes(b"not a database")
        code, _, err = self.run_main("--overwrite")
        self.assertEqual(code, 0, err)
        self.assertEqual(self.session_count(dest), 1)

    def test_d_demo_content_is_unchanged_for_data_seed_0(self) -> None:
        path, session_id = demo_db.build_demo_db(self.out_dir / "d.db", data_seed=0)
        self.assertEqual(
            fingerprint(path, session_id),
            (DEMO_SEED0_HR_ROWS, DEMO_SEED0_PREDICTION_ROWS, DEMO_SEED0_FINGERPRINT),
        )

    def test_h_fresh_db_works_with_the_apps_read_path(self) -> None:
        path, session_id = demo_db.build_demo_db(self.out_dir / "d.db", data_seed=3)
        result = stats.compute_session_stats(session_id, path=path)
        self.assertEqual(result["duration_s"], 340.0)
        self.assertEqual(sum(result["activity_seconds"].values()), 340.0)
        self.assertTrue(stats.is_summarizable(result))
        self.assertIn("Running", stats.format_stats_plain(result))


class FromRealModeTests(FakeRealDbCase):
    def test_b_from_real_leaves_the_real_db_untouched(self) -> None:
        before = snapshot(self.real_dir)
        code, _, err = self.run_main("--from-real")
        self.assertEqual(code, 0, err)
        self.assertEqual(snapshot(self.real_dir), before)

    def test_c_from_real_copy_is_complete(self) -> None:
        dest = self.out_dir / "demo.db"
        code, _, err = self.run_main("--from-real")
        self.assertEqual(code, 0, err)
        self.assertIn("Close HeartWatch before using --from-real", err)
        self.assertEqual(self.integrity(dest), "ok")
        with contextlib.closing(sqlite3.connect(str(dest))) as conn:
            devices = {r[0] for r in conn.execute("SELECT device_id FROM sessions")}
            real_hr = conn.execute(
                "SELECT COUNT(*) FROM hr_samples WHERE session_id = "
                "(SELECT id FROM sessions WHERE device_id = 'real-device')"
            ).fetchone()[0]
        # The real session's rows exist only in the real database's -wal, so
        # finding them proves the -wal was copied; the demo session is extra.
        self.assertEqual(devices, {"real-device", "demo-session"})
        self.assertEqual(real_hr, 2)

    def test_from_real_with_no_real_database_is_an_error(self) -> None:
        self.writer.close()
        for path in self.real_dir.iterdir():
            path.unlink()
        code, _, err = self.run_main("--from-real")
        self.assertNotEqual(code, 0)
        self.assertIn("no real database", err.lower())
        self.assertFalse((self.out_dir / "demo.db").exists())

    def test_i_integrity_failure_leaves_nothing_behind(self) -> None:
        dest = self.out_dir / "demo.db"
        before = snapshot(self.real_dir)
        with mock.patch.object(demo_db, "integrity_check", return_value="row 1 is bad"):
            code, _, err = self.run_main("--from-real")
        self.assertNotEqual(code, 0)
        self.assertIn("integrity", err.lower())
        for suffix in ("", *SIDECARS):
            self.assertFalse(Path(f"{dest}{suffix}").exists(), suffix)
        self.assertEqual(snapshot(self.real_dir), before)


class GuardTests(FakeRealDbCase):
    def assert_refused(self, *args: str, dest: Path | str) -> None:
        before = snapshot(self.real_dir)
        code, _, err = self.run_main(*args, dest=dest)
        self.assertEqual(code, 2)
        self.assertIn("real database", err)
        self.assertEqual(snapshot(self.real_dir), before)

    def test_f_same_path_is_refused_in_both_modes(self) -> None:
        self.assert_refused(dest=self.real)
        self.assert_refused("--from-real", dest=self.real)

    def test_overwrite_does_not_get_past_the_guard(self) -> None:
        # check_destination() runs the real-DB guard before it looks at
        # --overwrite, so --overwrite can't make the script delete the real DB.
        self.assert_refused("--overwrite", dest=self.real)
        self.assert_refused("--overwrite", "--from-real", dest=self.real)
        link = self.out_dir / "link.db"
        os.symlink(self.real, link)
        self.assert_refused("--overwrite", dest=link)

    def test_g_symlink_is_refused(self) -> None:
        link = self.out_dir / "link.db"
        os.symlink(self.real, link)
        self.assert_refused(dest=link)
        self.assert_refused("--from-real", dest=link)

    def test_g_hard_link_is_refused(self) -> None:
        hard = self.out_dir / "hard.db"
        os.link(self.real, hard)
        self.assert_refused(dest=hard)

    def test_g_relative_path_is_refused(self) -> None:
        old_cwd = os.getcwd()
        self.addCleanup(os.chdir, old_cwd)
        os.chdir(self.root)
        self.assert_refused(dest="demo/../real/heartwatch.db")

    def test_the_real_path_follows_the_env_var_like_the_app_does(self) -> None:
        for environ in ({}, {DB_PATH_ENV_VAR: ""}, {DB_PATH_ENV_VAR: str(self.real)}):
            with self.subTest(environ=environ):
                self.assertEqual(demo_db.real_db_path(environ), db.resolve_db_path(environ))


class EvalGuardTests(FakeRealDbCase):
    """eval_summaries.py shares check_destination(), so --overwrite can't
    get it past the real-DB guard either. The guard refuses before any
    Ollama call."""

    def test_eval_overwrite_does_not_get_past_the_guard(self) -> None:
        results = self.root / "results"
        results.mkdir()
        for dest in (self.real, "symlink"):
            with self.subTest(dest=dest):
                if dest == "symlink":
                    dest = self.out_dir / "link.db"
                    os.symlink(self.real, dest)
                before = snapshot(self.real_dir)
                stderr = io.StringIO()
                with (
                    mock.patch.dict(os.environ, {DB_PATH_ENV_VAR: str(self.real)}),
                    contextlib.redirect_stderr(stderr),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    code = eval_summaries.main(
                        ["--db", str(dest), "--overwrite", "--out", str(results)]
                    )
                self.assertEqual(code, 2)
                self.assertIn("real database", stderr.getvalue())
                self.assertEqual(snapshot(self.real_dir), before)
                self.assertEqual(list(results.iterdir()), [])


if __name__ == "__main__":
    unittest.main()


class HistoryTests(FakeRealDbCase):
    """Handoff 9a, 3.10 (K): --history-days adds earlier sessions from its
    own random generator; off, nothing changes."""

    def sessions(self, path: Path) -> list[tuple[int, int, int | None]]:
        with contextlib.closing(sqlite3.connect(str(path))) as conn:
            return conn.execute(
                "SELECT id, started_at, ended_at FROM sessions ORDER BY id").fetchall()

    def test_off_changes_nothing(self) -> None:
        path, session_id = demo_db.build_demo_db(self.out_dir / "d.db", data_seed=0,
                                                 history_days=0, real=self.real)
        self.assertEqual(fingerprint(path, session_id),
                         (DEMO_SEED0_HR_ROWS, DEMO_SEED0_PREDICTION_ROWS, DEMO_SEED0_FINGERPRINT))
        self.assertEqual(len(self.sessions(path)), 1)

    def test_on_keeps_the_demo_session_and_adds_one_per_earlier_day(self) -> None:
        path, session_id = demo_db.build_demo_db(self.out_dir / "d.db", data_seed=0,
                                                 history_days=7, real=self.real)
        # Same id, same content: the history uses its own generator.
        self.assertEqual(session_id, 1)
        self.assertEqual(fingerprint(path, session_id),
                         (DEMO_SEED0_HR_ROWS, DEMO_SEED0_PREDICTION_ROWS, DEMO_SEED0_FINGERPRINT))
        rows = self.sessions(path)
        self.assertEqual(len(rows), 8)
        self.assertTrue(all(ended is not None for _, _, ended in rows))
        # Every history session started before the demo session, on a
        # different day each, so the demo session stays the latest.
        demo_start = rows[0][1]
        days = {(demo_start - start) // 86_400_000 for _, start, _ in rows[1:]}
        self.assertTrue(all(start < demo_start for _, start, _ in rows[1:]))
        self.assertEqual(len({start for _, start, _ in rows[1:]}), 7)
        self.assertLessEqual(max(days), 7)

    def test_the_same_seed_gives_the_same_history(self) -> None:
        a, _ = demo_db.build_demo_db(self.out_dir / "a.db", data_seed=0, history_days=3,
                                     real=self.real)
        b, _ = demo_db.build_demo_db(self.out_dir / "b.db", data_seed=0, history_days=3,
                                     real=self.real)
        for sid in (2, 3, 4):
            self.assertEqual(fingerprint(a, sid), fingerprint(b, sid))

    def test_the_pinned_start_changes_no_content(self) -> None:
        path, session_id = demo_db.build_demo_db(self.out_dir / "d.db", data_seed=0,
                                                 started_at=1_760_000_000_000, real=self.real)
        self.assertEqual(fingerprint(path, session_id),
                         (DEMO_SEED0_HR_ROWS, DEMO_SEED0_PREDICTION_ROWS, DEMO_SEED0_FINGERPRINT))
        self.assertEqual(self.sessions(path)[0][1], 1_760_000_000_000)

    def test_command_line_option(self) -> None:
        code, out, err = self.run_main("--data-seed", "0", "--history-days", "3")
        self.assertEqual(code, 0, err)
        self.assertEqual(self.session_count(self.out_dir / "demo.db"), 4)
        self.assertIn("3 days before today", out)
        with self.assertRaises(SystemExit) as exit_, contextlib.redirect_stderr(io.StringIO()):
            self.run_main("--history-days", "-1", "--overwrite")
        self.assertEqual(exit_.exception.code, 2)
