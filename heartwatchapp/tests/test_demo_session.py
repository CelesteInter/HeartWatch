"""Tests for the demo tooling: scripts/demo_session.py's real-database
guard, and the dummy predictions "Seed demo session" now writes.

Stdlib unittest, same as the other test files. Run from the repo root:

    python -m unittest heartwatchapp.tests.test_demo_session

Never launches the app and never opens the real database -- the guard is
tested as a plain function, and seeding writes to a temporary fixture DB.
"""

from __future__ import annotations

import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path

from ..data import db, seed, stats
from ..data.writer import DatabaseWriter
from ..scripts import demo_db, demo_session


class RealDatabaseGuardTests(unittest.TestCase):
    def test_scripts_idea_of_the_real_db_matches_db_py(self) -> None:
        self.assertEqual(demo_session.REAL_DB_PATH, db.DEFAULT_DB_PATH)
        self.assertEqual(demo_session.DB_PATH_ENV_VAR, db.DB_PATH_ENV_VAR)

    def test_real_db_path_is_refused(self) -> None:
        with self.assertRaises(demo_session.RealDatabaseError):
            demo_session.ensure_not_real_db(db.DEFAULT_DB_PATH)

    def test_roundabout_path_to_real_db_is_refused(self) -> None:
        roundabout = db.DEFAULT_DB_PATH.parent / ".." / "data" / db.DEFAULT_DB_PATH.name
        with self.assertRaises(demo_session.RealDatabaseError):
            demo_session.ensure_not_real_db(roundabout)

    def test_symlink_to_real_db_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            real = Path(tmp) / "real.db"
            real.write_bytes(b"")
            link = Path(tmp) / "link.db"
            os.symlink(real, link)
            with self.assertRaises(demo_session.RealDatabaseError):
                demo_session.ensure_not_real_db(link, real=real)

    def test_other_path_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "copy.db"
            self.assertEqual(demo_session.ensure_not_real_db(target), target.resolve())

    def test_main_refuses_real_db_before_writing_anything(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = demo_session.main(["--db", str(db.DEFAULT_DB_PATH), "--no-launch"])
        self.assertEqual(code, 2)
        self.assertIn("real database", stderr.getvalue())


class DemoSessionFactsTests(unittest.TestCase):
    """The session scripts/demo_db.py builds, read back through
    data/stats.py. Built on new empty temporary databases (copy_real=False),
    so the real database isn't opened."""

    def test_every_phase_is_credited_in_full_and_the_peak_is_running(self) -> None:
        # Phases are Sitting 70s, Walking 140s, Running 45s, Standing 85s;
        # each phase's predictions end at 5s, 10s, ... into it, so each 5s
        # window lies inside its own phase and the four add up to 340s.
        # Running's heart rate (~128 bpm) is far above every other phase's.
        for data_seed in range(6):
            with self.subTest(data_seed=data_seed), tempfile.TemporaryDirectory() as tmp:
                path, session_id = demo_db.build_demo_db(
                    Path(tmp) / "demo.db", copy_real=False, data_seed=data_seed
                )
                result = stats.compute_session_stats(session_id, path=path)
            self.assertEqual(
                result["activity_seconds"],
                {"Sitting": 70.0, "Walking": 140.0, "Running": 45.0, "Standing": 85.0},
            )
            self.assertEqual(result["duration_s"], 340.0)
            self.assertEqual(result["peak_hr_activity"], "Running")

    def test_same_data_seed_gives_the_same_heart_rate(self) -> None:
        results = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as tmp:
                path, session_id = demo_db.build_demo_db(
                    Path(tmp) / "demo.db", copy_real=False, data_seed=7
                )
                results.append(stats.compute_session_stats(session_id, path=path))
        self.assertEqual(results[0]["avg_hr"], results[1]["avg_hr"])
        self.assertEqual(results[0]["max_hr"], results[1]["max_hr"])

    def test_builder_refuses_real_db(self) -> None:
        with self.assertRaises(demo_db.RealDatabaseError):
            demo_db.build_demo_db(db.DEFAULT_DB_PATH, copy_real=False)


class SeededPredictionTests(unittest.TestCase):
    def test_generated_predictions_cover_the_session_in_5s_windows(self) -> None:
        rows = seed.generate_prediction_rows(0, 120.0, "Walking")
        self.assertEqual([ts for ts, _, _ in rows], list(range(5_000, 120_001, 5_000)))
        for _, label, confidence in rows:
            self.assertIn(label, seed.ACTIVITY_LABELS)
            self.assertGreaterEqual(confidence, seed.DUMMY_CONFIDENCE_RANGE[0])
            self.assertLessEqual(confidence, seed.DUMMY_CONFIDENCE_RANGE[1])

    def test_seed_demo_session_writes_dummy_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.db"
            db.init_db(path)
            writer = DatabaseWriter(path)
            writer.start()
            try:
                session_id = writer.seed_demo_session(label="Sitting", duration_s=90.0)
            finally:
                writer.stop()

            timeline = db.get_session_timeline(session_id, path)
            self.assertEqual(len(timeline["predictions"]), 18)  # 90s / 5s windows
            result = stats.compute_session_stats(session_id, path=path)
            self.assertTrue(stats.is_summarizable(result))

            with contextlib.closing(db._connect(path)) as conn:
                versions = {
                    row[0]
                    for row in conn.execute(
                        "SELECT DISTINCT model_version FROM predictions WHERE session_id = ?",
                        (session_id,),
                    )
                }
            self.assertEqual(versions, {seed.DUMMY_MODEL_VERSION})


if __name__ == "__main__":
    unittest.main()
