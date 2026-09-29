"""Tests for data/stats.py's session-number crunching.

Uses stdlib unittest, not pytest -- pytest isn't installed in
heartwatchapp/.venv and the ground rules for this change say not to
install packages. Run with:

    python -m unittest heartwatchapp.tests.test_stats

or run every test file (this one and test_validate.py) at once:

    python -m unittest discover -s heartwatchapp/tests -t .

No live Ollama daemon is needed; this only exercises the SQLite/arithmetic
layer in data/stats.py, never llm/client.py.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ..data import db, stats


class ComputeSessionStatsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self._tmpdir.name) / "fixture.db"
        db.init_db(self.db_path)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _make_session(
        self, hr_rows=(), prediction_rows=(), ended_at: int | None = 60_000
    ) -> int:
        """Writes a fixture session directly through db.py's insert
        statements (the same ones data/writer.py calls), started at t=0."""
        import contextlib

        with contextlib.closing(db._connect(self.db_path)) as conn:
            with conn:
                session_id = db.start_session(
                    conn, "fixture", "Walking", 1.0, 1.0, started_at=0
                )
                if hr_rows:
                    db.insert_hr_batch(conn, session_id, list(hr_rows))
                if prediction_rows:
                    for ts, predicted, confidence in prediction_rows:
                        db.insert_prediction(
                            conn, session_id, ts, predicted, confidence, "test-model"
                        )
                if ended_at is not None:
                    db.end_session(conn, session_id, ended_at)
        return session_id

    def test_duration_and_heart_rate(self) -> None:
        session_id = self._make_session(
            hr_rows=[(0, 70, 0.9), (10_000, 80, 0.9), (20_000, 90, 0.9)],
            ended_at=60_000,
        )
        result = stats.compute_session_stats(session_id, path=self.db_path)

        self.assertEqual(result["duration_s"], 60.0)
        self.assertAlmostEqual(result["avg_hr"], 80.0)
        self.assertEqual(result["max_hr"], 90)
        self.assertEqual(result["hr_sample_count"], 3)

    def test_activity_seconds_from_predictions(self) -> None:
        session_id = self._make_session(
            prediction_rows=[
                (10_000, "Sitting", 0.95),
                (30_000, "Walking", 0.9),
                (40_000, "Sitting", 0.2),
            ],
            ended_at=60_000,
        )
        result = stats.compute_session_stats(session_id, path=self.db_path)

        # Sitting [10s,30s) = 20s, Walking [30s,40s) = 10s, Sitting [40s,60s] = 20s.
        self.assertAlmostEqual(result["activity_seconds"]["Sitting"], 40.0)
        self.assertAlmostEqual(result["activity_seconds"]["Walking"], 10.0)
        self.assertEqual(result["prediction_count"], 3)
        self.assertEqual(result["low_confidence_count"], 1)  # the 0.2-confidence row

    def test_no_hr_or_predictions_yields_none_and_empty(self) -> None:
        session_id = self._make_session()
        result = stats.compute_session_stats(session_id, path=self.db_path)

        self.assertIsNone(result["avg_hr"])
        self.assertIsNone(result["max_hr"])
        self.assertEqual(result["activity_seconds"], {})
        self.assertEqual(result["label"], "Walking")

    def test_unknown_session_raises(self) -> None:
        with self.assertRaises(ValueError):
            stats.compute_session_stats(999, path=self.db_path)


class RelationalFactsTests(unittest.TestCase):
    """Time per activity, and which activity the peak HR happened during."""

    # Predictions are window END times: Sitting [10s,30s), Walking [30s,40s),
    # Sitting [40s,60s] -- 40s Sitting, 10s Walking in total.
    PREDICTIONS = [(10_000, "Sitting", 0.95), (30_000, "Walking", 0.9), (40_000, "Sitting", 0.9)]

    setUp = ComputeSessionStatsTests.setUp
    tearDown = ComputeSessionStatsTests.tearDown
    _make_session = ComputeSessionStatsTests._make_session

    def _stats_with_peak_at(self, peak_ts: int) -> dict:
        hr = [(0, 70, 0.9), (20_000, 75, 0.9), (35_000, 72, 0.9), (50_000, 71, 0.9)]
        hr = sorted([row for row in hr if row[0] != peak_ts] + [(peak_ts, 120, 0.9)])
        session_id = self._make_session(
            hr_rows=hr, prediction_rows=self.PREDICTIONS, ended_at=60_000
        )
        return stats.compute_session_stats(session_id, path=self.db_path)

    def test_time_per_activity_sums_correctly(self) -> None:
        result = self._stats_with_peak_at(35_000)
        self.assertEqual(result["activity_seconds"], {"Sitting": 40.0, "Walking": 10.0})
        # Everything from the first prediction (10s) to the end (60s) is covered.
        self.assertAlmostEqual(sum(result["activity_seconds"].values()), 50.0)

    def test_peak_during_walking(self) -> None:
        self.assertEqual(self._stats_with_peak_at(35_000)["peak_hr_activity"], "Walking")

    def test_peak_during_sitting(self) -> None:
        self.assertEqual(self._stats_with_peak_at(20_000)["peak_hr_activity"], "Sitting")

    def test_peak_at_session_end_counts_as_last_activity(self) -> None:
        self.assertEqual(self._stats_with_peak_at(60_000)["peak_hr_activity"], "Sitting")

    def test_peak_outside_every_prediction_window_is_none(self) -> None:
        # 5s is before the first prediction's window end (10s), so no
        # activity covers it.
        self.assertIsNone(self._stats_with_peak_at(5_000)["peak_hr_activity"])

    def test_no_predictions_gives_no_peak_activity(self) -> None:
        session_id = self._make_session(hr_rows=[(0, 80, 0.9)], ended_at=60_000)
        result = stats.compute_session_stats(session_id, path=self.db_path)
        self.assertIsNone(result["peak_hr_activity"])

    def test_labels_get_canonical_capitalization(self) -> None:
        session_id = self._make_session(
            prediction_rows=[(10_000, "walking", 0.9)], ended_at=60_000
        )
        result = stats.compute_session_stats(session_id, path=self.db_path)
        self.assertEqual(result["activity_seconds"], {"Walking": 50.0})


class FormatStatsPlainTests(unittest.TestCase):
    FULL = {
        "duration_s": 380.0,
        "avg_hr": 92.6,
        "max_hr": 111,
        "activity_seconds": {"Walking": 150.0, "Sitting": 130.0, "Standing": 100.0},
        "peak_hr_activity": "Walking",
        "label": "Walking",
    }

    def test_full_facts(self) -> None:
        self.assertEqual(
            stats.format_stats_plain(self.FULL),
            "The session lasted 6 minutes 20 seconds. Average heart rate was 93 bpm "
            "and peak heart rate was 111 bpm, reached during Walking. Time by "
            "activity: Sitting 2 minutes 10 seconds, Walking 2 minutes 30 seconds, "
            "Standing 1 minute 40 seconds.",
        )

    def test_no_predictions_falls_back_to_logged_label(self) -> None:
        text = stats.format_stats_plain(
            {**self.FULL, "activity_seconds": {}, "peak_hr_activity": None, "label": "Running"}
        )
        self.assertNotIn("Time by activity", text)
        self.assertNotIn("reached during", text)
        self.assertTrue(text.endswith("peak heart rate was 111 bpm. Logged activity: Running."))

    def test_no_predictions_and_no_label(self) -> None:
        text = stats.format_stats_plain(
            {**self.FULL, "activity_seconds": {}, "peak_hr_activity": None, "label": None}
        )
        self.assertTrue(text.endswith("peak heart rate was 111 bpm."))

    def test_no_peak_activity(self) -> None:
        text = stats.format_stats_plain({**self.FULL, "peak_hr_activity": None})
        self.assertIn("peak heart rate was 111 bpm. Time by activity:", text)

    def test_no_hr_samples(self) -> None:
        text = stats.format_stats_plain(
            {**self.FULL, "avg_hr": None, "max_hr": None, "peak_hr_activity": None}
        )
        self.assertIn("No heart-rate samples were recorded.", text)
        self.assertNotIn("bpm", text)


class DbPathTests(unittest.TestCase):
    """HEARTWATCH_DB_PATH, read by data/db.py's resolve_db_path()."""

    def test_unset_gives_default_path(self) -> None:
        self.assertEqual(db.resolve_db_path({}), db.DEFAULT_DB_PATH)
        self.assertEqual(db.resolve_db_path({db.DB_PATH_ENV_VAR: ""}), db.DEFAULT_DB_PATH)

    def test_set_gives_that_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "copy.db"
            self.assertEqual(
                db.resolve_db_path({db.DB_PATH_ENV_VAR: str(target)}), target.resolve()
            )


def _stats(**overrides) -> dict:
    """A data-rich stats dict (5.5 minutes, 330 HR samples, two activities)
    that passes is_summarizable(); tests override one field at a time."""
    base = {
        "session_id": 1,
        "label": "Walking",
        "duration_s": 330.0,
        "avg_hr": 101.4,
        "max_hr": 134,
        "hr_sample_count": 330,
        "activity_seconds": {"Walking": 190.0, "Standing": 125.0},
        "prediction_count": 30,
        "low_confidence_count": 0,
    }
    base.update(overrides)
    return base


class IsSummarizableTests(unittest.TestCase):
    def test_data_rich_session_passes(self) -> None:
        self.assertTrue(stats.is_summarizable(_stats()))

    def test_no_predictions_fails(self) -> None:
        self.assertFalse(stats.is_summarizable(_stats(prediction_count=0)))

    def test_too_few_hr_samples_fails(self) -> None:
        self.assertFalse(
            stats.is_summarizable(_stats(hr_sample_count=stats.MIN_HR_SAMPLES - 1))
        )

    def test_too_short_fails(self) -> None:
        self.assertFalse(
            stats.is_summarizable(_stats(duration_s=stats.MIN_DURATION_S - 0.1))
        )

    def test_exactly_at_every_threshold_passes(self) -> None:
        self.assertTrue(
            stats.is_summarizable(
                _stats(
                    prediction_count=stats.MIN_PREDICTIONS,
                    hr_sample_count=stats.MIN_HR_SAMPLES,
                    duration_s=stats.MIN_DURATION_S,
                )
            )
        )

    def test_session_7_shape_fails(self) -> None:
        # Session #7 as described in the handoff: 0 predictions, 23 HR
        # samples, ~23 seconds.
        self.assertFalse(
            stats.is_summarizable(
                _stats(prediction_count=0, hr_sample_count=23, duration_s=22.8)
            )
        )
        # Session #7 as it actually is in the current dev database: 62 HR
        # samples over 62 seconds, still 0 predictions -- fails on
        # predictions alone.
        self.assertFalse(
            stats.is_summarizable(
                _stats(prediction_count=0, hr_sample_count=62, duration_s=62.062)
            )
        )


class ConfidenceCaveatTests(unittest.TestCase):
    def test_caveat_when_any_low_confidence(self) -> None:
        self.assertEqual(
            stats.confidence_caveat(_stats(low_confidence_count=4)),
            "Some windows had low classification confidence.",
        )

    def test_no_caveat_when_none_low(self) -> None:
        self.assertIsNone(stats.confidence_caveat(_stats(low_confidence_count=0)))


if __name__ == "__main__":
    unittest.main()
