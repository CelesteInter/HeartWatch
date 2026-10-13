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
from unittest import mock

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

        # Each prediction owns the 5s window ending at its timestamp:
        # Sitting [5s,10s) = 5s, Walking [25s,30s) = 5s, Sitting [35s,40s)
        # = 5s. 10s-25s, 30s-35s and 40s-60s have no prediction.
        self.assertAlmostEqual(result["activity_seconds"]["Sitting"], 10.0)
        self.assertAlmostEqual(result["activity_seconds"]["Walking"], 5.0)
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

    # Predictions are window END times, each owning the 5s before it:
    # Sitting [5s,10s), Walking [25s,30s), Sitting [35s,40s) -- 10s Sitting,
    # 5s Walking in total. [10s,25s), [30s,35s) and [40s,60s] are gaps.
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
        self.assertEqual(result["activity_seconds"], {"Sitting": 10.0, "Walking": 5.0})
        # Three 5s windows, no overlap: 15s attributed out of 60s.
        self.assertAlmostEqual(sum(result["activity_seconds"].values()), 15.0)

    def test_peak_during_walking(self) -> None:
        self.assertEqual(self._stats_with_peak_at(27_000)["peak_hr_activity"], "Walking")

    def test_peak_at_a_window_start_belongs_to_that_window(self) -> None:
        # 35s is where the Sitting window ending at 40s starts (inclusive).
        self.assertEqual(self._stats_with_peak_at(35_000)["peak_hr_activity"], "Sitting")

    def test_peak_between_prediction_windows_is_none(self) -> None:
        # 20s is in the [10s,25s) gap no prediction covers.
        self.assertIsNone(self._stats_with_peak_at(20_000)["peak_hr_activity"])

    def test_peak_at_session_end_after_last_prediction_is_none(self) -> None:
        # The last prediction (40s) owns [35s,40s); nothing after it.
        self.assertIsNone(self._stats_with_peak_at(60_000)["peak_hr_activity"])

    def test_peak_at_start_of_first_window_counts(self) -> None:
        # The first prediction (10s) owns [5s,10s), which includes 5s.
        self.assertEqual(self._stats_with_peak_at(5_000)["peak_hr_activity"], "Sitting")

    def test_no_predictions_gives_no_peak_activity(self) -> None:
        session_id = self._make_session(hr_rows=[(0, 80, 0.9)], ended_at=60_000)
        result = stats.compute_session_stats(session_id, path=self.db_path)
        self.assertIsNone(result["peak_hr_activity"])

    def test_labels_get_canonical_capitalization(self) -> None:
        session_id = self._make_session(
            prediction_rows=[(10_000, "walking", 0.9)], ended_at=60_000
        )
        result = stats.compute_session_stats(session_id, path=self.db_path)
        # The single prediction at 10s owns [5s,10s).
        self.assertEqual(result["activity_seconds"], {"Walking": 5.0})


class WindowConventionTests(unittest.TestCase):
    """PREDICTION_TIMESTAMP_MARKS: which time each prediction is credited
    with. Every expected value below is worked out by hand from the rule in
    data/stats.py's _activity_segments() docstring."""

    setUp = ComputeSessionStatsTests.setUp
    tearDown = ComputeSessionStatsTests.tearDown
    _make_session = ComputeSessionStatsTests._make_session

    def _transition_session(self) -> int:
        """Walking predictions at 5s, 10s, ..., 210s, then Running at 215s,
        ..., 255s; the session ends at 255s. One HR sample per second, 100
        bpm except a 150 bpm peak at exactly 210s -- the first second of
        Running."""
        predictions = [(t * 1000, "Walking", 0.9) for t in range(5, 211, 5)]
        predictions += [(t * 1000, "Running", 0.9) for t in range(215, 256, 5)]
        hr = [(t * 1000, 150 if t == 210 else 100, 0.9) for t in range(0, 255)]
        return self._make_session(hr_rows=hr, prediction_rows=predictions, ended_at=255_000)

    def test_peak_at_a_transition_belongs_to_the_new_activity(self) -> None:
        result = stats.compute_session_stats(self._transition_session(), path=self.db_path)
        # Walking 5s..210s owns [0s,210s) = 210s; Running 215s..255s owns
        # [210s,255s) = 45s. 210s is in the Running window ending at 215s.
        self.assertEqual(result["activity_seconds"], {"Walking": 210.0, "Running": 45.0})
        self.assertEqual(result["peak_hr_activity"], "Running")

    def test_start_convention_shifts_every_window_forward(self) -> None:
        session_id = self._transition_session()
        with mock.patch.object(stats, "PREDICTION_TIMESTAMP_MARKS", "start"):
            result = stats.compute_session_stats(session_id, path=self.db_path)
        # Walking 5s..210s owns [5s,215s) = 210s; Running 215s..250s owns
        # [215s,255s) = 40s, and the one at 255s would own [255s,260s),
        # which is past the session's end. [0s,5s) belongs to nothing.
        # 210s is now inside Walking's [210s,215s).
        self.assertEqual(result["activity_seconds"], {"Walking": 210.0, "Running": 40.0})
        self.assertEqual(result["peak_hr_activity"], "Walking")

    def test_predictions_every_window_cover_the_whole_session(self) -> None:
        # Sitting 5s..30s owns [0s,30s); Standing 35s..60s owns [30s,60s).
        predictions = [
            (t * 1000, "Sitting" if t <= 30 else "Standing", 0.9) for t in range(5, 61, 5)
        ]
        session_id = self._make_session(prediction_rows=predictions, ended_at=60_000)
        result = stats.compute_session_stats(session_id, path=self.db_path)
        self.assertEqual(result["activity_seconds"], {"Sitting": 30.0, "Standing": 30.0})
        self.assertEqual(sum(result["activity_seconds"].values()), result["duration_s"])

    def test_overlapping_windows_are_not_double_counted(self) -> None:
        # A prediction every 2.5s: each owns only the 2.5s since the one
        # before it, so 24 predictions cover 60s, not 24 x 5s = 120s.
        predictions = [
            (t, "Walking" if t <= 30_000 else "Running", 0.9) for t in range(2_500, 60_001, 2_500)
        ]
        session_id = self._make_session(prediction_rows=predictions, ended_at=60_000)
        result = stats.compute_session_stats(session_id, path=self.db_path)
        self.assertEqual(result["activity_seconds"], {"Walking": 30.0, "Running": 30.0})

    def _session_missing_30s(self, hr_rows=()) -> int:
        """Walking predictions every 5s from 5s to 60s, except 30s."""
        predictions = [(t * 1000, "Walking", 0.9) for t in range(5, 61, 5) if t != 30]
        return self._make_session(hr_rows=hr_rows, prediction_rows=predictions, ended_at=60_000)

    def test_missing_prediction_leaves_exactly_one_window_unattributed(self) -> None:
        result = stats.compute_session_stats(self._session_missing_30s(), path=self.db_path)
        # [25s,30s) had no prediction; 35s still owns only [30s,35s).
        self.assertEqual(result["activity_seconds"], {"Walking": 55.0})
        self.assertEqual(result["duration_s"] - 55.0, 5.0)

    def test_peak_inside_a_gap_is_none(self) -> None:
        hr = [(0, 80, 0.9), (27_000, 140, 0.9), (50_000, 90, 0.9)]
        result = stats.compute_session_stats(self._session_missing_30s(hr), path=self.db_path)
        self.assertEqual(result["max_hr"], 140)
        self.assertIsNone(result["peak_hr_activity"])

    def test_open_session_gets_nothing_past_the_last_prediction(self) -> None:
        # Still recording (no ended_at): the last prediction is at 15s, a
        # peak at 20s is after it, and "now" is far later.
        session_id = self._make_session(
            hr_rows=[(1_000, 90, 0.9), (20_000, 130, 0.9)],
            prediction_rows=[(t * 1000, "Walking", 0.9) for t in (5, 10, 15)],
            ended_at=None,
        )
        result = stats.compute_session_stats(session_id, path=self.db_path)
        self.assertEqual(result["activity_seconds"], {"Walking": 15.0})
        self.assertGreater(result["duration_s"], 15.0)
        self.assertIsNone(result["peak_hr_activity"])

    def test_activities_are_keyed_in_first_appearance_order(self) -> None:
        # Walking [0s,5s) + [10s,15s) = 10s; Sitting [5s,10s) + [15s,20s) =
        # 10s. A tie, so the one seen first (Walking) is listed first.
        predictions = [(5_000, "Walking", 0.9), (10_000, "Sitting", 0.9)]
        predictions += [(15_000, "Walking", 0.9), (20_000, "Sitting", 0.9)]
        session_id = self._make_session(prediction_rows=predictions, ended_at=20_000)
        result = stats.compute_session_stats(session_id, path=self.db_path)
        self.assertEqual(
            list(result["activity_seconds"].items()), [("Walking", 10.0), ("Sitting", 10.0)]
        )
        self.assertIn(
            "Time by activity: Walking 10 seconds, Sitting 10 seconds.",
            stats.format_stats_plain(result),
        )


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
            "activity: Walking 2 minutes 30 seconds, Sitting 2 minutes 10 seconds, "
            "Standing 1 minute 40 seconds.",
        )

    def test_ties_keep_first_appearance_order(self) -> None:
        # Standing and Walking tie at 60s; Standing appeared first in the
        # session (it's first in the dict), so it's listed first.
        text = stats.format_stats_plain(
            {**self.FULL, "activity_seconds": {"Standing": 60.0, "Walking": 60.0, "Sitting": 90.0}}
        )
        self.assertTrue(
            text.endswith(
                "Time by activity: Sitting 1 minute 30 seconds, Standing 1 minute, "
                "Walking 1 minute."
            )
        )

    def test_single_activity_is_the_whole_session(self) -> None:
        text = stats.format_stats_plain({**self.FULL, "activity_seconds": {"Walking": 380.0}})
        self.assertNotIn("Time by activity", text)
        self.assertTrue(
            text.endswith("reached during Walking. Activity: Walking for the whole session.")
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
