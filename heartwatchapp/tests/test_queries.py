"""Tests for the shared query layer (data/queries.py) and the metric
functions it hands rows to (data/stats.py).

Plain-language overview: every number the user reads is computed once, in
data/stats.py, and fetched through data/queries.py. These tests check that
rule from every side: the metric functions on hand-built rows (A), that the
Dashboard's path and the session summary agree on the demo session (B, C),
several sessions combined without a window crossing between them (D),
calendar-day ranges including daylight-saving days and the AI bar's week
and latest-session ranges (E), request validation (F), a scan for SQL
arithmetic outside stats.py (G), empty data (H), the summary text staying
byte-identical (I), the simulated-predictions flag, and the Sessions
table's rows.

Every expected value is worked out by hand from the rule in a comment, not
copied from the code's output. Every database is a temporary fixture or a
fresh demo database; the real database is never opened. Stdlib unittest:

    python -m unittest heartwatchapp.tests.test_queries
"""

from __future__ import annotations

import ast
import calendar
import contextlib
import hashlib
import re
import tempfile
import unittest
from pathlib import Path
from zoneinfo import ZoneInfo

from ..data import db, queries, stats
from ..data.queries import MetricRequest
from ..scripts import demo_db

APP_DIR = Path(__file__).resolve().parent.parent
DENVER = ZoneInfo("America/Denver")


def utc_ms(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> int:
    """Epoch ms of a UTC wall-clock time. Denver is UTC-6 on daylight time
    (MDT, Mar 8 - Nov 1 2026) and UTC-7 on standard time (MST), so local
    midnight is 06:00 or 07:00 UTC; the tests add those offsets by hand."""
    return calendar.timegm((year, month, day, hour, minute, 0)) * 1000


# Thursday October 8 2026, 12:00 MDT (18:00 UTC).
NOW = utc_ms(2026, 10, 8, 18)


def oct_midnight(day: int) -> int:
    """Local midnight on October `day` 2026 (MDT, so 06:00 UTC)."""
    return utc_ms(2026, 10, day, 6)


class FixtureDb:
    """A temporary database plus a helper to write one session into it
    through db.py's own insert statements."""

    def __init__(self, test: unittest.TestCase):
        tmp = tempfile.TemporaryDirectory()
        test.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "fixture.db"
        db.init_db(self.path)

    def add(
        self,
        started_at: int,
        ended_at: int | None,
        hr=(),
        predictions=(),
        label: str | None = None,
        model_version: str = "test-model",
    ) -> int:
        """hr: (offset_ms, bpm) pairs; predictions: (offset_ms, label)
        pairs. Offsets are from started_at."""
        with contextlib.closing(db._connect(self.path)) as conn, conn:
            sid = db.start_session(conn, "fixture", label, 1.0, 1.0, started_at=started_at)
            db.insert_hr_batch(conn, sid, [(started_at + off, bpm, None) for off, bpm in hr])
            for off, predicted in predictions:
                db.insert_prediction(conn, sid, started_at + off, predicted, 0.9, model_version)
            if ended_at is not None:
                db.end_session(conn, sid, ended_at)
        return sid

    def run(self, metric: str, activity: str | None = None, range_name: str | None = None,
            session_id: int | None = None) -> queries.MetricResult:
        return queries.run(
            MetricRequest(metric, activity, range_name, session_id),
            path=self.path, now_ms=NOW, tz=DENVER,
        )


# --- A. metric functions -------------------------------------------------

# The dropout example from the Phase A report: samples 1-5 at 0-4 s read
# 60, then 10 s with no readings, then samples 6-10 at 15-19 s read 120.
DROPOUT_SAMPLES = [(0, 60), (1000, 60), (2000, 60), (3000, 60), (4000, 60),
                   (15000, 120), (16000, 120), (17000, 120), (18000, 120), (19000, 120)]


class MetricFunctionTests(unittest.TestCase):
    def test_average_is_the_simple_mean_and_skips_the_dropout(self) -> None:
        # (5 x 60 + 5 x 120) / 10 = 900 / 10 = 90. The 10 s gap is not
        # filled in (a time-weighted mean would have given 75).
        self.assertEqual(stats.average_hr(DROPOUT_SAMPLES), 90.0)

    def test_average_of_nothing_is_none(self) -> None:
        self.assertIsNone(stats.average_hr([]))

    def test_peak_is_the_first_sample_to_hit_the_highest_value(self) -> None:
        # 120 first appears at 15 s.
        self.assertEqual(stats.peak_hr(DROPOUT_SAMPLES), (120, 15000))
        self.assertIsNone(stats.peak_hr([]))

    def test_time_per_activity_follows_the_window_rule(self) -> None:
        # Session 0-20 s. Walking at 5 s owns [0, 5); Walking at 10 s owns
        # [5, 10) -> Walking 10 s. Running at 20 s owns [max(10, 15), 20) =
        # [15, 20) -> 5 s. 10-15 s is a gap: no activity.
        predictions = [(5000, "Walking", 0.9), (10000, "Walking", 0.9), (20000, "Running", 0.9)]
        self.assertEqual(
            stats.time_per_activity(predictions, 0, 20000), {"Walking": 10.0, "Running": 5.0}
        )

    def test_hr_samples_during_an_activity(self) -> None:
        # Running owns [15, 20): the five 120 bpm samples at 15-19 s.
        predictions = [(5000, "Walking", 0.9), (10000, "Walking", 0.9), (20000, "Running", 0.9)]
        self.assertEqual(
            stats.hr_samples_during(DROPOUT_SAMPLES, predictions, 0, 20000, "Running"),
            DROPOUT_SAMPLES[5:],
        )

    def test_combine_keeps_first_appearance_order(self) -> None:
        # Walking 10 + 2 = 12, Running 5; Walking appeared first.
        combined = stats.combine_activity_seconds([{"Walking": 10.0}, {"Running": 5.0, "Walking": 2.0}])
        self.assertEqual(list(combined.items()), [("Walking", 12.0), ("Running", 5.0)])

    def test_session_count_and_total_time(self) -> None:
        # 20 s + 30 s = 50 s over 2 sessions.
        spans = [(0, 20000), (100000, 130000)]
        self.assertEqual(stats.session_count(spans), 2)
        self.assertEqual(stats.total_recorded_seconds(spans), 50.0)
        self.assertEqual(stats.session_duration_s(5000, 1000), 0.0)  # never negative


# --- B, C, I. the seed-0 demo session ----------------------------------------

# Captured from a fresh seed-0 demo database before this change (October 8,
# 2026), with the pre-change compute_session_stats() / format_stats_plain().
PLAIN_SEED0 = (
    "The session lasted 5 minutes 40 seconds. Average heart rate was 95 bpm and peak "
    "heart rate was 134 bpm, reached during Running. Time by activity: Walking 2 minutes "
    "20 seconds, Standing 1 minute 25 seconds, Sitting 1 minute 10 seconds, Running 45 seconds."
)
PLAIN_SEED0_SHA256 = "8c658286d7485c16da4ab76ff5a138ce7567200686646ec5b443f8e7d057dd80"


class DemoSessionTests(unittest.TestCase):
    """Built on a brand-new seed-0 demo database (build_demo_db()'s
    default), so the real database is never opened."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.path, cls.session_id = demo_db.build_demo_db(
            Path(cls._tmp.name) / "demo.db", data_seed=0
        )
        cls.summary_stats = stats.compute_session_stats(cls.session_id, cls.path)
        cls.plain = stats.format_stats_plain(cls.summary_stats)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def run_query(self, metric: str, activity: str | None = None) -> queries.MetricResult:
        return queries.run(MetricRequest(metric, activity, None, self.session_id), path=self.path)

    def test_b_average_hr_agrees_across_every_path(self) -> None:
        # The most important test: the Dashboard's path (run()), the summary's
        # numbers (compute_session_stats()) and the summary's text must show
        # the same average for the same session.
        result = self.run_query("avg_hr")
        self.assertFalse(result.empty)
        self.assertEqual(result.value, self.summary_stats["avg_hr"])
        self.assertIn(f"Average heart rate was {result.value:.0f} bpm", self.plain)

    def test_b_peak_hr_agrees_across_every_path(self) -> None:
        result = self.run_query("peak_hr")
        self.assertEqual(result.value, self.summary_stats["max_hr"])
        self.assertIn(f"peak heart rate was {result.value:.0f} bpm", self.plain)

    def test_b_each_activity_time_agrees_across_every_path(self) -> None:
        for label, seconds in self.summary_stats["activity_seconds"].items():
            with self.subTest(activity=label):
                result = self.run_query("time_in_activity", label)
                self.assertEqual(result.value, seconds)
                self.assertIn(f"{label} {stats.format_duration_words(result.value)}", self.plain)

    def test_c_breakdown_is_the_four_phases_most_time_first(self) -> None:
        # Phases: Sitting 70 s, Walking 140 s, Running 45 s, Standing 85 s
        # (scripts/demo_db.py's DEMO_PHASES), each fully credited.
        # Most time first: Walking 140, Standing 85, Sitting 70, Running 45
        # = 340 s.
        breakdown = queries.activity_breakdown(session_id=self.session_id, path=self.path)
        self.assertEqual(
            breakdown.items,
            (("Walking", 140.0), ("Standing", 85.0), ("Sitting", 70.0), ("Running", 45.0)),
        )
        self.assertEqual(sum(seconds for _, seconds in breakdown.items), 340.0)
        self.assertTrue(breakdown.simulated)  # the demo writes model_version "demo-dummy"

    def test_i_plain_summary_is_byte_identical_to_before(self) -> None:
        self.assertEqual(self.plain, PLAIN_SEED0)
        self.assertEqual(hashlib.sha256(self.plain.encode()).hexdigest(), PLAIN_SEED0_SHA256)


# --- D. several sessions ----------------------------------------------------

class MultiSessionTests(unittest.TestCase):
    """Three finished sessions on three days in the last 7 days, plus a
    live one today and an older one, which must both be left out."""

    def setUp(self) -> None:
        self.db = FixtureDb(self)
        # A: Oct 3, 09:00, 20 s. HR 60, 90. Walking at 5 s and 10 s:
        #    [0, 5) + [5, 10) = Walking 10 s.
        a = oct_midnight(3) + 9 * 3_600_000
        self.db.add(a, a + 20_000, hr=[(0, 60), (1000, 90)],
                    predictions=[(5000, "Walking"), (10000, "Walking")])
        # B: Oct 5, 09:00, 20 s. HR 100. Running at 5 s: Running 5 s.
        b = oct_midnight(5) + 9 * 3_600_000
        self.db.add(b, b + 20_000, hr=[(0, 100)], predictions=[(5000, "Running")])
        # C: Oct 8 (today), 08:00, 20 s. HR 120 x 3. Walking at 3 s owns
        #    [3 - 5, 3) clipped to the session's own start = [0, 3): 3 s.
        #    (Pooled with B, it would reach back into B's day and get 5 s.)
        #    Sitting at 8 s owns [max(3, 3), 8) = 5 s.
        c = oct_midnight(8) + 8 * 3_600_000
        self.db.add(c, c + 20_000, hr=[(0, 120), (1000, 120), (2000, 120)],
                    predictions=[(3000, "Walking"), (8000, "Sitting")])
        # Live (no ended_at), today: excluded everywhere.
        live = oct_midnight(8) + 10 * 3_600_000
        self.db.add(live, None, hr=[(0, 200)], predictions=[(5000, "Running")])
        # Old: Oct 1, before the 7-day range's Oct 2 midnight.
        old = oct_midnight(1) + 9 * 3_600_000
        self.db.add(old, old + 20_000, hr=[(0, 200)])

    def test_counts_and_total_time(self) -> None:
        # A, B, C: 3 sessions, 20 + 20 + 20 = 60 s.
        self.assertEqual(self.db.run("session_count", range_name="last_7_days").value, 3)
        result = self.db.run("total_time", range_name="last_7_days")
        self.assertEqual((result.value, result.n_sessions), (60.0, 3))

    def test_average_pools_every_sample(self) -> None:
        # (60 + 90 + 100 + 120 + 120 + 120) / 6 = 610 / 6 = 101.666...
        # (Averaging each session's mean instead would give
        # (75 + 100 + 120) / 3 = 98.33.)
        result = self.db.run("avg_hr", range_name="last_7_days")
        self.assertAlmostEqual(result.value, 610 / 6)
        self.assertEqual(result.n_sessions, 3)
        self.assertEqual(self.db.run("peak_hr", range_name="last_7_days").value, 120)

    def test_activity_time_is_worked_out_per_session(self) -> None:
        # Walking 10 (A) + 3 (C) = 13; Running 5 (B); Sitting 5 (C).
        self.assertEqual(self.db.run("time_in_activity", "Walking", "last_7_days").value, 13.0)
        self.assertEqual(self.db.run("time_in_activity", "Running", "last_7_days").value, 5.0)
        self.assertEqual(self.db.run("time_in_activity", "Sitting", "last_7_days").value, 5.0)
        # Standing: there is activity data, just none of it Standing: a real 0.
        standing = self.db.run("time_in_activity", "Standing", "last_7_days")
        self.assertEqual((standing.value, standing.empty), (0.0, False))
        # Any activity: 13 + 5 + 5 = 23.
        self.assertEqual(self.db.run("time_in_activity", None, "last_7_days").value, 23.0)

    def test_activity_filtered_counts_only_contributing_sessions(self) -> None:
        # A, B and C all have activity data, but Walking time comes only
        # from A (10 s) and C (3 s): 2 sessions, not 3. Running only from B:
        # 1. Standing is a real 0 that no session contributed to: 0.
        # Unfiltered, all three contribute: 3.
        cases = [("Walking", 13.0, 2), ("Running", 5.0, 1), ("Sitting", 5.0, 1),
                 ("Standing", 0.0, 0), (None, 23.0, 3)]
        for activity, value, n in cases:
            with self.subTest(activity=activity):
                result = self.db.run("time_in_activity", activity, "last_7_days")
                self.assertEqual((result.value, result.n_sessions), (value, n))

    def test_activity_filtered_heart_rate_counts_only_sessions_with_samples_in_it(self) -> None:
        # Walking windows hold samples in A (60, 90) and C (120 x 3): 2
        # sessions. Running's window in B, [0, 5 s), holds B's 0 s sample: 1.
        # Unfiltered: A, B, C all have samples: 3.
        for metric in ("avg_hr", "peak_hr"):
            with self.subTest(metric=metric):
                self.assertEqual(self.db.run(metric, "Walking", "last_7_days").n_sessions, 2)
                self.assertEqual(self.db.run(metric, "Running", "last_7_days").n_sessions, 1)
                self.assertEqual(self.db.run(metric, None, "last_7_days").n_sessions, 3)

    def test_breakdown_order_breaks_ties_by_first_appearance(self) -> None:
        # Walking 13 first; Running and Sitting tie at 5 s, and Running
        # appeared first (B, Oct 5, before C, Oct 8).
        breakdown = queries.activity_breakdown(
            "last_7_days", path=self.db.path, now_ms=NOW, tz=DENVER
        )
        self.assertEqual(breakdown.items, (("Walking", 13.0), ("Running", 5.0), ("Sitting", 5.0)))
        self.assertEqual(breakdown.n_sessions, 3)
        self.assertFalse(breakdown.simulated)

    def test_heart_rate_during_one_activity(self) -> None:
        # Walking covers A's [0, 10) (60, 90) and C's [0, 3) (120 x 3):
        # (60 + 90 + 360) / 5 = 510 / 5 = 102. Sitting is C's [3, 8): no
        # samples there, so no answer rather than a 0.
        self.assertEqual(self.db.run("avg_hr", "Walking", "last_7_days").value, 102.0)
        sitting = self.db.run("avg_hr", "Sitting", "last_7_days")
        self.assertEqual((sitting.value, sitting.empty), (None, True))

    def test_today_and_all(self) -> None:
        # Today: only C (the live one is left out). All: A, B, C and Old.
        self.assertEqual(self.db.run("session_count", range_name="today").value, 1)
        self.assertEqual(self.db.run("session_count", range_name="all").value, 4)

    def test_recent_sessions_are_finished_ones_newest_first(self) -> None:
        recent = queries.list_recent_sessions(3, path=self.db.path)
        # C (peak 120), B (100), A (90); the live session is left out.
        self.assertEqual([r["peak_bpm"] for r in recent], [120, 100, 90])
        self.assertEqual([r["duration_s"] for r in recent], [20.0, 20.0, 20.0])


# --- E. ranges --------------------------------------------------------------

class ResolveRangeTests(unittest.TestCase):
    def test_today_starts_at_local_midnight(self) -> None:
        # Oct 8 00:00 MDT = 06:00 UTC.
        self.assertEqual(queries.resolve_range("today", NOW, DENVER), (oct_midnight(8), NOW))

    def test_last_7_days_is_seven_calendar_days(self) -> None:
        # Today (Oct 8) and the 6 days before: from Oct 2 00:00 MDT.
        self.assertEqual(
            queries.resolve_range("last_7_days", NOW, DENVER), (oct_midnight(2), NOW)
        )

    def test_last_30_days(self) -> None:
        # Oct 8 minus 29 days = Sep 9; 00:00 MDT = 06:00 UTC.
        self.assertEqual(
            queries.resolve_range("last_30_days", NOW, DENVER), (utc_ms(2026, 9, 9, 6), NOW)
        )

    def test_fall_back_day_is_25_hours(self) -> None:
        # Nov 1 2026: 00:00 MDT = Nov 1 06:00 UTC; clocks fall back at 02:00,
        # so Nov 2 00:00 MST = Nov 2 07:00 UTC. 07:00 - 06:00 the next day
        # = 25 hours.
        nov1, nov2 = utc_ms(2026, 11, 1, 6), utc_ms(2026, 11, 2, 7)
        self.assertEqual(queries.resolve_range("today", nov2 - 1, DENVER)[0], nov1)
        self.assertEqual(queries.resolve_range("today", nov2, DENVER)[0], nov2)
        self.assertEqual(nov2 - nov1, 25 * 3_600_000)

    def test_spring_forward_day_is_23_hours(self) -> None:
        # Mar 8 2026: 00:00 MST = 07:00 UTC; Mar 9 00:00 MDT = 06:00 UTC.
        mar8, mar9 = utc_ms(2026, 3, 8, 7), utc_ms(2026, 3, 9, 6)
        self.assertEqual(queries.resolve_range("today", mar9 - 1, DENVER)[0], mar8)
        self.assertEqual(queries.resolve_range("today", mar9, DENVER)[0], mar9)
        self.assertEqual(mar9 - mar8, 23 * 3_600_000)

    def test_last_7_days_across_the_fall_back(self) -> None:
        # Thu Nov 5 2026 12:00 MST = 19:00 UTC. Seven days back to Fri Oct 30,
        # still daylight time: 00:00 MDT = 06:00 UTC.
        now = utc_ms(2026, 11, 5, 19)
        self.assertEqual(
            queries.resolve_range("last_7_days", now, DENVER), (utc_ms(2026, 10, 30, 6), now)
        )

    def test_all_covers_everything(self) -> None:
        start, end = queries.resolve_range("all", NOW, DENVER)
        self.assertLessEqual(start, 0)
        self.assertGreater(end, NOW * 1000)

    def test_unknown_range_name(self) -> None:
        # Was "yesterday", which is a real range since the AI bar (v1).
        with self.assertRaisesRegex(ValueError, "Unknown range 'fortnight'"):
            queries.resolve_range("fortnight", NOW, DENVER)

    def test_yesterday_is_the_previous_calendar_day(self) -> None:
        # NOW is Thu Oct 8 12:00 MDT: yesterday is Oct 7 00:00 to Oct 8 00:00.
        self.assertEqual(
            queries.resolve_range("yesterday", NOW, DENVER), (oct_midnight(7), oct_midnight(8))
        )

    def test_yesterday_around_midnight(self) -> None:
        # At exactly Oct 8 00:00 it's already Oct 8, so yesterday is Oct 7;
        # 1 ms earlier it's still Oct 7, so yesterday is Oct 6.
        self.assertEqual(
            queries.resolve_range("yesterday", oct_midnight(8), DENVER),
            (oct_midnight(7), oct_midnight(8)),
        )
        self.assertEqual(
            queries.resolve_range("yesterday", oct_midnight(8) - 1, DENVER),
            (oct_midnight(6), oct_midnight(7)),
        )

    def test_this_week_and_last_week(self) -> None:
        # Thu Oct 8: this ISO week began Mon Oct 5 00:00 MDT; last week is
        # Mon Sep 28 00:00 to Mon Oct 5 00:00 (Sep 28 00:00 MDT = 06:00 UTC).
        self.assertEqual(queries.resolve_range("this_week", NOW, DENVER), (oct_midnight(5), NOW))
        self.assertEqual(
            queries.resolve_range("last_week", NOW, DENVER),
            (utc_ms(2026, 9, 28, 6), oct_midnight(5)),
        )

    def test_week_boundary_on_monday_midnight(self) -> None:
        # At exactly Mon Oct 5 00:00 the new week has begun (and is empty so
        # far); 1 ms earlier it's still Sunday, in the week of Sep 28.
        monday = oct_midnight(5)
        self.assertEqual(queries.resolve_range("this_week", monday, DENVER), (monday, monday))
        self.assertEqual(
            queries.resolve_range("last_week", monday, DENVER), (utc_ms(2026, 9, 28, 6), monday)
        )
        self.assertEqual(
            queries.resolve_range("this_week", monday - 1, DENVER),
            (utc_ms(2026, 9, 28, 6), monday - 1),
        )
        self.assertEqual(
            queries.resolve_range("last_week", monday - 1, DENVER),
            (utc_ms(2026, 9, 21, 6), utc_ms(2026, 9, 28, 6)),
        )

    def test_last_week_across_the_fall_back_is_169_hours(self) -> None:
        # Wed Nov 4 2026 12:00 MST = 19:00 UTC. This week: Mon Nov 2 00:00
        # MST = 07:00 UTC. Last week: Mon Oct 26 00:00 MDT = 06:00 UTC, up
        # to Nov 2 07:00 UTC -- 7 days plus the extra fall-back hour.
        now = utc_ms(2026, 11, 4, 19)
        nov2 = utc_ms(2026, 11, 2, 7)
        self.assertEqual(queries.resolve_range("this_week", now, DENVER), (nov2, now))
        start, end = queries.resolve_range("last_week", now, DENVER)
        self.assertEqual((start, end), (utc_ms(2026, 10, 26, 6), nov2))
        self.assertEqual(end - start, 169 * 3_600_000)

    def test_latest_session_is_not_a_time_window(self) -> None:
        with self.assertRaisesRegex(ValueError, "latest_session isn't a time window"):
            queries.resolve_range("latest_session", NOW, DENVER)


class LatestSessionTests(unittest.TestCase):
    """range "latest_session": the most recently started finished session.
    Each session's heart rate is one flat value, so which session was
    picked can be read straight off its average."""

    def setUp(self) -> None:
        self.fixture = FixtureDb(self)
        self.base = oct_midnight(8) + 3_600_000   # Oct 8 01:00 MDT

    def add(self, offset_min: int, bpm: int, finished: bool = True, predictions=()) -> int:
        start = self.base + offset_min * 60_000
        return self.fixture.add(
            start, start + 10_000 if finished else None,
            hr=[(0, bpm), (5000, bpm)], predictions=predictions,
        )

    def test_no_sessions_is_empty(self) -> None:
        result = self.fixture.run("avg_hr", range_name="latest_session")
        self.assertEqual((result.value, result.n_sessions, result.empty), (None, 0, True))

    def test_one_session(self) -> None:
        self.add(0, 90)
        result = self.fixture.run("avg_hr", range_name="latest_session")
        self.assertEqual((result.value, result.n_sessions), (90.0, 1))
        self.assertEqual(self.fixture.run("session_count", range_name="latest_session").value, 1)

    def test_several_sessions_picks_the_latest_start(self) -> None:
        # Added out of order: the 30-minute one started last, so it's the
        # one used (80 bpm), not the one added last (70 bpm).
        self.add(10, 70)
        self.add(30, 80)
        self.add(20, 60)
        self.assertEqual(self.fixture.run("avg_hr", range_name="latest_session").value, 80.0)

    def test_same_start_picks_the_higher_id(self) -> None:
        self.add(10, 70)
        self.add(10, 75)
        self.assertEqual(self.fixture.run("avg_hr", range_name="latest_session").value, 75.0)

    def test_a_live_session_is_left_out(self) -> None:
        # The still-recording session started later, but only finished
        # sessions count, so the finished 90 bpm one is the latest.
        self.add(0, 90)
        self.add(60, 140, finished=False)
        self.assertEqual(self.fixture.run("avg_hr", range_name="latest_session").value, 90.0)

    def test_an_activity_means_that_activity_within_the_latest_session(self) -> None:
        # The earlier session was all Running at 150 bpm; the latest one has
        # Walking [0, 5 s) then Running [5, 10 s). Running in the latest
        # session is its 5 s sample only (90 bpm), not the earlier run.
        self.add(0, 150, predictions=[(5000, "Running"), (10000, "Running")])
        start = self.base + 30 * 60_000
        self.fixture.add(
            start, start + 10_000, hr=[(0, 100), (5000, 90)],
            predictions=[(5000, "Walking"), (10000, "Running")],
        )
        self.assertEqual(
            self.fixture.run("avg_hr", "Running", range_name="latest_session").value, 90.0
        )
        breakdown = queries.activity_breakdown("latest_session", path=self.fixture.path)
        self.assertEqual(breakdown.items, (("Walking", 5.0), ("Running", 5.0)))

    def test_a_session_at_midnight_belongs_to_the_new_day(self) -> None:
        fixture = FixtureDb(self)
        fixture.add(oct_midnight(8), oct_midnight(8) + 1000)          # exactly midnight
        fixture.add(oct_midnight(8) - 1, oct_midnight(8) + 1000)      # 1 ms before
        # Today has only the midnight one; the last 7 days have both.
        self.assertEqual(fixture.run("session_count", range_name="today").value, 1)
        self.assertEqual(fixture.run("session_count", range_name="last_7_days").value, 2)

    def test_all_counts_sessions_at_any_time(self) -> None:
        fixture = FixtureDb(self)
        fixture.add(0, 1000)
        fixture.add(NOW * 10, NOW * 10 + 1000)  # far in the future
        self.assertEqual(fixture.run("session_count", range_name="all").value, 2)


# --- F. validation ----------------------------------------------------------

class ValidationTests(unittest.TestCase):
    def test_invalid_requests_are_refused_in_plain_english(self) -> None:
        cases = [
            (MetricRequest("mean_hr", None, "today"), "Unknown metric 'mean_hr'"),
            (MetricRequest("avg_hr", "Swimming", "today"), "Unknown activity 'Swimming'"),
            (MetricRequest("avg_hr", None, None), "give a range or a session id"),
            (MetricRequest("avg_hr", None, "today", 3), "not both"),
            (MetricRequest("avg_hr", None, "fortnight"), "Unknown range 'fortnight'"),
            (MetricRequest("session_count", "Running", "today"), "can't be narrowed"),
            (MetricRequest("total_time", "Walking", "today"), "can't be narrowed"),
        ]
        for request, message in cases:
            with self.subTest(request=request):
                with self.assertRaisesRegex(ValueError, message):
                    request.validate()
                with self.assertRaisesRegex(ValueError, message):
                    queries.run(request, path="/nonexistent/never-opened.db")

    def test_valid_requests_pass(self) -> None:
        MetricRequest("avg_hr", "Running", "last_7_days").validate()
        MetricRequest("time_in_activity", None, None, 1).validate()

    def test_unknown_session_id(self) -> None:
        fixture = FixtureDb(self)
        with self.assertRaisesRegex(ValueError, "no session with id 99"):
            fixture.run("avg_hr", session_id=99)


# --- G. structural guard -------------------------------------------------------

# Why this exists: if SQL computed a user-facing number (AVG, SUM, MAX,
# MIN) in one place and data/stats.py computed it in another, the two could
# quietly drift apart -- the Dashboard and the session summary showing
# different "average heart rates" for the same data. So SQL may only
# filter, order, join and COUNT(*); every number is computed by a function
# in data/stats.py. This test reads the SQL text in the query layer, the
# database module and every UI file, and fails if an aggregate shows up.
SQL_AGGREGATE = re.compile(r"\b(AVG|SUM|MAX|MIN)\s*\(", re.IGNORECASE)
SCANNED_FILES = [APP_DIR / "data" / "queries.py", APP_DIR / "data" / "db.py",
                 *sorted((APP_DIR / "ui").glob("*.py")),
                 *sorted((APP_DIR / "ask").glob("*.py"))]
# (file name, matched text) -> why it's allowed. Empty: nothing in these
# files needs a SQL aggregate today.
ALLOWED_AGGREGATES: dict[tuple[str, str], str] = {}


def string_literals(source: str) -> list[str]:
    """Every string literal in the source (SQL lives in these), except
    docstrings, which may describe the rule. Python calls like max(a, b)
    aren't strings, so they're never flagged."""
    tree = ast.parse(source)
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                docstrings.add(id(body[0].value))
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        and id(node) not in docstrings
    ]


def aggregates_in(source: str) -> list[str]:
    return [m.group(0) for text in string_literals(source) for m in SQL_AGGREGATE.finditer(text)]


class NoSqlArithmeticTests(unittest.TestCase):
    def test_no_sql_aggregates_outside_stats(self) -> None:
        for path in SCANNED_FILES:
            with self.subTest(file=path.name):
                found = [
                    hit for hit in aggregates_in(path.read_text())
                    if (path.name, hit) not in ALLOWED_AGGREGATES
                ]
                self.assertEqual(found, [], f"SQL arithmetic in {path.name}; use data/stats.py")

    def test_the_ai_bar_package_is_scanned(self) -> None:
        names = {(p.parent.name, p.name) for p in SCANNED_FILES}
        for name in ("pipeline.py", "rules.py", "answers.py"):
            self.assertIn(("ask", name), names)

    def test_the_scanner_catches_sql_and_ignores_python(self) -> None:
        source = (
            'def f():\n'
            '    """Mentions AVG(bpm) in a docstring, which is fine."""\n'
            '    q = "SELECT avg(bpm) FROM hr_samples"\n'
            '    m = f"SELECT MAX (bpm) FROM t WHERE x = {1}"\n'
            '    return max(1, 2), min(3, 4), sum([5])\n'
        )
        self.assertEqual(aggregates_in(source), ["avg(", "MAX ("])


# --- H. empty states ------------------------------------------------------------

class EmptyDataTests(unittest.TestCase):
    def test_an_empty_database_answers_empty_everywhere(self) -> None:
        fixture = FixtureDb(self)
        for metric in queries.METRIC_UNITS:
            for range_name in queries.RANGE_NAMES:
                with self.subTest(metric=metric, range=range_name):
                    result = fixture.run(metric, range_name=range_name)
                    self.assertTrue(result.empty)
                    self.assertIsNone(result.value)  # never a 0 passed off as real
                    self.assertEqual(result.n_sessions, 0)
        breakdown = queries.activity_breakdown("all", path=fixture.path)
        self.assertEqual((breakdown.items, breakdown.simulated), ((), False))
        self.assertEqual(queries.list_recent_sessions(3, path=fixture.path), [])

    def test_sessions_without_predictions_have_heart_rate_but_no_breakdown(self) -> None:
        fixture = FixtureDb(self)
        # HR 70, 80: average (70 + 80) / 2 = 75, peak 80.
        start = oct_midnight(8) + 3_600_000
        fixture.add(start, start + 60_000, hr=[(0, 70), (1000, 80)])
        self.assertEqual(fixture.run("avg_hr", range_name="today").value, 75.0)
        self.assertEqual(fixture.run("peak_hr", range_name="today").value, 80)
        activity = fixture.run("time_in_activity", range_name="today")
        self.assertEqual((activity.value, activity.empty), (None, True))
        breakdown = queries.activity_breakdown("today", path=fixture.path, now_ms=NOW, tz=DENVER)
        self.assertEqual((breakdown.items, breakdown.n_sessions), ((), 0))


class SimulatedPredictionTests(unittest.TestCase):
    def test_run_flags_simulated_only_when_activity_windows_are_used(self) -> None:
        # One session today, Walking [0, 5 s) and [5, 10 s), samples at 0 and
        # 5 s. With the dummy classifier, anything that uses the windows
        # (an activity filter, or time in an activity) is simulated; plain
        # heart rate, session count and total time never are. With a real
        # model version, nothing is.
        start = oct_midnight(8) + 3_600_000
        cases = [
            ("avg_hr", "Walking", True), ("peak_hr", "Walking", True),
            ("time_in_activity", "Walking", True), ("time_in_activity", None, True),
            ("avg_hr", None, False), ("peak_hr", None, False),
            ("session_count", None, False), ("total_time", None, False),
        ]
        for model_version, dummy in (("demo-dummy", True), ("cnn-v1", False)):
            fixture = FixtureDb(self)
            fixture.add(start, start + 10_000, hr=[(0, 90), (5000, 100)],
                        predictions=[(5000, "Walking"), (10000, "Walking")],
                        model_version=model_version)
            for metric, activity, uses_windows in cases:
                with self.subTest(model_version=model_version, metric=metric, activity=activity):
                    result = fixture.run(metric, activity, range_name="today")
                    self.assertFalse(result.empty)
                    self.assertEqual(result.simulated, dummy and uses_windows)

    def test_run_ignores_dummy_sessions_that_fed_nothing(self) -> None:
        # Two sessions: a dummy one with no Running, and a real one with
        # Running. Heart rate while Running comes only from the real one, so
        # it isn't simulated -- the same "only sessions that fed it" rule as
        # the breakdown's.
        start = oct_midnight(8) + 3_600_000
        fixture = FixtureDb(self)
        fixture.add(start, start + 10_000, hr=[(0, 90)], predictions=[(5000, "Walking")],
                    model_version="demo-dummy")
        fixture.add(start + 60_000, start + 70_000, hr=[(0, 130)],
                    predictions=[(5000, "Running")], model_version="cnn-v1")
        result = fixture.run("avg_hr", "Running", range_name="today")
        self.assertEqual((result.value, result.n_sessions, result.simulated), (130.0, 1, False))

    def test_simulated_only_when_dummy_predictions_feed_the_breakdown(self) -> None:
        start = oct_midnight(8) + 3_600_000
        for model_version, expected in (("demo-dummy", True), ("cnn-v1", False)):
            with self.subTest(model_version=model_version):
                fixture = FixtureDb(self)
                fixture.add(start, start + 10_000, predictions=[(5000, "Walking")],
                            model_version=model_version)
                breakdown = queries.activity_breakdown(
                    "today", path=fixture.path, now_ms=NOW, tz=DENVER
                )
                self.assertEqual(breakdown.simulated, expected)


# --- the Sessions table --------------------------------------------------------

class SessionsListTests(unittest.TestCase):
    def setUp(self) -> None:
        self.db = FixtureDb(self)
        # Oct 3: HR 60, 90 -> avg (60 + 90) / 2 = 75, peak 90.
        a = oct_midnight(3) + 9 * 3_600_000
        self.a = self.db.add(a, a + 20_000, hr=[(0, 60), (1000, 90)], label="Walking")
        # Today, still recording: listed (unlike on the Dashboard), its
        # duration running up to now: 12:00 - 10:00 = 7200 s.
        live = oct_midnight(8) + 10 * 3_600_000
        self.live = self.db.add(live, None, hr=[(0, 100)])
        # Oct 1, 23:59:59.999 the night before the 7-day range: excluded.
        self.old = self.db.add(oct_midnight(2) - 1, oct_midnight(2) + 1000, label="Running")

    def rows(self, range_name: str, activity: str | None = None) -> list[dict]:
        return queries.list_sessions(
            50, activity, range_name, path=self.db.path, now_ms=NOW, tz=DENVER
        )

    def test_last_7_days_uses_the_same_calendar_days_as_the_dashboard(self) -> None:
        rows = self.rows("last_7_days")
        self.assertEqual([r["id"] for r in rows], [self.live, self.a])
        self.assertEqual((rows[1]["avg_bpm"], rows[1]["peak_bpm"]), (75.0, 90))
        self.assertEqual((rows[0]["duration_s"], rows[0]["ended_at"]), (7200.0, None))
        self.assertEqual([r["id"] for r in self.rows("all")], [self.live, self.a, self.old])

    def test_activity_filter_matches_the_session_label(self) -> None:
        self.assertEqual([r["id"] for r in self.rows("all", "Running")], [self.old])

    def test_a_session_without_heart_rate_has_no_average(self) -> None:
        self.assertEqual(
            (self.rows("all")[2]["avg_bpm"], self.rows("all")[2]["peak_bpm"]), (None, None)
        )


if __name__ == "__main__":
    unittest.main()
