"""Tests for explicit Start/Stop recording (handoff 8).

Plain-language overview: the Live Monitor used to open a database session
the moment the app launched and write dummy data until it closed. Now it
shows a live preview that is never saved, and only the Start button opens
a session. These tests build the real main window offscreen on a
temporary database (db.DB_PATH patched, see test_db_path.py), so the real
database is never opened.

Apart from test A, which lets the default source run on its real timers
for a moment, every test drives a DummySource by hand: a fake clock that
the source and the recording controller share, the source's timer off,
and poll() called after each 100 ms step. So "30 seconds" of recording
takes no real time and gives the same rows every run. Expected counts
are worked out from the schedule (HR every 1000 ms from the source's
start, IMU every 100 ms from start + 100, predictions every 5000 ms,
restarted at Start), not copied from a run.

    python -m unittest heartwatchapp.tests.test_recording
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import contextlib  # noqa: E402
import random  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import unittest  # noqa: E402

from PySide6.QtCore import QEvent, QObject, Signal  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication, QLabel, QTableWidget  # noqa: E402

from .. import main as app_main  # noqa: E402
from ..data import db, stats  # noqa: E402
from ..data.writer import DatabaseWriter  # noqa: E402
from ..ml.inference import CLASSES  # noqa: E402
from ..recording.controller import PREVIEW, RECORDING  # noqa: E402
from ..sources.base import HrReading, ImuReading, PredictionReading  # noqa: E402
from ..sources.dummy import DummySource  # noqa: E402
from ..ui import live_monitor  # noqa: E402
from ..ui.dashboard import SessionRow  # noqa: E402
from ..ui.main_window import HeartWatchApp  # noqa: E402
from ..ui.theme import Theme  # noqa: E402
from .test_db_path import TempDbCase  # noqa: E402

EMPTY_COUNTS = {"sessions": 0, "imu_samples": 0, "hr_samples": 0, "predictions": 0}
# Two minutes before the real now, on a whole second, so sessions land in
# the Dashboard's "last 7 days" and the arithmetic below stays readable.
T0 = (int(time.time()) - 120) * 1000


class FakeClock:
    def __init__(self, now_ms: int):
        self.now = now_ms

    def __call__(self) -> int:
        return self.now


class RecordingCase(TempDbCase):
    """A temporary database (db.DB_PATH patched) and a running writer."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        super().setUp()
        db.init_db()
        self.writer = DatabaseWriter()
        self.writer.start()
        self.addCleanup(self.writer.stop)

    def counts(self) -> dict[str, int]:
        self.writer.flush_and_wait()
        return db.get_db_info()["row_counts"]

    def window(self, **kwargs) -> HeartWatchApp:
        window = HeartWatchApp(Theme(mode="dark"), self.writer, **kwargs)
        self.addCleanup(window.deleteLater)
        self.addCleanup(window.shutdown)
        return window

    def sim_window(self) -> HeartWatchApp:
        """The main window on a hand-driven DummySource, clock at T0. The
        source starts with the window, so its first HR reading (at T0) is
        already shown, as a preview."""
        self.clock = FakeClock(T0)
        self.source = DummySource(clock=self.clock, rng=random.Random(8), use_timer=False)
        window = self.window(source=self.source, clock=self.clock)
        self.controller = window.recording
        return window

    def run_ms(self, ms: int) -> None:
        for _ in range(ms // 100):
            self.clock.now += 100
            self.source.poll()

    def sessions(self) -> list[dict]:
        self.writer.flush_and_wait()
        with contextlib.closing(db._connect()) as conn:
            return [dict(r) for r in conn.execute("SELECT * FROM sessions ORDER BY id")]

    def timeline(self, session_id: int) -> dict:
        self.writer.flush_and_wait()
        return db.get_session_timeline(session_id)


class LaunchTests(RecordingCase):
    def test_a_launch_creates_nothing(self) -> None:
        window = self.window()
        hr, imu = [], []
        window.recording.hr_received.connect(hr.extend)
        window.recording.imu_received.connect(imu.extend)
        # Long enough for the default source's real timers to show data;
        # then, on a slow machine, up to 5 s more for the first readings.
        QTest.qWait(1200)
        for _ in range(50):
            if hr and imu:
                break
            QTest.qWait(100)
        # The source really ran (the first HR, emitted while the window was
        # being built, isn't counted here) -- and none of it was saved.
        self.assertGreaterEqual(len(hr), 1)
        self.assertGreaterEqual(len(imu), 1)
        self.assertEqual(self.counts(), EMPTY_COUNTS)
        self.assertNotEqual(window.live.hr_number.text(), "--")

    def test_a_simulated_preview_saves_nothing(self) -> None:
        window = self.sim_window()
        hr, imu = [], []
        self.controller.hr_received.connect(hr.extend)
        self.controller.imu_received.connect(imu.extend)
        self.run_ms(30_000)
        # After the HR at T0 (emitted at construction): HR T0+1000 ..
        # T0+30000 = 30, IMU T0+100 .. T0+30000 = 300.
        self.assertEqual((len(hr), len(imu)), (30, 300))
        self.assertEqual(self.counts(), EMPTY_COUNTS)
        # ...while the screen showed all of it.
        self.assertNotEqual(window.live.hr_number.text(), "--")
        self.assertNotEqual(window.live.confidence_label.text(), "Confidence: --")
        self.assertEqual(window.live.status_label.text(), live_monitor.PREVIEW_TEXT)


class StartStopTests(RecordingCase):
    def test_b_start_then_stop_saves_exactly_the_recording(self) -> None:
        self.sim_window()
        emitted = {"hr": [], "imu": [], "pred": []}
        self.controller.hr_received.connect(lambda rs: emitted["hr"].extend(rs))
        self.controller.imu_received.connect(lambda rs: emitted["imu"].extend(rs))
        self.controller.prediction_received.connect(emitted["pred"].append)

        self.run_ms(3_000)
        sid = self.controller.start("Walking")
        started = T0 + 3_000
        recorded_from = {k: len(v) for k, v in emitted.items()}
        self.run_ms(10_000)
        self.controller.stop()

        [session] = self.sessions()
        self.assertEqual(session["id"], sid)
        self.assertEqual(session["started_at"], started)
        self.assertEqual(session["ended_at"], T0 + 13_000)
        self.assertLess(session["started_at"], session["ended_at"])
        timeline = self.timeline(sid)
        # HR at T0+3000 was shown in preview, just before Start: saved are
        # T0+4000 .. T0+13000 = 10. IMU: T0+3100 .. T0+13000 = 100.
        # Predictions restart at Start: T0+8000 and T0+13000 = 2.
        self.assertEqual(len(timeline["hr"]), 10)
        self.assertEqual(len(timeline["imu"]), 100)
        self.assertEqual([ts for ts, _, _ in timeline["predictions"]],
                         [T0 + 8_000, T0 + 13_000])
        # ...which is everything the source emitted while recording.
        self.assertEqual(len(timeline["hr"]), len(emitted["hr"]) - recorded_from["hr"])
        self.assertEqual(len(timeline["imu"]), len(emitted["imu"]) - recorded_from["imu"])
        self.assertEqual(len(timeline["predictions"]),
                         len(emitted["pred"]) - recorded_from["pred"])
        for series in ("hr", "imu", "predictions"):
            for row in timeline[series]:
                self.assertTrue(started <= row[0] <= session["ended_at"], (series, row))
        self.assertEqual({row[0] for row in timeline["hr"]},
                         {T0 + 1_000 * k for k in range(4, 14)})

    def test_b_stored_values_match_what_was_shown(self) -> None:
        self.sim_window()
        shown_hr: list[HrReading] = []
        self.controller.start(None)
        self.controller.hr_received.connect(shown_hr.extend)
        self.run_ms(5_000)
        self.controller.stop()
        sid = self.sessions()[0]["id"]
        self.assertEqual(self.timeline(sid)["hr"], [(r.ts_ms, r.bpm) for r in shown_hr])

    def test_c_label_comes_from_the_picker(self) -> None:
        window = self.sim_window()
        live = window.live
        self.assertEqual(live.activity_picker.itemText(0), live_monitor.NO_LABEL_TEXT)
        self.assertEqual([live.activity_picker.itemText(i) for i in range(1, 5)], CLASSES)
        for index in range(live.activity_picker.count()):
            live.activity_picker.setCurrentIndex(index)
            live.start_stop_button.click()
            self.assertEqual(live.start_stop_button.text(), live_monitor.STOP_TEXT)
            self.assertFalse(live.activity_picker.isEnabled())
            self.run_ms(1_000)
            live.start_stop_button.click()
            self.assertEqual(live.start_stop_button.text(), live_monitor.START_TEXT)
            self.assertTrue(live.activity_picker.isEnabled())
        self.assertEqual([s["label"] for s in self.sessions()], [None, *CLASSES])

    def test_d_readings_before_start_are_never_saved(self) -> None:
        self.sim_window()
        self.run_ms(3_000)
        started = T0 + 3_000
        sid = self.controller.start(None)
        # Readings arriving after Start but timed before it -- e.g. a late
        # Bluetooth packet -- are preview-only; at exactly started_at they
        # count.
        self.controller.on_hr([HrReading(started - 1, 70), HrReading(started, 71)])
        self.controller.on_imu([ImuReading(started - 1, *[1] * 6),
                                ImuReading(started, *[2] * 6)])
        # A prediction's whole window must be inside the recording.
        self.controller.on_prediction(PredictionReading(started + 4_999, "Sitting", 0.9, "t"))
        self.controller.on_prediction(PredictionReading(started + 5_000, "Running", 0.9, "t"))
        self.controller.stop()

        timeline = self.timeline(sid)
        self.assertEqual(timeline["hr"], [(started, 71)])
        self.assertEqual(timeline["imu"], [(started, *[2] * 6)])
        self.assertEqual(timeline["predictions"], [(started + 5_000, "Running", 0.9)])
        # The clock still reads `started`, but a saved prediction is at
        # started + 5000, so the session ends there.
        self.assertEqual(self.sessions()[0]["ended_at"], started + 5_000)

    def test_d_stop_straight_after_start_saves_no_samples(self) -> None:
        self.sim_window()
        self.run_ms(3_000)
        self.controller.start(None)
        self.controller.stop()
        counts = self.counts()
        self.assertEqual(counts, {**EMPTY_COUNTS, "sessions": 1})
        [session] = self.sessions()
        self.assertEqual(session["started_at"], session["ended_at"])

    def test_e_prediction_schedule_restarts_at_start(self) -> None:
        self.sim_window()
        self.run_ms(3_000)
        sid = self.controller.start("Walking")
        started = T0 + 3_000
        self.run_ms(30_000)
        self.controller.stop()
        ended = T0 + 33_000

        predictions = self.timeline(sid)["predictions"]
        self.assertEqual([ts for ts, _, _ in predictions],
                         [started + 5_000 * k for k in range(1, 7)])
        self.assertGreaterEqual(predictions[0][0] - stats.PREDICTION_WINDOW_MS, started)
        seconds = stats.time_per_activity(predictions, started, ended)
        # Six whole 5-second windows inside a 30-second session.
        self.assertEqual(sum(seconds.values()), 30.0)
        self.assertLessEqual(sum(seconds.values()), stats.session_duration_s(started, ended))


class CrashRecoveryTests(RecordingCase):
    def add(self, conn, started_at, ended_at=None, notes=None, hr=(), imu=(), preds=()) -> int:
        sid = db.start_session(conn, "demo", None, 1.0, 1.0, started_at=started_at)
        db.insert_hr_batch(conn, sid, [(ts, 80, None) for ts in hr])
        db.insert_imu_batch(conn, sid, [(ts, 0, 0, 0, 0, 0, 0) for ts in imu])
        for ts in preds:
            db.insert_prediction(conn, sid, ts, "Walking", 0.9, "demo-dummy")
        if ended_at is not None:
            db.end_session(conn, sid, ended_at)
        conn.execute("UPDATE sessions SET notes = ? WHERE id = ?", (notes, sid))
        return sid

    def test_f_orphans_are_closed_and_nothing_is_deleted(self) -> None:
        with contextlib.closing(db._connect()) as conn, conn:
            # Open, with samples: the newest is the IMU row at 12_345.
            open_with = self.add(conn, 1_000, hr=(2_000, 10_000), imu=(1_100, 12_345),
                                 preds=(6_000, 11_000))
            # Open, no samples, with a note of its own.
            open_empty = self.add(conn, 50_000, notes="Walked the dog")
            finished = self.add(conn, 90_000, ended_at=99_000, hr=(91_000,))
        before = self.counts()

        self.assertEqual(self.writer.close_orphan_sessions(), 2)

        rows = {s["id"]: s for s in self.sessions()}
        self.assertEqual(rows[open_with]["ended_at"], 12_345)
        self.assertEqual(rows[open_with]["notes"], db.ORPHAN_NOTE)
        self.assertEqual(rows[open_empty]["ended_at"], 50_000)
        self.assertEqual(rows[open_empty]["notes"], "Walked the dog " + db.ORPHAN_NOTE)
        self.assertEqual(rows[finished]["ended_at"], 99_000)
        self.assertIsNone(rows[finished]["notes"])
        self.assertEqual(self.counts(), before)
        # Running it again finds nothing to do.
        self.assertEqual(self.writer.close_orphan_sessions(), 0)


class QuitTests(RecordingCase):
    def test_g_quitting_mid_recording_saves_the_session(self) -> None:
        class FakeApp(QObject):
            about_to_quit = Signal()

        window = self.sim_window()
        sid = self.controller.start("Sitting")
        # Not flushed yet: the writer batches for up to a second.
        self.run_ms(2_000)
        fake = FakeApp()
        app_main.connect_shutdown(fake.about_to_quit, window, self.writer, None)
        fake.about_to_quit.emit()

        # The writer has stopped, so read the file directly.
        [session] = self.sessions_after_quit()
        self.assertEqual(session["id"], sid)
        self.assertEqual(session["ended_at"], T0 + 2_000)
        timeline = db.get_session_timeline(sid)
        # Start at T0, just after the preview's HR at T0: HR T0+1000 and
        # T0+2000; IMU T0+100 .. T0+2000; no 5-second window finished yet.
        self.assertEqual(len(timeline["hr"]), 2)
        self.assertEqual(len(timeline["imu"]), 20)
        self.assertEqual(timeline["predictions"], [])
        self.assertEqual(self.controller.state, PREVIEW)

    def sessions_after_quit(self) -> list[dict]:
        with contextlib.closing(db._connect()) as conn:
            return [dict(r) for r in conn.execute("SELECT * FROM sessions ORDER BY id")]


class MisuseTests(RecordingCase):
    def test_h_double_start_stop_in_preview_and_rapid_toggling(self) -> None:
        window = self.sim_window()
        self.assertIsNone(self.controller.stop())
        self.assertEqual(self.counts(), EMPTY_COUNTS)

        first = self.controller.start("Walking")
        self.assertIsNone(self.controller.start("Running"))
        self.assertEqual(self.controller.session_id, first)
        self.run_ms(1_000)
        self.controller.stop()
        self.assertIsNone(self.controller.stop())

        button = window.live.start_stop_button
        for _ in range(5):  # Start, Stop, Start, Stop, Start
            button.click()
        self.run_ms(500)
        button.click()  # Stop

        sessions = self.sessions()
        self.assertEqual(len(sessions), 4)
        self.assertEqual(len({s["id"] for s in sessions}), 4)
        self.assertEqual(sessions[0]["label"], "Walking")
        for s in sessions:
            self.assertIsNotNone(s["ended_at"])
            self.assertLessEqual(s["started_at"], s["ended_at"])
        self.assertEqual(self.controller.state, PREVIEW)


class OtherViewsTests(RecordingCase):
    def flush_deletes(self) -> None:
        QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    def test_k_dashboard_and_sessions_refresh_after_stop(self) -> None:
        window = self.sim_window()
        self.assertEqual(window.dashboard.findChildren(SessionRow), [])
        self.controller.start("Walking")
        self.run_ms(10_000)
        self.flush_deletes()
        self.assertEqual(window.dashboard.findChildren(SessionRow), [])  # still recording

        self.controller.stop()
        self.flush_deletes()
        [row] = window.dashboard.findChildren(SessionRow)
        self.assertIn("Walking", [lbl.text() for lbl in row.findChildren(QLabel)])
        [table] = window.sessions.findChildren(QTableWidget)
        self.assertEqual(table.rowCount(), 1)
        self.assertEqual(table.item(0, 2).text(), "Walking")
        self.assertEqual(window.stack.currentIndex(), 0)


class ThemeToggleTests(RecordingCase):
    def setUp(self) -> None:
        super().setUp()
        stylesheet = self.app.styleSheet()
        self.addCleanup(self.app.setStyleSheet, stylesheet)

    def test_l_toggling_the_theme_keeps_the_recording(self) -> None:
        window = self.sim_window()
        live_picker_index = window.live.activity_picker.findData("Running")
        window.live.activity_picker.setCurrentIndex(live_picker_index)
        window.live.start_stop_button.click()
        sid = self.controller.session_id
        self.run_ms(3_000)

        old_live = window.live
        window._toggle_theme()
        self.assertIsNot(window.live, old_live)
        self.assertEqual(self.controller.state, RECORDING)
        self.assertEqual(self.controller.session_id, sid)
        # The rebuilt view shows the recording in progress.
        self.assertEqual(window.live.status_label.text(), "● Recording 0:03")
        self.assertEqual(window.live.start_stop_button.text(), live_monitor.STOP_TEXT)
        self.assertFalse(window.live.activity_picker.isEnabled())
        self.assertEqual(window.live.activity_picker.currentData(), "Running")

        self.run_ms(3_000)
        self.assertNotEqual(window.live.hr_number.text(), "--")
        window.live.start_stop_button.click()

        [session] = self.sessions()
        self.assertEqual((session["id"], session["label"]), (sid, "Running"))
        self.assertEqual(session["ended_at"], T0 + 6_000)
        hr = self.timeline(sid)["hr"]
        # T0+1000 .. T0+6000, before and after the toggle alike.
        self.assertEqual([ts for ts, _ in hr], [T0 + 1_000 * k for k in range(1, 7)])

    def test_l_toggling_the_theme_in_preview_creates_nothing(self) -> None:
        window = self.sim_window()
        self.run_ms(2_000)
        window._toggle_theme()
        self.run_ms(2_000)
        self.assertEqual(self.counts(), EMPTY_COUNTS)
        self.assertEqual(window.live.status_label.text(), live_monitor.PREVIEW_TEXT)


class ThreadTests(RecordingCase):
    def test_readings_from_another_thread_arrive_on_the_main_thread(self) -> None:
        self.sim_window()
        received = []
        self.controller.hr_received.connect(
            lambda rs: received.append((rs, threading.current_thread()))
        )
        worker = threading.Thread(target=self.controller.on_hr, args=([HrReading(T0, 72)],))
        worker.start()
        worker.join()
        self.assertEqual(received, [])  # queued until the main thread runs events
        self.app.processEvents()
        self.assertEqual(received, [([HrReading(T0, 72)], threading.main_thread())])


class ElapsedFormatTests(unittest.TestCase):
    def test_format_elapsed(self) -> None:
        self.assertEqual(live_monitor.format_elapsed(0), "0:00")
        self.assertEqual(live_monitor.format_elapsed(192_999), "3:12")
        self.assertEqual(live_monitor.format_elapsed(3_723_000), "1:02:03")


if __name__ == "__main__":
    unittest.main()
