"""Tests for what happens when the database writer fails (handoff 11).

Plain-language overview: one background thread (data/writer.py) does all
writes to the database. It used to die silently if saving a batch of
samples raised an error -- a full disk, or rows for a session that had
just been deleted -- and every later call that waits for the writer
(Stop, quitting while recording, the developer panel) then waited
forever, freezing the app.

Now the first such error puts the writer into a "failed" state for the
rest of the run: it logs the error once, answers every later request
straight away with WriterFailed, and still stops when asked. Waiting
calls also give up after a few seconds (WriterUnavailable) if the
thread is stuck or gone, so quitting never takes more than about 4 s.
The recording controller shows the failure as an "error" state, and the
developer panel won't delete the session being recorded.

Every call that could block runs in a daemon side thread joined with a
timeout (call_bounded below), so a hang shows up as a test failure, never
as a test run that never finishes. Every test uses a temporary database
(db.DB_PATH patched, see test_db_path.py); the real database's files are
hashed before and after, never opened.

    python -m unittest heartwatchapp.tests.test_writer_failures
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import contextlib  # noqa: E402
import logging  # noqa: E402
import sqlite3  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import unittest  # noqa: E402
from unittest import mock  # noqa: E402

from .. import main as app_main  # noqa: E402
from ..data import db  # noqa: E402
from ..data import writer as writer_mod  # noqa: E402
from ..data.writer import DatabaseWriter  # noqa: E402
from ..ui import live_monitor  # noqa: E402
from ..ui import settings as settings_mod  # noqa: E402
from ..ui.main_window import HeartWatchApp  # noqa: E402
from ..ui.theme import Theme  # noqa: E402
from .test_db_path import TempDbCase, real_db_hashes  # noqa: E402
from .test_recording import RecordingCase  # noqa: E402

# How long a test waits for a call that should come back. Long enough to
# cover the writer's 2 s quit timeouts with room to spare, short enough
# that a hang on the old code fails in seconds.
BOUND_S = 3.0
# The writer's quit-path timeouts (end_session, stop): 2.0 s each.
QUIT_S = 2.0
# "Fails fast": an answer from an already-failed writer, no waiting.
FAST_S = 0.5

ERROR_STATE = "error"  # recording/controller.py's ERROR


def call_bounded(fn, *args, timeout: float = BOUND_S, **kwargs):
    """Runs fn(*args, **kwargs) in a daemon thread and waits at most
    `timeout` seconds. Returns (outcome, value, seconds): outcome is "ok"
    (value is the return value), "raised" (value is the exception) or
    "hung" (value None; the daemon thread is abandoned)."""
    box: dict = {}

    def run() -> None:
        try:
            box["ok"] = fn(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001 - reported to the test
            box["raised"] = exc

    t0 = time.monotonic()
    thread = threading.Thread(target=run, name="bounded-call", daemon=True)
    thread.start()
    thread.join(timeout)
    took = time.monotonic() - t0
    if thread.is_alive():
        return "hung", None, took
    if "raised" in box:
        return "raised", box["raised"], took
    return "ok", box.get("ok"), took


@contextlib.contextmanager
def db_error_logs(name: str = "heartwatch.db"):
    """Collects ERROR records from the `name` logger (kept off the
    terminal) into a list the test checks AFTER its hang check -- unlike
    assertLogs, which would fail on exit before a hang is reported."""
    records: list[logging.LogRecord] = []
    handler = logging.Handler(logging.ERROR)
    handler.emit = records.append
    logger = logging.getLogger(name)
    logger.addHandler(handler)
    propagate, logger.propagate = logger.propagate, False
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.propagate = propagate


def boom(*_args, **_kwargs):
    raise sqlite3.OperationalError("database or disk is full")


def assert_returns_error(case, outcome, value, took, error_name, within=BOUND_S):
    """outcome/value/took from call_bounded: the call came back (did not
    hang) within `within` s, by raising writer_mod.<error_name>. Checked
    in that order, so on code without the fix a hang fails first."""
    case.assertNotEqual(outcome, "hung", f"call still blocked after {took:.1f} s")
    case.assertLess(took, within)
    case.assertEqual(outcome, "raised", f"expected {error_name}, got {value!r}")
    case.assertIsInstance(value, getattr(writer_mod, error_name))
    case.assertIsInstance(value, writer_mod.WriterError)


class WriterCase(TempDbCase):
    """A temporary database and a running writer, with the real database's
    files checked unchanged at the end."""

    def setUp(self) -> None:
        super().setUp()
        before = real_db_hashes()
        self.addCleanup(lambda: self.assertEqual(real_db_hashes(), before))
        db.init_db()
        self.writer = DatabaseWriter()
        self.writer.start()
        self.addCleanup(lambda: call_bounded(self.writer.stop))


class FlushFailureTests(WriterCase):
    def fail_a_flush(self):
        """Opens a session, makes the next flush raise, queues one HR row,
        and calls end_session (which flushes first)."""
        sid = self.writer.start_session("fixture", None, 1.0, 1.0)
        with mock.patch.object(db, "_insert_hr_rows", side_effect=boom), \
                db_error_logs() as logs:
            self.writer.insert_hr_batch(sid, [(1_000, 70, 0.9)])
            result = call_bounded(self.writer.end_session, sid)
        return sid, result, logs

    def test_end_session_after_a_failed_flush_returns_writer_failed(self) -> None:
        _sid, (outcome, value, took), logs = self.fail_a_flush()
        assert_returns_error(self, outcome, value, took, "WriterFailed")
        # It carries the original error, which is also writer.failed.
        self.assertIsInstance(value.original, sqlite3.OperationalError)
        self.assertIs(self.writer.failed, value.original)
        # Logged once, with the traceback.
        self.assertEqual(len(logs), 1)
        self.assertIsNotNone(logs[0].exc_info)

    def test_after_a_failure_every_call_fails_fast_and_stop_returns(self) -> None:
        sid, (outcome, _value, took), _logs = self.fail_a_flush()
        self.assertNotEqual(outcome, "hung", f"end_session blocked {took:.1f} s")
        for name, call in (
            ("flush_and_wait", lambda: self.writer.flush_and_wait()),
            ("end_session", lambda: self.writer.end_session(sid)),
            ("start_session", lambda: self.writer.start_session("x", None, 1.0, 1.0)),
            ("delete_session", lambda: self.writer.delete_session(sid)),
            ("close_orphan_sessions", lambda: self.writer.close_orphan_sessions()),
            ("seed_demo_session", lambda: self.writer.seed_demo_session("Sitting", 5.0)),
        ):
            with self.subTest(call=name):
                assert_returns_error(self, *call_bounded(call), "WriterFailed", within=FAST_S)
        # Sample batches are dropped without raising or blocking.
        outcome, _value, _took = call_bounded(
            self.writer.insert_hr_batch, sid, [(2_000, 71, 0.9)], timeout=FAST_S
        )
        self.assertEqual(outcome, "ok")
        # The thread stayed alive to answer, and still exits on stop().
        self.assertTrue(self.writer._thread.is_alive())
        outcome, _value, took = call_bounded(self.writer.stop)
        self.assertEqual(outcome, "ok")
        self.assertLess(took, QUIT_S + 0.5)
        self.assertFalse(self.writer._thread.is_alive())

    def test_the_failed_batch_is_dropped_and_earlier_data_kept(self) -> None:
        sid, (outcome, _value, took), _logs = self.fail_a_flush()
        self.assertNotEqual(outcome, "hung", f"end_session blocked {took:.1f} s")
        # The session row (written before the failure) is still there and
        # still open; the HR row in the failed batch was not saved.
        sessions = db.list_sessions()
        self.assertEqual([s["id"] for s in sessions], [sid])
        self.assertIsNone(sessions[0]["ended_at"])
        self.assertEqual(db.get_db_info()["row_counts"]["hr_samples"], 0)


class StartupFailureTests(TempDbCase):
    def setUp(self) -> None:
        super().setUp()
        before = real_db_hashes()
        self.addCleanup(lambda: self.assertEqual(real_db_hashes(), before))
        db.init_db()
        self.writer = DatabaseWriter()
        self.addCleanup(lambda: call_bounded(self.writer.stop))

    def test_a_failed_connect_makes_the_first_sync_call_fail_fast(self) -> None:
        with mock.patch.object(
            db, "_connect", side_effect=sqlite3.OperationalError("unable to open database file")
        ), db_error_logs() as logs:
            self.writer.start()
            result = call_bounded(self.writer.close_orphan_sessions)
        assert_returns_error(self, *result, "WriterFailed")
        self.assertIsInstance(self.writer.failed, sqlite3.OperationalError)
        self.assertEqual(len(logs), 1)

    def test_startup_reports_the_error_and_exits_instead_of_hanging(self) -> None:
        """main.py: close_orphan_sessions failing shows a message box and
        tells main() to exit (recover_orphans returns False)."""
        with mock.patch.object(db, "_connect", side_effect=boom), \
                mock.patch.object(app_main, "QMessageBox", create=True) as box, \
                db_error_logs() as logs:
            self.writer.start()
            outcome, value, took = call_bounded(
                lambda: app_main.recover_orphans(self.writer)
            )
        self.assertEqual(outcome, "ok", f"{value!r} after {took:.1f} s")
        self.assertIs(value, False)
        box.critical.assert_called_once()
        # The writer's traceback once, plus main's one-line startup message.
        self.assertEqual(len([r for r in logs if r.exc_info]), 1)
        startup = [r.getMessage() for r in logs if not r.exc_info]
        self.assertEqual(len(startup), 1)
        self.assertTrue(startup[0].startswith("Startup: the database couldn't be opened"))

    def test_a_call_on_a_writer_that_was_never_started_fails_fast(self) -> None:
        assert_returns_error(
            self, *call_bounded(self.writer.flush_and_wait), "WriterUnavailable", within=FAST_S
        )


class InAppTriggerTests(WriterCase):
    def test_deleting_the_session_being_recorded_then_ending_it_does_not_hang(self) -> None:
        """No patches: samples keep arriving for a session deleted under
        them, so the next flush breaks the foreign key on session_id."""
        sid = self.writer.start_session("fixture", "Walking", 1.0, 1.0)
        self.writer.insert_hr_batch(sid, [(1_000, 70, 0.9)])
        self.writer.delete_session(sid)
        self.writer.insert_hr_batch(sid, [(2_000, 71, 0.9)])
        with db_error_logs() as logs:
            outcome, value, took = call_bounded(self.writer.end_session, sid)
        assert_returns_error(self, outcome, value, took, "WriterFailed")
        self.assertIsInstance(value.original, sqlite3.IntegrityError)
        self.assertEqual(len(logs), 1)


class StuckWriterTests(WriterCase):
    def test_quit_is_bounded_when_the_writer_is_stuck(self) -> None:
        """A flush that never finishes (say, a stalled disk): end_session
        gives up after 2 s with WriterUnavailable, then stop() after 2 s."""
        release = threading.Event()
        self.addCleanup(release.set)  # let the stuck flush finish at the end
        sid = self.writer.start_session("fixture", None, 1.0, 1.0)
        with mock.patch.object(db, "_insert_hr_rows", side_effect=lambda *a: release.wait(10)):
            self.writer.insert_hr_batch(sid, [(1_000, 70, 0.9)])
            outcome, value, took = call_bounded(
                lambda: self.writer.end_session(sid, timeout=QUIT_S), timeout=QUIT_S + 1.0
            )
            assert_returns_error(self, outcome, value, took, "WriterUnavailable", QUIT_S + 0.5)
            self.assertGreaterEqual(took, QUIT_S - 0.1)
            outcome, _value, took = call_bounded(self.writer.stop, timeout=QUIT_S + 1.0)
            self.assertEqual(outcome, "ok")
            self.assertLess(took, QUIT_S + 0.5)


class BoundedRecordingCase(RecordingCase):
    """RecordingCase, but the window's shutdown at cleanup runs in a side
    thread too: on a writer that hangs, a plain window.shutdown() cleanup
    would block the test run forever."""

    def setUp(self) -> None:
        super().setUp()
        before = real_db_hashes()
        self.addCleanup(lambda: self.assertEqual(real_db_hashes(), before))
        # The failures these tests cause are logged on purpose; keep them
        # off the test run's output (tests still see them via db_error_logs).
        for name in ("heartwatch.db", "heartwatch.recording"):
            log = logging.getLogger(name)
            self.addCleanup(setattr, log, "propagate", log.propagate)
            log.propagate = False
            quiet = logging.NullHandler()  # else logging's last-resort handler prints
            log.addHandler(quiet)
            self.addCleanup(log.removeHandler, quiet)

    def window(self, **kwargs) -> HeartWatchApp:
        window = HeartWatchApp(Theme(mode="dark"), self.writer, **kwargs)
        self.addCleanup(window.deleteLater)
        self.addCleanup(lambda: call_bounded(window.shutdown))
        return window


class ControllerFailureTests(BoundedRecordingCase):

    def recording_window(self):
        window = self.sim_window()
        self.states: list[str] = []
        self.controller.state_changed.connect(lambda state, *_: self.states.append(state))
        self.sid = self.controller.start("Walking")
        self.run_ms(500)
        return window

    def fail_the_next_flush(self) -> None:
        """Makes the next IMU flush raise and forces it (flush_and_wait, in a
        side thread), leaving the writer failed."""
        with mock.patch.object(db, "_insert_imu_rows", side_effect=boom), \
                db_error_logs() as logs:
            outcome, _value, took = call_bounded(self.writer.flush_and_wait)
        self.assertNotEqual(outcome, "hung", f"flush_and_wait blocked {took:.1f} s")
        self.assertEqual(len(logs), 1)

    def test_a_failed_flush_switches_to_error_on_the_next_batch(self) -> None:
        window = self.recording_window()
        self.fail_the_next_flush()
        self.assertEqual(self.states, ["recording"])  # no batch has arrived since
        self.run_ms(100)  # the next readings arrive
        self.assertEqual(self.states, ["recording", ERROR_STATE])
        self.assertEqual(self.controller.state, ERROR_STATE)
        self.assertFalse(self.controller.can_record)
        self.run_ms(1000)  # switched once only
        self.assertEqual(self.states, ["recording", ERROR_STATE])
        # Live Monitor: the message in the danger colour, Start disabled.
        live = window.live
        self.assertEqual(live.status_label.text(), live_monitor.ERROR_TEXT)
        self.assertEqual(
            live.status_label.text(), "Recording stopped: database error. Restart the app."
        )
        self.assertIn(window.theme.c("danger"), live.status_label.styleSheet())
        self.assertFalse(live.start_stop_button.isEnabled())
        self.assertFalse(live.activity_picker.isEnabled())
        # Stop and quit come back quickly and don't try to stamp ended_at.
        outcome, value, took = call_bounded(self.controller.stop, timeout=FAST_S)
        self.assertEqual((outcome, value), ("ok", None), f"after {took:.1f} s")
        outcome, value, took = call_bounded(self.controller.shutdown, timeout=FAST_S)
        self.assertEqual(outcome, "ok", f"{value!r} after {took:.1f} s")
        with contextlib.closing(db._connect()) as conn:
            row = conn.execute("SELECT ended_at FROM sessions WHERE id = ?", (self.sid,))
            self.assertIsNone(row.fetchone()[0])

    def test_shutdown_while_recording_on_a_failed_writer_returns_quickly(self) -> None:
        """Quit before any new batch arrives: the controller still thinks it
        is recording, end_session fails fast, shutdown logs and returns."""
        self.recording_window()
        self.fail_the_next_flush()
        with db_error_logs("heartwatch.recording") as logs:
            outcome, value, took = call_bounded(self.controller.shutdown, timeout=FAST_S + 0.5)
        self.assertEqual(outcome, "ok", f"{value!r} after {took:.1f} s")
        self.assertEqual(len(logs), 1)
        self.assertEqual(self.controller.state, ERROR_STATE)
        self.assertFalse(self.controller.can_record)

    def test_start_after_an_error_is_refused(self) -> None:
        window = self.recording_window()
        self.fail_the_next_flush()
        self.run_ms(100)
        self.assertEqual(self.states, ["recording", ERROR_STATE])
        outcome, value, _took = call_bounded(self.controller.start, "Sitting", timeout=FAST_S)
        self.assertEqual((outcome, value), ("ok", None))
        self.assertEqual(self.states, ["recording", ERROR_STATE])
        self.assertFalse(window.live.start_stop_button.isEnabled())


class SettingsDeleteRuleTests(unittest.TestCase):
    """The rule itself, without a window."""

    SESSIONS = [{"id": 3}, {"id": 2}, {"id": 1}]

    def test_the_recording_session_is_not_offered(self) -> None:
        self.assertEqual(
            [s["id"] for s in settings_mod.deletable_sessions(self.SESSIONS, 2)], [3, 1]
        )
        self.assertEqual(
            [s["id"] for s in settings_mod.deletable_sessions(self.SESSIONS, None)], [3, 2, 1]
        )

    def test_deleting_the_recording_session_is_refused(self) -> None:
        self.assertIsNotNone(settings_mod.delete_refusal(2, 2))
        self.assertIsNone(settings_mod.delete_refusal(1, 2))
        self.assertIsNone(settings_mod.delete_refusal(1, None))


class SettingsPanelTests(BoundedRecordingCase):

    def picker_ids(self, panel) -> list[int]:
        picker = panel._session_picker
        return [picker.itemData(i) for i in range(picker.count())]

    def test_the_recording_session_is_hidden_from_the_delete_picker(self) -> None:
        window = self.sim_window()
        past = self.writer.seed_demo_session("Sitting", 5.0)
        window.settings._refresh_session_picker()
        self.assertEqual(self.picker_ids(window.settings), [past])
        sid = self.controller.start("Walking")
        self.assertNotIn(sid, self.picker_ids(window.settings))
        window.settings._refresh_session_picker()  # any later refresh, too
        self.assertNotIn(sid, self.picker_ids(window.settings))
        self.assertIn(past, self.picker_ids(window.settings))
        # Offered again once it has stopped.
        self.controller.stop()
        self.assertIn(sid, self.picker_ids(window.settings))

    def test_a_delete_of_the_recording_session_is_refused(self) -> None:
        window = self.sim_window()
        sid = self.controller.start("Walking")
        panel = window.settings
        # Requested anyway (e.g. an item put there before Start).
        panel._session_picker.addItem(f"#{sid}", sid)
        panel._session_picker.setCurrentIndex(panel._session_picker.count() - 1)
        with mock.patch.object(settings_mod, "QMessageBox") as box, \
                mock.patch.object(self.writer, "delete_session") as delete:
            box.question.return_value = box.Yes
            panel._on_delete_clicked()
        delete.assert_not_called()
        box.warning.assert_called_once()
        self.assertEqual(self.controller.state, "recording")

    def test_panel_calls_on_a_failed_writer_show_a_message(self) -> None:
        window = self.sim_window()
        panel = window.settings
        sid = self.controller.start("Walking")
        self.run_ms(300)
        with mock.patch.object(db, "_insert_imu_rows", side_effect=boom), \
                db_error_logs() as logs:
            outcome, _value, took = call_bounded(self.writer.flush_and_wait)
        self.assertNotEqual(outcome, "hung", f"flush_and_wait blocked {took:.1f} s")
        for name, handler in (
            ("seed", panel._on_seed_clicked),
            ("run query", panel._on_run_query_clicked),
            ("refresh", panel._refresh_info),
            ("delete", panel._on_delete_clicked),
        ):
            with self.subTest(handler=name), \
                    mock.patch.object(settings_mod, "QMessageBox") as box:
                box.question.return_value = box.Yes
                if name == "delete":
                    # a past session id: not the recording one, so the
                    # writer is actually asked (and fails fast)
                    panel._session_picker.addItem("#999", sid + 1)
                    panel._session_picker.setCurrentIndex(panel._session_picker.count() - 1)
                t0 = time.monotonic()
                handler()
                self.assertLess(time.monotonic() - t0, FAST_S)
                box.warning.assert_called_once()


if __name__ == "__main__":
    unittest.main()
