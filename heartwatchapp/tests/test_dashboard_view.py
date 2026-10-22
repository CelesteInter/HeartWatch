"""Tests for the Dashboard widget (ui/dashboard.py), built offscreen.

Plain-language overview: builds the real DashboardView on a temporary
database and reads back the text it shows: the four cards must match what
data/queries.py's run() answers (J), the "simulated predictions" caption
appears only when dummy-classifier time feeds the breakdown, empty data
says so plainly, and an unlabeled session doesn't borrow an activity's
icon. Qt runs with the offscreen platform (like test_summary_mode.py), so
no window opens. The real database is never opened: every view is pointed
at a fixture or a fresh demo database.

    python -m unittest heartwatchapp.tests.test_dashboard_view
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import contextlib  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import unittest  # noqa: E402
from pathlib import Path  # noqa: E402

from PySide6.QtWidgets import QApplication, QLabel  # noqa: E402

from ..data import db, queries  # noqa: E402
from ..scripts import demo_db  # noqa: E402
from ..ui import dashboard  # noqa: E402
from ..ui.theme import Theme  # noqa: E402
from ..ui.widgets import MetricCard  # noqa: E402


class DashboardViewCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])
        cls.theme = Theme(mode="dark")

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def build(self, path: Path) -> dashboard.DashboardView:
        view = dashboard.DashboardView(self.theme, db_path=path)
        self.addCleanup(view.deleteLater)
        return view

    def cards(self, view: dashboard.DashboardView) -> dict[str, str]:
        """{card title: value text} for the four metric cards."""
        out = {}
        for card in view.findChildren(MetricCard):
            title = next(lbl.text() for lbl in card.findChildren(QLabel)
                         if lbl.objectName() == "CardLabel")
            out[title] = card.value_label.text()
        return out

    def texts(self, view: dashboard.DashboardView) -> list[str]:
        return [lbl.text() for lbl in view.findChildren(QLabel)]

    def fixture(self, sessions) -> Path:
        """A fixture DB with finished sessions that ended a minute ago, so
        they're in the Dashboard's last 7 days whatever the real clock says.
        sessions: (label, hr bpms, prediction labels, model_version)."""
        path = self.dir / "fixture.db"
        db.init_db(path)
        now = int(time.time() * 1000)
        with contextlib.closing(db._connect(path)) as conn, conn:
            for i, (label, bpms, predicted, model_version) in enumerate(sessions):
                start = now - 120_000 - i * 1_000
                sid = db.start_session(conn, "fixture", label, 1.0, 1.0, started_at=start)
                db.insert_hr_batch(
                    conn, sid, [(start + k * 1000, bpm, None) for k, bpm in enumerate(bpms)]
                )
                for k, activity in enumerate(predicted, start=1):
                    db.insert_prediction(conn, sid, start + k * 5000, activity, 0.9, model_version)
                db.end_session(conn, sid, start + 60_000)
        return path


class DashboardMatchesQueriesTests(DashboardViewCase):
    def test_j_cards_show_what_run_answers_for_the_demo_session(self) -> None:
        path, _ = demo_db.build_demo_db(self.dir / "demo.db", data_seed=0)
        view = self.build(path)

        def run(metric: str) -> queries.MetricResult:
            return queries.run(queries.MetricRequest(metric, None, "last_7_days"), path=path)

        self.assertEqual(
            self.cards(view),
            {
                "Heart rate": f"{run('avg_hr').value:.0f}",
                "Active time": f"{run('total_time').value / 60:.0f}",
                "Peak HR": f"{run('peak_hr').value:.0f}",
                "Sessions": f"{run('session_count').value:.0f}",
            },
        )
        # By hand for seed 0: avg 32349 / 340 = 95.1 -> "95"; 340 s / 60 =
        # 5.67 -> "6" min; peak 134; 1 session.
        self.assertEqual(
            self.cards(view),
            {"Heart rate": "95", "Active time": "6", "Peak HR": "134", "Sessions": "1"},
        )

    def test_j_bars_follow_the_breakdown_order(self) -> None:
        path, _ = demo_db.build_demo_db(self.dir / "demo.db", data_seed=0)
        view = self.build(path)
        bars = [bar.findChildren(QLabel)[0].text() for bar in view.findChildren(dashboard.ActivityBar)]
        self.assertEqual(bars, ["Walking", "Standing", "Sitting", "Running"])


class SimulatedCaptionTests(DashboardViewCase):
    def test_caption_appears_for_dummy_predictions(self) -> None:
        view = self.build(self.fixture([("Walking", [80], ["Walking"], "demo-dummy")]))
        self.assertIn(dashboard.SIMULATED_CAPTION, self.texts(view))

    def test_caption_absent_for_a_real_model(self) -> None:
        view = self.build(self.fixture([("Walking", [80], ["Walking"], "cnn-v1")]))
        self.assertNotIn(dashboard.SIMULATED_CAPTION, self.texts(view))

    def test_caption_absent_without_predictions(self) -> None:
        view = self.build(self.fixture([("Walking", [80], [], "demo-dummy")]))
        self.assertNotIn(dashboard.SIMULATED_CAPTION, self.texts(view))


class EmptyStateTests(DashboardViewCase):
    def assert_onboarding(self, view: dashboard.DashboardView) -> None:
        """The whole Dashboard is the onboarding card: no metric cards."""
        texts = self.texts(view)
        self.assertIn(dashboard.ONBOARDING_TITLE, texts)
        self.assertIn(dashboard.ONBOARDING_DETAIL, texts)
        self.assertEqual(self.cards(view), {})
        self.assertNotIn(dashboard.NO_SESSIONS_TEXT, texts)

    def test_empty_database_shows_the_onboarding_card(self) -> None:
        path = self.dir / "empty.db"
        db.init_db(path)
        self.assert_onboarding(self.build(path))

    def test_a_live_session_alone_counts_as_no_finished_sessions(self) -> None:
        # Like a fresh install with the app open: the Live Monitor's session
        # is still recording (ended_at NULL), so there's nothing finished.
        path = self.dir / "live.db"
        db.init_db(path)
        start = int(time.time() * 1000) - 60_000
        with contextlib.closing(db._connect(path)) as conn, conn:
            sid = db.start_session(conn, "demo", None, 1.0, 1.0, started_at=start)
            db.insert_hr_batch(conn, sid, [(start, 80, None)])
        self.assert_onboarding(self.build(path))

    def test_finished_sessions_outside_the_range_show_dashes_and_say_why(self) -> None:
        # One finished session 30 days ago: the database isn't empty, but
        # nothing started in the last 7 calendar days (at most 7 x 25 h ago).
        path = self.dir / "old.db"
        db.init_db(path)
        start = int(time.time() * 1000) - 30 * 86_400_000
        with contextlib.closing(db._connect(path)) as conn, conn:
            sid = db.start_session(conn, "fixture", "Walking", 1.0, 1.0, started_at=start)
            db.insert_hr_batch(conn, sid, [(start, 80, None)])
            db.end_session(conn, sid, start + 60_000)
        view = self.build(path)
        texts = self.texts(view)
        self.assertEqual(set(self.cards(view).values()), {dashboard.EMPTY_VALUE})
        self.assertIn(dashboard.NO_SESSIONS_TEXT, texts)
        self.assertNotIn(dashboard.ONBOARDING_TITLE, texts)
        # Recent sessions isn't limited to the range, so the old one is listed.
        self.assertEqual(len(view.findChildren(dashboard.SessionRow)), 1)

    def test_sessions_without_predictions_show_heart_rate_and_no_activity_data(self) -> None:
        # HR 70, 80: average 75, peak 80.
        view = self.build(self.fixture([(None, [70, 80], [], "cnn-v1")]))
        cards = self.cards(view)
        self.assertEqual((cards["Heart rate"], cards["Peak HR"]), ("75", "80"))
        self.assertIn(dashboard.NO_ACTIVITY_TEXT, self.texts(view))
        self.assertEqual(view.findChildren(dashboard.ActivityBar), [])


class RecentSessionRowTests(DashboardViewCase):
    def test_unlabeled_session_gets_a_neutral_icon(self) -> None:
        view = self.build(self.fixture([(None, [], [], "cnn-v1"), ("Sitting", [140], [], "cnn-v1")]))
        rows = view.findChildren(dashboard.SessionRow)
        icons = sorted(row.icon.text() for row in rows)
        self.assertEqual(icons, sorted([dashboard.UNLABELED_ICON, "S"]))
        # The unlabeled one has no heart rate: a dash, not "0 bpm".
        peaks = sorted(row.peak.text() for row in rows)
        self.assertEqual(peaks, sorted(["140 bpm", dashboard.EMPTY_VALUE]))


if __name__ == "__main__":
    unittest.main()
