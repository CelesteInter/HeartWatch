"""Tests for the "Ask about your data" bar (ui/ai_bar.py), its timeout
helper (ui/pending_request.py) and the topbar's AI pill, built offscreen.

Plain-language overview: a question the keyword parser understands is
answered straight away with the reply and the "what I understood" line; a
question that needs the AI shows "Thinking…" until the (fake) worker
replies; a newer question always replaces an older one, a stale or late
reply is ignored, and no reply within the timeout gives the "didn't
understand" reply. Qt runs with the offscreen platform (like
test_dashboard_view.py), no Ollama is needed, and the bar is pointed at a
fixture database, never the real one.

    python -m unittest heartwatchapp.tests.test_ai_bar
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json  # noqa: E402
import time  # noqa: E402
import unittest  # noqa: E402
from unittest import mock  # noqa: E402

from PySide6.QtCore import QEvent, QObject, Signal  # noqa: E402
from PySide6.QtGui import QFocusEvent  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from ..ask import answers  # noqa: E402
from ..llm import client as llm_client  # noqa: E402
from ..ui import ai_bar  # noqa: E402
from ..ui.pending_request import PendingRequest  # noqa: E402
from ..ui.theme import Theme  # noqa: E402
from ..ui.topbar import Topbar  # noqa: E402
from .test_queries import FixtureDb  # noqa: E402

# Needs the AI: "treadmill" is a word the keyword parser doesn't know.
AI_QUESTION = "average heart rate on the treadmill ever"


def form(metric, activity=None, range_name="all") -> str:
    return json.dumps({"intent": "query", "metric": metric, "activity": activity,
                       "range": range_name})


class FakeWorker(QObject):
    """Stands in for LlmWorker: same signals, records requests, never
    replies on its own."""

    parse_finished = Signal(int, str)
    parse_failed = Signal(int, str)
    health_checked = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.requests: list = []
        self.preloads = 0

    def request_parse(self, messages, format) -> int:
        self.requests.append((messages, format))
        return len(self.requests)

    def preload(self) -> None:
        self.preloads += 1


class AiBarCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])
        cls.theme = Theme(mode="dark")

    def setUp(self) -> None:
        # One finished session, an hour ago, heart rate 90 then 100. The
        # questions use "ever" so the answer doesn't depend on today's date.
        self.fixture = FixtureDb(self)
        start = int(time.time() * 1000) - 3_600_000
        self.fixture.add(start, start + 10_000, hr=[(0, 90), (5000, 100)])
        env = mock.patch.dict(os.environ, {"HEARTWATCH_ASK_MODE": "hybrid"})
        env.start()
        self.addCleanup(env.stop)
        self.worker = FakeWorker()

    def build(self, worker=None) -> ai_bar.AiBar:
        bar = ai_bar.AiBar(self.theme, worker, db_path=self.fixture.path)
        self.addCleanup(bar.deleteLater)
        return bar

    def ask(self, bar: ai_bar.AiBar, question: str) -> None:
        bar.input.setText(question)
        bar.input.returnPressed.emit()

    def shown(self, bar: ai_bar.AiBar) -> tuple:
        understood = None if bar.understood_text.isHidden() else bar.understood_text.text()
        caption = None if bar.caption_text.isHidden() else bar.caption_text.text()
        return bar.result_text.text(), understood, caption


class RulesPathTests(AiBarCase):
    def test_a_rules_question_is_answered_at_once(self) -> None:
        bar = self.build(self.worker)
        self.ask(bar, "peak heart rate ever")
        self.assertFalse(bar.result.isHidden())
        self.assertEqual(self.shown(bar), (
            "Your heart rate peaked at 100 bpm in your one recorded session.",
            "Peak heart rate · All activities · all sessions",
            None,
        ))
        self.assertEqual(self.worker.requests, [])
        self.assertEqual(bar.input.text(), "")

    def test_empty_input_does_nothing(self) -> None:
        bar = self.build(self.worker)
        self.ask(bar, "   ")
        self.assertTrue(bar.result.isHidden())

    def test_works_with_no_worker_at_all(self) -> None:
        bar = self.build(None)
        self.ask(bar, "average heart rate ever")
        self.assertEqual(bar.result_text.text(),
                         "Your heart rate averaged 95 bpm in your one recorded session.")
        self.ask(bar, AI_QUESTION)
        self.assertEqual(self.shown(bar), (answers.AI_UNAVAILABLE_REPLY, None, None))


class AiPathTests(AiBarCase):
    def test_thinking_then_the_answer(self) -> None:
        bar = self.build(self.worker)
        self.ask(bar, AI_QUESTION)
        self.assertEqual(self.shown(bar), (answers.THINKING_TEXT, None, None))
        ((messages, _schema),) = self.worker.requests
        self.assertEqual(messages[-1]["content"], AI_QUESTION)
        self.worker.parse_finished.emit(1, form("avg_hr"))
        self.assertEqual(self.shown(bar), (
            "Your heart rate averaged 95 bpm in your one recorded session.",
            "Average heart rate · All activities · all sessions",
            None,
        ))

    def test_a_stale_reply_is_ignored(self) -> None:
        bar = self.build(self.worker)
        self.ask(bar, AI_QUESTION)
        self.worker.parse_finished.emit(99, form("peak_hr"))
        self.assertEqual(bar.result_text.text(), answers.THINKING_TEXT)

    def test_a_newer_question_replaces_an_older_one(self) -> None:
        bar = self.build(self.worker)
        self.ask(bar, AI_QUESTION)                                # request 1
        self.ask(bar, "peak heart rate on the treadmill ever")    # request 2
        self.worker.parse_finished.emit(1, form("avg_hr"))        # late reply to 1
        self.assertEqual(bar.result_text.text(), answers.THINKING_TEXT)
        self.worker.parse_finished.emit(2, form("peak_hr"))
        self.assertEqual(bar.result_text.text(),
                         "Your heart rate peaked at 100 bpm in your one recorded session.")
        # A rules question asked while the AI is busy wins too.
        self.ask(bar, AI_QUESTION)                                # request 3
        self.ask(bar, "average heart rate ever")
        self.worker.parse_finished.emit(3, form("peak_hr"))
        self.assertEqual(bar.result_text.text(),
                         "Your heart rate averaged 95 bpm in your one recorded session.")

    def test_rapid_submits_dont_crash_or_mix_answers(self) -> None:
        bar = self.build(self.worker)
        for _ in range(20):
            self.ask(bar, AI_QUESTION)
            self.ask(bar, "peak heart rate ever")
        for request_id in range(1, 21):
            self.worker.parse_finished.emit(request_id, form("avg_hr"))
        self.assertEqual(bar.result_text.text(),
                         "Your heart rate peaked at 100 bpm in your one recorded session.")

    def test_failures(self) -> None:
        bar = self.build(self.worker)
        self.ask(bar, AI_QUESTION)
        self.worker.parse_failed.emit(1, "unavailable")
        self.assertEqual(bar.result_text.text(), answers.AI_UNAVAILABLE_REPLY)
        self.ask(bar, AI_QUESTION)
        with self.assertLogs("heartwatch.llm"):
            self.worker.parse_failed.emit(2, "error")
        self.assertEqual(bar.result_text.text(), answers.NOT_UNDERSTOOD_REPLY)

    def test_known_unavailable_skips_the_request(self) -> None:
        bar = self.build(self.worker)
        self.worker.health_checked.emit(llm_client.unreachable_status(ConnectionError("x")))
        self.ask(bar, AI_QUESTION)
        self.assertEqual(bar.result_text.text(), answers.AI_UNAVAILABLE_REPLY)
        self.assertEqual(self.worker.requests, [])

    def test_timeout(self) -> None:
        with mock.patch.object(ai_bar, "ASK_TIMEOUT_MS", 20):
            bar = self.build(self.worker)
        self.ask(bar, AI_QUESTION)
        with self.assertLogs("heartwatch.llm"):
            QTest.qWait(100)
        self.assertEqual(bar.result_text.text(), answers.NOT_UNDERSTOOD_REPLY)
        # The reply that arrives afterwards is ignored.
        self.worker.parse_finished.emit(1, form("avg_hr"))
        self.assertEqual(bar.result_text.text(), answers.NOT_UNDERSTOOD_REPLY)

    def test_rules_mode_never_asks_the_worker(self) -> None:
        bar = self.build(self.worker)
        with mock.patch.dict(os.environ, {"HEARTWATCH_ASK_MODE": "rules"}):
            self.ask(bar, AI_QUESTION)
        self.assertEqual(bar.result_text.text(), answers.NOT_UNDERSTOOD_REPLY)
        self.assertEqual(self.worker.requests, [])


class PreloadTests(AiBarCase):
    def focus(self, bar: ai_bar.AiBar) -> None:
        QApplication.sendEvent(bar.input, QFocusEvent(QEvent.FocusIn))

    def test_focus_preloads_once_per_keep_alive_period(self) -> None:
        bar = self.build(self.worker)
        self.focus(bar)
        self.focus(bar)
        self.assertEqual(self.worker.preloads, 1)
        later = time.monotonic() + llm_client.DEFAULT_KEEP_ALIVE_S + 1
        with mock.patch.object(ai_bar.time, "monotonic", return_value=later):
            self.focus(bar)
        self.assertEqual(self.worker.preloads, 2)

    def test_no_preload_in_rules_mode_or_when_unavailable(self) -> None:
        bar = self.build(self.worker)
        with mock.patch.dict(os.environ, {"HEARTWATCH_ASK_MODE": "rules"}):
            self.focus(bar)
        self.worker.health_checked.emit(llm_client.model_missing_status())
        self.focus(bar)
        self.assertEqual(self.worker.preloads, 0)


class PendingRequestTests(AiBarCase):
    def test_claim_cancel_and_timeout(self) -> None:
        pending = PendingRequest(20)
        self.addCleanup(pending.deleteLater)
        timed_out: list[int] = []
        pending.timed_out.connect(timed_out.append)

        pending.start(1)
        self.assertFalse(pending.claim(2))      # stale
        self.assertTrue(pending.claim(1))
        self.assertFalse(pending.claim(1))      # only once
        pending.start(3)
        pending.start(4)                        # replaces 3
        self.assertFalse(pending.claim(3))
        pending.cancel()
        self.assertIsNone(pending.pending_id)
        QTest.qWait(60)
        self.assertEqual(timed_out, [])         # cancelled: no timeout
        pending.start(5)
        QTest.qWait(60)
        self.assertEqual(timed_out, [5])        # once
        self.assertFalse(pending.claim(5))      # and too late now


class AiPillTests(AiBarCase):
    def test_three_states(self) -> None:
        topbar = Topbar(self.theme)
        self.addCleanup(topbar.deleteLater)
        for status, text in (
            (llm_client.ready_status(), "AI: ready"),
            (llm_client.model_missing_status(), "AI: model missing"),
            (llm_client.unreachable_status(ConnectionError("x")), "AI: unavailable"),
        ):
            with self.subTest(text=text):
                topbar.set_ai_status(status.ready, status.message, status.daemon_reachable)
                self.assertEqual(topbar.ai_pill.text(), text)
                self.assertEqual(topbar.ai_pill.toolTip(), status.message)


class SuggestionButtonTests(AiBarCase):
    """Handoff 9a, 3.7: buttons under a reply that can't answer; a click
    asks the button's question exactly as if typed."""

    def labels(self, bar: ai_bar.AiBar) -> list[str]:
        return [b.text() for b in bar.suggestion_buttons]

    def test_a_clarification_and_a_click(self) -> None:
        bar = self.build(self.worker)
        self.ask(bar, "how long was I on my feet ever")
        self.assertEqual(bar.result_text.text(), "Did you mean walking or standing?")
        self.assertEqual(self.labels(bar), ["Walking", "Standing"])
        bar.suggestion_buttons[0].click()
        # "how long was i walking ever": the fixture has no activity data.
        self.assertEqual(self.shown(bar), ("You haven't recorded any activity yet.",
                                           "Time in activity · Walking · all sessions", None))
        self.assertEqual(self.worker.requests, [])

    def test_which_activity_offers_all_four(self) -> None:
        bar = self.build(self.worker)
        self.ask(bar, "how much time did I spend ever")
        self.assertEqual(self.labels(bar), ["Sitting", "Walking", "Standing", "Running"])

    def test_buttons_clear_for_an_answer_and_while_thinking(self) -> None:
        bar = self.build(self.worker)
        self.ask(bar, "what's the weather")
        self.assertEqual(len(bar.suggestion_buttons), 3)
        self.ask(bar, AI_QUESTION)
        self.assertEqual((bar.result_text.text(), bar.suggestion_buttons),
                         (answers.THINKING_TEXT, []))
        self.ask(bar, "peak heart rate ever")
        self.assertEqual(bar.suggestion_buttons, [])

    def test_a_click_while_thinking_replaces_the_pending_question(self) -> None:
        # Same as typing a new question: the late model reply is ignored.
        bar = self.build(self.worker)
        self.ask(bar, AI_QUESTION)                                # request 1
        bar.submit_text("peak heart rate ever")
        self.assertEqual(bar.result_text.text(),
                         "Your heart rate peaked at 100 bpm in your one recorded session.")
        self.worker.parse_finished.emit(1, form("avg_hr"))
        self.assertEqual(bar.result_text.text(),
                         "Your heart rate peaked at 100 bpm in your one recorded session.")


class BarMatchesDashboardTests(AiBarCase):
    """A1: the bar's breakdown and the Dashboard's bars give the same
    percentage for every activity on the same data."""

    def test_same_percentages(self) -> None:
        import re
        import tempfile
        from pathlib import Path

        from PySide6.QtWidgets import QLabel

        from ..scripts import demo_db
        from ..ui import dashboard

        with tempfile.TemporaryDirectory() as tmp:
            path, _ = demo_db.build_demo_db(Path(tmp) / "demo.db", data_seed=0)
            view = dashboard.DashboardView(self.theme, db_path=path)
            self.addCleanup(view.deleteLater)
            on_dashboard = {}
            for row in view.findChildren(dashboard.ActivityBar):
                name, pct = (label.text() for label in row.findChildren(QLabel))
                on_dashboard[name.lower()] = pct
            bar = ai_bar.AiBar(self.theme, self.worker, db_path=path)
            self.addCleanup(bar.deleteLater)
            self.ask(bar, "give me an activity breakdown")
        parts = bar.result_text.text().split(": ", 1)[1].rstrip(".").split(", ")
        in_bar = dict(re.fullmatch(r"(\w+) .*\((\d+%)\)", part).groups() for part in parts)
        self.assertEqual(in_bar, on_dashboard)
        self.assertEqual(on_dashboard,
                         {"walking": "41%", "standing": "25%", "sitting": "21%", "running": "13%"})


class DevToolBarTests(AiBarCase):
    """J. HEARTWATCH_ASK_DEV: off writes nothing and adds no marker; on adds
    the marker and logs each question without any number from the reply."""

    def setUp(self) -> None:
        super().setUp()
        import tempfile
        from pathlib import Path

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.log = Path(tmp.name) / "dev_logs" / "ask_questions.jsonl"

    def build_logged(self) -> ai_bar.AiBar:
        bar = self.build(self.worker)
        bar._dev_log_path = self.log
        return bar

    def test_off_by_default(self) -> None:
        with mock.patch.dict(os.environ, {"HEARTWATCH_ASK_DEV": ""}):
            bar = self.build_logged()
            self.ask(bar, "avrage heart rate ever")
        self.assertEqual(self.shown(bar)[1],
                         "Average heart rate · All activities · all sessions · corrected: "
                         "avrage → average")
        self.assertFalse(self.log.exists())
        self.assertFalse(self.log.parent.exists())

    def test_on_marks_and_logs(self) -> None:
        with mock.patch.dict(os.environ, {"HEARTWATCH_ASK_DEV": "1"}):
            bar = self.build_logged()
            self.ask(bar, "avrage heart rate ever")
            self.assertEqual(bar.understood_text.text(),
                             "Average heart rate · All activities · all sessions · corrected: "
                             "avrage → average · rules (corrected)")
            self.ask(bar, "is my heart rate healthy")
            self.assertEqual(bar.understood_text.text(), "guard")
            self.ask(bar, AI_QUESTION)
            self.worker.parse_finished.emit(1, form("avg_hr"))
            self.assertTrue(bar.understood_text.text().endswith(" · AI"))
        lines = [json.loads(line) for line in self.log.read_text().splitlines()]
        self.assertEqual([e["decided_by"] for e in lines], ["rules", "guard", "llm"])
        self.assertEqual(lines[0]["question"], "avrage heart rate ever")
        self.assertEqual(lines[0]["corrections"], [["avrage", "average"]])
        # The replies said 95 bpm; the log never does (timing and clock
        # fields left out, as their digits are arbitrary).
        logged = json.dumps([{k: v for k, v in e.items() if k not in ("latency_ms", "timestamp")}
                             for e in lines])
        self.assertNotIn("95", logged)
        self.assertNotIn("bpm", logged)
