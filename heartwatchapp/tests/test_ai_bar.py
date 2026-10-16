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
            "Your peak heart rate across all sessions was 100 bpm (1 session).",
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
                         "Your average heart rate across all sessions was 95 bpm (1 session).")
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
            "Your average heart rate across all sessions was 95 bpm (1 session).",
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
                         "Your peak heart rate across all sessions was 100 bpm (1 session).")
        # A rules question asked while the AI is busy wins too.
        self.ask(bar, AI_QUESTION)                                # request 3
        self.ask(bar, "average heart rate ever")
        self.worker.parse_finished.emit(3, form("peak_hr"))
        self.assertEqual(bar.result_text.text(),
                         "Your average heart rate across all sessions was 95 bpm (1 session).")

    def test_rapid_submits_dont_crash_or_mix_answers(self) -> None:
        bar = self.build(self.worker)
        for _ in range(20):
            self.ask(bar, AI_QUESTION)
            self.ask(bar, "peak heart rate ever")
        for request_id in range(1, 21):
            self.worker.parse_finished.emit(request_id, form("avg_hr"))
        self.assertEqual(bar.result_text.text(),
                         "Your peak heart rate across all sessions was 100 bpm (1 session).")

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
