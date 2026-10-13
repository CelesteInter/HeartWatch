"""Tests for HEARTWATCH_SUMMARY_MODE: config.summary_mode(), and which
path the session detail dialog takes in each mode.

Stdlib unittest, same as the other test files. Run from the repo root:

    python -m unittest heartwatchapp.tests.test_summary_mode

No live Ollama daemon and no database: the dialog gets a fake LLM worker
(records requests, never answers) and patched-in session data, and Qt runs
with the offscreen platform so no window opens.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject, Signal  # noqa: E402
from PySide6.QtWidgets import QApplication, QLabel  # noqa: E402

from .. import config  # noqa: E402
from ..data import stats  # noqa: E402
from ..ui import sessions  # noqa: E402
from ..ui.theme import Theme  # noqa: E402


class SummaryModeTests(unittest.TestCase):
    def test_unset_is_plain(self) -> None:
        self.assertEqual(config.summary_mode({}), "plain")

    def test_empty_is_plain(self) -> None:
        self.assertEqual(config.summary_mode({config.SUMMARY_MODE_ENV_VAR: ""}), "plain")

    def test_plain(self) -> None:
        self.assertEqual(config.summary_mode({config.SUMMARY_MODE_ENV_VAR: "plain"}), "plain")

    def test_llm(self) -> None:
        self.assertEqual(config.summary_mode({config.SUMMARY_MODE_ENV_VAR: "llm"}), "llm")

    def test_case_insensitive(self) -> None:
        self.assertEqual(config.summary_mode({config.SUMMARY_MODE_ENV_VAR: "LLM"}), "llm")

    def test_invalid_is_plain_with_one_warning(self) -> None:
        environ = {config.SUMMARY_MODE_ENV_VAR: "gpt-please"}
        config._warned_summary_modes.discard("gpt-please")
        with self.assertLogs("heartwatch.llm", level="WARNING") as logs:
            self.assertEqual(config.summary_mode(environ), "plain")
            self.assertEqual(config.summary_mode(environ), "plain")
        self.assertEqual(len(logs.records), 1)
        self.assertIn("gpt-please", logs.output[0])


class _FakeWorker(QObject):
    """Stands in for llm.worker.LlmWorker: same signals, records every
    request, never replies."""

    summary_finished = Signal(int, str)
    summary_failed = Signal(int, str)

    def __init__(self) -> None:
        super().__init__()
        self.requests: list[dict] = []

    def request_session_summary(self, session_stats: dict) -> int:
        self.requests.append(session_stats)
        return len(self.requests)


# A session rich enough for the LLM path, and one that isn't (Session #7's
# shape: no predictions, a minute of heart rate).
SUMMARIZABLE = {
    "session_id": 1,
    "label": "Walking",
    "duration_s": 330.0,
    "avg_hr": 101.4,
    "max_hr": 134,
    "hr_sample_count": 330,
    "activity_seconds": {"Walking": 190.0, "Standing": 125.0},
    "peak_hr_activity": "Walking",
    "prediction_count": 63,
    "low_confidence_count": 0,
}
UNSUMMARIZABLE = {
    **SUMMARIZABLE,
    "duration_s": 62.0,
    "hr_sample_count": 62,
    "activity_seconds": {},
    "peak_hr_activity": None,
    "prediction_count": 0,
}
EMPTY_TIMELINE = {"session": {"label": "Walking"}, "imu": [], "hr": [], "predictions": []}


class SummaryCardModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])
        cls.theme = Theme(mode="dark")

    def _open(self, mode: str | None, session_stats: dict):
        """Builds a SessionDetailDialog for `session_stats` with the given
        HEARTWATCH_SUMMARY_MODE (None = unset). Returns (dialog, worker,
        every label's text)."""
        environ = {} if mode is None else {config.SUMMARY_MODE_ENV_VAR: mode}
        worker = _FakeWorker()
        with mock.patch.dict(os.environ, environ, clear=False), mock.patch.object(
            sessions.db, "get_session_timeline", return_value=EMPTY_TIMELINE
        ), mock.patch.object(
            sessions.stats, "compute_session_stats", return_value=session_stats
        ):
            if mode is None:
                os.environ.pop(config.SUMMARY_MODE_ENV_VAR, None)
            dialog = sessions.SessionDetailDialog(1, self.theme, worker)
        self.addCleanup(dialog.deleteLater)
        texts = [label.text() for label in dialog.findChildren(QLabel)]
        return dialog, worker, texts

    def test_plain_is_the_default_and_never_asks_the_worker(self) -> None:
        dialog, worker, texts = self._open(None, SUMMARIZABLE)
        self.assertEqual(worker.requests, [])
        self.assertIsNone(dialog._summary_timer)
        self.assertEqual(dialog._summary_label.text(), stats.format_stats_plain(SUMMARIZABLE))
        self.assertNotIn(sessions.GENERATING_TEXT, "\n".join(texts))

    def test_plain_unsummarizable_has_no_not_enough_data_note(self) -> None:
        dialog, worker, texts = self._open("plain", UNSUMMARIZABLE)
        self.assertEqual(worker.requests, [])
        self.assertEqual(dialog._summary_label.text(), stats.format_stats_plain(UNSUMMARIZABLE))
        self.assertNotIn(sessions.NOT_ENOUGH_DATA_TEXT, "\n".join(texts))

    def test_plain_still_shows_the_confidence_caveat(self) -> None:
        _dialog, _worker, texts = self._open("plain", {**SUMMARIZABLE, "low_confidence_count": 3})
        self.assertIn(stats.confidence_caveat({"low_confidence_count": 3}), texts)

    def test_llm_requests_a_summary_for_a_summarizable_session(self) -> None:
        dialog, worker, _texts = self._open("llm", SUMMARIZABLE)
        self.assertEqual(worker.requests, [SUMMARIZABLE])
        self.assertEqual(dialog._summary_label.text(), sessions.GENERATING_TEXT)
        dialog.done(0)  # stops the timeout timer and disconnects

    def test_llm_unsummarizable_keeps_the_not_enough_data_note(self) -> None:
        dialog, worker, _texts = self._open("llm", UNSUMMARIZABLE)
        self.assertEqual(worker.requests, [])
        self.assertIn(sessions.NOT_ENOUGH_DATA_TEXT, dialog._summary_label.text())


if __name__ == "__main__":
    unittest.main()
