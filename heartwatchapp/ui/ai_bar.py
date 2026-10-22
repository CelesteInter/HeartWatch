"""Always-visible "Ask about your data" bar.

Plain-language overview: a question typed here goes through ask/pipeline.py.
Questions the keyword parser understands are answered straight away, on
the GUI thread (no network, no model). Others go to the local model through
LlmWorker.request_parse() while the bar shows "Thinking…"; a
PendingRequest (ui/pending_request.py) gives that 10 s and makes sure a
newer question always replaces an older one's answer. Every word shown --
the reply, the "what I understood" line, the simulated-data caption and
the suggestion buttons -- is written by ask/answers.py and ask/suggest.py,
never by the model. Clicking a suggestion asks it exactly as if typed, so
a click while "Thinking…" replaces the pending question just like typing.
With HEARTWATCH_ASK_DEV on, the understood line also says who answered and
each question is logged locally (ask/devlog.py).

Focusing the input preloads the model (at most once per Ollama keep-alive
period) so the first AI answer is quicker.
"""

from __future__ import annotations

import time
from pathlib import Path

from PySide6.QtCore import QEvent, QObject, Qt, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .. import config
from ..ask import answers, devlog, pipeline
from ..llm import client as llm_client
from ..llm.worker import LlmWorker
from .pending_request import PendingRequest
from .theme import Theme

# How long an AI answer may take before the bar gives up on it (the same
# budget as the session summary's).
ASK_TIMEOUT_MS = 10_000

PLACEHOLDER = 'Ask about your data — e.g. "What was my avg HR during walking today?"'


class SparkleIcon(QWidget):
    def __init__(self, color: str, parent: QWidget | None = None):
        super().__init__(parent)
        self._color = color
        self.setFixedSize(16, 16)

    def paintEvent(self, _e) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setBrush(QColor(self._color))
        p.setPen(Qt.NoPen)
        cx, cy = 8, 8
        p.drawPolygon(
            _diamond(cx, cy, 7, 3)
        )


def _diamond(cx, cy, ry, rx):
    from PySide6.QtCore import QPointF

    return [
        QPointF(cx, cy - ry),
        QPointF(cx + rx, cy),
        QPointF(cx, cy + ry),
        QPointF(cx - rx, cy),
    ]


class AiBar(QWidget):
    query_submitted = Signal(str)

    def __init__(
        self,
        theme: Theme,
        llm_worker: LlmWorker | None = None,
        db_path: str | Path | None = None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setObjectName("AiBar")
        self.theme = theme
        self._llm_worker = llm_worker
        # None: the app's database. Tests point this at a fixture.
        self._db_path = db_path
        # None: ask/devlog.py's LOG_PATH (HEARTWATCH_ASK_DEV only). Tests
        # point this at a temporary file.
        self._dev_log_path: Path | None = None
        # Last known AI status: None until the first health check answers.
        self._ai_ready: bool | None = None
        self._last_preload: float | None = None
        self._pending = PendingRequest(ASK_TIMEOUT_MS, self)
        self._pending.timed_out.connect(self._on_timeout)
        self._pending_question: pipeline.PendingQuestion | None = None
        if llm_worker is not None:
            llm_worker.parse_finished.connect(self._on_parse_finished)
            llm_worker.parse_failed.connect(self._on_parse_failed)
            llm_worker.health_checked.connect(self._on_health_checked)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 10, 20, 12)
        outer.setSpacing(8)

        # -- expandable result area (hidden until a query is made) --
        self.result = QFrame()
        self.result.setObjectName("AiResult")
        res_lay = QHBoxLayout(self.result)
        res_lay.setContentsMargins(12, 10, 8, 10)
        texts = QVBoxLayout()
        texts.setSpacing(4)
        self.result_text = QLabel("")
        self.result_text.setWordWrap(True)
        # "Average heart rate · Running · this week": muted and smaller.
        self.understood_text = QLabel("")
        self.understood_text.setObjectName("Muted")
        font = self.understood_text.font()
        font.setPointSizeF(font.pointSizeF() * 0.9)
        self.understood_text.setFont(font)
        self.caption_text = QLabel("")
        self.caption_text.setWordWrap(True)
        self.caption_text.setObjectName("Muted")
        for label in (self.result_text, self.understood_text, self.caption_text):
            texts.addWidget(label)
        # Up to three clickable questions (suggestions, or the choices of
        # "Did you mean ...?"); clicking one asks it exactly as if typed.
        self.suggestion_row = QHBoxLayout()
        self.suggestion_row.setSpacing(6)
        self.suggestion_buttons: list[QPushButton] = []
        texts.addLayout(self.suggestion_row)
        dismiss = QPushButton("✕")
        dismiss.setObjectName("GhostButton")
        dismiss.setFixedSize(24, 24)
        dismiss.clicked.connect(lambda: self.result.setHidden(True))
        res_lay.addLayout(texts, 1)
        res_lay.addWidget(dismiss, 0, Qt.AlignTop)
        self.result.setHidden(True)
        outer.addWidget(self.result)

        # -- input row --
        row = QHBoxLayout()
        row.setSpacing(10)
        row.addWidget(SparkleIcon(theme.c("accent")))

        self.input = QLineEdit()
        self.input.setObjectName("AiInput")
        self.input.setPlaceholderText(PLACEHOLDER)
        self.input.returnPressed.connect(self._submit)
        self.input.installEventFilter(self)

        self.ask_btn = QPushButton("Ask")
        self.ask_btn.setObjectName("AccentButton")
        self.ask_btn.clicked.connect(self._submit)

        row.addWidget(self.input, 1)
        row.addWidget(self.ask_btn, 0)
        outer.addLayout(row)

    def _submit(self) -> None:
        text = self.input.text().strip()
        if not text:
            return
        self.query_submitted.emit(text)
        self.input.clear()
        # A new question replaces any answer still on its way.
        self._pending.cancel()
        self._pending_question = None

        step = pipeline.begin(text, conn_or_path=self._db_path)
        if isinstance(step, pipeline.Answer):
            self.show_answer(step, text)
            return
        if self._llm_worker is None or self._ai_ready is False:
            self.show_answer(self._finish(step, failure=pipeline.FAILURE_UNAVAILABLE), text)
            return
        self._pending_question = step
        self.show_result(answers.THINKING_TEXT)
        self._pending.start(self._llm_worker.request_parse(step.messages, step.format))

    def _finish(self, step: pipeline.PendingQuestion, **reply) -> pipeline.Answer:
        return pipeline.finish(step, conn_or_path=self._db_path, **reply)

    def _take_pending(self, request_id: int) -> pipeline.PendingQuestion | None:
        """The question `request_id` answers, if it's still the one shown."""
        if not self._pending.claim(request_id):
            return None
        step, self._pending_question = self._pending_question, None
        return step

    def submit_text(self, question: str) -> None:
        """Asks `question` exactly as if it had been typed (a suggestion or
        "Did you mean ...?" button). A question still waiting on the model
        is replaced, as for a typed one."""
        self.input.setText(question)
        self._submit()

    def _on_parse_finished(self, request_id: int, raw: str) -> None:
        step = self._take_pending(request_id)
        if step is not None:
            self.show_answer(self._finish(step, raw=raw), step.question)

    def _on_parse_failed(self, request_id: int, reason: str) -> None:
        step = self._take_pending(request_id)
        if step is None:
            return
        failure = (
            pipeline.FAILURE_UNAVAILABLE if reason == LlmWorker.PARSE_UNAVAILABLE
            else pipeline.FAILURE_ERROR
        )
        self.show_answer(self._finish(step, failure=failure), step.question)

    def _on_timeout(self, request_id: int) -> None:
        step, self._pending_question = self._pending_question, None
        if step is not None:
            self.show_answer(self._finish(step, failure=pipeline.FAILURE_TIMEOUT), step.question)

    def _on_health_checked(self, status) -> None:
        self._ai_ready = status.ready

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802
        if watched is self.input and event.type() == QEvent.FocusIn:
            self._maybe_preload()
        return super().eventFilter(watched, event)

    def _maybe_preload(self) -> None:
        """Loads the model when the input gets focus, at most once per
        Ollama keep-alive period, and only in modes that use the model."""
        if self._llm_worker is None or self._ai_ready is False:
            return
        if config.ask_mode() == config.ASK_MODE_RULES:
            return
        now = time.monotonic()
        last = self._last_preload
        if last is not None and now - last < llm_client.DEFAULT_KEEP_ALIVE_S:
            return
        self._last_preload = now
        self._llm_worker.preload()

    def show_answer(self, answer: pipeline.Answer, question: str | None = None) -> None:
        """Shows `answer`. With HEARTWATCH_ASK_DEV on, the understood line
        ends with who answered and `question` is added to the local log."""
        understood = answer.understood
        if config.ask_dev():
            understood = devlog.with_marker(understood, answer)
            if question is not None:
                devlog.append(question, answer, self._dev_log_path)
        self.show_result(answer.text, understood, answer.caption, answer.suggestions)

    def show_result(
        self, text: str, understood: str | None = None, caption: str | None = None,
        suggestions: tuple = (),
    ) -> None:
        self.result_text.setText(text)
        self.understood_text.setText(understood or "")
        self.understood_text.setHidden(understood is None)
        self.caption_text.setText(caption or "")
        self.caption_text.setHidden(caption is None)
        self._set_suggestions(suggestions)
        self.result.setHidden(False)

    def _set_suggestions(self, suggestions: tuple) -> None:
        for button in self.suggestion_buttons:
            self.suggestion_row.removeWidget(button)
            button.deleteLater()
        self.suggestion_buttons = []
        # ask/suggest.py keeps suggestions to three; a clarification shows
        # every choice ("Which activity?" has four).
        for option in suggestions:
            button = QPushButton(option.label)
            button.setObjectName("GhostButton")
            button.clicked.connect(lambda _=False, q=option.question: self.submit_text(q))
            self.suggestion_row.addWidget(button)
            self.suggestion_buttons.append(button)
