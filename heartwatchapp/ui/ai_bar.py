"""Always-visible data-query assistant bar.

Layout only for now (shell milestone). The Anthropic API worker thread lands in
step 4 of the build order -- submitting today just shows a stub result panel.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
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

from .theme import Theme

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

    def __init__(self, theme: Theme, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("AiBar")
        self.theme = theme

        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 10, 20, 12)
        outer.setSpacing(8)

        # -- expandable result area (hidden until a query is made) --
        self.result = QFrame()
        self.result.setObjectName("AiResult")
        res_lay = QHBoxLayout(self.result)
        res_lay.setContentsMargins(12, 10, 8, 10)
        self.result_text = QLabel("")
        self.result_text.setWordWrap(True)
        self.result_text.setObjectName("Muted")
        dismiss = QPushButton("✕")
        dismiss.setObjectName("GhostButton")
        dismiss.setFixedSize(24, 24)
        dismiss.clicked.connect(lambda: self.result.setHidden(True))
        res_lay.addWidget(self.result_text, 1)
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
        # Stub response until the Anthropic worker thread is wired (build step 4).
        self.show_result(
            f'Query received: "{text}"\n\n'
            "The data assistant is not connected yet — SQLite context + the "
            "Anthropic API worker land in build step 4."
        )
        self.input.clear()

    def show_result(self, text: str) -> None:
        self.result_text.setText(text)
        self.result.setHidden(False)
