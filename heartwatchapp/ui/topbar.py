"""52px topbar: section title (left), status subtitle (center-left),
connection + battery indicator (right)."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QHBoxLayout, QLabel, QWidget

from .theme import Theme


class BatteryIndicator(QWidget):
    def __init__(self, theme: Theme, parent: QWidget | None = None):
        super().__init__(parent)
        self.theme = theme
        self.level = 0.72  # 0..1, dummy
        self.setFixedSize(30, 16)

    def set_level(self, level: float) -> None:
        self.level = max(0.0, min(1.0, level))
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor(self.theme.c("text_muted")))
        pen.setWidth(1)
        p.setPen(pen)
        body = self.rect().adjusted(1, 2, -4, -2)
        p.drawRoundedRect(body, 2, 2)
        p.drawRect(body.right() + 1, body.center().y() - 3, 2, 6)
        fill_color = (
            self.theme.c("danger") if self.level < 0.2 else self.theme.c("success")
        )
        p.setBrush(QColor(fill_color))
        p.setPen(Qt.NoPen)
        fw = int((body.width() - 3) * self.level)
        p.drawRoundedRect(body.left() + 2, body.top() + 2, fw, body.height() - 4, 1, 1)


class Topbar(QWidget):
    def __init__(self, theme: Theme, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("Topbar")
        self.setFixedHeight(52)
        self.theme = theme

        lay = QHBoxLayout(self)
        lay.setContentsMargins(20, 0, 20, 0)
        lay.setSpacing(12)

        self.title = QLabel("Dashboard")
        self.title.setObjectName("TopbarTitle")

        self.subtitle = QLabel("Today's overview")
        self.subtitle.setObjectName("TopbarSubtitle")
        self.subtitle.setAlignment(Qt.AlignVCenter)

        lay.addWidget(self.title)
        lay.addWidget(self.subtitle)
        lay.addStretch(1)

        self.conn_pill = QLabel("Disconnected")
        self.conn_pill.setObjectName("ConnPill")
        self.battery = BatteryIndicator(theme)

        lay.addWidget(self.conn_pill)
        lay.addWidget(self.battery)

    def set_section(self, title: str, subtitle: str) -> None:
        self.title.setText(title)
        self.subtitle.setText(subtitle)

    def set_connection(self, connected: bool, name: str = "") -> None:
        self.conn_pill.setText(f"Connected · {name}" if connected else "Disconnected")
