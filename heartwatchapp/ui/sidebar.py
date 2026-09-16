"""64px icon-only vertical nav. Tabler-style glyphs drawn with QPainter so the
app has no icon-font dependency for the prototype.
"""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .theme import Theme

# name -> ordered nav index. Order matches the handoff.
SECTIONS = [
    ("layout-dashboard", "Dashboard"),
    ("activity", "Live Monitor"),
    ("history", "Sessions"),
    ("brain", "ML / Activity"),
    ("settings", "Settings"),
]


def _icon(name: str, color: str) -> QIcon:
    """Minimal line glyphs, 24x24 on a 44px canvas, stroked in `color`."""
    pm = QPixmap(44, 44)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    pen = QPen(QColor(color))
    pen.setWidth(2)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    p.setPen(pen)
    p.translate(10, 10)  # 24px glyph centered in 44px

    if name == "layout-dashboard":
        p.drawRoundedRect(1, 1, 9, 13, 2, 2)
        p.drawRoundedRect(14, 1, 9, 7, 2, 2)
        p.drawRoundedRect(14, 12, 9, 11, 2, 2)
        p.drawRoundedRect(1, 18, 9, 5, 2, 2)
    elif name == "activity":
        p.drawPolyline(_pts([(0, 12), (6, 12), (9, 3), (14, 21), (18, 12), (24, 12)]))
    elif name == "history":
        p.drawArc(1, 1, 22, 22, 40 * 16, 300 * 16)
        p.drawPolyline(_pts([(1, 2), (1, 8), (7, 8)]))
        p.drawPolyline(_pts([(12, 6), (12, 12), (17, 15)]))
    elif name == "brain":
        p.drawEllipse(3, 4, 8, 16)
        p.drawEllipse(13, 4, 8, 16)
        p.drawLine(12, 4, 12, 20)
    elif name == "settings":
        p.drawEllipse(8, 8, 8, 8)
        for ang in range(0, 360, 45):
            p.save()
            p.translate(12, 12)
            p.rotate(ang)
            p.drawLine(0, -11, 0, -8)
            p.restore()
    p.end()
    return QIcon(pm)


def _pts(seq):
    from PySide6.QtCore import QPointF

    return [QPointF(x, y) for x, y in seq]


class Sidebar(QWidget):
    section_changed = Signal(int)

    def __init__(self, theme: Theme, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("Sidebar")
        self.setFixedWidth(64)
        self.theme = theme

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 12, 8, 12)
        lay.setSpacing(6)

        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self._buttons: list[QToolButton] = []

        for idx, (icon_name, tooltip) in enumerate(SECTIONS):
            if idx == len(SECTIONS) - 1:
                lay.addStretch(1)
                div = QWidget()
                div.setFixedHeight(1)
                div.setStyleSheet(f"background: {theme.c('border')};")
                lay.addWidget(div)
                lay.addSpacing(6)

            btn = QToolButton(self)
            btn.setObjectName("NavButton")
            btn.setCheckable(True)
            btn.setToolTip(tooltip)
            btn.setIconSize(QSize(44, 44))
            btn.setIcon(_icon(icon_name, theme.c("text_muted")))
            btn.setFixedSize(48, 48)
            btn._icon_name = icon_name  # noqa: SLF001  (stash for restyle)
            self.group.addButton(btn, idx)
            self._buttons.append(btn)
            lay.addWidget(btn)

        self._buttons[0].setChecked(True)
        self.group.idClicked.connect(self._on_click)
        self.restyle()

    def _on_click(self, idx: int) -> None:
        self.restyle()
        self.section_changed.emit(idx)

    def restyle(self) -> None:
        """Recolor glyphs so checked = accent, others = muted (theme-aware)."""
        for btn in self._buttons:
            color = (
                self.theme.c("accent")
                if btn.isChecked()
                else self.theme.c("text_muted")
            )
            btn.setIcon(_icon(btn._icon_name, color))  # noqa: SLF001
