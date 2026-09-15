"""Small shared widgets used across sections: Card, MetricCard, Badge, Divider."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from .theme import Theme


class Card(QFrame):
    """Flat rounded surface. 14px inner padding per the handoff."""

    def __init__(self, parent: QWidget | None = None, pad: int = 14):
        super().__init__(parent)
        self.setObjectName("Card")
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(pad, pad, pad, pad)
        self._layout.setSpacing(10)

    def body(self) -> QVBoxLayout:
        return self._layout

    def add(self, w: QWidget) -> None:
        self._layout.addWidget(w)


class MetricCard(Card):
    """Dashboard top-row card: 11px muted label, 26px value, 12px colored unit."""

    def __init__(
        self,
        label: str,
        value: str,
        unit: str,
        subtitle: str,
        value_color: str,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        lbl = QLabel(label)
        lbl.setObjectName("CardLabel")

        row = QHBoxLayout()
        row.setSpacing(4)
        row.setContentsMargins(0, 0, 0, 0)
        self.value_label = QLabel(value)
        self.value_label.setObjectName("CardValue")
        self.value_label.setStyleSheet(f"color: {value_color};")
        unit_label = QLabel(unit)
        unit_label.setObjectName("CardUnit")
        unit_label.setStyleSheet(f"color: {value_color};")
        unit_label.setAlignment(Qt.AlignBottom)
        row.addWidget(self.value_label)
        row.addWidget(unit_label)
        row.addStretch(1)

        sub = QLabel(subtitle)
        sub.setObjectName("Muted")

        self.body().addWidget(lbl)
        self.body().addLayout(row)
        self.body().addWidget(sub)
        self.body().addStretch(1)


class Badge(QLabel):
    """Filled pill: colored background, white text. Used for activity labels."""

    def __init__(self, text: str, bg_color: str, parent: QWidget | None = None):
        super().__init__(text, parent)
        self.set_color(bg_color)
        self.setAlignment(Qt.AlignCenter)

    def set_color(self, bg_color: str) -> None:
        self.setStyleSheet(
            f"background: {bg_color}; color: #ffffff; font-weight: 500;"
            "font-size: 12px; border-radius: 8px; padding: 5px 12px;"
        )


class Divider(QFrame):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("Divider")
        self.setFrameShape(QFrame.HLine)


def muted(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("Muted")
    return lbl


def heading(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("SectionHeading")
    return lbl


def clear_layout(layout) -> None:
    """Removes and deletes every item in a layout, recursing into nested
    layouts. Used by views (Dashboard, Sessions) that rebuild their content
    in place on refresh() rather than being fully torn down and recreated."""
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.deleteLater()
            continue
        sub_layout = item.layout()
        if sub_layout is not None:
            clear_layout(sub_layout)


class PlaceholderState(QWidget):
    """Centered empty-state used by the stubbed sections (Sessions / ML / Settings)."""

    def __init__(self, title: str, detail: str, theme: Theme, parent: QWidget | None = None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignCenter)
        lay.setSpacing(8)

        t = QLabel(title)
        t.setObjectName("SectionHeading")
        t.setAlignment(Qt.AlignCenter)

        d = QLabel(detail)
        d.setObjectName("Muted")
        d.setAlignment(Qt.AlignCenter)
        d.setWordWrap(True)
        d.setMaximumWidth(420)

        eff = QGraphicsOpacityEffect(t)
        eff.setOpacity(0.9)
        t.setGraphicsEffect(eff)

        lay.addWidget(t)
        lay.addWidget(d)
