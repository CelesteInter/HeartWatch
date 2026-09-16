"""Section 1 — Dashboard. Apple-Health-style card grid, backed by SQLite.

get_dashboard_stats(days=7) supplies the four metric cards and the activity
breakdown; list_sessions(limit=3) supplies "Recent sessions" (not returned
by get_dashboard_stats itself, since that call is scoped to aggregates, not
individual session rows).
"""

from __future__ import annotations

import datetime

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from ..data import db
from .charts import DonutChart
from .theme import Theme
from .widgets import Card, Divider, MetricCard, PlaceholderState, clear_layout, heading, muted

DASHBOARD_WINDOW_DAYS = 7


def _format_session_meta(started_at_ms: int, duration_s: float) -> str:
    dt = datetime.datetime.fromtimestamp(started_at_ms / 1000)
    today = datetime.datetime.now().date()
    if dt.date() == today:
        day = "Today"
    elif dt.date() == today - datetime.timedelta(days=1):
        day = "Yesterday"
    else:
        day = dt.strftime("%b %d")
    dur = f"{duration_s / 3600:.1f} hr" if duration_s >= 3600 else f"{duration_s / 60:.0f} min"
    return f"{day} · {dur}"


class ActivityBar(QWidget):
    """One row: name (60px) · track (flex, 6px) · percent (right)."""

    def __init__(self, name: str, pct: int, color: str, theme: Theme):
        super().__init__()
        self._pct = pct
        self._color = color
        self._track = theme.c("surface_2")
        self.setFixedHeight(18)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)

        name_lbl = QLabel(name)
        name_lbl.setFixedWidth(60)
        name_lbl.setStyleSheet("font-size: 12px;")

        self._bar = _BarTrack(pct, color, self._track)

        pct_lbl = QLabel(f"{pct}%")
        pct_lbl.setObjectName("Muted")
        pct_lbl.setFixedWidth(34)
        pct_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        lay.addWidget(name_lbl)
        lay.addWidget(self._bar, 1)
        lay.addWidget(pct_lbl)


class _BarTrack(QWidget):
    def __init__(self, pct: int, color: str, track: str):
        super().__init__()
        self._pct = pct
        self._color = color
        self._track = track
        self.setMinimumHeight(6)

    def paintEvent(self, _e):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        h = 6
        y = (self.height() - h) // 2
        p.setBrush(QColor(self._track))
        p.drawRoundedRect(0, y, self.width(), h, 3, 3)
        p.setBrush(QColor(self._color))
        w = int(self.width() * self._pct / 100)
        p.drawRoundedRect(0, y, w, h, 3, 3)


class SessionRow(QWidget):
    def __init__(self, name: str, meta: str, activity: str, peak_hr: int, theme: Theme):
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 6, 0, 6)
        lay.setSpacing(12)

        icon = QLabel()
        icon.setFixedSize(32, 32)
        tint = theme.activity_color(activity)
        icon.setStyleSheet(
            f"background: {tint}; border-radius: 8px; color: #ffffff;"
            "font-weight: 500; font-size: 11px;"
        )
        icon.setAlignment(Qt.AlignCenter)
        icon.setText(activity[0])

        center = QVBoxLayout()
        center.setSpacing(1)
        center.setContentsMargins(0, 0, 0, 0)
        title = QLabel(name)
        title.setStyleSheet("font-size: 13px; font-weight: 500;")
        center.addWidget(title)
        center.addWidget(muted(meta))

        peak = QLabel(f"{peak_hr} bpm")
        peak.setStyleSheet(
            "font-size: 13px; font-weight: 500;"
            + (f"color: {theme.c('danger')};" if peak_hr > 130 else "")
        )
        peak.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        lay.addWidget(icon)
        lay.addLayout(center, 1)
        lay.addWidget(peak)


class DashboardView(QWidget):
    def __init__(self, theme: Theme, parent: QWidget | None = None):
        super().__init__(parent)
        self.theme = theme

        self._root = QVBoxLayout(self)
        self._root.setContentsMargins(0, 0, 0, 0)
        self._root.setSpacing(16)

        self.refresh()

    def refresh(self) -> None:
        clear_layout(self._root)
        stats = db.get_dashboard_stats(days=DASHBOARD_WINDOW_DAYS)

        if stats["session_count"] == 0:
            empty = Card()
            empty.body().addWidget(
                PlaceholderState(
                    "No sessions recorded yet",
                    "Metrics appear here once a session has been recorded -- try "
                    "Settings -> Developer -> Seed demo session.",
                    self.theme,
                )
            )
            self._root.addWidget(empty, 1)
            return

        recent = db.list_sessions(limit=3)

        cards = QHBoxLayout()
        cards.setSpacing(14)
        for label, value, unit, subtitle, color_key in self._metric_specs(stats):
            cards.addWidget(MetricCard(label, value, unit, subtitle, self.theme.c(color_key)))
        self._root.addLayout(cards)

        middle = QHBoxLayout()
        middle.setSpacing(16)
        middle.addWidget(self._activity_card(stats["activity_breakdown"]), 1)
        middle.addWidget(self._sessions_card(recent), 1)
        self._root.addLayout(middle)
        self._root.addStretch(1)

    def _metric_specs(self, stats: dict) -> list[tuple[str, str, str, str, str]]:
        avg_hr = stats["avg_hr"]
        peak_hr = stats["peak_hr"]
        active_min = stats["total_active_seconds"] / 60
        window = f"Last {DASHBOARD_WINDOW_DAYS} days"
        return [
            ("Heart rate", f"{avg_hr:.0f}" if avg_hr is not None else "--", "bpm", f"Avg · {window}", "accent"),
            ("Active time", f"{active_min:.0f}", "min", window, "text"),
            ("Peak HR", f"{peak_hr:.0f}" if peak_hr is not None else "--", "bpm", window, "success"),
            ("Sessions", str(stats["session_count"]), "", window, "text"),
        ]

    def _activity_card(self, breakdown: dict[str, float]) -> Card:
        card = Card()
        card.body().addWidget(heading(f"Activity breakdown · Last {DASHBOARD_WINDOW_DAYS} days"))

        total = sum(breakdown.values()) or 1.0
        pairs = sorted(
            ((name, round(100 * secs / total)) for name, secs in breakdown.items()),
            key=lambda pair: -pair[1],
        )

        split = QHBoxLayout()
        split.setSpacing(16)

        bars = QVBoxLayout()
        bars.setSpacing(8)
        for name, pct in pairs:
            bars.addWidget(
                ActivityBar(name, pct, self.theme.activity_color(name), self.theme)
            )
        bars.addStretch(1)
        split.addLayout(bars, 1)

        donut = DonutChart(self.theme)
        donut.set_data(pairs)
        split.addWidget(donut, 0, Qt.AlignVCenter)

        holder = QWidget()
        holder.setLayout(split)
        card.body().addWidget(holder)
        return card

    def _sessions_card(self, sessions: list[dict]) -> Card:
        card = Card()
        card.body().addWidget(heading("Recent sessions"))
        for i, s in enumerate(sessions):
            if i:
                card.body().addWidget(Divider())
            name = s["label"] or "Unlabeled session"
            meta = _format_session_meta(s["started_at"], s["duration_s"])
            peak = int(round(s["peak_bpm"])) if s["peak_bpm"] is not None else 0
            card.body().addWidget(SessionRow(name, meta, s["label"] or "Sitting", peak, self.theme))
        card.body().addStretch(1)
        return card
