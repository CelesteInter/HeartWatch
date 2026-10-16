"""Section 1 — Dashboard. Apple-Health-style card grid, backed by SQLite.

Every number, bar, donut slice and "Recent sessions" row comes from
data/queries.py: the four cards from run(), the bars and donut from one
activity_breakdown() result, the list from list_recent_sessions(). Nothing
here does arithmetic on the data or holds a demo value. The last 7 days
are calendar days in local time (queries.resolve_range()); sessions still
recording are left out until they end.
"""

from __future__ import annotations

import datetime
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from ..data import queries
from .charts import DonutChart
from .theme import Theme
from .widgets import (
    Card,
    Divider,
    MetricCard,
    PlaceholderState,
    clear_layout,
    heading,
    muted,
)

DASHBOARD_RANGE = "last_7_days"
RANGE_TEXT = "Last 7 days"
RECENT_SESSIONS = 3

EMPTY_VALUE = "—"
# No finished session in the whole database (a brand-new install, or only
# the Live Monitor's still-recording session): the whole Dashboard is this
# one onboarding card. Finished sessions that just aren't in the range get
# the "—" cards and NO_SESSIONS_TEXT instead.
ONBOARDING_TITLE = "No sessions recorded yet"
ONBOARDING_DETAIL = (
    "Metrics appear here once a session has been recorded -- try "
    "Settings -> Developer -> Seed demo session."
)
NO_SESSIONS_TEXT = "No sessions in the last 7 days"
NO_ACTIVITY_TEXT = "No activity data yet"
NO_RECENT_TEXT = "No sessions recorded yet"
# Shown under the breakdown when any of its time comes from the dummy
# classifier (model_version "demo-dummy": the Live Monitor, the dev-panel
# seed and the demo script all write those until milestone 6).
SIMULATED_CAPTION = queries.SIMULATED_CAPTION
UNLABELED_ICON = "?"


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
    """One "Recent sessions" row. `activity` None (an unlabeled session)
    gets a neutral "?" icon in the muted color rather than an activity's;
    `peak_hr` None (no heart-rate samples) shows as EMPTY_VALUE."""

    def __init__(
        self, name: str, meta: str, activity: str | None, peak_hr: int | None, theme: Theme
    ):
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 6, 0, 6)
        lay.setSpacing(12)

        self.icon = QLabel()
        self.icon.setFixedSize(32, 32)
        tint = theme.activity_color(activity) if activity else theme.c("text_muted")
        self.icon.setStyleSheet(
            f"background: {tint}; border-radius: 8px; color: #ffffff;"
            "font-weight: 500; font-size: 11px;"
        )
        self.icon.setAlignment(Qt.AlignCenter)
        self.icon.setText(activity[0] if activity else UNLABELED_ICON)

        center = QVBoxLayout()
        center.setSpacing(1)
        center.setContentsMargins(0, 0, 0, 0)
        title = QLabel(name)
        title.setStyleSheet("font-size: 13px; font-weight: 500;")
        center.addWidget(title)
        center.addWidget(muted(meta))

        self.peak = QLabel(f"{peak_hr} bpm" if peak_hr is not None else EMPTY_VALUE)
        self.peak.setStyleSheet(
            "font-size: 13px; font-weight: 500;"
            + (f"color: {theme.c('danger')};" if peak_hr is not None and peak_hr > 130 else "")
        )
        self.peak.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        lay.addWidget(self.icon)
        lay.addLayout(center, 1)
        lay.addWidget(self.peak)


class DashboardView(QWidget):
    def __init__(
        self,
        theme: Theme,
        parent: QWidget | None = None,
        db_path: str | Path | None = None,
    ):
        """`db_path`: the database to read; None means the app's own
        (db.DB_PATH). Tests pass a fixture database."""
        super().__init__(parent)
        self.theme = theme
        self._db_path = db_path

        self._root = QVBoxLayout(self)
        self._root.setContentsMargins(0, 0, 0, 0)
        self._root.setSpacing(16)

        self.refresh()

    def refresh(self) -> None:
        clear_layout(self._root)
        with queries.connection(path=self._db_path) as conn:
            if queries.run(queries.MetricRequest("session_count", None, "all"), conn).empty:
                self._show_onboarding()
                return
            results = {
                metric: queries.run(queries.MetricRequest(metric, None, DASHBOARD_RANGE), conn)
                for metric in ("avg_hr", "total_time", "peak_hr", "session_count")
            }
            breakdown = queries.activity_breakdown(DASHBOARD_RANGE, conn=conn)
            recent = queries.list_recent_sessions(RECENT_SESSIONS, conn=conn)
        has_sessions = not results["session_count"].empty

        cards = QHBoxLayout()
        cards.setSpacing(14)
        for label, value, unit, subtitle, color_key in self._metric_specs(results):
            cards.addWidget(MetricCard(label, value, unit, subtitle, self.theme.c(color_key)))
        self._root.addLayout(cards)
        if not has_sessions:
            self._root.addWidget(muted(NO_SESSIONS_TEXT))

        middle = QHBoxLayout()
        middle.setSpacing(16)
        middle.addWidget(self._activity_card(breakdown, has_sessions), 1)
        middle.addWidget(self._sessions_card(recent), 1)
        self._root.addLayout(middle)
        self._root.addStretch(1)

    def _show_onboarding(self) -> None:
        card = Card()
        card.body().addWidget(PlaceholderState(ONBOARDING_TITLE, ONBOARDING_DETAIL, self.theme))
        self._root.addWidget(card, 1)

    def _metric_specs(
        self, results: dict[str, queries.MetricResult]
    ) -> list[tuple[str, str, str, str, str]]:
        def shown(result: queries.MetricResult, scale: float = 1.0) -> str:
            return EMPTY_VALUE if result.empty else f"{result.value / scale:.0f}"

        return [
            ("Heart rate", shown(results["avg_hr"]), "bpm", f"Avg · {RANGE_TEXT}", "accent"),
            ("Active time", shown(results["total_time"], 60), "min", RANGE_TEXT, "text"),
            ("Peak HR", shown(results["peak_hr"]), "bpm", RANGE_TEXT, "success"),
            ("Sessions", shown(results["session_count"]), "", RANGE_TEXT, "text"),
        ]

    def _activity_card(self, breakdown: queries.ActivityBreakdown, has_sessions: bool) -> Card:
        card = Card()
        card.body().addWidget(heading(f"Activity breakdown · {RANGE_TEXT}"))

        if not breakdown.items:
            card.body().addWidget(muted(NO_ACTIVITY_TEXT if has_sessions else NO_SESSIONS_TEXT))
            card.body().addStretch(1)
            return card

        # The bars and the donut both draw breakdown.items, so they can't
        # disagree. Percentages are only for the bar labels.
        total = sum(seconds for _, seconds in breakdown.items)

        split = QHBoxLayout()
        split.setSpacing(16)

        bars = QVBoxLayout()
        bars.setSpacing(8)
        for name, seconds in breakdown.items:
            bars.addWidget(
                ActivityBar(
                    name, round(100 * seconds / total), self.theme.activity_color(name), self.theme
                )
            )
        bars.addStretch(1)
        split.addLayout(bars, 1)

        donut = DonutChart(self.theme)
        donut.set_data(list(breakdown.items))
        split.addWidget(donut, 0, Qt.AlignVCenter)

        holder = QWidget()
        holder.setLayout(split)
        card.body().addWidget(holder)
        if breakdown.simulated:
            card.body().addWidget(muted(SIMULATED_CAPTION))
        return card

    def _sessions_card(self, sessions: list[dict]) -> Card:
        card = Card()
        card.body().addWidget(heading("Recent sessions"))
        if not sessions:
            card.body().addWidget(muted(NO_RECENT_TEXT))
        for i, s in enumerate(sessions):
            if i:
                card.body().addWidget(Divider())
            name = s["label"] or "Unlabeled session"
            meta = _format_session_meta(s["started_at"], s["duration_s"])
            card.body().addWidget(SessionRow(name, meta, s["label"], s["peak_bpm"], self.theme))
        card.body().addStretch(1)
        return card
