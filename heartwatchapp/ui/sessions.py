"""Section 3 — Sessions. Filterable history backed by list_sessions();
double-clicking a row opens a detail dialog backed by get_session_timeline().
"""

from __future__ import annotations

import datetime

from matplotlib.figure import Figure
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..data import db
from .charts import _CanvasHost
from .theme import Theme
from .widgets import Card, PlaceholderState, clear_layout, heading, muted

DATE_RANGES = {"Last 7 days": 7, "Last 30 days": 30, "All time": None}
ACTIVITY_FILTERS = ["All activities", "Sitting", "Walking", "Standing", "Running"]


def _now_ms() -> int:
    return int(datetime.datetime.now().timestamp() * 1000)


def _format_dt(ts_ms: int | None) -> str:
    if ts_ms is None:
        return "--"
    return datetime.datetime.fromtimestamp(ts_ms / 1000).strftime("%b %d, %H:%M")


def _format_duration(duration_s: float) -> str:
    if duration_s >= 3600:
        return f"{duration_s / 3600:.1f} hr"
    return f"{duration_s / 60:.0f} min"


class SessionsView(QWidget):
    def __init__(self, theme: Theme, parent: QWidget | None = None):
        super().__init__(parent)
        self.theme = theme

        self._root = QVBoxLayout(self)
        self._root.setContentsMargins(0, 0, 0, 0)
        self._root.setSpacing(16)

        # -- filter bar --
        filters = Card(pad=12)
        row = QHBoxLayout()
        row.setSpacing(10)
        self._date_filter = QComboBox()
        self._date_filter.addItems(list(DATE_RANGES.keys()))
        self._date_filter.currentIndexChanged.connect(self.refresh)
        self._type_filter = QComboBox()
        self._type_filter.addItems(ACTIVITY_FILTERS)
        self._type_filter.currentIndexChanged.connect(self.refresh)
        row.addWidget(muted("Filter"))
        row.addWidget(self._date_filter)
        row.addWidget(self._type_filter)
        row.addStretch(1)
        holder = QWidget()
        holder.setLayout(row)
        filters.body().addWidget(holder)
        self._root.addWidget(filters)

        self._content = QVBoxLayout()
        self._root.addLayout(self._content, 1)

        self.refresh()

    def refresh(self) -> None:
        clear_layout(self._content)

        days = DATE_RANGES[self._date_filter.currentText()]
        start_date = _now_ms() - days * 86_400_000 if days is not None else None
        activity = self._type_filter.currentText()
        activity_filter = None if activity == "All activities" else activity

        sessions = db.list_sessions(
            limit=100, activity_filter=activity_filter, start_date=start_date
        )

        if not sessions:
            empty = Card()
            empty.body().addWidget(
                PlaceholderState(
                    "No sessions recorded yet",
                    "Recorded sessions will appear here once the device streams "
                    "data into SQLite, or after seeding one from "
                    "Settings -> Developer -> Seed demo session.",
                    self.theme,
                )
            )
            self._content.addWidget(empty, 1)
            return

        card = Card()
        table = QTableWidget(len(sessions), 6)
        table.setHorizontalHeaderLabels(
            ["Started", "Duration", "Activity", "Samples", "Avg BPM", "Peak BPM"]
        )
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        for row_idx, s in enumerate(sessions):
            avg_bpm = f"{s['avg_bpm']:.0f}" if s["avg_bpm"] is not None else "--"
            peak_bpm = f"{s['peak_bpm']:.0f}" if s["peak_bpm"] is not None else "--"
            values = [
                _format_dt(s["started_at"]),
                _format_duration(s["duration_s"]),
                s["label"] or "Unlabeled",
                str(s["sample_count"]),
                avg_bpm,
                peak_bpm,
            ]
            for col, text in enumerate(values):
                table.setItem(row_idx, col, QTableWidgetItem(text))

        session_ids = [s["id"] for s in sessions]
        table.cellDoubleClicked.connect(
            lambda r, _c, ids=session_ids: self._open_detail(ids[r])
        )
        card.body().addWidget(heading(f"{len(sessions)} session(s) · double-click for detail"))
        card.body().addWidget(table)
        self._content.addWidget(card, 1)

    def _open_detail(self, session_id: int) -> None:
        dlg = SessionDetailDialog(session_id, self.theme, self)
        dlg.exec()


class _SessionHRChart(_CanvasHost):
    """Static HR-over-time line for the session detail dialog."""

    def __init__(self, theme: Theme, hr_series: list[tuple[int, float]], parent: QWidget | None = None):
        super().__init__(parent)
        self.setMinimumHeight(200)
        fig = Figure(figsize=(5, 2.4), dpi=100)
        fig.patch.set_alpha(0.0)
        ax = fig.add_subplot(111)

        muted_c = theme.c("text_muted")
        border = theme.c("border")
        ax.set_facecolor("none")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(border)
        ax.tick_params(colors=muted_c, labelsize=8)
        ax.grid(True, color=border, linewidth=0.5, alpha=0.5)

        if hr_series:
            t0 = hr_series[0][0]
            xs = [(ts - t0) / 1000 for ts, _ in hr_series]
            ys = [bpm for _, bpm in hr_series]
            ax.plot(xs, ys, color=theme.c("danger"), linewidth=1.6)
            ax.fill_between(xs, ys, min(ys) - 3, color=theme.c("danger"), alpha=0.12)
            ax.set_xlabel("seconds into session", color=muted_c, fontsize=8)

        fig.tight_layout(pad=0.4)
        self._mount(fig)


class SessionDetailDialog(QDialog):
    def __init__(self, session_id: int, theme: Theme, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle(f"Session #{session_id}")
        self.resize(560, 460)

        timeline = db.get_session_timeline(session_id)
        session = timeline["session"] or {}

        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 20, 20, 20)
        lay.setSpacing(10)

        label = session.get("label") or "Unlabeled session"
        lay.addWidget(heading(f"Session #{session_id} · {label}"))
        lay.addWidget(
            muted(
                f"Started {_format_dt(session.get('started_at'))} · "
                f"Device: {session.get('device_id') or 'unknown'} · "
                f"Scales: accel x{session.get('accel_scale')}, gyro x{session.get('gyro_scale')}"
            )
        )
        lay.addWidget(
            muted(
                f"IMU samples: {len(timeline['imu'])} · "
                f"HR samples: {len(timeline['hr'])} · "
                f"Predictions: {len(timeline['predictions'])}"
            )
        )

        if timeline["hr"]:
            lay.addWidget(_SessionHRChart(theme, timeline["hr"]))
        else:
            lay.addWidget(muted("No heart-rate samples recorded for this session."))

        lay.addStretch(1)
