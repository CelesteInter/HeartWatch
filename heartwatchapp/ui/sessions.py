"""Section 3 — Sessions. Filterable history backed by list_sessions();
double-clicking a row opens a detail dialog backed by get_session_timeline().
"""

from __future__ import annotations

import datetime
import logging

from matplotlib.figure import Figure
from PySide6.QtCore import QTimer
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

from .. import config
from ..data import db, stats
from ..llm.worker import LlmWorker
from .charts import _CanvasHost
from .theme import Theme
from .widgets import Card, Divider, PlaceholderState, clear_layout, heading, muted

DATE_RANGES = {"Last 7 days": 7, "Last 30 days": 30, "All time": None}
ACTIVITY_FILTERS = ["All activities", "Sitting", "Walking", "Standing", "Running"]

# How long the session detail dialog waits for an AI summary before giving
# up and showing plain numbers. A cold model load takes ~2.3s on the dev
# M2, so this only fires if something is genuinely wrong.
SUMMARY_TIMEOUT_MS = 10_000
GENERATING_TEXT = "Generating summary…"
# Shown instead of calling the LLM when stats.is_summarizable() is False --
# the session is too short, has too few HR samples, or has no activity
# predictions yet. Worded to be true for all three.
NOT_ENOUGH_DATA_TEXT = "Not enough data for a generated summary yet."

logger = logging.getLogger("heartwatch.llm")


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
    def __init__(
        self,
        theme: Theme,
        llm_worker: LlmWorker | None = None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.theme = theme
        self.llm_worker = llm_worker

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
        dlg = SessionDetailDialog(session_id, self.theme, self.llm_worker, self)
        dlg.exec()
        # One dialog per double-click; free it once closed instead of
        # letting closed dialogs pile up as children of this view.
        dlg.deleteLater()


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
    def __init__(
        self,
        session_id: int,
        theme: Theme,
        llm_worker: LlmWorker | None = None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle(f"Session #{session_id}")
        self.resize(560, 460)
        self._session_id = session_id
        self._llm_worker = llm_worker

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

        lay.addWidget(Divider())
        lay.addWidget(self._summary_card())

        lay.addStretch(1)

    # -- Session summary: plain by default, AI-phrased if asked for ------
    #
    # In order: compute the numbers -> HEARTWATCH_SUMMARY_MODE is "plain"
    # (the default; see config.py)? show the app's own sentences and stop
    # -> too little data? show plain numbers and stop -> otherwise ask the
    # LLM worker (which strips, checks, and logs the reply before sending
    # it back) -> show the reply only if it passed, else plain numbers. The
    # whole reply arrives at once, never piece by piece, so nothing
    # unchecked is ever on screen.
    def _summary_card(self) -> Card:
        card = Card(pad=16)
        card.body().addWidget(heading("Session summary"))
        self._summary_label = muted("")
        self._summary_label.setWordWrap(True)
        card.body().addWidget(self._summary_label)
        # Id of the summary request this dialog is still waiting on, or
        # None. Any reply with a different id (or arriving after the
        # timeout or after the dialog closed) is thrown away.
        self._pending_request_id: int | None = None
        self._summary_timer: QTimer | None = None

        try:
            session_stats = stats.compute_session_stats(self._session_id)
        except ValueError:
            self._summary_label.setText("Not enough data to summarize this session yet.")
            return card

        # The raw numbers are always available; the LLM's prose on top of
        # them is optional. This is what's shown if Ollama can't run, if
        # the reply fails validation, or if it takes too long.
        self._fallback_text = stats.format_stats_plain(session_stats)

        # The classifier-confidence caveat is written by the app, not the
        # model, and shown under whichever summary ends up on screen.
        caveat = stats.confidence_caveat(session_stats)
        if caveat:
            caveat_label = muted(caveat)
            caveat_label.setWordWrap(True)
            card.body().addWidget(caveat_label)

        if config.summary_mode() == config.SUMMARY_MODE_PLAIN:
            # The plain summary IS the summary here, so there's no
            # generated one to be missing -- no "not enough data" note.
            self._summary_label.setText(self._fallback_text)
            return card

        if not stats.is_summarizable(session_stats):
            # Too little data for the model to say anything real -- don't
            # call it at all.
            self._summary_label.setText(f"{self._fallback_text}\n\n{NOT_ENOUGH_DATA_TEXT}")
            return card

        if self._llm_worker is None:
            self._summary_label.setText(self._fallback_text)
            return card

        self._summary_label.setText(GENERATING_TEXT)
        self._llm_worker.summary_finished.connect(self._on_summary_finished)
        self._llm_worker.summary_failed.connect(self._on_summary_failed)
        self._summary_timer = QTimer(self)
        self._summary_timer.setSingleShot(True)
        self._summary_timer.timeout.connect(self._on_summary_timeout)
        self._summary_timer.start(SUMMARY_TIMEOUT_MS)
        self._pending_request_id = self._llm_worker.request_session_summary(session_stats)
        return card

    def _claim_reply(self, request_id: int) -> bool:
        """True (and stops waiting) if `request_id` is the reply this
        dialog is still waiting for; False for anything stale."""
        if self._pending_request_id is None or request_id != self._pending_request_id:
            return False
        self._pending_request_id = None
        if self._summary_timer is not None:
            self._summary_timer.stop()
        return True

    def _on_summary_finished(self, request_id: int, text: str) -> None:
        if not self._claim_reply(request_id):
            return
        self._summary_label.setText(text)

    def _on_summary_failed(self, request_id: int, reason: str) -> None:
        if not self._claim_reply(request_id):
            return
        self._summary_label.setText(
            f"{self._fallback_text}\n\n(AI summary unavailable: {reason}.)"
        )

    def _on_summary_timeout(self) -> None:
        if self._pending_request_id is None:
            return
        logger.warning(
            "summary timed out: session=%s request=%s after %d ms",
            self._session_id,
            self._pending_request_id,
            SUMMARY_TIMEOUT_MS,
        )
        # Forget the request, so a reply that turns up later is discarded.
        self._pending_request_id = None
        self._summary_label.setText(
            f"{self._fallback_text}\n\n(AI summary unavailable: no reply within "
            f"{SUMMARY_TIMEOUT_MS // 1000} seconds.)"
        )

    def done(self, result: int) -> None:
        # Every way of closing a QDialog (Esc, the close button, accept or
        # reject) goes through done(). Stop waiting and drop our signal
        # connections so a late reply for a closed dialog can never try to
        # update a deleted label.
        self._pending_request_id = None
        # The timer exists only if this dialog actually sent a request (and
        # so connected to the worker's signals) -- nothing to undo otherwise.
        if self._summary_timer is not None:
            self._summary_timer.stop()
            for signal, slot in (
                (self._llm_worker.summary_finished, self._on_summary_finished),
                (self._llm_worker.summary_failed, self._on_summary_failed),
            ):
                try:
                    signal.disconnect(slot)
                except (TypeError, RuntimeError):
                    pass
        super().done(result)
