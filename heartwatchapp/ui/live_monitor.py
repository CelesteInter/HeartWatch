"""Section 2 — Live Monitor. Control row on top; below it, two columns:
HR chart (left ~60%) · stacked panels (right).

The view only displays. Readings come from recording/controller.py's
RecordingController (which the main window owns, so a recording survives
a theme rebuild of this view), fed by a data source -- today the dummy
generators in sources/dummy.py, later the real watch. Before Start the
readings are a preview and nothing is saved; Start opens a session
(optionally labeled with an activity) and Stop ends it.

The activity badge shows the source's predictions -- for now the dummy
classifier (model_version "demo-dummy", data/seed.py).
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..ml.inference import CLASSES
from ..recording.controller import ERROR, RECORDING, RecordingController
from ..sources.base import HrReading, ImuReading, PredictionReading
from .charts import LiveHRChart
from .theme import Theme
from .widgets import Badge, Card, heading, muted

# Raw stored IMU counts = on-screen value (g, deg/s) x this, for the dummy
# source (sources/dummy.py). The sessions row's accel_scale / gyro_scale are
# still placeholder 1.0s until BLE supplies real range settings.
_RAW_COUNT_SCALE = 1000

PREVIEW_TEXT = "Preview · not saved"
ERROR_TEXT = "Recording stopped: database error. Restart the app."
NO_LABEL_TEXT = "No label"
START_TEXT = "Start"
STOP_TEXT = "Stop"


def format_elapsed(ms: int) -> str:
    """"3:12" for 3 min 12 s; "1:02:03" once past an hour."""
    total_s = ms // 1000
    hours, rest = divmod(total_s, 3600)
    minutes, seconds = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


IMU_AXES = [
    ("ax", "accent", (-2.0, 2.0)),
    ("ay", "accent", (-2.0, 2.0)),
    ("az", "accent", (-2.0, 2.0)),
    ("gx", "warning", (-250.0, 250.0)),
    ("gy", "warning", (-250.0, 250.0)),
    ("gz", "warning", (-250.0, 250.0)),
]


class PulseDot(QWidget):
    def __init__(self, color: str, parent: QWidget | None = None):
        super().__init__(parent)
        self._color = color
        self._on = True
        self.setFixedSize(12, 12)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._blink)
        self._timer.start(600)

    def _blink(self) -> None:
        self._on = not self._on
        self.update()

    def paintEvent(self, _e):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        c = QColor(self._color)
        c.setAlpha(255 if self._on else 90)
        p.setBrush(c)
        p.drawEllipse(1, 1, 10, 10)


class ImuRow(QWidget):
    def __init__(self, label: str, color: str, rng: tuple[float, float], theme: Theme):
        super().__init__()
        self._lo, self._hi = rng
        self._color = color
        self._track = theme.c("surface_2")
        self._value = 0.0
        self.setFixedHeight(20)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)

        self._label = QLabel(label)
        self._label.setObjectName("Mono")
        self._label.setFixedWidth(24)

        self._bar = _MiniBar(color, self._track, rng)

        self._num = QLabel("0.00")
        self._num.setObjectName("Mono")
        self._num.setFixedWidth(64)
        self._num.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        lay.addWidget(self._label)
        lay.addWidget(self._bar)
        lay.addStretch(1)
        lay.addWidget(self._num)

    def set_value(self, v: float) -> None:
        self._value = v
        self._bar.set_value(v)
        self._num.setText(f"{v:7.2f}")


class _MiniBar(QWidget):
    """60px wide, 4px tall, fill proportional to value within expected range."""

    def __init__(self, color: str, track: str, rng: tuple[float, float]):
        super().__init__()
        self._color = color
        self._track = track
        self._lo, self._hi = rng
        self._value = 0.0
        self.setFixedSize(60, 20)

    def set_value(self, v: float) -> None:
        self._value = v
        self.update()

    def paintEvent(self, _e):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        h = 4
        y = (self.height() - h) // 2
        p.setBrush(QColor(self._track))
        p.drawRoundedRect(0, y, 60, h, 2, 2)
        frac = (self._value - self._lo) / (self._hi - self._lo)
        frac = max(0.0, min(1.0, frac))
        p.setBrush(QColor(self._color))
        p.drawRoundedRect(0, y, int(60 * frac), h, 2, 2)


class LiveMonitorView(QWidget):
    def __init__(
        self,
        theme: Theme,
        controller: RecordingController | None = None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.theme = theme
        self._controller = controller

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(16)
        root.addWidget(self._control_row())

        columns = QHBoxLayout()
        columns.setSpacing(16)
        columns.addWidget(self._left_column(), 6)
        columns.addWidget(self._right_column(), 4)
        root.addLayout(columns, 1)

        # Updates "● Recording m:ss" once a second while recording.
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.timeout.connect(self._show_elapsed)

        if controller is not None:
            controller.hr_received.connect(self._show_hr)
            controller.imu_received.connect(self._show_imu)
            controller.prediction_received.connect(self._show_prediction)
            controller.state_changed.connect(self._show_state)
            # A rebuilt view (theme toggle) picks up a recording in progress.
            self._show_state(controller.state, controller.session_id, controller.started_at)
            picker_index = self.activity_picker.findData(controller.label)
            self.activity_picker.setCurrentIndex(max(picker_index, 0))
        else:
            self._show_state(None, None, None)

    def detach(self) -> None:
        """Disconnects from the controller. Called before this view is torn
        down (theme rebuild), so a deleted view never receives readings."""
        self._elapsed_timer.stop()
        if self._controller is None:
            return
        self._controller.hr_received.disconnect(self._show_hr)
        self._controller.imu_received.disconnect(self._show_imu)
        self._controller.prediction_received.disconnect(self._show_prediction)
        self._controller.state_changed.disconnect(self._show_state)
        self._controller = None

    # -- control row -------------------------------------------------------
    def _control_row(self) -> Card:
        card = Card(pad=12)
        row = QHBoxLayout()
        row.setSpacing(10)
        self.status_label = QLabel(PREVIEW_TEXT)
        row.addWidget(self.status_label)
        row.addStretch(1)
        row.addWidget(muted("Activity"))
        self.activity_picker = QComboBox()
        self.activity_picker.addItem(NO_LABEL_TEXT, None)
        for activity in CLASSES:
            self.activity_picker.addItem(activity, activity)
        row.addWidget(self.activity_picker)
        self.start_stop_button = QPushButton(START_TEXT)
        self.start_stop_button.setObjectName("AccentButton")
        self.start_stop_button.clicked.connect(self._on_start_stop_clicked)
        row.addWidget(self.start_stop_button)
        holder = QWidget()
        holder.setLayout(row)
        card.body().addWidget(holder)
        return card

    def _on_start_stop_clicked(self) -> None:
        if self._controller is None:
            return
        if self._controller.state == RECORDING:
            self._controller.stop()
        else:
            self._controller.start(self.activity_picker.currentData())

    def _show_state(self, state: str | None, _session_id, _started_at) -> None:
        recording = state == RECORDING
        can_record = self._controller is not None and self._controller.can_record
        self.start_stop_button.setText(STOP_TEXT if recording else START_TEXT)
        self.start_stop_button.setEnabled(can_record)
        self.activity_picker.setEnabled(can_record and not recording)
        if recording:
            self.status_label.setStyleSheet(
                f"color: {self.theme.c('danger')}; font-weight: 600;"
            )
            self._show_elapsed()
            self._elapsed_timer.start(1000)
        elif state == ERROR:
            self._elapsed_timer.stop()
            self.status_label.setStyleSheet(
                f"color: {self.theme.c('danger')}; font-weight: 600;"
            )
            self.status_label.setText(ERROR_TEXT)
        else:
            self._elapsed_timer.stop()
            self.status_label.setStyleSheet(f"color: {self.theme.c('text_muted')};")
            self.status_label.setText(PREVIEW_TEXT)

    def _show_elapsed(self) -> None:
        if self._controller is not None:
            elapsed = format_elapsed(self._controller.elapsed_ms())
            self.status_label.setText(f"● Recording {elapsed}")

    # -- left ------------------------------------------------------------
    def _left_column(self) -> Card:
        card = Card(pad=16)

        hr_row = QHBoxLayout()
        hr_row.setSpacing(10)
        hr_row.addWidget(PulseDot(self.theme.c("danger")), 0, Qt.AlignVCenter)
        self.hr_number = QLabel("--")
        self.hr_number.setStyleSheet(
            f"font-size: 42px; font-weight: 500; color: {self.theme.c('accent')};"
        )
        hr_row.addWidget(self.hr_number)
        hr_row.addStretch(1)
        card.body().addLayout(hr_row)

        card.body().addWidget(muted("Live heart rate · MAX30101"))

        self.chart = LiveHRChart(self.theme)
        card.body().addWidget(self.chart, 1)
        return card

    # -- right ---------------------------------------------------------
    def _right_column(self) -> QWidget:
        holder = QWidget()
        lay = QVBoxLayout(holder)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(16)

        # activity classification panel
        act = Card(pad=16)
        act.body().addWidget(heading("Activity"))
        self.activity_badge = Badge("Walking", self.theme.c("success"))
        badge_row = QHBoxLayout()
        badge_row.addWidget(self.activity_badge, 0, Qt.AlignLeft)
        badge_row.addStretch(1)
        act.body().addLayout(badge_row)
        self.confidence_label = muted("Confidence: --")
        act.body().addWidget(self.confidence_label)
        act.body().addWidget(muted("Model: 1D CNN · TFLite"))
        lay.addWidget(act)

        # IMU axes panel
        imu = Card(pad=16)
        imu.body().addWidget(heading("IMU axes · MPU-9250"))
        self.imu_rows: list[ImuRow] = []
        for label, color_key, rng in IMU_AXES:
            row = ImuRow(label, self.theme.c(color_key), rng, self.theme)
            self.imu_rows.append(row)
            imu.body().addWidget(row)
        imu.body().addWidget(muted("Debug view · live per BLE packet"))
        lay.addWidget(imu)
        lay.addStretch(1)
        return holder

    # -- readings from the controller (preview and recording alike) -------
    def _show_hr(self, readings: list[HrReading]) -> None:
        for r in readings:
            self.chart.push(r.bpm)
        self.hr_number.setText(f"{readings[-1].bpm:.0f}")

    def _show_imu(self, readings: list[ImuReading]) -> None:
        r = readings[-1]
        for row, raw in zip(self.imu_rows, (r.ax, r.ay, r.az, r.gx, r.gy, r.gz)):
            row.set_value(raw / _RAW_COUNT_SCALE)

    def _show_prediction(self, r: PredictionReading) -> None:
        self.activity_badge.setText(r.label)
        self.activity_badge.set_color(self.theme.activity_color(r.label))
        self.confidence_label.setText(f"Confidence: {r.confidence:.0%}")

    def restyle(self, theme: Theme) -> None:
        self.theme = theme
        self.chart.restyle(theme)
