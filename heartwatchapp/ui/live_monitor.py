"""Section 2 — Live Monitor. Two-column: HR chart (left ~60%) · stacked panels (right).

The HR/IMU values are still a dummy generator (sine-wave HR, random-walk IMU) --
step 6 swaps the generator for the real BLE stream from the ESP32-S3. What's real
now is the write path: every generated sample is inserted through
DatabaseWriter.insert_imu_batch / insert_hr_batch against a session opened with
start_session(device_id="demo", ...), exactly like BLE data will be. That makes
step 6 a source swap and nothing else.
"""

from __future__ import annotations

import math
import random
import time

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from ..data.writer import DatabaseWriter
from .charts import LiveHRChart
from .theme import Theme
from .widgets import Badge, Card, heading, muted

# Placeholder scale factors for the demo session (per the SQLite handoff, §6) --
# raw stored counts equal the demo's float units 1:1 until BLE supplies real
# accel/gyro range settings. Multiplied in only when writing to the DB so the
# on-screen numbers (which read naturally as g / deg-per-second) don't change.
_RAW_COUNT_SCALE = 1000

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
        writer: DatabaseWriter | None = None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.theme = theme
        self._t = 0.0
        self._imu_state = [0.0] * 6

        self._writer = writer
        self._session_id: int | None = None
        if self._writer is not None:
            self._session_id = self._writer.start_session(
                device_id="demo", label=None, accel_scale=1.0, gyro_scale=1.0
            )

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(16)

        root.addWidget(self._left_column(), 6)
        root.addWidget(self._right_column(), 4)

        # -- dummy data timers --
        self._hr_timer = QTimer(self)
        self._hr_timer.timeout.connect(self._tick_hr)
        self._hr_timer.start(1000)

        self._imu_timer = QTimer(self)
        self._imu_timer.timeout.connect(self._tick_imu)
        self._imu_timer.start(100)

        self._tick_hr()

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
        act.body().addWidget(muted("Confidence: 94%"))
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

    # -- dummy generators, routed through the real insert path -----------
    def _tick_hr(self) -> None:
        self._t += 1.0
        bpm = 75 + 12 * math.sin(self._t / 6.0) + random.uniform(-2.5, 2.5)
        self.hr_number.setText(f"{bpm:.0f}")
        self.chart.push(bpm)

        if self._writer is not None and self._session_id is not None:
            ts = int(time.time() * 1000)
            self._writer.insert_hr_batch(self._session_id, [(ts, int(round(bpm)), None)])

    def _tick_imu(self) -> None:
        for i, (_, _, rng) in enumerate(IMU_AXES):
            span = rng[1] - rng[0]
            step = random.uniform(-0.04, 0.04) * span
            self._imu_state[i] = max(rng[0], min(rng[1], self._imu_state[i] + step))
            self.imu_rows[i].set_value(self._imu_state[i])

        if self._writer is not None and self._session_id is not None:
            ts = int(time.time() * 1000)
            raw = tuple(int(round(v * _RAW_COUNT_SCALE)) for v in self._imu_state)
            self._writer.insert_imu_batch(self._session_id, [(ts, *raw)])

    def shutdown(self) -> None:
        """Stops the generator timers and closes out the demo session.
        Must be called before this view is torn down (theme rebuild, app
        close) so `ended_at` gets stamped instead of staying NULL forever."""
        self._hr_timer.stop()
        self._imu_timer.stop()
        if self._writer is not None and self._session_id is not None:
            self._writer.end_session(self._session_id)
            self._session_id = None

    def restyle(self, theme: Theme) -> None:
        self.theme = theme
        self.chart.restyle(theme)
