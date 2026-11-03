"""DummySource -- fake live readings until the real watch is ready.

Plain-language overview: the same fake data the Live Monitor has always
shown, moved out of the view so the real watch can later plug into the
same slot (sources/base.py). Heart rate is a sine wave with noise, once a
second; motion is a random walk, ten times a second; the activity is the
dummy classifier from data/seed.py, one prediction per 5-second window,
tagged model_version "demo-dummy". Same formulas and rates as before.
Remove in milestone 6 along with the other dummy generators.

Timing: readings are produced on a fixed schedule and stamped with their
scheduled time (host epoch ms), not with whenever a timer happened to
fire. poll() emits everything that is due by the clock's current time;
the built-in timer just calls poll() every POLL_MS. Tests turn the timer
off, inject a clock and an rng, and call poll() themselves, so they run
instantly and give the same readings every time. align_windows() restarts
the prediction schedule, so when a recording starts at t its first window
ends exactly at t + 5 s.
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable

from PySide6.QtCore import Qt, QTimer

from ..data import seed
from .base import DataSource, HrReading, ImuReading, PredictionReading, ReadingSink

HR_PERIOD_MS = 1000
IMU_PERIOD_MS = 100
POLL_MS = 100
# After a stall longer than this (the laptop slept, the app was busy), the
# schedule skips ahead instead of emitting a burst of catch-up readings.
MAX_BACKLOG_MS = 2000

# Expected ranges in g (accelerometer) and deg/s (gyroscope), same as
# data/seed.py's and the Live Monitor's IMU bars.
_ACCEL_RANGE = (-2.0, 2.0)
_GYRO_RANGE = (-250.0, 250.0)
IMU_RANGES = (_ACCEL_RANGE,) * 3 + (_GYRO_RANGE,) * 3
# Raw stored count = value in g or deg/s x this (see the Live Monitor).
RAW_COUNT_SCALE = 1000
FIRST_ACTIVITY = "Walking"


def _now_ms() -> int:
    return int(time.time() * 1000)


def _due(next_ms: int, period_ms: int, now_ms: int) -> tuple[list[int], int]:
    """The scheduled times up to `now_ms`, starting at `next_ms` and every
    `period_ms`, and the next one after them. Skips (on the same grid) any
    backlog older than MAX_BACKLOG_MS."""
    if now_ms - next_ms > MAX_BACKLOG_MS:
        next_ms += (now_ms - next_ms) // period_ms * period_ms
    times = []
    while next_ms <= now_ms:
        times.append(next_ms)
        next_ms += period_ms
    return times, next_ms


class DummySource(DataSource):
    def __init__(
        self,
        clock: Callable[[], int] | None = None,
        rng: random.Random | None = None,
        use_timer: bool = True,
    ):
        """`clock`: returns host epoch ms (default: the real clock).
        `rng`: e.g. a seeded random.Random (default: the global `random`,
        as before). `use_timer=False`: nothing happens until poll() is
        called."""
        self._clock = clock or _now_ms
        self._rng = rng if rng is not None else random
        self._use_timer = use_timer
        self._timer: QTimer | None = None
        self._sink: ReadingSink | None = None
        self._t = 0.0
        self._imu_state = [0.0] * 6
        self._activity = FIRST_ACTIVITY
        self._next_hr = self._next_imu = self._next_window_end = 0

    @property
    def device_id(self) -> str:
        return "demo"

    @property
    def accel_scale(self) -> float:
        return 1.0

    @property
    def gyro_scale(self) -> float:
        return 1.0

    def start(self, sink: ReadingSink) -> None:
        if self._sink is not None:
            return
        self._sink = sink
        now = self._clock()
        # As before: a heart-rate reading straight away, the first motion
        # reading 100 ms later, the first prediction at the end of the
        # first 5-second window.
        self._next_hr = now
        self._next_imu = now + IMU_PERIOD_MS
        self._next_window_end = now + seed.PREDICTION_WINDOW_MS
        self.poll()
        if self._use_timer:
            self._timer = QTimer()
            self._timer.setTimerType(Qt.PreciseTimer)
            self._timer.timeout.connect(self.poll)
            self._timer.start(POLL_MS)

    def stop(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        self._sink = None

    def align_windows(self, t_ms: int) -> None:
        self._next_window_end = t_ms + seed.PREDICTION_WINDOW_MS

    def poll(self) -> None:
        """Emit every reading that is due by now. Called by the timer, or
        by tests with use_timer=False."""
        if self._sink is None:
            return
        now = self._clock()
        hr_times, self._next_hr = _due(self._next_hr, HR_PERIOD_MS, now)
        imu_times, self._next_imu = _due(self._next_imu, IMU_PERIOD_MS, now)
        window_ends, self._next_window_end = _due(
            self._next_window_end, seed.PREDICTION_WINDOW_MS, now
        )
        if hr_times:
            self._sink.on_hr([self._hr_reading(ts) for ts in hr_times])
        if imu_times:
            self._sink.on_imu([self._imu_reading(ts) for ts in imu_times])
        for ts in window_ends:
            self._activity, confidence = seed.next_dummy_prediction(self._activity, self._rng)
            self._sink.on_prediction(
                PredictionReading(ts, self._activity, confidence, seed.DUMMY_MODEL_VERSION)
            )

    def _hr_reading(self, ts: int) -> HrReading:
        self._t += 1.0
        bpm = 75 + 12 * math.sin(self._t / 6.0) + self._rng.uniform(-2.5, 2.5)
        return HrReading(ts, int(round(bpm)), None)

    def _imu_reading(self, ts: int) -> ImuReading:
        for i, (lo, hi) in enumerate(IMU_RANGES):
            step = self._rng.uniform(-0.04, 0.04) * (hi - lo)
            self._imu_state[i] = max(lo, min(hi, self._imu_state[i] + step))
        return ImuReading(ts, *(int(round(v * RAW_COUNT_SCALE)) for v in self._imu_state))
