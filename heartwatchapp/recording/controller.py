"""RecordingController -- whether live readings are only shown, or saved too.

Plain-language overview: the data source (sources/) always runs, so the
Live Monitor always shows live readings. The controller is in one of two
states:

- PREVIEW: readings are shown on screen ("Preview · not saved") and never
  written anywhere. The app starts in this state, so launching it creates
  no session.
- RECORDING: after Start, readings are also saved into a new session.
  Stop (or quitting the app) ends the session.

Every reading on screen comes through here in both states, so preview and
recording look the same apart from the status badge.

What gets saved while recording (started_at is the moment Start was
pressed, from the controller's clock):
- heart-rate and motion readings with ts >= started_at;
- predictions whose whole window is inside the recording:
  ts - PREDICTION_WINDOW_MS >= started_at. (A source that times its own
  windows is asked to restart them at Start -- DataSource.align_windows()
  -- so the dummy's first saved window ends at started_at + 5 s.)
On Stop, ended_at is the controller's clock, or the newest saved reading
if that is later, so every saved row lies inside [started_at, ended_at].

Threads: a source may call on_hr / on_imu / on_prediction from any
thread. They re-emit through Qt signals with the default (auto)
connection, which hands the readings to the Qt main thread when they come
from another thread and calls straight through when they don't.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from PySide6.QtCore import QObject, Signal

from ..data.stats import PREDICTION_WINDOW_MS
from ..data.writer import DatabaseWriter
from ..sources.base import DataSource, HrReading, ImuReading, PredictionReading

PREVIEW = "preview"
RECORDING = "recording"

logger = logging.getLogger("heartwatch.recording")


def _now_ms() -> int:
    return int(time.time() * 1000)


class RecordingController(QObject):
    hr_received = Signal(object)          # list[HrReading]
    imu_received = Signal(object)         # list[ImuReading]
    prediction_received = Signal(object)  # PredictionReading
    # (state, session id or None, started_at ms or None)
    state_changed = Signal(str, object, object)
    recording_finished = Signal(int)      # session id, after it has ended

    # Readings from the source, re-emitted so they reach the main thread.
    _hr_in = Signal(object)
    _imu_in = Signal(object)
    _prediction_in = Signal(object)

    def __init__(
        self,
        source: DataSource,
        writer: DatabaseWriter | None,
        clock: Callable[[], int] | None = None,
        parent: QObject | None = None,
    ):
        """`writer` None: preview only, Start does nothing. `clock`:
        returns host epoch ms (default: the real clock); tests share one
        fake clock between the source and the controller."""
        super().__init__(parent)
        self._source = source
        self._writer = writer
        self._clock = clock or _now_ms
        self._running = False
        self._state = PREVIEW
        self._session_id: int | None = None
        self._started_at: int | None = None
        self._label: str | None = None
        self._last_saved_ts: int | None = None

        self._hr_in.connect(self._handle_hr)
        self._imu_in.connect(self._handle_imu)
        self._prediction_in.connect(self._handle_prediction)

    # -- state -------------------------------------------------------------

    @property
    def state(self) -> str:
        return self._state

    @property
    def session_id(self) -> int | None:
        return self._session_id

    @property
    def started_at(self) -> int | None:
        return self._started_at

    @property
    def label(self) -> str | None:
        """The activity label of the current recording (None: no label)."""
        return self._label

    @property
    def can_record(self) -> bool:
        return self._writer is not None

    def elapsed_ms(self) -> int:
        """How long the current recording has run; 0 in preview."""
        if self._state != RECORDING:
            return 0
        return max(0, self._clock() - self._started_at)

    # -- lifecycle -----------------------------------------------------------

    def start_preview(self) -> None:
        """Starts the source. Called once, when the main window is built."""
        if not self._running:
            self._running = True
            self._source.start(self)

    def shutdown(self) -> None:
        """App quit: ends a recording in progress (so it is saved), then
        stops the source. Must run before the writer stops."""
        self.stop()
        if self._running:
            self._running = False
            self._source.stop()

    def start(self, activity_label: str | None = None) -> int | None:
        """Opens a session and starts saving. Returns its id, or None if
        already recording (ignored) or there's no writer."""
        if self._state == RECORDING:
            logger.info("Start ignored: already recording session #%d", self._session_id)
            return None
        if self._writer is None:
            logger.warning("Start ignored: no database writer")
            return None
        started_at = self._clock()
        self._source.align_windows(started_at)
        session_id = self._writer.start_session(
            self._source.device_id,
            activity_label,
            self._source.accel_scale,
            self._source.gyro_scale,
            started_at=started_at,
        )
        self._state = RECORDING
        self._session_id = session_id
        self._started_at = started_at
        self._label = activity_label
        self._last_saved_ts = None
        logger.info("Recording session #%d (label %r)", session_id, activity_label)
        self.state_changed.emit(RECORDING, session_id, started_at)
        return session_id

    def stop(self) -> int | None:
        """Ends the recording: flushes pending rows, stamps ended_at.
        Returns the session id, or None if there was nothing to stop."""
        if self._state != RECORDING:
            return None
        session_id = self._session_id
        ended_at = self._clock()
        if self._last_saved_ts is not None and self._last_saved_ts > ended_at:
            ended_at = self._last_saved_ts
        self._writer.end_session(session_id, ended_at=ended_at)
        self._state = PREVIEW
        self._session_id = None
        self._started_at = None
        self._label = None
        self._last_saved_ts = None
        logger.info("Recording session #%d saved", session_id)
        self.state_changed.emit(PREVIEW, None, None)
        self.recording_finished.emit(session_id)
        return session_id

    # -- sink (called by the source, from any thread) --------------------------

    def on_hr(self, readings: list[HrReading]) -> None:
        self._hr_in.emit(list(readings))

    def on_imu(self, readings: list[ImuReading]) -> None:
        self._imu_in.emit(list(readings))

    def on_prediction(self, reading: PredictionReading) -> None:
        self._prediction_in.emit(reading)

    # -- main thread -------------------------------------------------------------

    def _handle_hr(self, readings: list[HrReading]) -> None:
        self.hr_received.emit(readings)
        rows = [(r.ts_ms, r.bpm, r.confidence) for r in readings if self._saves(r.ts_ms)]
        if rows:
            self._writer.insert_hr_batch(self._session_id, rows)
            self._saved(rows)

    def _handle_imu(self, readings: list[ImuReading]) -> None:
        self.imu_received.emit(readings)
        rows = [
            (r.ts_ms, r.ax, r.ay, r.az, r.gx, r.gy, r.gz)
            for r in readings
            if self._saves(r.ts_ms)
        ]
        if rows:
            self._writer.insert_imu_batch(self._session_id, rows)
            self._saved(rows)

    def _handle_prediction(self, r: PredictionReading) -> None:
        self.prediction_received.emit(r)
        if self._saves(r.ts_ms - PREDICTION_WINDOW_MS):
            self._writer.insert_prediction(
                self._session_id, r.ts_ms, r.label, r.confidence, r.model_version
            )
            self._saved([(r.ts_ms,)])

    def _saves(self, ts_ms: int) -> bool:
        return self._state == RECORDING and ts_ms >= self._started_at

    def _saved(self, rows: list[tuple]) -> None:
        newest = max(row[0] for row in rows)
        if self._last_saved_ts is None or newest > self._last_saved_ts:
            self._last_saved_ts = newest
