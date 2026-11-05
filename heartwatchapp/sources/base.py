"""What a data source is: something that produces live readings.

Plain-language overview: a *source* is where the Live Monitor's numbers
come from -- today DummySource (sources/dummy.py, fake data), later the
real watch over Bluetooth (a BleSource) or a USB serial cable (a
SerialSource). A source only produces readings: heart rate, motion (IMU)
and activity predictions. It knows nothing about sessions, recording or
the database; recording/controller.py decides what is saved.

Thread-agnostic rule: this file must not import asyncio or PySide6
(tests/test_sources.py checks). A source may call its sink from ANY
thread -- a Bluetooth library may deliver data on its own event loop, a
serial reader on its own thread. Getting readings onto the Qt main thread
is the recording controller's job, so that choice stays inside each
source.

Timestamps: every `ts_ms` is the HOST computer's clock, in epoch
milliseconds, like every timestamp in the database. A source that
receives the device's own timestamps (serial or BLE) converts them to host
time itself, e.g. with an offset measured at connect; the controller and
the database never see device time.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class HrReading:
    ts_ms: int                       # host epoch ms
    bpm: int
    confidence: float | None = None  # sensor signal quality 0-1, if known


@dataclass(frozen=True)
class ImuReading:
    """One motion sample, as raw int16 counts (what imu_samples stores).
    Multiply by the source's accel_scale / gyro_scale to get g and deg/s."""

    ts_ms: int  # host epoch ms
    ax: int
    ay: int
    az: int
    gx: int
    gy: int
    gz: int


@dataclass(frozen=True)
class PredictionReading:
    """One activity-classifier window. `ts_ms` is the window's END time,
    the predictions table's convention (data/stats.py)."""

    ts_ms: int  # host epoch ms
    label: str
    confidence: float
    model_version: str


class ReadingSink(Protocol):
    """Where a source delivers readings. May be called from any thread."""

    def on_hr(self, readings: list[HrReading]) -> None: ...

    def on_imu(self, readings: list[ImuReading]) -> None: ...

    def on_prediction(self, reading: PredictionReading) -> None: ...


class DataSource(ABC):
    @property
    @abstractmethod
    def device_id(self) -> str:
        """Stored in sessions.device_id ("demo" for DummySource)."""

    @property
    @abstractmethod
    def accel_scale(self) -> float:
        """Stored in sessions.accel_scale: raw count x this = g."""

    @property
    @abstractmethod
    def gyro_scale(self) -> float:
        """Stored in sessions.gyro_scale: raw count x this = deg/s."""

    @abstractmethod
    def start(self, sink: ReadingSink) -> None:
        """Start producing readings into `sink`."""

    @abstractmethod
    def stop(self) -> None:
        """Stop producing readings. Safe to call more than once."""

    def align_windows(self, t_ms: int) -> None:
        """A recording starts at `t_ms`: a source that decides its own
        prediction windows may restart them so the first window ends at
        t_ms + the window length. Optional; does nothing by default (a
        real watch's windows are timed by the watch)."""
