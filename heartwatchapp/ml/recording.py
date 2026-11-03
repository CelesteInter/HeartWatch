"""One recording of motion data, in real units, ready for preprocessing.

Plain-language overview: the model is trained on recordings that come from
different places -- today the two text logs the watch printed over USB
(HeartWatch/test/Standing.txt and Walking.txt), later sessions saved in the
app's database. This file turns each kind into the same thing, a
`Recording`: timestamps, six motion channels in g and deg/s, an activity
label, and a `meta` dict saying exactly where the data came from.

Separate on purpose from sources/base.py: those readings are raw integer
counts stamped in host time, for the live app. A Recording holds real
units, and its timestamps may be the device's own uptime (legacy captures
have nothing else).

No PySide6 anywhere in this file or what it imports (the training
environment doesn't have it); tests/test_ml_pipeline.py checks.
"""

from __future__ import annotations

import collections
import hashlib
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import spec
from .inference import CLASSES

HARDWARE_NOTE = (
    "IMU range registers and PWR_MGMT_1 never written by firmware 1eabdb8; "
    "sensor defaults assumed; chip variant unconfirmed."
)

# One IMU line of the ESP-IDF log, timestamp = device uptime in ms.
LEGACY_IMU_RE = re.compile(
    r"^I \((\d+)\) mpu: AX: (-?\d+), AY: (-?\d+), AZ: (-?\d+), "
    r"GX: (-?\d+), GY: (-?\d+), GZ: (-?\d+)$"
)
_HR_RE = re.compile(r"^E \(\d+\) heartrate: ")
_ESP_RE = re.compile(r"^I \(\d+\) esp: ?$")
_MPU_PREFIX_RE = re.compile(r"^I \(\d+\) mpu: ")

# Sessions whose stored scales are known placeholders: these sources write
# value x 1000 as the "count" but store accel_scale = gyro_scale = 1.0
# (sources/dummy.py, data/seed.py, data/writer.py, scripts/demo_db.py), so
# count x scale would be 1000x too big. They're synthetic anyway.
UNTRUSTED_DEVICE_IDS = frozenset({"seed", "demo", "demo-session"})
PLACEHOLDER_SCALES = (1.0, 1.0)

# The app's real database. from_db_session() refuses it outright -- before
# opening anything -- so a mistake can't touch real data.
REAL_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "heartwatch.db"


@dataclass
class Recording:
    ts_ms: np.ndarray          # int64 [N], non-decreasing (relative or absolute ms)
    imu: np.ndarray            # float64 [N, 6], ax ay az in g, gx gy gz in deg/s
    label: str | None          # one of CLASSES, or None
    source: str                # "legacy-capture", "db-session", ...
    meta: dict = field(default_factory=dict)
    # Set by preprocess.resample(): True where a 100 ms bin had no samples
    # (that row of `imu` is NaN). None for raw recordings.
    gap: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.ts_ms)


def _check_label(label: str | None) -> str | None:
    if label is not None and label not in CLASSES:
        raise ValueError(f"label {label!r} is not one of {CLASSES}")
    return label


def counts_to_units(counts: np.ndarray, accel_scale: float, gyro_scale: float) -> np.ndarray:
    """Raw int counts [N, 6] -> float64 g and deg/s. The ONE conversion used
    by both readers, so a capture and the same data read back from a
    database give bit-identical values."""
    scales = np.array([accel_scale] * 3 + [gyro_scale] * 3, dtype=np.float64)
    return counts.astype(np.float64) * scales


def read_legacy_counts(path: str | Path) -> tuple[np.ndarray, np.ndarray, dict, str]:
    """Parse an ESP-IDF capture log: (ts_ms int64 [N], counts int64 [N, 6],
    line_counts, sha256). Keeps only lines matching LEGACY_IMU_RE; every
    other line is counted by kind: "hr", "esp", "blank", "spliced" (an mpu
    line cut short by another line, e.g. line 1 of both captures), "other"."""
    raw = Path(path).read_bytes()
    sha256 = hashlib.sha256(raw).hexdigest()
    kinds = collections.Counter()
    ts, counts = [], []
    for line in raw.decode("utf-8").splitlines():
        m = LEGACY_IMU_RE.match(line)
        if m:
            kinds["imu"] += 1
            ts.append(int(m[1]))
            counts.append([int(v) for v in m.groups()[1:]])
        elif not line.strip():
            kinds["blank"] += 1
        elif _ESP_RE.match(line):
            kinds["esp"] += 1
        elif _HR_RE.match(line):
            kinds["hr"] += 1
        elif _MPU_PREFIX_RE.match(line):
            kinds["spliced"] += 1
        else:
            kinds["other"] += 1
    line_counts = {k: kinds.get(k, 0) for k in ("imu", "hr", "esp", "blank", "spliced", "other")}
    ts_arr = np.array(ts, dtype=np.int64)
    counts_arr = np.array(counts, dtype=np.int64).reshape(-1, 6)
    if len(ts_arr) > 1 and np.any(np.diff(ts_arr) < 0):
        raise ValueError(f"{path}: IMU timestamps go backwards")
    return ts_arr, counts_arr, line_counts, sha256


def from_legacy_capture(path: str | Path, label: str | None = None) -> Recording:
    """A Recording from one capture log. `label` defaults to the file name
    (Standing.txt -> "Standing") when that is one of CLASSES, else None.
    A label (given or from the name) outside CLASSES is rejected."""
    path = Path(path)
    if label is None and path.stem in CLASSES:
        label = path.stem
    _check_label(label)
    ts, counts, line_counts, sha256 = read_legacy_counts(path)
    imu = counts_to_units(counts, spec.LEGACY_ACCEL_SCALE, spec.LEGACY_GYRO_SCALE)
    return Recording(
        ts_ms=ts,
        imu=imu,
        label=label,
        source="legacy-capture",
        meta={
            "path": str(path),
            "sha256": sha256,
            "accel_scale": spec.LEGACY_ACCEL_SCALE,
            "gyro_scale": spec.LEGACY_GYRO_SCALE,
            "timestamps": "device uptime ms",
            "line_counts": line_counts,
            "hardware_note": HARDWARE_NOTE,
        },
    )


def from_db_session(path: str | Path, session_id: int) -> Recording:
    """A Recording from one session of a database at `path` (required --
    never the app's real database, which is refused before anything is
    opened). Uses the session's stored scale factors, so sessions whose
    scales are known placeholders are refused: device_id seed / demo /
    demo-session, or scales exactly (1.0, 1.0)."""
    if path is None:
        raise ValueError("from_db_session needs an explicit database path")
    path = Path(path).expanduser().resolve()
    if path == REAL_DB_PATH.resolve():
        raise ValueError(
            f"refusing to read the app's real database ({path}); "
            "use a copy or a temp database"
        )
    if not path.is_file():
        raise FileNotFoundError(path)
    conn = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    try:
        row = conn.execute(
            "SELECT device_id, label, accel_scale, gyro_scale FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"no session {session_id} in {path}")
        device_id, label, accel_scale, gyro_scale = row
        if device_id in UNTRUSTED_DEVICE_IDS:
            raise ValueError(
                f"session {session_id} is from device_id {device_id!r}, whose stored "
                f"scales are placeholders (1.0, 1.0) while its counts are value x 1000; "
                "it can't be converted to real units"
            )
        if (accel_scale, gyro_scale) == PLACEHOLDER_SCALES:
            raise ValueError(
                f"session {session_id} has placeholder scales (1.0, 1.0); "
                "refusing to guess its units"
            )
        _check_label(label)
        rows = conn.execute(
            "SELECT ts, ax, ay, az, gx, gy, gz FROM imu_samples "
            "WHERE session_id = ? ORDER BY ts, rowid",
            (session_id,),
        ).fetchall()
    finally:
        conn.close()
    data = np.array(rows, dtype=np.int64).reshape(-1, 7)
    return Recording(
        ts_ms=data[:, 0].copy(),
        imu=counts_to_units(data[:, 1:], accel_scale, gyro_scale),
        label=label,
        source="db-session",
        meta={
            "path": str(path),
            "session_id": session_id,
            "device_id": device_id,
            "accel_scale": accel_scale,
            "gyro_scale": gyro_scale,
            "timestamps": "host epoch ms",
            "hardware_note": HARDWARE_NOTE,
        },
    )
