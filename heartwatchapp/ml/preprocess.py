"""Turning a Recording into model input windows. Shared by training and the app.

Plain-language overview, in the order prepare() runs the steps:

1. resample: average the raw readings in each 100 ms slot into one sample,
   giving 10 samples per second whatever rate the watch sent. A slot with
   no readings is marked as a gap (never filled in).
2. trim: drop the first and last 5 seconds (the start/stop fiddling).
3. windows: cut 5-second pieces (50 samples). A piece that touches a gap
   is dropped and counted. Each piece is stamped with its END time, like
   the predictions table.
4. normalize: divide accel by 4 g and gyro by 500 deg/s (fixed numbers
   from ml/spec.py, stored in the model's metadata).

prepare() is the ONLY entry point training and the app use, so both see
exactly the same numbers. No PySide6 here (tests/test_ml_pipeline.py).
"""

from __future__ import annotations

import dataclasses
from typing import NamedTuple

import numpy as np

from . import spec
from .recording import Recording


class Windows(NamedTuple):
    X: np.ndarray            # [M, WINDOW_SAMPLES, 6]
    end_ts: np.ndarray       # int64 [M]: each window's end (ms, same clock as the input)
    start_index: np.ndarray  # int64 [M]: first sample's index in the trimmed recording
    dropped: int             # windows skipped because they contained a gap
    n_samples: int           # length of the trimmed, resampled recording


def resample(rec: Recording) -> Recording:
    """Bins [t0 + k*100, t0 + (k+1)*100) ms, half-open, t0 = the first
    timestamp. Output sample k = mean of the raw samples in bin k, stamped
    at the bin's END (t0 + (k+1)*100). Empty bins: imu row NaN, gap True.
    Input already at 10 Hz on a 100 ms grid comes out with the same values
    (a mean of one sample), each stamped 100 ms later (its bin's end)."""
    if rec.gap is not None:
        raise ValueError("recording is already resampled")
    if len(rec) == 0:
        raise ValueError("empty recording")
    ts = np.asarray(rec.ts_ms, dtype=np.int64)
    k = (ts - ts[0]) // spec.BIN_MS
    n_bins = int(k[-1]) + 1
    per_bin = np.bincount(k, minlength=n_bins)
    sums = np.stack(
        [np.bincount(k, weights=rec.imu[:, c], minlength=n_bins) for c in range(rec.imu.shape[1])],
        axis=1,
    )
    gap = per_bin == 0
    imu = np.full_like(sums, np.nan)
    imu[~gap] = sums[~gap] / per_bin[~gap, None]
    out_ts = ts[0] + (np.arange(n_bins, dtype=np.int64) + 1) * spec.BIN_MS
    meta = dict(rec.meta, rate_hz=spec.RATE_HZ, bins=n_bins, gap_bins=int(gap.sum()))
    return dataclasses.replace(rec, ts_ms=out_ts, imu=imu, gap=gap, meta=meta)


def trim(rec: Recording, seconds: float) -> Recording:
    """Drop `seconds` from each end of a resampled recording (seconds x
    RATE_HZ samples at each end). Too short -> an empty recording."""
    if rec.gap is None:
        raise ValueError("trim() needs a resampled recording (run resample() first)")
    n = int(round(seconds * spec.RATE_HZ))
    keep = slice(n, max(n, len(rec) - n))
    return dataclasses.replace(
        rec, ts_ms=rec.ts_ms[keep], imu=rec.imu[keep], gap=rec.gap[keep],
        meta=dict(rec.meta, trim_s=seconds),
    )


def windows(rec: Recording, hop_samples: int) -> Windows:
    """Windows of WINDOW_SAMPLES starting at 0, hop, 2*hop, ... that fit
    entirely. Windows containing a gap are dropped and counted."""
    if rec.gap is None:
        raise ValueError("windows() needs a resampled recording")
    if hop_samples < 1:
        raise ValueError("hop_samples must be >= 1")
    w = spec.WINDOW_SAMPLES
    starts = np.arange(0, len(rec) - w + 1, hop_samples, dtype=np.int64)
    keep = np.array([not rec.gap[s:s + w].any() for s in starts], dtype=bool)
    starts_kept = starts[keep] if len(starts) else starts
    X = np.stack([rec.imu[s:s + w] for s in starts_kept]) if len(starts_kept) else (
        np.empty((0, w, rec.imu.shape[1]), dtype=rec.imu.dtype)
    )
    end_ts = rec.ts_ms[starts_kept + w - 1] if len(starts_kept) else np.empty(0, np.int64)
    return Windows(X, end_ts.astype(np.int64), starts_kept, int((~keep).sum()), len(rec))


def normalize(X: np.ndarray) -> np.ndarray:
    """Accel / ACCEL_NORM_G, gyro / GYRO_NORM_DPS, as float32 (the model's
    input type). Fixed constants -- nothing is learned from the data."""
    norm = np.array([spec.ACCEL_NORM_G] * 3 + [spec.GYRO_NORM_DPS] * 3, dtype=np.float64)
    return (np.asarray(X, dtype=np.float64) / norm).astype(np.float32)


def prepare(rec: Recording, hop_samples: int) -> Windows:
    """resample -> trim(TRIM_S) -> windows -> normalize. The one entry point."""
    w = windows(trim(resample(rec), spec.TRIM_S), hop_samples)
    return w._replace(X=normalize(w.X))
