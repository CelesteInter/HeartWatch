"""Synthetic sample generation for demo/seed sessions.

Pure functions, no database access -- callers (DatabaseWriter.seed_demo_session,
LiveMonitorView's tick timers) decide how the rows reach the database. Kept
separate from db.py/writer.py so the "what does fake sensor data look like"
logic isn't tangled with SQL.
"""

from __future__ import annotations

import math
import random

ACTIVITY_LABELS = ["Sitting", "Walking", "Standing", "Running"]

# Same shape as Live Monitor's IMU_AXES ranges (ui/live_monitor.py), so seeded
# history and live demo data look like they came from the same source.
_ACCEL_RANGE = (-2.0, 2.0)
_GYRO_RANGE = (-250.0, 250.0)
_RAW_COUNT_SCALE = 1000  # spreads the -2..2 / -250..250 walk into non-degenerate ints


# Dummy activity classifier (until step 6's real TFLite model). One
# prediction per 5-second window, like the real classifier will emit.
# Each window has a small chance of switching activity so sessions get a
# realistic mix; confidence is random, so now and then a window falls
# below data/stats.py's LOW_CONFIDENCE_THRESHOLD (0.6). Used by both the
# Live Monitor's live demo session and DatabaseWriter.seed_demo_session;
# rows are tagged DUMMY_MODEL_VERSION so they're easy to find and delete.
# Remove along with the other dummy generators in milestone 6.
PREDICTION_WINDOW_MS = 5000
ACTIVITY_SWITCH_CHANCE = 0.15
DUMMY_CONFIDENCE_RANGE = (0.55, 0.98)
DUMMY_MODEL_VERSION = "demo-dummy"


def random_label() -> str:
    return random.choice(ACTIVITY_LABELS)


def next_dummy_prediction(current_label: str) -> tuple[str, float]:
    """One dummy classifier window: (label, confidence). Usually keeps
    `current_label`; with ACTIVITY_SWITCH_CHANCE switches to a different
    activity."""
    label = current_label
    if random.random() < ACTIVITY_SWITCH_CHANCE:
        label = random.choice([a for a in ACTIVITY_LABELS if a != current_label])
    return label, round(random.uniform(*DUMMY_CONFIDENCE_RANGE), 2)


def generate_prediction_rows(
    started_at_ms: int, duration_s: float, first_label: str | None = None
) -> list[tuple[int, str, float]]:
    """[(ts, predicted, confidence), ...] -- one dummy prediction per
    PREDICTION_WINDOW_MS window over the session, `ts` being each window's
    END time (the predictions table's convention). Same generator the Live
    Monitor uses, just run over a whole session at once."""
    label = first_label or random_label()
    rows = []
    ts = started_at_ms + PREDICTION_WINDOW_MS
    end_ms = started_at_ms + int(duration_s * 1000)
    while ts <= end_ms:
        label, confidence = next_dummy_prediction(label)
        rows.append((ts, label, confidence))
        ts += PREDICTION_WINDOW_MS
    return rows


def generate_imu_rows(
    started_at_ms: int, duration_s: float, hz: float = 10.0
) -> list[tuple[int, int, int, int, int, int, int]]:
    """[(ts, ax, ay, az, gx, gy, gz), ...] -- random-walk, same shape as the
    Live Monitor's dummy generator (ui/live_monitor.py)."""
    n = int(duration_s * hz)
    ranges = (_ACCEL_RANGE, _ACCEL_RANGE, _ACCEL_RANGE, _GYRO_RANGE, _GYRO_RANGE, _GYRO_RANGE)
    state = [0.0] * 6
    rows = []
    for i in range(n):
        ts = started_at_ms + int(i * 1000 / hz)
        for j, (lo, hi) in enumerate(ranges):
            span = hi - lo
            state[j] = max(lo, min(hi, state[j] + random.uniform(-0.04, 0.04) * span))
        ax, ay, az, gx, gy, gz = (int(round(v * _RAW_COUNT_SCALE)) for v in state)
        rows.append((ts, ax, ay, az, gx, gy, gz))
    return rows


def generate_hr_rows(
    started_at_ms: int, duration_s: float, hz: float = 1.0
) -> list[tuple[int, int, float]]:
    """[(ts, bpm, confidence), ...] -- same sine-wave shape as the Live Monitor's
    dummy HR generator."""
    n = int(duration_s * hz)
    rows = []
    for i in range(n):
        ts = started_at_ms + int(i * 1000 / hz)
        bpm = 75 + 12 * math.sin(i / 6.0) + random.uniform(-2.5, 2.5)
        confidence = round(random.uniform(0.85, 0.99), 2)
        rows.append((ts, int(round(bpm)), confidence))
    return rows
