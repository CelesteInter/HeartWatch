"""Shared by scripts/demo_session.py and scripts/eval_summaries.py: build
the synthetic demo session on a throwaway database, never the real one.

Plain-language overview: both scripts need the same data-rich session --
about 6 minutes, 1 heart-rate sample per second, four activities, a few
low-confidence predictions -- written to a database file that is NOT the
app's real one. This module holds the guard that refuses the real
database, the copy step, and the session builder, so the two scripts can't
drift apart. Pass `data_seed` to get the same heart-rate values (and so the
same average and peak) every time; leave it out for fresh random values,
which is what demo_session.py does by default.

Nothing here imports heartwatchapp.data.db at module level: importing it
fixes the database path for the whole process (see config.py), so
demo_session.py has to set HEARTWATCH_DB_PATH first. Functions that need
it import it when called, and always pass the target path explicitly.
"""

from __future__ import annotations

import contextlib
import os
import random
import sqlite3
import time
from pathlib import Path

# Where the real database lives -- the same path as data/db.py's
# DEFAULT_DB_PATH (tests/test_demo_session.py checks they agree). Worked
# out here instead of imported, for the reason in the module docstring.
REAL_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "heartwatch.db"

# The synthetic session: (activity, seconds, typical bpm). Running is
# placed so the peak heart rate lands there. 1 HR sample per second, one
# prediction per 5-second window, stamped at the window's END (the
# convention data/stats.py's PREDICTION_TIMESTAMP_MARKS describes).
DEMO_PHASES = [
    ("Sitting", 70, 72),
    ("Walking", 140, 102),
    ("Running", 45, 128),
    ("Standing", 85, 86),
]
DEMO_MODEL_VERSION = "demo-dummy"


class RealDatabaseError(RuntimeError):
    """Raised when the target database is the app's real one."""


def ensure_not_real_db(target: str | Path, real: str | Path = REAL_DB_PATH) -> Path:
    """Returns `target` as an absolute path, or raises RealDatabaseError if
    it is (or links to) the real database. This is the only thing standing
    between the demo data and the real recordings, so it checks both the
    resolved path and, if both files exist, whether they're the same file."""
    target_path = Path(target).expanduser().resolve()
    real_path = Path(real).expanduser().resolve()
    same = target_path == real_path
    if not same and target_path.exists() and real_path.exists():
        same = os.path.samefile(target_path, real_path)
    if same:
        raise RealDatabaseError(
            f"Refusing to use {target_path}: that's the app's real database. "
            "The demo scripts only ever write to a copy -- pass a different --db path "
            "or leave --db out to use a temporary file."
        )
    return target_path


def copy_real_db(target: Path) -> None:
    """Copies the real database into `target` using SQLite's backup API
    (which, unlike a plain file copy, includes anything still sitting in
    the write-ahead log). The real file is opened read-only."""
    if not REAL_DB_PATH.exists():
        return  # nothing to copy; init_db() creates an empty schema
    target.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.closing(
        sqlite3.connect(f"file:{REAL_DB_PATH}?mode=ro", uri=True)
    ) as src, contextlib.closing(sqlite3.connect(str(target))) as dst:
        src.backup(dst)


def add_demo_session(
    path: Path, rng: random.Random | None = None, data_seed: int | None = None
) -> int:
    """Writes the synthetic session into `path` (which must already have
    the schema -- see build_demo_db()) and returns its id. Heart-rate
    values and prediction confidences come from `rng`, or from a new
    random.Random(data_seed) if only `data_seed` is given, or from fresh
    randomness if neither is. Refuses the real database."""
    from heartwatchapp.data import db, seed

    path = ensure_not_real_db(path)
    if rng is None:
        rng = random.Random(data_seed)
    total_s = sum(seconds for _, seconds, _ in DEMO_PHASES)
    # Ended two minutes ago, so the default "Last 7 days" filter shows it.
    t0 = int(time.time() * 1000) - (total_s + 120) * 1000

    hr_rows, pred_rows, offset = [], [], 0
    for label, seconds, bpm in DEMO_PHASES:
        for s in range(seconds):
            hr_rows.append((t0 + (offset + s) * 1000, int(round(bpm + rng.gauss(0, 3))), 0.9))
        for window_end in range(5, seconds + 1, 5):
            # The first window after each change of activity is the
            # "unsure" one, like a real classifier at a transition.
            confidence = 0.52 if window_end == 5 and offset else round(rng.uniform(0.8, 0.97), 2)
            pred_rows.append((t0 + (offset + window_end) * 1000, label, confidence))
        offset += seconds

    with contextlib.closing(db._connect(path)) as conn, conn:
        session_id = db.start_session(conn, "demo-session", "Walking", 1.0, 1.0, started_at=t0)
        db.insert_imu_batch(conn, session_id, seed.generate_imu_rows(t0, total_s))
        db.insert_hr_batch(conn, session_id, hr_rows)
        for ts, label, confidence in pred_rows:
            db.insert_prediction(conn, session_id, ts, label, confidence, DEMO_MODEL_VERSION)
        db.end_session(conn, session_id, t0 + total_s * 1000)
    return session_id


def build_demo_db(
    target: str | Path, *, copy_real: bool = True, data_seed: int | None = None
) -> tuple[Path, int]:
    """Refuses the real database (RealDatabaseError), then -- if `target`
    doesn't exist yet and `copy_real` is set -- copies the real database
    into it, applies the schema, and adds one demo session. Returns
    (absolute target path, new session id). With copy_real=False a new
    target starts empty, so the real database isn't opened at all."""
    from heartwatchapp.data import db

    target = ensure_not_real_db(target)
    if copy_real and not target.exists():
        copy_real_db(target)
    db.init_db(target)
    return target, add_demo_session(target, data_seed=data_seed)
