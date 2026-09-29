"""Try the AI session summaries on a throwaway copy of the database.

Plain-language overview: real recorded sessions don't have activity
predictions yet (the classifier arrives in milestone 6), so they never get
an AI summary. This script makes a copy of the real database, adds one
data-rich synthetic session to the COPY -- about 6 minutes, 1 heart-rate
sample per second, four activities, a few low-confidence predictions --
and launches the real app pointed at the copy, with the AI summary log
("heartwatch.llm") turned up to DEBUG so every pass and rejection prints
in the terminal. The real database is only ever read, never written.

Run from the repo root, with the venv active:

    python -m heartwatchapp.scripts.demo_session             # temp copy
    python -m heartwatchapp.scripts.demo_session --db x.db   # reuse/choose a copy
    python -m heartwatchapp.scripts.demo_session --no-launch # just build the copy

Then open Sessions, set the filter to "All time" if needed, and
double-click the session number the script prints.

How it points the app elsewhere: it sets HEARTWATCH_DB_PATH (read by
data/db.py) BEFORE importing any app code, and refuses to run if that
path turns out to be the real database.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import random
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Where the real database lives -- the same path as data/db.py's
# DEFAULT_DB_PATH (tests/test_demo_session.py checks they agree). Worked
# out here instead of imported, because importing data/db.py fixes the
# database path for the whole app, and that must not happen until
# HEARTWATCH_DB_PATH has been set.
REAL_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "heartwatch.db"
DB_PATH_ENV_VAR = "HEARTWATCH_DB_PATH"

# The synthetic session: (activity, seconds, typical bpm). Running is
# placed so the peak heart rate lands there. 1 HR sample per second, one
# prediction per 5-second window.
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
            "demo_session.py only ever writes to a copy -- pass a different --db path "
            "or leave --db out to use a temporary copy."
        )
    return target_path


def _copy_real_db(target: Path) -> None:
    """Copies the real database into `target` using SQLite's backup API
    (which, unlike a plain file copy, includes anything still sitting in
    the write-ahead log). The real file is opened read-only."""
    if not REAL_DB_PATH.exists():
        return  # nothing to copy; init_db() creates an empty schema below
    target.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.closing(
        sqlite3.connect(f"file:{REAL_DB_PATH}?mode=ro", uri=True)
    ) as src, contextlib.closing(sqlite3.connect(str(target))) as dst:
        src.backup(dst)


def _add_demo_session(db, seed, path: Path) -> int:
    """Writes the synthetic session into `path` and returns its id."""
    rng = random.Random()
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--db",
        help="database file to use (created as a copy of the real one if it "
        "doesn't exist yet). Default: a new temporary copy.",
    )
    parser.add_argument(
        "--no-launch", action="store_true", help="build the demo database, then exit"
    )
    args = parser.parse_args(argv)

    if args.db:
        target = Path(args.db)
    else:
        target = Path(tempfile.mkdtemp(prefix="heartwatch-demo-")) / "heartwatch_demo.db"
    try:
        target = ensure_not_real_db(target)
    except RealDatabaseError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # Point the whole app at the copy BEFORE importing any of it.
    os.environ[DB_PATH_ENV_VAR] = str(target)

    # DEBUG for the AI summary log so passes print too, not just rejections.
    # Configured before main.py's basicConfig, which then leaves it alone.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("heartwatch.llm").setLevel(logging.DEBUG)

    from heartwatchapp.data import db, seed

    if db.DB_PATH != target:  # belt and braces: never touch anything else
        print(
            f"error: the app would use {db.DB_PATH}, not {target} "
            "(was heartwatchapp.data.db imported too early?). Nothing was written.",
            file=sys.stderr,
        )
        return 2

    if not target.exists():
        _copy_real_db(target)
    db.init_db(target)
    session_id = _add_demo_session(db, seed, target)
    print(f"\n>>> Demo session #{session_id} added to {target}")
    print(">>> Open Sessions and double-click it to generate a summary.\n", flush=True)

    if args.no_launch:
        return 0
    from heartwatchapp import main as app_main

    return app_main.main()


if __name__ == "__main__":
    raise SystemExit(main())
