"""Shared by scripts/demo_session.py and scripts/eval_summaries.py: build
the synthetic demo session on a throwaway database, never the real one.

Plain-language overview: both scripts need the same data-rich session --
about 6 minutes, 1 heart-rate sample per second, four activities, a few
low-confidence predictions -- written to a database file that is NOT the
app's real one. By default that file is brand new (the app's own schema
plus the demo session), so the real database is never opened at all. This
module holds the guard that refuses the real database, the builder, and
the opt-in file-level copy of the real database (--from-real), so the two
scripts can't drift apart.

Why the copy is a plain file copy: opening the real database with SQLite,
even read-only (mode=ro), still writes to its -shm file. So the copy step
copies the .db, -wal and -shm files with shutil and only ever opens the
copy. Pass `data_seed` to get the same heart-rate values (and so the
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
import shutil
import sqlite3
import time
from collections.abc import Mapping
from pathlib import Path

from heartwatchapp.config import DB_PATH_ENV_VAR

# Where the real database lives -- the same path as data/db.py's
# DEFAULT_DB_PATH (tests/test_demo_session.py checks they agree). Worked
# out here instead of imported, for the reason in the module docstring.
REAL_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "heartwatch.db"

# SQLite's companion files for a database at "x.db": "x.db-wal", etc.
SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")

FROM_REAL_WARNING = (
    "Close HeartWatch before using --from-real; copying while the app is "
    "writing can produce an inconsistent copy."
)

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


class DemoDbError(RuntimeError):
    """Raised when --from-real can't produce a usable copy."""


class DestinationExistsError(RuntimeError):
    """Raised when a user-chosen --db file already exists and --overwrite
    wasn't given."""


def real_db_path(environ: Mapping[str, str] = os.environ) -> Path:
    """The database the app would use: HEARTWATCH_DB_PATH if it's set (and
    not empty), otherwise data/heartwatch.db -- the same rule as data/db.py's
    resolve_db_path() (tests/test_demo_db_fresh.py checks they agree).
    Only the path is worked out; the file is never opened. Callers that
    change HEARTWATCH_DB_PATH (demo_session.py does) must call this first."""
    override = environ.get(DB_PATH_ENV_VAR)
    return Path(override).expanduser().resolve() if override else REAL_DB_PATH


def ensure_not_real_db(target: str | Path, real: str | Path | None = None) -> Path:
    """Returns `target` as an absolute path, or raises RealDatabaseError if
    it is (or links to) the real database (`real`, or real_db_path() if
    not given). This is the only thing standing between the demo data and
    the real recordings, so it checks both the resolved path and, if both
    files exist, whether they're the same file."""
    target_path = Path(target).expanduser().resolve()
    real_path = Path(real if real is not None else real_db_path()).expanduser().resolve()
    same = target_path == real_path
    if not same and target_path.exists() and real_path.exists():
        same = os.path.samefile(target_path, real_path)
    if same:
        raise RealDatabaseError(
            f"Refusing to use {target_path}: that's the app's real database. "
            "The demo scripts never write to it -- pass a different --db path "
            "or leave --db out to use a temporary file."
        )
    return target_path


def check_destination(
    path: str | Path, overwrite: bool, real: str | Path | None = None
) -> Path:
    """The checks a script runs on its --db path before building anything,
    in this order: first the real-database guard (ensure_not_real_db() --
    RealDatabaseError, whatever `overwrite` says), then, unless `overwrite`
    is set, refuse a file that's already there (DestinationExistsError),
    since build_demo_db() deletes whatever is at the destination. Returns
    the absolute path. Nothing is deleted or opened here."""
    target = ensure_not_real_db(path, real)
    if target.exists() and not overwrite:
        raise DestinationExistsError(
            f"{target} already exists. Building the demo database would delete it "
            "-- pass --overwrite to replace it, or choose another --db path."
        )
    return target


def db_files(path: Path) -> list[Path]:
    """`path` and its SQLite companion files (whether they exist or not)."""
    return [path] + [Path(f"{path}{suffix}") for suffix in SIDECAR_SUFFIXES]


def remove_db_files(path: Path) -> None:
    """Deletes `path` and any -wal / -shm / -journal next to it. A leftover
    -wal from an earlier demo database would otherwise be replayed into
    the new file and could corrupt it."""
    for file in db_files(path):
        file.unlink(missing_ok=True)


def integrity_check(path: Path) -> str:
    """SQLite's PRAGMA integrity_check on `path`: "ok" if all is well."""
    with contextlib.closing(sqlite3.connect(str(path))) as conn:
        return conn.execute("PRAGMA integrity_check").fetchone()[0]


def copy_real_db(target: Path, real: Path) -> None:
    """Copies the real database's .db, then its -wal and -shm if they
    exist, into `target` at the file level (shutil.copy2). The real files
    are never opened with SQLite in any mode -- even mode=ro writes to the
    -shm. Only the copy is opened, for an integrity check; if that fails
    the copy is deleted and DemoDbError raised."""
    if not real.exists():
        raise DemoDbError(f"There's no real database at {real} to copy.")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(real, target)
    for suffix in ("-wal", "-shm"):
        sidecar = Path(f"{real}{suffix}")
        if sidecar.exists():
            shutil.copy2(sidecar, Path(f"{target}{suffix}"))
    result = integrity_check(target)
    if result != "ok":
        remove_db_files(target)
        raise DemoDbError(
            f"The copy of {real} failed SQLite's integrity check ({result!r}), "
            "so it was deleted. Close HeartWatch and try again."
        )


def add_demo_session(
    path: Path,
    rng: random.Random | None = None,
    data_seed: int | None = None,
    real: str | Path | None = None,
) -> int:
    """Writes the synthetic session into `path` (which must already have
    the schema -- see build_demo_db()) and returns its id. Heart-rate
    values and prediction confidences come from `rng`, or from a new
    random.Random(data_seed) if only `data_seed` is given, or from fresh
    randomness if neither is. Refuses the real database (`real`, or
    real_db_path())."""
    total_s = sum(seconds for _, seconds, _ in DEMO_PHASES)
    # Ended two minutes ago, so the default "Last 7 days" filter shows it.
    t0 = int(time.time() * 1000) - (total_s + 120) * 1000
    return add_phases_session(path, t0, rng=rng, data_seed=data_seed, real=real)


def add_phases_session(
    path: Path,
    t0: int,
    phases: list[tuple[str, int, int]] = DEMO_PHASES,
    rng: random.Random | None = None,
    data_seed: int | None = None,
    real: str | Path | None = None,
) -> int:
    """add_demo_session(), but starting at `t0` (epoch ms) and with your
    own (label, seconds, bpm) phases. For scripts that need sessions at
    fixed times, e.g. scripts/eval_ask.py placing sessions around a pinned
    "now". Same randomness and the same real-database refusal."""
    from heartwatchapp.data import db, seed

    path = ensure_not_real_db(path, real)
    if rng is None:
        rng = random.Random(data_seed)
    total_s = sum(seconds for _, seconds, _ in phases)

    hr_rows, pred_rows, offset = [], [], 0
    for label, seconds, bpm in phases:
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
    target: str | Path,
    *,
    from_real: bool = False,
    data_seed: int | None = None,
    real: str | Path | None = None,
) -> tuple[Path, int]:
    """Refuses the real database (RealDatabaseError), deletes any existing
    `target` and its companion files, then builds a new database there and
    adds one demo session. Returns (absolute target path, new session id).

    By default the new database is empty apart from the app's schema
    (data/db.py's init_db()), and the real database is never opened. With
    from_real=True it starts as a file-level copy of the real database
    (`real`, or real_db_path()) instead -- see copy_real_db(). If anything
    fails part-way, nothing is left at `target`."""
    from heartwatchapp.data import db

    real_path = Path(real) if real is not None else real_db_path()
    target = ensure_not_real_db(target, real_path)
    remove_db_files(target)
    try:
        if from_real:
            copy_real_db(target, real_path)
        db.init_db(target)
        return target, add_demo_session(target, data_seed=data_seed, real=real_path)
    except BaseException:
        remove_db_files(target)
        raise
