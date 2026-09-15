"""SQLite data layer for HeartWatch.

Plain-language overview: one SQLite file (`heartwatch.db`) holds every
recording the app has ever made. A *session* is one recording run; IMU
samples, heart-rate samples, and model predictions all reference the
session they belong to. Deleting a session cascades to delete everything
that hangs off it. See heartwatchapp/docs/SCHEMA.md for the full explanation.

Threading model: SQLite connections are not safe to share across threads.
Reads in this module open a short-lived connection per call and are safe
to call from any thread, including the Qt main thread. Writes are NOT
performed directly by this module's insert_* functions when a
DatabaseWriter is available -- see heartwatchapp/data/writer.py, which owns a
single dedicated writer thread and batches inserts for it. The functions
here (`insert_imu_batch`, `insert_hr_batch`, etc.) are the low-level
statements the writer thread calls; call them directly only if you are
implementing / testing the writer itself.
"""

from __future__ import annotations

import contextlib
import csv
import sqlite3
import time
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "heartwatch.db"
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"

# Bumped whenever schema.sql changes in a way old data files won't have.
EXPECTED_SCHEMA_VERSION = 1


def _now_ms() -> int:
    return int(time.time() * 1000)


# --- connection handling ---------------------------------------------

def _connect(path: str | Path = DB_PATH) -> sqlite3.Connection:
    """Open a connection with the pragmas HeartWatch requires.

    Applied on every connection, not just at init -- SQLite disables
    foreign-key enforcement by default, so ON DELETE CASCADE silently
    does nothing unless `PRAGMA foreign_keys = ON` is set on the specific
    connection doing the delete.
    """
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")     # readers don't block the writer
    conn.execute("PRAGMA synchronous  = NORMAL")  # safe under WAL, much faster
    conn.execute("PRAGMA foreign_keys = ON")      # OFF by default in SQLite
    return conn


def init_db(path: str | Path = DB_PATH) -> None:
    """Create the database file if absent, apply schema.sql, and verify
    schema_meta.version matches what this build of the app expects."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with contextlib.closing(_connect(path)) as conn:
        schema_sql = SCHEMA_PATH.read_text()
        conn.executescript(schema_sql)
        conn.commit()

        row = conn.execute("SELECT version FROM schema_meta WHERE id = 1").fetchone()
        if row is None:
            raise RuntimeError(
                "schema_meta has no row after init_db() -- schema.sql did not run "
                "as expected."
            )
        if row["version"] != EXPECTED_SCHEMA_VERSION:
            raise RuntimeError(
                f"Database at {path} has schema version {row['version']}, but this "
                f"build of HeartWatch expects version {EXPECTED_SCHEMA_VERSION}. "
                "Delete the .db/.db-wal/.db-shm files and relaunch, or migrate."
            )


# --- writes (called by the DatabaseWriter thread) ----------------------
#
# These take an explicit `conn` rather than opening their own: the writer
# thread in data/writer.py owns the single write connection for the app's
# lifetime and calls these as plain SQL statements. Application code (the
# UI) should call the DatabaseWriter's methods of the same name instead --
# those queue the work and hand back a result once the writer thread has
# actually run it, which is what keeps SQLite writes off the Qt thread.

def start_session(
    conn: sqlite3.Connection,
    device_id: str | None,
    label: str | None,
    accel_scale: float,
    gyro_scale: float,
    started_at: int,
) -> int:
    """Open a new session. Returns the new session id.
    ended_at is left NULL, which is how the app knows it is live."""
    cur = conn.execute(
        "INSERT INTO sessions (started_at, device_id, label, accel_scale, gyro_scale) "
        "VALUES (?, ?, ?, ?, ?)",
        (started_at, device_id, label, accel_scale, gyro_scale),
    )
    return cur.lastrowid


def end_session(conn: sqlite3.Connection, session_id: int, ended_at: int) -> None:
    """Stamp ended_at with the current time."""
    conn.execute("UPDATE sessions SET ended_at = ? WHERE id = ?", (ended_at, session_id))


def _insert_imu_rows(conn: sqlite3.Connection, rows: list[tuple]) -> None:
    """rows: [(session_id, ts, ax, ay, az, gx, gy, gz), ...] -- already flattened,
    possibly spanning multiple sessions. Called by the writer's batch flush."""
    conn.executemany(
        "INSERT INTO imu_samples (session_id, ts, ax, ay, az, gx, gy, gz) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )


def _insert_hr_rows(conn: sqlite3.Connection, rows: list[tuple]) -> None:
    """rows: [(session_id, ts, bpm, confidence), ...]"""
    conn.executemany(
        "INSERT INTO hr_samples (session_id, ts, bpm, confidence) VALUES (?, ?, ?, ?)",
        rows,
    )


def _insert_prediction_rows(conn: sqlite3.Connection, rows: list[tuple]) -> None:
    """rows: [(session_id, ts, predicted, confidence, model_version), ...]"""
    conn.executemany(
        "INSERT INTO predictions (session_id, ts, predicted, confidence, model_version) "
        "VALUES (?, ?, ?, ?, ?)",
        rows,
    )


def insert_imu_batch(conn: sqlite3.Connection, session_id: int, rows: list[tuple]) -> None:
    """rows: [(ts, ax, ay, az, gx, gy, gz), ...]"""
    _insert_imu_rows(conn, [(session_id, *r) for r in rows])


def insert_hr_batch(conn: sqlite3.Connection, session_id: int, rows: list[tuple]) -> None:
    """rows: [(ts, bpm, confidence), ...]"""
    _insert_hr_rows(conn, [(session_id, *r) for r in rows])


def insert_prediction(
    conn: sqlite3.Connection,
    session_id: int,
    ts: int,
    predicted: str,
    confidence: float,
    model_version: str,
) -> None:
    _insert_prediction_rows(conn, [(session_id, ts, predicted, confidence, model_version)])


def delete_session(conn: sqlite3.Connection, session_id: int) -> None:
    """Deletes the session; CASCADE removes its samples and predictions.
    Requires PRAGMA foreign_keys = ON on this connection (set by _connect())."""
    conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))


# --- reads (direct, short-lived connections; safe from the Qt main thread) --

def list_sessions(
    limit: int = 50,
    activity_filter: str | None = None,
    start_date: int | None = None,
    end_date: int | None = None,
) -> list[dict]:
    """Backs the Sessions view. Each dict has: id, started_at, ended_at
    (epoch ms; ended_at is None for a still-live session), duration_s,
    device_id, label, sample_count (IMU rows), avg_bpm, peak_bpm (both
    None if the session has no HR samples yet). Newest first."""
    query = """
        SELECT
            s.id, s.started_at, s.ended_at, s.device_id, s.label,
            (SELECT COUNT(*) FROM imu_samples i WHERE i.session_id = s.id) AS sample_count,
            (SELECT AVG(bpm) FROM hr_samples h WHERE h.session_id = s.id) AS avg_bpm,
            (SELECT MAX(bpm) FROM hr_samples h WHERE h.session_id = s.id) AS peak_bpm
        FROM sessions s
        WHERE (:label IS NULL OR s.label = :label)
          AND (:start_date IS NULL OR s.started_at >= :start_date)
          AND (:end_date IS NULL OR s.started_at <= :end_date)
        ORDER BY s.started_at DESC
        LIMIT :limit
    """
    params = {
        "label": activity_filter,
        "start_date": start_date,
        "end_date": end_date,
        "limit": limit,
    }
    with contextlib.closing(_connect()) as conn:
        rows = conn.execute(query, params).fetchall()

    now = _now_ms()
    out = []
    for r in rows:
        ended_at = r["ended_at"]
        duration_s = ((ended_at if ended_at is not None else now) - r["started_at"]) / 1000
        out.append(
            {
                "id": r["id"],
                "started_at": r["started_at"],
                "ended_at": ended_at,
                "duration_s": duration_s,
                "device_id": r["device_id"],
                "label": r["label"],
                "sample_count": r["sample_count"],
                "avg_bpm": r["avg_bpm"],
                "peak_bpm": r["peak_bpm"],
            }
        )
    return out


def get_session_timeline(session_id: int) -> dict:
    """Backs the per-session detail view.

    Returns {"session": {...}, "hr": [(ts, bpm), ...],
    "imu": [(ts, ax, ay, az, gx, gy, gz), ...],
    "predictions": [(ts, predicted, confidence), ...]}.

    Note on IMU: the handoff's API sketch describes each series as (ts,
    value) pairs. A single IMU sample carries six axis values, not one,
    so collapsing it to a scalar would throw away data the detail view
    needs -- the IMU series here is (ts, ax, ay, az, gx, gy, gz) tuples
    instead. HR and predictions match the (ts, value) shape as described.
    """
    with contextlib.closing(_connect()) as conn:
        session_row = conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        hr_rows = conn.execute(
            "SELECT ts, bpm FROM hr_samples WHERE session_id = ? ORDER BY ts",
            (session_id,),
        ).fetchall()
        imu_rows = conn.execute(
            "SELECT ts, ax, ay, az, gx, gy, gz FROM imu_samples "
            "WHERE session_id = ? ORDER BY ts",
            (session_id,),
        ).fetchall()
        pred_rows = conn.execute(
            "SELECT ts, predicted, confidence FROM predictions "
            "WHERE session_id = ? ORDER BY ts",
            (session_id,),
        ).fetchall()

    return {
        "session": dict(session_row) if session_row is not None else None,
        "hr": [(r["ts"], r["bpm"]) for r in hr_rows],
        "imu": [tuple(r) for r in imu_rows],
        "predictions": [(r["ts"], r["predicted"], r["confidence"]) for r in pred_rows],
    }


def get_dashboard_stats(days: int = 7) -> dict:
    """Backs the four Dashboard metric cards and the activity breakdown.

    Returns {"avg_hr": float|None, "peak_hr": int|None, "session_count": int,
    "total_active_seconds": float, "activity_breakdown": {label: seconds}}.
    Sessions are scoped to the window by started_at; a still-live session's
    duration counts up to "now"."""
    cutoff = _now_ms() - days * 86_400_000
    with contextlib.closing(_connect()) as conn:
        sessions = conn.execute(
            "SELECT id, started_at, ended_at, label FROM sessions WHERE started_at >= ?",
            (cutoff,),
        ).fetchall()
        hr_agg = conn.execute(
            "SELECT AVG(bpm) AS avg_bpm, MAX(bpm) AS peak_bpm FROM hr_samples "
            "WHERE session_id IN (SELECT id FROM sessions WHERE started_at >= ?)",
            (cutoff,),
        ).fetchone()

    now = _now_ms()
    breakdown: dict[str, float] = {}
    total_active_s = 0.0
    for r in sessions:
        ended_at = r["ended_at"] if r["ended_at"] is not None else now
        duration_s = max(0, ended_at - r["started_at"]) / 1000
        total_active_s += duration_s
        label = r["label"] or "Unknown"
        breakdown[label] = breakdown.get(label, 0.0) + duration_s

    return {
        "avg_hr": hr_agg["avg_bpm"],
        "peak_hr": hr_agg["peak_bpm"],
        "session_count": len(sessions),
        "total_active_seconds": total_active_s,
        "activity_breakdown": breakdown,
    }


def export_training_csv(session_ids: list[int], path: str) -> int:
    """Writes ts, ax..gz, label rows for the CNN training pipeline.
    Returns the number of rows written. Backs the ML section's
    'Export training data' button."""
    if not session_ids:
        return 0
    placeholders = ",".join("?" for _ in session_ids)
    query = f"""
        SELECT i.ts, i.ax, i.ay, i.az, i.gx, i.gy, i.gz, s.label
        FROM imu_samples i
        JOIN sessions s ON s.id = i.session_id
        WHERE i.session_id IN ({placeholders})
        ORDER BY i.session_id, i.ts
    """
    with contextlib.closing(_connect()) as conn:
        rows = conn.execute(query, session_ids).fetchall()

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ts", "ax", "ay", "az", "gx", "gy", "gz", "label"])
        writer.writerows(tuple(r) for r in rows)
    return len(rows)


# --- developer-panel helpers -------------------------------------------

def get_db_info(path: str | Path = DB_PATH) -> dict:
    """File path, file size on disk, row counts per table, schema version.
    Backs the Settings developer panel's 'Database info' control."""
    path = Path(path)
    table_names = ["sessions", "imu_samples", "hr_samples", "predictions"]
    with contextlib.closing(_connect(path)) as conn:
        row_counts = {
            name: conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
            for name in table_names
        }
        version_row = conn.execute("SELECT version FROM schema_meta WHERE id = 1").fetchone()
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "row_counts": row_counts,
        "schema_version": version_row["version"] if version_row else None,
    }


def run_samples_flat_demo(limit: int = 20) -> list[dict]:
    """SELECT * FROM samples_flat LIMIT ? -- the acceptance-review demo.
    Confirms the view returns exactly the task-specified format:
    (DATETIME, x, y, z, heartrate)."""
    with contextlib.closing(_connect()) as conn:
        rows = conn.execute("SELECT * FROM samples_flat LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]
