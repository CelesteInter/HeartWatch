"""DatabaseWriter -- the single thread allowed to write to heartwatch.db.

Why this exists: SQLite connections are not safe to share across threads,
and the app has several (Qt main thread, Live Monitor generator ticks,
eventually the BLE receive thread). Rather than open-and-close a
connection per write, one background thread owns a single connection for
the app's lifetime and pulls queued work off a queue.Queue.

Two kinds of work move through the queue:

- High-rate sample batches (insert_imu_batch / insert_hr_batch /
  insert_prediction) are fire-and-forget. They are buffered and flushed
  together -- whichever comes first of ~1 second elapsed or ~100 rows
  buffered -- as one executemany() per table inside one transaction. At
  10 Hz the IMU alone produces ~36,000 rows/hour; one disk sync per row
  would not keep up once BLE is live, so this batching is built now.
- Control operations (start_session, end_session, delete_session) need a
  result back on the caller's thread (a new session id, or confirmation
  a delete finished before the UI refreshes row counts). These block the
  calling thread on a one-shot queue.Queue until the writer thread has
  executed them -- flushing any buffered sample rows first, so a
  start_session/delete_session a caller issues is never reordered ahead
  of samples that were queued before it.
"""

from __future__ import annotations

import queue
import random
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any

from . import db as db_ops
from . import seed as seed_gen

FLUSH_MAX_ROWS = 100
FLUSH_MAX_SECONDS = 1.0
_QUEUE_POLL_SECONDS = 0.2


class _Kind(Enum):
    INSERT_IMU = auto()
    INSERT_HR = auto()
    INSERT_PREDICTION = auto()
    START_SESSION = auto()
    END_SESSION = auto()
    DELETE_SESSION = auto()
    SEED_DEMO = auto()
    FLUSH = auto()
    STOP = auto()


@dataclass
class _Job:
    kind: _Kind
    payload: Any = None
    reply: "queue.Queue[tuple[bool, Any]] | None" = None  # (ok, result_or_exception)


_BATCH_KINDS = (_Kind.INSERT_IMU, _Kind.INSERT_HR, _Kind.INSERT_PREDICTION)


class DatabaseWriter:
    """Owns the one write connection to heartwatch.db and its worker thread."""

    def __init__(self, db_path: str | Path = db_ops.DB_PATH):
        self._db_path = db_path
        self._queue: "queue.Queue[_Job]" = queue.Queue()
        self._thread: threading.Thread | None = None
        self._started = False

    # -- lifecycle -------------------------------------------------------

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._thread = threading.Thread(
            target=self._run, name="HeartWatchDBWriter", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        if not self._started:
            return
        self._queue.put(_Job(_Kind.STOP))
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self._started = False

    # -- fire-and-forget batch inserts (called from any thread) ----------

    def insert_imu_batch(self, session_id: int, rows: list[tuple]) -> None:
        if rows:
            self._queue.put(_Job(_Kind.INSERT_IMU, (session_id, rows)))

    def insert_hr_batch(self, session_id: int, rows: list[tuple]) -> None:
        if rows:
            self._queue.put(_Job(_Kind.INSERT_HR, (session_id, rows)))

    def insert_prediction(
        self, session_id: int, ts: int, predicted: str, confidence: float, model_version: str
    ) -> None:
        row = (ts, predicted, confidence, model_version)
        self._queue.put(_Job(_Kind.INSERT_PREDICTION, (session_id, [row])))

    # -- synchronous control operations (block the calling thread) -------

    def start_session(
        self,
        device_id: str | None,
        label: str | None,
        accel_scale: float,
        gyro_scale: float,
    ) -> int:
        started_at = int(time.time() * 1000)
        payload = (device_id, label, accel_scale, gyro_scale, started_at)
        return self._submit_sync(_Kind.START_SESSION, payload)

    def end_session(self, session_id: int) -> None:
        ended_at = int(time.time() * 1000)
        self._submit_sync(_Kind.END_SESSION, (session_id, ended_at))

    def delete_session(self, session_id: int) -> None:
        self._submit_sync(_Kind.DELETE_SESSION, session_id)

    def seed_demo_session(
        self, label: str | None = None, duration_s: float | None = None
    ) -> int:
        """Generates 2-3 minutes of synthetic IMU/HR data and writes it through
        the same insert_imu_batch / insert_hr_batch statements the live batch
        flush uses -- not a separate shortcut, just run synchronously as one
        control job so the session can be backdated to look like a real past
        recording. Returns the new session id."""
        label = label or seed_gen.random_label()
        duration_s = duration_s if duration_s is not None else random.uniform(120, 180)
        ended_at = int(time.time() * 1000)
        started_at = ended_at - int(duration_s * 1000)
        imu_rows = seed_gen.generate_imu_rows(started_at, duration_s)
        hr_rows = seed_gen.generate_hr_rows(started_at, duration_s)
        payload = (label, started_at, ended_at, imu_rows, hr_rows)
        return self._submit_sync(_Kind.SEED_DEMO, payload)

    def flush_and_wait(self) -> None:
        """Blocks until every batch queued before this call has been committed.
        Used before reads (e.g. the developer panel's row counts) that must
        reflect data submitted moments earlier."""
        self._submit_sync(_Kind.FLUSH, None)

    def _submit_sync(self, kind: _Kind, payload: Any) -> Any:
        reply: "queue.Queue[tuple[bool, Any]]" = queue.Queue(maxsize=1)
        self._queue.put(_Job(kind, payload, reply))
        ok, result = reply.get()
        if not ok:
            raise result
        return result

    # -- worker thread -----------------------------------------------------

    def _run(self) -> None:
        conn = db_ops._connect(self._db_path)
        pending: list[_Job] = []
        pending_rows = 0
        oldest_pending: float | None = None
        try:
            while True:
                try:
                    job = self._queue.get(timeout=_QUEUE_POLL_SECONDS)
                except queue.Empty:
                    job = None

                if job is not None:
                    if job.kind in _BATCH_KINDS:
                        pending.append(job)
                        pending_rows += len(job.payload[1])
                        if oldest_pending is None:
                            oldest_pending = time.monotonic()
                    elif job.kind is _Kind.STOP:
                        self._flush(conn, pending)
                        pending, pending_rows, oldest_pending = [], 0, None
                        break
                    else:
                        # Control op or explicit flush: drain buffered samples
                        # first so ordering matches submission order, then run.
                        self._flush(conn, pending)
                        pending, pending_rows, oldest_pending = [], 0, None
                        self._run_control(conn, job)

                due = pending and (
                    pending_rows >= FLUSH_MAX_ROWS
                    or (oldest_pending is not None
                        and time.monotonic() - oldest_pending >= FLUSH_MAX_SECONDS)
                )
                if due:
                    self._flush(conn, pending)
                    pending, pending_rows, oldest_pending = [], 0, None
        finally:
            conn.close()

    def _flush(self, conn, jobs: list[_Job]) -> None:
        if not jobs:
            return
        imu_rows: list[tuple] = []
        hr_rows: list[tuple] = []
        pred_rows: list[tuple] = []
        for job in jobs:
            session_id, rows = job.payload
            if job.kind is _Kind.INSERT_IMU:
                imu_rows.extend((session_id, *r) for r in rows)
            elif job.kind is _Kind.INSERT_HR:
                hr_rows.extend((session_id, *r) for r in rows)
            elif job.kind is _Kind.INSERT_PREDICTION:
                pred_rows.extend((session_id, *r) for r in rows)

        with conn:  # one transaction for the whole flush
            if imu_rows:
                db_ops._insert_imu_rows(conn, imu_rows)
            if hr_rows:
                db_ops._insert_hr_rows(conn, hr_rows)
            if pred_rows:
                db_ops._insert_prediction_rows(conn, pred_rows)

    def _run_control(self, conn, job: _Job) -> None:
        try:
            if job.kind is _Kind.START_SESSION:
                device_id, label, accel_scale, gyro_scale, started_at = job.payload
                with conn:
                    result = db_ops.start_session(
                        conn, device_id, label, accel_scale, gyro_scale, started_at
                    )
            elif job.kind is _Kind.END_SESSION:
                session_id, ended_at = job.payload
                with conn:
                    db_ops.end_session(conn, session_id, ended_at)
                result = None
            elif job.kind is _Kind.DELETE_SESSION:
                with conn:
                    db_ops.delete_session(conn, job.payload)
                result = None
            elif job.kind is _Kind.SEED_DEMO:
                label, started_at, ended_at, imu_rows, hr_rows = job.payload
                with conn:
                    result = db_ops.start_session(
                        conn, "seed", label, 1.0, 1.0, started_at
                    )
                    db_ops.insert_imu_batch(conn, result, imu_rows)
                    db_ops.insert_hr_batch(conn, result, hr_rows)
                    db_ops.end_session(conn, result, ended_at)
            elif job.kind is _Kind.FLUSH:
                result = None
            else:  # pragma: no cover - defensive
                raise ValueError(f"unknown job kind {job.kind}")
        except Exception as exc:  # noqa: BLE001 - relayed to the caller's thread
            if job.reply is not None:
                job.reply.put((False, exc))
            return
        if job.reply is not None:
            job.reply.put((True, result))
