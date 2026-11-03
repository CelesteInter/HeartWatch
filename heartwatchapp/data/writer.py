"""DatabaseWriter -- the single thread allowed to write to heartwatch.db.

Why this exists: SQLite connections are not safe to share across threads,
and the app has several (Qt main thread, the recording controller's
sample writes, eventually the BLE receive thread). Rather than open-and-close a
connection per write, one background thread owns a single connection for
the app's lifetime and pulls queued work off a queue.Queue.

Two kinds of work move through the queue:

- High-rate sample batches (insert_imu_batch / insert_hr_batch /
  insert_prediction) are fire-and-forget. They are buffered and flushed
  together -- whichever comes first of ~1 second elapsed or ~100 rows
  buffered -- as one executemany() per table inside one transaction. At
  10 Hz the IMU alone produces ~36,000 rows/hour; one disk sync per row
  would not keep up once BLE is live, so this batching is built now.
- Control operations (start_session, end_session, delete_session,
  close_orphan_sessions) need a
  result back on the caller's thread (a new session id, or confirmation
  a delete finished before the UI refreshes row counts). These block the
  calling thread on a one-shot queue.Queue until the writer thread has
  executed them -- flushing any buffered sample rows first, so a
  start_session/delete_session a caller issues is never reordered ahead
  of samples that were queued before it.

When something goes wrong (handoff 11): the first error while opening the
database or saving a batch of samples -- a full disk, or rows for a
session that was deleted under them -- puts the writer into a "failed"
state until the app restarts. The error is logged once (logger
"heartwatch.db", with the traceback), the batch that failed is dropped,
and from then on the thread stays alive only to answer: every control
operation gets WriterFailed straight away, sample batches are dropped,
and stop() still stops it. No retries.

A caller never waits forever: control operations give up after
SYNC_TIMEOUT_S (5 s; the quit path passes 2 s) with WriterUnavailable if
the thread is stuck or gone. A timed-out operation stays queued and may
still run later -- e.g. a session that ends up opened after start_session
gave up; next launch's crash recovery closes it.

An error raised by a control operation itself (not by the sample flush
before it) is still handed back to that one caller unchanged, as before,
and does not fail the writer.
"""

from __future__ import annotations

import logging
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
# How long a control operation waits for the writer thread by default.
SYNC_TIMEOUT_S = 5.0
# The quit path's limit for end_session and stop() each, so quitting
# while recording takes at most about 4 s even if the writer is stuck.
QUIT_TIMEOUT_S = 2.0

logger = logging.getLogger("heartwatch.db")


class WriterError(Exception):
    """The database writer couldn't do what was asked."""


class WriterFailed(WriterError):
    """The writer hit an error earlier in this run and has stopped writing
    until the app restarts. `original` is that first error."""

    def __init__(self, original: BaseException):
        super().__init__(f"database writer failed earlier: {original!r}")
        self.original = original


class WriterUnavailable(WriterError):
    """The writer thread didn't answer in time, isn't running, or is gone."""


class _Kind(Enum):
    INSERT_IMU = auto()
    INSERT_HR = auto()
    INSERT_PREDICTION = auto()
    START_SESSION = auto()
    END_SESSION = auto()
    DELETE_SESSION = auto()
    CLOSE_ORPHANS = auto()
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

    def __init__(self, db_path: str | Path | None = None):
        """`db_path`: the database to write; None means the app's own
        (db.DB_PATH, read when the writer is created)."""
        self._db_path = db_path if db_path is not None else db_ops.DB_PATH
        self._queue: "queue.Queue[_Job]" = queue.Queue()
        self._thread: threading.Thread | None = None
        self._started = False
        self._failed: BaseException | None = None
        self._failed_lock = threading.Lock()

    @property
    def failed(self) -> BaseException | None:
        """None while the writer works; otherwise the first error that made
        it fail. Safe to read from any thread."""
        with self._failed_lock:
            return self._failed

    def _fail(self, exc: BaseException, what: str) -> None:
        """Writer thread only: enter the failed state (once) and log it."""
        with self._failed_lock:
            if self._failed is not None:
                return
            self._failed = exc
        logger.error(
            "Database writer failed while %s; nothing more will be saved until "
            "the app restarts", what, exc_info=exc,
        )

    # -- lifecycle -------------------------------------------------------

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._thread = threading.Thread(
            target=self._run, name="HeartWatchDBWriter", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = QUIT_TIMEOUT_S) -> None:
        """Flushes what's buffered and stops the thread, waiting at most
        `timeout` seconds. The thread is a daemon, so if it is stuck the
        process can still exit."""
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
        started_at: int | None = None,
    ) -> int:
        """`started_at` (epoch ms) defaults to now. The recording
        controller passes its own, so the time it filters samples by and
        the stored start are the same number."""
        if started_at is None:
            started_at = int(time.time() * 1000)
        payload = (device_id, label, accel_scale, gyro_scale, started_at)
        return self._submit_sync(_Kind.START_SESSION, payload)

    def end_session(
        self, session_id: int, ended_at: int | None = None, timeout: float = SYNC_TIMEOUT_S
    ) -> None:
        """`ended_at` (epoch ms) defaults to now. `timeout`: the quit path
        passes QUIT_TIMEOUT_S."""
        if ended_at is None:
            ended_at = int(time.time() * 1000)
        self._submit_sync(_Kind.END_SESSION, (session_id, ended_at), timeout)

    def close_orphan_sessions(self) -> int:
        """Closes every session a crash left unfinished (data/db.py's
        close_orphan_sessions()) and returns how many. Run once at startup,
        before anything starts recording."""
        return self._submit_sync(_Kind.CLOSE_ORPHANS, None)

    def delete_session(self, session_id: int) -> None:
        self._submit_sync(_Kind.DELETE_SESSION, session_id)

    def seed_demo_session(
        self, label: str | None = None, duration_s: float | None = None
    ) -> int:
        """Generates 2-3 minutes of synthetic IMU/HR data plus dummy activity
        predictions, and writes it through the same insert statements the live batch
        flush uses -- not a separate shortcut, just run synchronously as one
        control job so the session can be backdated to look like a real past
        recording. Returns the new session id."""
        label = label or seed_gen.random_label()
        duration_s = duration_s if duration_s is not None else random.uniform(120, 180)
        ended_at = int(time.time() * 1000)
        started_at = ended_at - int(duration_s * 1000)
        imu_rows = seed_gen.generate_imu_rows(started_at, duration_s)
        hr_rows = seed_gen.generate_hr_rows(started_at, duration_s)
        # Dummy activity predictions (until step 6's real TFLite model), so a
        # seeded session can reach the AI summary path -- without any, it
        # always fails data/stats.py's is_summarizable(). Same generator as
        # the Live Monitor; tagged model_version "demo-dummy". Remove along
        # with the other dummy generators in milestone 6.
        pred_rows = seed_gen.generate_prediction_rows(started_at, duration_s, label)
        payload = (label, started_at, ended_at, imu_rows, hr_rows, pred_rows)
        return self._submit_sync(_Kind.SEED_DEMO, payload)

    def flush_and_wait(self) -> None:
        """Blocks until every batch queued before this call has been committed.
        Used before reads (e.g. the developer panel's row counts) that must
        reflect data submitted moments earlier."""
        self._submit_sync(_Kind.FLUSH, None)

    def _submit_sync(self, kind: _Kind, payload: Any, timeout: float = SYNC_TIMEOUT_S) -> Any:
        """Queues a control job and waits at most `timeout` s for its result.
        Raises WriterFailed (failed state), WriterUnavailable (not running,
        dead, or no answer in time), or the job's own error."""
        failed = self.failed
        if failed is not None:
            raise WriterFailed(failed)
        if not self._started or self._thread is None or not self._thread.is_alive():
            raise WriterUnavailable("the database writer is not running")
        reply: "queue.Queue[tuple[bool, Any]]" = queue.Queue(maxsize=1)
        self._queue.put(_Job(kind, payload, reply))
        try:
            ok, result = reply.get(timeout=timeout)
        except queue.Empty:
            failed = self.failed
            if failed is not None:
                raise WriterFailed(failed) from None
            raise WriterUnavailable(
                f"the database writer didn't answer within {timeout:g} s"
            ) from None
        if not ok:
            raise result
        return result

    # -- worker thread -----------------------------------------------------

    def _run(self) -> None:
        try:
            conn = db_ops._connect(self._db_path)
        except Exception as exc:  # noqa: BLE001 - becomes the failed state
            self._fail(exc, "opening the database")
            conn = None
        pending: list[_Job] = []
        pending_rows = 0
        oldest_pending: float | None = None
        try:
            while True:
                try:
                    job = self._queue.get(timeout=_QUEUE_POLL_SECONDS)
                except queue.Empty:
                    job = None

                if self.failed is not None:
                    # Failed mode: answer, drop, and wait for stop().
                    pending, pending_rows, oldest_pending = [], 0, None
                    if job is None or job.kind in _BATCH_KINDS:
                        continue
                    if job.kind is _Kind.STOP:
                        break
                    if job.reply is not None:
                        job.reply.put((False, WriterFailed(self.failed)))
                    continue

                if job is not None:
                    if job.kind in _BATCH_KINDS:
                        pending.append(job)
                        pending_rows += len(job.payload[1])
                        if oldest_pending is None:
                            oldest_pending = time.monotonic()
                    elif job.kind is _Kind.STOP:
                        self._safe_flush(conn, pending)
                        pending, pending_rows, oldest_pending = [], 0, None
                        break
                    else:
                        # Control op or explicit flush: drain buffered samples
                        # first so ordering matches submission order, then run.
                        flushed = self._safe_flush(conn, pending)
                        pending, pending_rows, oldest_pending = [], 0, None
                        if flushed:
                            self._run_control(conn, job)
                        elif job.reply is not None:
                            job.reply.put((False, WriterFailed(self.failed)))

                due = pending and (
                    pending_rows >= FLUSH_MAX_ROWS
                    or (oldest_pending is not None
                        and time.monotonic() - oldest_pending >= FLUSH_MAX_SECONDS)
                )
                if due:
                    self._safe_flush(conn, pending)
                    pending, pending_rows, oldest_pending = [], 0, None
        finally:
            if conn is not None:
                conn.close()

    def _safe_flush(self, conn, jobs: list[_Job]) -> bool:
        """_flush, but an error fails the writer (logged once) instead of
        ending the thread. The failed batch is dropped. Returns whether it
        worked."""
        try:
            self._flush(conn, jobs)
        except Exception as exc:  # noqa: BLE001 - becomes the failed state
            self._fail(exc, "saving a batch of samples")
            return False
        return True

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
            elif job.kind is _Kind.CLOSE_ORPHANS:
                with conn:
                    result = db_ops.close_orphan_sessions(conn)
            elif job.kind is _Kind.SEED_DEMO:
                label, started_at, ended_at, imu_rows, hr_rows, pred_rows = job.payload
                with conn:
                    result = db_ops.start_session(
                        conn, "seed", label, 1.0, 1.0, started_at
                    )
                    db_ops.insert_imu_batch(conn, result, imu_rows)
                    db_ops.insert_hr_batch(conn, result, hr_rows)
                    db_ops._insert_prediction_rows(
                        conn,
                        [
                            (result, ts, predicted, confidence, seed_gen.DUMMY_MODEL_VERSION)
                            for ts, predicted, confidence in pred_rows
                        ],
                    )
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
