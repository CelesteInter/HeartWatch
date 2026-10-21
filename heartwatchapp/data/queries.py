"""The one query layer behind every number the Dashboard and the Sessions
table show (and, from milestone 4, the AI bar).

Plain-language overview: a question like "average heart rate, while
Running, over the last 7 days" is written down as a MetricRequest. run()
fetches the rows that question needs and hands them to the metric
functions in data/stats.py, which do all the arithmetic. Because every
screen asks through here, and every calculation lives in stats.py, two
screens can't show different numbers for the same data.

The rule that makes that true:

    SQL may filter, order, join and COUNT(*) rows. Every number shown to
    the user is computed by a function in data/stats.py.

tests/test_queries.py scans this file, data/db.py and ui/ to enforce it.

Which sessions a range covers: a session belongs to a range if its
started_at falls in the range's half-open [start, end), and then all of its
samples count, so a session that runs past midnight counts on the day it
started. Sessions still recording (ended_at IS NULL) are left out of every
range-based number; see resolve_range() for the ranges themselves. The one
range that isn't a time window, "latest_session", is the single most
recently started finished session (see _load_sessions()).

Every function takes `conn` (an open connection to use) or `path` (a
database file to open); with neither, the app's database (db.DB_PATH, read
when the function is called, not when this module is imported).
"""

from __future__ import annotations

import contextlib
import datetime
import os
import sqlite3
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from ..ml.inference import CLASSES
from . import db, stats
from .seed import DUMMY_MODEL_VERSION

# Shown under activity times that come from the dummy classifier. The
# Dashboard (ui/dashboard.py) and the AI bar (ask/answers.py) both use it.
SIMULATED_CAPTION = "Activity times are from simulated predictions."

# metric name -> unit of its value.
METRIC_UNITS = {
    "avg_hr": "bpm",
    "peak_hr": "bpm",
    "time_in_activity": "seconds",
    "session_count": "sessions",
    "total_time": "seconds",
}
# Metrics that can be narrowed to one activity. A session or its total
# length isn't a single activity, so session_count and total_time can't.
ACTIVITY_METRICS = frozenset({"avg_hr", "peak_hr", "time_in_activity"})

# Calendar-day ranges: name -> how many calendar days, counting today.
_CALENDAR_DAY_RANGES = {"today": 1, "last_7_days": 7, "last_30_days": 30}
# Not a time window: the one most recently started finished session.
LATEST_SESSION = "latest_session"
# Every range a MetricRequest can name. The AI bar's parsers and its JSON
# schema read this tuple (ask/vocab.py), so they can't offer a range that
# resolve_range() doesn't know.
RANGE_NAMES = (
    "today", "yesterday", "this_week", "last_week", "last_7_days", "last_30_days", "all",
    LATEST_SESSION,
)
# "all": SQLite's whole INTEGER range, so every started_at falls inside it.
_ALL_RANGE = (-(2**63), 2**63 - 1)

# Rows are fetched with "session_id IN (?, ?, ...)" in chunks this size,
# well under SQLite's limit on query parameters.
_ID_CHUNK = 500


def _now_ms() -> int:
    return int(time.time() * 1000)


# --- connections ---------------------------------------------------------

@contextlib.contextmanager
def connection(
    conn: sqlite3.Connection | None = None, path: str | Path | None = None
) -> Iterator[sqlite3.Connection]:
    """`conn` itself if given (left open), otherwise a new connection to
    `path` (or db.DB_PATH) that is closed afterwards. The Dashboard opens one
    of these per refresh and passes it to every call."""
    if conn is not None:
        yield conn
        return
    with contextlib.closing(db._connect(path if path is not None else db.DB_PATH)) as new:
        yield new


# --- time ranges ---------------------------------------------------------

def local_timezone() -> datetime.tzinfo:
    """The system's local timezone, with its daylight-saving rules: the TZ
    environment variable if it names a zone, otherwise /etc/localtime.
    Falls back to the current fixed UTC offset only if neither works."""
    key = (os.environ.get("TZ") or "").lstrip(":")
    if key:
        with contextlib.suppress(ZoneInfoNotFoundError, ValueError):
            return ZoneInfo(key)
    with contextlib.suppress(OSError, ValueError):
        with open("/etc/localtime", "rb") as f:
            return ZoneInfo.from_file(f)
    return datetime.datetime.now().astimezone().tzinfo


def resolve_range(
    range_name: str, now_ms: int | None = None, tz: datetime.tzinfo | None = None
) -> tuple[int, int]:
    """(start_ms, end_ms) for a named range, half-open: [start, end).

    - "today": from local midnight to now.
    - "yesterday": the previous calendar day, midnight to midnight.
    - "this_week": from Monday 00:00 of the current ISO week to now.
    - "last_week": the previous ISO week, Monday 00:00 to Monday 00:00.
    - "last_7_days": today plus the 6 calendar days before it, from local
      midnight 6 days ago to now. "last_30_days" likewise (the Sessions
      view's other filter).
    - "all": everything.

    "latest_session" isn't a time window, so it raises ValueError here;
    run() and activity_breakdown() handle it in _load_sessions().

    Days are calendar days in `tz` (default: the system's local timezone),
    so a day that gains or loses an hour for daylight saving is still one
    day. A session belongs to the range if its started_at is in [start,
    end); a session starting at exactly local midnight belongs to the new
    day. `now_ms` defaults to the current time; tests pass both."""
    if range_name == "all":
        return _ALL_RANGE
    if range_name == LATEST_SESSION:
        raise ValueError(
            "latest_session isn't a time window: it's the most recent finished "
            "session, so it has no start and end to resolve."
        )
    if range_name not in RANGE_NAMES:
        raise ValueError(
            f"Unknown range {range_name!r}. Choose one of: {', '.join(RANGE_NAMES)}."
        )
    now_ms = _now_ms() if now_ms is None else now_ms
    tz = local_timezone() if tz is None else tz
    today = datetime.datetime.fromtimestamp(now_ms // 1000, tz).date()
    monday = today - datetime.timedelta(days=today.weekday())
    if range_name == "yesterday":
        return _midnight_ms(today - datetime.timedelta(days=1), tz), _midnight_ms(today, tz)
    if range_name == "this_week":
        return _midnight_ms(monday, tz), now_ms
    if range_name == "last_week":
        return _midnight_ms(monday - datetime.timedelta(days=7), tz), _midnight_ms(monday, tz)
    days = _CALENDAR_DAY_RANGES[range_name]
    return _midnight_ms(today - datetime.timedelta(days=days - 1), tz), now_ms


def _midnight_ms(day: datetime.date, tz: datetime.tzinfo) -> int:
    """Epoch ms of 00:00 on `day` in `tz`."""
    return round(datetime.datetime.combine(day, datetime.time(0), tzinfo=tz).timestamp() * 1000)


# --- the request format (the contract the AI bar will fill in) -----------

@dataclass(frozen=True)
class MetricRequest:
    """One question about the data, e.g. MetricRequest("avg_hr", "Running",
    "last_7_days"): average heart rate while Running over the last 7 days.
    Give either a range name or a session id, not both."""

    metric: str            # one of METRIC_UNITS
    activity: str | None   # one of ml.inference.CLASSES, or None for all activities
    range: str | None      # one of RANGE_NAMES; None when session_id is given
    session_id: int | None = None

    def validate(self) -> None:
        """Raises ValueError, in plain English, if this request can't be
        answered as written."""
        if self.metric not in METRIC_UNITS:
            raise ValueError(
                f"Unknown metric {self.metric!r}. Choose one of: {', '.join(METRIC_UNITS)}."
            )
        if self.activity is not None:
            if self.activity not in CLASSES:
                raise ValueError(
                    f"Unknown activity {self.activity!r}. Choose one of: "
                    f"{', '.join(CLASSES)}, or none for all activities."
                )
            if self.metric not in ACTIVITY_METRICS:
                raise ValueError(
                    f"{self.metric} can't be narrowed to one activity: a session isn't "
                    "a single activity. Leave the activity out."
                )
        if self.range is None and self.session_id is None:
            raise ValueError("Say which data to use: give a range or a session id.")
        if self.range is not None and self.session_id is not None:
            raise ValueError("Give a range or a session id, not both.")
        if self.range is not None and self.range not in RANGE_NAMES:
            raise ValueError(
                f"Unknown range {self.range!r}. Choose one of: {', '.join(RANGE_NAMES)}."
            )


@dataclass(frozen=True)
class MetricResult:
    """run()'s answer. `empty` is True when there was no data to answer
    from -- `value` is then None, never a stand-in zero. `n_sessions` is how
    many sessions fed the answer."""

    value: float | int | None
    unit: str
    n_sessions: int
    empty: bool
    # True if the answer used activity windows (an activity filter, or
    # time_in_activity) and any session that fed it has predictions from
    # the dummy classifier -- the same rule as ActivityBreakdown.simulated
    # (both use _any_simulated()).
    simulated: bool = False


@dataclass(frozen=True)
class ActivityBreakdown:
    """Time per activity, most time first (ties in first-appearance order).
    The Dashboard's bars and donut both draw `items`, so they can't
    disagree. `simulated` is True if any session that fed it has
    predictions from the dummy classifier (model_version "demo-dummy")."""

    items: tuple[tuple[str, float], ...]
    simulated: bool
    n_sessions: int

    def percentages(self) -> tuple[int, ...]:
        """Each item's share of the total, rounded on its own to a whole
        percent (Python's round()), in `items` order. The Dashboard's bar
        labels and the AI bar's breakdown reply both use this, so they
        can't disagree. Because each share is rounded separately the total
        can be 99 or 101."""
        total = sum(seconds for _, seconds in self.items)
        return tuple(round(100 * seconds / total) for _, seconds in self.items)


# --- rows ------------------------------------------------------------------

@dataclass
class _Session:
    id: int
    started_at: int
    end_ms: int            # ended_at, or now for a session still recording
    label: str | None
    hr: list[tuple[int, int]] = field(default_factory=list)
    predictions: list[tuple[int, str, float]] = field(default_factory=list)
    model_versions: set[str] = field(default_factory=set)

    @property
    def span(self) -> tuple[int, int]:
        return self.started_at, self.end_ms


def _any_simulated(sessions: Iterable[_Session]) -> bool:
    """True if any of `sessions` has predictions from the dummy classifier
    (model_version "demo-dummy"). Pass only the sessions that fed an
    answer. Shared by run() and activity_breakdown()."""
    return any(DUMMY_MODEL_VERSION in s.model_versions for s in sessions)


def _load_sessions(
    conn: sqlite3.Connection,
    *,
    range_name: str | None = None,
    session_id: int | None = None,
    with_predictions: bool = True,
    with_hr: bool = True,
    now_ms: int | None = None,
    tz: datetime.tzinfo | None = None,
) -> list[_Session]:
    """The sessions a request covers, oldest first, with their HR samples
    and predictions (each only if asked for). By range: finished sessions only. By
    id: that session, finished or not (a live one runs up to now, as in
    stats.compute_session_stats()); ValueError if it doesn't exist.
    "latest_session": the one finished session that started last (ties:
    the higher id), or none if there isn't one."""
    now_ms = _now_ms() if now_ms is None else now_ms
    if session_id is not None:
        rows = conn.execute(
            "SELECT id, started_at, ended_at, label FROM sessions WHERE id = ?", (session_id,)
        ).fetchall()
        if not rows:
            raise ValueError(f"no session with id {session_id}")
    elif range_name == LATEST_SESSION:
        rows = conn.execute(
            "SELECT id, started_at, ended_at, label FROM sessions "
            "WHERE ended_at IS NOT NULL ORDER BY started_at DESC, id DESC LIMIT 1"
        ).fetchall()
    else:
        start_ms, end_ms = resolve_range(range_name, now_ms, tz)
        rows = conn.execute(
            "SELECT id, started_at, ended_at, label FROM sessions "
            "WHERE ended_at IS NOT NULL AND started_at >= ? AND started_at < ? "
            "ORDER BY started_at, id",
            (start_ms, end_ms),
        ).fetchall()
    sessions = {
        r[0]: _Session(r[0], r[1], r[2] if r[2] is not None else now_ms, r[3]) for r in rows
    }
    _attach_rows(conn, sessions, with_predictions, with_hr)
    return list(sessions.values())


def _attach_rows(
    conn: sqlite3.Connection,
    sessions: dict[int, _Session],
    with_predictions: bool,
    with_hr: bool = True,
) -> None:
    ids = list(sessions)
    for i in range(0, len(ids), _ID_CHUNK):
        chunk = ids[i:i + _ID_CHUNK]
        marks = ",".join("?" * len(chunk))
        if with_hr:
            for sid, ts, bpm in conn.execute(
                f"SELECT session_id, ts, bpm FROM hr_samples WHERE session_id IN ({marks}) "
                "ORDER BY session_id, ts",
                chunk,
            ):
                sessions[sid].hr.append((ts, bpm))
        if not with_predictions:
            continue
        for sid, ts, predicted, confidence, model_version in conn.execute(
            "SELECT session_id, ts, predicted, confidence, model_version FROM predictions "
            f"WHERE session_id IN ({marks}) ORDER BY session_id, ts",
            chunk,
        ):
            sessions[sid].predictions.append((ts, predicted, confidence))
            sessions[sid].model_versions.add(model_version)


def _hr_samples(session: _Session, activity: str | None) -> list[tuple[int, int]]:
    if activity is None:
        return session.hr
    return stats.hr_samples_during(
        session.hr, session.predictions, session.started_at, session.end_ms, activity
    )


# --- the public entry points ----------------------------------------------

def run(
    request: MetricRequest,
    conn: sqlite3.Connection | None = None,
    path: str | Path | None = None,
    *,
    now_ms: int | None = None,
    tz: datetime.tzinfo | None = None,
) -> MetricResult:
    """Answers one MetricRequest. The only way the Dashboard gets its
    numbers. Raises ValueError for an invalid request or an unknown session
    id.

    - avg_hr / peak_hr: over every HR sample of the covered sessions (only
      those during `activity`, if given), pooled -- see stats.average_hr().
      Empty if there are no such samples.
    - time_in_activity: seconds credited to `activity` (or to any activity,
      if None), worked out per session and then added up. Empty if no
      covered session has any activity data; a real 0 if they do but none
      of it was `activity`.

    `n_sessions` counts the sessions that contributed to the number: with
    an activity, only those with HR samples inside its windows (avg_hr,
    peak_hr) or with time in it (time_in_activity -- so a real 0 has 0
    sessions); without one, every covered session with the data needed.
    - session_count / total_time: the covered sessions, and their combined
      length. Empty if there are none.

    `simulated` is set only when activity windows were used (an activity
    filter, or time_in_activity), from the sessions that fed the answer."""
    request.validate()
    metric, activity = request.metric, request.activity
    unit = METRIC_UNITS[metric]
    needs_predictions = metric == "time_in_activity" or activity is not None
    with connection(conn, path) as c:
        sessions = _load_sessions(
            c,
            range_name=request.range,
            session_id=request.session_id,
            with_predictions=needs_predictions,
            with_hr=metric in ("avg_hr", "peak_hr"),
            now_ms=now_ms,
            tz=tz,
        )

    if metric in ("session_count", "total_time"):
        if not sessions:
            return MetricResult(None, unit, 0, empty=True)
        spans = [s.span for s in sessions]
        value = stats.session_count(spans) if metric == "session_count" else (
            stats.total_recorded_seconds(spans)
        )
        return MetricResult(value, unit, len(sessions), empty=False)

    if metric in ("avg_hr", "peak_hr"):
        fed = [(s, samples) for s in sessions if (samples := _hr_samples(s, activity))]
        pooled = [sample for _, samples in fed for sample in samples]
        if metric == "avg_hr":
            value = stats.average_hr(pooled)
        else:
            peak = stats.peak_hr(pooled)
            value = peak[0] if peak is not None else None
        return MetricResult(
            value, unit, len(fed), empty=value is None,
            simulated=_any_simulated(s for s, _ in fed),
        )

    # time_in_activity
    fed = [
        (s, seconds)
        for s in sessions
        if (seconds := stats.time_per_activity(s.predictions, s.started_at, s.end_ms))
    ]
    if not fed:
        return MetricResult(None, unit, 0, empty=True)
    combined = stats.combine_activity_seconds(seconds for _, seconds in fed)
    if activity is None:
        value, n_sessions = sum(combined.values()), len(fed)
    else:
        # Only sessions with some time in this activity contributed.
        value = combined.get(activity, 0.0)
        n_sessions = sum(1 for _, seconds in fed if seconds.get(activity, 0.0) > 0)
    return MetricResult(
        value, unit, n_sessions, empty=False, simulated=_any_simulated(s for s, _ in fed)
    )


def activity_breakdown(
    range_name: str | None = None,
    session_id: int | None = None,
    conn: sqlite3.Connection | None = None,
    path: str | Path | None = None,
    *,
    now_ms: int | None = None,
    tz: datetime.tzinfo | None = None,
) -> ActivityBreakdown:
    """Time per activity over a range or one session, for the Dashboard's
    bars and donut. Each session is worked out on its own
    (stats.time_per_activity()), then the sessions are added up and ordered
    by stats.ordered_activities(). Empty `items` means no activity data."""
    MetricRequest("time_in_activity", None, range_name, session_id).validate()
    with connection(conn, path) as c:
        sessions = _load_sessions(
            c, range_name=range_name, session_id=session_id, now_ms=now_ms, tz=tz
        )
    fed = []
    for s in sessions:
        seconds = stats.time_per_activity(s.predictions, s.started_at, s.end_ms)
        if seconds:
            fed.append((s, seconds))
    combined = stats.combine_activity_seconds(seconds for _, seconds in fed)
    return ActivityBreakdown(
        items=tuple(stats.ordered_activities(combined)),
        simulated=_any_simulated(s for s, _ in fed),
        n_sessions=len(fed),
    )


def list_recent_sessions(
    n: int = 3,
    conn: sqlite3.Connection | None = None,
    path: str | Path | None = None,
) -> list[dict]:
    """The `n` newest finished sessions, newest first, for the Dashboard's
    "Recent sessions" list. Each dict has: id, started_at, ended_at,
    duration_s, label, peak_bpm (None if the session has no HR samples)."""
    with connection(conn, path) as c:
        rows = c.execute(
            "SELECT id, started_at, ended_at, label FROM sessions "
            "WHERE ended_at IS NOT NULL ORDER BY started_at DESC, id DESC LIMIT ?",
            (n,),
        ).fetchall()
        sessions = {r[0]: _Session(r[0], r[1], r[2], r[3]) for r in rows}
        _attach_rows(c, sessions, with_predictions=False)
    out = []
    for s in sessions.values():
        peak = stats.peak_hr(s.hr)
        out.append(
            {
                "id": s.id,
                "started_at": s.started_at,
                "ended_at": s.end_ms,
                "duration_s": stats.session_duration_s(s.started_at, s.end_ms),
                "label": s.label,
                "peak_bpm": peak[0] if peak is not None else None,
            }
        )
    return out


def list_sessions(
    limit: int = 50,
    activity_filter: str | None = None,
    range_name: str = "all",
    conn: sqlite3.Connection | None = None,
    path: str | Path | None = None,
    *,
    now_ms: int | None = None,
    tz: datetime.tzinfo | None = None,
) -> list[dict]:
    """Backs the Sessions view's table, newest first. Unlike the Dashboard
    it includes a session that's still recording (its duration counts up
    to now). `range_name` uses resolve_range(), so "Last 7 days" means the
    same here as on the Dashboard. `activity_filter` matches the session's
    own label. Each dict has: id, started_at, ended_at, duration_s,
    device_id, label, sample_count (IMU rows), avg_bpm, peak_bpm (both None
    if the session has no HR samples)."""
    now_ms = _now_ms() if now_ms is None else now_ms
    start_ms, end_ms = resolve_range(range_name, now_ms, tz)
    with connection(conn, path) as c:
        rows = c.execute(
            "SELECT s.id, s.started_at, s.ended_at, s.device_id, s.label, "
            "(SELECT COUNT(*) FROM imu_samples i WHERE i.session_id = s.id) "
            "FROM sessions s "
            "WHERE (? IS NULL OR s.label = ?) AND s.started_at >= ? AND s.started_at < ? "
            "ORDER BY s.started_at DESC, s.id DESC LIMIT ?",
            (activity_filter, activity_filter, start_ms, end_ms, limit),
        ).fetchall()
        sessions = {
            r[0]: _Session(r[0], r[1], r[2] if r[2] is not None else now_ms, r[4]) for r in rows
        }
        _attach_rows(c, sessions, with_predictions=False)
    out = []
    for r in rows:
        s = sessions[r[0]]
        peak = stats.peak_hr(s.hr)
        out.append(
            {
                "id": s.id,
                "started_at": s.started_at,
                "ended_at": r[2],
                "duration_s": stats.session_duration_s(s.started_at, s.end_ms),
                "device_id": r[3],
                "label": s.label,
                "sample_count": r[5],
                "avg_bpm": stats.average_hr(s.hr),
                "peak_bpm": peak[0] if peak is not None else None,
            }
        )
    return out
