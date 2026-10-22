"""Session-summary number crunching.

Plain-language overview: this file turns one session's raw SQLite rows
into a small handful of finished facts -- how long it lasted, average
and peak heart rate, how much time was spent in each detected activity,
which activity the peak heart rate happened during, and how many activity
readings the classifier itself was unsure about. Nothing in this file
talks to an LLM. The facts computed here are shown to the user directly
as readable sentences (format_stats_plain()) -- by default, and whenever
the AI summary is unavailable -- and are the exact facts handed to the
LLM to phrase when HEARTWATCH_SUMMARY_MODE=llm (see config.py). All arithmetic on a session's rows -- and
every relationship between numbers, like "the peak happened during
Walking" -- is worked out in this file and nowhere else, so the model
never has to (and never gets the chance to) invent one.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from pathlib import Path

from ..ml.inference import CLASSES
from . import db

# A classifier prediction below this confidence is flagged as "low
# confidence". The app shows that as its own fixed caveat line (see
# confidence_caveat() below) -- it is never handed to the LLM, which can't
# tell "the classifier was unsure" apart from "the user did something".
LOW_CONFIDENCE_THRESHOLD = 0.6

# Minimum amount of data before a session is worth sending to the LLM at
# all. With less than this the model has almost nothing to describe, and a
# small model fills the gap by padding and inventing things. A session
# below ANY one of these gets plain numbers only (see is_summarizable()).
# Tune these by hand.
MIN_PREDICTIONS = 1       # at least one activity-classifier reading
MIN_HR_SAMPLES = 60       # heart-rate readings (~1 per second from the MAX30101)
MIN_DURATION_S = 60.0     # seconds

# Each classifier prediction describes one PREDICTION_WINDOW_MS-long window
# of sensor data. PREDICTION_TIMESTAMP_MARKS says which edge of that window
# the prediction's timestamp is: "end" (every generator in the app --
# data/seed.py, data/writer.py, ui/live_monitor.py, scripts/demo_session.py
# -- writes the first prediction at start + 5 s, then every 5 s) or
# "start". See _activity_segments() for exactly which time each prediction
# is credited with.
PREDICTION_WINDOW_MS = 5000
PREDICTION_TIMESTAMP_MARKS = "end"   # "end" (current convention) or "start"

# Canonical spelling of each activity label, looked up case-insensitively,
# so "walking" from some future classifier build still reads "Walking".
_CANONICAL_LABELS = {label.lower(): label for label in CLASSES}


def compute_session_stats(session_id: int, path: str | Path = db.DB_PATH) -> dict:
    """Reads one session's rows out of SQLite and reduces them to the
    facts a summary needs. Raises ValueError if the session id doesn't
    exist. `path` defaults to the app's real database; tests pass a
    fixture path instead.

    Returns a dict with: session_id, label, duration_s, avg_hr, max_hr
    (None if there are no HR samples), hr_sample_count, activity_seconds
    ({label: seconds} for every activity with at least one prediction, in
    the order each first appeared in the session; empty if there are none
    yet), peak_hr_activity (the activity the peak
    heart rate happened during, or None if that can't be known),
    prediction_count, low_confidence_count.
    """
    timeline = db.get_session_timeline(session_id, path)
    session = timeline["session"]
    if session is None:
        raise ValueError(f"no session with id {session_id}")

    started_at = session["started_at"]
    ended_at = session["ended_at"]
    now_ms = int(time.time() * 1000)
    session_end_ms = ended_at if ended_at is not None else now_ms
    duration_s = session_duration_s(started_at, session_end_ms)

    hr = timeline["hr"]
    avg_hr = average_hr(hr)
    peak = peak_hr(hr)
    max_hr = peak[0] if peak is not None else None

    predictions = timeline["predictions"]
    segments = _activity_segments(predictions, started_at, session_end_ms)
    activity_seconds = _seconds_by_label(segments)

    peak_hr_activity = _activity_at(segments, peak[1]) if peak is not None else None

    low_confidence_count = sum(
        1 for _, _, confidence in predictions if confidence < LOW_CONFIDENCE_THRESHOLD
    )

    return {
        "session_id": session_id,
        "label": session.get("label"),
        "duration_s": duration_s,
        "avg_hr": avg_hr,
        "max_hr": max_hr,
        "hr_sample_count": len(hr),
        "activity_seconds": activity_seconds,
        "peak_hr_activity": peak_hr_activity,
        "prediction_count": len(predictions),
        "low_confidence_count": low_confidence_count,
    }


# --- metric functions -------------------------------------------------
#
# Every number the user reads -- in the session summary, on the Dashboard,
# in the Sessions table, and later from the AI bar -- is worked out by one
# of these functions, and only here. data/queries.py fetches the rows and
# hands them over; SQL never does the arithmetic. Pure functions: no
# database access, so they're easy to test on hand-built rows.


def average_hr(samples: Iterable[tuple[int, int]]) -> float | None:
    """Average heart rate in bpm, from (ts, bpm) samples. None if there are
    no samples.

    The rule: the simple mean of the measured samples -- add up every bpm
    reading and divide by how many there are.

    - Only measured samples count. A gap in the readings (a sensor
      dropout, say) is left out: it is never filled in, and the samples on
      either side of it don't count for any longer because of it.
    - Across several sessions, the samples are pooled: every sample counts
      once, whichever session it came from, so a long session weighs more
      than a short one. That's the same answer SQL's AVG(bpm) gives over
      the same rows."""
    values = [bpm for _, bpm in samples]
    return sum(values) / len(values) if values else None


def peak_hr(samples: Iterable[tuple[int, int]]) -> tuple[int, int] | None:
    """(bpm, ts) of the highest heart-rate sample, or None if there are no
    samples. If the peak value repeats, the earliest sample that hit it."""
    peak = None
    for ts, bpm in samples:
        if peak is None or bpm > peak[0]:
            peak = (bpm, ts)
    return peak


def time_per_activity(
    predictions: list[tuple[int, str, float]], session_start_ms: int, session_end_ms: int
) -> dict[str, float]:
    """{label: seconds} for one session, in the order each activity first
    appeared; empty if no time is credited to any activity. Built on
    _activity_segments(), the one implementation of the prediction window
    rule. Takes the session's start as well as its end, because the rule
    clips every window to the session."""
    return _seconds_by_label(_activity_segments(predictions, session_start_ms, session_end_ms))


def hr_samples_during(
    samples: Iterable[tuple[int, int]],
    predictions: list[tuple[int, str, float]],
    session_start_ms: int,
    session_end_ms: int,
    activity: str,
) -> list[tuple[int, int]]:
    """One session's (ts, bpm) samples that fall in time credited to
    `activity` (same pieces as time_per_activity(), start inclusive, end
    exclusive)."""
    segments = _activity_segments(predictions, session_start_ms, session_end_ms)
    return [(ts, bpm) for ts, bpm in samples if _activity_at(segments, ts) == activity]


def combine_activity_seconds(per_session: Iterable[dict[str, float]]) -> dict[str, float]:
    """Adds up several sessions' time_per_activity() results. Each session
    is worked out on its own first, so no prediction window ever reaches
    across from one session into another. Pass sessions oldest first: the
    combined dict keeps first-appearance order, which ordered_activities()
    uses to break ties."""
    combined: dict[str, float] = {}
    for activity_seconds in per_session:
        for label, seconds in activity_seconds.items():
            combined[label] = combined.get(label, 0.0) + seconds
    return combined


def session_duration_s(started_at_ms: int, ended_at_ms: int) -> float:
    """A session's length in seconds (never negative)."""
    return max(0, ended_at_ms - started_at_ms) / 1000


def session_count(sessions: Iterable[tuple[int, int]]) -> int:
    """How many sessions there are, from (started_at_ms, ended_at_ms) pairs."""
    return sum(1 for _ in sessions)


def total_recorded_seconds(sessions: Iterable[tuple[int, int]]) -> float:
    """Total recorded time in seconds over (started_at_ms, ended_at_ms)
    pairs: each session's full length, whether or not any of it was
    credited to an activity."""
    return sum(session_duration_s(start, end) for start, end in sessions)


def _seconds_by_label(segments: list[tuple[int, int, str | None]]) -> dict[str, float]:
    """{label: seconds} from _activity_segments() pieces, skipping unlabeled
    ones, in the order each label first appears."""
    activity_seconds: dict[str, float] = {}
    for start_ms, end_ms, label in segments:
        if label is not None:
            activity_seconds[label] = activity_seconds.get(label, 0.0) + (end_ms - start_ms) / 1000
    return activity_seconds


def _canonical_label(label: str | None) -> str | None:
    """Canonical capitalization for a known activity label; unknown labels
    are kept as written; an empty label counts as no label."""
    if not label:
        return None
    return _CANONICAL_LABELS.get(label.lower(), label)


def _activity_segments(
    predictions: list[tuple[int, str, float]], session_start_ms: int, session_end_ms: int
) -> list[tuple[int, int, str | None]]:
    """Splits the session's timeline into (start_ms, end_ms, label) pieces:
    the stretch of time each classifier prediction describes. Every piece
    is half-open, [start_ms, end_ms), so any instant belongs to at most one
    prediction. With W = PREDICTION_WINDOW_MS:

    - "end" convention (a prediction's timestamp t is the END of its
      window): it owns [max(t_prev, t - W), t), where t_prev is the
      previous prediction's timestamp. The first one owns [t - W, t).
      E.g. the prediction at 215 s owns [210 s, 215 s), so a heart-rate
      sample at exactly 210 s belongs to it, not to the window ending at
      210 s. Nothing after the last prediction is attributed.
    - "start" convention: it owns [t, min(t_next, t + W)); the last one
      owns [t, t + W).

    No prediction owns more than W, so if a prediction is missing, the
    time it would have covered belongs to no activity (instead of being
    stretched onto a neighbour). Every piece is clipped to the session's
    [start, end]. Neighbouring pieces with the same label that touch are
    merged into one.

    Both time-per-activity and the peak-HR activity are read off these same
    pieces, so the two facts can never disagree with each other."""
    window = PREDICTION_WINDOW_MS
    if PREDICTION_TIMESTAMP_MARKS not in ("end", "start"):
        raise ValueError(f"unknown PREDICTION_TIMESTAMP_MARKS {PREDICTION_TIMESTAMP_MARKS!r}")

    segments: list[tuple[int, int, str | None]] = []
    for i, (ts, predicted, _confidence) in enumerate(predictions):
        if PREDICTION_TIMESTAMP_MARKS == "end":
            start_ms = ts - window if i == 0 else max(predictions[i - 1][0], ts - window)
            end_ms = ts
        else:
            start_ms = ts
            is_last = i + 1 == len(predictions)
            end_ms = ts + window if is_last else min(predictions[i + 1][0], ts + window)
        start_ms = max(start_ms, session_start_ms)
        end_ms = min(end_ms, session_end_ms)
        if end_ms <= start_ms:
            continue
        label = _canonical_label(predicted)
        if segments and segments[-1][2] == label and segments[-1][1] == start_ms:
            segments[-1] = (segments[-1][0], end_ms, label)
        else:
            segments.append((start_ms, end_ms, label))
    return segments


def _activity_at(segments: list[tuple[int, int, str | None]], ts: int) -> str | None:
    """The label of the piece containing `ts` (start inclusive, end
    exclusive, for every piece). None if `ts` falls outside every piece or
    in an unlabeled one."""
    for start_ms, end_ms, label in segments:
        if start_ms <= ts < end_ms:
            return label
    return None


def ordered_activities(activity_seconds: dict[str, float]) -> list[tuple[str, float]]:
    """(label, seconds) pairs, most time first. Ties keep the dict's own
    order, which for compute_session_stats() output is the order each
    activity first appeared in the session -- so an equal split lists
    whatever the user did first, first. The one ordering helper for both
    the LLM prompt (llm/prompts.py's activity_fact_lines()) and
    format_stats_plain(), so the two can never list activities
    differently."""
    # sorted() is stable: equal times stay in first-appearance order.
    return sorted(activity_seconds.items(), key=lambda kv: -kv[1])


def format_duration_words(seconds: float) -> str:
    """Writes a length of time the way it should read in a sentence, e.g.
    252.4 -> "4 minutes 12 seconds", 60 -> "1 minute", 3725 -> "1 hour 2
    minutes 5 seconds". Rounded to the nearest whole second. The LLM prompt
    uses this so the model copies these words as-is and has nothing to
    convert, and format_stats_plain() uses it so both read the same."""
    total = int(round(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    parts = [
        f"{value} {unit}{'' if value == 1 else 's'}"
        for value, unit in ((hours, "hour"), (minutes, "minute"), (secs, "second"))
        if value
    ]
    return " ".join(parts) if parts else "0 seconds"


def is_summarizable(stats: dict) -> bool:
    """True if a session has enough data to be worth asking the LLM about.

    False if ANY of these is true: no activity-classifier predictions at
    all, fewer than MIN_HR_SAMPLES heart-rate readings, or shorter than
    MIN_DURATION_S seconds. Values exactly at a threshold pass. When this
    is False the app skips the LLM entirely and shows plain numbers.
    """
    return (
        (stats.get("prediction_count") or 0) >= MIN_PREDICTIONS
        and (stats.get("hr_sample_count") or 0) >= MIN_HR_SAMPLES
        and stats["duration_s"] >= MIN_DURATION_S
    )


def confidence_caveat(stats: dict) -> str | None:
    """A fixed, app-written sentence noting that the activity classifier
    was unsure during part of the session, or None if it never was. Shown
    by the UI underneath the summary -- whether that summary came from the
    LLM or from format_stats_plain() -- and never passed to the LLM."""
    if stats.get("low_confidence_count"):
        return "Some windows had low classification confidence."
    return None


def format_stats_plain(stats: dict) -> str:
    """Plain-language, non-AI rendering of compute_session_stats()'s
    output, in short readable sentences, e.g.:

        The session lasted 6 minutes 20 seconds. Average heart rate was
        93 bpm and peak heart rate was 111 bpm, reached during Walking.
        Time by activity: Walking 2 minutes 30 seconds, Sitting 2 minutes
        10 seconds, Standing 1 minute 40 seconds.

    Activities are listed most time first (see ordered_activities()). A
    session with exactly one predicted activity says "Activity: Walking
    for the whole session." instead of a one-item list.

    This is what the user sees by default (HEARTWATCH_SUMMARY_MODE=plain)
    and, in llm mode, whenever there's no AI summary -- Ollama
    unavailable, too little data, or the generated text was rejected -- so
    it says everything a passing AI summary could. Any fact that's missing
    (no HR samples, no predictions, peak activity unknown) is left out
    cleanly."""
    sentences = [f"The session lasted {format_duration_words(stats['duration_s'])}."]

    if stats.get("avg_hr") is not None and stats.get("max_hr") is not None:
        hr_sentence = (
            f"Average heart rate was {stats['avg_hr']:.0f} bpm and peak heart "
            f"rate was {stats['max_hr']:.0f} bpm"
        )
        if stats.get("peak_hr_activity"):
            hr_sentence += f", reached during {stats['peak_hr_activity']}"
        sentences.append(hr_sentence + ".")
    else:
        sentences.append("No heart-rate samples were recorded.")

    activities = ordered_activities(stats.get("activity_seconds") or {})
    if len(activities) == 1:
        sentences.append(f"Activity: {activities[0][0]} for the whole session.")
    elif activities:
        breakdown = ", ".join(
            f"{label} {format_duration_words(seconds)}" for label, seconds in activities
        )
        sentences.append(f"Time by activity: {breakdown}.")
    elif stats.get("label"):
        sentences.append(f"Logged activity: {stats['label']}.")

    return " ".join(sentences)
