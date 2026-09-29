"""Session-summary number crunching.

Plain-language overview: this file turns one session's raw SQLite rows
into a small handful of finished facts -- how long it lasted, average
and peak heart rate, how much time was spent in each detected activity,
which activity the peak heart rate happened during, and how many activity
readings the classifier itself was unsure about. Nothing in this file
talks to an LLM. The facts computed here are shown to the user directly
as readable sentences when the AI summary is unavailable
(format_stats_plain()), and are the exact facts handed to the LLM to
phrase when it is available. All arithmetic on a session's rows -- and
every relationship between numbers, like "the peak happened during
Walking" -- is worked out in this file and nowhere else, so the model
never has to (and never gets the chance to) invent one.
"""

from __future__ import annotations

import time
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
    ({label: seconds} for every activity with at least one prediction,
    empty if there are none yet), peak_hr_activity (the activity the peak
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
    duration_s = max(0, session_end_ms - started_at) / 1000

    hr = timeline["hr"]
    hr_values = [bpm for _, bpm in hr]
    avg_hr = sum(hr_values) / len(hr_values) if hr_values else None
    max_hr = max(hr_values) if hr_values else None

    predictions = timeline["predictions"]
    segments = _activity_segments(predictions, session_end_ms)
    activity_seconds: dict[str, float] = {}
    for start_ms, end_ms, label in segments:
        if label is not None:
            activity_seconds[label] = activity_seconds.get(label, 0.0) + (end_ms - start_ms) / 1000

    peak_hr_activity = None
    if hr:
        # The first sample that hit the peak, if the peak value repeats.
        peak_ts = next(ts for ts, bpm in hr if bpm == max_hr)
        peak_hr_activity = _activity_at(segments, peak_ts)

    low_confidence_count = sum(
        1 for _, _, confidence in predictions if confidence < LOW_CONFIDENCE_THRESHOLD
    )

    return {
        "session_id": session_id,
        "label": session.get("label"),
        "duration_s": duration_s,
        "avg_hr": avg_hr,
        "max_hr": max_hr,
        "hr_sample_count": len(hr_values),
        "activity_seconds": activity_seconds,
        "peak_hr_activity": peak_hr_activity,
        "prediction_count": len(predictions),
        "low_confidence_count": low_confidence_count,
    }


def _canonical_label(label: str | None) -> str | None:
    """Canonical capitalization for a known activity label; unknown labels
    are kept as written; an empty label counts as no label."""
    if not label:
        return None
    return _CANONICAL_LABELS.get(label.lower(), label)


def _activity_segments(
    predictions: list[tuple[int, str, float]], session_end_ms: int
) -> list[tuple[int, int, str | None]]:
    """Splits the session's timeline into (start_ms, end_ms, label) pieces,
    one per classifier prediction. Each prediction row is a window ending
    at `ts`; the time from one window's end to the next window's end (or
    the session's end, for the last window) is attributed to the earlier
    window's label, so a run of "Walking" predictions adds up to however
    long the app was actually calling it Walking.

    Both time-per-activity and the peak-HR activity are read off these same
    pieces, so the two facts can never disagree with each other. Time
    before the first prediction belongs to no activity."""
    segments = []
    for i, (ts, predicted, _confidence) in enumerate(predictions):
        next_ts = predictions[i + 1][0] if i + 1 < len(predictions) else session_end_ms
        segments.append((ts, max(ts, next_ts), _canonical_label(predicted)))
    return segments


def _activity_at(segments: list[tuple[int, int, str | None]], ts: int) -> str | None:
    """The label of the piece containing `ts` (start inclusive, end
    exclusive -- except the very last piece, which includes the session's
    end). None if `ts` falls outside every piece or in an unlabeled one."""
    for i, (start_ms, end_ms, label) in enumerate(segments):
        is_last = i == len(segments) - 1
        if start_ms <= ts < end_ms or (is_last and ts == end_ms):
            return label
    return None


def ordered_activities(activity_seconds: dict[str, float]) -> list[tuple[str, float]]:
    """(label, seconds) pairs in the codebase's canonical order -- Sitting,
    Walking, Standing, Running -- with any unknown label after them. Used
    by both the LLM prompt and format_stats_plain() so they list activities
    the same way."""
    order = {label: i for i, label in enumerate(CLASSES)}
    return sorted(activity_seconds.items(), key=lambda kv: (order.get(kv[0], len(order)), kv[0]))


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
        Time by activity: Sitting 2 minutes 10 seconds, Walking 2 minutes
        30 seconds, Standing 1 minute 40 seconds.

    This is what the user sees whenever there's no AI summary -- Ollama
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
    if activities:
        breakdown = ", ".join(
            f"{label} {format_duration_words(seconds)}" for label, seconds in activities
        )
        sentences.append(f"Time by activity: {breakdown}.")
    elif stats.get("label"):
        sentences.append(f"Logged activity: {stats['label']}.")

    return " ".join(sentences)
