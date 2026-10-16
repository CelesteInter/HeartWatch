"""Every sentence the AI bar shows, written by Python.

Plain-language overview: the answer to a question is built here from the
checked request and run()'s result -- the model never writes any of it.
Each answer has three parts: the reply ("Your average heart rate while
Running this week was 128 bpm (3 sessions)."), the "what I understood"
line ("Average heart rate · Running · this week"), built from the request
so a misunderstanding is visible straight away, and the simulated-data
caption when activity times come from the dummy classifier. Heart rates
are rounded the same way as the Dashboard's cards; durations use the
session summaries' format_duration_words(). Missing data is said in words,
never shown as 0.
"""

from __future__ import annotations

from ..data.queries import SIMULATED_CAPTION, MetricRequest, MetricResult
from ..data.stats import format_duration_words
from . import vocab

MEDICAL_REPLY = (
    "I can't judge health or give medical advice. I can tell you numbers from your recorded "
    "sessions: average or peak heart rate, time in each activity, number of sessions, or "
    "total recorded time."
)
UNSUPPORTED_REPLY = (
    "I can only answer questions about your recorded sessions, like 'average heart rate this "
    "week' or 'how long did I walk yesterday?'"
)
SESSION_COUNT_BY_ACTIVITY_REPLY = (
    "I can count all your sessions, but not sessions by activity yet."
)
NOT_UNDERSTOOD_REPLY = (
    "Sorry, I didn't understand that. Try something like 'peak heart rate last week'."
)
# Questions the keyword parser answers on its own (tests/test_ask_pipeline.py
# checks that it does), offered when the AI model isn't available.
RULES_EXAMPLES = (
    "average heart rate this week",
    "how long did I walk yesterday",
    "how many sessions in the last 30 days",
)
AI_UNAVAILABLE_REPLY = (
    "I couldn't understand that without the AI model, which isn't running right now. "
    "Questions like these work without it: "
    + "; ".join(f"'{q}'" for q in RULES_EXAMPLES)
    + "."
)
THINKING_TEXT = "Thinking…"


def needs_activity_reply() -> str:
    """"Which activity? I can answer for Sitting, Walking, Standing or
    Running." -- built from the classifier's own labels."""
    *rest, last = vocab.ACTIVITIES
    return f"Which activity? I can answer for {', '.join(rest)} or {last}."


METRIC_LABELS = {
    "avg_hr": "Average heart rate",
    "peak_hr": "Peak heart rate",
    "time_in_activity": "Time in activity",
    "session_count": "Number of sessions",
    "total_time": "Total recorded time",
}
# How a range reads inside a sentence.
RANGE_PHRASES = {
    "today": "today",
    "yesterday": "yesterday",
    "this_week": "this week",
    "last_week": "last week",
    "last_7_days": "in the last 7 days",
    "last_30_days": "in the last 30 days",
    "all": "across all sessions",
    "latest_session": "in your latest session",
}
# How a range reads in the "what I understood" line.
RANGE_LABELS = {
    "today": "today",
    "yesterday": "yesterday",
    "this_week": "this week",
    "last_week": "last week",
    "last_7_days": "last 7 days",
    "last_30_days": "last 30 days",
    "all": "all sessions",
    "latest_session": "latest session",
}


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def understood_line(request: MetricRequest, range_defaulted: bool = False) -> str:
    """e.g. "Average heart rate · Running · this week". Heart-rate metrics
    without an activity say "All activities"; a range the question didn't
    name is marked "(default)"."""
    parts = [METRIC_LABELS[request.metric]]
    if request.activity is not None:
        parts.append(request.activity)
    elif request.metric in ("avg_hr", "peak_hr"):
        parts.append("All activities")
    label = RANGE_LABELS[request.range]
    parts.append(f"{label} (default)" if range_defaulted else label)
    return " · ".join(parts)


def reply_text(request: MetricRequest, result: MetricResult) -> str:
    """The sentence for a computed answer, or for no data."""
    metric, activity, range_name = request.metric, request.activity, request.range
    during = f" while {activity}" if activity else ""
    when = RANGE_PHRASES[range_name]
    # Only the sessions that fed the number are counted; a real 0 (activity
    # data, but none of this activity) has none, so no count is shown.
    sessions = f" ({_plural(result.n_sessions, 'session')})" if result.n_sessions else ""

    if result.empty:
        if metric in ("avg_hr", "peak_hr"):
            return f"No heart-rate data recorded{during} {when}."
        if metric == "time_in_activity":
            return f"No activity data recorded {when}."
        if range_name in ("all", "latest_session"):
            return "No sessions recorded yet."
        return f"No sessions recorded {when}."

    if metric == "avg_hr":
        return f"Your average heart rate{during} {when} was {result.value:.0f} bpm{sessions}."
    if metric == "peak_hr":
        return f"Your peak heart rate{during} {when} was {result.value:.0f} bpm{sessions}."
    if metric == "time_in_activity":
        return f"Time spent {activity} {when}: {format_duration_words(result.value)}{sessions}."
    if metric == "session_count":
        count_when = "in total" if range_name == "all" else when
        return f"You recorded {_plural(result.value, 'session')} {count_when}."
    return f"Total recorded time {when}: {format_duration_words(result.value)}{sessions}."


def caption(result: MetricResult) -> str | None:
    """The Dashboard's simulated-predictions caption, by the same rule
    (MetricResult.simulated is set only when activity windows were used)."""
    return SIMULATED_CAPTION if result.simulated else None
