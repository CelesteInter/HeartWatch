"""Clickable example questions for replies that can't answer (handoff 9a, 3.7).

Plain-language overview: when the bar can't give a number -- the question
isn't supported, wasn't understood, needs an activity, asks for something
not built yet, found no data, or the AI model isn't running -- it offers
up to three questions the person can click instead. They're chosen from
what was understood: an activity the question named gives questions about
that activity; a time period gives questions about that period; otherwise
a fixed trio. Nothing is random.

Every question built here is answered by the keyword parser alone, with no
AI model (tests/test_ask_suggest.py checks every one this module can
produce), so a suggestion always works, even when Ollama isn't running.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from ..data.queries import LATEST_SESSION, MetricRequest
from . import vocab
from .parse_result import Option

MAX_SUGGESTIONS = 3

# How each range reads at the end of a suggested question. Each is one of
# vocab.RANGE_SYNONYMS's phrases, so the keyword parser reads it back.
RANGE_WORDS = {
    "today": "today",
    "yesterday": "yesterday",
    "this_week": "this week",
    "last_week": "last week",
    "last_7_days": "in the last 7 days",
    "last_30_days": "in the last 30 days",
    "all": "ever",
    LATEST_SESSION: "in my latest session",
}
# The present-tense verb for "how long did I ...".
ACTIVITY_VERBS = {"Sitting": "sit", "Walking": "walk", "Standing": "stand", "Running": "run"}
DEFAULT_RANGE = "this_week"
DEFAULT_QUESTIONS = (
    "average heart rate this week",
    "what did I do today",
    "tell me about my last session",
)


def _when(range_name: str | None) -> str:
    return f" {RANGE_WORDS[range_name]}" if range_name else ""


def metric_question(metric: str, activity: str | None, range_name: str | None) -> str:
    """A question the keyword parser reads as exactly this request (time in
    an activity with no activity becomes the breakdown question)."""
    when = _when(range_name)
    during = f" while {activity.lower()}" if activity else ""
    if metric == "avg_hr":
        return f"average heart rate{during}{when}"
    if metric == "peak_hr":
        return f"peak heart rate{during}{when}"
    if metric == "time_in_activity":
        if activity is None:
            return f"what did I do{when}"
        return f"how long did I {ACTIVITY_VERBS[activity]}{when}"
    if metric == "session_count":
        if range_name == LATEST_SESSION:
            return "tell me about my last session"
        return f"how many sessions{when}"
    return f"total recorded time{when}"


def _activity_questions(activity: str, range_name: str) -> list[str]:
    return [
        metric_question("time_in_activity", activity, range_name),
        metric_question("avg_hr", activity, range_name),
        metric_question("peak_hr", activity, range_name),
    ]


def _range_questions(range_name: str) -> list[str]:
    if range_name == LATEST_SESSION:
        return ["tell me about my last session", metric_question("avg_hr", None, range_name),
                metric_question("time_in_activity", None, range_name)]
    return [metric_question("avg_hr", None, range_name),
            metric_question("time_in_activity", None, range_name),
            metric_question("session_count", None, range_name)]


def _options(questions: Iterable[str]) -> tuple[Option, ...]:
    out: list[str] = []
    for q in questions:
        if q not in out:
            out.append(q)
    return tuple(Option(q, q) for q in out[:MAX_SUGGESTIONS])


def from_detected(activities: Sequence[str] = (), ranges: Sequence[str] = ()) -> tuple[Option, ...]:
    """Suggestions from what a question named: its activity (with its
    range, or this week), else its range, else the default trio."""
    if activities:
        return _options(_activity_questions(activities[0], ranges[0] if ranges else DEFAULT_RANGE))
    if ranges:
        return _options(_range_questions(ranges[0]))
    return _options(DEFAULT_QUESTIONS)


def for_comparison(metrics: Sequence[str], activities: Sequence[str],
                   ranges: Sequence[str]) -> tuple[Option, ...]:
    """One question per period the comparison named (the first two), for
    the metric it named, or the breakdown if it named none or several."""
    if len(ranges) < 2:
        return from_detected(activities, ranges)
    metric = metrics[0] if len(metrics) == 1 else "time_in_activity"
    activity = activities[0] if len(activities) == 1 else None
    if metric in ("session_count", "total_time"):
        activity = None
    return _options(metric_question(metric, activity, r) for r in ranges[:2])


def for_no_data(request: MetricRequest) -> tuple[Option, ...]:
    """The same question over longer periods, then the session summary."""
    wider = [r for r in ("last_30_days", "all") if r != request.range]
    activity = request.activity
    questions = [metric_question(request.metric, activity, r) for r in wider]
    return _options([*questions, "tell me about my last session"])


def which_activity(range_name: str | None, activities: Sequence[str] = vocab.ACTIVITIES
                   ) -> tuple[Option, ...]:
    """The "Which activity?" buttons: the time-in-activity question again,
    once per activity, labelled with the activity."""
    return tuple(
        Option(a, metric_question("time_in_activity", a, range_name)) for a in activities
    )


def every_question() -> list[str]:
    """Every question this module can produce (for the test that each one
    is answered by the keyword parser alone)."""
    ranges = [None, *vocab.RANGES]
    questions = list(DEFAULT_QUESTIONS)
    for metric in vocab.METRICS:
        for range_name in ranges:
            for activity in (None, *vocab.ACTIVITIES):
                if activity is not None and metric in ("session_count", "total_time"):
                    continue
                questions.append(metric_question(metric, activity, range_name))
    for range_name in vocab.RANGES:
        questions += _range_questions(range_name)
        for activity in vocab.ACTIVITIES:
            questions += _activity_questions(activity, range_name)
    return sorted(set(questions))
