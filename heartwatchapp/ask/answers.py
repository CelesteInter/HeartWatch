"""Every sentence the AI bar shows, written by Python.

Plain-language overview: the answer to a question is built here from the
checked request and run()'s result -- the model never writes any of it.
Each answer has up to four parts: the reply ("Your heart rate averaged 128
bpm while running this week, across 3 sessions."), the "what I
understood" line ("Average heart rate · Running · this week"), built from
the request so a misunderstanding is visible straight away (and naming any
typo that was corrected), the simulated-data caption when activity times
come from the dummy classifier, and -- for replies that can't answer --
up to three clickable example questions (ask/suggest.py).

Wording rules (handoff 9a, 3.3): activities read as verbs where natural
("you walked", "you ran") and lower-case inside sentences (the understood
line keeps the labels); the number of sessions is mentioned only when it
adds something ("in one session", "across 3 sessions", never for the
latest session); a real zero ("You didn't spend any time walking") is
worded differently from no data ("You didn't record any activity"); and
no reply judges or compares. Heart rates are rounded the same way as the
Dashboard's cards; durations use the session summaries'
format_duration_words(); percentages use ActivityBreakdown.percentages(),
the Dashboard's own rounding.
"""

from __future__ import annotations

import datetime

from ..data.queries import (
    LATEST_SESSION,
    SIMULATED_CAPTION,
    ActivityBreakdown,
    MetricRequest,
    MetricResult,
)
from ..data.stats import format_duration_words
from . import suggest, vocab

MEDICAL_REPLY = (
    "I can't judge health or give medical advice. I can tell you numbers from your recorded "
    "sessions: average or peak heart rate, time in each activity, number of sessions, or "
    "total recorded time."
)
UNSUPPORTED_REPLY = (
    "That isn't something I can answer from your recorded sessions. Try one of these:"
)
SESSION_COUNT_BY_ACTIVITY_REPLY = (
    "I can count all your sessions, but not sessions by activity yet. Try one of these:"
)
NOT_UNDERSTOOD_REPLY = "Sorry, I didn't catch that. Try one of these:"
# Questions the keyword parser answers on its own (tests/test_ask_suggest.py
# checks that it does), offered when the AI model isn't available and as
# the default suggestions.
RULES_EXAMPLES = suggest.DEFAULT_QUESTIONS
AI_UNAVAILABLE_REPLY = (
    "I'd need the AI model for that one, and it isn't running right now. These work without it:"
)
THINKING_TEXT = "Thinking…"
NO_SESSIONS_REPLY = "You haven't recorded any sessions yet."
COMPARISON_REPLY = "I can't compare two periods yet. You can ask about each one on its own:"
LONGEST_REPLY = "I can't find your longest session yet. Try one of these:"


def follow_up_reply(example: str) -> str:
    """For "and walking?": the bar has no memory of earlier questions."""
    return (
        "I can't follow up on earlier questions yet. Try asking the whole question, "
        f"like '{example}'."
    )


def _or_list(words: list[str]) -> str:
    *rest, last = words
    return f"{', '.join(rest)} or {last}" if rest else last


def needs_activity_reply() -> str:
    """"Which activity do you mean: sitting, walking, standing or running?"
    -- built from the classifier's own labels."""
    return f"Which activity do you mean: {_or_list([a.lower() for a in vocab.ACTIVITIES])}?"


def clarify_reply(labels: list[str]) -> str:
    """"Did you mean walking or standing?" for two (or more) readings."""
    return f"Did you mean {_or_list([label.lower() for label in labels])}?"


METRIC_LABELS = {
    "avg_hr": "Average heart rate",
    "peak_hr": "Peak heart rate",
    "time_in_activity": "Time in activity",
    "session_count": "Number of sessions",
    "total_time": "Total recorded time",
}
SUMMARY_LABEL = "Session summary"
BREAKDOWN_LABEL = "Activity breakdown"
# How a range reads inside a sentence.
RANGE_PHRASES = {
    "today": "today",
    "yesterday": "yesterday",
    "this_week": "this week",
    "last_week": "last week",
    "last_7_days": "in the last 7 days",
    "last_30_days": "in the last 30 days",
    "all": "across all your sessions",
    LATEST_SESSION: "in your latest session",
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
    LATEST_SESSION: "latest session",
}
# How the breakdown reply begins, per range.
BREAKDOWN_LEADS = {
    "today": "Today",
    "yesterday": "Yesterday",
    "this_week": "This week",
    "last_week": "Last week",
    "last_7_days": "Over the last 7 days",
    "last_30_days": "Over the last 30 days",
    "all": "Across all your sessions",
    LATEST_SESSION: "In your latest session",
}
# "You walked for ...": each activity's past tense.
ACTIVITY_PAST = {"Sitting": "sat", "Walking": "walked", "Standing": "stood", "Running": "ran"}


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


def summary_understood_line() -> str:
    """The session summary is always of the latest session (B5): "Session
    summary · latest session", even when the question said "my last run"."""
    return f"{SUMMARY_LABEL} · {RANGE_LABELS[LATEST_SESSION]}"


def breakdown_understood_line(range_name: str, range_defaulted: bool = False,
                              summary_fallback: bool = False) -> str:
    """e.g. "Activity breakdown · this week". When a summary was asked for
    over several sessions, says that a summary covers one session."""
    label = RANGE_LABELS[range_name]
    line = f"{BREAKDOWN_LABEL} · {label} (default)" if range_defaulted else (
        f"{BREAKDOWN_LABEL} · {label}"
    )
    if summary_fallback:
        line += " · a summary covers one session, so here's the breakdown"
    return line


def with_corrections(line: str | None, corrections) -> str | None:
    """The understood line, followed by any typo corrections:
    "... · corrected: walkign → walking"."""
    if not corrections:
        return line
    fixed = "corrected: " + ", ".join(f"{a} → {b}" for a, b in corrections)
    return f"{line} · {fixed}" if line else fixed


def _over(range_name: str, n: int) -> str:
    """The range, plus the number of sessions when it adds something."""
    if range_name == "all":
        if n == 1:
            return "in your one recorded session"
        return f"across all {n} of your sessions" if n else RANGE_PHRASES["all"]
    when = RANGE_PHRASES[range_name]
    if n == 1:
        return f"{when}, in one session"
    return f"{when}, across {n} sessions" if n else when


def _empty_text(metric: str, activity: str | None, range_name: str) -> str:
    during = f" while {activity.lower()}" if activity else ""
    when = RANGE_PHRASES[range_name]
    if metric in ("avg_hr", "peak_hr"):
        if range_name == LATEST_SESSION:
            return f"There's no heart-rate data{during} in your latest session."
        if range_name == "all":
            return f"You haven't recorded any heart-rate data{during} yet."
        return f"You didn't record any heart-rate data{during} {when}."
    if metric == "time_in_activity":
        if range_name == LATEST_SESSION:
            return "There's no activity data in your latest session."
        if range_name == "all":
            return "You haven't recorded any activity yet."
        return f"You didn't record any activity {when}."
    if range_name in ("all", LATEST_SESSION):
        return NO_SESSIONS_REPLY
    return f"You didn't record any sessions {when}."


def reply_text(request: MetricRequest, result: MetricResult) -> str:
    """The sentence for a computed answer, or for no data."""
    metric, activity, range_name = request.metric, request.activity, request.range
    if result.empty:
        return _empty_text(metric, activity, range_name)

    latest = range_name == LATEST_SESSION
    during = f" while {activity.lower()}" if activity else ""
    over = _over(range_name, result.n_sessions)
    if metric in ("avg_hr", "peak_hr"):
        verb = "averaged" if metric == "avg_hr" else "peaked at"
        if latest:
            return f"In your latest session, your heart rate {verb} {result.value:.0f} bpm{during}."
        return f"Your heart rate {verb} {result.value:.0f} bpm{during} {over}."
    if metric == "time_in_activity":
        if result.value == 0:
            if latest:
                return f"In your latest session, you didn't spend any time {activity.lower()}."
            return f"You didn't spend any time {activity.lower()} {RANGE_PHRASES[range_name]}."
        spent = f"{ACTIVITY_PAST[activity]} for {format_duration_words(result.value)}"
        if latest:
            return f"In your latest session, you {spent}."
        return f"You {spent} {over}."
    if metric == "session_count":
        when = "in total" if range_name == "all" else RANGE_PHRASES[range_name]
        return f"You recorded {_plural(result.value, 'session')} {when}."
    duration = format_duration_words(result.value)
    if latest:
        return f"Your latest session lasted {duration}."
    return f"You recorded {duration} in total {over}."


def breakdown_text(range_name: str, breakdown: ActivityBreakdown) -> str:
    """Total activity time, then each activity, most time first, with its
    share. Same items, order and percentages as the Dashboard's card."""
    if not breakdown.items:
        return _empty_text("time_in_activity", None, range_name)
    lead = BREAKDOWN_LEADS[range_name]
    total = format_duration_words(sum(seconds for _, seconds in breakdown.items))
    if len(breakdown.items) == 1:
        return f"{lead} you recorded {total} of activity, all of it {breakdown.items[0][0].lower()}."
    parts = ", ".join(
        f"{label.lower()} {format_duration_words(seconds)} ({pct}%)"
        for (label, seconds), pct in zip(breakdown.items, breakdown.percentages())
    )
    return f"{lead} you recorded {total} of activity: {parts}."


def _clock(dt: datetime.datetime) -> str:
    hour = dt.hour % 12 or 12
    return f"{hour}:{dt.minute:02d} {'AM' if dt.hour < 12 else 'PM'}"


def summary_lead(started_at_ms: int, tz: datetime.tzinfo) -> str:
    """"Your latest session started at 7:30 AM on Monday, October 12." in
    local time."""
    dt = datetime.datetime.fromtimestamp(started_at_ms / 1000, tz)
    return f"Your latest session started at {_clock(dt)} on {dt:%A}, {dt:%B} {dt.day}."


def summary_text(lead: str, plain: str, caveat: str | None) -> str:
    """The lead-in, then format_stats_plain()'s summary unchanged, then the
    low-confidence caveat if there is one -- as the Sessions view shows it."""
    return " ".join(part for part in (lead, plain, caveat) if part)


def caption(result: MetricResult | ActivityBreakdown) -> str | None:
    """The Dashboard's simulated-predictions caption, by the same rule
    (`simulated` is set only when activity windows were used)."""
    return SIMULATED_CAPTION if result.simulated else None
