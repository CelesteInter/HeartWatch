"""Prompt text for the local LLM, kept separate from client.py so wording
can be tuned by anyone on the team without touching code that talks to
Ollama.

Plain-language overview: the model is only ever asked to turn facts
Python has already worked out into a few sentences of plain English. It
is never asked to compute anything, never asked to connect one number to
another (Python already did -- e.g. "Peak heart rate occurred during:
Walking"), and never asked for medical or health advice. Those rules are
spelled out in SYSTEM_PROMPT below because that's the text that actually
reaches the model.

The rules in SYSTEM_PROMPT are only a first line of defense: a 1B model
will not reliably follow them. What actually enforces them is
llm/validate.py, which checks every reply before it is shown. Three things
here exist to make that checking fair:

- Every number is written out exactly the way it should appear in the
  summary (e.g. "4 minutes 12 seconds", not "252 seconds"), so the model
  only ever has to copy numbers, never convert or round them.
- The instruction text itself contains no digits, so the set of numbers
  in the user prompt is exactly the set of session numbers the model is
  allowed to repeat (validate.py's number-grounding check relies on this).
- Facts are split into session-level lines and one line per activity
  (session_fact_lines() / activity_fact_lines()), and validate.py reads
  those same functions to check that each number stays attached to the
  activity it was given with.
"""

from __future__ import annotations

import json

from ..ask import vocab as _ask_vocab

# Lives in data/stats.py so the plain-text fallback formats durations the
# same way; re-exported here because the prompt is where it matters most.
from ..data.stats import format_duration_words, ordered_activities

__all__ = [
    "ASK_SYSTEM_PROMPT",
    "SYSTEM_PROMPT",
    "activity_fact_lines",
    "build_ask_messages",
    "build_session_summary_messages",
    "format_duration_words",
    "render_session_summary_prompt",
    "session_fact_lines",
]

SYSTEM_PROMPT = (
    "You are a plain-language narrator for a fitness-tracking app called "
    "HeartWatch. You will be given a short list of facts that the app has "
    "already worked out from one recorded session -- you do not calculate "
    "anything yourself, and you never see raw sensor data. Summarize the "
    "facts you are given in a few short, plain-English sentences.\n"
    "Rules:\n"
    "- Begin directly with the first sentence of the summary. No preamble, "
    "no greeting, no restating the task, no markdown headers, no bullet "
    "points.\n"
    "- Mention the time spent in each activity, using the exact wording "
    "given.\n"
    "- Use only the numbers provided, written exactly as given. Never "
    "invent, convert, or round a number, activity, or time.\n"
    "- Keep each number with the activity it is listed with. Only say "
    "which activity the peak heart rate happened during if that fact is "
    "given, and then name exactly that activity.\n"
    "- The numbers are exact values from the database. Do not hedge them "
    "with words like around, about, or approximately.\n"
    "- Mention each number once. Do not repeat the same figure as if it "
    "were two separate findings.\n"
    "- Do not refer to anything that is not in the input: no earlier "
    "sessions, no usual or typical values, no comparisons.\n"
    "- Do not give medical or health advice, do not diagnose or interpret "
    "anything, and do not say whether the numbers are healthy, unhealthy, "
    "good, or bad -- just describe what happened."
)


def session_fact_lines(stats: dict) -> list[str]:
    """The facts about the session as a whole: length, average and peak
    heart rate, and (if known) the activity the peak happened during.
    Each line is written exactly as it should appear in prose."""
    lines = [f"Session length: {format_duration_words(stats['duration_s'])}"]
    if stats.get("avg_hr") is not None:
        lines.append(f"Average heart rate: {stats['avg_hr']:.0f} bpm")
    if stats.get("max_hr") is not None:
        lines.append(f"Peak heart rate: {stats['max_hr']:.0f} bpm")
        if stats.get("peak_hr_activity"):
            lines.append(f"Peak heart rate occurred during: {stats['peak_hr_activity']}")
    return lines


def activity_fact_lines(stats: dict) -> dict[str, str]:
    """{activity label: its fact line}, one per activity with at least one
    prediction, most time first -- the same order format_stats_plain()
    uses (both go through data/stats.py's ordered_activities())."""
    return {
        label: f"Time {label}: {format_duration_words(seconds)}"
        for label, seconds in ordered_activities(stats.get("activity_seconds") or {})
    }


def render_session_summary_prompt(stats: dict) -> str:
    """Turns compute_session_stats()'s output (see data/stats.py) into the
    text the model reads: one short fact per line, e.g.

        Session length: 6 minutes 20 seconds
        Average heart rate: 93 bpm
        Peak heart rate: 111 bpm
        Peak heart rate occurred during: Walking
        Time Walking: 3 minutes 10 seconds
        Time Sitting: 1 minute 10 seconds

    Every number in the returned text comes from a fact line and is one the
    model is allowed to use; llm/validate.py rejects any reply containing a
    number not in here."""
    lines = session_fact_lines(stats) + list(activity_fact_lines(stats).values())

    # Deliberately NOT included: the classifier's low-confidence count.
    # A 1B model has no way to understand "the activity classifier was
    # unsure" and has been seen describing it as something the user's body
    # did. The app shows that caveat itself instead -- see
    # data/stats.py's confidence_caveat() and ui/sessions.py.

    lines.append(
        "Summarize these facts in a few short plain sentences, including the "
        "time spent in each activity."
    )
    return "\n".join(lines)


def build_session_summary_messages(stats: dict) -> list[dict]:
    """The exact chat messages sent to the model for one session summary:
    SYSTEM_PROMPT, then render_session_summary_prompt(stats). Shared by
    llm/worker.py and scripts/eval_summaries.py, so the evaluation measures
    the same request the app makes."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": render_session_summary_prompt(stats)},
    ]


# -- the AI bar's question parser ---------------------------------------------
#
# A different job from the summaries above: the model reads one question
# and fills in a JSON form (ask/llm_parser.py passes the schema as Ollama's
# `format=`). It never answers, and nothing it writes is shown -- Python
# checks the form and writes the reply (ask/answers.py). The field lists
# below are built from ask/vocab.py, the same source as the schema, so the
# prompt can't describe a value the schema doesn't allow.

ASK_METRIC_DESCRIPTIONS = {
    "avg_hr": "average heart rate",
    "peak_hr": "highest (peak) heart rate",
    "time_in_activity": "how long the user spent doing one activity",
    "session_count": "how many sessions were recorded",
    "total_time": "total recorded time",
}
ASK_RANGE_DESCRIPTIONS = {
    "today": "today",
    "yesterday": "yesterday",
    "this_week": "this week: since Monday of the current week",
    "last_week": "last week: the previous Monday-to-Sunday week",
    "last_7_days": "the last 7 days, counting today",
    "last_30_days": "the last 30 days, counting today",
    "all": "all sessions ever recorded",
    "latest_session": "the user's most recent session only",
}
# (question, intent, metric, activity, range) -- includes the this_week /
# last_week / last_7_days contrast the model got wrong in the Phase A
# smoke test ("this week" came back as last_week). Every activity and range
# in an example is named literally in its question (handoff 9a, 3.9): the
# v1 examples that weren't ("the week before this one", "newest
# recording") taught the model to fill in values the question doesn't
# give. None of these may repeat or closely paraphrase a dev question
# (eval/ask_questions_dev*.jsonl): tests/test_ask_vocab.py fails above 0.6
# word overlap.
ASK_EXAMPLES = (
    ("avg pulse for this week so far", "query", "avg_hr", None, "this_week"),
    ("peak heart rate today", "query", "peak_hr", None, "today"),
    ("mean bpm for last week", "query", "avg_hr", None, "last_week"),
    ("average bpm across the past 7 days", "query", "avg_hr", None, "last_7_days"),
    ("walking duration yesterday", "query", "time_in_activity", "Walking", "yesterday"),
    ("max bpm while jogging in my latest recording", "query", "peak_hr", "Running",
     "latest_session"),
    ("count of recorded sessions", "query", "session_count", None, None),
    ("number of sessions this week", "query", "session_count", None, "this_week"),
    ("recap of my latest recording", "session_summary", None, None, "latest_session"),
    ("how was my time split between activities yesterday", "activity_breakdown", None, None,
     "yesterday"),
    ("how much time have I recorded this month so far", "unsupported", None, None, None),
    ("tell me a joke", "unsupported", None, None, None),
)


def _ask_example_line(question: str, intent: str, metric, activity, range_name) -> str:
    form = {
        "intent": intent,
        "metric": metric,
        "activity": activity,
        "range": range_name,
    }
    return f"{question} -> {json.dumps(form)}"


ASK_SYSTEM_PROMPT = (
    "You turn one question about the user's own recorded sessions into a JSON form. "
    "You never answer the question yourself.\n\n"
    "Fields:\n"
    '- intent: "query" if the question asks for one of the metrics below; '
    '"session_summary" if it asks for an overview of one single recorded session; '
    '"activity_breakdown" if it asks what the user did, or how their time was split '
    'between activities, over a time period; otherwise "unsupported".\n'
    '- metric (null unless intent is "query"):\n'
    + "".join(f"  {m}: {ASK_METRIC_DESCRIPTIONS[m]}\n" for m in _ask_vocab.METRICS)
    + f"- activity: one of {', '.join(_ask_vocab.ACTIVITIES)}, only if the question itself "
    "names that activity (e.g. walk, run, jog, sit, stand). Most questions name none: then "
    "activity is null. Never guess one.\n"
    "- range: the time period the question names, or null if it names none:\n"
    + "".join(f"  {r}: {ASK_RANGE_DESCRIPTIONS[r]}\n" for r in _ask_vocab.RANGES)
    + "\nExamples:\n"
    + "".join(_ask_example_line(*example) + "\n" for example in ASK_EXAMPLES)
)


def build_ask_messages(question: str) -> list[dict]:
    """The exact chat messages for one question: ASK_SYSTEM_PROMPT, then
    the question as typed. Shared by the app and scripts/eval_ask.py."""
    return [
        {"role": "system", "content": ASK_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
