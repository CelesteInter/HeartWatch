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

# Lives in data/stats.py so the plain-text fallback formats durations the
# same way; re-exported here because the prompt is where it matters most.
from ..data.stats import format_duration_words, ordered_activities

__all__ = [
    "SYSTEM_PROMPT",
    "activity_fact_lines",
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
