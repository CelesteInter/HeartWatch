"""Every word list the AI bar uses, in one place.

Plain-language overview: the metrics, activities and time ranges a question
can name come straight from the query layer (data/queries.py) and the
activity classifier (ml/inference.py), so the keyword parser, the AI
model's JSON schema and the code that answers can never disagree about
what exists. The synonym tables below are what the keyword parser
(ask/rules.py) and the evaluation (scripts/eval_ask.py) match against.

Every phrase is written lower-case with single spaces, in the same form
normalize() turns a question into, so "Heart-Rate" in a question matches
"heart rate" here.
"""

from __future__ import annotations

import re

from ..data.queries import LATEST_SESSION, METRIC_UNITS, RANGE_NAMES
from ..ml.inference import CLASSES

METRICS = tuple(METRIC_UNITS)
ACTIVITIES = tuple(CLASSES)
RANGES = RANGE_NAMES
# Used when a question names no time period -- the same as the Dashboard.
DEFAULT_RANGE = "last_7_days"

ACTIVITY_SYNONYMS: dict[str, tuple[str, ...]] = {
    "Sitting": ("sit", "sits", "sitting", "sat", "seated"),
    "Walking": ("walk", "walks", "walking", "walked", "stroll", "strolls", "strolling", "strolled"),
    "Standing": ("stand", "stands", "standing", "stood"),
    "Running": ("run", "runs", "running", "ran", "jog", "jogs", "jogging", "jogged"),
}

# Words that name heart rate itself. avg_hr and peak_hr need one of these
# plus a modifier below ("average ... heart rate").
HR_NOUNS = ("heart rate", "heartrate", "hr", "bpm", "pulse")
HR_MODIFIERS: dict[str, tuple[str, ...]] = {
    "avg_hr": ("average", "avg", "mean"),
    "peak_hr": ("peak", "max", "maximum", "highest", "top", "how high"),
}
# Phrases that name a metric on their own.
METRIC_PHRASES: dict[str, tuple[str, ...]] = {
    "time_in_activity": (
        "how long", "how much time", "how many minutes", "how many hours", "minutes",
        "hours", "time spent", "spent", "spend",
    ),
    "session_count": (
        "how many sessions", "how many recordings", "number of sessions",
        "number of recordings", "session count", "count my sessions",
    ),
    "total_time": (
        "total time", "total recorded time", "total recording time", "time recorded",
        "recorded time", "recording time", "how long did i record",
        "how long have i recorded", "how long have i been recording",
        "how much time did i record", "how much time have i recorded",
    ),
}
# An explicit request to count sessions of one activity ("how many walking
# sessions"). v1 can't answer it, but it's unambiguous, so the keyword
# parser hands it straight to the fixed "not by activity yet" reply
# instead of to the AI. A bare "sessions" elsewhere in a question doesn't
# count (see ask/rules.py).
SESSION_COUNT_BY_ACTIVITY_PHRASES = tuple(
    f"{prefix} {word} {noun}"
    for prefix in ("how many", "number of")
    for words in ACTIVITY_SYNONYMS.values()
    for word in words
    for noun in ("sessions", "recordings")
)

# Words that mean "a session". An activity together with one of these
# ("how many running sessions") isn't something v1 can answer, so the
# keyword parser defers it -- unless the word is part of a range phrase
# ("my last session").
SESSION_WORDS = ("session", "sessions", "recording", "recordings")

_LATEST_NOUNS = ("session", "recording", "run", "walk", "jog")
RANGE_SYNONYMS: dict[str, tuple[str, ...]] = {
    "today": ("today", "so far today"),
    "yesterday": ("yesterday",),
    "this_week": ("this week", "so far this week", "this calendar week"),
    "last_week": ("last week", "previous week", "last calendar week"),
    "last_7_days": ("last 7 days", "past 7 days", "last seven days", "past seven days"),
    "last_30_days": ("last 30 days", "past 30 days", "last thirty days", "past thirty days"),
    "all": (
        "all time", "ever", "all sessions", "all my sessions", "all of my sessions",
        "every session", "across all sessions",
    ),
    # "my last run" is read as the latest session, Running (a v1
    # limitation, see the README) -- the activity word still counts.
    LATEST_SESSION: tuple(
        f"{prefix} {noun}" for prefix in ("last", "latest", "most recent") for noun in _LATEST_NOUNS
    ),
}
# Time phrases v1 can't map to exactly one range. The keyword parser defers
# any question containing one (a longer phrase wins, so "the last week"
# is ambiguous even though it contains "last week").
AMBIGUOUS_RANGE_PHRASES = (
    "the last week", "the past week", "past week", "this past week", "recent", "recently",
    "lately", "this month", "last month", "past month", "the last month", "this year",
    "last year", "tonight", "this morning", "this afternoon", "this evening", "last night",
    "tomorrow", "next week", "last time", "weekend", "monday", "tuesday", "wednesday",
    "thursday", "friday", "saturday", "sunday", "january", "february", "march", "april",
    "may", "june", "july", "august", "september", "october", "november", "december",
)

# Things v1 can't do at all -- charts, commands, data the watch doesn't
# record (README, "Known limitations"). Any of these makes the keyword
# parser answer "unsupported" straight away, even next to metric words
# ("show me a chart of my heart rate"), instead of handing the question to
# the AI, which would answer the metric and drop the rest.
UNSUPPORTED_WORDS = (
    "chart", "charts", "graph", "graphs", "plot", "plots", "draw", "start", "stop", "pause",
    "delete", "export", "step", "steps", "calorie", "calories", "sleep", "weather",
    "distance", "pace", "speed", "km", "kilometers", "miles",
)

_NON_WORD = re.compile(r"[^a-z0-9']+")


def normalize(text: str) -> str:
    """Lower-case, curly apostrophes made straight, everything that isn't
    a letter, digit or apostrophe turned into one space."""
    text = text.lower().replace("’", "'")
    return _NON_WORD.sub(" ", text).strip()


def phrase_pattern(phrase: str) -> re.Pattern[str]:
    """Matches `phrase` as whole words in normalize()d text."""
    return re.compile(r"(?<![a-z0-9'])" + re.escape(phrase).replace(r"\ ", " ") + r"(?![a-z0-9'])")


def all_synonym_phrases() -> list[str]:
    """Every phrase above that a supported question may contain. The
    medical guard must never match any of them (tests/test_ask_vocab.py)."""
    phrases = [p for words in ACTIVITY_SYNONYMS.values() for p in words]
    phrases += list(HR_NOUNS)
    phrases += [p for words in HR_MODIFIERS.values() for p in words]
    phrases += [p for words in METRIC_PHRASES.values() for p in words]
    phrases += list(SESSION_COUNT_BY_ACTIVITY_PHRASES)
    phrases += list(SESSION_WORDS)
    phrases += [p for words in RANGE_SYNONYMS.values() for p in words]
    return phrases
