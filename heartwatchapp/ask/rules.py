"""The keyword parser: answers simple questions with no AI at all.

Plain-language overview: the question is matched, whole words only,
against the phrase tables in ask/vocab.py. The parser only accepts when it
is sure -- exactly one metric, at most one activity, at most one time
range, and every other word is a known filler word ("what", "was", "my",
...). Anything else is "deferred" (in hybrid mode the AI model gets it), and
a question that names no metric and no activity at all, or asks for
something v1 can't do (a chart, a command, steps, a distance filter --
vocab.UNSUPPORTED_WORDS and the command phrases), is "unsupported".
A wrong answer is much worse than passing the question on, so when in
doubt it defers.

A longer phrase wins over a shorter one inside it: "how long did I record"
is total_time, not "how long" (time in an activity); "the last week" is
ambiguous, not "last week"; "my last session" is a range, so its "session"
doesn't count as a sessions word. Activity words always count, so "my last
run" is the latest session, Running, and "how many walking sessions" is a
session count with an activity (which pipeline.normalize() answers with
the fixed "not by activity yet" reply).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..data.queries import LATEST_SESSION, MetricRequest
from . import vocab
from .parse_result import (
    ACCEPTED,
    CLARIFY,
    DEFERRED,
    INTENT_BREAKDOWN,
    INTENT_SUMMARY,
    NOT_BUILT,
    NOT_BUILT_COMPARISON,
    NOT_BUILT_FOLLOW_UP,
    NOT_BUILT_LONGEST,
    UNSUPPORTED,
    Option,
    ParseResult,
)

# Words that carry no meaning of their own in a supported question. Any
# other word that isn't part of a matched phrase makes the parser defer.
FILLER_WORDS = frozenset(
    """
    what what's whats which was were is are am my me i i've ive i'd did do does doing have has
    had been how the a an in on during while for of over across at from to show tell give get
    got go went up reach reached hit find see check please can could would you so far record
    recorded logged
    """.split()
)

# "last 3 days", "past two weeks", ... -- a count v1 has no range for. The
# ranges it does have ("last 7 days", "last 30 days") are left to
# vocab.RANGE_SYNONYMS.
_UNSUPPORTED_COUNT_RANGE = re.compile(
    r"(?<![a-z0-9'])(?:last|past|previous|next) (?!(?:7|30|seven|thirty) days?(?![a-z0-9']))"
    r"(?:\d+|a few|few|a couple of|couple of|two|three|four|five|six|eight|nine|ten|"
    r"fourteen|twenty) (?:days?|weeks?|months?|years?|hours?|minutes?)(?![a-z0-9'])"
)

_TOKEN = re.compile(r"[a-z0-9']+")


@dataclass(frozen=True)
class _Match:
    kind: str    # range | ambiguous | metric | modifier | hr | activity | session
    value: str   # the range key, metric or activity it names ("" for hr/session/ambiguous)
    start: int
    end: int

    def inside(self, other: _Match) -> bool:
        """True if this match lies within a strictly longer `other`."""
        return (
            other.start <= self.start and self.end <= other.end
            and other.end - other.start > self.end - self.start
        )


def _table() -> list[tuple[str, str, re.Pattern[str]]]:
    rows = []
    for key, phrases in vocab.RANGE_SYNONYMS.items():
        rows += [("range", key, vocab.phrase_pattern(p)) for p in phrases]
    rows += [("ambiguous", "", vocab.phrase_pattern(p)) for p in vocab.AMBIGUOUS_RANGE_PHRASES]
    rows.append(("ambiguous", "", _UNSUPPORTED_COUNT_RANGE))
    for metric, phrases in vocab.METRIC_PHRASES.items():
        rows += [("metric", metric, vocab.phrase_pattern(p)) for p in phrases]
    rows += [("metric", "session_count", vocab.phrase_pattern(p))
             for p in vocab.SESSION_COUNT_BY_ACTIVITY_PHRASES]
    for metric, phrases in vocab.HR_MODIFIERS.items():
        rows += [("modifier", metric, vocab.phrase_pattern(p)) for p in phrases]
    rows += [("hr", "", vocab.phrase_pattern(p)) for p in vocab.HR_NOUNS]
    for activity, phrases in vocab.ACTIVITY_SYNONYMS.items():
        rows += [("activity", activity, vocab.phrase_pattern(p)) for p in phrases]
    rows += [("session", "", vocab.phrase_pattern(p)) for p in vocab.SESSION_WORDS]
    rows += [("ambiguous_activity", phrase, vocab.phrase_pattern(phrase))
             for phrase in vocab.AMBIGUOUS_ACTIVITY_PHRASES]
    rows += [("summary", "", vocab.phrase_pattern(p)) for p in vocab.SUMMARY_PHRASES]
    rows += [("breakdown", "", vocab.phrase_pattern(p)) for p in vocab.BREAKDOWN_PHRASES]
    rows += [("not_built", NOT_BUILT_COMPARISON, vocab.phrase_pattern(p))
             for p in vocab.COMPARISON_WORDS]
    rows += [("not_built", NOT_BUILT_LONGEST, vocab.phrase_pattern(p))
             for p in vocab.LONGEST_WORDS]
    rows += [("unsupported", "", vocab.phrase_pattern(p)) for p in vocab.UNSUPPORTED_WORDS]
    rows += [("unsupported", "", vocab.phrase_pattern(p))
             for p in vocab.UNSUPPORTED_COMMAND_PHRASES]
    # A command word as the very first word of the question ("start
    # tracking my run").
    rows += [("unsupported", "", re.compile(rf"^{word}(?![a-z0-9'])"))
             for word in vocab.COMMAND_FIRST_WORDS]
    return rows


_TABLE = _table()
# Which kinds of longer match swallow a shorter one inside them.
_SWALLOWS = {
    "range": {"range", "ambiguous", "metric", "modifier"},
    "ambiguous": {"range", "ambiguous", "metric", "modifier"},
    "metric": {"range", "ambiguous", "metric", "modifier"},
    "modifier": {"range", "ambiguous", "metric", "modifier"},
    "hr": {"range", "ambiguous", "metric", "modifier"},
    # A sessions word inside "how many sessions" still counts; inside
    # "my last session" it doesn't.
    "session": {"range", "ambiguous"},
    "activity": set(),
    "unsupported": set(),
    "ambiguous_activity": set(),
    "summary": {"summary", "breakdown"},
    "breakdown": {"summary", "breakdown"},
    "not_built": set(),
}
_FOLLOW_UP_START = re.compile(
    "^(?:" + "|".join(re.escape(p) for p in vocab.FOLLOW_UP_STARTS) + ")(?![a-z0-9'])"
)


def _matches(text: str) -> list[_Match]:
    found = [
        _Match(kind, value, m.start(), m.end())
        for kind, value, pattern in _TABLE
        for m in pattern.finditer(text)
    ]
    return [
        m for m in found
        if not any(m.inside(o) for o in found if o.kind in _SWALLOWS[m.kind])
    ]


def parse(question: str) -> ParseResult:
    """Accepted (a MetricRequest whose range may be None, or a summary /
    breakdown intent), deferred, unsupported, clarify (which activity?) or
    not built (a comparison, a follow-up, the longest session). Never
    raises. `question` may already be typo-corrected (ask/typos.py)."""
    text = vocab.normalize(question)
    matches = _matches(text)
    by_kind: dict[str, list[_Match]] = {}
    for m in matches:
        by_kind.setdefault(m.kind, []).append(m)

    ranges = {m.value for m in by_kind.get("range", [])}
    ambiguous = bool(by_kind.get("ambiguous"))
    range_hint = next(iter(ranges)) if len(ranges) == 1 and not ambiguous else None
    activities = {m.value for m in by_kind.get("activity", [])}
    has_hr = bool(by_kind.get("hr"))
    modifiers = {m.value for m in by_kind.get("modifier", [])}
    has_session_word = bool(by_kind.get("session"))
    metrics = {m.value for m in by_kind.get("metric", [])}
    if has_hr:
        metrics |= modifiers
    detected = {
        "activities": tuple(sorted(activities)),
        "ranges": tuple(r for r in vocab.RANGES if r in ranges),
        "metrics": tuple(sorted(metrics)),
    }

    def result(status: str, reason: str, **fields) -> ParseResult:
        return ParseResult(status, None, "rules", reason, range_hint=range_hint,
                           **{**detected, **fields})

    if by_kind.get("unsupported"):
        words = ", ".join(text[m.start:m.end] for m in by_kind["unsupported"])
        return result(UNSUPPORTED, f"asks for something v1 can't do: {words}")
    metric_words = metrics or modifiers or has_hr or has_session_word
    if by_kind.get("not_built"):
        kind = by_kind["not_built"][0].value
        return result(NOT_BUILT, f"not built yet: {kind}", kind=kind)
    if _FOLLOW_UP_START.search(text) and not (metrics or modifiers or has_hr):
        return result(NOT_BUILT, "a follow-up to an earlier question",
                      kind=NOT_BUILT_FOLLOW_UP)

    unknown = [
        t.group(0) for t in _TOKEN.finditer(text)
        if t.group(0) not in FILLER_WORDS
        and not any(m.start <= t.start() and t.end() <= m.end for m in matches)
    ]
    if (by_kind.get("summary") or by_kind.get("breakdown")) and not (
        metrics or modifiers or has_hr
    ):
        return _intent(by_kind, ranges, ambiguous, unknown, result)

    if not metric_words and not activities:
        return result(UNSUPPORTED, "no metric or activity words")
    if unknown:
        return result(DEFERRED, f"unrecognized words: {', '.join(unknown)}")
    if ambiguous:
        return result(DEFERRED, "a time phrase with no single matching range")
    if len(metrics) != 1:
        if not metrics:
            return result(DEFERRED, "no metric")
        return result(DEFERRED, f"several metrics: {', '.join(sorted(metrics))}")
    if modifiers and not has_hr:
        return result(DEFERRED, "average/peak without heart rate")
    if has_hr and not modifiers:
        return result(DEFERRED, "heart rate without average or peak")
    if len(activities) > 1:
        return result(DEFERRED, f"several activities: {', '.join(sorted(activities))}")
    if len(ranges) > 1:
        return result(DEFERRED, f"several ranges: {', '.join(sorted(ranges))}")
    # An activity with a sessions word is deferred -- unless the metric is
    # an explicit session count ("how many sessions did I run", "how many
    # walking sessions"), which normalize() turns into the fixed "not by
    # activity yet" reply. A bare "sessions" ("... across my sessions")
    # still defers.
    if activities and has_session_word and metrics != {"session_count"}:
        return result(DEFERRED, "an activity together with a sessions word")

    (metric,) = metrics
    if not activities and by_kind.get("ambiguous_activity"):
        # "on my feet": ask which activity, never guess. Each option is the
        # whole question with the phrase replaced by that activity.
        phrase = by_kind["ambiguous_activity"][0].value
        options = tuple(
            Option(activity, vocab.phrase_pattern(phrase).sub(activity.lower(), text, count=1))
            for activity in vocab.AMBIGUOUS_ACTIVITY_PHRASES[phrase]
        )
        return result(CLARIFY, f"{phrase!r} could be more than one activity", options=options,
                      kind="activity")
    activity = next(iter(activities), None)
    range_name = next(iter(ranges), None)
    return ParseResult(
        ACCEPTED,
        MetricRequest(metric, activity, range_name),
        "rules",
        "keyword match",
        range_source="rules" if range_name else None,
        range_hint=range_hint,
        **detected,
    )


def _intent(by_kind: dict[str, list[_Match]], ranges: set[str], ambiguous: bool,
            unknown: list[str], result) -> ParseResult:
    """A summary or breakdown question (no metric words). A summary of a
    single session ("my last session", "my last run") is the session
    summary; a summary of any other range, or a breakdown phrase, is the
    activity breakdown for that range. Activity words inside the range
    phrase ("my last run") don't count; any other activity word, an
    unknown word, or more than one range defers."""
    range_matches = by_kind.get("range", [])
    loose_activities = [
        m for m in by_kind.get("activity", [])
        if not any(m.inside(r) for r in range_matches)
    ]
    if unknown:
        return result(DEFERRED, f"unrecognized words: {', '.join(unknown)}")
    if ambiguous or len(ranges) > 1:
        return result(DEFERRED, "a time phrase with no single matching range")
    if loose_activities:
        return result(DEFERRED, "an activity in a summary or breakdown question")
    range_name = next(iter(ranges), None)
    if by_kind.get("summary") and range_name == LATEST_SESSION:
        return result(ACCEPTED, "keyword match", intent=INTENT_SUMMARY,
                      intent_range=LATEST_SESSION)
    return result(ACCEPTED, "keyword match", intent=INTENT_BREAKDOWN, intent_range=range_name,
                  summary_fallback=bool(by_kind.get("summary")))
