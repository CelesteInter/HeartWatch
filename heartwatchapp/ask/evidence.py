"""The evidence check on the AI parser's form (handoff 9a, 3.5).

Plain-language overview: gemma3:1b sometimes fills in an activity or a time
period the question never mentions -- copied from an example in its prompt,
or simply guessed. So after the model fills in the form, and before
anything is answered, each activity and range it chose must be backed by
the question itself (after typo correction):

- Activity: kept only if the question contains that activity's word or a
  synonym ("jogged" for Running). If the only backing is a phrase that
  could mean several activities ("on my feet"), the bar asks which one.
  Otherwise the activity is dropped (time in an activity then asks
  "Which activity?").
- Range: kept if the question contains a time phrase for it -- the keyword
  parser's own phrases plus a few more (vocab.RANGE_EVIDENCE_PHRASES, e.g.
  "the week before this one" for last_week). If the question names
  exactly one range and the model picked another (or none), the
  question's range is used: this is v1's hybrid range override, now
  applied in every mode. Otherwise the range is dropped and the default
  (last 7 days, shown as "(default)") applies.
- Metric: not checked. Understanding a paraphrased metric is the model's
  actual job.
- Summary or breakdown: the question must ask for one ("tell me about",
  "summarize", "breakdown", "what did I do" -- vocab.SUMMARY_PHRASES and
  BREAKDOWN_PHRASES). If it doesn't, and the model also named a metric,
  the form is checked as that metric question instead; with no metric
  it's unsupported. A session summary also needs a single-session phrase
  ("my last session"); without one the activity breakdown for the
  question's range is given instead.

Every drop is recorded in ParseResult.dropped (and in the developer log,
if it's on). A longer phrase wins over a shorter one inside it, as in the
keyword parser, so "the last week" (last_week or last_7_days) doesn't
count as "last week".
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from ..data.queries import LATEST_SESSION
from . import vocab
from .parse_result import (
    ACCEPTED,
    CLARIFY,
    INTENT_BREAKDOWN,
    INTENT_METRIC,
    INTENT_SUMMARY,
    UNSUPPORTED,
    Option,
    ParseResult,
)


@dataclass(frozen=True)
class Evidence:
    activities: frozenset[str]
    # Activities backed only by an ambiguous phrase: phrase -> its activities.
    ambiguous_activities: dict[str, tuple[str, ...]]
    ranges: frozenset[str]
    # The ranges named by each separate time phrase (so "exactly one range
    # named" can be told apart from one phrase that allows two).
    range_phrases: tuple[frozenset[str], ...]
    # The question uses a summary or breakdown phrase.
    asks_for_summary_or_breakdown: bool = False


def _range_table() -> list[tuple[str, frozenset[str]]]:
    rows = [(p, frozenset({r})) for r, phrases in vocab.RANGE_SYNONYMS.items() for p in phrases]
    rows += [(p, frozenset({r})) for r, phrases in vocab.RANGE_EVIDENCE_PHRASES.items()
             for p in phrases]
    rows += [(p, frozenset(rs)) for p, rs in vocab.AMBIGUOUS_RANGE_EVIDENCE.items()]
    # Vague phrases back no range, but still swallow what's inside them.
    rows += [(p, frozenset()) for p in vocab.AMBIGUOUS_RANGE_PHRASES]
    return rows


_RANGE_TABLE = [(phrase, ranges, vocab.phrase_pattern(phrase)) for phrase, ranges in _range_table()]


def find(text: str) -> Evidence:
    """What `text` (normalize()d, typo-corrected) names."""
    text = vocab.normalize(text)
    found = [
        (m.start(), m.end(), ranges)
        for _, ranges, pattern in _RANGE_TABLE
        for m in pattern.finditer(text)
    ]
    kept = [
        f for f in found
        if not any(o[0] <= f[0] and f[1] <= o[1] and o[1] - o[0] > f[1] - f[0] for o in found)
    ]
    range_phrases = tuple(ranges for _, _, ranges in kept if ranges)
    activities = {
        activity
        for activity, words in vocab.ACTIVITY_SYNONYMS.items()
        for word in words
        if vocab.phrase_pattern(word).search(text)
    }
    ambiguous = {
        phrase: options
        for phrase, options in vocab.AMBIGUOUS_ACTIVITY_PHRASES.items()
        if vocab.phrase_pattern(phrase).search(text)
    }
    return Evidence(
        frozenset(activities),
        ambiguous,
        frozenset(r for ranges in range_phrases for r in ranges),
        range_phrases,
        any(vocab.phrase_pattern(p).search(text)
            for p in (*vocab.SUMMARY_PHRASES, *vocab.BREAKDOWN_PHRASES)),
    )


def _single_range(evidence: Evidence) -> str | None:
    """The one range the question names: every time phrase in it names
    exactly that one range (v1's range_hint rule)."""
    if evidence.range_phrases and all(len(r) == 1 for r in evidence.range_phrases):
        distinct = {r for ranges in evidence.range_phrases for r in ranges}
        if len(distinct) == 1:
            return next(iter(distinct))
    return None


def _checked_range(chosen: str | None, evidence: Evidence) -> tuple[str | None, str | None,
                                                                     str | None]:
    """(final range, range_source, what was dropped or None)."""
    single = _single_range(evidence)
    if chosen is not None and chosen in evidence.ranges and (single is None or single == chosen):
        return chosen, "llm", None
    if single is not None:
        return single, "rules", None
    if chosen is not None:
        return None, None, f"range {chosen}"
    return None, None, None


def check(parse: ParseResult, text: str) -> ParseResult:
    """`parse` (the AI parser's accepted form) with every activity and
    range the question doesn't back removed or replaced. Anything that
    isn't an accepted AI form is returned unchanged."""
    if parse.status != ACCEPTED or parse.source != "llm":
        return parse
    evidence = find(text)
    dropped: list[str] = []

    if parse.intent != INTENT_METRIC and not evidence.asks_for_summary_or_breakdown:
        dropped.append(f"intent {parse.intent}")
        if parse.request is None:
            return dataclasses.replace(parse, status=UNSUPPORTED, intent=INTENT_METRIC,
                                       intent_range=None, dropped=tuple(dropped))
        parse = dataclasses.replace(parse, intent=INTENT_METRIC, intent_range=None)

    if parse.intent != INTENT_METRIC:
        chosen = parse.intent_range
        final, source, drop = _checked_range(chosen, evidence)
        if drop:
            dropped.append(drop)
        intent, fallback = parse.intent, False
        if intent == INTENT_SUMMARY and final != LATEST_SESSION:
            # A summary needs a single-session phrase; otherwise give the
            # breakdown for whatever range the question named.
            dropped.append("intent session_summary")
            intent, fallback = INTENT_BREAKDOWN, True
        return dataclasses.replace(
            parse, request=None, intent=intent, intent_range=final, summary_fallback=fallback,
            range_source=source, llm_range=chosen if final != chosen else None,
            dropped=tuple(dropped),
        )

    request = parse.request
    activity = request.activity
    if activity is not None and activity not in evidence.activities:
        options = [
            (phrase, acts) for phrase, acts in evidence.ambiguous_activities.items()
            if activity in acts
        ]
        if options and not evidence.activities:
            phrase, acts = options[0]
            normalized = vocab.normalize(text)
            return dataclasses.replace(
                parse, status=CLARIFY, request=None, kind="activity",
                options=tuple(
                    Option(a, vocab.phrase_pattern(phrase).sub(a.lower(), normalized, count=1))
                    for a in acts
                ),
                dropped=(f"activity {activity}",),
            )
        dropped.append(f"activity {activity}")
        activity = None

    final, source, drop = _checked_range(request.range, evidence)
    if drop:
        dropped.append(drop)
    return dataclasses.replace(
        parse,
        request=dataclasses.replace(request, activity=activity, range=final),
        range_source=source,
        llm_range=request.range if final != request.range else None,
        dropped=tuple(dropped),
    )
