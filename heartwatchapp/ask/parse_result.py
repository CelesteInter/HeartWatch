"""What a parser hands back: ParseResult. Its own module so the keyword
parser (rules.py) and the AI parser (llm_parser.py) share it without
importing each other."""

from __future__ import annotations

from dataclasses import dataclass

from ..data.queries import MetricRequest

ACCEPTED = "accepted"        # `request` is filled in (its range may be None), or
                             # `intent` is a summary / breakdown
DEFERRED = "deferred"        # rules only: not sure, ask the AI (or give up)
UNSUPPORTED = "unsupported"  # not a question about the recorded data
INVALID = "invalid"          # AI only: the reply didn't parse or fit the schema
CLARIFY = "clarify"          # two readings are equally likely: `options` asks which
NOT_BUILT = "not_built"      # understood, but not built yet: `kind` says what

# What an accepted parse asks for.
INTENT_METRIC = "metric"                         # one number: `request`
INTENT_SUMMARY = "session_summary"               # the latest session's plain summary
INTENT_BREAKDOWN = "activity_breakdown"          # time in each activity over `intent_range`

# NOT_BUILT kinds (handoff 9a, 3.6).
NOT_BUILT_COMPARISON = "comparison"
NOT_BUILT_FOLLOW_UP = "follow_up"
NOT_BUILT_LONGEST = "longest_session"
NOT_BUILT_KINDS = (NOT_BUILT_COMPARISON, NOT_BUILT_FOLLOW_UP, NOT_BUILT_LONGEST)


@dataclass(frozen=True)
class Option:
    """One clickable choice: `label` on the button, `question` submitted
    when it's clicked (exactly as if typed)."""

    label: str
    question: str


@dataclass(frozen=True)
class ParseResult:
    status: str
    request: MetricRequest | None
    source: str                       # "rules" or "llm"
    reason: str
    # Which parser picked request.range: "rules", "llm", or None when the
    # question named no range (pipeline.normalize() then applies the
    # default). In hybrid mode the rules parser's range can override the
    # AI's (see pipeline.finish()).
    range_source: str | None = None
    # Rules only: the one unambiguous range phrase the question contained,
    # even when the parse as a whole was deferred. None if there were zero,
    # several, or an ambiguous one.
    range_hint: str | None = None
    # AI only, after a hybrid override: the range the AI itself chose.
    llm_range: str | None = None
    # AI only: the model's raw reply, for logs and the evaluation.
    raw_reply: str | None = None
    # INTENT_METRIC, or a summary / breakdown (then `request` is None and
    # `intent_range` is its range; None means the default).
    intent: str = INTENT_METRIC
    intent_range: str | None = None
    # A summary was asked for over a range of several sessions, so the
    # breakdown is given instead (the understood line says so).
    summary_fallback: bool = False
    # CLARIFY: the choices; NOT_BUILT: which feature.
    options: tuple[Option, ...] = ()
    kind: str | None = None
    # What was detected even if the parse wasn't accepted, for building
    # suggestions (ask/suggest.py).
    activities: tuple[str, ...] = ()
    ranges: tuple[str, ...] = ()
    metrics: tuple[str, ...] = ()
    # Typo corrections made before parsing: (original, corrected).
    corrections: tuple[tuple[str, str], ...] = ()
    # AI only: what the evidence check (ask/evidence.py) removed, e.g.
    # ("activity Walking", "range yesterday").
    dropped: tuple[str, ...] = ()
