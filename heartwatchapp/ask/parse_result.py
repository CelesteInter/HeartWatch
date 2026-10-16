"""What a parser hands back: ParseResult. Its own module so the keyword
parser (rules.py) and the AI parser (llm_parser.py) share it without
importing each other."""

from __future__ import annotations

from dataclasses import dataclass

from ..data.queries import MetricRequest

ACCEPTED = "accepted"        # `request` is filled in (its range may be None)
DEFERRED = "deferred"        # rules only: not sure, ask the AI (or give up)
UNSUPPORTED = "unsupported"  # not a question about the recorded data
INVALID = "invalid"          # AI only: the reply didn't parse or fit the schema


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
