"""The AI parser: the local model fills in the question form.

Plain-language overview: questions the keyword parser isn't sure about are
sent to gemma3:1b with a JSON schema (Ollama's structured output, the
`format=` option), so the reply can only be a form with four fields --
intent (a metric query, the session summary, the activity breakdown, or
unsupported), metric, activity, range -- each limited to the values in
ask/vocab.py. This module builds that request and checks the reply; it
never talks to Ollama itself. The app sends the request through
llm/worker.py's LlmWorker.request_parse() (off the GUI thread, with a
timeout in ui/pending_request.py), and scripts/eval_ask.py through
llm/client.py. Nothing the model writes is ever shown: a reply that isn't
valid JSON, doesn't fit the schema, or fails MetricRequest.validate() just
gets the fixed "didn't understand" reply.
"""

from __future__ import annotations

import json

from ..data.queries import MetricRequest
from ..llm import prompts
from . import vocab
from .parse_result import (
    ACCEPTED,
    INTENT_BREAKDOWN,
    INTENT_SUMMARY,
    INVALID,
    UNSUPPORTED,
    ParseResult,
)

FIELDS = ("intent", "metric", "activity", "range")
INTENTS = ("query", INTENT_SUMMARY, INTENT_BREAKDOWN, "unsupported")


class LlmUnavailable(Exception):
    """Ollama isn't running, or the model isn't pulled."""


def _nullable_enum(values) -> dict:
    return {"anyOf": [{"type": "string", "enum": list(values)}, {"type": "null"}]}


def response_schema() -> dict:
    """The JSON schema passed as `format=`. Its enums are vocab.METRICS,
    ACTIVITIES and RANGES themselves, not hand-typed copies."""
    return {
        "type": "object",
        "properties": {
            "intent": {"type": "string", "enum": list(INTENTS)},
            "metric": _nullable_enum(vocab.METRICS),
            "activity": _nullable_enum(vocab.ACTIVITIES),
            "range": _nullable_enum(vocab.RANGES),
        },
        "required": list(FIELDS),
        "additionalProperties": False,
    }


def build_messages(question: str) -> list[dict]:
    return prompts.build_ask_messages(question)


def _invalid(reason: str, raw: str) -> ParseResult:
    return ParseResult(INVALID, None, "llm", reason, raw_reply=raw)


def parse_reply(raw: str) -> ParseResult:
    """Checks the model's raw reply against the schema (Ollama enforces it
    too, but the app doesn't rely on that) and turns it into a ParseResult:
    accepted with a MetricRequest, unsupported, or invalid. Never raises."""
    try:
        form = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return _invalid("reply isn't JSON", raw)
    if not isinstance(form, dict) or set(form) != set(FIELDS):
        return _invalid("reply doesn't have exactly the form's fields", raw)
    allowed = {
        "intent": INTENTS,
        "metric": (*vocab.METRICS, None),
        "activity": (*vocab.ACTIVITIES, None),
        "range": (*vocab.RANGES, None),
    }
    for field in FIELDS:
        if form[field] not in allowed[field]:
            return _invalid(f"{field} {form[field]!r} isn't an allowed value", raw)
    if form["intent"] == "unsupported":
        return ParseResult(UNSUPPORTED, None, "llm", "model said unsupported", raw_reply=raw)
    if form["intent"] in (INTENT_SUMMARY, INTENT_BREAKDOWN):
        # Not a number. If the model also filled in a metric, it's kept as
        # `request`: the evidence check (ask/evidence.py) falls back to it
        # when the question doesn't actually ask for a summary or breakdown.
        fallback = (
            MetricRequest(form["metric"], form["activity"], form["range"])
            if form["metric"] is not None else None
        )
        return ParseResult(
            ACCEPTED, fallback, "llm", "model filled in the form", raw_reply=raw,
            intent=form["intent"], intent_range=form["range"],
            range_source="llm" if form["range"] else None,
        )
    if form["metric"] is None:
        return _invalid("intent is query but metric is null", raw)
    return ParseResult(
        ACCEPTED,
        MetricRequest(form["metric"], form["activity"], form["range"]),
        "llm",
        "model filled in the form",
        range_source="llm" if form["range"] else None,
        raw_reply=raw,
    )
