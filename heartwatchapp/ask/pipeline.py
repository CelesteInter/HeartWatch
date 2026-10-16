"""Question in, answer out: guard -> rules -> (AI) -> normalize -> validate
-> run -> answer.

Plain-language overview: answer_question() is the whole pipeline for code
that can wait (tests, scripts/eval_ask.py). The GUI can't wait on the AI
model, so it uses the two halves: begin() does everything that needs no
model -- the medical check, the keyword parser and, if that's enough, the
answer -- and returns either a finished Answer or a PendingQuestion; the
GUI sends the pending question's messages to the model in the background
and hands the reply (or the failure) to finish().

Modes (config.ask_mode()):
- hybrid: the keyword parser first; only a question it defers goes to the
  AI. If the keyword parser found exactly one clear time phrase, its range
  replaces the AI's (the model mixes up "this week" and "last week"); the
  override is recorded in ParseResult.range_source / llm_range.
- rules: the keyword parser only; a deferred question isn't understood.
- llm: every question (after the medical check) goes to the AI, with no
  override, so the evaluation can compare the two parsers cleanly.

normalize() applies the v1 rules for metric x activity (README, "Ask about
your data"): time in an activity needs an activity; total time narrowed to
an activity is time in that activity; counting sessions by activity isn't
supported; no range means the last 7 days, as on the Dashboard.
"""

from __future__ import annotations

import dataclasses
import datetime
import logging
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .. import config
from ..data import queries
from ..data.queries import MetricRequest, MetricResult
from . import answers, guard, llm_parser, rules, vocab
from .parse_result import ACCEPTED, DEFERRED, UNSUPPORTED, ParseResult

logger = logging.getLogger("heartwatch.llm")

OUTCOME_QUERY = "query"
OUTCOME_MEDICAL = "medical"
OUTCOME_UNSUPPORTED = "unsupported"
OUTCOME_NEEDS_ACTIVITY = "needs_activity"
OUTCOME_SESSION_COUNT_BY_ACTIVITY = "session_count_by_activity"
OUTCOME_NOT_UNDERSTOOD = "not_understood"
OUTCOME_AI_UNAVAILABLE = "ai_unavailable"
OUTCOME_TIMEOUT = "timeout"

# Why the model gave no reply (finish()'s `failure`).
FAILURE_UNAVAILABLE = "unavailable"
FAILURE_TIMEOUT = "timeout"
FAILURE_ERROR = "error"

ConnOrPath = sqlite3.Connection | str | Path | None
# answer_question()'s model: (messages, JSON schema) -> raw reply. Raises
# llm_parser.LlmUnavailable or TimeoutError; anything else counts as an error.
LlmCall = Callable[[list[dict], dict], str]


@dataclass(frozen=True)
class Answer:
    text: str                       # the reply, always Python's wording
    understood: str | None          # "Average heart rate · Running · this week"
    caption: str | None             # the simulated-predictions caption
    outcome: str                    # one of the OUTCOME_ names
    source: str                     # "guard", "rules" or "llm": who decided
    parse: ParseResult | None       # None when the guard caught the question
    request: MetricRequest | None   # after normalize(), when there is one
    result: MetricResult | None
    timings: dict                   # milliseconds: guard, rules, llm, run


@dataclass(frozen=True)
class PendingQuestion:
    """begin()'s result when the AI model is needed. Send `messages` with
    `format` to the model, then call finish()."""

    question: str
    mode: str
    rules: ParseResult | None       # the keyword parser's deferral (hybrid only)
    messages: list[dict]
    format: dict
    timings: dict


@dataclass(frozen=True)
class Normalized:
    outcome: str                    # OUTCOME_QUERY or a fixed-reply outcome
    request: MetricRequest | None   # the request to run, for OUTCOME_QUERY
    range_defaulted: bool = False


def normalize(request: MetricRequest) -> Normalized:
    """The metric x activity rules, applied after either parser and before
    validate(). See the module docstring."""
    if request.metric == "total_time" and request.activity is not None:
        request = dataclasses.replace(request, metric="time_in_activity")
    if request.metric == "time_in_activity" and request.activity is None:
        return Normalized(OUTCOME_NEEDS_ACTIVITY, None)
    if request.metric == "session_count" and request.activity is not None:
        return Normalized(OUTCOME_SESSION_COUNT_BY_ACTIVITY, None)
    if request.range is None and request.session_id is None:
        return Normalized(
            OUTCOME_QUERY, dataclasses.replace(request, range=vocab.DEFAULT_RANGE), True
        )
    return Normalized(OUTCOME_QUERY, request)


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)


def _fixed(text: str, outcome: str, source: str, parse: ParseResult | None,
           timings: dict) -> Answer:
    return Answer(text, None, None, outcome, source, parse, None, None, timings)


def _run(request: MetricRequest, conn_or_path: ConnOrPath,
         now: datetime.datetime | None) -> MetricResult:
    now_ms = round(now.timestamp() * 1000) if now is not None else None
    tz = now.tzinfo if now is not None else None
    if isinstance(conn_or_path, sqlite3.Connection):
        return queries.run(request, conn=conn_or_path, now_ms=now_ms, tz=tz)
    return queries.run(request, path=conn_or_path, now_ms=now_ms, tz=tz)


def _answer_parse(parse: ParseResult, conn_or_path: ConnOrPath,
                  now: datetime.datetime | None, timings: dict) -> Answer:
    """An accepted or unsupported parse, normalized, checked and answered."""
    source = parse.source
    if parse.status == UNSUPPORTED:
        return _fixed(answers.UNSUPPORTED_REPLY, OUTCOME_UNSUPPORTED, source, parse, timings)
    if parse.status != ACCEPTED:
        if parse.raw_reply is not None:
            logger.warning("ask: couldn't use the model's reply (%s): %r",
                           parse.reason, parse.raw_reply)
        return _fixed(answers.NOT_UNDERSTOOD_REPLY, OUTCOME_NOT_UNDERSTOOD, source, parse,
                      timings)

    normalized = normalize(parse.request)
    if normalized.outcome == OUTCOME_NEEDS_ACTIVITY:
        return _fixed(answers.needs_activity_reply(), normalized.outcome, source, parse, timings)
    if normalized.outcome == OUTCOME_SESSION_COUNT_BY_ACTIVITY:
        return _fixed(answers.SESSION_COUNT_BY_ACTIVITY_REPLY, normalized.outcome, source,
                      parse, timings)
    request = normalized.request
    try:
        request.validate()
    except ValueError as exc:
        logger.warning("ask: request failed validation (%s): %r raw=%r",
                       exc, request, parse.raw_reply)
        return _fixed(answers.NOT_UNDERSTOOD_REPLY, OUTCOME_NOT_UNDERSTOOD, source, parse,
                      timings)

    started = time.perf_counter()
    result = _run(request, conn_or_path, now)
    timings = {**timings, "run": _ms(started)}
    return Answer(
        answers.reply_text(request, result),
        answers.understood_line(request, normalized.range_defaulted),
        answers.caption(result),
        OUTCOME_QUERY,
        source,
        parse,
        request,
        result,
        timings,
    )


def begin(question: str, *, mode: str | None = None, conn_or_path: ConnOrPath = None,
          now: datetime.datetime | None = None) -> Answer | PendingQuestion:
    """Everything that needs no model. `now` (timezone-aware) pins the
    current time and timezone for ranges; None means the real ones."""
    mode = config.ask_mode() if mode is None else mode
    started = time.perf_counter()
    term = guard.check(question)
    timings = {"guard": _ms(started)}
    if term is not None:
        return _fixed(answers.MEDICAL_REPLY, OUTCOME_MEDICAL, "guard", None, timings)

    deferral = None
    if mode != config.ASK_MODE_LLM:
        started = time.perf_counter()
        parsed = rules.parse(question)
        timings["rules"] = _ms(started)
        if parsed.status != DEFERRED:
            return _answer_parse(parsed, conn_or_path, now, timings)
        if mode == config.ASK_MODE_RULES:
            return _fixed(answers.NOT_UNDERSTOOD_REPLY, OUTCOME_NOT_UNDERSTOOD, "rules", parsed,
                          timings)
        deferral = parsed
    return PendingQuestion(
        question, mode, deferral, llm_parser.build_messages(question),
        llm_parser.response_schema(), timings,
    )


def finish(pending: PendingQuestion, *, raw: str | None = None, failure: str | None = None,
           conn_or_path: ConnOrPath = None, now: datetime.datetime | None = None,
           llm_ms: float | None = None) -> Answer:
    """The model's raw reply (or why there isn't one: FAILURE_UNAVAILABLE,
    FAILURE_TIMEOUT, FAILURE_ERROR) turned into an Answer."""
    timings = dict(pending.timings)
    if llm_ms is not None:
        timings["llm"] = llm_ms
    if raw is None:
        if failure == FAILURE_UNAVAILABLE:
            return _fixed(answers.AI_UNAVAILABLE_REPLY, OUTCOME_AI_UNAVAILABLE, "llm",
                          pending.rules, timings)
        logger.warning("ask: no reply from the model (%s)", failure)
        outcome = OUTCOME_TIMEOUT if failure == FAILURE_TIMEOUT else OUTCOME_NOT_UNDERSTOOD
        return _fixed(answers.NOT_UNDERSTOOD_REPLY, outcome, "llm", pending.rules, timings)

    parse = llm_parser.parse_reply(raw)
    hint = pending.rules.range_hint if pending.rules is not None else None
    if (
        pending.mode == config.ASK_MODE_HYBRID
        and parse.status == ACCEPTED
        and hint is not None
        and parse.request.range != hint
    ):
        parse = dataclasses.replace(
            parse,
            request=dataclasses.replace(parse.request, range=hint),
            range_source="rules",
            llm_range=parse.request.range,
        )
    return _answer_parse(parse, conn_or_path, now, timings)


def answer_question(text: str, *, conn_or_path: ConnOrPath = None,
                    now: datetime.datetime | None = None, mode: str | None = None,
                    llm: LlmCall | None = None) -> Answer:
    """The whole pipeline, synchronously. `llm` is the model call (None:
    no model available)."""
    step = begin(text, mode=mode, conn_or_path=conn_or_path, now=now)
    if isinstance(step, Answer):
        return step
    if llm is None:
        return finish(step, failure=FAILURE_UNAVAILABLE, conn_or_path=conn_or_path, now=now)
    started = time.perf_counter()
    try:
        raw = llm(step.messages, step.format)
    except llm_parser.LlmUnavailable:
        failure = FAILURE_UNAVAILABLE
    except TimeoutError:
        failure = FAILURE_TIMEOUT
    except Exception:  # noqa: BLE001 - any other failure is "didn't understand"
        logger.exception("ask: model call failed")
        failure = FAILURE_ERROR
    else:
        return finish(step, raw=raw, conn_or_path=conn_or_path, now=now, llm_ms=_ms(started))
    return finish(step, failure=failure, conn_or_path=conn_or_path, now=now, llm_ms=_ms(started))
