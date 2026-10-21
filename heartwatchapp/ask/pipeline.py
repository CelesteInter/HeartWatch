"""Question in, answer out: typos -> guard -> rules -> (AI -> evidence check)
-> normalize -> validate -> run -> answer.

Plain-language overview: answer_question() is the whole pipeline for code
that can wait (tests, scripts/eval_ask.py). The GUI can't wait on the AI
model, so it uses the two halves: begin() does everything that needs no
model -- typo correction, the medical check, the keyword parser and, if
that's enough, the answer -- and returns either a finished Answer or a
PendingQuestion; the GUI sends the pending question's messages to the
model in the background and hands the reply (or the failure) to finish().

Typos (ask/typos.py) are corrected first, and the corrections are shown in
the "what I understood" line. The medical guard then checks the question
as typed, the corrected question, and its own looser fuzzy match
(guard.check_question()); any match gets the medical reply.

Modes (config.ask_mode()):
- hybrid: the keyword parser first; only a question it defers goes to the
  AI.
- rules: the keyword parser only; a deferred question isn't understood.
- llm: every question (after the medical check) goes to the AI, so the
  evaluation can compare the two parsers.

In every mode, the AI's form goes through the evidence check
(ask/evidence.py): an activity or range the question doesn't name is
dropped, and a range the question does name replaces the model's (v1's
hybrid-only range override, generalised). The evaluation can switch the
check off (`check_evidence=False`) to measure the model on its own.

Besides one number, a question can ask for the latest session's summary
(the Sessions view's plain summary, unchanged, after a short lead-in) or
the activity breakdown over a range (the Dashboard's breakdown). Replies
that can't answer carry up to three clickable suggestions (ask/suggest.py).

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
from ..data import db, queries, stats
from ..data.queries import ActivityBreakdown, MetricRequest, MetricResult
from . import answers, evidence, guard, llm_parser, rules, suggest, typos, vocab
from .parse_result import (
    ACCEPTED,
    CLARIFY,
    DEFERRED,
    INTENT_BREAKDOWN,
    INTENT_SUMMARY,
    NOT_BUILT,
    NOT_BUILT_COMPARISON,
    NOT_BUILT_FOLLOW_UP,
    UNSUPPORTED,
    Option,
    ParseResult,
)

logger = logging.getLogger("heartwatch.llm")

OUTCOME_QUERY = "query"
OUTCOME_SESSION_SUMMARY = "session_summary"
OUTCOME_ACTIVITY_BREAKDOWN = "activity_breakdown"
OUTCOME_MEDICAL = "medical"
OUTCOME_UNSUPPORTED = "unsupported"
OUTCOME_NEEDS_ACTIVITY = "needs_activity"
OUTCOME_SESSION_COUNT_BY_ACTIVITY = "session_count_by_activity"
OUTCOME_CLARIFY = "clarify"
OUTCOME_NOT_BUILT = "not_built"
OUTCOME_NOT_UNDERSTOOD = "not_understood"
OUTCOME_AI_UNAVAILABLE = "ai_unavailable"
OUTCOME_TIMEOUT = "timeout"
# Outcomes that answer from the data (everything else is a fixed reply).
ANSWERED_OUTCOMES = (OUTCOME_QUERY, OUTCOME_SESSION_SUMMARY, OUTCOME_ACTIVITY_BREAKDOWN)

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
    # Buttons under the reply: suggestions, or the choices of a clarification.
    suggestions: tuple[Option, ...] = ()
    # Typo corrections made before parsing: (original, corrected).
    corrections: tuple[tuple[str, str], ...] = ()
    # The activity breakdown's numbers, for that kind of answer.
    breakdown: ActivityBreakdown | None = None


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
    corrected: str = ""             # the typo-corrected question, for the evidence check
    corrections: tuple[tuple[str, str], ...] = ()


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


def _fixed(text: str, outcome: str, source: str, parse: ParseResult | None, timings: dict,
           suggestions: tuple[Option, ...] = ()) -> Answer:
    corrections = parse.corrections if parse is not None else ()
    return Answer(text, answers.with_corrections(None, corrections), None, outcome, source,
                  parse, None, None, timings, suggestions, corrections)


def _detected(parse: ParseResult | None) -> tuple[Option, ...]:
    if parse is None:
        return suggest.from_detected()
    return suggest.from_detected(parse.activities, parse.ranges)


def _now_ms_tz(now: datetime.datetime | None) -> tuple[int | None, datetime.tzinfo | None]:
    if now is None:
        return None, None
    return round(now.timestamp() * 1000), now.tzinfo


def _run(request: MetricRequest, conn_or_path: ConnOrPath,
         now: datetime.datetime | None) -> MetricResult:
    now_ms, tz = _now_ms_tz(now)
    if isinstance(conn_or_path, sqlite3.Connection):
        return queries.run(request, conn=conn_or_path, now_ms=now_ms, tz=tz)
    return queries.run(request, path=conn_or_path, now_ms=now_ms, tz=tz)


def _breakdown(range_name: str | None, session_id: int | None, conn_or_path: ConnOrPath,
               now: datetime.datetime | None) -> ActivityBreakdown:
    now_ms, tz = _now_ms_tz(now)
    if isinstance(conn_or_path, sqlite3.Connection):
        return queries.activity_breakdown(range_name, session_id, conn=conn_or_path,
                                          now_ms=now_ms, tz=tz)
    return queries.activity_breakdown(range_name, session_id, path=conn_or_path, now_ms=now_ms,
                                      tz=tz)


def _db_path(conn_or_path: ConnOrPath) -> Path:
    """A file path for code that only takes one (stats.compute_session_stats)."""
    if isinstance(conn_or_path, sqlite3.Connection):
        return Path(conn_or_path.execute("PRAGMA database_list").fetchone()[2])
    return Path(conn_or_path) if conn_or_path is not None else db.DB_PATH


def _answer_summary(parse: ParseResult, conn_or_path: ConnOrPath,
                    now: datetime.datetime | None, timings: dict) -> Answer:
    """The latest finished session's plain summary, exactly as the Sessions
    view shows it, after a lead-in with its local start time."""
    understood = answers.with_corrections(answers.summary_understood_line(), parse.corrections)
    started = time.perf_counter()
    path = _db_path(conn_or_path)
    recent = queries.list_recent_sessions(1, path=path)
    if not recent:
        return Answer(answers.NO_SESSIONS_REPLY, understood, None, OUTCOME_SESSION_SUMMARY,
                      parse.source, parse, None, None, {**timings, "run": _ms(started)},
                      suggest.from_detected(), parse.corrections)
    session = recent[0]
    session_stats = stats.compute_session_stats(session["id"], path)
    tz = now.tzinfo if now is not None else queries.local_timezone()
    text = answers.summary_text(
        answers.summary_lead(session["started_at"], tz),
        stats.format_stats_plain(session_stats),
        stats.confidence_caveat(session_stats),
    )
    breakdown = queries.activity_breakdown(session_id=session["id"], path=path)
    return Answer(text, understood, answers.caption(breakdown), OUTCOME_SESSION_SUMMARY,
                  parse.source, parse, None, None, {**timings, "run": _ms(started)},
                  corrections=parse.corrections, breakdown=breakdown)


def _answer_breakdown(parse: ParseResult, conn_or_path: ConnOrPath,
                      now: datetime.datetime | None, timings: dict) -> Answer:
    range_name = parse.intent_range or vocab.DEFAULT_RANGE
    understood = answers.with_corrections(
        answers.breakdown_understood_line(range_name, parse.intent_range is None,
                                          parse.summary_fallback),
        parse.corrections,
    )
    started = time.perf_counter()
    breakdown = _breakdown(range_name, None, conn_or_path, now)
    timings = {**timings, "run": _ms(started)}
    suggestions = () if breakdown.items else suggest.from_detected((), (range_name,))
    return Answer(answers.breakdown_text(range_name, breakdown), understood,
                  answers.caption(breakdown), OUTCOME_ACTIVITY_BREAKDOWN, parse.source, parse,
                  None, None, timings, suggestions, parse.corrections, breakdown)


def _not_built(parse: ParseResult, timings: dict) -> Answer:
    if parse.kind == NOT_BUILT_COMPARISON:
        return _fixed(answers.COMPARISON_REPLY, OUTCOME_NOT_BUILT, parse.source, parse, timings,
                      suggest.for_comparison(parse.metrics, parse.activities, parse.ranges))
    suggestions = _detected(parse)
    if parse.kind == NOT_BUILT_FOLLOW_UP:
        return _fixed(answers.follow_up_reply(suggestions[0].question), OUTCOME_NOT_BUILT,
                      parse.source, parse, timings, suggestions)
    return _fixed(answers.LONGEST_REPLY, OUTCOME_NOT_BUILT, parse.source, parse, timings,
                  suggestions)


def _answer_parse(parse: ParseResult, conn_or_path: ConnOrPath,
                  now: datetime.datetime | None, timings: dict) -> Answer:
    """A parse that isn't deferred, normalized, checked and answered."""
    source = parse.source
    if parse.status == UNSUPPORTED:
        return _fixed(answers.UNSUPPORTED_REPLY, OUTCOME_UNSUPPORTED, source, parse, timings,
                      _detected(parse))
    if parse.status == NOT_BUILT:
        return _not_built(parse, timings)
    if parse.status == CLARIFY:
        return _fixed(answers.clarify_reply([o.label for o in parse.options]), OUTCOME_CLARIFY,
                      source, parse, timings, parse.options)
    if parse.status != ACCEPTED:
        if parse.raw_reply is not None:
            logger.warning("ask: couldn't use the model's reply (%s): %r",
                           parse.reason, parse.raw_reply)
        return _fixed(answers.NOT_UNDERSTOOD_REPLY, OUTCOME_NOT_UNDERSTOOD, source, parse,
                      timings, _detected(parse))
    if parse.intent == INTENT_SUMMARY:
        return _answer_summary(parse, conn_or_path, now, timings)
    if parse.intent == INTENT_BREAKDOWN:
        return _answer_breakdown(parse, conn_or_path, now, timings)

    normalized = normalize(parse.request)
    if normalized.outcome == OUTCOME_NEEDS_ACTIVITY:
        return _fixed(answers.needs_activity_reply(), normalized.outcome, source, parse, timings,
                      suggest.which_activity(parse.request.range))
    if normalized.outcome == OUTCOME_SESSION_COUNT_BY_ACTIVITY:
        return _fixed(answers.SESSION_COUNT_BY_ACTIVITY_REPLY, normalized.outcome, source, parse,
                      timings, suggest.from_detected((parse.request.activity,), parse.ranges))
    request = normalized.request
    try:
        request.validate()
    except ValueError as exc:
        logger.warning("ask: request failed validation (%s): %r raw=%r",
                       exc, request, parse.raw_reply)
        return _fixed(answers.NOT_UNDERSTOOD_REPLY, OUTCOME_NOT_UNDERSTOOD, source, parse,
                      timings, _detected(parse))

    started = time.perf_counter()
    result = _run(request, conn_or_path, now)
    timings = {**timings, "run": _ms(started)}
    return Answer(
        answers.reply_text(request, result),
        answers.with_corrections(
            answers.understood_line(request, normalized.range_defaulted), parse.corrections
        ),
        answers.caption(result),
        OUTCOME_QUERY,
        source,
        parse,
        request,
        result,
        timings,
        suggest.for_no_data(request) if result.empty else (),
        parse.corrections,
    )


def _clarify_typo(typo: typos.TypoResult, timings: dict) -> Answer:
    """"Did you mean sitting or standing?", with the question re-asked
    with each reading as the buttons."""
    word, candidates = typo.ambiguous
    options = tuple(
        Option(vocab.activity_of_word(c) or c, typo.replace(word, c)) for c in candidates
    )
    parse = ParseResult(CLARIFY, None, "rules", f"{word!r} is close to {', '.join(candidates)}",
                        options=options, kind="typo", corrections=typo.corrections)
    return _fixed(answers.clarify_reply([o.label for o in options]), OUTCOME_CLARIFY, "rules",
                  parse, timings, options)


def begin(question: str, *, mode: str | None = None, conn_or_path: ConnOrPath = None,
          now: datetime.datetime | None = None) -> Answer | PendingQuestion:
    """Everything that needs no model. `now` (timezone-aware) pins the
    current time and timezone for ranges; None means the real ones."""
    mode = config.ask_mode() if mode is None else mode
    started = time.perf_counter()
    typo = typos.correct(question, guard.GUARD_WORDS)
    term = guard.check_question(question, typo.text)
    timings = {"guard": _ms(started)}
    if term is not None:
        return Answer(answers.MEDICAL_REPLY, None, None, OUTCOME_MEDICAL, "guard", None, None,
                      None, timings, (), typo.corrections)

    deferral = None
    if mode != config.ASK_MODE_LLM:
        if typo.ambiguous is not None:
            return _clarify_typo(typo, timings)
        started = time.perf_counter()
        parsed = dataclasses.replace(rules.parse(typo.text), corrections=typo.corrections)
        timings["rules"] = _ms(started)
        if parsed.status != DEFERRED:
            return _answer_parse(parsed, conn_or_path, now, timings)
        if mode == config.ASK_MODE_RULES:
            return _fixed(answers.NOT_UNDERSTOOD_REPLY, OUTCOME_NOT_UNDERSTOOD, "rules", parsed,
                          timings, _detected(parsed))
        deferral = parsed
    return PendingQuestion(
        question, mode, deferral, llm_parser.build_messages(question),
        llm_parser.response_schema(), timings, typo.text, typo.corrections,
    )


def finish(pending: PendingQuestion, *, raw: str | None = None, failure: str | None = None,
           conn_or_path: ConnOrPath = None, now: datetime.datetime | None = None,
           llm_ms: float | None = None, check_evidence: bool = True) -> Answer:
    """The model's raw reply (or why there isn't one: FAILURE_UNAVAILABLE,
    FAILURE_TIMEOUT, FAILURE_ERROR) turned into an Answer. The evidence
    check runs unless `check_evidence` is False (the evaluation's "before"
    measurement only)."""
    timings = dict(pending.timings)
    if llm_ms is not None:
        timings["llm"] = llm_ms
    if raw is None:
        if failure == FAILURE_UNAVAILABLE:
            return _fixed(answers.AI_UNAVAILABLE_REPLY, OUTCOME_AI_UNAVAILABLE, "llm",
                          pending.rules, timings, suggest.from_detected())
        logger.warning("ask: no reply from the model (%s)", failure)
        outcome = OUTCOME_TIMEOUT if failure == FAILURE_TIMEOUT else OUTCOME_NOT_UNDERSTOOD
        return _fixed(answers.NOT_UNDERSTOOD_REPLY, outcome, "llm", pending.rules, timings,
                      _detected(pending.rules))

    parse = llm_parser.parse_reply(raw)
    if check_evidence:
        parse = evidence.check(parse, pending.corrected or pending.question)
    parse = dataclasses.replace(parse, corrections=pending.corrections)
    return _answer_parse(parse, conn_or_path, now, timings)


def answer_question(text: str, *, conn_or_path: ConnOrPath = None,
                    now: datetime.datetime | None = None, mode: str | None = None,
                    llm: LlmCall | None = None, check_evidence: bool = True) -> Answer:
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
        return finish(step, raw=raw, conn_or_path=conn_or_path, now=now, llm_ms=_ms(started),
                      check_evidence=check_evidence)
    return finish(step, failure=failure, conn_or_path=conn_or_path, now=now, llm_ms=_ms(started))
