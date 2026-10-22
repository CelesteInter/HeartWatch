"""Repeatable evaluation of the AI bar's question parsers, for the writeup.

Plain-language overview: every question in a question file is put through
the app's own pipeline (ask/pipeline.py) in each mode -- keyword parser
only ("rules"), AI only ("llm"), and both ("hybrid", the app's default) --
and the form each mode understood is compared with the expected one written
by hand in the file. Scoring is on the form (metric, activity, range, or
which fixed reply), never on the prose, which Python writes anyway. The
headline number is false accepts: a question answered with a number when
it should have been refused, or answered for the wrong metric, activity or
range.

Nothing about it is random between runs: the model runs at temperature 0
with seed 0 (llm/client.py's ASK_OPTIONS), "now" is pinned (--now, default
Wednesday 2026-10-14 15:00 in --tz America/Denver) so "yesterday" and "this
week" always mean the same days, and the sessions are written by
scripts/demo_db.py from --data-seed into a new throwaway database (the real
database is never opened). Running it twice gives the same runs.jsonl
apart from the timings.

Run from the repo root (HeartWatch/), with the venv active and, for the
llm and hybrid modes, Ollama running:

    python -m heartwatchapp.scripts.eval_ask
    python -m heartwatchapp.scripts.eval_ask --questions heartwatchapp/eval/ask_questions_heldout.jsonl
    python -m heartwatchapp.scripts.eval_ask --modes rules

If Ollama isn't available, the rules mode still runs and the others are
marked skipped. The dev questions (eval/ask_questions_dev.jsonl, and
v1.1's eval/ask_questions_dev_v11.jsonl) may be used to tune the keyword
parser; a held-out file must not be.

v1.1 (handoff 9a): a question may also expect a session summary or an
activity breakdown (with its range), a clarification (with its options),
or a not-built reply (with its kind), and may list the typo corrections it
expects (scored only when listed). In llm mode each question's model reply
is scored twice -- after the evidence check (ask/evidence.py), as the app
answers, and before it, from the raw form -- using the same reply, and
summary.md shows the two side by side.

Writes heartwatchapp/eval_results/ask-<YYYYMMDD-HHMMSS>/:
    runs.jsonl  one JSON object per question per mode
    meta.json   model, digest, Ollama version, options, modes, now, the
                question file's SHA-256
    summary.md  a table per mode, then every false accept
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import hashlib
import importlib.metadata
import json
import logging
import statistics
import sys
import tempfile
from pathlib import Path
from zoneinfo import ZoneInfo

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from heartwatchapp import config  # noqa: E402
from heartwatchapp.ask import pipeline, vocab  # noqa: E402
from heartwatchapp.ask.llm_parser import LlmUnavailable  # noqa: E402
from heartwatchapp.ask.parse_result import NOT_BUILT_KINDS  # noqa: E402
from heartwatchapp.llm import client as llm_client  # noqa: E402
from heartwatchapp.scripts.demo_db import (  # noqa: E402
    DestinationExistsError,
    RealDatabaseError,
    add_phases_session,
    build_demo_db,
    check_destination,
)
from heartwatchapp.scripts.eval_summaries import _cell, make_output_dir  # noqa: E402

APP_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = APP_DIR / "eval_results"
DEFAULT_QUESTIONS = APP_DIR / "eval" / "ask_questions_dev.jsonl"
DEFAULT_NOW = "2026-10-14T15:00"
DEFAULT_TZ = "America/Denver"
MODES = (config.ASK_MODE_RULES, config.ASK_MODE_LLM, config.ASK_MODE_HYBRID)
LLM_TIMEOUT_S = 10

EXPECTED_OUTCOMES = (
    pipeline.OUTCOME_QUERY,
    pipeline.OUTCOME_UNSUPPORTED,
    pipeline.OUTCOME_MEDICAL,
    pipeline.OUTCOME_NEEDS_ACTIVITY,
    pipeline.OUTCOME_SESSION_COUNT_BY_ACTIVITY,
    # v1.1 (eval/ask_questions_dev_v11.jsonl):
    pipeline.OUTCOME_SESSION_SUMMARY,
    pipeline.OUTCOME_ACTIVITY_BREAKDOWN,
    pipeline.OUTCOME_CLARIFY,
    pipeline.OUTCOME_NOT_BUILT,
)
REFUSALS = (pipeline.OUTCOME_MEDICAL, pipeline.OUTCOME_UNSUPPORTED)
FORM_FIELDS = ("metric", "activity", "range")
INTENT_OUTCOMES = (pipeline.OUTCOME_SESSION_SUMMARY, pipeline.OUTCOME_ACTIVITY_BREAKDOWN)

# Sessions placed around the pinned "now": (days before now's date, hour,
# minute, phases). With the default now (Wed Oct 14) they fall on Sep 25
# (last 30 days only), Thu Oct 8 (last week), Mon Oct 12 (this week), Tue
# Oct 13 (yesterday) and this morning. scripts/demo_db.py's own demo
# session is also there, DEMO_SESSION_DELAY after the pinned now (so,
# as when it was timed from the real clock, only "all" and
# "latest_session" include it); pinning it makes the session summary's
# date and time the same on every run.
DEMO_SESSION_DELAY = datetime.timedelta(hours=1)
EVAL_SESSIONS = (
    (19, 9, 0, [("Walking", 600, 98), ("Sitting", 300, 70)]),
    (6, 18, 0, [("Running", 900, 142), ("Walking", 300, 105)]),
    (2, 7, 30, [("Sitting", 600, 68), ("Standing", 300, 80), ("Walking", 240, 101)]),
    (1, 18, 0, [("Walking", 420, 103), ("Running", 180, 138)]),
    (0, 8, 0, [("Standing", 200, 84), ("Walking", 140, 100)]),
)


# -- pure parts (unit-tested without Ollama) ----------------------------------


def load_questions(path: Path) -> list[dict]:
    """The question file, one JSON object per non-empty line. Raises
    ValueError naming the line if a question is malformed."""
    questions, seen = [], set()
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        q = json.loads(line)
        expected = q.get("expected", {})
        problems = []
        if not q.get("id") or q["id"] in seen:
            problems.append("missing or repeated id")
        if not q.get("question"):
            problems.append("no question")
        if expected.get("outcome") not in EXPECTED_OUTCOMES:
            problems.append(f"outcome must be one of {', '.join(EXPECTED_OUTCOMES)}")
        if expected.get("outcome") == pipeline.OUTCOME_QUERY and (
            expected.get("metric") not in vocab.METRICS
            or expected.get("activity") not in (*vocab.ACTIVITIES, None)
            or expected.get("range") not in vocab.RANGES
        ):
            problems.append("a query needs a valid metric, activity (or null) and range")
        if expected.get("outcome") in INTENT_OUTCOMES and expected.get("range") not in vocab.RANGES:
            problems.append("a summary or breakdown needs a valid range")
        if expected.get("outcome") == pipeline.OUTCOME_CLARIFY and not expected.get("options"):
            problems.append("a clarification needs its options")
        if (expected.get("outcome") == pipeline.OUTCOME_NOT_BUILT
                and expected.get("kind") not in NOT_BUILT_KINDS):
            problems.append(f"not_built needs a kind: one of {', '.join(NOT_BUILT_KINDS)}")
        if "corrections" in expected and not all(
            isinstance(c, list) and len(c) == 2 for c in expected["corrections"]
        ):
            problems.append("corrections must be [original, corrected] pairs")
        if problems:
            raise ValueError(f"{path.name} line {number}: {'; '.join(problems)}")
        seen.add(q["id"])
        questions.append(q)
    return questions


def parse_now(text: str, tz_name: str) -> datetime.datetime:
    """--now (ISO, local wall-clock time) in --tz, as an aware datetime."""
    return datetime.datetime.fromisoformat(text).replace(tzinfo=ZoneInfo(tz_name))


def build_eval_db(db_path: Path, now: datetime.datetime, data_seed: int) -> Path:
    """A new database at `db_path` (never the real one): the demo session,
    then EVAL_SESSIONS around `now`. Returns its absolute path."""
    demo_start = round((now + DEMO_SESSION_DELAY).timestamp() * 1000)
    target, _ = build_demo_db(db_path, data_seed=data_seed, started_at=demo_start)
    for i, (days_before, hour, minute, phases) in enumerate(EVAL_SESSIONS):
        day = now.date() - datetime.timedelta(days=days_before)
        start = datetime.datetime.combine(day, datetime.time(hour, minute), tzinfo=now.tzinfo)
        add_phases_session(
            target, round(start.timestamp() * 1000), phases, data_seed=data_seed + 1 + i
        )
    return target


def form_of(answer: pipeline.Answer) -> dict:
    """What a mode understood: the outcome, plus the normalized form for a
    computed answer, the range of a summary or breakdown, the choices of a
    clarification, which feature isn't built, and the typo corrections."""
    request = answer.request if answer.outcome == pipeline.OUTCOME_QUERY else None
    parse = answer.parse
    form = {
        "outcome": answer.outcome,
        "metric": request.metric if request else None,
        "activity": request.activity if request else None,
        "range": request.range if request else None,
        "corrections": [list(c) for c in answer.corrections],
    }
    if answer.outcome in INTENT_OUTCOMES:
        form["range"] = parse.intent_range or vocab.DEFAULT_RANGE
    if answer.outcome == pipeline.OUTCOME_CLARIFY:
        form["options"] = sorted(o.label for o in answer.suggestions)
    if answer.outcome == pipeline.OUTCOME_NOT_BUILT:
        form["kind"] = parse.kind
    return form


def score(expected: dict, actual: dict) -> dict:
    """Exact match, false accept, and per-field correctness for one question.

    v1's rules are unchanged for v1's outcomes. For v1.1: a summary or
    breakdown must have the right range, a clarification the right options
    and a not-built reply the right kind; when the question lists its
    expected typo corrections they must match exactly (none wrongly, none
    missed), and only then. A false accept is any answer from the data
    (a number, a summary or a breakdown) that isn't exact -- which covers
    answering an ambiguous typo or phrase instead of asking, a medical
    question with typos that isn't refused, and a wrong typo correction
    that leads to a number."""
    is_query = expected["outcome"] == pipeline.OUTCOME_QUERY
    answered = actual["outcome"] in pipeline.ANSWERED_OUTCOMES
    fields = {
        f: (actual["outcome"] == pipeline.OUTCOME_QUERY and actual[f] == expected.get(f))
        if is_query else None
        for f in FORM_FIELDS
    }
    corrections = (
        [list(c) for c in actual.get("corrections", [])] == expected["corrections"]
        if "corrections" in expected else None
    )
    same_outcome = actual["outcome"] == expected["outcome"]
    if expected["outcome"] in INTENT_OUTCOMES:
        detail = actual.get("range") == expected["range"]
    elif expected["outcome"] == pipeline.OUTCOME_CLARIFY:
        detail = actual.get("options") == sorted(expected["options"])
    elif expected["outcome"] == pipeline.OUTCOME_NOT_BUILT:
        detail = actual.get("kind") == expected["kind"]
    else:
        detail = not is_query or all(fields.values())
    exact = same_outcome and detail and corrections is not False
    return {
        "exact": exact,
        "false_accept": answered and not exact,
        "fields": fields,
        "corrections_ok": corrections,
    }


def override_effect(expected: dict, answer: pipeline.Answer) -> str | None:
    """AI answers only: None if the question's own range didn't override
    the AI's (the evidence check, ask/evidence.py); otherwise "fixed",
    "broke", "neutral" (wrong either way, or both right), or "n/a" (the
    question shouldn't have been answered with a number)."""
    parse = answer.parse
    if parse is None or parse.source != "llm" or parse.range_source != "rules":
        return None
    if expected["outcome"] != pipeline.OUTCOME_QUERY or parse.request is None:
        return "n/a"
    llm_range = parse.llm_range or vocab.DEFAULT_RANGE
    final = parse.request.range
    if final == expected["range"] and llm_range != expected["range"]:
        return "fixed"
    if llm_range == expected["range"] and final != expected["range"]:
        return "broke"
    return "neutral"


def run_record(mode: str, question: dict, answer: pipeline.Answer, llm_timing: dict,
               before: pipeline.Answer | None = None) -> dict:
    """One line of runs.jsonl. `before` (llm mode only) is the same model
    reply answered without the evidence check."""
    actual = form_of(answer)
    parse = answer.parse
    before_record = None
    if before is not None:
        before_actual = form_of(before)
        before_record = {"actual": before_actual,
                         **score(question["expected"], before_actual)}
    return {
        "before_evidence": before_record,
        "dropped": list(parse.dropped) if parse else [],
        "mode": mode,
        "id": question["id"],
        "category": question.get("category"),
        "question": question["question"],
        "expected": question["expected"],
        "actual": actual,
        **score(question["expected"], actual),
        "decided_by": answer.source,
        "range_source": parse.range_source if parse else None,
        "llm_range": parse.llm_range if parse else None,
        "override": override_effect(question["expected"], answer),
        "raw_reply": parse.raw_reply if parse else None,
        "reply": answer.text,
        "understood": answer.understood,
        "caption": answer.caption,
        "timings_ms": answer.timings,
        "llm_load_duration_ns": llm_timing.get("load_duration"),
    }


def _display_path(path: Path) -> str:
    """`path` relative to the repo root when it's inside it."""
    try:
        return str(path.resolve().relative_to(APP_DIR.parent))
    except ValueError:
        return str(path)


def _pct(n: int, d: int) -> str:
    return f"{n}/{d} ({100 * n / d:.0f}%)" if d else "0/0"


def mode_stats(records: list[dict]) -> dict:
    """The numbers in one mode's summary table."""
    queries_expected = [r for r in records if r["expected"]["outcome"] == pipeline.OUTCOME_QUERY]
    refusals = [r for r in records if r["expected"]["outcome"] in REFUSALS]
    rules_ms = [r["timings_ms"]["rules"] for r in records if "rules" in r["timings_ms"]]
    llm_ms = [r["timings_ms"]["llm"] for r in records if "llm" in r["timings_ms"]]
    overrides = [r["override"] for r in records if r["override"]]

    def of(outcomes: tuple[str, ...]) -> list[dict]:
        return [r for r in records if r["expected"]["outcome"] in outcomes]

    intents = of(INTENT_OUTCOMES)
    clarify = of((pipeline.OUTCOME_CLARIFY,))
    not_built = of((pipeline.OUTCOME_NOT_BUILT,))
    with_corrections = [r for r in records if r.get("corrections_ok") is not None]
    before = [r["before_evidence"] for r in records if r.get("before_evidence")]
    return {
        "n": len(records),
        "exact": sum(r["exact"] for r in records),
        "fields": {
            f: sum(bool(r["fields"][f]) for r in queries_expected) for f in FORM_FIELDS
        },
        "n_queries": len(queries_expected),
        "false_accepts": sum(r["false_accept"] for r in records),
        "intent_exact": sum(r["exact"] for r in intents),
        "n_intents": len(intents),
        "clarify_exact": sum(r["exact"] for r in clarify),
        "n_clarify": len(clarify),
        "not_built_exact": sum(r["exact"] for r in not_built),
        "n_not_built": len(not_built),
        "corrections_ok": sum(bool(r["corrections_ok"]) for r in with_corrections),
        "n_corrections": len(with_corrections),
        "dropped": sum(bool(r.get("dropped")) for r in records),
        "before_exact": sum(b["exact"] for b in before) if before else None,
        "before_false_accepts": sum(b["false_accept"] for b in before) if before else None,
        "refusals_caught": sum(
            r["actual"]["outcome"] not in pipeline.ANSWERED_OUTCOMES for r in refusals
        ),
        "refusals_exact": sum(r["actual"]["outcome"] == r["expected"]["outcome"] for r in refusals),
        "n_refusals": len(refusals),
        "decided_by": {
            who: sum(r["decided_by"] == who for r in records) for who in ("guard", "rules", "llm")
        },
        "overrides": {k: overrides.count(k) for k in ("fixed", "broke", "neutral", "n/a")},
        "n_overrides": len(overrides),
        "rules_median_ms": statistics.median(rules_ms) if rules_ms else None,
        "llm_first_ms": llm_ms[0] if llm_ms else None,
        "llm_warm_median_ms": statistics.median(llm_ms[1:]) if len(llm_ms) > 1 else None,
    }


def _ms_text(value: float | None) -> str:
    return "—" if value is None else f"{value:.1f} ms"


def format_summary_md(meta: dict, records: list[dict]) -> str:
    lines = [
        "# Ask-bar parser evaluation",
        "",
        f"- Questions: `{meta['questions_file']}` ({meta['n_questions']} questions, "
        f"sha256 `{meta['questions_sha256'][:12]}…`)",
        f"- Now: {meta['now']} ({meta['tz']})",
    ]
    if meta["digest"] is None:
        lines.append("- Model: not used (rules mode only)")
    else:
        lines.append(
            f"- Model: `{meta['model']}` (digest `{meta['digest']}`), Ollama "
            f"{meta['ollama_version']}, options `{json.dumps(meta['options'], sort_keys=True)}`"
        )
    if meta["skipped_modes"]:
        lines.append(f"- Skipped (Ollama unavailable): {', '.join(meta['skipped_modes'])}")
    for mode in meta["modes_run"]:
        rows = [r for r in records if r["mode"] == mode]
        s = mode_stats(rows)
        lines += [
            "",
            f"## {mode}",
            "",
            "| Measure | Value |",
            "|---|---|",
            f"| Exact match (whole form) | {_pct(s['exact'], s['n'])} |",
            *[
                f"| {f} correct (of expected queries) | {_pct(s['fields'][f], s['n_queries'])} |"
                for f in FORM_FIELDS
            ],
            f"| **False accepts** | **{_pct(s['false_accepts'], s['n'])}** |",
            f"| Refusals caught (medical/unsupported not answered) "
            f"| {_pct(s['refusals_caught'], s['n_refusals'])} |",
            f"| Refusals with the exact expected reply "
            f"| {_pct(s['refusals_exact'], s['n_refusals'])} |",
            f"| Decided by guard / rules / AI | {s['decided_by']['guard']} / "
            f"{s['decided_by']['rules']} / {s['decided_by']['llm']} |",
            f"| Rules parse, median | {_ms_text(s['rules_median_ms'])} |",
            f"| AI call, first (cold if the model wasn't loaded) | {_ms_text(s['llm_first_ms'])} |",
            f"| AI call, warm median | {_ms_text(s['llm_warm_median_ms'])} |",
        ]
        if s["n_intents"] or s["n_clarify"] or s["n_not_built"] or s["n_corrections"]:
            lines += [
                f"| Intent correct (summary / breakdown) | {_pct(s['intent_exact'], s['n_intents'])} |",
                f"| Clarification correct (right options) "
                f"| {_pct(s['clarify_exact'], s['n_clarify'])} |",
                f"| Not-built reply correct | {_pct(s['not_built_exact'], s['n_not_built'])} |",
                f"| Corrections correct (right words, none wrong) "
                f"| {_pct(s['corrections_ok'], s['n_corrections'])} |",
            ]
        if mode != config.ASK_MODE_RULES:
            o = s["overrides"]
            lines += [
                f"| Answers where the evidence check dropped something | {s['dropped']} |",
                f"| Question's range overrode the AI's | {s['n_overrides']} "
                f"(fixed {o['fixed']}, broke {o['broke']}, neutral {o['neutral']}, "
                f"n/a {o['n/a']}) |",
            ]
        if s["before_exact"] is not None:
            lines += [
                "",
                "AI only, the same model replies with and without the evidence check:",
                "",
                "| Measure | Before the check (raw form) | After the check |",
                "|---|---|---|",
                f"| Exact match | {_pct(s['before_exact'], s['n'])} | {_pct(s['exact'], s['n'])} |",
                f"| **False accepts** | **{_pct(s['before_false_accepts'], s['n'])}** "
                f"| **{_pct(s['false_accepts'], s['n'])}** |",
            ]
    lines += ["", "## False accepts", ""]
    false = [r for r in records if r["false_accept"]]
    if not false:
        lines.append("None.")
    else:
        lines += [
            "| Mode | Id | Question | Expected | Understood as | Reply |",
            "|---|---|---|---|---|---|",
        ]
        for r in false:
            e, a = r["expected"], r["actual"]
            exp = e["outcome"] if e["outcome"] != "query" else (
                f"{e['metric']} · {e.get('activity')} · {e['range']}"
            )
            act = f"{a['metric']} · {a['activity']} · {a['range']}" if (
                a["outcome"] == pipeline.OUTCOME_QUERY
            ) else f"{a['outcome']} · {a['range']}"
            lines.append(
                f"| {r['mode']} | {r['id']} | {_cell(r['question'])} | {_cell(exp)} "
                f"| {_cell(act)} | {_cell(r['reply'])} |"
            )
    return "\n".join(lines) + "\n"


# -- live part -----------------------------------------------------------------


class _ModelCall:
    """answer_question()'s `llm`: one call through llm/client.py on the
    eval's own event loop (the client is tied to the loop it first ran on).
    Keeps the last response's Ollama timings. The same messages asked again
    in one run get the first reply back without calling the model, so the
    "before the evidence check" answer uses exactly the same reply."""

    def __init__(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop
        self.last_timing: dict = {}
        self._replies: dict[str, str] = {}

    def __call__(self, messages: list[dict], format: dict) -> str:
        key = json.dumps([messages, format], sort_keys=True)
        if key in self._replies:
            return self._replies[key]
        self.last_timing = {}
        try:
            response = self._loop.run_until_complete(
                asyncio.wait_for(
                    llm_client.chat_response(messages, llm_client.ASK_OPTIONS, format),
                    LLM_TIMEOUT_S,
                )
            )
        except TimeoutError:
            raise
        except llm_client.LLM_ERRORS as exc:
            raise LlmUnavailable(str(exc)) from exc
        self.last_timing = {"load_duration": getattr(response, "load_duration", None)}
        self._replies[key] = response.message.content or ""
        return self._replies[key]


def evaluate(questions: list[dict], db_path: Path, now: datetime.datetime, modes: list[str],
             loop: asyncio.AbstractEventLoop | None) -> list[dict]:
    """Runs every question in every mode. `loop` is None when Ollama isn't
    available (then only modes that don't need it may be given)."""
    model = _ModelCall(loop) if loop is not None else None
    records = []
    for mode in modes:
        for q in questions:
            if model is not None:
                model.last_timing = {}
            answer = pipeline.answer_question(
                q["question"], conn_or_path=db_path, now=now, mode=mode, llm=model
            )
            timing = model.last_timing if model is not None else {}
            before = None
            if mode == config.ASK_MODE_LLM:
                # The same (cached) model reply, answered without the
                # evidence check: the AI on its own.
                before = pipeline.answer_question(
                    q["question"], conn_or_path=db_path, now=now, mode=mode, llm=model,
                    check_evidence=False,
                )
            records.append(run_record(mode, q, answer, timing, before))
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS,
                        help=f"question file (default {DEFAULT_QUESTIONS.name})")
    parser.add_argument("--now", default=DEFAULT_NOW,
                        help=f"pinned local time, ISO format (default {DEFAULT_NOW})")
    parser.add_argument("--tz", default=DEFAULT_TZ, help=f"timezone for --now (default {DEFAULT_TZ})")
    parser.add_argument("--modes", default=",".join(MODES),
                        help=f"comma-separated, from {', '.join(MODES)} (default all)")
    parser.add_argument("--data-seed", type=int, default=0,
                        help="seed for the sessions' values (default 0)")
    parser.add_argument(
        "--db",
        help="new database file for the sessions; refused if it already exists, "
        "unless --overwrite is given (default: a temporary file)",
    )
    parser.add_argument("--overwrite", action="store_true",
                        help="let --db replace an existing file (and its -wal/-shm/-journal)")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR,
                        help=f"results root (default {RESULTS_DIR})")
    args = parser.parse_args(argv)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    if not modes or any(m not in MODES for m in modes):
        parser.error(f"--modes must be a comma-separated list from {', '.join(MODES)}")

    if args.db:
        db_path = Path(args.db)
    else:
        db_path = Path(tempfile.mkdtemp(prefix="heartwatch-eval-ask-")) / "heartwatch_eval.db"
    try:
        # The default temporary folder is brand new, so only a --db path
        # can already be taken.
        check_destination(db_path, args.overwrite or not args.db)
    except (RealDatabaseError, DestinationExistsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        questions = load_questions(args.questions)
        now = parse_now(args.now, args.tz)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # Rejected replies are recorded in runs.jsonl; don't also log each one.
    logging.getLogger("heartwatch.llm").setLevel(logging.CRITICAL)

    target = build_eval_db(db_path, now, args.data_seed)
    loop = None
    server = {"model": llm_client.MODEL_NAME, "digest": None, "version": None}
    skipped: list[str] = []
    needs_model = [m for m in modes if m != config.ASK_MODE_RULES]
    if needs_model:
        loop = asyncio.new_event_loop()
        health = loop.run_until_complete(llm_client.health_check())
        if health.ready:
            server = loop.run_until_complete(llm_client.server_info())
        else:
            print(f"Ollama unavailable ({health.message}); skipping {', '.join(needs_model)}.")
            skipped = needs_model
            loop.close()
            loop = None
    modes_run = [m for m in modes if m not in skipped]
    try:
        records = evaluate(questions, target, now, modes_run, loop)
    finally:
        if loop is not None:
            loop.close()

    meta = {
        "model": server["model"],
        "digest": server["digest"],
        "ollama_version": server["version"],
        "ollama_python_package": importlib.metadata.version("ollama"),
        "options": dict(llm_client.ASK_OPTIONS),
        "llm_timeout_s": LLM_TIMEOUT_S,
        "modes_run": modes_run,
        "skipped_modes": skipped,
        "now": now.isoformat(),
        "tz": args.tz,
        "data_seed": args.data_seed,
        "questions_file": _display_path(args.questions),
        "questions_sha256": hashlib.sha256(args.questions.read_bytes()).hexdigest(),
        "n_questions": len(questions),
    }
    out_dir = make_output_dir(args.out, prefix="ask-")
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    with open(out_dir / "runs.jsonl", "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    (out_dir / "summary.md").write_text(format_summary_md(meta, records))

    for mode in modes_run:
        s = mode_stats([r for r in records if r["mode"] == mode])
        print(f"{mode}: exact {s['exact']}/{s['n']}, false accepts {s['false_accepts']}")
    print(f"Results: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
