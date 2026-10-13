"""Repeatable evaluation of the AI session summaries, for the writeup.

Plain-language overview: this re-runs the side-by-side experiment behind
"the plain summary is the default" -- ask the local model to summarize the
same demo session N times, run every reply through the app's own checks,
and save everything so each reply can be labelled by hand. Nothing about it
is random between runs of the script:

- The session is built by scripts/demo_db.py from `--data-seed`, on a new
  throwaway database file (the real database is never opened), so the
  facts are identical every time.
- Run i asks the model with Ollama's `seed` option set to `--seed-base + i`.
  With a fixed seed gemma3:1b's reply is byte-for-byte the same, so running
  the script twice gives the same replies; different seeds give the runs
  some variety.

The request is exactly the app's (llm/prompts.py's
build_session_summary_messages() and SUMMARY_OPTIONS, plus the seed), and
each reply goes through the app's own validate.screen_summary(). It runs
the same whatever HEARTWATCH_SUMMARY_MODE is set to.

Run from the repo root, with the venv active and Ollama running:

    python -m heartwatchapp.scripts.eval_summaries                  # 10 runs
    python -m heartwatchapp.scripts.eval_summaries --runs 20 --seed-base 100

Writes heartwatchapp/eval_results/<YYYYMMDD-HHMMSS>/:
    runs.jsonl  one JSON object per run (seed, raw reply, text after
                strip_preamble(), pass/fail, rejecting check, timings)
    meta.json   recorded once: model, digest, Ollama version, options, data
                seed, the fact lines sent, and the plain fallback
    summary.md  the facts and fallback, then one table row per run with two
                empty columns to fill in by hand

All the live-model work happens inside one asyncio.run() (llm/client.py's
AsyncClient is tied to the event loop it first ran on), and nothing here
imports Qt.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import logging
import sys
import tempfile
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import ollama  # noqa: E402

from heartwatchapp.data import stats as stats_mod  # noqa: E402
from heartwatchapp.llm import client as llm_client  # noqa: E402
from heartwatchapp.llm import prompts, validate  # noqa: E402
from heartwatchapp.scripts.demo_db import (  # noqa: E402
    RealDatabaseError,
    build_demo_db,
    ensure_not_real_db,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "eval_results"

# validate.py's check names, as the writeup refers to them.
CHECK_LABELS = {
    validate.CHECK_EMPTY: "empty reply",
    validate.CHECK_BLOCKED: "A (blocked vocabulary)",
    validate.CHECK_HEDGED: "B (hedged number)",
    validate.CHECK_UNGROUNDED: "C (number not in the input)",
    validate.CHECK_LABEL_PAIRING: "D (label-number pairing)",
}


class OllamaUnavailableError(RuntimeError):
    """Ollama isn't reachable, or the model isn't pulled."""


# -- pure parts (unit-tested without Ollama) ----------------------------------


def run_seeds(seed_base: int, runs: int) -> list[int]:
    """Run i uses seed seed_base + i."""
    return [seed_base + i for i in range(runs)]


def run_options(seed: int) -> dict:
    """The app's summary options, plus a fixed seed."""
    return {**llm_client.SUMMARY_OPTIONS, "seed": seed}


def build_session(db_path: str | Path, data_seed: int) -> dict:
    """Builds the demo session from `data_seed` on a new database at
    `db_path` (never the real one -- RealDatabaseError) and returns its
    compute_session_stats() facts."""
    target, session_id = build_demo_db(db_path, copy_real=False, data_seed=data_seed)
    return stats_mod.compute_session_stats(session_id, path=target)


def session_facts(session_stats: dict) -> dict:
    """What the model is told, and what the app would show instead."""
    return {
        "fact_lines": prompts.session_fact_lines(session_stats)
        + list(prompts.activity_fact_lines(session_stats).values()),
        "plain_fallback": stats_mod.format_stats_plain(session_stats),
        "confidence_caveat": stats_mod.confidence_caveat(session_stats),
    }


def run_record(
    seed: int, raw_text: str, result: validate.ValidationResult, latency_s: float, timings: dict
) -> dict:
    """One line of runs.jsonl. `timings` holds Ollama's own durations
    (nanoseconds) where it reported them."""
    return {
        "seed": seed,
        "raw_reply": raw_text,
        "stripped_text": result.text,
        "passed": result.passed,
        "rejected_by": None if result.passed else CHECK_LABELS.get(result.check, result.check),
        "matched": result.matched,
        "latency_s": round(latency_s, 3),
        "load_duration_ns": timings.get("load_duration"),
        "eval_duration_ns": timings.get("eval_duration"),
    }


def _cell(text: str | None) -> str:
    """A value made safe for one Markdown table cell."""
    if text is None:
        return ""
    return str(text).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " <br> ")


def format_summary_md(meta: dict, records: list[dict]) -> str:
    """summary.md: the session's facts and plain fallback, the pass rate,
    then one row per run with two columns left empty for hand-labelling."""
    passed = sum(1 for r in records if r["passed"])
    lines = [
        "# Summary evaluation",
        "",
        f"- Model: `{meta['model']}` (digest `{meta['digest']}`), Ollama {meta['ollama_version']}",
        f"- Options: `{json.dumps(meta['options_without_seed'], sort_keys=True)}` "
        f"+ seed {meta['seed_base']}..{meta['seed_base'] + len(records) - 1}",
        f"- Data seed: {meta['data_seed']}",
        f"- Passed validation: {passed} of {len(records)}",
        "",
        "## Fact lines sent to the model",
        "",
        *[f"    {line}" for line in meta["fact_lines"]],
        "",
        "## Plain fallback for the same session",
        "",
        meta["plain_fallback"],
        "",
    ]
    if meta.get("confidence_caveat"):
        lines += [f"(App caveat shown under either: {meta['confidence_caveat']})", ""]
    lines += [
        "## Runs",
        "",
        "| Run | Seed | Passed | Rejected by | Latency (s) | Reply (after strip_preamble) "
        "| False statement? | Just restated the fact lines? |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(records):
        rejected = r["rejected_by"] or ""
        if r["matched"]:
            rejected += f": {r['matched']}"
        lines.append(
            f"| {i} | {r['seed']} | {'yes' if r['passed'] else 'no'} | {_cell(rejected)} "
            f"| {r['latency_s']:.2f} | {_cell(r['stripped_text'])} |  |  |"
        )
    return "\n".join(lines) + "\n"


def make_output_dir(root: Path, now: datetime.datetime | None = None) -> Path:
    """root/<YYYYMMDD-HHMMSS>/, with -2, -3, ... added if that already exists."""
    stamp = (now or datetime.datetime.now()).strftime("%Y%m%d-%H%M%S")
    candidate, n = root / stamp, 1
    while candidate.exists():
        n += 1
        candidate = root / f"{stamp}-{n}"
    candidate.mkdir(parents=True)
    return candidate


# -- live part -----------------------------------------------------------------


async def _evaluate(messages: list[dict], session_stats: dict, seeds: list[int]):
    """Everything that talks to Ollama, in one event loop. Returns
    (server info, run records)."""
    health = await llm_client.health_check()
    if not health.ready:
        raise OllamaUnavailableError(health.message)
    try:
        server = await llm_client.server_info()
        records = []
        for seed in seeds:
            started = time.perf_counter()
            response = await llm_client.chat_response(messages, options=run_options(seed))
            latency_s = time.perf_counter() - started
            raw_text = response.message.content or ""
            timings = {
                key: getattr(response, key, None) for key in ("load_duration", "eval_duration")
            }
            result = validate.screen_summary(raw_text, session_stats)
            records.append(run_record(seed, raw_text, result, latency_s, timings))
    except ollama.ResponseError as exc:
        raise OllamaUnavailableError(f"Ollama error: {exc}") from exc
    except llm_client.CONNECTION_ERRORS as exc:
        raise OllamaUnavailableError(f"Ollama isn't reachable ({exc})") from exc
    return server, records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=int, default=10, help="replies to generate (default 10)")
    parser.add_argument("--seed-base", type=int, default=0, help="run i uses seed S + i (default 0)")
    parser.add_argument(
        "--data-seed", type=int, default=0, help="seed for the demo session's values (default 0)"
    )
    parser.add_argument(
        "--db", help="new database file for the demo session (default: a temporary file)"
    )
    parser.add_argument(
        "--out", type=Path, default=RESULTS_DIR, help=f"results root (default {RESULTS_DIR})"
    )
    args = parser.parse_args(argv)
    if args.runs < 1:
        parser.error("--runs must be at least 1")

    if args.db:
        db_path = Path(args.db)
    else:
        db_path = Path(tempfile.mkdtemp(prefix="heartwatch-eval-")) / "heartwatch_eval.db"
    try:
        ensure_not_real_db(db_path)
    except RealDatabaseError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # Every rejection is recorded in runs.jsonl; don't also print each one.
    logging.getLogger("heartwatch.llm").setLevel(logging.ERROR)

    session_stats = build_session(db_path, args.data_seed)
    messages = prompts.build_session_summary_messages(session_stats)
    seeds = run_seeds(args.seed_base, args.runs)
    try:
        server, records = asyncio.run(_evaluate(messages, session_stats, seeds))
    except OllamaUnavailableError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    meta = {
        "model": server["model"],
        "digest": server["digest"],
        "ollama_version": server["version"],
        "options_without_seed": dict(llm_client.SUMMARY_OPTIONS),
        "seed_base": args.seed_base,
        "runs": args.runs,
        "data_seed": args.data_seed,
        **session_facts(session_stats),
    }
    out_dir = make_output_dir(args.out)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    with open(out_dir / "runs.jsonl", "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    (out_dir / "summary.md").write_text(format_summary_md(meta, records))

    passed = sum(1 for r in records if r["passed"])
    print(f"{passed} of {len(records)} replies passed validation. Results: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
