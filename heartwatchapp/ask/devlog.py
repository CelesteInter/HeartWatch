"""Developer tools for the "Ask about your data" bar (handoff 9a, 3.8).

Plain-language overview: with HEARTWATCH_ASK_DEV=1 (config.ask_dev()) the
bar shows who answered at the end of the "what I understood" line
("· rules", "· AI", "· guard", "· rules (corrected)") and appends one JSON
line per question to a local log, so the questions people actually ask
can be reviewed when tuning the keyword parser. Off by default; when off,
nothing is written and the understood line is unchanged.

The log never records the reply or any number from the data -- only the
question, how it was corrected and understood, what the evidence check
dropped, and how long it took. It lives in heartwatchapp/dev_logs/, which
heartwatchapp/.gitignore keeps out of git. Tests pass their own path.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

from . import guard, typos
from .pipeline import OUTCOME_QUERY, Answer

LOG_PATH = Path(__file__).resolve().parent.parent / "dev_logs" / "ask_questions.jsonl"

_WHO = {"guard": "guard", "rules": "rules", "llm": "AI"}


def marker(answer: Answer) -> str:
    """"rules", "AI", "guard", or "rules (corrected)" etc. when a typo was
    corrected first."""
    who = _WHO.get(answer.source, answer.source)
    return f"{who} (corrected)" if answer.corrections else who


def with_marker(understood: str | None, answer: Answer) -> str:
    """The understood line with "· <who answered>" on the end."""
    return f"{understood} · {marker(answer)}" if understood else marker(answer)


def _form(answer: Answer) -> dict:
    """What was understood, without any number from the data."""
    if answer.outcome == OUTCOME_QUERY and answer.request is not None:
        r = answer.request
        return {"metric": r.metric, "activity": r.activity, "range": r.range}
    parse = answer.parse
    form: dict = {}
    if parse is not None and parse.intent != "metric":
        form = {"intent": parse.intent, "range": parse.intent_range}
    if answer.suggestions and answer.outcome == "clarify":
        form["options"] = [o.label for o in answer.suggestions]
    if parse is not None and parse.kind:
        form["kind"] = parse.kind
    return form


def entry(question: str, answer: Answer, now: datetime.datetime | None = None) -> dict:
    """One log line's fields (handoff 9a, 3.8)."""
    now = now or datetime.datetime.now().astimezone()
    return {
        "timestamp": now.isoformat(timespec="seconds"),
        "question": question,
        "corrected": typos.correct(question, guard.GUARD_WORDS).text,
        "corrections": [list(c) for c in answer.corrections],
        "decided_by": answer.source,
        "outcome": answer.outcome,
        "form": _form(answer),
        "dropped": list(answer.parse.dropped) if answer.parse is not None else [],
        "latency_ms": round(sum(answer.timings.values()), 3),
    }


def append(question: str, answer: Answer, path: Path | None = None) -> None:
    """Appends one line to the log (LOG_PATH unless `path` is given),
    creating its folder if needed."""
    path = LOG_PATH if path is None else Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry(question, answer)) + "\n")
