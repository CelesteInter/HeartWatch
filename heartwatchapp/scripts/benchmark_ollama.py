"""Standalone diagnostic: is gemma3:1b viable on this 8 GB M2 machine?

Plain-language overview: this script asks the same questions Dylan would
ask by hand -- is Ollama running, is the model downloaded, how long does a
reply take, and does streaming actually get text on screen sooner? It
writes nothing to the app's database and imports nothing from the GUI, so
it's safe to run any time, repeatedly, without side effects.

Run from the repo root, with the venv active:

    python -m heartwatchapp.scripts.benchmark_ollama
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import ollama  # noqa: E402

from heartwatchapp.llm import prompts  # noqa: E402

MODEL_NAME = "gemma3:1b"
NUM_CTX = 2048

# A realistic hard-coded stats dict, shaped exactly like
# data/stats.py's compute_session_stats() output.
SAMPLE_STATS = {
    "duration_s": 642.0,
    "avg_hr": 118.4,
    "max_hr": 151,
    "activity_seconds": {"Walking": 420.0, "Standing": 132.0, "Sitting": 90.0},
    "label": "Walking",
    "prediction_count": 40,
    "low_confidence_count": 3,
}


def _ollama_rss_kb() -> int | None:
    """Total resident memory (KB) of any process with 'ollama' in its
    command line, via `ps` -- no psutil dependency. None if unavailable."""
    try:
        out = subprocess.run(
            ["ps", "-axo", "rss,comm"], capture_output=True, text=True, timeout=5
        ).stdout
    except Exception:
        return None
    rows = (line.split(None, 1) for line in out.splitlines()[1:])
    matches = [int(rss) for rss, comm in rows if "ollama" in comm.lower() and rss.isdigit()]
    return sum(matches) if matches else None


def main() -> int:
    client = ollama.Client()

    print("== 1. Daemon + model check ==")
    try:
        models = {m.model for m in client.list().models if m.model}
    except Exception as exc:
        print(f"Ollama daemon is not reachable: {exc}")
        print("Start the Ollama app, then re-run this script.")
        return 1
    print(f"Daemon reachable. Local models: {sorted(models) or '(none)'}")
    print(f"{MODEL_NAME} present: {MODEL_NAME in models}")
    if MODEL_NAME not in models:
        print(f"Run `ollama pull {MODEL_NAME}` before continuing.")
        return 1

    rss_before = _ollama_rss_kb()

    print("\n== 2. Non-streaming chat ==")
    t0 = time.monotonic()
    client.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Say hello in one short sentence."}],
        options={"num_ctx": NUM_CTX},
    )
    print(f"Latency: {time.monotonic() - t0:.2f}s")

    print("\n== 3. Streaming chat (time to first token vs. total) ==")
    t0 = time.monotonic()
    first_token_at = None
    chunks = []
    for part in client.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Say hello in one short sentence."}],
        stream=True,
        options={"num_ctx": NUM_CTX},
    ):
        if first_token_at is None and part.message.content:
            first_token_at = time.monotonic() - t0
        chunks.append(part.message.content or "")
    total = time.monotonic() - t0
    print(f"Time to first token: {first_token_at:.2f}s")
    print(f"Total time: {total:.2f}s")
    print(f"Reply: {''.join(chunks)!r}")

    print("\n== 4. Realistic session-summary prompt ==")
    messages = [
        {"role": "system", "content": prompts.SYSTEM_PROMPT},
        {"role": "user", "content": prompts.render_session_summary_prompt(SAMPLE_STATS)},
    ]
    t0 = time.monotonic()
    response = client.chat(model=MODEL_NAME, messages=messages, options={"num_ctx": NUM_CTX})
    print(f"Latency: {time.monotonic() - t0:.2f}s")
    print(f"Output:\n{response.message.content}")

    print("\n== 5. Ollama process memory ==")
    rss_after = _ollama_rss_kb()
    if rss_before is not None and rss_after is not None:
        print(f"Before: {rss_before / 1024:.0f} MB   After: {rss_after / 1024:.0f} MB")
    else:
        print("Could not read process memory via `ps` -- watch Activity Monitor instead.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
