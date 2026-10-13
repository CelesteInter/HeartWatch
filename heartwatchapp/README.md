# HeartWatch — Host App

PySide6 (Qt6) desktop application for the HeartWatch capstone. Sits between
Apple Health (clean card layout) and Garmin Connect (data-dense, metric-forward).

## Run

Quickest — the launcher creates the venv on first run, then reuses it:

```bash
./heartwatchapp/run.sh
```

Manual equivalent (venv lives at `heartwatchapp/.venv/`, gitignored):

```bash
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"   # repo root
python3 -m venv heartwatchapp/.venv
source heartwatchapp/.venv/bin/activate
pip install -r heartwatchapp/requirements.txt
python -m heartwatchapp.main
```

After the first setup, each new terminal session is just:

```bash
source heartwatchapp/.venv/bin/activate
python -m heartwatchapp.main
```

Always run from the **repo root**, not from inside `heartwatchapp/`.

## Session summaries

Double-clicking a session shows a short plain-language summary of it. **By
default the app writes this summary itself**, in Python
(`data/stats.py`'s `format_stats_plain()`), and shows it straight away --
Ollama isn't contacted for it and there's no "Generating summary…" wait.

The summary mode is set by the `HEARTWATCH_SUMMARY_MODE` environment
variable (read by `config.py`, each time a session is opened):

| Value | What the session detail view shows |
|---|---|
| `plain` (default; also used if unset, empty, or unrecognized) | The app's own sentences, instantly |
| `llm` | Sentences phrased by a local model (below), checked before they're shown; the plain summary if the model is unavailable or its reply is rejected |

Why plain is the default: in the side-by-side test the plain summary was
correct every time, while about half of the AI summaries that passed every
check below still said something false. The AI path is kept for the
capstone writeup (see `scripts/eval_summaries.py` under Development notes).

### AI summaries (optional, local-only)

With `HEARTWATCH_SUMMARY_MODE=llm`, the Sessions view asks
[Ollama](https://ollama.com) running `gemma3:1b` on your own machine to
phrase the summary in two or three sentences -- nothing about a session is
sent anywhere over the internet. **The app works fully without this**; if
Ollama isn't installed or isn't running, the session detail view just
shows the plain summary instead. See `llm/client.py`'s module docstring and
`llm/prompts.py`'s `SYSTEM_PROMPT` for the exact constraints placed on the
model.

To turn it on:

1. Install the Ollama desktop app from [ollama.com](https://ollama.com).
2. Pull the model: `ollama pull gemma3:1b`
3. Create the venv and install requirements as described above (`ollama`
   is already listed in `requirements.txt`).
4. Launch with `HEARTWATCH_SUMMARY_MODE=llm python -m heartwatchapp.main`.
   The topbar's "AI" pill shows whether it found Ollama and the model;
   hover it for details if it says "unavailable." (The pill is checked at
   startup in either mode.)

### What the model is for

The model **never computes anything**. Python already worked out every
number it sees -- session duration, average/peak heart rate, time spent
per detected activity -- from the SQLite database (`data/stats.py`). All
the model does is turn that short list of numbers into two or three plain
English sentences. It never looks at raw sensor data, never overrides the
activity classifier, and it is explicitly told not to give medical or
health advice, diagnose anything, or say whether a number is good or bad
-- just to describe what happened. If you want to sanity-check whether
this is viable on your machine before touching any of this, run
`scripts/benchmark_ollama.py` (see below).

### LLM behavior and safeguards

A 1B-parameter model does not reliably follow instructions, so the rules
below are enforced by code, not just asked for in the prompt. Each one
holds even when the model misbehaves.

- **The model only phrases facts Python already worked out.** Python
  computes every number (`data/stats.py`) and writes it out exactly as it
  should appear in the summary -- e.g. "4 minutes 12 seconds", never "252
  seconds" -- so the model only ever copies numbers. Python also works
  out how the numbers relate: the time spent in each activity and which
  activity the peak heart rate happened during. These go to the model as
  short fact lines ("Time Walking: 2 minutes 30 seconds", "Peak heart rate
  occurred during: Walking"), so there is no relationship left for it to
  guess. It never computes, never interprets what a number means for your
  body, and never gives advice.
- **Short sessions skip the model entirely.** In `llm` mode, a session
  with no activity predictions, fewer than 60 heart-rate samples, or
  shorter than 60 seconds gets the plain numbers plus "Not enough data for
  a generated summary yet." (thresholds: `MIN_*` constants in
  `data/stats.py`). In `plain` mode there's no generated summary to be
  missing, so that note never appears.
- **Every reply is checked before it's shown** (`llm/validate.py`). A
  chatty opener like "Okay, here's the summary:" is stripped first; then
  the reply is rejected if it contains blocked wording (medical
  interpretation, advice, praise, or comparisons to a "usual" the model
  was never shown), a hedged number ("around 89 bpm" -- every number is
  exact), any number that wasn't in the input, or a number attached to
  the wrong activity. That last check goes sentence by sentence: a
  sentence about one activity may only use that activity's time or the
  session-wide numbers, a sentence saying when the peak happened must
  name the activity Python reported, and naming an activity the session
  doesn't have is rejected. A rejected reply is never displayed; the app
  shows the plain-language fallback instead. The whole reply is generated
  before anything is shown (no streaming), with a 10-second timeout that
  also falls back.
- **The plain summary reads like a summary too.** It's what the app shows
  by default, and whenever there's no AI text: the app's own sentences
  from the same facts, e.g. "The session lasted 6 minutes 20 seconds.
  Average heart rate was 93 bpm and peak heart rate was 111 bpm, reached
  during Walking. Time by activity: Walking 2 minutes 30 seconds, Sitting
  2 minutes 10 seconds, Standing 1 minute 40 seconds." Activities are
  listed most time first (ties: whichever came first in the session), in
  both this and the facts sent to the model; a session with only one
  activity says "Activity: Walking for the whole session." instead.
- **The AI pill follows Ollama at runtime.** It is set at startup, flips
  to "unavailable" if a summary request can't reach Ollama (or the model
  is missing), and back to "ready" when a later request gets an answer.
  A reply rejected by the checks above doesn't count as Ollama failing.
- **Classification confidence is shown by the app, not the model.** If
  the activity classifier was unsure during part of the session, the app
  adds its own fixed line, "Some windows had low classification
  confidence." The model is never told about confidence, because it has
  been seen describing classifier uncertainty as something the user did.
- **Rejections are logged** on the `heartwatch.llm` logger (printed to
  the terminal) with the session id, which check failed, what matched,
  and the full rejected text. Passes are logged at DEBUG; set the level
  in `main.py` to DEBUG to see them and work out a rejection rate.

Known gaps: numbers written as words ("two activities") are not checked;
activities are only recognized by their label ("walking" counts,
"walked" doesn't); and a sentence naming two or more activities isn't
checked for which number goes with which (only the peak rule applies).

### Trying the summaries

1. Run the unit tests (no Ollama needed), from the repo root:
   `python -m unittest discover -s heartwatchapp/tests -t .`
2. Launch the app normally. Real recorded sessions have no activity
   predictions yet (that's milestone 6), so their plain summary has no
   activity breakdown, and in `llm` mode they take the "not enough data"
   path -- that's expected.
3. To see a data-rich session, either:
   - run `python -m heartwatchapp.scripts.demo_session` (plain summary)
     or `HEARTWATCH_SUMMARY_MODE=llm python -m
     heartwatchapp.scripts.demo_session` (AI path), which copies the
     database to a temporary file, adds one data-rich session to the
     copy, launches the app on the copy with the `heartwatch.llm` log at
     DEBUG, and prints the new session's number -- the real database is
     never written (the script refuses to run against it); or
   - click Settings -> Developer -> **Seed demo session** (this *does*
     write to your real database -- seeded sessions include dummy
     predictions, `model_version "demo-dummy"`), or leave the app running
     on the Live Monitor for more than a minute.
4. In Sessions, double-click the session. In `llm` mode the terminal
   shows each pass or rejection and why.

To point the app at any other database file, set `HEARTWATCH_DB_PATH`
before launching; with it unset the app uses `data/heartwatch.db` as usual.

**Measured benchmark** (Dylan's MacBook Air, Apple M2, 8 GB):
`gemma3:1b` -- 883 MB, 100% GPU-resident on Apple M2, 2048 context. Time
to first token 0.04s; warm generation 0.11–0.94s; cold load 2.31s.
Selected to fit an 8 GB development machine.

## Development notes

Things we learned the hard way while building the AI summaries. Read this
before touching `llm/` or the database.

| Don't | Do instead |
|---|---|
| Quit Ollama with `osascript` or force-kill it | Point a client at a dead port (`127.0.0.1:9`) to simulate Ollama being down, or quit the menu-bar app by hand, run `ollama serve` in a terminal, and Ctrl-C it |
| Reuse the module-level `AsyncClient` across several `asyncio.run()` calls | Keep live-model work inside one `asyncio.run()`, and run Qt checks in a separate process. The client is tied to its event loop. |
| Catch only `httpx.RequestError` for "Ollama down" | Catch `CONNECTION_ERRORS` (includes built-in `ConnectionError`) |
| Check numbers as substrings in tests ("134" contains "4") | Compare digit tokens from `re.findall(r"\d+")` |
| Open the real DB with SQLite `mode=ro` | Use copies via `HEARTWATCH_DB_PATH`. WAL mode still touches the `-shm`/`-wal` files. |
| Use pytest | It isn't installed. Use stdlib `unittest`. |
| Update test expectations by copying new outputs | Recompute expected values by hand from the rule being tested |

**Summary mode.** `HEARTWATCH_SUMMARY_MODE=plain` (the default) shows the
app's own summary; `llm` brings back the AI path. Plain is the default
because it was correct every time in the side-by-side test, while about
half of the AI summaries that passed validation still said something false.

**Prediction windows.** An activity prediction's timestamp marks the *end*
of the 5-second window of sensor data it was computed from, and it owns
the half-open interval before it: a prediction at `t` covers
`[max(t_prev, t - 5 s), t)`. So a heart-rate sample at exactly a
transition belongs to the new activity, a missing prediction leaves its 5
seconds unattributed, and nothing after the last prediction counts toward
any activity. The rule lives in `data/stats.py` (`PREDICTION_WINDOW_MS`,
`PREDICTION_TIMESTAMP_MARKS`, `_activity_segments()`); every data generator
already stamps predictions this way.

**Evaluating the AI summaries.** From the repo root, with Ollama running:

```bash
python -m heartwatchapp.scripts.eval_summaries --runs 10   # --seed-base 0 --data-seed 0 by default
```

It builds the demo session from `--data-seed` on a throwaway database (the
real one is never opened), asks the model for `--runs` summaries with the
app's exact prompt and options plus Ollama seed `--seed-base + i` for run
`i`, and runs each reply through the app's checks. Results go to
`heartwatchapp/eval_results/<YYYYMMDD-HHMMSS>/` (gitignored): `runs.jsonl`
(one line per run), `meta.json` (model digest, Ollama version, options,
facts sent, plain fallback), and `summary.md` (a table with two empty
columns to label by hand). With fixed seeds `gemma3:1b`'s replies are
byte-identical, so running it again with the same arguments reproduces
the same replies.

**Ollama facts.** Ollama runs from the macOS menu-bar app. It unloads the
model after 5 idle minutes by default, so the first request after that
takes about 3 s instead of about 1 s.

**Ollama restarts are handled.** Stopping and restarting Ollama while the
app was open was checked by hand on September 30, 2026 (`llm` mode, demo
session, `ollama serve` in a terminal), and the app recovered without a
relaunch.

## Layout

```
heartwatchapp/
├── main.py              app entry point
├── config.py            environment-variable settings (HEARTWATCH_DB_PATH, HEARTWATCH_SUMMARY_MODE)
├── ui/
│   ├── main_window.py   QMainWindow shell (sidebar · topbar · stack · AI bar)
│   ├── sidebar.py       64px icon-only nav (QToolButton + QButtonGroup)
│   ├── topbar.py        52px section title / status / connection + battery
│   ├── theme.py         design tokens + QSS (light/dark, no scattered hex)
│   ├── widgets.py       Card, MetricCard, Badge, Divider, PlaceholderState
│   ├── charts.py        Matplotlib DonutChart + scrolling LiveHRChart
│   ├── dashboard.py     Section 1 — metric cards, activity breakdown, sessions
│   ├── live_monitor.py  Section 2 — live HR + activity badge + IMU axes
│   ├── sessions.py      Section 3 — sessions list + SessionDetailDialog (chart, summary)
│   ├── ml_view.py       Section 4 — stub (pipeline summary + matrix placeholder)
│   ├── settings.py      Section 5 — stub (placeholder controls, theme toggle)
│   └── ai_bar.py        always-visible data-query bar (layout only)
├── data/
│   ├── db.py            SQLite connection + schema + query stubs
│   └── stats.py         session-summary number crunching (no LLM code)
├── llm/
│   ├── client.py        the only module that imports `ollama`
│   ├── prompts.py       system prompt + session-summary prompt text
│   ├── validate.py      checks every generated summary before it's shown
│   └── worker.py        Qt-thread bridge (sends checked replies to the GUI)
├── ml/inference.py      TFLite activity classifier stub
├── scripts/
│   ├── benchmark_ollama.py  standalone viability check, run by hand
│   ├── demo_db.py           builds the demo session on a DB copy (shared by the two below)
│   ├── demo_session.py      launch the app on a DB copy with a data-rich demo session
│   └── eval_summaries.py    repeatable AI-summary evaluation (writes eval_results/)
└── tests/
    ├── test_stats.py          unit tests for data/stats.py (no live Ollama needed)
    ├── test_validate.py       unit tests for llm/validate.py (no live Ollama needed)
    ├── test_demo_session.py   demo tooling: real-DB guard, demo facts, seeded predictions
    ├── test_summary_mode.py   HEARTWATCH_SUMMARY_MODE and the dialog's two paths
    └── test_eval_summaries.py eval script's pure parts (no live Ollama needed)
```

## Build status

| # | Milestone | State |
|---|---|---|
| 1 | App shell (window, sidebar, topbar, stack, AI bar) | done |
| 2 | Dashboard — cards / bars / donut / sessions (demo data) | done |
| 3 | Live Monitor — dummy sine HR + IMU axes; also writes dummy activity predictions (`model_version "demo-dummy"`) until milestone 6 | done |
| 4 | AI bar wired to Anthropic API | layout only |
| 5 | SQLite integration (replace demo data) | partial — Sessions, `DatabaseWriter`, and `data/stats.py` use SQLite; the Dashboard still uses hardcoded demo data |
| 6 | BLE stream from ESP32-S3 + real TFLite inference | not started |
| — | Session summaries | done — plain (Python-written) by default; AI phrasing via Ollama / gemma3:1b with `HEARTWATCH_SUMMARY_MODE=llm`, degrades to the plain summary otherwise |

All Dashboard / Live Monitor values are hardcoded demo data or dummy generators
until step 5–6.
