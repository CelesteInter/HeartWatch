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

## AI session summaries (optional, local-only)

After a recorded session, the Sessions view can show two or three plain
sentences describing it, generated on your own machine by
[Ollama](https://ollama.com) running `gemma3:1b` -- nothing about a session
is sent anywhere over the internet. **The app works fully without this**;
if Ollama isn't installed or isn't running, the session detail view just
shows the same numbers as plain text instead of AI-written sentences. See
`llm/client.py`'s module docstring and `llm/prompts.py`'s `SYSTEM_PROMPT`
for the exact constraints placed on the model.

To turn it on:

1. Install the Ollama desktop app from [ollama.com](https://ollama.com).
2. Pull the model: `ollama pull gemma3:1b`
3. Create the venv and install requirements as described above (`ollama`
   is already listed in `requirements.txt`).
4. Run the app as usual. The topbar's "AI" pill shows whether it found
   Ollama and the model; hover it for details if it says "unavailable."

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
- **Short sessions skip the model entirely.** A session with no activity
  predictions, fewer than 60 heart-rate samples, or shorter than 60
  seconds gets the plain numbers plus "Not enough data for a generated
  summary yet." (thresholds: `MIN_*` constants in `data/stats.py`).
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
- **The fallback reads like a summary too.** Whenever there's no AI text,
  the app writes its own sentences from the same facts, e.g. "The session
  lasted 6 minutes 20 seconds. Average heart rate was 93 bpm and peak
  heart rate was 111 bpm, reached during Walking. Time by activity:
  Sitting 2 minutes 10 seconds, Walking 2 minutes 30 seconds, Standing 1
  minute 40 seconds." Comparing it with a passing AI summary is the
  fairest test of whether the model earns its place here.
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

### Trying the AI summaries

1. Run the unit tests (no Ollama needed), from the repo root:
   `python -m unittest discover -s heartwatchapp/tests -t .`
2. Launch the app normally. Real recorded sessions have no activity
   predictions yet (that's milestone 6), so they take the "not enough
   data" path -- that's expected.
3. To see the AI path, either:
   - run `python -m heartwatchapp.scripts.demo_session`, which copies the
     database to a temporary file, adds one data-rich session to the
     copy, launches the app on the copy with the `heartwatch.llm` log at
     DEBUG, and prints the new session's number -- the real database is
     never written (the script refuses to run against it); or
   - click Settings -> Developer -> **Seed demo session** (this *does*
     write to your real database -- seeded sessions include dummy
     predictions, `model_version "demo-dummy"`), or leave the app running
     on the Live Monitor for more than a minute.
4. In Sessions, double-click the session. The terminal shows each pass or
   rejection and why.

To point the app at any other database file, set `HEARTWATCH_DB_PATH`
before launching; with it unset the app uses `data/heartwatch.db` as usual.

**Measured benchmark** (Dylan's MacBook Air, Apple M2, 8 GB):
`gemma3:1b` -- 883 MB, 100% GPU-resident on Apple M2, 2048 context. Time
to first token 0.04s; warm generation 0.11–0.94s; cold load 2.31s.
Selected to fit an 8 GB development machine.

## Layout

```
heartwatchapp/
├── main.py              app entry point
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
│   └── demo_session.py      launch the app on a DB copy with a data-rich demo session
└── tests/
    ├── test_stats.py        unit tests for data/stats.py (no live Ollama needed)
    ├── test_validate.py     unit tests for llm/validate.py (no live Ollama needed)
    └── test_demo_session.py demo tooling: real-DB guard, seeded predictions
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
| — | Local LLM session summaries (Ollama / gemma3:1b) | done — needs Ollama installed to activate, degrades to plain stats otherwise |

All Dashboard / Live Monitor values are hardcoded demo data or dummy generators
until step 5–6.
