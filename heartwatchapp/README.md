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

Always run from the **repo root** (`HeartWatch/`), not from inside `heartwatchapp/`.

## Ask about your data

The bar at the bottom of the window answers simple questions about your
own recorded sessions, for example:

- "What was my average heart rate while running this week?"
- "How long did I walk yesterday?"
- "Peak heart rate in my most recent session"
- "How many sessions did I record in the last 30 days?"

It can answer five things -- **average heart rate, peak heart rate, time
spent in one activity (Sitting, Walking, Standing, Running), number of
sessions, and total recorded time** -- for today, yesterday, this week
(since Monday), last week (the previous Monday to Sunday), the last 7 days,
the last 30 days, all sessions, or your latest session. With no time
period it uses the last 7 days, the same as the Dashboard.

**The AI only understands; Python answers.** The question is turned into a
small form -- which number, which activity, which time period -- and then
Python looks the answer up with the same code the Dashboard uses
(`data/queries.py`) and writes the sentence (`ask/answers.py`). The model
never writes a number or a word you read. Under every answer the bar shows
what it understood, e.g. *Average heart rate · Running · this week*, so a
misunderstanding is visible straight away. If the activity times come from
the dummy classifier it also says *"Activity times are from simulated
predictions."*, by the same rule as the Dashboard.

**Health questions get a fixed reply.** Anything asking whether a number is
healthy, normal, safe, too high, higher than usual, or what you should do
gets the same reply and no number, even if it also names a metric. This
check is plain code (`ask/guard.py`) and runs before anything else.

How a question is understood (`HEARTWATCH_ASK_MODE`, read each time a
question is asked):

| Value | What happens |
|---|---|
| `hybrid` (default; also used if unset, empty, or unrecognized) | A keyword parser (`ask/rules.py`) answers simple questions instantly with no AI; only questions it isn't sure about go to the local model (`gemma3:1b` via Ollama). If the keyword parser saw exactly one clear time phrase, its time period is used even if the model picked another. |
| `rules` | Keyword parser only; Ollama is never contacted. Anything it isn't sure about gets "Sorry, I didn't understand that." |
| `llm` | Every question (after the health check) goes to the model. Used by the evaluation to compare the parsers. |

Without Ollama, `hybrid` still answers everything the keyword parser
understands; anything else gets a reply saying the AI model isn't running,
with example questions that work without it. The keyword parser only
accepts when it's sure (one metric, at most one activity and one time
period, and no words it doesn't know); otherwise it passes the question on.

### Evaluating the question parsers

From the repo root, with Ollama running:

```bash
python -m heartwatchapp.scripts.eval_ask              # all three modes, dev questions
python -m heartwatchapp.scripts.eval_ask --modes rules  # no Ollama needed
```

It puts every question in `heartwatchapp/eval/ask_questions_dev.jsonl`
through the app's own pipeline in `rules`, `llm` and `hybrid` mode and
compares the form each understood with the expected form written by hand.
The headline number is **false accepts**: a question answered with a number
when it should have been refused, or answered for the wrong metric,
activity or time period. "Now" is pinned (`--now 2026-10-14T15:00 --tz
America/Denver`, a Wednesday) and the sessions are built on a throwaway
database from `--data-seed`, so results are reproducible; the model runs at
temperature 0 with seed 0, and two runs give identical results apart from
timings. If Ollama is down, `rules` still runs and the others are marked
skipped. `--db` / `--overwrite` work as for the summaries' evaluation
below, and the real database is never opened. Results go to
`heartwatchapp/eval_results/ask-<YYYYMMDD-HHMMSS>/` (gitignored):
`runs.jsonl`, `meta.json` (model digest, Ollama version, options, now,
question file hash) and `summary.md` (a table per mode and every false
accept).

**Dev vs held-out questions.** The dev file may be used to tune the keyword
parser and the prompt. A held-out file (e.g.
`heartwatchapp/eval/ask_questions_heldout.jsonl`, run with `--questions
PATH`) is written by someone who hasn't seen the parser, and **the parser
must never be tuned against it** -- otherwise its score stops meaning
anything.

On the dev questions (October 16, 2026, experiment 3): false accepts were
0 of 30 for `rules`, 12 of 30 for `llm` and 0 of 30 for `hybrid` (details
in the run's `summary.md`). The model on its own mostly fails by inventing
an activity the question never named, answering an off-topic question, or
reading "this week" as the last 7 days. Experiments 1 and 2 reported
better `llm` scores (8 and 10 false accepts), but those were inflated:
several prompt examples repeated or closely paraphrased dev questions.
Experiment 3 is the first with no prompt example near a dev question
(`tests/test_ask_vocab.py` now enforces it), so it is the clean dev
baseline. The held-out set remains the real measure.

The "(N sessions)" after an answer counts only the sessions that
contributed to the number: for "time spent Running", sessions with some
Running; for heart rate while Running, sessions with heart-rate samples
during Running. A real 0 ("Time spent Standing last week: 0 seconds.") has
no count.

### Known limitations (v1)

- One question at a time: a follow-up like "and what about walking?" isn't
  understood.
- Text answers only -- no charts -- and no commands ("start recording").
- "My last run" is read as *latest session, Running*: the running part of
  your most recent session, not your most recent session that included
  running.
- Counting sessions by activity ("how many running sessions") isn't
  supported yet and gets a fixed reply (straight from the keyword parser
  when the question says "how many / number of ... sessions").
- Session counts on the real database are inflated by the Live Monitor
  opening a session every time the app starts (an open item, fixed with the
  data-source abstraction).
- Activity answers use simulated predictions until the real classifier
  lands (milestone 6).

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
     heartwatchapp.scripts.demo_session` (AI path) -- see **Demo session**
     below; or
   - click Settings -> Developer -> **Seed demo session** (this *does*
     write to your real database -- seeded sessions include dummy
     predictions, `model_version "demo-dummy"`), or leave the app running
     on the Live Monitor for more than a minute.
4. In Sessions, double-click the session. In `llm` mode the terminal
   shows each pass or rejection and why.

To point the app at any other database file, set `HEARTWATCH_DB_PATH`
before launching; with it unset the app uses `data/heartwatch.db` as usual.

### Demo session

The demo script makes a separate, brand-new database with one example
session in it, so you can see a full summary without touching anyone's
real recordings.

- **Default:** `python -m heartwatchapp.scripts.demo_session` builds a new
  database in a temporary folder (the app's own schema from
  `data/db.py`'s `init_db()`, plus one demo session), launches the app on
  it with the `heartwatch.llm` log at DEBUG, and prints the new session's
  number. The real database is never opened, copied or read.
- `--db PATH` chooses where the demo database goes. If a file is already
  there, the script stops (exit code 2) and leaves it alone, so a mistyped
  path can't delete anything. Add `--overwrite` to replace it; the file and
  any `-wal` / `-shm` / `-journal` next to it are deleted first (a leftover
  `-wal` would otherwise be replayed into the new file). The default
  temporary folder is always new, so it never needs `--overwrite`.
- `--data-seed N` fixes the heart-rate values; leave it out for random
  ones. `--no-launch` builds the database and exits.
- `--from-real` starts from a copy of the real database instead of an
  empty one, for the rare case you want real sessions alongside the demo.
  It prints *"Close HeartWatch before using --from-real; copying while the
  app is writing can produce an inconsistent copy."* It copies the real
  `.db`, `-wal` and `-shm` files at the filesystem level (`shutil.copy2`)
  and never opens the real database with SQLite in any mode; it opens only
  the copy, runs `PRAGMA integrity_check`, and if that isn't `ok` deletes
  the copy and exits with an error.
- In both modes the script refuses to run (exit code 2) if the
  destination is the real database, comparing resolved paths and, if both
  exist, `os.path.samefile`, so symlinks, hard links and relative paths
  can't get around it. "The real database" means what the app would open:
  `HEARTWATCH_DB_PATH` if set, otherwise `data/heartwatch.db`.

**Measured benchmark** (Dylan's MacBook Air, Apple M2, 8 GB):
`gemma3:1b` -- 883 MB, 100% GPU-resident on Apple M2, 2048 context. Time
to first token 0.04s; warm generation 0.11–0.94s; cold load 2.31s.
Selected to fit an 8 GB development machine.

## Where numbers come from

Every number the app shows -- the Dashboard's cards, bars and donut, the
Sessions table's Avg/Peak BPM, and the session summary -- is computed by
one function in `data/stats.py`, and fetched through `data/queries.py`
(`run()` with a `MetricRequest`, plus `activity_breakdown()` and the two
session lists). **SQL may filter, order, join and `COUNT(*)` rows; it never
does the arithmetic.** `tests/test_queries.py` scans `data/queries.py`,
`data/db.py`, `ui/` and `ask/` and fails on `AVG(`, `SUM(`, `MAX(` or `MIN(` in SQL.
That's what keeps the Dashboard and the summary from ever disagreeing. The
AI bar (`ask/`, also scanned) answers through the same `run()`.

- **Average heart rate** is the simple mean of the measured samples: add
  up the readings, divide by how many. Gaps (sensor dropouts) are left
  out, never filled in. Across several sessions the samples are pooled, so
  every sample counts once (the same answer as SQL's `AVG(bpm)`).
- **Time per activity** comes from the prediction window rule in
  `stats._activity_segments()` (each prediction owns the 5 s before it),
  worked out per session and then added up, so no window crosses from one
  session into another.
- **Which sessions a range covers:** a session belongs to "Today", "Last 7
  days" or "Last 30 days" if it *started* in that range, and then all of
  its samples count; a session that runs past midnight counts on the day it
  started. Days are local calendar days (7 days = today plus the 6 before
  it, from local midnight), so the Dashboard and the Sessions filter mean
  the same thing. Sessions still recording are left out of the Dashboard
  until they end; the Sessions table lists them.
- **Simulated predictions:** until milestone 6 every prediction comes from
  the dummy classifier (`model_version "demo-dummy"`). When any of the
  breakdown's time does, the Dashboard says so under it: *"Activity times
  are from simulated predictions."*

## Development notes

Things we learned the hard way while building the AI summaries. Read this
before touching `llm/` or the database.

| Don't | Do instead |
|---|---|
| Quit Ollama with `osascript` or force-kill it | Point a client at a dead port (`127.0.0.1:9`) to simulate Ollama being down, or quit the menu-bar app by hand, run `ollama serve` in a terminal, and Ctrl-C it |
| Reuse the module-level `AsyncClient` across several `asyncio.run()` calls | Keep live-model work inside one `asyncio.run()`, and run Qt checks in a separate process. The client is tied to its event loop. |
| Catch only `httpx.RequestError` for "Ollama down" | Catch `CONNECTION_ERRORS` (includes built-in `ConnectionError`) |
| Check numbers as substrings in tests ("134" contains "4") | Compare digit tokens from `re.findall(r"\d+")` |
| Open the real DB from scripts, even with `mode=ro` (it still changes `-shm`) | Build a fresh DB; if a copy is truly needed, copy the files at the filesystem level and open only the copy. |
| Use pytest | It isn't installed. Use stdlib `unittest`. |
| Update test expectations by copying new outputs | Recompute expected values by hand from the rule being tested |
| Let the model's reply reach the screen, even "just the numbers" | Parse it into a form (`ask/llm_parser.py`), check it, answer with `run()` and a Python template |
| Trust `gemma3:1b` to leave a field empty | It invents activities and answers off-topic questions; keep the keyword parser first and the health check before both |
| Ping Ollama with a chat request to see if it's up | Use `list` (`health_check()`): a chat loads the model into memory |
| Pass `keep_alive=-1` to keep the model warm | Use Ollama's default (5 min) and preload on focus at most once per period (`ui/ai_bar.py`) |
| Tune the keyword parser on the held-out questions | Tune on `eval/ask_questions_dev.jsonl` only |

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

`--db PATH` keeps the throwaway database somewhere you choose instead of a
temporary folder. Like the demo script, it won't replace a file that's
already there: it stops with exit code 2 and leaves the file alone unless
you add `--overwrite`, which deletes that file (and any `-wal` / `-shm` /
`-journal` next to it) before building. The default temporary folder is
always new, so it never needs `--overwrite`.

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
├── config.py            environment-variable settings (HEARTWATCH_DB_PATH, HEARTWATCH_SUMMARY_MODE, HEARTWATCH_ASK_MODE)
├── ui/
│   ├── main_window.py   QMainWindow shell (sidebar · topbar · stack · AI bar)
│   ├── sidebar.py       64px icon-only nav (QToolButton + QButtonGroup)
│   ├── topbar.py        52px section title / status / connection + battery
│   ├── theme.py         design tokens + QSS (light/dark, no scattered hex)
│   ├── widgets.py       Card, MetricCard, Badge, Divider, PlaceholderState
│   ├── charts.py        Matplotlib DonutChart + scrolling LiveHRChart
│   ├── dashboard.py     Section 1 — metric cards, activity breakdown, sessions (all via data/queries.py)
│   ├── live_monitor.py  Section 2 — live HR + activity badge + IMU axes
│   ├── sessions.py      Section 3 — sessions list + SessionDetailDialog (chart, summary)
│   ├── ml_view.py       Section 4 — stub (pipeline summary + matrix placeholder)
│   ├── settings.py      Section 5 — stub (placeholder controls, theme toggle)
│   ├── ai_bar.py        "Ask about your data" bar (rules answers at once; AI answers via LlmWorker)
│   └── pending_request.py  waits for one background reply at a time, with a timeout
├── data/
│   ├── db.py            SQLite connection + schema + writes and plain row reads
│   ├── queries.py       the one query layer: MetricRequest, run(), ranges, breakdown, session lists
│   └── stats.py         every metric (average/peak HR, time per activity, totals) + session summaries
├── ask/                 the AI bar's question pipeline (no Qt)
│   ├── vocab.py         metrics, activities, ranges and synonyms -- one source
│   ├── guard.py         health/advice check, before any parser
│   ├── rules.py         keyword parser
│   ├── llm_parser.py    JSON schema + checks for the model's form
│   ├── parse_result.py  ParseResult, shared by both parsers
│   ├── pipeline.py      guard -> rules -> (AI) -> normalize -> validate -> run -> answer
│   └── answers.py       every sentence the bar shows
├── eval/
│   └── ask_questions_dev.jsonl  dev questions for scripts/eval_ask.py
├── llm/
│   ├── client.py        the only app module that imports `ollama`
│   ├── prompts.py       system prompts + session-summary and question-parser prompt text
│   ├── validate.py      checks every generated summary before it's shown
│   └── worker.py        Qt-thread bridge (sends checked replies to the GUI)
├── ml/inference.py      TFLite activity classifier stub
├── scripts/
│   ├── benchmark_ollama.py  standalone viability check, run by hand
│   ├── demo_db.py           builds the demo session on a fresh DB (shared by the two below)
│   ├── demo_session.py      launch the app on a fresh demo DB with a data-rich session
│   ├── eval_summaries.py    repeatable AI-summary evaluation (writes eval_results/)
│   └── eval_ask.py          repeatable ask-bar parser evaluation (writes eval_results/ask-*)
└── tests/
    ├── test_stats.py          unit tests for data/stats.py (no live Ollama needed)
    ├── test_validate.py       unit tests for llm/validate.py (no live Ollama needed)
    ├── test_demo_session.py   demo tooling: real-DB guard, demo facts, seeded predictions
    ├── test_demo_db_fresh.py  demo DB never touches the real DB's files (fake real DB, live WAL)
    ├── test_summary_mode.py   HEARTWATCH_SUMMARY_MODE and the dialog's two paths
    ├── test_queries.py        query layer + metric functions, ranges, no-SQL-arithmetic guard
    ├── test_dashboard_view.py Dashboard widget offscreen: cards match run(), caption, empty states
    ├── test_eval_summaries.py eval script's pure parts (no live Ollama needed)
    ├── test_ask_vocab.py      one source of words; guard never blocks a supported phrase
    ├── test_ask_rules.py      keyword parser: accepts, deferrals, unsupported
    ├── test_ask_pipeline.py   metric x activity table, modes, range override, AI parser (mocked)
    ├── test_ask_answers.py    every answer template and fixed reply
    ├── test_eval_ask.py       ask eval: real-DB refusal, scoring, rules-mode run
    └── test_ai_bar.py         AI bar offscreen: answers, stale replies, timeout, preload
```

## Build status

| # | Milestone | State |
|---|---|---|
| 1 | App shell (window, sidebar, topbar, stack, AI bar) | done |
| 2 | Dashboard — cards / bars / donut / sessions (demo data) | done |
| 3 | Live Monitor — dummy sine HR + IMU axes; also writes dummy activity predictions (`model_version "demo-dummy"`) until milestone 6 | done |
| 4 | "Ask about your data" bar — keyword parser + local model (gemma3:1b) fill a form; Python answers via `data/queries.py` | done (v1; see Known limitations) |
| 5 | SQLite integration (replace demo data) | done for the Dashboard, Sessions and the AI bar (all through `data/queries.py`); the Live Monitor still streams dummy data |
| 6 | BLE stream from ESP32-S3 + real TFLite inference | not started |
| — | Session summaries | done — plain (Python-written) by default; AI phrasing via Ollama / gemma3:1b with `HEARTWATCH_SUMMARY_MODE=llm`, degrades to the plain summary otherwise |

The Dashboard, Sessions and AI bar read the SQLite database. The Live Monitor's
values come from dummy generators until step 6, and every activity prediction
is from the dummy classifier (`model_version "demo-dummy"`) until then.
