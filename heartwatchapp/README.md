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
│   ├── sessions.py      Section 3 — stub (empty state)
│   ├── ml_view.py       Section 4 — stub (pipeline summary + matrix placeholder)
│   ├── settings.py      Section 5 — stub (placeholder controls, theme toggle)
│   └── ai_bar.py        always-visible data-query bar (layout only)
├── data/db.py           SQLite connection + schema + query stubs
└── ml/inference.py      TFLite activity classifier stub
```

## Build status

| # | Milestone | State |
|---|---|---|
| 1 | App shell (window, sidebar, topbar, stack, AI bar) | done |
| 2 | Dashboard — cards / bars / donut / sessions (demo data) | done |
| 3 | Live Monitor — dummy sine HR + IMU axes | done |
| 4 | AI bar wired to Anthropic API | layout only |
| 5 | SQLite integration (replace demo data) | not started |
| 6 | BLE stream from ESP32-S3 + real TFLite inference | not started |

All Dashboard / Live Monitor values are hardcoded demo data or dummy generators
until step 5–6.
