# HeartWatch database schema

Plain-language explanation first, technical detail after. This file describes
`heartwatchapp/data/schema.sql`, which is the single source of truth for the
database's shape — nothing in the Python code creates tables directly.

## What's stored, and why

HeartWatch records a wrist-worn watch's sensors while someone wears it. Two
sensors, at two different speeds:

- The **MPU-9250** (motion sensor: accelerometer + gyroscope) reports about
  10 times a second.
- The **MAX30101** (heart-rate sensor) reports about once a second.

Those two streams are kept in **separate tables** (`imu_samples` and
`hr_samples`) rather than one combined table. If they shared a table, nine
out of every ten rows would need an empty or stale heart-rate value just to
keep the IMU rows lined up — every row in these tables is a real,
independently-timed measurement.

Everything hangs off a **session**: one row in `sessions` per recording run
("start monitoring" to "stop monitoring", or one seeded demo). Delete a
session and its samples go with it — see "Cascading deletes" below.

## Tables

### `sessions`

One row per recording run.

| Column | Type | Meaning |
|---|---|---|
| `id` | INTEGER PK | Session id. Everything else references this. |
| `started_at` | INTEGER | Epoch milliseconds when the session began. |
| `ended_at` | INTEGER, nullable | Epoch milliseconds when it ended. **NULL means the session is still live** — that's how the app tells a recording session apart from a finished one, rather than a separate boolean flag. |
| `device_id` | TEXT | BLE address of the watch, or `'demo'` for the Live Monitor's generator, or `'seed'` for a synthetic session created from the developer panel. |
| `label` | TEXT | Activity label: walking / running / idle / etc. Nullable — a live session doesn't know its label yet. |
| `accel_scale` | REAL | Multiply a raw `imu_samples.ax/ay/az` count by this to get g-force. |
| `gyro_scale` | REAL | Multiply a raw `imu_samples.gx/gy/gz` count by this to get degrees/second. |
| `notes` | TEXT | Free text, nullable. |

### `imu_samples`

High-rate motion data. No id column — a row is never looked up
individually, only as part of a session's time series, so a surrogate key
would be pure overhead at 10 rows/second.

| Column | Type | Meaning |
|---|---|---|
| `session_id` | INTEGER | FK → `sessions.id`, `ON DELETE CASCADE`. |
| `ts` | INTEGER | Epoch milliseconds. |
| `ax, ay, az` | INTEGER | Raw accelerometer counts, one axis each. |
| `gx, gy, gz` | INTEGER | Raw gyroscope counts, one axis each. |

Indexed on `(session_id, ts)` — every read in this app asks for one
session's samples in time order.

Magnetometer columns (`mx, my, mz`) are a known future addition, left out
for now rather than added as unused placeholders.

### `hr_samples`

Heart rate, on its own cadence (roughly 1 Hz, vs. IMU's ~10 Hz).

| Column | Type | Meaning |
|---|---|---|
| `session_id` | INTEGER | FK → `sessions.id`, `ON DELETE CASCADE`. |
| `ts` | INTEGER | Epoch milliseconds. |
| `bpm` | INTEGER | Beats per minute. |
| `confidence` | REAL, nullable | Sensor signal quality, 0–1. Nullable because not every code path that writes an HR sample has a confidence figure to report. |

### `predictions`

One row per activity-classification window from the 1D CNN.

| Column | Type | Meaning |
|---|---|---|
| `session_id` | INTEGER | FK → `sessions.id`, `ON DELETE CASCADE`. |
| `ts` | INTEGER | Epoch milliseconds — the **end** of the inference window. |
| `predicted` | TEXT | The predicted activity label. |
| `confidence` | REAL | Model confidence, 0–1. |
| `model_version` | TEXT | Lets the same session's raw data be re-scored and compared across model builds without losing the original prediction. |

Nothing calls `insert_prediction` yet — see "Open items" below.

### `schema_meta`

A single row (`id = 1`) holding a `version` integer. `init_db()` checks this
against the version the running code expects, and refuses to proceed
silently against a mismatched database file. This is what lets a schema
change be *detected* instead of causing confusing query errors later.

## Cascading deletes

Every samples/predictions table declares its `session_id` column as
`REFERENCES sessions(id) ON DELETE CASCADE`. In plain terms: delete a
session, and SQLite automatically deletes every `imu_samples`, `hr_samples`,
and `predictions` row that pointed at it. The app never has to remember to
clean those up manually, and it's not possible to end up with samples that
reference a session that no longer exists.

**This only works if `PRAGMA foreign_keys = ON` is set on the connection
doing the delete.** SQLite ships with foreign-key enforcement *off* by
default for backward compatibility — without the pragma, `ON DELETE CASCADE`
is silently ignored and deleting a session leaves its samples behind as
orphaned rows. See `DATABASE_SETUP.md` for where this is applied in the code
and how to verify it's working.

## Raw counts, not converted units

`imu_samples` stores whatever integer the MPU-9250's analog-to-digital
converter reports — not g-force, not degrees/second. The conversion factor
(`sessions.accel_scale`, `sessions.gyro_scale`) is stored per-session instead
of applied at write time, because that factor depends on a sensor range
setting that may change during the project. If the range setting changes
between two sessions, both are still stored correctly and both can still be
converted correctly — an old recording never has to be reprocessed or
becomes unreadable just because a later session used a different range.

## Integer millisecond timestamps, not TEXT dates

SQLite has no native DATETIME type — its own documentation describes dates
as being stored as TEXT, REAL, or INTEGER, and the application is
responsible for interpreting them consistently. This schema stores every
timestamp as **epoch milliseconds, as an INTEGER**:

- Compact — an 8-byte integer vs. a ~19-character ISO-8601 string.
- Sorts correctly with a plain `ORDER BY`, with no string-to-date parsing.
- Exact. At 10 samples/second, two samples can be 100ms apart; a
  timestamp format with second-level precision would collapse them.

Anywhere a human needs to read a timestamp (the developer panel, the
Sessions list), formatting to a readable date happens at display time in
Python, or — for the `samples_flat` view specifically — via SQLite's
`datetime()` function. The stored value itself is never a string.

## The `samples_flat` view, and why it exists

The team lead's original task specification described the stored data as a
table shaped `(DATETIME, x INT, y INT, z INT, heartrate INT)`. That shape
comes from an earlier ECE 5780 breadboard prototype, which used a 3-axis
accelerometer and nothing else.

HeartWatch's MPU-9250 is a 6-axis sensor (3-axis accelerometer + 3-axis
gyroscope), and the activity-classification CNN's input tensor is a window
of **all six axes** — accelerometer alone is not enough data for the model
to run against. Storing only three axes, to match the older spec literally,
would make every session unusable for training or running the model.

The resolution: **store all six axes** in `imu_samples` (as described
above), and additionally expose a SQL **view** —`samples_flat`— that returns
exactly the specified five columns, for the acceptance demo and for anyone
opening the file in a SQLite browser:

```sql
SELECT * FROM samples_flat LIMIT 20;
```

returns rows shaped `(DATETIME, x, y, z, heartrate)`. Specifically:

- `DATETIME` is `imu_samples.ts` formatted as a human-readable string via
  SQLite's `datetime()` function — the one place in this project a
  timestamp is presented as text rather than an integer, because this view
  exists to be read by a person, not by application code.
- `x, y, z` are the **accelerometer** axes (`ax, ay, az`). The gyroscope is
  intentionally absent — the specified five-column format has no columns
  for it, and the view's job is to match that format, not to be a complete
  data export (that's what `export_training_csv()` is for).
- `heartrate` is the most recent `hr_samples.bpm` at or before that IMU
  sample's timestamp (last-known-value carry-forward), since heart rate
  updates roughly 10x less often than IMU data and the two streams don't
  share timestamps.

Nothing in the application reads `samples_flat` on any normal code path —
the developer panel's "Run samples_flat query" button exists specifically to
demonstrate it. The view's carry-forward logic uses a correlated subquery,
which is slow across a large table; that's acceptable because it is a
manual-inspection tool, not something on a hot path. Always add
`WHERE session_id = ?` or a `LIMIT` when querying it directly.

## Open items (deferred, not resolved)

- **Retention.** All raw samples are kept indefinitely for the project's
  duration. No downsampling or archiving of old sessions is implemented.
  Fine at capstone scale (see `DATABASE_SETUP.md` for rough volume numbers);
  would need addressing before any long-term deployment.
- **Live prediction writes.** `insert_prediction` / `predictions` exist and
  work, but nothing calls them yet — the TFLite classifier isn't wired in.
  The intent is for predictions to be written live during monitoring
  (one row per inference window), not computed after the fact on review.
