# Database setup

How to get a working `heartwatch.db` from a clean checkout, how to look
inside it, and how to reset it. See `SCHEMA.md` for what's actually in the
database and why it's shaped the way it is.

## First run

Nothing to set up by hand. Run the app from the repo root:

```bash
cd heartwatchapp
source .venv/bin/activate   # or create one: python3 -m venv .venv && pip install -r requirements.txt
python -m heartwatchapp.main
```

On first launch, `main.py` calls `db.init_db()` before the window opens,
which creates `heartwatchapp/data/heartwatch.db` if it doesn't already exist and
applies `heartwatchapp/data/schema.sql`. There's no separate "run this migration
script" step.

## Where the file lives, and why it's gitignored

`heartwatchapp/data/heartwatch.db` (plus its `-wal` and `-shm` sidecar files
while the app is running — see "WAL mode" below). A per-user OS
application-data directory (via Qt's `QStandardPaths`) would be the more
"correct" choice for a shipped application, but for a four-person capstone
team actively developing against this database, a path inside the repo that
everyone can find and open in a SQLite browser is far more useful. This is a
deliberate tradeoff, not an oversight.

Because it's real (if synthetic) data that changes on every run and every
teammate's machine, `*.db`, `*.db-wal`, and `*.db-shm` are in `.gitignore`.
Never commit these files.

## The three pragmas

Every connection this app opens — not just the one `init_db()` uses — runs:

```sql
PRAGMA journal_mode = WAL;     -- readers don't block the writer
PRAGMA synchronous  = NORMAL;  -- safe under WAL, much faster than FULL
PRAGMA foreign_keys = ON;      -- OFF by default in SQLite!
```

The third one is the one to actually pay attention to.

**SQLite disables foreign-key enforcement by default**, for backward
compatibility with databases created before the feature existed. This is
easy to forget because the schema *looks* like it enforces referential
integrity — `imu_samples.session_id` really is declared
`REFERENCES sessions(id) ON DELETE CASCADE`. But without
`PRAGMA foreign_keys = ON` set on the specific connection performing a
delete, that `ON DELETE CASCADE` is **silently ignored**. Deleting a session
would leave its samples behind as orphaned rows that reference a session
that no longer exists — no error, no warning, and the "delete a session"
acceptance demo would *look* like it worked (the session disappears from the
list) while actually leaving broken data behind.

This is why `_connect()` in `data/db.py` applies the pragma on every open,
and why `data/writer.py`'s background thread opens its one long-lived
connection through that same helper rather than a bare `sqlite3.connect()`.

**To verify it's actually working:** open Settings → Developer, note the row
counts in "Database info," seed a demo session, delete it, and check the
counts again. `sessions` should drop by 1, and `imu_samples`/`hr_samples`
should drop by exactly however many rows that session had — not stay flat.

## WAL mode

`journal_mode = WAL` means SQLite keeps a separate write-ahead log
(`heartwatch.db-wal`) instead of locking the whole database file on every
write. This is what lets the Sessions view and Dashboard read from the
database at the same time the Live Monitor's background writer thread is
inserting samples, without either one blocking the other. A `heartwatch.db-shm`
shared-memory file also appears alongside it while the app is running —
both are normal and are cleaned up automatically on a clean shutdown (they
get gitignored either way).

## Inspecting the file directly

With the `sqlite3` CLI:

```bash
sqlite3 heartwatchapp/data/heartwatch.db
sqlite> .tables
sqlite> .schema sessions
sqlite> SELECT * FROM samples_flat LIMIT 20;
sqlite> SELECT COUNT(*) FROM imu_samples;
```

Or open the file in [DB Browser for SQLite](https://sqlitebrowser.org/) —
double-click `heartwatch.db`, use the "Browse Data" tab. Since the file is
in WAL mode, close the app first (or DB Browser may show a slightly stale
view if it opens the file while the writer thread has uncommitted changes
buffered — see "Buffered writes" below).

## Using the Settings developer panel

Settings → Developer is a set of GUI controls that exercise the SQLite layer
directly, for demoing and debugging without touching a SQL client:

- **Seed demo session** — writes 2–3 minutes of synthetic IMU + HR data
  through the same insert functions the Live Monitor and (eventually) BLE
  use, backdated to look like a real past recording, with a random activity
  label.
- **Delete session** — pick a session from the dropdown, confirm, and it's
  gone (cascade included). The Dashboard and Sessions views refresh
  automatically afterward.
- **Run samples_flat query** — runs `SELECT * FROM samples_flat LIMIT 20`
  and shows the result in a table. This is the acceptance-review demo for
  the `(DATETIME, x, y, z, heartrate)` format.
- **Database info** — file path, file size on disk, row count per table,
  and schema version, with a Refresh button.

## Buffered writes

High-rate inserts (IMU at ~10 Hz, HR at ~1 Hz) don't hit disk immediately —
`data/writer.py`'s background thread buffers them and flushes in batches
(whichever comes first: ~100 rows buffered, or ~1 second elapsed). This
means row counts can lag reality by up to a second under load. The
developer panel's "Database info" and "Run samples_flat query" both call
`flush_and_wait()` first, which blocks until every write queued before that
call has actually committed — so what you see in the panel is always
current. If you're inspecting the file with an external tool while the app
is running, keep this buffering in mind.

## Resetting

Delete `heartwatch.db`, `heartwatch.db-wal`, and `heartwatch.db-shm` from
`heartwatchapp/data/`, then relaunch. `init_db()` recreates an empty database
from `schema.sql` on the next start — there's no data migration to run
because there's nothing to migrate from.

```bash
rm heartwatchapp/data/heartwatch.db heartwatchapp/data/heartwatch.db-wal heartwatchapp/data/heartwatch.db-shm
python -m heartwatchapp.main
```
