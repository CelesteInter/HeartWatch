-- HeartWatch SQLite schema. Loaded once by init_db(). Do not scatter
-- CREATE TABLE strings through the Python code -- this file is the
-- single source of truth for the on-disk shape of the database.
--
-- See heartwatchapp/docs/SCHEMA.md for the plain-language explanation of
-- every table and the reasoning behind these choices.

PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------
-- sessions: one row per recording run. Everything else hangs off this.
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sessions (
    id           INTEGER PRIMARY KEY,
    started_at   INTEGER NOT NULL,          -- epoch milliseconds
    ended_at     INTEGER,                   -- NULL while still recording
    device_id    TEXT,                      -- BLE address, or 'demo'
    label        TEXT,                      -- walking / running / idle / ...
    accel_scale  REAL NOT NULL,             -- multiply raw count by this to get g
    gyro_scale   REAL NOT NULL,             -- multiply raw count by this to get dps
    notes        TEXT
);

-- ---------------------------------------------------------------
-- imu_samples: high-rate motion data. Raw int16 counts from the MPU-9250.
-- No surrogate id column: rows are never addressed individually.
-- Magnetometer columns (mx, my, mz) are added later as nullable columns.
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS imu_samples (
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    ts         INTEGER NOT NULL,            -- epoch milliseconds
    ax INTEGER NOT NULL, ay INTEGER NOT NULL, az INTEGER NOT NULL,
    gx INTEGER NOT NULL, gy INTEGER NOT NULL, gz INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_imu_session_ts ON imu_samples(session_id, ts);

-- ---------------------------------------------------------------
-- hr_samples: heart rate from the MAX30101, on its own cadence.
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS hr_samples (
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    ts         INTEGER NOT NULL,
    bpm        INTEGER NOT NULL,
    confidence REAL                         -- sensor signal quality 0-1, nullable
);
CREATE INDEX IF NOT EXISTS idx_hr_session_ts ON hr_samples(session_id, ts);

-- ---------------------------------------------------------------
-- predictions: one row per CNN inference window.
-- model_version lets the same session be compared across model builds.
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS predictions (
    session_id    INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    ts            INTEGER NOT NULL,         -- window END time
    predicted     TEXT NOT NULL,
    confidence    REAL NOT NULL,
    model_version TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_pred_session_ts ON predictions(session_id, ts);

-- ---------------------------------------------------------------
-- schema_meta: single row holding the schema version integer.
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS schema_meta (
    id      INTEGER PRIMARY KEY CHECK (id = 1),
    version INTEGER NOT NULL
);
INSERT OR IGNORE INTO schema_meta (id, version) VALUES (1, 1);

-- ---------------------------------------------------------------
-- samples_flat: compatibility view matching the specified task format
-- (DATETIME, x, y, z, heartrate). x/y/z are the ACCELEROMETER axes.
-- Gyroscope is intentionally absent: the specified format has no columns
-- for it. Nothing in the application reads this view; it exists for the
-- acceptance demo and for anyone inspecting the file with a SQLite browser.
-- Heart rate uses last-known-value carry-forward, since HR and IMU sample
-- at different rates.
-- ---------------------------------------------------------------
CREATE VIEW IF NOT EXISTS samples_flat AS
SELECT
    datetime(i.ts / 1000, 'unixepoch') AS DATETIME,
    i.ax AS x,
    i.ay AS y,
    i.az AS z,
    (SELECT h.bpm
       FROM hr_samples h
      WHERE h.session_id = i.session_id
        AND h.ts <= i.ts
      ORDER BY h.ts DESC
      LIMIT 1) AS heartrate
FROM imu_samples i;
