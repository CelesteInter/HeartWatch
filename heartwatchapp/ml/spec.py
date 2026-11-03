"""The activity model's pipeline numbers, in one place.

Plain-language overview: every number that decides what the model sees --
how often samples come (10 per second), how long a window is (5 seconds =
50 samples), how much is cut off each recording's ends, how the values are
scaled -- lives here. Training, export and the app's classifier all read
these, and ml/inference.py's INPUT_SHAPE is built from them, so the
training code and the app can't silently disagree.

Rules for this file:
- Imports nothing from the app (in particular nothing from data/, because
  data/stats.py imports ml/inference.py, which imports this file). The
  check that WINDOW_SAMPLES matches data/stats.py's PREDICTION_WINDOW_MS
  lives in tests/test_ml_pipeline.py only.
- No numpy, no PySide6: the training environment has no PySide6, and
  importing ml.inference must stay cheap.
"""

from __future__ import annotations

RATE_HZ = 10                 # model samples per second
BIN_MS = 1000 // RATE_HZ     # 100 ms: each model sample = mean of one bin
WINDOW_SAMPLES = 50          # 5 s window
TRAIN_HOP_SAMPLES = 25       # 50% overlap, training only
LIVE_HOP_SAMPLES = WINDOW_SAMPLES  # live predictions don't overlap
TRIM_S = 5                   # seconds dropped from each end of a recording
ACCEL_NORM_G = 4.0           # accel (g) / this
GYRO_NORM_DPS = 500.0        # gyro (deg/s) / this
CHANNELS = ("ax", "ay", "az", "gx", "gy", "gz")

# Legacy captures (firmware 1eabdb8) print raw MPU counts. That firmware
# never writes the range registers, so the sensor's power-on defaults
# apply: +-2 g and +-250 deg/s.
LEGACY_ACCEL_SCALE = 1 / 16384   # g per count
LEGACY_GYRO_SCALE = 1 / 131      # deg/s per count

WINDOW_MS = WINDOW_SAMPLES * BIN_MS
INPUT_SHAPE = (1, WINDOW_SAMPLES, len(CHANNELS))
