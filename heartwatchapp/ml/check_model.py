"""Checks the exported model in the APP environment (LiteRT, no TensorFlow).

    heartwatchapp/.venv/bin/python -m heartwatchapp.ml.check_model

Loads ml/models/har_cnn.tflite through ActivityClassifier.load() (which
checks model_meta.json against ml/spec.py), runs the parity windows that
training saved, and prints the largest difference from Keras's
probabilities plus a warm per-window inference time. Writes nothing.
"""

from __future__ import annotations

import statistics
import time

import numpy as np

from .inference import METADATA_PATH, MODEL_PATH, PARITY_PATH, ActivityClassifier

WARMUP, TIMED = 20, 500


def main() -> int:
    clf = ActivityClassifier.load(MODEL_PATH, METADATA_PATH)
    fixture = np.load(PARITY_PATH)
    X, keras_probs = fixture["X"], fixture["probs"]
    probs = clf.predict_proba(X)
    diff = float(np.abs(probs - keras_probs).max())
    same = int((probs.argmax(1) == keras_probs.argmax(1)).sum())
    one = X[:1]
    for _ in range(WARMUP):
        clf.predict_proba(one)
    times = []
    for _ in range(TIMED):
        t = time.perf_counter()
        clf.predict_proba(one)
        times.append((time.perf_counter() - t) * 1e6)
    print(f"model {clf.model_version}, {MODEL_PATH.stat().st_size:,} bytes")
    print(f"parity: {len(X)} windows, max |LiteRT - Keras| = {diff:.3e}, "
          f"same argmax {same}/{len(X)}")
    print(f"warm inference per window: median {statistics.median(times):.1f} us, "
          f"p95 {sorted(times)[int(0.95 * len(times))]:.1f} us ({TIMED} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
