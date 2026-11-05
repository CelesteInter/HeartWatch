"""Exports a trained Keras model for the app (training environment only).

Writes into ml/models/ (git-ignored):
- har_cnn.tflite   float32, no quantization
- model_meta.json  what the model expects + where it came from; the app's
                   ActivityClassifier.load() checks it against ml/spec.py
- parity.npz       a few test windows and Keras's probabilities for them,
                   so tests/test_ml_pipeline.py can check LiteRT gives the
                   same answers in the app environment

Converter: tf.lite.TFLiteConverter.from_keras_model first; if that fails,
model.export(..., "tf_saved_model") + from_saved_model. The path used (and
the first error, if any) is recorded in model_meta.json.
"""

from __future__ import annotations

import datetime
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from .. import spec
from ..inference import CLASSES, METADATA_PATH, MODEL_PATH, PARITY_PATH, expected_metadata
from .dataset import REPO_ROOT

MODEL_VERSION = "har-cnn-v0"


def convert(model) -> tuple[bytes, dict]:
    import tensorflow as tf

    try:
        data = tf.lite.TFLiteConverter.from_keras_model(model).convert()
        return data, {"path": "from_keras_model", "from_keras_model_error": None}
    except Exception as e:  # recorded, then fall back
        first_error = f"{type(e).__name__}: {e}"
    with tempfile.TemporaryDirectory() as tmp:
        model.export(tmp, format="tf_saved_model")
        data = tf.lite.TFLiteConverter.from_saved_model(tmp).convert()
    return data, {"path": "export+from_saved_model", "from_keras_model_error": first_error}


def git_commit() -> dict:
    def run(*args):
        return subprocess.run(["git", "-C", str(REPO_ROOT), *args],
                              capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD") or None,
            "dirty": bool(run("status", "--porcelain"))}


def pip_freeze() -> list[str]:
    out = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    return out.stdout.splitlines()


def build_metadata(ds, converter: dict, extra: dict) -> dict:
    import keras
    import tensorflow as tf

    from .dataset import counts_by_class

    meta = {
        "model_version": MODEL_VERSION,
        **expected_metadata(),
        "bin_ms": spec.BIN_MS,
        "trim_s": spec.TRIM_S,
        "train_hop_samples": spec.TRAIN_HOP_SAMPLES,
        "input_shape": list(spec.INPUT_SHAPE),
        "captures": [{"path": c["path"], "label": c["label"], "sha256": c["sha256"]}
                     for c in ds.captures],
        "hardware_note": ds.captures[0]["hardware_note"] if ds.captures else None,
        "git": git_commit(),
        "tensorflow_version": tf.__version__,
        "keras_version": keras.__version__,
        "numpy_version": np.__version__,
        "python_version": sys.version.split()[0],
        "converter": converter,
        "train_windows": counts_by_class(ds.train),
        "test_windows": counts_by_class(ds.test),
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "pip_freeze": pip_freeze(),
    }
    meta.update(extra)
    return meta


def write_artifacts(tflite: bytes, meta: dict, parity_X: np.ndarray, parity_probs: np.ndarray,
                    model_path: Path = MODEL_PATH, metadata_path: Path = METADATA_PATH,
                    parity_path: Path = PARITY_PATH) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(tflite)
    metadata_path.write_text(json.dumps(meta, indent=2) + "\n")
    np.savez(parity_path, X=parity_X.astype(np.float32), probs=parity_probs.astype(np.float32),
             classes=np.array(CLASSES))
