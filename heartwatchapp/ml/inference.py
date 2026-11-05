"""Activity classifier: runs the exported .tflite model with LiteRT.

Plain-language overview: training (ml/training/, its own environment)
exports a small model file plus a model_meta.json describing what it
expects. ActivityClassifier.load() reads both, refuses the model if the
metadata disagrees with ml/spec.py or CLASSES (different classes, rate,
window or scaling would mean silently wrong answers), and then predict()
labels windows produced by ml/preprocess.prepare().

Not wired into any screen yet: the Live Monitor and the predictions table
still use the "demo-dummy" generator.

Importing this module stays cheap: no numpy, no LiteRT at module level
(data/stats.py, the Ask bar and others import CLASSES from here). LiteRT
is imported inside load().
"""

from __future__ import annotations

import json
from pathlib import Path

from . import spec

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODEL_DIR / "har_cnn.tflite"
METADATA_PATH = MODEL_DIR / "model_meta.json"
PARITY_PATH = MODEL_DIR / "parity.npz"
CLASSES = ["Sitting", "Walking", "Standing", "Running"]
INPUT_SHAPE = spec.INPUT_SHAPE  # (batch, window, channels) = (1, 50, 6)


class ModelMetadataError(ValueError):
    """model_meta.json doesn't match what this build of the app expects."""


def expected_metadata() -> dict:
    """The fields a model's metadata must match exactly."""
    return {
        "classes": list(CLASSES),
        "rate_hz": spec.RATE_HZ,
        "window_samples": spec.WINDOW_SAMPLES,
        "channels": list(spec.CHANNELS),
        "normalization": {"accel_g": spec.ACCEL_NORM_G, "gyro_dps": spec.GYRO_NORM_DPS},
    }


def check_metadata(meta: dict) -> None:
    """Raise ModelMetadataError listing every field that differs."""
    problems = [
        f"{key}: model has {meta.get(key)!r}, app expects {want!r}"
        for key, want in expected_metadata().items()
        if meta.get(key) != want
    ]
    if problems:
        raise ModelMetadataError("model metadata mismatch -- " + "; ".join(problems))


class ActivityClassifier:
    """Use ActivityClassifier.load(...), not the constructor."""

    def __init__(self, interpreter, metadata: dict):
        self._interpreter = interpreter
        self.metadata = metadata
        self.model_version = metadata.get("model_version")
        self._input = interpreter.get_input_details()[0]
        self._output = interpreter.get_output_details()[0]

    @classmethod
    def load(
        cls, model_path: str | Path = MODEL_PATH, metadata_path: str | Path = METADATA_PATH
    ) -> "ActivityClassifier":
        meta = json.loads(Path(metadata_path).read_text())
        check_metadata(meta)  # before touching the model or importing LiteRT
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        from ai_edge_litert.interpreter import Interpreter

        interpreter = Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()
        shape = tuple(int(d) for d in interpreter.get_input_details()[0]["shape"])
        if shape != INPUT_SHAPE:
            raise ModelMetadataError(f"model input shape {shape}, app expects {INPUT_SHAPE}")
        n_out = int(interpreter.get_output_details()[0]["shape"][-1])
        if n_out != len(CLASSES):
            raise ModelMetadataError(f"model has {n_out} outputs, app expects {len(CLASSES)}")
        return cls(interpreter, meta)

    def predict_proba(self, windows):
        """windows: float32 [M, WINDOW_SAMPLES, 6] from preprocess.prepare()
        (already normalized). Returns float32 [M, len(CLASSES)]."""
        import numpy as np

        X = np.asarray(windows, dtype=np.float32)
        if X.ndim != 3 or X.shape[1:] != INPUT_SHAPE[1:]:
            raise ValueError(f"windows must be [M, {INPUT_SHAPE[1]}, {INPUT_SHAPE[2]}], got {X.shape}")
        out = np.empty((len(X), len(CLASSES)), dtype=np.float32)
        for i, window in enumerate(X):
            self._interpreter.set_tensor(self._input["index"], window[None, ...])
            self._interpreter.invoke()
            out[i] = self._interpreter.get_tensor(self._output["index"])[0]
        return out

    def predict(self, windows) -> list[tuple[str, float]]:
        """(class_name, confidence) per window, CLASSES order."""
        probs = self.predict_proba(windows)
        return [(CLASSES[int(p.argmax())], float(p.max())) for p in probs]
