"""TFLite model loader + inference for activity classification.

Stub for the shell milestone. Pipeline (per the handoff):

    windowed [ax, ay, az, gx, gy, gz]  ->  1D CNN  ->  TFLite inference

Build step 6 loads a real model and runs it on the dedicated inference thread.
"""

from __future__ import annotations

from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent / "models" / "har_cnn.tflite"
CLASSES = ["Sitting", "Walking", "Standing", "Running"]
INPUT_SHAPE = (1, 128, 6)  # (batch, window, channels)


class ActivityClassifier:
    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = model_path
        self._interpreter = None

    def load(self) -> None:
        raise NotImplementedError("wired in build step 6 (needs tflite-runtime)")

    def predict(self, window) -> tuple[str, float]:
        """window: array-like shaped (128, 6). Returns (class_name, confidence)."""
        raise NotImplementedError("wired in build step 6")
