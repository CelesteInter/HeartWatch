"""Activity classifier: runs the exported .tflite model with LiteRT.

Plain-language overview: training (ml/training/, its own environment)
exports a small model file plus a model_meta.json describing what it
expects. ActivityClassifier.load() reads both, refuses the model if the
metadata disagrees with ml/spec.py (a different rate, window or scaling
would mean silently wrong answers), and then predict() labels windows
produced by ml/preprocess.prepare().

Since v2 a model can see its windows through an input transform
(model_meta.json "input_transform", ml/transforms.py) and can have its own
classes. The contract:
- "input_transform" must be a known transform. Metadata without one is an
  older model and loads as "raw" (logged); for those, "classes" must still
  be exactly CLASSES.
- "input_channels" (default: 6) must equal the transform's channel count
  and the .tflite input shape [1, 50, input_channels].
- "classes" come from the metadata and must match the model's output
  count. The global CLASSES (the app's four activities) never change.
- "label_map" says which model class each manifest/app label counts as
  (v2: Standing and Sitting -> Still, Walking -> Walking). Without one,
  each class maps to itself.
"channels" stays the six sensor channels prepare() produces.
predict_proba() always takes prepare()'s [M, 50, 6] windows and applies
the model's transform itself, through preprocess.model_input() -- the same
function the export script trains with.

Not wired into any screen yet: the Live Monitor and the predictions table
still use the "demo-dummy" generator.

Importing this module stays cheap: no numpy, no LiteRT at module level
(data/stats.py, the Ask bar and others import CLASSES from here). LiteRT
is imported inside load().
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from . import spec, transforms

log = logging.getLogger(__name__)

DEFAULT_TRANSFORM = "raw"   # what metadata without "input_transform" means
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


def check_metadata(meta: dict) -> dict:
    """Raise ModelMetadataError listing every problem: a field that differs
    from what the app expects, an unknown input_transform, an
    input_channels that doesn't fit it, bad classes or label_map, or no
    model_version (every prediction stored in the database is tagged with
    it, so an unnamed model can't be traced). Returns what the model
    expects: input_transform, input_channels, classes, label_map."""
    transform = meta.get("input_transform")
    if transform is None:
        transform = DEFAULT_TRANSFORM
        log.info("model %r has no input_transform in its metadata: loading it as %r",
                 meta.get("model_version"), DEFAULT_TRANSFORM)
    expected = expected_metadata()
    if transform != DEFAULT_TRANSFORM:
        expected.pop("classes")   # a transformed model brings its own classes
    problems = [
        f"{key}: model has {meta.get(key)!r}, app expects {want!r}"
        for key, want in expected.items()
        if meta.get(key) != want
    ]
    version = meta.get("model_version")
    if not isinstance(version, str) or not version.strip():
        problems.insert(0, f"model_version: missing or empty ({version!r})")

    channels = None
    if transform not in transforms.TRANSFORMS:
        problems.append(f"input_transform: {transform!r} is unknown (known: {sorted(transforms.TRANSFORMS)})")
    else:
        channels = transforms.TRANSFORMS[transform]
        got = meta.get("input_channels", len(spec.CHANNELS))
        if got != channels:
            problems.append(f"input_channels: model has {got!r}, transform {transform!r} gives {channels}")

    classes = meta.get("classes")
    if not (isinstance(classes, list) and classes and all(isinstance(c, str) and c for c in classes)
            and len(set(classes)) == len(classes)):
        problems.append(f"classes: {classes!r} is not a list of distinct names")
        classes = None
    label_map = meta.get("label_map")
    if label_map is None:
        label_map = {c: c for c in classes or []}
    elif not (isinstance(label_map, dict) and classes is not None
              and all(isinstance(k, str) and v in classes for k, v in label_map.items())):
        problems.append(f"label_map: {label_map!r} must map label names to the model's classes {classes!r}")

    if problems:
        raise ModelMetadataError("model metadata mismatch -- " + "; ".join(problems))
    return {"input_transform": transform, "input_channels": channels,
            "classes": list(classes), "label_map": dict(label_map)}


class ActivityClassifier:
    """Use ActivityClassifier.load(...), not the constructor."""

    def __init__(self, interpreter, metadata: dict, info: dict | None = None):
        info = info or check_metadata(metadata)
        self._interpreter = interpreter
        self.metadata = metadata
        self.model_version = metadata.get("model_version")
        self.input_transform = info["input_transform"]
        self.input_channels = info["input_channels"]
        self.input_shape = (1, spec.WINDOW_SAMPLES, self.input_channels)
        self.classes = info["classes"]
        self.label_map = info["label_map"]
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
        return cls.from_interpreter(interpreter, meta)

    @classmethod
    def from_interpreter(cls, interpreter, meta: dict) -> "ActivityClassifier":
        """The checks between the metadata and the model itself (input
        shape, output count), for an interpreter that's already allocated."""
        info = check_metadata(meta)
        want = (1, spec.WINDOW_SAMPLES, info["input_channels"])
        shape = tuple(int(d) for d in interpreter.get_input_details()[0]["shape"])
        if shape != want:
            raise ModelMetadataError(f"model input shape {shape}, its metadata "
                                     f"({info['input_transform']!r}, input_channels "
                                     f"{info['input_channels']}) says {want}")
        n_out = int(interpreter.get_output_details()[0]["shape"][-1])
        if n_out != len(info["classes"]):
            raise ModelMetadataError(f"model has {n_out} outputs, its metadata lists "
                                     f"{len(info['classes'])} classes {info['classes']}")
        return cls(interpreter, meta, info)

    def model_class(self, label: str | None) -> str | None:
        """The model class an app/manifest label counts as (label_map), or
        None when this model has no class for it (e.g. Running for v2)."""
        return self.label_map.get(label) if label is not None else None

    def predict_proba(self, windows):
        """windows: float32 [M, WINDOW_SAMPLES, 6] from preprocess.prepare()
        (already normalized, NOT transformed: the model's input_transform
        is applied here). Returns float32 [M, len(self.classes)]."""
        import numpy as np

        from .preprocess import model_input

        X = np.asarray(windows, dtype=np.float32)
        if X.ndim != 3 or X.shape[1:] != INPUT_SHAPE[1:]:
            raise ValueError(f"windows must be preprocess.prepare() output [M, {INPUT_SHAPE[1]}, "
                             f"{INPUT_SHAPE[2]}], got {X.shape}")
        X = np.asarray(model_input(X, self.input_transform), dtype=np.float32)
        out = np.empty((len(X), len(self.classes)), dtype=np.float32)
        for i, window in enumerate(X):
            self._interpreter.set_tensor(self._input["index"], window[None, ...])
            self._interpreter.invoke()
            out[i] = self._interpreter.get_tensor(self._output["index"])[0]
        return out

    def predict(self, windows) -> list[tuple[str, float]]:
        """(class_name, confidence) per window, in this model's classes."""
        probs = self.predict_proba(windows)
        return [(self.classes[int(p.argmax())], float(p.max())) for p in probs]
