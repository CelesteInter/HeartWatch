"""Model input transforms, by name. The one copy, shared by training and the app.

Plain-language overview: preprocess.prepare() turns a recording into
normalized 5-second windows of the six motion channels. A model may then
want those windows changed before it sees them -- v2 only wants "how hard
the wrist moves and turns", not which way it points. Each model's
model_meta.json names its transform ("input_transform"), and both the
export script (training) and ActivityClassifier (prediction) apply it
through preprocess.model_input(), which calls apply() here. There is no
second copy anywhere, so training and the app can't drift apart.

- raw:       the windows unchanged, 6 channels (v0, v1).
- magnitude: 2 channels that don't depend on the wrist angle:
             |accel| minus its window average, and |gyro| (handoff 12's
             "magnitude" input variant; input_variants.py imports it from
             here).

No numpy at import time (ml.inference imports this module and must stay
cheap); numpy is imported inside the functions.
"""

from __future__ import annotations

# name -> number of channels the model receives
TRANSFORMS = {"raw": 6, "magnitude": 2}


def raw(X):
    return X


def magnitude(X):
    """[n, 50, 2]: |a| - mean(|a|) over the window, and |g|."""
    import numpy as np

    a = np.linalg.norm(X[:, :, :3], axis=2)
    g = np.linalg.norm(X[:, :, 3:6], axis=2)
    return np.stack([a - a.mean(axis=1, keepdims=True), g], axis=2).astype(X.dtype)


_FUNCTIONS = {"raw": raw, "magnitude": magnitude}


def output_channels(name: str) -> int:
    if name not in TRANSFORMS:
        raise ValueError(f"unknown input transform {name!r} (known: {sorted(TRANSFORMS)})")
    return TRANSFORMS[name]


def apply(name: str, X):
    """X: prepare()'s windows [n, 50, 6]. Returns [n, 50, output_channels(name)]."""
    output_channels(name)
    return _FUNCTIONS[name](X)
