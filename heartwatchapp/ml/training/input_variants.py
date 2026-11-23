"""Input variants for the input experiment (handoff 12). numpy only.

Plain-language overview: v1 learned which way the watch was turned (the
average of each accelerometer axis is gravity, i.e. the wrist angle)
instead of how the wrist moved. Each variant below changes the network's
input so the angle is removed or made useless, while keeping the motion:

- raw:          v1's input, unchanged (the reference).
- centered:     subtract each window's own average from each accel axis
                (removes gravity, keeps the motion shape per axis); gyro
                unchanged.
- magnitude:    two channels that don't depend on the angle at all:
                |accel| minus its window average, and |gyro|.
- rotation_aug: v1's input, but every TRAINING window also gets 9 copies
                turned by random 3D rotations, so the angle stops being
                a useful clue. Test windows are never rotated.

All of these work on prepare()'s normalized windows (accel / 4 g, gyro /
500 deg/s), so |accel| and |gyro| are in those units.

The experiment uses two labels of its own: Still (Standing or Sitting --
without the angle they're both just "not moving") and Walking. The app's
CLASSES don't change.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .dataset import Split
from .folds import FileWindows

EXPERIMENT_CLASSES = ("Still", "Walking")
LABEL_MAP = {"Standing": "Still", "Sitting": "Still", "Walking": "Walking"}
ROTATION_COPIES = 9


def experiment_label(label: str) -> str:
    """Standing / Sitting -> Still, Walking -> Walking. Anything else
    (Running, which has no usable data) is a ValueError."""
    if label not in LABEL_MAP:
        raise ValueError(f"label {label!r} has no experiment class (only {sorted(LABEL_MAP)})")
    return LABEL_MAP[label]


def relabel(files: list[FileWindows]) -> list[FileWindows]:
    """The manifest's `train` files with Still/Walking labels: entry.label
    becomes the experiment label and y becomes its index in
    EXPERIMENT_CLASSES. Windows are the same arrays. `exclude` rows are
    dropped (they never take part)."""
    out = []
    for f in files:
        if f.entry.use != "train":
            continue
        label = experiment_label(f.entry.label)
        k = EXPERIMENT_CLASSES.index(label)

        def re(split: Split) -> Split:
            return Split(split.X, np.full(len(split.y), k, dtype=np.int64), split.end_ts, list(split.source))

        out.append(FileWindows(dataclasses.replace(f.entry, label=label), re(f.train), re(f.live),
                               dict(f.info, original_label=f.entry.label)))
    return out


# ------------------------------------------------------------- transforms

def raw(X: np.ndarray) -> np.ndarray:
    return X


def centered(X: np.ndarray) -> np.ndarray:
    """Each accel axis minus its own window mean; gyro unchanged."""
    out = np.array(X, copy=True)
    out[:, :, :3] = X[:, :, :3] - X[:, :, :3].mean(axis=1, keepdims=True)
    return out


def magnitude(X: np.ndarray) -> np.ndarray:
    """[n, 50, 2]: |a| - mean(|a|) over the window, and |g|."""
    a = np.linalg.norm(X[:, :, :3], axis=2)
    g = np.linalg.norm(X[:, :, 3:6], axis=2)
    return np.stack([a - a.mean(axis=1, keepdims=True), g], axis=2).astype(X.dtype)


def random_rotations(n: int, rng: np.random.Generator) -> np.ndarray:
    """[n, 3, 3] uniformly random rotations (random unit quaternions)."""
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], axis=1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], axis=1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=1),
    ], axis=1)


def rotate(X: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Rotate each window's accel and gyro by that window's R (the same R
    for both: they're measured in the same sensor frame)."""
    a = np.einsum("nij,ntj->nti", R, X[:, :, :3])
    g = np.einsum("nij,ntj->nti", R, X[:, :, 3:6])
    return np.concatenate([a, g], axis=2).astype(X.dtype)


def rotation_augment(X: np.ndarray, y: np.ndarray, seed: int,
                     copies: int = ROTATION_COPIES) -> tuple[np.ndarray, np.ndarray]:
    """The windows themselves, then `copies` rotated copies of all of them
    (a fresh random rotation per window per copy). Same seed, same output."""
    rng = np.random.default_rng(seed)
    parts = [X] + [rotate(X, random_rotations(len(X), rng)) for _ in range(copies)]
    return np.concatenate(parts), np.tile(np.asarray(y), copies + 1)


# --------------------------------------------------------------- registry

@dataclass(frozen=True)
class Variant:
    name: str
    channels: int
    description: str
    transform: Callable[[np.ndarray], np.ndarray]       # applied to training AND test windows
    augment: Callable | None = None                     # training windows only: (X, y, seed) -> (X, y)

    def prepare_train(self, X: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
        Xt = self.transform(X)
        return self.augment(Xt, y, seed) if self.augment else (Xt, np.asarray(y))


VARIANTS = {v.name: v for v in (
    Variant("raw", 6, "v1's input, unchanged (reference)", raw),
    Variant("centered", 6, "each accel axis minus its window mean; gyro unchanged", centered),
    Variant("magnitude", 2, "|a| - window mean of |a|, and |g|", magnitude),
    Variant("rotation_aug", 6, "raw input; each training window + 9 randomly rotated copies "
            "(seed = the run's seed); test windows unrotated", raw, rotation_augment),
)}
