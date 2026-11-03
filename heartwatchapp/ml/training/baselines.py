"""Three simple non-AI comparisons for the CNN. numpy only.

1. Majority class: always guess the most common training activity.
2. Motion rule: how much the accelerometer magnitude varies within a
   window (its standard deviation, in g). Each activity's training average
   of that number is computed; a window gets the activity whose average is
   closest -- with two activities that is one threshold halfway between.
3. Tilt only: the average of each accelerometer axis over the window
   (which way gravity points, i.e. how the wrist is held), nearest
   activity average. If this alone scores near 100%, the recordings may be
   told apart by how the watch was tilted, not by the activity.

Everything is fitted on the training windows only.
"""

from __future__ import annotations

import numpy as np

from .. import spec
from ..inference import CLASSES


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """[len(CLASSES), len(CLASSES)] counts: rows = true, columns = predicted."""
    m = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        m[int(t), int(p)] += 1
    return m


def score(y_true, y_pred) -> tuple[int, int]:
    """(correct, total)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return int((y_true == y_pred).sum()), int(len(y_true))


def _accel_g(X: np.ndarray) -> np.ndarray:
    """Normalized windows -> accelerometer channels back in g."""
    return np.asarray(X, dtype=np.float64)[:, :, :3] * spec.ACCEL_NORM_G


def motion_feature(X: np.ndarray) -> np.ndarray:
    """[M, 1]: std over time of |a| (g)."""
    return np.linalg.norm(_accel_g(X), axis=2).std(axis=1)[:, None]


def tilt_feature(X: np.ndarray) -> np.ndarray:
    """[M, 3]: mean of ax, ay, az (g)."""
    return _accel_g(X).mean(axis=1)


class NearestCentroid:
    def fit(self, F: np.ndarray, y: np.ndarray) -> "NearestCentroid":
        self.classes_ = np.array(sorted(set(int(v) for v in y)), dtype=np.int64)
        self.centroids_ = np.stack([F[y == c].mean(axis=0) for c in self.classes_])
        return self

    def predict(self, F: np.ndarray) -> np.ndarray:
        d = np.linalg.norm(F[:, None, :] - self.centroids_[None, :, :], axis=2)
        return self.classes_[d.argmin(axis=1)]


def majority_class(y_train: np.ndarray) -> int:
    """Most common training class; ties go to the earlier one in CLASSES."""
    counts = np.bincount(y_train, minlength=len(CLASSES))
    return int(counts.argmax())


def run_baselines(train, test) -> list[dict]:
    """Each: name, description, train/test (correct, total), confusion
    matrices, and the fitted numbers."""
    results = []

    m = majority_class(train.y)
    results.append(_result(
        "Majority class", f"always '{CLASSES[m]}'", train, test,
        np.full(len(train), m), np.full(len(test), m), {"class": CLASSES[m]},
    ))

    for name, desc, feature in (
        ("Motion rule", "std of accel magnitude per window, nearest class mean (train-fitted)",
         motion_feature),
        ("Tilt only", "per-axis accel mean per window, nearest centroid (train-fitted)",
         tilt_feature),
    ):
        nc = NearestCentroid().fit(feature(train.X), train.y)
        fitted = {CLASSES[c]: [round(float(v), 4) for v in cen]
                  for c, cen in zip(nc.classes_, nc.centroids_)}
        if name == "Motion rule" and len(nc.classes_) == 2:
            fitted["threshold_g"] = round(float(nc.centroids_[:, 0].mean()), 4)
        results.append(_result(
            name, desc, train, test,
            nc.predict(feature(train.X)) if len(train) else np.empty(0, np.int64),
            nc.predict(feature(test.X)) if len(test) else np.empty(0, np.int64),
            fitted,
        ))
    return results


def _result(name, desc, train, test, pred_train, pred_test, fitted) -> dict:
    return {
        "name": name,
        "description": desc,
        "train": score(train.y, pred_train),
        "test": score(test.y, pred_test),
        "confusion_train": confusion(train.y, pred_train),
        "confusion_test": confusion(test.y, pred_test),
        "fitted": fitted,
    }
