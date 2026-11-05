"""Builds the train/test windows from labeled captures.

Plain-language overview: each capture goes through ml/preprocess.prepare()
(the same path the app uses), with 50% overlapping windows. Because there
is only one recording per activity, the split is by time WITHIN each
recording: windows that end in the first 70% are training data, windows
that start at least one full window after that point are test data, and
windows in between are thrown away -- so no overlapping window ever has
samples in both sets.

numpy only (no TensorFlow, no PySide6), so tests can check it in the app
environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import spec
from ..inference import CLASSES
from ..preprocess import prepare
from ..recording import from_legacy_capture

REPO_ROOT = Path(__file__).resolve().parents[3]
CAPTURES = [
    (REPO_ROOT / "HeartWatch" / "test" / "Standing.txt", "Standing"),
    (REPO_ROOT / "HeartWatch" / "test" / "Walking.txt", "Walking"),
]
TRAIN_FRACTION = 0.7


@dataclass
class Split:
    X: np.ndarray                  # float32 [M, 50, 6], normalized
    y: np.ndarray                  # int64 [M], index into CLASSES
    end_ts: np.ndarray             # int64 [M]
    source: list[str] = field(default_factory=list)  # capture file name per window

    def __len__(self) -> int:
        return len(self.y)


@dataclass
class Dataset:
    train: Split
    test: Split
    captures: list[dict]           # per-capture provenance and counts


def blocked_split(n_samples: int, start_index: np.ndarray, train_fraction: float = TRAIN_FRACTION):
    """(train_mask, test_mask) over windows. cut = floor(n * fraction);
    train: window ends at or before cut; test: window starts at or after
    cut + WINDOW_SAMPLES (a gap of at least one window); others dropped."""
    w = spec.WINDOW_SAMPLES
    cut = int(n_samples * train_fraction)
    start_index = np.asarray(start_index, dtype=np.int64)
    train = start_index + w <= cut
    test = start_index >= cut + w
    return train, test


def _concat(parts: list[Split]) -> Split:
    if not parts:
        return Split(np.empty((0, spec.WINDOW_SAMPLES, len(spec.CHANNELS)), np.float32),
                     np.empty(0, np.int64), np.empty(0, np.int64), [])
    return Split(
        np.concatenate([p.X for p in parts]),
        np.concatenate([p.y for p in parts]),
        np.concatenate([p.end_ts for p in parts]),
        [s for p in parts for s in p.source],
    )


def build_dataset(captures=CAPTURES, hop_samples: int = spec.TRAIN_HOP_SAMPLES) -> Dataset:
    train_parts, test_parts, info = [], [], []
    for path, label in captures:
        rec = from_legacy_capture(path, label)
        w = prepare(rec, hop_samples)
        train_mask, test_mask = blocked_split(w.n_samples, w.start_index)
        y = np.full(len(w.end_ts), CLASSES.index(rec.label), dtype=np.int64)
        name = Path(path).name
        for parts, mask in ((train_parts, train_mask), (test_parts, test_mask)):
            parts.append(Split(w.X[mask], y[mask], w.end_ts[mask], [name] * int(mask.sum())))
        info.append({
            "path": str(Path(path).resolve().relative_to(REPO_ROOT)),
            "label": rec.label,
            "sha256": rec.meta["sha256"],
            "line_counts": rec.meta["line_counts"],
            "hardware_note": rec.meta["hardware_note"],
            "raw_samples": len(rec),
            "duration_s": float(rec.ts_ms[-1] - rec.ts_ms[0]) / 1000,
            "samples_after_trim": w.n_samples,
            "windows": int(len(w.end_ts)),
            "windows_dropped_gap": w.dropped,
            "train_windows": int(train_mask.sum()),
            "test_windows": int(test_mask.sum()),
            "windows_dropped_split_gap": int((~train_mask & ~test_mask).sum()),
        })
    return Dataset(_concat(train_parts), _concat(test_parts), info)


def counts_by_class(split: Split) -> dict[str, int]:
    return {c: int((split.y == i).sum()) for i, c in enumerate(CLASSES)}
