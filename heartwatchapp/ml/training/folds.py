"""Whole-file and whole-wearer evaluation folds (no TensorFlow).

Plain-language overview: a score only means something if the model is
tested on data it never saw. Neighbouring windows from the same file look
almost alike, so a score from windows shuffled across files mostly
measures "does it recognise this file?". So scores here always hold out
whole files:

- Leave one file out: one fold per `train` file in the manifest. That file
  is the test set, every other `train` file is the training set. If the
  file is the only one of its activity (BobbySitting today), there's
  nothing to learn that activity from, so the fold is marked "can't test:
  only file of its class" and skipped -- never scored as 0 or 100.
- Leave one wearer out (Standing vs Walking only): train on one person's
  files, test on the other's.

Test windows don't overlap (the live hop, as the app sees them); training
windows use the 50% training hop. For every fold the CNN (trained by the
caller's `fit_predict`, once per seed) and the three baselines (majority,
motion rule, tilt only) are all fitted on that fold's training set only.

numpy only, so tests run it in the app environment with a stub CNN.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .. import spec
from ..inference import CLASSES
from ..manifest import REPO_ROOT, CaptureEntry, load_recording
from ..preprocess import prepare
from . import baselines as bl
from .dataset import Split, _concat, counts_by_class

SEEDS = (0, 1, 2)
WEARER_CLASSES = ("Standing", "Walking")
ONLY_FILE = "can't test: only file of its class"

# fit_predict(train_X, train_y, test_X, seed) -> probabilities [len(test_X), len(CLASSES)]
FitPredict = Callable[[np.ndarray, np.ndarray, np.ndarray, int], np.ndarray]


@dataclass
class FileWindows:
    entry: CaptureEntry
    train: Split                 # 50% overlap (training hop)
    live: Split                  # no overlap (live hop): what a test sees
    info: dict = field(default_factory=dict)

    @property
    def path(self) -> str:
        return self.entry.path


@dataclass(frozen=True)
class Fold:
    name: str
    kind: str                    # "file" or "wearer"
    test_paths: tuple[str, ...]
    train_paths: tuple[str, ...]
    skipped: str | None = None   # reason, or None when the fold is scored


def _split(windows, label: str, path: str) -> Split:
    y = np.full(len(windows.end_ts), CLASSES.index(label), dtype=np.int64)
    return Split(windows.X, y, windows.end_ts, [path] * len(y))


def load_file_windows(entries: list[CaptureEntry], repo_root: str | Path = REPO_ROOT) -> list[FileWindows]:
    """Every manifest row's windows at both hops, plus what the report needs."""
    out = []
    for e in entries:
        rec = load_recording(e, repo_root)
        tw = prepare(rec, spec.TRAIN_HOP_SAMPLES)
        lw = prepare(rec, spec.LIVE_HOP_SAMPLES)
        out.append(FileWindows(e, _split(tw, e.label, e.path), _split(lw, e.label, e.path), {
            "sha256": rec.meta["sha256"],
            "raw_samples": len(rec),
            "eol_rows": rec.meta["eol_rows"],
            "segments": rec.meta["segments"],
            "clipping": rec.meta["clipping"],
            "dropped_train": tw.dropped,
            "dropped_live": lw.dropped,
        }))
    return out


def train_files(files: list[FileWindows]) -> list[FileWindows]:
    return [f for f in files if f.entry.use == "train"]


def leave_one_file_out(files: list[FileWindows]) -> list[Fold]:
    usable = train_files(files)
    folds = []
    for f in usable:
        others = tuple(g.path for g in usable if g is not f)
        same_class = [g for g in usable if g is not f and g.entry.label == f.entry.label]
        skipped = None
        if not same_class:
            skipped = ONLY_FILE
        elif len(f.live) == 0:
            skipped = "can't test: no test windows"
        folds.append(Fold(f"file {f.path}", "file", (f.path,), others, skipped))
    return folds


def leave_one_wearer_out(files: list[FileWindows], classes=WEARER_CLASSES) -> list[Fold]:
    usable = [f for f in train_files(files) if f.entry.label in classes]
    wearers = sorted({f.entry.wearer for f in usable})
    folds = []
    for w in wearers:
        test = [f for f in usable if f.entry.wearer == w]
        train = [f for f in usable if f.entry.wearer != w]
        missing = sorted({f.entry.label for f in test} - {f.entry.label for f in train})
        skipped = f"can't test: no other wearer has {', '.join(missing)}" if missing else None
        folds.append(Fold(f"wearer {w}", "wearer", tuple(f.path for f in test),
                          tuple(f.path for f in train), skipped))
    return folds


def fold_splits(fold: Fold, files: list[FileWindows]) -> tuple[Split, Split]:
    """(training windows, test windows) for one fold. Training: the training
    hop of every training file; test: the live hop of the held-out files."""
    by_path = {f.path: f for f in files}
    train = _concat([by_path[p].train for p in fold.train_paths])
    test = _concat([by_path[p].live for p in fold.test_paths])
    return train, test


def window_counts(files: list[FileWindows]) -> dict:
    usable = train_files(files)
    return {"train": counts_by_class(_concat([f.train for f in usable])),
            "live": counts_by_class(_concat([f.live for f in usable]))}


def _baseline_json(b: dict) -> dict:
    return {"name": b["name"], "description": b["description"], "train": list(b["train"]),
            "test": list(b["test"]), "fitted": b["fitted"],
            "confusion_test": np.asarray(b["confusion_test"]).tolist()}


def evaluate_folds(folds: list[Fold], files: list[FileWindows], fit_predict: FitPredict,
                   seeds=SEEDS) -> list[dict]:
    """Score every fold: the CNN once per seed (mean and worst reported) and
    the three baselines, all fitted on the fold's training set only.
    Skipped folds keep their reason in "status" and get no scores."""
    results = []
    for fold in folds:
        r = {"name": fold.name, "kind": fold.kind, "test_paths": list(fold.test_paths),
             "train_paths": list(fold.train_paths)}
        if fold.skipped:
            results.append({**r, "status": fold.skipped})
            continue
        train, test = fold_splits(fold, files)
        test_counts = {c: n for c, n in counts_by_class(test).items() if n}
        r.update({
            "status": "scored",
            "true_class": next(iter(test_counts)) if len(test_counts) == 1 else "mixed",
            "test_counts": test_counts,
            "train_counts": {c: n for c, n in counts_by_class(train).items() if n},
            "test_windows": len(test),
            "baselines": [_baseline_json(b) for b in bl.run_baselines(train, test)],
        })
        per_seed, predictions, confusions = [], [], []
        for seed in seeds:
            probs = np.asarray(fit_predict(train.X, train.y, test.X, seed))
            pred = probs.argmax(axis=1)
            per_seed.append(list(bl.score(test.y, pred)))
            predictions.append([CLASSES[i] for i in pred])
            confusions.append(bl.confusion(test.y, pred).tolist())
        correct = [c for c, _ in per_seed]
        r["cnn"] = {
            "seeds": list(seeds),
            "per_seed": per_seed,
            "mean_correct": float(np.mean(correct)),
            "worst": per_seed[int(np.argmin(correct))],
            "predictions": predictions,
            "confusion_per_seed": confusions,
        }
        r["test_windows_detail"] = [{"file": s, "end_ts": int(t), "true": CLASSES[int(y)]}
                                    for s, t, y in zip(test.source, test.end_ts, test.y)]
        results.append(r)
    return results


def fold_summary(folds: list[Fold], files: list[FileWindows]) -> list[dict]:
    """Window counts per fold, without training anything (--folds-only)."""
    out = []
    for fold in folds:
        train, test = fold_splits(fold, files)
        out.append({"name": fold.name, "kind": fold.kind, "test_paths": list(fold.test_paths),
                    "skipped": fold.skipped,
                    "train_windows": {c: n for c, n in counts_by_class(train).items() if n},
                    "test_windows": {c: n for c, n in counts_by_class(test).items() if n}})
    return out
