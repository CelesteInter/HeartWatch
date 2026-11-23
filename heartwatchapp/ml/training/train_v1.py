"""Trains and scores the v1 activity CNN from the capture manifest.

Run from the repo root, in the training environment:

    heartwatchapp/ml/training/.venv/bin/python -B -m heartwatchapp.ml.training.train_v1
    heartwatchapp/ml/training/.venv/bin/python -B -m heartwatchapp.ml.training.train_v1 --folds-only

Plain-language overview, in order:
1. Read ml/captures.csv and the `train` files (Running.txt is `exclude`).
2. Leave one file out, and leave one wearer out (Standing vs Walking): for
   every fold, train the CNN three times (seeds 0, 1, 2) on that fold's
   training files only, and score it next to the majority, motion-rule
   and tilt-only baselines on the held-out files (ml/training/folds.py).
3. Copy the current model in ml/models/ to ml/models/archive/<its
   version>/ (once; an existing archive is never overwritten, nothing is
   ever deleted), then train the final v1 model on every `train` file and
   export .tflite + model_meta.json + parity.npz.
4. Write ml/runs/<timestamp>/report.md and scores.json.

--folds-only does step 1 and prints the window counts per fold. It trains
nothing, imports no TensorFlow and writes nothing.

The CNN, its hyperparameters (40 epochs, batch 16, Adam 1e-3, no class
weights) and seeding are train.py's, unchanged. Importing this module
doesn't import TensorFlow; only the training functions do.
"""

from __future__ import annotations

import argparse
import datetime
import json
import shutil
from pathlib import Path
from typing import Callable

import numpy as np

from .. import spec
from ..inference import CLASSES, METADATA_PATH, MODEL_DIR, expected_metadata
from ..manifest import MANIFEST_PATH, REPO_ROOT, load_manifest, manifest_sha256
from . import export, folds, report
from .dataset import _concat, counts_by_class
from .train import RUNS_DIR, build_model

MODEL_VERSION = "har-cnn-v1"
FINAL_SEED = 0
EPOCHS, BATCH_SIZE = 40, 16
PARITY_MAX_WINDOWS = 16
MODEL_FILES = ("har_cnn.tflite", "model_meta.json", "parity.npz")


# ------------------------------------------------------- Keras (TF only here)

def keras_fit(X: np.ndarray, y: np.ndarray, seed: int, epochs: int = EPOCHS,
              batch_size: int = BATCH_SIZE):
    """train.py's CNN and training settings, for one seed."""
    import keras
    import tensorflow as tf

    tf.config.experimental.enable_op_determinism()
    keras.utils.set_random_seed(seed)
    model = build_model(seed)
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(X, y, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=0)
    return model


class _KerasPredictor:
    """A trained Keras model as a plain X -> probabilities callable."""

    def __init__(self, model):
        self.keras_model = model

    def __call__(self, X):
        if not len(X):
            return np.empty((0, len(CLASSES)), np.float32)
        return self.keras_model.predict(X, verbose=0)


def keras_fit_predict(train_X, train_y, test_X, seed, epochs=EPOCHS, batch_size=BATCH_SIZE):
    return _KerasPredictor(keras_fit(train_X, train_y, seed, epochs, batch_size))(test_X)


def keras_fit_final(X, y, seed, epochs=EPOCHS, batch_size=BATCH_SIZE):
    return _KerasPredictor(keras_fit(X, y, seed, epochs, batch_size))


def keras_convert(predictor):
    return export.convert(predictor.keras_model)


def keras_versions() -> dict:
    import sys

    import keras
    import tensorflow as tf

    return {"tensorflow_version": tf.__version__, "keras_version": keras.__version__,
            "numpy_version": np.__version__, "python_version": sys.version.split()[0],
            "pip_freeze": export.pip_freeze()}


# ---------------------------------------------------------------- helpers

def archive_existing(models_dir: Path) -> Path | None:
    """Copy the model currently in models_dir to models_dir/archive/<its
    model_version>/, unless that folder already exists. Never deletes or
    overwrites anything. Returns the archive folder, or None."""
    meta_path = models_dir / "model_meta.json"
    if not meta_path.is_file():
        return None
    try:
        version = json.loads(meta_path.read_text()).get("model_version") or "unknown"
    except (ValueError, OSError):
        version = "unknown"
    dest = models_dir / "archive" / str(version)
    if dest.exists():
        return dest
    dest.mkdir(parents=True)
    for name in MODEL_FILES:
        if (models_dir / name).is_file():
            shutil.copy2(models_dir / name, dest / name)
    return dest


def _new_run_dir(runs_dir: Path) -> Path:
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir, k = runs_dir / stamp, 2
    while run_dir.exists():
        run_dir, k = runs_dir / f"{stamp}-{k}", k + 1
    run_dir.mkdir(parents=True)
    return run_dir


def _file_rows(files: list[folds.FileWindows]) -> list[dict]:
    return [{"path": f.path, "label": f.entry.label, "wearer": f.entry.wearer, "use": f.entry.use,
             "sha256": f.info["sha256"], "train_windows": len(f.train), "live_windows": len(f.live),
             "eol_rows": f.info["eol_rows"], "segments": f.info["segments"],
             "clipping": f.info["clipping"]} for f in files]


def _fold_brief(results: list[dict]) -> list[dict]:
    """The fold results for model_meta.json (no per-window detail)."""
    keep = ("name", "kind", "test_paths", "status", "true_class", "test_windows", "test_counts")
    out = []
    for r in results:
        b = {k: r[k] for k in keep if k in r}
        if "cnn" in r:
            b["cnn"] = {k: r["cnn"][k] for k in ("seeds", "per_seed", "mean_correct", "worst")}
            b["baselines"] = {x["name"]: x["test"] for x in r["baselines"]}
        out.append(b)
    return out


# ---------------------------------------------------------------- the run

def run(manifest_path: str | Path = MANIFEST_PATH, *, repo_root: str | Path = REPO_ROOT,
        models_dir: str | Path = MODEL_DIR, runs_dir: str | Path = RUNS_DIR,
        folds_only: bool = False, fit_predict: Callable | None = None,
        fit_final: Callable | None = None, convert: Callable | None = None,
        versions: dict | None = None, seeds=folds.SEEDS, final_seed: int = FINAL_SEED,
        epochs: int = EPOCHS, batch_size: int = BATCH_SIZE) -> dict:
    """The whole v1 run; see the module docstring. The fit/convert hooks
    default to Keras; tests pass stubs so nothing is trained."""
    manifest_path, models_dir, runs_dir = Path(manifest_path), Path(models_dir), Path(runs_dir)
    entries = load_manifest(manifest_path, repo_root)
    files = folds.load_file_windows(entries, repo_root)
    usable = folds.train_files(files)
    all_folds = folds.leave_one_file_out(files) + folds.leave_one_wearer_out(files)
    counts = folds.window_counts(files)
    if folds_only:
        return {"folds": folds.fold_summary(all_folds, files), "window_counts": counts,
                "files": _file_rows(files)}

    real = fit_predict is None
    if real:
        fit_predict = lambda a, b, c, s: keras_fit_predict(a, b, c, s, epochs, batch_size)  # noqa: E731
        fit_final = lambda a, b, s: keras_fit_final(a, b, s, epochs, batch_size)  # noqa: E731
        convert = keras_convert
    archive = archive_existing(models_dir)   # before anything is written to models_dir

    results = folds.evaluate_folds(all_folds, files, fit_predict, seeds)

    train_all = _concat([f.train for f in usable])
    live_all = _concat([f.live for f in usable])
    model = fit_final(train_all.X, train_all.y, final_seed)
    idx = np.unique(np.linspace(0, len(live_all) - 1, min(PARITY_MAX_WINDOWS, len(live_all))).astype(int))
    parity_X = live_all.X[idx]
    parity_probs = np.asarray(model(parity_X))
    tflite, converter = convert(model)

    meta = {
        "model_version": MODEL_VERSION,
        **expected_metadata(),
        "bin_ms": spec.BIN_MS, "trim_s": spec.TRIM_S,
        "train_hop_samples": spec.TRAIN_HOP_SAMPLES, "live_hop_samples": spec.LIVE_HOP_SAMPLES,
        "input_shape": list(spec.INPUT_SHAPE),
        "manifest": {"path": str(manifest_path), "sha256": manifest_sha256(manifest_path)},
        "captures": [{k: r[k] for k in ("path", "label", "wearer", "sha256", "eol_rows")}
                     for r in _file_rows(usable)],
        "excluded": [f.path for f in files if f.entry.use != "train"],
        "train_windows": counts_by_class(train_all),
        "live_windows": counts_by_class(live_all),
        "folds": _fold_brief(results),
        "seeds": {"folds": list(seeds), "final": final_seed},
        "training": {"epochs": epochs, "batch_size": batch_size, "optimizer": "Adam(1e-3)",
                     "class_weights": None, "architecture": "train.build_model"},
        "notes": ["Running has no training data: this model can't predict Running.",
                  "Sitting: decided by tilt only, one session, one wearer (untestable).",
                  "Accelerometer Z offset (~+0.24 g) not corrected.",
                  "Sensor at +-2 g / +-250 deg/s defaults for every capture."],
        "git": export.git_commit(),
        "converter": converter,
        "parity_windows": int(len(parity_X)),
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        **(keras_versions() if real else (versions or {})),
    }
    export.write_artifacts(tflite, meta, parity_X, parity_probs,
                           model_path=models_dir / MODEL_FILES[0],
                           metadata_path=models_dir / MODEL_FILES[1],
                           parity_path=models_dir / MODEL_FILES[2])

    run_dir = _new_run_dir(runs_dir)
    (run_dir / "scores.json").write_text(json.dumps(results, indent=1) + "\n")
    summary = {"meta": meta, "files": _file_rows(files), "window_counts": counts, "folds": results,
               "archive": str(archive) if archive else None,
               "final": {"train_windows": len(train_all), "seed": final_seed, "epochs": epochs,
                         "batch_size": batch_size, "model_bytes": len(tflite)}}
    path = report.write_v1_report(run_dir, summary)
    return {"report": str(path), "folds": results, "meta": meta, "window_counts": counts,
            "archive": summary["archive"]}


def _print_folds(out: dict) -> None:
    print("Files:")
    for f in out["files"]:
        c = f["clipping"]
        print(f"  {f['path']}: {f['label']}, {f['wearer']}, use={f['use']}, training windows "
              f"{f['train_windows']}, live windows {f['live_windows']}, EOL rows {f['eol_rows']}, "
              f"clipped rows {c['rows_any']}/{c['rows']}")
    wc = out["window_counts"]
    print("Windows per class (train files): " + ", ".join(
        f"{c} {wc['train'][c]} training / {wc['live'][c]} live" for c in CLASSES))
    print("Folds:")
    for f in out["folds"]:
        status = f["skipped"] or "scored"
        print(f"  {f['name']}: {status}; train {f['train_windows']}; test {f['test_windows']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", default=str(MANIFEST_PATH))
    ap.add_argument("--folds-only", action="store_true",
                    help="load the manifest, build the folds, print window counts; train nothing")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = ap.parse_args(argv)
    out = run(args.manifest, folds_only=args.folds_only, epochs=args.epochs,
              batch_size=args.batch_size)
    if args.folds_only:
        _print_folds(out)
        return 0
    print(f"model: {MODEL_DIR / MODEL_FILES[0]} ({MODEL_VERSION})")
    print(f"metadata: {METADATA_PATH}")
    print(f"archive: {out['archive']}")
    print(f"report: {out['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
