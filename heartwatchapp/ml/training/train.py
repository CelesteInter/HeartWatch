"""Trains the v0 activity CNN, exports it, and writes a report.

Run from the repo root, in the training environment:

    heartwatchapp/ml/training/.venv/bin/python -m heartwatchapp.ml.training.train

Steps: build the windows (dataset.py, through ml/preprocess.prepare()),
score the three baselines, train a small 1D CNN on the training windows
only (fixed seeds), export .tflite + model_meta.json + parity.npz to
ml/models/, and write ml/runs/<timestamp>/report.md.
"""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path

import numpy as np

from .. import spec
from ..inference import CLASSES
from . import baselines as bl
from . import export, report
from .dataset import build_dataset

RUNS_DIR = Path(__file__).resolve().parents[1] / "runs"
PARITY_MAX_WINDOWS = 16


def build_model(seed: int):
    import keras

    keras.utils.set_random_seed(seed)
    return keras.Sequential([
        keras.Input(shape=(spec.WINDOW_SAMPLES, len(spec.CHANNELS))),
        keras.layers.Conv1D(16, 5, padding="same", activation="relu"),
        keras.layers.Conv1D(32, 5, padding="same", activation="relu"),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(len(CLASSES), activation="softmax"),
    ])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    import keras
    import tensorflow as tf

    tf.config.experimental.enable_op_determinism()
    keras.utils.set_random_seed(args.seed)

    ds = build_dataset()
    if len(ds.train) == 0:
        raise SystemExit("no training windows")
    baselines = bl.run_baselines(ds.train, ds.test)

    model = build_model(args.seed)
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(ds.train.X, ds.train.y, epochs=args.epochs,
                        batch_size=args.batch_size, shuffle=True, verbose=0)

    def predict(X):
        return model.predict(X, verbose=0) if len(X) else np.empty((0, len(CLASSES)), np.float32)

    p_train, p_test = predict(ds.train.X), predict(ds.test.X)
    y_tr, y_te = p_train.argmax(axis=1), p_test.argmax(axis=1)
    cnn = {
        "seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size,
        "final_loss": float(history.history["loss"][-1]),
        "train": bl.score(ds.train.y, y_tr), "test": bl.score(ds.test.y, y_te),
        "confusion_train": bl.confusion(ds.train.y, y_tr),
        "confusion_test": bl.confusion(ds.test.y, y_te),
    }

    tflite, converter = export.convert(model)
    parity_X = ds.test.X[:PARITY_MAX_WINDOWS] if len(ds.test) else ds.train.X[:PARITY_MAX_WINDOWS]
    meta = export.build_metadata(ds, converter, {
        "training": {"seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size,
                     "final_loss": cnn["final_loss"]},
        "scores": {"cnn_train": list(cnn["train"]), "cnn_test": list(cnn["test"]),
                   **{f"{b['name']}": {"train": list(b["train"]), "test": list(b["test"])}
                      for b in baselines}},
        "parity_windows": int(len(parity_X)),
    })
    export.write_artifacts(tflite, meta, parity_X, predict(parity_X))

    run_dir = RUNS_DIR / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    _plot_loss(history.history["loss"], run_dir / "loss.png")
    path = report.write_report(run_dir, ds, baselines, cnn, meta, len(tflite))

    print(f"converter: {converter['path']}")
    print(f"model: {export.MODEL_PATH} ({len(tflite):,} bytes)")
    print(f"CNN train {report.frac(cnn['train'], 'train')}, test {report.frac(cnn['test'], 'test')}")
    print(f"report: {path}")
    return 0


def _plot_loss(loss: list[float], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(range(1, len(loss) + 1), loss)
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
