"""Writes the plain-language report.md for one training run (git-ignored
ml/runs/<timestamp>/). Every percentage is shown next to its raw window
count, because with this little data the counts are the honest number."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..inference import CLASSES
from .dataset import counts_by_class

CAVEAT = (
    "> **Plumbing test, not an accuracy result.** One person, one recording per "
    "activity, two of the four activities (Standing, Walking), about 2.7 minutes "
    "of data in total, split by time inside each recording. Every number below "
    "means \"the pipeline runs end to end\", nothing more."
)


def frac(score: tuple[int, int], what: str) -> str:
    correct, total = score
    pct = f"{100 * correct / total:.0f}%" if total else "n/a"
    return f"{pct} ({correct}/{total} {what} windows)"


def matrix_md(m: np.ndarray) -> str:
    head = "| true \\ predicted | " + " | ".join(CLASSES) + " |"
    sep = "|---|" + "---|" * len(CLASSES)
    rows = [f"| {c} | " + " | ".join(str(int(v)) for v in m[i]) + " |"
            for i, c in enumerate(CLASSES)]
    return "\n".join([head, sep, *rows])


def write_report(run_dir: Path, ds, baselines: list[dict], cnn: dict, meta: dict,
                 model_bytes: int) -> Path:
    lines = ["# Activity model v0: training report", "", CAVEAT, ""]
    lines += [f"- Model: `{meta['model_version']}`, {model_bytes:,} bytes (float32 .tflite)",
              f"- Converter path: `{meta['converter']['path']}`",
              f"- TensorFlow {meta['tensorflow_version']}, Keras {meta['keras_version']}, "
              f"numpy {meta['numpy_version']}",
              f"- Git: `{meta['git']['commit']}`" + (" (working tree dirty)" if meta['git']['dirty'] else ""),
              f"- Hardware: {meta['hardware_note']}", ""]

    lines += ["## Windows", "",
              "5 s windows (50 samples at 10 Hz), 50% overlap for training. Split by time "
              "inside each recording: first 70% train, then a gap of at least one window, "
              "then test.", "",
              "| Capture | Label | Duration | Samples after trim | Windows | Train | Test | "
              "Dropped (split gap) | Dropped (data gap) |",
              "|---|---|---|---|---|---|---|---|---|"]
    for c in ds.captures:
        lines.append(f"| `{c['path']}` | {c['label']} | {c['duration_s']:.1f} s | "
                     f"{c['samples_after_trim']} | {c['windows']} | {c['train_windows']} | "
                     f"{c['test_windows']} | {c['windows_dropped_split_gap']} | "
                     f"{c['windows_dropped_gap']} |")
    tr, te = counts_by_class(ds.train), counts_by_class(ds.test)
    lines += ["", "| Class | Train windows | Test windows |", "|---|---|---|"]
    lines += [f"| {c} | {tr[c]} | {te[c]} |" for c in CLASSES]
    lines += [""]

    lines += ["## Baselines (fitted on train only)", "",
              "| Baseline | How | Train | Test |", "|---|---|---|---|"]
    for b in baselines:
        lines.append(f"| {b['name']} | {b['description']} | {frac(b['train'], 'train')} | "
                     f"{frac(b['test'], 'test')} |")
    lines += [""]
    for b in baselines:
        lines.append(f"- {b['name']} fitted values: `{b['fitted']}`")
    tilt = next(b for b in baselines if b["name"] == "Tilt only")
    c, n = tilt["test"]
    lines += ["", f"**Tilt-only check:** using nothing but the average wrist orientation "
              f"per window, it gets {frac(tilt['test'], 'test')} and "
              f"{frac(tilt['train'], 'train')}. "
              + ("That is high: these two recordings can largely be told apart by how the "
                 "watch was tilted, not necessarily by the activity, so the CNN's score "
                 "doesn't show it learned the activity." if n and c / n >= 0.75 else
                 "Orientation alone does not separate these recordings well."),
              "", "Tilt-only test confusion matrix:", "", matrix_md(tilt["confusion_test"]), ""]

    lines += ["## CNN", "",
              f"Conv1D(16, k=5) → Conv1D(32, k=5) → global average pooling → Dense({len(CLASSES)}, "
              f"softmax). Output is always {len(CLASSES)} classes in `CLASSES` order "
              f"({', '.join(CLASSES)}), including the ones with no data. "
              f"Seed {cnn['seed']}, {cnn['epochs']} epochs, batch {cnn['batch_size']}, "
              f"final training loss {cnn['final_loss']:.4f}.", "",
              f"- Train: {frac(cnn['train'], 'train')}",
              f"- Test: {frac(cnn['test'], 'test')}", "",
              "Train confusion matrix:", "", matrix_md(cnn["confusion_train"]), "",
              "Test confusion matrix:", "", matrix_md(cnn["confusion_test"]), ""]
    if (run_dir / "loss.png").exists():
        lines += ["![training loss](loss.png)", ""]

    path = run_dir / "report.md"
    path.write_text("\n".join(lines))
    return path
