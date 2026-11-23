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


# ---------------------------------------------------------------- v1 report

V1_CAVEAT = (
    "> **Read this first.** Two wearers (Dylan, Bobby), one or two short recordings per "
    "activity, wrists and watch orientation unknown, sensor at its +-2 g / +-250 deg/s "
    "defaults, accelerometer Z offset (about +0.24 g) not corrected. **Running has no "
    "training data** (Running.txt clips in 24.3% of rows and is excluded), so this model "
    "can't predict Running. **Sitting is decided by tilt only, one session, one wearer**: "
    "it can't be tested, because there's no second Sitting file to hold out. Every score "
    "below comes from files (or wearers) the model never saw during that fold's training."
)


def _pct(c: float, n: int) -> str:
    return f"{100 * c / n:.1f}%" if n else "n/a"


def _cnn_cells(r: dict) -> tuple[str, str]:
    n = r["test_windows"]
    mean, (worst, _) = r["cnn"]["mean_correct"], r["cnn"]["worst"]
    return f"{mean:.1f}/{n} ({_pct(mean, n)})", f"{worst}/{n} ({_pct(worst, n)})"


def _base_cell(r: dict, name: str) -> str:
    b = next(b for b in r["baselines"] if b["name"] == name)
    c, n = b["test"]
    return f"{c}/{n} ({_pct(c, n)})"


def _fold_table(results: list[dict], kind: str) -> list[str]:
    rows = [r for r in results if r["kind"] == kind]
    lines = ["| Held out | True class | Test windows | CNN mean of seeds | CNN worst seed | "
             "Majority | Motion rule | Tilt only |", "|---|---|---|---|---|---|---|---|"]
    totals = {"n": 0, "cnn": 0.0, "worst": 0, "Majority class": 0, "Motion rule": 0, "Tilt only": 0}
    for r in rows:
        held = ", ".join(f"`{p}`" for p in r["test_paths"])
        if r["status"] != "scored":
            lines.append(f"| {held} | {r.get('true_class', '-')} | - | **{r['status']}** | | | | |")
            continue
        cnn_mean, cnn_worst = _cnn_cells(r)
        lines.append(f"| {held} | {r['true_class']} | {r['test_windows']} | {cnn_mean} | {cnn_worst} | "
                     f"{_base_cell(r, 'Majority class')} | {_base_cell(r, 'Motion rule')} | "
                     f"{_base_cell(r, 'Tilt only')} |")
        totals["n"] += r["test_windows"]
        totals["cnn"] += r["cnn"]["mean_correct"]
        totals["worst"] += r["cnn"]["worst"][0]
        for b in r["baselines"]:
            totals[b["name"]] += b["test"][0]
    n = totals["n"]
    if n:
        lines.append(f"| **All scored folds** | | {n} | {totals['cnn']:.1f}/{n} ({_pct(totals['cnn'], n)}) | "
                     f"{totals['worst']}/{n} ({_pct(totals['worst'], n)}) | "
                     f"{totals['Majority class']}/{n} ({_pct(totals['Majority class'], n)}) | "
                     f"{totals['Motion rule']}/{n} ({_pct(totals['Motion rule'], n)}) | "
                     f"{totals['Tilt only']}/{n} ({_pct(totals['Tilt only'], n)}) |")
    return lines


def write_v1_report(run_dir: Path, summary: dict) -> Path:
    """summary: what train_v1.run() gathered (files, window counts, fold
    results, final model, metadata). Plain language first."""
    meta = summary["meta"]
    lines = [f"# Activity model {meta['model_version']}: training report", "", V1_CAVEAT, ""]
    lines += ["## In plain words", "",
              "Each test below hides one whole recording (or one whole person) from the model, "
              "trains on the rest, and checks the hidden one. The CNN is trained three times "
              "per test with different random starts; we show the average and the worst. The "
              "three simple rules next to it are the bar to clear: *majority* always guesses the "
              "most common training activity, *motion rule* looks only at how much the wrist "
              "moves, and *tilt only* looks only at which way the watch points. If the CNN can't "
              "beat *tilt only* on hidden files, it learned the wrist angle, not the activity.", ""]

    lines += ["## Data", "",
              "| File | Label | Wearer | Use | Training windows | Live windows | EOL rows | "
              "Rows with clipping |", "|---|---|---|---|---|---|---|---|"]
    for f in summary["files"]:
        c = f["clipping"]
        lines.append(f"| `{f['path']}` | {f['label']} | {f['wearer']} | {f['use']} | "
                     f"{f['train_windows']} | {f['live_windows']} | {f['eol_rows']} | "
                     f"{c['rows_any']}/{c['rows']} ({_pct(c['rows_any'], c['rows'])})"
                     f"{' **warning**' if c['warn'] else ''} |")
    wc = summary["window_counts"]
    lines += ["", "Windows per class (`train` files only; training hop = 50% overlap, live hop = "
              "no overlap):", "", "| Class | Training hop | Live hop |", "|---|---|---|"]
    lines += [f"| {c} | {wc['train'][c]} | {wc['live'][c]} |" for c in CLASSES]
    lines += [""]

    lines += ["## Leave one file out", ""] + _fold_table(summary["folds"], "file") + [""]
    lines += ["## Leave one wearer out (Standing vs Walking only)", ""]
    lines += _fold_table(summary["folds"], "wearer") + [""]

    lines += ["## Sitting", "",
              "Sitting is in the training set (BobbySitting.txt) but **can't be tested**: it's the "
              "only Sitting file, so no fold can hide it and still have Sitting to learn from. In "
              "the recon, Sitting vs Standing separated only by wrist angle (tilt only 134/134, "
              "best motion threshold 103/134). Treat any Sitting prediction as decided by tilt "
              "only, one session, one wearer.", ""]

    fm = summary["final"]
    lines += ["## Final model", "",
              f"- Trained on every `train` file: {fm['train_windows']} training windows, seed "
              f"{fm['seed']}, {fm['epochs']} epochs, batch {fm['batch_size']}.",
              f"- `{meta['model_version']}`, {fm['model_bytes']:,} bytes, converter "
              f"`{meta['converter']['path']}`.",
              f"- Manifest SHA-256 `{meta['manifest']['sha256']}`.",
              f"- Git `{meta['git']['commit']}`" + (" (**working tree dirty**)" if meta["git"]["dirty"]
                                                    else " (clean tree)"),
              f"- Previous model archived to: {summary['archive'] or 'nothing to archive'}", ""]
    path = run_dir / "report.md"
    path.write_text("\n".join(lines))
    return path
