"""Exports the v2 Still/Walking activity model (`magnitude` input), if it passes.

Run from the repo root, in the training environment:

    heartwatchapp/ml/training/.venv/bin/python -B -m heartwatchapp.ml.training.train_v2_magnitude --folds-only
    heartwatchapp/ml/training/.venv/bin/python -B -m heartwatchapp.ml.training.train_v2_magnitude

Plain-language overview, in order (handoff 13):
1. Read ml/captures.csv; the `train` files get two labels, Still
   (Standing + Sitting) and Walking (input_variants.LABEL_MAP).
2. Score the `magnitude` input exactly as the input experiment did
   (experiment_inputs.py): hold out one whole file, then one whole wearer;
   for every fold train the network three times (seeds 0, 1, 2) with v1's
   settings (40 epochs, batch 16, Adam 1e-3, no class weights) and compare
   it with the motion rule and tilt only, fitted on the same training files.
3. Apply the fixed pass rule (experiment_inputs.SUCCESS_RULE, exact
   arithmetic): on both held-out files and held-out wearers, mean >= motion
   rule - 5 points, worst seed >= motion rule - 10, mean >= tilt only + 20.
4. Only if it passes: copy the current model (v1) to
   ml/models/archive/<its version>/ (never overwritten), train one final
   model on every `train` file's windows with seed 0, and export
   .tflite + model_meta.json + parity.npz to ml/models/.
   If it fails, ml/models/ is not touched.
5. Always write ml/runs/<timestamp>-v2/report.md and scores.json.

The input transform is ml/transforms.py's, applied through
preprocess.model_input() -- the same call ActivityClassifier makes in the
app, so training and prediction can't disagree. train_v1.py is left as it
is, so v1 stays reproducible.

--folds-only prints the folds and window counts. It trains nothing,
imports no TensorFlow and writes nothing. Importing this module imports no
TensorFlow either.
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
from typing import Callable

import numpy as np

from .. import spec, transforms
from ..inference import MODEL_DIR, check_metadata, expected_metadata
from ..manifest import MANIFEST_PATH, REPO_ROOT, load_manifest, manifest_sha256
from ..preprocess import model_input
from . import experiment_inputs as ex
from . import export, folds
from .dataset import _concat
from .input_variants import EXPERIMENT_CLASSES, LABEL_MAP, VARIANTS, Variant, relabel
from .train import RUNS_DIR
from .train_v1 import (BATCH_SIZE, EPOCHS, MODEL_FILES, PARITY_MAX_WINDOWS, _KerasPredictor,
                       archive_existing, keras_convert, keras_fit, keras_versions)

MODEL_VERSION = "har-cnn-v2-magnitude"
INPUT_TRANSFORM = "magnitude"
SEEDS = ex.SEEDS
FINAL_SEED = SEEDS[0]
SUCCESS_RULE = ex.SUCCESS_RULE
SCOPES = (("files", "file"), ("wearers", "wearer"))
NOTES = [
    "Still vs Walking only: Sitting and Standing are both Still.",
    "Running is not a model class (its only capture clips at +-2 g).",
    "Reads only the old ESP-IDF log format (from_legacy_capture); sensor at the "
    "+-2 g / +-250 deg/s defaults for these captures.",
    "Ties a simple motion-threshold rule; not shown to be better than it yet.",
    "Model files are git-ignored: teammates run this script to get the model.",
]


def to_model_input(X: np.ndarray) -> np.ndarray:
    """prepare()'s windows -> this model's input, via the shared function."""
    return model_input(X, INPUT_TRANSFORM)


VARIANT = Variant(INPUT_TRANSFORM, transforms.output_channels(INPUT_TRANSFORM),
                  VARIANTS[INPUT_TRANSFORM].description, to_model_input)


# ------------------------------------------------------- Keras (TF only here)

def keras_fit_final(X, y, seed):
    return _KerasPredictor(keras_fit(X, y, seed, EPOCHS, BATCH_SIZE, channels=X.shape[2],
                                     n_outputs=len(EXPERIMENT_CLASSES)))


# --------------------------------------------------------------- the rule

def score_rule(results: list[dict], seeds=SEEDS) -> dict:
    """Totals per scope and the fixed pass rule's checks. A scope with no
    scored fold can't pass."""
    out = {"text": SUCCESS_RULE["text"], "totals": {}, "checks": {}, "passed": True}
    for scope, kind in SCOPES:
        tot = ex.totals(results, kind, seeds)
        out["totals"][scope] = tot
        if not tot["folds"]:
            out["checks"][scope] = None
            out["passed"] = False
            continue
        checks = ex.success_checks(tot["variants"][INPUT_TRANSFORM]["per_seed"],
                                   tot["rules"]["motion_rule"], tot["rules"]["tilt_only"])
        out["checks"][scope] = checks
        out["passed"] = out["passed"] and all(c["pass"] for c in checks.values())
    return out


# ---------------------------------------------------------------- helpers

def _new_run_dir(runs_dir: Path) -> Path:
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir, k = runs_dir / f"{stamp}-v2", 2
    while run_dir.exists():
        run_dir, k = runs_dir / f"{stamp}-{k}-v2", k + 1
    run_dir.mkdir(parents=True)
    return run_dir


def _fold_brief(results: list[dict]) -> list[dict]:
    out = []
    for r in results:
        b = {k: r[k] for k in ("name", "kind", "test_paths", "status", "test_windows", "test_counts")
             if k in r}
        if r.get("status") == "scored":
            v = r["variants"][INPUT_TRANSFORM]
            b["cnn"] = {"seeds": list(SEEDS), "per_seed": v["per_seed"], "mean_correct": v["mean_correct"],
                        "worst": v["worst"]}
            b["rules"] = {k: r["rules"][k]["score"] for k in ex.RULE_NAMES}
        out.append(b)
    return out


def _captures(files) -> list[dict]:
    return [{"path": f.path, "label": f.info["original_label"], "model_class": f.entry.label,
             "wearer": f.entry.wearer, "sha256": f.info["sha256"], "eol_rows": f.info["eol_rows"]}
            for f in files]


def write_report(run_dir: Path, s: dict) -> Path:
    rule, names = s["pass_rule"], ex.RULE_NAMES
    if rule["passed"]:
        verdict = (f"**The `magnitude` model passed the fixed rule and was exported** as "
                   f"`{MODEL_VERSION}` to `{s['exported']['model']}`. The previous model was archived "
                   f"to `{s['archive']}`.")
    else:
        verdict = ("**The `magnitude` model did not pass the fixed rule (FAIL). Nothing was exported**; "
                   "`ml/models/` was not touched.")
    L = ["# v2 export: Still vs Walking from movement size (`magnitude` input)", "",
         verdict, "",
         "> Two labels: **Still** (Standing + Sitting) and **Walking**. The network sees only how hard "
         "the wrist moves (|accel| minus its window average) and turns (|gyro|), never its angle. "
         "Scores come from files and wearers the network never saw, three seeds each, with the "
         "input experiment's settings.", "",
         "## The pass rule (fixed in advance)", "", rule["text"], "",
         "| Scope | Mean >= motion - 5 | Worst >= motion - 10 | Mean >= tilt + 20 |", "|---|---|---|---|"]
    for scope, _ in SCOPES:
        c = rule["checks"][scope]
        if c is None:
            L.append(f"| {scope} | FAIL (no scored fold) | | |")
            continue
        L.append(f"| {scope} | " + " | ".join(
            f"{'pass' if x['pass'] else 'FAIL'} ({x['value_pct']:.1f}% vs {x['needed_pct']:.1f}% needed)"
            for x in c.values()) + " |")
    L.append("")
    for scope, kind in SCOPES:
        L += [f"## Held-out {scope}", "",
              "| Held out | Test windows | Majority | Motion rule | Tilt only | magnitude mean [worst] |",
              "|---|---|---|---|---|---|"]
        for r in (r for r in s["folds"] if r["kind"] == kind):
            held = ", ".join(ex._short(p) for p in r["test_paths"])
            if r["status"] != "scored":
                L.append(f"| {held} | - | **{r['status']}** | | | |")
                continue
            n, v = r["test_windows"], r["variants"][INPUT_TRANSFORM]
            counts = ", ".join(f"{c} {k}" for k, c in r["test_counts"].items())
            L.append(f"| {held} ({counts}) | {n} | "
                     + " | ".join(ex._cell(r["rules"][k]["score"][0], n) for k in names)
                     + f" | {ex._cell(v['mean_correct'], n)} [{v['worst'][0]}/{n}] |")
        t = rule["totals"][scope]
        if t["windows"]:
            n, v = t["windows"], t["variants"][INPUT_TRANSFORM]
            L.append(f"| **Total** | {n} | " + " | ".join(ex._cell(t["rules"][k][0], n) for k in names)
                     + f" | {ex._cell(v['mean_correct'], n)} [{v['worst'][0]}/{n}] |")
            L.append(f"\nPer seed: {', '.join(f'{c}/{m}' for c, m in v['per_seed'])}")
        L.append("")
    if s.get("final"):
        f = s["final"]
        L += ["## Exported model", "",
              f"- `{MODEL_VERSION}`, input transform `{INPUT_TRANSFORM}`, input shape {f['input_shape']}, "
              f"classes {', '.join(EXPERIMENT_CLASSES)}.",
              f"- Trained on {f['train_windows']} training-hop windows ({f['train_counts']}) from every "
              f"`train` file, seed {f['seed']}, {f['epochs']} epochs, batch {f['batch_size']}.",
              f"- {f['model_bytes']:,} bytes; parity fixture: {f['parity_windows']} windows.", ""]
    L += ["## Known limits", "", *[f"- {n}" for n in NOTES], "",
          "## Provenance", "",
          f"- Git `{s['git']['commit']}`" + (" (**working tree dirty**)" if s["git"]["dirty"] else " (clean tree)")
          + f"; manifest SHA-256 `{s['manifest']['sha256']}`.", ""]
    path = run_dir / "report.md"
    path.write_text("\n".join(L))
    return path


# ---------------------------------------------------------------- the run

def run(manifest_path: str | Path = MANIFEST_PATH, *, repo_root: str | Path = REPO_ROOT,
        models_dir: str | Path = MODEL_DIR, runs_dir: str | Path = RUNS_DIR,
        folds_only: bool = False, fit_predict: Callable | None = None,
        fit_final: Callable | None = None, convert: Callable | None = None,
        versions: dict | None = None, seeds=SEEDS) -> dict:
    """The whole v2 run; see the module docstring. The fit/convert hooks
    default to Keras; tests pass stubs so nothing is trained."""
    manifest_path, models_dir, runs_dir = Path(manifest_path), Path(models_dir), Path(runs_dir)
    all_files = folds.load_file_windows(load_manifest(manifest_path, repo_root), repo_root)
    files = relabel(all_files)
    fl = ex.build_folds(files)
    if folds_only:
        return {"folds": ex.fold_summary(fl, files), "files": ex._file_rows(files)}

    real = fit_predict is None
    if real:
        fit_predict, fit_final, convert = ex.keras_fit_predict, keras_fit_final, keras_convert
    results = ex.evaluate(files, fit_predict, {INPUT_TRANSFORM: VARIANT}, seeds)
    rule = score_rule(results, seeds)
    git = export.git_commit()

    s = {"model_version": MODEL_VERSION, "input_transform": INPUT_TRANSFORM, "git": git,
         "manifest": {"path": str(manifest_path), "sha256": manifest_sha256(manifest_path)},
         "classes": list(EXPERIMENT_CLASSES), "label_map": dict(LABEL_MAP), "seeds": list(seeds),
         "training": {"epochs": EPOCHS, "batch_size": BATCH_SIZE, "optimizer": "Adam(1e-3)",
                      "class_weights": None, "architecture": "train.build_model",
                      "outputs": len(EXPERIMENT_CLASSES)},
         "files": ex._file_rows(files), "folds": results, "pass_rule": rule,
         "archive": None, "exported": None, "final": None,
         "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}

    if rule["passed"]:
        archive = archive_existing(models_dir)   # before anything is written to models_dir
        train_all = _concat([f.train for f in files])
        live_all = _concat([f.live for f in files])
        model = fit_final(to_model_input(train_all.X), train_all.y, FINAL_SEED)
        idx = np.unique(np.linspace(0, len(live_all) - 1, min(PARITY_MAX_WINDOWS, len(live_all))).astype(int))
        parity_X = live_all.X[idx]                                # prepare() output, as the app passes it
        parity_probs = np.asarray(model(to_model_input(parity_X)))
        tflite, converter = convert(model)
        channels = VARIANT.channels
        meta = {
            "model_version": MODEL_VERSION,
            **expected_metadata(),
            "classes": list(EXPERIMENT_CLASSES),
            "input_transform": INPUT_TRANSFORM,
            "input_channels": channels,
            "label_map": dict(LABEL_MAP),
            "bin_ms": spec.BIN_MS, "trim_s": spec.TRIM_S,
            "train_hop_samples": spec.TRAIN_HOP_SAMPLES, "live_hop_samples": spec.LIVE_HOP_SAMPLES,
            "input_shape": [1, spec.WINDOW_SAMPLES, channels],
            "manifest": s["manifest"],
            "captures": _captures(files),
            "excluded": [f.path for f in all_files if f.entry.use != "train"],
            "train_windows": {c: int((train_all.y == i).sum()) for i, c in enumerate(EXPERIMENT_CLASSES)},
            "live_windows": {c: int((live_all.y == i).sum()) for i, c in enumerate(EXPERIMENT_CLASSES)},
            "folds": _fold_brief(results),
            "pass_rule": {"text": rule["text"], "checks": rule["checks"], "passed": rule["passed"]},
            "seeds": {"folds": list(seeds), "final": FINAL_SEED},
            "training": s["training"],
            "notes": NOTES,
            "git": git,
            "converter": converter,
            "parity_windows": int(len(parity_X)),
            "created_at": s["created_at"],
            **(keras_versions() if real else (versions or {})),
        }
        check_metadata(meta)   # what the app will check; fail here, not in the app
        paths = {"model": models_dir / MODEL_FILES[0], "metadata": models_dir / MODEL_FILES[1],
                 "parity": models_dir / MODEL_FILES[2]}
        export.write_artifacts(tflite, meta, parity_X, parity_probs, model_path=paths["model"],
                               metadata_path=paths["metadata"], parity_path=paths["parity"],
                               classes=EXPERIMENT_CLASSES)
        s["archive"] = str(archive) if archive else None
        s["exported"] = {k: str(p) for k, p in paths.items()}
        s["final"] = {"train_windows": len(train_all), "train_counts": meta["train_windows"],
                      "seed": FINAL_SEED, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
                      "input_shape": meta["input_shape"], "model_bytes": len(tflite),
                      "parity_windows": meta["parity_windows"]}

    run_dir = _new_run_dir(runs_dir)
    (run_dir / "scores.json").write_text(json.dumps(s, indent=1) + "\n")
    report = write_report(run_dir, s)
    return {"run_dir": str(run_dir), "report": str(report), "passed": rule["passed"],
            "pass_rule": rule, "exported": s["exported"], "archive": s["archive"], "git": git}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", default=str(MANIFEST_PATH))
    ap.add_argument("--folds-only", action="store_true",
                    help="print the folds and Still/Walking window counts; train nothing, write nothing")
    args = ap.parse_args(argv)
    out = run(args.manifest, folds_only=args.folds_only)
    if args.folds_only:
        print("Files (model class <- manifest label):")
        for f in out["files"]:
            print(f"  {f['path']}: {f['label']} <- {f['original_label']}, {f['wearer']}, "
                  f"training windows {f['train_windows']}, live windows {f['live_windows']}")
        print("Folds:")
        for f in out["folds"]:
            print(f"  {f['name']}: {f['skipped'] or 'scored'}; train {f['train_windows']}; "
                  f"test {f['test_windows']}")
        return 0
    for scope, checks in out["pass_rule"]["checks"].items():
        if checks is None:
            print(f"  {scope}: FAIL (no scored fold)")
            continue
        print(f"  {scope}: " + ", ".join(f"{name} {'pass' if c['pass'] else 'FAIL'} "
                                          f"({c['value_pct']:.2f}% vs {c['needed_pct']:.2f}%)"
                                          for name, c in checks.items()))
    if out["git"]["dirty"]:
        print("WARNING: the working tree is dirty; the provenance in the report says so.")
    if out["passed"]:
        print(f"PASS: exported {MODEL_VERSION} to {out['exported']['model']}")
        print(f"archive: {out['archive']}")
    else:
        print("FAIL: nothing exported; ml/models/ untouched")
    print(f"report: {out['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
