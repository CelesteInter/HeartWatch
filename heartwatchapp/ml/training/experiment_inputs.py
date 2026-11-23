"""The input experiment (handoff 12): does the network learn motion once the
wrist angle is taken out of its input?

Run from the repo root, in the training environment:

    heartwatchapp/ml/training/.venv/bin/python -B -m heartwatchapp.ml.training.experiment_inputs --folds-only
    heartwatchapp/ml/training/.venv/bin/python -B -m heartwatchapp.ml.training.experiment_inputs

Plain-language overview: two labels only, Still (Standing + Sitting) and
Walking. For every fold (hold out one whole file; hold out one whole
wearer) and every input variant in input_variants.VARIANTS, the v1 network
is trained three times (seeds 0, 1, 2) with v1's exact settings, then
scored on the held-out windows next to three fixed rules fitted on the
same training files: majority guess, motion rule, tilt only. Everything
is fixed in advance (variants, settings, success rule); nothing is tuned.

Writes ml/runs/<timestamp>-inputs/report.md and scores.json. Never writes
to ml/models/: experiment models are not exported for the app.
--folds-only prints the folds and window counts, imports no TensorFlow
and writes nothing. Importing this module imports no TensorFlow either.
"""

from __future__ import annotations

import argparse
import datetime
import json
from fractions import Fraction
from pathlib import Path
from typing import Callable

import numpy as np

from ..manifest import MANIFEST_PATH, REPO_ROOT, load_manifest, manifest_sha256
from . import baselines as bl
from . import export, folds
from .input_variants import EXPERIMENT_CLASSES, LABEL_MAP, ROTATION_COPIES, VARIANTS, relabel
from .train import RUNS_DIR
from .train_v1 import BATCH_SIZE, EPOCHS, _KerasPredictor, keras_fit

SEEDS = folds.SEEDS
RULE_NAMES = ("majority", "motion_rule", "tilt_only")
SUCCESS_RULE = {
    "text": ("A variant counts as learning motion only if, on BOTH held-out files and held-out "
             "wearers: (1) its mean over 3 seeds >= motion rule - 5 percentage points; (2) its "
             "worst seed >= motion rule - 10 points; (3) its mean >= tilt only + 20 points. "
             "Scores are totals over the scored folds; a seed's score is its total over all folds."),
    "mean_vs_motion_pp": -5,
    "worst_vs_motion_pp": -10,
    "mean_vs_tilt_pp": 20,
}

# fit_predict(train_X, train_y, test_X, seed) -> probabilities [len(test_X), 2]
FitPredict = Callable[[np.ndarray, np.ndarray, np.ndarray, int], np.ndarray]


def keras_fit_predict(train_X, train_y, test_X, seed, epochs=EPOCHS, batch_size=BATCH_SIZE):
    """v1's network and settings; only the channel count (from the input)
    and the 2 outputs differ."""
    model = keras_fit(train_X, train_y, seed, epochs, batch_size,
                      channels=train_X.shape[2], n_outputs=len(EXPERIMENT_CLASSES))
    return _KerasPredictor(model)(test_X)


# ------------------------------------------------------------------ folds

def build_folds(files: list[folds.FileWindows]) -> list[folds.Fold]:
    """Leave one file out (all five; Still has three files, so BobbySitting
    is scored), then leave one wearer out on Still vs Walking."""
    return folds.leave_one_file_out(files) + folds.leave_one_wearer_out(files, classes=EXPERIMENT_CLASSES)


def class_counts(y: np.ndarray) -> dict[str, int]:
    counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=len(EXPERIMENT_CLASSES))
    return {c: int(n) for c, n in zip(EXPERIMENT_CLASSES, counts) if n}


def _names(pred) -> list[str]:
    return [EXPERIMENT_CLASSES[int(i)] for i in pred]


def _confusion(y_true, y_pred) -> list[list[int]]:
    m = np.zeros((len(EXPERIMENT_CLASSES),) * 2, dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        m[int(t), int(p)] += 1
    return m.tolist()


def _rules(train, test) -> dict:
    """The three fixed yardsticks, fitted on the fold's training windows
    (raw), exactly as baselines.run_baselines does it."""
    preds = {
        "majority": np.full(len(test), bl.majority_class(train.y)),
        "motion_rule": bl.NearestCentroid().fit(bl.motion_feature(train.X), train.y)
                                           .predict(bl.motion_feature(test.X)),
        "tilt_only": bl.NearestCentroid().fit(bl.tilt_feature(train.X), train.y)
                                         .predict(bl.tilt_feature(test.X)),
    }
    return {k: {"score": list(bl.score(test.y, p)), "predictions": _names(p)} for k, p in preds.items()}


def evaluate(files: list[folds.FileWindows], fit_predict: FitPredict, variants=VARIANTS,
             seeds=SEEDS) -> list[dict]:
    """Every fold x every variant x every seed, plus the three rules."""
    results = []
    for fold in build_folds(files):
        r = {"name": fold.name, "kind": fold.kind, "test_paths": list(fold.test_paths),
             "train_paths": list(fold.train_paths)}
        if fold.skipped:
            results.append({**r, "status": fold.skipped})
            continue
        train, test = folds.fold_splits(fold, files)
        r.update({"status": "scored", "test_counts": class_counts(test.y),
                  "train_counts": class_counts(train.y), "test_windows": len(test),
                  "test_detail": [{"file": s, "end_ts": int(t), "true": EXPERIMENT_CLASSES[int(y)]}
                                  for s, t, y in zip(test.source, test.end_ts, test.y)],
                  "rules": _rules(train, test), "variants": {}})
        for name, v in variants.items():
            test_X = v.transform(test.X)
            per_seed, predictions, confusions = [], [], []
            for seed in seeds:
                train_X, train_y = v.prepare_train(train.X, train.y, seed)
                pred = np.asarray(fit_predict(train_X, train_y, test_X, seed)).argmax(axis=1)
                per_seed.append(list(bl.score(test.y, pred)))
                predictions.append(_names(pred))
                confusions.append(_confusion(test.y, pred))
            correct = [c for c, _ in per_seed]
            r["variants"][name] = {"per_seed": per_seed, "mean_correct": float(np.mean(correct)),
                                   "worst": per_seed[int(np.argmin(correct))],
                                   "predictions": predictions, "confusion_per_seed": confusions}
        results.append(r)
    return results


# ---------------------------------------------------------- success rule

def success_checks(per_seed: list[tuple[int, int]], motion: tuple[int, int],
                   tilt: tuple[int, int]) -> dict:
    """The fixed success rule for one scope (files or wearers), in exact
    arithmetic (fractions), so nothing rounds towards a pass. per_seed: each
    seed's (correct, total) over the scope; motion / tilt: the rules'."""
    def pct(c, n) -> Fraction:
        return Fraction(c) * 100 / n

    n = per_seed[0][1]
    mean = pct(sum(c for c, _ in per_seed), n * len(per_seed))
    worst = pct(min(c for c, _ in per_seed), n)
    m, t = pct(*motion), pct(*tilt)
    rule = SUCCESS_RULE

    def check(value, needed):
        return {"pass": bool(value >= needed), "value_pct": float(value), "needed_pct": float(needed)}

    return {
        "mean_vs_motion": check(mean, m + rule["mean_vs_motion_pp"]),
        "worst_vs_motion": check(worst, m + rule["worst_vs_motion_pp"]),
        "mean_vs_tilt": check(mean, t + rule["mean_vs_tilt_pp"]),
    }


def learns_motion(file_checks: dict, wearer_checks: dict) -> bool:
    return all(c["pass"] for c in (*file_checks.values(), *wearer_checks.values()))


def totals(results: list[dict], kind: str, seeds=SEEDS) -> dict:
    scored = [r for r in results if r["kind"] == kind and r["status"] == "scored"]
    n = sum(r["test_windows"] for r in scored)
    out = {"folds": len(scored), "windows": n,
           "rules": {k: [sum(r["rules"][k]["score"][0] for r in scored), n] for k in RULE_NAMES},
           "variants": {}}
    names = scored[0]["variants"] if scored else {}
    for name in names:
        per_seed = [[sum(r["variants"][name]["per_seed"][i][0] for r in scored), n] for i in range(len(seeds))]
        correct = [c for c, _ in per_seed]
        out["variants"][name] = {"per_seed": per_seed, "mean_correct": float(np.mean(correct)),
                                 "worst": per_seed[int(np.argmin(correct))]}
    return out


def agreement(results: list[dict], kind: str, variant: str) -> dict:
    """How often the variant's seed-0 guesses equal the tilt-only / motion
    rule's guesses on the held-out windows."""
    scored = [r for r in results if r["kind"] == kind and r["status"] == "scored"]
    n = tilt = motion = 0
    for r in scored:
        p = r["variants"][variant]["predictions"][0]
        n += len(p)
        tilt += sum(a == b for a, b in zip(p, r["rules"]["tilt_only"]["predictions"]))
        motion += sum(a == b for a, b in zip(p, r["rules"]["motion_rule"]["predictions"]))
    return {"tilt_only": [tilt, n], "motion_rule": [motion, n]}


# ------------------------------------------------------------------ report

def _pct(c: float, n: int) -> str:
    return f"{100 * c / n:.1f}%" if n else "n/a"


def _cell(c, n) -> str:
    return f"{c:.1f}/{n} ({_pct(c, n)})" if isinstance(c, float) else f"{c}/{n} ({_pct(c, n)})"


def _short(path: str) -> str:
    return Path(path).name


def write_report(run_dir: Path, s: dict) -> Path:
    names = list(s["variants"])
    L = ["# Input experiment: can the network learn motion once wrist angle is removed?", "",
         "> Two labels only: **Still** (Standing + Sitting) and **Walking**. v1's network and "
         "training settings, unchanged; three seeds per fold. Scores always come from files or "
         "wearers the network never saw. The three rules (majority, motion rule, tilt only) are "
         "the fixed yardsticks. Nothing here is exported for the app.", "",
         "## The success rule (fixed in advance)", "", s["success_rule"]["text"], ""]
    for kind, title in (("file", "Held-out files"), ("wearer", "Held-out wearers")):
        L += [f"## {title}", "",
              "| Held out | Test windows | Majority | Motion rule | Tilt only | "
              + " | ".join(f"{n} mean [worst]" for n in names) + " |",
              "|---|---|---|---|---|" + "---|" * len(names)]
        for r in (r for r in s["folds"] if r["kind"] == kind):
            held = ", ".join(_short(p) for p in r["test_paths"])
            if r["status"] != "scored":
                L.append(f"| {held} | - | **{r['status']}** |" + " |" * (2 + len(names)))
                continue
            n = r["test_windows"]
            rules = " | ".join(_cell(r["rules"][k]["score"][0], n) for k in RULE_NAMES)
            cnn = " | ".join(f"{_cell(r['variants'][v]['mean_correct'], n)} [{r['variants'][v]['worst'][0]}/{n}]"
                             for v in names)
            counts = ", ".join(f"{c} {k}" for k, c in r["test_counts"].items())
            L.append(f"| {held} ({counts}) | {n} | {rules} | {cnn} |")
        t = s["totals"][kind + "s"]
        n = t["windows"]
        if n:
            rules = " | ".join(_cell(t["rules"][k][0], n) for k in RULE_NAMES)
            cnn = " | ".join(f"{_cell(t['variants'][v]['mean_correct'], n)} [{t['variants'][v]['worst'][0]}/{n}]"
                             for v in names)
            L.append(f"| **Total** | {n} | {rules} | {cnn} |")
        L.append("")

    L += ["## Success-rule checks", "",
          "| Variant | Scope | Mean >= motion - 5 | Worst >= motion - 10 | Mean >= tilt + 20 | "
          "Learns motion? |", "|---|---|---|---|---|---|"]
    for v in names:
        verdict = "**yes**" if s["success"][v]["learns_motion"] else "no"
        for scope in ("files", "wearers"):
            c = s["success"][v][scope]
            cells = " | ".join(f"{'pass' if x['pass'] else 'FAIL'} ({x['value_pct']:.1f}% vs "
                               f"{x['needed_pct']:.1f}% needed)" for x in c.values())
            L.append(f"| {v} | {scope} | {cells} | {verdict} |")
    L.append("")

    L += ["## Agreement with the rules (seed 0, held-out windows)", "",
          "| Variant | Files: = tilt only | Files: = motion rule | Wearers: = tilt only | "
          "Wearers: = motion rule |", "|---|---|---|---|---|"]
    for v in names:
        a, b = s["agreement"]["files"][v], s["agreement"]["wearers"][v]
        L.append(f"| {v} | {_cell(*a['tilt_only'])} | {_cell(*a['motion_rule'])} | "
                 f"{_cell(*b['tilt_only'])} | {_cell(*b['motion_rule'])} |")
    L.append("")

    L += ["## Seed-0 confusion tables (rows = true class, counts predicted Still / Walking)", ""]
    for r in (r for r in s["folds"] if r["status"] == "scored"):
        L.append(f"- **{', '.join(_short(p) for p in r['test_paths'])}**")
        for v in names:
            m = r["variants"][v]["confusion_per_seed"][0]
            rows = [f"{EXPERIMENT_CLASSES[i]} -> Still {m[i][0]}, Walking {m[i][1]}"
                    for i in range(len(EXPERIMENT_CLASSES)) if sum(m[i])]
            L.append(f"  - {v}: " + "; ".join(rows))
    L += ["", "## Notes", "",
          "- Sitting vs Standing is gone by design (both are Still). Running is excluded and untested.",
          f"- Git `{s['git']['commit']}`" + (" (**working tree dirty**)" if s["git"]["dirty"] else " (clean tree)")
          + f"; manifest SHA-256 `{s['manifest']['sha256']}`.", ""]
    path = run_dir / "report.md"
    path.write_text("\n".join(L))
    return path


# ------------------------------------------------------------------ run

def _new_run_dir(runs_dir: Path) -> Path:
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir, k = runs_dir / f"{stamp}-inputs", 2
    while run_dir.exists():
        run_dir, k = runs_dir / f"{stamp}-{k}-inputs", k + 1
    run_dir.mkdir(parents=True)
    return run_dir


def _file_rows(files) -> list[dict]:
    return [{"path": f.path, "label": f.entry.label, "original_label": f.info["original_label"],
             "wearer": f.entry.wearer, "train_windows": len(f.train), "live_windows": len(f.live)}
            for f in files]


def fold_summary(fl, files) -> list[dict]:
    out = []
    for fold in fl:
        train, test = folds.fold_splits(fold, files)
        out.append({"name": fold.name, "kind": fold.kind, "test_paths": list(fold.test_paths),
                    "skipped": fold.skipped, "train_windows": class_counts(train.y),
                    "test_windows": class_counts(test.y)})
    return out


def run(manifest_path: str | Path = MANIFEST_PATH, *, repo_root: str | Path = REPO_ROOT,
        runs_dir: str | Path = RUNS_DIR, folds_only: bool = False,
        fit_predict: FitPredict | None = None, seeds=SEEDS) -> dict:
    manifest_path, runs_dir = Path(manifest_path), Path(runs_dir)
    files = relabel(folds.load_file_windows(load_manifest(manifest_path, repo_root), repo_root))
    fl = build_folds(files)
    if folds_only:
        return {"folds": fold_summary(fl, files), "files": _file_rows(files)}

    real = fit_predict is None
    if real:
        fit_predict = keras_fit_predict
    results = evaluate(files, fit_predict, VARIANTS, seeds)
    tot = {"files": totals(results, "file", seeds), "wearers": totals(results, "wearer", seeds)}
    success = {}
    for v in VARIANTS:
        checks = {scope: success_checks(tot[scope]["variants"][v]["per_seed"],
                                        tot[scope]["rules"]["motion_rule"], tot[scope]["rules"]["tilt_only"])
                  for scope in ("files", "wearers")}
        success[v] = {**checks, "learns_motion": learns_motion(checks["files"], checks["wearers"])}
    s = {
        "git": export.git_commit(),
        "manifest": {"path": str(manifest_path), "sha256": manifest_sha256(manifest_path)},
        "classes": list(EXPERIMENT_CLASSES),
        "label_map": LABEL_MAP,
        "variants": {k: {"channels": v.channels, "description": v.description} for k, v in VARIANTS.items()},
        "seeds": list(seeds),
        "success_rule": SUCCESS_RULE,
        "training": {"epochs": EPOCHS, "batch_size": BATCH_SIZE, "optimizer": "Adam(1e-3)",
                     "class_weights": None, "architecture": "train.build_model",
                     "outputs": len(EXPERIMENT_CLASSES), "rotation_copies": ROTATION_COPIES,
                     "rotation_seed": "the run's seed"},
        "files": _file_rows(files),
        "folds": results,
        "totals": tot,
        "success": success,
        "agreement": {scope: {v: agreement(results, kind, v) for v in VARIANTS}
                      for scope, kind in (("files", "file"), ("wearers", "wearer"))},
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }
    if real:
        import sys

        import keras
        import tensorflow as tf

        s["versions"] = {"tensorflow": tf.__version__, "keras": keras.__version__,
                         "numpy": np.__version__, "python": sys.version.split()[0]}
    run_dir = _new_run_dir(runs_dir)
    (run_dir / "scores.json").write_text(json.dumps(s, indent=1) + "\n")
    report = write_report(run_dir, s)
    return {"run_dir": str(run_dir), "report": str(report), "totals": tot, "success": success}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", default=str(MANIFEST_PATH))
    ap.add_argument("--folds-only", action="store_true",
                    help="print the folds and Still/Walking window counts; train nothing, write nothing")
    args = ap.parse_args(argv)
    out = run(args.manifest, folds_only=args.folds_only)
    if args.folds_only:
        print("Files (experiment label <- manifest label):")
        for f in out["files"]:
            print(f"  {f['path']}: {f['label']} <- {f['original_label']}, {f['wearer']}, "
                  f"training windows {f['train_windows']}, live windows {f['live_windows']}")
        print("Folds:")
        for f in out["folds"]:
            print(f"  {f['name']}: {f['skipped'] or 'scored'}; train {f['train_windows']}; "
                  f"test {f['test_windows']}")
        return 0
    print(f"report: {out['report']}")
    for v, r in out["success"].items():
        print(f"  {v}: learns motion = {r['learns_motion']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
