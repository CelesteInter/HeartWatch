"""Runs the exported activity model over a capture and prints what it says.

App environment only (LiteRT, no TensorFlow). From the repo root:

    heartwatchapp/.venv/bin/python -B -m heartwatchapp.ml.tools.predict_capture \\
        HeartWatch/test/BobbyStand.txt --accel-range-g 2 --gyro-range-dps 250
    heartwatchapp/.venv/bin/python -B -m heartwatchapp.ml.tools.predict_capture --manifest
    heartwatchapp/.venv/bin/python -B -m heartwatchapp.ml.tools.predict_capture heartwatchapp/ml/captures.csv

Plain-language overview: it uses whatever model is in ml/models/ and
first prints which one (its model_version, input transform and classes).
Each capture is cut into non-overlapping 5-second windows (the way the app
will see it) and the model guesses a class for each. For every file it
prints:
- each window's time (from the start of the log), class and confidence,
- the windows merged into time segments ("Still 0:05–1:20"): neighbouring
  windows with the same class, with no time gap between them,
- how many windows got each class, the average confidence, the end-of-log
  rows and the clipping line (WARNING when too many readings hit the
  sensor's limits).
For a single capture file you must say the sensor ranges it was recorded
at (nothing is guessed). With --manifest (or a .csv), every manifest row
is run with its own ranges and label, and a summary line per file shows
the manifest label mapped through the model's label_map (✓ when most
windows agree, ✗ when not; a label the model has no class for, like
Running for v2, is "not a model class").
An unreadable, empty or too-short log gives a one-line error, not a
traceback. Writes nothing.
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

from .. import spec
from ..inference import CLASSES, METADATA_PATH, MODEL_PATH, ActivityClassifier, ModelMetadataError
from ..manifest import MANIFEST_PATH, REPO_ROOT, ManifestError, load_manifest, load_recording
from ..preprocess import prepare
from ..recording import from_legacy_capture


class CaptureError(Exception):
    """A capture that can't be classified, with a one-line reason."""


def _pct(c: int, n: int) -> str:
    return f"{100 * c / n:.0f}%" if n else "n/a"


def fmt_time(ms: int) -> str:
    """Milliseconds -> m:ss (h:mm:ss from an hour), whole seconds rounded down."""
    s = int(ms) // 1000
    h, rest = divmod(s, 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def merge_segments(windows) -> list[tuple[int, int, str, int]]:
    """windows: (start_ms, end_ms, label) in time order. Neighbouring
    windows with the same label merge when one starts where the previous
    ended; a time gap (a dropped window, a pause) starts a new segment.
    Returns (start_ms, end_ms, label, n_windows) per segment."""
    out: list[list] = []
    for start, end, label in windows:
        if out and out[-1][2] == label and out[-1][1] == start:
            out[-1][1] = end
            out[-1][3] += 1
        else:
            out.append([start, end, label, 1])
    return [tuple(s) for s in out]


def _model_info(classifier) -> tuple[list[str], dict, str]:
    """classes, label_map, transform -- with the v1 defaults for a
    classifier that doesn't say (test stubs)."""
    classes = list(getattr(classifier, "classes", CLASSES))
    label_map = getattr(classifier, "label_map", None) or {c: c for c in classes}
    return classes, dict(label_map), getattr(classifier, "input_transform", "raw")


def _read_capture(load):
    """Run a loader; turn the ways a log can be unreadable into CaptureError."""
    try:
        return load()
    except FileNotFoundError:
        raise CaptureError("file not found") from None
    except IsADirectoryError:
        raise CaptureError("is a directory, not a log file") from None
    except UnicodeDecodeError:
        raise CaptureError("not a text log (not UTF-8)") from None
    except PermissionError:
        raise CaptureError("no permission to read it") from None
    except OSError as e:
        raise CaptureError(f"can't read it ({e.strerror or e})") from None
    except ValueError as e:
        raise CaptureError(f"can't read it as a capture log ({e})") from None


def classify(rec, classifier) -> dict:
    """Windows, per-window predictions and segments for one recording, or
    CaptureError when there is nothing to classify."""
    if len(rec) == 0:
        raise CaptureError("no motion readings in the log (empty, or not an ESP-IDF capture log)")
    try:
        w = prepare(rec, spec.LIVE_HOP_SAMPLES)
    except ValueError as e:
        raise CaptureError(f"can't prepare windows ({e})") from None
    if len(w.X) == 0:
        why = (f"too short for one {spec.WINDOW_MS // 1000} s window after trimming {spec.TRIM_S} s "
               f"at each end ({w.n_samples} samples of {spec.WINDOW_SAMPLES} needed)")
        if w.dropped:
            why = f"no complete {spec.WINDOW_MS // 1000} s window ({w.dropped} touched a gap)"
        raise CaptureError(why)
    preds = classifier.predict(w.X)
    t0 = int(rec.ts_ms[0])
    rows = [(int(end) - spec.WINDOW_MS - t0, int(end) - t0, label, conf)
            for end, (label, conf) in zip(w.end_ts, preds)]
    return {"windows": w, "rows": rows, "segments": merge_segments([r[:3] for r in rows]),
            "counts": collections.Counter(label for _, _, label, _ in rows)}


def report_one(name: str, rec, classifier, out) -> dict:
    """Print one capture's windows, segments and counts. Returns the
    classify() result; raises CaptureError."""
    classes, label_map, _ = _model_info(classifier)
    target = label_map.get(rec.label) if rec.label else None
    shown = rec.label or "none"
    if rec.label:
        shown += f" → {target}" if target else " (not a model class)"
    print(f"== {name}  (label: {shown})", file=out)
    r = classify(rec, classifier)
    w, n = r["windows"], len(r["rows"])
    print(f"windows: {n} (dropped next to gaps: {w.dropped})", file=out)
    print("per window (start–end, class, confidence):", file=out)
    for start, end, label, conf in r["rows"]:
        print(f"  {fmt_time(start)}–{fmt_time(end)}  {label}  {conf:.3f}", file=out)
    print("segments:", file=out)
    for start, end, label, k in r["segments"]:
        print(f"  {label} {fmt_time(start)}–{fmt_time(end)} ({k} window{'s' if k != 1 else ''})", file=out)
    print("counts:", file=out)
    for c in classes:
        print(f"  {c}: {r['counts'].get(c, 0)}/{n} ({_pct(r['counts'].get(c, 0), n)})", file=out)
    print(f"mean top probability: {sum(p for *_, p in r['rows']) / n:.3f}", file=out)
    clip = rec.meta["clipping"]
    print(f"EOL rows: {rec.meta['eol_rows']} (segments: {rec.meta['segments']})", file=out)
    axes = ", ".join(f"{a} {clip['per_axis'][a]}" for a in spec.CHANNELS)
    print(f"clipping (raw readings within {spec.CLIP_MARGIN_COUNTS} counts of the limits): {axes}; "
          f"rows with any clipped axis {clip['rows_any']}/{clip['rows']} "
          f"({100 * clip['fraction_rows']:.2f}%)" + (" WARNING" if clip["warn"] else ""), file=out)
    return r


def summary_line(name: str, label: str | None, result: dict | None, error: str | None,
                 classes: list[str], label_map: dict) -> str:
    if error is not None:
        return f"! {name}: error: {error}"
    n = len(result["rows"])
    counts = result["counts"]
    top = max(classes, key=lambda c: (counts.get(c, 0), -classes.index(c)))
    predicted = f"predicted {top} {counts.get(top, 0)}/{n} ({_pct(counts.get(top, 0), n)})"
    target = label_map.get(label) if label else None
    if label is None:
        return f"? {name}: no label; {predicted}"
    if target is None:
        return f"- {name}: {label}: not a model class; {predicted}"
    right = counts.get(target, 0)
    mark = "✓" if top == target else "✗"
    return f"{mark} {name}: {label} → {target}; {predicted}; {right}/{n} windows are {target}"


def main(argv=None, classifier=None, out=None) -> int:
    out = out or sys.stdout
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("target", nargs="?", help="a capture file, or the manifest (a .csv)")
    ap.add_argument("--manifest", nargs="?", const=str(MANIFEST_PATH), default=None,
                    help=f"run every manifest row (default {MANIFEST_PATH.name}) and print a summary")
    ap.add_argument("--accel-range-g", type=float, help="required for a capture file")
    ap.add_argument("--gyro-range-dps", type=float, help="required for a capture file")
    ap.add_argument("--label", help="capture file only: the true activity, if known")
    ap.add_argument("--repo-root", default=str(REPO_ROOT), help="manifest paths are relative to this")
    ap.add_argument("--model", default=str(MODEL_PATH))
    ap.add_argument("--meta", default=str(METADATA_PATH))
    args = ap.parse_args(argv)

    manifest = args.manifest
    if manifest is None and args.target and Path(args.target).suffix.lower() == ".csv":
        manifest = args.target
    if manifest is None:
        if not args.target:
            ap.error("give a capture file, or --manifest")
        if args.accel_range_g is None or args.gyro_range_dps is None:
            ap.error("a capture file needs --accel-range-g and --gyro-range-dps "
                     "(the ranges it was recorded at; nothing is guessed)")

    if classifier is None:
        if not (Path(args.model).is_file() and Path(args.meta).is_file()):
            print(f"no model at {args.model} (+ {args.meta}); train one first "
                  "(README, 'Activity model')", file=out)
            return 1
        try:
            classifier = ActivityClassifier.load(args.model, args.meta)
        except (ModelMetadataError, ValueError, OSError) as e:
            print(f"error: can't load the model: {e}", file=out)
            return 1
    classes, label_map, transform = _model_info(classifier)
    print(f"model: {classifier.model_version}  input transform: {transform}  "
          f"classes: {', '.join(classes)}", file=out)

    if manifest is not None:
        try:
            entries = load_manifest(manifest, args.repo_root)
        except (ManifestError, OSError, ValueError) as e:
            print(f"error: can't read the manifest {manifest}: {e}", file=out)
            return 1
        items = [(e.path, e.label, lambda e=e: load_recording(e, args.repo_root)) for e in entries]
    else:
        target = Path(args.target)
        items = [(str(target), args.label,
                  lambda: from_legacy_capture(target, args.label, accel_range_g=args.accel_range_g,
                                              gyro_range_dps=args.gyro_range_dps))]

    failed, summary = 0, []
    for name, label, load in items:
        result = error = None
        try:
            rec = _read_capture(load)
            label = rec.label or label
            result = report_one(name, rec, classifier, out)
        except CaptureError as e:
            error = str(e)
            failed += 1
            print(f"error: {name}: {error}", file=out)
        summary.append(summary_line(name, label, result, error, classes, label_map))
        print(file=out)
    if manifest is not None:
        print("summary (most common predicted class vs the manifest label through the model's label_map):",
              file=out)
        for line in summary:
            print(f"  {line}", file=out)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
