"""Runs the exported activity model over a capture and prints what it says.

App environment only (LiteRT, no TensorFlow). From the repo root:

    heartwatchapp/.venv/bin/python -B -m heartwatchapp.ml.tools.predict_capture \\
        HeartWatch/test/BobbyStand.txt --accel-range-g 2 --gyro-range-dps 250
    heartwatchapp/.venv/bin/python -B -m heartwatchapp.ml.tools.predict_capture heartwatchapp/ml/captures.csv

Plain-language overview: the capture is cut into non-overlapping 5-second
windows (the way the app will see it), the model guesses an activity for
each, and the tool prints how many windows got each activity, the average
confidence, how many end-of-log rows were dropped, and how many raw
readings hit the sensor's limits. For a single capture file you must say
the sensor ranges it was recorded at (nothing is guessed); with the
manifest (a .csv) every row is run with its own ranges and label. Writes
nothing.
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

from .. import spec
from ..inference import CLASSES, METADATA_PATH, MODEL_PATH, ActivityClassifier
from ..manifest import REPO_ROOT, load_manifest, load_recording
from ..preprocess import prepare
from ..recording import from_legacy_capture


def _pct(c: int, n: int) -> str:
    return f"{100 * c / n:.0f}%" if n else "n/a"


def report_one(name: str, rec, classifier, out) -> None:
    w = prepare(rec, spec.LIVE_HOP_SAMPLES)
    preds = classifier.predict(w.X) if len(w.X) else []
    n = len(preds)
    counts = collections.Counter(label for label, _ in preds)
    clip = rec.meta["clipping"]
    print(f"== {name}  (label: {rec.label or 'none'}, model: {classifier.model_version})", file=out)
    print(f"windows: {n} (dropped next to gaps: {w.dropped})", file=out)
    for c in CLASSES:
        print(f"  {c}: {counts.get(c, 0)}/{n} ({_pct(counts.get(c, 0), n)})", file=out)
    mean = f"{sum(p for _, p in preds) / n:.3f}" if n else "n/a"
    print(f"mean top probability: {mean}", file=out)
    print(f"EOL rows: {rec.meta['eol_rows']} (segments: {rec.meta['segments']})", file=out)
    axes = ", ".join(f"{a} {clip['per_axis'][a]}" for a in spec.CHANNELS)
    print(f"clipping (raw readings within {spec.CLIP_MARGIN_COUNTS} counts of the limits): {axes}; "
          f"rows with any clipped axis {clip['rows_any']}/{clip['rows']} "
          f"({100 * clip['fraction_rows']:.2f}%)" + (" WARNING" if clip["warn"] else ""), file=out)


def main(argv=None, classifier=None, out=None) -> int:
    out = out or sys.stdout
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("target", help="a capture file, or the manifest (a .csv)")
    ap.add_argument("--accel-range-g", type=float, help="required for a capture file")
    ap.add_argument("--gyro-range-dps", type=float, help="required for a capture file")
    ap.add_argument("--label", help="capture file only: the true activity, if known")
    ap.add_argument("--repo-root", default=str(REPO_ROOT), help="manifest paths are relative to this")
    ap.add_argument("--model", default=str(MODEL_PATH))
    ap.add_argument("--meta", default=str(METADATA_PATH))
    args = ap.parse_args(argv)

    target = Path(args.target)
    if target.suffix.lower() == ".csv":
        items = [(e.path, load_recording(e, args.repo_root))
                 for e in load_manifest(target, args.repo_root)]
    else:
        if args.accel_range_g is None or args.gyro_range_dps is None:
            ap.error("a capture file needs --accel-range-g and --gyro-range-dps "
                     "(the ranges it was recorded at; nothing is guessed)")
        items = [(str(target), from_legacy_capture(target, args.label,
                                                   accel_range_g=args.accel_range_g,
                                                   gyro_range_dps=args.gyro_range_dps))]

    if classifier is None:
        if not (Path(args.model).is_file() and Path(args.meta).is_file()):
            print(f"no model at {args.model} (+ {args.meta}); train one first "
                  "(README, 'Activity model')", file=out)
            return 1
        classifier = ActivityClassifier.load(args.model, args.meta)
    for name, rec in items:
        report_one(name, rec, classifier, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
