"""The capture manifest: what each recorded capture is.

Plain-language overview: ml/captures.csv has one row per capture file --
its activity label, who wore the watch, which firmware and sensor ranges
recorded it, and whether it's used for training (`train`) or kept out
(`exclude`). Labels and facts come from this file, not from file names
(Bobby's files are called BobbyStand.txt and so on). To add a capture,
put the file in HeartWatch/test/ and add a row.

load_manifest() checks every row (label in CLASSES, file exists, no path
twice, `use` valid, ranges the sensor supports, and that a file name which
itself names an activity agrees with the label) and raises ManifestError
naming the row. load_recording() reads one row's file as a Recording,
always passing the manifest's label and ranges.

Standard library + numpy only (no TensorFlow, no PySide6). Every capture
so far is in the old ESP-IDF log format, so rows are read with
recording.from_legacy_capture().
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

from . import spec
from .inference import CLASSES
from .recording import Recording, from_legacy_capture

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = Path(__file__).resolve().parent / "captures.csv"
COLUMNS = ("path", "label", "wearer", "wrist", "orientation", "uploaded_by", "commit",
           "firmware", "accel_range_g", "gyro_range_dps", "use", "notes")
USES = ("train", "exclude")


class ManifestError(ValueError):
    """captures.csv has a row that can't be trusted."""


@dataclass(frozen=True)
class CaptureEntry:
    path: str              # repo-relative, e.g. HeartWatch/test/BobbyStand.txt
    label: str             # one of CLASSES
    wearer: str
    wrist: str             # "unknown" until someone records it
    orientation: str
    uploaded_by: str
    commit: str            # commit that added the file
    firmware: str
    accel_range_g: float
    gyro_range_dps: float
    use: str               # "train" or "exclude"
    notes: str


def label_implied_by_name(path: str | Path) -> str | None:
    """The activity a file name itself names, using the parsers' rules:
    the whole stem (Standing.txt) or the part before the first "_"
    (Sitting_20261110_03.txt), matched exactly against CLASSES."""
    stem = Path(path).stem
    if stem in CLASSES:
        return stem
    head = stem.split("_", 1)[0]
    return head if head in CLASSES else None


def _range(row_no: int, name: str, value: str, supported: dict) -> float:
    try:
        number = float(value)
    except ValueError:
        raise ManifestError(f"row {row_no}: {name} {value!r} is not a number") from None
    if not math.isfinite(number) or number <= 0:
        raise ManifestError(f"row {row_no}: {name} {value!r} must be a positive number")
    if number not in supported:
        raise ManifestError(f"row {row_no}: {name} {value!r} is not a range the sensor "
                            f"supports {sorted(supported)}")
    return number


def load_manifest(path: str | Path = MANIFEST_PATH, repo_root: str | Path = REPO_ROOT) -> list[CaptureEntry]:
    """Every row of the manifest, checked. Row numbers in errors count the
    header as row 1, like a spreadsheet."""
    path, repo_root = Path(path), Path(repo_root)
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = tuple(reader.fieldnames or ())
        missing = [c for c in COLUMNS if c not in header]
        if missing:
            raise ManifestError(f"{path}: missing column(s) {missing}; expected {list(COLUMNS)}")
        rows = list(reader)

    entries, seen = [], {}
    for row_no, row in enumerate(rows, start=2):
        values = {c: (row.get(c) or "").strip() for c in COLUMNS}
        rel = values["path"]
        if not rel:
            raise ManifestError(f"row {row_no}: empty path")
        if values["label"] not in CLASSES:
            raise ManifestError(f"row {row_no} ({rel}): label {values['label']!r} is not one of {CLASSES}")
        if values["use"] not in USES:
            raise ManifestError(f"row {row_no} ({rel}): use {values['use']!r} must be one of {USES}")
        if rel in seen:
            raise ManifestError(f"row {row_no}: duplicate path {rel!r} (also row {seen[rel]})")
        seen[rel] = row_no
        if not (repo_root / rel).is_file():
            raise ManifestError(f"row {row_no}: file not found: {rel}")
        implied = label_implied_by_name(rel)
        if implied is not None and implied != values["label"]:
            raise ManifestError(f"row {row_no} ({rel}): the file name says {implied!r} but the "
                                f"manifest label is {values['label']!r}")
        accel = _range(row_no, "accel_range_g", values["accel_range_g"], spec.ACCEL_COUNTS_PER_G)
        gyro = _range(row_no, "gyro_range_dps", values["gyro_range_dps"], spec.GYRO_COUNTS_PER_DPS)
        entries.append(CaptureEntry(**{**values, "accel_range_g": accel, "gyro_range_dps": gyro}))
    return entries


def manifest_sha256(path: str | Path = MANIFEST_PATH) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_recording(entry: CaptureEntry, repo_root: str | Path = REPO_ROOT) -> Recording:
    """One manifest row's capture as a Recording: the manifest's label
    (it always wins over the file name) and the manifest's sensor ranges.
    meta["manifest"] holds the row."""
    rec = from_legacy_capture(Path(repo_root) / entry.path, entry.label,
                              accel_range_g=entry.accel_range_g,
                              gyro_range_dps=entry.gyro_range_dps)
    rec.meta["manifest"] = dataclasses.asdict(entry)
    return rec
