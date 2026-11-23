"""Tests for the v1 data pipeline (handoff 11, Phase A): end-of-log rows,
raw clipping counts, explicit sensor ranges, the capture manifest,
whole-file / whole-wearer folds, the v1 training orchestration (with stub
models -- nothing is trained here), the prediction tool, and the
model_version check.

Plain-language overview: Bobby's captures mark every pause with one
all-zero reading (the firmware's "end of log" marker). It must never be
treated as a reading, and no 5-second window may mix data from before and
after it. Scores must come from files (and wearers) the model never saw.
These tests build small synthetic captures in temp folders so the expected
numbers can be worked out by hand.

New modules are imported inside the tests, so on the old code each test
fails on its own (with its own reason) rather than the whole file failing
to import.

    python -m unittest heartwatchapp.tests.test_ml_v1
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ..ml import inference, preprocess, recording, spec

REPO_ROOT = Path(__file__).resolve().parents[2]
CAPTURE_DIR = REPO_ROOT / "HeartWatch" / "test"
RANGES = {"accel_range_g": 2, "gyro_range_dps": 250}
EOL = [0, 0, 0, 0, 0, 0]

# SHA-256 of preprocess.prepare(...).X.tobytes() for the captures that have
# no end-of-log rows, recorded on the code BEFORE this change (commit
# 196fd30). A file without EOL rows must give byte-identical windows.
REFERENCE_WINDOW_SHA256 = {
    ("Standing", 25): "ebf304fff3a35685183b570285c867e9f24e8b1e701bc8fded285172b1c2b0ea",
    ("Standing", 50): "4bb84e5d662d8989765d224cf8c98e665d4f021ee9d9a0d0230cb53170f2ef49",
    ("Walking", 25): "d54d99b2b0bea12d5c768cfe37b84b6171a3c5903761c054301a5ecb3ec8b0b5",
    ("Walking", 50): "da3d9c1ad5afe0dc80bab4abc57252a8b2e3361513e0babb1fb9bc615ba648d8",
    ("Running", 25): "2f0095e2b7e0c789f435a78b633b9a36e53cce17af53abd353a2f86e6a49b1f9",
    ("Running", 50): "16df8b3bce936bbb1c82ac55cdc61d784215f7d356294f839ae24e1fe3e24000",
}


def imu_line(t: int, c) -> str:
    return f"I ({t}) mpu: AX: {c[0]}, AY: {c[1]}, AZ: {c[2]}, GX: {c[3]}, GY: {c[4]}, GZ: {c[5]}"


def write_capture(path: Path, rows, t0: int = 100_000, step_ms: int = 100) -> Path:
    """A capture in the layout of Bobby's files: HR line, IMU line, blank,
    esp line, blank -- one block per reading, step_ms apart, no time gaps."""
    lines = []
    for i, c in enumerate(rows):
        t = t0 + i * step_ms
        lines += [f"E ({t}) heartrate: (stable) Heart rate:  65.5 BPM", imu_line(t, c), "",
                  f"I ({t}) esp: ", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def dump_line(n: int, c, hr: int = 54) -> str:
    return (f"I ({52310 + n}) esp: log{n}: ax: {c[0]}, ay: {c[1]}, az: {c[2]}, "
            f"gx: {c[3]}, gy: {c[4]}, gz: {c[5]}, heartrate: {hr}")


def plus(n):   # side A of a break: ax = +0.5 g
    return [[8192, 0, 14000, 10, 10, 10]] * n


def minus(n):  # side B of a break: ax = -0.5 g
    return [[-8192, 0, 14000, -10, -10, -10]] * n


class TempDirTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()


def assert_no_straddle(test: unittest.TestCase, X: np.ndarray) -> tuple[int, int]:
    """Every window is all side A (ax > 0) or all side B (ax < 0), and no
    sample is all zeros. Returns (A windows, B windows)."""
    ax = X[:, :, 0]
    a = np.all(ax > 0, axis=1)
    b = np.all(ax < 0, axis=1)
    test.assertTrue(np.all(a | b), "a window mixes samples from both sides of an EOL row")
    test.assertFalse(np.any(np.all(X == 0, axis=2)), "a window contains an all-zero sample")
    return int(a.sum()), int(b.sum())


class EndOfLogLegacyTest(TempDirTest):
    def test_eol_row_is_a_break_even_without_a_time_gap(self):
        # 40 s of side A, the EOL row, 40 s of side B; timestamps keep
        # stepping with no gap (at 100 ms the EOL row sits alone in its
        # 100 ms bin; at 10 ms it shares a bin with both sides).
        for step_ms, per_side in ((100, 400), (10, 4000)):
            with self.subTest(step_ms=step_ms):
                path = write_capture(self.dir / f"eol{step_ms}.txt",
                                     plus(per_side) + [EOL] + minus(per_side), step_ms=step_ms)
                rec = recording.from_legacy_capture(path, "Walking")
                self.assertEqual(len(rec), 2 * per_side)
                self.assertEqual(rec.meta["eol_rows"], 1)
                self.assertFalse(np.any(np.all(rec.imu == 0, axis=1)))
                # Each side: 400 bins - 2 x 50 trimmed = 300 samples;
                # hop 25: starts 0..250 -> 11 windows; hop 50 -> 6.
                w = preprocess.prepare(rec, spec.TRAIN_HOP_SAMPLES)
                self.assertEqual(assert_no_straddle(self, w.X), (11, 11))
                live = preprocess.prepare(rec, spec.LIVE_HOP_SAMPLES)
                self.assertEqual(assert_no_straddle(self, live.X), (6, 6))

    def test_eol_rows_at_the_ends_disappear(self):
        body = plus(400)
        plain = recording.from_legacy_capture(write_capture(self.dir / "a.txt", body), "Walking")
        ends = recording.from_legacy_capture(
            write_capture(self.dir / "b.txt", [EOL] + body + [EOL], t0=99_900), "Walking")
        self.assertEqual(len(ends), 400)
        self.assertEqual(ends.meta["eol_rows"], 2)
        self.assertEqual(ends.ts_ms.tolist(), plain.ts_ms.tolist())
        self.assertEqual(ends.meta["segments"], 1)
        for hop in (spec.TRAIN_HOP_SAMPLES, spec.LIVE_HOP_SAMPLES):
            self.assertEqual(preprocess.prepare(ends, hop).X.tobytes(),
                             preprocess.prepare(plain, hop).X.tobytes())

    def test_resample_refuses_a_recording_with_breaks(self):
        path = write_capture(self.dir / "c.txt", plus(100) + [EOL] + minus(100))
        rec = recording.from_legacy_capture(path, "Walking")
        with self.assertRaisesRegex(ValueError, "segment"):
            preprocess.resample(rec)
        parts = preprocess.split_segments(rec)
        self.assertEqual([len(p) for p in parts], [100, 100])
        self.assertTrue(all(p.breaks is None for p in parts))

    def test_file_without_eol_rows_gives_the_same_windows_as_before(self):
        rows = [[(i * 37) % 9000, -(i * 11) % 7000, 15000, i % 300, -i % 200, 7] for i in range(900)]
        rec = recording.from_legacy_capture(write_capture(self.dir / "d.txt", rows, step_ms=10),
                                            "Walking")
        self.assertEqual(rec.meta["eol_rows"], 0)
        for hop in (spec.TRAIN_HOP_SAMPLES, spec.LIVE_HOP_SAMPLES):
            by_hand = preprocess.normalize(
                preprocess.windows(preprocess.trim(preprocess.resample(rec), spec.TRIM_S), hop).X)
            self.assertEqual(preprocess.prepare(rec, hop).X.tobytes(), by_hand.tobytes())

    @unittest.skipUnless(all((CAPTURE_DIR / f"{n}.txt").is_file()
                             for n in ("Standing", "Walking", "Running")),
                         "captures not present in HeartWatch/test/")
    def test_real_captures_without_eol_rows_are_byte_identical_to_before(self):
        for (name, hop), want in REFERENCE_WINDOW_SHA256.items():
            with self.subTest(name=name, hop=hop):
                rec = recording.from_legacy_capture(CAPTURE_DIR / f"{name}.txt")
                self.assertEqual(rec.meta["eol_rows"], 0)
                got = hashlib.sha256(preprocess.prepare(rec, hop).X.tobytes()).hexdigest()
                self.assertEqual(got, want)


class EndOfLogDumpTest(TempDirTest):
    def write_dump(self, rows) -> Path:
        path = self.dir / "Walking_20261122_01.txt"
        path.write_text("\n".join(dump_line(i, c, 60 + i % 3) for i, c in enumerate(rows)) + "\n")
        return path

    def test_eol_record_is_a_break(self):
        path = self.write_dump(plus(400) + [EOL] + minus(400))
        rec = recording.from_log_dump(path, rate_hz=10, **RANGES)
        self.assertEqual(len(rec), 800)
        self.assertEqual(rec.meta["eol_rows"], 1)
        self.assertEqual(len(rec.meta["heartrate"]), 800)
        self.assertEqual(assert_no_straddle(self, preprocess.prepare(rec, 25).X), (11, 11))

    def test_ranges_are_required_and_used(self):
        path = self.write_dump(plus(3))
        with self.assertRaises(TypeError):
            recording.from_log_dump(path, rate_hz=50)  # no silent +-2 g default
        rec = recording.from_log_dump(path, rate_hz=50, accel_range_g=4, gyro_range_dps=500)
        np.testing.assert_allclose(rec.imu[0, :2], [1, 0], rtol=0, atol=1e-12)  # 8192 = 1 g at +-4 g
        np.testing.assert_allclose(rec.imu[0, 3], 10 / 65.5, rtol=1e-12)
        self.assertEqual((rec.meta["accel_range_g"], rec.meta["gyro_range_dps"]), (4, 500))


class RangesTest(TempDirTest):
    def test_legacy_default_is_old_captures_2g_250dps(self):
        rec = recording.from_legacy_capture(write_capture(self.dir / "x.txt", [[16384, 0, 0, 131, 0, 0]]))
        np.testing.assert_allclose(rec.imu[0], [1, 0, 0, 1, 0, 0], rtol=0, atol=1e-12)
        self.assertEqual((rec.meta["accel_range_g"], rec.meta["gyro_range_dps"]), (2, 250))
        self.assertEqual((rec.meta["accel_scale"], rec.meta["gyro_scale"]),
                         (spec.LEGACY_ACCEL_SCALE, spec.LEGACY_GYRO_SCALE))

    def test_legacy_explicit_ranges(self):
        rec = recording.from_legacy_capture(write_capture(self.dir / "x.txt", [[4096, -4096, 0, 328, 0, 0]]),
                                            accel_range_g=8, gyro_range_dps=1000)
        np.testing.assert_allclose(rec.imu[0, :4], [1, -1, 0, 10], rtol=1e-12, atol=1e-12)
        self.assertEqual(rec.meta["accel_scale"], 1 / 4096)

    def test_unsupported_ranges_are_rejected(self):
        path = write_capture(self.dir / "x.txt", [[1, 1, 1, 1, 1, 1]])
        for kwargs in ({"accel_range_g": 3}, {"accel_range_g": 0}, {"accel_range_g": -2},
                       {"gyro_range_dps": 300}, {"gyro_range_dps": float("nan")}):
            with self.subTest(**kwargs):
                with self.assertRaisesRegex(ValueError, "range"):
                    recording.from_legacy_capture(path, **kwargs)


class ClippingTest(TempDirTest):
    ROWS = [
        [32767, 0, 16384, 0, 0, 0],    # ax at +limit
        [32757, 0, 16384, 0, 0, 0],    # ax within 10 counts
        [32756, 0, 16384, 0, 0, 0],    # 11 counts away: not clipped
        [0, -32768, 16384, 0, 0, 0],   # ay at -limit
        [0, -32758, 16384, 0, 0, 0],   # ay within 10 counts
        [0, -32757, 16384, 0, 0, 0],   # 11 counts away: not clipped
        [0, 0, 16384, 32767, -32768, 0],  # gx and gy, one row
        [100, 200, 16384, 5, 5, 5],
        [100, 200, 16384, 5, 5, 5],
        [100, 200, 16384, 5, 5, 5],
    ]

    def check(self, clip):
        self.assertEqual(clip["per_axis"], {"ax": 2, "ay": 2, "az": 0, "gx": 1, "gy": 1, "gz": 0})
        self.assertEqual((clip["rows"], clip["rows_any"], clip["rows_accel"], clip["rows_gyro"]),
                         (10, 5, 4, 1))
        self.assertEqual(clip["fraction_rows"], 0.5)
        self.assertTrue(clip["warn"])

    def test_spec_numbers(self):
        self.assertEqual(spec.CLIP_MARGIN_COUNTS, 10)
        self.assertEqual(spec.CLIP_WARN_FRACTION, 0.01)

    def test_legacy_counts(self):
        rec = recording.from_legacy_capture(write_capture(self.dir / "c.txt", self.ROWS))
        self.check(rec.meta["clipping"])

    def test_log_dump_counts(self):
        path = self.dir / "Running_20261122_01.txt"
        path.write_text("\n".join(dump_line(i, c) for i, c in enumerate(self.ROWS)) + "\n")
        self.check(recording.from_log_dump(path, rate_hz=50, **RANGES).meta["clipping"])

    def test_unclipped_and_the_one_percent_line(self):
        ok = [[100, 200, 16384, 5, 5, 5]] * 100
        clip = recording.from_legacy_capture(write_capture(self.dir / "u.txt", ok)).meta["clipping"]
        self.assertEqual(clip["per_axis"], dict.fromkeys(spec.CHANNELS, 0))
        self.assertEqual((clip["rows_any"], clip["warn"]), (0, False))
        one = [[32767, 0, 16384, 0, 0, 0]] + ok[1:]
        clip = recording.from_legacy_capture(write_capture(self.dir / "o.txt", one)).meta["clipping"]
        self.assertEqual((clip["rows_any"], clip["rows"], clip["warn"]), (1, 100, True))

    def test_eol_rows_are_not_counted_as_rows(self):
        rec = recording.from_legacy_capture(write_capture(self.dir / "e.txt", self.ROWS + [EOL]))
        self.assertEqual(rec.meta["clipping"]["rows"], 10)


# ---------------------------------------------------------------- manifest

MANIFEST_HEADER = ["path", "label", "wearer", "wrist", "orientation", "uploaded_by", "commit",
                   "firmware", "accel_range_g", "gyro_range_dps", "use", "notes"]


def manifest_row(path, label, wearer="Dylan", use="train", accel="2", gyro="250", notes=""):
    return {"path": path, "label": label, "wearer": wearer, "wrist": "unknown",
            "orientation": "unknown", "uploaded_by": wearer, "commit": "abc1234",
            "firmware": "test", "accel_range_g": accel, "gyro_range_dps": gyro, "use": use,
            "notes": notes}


def write_manifest(path: Path, rows, header=MANIFEST_HEADER) -> Path:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return path


class ManifestTest(TempDirTest):
    def test_the_real_manifest(self):
        from ..ml import manifest

        entries = manifest.load_manifest()
        self.assertEqual(list(manifest.COLUMNS), MANIFEST_HEADER)
        by_name = {Path(e.path).name: e for e in entries}
        self.assertEqual(set(by_name), {"Standing.txt", "Walking.txt", "Running.txt", "BobbyStand.txt",
                                        "BobbyWalking.txt", "BobbySitting.txt"})
        self.assertEqual({n: e.label for n, e in by_name.items()}, {
            "Standing.txt": "Standing", "Walking.txt": "Walking", "Running.txt": "Running",
            "BobbyStand.txt": "Standing", "BobbyWalking.txt": "Walking", "BobbySitting.txt": "Sitting"})
        self.assertEqual({n: e.wearer for n, e in by_name.items()}, {
            "Standing.txt": "Dylan", "Walking.txt": "Dylan", "Running.txt": "Bobby",
            "BobbyStand.txt": "Bobby", "BobbyWalking.txt": "Bobby", "BobbySitting.txt": "Bobby"})
        self.assertEqual({n for n, e in by_name.items() if e.use == "exclude"}, {"Running.txt"})
        for e in entries:
            self.assertEqual((e.wrist, e.orientation), ("unknown", "unknown"))
            self.assertEqual((e.accel_range_g, e.gyro_range_dps), (2, 250))
            self.assertTrue(e.path.startswith("HeartWatch/test/"))
        self.assertEqual(by_name["Standing.txt"].firmware, "1eabdb8")
        self.assertIn("ef150a7", by_name["BobbySitting.txt"].firmware)
        self.assertIn("tilt only", by_name["BobbySitting.txt"].notes)
        self.assertIn("24.3%", by_name["Running.txt"].notes)
        self.assertEqual(len(manifest.manifest_sha256()), 64)

    def fake_repo(self, rows, files=None):
        root = self.dir / "repo"
        for rel in files if files is not None else [r["path"] for r in rows]:
            write_capture(root / rel, [[4096, 0, 0, 0, 0, 0]] * 3)
        return root, write_manifest(self.dir / "captures.csv", rows)

    def test_manifest_label_wins_and_ranges_are_passed(self):
        from ..ml import manifest

        root, path = self.fake_repo([manifest_row("cap/BobbyStand.txt", "Standing", accel="8")])
        (entry,) = manifest.load_manifest(path, repo_root=root)
        self.assertEqual(entry.accel_range_g, 8)
        rec = manifest.load_recording(entry, repo_root=root)
        self.assertEqual(rec.label, "Standing")
        np.testing.assert_allclose(rec.imu[0, 0], 1.0, rtol=0, atol=1e-12)  # 4096 = 1 g at +-8 g
        self.assertEqual(rec.meta["manifest"]["wearer"], "Dylan")

    def test_validation_errors(self):
        from ..ml import manifest

        cases = {
            "not one of": [manifest_row("a/x.txt", "Jogging")],
            "duplicate": [manifest_row("a/x.txt", "Walking"), manifest_row("a/x.txt", "Walking")],
            "use": [manifest_row("a/x.txt", "Walking", use="maybe")],
            "accel_range_g": [manifest_row("a/x.txt", "Walking", accel="0")],
            "gyro_range_dps": [manifest_row("a/x.txt", "Walking", gyro="fast")],
            "range": [manifest_row("a/x.txt", "Walking", accel="-2")],
        }
        for what, rows in cases.items():
            with self.subTest(what):
                root, path = self.fake_repo(rows)
                with self.assertRaisesRegex(manifest.ManifestError, what):
                    manifest.load_manifest(path, repo_root=root)

    def test_missing_file_is_an_error(self):
        from ..ml import manifest

        root, path = self.fake_repo([manifest_row("a/gone.txt", "Walking")], files=[])
        with self.assertRaisesRegex(manifest.ManifestError, "gone.txt"):
            manifest.load_manifest(path, repo_root=root)

    def test_file_name_label_conflict_names_both(self):
        from ..ml import manifest

        for name in ("Standing.txt", "Standing_20261122_01.txt"):
            with self.subTest(name):
                root, path = self.fake_repo([manifest_row(f"a/{name}", "Walking")])
                with self.assertRaisesRegex(manifest.ManifestError, r"(?s)Standing.*Walking|Walking.*Standing"):
                    manifest.load_manifest(path, repo_root=root)

    def test_missing_column_is_an_error(self):
        from ..ml import manifest

        root, _ = self.fake_repo([manifest_row("a/x.txt", "Walking")])
        path = write_manifest(self.dir / "short.csv", [manifest_row("a/x.txt", "Walking")],
                              header=MANIFEST_HEADER[:-1])
        with self.assertRaisesRegex(manifest.ManifestError, "notes"):
            manifest.load_manifest(path, repo_root=root)


# ------------------------------------------------------------------ folds

# Fake study: two wearers. Each file gets its own az value, so every window
# can be traced back to the file it came from.
FAKE_FILES = [
    ("t/DylanStand.txt", "Standing", "Dylan", "train"),
    ("t/DylanWalk.txt", "Walking", "Dylan", "train"),
    ("t/BobbyStand.txt", "Standing", "Bobby", "train"),
    ("t/BobbyWalk.txt", "Walking", "Bobby", "train"),
    ("t/BobbySit.txt", "Sitting", "Bobby", "train"),
    ("t/BobbyRun.txt", "Running", "Bobby", "exclude"),
]


def az_counts(k: int) -> int:
    return 16000 - 500 * k


class FakeStudy(TempDirTest):
    """60 s at 10 Hz per file -> 600 bins - 100 trimmed = 500 samples:
    training hop 25 -> starts 0..450 = 19 windows; live hop 50 -> 10."""

    def setUp(self):
        super().setUp()
        self.root = self.dir / "repo"
        rng = np.random.default_rng(0)
        for k, (rel, label, _, _) in enumerate(FAKE_FILES):
            moving = label in ("Walking", "Running")
            rows = [[int(rng.normal(0, 3000 if moving else 50)), 0, az_counts(k),
                     int(rng.normal(0, 4000 if moving else 30)), 0, 0] for _ in range(600)]
            write_capture(self.root / rel, rows)
        self.manifest_path = write_manifest(self.dir / "captures.csv", [
            manifest_row(rel, label, wearer, use) for rel, label, wearer, use in FAKE_FILES])

    def files(self):
        from ..ml import manifest
        from ..ml.training import folds

        return folds.load_file_windows(manifest.load_manifest(self.manifest_path, repo_root=self.root),
                                       repo_root=self.root)

    def file_of(self, X) -> list[str]:
        """Which fake file each window came from, from its az value."""
        az = X[:, 0, 2] * spec.ACCEL_NORM_G / spec.LEGACY_ACCEL_SCALE
        return [FAKE_FILES[int(round((16000 - v) / 500))][0] for v in az]


class FoldsTest(FakeStudy):
    def test_window_counts_use_both_hops(self):
        for f in self.files():
            self.assertEqual((len(f.train), len(f.live)), (19, 10), f.entry.path)
            self.assertEqual(set(f.train.source), {f.entry.path})

    def test_leave_one_file_out(self):
        from ..ml.training import folds

        files = self.files()
        fl = folds.leave_one_file_out(files)
        self.assertEqual([f.test_paths for f in fl],
                         [(p,) for p, _, _, use in FAKE_FILES if use == "train"])
        skipped = {f.test_paths[0]: f.skipped for f in fl if f.skipped}
        self.assertEqual(skipped, {"t/BobbySit.txt": "can't test: only file of its class"})
        for fold in fl:
            if fold.skipped:
                continue
            train, test = folds.fold_splits(fold, files)
            held = fold.test_paths[0]
            self.assertEqual(set(self.file_of(test.X)), {held})
            self.assertEqual(len(test), 10)                       # live hop, non-overlapping
            self.assertNotIn(held, set(self.file_of(train.X)))    # never in training
            self.assertNotIn("t/BobbyRun.txt", set(self.file_of(train.X)))
            self.assertEqual(len(train), 19 * 4)                  # the other 4 train files
            self.assertEqual(set(train.source), set(fold.train_paths))

    def test_leave_one_wearer_out_uses_standing_and_walking_only(self):
        from ..ml.training import folds

        files = self.files()
        wl = folds.leave_one_wearer_out(files)
        self.assertEqual([(f.name, f.skipped) for f in wl],
                         [("wearer Bobby", None), ("wearer Dylan", None)])
        for fold in wl:
            train, test = folds.fold_splits(fold, files)
            wearer = fold.name.split()[-1]
            other = {"Bobby": "Dylan", "Dylan": "Bobby"}[wearer]
            self.assertEqual({p.split("/")[1][:5] for p in self.file_of(test.X)}, {wearer})
            self.assertEqual({p.split("/")[1][:5] for p in self.file_of(train.X)}, {other})
            used = {inference.CLASSES[i] for i in np.r_[train.y, test.y]}
            self.assertEqual(used, {"Standing", "Walking"})
            self.assertEqual((len(train), len(test)), (38, 20))

    def test_evaluate_folds_with_a_stub_cnn(self):
        from ..ml.training import folds

        calls = []
        walking, standing = inference.CLASSES.index("Walking"), inference.CLASSES.index("Standing")

        def stub(train_X, train_y, test_X, seed):
            calls.append(seed)
            probs = np.zeros((len(test_X), len(inference.CLASSES)), np.float32)
            probs[:, standing if seed == 2 else walking] = 1
            return probs

        files = self.files()
        results = folds.evaluate_folds(folds.leave_one_file_out(files), files, stub)
        self.assertEqual(calls, [0, 1, 2] * 4)                  # 4 scored folds, 3 seeds each
        by = {r["test_paths"][0]: r for r in results}
        sit = by["t/BobbySit.txt"]
        self.assertEqual(sit["status"], "can't test: only file of its class")
        self.assertNotIn("cnn", sit)
        walk = by["t/DylanWalk.txt"]
        self.assertEqual(walk["cnn"]["per_seed"], [[10, 10], [10, 10], [0, 10]])
        self.assertEqual(walk["cnn"]["worst"], [0, 10])
        self.assertAlmostEqual(walk["cnn"]["mean_correct"], 20 / 3)
        self.assertEqual(walk["true_class"], "Walking")
        self.assertEqual(walk["test_windows"], 10)
        names = [b["name"] for b in walk["baselines"]]
        self.assertEqual(names, ["Majority class", "Motion rule", "Tilt only"])
        for b in walk["baselines"]:
            self.assertEqual(b["test"][1], 10)
        # Majority is fitted on the fold's training files only: without
        # DylanWalk they are 2 Standing + 1 Walking + 1 Sitting -> Standing.
        self.assertEqual(walk["baselines"][0]["fitted"], {"class": "Standing"})
        self.assertEqual(walk["baselines"][0]["test"], [0, 10])


class TrainV1OrchestrationTest(FakeStudy):
    """train_v1.run() with stub fit/convert functions: nothing is trained,
    and everything is written into temp folders."""

    def stubs(self):
        def fit_predict(train_X, train_y, test_X, seed):
            probs = np.zeros((len(test_X), len(inference.CLASSES)), np.float32)
            probs[:, int(np.bincount(train_y).argmax())] = 1
            return probs

        def fit_final(X, y, seed):
            return lambda Z: np.full((len(Z), len(inference.CLASSES)), 0.25, np.float32)

        def convert(model):
            return b"fake tflite", {"path": "stub", "from_keras_model_error": None}

        return {"fit_predict": fit_predict, "fit_final": fit_final, "convert": convert,
                "versions": {"tensorflow_version": "stub"}}

    def run_once(self, models, runs, **kw):
        from ..ml.training import train_v1

        return train_v1.run(self.manifest_path, repo_root=self.root, models_dir=models,
                            runs_dir=runs, **self.stubs(), **kw)

    def test_archives_v0_then_writes_v1(self):
        models, runs = self.dir / "models", self.dir / "runs"
        models.mkdir()
        old = {"har_cnn.tflite": b"v0 model", "parity.npz": b"v0 parity",
               "model_meta.json": json.dumps({"model_version": "har-cnn-v0"}).encode()}
        for name, data in old.items():
            (models / name).write_bytes(data)
        out = self.run_once(models, runs)
        archive = models / "archive" / "har-cnn-v0"
        self.assertEqual({p.name: p.read_bytes() for p in archive.iterdir()}, old)
        self.assertEqual((models / "har_cnn.tflite").read_bytes(), b"fake tflite")
        meta = json.loads((models / "model_meta.json").read_text())
        self.assertEqual(meta["model_version"], "har-cnn-v1")
        inference.check_metadata(meta)
        self.assertEqual(meta["manifest"]["sha256"],
                         hashlib.sha256(self.manifest_path.read_bytes()).hexdigest())
        self.assertEqual([c["path"] for c in meta["captures"]],
                         [p for p, _, _, use in FAKE_FILES if use == "train"])
        self.assertEqual(meta["train_windows"], {"Sitting": 19, "Walking": 38, "Standing": 38, "Running": 0})
        self.assertEqual(meta["seeds"], {"folds": [0, 1, 2], "final": 0})
        self.assertIn("dirty", meta["git"])
        self.assertEqual(len(meta["folds"]), 5 + 2)
        report = Path(out["report"]).read_text()
        self.assertIn("can't test", report)
        self.assertIn("tilt only", report)
        self.assertTrue(Path(out["report"]).parent.parent == runs)
        # A second run never overwrites the v0 archive.
        self.run_once(models, runs)
        self.assertEqual({p.name: p.read_bytes() for p in archive.iterdir()}, old)

    def test_folds_only_writes_nothing(self):
        models, runs = self.dir / "models", self.dir / "runs"
        out = self.run_once(models, runs, folds_only=True)
        self.assertFalse(models.exists())
        self.assertFalse(runs.exists())
        self.assertEqual(len(out["folds"]), 7)
        self.assertEqual(out["window_counts"]["train"]["Standing"], 38)


# ---------------------------------------------------- tool, guard, imports

class StubClassifier:
    model_version = "stub"

    def predict(self, X):
        return [("Walking", 0.75)] * len(X)


class PredictToolTest(TempDirTest):
    def capture(self):
        return write_capture(self.dir / "BobbyWalking.txt",
                             plus(300) + [EOL] + [[32767, 0, 14000, 0, 0, 0]] + plus(300))

    def test_capture_form_with_a_stub(self):
        from ..ml.tools import predict_capture

        out = io.StringIO()
        code = predict_capture.main([str(self.capture()), "--accel-range-g", "2",
                                     "--gyro-range-dps", "250"], classifier=StubClassifier(), out=out)
        text = out.getvalue()
        self.assertEqual(code, 0, text)
        # 30 s + 30.1 s segments: 300 - 100 = 200 -> live starts 0..150 = 4
        # windows; 301 - 100 = 201 -> 4. 8 windows, all "Walking".
        self.assertIn("Walking: 8/8", text)
        self.assertIn("mean top probability: 0.750", text)
        self.assertIn("EOL rows: 1", text)
        self.assertIn("ax 1", text)

    def test_capture_form_needs_both_ranges(self):
        from ..ml.tools import predict_capture

        err = io.StringIO()
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(err):
            predict_capture.main([str(self.capture())], classifier=StubClassifier(), out=io.StringIO())
        self.assertIn("--accel-range-g", err.getvalue())

    def test_manifest_form(self):
        from ..ml.tools import predict_capture

        root = self.dir / "repo"
        write_capture(root / "t/A.txt", plus(300))
        path = write_manifest(self.dir / "captures.csv", [manifest_row("t/A.txt", "Walking")])
        out = io.StringIO()
        code = predict_capture.main([str(path), "--repo-root", str(root)],
                                    classifier=StubClassifier(), out=out)
        self.assertEqual(code, 0, out.getvalue())
        self.assertIn("t/A.txt", out.getvalue())
        self.assertIn("Walking: 4/4", out.getvalue())

    @unittest.skipUnless(inference.MODEL_PATH.is_file() and inference.METADATA_PATH.is_file(),
                         "no exported model in ml/models/")
    def test_with_the_real_model(self):
        from ..ml.tools import predict_capture

        out = io.StringIO()
        code = predict_capture.main([str(self.capture()), "--accel-range-g", "2",
                                     "--gyro-range-dps", "250"], out=out)
        self.assertEqual(code, 0, out.getvalue())
        self.assertIn("windows: 8", out.getvalue())


class ModelVersionGuardTest(TempDirTest):
    def test_metadata_without_model_version_is_refused(self):
        for meta in (inference.expected_metadata(),
                     {"model_version": "", **inference.expected_metadata()}):
            with self.subTest(meta.get("model_version")):
                path = self.dir / "meta.json"
                path.write_text(json.dumps(meta))
                with self.assertRaisesRegex(inference.ModelMetadataError, "model_version"):
                    inference.ActivityClassifier.load(self.dir / "missing.tflite", path)


class NoTensorFlowImportTest(unittest.TestCase):
    def test_new_modules_import_with_tensorflow_keras_and_pyside6_blocked(self):
        code = (
            "import sys\n"
            "BLOCKED = {'tensorflow', 'keras', 'PySide6'}\n"
            "class Block:\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name.split('.')[0] in BLOCKED:\n"
            "            raise ImportError(name + ' is blocked in this test')\n"
            "sys.meta_path.insert(0, Block())\n"
            "import heartwatchapp.ml.manifest, heartwatchapp.ml.recording, heartwatchapp.ml.preprocess\n"
            "import heartwatchapp.ml.training.folds, heartwatchapp.ml.training.train_v1\n"
            "import heartwatchapp.ml.tools.predict_capture\n"
            "assert not [m for m in sys.modules if m.split('.')[0] in BLOCKED]\n"
            "print('ok')\n"
        )
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=REPO_ROOT,
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "ok")


if __name__ == "__main__":
    unittest.main()
