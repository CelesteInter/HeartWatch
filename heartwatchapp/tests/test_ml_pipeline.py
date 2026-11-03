"""Tests for the activity-model pipeline (ml/spec, recording, preprocess,
inference, training/dataset, training/baselines).

Plain-language overview: the numbers the model is built on must agree with
the app's 5-second prediction window; the capture parser must read the
watch's log lines into real units and count everything it skips;
resampling, trimming and windowing must give the hand-computed results;
the same capture read straight from its file and read back from a
database must give byte-identical model input (the "skew" test); the
classifier must refuse a model whose metadata disagrees with the app; and,
when a model has been exported, LiteRT must give the same answers Keras
did. Expected values are worked out by hand from the rules, not copied
from a run. Only temp databases are used, never the real one.

    python -m unittest heartwatchapp.tests.test_ml_pipeline
"""

from __future__ import annotations

import contextlib
import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from ..data import db, stats
from ..ml import inference, preprocess, recording, spec
from ..ml.recording import Recording
from ..ml.training import baselines, dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
CAPTURE_DIR = REPO_ROOT / "HeartWatch" / "test"
EPOCH_OFFSET = 1_700_000_000_000

# The skew test's database copy uses these scales. To confirm the skew
# test can fail, set SKEW_ACCEL_SCALE to a wrong value (e.g. 1 / 8192)
# and watch test_capture_and_db_copy_give_identical_windows fail.
SKEW_ACCEL_SCALE = spec.LEGACY_ACCEL_SCALE
SKEW_GYRO_SCALE = spec.LEGACY_GYRO_SCALE


def imu_line(t: int, c) -> str:
    return f"I ({t}) mpu: AX: {c[0]}, AY: {c[1]}, AZ: {c[2]}, GX: {c[3]}, GY: {c[4]}, GZ: {c[5]}"


def write_synthetic_capture(path: Path, seconds: float = 30.0, seed: int = 1) -> None:
    """A capture in the firmware's log layout: IMU line, blank, esp, blank,
    HR line; 10 or 20 ms between IMU lines, like the real captures."""
    rng = random.Random(seed)
    t, end = 500_000, 500_000 + int(seconds * 1000)
    lines = ["I (499000) mpu: AX: 1, AY: 2, GX: 3E (499990) heartrate: (stable) Heart rate:  46 BPM"]
    while t <= end:
        counts = [rng.randint(-16000, 16000) for _ in range(3)] + [rng.randint(-3000, 3000) for _ in range(3)]
        lines += [imu_line(t, counts), "", f"I ({t}) esp: ", "",
                  f"E ({t + 10}) heartrate: (stable) Heart rate:  46 BPM"]
        t += rng.choice((10, 20))
    path.write_text("\n".join(lines) + "\n")


def write_db_copy(path: Path, ts, counts, accel_scale, gyro_scale, offset,
                  device_id="legacy-capture", label="Walking") -> int:
    """The capture's raw counts in a temp database, timestamps + offset."""
    db.init_db(path)
    with contextlib.closing(db._connect(path)) as conn, conn:
        sid = db.start_session(conn, device_id, label, accel_scale, gyro_scale,
                               started_at=int(ts[0]) + offset)
        db.insert_imu_batch(conn, sid, [(int(t) + offset, *(int(v) for v in c))
                                        for t, c in zip(ts, counts)])
        db.end_session(conn, sid, int(ts[-1]) + offset)
    return sid


def resampled(ts, imu, gap=None) -> Recording:
    imu = np.asarray(imu, dtype=np.float64)
    gap = np.zeros(len(ts), dtype=bool) if gap is None else np.asarray(gap)
    return Recording(np.asarray(ts, dtype=np.int64), imu, None, "test", {}, gap)


class SpecTest(unittest.TestCase):
    def test_window_matches_the_apps_prediction_window(self):
        self.assertEqual(spec.WINDOW_SAMPLES * 1000 / spec.RATE_HZ, stats.PREDICTION_WINDOW_MS)
        self.assertEqual(spec.WINDOW_MS, stats.PREDICTION_WINDOW_MS)
        self.assertEqual(spec.BIN_MS, 100)

    def test_input_shape(self):
        self.assertEqual(inference.INPUT_SHAPE, (1, 50, 6))
        self.assertEqual(len(spec.CHANNELS), 6)

    def test_ml_view_label_is_derived_from_input_shape(self):
        from ..ui import ml_view

        self.assertEqual(ml_view.INPUT_SHAPE_TEXT, "[1, 50, 6]  (window, channels)")


class ImportIsolationTest(unittest.TestCase):
    """The training environment has no PySide6; ml.inference must stay cheap."""

    def run_python(self, code: str) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT,
                              capture_output=True, text=True, timeout=120)

    def test_pipeline_modules_import_with_pyside6_blocked(self):
        code = (
            "import sys\n"
            "class Block:\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name.split('.')[0] == 'PySide6':\n"
            "            raise ImportError('PySide6 is blocked in this test')\n"
            "sys.meta_path.insert(0, Block())\n"
            "try:\n"
            "    import PySide6\n"
            "    raise SystemExit('block did not work')\n"
            "except ImportError:\n"
            "    pass\n"
            "import heartwatchapp.ml.spec, heartwatchapp.ml.inference\n"
            "import heartwatchapp.ml.recording, heartwatchapp.ml.preprocess\n"
            "import heartwatchapp.ml.training.dataset, heartwatchapp.ml.training.baselines\n"
            "assert not [m for m in sys.modules if m.split('.')[0] == 'PySide6']\n"
            "print('ok')\n"
        )
        result = self.run_python(code)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "ok")

    def test_inference_import_loads_no_numpy_or_litert(self):
        result = self.run_python(
            "import sys, heartwatchapp.ml.inference\n"
            "heavy = [m for m in sys.modules if m.split('.')[0] in "
            "('numpy', 'ai_edge_litert', 'tensorflow', 'PySide6')]\n"
            "print(heavy)\n"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "[]")


class LegacyParserTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name: str, lines: list[str]) -> Path:
        path = self.dir / name
        path.write_text("\n".join(lines) + "\n")
        return path

    def fixture(self, name="Walking.txt") -> Path:
        return self.write(name, [
            "I (900) mpu: AX: 100, AY: -2, GX: 1E (1000) heartrate: (stable) Heart rate:  46 BPM",
            "I (1000) mpu: AX: 16384, AY: -16384, AZ: 8192, GX: 131, GY: -262, GZ: 0",
            "",
            "I (1000) esp: ",
            "",
            "E (1010) heartrate: (stable) Heart rate:  46 BPM",
            "I (1010) mpu: AX: 0, AY: 0, AZ: 16384, GX: 0, GY: 0, GZ: -131",
            "something unexpected",
        ])

    def test_samples_units_and_skip_counts(self):
        rec = recording.from_legacy_capture(self.fixture())
        self.assertEqual(rec.ts_ms.tolist(), [1000, 1010])
        # 16384 counts = 1 g; 131 counts = 1 deg/s (sensor defaults).
        np.testing.assert_allclose(rec.imu, [[1, -1, 0.5, 1, -2, 0], [0, 0, 1, 0, 0, -1]],
                                   rtol=0, atol=1e-12)
        self.assertEqual(rec.meta["line_counts"],
                         {"imu": 2, "hr": 1, "esp": 1, "blank": 2, "spliced": 1, "other": 1})
        self.assertEqual(rec.label, "Walking")
        self.assertEqual(rec.source, "legacy-capture")
        self.assertEqual(rec.meta["hardware_note"], recording.HARDWARE_NOTE)
        self.assertIn("1eabdb8", rec.meta["hardware_note"])
        self.assertEqual(len(rec.meta["sha256"]), 64)

    def test_labels(self):
        self.assertIsNone(recording.from_legacy_capture(self.fixture("capture.txt")).label)
        self.assertEqual(recording.from_legacy_capture(self.fixture("x.txt"), "Sitting").label,
                         "Sitting")
        with self.assertRaises(ValueError):
            recording.from_legacy_capture(self.fixture(), "Jogging")

    def test_backwards_timestamps_rejected(self):
        path = self.write("x.txt", [imu_line(2000, [0] * 6), imu_line(1990, [0] * 6)])
        with self.assertRaises(ValueError):
            recording.from_legacy_capture(path)


@unittest.skipUnless((CAPTURE_DIR / "Standing.txt").is_file() and (CAPTURE_DIR / "Walking.txt").is_file(),
                     "captures not present in HeartWatch/test/")
class RealCapturesTest(unittest.TestCase):
    def test_counts_match_the_reconnaissance(self):
        expected = {
            "Standing.txt": {"imu": 7984, "hr": 7984, "esp": 7984, "blank": 15968, "spliced": 1, "other": 0},
            "Walking.txt": {"imu": 3442, "hr": 3453, "esp": 3442, "blank": 6884, "spliced": 1, "other": 0},
        }
        for name, counts in expected.items():
            with self.subTest(name):
                rec = recording.from_legacy_capture(CAPTURE_DIR / name)
                self.assertEqual(rec.meta["line_counts"], counts)
                self.assertEqual(len(rec), counts["imu"])
                self.assertEqual(rec.label, name.removesuffix(".txt"))

    def test_dataset_window_counts(self):
        # Standing: 1103 bins - 2 x 50 trimmed = 1003 samples; cut = 702.
        # Train: starts 0..650 step 25 (start + 50 <= 702) = 27 windows.
        # Test: starts >= 752 and <= 953 -> 775..950 = 8 windows.
        # Walking: 513 - 100 = 413; cut = 289; train 0..225 = 10; test
        # starts in [339, 363] -> 350 = 1 window.
        ds = dataset.build_dataset()
        self.assertEqual(dataset.counts_by_class(ds.train),
                         {"Sitting": 0, "Walking": 10, "Standing": 27, "Running": 0})
        self.assertEqual(dataset.counts_by_class(ds.test),
                         {"Sitting": 0, "Walking": 1, "Standing": 8, "Running": 0})


class ResampleTest(unittest.TestCase):
    def test_bin_means_half_open_bins_and_gaps(self):
        ts = [0, 30, 90, 100, 350]
        ax = [1.0, 2.0, 6.0, 10.0, 4.0]
        imu = np.zeros((5, 6))
        imu[:, 0] = ax
        out = preprocess.resample(Recording(np.array(ts), imu, None, "t", {}))
        # Bins [0,100): 1,2,6 -> 3; [100,200): 10 (100 is NOT in bin 0);
        # [200,300): empty -> gap; [300,400): 4. Stamped at bin ends.
        self.assertEqual(out.ts_ms.tolist(), [100, 200, 300, 400])
        self.assertEqual(out.gap.tolist(), [False, False, True, False])
        self.assertEqual(out.imu[[0, 1, 3], 0].tolist(), [3.0, 10.0, 4.0])
        self.assertTrue(np.isnan(out.imu[2]).all())
        self.assertEqual(out.meta["gap_bins"], 1)

    def test_10hz_input_passes_through(self):
        rng = np.random.default_rng(0)
        ts = 5000 + 100 * np.arange(20)
        imu = rng.normal(size=(20, 6))
        out = preprocess.resample(Recording(ts, imu, None, "t", {}))
        self.assertEqual(out.imu.tobytes(), imu.tobytes())
        self.assertEqual(out.ts_ms.tolist(), (ts + 100).tolist())
        self.assertFalse(out.gap.any())


class TrimWindowNormalizeTest(unittest.TestCase):
    def test_trim(self):
        ts = 100 * np.arange(1, 201)
        rec = preprocess.trim(resampled(ts, np.zeros((200, 6))), 5)
        self.assertEqual(len(rec), 100)          # 200 - 2 x 50
        self.assertEqual(rec.ts_ms[0], ts[50])
        self.assertEqual(rec.ts_ms[-1], ts[149])
        with self.assertRaises(ValueError):
            preprocess.trim(Recording(ts, np.zeros((200, 6)), None, "t", {}), 5)

    def test_window_counts_and_end_timestamps(self):
        ts = 100 * np.arange(1, 131)
        rec = resampled(ts, np.arange(130 * 6, dtype=float).reshape(130, 6))
        w = preprocess.windows(rec, 25)
        # Starts 0, 25, 50, 75 (75 + 50 = 125 <= 130; 100 + 50 > 130).
        self.assertEqual(w.start_index.tolist(), [0, 25, 50, 75])
        self.assertEqual(w.end_ts.tolist(), [ts[49], ts[74], ts[99], ts[124]])
        self.assertEqual(w.X.shape, (4, 50, 6))
        self.assertEqual(w.X[1, 0].tolist(), rec.imu[25].tolist())
        self.assertEqual((w.dropped, w.n_samples), (0, 130))
        self.assertEqual(preprocess.windows(rec, 50).start_index.tolist(), [0, 50])

    def test_windows_with_gaps_are_dropped(self):
        ts = 100 * np.arange(1, 131)
        gap = np.zeros(130, dtype=bool)
        gap[60] = True                            # inside windows starting 25 and 50
        w = preprocess.windows(resampled(ts, np.zeros((130, 6)), gap), 25)
        self.assertEqual(w.start_index.tolist(), [0, 75])
        self.assertEqual(w.dropped, 2)

    def test_normalize(self):
        X = np.zeros((1, 50, 6))
        X[0, 0] = [4.0, -2.0, 1.0, 500.0, -250.0, 50.0]
        out = preprocess.normalize(X)
        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(out[0, 0].tolist(), [1.0, -0.5, 0.25, 1.0, -0.5, 0.10000000149011612])

    def test_blocked_split(self):
        starts = np.arange(0, 1003 - 50 + 1, 25)
        train, test = dataset.blocked_split(1003, starts)
        self.assertEqual(int(train.sum()), 27)    # starts 0..650
        self.assertEqual(int(test.sum()), 8)      # starts 775..950
        self.assertFalse((train & test).any())
        last_train_end = starts[train].max() + 50
        self.assertGreaterEqual(starts[test].min() - last_train_end, 50)


class SkewTest(unittest.TestCase):
    """Same capture, two ways in: straight from the file, and via a temp DB."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def captures(self):
        synthetic = self.dir / "Walking.txt"
        write_synthetic_capture(synthetic)
        yield synthetic
        if (CAPTURE_DIR / "Walking.txt").is_file():
            yield CAPTURE_DIR / "Walking.txt"

    def test_capture_and_db_copy_give_identical_windows(self):
        for i, path in enumerate(self.captures()):
            with self.subTest(path.name, i=i):
                ts, counts, _, _ = recording.read_legacy_counts(path)
                db_path = self.dir / f"copy{i}.db"
                sid = write_db_copy(db_path, ts, counts, SKEW_ACCEL_SCALE, SKEW_GYRO_SCALE,
                                    EPOCH_OFFSET)
                a = preprocess.prepare(recording.from_legacy_capture(path, "Walking"),
                                       spec.TRAIN_HOP_SAMPLES)
                b = preprocess.prepare(recording.from_db_session(db_path, sid),
                                       spec.TRAIN_HOP_SAMPLES)
                self.assertGreater(len(a.X), 0)
                self.assertEqual(a.X.tobytes(), b.X.tobytes(),
                                 "prepare() output differs between the capture and its DB copy")
                self.assertEqual((b.end_ts - a.end_ts).tolist(), [EPOCH_OFFSET] * len(a.end_ts))

    def test_a_wrong_scale_is_detected(self):
        path = next(self.captures())
        ts, counts, _, _ = recording.read_legacy_counts(path)
        sid = write_db_copy(self.dir / "wrong.db", ts, counts, 1 / 8192, spec.LEGACY_GYRO_SCALE,
                            EPOCH_OFFSET)
        a = preprocess.prepare(recording.from_legacy_capture(path), spec.TRAIN_HOP_SAMPLES)
        b = preprocess.prepare(recording.from_db_session(self.dir / "wrong.db", sid),
                               spec.TRAIN_HOP_SAMPLES)
        self.assertNotEqual(a.X.tobytes(), b.X.tobytes())


class FromDbSessionGuardsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.ts = np.array([0, 100, 200])
        self.counts = np.zeros((3, 6), dtype=np.int64)

    def tearDown(self):
        self.tmp.cleanup()

    def test_untrusted_sources_are_refused(self):
        for i, device in enumerate(("seed", "demo", "demo-session")):
            with self.subTest(device):
                path = self.dir / f"{device}.db"
                sid = write_db_copy(path, self.ts, self.counts, spec.LEGACY_ACCEL_SCALE,
                                    spec.LEGACY_GYRO_SCALE, 0, device_id=device)
                with self.assertRaisesRegex(ValueError, "placeholders"):
                    recording.from_db_session(path, sid)

    def test_placeholder_scales_are_refused(self):
        path = self.dir / "p.db"
        sid = write_db_copy(path, self.ts, self.counts, 1.0, 1.0, 0, device_id="watch")
        with self.assertRaisesRegex(ValueError, r"placeholder scales \(1.0, 1.0\)"):
            recording.from_db_session(path, sid)

    def test_real_database_is_refused_before_opening(self):
        self.assertEqual(recording.REAL_DB_PATH.resolve(), db.DEFAULT_DB_PATH.resolve())
        fake_real = self.dir / "does-not-exist.db"
        with mock.patch.object(recording, "REAL_DB_PATH", fake_real):
            # ValueError, not FileNotFoundError: refused before any file access.
            with self.assertRaisesRegex(ValueError, "real database"):
                recording.from_db_session(fake_real, 1)

    def test_missing_path_or_session(self):
        with self.assertRaises(ValueError):
            recording.from_db_session(None, 1)
        path = self.dir / "ok.db"
        write_db_copy(path, self.ts, self.counts, spec.LEGACY_ACCEL_SCALE, spec.LEGACY_GYRO_SCALE, 0)
        with self.assertRaisesRegex(ValueError, "no session 99"):
            recording.from_db_session(path, 99)


class BaselinesTest(unittest.TestCase):
    def test_score_confusion_and_majority(self):
        self.assertEqual(baselines.score([1, 1, 2], [1, 2, 2]), (2, 3))
        m = baselines.confusion(np.array([1, 1, 2]), np.array([1, 2, 2]))
        self.assertEqual(m[1].tolist(), [0, 1, 1, 0])
        self.assertEqual(m[2].tolist(), [0, 0, 1, 0])
        self.assertEqual(baselines.majority_class(np.array([2, 2, 1])), 2)
        self.assertEqual(baselines.majority_class(np.array([2, 1])), 1)  # tie -> earlier


class ClassifierGuardsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def load_with(self, **changes):
        meta = {"model_version": "test", **inference.expected_metadata(), **changes}
        path = self.dir / "meta.json"
        path.write_text(json.dumps(meta))
        return inference.ActivityClassifier.load(self.dir / "missing.tflite", path)

    def test_mismatched_metadata_is_refused(self):
        cases = {
            "classes": list(reversed(inference.CLASSES)),
            "rate_hz": 50,
            "window_samples": 128,
            "channels": ["ax", "ay", "az"],
            "normalization": {"accel_g": 2.0, "gyro_dps": 500.0},
        }
        for key, value in cases.items():
            with self.subTest(key):
                with self.assertRaisesRegex(inference.ModelMetadataError, key):
                    self.load_with(**{key: value})

    def test_matching_metadata_reaches_the_model_file(self):
        with self.assertRaises(FileNotFoundError):
            self.load_with()


@unittest.skipUnless(
    inference.MODEL_PATH.is_file() and inference.METADATA_PATH.is_file()
    and inference.PARITY_PATH.is_file(),
    "no exported model in ml/models/ -- run training first (README, 'Activity model')",
)
class LiteRtParityTest(unittest.TestCase):
    def test_litert_matches_keras(self):
        clf = inference.ActivityClassifier.load()
        fixture = np.load(inference.PARITY_PATH)
        probs = clf.predict_proba(fixture["X"])
        self.assertEqual(probs.shape, fixture["probs"].shape)
        self.assertLessEqual(float(np.abs(probs - fixture["probs"]).max()), 1e-5)
        labels = clf.predict(fixture["X"])
        self.assertTrue(all(label in inference.CLASSES for label, _ in labels))


if __name__ == "__main__":
    unittest.main()
