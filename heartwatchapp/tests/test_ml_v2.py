"""Tests for the v2 `magnitude` model export (handoff 13, Phase A): the one
shared input transform, the model metadata contract, the export script
(with stub models -- nothing is trained here), predict_capture's segments,
summary and error handling, and the ML view's model status.

Plain-language overview: the input experiment showed that a network fed
only "how hard the wrist moves and turns" (no angle) tells Still from
Walking on recordings it never saw. v2 exports that model. The danger is
that training and the app turn a recording into model input in two
slightly different ways (training/serving skew), or that the app loads a
model whose input or classes it misunderstands. These tests pin both down.

New modules are imported inside the tests, so on the old code each test
fails on its own.

    python -m unittest heartwatchapp.tests.test_ml_v2
"""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from ..ml import inference, preprocess, spec
from .test_ml_v1 import FAKE_FILES, FakeStudy, TempDirTest, manifest_row, plus, write_capture, write_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]
ORIGINAL_CLASSES = ["Sitting", "Walking", "Standing", "Running"]

# SHA-256 of input_variants.magnitude(preprocess.prepare(rec, hop).X).tobytes()
# for every manifest capture, recorded on the code BEFORE this change
# (commit a218739). The shared transform must give byte-identical windows.
REFERENCE_MAGNITUDE_SHA256 = {
    ("Standing.txt", 25): "a229c750b00cf33d41b8bb78a300cca849acba9f94d9fbc756a6e0acf283fc40",
    ("Standing.txt", 50): "a46e11022e4ad446e02ee595b728f651597f82c8780c4315bb8622d46a010229",
    ("Walking.txt", 25): "1153d178a2980f6f489cc06c67ce2a22ddbdf912e5801c0ae4e13c7d58b997ec",
    ("Walking.txt", 50): "f87331c7d8254e42d75292a321feb770fe87cb086457054daec15f7639be7ce2",
    ("Running.txt", 25): "1d748b8ede486f27704394ae432ea834d5a0392a9f6c56301cc4932cf4751d62",
    ("Running.txt", 50): "6516ed03e22e16f08a9716ac9f1301d6ea5cd3948869a1b67b552190e29bd95f",
    ("BobbyStand.txt", 25): "fc7d457dc0bdf29d07753ad9855cd960b876e985bf540a5f36ea84de89a5bfa2",
    ("BobbyStand.txt", 50): "4ae63d0ee5594e3ad03eb8928fe8bd9499279cb60006134bfd1434aba9b01440",
    ("BobbyWalking.txt", 25): "9b4a4303aba97cbb42e2d3d826a99ecf81532dd2b719b0b8b4a7b15f7f213511",
    ("BobbyWalking.txt", 50): "779a5d7720c27c4db0c62140b6bdb8a6ffc42c9a2ce51ebf253b0daf6b90baa0",
    ("BobbySitting.txt", 25): "f1a161d6947d92251550e346cf50c8dea05559a2f9effe360eff0884fb08e5aa",
    ("BobbySitting.txt", 50): "955f5718354df2e0f86d54a7bed1ba3e6e8b68c656cd3086604d29e10f63af99",
}


def v2_meta(**changes) -> dict:
    meta = {"model_version": "har-cnn-v2-magnitude", **inference.expected_metadata(),
            "classes": ["Still", "Walking"], "input_transform": "magnitude", "input_channels": 2,
            "label_map": {"Standing": "Still", "Sitting": "Still", "Walking": "Walking"},
            "input_shape": [1, spec.WINDOW_SAMPLES, 2]}
    meta.update(changes)
    return meta


class FakeInterpreter:
    """Just enough of a LiteRT interpreter: records every input tensor and
    answers with a fixed probability row."""

    def __init__(self, channels: int, outputs: int, probs=None):
        self.channels, self.outputs = channels, outputs
        self.inputs = []
        self.probs = np.asarray(probs if probs is not None else [1.0] + [0.0] * (outputs - 1),
                                np.float32)

    def get_input_details(self):
        return [{"index": 0, "shape": np.array([1, spec.WINDOW_SAMPLES, self.channels])}]

    def get_output_details(self):
        return [{"index": 1, "shape": np.array([1, self.outputs])}]

    def set_tensor(self, index, value):
        self.inputs.append(np.array(value, copy=True))

    def invoke(self):
        pass

    def get_tensor(self, index):
        return self.probs[None, :]


def real_recording(name: str):
    from ..ml import manifest

    entry = next(e for e in manifest.load_manifest(manifest.MANIFEST_PATH) if e.path.endswith("/" + name))
    return manifest.load_recording(entry)


# --------------------------------------------------- A1: shared transform

class SharedTransformTest(unittest.TestCase):
    def test_magnitude_is_byte_identical_to_the_experiment_on_every_capture(self):
        from ..ml.training import input_variants as iv

        for (name, hop), want in REFERENCE_MAGNITUDE_SHA256.items():
            with self.subTest(file=name, hop=hop):
                X = preprocess.prepare(real_recording(name), hop).X
                self.assertEqual(hashlib.sha256(preprocess.model_input(X, "magnitude").tobytes()).hexdigest(), want)
                self.assertEqual(hashlib.sha256(iv.VARIANTS["magnitude"].transform(X).tobytes()).hexdigest(), want)

    def test_one_implementation(self):
        from ..ml import transforms
        from ..ml.training import input_variants as iv

        self.assertIs(iv.magnitude, transforms.magnitude)
        self.assertIs(iv.raw, transforms.raw)
        self.assertEqual(transforms.TRANSFORMS, {"raw": 6, "magnitude": 2})
        X = preprocess.prepare(real_recording("Walking.txt"), spec.LIVE_HOP_SAMPLES).X
        self.assertEqual(preprocess.model_input(X, "raw").tobytes(), X.tobytes())
        with self.assertRaisesRegex(ValueError, "bogus"):
            preprocess.model_input(X, "bogus")

    def test_training_and_inference_give_identical_magnitude_windows(self):
        """The skew test: the same capture through the export script's
        training path and through ActivityClassifier gives the same bytes."""
        from ..ml.training import train_v2_magnitude as v2

        rec = real_recording("BobbyWalking.txt")
        X = preprocess.prepare(rec, spec.LIVE_HOP_SAMPLES).X
        fake = FakeInterpreter(2, 2)
        clf = inference.ActivityClassifier.from_interpreter(fake, v2_meta())
        clf.predict_proba(X)
        served = np.concatenate(fake.inputs)
        trained = v2.VARIANT.transform(X)
        self.assertEqual(served.shape, (len(X), spec.WINDOW_SAMPLES, 2))
        self.assertEqual(served.dtype, np.float32)
        self.assertEqual(served.tobytes(), trained.tobytes())
        self.assertEqual(served.tobytes(), v2.to_model_input(X).tobytes())


# ---------------------------------------------------- A2: metadata contract

class LoaderContractTest(TempDirTest):
    def write(self, meta) -> Path:
        path = self.dir / "meta.json"
        path.write_text(json.dumps(meta))
        return path

    def test_unknown_transform_is_refused(self):
        with self.assertRaisesRegex(inference.ModelMetadataError, "input_transform.*bogus"):
            inference.ActivityClassifier.load(self.dir / "missing.tflite",
                                              self.write(v2_meta(input_transform="bogus")))

    def test_channel_count_that_doesnt_fit_the_transform_is_refused(self):
        with self.assertRaisesRegex(inference.ModelMetadataError, "input_channels"):
            inference.ActivityClassifier.load(self.dir / "missing.tflite",
                                              self.write(v2_meta(input_channels=6)))

    def test_model_input_shape_that_doesnt_match_the_metadata_is_refused(self):
        with self.assertRaisesRegex(inference.ModelMetadataError, "input shape"):
            inference.ActivityClassifier.from_interpreter(FakeInterpreter(6, 2), v2_meta())

    def test_output_count_that_doesnt_match_the_classes_is_refused(self):
        with self.assertRaisesRegex(inference.ModelMetadataError, "outputs"):
            inference.ActivityClassifier.from_interpreter(FakeInterpreter(2, 4), v2_meta())

    def test_label_map_must_point_at_model_classes(self):
        with self.assertRaisesRegex(inference.ModelMetadataError, "label_map"):
            inference.ActivityClassifier.load(
                self.dir / "missing.tflite", self.write(v2_meta(label_map={"Walking": "Jogging"})))

    def test_old_metadata_loads_as_raw_and_says_so(self):
        meta = {"model_version": "har-cnn-v1", **inference.expected_metadata()}
        with self.assertLogs("heartwatchapp.ml.inference", level="INFO") as logs:
            clf = inference.ActivityClassifier.from_interpreter(FakeInterpreter(6, 4), meta)
        self.assertIn("raw", "\n".join(logs.output))
        self.assertEqual(clf.input_transform, "raw")
        self.assertEqual(clf.classes, ORIGINAL_CLASSES)
        self.assertEqual(clf.label_map, {c: c for c in ORIGINAL_CLASSES})
        X = preprocess.prepare(real_recording("Walking.txt"), spec.LIVE_HOP_SAMPLES).X
        self.assertEqual(clf.predict(X[:2]), [("Sitting", 1.0)] * 2)
        # Without input_transform the old rule still holds: classes must be the global CLASSES.
        with self.assertRaisesRegex(inference.ModelMetadataError, "classes"):
            inference.ActivityClassifier.load(self.dir / "missing.tflite",
                                              self.write({**meta, "classes": ["Still", "Walking"]}))

    def test_v2_classes_come_from_the_metadata_and_global_classes_stay(self):
        clf = inference.ActivityClassifier.from_interpreter(FakeInterpreter(2, 2, [0.1, 0.9]), v2_meta())
        self.assertEqual(clf.classes, ["Still", "Walking"])
        self.assertEqual(clf.input_transform, "magnitude")
        self.assertEqual(clf.input_shape, (1, spec.WINDOW_SAMPLES, 2))
        X = preprocess.prepare(real_recording("Walking.txt"), spec.LIVE_HOP_SAMPLES).X
        preds = clf.predict(X[:3])
        self.assertEqual([p[0] for p in preds], ["Walking"] * 3)
        self.assertAlmostEqual(preds[0][1], 0.9, places=6)
        self.assertEqual(clf.model_class("Standing"), "Still")
        self.assertEqual(clf.model_class("Sitting"), "Still")
        self.assertEqual(clf.model_class("Walking"), "Walking")
        self.assertIsNone(clf.model_class("Running"))
        self.assertEqual(inference.CLASSES, ORIGINAL_CLASSES)
        self.assertEqual(inference.INPUT_SHAPE, (1, 50, 6))

    def test_classifier_takes_prepare_output_not_transformed_windows(self):
        clf = inference.ActivityClassifier.from_interpreter(FakeInterpreter(2, 2), v2_meta())
        with self.assertRaisesRegex(ValueError, "prepare"):
            clf.predict_proba(np.zeros((1, spec.WINDOW_SAMPLES, 2), np.float32))


# -------------------------------------------------------- A3: export script

def still_walking_by_gyro(train_X, train_y, test_X, seed):
    """A stand-in for the network on magnitude input: Walking when the
    window's mean |gyro| (channel 1) is large. Perfect on the fake study."""
    walking = test_X[:, :, 1].mean(axis=1) > 0.01
    return np.stack([~walking, walking], axis=1).astype(np.float32)


def always_wrong(train_X, train_y, test_X, seed):
    walking = test_X[:, :, 1].mean(axis=1) > 0.01
    return np.stack([walking, ~walking], axis=1).astype(np.float32)


class TrainV2Test(FakeStudy):
    OLD = {"har_cnn.tflite": b"v1 model", "parity.npz": b"v1 parity",
           "model_meta.json": json.dumps({"model_version": "har-cnn-v1"}).encode()}

    def setUp(self):
        super().setUp()
        self.models, self.runs = self.dir / "models", self.dir / "runs"
        self.models.mkdir()
        for name, data in self.OLD.items():
            (self.models / name).write_bytes(data)
        self.final_calls = []

    def run_v2(self, fit_predict, **kw):
        from ..ml.training import train_v2_magnitude as v2

        def fit_final(X, y, seed):
            self.final_calls.append((X.shape, sorted(set(np.asarray(y).tolist())), seed))
            return lambda Z: still_walking_by_gyro(None, None, Z, seed)

        def convert(model):
            return b"fake v2 tflite", {"path": "stub", "from_keras_model_error": None}

        return v2.run(self.manifest_path, repo_root=self.root, models_dir=self.models,
                      runs_dir=self.runs, fit_predict=fit_predict, fit_final=fit_final,
                      convert=convert, versions={"tensorflow_version": "stub"}, **kw)

    def models_snapshot(self):
        return {str(p.relative_to(self.models)): p.read_bytes()
                for p in sorted(self.models.rglob("*")) if p.is_file()}

    def test_failing_the_rule_exports_nothing_but_writes_the_report(self):
        before = self.models_snapshot()
        out = self.run_v2(always_wrong)
        self.assertFalse(out["passed"])
        self.assertIsNone(out["exported"])
        self.assertEqual(self.models_snapshot(), before)
        self.assertFalse((self.models / "archive").exists())
        self.assertEqual(self.final_calls, [])
        run_dir = Path(out["run_dir"])
        self.assertEqual(run_dir.parent, self.runs)
        self.assertTrue(run_dir.name.endswith("-v2"))
        self.assertEqual(sorted(p.name for p in run_dir.iterdir()), ["report.md", "scores.json"])
        report = (run_dir / "report.md").read_text()
        self.assertIn("FAIL", report)
        self.assertIn("Nothing was exported", report)
        s = json.loads((run_dir / "scores.json").read_text())
        self.assertFalse(s["pass_rule"]["passed"])

    def test_passing_the_rule_archives_v1_and_exports_v2(self):
        out = self.run_v2(still_walking_by_gyro)
        self.assertTrue(out["passed"])
        archive = self.models / "archive" / "har-cnn-v1"
        self.assertEqual({p.name: p.read_bytes() for p in archive.iterdir()}, self.OLD)
        self.assertEqual((self.models / "har_cnn.tflite").read_bytes(), b"fake v2 tflite")
        meta = json.loads((self.models / "model_meta.json").read_text())
        info = inference.check_metadata(meta)
        self.assertEqual(info["input_transform"], "magnitude")
        self.assertEqual(meta["model_version"], "har-cnn-v2-magnitude")
        self.assertEqual(meta["input_transform"], "magnitude")
        self.assertEqual(meta["input_channels"], 2)
        self.assertEqual(meta["input_shape"], [1, 50, 2])
        self.assertEqual(meta["classes"], ["Still", "Walking"])
        self.assertEqual(meta["label_map"], {"Standing": "Still", "Sitting": "Still", "Walking": "Walking"})
        self.assertEqual(meta["seeds"], {"folds": [0, 1, 2], "final": 0})
        self.assertEqual(meta["training"]["epochs"], 40)
        self.assertEqual(meta["training"]["batch_size"], 16)
        self.assertIn("dirty", meta["git"])
        self.assertTrue(meta["pass_rule"]["passed"])
        self.assertEqual([c["path"] for c in meta["captures"]],
                         [p for p, _, _, use in FAKE_FILES if use == "train"])
        self.assertEqual(meta["train_windows"], {"Still": 57, "Walking": 38})
        # Final model: every train file's training-hop windows, magnitude input, seed 0.
        self.assertEqual(self.final_calls, [((95, 50, 2), [0, 1], 0)])
        parity = np.load(self.models / "parity.npz")
        self.assertEqual(parity["X"].shape[1:], (50, 6))            # prepare() output: raw windows
        self.assertEqual(parity["probs"].shape, (len(parity["X"]), 2))
        self.assertEqual(parity["classes"].tolist(), ["Still", "Walking"])
        self.assertEqual(Path(out["exported"]["model"]), self.models / "har_cnn.tflite")
        report = Path(out["report"]).read_text()
        self.assertIn("pass", report)
        self.assertIn("har-cnn-v2-magnitude", report)

    def test_folds_only_writes_nothing(self):
        before = self.models_snapshot()
        out = self.run_v2(None, folds_only=True)
        self.assertEqual(self.models_snapshot(), before)
        self.assertFalse(self.runs.exists())
        self.assertEqual(len(out["folds"]), 7)

    def test_settings_are_the_experiments(self):
        from ..ml.training import experiment_inputs as ex
        from ..ml.training import train_v1
        from ..ml.training import train_v2_magnitude as v2

        self.assertEqual((v2.EPOCHS, v2.BATCH_SIZE), (train_v1.EPOCHS, train_v1.BATCH_SIZE))
        self.assertEqual(v2.SEEDS, ex.SEEDS)
        self.assertEqual(v2.FINAL_SEED, ex.SEEDS[0])
        self.assertIs(v2.SUCCESS_RULE, ex.SUCCESS_RULE)
        self.assertEqual(v2.VARIANT.channels, 2)


# ------------------------------------------------------ A4: predict_capture

class V2StubClassifier:
    model_version = "stub-v2"
    input_transform = "magnitude"
    classes = ["Still", "Walking"]
    label_map = {"Standing": "Still", "Sitting": "Still", "Walking": "Walking"}

    def __init__(self, labels=None):
        self.labels = labels

    def model_class(self, label):
        return self.label_map.get(label)

    def predict(self, X):
        labels = self.labels or ["Still"] * len(X)
        return [(labels[i % len(labels)], 0.875) for i in range(len(X))]


class SegmentMergeTest(unittest.TestCase):
    def merge(self, labels, gaps=()):
        from ..ml.tools import predict_capture as pc

        windows, t = [], 5000
        for i, label in enumerate(labels):
            if i in gaps:
                t += 5000
            windows.append((t, t + 5000, label))
            t += 5000
        return pc.merge_segments(windows)

    def test_single_window(self):
        self.assertEqual(self.merge(["Still"]), [(5000, 10000, "Still", 1)])

    def test_runs_merge(self):
        self.assertEqual(self.merge(["Still"] * 3 + ["Walking"] * 2),
                         [(5000, 20000, "Still", 3), (20000, 30000, "Walking", 2)])

    def test_alternating_labels(self):
        self.assertEqual(self.merge(["Still", "Walking", "Still", "Walking"]),
                         [(5000, 10000, "Still", 1), (10000, 15000, "Walking", 1),
                          (15000, 20000, "Still", 1), (20000, 25000, "Walking", 1)])

    def test_a_time_gap_splits_a_run(self):
        self.assertEqual(self.merge(["Still", "Still"], gaps=(1,)),
                         [(5000, 10000, "Still", 1), (15000, 20000, "Still", 1)])

    def test_empty(self):
        self.assertEqual(self.merge([]), [])

    def test_time_format(self):
        from ..ml.tools import predict_capture as pc

        self.assertEqual([pc.fmt_time(ms) for ms in (5000, 80_000, 599_999, 3_725_000)],
                         ["0:05", "1:20", "9:59", "1:02:05"])


class PredictV2OutputTest(TempDirTest):
    def run_tool(self, argv, classifier):
        from ..ml.tools import predict_capture

        out = io.StringIO()
        code = predict_capture.main(argv, classifier=classifier, out=out)
        return code, out.getvalue()

    def test_single_file_shows_model_windows_segments_and_counts(self):
        # 300 rows -> 200 samples after trimming -> 4 live windows; the last
        # 150 rows clip on ax, so the clipping WARNING must still be printed.
        path = write_capture(self.dir / "w.txt", plus(150) + [[32767, 0, 14000, 0, 0, 0]] * 150)
        code, text = self.run_tool([str(path), "--accel-range-g", "2", "--gyro-range-dps", "250"],
                                   V2StubClassifier(["Still", "Still", "Walking", "Walking"]))
        self.assertEqual(code, 0, text)
        self.assertIn("WARNING", text)
        self.assertTrue(text.startswith("model: stub-v2"), text)
        self.assertIn("input transform: magnitude", text.splitlines()[0])
        self.assertIn("0:05–0:10  Still  0.875", text)
        self.assertIn("Still 0:05–0:15 (2 windows)", text)
        self.assertIn("Walking 0:15–0:25 (2 windows)", text)
        self.assertIn("Still: 2/4 (50%)", text)
        self.assertIn("Walking: 2/4 (50%)", text)
        self.assertNotIn("Sitting", text)

    def test_manifest_summary_maps_labels_through_the_label_map(self):
        root = self.dir / "repo"
        for name in ("Stand.txt", "Walk.txt", "Run.txt"):
            write_capture(root / "t" / name, plus(300))
        mpath = write_manifest(self.dir / "captures.csv", [
            manifest_row("t/Stand.txt", "Standing"), manifest_row("t/Walk.txt", "Walking"),
            manifest_row("t/Run.txt", "Running", use="exclude")])
        code, text = self.run_tool(["--manifest", str(mpath), "--repo-root", str(root)], V2StubClassifier())
        self.assertEqual(code, 0, text)
        summary = text[text.index("summary"):]
        self.assertIn("✓ t/Stand.txt: Standing → Still; predicted Still", summary)
        self.assertIn("✗ t/Walk.txt: Walking → Walking; predicted Still", summary)
        self.assertIn("t/Run.txt: Running: not a model class", summary)


class PredictErrorsTest(TempDirTest):
    def run_tool(self, path: Path):
        from ..ml.tools import predict_capture

        out = io.StringIO()
        code = predict_capture.main([str(path), "--accel-range-g", "2", "--gyro-range-dps", "250"],
                                    classifier=V2StubClassifier(), out=out)
        return code, out.getvalue()

    def assert_clean_error(self, path: Path, *phrases):
        code, text = self.run_tool(path)   # must not raise
        self.assertNotEqual(code, 0, text)
        self.assertIn("error", text)
        self.assertNotIn("Traceback", text)
        for p in phrases:
            self.assertIn(p, text)

    def test_empty_log(self):
        path = self.dir / "empty.txt"
        path.write_text("")
        self.assert_clean_error(path, "no motion readings")

    def test_log_with_no_readings(self):
        path = self.dir / "notes.txt"
        path.write_text("hello\nthis is not a capture\n")
        self.assert_clean_error(path, "no motion readings")

    def test_too_short_for_one_window(self):
        path = write_capture(self.dir / "short.txt", plus(120))   # 12 s - 10 s trimmed = 2 s
        self.assert_clean_error(path, "too short")

    def test_unreadable(self):
        binary = self.dir / "bin.txt"
        binary.write_bytes(b"\xff\xfe\x00garbage")
        self.assert_clean_error(binary, "not a text log")
        self.assert_clean_error(self.dir / "missing.txt", "not found")
        self.assert_clean_error(self.dir, "directory")

    def test_one_bad_file_in_the_manifest_doesnt_stop_the_rest(self):
        from ..ml.tools import predict_capture

        root = self.dir / "repo"
        write_capture(root / "t/Stand.txt", plus(300))
        write_capture(root / "t/Walk.txt", plus(120))   # too short
        mpath = write_manifest(self.dir / "captures.csv", [
            manifest_row("t/Stand.txt", "Standing"), manifest_row("t/Walk.txt", "Walking")])
        out = io.StringIO()
        code = predict_capture.main(["--manifest", str(mpath), "--repo-root", str(root)],
                                    classifier=V2StubClassifier(), out=out)
        text = out.getvalue()
        self.assertEqual(code, 1, text)
        self.assertIn("✓ t/Stand.txt", text)
        self.assertIn("t/Walk.txt: error", text)


# ----------------------------------------------------------- A5: ML view

class MlViewStatusTest(TempDirTest):
    def test_status_shows_the_exported_models_shape_and_classes(self):
        from ..ui import ml_view

        path = self.dir / "model_meta.json"
        path.write_text(json.dumps(v2_meta(created_at="2026-11-24T10:00:00+00:00")))
        status = dict(ml_view.model_status(path))
        self.assertEqual(status["Input shape"], "[1, 50, 2]  (window, channels)")
        self.assertEqual(status["Output classes"], "Still · Walking")
        self.assertIn("har-cnn-v2-magnitude", status["Model file"])
        self.assertIn("magnitude", status["Model file"])
        self.assertEqual(status["Last updated"], "2026-11-24T10:00:00+00:00")

    def test_status_without_a_model_or_with_a_broken_file(self):
        from ..ui import ml_view

        broken = self.dir / "broken.json"
        broken.write_text("{not json")
        for path in (self.dir / "missing.json", broken):
            with self.subTest(path.name):
                status = dict(ml_view.model_status(path))
                self.assertEqual(status["Input shape"], "[1, 50, 6]  (window, channels)")
                self.assertEqual(status["Output classes"], "Sitting · Walking · Standing · Running")
                self.assertIn("no model", status["Model file"])

    def test_view_builds_offscreen_with_a_v2_model(self):
        path = self.dir / "model_meta.json"
        path.write_text(json.dumps(v2_meta()))
        code = (
            "import os, sys\n"
            "os.environ['QT_QPA_PLATFORM'] = 'offscreen'\n"
            "from pathlib import Path\n"
            "from PySide6.QtWidgets import QApplication, QLabel\n"
            "from heartwatchapp.ui import ml_view\n"
            "from heartwatchapp.ui.theme import Theme\n"
            "app = QApplication([])\n"
            f"view = ml_view.MlView(Theme(), meta_path=Path({str(path)!r}))\n"
            "texts = [w.text() for w in view.findChildren(QLabel)]\n"
            "assert any('[1, 50, 2]' in t for t in texts), texts\n"
            "print('ok')\n"
        )
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=REPO_ROOT,
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip().splitlines()[-1], "ok")


class NoTensorFlowV2Test(unittest.TestCase):
    def test_modules_and_folds_only_run_with_tensorflow_blocked(self):
        code = (
            "import sys\n"
            "BLOCKED = {'tensorflow', 'keras', 'PySide6'}\n"
            "class Block:\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name.split('.')[0] in BLOCKED:\n"
            "            raise ImportError(name + ' is blocked in this test')\n"
            "sys.meta_path.insert(0, Block())\n"
            "import io, contextlib\n"
            "import heartwatchapp.ml.transforms\n"
            "from heartwatchapp.ml.training import train_v2_magnitude\n"
            "buf = io.StringIO()\n"
            "with contextlib.redirect_stdout(buf):\n"
            "    code = train_v2_magnitude.main(['--folds-only'])\n"
            "assert code == 0, code\n"
            "assert 'BobbySitting' in buf.getvalue()\n"
            "assert not [m for m in sys.modules if m.split('.')[0] in BLOCKED]\n"
            "print('ok')\n"
        )
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=REPO_ROOT,
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip().splitlines()[-1], "ok")

    def test_transforms_import_loads_no_numpy(self):
        result = subprocess.run(
            [sys.executable, "-c", "import sys, heartwatchapp.ml.transforms, heartwatchapp.ml.inference\n"
             "print([m for m in sys.modules if m.split('.')[0] in ('numpy', 'ai_edge_litert')])"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "[]")


if __name__ == "__main__":
    unittest.main()
