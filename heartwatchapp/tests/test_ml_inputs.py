"""Tests for the input experiment (handoff 12, Phase A): the four input
variants, the Still/Walking label map, folds with that map, the fixed
success rule, and the experiment run with a stub network (nothing is
trained here).

Plain-language overview: v1 learned the watch's angle instead of the
activity. The experiment feeds the same network inputs that don't depend
on the angle (gravity removed, or only magnitudes, or randomly rotated
training copies) and checks, on recordings it never saw, whether it then
matches the simple motion rule. These tests pin down the transforms and
the scoring so the experiment can't quietly change after results are seen.

New modules are imported inside the tests, so on the old code each test
fails on its own.

    python -m unittest heartwatchapp.tests.test_ml_inputs
"""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from ..ml import spec
from ..ml.inference import CLASSES
from .test_ml_v1 import FAKE_FILES, FakeStudy

REPO_ROOT = Path(__file__).resolve().parents[2]


def random_windows(n=7, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 0.1, size=(n, spec.WINDOW_SAMPLES, 6)).astype(np.float32)
    X[:, :, 2] += 0.25  # ~1 g on z after the 4 g normalisation
    return X


def norms(X):
    return np.linalg.norm(X[:, :, :3], axis=2), np.linalg.norm(X[:, :, 3:6], axis=2)


class TransformTest(unittest.TestCase):
    def test_centered_removes_each_window_accel_mean_and_keeps_gyro(self):
        from ..ml.training import input_variants as iv

        X = random_windows()
        out = iv.centered(X)
        self.assertEqual(out.shape, X.shape)
        np.testing.assert_allclose(out[:, :, :3].mean(axis=1), 0, atol=1e-6)
        self.assertEqual(out[:, :, 3:].tobytes(), X[:, :, 3:].tobytes())
        np.testing.assert_allclose(out[:, :, :3] - X[:, :, :3],
                                   np.broadcast_to(-X[:, :, :3].mean(axis=1, keepdims=True), out[:, :, :3].shape),
                                   atol=1e-6)

    def test_magnitude_shape_and_rotation_invariance(self):
        from ..ml.training import input_variants as iv

        X = random_windows()
        out = iv.magnitude(X)
        self.assertEqual(out.shape, (7, spec.WINDOW_SAMPLES, 2))
        a, g = norms(X)
        np.testing.assert_allclose(out[:, :, 0], a - a.mean(axis=1, keepdims=True), atol=1e-6)
        np.testing.assert_allclose(out[:, :, 1], g, atol=1e-6)
        R = iv.random_rotations(len(X), np.random.default_rng(5))
        np.testing.assert_allclose(iv.magnitude(iv.rotate(X, R)), out, atol=1e-5)

    def test_random_rotations_are_proper_rotations(self):
        from ..ml.training import input_variants as iv

        R = iv.random_rotations(50, np.random.default_rng(1))
        np.testing.assert_allclose(R @ np.transpose(R, (0, 2, 1)), np.broadcast_to(np.eye(3), (50, 3, 3)),
                                   atol=1e-12)
        np.testing.assert_allclose(np.linalg.det(R), 1, atol=1e-12)

    def test_rotation_augmentation(self):
        from ..ml.training import input_variants as iv

        X, y = random_windows(), np.array([0, 1, 0, 1, 1, 0, 0])
        Xa, ya = iv.rotation_augment(X, y, seed=3)
        self.assertEqual(Xa.shape, (70, spec.WINDOW_SAMPLES, 6))   # original + 9 copies
        self.assertEqual(Xa.dtype, X.dtype)
        self.assertEqual(Xa[:7].tobytes(), X.tobytes())           # originals kept as-is
        self.assertEqual(ya.tolist(), np.tile(y, 10).tolist())
        a, g = norms(X)
        a2, g2 = norms(Xa)
        np.testing.assert_allclose(a2, np.tile(a, (10, 1)), atol=1e-5)
        np.testing.assert_allclose(g2, np.tile(g, (10, 1)), atol=1e-5)
        self.assertFalse(np.allclose(Xa[7:14], X, atol=1e-3))      # the copies really are rotated
        again, _ = iv.rotation_augment(X, y, seed=3)
        self.assertEqual(again.tobytes(), Xa.tobytes())
        other, _ = iv.rotation_augment(X, y, seed=4)
        self.assertNotEqual(other.tobytes(), Xa.tobytes())

    def test_variant_registry(self):
        from ..ml.training import input_variants as iv

        self.assertEqual(list(iv.VARIANTS), ["raw", "centered", "magnitude", "rotation_aug"])
        self.assertEqual({k: v.channels for k, v in iv.VARIANTS.items()},
                         {"raw": 6, "centered": 6, "magnitude": 2, "rotation_aug": 6})
        X, y = random_windows(), np.zeros(7, np.int64)
        for name, v in iv.VARIANTS.items():
            with self.subTest(name):
                Xt = v.transform(X)
                self.assertEqual(Xt.shape[2], v.channels)
                Xtr, ytr = v.prepare_train(X, y, seed=0)
                self.assertEqual(len(Xtr), 70 if name == "rotation_aug" else 7)
                self.assertEqual(len(ytr), len(Xtr))
        self.assertEqual(iv.VARIANTS["raw"].transform(X).tobytes(), X.tobytes())
        self.assertEqual(iv.VARIANTS["rotation_aug"].transform(X).tobytes(), X.tobytes())  # test windows: never rotated


class LabelMapTest(unittest.TestCase):
    def test_map(self):
        from ..ml.training import input_variants as iv

        self.assertEqual(iv.EXPERIMENT_CLASSES, ("Still", "Walking"))
        self.assertEqual([iv.experiment_label(c) for c in ("Standing", "Sitting", "Walking")],
                         ["Still", "Still", "Walking"])
        for bad in ("Running", "Jogging", "Still"):
            with self.subTest(bad):
                with self.assertRaises(ValueError):
                    iv.experiment_label(bad)


class ExperimentFoldsTest(FakeStudy):
    """The fake study from test_ml_v1: Dylan Standing/Walking, Bobby
    Standing/Walking/Sitting, Bobby Running (exclude). 19 training-hop and
    10 live windows per file."""

    def relabelled(self):
        from ..ml.training import input_variants as iv

        return iv.relabel(self.files())

    def test_relabel_drops_excluded_and_maps_labels(self):
        files = self.relabelled()
        self.assertEqual([f.path for f in files], [p for p, _, _, use in FAKE_FILES if use == "train"])
        self.assertEqual([f.entry.label for f in files], ["Still", "Walking", "Still", "Walking", "Still"])
        for f in files:
            want = 0 if f.entry.label == "Still" else 1
            self.assertEqual(set(f.train.y.tolist()) | set(f.live.y.tolist()), {want})

    def test_folds(self):
        from ..ml.training import experiment_inputs as ex
        from ..ml.training import folds

        files = self.relabelled()
        fl = ex.build_folds(files)
        file_folds = [f for f in fl if f.kind == "file"]
        wearer_folds = [f for f in fl if f.kind == "wearer"]
        self.assertEqual(len(file_folds), 5)
        self.assertTrue(all(f.skipped is None for f in fl))          # BobbySit is scored now
        self.assertIn(("t/BobbySit.txt",), [f.test_paths for f in file_folds])
        self.assertEqual({f.name: set(f.train_paths) for f in wearer_folds}, {
            "wearer Bobby": {"t/DylanStand.txt", "t/DylanWalk.txt"},
            "wearer Dylan": {"t/BobbyStand.txt", "t/BobbyWalk.txt", "t/BobbySit.txt"},
        })
        train, test = folds.fold_splits(wearer_folds[1], files)
        self.assertEqual((len(train), len(test)), (19 * 3, 10 * 2))

    def test_no_held_out_window_reaches_training_even_rotated(self):
        from ..ml.training import experiment_inputs as ex
        from ..ml.training import folds

        files = self.relabelled()
        calls = []

        def stub(train_X, train_y, test_X, seed):
            calls.append((train_X, test_X))
            return np.tile([0.9, 0.1], (len(test_X), 1)).astype(np.float32)

        ex.evaluate(files, stub)
        all_folds = ex.build_folds(files)
        names = list(ex.VARIANTS)
        self.assertEqual(len(calls), len(all_folds) * len(names) * 3)
        k = 0

        def gyro_norms(X):  # rotation-invariant fingerprint; also channel 1 of "magnitude"
            return X[:, :, 1] if X.shape[2] == 2 else np.linalg.norm(X[:, :, 3:6], axis=2)

        for fold in all_folds:
            train, test = folds.fold_splits(fold, files)
            held = gyro_norms(test.X)
            own = gyro_norms(train.X)
            for name in names:
                for _seed in range(3):
                    train_X, test_X = calls[k]
                    k += 1
                    with self.subTest(fold=fold.name, variant=name):
                        expect = 10 * len(train) if name == "rotation_aug" else len(train)
                        self.assertEqual(len(train_X), expect)
                        self.assertEqual(len(test_X), len(test))           # live hop, never augmented
                        fp = gyro_norms(train_X)
                        d_held = np.abs(fp[:, None, :] - held[None, :, :]).max(axis=2).min()
                        d_own = np.abs(fp[:, None, :] - own[None, :, :]).max(axis=2).min(axis=1).max()
                        self.assertGreater(d_held, 1e-6)                    # nothing from the held-out file
                        self.assertLess(d_own, 1e-6)                        # positive control
                        if name == "rotation_aug":
                            self.assertEqual(test_X.tobytes(), test.X.tobytes())

    def test_rules_match_run_baselines(self):
        from ..ml.training import baselines as bl
        from ..ml.training import experiment_inputs as ex
        from ..ml.training import folds

        files = self.relabelled()
        results = ex.evaluate(files, lambda a, b, c, s: np.tile([1.0, 0.0], (len(c), 1)))
        for fold, r in zip(ex.build_folds(files), results):
            train, test = folds.fold_splits(fold, files)
            ref = {b["name"]: list(b["test"]) for b in bl.run_baselines(train, test)}
            self.assertEqual({k: v["score"] for k, v in r["rules"].items()},
                             {"majority": ref["Majority class"], "motion_rule": ref["Motion rule"],
                              "tilt_only": ref["Tilt only"]})
            self.assertEqual(r["variants"]["raw"]["per_seed"], [[int((test.y == 0).sum()), len(test)]] * 3)


class SuccessRuleTest(unittest.TestCase):
    def check(self, per_seed, motion, tilt, n=100):
        from ..ml.training import experiment_inputs as ex

        return ex.success_checks([(c, n) for c in per_seed], (motion, n), (tilt, n))

    def passes(self, checks):
        return {k: v["pass"] for k, v in checks.items()}

    def test_mean_vs_motion_boundary(self):
        self.assertTrue(self.check([85, 85, 85], 90, 50)["mean_vs_motion"]["pass"])   # exactly -5
        self.assertFalse(self.check([85, 85, 84], 90, 50)["mean_vs_motion"]["pass"])  # 84.67

    def test_worst_vs_motion_boundary(self):
        self.assertTrue(self.check([95, 95, 80], 90, 50)["worst_vs_motion"]["pass"])  # exactly -10
        self.assertFalse(self.check([96, 95, 79], 90, 50)["worst_vs_motion"]["pass"])

    def test_mean_vs_tilt_boundary(self):
        self.assertEqual(self.passes(self.check([60, 60, 60], 60, 40)),
                         {"mean_vs_motion": True, "worst_vs_motion": True, "mean_vs_tilt": True})
        self.assertEqual(self.passes(self.check([60, 60, 59], 60, 40)),
                         {"mean_vs_motion": True, "worst_vs_motion": True, "mean_vs_tilt": False})

    def test_exact_arithmetic_at_awkward_totals(self):
        # 109/136 = 80.147% vs 116/136 - 5 = 80.294%: fails; 110/136 = 80.88%: passes.
        self.assertFalse(self.check([109, 109, 109], 116, 0, n=136)["mean_vs_motion"]["pass"])
        self.assertTrue(self.check([110, 110, 110], 116, 0, n=136)["mean_vs_motion"]["pass"])
        # A boundary that float arithmetic gets wrong: 0.9 - 0.05 != 0.85 in floats.
        self.assertTrue(self.check([17, 17, 17], 18, 0, n=20)["mean_vs_motion"]["pass"])  # 85% vs 90% - 5

    def test_needs_all_checks_in_both_scopes(self):
        from ..ml.training import experiment_inputs as ex

        good = self.check([90, 90, 90], 90, 10)
        bad = self.check([90, 90, 70], 90, 10)
        self.assertTrue(ex.learns_motion(good, good))
        self.assertFalse(ex.learns_motion(good, bad))
        self.assertFalse(ex.learns_motion(bad, good))


class BuildModelDefaultsTest(unittest.TestCase):
    """v1 must stay byte-identical: the new options default to v1's shape."""

    def test_defaults_reproduce_v1(self):
        from ..ml.training import train, train_v1

        p = inspect.signature(train.build_model).parameters
        self.assertEqual((p["channels"].default, p["n_outputs"].default), (len(spec.CHANNELS), len(CLASSES)))
        q = inspect.signature(train_v1.keras_fit).parameters
        self.assertEqual((q["channels"].default, q["n_outputs"].default), (len(spec.CHANNELS), len(CLASSES)))
        self.assertEqual((q["epochs"].default, q["batch_size"].default), (40, 16))


class ExperimentRunTest(FakeStudy):
    def test_run_writes_report_and_scores_only_in_its_run_folder(self):
        from ..ml.training import experiment_inputs as ex

        runs = self.dir / "runs"
        out = ex.run(self.manifest_path, repo_root=self.root, runs_dir=runs,
                     fit_predict=lambda a, b, c, s: np.tile([0.2, 0.8], (len(c), 1)))
        run_dir = Path(out["run_dir"])
        self.assertEqual(run_dir.parent, runs)
        self.assertTrue(run_dir.name.endswith("-inputs"))
        self.assertEqual(sorted(p.name for p in run_dir.iterdir()), ["report.md", "scores.json"])
        s = json.loads((run_dir / "scores.json").read_text())
        for key in ("git", "manifest", "variants", "seeds", "success_rule", "classes", "label_map",
                    "folds", "totals", "success", "training"):
            self.assertIn(key, s)
        self.assertIn("dirty", s["git"])
        self.assertEqual(len(s["manifest"]["sha256"]), 64)
        self.assertEqual(s["seeds"], [0, 1, 2])
        self.assertEqual(list(s["variants"]), ["raw", "centered", "magnitude", "rotation_aug"])
        self.assertEqual(s["classes"], ["Still", "Walking"])
        self.assertEqual(len(s["folds"]), 7)
        self.assertEqual(set(s["success"]), set(s["variants"]))
        report = (run_dir / "report.md").read_text()
        self.assertIn("rotation_aug", report)
        self.assertIn("BobbySit", report)
        # Always "Walking": over the 5 file folds, 2 Walking files x 10 windows right out of 50.
        self.assertEqual(s["totals"]["files"]["variants"]["raw"]["per_seed"][0], [20, 50])
        self.assertEqual(sorted(p.name for p in self.dir.iterdir()), ["captures.csv", "repo", "runs"])

    def test_folds_only_writes_nothing(self):
        from ..ml.training import experiment_inputs as ex

        runs = self.dir / "runs"
        out = ex.run(self.manifest_path, repo_root=self.root, runs_dir=runs, folds_only=True)
        self.assertFalse(runs.exists())
        self.assertEqual(len(out["folds"]), 7)
        self.assertEqual(out["folds"][4]["test_windows"], {"Still": 10})


class NoTensorFlowTest(unittest.TestCase):
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
            "from heartwatchapp.ml.training import input_variants, experiment_inputs\n"
            "buf = io.StringIO()\n"
            "with contextlib.redirect_stdout(buf):\n"
            "    code = experiment_inputs.main(['--folds-only'])\n"
            "assert code == 0, code\n"
            "assert 'BobbySitting' in buf.getvalue()\n"
            "assert not [m for m in sys.modules if m.split('.')[0] in BLOCKED]\n"
            "print('ok')\n"
        )
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=REPO_ROOT,
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip().splitlines()[-1], "ok")


if __name__ == "__main__":
    unittest.main()
