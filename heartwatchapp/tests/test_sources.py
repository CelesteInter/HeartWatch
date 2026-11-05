"""Tests for the data-source layer (sources/).

Plain-language overview: sources/base.py must stay free of asyncio and
PySide6 (J), so the future Bluetooth or serial source can pick its own
threading; DummySource must produce the same fake data as before, on a
fixed schedule, deterministically when given a clock and an rng.

Expected values are worked out from the rules (the schedule, and the
formulas the Live Monitor has always used), not copied from a run.

    python -m unittest heartwatchapp.tests.test_sources
"""

from __future__ import annotations

import ast
import math
import random
import unittest
from pathlib import Path

from ..data import seed
from ..sources import base
from ..sources.dummy import DummySource

BASE_PY = Path(base.__file__)
FORBIDDEN_MODULES = {"asyncio", "PySide6"}


def forbidden_imports(source: str) -> list[str]:
    """Top-level module names in FORBIDDEN_MODULES that `source` imports."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names = [node.module]
        else:
            continue
        found += [name for name in names if name.split(".")[0] in FORBIDDEN_MODULES]
    return found


class Sink:
    def __init__(self):
        self.hr, self.imu, self.predictions = [], [], []

    def on_hr(self, readings):
        self.hr += readings

    def on_imu(self, readings):
        self.imu += readings

    def on_prediction(self, reading):
        self.predictions.append(reading)


class Clock:
    def __init__(self, now_ms: int):
        self.now = now_ms

    def __call__(self) -> int:
        return self.now


T = 1_000_000


def started(rng=None) -> tuple[DummySource, Sink, Clock]:
    clock, sink = Clock(T), Sink()
    source = DummySource(clock=clock, rng=rng if rng is not None else random.Random(1),
                         use_timer=False)
    source.start(sink)
    return source, sink, clock


class InterfaceGuardTests(unittest.TestCase):
    def test_j_base_imports_neither_asyncio_nor_pyside6(self) -> None:
        self.assertEqual(forbidden_imports(BASE_PY.read_text()), [])

    def test_j_the_guard_catches_each_form_of_import(self) -> None:
        clean = BASE_PY.read_text()
        for line in ("import asyncio", "from asyncio import Queue",
                     "import PySide6.QtCore as qc", "from PySide6.QtCore import QObject"):
            with self.subTest(line=line):
                self.assertNotEqual(forbidden_imports(clean + "\n" + line + "\n"), [])
        self.assertEqual(forbidden_imports("import dataclasses\nfrom . import dummy\n"), [])

    def test_dummy_source_is_a_data_source(self) -> None:
        source = DummySource(use_timer=False)
        self.assertIsInstance(source, base.DataSource)
        self.assertEqual((source.device_id, source.accel_scale, source.gyro_scale),
                         ("demo", 1.0, 1.0))


class ScheduleTests(unittest.TestCase):
    def test_first_readings_follow_the_old_timers(self) -> None:
        source, sink, clock = started()
        # Heart rate straight away, nothing else yet.
        self.assertEqual([r.ts_ms for r in sink.hr], [T])
        self.assertEqual(sink.imu, [])
        clock.now = T + 1_000
        source.poll()
        self.assertEqual([r.ts_ms for r in sink.hr], [T, T + 1_000])
        self.assertEqual([r.ts_ms for r in sink.imu], [T + 100 * k for k in range(1, 11)])
        self.assertEqual(sink.predictions, [])
        clock.now = T + 5_000
        source.poll()
        self.assertEqual([p.ts_ms for p in sink.predictions], [T + 5_000])
        self.assertEqual(sink.predictions[0].model_version, seed.DUMMY_MODEL_VERSION)

    def test_align_windows_restarts_the_prediction_schedule(self) -> None:
        source, sink, clock = started()
        clock.now = T + 2_000
        source.poll()
        source.align_windows(T + 2_000)
        for _ in range(100):  # to T + 12_000, in the timer's 100 ms steps
            clock.now += 100
            source.poll()
        self.assertEqual([p.ts_ms for p in sink.predictions], [T + 7_000, T + 12_000])

    def test_a_long_stall_skips_ahead_on_the_same_grid(self) -> None:
        source, sink, clock = started()
        clock.now = T + 10_000
        source.poll()
        # The backlog (more than 2 s) is skipped; one reading of each kind,
        # at the grid time that is due now.
        self.assertEqual([r.ts_ms for r in sink.hr], [T, T + 10_000])
        self.assertEqual([r.ts_ms for r in sink.imu], [T + 10_000])
        self.assertEqual([p.ts_ms for p in sink.predictions], [T + 10_000])

    def test_stop_ends_the_readings(self) -> None:
        source, sink, clock = started()
        source.stop()
        clock.now = T + 5_000
        source.poll()
        self.assertEqual(len(sink.hr), 1)


class ValueTests(unittest.TestCase):
    def test_values_follow_the_live_monitor_formulas(self) -> None:
        source, sink, clock = started(random.Random(7))
        clock.now = T + 100
        source.poll()  # first IMU reading
        # Same draws, by hand: HR (t = 1) first, then the six IMU steps.
        rng = random.Random(7)
        bpm = 75 + 12 * math.sin(1 / 6.0) + rng.uniform(-2.5, 2.5)
        self.assertEqual(sink.hr[0].bpm, int(round(bpm)))
        self.assertIsNone(sink.hr[0].confidence)
        spans = [4.0] * 3 + [500.0] * 3
        expected = tuple(int(round(rng.uniform(-0.04, 0.04) * span * 1000)) for span in spans)
        imu = sink.imu[0]
        self.assertEqual((imu.ax, imu.ay, imu.az, imu.gx, imu.gy, imu.gz), expected)

    def test_same_rng_seed_gives_the_same_readings(self) -> None:
        runs = []
        for _ in range(2):
            source, sink, clock = started(random.Random(11))
            for _ in range(300):
                clock.now += 100
                source.poll()
            runs.append((sink.hr, sink.imu, sink.predictions))
        self.assertEqual(runs[0], runs[1])
        self.assertEqual(len(runs[0][2]), 6)  # 30 s, one prediction per 5 s

    def test_without_an_rng_it_uses_the_global_random(self) -> None:
        runs = []
        for _ in range(2):
            random.seed(5)
            clock, sink = Clock(T), Sink()
            source = DummySource(clock=clock, use_timer=False)
            source.start(sink)
            clock.now = T + 5_000
            source.poll()
            runs.append((sink.hr, sink.imu, sink.predictions))
        self.assertEqual(runs[0], runs[1])

    def test_next_dummy_prediction_takes_an_optional_rng(self) -> None:
        random.seed(9)
        with_global = [seed.next_dummy_prediction("Walking") for _ in range(20)]
        rng = random.Random(9)
        with_rng = [seed.next_dummy_prediction("Walking", rng) for _ in range(20)]
        self.assertEqual(with_global, with_rng)


if __name__ == "__main__":
    unittest.main()
