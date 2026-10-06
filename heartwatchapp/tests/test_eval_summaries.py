"""Tests for scripts/eval_summaries.py's pure parts: seeds, options, result
formatting, the real-database refusal, and that the same --data-seed gives
the same session facts.

Stdlib unittest, same as the other test files. Run from the repo root:

    python -m unittest heartwatchapp.tests.test_eval_summaries

No live Ollama daemon is needed and nothing here calls it -- the live part
(_evaluate) only runs from the command line. Demo sessions are built on
new temporary databases; the real one is never opened.
"""

from __future__ import annotations

import contextlib
import datetime
import io
import re
import tempfile
import unittest
from pathlib import Path

from ..data import db
from ..llm import client as llm_client
from ..llm import validate
from ..scripts import eval_summaries

META = {
    "model": "gemma3:1b",
    "digest": "8648f39daa8f",
    "ollama_version": "0.34.0",
    "options_without_seed": {"num_ctx": 2048, "temperature": 0.2},
    "seed_base": 3,
    "runs": 2,
    "data_seed": 0,
    "fact_lines": ["Session length: 5 minutes 40 seconds", "Time Walking: 2 minutes 20 seconds"],
    "plain_fallback": "The session lasted 5 minutes 40 seconds.",
    "confidence_caveat": None,
}


class SeedAndOptionTests(unittest.TestCase):
    def test_run_i_uses_seed_base_plus_i(self) -> None:
        self.assertEqual(eval_summaries.run_seeds(0, 3), [0, 1, 2])
        self.assertEqual(eval_summaries.run_seeds(100, 2), [100, 101])

    def test_options_are_the_apps_plus_a_seed(self) -> None:
        options = eval_summaries.run_options(7)
        self.assertEqual(options, {**llm_client.SUMMARY_OPTIONS, "seed": 7})
        self.assertNotIn("seed", llm_client.SUMMARY_OPTIONS)


class ResultFormattingTests(unittest.TestCase):
    def _records(self) -> list[dict]:
        good = validate.ValidationResult(True, "Walking took 2 minutes 20 seconds.")
        bad = validate.ValidationResult(False, "A | B\nline two", validate.CHECK_UNGROUNDED, "99")
        timings = {"load_duration": 10, "eval_duration": 20}
        return [
            eval_summaries.run_record(
                3, "Sure: Walking took 2 minutes 20 seconds.", good, 1.2344, timings
            ),
            eval_summaries.run_record(4, "A | B\nline two", bad, 0.5, {}),
        ]

    def test_run_record_fields(self) -> None:
        good, bad = self._records()
        self.assertEqual(good["seed"], 3)
        self.assertTrue(good["passed"])
        self.assertIsNone(good["rejected_by"])
        self.assertEqual(good["stripped_text"], "Walking took 2 minutes 20 seconds.")
        self.assertEqual(good["raw_reply"], "Sure: Walking took 2 minutes 20 seconds.")
        self.assertEqual(good["latency_s"], 1.234)
        self.assertEqual((good["load_duration_ns"], good["eval_duration_ns"]), (10, 20))
        self.assertFalse(bad["passed"])
        self.assertEqual(bad["rejected_by"], "C (number not in the input)")
        self.assertEqual(bad["matched"], "99")
        self.assertIsNone(bad["load_duration_ns"])

    def test_summary_md_has_one_row_per_run_and_empty_label_columns(self) -> None:
        text = eval_summaries.format_summary_md(META, self._records())
        self.assertIn("| False statement? | Just restated the fact lines? |", text)
        self.assertIn("Passed validation: 1 of 2", text)
        self.assertIn("seed 3..4", text)
        self.assertIn("The session lasted 5 minutes 40 seconds.", text)
        self.assertIn("    Time Walking: 2 minutes 20 seconds", text)
        rows = [line for line in text.splitlines() if re.match(r"\| \d+ \|", line)]
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertTrue(row.endswith("|  |  |"))
        # A "|" or newline inside a reply can't break the table.
        self.assertIn("A \\| B <br> line two", rows[1])

    def test_output_dir_gets_a_suffix_instead_of_overwriting(self) -> None:
        now = datetime.datetime(2026, 9, 30, 14, 5, 9)
        with tempfile.TemporaryDirectory() as tmp:
            first = eval_summaries.make_output_dir(Path(tmp), now)
            second = eval_summaries.make_output_dir(Path(tmp), now)
            self.assertEqual(first.name, "20260930-140509")
            self.assertEqual(second.name, "20260930-140509-2")


class RealDatabaseTests(unittest.TestCase):
    def test_main_refuses_real_db_before_anything_else(self) -> None:
        stderr = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stderr(stderr):
            code = eval_summaries.main(["--db", str(db.DEFAULT_DB_PATH), "--out", tmp])
            self.assertEqual(list(Path(tmp).iterdir()), [])
        self.assertEqual(code, 2)
        self.assertIn("real database", stderr.getvalue())


class DataSeedTests(unittest.TestCase):
    def _facts(self, data_seed: int) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            session_stats = eval_summaries.build_session(Path(tmp) / "eval.db", data_seed)
        return eval_summaries.session_facts(session_stats)

    def test_same_data_seed_gives_the_same_facts(self) -> None:
        first, second = self._facts(0), self._facts(0)
        self.assertEqual(first, second)
        # The demo phases (scripts/demo_db.py) are 70 + 140 + 45 + 85 =
        # 340s, fully covered by 5s windows; most time first.
        self.assertEqual(
            first["fact_lines"][-4:],
            [
                "Time Walking: 2 minutes 20 seconds",
                "Time Standing: 1 minute 25 seconds",
                "Time Sitting: 1 minute 10 seconds",
                "Time Running: 45 seconds",
            ],
        )
        self.assertIn("Session length: 5 minutes 40 seconds", first["fact_lines"])


if __name__ == "__main__":
    unittest.main()
