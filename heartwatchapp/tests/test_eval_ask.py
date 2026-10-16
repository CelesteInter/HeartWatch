"""Tests for scripts/eval_ask.py, the ask-bar parser evaluation.

Plain-language overview: the evaluation must refuse the real database
(whatever --overwrite says) before building anything, score forms the way
the writeup describes (a wrong field on an answered question is a false
accept), and run end to end in rules mode with no Ollama. Everything lives
in temporary folders; the real database is never opened.

    python -m unittest heartwatchapp.tests.test_eval_ask
"""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ..ask.parse_result import ParseResult
from ..data import db
from ..data.queries import MetricRequest
from ..scripts import eval_ask

TINY_QUESTIONS = [
    {"id": "a", "category": "straightforward", "question": "How long did I walk yesterday?",
     "expected": {"outcome": "query", "metric": "time_in_activity", "activity": "Walking",
                  "range": "yesterday"}},
    {"id": "b", "category": "medical", "question": "Is my heart rate healthy?",
     "expected": {"outcome": "medical"}},
    {"id": "c", "category": "unsupported", "question": "What's the weather?",
     "expected": {"outcome": "unsupported"}},
    # Deliberately wrong expectation: the parser answers "this week", so
    # this must be counted as a false accept.
    {"id": "d", "category": "relative_dates", "question": "Peak heart rate this week",
     "expected": {"outcome": "query", "metric": "peak_hr", "activity": None,
                  "range": "last_week"}},
]


class EvalCase(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.out = self.root / "results"
        self.out.mkdir()
        self.questions = self.root / "questions.jsonl"
        self.questions.write_text("".join(json.dumps(q) + "\n" for q in TINY_QUESTIONS))

    def run_main(self, *args: str) -> tuple[int, str, str]:
        stderr, stdout = io.StringIO(), io.StringIO()
        with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(stdout):
            code = eval_ask.main(["--questions", str(self.questions), "--out", str(self.out),
                                  *args])
        return code, stdout.getvalue(), stderr.getvalue()


class RealDatabaseTests(EvalCase):
    def test_refuses_the_real_db_with_and_without_overwrite(self) -> None:
        for extra in ([], ["--overwrite"]):
            with self.subTest(args=extra):
                code, _, err = self.run_main("--db", str(db.DEFAULT_DB_PATH), "--modes", "rules",
                                             *extra)
                self.assertEqual(code, 2)
                self.assertIn("real database", err)
                self.assertEqual(list(self.out.iterdir()), [])

    def test_refuses_an_existing_file_without_overwrite(self) -> None:
        dest = self.root / "eval.db"
        dest.write_bytes(b"someone's file")
        code, _, err = self.run_main("--db", str(dest), "--modes", "rules")
        self.assertEqual(code, 2)
        self.assertIn("--overwrite", err)
        self.assertEqual(dest.read_bytes(), b"someone's file")


class EndToEndTests(EvalCase):
    def test_rules_mode_runs_without_ollama(self) -> None:
        with mock.patch.object(eval_ask.llm_client, "health_check") as health:
            code, out, err = self.run_main("--modes", "rules")
        self.assertEqual(code, 0, err)
        health.assert_not_called()
        (run_dir,) = self.out.iterdir()
        self.assertTrue(run_dir.name.startswith("ask-"))
        records = [json.loads(line) for line in (run_dir / "runs.jsonl").read_text().splitlines()]
        by_id = {r["id"]: r for r in records}
        self.assertEqual([r["mode"] for r in records], ["rules"] * 4)
        self.assertTrue(by_id["a"]["exact"])
        # Pinned now is Wed Oct 14; yesterday's session (Oct 13) walked 420 s.
        self.assertEqual(by_id["a"]["reply"],
                         "Time spent Walking yesterday: 7 minutes (1 session).")
        self.assertEqual(by_id["b"]["actual"]["outcome"], "medical")
        self.assertEqual(by_id["c"]["actual"]["outcome"], "unsupported")
        self.assertTrue(by_id["d"]["false_accept"])
        meta = json.loads((run_dir / "meta.json").read_text())
        self.assertEqual((meta["modes_run"], meta["now"]),
                         (["rules"], "2026-10-14T15:00:00-06:00"))
        self.assertEqual(len(meta["questions_sha256"]), 64)
        summary = (run_dir / "summary.md").read_text()
        self.assertIn("**False accepts** | **1/4 (25%)**", summary)
        self.assertIn("| rules | d | Peak heart rate this week |", summary)

    def test_llm_modes_are_skipped_when_ollama_is_down(self) -> None:
        down = eval_ask.llm_client.unreachable_status(ConnectionError("refused"))

        async def fake_health():
            return down

        with mock.patch.object(eval_ask.llm_client, "health_check", fake_health):
            code, out, _ = self.run_main()
        self.assertEqual(code, 0)
        self.assertIn("skipping llm, hybrid", out)
        (run_dir,) = self.out.iterdir()
        meta = json.loads((run_dir / "meta.json").read_text())
        self.assertEqual((meta["modes_run"], meta["skipped_modes"]),
                         (["rules"], ["llm", "hybrid"]))


class ScoringTests(unittest.TestCase):
    def test_score(self) -> None:
        expected = {"outcome": "query", "metric": "avg_hr", "activity": None, "range": "today"}
        right = {"outcome": "query", "metric": "avg_hr", "activity": None, "range": "today"}
        wrong_range = {**right, "range": "yesterday"}
        refused = {"outcome": "not_understood", "metric": None, "activity": None, "range": None}
        self.assertTrue(eval_ask.score(expected, right)["exact"])
        self.assertTrue(eval_ask.score(expected, wrong_range)["false_accept"])
        self.assertEqual(eval_ask.score(expected, wrong_range)["fields"],
                         {"metric": True, "activity": True, "range": False})
        # Not answering isn't a false accept, just not a match.
        self.assertEqual(eval_ask.score(expected, refused),
                         {"exact": False, "false_accept": False,
                          "fields": {"metric": False, "activity": False, "range": False}})
        # Answering a question that should have been refused is.
        medical = {"outcome": "medical"}
        self.assertTrue(eval_ask.score(medical, right)["false_accept"])
        self.assertTrue(eval_ask.score(medical, {**refused, "outcome": "medical"})["exact"])

    def test_override_effect(self) -> None:
        def answer(final: str, llm_range: str | None):
            parse = ParseResult("accepted", MetricRequest("avg_hr", None, final), "llm", "",
                                range_source="rules", llm_range=llm_range)
            return mock.Mock(parse=parse)

        query = {"outcome": "query", "metric": "avg_hr", "activity": None, "range": "this_week"}
        self.assertEqual(eval_ask.override_effect(query, answer("this_week", "last_week")),
                         "fixed")
        self.assertEqual(eval_ask.override_effect({**query, "range": "last_week"},
                                                  answer("this_week", "last_week")), "broke")
        self.assertEqual(eval_ask.override_effect({**query, "range": "today"},
                                                  answer("this_week", "last_week")), "neutral")
        # A missing AI range would have defaulted to last_7_days.
        self.assertEqual(eval_ask.override_effect({**query, "range": "last_7_days"},
                                                  answer("this_week", None)), "broke")
        self.assertEqual(eval_ask.override_effect({"outcome": "medical"},
                                                  answer("this_week", "today")), "n/a")

    def test_question_file_errors_name_the_line(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "q.jsonl"
            path.write_text(json.dumps({"id": "x", "question": "hi",
                                        "expected": {"outcome": "maybe"}}) + "\n")
            with self.assertRaisesRegex(ValueError, "line 1: outcome must be one of"):
                eval_ask.load_questions(path)

    def test_the_dev_file_has_the_planned_mix(self) -> None:
        questions = eval_ask.load_questions(eval_ask.DEFAULT_QUESTIONS)
        counts: dict[str, int] = {}
        for q in questions:
            counts[q["category"]] = counts.get(q["category"], 0) + 1
        self.assertEqual(counts, {"straightforward": 6, "paraphrased": 6, "activity_synonyms": 5,
                                  "relative_dates": 5, "unsupported": 4, "medical": 4})
        texts = {q["question"].lower().rstrip("?") for q in questions}
        self.assertIn("is my heart rate healthy", texts)
        self.assertIn("is that higher than usual", texts)
