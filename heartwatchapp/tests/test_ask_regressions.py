"""Regression tests for the ask bar's v1 bugs (handoff 9a, 3.5 and the
Phase A report).

Plain-language overview: each test is one wrong answer v1 gave. The model
is a stand-in that replies with exactly the form gemma3:1b gave in the
reconnaissance run, so no Ollama is needed. These tests use only what v1
already had (answer_question(), Answer.outcome, Answer.request), so they
run -- and fail -- against the ask-v1 code too.

    python -m unittest heartwatchapp.tests.test_ask_regressions
"""

from __future__ import annotations

import unittest

from .test_ask_pipeline import FakeModel, PipelineCase, form


class EvidenceCheckRegressionTests(PipelineCase):
    """The model's activity or range must appear in the question. llm
    mode, so the question reaches the model whatever the keyword parser
    would have done."""

    def test_and_walking_doesnt_become_yesterday(self) -> None:
        # Copied from the prompt example "walking duration yesterday".
        model = FakeModel(form("time_in_activity", "Walking", "yesterday"))
        answer = self.ask("and walking?", "llm", model)
        self.assertEqual(len(model.calls), 1)
        self.assertNotEqual(answer.request and answer.request.range, "yesterday")

    def test_latest_session_doesnt_gain_running(self) -> None:
        # Copied from "max bpm while jogging in my newest recording".
        model = FakeModel(form("peak_hr", "Running", "latest_session"))
        answer = self.ask("what was my peak heart rate in my latest session", "llm", model)
        self.assertEqual(answer.outcome, "query")
        self.assertIsNone(answer.request.activity)
        self.assertEqual(answer.request.range, "latest_session")

    def test_what_did_i_do_today_doesnt_become_walking(self) -> None:
        model = FakeModel(form("time_in_activity", "Walking", "today"))
        answer = self.ask("what did I do today", "llm", model)
        self.assertFalse(answer.request is not None and answer.request.activity == "Walking",
                         answer.text)


class MedicalTypoRegressionTests(PipelineCase):
    """v11-v13 of eval/ask_questions_dev_v11.jsonl. In v1 the misspelled
    health word was unknown, so the question went to the model, which
    answered with a number."""

    QUESTIONS = (
        "is my heart rate helthy this week",
        "shuold i be concernd about my avrage pulse",
        "was my peak bpm dangerus yesterday",
    )

    def test_medical_questions_with_typos_are_refused_in_every_mode(self) -> None:
        for mode in ("rules", "hybrid", "llm"):
            for question in self.QUESTIONS:
                with self.subTest(mode=mode, question=question):
                    model = FakeModel(form("avg_hr", "Walking", "this_week"))
                    answer = self.ask(question, mode, model)
                    self.assertEqual(answer.outcome, "medical", answer.text)
                    self.assertIsNone(answer.result)
                    self.assertEqual(model.calls, [])


if __name__ == "__main__":
    unittest.main()
