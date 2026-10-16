"""Tests for the AI bar's pipeline (ask/pipeline.py), the AI parser
(ask/llm_parser.py), HEARTWATCH_ASK_MODE (config.ask_mode()) and the
worker's parse request (llm/worker.py).

Plain-language overview: every cell of the metric x activity table is
answered end to end on a small fixture database, with the expected numbers
worked out by hand; each mode is checked with a stand-in for the AI model
(no Ollama needed), including the model being unavailable, timing out, or
replying with something that isn't a valid form; and the hybrid mode's
range override is checked both ways. The real database is never opened.

    python -m unittest heartwatchapp.tests.test_ask_pipeline
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import unittest
from unittest import mock

from .. import config
from ..ask import answers, llm_parser, pipeline
from ..ask.llm_parser import LlmUnavailable
from ..data import queries
from ..data.queries import MetricRequest
from ..llm import client as llm_client
from .test_queries import DENVER, NOW, FixtureDb, oct_midnight

NOW_DT = datetime.datetime.fromtimestamp(NOW / 1000, DENVER)   # Thu Oct 8 12:00 MDT


def form(metric=None, activity=None, range_name=None, intent="query") -> str:
    return json.dumps({"intent": intent, "metric": metric, "activity": activity,
                       "range": range_name})


class FakeModel:
    """Stands in for the AI model: returns `reply` (or raises `error`) and
    records every call."""

    def __init__(self, reply: str | None = None, error: Exception | None = None):
        self.reply, self.error, self.calls = reply, error, []

    def __call__(self, messages: list[dict], format: dict) -> str:
        self.calls.append((messages, format))
        if self.error is not None:
            raise self.error
        return self.reply


class PipelineCase(unittest.TestCase):
    """One finished session today (Oct 8, 01:00 MDT), 10 s long, from the
    dummy classifier: Walking [0, 5 s), Running [5, 10 s); heart rate 90
    at 0 s and 100 at 5 s."""

    def setUp(self) -> None:
        self.fixture = FixtureDb(self)
        start = oct_midnight(8) + 3_600_000
        self.fixture.add(start, start + 10_000, hr=[(0, 90), (5000, 100)],
                         predictions=[(5000, "Walking"), (10000, "Running")],
                         model_version="demo-dummy")

    def ask(self, question: str, mode: str = "rules", llm=None) -> pipeline.Answer:
        return pipeline.answer_question(question, conn_or_path=self.fixture.path, now=NOW_DT,
                                        mode=mode, llm=llm)


class MetricActivityTableTests(PipelineCase):
    """Handoff 3.4, every cell, through the rules parser and run()."""

    def test_every_cell(self) -> None:
        sim = queries.SIMULATED_CAPTION
        cases = [
            # avg_hr: all samples (90 + 100) / 2 = 95; while Running only the
            # 5 s sample, 100.
            ("average heart rate today", "Your average heart rate today was 95 bpm (1 session).",
             "Average heart rate · All activities · today", None),
            ("average heart rate while running today",
             "Your average heart rate while Running today was 100 bpm (1 session).",
             "Average heart rate · Running · today", sim),
            # peak_hr: 100 overall; while Walking only the 0 s sample, 90.
            ("peak heart rate today", "Your peak heart rate today was 100 bpm (1 session).",
             "Peak heart rate · All activities · today", None),
            ("peak heart rate while walking today",
             "Your peak heart rate while Walking today was 90 bpm (1 session).",
             "Peak heart rate · Walking · today", sim),
            # time_in_activity: Running owns [5, 10 s) = 5 s.
            ("how long did I run today", "Time spent Running today: 5 seconds (1 session).",
             "Time in activity · Running · today", sim),
            # session_count: one finished session today.
            ("how many sessions today", "You recorded 1 session today.",
             "Number of sessions · today", None),
            # total_time: 10 s; with an activity it becomes Running's 5 s.
            ("total time today", "Total recorded time today: 10 seconds (1 session).",
             "Total recorded time · today", None),
            ("total time running today", "Time spent Running today: 5 seconds (1 session).",
             "Time in activity · Running · today", sim),
        ]
        for question, text, understood, caption in cases:
            with self.subTest(question=question):
                answer = self.ask(question)
                self.assertEqual(answer.outcome, "query")
                self.assertEqual(answer.text, text)
                self.assertEqual(answer.understood, understood)
                self.assertEqual(answer.caption, caption)

    def test_time_without_an_activity_asks_which(self) -> None:
        answer = self.ask("how much time did I spend today")
        self.assertEqual(answer.outcome, "needs_activity")
        self.assertEqual(answer.text,
                         "Which activity? I can answer for Sitting, Walking, Standing or Running.")
        self.assertIsNone(answer.result)

    def test_session_count_by_activity_isnt_supported(self) -> None:
        # Through the AI parser (the model's form says so)...
        model = FakeModel(form("session_count", "Running", "today"))
        answer = self.ask("how many sessions with a jog in them today", mode="hybrid", llm=model)
        self.assertEqual((answer.source, answer.outcome), ("llm", "session_count_by_activity"))
        self.assertEqual(answer.text, answers.SESSION_COUNT_BY_ACTIVITY_REPLY)
        self.assertIsNone(answer.result)

    def test_session_count_by_activity_skips_the_model_in_rules_and_hybrid(self) -> None:
        # ...but an explicit "how many <activity> sessions" never reaches
        # it, even when the model would have answered something else.
        for mode in ("rules", "hybrid"):
            with self.subTest(mode=mode):
                model = FakeModel(form("time_in_activity", "Walking", "last_week"))
                answer = self.ask("How many walking sessions did I have last week?", mode, model)
                self.assertEqual((answer.source, answer.outcome),
                                 ("rules", "session_count_by_activity"))
                self.assertEqual(model.calls, [])
        # llm mode stays pure: the model decides.
        model = FakeModel(form("time_in_activity", "Walking", "last_week"))
        answer = self.ask("How many walking sessions did I have last week?", "llm", model)
        self.assertEqual((answer.source, answer.outcome), ("llm", "query"))
        self.assertEqual(len(model.calls), 1)

    def test_a_bare_sessions_word_still_goes_to_the_model(self) -> None:
        model = FakeModel(form("time_in_activity", "Walking", "today"))
        answer = self.ask("how many minutes did I walk across my sessions today", "hybrid", model)
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(answer.source, "llm")

    def test_default_range_is_applied_and_shown(self) -> None:
        answer = self.ask("average heart rate")
        self.assertEqual(answer.request.range, "last_7_days")
        self.assertEqual(answer.understood,
                         "Average heart rate · All activities · last 7 days (default)")
        self.assertEqual(answer.text,
                         "Your average heart rate in the last 7 days was 95 bpm (1 session).")

    def test_normalize_table(self) -> None:
        cases = [
            (MetricRequest("time_in_activity", None, "today"), "needs_activity", None),
            (MetricRequest("session_count", "Walking", "today"), "session_count_by_activity",
             None),
            (MetricRequest("total_time", "Walking", "today"), "query",
             MetricRequest("time_in_activity", "Walking", "today")),
            (MetricRequest("total_time", None, "today"), "query",
             MetricRequest("total_time", None, "today")),
            (MetricRequest("avg_hr", "Walking", None), "query",
             MetricRequest("avg_hr", "Walking", "last_7_days")),
        ]
        for request, outcome, normalized in cases:
            with self.subTest(request=request):
                result = pipeline.normalize(request)
                self.assertEqual((result.outcome, result.request), (outcome, normalized))


class GuardInPipelineTests(PipelineCase):
    def test_medical_questions_get_no_number_in_any_mode(self) -> None:
        for mode in ("rules", "hybrid", "llm"):
            with self.subTest(mode=mode):
                model = FakeModel(form("avg_hr", None, "today"))
                answer = self.ask("Was my average heart rate too high today?", mode, model)
                self.assertEqual(answer.outcome, "medical")
                self.assertEqual(answer.text, answers.MEDICAL_REPLY)
                self.assertEqual(model.calls, [])          # the model never sees it
                self.assertIsNone(answer.result)
                self.assertFalse(any(ch.isdigit() for ch in answer.text))


class ModeTests(PipelineCase):
    def test_rules_mode_never_calls_the_model(self) -> None:
        model = FakeModel(form("avg_hr", None, "today"))
        answer = self.ask("average heart rate on the treadmill", "rules", model)
        self.assertEqual(answer.outcome, "not_understood")
        self.assertEqual(answer.text, answers.NOT_UNDERSTOOD_REPLY)
        self.assertEqual(model.calls, [])

    def test_hybrid_answers_simple_questions_without_the_model(self) -> None:
        model = FakeModel(form("peak_hr", None, "all"))
        answer = self.ask("average heart rate today", "hybrid", model)
        self.assertEqual((answer.source, answer.request.metric), ("rules", "avg_hr"))
        self.assertEqual(model.calls, [])

    def test_hybrid_sends_deferred_questions_to_the_model(self) -> None:
        model = FakeModel(form("avg_hr", None, "today"))
        answer = self.ask("average heart rate on the treadmill today", "hybrid", model)
        self.assertEqual(len(model.calls), 1)
        self.assertEqual((answer.source, answer.outcome), ("llm", "query"))
        self.assertEqual(answer.text, "Your average heart rate today was 95 bpm (1 session).")
        messages, schema = model.calls[0]
        self.assertEqual(messages[-1]["content"], "average heart rate on the treadmill today")
        self.assertEqual(schema, llm_parser.response_schema())

    def test_llm_mode_sends_every_question_to_the_model(self) -> None:
        model = FakeModel(form("peak_hr", None, "today"))
        answer = self.ask("average heart rate today", "llm", model)
        self.assertEqual(len(model.calls), 1)
        # The model's form is what's answered, and the understood line shows it.
        self.assertEqual(answer.understood, "Peak heart rate · All activities · today")

    def test_hybrid_with_the_model_unavailable(self) -> None:
        for model in (None, FakeModel(error=LlmUnavailable("down"))):
            with self.subTest(model=model):
                answer = self.ask("average heart rate on the treadmill", "hybrid", model)
                self.assertEqual(answer.outcome, "ai_unavailable")
                self.assertEqual(answer.text, answers.AI_UNAVAILABLE_REPLY)
                self.assertIsNone(answer.result)
        # Simple questions still work.
        self.assertEqual(self.ask("average heart rate today", "hybrid", None).outcome, "query")

    def test_unsupported_from_either_parser(self) -> None:
        self.assertEqual(self.ask("what's the weather", "hybrid").text, answers.UNSUPPORTED_REPLY)
        model = FakeModel(form(intent="unsupported"))
        self.assertEqual(self.ask("tell me a joke", "llm", model).outcome, "unsupported")


class RangeOverrideTests(PipelineCase):
    def test_hybrid_uses_the_rules_range(self) -> None:
        # The rules parser defers ("treadmill") but saw exactly one range,
        # "today"; the model says last_week. Today wins, and it's recorded.
        model = FakeModel(form("avg_hr", None, "last_week"))
        answer = self.ask("average heart rate on the treadmill today", "hybrid", model)
        self.assertEqual(answer.request.range, "today")
        self.assertEqual((answer.parse.range_source, answer.parse.llm_range),
                         ("rules", "last_week"))
        self.assertEqual(answer.understood, "Average heart rate · All activities · today")

    def test_hybrid_fills_a_missing_ai_range(self) -> None:
        model = FakeModel(form("avg_hr", None, None))
        answer = self.ask("average heart rate on the treadmill today", "hybrid", model)
        self.assertEqual((answer.request.range, answer.parse.llm_range), ("today", None))
        self.assertEqual(answer.parse.range_source, "rules")

    def test_no_override_when_they_agree_or_without_a_single_range(self) -> None:
        agree = self.ask("average heart rate on the treadmill today", "hybrid",
                         FakeModel(form("avg_hr", None, "today")))
        self.assertEqual(agree.parse.range_source, "llm")
        # "the last week" is ambiguous, so there's no hint to override with.
        vague = self.ask("average bpm in the last week", "hybrid",
                         FakeModel(form("avg_hr", None, "last_7_days")))
        self.assertEqual((vague.parse.range_source, vague.request.range), ("llm", "last_7_days"))

    def test_llm_mode_never_overrides(self) -> None:
        model = FakeModel(form("avg_hr", None, "last_week"))
        answer = self.ask("average heart rate on the treadmill today", "llm", model)
        self.assertEqual((answer.request.range, answer.parse.range_source), ("last_week", "llm"))


class BadModelReplyTests(PipelineCase):
    def ask_hybrid(self, model: FakeModel) -> pipeline.Answer:
        with self.assertLogs("heartwatch.llm", logging.WARNING) as logs:
            answer = self.ask("average heart rate on the treadmill", "hybrid", model)
        self.logs = logs.output
        return answer

    def test_invalid_replies_get_the_didnt_understand_reply(self) -> None:
        for raw in ("not json", "[1, 2]", form("avg_hr", "Swimming", "today"),
                    json.dumps({"intent": "query", "metric": "avg_hr", "activity": None}),
                    form(None, None, "today")):
            with self.subTest(raw=raw):
                answer = self.ask_hybrid(FakeModel(raw))
                self.assertEqual(answer.outcome, "not_understood")
                self.assertEqual(answer.text, answers.NOT_UNDERSTOOD_REPLY)
                self.assertIsNone(answer.result)
                # The raw reply is logged.
                self.assertTrue(any(repr(raw) in line for line in self.logs))

    def test_a_request_that_fails_validation(self) -> None:
        with mock.patch.object(MetricRequest, "validate", side_effect=ValueError("nope")):
            answer = self.ask_hybrid(FakeModel(form("avg_hr", None, "today")))
        self.assertEqual(answer.outcome, "not_understood")
        self.assertIsNone(answer.result)

    def test_timeout_and_other_errors(self) -> None:
        answer = self.ask_hybrid(FakeModel(error=TimeoutError()))
        self.assertEqual((answer.outcome, answer.text), ("timeout", answers.NOT_UNDERSTOOD_REPLY))
        with self.assertLogs("heartwatch.llm", logging.ERROR):
            answer = self.ask("average heart rate on the treadmill", "hybrid",
                              FakeModel(error=RuntimeError("boom")))
        self.assertEqual(answer.outcome, "not_understood")


class ParseReplyTests(unittest.TestCase):
    def test_valid_form(self) -> None:
        result = llm_parser.parse_reply(form("time_in_activity", "Walking", "yesterday"))
        self.assertEqual(result.status, "accepted")
        self.assertEqual(result.request,
                         MetricRequest("time_in_activity", "Walking", "yesterday"))
        self.assertEqual((result.source, result.range_source), ("llm", "llm"))

    def test_unsupported_and_invalid(self) -> None:
        self.assertEqual(llm_parser.parse_reply(form(intent="unsupported")).status,
                         "unsupported")
        for raw in ("", "{", "null", json.dumps({**json.loads(form("avg_hr")), "extra": 1})):
            with self.subTest(raw=raw):
                self.assertEqual(llm_parser.parse_reply(raw).status, "invalid")


class AskModeTests(unittest.TestCase):
    def setUp(self) -> None:
        config._warned_ask_modes.clear()

    def test_default_and_explicit_modes(self) -> None:
        self.assertEqual(config.ask_mode({}), "hybrid")
        self.assertEqual(config.ask_mode({"HEARTWATCH_ASK_MODE": ""}), "hybrid")
        self.assertEqual(config.ask_mode({"HEARTWATCH_ASK_MODE": " Rules "}), "rules")
        self.assertEqual(config.ask_mode({"HEARTWATCH_ASK_MODE": "LLM"}), "llm")

    def test_unknown_value_warns_once(self) -> None:
        environ = {"HEARTWATCH_ASK_MODE": "gpt-please"}
        with self.assertLogs("heartwatch.llm", logging.WARNING) as logs:
            self.assertEqual(config.ask_mode(environ), "hybrid")
            self.assertEqual(config.ask_mode(environ), "hybrid")
        self.assertEqual(len(logs.records), 1)


class WorkerParseTests(unittest.TestCase):
    """LlmWorker._do_parse() run directly (no thread), with the Ollama
    client patched. Signals are connected directly, so they fire inline."""

    def setUp(self) -> None:
        from ..llm.worker import LlmWorker

        self.worker = LlmWorker()
        self.events: list = []
        self.worker.parse_finished.connect(lambda i, raw: self.events.append(("ok", i, raw)))
        self.worker.parse_failed.connect(lambda i, why: self.events.append(("failed", i, why)))
        self.worker.health_checked.connect(lambda s: self.events.append(("health", s.message)))

    def run_parse(self, chat) -> None:
        with mock.patch.object(llm_client._client, "chat", chat):
            asyncio.run(self.worker._do_parse(7, [{"role": "user", "content": "q"}], {"x": 1}))

    def test_success_forwards_format_and_options(self) -> None:
        response = mock.Mock()
        response.message.content = form("avg_hr")
        chat = mock.AsyncMock(return_value=response)
        self.run_parse(chat)
        self.assertEqual(self.events[-1], ("ok", 7, form("avg_hr")))
        kwargs = chat.call_args.kwargs
        self.assertEqual(kwargs["format"], {"x": 1})
        self.assertEqual(kwargs["options"], llm_client.ASK_OPTIONS)
        self.assertNotIn("keep_alive", kwargs)   # Ollama's default, never pinned

    def test_unreachable_is_unavailable(self) -> None:
        self.run_parse(mock.AsyncMock(side_effect=ConnectionError("refused")))
        self.assertEqual(self.events[-1], ("failed", 7, "unavailable"))
        self.assertIn("isn't reachable", self.events[0][1])

    def test_model_missing_is_unavailable(self) -> None:
        import ollama

        self.run_parse(mock.AsyncMock(side_effect=ollama.ResponseError("missing", 404)))
        self.assertEqual(self.events[-1], ("failed", 7, "unavailable"))
        self.assertIn("isn't pulled", self.events[0][1])

    def test_other_errors(self) -> None:
        self.run_parse(mock.AsyncMock(side_effect=RuntimeError("boom")))
        self.assertEqual(self.events, [("failed", 7, "error")])

    def test_summaries_still_send_no_format(self) -> None:
        # The summary path is unchanged: format stays Ollama's default (None).
        response = mock.Mock()
        response.message.content = "x"
        chat = mock.AsyncMock(return_value=response)
        with mock.patch.object(llm_client._client, "chat", chat):
            asyncio.run(llm_client.chat([], options=llm_client.SUMMARY_OPTIONS))
        self.assertIsNone(chat.call_args.kwargs["format"])
        self.assertEqual(chat.call_args.kwargs["options"], llm_client.SUMMARY_OPTIONS)
