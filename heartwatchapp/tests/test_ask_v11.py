"""Tests for the ask bar's v1.1 features (handoff 9a).

Plain-language overview, one class per item in the handoff's test table:
A typo correction (ask/typos.py); B the guard with typos; the evidence
check on the AI's form both ways (ask/evidence.py, A4); D the session
summary; E the activity breakdown; G ambiguous activity phrases; H every
suggestion answered by the keyword parser alone; I the replies for things
not built yet; J the developer log and marker setting. The model is always
a stand-in (no Ollama), and every database is a temporary one -- the real
database is never opened.

    python -m unittest heartwatchapp.tests.test_ask_v11
"""

from __future__ import annotations

import datetime
import json
import logging
import re
import tempfile
import unittest
from pathlib import Path

from .. import config
from ..ask import answers, devlog, evidence, guard, pipeline, rules, suggest, typos, vocab
from ..data import queries, stats
from ..scripts import demo_db, eval_ask
from .test_ask_pipeline import NOW_DT, FakeModel, PipelineCase, form
from .test_queries import DENVER, PLAIN_SEED0, FixtureDb, oct_midnight

DEV_V11 = eval_ask.APP_DIR / "eval" / "ask_questions_dev_v11.jsonl"


def intent_form(intent: str, range_name: str | None) -> str:
    return json.dumps({"intent": intent, "metric": None, "activity": None, "range": range_name})


class TypoTests(unittest.TestCase):
    """A. Clear typos corrected; known words untouched; short words
    untouched; two close readings ask."""

    def correct(self, question: str) -> typos.TypoResult:
        return typos.correct(question, guard.GUARD_WORDS)

    def test_clear_typos_are_corrected(self) -> None:
        cases = {
            "how long did i walkign yesterday": [("walkign", "walking")],
            "avrage heart rate this week": [("avrage", "average")],
            "peak heart rate yesterdy": [("yesterdy", "yesterday")],
            "how many sesions today": [("sesions", "sessions")],
            "average bpm while sittting": [("sittting", "sitting")],
            # Two typos, one of them a filler word.
            "waht was my max pulse while runing": [("waht", "what"), ("runing", "running")],
            # Four-letter swaps score only 0.75, but are one swap away.
            "total recorded tiem": [("tiem", "time")],
        }
        for question, corrections in cases.items():
            with self.subTest(question=question):
                self.assertEqual(list(self.correct(question).corrections), corrections)

    def test_a_neighbouring_phrase_settles_a_tie(self) -> None:
        # "lsat" is one deletion from "sat" (Sitting) and one swap from
        # "last"; only "last" makes a phrase with "week".
        result = self.correct("how long did I stand lsat week")
        self.assertEqual(result.corrections, (("lsat", "last"),))
        self.assertIsNone(result.ambiguous)

    def test_known_words_are_never_touched(self) -> None:
        for word in [*typos.TARGETS, *guard.GUARD_WORDS]:
            with self.subTest(word=word):
                self.assertEqual(self.correct(word).corrections, ())

    def test_short_words_are_never_guessed(self) -> None:
        for question in ("how long did I rn today", "avg hr wlk today", "peek hr"):
            with self.subTest(question=question):
                self.assertEqual(self.correct(question).corrections, ())

    def test_two_close_readings_are_ambiguous(self) -> None:
        # standing 0.875, sitting 0.800: closer than the 0.1 margin.
        result = self.correct("how long was I sitnding yesterday")
        self.assertEqual(result.corrections, ())
        self.assertEqual(result.ambiguous, ("sitnding", ("standing", "sitting")))
        self.assertEqual(result.replace("sitnding", "standing"),
                         "how long was i standing yesterday")

    def test_same_meaning_isnt_ambiguous(self) -> None:
        # sessions 0.933, session 0.857: one meaning, so it's corrected.
        self.assertEqual(self.correct("sesions").corrections, (("sesions", "sessions"),))

    def test_unknown_words_in_the_dev_questions_are_left_alone(self) -> None:
        # The v1 dev questions have no typos, so nothing may change.
        for q in eval_ask.load_questions(eval_ask.DEFAULT_QUESTIONS):
            with self.subTest(id=q["id"]):
                self.assertEqual(self.correct(q["question"]).corrections, ())

    def test_the_v11_questions_get_exactly_their_expected_corrections(self) -> None:
        for q in eval_ask.load_questions(DEV_V11):
            if "corrections" in q["expected"]:
                with self.subTest(id=q["id"]):
                    got = [list(c) for c in self.correct(q["question"]).corrections]
                    self.assertEqual(got, q["expected"]["corrections"])


class GuardTypoTests(unittest.TestCase):
    """B. Medical questions with typos are refused; "heart" is never
    "health"; nothing supported is newly refused (tests/test_ask_vocab.py
    checks both dev files)."""

    def check(self, question: str) -> str | None:
        return guard.check_question(question, typos.correct(question, guard.GUARD_WORDS).text)

    def test_misspelled_health_words_are_refused(self) -> None:
        for question, term in (
            ("is my hart rate helthy", "healthy"),
            ("is my heart rate normall", "normal"),
            ("shuold i run today", "should i"),
            ("was that dangerus", "dangerous"),
            ("my pulse is to hihg", None),   # "to" isn't "too": not refused
        ):
            with self.subTest(question=question):
                self.assertEqual(self.check(question), term)

    def test_heart_is_never_health(self) -> None:
        self.assertEqual(guard.guard_correct("heart rate"), "heart rate")
        self.assertIsNone(self.check("average heart rate this week"))
        # A typo of a supported word that is as close to a guard word goes
        # to the guard only on a tie or better; "hart" is closer to heart.
        self.assertIsNone(self.check("average hart rate today"))

    def test_no_supported_phrase_or_suggestion_is_refused(self) -> None:
        for phrase in [*vocab.all_synonym_phrases(), *suggest.every_question()]:
            with self.subTest(phrase=phrase):
                self.assertIsNone(self.check(phrase))


class EvidenceTests(PipelineCase):
    """A4: the range evidence table, both ways, and the activity check."""

    def test_paraphrases_count_as_evidence(self) -> None:
        for text, range_name in (
            ("max bpm in my newest recording", "latest_session"),
            ("avg pulse for the current week", "this_week"),
            ("mean bpm for the week before this one", "last_week"),
            ("average heart rate last week", "last_week"),
        ):
            with self.subTest(text=text):
                self.assertEqual(evidence.find(text).ranges, {range_name})

    def test_no_range_named_means_no_evidence(self) -> None:
        for text in ("and walking?", "average heart rate on the treadmill", "this month"):
            with self.subTest(text=text):
                self.assertEqual(evidence.find(text).ranges, frozenset())

    def test_a_vague_week_allows_both_weeks(self) -> None:
        self.assertEqual(evidence.find("average bpm in the last week").ranges,
                         {"last_week", "last_7_days"})

    def test_a_backed_range_is_kept_and_an_unbacked_one_dropped(self) -> None:
        kept = self.ask("avg pulse for the current week on the treadmill", "llm",
                        FakeModel(form("avg_hr", None, "this_week")))
        self.assertEqual((kept.request.range, kept.parse.range_source, kept.parse.dropped),
                         ("this_week", "llm", ()))
        dropped = self.ask("and walking?", "llm",
                           FakeModel(form("time_in_activity", "Walking", "yesterday")))
        self.assertEqual(dropped.request.range, "last_7_days")
        self.assertIn("(default)", dropped.understood)
        self.assertEqual(dropped.parse.dropped, ("range yesterday",))

    def test_an_unbacked_activity_asks_which(self) -> None:
        answer = self.ask("how long on the treadmill today", "hybrid",
                          FakeModel(form("time_in_activity", "Running", "today")))
        self.assertEqual(answer.outcome, "needs_activity")
        self.assertEqual(answer.parse.dropped, ("activity Running",))
        self.assertEqual([o.label for o in answer.suggestions], list(vocab.ACTIVITIES))
        self.assertIsNone(answer.result)

    def test_a_synonym_backs_its_activity(self) -> None:
        answer = self.ask("how long did i jog on the treadmill today", "hybrid",
                          FakeModel(form("time_in_activity", "Running", "today")))
        self.assertEqual((answer.request.activity, answer.parse.dropped), ("Running", ()))

    def test_an_intent_needs_a_summary_or_breakdown_phrase(self) -> None:
        # Dev q09, found by the v1.1 eval: the model chose the breakdown
        # for a time-in-walking question (its form still named the metric
        # and the activity), so hybrid answered with a breakdown.
        model = FakeModel(json.dumps({"intent": "activity_breakdown",
                                      "metric": "time_in_activity", "activity": "Walking",
                                      "range": "yesterday"}))
        answer = self.ask("How much of yesterday did I spend on my feet walking around?",
                          "hybrid", model)
        self.assertEqual((answer.outcome, answer.request),
                         ("query", queries.MetricRequest("time_in_activity", "Walking",
                                                         "yesterday")))
        self.assertIn("intent activity_breakdown", answer.parse.dropped)
        # No metric to fall back to (dev q24): unsupported, never a number.
        answer = self.ask("Start recording a new session", "llm",
                          FakeModel(intent_form("session_summary", None)))
        self.assertEqual(answer.outcome, "unsupported")
        self.assertIsNone(answer.result)

    def test_a_summary_needs_a_single_session_phrase(self) -> None:
        answer = self.ask("give me an overview of this week please", "llm",
                          FakeModel(intent_form("session_summary", "this_week")))
        self.assertEqual(answer.outcome, "activity_breakdown")
        self.assertEqual(answer.parse.intent_range, "this_week")
        self.assertIn("intent session_summary", answer.parse.dropped)


class SummaryTests(unittest.TestCase):
    """D. The reply contains format_stats_plain() for the latest finished
    session, byte for byte; live sessions are skipped; an empty database
    gets the no-sessions reply."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def ask(self, question: str, path: Path, mode: str = "rules") -> pipeline.Answer:
        return pipeline.answer_question(question, conn_or_path=path, mode=mode,
                                        now=datetime.datetime.now(DENVER))

    def test_the_sessions_views_summary_unchanged(self) -> None:
        path, session_id = demo_db.build_demo_db(self.dir / "demo.db", data_seed=0)
        plain = stats.format_stats_plain(stats.compute_session_stats(session_id, path))
        self.assertEqual(plain, PLAIN_SEED0)
        for question in ("tell me about my last session", "How did my latest workout go?",
                         "summarize my most recent recording", "give me a recap of my last run"):
            with self.subTest(question=question):
                answer = self.ask(question, path)
                self.assertEqual(answer.outcome, "session_summary")
                self.assertIn(PLAIN_SEED0, answer.text)
                # The demo session has unsure windows, so the caveat follows.
                self.assertTrue(answer.text.startswith("Your latest session started at "))
                self.assertTrue(answer.text.endswith(
                    PLAIN_SEED0 + " Some windows had low classification confidence."))
                # "my last run" still summarises the latest session (B5).
                self.assertEqual(answer.understood, "Session summary · latest session")
                self.assertEqual(answer.caption, queries.SIMULATED_CAPTION)

    def test_a_live_session_is_skipped(self) -> None:
        fixture = FixtureDb(self)
        done = fixture.add(oct_midnight(8) + 3_600_000, oct_midnight(8) + 3_610_000,
                           hr=[(0, 90)], predictions=[(5000, "Walking")])
        fixture.add(oct_midnight(8) + 7_200_000, None, hr=[(0, 120)])  # still recording
        answer = pipeline.answer_question("tell me about my last session",
                                          conn_or_path=fixture.path, mode="rules", now=NOW_DT)
        plain = stats.format_stats_plain(stats.compute_session_stats(done, fixture.path))
        self.assertIn(plain, answer.text)
        # 01:00 MDT on Thursday Oct 8.
        self.assertTrue(answer.text.startswith(
            "Your latest session started at 1:00 AM on Thursday, October 8."))

    def test_no_finished_session(self) -> None:
        fixture = FixtureDb(self)
        answer = pipeline.answer_question("tell me about my last session",
                                          conn_or_path=fixture.path, mode="rules", now=NOW_DT)
        self.assertEqual(answer.text, answers.NO_SESSIONS_REPLY)
        self.assertEqual(len(answer.suggestions), 3)

    def test_the_eval_databases_demo_session_is_pinned(self) -> None:
        now = eval_ask.parse_now(eval_ask.DEFAULT_NOW, eval_ask.DEFAULT_TZ)
        path = eval_ask.build_eval_db(self.dir / "eval.db", now, 0)
        answer = pipeline.answer_question("tell me about my last session", conn_or_path=path,
                                          mode="rules", now=now)
        # Pinned now (Wed Oct 14 15:00 MDT) + 1 hour.
        self.assertTrue(answer.text.startswith(
            "Your latest session started at 4:00 PM on Wednesday, October 14."))


class BreakdownTests(unittest.TestCase):
    """E. The same numbers and order as the Dashboard's ActivityBreakdown,
    with its percentages."""

    def test_same_as_the_dashboards_breakdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path, _ = demo_db.build_demo_db(Path(tmp) / "demo.db", data_seed=0)
            answer = pipeline.answer_question("give me an activity breakdown",
                                              conn_or_path=path, mode="rules")
            dashboard = queries.activity_breakdown("last_7_days", path=path)
        self.assertEqual(answer.outcome, "activity_breakdown")
        self.assertEqual(answer.breakdown, dashboard)
        self.assertEqual(answer.understood, "Activity breakdown · last 7 days (default)")
        parts = answer.text.split(": ", 1)[1].rstrip(".").split(", ")
        self.assertEqual(
            [re.fullmatch(r"(\w+) .*\((\d+)%\)", part).groups() for part in parts],
            [(label.lower(), str(p)) for (label, _), p in
             zip(dashboard.items, dashboard.percentages())],
        )
        self.assertEqual(answer.caption, queries.SIMULATED_CAPTION)

    def test_ranges_and_the_summary_fallback(self) -> None:
        for question, range_name, fallback in (
            ("What did I do today?", "today", False),
            ("breakdown of my activities this week", "this_week", False),
            ("summarize my activity in the last 30 days", "last_30_days", True),
            ("what did I do in my latest session", "latest_session", False),
        ):
            with self.subTest(question=question):
                parse = rules.parse(question)
                self.assertEqual((parse.status, parse.intent, parse.intent_range,
                                  parse.summary_fallback),
                                 ("accepted", "activity_breakdown", range_name, fallback))


class AmbiguousActivityTests(PipelineCase):
    """G. "on my feet" asks Walking or Standing; each button's question is
    answered by the keyword parser alone."""

    def test_buttons_and_their_questions(self) -> None:
        for question in ("How long was I on my feet today?",
                         "average heart rate while on my feet this week"):
            with self.subTest(question=question):
                answer = self.ask(question)
                self.assertEqual(answer.outcome, "clarify")
                self.assertEqual(answer.text, "Did you mean walking or standing?")
                self.assertEqual([o.label for o in answer.suggestions], ["Walking", "Standing"])
                for option in answer.suggestions:
                    parsed = rules.parse(option.question)
                    self.assertEqual((parsed.status, parsed.request.activity),
                                     ("accepted", option.label))

    def test_a_named_activity_wins(self) -> None:
        parsed = rules.parse("how long was i on my feet walking today")
        self.assertEqual((parsed.status, parsed.request.activity), ("accepted", "Walking"))

    def test_the_ai_picking_one_still_asks(self) -> None:
        answer = self.ask("how long was i on my feet at the gym today", "hybrid",
                          FakeModel(form("time_in_activity", "Walking", "today")))
        self.assertEqual(answer.outcome, "clarify")
        self.assertEqual([o.label for o in answer.suggestions], ["Walking", "Standing"])

    def test_an_ambiguous_typo_asks_too(self) -> None:
        answer = self.ask("how long was I sitnding yesterday")
        self.assertEqual((answer.outcome, answer.text),
                         ("clarify", "Did you mean standing or sitting?"))
        for option in answer.suggestions:
            self.assertEqual(rules.parse(option.question).status, "accepted")


class SuggestionTests(PipelineCase):
    """H. Every suggestion the code can produce is answered by the keyword
    parser alone, with a valid request or intent."""

    def assert_rules_answer(self, question: str) -> None:
        parsed = rules.parse(question)
        self.assertEqual(parsed.status, "accepted", f"{question!r}: {parsed.reason}")
        if parsed.request is not None:
            normalized = pipeline.normalize(parsed.request)
            self.assertEqual(normalized.outcome, "query", question)
            normalized.request.validate()

    def test_every_producible_question(self) -> None:
        questions = suggest.every_question()
        self.assertGreater(len(questions), 100)
        for question in questions:
            with self.subTest(question=question):
                self.assert_rules_answer(question)

    def test_every_suggestion_shown_for_the_dev_questions(self) -> None:
        questions = [q["question"] for q in eval_ask.load_questions(eval_ask.DEFAULT_QUESTIONS)]
        questions += [q["question"] for q in eval_ask.load_questions(DEV_V11)]
        for mode in ("rules", "hybrid"):
            for question in questions:
                answer = self.ask(question, mode, None)
                # Suggestions: up to three. "Which activity?" offers all four.
                limit = 4 if answer.outcome == "needs_activity" else 3
                self.assertLessEqual(len(answer.suggestions), limit, question)
                for option in answer.suggestions:
                    with self.subTest(mode=mode, question=question, suggestion=option.question):
                        self.assert_rules_answer(option.question)

    def test_chosen_from_what_was_understood(self) -> None:
        self.assertEqual([o.question for o in suggest.from_detected(("Walking",), ("today",))],
                         ["how long did I walk today", "average heart rate while walking today",
                          "peak heart rate while walking today"])
        self.assertEqual([o.question for o in suggest.from_detected((), ("yesterday",))],
                         ["average heart rate yesterday", "what did I do yesterday",
                          "how many sessions yesterday"])
        self.assertEqual([o.question for o in suggest.from_detected()],
                         list(suggest.DEFAULT_QUESTIONS))


class NotBuiltTests(PipelineCase):
    """I. Comparisons, follow-ups and the longest session get their own
    replies, never a number."""

    def test_specific_replies(self) -> None:
        cases = [
            ("compare this week to last week", "comparison", answers.COMPARISON_REPLY),
            ("my average heart rate this week versus last week", "comparison",
             answers.COMPARISON_REPLY),
            ("which session was my longest", "longest_session", answers.LONGEST_REPLY),
            ("and walking?", "follow_up",
             answers.follow_up_reply("how long did I walk this week")),
            ("what about yesterday?", "follow_up",
             answers.follow_up_reply("average heart rate yesterday")),
        ]
        for question, kind, text in cases:
            for mode in ("rules", "hybrid"):
                with self.subTest(question=question, mode=mode):
                    model = FakeModel(form("avg_hr", None, "today"))
                    answer = self.ask(question, mode, model)
                    self.assertEqual((answer.outcome, answer.parse.kind, answer.text),
                                     ("not_built", kind, text))
                    self.assertIsNone(answer.result)
                    self.assertEqual(model.calls, [])

    def test_a_comparison_suggests_each_period(self) -> None:
        answer = self.ask("my average heart rate this week versus last week")
        self.assertEqual([o.question for o in answer.suggestions],
                         ["average heart rate this week", "average heart rate last week"])


class DevToolTests(PipelineCase):
    """J (the parts that need no window; the bar's are in test_ai_bar.py)."""

    def setUp(self) -> None:
        super().setUp()
        config._warned_ask_dev.clear()

    def test_setting(self) -> None:
        self.assertFalse(config.ask_dev({}))
        for value in ("1", "true", " ON ", "yes"):
            self.assertTrue(config.ask_dev({"HEARTWATCH_ASK_DEV": value}), value)
        for value in ("0", "false", "", "off"):
            self.assertFalse(config.ask_dev({"HEARTWATCH_ASK_DEV": value}), value)
        with self.assertLogs("heartwatch.llm", logging.WARNING) as logs:
            self.assertFalse(config.ask_dev({"HEARTWATCH_ASK_DEV": "maybe"}))
            self.assertFalse(config.ask_dev({"HEARTWATCH_ASK_DEV": "maybe"}))
        self.assertEqual(len(logs.records), 1)

    def test_marker(self) -> None:
        self.assertEqual(devlog.marker(self.ask("average heart rate today")), "rules")
        self.assertEqual(devlog.marker(self.ask("avrage heart rate today")), "rules (corrected)")
        self.assertEqual(devlog.marker(self.ask("is my heart rate healthy")), "guard")
        ai = self.ask("average heart rate on the treadmill today", "hybrid",
                      FakeModel(form("avg_hr", None, "today")))
        self.assertEqual(devlog.with_marker(ai.understood, ai),
                         "Average heart rate · All activities · today · AI")

    def test_entry_has_the_fields_and_no_numbers_from_the_reply(self) -> None:
        question = "avrage heart rate today"
        answer = self.ask(question)
        self.assertEqual(answer.text, "Your heart rate averaged 95 bpm today, in one session.")
        entry = devlog.entry(question, answer)
        self.assertEqual(set(entry), {"timestamp", "question", "corrected", "corrections",
                                      "decided_by", "outcome", "form", "dropped", "latency_ms"})
        self.assertEqual(entry["corrected"], "average heart rate today")
        self.assertEqual(entry["form"], {"metric": "avg_hr", "activity": None, "range": "today"})
        logged = json.dumps({k: v for k, v in entry.items() if k != "latency_ms"})
        self.assertNotIn("95", logged)
        self.assertNotIn(answer.text, logged)


if __name__ == "__main__":
    unittest.main()
