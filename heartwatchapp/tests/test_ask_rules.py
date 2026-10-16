"""Tests for the keyword parser (ask/rules.py).

Plain-language overview: the parser may only accept a question when it's
sure -- one metric, at most one activity, at most one range, and no words
it doesn't know. These tests check what it accepts, that it defers on
every kind of ambiguity, and what it calls unsupported. Stdlib unittest:

    python -m unittest heartwatchapp.tests.test_ask_rules
"""

from __future__ import annotations

import unittest

from ..ask import rules
from ..ask.answers import RULES_EXAMPLES
from ..data.queries import MetricRequest


class AcceptTests(unittest.TestCase):
    def assert_accepts(self, question: str, metric: str, activity: str | None,
                       range_name: str | None) -> None:
        result = rules.parse(question)
        self.assertEqual(result.status, "accepted", f"{question!r}: {result.reason}")
        self.assertEqual(result.request, MetricRequest(metric, activity, range_name))
        self.assertEqual(result.source, "rules")
        self.assertEqual(result.range_source, "rules" if range_name else None)

    def test_simple_questions(self) -> None:
        cases = [
            ("What was my average heart rate while running this week?",
             "avg_hr", "Running", "this_week"),
            ("Peak heart rate last week", "peak_hr", None, "last_week"),
            ("How long did I walk yesterday?", "time_in_activity", "Walking", "yesterday"),
            ("How many sessions did I record in the last 30 days?",
             "session_count", None, "last_30_days"),
            ("Total recorded time today", "total_time", None, "today"),
            ("highest pulse ever", "peak_hr", None, "all"),
            ("avg bpm while seated", "avg_hr", "Sitting", None),
        ]
        for question, metric, activity, range_name in cases:
            with self.subTest(question=question):
                self.assert_accepts(question, metric, activity, range_name)

    def test_since_the_start_of_the_week(self) -> None:
        # "start" here is part of a this_week phrase, not a command.
        self.assert_accepts("average heart rate since the start of the week",
                            "avg_hr", None, "this_week")

    def test_a_longer_phrase_wins(self) -> None:
        # "how long did I record" is total_time, not "how long" + activity time.
        self.assert_accepts("how long did I record today", "total_time", None, "today")
        # "my last run": the latest session, and the run still counts as Running.
        self.assert_accepts("what was my max HR during my last run", "peak_hr", "Running",
                            "latest_session")
        # "session" inside "my most recent session" is the range, not a
        # sessions word, so it doesn't clash with the activity.
        self.assert_accepts("average heart rate while walking in my most recent session",
                            "avg_hr", "Walking", "latest_session")

    def test_total_time_with_an_activity_is_left_for_normalize(self) -> None:
        # The parser reports what was asked; pipeline.normalize() turns it
        # into time_in_activity.
        self.assert_accepts("total time walking", "total_time", "Walking", None)

    def test_time_without_an_activity_is_accepted_for_normalize(self) -> None:
        # normalize() then asks which activity -- no number is shown.
        self.assert_accepts("how much time did I spend yesterday", "time_in_activity", None,
                            "yesterday")

    def test_session_count_with_an_activity(self) -> None:
        # Explicit session-count phrases with an activity are accepted as
        # what they are; normalize() then gives the fixed reply.
        for question, range_name in (
            ("How many walking sessions did I have last week?", "last_week"),
            ("how many sessions did I run", None),
            ("number of jogging recordings this week", "this_week"),
            ("how many sessions did I record while standing yesterday", "yesterday"),
        ):
            with self.subTest(question=question):
                result = rules.parse(question)
                self.assertEqual(result.status, "accepted", result.reason)
                self.assertEqual((result.request.metric, result.request.range),
                                 ("session_count", range_name))
                self.assertIsNotNone(result.request.activity)

    def test_the_fallback_examples_are_accepted(self) -> None:
        # The "AI isn't running" reply offers these, so they must work
        # without the model.
        for question in RULES_EXAMPLES:
            with self.subTest(question=question):
                self.assertEqual(rules.parse(question).status, "accepted")


class DeferTests(unittest.TestCase):
    def assert_defers(self, question: str, reason: str, range_hint: str | None = None) -> None:
        result = rules.parse(question)
        self.assertEqual(result.status, "deferred", f"{question!r}: {result.reason}")
        self.assertIsNone(result.request)
        self.assertIn(reason, result.reason)
        self.assertEqual(result.range_hint, range_hint)

    def test_two_activities(self) -> None:
        self.assert_defers("average heart rate running or walking this week",
                           "unrecognized words: or", "this_week")
        self.assert_defers("average heart rate running walking", "several activities")

    def test_two_ranges(self) -> None:
        self.assert_defers("peak heart rate today yesterday", "several ranges")

    def test_two_metrics(self) -> None:
        self.assert_defers("average peak heart rate today", "several metrics", "today")

    def test_activity_with_a_bare_sessions_word(self) -> None:
        # "sessions" without a session-count phrase: still deferred.
        self.assert_defers("how many minutes did I walk across my sessions",
                           "an activity together with a sessions word")
        self.assert_defers("average heart rate while running in my sessions this week",
                           "an activity together with a sessions word", "this_week")

    def test_unknown_words(self) -> None:
        self.assert_defers("what was my average heart rate on the treadmill",
                           "unrecognized words: treadmill")

    def test_ambiguous_time_phrases(self) -> None:
        # "the last week" could be last_week or last_7_days; "last 3 days"
        # has no range; "recently" is vague. None gives a range hint.
        for question in ("average bpm in the last week", "average heart rate last 3 days",
                         "peak heart rate recently", "average heart rate on monday"):
            with self.subTest(question=question):
                self.assert_defers(question, "no single matching range")

    def test_heart_rate_without_average_or_peak(self) -> None:
        self.assert_defers("what was my heart rate yesterday", "no metric", "yesterday")
        self.assert_defers("how long was my heart rate high", "unrecognized words: high")

    def test_average_without_heart_rate(self) -> None:
        self.assert_defers("what was my max yesterday", "no metric", "yesterday")


class UnsupportedTests(unittest.TestCase):
    def test_no_metric_or_activity_words(self) -> None:
        for question in ("what's the weather", "hello", "what about yesterday?"):
            with self.subTest(question=question):
                result = rules.parse(question)
                self.assertEqual(result.status, "unsupported")
                self.assertIsNone(result.request)

    def test_things_v1_cant_do(self) -> None:
        # Charts, commands and filters v1 can't apply are refused outright,
        # even next to metric words, instead of being passed to the AI.
        for question in ("show me a chart of my heart rate this week",
                         "start recording a new session", "how many steps did I walk",
                         "average heart rate on my 5 km run",
                         "peak heart rate over the last 3 miles"):
            with self.subTest(question=question):
                self.assertEqual(rules.parse(question).status, "unsupported")

    def test_start_stop_pause_only_as_commands(self) -> None:
        # Command phrases, or the question's first word: unsupported.
        for question in ("Start recording", "please stop recording", "pause the session",
                         "can you start a session", "Start tracking my run",
                         "stop", "pause my heart rate"):
            with self.subTest(question=question):
                self.assertEqual(rules.parse(question).status, "unsupported")
        # Anywhere else they're ordinary (unknown) words: deferred, not refused.
        for question in ("average heart rate from the start of my last run",
                         "peak heart rate before I had to stop yesterday",
                         "how long did I walk without a pause today"):
            with self.subTest(question=question):
                self.assertEqual(rules.parse(question).status, "deferred")

    def test_pace_and_speed_are_not_refused(self) -> None:
        # They describe an activity, not a filter v1 can't apply; the
        # parser doesn't know them, so it defers to the AI.
        for question in ("average heart rate at a walking pace",
                         "peak heart rate at running speed"):
            with self.subTest(question=question):
                result = rules.parse(question)
                self.assertEqual(result.status, "deferred")
                self.assertRegex(result.reason, "unrecognized words: .*(pace|speed)")
