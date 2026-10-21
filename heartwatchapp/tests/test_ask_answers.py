"""Tests for the AI bar's wording (ask/answers.py).

Plain-language overview: every sentence the bar shows is a Python
template. These tests pin each template's exact text for hand-made
results: the five metrics, every range phrase, empty data (said in words,
never a 0), a real zero (worded differently from no data), "in one
session" vs "across 2 sessions", the "what I understood" line, the
simulated-predictions caption, the breakdown and summary replies, and the
fixed replies. They also render every metric x activity x range x (value,
zero, no data) combination and check that nothing is left unfilled, that
every number in a reply comes from the result through the formatting
helpers, and that no reply contains a blocked or judging word. No database
needed.

    python -m unittest heartwatchapp.tests.test_ask_answers
"""

from __future__ import annotations

import re
import unittest

from ..ask import answers, vocab
from ..data.queries import (
    ACTIVITY_METRICS,
    SIMULATED_CAPTION,
    ActivityBreakdown,
    MetricRequest,
    MetricResult,
)
from ..data.stats import format_duration_words
from ..llm.validate import _BLOCKED_RE
from .test_queries import DENVER, utc_ms

# Handoff 9a, 3.3: no judgments in a reply.
JUDGMENT_RE = re.compile(r"\b(?:only|great|a lot|high|low)\b", re.IGNORECASE)


def result(value, unit="bpm", n=1, empty=False, simulated=False) -> MetricResult:
    return MetricResult(value, unit, n, empty, simulated)


def digits(text: str) -> set[str]:
    return set(re.findall(r"\d+", text))


class ReplyTextTests(unittest.TestCase):
    def test_each_metric(self) -> None:
        cases = [
            # 127.6 rounds to 128, the same as the Dashboard's cards (.0f).
            (MetricRequest("avg_hr", "Running", "this_week"), result(127.6, n=3),
             "Your heart rate averaged 128 bpm while running this week, across 3 sessions."),
            (MetricRequest("peak_hr", None, "last_7_days"), result(134),
             "Your heart rate peaked at 134 bpm in the last 7 days, in one session."),
            # 140 s = 2 minutes 20 seconds; Walking reads as "you walked".
            (MetricRequest("time_in_activity", "Walking", "yesterday"),
             result(140.0, "seconds"),
             "You walked for 2 minutes 20 seconds yesterday, in one session."),
            (MetricRequest("session_count", None, "last_30_days"), result(4, "sessions", n=4),
             "You recorded 4 sessions in the last 30 days."),
            # 1085 s = 18 minutes 5 seconds.
            (MetricRequest("total_time", None, "this_week"), result(1085.0, "seconds", n=4),
             "You recorded 18 minutes 5 seconds in total this week, across 4 sessions."),
        ]
        for request, res, text in cases:
            with self.subTest(metric=request.metric):
                self.assertEqual(answers.reply_text(request, res), text)

    def test_every_range_phrase(self) -> None:
        expected = {
            "today": "Your heart rate peaked at 120 bpm today, in one session.",
            "yesterday": "Your heart rate peaked at 120 bpm yesterday, in one session.",
            "this_week": "Your heart rate peaked at 120 bpm this week, in one session.",
            "last_week": "Your heart rate peaked at 120 bpm last week, in one session.",
            "last_7_days": "Your heart rate peaked at 120 bpm in the last 7 days, in one session.",
            "last_30_days":
                "Your heart rate peaked at 120 bpm in the last 30 days, in one session.",
            "all": "Your heart rate peaked at 120 bpm in your one recorded session.",
            # The latest session is one session: no count.
            "latest_session": "In your latest session, your heart rate peaked at 120 bpm.",
        }
        self.assertEqual(set(answers.RANGE_PHRASES), set(vocab.RANGES))
        self.assertEqual(set(answers.RANGE_LABELS), set(vocab.RANGES))
        self.assertEqual(set(expected), set(vocab.RANGES))
        for range_name, text in expected.items():
            with self.subTest(range=range_name):
                self.assertEqual(
                    answers.reply_text(MetricRequest("peak_hr", None, range_name), result(120)),
                    text,
                )

    def test_session_counts_in_words(self) -> None:
        one = answers.reply_text(MetricRequest("session_count", None, "today"),
                                 result(1, "sessions"))
        self.assertEqual(one, "You recorded 1 session today.")
        two = answers.reply_text(MetricRequest("avg_hr", None, "today"), result(90, n=2))
        self.assertEqual(two, "Your heart rate averaged 90 bpm today, across 2 sessions.")
        # "in total" rather than "across all your sessions" after "sessions".
        total = answers.reply_text(MetricRequest("session_count", None, "all"),
                                   result(7, "sessions", n=7))
        self.assertEqual(total, "You recorded 7 sessions in total.")
        everything = answers.reply_text(MetricRequest("total_time", None, "all"),
                                        result(600.0, "seconds", n=3))
        self.assertEqual(everything, "You recorded 10 minutes in total across all 3 of your "
                                     "sessions.")

    def test_every_activity_reads_as_a_verb(self) -> None:
        for activity, verb in (("Sitting", "sat"), ("Walking", "walked"),
                               ("Standing", "stood"), ("Running", "ran")):
            with self.subTest(activity=activity):
                text = answers.reply_text(MetricRequest("time_in_activity", activity, "today"),
                                          result(60.0, "seconds"))
                self.assertEqual(text, f"You {verb} for 1 minute today, in one session.")
        latest = answers.reply_text(MetricRequest("time_in_activity", "Running", "latest_session"),
                                    result(45.0, "seconds"))
        self.assertEqual(latest, "In your latest session, you ran for 45 seconds.")

    def test_a_real_zero_is_worded_differently_from_no_data(self) -> None:
        # Activity data exists but none of it was Running: a real 0, which
        # no session contributed to.
        zero = answers.reply_text(MetricRequest("time_in_activity", "Running", "today"),
                                  result(0.0, "seconds", n=0))
        self.assertEqual(zero, "You didn't spend any time running today.")
        empty = result(None, n=0, empty=True)
        cases = [
            (MetricRequest("avg_hr", "Running", "this_week"),
             "You didn't record any heart-rate data while running this week."),
            (MetricRequest("peak_hr", None, "today"), "You didn't record any heart-rate data today."),
            (MetricRequest("peak_hr", None, "latest_session"),
             "There's no heart-rate data in your latest session."),
            (MetricRequest("time_in_activity", "Walking", "yesterday"),
             "You didn't record any activity yesterday."),
            (MetricRequest("session_count", None, "yesterday"),
             "You didn't record any sessions yesterday."),
            (MetricRequest("total_time", None, "last_week"), "You didn't record any sessions last week."),
            (MetricRequest("session_count", None, "all"), "You haven't recorded any sessions yet."),
            (MetricRequest("total_time", None, "latest_session"),
             "You haven't recorded any sessions yet."),
        ]
        for request, text in cases:
            with self.subTest(request=request):
                reply = answers.reply_text(request, empty)
                self.assertEqual(reply, text)
                self.assertNotIn("0", reply)


class TemplateMatrixTests(unittest.TestCase):
    """Handoff 9a, 3.3: every metric x activity x range x (value, zero,
    no data), with one and with three sessions."""

    def combinations(self):
        for metric in vocab.METRICS:
            activities = (None, *vocab.ACTIVITIES) if metric in ACTIVITY_METRICS else (None,)
            for activity in activities:
                if metric == "time_in_activity" and activity is None:
                    continue    # normalize() asks which activity first
                for range_name in vocab.RANGES:
                    request = MetricRequest(metric, activity, range_name)
                    unit = "bpm" if metric in ("avg_hr", "peak_hr") else (
                        "sessions" if metric == "session_count" else "seconds")
                    values = {"avg_hr": 95.4, "peak_hr": 134, "session_count": 3,
                              "time_in_activity": 3725.0, "total_time": 340.0}
                    for n in (1, 3):
                        yield request, result(values[metric], unit, n=n)
                    if metric == "time_in_activity":
                        yield request, result(0.0, unit, n=0)
                    yield request, result(None, unit, n=0, empty=True)

    def allowed_digits(self, request: MetricRequest, res: MetricResult) -> set[str]:
        """Digits the formatting helpers can produce from this result, plus
        those in the range's own name ("the last 7 days")."""
        named = digits(answers.RANGE_PHRASES[request.range])
        if res.empty:
            return named
        if res.unit == "bpm":
            value = f"{res.value:.0f}"
        elif res.unit == "seconds":
            value = format_duration_words(res.value)
        else:
            value = str(res.value)
        return digits(value) | {str(res.n_sessions)} | named

    def test_every_combination_renders_cleanly(self) -> None:
        count = 0
        for request, res in self.combinations():
            count += 1
            with self.subTest(request=request, result=res):
                reply = answers.reply_text(request, res)
                self.assertNotRegex(reply, r"[{}]")
                self.assertNotRegex(reply, r"\b1 (?:sessions|minutes|seconds|hours)\b")
                self.assertNotRegex(reply, r"\b(?:[02-9]|\d{2,}) (?:session|minute|second|hour)\b")
                self.assertNotIn("None", reply)
                self.assertTrue(reply.endswith("."), reply)
                self.assertLessEqual(digits(reply), self.allowed_digits(request, res), reply)
                self.assertIsNone(_BLOCKED_RE.search(reply), reply)
                self.assertIsNone(JUDGMENT_RE.search(reply), reply)
                self.assertNotRegex(reply, r"\b(?:Sitting|Walking|Standing|Running)\b")
        self.assertGreater(count, 200)

    def test_fixed_replies_have_no_blocked_or_judging_words(self) -> None:
        for text in (answers.UNSUPPORTED_REPLY, answers.NOT_UNDERSTOOD_REPLY,
                     answers.SESSION_COUNT_BY_ACTIVITY_REPLY, answers.AI_UNAVAILABLE_REPLY,
                     answers.NO_SESSIONS_REPLY, answers.COMPARISON_REPLY, answers.LONGEST_REPLY,
                     answers.needs_activity_reply(), answers.MEDICAL_REPLY,
                     answers.follow_up_reply("how long did I walk this week")):
            with self.subTest(text=text):
                self.assertIsNone(_BLOCKED_RE.search(text))
                self.assertIsNone(JUDGMENT_RE.search(text))


class BreakdownTextTests(unittest.TestCase):
    # The seed-0 demo session's own split (tests/test_queries.py PLAIN_SEED0).
    DEMO = ActivityBreakdown(
        (("Walking", 140.0), ("Standing", 85.0), ("Sitting", 70.0), ("Running", 45.0)),
        simulated=True, n_sessions=1,
    )

    def test_every_activity_with_its_share(self) -> None:
        # 340 s in all = 5 minutes 40 seconds. Shares, each rounded on its
        # own: 140/340 = 41.2 -> 41, 85/340 = 25.0 -> 25, 70/340 = 20.6 ->
        # 21, 45/340 = 13.2 -> 13.
        self.assertEqual(self.DEMO.percentages(), (41, 25, 21, 13))
        self.assertEqual(
            answers.breakdown_text("last_7_days", self.DEMO),
            "Over the last 7 days you recorded 5 minutes 40 seconds of activity: walking 2 "
            "minutes 20 seconds (41%), standing 1 minute 25 seconds (25%), sitting 1 minute 10 "
            "seconds (21%), running 45 seconds (13%).",
        )

    def test_shares_are_rounded_like_the_dashboard_not_forced_to_100(self) -> None:
        thirds = ActivityBreakdown((("Walking", 10.0), ("Sitting", 10.0), ("Running", 10.0)),
                                   simulated=False, n_sessions=1)
        self.assertEqual(thirds.percentages(), (33, 33, 33))

    def test_one_activity_and_no_data(self) -> None:
        one = ActivityBreakdown((("Walking", 140.0),), simulated=True, n_sessions=1)
        self.assertEqual(answers.breakdown_text("today", one),
                         "Today you recorded 2 minutes 20 seconds of activity, all of it walking.")
        none = ActivityBreakdown((), simulated=False, n_sessions=0)
        self.assertEqual(answers.breakdown_text("today", none),
                         "You didn't record any activity today.")
        self.assertEqual(answers.breakdown_text("latest_session", none),
                         "There's no activity data in your latest session.")
        self.assertEqual(answers.breakdown_text("all", none), "You haven't recorded any activity yet.")

    def test_every_range_and_number_comes_from_the_breakdown(self) -> None:
        allowed = digits(format_duration_words(340.0)) | {
            d for _, s in self.DEMO.items for d in digits(format_duration_words(s))
        } | {str(p) for p in self.DEMO.percentages()}
        for range_name in vocab.RANGES:
            with self.subTest(range=range_name):
                text = answers.breakdown_text(range_name, self.DEMO)
                named = digits(answers.BREAKDOWN_LEADS[range_name])
                self.assertLessEqual(digits(text), allowed | named, text)
                self.assertIsNone(_BLOCKED_RE.search(text))
                self.assertIsNone(JUDGMENT_RE.search(text))


class SummaryTextTests(unittest.TestCase):
    def test_lead_in_names_the_local_start(self) -> None:
        # 13:30 UTC on Monday Oct 12 2026 is 7:30 AM MDT.
        self.assertEqual(answers.summary_lead(utc_ms(2026, 10, 12, 13, 30), DENVER),
                         "Your latest session started at 7:30 AM on Monday, October 12.")
        # 00:05 MDT is 12:05 AM; 18:45 MDT is 6:45 PM.
        self.assertIn("12:05 AM", answers.summary_lead(utc_ms(2026, 10, 12, 6, 5), DENVER))
        self.assertIn("6:45 PM", answers.summary_lead(utc_ms(2026, 10, 13, 0, 45), DENVER))

    def test_summary_is_lead_then_plain_then_caveat(self) -> None:
        self.assertEqual(answers.summary_text("Lead.", "Plain.", None), "Lead. Plain.")
        self.assertEqual(answers.summary_text("Lead.", "Plain.", "Caveat."),
                         "Lead. Plain. Caveat.")


class UnderstoodLineTests(unittest.TestCase):
    def test_lines(self) -> None:
        cases = [
            (MetricRequest("avg_hr", "Running", "this_week"), False,
             "Average heart rate · Running · this week"),
            (MetricRequest("peak_hr", None, "last_7_days"), True,
             "Peak heart rate · All activities · last 7 days (default)"),
            (MetricRequest("time_in_activity", "Walking", "latest_session"), False,
             "Time in activity · Walking · latest session"),
            (MetricRequest("session_count", None, "all"), False,
             "Number of sessions · all sessions"),
            (MetricRequest("total_time", None, "yesterday"), False,
             "Total recorded time · yesterday"),
        ]
        for request, defaulted, line in cases:
            with self.subTest(request=request):
                self.assertEqual(answers.understood_line(request, defaulted), line)

    def test_summary_breakdown_and_corrections(self) -> None:
        self.assertEqual(answers.summary_understood_line(), "Session summary · latest session")
        self.assertEqual(answers.breakdown_understood_line("today"), "Activity breakdown · today")
        self.assertEqual(answers.breakdown_understood_line("last_7_days", True),
                         "Activity breakdown · last 7 days (default)")
        self.assertEqual(
            answers.breakdown_understood_line("last_30_days", summary_fallback=True),
            "Activity breakdown · last 30 days · a summary covers one session, so here's the "
            "breakdown",
        )
        self.assertEqual(
            answers.with_corrections("Average heart rate · All activities · this week",
                                     (("avrage", "average"),)),
            "Average heart rate · All activities · this week · corrected: avrage → average",
        )
        self.assertEqual(answers.with_corrections(None, (("a", "b"),)), "corrected: a → b")
        self.assertIsNone(answers.with_corrections(None, ()))


class CaptionTests(unittest.TestCase):
    def test_caption_follows_the_simulated_flag(self) -> None:
        self.assertEqual(answers.caption(result(5.0, "seconds", simulated=True)),
                         SIMULATED_CAPTION)
        self.assertIsNone(answers.caption(result(5.0, "seconds", simulated=False)))
        self.assertEqual(SIMULATED_CAPTION, "Activity times are from simulated predictions.")


class FixedReplyTests(unittest.TestCase):
    def test_fixed_replies(self) -> None:
        # The medical reply's wording is unchanged from v1.
        self.assertEqual(
            answers.MEDICAL_REPLY,
            "I can't judge health or give medical advice. I can tell you numbers from your "
            "recorded sessions: average or peak heart rate, time in each activity, number of "
            "sessions, or total recorded time.",
        )
        self.assertEqual(
            answers.needs_activity_reply(),
            "Which activity do you mean: sitting, walking, standing or running?",
        )
        self.assertEqual(answers.SESSION_COUNT_BY_ACTIVITY_REPLY,
                         "I can count all your sessions, but not sessions by activity yet. Try "
                         "one of these:")
        self.assertEqual(answers.NOT_UNDERSTOOD_REPLY,
                         "Sorry, I didn't catch that. Try one of these:")
        self.assertEqual(answers.UNSUPPORTED_REPLY,
                         "That isn't something I can answer from your recorded sessions. Try "
                         "one of these:")
        self.assertEqual(answers.AI_UNAVAILABLE_REPLY,
                         "I'd need the AI model for that one, and it isn't running right now. "
                         "These work without it:")
        self.assertTrue(2 <= len(answers.RULES_EXAMPLES) <= 3)
        self.assertEqual(answers.clarify_reply(["Walking", "Standing"]),
                         "Did you mean walking or standing?")
        self.assertEqual(
            answers.follow_up_reply("how long did I walk this week"),
            "I can't follow up on earlier questions yet. Try asking the whole question, like "
            "'how long did I walk this week'.",
        )
        self.assertEqual(answers.COMPARISON_REPLY,
                         "I can't compare two periods yet. You can ask about each one on its "
                         "own:")
        self.assertEqual(answers.LONGEST_REPLY,
                         "I can't find your longest session yet. Try one of these:")
