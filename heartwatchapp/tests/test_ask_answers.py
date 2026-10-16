"""Tests for the AI bar's wording (ask/answers.py).

Plain-language overview: every sentence the bar shows is a Python
template. These tests pin each template's exact text for hand-made
results: the five metrics, every range phrase, empty data (said in words,
never a 0), "1 session" vs "2 sessions", the "what I understood" line, the
simulated-predictions caption, and the fixed replies. No database needed.

    python -m unittest heartwatchapp.tests.test_ask_answers
"""

from __future__ import annotations

import unittest

from ..ask import answers, vocab
from ..data.queries import SIMULATED_CAPTION, MetricRequest, MetricResult


def result(value, unit="bpm", n=1, empty=False, simulated=False) -> MetricResult:
    return MetricResult(value, unit, n, empty, simulated)


class ReplyTextTests(unittest.TestCase):
    def test_each_metric(self) -> None:
        cases = [
            # 127.6 rounds to 128, the same as the Dashboard's cards (.0f).
            (MetricRequest("avg_hr", "Running", "this_week"), result(127.6, n=3),
             "Your average heart rate while Running this week was 128 bpm (3 sessions)."),
            (MetricRequest("peak_hr", None, "last_7_days"), result(134),
             "Your peak heart rate in the last 7 days was 134 bpm (1 session)."),
            # 140 s = 2 minutes 20 seconds.
            (MetricRequest("time_in_activity", "Walking", "yesterday"),
             result(140.0, "seconds"),
             "Time spent Walking yesterday: 2 minutes 20 seconds (1 session)."),
            (MetricRequest("session_count", None, "last_30_days"), result(4, "sessions", n=4),
             "You recorded 4 sessions in the last 30 days."),
            # 1085 s = 18 minutes 5 seconds.
            (MetricRequest("total_time", None, "this_week"), result(1085.0, "seconds", n=4),
             "Total recorded time this week: 18 minutes 5 seconds (4 sessions)."),
        ]
        for request, res, text in cases:
            with self.subTest(metric=request.metric):
                self.assertEqual(answers.reply_text(request, res), text)

    def test_every_range_phrase(self) -> None:
        expected = {
            "today": "today", "yesterday": "yesterday", "this_week": "this week",
            "last_week": "last week", "last_7_days": "in the last 7 days",
            "last_30_days": "in the last 30 days", "all": "across all sessions",
            "latest_session": "in your latest session",
        }
        self.assertEqual(set(answers.RANGE_PHRASES), set(vocab.RANGES))
        self.assertEqual(set(answers.RANGE_LABELS), set(vocab.RANGES))
        for range_name, phrase in expected.items():
            with self.subTest(range=range_name):
                text = answers.reply_text(MetricRequest("peak_hr", None, range_name), result(120))
                self.assertEqual(text, f"Your peak heart rate {phrase} was 120 bpm (1 session).")

    def test_pluralization(self) -> None:
        one = answers.reply_text(MetricRequest("session_count", None, "today"),
                                 result(1, "sessions"))
        self.assertEqual(one, "You recorded 1 session today.")
        two = answers.reply_text(MetricRequest("avg_hr", None, "today"), result(90, n=2))
        self.assertEqual(two, "Your average heart rate today was 90 bpm (2 sessions).")
        # "in total" rather than "across all sessions" after "sessions".
        total = answers.reply_text(MetricRequest("session_count", None, "all"),
                                   result(7, "sessions", n=7))
        self.assertEqual(total, "You recorded 7 sessions in total.")

    def test_a_real_zero_is_shown_but_missing_data_never_is(self) -> None:
        # Activity data exists but none of it was Running: a real 0, which
        # no session contributed to, so no session count is shown.
        zero = answers.reply_text(MetricRequest("time_in_activity", "Running", "today"),
                                  result(0.0, "seconds", n=0))
        self.assertEqual(zero, "Time spent Running today: 0 seconds.")
        empty = result(None, n=0, empty=True)
        cases = [
            (MetricRequest("avg_hr", "Running", "this_week"),
             "No heart-rate data recorded while Running this week."),
            (MetricRequest("peak_hr", None, "today"), "No heart-rate data recorded today."),
            (MetricRequest("time_in_activity", "Walking", "yesterday"),
             "No activity data recorded yesterday."),
            (MetricRequest("session_count", None, "yesterday"), "No sessions recorded yesterday."),
            (MetricRequest("total_time", None, "last_week"), "No sessions recorded last week."),
            (MetricRequest("session_count", None, "all"), "No sessions recorded yet."),
            (MetricRequest("total_time", None, "latest_session"), "No sessions recorded yet."),
        ]
        for request, text in cases:
            with self.subTest(request=request):
                reply = answers.reply_text(request, empty)
                self.assertEqual(reply, text)
                self.assertNotIn("0", reply)


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


class CaptionTests(unittest.TestCase):
    def test_caption_follows_the_simulated_flag(self) -> None:
        self.assertEqual(answers.caption(result(5.0, "seconds", simulated=True)),
                         SIMULATED_CAPTION)
        self.assertIsNone(answers.caption(result(5.0, "seconds", simulated=False)))
        self.assertEqual(SIMULATED_CAPTION, "Activity times are from simulated predictions.")


class FixedReplyTests(unittest.TestCase):
    def test_fixed_replies(self) -> None:
        self.assertEqual(
            answers.MEDICAL_REPLY,
            "I can't judge health or give medical advice. I can tell you numbers from your "
            "recorded sessions: average or peak heart rate, time in each activity, number of "
            "sessions, or total recorded time.",
        )
        self.assertEqual(
            answers.needs_activity_reply(),
            "Which activity? I can answer for Sitting, Walking, Standing or Running.",
        )
        self.assertEqual(answers.SESSION_COUNT_BY_ACTIVITY_REPLY,
                         "I can count all your sessions, but not sessions by activity yet.")
        self.assertEqual(answers.NOT_UNDERSTOOD_REPLY,
                         "Sorry, I didn't understand that. Try something like 'peak heart rate "
                         "last week'.")
        self.assertTrue(answers.AI_UNAVAILABLE_REPLY.startswith(
            "I couldn't understand that without the AI model, which isn't running right now."
        ))
        for example in answers.RULES_EXAMPLES:
            self.assertIn(f"'{example}'", answers.AI_UNAVAILABLE_REPLY)
        self.assertTrue(2 <= len(answers.RULES_EXAMPLES) <= 3)
