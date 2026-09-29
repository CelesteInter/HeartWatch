"""Tests for llm/validate.py (the checks every AI summary must pass before
it's shown) and the parts of llm/prompts.py those checks depend on.

Uses stdlib unittest, same as test_stats.py. Run from the repo root with:

    python -m unittest heartwatchapp.tests.test_validate

No live Ollama daemon is needed -- every "model reply" here is a fixed
string, including the real Session #7 output from the handoff.
"""

from __future__ import annotations

import re
import unittest

from ..llm import prompts, validate
from ..ml.inference import CLASSES

# The real gemma3:1b reply for Session #7 that motivated this handoff.
SESSION_7_OUTPUT = (
    "Okay, here's the summary: The user's heart rate was higher in the last "
    "minute of the session than usual, reaching around 89 bpm. They also had "
    "a noticeable peak in heart rate of approximately 89 bpm during this "
    "time, suggesting a short period of heightened exertion."
)

# A data-rich session. Its rendered prompt contains exactly these numbers:
# 5 minutes 30 seconds, 101 bpm, 134 bpm, Walking 3 minutes 10 seconds,
# Standing 2 minutes 5 seconds.
RICH_STATS = {
    "session_id": 42,
    "label": "Walking",
    "duration_s": 330.0,
    "avg_hr": 101.4,
    "max_hr": 134,
    "hr_sample_count": 330,
    "activity_seconds": {"Walking": 190.0, "Standing": 125.0},
    "prediction_count": 30,
    "low_confidence_count": 4,
}

GOOD_SUMMARY = (
    "The session lasted 5 minutes 30 seconds, with an average heart rate of "
    "101 bpm and a peak of 134 bpm. Walking took 3 minutes 10 seconds and "
    "Standing took 2 minutes 5 seconds."
)


# The handoff-3 example session: three activities, peak during Walking.
# Prompt numbers: 6 minutes 20 seconds, 93 bpm, 111 bpm; Sitting 2 minutes
# 10 seconds, Walking 2 minutes 30 seconds, Standing 1 minute 40 seconds.
FACT_STATS = {
    "session_id": 43,
    "label": "Walking",
    "duration_s": 380.0,
    "avg_hr": 92.6,
    "max_hr": 111,
    "hr_sample_count": 380,
    "activity_seconds": {"Walking": 150.0, "Sitting": 130.0, "Standing": 100.0},
    "peak_hr_activity": "Walking",
    "prediction_count": 76,
    "low_confidence_count": 3,
}


def _check(text: str) -> validate.ValidationResult:
    return validate.validate_summary(text, RICH_STATS)


class PromptTests(unittest.TestCase):
    def test_confidence_is_not_sent_to_the_model(self) -> None:
        prompt = prompts.render_session_summary_prompt(
            {**RICH_STATS, "low_confidence_count": 17}
        )
        self.assertNotIn("confidence", prompt.lower())
        self.assertNotIn("17", re.findall(r"\d+", prompt))

    def test_durations_are_spelled_out_for_the_model(self) -> None:
        prompt = prompts.render_session_summary_prompt(RICH_STATS)
        self.assertIn("Session length: 5 minutes 30 seconds", prompt)
        self.assertIn("Time Walking: 3 minutes 10 seconds", prompt)
        self.assertIn("Time Standing: 2 minutes 5 seconds", prompt)

    def test_every_number_comes_from_a_fact_line(self) -> None:
        fact_lines = prompts.session_fact_lines(FACT_STATS) + list(
            prompts.activity_fact_lines(FACT_STATS).values()
        )
        prompt_lines = prompts.render_session_summary_prompt(FACT_STATS).splitlines()
        # Every line of the prompt with a digit in it is a fact line; the
        # instruction line has no digits at all.
        for line in prompt_lines:
            if re.search(r"\d", line):
                self.assertIn(line, fact_lines)
        self.assertFalse(re.search(r"\d", prompt_lines[-1]))
        self.assertEqual(
            re.findall(r"\d+", "\n".join(prompt_lines)),
            re.findall(r"\d+", "\n".join(fact_lines)),
        )

    def test_one_activity_line_per_predicted_class_in_canonical_order(self) -> None:
        prompt = prompts.render_session_summary_prompt(FACT_STATS)
        self.assertEqual(
            [line for line in prompt.splitlines() if line.startswith("Time ")],
            [
                "Time Sitting: 2 minutes 10 seconds",
                "Time Walking: 2 minutes 30 seconds",
                "Time Standing: 1 minute 40 seconds",
            ],
        )
        self.assertIn("Peak heart rate occurred during: Walking", prompt)

    def test_peak_activity_line_left_out_when_unknown(self) -> None:
        prompt = prompts.render_session_summary_prompt({**FACT_STATS, "peak_hr_activity": None})
        self.assertNotIn("occurred during", prompt)

    def test_format_duration_words(self) -> None:
        self.assertEqual(prompts.format_duration_words(252.4), "4 minutes 12 seconds")
        self.assertEqual(prompts.format_duration_words(60), "1 minute")
        self.assertEqual(prompts.format_duration_words(3725), "1 hour 2 minutes 5 seconds")
        self.assertEqual(prompts.format_duration_words(0.2), "0 seconds")


class BlockedVocabularyTests(unittest.TestCase):
    """Check A -- one failing case per category, plus word-boundary cases."""

    def _assert_blocked(self, text: str, term: str) -> None:
        result = _check(text)
        self.assertFalse(result.passed)
        self.assertEqual(result.check, validate.CHECK_BLOCKED)
        self.assertEqual(result.matched.lower(), term)

    def test_medical_interpretive(self) -> None:
        self._assert_blocked(
            "The peak of 134 bpm indicates a hard effort.", "indicates"
        )

    def test_advice(self) -> None:
        self._assert_blocked("You should rest after 5 minutes 30 seconds.", "you should")

    def test_judgment(self) -> None:
        self._assert_blocked("A great session of 5 minutes 30 seconds.", "great")

    def test_fabricated_comparison(self) -> None:
        self._assert_blocked("Your heart rate of 101 bpm was higher than usual.", "than usual")

    def test_matching_ignores_case(self) -> None:
        self._assert_blocked("NOTABLY, the peak was 134 bpm.", "notably")

    def test_word_containing_blocked_term_passes(self) -> None:
        # "asterisk" contains "risk"; whole-word matching must not trip on it.
        self.assertTrue(_check("The peak, marked with an asterisk, was 134 bpm.").passed)

    def test_average_heart_rate_passes(self) -> None:
        self.assertTrue(_check("The average heart rate was 101 bpm.").passed)

    def test_average_for_you_fails(self) -> None:
        self._assert_blocked("101 bpm is about average for you.", "average for you")

    def test_no_blocked_term_is_part_of_an_activity_label(self) -> None:
        # Otherwise every summary of that activity would be rejected.
        every_class = {**RICH_STATS, "activity_seconds": {label: 190.0 for label in CLASSES}}
        for label in CLASSES:
            with self.subTest(label=label):
                result = validate.validate_summary(
                    f"{label} took 3 minutes 10 seconds.", every_class
                )
                self.assertTrue(
                    result.passed,
                    f"blocked vocabulary collides with activity label {label!r}: {result}",
                )


class HedgedNumberTests(unittest.TestCase):
    """Check B."""

    def test_around_before_number_fails(self) -> None:
        result = _check("The peak was around 134 bpm.")
        self.assertFalse(result.passed)
        self.assertEqual(result.check, validate.CHECK_HEDGED)
        self.assertEqual(result.matched, "around 1")

    def test_tilde_before_number_fails(self) -> None:
        self.assertEqual(_check("The peak was ~134 bpm.").check, validate.CHECK_HEDGED)

    def test_about_without_following_number_passes(self) -> None:
        self.assertTrue(
            _check("This is about a session of 5 minutes 30 seconds.").passed
        )


class NumberGroundingTests(unittest.TestCase):
    """Check C."""

    def test_number_not_in_input_fails(self) -> None:
        result = _check("The peak was 140 bpm.")
        self.assertFalse(result.passed)
        self.assertEqual(result.check, validate.CHECK_UNGROUNDED)
        self.assertEqual(result.matched, "140")

    def test_converted_duration_fails(self) -> None:
        # 5 minutes 30 seconds converted by the model into 5.5 minutes.
        result = _check("The session lasted 5.5 minutes.")
        self.assertFalse(result.passed)
        self.assertEqual(result.check, validate.CHECK_UNGROUNDED)
        self.assertEqual(result.matched, "5.5")

    def test_echoing_input_numbers_passes(self) -> None:
        self.assertTrue(_check(GOOD_SUMMARY).passed)

    def test_trailing_zero_decimal_is_the_same_number(self) -> None:
        self.assertTrue(_check("The peak was 134.0 bpm.").passed)


class LabelPairingTests(unittest.TestCase):
    """Check D."""

    INVENTED_PEAK = "Heart rate reached a peak of 111 bpm during the walking activity."

    def _d(self, text: str, **overrides) -> validate.ValidationResult:
        return validate.validate_summary(text, {**FACT_STATS, **overrides})

    def _assert_d_fails(self, result: validate.ValidationResult) -> None:
        self.assertFalse(result.passed, result)
        self.assertEqual(result.check, validate.CHECK_LABEL_PAIRING)

    def test_peak_during_wrong_activity_fails(self) -> None:
        self._assert_d_fails(self._d(self.INVENTED_PEAK, peak_hr_activity="Standing"))

    def test_peak_during_right_activity_passes(self) -> None:
        self.assertTrue(self._d(self.INVENTED_PEAK, peak_hr_activity="Walking").passed)

    def test_walking_sentence_with_sittings_duration_fails(self) -> None:
        result = self._d("Walking took 2 minutes 10 seconds.")
        self._assert_d_fails(result)
        self.assertEqual(result.matched, "Walking with 10")

    def test_walking_sentence_with_session_level_number_passes(self) -> None:
        self.assertTrue(
            self._d("While Walking for 2 minutes 30 seconds, average heart rate was 93 bpm.").passed
        )

    def test_two_activities_skip_pairing(self) -> None:
        # Sitting's and Walking's times swapped -- not caught, by design.
        self.assertTrue(
            self._d("Sitting took 2 minutes 30 seconds and Walking took 2 minutes 10 seconds.").passed
        )

    def test_two_activities_still_enforce_the_peak_rule(self) -> None:
        self._assert_d_fails(
            self._d("The peak of 111 bpm came while Standing, after Walking for 2 minutes 30 seconds.")
        )

    def test_peak_with_a_label_when_peak_activity_unknown_fails(self) -> None:
        self._assert_d_fails(self._d(self.INVENTED_PEAK, peak_hr_activity=None))

    def test_peak_without_a_label_passes(self) -> None:
        self.assertTrue(self._d("The peak heart rate was 111 bpm.", peak_hr_activity=None).passed)

    def test_activity_not_in_the_session_fails(self) -> None:
        result = self._d("Running took 2 minutes.")
        self._assert_d_fails(result)
        self.assertIn("Running", result.matched)

    def test_lines_without_full_stops_are_separate_sentences(self) -> None:
        self._assert_d_fails(
            self._d("Session length was 6 minutes 20 seconds\nWalking took 2 minutes 10 seconds")
        )

    def test_full_grounded_summary_passes(self) -> None:
        self.assertTrue(
            self._d(
                "The session lasted 6 minutes 20 seconds, with an average heart rate "
                "of 93 bpm. The peak heart rate of 111 bpm occurred during Walking. "
                "Sitting took 2 minutes 10 seconds, Walking took 2 minutes 30 seconds, "
                "and Standing took 1 minute 40 seconds."
            ).passed
        )

    def test_rejection_is_logged_with_check_d(self) -> None:
        with self.assertLogs("heartwatch.llm", level="WARNING") as logs:
            validate.screen_summary(self.INVENTED_PEAK, {**FACT_STATS, "peak_hr_activity": "Standing"})
        self.assertIn("check=D", logs.output[0])


class StripPreambleTests(unittest.TestCase):
    def test_same_line_preamble_is_removed(self) -> None:
        self.assertEqual(
            validate.strip_preamble(
                "Sure! Here's the summary: Session duration was 6 minutes 20 seconds."
            ),
            "Session duration was 6 minutes 20 seconds.",
        )

    def test_ok_opener_is_removed(self) -> None:
        self.assertEqual(validate.strip_preamble(f"Ok: {GOOD_SUMMARY}"), GOOD_SUMMARY)

    def test_sentence_with_a_colon_but_no_opener_is_untouched(self) -> None:
        text = "Session duration: 6 minutes 20 seconds."
        self.assertEqual(validate.strip_preamble(text), text)

    def test_opener_without_colon_in_first_80_characters_is_untouched(self) -> None:
        text = "Okay " + "so " * 30 + "here it is: the session lasted 6 minutes 20 seconds."
        self.assertGreater(text.index(":"), validate.PREAMBLE_COLON_WINDOW)
        self.assertEqual(validate.strip_preamble(text), text)

    def test_opener_with_a_time_colon_is_untouched(self) -> None:
        text = "Sure, it lasted 6:20 in total."
        self.assertEqual(validate.strip_preamble(text), text)

    def test_session_7_preamble_is_removed(self) -> None:
        stripped = validate.strip_preamble(SESSION_7_OUTPUT)
        self.assertTrue(stripped.startswith("The user's heart rate was higher"))

    def test_first_line_ending_in_colon_is_removed(self) -> None:
        text = f"Here is a summary of your session:\n\n{GOOD_SUMMARY}"
        self.assertEqual(validate.strip_preamble(text), GOOD_SUMMARY)

    def test_markdown_wrapped_preamble_is_removed(self) -> None:
        text = f"**Summary:** {GOOD_SUMMARY}"
        self.assertEqual(validate.strip_preamble(text), GOOD_SUMMARY)

    def test_normal_first_sentence_is_untouched(self) -> None:
        self.assertEqual(validate.strip_preamble(GOOD_SUMMARY), GOOD_SUMMARY)

    def test_colon_after_a_full_sentence_is_untouched(self) -> None:
        # Starts with "Here is", but the first colon is past the end of the
        # first sentence, so it isn't a preamble.
        text = "Here is how the session went. Walking: 3 minutes 10 seconds."
        self.assertEqual(validate.strip_preamble(text), text)

    def test_preamble_only_is_rejected(self) -> None:
        for text in ("Okay, here's the summary:", "Summary:", "Sure, here it is:\n"):
            with self.subTest(text=text):
                stripped = validate.strip_preamble(text)
                self.assertEqual(stripped, "")
                result = validate.validate_summary(stripped, RICH_STATS)
                self.assertFalse(result.passed)
                self.assertEqual(result.check, validate.CHECK_EMPTY)


class EndToEndTests(unittest.TestCase):
    def test_real_session_7_output_still_fails_after_stripping(self) -> None:
        # Contains "than usual", "around 89", "approximately 89", and
        # "suggesting" -- any one of these must be enough.
        session_7_stats = {
            "session_id": 7,
            "label": None,
            "duration_s": 22.8,
            "avg_hr": 78.0,
            "max_hr": 89,
            "hr_sample_count": 23,
            "activity_seconds": {},
            "prediction_count": 0,
            "low_confidence_count": 0,
        }
        with self.assertLogs("heartwatch.llm", level="WARNING") as logs:
            result = validate.screen_summary(SESSION_7_OUTPUT, session_7_stats)
        self.assertFalse(result.passed)
        self.assertEqual(result.check, validate.CHECK_BLOCKED)
        # The log carries the session id, the check, and the full raw text.
        self.assertIn("session=7", logs.output[0])
        self.assertIn(validate.CHECK_BLOCKED, logs.output[0])
        self.assertIn("Okay, here's the summary", logs.output[0])

    def test_session_7_output_fails_every_check_it_should(self) -> None:
        # Each of the four problems on its own is enough to reject it.
        stats_7 = {"duration_s": 22.8, "avg_hr": 78.0, "max_hr": 89, "activity_seconds": {}}
        for sentence, check in (
            ("It was higher than usual at 89 bpm.", validate.CHECK_BLOCKED),
            ("It reached around 89 bpm.", validate.CHECK_HEDGED),
            ("It peaked at approximately 89 bpm.", validate.CHECK_HEDGED),
            ("It peaked at 89 bpm, suggesting effort.", validate.CHECK_BLOCKED),
        ):
            with self.subTest(sentence=sentence):
                self.assertEqual(validate.validate_summary(sentence, stats_7).check, check)

    def test_good_summary_passes_and_logs_at_debug(self) -> None:
        with self.assertLogs("heartwatch.llm", level="DEBUG") as logs:
            result = validate.screen_summary(f"Sure, here it is: {GOOD_SUMMARY}", RICH_STATS)
        self.assertTrue(result.passed)
        self.assertEqual(result.text, GOOD_SUMMARY)
        self.assertTrue(logs.output[0].startswith("DEBUG:heartwatch.llm:summary passed"))


if __name__ == "__main__":
    unittest.main()
