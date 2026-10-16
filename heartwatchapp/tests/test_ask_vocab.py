"""Tests for the AI bar's single source of words (ask/vocab.py) and the
medical guard (ask/guard.py).

Plain-language overview: the parser, the AI model's JSON schema, the
prompt and the query layer must all offer exactly the same metrics,
activities and ranges, and the medical guard must never block a word that
a supported question uses -- like handoff 2's rule that no blocked summary
word may be part of an activity label, but for multi-word phrases too.
Stdlib unittest, no Ollama:

    python -m unittest heartwatchapp.tests.test_ask_vocab
"""

from __future__ import annotations

import json
import unittest

from ..ask import guard, llm_parser, rules, vocab
from ..ask.answers import RULES_EXAMPLES
from ..data import queries
from ..llm import prompts
from ..llm.validate import BLOCKED_TERMS
from ..ml.inference import CLASSES
from ..scripts import eval_ask


def token_jaccard(a: str, b: str) -> float:
    """|shared words| / |all words|, on vocab.normalize()d word sets."""
    x, y = set(vocab.normalize(a).split()), set(vocab.normalize(b).split())
    return len(x & y) / len(x | y) if x | y else 0.0


class SingleSourceTests(unittest.TestCase):
    def test_vocab_is_the_query_layer_and_the_classifier(self) -> None:
        self.assertEqual(vocab.METRICS, tuple(queries.METRIC_UNITS))
        self.assertEqual(vocab.ACTIVITIES, tuple(CLASSES))
        self.assertIs(vocab.RANGES, queries.RANGE_NAMES)
        self.assertIn(vocab.DEFAULT_RANGE, vocab.RANGES)

    def test_schema_enums_are_the_vocab(self) -> None:
        props = llm_parser.response_schema()["properties"]

        def enum(field: str) -> tuple:
            return tuple(props[field]["anyOf"][0]["enum"])

        self.assertEqual(enum("metric"), vocab.METRICS)
        self.assertEqual(enum("activity"), vocab.ACTIVITIES)
        self.assertEqual(enum("range"), vocab.RANGES)
        self.assertEqual(tuple(props["intent"]["enum"]), ("query", "unsupported"))

    def test_every_activity_range_and_metric_has_synonyms(self) -> None:
        self.assertEqual(set(vocab.ACTIVITY_SYNONYMS), set(vocab.ACTIVITIES))
        self.assertEqual(set(vocab.RANGE_SYNONYMS), set(vocab.RANGES))
        self.assertEqual(
            set(vocab.METRIC_PHRASES) | set(vocab.HR_MODIFIERS), set(vocab.METRICS)
        )

    def test_the_prompt_describes_exactly_the_vocab(self) -> None:
        self.assertEqual(tuple(prompts.ASK_METRIC_DESCRIPTIONS), vocab.METRICS)
        self.assertEqual(tuple(prompts.ASK_RANGE_DESCRIPTIONS), vocab.RANGES)
        for name in (*vocab.METRICS, *vocab.RANGES):
            self.assertIn(f"  {name}: ", prompts.ASK_SYSTEM_PROMPT)

    def test_prompt_examples_fit_the_schema(self) -> None:
        # Every example form in the prompt must itself be a valid reply.
        for question, metric, activity, range_name in prompts.ASK_EXAMPLES:
            with self.subTest(question=question):
                line = prompts._ask_example_line(question, metric, activity, range_name)
                reply = line.split(" -> ", 1)[1]
                self.assertNotEqual(llm_parser.parse_reply(reply).status, "invalid")

    def test_the_prompt_contrasts_the_three_week_like_ranges(self) -> None:
        ranges = {example[3] for example in prompts.ASK_EXAMPLES}
        self.assertTrue({"this_week", "last_week", "last_7_days"} <= ranges)

    def test_no_prompt_example_is_close_to_a_dev_question(self) -> None:
        # A prompt example that (nearly) repeats a dev question lets the
        # model score well on that question for the wrong reason. Limit:
        # Jaccard overlap of normalized word sets at most 0.6.
        dev = eval_ask.load_questions(eval_ask.DEFAULT_QUESTIONS)
        for example, *_ in prompts.ASK_EXAMPLES:
            for q in dev:
                with self.subTest(example=example, id=q["id"]):
                    overlap = token_jaccard(example, q["question"])
                    self.assertLessEqual(overlap, 0.6, f"{example!r} ~ {q['question']!r}")

    def test_token_jaccard(self) -> None:
        # {how, long, did, i, walk, yesterday} vs the same plus nothing: 1.0;
        # {a, b} vs {b, c}: 1 shared of 3 = 1/3.
        self.assertEqual(token_jaccard("How long did I walk yesterday?",
                                       "how long did i walk yesterday"), 1.0)
        self.assertAlmostEqual(token_jaccard("a b", "b c"), 1 / 3)

    def test_the_prompt_never_mentions_health(self) -> None:
        # The prompt must not invite interpretation: no guard word appears.
        self.assertIsNone(guard.check(prompts.ASK_SYSTEM_PROMPT))


class GuardVocabularyTests(unittest.TestCase):
    def test_reuses_validate_lists_except_judgment(self) -> None:
        for category in ("medical_interpretive", "advice", "fabricated_comparison"):
            for term in BLOCKED_TERMS[category]:
                self.assertIn(term, guard.GUARD_TERMS, f"{category}: {term}")
        for term in BLOCKED_TERMS["judgment"]:
            if term not in guard.QUESTION_TERMS:
                self.assertNotIn(term, guard.GUARD_TERMS, term)

    def test_no_guard_term_collides_with_a_supported_phrase(self) -> None:
        # Otherwise every question using that phrase would get the medical
        # reply. Checks every synonym, filler word and example question.
        phrases = vocab.all_synonym_phrases() + sorted(rules.FILLER_WORDS) + list(RULES_EXAMPLES)
        for phrase in phrases:
            with self.subTest(phrase=phrase):
                self.assertIsNone(guard.check(phrase), f"guard blocks {phrase!r}")

    def test_the_collision_check_catches_multi_word_phrases(self) -> None:
        # The check above only means something if it would catch a
        # collision: a multi-word term inside a multi-word phrase, a single
        # word inside a phrase, and a hyphenated spelling.
        self.assertEqual(guard.check("so far this week", terms=("this week",)), "this week")
        self.assertEqual(guard.check("how many sessions", terms=("many",)), "many")
        self.assertEqual(guard.check("Heart-Rate", terms=("heart rate",)), "heart rate")
        # Whole words only: "fit" is not inside "outfit".
        self.assertIsNone(guard.check("my outfit", terms=("fit",)))

    def test_dev_questions_are_guarded_exactly_when_medical(self) -> None:
        for q in eval_ask.load_questions(eval_ask.DEFAULT_QUESTIONS):
            with self.subTest(id=q["id"]):
                blocked = guard.check(q["question"]) is not None
                self.assertEqual(blocked, q["expected"]["outcome"] == "medical", q["question"])


class GuardTests(unittest.TestCase):
    def test_blocks_health_questions_even_when_they_name_a_metric(self) -> None:
        for question in (
            "Is my heart rate healthy?",
            "Is that higher than usual?",
            "Should I be worried about my peak heart rate?",
            "Was my average heart rate too high while running?",
            "is 128 bpm normal for walking",
            "Is it OK that my HR hit 150?",
            "what does my average heart rate suggest",
            "do I need a doctor",
        ):
            with self.subTest(question=question):
                self.assertIsNotNone(guard.check(question))

    def test_reports_the_most_specific_match(self) -> None:
        # "than usual" and "usual" both match; the longer one is reported.
        self.assertEqual(guard.check("is that higher than usual"), "than usual")

    def test_whole_words_only(self) -> None:
        self.assertIsNone(guard.check("average heart rate in my outfit"))
        self.assertIsNone(guard.check("peak heart rate this week"))

    def test_ignores_case_and_punctuation(self) -> None:
        self.assertEqual(guard.check("IS IT  OKAY?!"), "is it okay")
