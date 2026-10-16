"""The medical/advice check that runs before any parser.

Plain-language overview: the app never judges health. Any question that
asks whether a number is healthy, normal, safe or worrying, asks for
advice, or compares against a "usual" the app doesn't have, gets one fixed
reply (answers.MEDICAL_REPLY) and no number at all -- even if it also names
a metric. This is plain code, so it works the same with or without the AI
model, and it runs before either parser sees the question.

The word lists are the session-summary checks' own (llm/validate.py's
BLOCKED_TERMS -- imported, not copied, so the two can't drift), minus the
"judgment" list, which is about how a summary is worded rather than what a
question asks; plus question-style phrases ("should I", "is it okay",
"too high"). Matching is whole-word and ignores case, on normalize()d
text. tests/test_ask_vocab.py checks that no term here matches any phrase
a supported question uses.
"""

from __future__ import annotations

from ..llm.validate import BLOCKED_TERMS
from .vocab import normalize, phrase_pattern

REUSED_CATEGORIES = ("medical_interpretive", "advice", "fabricated_comparison")

QUESTION_TERMS = (
    "should i", "should we", "is it okay", "is it ok", "is that okay", "is that ok",
    "is this okay", "is this ok", "okay for me", "ok for me", "too high", "too low",
    "too fast", "too slow", "worried", "worry", "worrying", "concerned", "concern",
    "dangerous", "danger", "safe", "unsafe", "doctor", "physician", "cardiologist",
    "disease", "illness", "sick", "symptom", "symptoms", "diagnose", "diagnosis", "medical",
    "medicine", "medication", "heart attack", "arrhythmia", "tachycardia", "bradycardia",
    "afib", "health", "healthier", "good", "bad", "better", "worse", "improve", "improved",
    "improving", "fit",
)

GUARD_TERMS: tuple[str, ...] = tuple(
    dict.fromkeys(
        [term for category in REUSED_CATEGORIES for term in BLOCKED_TERMS[category]]
        + list(QUESTION_TERMS)
    )
)


def _patterns(terms) -> list:
    # Longest first, so the reported match is the most specific one.
    return [(term, phrase_pattern(term)) for term in sorted(terms, key=len, reverse=True)]


_PATTERNS = _patterns(GUARD_TERMS)


def check(question: str, terms: tuple[str, ...] | None = None) -> str | None:
    """The first guard term found in `question`, or None if it's clear.
    `terms` replaces GUARD_TERMS (tests use it to check the matching
    itself)."""
    text = normalize(question)
    for term, pattern in _PATTERNS if terms is None else _patterns(terms):
        if pattern.search(text):
            return term
    return None
