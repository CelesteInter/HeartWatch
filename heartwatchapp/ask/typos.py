"""Typo correction for the "Ask about your data" bar.

Plain-language overview: before either parser sees a question, each word
the bar doesn't know is compared (difflib, standard library) with the words
it does know -- metric, activity and time words, their synonyms, and filler
words like "what". A word is corrected only when the match is clear:

- it isn't already a known word, and has at least MIN_LETTERS letters
  ("rn" is never guessed at);
- difflib's similarity is at least CUTOFF, or the word is exactly one swap
  of two neighbouring letters away from a known word ("waht" -> "what":
  four-letter swaps only score 0.75, so the cutoff alone misses them);
- and only one meaning is that close. Words that mean the same thing
  ("session" / "sessions", "walk" / "walking") count as one meaning; if
  two different meanings are within MARGIN of each other ("sitnding":
  standing 0.875, sitting 0.800) nothing is corrected and the question is
  marked ambiguous, so the bar can ask which was meant.

Multi-word phrases ("heart rate", "last week") need no special handling:
the corrected words are put back in place and the parsers match phrases on
the corrected text as usual. The medical guard (ask/guard.py) does its own,
looser fuzzy matching on top of this, because a missed refusal is worse
than an extra one.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Iterable
from dataclasses import dataclass

from . import rules, vocab

CUTOFF = 0.8
MARGIN = 0.1
MIN_LETTERS = 4

_WORD = re.compile(r"[a-z']+")


def _meaning(word: str) -> str:
    """Words that mean the same thing share a key: an activity's synonyms
    share the activity, and a plural shares its singular."""
    activity = vocab.activity_of_word(word)
    if activity is not None:
        return f"activity:{activity}"
    return re.sub(r"'?s$", "", word)


def _target_words() -> dict[str, str]:
    """Every word a correction may produce -> its meaning key."""
    phrases = (
        vocab.all_synonym_phrases()
        + list(vocab.AMBIGUOUS_RANGE_PHRASES)
        + list(vocab.UNSUPPORTED_WORDS)
        + list(vocab.UNSUPPORTED_COMMAND_PHRASES)
        + list(vocab.COMMAND_FIRST_WORDS)
        + sorted(rules.FILLER_WORDS)
    )
    words = {w for phrase in phrases for w in phrase.split() if _WORD.fullmatch(w)}
    return {w: _meaning(w) for w in sorted(words)}


TARGETS: dict[str, str] = _target_words()


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def _swaps(word: str) -> set[str]:
    """Every word one swap of neighbouring letters away from `word`."""
    return {word[:i] + word[i + 1] + word[i] + word[i + 2:] for i in range(len(word) - 1)}


def best_target_score(word: str) -> float:
    """How close `word` gets to any known word (1.0 if it is one)."""
    if word in TARGETS:
        return 1.0
    return max(similarity(word, t) for t in TARGETS)


def _candidates(word: str) -> list[tuple[float, str]]:
    """The best (score, word) per meaning at or above the cutoff, best first."""
    best: dict[str, tuple[float, str]] = {}
    swapped = _swaps(word) & set(TARGETS)
    for target, meaning in TARGETS.items():
        score = similarity(word, target)
        if target in swapped:
            score = max(score, CUTOFF)
        if score >= CUTOFF and score > best.get(meaning, (0.0, ""))[0]:
            best[meaning] = (score, target)
    return sorted(best.values(), key=lambda pair: (-pair[0], pair[1]))


def _letters(word: str) -> int:
    return sum(ch.isalpha() for ch in word)


@dataclass(frozen=True)
class TypoResult:
    text: str                                    # the normalize()d question, corrected
    corrections: tuple[tuple[str, str], ...]     # (original word, corrected word)
    # The first word that was close to two meanings, with the closest word
    # of each (best first); None if there wasn't one.
    ambiguous: tuple[str, tuple[str, ...]] | None = None

    def replace(self, word: str, replacement: str) -> str:
        """The corrected text with every `word` replaced -- for building
        the question behind a "Did you mean ...?" button."""
        return " ".join(replacement if t == word else t for t in self.text.split())


_MULTI_WORD_PHRASES = tuple(p for p in vocab.all_synonym_phrases() if " " in p)


def _completes_a_phrase(tokens: list[str], i: int, word: str) -> bool:
    """True if `word` at position i forms part of a known multi-word phrase
    with the word before or after it ("lsat week" -> "last week")."""
    pairs = []
    if i > 0:
        pairs.append(f"{tokens[i - 1]} {word}")
    if i + 1 < len(tokens):
        pairs.append(f"{word} {tokens[i + 1]}")
    return any(
        vocab.phrase_pattern(pair).search(phrase) for pair in pairs for phrase in _MULTI_WORD_PHRASES
    )


def correct(question: str, also_known: Iterable[str] = ()) -> TypoResult:
    """Corrects clear typos in `question` (see the module docstring).
    `also_known` are extra words never to touch (the guard's terms). Never
    raises.

    When two meanings are equally close, the one that completes a known
    phrase with a neighbouring word wins if it's the only one ("stand lsat
    week": "last week", not "sat week"); otherwise the word is ambiguous."""
    known = set(also_known)
    tokens = vocab.normalize(question).split()
    out = list(tokens)
    corrections: list[tuple[str, str]] = []
    ambiguous = None
    for i, token in enumerate(tokens):
        if (
            token in TARGETS or token in known or not _WORD.fullmatch(token)
            or _letters(token) < MIN_LETTERS
        ):
            continue
        candidates = _candidates(token)
        if not candidates:
            continue
        close = [w for s, w in candidates if candidates[0][0] - s < MARGIN]
        if len(close) > 1:
            in_phrase = [w for w in close if _completes_a_phrase(tokens, i, w)]
            if len(in_phrase) != 1:
                if ambiguous is None:
                    ambiguous = (token, tuple(close))
                continue
            close = in_phrase
        out[i] = close[0]
        corrections.append((token, close[0]))
    return TypoResult(" ".join(out), tuple(corrections), ambiguous)
