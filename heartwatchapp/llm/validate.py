"""Checks every AI-generated session summary before the user sees it.

Plain-language overview: the local model is small (1B parameters) and
does not reliably follow the rules in llm/prompts.py's SYSTEM_PROMPT. It
has been seen opening with "Okay, here's the summary:", comparing a
session to a "usual" it was never shown, hedging exact numbers ("around
89 bpm"), guessing at what a heart rate means medically, and attaching a
number to an activity it never belonged to ("a peak of 111 bpm during
the walking activity" when it was never told when the peak was). So instead
of trusting the prompt, this file checks the reply itself. If a reply
fails any check it is thrown away and the app shows plain numbers
instead (data/stats.py's format_stats_plain()). A rejected reply is never
shown on screen.

The order things run in, from llm/worker.py:

    strip_preamble(raw)        drop a clear "Here's the summary:" opener
    validate_summary(text)     A) blocked words  B) hedged numbers
                               C) any number that wasn't in the input
                               D) a number or the peak attached to the
                                  wrong activity
    screen_summary(raw)        both of the above, plus logging

Every rejection is logged on the "heartwatch.llm" logger (WARNING) with
the session id, the check that failed, what matched, and the full
rejected text; passes are logged at DEBUG so a rejection rate can be
worked out from the log.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from ..ml.inference import CLASSES
from . import prompts

logger = logging.getLogger("heartwatch.llm")

# Names of the checks, as reported in ValidationResult.check and the log.
CHECK_EMPTY = "empty"
CHECK_BLOCKED = "blocked_vocabulary"
CHECK_HEDGED = "hedged_number"
CHECK_UNGROUNDED = "ungrounded_number"
CHECK_LABEL_PAIRING = "D"  # label-number pairing; logged as check=D per handoff 3

# -- Check A: blocked vocabulary --------------------------------------------
# Words and phrases a summary must never contain. Matching ignores case and
# only matches whole words, so "asterisk" does not trip "risk" and
# "average heart rate" is fine (only the phrase "average for you" is
# blocked). Tune by hand. Before adding a word here, make sure it isn't
# (part of) an activity label -- currently Sitting / Walking / Standing /
# Running (ml/inference.py's CLASSES) -- or every summary of that activity
# would be rejected. tests/test_validate.py checks this.
BLOCKED_TERMS: dict[str, tuple[str, ...]] = {
    # Medical or interpretive: says what the numbers *mean* for the body.
    "medical_interpretive": (
        "suggest", "suggests", "suggested", "suggesting",
        "indicate", "indicates", "indicated", "indicating",
        "exertion", "healthy", "unhealthy", "normal", "abnormal",
        "concerning", "elevated", "recovery", "fitness", "condition",
        "risk", "stress",
    ),
    # Advice: tells the user what to do.
    "advice": (
        "you should", "recommend", "recommends", "recommended",
        "try to", "consider", "make sure",
    ),
    # Judgment or editorializing: grades the session instead of describing it.
    "judgment": (
        "achieved", "impressive", "great", "good job", "interestingly",
        "notably", "unfortunately",
    ),
    # Fabricated comparison: the model is only ever shown ONE session, so
    # any comparison to "usual" or "typical" is made up.
    "fabricated_comparison": (
        "than usual", "usual", "usually", "higher than normal", "typical",
        "typically", "baseline", "average for you",
    ),
}

# -- Check B: hedged numbers --------------------------------------------------
# Every number the model is given is exact (it came straight from SQLite),
# so a hedge word directly in front of a number is always wrong. "about"
# elsewhere in a sentence ("a session about walking") is fine.
HEDGE_WORDS = ("around", "approximately", "about", "roughly", "nearly", "almost")


def _phrase_pattern(term: str) -> str:
    # Allow any run of whitespace between the words of a phrase.
    return r"\s+".join(re.escape(word) for word in term.split())


_ALL_BLOCKED = sorted(
    {term for terms in BLOCKED_TERMS.values() for term in terms}, key=len, reverse=True
)
_BLOCKED_RE = re.compile(
    r"\b(?:" + "|".join(_phrase_pattern(t) for t in _ALL_BLOCKED) + r")\b",
    re.IGNORECASE,
)
_HEDGED_RE = re.compile(
    r"(?:\b(?:" + "|".join(HEDGE_WORDS) + r")\s+|~\s*)\d",
    re.IGNORECASE,
)
# Digits, optionally with a decimal part: "89", "4", "89.0". A sentence's
# final full stop ("89.") is not swallowed because a digit must follow it.
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")

# -- Check D: label-number pairing --------------------------------------------
# Words that mean a sentence is talking about the peak heart rate.
_PEAK_WORD_RE = re.compile(r"\b(?:peak\w*|highest|maximum)\b", re.IGNORECASE)
# Sentence boundaries: ". ! ?" followed by whitespace, or a line break (the
# model often puts one sentence per line without a full stop).
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")

# -- preamble stripping -------------------------------------------------------
# A chatty opener at the very start of the reply: "Okay, here's the
# summary:", "Sure! Here's the summary:", "Summary:", "**Here's the
# summary:**". See strip_preamble() for exactly when it's removed.
_OPENER_RE = re.compile(
    r"^[\s*#_]*(?:okay|ok|sure|here(?:'|’)s|here\s+is|summary)\b",
    re.IGNORECASE,
)
# The opener's colon has to be within this many characters of the start.
PREAMBLE_COLON_WINDOW = 80


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of validate_summary(). `text` is the text that was checked.
    On failure, `check` is one of the CHECK_* names above and `matched` is
    the word, phrase, or number that tripped it (None for CHECK_EMPTY)."""

    passed: bool
    text: str
    check: str | None = None
    matched: str | None = None


def strip_preamble(text: str) -> str:
    """Removes a clearly conversational opener from the start of a reply,
    and nothing else. Two cases only:

    - The first line ends in a colon ("Here's a summary of your session:"
      on its own line) -- that whole line is dropped.
    - The text starts with Okay / Ok / Sure / Here's / Here is / Summary
      and its first colon is within the first PREAMBLE_COLON_WINDOW (80)
      characters -- everything up to and including that colon is dropped,
      even on the same line as the summary ("Sure! Here's the summary:
      Session duration was...").

    Kept conservative so a real first sentence is never mangled. The
    opener rule does NOT fire if:
    - the text doesn't start with an opener ("Session duration: 6 minutes
      20 seconds." is left alone),
    - there's no colon in the first 80 characters,
    - the colon is part of a time like "4:12",
    - a full sentence ends between the opener and the colon ("Here is how
      the session went. Walking: 3 minutes" is left alone). A "." or "!"
      right after the opener word itself ("Okay. Here's...", "Sure! ...")
      doesn't count.

    Returns "" if nothing but preamble was there; validate_summary()
    rejects "".
    """
    text = text.strip()
    first_line, newline, rest = text.partition("\n")
    if newline and first_line.rstrip(" *_").endswith(":"):
        text = rest.strip()
    elif not newline and text.rstrip(" *_").endswith(":"):
        # The whole reply is a single line ending in a colon -- preamble
        # with no summary after it.
        return ""

    opener = _OPENER_RE.match(text)
    if not opener:
        return text
    colon = text.find(":", 0, PREAMBLE_COLON_WINDOW)
    if colon == -1 or text[colon + 1 : colon + 2].isdigit():
        return text
    lead_in = text[opener.end() : colon].lstrip(".!,")
    if re.search(r"[.?]", lead_in):
        return text
    return text[colon + 1 :].lstrip(" \t\n*_").strip()


def _numbers(text: str) -> set[float]:
    # float() so "89.0" and "89" compare equal.
    return {float(token) for token in _NUMBER_RE.findall(text)}


def validate_summary(text: str, stats: dict) -> ValidationResult:
    """Checks a (preamble-stripped) summary against the stats it was
    generated from. Runs the checks in order and reports the first failure:

    A) Blocked vocabulary -- any word/phrase in BLOCKED_TERMS.
    B) Hedged number -- a HEDGE_WORDS word (or "~") right before a digit.
    C) Number grounding -- every number in the summary must also appear in
       the prompt the model was given (llm/prompts.py's
       render_session_summary_prompt(stats)). This catches invented
       numbers however they are worded, and also catches the model doing
       its own conversions ("4.2 minutes" when the input said "4 minutes
       12 seconds") -- which is intended; the prompt already spells every
       value out the way it should be written.
    D) Label-number pairing -- see _check_label_pairing(). Catches a real
       number attached to the wrong activity, which C can't see.

    Known gaps: numbers written as words ("two activities", "five
    minutes") are not checked by C or D, and D only recognizes activities
    by their label ("walking" counts, "walked" doesn't).

    Empty text fails with check=CHECK_EMPTY.
    """
    if not text.strip():
        return ValidationResult(False, text, CHECK_EMPTY)

    match = _BLOCKED_RE.search(text)
    if match:
        return ValidationResult(False, text, CHECK_BLOCKED, match.group(0))

    match = _HEDGED_RE.search(text)
    if match:
        return ValidationResult(False, text, CHECK_HEDGED, match.group(0))

    allowed = _numbers(prompts.render_session_summary_prompt(stats))
    for token in _NUMBER_RE.findall(text):
        if float(token) not in allowed:
            return ValidationResult(False, text, CHECK_UNGROUNDED, token)

    mismatch = _check_label_pairing(text, stats)
    if mismatch:
        return ValidationResult(False, text, CHECK_LABEL_PAIRING, mismatch)

    return ValidationResult(True, text)


def _check_label_pairing(text: str, stats: dict) -> str | None:
    """Check D. Goes sentence by sentence and returns a short description
    of the first problem, or None if there isn't one. For each sentence
    that names an activity (case-insensitive, whole word -- "the walking
    activity" names Walking):

    1. The activity must be one this session actually has facts for.
       Naming an activity with no predictions ("Running", when nothing was
       classified as Running) is rejected.
    2. Peak rule: if the sentence talks about the peak ("peak", "peaked",
       "highest", "maximum"), every activity it names must be the one
       Python reported the peak happened during. If Python couldn't tell
       (peak_hr_activity is None), any peak sentence naming an activity is
       rejected. A peak sentence that names a second activity is therefore
       always rejected -- deliberately strict, since we can't tell which
       activity the peak is being attached to.
    3. Pairing rule, for a sentence naming EXACTLY ONE activity: every
       number in it must come from that activity's own fact line or from
       the session-level facts (length, average HR, peak HR). "Walking took
       1 minute 10 seconds" fails if 1 minute 10 seconds was Sitting's time.

    Known gap: a sentence naming two or more activities skips the pairing
    rule (too ambiguous to check reliably) -- only rules 1 and 2 apply.
    """
    session_numbers = _numbers("\n".join(prompts.session_fact_lines(stats)))
    activity_lines = prompts.activity_fact_lines(stats)
    activity_numbers = {label: _numbers(line) for label, line in activity_lines.items()}
    peak_activity = stats.get("peak_hr_activity")

    known = {label.lower(): label for label in [*CLASSES, *activity_lines]}
    label_re = re.compile(
        r"\b(" + "|".join(re.escape(label) for label in known) + r")\b", re.IGNORECASE
    )

    for sentence in _SENTENCE_SPLIT_RE.split(text):
        named = sorted({known[m.lower()] for m in label_re.findall(sentence)})
        if not named:
            continue
        for label in named:
            if label not in activity_lines:
                return f"{label} (no {label} in this session)"
        if _PEAK_WORD_RE.search(sentence):
            for label in named:
                if label != peak_activity:
                    return f"peak during {label} (peak was during {peak_activity})"
        if len(named) == 1:
            label = named[0]
            allowed = activity_numbers[label] | session_numbers
            for token in _NUMBER_RE.findall(sentence):
                if float(token) not in allowed:
                    return f"{label} with {token}"
    return None


def screen_summary(raw_text: str, stats: dict) -> ValidationResult:
    """The full post-generation pipeline: strip_preamble(), then
    validate_summary(), then log the outcome on "heartwatch.llm". This is
    what llm/worker.py calls. Only show `result.text` if `result.passed`."""
    result = validate_summary(strip_preamble(raw_text), stats)
    session_id = stats.get("session_id")
    if result.passed:
        logger.debug("summary passed: session=%s text=%r", session_id, result.text)
    else:
        logger.warning(
            "summary rejected: session=%s check=%s matched=%r text=%r",
            session_id,
            result.check,
            result.matched,
            raw_text,
        )
    return result
