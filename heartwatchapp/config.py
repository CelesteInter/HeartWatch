"""App-wide settings read from environment variables.

Plain-language overview: a couple of things about how the app runs can be
changed without editing code, by setting an environment variable before
launching it. Each one is read through a small function that takes the
environment as an argument (defaulting to the real one), so tests can pass
a plain dict instead of touching os.environ.

- HEARTWATCH_DB_PATH -- run against a different database file (read by
  data/db.py's resolve_db_path(); see scripts/demo_session.py).
- HEARTWATCH_SUMMARY_MODE -- how the session detail dialog writes its
  summary; see summary_mode() below.

Nothing here imports the rest of the app, so scripts can read these names
before anything else (in particular data/db.py, which fixes the database
path the moment it's imported).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping

# Set this environment variable to run the whole app against a different
# database file (e.g. a throwaway copy -- see scripts/demo_session.py).
DB_PATH_ENV_VAR = "HEARTWATCH_DB_PATH"

SUMMARY_MODE_ENV_VAR = "HEARTWATCH_SUMMARY_MODE"
# "plain": the app's own sentences from data/stats.py's format_stats_plain(),
# shown straight away, with no call to Ollama. The default -- in the
# side-by-side test the plain summary was correct every time, while about
# half of the AI summaries that passed every check still said something
# false.
SUMMARY_MODE_PLAIN = "plain"
# "llm": ask the local model (llm/worker.py) to phrase the same facts,
# falling back to the plain summary if it's unavailable or rejected.
SUMMARY_MODE_LLM = "llm"
SUMMARY_MODES = (SUMMARY_MODE_PLAIN, SUMMARY_MODE_LLM)

logger = logging.getLogger("heartwatch.llm")
# Unrecognized values already warned about, so a typo is reported once
# instead of every time a session is opened.
_warned_summary_modes: set[str] = set()


def summary_mode(environ: Mapping[str, str] = os.environ) -> str:
    """"plain" or "llm", from HEARTWATCH_SUMMARY_MODE (case-insensitive).
    Unset or empty gives "plain". Anything else also gives "plain", with
    one warning on the "heartwatch.llm" logger. Read each time a summary is
    built, not once at import, so it can be changed between runs of a
    script or test without reloading anything."""
    raw = environ.get(SUMMARY_MODE_ENV_VAR) or ""
    mode = raw.strip().lower()
    if not mode:
        return SUMMARY_MODE_PLAIN
    if mode in SUMMARY_MODES:
        return mode
    if raw not in _warned_summary_modes:
        _warned_summary_modes.add(raw)
        logger.warning(
            "%s=%r isn't one of %s; using %r",
            SUMMARY_MODE_ENV_VAR,
            raw,
            ", ".join(SUMMARY_MODES),
            SUMMARY_MODE_PLAIN,
        )
    return SUMMARY_MODE_PLAIN
