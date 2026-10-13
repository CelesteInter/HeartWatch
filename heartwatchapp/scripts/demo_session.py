"""Try the session summaries on a throwaway copy of the database.

Plain-language overview: real recorded sessions don't have activity
predictions yet (the classifier arrives in milestone 6), so their summaries
have little to say. This script makes a copy of the real database, adds one
data-rich synthetic session to the COPY -- about 6 minutes, 1 heart-rate
sample per second, four activities, a few low-confidence predictions (see
scripts/demo_db.py) -- and launches the real app pointed at the copy, with
the summary log ("heartwatch.llm") turned up to DEBUG so every AI pass and
rejection prints in the terminal. The real database is only ever read,
never written.

Run from the repo root, with the venv active:

    python -m heartwatchapp.scripts.demo_session             # temp copy
    python -m heartwatchapp.scripts.demo_session --db x.db   # reuse/choose a copy
    python -m heartwatchapp.scripts.demo_session --no-launch # just build the copy
    HEARTWATCH_SUMMARY_MODE=llm python -m heartwatchapp.scripts.demo_session  # AI path

Then open Sessions, set the filter to "All time" if needed, and
double-click the session number the script prints.

How it points the app elsewhere: it sets HEARTWATCH_DB_PATH (read by
data/db.py) BEFORE importing any app code, and refuses to run if that
path turns out to be the real database.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# config.py and demo_db.py import nothing that fixes the database path.
from heartwatchapp.config import DB_PATH_ENV_VAR  # noqa: E402
from heartwatchapp.scripts.demo_db import (  # noqa: E402,F401 - re-exported for tests
    REAL_DB_PATH,
    RealDatabaseError,
    build_demo_db,
    ensure_not_real_db,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--db",
        help="database file to use (created as a copy of the real one if it "
        "doesn't exist yet). Default: a new temporary copy.",
    )
    parser.add_argument(
        "--no-launch", action="store_true", help="build the demo database, then exit"
    )
    parser.add_argument(
        "--data-seed",
        type=int,
        default=None,
        help="fix the demo session's random heart-rate values (default: random)",
    )
    args = parser.parse_args(argv)

    if args.db:
        target = Path(args.db)
    else:
        target = Path(tempfile.mkdtemp(prefix="heartwatch-demo-")) / "heartwatch_demo.db"
    try:
        target = ensure_not_real_db(target)
    except RealDatabaseError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # Point the whole app at the copy BEFORE importing any of it.
    os.environ[DB_PATH_ENV_VAR] = str(target)

    # DEBUG for the AI summary log so passes print too, not just rejections.
    # Configured before main.py's basicConfig, which then leaves it alone.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("heartwatch.llm").setLevel(logging.DEBUG)

    from heartwatchapp.data import db

    if db.DB_PATH != target:  # belt and braces: never touch anything else
        print(
            f"error: the app would use {db.DB_PATH}, not {target} "
            "(was heartwatchapp.data.db imported too early?). Nothing was written.",
            file=sys.stderr,
        )
        return 2

    target, session_id = build_demo_db(target, data_seed=args.data_seed)
    print(f"\n>>> Demo session #{session_id} added to {target}")
    print(">>> Open Sessions and double-click it to see its summary.\n", flush=True)

    if args.no_launch:
        return 0
    from heartwatchapp import main as app_main

    return app_main.main()


if __name__ == "__main__":
    raise SystemExit(main())
