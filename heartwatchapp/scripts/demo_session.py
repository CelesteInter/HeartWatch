"""Try the session summaries on a throwaway demo database.

Plain-language overview: real recorded sessions don't have activity
predictions yet (the classifier arrives in milestone 6), so their summaries
have little to say. This script builds a brand-new database holding one
data-rich synthetic session -- about 6 minutes, 1 heart-rate sample per
second, four activities, a few low-confidence predictions (see
scripts/demo_db.py) -- and launches the real app pointed at it, with the
summary log ("heartwatch.llm") turned up to DEBUG so every AI pass and
rejection prints in the terminal. By default the real database is never
opened, copied or read.

Run from the repo root, with the venv active:

    python -m heartwatchapp.scripts.demo_session             # new temp demo DB
    python -m heartwatchapp.scripts.demo_session --db x.db   # choose where (must not exist yet)
    python -m heartwatchapp.scripts.demo_session --db x.db --overwrite  # replace x.db
    python -m heartwatchapp.scripts.demo_session --no-launch # just build the demo DB
    python -m heartwatchapp.scripts.demo_session --from-real # start from a copy of the real DB
    HEARTWATCH_SUMMARY_MODE=llm python -m heartwatchapp.scripts.demo_session  # AI path

--from-real copies the real database's files (.db, -wal, -shm) at the file
level and only ever opens the copy; close HeartWatch first, since copying
while the app is writing can produce an inconsistent copy.

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
    FROM_REAL_WARNING,
    REAL_DB_PATH,
    DemoDbError,
    DestinationExistsError,
    RealDatabaseError,
    build_demo_db,
    check_destination,
    ensure_not_real_db,
    real_db_path,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--db",
        help="demo database file to build; refused if it already exists, unless "
        "--overwrite is given. Default: a new temporary file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="let --db replace an existing file (and its -wal/-shm/-journal)",
    )
    parser.add_argument(
        "--from-real",
        action="store_true",
        help="start from a file-level copy of the real database instead of an "
        "empty one (close HeartWatch first)",
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

    # Work out the real database's path BEFORE pointing HEARTWATCH_DB_PATH
    # at the demo file below. Only the path -- the file isn't opened.
    real = real_db_path()
    if args.db:
        target = Path(args.db)
    else:
        target = Path(tempfile.mkdtemp(prefix="heartwatch-demo-")) / "heartwatch_demo.db"
    try:
        # The default temporary folder is brand new, so only a --db path
        # can already be taken.
        target = check_destination(target, args.overwrite or not args.db, real)
    except (RealDatabaseError, DestinationExistsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # Point the whole app at the demo file BEFORE importing any of it.
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

    if args.from_real:
        print(FROM_REAL_WARNING, file=sys.stderr, flush=True)
    try:
        target, session_id = build_demo_db(
            target, from_real=args.from_real, data_seed=args.data_seed, real=real
        )
    except RealDatabaseError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except DemoDbError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"\n>>> Demo session #{session_id} added to {target}")
    print(">>> Open Sessions and double-click it to see its summary.\n", flush=True)

    if args.no_launch:
        return 0
    from heartwatchapp import main as app_main

    return app_main.main()


if __name__ == "__main__":
    raise SystemExit(main())
