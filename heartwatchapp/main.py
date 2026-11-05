"""HeartWatch host app — entry point.

Run:  python -m heartwatchapp.main   (from the repo root)
  or: python heartwatchapp/main.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Allow `python heartwatchapp/main.py` as well as `python -m heartwatchapp.main`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtCore import SignalInstance  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from heartwatchapp.data import db  # noqa: E402
from heartwatchapp.data.writer import DatabaseWriter  # noqa: E402
from heartwatchapp.llm.worker import LlmWorker  # noqa: E402
from heartwatchapp.ui.main_window import HeartWatchApp  # noqa: E402
from heartwatchapp.ui.theme import Theme  # noqa: E402


def connect_shutdown(
    about_to_quit: SignalInstance,
    window: HeartWatchApp,
    writer: DatabaseWriter,
    llm_worker: LlmWorker | None,
) -> None:
    """What runs when the app quits, in this order (Qt calls slots in the
    order they were connected): the window first, which stops and saves a
    recording in progress -- its last samples and ended_at still go
    through the writer -- then the writer (flushes and closes the
    database), then the LLM worker's loop."""
    about_to_quit.connect(window.shutdown)
    about_to_quit.connect(writer.stop)
    if llm_worker is not None:
        about_to_quit.connect(llm_worker.stop)


def main() -> int:
    # Prints log messages to the terminal. The main thing that logs today is
    # the "heartwatch.llm" logger, which records every AI summary the app
    # rejected and why (llm/validate.py). Change INFO to DEBUG to also see
    # every summary that passed, e.g. to work out a rejection rate.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    app = QApplication(sys.argv)
    app.setApplicationName("HeartWatch")

    theme = Theme(mode="dark")
    app.setStyleSheet(theme.stylesheet())

    db.init_db()
    writer = DatabaseWriter()
    writer.start()
    # Crash recovery: close any session left unfinished by a crash or a
    # force-quit, before anything can start recording.
    closed = writer.close_orphan_sessions()
    logging.getLogger("heartwatch.recording").info(
        "Closed %d unfinished session(s) left by an unclean exit", closed
    )

    # Ollama may not be installed or running -- llm_worker.start() only
    # opens a background event loop, it never touches the network, so
    # this is safe even if Ollama is completely absent. The actual
    # reachability check runs off the Qt main thread, right below.
    llm_worker = LlmWorker()
    llm_worker.start()
    llm_worker.check_health()

    window = HeartWatchApp(theme, writer, llm_worker)
    connect_shutdown(app.aboutToQuit, window, writer, llm_worker)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
