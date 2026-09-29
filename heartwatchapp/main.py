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

from PySide6.QtWidgets import QApplication  # noqa: E402

from heartwatchapp.data import db  # noqa: E402
from heartwatchapp.data.writer import DatabaseWriter  # noqa: E402
from heartwatchapp.llm.worker import LlmWorker  # noqa: E402
from heartwatchapp.ui.main_window import HeartWatchApp  # noqa: E402
from heartwatchapp.ui.theme import Theme  # noqa: E402


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

    # Ollama may not be installed or running -- llm_worker.start() only
    # opens a background event loop, it never touches the network, so
    # this is safe even if Ollama is completely absent. The actual
    # reachability check runs off the Qt main thread, right below.
    llm_worker = LlmWorker()
    llm_worker.start()
    llm_worker.check_health()

    window = HeartWatchApp(theme, writer, llm_worker)
    app.aboutToQuit.connect(window.shutdown)
    app.aboutToQuit.connect(writer.stop)
    app.aboutToQuit.connect(llm_worker.stop)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
