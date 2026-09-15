"""HeartWatch host app — entry point.

Run:  python -m heartwatchapp.main   (from the repo root)
  or: python heartwatchapp/main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `python heartwatchapp/main.py` as well as `python -m heartwatchapp.main`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtWidgets import QApplication  # noqa: E402

from heartwatchapp.data import db  # noqa: E402
from heartwatchapp.data.writer import DatabaseWriter  # noqa: E402
from heartwatchapp.ui.main_window import HeartWatchApp  # noqa: E402
from heartwatchapp.ui.theme import Theme  # noqa: E402


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("HeartWatch")

    theme = Theme(mode="dark")
    app.setStyleSheet(theme.stylesheet())

    db.init_db()
    writer = DatabaseWriter()
    writer.start()

    window = HeartWatchApp(theme, writer)
    app.aboutToQuit.connect(window.shutdown)
    app.aboutToQuit.connect(writer.stop)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
