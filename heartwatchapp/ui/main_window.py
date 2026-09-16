"""HeartWatchApp — the QMainWindow shell.

Layout: sidebar (64px) · [ topbar / QStackedWidget / AI bar ] on the right.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QMainWindow,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..data.writer import DatabaseWriter
from .ai_bar import AiBar
from .dashboard import DashboardView
from .live_monitor import LiveMonitorView
from .ml_view import MlView
from .sessions import SessionsView
from .settings import SettingsView
from .sidebar import Sidebar
from .theme import PAD_CONTENT, Theme
from .topbar import Topbar

SECTION_META = [
    ("Dashboard", "Today's overview"),
    ("Live Monitor", "Streaming from MAX30101 · MPU-9250"),
    ("Sessions", "Past recorded sessions"),
    ("ML / Activity", "Model status and training pipeline"),
    ("Settings", "Pairing, streaming, data export"),
]


class HeartWatchApp(QMainWindow):
    def __init__(self, theme: Theme, writer: DatabaseWriter | None = None):
        super().__init__()
        self.theme = theme
        self.writer = writer
        self.setWindowTitle("HeartWatch")
        self.resize(1180, 780)
        self.setMinimumSize(940, 640)

        root = QWidget()
        root.setObjectName("RootWidget")
        from PySide6.QtWidgets import QHBoxLayout

        h = QHBoxLayout(root)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)

        self.sidebar = Sidebar(theme)
        self.sidebar.section_changed.connect(self._go_to_section)
        h.addWidget(self.sidebar)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(0)

        self.topbar = Topbar(theme)
        rv.addWidget(self.topbar)

        self.stack = QStackedWidget()
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.stack)
        rv.addWidget(self.scroll, 1)

        self.ai_bar = AiBar(theme)
        rv.addWidget(self.ai_bar)

        h.addWidget(right, 1)
        self.setCentralWidget(root)

        self._build_views()
        self._go_to_section(0)

    # -- views ---------------------------------------------------------
    def _build_views(self) -> None:
        # Close out the previous Live Monitor's demo session (stamps ended_at)
        # before its view -- and the session it opened -- is torn down.
        if getattr(self, "live", None) is not None:
            self.live.shutdown()

        while self.stack.count():
            w = self.stack.widget(0)
            self.stack.removeWidget(w)
            w.deleteLater()

        self.dashboard = DashboardView(self.theme)
        self.live = LiveMonitorView(self.theme, self.writer)
        self.sessions = SessionsView(self.theme)
        self.ml = MlView(self.theme)
        self.settings = SettingsView(self.theme, self.writer)
        self.settings.theme_toggled.connect(self._toggle_theme)
        self.settings.data_changed.connect(self._on_data_changed)

        for view in (self.dashboard, self.live, self.sessions, self.ml, self.settings):
            self.stack.addWidget(self._pad(view))

    def _on_data_changed(self) -> None:
        """Seeding or deleting a session from the developer panel should be
        reflected immediately without switching sections."""
        if hasattr(self.dashboard, "refresh"):
            self.dashboard.refresh()
        if hasattr(self.sessions, "refresh"):
            self.sessions.refresh()

    def shutdown(self) -> None:
        """Called once, from app.aboutToQuit, before the writer thread stops."""
        if getattr(self, "live", None) is not None:
            self.live.shutdown()

    def _pad(self, view: QWidget) -> QWidget:
        wrap = QWidget()
        lay = QVBoxLayout(wrap)
        lay.setContentsMargins(PAD_CONTENT, PAD_CONTENT, PAD_CONTENT, PAD_CONTENT)
        lay.addWidget(view)
        return wrap

    def _go_to_section(self, idx: int) -> None:
        self.stack.setCurrentIndex(idx)
        title, subtitle = SECTION_META[idx]
        self.topbar.set_section(title, subtitle)
        btn = self.sidebar.group.button(idx)
        if btn is not None and not btn.isChecked():
            btn.setChecked(True)
            self.sidebar.restyle()

    # -- theme -------------------------------------------------------
    def _toggle_theme(self) -> None:
        from PySide6.QtWidgets import QApplication

        self.theme.toggle()
        current = self.stack.currentIndex()
        QApplication.instance().setStyleSheet(self.theme.stylesheet())
        self.sidebar.restyle()
        self._build_views()
        self.sidebar.group.button(current).setChecked(True)
        self._go_to_section(current)
