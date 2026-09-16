"""Matplotlib widgets embedded via FigureCanvasQTAgg.

- DonutChart: static day-distribution donut for the Dashboard.
- LiveHRChart: scrolling 60s BPM line for the Live Monitor, advanced on a QTimer
  (stands in for FuncAnimation; real BLE data replaces the dummy source later).
"""

from __future__ import annotations

import collections

import matplotlib

matplotlib.use("QtAgg")

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from PySide6.QtWidgets import QVBoxLayout, QWidget  # noqa: E402

from .theme import Theme  # noqa: E402


class _CanvasHost(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)

    def _mount(self, fig: Figure) -> FigureCanvasQTAgg:
        canvas = FigureCanvasQTAgg(fig)
        canvas.setStyleSheet("background: transparent;")
        self._layout.addWidget(canvas)
        return canvas


class DonutChart(_CanvasHost):
    def __init__(self, theme: Theme, parent: QWidget | None = None):
        super().__init__(parent)
        self.theme = theme
        self.setFixedSize(140, 140)
        self.fig = Figure(figsize=(1.4, 1.4), dpi=100)
        self.fig.patch.set_alpha(0.0)
        self.ax = self.fig.add_subplot(111)
        self.canvas = self._mount(self.fig)

    def set_data(self, pairs: list[tuple[str, float]]) -> None:
        self.ax.clear()
        labels = [p[0] for p in pairs]
        values = [p[1] for p in pairs]
        colors = [self.theme.activity_color(lbl) for lbl in labels]
        self.ax.pie(
            values,
            colors=colors,
            startangle=90,
            counterclock=False,
            wedgeprops=dict(width=0.4, edgecolor="none"),
        )
        self.ax.set_aspect("equal")
        self.fig.tight_layout(pad=0.1)
        self.canvas.draw_idle()


class LiveHRChart(_CanvasHost):
    WINDOW_SECONDS = 60

    def __init__(self, theme: Theme, parent: QWidget | None = None):
        super().__init__(parent)
        self.theme = theme
        self.fig = Figure(figsize=(5, 2.6), dpi=100)
        self.fig.patch.set_alpha(0.0)
        self.ax = self.fig.add_subplot(111)
        self.canvas = self._mount(self.fig)

        self._samples: collections.deque[float] = collections.deque(
            maxlen=self.WINDOW_SECONDS
        )
        self._style_axes()

    def _style_axes(self) -> None:
        muted = self.theme.c("text_muted")
        border = self.theme.c("border")
        self.ax.set_facecolor("none")
        for spine in ("top", "right"):
            self.ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            self.ax.spines[spine].set_color(border)
        self.ax.tick_params(colors=muted, labelsize=8)
        self.ax.set_xlim(0, self.WINDOW_SECONDS)
        self.ax.set_xticks([0, 30, 60])
        self.ax.set_xticklabels(["0s", "30s", "60s"])
        self.ax.grid(True, color=border, linewidth=0.5, alpha=0.5)

    def push(self, bpm: float) -> None:
        self._samples.append(bpm)
        self.redraw()

    def redraw(self) -> None:
        self.ax.clear()
        self._style_axes()
        if not self._samples:
            self.canvas.draw_idle()
            return
        n = len(self._samples)
        xs = list(range(self.WINDOW_SECONDS - n, self.WINDOW_SECONDS))
        ys = list(self._samples)
        danger = self.theme.c("danger")
        self.ax.plot(xs, ys, color=danger, linewidth=1.8)
        self.ax.fill_between(xs, ys, min(ys) - 3, color=danger, alpha=0.12)
        lo, hi = min(ys), max(ys)
        pad = max(4.0, (hi - lo) * 0.25)
        self.ax.set_ylim(lo - pad, hi + pad)
        self.fig.tight_layout(pad=0.4)
        self.canvas.draw_idle()

    def restyle(self, theme: Theme) -> None:
        self.theme = theme
        self.redraw()
