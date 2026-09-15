"""Section 4 — ML / Activity. Stub: pipeline summary + confusion-matrix placeholder.
"Export training data" is real -- it backs export_training_csv()."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from ..data import db
from .theme import Theme
from .widgets import Card, PlaceholderState, heading, muted


def _left(w: QWidget) -> QWidget:
    holder = QWidget()
    lay = QHBoxLayout(holder)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.addWidget(w)
    lay.addStretch(1)
    return holder

PIPELINE_SUMMARY = (
    "windowed [ax, ay, az, gx, gy, gz]  →  1D CNN  →  TFLite inference"
)


class MlView(QWidget):
    def __init__(self, theme: Theme, parent: QWidget | None = None):
        super().__init__(parent)
        self.theme = theme

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(16)

        status = Card()
        status.body().addWidget(heading("Model status"))
        for k, v in [
            ("Model file", "ml/models/har_cnn.tflite (not loaded)"),
            ("Input shape", "[1, 128, 6]  (window, channels)"),
            ("Output classes", "Sitting · Walking · Standing · Running"),
            ("Last updated", "—"),
        ]:
            line = QLabel(f"{k}:  {v}")
            line.setObjectName("Mono")
            status.body().addWidget(line)
        root.addWidget(status)

        pipeline = Card()
        pipeline.body().addWidget(heading("Pipeline"))
        summary = QLabel(PIPELINE_SUMMARY)
        summary.setObjectName("Mono")
        summary.setWordWrap(True)
        pipeline.body().addWidget(summary)
        pipeline.body().addWidget(muted("Export training data → labeled CSV dump from SQLite."))
        export_btn = QPushButton("Export training data")
        export_btn.setObjectName("GhostButton")
        export_btn.clicked.connect(self._on_export_clicked)
        pipeline.body().addWidget(_left(export_btn))
        self._export_status = muted(" ")
        pipeline.body().addWidget(self._export_status)
        root.addWidget(pipeline)

        matrix = Card()
        matrix.body().addWidget(heading("Confusion matrix"))
        matrix.body().addWidget(
            PlaceholderState(
                "Awaiting evaluation data",
                "The confusion matrix and per-class confidence histogram render "
                "here once a labeled evaluation set is available.",
                theme,
            )
        )
        root.addWidget(matrix, 1)

    def _on_export_clicked(self) -> None:
        session_ids = [s["id"] for s in db.list_sessions(limit=1000)]
        if not session_ids:
            self._export_status.setText("No sessions recorded yet -- nothing to export.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export training data", "training_data.csv", "CSV files (*.csv)"
        )
        if not path:
            return
        n = db.export_training_csv(session_ids, path)
        self._export_status.setText(f"Wrote {n} rows from {len(session_ids)} session(s) to {path}")
