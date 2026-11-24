"""Section 4 — ML / Activity. Stub: pipeline summary + confusion-matrix placeholder.
"Export training data" is real -- it backs export_training_csv()."""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from ..data import db
from ..ml.inference import CLASSES, INPUT_SHAPE, METADATA_PATH
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
# Derived from ml.inference.INPUT_SHAPE, so the label can't go stale. Shown
# when there is no exported model (or its metadata can't be read).
INPUT_SHAPE_TEXT = f"[{', '.join(str(d) for d in INPUT_SHAPE)}]  (window, channels)"


def _shape_text(shape) -> str:
    return f"[{', '.join(str(d) for d in shape)}]  (window, channels)"


def model_status(meta_path: str | Path = METADATA_PATH) -> list[tuple[str, str]]:
    """The "Model status" lines, from the exported model's model_meta.json.
    Only reads the JSON (the model itself isn't loaded, nothing is
    checked); a missing or unreadable file shows the app's defaults."""
    try:
        meta = json.loads(Path(meta_path).read_text())
    except (OSError, ValueError):
        meta = None
    if not isinstance(meta, dict):
        return [("Model file", "ml/models/har_cnn.tflite (no model exported)"),
                ("Input shape", INPUT_SHAPE_TEXT),
                ("Output classes", " · ".join(CLASSES)),
                ("Last updated", "—")]
    shape = meta.get("input_shape")
    if not (isinstance(shape, list) and shape and all(isinstance(d, int) for d in shape)):
        shape = list(INPUT_SHAPE)
    classes = meta.get("classes")
    if not (isinstance(classes, list) and classes):
        classes = CLASSES
    version = meta.get("model_version") or "unnamed"
    transform = meta.get("input_transform") or "raw"
    return [("Model file", f"ml/models/har_cnn.tflite — {version}, input transform {transform} (not loaded)"),
            ("Input shape", _shape_text(shape)),
            ("Output classes", " · ".join(str(c) for c in classes)),
            ("Last updated", str(meta.get("created_at") or "—"))]


class MlView(QWidget):
    def __init__(self, theme: Theme, parent: QWidget | None = None,
                 meta_path: str | Path = METADATA_PATH):
        super().__init__(parent)
        self.theme = theme

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(16)

        status = Card()
        status.body().addWidget(heading("Model status"))
        for k, v in model_status(meta_path):
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
