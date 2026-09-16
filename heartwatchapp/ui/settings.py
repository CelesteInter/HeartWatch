"""Section 5 — Settings. BLE/streaming/export controls are still placeholders;
the Developer group is real and backs the SQLite acceptance demo."""

from __future__ import annotations

import datetime

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..data import db
from ..data.writer import DatabaseWriter
from .theme import Theme
from .widgets import Badge, Card, Divider, heading, muted


def _left(w: QWidget) -> QWidget:
    """Wrap a control so it hugs its content instead of stretching full-width."""
    holder = QWidget()
    lay = QHBoxLayout(holder)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.addWidget(w)
    lay.addStretch(1)
    return holder


class SettingsView(QWidget):
    theme_toggled = Signal()
    data_changed = Signal()  # emitted after the developer panel seeds/deletes data

    def __init__(
        self,
        theme: Theme,
        writer: DatabaseWriter | None = None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.theme = theme
        self.writer = writer

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(16)

        # -- BLE device --
        ble = Card()
        ble.body().addWidget(heading("BLE device"))
        ble.body().addWidget(muted("Paired device: none"))
        scan_row = QHBoxLayout()
        scan_btn = QPushButton("Scan for devices")
        scan_btn.setObjectName("GhostButton")
        scan_btn.setEnabled(False)
        disconnect_btn = QPushButton("Disconnect")
        disconnect_btn.setObjectName("GhostButton")
        disconnect_btn.setEnabled(False)
        scan_row.addWidget(scan_btn)
        scan_row.addWidget(disconnect_btn)
        scan_row.addStretch(1)
        holder = QWidget()
        holder.setLayout(scan_row)
        ble.body().addWidget(holder)
        root.addWidget(ble)

        # -- Streaming mode --
        stream = Card()
        stream.body().addWidget(heading("Streaming mode"))
        stream.body().addWidget(muted("Transport decision not finalized — disabled for now."))
        grp = QButtonGroup(self)
        wifi = QRadioButton("WiFi")
        usb = QRadioButton("USB-C serial")
        wifi.setChecked(True)
        for rb in (wifi, usb):
            rb.setEnabled(False)
            grp.addButton(rb)
            stream.body().addWidget(rb)
        root.addWidget(stream)

        # -- Data export --
        export = Card()
        export.body().addWidget(heading("Data export"))
        export_btn = QPushButton("Export all sessions to CSV")
        export_btn.setObjectName("GhostButton")
        export_btn.setEnabled(False)
        export.body().addWidget(_left(export_btn))
        root.addWidget(export)

        # -- Display --
        display = Card()
        display.body().addWidget(heading("Display"))
        theme_btn = QPushButton("Toggle light / dark")
        theme_btn.setObjectName("GhostButton")
        theme_btn.clicked.connect(self.theme_toggled.emit)
        display.body().addWidget(_left(theme_btn))
        display.body().addWidget(Divider())
        display.body().addWidget(muted("Chart refresh rate: 1 Hz (fixed for prototype)"))
        root.addWidget(display)

        # -- Developer (SQLite demo tools) --
        root.addWidget(self._developer_card())

        root.addStretch(1)

    # -- Developer panel ---------------------------------------------------
    def _developer_card(self) -> Card:
        card = Card()

        title_row = QHBoxLayout()
        title_row.setSpacing(8)
        title_row.addWidget(heading("Developer"))
        dev_badge = Badge("DEV TOOL", self.theme.c("warning"))
        title_row.addWidget(dev_badge)
        title_row.addStretch(1)
        title_holder = QWidget()
        title_holder.setLayout(title_row)
        card.body().addWidget(title_holder)

        card.body().addWidget(
            muted(
                "Exercises the SQLite layer directly from the GUI: seed, "
                "inspect, and delete session data. Not part of the shipped app."
            )
        )
        card.body().addWidget(Divider())

        # 1. Seed demo session
        card.body().addWidget(heading("Seed demo session"))
        card.body().addWidget(
            muted("Writes 2-3 minutes of synthetic IMU + HR data through the real insert path.")
        )
        seed_btn = QPushButton("Seed demo session")
        seed_btn.setObjectName("GhostButton")
        seed_btn.setEnabled(self.writer is not None)
        seed_btn.clicked.connect(self._on_seed_clicked)
        card.body().addWidget(_left(seed_btn))
        self._seed_status = muted(" ")
        card.body().addWidget(self._seed_status)
        card.body().addWidget(Divider())

        # 2. Session picker + delete
        card.body().addWidget(heading("Delete session"))
        card.body().addWidget(
            muted("Deletes the selected session; CASCADE removes its samples and predictions.")
        )
        picker_row = QHBoxLayout()
        picker_row.setSpacing(8)
        self._session_picker = QComboBox()
        self._session_picker.setMinimumWidth(280)
        picker_row.addWidget(self._session_picker)
        delete_btn = QPushButton("Delete session")
        delete_btn.setObjectName("GhostButton")
        delete_btn.setEnabled(self.writer is not None)
        delete_btn.clicked.connect(self._on_delete_clicked)
        picker_row.addWidget(delete_btn)
        picker_row.addStretch(1)
        picker_holder = QWidget()
        picker_holder.setLayout(picker_row)
        card.body().addWidget(picker_holder)
        card.body().addWidget(Divider())

        # 3. samples_flat demo query
        card.body().addWidget(heading("Run samples_flat query"))
        card.body().addWidget(
            muted(
                "SELECT * FROM samples_flat LIMIT 20 -- the acceptance-review demo. "
                "x/y/z are the accelerometer axes; heartrate is carried forward from "
                "the nearest earlier HR sample."
            )
        )
        query_btn = QPushButton("Run samples_flat query")
        query_btn.setObjectName("GhostButton")
        query_btn.setEnabled(self.writer is not None)
        query_btn.clicked.connect(self._on_run_query_clicked)
        card.body().addWidget(_left(query_btn))
        self._query_table = QTableWidget(0, 5)
        self._query_table.setHorizontalHeaderLabels(["DATETIME", "x", "y", "z", "heartrate"])
        self._query_table.verticalHeader().setVisible(False)
        self._query_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._query_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._query_table.setMaximumHeight(220)
        card.body().addWidget(self._query_table)
        card.body().addWidget(Divider())

        # 4. Database info
        card.body().addWidget(heading("Database info"))
        self._info_label = QLabel("")
        self._info_label.setObjectName("Mono")
        self._info_label.setWordWrap(True)
        card.body().addWidget(self._info_label)
        refresh_btn = QPushButton("Refresh")
        refresh_btn.setObjectName("GhostButton")
        refresh_btn.clicked.connect(self._refresh_info)
        card.body().addWidget(_left(refresh_btn))

        self._refresh_session_picker()
        self._refresh_info()
        return card

    def _on_seed_clicked(self) -> None:
        if self.writer is None:
            return
        session_id = self.writer.seed_demo_session()
        self._seed_status.setText(f"Seeded session #{session_id}.")
        self._refresh_session_picker()
        self._refresh_info()
        self.data_changed.emit()

    def _refresh_session_picker(self) -> None:
        self._session_picker.clear()
        for s in db.list_sessions(limit=200):
            started = datetime.datetime.fromtimestamp(s["started_at"] / 1000)
            text = f"#{s['id']} · {s['label'] or 'Unlabeled'} · {started:%b %d, %H:%M}"
            self._session_picker.addItem(text, s["id"])

    def _on_delete_clicked(self) -> None:
        if self.writer is None or self._session_picker.count() == 0:
            return
        session_id = self._session_picker.currentData()
        label = self._session_picker.currentText()
        reply = QMessageBox.question(
            self,
            "Delete session",
            f"Delete {label}?\n\nThis permanently removes its IMU samples, "
            "HR samples, and predictions.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self.writer.delete_session(session_id)
        self._seed_status.setText(f"Deleted session #{session_id}.")
        self._refresh_session_picker()
        self._refresh_info()
        self.data_changed.emit()

    def _on_run_query_clicked(self) -> None:
        if self.writer is not None:
            self.writer.flush_and_wait()
        rows = db.run_samples_flat_demo(limit=20)
        self._query_table.setRowCount(len(rows))
        for row_idx, row in enumerate(rows):
            values = [row["DATETIME"], row["x"], row["y"], row["z"], row["heartrate"]]
            for col, value in enumerate(values):
                self._query_table.setItem(row_idx, col, QTableWidgetItem(str(value)))

    def _refresh_info(self) -> None:
        if self.writer is not None:
            self.writer.flush_and_wait()  # any buffered inserts land before we count
        info = db.get_db_info()
        lines = [
            f"path            {info['path']}",
            f"size            {info['size_bytes'] / 1024:.1f} KB",
            f"schema version  {info['schema_version']}",
            "",
        ]
        for table, count in info["row_counts"].items():
            lines.append(f"{table:<14}  {count} rows")
        self._info_label.setText("\n".join(lines))
