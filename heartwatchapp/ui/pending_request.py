"""Waits for one background reply at a time, with a timeout.

Plain-language overview: LlmWorker gives every request an id and answers
later, possibly after the user has moved on. A PendingRequest remembers
the one id a widget is still waiting for: claim(id) says whether a reply
is the one to show (and stops waiting), start() with a new id replaces the
old one -- so a newer question always wins over an older one -- and if no
reply comes within the timeout, timed_out fires once and the late reply,
whenever it arrives, is ignored. Nothing is cancelled on the worker; stale
replies are simply dropped.

Used by ui/ai_bar.py. The session detail dialog (ui/sessions.py) still has
its own copy of this logic; moving it here is a follow-up.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, QTimer, Signal


class PendingRequest(QObject):
    # The id that ran out of time. Emitted once, after which nothing is pending.
    timed_out = Signal(int)

    def __init__(self, timeout_ms: int, parent: QObject | None = None):
        super().__init__(parent)
        self._pending_id: int | None = None
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(timeout_ms)
        self._timer.timeout.connect(self._on_timeout)

    @property
    def pending_id(self) -> int | None:
        return self._pending_id

    def start(self, request_id: int) -> None:
        """Wait for `request_id` (forgetting any earlier one) and restart
        the timeout."""
        self._pending_id = request_id
        self._timer.start()

    def claim(self, request_id: int) -> bool:
        """True (and stops waiting) if `request_id` is the one still
        pending; False for anything stale."""
        if self._pending_id is None or request_id != self._pending_id:
            return False
        self.cancel()
        return True

    def cancel(self) -> None:
        """Stop waiting; any reply that still arrives is stale."""
        self._pending_id = None
        self._timer.stop()

    def _on_timeout(self) -> None:
        request_id, self._pending_id = self._pending_id, None
        if request_id is not None:
            self.timed_out.emit(request_id)
