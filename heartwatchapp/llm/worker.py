"""Qt-facing bridge between the async Ollama client and the GUI thread.

Plain-language overview: Qt's signal/slot system and Python's asyncio event
loop are two different ways of waiting for things to happen, and they don't
talk to each other on their own. This module owns one background thread
with one asyncio event loop for the whole app's lifetime -- the same idea
as data/writer.py owning one background thread for every SQLite write --
and turns each finished, checked reply into a Qt signal, which Qt
automatically delivers to the main thread. GUI code below only ever
touches Signals; nothing here blocks the Qt main thread.

Design note: the rest of the app has no existing pattern for "a background
thread pushes updates to the GUI" to reuse -- DatabaseWriter's thread never
talks back to Qt directly (the UI just calls refresh() after a blocking
write finishes), and the BLE/asyncio IO thread the handoff describes
doesn't exist yet (that's build step 6, not started). This worker is the
first of its kind, built the same way the rest of the app builds a
dedicated long-lived worker thread, using ordinary Qt Signals (the same
mechanism ui/ai_bar.py and ui/settings.py already use) to get data back to
the GUI safely across threads.
"""

from __future__ import annotations

import asyncio
import itertools
import threading

import ollama
from PySide6.QtCore import QObject, Signal

from . import client as llm_client
from . import prompts, validate


class LlmWorker(QObject):
    """Owns the background asyncio loop that talks to Ollama.

    One instance lives for the whole app (created in main.py), same as
    DatabaseWriter. Every public method here is safe to call from the Qt
    main thread; the network/model work always happens on this worker's
    own thread so it can never freeze the GUI.
    """

    # HealthStatus, as returned by llm.client.health_check(). Emitted once at
    # startup, and again after every summary request that shows whether
    # Ollama is reachable -- so the topbar's AI pill (wired up in
    # ui/main_window.py) flips to "unavailable" if Ollama quits and back to
    # "ready" once it answers again, with no polling.
    health_checked = Signal(object)
    # (request_id, summary text) -- emitted once, only for a reply that
    # passed every check in llm/validate.py.
    summary_finished = Signal(int, str)
    # (request_id, human-readable reason) -- emitted instead of _finished
    # when Ollama is unavailable, the model isn't pulled, or the reply
    # failed validation. The rejected text itself is never sent.
    summary_failed = Signal(int, str)

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._loop_ready = threading.Event()
        # Every summary request gets a fresh id, so a reply can be matched
        # to the exact request (and dialog) that asked for it.
        self._request_ids = itertools.count(1)

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Starts the background thread and its event loop. Call once,
        before check_health() or request_session_summary()."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run_loop, name="HeartWatchLLM", daemon=True
        )
        self._thread.start()
        self._loop_ready.wait()

    def stop(self) -> None:
        """Stops the background loop. Call once, from app shutdown -- same
        moment main.py stops the DatabaseWriter."""
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop_ready.set()
        self._loop.run_forever()
        self._loop.close()

    # -- requests (called from the Qt main thread) --------------------------

    def check_health(self) -> None:
        """Runs health_check() on the worker thread; emits health_checked
        with the result once it's done."""
        self._submit(self._do_health_check())

    def request_session_summary(self, stats: dict) -> int:
        """Asks for a plain-language summary of `stats` (see data/stats.py's
        compute_session_stats) and returns this request's id straight away.
        Later emits summary_finished(request_id, text) if the reply passed
        llm/validate.py's checks, or summary_failed(request_id, reason) if
        Ollama isn't available or the reply was rejected. Callers should
        ignore any signal whose request_id isn't the one they're waiting on."""
        request_id = next(self._request_ids)
        self._submit(self._do_session_summary(request_id, stats))
        return request_id

    def _submit(self, coro) -> None:
        if self._loop is None:
            raise RuntimeError("LlmWorker.start() must be called before use")
        asyncio.run_coroutine_threadsafe(coro, self._loop)

    # -- work that actually runs on the worker thread -----------------------

    async def _do_health_check(self) -> None:
        status = await llm_client.health_check()
        self.health_checked.emit(status)

    async def _do_session_summary(self, request_id: int, stats: dict) -> None:
        messages = [
            {"role": "system", "content": prompts.SYSTEM_PROMPT},
            {"role": "user", "content": prompts.render_session_summary_prompt(stats)},
        ]
        try:
            raw_text = await llm_client.chat(messages, options=llm_client.SUMMARY_OPTIONS)
        except ollama.ResponseError as exc:
            if exc.status_code == 404:
                reason = f"the model isn't pulled yet (run: ollama pull {llm_client.MODEL_NAME})"
                self.health_checked.emit(llm_client.model_missing_status())
            else:
                reason = str(exc)
            self.summary_failed.emit(request_id, reason)
            return
        except llm_client.CONNECTION_ERRORS as exc:
            self.health_checked.emit(llm_client.unreachable_status(exc))
            self.summary_failed.emit(request_id, "Ollama isn't running")
            return
        except Exception as exc:  # noqa: BLE001 - worker thread must never crash the app
            self.summary_failed.emit(request_id, str(exc))
            return

        # Ollama answered, so it's available -- even if the text below ends
        # up rejected (that's the model's wording failing, not Ollama).
        self.health_checked.emit(llm_client.ready_status())

        # Strip any preamble, check the wording and numbers, and log the
        # outcome. A rejected reply is dropped here and never reaches the UI.
        result = validate.screen_summary(raw_text, stats)
        if result.passed:
            self.summary_finished.emit(request_id, result.text)
        else:
            self.summary_failed.emit(
                request_id, "the generated text didn't pass the app's wording checks"
            )
