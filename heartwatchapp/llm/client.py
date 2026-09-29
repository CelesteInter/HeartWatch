"""The one place in the app that talks to Ollama.

Plain-language overview: Ollama is a program that runs a small language
model on Dylan's own laptop (nothing goes over the internet). This module
wraps the handful of calls the rest of the app needs -- "is it running and
ready?" and "ask it to phrase these numbers" -- so that no other file has
to import the `ollama` package or know anything about how it works. If we
ever swap models or add retry logic, this is the only file that changes.

Model choice and memory: MODEL_NAME is gemma3:1b specifically because it
fits comfortably next to TensorFlow and PySide6 on an 8 GB M2 machine.
MODEL_OPTIONS keeps num_ctx small (2048) because context window is where
memory actually goes -- see the handoff doc for the full reasoning. Do not
raise num_ctx or point this at a bigger model without re-checking memory
headroom on the target machine.

What this model is (and is not) for: it phrases numbers Python has already
computed. It never does arithmetic on sensor data and never overrides the
activity classifier. See llm/prompts.py's SYSTEM_PROMPT for the rule as
stated to the model itself.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx
import ollama

MODEL_NAME = "gemma3:1b"
MODEL_OPTIONS = {"num_ctx": 2048}
# Options for session summaries only. Low temperature makes the model pick
# its most likely wording instead of getting creative, which is what we
# want when all it should do is restate numbers. Same num_ctx as above;
# MODEL_OPTIONS itself is left alone for any other caller.
SUMMARY_OPTIONS = {**MODEL_OPTIONS, "temperature": 0.2}

# One shared client for the app's lifetime, not one per call -- mirrors how
# data/writer.py keeps a single SQLite connection alive instead of opening
# a new one per query.
_client = ollama.AsyncClient()


@dataclass
class HealthStatus:
    """Result of health_check(). `message` is written to be shown as-is in
    the GUI status bar -- it already says what the user should do."""

    daemon_reachable: bool
    model_pulled: bool
    message: str

    @property
    def ready(self) -> bool:
        """True only when both the Ollama app is running and gemma3:1b is
        actually downloaded -- the two things that have to be true before
        any summary request can succeed."""
        return self.daemon_reachable and self.model_pulled


# Errors that mean "couldn't reach the Ollama app at all". The ollama
# package (0.6.x) wraps connection failures in Python's built-in
# ConnectionError; older versions let httpx's RequestError through.
CONNECTION_ERRORS = (ConnectionError, httpx.RequestError)


# The three statuses the topbar's AI pill can show. Built in one place so
# the startup check and a failed/successful summary later on (see
# llm/worker.py) show exactly the same hover text.
def unreachable_status(exc: Exception) -> HealthStatus:
    return HealthStatus(
        daemon_reachable=False,
        model_pulled=False,
        message=(
            f"Ollama isn't reachable ({exc}). Install the Ollama app from "
            "ollama.com, then make sure it's running."
        ),
    )


def model_missing_status() -> HealthStatus:
    return HealthStatus(
        daemon_reachable=True,
        model_pulled=False,
        message=f"Ollama is running, but {MODEL_NAME} isn't pulled yet. Run: ollama pull {MODEL_NAME}",
    )


def ready_status() -> HealthStatus:
    return HealthStatus(daemon_reachable=True, model_pulled=True, message=f"{MODEL_NAME} ready.")


async def health_check() -> HealthStatus:
    """Checks whether Ollama is running and whether gemma3:1b has been
    pulled, without ever raising -- this runs once at app startup, and a
    startup check that can throw would defeat the point of having it. Any
    failure is reported through the returned message instead."""
    try:
        response = await _client.list()
    except (*CONNECTION_ERRORS, ollama.ResponseError) as exc:
        return unreachable_status(exc)
    except Exception as exc:  # noqa: BLE001 - a health check must never raise
        return HealthStatus(
            daemon_reachable=False,
            model_pulled=False,
            message=f"Couldn't check Ollama's status ({exc}).",
        )

    pulled_names = {m.model for m in response.models if m.model}
    if MODEL_NAME not in pulled_names:
        return model_missing_status()
    return ready_status()


async def stream_chat(messages: list[dict]) -> AsyncIterator[str]:
    """Sends a conversation to gemma3:1b and yields its reply piece by
    piece, as each piece is generated, instead of waiting for the whole
    thing. Callers see partial text sooner, which matters because a full
    reply can take several seconds on this hardware.

    Raises ollama.ResponseError (e.g. the model isn't pulled) or an
    httpx connection error if Ollama isn't reachable -- callers decide how
    to fall back; this function doesn't swallow errors so a caller that
    already showed a few chunks knows generation stopped partway through.
    """
    stream = await _client.chat(
        model=MODEL_NAME,
        messages=messages,
        stream=True,
        options=MODEL_OPTIONS,
    )
    async for part in stream:
        content = part.message.content
        if content:
            yield content


async def chat(messages: list[dict], options: dict | None = None) -> str:
    """Like stream_chat, but waits for the whole reply and hands it back as
    one string. This is what session summaries use: the reply has to be
    checked (llm/validate.py) before any of it is shown, so there's no
    point displaying it piece by piece.

    `options` defaults to MODEL_OPTIONS; summaries pass SUMMARY_OPTIONS.
    Raises the same errors as stream_chat."""
    response = await _client.chat(
        model=MODEL_NAME,
        messages=messages,
        stream=False,
        options=options if options is not None else MODEL_OPTIONS,
    )
    return response.message.content or ""
