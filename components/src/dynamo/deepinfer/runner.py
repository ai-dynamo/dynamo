# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-threaded owner of the native DeepInfer engine.

DeepInfer is an offline engine: its ``step()`` loop is not thread-safe, and the
``RequestOutput`` objects it returns point into buffers that are recycled by the
next step. :class:`EngineRunner` therefore confines every native call (add,
step, abort, close) to one dedicated thread and converts each output into an
owned :class:`TokenOutput` snapshot before anything leaves that thread.

Requests are submitted from the event loop through a thread-safe command
queue. Results are pushed to a per-request ``sink`` callable that runs on the
runner thread; the caller is expected to hop back to its event loop there.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# What a sink receives: a batch of snapshots for the request, or the exception
# that terminated it.
Sink = Callable[["list[TokenOutput] | BaseException"], None]


@dataclass(frozen=True)
class TokenOutput:
    """Owned copy of one native ``RequestOutput``."""

    request_id: str
    token_id: int
    finished: bool
    finish_reason: str
    prompt_len: int
    logprob: Optional[float] = None
    top_logprobs: list[float] = field(default_factory=list)
    top_token_ids: list[int] = field(default_factory=list)


def _as_list(value: Any) -> list:
    if value is None:
        return []
    tolist = getattr(value, "tolist", None)
    return list(tolist() if callable(tolist) else value)


def snapshot(output: Any) -> TokenOutput:
    """Copy every field out of a (cloned) native output."""
    logprob = output.token_logprobs
    return TokenOutput(
        request_id=str(output.request_id),
        token_id=int(output.token_id),
        finished=bool(output.finished),
        finish_reason=str(output.finish_reason or ""),
        prompt_len=int(output.prompt_len),
        logprob=None if logprob is None else float(logprob),
        top_logprobs=[float(x) for x in _as_list(output.top_logprobs)],
        top_token_ids=[int(x) for x in _as_list(output.top_token_ids)],
    )


class RunnerClosed(RuntimeError):
    """Raised by :meth:`EngineRunner.submit` once the runner stopped or failed."""


class EngineRunner:
    """Drives ``engine.step()`` on a dedicated thread, then closes the engine there.

    ``engine`` is DeepInfer's ``LLMEngine``; ``clone`` turns a raw output
    pointer returned by ``step()`` into an object that outlives the next step
    (``RequestOutput.clone`` in production).
    """

    def __init__(self, engine: Any, clone: Callable[[Any], Any]):
        self._engine = engine
        self._clone = clone
        self._commands: queue.SimpleQueue[tuple] = queue.SimpleQueue()
        self._lock = threading.Lock()
        self._closed = False
        self._failure: Optional[BaseException] = None
        self._sinks: dict[str, Sink] = {}
        self._needs_flush = False
        self._thread = threading.Thread(
            target=self._run, name="deepinfer-engine", daemon=True
        )

    @property
    def failure(self) -> Optional[BaseException]:
        return self._failure

    def start(self) -> None:
        self._thread.start()

    def submit(
        self, request_id: str, token_ids: Any, sampling_params: Any, sink: Sink
    ) -> None:
        self._post(("add", request_id, token_ids, sampling_params, sink))

    def abort(self, request_id: str) -> None:
        try:
            self._post(("abort", request_id))
        except RunnerClosed:
            pass

    def stop(self, timeout: Optional[float] = None) -> bool:
        """Ask the thread to exit and wait for it. Safe to call repeatedly."""
        with self._lock:
            already_closed, self._closed = self._closed, True
        if not already_closed:
            self._commands.put(("stop",))
        if self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout)
        return not self._thread.is_alive()

    def _post(self, command: tuple) -> None:
        with self._lock:
            if self._closed:
                raise RunnerClosed(
                    "DeepInfer engine is not running"
                    if self._failure is None
                    else f"DeepInfer engine failed: {self._failure!r}"
                )
            self._commands.put(command)

    # -- runner thread ---------------------------------------------------

    def _run(self) -> None:
        try:
            while self._drain():
                if self._engine.has_unfinished_requests():
                    self._needs_flush = True
                    self._dispatch(self._engine.step())
                elif self._needs_flush:
                    self._needs_flush = False
                    self._dispatch(self._engine.flush_inflight_reqs())
        except BaseException as exc:  # noqa: BLE001 - fail every waiter
            logger.exception("DeepInfer engine loop failed")
            with self._lock:
                self._failure = exc
                self._closed = True
            self._fail_all(exc)
        else:
            self._fail_all(RunnerClosed("DeepInfer engine stopped"))
        finally:
            self._close_engine()

    def _drain(self) -> bool:
        """Apply queued commands; block only when there is nothing to step."""
        idle = not self._sinks and not self._needs_flush
        while True:
            try:
                command = self._commands.get() if idle else self._commands.get_nowait()
            except queue.Empty:
                return True
            idle = False
            kind = command[0]
            if kind == "stop":
                return False
            if kind == "add":
                self._add(*command[1:])
            elif kind == "abort":
                self._abort(command[1])

    def _add(
        self, request_id: str, token_ids: Any, sampling_params: Any, sink: Sink
    ) -> None:
        try:
            self._engine.add_request(
                user_id="0",
                request_id=request_id,
                sampling_params=sampling_params,
                prompt_token_ids=token_ids,
            )
        except Exception as exc:  # noqa: BLE001 - reject only this request
            sink(exc)
            return
        self._sinks[request_id] = sink

    def _abort(self, request_id: str) -> None:
        if self._sinks.pop(request_id, None) is None:
            return
        self._engine.abort_request(request_id)

    def _dispatch(self, raw_outputs: Any) -> None:
        batches: dict[str, list[TokenOutput]] = defaultdict(list)
        for raw in raw_outputs:
            out = snapshot(self._clone(raw))
            batches[out.request_id].append(out)
        for request_id, batch in batches.items():
            sink = self._sinks.get(request_id)
            if sink is None:  # aborted while this step was in flight
                continue
            if batch[-1].finished:
                del self._sinks[request_id]
            sink(batch)

    def _close_engine(self) -> None:
        try:
            self._engine.close()
        except Exception:  # noqa: BLE001 - nothing left to recover
            logger.exception("DeepInfer engine close failed")

    def _fail_all(self, exc: BaseException) -> None:
        sinks, self._sinks = self._sinks, {}
        for sink in sinks.values():
            try:
                sink(exc)
            except Exception:  # noqa: BLE001
                logger.debug("sink failed while reporting %r", exc, exc_info=True)
