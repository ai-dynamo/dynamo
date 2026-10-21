# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Event-loop driven owner of the native DeepInfer engine.

DeepInfer is an offline engine: its ``step()`` loop is not thread-safe, and the
``RequestOutput`` objects it returns point into buffers that are recycled by the
next step. :class:`EngineRunner` therefore runs every native call on one
single-worker executor thread: engine construction, ``add_request``, ``step``,
``abort_request`` and ``close``. Pinning construction and execution to the same
thread also keeps per-thread CUDA/torch state consistent.

A pump task on the event loop drives the engine. Each tick awaits one executor
job that steps the engine, clones and snapshots the outputs on the engine
thread, and returns plain :class:`TokenOutput` objects; the pump then pushes
them into the per-request ``asyncio.Queue`` returned by :meth:`submit`. All
request bookkeeping therefore lives on the event loop and needs no locks.

Two rules keep tokens from being lost:

* A step that has started cannot be interrupted, and its tokens are consumed
  from the engine. The pump is never cancelled to stop it: :meth:`stop` sets a
  flag, the in-flight step finishes and is delivered, and only then does the
  pump close the engine.
* :meth:`submit` does not wait for ``add_request``. A caller that goes away
  right after submitting therefore cannot leave the add half-done; its
  :meth:`abort` is queued behind the add on the same single worker.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Pause between ticks when requests are registered but the engine has nothing
# to step, so a request the engine lost cannot make the pump spin.
_IDLE_BACKOFF_S = 0.005

EngineFactory = Callable[[], "tuple[Any, Callable[[Any], Any]]"]


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
    """Drives DeepInfer's ``LLMEngine`` from the event loop via one worker thread.

    Usage::

        runner = EngineRunner()
        await runner.start(lambda: (engine, RequestOutput.clone))
        results = runner.submit(request_id, token_ids, sampling_params)
        batch = await results.get()   # list[TokenOutput] | BaseException
        ...
        await runner.stop()
    """

    def __init__(self, thread_name: str = "deepinfer-engine"):
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=thread_name
        )
        self._engine: Any = None
        self._clone: Callable[[Any], Any] = lambda raw: raw
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._wake: Optional[asyncio.Event] = None
        self._pump_task: Optional[asyncio.Task] = None
        self._streams: dict[str, asyncio.Queue] = {}
        self._needs_flush = False
        self._closed = False
        self._failure: Optional[BaseException] = None

    @property
    def failure(self) -> Optional[BaseException]:
        return self._failure

    async def start(self, factory: EngineFactory) -> None:
        """Build the engine on the worker thread, then start the pump.

        ``factory`` returns ``(engine, clone)``, where ``clone`` turns a raw
        output pointer from ``engine.step()`` into an object that outlives the
        next step (``RequestOutput.clone`` in production).
        """
        self._loop = asyncio.get_running_loop()
        try:
            self._engine, self._clone = await self._loop.run_in_executor(
                self._executor, factory
            )
        except BaseException:
            self._closed = True
            self._executor.shutdown(wait=False)
            raise
        self._wake = asyncio.Event()
        # The loop keeps only a weak reference to tasks: hold the pump.
        self._pump_task = asyncio.create_task(self._pump(), name="deepinfer-pump")
        self._pump_task.add_done_callback(self._on_pump_done)

    def submit(
        self, request_id: str, token_ids: Any, sampling_params: Any
    ) -> "asyncio.Queue[list[TokenOutput] | BaseException]":
        """Register a request and return the queue its results arrive on.

        The queue yields batches of :class:`TokenOutput`; the batch whose last
        item is ``finished`` is the final one. An exception object is put on
        the queue instead if the request fails. Must be called on the loop.
        """
        self._check_open()
        if request_id in self._streams:
            raise ValueError(f"duplicate request id {request_id!r}")
        results: asyncio.Queue = asyncio.Queue()
        self._streams[request_id] = results
        assert self._loop is not None and self._wake is not None
        added = self._loop.run_in_executor(
            self._executor, self._add_job, request_id, token_ids, sampling_params
        )
        added.add_done_callback(lambda fut: self._on_added(request_id, fut))
        self._wake.set()
        return results

    def abort(self, request_id: str) -> None:
        """Stop delivering to ``request_id`` and free its native request.

        The native abort is queued on the worker thread, behind any add still
        pending for the same request. Unknown or finished ids are ignored.
        """
        if self._streams.pop(request_id, None) is None or self._closed:
            return
        assert self._loop is not None
        aborted = self._loop.run_in_executor(
            self._executor, self._abort_job, request_id
        )
        aborted.add_done_callback(self._log_abort_failure)

    async def stop(self, timeout: Optional[float] = None) -> bool:
        """Finish the in-flight step, fail pending requests, close the engine.

        Returns False if the pump did not finish within ``timeout`` (for
        example a native step that never returns). Safe to call repeatedly.
        """
        self._closed = True
        task = self._pump_task
        if task is None:
            self._executor.shutdown(wait=False)
            return True
        assert self._wake is not None
        self._wake.set()
        _, pending = await asyncio.wait({task}, timeout=timeout)
        return not pending

    def _check_open(self) -> None:
        if self._closed or self._pump_task is None:
            raise RunnerClosed(
                "DeepInfer engine is not running"
                if self._failure is None
                else f"DeepInfer engine failed: {self._failure!r}"
            )

    # -- event loop ------------------------------------------------------

    async def _pump(self) -> None:
        assert self._loop is not None and self._wake is not None
        try:
            while not self._closed:
                if not self._streams and not self._needs_flush:
                    self._wake.clear()
                    await self._wake.wait()
                    continue
                # Shielded so that cancelling the pump can never abandon a step
                # whose tokens the engine has already consumed.
                batches, stepped = await asyncio.shield(
                    self._loop.run_in_executor(self._executor, self._step_once)
                )
                self._needs_flush = stepped
                self._deliver(batches)
                if not batches and not stepped and self._streams:
                    await asyncio.sleep(_IDLE_BACKOFF_S)
        except Exception as exc:  # noqa: BLE001 - fail every waiter
            logger.exception("DeepInfer engine loop failed")
            self._failure = exc
        finally:
            self._closed = True
            self._fail_all(self._failure or RunnerClosed("DeepInfer engine stopped"))
            await self._close_engine()

    def _deliver(self, batches: dict[str, "list[TokenOutput]"]) -> None:
        for request_id, batch in batches.items():
            results = self._streams.get(request_id)
            if results is None:  # aborted while this step was in flight
                continue
            if batch[-1].finished:
                del self._streams[request_id]
            results.put_nowait(batch)

    def _on_added(self, request_id: str, added: "asyncio.Future[None]") -> None:
        if added.cancelled():
            error: Optional[BaseException] = RunnerClosed("DeepInfer engine stopped")
        else:
            error = added.exception()
        if error is None:
            return
        results = self._streams.pop(request_id, None)
        if results is not None:  # not aborted meanwhile
            results.put_nowait(error)

    def _on_pump_done(self, task: "asyncio.Task[None]") -> None:
        if not task.cancelled() and task.exception() is not None:
            logger.error("DeepInfer pump task died", exc_info=task.exception())

    @staticmethod
    def _log_abort_failure(aborted: "asyncio.Future[None]") -> None:
        if not aborted.cancelled() and aborted.exception() is not None:
            logger.warning("DeepInfer abort_request failed: %r", aborted.exception())

    def _fail_all(self, exc: BaseException) -> None:
        streams, self._streams = self._streams, {}
        for results in streams.values():
            results.put_nowait(exc)

    async def _close_engine(self) -> None:
        assert self._loop is not None
        try:
            await self._loop.run_in_executor(self._executor, self._engine.close)
        except Exception:  # noqa: BLE001 - nothing left to recover
            logger.exception("DeepInfer engine close failed")
        finally:
            self._executor.shutdown(wait=False)

    # -- worker thread: the only place the native engine is touched ------

    def _add_job(self, request_id: str, token_ids: Any, sampling_params: Any) -> None:
        self._engine.add_request(
            user_id="0",
            request_id=request_id,
            sampling_params=sampling_params,
            prompt_token_ids=token_ids,
        )

    def _abort_job(self, request_id: str) -> None:
        self._engine.abort_request(request_id)

    def _step_once(self) -> "tuple[dict[str, list[TokenOutput]], bool]":
        """One engine tick: step if work remains, else flush async leftovers.

        Returns the snapshots grouped by request and whether a step ran.
        """
        engine = self._engine
        if engine.has_unfinished_requests():
            raw_outputs, stepped = engine.step(), True
        else:
            raw_outputs, stepped = engine.flush_inflight_reqs(), False
        batches: dict[str, list[TokenOutput]] = defaultdict(list)
        for raw in raw_outputs:
            out = snapshot(self._clone(raw))
            batches[out.request_id].append(out)
        return dict(batches), stepped
