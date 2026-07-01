# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coalesce concurrent async calls into batched calls of a blocking fn on one thread.

``ThreadedMicroBatcher`` is the generic execution mechanism behind
``AsyncVisionEncoder`` — it has no model/vision knowledge and stays torch-free.
It owns:

- a **dedicated worker thread** that runs an optional ``on_start`` (e.g. build +
  CUDA-graph capture) and then every ``fn`` call, so anything thread-affine
  (CUDA graphs, the current device/stream) is captured and replayed on the same
  thread; an optional ``on_stop`` runs on that same thread at teardown;
- a **coalescing micro-batcher**: items from concurrent ``submit()`` calls are
  pooled and split into batches whose summed ``cost`` stays within
  ``max_batch_cost`` (a compute/token budget, not a raw count). Packing is
  **one-dimensional** — by scalar ``cost`` alone; the batcher never inspects item
  shape.

The caller speaks in opaque items plus a per-item scalar ``cost`` (int), computed
once off-thread (see ``Preprocessed``); the batcher never interprets the items, so
all model knowledge stays in the caller.

Coalescing window — **eager drain-on-completion, no timer**: whenever the worker
is free it pulls everything queued and runs it, then repeats. A lone item runs on
the next free iteration (liveness for free); batch size auto-scales with load
(arrivals during ``fn`` pile up and are scooped next loop).

Concurrency contract (cross-thread correctness):

- The event-loop bridge is a ``concurrent.futures.Future`` adapted with
  ``asyncio.wrap_future``, so callers on **any** event loop work.
  TODO: cancelling an ``await submit(...)`` does not yet tombstone the request —
  its items still run through ``fn`` (the result is discarded) and admission frees
  only when they finish. Add cancel-aware tombstoning + retirement here.
- A request is finalised **per item**: admission release and ``completion`` fire
  exactly once, only after *all* of the request's items have been delivered or
  failed — never on the first item of a multi-batch request. Every request-state
  transition is under one short lock, so the worker and a concurrent
  ``shutdown()`` cannot race on ``remaining``.
- The worker runs under a **supervisor**: any unexpected crash fails every live
  request (no hung awaiter) and moves the batcher to ``FAILED``.

TODO: no admission backpressure — ``submit()`` always accepts. Add a bound on
accepted-but-incomplete cost if a producer can outrun the encoder.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import queue
import threading
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Generic, List, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")

# Sentinel pushed onto the queue to stop the worker thread.
_SHUTDOWN = object()
# Sentinel distinguishing "no result" (failed / tombstoned item) from a real
# ``None`` result returned by ``fn``.
_NO_RESULT = object()


class _State(Enum):
    NEW = auto()
    RUNNING = auto()
    CLOSED = auto()  # shutdown begun — no further submits (worker may still be exiting)
    FAILED = auto()


@dataclass(eq=False)  # identity-hashable: tracked in a set, compared by identity
class _Request(Generic[R]):
    """One ``submit()`` call: its future resolves to one result per item, in order.

    ``completion`` is a thread-safe ``concurrent.futures.Future``. All mutation
    (``remaining`` / ``done`` / ``error`` / ``results``) happens under the batcher
    lock, so the worker and a concurrent ``shutdown()`` never race.
    """

    completion: "concurrent.futures.Future[List[R]]"
    results: List[Optional[R]]
    remaining: int
    error: Optional[BaseException] = None
    done: bool = False


@dataclass
class _Work(Generic[T]):
    """A single item plus where its result belongs in the owning request."""

    item: T
    cost: int
    request: _Request
    index: int


class ThreadedMicroBatcher(Generic[T, R]):
    """Run ``fn(list[item]) -> list[result]`` on a dedicated thread, coalescing
    concurrent ``submit()`` calls into cost-bounded batches.

    Args:
        fn: Batched work; one result per item, in order. Runs on the worker thread.
        max_batch_cost: Max summed ``cost`` of a single ``fn`` batch (>= 1).
            ``None`` (default) ⇒ **pass-through**: no cap — the whole drained set
            runs as one ``fn`` call (``cost`` ignored).
        on_start: Optional callable run once on the worker thread before serving
            (model build / warmup); its failure surfaces from ``start()``.
        on_stop: Optional callable run once on the worker thread at teardown (after
            the serving loop ends), iff ``on_start`` succeeded. Its failure is
            logged, never raised.
        name: Worker thread name.
        join_timeout_s: Seconds ``shutdown()`` waits for an in-flight ``fn``.
    """

    def __init__(
        self,
        fn: Callable[[List[T]], List[R]],
        *,
        max_batch_cost: Optional[int] = None,
        on_start: Optional[Callable[[], None]] = None,
        on_stop: Optional[Callable[[], None]] = None,
        name: str = "micro-batcher",
        join_timeout_s: float = 10.0,
    ) -> None:
        if max_batch_cost is not None and max_batch_cost < 1:
            raise ValueError("max_batch_cost must be >= 1 (or None for pass-through)")
        self._fn = fn
        self._max_batch_cost = max_batch_cost
        self._on_start = on_start
        self._on_stop = on_stop
        self._name = name
        self._join_timeout_s = join_timeout_s

        self._queue: queue.Queue = queue.Queue()
        # Completed by the worker once on_start settles: result(None) on success,
        # set_exception(exc) on failure. start() blocks on it and re-raises —
        # one primitive in place of a ready-Event + a start-error field.
        self._started: "concurrent.futures.Future[None]" = concurrent.futures.Future()
        self._terminal_error: Optional[BaseException] = None
        # Guards _state, _live, every _Request transition, and the queue-commit so
        # a state change and its work enqueue happen atomically. Bookkeeping only
        # — never held across fn / join.
        self._lock = threading.Lock()
        self._state = _State.NEW
        self._live: set[_Request] = set()
        self._thread: Optional[threading.Thread] = None

    # ---- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Start the worker thread, run ``on_start`` on it, and re-raise its error.

        Single-shot: once ``start()`` has been called — even if the thread failed
        to spawn — a second call raises rather than spawning a second consumer /
        orphaning the first thread."""
        with self._lock:
            # Guard on state, not `_thread`: a failed spawn moves to FAILED with
            # `_thread` still None, and rejecting any non-NEW state here stops a
            # retry from spawning a second (orphaned) worker.
            if self._state is not _State.NEW:
                raise RuntimeError("ThreadedMicroBatcher.start() called twice")
            # Non-daemon: a clean stop is via shutdown(); a daemon worker could be
            # torn down mid-fn at interpreter exit. Pin daemon=False explicitly so
            # it never inherits a daemon creator thread. Start under the lock and
            # publish _thread only after a successful start(), so a racing
            # shutdown() can never join() an unstarted thread.
            thread = threading.Thread(target=self._run, name=self._name, daemon=False)
            try:
                thread.start()
            except BaseException:
                self._state = _State.FAILED
                raise
            self._thread = thread
        try:
            self._started.result()  # blocks until on_start settles; re-raises it
        except BaseException:
            # on_start ran on the thread, which then exited — mark closed so a
            # later submit() raises instead of queueing to a dead consumer.
            self.shutdown()
            raise
        with self._lock:
            if self._state is _State.NEW:
                self._state = _State.RUNNING
            elif self._state is not _State.RUNNING:
                # a concurrent shutdown() closed us during startup
                raise RuntimeError("ThreadedMicroBatcher shut down during start()")

    async def submit(
        self,
        items: List[T],
        costs: Optional[List[int]] = None,
    ) -> List[R]:
        """Submit a group of items; await one result per item, in order.

        ``costs`` is computed off-thread by the caller (see ``Preprocessed``);
        when omitted it defaults to ``1`` per item (plain count-based batching).
        Batching is one-dimensional — the batcher packs by ``cost`` alone and
        never inspects item shape.

        TODO: not cancel-aware. Cancelling the await abandons the result, but the
        items still run through ``fn`` and admission frees only when they finish.
        """
        if self._thread is None:
            raise RuntimeError("ThreadedMicroBatcher.submit() called before start()")
        if not items:
            return []
        if costs is None:
            costs = [1] * len(items)
        elif len(costs) != len(items):
            raise ValueError(f"costs has {len(costs)} entries for {len(items)} items")
        for c in costs:
            if not isinstance(c, int) or isinstance(c, bool) or c < 1:
                raise ValueError(f"cost must be a positive int, got {c!r}")
            if self._max_batch_cost is not None and c > self._max_batch_cost:
                raise ValueError(
                    f"item cost {c} exceeds max_batch_cost {self._max_batch_cost}; "
                    "it has no batch it can fit"
                )
        request: _Request = _Request(
            completion=concurrent.futures.Future(),
            results=[None] * len(items),
            remaining=len(items),
        )
        works = [
            _Work(item, c, request, i) for i, (item, c) in enumerate(zip(items, costs))
        ]
        # State check + admission + queue-commit under one lock so a concurrent
        # shutdown() cannot strand the request and capacity cannot leak.
        with self._lock:
            if self._state is _State.RUNNING:
                pass
            elif self._state is _State.FAILED:
                raise RuntimeError(
                    "ThreadedMicroBatcher.submit() after worker failure"
                ) from self._terminal_error
            else:  # NEW / CLOSED
                raise RuntimeError(
                    "ThreadedMicroBatcher.submit() called after shutdown()"
                )
            self._live.add(request)
            for work in works:
                self._queue.put(work)
        return await asyncio.wrap_future(request.completion)

    def shutdown(self) -> None:
        """Stop the worker, failing not-yet-run items. Idempotent and best-effort:
        a slow in-flight ``fn`` keeps running and the worker exits on its own once
        it returns and reads the stop signal."""
        if self._thread is None:
            return
        to_fail: List[_Work] = []
        with self._lock:
            if self._state not in (_State.CLOSED, _State.FAILED):
                self._state = _State.CLOSED
                to_fail = self._drain_queue_locked()  # fail queued, then signal stop
                self._queue.put(_SHUTDOWN)
        # Admission is now closed and the queue drained under the lock, so no new
        # work can arrive; failing the drained set covers everything not already
        # in-flight on the worker. No post-join re-drain is needed.
        for work in to_fail:
            self._consume(work, error=RuntimeError("ThreadedMicroBatcher shut down"))
        self._thread.join(timeout=self._join_timeout_s)
        if self._thread.is_alive():
            logger.warning(
                "ThreadedMicroBatcher(%s): worker still finishing an in-flight fn "
                "after %gs; it will exit on its own.",
                self._name,
                self._join_timeout_s,
            )

    # ---- worker thread -----------------------------------------------------

    def _run(self) -> None:
        try:
            if self._on_start is not None:
                self._on_start()  # build / warmup / CUDA-graph capture HERE
        except BaseException as exc:  # noqa: BLE001 — surface to start()
            self._started.set_exception(exc)
            return
        self._started.set_result(None)
        try:
            while True:
                works = self._collect()
                if works is None:
                    return
                self._dispatch(works)
        except BaseException as exc:  # noqa: BLE001 — supervisor: never hang awaiters
            logger.exception(
                "ThreadedMicroBatcher(%s): worker crashed; failing live requests",
                self._name,
            )
            with self._lock:
                self._state = _State.FAILED
                self._terminal_error = exc
                live = list(self._live)
                self._drain_queue_locked()  # clear queue; those reqs are in `live`
            for req in live:
                self._abort(req, exc)
        finally:
            # on_stop runs on the actor thread (so CUDA teardown is same-thread),
            # only after on_start succeeded (this finally is unreachable otherwise).
            self._run_on_stop()

    def _run_on_stop(self) -> None:
        if self._on_stop is None:
            return
        try:
            self._on_stop()
        except BaseException:  # noqa: BLE001 — teardown best-effort, never raise
            logger.exception(
                "ThreadedMicroBatcher(%s): on_stop raised during teardown",
                self._name,
            )

    def _collect(self) -> Optional[List[_Work]]:
        """Block for one item, then eager-drain everything else already queued.

        No timed hold: pull only what is immediately available, then run."""
        first = self._queue.get()
        if first is _SHUTDOWN:
            return None
        works: List[_Work] = [first]
        # Eager drain: pull everything immediately available, then run.
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item is _SHUTDOWN:
                self._queue.put(_SHUTDOWN)  # drain this round, stop next loop
                break
            works.append(item)
        return works

    def _dispatch(self, works: List[_Work]) -> None:
        """Split live items by cost budget, run ``fn`` (one-dimensional packing).

        Tombstoned (failed / done) items are dropped before batching —
        an already-failed request never reaches ``fn``."""
        live: List[_Work] = []
        for work in works:
            if self._is_tombstoned(work.request):
                self._consume(work)  # account the dropped item
            else:
                live.append(work)
        if not live:
            return
        if self._max_batch_cost is None:
            # Pass-through: no cost cap — the whole drained set is one batch.
            self._run_batch(live)
            return
        batch: List[_Work] = []
        batch_cost = 0
        for work in live:
            if batch and batch_cost + work.cost > self._max_batch_cost:
                self._run_batch(batch)
                batch, batch_cost = [], 0
            batch.append(work)
            batch_cost += work.cost
        if batch:
            self._run_batch(batch)

    def _run_batch(self, batch: List[_Work]) -> None:
        # Re-filter immediately before fn: a request may have been failed by a
        # sibling batch after grouping, so a failed request's remaining items are
        # dropped at each fn call rather than only at the per-dispatch snapshot.
        runnable: List[_Work] = []
        for work in batch:
            if self._is_tombstoned(work.request):
                self._consume(work)
            else:
                runnable.append(work)
        if not runnable:
            return
        # Once shutdown/failure has begun, do not START new fn calls: fail these
        # collected-but-not-yet-run items with the shutdown error. The fn already
        # in flight when shutdown() was called still finishes (it is past this
        # check); this just bounds work after teardown intent to that one batch.
        with self._lock:
            stopping = self._state is not _State.RUNNING
        if stopping:
            for work in runnable:
                self._consume(
                    work, error=RuntimeError("ThreadedMicroBatcher shut down")
                )
            return
        items = [w.item for w in runnable]
        try:
            results = self._fn(items)
        except (
            BaseException
        ) as exc:  # noqa: BLE001 — a bad batch must not hang awaiters
            for work in runnable:
                self._consume(work, error=exc)
            return
        if len(results) != len(items):
            err = RuntimeError(
                f"batch fn returned {len(results)} results for {len(items)} items; "
                "it must return one result per item"
            )
            for work in runnable:
                self._consume(work, error=err)
            return
        for work, result in zip(runnable, results):
            self._consume(work, result=result)

    def _is_tombstoned(self, req: _Request) -> bool:
        """True once the request must not send further items to ``fn``."""
        return req.done or req.error is not None

    def _consume(self, work: _Work, result: object = _NO_RESULT, error=None) -> None:
        """Account one item of a request (delivered / failed / tombstoned).

        Decrements ``remaining`` under the lock and finalises the request only
        when the last item is consumed — so admission release and ``completion``
        are exactly-once even when items span batches or threads."""
        req = work.request
        finalize = False
        with self._lock:
            if req.done:
                return
            if error is not None and req.error is None:
                req.error = error
            elif result is not _NO_RESULT and req.error is None:
                req.results[work.index] = result
            req.remaining -= 1
            if req.remaining == 0:
                req.done = True
                self._live.discard(req)
                finalize = True
        if finalize:
            self._complete(req)

    def _abort(self, req: _Request, exc: BaseException) -> None:
        """Force-finalise a live request (worker crash): items may be lost, so do
        not wait for ``remaining``."""
        with self._lock:
            if req.done:
                return
            req.done = True
            if req.error is None:
                req.error = exc
            self._live.discard(req)
        self._complete(req)

    def _complete(self, req: _Request) -> None:
        """Settle a finalised request's futures (outside the lock; idempotent)."""
        try:
            if req.error is not None:
                req.completion.set_exception(req.error)
            else:
                req.completion.set_result(list(req.results))
        except concurrent.futures.InvalidStateError:
            pass  # caller already cancelled / abandoned the future

    def _drain_queue_locked(self) -> List[_Work]:
        """Pop all queued works (caller holds the lock); returns them to consume
        outside the lock (``_consume`` re-takes the lock)."""
        drained: List[_Work] = []
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return drained
            if item is _SHUTDOWN:
                # Preserve the stop signal so a still-running worker reads it and
                # exits — draining must never strand the worker.
                self._queue.put(_SHUTDOWN)
                return drained
            drained.append(item)
