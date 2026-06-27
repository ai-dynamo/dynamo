# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coalesce concurrent async calls into batched calls of a blocking fn on one thread.

``ThreadedMicroBatcher`` is the generic execution mechanism behind
``BatchedCustomEncoder`` — it has no model/vision knowledge. It owns:

- a **dedicated worker thread** that runs an optional ``on_start`` (e.g. build +
  CUDA-graph capture) and then every ``fn`` call, so anything thread-affine
  (CUDA graphs, the current device/stream) is captured and replayed on the same
  thread;
- a **coalescing micro-batcher**: items from concurrent ``submit()`` calls are
  pooled, grouped by ``bucket_key`` (only same-bucket items batch together — the
  shape constraint a CUDA graph needs), and split into batches whose summed
  ``cost`` stays within ``max_batch_cost`` (a compute/token budget, not a raw
  count). Defaults — ``cost=1``, ``bucket_key=None`` — reduce to plain
  count-based batching up to ``max_batch_cost`` items.

The caller speaks in opaque items: ``cost`` returns an int and ``bucket_key`` a
hashable; the batcher never interprets them, so all model knowledge stays in the
caller. Final padding of a batch to a captured CUDA-graph shape is the ``fn``'s
job (it owns the model), not the batcher's.

Concurrency contract (cross-thread correctness):

- The event-loop bridge is a ``concurrent.futures.Future`` adapted with
  ``asyncio.wrap_future`` — so callers on **any** event loop work, and cancelling
  an ``await submit(...)`` actually retires the request: its not-yet-dispatched
  items are tombstoned and its admission released, and the caller only returns
  once the worker can no longer touch its items (``retired``).
- The worker runs under a **supervisor**: any unexpected crash fails every live
  request (no hung awaiter) and moves the batcher to ``FAILED`` instead of
  silently dying.
- Admission is bounded by an optional ``max_outstanding_cost``; state +
  admission are mutated together under one short lock.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import queue
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Generic, Hashable, List, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")

# Sentinel pushed onto the queue to stop the worker thread.
_SHUTDOWN = object()


class BatcherOverloaded(RuntimeError):
    """Raised by ``submit()`` when admitting the request would exceed
    ``max_outstanding_cost`` (accepted-but-incomplete cost)."""


class _State(Enum):
    NEW = auto()
    RUNNING = auto()
    CLOSING = auto()
    CLOSED = auto()
    FAILED = auto()


@dataclass(eq=False)  # identity-hashable: tracked in a set, compared by identity
class _Request(Generic[R]):
    """One ``submit()`` call: its future resolves to one result per item, in order.

    ``completion`` and ``retired`` are thread-safe ``concurrent.futures.Future``s.
    The worker is the sole finaliser (``_finalize``), so every transition is
    exactly-once.
    """

    completion: "concurrent.futures.Future[List[R]]"
    retired: "concurrent.futures.Future[None]"
    results: List[Optional[R]]
    remaining: int
    cost_total: int
    cancelled: bool = False
    done: bool = False


@dataclass
class _Work(Generic[T]):
    """A single item plus where its result belongs in the owning request."""

    item: T
    bucket: Hashable
    cost: int
    request: _Request
    index: int


class ThreadedMicroBatcher(Generic[T, R]):
    """Run ``fn(list[item]) -> list[result]`` on a dedicated thread, coalescing
    concurrent ``submit()`` calls into cost-bounded, same-bucket batches.

    Args:
        fn: Batched work; one result per item, in order. Runs on the worker thread.
        max_batch_cost: Max summed ``cost`` of a single ``fn`` batch.
        max_wait_ms: Window to gather more items after the first arrives.
        cost: Per-item cost (default 1 → count-based).
        bucket_key: Items with different keys never share a batch (default: one
            bucket).
        on_start: Optional callable run once on the worker thread before serving
            (model build / warmup); its failure surfaces from ``start()``.
        max_outstanding_cost: Optional admission ceiling on accepted-but-incomplete
            cost; ``submit()`` raises ``BatcherOverloaded`` when exceeded.
        name: Worker thread name.
        join_timeout_s: Seconds ``shutdown()`` waits for an in-flight ``fn``.
    """

    def __init__(
        self,
        fn: Callable[[List[T]], List[R]],
        *,
        max_batch_cost: int = 8,
        max_wait_ms: float = 5.0,
        cost: Callable[[T], int] = lambda _item: 1,
        bucket_key: Callable[[T], Hashable] = lambda _item: None,
        on_start: Optional[Callable[[], None]] = None,
        max_outstanding_cost: Optional[int] = None,
        name: str = "micro-batcher",
        join_timeout_s: float = 10.0,
    ) -> None:
        if max_batch_cost < 1:
            raise ValueError("max_batch_cost must be >= 1")
        self._fn = fn
        self._max_batch_cost = max_batch_cost
        self._max_wait_s = max_wait_ms / 1000.0
        self._cost = cost
        self._bucket_key = bucket_key
        self._on_start = on_start
        self._max_outstanding_cost = max_outstanding_cost
        self._name = name
        self._join_timeout_s = join_timeout_s

        self._queue: queue.Queue = queue.Queue()
        self._ready = threading.Event()
        self._terminated = threading.Event()
        self._start_error: Optional[BaseException] = None
        self._terminal_error: Optional[BaseException] = None
        # Guards _state, _outstanding, _live, and the queue-commit so a state
        # transition and its work enqueue happen atomically. Bookkeeping only —
        # never held across fn / join.
        self._lock = threading.Lock()
        self._state = _State.NEW
        self._outstanding = 0
        self._live: set[_Request] = set()
        # Non-daemon: shutdown() is the contract for a clean stop, and a daemon
        # thread could be torn down mid-fn at interpreter exit.
        self._thread: Optional[threading.Thread] = None

    # ---- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Start the worker thread, run ``on_start`` on it, and re-raise its error."""
        self._thread = threading.Thread(target=self._run, name=self._name)
        self._thread.start()
        self._ready.wait()
        if self._start_error is not None:
            # on_start ran on the thread, which then exited — mark closed so a
            # later submit() raises instead of queueing to a dead consumer.
            self.shutdown()
            raise self._start_error
        with self._lock:
            if self._state is _State.NEW:
                self._state = _State.RUNNING

    async def submit(self, items: List[T]) -> List[R]:
        """Submit a group of items; await one result per item, in order.

        Cancellation-safe: cancelling the await tombstones not-yet-dispatched
        items, releases admission, and returns only once the worker has retired
        the request (so the caller may then release the items' backing memory).
        """
        if self._thread is None:
            raise RuntimeError("ThreadedMicroBatcher.submit() called before start()")
        if not items:
            return []
        costs = [self._cost(item) for item in items]
        request: _Request = _Request(
            completion=concurrent.futures.Future(),
            retired=concurrent.futures.Future(),
            results=[None] * len(items),
            remaining=len(items),
            cost_total=sum(costs),
        )
        works = [
            _Work(item, self._bucket_key(item), c, request, i)
            for i, (item, c) in enumerate(zip(items, costs))
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
            else:  # NEW / CLOSING / CLOSED
                raise RuntimeError(
                    "ThreadedMicroBatcher.submit() called after shutdown()"
                )
            if (
                self._max_outstanding_cost is not None
                and self._outstanding + request.cost_total > self._max_outstanding_cost
            ):
                raise BatcherOverloaded(
                    f"submit cost {request.cost_total} would exceed outstanding "
                    f"budget {self._max_outstanding_cost} "
                    f"(in flight: {self._outstanding})"
                )
            self._outstanding += request.cost_total
            self._live.add(request)
            for work in works:
                self._queue.put(work)
        try:
            return await asyncio.wrap_future(request.completion)
        except asyncio.CancelledError:
            # Tombstone: the worker skips not-yet-dispatched items and finalises
            # (releasing admission). Wait — through repeated cancellation — until
            # the worker is provably done with this request's items before
            # propagating, so the caller can safely drop them.
            request.cancelled = True
            retirement = asyncio.wrap_future(request.retired)
            while not retirement.done():
                try:
                    await asyncio.shield(retirement)
                except asyncio.CancelledError:
                    continue
            raise

    def shutdown(self) -> None:
        """Stop the worker, failing not-yet-started items. Idempotent; retries the
        join if a slow ``fn`` is still in flight."""
        if self._thread is None:
            return
        to_fail: List[_Work] = []
        with self._lock:
            if self._state not in (_State.CLOSING, _State.CLOSED, _State.FAILED):
                self._state = _State.CLOSING
                to_fail = self._drain_queue_locked()  # fail queued, then signal stop
                self._queue.put(_SHUTDOWN)
        self._fail_works(to_fail, RuntimeError("ThreadedMicroBatcher shut down"))
        self._thread.join(timeout=self._join_timeout_s)
        if self._thread.is_alive():
            logger.warning(
                "ThreadedMicroBatcher(%s): worker still running after %gs; will "
                "reap on a later call",
                self._name,
                self._join_timeout_s,
            )
            return
        with self._lock:
            if self._state is _State.CLOSING:
                self._state = _State.CLOSED
            leftover = self._drain_queue_locked()  # belt-and-suspenders
        self._fail_works(leftover, RuntimeError("ThreadedMicroBatcher shut down"))

    # ---- worker thread -----------------------------------------------------

    def _run(self) -> None:
        try:
            if self._on_start is not None:
                self._on_start()  # build / warmup / CUDA-graph capture HERE
        except BaseException as exc:  # noqa: BLE001 — surface to start()
            self._start_error = exc
            self._ready.set()
            self._terminated.set()
            return
        self._ready.set()
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
                queued = self._drain_queue_locked()
            for req in live:
                self._finalize(req, exc=exc)
            self._fail_works(queued, exc)
        finally:
            self._terminated.set()

    def _collect(self) -> Optional[List[_Work]]:
        """Block for one item, then drain more within the coalescing window."""
        first = self._queue.get()
        if first is _SHUTDOWN:
            return None
        works: List[_Work] = [first]
        deadline = time.monotonic() + self._max_wait_s
        while True:
            timeout = deadline - time.monotonic()
            if timeout <= 0:
                break
            try:
                item = self._queue.get(timeout=timeout)
            except queue.Empty:
                break
            if item is _SHUTDOWN:
                self._queue.put(_SHUTDOWN)  # drain this round, stop next loop
                break
            works.append(item)
        return works

    def _dispatch(self, works: List[_Work]) -> None:
        """Group the drained items by bucket, split by cost budget, run ``fn``.

        Tombstoned (cancelled / already-finalised) items are dropped before
        batching — a cancelled request never reaches ``fn``."""
        live: List[_Work] = []
        for work in works:
            if work.request.done or work.request.cancelled:
                self._retire_tombstoned(work)
            else:
                live.append(work)
        by_bucket: dict = {}
        for work in live:
            by_bucket.setdefault(work.bucket, []).append(work)
        for group in by_bucket.values():
            batch: List[_Work] = []
            batch_cost = 0
            for work in group:
                if batch and batch_cost + work.cost > self._max_batch_cost:
                    self._run_batch(batch)
                    batch, batch_cost = [], 0
                batch.append(work)
                batch_cost += work.cost
            if batch:
                self._run_batch(batch)

    def _run_batch(self, batch: List[_Work]) -> None:
        items = [w.item for w in batch]
        try:
            results = self._fn(items)
        except (
            BaseException
        ) as exc:  # noqa: BLE001 — a bad batch must not hang awaiters
            for work in batch:
                self._finalize(work.request, exc=exc)
            return
        if len(results) != len(items):
            err = RuntimeError(
                f"batch fn returned {len(results)} results for {len(items)} items; "
                "it must return one result per item"
            )
            for work in batch:
                self._finalize(work.request, exc=err)
            return
        for work, result in zip(batch, results):
            self._deliver(work, result)

    def _deliver(self, work: _Work, result: R) -> None:
        req = work.request
        if req.done:
            return
        req.results[work.index] = result
        req.remaining -= 1
        if req.remaining == 0:
            self._finalize(req, result=list(req.results))

    def _retire_tombstoned(self, work: _Work) -> None:
        """Account a dropped item of a cancelled/finalised request; finalise at 0."""
        req = work.request
        if req.done:
            return
        req.remaining -= 1
        if req.remaining == 0:
            self._finalize(req)

    def _drain_queue_locked(self) -> List[_Work]:
        """Pop all queued works (caller holds the lock); returns them to fail
        outside the lock (``_finalize`` takes the lock)."""
        drained: List[_Work] = []
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return drained
            if item is _SHUTDOWN:
                continue
            drained.append(item)

    def _fail_works(self, works: List[_Work], exc: BaseException) -> None:
        for work in works:
            self._finalize(work.request, exc=exc)

    def _finalize(
        self,
        req: _Request,
        result: Optional[List[R]] = None,
        exc: Optional[BaseException] = None,
    ) -> None:
        """Resolve a request exactly once: release admission, settle ``completion``
        (unless the caller already cancelled it), and signal ``retired``."""
        with self._lock:
            if req.done:
                return
            req.done = True
            self._live.discard(req)
            self._outstanding -= req.cost_total
        try:
            if exc is not None:
                req.completion.set_exception(exc)
            elif not req.cancelled:
                req.completion.set_result(result)
            # cancelled + no error: leave completion (already cancelled by the waiter)
        except concurrent.futures.InvalidStateError:
            pass  # caller already cancelled the future
        try:
            req.retired.set_result(None)  # caller's cancel path awaits this
        except concurrent.futures.InvalidStateError:
            pass
