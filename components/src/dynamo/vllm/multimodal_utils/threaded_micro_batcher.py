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
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Generic, Hashable, List, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")

# Sentinel pushed onto the queue to stop the worker thread.
_SHUTDOWN = object()


@dataclass
class _Request(Generic[R]):
    """One ``submit()`` call: its future resolves to one result per item, in order."""

    loop: asyncio.AbstractEventLoop
    future: asyncio.Future
    results: List[Optional[R]]
    remaining: int


@dataclass
class _Work(Generic[T, R]):
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
        cost: Per-item cost (default 1 → count-based).
        bucket_key: Items with different keys never share a batch (default: one
            bucket).
        on_start: Optional callable run once on the worker thread before serving
            (model build / warmup); its failure surfaces from ``start()``.
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
        name: str = "micro-batcher",
        join_timeout_s: float = 10.0,
    ) -> None:
        self._fn = fn
        self._max_batch_cost = max_batch_cost
        self._max_wait_s = max_wait_ms / 1000.0
        self._cost = cost
        self._bucket_key = bucket_key
        self._on_start = on_start
        self._name = name
        self._join_timeout_s = join_timeout_s

        self._queue: queue.Queue = queue.Queue()
        self._ready = threading.Event()
        self._start_error: Optional[BaseException] = None
        self._lock = threading.Lock()
        self._closed = False
        self._thread: Optional[threading.Thread] = None

    # ---- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Start the worker thread, run ``on_start`` on it, and re-raise its error."""
        self._thread = threading.Thread(target=self._run, name=self._name, daemon=True)
        self._thread.start()
        self._ready.wait()
        if self._start_error is not None:
            # on_start ran on the thread, which then exited — mark closed so a
            # later submit() raises instead of queueing to a dead consumer.
            self.shutdown()
            raise self._start_error

    async def submit(self, items: List[T]) -> List[R]:
        """Submit a group of items; await one result per item, in order."""
        if self._thread is None:
            raise RuntimeError("ThreadedMicroBatcher.submit() called before start()")
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        request = _Request(loop, future, [None] * len(items), len(items))
        works = [
            _Work(item, self._bucket_key(item), self._cost(item), request, i)
            for i, item in enumerate(items)
        ]
        # Hold the lock across the closed-check and the puts so a concurrent
        # shutdown() cannot slip the sentinel between them and strand the request.
        with self._lock:
            if self._closed:
                raise RuntimeError(
                    "ThreadedMicroBatcher.submit() called after shutdown()"
                )
            if not works:
                future.set_result([])
                return await future
            for work in works:
                self._queue.put(work)
        return await future

    def shutdown(self) -> None:
        """Stop the worker, failing not-yet-started items. Idempotent; retries the
        join if a slow ``fn`` is still in flight."""
        if self._thread is None:
            return
        with self._lock:
            if not self._closed:
                self._closed = True
                self._drain_pending()  # fail queued items, then signal stop
                self._queue.put(_SHUTDOWN)
        self._thread.join(timeout=self._join_timeout_s)
        if self._thread.is_alive():
            logger.warning(
                "ThreadedMicroBatcher(%s): worker still running after %gs; will "
                "reap on a later call",
                self._name,
                self._join_timeout_s,
            )
            return
        self._drain_pending()  # belt-and-suspenders

    # ---- worker thread -----------------------------------------------------

    def _run(self) -> None:
        try:
            if self._on_start is not None:
                self._on_start()  # build / warmup / CUDA-graph capture HERE
        except BaseException as exc:  # noqa: BLE001 — surface to start()
            self._start_error = exc
            self._ready.set()
            return
        self._ready.set()
        while True:
            works = self._collect()
            if works is None:
                return
            self._dispatch(works)

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
        """Group the drained items by bucket, split by cost budget, run ``fn``."""
        by_bucket: dict = {}
        for work in works:
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
                self._fail(work.request, exc)
            return
        if len(results) != len(items):
            err = RuntimeError(
                f"batch fn returned {len(results)} results for {len(items)} items; "
                "it must return one result per item"
            )
            for work in batch:
                self._fail(work.request, err)
            return
        for work, result in zip(batch, results):
            self._deliver(work, result)

    def _deliver(self, work: _Work, result: R) -> None:
        req = work.request
        req.results[work.index] = result
        req.remaining -= 1
        if req.remaining == 0:
            self._resolve(req, result=list(req.results))

    def _drain_pending(self) -> None:
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return
            if item is _SHUTDOWN:
                continue
            self._fail(item.request, RuntimeError("ThreadedMicroBatcher shut down"))

    def _fail(self, req: _Request, exc: BaseException) -> None:
        self._resolve(req, exc=exc)

    def _resolve(
        self,
        req: _Request,
        result: Optional[List[R]] = None,
        exc: Optional[BaseException] = None,
    ) -> None:
        """Resolve a request's future on its own event loop (cross-thread safe)."""

        def _set() -> None:
            if req.future.done():
                return
            if exc is not None:
                req.future.set_exception(exc)
            else:
                req.future.set_result(result)

        req.loop.call_soon_threadsafe(_set)
