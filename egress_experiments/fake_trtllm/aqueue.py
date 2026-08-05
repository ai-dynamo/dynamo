# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``AsyncQueue`` / ``_SyncQueue``, ported from ``tensorrt_llm/llmapi/utils.py``.

This is the single most load-bearing piece of the whole simulation, because it
is what makes the response path cost ONE event-loop ready-deque entry per IPC
batch instead of one per response.

The split matters:

* :meth:`AsyncQueue.put_nowait` appends to a ``collections.deque`` and touches
  the loop not at all -- ``deque.append`` is thread safe, so the proxy dispatch
  thread can do this for every response in the batch for free.
* :meth:`SyncQueue.notify_many` is issued ONCE per batch and is the only thing
  that reaches the loop: one ``run_coroutine_threadsafe`` -> one
  ``call_soon_threadsafe`` -> ONE ready-deque entry, however many responses the
  batch carried.

``ASYNCIO_GIL_PATH.md`` measured that ratio on the real worker at ~132
responses per deque entry. Here it is exactly ``len(batch)``, which is what the
engine's in-flight count makes it.
"""

from __future__ import annotations

import asyncio
import collections
from typing import Iterable, List, Optional


class AsyncQueue:
    """Sync-producer / async-consumer queue with decoupled put and notify."""

    class EventLoopShutdownError(Exception):
        pass

    def __init__(self) -> None:
        self._q: collections.deque = collections.deque()
        self._event = asyncio.Event()
        self._sync_q = SyncQueue(self)

    @property
    def sync_q(self) -> "SyncQueue":
        return self._sync_q

    def empty(self) -> bool:
        return not self._q

    def put(self, item) -> None:
        self._q.append(item)
        self._event.set()

    def put_nowait(self, item) -> None:
        """Append without waking the consumer. Safe from any thread."""
        self._q.append(item)

    def notify(self) -> None:
        """Wake the consumer. MUST run on the event loop."""
        if self._q:
            self._event.set()

    async def get(self):
        # Several coroutines can be woken by one `set()`, so re-check rather
        # than trusting the event -- same as the real implementation.
        while not self._q:
            await self._event.wait()
        res = self._q.popleft()
        if not self._q:
            self._event.clear()
        return res


class SyncQueue:
    """The producer-side handle the proxy dispatch thread holds.

    Named ``_SyncQueue`` in TRT-LLM; the underscore is dropped here only so it
    can be imported by name from tests.
    """

    def __init__(
        self, queue: AsyncQueue, loop: Optional[asyncio.AbstractEventLoop] = None
    ) -> None:
        self._aq = queue
        # TRT-LLM uses get_event_loop() here, which only works because
        # generate_async is called from the loop. get_running_loop() is the
        # same thing in that situation and does not warn.
        if loop is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.get_event_loop()
        self._loop = loop

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        return self._loop

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Attach the loop the consuming coroutine will run on.

        TRT-LLM captures this at ``GenerationResult`` construction via
        ``asyncio.get_event_loop()``, which only works because
        ``generate_async`` is called from the loop. The simulation constructs
        results from the loop too, but keeps this hook so a test can build a
        result outside the loop and bind afterwards.
        """
        self._loop = loop

    def put_nowait(self, item) -> None:
        """Enqueue without notifying. This is the per-response cost."""
        self._aq.put_nowait(item)

    async def _notify(self) -> None:
        self._aq.notify()

    @staticmethod
    async def _notify_many(queues: Iterable["SyncQueue"]) -> None:
        for queue in queues:
            queue._aq.notify()

    @staticmethod
    def notify_many(loop: asyncio.AbstractEventLoop, queues: List["SyncQueue"]) -> None:
        """One coroutine hop for the whole batch. This is the per-BATCH cost."""
        if loop.is_running():
            asyncio.run_coroutine_threadsafe(
                SyncQueue._notify_many(frozenset(queues)), loop
            )
        else:
            raise AsyncQueue.EventLoopShutdownError()

    def __hash__(self) -> int:
        return id(self)
