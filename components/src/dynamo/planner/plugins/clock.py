# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic clock abstraction.

All time access in the orchestrator and PluginRegistry MUST go through
``Clock`` — direct ``time.time()`` / ``time.monotonic()`` /
``asyncio.sleep`` is forbidden.

**Two time sources**:
- ``now()``: epoch float (wall-clock); use for audit log timestamps,
  ``decision_id`` generation
- ``monotonic()``: monotonic float; use for duration / scheduling
  (immune to NTP / clock skew)

**Two implementations**:
- ``WallClock``: production
- ``VirtualClock``: replay / test; ``advance(N)`` warps time forward and
  wakes pending sleepers

**Production safety**: ``VirtualClock`` MUST NOT be used in production
(NativePlannerBase startup checks ``clock.type=virtual`` and refuses to
start unless ``DYNAMO_PLANNER_TEST=1``).
"""

from __future__ import annotations

import abc
import asyncio
import heapq
import itertools
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class Clock(abc.ABC):
    """Abstract clock interface."""

    @abc.abstractmethod
    def now(self) -> float:
        """Wall-clock seconds since epoch (UTC).

        Use for audit log timestamps, ``decision_id`` generation, anything
        that must be human-readable / NTP-aligned.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def monotonic(self) -> float:
        """Monotonic seconds (independent of wall-clock adjustments).

        Use for duration measurement, scheduling intervals, circuit breaker
        cooldown — anything where clock skew / NTP jumps would cause bugs.
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def sleep(self, seconds: float) -> None:
        """Asynchronously sleep for the given number of seconds.

        ``WallClock`` delegates to ``asyncio.sleep``.
        ``VirtualClock`` parks on a future awaiting ``advance()`` to elapse.

        Cancellation: standard asyncio cancellation semantics — if the
        awaiting task is cancelled, ``CancelledError`` propagates.
        """
        raise NotImplementedError


class WallClock(Clock):
    """Production clock — real wall-clock and event-loop sleep."""

    def now(self) -> float:
        return time.time()

    def monotonic(self) -> float:
        return time.monotonic()

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


class VirtualClock(Clock):
    """Test / replay clock — time advances only via explicit ``advance()`` call.

    ``sleep()`` parks the caller on a future; ``advance(N)`` adds N to virtual
    time and resolves all futures whose deadlines have passed.

    **Cancellation cleanup** (P1-4 review v11):
    - ``advance()`` does a cleanup pass to discard already-cancelled
      futures from the heap, preventing memory leak in long-running tests.
    """

    def __init__(self, start_now: float = 0.0, start_mono: float = 0.0) -> None:
        self._now = start_now
        self._mono = start_mono
        # heap of (wake_at_monotonic, sequence_id, future);
        # sequence_id ensures FIFO when wake_at ties (heap requires comparable tuples)
        self._sleepers: list[tuple[float, int, asyncio.Future[None]]] = []
        self._counter = itertools.count()

    def now(self) -> float:
        return self._now

    def monotonic(self) -> float:
        return self._mono

    async def sleep(self, seconds: float) -> None:
        if seconds <= 0:
            # Immediate yield — let other coroutines run, but no virtual time passes
            await asyncio.sleep(0)
            return
        loop = asyncio.get_event_loop()
        fut: asyncio.Future[None] = loop.create_future()
        wake_at = self._mono + seconds
        heapq.heappush(self._sleepers, (wake_at, next(self._counter), fut))
        try:
            await fut
        except asyncio.CancelledError:
            # Don't try to remove from heap here (heap removal is O(n)); just
            # let the cancelled future stay in heap and skip on advance().
            raise

    def advance(self, seconds: float) -> None:
        """Warp virtual time forward by ``seconds`` and wake any due sleepers.

        Sleepers whose ``wake_at <= mono + seconds`` resolve immediately
        (without calling their awaiting coroutine — the awaiter resumes on
        the next event loop iteration).

        **Cleanup pass** (v11 P1-4): cancelled / done futures are silently
        discarded from the heap to bound memory in long-running tests.
        """
        if seconds < 0:
            raise ValueError(f"VirtualClock.advance: seconds must be >= 0, got {seconds}")
        self._now += seconds
        self._mono += seconds
        while self._sleepers and self._sleepers[0][0] <= self._mono:
            _wake_at, _seq, fut = heapq.heappop(self._sleepers)
            if not fut.done():
                fut.set_result(None)
            # else: cancelled or already-resolved; silently drop


__all__ = ["Clock", "WallClock", "VirtualClock"]
