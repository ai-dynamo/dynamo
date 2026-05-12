# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Clock implementations."""

from __future__ import annotations

import asyncio
import time

import pytest

from dynamo.planner.plugins.clock import VirtualClock, WallClock

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


# ----- WallClock -----


def test_wall_clock_now_close_to_time_time():
    c = WallClock()
    delta = abs(c.now() - time.time())
    assert delta < 0.5  # generous; just sanity


def test_wall_clock_monotonic_strictly_increasing():
    c = WallClock()
    a = c.monotonic()
    time.sleep(0.001)
    b = c.monotonic()
    assert b > a


@pytest.mark.asyncio
async def test_wall_clock_sleep_actually_sleeps():
    c = WallClock()
    t_start = c.monotonic()
    await c.sleep(0.05)
    elapsed = c.monotonic() - t_start
    assert 0.04 < elapsed < 0.5  # generous upper for slow CI


# ----- VirtualClock -----


def test_virtual_clock_initial_state():
    c = VirtualClock(start_now=1000.0, start_mono=0.0)
    assert c.now() == 1000.0
    assert c.monotonic() == 0.0


def test_virtual_clock_advance_updates_both():
    c = VirtualClock(start_now=1000.0, start_mono=5.0)
    c.advance(7.5)
    assert c.now() == 1007.5
    assert c.monotonic() == 12.5


def test_virtual_clock_advance_negative_rejected():
    c = VirtualClock()
    with pytest.raises(ValueError, match="seconds must be >= 0"):
        c.advance(-1.0)


@pytest.mark.asyncio
async def test_virtual_clock_sleeper_resumes_on_advance():
    c = VirtualClock()
    woke = []

    async def sleeper(name: str, secs: float):
        await c.sleep(secs)
        woke.append((name, c.monotonic()))

    task1 = asyncio.create_task(sleeper("a", 5.0))
    task2 = asyncio.create_task(sleeper("b", 10.0))
    task3 = asyncio.create_task(sleeper("c", 3.0))

    # Let coroutines schedule and queue their futures
    await asyncio.sleep(0)

    c.advance(7.0)
    # Let resumed coroutines run
    for _ in range(3):
        await asyncio.sleep(0)

    # a (5s) and c (3s) should have woken at mono=7.0
    assert ("c", 7.0) in woke
    assert ("a", 7.0) in woke
    # b (10s) still pending
    assert not any(name == "b" for name, _ in woke)
    assert not task2.done()

    c.advance(5.0)
    for _ in range(3):
        await asyncio.sleep(0)
    assert ("b", 12.0) in woke
    await task1
    await task2
    await task3


@pytest.mark.asyncio
async def test_virtual_clock_immediate_yield_for_zero_sleep():
    c = VirtualClock()
    initial = c.monotonic()
    await c.sleep(0)
    assert c.monotonic() == initial  # zero sleep does not advance virtual time


@pytest.mark.asyncio
async def test_virtual_clock_advance_skips_past_deadline():
    """Sleeper with deadline T resolves even when advance(N) where N > T."""
    c = VirtualClock()

    async def sleeper():
        await c.sleep(2.0)
        return c.monotonic()

    task = asyncio.create_task(sleeper())
    await asyncio.sleep(0)
    c.advance(100.0)  # way past 2s
    for _ in range(3):
        await asyncio.sleep(0)
    result = await task
    assert result == 100.0


@pytest.mark.asyncio
async def test_virtual_clock_cancellation_does_not_leak_heap():
    """v11 P1-4: cancelled sleeper futures must not keep heap growing."""
    c = VirtualClock()

    async def cancelled_sleeper():
        await c.sleep(1000)

    task = asyncio.create_task(cancelled_sleeper())
    await asyncio.sleep(0)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # After advance past deadline, the cancelled future is dropped from heap
    assert len(c._sleepers) == 1  # before advance, still in heap
    c.advance(2000)
    # Heap should now be empty (cancelled future popped + silently discarded)
    assert len(c._sleepers) == 0
