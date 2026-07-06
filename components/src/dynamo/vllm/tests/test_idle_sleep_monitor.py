# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the idle sleep TTL ladder monitor.

`dynamo.vllm.idle_sleep` is stdlib-only, so these tests run without
vLLM/torch installed (hence the file is not named ``test_vllm_*``).
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from dynamo.vllm.idle_sleep import IdleSleepMonitor, IdleSleepState

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
]

TIMEOUT = 10.0
ESCALATE_TIMEOUT = 30.0


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def make_monitor(clock: FakeClock, **overrides) -> IdleSleepMonitor:
    kwargs = dict(
        sleep_fn=AsyncMock(return_value={"status": "ok"}),
        timeout=TIMEOUT,
        level=1,
        escalate_fn=AsyncMock(return_value={"status": "ok"}),
        escalate_timeout=ESCALATE_TIMEOUT,
        escalate_level=2,
        time_source=clock,
    )
    kwargs.update(overrides)
    return IdleSleepMonitor(**kwargs)


def test_constructor_rejects_bad_config():
    with pytest.raises(ValueError):
        IdleSleepMonitor(sleep_fn=AsyncMock(), timeout=0)
    with pytest.raises(ValueError):
        IdleSleepMonitor(sleep_fn=AsyncMock(), timeout=10, escalate_timeout=0)
    with pytest.raises(ValueError):
        IdleSleepMonitor(sleep_fn=AsyncMock(), timeout=10, escalate_timeout=30)


@pytest.mark.asyncio
async def test_sleeps_after_idle_ttl():
    clock = FakeClock()
    monitor = make_monitor(clock)

    clock.advance(TIMEOUT - 0.1)
    await monitor._tick()
    monitor._sleep_fn.assert_not_awaited()
    assert monitor.state is IdleSleepState.WARM

    clock.advance(0.1)
    await monitor._tick()
    monitor._sleep_fn.assert_awaited_once_with(1)
    assert monitor.state is IdleSleepState.ASLEEP
    assert monitor.current_level == 1


@pytest.mark.asyncio
async def test_in_flight_request_blocks_sleep():
    clock = FakeClock()
    monitor = make_monitor(clock)

    monitor.record_request_started()
    clock.advance(TIMEOUT * 5)
    await monitor._tick()
    monitor._sleep_fn.assert_not_awaited()

    # The idle window restarts when the request finishes.
    monitor.record_request_finished()
    await monitor._tick()
    monitor._sleep_fn.assert_not_awaited()

    clock.advance(TIMEOUT)
    await monitor._tick()
    monitor._sleep_fn.assert_awaited_once_with(1)


@pytest.mark.asyncio
async def test_request_activity_resets_idle_timer():
    clock = FakeClock()
    monitor = make_monitor(clock)

    clock.advance(TIMEOUT * 0.6)
    with monitor.track_request():
        pass

    # Total elapsed exceeds the TTL, but only 0.6 * TIMEOUT has passed
    # since the last request finished.
    clock.advance(TIMEOUT * 0.6)
    await monitor._tick()
    monitor._sleep_fn.assert_not_awaited()

    clock.advance(TIMEOUT * 0.4)
    await monitor._tick()
    monitor._sleep_fn.assert_awaited_once_with(1)


@pytest.mark.asyncio
async def test_escalates_to_deeper_level():
    clock = FakeClock()
    monitor = make_monitor(clock)

    clock.advance(TIMEOUT)
    await monitor._tick()
    assert monitor.state is IdleSleepState.ASLEEP

    clock.advance(ESCALATE_TIMEOUT - 0.1)
    await monitor._tick()
    monitor._escalate_fn.assert_not_awaited()

    clock.advance(0.1)
    await monitor._tick()
    monitor._escalate_fn.assert_awaited_once_with(1, 2)
    assert monitor.current_level == 2

    # Already at the deepest configured level: no further escalation.
    clock.advance(ESCALATE_TIMEOUT * 5)
    await monitor._tick()
    monitor._escalate_fn.assert_awaited_once()


@pytest.mark.asyncio
async def test_no_escalation_when_not_configured():
    clock = FakeClock()
    monitor = make_monitor(clock, escalate_fn=None, escalate_timeout=None)

    clock.advance(TIMEOUT)
    await monitor._tick()
    assert monitor.state is IdleSleepState.ASLEEP

    clock.advance(ESCALATE_TIMEOUT * 10)
    await monitor._tick()
    assert monitor.current_level == 1


@pytest.mark.asyncio
async def test_wake_resets_state_and_idle_timer():
    clock = FakeClock()
    monitor = make_monitor(clock)

    clock.advance(TIMEOUT)
    await monitor._tick()
    assert monitor.state is IdleSleepState.ASLEEP

    monitor.notify_woken()
    assert monitor.state is IdleSleepState.WARM
    assert monitor.current_level is None

    # A fresh full TTL must elapse before the next auto-sleep.
    clock.advance(TIMEOUT - 0.1)
    await monitor._tick()
    monitor._sleep_fn.assert_awaited_once()

    clock.advance(0.1)
    await monitor._tick()
    assert monitor._sleep_fn.await_count == 2


@pytest.mark.asyncio
async def test_external_sleep_is_tracked_and_escalated():
    clock = FakeClock()
    monitor = make_monitor(clock)

    # control/sleep route slept the engine directly.
    monitor.notify_slept(1)
    assert monitor.state is IdleSleepState.ASLEEP

    clock.advance(ESCALATE_TIMEOUT)
    await monitor._tick()
    monitor._sleep_fn.assert_not_awaited()
    monitor._escalate_fn.assert_awaited_once_with(1, 2)


@pytest.mark.asyncio
async def test_sleep_failure_backs_off_full_ttl():
    clock = FakeClock()
    monitor = make_monitor(
        clock, sleep_fn=AsyncMock(return_value={"status": "error", "message": "boom"})
    )

    clock.advance(TIMEOUT)
    await monitor._tick()
    assert monitor._sleep_fn.await_count == 1
    assert monitor.state is IdleSleepState.WARM

    # No hot-loop: the retry waits out another full TTL.
    await monitor._tick()
    assert monitor._sleep_fn.await_count == 1

    clock.advance(TIMEOUT)
    await monitor._tick()
    assert monitor._sleep_fn.await_count == 2


@pytest.mark.asyncio
async def test_escalation_failure_backs_off():
    clock = FakeClock()
    monitor = make_monitor(
        clock,
        escalate_fn=AsyncMock(return_value={"status": "error", "message": "boom"}),
    )

    clock.advance(TIMEOUT)
    await monitor._tick()
    clock.advance(ESCALATE_TIMEOUT)
    await monitor._tick()
    assert monitor._escalate_fn.await_count == 1
    assert monitor.current_level == 1

    await monitor._tick()
    assert monitor._escalate_fn.await_count == 1

    clock.advance(ESCALATE_TIMEOUT)
    await monitor._tick()
    assert monitor._escalate_fn.await_count == 2


def test_track_request_decrements_on_error():
    clock = FakeClock()
    monitor = make_monitor(clock)

    with pytest.raises(RuntimeError):
        with monitor.track_request():
            assert monitor.in_flight == 1
            raise RuntimeError("request failed")
    assert monitor.in_flight == 0


@pytest.mark.asyncio
async def test_background_loop_sleeps_and_stops():
    sleep_fn = AsyncMock(return_value={"status": "ok"})
    slept = asyncio.Event()

    async def sleep_and_signal(level: int) -> dict:
        result = await sleep_fn(level)
        slept.set()
        return result

    monitor = IdleSleepMonitor(
        sleep_fn=sleep_and_signal,
        timeout=0.05,
        level=2,
        poll_interval=0.01,
    )
    monitor.start()
    try:
        await asyncio.wait_for(slept.wait(), timeout=2.0)
    finally:
        monitor.stop()

    sleep_fn.assert_awaited_once_with(2)
    assert monitor.state is IdleSleepState.ASLEEP


@pytest.mark.asyncio
async def test_background_loop_exits_on_shutdown_event():
    shutdown_event = asyncio.Event()
    monitor = IdleSleepMonitor(
        sleep_fn=AsyncMock(return_value={"status": "ok"}),
        timeout=60.0,
        shutdown_event=shutdown_event,
        poll_interval=0.01,
    )
    monitor.start()
    task = monitor._task
    shutdown_event.set()
    await asyncio.wait_for(task, timeout=2.0)
