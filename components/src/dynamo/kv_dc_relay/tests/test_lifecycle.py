# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import AsyncIterator, Awaitable

import pytest
import pytest_asyncio

from dynamo.kv_dc_relay.cli import monitor_relay

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.asyncio,
    pytest.mark.timeout(5),
]


class FakeRelay:
    def __init__(self, *, is_future: bool) -> None:
        self.is_future = is_future
        self.shutdown: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self.waiter_started = asyncio.Event()

    def wait_for_shutdown(self) -> Awaitable[None]:
        if self.is_future:
            self.waiter_started.set()
            return self.shutdown
        return self.wait_for_shutdown_coroutine()

    async def wait_for_shutdown_coroutine(self) -> None:
        self.waiter_started.set()
        await self.shutdown


Monitor = tuple[FakeRelay, asyncio.Future[object], asyncio.Task[None]]


@pytest_asyncio.fixture(params=[False, True], ids=["coroutine", "future"])
async def running_monitor(request: pytest.FixtureRequest) -> AsyncIterator[Monitor]:
    relay = FakeRelay(is_future=request.param)
    endpoint: asyncio.Future[object] = asyncio.get_running_loop().create_future()
    monitor = asyncio.create_task(monitor_relay(relay, [endpoint]))
    try:
        await asyncio.wait_for(relay.waiter_started.wait(), timeout=1)
        yield relay, endpoint, monitor
    finally:
        monitor.cancel()
        endpoint.cancel()
        relay.shutdown.cancel()
        await asyncio.gather(monitor, endpoint, relay.shutdown, return_exceptions=True)


@pytest.mark.parametrize("fails", [False, True], ids=["completed", "failed"])
async def test_relay_shutdown_ends_monitor(
    running_monitor: Monitor, fails: bool
) -> None:
    relay, endpoint, monitor = running_monitor
    if fails:
        failure = RuntimeError("relay shutdown failure")
        relay.shutdown.set_exception(failure)
        with pytest.raises(RuntimeError) as error:
            await asyncio.wait_for(monitor, timeout=1)
        assert error.value is failure
    else:
        relay.shutdown.set_result(None)
        await asyncio.wait_for(monitor, timeout=1)
    assert not endpoint.done()


@pytest.mark.parametrize("fails", [False, True], ids=["completed", "failed"])
async def test_endpoint_completion_releases_shutdown_waiter(
    running_monitor: Monitor,
    fails: bool,
) -> None:
    relay, endpoint, monitor = running_monitor
    if fails:
        failure = RuntimeError("endpoint failure")
        endpoint.set_exception(failure)
        with pytest.raises(RuntimeError) as error:
            await asyncio.wait_for(monitor, timeout=1)
        assert error.value is failure
    else:
        endpoint.set_result(None)
        await asyncio.wait_for(monitor, timeout=1)
    assert relay.shutdown.cancelled()


async def test_monitor_cancellation_releases_shutdown_waiter(
    running_monitor: Monitor,
) -> None:
    relay, endpoint, monitor = running_monitor
    monitor.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(monitor, timeout=1)
    assert relay.shutdown.cancelled()
    assert not endpoint.done()
