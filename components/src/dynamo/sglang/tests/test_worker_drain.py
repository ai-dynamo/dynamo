# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from dynamo._core import WorkerDraining
from dynamo.health_check import HEALTH_CHECK_KEY
from dynamo.sglang.drain import DrainConflict, WorkerDrain

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


@pytest.fixture
def ready():
    event = asyncio.Event()
    event.set()
    return event


@pytest.fixture
def endpoint():
    return SimpleNamespace(
        unregister_endpoint_instance=AsyncMock(),
        register_endpoint_instance=AsyncMock(),
    )


@pytest.mark.asyncio
async def test_drain_preserves_admitted_stream_and_rejects_late_work(endpoint, ready):
    release = asyncio.Event()

    async def generate(request, context):
        yield "first"
        await release.wait()
        yield "last"

    drain = WorkerDrain(
        endpoint,
        lock=asyncio.Lock(),
        shutdown_event=asyncio.Event(),
        terminal_event=asyncio.Event(),
        ready_event=ready,
        is_paused=lambda: False,
        grace_period=0,
    )
    tracked = drain.wrap(generate)
    assert tracked.__wrapped__ is generate
    stream = tracked({}, None)
    assert await anext(stream) == "first"
    status = await drain.drain({})
    assert status["inflight_requests"] == 1
    assert not status["accepting_requests"]
    assert status["admin_withdrawn"]
    with pytest.raises(WorkerDraining):
        await anext(tracked({}, None))
    release.set()
    assert await asyncio.wait_for(anext(stream), 1) == "last"
    with pytest.raises(StopAsyncIteration):
        await anext(stream)
    status = await drain.status({})
    assert status["state"] == "draining"
    assert status["inflight_requests"] == 0
    assert status["engine_quiescent"] is None
    assert not status["completion_supported"]


@pytest.mark.asyncio
async def test_cancelled_drain_request_does_not_cancel_withdrawal(endpoint, ready):
    entered = asyncio.Event()
    release = asyncio.Event()

    async def unregister():
        entered.set()
        await release.wait()

    endpoint.unregister_endpoint_instance.side_effect = unregister
    drain = WorkerDrain(
        endpoint,
        lock=asyncio.Lock(),
        shutdown_event=asyncio.Event(),
        terminal_event=asyncio.Event(),
        ready_event=ready,
        is_paused=lambda: False,
        grace_period=0,
    )
    request = asyncio.create_task(drain.drain({}))
    await asyncio.wait_for(entered.wait(), 1)
    request.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request
    release.set()
    status = await asyncio.wait_for(drain.drain({}), 1)
    assert status["state"] == "draining"
    assert not status["accepting_requests"]
    endpoint.unregister_endpoint_instance.assert_awaited_once()


@pytest.mark.asyncio
async def test_discovery_failures_are_observable_and_resume_can_retry(endpoint, ready):
    drain = WorkerDrain(
        endpoint,
        lock=asyncio.Lock(),
        shutdown_event=asyncio.Event(),
        terminal_event=asyncio.Event(),
        ready_event=ready,
        is_paused=lambda: False,
        grace_period=0,
    )
    endpoint.unregister_endpoint_instance.side_effect = RuntimeError("discovery down")
    with pytest.raises(RuntimeError, match="discovery down"):
        await drain.drain({})
    assert (await drain.status({}))["state"] == "serving"
    assert not drain.withdrawn
    endpoint.unregister_endpoint_instance.side_effect = None
    await drain.drain({})
    endpoint.register_endpoint_instance.side_effect = RuntimeError("register failed")
    with pytest.raises(RuntimeError, match="register failed"):
        await drain.resume({})
    status = await drain.status({})
    assert status["state"] == "draining"
    assert not status["accepting_requests"]
    assert status["last_error"] == "register failed"
    endpoint.register_endpoint_instance.side_effect = None
    assert (await drain.resume({}))["state"] == "serving"
    assert not drain.withdrawn


@pytest.mark.asyncio
async def test_shutdown_during_resume_undoes_registration(endpoint, ready):
    entered = asyncio.Event()
    release = asyncio.Event()
    shutdown = asyncio.Event()

    async def register():
        entered.set()
        await release.wait()

    drain = WorkerDrain(
        endpoint,
        lock=asyncio.Lock(),
        shutdown_event=shutdown,
        terminal_event=asyncio.Event(),
        ready_event=ready,
        is_paused=lambda: False,
        grace_period=0,
    )
    await drain.drain({})
    endpoint.register_endpoint_instance.side_effect = register
    request = asyncio.create_task(drain.resume({}))
    await asyncio.wait_for(entered.wait(), 1)
    shutdown.set()
    release.set()
    with pytest.raises(DrainConflict, match="superseded"):
        await asyncio.wait_for(request, 1)
    status = await drain.status({})
    assert status["state"] == "stopping"
    assert status["admin_withdrawn"]
    assert not status["accepting_requests"]
    assert endpoint.unregister_endpoint_instance.await_count == 2
    with pytest.raises(DrainConflict, match="shutdown"):
        await drain.resume({})


@pytest.mark.asyncio
async def test_resume_cancels_pending_router_grace(endpoint, ready):
    drain = WorkerDrain(
        endpoint,
        lock=asyncio.Lock(),
        shutdown_event=asyncio.Event(),
        terminal_event=asyncio.Event(),
        ready_event=ready,
        is_paused=lambda: False,
        grace_period=60,
    )
    status = await drain.drain({})
    assert status["accepting_requests"]
    grace = drain._grace_task
    assert (await drain.resume({}))["accepting_requests"]
    with pytest.raises(asyncio.CancelledError):
        await grace
    await drain.resume({})
    endpoint.register_endpoint_instance.assert_awaited_once()


@pytest.mark.asyncio
async def test_stream_close_releases_admission_count(endpoint, ready):
    closed = asyncio.Event()

    async def generate():
        try:
            yield "first"
            await asyncio.Event().wait()
        finally:
            closed.set()

    drain = WorkerDrain(
        endpoint,
        lock=asyncio.Lock(),
        shutdown_event=asyncio.Event(),
        terminal_event=asyncio.Event(),
        ready_event=ready,
        is_paused=lambda: False,
        grace_period=0,
    )
    stream = drain.wrap(generate)()
    await anext(stream)
    assert (await drain.status({}))["inflight_requests"] == 1
    await stream.aclose()
    assert closed.is_set()
    assert (await drain.status({}))["inflight_requests"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload", [{HEALTH_CHECK_KEY: True}, {"request": {HEALTH_CHECK_KEY: True}}]
)
async def test_startup_controls_are_fenced_but_health_remains_callable(
    endpoint, payload
):
    ready = asyncio.Event()
    drain = WorkerDrain(
        endpoint,
        lock=asyncio.Lock(),
        shutdown_event=asyncio.Event(),
        terminal_event=asyncio.Event(),
        ready_event=ready,
        is_paused=lambda: False,
        grace_period=0,
    )

    async def generate(request):
        yield "healthy"

    tracked = drain.wrap(generate)
    assert (await drain.status({}))["state"] == "starting"
    with pytest.raises(DrainConflict, match="not ready"):
        await drain.drain({})
    with pytest.raises(WorkerDraining):
        await anext(tracked({}))
    assert [item async for item in tracked(payload)] == ["healthy"]
    ready.set()
    await drain.drain({})
    assert [item async for item in tracked(payload)] == ["healthy"]
    endpoint.register_endpoint_instance.assert_not_awaited()


@pytest.mark.asyncio
async def test_resume_does_not_override_engine_pause(endpoint, ready):
    paused = True
    drain = WorkerDrain(
        endpoint,
        lock=asyncio.Lock(),
        shutdown_event=asyncio.Event(),
        terminal_event=asyncio.Event(),
        ready_event=ready,
        is_paused=lambda: paused,
        grace_period=0,
    )
    await drain.drain({})
    with pytest.raises(DrainConflict, match="continue engine"):
        await drain.resume({})
    endpoint.register_endpoint_instance.assert_not_awaited()
    assert drain.withdrawn
    paused = False
    assert (await drain.resume({}))["state"] == "serving"


@pytest.mark.asyncio
async def test_terminal_fence_preserves_existing_shutdown_router_grace(endpoint, ready):
    terminal = asyncio.Event()
    shutdown = asyncio.Event()
    drain = WorkerDrain(
        endpoint,
        lock=asyncio.Lock(),
        shutdown_event=shutdown,
        terminal_event=terminal,
        ready_event=ready,
        is_paused=lambda: False,
        grace_period=0,
    )

    async def generate(request):
        yield "already routed"

    terminal.set()
    status = await drain.status({})
    assert status["state"] == "stopping"
    assert status["accepting_requests"]
    with pytest.raises(DrainConflict):
        await drain.resume({})
    assert [item async for item in drain.wrap(generate)({})] == ["already routed"]
    shutdown.set()
    with pytest.raises(WorkerDraining):
        await anext(drain.wrap(generate)({}))


@pytest.mark.asyncio
async def test_close_joins_cancelled_http_resume_before_final_unregister(
    endpoint, ready
):
    entered = asyncio.Event()
    cancelled = asyncio.Event()
    order = []

    async def unregister():
        order.append("unregister")

    async def register():
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            order.append("registration cancelled")
            cancelled.set()
            raise

    endpoint.unregister_endpoint_instance.side_effect = unregister
    endpoint.register_endpoint_instance.side_effect = register
    drain = WorkerDrain(
        endpoint,
        lock=asyncio.Lock(),
        shutdown_event=asyncio.Event(),
        terminal_event=asyncio.Event(),
        ready_event=ready,
        is_paused=lambda: False,
        grace_period=0,
    )
    await drain.drain({})
    request = asyncio.create_task(drain.resume({}))
    await asyncio.wait_for(entered.wait(), 1)
    request.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request
    operations = list(drain._operations)
    await asyncio.wait_for(drain.close(), 1)
    assert cancelled.is_set()
    assert all(task.done() for task in operations)
    assert order == ["unregister", "registration cancelled", "unregister"]
    assert (await drain.status({}))["admin_withdrawn"]
    await drain.close()
    assert len(order) == 3

    async def health(request):
        yield "must not run after cleanup"

    with pytest.raises(WorkerDraining):
        await anext(drain.wrap(health)({HEALTH_CHECK_KEY: True}))
