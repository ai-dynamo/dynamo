# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from dynamo._core import WorkerDraining
from dynamo.health_check import HEALTH_CHECK_KEY
from dynamo.sglang import drain as drain_module
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


@pytest.fixture
def clock(monkeypatch):
    now = Mock(return_value=100.0)
    monkeypatch.setattr(drain_module, "monotonic", now, raising=False)
    return now


@pytest.mark.asyncio
@pytest.mark.parametrize("grace_period", [0, 0.001])
async def test_drain_preserves_admitted_stream_and_rejects_late_work(
    endpoint, ready, grace_period, clock
):
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
        grace_period=grace_period,
    )
    tracked = drain.wrap(generate)
    assert tracked.__wrapped__ is generate
    stream = tracked({}, None)
    assert await anext(stream) == "first"
    status = await drain.drain({})
    if grace_period:
        assert status["state"] == "withdrawing"
        assert status["accepting_requests"]
        clock.return_value += grace_period
        status = await drain.status({})
    assert status["state"] == "withdrawn"
    assert not status["accepting_requests"]
    assert status["admin_withdrawn"]
    with pytest.raises(WorkerDraining):
        await anext(tracked({}, None))
    release.set()
    assert await asyncio.wait_for(anext(stream), 1) == "last"
    with pytest.raises(StopAsyncIteration):
        await anext(stream)
    status = await drain.status({})
    assert status["state"] == "withdrawn"


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
    assert status["state"] == "withdrawn"
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
    endpoint.published = True

    async def committed_unregister():
        endpoint.published = False
        raise RuntimeError("discovery down")

    async def register():
        endpoint.published = True

    endpoint.unregister_endpoint_instance.side_effect = committed_unregister
    endpoint.register_endpoint_instance.side_effect = register
    with pytest.raises(RuntimeError, match="discovery down"):
        await drain.drain({})
    assert (await drain.status({}))["state"] == "withdrawing"
    assert drain.withdrawn
    assert not endpoint.published
    await drain.resume({})
    endpoint.register_endpoint_instance.assert_awaited_once()
    assert endpoint.published
    with pytest.raises(RuntimeError, match="discovery down"):
        await drain.drain({})
    endpoint.unregister_endpoint_instance.side_effect = None
    status = await drain.drain({})
    assert endpoint.unregister_endpoint_instance.await_count == 3
    assert status["state"] == "withdrawn"
    assert not status["accepting_requests"]
    endpoint.register_endpoint_instance.side_effect = RuntimeError("register failed")
    with pytest.raises(RuntimeError, match="register failed"):
        await drain.resume({})
    status = await drain.status({})
    assert status["state"] == "serving"
    assert status["accepting_requests"]
    assert status["last_error"] == "register failed"
    endpoint.register_endpoint_instance.side_effect = None
    assert (await drain.resume({}))["state"] == "serving"
    assert not drain.withdrawn


@pytest.mark.asyncio
async def test_shutdown_supersedes_drain_during_discovery_withdrawal(endpoint, ready):
    terminal = asyncio.Event()
    endpoint.unregister_endpoint_instance.side_effect = terminal.set
    drain = WorkerDrain(
        endpoint,
        lock=asyncio.Lock(),
        shutdown_event=asyncio.Event(),
        terminal_event=terminal,
        ready_event=ready,
        is_paused=lambda: False,
        grace_period=0,
    )

    with pytest.raises(DrainConflict, match="shutdown superseded drain"):
        await drain.drain({})
    assert (await drain.status({}))["state"] == "stopping"


@pytest.mark.asyncio
@pytest.mark.parametrize("registration_error", [False, True])
async def test_shutdown_during_resume_undoes_registration(
    endpoint, ready, registration_error
):
    entered = asyncio.Event()
    release = asyncio.Event()
    shutdown = asyncio.Event()

    async def register():
        entered.set()
        await release.wait()
        if registration_error:
            raise RuntimeError("registration reply lost")

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
    assert (await drain.status({}))["accepting_requests"]
    shutdown.set()
    release.set()
    error_type = RuntimeError if registration_error else DrainConflict
    message = "registration reply lost" if registration_error else "superseded"
    with pytest.raises(error_type, match=message):
        await asyncio.wait_for(request, 1)
    status = await drain.status({})
    assert status["state"] == "stopping"
    assert status["admin_withdrawn"]
    assert not status["accepting_requests"]
    assert endpoint.unregister_endpoint_instance.await_count == 2
    with pytest.raises(DrainConflict, match="shutdown"):
        await drain.resume({})


@pytest.mark.asyncio
async def test_resume_clears_previous_deadline(endpoint, ready, clock):
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
    assert status["state"] == "withdrawing"
    clock.return_value += 30
    await drain.drain({})
    clock.return_value += 30
    assert not (await drain.status({}))["accepting_requests"]
    assert (await drain.resume({}))["accepting_requests"]
    clock.return_value += 60

    async def generate(request):
        yield "accepted"

    assert [item async for item in drain.wrap(generate)({})] == ["accepted"]
    await drain.resume({})
    assert endpoint.register_endpoint_instance.await_count == 2


@pytest.mark.asyncio
async def test_stream_close_reaches_engine_generator(endpoint, ready):
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
    await stream.aclose()
    assert closed.is_set()


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
async def test_drain_requires_running_generation(endpoint, ready):
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
    with pytest.raises(DrainConflict, match="continue engine"):
        await drain.drain({})
    endpoint.unregister_endpoint_instance.assert_not_awaited()

    paused = False
    await drain.drain({})
    with pytest.raises(DrainConflict, match="resume worker"):
        drain.check_generation_control()
    assert (await drain.resume({}))["state"] == "serving"
    drain.check_generation_control()


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


@pytest.mark.asyncio
@pytest.mark.parametrize("registration_error", [False, True])
async def test_resume_accepts_requests_before_registration_reply(
    endpoint, ready, registration_error
):
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
    entered, release = asyncio.Event(), asyncio.Event()

    async def register():
        entered.set()
        await release.wait()
        if registration_error:
            raise RuntimeError("registration reply lost")

    async def generate(request):
        yield "accepted"

    endpoint.register_endpoint_instance.side_effect = register
    resume = asyncio.create_task(drain.resume({}))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        assert [item async for item in drain.wrap(generate)({})] == ["accepted"]
    finally:
        release.set()
        if registration_error:
            with pytest.raises(RuntimeError, match="registration reply lost"):
                await asyncio.wait_for(resume, 1)
        else:
            await asyncio.wait_for(resume, 1)
    assert [item async for item in drain.wrap(generate)({})] == ["accepted"]


@pytest.mark.asyncio
async def test_admission_deadline_does_not_wait_for_engine_lock(endpoint, ready, clock):
    lock = asyncio.Lock()
    drain = WorkerDrain(
        endpoint,
        lock=lock,
        shutdown_event=asyncio.Event(),
        terminal_event=asyncio.Event(),
        ready_event=ready,
        is_paused=lambda: False,
        grace_period=10,
    )
    await drain.drain({})

    async def generate(request):
        yield "accepted"

    async with lock:
        clock.return_value += 10
        with pytest.raises(WorkerDraining):
            await anext(drain.wrap(generate)({}))
