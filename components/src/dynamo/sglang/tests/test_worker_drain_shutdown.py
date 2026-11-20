# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import signal
from contextlib import aclosing
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from dynamo._core import WorkerDraining
from dynamo.common.utils import worker_shutdown
from dynamo.sglang import shutdown
from dynamo.sglang.drain import DrainConflict
from dynamo.sglang.drain_controls import configure_worker_drain

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


@pytest.mark.asyncio
async def test_sigterm_without_system_listener_preserves_grace_and_rejects_late_work(
    monkeypatch,
):
    loop = asyncio.get_running_loop()
    handlers = {}
    unregistered, release = asyncio.Event(), asyncio.Event()
    shutdown_event, ready = asyncio.Event(), asyncio.Event()
    ready.set()
    endpoint = SimpleNamespace(
        unregister_endpoint_instance=AsyncMock(side_effect=unregistered.set),
        register_endpoint_instance=AsyncMock(),
    )
    handler = SimpleNamespace(
        generate_endpoint=endpoint,
        _engine_route_lock=asyncio.Lock(),
        engine=SimpleNamespace(tokenizer_manager=SimpleNamespace(is_pause=False)),
    )
    runtime = SimpleNamespace(
        system_status_server_url=Mock(return_value=None),
        register_engine_http_handler=Mock(),
        close_engine_routes=Mock(return_value=False),
        shutdown_and_wait=AsyncMock(),
    )
    monkeypatch.setenv("DYN_GRACEFUL_SHUTDOWN_GRACE_PERIOD_SECS", "0.05")
    monkeypatch.setenv("DYN_WORKER_SHUTDOWN_TOTAL_TIMEOUT_SECS", "10")
    monkeypatch.setattr(worker_shutdown, "ShutdownWatchdog", Mock())
    coordinator = worker_shutdown.WorkerShutdown(runtime, [endpoint], shutdown_event)
    controller = configure_worker_drain(
        handler, runtime, ready, shutdown_event, coordinator.requested_event
    )
    runtime.register_engine_http_handler.assert_not_called()

    async def generate(request, context):
        yield "first"
        await release.wait()
        yield "last"

    admitted = coordinator.wrap(controller.wrap(generate))
    monkeypatch.setattr(
        signal, "signal", lambda signum, callback: handlers.update({signum: callback})
    )
    with shutdown.defer_engine_signals(loop, coordinator) as deferred:
        coordinator.post_shutdown = deferred
        owner = asyncio.create_task(
            coordinator.run(asyncio.Event().wait(), install_signals=False)
        )
        try:
            async with aclosing(admitted({}, None)) as original:
                assert await anext(original) == "first"
                handlers[signal.SIGTERM](signal.SIGTERM, None)
                await asyncio.wait_for(unregistered.wait(), 1)
                assert coordinator.requested_event.is_set()
                assert not shutdown_event.is_set()
                with pytest.raises(DrainConflict, match="shutdown has started"):
                    await controller.resume({})
                async with aclosing(admitted({}, None)) as during_grace:
                    assert await anext(during_grace) == "first"
                async with asyncio.timeout(1):
                    while coordinator.accepting:
                        await asyncio.sleep(0.001)
                with pytest.raises(WorkerDraining):
                    await anext(admitted({}, None))
                runtime.shutdown_and_wait.assert_not_called()
                release.set()
                assert await asyncio.wait_for(anext(original), 1) == "last"
        finally:
            release.set()
            if not coordinator.started:
                coordinator.request_shutdown()
            await asyncio.wait_for(owner, 2)
    endpoint.register_endpoint_instance.assert_not_called()
    runtime.shutdown_and_wait.assert_awaited_once()
