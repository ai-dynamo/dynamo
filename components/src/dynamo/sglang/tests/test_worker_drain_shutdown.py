# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import signal
from contextlib import aclosing
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from dynamo._core import WorkerDraining
from dynamo.common.utils import graceful_shutdown
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
    unregistered, completed, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    terminal_event, shutdown_event, ready = (
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
    )
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
        shutdown=completed.set,
    )
    monkeypatch.setenv("DYN_GRACEFUL_SHUTDOWN_GRACE_PERIOD_SECS", "0.05")
    monkeypatch.setattr(graceful_shutdown, "_shutdown_started", asyncio.Event())
    controller = configure_worker_drain(
        handler, runtime, ready, shutdown_event, terminal_event
    )
    runtime.register_engine_http_handler.assert_not_called()

    async def generate(request, context):
        yield "first"
        await release.wait()
        yield "last"

    admitted = controller.wrap(generate)
    async with aclosing(admitted({}, None)) as original:
        assert await anext(original) == "first"
        monkeypatch.setattr(
            signal,
            "signal",
            lambda signum, callback: handlers.update({signum: callback}),
        )
        # Restore the loop's method after the production installer wraps it.
        monkeypatch.setattr(loop, "add_signal_handler", loop.add_signal_handler)
        deferred = shutdown.install_graceful_shutdown(
            loop,
            runtime,
            [endpoint],
            shutdown_event,
            signals=(signal.SIGTERM,),
            terminal_event=terminal_event,
        )
        handlers[signal.SIGTERM](signal.SIGTERM, None)
        await asyncio.wait_for(unregistered.wait(), 1)
        assert terminal_event.is_set()
        assert not shutdown_event.is_set()
        with pytest.raises(DrainConflict, match="shutdown has started"):
            await controller.resume({})
        async with aclosing(admitted({}, None)) as during_grace:
            assert await anext(during_grace) == "first"
        await asyncio.wait_for(completed.wait(), 1)
        assert shutdown_event.is_set()
        with pytest.raises(WorkerDraining):
            await anext(admitted({}, None))
        release.set()
        assert await asyncio.wait_for(anext(original), 1) == "last"
        endpoint.register_endpoint_instance.assert_not_called()
        await deferred()
