# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import uuid

import pytest

from dynamo._core import DistributedRuntime
from dynamo.common.utils.worker_shutdown import WorkerShutdown, serve_endpoint

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.integration,
    pytest.mark.gpu_0,
    pytest.mark.core,
    pytest.mark.timeout(30),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("push", [False, True], ids=["pull", "push"])
async def test_python_draining_rejection_survives_tcp(push, monkeypatch):
    """A closed gate must reach the caller as Unavailable on either transport."""
    monkeypatch.setenv("DYN_SYSTEM_PORT", "0")  # OS-assigned, no shared port.
    monkeypatch.setenv("DYN_GRACEFUL_SHUTDOWN_GRACE_PERIOD_SECS", "0")
    monkeypatch.setenv("DYN_WORKER_GRACEFUL_SHUTDOWN_TIMEOUT", "5")
    runtime = DistributedRuntime(
        asyncio.get_running_loop(), "mem", "tcp", event_plane="zmq"
    )
    endpoint = runtime.endpoint(f"shutdown{uuid.uuid4().hex}.worker.generate")
    shutdown = WorkerShutdown(runtime, [endpoint], asyncio.Event())
    cleaned = asyncio.Event()

    async def pull_handler(request, context=None):
        yield {"value": 1}

    async def push_handler(request, context=None, response_sender=None):
        if response_sender is None:
            yield {"value": 1}
        else:
            response_sender.send({"value": 1})
            response_sender.close()

    async def worker():
        try:
            await serve_endpoint(
                endpoint, push_handler if push else pull_handler, shutdown=shutdown
            )
        finally:
            cleaned.set()

    serving = asyncio.create_task(shutdown.run(worker(), install_signals=False))
    try:
        client = await endpoint.client()
        instances = await asyncio.wait_for(client.wait_for_instances(), 10)
        stream = await client.direct({}, instances[0], annotated=False)
        assert [item async for item in stream] == [{"value": 1}]
        shutdown.accepting = False
        stream = await client.direct({}, instances[0])
        with pytest.raises(ValueError, match="Unavailable: worker is not accepting"):
            async for _ in stream:
                pytest.fail("a draining worker emitted data")
    finally:
        shutdown.request_shutdown()
        await serving
    assert cleaned.is_set()
