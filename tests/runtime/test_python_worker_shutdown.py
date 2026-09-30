# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import uuid

import httpx
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
@pytest.mark.parametrize("request_plane", ["tcp", "nats"])
@pytest.mark.parametrize("discovery_backend", ["mem"])
@pytest.mark.parametrize("event_plane", ["zmq"])
async def test_python_draining_rejection_survives_transport(
    push,
    monkeypatch,
    request_plane,
    discovery_backend,
    event_plane,
    runtime_services_dynamic_ports,
    dynamo_dynamic_ports,
):
    """A closed gate must reach the caller as Unavailable on either transport."""
    system_port = dynamo_dynamic_ports.system_ports[0]
    monkeypatch.setenv("DYN_SYSTEM_PORT", str(system_port))
    monkeypatch.setenv("DYN_GRACEFUL_SHUTDOWN_GRACE_PERIOD_SECS", "0")
    monkeypatch.setenv("DYN_WORKER_GRACEFUL_SHUTDOWN_TIMEOUT", "5")
    runtime = DistributedRuntime(
        asyncio.get_running_loop(),
        discovery_backend,
        request_plane,
        event_plane=event_plane,
    )
    endpoint = runtime.endpoint(f"shutdown{uuid.uuid4().hex}.worker.generate")
    shutdown = WorkerShutdown(runtime, [endpoint], asyncio.Event())
    cleaned = asyncio.Event()
    cleanup_entered = asyncio.Event()
    release_cleanup = asyncio.Event()

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
            cleanup_entered.set()
            await release_cleanup.wait()
            cleaned.set()

    serving = asyncio.create_task(shutdown.run(worker(), install_signals=False))
    requested = False
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
        requested = True
        shutdown.request_shutdown()
        await cleanup_entered.wait()
        if request_plane == "tcp" and not push:
            # One representative case checks publication through the shared handle.
            async with httpx.AsyncClient(timeout=2, trust_env=False) as http:
                response = await http.get(f"http://127.0.0.1:{system_port}/metrics")
                response.raise_for_status()
            samples = [
                line for line in response.text.splitlines() if not line.startswith("#")
            ]
            assert any(
                "shutdown_stage_seconds{" in line
                and 'stage="cleanup"' in line
                and 'reason="started"' in line
                for line in samples
            )
            assert any(
                "shutdown_inflight_requests{" in line and line.endswith(" 0")
                for line in samples
            )
            assert any(
                "shutdown_kv_quiescent{" in line and line.endswith(" -1")
                for line in samples
            )
    finally:
        release_cleanup.set()
        if not requested:
            shutdown.request_shutdown()
        await serving
    assert cleaned.is_set()
