# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
import uuid
from dataclasses import dataclass

import pytest

from dynamo.llm import (
    KvRouter,
    KvRouterConfig,
    ModelInput,
    ModelType,
    WorkerType,
    register_model,
)
from dynamo.runtime import DistributedRuntime

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.integration,
]


async def _generate_error(_request, _context=None):
    raise RuntimeError("intentional KV-router failure")
    yield


@dataclass
class _Worker:
    """One endpoint served on its own runtime, torn down the way a fixture does."""

    endpoint_path: str
    runtime: DistributedRuntime
    server_task: asyncio.Task
    stopped: bool = False

    @classmethod
    async def start(cls, endpoint_path):
        runtime = DistributedRuntime(asyncio.get_running_loop(), "file", "tcp")
        endpoint = runtime.endpoint(endpoint_path)
        await register_model(
            ModelInput.Tensor,
            ModelType.TensorBased,
            endpoint,
            "test-router-worker",
            worker_type=WorkerType.Aggregated,
            tensor_model_config={
                "name": "test-router-worker",
                "inputs": [],
                "outputs": [],
            },
        )
        server_task = asyncio.ensure_future(endpoint.serve_endpoint(_generate_error))
        return cls(endpoint_path, runtime, server_task)

    async def stop(self):
        if self.stopped:
            return
        self.stopped = True
        self.server_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self.server_task
        self.runtime.shutdown()


async def _wait_for_single_instance(endpoint):
    client = await endpoint.client()
    instances = await client.wait_for_instances()
    assert len(instances) == 1
    return client


async def _generate_and_collect(router, response_buffer_size):
    stream = await router.generate(
        [1, 2, 3],
        "test-model",
        response_buffer_size=response_buffer_size,
    )
    return [response async for response in stream]


@pytest.fixture
async def router_runtime(temp_file_store):
    runtime = DistributedRuntime(asyncio.get_running_loop(), "file", "tcp")
    yield runtime
    runtime.shutdown()


@pytest.fixture
async def error_router_endpoint(router_runtime):
    worker = await _Worker.start(f"error-router-{uuid.uuid4().hex}.worker.generate")
    endpoint = router_runtime.endpoint(worker.endpoint_path)
    await _wait_for_single_instance(endpoint)
    yield endpoint
    await worker.stop()


@pytest.fixture
async def worker_pair(router_runtime):
    suffix = uuid.uuid4().hex
    workers = [
        await _Worker.start(f"worker-{name}-{suffix}.worker.generate")
        for name in ("a", "b")
    ]
    yield workers
    for worker in workers:
        await worker.stop()


@pytest.mark.asyncio
@pytest.mark.timeout(30)
@pytest.mark.parametrize("response_buffer_size", [0, 100])
async def test_kv_router_propagates_stream_errors(
    error_router_endpoint, response_buffer_size, monkeypatch
):
    # Wait for the router's worker watcher to observe the registered endpoint before
    # issuing the request. The fixture's endpoint client has a separate watcher.
    monkeypatch.setenv("DYN_ROUTER_MIN_INITIAL_WORKERS", "1")
    router = KvRouter(
        error_router_endpoint,
        4,
        KvRouterConfig(use_kv_events=False),
    )

    with pytest.raises(ValueError, match="intentional KV-router failure"):
        await _generate_and_collect(router, response_buffer_size)


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_worker_teardown_leaves_sibling_worker_reachable(
    worker_pair, router_runtime, monkeypatch
):
    """Regression test for ai-dynamo/dynamo#14261."""
    monkeypatch.setenv("DYN_ROUTER_MIN_INITIAL_WORKERS", "1")
    torn_down, survivor = worker_pair
    torn_down_client = await _wait_for_single_instance(
        router_runtime.endpoint(torn_down.endpoint_path)
    )
    survivor_endpoint = router_runtime.endpoint(survivor.endpoint_path)
    await _wait_for_single_instance(survivor_endpoint)

    await torn_down.stop()
    # The endpoint's detached cleanup task unregisters from discovery and then from the
    # shared TCP server, so the instance disappearing is the cue that the handler sweep
    # is about to run; the short sleep covers the rest of that task.
    while torn_down_client.instance_ids():
        await asyncio.sleep(0.05)
    await asyncio.sleep(0.2)

    router = KvRouter(survivor_endpoint, 4, KvRouterConfig(use_kv_events=False))
    with pytest.raises(ValueError, match="intentional KV-router failure"):
        await asyncio.wait_for(_generate_and_collect(router, 100), timeout=10)
