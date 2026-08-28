# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
import uuid

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


@pytest.fixture
async def error_router_endpoint(temp_file_store):
    endpoint_path = f"error-router-{uuid.uuid4().hex}.worker.generate"
    loop = asyncio.get_running_loop()
    worker_runtime = DistributedRuntime(loop, "file", "tcp")
    router_runtime = DistributedRuntime(loop, "file", "tcp")
    worker_endpoint = worker_runtime.endpoint(endpoint_path)
    await register_model(
        ModelInput.Tensor,
        ModelType.TensorBased,
        worker_endpoint,
        "test-router-worker",
        worker_type=WorkerType.Aggregated,
        tensor_model_config={
            "name": "test-router-worker",
            "inputs": [],
            "outputs": [],
        },
    )
    server_task = asyncio.ensure_future(worker_endpoint.serve_endpoint(_generate_error))
    endpoint = router_runtime.endpoint(endpoint_path)
    client = await endpoint.client()
    try:
        instances = await client.wait_for_instances()
        assert len(instances) == 1
        yield endpoint
    finally:
        server_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await server_task
        router_runtime.shutdown()
        worker_runtime.shutdown()


async def _drain_once(router, response_buffer_size):
    stream = await router.generate(
        [1, 2, 3],
        "test-model",
        response_buffer_size=response_buffer_size,
    )
    return [response async for response in stream]


async def _drain_when_routable(router, response_buffer_size, timeout=10.0):
    """Drain the router once it has a worker to route to.

    The endpoint fixture waits on discovery, but KvRouter tracks workers on its
    own background task, so a generate() issued right after construction can lose
    that race and fail with "no endpoints available to route work" instead of the
    stream error under test. Retry only that condition; every other exception --
    including the intentional failure this test asserts on -- propagates
    immediately, so a genuine regression still fails the test.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        try:
            return await _drain_once(router, response_buffer_size)
        except Exception as exc:
            if "no endpoints available to route work" not in str(exc):
                raise
            if loop.time() >= deadline:
                raise AssertionError(
                    f"KvRouter never discovered a worker within {timeout}s: {exc}"
                ) from exc
            await asyncio.sleep(0.02)


@pytest.mark.asyncio
@pytest.mark.timeout(30)
@pytest.mark.parametrize("response_buffer_size", [0, 100])
async def test_kv_router_propagates_stream_errors(
    error_router_endpoint, response_buffer_size
):
    router = KvRouter(
        error_router_endpoint,
        4,
        KvRouterConfig(use_kv_events=False),
    )

    with pytest.raises(ValueError, match="intentional KV-router failure"):
        await _drain_when_routable(router, response_buffer_size)
