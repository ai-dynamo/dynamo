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
    """One endpoint served on its own runtime."""

    endpoint_path: str
    runtime: DistributedRuntime
    server_task: asyncio.Task
    stopped: bool = False

    @classmethod
    @contextlib.asynccontextmanager
    async def serve(cls, endpoint_path):
        runtime = DistributedRuntime(asyncio.get_running_loop(), "file", "tcp")
        try:
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
            server_task = asyncio.ensure_future(
                endpoint.serve_endpoint(_generate_error)
            )
        except BaseException:
            runtime.shutdown()
            raise
        worker = cls(endpoint_path, runtime, server_task)
        try:
            yield worker
        finally:
            await worker.stop()

    async def stop(self):
        """Shut the runtime down and return once the endpoint's cleanup has finished.

        `serve_endpoint` resolves only after the endpoint has unregistered from discovery
        and from the request-plane server, so awaiting it after `shutdown()` is the barrier
        that proves the handler removal has run.
        """
        if self.stopped:
            return
        self.stopped = True
        self.runtime.shutdown()
        await asyncio.wait_for(self.server_task, timeout=10)


async def _wait_for_single_instance(endpoint):
    client = await endpoint.client()
    assert len(await client.wait_for_instances()) == 1


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
    async with _Worker.serve(
        f"error-router-{uuid.uuid4().hex}.worker.generate"
    ) as worker:
        endpoint = router_runtime.endpoint(worker.endpoint_path)
        await _wait_for_single_instance(endpoint)
        yield endpoint


@pytest.fixture
async def worker_pair(router_runtime):
    suffix = uuid.uuid4().hex
    async with contextlib.AsyncExitStack() as stack:
        yield [
            await stack.enter_async_context(
                _Worker.serve(f"worker-{name}-{suffix}.worker.generate")
            )
            for name in ("a", "b")
        ]


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
    # Both handlers must be registered before the teardown, or the removal it triggers
    # has nothing of the survivor's to remove and the test proves nothing.
    await _wait_for_single_instance(router_runtime.endpoint(torn_down.endpoint_path))
    survivor_endpoint = router_runtime.endpoint(survivor.endpoint_path)
    await _wait_for_single_instance(survivor_endpoint)

    # Returns after the torn-down endpoint's handler removal has run, so the survivor's
    # request is issued after that removal rather than racing it.
    await torn_down.stop()

    router = KvRouter(survivor_endpoint, 4, KvRouterConfig(use_kv_events=False))
    with pytest.raises(ValueError, match="intentional KV-router failure"):
        await asyncio.wait_for(_generate_and_collect(router, 100), timeout=10)


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_prefetch_targets_worker_without_generation(router_runtime, monkeypatch):
    monkeypatch.setenv("DYN_ROUTER_MIN_INITIAL_WORKERS", "1")
    component = f"prefetch-{uuid.uuid4().hex}.worker"
    calls = []
    tasks = []
    async with contextlib.AsyncExitStack() as stack:
        workers = [
            await stack.enter_async_context(_Worker.serve(f"{component}.generate"))
            for _ in range(2)
        ]
        try:
            for index, worker in enumerate(workers):

                async def prefetch(request, _context=None, index=index):
                    calls.append((index, request))
                    if request["token_ids"] == [0]:
                        raise RuntimeError("intentional prefetch failure")
                    yield {"success": True, "prefetched_tokens": 4, "cached_tokens": 4}

                endpoint = worker.runtime.endpoint(f"{component}.prefetch")
                tasks.append(asyncio.ensure_future(endpoint.serve_endpoint(prefetch)))
            control_client = await router_runtime.endpoint(
                f"{component}.prefetch"
            ).client()
            while len(await control_client.wait_for_instances()) != 2:
                await asyncio.sleep(0.01)
            endpoint = router_runtime.endpoint(f"{component}.generate")
            await (await endpoint.client()).wait_for_instances()
            router = KvRouter(endpoint, 4, KvRouterConfig(use_kv_events=False))
            tokens = [1, 2, 3, 4, 5]
            for index, worker in enumerate(workers):
                worker_id = worker.runtime.endpoint(
                    f"{component}.generate"
                ).connection_id()
                result = await router.prefetch(worker_id, tokens, cache_salt="tenant-a")
                assert result == {
                    "success": True,
                    "prefetched_tokens": 4,
                    "cached_tokens": 4,
                }
                assert calls[-1] == (
                    index,
                    {"token_ids": tokens, "cache_salt": "tenant-a"},
                )
            assert len(calls) == 2
            with pytest.raises(Exception, match="intentional prefetch failure"):
                await router.prefetch(worker_id, [0])
            with pytest.raises(ValueError, match="must not be empty"):
                await router.prefetch(worker_id, [])
        finally:
            for worker in workers:
                worker.runtime.shutdown()
            await asyncio.wait_for(asyncio.gather(*tasks), timeout=10)
