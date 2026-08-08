# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
import uuid
from pathlib import Path

import pytest

from dynamo.llm import (
    KvRouter,
    KvRouterConfig,
    ModelInput,
    ModelType,
    RouterMode,
    WorkerType,
    register_model,
)
from dynamo.runtime import DistributedRuntime

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.integration,
]


async def _generate(_request, _context=None):
    yield {"token_ids": [4], "finish_reason": "stop"}


@pytest.fixture
async def router_endpoint(temp_file_store):
    endpoint_path = f"router-{uuid.uuid4().hex}.worker.generate"
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
    server_task = asyncio.ensure_future(worker_endpoint.serve_endpoint(_generate))
    endpoint = router_runtime.endpoint(endpoint_path)
    client = await endpoint.client()
    try:
        instances = await client.wait_for_instances()
        assert len(instances) == 1
        yield endpoint, instances[0]
    finally:
        server_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await server_task
        router_runtime.shutdown()
        worker_runtime.shutdown()


@pytest.fixture
async def bare_router_endpoint(temp_file_store):
    endpoint_path = f"bare-router-{uuid.uuid4().hex}.worker.generate"
    loop = asyncio.get_running_loop()
    worker_runtime = DistributedRuntime(loop, "file", "tcp")
    router_runtime = DistributedRuntime(loop, "file", "tcp")
    worker_endpoint = worker_runtime.endpoint(endpoint_path)
    server_task = asyncio.ensure_future(worker_endpoint.serve_endpoint(_generate))
    endpoint = router_runtime.endpoint(endpoint_path)
    client = await endpoint.client()
    try:
        instances = await client.wait_for_instances()
        assert len(instances) == 1
        yield endpoint, instances[0], Path(temp_file_store)
    finally:
        server_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await server_task
        router_runtime.shutdown()
        worker_runtime.shutdown()


def _router(endpoint, mode):
    if mode == RouterMode.KV:
        return KvRouter(
            endpoint,
            4,
            KvRouterConfig(use_kv_events=False),
        )
    return KvRouter(endpoint, router_mode=mode)


@pytest.mark.asyncio
@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "mode",
    [
        RouterMode.RoundRobin,
        RouterMode.Random,
        RouterMode.PowerOfTwoChoices,
        RouterMode.LeastLoaded,
        RouterMode.DeviceAwareWeighted,
        RouterMode.Direct,
        RouterMode.KV,
    ],
)
async def test_kv_router_generate_supports_every_mode(router_endpoint, mode):
    endpoint, worker_id = router_endpoint
    router = _router(endpoint, mode)

    if mode == RouterMode.Direct:
        with pytest.raises(ValueError, match="explicit worker target"):
            await router.best_worker([1, 2, 3])
    else:
        selection_options = {}
        if mode == RouterMode.DeviceAwareWeighted:
            selection_options = {
                "multi_modal_data": {
                    "image_url": [{"RawUrl": "https://example.invalid/image.png"}]
                },
                "mm_routing_info": {
                    "routing_token_ids": [1, 2, 3],
                    "block_mm_infos": [None],
                    "expanded_prompt_len": 3,
                },
            }
        selected_worker, dp_rank, overlap = await router.best_worker(
            [1, 2, 3], **selection_options
        )
        assert selected_worker == worker_id
        assert dp_rank == (0 if mode == RouterMode.KV else None)
        assert overlap == 0

    stream = await router.generate(
        [1, 2, 3],
        "test-model",
        worker_id=worker_id if mode == RouterMode.Direct else None,
    )
    responses = [response async for response in stream]
    assert responses


@pytest.mark.asyncio
@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "mode",
    [RouterMode.RoundRobin, RouterMode.Random, RouterMode.LeastLoaded],
)
async def test_non_kv_reservation_ids_are_atomic_and_reusable(router_endpoint, mode):
    endpoint, worker_id = router_endpoint
    router = KvRouter(endpoint, router_mode=mode)

    outcomes = await asyncio.gather(
        router.best_worker([1, 2, 3], request_id="reservation"),
        router.best_worker([1, 2, 3], request_id="reservation"),
        return_exceptions=True,
    )
    successes = [outcome for outcome in outcomes if not isinstance(outcome, Exception)]
    failures = [outcome for outcome in outcomes if isinstance(outcome, Exception)]
    assert successes == [(worker_id, None, 0)]
    assert len(failures) == 1
    assert isinstance(failures[0], ValueError)
    assert "active routing reservation" in str(failures[0])

    await router.free("reservation")
    await router.free("reservation")
    assert await router.best_worker([1, 2, 3], request_id="reservation") == (
        worker_id,
        None,
        0,
    )
    await router.free("reservation")


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_non_kv_rejects_kv_only_options_and_introspection(router_endpoint):
    endpoint, _ = router_endpoint
    config = KvRouterConfig(use_kv_events=False)

    with pytest.raises(ValueError, match="block_size is only valid"):
        KvRouter(endpoint, block_size=4, router_mode=RouterMode.RoundRobin)
    with pytest.raises(ValueError, match="kv_router_config is only valid"):
        KvRouter(
            endpoint,
            kv_router_config=config,
            router_mode=RouterMode.RoundRobin,
        )
    with pytest.raises(ValueError, match="only valid for DeviceAwareWeighted"):
        KvRouter(
            endpoint,
            router_mode=RouterMode.RoundRobin,
            enable_multimodal_cache_indexer=True,
        )

    router = KvRouter(endpoint, router_mode=RouterMode.Random)
    with pytest.raises(ValueError, match="only available in KV routing mode"):
        await router.get_potential_loads([1, 2, 3])


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_non_kv_modes_need_no_kv_prerequisites_or_registrations(
    bare_router_endpoint, monkeypatch
):
    endpoint, worker_id, discovery_root = bare_router_endpoint
    monkeypatch.setenv("DYN_ROUTER_MIN_INITIAL_WORKERS", "poison-kv-initialization")

    for mode in [
        RouterMode.RoundRobin,
        RouterMode.Random,
        RouterMode.PowerOfTwoChoices,
        RouterMode.LeastLoaded,
        RouterMode.DeviceAwareWeighted,
        RouterMode.Direct,
    ]:
        router = KvRouter(endpoint, router_mode=mode)
        stream = await router.generate(
            [1, 2, 3],
            "test-model",
            worker_id=worker_id if mode == RouterMode.Direct else None,
        )
        assert [response async for response in stream]
        del router

    assert not (discovery_root / "v1" / "event_channels").exists()
