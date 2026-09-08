# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest

from dynamo.llm import (
    ModelInput,
    ModelRuntimeConfig,
    ModelType,
    WorkerType,
    register_model,
    update_model_taints,
)
from dynamo.runtime import DistributedRuntime, Endpoint

pytestmark = [
    pytest.mark.unit,
    pytest.mark.none,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]

ENDPOINT_PATH = "test.taints.generate"


def _runtime() -> DistributedRuntime:
    return DistributedRuntime(asyncio.get_running_loop(), "file", "tcp")


async def _register_worker(runtime: DistributedRuntime, taints: set[str]) -> Endpoint:
    endpoint = runtime.endpoint(ENDPOINT_PATH)
    await endpoint.register_endpoint_instance()

    runtime_config = ModelRuntimeConfig()
    runtime_config.taints = taints
    await register_model(
        ModelInput.Tensor,
        ModelType.TensorBased,
        endpoint,
        "tensor",
        runtime_config=runtime_config,
        worker_type=WorkerType.Aggregated,
    )
    return endpoint


async def _snapshot(runtime: DistributedRuntime, only_live: bool = True):
    return await runtime.endpoint(ENDPOINT_PATH).list_endpoint_taints(
        only_live=only_live
    )


@pytest.mark.asyncio
async def test_list_endpoint_taints_snapshots_both_live_workers(temp_file_store):
    worker_a = _runtime()
    worker_b = _runtime()
    observer = _runtime()
    try:
        endpoint_a = await _register_worker(worker_a, {"pool=fast", "zone=zone-a"})
        endpoint_b = await _register_worker(worker_b, {"pool=slow", "zone=zone-b"})

        snapshot = await _snapshot(observer)

        assert snapshot == {
            str(endpoint_a.connection_id()): {"pool=fast", "zone=zone-a"},
            str(endpoint_b.connection_id()): {"pool=slow", "zone=zone-b"},
        }
    finally:
        worker_a.shutdown()
        worker_b.shutdown()
        observer.shutdown()


@pytest.mark.asyncio
async def test_list_endpoint_taints_only_live_filters_sleeping_worker(
    temp_file_store,
):
    worker = _runtime()
    observer = _runtime()
    try:
        endpoint = await _register_worker(worker, {"pool=fast"})

        # Sleeping worker: unregister the endpoint instance but keep the
        # advertised model card. only_live=True must filter it out while
        # only_live=False still reports its advertised taints.
        await endpoint.unregister_endpoint_instance()

        assert await _snapshot(observer) == {}

        advertised = await _snapshot(observer, only_live=False)
        assert advertised == {str(endpoint.connection_id()): {"pool=fast"}}
    finally:
        worker.shutdown()
        observer.shutdown()


@pytest.mark.asyncio
async def test_list_endpoint_taints_reports_worker_without_taints(temp_file_store):
    worker = _runtime()
    observer = _runtime()
    try:
        endpoint = await _register_worker(worker, set())

        snapshot = await _snapshot(observer)

        assert snapshot == {str(endpoint.connection_id()): set()}
    finally:
        worker.shutdown()
        observer.shutdown()


@pytest.mark.asyncio
async def test_list_endpoint_taints_reflects_updates_and_excludes_lora(
    temp_file_store,
):
    worker = _runtime()
    observer = _runtime()
    try:
        endpoint = await _register_worker(worker, {"pool=fast"})
        await register_model(
            ModelInput.Tensor,
            ModelType.TensorBased,
            endpoint,
            "tensor",
            worker_type=WorkerType.Aggregated,
            lora_name="adapterA",
            base_model_path="tensor",
            runtime_config=ModelRuntimeConfig(),
        )

        await update_model_taints(endpoint, {"pool=slow", "zone=zone-b"})

        assert await _snapshot(observer) == {
            str(endpoint.connection_id()): {"pool=slow", "zone=zone-b"}
        }
    finally:
        worker.shutdown()
        observer.shutdown()
