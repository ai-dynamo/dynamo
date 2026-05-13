# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json

import pytest

from dynamo.llm import ModelInput, ModelRuntimeConfig, ModelType, register_model
from dynamo.runtime import DistributedRuntime

pytestmark = [
    pytest.mark.unit,
    pytest.mark.none,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]

SGLANG_WORKER_GROUP_ID_KEY = "sglang_worker_group_id"
ENDPOINT_PATH = "test.sglang.generate"


def _runtime() -> DistributedRuntime:
    return DistributedRuntime(asyncio.get_running_loop(), "file", "tcp")


async def _register_leader(
    runtime: DistributedRuntime, group_id, endpoint_path=ENDPOINT_PATH
):
    endpoint = runtime.endpoint(endpoint_path)
    await endpoint.register_endpoint_instance()

    runtime_config = ModelRuntimeConfig()
    runtime_config.data_parallel_size = 2
    runtime_config.set_engine_specific(
        SGLANG_WORKER_GROUP_ID_KEY,
        json.dumps(group_id),
    )
    await register_model(
        ModelInput.Tensor,
        ModelType.TensorBased,
        endpoint,
        "tensor",
        runtime_config=runtime_config,
    )
    return endpoint.connection_id()


@pytest.mark.asyncio
async def test_wait_for_instance_by_runtime_data_resolves_current_snapshot(
    temp_file_store,
):
    leader = _runtime()
    resolver = _runtime()
    try:
        leader_id = await _register_leader(leader, "group-a")
        client = await resolver.endpoint(ENDPOINT_PATH).client()

        worker_id = await client.wait_for_instance_by_runtime_data(
            SGLANG_WORKER_GROUP_ID_KEY,
            "group-a",
            timeout_s=5.0,
        )

        assert worker_id == leader_id
    finally:
        leader.shutdown()
        resolver.shutdown()


@pytest.mark.asyncio
async def test_wait_for_instance_by_runtime_data_resolves_from_watch_event(
    temp_file_store,
):
    leader = _runtime()
    resolver = _runtime()
    try:
        client = await resolver.endpoint(ENDPOINT_PATH).client()
        wait_task = asyncio.ensure_future(
            client.wait_for_instance_by_runtime_data(
                SGLANG_WORKER_GROUP_ID_KEY,
                "group-a",
                timeout_s=5.0,
            )
        )
        await asyncio.sleep(0)

        leader_id = await _register_leader(leader, "group-a")

        assert await wait_task == leader_id
    finally:
        leader.shutdown()
        resolver.shutdown()


@pytest.mark.asyncio
async def test_wait_for_instance_by_runtime_data_times_out_without_match(
    temp_file_store,
):
    resolver = _runtime()
    try:
        client = await resolver.endpoint(ENDPOINT_PATH).client()

        with pytest.raises(TimeoutError, match="last_match_count=0"):
            await client.wait_for_instance_by_runtime_data(
                SGLANG_WORKER_GROUP_ID_KEY,
                "missing-group",
                timeout_s=0.1,
            )
    finally:
        resolver.shutdown()


@pytest.mark.asyncio
async def test_wait_for_instance_by_runtime_data_times_out_on_duplicate_match(
    temp_file_store,
):
    leader_a = _runtime()
    leader_b = _runtime()
    resolver = _runtime()
    try:
        await _register_leader(leader_a, "group-a")
        await _register_leader(leader_b, "group-a")
        client = await resolver.endpoint(ENDPOINT_PATH).client()

        with pytest.raises(TimeoutError, match="last_match_count=2"):
            await client.wait_for_instance_by_runtime_data(
                SGLANG_WORKER_GROUP_ID_KEY,
                "group-a",
                timeout_s=0.1,
            )
    finally:
        leader_a.shutdown()
        leader_b.shutdown()
        resolver.shutdown()


@pytest.mark.asyncio
async def test_wait_for_instance_by_runtime_data_matches_only_json_strings(
    temp_file_store,
):
    leader = _runtime()
    resolver = _runtime()
    try:
        await _register_leader(leader, 123)
        client = await resolver.endpoint(ENDPOINT_PATH).client()

        with pytest.raises(TimeoutError, match="last_match_count=0"):
            await client.wait_for_instance_by_runtime_data(
                SGLANG_WORKER_GROUP_ID_KEY,
                "123",
                timeout_s=0.1,
            )
    finally:
        leader.shutdown()
        resolver.shutdown()


@pytest.mark.asyncio
async def test_wait_for_instance_by_runtime_data_links_two_multinode_groups(
    temp_file_store,
):
    leader_a = _runtime()
    leader_b = _runtime()
    nonleader_a = _runtime()
    nonleader_b = _runtime()
    try:
        client_a = await nonleader_a.endpoint(ENDPOINT_PATH).client()
        client_b = await nonleader_b.endpoint(ENDPOINT_PATH).client()
        wait_a = asyncio.ensure_future(
            client_a.wait_for_instance_by_runtime_data(
                SGLANG_WORKER_GROUP_ID_KEY,
                "group-a",
                timeout_s=5.0,
            )
        )
        wait_b = asyncio.ensure_future(
            client_b.wait_for_instance_by_runtime_data(
                SGLANG_WORKER_GROUP_ID_KEY,
                "group-b",
                timeout_s=5.0,
            )
        )
        await asyncio.sleep(0)

        leader_a_id = await _register_leader(leader_a, "group-a")
        leader_b_id = await _register_leader(leader_b, "group-b")

        assert await wait_a == leader_a_id
        assert await wait_b == leader_b_id
    finally:
        leader_a.shutdown()
        leader_b.shutdown()
        nonleader_a.shutdown()
        nonleader_b.shutdown()


@pytest.mark.asyncio
async def test_wait_for_instance_by_runtime_data_links_two_late_subscribers(
    temp_file_store,
):
    leader_a = _runtime()
    leader_b = _runtime()
    nonleader_a = _runtime()
    nonleader_b = _runtime()
    try:
        leader_a_id = await _register_leader(leader_a, "group-a")
        leader_b_id = await _register_leader(leader_b, "group-b")

        client_a = await nonleader_a.endpoint(ENDPOINT_PATH).client()
        client_b = await nonleader_b.endpoint(ENDPOINT_PATH).client()

        assert (
            await client_a.wait_for_instance_by_runtime_data(
                SGLANG_WORKER_GROUP_ID_KEY,
                "group-a",
                timeout_s=5.0,
            )
            == leader_a_id
        )
        assert (
            await client_b.wait_for_instance_by_runtime_data(
                SGLANG_WORKER_GROUP_ID_KEY,
                "group-b",
                timeout_s=5.0,
            )
            == leader_b_id
        )
    finally:
        leader_a.shutdown()
        leader_b.shutdown()
        nonleader_a.shutdown()
        nonleader_b.shutdown()
