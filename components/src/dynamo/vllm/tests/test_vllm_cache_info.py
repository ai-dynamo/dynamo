# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine startup metadata is authoritative even without KV event publishing."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from dynamo.vllm.cache_info import (
    DYNAMO_KV_CACHE_GROUP_COUNT_KEY,
    DYNAMO_KV_EVENT_BLOCK_SIZE_KEY,
    configure_kv_event_block_size,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.core,
]


def _group(index=0, kind="full_attention", block_size=32):
    return {"group_idx": index, "kind": kind, "block_size": block_size}


@pytest.mark.asyncio
@pytest.mark.parametrize("groups", [[_group()], [_group(), _group(1, "mla_attention")]])
async def test_startup_records_actual_groups_without_kv_events(groups):
    engine = SimpleNamespace(
        engine_core=SimpleNamespace(call_utility_async=AsyncMock(return_value=groups))
    )
    config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=16),
        additional_config={DYNAMO_KV_CACHE_GROUP_COUNT_KEY: 99},
        kv_events_config=None,
    )
    assert await configure_kv_event_block_size(engine, config) == 32
    assert config.additional_config[DYNAMO_KV_CACHE_GROUP_COUNT_KEY] == len(groups)
    assert config.additional_config[DYNAMO_KV_EVENT_BLOCK_SIZE_KEY] == 32
    assert config.cache_config.block_size == 16
    engine.engine_core.call_utility_async.assert_awaited_once_with(
        "get_kv_cache_group_metadata"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "groups",
    [
        None,
        {},
        [],
        [None],
        [_group(index=True)],
        [_group(index=1)],
        [_group(block_size=0)],
        [_group(block_size=True)],
        [_group(kind=[])],
        [{"kind": "full_attention", "block_size": 32}],
    ],
)
async def test_unavailable_or_malformed_group_metadata_cannot_authorize_reuse(groups):
    engine = SimpleNamespace(
        engine_core=SimpleNamespace(call_utility_async=AsyncMock(return_value=groups))
    )
    config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=16),
        additional_config={DYNAMO_KV_CACHE_GROUP_COUNT_KEY: 1},
    )
    block_size = await configure_kv_event_block_size(engine, config)
    assert type(block_size) is int and block_size > 0
    assert DYNAMO_KV_CACHE_GROUP_COUNT_KEY not in config.additional_config
    assert config.cache_config.block_size == 16


@pytest.mark.asyncio
async def test_failed_metadata_rpc_clears_configuration_spoof_and_keeps_inference_defaults():
    engine = SimpleNamespace(
        engine_core=SimpleNamespace(
            call_utility_async=AsyncMock(
                side_effect=RuntimeError("unsupported utility")
            )
        )
    )
    config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=16),
        additional_config={DYNAMO_KV_CACHE_GROUP_COUNT_KEY: 1},
    )
    assert await configure_kv_event_block_size(engine, config) == 16
    assert DYNAMO_KV_CACHE_GROUP_COUNT_KEY not in config.additional_config
    assert config.additional_config[DYNAMO_KV_EVENT_BLOCK_SIZE_KEY] == 16
