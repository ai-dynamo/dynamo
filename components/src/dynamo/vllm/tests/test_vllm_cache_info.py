# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from dynamo.vllm.cache_info import (
    configure_kv_event_block_size,
    get_configured_kv_event_block_size,
    select_main_attention_block_size,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind,logical_size",
    [("full_attention", 32), ("mla_attention", 1056), ("sink_full_attention", 16)],
)
async def test_configure_uses_main_attention_logical_size(kind, logical_size):
    engine = SimpleNamespace(
        get_kv_cache_group_metadata=AsyncMock(
            return_value=[
                {
                    "group_id": 0,
                    "kind": "mamba",
                    "block_size": 64,
                    "logical_block_size": 64,
                },
                {
                    "group_id": 1,
                    "kind": kind,
                    "block_size": 16,
                    "logical_block_size": logical_size,
                },
            ]
        )
    )
    config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=16), additional_config=None
    )

    assert await configure_kv_event_block_size(engine, config) == logical_size
    assert get_configured_kv_event_block_size(config) == logical_size
    assert config.cache_config.block_size == 16
    engine.get_kv_cache_group_metadata.assert_awaited_once_with()


@pytest.mark.parametrize("logical_size", [None, 0, -1, True, "32"])
def test_rejects_invalid_logical_size(logical_size):
    group = {"kind": "full_attention", "block_size": 16}
    if logical_size is not None:
        group["logical_block_size"] = logical_size
    with pytest.raises(ValueError, match="positive integer logical_block_size"):
        select_main_attention_block_size([group], 16)


def test_distinguishes_unavailable_metadata_from_no_cache_groups():
    with pytest.raises(ValueError, match="does not provide initialized"):
        select_main_attention_block_size(None, 16)
    assert select_main_attention_block_size([], 16) == 16


@pytest.mark.parametrize(
    "groups",
    [
        [{"kind": "unknown", "logical_block_size": 32}],
        [
            {"kind": "full_attention", "logical_block_size": 32},
            {"kind": "mla_attention", "logical_block_size": 64},
        ],
    ],
)
def test_requires_unambiguous_main_attention_size(groups):
    with pytest.raises(ValueError, match="one shared main-attention"):
        select_main_attention_block_size(groups, 16)


@pytest.mark.asyncio
async def test_metadata_fetch_failure_prevents_configuration():
    engine = SimpleNamespace(
        get_kv_cache_group_metadata=AsyncMock(side_effect=RuntimeError("unavailable"))
    )
    config = SimpleNamespace(additional_config=None)
    with pytest.raises(RuntimeError, match="unavailable"):
        await configure_kv_event_block_size(engine, config)
    assert config.additional_config is None
