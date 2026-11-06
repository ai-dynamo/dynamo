# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from dynamo.vllm.cache_info import (
    DYNAMO_KV_CACHE_GROUPS_KEY,
    configure_kv_event_block_size,
    recurrent_alignment_blocks,
    recurrent_kv_groups,
)


def test_group_layout_is_preserved_and_recurrent_publishing_is_explicit(monkeypatch):
    metadata = [
        {"group_idx": 0, "kind": "mamba", "block_size": 576},
        {"group_idx": 1, "kind": "mamba", "block_size": 576},
        {"group_idx": 2, "kind": "full_attention", "block_size": 576},
    ]
    config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=16), additional_config=None
    )
    utility = AsyncMock(return_value=metadata)
    engine = SimpleNamespace(engine_core=SimpleNamespace(call_utility_async=utility))
    assert asyncio.run(configure_kv_event_block_size(engine, config)) == 576
    assert config.additional_config[DYNAMO_KV_CACHE_GROUPS_KEY] == metadata
    monkeypatch.delenv("DYN_VLLM_RECURRENT_KV_EVENTS", raising=False)
    assert recurrent_kv_groups(config) is None
    monkeypatch.setenv("DYN_VLLM_RECURRENT_KV_EVENTS", "1")
    assert recurrent_kv_groups(config, enabled=True) == [0, 1]


def test_recurrent_opt_in_requires_successfully_fetched_layout(monkeypatch):
    config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=16), additional_config={}
    )
    engine = SimpleNamespace(
        engine_core=SimpleNamespace(
            call_utility_async=AsyncMock(
                side_effect=RuntimeError("unsupported utility")
            )
        )
    )
    assert asyncio.run(configure_kv_event_block_size(engine, config)) == 16
    monkeypatch.setenv("DYN_VLLM_RECURRENT_KV_EVENTS", "1")
    with pytest.raises(ValueError, match="group metadata"):
        recurrent_kv_groups(config, enabled=True)


def test_recurrent_alignment_uses_all_group_block_sizes():
    config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=16),
        additional_config={
            DYNAMO_KV_CACHE_GROUPS_KEY: [{"block_size": 16}, {"block_size": 24}]
        },
    )
    assert recurrent_alignment_blocks(config) == 3
