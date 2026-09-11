# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from dynamo.vllm.cache_info import (
    DYNAMO_KV_EVENT_BLOCK_SIZE_KEY,
    configure_kv_event_block_size,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "effective_size,expected", [(16, 16), (32, 32), (1056, 1056), (None, 16)]
)
async def test_configure_uses_effective_attention_block_size(effective_size, expected):
    config = SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=16, effective_attention_block_size=effective_size
        ),
        additional_config=None,
    )
    engine = SimpleNamespace(vllm_config=config)

    assert await configure_kv_event_block_size(engine, config) == expected
    assert config.additional_config[DYNAMO_KV_EVENT_BLOCK_SIZE_KEY] == expected
    assert config.cache_config.block_size == 16
