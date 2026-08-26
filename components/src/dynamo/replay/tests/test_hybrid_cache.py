# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from dynamo.replay.hybrid_cache import (
    HybridCacheConfig,
    HybridCacheRequest,
    VllmHybridCacheSimulator,
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
]


def _deepseek_config(**overrides) -> HybridCacheConfig:
    raw = {
        "scheduler_block_size": 256,
        "hash_block_size": 4,
        "gpu_capacity_slots": 1000,
        "cpu_capacity_slots": 1000,
        "cpu_slot_bytes": 8_317_440,
        "store_threshold": 1,
        "groups": [
            {"group_index": 0, "block_size": 256},
            {
                "group_index": 1,
                "block_size": 64,
                "sliding_window": 128,
                "gpu_spec_group": "b64-w128",
            },
            {
                "group_index": 2,
                "block_size": 64,
                "sliding_window": 128,
                "gpu_spec_group": "b64-w128",
            },
            {"group_index": 3, "block_size": 4, "sliding_window": 8},
            {"group_index": 4, "block_size": 8, "sliding_window": 128},
            {
                "group_index": 5,
                "block_size": 64,
                "sliding_window": 128,
                "gpu_spec_group": "b64-w128",
                "use_eagle": True,
                "offload": False,
            },
        ],
    }
    raw.update(overrides)
    return HybridCacheConfig.from_dict(raw)


def _request(request_id: str = "request-1", tokens: int = 256):
    return HybridCacheRequest(
        request_id=request_id,
        input_length=tokens,
        output_length=0,
        lineage=tuple(f"h-{index}" for index in range(tokens // 4)),
    )


def test_deepseek_store_geometry_uses_group_specific_fixed_slots() -> None:
    simulator = VllmHybridCacheSimulator(_deepseek_config())

    result = simulator.process(_request())

    assert result.cpu_store_offers_by_group == {0: 1, 1: 2, 2: 2, 3: 2, 4: 16}
    assert result.cpu_store_offers == 23
    assert result.cpu_occupancy_slots == 23
    assert result.cpu_reserved_bytes == 23 * 8_317_440


def test_store_threshold_counts_post_mask_offers() -> None:
    simulator = VllmHybridCacheSimulator(_deepseek_config(store_threshold=2))
    request = _request()

    first = simulator.process(request)
    second = simulator.process(request)

    assert first.cpu_store_offers == 23
    assert first.cpu_admissions == 0
    assert second.cpu_store_offers == 23
    assert second.cpu_admissions == 23


def test_physical_gpu_availability_filters_store_offers_below_hit() -> None:
    simulator = VllmHybridCacheSimulator(_deepseek_config())
    request = _request(tokens=512)
    simulator.process(request)
    group = simulator._groups[4]
    missing = simulator._prompt_key(request, group, 31)
    simulator._gpu_by_group[4].remove(missing)

    offered = simulator._store_offer_keys(
        request,
        combined_hit_tokens=256,
        external_hit_tokens=0,
        gpu_hit_tokens=256,
    )

    group_four = [key for key in offered if key[0] == 4]
    assert missing not in group_four
    assert len(group_four) == 31


def test_draft_group_is_gpu_resident_but_excluded_from_cpu() -> None:
    simulator = VllmHybridCacheSimulator(_deepseek_config())

    result = simulator.process(_request())

    assert simulator.effective_eagle_groups == frozenset({1, 2, 5})
    assert len(simulator._gpu_by_group[5]) == 1
    assert 5 not in result.cpu_store_offers_by_group


def test_cpu_capacity_bytes_are_converted_to_fixed_slots() -> None:
    config = _deepseek_config(
        cpu_capacity_slots=None,
        cpu_capacity_bytes=3 * 8_317_440 + 1,
    )

    assert config.cpu_capacity_slots == 3


def test_store_batch_larger_than_cpu_pool_is_rejected() -> None:
    simulator = VllmHybridCacheSimulator(_deepseek_config(cpu_capacity_slots=22))

    result = simulator.process(_request())

    assert result.cpu_store_offers == 23
    assert result.cpu_admissions == 0
    assert result.cpu_occupancy_slots == 0
