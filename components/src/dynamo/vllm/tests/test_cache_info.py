# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from dynamo.vllm.cache_info import (
    DYNAMO_KV_CACHE_GROUPS_KEY,
    DYNAMO_KV_EVENT_BLOCK_SIZE_KEY,
    configure_kv_event_block_size,
    recurrent_alignment_blocks,
    recurrent_kv_groups,
    recurrent_kv_layout,
)

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def _config(unit=576):
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=unit,
            prefix_match_unit=None,
            enable_prefix_caching=True,
            mamba_cache_mode="align",
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        speculative_config=None,
        kv_transfer_config=None,
        additional_config={
            DYNAMO_KV_EVENT_BLOCK_SIZE_KEY: unit,
            DYNAMO_KV_CACHE_GROUPS_KEY: [
                {"group_idx": 0, "kind": "mamba", "block_size": unit},
                {"group_idx": 1, "kind": "mamba", "block_size": unit},
                {"group_idx": 2, "kind": "full_attention", "block_size": unit},
            ],
        },
    )


def test_group_layout_is_preserved_and_recurrent_publishing_is_explicit(monkeypatch):
    config = _config()
    metadata = config.additional_config[DYNAMO_KV_CACHE_GROUPS_KEY]
    config.cache_config.block_size = 16
    config.additional_config = None
    utility = AsyncMock(return_value=metadata)
    engine = SimpleNamespace(engine_core=SimpleNamespace(call_utility_async=utility))
    assert asyncio.run(configure_kv_event_block_size(engine, config)) == 576
    assert config.additional_config[DYNAMO_KV_CACHE_GROUPS_KEY] == metadata
    monkeypatch.delenv("DYN_VLLM_RECURRENT_KV_EVENTS", raising=False)
    assert recurrent_kv_groups(config) is None
    monkeypatch.setenv("DYN_VLLM_RECURRENT_KV_EVENTS", "1")
    assert recurrent_kv_groups(config) is None
    assert recurrent_kv_groups(config, enabled=True) == [0, 1]


def test_recurrent_opt_in_requires_successfully_fetched_layout():
    config = _config()
    config.additional_config = {}
    engine = SimpleNamespace(
        engine_core=SimpleNamespace(
            call_utility_async=AsyncMock(
                side_effect=RuntimeError("unsupported utility")
            )
        )
    )
    assert asyncio.run(configure_kv_event_block_size(engine, config)) == 576
    with pytest.raises(ValueError, match="group metadata"):
        recurrent_kv_groups(config, enabled=True)


@pytest.mark.parametrize("unit", [32, 64, 576])
@pytest.mark.parametrize("explicit_prefix_unit", [False, True])
def test_supported_geometry_publishes_fixed_common_unit_contract(
    unit, explicit_prefix_unit
):
    config = _config(unit)
    if explicit_prefix_unit:
        config.cache_config.prefix_match_unit = unit
    layout = recurrent_kv_layout(config, enabled=True)
    assert layout is not None
    assert layout.runtime_metadata() == {
        "contract": "vllm_fixed_common_unit_v1",
        "geometry_source": "validated_common_unit",
        "recurrent_groups": [0, 1],
        "alignment_blocks": 1,
        "indexer_block_size": unit,
        "hash_block_size": unit,
        "scheduler_block_size": unit,
        "cache_hit_alignment_tokens": unit,
        "cache_groups": config.additional_config[DYNAMO_KV_CACHE_GROUPS_KEY],
    }
    assert recurrent_alignment_blocks(config) == 1


def test_known_cpu_offloading_geometry_is_supported():
    config = _config()
    config.kv_transfer_config = SimpleNamespace(
        kv_connector="OffloadingConnector",
        kv_connector_extra_config={
            "self_describing_kv_events": True,
            "blocks_per_chunk": 1,
            "cpu_bytes_to_use": 1 << 30,
        },
    )
    assert recurrent_kv_groups(config, enabled=True) == [0, 1]


@pytest.mark.parametrize(
    "field,value",
    [
        ("prefix_match_unit", 16),
        ("prefix_match_unit", 1152),
        ("prefix_match_unit", "576"),
        ("enable_prefix_caching", False),
        ("mamba_cache_mode", "none"),
    ],
)
def test_unsupported_cache_geometry_fails_before_publication(field, value):
    config = _config()
    setattr(config.cache_config, field, value)
    with pytest.raises(ValueError, match="Recurrent KV events"):
        recurrent_kv_layout(config, enabled=True)


@pytest.mark.parametrize(
    "field", ["decode_context_parallel_size", "prefill_context_parallel_size"]
)
def test_context_parallel_geometry_requires_separate_support(field):
    config = _config()
    setattr(config.parallel_config, field, 2)
    with pytest.raises(ValueError, match="context parallelism"):
        recurrent_kv_layout(config, enabled=True)


def test_speculative_checkpoint_rules_require_separate_support():
    config = _config()
    config.speculative_config = SimpleNamespace(method="eagle")
    with pytest.raises(ValueError, match="EAGLE or speculative"):
        recurrent_kv_layout(config, enabled=True)


@pytest.mark.parametrize(
    "kind", ["sliding_window", "chunked_local_attention", "future_kind", None]
)
def test_unsupported_group_kinds_fail_closed(kind):
    config = _config()
    config.additional_config[DYNAMO_KV_CACHE_GROUPS_KEY].append(
        {"group_idx": 3, "kind": kind, "block_size": 576}
    )
    with pytest.raises(ValueError, match="sliding-window, chunked-local, and unknown"):
        recurrent_kv_layout(config, enabled=True)


@pytest.mark.parametrize("size", [0, -1, True, "576"])
def test_invalid_physical_group_size_is_rejected(size):
    config = _config()
    config.additional_config[DYNAMO_KV_CACHE_GROUPS_KEY][0]["block_size"] = size
    with pytest.raises(ValueError, match="positive group block sizes"):
        recurrent_kv_layout(config, enabled=True)


def test_mixed_group_sizes_are_not_reduced_to_an_unverified_lcm():
    config = _config(16)
    config.additional_config[DYNAMO_KV_CACHE_GROUPS_KEY][0]["block_size"] = 24
    with pytest.raises(ValueError, match="one common block size"):
        recurrent_alignment_blocks(config)


def test_different_indexer_unit_is_rejected():
    config = _config()
    config.additional_config[DYNAMO_KV_EVENT_BLOCK_SIZE_KEY] = 16
    with pytest.raises(ValueError, match="one common block size"):
        recurrent_kv_layout(config, enabled=True)


@pytest.mark.parametrize("group_id", [0, -1, True, "1", None])
def test_ambiguous_group_identity_is_rejected(group_id):
    config = _config()
    config.additional_config[DYNAMO_KV_CACHE_GROUPS_KEY][1]["group_idx"] = group_id
    with pytest.raises(ValueError, match="unique unsigned integers"):
        recurrent_kv_layout(config, enabled=True)


def test_unknown_engine_capability_is_not_assumed():
    config = _config()
    del config.cache_config.prefix_match_unit
    with pytest.raises(ValueError, match="capability fields"):
        recurrent_kv_layout(config, enabled=True)


@pytest.mark.parametrize(
    "extra",
    [
        {"self_describing_kv_events": True, "blocks_per_chunk": 2},
        {"self_describing_kv_events": False, "blocks_per_chunk": 1},
        {"self_describing_kv_events": True},
        {"self_describing_kv_events": True, "blocks_per_chunk": 1, "block_size": 1152},
        {
            "self_describing_kv_events": True,
            "blocks_per_chunk": 1,
            "spec_name": "TieringOffloadingSpec",
        },
    ],
)
def test_unknown_offload_geometry_is_not_advertised(extra):
    config = _config()
    config.kv_transfer_config = SimpleNamespace(
        kv_connector="OffloadingConnector", kv_connector_extra_config=extra
    )
    with pytest.raises(ValueError, match="OffloadingConnector"):
        recurrent_kv_layout(config, enabled=True)


def test_unknown_connector_is_not_advertised():
    config = _config()
    config.kv_transfer_config = SimpleNamespace(
        kv_connector="NixlConnector", kv_connector_extra_config={}
    )
    with pytest.raises(ValueError, match="OffloadingConnector"):
        recurrent_kv_layout(config, enabled=True)


def test_default_off_does_not_read_or_validate_new_capabilities():
    assert recurrent_kv_layout(SimpleNamespace()) is None
    assert recurrent_kv_groups(SimpleNamespace()) is None


def test_dense_worker_does_not_need_a_recurrent_capability_contract():
    config = SimpleNamespace(
        additional_config={
            DYNAMO_KV_CACHE_GROUPS_KEY: [
                {"group_idx": 0, "kind": "full_attention", "block_size": 16}
            ]
        }
    )
    assert recurrent_kv_layout(config, enabled=True) is None
    assert recurrent_kv_groups(config, enabled=True) is None
