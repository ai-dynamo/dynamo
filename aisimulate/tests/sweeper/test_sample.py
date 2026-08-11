# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend-only sample materialization."""

import pytest

from aisimulate.sweeper.config import SearchSpace
from aisimulate.sweeper.parallel_enum import (
    DisaggParallelConfig,
    ParallelShape,
    ReplicaParallelConfig,
)
from aisimulate.sweeper.sample import unroll_sample


def _space(**overrides) -> SearchSpace:
    values = {"model_name": "example/model", "hardware_sku": "example_sku"}
    values.update(overrides)
    return SearchSpace(**values)


def _agg_selection(**overrides) -> dict:
    values = {
        "deployment_mode": "agg",
        "backend": "trtllm",
        "agg_max_num_batched_tokens": 8192,
        "agg_max_num_seqs": 256,
    }
    values.update(overrides)
    return values


AGG_CONFIG = ReplicaParallelConfig(
    shape=ParallelShape(tp=4, dp=1, moe_tp=1, moe_ep=4), replicas=2
)
DISAGG_CONFIG = DisaggParallelConfig(
    prefill=ReplicaParallelConfig(ParallelShape(tp=8, dp=1, moe_tp=1, moe_ep=8), 1),
    decode=ReplicaParallelConfig(ParallelShape(tp=1, dp=8, moe_tp=1, moe_ep=8), 2),
)


def test_agg_unroll_preserves_backend_shape_and_batching():
    sample = unroll_sample(
        search_space=_space(),
        selection=_agg_selection(),
        parallel_config=AGG_CONFIG,
    )

    assert {
        key: sample[key]
        for key in (
            "deployment_mode",
            "backend",
            "tp",
            "attention_dp",
            "moe_tp",
            "moe_ep",
            "pp",
            "replicas",
            "strategy",
            "used_gpus",
            "agg_max_num_batched_tokens",
            "agg_max_num_seqs",
        )
    } == {
        "deployment_mode": "agg",
        "backend": "trtllm",
        "tp": 4,
        "attention_dp": 1,
        "moe_tp": 1,
        "moe_ep": 4,
        "pp": 1,
        "replicas": 2,
        "strategy": "tep",
        "used_gpus": 8,
        "agg_max_num_batched_tokens": 8192,
        "agg_max_num_seqs": 256,
    }
    assert "prefill_tp" not in sample
    assert "decode_tp" not in sample


def test_disagg_unroll_preserves_both_roles():
    selection = _agg_selection(
        deployment_mode="disagg",
        prefill_max_num_batched_tokens=16384,
        prefill_max_num_seqs=4,
        decode_max_num_batched_tokens=8192,
        decode_max_num_seqs=512,
    )

    sample = unroll_sample(
        search_space=_space(),
        selection=selection,
        parallel_config=DISAGG_CONFIG,
    )

    assert (
        sample["prefill_tp"],
        sample["prefill_moe_ep"],
        sample["prefill_replicas"],
    ) == (8, 8, 1)
    assert (
        sample["decode_attention_dp"],
        sample["decode_moe_ep"],
        sample["decode_replicas"],
    ) == (8, 8, 2)
    assert sample["prefill_strategy"] == "tep"
    assert sample["decode_strategy"] == "dep"
    assert sample["used_gpus"] == 24
    assert "tp" not in sample
    assert "agg_max_num_seqs" not in sample


def test_unroll_folds_only_backend_pinned_values():
    sample = unroll_sample(
        search_space=_space(
            gpu_budget=16,
            min_gpu_budget=8,
            context_length=4096,
            startup_time=300.0,
            aic_nextn=2,
            agg_block_size=32,
            agg_gpu_memory_utilization=0.8,
            agg_enable_prefix_caching=False,
        ),
        selection=_agg_selection(),
        parallel_config=AGG_CONFIG,
    )

    assert sample["gpu_budget"] == 16
    assert sample["min_gpu_budget"] == 8
    assert sample["context_length"] == 4096
    assert sample["startup_time"] == 300.0
    assert sample["aic_nextn"] == 2
    assert sample["agg_block_size"] == 32
    assert sample["agg_gpu_memory_utilization"] == 0.8
    assert sample["agg_enable_prefix_caching"] is False
    assert not any(key.startswith(("planner_", "router_")) for key in sample)


@pytest.mark.parametrize(
    ("mode", "parallel_config", "message"),
    [
        ("agg", DISAGG_CONFIG, "ReplicaParallelConfig"),
        ("disagg", AGG_CONFIG, "DisaggParallelConfig"),
    ],
)
def test_unroll_rejects_parallel_config_for_wrong_topology(
    mode, parallel_config, message
):
    selection = _agg_selection(deployment_mode=mode)
    if mode == "disagg":
        selection.update(
            prefill_max_num_batched_tokens=8192,
            prefill_max_num_seqs=4,
            decode_max_num_batched_tokens=8192,
            decode_max_num_seqs=256,
        )

    with pytest.raises(TypeError, match=message):
        unroll_sample(
            search_space=_space(),
            selection=selection,
            parallel_config=parallel_config,
        )
