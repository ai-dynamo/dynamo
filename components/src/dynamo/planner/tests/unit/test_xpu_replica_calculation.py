# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replica calculation coverage using Intel B60 profiling results."""

import math
from pathlib import Path

import pytest

from dynamo.planner.config.defaults import SubComponentType
from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.budget import _apply_global_gpu_budget
from dynamo.planner.core.perf_model import DecodeRegressionModel, PrefillRegressionModel
from dynamo.planner.monitoring.perf_metrics import _convert_profiling_data_to_fpms

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]

PROFILE_ROOT = Path(__file__).parent.parent / "data/profiling_results"
B60_PROFILE_DIR = PROFILE_ROOT / "B60_TP1P_TP1D"
H200_PROFILE_DIR = PROFILE_ROOT / "H200_TP1P_TP1D"


def _models(profile_dir: Path):
    prefill = PrefillRegressionModel(max_num_fpm_samples=50, min_observations=5)
    decode = DecodeRegressionModel(max_num_fpm_samples=50, min_observations=5)
    prefill.load_benchmark_fpms(
        _convert_profiling_data_to_fpms(str(profile_dir), SubComponentType.PREFILL)
    )
    decode.load_benchmark_fpms(
        _convert_profiling_data_to_fpms(str(profile_dir), SubComponentType.DECODE)
    )
    return prefill, decode


def _replicas(profile_dir: Path, demand_rps: float) -> tuple[int, int]:
    prefill, decode = _models(profile_dir)
    prefill_rps, _ = prefill.find_best_engine_prefill_rps(
        ttft_sla=1_000.0, isl=3_000.0, max_num_batched_tokens=8_192
    )
    decode_rps, _ = decode.find_best_engine_decode_rps(
        itl=50.0,
        context_length=3_075.0,
        osl=150.0,
        max_kv_tokens=55_000,
        max_num_seqs=256,
    )
    assert prefill_rps > 0
    assert decode_rps > 0
    return math.ceil(demand_rps / prefill_rps), math.ceil(demand_rps / decode_rps)


def test_b60_replica_counts_scale_with_demand():
    low = _replicas(B60_PROFILE_DIR, demand_rps=10.0)
    high = _replicas(B60_PROFILE_DIR, demand_rps=100.0)

    assert high[0] > low[0]
    assert high[1] > low[1]


def test_b60_requires_more_replicas_than_h200():
    b60_prefill, b60_decode = _replicas(B60_PROFILE_DIR, demand_rps=100.0)
    h200_prefill, h200_decode = _replicas(H200_PROFILE_DIR, demand_rps=100.0)

    assert b60_prefill > h200_prefill
    assert b60_decode > h200_decode


def test_b60_replica_counts_respect_global_gpu_budget():
    desired_prefill, desired_decode = _replicas(B60_PROFILE_DIR, demand_rps=100.0)
    config = PlannerConfig.model_construct(
        prefill_engine_num_gpu=1,
        decode_engine_num_gpu=1,
        min_endpoint=1,
        min_gpu_budget=-1,
        max_gpu_budget=10,
    )

    actual_prefill, actual_decode = _apply_global_gpu_budget(
        desired_prefill, desired_decode, config
    )

    assert actual_prefill >= config.min_endpoint
    assert actual_decode >= config.min_endpoint
    assert actual_prefill + actual_decode <= config.max_gpu_budget
