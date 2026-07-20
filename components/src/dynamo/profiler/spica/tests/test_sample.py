# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unroll a selected sample: composite knobs expanded, flattened per mode/router/planner.

Pure — selection + parallel_config are inputs, so no perf DB / aiconfigurator
model build is needed."""

from dynamo.profiler.spica.config import SearchSpace
from dynamo.profiler.spica.load_predictor_sweep import LoadPredictorResult
from dynamo.profiler.spica.parallel_enum import (
    DisaggParallelConfig,
    ParallelShape,
    ReplicaParallelConfig,
)
from dynamo.profiler.spica.sample import unroll_sample


def _space(**overrides) -> SearchSpace:
    base = {"model_name": "deepseek-ai/DeepSeek-V3", "hardware_sku": "gb200"}
    base.update(overrides)
    return SearchSpace(**base)


def _agg_selection(**overrides) -> dict:
    sel = {
        "deployment_mode": "agg",
        "backend": "trtllm",
        "agg_max_num_batched_tokens": 8192,
        "agg_max_num_seqs": 256,
        "router_mode": "round_robin",
        "planner_scaling_policy": "disabled",
        "planner_fpm_sampling": "default",
        "planner_load_sensitivity": "default",
    }
    sel.update(overrides)
    return sel


AGG_CFG = ReplicaParallelConfig(
    shape=ParallelShape(tp=4, dp=1, moe_tp=1, moe_ep=4), replicas=2
)
DISAGG_CFG = DisaggParallelConfig(
    prefill=ReplicaParallelConfig(ParallelShape(tp=8, dp=1, moe_tp=1, moe_ep=8), 1),
    decode=ReplicaParallelConfig(ParallelShape(tp=1, dp=8, moe_tp=1, moe_ep=8), 2),
)


def test_agg_parallel_unrolls_to_flat_shape():
    s = unroll_sample(
        search_space=_space(), selection=_agg_selection(), parallel_config=AGG_CFG
    )
    assert (s["tp"], s["attention_dp"], s["moe_tp"], s["moe_ep"], s["pp"]) == (
        4,
        1,
        1,
        4,
        1,
    )
    assert s["replicas"] == 2
    assert s["strategy"] == "tep"
    assert s["used_gpus"] == 8
    assert s["agg_max_num_batched_tokens"] == 8192 and s["agg_max_num_seqs"] == 256
    # the other branch's engine knobs are absent
    assert "prefill_tp" not in s and "decode_tp" not in s
    assert "prefill_max_num_seqs" not in s and "decode_max_num_seqs" not in s


def test_disagg_parallel_unrolls_prefill_and_decode():
    sel = _agg_selection(
        deployment_mode="disagg",
        prefill_max_num_batched_tokens=16384,
        prefill_max_num_seqs=4,
        decode_max_num_batched_tokens=8192,
        decode_max_num_seqs=512,
    )
    s = unroll_sample(search_space=_space(), selection=sel, parallel_config=DISAGG_CFG)
    assert (s["prefill_tp"], s["prefill_moe_ep"], s["prefill_replicas"]) == (8, 8, 1)
    assert (s["decode_attention_dp"], s["decode_moe_ep"], s["decode_replicas"]) == (
        8,
        8,
        2,
    )
    assert s["prefill_strategy"] == "tep" and s["decode_strategy"] == "dep"
    assert s["used_gpus"] == 8 + 16  # 8*1 + 8*2
    assert s["prefill_max_num_seqs"] == 4 and s["decode_max_num_seqs"] == 512
    assert "tp" not in s and "agg_max_num_seqs" not in s


def test_scaling_policy_disabled_emits_only_flags():
    s = unroll_sample(
        search_space=_space(), selection=_agg_selection(), parallel_config=AGG_CFG
    )
    assert s["enable_throughput_scaling"] is False and s["enable_load_scaling"] is False
    assert "throughput_adjustment_interval_seconds" not in s
    assert "max_num_fpm_samples" not in s and "load_scaling_down_sensitivity" not in s
    assert "load_predictor" not in s


def test_scaling_policy_decodes_intervals():
    sel = _agg_selection(planner_scaling_policy="throughput_180_5")
    s = unroll_sample(search_space=_space(), selection=sel, parallel_config=AGG_CFG)
    assert s["enable_throughput_scaling"] is True and s["enable_load_scaling"] is False
    assert s["throughput_adjustment_interval_seconds"] == 180
    assert s["load_adjustment_interval_seconds"] == 5


def test_fpm_and_sensitivity_numeric_expansion():
    sel = _agg_selection(
        planner_scaling_policy="hybrid_180_5",
        planner_fpm_sampling="large",
        planner_load_sensitivity="aggressive",
    )
    s = unroll_sample(search_space=_space(), selection=sel, parallel_config=AGG_CFG)
    assert s["max_num_fpm_samples"] == 128 and s["fpm_sample_bucket_size"] == 16
    assert s["load_scaling_down_sensitivity"] == 70 and s["load_min_observations"] == 3


def test_load_predictor_resolved_from_sweep_winner():
    lp = LoadPredictorResult(
        best_by_interval={180: "prophet_w20_log1p", 600: "constant_last"},
        reason="swept",
    )
    sel = _agg_selection(planner_scaling_policy="throughput_180_5")
    s = unroll_sample(
        search_space=_space(), selection=sel, parallel_config=AGG_CFG, load_predictor=lp
    )
    assert s["load_predictor"] == "prophet" and s["load_predictor_log1p"] is True
    assert s["prophet_window_size"] == 20
    assert "kalman_q_level" not in s  # only the chosen family's knobs


def test_load_predictor_omitted_without_throughput_scaling():
    lp = LoadPredictorResult(
        best_by_interval={180: "prophet_w20_log1p"}, reason="swept"
    )
    sel = _agg_selection(
        planner_scaling_policy="load_180_5"
    )  # load-only -> no forecaster
    s = unroll_sample(
        search_space=_space(), selection=sel, parallel_config=AGG_CFG, load_predictor=lp
    )
    assert s["enable_load_scaling"] is True and s["enable_throughput_scaling"] is False
    assert "load_predictor" not in s


def test_router_knobs_gated_by_router_mode():
    s = unroll_sample(
        search_space=_space(),
        selection=_agg_selection(router_mode="round_robin"),
        parallel_config=AGG_CFG,
    )
    assert s["router_mode"] == "round_robin"
    assert "overlap_score_credit" not in s and "router_temperature" not in s
    assert "no_admission_control" not in s

    sel = _agg_selection(
        router_mode="kv_router",
        overlap_score_credit=0.5,
        prefill_load_scale=1.0,
        host_cache_hit_weight=0.75,
        disk_cache_hit_weight=0.25,
        router_temperature=0.2,
    )
    s2 = unroll_sample(search_space=_space(), selection=sel, parallel_config=AGG_CFG)
    assert s2["overlap_score_credit"] == 0.5 and s2["router_temperature"] == 0.2
    assert "no_admission_control" in s2  # admission scalars folded in for kv_router


def test_planner_composites_accept_raw_dicts():
    # scaling + fpm + sensitivity all pinned as raw dicts (custom interval=240)
    sel = _agg_selection(
        planner_scaling_policy={
            "enable_throughput_scaling": True,
            "enable_load_scaling": False,
            "throughput_adjustment_interval_seconds": 240,
            "load_adjustment_interval_seconds": 5,
        },
        planner_fpm_sampling={"max_num_fpm_samples": 96, "fpm_sample_bucket_size": 16},
        planner_load_sensitivity={
            "load_scaling_down_sensitivity": 75,
            "load_min_observations": 4,
        },
    )
    s = unroll_sample(search_space=_space(), selection=sel, parallel_config=AGG_CFG)
    assert s["enable_throughput_scaling"] is True and s["enable_load_scaling"] is False
    assert s["throughput_adjustment_interval_seconds"] == 240  # not a preset value
    assert s["max_num_fpm_samples"] == 96 and s["fpm_sample_bucket_size"] == 16
    assert s["load_scaling_down_sensitivity"] == 75 and s["load_min_observations"] == 4


def test_scaling_dict_both_flags_off_is_disabled():
    sel = _agg_selection(
        planner_scaling_policy={
            "enable_throughput_scaling": False,
            "enable_load_scaling": False,
        }
    )
    s = unroll_sample(search_space=_space(), selection=sel, parallel_config=AGG_CFG)
    assert s["enable_throughput_scaling"] is False and s["enable_load_scaling"] is False
    assert (
        "throughput_adjustment_interval_seconds" not in s
    )  # disabled -> nothing else emitted
    assert "max_num_fpm_samples" not in s


def test_load_predictor_winner_can_be_a_custom_dict():
    # the sweep winner for the interval is a raw predictor dict, not a preset id
    lp = LoadPredictorResult(
        best_by_interval={
            240: {
                "load_predictor": "kalman",
                "load_predictor_log1p": True,
                "kalman_q_level": 3.0,
            }
        },
        reason="swept",
    )
    sel = _agg_selection(
        planner_scaling_policy={
            "enable_throughput_scaling": True,
            "enable_load_scaling": False,
            "throughput_adjustment_interval_seconds": 240,
            "load_adjustment_interval_seconds": 5,
        }
    )
    s = unroll_sample(
        search_space=_space(), selection=sel, parallel_config=AGG_CFG, load_predictor=lp
    )
    assert s["load_predictor"] == "kalman" and s["load_predictor_log1p"] is True
    assert (
        s["kalman_q_level"] == 3.0 and s["kalman_min_points"] == 5
    )  # custom value + default


def test_pinned_scalars_folded_in():
    s = unroll_sample(
        search_space=_space(
            gpu_budget=16,
            min_gpu_budget=8,
            min_endpoint=2,
            context_length=4096,
            startup_time=300.0,
            aic_nextn=2,
            kv_bytes_per_token=131072,
        ),
        selection=_agg_selection(),
        parallel_config=AGG_CFG,
    )
    assert s["model_name"] == "deepseek-ai/DeepSeek-V3" and s["hardware_sku"] == "gb200"
    assert s["aic_nextn"] == 2
    assert s["num_g2_blocks"] == 0  # kv-manager pinned
    assert s["kv_bytes_per_token"] == 131072
    assert s["agg_block_size"] == 64 and s["agg_gpu_memory_utilization"] == 0.9
    assert s["gpu_budget"] == 16 and s["min_gpu_budget"] == 8 and s["min_endpoint"] == 2
    assert s["context_length"] == 4096 and s["startup_time"] == 300.0
