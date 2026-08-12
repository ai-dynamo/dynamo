# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo-independent Sweeper configuration contracts."""

import pytest
from pydantic import ValidationError

from aisimulate.sweeper import OptimizationTarget, SmartSearchConfig
from aisimulate.sweeper.config import (
    OptimizationGoal,
    SearchSpace,
    SLATarget,
    SweepConfig,
    Workload,
)


def _search_space(**overrides):
    return {"model_name": "m", "hardware_sku": "h200_sxm", **overrides}


def _workload(**overrides):
    return {
        "isl": 4000,
        "osl": 1000,
        "concurrency": 2,
        "num_request_ratio": 10,
        **overrides,
    }


def test_backend_only_yaml_and_adapter_search_space_load(tmp_path):
    path = tmp_path / "sweep.yaml"
    path.write_text(
        """
search_space:
  deployment_mode: [agg]
  backend: [vllm]
  model_name: example/model
  hardware_sku: h200_sxm
  gpu_budget: 8
adapters:
  dynamo.planner:
    search_space:
      scaling_interval: [5, 10]
      load_predictor: [default, conservative]
workload:
  isl: 128
  osl: 16
  request_rate: 2
  num_request_ratio: 3
sweep:
  max_rounds: 2
  candidates_per_round: 1
"""
    )

    config = SmartSearchConfig.from_yaml(path)

    assert config.search_space.deployment_mode == ["agg"]
    assert config.search_space.backend == ["vllm"]
    assert config.adapters["dynamo.planner"].search_space == {
        "scaling_interval": [5, 10],
        "load_predictor": ["default", "conservative"],
    }
    assert config.workload.request_rate == 2
    assert config.sweep.max_rounds == 2


def test_defaults_are_backend_only():
    config = SmartSearchConfig(
        search_space=_search_space(),
        workload=_workload(),
    )

    assert config.search_space.gpu_budget == 32
    assert config.search_space.prefill_block_size == 64
    assert config.search_space.prefill_max_num_seqs == [
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
    ]
    assert config.adapters == {}
    assert config.goal.target is OptimizationTarget.THROUGHPUT
    assert config.sweep.parallel_evals == 16
    dumped = config.search_space.model_dump()
    assert (
        not {"planner_scaling_policy", "router_mode", "num_g2_blocks"} & dumped.keys()
    )


def test_extra_fields_are_forbidden_at_each_boundary():
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        SmartSearchConfig(
            search_space=_search_space(bogus=1),
            workload=_workload(),
        )
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        SmartSearchConfig(
            search_space=_search_space(),
            workload=_workload(),
            adapters={"custom": {"search_space": {}, "bogus": 1}},
        )


@pytest.mark.parametrize(
    "field",
    [
        "min_endpoint",
        "prefill_min_endpoint",
        "decode_min_endpoint",
        "planner_scaling_policy",
        "planner_fpm_sampling",
        "planner_load_sensitivity",
        "load_predictor_candidates",
    ],
)
def test_legacy_planner_fields_point_to_adapter_migration(field):
    with pytest.raises(
        ValidationError,
        match=r"adapters\['dynamo\.planner'\]\.search_space",
    ):
        SmartSearchConfig(
            search_space=_search_space(**{field: 1}),
            workload=_workload(),
        )


@pytest.mark.parametrize(
    "field",
    [
        "router_mode",
        "overlap_score_credit",
        "prefill_load_scale",
        "router_temperature",
        "active_decode_blocks_threshold",
        "active_decode_tokens_threshold",
        "active_prefill_tokens_threshold",
        "active_prefill_tokens_threshold_frac",
        "no_admission_control",
    ],
)
def test_legacy_router_fields_point_to_adapter_migration(field):
    with pytest.raises(
        ValidationError,
        match=r"adapters\['dynamo\.router'\]\.search_space",
    ):
        SmartSearchConfig(
            search_space=_search_space(**{field: 1}),
            workload=_workload(),
        )


@pytest.mark.parametrize(
    "field",
    [
        "num_g2_blocks",
        "kv_bytes_per_token",
        "bandwidth_g1_to_g2_gbps",
        "bandwidth_g2_to_g1_gbps",
        "offload_batch_size",
        "host_cache_hit_weight",
        "disk_cache_hit_weight",
    ],
)
def test_legacy_kvbm_fields_are_deprecated_without_migration(field):
    with pytest.raises(
        ValidationError,
        match="not supported by the AISimulate engine and replay path",
    ):
        SmartSearchConfig(
            search_space=_search_space(**{field: 1}),
            workload=_workload(),
        )


def test_backend_choice_subset_is_accepted():
    config = SmartSearchConfig(
        search_space=_search_space(deployment_mode=["agg"], backend=["vllm", "sglang"]),
        workload=_workload(),
    )

    assert config.search_space.deployment_mode == ["agg"]
    assert config.search_space.backend == ["vllm", "sglang"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("backend", ["bogus"]),
        ("deployment_mode", ["bogus"]),
        ("prefill_max_num_seqs", [1, 999]),
        ("agg_max_num_batched_tokens", [8192, 999]),
        ("backend", []),
    ],
)
def test_invalid_or_empty_backend_choices_are_rejected(field, value):
    with pytest.raises(ValidationError):
        SearchSpace(**_search_space(**{field: value}))


def test_pinned_parallel_config_requires_one_deployment_mode():
    with pytest.raises(ValidationError, match="exactly one mode"):
        SearchSpace(
            **_search_space(
                deployment_mode=["agg", "disagg"],
                parallel_configs=[{"tp": 4}],
            )
        )


@pytest.mark.parametrize(
    ("mode", "entry", "message"),
    [
        ("agg", {"replicas": 2}, "'tp' field"),
        ("disagg", {"tp": 4}, "prefill"),
        (
            "disagg",
            {"prefill": 1, "decode": {"tp": 1}},
            "prefill.*dict",
        ),
        (
            "disagg",
            {"prefill": {"tp": 1}, "decode": {"replicas": 1}},
            "decode.*'tp' field",
        ),
    ],
)
def test_parallel_config_shape_matches_mode(mode, entry, message):
    with pytest.raises(ValidationError, match=message):
        SearchSpace(
            **_search_space(
                deployment_mode=[mode],
                parallel_configs=[entry],
            )
        )


def test_well_formed_agg_and_disagg_parallel_configs_are_accepted():
    SearchSpace(
        **_search_space(
            deployment_mode=["agg"],
            parallel_configs=[{"tp": 4, "moe_ep": 4, "replicas": 2}],
        )
    )
    SearchSpace(
        **_search_space(
            deployment_mode=["disagg"],
            parallel_configs=[
                {
                    "prefill": {"tp": 8, "moe_ep": 8},
                    "decode": {"tp": 1, "attention_dp": 8, "moe_ep": 8},
                }
            ],
        )
    )


def test_trace_and_synthetic_workloads_are_mutually_exclusive():
    with pytest.raises(ValidationError, match="must not set synthetic fields"):
        Workload(
            trace_path="/tmp/trace.jsonl",
            isl=1,
            osl=1,
            concurrency=1,
            num_request_ratio=1,
        )
    with pytest.raises(ValidationError, match="exactly one"):
        Workload(isl=1, osl=1, concurrency=1, request_rate=1, num_request_ratio=1)


def test_trace_closed_loop_cap_and_synthetic_helpers():
    trace = Workload(trace_path="/tmp/trace.jsonl", replay_concurrency=8)
    concurrency = Workload(isl=4000, osl=1000, concurrency=256, num_request_ratio=10)
    rate = Workload(isl=4000, osl=1000, request_rate=25, num_request_ratio=4)

    assert trace.is_trace_based
    assert trace.effective_in_flight_cap() == 8
    assert concurrency.effective_in_flight_cap() == 256
    assert concurrency.effective_in_flight_cap(8) == 8
    assert concurrency.resolved_request_count() == 2560
    assert concurrency.resolved_request_count(8) == 80
    assert concurrency.synthetic_arrival_interval_ms is None
    assert rate.synthetic_arrival_interval_ms == 40
    assert rate.resolved_request_count() == 100


@pytest.mark.parametrize(
    "workload",
    [
        {"isl": 1, "osl": 1, "concurrency": 0, "num_request_ratio": 1},
        {"isl": 1, "osl": 1, "concurrency": 1.9, "num_request_ratio": 1},
        {"isl": 1, "osl": 1, "concurrency": 1},
        {"trace_path": "/tmp/trace.jsonl", "replay_concurrency": 0},
    ],
)
def test_invalid_workloads_are_rejected(workload):
    with pytest.raises(ValidationError):
        Workload(**workload)


def test_goodput_requires_complete_sla():
    with pytest.raises(ValidationError, match="require an SLA"):
        OptimizationGoal(target=OptimizationTarget.GOODPUT)
    with pytest.raises(ValidationError, match="require an SLA"):
        OptimizationGoal(
            target=OptimizationTarget.GOODPUT,
            sla=SLATarget(ttft_ms=2000),
        )

    OptimizationGoal(
        target=OptimizationTarget.GOODPUT,
        sla=SLATarget(ttft_ms=2000, itl_ms=30),
    )
    OptimizationGoal(
        target=OptimizationTarget.GOODPUT_PER_GPU,
        sla=SLATarget(e2e_ms=5000),
    )


def test_scalar_target_directions():
    assert OptimizationTarget.THROUGHPUT.maximize
    assert OptimizationTarget.THROUGHPUT_PER_GPU.maximize
    assert OptimizationTarget.THROUGHPUT_PER_USER.maximize
    assert OptimizationTarget.GOODPUT_PER_GPU.maximize
    assert not OptimizationTarget.E2E_LATENCY.maximize
    with pytest.raises(ValueError, match="no scalar direction"):
        _ = OptimizationTarget.PARETO.maximize


def test_pareto_defaults_and_custom_objectives():
    default = OptimizationGoal(target=OptimizationTarget.PARETO)
    custom = OptimizationGoal(
        target=OptimizationTarget.PARETO,
        pareto_objectives=[
            OptimizationTarget.THROUGHPUT_PER_USER,
            OptimizationTarget.GOODPUT_PER_GPU,
        ],
        sla=SLATarget(ttft_ms=2000, itl_ms=30),
    )

    assert default.resolved_pareto_objectives == [
        OptimizationTarget.THROUGHPUT_PER_GPU,
        OptimizationTarget.THROUGHPUT_PER_USER,
    ]
    assert custom.resolved_pareto_objectives[1] is OptimizationTarget.GOODPUT_PER_GPU


@pytest.mark.parametrize(
    "goal",
    [
        {
            "target": "throughput",
            "pareto_objectives": ["throughput_per_gpu", "throughput_per_user"],
        },
        {"target": "pareto", "pareto_objectives": []},
        {"target": "pareto", "pareto_objectives": ["throughput_per_gpu"]},
        {
            "target": "pareto",
            "pareto_objectives": ["throughput_per_gpu", "throughput_per_gpu"],
        },
        {
            "target": "pareto",
            "pareto_objectives": ["goodput_per_gpu", "throughput_per_user"],
        },
    ],
)
def test_invalid_pareto_goals_are_rejected(goal):
    with pytest.raises(ValidationError):
        OptimizationGoal.model_validate(goal)


def test_kv_load_range_is_pareto_only_and_defaults_for_synthetic_pareto():
    with pytest.raises(ValidationError, match="ranged workload.kv_load_ratio"):
        SmartSearchConfig(
            search_space=_search_space(),
            workload={
                "isl": 1024,
                "osl": 1024,
                "kv_load_ratio": [0.0, 1.0],
                "num_request_ratio": 10,
            },
            goal={"target": "throughput_per_gpu"},
        )

    explicit = SmartSearchConfig(
        search_space=_search_space(),
        workload={
            "isl": 1024,
            "osl": 1024,
            "kv_load_ratio": [0.25, 0.75],
            "num_request_ratio": 10,
        },
        goal={"target": "pareto"},
    )
    defaulted = SmartSearchConfig(
        search_space=_search_space(),
        workload={"isl": 1024, "osl": 1024, "num_request_ratio": 10},
        goal={"target": "pareto"},
    )

    assert explicit.workload.kv_load_ratio_range == (0.25, 0.75)
    assert defaulted.workload.kv_load_ratio == [0.0, 1.0]


def test_scalar_kv_load_works_for_scalar_goal():
    config = SmartSearchConfig(
        search_space=_search_space(),
        workload={
            "isl": 1024,
            "osl": 1024,
            "kv_load_ratio": 0.75,
            "num_request_ratio": 10,
        },
        goal={"target": "throughput_per_gpu"},
    )

    assert config.workload.kv_load_ratio == 0.75
    assert config.workload.kv_load_ratio_range is None


@pytest.mark.parametrize(
    "value",
    [[], [0.0], [0.0, 0.5, 1.0], [-0.1, 1.0], [1.0, 1.0], float("inf")],
)
def test_invalid_kv_load_ratio_is_rejected(value):
    with pytest.raises(ValidationError):
        SmartSearchConfig(
            search_space=_search_space(),
            workload={
                "isl": 1024,
                "osl": 1024,
                "kv_load_ratio": value,
                "num_request_ratio": 10,
            },
            goal={"target": "pareto"},
        )


@pytest.mark.parametrize("kwargs", [{"ttft_ms": 0}, {"itl_ms": -1}, {"e2e_ms": -5}])
def test_non_positive_sla_is_rejected(kwargs):
    with pytest.raises(ValidationError):
        SLATarget(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_rounds": 0},
        {"parallel_evals": 0},
        {"candidates_per_round": 0},
        {"max_eval_seconds": 0},
    ],
)
def test_non_positive_sweep_control_is_rejected(kwargs):
    with pytest.raises(ValidationError):
        SweepConfig(**kwargs)


@pytest.mark.parametrize(
    ("minimum", "maximum"),
    [(0, 16), (32, 16)],
)
def test_invalid_min_gpu_budget_is_rejected(minimum, maximum):
    with pytest.raises(ValidationError, match="min_gpu_budget"):
        SearchSpace(
            model_name="m",
            hardware_sku="h200_sxm",
            gpu_budget=maximum,
            min_gpu_budget=minimum,
        )


def test_valid_min_gpu_budget_is_accepted():
    space = SearchSpace(
        model_name="m",
        hardware_sku="h200_sxm",
        gpu_budget=32,
        min_gpu_budget=8,
    )

    assert space.min_gpu_budget == 8
