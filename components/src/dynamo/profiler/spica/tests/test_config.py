# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from pydantic import ValidationError

from dynamo.profiler.spica import OptimizationTarget, SmartSearchConfig
from dynamo.profiler.spica.config import (
    OptimizationGoal,
    SearchSpace,
    SLATarget,
    SweepConfig,
    Workload,
)

EXAMPLE = (
    Path(__file__).resolve().parents[6]
    / "examples"
    / "profiler"
    / "spica"
    / "configs"
    / "smart_sweep.yaml"
)
NATIVE_FEATURES = Path(__file__).parent / "data" / "spica_native_features.yaml"


def test_example_yaml_loads():
    cfg = SmartSearchConfig.from_yaml(EXAMPLE)
    assert cfg.search_space.model_name == "deepseek-ai/DeepSeek-V3"
    assert cfg.search_space.deployment_mode == ["disagg", "agg"]
    assert "disabled" in cfg.search_space.planner_scaling_policy
    assert cfg.workload.is_trace_based
    assert cfg.goal.target is OptimizationTarget.GOODPUT_PER_GPU
    assert cfg.sweep.max_rounds == 40


def test_native_feature_fixture_loads_from_migrated_test_data():
    cfg = SmartSearchConfig.from_yaml(NATIVE_FEATURES)

    assert cfg.search_space.num_g2_blocks == 128
    assert cfg.search_space.router_mode == ["kv_router"]
    assert cfg.search_space.parallel_configs == [{"tp": 1, "replicas": 2}]
    assert cfg.workload.request_rate == 1
    assert cfg.sweep.candidates_per_round == 1


def test_defaults_fill_in():
    cfg = SmartSearchConfig(
        search_space={"model_name": "m", "hardware_sku": "h200_sxm"},
        workload={
            "isl": 4000,
            "osl": 1000,
            "request_rate": 25,
            "num_request_ratio": 1000,
        },
    )
    # search-space defaults
    assert cfg.search_space.gpu_budget == 32
    assert cfg.search_space.prefill_block_size == 64
    assert cfg.search_space.prefill_max_num_seqs == [1, 2, 4, 8, 16, 32, 64, 128, 256]
    assert len(cfg.search_space.load_predictor_candidates) == 11
    # goal/sweep default factories
    assert cfg.goal.target is OptimizationTarget.THROUGHPUT
    assert cfg.sweep.parallel_evals == 16


def test_extra_field_forbidden():
    with pytest.raises(ValidationError):
        SmartSearchConfig(
            search_space={"model_name": "m", "hardware_sku": "h", "bogus": 1},
            workload={"isl": 1, "osl": 1, "concurrency": 1, "num_request_ratio": 1},
        )


def test_goodput_requires_sla():
    with pytest.raises(ValidationError):
        OptimizationGoal(target=OptimizationTarget.GOODPUT)  # no SLA
    # ttft alone is not enough (needs ttft+itl, or e2e)
    with pytest.raises(ValidationError):
        OptimizationGoal(target=OptimizationTarget.GOODPUT, sla=SLATarget(ttft_ms=2000))
    # ttft + itl is fine
    OptimizationGoal(
        target=OptimizationTarget.GOODPUT, sla=SLATarget(ttft_ms=2000, itl_ms=30)
    )


def test_target_direction():
    assert OptimizationTarget.THROUGHPUT.maximize
    assert OptimizationTarget.THROUGHPUT_PER_GPU.maximize
    assert OptimizationTarget.THROUGHPUT_PER_USER.maximize
    assert OptimizationTarget.GOODPUT_PER_GPU.maximize
    assert not OptimizationTarget.E2E_LATENCY.maximize


def test_planner_optimization_target_mapping():
    # the planner's scaling objective is derived from the sweep goal
    assert OptimizationTarget.THROUGHPUT.planner_optimization_target == "throughput"
    assert (
        OptimizationTarget.THROUGHPUT_PER_GPU.planner_optimization_target
        == "throughput"
    )
    assert (
        OptimizationTarget.THROUGHPUT_PER_USER.planner_optimization_target
        == "throughput"
    )
    assert OptimizationTarget.E2E_LATENCY.planner_optimization_target == "latency"
    assert OptimizationTarget.GOODPUT.planner_optimization_target == "sla"
    assert OptimizationTarget.GOODPUT_PER_GPU.planner_optimization_target == "sla"
    assert OptimizationTarget.PARETO.planner_optimization_target == "throughput"


def test_throughput_per_gpu_needs_no_sla():
    # throughput_per_gpu is throughput-based -> no SLA required (unlike goodput*)
    OptimizationGoal(
        target=OptimizationTarget.THROUGHPUT_PER_GPU
    )  # must validate without an SLA


def _search_space(**overrides):
    return {"model_name": "m", "hardware_sku": "h200_sxm", **overrides}


def test_subset_of_choices_is_accepted():
    cfg = SmartSearchConfig(
        search_space=_search_space(
            router_mode=["kv_router"], backend=["vllm", "sglang"]
        ),
        workload={"isl": 1, "osl": 1, "concurrency": 1, "num_request_ratio": 1},
    )
    assert cfg.search_space.router_mode == ["kv_router"]
    assert cfg.search_space.backend == ["vllm", "sglang"]


def test_value_outside_choices_rejected():
    with pytest.raises(ValidationError):
        SmartSearchConfig(
            search_space=_search_space(router_mode=["bogus"]),
            workload={"isl": 1, "osl": 1, "concurrency": 1, "num_request_ratio": 1},
        )
    # numeric dimension: 999 is not a listed prefill_max_num_seqs choice
    with pytest.raises(ValidationError):
        SmartSearchConfig(
            search_space=_search_space(prefill_max_num_seqs=[1, 999]),
            workload={"isl": 1, "osl": 1, "concurrency": 1, "num_request_ratio": 1},
        )
    # planner: only listed presets allowed
    with pytest.raises(ValidationError):
        SmartSearchConfig(
            search_space=_search_space(planner_scaling_policy=["disabled", "made_up"]),
            workload={"isl": 1, "osl": 1, "concurrency": 1, "num_request_ratio": 1},
        )


def test_empty_choice_list_rejected():
    with pytest.raises(ValidationError):
        SmartSearchConfig(
            search_space=_search_space(router_mode=[]),
            workload={"isl": 1, "osl": 1, "concurrency": 1, "num_request_ratio": 1},
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("active_decode_blocks_threshold", 64),
        ("active_prefill_tokens_threshold", 4096),
        ("active_prefill_tokens_threshold_frac", 0.5),
        ("no_admission_control", True),
    ],
)
def test_kv_router_admission_controls_fail_closed_until_replay_supports_them(
    field, value
):
    with pytest.raises(ValidationError, match="not supported by the Dynamo replay API"):
        SearchSpace(**_search_space(router_mode=["kv_router"], **{field: value}))


def test_round_robin_ignores_router_admission_controls():
    cfg = SearchSpace(
        **_search_space(
            router_mode=["round_robin"],
            active_prefill_tokens_threshold=4096,
            no_admission_control=True,
        )
    )
    assert cfg.active_prefill_tokens_threshold == 4096
    assert cfg.no_admission_control is True


def test_kv_offload_requires_explicit_replay_transfer_sizing():
    with pytest.raises(ValidationError, match="kv_bytes_per_token is required"):
        SearchSpace(**_search_space(num_g2_blocks=128))

    cfg = SearchSpace(**_search_space(num_g2_blocks=128, kv_bytes_per_token=131072))
    assert cfg.kv_bytes_per_token == 131072


@pytest.mark.parametrize("value", [0, -1])
def test_kv_offload_rejects_non_positive_transfer_sizing(value):
    with pytest.raises(ValidationError, match="kv_bytes_per_token"):
        SearchSpace(**_search_space(num_g2_blocks=128, kv_bytes_per_token=value))


def test_negative_g2_block_count_rejected():
    with pytest.raises(ValidationError, match="num_g2_blocks"):
        SearchSpace(**_search_space(num_g2_blocks=-1))


# --- composite knobs accept raw dicts (preset OR pinned-dict per entry) ---


def test_composite_dict_entries_accepted():
    cfg = SmartSearchConfig(
        search_space=_search_space(
            # a planner dict must be self-contained (all 4 scaling fields)
            planner_scaling_policy=[
                "disabled",
                {
                    "enable_throughput_scaling": True,
                    "enable_load_scaling": False,
                    "throughput_adjustment_interval_seconds": 240,
                    "load_adjustment_interval_seconds": 5,
                },
            ],
            planner_fpm_sampling=[
                {"max_num_fpm_samples": 96, "fpm_sample_bucket_size": 16}
            ],
            # load_predictor needs only the family; params default per family
            load_predictor_candidates=[
                {"load_predictor": "prophet", "prophet_window_size": 30}
            ],
        ),
        workload={"isl": 1, "osl": 1, "concurrency": 1, "num_request_ratio": 1},
    )
    assert isinstance(cfg.search_space.planner_scaling_policy[1], dict)
    assert cfg.search_space.planner_fpm_sampling[0]["max_num_fpm_samples"] == 96


def test_composite_dict_unknown_key_rejected():
    with pytest.raises(ValidationError, match="unknown keys"):
        SmartSearchConfig(
            search_space=_search_space(
                planner_fpm_sampling=[{"max_num_fpm_samples": 64, "bogus": 1}]
            ),
            workload={"isl": 1, "osl": 1, "concurrency": 1, "num_request_ratio": 1},
        )


def test_composite_dict_missing_required_key_rejected():
    # load_predictor dict without the family would KeyError mid-sweep -> reject upfront
    with pytest.raises(ValidationError, match="missing required keys"):
        SmartSearchConfig(
            search_space=_search_space(
                load_predictor_candidates=[{"load_predictor_log1p": True}]
            ),
            workload={"isl": 1, "osl": 1, "concurrency": 1, "num_request_ratio": 1},
        )
    # a planner dict must be self-contained (all of its fields)
    with pytest.raises(ValidationError, match="missing required keys"):
        SmartSearchConfig(
            search_space=_search_space(
                planner_scaling_policy=[
                    {
                        "enable_throughput_scaling": True,
                        "throughput_adjustment_interval_seconds": 240,
                    }
                ]
            ),
            workload={"isl": 1, "osl": 1, "concurrency": 1, "num_request_ratio": 1},
        )


def test_dict_rejected_on_non_composite_knob():
    # router_mode is list[str]; a dict entry is not allowed
    with pytest.raises(ValidationError):
        SmartSearchConfig(
            search_space=_search_space(router_mode=[{"foo": 1}]),
            workload={"isl": 1, "osl": 1, "concurrency": 1, "num_request_ratio": 1},
        )


# --- pinned parallel_configs (structural validation; legality is in enumerate_branches) ---


def test_parallel_configs_pin_requires_single_mode():
    with pytest.raises(ValidationError, match="exactly one mode"):
        SmartSearchConfig(
            search_space=_search_space(
                deployment_mode=["agg", "disagg"], parallel_configs=[{"tp": 4}]
            ),
            workload={"isl": 1, "osl": 1, "concurrency": 1, "num_request_ratio": 1},
        )


def test_parallel_configs_shape_matches_mode():
    # agg entry needs tp
    with pytest.raises(ValidationError, match="'tp' field"):
        SmartSearchConfig(
            search_space=_search_space(
                deployment_mode=["agg"], parallel_configs=[{"replicas": 2}]
            ),
            workload={"isl": 1, "osl": 1, "concurrency": 1, "num_request_ratio": 1},
        )
    # disagg entry needs prefill + decode
    with pytest.raises(ValidationError, match="prefill"):
        SmartSearchConfig(
            search_space=_search_space(
                deployment_mode=["disagg"], parallel_configs=[{"tp": 4}]
            ),
            workload={"isl": 1, "osl": 1, "concurrency": 1, "num_request_ratio": 1},
        )
    with pytest.raises(ValidationError, match=r"prefill.*dict"):
        SmartSearchConfig(
            search_space=_search_space(
                deployment_mode=["disagg"],
                parallel_configs=[{"prefill": 1, "decode": {"tp": 1}}],
            ),
            workload={"isl": 1, "osl": 1, "concurrency": 1, "num_request_ratio": 1},
        )
    with pytest.raises(ValidationError, match=r"decode.*'tp' field"):
        SmartSearchConfig(
            search_space=_search_space(
                deployment_mode=["disagg"],
                parallel_configs=[{"prefill": {"tp": 1}, "decode": {"replicas": 1}}],
            ),
            workload={"isl": 1, "osl": 1, "concurrency": 1, "num_request_ratio": 1},
        )
    # well-formed agg + disagg entries pass structural validation
    SmartSearchConfig(
        search_space=_search_space(
            deployment_mode=["agg"],
            parallel_configs=[{"tp": 4, "moe_ep": 4, "replicas": 2}],
        ),
        workload={"isl": 1, "osl": 1, "concurrency": 1, "num_request_ratio": 1},
    )
    SmartSearchConfig(
        search_space=_search_space(
            deployment_mode=["disagg"],
            parallel_configs=[
                {
                    "prefill": {"tp": 8, "moe_ep": 8},
                    "decode": {"tp": 1, "attention_dp": 8, "moe_ep": 8},
                }
            ],
        ),
        workload={"isl": 1, "osl": 1, "concurrency": 1, "num_request_ratio": 1},
    )


# --- synthetic concurrency is an integer in-flight cap ---


def test_fractional_concurrency_rejected():
    # a float concurrency would degenerate the closed-loop cap (e.g. 0.5 -> 0)
    with pytest.raises(ValidationError):
        Workload(isl=1, osl=1, concurrency=1.9, num_request_ratio=1)


def test_concurrency_is_integer_in_flight_cap():
    wl = Workload(isl=4000, osl=1000, concurrency=2, num_request_ratio=1000)
    assert wl.effective_in_flight_cap() == 2
    assert isinstance(wl.effective_in_flight_cap(), int)


def test_non_positive_concurrency_rejected():
    with pytest.raises(ValidationError):
        Workload(isl=1, osl=1, concurrency=0, num_request_ratio=1)


# --- num_request_ratio: request count = round(ratio * load) ---


def test_num_request_ratio_scales_with_concurrency():
    wl = Workload(isl=4000, osl=1000, concurrency=256, num_request_ratio=10)
    assert wl.resolved_request_count() == 2560  # 10 * 256
    # the candidate-derived override wins over the workload's own concurrency
    assert wl.resolved_request_count(concurrency_override=8) == 80  # 10 * 8


def test_num_request_ratio_scales_with_request_rate():
    wl = Workload(isl=4000, osl=1000, request_rate=25, num_request_ratio=4)
    assert wl.resolved_request_count() == 100  # round(4 * 25)


def test_num_request_ratio_required_for_synthetic():
    with pytest.raises(ValidationError, match="num_request_ratio"):
        Workload(isl=1, osl=1, concurrency=1)  # missing num_request_ratio


# --- pareto goal + candidate-relative KV load ---


def test_pareto_goal_defaults_to_tput_per_gpu_and_per_user():
    goal = OptimizationGoal(target=OptimizationTarget.PARETO)
    assert goal.is_pareto
    assert goal.resolved_pareto_objectives == [
        OptimizationTarget.THROUGHPUT_PER_GPU,
        OptimizationTarget.THROUGHPUT_PER_USER,
    ]


def test_pareto_objectives_configurable():
    goal = OptimizationGoal(
        target=OptimizationTarget.PARETO,
        pareto_objectives=[
            OptimizationTarget.THROUGHPUT_PER_USER,
            OptimizationTarget.GOODPUT_PER_GPU,
        ],
        sla=SLATarget(
            ttft_ms=2000, itl_ms=30
        ),  # goodput_per_gpu objective needs an SLA
    )
    assert goal.resolved_pareto_objectives[1] is OptimizationTarget.GOODPUT_PER_GPU


def test_pareto_objectives_only_with_pareto_target():
    with pytest.raises(ValidationError, match="only valid when target is 'pareto'"):
        OptimizationGoal(
            target=OptimizationTarget.THROUGHPUT,
            pareto_objectives=[
                OptimizationTarget.THROUGHPUT_PER_GPU,
                OptimizationTarget.THROUGHPUT_PER_USER,
            ],
        )


def test_pareto_goodput_objective_requires_sla():
    with pytest.raises(ValidationError, match="require an SLA"):
        OptimizationGoal(
            target=OptimizationTarget.PARETO,
            pareto_objectives=[
                OptimizationTarget.GOODPUT_PER_GPU,
                OptimizationTarget.THROUGHPUT_PER_USER,
            ],
        )


def test_pareto_objectives_must_be_distinct():
    with pytest.raises(ValidationError, match="distinct"):
        OptimizationGoal(
            target=OptimizationTarget.PARETO,
            pareto_objectives=[
                OptimizationTarget.THROUGHPUT_PER_GPU,
                OptimizationTarget.THROUGHPUT_PER_GPU,
            ],
        )


def test_pareto_objectives_explicit_empty_or_singleton_rejected():
    # an explicit [] must NOT silently fall back to the default pair (None does that)
    with pytest.raises(ValidationError, match="at least 2 objectives"):
        OptimizationGoal(target=OptimizationTarget.PARETO, pareto_objectives=[])
    with pytest.raises(ValidationError, match="at least 2 objectives"):
        OptimizationGoal(
            target=OptimizationTarget.PARETO,
            pareto_objectives=[OptimizationTarget.THROUGHPUT_PER_GPU],
        )


def test_concurrency_is_always_scalar():
    for target in ("throughput_per_gpu", "pareto"):
        with pytest.raises(ValidationError):
            SmartSearchConfig(
                search_space=_search_space(),
                workload={
                    "isl": 1024,
                    "osl": 1024,
                    "concurrency": [8, 16, 32],
                    "num_request_ratio": 10,
                },
                goal={"target": target},
            )


def test_kv_load_ratio_range_only_under_pareto():
    with pytest.raises(ValidationError, match=r"ranged workload.kv_load_ratio"):
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
    cfg = SmartSearchConfig(
        search_space=_search_space(),
        workload={
            "isl": 1024,
            "osl": 1024,
            "kv_load_ratio": [0.0, 1.0],
            "num_request_ratio": 10,
        },
        goal={"target": "pareto"},
    )
    assert cfg.workload.kv_load_ratio_range == (0.0, 1.0)


def test_scalar_kv_load_ratio_works_for_non_pareto():
    cfg = SmartSearchConfig(
        search_space=_search_space(),
        workload={
            "isl": 1024,
            "osl": 1024,
            "kv_load_ratio": 0.75,
            "num_request_ratio": 10,
        },
        goal={"target": "throughput_per_gpu"},
    )
    assert cfg.workload.kv_load_ratio == 0.75
    assert cfg.workload.kv_load_ratio_range is None


def test_pareto_defaults_to_full_kv_load_range():
    cfg = SmartSearchConfig(
        search_space=_search_space(),
        workload={"isl": 1024, "osl": 1024, "num_request_ratio": 10},
        goal={"target": "pareto"},
    )
    assert cfg.workload.kv_load_ratio == [0.0, 1.0]


@pytest.mark.parametrize(
    "value", [[], [0.0], [0.0, 0.5, 1.0], [-0.1, 1.0], [1.0, 1.0], float("inf")]
)
def test_invalid_kv_load_ratio_rejected(value):
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


def test_pareto_target_has_no_scalar_direction():
    with pytest.raises(ValueError, match="no scalar direction"):
        _ = OptimizationTarget.PARETO.maximize


# --- SLA targets must be strictly positive ---


@pytest.mark.parametrize("kwargs", [{"ttft_ms": 0}, {"itl_ms": -1}, {"e2e_ms": -5}])
def test_non_positive_sla_rejected(kwargs):
    with pytest.raises(ValidationError):
        SLATarget(**kwargs)


def test_positive_sla_accepted():
    SLATarget(ttft_ms=2000, itl_ms=30)
    SLATarget(e2e_ms=5000)


# --- sweep run-control must be positive ---


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_rounds": 0},
        {"max_rounds": -1},
        {"parallel_evals": 0},
        {"parallel_evals": -3},
        {"candidates_per_round": 0},
        {"candidates_per_round": -2},
    ],
)
def test_sweep_non_positive_rejected(kwargs):
    with pytest.raises(ValidationError):
        SweepConfig(**kwargs)


# --- min_gpu_budget bounds ---


def test_min_gpu_budget_exceeding_budget_rejected():
    with pytest.raises(ValidationError, match="min_gpu_budget"):
        SearchSpace(
            model_name="m", hardware_sku="h200_sxm", gpu_budget=16, min_gpu_budget=32
        )


def test_non_positive_min_gpu_budget_rejected():
    with pytest.raises(ValidationError, match="min_gpu_budget"):
        SearchSpace(
            model_name="m", hardware_sku="h200_sxm", gpu_budget=16, min_gpu_budget=0
        )


def test_min_gpu_budget_within_budget_accepted():
    ss = SearchSpace(
        model_name="m", hardware_sku="h200_sxm", gpu_budget=32, min_gpu_budget=8
    )
    assert ss.min_gpu_budget == 8
