# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Vizier-backed sampler. Needs the pinned Vizier/JAX stack; skips otherwise."""

import json
import uuid

import pytest

from dynamo.profiler.spica.parallel_enum import ParallelShape, ReplicaParallelConfig
from dynamo.profiler.spica.parallel_projection import (
    AGG_ATTENTION_MODE,
    AGG_FFN_MODE,
    AGG_GPUS_PER_ENGINE,
    USED_GPU_RATIO,
)
from dynamo.profiler.spica.sampler import (
    Suggestion,
    _decoder_for,
    _index_decoder,
    make_branch_sampler,
)
from dynamo.profiler.spica.search_space import BranchSpace

pytest.importorskip("vizier")

pytestmark = [
    pytest.mark.timeout(300),
    pytest.mark.filterwarnings(
        r"ignore::DeprecationWarning:(jax|jaxlib|equinox|jaxopt)(\..*)?"
    ),
    pytest.mark.filterwarnings("ignore:.*JAXopt is no longer maintained.*"),
    # google-vizier 0.1.21 still uses RandomState.random_integers internally.
    pytest.mark.filterwarnings(
        "ignore:This function is deprecated.*call randint.*:DeprecationWarning"
    ),
]


def test_decoder_for_categorical_strings_returns_str():
    decode = _decoder_for(["round_robin", "kv_router"])
    assert decode("kv_router") == "kv_router"
    assert isinstance(decode("kv_router"), str)


def test_decoder_for_discrete_int_rounds_to_native_int():
    decode = _decoder_for([256, 512, 1024])
    # Vizier stores discrete params as floats; decode must round back to int.
    assert decode("512.0") == 512
    assert decode(512.0) == 512
    assert isinstance(decode("512.0"), int)


def test_decoder_for_discrete_float_returns_native_float():
    decode = _decoder_for([0.0, 0.5, 1.0])
    result = decode("0.5")
    assert result == 0.5
    assert isinstance(result, float)


def test_index_decoder_maps_index_back_to_entry():
    raw = {"enable_throughput_scaling": True}
    decode = _index_decoder(["disabled", raw])
    assert decode("0.0") == "disabled"
    assert decode("1") == raw  # exact entry, including the dict


def _branch() -> BranchSpace:
    configs = tuple(
        ReplicaParallelConfig(ParallelShape(tp=4, dp=1, moe_tp=1, moe_ep=4), replicas=r)
        for r in (1, 2, 4)
    )
    return BranchSpace(
        deployment_mode="agg",
        parallel_configs=configs,
        supported_backends={c: frozenset({"trtllm"}) for c in configs},
        knob_choices={
            "backend": ["trtllm"],  # single choice -> constant
            "router_mode": ["round_robin"],  # single choice -> constant, not a param
            "planner_scaling_policy": ["disabled", "throughput_180_5"],  # categorical
            "planner_fpm_sampling": ["default", "large"],
            "planner_load_sensitivity": ["default"],  # single -> constant
            "agg_max_num_batched_tokens": [8192, 16384],  # discrete int
            "agg_max_num_seqs": [256, 512, 1024],  # discrete int
            "overlap_score_credit": [
                0.0,
                0.5,
                1.0,
            ],  # discrete float (ignored under round_robin)
        },
    )


def test_suggest_produces_valid_selections():
    branch = _branch()
    sampler = make_branch_sampler(branch, study_id="test_valid")
    suggestions = sampler.suggest(count=4)
    assert len(suggestions) == 4
    for s in suggestions:
        assert isinstance(s, Suggestion)
        # constants injected; branch identity present
        assert (
            s.selection["deployment_mode"] == "agg"
            and s.selection["backend"] == "trtllm"
        )
        assert s.selection["router_mode"] == "round_robin"
        assert s.selection["planner_load_sensitivity"] == "default"
        # searched knobs land within their choice sets, native types preserved
        assert s.selection["planner_scaling_policy"] in {"disabled", "throughput_180_5"}
        assert s.selection["agg_max_num_seqs"] in {256, 512, 1024}
        assert isinstance(s.selection["agg_max_num_seqs"], int)
        assert s.selection["overlap_score_credit"] in {0.0, 0.5, 1.0}
        # the chosen parallel config is one of the branch's
        assert s.parallel_config in branch.parallel_configs


def test_dict_form_composite_decodes_via_index():
    # planner_scaling_policy mixes a preset id and a pinned dict; the sampler
    # categorizes over the index and decodes back to the exact entry (str or dict).
    raw = {
        "enable_throughput_scaling": True,
        "enable_load_scaling": False,
        "throughput_adjustment_interval_seconds": 240,
        "load_adjustment_interval_seconds": 5,
    }
    pc = ReplicaParallelConfig(
        ParallelShape(tp=4, dp=1, moe_tp=1, moe_ep=4), replicas=1
    )
    branch = BranchSpace(
        deployment_mode="agg",
        parallel_configs=(pc,),
        supported_backends={pc: frozenset({"trtllm"})},
        knob_choices={
            "backend": ["trtllm"],
            "planner_scaling_policy": [
                "disabled",
                raw,
            ],  # str | dict -> index categorical
            "planner_fpm_sampling": ["default"],
            "planner_load_sensitivity": ["default"],
        },
    )
    sampler = make_branch_sampler(branch, study_id="test_dict_index")
    seen = [s.selection["planner_scaling_policy"] for s in sampler.suggest(count=5)]
    assert seen, "expected suggestions"
    # every decoded entry is exactly one of the two originals (native type preserved)
    for entry in seen:
        assert entry == "disabled" or entry == raw
        assert isinstance(entry, (str, dict))


def test_parallel_suggestions_project_to_valid_configs(monkeypatch):
    monkeypatch.setenv("SPICA_VIZIER_ALGO", "RANDOM_SEARCH")
    branch = _branch()
    sampler = make_branch_sampler(branch, study_id="test_structured_parallel")

    suggestions = sampler.suggest(count=4)

    assert suggestions
    for suggestion in suggestions:
        assert USED_GPU_RATIO in suggestion.handle.parameters
        assert suggestion.projection is not None
        assert suggestion.parallel_config in branch.parallel_configs
        assert (
            suggestion.selection["backend"]
            in branch.supported_backends[suggestion.parallel_config]
        )


def test_parallel_search_exposes_ordered_size_and_mode_dimensions(monkeypatch):
    monkeypatch.setenv("SPICA_VIZIER_ALGO", "RANDOM_SEARCH")
    tep4 = ReplicaParallelConfig(
        ParallelShape(tp=4, dp=1, moe_tp=1, moe_ep=4), replicas=4
    )
    dtp8 = ReplicaParallelConfig(
        ParallelShape(tp=1, dp=8, moe_tp=8, moe_ep=1), replicas=2
    )
    configs = (tep4, dtp8)
    branch = BranchSpace(
        deployment_mode="agg",
        parallel_configs=configs,
        supported_backends={config: frozenset({"vllm"}) for config in configs},
        knob_choices={"backend": ["vllm"]},
        gpu_budget=32,
    )
    sampler = make_branch_sampler(branch, study_id="test_structured_dimensions")

    suggestions = sampler.suggest(count=6)

    assert suggestions
    for suggestion in suggestions:
        params = suggestion.handle.parameters
        assert params[AGG_GPUS_PER_ENGINE] in (4, 8)
        assert params[AGG_ATTENTION_MODE] in ("tp", "dp")
        assert params[AGG_FFN_MODE] in ("tp", "ep")
        assert suggestion.parallel_config in configs


def test_single_parallel_config_is_pinned(monkeypatch):
    monkeypatch.setenv("SPICA_VIZIER_ALGO", "RANDOM_SEARCH")
    pinned = ReplicaParallelConfig(
        ParallelShape(tp=4, dp=1, moe_tp=1, moe_ep=4), replicas=8
    )
    branch = BranchSpace(
        deployment_mode="agg",
        parallel_configs=(pinned,),
        supported_backends={pinned: frozenset({"vllm"})},
        knob_choices={"backend": ["vllm"], "agg_max_num_seqs": [256, 512]},
        gpu_budget=32,
    )
    sampler = make_branch_sampler(branch, study_id="test_structured_pin")

    suggestions = sampler.suggest(count=3)

    assert suggestions
    assert all(suggestion.parallel_config == pinned for suggestion in suggestions)
    assert all(suggestion.projection is None for suggestion in suggestions)
    assert all(
        USED_GPU_RATIO not in suggestion.handle.parameters for suggestion in suggestions
    )


def test_fully_pinned_study_uses_only_internal_constant(monkeypatch):
    monkeypatch.setenv("SPICA_VIZIER_ALGO", "RANDOM_SEARCH")
    pinned = ReplicaParallelConfig(
        ParallelShape(tp=4, dp=1, moe_tp=1, moe_ep=4), replicas=8
    )
    branch = BranchSpace(
        deployment_mode="agg",
        parallel_configs=(pinned,),
        supported_backends={pinned: frozenset({"vllm"})},
        knob_choices={"backend": ["vllm"]},
        gpu_budget=32,
    )
    sampler = make_branch_sampler(branch, study_id="test_structured_fully_pinned")

    suggestion = sampler.suggest(count=1)[0]

    assert suggestion.parallel_config == pinned
    assert suggestion.projection is None
    assert dict(suggestion.handle.parameters) == {"_spica_constant": "0"}


def test_projection_is_written_to_trial_metadata(monkeypatch):
    monkeypatch.setenv("SPICA_VIZIER_ALGO", "RANDOM_SEARCH")
    sampler = make_branch_sampler(_branch(), study_id="test_structured_metadata")
    suggestion = sampler.suggest(count=1)[0]

    sampler.observe(suggestion, {"objective": 1.0})

    metadata = suggestion.handle.materialize().metadata
    projection = json.loads(metadata["spica_projection"])
    assert projection["actual_parallel_config"]
    assert projection["requested_features"]
    assert projection["actual_features"]


def test_suggest_observe_round_trips():
    # Verify the ask/tell round-trip feeds back without error and the study
    # tracks the best observed score. (Convergence quality isn't asserted —
    # Vizier GP-bandit is slow, ~seconds per suggest, so keep trial counts low.)
    branch = _branch()
    sampler = make_branch_sampler(
        branch, study_id=f"test_round_trip_{uuid.uuid4().hex}"
    )
    scores = []
    for _ in range(2):
        for s in sampler.suggest(count=2):
            score = float(s.selection["agg_max_num_seqs"])
            scores.append(score)
            sampler.observe(s, {"objective": score})
    assert len(scores) == 4
    best = list(sampler._study.optimal_trials())[0].materialize()
    assert best.final_measurement.metrics["objective"].value == max(scores)


def _branch_with_kv_load() -> BranchSpace:
    b = _branch()
    return BranchSpace(
        deployment_mode=b.deployment_mode,
        parallel_configs=b.parallel_configs,
        supported_backends=b.supported_backends,
        knob_choices=b.knob_choices,
        float_ranges={"kv_load_ratio": (0.0, 1.0)},
    )


def test_pareto_study_sweeps_kv_load_and_returns_front(monkeypatch):
    # A multi-objective study: a continuous KV-load ratio + two maximized metrics.
    # Verifies the >=2-metric study builds, observe carries both, and optimal_trials() returns
    # a non-empty Pareto set whose trials carry both objectives.
    monkeypatch.setenv("SPICA_VIZIER_ALGO", "RANDOM_SEARCH")
    branch = _branch_with_kv_load()
    sampler = make_branch_sampler(
        branch,
        study_id=f"test_pareto_kv_load_{uuid.uuid4().hex}",
        objectives=[("throughput_per_gpu", True), ("throughput_per_user", True)],
    )
    for s in sampler.suggest(count=3):
        ratio = s.selection["kv_load_ratio"]
        assert 0.0 <= ratio <= 1.0
        # a tradeoff: throughput-per-gpu rises with load, per-user falls
        sampler.observe(
            s,
            {
                "throughput_per_gpu": ratio * 10.0,
                "throughput_per_user": 100.0 / (ratio + 1.0),
            },
        )
    optimal = list(sampler._study.optimal_trials())
    assert optimal  # multi-objective study -> non-empty Pareto frontier
    for t in optimal:
        metrics = t.materialize().final_measurement.metrics
        assert "throughput_per_gpu" in metrics and "throughput_per_user" in metrics
