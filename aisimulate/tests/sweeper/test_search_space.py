# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend branch enumeration and runner-capability filtering."""

from pathlib import Path

import pytest

from aisimulate.sweeper.config import SmartSearchConfig
from aisimulate.sweeper.kv_estimate import NoPerfDatabase
from aisimulate.sweeper.model_hw import NoViableParallelConfig
from aisimulate.sweeper.parallel_enum import ParallelShape, ReplicaParallelConfig
from aisimulate.sweeper.replay import RunnerCapabilities
from aisimulate.sweeper.search_space import branch_knob_choices, enumerate_branches

TRACE = str(Path(__file__).parent / "data" / "mooncake_tiny.jsonl")

_AGG_CFG = ReplicaParallelConfig(
    ParallelShape(tp=1, dp=1, moe_tp=1, moe_ep=1), replicas=1
)


def _config(**search_overrides) -> SmartSearchConfig:
    search_space = {
        "model_name": "deepseek-ai/DeepSeek-V3",
        "hardware_sku": "gb200",
        "backend": ["trtllm"],
        "deployment_mode": ["agg"],
        "gpu_budget": 16,
    }
    search_space.update(search_overrides)
    return SmartSearchConfig(
        search_space=search_space,
        workload={"trace_path": TRACE},
    )


def _capabilities(*pairs):
    return RunnerCapabilities(supported_backend_topologies=tuple(pairs))


def test_branch_knobs_are_backend_only_and_mode_specific():
    search_space = _config().search_space

    agg = branch_knob_choices(search_space, "agg")
    disagg = branch_knob_choices(search_space, "disagg")

    assert set(agg) == {"agg_max_num_batched_tokens", "agg_max_num_seqs"}
    assert set(disagg) == {
        "prefill_max_num_batched_tokens",
        "prefill_max_num_seqs",
        "decode_max_num_batched_tokens",
        "decode_max_num_seqs",
    }
    assert not {"router_mode", "planner_scaling_policy", "num_g2_blocks"} & set(agg)


def test_enumerate_real_backend_space_honors_runner_topologies():
    config = _config(
        deployment_mode=["agg", "disagg"],
        backend=["trtllm"],
        gpu_budget=16,
    )
    capabilities = _capabilities(("trtllm", "agg"))

    with pytest.warns(UserWarning, match="runner-incompatible.*trtllm"):
        branches = enumerate_branches(config, runner_capabilities=capabilities)

    assert [branch.deployment_mode for branch in branches] == ["agg"]
    branch = branches[0]
    assert branch.knob_choices["backend"] == ["trtllm"]
    assert branch.parallel_configs
    assert all(config.total_gpus <= 16 for config in branch.parallel_configs)
    assert all(
        branch.supported_backends[config] == frozenset({"trtllm"})
        for config in branch.parallel_configs
    )
    assert "agg_max_num_seqs" in branch.knob_choices


def test_runner_incompatible_backend_is_removed_before_perf_lookup(monkeypatch):
    calls = []

    def fake_parallel_configs(
        model,
        hardware,
        *,
        gpu_budget,
        deployment_mode,
        backend,
        min_gpu_budget=None,
        max_seq_len=None,
    ):
        calls.append((deployment_mode, backend))
        return [_AGG_CFG]

    monkeypatch.setattr(
        "aisimulate.sweeper.search_space.parallel_configs_for", fake_parallel_configs
    )
    config = _config(
        deployment_mode=["disagg"],
        backend=["trtllm", "vllm"],
        gpu_budget=8,
    )

    (branch,) = enumerate_branches(
        config,
        runner_capabilities=_capabilities(("vllm", "disagg")),
    )

    assert calls == [("disagg", "vllm")]
    assert branch.knob_choices["backend"] == ["vllm"]
    assert branch.supported_backends[_AGG_CFG] == frozenset({"vllm"})


def test_pinned_parallel_configs_replace_generated_menu():
    config = _config(
        model_name="meta-llama/Meta-Llama-3.1-8B",
        deployment_mode=["agg"],
        gpu_budget=32,
        parallel_configs=[{"tp": 4, "replicas": 2}, {"tp": 8, "replicas": 1}],
    )

    (branch,) = enumerate_branches(config)

    assert {(item.shape.tp, item.replicas) for item in branch.parallel_configs} == {
        (4, 2),
        (8, 1),
    }
    assert all(item.total_gpus == 8 for item in branch.parallel_configs)


def test_illegal_pinned_parallel_config_is_rejected():
    config = _config(
        model_name="meta-llama/Meta-Llama-3.1-8B",
        deployment_mode=["agg"],
        gpu_budget=32,
        parallel_configs=[{"tp": 3, "replicas": 1}],
    )

    with pytest.raises(NoViableParallelConfig):
        enumerate_branches(config)


def test_infeasible_mode_is_skipped_while_viable_mode_remains(monkeypatch):
    def fake_parallel_configs(
        model,
        hardware,
        *,
        gpu_budget,
        deployment_mode,
        backend,
        min_gpu_budget=None,
        max_seq_len=None,
    ):
        if deployment_mode == "disagg":
            raise NoViableParallelConfig("disagg does not fit")
        return [_AGG_CFG]

    monkeypatch.setattr(
        "aisimulate.sweeper.search_space.parallel_configs_for", fake_parallel_configs
    )
    config = _config(
        deployment_mode=["agg", "disagg"],
        backend=["trtllm"],
        gpu_budget=8,
    )

    with pytest.warns(UserWarning, match="disagg.*skipped"):
        branches = enumerate_branches(config)

    assert [branch.deployment_mode for branch in branches] == ["agg"]
    assert branches[0].supported_backends[_AGG_CFG] == frozenset({"trtllm"})


@pytest.mark.filterwarnings(
    "ignore:smart-sweep.*deployment_mode=.* skipped.*:UserWarning"
)
def test_all_modes_infeasible_raises(monkeypatch):
    def always_raise(*args, **kwargs):
        raise NoViableParallelConfig("nothing fits")

    monkeypatch.setattr(
        "aisimulate.sweeper.search_space.parallel_configs_for", always_raise
    )
    config = _config(
        deployment_mode=["agg", "disagg"],
        backend=["trtllm"],
        gpu_budget=1,
    )

    with pytest.raises(NoViableParallelConfig, match="no deployment_mode"):
        enumerate_branches(config)


def test_backend_without_perf_database_is_dropped(monkeypatch):
    def fake_parallel_configs(
        model,
        hardware,
        *,
        gpu_budget,
        deployment_mode,
        backend,
        min_gpu_budget=None,
        max_seq_len=None,
    ):
        if backend == "vllm":
            raise NoPerfDatabase("no vLLM perf database")
        return [_AGG_CFG]

    monkeypatch.setattr(
        "aisimulate.sweeper.search_space.parallel_configs_for", fake_parallel_configs
    )
    config = _config(
        deployment_mode=["agg"],
        backend=["vllm", "trtllm"],
        gpu_budget=8,
    )

    (branch,) = enumerate_branches(config)

    assert branch.knob_choices["backend"] == ["trtllm"]
    assert branch.supported_backends[_AGG_CFG] == frozenset({"trtllm"})


def test_viable_backend_choices_preserve_user_order(monkeypatch):
    monkeypatch.setattr(
        "aisimulate.sweeper.search_space.parallel_configs_for",
        lambda *args, **kwargs: [_AGG_CFG],
    )
    config = _config(
        deployment_mode=["agg"],
        backend=["trtllm", "vllm"],
        gpu_budget=8,
    )

    (branch,) = enumerate_branches(config)

    assert branch.knob_choices["backend"] == ["trtllm", "vllm"]


def test_kv_load_range_becomes_continuous_branch_dimension(monkeypatch):
    monkeypatch.setattr(
        "aisimulate.sweeper.search_space.parallel_configs_for",
        lambda *args, **kwargs: [_AGG_CFG],
    )
    config = SmartSearchConfig(
        search_space={
            "model_name": "m",
            "hardware_sku": "h200_sxm",
            "backend": ["trtllm"],
            "deployment_mode": ["agg"],
        },
        workload={
            "isl": 1024,
            "osl": 1024,
            "kv_load_ratio": [0.0, 1.0],
            "num_request_ratio": 10,
        },
        goal={"target": "pareto"},
    )

    (branch,) = enumerate_branches(config)

    assert branch.float_ranges == {"kv_load_ratio": (0.0, 1.0)}
    assert "kv_load_ratio" not in branch.knob_choices


def test_scalar_kv_load_is_pinned_in_branch_selection(monkeypatch):
    monkeypatch.setattr(
        "aisimulate.sweeper.search_space.parallel_configs_for",
        lambda *args, **kwargs: [_AGG_CFG],
    )
    config = SmartSearchConfig(
        search_space={
            "model_name": "m",
            "hardware_sku": "h200_sxm",
            "backend": ["trtllm"],
            "deployment_mode": ["agg"],
        },
        workload={
            "isl": 1024,
            "osl": 1024,
            "kv_load_ratio": 0.6,
            "num_request_ratio": 10,
        },
    )

    (branch,) = enumerate_branches(config)

    assert branch.float_ranges == {}
    assert branch.knob_choices["kv_load_ratio"] == [0.6]


def test_partial_illegal_pinned_config_raises(monkeypatch):
    monkeypatch.setattr(
        "aisimulate.sweeper.search_space.parallel_configs_for",
        lambda *args, **kwargs: [_AGG_CFG],
    )
    config = _config(
        deployment_mode=["agg"],
        backend=["trtllm"],
        gpu_budget=8,
        parallel_configs=[{"tp": 1, "replicas": 1}, {"tp": 2, "replicas": 1}],
    )

    with pytest.raises(
        NoViableParallelConfig,
        match="legal/KV-feasible for no configured backend",
    ):
        enumerate_branches(config)
