# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: E402

"""Per-branch candidate space. The branch enumeration calls the KV-feasibility
path, so it needs aiconfigurator + a perf DB (skips otherwise)."""

import pytest

pytest.importorskip("aiconfigurator")

from dynamo._internal.aic import AicMemoryEstimatorUnavailableError
from dynamo.profiler.spica.config import SmartSearchConfig
from dynamo.profiler.spica.kv_estimate import NoPerfDatabase, _load_memory_estimator
from dynamo.profiler.spica.model_hw import NoViableParallelConfig
from dynamo.profiler.spica.parallel_enum import ParallelShape, ReplicaParallelConfig
from dynamo.profiler.spica.search_space import branch_knob_choices, enumerate_branches


def _config(**ss_overrides) -> SmartSearchConfig:
    ss = {
        "model_name": "deepseek-ai/DeepSeek-V3",
        "hardware_sku": "gb200",
        "backend": ["trtllm"],
        "deployment_mode": ["agg"],
        "gpu_budget": 16,
    }
    ss.update(ss_overrides)
    return SmartSearchConfig(search_space=ss, workload={"trace_path": "/tmp/t.jsonl"})


def _require_memory_estimator() -> None:
    try:
        _load_memory_estimator()
    except AicMemoryEstimatorUnavailableError as exc:
        pytest.skip(str(exc))


def test_branch_knob_choices_by_mode():
    ss = _config().search_space
    agg = branch_knob_choices(ss, "agg")
    assert "agg_max_num_seqs" in agg and "prefill_max_num_seqs" not in agg
    assert "router_mode" in agg and "planner_scaling_policy" in agg
    disagg = branch_knob_choices(ss, "disagg")
    assert "prefill_max_num_seqs" in disagg and "decode_max_num_seqs" in disagg
    assert "agg_max_num_seqs" not in disagg


def test_host_disk_cache_weights_gated_on_offload():
    # host/disk cache-hit weights are dead in the replay unless multi-tier KV offload is
    # on (they multiply host/disk extension blocks, which are 0 when num_g2_blocks==0),
    # so they are only swept when offload is enabled.
    off = branch_knob_choices(
        _config().search_space, "agg"
    )  # num_g2_blocks defaults to 0
    assert "host_cache_hit_weight" not in off and "disk_cache_hit_weight" not in off
    assert (
        "overlap_score_credit" in off and "prefill_load_scale" in off
    )  # live knobs kept

    on = branch_knob_choices(
        _config(num_g2_blocks=4096, kv_bytes_per_token=131072).search_space, "agg"
    )
    assert "host_cache_hit_weight" in on and "disk_cache_hit_weight" in on


def test_round_robin_only_prunes_router_dependent_knobs():
    choices = branch_knob_choices(
        _config(router_mode=["round_robin"]).search_space, "agg"
    )

    assert "router_mode" in choices
    assert (
        not {
            "overlap_score_credit",
            "prefill_load_scale",
            "host_cache_hit_weight",
            "disk_cache_hit_weight",
            "router_temperature",
        }
        & choices.keys()
    )


def test_mixed_router_modes_keep_live_router_knobs():
    choices = branch_knob_choices(
        _config(router_mode=["round_robin", "kv_router"]).search_space, "agg"
    )

    assert "overlap_score_credit" in choices
    assert "prefill_load_scale" in choices
    assert "router_temperature" in choices


def test_disabled_planner_prunes_dependent_knobs():
    choices = branch_knob_choices(
        _config(planner_scaling_policy=["disabled"]).search_space, "agg"
    )

    assert "planner_scaling_policy" in choices
    assert "planner_fpm_sampling" not in choices
    assert "planner_load_sensitivity" not in choices


def test_mixed_planner_policies_keep_dependent_knobs():
    choices = branch_knob_choices(
        _config(planner_scaling_policy=["disabled", "load_180_5"]).search_space,
        "agg",
    )

    assert "planner_fpm_sampling" in choices
    assert "planner_load_sensitivity" in choices


@pytest.mark.filterwarnings(
    "ignore:\\[EXPERIMENTAL\\] Spica cannot apply KV-capacity filtering.*:UserWarning"
)
def test_enumerate_branches_deepseek_gb200():
    cfg = _config(deployment_mode=["agg", "disagg"], backend=["trtllm"], gpu_budget=16)
    try:
        with pytest.warns(UserWarning, match="disagg.*replay-incompatible.*trtllm"):
            branches = enumerate_branches(cfg)
    except (NoPerfDatabase, NoViableParallelConfig):
        pytest.skip("no gb200/trtllm perf DB")
    except ValueError as exc:
        if "unsupported model/backend/GPU" in str(exc):
            pytest.skip(f"native KV build unavailable: {exc}")
        raise
    # Dynamo replay rejects TRT-LLM disaggregation, so only agg is searchable.
    assert {b.deployment_mode for b in branches} == {"agg"}
    for b in branches:
        assert b.knob_choices["backend"] == ["trtllm"]  # only the viable backend(s)
        assert len(b.parallel_configs) > 0  # KV-feasible configs exist
        assert all(c.total_gpus <= 16 for c in b.parallel_configs)
        # every config is tagged with the backends that support it
        assert all(
            b.supported_backends[c] == frozenset({"trtllm"}) for c in b.parallel_configs
        )
        # planner + router knobs always present; engine knobs match the mode
        assert "planner_scaling_policy" in b.knob_choices
        key = (
            "agg_max_num_seqs" if b.deployment_mode == "agg" else "decode_max_num_seqs"
        )
        assert key in b.knob_choices


def test_replay_incompatible_backend_is_removed_before_sampling(monkeypatch):
    calls = []

    def fake_pcf(
        model,
        hw,
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
        "dynamo.profiler.spica.search_space.parallel_configs_for", fake_pcf
    )
    cfg = _config(deployment_mode=["disagg"], backend=["trtllm", "vllm"], gpu_budget=8)

    (branch,) = enumerate_branches(cfg)

    assert calls == [("disagg", "vllm")]
    assert branch.knob_choices["backend"] == ["vllm"]
    assert branch.supported_backends[_AGG_CFG] == frozenset({"vllm"})


def test_pinned_parallel_configs_replace_the_menu():
    # a dense, KV-trivial model so the pinned shapes are guaranteed feasible
    _require_memory_estimator()
    cfg = _config(
        model_name="meta-llama/Meta-Llama-3.1-8B",
        deployment_mode=["agg"],
        gpu_budget=32,
        parallel_configs=[{"tp": 4, "replicas": 2}, {"tp": 8, "replicas": 1}],
    )
    try:
        branches = enumerate_branches(cfg)
    except NoPerfDatabase:
        pytest.skip("no gb200/trtllm perf DB")
    (branch,) = branches
    menu = {(c.shape.tp, c.replicas) for c in branch.parallel_configs}
    assert menu == {(4, 2), (8, 1)}  # exactly the pinned set, not the full enumeration
    assert all(c.total_gpus == 8 for c in branch.parallel_configs)


@pytest.mark.filterwarnings(
    "ignore:\\[EXPERIMENTAL\\] Spica cannot apply KV-capacity filtering.*:UserWarning"
)
def test_pinned_parallel_config_illegal_is_rejected():
    cfg = _config(
        model_name="meta-llama/Meta-Llama-3.1-8B",
        deployment_mode=["agg"],
        gpu_budget=32,
        parallel_configs=[{"tp": 3, "replicas": 1}],  # tp=3 not on the GPU ladder
    )
    try:
        with pytest.raises(NoViableParallelConfig):
            enumerate_branches(cfg)
    except NoPerfDatabase:
        pytest.skip("no gb200/trtllm perf DB")


# --- per-mode failure policy: skip an infeasible mode, keep the viable ones --------
# (stub parallel_configs_for so feasibility is controlled, not perf-DB-dependent)

_AGG_CFG = ReplicaParallelConfig(
    ParallelShape(tp=1, dp=1, moe_tp=1, moe_ep=1), replicas=1
)


def test_infeasible_mode_is_skipped_with_warning(monkeypatch):
    # disagg infeasible (raises), agg viable -> only the agg branch survives, with a warning.
    def fake_pcf(
        model,
        hw,
        *,
        gpu_budget,
        deployment_mode,
        backend,
        min_gpu_budget=None,
        max_seq_len=None,
    ):
        if deployment_mode == "disagg":
            raise NoViableParallelConfig("disagg doesn't fit the budget")
        return [_AGG_CFG]

    monkeypatch.setattr(
        "dynamo.profiler.spica.search_space.parallel_configs_for", fake_pcf
    )
    cfg = _config(deployment_mode=["agg", "disagg"], backend=["trtllm"], gpu_budget=8)
    with pytest.warns(UserWarning, match="disagg.* skipped"):
        branches = enumerate_branches(cfg)
    assert [b.deployment_mode for b in branches] == ["agg"]  # disagg dropped, agg kept
    assert branches[0].supported_backends[_AGG_CFG] == frozenset({"trtllm"})


@pytest.mark.filterwarnings(
    "ignore:smart-sweep.*deployment_mode=.* skipped.*:UserWarning"
)
def test_all_modes_infeasible_raises(monkeypatch):
    def _always_raise(*args, **kwargs):
        raise NoViableParallelConfig("nothing fits")

    monkeypatch.setattr(
        "dynamo.profiler.spica.search_space.parallel_configs_for", _always_raise
    )
    cfg = _config(deployment_mode=["agg", "disagg"], backend=["trtllm"], gpu_budget=1)
    with pytest.raises(NoViableParallelConfig, match="no deployment_mode"):
        enumerate_branches(cfg)


def test_backend_without_perf_db_is_dropped_from_knob(monkeypatch):
    # Multi-backend search: vllm has no perf DB for this mode (raises NoPerfDatabase),
    # trtllm is viable. vllm must be dropped from the backend knob and the parallel
    # config tagged only with trtllm.
    def fake_pcf(
        model,
        hw,
        *,
        gpu_budget,
        deployment_mode,
        backend,
        min_gpu_budget=None,
        max_seq_len=None,
    ):
        if backend == "vllm":
            raise NoPerfDatabase("no vllm perf DB for this mode")
        return [_AGG_CFG]

    monkeypatch.setattr(
        "dynamo.profiler.spica.search_space.parallel_configs_for", fake_pcf
    )
    cfg = _config(deployment_mode=["agg"], backend=["vllm", "trtllm"], gpu_budget=8)
    (branch,) = enumerate_branches(cfg)
    assert branch.knob_choices["backend"] == ["trtllm"]  # vllm dropped (no perf DB)
    assert branch.supported_backends[_AGG_CFG] == frozenset(
        {"trtllm"}
    )  # only trtllm supports it


def test_viable_backend_knob_preserves_user_order(monkeypatch):
    monkeypatch.setattr(
        "dynamo.profiler.spica.search_space.parallel_configs_for",
        lambda *args, **kwargs: [_AGG_CFG],
    )
    cfg = _config(deployment_mode=["agg"], backend=["trtllm", "vllm"], gpu_budget=8)

    (branch,) = enumerate_branches(cfg)

    assert branch.knob_choices["backend"] == ["trtllm", "vllm"]


def test_kv_load_range_becomes_continuous_branch_dimension(monkeypatch):
    monkeypatch.setattr(
        "dynamo.profiler.spica.search_space.parallel_configs_for",
        lambda *args, **kwargs: [_AGG_CFG],
    )
    cfg = SmartSearchConfig(
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

    (branch,) = enumerate_branches(cfg)

    assert branch.float_ranges == {"kv_load_ratio": (0.0, 1.0)}
    assert "kv_load_ratio" not in branch.knob_choices


def test_scalar_kv_load_is_pinned_in_branch_selection(monkeypatch):
    monkeypatch.setattr(
        "dynamo.profiler.spica.search_space.parallel_configs_for",
        lambda *args, **kwargs: [_AGG_CFG],
    )
    cfg = SmartSearchConfig(
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

    (branch,) = enumerate_branches(cfg)

    assert branch.float_ranges == {}
    assert branch.knob_choices["kv_load_ratio"] == [0.6]


def test_partial_illegal_pinned_config_raises(monkeypatch):
    # Two pinned configs, only one of which any backend can run -> the pin is wrong, so
    # enumerate_branches fails fast (NoViableParallelConfig). _AGG_CFG is the tp=1 shape.
    def fake_pcf(
        model,
        hw,
        *,
        gpu_budget,
        deployment_mode,
        backend,
        min_gpu_budget=None,
        max_seq_len=None,
    ):
        return [_AGG_CFG]  # only the tp=1 config is ever legal; the pinned tp=2 is not

    monkeypatch.setattr(
        "dynamo.profiler.spica.search_space.parallel_configs_for", fake_pcf
    )
    cfg = _config(
        deployment_mode=["agg"],
        backend=["trtllm"],
        gpu_budget=8,
        parallel_configs=[{"tp": 1, "replicas": 1}, {"tp": 2, "replicas": 1}],
    )
    with pytest.raises(
        NoViableParallelConfig, match="legal/KV-feasible for no configured backend"
    ):
        enumerate_branches(cfg)
