# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest

try:
    from dynamo.llm import KvRouterConfig, MockEngineArgs
except ImportError:
    pytest.skip("dynamo.llm bindings not available", allow_module_level=True)
from dynamo.profiler.utils import replay_optimize
from dynamo.profiler.utils.replay_optimize import (
    DenseAggReplayState,
    DenseReplayState,
    EngineSpec,
    HardwareSpec,
    ReplayObjective,
    ReplayOptimizeSpec,
    RouterSpec,
    SLASpec,
    WorkloadSpec,
    compare_agg_and_disagg_with_replay,
    optimize_dense_agg_with_replay,
    optimize_dense_disagg_with_replay,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.parallel,
]

_AIC_MODEL = "Qwen/Qwen3-32B"
_AIC_SYSTEM = "h200_sxm"


def _base_prefill_args() -> MockEngineArgs:
    return MockEngineArgs(
        engine_type="vllm",
        num_gpu_blocks=128,
        block_size=64,
        max_num_seqs=16,
        max_num_batched_tokens=4096,
        enable_prefix_caching=True,
        enable_chunked_prefill=False,
        worker_type="prefill",
    )


def _base_decode_args() -> MockEngineArgs:
    return MockEngineArgs(
        engine_type="vllm",
        num_gpu_blocks=192,
        block_size=64,
        max_num_seqs=32,
        max_num_batched_tokens=4096,
        enable_prefix_caching=True,
        enable_chunked_prefill=False,
        worker_type="decode",
    )


def _base_agg_args() -> MockEngineArgs:
    return MockEngineArgs(
        engine_type="vllm",
        num_gpu_blocks=160,
        block_size=64,
        max_num_seqs=24,
        max_num_batched_tokens=4096,
        enable_prefix_caching=True,
        enable_chunked_prefill=False,
        worker_type="aggregated",
    )


def _write_trace(tmp_path: Path) -> Path:
    trace_path = tmp_path / "optimizer_trace.jsonl"
    records = [
        {
            "timestamp": 1000.0,
            "input_length": 32,
            "output_length": 8,
            "hash_ids": [1, 2, 3, 4],
        },
        {
            "timestamp": 1001.0,
            "input_length": 48,
            "output_length": 6,
            "hash_ids": [1, 2, 3, 5],
        },
    ]
    trace_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    return trace_path


def _synthetic_workload(
    *,
    isl: int = 64,
    osl: int = 32,
    request_count: int = 8,
    concurrency: float = 4,
    **extras: Any,
) -> WorkloadSpec:
    return WorkloadSpec(
        isl=isl,
        osl=osl,
        requestCount=request_count,
        concurrency=concurrency,
        **extras,
    )


def _trace_workload(trace_file: Path, speedup: float = 100.0) -> WorkloadSpec:
    return WorkloadSpec(traceFile=str(trace_file), arrivalSpeedupRatio=speedup)


def _applied_compute_agentic_trace_workload(
    trace_file: Path,
    *,
    concurrency: int = 8,
    shared_prefix_ratio: float = 0.5,
    num_prefix_groups: int = 1,
) -> WorkloadSpec:
    return WorkloadSpec(
        traceFile=str(trace_file),
        traceFormat="applied_compute_agentic",
        traceReplayConcurrency=concurrency,
        traceSharedPrefixRatio=shared_prefix_ratio,
        traceNumPrefixGroups=num_prefix_groups,
    )


def _sla(**bounds: float) -> SLASpec:
    """Build an SLASpec from keyword bounds.

    Accepts the legacy `mean_e2e_latency_ms`, `mean_ttft_ms`, `mean_tpot_ms`,
    etc. names for test readability; maps to camelCase DGDR field names.
    """
    translate = {
        "mean_ttft_ms": "ttft",
        "mean_tpot_ms": "itl",
        "mean_e2e_latency_ms": "e2eLatency",
        "p95_ttft_ms": "p95Ttft",
        "p95_tpot_ms": "p95Itl",
        "p95_e2e_latency_ms": "p95E2eLatency",
    }
    return SLASpec(**{translate.get(k, k): v for k, v in bounds.items()})


def test_applied_compute_agentic_trace_workload_requires_trace_replay_concurrency(
    tmp_path,
) -> None:
    trace_path = _write_trace(tmp_path)
    with pytest.raises(ValueError, match="traceReplayConcurrency"):
        WorkloadSpec(
            traceFile=str(trace_path),
            traceFormat="applied_compute_agentic",
        )


def test_applied_compute_agentic_trace_workload_rejects_shared_prefix_ratio_without_groups(
    tmp_path,
) -> None:
    trace_path = _write_trace(tmp_path)
    with pytest.raises(ValueError, match="traceNumPrefixGroups"):
        WorkloadSpec(
            traceFile=str(trace_path),
            traceFormat="applied_compute_agentic",
            traceReplayConcurrency=8,
            traceSharedPrefixRatio=0.5,
            traceNumPrefixGroups=0,
        )


def test_run_replay_for_state_passes_applied_compute_agentic_trace_knobs(
    tmp_path, monkeypatch
) -> None:
    trace_path = _write_trace(tmp_path)
    workload = _applied_compute_agentic_trace_workload(trace_path, concurrency=9)
    captured: dict[str, Any] = {}

    def fake_run_trace_replay(trace_file, **kwargs):
        captured["trace_file"] = trace_file
        captured["kwargs"] = kwargs
        return {"output_throughput_tok_s": 1.0}

    monkeypatch.setattr(
        "dynamo.profiler.utils.replay_optimize.evaluate.run_trace_replay",
        fake_run_trace_replay,
    )

    replay_optimize.evaluate._run_replay_for_state(
        state=DenseReplayState(1, 1, 1, 1, 0.5),
        workload=workload,
        prefill_engine_args=_base_prefill_args(),
        decode_engine_args=_base_decode_args(),
        router_config=KvRouterConfig(),
    )

    assert Path(captured["trace_file"]) == trace_path
    assert captured["kwargs"]["replay_concurrency"] == 9
    assert captured["kwargs"]["trace_format"] == "applied_compute_agentic"
    assert captured["kwargs"]["trace_shared_prefix_ratio"] == 0.5
    assert captured["kwargs"]["trace_num_prefix_groups"] == 1


def _disagg_spec(
    *,
    workload: WorkloadSpec | None = None,
    total_gpus: int = 4,
    sla: SLASpec | None = None,
    objective: ReplayObjective = ReplayObjective.THROUGHPUT,
    router_mode: str = "kv_router",
    overlap_weights: list[float] | None = None,
    base_router_config: KvRouterConfig | None = None,
    max_parallel_evals: int = 1,
) -> ReplayOptimizeSpec:
    return ReplayOptimizeSpec(
        engine=EngineSpec(
            model=_AIC_MODEL,
            backend="vllm",
            basePrefillEngineArgs=_base_prefill_args(),
            baseDecodeEngineArgs=_base_decode_args(),
        ),
        hardware=HardwareSpec(gpuSku=_AIC_SYSTEM, totalGpus=total_gpus),
        workload=workload if workload is not None else _synthetic_workload(),
        sla=sla if sla is not None else SLASpec(),
        router=RouterSpec(
            mode=router_mode,
            overlapWeights=overlap_weights,
            baseRouterConfig=base_router_config,
        ),
        objective=objective,
        maxParallelEvals=max_parallel_evals,
    )


def _agg_spec(
    *,
    workload: WorkloadSpec | None = None,
    total_gpus: int = 4,
    sla: SLASpec | None = None,
    router_mode: str = "kv_router",
    overlap_weights: list[float] | None = None,
    base_router_config: KvRouterConfig | None = None,
    max_parallel_evals: int = 1,
) -> ReplayOptimizeSpec:
    return ReplayOptimizeSpec(
        engine=EngineSpec(
            model=_AIC_MODEL,
            backend="vllm",
            baseEngineArgs=_base_agg_args(),
        ),
        hardware=HardwareSpec(gpuSku=_AIC_SYSTEM, totalGpus=total_gpus),
        workload=workload if workload is not None else _synthetic_workload(),
        sla=sla if sla is not None else SLASpec(),
        router=RouterSpec(
            mode=router_mode,
            overlapWeights=overlap_weights,
            baseRouterConfig=base_router_config,
        ),
        maxParallelEvals=max_parallel_evals,
    )


# ---- internal-helper tests (unchanged from Phase 1 interface) ----


def test_enumerate_dense_tp_candidates_filters_to_tp_only(monkeypatch) -> None:
    common = SimpleNamespace(BackendName=SimpleNamespace(vllm="vllm"))
    task = SimpleNamespace(
        build_disagg_parallel_lists=lambda **_: (
            {
                "num_gpu_per_worker": [1, 2, 4],
                "tp_list": [1, 2, 4],
                "pp_list": [1],
                "dp_list": [1],
                "moe_tp_list": [1],
                "moe_ep_list": [1],
            },
            {
                "num_gpu_per_worker": [1, 2, 4],
                "tp_list": [1, 2, 4],
                "pp_list": [1],
                "dp_list": [1],
                "moe_tp_list": [1],
                "moe_ep_list": [1],
            },
        )
    )
    utils = SimpleNamespace(
        enumerate_parallel_config=lambda **_: [
            [1, 1, 1, 1, 1],
            [2, 1, 1, 1, 1],
            [2, 2, 1, 1, 1],
            [4, 1, 2, 1, 1],
            [4, 1, 1, 1, 1],
        ]
    )
    monkeypatch.setattr(
        replay_optimize.aic,
        "_load_aiconfigurator_modules",
        lambda: (common, task, utils),
    )

    prefill_tps, decode_tps = replay_optimize._enumerate_dense_tp_candidates(
        "vllm", "h200_sxm"
    )

    assert prefill_tps == [1, 2, 4]
    assert decode_tps == [1, 2, 4]


def test_iter_tp_states_with_equal_workers_respects_gpu_budget() -> None:
    states = replay_optimize._iter_tp_states_with_equal_workers(
        prefill_tps=[1, 2, 4, 8],
        decode_tps=[1, 2, 4, 8],
        router_mode="round_robin",
        overlap_score_weight=1.0,
        max_total_gpus=8,
    )

    states_by_tp = {
        (state.prefill_tp, state.decode_tp): (
            state.prefill_workers,
            state.decode_workers,
        )
        for state in states
    }

    assert (8, 8) not in states_by_tp
    assert states_by_tp[(1, 1)] == (4, 4)
    assert states_by_tp[(2, 1)] == (2, 2)
    assert states_by_tp[(4, 4)] == (1, 1)
    assert all(state.total_gpus_used <= 8 for state in states)


def test_iter_agg_tp_states_with_max_workers_respects_gpu_budget() -> None:
    states = replay_optimize._iter_agg_tp_states_with_max_workers(
        tps=[1, 2, 4, 8],
        router_mode="round_robin",
        overlap_score_weight=0.0,
        max_total_gpus=8,
    )

    states_by_tp = {state.tp: state.workers for state in states}

    assert states_by_tp == {1: 8, 2: 4, 4: 2, 8: 1}
    assert all(state.total_gpus_used <= 8 for state in states)
    assert set(state.router_mode for state in states) == {"round_robin"}


def test_mock_engine_args_dump_json_round_trips_explicit_none_fields() -> None:
    base_args = MockEngineArgs(
        engine_type="vllm",
        num_gpu_blocks=128,
        block_size=64,
        max_num_seqs=None,
        max_num_batched_tokens=None,
        enable_prefix_caching=True,
        worker_type="decode",
    )

    restored = MockEngineArgs.from_json(base_args.dump_json())

    assert restored.worker_type == "decode"
    assert restored.max_num_seqs is None
    assert restored.max_num_batched_tokens is None


def test_iter_agg_worker_states_collapses_round_robin_overlap() -> None:
    states = replay_optimize._iter_agg_worker_states(
        tp=2,
        router_mode="round_robin",
        overlap_score_weight=0.0,
        max_total_gpus=8,
    )

    assert [(state.tp, state.workers) for state in states] == [
        (2, 1),
        (2, 2),
        (2, 3),
        (2, 4),
    ]
    assert set(state.router_mode for state in states) == {"round_robin"}
    assert set(state.overlap_score_weight for state in states) == {0.0}


# ---- public-API tests (reshaped to ReplayOptimizeSpec) ----


def test_optimizer_finds_coordinate_optimum_and_reuses_cache(monkeypatch) -> None:
    call_counter: Counter = Counter()
    target_state = replay_optimize.DenseReplayState(2, 4, 2, 1, 2.0)

    def fake_run(**kwargs):
        state = kwargs["state"]
        call_counter[state] += 1
        desired_score = (
            1000.0
            - 100.0 * abs(state.prefill_tp - target_state.prefill_tp)
            - 100.0 * abs(state.decode_tp - target_state.decode_tp)
            - 50.0 * abs(state.prefill_workers - target_state.prefill_workers)
            - 50.0 * abs(state.decode_workers - target_state.decode_workers)
            - 10.0 * abs(state.overlap_score_weight - target_state.overlap_score_weight)
        )
        return {
            "output_throughput_tok_s": desired_score,
            "mean_ttft_ms": 100.0,
            "p95_ttft_ms": 120.0,
            "mean_tpot_ms": 10.0,
            "p95_tpot_ms": 12.0,
            "mean_e2e_latency_ms": 200.0,
            "p95_e2e_latency_ms": 220.0,
        }

    monkeypatch.setattr(
        replay_optimize.aic,
        "_enumerate_dense_tp_candidates",
        lambda backend, system: ([1, 2, 4], [1, 2, 4]),
    )
    monkeypatch.setattr(replay_optimize.evaluate, "_run_replay_for_state", fake_run)

    result = optimize_dense_disagg_with_replay(
        _disagg_spec(
            total_gpus=8,
            sla=_sla(mean_e2e_latency_ms=500.0),
            overlap_weights=[0.0, 1.0, 2.0],
        )
    )

    assert result.best_feasible is not None
    assert result.best_feasible["prefill_tp"] == 2
    assert result.best_feasible["decode_tp"] == 4
    assert result.best_feasible["prefill_workers"] == 2
    assert result.best_feasible["decode_workers"] == 1
    assert result.best_feasible["overlap_score_weight"] == 2.0
    assert sum(call_counter.values()) == len(call_counter)
    assert len(call_counter) == len(result.evaluated_df)


def test_agg_optimizer_finds_coordinate_optimum_and_reuses_cache(monkeypatch) -> None:
    call_counter: Counter = Counter()
    target_state = DenseAggReplayState(2, 3, "kv_router", 2.0)

    def fake_run(**kwargs):
        state = kwargs["state"]
        call_counter[state] += 1
        desired_score = (
            1000.0
            - 100.0 * abs(state.tp - target_state.tp)
            - 50.0 * abs(state.workers - target_state.workers)
            - 100.0 * (state.router_mode != target_state.router_mode)
            - 10.0 * abs(state.overlap_score_weight - target_state.overlap_score_weight)
        )
        return {
            "output_throughput_tok_s": desired_score,
            "mean_ttft_ms": 100.0,
            "p95_ttft_ms": 120.0,
            "mean_tpot_ms": 10.0,
            "p95_tpot_ms": 12.0,
            "mean_e2e_latency_ms": 200.0,
            "p95_e2e_latency_ms": 220.0,
        }

    monkeypatch.setattr(
        replay_optimize.aic,
        "_enumerate_dense_tp_candidates",
        lambda backend, system: ([1, 2, 4], [1, 2, 4]),
    )
    monkeypatch.setattr(replay_optimize.evaluate, "_run_agg_replay_for_state", fake_run)

    result = optimize_dense_agg_with_replay(
        _agg_spec(
            total_gpus=8,
            sla=_sla(mean_e2e_latency_ms=500.0),
            router_mode="both",
            overlap_weights=[0.0, 1.0, 2.0],
        )
    )

    assert result.best_feasible is not None
    assert result.best_feasible["tp"] == 2
    assert result.best_feasible["workers"] == 3
    assert result.best_feasible["router_mode"] == "kv_router"
    assert result.best_feasible["overlap_score_weight"] == 2.0
    assert sum(call_counter.values()) == len(call_counter)
    assert len(call_counter) == len(result.evaluated_df)


def test_optimizer_uses_violation_penalty_when_no_state_is_feasible(
    monkeypatch,
) -> None:
    target_state = replay_optimize.DenseReplayState(1, 2, 2, 2, 1.0)

    def fake_run(**kwargs):
        state = kwargs["state"]
        latency = (
            60.0
            + 10.0 * abs(state.prefill_tp - target_state.prefill_tp)
            + 10.0 * abs(state.decode_tp - target_state.decode_tp)
            + 5.0 * abs(state.prefill_workers - target_state.prefill_workers)
            + 5.0 * abs(state.decode_workers - target_state.decode_workers)
            + abs(state.overlap_score_weight - target_state.overlap_score_weight)
        )
        return {
            "output_throughput_tok_s": 1000.0,
            "mean_ttft_ms": latency,
            "p95_ttft_ms": latency,
            "mean_tpot_ms": 10.0,
            "p95_tpot_ms": 10.0,
            "mean_e2e_latency_ms": latency,
            "p95_e2e_latency_ms": latency,
        }

    monkeypatch.setattr(
        replay_optimize.aic,
        "_enumerate_dense_tp_candidates",
        lambda backend, system: ([1, 2], [1, 2]),
    )
    monkeypatch.setattr(replay_optimize.evaluate, "_run_replay_for_state", fake_run)

    result = optimize_dense_disagg_with_replay(
        _disagg_spec(
            total_gpus=6,
            sla=_sla(mean_e2e_latency_ms=50.0),
            overlap_weights=[0.0, 1.0],
        )
    )

    assert result.best_feasible is None
    assert result.best_infeasible is not None
    assert result.best_infeasible["prefill_tp"] == 1
    assert result.best_infeasible["decode_tp"] == 2
    assert result.best_infeasible["prefill_workers"] == 2
    assert result.best_infeasible["decode_workers"] == 2
    assert result.best_infeasible["overlap_score_weight"] == 1.0


def test_agg_optimizer_uses_violation_penalty_when_no_state_is_feasible(
    monkeypatch,
) -> None:
    target_state = DenseAggReplayState(2, 3, "kv_router", 1.0)

    def fake_run(**kwargs):
        state = kwargs["state"]
        latency = (
            60.0
            + 10.0 * abs(state.tp - target_state.tp)
            + 5.0 * abs(state.workers - target_state.workers)
            + 3.0 * (state.router_mode != target_state.router_mode)
            + abs(state.overlap_score_weight - target_state.overlap_score_weight)
        )
        return {
            "output_throughput_tok_s": 1000.0,
            "mean_ttft_ms": latency,
            "p95_ttft_ms": latency,
            "mean_tpot_ms": 10.0,
            "p95_tpot_ms": 10.0,
            "mean_e2e_latency_ms": latency,
            "p95_e2e_latency_ms": latency,
        }

    monkeypatch.setattr(
        replay_optimize.aic,
        "_enumerate_dense_tp_candidates",
        lambda backend, system: ([1, 2], [1, 2]),
    )
    monkeypatch.setattr(replay_optimize.evaluate, "_run_agg_replay_for_state", fake_run)

    result = optimize_dense_agg_with_replay(
        _agg_spec(
            total_gpus=8,
            sla=_sla(mean_e2e_latency_ms=50.0),
            router_mode="both",
            overlap_weights=[0.0, 1.0],
        )
    )

    assert result.best_feasible is None
    assert result.best_infeasible is not None
    assert result.best_infeasible["tp"] == 2
    assert result.best_infeasible["workers"] == 3
    assert result.best_infeasible["router_mode"] == "kv_router"
    assert result.best_infeasible["overlap_score_weight"] == 1.0


def test_optimizer_supports_round_robin_router_mode(monkeypatch) -> None:
    seen_router_modes: list[str] = []
    seen_weights: list[float] = []

    def fake_run(**kwargs):
        seen_router_modes.append(kwargs["state"].router_mode)
        seen_weights.append(kwargs["state"].overlap_score_weight)
        return {
            "output_throughput_tok_s": 1000.0,
            "mean_ttft_ms": 100.0,
            "p95_ttft_ms": 120.0,
            "mean_tpot_ms": 10.0,
            "p95_tpot_ms": 12.0,
            "mean_e2e_latency_ms": 200.0,
            "p95_e2e_latency_ms": 220.0,
        }

    monkeypatch.setattr(
        replay_optimize.aic,
        "_enumerate_dense_tp_candidates",
        lambda backend, system: ([1, 2], [1, 2]),
    )
    monkeypatch.setattr(replay_optimize.evaluate, "_run_replay_for_state", fake_run)

    result = optimize_dense_disagg_with_replay(
        _disagg_spec(
            total_gpus=4,
            sla=_sla(mean_e2e_latency_ms=500.0),
            router_mode="round_robin",
            overlap_weights=[0.0, 1.0, 2.0],
        )
    )

    assert result.best_feasible is not None
    assert set(seen_router_modes) == {"round_robin"}
    # Guardrail #5: round_robin auto-collapses overlap weights to (0.0,)
    assert set(seen_weights) == {0.0}


def test_disagg_optimizer_supports_latency_objective(monkeypatch) -> None:
    def fake_run(**kwargs):
        state = kwargs["state"]
        if state.prefill_tp == 1 and state.decode_tp == 1:
            return {
                "output_throughput_tok_s": 1200.0,
                "mean_ttft_ms": 140.0,
                "p95_ttft_ms": 160.0,
                "mean_tpot_ms": 10.0,
                "p95_tpot_ms": 12.0,
                "mean_e2e_latency_ms": 300.0,
                "p95_e2e_latency_ms": 320.0,
            }
        return {
            "output_throughput_tok_s": 1000.0,
            "mean_ttft_ms": 100.0,
            "p95_ttft_ms": 120.0,
            "mean_tpot_ms": 10.0,
            "p95_tpot_ms": 12.0,
            "mean_e2e_latency_ms": 200.0,
            "p95_e2e_latency_ms": 220.0,
        }

    monkeypatch.setattr(
        replay_optimize.aic,
        "_enumerate_dense_tp_candidates",
        lambda backend, system: ([1, 2], [1, 2]),
    )
    monkeypatch.setattr(replay_optimize.evaluate, "_run_replay_for_state", fake_run)

    result = optimize_dense_disagg_with_replay(
        _disagg_spec(
            total_gpus=4,
            sla=_sla(mean_e2e_latency_ms=500.0),
            objective=ReplayObjective.MEAN_E2E_LATENCY,
            overlap_weights=[0.0],
        )
    )

    assert result.best_feasible is not None
    assert (
        result.best_feasible["prefill_tp"],
        result.best_feasible["decode_tp"],
    ) in {(1, 2), (2, 1), (2, 2)}
    assert result.best_feasible["score"] == -200.0
    assert result.best_feasible["objective"] == "mean_e2e_latency"


def test_disagg_optimizer_rejects_invalid_objective() -> None:
    # Invalid objective strings raise at spec construction (Pydantic validates
    # the enum field) and at direct ReplayObjective construction.
    with pytest.raises(ValueError, match="not a valid ReplayObjective"):
        ReplayObjective("bad_objective")
    with pytest.raises(ValueError):
        ReplayOptimizeSpec(
            engine=EngineSpec(
                model=_AIC_MODEL,
                backend="vllm",
                basePrefillEngineArgs=_base_prefill_args(),
                baseDecodeEngineArgs=_base_decode_args(),
            ),
            hardware=HardwareSpec(gpuSku=_AIC_SYSTEM, totalGpus=4),
            workload=_synthetic_workload(),
            objective="bad_objective",
        )


def test_disagg_optimizer_supports_router_mode_search(monkeypatch) -> None:
    seen_router_modes: list[str] = []
    seen_weights: list[float] = []

    def fake_run(**kwargs):
        state = kwargs["state"]
        seen_router_modes.append(state.router_mode)
        seen_weights.append(state.overlap_score_weight)
        return {
            "output_throughput_tok_s": 1000.0 * state.total_gpus_used,
            "mean_ttft_ms": 100.0,
            "p95_ttft_ms": 120.0,
            "mean_tpot_ms": 10.0,
            "p95_tpot_ms": 12.0,
            "mean_e2e_latency_ms": 200.0,
            "p95_e2e_latency_ms": 220.0,
        }

    monkeypatch.setattr(
        replay_optimize.aic,
        "_enumerate_dense_tp_candidates",
        lambda backend, system: ([1, 2], [1, 2]),
    )
    monkeypatch.setattr(replay_optimize.evaluate, "_run_replay_for_state", fake_run)

    result = optimize_dense_disagg_with_replay(
        _disagg_spec(
            total_gpus=4,
            sla=_sla(mean_e2e_latency_ms=500.0),
            router_mode="both",
            overlap_weights=[0.0, 1.0, 2.0],
        )
    )

    assert result.best_feasible is not None
    assert "round_robin" in seen_router_modes
    assert "kv_router" in seen_router_modes
    assert 0.0 in seen_weights
    assert 1.0 in seen_weights
    assert 2.0 in seen_weights


def test_agg_optimizer_supports_router_mode_search(monkeypatch) -> None:
    seen_router_modes: list[str] = []
    seen_weights: list[float] = []

    def fake_run(**kwargs):
        state = kwargs["state"]
        seen_router_modes.append(state.router_mode)
        seen_weights.append(state.overlap_score_weight)
        return {
            "output_throughput_tok_s": 1000.0 * state.workers,
            "mean_ttft_ms": 100.0,
            "p95_ttft_ms": 120.0,
            "mean_tpot_ms": 10.0,
            "p95_tpot_ms": 12.0,
            "mean_e2e_latency_ms": 200.0,
            "p95_e2e_latency_ms": 220.0,
        }

    monkeypatch.setattr(
        replay_optimize.aic,
        "_enumerate_dense_tp_candidates",
        lambda backend, system: ([1, 2], [1, 2]),
    )
    monkeypatch.setattr(replay_optimize.evaluate, "_run_agg_replay_for_state", fake_run)

    result = optimize_dense_agg_with_replay(
        _agg_spec(
            total_gpus=4,
            sla=_sla(mean_e2e_latency_ms=500.0),
            router_mode="both",
            overlap_weights=[0.0, 1.0, 2.0],
        )
    )

    assert result.best_feasible is not None
    assert "round_robin" in seen_router_modes
    assert "kv_router" in seen_router_modes
    assert 0.0 in seen_weights
    assert 1.0 in seen_weights
    assert 2.0 in seen_weights


def test_compare_agg_and_disagg_with_replay_picks_expected_mode(monkeypatch) -> None:
    agg_result = replay_optimize.DenseReplayOptimizationResult(
        best_feasible={
            "tp": 2,
            "workers": 3,
            "router_mode": "kv_router",
            "overlap_score_weight": 1.0,
            "total_gpus_used": 6,
            "output_throughput_tok_s": 3000.0,
            "score": 500.0,
            "feasible": True,
            "violation_penalty": 0.0,
            "mean_e2e_latency_ms": 100.0,
        },
        best_infeasible=None,
        evaluated_df=pd.DataFrame(),
        feasible_df=pd.DataFrame(),
    )
    disagg_result = replay_optimize.DenseReplayOptimizationResult(
        best_feasible={
            "prefill_tp": 1,
            "decode_tp": 1,
            "prefill_workers": 2,
            "decode_workers": 2,
            "overlap_score_weight": 0.0,
            "total_gpus_used": 4,
            "output_throughput_tok_s": 1200.0,
            "score": 300.0,
            "feasible": True,
            "violation_penalty": 0.0,
            "mean_e2e_latency_ms": 150.0,
        },
        best_infeasible=None,
        evaluated_df=pd.DataFrame(),
        feasible_df=pd.DataFrame(),
    )

    monkeypatch.setattr(
        replay_optimize.bench,
        "optimize_dense_agg_with_replay",
        lambda *_, **__: agg_result,
    )
    monkeypatch.setattr(
        replay_optimize.bench,
        "optimize_dense_disagg_with_replay",
        lambda *_, **__: disagg_result,
    )

    # Build a spec that populates both agg and disagg engine args so the
    # comparison function accepts it (though the patched optimize_* bypass
    # the actual engine-args assertions).
    spec = ReplayOptimizeSpec(
        engine=EngineSpec(
            model=_AIC_MODEL,
            backend="vllm",
            baseEngineArgs=_base_agg_args(),
            basePrefillEngineArgs=_base_prefill_args(),
            baseDecodeEngineArgs=_base_decode_args(),
        ),
        hardware=HardwareSpec(gpuSku=_AIC_SYSTEM, totalGpus=8),
        workload=_synthetic_workload(),
        sla=_sla(mean_e2e_latency_ms=500.0),
    )

    comparison = compare_agg_and_disagg_with_replay(spec)

    assert comparison["chosen_mode"] == "agg"
    assert comparison["chosen_best"] == agg_result.best_feasible


def test_evaluate_state_prefers_normalized_metrics_over_report_payload() -> None:
    state = replay_optimize.DenseReplayState(
        prefill_tp=1,
        decode_tp=1,
        prefill_workers=1,
        decode_workers=1,
        overlap_score_weight=0.0,
        router_mode="round_robin",
    )
    cache: dict[replay_optimize.DenseReplayState, dict[str, Any]] = {}

    spec = ReplayOptimizeSpec(
        engine=EngineSpec(
            model="meta-llama/Llama-3.1-8B-Instruct",
            backend="vllm",
            basePrefillEngineArgs=_base_prefill_args(),
            baseDecodeEngineArgs=_base_decode_args(),
        ),
        hardware=HardwareSpec(gpuSku="h100_sxm", totalGpus=4),
        workload=_synthetic_workload(isl=128, request_count=16),
        sla=_sla(mean_e2e_latency_ms=1000.0),
    )

    with patch(
        "dynamo.profiler.utils.replay_optimize.evaluate._run_replay_for_state",
        return_value={
            "output_throughput_tok_s": "11.0",
            "score": -1.0,
            "feasible": False,
            "violation_penalty": 7.0,
            "mean_e2e_latency_ms": 100.0,
        },
    ):
        record = replay_optimize.evaluate._evaluate_state(
            state=state,
            spec=spec,
            cache=cache,
        )

    assert record["output_throughput_tok_s"] == 11.0
    assert record["score"] == 11.0
    assert record["feasible"] is True
    assert record["violation_penalty"] == 0.0


def test_evaluate_agg_state_prefers_normalized_metrics_over_report_payload() -> None:
    state = DenseAggReplayState(
        tp=2,
        workers=2,
        router_mode="round_robin",
        overlap_score_weight=0.0,
    )
    cache: dict[DenseAggReplayState, dict[str, Any]] = {}

    spec = ReplayOptimizeSpec(
        engine=EngineSpec(
            model="meta-llama/Llama-3.1-8B-Instruct",
            backend="vllm",
            baseEngineArgs=_base_agg_args(),
        ),
        hardware=HardwareSpec(gpuSku="h100_sxm", totalGpus=4),
        workload=_synthetic_workload(isl=128, request_count=16),
        sla=_sla(mean_e2e_latency_ms=1000.0),
    )

    with patch(
        "dynamo.profiler.utils.replay_optimize.evaluate._run_agg_replay_for_state",
        return_value={
            "output_throughput_tok_s": "24.0",
            "score": -1.0,
            "feasible": False,
            "violation_penalty": 9.0,
            "mean_e2e_latency_ms": 200.0,
        },
    ):
        record = replay_optimize.evaluate._evaluate_agg_state(
            state=state,
            spec=spec,
            cache=cache,
        )

    assert record["output_throughput_tok_s"] == 24.0
    assert record["score"] == 24.0
    assert record["feasible"] is True
    assert record["violation_penalty"] == 0.0


def test_kv_router_config_rejects_negative_overlap_weight() -> None:
    config = KvRouterConfig(overlap_score_weight=1.0)

    with pytest.raises(ValueError, match="overlap_score_weight must be non-negative"):
        config.overlap_score_weight = -1.0

    with pytest.raises(ValueError, match="overlap_score_weight must be non-negative"):
        config.with_overrides(overlap_score_weight=-1.0)


@pytest.mark.timeout(30)
def test_agg_optimizer_synthetic_replay_smoke(monkeypatch) -> None:
    pytest.importorskip("aiconfigurator")
    monkeypatch.setattr(
        replay_optimize.aic,
        "_enumerate_dense_tp_candidates",
        lambda backend, system: ([1, 2], [1, 2]),
    )

    result = optimize_dense_agg_with_replay(
        _agg_spec(
            workload=_synthetic_workload(isl=128, request_count=8),
            total_gpus=4,
            sla=_sla(
                mean_ttft_ms=100000.0,
                mean_tpot_ms=100000.0,
                mean_e2e_latency_ms=100000.0,
            ),
            router_mode="both",
            overlap_weights=[0.0, 1.0],
        )
    )

    assert not result.evaluated_df.empty
    assert result.best_feasible is not None


@pytest.mark.timeout(30)
def test_agg_optimizer_timed_trace_smoke(tmp_path, monkeypatch) -> None:
    pytest.importorskip("aiconfigurator")
    monkeypatch.setattr(
        replay_optimize.aic,
        "_enumerate_dense_tp_candidates",
        lambda backend, system: ([1, 2], [1, 2]),
    )

    result = optimize_dense_agg_with_replay(
        _agg_spec(
            workload=_trace_workload(_write_trace(tmp_path)),
            total_gpus=4,
            sla=_sla(
                mean_ttft_ms=100000.0,
                mean_tpot_ms=100000.0,
                mean_e2e_latency_ms=100000.0,
            ),
            router_mode="both",
            overlap_weights=[0.0, 1.0],
        )
    )

    assert not result.evaluated_df.empty
    assert result.best_feasible is not None


@pytest.mark.timeout(30)
def test_optimizer_synthetic_replay_smoke(tmp_path, monkeypatch) -> None:
    pytest.importorskip("aiconfigurator")
    monkeypatch.setattr(
        replay_optimize.aic,
        "_enumerate_dense_tp_candidates",
        lambda backend, system: ([1, 2], [1, 2]),
    )

    result = optimize_dense_disagg_with_replay(
        _disagg_spec(
            workload=_synthetic_workload(isl=128, request_count=8),
            total_gpus=4,
            sla=_sla(
                mean_ttft_ms=100000.0,
                mean_tpot_ms=100000.0,
                mean_e2e_latency_ms=100000.0,
            ),
            overlap_weights=[0.0, 1.0],
        )
    )

    assert not result.evaluated_df.empty
    assert result.best_feasible is not None


@pytest.mark.timeout(30)
def test_optimizer_timed_trace_smoke(tmp_path, monkeypatch) -> None:
    pytest.importorskip("aiconfigurator")
    monkeypatch.setattr(
        replay_optimize.aic,
        "_enumerate_dense_tp_candidates",
        lambda backend, system: ([1, 2], [1, 2]),
    )

    result = optimize_dense_disagg_with_replay(
        _disagg_spec(
            workload=_trace_workload(_write_trace(tmp_path)),
            total_gpus=4,
            sla=_sla(
                mean_ttft_ms=100000.0,
                mean_tpot_ms=100000.0,
                mean_e2e_latency_ms=100000.0,
            ),
            overlap_weights=[0.0, 1.0],
        )
    )

    assert not result.evaluated_df.empty
