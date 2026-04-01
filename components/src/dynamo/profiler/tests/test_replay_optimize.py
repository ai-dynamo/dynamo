# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from dynamo.llm import MockEngineArgs
from dynamo.profiler.utils import replay_optimize
from dynamo.profiler.utils.replay_optimize import (
    SyntheticReplayWorkload,
    TraceReplayWorkload,
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
        total_gpus = state.total_gpus_used
        return {
            "output_throughput_tok_s": desired_score * total_gpus,
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
        model=_AIC_MODEL,
        backend="vllm",
        system=_AIC_SYSTEM,
        workload=SyntheticReplayWorkload(
            isl=64,
            osl=32,
            request_count=8,
            replay_concurrency=4,
        ),
        base_prefill_engine_args=_base_prefill_args(),
        base_decode_engine_args=_base_decode_args(),
        max_total_gpus=8,
        constraints={"mean_e2e_latency_ms": 500.0},
        overlap_score_weights=[0.0, 1.0, 2.0],
        max_parallel_evals=1,
    )

    assert result.best_feasible is not None
    assert result.best_feasible["prefill_tp"] == 2
    assert result.best_feasible["decode_tp"] == 4
    assert result.best_feasible["prefill_workers"] == 2
    assert result.best_feasible["decode_workers"] == 1
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
        model=_AIC_MODEL,
        backend="vllm",
        system=_AIC_SYSTEM,
        workload=SyntheticReplayWorkload(
            isl=64,
            osl=32,
            request_count=8,
            replay_concurrency=4,
        ),
        base_prefill_engine_args=_base_prefill_args(),
        base_decode_engine_args=_base_decode_args(),
        max_total_gpus=6,
        constraints={"mean_e2e_latency_ms": 50.0},
        overlap_score_weights=[0.0, 1.0],
        max_parallel_evals=1,
    )

    assert result.best_feasible is None
    assert result.best_infeasible is not None
    assert result.best_infeasible["prefill_tp"] == 1
    assert result.best_infeasible["decode_tp"] == 2
    assert result.best_infeasible["prefill_workers"] == 2
    assert result.best_infeasible["decode_workers"] == 2
    assert result.best_infeasible["overlap_score_weight"] == 1.0


def test_optimizer_supports_round_robin_router_mode(monkeypatch) -> None:
    seen_router_modes: list[str] = []
    seen_weights: list[float] = []

    def fake_run(**kwargs):
        seen_router_modes.append(kwargs["router_mode"])
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
        model=_AIC_MODEL,
        backend="vllm",
        system=_AIC_SYSTEM,
        workload=SyntheticReplayWorkload(
            isl=64,
            osl=32,
            request_count=8,
            replay_concurrency=4,
        ),
        base_prefill_engine_args=_base_prefill_args(),
        base_decode_engine_args=_base_decode_args(),
        max_total_gpus=4,
        constraints={"mean_e2e_latency_ms": 500.0},
        router_mode="round_robin",
        overlap_score_weights=[0.0, 1.0, 2.0],
        max_parallel_evals=1,
    )

    assert result.best_feasible is not None
    assert set(seen_router_modes) == {"round_robin"}
    assert set(seen_weights) == {0.0}


@pytest.mark.timeout(30)
def test_optimizer_synthetic_replay_smoke(tmp_path, monkeypatch) -> None:
    pytest.importorskip("aiconfigurator")
    monkeypatch.setattr(
        replay_optimize.aic,
        "_enumerate_dense_tp_candidates",
        lambda backend, system: ([1, 2], [1, 2]),
    )

    result = optimize_dense_disagg_with_replay(
        model=_AIC_MODEL,
        backend="vllm",
        system=_AIC_SYSTEM,
        workload=SyntheticReplayWorkload(
            isl=128,
            osl=32,
            request_count=8,
            replay_concurrency=4,
        ),
        base_prefill_engine_args=_base_prefill_args(),
        base_decode_engine_args=_base_decode_args(),
        max_total_gpus=4,
        constraints={
            "mean_ttft_ms": 100000.0,
            "mean_tpot_ms": 100000.0,
            "mean_e2e_latency_ms": 100000.0,
        },
        overlap_score_weights=[0.0, 1.0],
        max_parallel_evals=1,
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
        model=_AIC_MODEL,
        backend="vllm",
        system=_AIC_SYSTEM,
        workload=TraceReplayWorkload(
            trace_file=_write_trace(tmp_path),
            arrival_speedup_ratio=100.0,
        ),
        base_prefill_engine_args=_base_prefill_args(),
        base_decode_engine_args=_base_decode_args(),
        max_total_gpus=4,
        constraints={
            "mean_ttft_ms": 100000.0,
            "mean_tpot_ms": 100000.0,
            "mean_e2e_latency_ms": 100000.0,
        },
        overlap_score_weights=[0.0, 1.0],
        max_parallel_evals=1,
    )

    assert not result.evaluated_df.empty
    assert result.best_feasible is not None
