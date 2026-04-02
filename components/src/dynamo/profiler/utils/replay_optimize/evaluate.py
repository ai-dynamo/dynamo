# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay evaluation helpers for the budget-focused dense search heuristic.

The search in `search.py` assumes we prefer to consume the available GPU budget
and therefore ranks visited states by raw output throughput, subject to replay
constraints, rather than by throughput normalized per GPU.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import Executor
from dataclasses import asdict
from pathlib import Path
from typing import Any

from dynamo.llm import KvRouterConfig, MockEngineArgs
from dynamo.replay import run_synthetic_trace_replay, run_trace_replay

from .engine_args import (
    _build_agg_candidate_engine_args,
    _build_candidate_engine_args,
    _build_router_config,
)
from .models import (
    DenseAggReplayState,
    DenseReplayState,
    SyntheticReplayWorkload,
    TraceReplayWorkload,
)
from .scoring import _violation_penalty


def _run_replay_for_state(
    *,
    state: DenseReplayState,
    workload: SyntheticReplayWorkload | TraceReplayWorkload,
    prefill_engine_args: MockEngineArgs,
    decode_engine_args: MockEngineArgs,
    router_config: KvRouterConfig | None,
) -> dict[str, Any]:
    if isinstance(workload, SyntheticReplayWorkload):
        return run_synthetic_trace_replay(
            workload.isl,
            workload.osl,
            workload.request_count,
            prefill_engine_args=prefill_engine_args,
            decode_engine_args=decode_engine_args,
            router_config=router_config,
            num_prefill_workers=state.prefill_workers,
            num_decode_workers=state.decode_workers,
            replay_concurrency=workload.replay_concurrency,
            replay_mode="offline",
            router_mode=state.router_mode,
            arrival_interval_ms=workload.arrival_interval_ms,
            turns_per_session=workload.turns_per_session,
            shared_prefix_ratio=workload.shared_prefix_ratio,
            num_prefix_groups=workload.num_prefix_groups,
            inter_turn_delay_ms=workload.inter_turn_delay_ms,
        )

    return run_trace_replay(
        Path(workload.trace_file),
        prefill_engine_args=prefill_engine_args,
        decode_engine_args=decode_engine_args,
        router_config=router_config,
        num_prefill_workers=state.prefill_workers,
        num_decode_workers=state.decode_workers,
        replay_mode="offline",
        router_mode=state.router_mode,
        arrival_speedup_ratio=workload.arrival_speedup_ratio,
    )


def _run_agg_replay_for_state(
    *,
    state: DenseAggReplayState,
    workload: SyntheticReplayWorkload | TraceReplayWorkload,
    engine_args: MockEngineArgs,
    router_config: KvRouterConfig | None,
) -> dict[str, Any]:
    if isinstance(workload, SyntheticReplayWorkload):
        return run_synthetic_trace_replay(
            workload.isl,
            workload.osl,
            workload.request_count,
            extra_engine_args=engine_args,
            router_config=router_config,
            num_workers=state.workers,
            replay_concurrency=workload.replay_concurrency,
            replay_mode="offline",
            router_mode=state.router_mode,
            arrival_interval_ms=workload.arrival_interval_ms,
            turns_per_session=workload.turns_per_session,
            shared_prefix_ratio=workload.shared_prefix_ratio,
            num_prefix_groups=workload.num_prefix_groups,
            inter_turn_delay_ms=workload.inter_turn_delay_ms,
        )

    return run_trace_replay(
        Path(workload.trace_file),
        extra_engine_args=engine_args,
        router_config=router_config,
        num_workers=state.workers,
        replay_mode="offline",
        router_mode=state.router_mode,
        arrival_speedup_ratio=workload.arrival_speedup_ratio,
    )


def _evaluate_state(
    *,
    state: DenseReplayState,
    workload: SyntheticReplayWorkload | TraceReplayWorkload,
    base_prefill_engine_args: MockEngineArgs,
    base_decode_engine_args: MockEngineArgs,
    base_router_config: KvRouterConfig | None,
    model: str,
    backend: str,
    system: str,
    constraints: Mapping[str, float],
    cache: dict[DenseReplayState, dict[str, Any]],
) -> dict[str, Any]:
    cached = cache.get(state)
    if cached is not None:
        return cached

    prefill_args = _build_candidate_engine_args(
        base_args=base_prefill_engine_args,
        tp_size=state.prefill_tp,
        worker_type="prefill",
        backend=backend,
        system=system,
        model=model,
    )
    decode_args = _build_candidate_engine_args(
        base_args=base_decode_engine_args,
        tp_size=state.decode_tp,
        worker_type="decode",
        backend=backend,
        system=system,
        model=model,
    )
    router_config = None
    if state.router_mode == "kv_router":
        router_config = _build_router_config(
            base_router_config, state.overlap_score_weight
        )
    report = _run_replay_for_state(
        state=state,
        workload=workload,
        prefill_engine_args=prefill_args,
        decode_engine_args=decode_args,
        router_config=router_config,
    )

    total_gpus_used = state.total_gpus_used
    throughput = float(report["output_throughput_tok_s"])
    score = throughput
    penalty = _violation_penalty(report, constraints, total_gpus_used)
    feasible = penalty == 0.0
    record = {
        **report,
        **asdict(state),
        "total_gpus_used": total_gpus_used,
        "output_throughput_tok_s": throughput,
        "score": score,
        "feasible": feasible,
        "violation_penalty": penalty,
    }
    cache[state] = record
    return record


def _evaluate_agg_state(
    *,
    state: DenseAggReplayState,
    workload: SyntheticReplayWorkload | TraceReplayWorkload,
    base_engine_args: MockEngineArgs,
    base_router_config: KvRouterConfig | None,
    model: str,
    backend: str,
    system: str,
    constraints: Mapping[str, float],
    cache: dict[DenseAggReplayState, dict[str, Any]],
) -> dict[str, Any]:
    cached = cache.get(state)
    if cached is not None:
        return cached

    engine_args = _build_agg_candidate_engine_args(
        base_args=base_engine_args,
        tp_size=state.tp,
        backend=backend,
        system=system,
        model=model,
    )
    router_config = None
    if state.router_mode == "kv_router":
        router_config = _build_router_config(
            base_router_config, state.overlap_score_weight
        )
    report = _run_agg_replay_for_state(
        state=state,
        workload=workload,
        engine_args=engine_args,
        router_config=router_config,
    )

    total_gpus_used = state.total_gpus_used
    throughput = float(report["output_throughput_tok_s"])
    score = throughput
    penalty = _violation_penalty(report, constraints, total_gpus_used)
    feasible = penalty == 0.0
    record = {
        **report,
        **asdict(state),
        "total_gpus_used": total_gpus_used,
        "output_throughput_tok_s": throughput,
        "score": score,
        "feasible": feasible,
        "violation_penalty": penalty,
    }
    cache[state] = record
    return record


def _evaluate_state_from_json_payloads(payload: Mapping[str, Any]) -> dict[str, Any]:
    return _evaluate_state(
        state=payload["state"],
        workload=payload["workload"],
        base_prefill_engine_args=MockEngineArgs.from_json(
            payload["base_prefill_engine_args_json"]
        ),
        base_decode_engine_args=MockEngineArgs.from_json(
            payload["base_decode_engine_args_json"]
        ),
        base_router_config=(
            KvRouterConfig.from_json(payload["base_router_config_json"])
            if payload["base_router_config_json"] is not None
            else None
        ),
        model=payload["model"],
        backend=payload["backend"],
        system=payload["system"],
        constraints=payload["constraints"],
        cache={},
    )


def _evaluate_agg_state_from_json_payloads(
    payload: Mapping[str, Any]
) -> dict[str, Any]:
    return _evaluate_agg_state(
        state=payload["state"],
        workload=payload["workload"],
        base_engine_args=MockEngineArgs.from_json(payload["base_engine_args_json"]),
        base_router_config=(
            KvRouterConfig.from_json(payload["base_router_config_json"])
            if payload["base_router_config_json"] is not None
            else None
        ),
        model=payload["model"],
        backend=payload["backend"],
        system=payload["system"],
        constraints=payload["constraints"],
        cache={},
    )


def _evaluate_states(
    *,
    states: Sequence[DenseReplayState],
    workload: SyntheticReplayWorkload | TraceReplayWorkload,
    base_prefill_engine_args: MockEngineArgs,
    base_decode_engine_args: MockEngineArgs,
    base_router_config: KvRouterConfig | None,
    model: str,
    backend: str,
    system: str,
    constraints: Mapping[str, float],
    cache: dict[DenseReplayState, dict[str, Any]],
    max_parallel_evals: int,
    executor: Executor | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any] | None] = [None] * len(states)
    uncached_indices: list[int] = []
    uncached_states: list[DenseReplayState] = []

    for index, state in enumerate(states):
        cached = cache.get(state)
        if cached is not None:
            records[index] = cached
            continue
        uncached_indices.append(index)
        uncached_states.append(state)

    if not uncached_states:
        return [record for record in records if record is not None]

    if max_parallel_evals <= 1 or len(uncached_states) == 1 or executor is None:
        for index, state in zip(uncached_indices, uncached_states, strict=True):
            records[index] = _evaluate_state(
                state=state,
                workload=workload,
                base_prefill_engine_args=base_prefill_engine_args,
                base_decode_engine_args=base_decode_engine_args,
                base_router_config=base_router_config,
                model=model,
                backend=backend,
                system=system,
                constraints=constraints,
                cache=cache,
            )
        return [record for record in records if record is not None]

    base_prefill_engine_args_json = base_prefill_engine_args.dump_json()
    base_decode_engine_args_json = base_decode_engine_args.dump_json()
    base_router_config_json = (
        None if base_router_config is None else base_router_config.dump_json()
    )
    payloads = [
        {
            "state": state,
            "workload": workload,
            "base_prefill_engine_args_json": base_prefill_engine_args_json,
            "base_decode_engine_args_json": base_decode_engine_args_json,
            "base_router_config_json": base_router_config_json,
            "model": model,
            "backend": backend,
            "system": system,
            "constraints": constraints,
        }
        for state in uncached_states
    ]

    future_records = list(executor.map(_evaluate_state_from_json_payloads, payloads))

    for index, state, record in zip(
        uncached_indices,
        uncached_states,
        future_records,
        strict=True,
    ):
        cache[state] = record
        records[index] = record

    return [record for record in records if record is not None]


def _evaluate_agg_states(
    *,
    states: Sequence[DenseAggReplayState],
    workload: SyntheticReplayWorkload | TraceReplayWorkload,
    base_engine_args: MockEngineArgs,
    base_router_config: KvRouterConfig | None,
    model: str,
    backend: str,
    system: str,
    constraints: Mapping[str, float],
    cache: dict[DenseAggReplayState, dict[str, Any]],
    max_parallel_evals: int,
    executor: Executor | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any] | None] = [None] * len(states)
    uncached_indices: list[int] = []
    uncached_states: list[DenseAggReplayState] = []

    for index, state in enumerate(states):
        cached = cache.get(state)
        if cached is not None:
            records[index] = cached
            continue
        uncached_indices.append(index)
        uncached_states.append(state)

    if not uncached_states:
        return [record for record in records if record is not None]

    if max_parallel_evals <= 1 or len(uncached_states) == 1 or executor is None:
        for index, state in zip(uncached_indices, uncached_states, strict=True):
            records[index] = _evaluate_agg_state(
                state=state,
                workload=workload,
                base_engine_args=base_engine_args,
                base_router_config=base_router_config,
                model=model,
                backend=backend,
                system=system,
                constraints=constraints,
                cache=cache,
            )
        return [record for record in records if record is not None]

    base_engine_args_json = base_engine_args.dump_json()
    base_router_config_json = (
        None if base_router_config is None else base_router_config.dump_json()
    )
    payloads = [
        {
            "state": state,
            "workload": workload,
            "base_engine_args_json": base_engine_args_json,
            "base_router_config_json": base_router_config_json,
            "model": model,
            "backend": backend,
            "system": system,
            "constraints": constraints,
        }
        for state in uncached_states
    ]

    future_records = list(
        executor.map(_evaluate_agg_state_from_json_payloads, payloads)
    )

    for index, state, record in zip(
        uncached_indices,
        uncached_states,
        future_records,
        strict=True,
    ):
        cache[state] = record
        records[index] = record

    return [record for record in records if record is not None]
