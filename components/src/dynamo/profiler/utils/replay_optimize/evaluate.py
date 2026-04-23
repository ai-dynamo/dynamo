# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay evaluation helpers for the budget-focused dense search heuristic.

The search in `search.py` assumes we prefer to consume the available GPU budget
and therefore ranks visited states by the selected `spec.objective`, subject to
SLA and budget constraints, rather than by throughput normalized per GPU.
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
from .logging import ensure_dynamo_logging, log_state_finish, log_state_start
from .models import DenseAggReplayState, DenseReplayState
from .specs import (
    EngineSpec,
    HardwareSpec,
    ReplayObjective,
    ReplayOptimizeSpec,
    RouterSpec,
    SLASpec,
    WorkloadSpec,
)


def _run_replay_for_state(
    *,
    state: DenseReplayState,
    workload: WorkloadSpec,
    prefill_engine_args: MockEngineArgs,
    decode_engine_args: MockEngineArgs,
    router_config: KvRouterConfig | None,
) -> dict[str, Any]:
    if workload.isTraceBased:
        return run_trace_replay(
            Path(workload.traceFile),
            prefill_engine_args=prefill_engine_args,
            decode_engine_args=decode_engine_args,
            router_config=router_config,
            num_prefill_workers=state.prefill_workers,
            num_decode_workers=state.decode_workers,
            replay_mode="offline",
            router_mode=state.router_mode,
            arrival_speedup_ratio=workload.arrivalSpeedupRatio,
        )

    return run_synthetic_trace_replay(
        workload.isl,
        workload.osl,
        int(workload.requestCount),
        prefill_engine_args=prefill_engine_args,
        decode_engine_args=decode_engine_args,
        router_config=router_config,
        num_prefill_workers=state.prefill_workers,
        num_decode_workers=state.decode_workers,
        replay_concurrency=int(workload.concurrency),
        replay_mode="offline",
        router_mode=state.router_mode,
        arrival_interval_ms=workload.arrivalIntervalMs,
        turns_per_session=workload.turnsPerSession,
        shared_prefix_ratio=workload.sharedPrefixRatio,
        num_prefix_groups=workload.numPrefixGroups,
        inter_turn_delay_ms=workload.interTurnDelayMs,
    )


def _run_agg_replay_for_state(
    *,
    state: DenseAggReplayState,
    workload: WorkloadSpec,
    engine_args: MockEngineArgs,
    router_config: KvRouterConfig | None,
) -> dict[str, Any]:
    if workload.isTraceBased:
        return run_trace_replay(
            Path(workload.traceFile),
            extra_engine_args=engine_args,
            router_config=router_config,
            num_workers=state.workers,
            replay_mode="offline",
            router_mode=state.router_mode,
            arrival_speedup_ratio=workload.arrivalSpeedupRatio,
        )

    return run_synthetic_trace_replay(
        workload.isl,
        workload.osl,
        int(workload.requestCount),
        extra_engine_args=engine_args,
        router_config=router_config,
        num_workers=state.workers,
        replay_concurrency=int(workload.concurrency),
        replay_mode="offline",
        router_mode=state.router_mode,
        arrival_interval_ms=workload.arrivalIntervalMs,
        turns_per_session=workload.turnsPerSession,
        shared_prefix_ratio=workload.sharedPrefixRatio,
        num_prefix_groups=workload.numPrefixGroups,
        inter_turn_delay_ms=workload.interTurnDelayMs,
    )


def _feasibility(
    *,
    report: Mapping[str, Any],
    state: DenseReplayState | DenseAggReplayState,
    sla: SLASpec,
    hardware: HardwareSpec,
) -> tuple[float, bool]:
    """Split SLA violation and hardware-budget gate, combine into one penalty.

    The record's `violation_penalty` field keeps its Phase 1 role (drives
    infeasible-record sorting in `_pick_best_record`); over-budget adds a
    constant 1.0 so it outranks small SLA misses.
    """
    sla_penalty = sla.violation_penalty(report)
    over_budget = state.total_gpus_used > hardware.totalGpus
    penalty = sla_penalty + (1.0 if over_budget else 0.0)
    feasible = sla_penalty == 0.0 and not over_budget
    return penalty, feasible


def _evaluate_state(
    *,
    state: DenseReplayState,
    spec: ReplayOptimizeSpec,
    cache: dict[DenseReplayState, dict[str, Any]],
) -> dict[str, Any]:
    ensure_dynamo_logging()
    cached = cache.get(state)
    if cached is not None:
        return cached

    log_state_start(state)

    backend = spec.engine.backend.value
    system = str(spec.hardware.gpuSku)
    prefill_args = _build_candidate_engine_args(
        base_args=spec.engine.basePrefillEngineArgs,
        tp_size=state.prefill_tp,
        worker_type="prefill",
        backend=backend,
        system=system,
        model=spec.engine.model,
    )
    decode_args = _build_candidate_engine_args(
        base_args=spec.engine.baseDecodeEngineArgs,
        tp_size=state.decode_tp,
        worker_type="decode",
        backend=backend,
        system=system,
        model=spec.engine.model,
    )
    router_config = None
    if state.router_mode == "kv_router":
        router_config = _build_router_config(
            spec.router.baseRouterConfig, state.overlap_score_weight
        )
    report = _run_replay_for_state(
        state=state,
        workload=spec.workload,
        prefill_engine_args=prefill_args,
        decode_engine_args=decode_args,
        router_config=router_config,
    )

    throughput = float(report["output_throughput_tok_s"])
    score = spec.objective.score(report)
    penalty, feasible = _feasibility(
        report=report, state=state, sla=spec.sla, hardware=spec.hardware
    )
    record = {
        **report,
        **asdict(state),
        "total_gpus_used": state.total_gpus_used,
        "output_throughput_tok_s": throughput,
        "score": score,
        "objective": spec.objective.value,
        "feasible": feasible,
        "violation_penalty": penalty,
    }
    log_state_finish(
        state=state,
        report=report,
        sla=spec.sla,
        hardware=spec.hardware,
        score=score,
        feasible=feasible,
        violation_penalty=penalty,
    )
    cache[state] = record
    return record


def _evaluate_agg_state(
    *,
    state: DenseAggReplayState,
    spec: ReplayOptimizeSpec,
    cache: dict[DenseAggReplayState, dict[str, Any]],
) -> dict[str, Any]:
    ensure_dynamo_logging()
    cached = cache.get(state)
    if cached is not None:
        return cached

    log_state_start(state)

    backend = spec.engine.backend.value
    system = str(spec.hardware.gpuSku)
    engine_args = _build_agg_candidate_engine_args(
        base_args=spec.engine.baseEngineArgs,
        tp_size=state.tp,
        backend=backend,
        system=system,
        model=spec.engine.model,
    )
    router_config = None
    if state.router_mode == "kv_router":
        router_config = _build_router_config(
            spec.router.baseRouterConfig, state.overlap_score_weight
        )
    report = _run_agg_replay_for_state(
        state=state,
        workload=spec.workload,
        engine_args=engine_args,
        router_config=router_config,
    )

    throughput = float(report["output_throughput_tok_s"])
    score = spec.objective.score(report)
    penalty, feasible = _feasibility(
        report=report, state=state, sla=spec.sla, hardware=spec.hardware
    )
    record = {
        **report,
        **asdict(state),
        "total_gpus_used": state.total_gpus_used,
        "output_throughput_tok_s": throughput,
        "score": score,
        "objective": spec.objective.value,
        "feasible": feasible,
        "violation_penalty": penalty,
    }
    log_state_finish(
        state=state,
        report=report,
        sla=spec.sla,
        hardware=spec.hardware,
        score=score,
        feasible=feasible,
        violation_penalty=penalty,
    )
    cache[state] = record
    return record


# ---- Cross-process payload bridge ----
# `MockEngineArgs` and `KvRouterConfig` are Rust-bound and don't pickle through
# `ProcessPoolExecutor`; we round-trip them via their own `dump_json()` /
# `from_json()` methods. Everything else on `ReplayOptimizeSpec` is a Pydantic
# model and pickles natively, but we serialize the whole payload dict to keep
# cross-process transport explicit.


def _spec_to_payload(spec: ReplayOptimizeSpec) -> dict[str, Any]:
    engine = spec.engine
    router = spec.router
    return {
        "engine_model": engine.model,
        "engine_backend": engine.backend.value,
        "engine_base_agg_json": (
            engine.baseEngineArgs.dump_json()
            if engine.baseEngineArgs is not None
            else None
        ),
        "engine_base_prefill_json": (
            engine.basePrefillEngineArgs.dump_json()
            if engine.basePrefillEngineArgs is not None
            else None
        ),
        "engine_base_decode_json": (
            engine.baseDecodeEngineArgs.dump_json()
            if engine.baseDecodeEngineArgs is not None
            else None
        ),
        "hardware_json": spec.hardware.model_dump_json(),
        "workload_json": spec.workload.model_dump_json(),
        "sla_json": spec.sla.model_dump_json(),
        "router_mode": router.mode,
        "router_overlap_weights": (
            None if router.overlapWeights is None else list(router.overlapWeights)
        ),
        "router_base_config_json": (
            router.baseRouterConfig.dump_json()
            if router.baseRouterConfig is not None
            else None
        ),
        "objective": spec.objective.value,
        "max_parallel_evals": spec.maxParallelEvals,
    }


def _spec_from_payload(payload: Mapping[str, Any]) -> ReplayOptimizeSpec:
    return ReplayOptimizeSpec(
        engine=EngineSpec(
            model=payload["engine_model"],
            backend=payload["engine_backend"],
            baseEngineArgs=(
                MockEngineArgs.from_json(payload["engine_base_agg_json"])
                if payload["engine_base_agg_json"] is not None
                else None
            ),
            basePrefillEngineArgs=(
                MockEngineArgs.from_json(payload["engine_base_prefill_json"])
                if payload["engine_base_prefill_json"] is not None
                else None
            ),
            baseDecodeEngineArgs=(
                MockEngineArgs.from_json(payload["engine_base_decode_json"])
                if payload["engine_base_decode_json"] is not None
                else None
            ),
        ),
        hardware=HardwareSpec.model_validate_json(payload["hardware_json"]),
        workload=WorkloadSpec.model_validate_json(payload["workload_json"]),
        sla=SLASpec.model_validate_json(payload["sla_json"]),
        router=RouterSpec(
            mode=payload["router_mode"],
            overlapWeights=payload["router_overlap_weights"],
            baseRouterConfig=(
                KvRouterConfig.from_json(payload["router_base_config_json"])
                if payload["router_base_config_json"] is not None
                else None
            ),
        ),
        objective=ReplayObjective(payload["objective"]),
        maxParallelEvals=payload["max_parallel_evals"],
    )


def _evaluate_state_from_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    return _evaluate_state(
        state=payload["state"],
        spec=_spec_from_payload(payload["spec"]),
        cache={},
    )


def _evaluate_agg_state_from_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    return _evaluate_agg_state(
        state=payload["state"],
        spec=_spec_from_payload(payload["spec"]),
        cache={},
    )


def _evaluate_states(
    *,
    states: Sequence[DenseReplayState],
    spec: ReplayOptimizeSpec,
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
            records[index] = _evaluate_state(state=state, spec=spec, cache=cache)
        return [record for record in records if record is not None]

    spec_payload = _spec_to_payload(spec)
    payloads = [{"state": state, "spec": spec_payload} for state in uncached_states]

    future_records = list(executor.map(_evaluate_state_from_payload, payloads))

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
    spec: ReplayOptimizeSpec,
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
            records[index] = _evaluate_agg_state(state=state, spec=spec, cache=cache)
        return [record for record in records if record is not None]

    spec_payload = _spec_to_payload(spec)
    payloads = [{"state": state, "spec": spec_payload} for state in uncached_states]

    future_records = list(executor.map(_evaluate_agg_state_from_payload, payloads))

    for index, state, record in zip(
        uncached_indices,
        uncached_states,
        future_records,
        strict=True,
    ):
        cache[state] = record
        records[index] = record

    return [record for record in records if record is not None]
