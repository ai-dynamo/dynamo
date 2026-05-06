# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Heuristic replay search over dense aggregated and disaggregated configs.

This module intentionally assumes the optimizer should try to consume as much of
`max_total_gpus` as possible once a TP family is under consideration.
Accordingly, the search prunes to near-budget-edge states instead of treating
throughput-per-GPU as the primary objective.

The descent dimensions are:
- Disaggregated replay:
  1. TP shape: `(prefill_tp, decode_tp)` probed at equal worker counts that fit
     the budget.
  2. Worker split: `(prefill_workers, decode_workers)` probed only among states
     that maximize GPU usage for the current TP shape.
  3. Router settings: `(router_mode, overlap_score_weight)`.
- Aggregated replay:
  1. TP size: `tp` probed at the maximum worker count that fits the budget.
  2. Worker count: `workers` for the incumbent `tp`.
  3. Router settings: `(router_mode, overlap_score_weight)`.

This is a budget-focused heuristic, not an exact optimizer over all feasible
replay states.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import Literal

from . import aic, evaluate
from .constants import DEFAULT_SEARCH_ROUNDS
from .models import DenseAggReplayState, DenseReplayOptimizationResult, DenseReplayState
from .scoring import _finalize_result, _pick_best_record
from .specs import ReplayOptimizeSpec, RouterMode


def _router_states(
    *,
    router_mode: RouterMode,
    overlap_score_weights: Sequence[float],
) -> list[tuple[str, float]]:
    if router_mode is RouterMode.ROUND_ROBIN:
        return [("round_robin", 0.0)]
    if router_mode is RouterMode.KV_ROUTER:
        return [("kv_router", float(weight)) for weight in overlap_score_weights]
    return [("round_robin", 0.0)] + [
        ("kv_router", float(weight)) for weight in overlap_score_weights
    ]


def _supports_agg_router_mode(*, workers: int, router_mode: str) -> bool:
    return router_mode == "round_robin" or workers > 1


def _iter_budget_edge_worker_states(
    *,
    prefill_tp: int,
    decode_tp: int,
    router_mode: Literal["kv_router", "round_robin"],
    overlap_score_weight: float,
    max_total_gpus: int,
) -> list[DenseReplayState]:
    states: list[DenseReplayState] = []
    max_gpus_used = 0
    for prefill_workers in range(1, max_total_gpus // prefill_tp + 1):
        for decode_workers in range(1, max_total_gpus // decode_tp + 1):
            total_gpus_used = prefill_tp * prefill_workers + decode_tp * decode_workers
            if total_gpus_used > max_total_gpus:
                continue
            state = DenseReplayState(
                prefill_tp=prefill_tp,
                decode_tp=decode_tp,
                prefill_workers=prefill_workers,
                decode_workers=decode_workers,
                overlap_score_weight=overlap_score_weight,
                router_mode=router_mode,
            )
            if total_gpus_used > max_gpus_used:
                max_gpus_used = total_gpus_used
                states = [state]
                continue
            if total_gpus_used == max_gpus_used:
                states.append(state)
    return states


def _iter_agg_worker_states(
    *,
    tp: int,
    router_mode: Literal["kv_router", "round_robin"],
    overlap_score_weight: float,
    max_total_gpus: int,
) -> list[DenseAggReplayState]:
    return [
        DenseAggReplayState(
            tp=tp,
            workers=workers,
            router_mode=router_mode,
            overlap_score_weight=overlap_score_weight,
        )
        for workers in range(1, max_total_gpus // tp + 1)
        if _supports_agg_router_mode(workers=workers, router_mode=router_mode)
    ]


def _iter_tp_states_with_equal_workers(
    *,
    prefill_tps: Sequence[int],
    decode_tps: Sequence[int],
    router_mode: Literal["kv_router", "round_robin"],
    overlap_score_weight: float,
    max_total_gpus: int,
) -> list[DenseReplayState]:
    states: list[DenseReplayState] = []
    for prefill_tp in prefill_tps:
        for decode_tp in decode_tps:
            max_equal_workers = max_total_gpus // (prefill_tp + decode_tp)
            if max_equal_workers < 1:
                continue
            states.append(
                DenseReplayState(
                    prefill_tp=prefill_tp,
                    decode_tp=decode_tp,
                    prefill_workers=max_equal_workers,
                    decode_workers=max_equal_workers,
                    overlap_score_weight=overlap_score_weight,
                    router_mode=router_mode,
                )
            )
    return states


def _iter_agg_tp_states_with_max_workers(
    *,
    tps: Sequence[int],
    router_mode: Literal["kv_router", "round_robin"],
    overlap_score_weight: float,
    max_total_gpus: int,
) -> list[DenseAggReplayState]:
    states: list[DenseAggReplayState] = []
    for tp in tps:
        workers = max_total_gpus // tp
        if workers < 1:
            continue
        states.append(
            DenseAggReplayState(
                tp=tp,
                workers=workers,
                router_mode=router_mode,
                overlap_score_weight=overlap_score_weight,
            )
        )
    return states


def _select_initial_state(
    *,
    prefill_tps: Sequence[int],
    decode_tps: Sequence[int],
    overlap_score_weight: float,
    max_total_gpus: int,
) -> DenseReplayState:
    initial_states = _iter_tp_states_with_equal_workers(
        prefill_tps=prefill_tps,
        decode_tps=decode_tps,
        router_mode="round_robin",
        overlap_score_weight=overlap_score_weight,
        max_total_gpus=max_total_gpus,
    )
    if initial_states:
        return initial_states[0]

    raise ValueError(
        "no TP candidates fit within "
        f"max_total_gpus={max_total_gpus} with equal prefill and decode workers"
    )


def _select_initial_agg_state(
    *,
    tps: Sequence[int],
    max_total_gpus: int,
) -> DenseAggReplayState:
    states = _iter_agg_tp_states_with_max_workers(
        tps=tps,
        router_mode="round_robin",
        overlap_score_weight=0.0,
        max_total_gpus=max_total_gpus,
    )
    if states:
        return states[0]

    raise ValueError(
        "no TP candidates fit within "
        f"max_total_gpus={max_total_gpus} for aggregated replay"
    )


def _record_to_state(record: Mapping[str, float | int]) -> DenseReplayState:
    return DenseReplayState(
        prefill_tp=int(record["prefill_tp"]),
        decode_tp=int(record["decode_tp"]),
        prefill_workers=int(record["prefill_workers"]),
        decode_workers=int(record["decode_workers"]),
        overlap_score_weight=float(record["overlap_score_weight"]),
        router_mode=str(record.get("router_mode", "kv_router")),
    )


def _record_to_agg_state(
    record: Mapping[str, float | int | str]
) -> DenseAggReplayState:
    return DenseAggReplayState(
        tp=int(record["tp"]),
        workers=int(record["workers"]),
        router_mode=str(record["router_mode"]),
        overlap_score_weight=float(record["overlap_score_weight"]),
    )


def optimize_dense_disagg_with_replay(
    spec: ReplayOptimizeSpec,
) -> DenseReplayOptimizationResult:
    """Run a heuristic block search over dense disaggregated offline replay configs.

    Assumes the optimizer should consume as much of `spec.hardware.totalGpus` as
    possible, then ranks visited states by `spec.objective` subject to the SLA
    bounds in `spec.sla`. Supported objectives: `ReplayObjective.THROUGHPUT`
    (default, maximize `output_throughput_tok_s`),
    `ReplayObjective.MEAN_E2E_LATENCY` and `ReplayObjective.MEAN_TTFT` (minimize
    the corresponding report metric). The descended dimensions are:
    1. `(prefill_tp, decode_tp)` at equal worker counts that fit the budget.
    2. `(prefill_workers, decode_workers)` on the budget edge for the incumbent TP
       shape.
    3. `(router_mode, overlap_score_weight)`.

    Returned "best" records are best among visited states, not a global optimum.
    """
    # Guardrail #8: disagg path requires both prefill and decode engine args.
    if (
        spec.engine.basePrefillEngineArgs is None
        or spec.engine.baseDecodeEngineArgs is None
    ):
        raise ValueError(
            "optimize_dense_disagg_with_replay requires both "
            "EngineSpec.basePrefillEngineArgs and EngineSpec.baseDecodeEngineArgs"
        )
    if spec.hardware.totalGpus < 2:
        raise ValueError(
            "hardware.totalGpus must be at least 2 for disaggregated replay"
        )

    backend = spec.engine.backend.value
    system = str(spec.hardware.gpuSku)
    overlap_weights = spec.router.effectiveOverlapWeights
    max_parallel_evals = max(1, int(spec.maxParallelEvals))
    prefill_tps, decode_tps = aic._enumerate_dense_tp_candidates(backend, system)
    if not prefill_tps or not decode_tps:
        raise ValueError(
            f"no dense TP candidates found for backend={backend!r}, system={system!r}"
        )

    cache: dict[DenseReplayState, dict[str, float | int | bool | str]] = {}
    incumbent = _select_initial_state(
        prefill_tps=prefill_tps,
        decode_tps=decode_tps,
        overlap_score_weight=overlap_weights[0],
        max_total_gpus=spec.hardware.totalGpus,
    )

    executor = (
        ProcessPoolExecutor(max_workers=max_parallel_evals)
        if max_parallel_evals > 1
        else None
    )
    try:
        for _ in range(DEFAULT_SEARCH_ROUNDS):
            round_start = incumbent

            tp_states = _iter_tp_states_with_equal_workers(
                prefill_tps=prefill_tps,
                decode_tps=decode_tps,
                router_mode=incumbent.router_mode,
                overlap_score_weight=incumbent.overlap_score_weight,
                max_total_gpus=spec.hardware.totalGpus,
            )
            tp_records = evaluate._evaluate_states(
                states=tp_states,
                spec=spec,
                cache=cache,
                max_parallel_evals=max_parallel_evals,
                executor=executor,
            )
            incumbent = _record_to_state(_pick_best_record(tp_records))

            worker_states = _iter_budget_edge_worker_states(
                prefill_tp=incumbent.prefill_tp,
                decode_tp=incumbent.decode_tp,
                router_mode=incumbent.router_mode,
                overlap_score_weight=incumbent.overlap_score_weight,
                max_total_gpus=spec.hardware.totalGpus,
            )
            worker_records = evaluate._evaluate_states(
                states=worker_states,
                spec=spec,
                cache=cache,
                max_parallel_evals=max_parallel_evals,
                executor=executor,
            )
            incumbent = _record_to_state(_pick_best_record(worker_records))

            router_records = evaluate._evaluate_states(
                states=[
                    DenseReplayState(
                        prefill_tp=incumbent.prefill_tp,
                        decode_tp=incumbent.decode_tp,
                        prefill_workers=incumbent.prefill_workers,
                        decode_workers=incumbent.decode_workers,
                        overlap_score_weight=weight,
                        router_mode=mode,
                    )
                    for mode, weight in _router_states(
                        router_mode=spec.router.mode,
                        overlap_score_weights=overlap_weights,
                    )
                ],
                spec=spec,
                cache=cache,
                max_parallel_evals=max_parallel_evals,
                executor=executor,
            )
            incumbent = _record_to_state(_pick_best_record(router_records))

            if incumbent == round_start:
                break
    finally:
        if executor is not None:
            executor.shutdown()

    return _finalize_result(cache)


def optimize_dense_agg_with_replay(
    spec: ReplayOptimizeSpec,
) -> DenseReplayOptimizationResult:
    """Run a heuristic block search over dense aggregated offline replay configs.

    Assumes the optimizer should consume as much of `spec.hardware.totalGpus` as
    possible, then ranks visited states by `spec.objective` subject to the SLA
    bounds in `spec.sla`. The descended dimensions are:
    1. `tp` at the maximum worker count that fits the budget.
    2. `workers` for the incumbent `tp`.
    3. `(router_mode, overlap_score_weight)`.

    Returned "best" records are best among visited states, not a global optimum.
    """
    # Guardrail #8: agg path requires the single baseEngineArgs.
    if spec.engine.baseEngineArgs is None:
        raise ValueError(
            "optimize_dense_agg_with_replay requires EngineSpec.baseEngineArgs"
        )
    if spec.hardware.totalGpus < 1:
        raise ValueError("hardware.totalGpus must be at least 1 for aggregated replay")

    backend = spec.engine.backend.value
    system = str(spec.hardware.gpuSku)
    overlap_weights = spec.router.effectiveOverlapWeights
    max_parallel_evals = max(1, int(spec.maxParallelEvals))
    tps, _ = aic._enumerate_dense_tp_candidates(backend, system)
    if not tps:
        raise ValueError(
            f"no dense TP candidates found for backend={backend!r}, system={system!r}"
        )

    cache: dict[DenseAggReplayState, dict[str, float | int | bool | str]] = {}
    incumbent = _select_initial_agg_state(
        tps=tps, max_total_gpus=spec.hardware.totalGpus
    )

    executor = (
        ProcessPoolExecutor(max_workers=max_parallel_evals)
        if max_parallel_evals > 1
        else None
    )
    try:
        for _ in range(DEFAULT_SEARCH_ROUNDS):
            round_start = incumbent

            tp_states = _iter_agg_tp_states_with_max_workers(
                tps=tps,
                router_mode=incumbent.router_mode,
                overlap_score_weight=incumbent.overlap_score_weight,
                max_total_gpus=spec.hardware.totalGpus,
            )
            tp_records = evaluate._evaluate_agg_states(
                states=tp_states,
                spec=spec,
                cache=cache,
                max_parallel_evals=max_parallel_evals,
                executor=executor,
            )
            incumbent = _record_to_agg_state(_pick_best_record(tp_records))

            worker_states = _iter_agg_worker_states(
                tp=incumbent.tp,
                router_mode=incumbent.router_mode,
                overlap_score_weight=incumbent.overlap_score_weight,
                max_total_gpus=spec.hardware.totalGpus,
            )
            worker_records = evaluate._evaluate_agg_states(
                states=worker_states,
                spec=spec,
                cache=cache,
                max_parallel_evals=max_parallel_evals,
                executor=executor,
            )
            incumbent = _record_to_agg_state(_pick_best_record(worker_records))

            router_records = evaluate._evaluate_agg_states(
                states=[
                    DenseAggReplayState(
                        tp=incumbent.tp,
                        workers=incumbent.workers,
                        router_mode=mode,
                        overlap_score_weight=weight,
                    )
                    for mode, weight in _router_states(
                        router_mode=spec.router.mode,
                        overlap_score_weights=overlap_weights,
                    )
                    if _supports_agg_router_mode(
                        workers=incumbent.workers,
                        router_mode=mode,
                    )
                ],
                spec=spec,
                cache=cache,
                max_parallel_evals=max_parallel_evals,
                executor=executor,
            )
            if router_records:
                incumbent = _record_to_agg_state(_pick_best_record(router_records))

            if incumbent == round_start:
                break
    finally:
        if executor is not None:
            executor.shutdown()

    return _finalize_result(cache)
