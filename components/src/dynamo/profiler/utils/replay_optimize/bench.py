# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd
from aiconfigurator.sdk.task import TaskConfig, TaskRunner

from dynamo.llm import MockEngineArgs

from .models import ReplayConstraints, SyntheticReplayWorkload, TraceReplayWorkload
from .scoring import _pick_best_record
from .search import optimize_dense_agg_with_replay, optimize_dense_disagg_with_replay


def compare_aic_and_replay_disagg(
    *,
    model: str,
    backend: str,
    system: str,
    isl: int,
    osl: int,
    max_total_gpus: int,
    replay_request_count: int,
    replay_concurrency: int,
    base_prefill_engine_args: MockEngineArgs,
    base_decode_engine_args: MockEngineArgs,
    constraints: Mapping[str, float] | None = None,
    max_parallel_evals: int = 1,
) -> dict[str, Any]:
    aic_constraints = ReplayConstraints.from_mapping(constraints, max_total_gpus)
    aic_task = TaskConfig(
        serving_mode="disagg",
        model_path=model,
        system_name=system,
        backend_name=backend,
        total_gpus=max_total_gpus,
        isl=isl,
        osl=osl,
        **aic_constraints.aic_task_kwargs(),
    )
    aic_result = TaskRunner().run(aic_task)
    aic_df = aic_result.get("pareto_df", pd.DataFrame())

    replay_result = optimize_dense_disagg_with_replay(
        model=model,
        backend=backend,
        system=system,
        workload=SyntheticReplayWorkload(
            isl=isl,
            osl=osl,
            request_count=replay_request_count,
            replay_concurrency=replay_concurrency,
        ),
        base_prefill_engine_args=base_prefill_engine_args,
        base_decode_engine_args=base_decode_engine_args,
        max_total_gpus=max_total_gpus,
        constraints=constraints,
        router_mode="round_robin",
        max_parallel_evals=max_parallel_evals,
    )

    aic_best = None
    if not aic_df.empty:
        row = aic_df.iloc[0]
        aic_best = {
            "prefill_tp": int(row.get("(p)tp", 0)),
            "decode_tp": int(row.get("(d)tp", 0)),
            "prefill_workers": int(row.get("(p)workers", 0)),
            "decode_workers": int(row.get("(d)workers", 0)),
            "total_gpus_used": int(row.get("num_total_gpus", 0)),
            "ttft": float(row.get("ttft", 0.0)),
            "tpot": float(row.get("tpot", 0.0)),
            "request_latency": float(row.get("request_latency", 0.0)),
            "tokens_per_s": float(row.get("tokens/s", 0.0)),
            "tokens_per_s_per_gpu": float(row.get("tokens/s/gpu", 0.0)),
        }

    replay_best = None
    if replay_result.best_feasible is not None:
        replay_best_record = replay_result.best_feasible
        replay_best = {
            "prefill_tp": int(replay_best_record["prefill_tp"]),
            "decode_tp": int(replay_best_record["decode_tp"]),
            "prefill_workers": int(replay_best_record["prefill_workers"]),
            "decode_workers": int(replay_best_record["decode_workers"]),
            "total_gpus_used": int(replay_best_record["total_gpus_used"]),
            "mean_ttft_ms": float(replay_best_record.get("mean_ttft_ms", 0.0)),
            "mean_tpot_ms": float(replay_best_record.get("mean_tpot_ms", 0.0)),
            "mean_e2e_latency_ms": float(
                replay_best_record.get("mean_e2e_latency_ms", 0.0)
            ),
            "output_throughput_tok_s": float(
                replay_best_record.get("output_throughput_tok_s", 0.0)
            ),
            "score": float(replay_best_record.get("score", 0.0)),
        }

    return {
        "aic_pareto_df": aic_df,
        "aic_best": aic_best,
        "replay_result": replay_result,
        "replay_best": replay_best,
    }


def compare_agg_and_disagg_with_replay(
    *,
    model: str,
    backend: str,
    system: str,
    workload: SyntheticReplayWorkload | TraceReplayWorkload,
    base_engine_args: MockEngineArgs,
    base_prefill_engine_args: MockEngineArgs,
    base_decode_engine_args: MockEngineArgs,
    max_total_gpus: int,
    constraints: Mapping[str, float] | None = None,
    router_mode: str = "kv_router",
    overlap_score_weights: tuple[float, ...] | list[float] | None = None,
    max_parallel_evals: int = 1,
) -> dict[str, Any]:
    agg_result = optimize_dense_agg_with_replay(
        model=model,
        backend=backend,
        system=system,
        workload=workload,
        base_engine_args=base_engine_args,
        max_total_gpus=max_total_gpus,
        constraints=constraints,
        router_mode=router_mode,
        overlap_score_weights=overlap_score_weights,
        max_parallel_evals=max_parallel_evals,
    )
    disagg_result = optimize_dense_disagg_with_replay(
        model=model,
        backend=backend,
        system=system,
        workload=workload,
        base_prefill_engine_args=base_prefill_engine_args,
        base_decode_engine_args=base_decode_engine_args,
        max_total_gpus=max_total_gpus,
        constraints=constraints,
        router_mode=router_mode,
        overlap_score_weights=overlap_score_weights,
        max_parallel_evals=max_parallel_evals,
    )

    agg_best = agg_result.best_feasible
    disagg_best = disagg_result.best_feasible
    if agg_best is None and disagg_best is None:
        candidates = [
            result.best_infeasible
            for result in (agg_result, disagg_result)
            if result.best_infeasible is not None
        ]
        chosen_best = None if not candidates else _pick_best_record(candidates)
    elif agg_best is None:
        chosen_best = disagg_best
    elif disagg_best is None:
        chosen_best = agg_best
    else:
        chosen_best = _pick_best_record([agg_best, disagg_best])

    chosen_mode = None
    if chosen_best is not None:
        chosen_mode = (
            "agg" if "tp" in chosen_best and "workers" in chosen_best else "disagg"
        )

    return {
        "agg_result": agg_result,
        "disagg_result": disagg_result,
        "chosen_mode": chosen_mode,
        "chosen_best": chosen_best,
    }
