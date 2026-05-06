# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pandas as pd
from aiconfigurator.sdk.task import TaskConfig, TaskRunner

from .scoring import _pick_best_record
from .search import optimize_dense_agg_with_replay, optimize_dense_disagg_with_replay
from .specs import ReplayOptimizeSpec


def compare_aic_and_replay_disagg(
    spec: ReplayOptimizeSpec,
) -> dict[str, Any]:
    """Run AIC pareto + replay optimization side-by-side for a disagg config.

    Uses SLA bounds from `spec.sla` as AIC's latency targets; round_robin router
    mode is forced for the replay run (AIC itself has no router sweep).
    """
    if spec.workload.isTraceBased:
        raise ValueError("compare_aic_and_replay_disagg requires a synthetic workload")
    if spec.workload.requestCount is None or spec.workload.concurrency is None:
        raise ValueError(
            "compare_aic_and_replay_disagg requires synthetic WorkloadSpec with "
            "requestCount and concurrency"
        )

    aic_task = TaskConfig(
        serving_mode="disagg",
        model_path=spec.engine.model,
        system_name=str(spec.hardware.gpuSku),
        backend_name=spec.engine.backend.value,
        total_gpus=spec.hardware.totalGpus,
        isl=spec.workload.isl,
        osl=spec.workload.osl,
        **spec.sla.aic_task_kwargs(),
    )
    aic_result = TaskRunner().run(aic_task)
    aic_df = aic_result.get("pareto_df", pd.DataFrame())

    replay_spec = spec.model_copy(
        update={"router": spec.router.model_copy(update={"mode": "round_robin"})}
    )
    replay_result = optimize_dense_disagg_with_replay(replay_spec)

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
    spec: ReplayOptimizeSpec,
) -> dict[str, Any]:
    """Run both agg and disagg replay optimizations on the same spec and pick the winner.

    The spec must populate `spec.engine.baseEngineArgs` (agg path) plus
    `basePrefillEngineArgs` / `baseDecodeEngineArgs` (disagg path).
    """
    agg_result = optimize_dense_agg_with_replay(spec)
    disagg_result = optimize_dense_disagg_with_replay(spec)

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
