# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from .models import DenseReplayOptimizationResult


def _rank_record(record: Mapping[str, Any]) -> tuple[float, float, float]:
    return (
        float(record["score"]),
        float(record["output_throughput_tok_s"]),
        -float(record.get("mean_e2e_latency_ms", math.inf)),
    )


def _pick_best_record(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    feasible_records = [record for record in records if record["feasible"]]
    if feasible_records:
        return max(
            feasible_records,
            key=lambda record: (
                *_rank_record(record),
                -float(record["total_gpus_used"]),
            ),
        )

    return min(
        records,
        key=lambda record: (
            float(record["violation_penalty"]),
            -float(record["output_throughput_tok_s"]),
            float(record.get("mean_e2e_latency_ms", math.inf)),
        ),
    )


def _finalize_result(
    cache: Mapping[Any, dict[str, Any]],
) -> DenseReplayOptimizationResult:
    evaluated_df = pd.DataFrame.from_records(list(cache.values()))
    feasible_df = (
        evaluated_df[evaluated_df["feasible"]]
        if not evaluated_df.empty
        else evaluated_df
    )
    if not feasible_df.empty:
        feasible_df = feasible_df.sort_values(
            by=[
                "score",
                "output_throughput_tok_s",
                "mean_e2e_latency_ms",
                "total_gpus_used",
            ],
            ascending=[False, False, True, True],
        ).reset_index(drop=True)
    best_feasible = feasible_df.iloc[0].to_dict() if not feasible_df.empty else None
    best_infeasible = None
    if not evaluated_df.empty:
        infeasible_df = evaluated_df[~evaluated_df["feasible"]]
        if not infeasible_df.empty:
            best_infeasible = (
                infeasible_df.sort_values(
                    by=[
                        "violation_penalty",
                        "output_throughput_tok_s",
                        "mean_e2e_latency_ms",
                    ],
                    ascending=[True, False, True],
                )
                .iloc[0]
                .to_dict()
            )

    return DenseReplayOptimizationResult(
        best_feasible=best_feasible,
        best_infeasible=best_infeasible,
        evaluated_df=evaluated_df.reset_index(drop=True),
        feasible_df=feasible_df,
    )
