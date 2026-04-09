# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def _metric_value(report: Mapping[str, Any], key: str, total_gpus_used: int) -> float:
    if key == "max_total_gpus":
        return float(total_gpus_used)
    value = report.get(key)
    if value is None:
        return math.inf
    return float(value)


def _violation_penalty(
    report: Mapping[str, Any],
    constraints: Mapping[str, float],
    total_gpus_used: int,
) -> float:
    penalty = 0.0
    for key, bound in constraints.items():
        if bound <= 0:
            continue
        metric = _metric_value(report, key, total_gpus_used)
        penalty += max(metric / bound - 1.0, 0.0)
    return penalty


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
