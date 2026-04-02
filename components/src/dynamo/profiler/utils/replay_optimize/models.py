# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class SyntheticReplayWorkload:
    isl: int
    osl: int
    request_count: int
    replay_concurrency: int
    arrival_interval_ms: float = 0.0
    turns_per_session: int = 1
    shared_prefix_ratio: float = 0.0
    num_prefix_groups: int = 0
    inter_turn_delay_ms: float = 0.0


@dataclass(frozen=True)
class TraceReplayWorkload:
    trace_file: str | os.PathLike[str]
    arrival_speedup_ratio: float = 1.0


@dataclass(frozen=True)
class DenseReplayState:
    prefill_tp: int
    decode_tp: int
    prefill_workers: int
    decode_workers: int
    overlap_score_weight: float
    router_mode: str = "kv_router"

    @property
    def total_gpus_used(self) -> int:
        return (
            self.prefill_tp * self.prefill_workers
            + self.decode_tp * self.decode_workers
        )


@dataclass(frozen=True)
class DenseAggReplayState:
    tp: int
    workers: int
    router_mode: str
    overlap_score_weight: float

    @property
    def total_gpus_used(self) -> int:
        return self.tp * self.workers


@dataclass(frozen=True)
class DenseReplayOptimizationResult:
    best_feasible: dict[str, Any] | None
    best_infeasible: dict[str, Any] | None
    evaluated_df: pd.DataFrame
    feasible_df: pd.DataFrame
