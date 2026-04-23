# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optimizer-internal dataclasses for replay_optimize search states.

User-facing configuration lives in `specs.py`. This module carries only the
hot-path state types (used as dict keys in the search cache, instantiated
once per visited candidate) and the result bundle returned from the
public API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


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

    def format_summary(self) -> str:
        return (
            f"prefill_tp={self.prefill_tp} decode_tp={self.decode_tp} "
            f"prefill_workers={self.prefill_workers} decode_workers={self.decode_workers} "
            f"router_mode={self.router_mode} overlap_score_weight={self.overlap_score_weight} "
            f"total_gpus={self.total_gpus_used}"
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

    def format_summary(self) -> str:
        return (
            f"tp={self.tp} workers={self.workers} "
            f"router_mode={self.router_mode} overlap_score_weight={self.overlap_score_weight} "
            f"total_gpus={self.total_gpus_used}"
        )


@dataclass(frozen=True)
class DenseReplayOptimizationResult:
    best_feasible: dict[str, Any] | None
    best_infeasible: dict[str, Any] | None
    evaluated_df: pd.DataFrame
    feasible_df: pd.DataFrame
