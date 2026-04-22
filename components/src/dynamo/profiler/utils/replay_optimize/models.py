# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, fields
from enum import Enum
from typing import Any

import pandas as pd

from .constants import SUPPORTED_CONSTRAINTS


@dataclass(frozen=True)
class ReplayConstraints:
    mean_ttft_ms: float | None = None
    p95_ttft_ms: float | None = None
    mean_tpot_ms: float | None = None
    p95_tpot_ms: float | None = None
    mean_e2e_latency_ms: float | None = None
    p95_e2e_latency_ms: float | None = None
    max_total_gpus: int | None = None

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, float] | None,
        max_total_gpus: int,
    ) -> ReplayConstraints:
        raw = dict(mapping or {})
        unknown = sorted(set(raw) - SUPPORTED_CONSTRAINTS)
        if unknown:
            raise ValueError(
                "unsupported constraints: "
                + ", ".join(unknown)
                + f"; supported constraints are {sorted(SUPPORTED_CONSTRAINTS)}"
            )

        raw_gpus = raw.get("max_total_gpus")
        if raw_gpus is not None and int(raw_gpus) != max_total_gpus:
            raise ValueError(
                "constraints['max_total_gpus'] must match max_total_gpus when both are provided"
            )

        def _bound(key: str) -> float | None:
            value = raw.get(key)
            return None if value is None or value <= 0 else float(value)

        return cls(
            mean_ttft_ms=_bound("mean_ttft_ms"),
            p95_ttft_ms=_bound("p95_ttft_ms"),
            mean_tpot_ms=_bound("mean_tpot_ms"),
            p95_tpot_ms=_bound("p95_tpot_ms"),
            mean_e2e_latency_ms=_bound("mean_e2e_latency_ms"),
            p95_e2e_latency_ms=_bound("p95_e2e_latency_ms"),
            max_total_gpus=int(max_total_gpus),
        )

    def _active(
        self, report: Mapping[str, Any], total_gpus_used: int
    ) -> Iterator[tuple[str, float | None, float]]:
        for field in fields(self):
            if field.name == "max_total_gpus":
                continue
            bound = getattr(self, field.name)
            if bound is None:
                continue
            value = report.get(field.name)
            yield field.name, None if value is None else float(value), bound
        if self.max_total_gpus is not None:
            yield (
                "max_total_gpus",
                float(total_gpus_used),
                float(self.max_total_gpus),
            )

    def violation_penalty(
        self, report: Mapping[str, Any], total_gpus_used: int
    ) -> float:
        penalty = 0.0
        for _, metric, bound in self._active(report, total_gpus_used):
            if metric is None:
                penalty += math.inf
                continue
            penalty += max(metric / bound - 1.0, 0.0)
        return penalty

    def summarize(self, report: Mapping[str, Any], total_gpus_used: int) -> str:
        statuses: list[str] = []
        for name, metric, bound in self._active(report, total_gpus_used):
            if metric is None:
                statuses.append(f"{name}=missing<={bound:g} unsatisfied")
                continue
            state = "satisfied" if metric <= bound else "unsatisfied"
            statuses.append(f"{name}={metric:.3f}<={bound:g} {state}")
        return "constraints=" + ", ".join(statuses) if statuses else "constraints=none"

    def aic_task_kwargs(self) -> dict[str, float | None]:
        return {
            "ttft": self.mean_ttft_ms,
            "tpot": self.mean_tpot_ms,
            "request_latency": self.mean_e2e_latency_ms,
        }


class ReplayObjective(str, Enum):
    THROUGHPUT = "throughput"
    MEAN_TTFT = "mean_ttft"
    MEAN_E2E_LATENCY = "mean_e2e_latency"

    def score(self, report: Mapping[str, Any]) -> float:
        if self is ReplayObjective.THROUGHPUT:
            return float(report["output_throughput_tok_s"])
        if self is ReplayObjective.MEAN_TTFT:
            return -float(report["mean_ttft_ms"])
        return -float(report["mean_e2e_latency_ms"])


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
