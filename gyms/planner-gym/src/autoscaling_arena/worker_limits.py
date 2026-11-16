# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared feasible fleet targets, independent of any planner implementation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorkerLimits:
    min_total: int
    max_total: int

    def __post_init__(self) -> None:
        if any(type(n) is not int or n < 1 for n in (self.min_total, self.max_total)):
            raise ValueError("worker limits must be positive integers")
        if self.min_total > self.max_total:
            raise ValueError("worker_limits.min_total must be <= max_total")

    def targets(
        self,
        *,
        topology: str,
        gpu_budget: int,
        prefill_gpus: int,
        decode_gpus: int,
        min_prefill: int = 1,
        max_prefill: int | None = None,
        min_decode: int = 1,
        max_decode: int | None = None,
    ) -> tuple[tuple[int, int], ...]:
        """Return (prefill, decode) pairs allowed by all worker/GPU limits."""
        if topology not in {"agg", "disagg"}:
            raise ValueError("worker limits require agg or disagg topology")
        if decode_gpus < 1 or (topology == "disagg" and prefill_gpus < 1):
            raise ValueError("active pools must have positive GPU costs")
        p_max = min(self.max_total, max_prefill or self.max_total)
        d_max = min(self.max_total, max_decode or self.max_total)
        prefills = range(min_prefill, p_max + 1) if topology == "disagg" else (0,)
        return tuple(
            (p, d)
            for p in prefills
            for d in range(min_decode, d_max + 1)
            if self.min_total <= p + d <= self.max_total
            and p * prefill_gpus + d * decode_gpus <= gpu_budget
        )


def resident_workers(snapshot, pool):
    """Count every allocated worker, including starts and drains."""
    states = [
        set(snapshot[state + "_" + pool + "_ids"])
        for state in ("active", "starting", "draining")
    ]
    if any(a & b for i, a in enumerate(states) for b in states[i + 1 :]):
        raise ValueError("Overlapping worker lifecycle states")
    return len(set().union(*states))
