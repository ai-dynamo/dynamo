# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Paired-replicate noise floor and the contract's "differs beyond noise" rule.

Noise comes only from CRN replicates (:mod:`learned_routing.replicates`): replicate ``k`` of a
cell is one perturbed workload plus policy seed ``k + 1``, shared by the policy and its reference.
The paired ratio ``r[c][k] = goodput(policy, c, k) / goodput(reference, c, k)`` is the unit of
noise. Identical re-runs carry no information about noise, so a zero pooled spread is reported
as degenerate instead of being used as a floor.

Rule (contract "Noise, determinism, gates", as amended by setup fix F1):

1. some cell has ``|mean_k r[c][k] - 1| > k_sd * pooled_sd``, where ``pooled_sd`` is the root mean
   of the per-cell sample variances of ``r`` over replicates; and
2. over independent workload segments (cells that share a trace segment are averaged first),
   ``|mean Δ| > k_se * SE``, with at least ``min_segments`` segments.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

Key = tuple[str, int]  # (cell_id, replicate index k)


@dataclass(frozen=True)
class NoiseVerdict:
    cells: int
    replicates_per_cell: int
    pooled_sd: float
    max_abs_cell_delta: float
    cell_clause: bool
    segments: int
    mean_segment_delta: float
    segment_se: float
    segment_clause: bool
    degenerate: bool

    @property
    def differs(self) -> bool:
        return self.cell_clause and self.segment_clause and not self.degenerate


def paired_ratios(
    policy: Mapping[Key, float], reference: Mapping[Key, float]
) -> dict[str, list[float]]:
    """Per-cell paired ratios over replicates; both sides must cover the same (cell, k) set."""
    if set(policy) != set(reference):
        missing = sorted(set(policy) ^ set(reference))[:5]
        raise ValueError(f"policy and reference replicate sets differ, e.g. {missing}")
    by_cell: dict[str, dict[int, float]] = {}
    for (cell, k), goodput in policy.items():
        base = reference[(cell, k)]
        if not base > 0:
            raise ValueError(
                f"reference goodput must be > 0 for {(cell, k)}, got {base}"
            )
        by_cell.setdefault(cell, {})[k] = goodput / base
    return {
        cell: [ratios[k] for k in sorted(ratios)] for cell, ratios in by_cell.items()
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _sample_var(values: list[float]) -> float:
    mean = _mean(values)
    return sum((v - mean) ** 2 for v in values) / (len(values) - 1)


def pooled_sd(ratios_by_cell: Mapping[str, list[float]]) -> float:
    """Root mean of per-cell sample variances; every cell needs >= 2 replicates."""
    if not ratios_by_cell:
        raise ValueError("no cells")
    short = [cell for cell, ratios in ratios_by_cell.items() if len(ratios) < 2]
    if short:
        raise ValueError(f"cells need >= 2 replicates to estimate noise: {short[:5]}")
    return math.sqrt(_mean([_sample_var(r) for r in ratios_by_cell.values()]))


def differs_beyond_noise(
    ratios_by_cell: Mapping[str, list[float]],
    segment_of: Mapping[str, str],
    *,
    k_sd: float = 3.0,
    k_se: float = 2.0,
    min_segments: int = 3,
) -> NoiseVerdict:
    sd = pooled_sd(ratios_by_cell)
    deltas = {cell: _mean(ratios) - 1.0 for cell, ratios in ratios_by_cell.items()}
    max_abs = max(abs(d) for d in deltas.values())

    by_segment: dict[str, list[float]] = {}
    for cell, delta in deltas.items():
        by_segment.setdefault(segment_of[cell], []).append(delta)
    segment_deltas = [_mean(values) for values in by_segment.values()]
    mean_delta = _mean(segment_deltas)
    if len(segment_deltas) >= 2:
        se = math.sqrt(_sample_var(segment_deltas) / len(segment_deltas))
    else:
        se = math.inf

    degenerate = sd == 0.0
    return NoiseVerdict(
        cells=len(deltas),
        replicates_per_cell=min(len(r) for r in ratios_by_cell.values()),
        pooled_sd=sd,
        max_abs_cell_delta=max_abs,
        cell_clause=(not degenerate) and max_abs > k_sd * sd,
        segments=len(segment_deltas),
        mean_segment_delta=mean_delta,
        segment_se=se,
        segment_clause=len(segment_deltas) >= min_segments
        and abs(mean_delta) > k_se * se,
        degenerate=degenerate,
    )
