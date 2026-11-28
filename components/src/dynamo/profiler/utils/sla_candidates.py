# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collect every simulated configuration that meets the requested SLA.

The rapid search strategy deploys only the rank-1 pick. The Pareto frontiers
and ranked lists that AIC computes along the way hold other configurations
that also meet the SLA and trade per-user speed for per-GPU throughput
differently. This module filters them down to SLA-compliant rows and writes
them to a CSV, so users can review the alternatives and choose one.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Mapping

import pandas as pd

logger = logging.getLogger(__name__)

SLA_CANDIDATES_FILE_NAME = "sla_candidates.csv"

# Columns moved to the front of the CSV, in this order, when present.
_LEADING_COLUMNS = (
    "mode",
    "selected",
    "ttft",
    "tpot",
    "request_latency",
    "tokens/s/user",
    "tokens/s/gpu",
    "tokens/s/gpu_cluster",
    "num_total_gpus",
    "total_gpus_needed",
)


def _meets_sla(
    df: pd.DataFrame,
    target_ttft: float | None,
    target_tpot: float | None,
    target_request_latency: float | None,
) -> pd.Series:
    """Return a boolean mask of rows that meet every SLA target that is set.

    When a request-latency target is set it replaces the TPOT target, which
    matches how the AIC pickers rank configurations.
    """
    mask = pd.Series(True, index=df.index)

    def _bound(column: str, target: float | None) -> None:
        nonlocal mask
        if target is None or target <= 0:
            return
        if column not in df.columns:
            # A target we cannot check must not let rows through silently.
            mask &= False
            return
        values = pd.to_numeric(df[column], errors="coerce")
        mask &= values.le(target).fillna(False)

    _bound("ttft", target_ttft)
    if target_request_latency is not None and target_request_latency > 0:
        _bound("request_latency", target_request_latency)
    else:
        _bound("tpot", target_tpot)
    return mask


def _same_config(row: pd.Series, selected: pd.Series) -> bool:
    """True when ``row`` matches ``selected`` on every column they share."""
    shared = [c for c in selected.index if c in row.index]
    if not shared:
        return False
    for column in shared:
        a, b = row[column], selected[column]
        if pd.isna(a) and pd.isna(b):
            continue
        if isinstance(a, float) or isinstance(b, float):
            try:
                if not math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-9):
                    return False
            except (TypeError, ValueError):
                return False
        elif a != b:
            return False
    return True


def collect_sla_candidates(
    pareto_fronts: Mapping[str, pd.DataFrame | None] | None,
    best_configs: Mapping[str, pd.DataFrame | None] | None,
    target_ttft: float | None,
    target_tpot: float | None,
    target_request_latency: float | None = None,
    chosen_exp: str | None = None,
) -> pd.DataFrame:
    """Merge the per-experiment results into one table of SLA-compliant configs.

    AIC returns two views per experiment or serving mode: the Pareto frontier
    of tokens/s/user against tokens/s/gpu, and the top-N configs it ranked
    under the SLA. The rank-1 config of ``chosen_exp`` is what the profiler
    deploys, and it is not always on the frontier, so both views are merged.

    Args:
        pareto_fronts: Pareto frontier per experiment or serving mode, as
            returned by AIC's ``_execute_tasks``. ``None`` counts as empty.
        best_configs: Ranked top-N configs per experiment or serving mode,
            from the same call. ``None`` counts as empty.
        target_ttft: TTFT target in ms. ``None`` or ``<= 0`` disables it.
        target_tpot: TPOT target in ms. Ignored when a request-latency
            target is set.
        target_request_latency: End-to-end request latency target in ms.
        chosen_exp: Experiment whose rank-1 config became the deployment.

    Returns:
        One row per distinct compliant config, with a ``mode`` column naming
        its experiment and a ``selected`` column that is ``True`` only for
        the config the profiler deployed. The selected config is kept even if
        it misses the SLA, since the profiler deploys it anyway. Rows are
        sorted by ``tokens/s/gpu``, highest first, when that column is present.
    """
    pareto_fronts = pareto_fronts or {}
    best_configs = best_configs or {}
    frames: list[pd.DataFrame] = []
    for exp_name in dict.fromkeys([*best_configs, *pareto_fronts]):
        ranked = best_configs.get(exp_name)
        frontier = pareto_fronts.get(exp_name)
        sources = [df for df in (ranked, frontier) if df is not None and not df.empty]
        if not sources:
            continue

        rows: list[pd.Series] = []
        selected_flags: list[bool] = []
        for source in sources:
            compliant = _meets_sla(
                source, target_ttft, target_tpot, target_request_latency
            ).to_numpy()
            for pos, (_, row) in enumerate(source.iterrows()):
                is_selected = exp_name == chosen_exp and source is ranked and pos == 0
                if not (compliant[pos] or is_selected):
                    continue
                if any(_same_config(row, kept) for kept in rows):
                    continue
                rows.append(row)
                selected_flags.append(is_selected)

        if not rows:
            continue
        merged = pd.DataFrame(rows).reset_index(drop=True)
        merged.insert(0, "mode", exp_name)
        merged.insert(1, "selected", selected_flags)
        frames.append(merged)

    if not frames:
        return pd.DataFrame(columns=["mode", "selected"])

    candidates = pd.concat(frames, ignore_index=True, sort=False)
    if "tokens/s/gpu" in candidates.columns:
        candidates = candidates.sort_values(
            "tokens/s/gpu", ascending=False, kind="stable"
        ).reset_index(drop=True)

    leading = [c for c in _LEADING_COLUMNS if c in candidates.columns]
    rest = [c for c in candidates.columns if c not in leading]
    return candidates[leading + rest]


def write_sla_candidates(candidates: pd.DataFrame, output_dir: str) -> str:
    """Write ``candidates`` to ``sla_candidates.csv`` and return the file name.

    The file is written even when no config meets the SLA, so its absence
    never has to be told apart from "nothing qualified".
    """
    path = os.path.join(output_dir, SLA_CANDIDATES_FILE_NAME)
    candidates.to_csv(path, index=False)
    logger.info(
        "Wrote %d SLA-compliant candidate config(s) to %s", len(candidates), path
    )
    return SLA_CANDIDATES_FILE_NAME
