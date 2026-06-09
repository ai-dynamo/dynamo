# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""AIC-based pre-filter for THOROUGH-mode profiling candidates.

Runs a single AIC simulation (offline, no GPUs) to predict performance metrics
for all parallelism configs, then matches predictions to enumerated candidates
and keeps only the top-N per side.

Best-effort: any AIC failure returns the original candidates unchanged.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import pandas as pd

from dynamo.profiler.utils.aic_dataframe import make_parallel_label

if TYPE_CHECKING:
    from collections.abc import Sequence

try:
    from aiconfigurator.sdk.task import TaskConfig, TaskRunner
except ImportError:
    TaskConfig = None
    TaskRunner = None

logger = logging.getLogger(__name__)


def _candidate_label(candidate) -> str:
    return make_parallel_label(
        candidate.tp,
        candidate.pp,
        candidate.dp,
        getattr(candidate, "moe_tp", 1),
        getattr(candidate, "moe_ep", 1),
    )


def run_aic_simulation(
    model: str,
    system: str,
    backend: str,
    total_gpus: int,
    isl: int,
    osl: int,
) -> pd.DataFrame:
    """Run a single AIC simulation and return the pareto DataFrame.

    The TaskConfig does not take per-candidate parallelism params — AIC
    explores those internally and returns a pareto_df with predicted metrics
    for each config it considers viable.

    Returns an empty DataFrame on failure.
    """
    if TaskRunner is None or TaskConfig is None:
        logger.warning("aiconfigurator not available; returning empty pareto_df.")
        return pd.DataFrame()

    task = TaskConfig(
        serving_mode="disagg",
        model_path=model,
        system_name=system,
        backend_name=backend,
        total_gpus=total_gpus,
        isl=isl,
        osl=osl,
    )
    runner = TaskRunner()
    result = runner.run(task)
    if not isinstance(result, dict):
        return pd.DataFrame()
    pareto_df = result.get("pareto_df")
    if pareto_df is None or not isinstance(pareto_df, pd.DataFrame):
        return pd.DataFrame()
    return pareto_df


def prefilter_prefill_candidates(
    candidates: Sequence,
    top_n: int,
    pareto_df: pd.DataFrame,
) -> list:
    """Keep the top-N prefill candidates ranked by predicted TTFT (ascending).

    Matches the ``parallel`` labels in *pareto_df* (from a prior
    ``run_aic_simulation`` call) to enumerated candidates and sorts by
    predicted TTFT ascending. Candidates with no AIC prediction sort last.

    Returns the original list unchanged when *pareto_df* lacks the needed
    columns.
    """
    if top_n is None or top_n <= 0 or len(candidates) <= top_n:
        return list(candidates)

    try:
        if pareto_df.empty or "ttft" not in pareto_df.columns:
            logger.warning("AIC returned no prefill predictions; skipping pre-filter.")
            return list(candidates)

        label_to_ttft: dict[str, float] = {}
        if "parallel" not in pareto_df.columns:
            logger.warning("AIC pareto_df has no 'parallel' column; skipping pre-filter.")
            return list(candidates)
        for _, row in pareto_df.iterrows():
            label = str(row["parallel"])
            ttft = float(row.get("ttft", float("inf")))
            if not math.isfinite(ttft):
                ttft = float("inf")
            if label not in label_to_ttft or ttft < label_to_ttft[label]:
                label_to_ttft[label] = ttft

        scored = []
        for candidate in candidates:
            label = _candidate_label(candidate)
            ttft = label_to_ttft.get(label, float("inf"))
            scored.append((ttft, candidate))

        scored.sort(key=lambda x: x[0])
        selected = [c for _, c in scored[:top_n]]

        for candidate in selected:
            logger.info(
                "Selected prefill DGD: tp=%d pp=%d dp=%d moe_tp=%d moe_ep=%d",
                candidate.tp, candidate.pp, candidate.dp,
                getattr(candidate, "moe_tp", 1),
                getattr(candidate, "moe_ep", 1),
            )
        return selected

    except Exception:
        logger.warning(
            "AIC pre-filter failed for prefill candidates; using full enumeration.",
            exc_info=True,
        )
        return list(candidates)


def prefilter_decode_candidates(
    candidates: Sequence,
    top_n: int,
    pareto_df: pd.DataFrame,
) -> list:
    """Keep the top-N decode candidates ranked by predicted tokens/s/gpu (descending).

    Matches the ``parallel`` labels in *pareto_df* (from a prior
    ``run_aic_simulation`` call) to enumerated candidates and sorts by
    predicted throughput descending. Candidates with no AIC prediction sort
    last.

    Returns the original list unchanged when *pareto_df* lacks the needed
    columns.
    """
    if top_n is None or top_n <= 0 or len(candidates) <= top_n:
        return list(candidates)

    try:
        if pareto_df.empty:
            logger.warning("AIC returned no decode predictions; skipping pre-filter.")
            return list(candidates)

        thpt_col = None
        for col_name in ("tokens/s/gpu", "seq/s/gpu"):
            if col_name in pareto_df.columns:
                thpt_col = col_name
                break

        if thpt_col is None:
            logger.warning(
                "AIC pareto_df has no throughput column; skipping decode pre-filter."
            )
            return list(candidates)

        label_to_thpt: dict[str, float] = {}
        if "parallel" not in pareto_df.columns:
            logger.warning("AIC pareto_df has no 'parallel' column; skipping pre-filter.")
            return list(candidates)
        for _, row in pareto_df.iterrows():
            label = str(row["parallel"])
            thpt = float(row.get(thpt_col, 0.0))
            if not math.isfinite(thpt):
                thpt = 0.0
            if label not in label_to_thpt or thpt > label_to_thpt[label]:
                label_to_thpt[label] = thpt

        scored = []
        for candidate in candidates:
            label = _candidate_label(candidate)
            thpt = label_to_thpt.get(label, 0.0)
            scored.append((thpt, candidate))

        scored.sort(key=lambda x: x[0], reverse=True)
        selected = [c for _, c in scored[:top_n]]

        for candidate in selected:
            logger.info(
                "Selected decode DGD: tp=%d pp=%d dp=%d moe_tp=%d moe_ep=%d",
                candidate.tp, candidate.pp, candidate.dp,
                getattr(candidate, "moe_tp", 1),
                getattr(candidate, "moe_ep", 1),
            )
        return selected

    except Exception:
        logger.warning(
            "AIC pre-filter failed for decode candidates; using full enumeration.",
            exc_info=True,
        )
        return list(candidates)
