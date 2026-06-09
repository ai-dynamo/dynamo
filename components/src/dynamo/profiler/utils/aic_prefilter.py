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

Scores enumerated candidates offline (no GPUs) using the aiconfigurator SDK
and keeps only the top-N per side, reducing expensive GPU benchmark time.

Best-effort: any AIC failure returns the original candidates unchanged.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

try:
    from aiconfigurator.sdk.task import TaskConfig, TaskRunner
except ImportError:
    TaskConfig = None
    TaskRunner = None

logger = logging.getLogger(__name__)


def prefilter_prefill_candidates(
    candidates: Sequence,
    top_n: int,
    model: str,
    system: str,
    backend: str,
    isl: int,
    osl: int,
) -> list:
    """Keep the top-N prefill candidates ranked by predicted TTFT (ascending).

    Uses ``aiconfigurator.sdk.task.TaskRunner`` to simulate each candidate
    offline and predict TTFT. Candidates with lower predicted TTFT are
    preferred.

    Returns the original list unchanged on any AIC error (best-effort).
    """
    if top_n is None or top_n <= 0 or len(candidates) <= top_n:
        return list(candidates)

    if TaskRunner is None or TaskConfig is None:
        logger.warning("aiconfigurator not available; skipping prefill pre-filter.")
        return list(candidates)

    try:
        runner = TaskRunner()
        scored = []
        for candidate in candidates:
            tc = TaskConfig(
                serving_mode="disagg",
                model_path=model,
                system_name=system,
                backend_name=backend,
                total_gpus=candidate.num_gpus,
                isl=isl,
                osl=osl,
                tp=candidate.tp,
                pp=candidate.pp,
                dp=candidate.dp,
                moe_tp=getattr(candidate, "moe_tp", 1),
                moe_ep=getattr(candidate, "moe_ep", 1),
            )
            result = runner.run(tc)
            ttft = result.get("ttft", float("inf")) if result else float("inf")
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
    model: str,
    system: str,
    backend: str,
    isl: int,
    osl: int,
) -> list:
    """Keep the top-N decode candidates ranked by predicted tokens/s/gpu (descending).

    Uses ``aiconfigurator.sdk.task.TaskRunner`` to simulate each candidate
    offline and predict throughput per GPU. Candidates with higher predicted
    throughput are preferred.

    Returns the original list unchanged on any AIC error (best-effort).
    """
    if top_n is None or top_n <= 0 or len(candidates) <= top_n:
        return list(candidates)

    if TaskRunner is None or TaskConfig is None:
        logger.warning("aiconfigurator not available; skipping decode pre-filter.")
        return list(candidates)

    try:
        runner = TaskRunner()
        scored = []
        for candidate in candidates:
            tc = TaskConfig(
                serving_mode="disagg",
                model_path=model,
                system_name=system,
                backend_name=backend,
                total_gpus=candidate.num_gpus,
                isl=isl,
                osl=osl,
                tp=candidate.tp,
                pp=candidate.pp,
                dp=candidate.dp,
                moe_tp=getattr(candidate, "moe_tp", 1),
                moe_ep=getattr(candidate, "moe_ep", 1),
            )
            result = runner.run(tc)
            thpt_per_gpu = (
                result.get("thpt_per_gpu", 0.0) if result else 0.0
            )
            scored.append((thpt_per_gpu, candidate))

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
