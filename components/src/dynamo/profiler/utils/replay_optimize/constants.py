# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

AIC_BACKEND_VERSIONS = {
    "vllm": "0.12.0",
    "sglang": "0.5.6.post2",
}

DEFAULT_OVERLAP_SCORE_WEIGHTS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
DEFAULT_MAX_PARALLEL_EVALS = min(8, os.cpu_count() or 1)
DEFAULT_SEARCH_ROUNDS = 3
SUPPORTED_CONSTRAINTS = frozenset(
    {
        "mean_ttft_ms",
        "p95_ttft_ms",
        "mean_tpot_ms",
        "p95_tpot_ms",
        "mean_e2e_latency_ms",
        "p95_e2e_latency_ms",
        "max_total_gpus",
    }
)
