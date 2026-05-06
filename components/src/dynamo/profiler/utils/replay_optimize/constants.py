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
