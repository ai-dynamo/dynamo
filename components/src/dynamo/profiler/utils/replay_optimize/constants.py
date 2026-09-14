# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

# Backends replay_optimize can evaluate through the AIC perf model. Versions are
# not pinned here; replay materialization resolves them from the installed
# aisimulate perf database (dynamo._internal.aic.resolve_backend_version).
AIC_REPLAY_BACKENDS = frozenset({"vllm", "sglang"})

DEFAULT_OVERLAP_SCORE_CREDITS = (1.0,)
DEFAULT_PREFILL_LOAD_SCALES = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
DEFAULT_MAX_PARALLEL_EVALS = min(8, os.cpu_count() or 1)
DEFAULT_SEARCH_ROUNDS = 3
