# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wall-clock phase log for Dynamo wrapper overhead.

Logs are grep-able as ``[dyn-phase]`` so a restart measurement can subtract
wrapper time from stock-vLLM time. Process-global: one timer per engine
container, started at ``worker()`` entry.
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)

_t0: float | None = None
_last: float | None = None


def start(label: str = "process") -> None:
    global _t0, _last
    now = time.perf_counter()
    _t0 = now
    _last = now
    logger.info("[dyn-phase] phase=%s elapsed_s=0.000 total_s=0.000", label)


def mark(phase: str) -> float:
    """Record a phase boundary. Returns seconds since the previous mark."""
    global _last
    now = time.perf_counter()
    if _t0 is None:
        start("implicit")
    assert _t0 is not None and _last is not None
    elapsed = now - _last
    total = now - _t0
    logger.info(
        "[dyn-phase] phase=%s elapsed_s=%.3f total_s=%.3f",
        phase,
        elapsed,
        total,
    )
    _last = now
    return elapsed
