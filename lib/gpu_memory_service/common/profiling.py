# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lightweight opt-in profiling helpers for GMS restore setup.

The helpers in this module intentionally avoid external dependencies and are
disabled by default.  Set either of the following environment variables to a
truthy value to emit INFO-level profiling logs:

* ``GMS_PROFILE_RESTORE_SETUP=1``
* ``GMS_RESTORE_PROFILE=1``
"""

from __future__ import annotations

import logging
import math
import os
from typing import Sequence

_TRUE_VALUES = {"1", "true", "yes", "on", "y"}


def restore_setup_profiling_enabled() -> bool:
    return (
        os.getenv("GMS_PROFILE_RESTORE_SETUP", "").strip().lower() in _TRUE_VALUES
        or os.getenv("GMS_RESTORE_PROFILE", "").strip().lower() in _TRUE_VALUES
    )


def profile_log(
    logger: logging.Logger,
    message: str,
    *args: object,
    **kwargs: object,
) -> None:
    """Emit an INFO profiling log only when restore setup profiling is enabled."""

    if not restore_setup_profiling_enabled():
        return
    logger.info("[GMS profile] " + message, *args, **kwargs)


def seconds_summary(samples: Sequence[float]) -> str:
    """Return a compact summary for a list of durations in seconds."""

    if not samples:
        return "count=0"
    ordered = sorted(float(sample) for sample in samples)
    count = len(ordered)

    def percentile(fraction: float) -> float:
        index = min(count - 1, max(0, math.ceil(count * fraction) - 1))
        return ordered[index]

    total = sum(ordered)
    return (
        f"count={count} total={total:.6f}s avg={total / count:.6f}s "
        f"p50={percentile(0.50):.6f}s p95={percentile(0.95):.6f}s "
        f"max={ordered[-1]:.6f}s"
    )
