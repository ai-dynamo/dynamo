# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic clock for the testbed.

Replaces ``time.monotonic()`` inside ``AICPowerOptimizer`` so that rate-limit
semantics for ``aic_reoptimize_interval`` are fully controllable.  The current
tick × interval_s gives a stable monotonic value without relying on wall-clock.
"""

from __future__ import annotations


class Clock:
    """Deterministic virtual clock tied to the scenario tick counter."""

    def __init__(self, interval_s: float = 60.0) -> None:
        self._interval_s = interval_s
        self._tick: int = 0

    def advance(self, tick: int) -> None:
        """Called by the runner at the start of each tick."""
        self._tick = tick

    def now(self) -> float:
        """Return virtual monotonic time in seconds."""
        return self._tick * self._interval_s
