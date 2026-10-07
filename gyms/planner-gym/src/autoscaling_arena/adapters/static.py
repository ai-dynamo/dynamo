# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static (fixed-count) autoscaler — the continuity baseline.

Holds a constant replica count for the whole run; it never reacts. This is the
"best static config" the project has historically compared the Planner against,
brought behind the common adapter interface so it ranks on the same board as
every reactive autoscaler.

It is an ``EngineProtocol`` implementation (structural — no inheritance needed):
the replay harness drives ``initial_tick`` / ``tick`` / ``shutdown`` and routes
the returned :class:`ScalingDecision` through the same Rust scaling-policy loop
as every other autoscaler. Because the runtime no-ops a decision that matches
the current fleet, a static adapter started at its target count produces zero
scaling events.
"""

from __future__ import annotations

from autoscaling_arena.adapters._regression_bootstrap import _NoopRegressionBootstrap

from dynamo.planner.core.types import (
    PlannerEffects,
    ScalingDecision,
    ScheduledTick,
    TickInput,
)


class StaticAutoscaler(_NoopRegressionBootstrap):
    """Returns a fixed ``(num_prefill, num_decode)`` every tick.

    Args:
        num_prefill: fixed prefill replica count (ignored in agg mode).
        num_decode: fixed decode (or aggregated) replica count.
        mode: ``"disagg"`` sets both pools; ``"agg"`` sets decode only.
        poll_interval_s: virtual-clock cadence between ticks. Does not affect
            the simulated result (no scaling happens); it only paces how finely
            the harness advances the clock and integrates GPU-hours.
    """

    def __init__(
        self,
        *,
        num_prefill: int,
        num_decode: int,
        mode: str = "disagg",
        poll_interval_s: float = 5.0,
    ) -> None:
        self._num_prefill = num_prefill
        self._num_decode = num_decode
        self._is_disagg = mode == "disagg"
        self._poll_interval_s = poll_interval_s

    def initial_tick(self, start_s: float) -> ScheduledTick:
        # Needs current worker states only so the harness can report counts;
        # no traffic / FPM signals are consumed (it never reacts).
        return ScheduledTick(at_s=start_s, need_worker_states=True)

    async def tick(
        self, scheduled_tick: ScheduledTick, tick_input: TickInput
    ) -> PlannerEffects:
        target = ScalingDecision(
            num_prefill=self._num_prefill if self._is_disagg else None,
            num_decode=self._num_decode,
        )
        return PlannerEffects(
            scale_to=target,
            next_tick=ScheduledTick(
                at_s=scheduled_tick.at_s + self._poll_interval_s,
                need_worker_states=True,
            ),
        )

    async def shutdown(self) -> None:
        return None
