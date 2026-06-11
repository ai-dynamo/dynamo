# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``EngineProtocol`` — shared abstraction for the tick engine.

``NativePlannerBase`` drives its tick loop through an ``EngineProtocol``
rather than a concrete implementation. Runtime planner execution uses
``OrchestratorEngineAdapter``, which bridges ``TickInput`` →
``PipelineContext`` and projects ``PipelineOutcome`` back onto
``PlannerEffects``.

"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from dynamo.planner.core.types import PlannerEffects, ScheduledTick, TickInput


@runtime_checkable
class EngineProtocol(Protocol):
    """Tick-engine abstraction consumed by ``NativePlannerBase``."""

    def initial_tick(self, start_s: float) -> ScheduledTick:
        """Build the first ``ScheduledTick`` for the main loop to wait on."""

    async def tick(
        self,
        scheduled_tick: ScheduledTick,
        tick_input: TickInput,
    ) -> PlannerEffects:
        """Drive one tick; return the decision + next scheduled tick +
        diagnostics. Path implementations absorb the concrete
        sync/async difference of their underlying engine."""

    async def shutdown(self) -> None:
        """Release any engine-owned resources. Idempotent."""


__all__ = ["EngineProtocol"]
