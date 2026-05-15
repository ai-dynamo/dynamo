# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``EngineProtocol`` — shared abstraction for the tick engine.

``NativePlannerBase`` drives its tick loop through an ``EngineProtocol``
rather than a concrete ``PlannerStateMachine`` so the planner can run
under two paths that are selectable at runtime via
``PlannerConfig.scheduling.use_orchestrator``:

- **PSM path** (default, ``use_orchestrator=False``): legacy behaviour.
  ``_PSMEngineAdapter`` wraps a ``PlannerStateMachine`` instance and
  forwards tick calls to its synchronous ``on_tick``.
- **Orchestrator path** (``use_orchestrator=True``): plugin
  decomposition. A separate orchestrator adapter bridges
  ``TickInput`` → ``PipelineContext`` and projects ``PipelineOutcome``
  back onto ``PlannerEffects``.

Both paths produce the same ``PlannerEffects`` shape so
``NativePlannerBase._apply_effects`` and downstream metric emission
stay unchanged.

Bootstrap paths are deliberately **not** on the protocol —
``NativePlannerBase._bootstrap_regression`` branches on the config flag
explicitly because PSM's ``load_benchmark_fpms`` + ``warm_load_predictors``
and orchestrator's ``install_regressions`` + ``bootstrap_plugins`` have
different input shapes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from dynamo.planner.core.types import PlannerEffects, ScheduledTick, TickInput

if TYPE_CHECKING:
    from dynamo.planner.core.state_machine import PlannerStateMachine


@runtime_checkable
class EngineProtocol(Protocol):
    """Tick-engine abstraction shared by PSM and LocalPlannerOrchestrator."""

    def initial_tick(self, start_s: float) -> ScheduledTick:
        """Build the first ``ScheduledTick`` for the main loop to wait on."""
        ...

    async def tick(
        self,
        scheduled_tick: ScheduledTick,
        tick_input: TickInput,
    ) -> PlannerEffects:
        """Drive one tick; return the decision + next scheduled tick +
        diagnostics. Path implementations absorb the concrete
        sync/async difference of their underlying engine."""
        ...

    async def shutdown(self) -> None:
        """Release any engine-owned resources. Idempotent."""
        ...


class _PSMEngineAdapter:
    """Adapts ``PlannerStateMachine`` (synchronous ``on_tick``) to
    ``EngineProtocol``. Zero behaviour change from legacy — just an
    async wrapper around the sync call so the protocol stays uniform."""

    def __init__(self, psm: "PlannerStateMachine") -> None:
        self._psm = psm

    def initial_tick(self, start_s: float) -> ScheduledTick:
        return self._psm.initial_tick(start_s)

    async def tick(
        self,
        scheduled_tick: ScheduledTick,
        tick_input: TickInput,
    ) -> PlannerEffects:
        return self._psm.on_tick(scheduled_tick, tick_input)

    async def shutdown(self) -> None:
        # PSM is in-process + holds no transports; nothing to release.
        return None


__all__ = ["EngineProtocol", "_PSMEngineAdapter"]
