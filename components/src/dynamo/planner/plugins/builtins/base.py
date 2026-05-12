# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``BuiltinPluginBase`` — base class for real builtin plugins.

Holds a reference to the owning ``LocalPlannerOrchestrator`` so
subclasses can reach the shared regression-model store via
``get_regression`` / ``update_regression`` forwarders.

Subclasses typically also hold the ``PlannerConfig`` so they can
inspect ``enable_load_scaling`` / ``enable_throughput_scaling`` /
``optimization_target`` / ``mode`` at dispatch time — the config is
passed in by ``LocalPlannerOrchestrator`` at registration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from dynamo.planner.plugins.lifecycle import PluginLifecycle
from dynamo.planner.plugins.types import (
    BootstrapRequest,
    BootstrapResponse,
    ResetRequest,
    ResetResponse,
)

if TYPE_CHECKING:
    from dynamo.planner.config.planner_config import PlannerConfig
    from dynamo.planner.plugins.orchestrator.orchestrator import (
        LocalPlannerOrchestrator,
    )


class BuiltinPluginBase(PluginLifecycle):
    """Shared scaffolding for all builtin plugins.

    Subclasses override ``Bootstrap`` / ``Reset`` when they have state
    to seed or zero; the default implementations return ``ok=True`` so
    stateless builtins (e.g. RECONCILE wrapping a pure merge function)
    don't need to write boilerplate.
    """

    def __init__(
        self,
        orchestrator: "LocalPlannerOrchestrator",
        config: "PlannerConfig",
    ) -> None:
        self._orch = orchestrator
        self._config = config

    # ------------------------------------------------------------------
    # Regression-model forwarders (single-threaded asyncio)
    # ------------------------------------------------------------------

    def get_regression(self, kind: str) -> Optional[Any]:
        """Live reference to the orchestrator-owned regression model for
        ``kind``. Single-threaded asyncio: do NOT hold the reference
        across an ``await`` (see ``LocalPlannerOrchestrator.get_regression``)."""
        return self._orch.get_regression(kind)

    def update_regression(self, kind: str, model: Any) -> None:
        """Install / replace the regression model for ``kind`` on the
        orchestrator-owned store."""
        self._orch.update_regression(kind, model)

    # ------------------------------------------------------------------
    # Default lifecycle implementations (stateless builtins inherit as-is)
    # ------------------------------------------------------------------

    async def Bootstrap(self, request: BootstrapRequest) -> BootstrapResponse:
        return BootstrapResponse(ok=True)

    async def Reset(self, request: ResetRequest) -> ResetResponse:
        return ResetResponse(ok=True)


__all__ = ["BuiltinPluginBase"]
