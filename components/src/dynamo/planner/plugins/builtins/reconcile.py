# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``BuiltinReconcile``.

The simplest real builtin: a passthrough that re-emits
``PipelineContext.proposal`` as an ``OverrideResult`` with ``SET``
targets during the RECONCILE stage.

Why have it at all? Two reasons:

1. The RECONCILE stage exists so user-provided reconcile plugins can
   participate in priority-aware merging. With at least one builtin
   present that always emits the current proposal, the merge at this
   stage is well-defined even when no user plugin fires
   (``type_aware_merge`` collapses the single entry back to the same
   proposal).
2. Future work: when ``ReconcileStageRequest.proposals`` is threaded
   through the pipeline with the full list of per-plugin PROPOSE
   outputs, this builtin can fold that list into its own merge call
   and surface richer reconciled output. That extension is additive —
   we keep the same plugin class.

Until the proposals list is threaded, this plugin is a passthrough.
That is **sufficient** for G3 parity because the orchestrator pipeline
already re-invokes ``type_aware_merge`` on the aggregated plugin
results at the RECONCILE stage; this builtin only adds a single
priority-1 OverrideResult that matches the incoming proposal.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from dynamo.planner.plugins.builtins.base import BuiltinPluginBase
from dynamo.planner.plugins.types import (
    AcceptResult,
    ComponentTarget,
    OverrideResult,
    OverrideType,
    ReconcileStageRequest,
    ReconcileStageResponse,
)

if TYPE_CHECKING:
    from dynamo.planner.config.planner_config import PlannerConfig
    from dynamo.planner.plugins.orchestrator.orchestrator import (
        LocalPlannerOrchestrator,
    )


class BuiltinReconcile(BuiltinPluginBase):
    """RECONCILE-stage passthrough: re-emits ``ctx.proposal`` as SET."""

    def __init__(
        self,
        orchestrator: "LocalPlannerOrchestrator",
        config: "PlannerConfig",
    ) -> None:
        super().__init__(orchestrator, config)

    async def Reconcile(
        self, request: ReconcileStageRequest
    ) -> ReconcileStageResponse:
        ctx = request.context
        proposal = ctx.proposal if ctx is not None else None

        if proposal is None or not proposal.targets:
            # Nothing to re-emit — stay out of the merge entirely.
            return ReconcileStageResponse(
                result_kind="accept", accept=AcceptResult()
            )

        targets: list[ComponentTarget] = [
            ComponentTarget(
                sub_component_type=t.sub_component_type,
                component_name=t.component_name,
                replicas=t.replicas,
                type=OverrideType.SET,
            )
            for t in proposal.targets
            if t.replicas is not None
        ]
        if not targets:
            return ReconcileStageResponse(
                result_kind="accept", accept=AcceptResult()
            )
        return ReconcileStageResponse(
            result_kind="override",
            override=OverrideResult(targets=targets, reason="builtin_reconcile"),
        )


__all__ = ["BuiltinReconcile"]
