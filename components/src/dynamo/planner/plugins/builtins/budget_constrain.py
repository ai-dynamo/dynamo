# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``BuiltinBudgetConstrain`` — CONSTRAIN-stage builtin (DEP-XXXX PR 6 sub-task 6-6).

Semantic rewrite of PSM's ``_apply_global_budget`` / ``_apply_single_budget``:
instead of clamping inside the ``_advance_*`` decision methods, emit
``AT_LEAST`` (min_endpoint floor) and ``AT_MOST`` (max_gpu_budget ceiling)
at the CONSTRAIN stage. PR 4 ``type_aware_merge`` with ``set_allowed=False``
applies them to the upstream proposal.

Output shape
------------

- **min_endpoint floor**: one ``AT_LEAST`` per engine-configured component
  — prefill gets one when ``capabilities.prefill`` is set, decode when
  ``capabilities.decode`` is set.
- **max_gpu_budget ceiling**: per-component ``AT_MOST``. The formula
  (per component X, other component Y):

      ceiling_X = floor((max_gpu_budget - min_endpoint * gpu_Y) / gpu_X)

  This reserves enough GPUs for Y's ``min_endpoint`` floor before
  giving X its ceiling. Matches PSM's ``_apply_global_budget`` allocation
  order ("prefill priority, decode uses remaining") for the single-
  component case; **diverges** from PSM when BOTH components exceed
  their independent ceiling simultaneously (PSM does proportional
  scaling, this emits independent ceilings). Documented trade-off per
  PR 6 design.
- **scaling_in_progress freeze**: when ``expected_num_*`` differs from
  ``ready_num_*``, emit ``AT_LEAST=current`` **and** ``AT_MOST=current``
  so the merge clamps replicas to their current value (no scaling
  while a prior scale operation is still completing).
- **budget starvation (Q1)**: when ``max_gpu_budget < min_endpoint *
  (prefill_gpu + decode_gpu)``, the plugin emits ``AT_MOST=0`` per
  component and **does not** emit ``AT_LEAST=min_endpoint`` — otherwise
  the merge's ``max(floor, min(ceiling, rec))`` would produce
  ``min_endpoint`` when we want ``0``.

Input side-channel
------------------

Like ``BuiltinLoadPropose``, the plugin reads ``WorkerCounts`` via a
side-channel ``prime_tick`` helper because ``PipelineContext.observations.workers``
in PR 1 format doesn't fully round-trip the ``expected_num_*`` fields
the scaling-in-progress check needs. PR 6 6-9 / PR 7 wire this through
the orchestrator.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from dynamo.planner.core.types import WorkerCounts
from dynamo.planner.plugins.builtins.base import BuiltinPluginBase
from dynamo.planner.plugins.types import (
    AcceptResult,
    ComponentTarget,
    ConstrainStageRequest,
    ConstrainStageResponse,
    OverrideResult,
    OverrideType,
)

if TYPE_CHECKING:
    from dynamo.planner.config.planner_config import PlannerConfig
    from dynamo.planner.plugins.orchestrator.orchestrator import (
        LocalPlannerOrchestrator,
    )

log = logging.getLogger(__name__)


def _accept() -> ConstrainStageResponse:
    return ConstrainStageResponse(result_kind="accept", accept=AcceptResult())


class BuiltinBudgetConstrain(BuiltinPluginBase):
    """CONSTRAIN-stage builtin emitting AT_LEAST/AT_MOST per component."""

    def __init__(
        self,
        orchestrator: "LocalPlannerOrchestrator",
        config: "PlannerConfig",
    ) -> None:
        super().__init__(orchestrator, config)
        self._cached_counts: Optional[WorkerCounts] = None

    def prime_tick(self, counts: Optional[WorkerCounts]) -> None:
        self._cached_counts = counts

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _has_component(self, component: str) -> bool:
        """Mode-based (mirrors PSM ``_has_prefill`` / ``_has_decode``).

        Capabilities may carry entries for both engines regardless of
        mode (config flexibility), so the authoritative signal is the
        planner's ``mode`` field — agg / decode use only the decode
        engine; prefill uses only prefill; disagg uses both.
        """
        mode = self._config.mode
        if component == "prefill":
            return mode in ("disagg", "prefill")
        return mode in ("disagg", "decode", "agg")

    def _gpu_for(self, component: str) -> Optional[int]:
        caps = self._orch.capabilities
        if caps is None:
            return None
        engine_caps = caps.prefill if component == "prefill" else caps.decode
        return engine_caps.num_gpu if engine_caps else None

    def _scaling_in_progress(self, counts: WorkerCounts, component: str) -> bool:
        if component == "prefill":
            return (
                counts.expected_num_prefill is not None
                and counts.expected_num_prefill != (counts.ready_num_prefill or 0)
            )
        return (
            counts.expected_num_decode is not None
            and counts.expected_num_decode != (counts.ready_num_decode or 0)
        )

    # ------------------------------------------------------------------
    # Stage dispatch
    # ------------------------------------------------------------------

    async def Constrain(
        self, request: ConstrainStageRequest
    ) -> ConstrainStageResponse:
        targets: list[ComponentTarget] = []

        # Determine which components this deployment has.
        has_prefill = self._has_component("prefill")
        has_decode = self._has_component("decode")
        if not has_prefill and not has_decode:
            return _accept()

        # Budget state — resolved first so we know whether to emit the
        # min_endpoint AT_LEAST floor (suppressed under starvation).
        budget = self._config.max_gpu_budget
        p_gpu = self._gpu_for("prefill") if has_prefill else 0
        d_gpu = self._gpu_for("decode") if has_decode else 0

        starved = False
        if budget >= 0:
            min_total = self._config.min_endpoint * ((p_gpu or 0) + (d_gpu or 0))
            starved = budget < min_total

        # 1. min_endpoint AT_LEAST floor (skipped when budget-starved per Q1).
        if not starved:
            for component, present in (("prefill", has_prefill), ("decode", has_decode)):
                if present:
                    targets.append(
                        ComponentTarget(
                            sub_component_type=component,
                            replicas=self._config.min_endpoint,
                            type=OverrideType.AT_LEAST,
                        )
                    )

        # 2. max_gpu_budget AT_MOST ceiling per component.
        if budget >= 0:
            if starved:
                # Q1: emit AT_MOST=0; NO AT_LEAST (above skipped) → merge
                # yields 0 without floor>ceiling conflict.
                for component, present in (
                    ("prefill", has_prefill),
                    ("decode", has_decode),
                ):
                    if present:
                        targets.append(
                            ComponentTarget(
                                sub_component_type=component,
                                replicas=0,
                                type=OverrideType.AT_MOST,
                            )
                        )
            else:
                # Reserve min_endpoint of the *other* component before
                # computing this component's ceiling — matches PSM's
                # "prefill priority, decode uses remaining" intent at
                # the level of per-component independence.
                if has_prefill and p_gpu and p_gpu > 0:
                    ceiling_p = (budget - self._config.min_endpoint * (d_gpu or 0)) // p_gpu
                    targets.append(
                        ComponentTarget(
                            sub_component_type="prefill",
                            replicas=max(0, ceiling_p),
                            type=OverrideType.AT_MOST,
                        )
                    )
                if has_decode and d_gpu and d_gpu > 0:
                    ceiling_d = (budget - self._config.min_endpoint * (p_gpu or 0)) // d_gpu
                    targets.append(
                        ComponentTarget(
                            sub_component_type="decode",
                            replicas=max(0, ceiling_d),
                            type=OverrideType.AT_MOST,
                        )
                    )

        # 3. scaling_in_progress freeze per component.
        counts = self._cached_counts
        if counts is not None:
            for component, present in (
                ("prefill", has_prefill),
                ("decode", has_decode),
            ):
                if not present:
                    continue
                if self._scaling_in_progress(counts, component):
                    current = (
                        counts.ready_num_prefill or 0
                        if component == "prefill"
                        else counts.ready_num_decode or 0
                    )
                    # Emit BOTH AT_LEAST and AT_MOST at current → merge
                    # pins the replica count to the current value.
                    targets.append(
                        ComponentTarget(
                            sub_component_type=component,
                            replicas=current,
                            type=OverrideType.AT_LEAST,
                        )
                    )
                    targets.append(
                        ComponentTarget(
                            sub_component_type=component,
                            replicas=current,
                            type=OverrideType.AT_MOST,
                        )
                    )

        if not targets:
            return _accept()
        return ConstrainStageResponse(
            result_kind="override",
            override=OverrideResult(
                targets=targets,
                reason="builtin_budget_constrain",
            ),
        )


__all__ = ["BuiltinBudgetConstrain"]
