# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``BuiltinThroughputPropose`` — PROPOSE-stage builtin (DEP-XXXX PR 6 sub-task 6-3).

Ports PSM's ``_advance_throughput`` pipeline
(``_throughput_{agg,disagg,single}`` + ``_compute_{prefill,decode}_replicas``)
into a standalone plugin. Reads ``PredictionData`` from ``ctx.predictions``
(produced by ``BuiltinLoadPredictor``) and the raw traffic duration from
``ctx.observations.traffic``.

Output type
-----------

The plugin emits an ``OverrideResult`` where every target has the same
``type`` field, determined by ``config.enable_load_scaling``:

- ``enable_load_scaling=True`` → ``AT_LEAST(desired)`` — load-propose
  will later emit ``SET(load_desired)``; PR 4 ``type_aware_merge`` then
  computes ``max(floor, min(ceiling, rec))``, matching PSM's
  ``max(load_desired, throughput_lower_bound)`` behaviour.
- ``enable_load_scaling=False`` → ``SET(desired)``; the CONSTRAIN stage
  applies the ``AT_MOST`` budget ceiling. This matches PSM's
  ``apply_single_budget(desired)`` call inside ``_throughput_*``.

Budget application
------------------

This plugin **does not** apply ``max_gpu_budget`` / ``min_endpoint``
clamps itself — that's the job of ``BuiltinBudgetConstrain`` (PR 6 6-6)
at the CONSTRAIN stage, which emits ``AT_LEAST=min_endpoint`` and
``AT_MOST=max_gpu_budget/gpu_per_component``. The only floor this
plugin still applies (matching PSM) is ``config.min_endpoint`` inside
``_compute_*_replicas`` — that's the PSM-internal "never return less
than min_endpoint" contract.

Regression models + capabilities
--------------------------------

- ``self.get_regression("prefill" / "decode" / "agg")`` — live references
  to the orchestrator-owned regression models. PR 7 NativePlannerBase
  seeds these at startup from mode-specific regression instances.
- ``self._orch.capabilities`` — static per-engine
  ``WorkerCapabilities`` (max_num_batched_tokens, max_kv_tokens, etc.).

If either is missing, the plugin degrades gracefully to ``AcceptResult``
(PSM equivalent: ``_diag_throughput_reason="model_not_ready"``).
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Optional

from dynamo.planner.core.types import ScalingDecision
from dynamo.planner.plugins.builtins.base import BuiltinPluginBase
from dynamo.planner.plugins.types import (
    AcceptResult,
    ComponentTarget,
    OverrideResult,
    OverrideType,
    ProposeStageRequest,
    ProposeStageResponse,
)

if TYPE_CHECKING:
    from dynamo.planner.config.planner_config import PlannerConfig
    from dynamo.planner.plugins.orchestrator.orchestrator import (
        LocalPlannerOrchestrator,
    )

log = logging.getLogger(__name__)


def _accept() -> ProposeStageResponse:
    return ProposeStageResponse(result_kind="accept", accept=AcceptResult())


class BuiltinThroughputPropose(BuiltinPluginBase):
    """PROPOSE-stage builtin porting PSM's throughput-scaling path."""

    def __init__(
        self,
        orchestrator: "LocalPlannerOrchestrator",
        config: "PlannerConfig",
    ) -> None:
        super().__init__(orchestrator, config)

        # PR 8 A2: last-tick diagnostic summary, read by
        # ``OrchestratorEngineAdapter`` after ``orchestrator.tick`` so
        # the observable ``TickDiagnostics.throughput_decision_reason*``
        # fields match the values PSM path surfaces. Mirrors 8-9's
        # ``BuiltinLoadPropose._last_load_diagnostics`` shape so the
        # adapter projection helper can stay symmetric.
        #
        # Vocabulary matches PSM's ``_diag_throughput_reason`` strings
        # (see ``THROUGHPUT_DECISION_STATES`` in monitoring) so existing
        # dashboards keep working unchanged when the orchestrator path
        # populates them: ``disabled`` / ``no_traffic_data`` /
        # ``predict_failed`` / ``model_not_ready`` / ``set_lower_bound`` /
        # ``scale``.
        self._last_throughput_diagnostics: dict = self._empty_diagnostics()

    @staticmethod
    def _empty_diagnostics() -> dict:
        return {"agg": None, "prefill": None, "decode": None}

    def _set_reason(self, component: str, reason: str) -> None:
        """Record the classification taken for ``component``
        (``agg``/``prefill``/``decode``). Called at every decision branch
        so the adapter has a populated reason regardless of which
        early-return fired."""
        self._last_throughput_diagnostics[component] = reason

    def _set_disagg_reason(self, reason: str) -> None:
        """Convenience: stamp both prefill + decode with the same reason
        — used for shared early-return branches (``disabled``,
        ``no_traffic_data``, ``predict_failed``)."""
        self._set_reason("prefill", reason)
        self._set_reason("decode", reason)

    # ------------------------------------------------------------------
    # Stage dispatch
    # ------------------------------------------------------------------

    async def Propose(
        self, request: ProposeStageRequest
    ) -> ProposeStageResponse:
        # PR 8 A2: reset diagnostics at the start of every evaluation so
        # the adapter never reads a stale reason from the previous tick.
        self._last_throughput_diagnostics = self._empty_diagnostics()
        mode = self._config.mode

        if not self._config.enable_throughput_scaling:
            self._stamp_mode_reason(mode, "disabled")
            return _accept()

        ctx = request.context
        if ctx is None or ctx.predictions is None or ctx.observations is None:
            self._stamp_mode_reason(mode, "predict_failed")
            return _accept()

        preds = ctx.predictions
        if (
            preds.predicted_num_req is None
            or preds.predicted_isl is None
            or preds.predicted_osl is None
        ):
            self._stamp_mode_reason(mode, "predict_failed")
            return _accept()

        traffic = ctx.observations.traffic
        if traffic is None or traffic.duration_s <= 0:
            self._stamp_mode_reason(mode, "no_traffic_data")
            return _accept()

        demand_rps = preds.predicted_num_req / traffic.duration_s
        isl = preds.predicted_isl
        osl = preds.predicted_osl

        if mode == "agg":
            decision = self._throughput_agg(demand_rps, isl, osl)
        elif mode == "disagg":
            decision = self._throughput_disagg(demand_rps, isl, osl)
        else:
            decision = self._throughput_single(demand_rps, isl, osl, mode)

        if decision is None:
            # ``_throughput_*`` already stamped per-component
            # ``model_not_ready`` for the side(s) that failed; if neither
            # stamped (shouldn't happen), default to model_not_ready so
            # the adapter never reads None on a real failure path.
            self._stamp_mode_reason_if_unset(mode, "model_not_ready")
            return _accept()

        # v11 § M-K (plugin decomposition): when ``enable_load_scaling=True``
        # PSM's ``_advance_throughput`` sets ``_throughput_lower_bound_p/d``
        # (a side effect) and returns ``None`` — load-propose later reads
        # that bound and incorporates it into its decision. Mirror that
        # here: publish the bounds on the orchestrator's shared state and
        # return Accept so we don't double-apply via merge AT_LEAST.
        if self._config.enable_load_scaling:
            if decision.num_prefill is not None:
                self._orch.set_throughput_lower_bound(
                    "prefill", decision.num_prefill
                )
                self._set_reason("prefill", "set_lower_bound")
            if decision.num_decode is not None:
                self._orch.set_throughput_lower_bound(
                    "decode", decision.num_decode
                )
                # agg mode emits a single decode-side decision — record
                # it under the agg slot so the adapter projects it onto
                # the aggregate ``throughput_decision_reason`` field.
                if mode == "agg":
                    self._set_reason("agg", "set_lower_bound")
                else:
                    self._set_reason("decode", "set_lower_bound")
            return _accept()

        # enable_load_scaling=False: throughput is authoritative; emit SET.
        targets: list[ComponentTarget] = []
        if decision.num_prefill is not None:
            targets.append(
                ComponentTarget(
                    sub_component_type="prefill",
                    replicas=decision.num_prefill,
                    type=OverrideType.SET,
                )
            )
            self._set_reason("prefill", "scale")
        if decision.num_decode is not None:
            targets.append(
                ComponentTarget(
                    sub_component_type="decode",
                    replicas=decision.num_decode,
                    type=OverrideType.SET,
                )
            )
            if mode == "agg":
                self._set_reason("agg", "scale")
            else:
                self._set_reason("decode", "scale")
        if not targets:
            self._stamp_mode_reason_if_unset(mode, "model_not_ready")
            return _accept()

        return ProposeStageResponse(
            result_kind="override",
            override=OverrideResult(
                targets=targets,
                reason="builtin_throughput_propose",
            ),
        )

    def _stamp_mode_reason(self, mode: str, reason: str) -> None:
        """Stamp the right slot(s) for the active mode."""
        if mode == "agg":
            self._set_reason("agg", reason)
        elif mode == "disagg":
            self._set_disagg_reason(reason)
        else:
            self._set_reason(mode, reason)

    def _stamp_mode_reason_if_unset(self, mode: str, reason: str) -> None:
        d = self._last_throughput_diagnostics
        if mode == "agg":
            if d["agg"] is None:
                self._set_reason("agg", reason)
        elif mode == "disagg":
            if d["prefill"] is None:
                self._set_reason("prefill", reason)
            if d["decode"] is None:
                self._set_reason("decode", reason)
        else:
            if d.get(mode) is None:
                self._set_reason(mode, reason)

    # ------------------------------------------------------------------
    # Ported algorithm: per-mode throughput computation
    # ------------------------------------------------------------------

    def _throughput_single(
        self, demand_rps: float, isl: float, osl: float, component: str
    ) -> Optional[ScalingDecision]:
        desired = (
            self._compute_prefill_replicas(demand_rps, isl, osl)
            if component == "prefill"
            else self._compute_decode_replicas(demand_rps, isl, osl)
        )
        if desired is None:
            return None
        return (
            ScalingDecision(num_prefill=desired)
            if component == "prefill"
            else ScalingDecision(num_decode=desired)
        )

    def _throughput_disagg(
        self, demand_rps: float, isl: float, osl: float
    ) -> Optional[ScalingDecision]:
        num_p = self._compute_prefill_replicas(demand_rps, isl, osl)
        num_d = self._compute_decode_replicas(demand_rps, isl, osl)
        if num_p is None or num_d is None:
            return None
        return ScalingDecision(num_prefill=num_p, num_decode=num_d)

    def _throughput_agg(
        self, demand_rps: float, isl: float, osl: float
    ) -> Optional[ScalingDecision]:
        caps = self._orch.capabilities
        d_caps = caps.decode if caps is not None else None
        max_tokens = d_caps.max_num_batched_tokens if d_caps else None
        if not max_tokens or max_tokens <= 0:
            self._set_reason("agg", "model_not_ready")
            return None

        agg_reg = self.get_regression("agg")
        if agg_reg is None:
            self._set_reason("agg", "model_not_ready")
            return None

        engine_rps, actual_ttft, actual_itl = agg_reg.find_best_engine_agg_rps(
            isl=isl,
            osl=osl,
            max_num_batched_tokens=max_tokens,
            ttft_sla=self._config.ttft,
            itl_sla=self._config.itl,
            max_kv_tokens=d_caps.max_kv_tokens if d_caps else None,
            max_num_seqs=d_caps.max_num_seqs if d_caps else None,
        )
        if engine_rps <= 0:
            self._set_reason("agg", "model_not_ready")
            return None
        # PSM logs a warning on SLA miss but still returns the decision;
        # behaviour mirrored here via log.info.
        if actual_ttft > self._config.ttft or actual_itl > self._config.itl:
            log.info(
                "builtin_throughput_propose agg SLA miss: ttft=%.1fms itl=%.1fms",
                actual_ttft,
                actual_itl,
            )

        desired = max(math.ceil(demand_rps / engine_rps), self._config.min_endpoint)
        return ScalingDecision(num_decode=desired)

    def _compute_prefill_replicas(
        self, demand_rps: float, isl: float, osl: float
    ) -> Optional[int]:
        p_reg = self.get_regression("prefill")
        if p_reg is None:
            self._set_reason("prefill", "model_not_ready")
            return None
        caps = self._orch.capabilities
        p_caps = caps.prefill if caps is not None else None
        engine_rps, ttft_ms = p_reg.find_best_engine_prefill_rps(
            ttft_sla=self._config.ttft,
            isl=isl,
            max_num_batched_tokens=p_caps.max_num_batched_tokens if p_caps else None,
        )
        if engine_rps <= 0:
            self._set_reason("prefill", "model_not_ready")
            return None
        if ttft_ms > self._config.ttft:
            log.info(
                "builtin_throughput_propose prefill TTFT miss: %.1fms > %.1fms",
                ttft_ms,
                self._config.ttft,
            )
        return max(math.ceil(demand_rps / engine_rps), self._config.min_endpoint)

    def _compute_decode_replicas(
        self, demand_rps: float, isl: float, osl: float
    ) -> Optional[int]:
        d_reg = self.get_regression("decode")
        if d_reg is None:
            self._set_reason("decode", "model_not_ready")
            return None
        caps = self._orch.capabilities
        d_caps = caps.decode if caps is not None else None
        engine_rps, itl_ms = d_reg.find_best_engine_decode_rps(
            itl=self._config.itl,
            context_length=isl + osl / 2,
            osl=osl,
            max_kv_tokens=d_caps.max_kv_tokens if d_caps else None,
            max_num_seqs=d_caps.max_num_seqs if d_caps else None,
        )
        if engine_rps <= 0:
            self._set_reason("decode", "model_not_ready")
            return None
        if itl_ms > self._config.itl:
            log.info(
                "builtin_throughput_propose decode ITL miss: %.1fms > %.1fms",
                itl_ms,
                self._config.itl,
            )
        return max(math.ceil(demand_rps / engine_rps), self._config.min_endpoint)


__all__ = ["BuiltinThroughputPropose"]
