# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``BuiltinLoadPropose`` — PROPOSE-stage builtin (DEP-XXXX PR 6 sub-task 6-4).

Ports PSM's ``_advance_load`` family:

- ``_advance_load_single`` / ``_advance_load_disagg`` / ``_advance_load_agg``
- ``_prefill_easy_decision`` / ``_decode_easy_decision`` / ``_agg_easy_decision``
- ``_prefill_load_decision`` / ``_decode_load_decision``
  (SLA paths via ``_prefill_regression.estimate_next_ttft`` /
  ``_decode_regression.estimate_next_itl``)
- ``_agg_prefill_scaling`` / ``_agg_decode_scaling``
- ``_scale_decision``

Scope divergences from PSM (encoded in the implementation):

1. **FPM wire format**: ``PipelineContext.observations.fpm`` uses the
   encoded ``FpmData`` (bytes per engine) shape from PR 1, not PSM's
   Python ``FpmObservations`` dict. Until a decoder lands, the plugin
   reads FPM + worker counts from a side-channel primed via
   ``prime_tick()`` before each ``orchestrator.tick`` call. Unit tests
   and the ``OrchestratorEngineAdapter`` use this priming hook;
   production wire-format deserialisation is a follow-up.
2. **Throughput lower bounds**: PSM's ``_advance_load`` reads
   ``self._throughput_lower_bound_p/d``, set by ``_advance_throughput``
   earlier in the same tick. In the full plugin decomposition, the
   throughput-propose builtin will emit those as ``AT_LEAST`` and the
   merge will compute the floor. Until 6-7 switchover, this plugin
   holds its own copies (default 1 per PSM) and provides
   ``update_throughput_lower_bounds(p, d)`` so the throughput plugin /
   orchestrator can sync them; matches PSM output bit-for-bit in unit
   tests.
3. **Output type**: SET always (PSM returns a concrete
   ``ScalingDecision``). Budget clamping via ``min_endpoint`` /
   ``max_gpu_budget`` is applied inline per PSM; 6-6 will refactor this
   into ``AT_LEAST`` / ``AT_MOST`` at CONSTRAIN stage and strip the
   clamp from this plugin.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Optional

from dynamo.planner.core.load_scaling import (
    _DECODE_LATENCY_SCALE_DOWN,
    _DECODE_LATENCY_SCALE_UP,
    _DECODE_THROUGHPUT_SCALE_DOWN,
    _DECODE_THROUGHPUT_SCALE_UP,
    _PREFILL_LATENCY_SCALE_DOWN,
    _PREFILL_LATENCY_SCALE_UP,
    _PREFILL_THROUGHPUT_SCALE_DOWN,
    _PREFILL_THROUGHPUT_SCALE_UP,
)
from dynamo.planner.core.types import (
    FpmObservations,
    ScalingDecision,
    WorkerCounts,
)
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
    from dynamo.common.forward_pass_metrics import ForwardPassMetrics
    from dynamo.planner.config.planner_config import PlannerConfig
    from dynamo.planner.plugins.orchestrator.orchestrator import (
        LocalPlannerOrchestrator,
    )

log = logging.getLogger(__name__)


def _accept() -> ProposeStageResponse:
    return ProposeStageResponse(result_kind="accept", accept=AcceptResult())


class BuiltinLoadPropose(BuiltinPluginBase):
    """PROPOSE-stage load-scaling builtin ported from PSM."""

    def __init__(
        self,
        orchestrator: "LocalPlannerOrchestrator",
        config: "PlannerConfig",
    ) -> None:
        super().__init__(orchestrator, config)
        # Side-channel tick state primed by ``prime_tick`` before each
        # orchestrator invocation; becomes a wire-format decode call
        # when PR 1 FpmData gains a Python round-trip.
        self._cached_fpm_obs: Optional[FpmObservations] = None
        self._cached_counts: Optional[WorkerCounts] = None

        # PR 8 8-9: last-tick diagnostic summary, read by
        # ``OrchestratorEngineAdapter`` after ``orchestrator.tick`` so
        # the observable ``TickDiagnostics.load_decision_reason*`` +
        # ``estimated_*_ms`` fields match the values PSM path surfaces.
        # Reset to all-None at each ``Propose`` entry; populated by the
        # ``_advance_load_*`` helpers as they compute the decision.
        #
        # Shape: {
        #   "agg": str | None,            # agg mode single reason
        #   "prefill": str | None,        # disagg mode per-component
        #   "decode": str | None,
        #   "estimated_ttft_ms": float | None,  # max across engines
        #   "estimated_itl_ms": float | None,   # max across engines
        # }
        self._last_load_diagnostics: dict = self._empty_diagnostics()

    @staticmethod
    def _empty_diagnostics() -> dict:
        return {
            "agg": None,
            "prefill": None,
            "decode": None,
            "estimated_ttft_ms": None,
            "estimated_itl_ms": None,
        }

    def _set_reason(self, component: str, reason: str) -> None:
        """Record the classification the decision path took for
        ``component`` (``agg``/``prefill``/``decode``). Called at every
        decision branch so the adapter has a populated reason
        regardless of which early-return fired."""
        self._last_load_diagnostics[component] = reason

    def _record_estimate(self, metric: str, value_ms: float) -> None:
        """Track the max across engines for either ``estimated_ttft_ms``
        or ``estimated_itl_ms`` — matches PSM's ``_diag_*`` surface."""
        cur = self._last_load_diagnostics.get(metric)
        if cur is None or value_ms > cur:
            self._last_load_diagnostics[metric] = value_ms

    # ------------------------------------------------------------------
    # Tick priming (side-channel until FpmData decode lands)
    # ------------------------------------------------------------------

    def prime_tick(
        self,
        fpm_obs: Optional[FpmObservations],
        counts: Optional[WorkerCounts],
    ) -> None:
        self._cached_fpm_obs = fpm_obs
        self._cached_counts = counts

    def update_throughput_lower_bounds(self, p: int, d: int) -> None:
        """Legacy hook retained for unit tests. In the plugin chain the
        bounds flow through ``LocalPlannerOrchestrator.set_throughput_lower_bound``
        (written by throughput-propose, read here)."""
        self._orch.set_throughput_lower_bound("prefill", p)
        self._orch.set_throughput_lower_bound("decode", d)

    # Accessors via orchestrator (match PSM's ``_throughput_lower_bound_p/d``).
    @property
    def _throughput_lower_bound_p(self) -> int:
        return self._orch.get_throughput_lower_bound("prefill")

    @property
    def _throughput_lower_bound_d(self) -> int:
        return self._orch.get_throughput_lower_bound("decode")

    # ------------------------------------------------------------------
    # Helpers that mirror PSM state accessors
    # ------------------------------------------------------------------

    @property
    def _is_easy(self) -> bool:
        return self._config.optimization_target != "sla"

    def _num_workers(self, counts: WorkerCounts, component: str) -> int:
        """Mirror PSM ``self._num_p_workers`` / ``_num_d_workers``.
        Missing ``ready_num_*`` treated as 0 (PSM initialises to 0)."""
        if component == "prefill":
            return counts.ready_num_prefill or 0
        return counts.ready_num_decode or 0

    def _expected_mismatch(self, counts: WorkerCounts, component: str) -> bool:
        """Mirror PSM ``_scaling_in_progress``."""
        if component == "prefill":
            return (
                counts.expected_num_prefill is not None
                and counts.expected_num_prefill != self._num_workers(counts, "prefill")
            )
        return (
            counts.expected_num_decode is not None
            and counts.expected_num_decode != self._num_workers(counts, "decode")
        )

    # ------------------------------------------------------------------
    # Stage dispatch
    # ------------------------------------------------------------------

    async def Propose(
        self, request: ProposeStageRequest
    ) -> ProposeStageResponse:
        # PR 8 8-9: reset diagnostics at the start of every evaluation so
        # the adapter never reads a stale reason from the previous tick.
        self._last_load_diagnostics = self._empty_diagnostics()

        if not self._config.enable_load_scaling:
            self._set_reason(self._config.mode if self._config.mode != "disagg" else "prefill", "disabled")
            if self._config.mode == "disagg":
                self._set_reason("decode", "disabled")
            return _accept()
        if self._cached_fpm_obs is None or self._cached_counts is None:
            self._set_reason(self._config.mode if self._config.mode != "disagg" else "prefill", "no_fpm_data")
            if self._config.mode == "disagg":
                self._set_reason("decode", "no_fpm_data")
            return _accept()

        decision = self._advance_load(self._cached_fpm_obs, self._cached_counts)
        if decision is None:
            return _accept()

        targets: list[ComponentTarget] = []
        if decision.num_prefill is not None:
            targets.append(
                ComponentTarget(
                    sub_component_type="prefill",
                    replicas=decision.num_prefill,
                    type=OverrideType.SET,
                )
            )
        if decision.num_decode is not None:
            targets.append(
                ComponentTarget(
                    sub_component_type="decode",
                    replicas=decision.num_decode,
                    type=OverrideType.SET,
                )
            )
        if not targets:
            return _accept()
        return ProposeStageResponse(
            result_kind="override",
            override=OverrideResult(
                targets=targets,
                reason="builtin_load_propose",
            ),
        )

    # ------------------------------------------------------------------
    # Core algorithm (ports ``_advance_load`` family from PSM)
    # ------------------------------------------------------------------

    def _advance_load(
        self, obs: FpmObservations, counts: WorkerCounts
    ) -> Optional[ScalingDecision]:
        mode = self._config.mode
        if mode == "agg":
            return self._advance_load_agg(obs, counts)
        if mode == "disagg":
            return self._advance_load_disagg(obs, counts)
        return self._advance_load_single(obs, counts, mode)

    def _advance_load_single(
        self, obs: FpmObservations, counts: WorkerCounts, component: str
    ) -> Optional[ScalingDecision]:
        if self._expected_mismatch(counts, component):
            self._set_reason(component, "scaling_in_progress")
            return None

        fpm_stats = obs.prefill if component == "prefill" else obs.decode
        num_workers = self._num_workers(counts, component)

        if not fpm_stats:
            self._set_reason(component, "no_fpm_data")
            return None
        if not self._reconcile_fpm_worker_count(fpm_stats, num_workers, component):
            self._set_reason(component, "worker_count_mismatch")
            return None

        if self._is_easy:
            desired = (
                self._prefill_easy_decision(fpm_stats, num_workers)
                if component == "prefill"
                else self._decode_easy_decision(fpm_stats, num_workers)
            )
        else:
            desired = (
                self._prefill_load_decision(fpm_stats, num_workers)
                if component == "prefill"
                else self._decode_load_decision(fpm_stats, num_workers)
            )
        if desired is None:
            self._set_reason(component, "insufficient_data")
            return None

        if self._config.enable_throughput_scaling:
            bound = (
                self._throughput_lower_bound_p
                if component == "prefill"
                else self._throughput_lower_bound_d
            )
            desired = max(desired, bound)

        desired = self._apply_single_budget(desired, component)
        self._set_reason(
            component,
            "no_change" if desired == num_workers
            else "scale_up" if desired > num_workers
            else "scale_down",
        )
        return (
            ScalingDecision(num_prefill=desired)
            if component == "prefill"
            else ScalingDecision(num_decode=desired)
        )

    def _advance_load_disagg(
        self, obs: FpmObservations, counts: WorkerCounts
    ) -> Optional[ScalingDecision]:
        p_stats, d_stats = obs.prefill, obs.decode
        num_p = self._num_workers(counts, "prefill")
        num_d = self._num_workers(counts, "decode")

        if not p_stats and not d_stats:
            self._set_reason("prefill", "no_fpm_data")
            self._set_reason("decode", "no_fpm_data")
            return None
        if p_stats and not self._reconcile_fpm_worker_count(p_stats, num_p, "prefill"):
            self._set_reason("prefill", "worker_count_mismatch")
            self._set_reason("decode", "worker_count_mismatch")
            return None
        if d_stats and not self._reconcile_fpm_worker_count(d_stats, num_d, "decode"):
            self._set_reason("prefill", "worker_count_mismatch")
            self._set_reason("decode", "worker_count_mismatch")
            return None

        p_desired: Optional[int]
        d_desired: Optional[int]
        if self._is_easy:
            p_desired = (
                self._prefill_easy_decision(p_stats, num_p) if p_stats else None
            )
            d_desired = (
                self._decode_easy_decision(d_stats, num_d) if d_stats else None
            )
        else:
            p_desired = (
                self._prefill_load_decision(p_stats, num_p) if p_stats else None
            )
            d_desired = (
                self._decode_load_decision(d_stats, num_d) if d_stats else None
            )

        final_p = p_desired if p_desired is not None else num_p
        final_d = d_desired if d_desired is not None else num_d

        if self._config.enable_throughput_scaling:
            final_p = max(final_p, self._throughput_lower_bound_p)
            final_d = max(final_d, self._throughput_lower_bound_d)

        final_p = max(final_p, self._config.min_endpoint)
        final_d = max(final_d, self._config.min_endpoint)
        final_p, final_d = self._apply_global_budget(final_p, final_d)

        # Per-component reason: classify each direction independently,
        # otherwise a dashboard can't tell "prefill is fine, decode
        # needs to scale" apart from "both need to scale".
        self._set_reason(
            "prefill",
            "insufficient_data"
            if p_desired is None
            else "no_change"
            if final_p == num_p
            else "scale_up"
            if final_p > num_p
            else "scale_down",
        )
        self._set_reason(
            "decode",
            "insufficient_data"
            if d_desired is None
            else "no_change"
            if final_d == num_d
            else "scale_up"
            if final_d > num_d
            else "scale_down",
        )

        if final_p == num_p and final_d == num_d:
            return None
        return ScalingDecision(num_prefill=final_p, num_decode=final_d)

    def _advance_load_agg(
        self, obs: FpmObservations, counts: WorkerCounts
    ) -> Optional[ScalingDecision]:
        fpm_stats = obs.decode
        if not fpm_stats:
            self._set_reason("agg", "no_fpm_data")
            return None
        num_workers = self._num_workers(counts, "decode")

        if self._expected_mismatch(counts, "decode"):
            self._set_reason("agg", "scaling_in_progress")
            return None
        if not self._reconcile_fpm_worker_count(fpm_stats, num_workers, "agg"):
            self._set_reason("agg", "worker_count_mismatch")
            return None

        if self._is_easy:
            desired = self._agg_easy_decision(fpm_stats, num_workers)
            if desired is None:
                self._set_reason("agg", "insufficient_data")
                return None
            desired = max(desired, self._config.min_endpoint)
            if self._config.enable_throughput_scaling:
                desired = max(desired, self._throughput_lower_bound_d)
            desired = self._apply_single_budget(desired, "decode")
            self._set_reason(
                "agg",
                "no_change"
                if desired == num_workers
                else "scale_up"
                if desired > num_workers
                else "scale_down",
            )
            return ScalingDecision(num_decode=desired)

        agg_reg = self.get_regression("agg")
        if agg_reg is None or not agg_reg.has_sufficient_data():
            self._set_reason("agg", "insufficient_data")
            return None

        caps = self._orch.capabilities
        d_caps = caps.decode if caps is not None else None
        max_tokens = d_caps.max_num_batched_tokens if d_caps else None
        if not max_tokens or max_tokens <= 0:
            p_desired: Optional[int] = None
        else:
            p_desired = self._agg_prefill_scaling(fpm_stats, num_workers, max_tokens)
        d_desired = self._agg_decode_scaling(fpm_stats, num_workers)

        if p_desired is not None and p_desired > num_workers:
            desired = p_desired
        elif d_desired is not None and d_desired > num_workers:
            desired = d_desired
        elif p_desired is None and d_desired is not None and d_desired < num_workers:
            desired = d_desired
        elif (
            p_desired is not None
            and p_desired < num_workers
            and d_desired is not None
            and d_desired < num_workers
        ):
            desired = max(p_desired, d_desired)
        else:
            desired = num_workers

        desired = max(desired, self._config.min_endpoint)
        if self._config.enable_throughput_scaling:
            desired = max(desired, self._throughput_lower_bound_d)
        desired = self._apply_single_budget(desired, "decode")

        self._set_reason(
            "agg",
            "no_change"
            if desired == num_workers
            else "scale_up"
            if desired > num_workers
            else "scale_down",
        )
        if desired == num_workers:
            return None
        return ScalingDecision(num_decode=desired)

    # ------------------------------------------------------------------
    # SLA decision methods (ported verbatim modulo _diag_* side effects)
    # ------------------------------------------------------------------

    def _prefill_load_decision(self, fpm_stats, num_workers):
        p_reg = self.get_regression("prefill")
        if p_reg is None or not p_reg.has_sufficient_data():
            return None
        if num_workers == 0:
            return None
        caps = self._orch.capabilities
        p_caps = caps.prefill if caps is not None else None
        max_tokens = p_caps.max_num_batched_tokens if p_caps else None
        if not max_tokens or max_tokens <= 0:
            return None

        estimates: list[float] = []
        for _key, fpm in fpm_stats.items():
            est = p_reg.estimate_next_ttft(
                queued_prefill_tokens=fpm.queued_requests.sum_prefill_tokens,
                max_num_batched_tokens=max_tokens,
            )
            if est is not None:
                ttft_ms = est * 1000
                estimates.append(ttft_ms)
                self._record_estimate("estimated_ttft_ms", ttft_ms)

        return self._scale_decision(estimates, self._config.ttft, num_workers)

    def _decode_load_decision(self, fpm_stats, num_workers):
        d_reg = self.get_regression("decode")
        if d_reg is None or not d_reg.has_sufficient_data():
            return None
        if num_workers == 0:
            return None

        estimates: list[float] = []
        for _key, fpm in fpm_stats.items():
            est = d_reg.estimate_next_itl(
                scheduled_decode_kv=fpm.scheduled_requests.sum_decode_kv_tokens,
                queued_decode_kv=fpm.queued_requests.sum_decode_kv_tokens,
            )
            if est is not None:
                itl_ms = est * 1000
                estimates.append(itl_ms)
                self._record_estimate("estimated_itl_ms", itl_ms)

        return self._scale_decision(estimates, self._config.itl, num_workers)

    def _agg_prefill_scaling(self, fpm_stats, num_workers, max_tokens):
        agg_reg = self.get_regression("agg")
        assert agg_reg is not None
        estimates: list[float] = []
        for fpm in fpm_stats.values():
            est = agg_reg.estimate_next_ttft(
                queued_prefill_tokens=fpm.queued_requests.sum_prefill_tokens,
                max_num_batched_tokens=max_tokens,
                current_decode_kv=fpm.scheduled_requests.sum_decode_kv_tokens,
            )
            if est is not None:
                estimates.append(est * 1000)
        return self._scale_decision(estimates, self._config.ttft, num_workers)

    def _agg_decode_scaling(self, fpm_stats, num_workers):
        agg_reg = self.get_regression("agg")
        assert agg_reg is not None
        estimates: list[float] = []
        for fpm in fpm_stats.values():
            est = agg_reg.estimate_next_itl(
                scheduled_decode_kv=fpm.scheduled_requests.sum_decode_kv_tokens,
                queued_decode_kv=fpm.queued_requests.sum_decode_kv_tokens,
            )
            if est is not None:
                estimates.append(est * 1000)
        return self._scale_decision(estimates, self._config.itl, num_workers)

    def _scale_decision(
        self, estimates: list[float], sla: float, num_workers: int
    ) -> Optional[int]:
        if not estimates:
            return None
        sensitivity = self._config.load_scaling_down_sensitivity / 100.0
        if all(t > sla for t in estimates):
            return num_workers + 1
        if num_workers > 1:
            threshold = sla * sensitivity
            if all(t < threshold for t in estimates):
                return max(num_workers - 1, self._config.min_endpoint)
        return None

    # ------------------------------------------------------------------
    # Easy-mode decisions
    # ------------------------------------------------------------------

    def _prefill_easy_decision(self, fpm_stats, num_workers):
        caps = self._orch.capabilities
        p_caps = caps.prefill if caps is not None else None
        ctx_len = p_caps.context_length if p_caps else None
        if not ctx_len or ctx_len <= 0 or num_workers == 0:
            return None

        is_latency = self._config.optimization_target == "latency"
        up_thresh = (
            _PREFILL_LATENCY_SCALE_UP if is_latency else _PREFILL_THROUGHPUT_SCALE_UP
        )
        down_thresh = (
            _PREFILL_LATENCY_SCALE_DOWN
            if is_latency
            else _PREFILL_THROUGHPUT_SCALE_DOWN
        )

        ratios = [
            fpm.queued_requests.sum_prefill_tokens / ctx_len
            for fpm in fpm_stats.values()
        ]
        if not ratios:
            return None
        if any(r >= up_thresh for r in ratios):
            return num_workers + 1
        if num_workers > 1:
            if is_latency:
                if all(r <= down_thresh for r in ratios):
                    return max(num_workers - 1, self._config.min_endpoint)
            else:
                if all(r < down_thresh for r in ratios):
                    return max(num_workers - 1, self._config.min_endpoint)
        return None

    def _decode_easy_decision(self, fpm_stats, num_workers):
        caps = self._orch.capabilities
        d_caps = caps.decode if caps is not None else None
        max_kv = d_caps.max_kv_tokens if d_caps else None
        if not max_kv or max_kv <= 0 or num_workers == 0:
            return None

        is_latency = self._config.optimization_target == "latency"
        up_thresh = (
            _DECODE_LATENCY_SCALE_UP if is_latency else _DECODE_THROUGHPUT_SCALE_UP
        )
        down_thresh = (
            _DECODE_LATENCY_SCALE_DOWN
            if is_latency
            else _DECODE_THROUGHPUT_SCALE_DOWN
        )

        utils = [
            (
                fpm.scheduled_requests.sum_decode_kv_tokens
                + fpm.queued_requests.sum_decode_kv_tokens
            )
            / max_kv
            for fpm in fpm_stats.values()
        ]
        if not utils:
            return None
        if any(u > up_thresh for u in utils):
            return num_workers + 1
        if num_workers > 1 and all(u < down_thresh for u in utils):
            return max(num_workers - 1, self._config.min_endpoint)
        return None

    def _agg_easy_decision(self, fpm_stats, num_workers):
        caps = self._orch.capabilities
        d_caps = caps.decode if caps is not None else None
        max_kv = d_caps.max_kv_tokens if d_caps else None
        if not max_kv or max_kv <= 0 or num_workers == 0:
            return None

        is_latency = self._config.optimization_target == "latency"
        up_thresh = (
            _DECODE_LATENCY_SCALE_UP if is_latency else _DECODE_THROUGHPUT_SCALE_UP
        )
        down_thresh = (
            _DECODE_LATENCY_SCALE_DOWN
            if is_latency
            else _DECODE_THROUGHPUT_SCALE_DOWN
        )

        utils = [
            (
                fpm.scheduled_requests.sum_decode_kv_tokens
                + fpm.queued_requests.sum_decode_kv_tokens
                + fpm.queued_requests.sum_prefill_tokens
            )
            / max_kv
            for fpm in fpm_stats.values()
        ]
        if not utils:
            return None
        if any(u > up_thresh for u in utils):
            return num_workers + 1
        if num_workers > 1 and all(u < down_thresh for u in utils):
            return max(num_workers - 1, self._config.min_endpoint)
        return None

    # ------------------------------------------------------------------
    # Budget (ported from PSM; 6-6 will move these to CONSTRAIN stage)
    # ------------------------------------------------------------------

    def _apply_single_budget(self, desired: int, component: str) -> int:
        caps = self._orch.capabilities
        engine_caps = (
            caps.prefill if (component == "prefill" and caps) else (caps.decode if caps else None)
        )
        gpu = engine_caps.num_gpu if engine_caps else None
        if gpu is None:
            return desired
        return self._budget_clamp(max(desired, self._config.min_endpoint), gpu)

    def _apply_global_budget(self, num_p: int, num_d: int) -> tuple[int, int]:
        budget = self._config.max_gpu_budget
        caps = self._orch.capabilities
        p_gpu = caps.prefill.num_gpu if (caps and caps.prefill) else None
        d_gpu = caps.decode.num_gpu if (caps and caps.decode) else None
        if budget < 0 or p_gpu is None or d_gpu is None:
            return num_p, num_d
        total = num_p * p_gpu + num_d * d_gpu
        if total <= budget:
            return num_p, num_d
        min_req = self._config.min_endpoint * p_gpu + self._config.min_endpoint * d_gpu
        if budget < min_req:
            return 0, 0
        scale = budget / total
        max_p = math.floor((budget - self._config.min_endpoint * d_gpu) / p_gpu)
        num_p = max(self._config.min_endpoint, min(max_p, math.floor(num_p * scale)))
        remaining = budget - num_p * p_gpu
        num_d = max(self._config.min_endpoint, math.floor(remaining / d_gpu))
        return num_p, num_d

    def _budget_clamp(self, desired: int, engine_gpu: int) -> int:
        budget = self._config.max_gpu_budget
        if budget < 0:
            return desired
        total = desired * engine_gpu
        if total <= budget:
            return desired
        min_req = self._config.min_endpoint * engine_gpu
        if budget < min_req:
            return 0
        return max(self._config.min_endpoint, math.floor(budget / engine_gpu))

    # ------------------------------------------------------------------
    # FPM worker-count reconciliation (ported verbatim)
    # ------------------------------------------------------------------

    @staticmethod
    def _reconcile_fpm_worker_count(
        fpm_stats: dict, dgd_count: int, label: str
    ) -> bool:
        workers_to_dp: dict[str, set[int]] = {}
        for wid, dp in fpm_stats:
            workers_to_dp.setdefault(wid, set()).add(dp)
        if len(workers_to_dp) != dgd_count:
            return False
        dp_sizes = {len(dps) for dps in workers_to_dp.values()}
        if len(dp_sizes) > 1:
            return False
        dp_size = dp_sizes.pop() if dp_sizes else 1
        if len(fpm_stats) != dgd_count * dp_size:
            return False
        return True


__all__ = ["BuiltinLoadPropose"]
