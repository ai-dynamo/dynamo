# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure discrete-event state machine for planner scaling decisions.

``PlannerStateMachine`` receives events (``ScheduledTick`` + ``TickInput``),
updates internal state (regression models, load predictors, worker inventory),
and returns effects (``PlannerEffects``: optional scaling decision + next tick).

This module contains **zero I/O** -- no runtime, connector, subscriber, asyncio,
or Prometheus dependencies.  All external interaction is done by the adapter
layer (``NativePlannerBase`` and its subclasses) which feeds data in and
applies decisions out.

Load-based scaling logic lives in ``load_scaling.py``.
Throughput-based scaling logic lives in ``throughput_scaling.py``.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Optional

from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.load.predictors import LOAD_PREDICTORS
from dynamo.planner.core.load_scaling import LoadScalingMixin
from dynamo.planner.core.perf_model import (
    AggRegressionModel,
    DecodeRegressionModel,
    PrefillRegressionModel,
)
from dynamo.planner.core.throughput_scaling import ThroughputScalingMixin
from dynamo.planner.core.types import (
    FpmObservations,
    PlannerEffects,
    ScheduledTick,
    TickDiagnostics,
    TickInput,
    TrafficObservation,
    WorkerCapabilities,
    WorkerCounts,
)

if TYPE_CHECKING:
    from dynamo.common.forward_pass_metrics import ForwardPassMetrics

logger = logging.getLogger(__name__)


class PlannerStateMachine(LoadScalingMixin, ThroughputScalingMixin):
    """Discrete-event state machine for all planner modes.

    Owns regression models, load predictors, throughput lower bounds,
    and all scaling decision logic.  Receives events, returns effects.
    Has no runtime dependencies.
    """

    def __init__(
        self,
        config: PlannerConfig,
        capabilities: Optional[WorkerCapabilities] = None,
    ) -> None:
        self._config = config
        self._capabilities = capabilities or WorkerCapabilities()

        self._is_agg = config.mode == "agg"
        self._has_prefill = config.mode in ("disagg", "prefill")
        self._has_decode = config.mode in ("disagg", "decode", "agg")
        self._is_easy = config.optimization_target != "sla"

        # Easy mode uses static thresholds -- no regression or predictors needed
        if not self._is_easy:
            if self._is_agg:
                self._agg_regression = AggRegressionModel(
                    max_num_fpm_samples=config.max_num_fpm_samples,
                    min_observations=config.load_min_observations,
                    bucket_count=config.fpm_sample_bucket_size,
                )
            else:
                if self._has_prefill:
                    self._prefill_regression = PrefillRegressionModel(
                        max_num_fpm_samples=config.max_num_fpm_samples,
                        min_observations=config.load_min_observations,
                        bucket_count=config.fpm_sample_bucket_size,
                    )
                if self._has_decode:
                    self._decode_regression = DecodeRegressionModel(
                        max_num_fpm_samples=config.max_num_fpm_samples,
                        min_observations=config.load_min_observations,
                        bucket_count=config.fpm_sample_bucket_size,
                    )

            predictor_cls = LOAD_PREDICTORS[config.load_predictor]
            self._num_req_predictor = predictor_cls(config)
            self._isl_predictor = predictor_cls(config)
            self._osl_predictor = predictor_cls(config)

        self._num_p_workers: int = 0
        self._num_d_workers: int = 0
        self._expected_num_p: Optional[int] = None
        self._expected_num_d: Optional[int] = None

        self._throughput_lower_bound_p: int = 1
        self._throughput_lower_bound_d: int = 1

        self._next_load_s: float = float("inf")
        self._next_throughput_s: float = float("inf")

        # Diagnostics scratch fields populated by mixins, read by on_tick
        self._diag_estimated_ttft_ms: Optional[float] = None
        self._diag_estimated_itl_ms: Optional[float] = None
        self._diag_predicted_num_req: Optional[float] = None
        self._diag_predicted_isl: Optional[float] = None
        self._diag_predicted_osl: Optional[float] = None
        self._diag_engine_rps_prefill: Optional[float] = None
        self._diag_engine_rps_decode: Optional[float] = None
        self._diag_load_reason: Optional[str] = None
        self._diag_throughput_reason: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_capabilities(self, capabilities: WorkerCapabilities) -> None:
        """Replace the current worker capabilities."""
        self._capabilities = capabilities

    def initial_tick(self, start_s: float) -> ScheduledTick:
        self._next_load_s = start_s + self._config.load_adjustment_interval
        if self._config.enable_throughput_scaling:
            self._next_throughput_s = (
                start_s + self._config.throughput_adjustment_interval
            )
        return self._next_scheduled_tick()

    def load_benchmark_fpms(
        self,
        prefill_fpms: Optional[list[ForwardPassMetrics]] = None,
        decode_fpms: Optional[list[ForwardPassMetrics]] = None,
        agg_fpms: Optional[list[ForwardPassMetrics]] = None,
    ) -> None:
        if self._is_easy:
            logger.debug("Skipping benchmark FPM loading in easy mode")
            return
        if agg_fpms and self._is_agg:
            self._agg_regression.load_benchmark_fpms(agg_fpms)
            logger.info(f"Bootstrapped agg regression with {len(agg_fpms)} FPMs")
        if prefill_fpms and self._has_prefill and not self._is_agg:
            self._prefill_regression.load_benchmark_fpms(prefill_fpms)
            logger.info(
                f"Bootstrapped prefill regression with {len(prefill_fpms)} FPMs"
            )
        if decode_fpms and self._has_decode and not self._is_agg:
            self._decode_regression.load_benchmark_fpms(decode_fpms)
            logger.info(f"Bootstrapped decode regression with {len(decode_fpms)} FPMs")

    def warm_load_predictors(self, observations: list[TrafficObservation]) -> None:
        if self._is_easy:
            logger.debug("Skipping load predictor warmup in easy mode")
            return
        for obs in observations:
            self._num_req_predictor.add_data_point(obs.num_req)
            self._isl_predictor.add_data_point(obs.isl)
            self._osl_predictor.add_data_point(obs.osl)
        logger.info(f"Warmed load predictors with {len(observations)} intervals")
        for p in (self._num_req_predictor, self._isl_predictor, self._osl_predictor):
            if hasattr(p, "reset_idle_skip"):
                p.reset_idle_skip()

    def on_tick(self, tick: ScheduledTick, tick_input: TickInput) -> PlannerEffects:
        effects = PlannerEffects()
        self._reset_diag()

        if tick_input.worker_counts is not None:
            self._update_inventory(tick_input.worker_counts)

        if tick.run_load_scaling:
            if tick_input.fpm_observations is not None:
                if not self._is_easy:
                    self._observe_fpm(tick_input.fpm_observations)
                load_decision = self._advance_load(tick_input.fpm_observations)
                if load_decision is not None:
                    effects.scale_to = load_decision
            self._next_load_s = tick_input.now_s + self._config.load_adjustment_interval

        if tick.run_throughput_scaling:
            if tick_input.traffic is not None:
                self._observe_traffic(tick_input.traffic)
                throughput_decision = self._advance_throughput(tick_input.traffic)
                if throughput_decision is not None:
                    if effects.scale_to is None:
                        effects.scale_to = throughput_decision
            self._next_throughput_s = (
                tick_input.now_s + self._config.throughput_adjustment_interval
            )

        effects.diagnostics = self._build_diagnostics()
        effects.next_tick = self._next_scheduled_tick()
        return effects

    def _reset_diag(self) -> None:
        self._diag_estimated_ttft_ms = None
        self._diag_estimated_itl_ms = None
        self._diag_predicted_num_req = None
        self._diag_predicted_isl = None
        self._diag_predicted_osl = None
        self._diag_engine_rps_prefill = None
        self._diag_engine_rps_decode = None
        self._diag_load_reason = None
        self._diag_throughput_reason = None

    def _build_diagnostics(self) -> TickDiagnostics:
        return TickDiagnostics(
            estimated_ttft_ms=self._diag_estimated_ttft_ms,
            estimated_itl_ms=self._diag_estimated_itl_ms,
            predicted_num_req=self._diag_predicted_num_req,
            predicted_isl=self._diag_predicted_isl,
            predicted_osl=self._diag_predicted_osl,
            engine_rps_prefill=self._diag_engine_rps_prefill,
            engine_rps_decode=self._diag_engine_rps_decode,
            load_decision_reason=self._diag_load_reason,
            throughput_decision_reason=self._diag_throughput_reason,
        )

    # ------------------------------------------------------------------
    # Tick scheduling
    # ------------------------------------------------------------------

    _MERGE_TOLERANCE_S = 0.5

    def _next_scheduled_tick(self) -> ScheduledTick:
        """Build the single next tick, merging cadences if they coincide."""
        at_s = min(self._next_load_s, self._next_throughput_s)
        is_load = self._next_load_s <= at_s + self._MERGE_TOLERANCE_S
        is_throughput = self._next_throughput_s <= at_s + self._MERGE_TOLERANCE_S
        return ScheduledTick(
            at_s=at_s,
            run_load_scaling=is_load,
            run_throughput_scaling=is_throughput,
            need_worker_states=True,
            need_worker_fpm=is_load,
            need_traffic_metrics=is_throughput,
            traffic_metrics_duration_s=(
                self._config.throughput_adjustment_interval if is_throughput else 0.0
            ),
        )

    # ------------------------------------------------------------------
    # Inventory
    # ------------------------------------------------------------------

    def _update_inventory(self, counts: WorkerCounts) -> None:
        if counts.ready_num_prefill is not None:
            self._num_p_workers = counts.ready_num_prefill
        if counts.ready_num_decode is not None:
            self._num_d_workers = counts.ready_num_decode
        self._expected_num_p = counts.expected_num_prefill
        self._expected_num_d = counts.expected_num_decode

    def _scaling_in_progress(self, component: str) -> bool:
        if component == "prefill":
            return (
                self._expected_num_p is not None
                and self._expected_num_p != self._num_p_workers
            )
        return (
            self._expected_num_d is not None
            and self._expected_num_d != self._num_d_workers
        )

    # ------------------------------------------------------------------
    # FPM / traffic observation
    # ------------------------------------------------------------------

    def _observe_fpm(self, obs: FpmObservations) -> None:
        if self._is_agg:
            if obs.decode:
                for fpm in obs.decode.values():
                    self._agg_regression.add_observation(fpm)
                logger.info(f"FPM load stats: {len(obs.decode)} agg engines observed")
            return

        if obs.prefill and self._has_prefill:
            for fpm in obs.prefill.values():
                self._prefill_regression.add_observation(fpm)
            logger.info(f"FPM load stats: {len(obs.prefill)} prefill engines observed")
        if obs.decode and self._has_decode:
            for fpm in obs.decode.values():
                self._decode_regression.add_observation(fpm)
            logger.info(f"FPM load stats: {len(obs.decode)} decode engines observed")

    def _observe_traffic(self, traffic: TrafficObservation) -> None:
        self._num_req_predictor.add_data_point(traffic.num_req)
        self._isl_predictor.add_data_point(traffic.isl)
        self._osl_predictor.add_data_point(traffic.osl)

    # ------------------------------------------------------------------
    # Budget
    # ------------------------------------------------------------------

    def _apply_single_budget(self, desired: int, component: str) -> int:
        caps = (
            self._capabilities.prefill
            if component == "prefill"
            else self._capabilities.decode
        )
        gpu = caps.num_gpu if caps else None
        if gpu is None:
            return desired
        return self._budget_clamp(max(desired, self._config.min_endpoint), gpu)

    def _apply_global_budget(self, num_p: int, num_d: int) -> tuple[int, int]:
        budget = self._config.max_gpu_budget
        p_gpu = (
            self._capabilities.prefill.num_gpu if self._capabilities.prefill else None
        )
        d_gpu = self._capabilities.decode.num_gpu if self._capabilities.decode else None
        if budget < 0 or p_gpu is None or d_gpu is None:
            return num_p, num_d
        total = num_p * p_gpu + num_d * d_gpu
        if total <= budget:
            return num_p, num_d
        min_req = self._config.min_endpoint * p_gpu + self._config.min_endpoint * d_gpu
        if budget < min_req:
            logger.warning(
                f"max_gpu_budget ({budget}) below min ({min_req}); zero replicas"
            )
            return 0, 0
        scale = budget / total
        max_p = math.floor((budget - self._config.min_endpoint * d_gpu) / p_gpu)
        num_p = max(self._config.min_endpoint, min(max_p, math.floor(num_p * scale)))
        remaining = budget - num_p * p_gpu
        num_d = max(self._config.min_endpoint, math.floor(remaining / d_gpu))
        logger.warning(f"GPUs ({total}) > budget ({budget}), -> {num_p}P + {num_d}D")
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
            logger.warning(
                f"max_gpu_budget ({budget}) below min ({min_req}); zero replicas"
            )
            return 0
        result = max(self._config.min_endpoint, math.floor(budget / engine_gpu))
        logger.warning(f"GPUs ({total}) > budget ({budget}), -> {result} replicas")
        return result

    # ------------------------------------------------------------------
    # FPM / worker count reconciliation
    # ------------------------------------------------------------------

    @staticmethod
    def _reconcile_fpm_worker_count(
        fpm_stats: dict[tuple[str, int], ForwardPassMetrics], dgd_count: int, label: str
    ) -> bool:
        workers_to_dp: dict[str, set[int]] = {}
        for wid, dp in fpm_stats:
            workers_to_dp.setdefault(wid, set()).add(dp)

        if len(workers_to_dp) != dgd_count:
            logger.warning(
                f"Worker count mismatch: DGD={dgd_count}, FPM={len(workers_to_dp)} for {label}"
            )
            return False

        dp_sizes = {len(dps) for dps in workers_to_dp.values()}
        if len(dp_sizes) > 1:
            logger.warning(f"Inconsistent DP ranks for {label}: {dict(workers_to_dp)}")
            return False

        dp_size = dp_sizes.pop() if dp_sizes else 1
        if len(fpm_stats) != dgd_count * dp_size:
            logger.warning(
                f"Incomplete FPM coverage for {label}: expected {dgd_count}x{dp_size}, got {len(fpm_stats)}"
            )
            return False
        return True

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def prefill_regression(self) -> PrefillRegressionModel:
        if not self._has_prefill:
            raise AttributeError(f"No prefill regression in mode={self._config.mode}")
        return self._prefill_regression

    @property
    def decode_regression(self) -> DecodeRegressionModel:
        if not self._has_decode or self._is_agg:
            raise AttributeError(f"No decode regression in mode={self._config.mode}")
        return self._decode_regression

    @property
    def agg_regression(self) -> AggRegressionModel:
        if not self._is_agg:
            raise AttributeError(f"No agg regression in mode={self._config.mode}")
        return self._agg_regression

    @property
    def regression(self) -> AggRegressionModel:
        return self.agg_regression
