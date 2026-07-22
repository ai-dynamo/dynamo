# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared builtin planner state and algorithm helpers.

``PlannerScalingState`` owns perf models, worker inventory, throughput floors,
runtime metadata, and the load/throughput scaling calculations used by the
builtin plugin bundle.

This module contains **zero I/O** -- no runtime, connector, subscriber, asyncio,
or Prometheus dependencies.  All external interaction is done by the adapter
layer, which feeds observations into the plugin pipeline and applies decisions
out.

Load-based scaling logic lives in ``load_scaling.py``.
Throughput-based scaling logic lives in ``throughput_scaling.py``.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Optional

from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.budget import (
    proportional_clamp_pair,
    proportional_clamp_single,
)
from dynamo.planner.core.load_scaling import LoadScalingMixin
from dynamo.planner.core.perf_model import PlannerEnginePerfModel
from dynamo.planner.core.throughput_scaling import ThroughputScalingMixin
from dynamo.planner.core.types import (
    FpmObservations,
    ScalingDecision,
    TickDiagnostics,
    TrafficObservation,
    WorkerCapabilities,
    WorkerCounts,
)

if TYPE_CHECKING:
    from dynamo.common.forward_pass_metrics import ForwardPassMetrics

logger = logging.getLogger(__name__)


class PlannerScalingState(LoadScalingMixin, ThroughputScalingMixin):
    """Shared in-memory scaling state for all planner modes.

    Owns perf models, throughput lower bounds, worker inventory,
    last-value runtime metadata, and all scaling decision logic. It
    deliberately has no runtime dependencies. Load prediction state
    lives in the builtin PREDICT plugin and is passed in explicitly.

    Builtin orchestrator plugins use this class directly as their private
    shared core while the remaining cross-plugin state is being split into
    explicit pipeline artifacts.
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

        # Easy mode uses static thresholds -- no perf models needed.
        if not self._is_easy:
            if self._is_agg:
                self._agg_regression = PlannerEnginePerfModel(
                    worker_type="aggregated",
                    config=config,
                    capabilities=self._capabilities.decode,
                )
            else:
                if self._has_prefill:
                    self._prefill_regression = PlannerEnginePerfModel(
                        worker_type="prefill",
                        config=config,
                        capabilities=self._capabilities.prefill,
                    )
                if self._has_decode:
                    self._decode_regression = PlannerEnginePerfModel(
                        worker_type="decode",
                        config=config,
                        capabilities=self._capabilities.decode,
                    )
        self._num_p_workers: int = 0
        self._num_d_workers: int = 0
        self._expected_num_p: Optional[int] = None
        self._expected_num_d: Optional[int] = None
        self._prefill_scaling_in_progress: bool = False
        self._decode_scaling_in_progress: bool = False

        self._throughput_lower_bound_p: int = 1
        self._throughput_lower_bound_d: int = 1

        # Most recent observed KV hit rate from the router. Runtime metadata like
        # this is intentionally last-value only, not fed through the traffic load
        # predictor. ``None`` means "no observation yet" -> no discount.
        self._last_kv_hit_rate: Optional[float] = None
        # Most recent speculative decode accept length. This uses last-value
        # semantics as runtime metadata; FPM observations remain raw per-forward
        # data and cold start/no observation falls back to 1.0.
        self._last_accept_length: float = 1.0

        # Diagnostics scratch fields populated by mixins and read by adapters.
        self._diag_estimated_ttft_ms: Optional[float] = None
        self._diag_estimated_itl_ms: Optional[float] = None
        self._diag_predicted_num_req: Optional[float] = None
        self._diag_predicted_isl: Optional[float] = None
        self._diag_predicted_osl: Optional[float] = None
        self._diag_predicted_kv_hit_rate: Optional[float] = None
        self._diag_engine_rps_prefill: Optional[float] = None
        self._diag_engine_rps_decode: Optional[float] = None
        self._diag_load_reason: Optional[str] = None
        self._diag_throughput_reason: Optional[str] = None
        self._diag_load_reason_prefill: Optional[str] = None
        self._diag_load_reason_decode: Optional[str] = None
        self._diag_throughput_reason_prefill: Optional[str] = None
        self._diag_throughput_reason_decode: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_capabilities(self, capabilities: WorkerCapabilities) -> None:
        """Replace the current worker capabilities."""
        self._capabilities = capabilities
        self._last_accept_length = self._clamp_accept_length(self._last_accept_length)
        if self._is_easy:
            return
        if self._is_agg and hasattr(self, "_agg_regression"):
            self._agg_regression.update_capabilities(self._capabilities.decode)
        if (
            self._has_prefill
            and not self._is_agg
            and hasattr(self, "_prefill_regression")
        ):
            self._prefill_regression.update_capabilities(self._capabilities.prefill)
        if (
            self._has_decode
            and not self._is_agg
            and hasattr(self, "_decode_regression")
        ):
            self._decode_regression.update_capabilities(self._capabilities.decode)

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
            logger.info(f"Bootstrapped agg perf model with {len(agg_fpms)} FPMs")
        if prefill_fpms and self._has_prefill and not self._is_agg:
            self._prefill_regression.load_benchmark_fpms(prefill_fpms)
            logger.info(
                f"Bootstrapped prefill perf model with {len(prefill_fpms)} FPMs"
            )
        if decode_fpms and self._has_decode and not self._is_agg:
            self._decode_regression.load_benchmark_fpms(decode_fpms)
            logger.info(f"Bootstrapped decode perf model with {len(decode_fpms)} FPMs")

    def begin_tick(self) -> None:
        """Reset per-tick diagnostics before builtin plugins run."""
        self._reset_diag()

    def observe_worker_counts(self, counts: WorkerCounts) -> None:
        self._update_inventory(counts)

    def observe_fpm(self, obs: FpmObservations) -> None:
        if self._is_easy:
            return
        self._observe_fpm(obs)

    def observe_runtime_metadata(
        self,
        *,
        kv_hit_rate: Optional[float] = None,
        accept_length: Optional[float] = None,
    ) -> None:
        """Update last-value runtime metadata without touching prediction history."""
        if kv_hit_rate is not None and not math.isnan(kv_hit_rate):
            self._last_kv_hit_rate = kv_hit_rate
        self._observe_accept_length(accept_length)

    def install_regressions(
        self,
        *,
        prefill: Optional[PlannerEnginePerfModel] = None,
        decode: Optional[PlannerEnginePerfModel] = None,
        agg: Optional[PlannerEnginePerfModel] = None,
    ) -> None:
        if prefill is not None:
            self._prefill_regression = prefill
        if decode is not None:
            self._decode_regression = decode
        if agg is not None:
            self._agg_regression = agg

    def advance_load(
        self,
        obs: FpmObservations,
        *,
        predicted_kv_hit_rate: Optional[float] = None,
        predicted_accept_length: Optional[float] = None,
    ) -> Optional[ScalingDecision]:
        self.observe_runtime_metadata(
            kv_hit_rate=predicted_kv_hit_rate,
            accept_length=predicted_accept_length,
        )
        return self._advance_load(obs)

    def advance_throughput_from_prediction(
        self,
        traffic: TrafficObservation,
        *,
        predicted_num_req: Optional[float],
        predicted_isl: Optional[float],
        predicted_osl: Optional[float],
        predicted_kv_hit_rate: Optional[float],
        predicted_accept_length: Optional[float] = None,
    ) -> Optional[ScalingDecision]:
        """Run the throughput decision using PREDICT-stage output.

        The PREDICT plugin owns load prediction history. This method consumes
        only explicit prediction output so PROPOSE never re-runs prediction or
        depends on hidden predictor state.
        """
        if not self._config.enable_throughput_scaling:
            self._diag_throughput_reason = "disabled"
            return None

        if predicted_num_req is None or predicted_isl is None or predicted_osl is None:
            return None

        self._diag_predicted_num_req = predicted_num_req
        self._diag_predicted_isl = predicted_isl
        self._diag_predicted_osl = predicted_osl
        self._diag_predicted_kv_hit_rate = predicted_kv_hit_rate
        self.observe_runtime_metadata(
            kv_hit_rate=predicted_kv_hit_rate,
            accept_length=predicted_accept_length,
        )

        if traffic.duration_s <= 0:
            logger.warning("Traffic observation has non-positive duration, skipping")
            self._diag_throughput_reason = "no_traffic_data"
            return None

        demand_rps = predicted_num_req / traffic.duration_s
        mode = self._config.mode
        if mode == "agg":
            return self._throughput_agg(
                demand_rps, predicted_isl, predicted_osl, predicted_kv_hit_rate
            )
        if mode == "disagg":
            return self._throughput_disagg(
                demand_rps, predicted_isl, predicted_osl, predicted_kv_hit_rate
            )
        return self._throughput_single(
            demand_rps,
            predicted_isl,
            predicted_osl,
            mode,
            predicted_kv_hit_rate,
        )

    def diagnostics(self) -> TickDiagnostics:
        return self._build_diagnostics()

    def _reset_diag(self) -> None:
        self._diag_estimated_ttft_ms = None
        self._diag_estimated_itl_ms = None
        self._diag_predicted_num_req = None
        self._diag_predicted_isl = None
        self._diag_predicted_osl = None
        self._diag_predicted_kv_hit_rate = None
        self._diag_engine_rps_prefill = None
        self._diag_engine_rps_decode = None
        self._diag_load_reason = None
        self._diag_throughput_reason = None
        self._diag_load_reason_prefill = None
        self._diag_load_reason_decode = None
        self._diag_throughput_reason_prefill = None
        self._diag_throughput_reason_decode = None

    def _build_diagnostics(self) -> TickDiagnostics:
        return TickDiagnostics(
            estimated_ttft_ms=self._diag_estimated_ttft_ms,
            estimated_itl_ms=self._diag_estimated_itl_ms,
            predicted_num_req=self._diag_predicted_num_req,
            predicted_isl=self._diag_predicted_isl,
            predicted_osl=self._diag_predicted_osl,
            predicted_kv_hit_rate=self._diag_predicted_kv_hit_rate,
            engine_rps_prefill=self._diag_engine_rps_prefill,
            engine_rps_decode=self._diag_engine_rps_decode,
            throughput_lower_bound_prefill=self._throughput_lower_bound_p,
            throughput_lower_bound_decode=self._throughput_lower_bound_d,
            load_decision_reason=self._diag_load_reason,
            throughput_decision_reason=self._diag_throughput_reason,
            load_decision_reason_prefill=self._diag_load_reason_prefill,
            load_decision_reason_decode=self._diag_load_reason_decode,
            throughput_decision_reason_prefill=self._diag_throughput_reason_prefill,
            throughput_decision_reason_decode=self._diag_throughput_reason_decode,
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
        self._prefill_scaling_in_progress = counts.prefill_scaling_in_progress
        self._decode_scaling_in_progress = counts.decode_scaling_in_progress

    def _scaling_in_progress(self, component: str) -> bool:
        if component == "prefill":
            return self._prefill_scaling_in_progress or (
                self._expected_num_p is not None
                and self._expected_num_p != self._num_p_workers
            )
        return self._decode_scaling_in_progress or (
            self._expected_num_d is not None
            and self._expected_num_d != self._num_d_workers
        )

    # ------------------------------------------------------------------
    # FPM / traffic observation
    # ------------------------------------------------------------------

    def _observe_fpm(self, obs: FpmObservations) -> None:
        if self._is_agg:
            if obs.decode:
                self._agg_regression.add_observations(obs.decode)
                logger.info(f"FPM load stats: {len(obs.decode)} agg engines observed")
            return

        if obs.prefill and self._has_prefill:
            self._prefill_regression.add_observations(obs.prefill)
            logger.info(f"FPM load stats: {len(obs.prefill)} prefill engines observed")
        if obs.decode and self._has_decode:
            self._decode_regression.add_observations(obs.decode)
            logger.info(f"FPM load stats: {len(obs.decode)} decode engines observed")

    def _effective_speculative_nextn(self) -> int:
        d_caps = self._capabilities.decode
        if d_caps and d_caps.speculative_nextn and d_caps.speculative_nextn > 0:
            return d_caps.speculative_nextn
        return max(0, int(self._config.speculative_nextn))

    def _clamp_accept_length(self, accept_length: Optional[float]) -> float:
        nextn = self._effective_speculative_nextn()
        if nextn <= 0:
            return 1.0
        if accept_length is None or not math.isfinite(accept_length):
            return 1.0
        return min(max(float(accept_length), 1.0), float(nextn + 1))

    def _observe_accept_length(self, accept_length: Optional[float]) -> None:
        if accept_length is None or not math.isfinite(accept_length):
            return
        self._last_accept_length = self._clamp_accept_length(accept_length)

    def _current_decode_accept_length(self) -> float:
        return self._clamp_accept_length(self._last_accept_length)

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
        """Apply the GPU budget band (ceiling and optional floor) to
        ``(num_p, num_d)``. Delegates to ``budget.proportional_clamp_pair``
        for the actual math; this method only resolves the per-engine GPU
        counts from capabilities."""
        p_gpu = (
            self._capabilities.prefill.num_gpu if self._capabilities.prefill else None
        )
        d_gpu = self._capabilities.decode.num_gpu if self._capabilities.decode else None
        if p_gpu is None or d_gpu is None:
            return num_p, num_d

        new_p, new_d = proportional_clamp_pair(
            num_p,
            num_d,
            p_gpu,
            d_gpu,
            self._config.min_gpu_budget,
            self._config.max_gpu_budget,
            self._config.min_endpoint,
        )
        if (new_p, new_d) != (num_p, num_d):
            old_total = num_p * p_gpu + num_d * d_gpu
            new_total = new_p * p_gpu + new_d * d_gpu
            logger.warning(
                f"GPU budget band [min={self._config.min_gpu_budget}, "
                f"max={self._config.max_gpu_budget}] clamped "
                f"({num_p}P + {num_d}D = {old_total}) -> "
                f"({new_p}P + {new_d}D = {new_total})"
            )
        return new_p, new_d

    def _budget_clamp(self, desired: int, engine_gpu: int) -> int:
        """Apply the GPU budget band to a single component's desired replica
        count (agg, prefill-only, or decode-only mode)."""
        new_replicas = proportional_clamp_single(
            desired,
            engine_gpu,
            self._config.min_gpu_budget,
            self._config.max_gpu_budget,
            self._config.min_endpoint,
        )
        if new_replicas != desired:
            logger.warning(
                f"GPU budget band [min={self._config.min_gpu_budget}, "
                f"max={self._config.max_gpu_budget}] clamped "
                f"{desired} replicas (= {desired * engine_gpu} GPUs) -> "
                f"{new_replicas} replicas (= {new_replicas * engine_gpu} GPUs)"
            )
        return new_replicas

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
    def prefill_regression(self) -> PlannerEnginePerfModel:
        if not self._has_prefill:
            raise AttributeError(f"No prefill regression in mode={self._config.mode}")
        return self._prefill_regression

    @property
    def decode_regression(self) -> PlannerEnginePerfModel:
        if not self._has_decode or self._is_agg:
            raise AttributeError(f"No decode regression in mode={self._config.mode}")
        return self._decode_regression

    @property
    def agg_regression(self) -> PlannerEnginePerfModel:
        if not self._is_agg:
            raise AttributeError(f"No agg regression in mode={self._config.mode}")
        return self._agg_regression

    @property
    def regression(self) -> PlannerEnginePerfModel:
        return self.agg_regression
