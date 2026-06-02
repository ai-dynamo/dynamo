# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Planner adapter for the Rust engine performance shim.

The Rust shim is the preferred engine-level query path for SLA planning.  This
adapter keeps the planner's policy decisions local: how to add the hypothetical
next request, how to apply prefix-cache discounts for FPM v1 queued prefill, and
how to group attention-DP ranks.  The legacy Python regression model remains as
the compatibility fallback when the Python extension is built without the
``aic-forward-pass`` feature.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Optional

from dynamo.common.forward_pass_metrics import (
    FPM_VERSION,
    ForwardPassMetrics,
    QueuedRequestMetrics,
    ScheduledRequestMetrics,
)
from dynamo.planner.config.parallelization import (
    PickedParallelConfig,
    picked_to_aic_model_config_kwargs,
)
from dynamo.planner.config.planner_config import AICPerfModelSpec, PlannerConfig
from dynamo.planner.core.perf_model.agg import AggRegressionModel
from dynamo.planner.core.perf_model.base import _clamp_kv_hit_rate
from dynamo.planner.core.perf_model.decode import DecodeRegressionModel
from dynamo.planner.core.perf_model.prefill import PrefillRegressionModel
from dynamo.planner.core.types import EngineCapabilities

logger = logging.getLogger(__name__)

try:  # pragma: no cover - availability depends on the optional Rust feature.
    from dynamo.llm import (
        AicEngineConfig,
        EngineCapacityRequest,
        EnginePerfLimits,
        OptimizationTarget,
        RustEnginePerfModel,
        RustEnginePerfOptions,
    )

    _RUST_SHIM_AVAILABLE = True
except Exception:  # pragma: no cover - exercised in pure-Python planner tests.
    AicEngineConfig = None  # type: ignore[assignment]
    EngineCapacityRequest = None  # type: ignore[assignment]
    EnginePerfLimits = None  # type: ignore[assignment]
    OptimizationTarget = None  # type: ignore[assignment]
    RustEnginePerfModel = None  # type: ignore[assignment]
    RustEnginePerfOptions = None  # type: ignore[assignment]
    _RUST_SHIM_AVAILABLE = False

LegacyPerfModel = PrefillRegressionModel | DecodeRegressionModel | AggRegressionModel


@dataclass(frozen=True)
class PlannerEngineCapacity:
    """Normalized capacity result consumed by planner throughput scaling."""

    rps: float
    ttft_ms: Optional[float] = None
    itl_ms: Optional[float] = None
    e2e_latency_ms: Optional[float] = None
    eligible: bool = True


class PlannerEnginePerfModel:
    """Planner-facing wrapper around the Rust shim and legacy regression.

    ``worker_type`` is one of ``prefill``, ``decode``, or ``aggregated``.
    ``legacy_model`` is always updated so existing tests and fallback behavior
    keep the same moving averages and regression readiness semantics.
    """

    def __init__(
        self,
        *,
        worker_type: str,
        config: PlannerConfig,
        capabilities: Optional[EngineCapabilities],
        legacy_model: LegacyPerfModel,
    ) -> None:
        self._worker_type = worker_type
        self._config = config
        self._capabilities = capabilities
        self._legacy_model = legacy_model
        self._rust_model: Optional[Any] = None
        self._pending_iterations: list[list[ForwardPassMetrics]] = []

        self._init_rust_model()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @property
    def _uses_native_aic(self) -> bool:
        return self._config.aic_perf_model is not None

    def update_capabilities(self, capabilities: Optional[EngineCapabilities]) -> None:
        self._capabilities = capabilities
        if self._rust_model is None:
            self._init_rust_model()

    def _init_rust_model(self) -> None:
        if not _RUST_SHIM_AVAILABLE or RustEnginePerfModel is None:
            logger.debug("Rust engine perf shim unavailable; using Python regression")
            return

        limits = self._build_limits()
        if limits is None:
            logger.debug(
                "Engine limits are incomplete; delaying Rust perf model init for %s",
                self._worker_type,
            )
            return

        try:
            options = self._build_options()
            self._rust_model = RustEnginePerfModel.best_available(
                aic_config=self._build_aic_config(),
                worker_type=self._worker_type,
                limits=limits,
                options=options,
            )
            logger.info(
                "Initialized Rust engine perf model for %s with native_aic=%s",
                self._worker_type,
                self._uses_native_aic,
            )
            if self._pending_iterations:
                self._rust_model.tune_with_fpms(self._pending_iterations)
                self._pending_iterations.clear()
        except Exception as e:
            logger.warning(
                "Failed to initialize Rust engine perf model for %s; "
                "falling back to Python regression: %s",
                self._worker_type,
                e,
            )
            self._rust_model = None

    def _build_limits(self) -> Optional[Any]:
        caps = self._capabilities
        if caps is None or EnginePerfLimits is None:
            return None
        values = (
            caps.max_num_batched_tokens,
            caps.max_num_seqs,
            caps.max_kv_tokens,
        )
        if any(v is None or v <= 0 for v in values):
            return None
        return EnginePerfLimits(
            max_num_batched_tokens=int(caps.max_num_batched_tokens),
            max_num_seqs=int(caps.max_num_seqs),
            max_kv_tokens=int(caps.max_kv_tokens),
        )

    def _build_options(self) -> Any:
        assert RustEnginePerfOptions is not None
        caps = self._capabilities
        assert caps is not None
        return RustEnginePerfOptions(
            max_observations=self._config.max_num_fpm_samples,
            min_observations=self._config.load_min_observations,
            bucket_count=self._config.fpm_sample_bucket_size,
            max_num_tokens=int(caps.max_num_batched_tokens),
            max_batch_size=int(caps.max_num_seqs),
            max_kv_tokens=int(caps.max_kv_tokens),
        )

    def _build_aic_config(self) -> Optional[Any]:
        spec = self._config.aic_perf_model
        if spec is None or AicEngineConfig is None:
            return None
        pick = self._pick_for_worker(spec)
        if pick is None:
            return None
        kwargs = picked_to_aic_model_config_kwargs(pick)
        return AicEngineConfig(
            model_name=spec.hf_id,
            backend=spec.backend,
            system_name=spec.system,
            backend_version=spec.backend_version,
            kv_block_size=(
                self._capabilities.kv_cache_block_size
                if self._capabilities is not None
                else None
            ),
            model_arch=spec.model_arch,
            weight_dtype=spec.weight_dtype,
            moe_dtype=spec.moe_dtype,
            activation_dtype=spec.activation_dtype,
            kv_cache_dtype=spec.kv_cache_dtype,
            **kwargs,
        )

    def _pick_for_worker(
        self, spec: AICPerfModelSpec
    ) -> Optional[PickedParallelConfig]:
        if self._worker_type == "prefill":
            return spec.prefill_pick
        return spec.decode_pick

    def _attention_dp_size(self) -> Optional[int]:
        spec = self._config.aic_perf_model
        if spec is None:
            return None
        pick = self._pick_for_worker(spec)
        if pick is None:
            return None
        return pick.dp

    # ------------------------------------------------------------------
    # Observation and bootstrap
    # ------------------------------------------------------------------

    def add_observation(self, fpm: ForwardPassMetrics) -> None:
        """Compatibility hook used by older state-machine tests."""
        self.add_observations({(fpm.worker_id, fpm.dp_rank): fpm})

    def add_observations(
        self, fpm_stats: dict[tuple[str, int], ForwardPassMetrics]
    ) -> None:
        valid: dict[tuple[str, int], ForwardPassMetrics] = {}
        for key, fpm in fpm_stats.items():
            if self._is_supported_fpm(fpm):
                valid[key] = fpm
                self._legacy_model.add_observation(fpm)
        if not valid:
            return
        self._tune(self._iteration_groups(valid, for_query=False))

    def load_benchmark_fpms(self, fpms: list[ForwardPassMetrics]) -> None:
        valid = [fpm for fpm in fpms if self._is_supported_fpm(fpm)]
        if not valid:
            return
        self._legacy_model.load_benchmark_fpms(valid)
        self._tune(self._iteration_groups_from_list(valid))

    def _tune(self, iterations: list[list[ForwardPassMetrics]]) -> None:
        if not iterations:
            return
        if self._rust_model is not None:
            try:
                self._rust_model.tune_with_fpms(iterations)
            except Exception as e:
                logger.warning("Rust perf model tuning failed: %s", e)
        else:
            self._pending_iterations.extend(iterations)
            if len(self._pending_iterations) > self._config.max_num_fpm_samples:
                self._pending_iterations = self._pending_iterations[
                    -self._config.max_num_fpm_samples :
                ]

    def _is_supported_fpm(self, fpm: ForwardPassMetrics) -> bool:
        if fpm.version != FPM_VERSION:
            logger.warning(
                "Skipping unsupported FPM version %s; planner supports version %s",
                fpm.version,
                FPM_VERSION,
            )
            return False
        return True

    # ------------------------------------------------------------------
    # Query grouping
    # ------------------------------------------------------------------

    def query_groups(
        self, fpm_stats: dict[tuple[str, int], ForwardPassMetrics]
    ) -> list[tuple[str, list[ForwardPassMetrics]]]:
        """Group live FPMs for query-time estimates.

        Native AIC estimates require one FPM per attention-DP rank. When the
        Rust shim is unavailable, preserve legacy rank-local behavior.
        """
        groups = self._iteration_groups(fpm_stats, for_query=True)
        return [(self._group_label(group), group) for group in groups]

    def _iteration_groups(
        self,
        fpm_stats: dict[tuple[str, int], ForwardPassMetrics],
        *,
        for_query: bool,
    ) -> list[list[ForwardPassMetrics]]:
        should_group_for_rust = (
            self._rust_model is not None if for_query else _RUST_SHIM_AVAILABLE
        )
        if should_group_for_rust:
            dp_size = self._attention_dp_size()
            if dp_size is not None and dp_size > 1:
                by_worker: dict[str, list[ForwardPassMetrics]] = {}
                for (worker_id, _dp_rank), fpm in fpm_stats.items():
                    by_worker.setdefault(worker_id, []).append(fpm)
                return [
                    sorted(group, key=lambda item: item.dp_rank)
                    for group in by_worker.values()
                ]
        return [
            [fpm]
            for (_worker_id, _dp_rank), fpm in sorted(
                fpm_stats.items(), key=lambda item: item[0]
            )
        ]

    def _iteration_groups_from_list(
        self, fpms: list[ForwardPassMetrics]
    ) -> list[list[ForwardPassMetrics]]:
        # Bootstrap FPMs loaded from profiler/AIC interpolation are flat
        # historical samples. They do not encode which records belonged to the
        # same attention-DP iteration, so keep each sample as one iteration.
        return [[fpm] for fpm in fpms]

    @staticmethod
    def _group_label(group: list[ForwardPassMetrics]) -> str:
        if not group:
            return "unknown"
        worker_id = group[0].worker_id
        if len(group) == 1:
            return f"{worker_id}:dp{group[0].dp_rank}"
        ranks = ",".join(str(fpm.dp_rank) for fpm in group)
        return f"{worker_id}:dp[{ranks}]"

    # ------------------------------------------------------------------
    # Planner query helpers
    # ------------------------------------------------------------------

    def estimate_queued_prefill_time(
        self,
        metrics_by_rank: list[ForwardPassMetrics],
        *,
        max_num_batched_tokens: int,
        kv_hit_rate: Optional[float] = None,
        queue_scale: float = 1.0,
        decode_scale: float = 1.0,
        include_queued_decode: bool = False,
        add_next_request: bool = True,
    ) -> Optional[float]:
        """Estimate next-request TTFT from queued prefill work.

        FPM v1 queued prefill does not know KV reuse. The planner applies the
        router-provided prefix-cache discount before calling the shim.
        """
        if self._rust_model is None:
            return self._legacy_queued_prefill_time(
                metrics_by_rank,
                max_num_batched_tokens=max_num_batched_tokens,
                kv_hit_rate=kv_hit_rate,
                queue_scale=queue_scale,
                decode_scale=decode_scale,
                include_queued_decode=include_queued_decode,
            )

        scale = 1.0 - _clamp_kv_hit_rate(kv_hit_rate)
        fpms = [
            self._synthetic_prefill_fpm(
                fpm,
                queue_scale=queue_scale,
                decode_scale=decode_scale,
                include_queued_decode=include_queued_decode,
                prefill_scale=scale,
                add_next_request=add_next_request,
            )
            for fpm in metrics_by_rank
        ]
        try:
            return self._rust_model.get_queued_prefill_time(fpms)
        except Exception as e:
            logger.warning("Rust queued prefill estimate failed: %s", e)
            return None

    def estimate_scheduled_decode_itl(
        self,
        metrics_by_rank: list[ForwardPassMetrics],
        *,
        decode_scale: float = 1.0,
        include_queued_decode: bool = True,
        include_queued_prefill_as_kv: bool = False,
        add_next_request: bool = True,
    ) -> Optional[float]:
        """Estimate next-request ITL from scheduled decode work."""
        if self._rust_model is None:
            return self._legacy_scheduled_decode_itl(
                metrics_by_rank,
                decode_scale=decode_scale,
                include_queued_decode=include_queued_decode,
                include_queued_prefill_as_kv=include_queued_prefill_as_kv,
            )

        fpms = [
            self._synthetic_decode_fpm(
                fpm,
                decode_scale=decode_scale,
                include_queued_decode=include_queued_decode,
                include_queued_prefill_as_kv=include_queued_prefill_as_kv,
                add_next_request=add_next_request,
            )
            for fpm in metrics_by_rank
        ]
        try:
            return self._rust_model.get_scheduled_decode_itl(fpms)
        except Exception as e:
            logger.warning("Rust scheduled decode estimate failed: %s", e)
            return None

    def find_engine_capacity_rps(
        self,
        *,
        isl: float,
        osl: float,
        ttft_sla_ms: Optional[float] = None,
        itl_sla_ms: Optional[float] = None,
        e2e_latency_sla_ms: Optional[float] = None,
        kv_hit_rate: Optional[float] = None,
    ) -> Optional[PlannerEngineCapacity]:
        """Estimate sustainable single-engine RPS for one request shape."""
        if self._rust_model is None:
            return self._legacy_capacity(
                isl=isl,
                osl=osl,
                ttft_sla_ms=ttft_sla_ms,
                itl_sla_ms=itl_sla_ms,
                kv_hit_rate=kv_hit_rate,
            )
        if isl <= 0 or osl <= 0 or EngineCapacityRequest is None:
            return None
        try:
            request = EngineCapacityRequest(
                isl=int(math.ceil(isl)),
                osl=int(math.ceil(osl)),
                ttft_sla_ms=ttft_sla_ms,
                itl_sla_ms=itl_sla_ms,
                e2e_latency_sla_ms=e2e_latency_sla_ms,
                optimization_target=OptimizationTarget.Throughput,
            )
            result = self._rust_model.find_engine_capacity_rps(request)
        except Exception as e:
            logger.warning("Rust capacity query failed: %s", e)
            return None
        if result is None:
            return None
        return PlannerEngineCapacity(
            rps=result.rps,
            ttft_ms=result.ttft_ms,
            itl_ms=result.itl_ms,
            e2e_latency_ms=result.e2e_latency_ms,
            eligible=result.eligible,
        )

    # ------------------------------------------------------------------
    # Compatibility methods for existing tests and fallback call sites
    # ------------------------------------------------------------------

    def has_sufficient_data(self) -> bool:
        if self._rust_model is not None and self._uses_native_aic:
            return True
        return self._legacy_model.has_sufficient_data()

    @property
    def num_observations(self) -> int:
        return self._legacy_model.num_observations

    @property
    def min_observations(self) -> int:
        return self._legacy_model.min_observations

    @property
    def avg_isl(self) -> float:
        return getattr(self._legacy_model, "avg_isl", 0.0)

    @property
    def avg_decode_length(self) -> float:
        return getattr(self._legacy_model, "avg_decode_length", 0.0)

    def estimate_next_ttft(self, *args: Any, **kwargs: Any) -> Optional[float]:
        return self._legacy_model.estimate_next_ttft(*args, **kwargs)

    def estimate_next_itl(self, *args: Any, **kwargs: Any) -> Optional[float]:
        return self._legacy_model.estimate_next_itl(*args, **kwargs)

    def find_best_engine_prefill_rps(self, *args: Any, **kwargs: Any) -> Any:
        return self._legacy_model.find_best_engine_prefill_rps(*args, **kwargs)

    def find_best_engine_decode_rps(self, *args: Any, **kwargs: Any) -> Any:
        return self._legacy_model.find_best_engine_decode_rps(*args, **kwargs)

    def find_best_engine_agg_rps(self, *args: Any, **kwargs: Any) -> Any:
        return self._legacy_model.find_best_engine_agg_rps(*args, **kwargs)

    # ------------------------------------------------------------------
    # Synthetic FPM builders
    # ------------------------------------------------------------------

    def _synthetic_prefill_fpm(
        self,
        fpm: ForwardPassMetrics,
        *,
        queue_scale: float,
        decode_scale: float,
        include_queued_decode: bool,
        prefill_scale: float,
        add_next_request: bool,
    ) -> ForwardPassMetrics:
        queued = fpm.queued_requests
        scheduled = fpm.scheduled_requests
        queued_tokens = float(queued.sum_prefill_tokens) * queue_scale
        queued_requests = float(queued.num_prefill_requests) * queue_scale
        if add_next_request and self.avg_isl > 0:
            queued_tokens += self.avg_isl
            queued_requests += 1.0
        queued_tokens *= prefill_scale

        decode_kv = float(scheduled.sum_decode_kv_tokens)
        decode_requests = float(scheduled.num_decode_requests)
        if include_queued_decode:
            decode_kv += float(queued.sum_decode_kv_tokens)
            decode_requests += float(queued.num_decode_requests)
        decode_kv *= decode_scale
        decode_requests *= decode_scale

        return self._replace_fpm(
            fpm,
            scheduled=ScheduledRequestMetrics(
                num_decode_requests=self._ceil_nonnegative(decode_requests),
                sum_decode_kv_tokens=self._ceil_nonnegative(decode_kv),
            ),
            queued=QueuedRequestMetrics(
                num_prefill_requests=self._ceil_nonnegative(queued_requests),
                sum_prefill_tokens=self._ceil_nonnegative(queued_tokens),
            ),
        )

    def _synthetic_decode_fpm(
        self,
        fpm: ForwardPassMetrics,
        *,
        decode_scale: float,
        include_queued_decode: bool,
        include_queued_prefill_as_kv: bool,
        add_next_request: bool,
    ) -> ForwardPassMetrics:
        queued = fpm.queued_requests
        scheduled = fpm.scheduled_requests

        prefill_tokens = float(scheduled.sum_prefill_tokens)
        prefill_requests = float(scheduled.num_prefill_requests)

        decode_kv = float(scheduled.sum_decode_kv_tokens)
        decode_requests = float(scheduled.num_decode_requests)
        if include_queued_decode:
            decode_kv += float(queued.sum_decode_kv_tokens)
            decode_requests += float(queued.num_decode_requests)
        if include_queued_prefill_as_kv:
            decode_kv += float(queued.sum_prefill_tokens)
            decode_requests += float(queued.num_prefill_requests)
        decode_kv *= decode_scale
        decode_requests *= decode_scale
        if add_next_request and self.avg_decode_length > 0:
            decode_kv += self.avg_decode_length
            decode_requests += 1.0

        return self._replace_fpm(
            fpm,
            scheduled=ScheduledRequestMetrics(
                num_prefill_requests=self._ceil_nonnegative(prefill_requests),
                sum_prefill_tokens=self._ceil_nonnegative(prefill_tokens),
                num_decode_requests=self._ceil_nonnegative(decode_requests),
                sum_decode_kv_tokens=self._ceil_nonnegative(decode_kv),
            ),
            queued=QueuedRequestMetrics(),
        )

    @staticmethod
    def _replace_fpm(
        fpm: ForwardPassMetrics,
        *,
        scheduled: ScheduledRequestMetrics,
        queued: QueuedRequestMetrics,
    ) -> ForwardPassMetrics:
        return ForwardPassMetrics(
            version=FPM_VERSION,
            worker_id=fpm.worker_id,
            dp_rank=fpm.dp_rank,
            counter_id=fpm.counter_id,
            wall_time=0.0,
            scheduled_requests=scheduled,
            queued_requests=queued,
        )

    @staticmethod
    def _ceil_nonnegative(value: float) -> int:
        if value <= 0:
            return 0
        return int(math.ceil(value))

    # ------------------------------------------------------------------
    # Legacy fallback query implementations
    # ------------------------------------------------------------------

    def _legacy_queued_prefill_time(
        self,
        metrics_by_rank: list[ForwardPassMetrics],
        *,
        max_num_batched_tokens: int,
        kv_hit_rate: Optional[float],
        queue_scale: float,
        decode_scale: float,
        include_queued_decode: bool,
    ) -> Optional[float]:
        queued = max(
            (fpm.queued_requests.sum_prefill_tokens * queue_scale)
            for fpm in metrics_by_rank
        )
        if isinstance(self._legacy_model, PrefillRegressionModel):
            return self._legacy_model.estimate_next_ttft(
                queued_prefill_tokens=int(queued),
                max_num_batched_tokens=max_num_batched_tokens,
                kv_hit_rate=kv_hit_rate,
            )
        if isinstance(self._legacy_model, DecodeRegressionModel):
            return None
        decode_kv = 0.0
        for fpm in metrics_by_rank:
            value = float(fpm.scheduled_requests.sum_decode_kv_tokens)
            if include_queued_decode:
                value += float(fpm.queued_requests.sum_decode_kv_tokens)
            decode_kv = max(decode_kv, value * decode_scale)
        return self._legacy_model.estimate_next_ttft(
            queued_prefill_tokens=int(queued),
            max_num_batched_tokens=max_num_batched_tokens,
            current_decode_kv=int(decode_kv),
            kv_hit_rate=kv_hit_rate,
        )

    def _legacy_scheduled_decode_itl(
        self,
        metrics_by_rank: list[ForwardPassMetrics],
        *,
        decode_scale: float,
        include_queued_decode: bool,
        include_queued_prefill_as_kv: bool,
    ) -> Optional[float]:
        if isinstance(self._legacy_model, PrefillRegressionModel):
            return None
        scheduled_kv = 0.0
        queued_kv = 0.0
        for fpm in metrics_by_rank:
            sched = float(fpm.scheduled_requests.sum_decode_kv_tokens)
            queued = (
                float(fpm.queued_requests.sum_decode_kv_tokens)
                if include_queued_decode
                else 0.0
            )
            if include_queued_prefill_as_kv:
                sched += float(fpm.queued_requests.sum_prefill_tokens)
            scheduled_kv = max(scheduled_kv, sched * decode_scale)
            queued_kv = max(queued_kv, queued * decode_scale)
        return self._legacy_model.estimate_next_itl(
            scheduled_decode_kv=int(scheduled_kv),
            queued_decode_kv=int(queued_kv),
        )

    def _legacy_capacity(
        self,
        *,
        isl: float,
        osl: float,
        ttft_sla_ms: Optional[float],
        itl_sla_ms: Optional[float],
        kv_hit_rate: Optional[float],
    ) -> Optional[PlannerEngineCapacity]:
        if isl <= 0 or osl <= 0:
            return None
        caps = self._capabilities
        if isinstance(self._legacy_model, PrefillRegressionModel):
            rps, ttft_ms = self._legacy_model.find_best_engine_prefill_rps(
                ttft_sla=ttft_sla_ms or self._config.ttft_ms,
                isl=isl,
                max_num_batched_tokens=(
                    caps.max_num_batched_tokens if caps is not None else None
                ),
            )
            return PlannerEngineCapacity(rps=rps, ttft_ms=ttft_ms)
        if isinstance(self._legacy_model, DecodeRegressionModel):
            rps, itl_ms = self._legacy_model.find_best_engine_decode_rps(
                itl=itl_sla_ms or self._config.itl_ms,
                context_length=isl + osl / 2,
                osl=osl,
                max_kv_tokens=caps.max_kv_tokens if caps is not None else None,
                max_num_seqs=caps.max_num_seqs if caps is not None else None,
            )
            return PlannerEngineCapacity(rps=rps, itl_ms=itl_ms)
        if caps is None or caps.max_num_batched_tokens is None:
            return None
        rps, ttft_ms, itl_ms = self._legacy_model.find_best_engine_agg_rps(
            isl=isl,
            osl=osl,
            max_num_batched_tokens=caps.max_num_batched_tokens,
            ttft_sla=ttft_sla_ms or self._config.ttft_ms,
            itl_sla=itl_sla_ms or self._config.itl_ms,
            max_kv_tokens=caps.max_kv_tokens,
            max_num_seqs=caps.max_num_seqs,
            kv_hit_rate=kv_hit_rate,
        )
        return PlannerEngineCapacity(rps=rps, ttft_ms=ttft_ms, itl_ms=itl_ms)
