# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Planner adapter for the Rust engine performance shim.

The Rust shim is the preferred engine-level query path for SLA planning.  This
adapter keeps the planner's policy decisions local: how to add the hypothetical
next request, how to apply prefix-cache discounts for FPM v1 queued prefill, and
how to group attention-DP ranks.
"""

from __future__ import annotations

import json
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
from dynamo.planner.config.parallelization import PickedParallelConfig
from dynamo.planner.config.planner_config import AICPerfModelSpec, PlannerConfig
from dynamo.planner.core.perf_model.base import _clamp_kv_hit_rate
from dynamo.planner.core.types import EngineCapabilities

logger = logging.getLogger(__name__)

AicEngineConfig: Any = None
EngineCapacityRequest: Any = None
EnginePerfLimits: Any = None
OptimizationTarget: Any = None
RustEnginePerfModel: Any = None
RustEnginePerfOptions: Any = None

try:  # pragma: no cover - availability depends on the optional Rust feature.
    from dynamo.mocker import AicEngineConfig as _AicEngineConfig
    from dynamo.mocker import EngineCapacityRequest as _EngineCapacityRequest
    from dynamo.mocker import EnginePerfLimits as _EnginePerfLimits
    from dynamo.mocker import OptimizationTarget as _OptimizationTarget
    from dynamo.mocker import RustEnginePerfModel as _RustEnginePerfModel
    from dynamo.mocker import RustEnginePerfOptions as _RustEnginePerfOptions

    AicEngineConfig = _AicEngineConfig
    EngineCapacityRequest = _EngineCapacityRequest
    EnginePerfLimits = _EnginePerfLimits
    OptimizationTarget = _OptimizationTarget
    RustEnginePerfModel = _RustEnginePerfModel
    RustEnginePerfOptions = _RustEnginePerfOptions
    _RUST_SHIM_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised in pure-Python planner tests.
    _RUST_SHIM_AVAILABLE = False

_RUST_SHIM_FALLBACK_EXCEPTIONS = (RuntimeError, ValueError, TypeError)

DEFAULT_MAX_NUM_BATCHED_TOKENS = 8192
DEFAULT_MAX_NUM_SEQS = 512
DEFAULT_MAX_KV_TOKENS = 2_000_000


@dataclass(frozen=True)
class PlannerEngineCapacity:
    """Normalized capacity result consumed by planner throughput scaling."""

    rps: float
    ttft_ms: Optional[float] = None
    itl_ms: Optional[float] = None
    e2e_latency_ms: Optional[float] = None
    eligible: bool = True


class _MovingAverage:
    """Fixed-window average for next-request synthesis, skipping leading zeros."""

    def __init__(self, max_len: int) -> None:
        self._max_len = max(1, max_len)
        self._values: list[float] = []
        self._sum = 0.0
        self._seen_nonzero = False

    def add_after_first_nonzero(self, value: float) -> None:
        if value > 0.0:
            self._seen_nonzero = True
        if not self._seen_nonzero:
            return
        value = max(0.0, value)
        self._values.append(value)
        self._sum += value
        if len(self._values) > self._max_len:
            self._sum -= self._values.pop(0)

    @property
    def value(self) -> float:
        if not self._values:
            return 0.0
        return self._sum / len(self._values)


class PlannerEnginePerfModel:
    """Planner-facing wrapper around the Rust shim.

    ``worker_type`` is one of ``prefill``, ``decode``, or ``aggregated``.
    """

    def __init__(
        self,
        *,
        worker_type: str,
        config: PlannerConfig,
        capabilities: Optional[EngineCapabilities],
    ) -> None:
        self._worker_type = worker_type
        self._config = config
        self._capabilities = capabilities
        self._rust_model: Optional[Any] = None
        self._pending_iterations: list[list[ForwardPassMetrics]] = []
        self._avg_isl = _MovingAverage(config.max_num_fpm_samples)
        self._avg_decode_length = _MovingAverage(config.max_num_fpm_samples)

        self._init_rust_model()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def update_capabilities(self, capabilities: Optional[EngineCapabilities]) -> None:
        self._capabilities = capabilities
        if self._rust_model is None:
            self._init_rust_model()

    def _init_rust_model(self) -> None:
        if not _RUST_SHIM_AVAILABLE or RustEnginePerfModel is None:
            logger.debug("Rust engine perf shim unavailable")
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
            diagnostics = self._rust_diagnostics()
            logger.info(
                "Initialized Rust engine perf model for %s with source=%s readiness=%s",
                self._worker_type,
                diagnostics.get("source", "unknown"),
                diagnostics.get("readiness", "unknown"),
            )
            if self._pending_iterations:
                self._rust_model.tune_with_fpms(self._pending_iterations)
                self._pending_iterations.clear()
        except _RUST_SHIM_FALLBACK_EXCEPTIONS as e:
            logger.warning(
                "Failed to initialize Rust engine perf model for %s; "
                "perf model will stay unavailable until capabilities/config are fixed: %s",
                self._worker_type,
                e,
            )
            self._rust_model = None

    def _rust_diagnostics(self) -> dict[str, Any]:
        if self._rust_model is None:
            return {}
        try:
            diagnostics = self._rust_model.diagnostics()
            if isinstance(diagnostics, str):
                loaded = json.loads(diagnostics)
                return loaded if isinstance(loaded, dict) else {}
            return diagnostics if isinstance(diagnostics, dict) else {}
        except json.JSONDecodeError as e:
            logger.warning("Rust perf model diagnostics JSON decode failed: %s", e)
            return {}
        except _RUST_SHIM_FALLBACK_EXCEPTIONS as e:
            logger.warning("Rust perf model diagnostics failed: %s", e)
            return {}

    def _rust_ready(self) -> bool:
        return self._rust_diagnostics().get("readiness") == "ready"

    def _limit_values(self) -> Optional[tuple[int, int, int]]:
        caps = self._capabilities
        if caps is None:
            return None
        if caps.max_num_batched_tokens is not None and caps.max_num_batched_tokens <= 0:
            return None
        if caps.max_num_seqs is not None and caps.max_num_seqs <= 0:
            return None
        if caps.max_kv_tokens is not None and caps.max_kv_tokens <= 0:
            return None

        # Match the Rust shim's MockEngineArgs defaults when the planner has a
        # capability object but an individual runtime limit has not been
        # published yet. This keeps the FPM regression path usable for tests
        # and partial worker metadata while still rejecting explicit invalid
        # values above.
        max_num_batched_tokens = (
            caps.max_num_batched_tokens or DEFAULT_MAX_NUM_BATCHED_TOKENS
        )
        max_num_seqs = caps.max_num_seqs or DEFAULT_MAX_NUM_SEQS
        max_kv_tokens = caps.max_kv_tokens or DEFAULT_MAX_KV_TOKENS
        return (max_num_batched_tokens, max_num_seqs, max_kv_tokens)

    def _build_limits(self) -> Optional[Any]:
        values = self._limit_values()
        if values is None or EnginePerfLimits is None:
            return None
        max_num_batched_tokens, max_num_seqs, max_kv_tokens = values
        return EnginePerfLimits(
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
            max_kv_tokens=max_kv_tokens,
        )

    def _build_options(self) -> Any:
        assert RustEnginePerfOptions is not None
        values = self._limit_values()
        assert values is not None
        max_num_batched_tokens, max_num_seqs, max_kv_tokens = values
        return RustEnginePerfOptions(
            max_observations=self._config.max_num_fpm_samples,
            min_observations=self._config.load_min_observations,
            bucket_count=self._config.fpm_sample_bucket_size,
            max_num_tokens=max_num_batched_tokens,
            max_batch_size=max_num_seqs,
            max_kv_tokens=max_kv_tokens,
        )

    def _build_aic_config(self) -> Optional[Any]:
        spec = self._config.aic_perf_model
        if spec is None or AicEngineConfig is None:
            return None
        pick = self._pick_for_worker(spec)
        if pick is None:
            return None
        return AicEngineConfig(
            model_name=spec.hf_id,
            backend=spec.backend,
            system_name=spec.system,
            backend_version=spec.backend_version,
            tp_size=pick.tp,
            pp_size=pick.pp,
            moe_tp_size=pick.moe_tp,
            moe_ep_size=pick.moe_ep,
            attention_dp_size=pick.dp,
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

    def add_observations(
        self, fpm_stats: dict[tuple[str, int], ForwardPassMetrics]
    ) -> None:
        valid: dict[tuple[str, int], ForwardPassMetrics] = {}
        for key, fpm in fpm_stats.items():
            if self._is_supported_fpm(fpm):
                valid[key] = fpm
                self._observe_for_next_request(fpm)
        if not valid:
            return
        self._tune(self._iteration_groups(valid, for_query=False))

    def load_benchmark_fpms(self, fpms: list[ForwardPassMetrics]) -> None:
        valid = [fpm for fpm in fpms if self._is_supported_fpm(fpm)]
        if not valid:
            return
        for fpm in valid:
            self._observe_for_next_request(fpm)
        self._tune(self._iteration_groups_from_list(valid))

    def _observe_for_next_request(self, fpm: ForwardPassMetrics) -> None:
        scheduled = fpm.scheduled_requests
        if scheduled.num_prefill_requests > 0:
            self._avg_isl.add_after_first_nonzero(
                scheduled.sum_prefill_tokens / scheduled.num_prefill_requests
            )
        else:
            self._avg_isl.add_after_first_nonzero(0.0)
        if scheduled.num_decode_requests > 0:
            self._avg_decode_length.add_after_first_nonzero(
                scheduled.sum_decode_kv_tokens / scheduled.num_decode_requests
            )

    def _tune(self, iterations: list[list[ForwardPassMetrics]]) -> None:
        if not iterations:
            return
        if self._rust_model is not None:
            try:
                self._rust_model.tune_with_fpms(iterations)
            except _RUST_SHIM_FALLBACK_EXCEPTIONS as e:
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
        # same attention-DP iteration. For ADP>1, skip Rust bootstrap tuning
        # instead of sending singleton-rank iterations to a model that requires
        # one FPM per rank.
        # TODO: consume grouped per-rank bootstrap data when profiling exports
        # iteration/rank grouping for attention-DP workloads.
        dp_size = self._attention_dp_size()
        if dp_size is not None and dp_size > 1:
            logger.info(
                "Skipping Rust bootstrap tuning for %s because flat bootstrap "
                "FPMs do not preserve attention-DP rank groups",
                self._worker_type,
            )
            return []
        # For single-rank workers, each flat sample is one complete iteration.
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
        Rust query failures are treated as unavailable estimates so load
        scaling can skip the current tick.
        """
        if self._rust_model is None:
            return None

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
            result = self._rust_model.get_queued_prefill_time(fpms)
        except _RUST_SHIM_FALLBACK_EXCEPTIONS as e:
            logger.warning("Rust queued prefill estimate failed: %s", e)
            return None
        return result

    def estimate_scheduled_decode_itl(
        self,
        metrics_by_rank: list[ForwardPassMetrics],
        *,
        decode_scale: float = 1.0,
        include_queued_decode: bool = True,
        include_queued_prefill_as_kv: bool = False,
        add_next_request: bool = True,
    ) -> Optional[float]:
        """Estimate next-request ITL from scheduled decode work.

        Rust query failures are treated as unavailable estimates so load
        scaling can skip the current tick.
        """
        if self._rust_model is None:
            return None

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
            result = self._rust_model.get_scheduled_decode_itl(fpms)
        except _RUST_SHIM_FALLBACK_EXCEPTIONS as e:
            logger.warning("Rust scheduled decode estimate failed: %s", e)
            return None
        return result

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
        """Estimate sustainable single-engine RPS for one request shape.

        Rust query failures are treated as unavailable estimates so throughput
        scaling can skip the current decision.
        """
        if self._rust_model is None:
            return None
        if isl <= 0 or EngineCapacityRequest is None:
            return None
        if self._worker_type != "prefill" and osl <= 0:
            return None
        try:
            request = EngineCapacityRequest(
                isl=int(math.ceil(isl)),
                osl=max(1, int(math.ceil(osl))),
                ttft_sla_ms=ttft_sla_ms,
                itl_sla_ms=itl_sla_ms,
                e2e_latency_sla_ms=e2e_latency_sla_ms,
                kv_hit_rate=kv_hit_rate,
                optimization_target=OptimizationTarget.Throughput,
            )
            result = self._rust_model.find_engine_capacity_rps(request)
        except _RUST_SHIM_FALLBACK_EXCEPTIONS as e:
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
    # Readiness and moving-average accessors used by load scaling and query
    # synthesis. Direct legacy prediction/capacity calls should not go through
    # this adapter; use the engine-query helpers above or the regression
    # classes directly in regression-specific tests.
    # ------------------------------------------------------------------

    def has_sufficient_data(self) -> bool:
        if self._rust_model is not None and self._rust_ready():
            return True
        return False

    @property
    def num_observations(self) -> int:
        value = self._rust_diagnostics().get("retained_observations", 0)
        return int(value) if isinstance(value, int) else 0

    @property
    def min_observations(self) -> int:
        return self._config.load_min_observations

    @property
    def avg_isl(self) -> float:
        return self._avg_isl.value

    @property
    def avg_decode_length(self) -> float:
        return self._avg_decode_length.value

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
        if self._worker_type == "aggregated":
            # Match the old aggregated ITL path: per-iteration scheduled
            # prefill is intentionally not used for decode load decisions.
            # The shim falls back to its learned average prefill/decode mix.
            prefill_tokens = 0.0
            prefill_requests = 0.0

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
