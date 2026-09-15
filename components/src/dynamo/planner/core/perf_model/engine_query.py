# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Planner-owned engine-level queries over AIC forward-pass estimates.

``aiconfigurator_core.sdk.RustForwardPassPerfModel`` owns native AIC
estimation, online correction, and regression fallback. This module owns the
Dynamo policy above that forward-pass abstraction: queue-drain estimates,
TTFT/ITL derivation, engine-limit checks, and bounded capacity searches.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from enum import Enum
from itertools import pairwise
from typing import Any, Callable, Literal, Optional

import msgspec
from aiconfigurator_core.sdk import RustForwardPassPerfModel as AicForwardPassPerfModel

from dynamo.common.forward_pass_metrics import (
    FPM_VERSION,
    ForwardPassMetrics,
    QueuedRequestMetrics,
    ScheduledRequestMetrics,
)

MAX_CAPACITY_SEARCH_CANDIDATES = 128
MAX_KV_HIT_RATE_DISCOUNT = 0.95
_U32_MAX = (1 << 32) - 1
_DURATION_EXCLUSIVE_MAX_SECONDS = float(1 << 64)

WorkerType = Literal["prefill", "decode", "aggregated"]


@dataclass(frozen=True)
class EnginePerfLimits:
    """Runtime limits used by Planner engine-level queries."""

    max_num_batched_tokens: int
    max_num_seqs: int
    max_kv_tokens: int

    def __post_init__(self) -> None:
        for name, value in (
            ("max_num_batched_tokens", self.max_num_batched_tokens),
            ("max_num_seqs", self.max_num_seqs),
            ("max_kv_tokens", self.max_kv_tokens),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
            if value > _U32_MAX:
                raise ValueError(f"{name} exceeds u32::MAX")


class OptimizationTarget(str, Enum):
    """Capacity-search objective."""

    THROUGHPUT = "throughput"
    LATENCY = "latency"


@dataclass(frozen=True)
class EngineCapacityRequest:
    """Request shape and SLA policy for a per-engine capacity search."""

    isl: int
    osl: int
    ttft_sla_s: Optional[float] = None
    itl_sla_s: Optional[float] = None
    e2e_latency_sla_s: Optional[float] = None
    accept_length: float = 1.0
    kv_hit_rate: Optional[float] = None
    optimization_target: OptimizationTarget = OptimizationTarget.THROUGHPUT

    def __post_init__(self) -> None:
        if self.isl < 0 or self.isl > _U32_MAX:
            raise ValueError(f"isl must fit u32, got {self.isl}")
        if self.osl < 0 or self.osl > _U32_MAX:
            raise ValueError(f"osl must fit u32, got {self.osl}")
        for name, value in (
            ("ttft_sla_s", self.ttft_sla_s),
            ("itl_sla_s", self.itl_sla_s),
            ("e2e_latency_sla_s", self.e2e_latency_sla_s),
        ):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and non-negative, got {value}")


@dataclass(frozen=True)
class EngineCapacity:
    """Sustainable per-engine request rate and modeled latency."""

    rps: float
    ttft_s: Optional[float] = None
    itl_s: Optional[float] = None
    e2e_latency_s: Optional[float] = None
    eligible: bool = True


class _MovingAverage:
    def __init__(self, max_len: int) -> None:
        self._values: deque[float] = deque()
        self._sum = 0.0
        self._max_len = max(1, max_len)
        self._seen_nonzero = False

    def add_after_first_nonzero(self, value: float) -> None:
        if value > 0:
            self._seen_nonzero = True
        if self._seen_nonzero:
            self.add(max(0.0, value))

    def add(self, value: float) -> None:
        self._values.append(value)
        self._sum += value
        while len(self._values) > self._max_len:
            self._sum -= self._values.popleft()

    @property
    def value(self) -> float:
        if not self._values:
            return 0.0
        return self._sum / len(self._values)


class _AggLoadAverages:
    def __init__(self, max_observations: int) -> None:
        self._avg_num_prefill = _MovingAverage(max_observations)
        self._avg_prefill_tokens = _MovingAverage(max_observations)

    def observe_iterations(self, iterations: list[list[ForwardPassMetrics]]) -> None:
        for metrics_by_rank in iterations:
            self._observe_iteration(metrics_by_rank)

    def _observe_iteration(self, metrics_by_rank: list[ForwardPassMetrics]) -> None:
        if not metrics_by_rank:
            return
        snapshot = max(
            metrics_by_rank,
            key=lambda item: (
                item.scheduled_requests.sum_prefill_tokens
                + item.scheduled_requests.sum_decode_kv_tokens
                + item.scheduled_requests.num_decode_requests
            ),
        )
        scheduled = snapshot.scheduled_requests
        if scheduled.num_prefill_requests > 0:
            self._avg_num_prefill.add_after_first_nonzero(
                float(scheduled.num_prefill_requests)
            )
        else:
            self._avg_num_prefill.add_after_first_nonzero(0.0)
        self._avg_prefill_tokens.add_after_first_nonzero(
            float(scheduled.sum_prefill_tokens)
        )

    def typical_prefill_for_mixed_itl(self) -> tuple[int, int]:
        tokens = _ceil_u32(self._avg_prefill_tokens.value, "avg prefill tokens")
        requests = _ceil_u32(self._avg_num_prefill.value, "avg prefill requests")
        return (max(requests, int(tokens > 0)), tokens)


@dataclass(frozen=True)
class _PrefillChunkPlan:
    full_chunks: int
    tail_tokens: int

    @classmethod
    def build(cls, tokens: int, max_chunk: int) -> _PrefillChunkPlan:
        if max_chunk <= 0:
            raise ValueError(f"max_chunk must be positive, got {max_chunk}")
        return cls(full_chunks=tokens // max_chunk, tail_tokens=tokens % max_chunk)

    def chunk_at(self, iteration: int, max_chunk: int) -> int:
        if iteration < self.full_chunks:
            return max_chunk
        if iteration == self.full_chunks:
            return self.tail_tokens
        return 0


class AicCoreEnginePerfModel:
    """Dynamo engine-query layer backed by the AIC SDK shipped in AISimulate."""

    def __init__(
        self,
        *,
        model: Any,
        worker_type: WorkerType,
        limits: EnginePerfLimits,
        attention_dp_size: int,
        max_observations: int,
    ) -> None:
        if worker_type not in ("prefill", "decode", "aggregated"):
            raise ValueError(f"invalid worker_type {worker_type!r}")
        if attention_dp_size <= 0:
            raise ValueError(
                f"attention_dp_size must be positive, got {attention_dp_size}"
            )
        self._model = model
        self._worker_type = worker_type
        self._limits = limits
        self._attention_dp_size = attention_dp_size
        self._load_averages = _AggLoadAverages(max_observations)

    @classmethod
    def best_available(
        cls,
        *,
        aic_config: Optional[dict[str, Any]],
        worker_type: WorkerType,
        limits: EnginePerfLimits,
        options: dict[str, int],
        attention_dp_size: int,
    ) -> AicCoreEnginePerfModel:
        if aic_config is None:
            model = AicForwardPassPerfModel.from_regression(options)
        else:
            model = AicForwardPassPerfModel.best_available(aic_config, options)
        return cls(
            model=model,
            worker_type=worker_type,
            limits=limits,
            attention_dp_size=attention_dp_size,
            max_observations=options["max_observations"],
        )

    def tune_with_fpms(self, iterations: list[list[ForwardPassMetrics]]) -> None:
        for metrics_by_rank in iterations:
            self._validate_metrics_by_rank(metrics_by_rank)
        self._model.tune_with_fpms(
            [
                [_fpm_to_aic_dict(metrics) for metrics in metrics_by_rank]
                for metrics_by_rank in iterations
            ]
        )
        self._load_averages.observe_iterations(iterations)

    def diagnostics(self) -> dict[str, Any]:
        diagnostics = self._model.diagnostics()
        if not isinstance(diagnostics, dict):
            raise TypeError(
                "aiconfigurator-core diagnostics must be a mapping, "
                f"got {type(diagnostics).__name__}"
            )
        return diagnostics

    def get_queued_prefill_time(
        self, metrics_by_rank: list[ForwardPassMetrics]
    ) -> Optional[float]:
        """Estimate queued-prefill drain time in seconds."""
        if self._worker_type == "decode":
            return None
        self._validate_metrics_by_rank(metrics_by_rank)

        plans = [
            _PrefillChunkPlan.build(
                metrics.queued_requests.sum_prefill_tokens,
                self._limits.max_num_batched_tokens,
            )
            for metrics in metrics_by_rank
        ]
        if all(plan.full_chunks == 0 and plan.tail_tokens == 0 for plan in plans):
            return 0.0

        breakpoints = {0}
        for plan in plans:
            if plan.full_chunks > 0:
                breakpoints.add(plan.full_chunks)
            if plan.tail_tokens > 0:
                breakpoints.add(plan.full_chunks + 1)

        total_s = 0.0
        for start, end in pairwise(sorted(breakpoints)):
            repeat_count = end - start
            chunk_metrics: list[ForwardPassMetrics] = []
            for snapshot, plan in zip(metrics_by_rank, plans, strict=True):
                chunk = plan.chunk_at(start, self._limits.max_num_batched_tokens)
                scheduled = ScheduledRequestMetrics(
                    num_prefill_requests=_estimate_prefill_request_count(
                        snapshot.queued_requests.num_prefill_requests, chunk
                    ),
                    sum_prefill_tokens=chunk,
                    var_prefill_length=(snapshot.queued_requests.var_prefill_length),
                    num_decode_requests=(
                        snapshot.scheduled_requests.num_decode_requests
                        if self._worker_type == "aggregated"
                        else 0
                    ),
                    sum_decode_kv_tokens=(
                        snapshot.scheduled_requests.sum_decode_kv_tokens
                        if self._worker_type == "aggregated"
                        else 0
                    ),
                    var_decode_kv_tokens=(
                        snapshot.scheduled_requests.var_decode_kv_tokens
                        if self._worker_type == "aggregated"
                        else 0.0
                    ),
                )
                chunk_metrics.append(_replace_scheduled(snapshot, scheduled))
            duration_s = self._estimate_forward_pass_seconds(chunk_metrics)
            if duration_s is None:
                return None
            total_s = _checked_duration_add(
                total_s,
                _checked_duration_mul(duration_s, repeat_count, "queued prefill time"),
                "queued prefill time",
            )
        return total_s

    def get_scheduled_decode_itl(
        self, metrics_by_rank: list[ForwardPassMetrics]
    ) -> Optional[float]:
        """Estimate one scheduled decode iteration in seconds."""
        if self._worker_type == "prefill":
            return None
        self._validate_metrics_by_rank(metrics_by_rank)

        estimates: list[ForwardPassMetrics] = []
        for snapshot in metrics_by_rank:
            source = snapshot.scheduled_requests
            prefill_requests = 0
            prefill_tokens = 0
            prefill_variance = 0.0
            prefill_kv_tokens = 0
            if self._worker_type == "aggregated":
                if source.sum_prefill_tokens > 0 or source.num_prefill_requests > 0:
                    prefill_requests = source.num_prefill_requests
                    prefill_tokens = source.sum_prefill_tokens
                else:
                    (
                        prefill_requests,
                        prefill_tokens,
                    ) = self._load_averages.typical_prefill_for_mixed_itl()
                prefill_variance = source.var_prefill_length
                prefill_kv_tokens = source.sum_prefill_kv_tokens

            estimates.append(
                _replace_scheduled(
                    snapshot,
                    ScheduledRequestMetrics(
                        num_prefill_requests=prefill_requests,
                        sum_prefill_tokens=prefill_tokens,
                        var_prefill_length=prefill_variance,
                        sum_prefill_kv_tokens=prefill_kv_tokens,
                        num_decode_requests=source.num_decode_requests,
                        sum_decode_kv_tokens=source.sum_decode_kv_tokens,
                        var_decode_kv_tokens=source.var_decode_kv_tokens,
                    ),
                )
            )
        return self._estimate_forward_pass_seconds(estimates)

    def find_engine_capacity_rps(
        self, request: EngineCapacityRequest
    ) -> Optional[EngineCapacity]:
        """Search sustainable per-engine RPS under request/SLA constraints."""
        if request.isl == 0:
            return None
        if self._worker_type == "prefill":
            return self._find_prefill_capacity(request)
        if self._worker_type == "decode":
            return self._find_decode_capacity(request)
        return self._find_aggregated_capacity(request)

    def _find_prefill_capacity(
        self, request: EngineCapacityRequest
    ) -> Optional[EngineCapacity]:
        prefill_isl = _effective_prefill_isl(request)
        max_batch = self._prefill_max_batch(prefill_isl)
        if max_batch == 0:
            return None

        best: Optional[EngineCapacity] = None
        for batch_size in _capacity_batch_sizes(max_batch):
            ttft_s = self._prefill_time_for_tokens(
                _saturating_mul_u32(prefill_isl, batch_size)
            )
            if ttft_s is None:
                return None
            if ttft_s == 0:
                continue
            capacity = EngineCapacity(
                rps=batch_size / ttft_s,
                ttft_s=ttft_s,
                e2e_latency_s=ttft_s,
                eligible=(
                    _sla_ok(ttft_s, request.ttft_sla_s)
                    and _sla_ok(None, request.itl_sla_s)
                    and _sla_ok(ttft_s, request.e2e_latency_sla_s)
                ),
            )
            best = _select_capacity(best, capacity, request.optimization_target)
        return best

    def _find_decode_capacity(
        self, request: EngineCapacityRequest
    ) -> Optional[EngineCapacity]:
        if request.osl == 0:
            return None
        accept_length = _normalized_accept_length(request.accept_length)
        context_length = _decode_context_length(request)
        max_batch = self._decode_max_batch(context_length)
        if max_batch == 0:
            return None

        best: Optional[EngineCapacity] = None
        for batch_size in _capacity_batch_sizes(max_batch):
            forward_s = self._decode_time_for_batch(batch_size, context_length)
            if forward_s is None:
                return None
            itl_s = forward_s / accept_length
            if itl_s == 0:
                continue
            capacity = EngineCapacity(
                rps=batch_size / (request.osl * itl_s),
                itl_s=itl_s,
                eligible=(
                    _sla_ok(None, request.ttft_sla_s)
                    and _sla_ok(itl_s, request.itl_sla_s)
                    and _sla_ok(None, request.e2e_latency_sla_s)
                ),
            )
            best = _select_capacity(best, capacity, request.optimization_target)
        return best

    def _find_aggregated_capacity(
        self, request: EngineCapacityRequest
    ) -> Optional[EngineCapacity]:
        if request.osl == 0 or self._limits.max_num_batched_tokens <= 1:
            return None

        prefill_isl = _effective_prefill_isl(request)
        accept_length = _normalized_accept_length(request.accept_length)
        context_length = _decode_context_length(request)
        hard_cap = min(
            self._decode_max_batch(context_length),
            max(1, self._limits.max_num_batched_tokens - 1),
        )
        if hard_cap == 0:
            return None

        best: Optional[EngineCapacity] = None
        for batch_size in _capacity_batch_sizes(hard_cap):
            if not _prefill_decode_balanced(
                prefill_isl,
                request.osl,
                batch_size,
                self._limits.max_num_batched_tokens,
                accept_length,
            ):
                continue

            decode_kv = _saturating_mul_u32(batch_size, context_length)
            prefill_per_iteration = min(
                self._limits.max_num_batched_tokens,
                _aggregate_prefill_per_iteration(
                    prefill_isl,
                    request.osl,
                    batch_size,
                    accept_length,
                ),
            )
            forward_s = self._mixed_time(prefill_per_iteration, batch_size, decode_kv)
            if forward_s is None:
                return None
            itl_s = forward_s / accept_length
            if itl_s == 0:
                continue

            ttft_s = self._aggregated_ttft(
                min(_U32_MAX, prefill_per_iteration + prefill_isl),
                batch_size,
                decode_kv,
            )
            if ttft_s is None:
                return None
            decode_tail_s = _checked_duration_mul(
                itl_s,
                max(0, request.osl - 1),
                "aggregate E2E latency",
            )
            e2e_s = _checked_duration_add(
                ttft_s, decode_tail_s, "aggregate E2E latency"
            )
            capacity = EngineCapacity(
                rps=batch_size / (request.osl * itl_s),
                ttft_s=ttft_s,
                itl_s=itl_s,
                e2e_latency_s=e2e_s,
                eligible=(
                    _sla_ok(ttft_s, request.ttft_sla_s)
                    and _sla_ok(itl_s, request.itl_sla_s)
                    and _sla_ok(e2e_s, request.e2e_latency_sla_s)
                ),
            )
            best = _select_capacity(best, capacity, request.optimization_target)
        return best

    def _prefill_time_for_tokens(self, tokens: int) -> Optional[float]:
        return self._prefill_chunk_time(
            tokens,
            lambda chunk: self._estimate_forward_pass_seconds(
                _synthetic_prefill_by_rank(chunk, self._attention_dp_size)
            ),
        )

    def _decode_time_for_batch(
        self, batch_size: int, context_length: int
    ) -> Optional[float]:
        return self._estimate_forward_pass_seconds(
            _synthetic_decode_by_rank(
                batch_size,
                _saturating_mul_u32(batch_size, context_length),
                self._attention_dp_size,
            )
        )

    def _mixed_time(
        self, prefill_tokens: int, decode_requests: int, decode_kv: int
    ) -> Optional[float]:
        return self._estimate_forward_pass_seconds(
            _synthetic_mixed_by_rank(
                prefill_tokens,
                decode_requests,
                decode_kv,
                self._attention_dp_size,
            )
        )

    def _aggregated_ttft(
        self,
        queued_prefill_tokens: int,
        current_decode_requests: int,
        current_decode_kv: int,
    ) -> Optional[float]:
        return self._prefill_chunk_time(
            queued_prefill_tokens,
            lambda chunk: self._mixed_time(
                chunk, current_decode_requests, current_decode_kv
            ),
        )

    def _prefill_chunk_time(
        self,
        tokens: int,
        estimate_chunk: Callable[[int], Optional[float]],
    ) -> Optional[float]:
        plan = _PrefillChunkPlan.build(tokens, self._limits.max_num_batched_tokens)
        total_s = 0.0
        if plan.full_chunks > 0:
            duration_s = estimate_chunk(self._limits.max_num_batched_tokens)
            if duration_s is None:
                return None
            total_s = _checked_duration_add(
                total_s,
                _checked_duration_mul(
                    duration_s, plan.full_chunks, "prefill chunk time"
                ),
                "prefill chunk time",
            )
        if plan.tail_tokens > 0:
            duration_s = estimate_chunk(plan.tail_tokens)
            if duration_s is None:
                return None
            total_s = _checked_duration_add(total_s, duration_s, "prefill chunk time")
        return total_s

    def _prefill_max_batch(self, isl: int) -> int:
        if isl > self._limits.max_num_batched_tokens:
            return 1
        return min(
            self._limits.max_num_seqs,
            self._limits.max_num_batched_tokens // max(1, isl),
        )

    def _decode_max_batch(self, context_length: int) -> int:
        if context_length == 0:
            return 0
        return min(
            self._limits.max_num_seqs,
            self._limits.max_kv_tokens // context_length,
        )

    def _validate_metrics_by_rank(
        self, metrics_by_rank: list[ForwardPassMetrics]
    ) -> None:
        if not metrics_by_rank:
            raise ValueError("at least one attention-DP rank metric is required")
        if len(metrics_by_rank) != self._attention_dp_size:
            raise ValueError(
                f"expected {self._attention_dp_size} attention-DP rank metrics, "
                f"got {len(metrics_by_rank)}"
            )
        seen: set[int] = set()
        for metrics in metrics_by_rank:
            if metrics.version not in (0, FPM_VERSION):
                raise ValueError(
                    f"unsupported FPM version {metrics.version}; expected {FPM_VERSION}"
                )
            if self._attention_dp_size > 1:
                if not 0 <= metrics.dp_rank < self._attention_dp_size:
                    raise ValueError(
                        f"dp_rank {metrics.dp_rank} out of range for "
                        f"attention_dp_size {self._attention_dp_size}"
                    )
                if metrics.dp_rank in seen:
                    raise ValueError(f"duplicate dp_rank {metrics.dp_rank}")
                seen.add(metrics.dp_rank)

    def _estimate_forward_pass_seconds(
        self, metrics_by_rank: list[ForwardPassMetrics]
    ) -> Optional[float]:
        estimate_ms = self._model.estimate_forward_pass_time_ms(
            [_fpm_to_aic_dict(metrics) for metrics in metrics_by_rank]
        )
        if estimate_ms is None:
            return None
        if not math.isfinite(estimate_ms):
            raise ValueError(
                f"AIC forward-pass estimate must be finite, got {estimate_ms}"
            )
        return _checked_duration_seconds(
            max(0.0, estimate_ms) / 1000.0,
            "AIC forward-pass estimate",
        )


def _fpm_to_aic_dict(metrics: ForwardPassMetrics) -> dict[str, Any]:
    payload = msgspec.to_builtins(metrics)
    if not isinstance(payload, dict):
        raise TypeError(
            f"ForwardPassMetrics must serialize to a mapping, got {type(payload).__name__}"
        )
    if metrics.version == 0:
        payload["version"] = FPM_VERSION
    return payload


def _replace_scheduled(
    source: ForwardPassMetrics, scheduled: ScheduledRequestMetrics
) -> ForwardPassMetrics:
    return ForwardPassMetrics(
        version=FPM_VERSION if source.version == 0 else source.version,
        worker_id=source.worker_id,
        dp_rank=source.dp_rank,
        counter_id=source.counter_id,
        wall_time=source.wall_time,
        scheduled_requests=scheduled,
        queued_requests=QueuedRequestMetrics(),
    )


def _empty_fpm(dp_rank: int) -> ForwardPassMetrics:
    return ForwardPassMetrics(dp_rank=dp_rank)


def _synthetic_prefill_by_rank(tokens: int, ranks: int) -> list[ForwardPassMetrics]:
    metrics_by_rank: list[ForwardPassMetrics] = []
    for dp_rank in range(max(1, ranks)):
        rank_tokens = _split_total(tokens, ranks, dp_rank)
        metrics_by_rank.append(
            _replace_scheduled(
                _empty_fpm(dp_rank),
                ScheduledRequestMetrics(
                    num_prefill_requests=int(rank_tokens > 0),
                    sum_prefill_tokens=rank_tokens,
                ),
            )
        )
    return metrics_by_rank


def _synthetic_decode_by_rank(
    num_requests: int, kv_tokens: int, ranks: int
) -> list[ForwardPassMetrics]:
    metrics_by_rank: list[ForwardPassMetrics] = []
    for dp_rank in range(max(1, ranks)):
        metrics_by_rank.append(
            _replace_scheduled(
                _empty_fpm(dp_rank),
                ScheduledRequestMetrics(
                    num_decode_requests=_split_total(num_requests, ranks, dp_rank),
                    sum_decode_kv_tokens=_split_total(kv_tokens, ranks, dp_rank),
                ),
            )
        )
    return metrics_by_rank


def _synthetic_mixed_by_rank(
    prefill_tokens: int,
    decode_requests: int,
    decode_kv_tokens: int,
    ranks: int,
) -> list[ForwardPassMetrics]:
    metrics_by_rank: list[ForwardPassMetrics] = []
    for dp_rank in range(max(1, ranks)):
        rank_prefill = _split_total(prefill_tokens, ranks, dp_rank)
        metrics_by_rank.append(
            _replace_scheduled(
                _empty_fpm(dp_rank),
                ScheduledRequestMetrics(
                    num_prefill_requests=int(rank_prefill > 0),
                    sum_prefill_tokens=rank_prefill,
                    num_decode_requests=_split_total(decode_requests, ranks, dp_rank),
                    sum_decode_kv_tokens=_split_total(decode_kv_tokens, ranks, dp_rank),
                ),
            )
        )
    return metrics_by_rank


def _split_total(total: int, ranks: int, rank: int) -> int:
    normalized_ranks = max(1, ranks)
    base, remainder = divmod(total, normalized_ranks)
    return base + int(rank < remainder)


def _saturating_mul_u32(lhs: int, rhs: int) -> int:
    return min(_U32_MAX, lhs * rhs)


def _checked_duration_seconds(value: float, context: str) -> float:
    if not math.isfinite(value) or value >= _DURATION_EXCLUSIVE_MAX_SECONDS:
        raise ValueError(f"{context} overflow")
    return max(0.0, value)


def _checked_duration_add(lhs: float, rhs: float, context: str) -> float:
    return _checked_duration_seconds(lhs + rhs, context)


def _checked_duration_mul(value: float, factor: int, context: str) -> float:
    return _checked_duration_seconds(value * factor, context)


def _estimate_prefill_request_count(num_requests: int, chunk_tokens: int) -> int:
    if chunk_tokens == 0:
        return 0
    return max(1, num_requests)


def _decode_context_length(request: EngineCapacityRequest) -> int:
    return max(1, min(_U32_MAX, request.isl + request.osl // 2))


def _effective_prefill_isl(request: EngineCapacityRequest) -> int:
    scale = 1.0 - _clamp_kv_hit_rate(request.kv_hit_rate)
    return max(
        1,
        _ceil_u32(
            request.isl * scale,
            "effective prefill ISL after kv_hit_rate discount",
        ),
    )


def _normalized_accept_length(value: float) -> float:
    if math.isfinite(value) and value > 1.0:
        return value
    return 1.0


def _clamp_kv_hit_rate(value: Optional[float]) -> float:
    if value is None or not math.isfinite(value):
        return 0.0
    return min(MAX_KV_HIT_RATE_DISCOUNT, max(0.0, value))


def _prefill_decode_balanced(
    isl: int,
    osl: int,
    batch_size: int,
    max_num_batched_tokens: int,
    accept_length: float,
) -> bool:
    prefill_budget = max(0, max_num_batched_tokens - batch_size)
    if prefill_budget == 0:
        return False
    return (
        _aggregate_prefill_per_iteration(isl, osl, batch_size, accept_length)
        <= prefill_budget
    )


def _aggregate_prefill_per_iteration(
    isl: int, osl: int, batch_size: int, accept_length: float
) -> int:
    return _ceil_u32(
        batch_size * _normalized_accept_length(accept_length) * isl / max(1, osl),
        "aggregate prefill per iteration",
    )


def _capacity_batch_sizes(max_batch: int) -> list[int]:
    if max_batch <= 0:
        return []
    if max_batch <= MAX_CAPACITY_SEARCH_CANDIDATES:
        return list(range(1, max_batch + 1))
    span = max_batch - 1
    denominator = MAX_CAPACITY_SEARCH_CANDIDATES - 1
    return [
        1 + index * span // denominator
        for index in range(MAX_CAPACITY_SEARCH_CANDIDATES)
    ]


def _sla_ok(value_s: Optional[float], sla_s: Optional[float]) -> bool:
    if sla_s is None:
        return True
    return value_s is not None and value_s <= sla_s


def _select_capacity(
    current: Optional[EngineCapacity],
    candidate: EngineCapacity,
    target: OptimizationTarget,
) -> EngineCapacity:
    if current is None:
        return candidate
    if not current.eligible and candidate.eligible:
        return candidate
    if current.eligible and not candidate.eligible:
        return current
    if target == OptimizationTarget.THROUGHPUT:
        return candidate if candidate.rps > current.rps else current

    current_latency = _capacity_latency(current)
    candidate_latency = _capacity_latency(candidate)
    return candidate if candidate_latency < current_latency else current


def _capacity_latency(capacity: EngineCapacity) -> float:
    for value in (
        capacity.e2e_latency_s,
        capacity.ttft_s,
        capacity.itl_s,
    ):
        if value is not None:
            return value
    return math.inf


def _ceil_u32(value: float, name: str) -> int:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value <= 0:
        return 0
    rounded = math.ceil(value)
    if rounded > _U32_MAX:
        raise ValueError(f"{name} exceeds u32::MAX")
    return rounded
