# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit-input types for the planner core.

These types form the boundary between the planner core (pure decision logic)
and any adapter (native runtime, replay harness, tests).  The core receives
``TickInput`` and returns ``PlannerEffects``; the adapter fills the input
based on the previous tick's ``ScheduledTick`` requirements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from dynamo.common.forward_pass_metrics import ForwardPassMetrics


@dataclass
class ScheduledTick:
    """Declares when the core next needs to be called, what data it needs,
    and what decisions to make.

    All times are absolute seconds (wall clock for native adapter,
    simulated clock for replay).
    """

    at_s: float

    # What decisions the core will make on this tick
    run_load_scaling: bool = False
    run_throughput_scaling: bool = False

    # What data the adapter should collect before calling on_tick
    need_traffic_metrics: bool = False
    traffic_metrics_duration_s: float = 0.0
    need_worker_states: bool = False
    need_worker_fpm: bool = False


@dataclass
class TrafficObservation:
    """Aggregated traffic metrics over an observation window."""

    duration_s: float
    num_req: float
    isl: float
    osl: float
    kv_hit_rate: Optional[float] = None


@dataclass
class WorkerCounts:
    """Current worker inventory as reported by the adapter."""

    ready_num_prefill: Optional[int] = None
    ready_num_decode: Optional[int] = None
    expected_num_prefill: Optional[int] = None
    expected_num_decode: Optional[int] = None


@dataclass
class FpmObservations:
    """Per-engine ForwardPassMetrics keyed by (worker_id, dp_rank)."""

    prefill: Optional[dict[tuple[str, int], ForwardPassMetrics]] = None
    decode: Optional[dict[tuple[str, int], ForwardPassMetrics]] = None


@dataclass
class TickInput:
    """What the adapter provides to the core on each tick.

    Fields are filled according to the previous ``ScheduledTick``'s
    declared requirements.
    """

    now_s: float
    traffic: Optional[TrafficObservation] = None
    worker_counts: Optional[WorkerCounts] = None
    fpm_observations: Optional[FpmObservations] = None


@dataclass
class ScalingDecision:
    """Desired replica counts.  ``None`` means the core has no opinion
    on that component (e.g. prefill-only planner leaves decode as None).
    """

    num_prefill: Optional[int] = None
    num_decode: Optional[int] = None


@dataclass
class TickDiagnostics:
    """Intermediate decision data populated by the state machine for
    observability.  The adapter layer reads these to set Prometheus
    metrics and feed the diagnostics recorder.
    """

    # Load-scaling: max estimated latency across engines (ms)
    estimated_ttft_ms: Optional[float] = None
    estimated_itl_ms: Optional[float] = None

    # Throughput-scaling: predicted next-interval traffic
    predicted_num_req: Optional[float] = None
    predicted_isl: Optional[float] = None
    predicted_osl: Optional[float] = None
    predicted_kv_hit_rate: Optional[float] = None

    # Throughput-scaling: single-engine capacity under SLA (req/s)
    engine_rps_prefill: Optional[float] = None
    engine_rps_decode: Optional[float] = None

    # Throughput-scaling: lower bound on replicas
    throughput_lower_bound_prefill: Optional[int] = None
    throughput_lower_bound_decode: Optional[int] = None

    # Scaling decision reasons (set by the mixin that ran)
    # Aggregate reasons (agg mode, or combined disagg).
    load_decision_reason: Optional[str] = None
    throughput_decision_reason: Optional[str] = None
    # Per-component reasons (populated in disagg mode for separate
    # prefill / decode decision timelines).
    load_decision_reason_prefill: Optional[str] = None
    load_decision_reason_decode: Optional[str] = None
    throughput_decision_reason_prefill: Optional[str] = None
    throughput_decision_reason_decode: Optional[str] = None


@dataclass
class PlannerEffects:
    """What the core returns after processing a tick."""

    scale_to: Optional[ScalingDecision] = None
    next_tick: Optional[ScheduledTick] = None
    diagnostics: TickDiagnostics = field(default_factory=TickDiagnostics)


@dataclass
class EngineCapabilities:
    """Static capabilities for a single engine stage (prefill or decode)."""

    num_gpu: Optional[int] = None
    max_num_batched_tokens: Optional[int] = None
    max_num_seqs: Optional[int] = None
    context_length: Optional[int] = None
    max_kv_tokens: Optional[int] = None


@dataclass
class WorkerCapabilities:
    """Static per-engine capabilities discovered at startup from MDC.

    Provided once when constructing the planner core.  In native mode
    these come from ``WorkerInfo`` (resolved via MDC / DGD); in replay
    they come from the simulated engine args.

    For agg mode, only ``decode`` is populated (single engine type).
    """

    prefill: Optional[EngineCapabilities] = None
    decode: Optional[EngineCapabilities] = None
