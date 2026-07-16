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

    ``at_s`` is an absolute wall-clock time for the native adapter and a
    simulated time for replay. ``at_monotonic_s`` is the matching scheduler
    timestamp used to make observation-prefetch and plugin-dispatch cadence
    decisions against the same clock value.
    """

    at_s: float
    at_monotonic_s: Optional[float] = None

    # What decisions the core will make on this tick
    run_load_scaling: bool = False
    run_throughput_scaling: bool = False

    # What data the adapter should collect before calling on_tick
    need_traffic_metrics: bool = False
    # True requests the full throughput traffic snapshot; False with
    # need_traffic_metrics=True requests the cheaper load-only
    # kv-hit-rate observation.
    use_full_traffic_metrics: bool = False
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
    accept_length: Optional[float] = None


@dataclass
class WorkerCounts:
    """Current worker inventory as reported by the adapter."""

    ready_num_prefill: Optional[int] = None
    ready_num_decode: Optional[int] = None
    expected_num_prefill: Optional[int] = None
    expected_num_decode: Optional[int] = None
    prefill_scaling_in_progress: bool = False
    decode_scaling_in_progress: bool = False


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
    """Intermediate decision data populated by the planner core for
    observability.  The adapter layer reads these to set Prometheus
    metrics and feed the diagnostics recorder.
    """

    # Load-scaling: max estimated latency across engines (ms)
    estimated_ttft_ms: Optional[float] = None
    estimated_itl_ms: Optional[float] = None

    # Throughput-scaling: predicted next-interval traffic and last-value
    # runtime metadata used by throughput decisions.
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

    # Plugin-pipeline fields below. Legacy callers that bypass the pipeline
    # may leave them empty. Downstream readers must treat "empty" as
    # "not available for this tick".

    # PROPOSE/RECONCILE/CONSTRAIN overrides contributed this tick.
    # Tuple: (plugin_id, stage, override_type, component_key, value).
    # override_type ∈ {"SET", "AT_LEAST", "AT_MOST", "REJECT"};
    # component_key = ``sub_component_type`` (one bucket per type in this
    # PR — multi-pool addressing is deferred to the hierarchical planner
    # PR); value = replica target (``-1`` for REJECT).
    plugin_overrides: list[tuple[str, str, str, str, int]] = field(default_factory=list)

    # Per-component reconcile reason.  Keyed by ``component_key`` as
    # above; value is a short audit string such as
    # ``"set_by_<plugin_id>"``, ``"clamped_to_floor"``,
    # ``"clamped_to_ceiling"``, or ``"passthrough"``.
    reconcile_reasons: dict[str, str] = field(default_factory=dict)

    # plugin_id list of HOLD_LAST cache replays this tick (plugin was
    # skipped because its execution_interval hadn't elapsed, and its
    # previous output is being reused).
    held_over_plugins: list[str] = field(default_factory=list)

    # Pipeline execute_action — one of ``"apply"``,
    # ``"skip_short_circuit"``, ``"skip_no_targets"``,
    # ``"skip_tick_timeout"``.  Mirrors
    # ``PipelineOutcome.execute_action``. ``None`` means no pipeline action
    # was recorded. Same information is also emitted as Prometheus
    # ``tick_skip_reasons_total`` etc., but exposing it on
    # ``PlannerEffects.diagnostics`` lets in-process consumers (replay
    # adapter, diagnostics recorder) see the action without scraping
    # metrics.
    execute_action: Optional[str] = None

    # Why a tick short-circuited (e.g. ``"propose: my-plugin: ..."``).
    # Populated when ``execute_action == "skip_short_circuit"``; empty
    # otherwise.  Mirrors ``PipelineOutcome.short_circuit_reason``.
    short_circuit_reason: str = ""

    # Audit-quality breadcrumbs emitted by the pipeline (chain-augment
    # warnings, CONSTRAIN SET drops, etc.).  Mirrors
    # ``PipelineOutcome.audit_events``.
    audit_events: list[str] = field(default_factory=list)


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
    kv_cache_block_size: Optional[int] = None
    speculative_nextn: Optional[int] = None


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
