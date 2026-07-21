# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Input schema for a Spica smart-search run.

These Pydantic models are the single source of truth for the search inputs. See
``docs/components/profiler/spica/overview.md`` for the experimental design:

- :class:`SearchSpace`        — the knobs to sweep + pinned context, per component
- :class:`Workload`           — the traffic every candidate is evaluated against
- :class:`OptimizationGoal`   — what "better" means + the SLA constraint
- :class:`SweepConfig`        — sweep run-control
- :class:`SmartSearchConfig`  — top-level bundle; one YAML maps to this
- :class:`Candidate`          — one evaluated configuration + its replay metrics

Field names are snake_case to match AIConfigurator's ``Task`` convention so the
eventual merge into an AIC sweep task is mechanical.
"""

from __future__ import annotations

import math
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class OptimizationTarget(str, Enum):
    """What the search optimizes for.

    All members except ``pareto`` are scalar (single-objective) targets. ``pareto`` is a
    multi-objective mode: the search optimizes the Pareto tradeoff between the scalar
    targets listed in :attr:`OptimizationGoal.pareto_objectives` (default: throughput per
    GPU vs per-user throughput — the InferenceX tok/s/gpu vs tok/s/user frontier).
    """

    THROUGHPUT = "throughput"  # maximize replay throughput
    THROUGHPUT_PER_GPU = (
        "throughput_per_gpu"  # maximize throughput / avg GPU (tok/s/gpu)
    )
    THROUGHPUT_PER_USER = (
        "throughput_per_user"  # maximize mean per-user output throughput (tok/s/user)
    )
    E2E_LATENCY = "e2e_latency"  # minimize mean end-to-end latency
    GOODPUT = "goodput"  # maximize SLA-satisfying throughput
    GOODPUT_PER_GPU = "goodput_per_gpu"  # maximize goodput / avg GPU (tok/s/gpu)
    PARETO = "pareto"  # multi-objective: Pareto front over pareto_objectives

    @property
    def maximize(self) -> bool:
        """True when larger is better (everything except e2e_latency).

        Raises for ``pareto`` — it has no single direction; use the per-objective
        directions in :attr:`OptimizationGoal.pareto_objectives` instead.
        """
        if self is OptimizationTarget.PARETO:
            raise ValueError("'pareto' is multi-objective and has no scalar direction")
        return self is not OptimizationTarget.E2E_LATENCY

    @property
    def planner_optimization_target(self) -> str:
        """The dynamo planner ``optimization_target`` this sweep goal maps to.

        The planner's scaling objective should match what the sweep optimizes:
        ``goodput``/``goodput_per_gpu`` -> ``"sla"`` (SLA-based scaling, the only
        mode that uses ttft/itl and enables predictive throughput scaling);
        ``throughput``/``throughput_per_gpu``/``throughput_per_user`` -> ``"throughput"``;
        ``e2e_latency`` -> ``"latency"`` (both reactive, no SLA). ``pareto`` ->
        ``"throughput"`` (its default objectives are throughput-based; no SLA).
        """
        return {
            OptimizationTarget.THROUGHPUT: "throughput",
            OptimizationTarget.THROUGHPUT_PER_GPU: "throughput",
            OptimizationTarget.THROUGHPUT_PER_USER: "throughput",
            OptimizationTarget.E2E_LATENCY: "latency",
            OptimizationTarget.GOODPUT: "sla",
            OptimizationTarget.GOODPUT_PER_GPU: "sla",
            OptimizationTarget.PARETO: "throughput",
        }[self]


class SLATarget(BaseModel):
    """Per-request latency bounds in ms. Set ttft_ms+itl_ms, or e2e_ms."""

    model_config = ConfigDict(extra="forbid")

    ttft_ms: float | None = Field(default=None, gt=0)
    itl_ms: float | None = Field(default=None, gt=0)
    e2e_ms: float | None = Field(default=None, gt=0)


# Goodput-based scalar targets — the only ones that need an SLA (their metric counts
# only SLA-satisfying requests). Used to gate the SLA requirement on both the scalar
# target and the per-objective list under a pareto goal.
_SLA_TARGETS = frozenset(
    {OptimizationTarget.GOODPUT, OptimizationTarget.GOODPUT_PER_GPU}
)

# Default Pareto objectives: throughput per GPU (y) vs mean per-user throughput (x) —
# the InferenceX tok/s/gpu vs tok/s/user frontier.
_DEFAULT_PARETO_OBJECTIVES = (
    OptimizationTarget.THROUGHPUT_PER_GPU,
    OptimizationTarget.THROUGHPUT_PER_USER,
)


class OptimizationGoal(BaseModel):
    """User-owned objective and SLA. Pinned; never searched."""

    model_config = ConfigDict(extra="forbid")

    target: OptimizationTarget = OptimizationTarget.THROUGHPUT
    sla: SLATarget | None = (
        None  # required for goodput / goodput_per_gpu (scalar or pareto objective)
    )
    # Only meaningful when target == pareto: the >=2 scalar objectives whose Pareto
    # front is sought. None -> the default pair (throughput_per_gpu, throughput_per_user).
    pareto_objectives: list[OptimizationTarget] | None = None

    @property
    def resolved_pareto_objectives(self) -> list[OptimizationTarget]:
        """The effective Pareto objective list: the configured one, or the default pair only
        when unset (``None``). An explicitly-supplied empty/short list is kept as-is so the
        validator's ``len < 2`` guard rejects it (rather than silently using the default).
        """
        return (
            list(_DEFAULT_PARETO_OBJECTIVES)
            if self.pareto_objectives is None
            else list(self.pareto_objectives)
        )

    @property
    def is_pareto(self) -> bool:
        return self.target is OptimizationTarget.PARETO

    @model_validator(mode="after")
    def _validate_goal(self) -> OptimizationGoal:
        # pareto_objectives only applies to a pareto target.
        if not self.is_pareto and self.pareto_objectives is not None:
            raise ValueError("pareto_objectives is only valid when target is 'pareto'")
        if self.is_pareto:
            objs = self.resolved_pareto_objectives
            if len(objs) < 2:
                raise ValueError("a pareto goal needs at least 2 objectives")
            if OptimizationTarget.PARETO in objs:
                raise ValueError(
                    "pareto_objectives cannot contain 'pareto' itself (objectives must be scalar)"
                )
            if len(set(objs)) != len(objs):
                raise ValueError(
                    f"pareto_objectives must be distinct, got {[o.value for o in objs]}"
                )
            effective = set(objs)
        else:
            effective = {self.target}
        # Any goodput-based objective (scalar target or pareto objective) needs an SLA.
        needs_sla = bool(effective & _SLA_TARGETS)
        has_sla = self.sla is not None and (
            self.sla.e2e_ms is not None
            or (self.sla.ttft_ms is not None and self.sla.itl_ms is not None)
        )
        if needs_sla and not has_sla:
            culprits = sorted(t.value for t in (effective & _SLA_TARGETS))
            raise ValueError(
                f"{culprits} require an SLA target (ttft_ms+itl_ms or e2e_ms)"
            )
        return self


class Workload(BaseModel):
    """Traffic every candidate is evaluated against (KV load may be searched for Pareto).

    Exactly one of **four load shapes** (all replayable with or without the planner):

    1. **mooncake trace** — set ``trace_path``. Open-loop at the trace's arrival
       timestamps (scale with ``arrival_speedup_ratio``); set ``replay_concurrency``
       to drive it **closed-loop** (cap N in flight, ignore timestamps).
    2. **synthetic request-rate** — set ``request_rate`` (+ ``isl``/``osl``/``num_request_ratio``):
       open-loop at a fixed QPS.
    3. **synthetic concurrency** — set ``concurrency`` (+ ``isl``/``osl``/``num_request_ratio``):
       closed-loop, cap N in flight.
    4. **synthetic KV load** — set ``kv_load_ratio`` (+ ``isl``/``osl``/``num_request_ratio``):
       closed-loop, with concurrency derived from each candidate's aggregate decode/agg KV
       capacity. A two-value range is searchable only under a ``pareto`` goal.

    The mode is inferred from which field is set; see the validator.

    ``concurrency`` is always one fixed positive integer. Under a ``pareto`` goal,
    ``kv_load_ratio`` may instead be a ``[min, max]`` continuous search range; when no
    synthetic load is specified, :class:`SmartSearchConfig` defaults that range to ``[0, 1]``.

    ``num_request_ratio`` (synthetic only) sets the request count **relative to the load**:
    ``num_requests = round(num_request_ratio * load)`` where ``load`` is ``concurrency``
    (closed-loop) or ``request_rate`` (open-loop). So the synthetic trace length scales with
    the concrete concurrency automatically — e.g. ratio 10 at concurrency 256 -> 2560 requests.
    """

    model_config = ConfigDict(extra="forbid")

    # synthetic workload (used when trace_path is unset): exactly one of
    # request_rate (open-loop QPS), concurrency (fixed closed-loop in-flight cap), or
    # kv_load_ratio (candidate-relative closed-loop load).
    isl: int | None = None
    osl: int | None = None
    concurrency: int | None = None
    kv_load_ratio: float | list[float] | None = None
    request_rate: float | None = None
    num_request_ratio: float | None = (
        None  # request count multiplier for concrete concurrency or request_rate
    )
    shared_prefix_ratio: float = 0.0  # cache-locality / prefix sharing
    num_prefix_groups: int = 0
    turns_per_session: int = 1  # multi-turn sessions
    inter_turn_delay_ms: float = 0.0  # think-time between turns (multi-turn synthetic)

    # dynamic trace source (mutually exclusive with the synthetic fields)
    trace_path: str | None = None
    trace_format: str = "mooncake"  # replay-ready trace schema
    arrival_speedup_ratio: float = 1.0  # scale trace inter-arrival times
    # Closed-loop replay over a *trace*: cap in-flight requests at this many (the
    # trace's timestamps are ignored; a new request starts as one finishes). For a
    # *synthetic* closed-loop workload use ``concurrency`` or ``kv_load_ratio`` instead.
    replay_concurrency: int | None = None

    @property
    def is_trace_based(self) -> bool:
        return self.trace_path is not None

    @property
    def is_synthetic(self) -> bool:
        return self.trace_path is None

    @property
    def kv_load_ratio_range(self) -> tuple[float, float] | None:
        """The Pareto-only continuous KV-load range, or ``None`` for a scalar/other load."""
        if isinstance(self.kv_load_ratio, list):
            return float(self.kv_load_ratio[0]), float(self.kv_load_ratio[1])
        return None

    def effective_in_flight_cap(
        self, concurrency_override: int | None = None
    ) -> int | None:
        """Closed-loop in-flight cap (``None`` = open-loop). ``concurrency_override`` (the
        per-trial value derived from KV load) wins; then ``replay_concurrency`` for a
        trace, then the fixed ``concurrency`` for a synthetic workload. KV-load mode always
        supplies its candidate-derived concurrency as the override."""
        if self.trace_path is not None:
            return self.replay_concurrency
        if concurrency_override is not None:
            return concurrency_override
        if self.concurrency is not None:
            return self.concurrency
        return None

    def resolved_request_count(self, concurrency_override: int | None = None) -> int:
        """Synthetic request count = ``round(num_request_ratio * load)`` (>= 1), where
        ``load`` is the in-flight concurrency (closed-loop) or the request rate (open-loop).
        ``concurrency_override`` is the candidate-specific concurrency in KV-load mode.
        """
        if concurrency_override is not None:
            load: float = concurrency_override
        elif self.concurrency is not None:
            load = self.concurrency
        elif self.request_rate is not None:
            load = self.request_rate
        else:
            raise ValueError(
                "resolved_request_count needs a concurrency_override for a kv_load_ratio workload"
            )
        return max(1, round((self.num_request_ratio or 0.0) * load))

    @property
    def synthetic_arrival_interval_ms(self) -> float:
        """Mean inter-arrival for a synthetic request-rate workload (1.0 default;
        ignored in closed-loop / concurrency mode)."""
        if self.request_rate:
            return 1000.0 / self.request_rate
        return 1.0

    @model_validator(mode="after")
    def _validate_workload(self) -> Workload:
        synthetic_only = (
            "isl",
            "osl",
            "request_rate",
            "concurrency",
            "kv_load_ratio",
            "num_request_ratio",
        )
        if self.trace_path is not None:
            set_syn = [n for n in synthetic_only if getattr(self, n) is not None]
            if set_syn:
                raise ValueError(
                    f"trace workload (trace_path set) must not set synthetic fields {set_syn}"
                )
            if self.replay_concurrency is not None and self.replay_concurrency <= 0:
                raise ValueError(
                    f"replay_concurrency must be a positive integer, got {self.replay_concurrency}"
                )
            return self
        # synthetic: exactly one load mode, plus isl/osl/num_request_ratio
        loads = [
            n
            for n in ("request_rate", "concurrency", "kv_load_ratio")
            if getattr(self, n) is not None
        ]
        if len(loads) != 1:
            raise ValueError(
                "a synthetic workload needs exactly one of request_rate, concurrency, or kv_load_ratio "
                "(or set trace_path for a trace workload)"
            )
        missing = [
            n for n in ("isl", "osl", "num_request_ratio") if getattr(self, n) is None
        ]
        if missing:
            raise ValueError(f"a synthetic workload requires {missing}")
        if self.replay_concurrency is not None:
            raise ValueError(
                "replay_concurrency is for trace workloads; use 'concurrency' for synthetic closed-loop"
            )
        if self.concurrency is not None and (
            isinstance(self.concurrency, bool) or self.concurrency <= 0
        ):
            raise ValueError(
                f"concurrency must be a positive integer, got {self.concurrency!r}"
            )
        if self.kv_load_ratio is not None:
            ratios = (
                self.kv_load_ratio
                if isinstance(self.kv_load_ratio, list)
                else [self.kv_load_ratio]
            )
            if isinstance(self.kv_load_ratio, list) and len(ratios) != 2:
                raise ValueError("kv_load_ratio range must contain exactly [min, max]")
            if any(
                not math.isfinite(float(value)) or float(value) < 0.0
                for value in ratios
            ):
                raise ValueError(
                    f"kv_load_ratio values must be finite and non-negative, got {self.kv_load_ratio!r}"
                )
            if isinstance(self.kv_load_ratio, list) and float(ratios[0]) >= float(
                ratios[1]
            ):
                raise ValueError(
                    f"kv_load_ratio range needs min < max, got {self.kv_load_ratio!r}"
                )
        for name in ("request_rate", "isl", "osl", "num_request_ratio"):
            v = getattr(self, name)
            if v is not None and v <= 0:
                raise ValueError(f"{name} must be positive, got {v}")
        return self


# Allowed choices for each swept search-space dimension. A configured value must
# be a non-empty subset of these (one or more); the field defaults below use the
# full set (or a sensible subset, e.g. ``backend``). Centralized here so the
# candidate generator can reuse it. Pinned scalars and the generated
# ``parallel_configs`` are intentionally not choice-constrained.
SEARCH_CHOICES: dict[str, tuple] = {
    "deployment_mode": ("disagg", "agg"),
    "backend": ("vllm", "sglang", "trtllm"),
    "prefill_max_num_batched_tokens": (8192, 16384, 32768),
    "prefill_max_num_seqs": (1, 2, 4, 8, 16, 32, 64, 128, 256),
    "decode_max_num_batched_tokens": (8192,),
    "decode_max_num_seqs": (256, 512, 1024),
    "agg_max_num_batched_tokens": (8192, 16384, 32768),
    "agg_max_num_seqs": (256, 512, 1024),
    "router_mode": ("kv_router", "round_robin"),
    "overlap_score_credit": (0.0, 0.5, 1.0),  # kv-router hard-caps this at 1.0
    "prefill_load_scale": (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0),
    "host_cache_hit_weight": (0.5, 0.75, 1.0),
    "disk_cache_hit_weight": (0.0, 0.25, 0.5),
    "router_temperature": (0.0, 0.2, 0.5, 1.0),
    "planner_scaling_policy": (
        "disabled",
        "throughput_180_5",
        "throughput_600_5",
        "load_180_5",
        "load_180_10",
        "hybrid_180_5",
        "hybrid_600_5",
    ),
    "planner_fpm_sampling": ("small", "default", "large", "fine"),
    "planner_load_sensitivity": ("aggressive", "default", "conservative"),
    "load_predictor_candidates": (
        "constant_last",
        "arima_raw",
        "arima_log1p",
        "prophet_w20_raw",
        "prophet_w20_log1p",
        "prophet_w50_raw",
        "prophet_w50_log1p",
        "kalman_default_raw",
        "kalman_default_log1p",
        "kalman_reactive_raw",
        "kalman_reactive_log1p",
    ),
}

# Composite knobs accept either a preset id (a string from SEARCH_CHOICES) or a
# dict that pins the unrolled fields directly (the escape hatch — search a custom
# value a preset doesn't offer). A dict entry must be self-contained: its keys are
# exactly that composite's unrolled field names (no partial/merge). The legality of
# the values (perfect-square fpm bucket, interval > 0, etc.) is validated downstream
# by Dynamo's PlannerConfig; here we only gate the key set. See
# docs/components/profiler/spica/search-space.md.
COMPOSITE_DICT_KEYS: dict[str, frozenset[str]] = {
    "planner_scaling_policy": frozenset(
        {
            "enable_throughput_scaling",
            "enable_load_scaling",
            "throughput_adjustment_interval_seconds",
            "load_adjustment_interval_seconds",
        }
    ),
    "planner_fpm_sampling": frozenset(
        {"max_num_fpm_samples", "fpm_sample_bucket_size"}
    ),
    "planner_load_sensitivity": frozenset(
        {"load_scaling_down_sensitivity", "load_min_observations"}
    ),
    "load_predictor_candidates": frozenset(
        {
            "load_predictor",
            "load_predictor_log1p",
            "prophet_window_size",
            "kalman_q_level",
            "kalman_q_trend",
            "kalman_r",
            "kalman_min_points",
        }
    ),
}

# Keys a composite dict MUST provide (any others default downstream). The three
# planner composites are small coupled sets, so a dict must give all of them (the
# doc's "self-contained" contract); a load-predictor dict needs at least the family
# (``load_predictor``) — there is no sensible default for it, and omitting it would
# crash the sub-sweep; the remaining family params default per family.
COMPOSITE_REQUIRED_KEYS: dict[str, frozenset[str]] = {
    "planner_scaling_policy": COMPOSITE_DICT_KEYS["planner_scaling_policy"],
    "planner_fpm_sampling": COMPOSITE_DICT_KEYS["planner_fpm_sampling"],
    "planner_load_sensitivity": COMPOSITE_DICT_KEYS["planner_load_sensitivity"],
    "load_predictor_candidates": frozenset({"load_predictor"}),
}


class SearchSpace(BaseModel):
    """Inputs to one search run, grouped by component.

    Each group lists its swept knobs (list-typed candidate sets; a
    single-element list pins that knob) followed by the pinned knobs that group
    needs (scalars). When ``deployment_mode`` lists both branches the optimizer
    runs one flat study per branch and ranks across both. Most fields drive the
    main Vizier sweep; ``load_predictor_candidates`` is swept by a separate
    forecast-loss grid, with its winner pinned into the main sweep.
    """

    model_config = ConfigDict(extra="forbid")

    # deployment: branch + backend + legal parallel shapes
    deployment_mode: list[str] = ["disagg", "agg"]  # branches to explore; pin with one
    backend: list[str] = ["vllm"]  # vllm | sglang | trtllm
    parallel_configs: list[dict[str, Any]] = Field(
        default_factory=list
    )  # generated when empty
    # pinned
    model_name: str  # HF id or private model name
    hardware_sku: str  # e.g. "h200_sxm"
    gpu_budget: int = 32  # max GPUs per candidate
    min_gpu_budget: int | None = None
    min_endpoint: int | None = None
    context_length: int | None = None
    startup_time: float | None = None
    aic_nextn: int | None = None  # speculative-decode (MTP) depth, 1..5

    # prefill engine (disagg branch): scheduler batching capacity
    prefill_max_num_batched_tokens: list[int] = [8192, 16384, 32768]
    prefill_max_num_seqs: list[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    # pinned
    prefill_block_size: int = 64
    prefill_gpu_memory_utilization: float = 0.9
    prefill_enable_prefix_caching: bool = True

    # decode engine (disagg branch): scheduler batching capacity
    decode_max_num_batched_tokens: list[int] = [8192]
    decode_max_num_seqs: list[int] = [256, 512, 1024]
    # pinned
    decode_block_size: int = 64
    decode_gpu_memory_utilization: float = 0.9
    decode_enable_prefix_caching: bool = False  # forced off for decode workers

    # agg engine (agg branch): scheduler batching capacity
    agg_max_num_batched_tokens: list[int] = [8192, 16384, 32768]
    agg_max_num_seqs: list[int] = [256, 512, 1024]
    # pinned
    agg_block_size: int = 64
    agg_gpu_memory_utilization: float = 0.9
    agg_enable_prefix_caching: bool = True

    # kv manager: multi-tier offload policy (all pinned; G3/G4 extend G2)
    num_g2_blocks: int = Field(default=0, ge=0)  # 0 disables host offload
    kv_bytes_per_token: int | None = Field(
        default=None, gt=0
    )  # required replay transfer sizing when G2 is on
    bandwidth_g1_to_g2_gbps: float | None = None
    bandwidth_g2_to_g1_gbps: float | None = None
    offload_batch_size: int | None = None

    # router (KV-router knobs are ignored under round_robin)
    router_mode: list[str] = ["kv_router", "round_robin"]
    overlap_score_credit: list[float] = [
        0.0,
        0.5,
        1.0,
    ]  # kv-router hard-caps this at 1.0
    prefill_load_scale: list[float] = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    host_cache_hit_weight: list[float] = [0.5, 0.75, 1.0]
    disk_cache_hit_weight: list[float] = [0.0, 0.25, 0.5]
    router_temperature: list[float] = [0.0, 0.2, 0.5, 1.0]
    # pinned (admission control)
    active_decode_blocks_threshold: int | None = None
    active_prefill_tokens_threshold: int | None = None
    active_prefill_tokens_threshold_frac: float | None = None
    no_admission_control: bool = False

    # planner: composite knobs — each entry is a preset id (str) OR a dict pinning
    # the unrolled fields directly (see COMPOSITE_DICT_KEYS and
    # docs/components/profiler/spica/search-space.md).
    # "disabled" = planner not enabled (no autoscaling, static replica count).
    planner_scaling_policy: list[str | dict[str, Any]] = [
        "disabled",
        "throughput_180_5",
        "throughput_600_5",
        "load_180_5",
        "load_180_10",
        "hybrid_180_5",
        "hybrid_600_5",
    ]
    planner_fpm_sampling: list[str | dict[str, Any]] = [
        "small",
        "default",
        "large",
        "fine",
    ]
    planner_load_sensitivity: list[str | dict[str, Any]] = [
        "aggressive",
        "default",
        "conservative",
    ]

    # planner load predictor — independent grid sweep (ranked by one-step-ahead
    # forecast loss, NOT the main Vizier loop); the winning preset is pinned
    # into the main sweep. Only relevant under predictive throughput scaling.
    load_predictor_candidates: list[str | dict[str, Any]] = [
        "constant_last",
        "arima_raw",
        "arima_log1p",
        "prophet_w20_raw",
        "prophet_w20_log1p",
        "prophet_w50_raw",
        "prophet_w50_log1p",
        "kalman_default_raw",
        "kalman_default_log1p",
        "kalman_reactive_raw",
        "kalman_reactive_log1p",
    ]

    @model_validator(mode="after")
    def _validate_search_choices(self) -> SearchSpace:
        """Each swept dimension is a non-empty list whose entries are valid: a string
        must be one of the listed choices; a dict (only on a composite knob) must have
        exactly that composite's unrolled field names (value legality is checked
        downstream by dynamo's PlannerConfig)."""
        for field_name, allowed in SEARCH_CHOICES.items():
            values = getattr(self, field_name)
            if not values:
                raise ValueError(
                    f"{field_name} must list at least one choice; allowed: {list(allowed)}"
                )
            dict_keys = COMPOSITE_DICT_KEYS.get(field_name)
            for v in values:
                if isinstance(v, dict):
                    if dict_keys is None:
                        raise ValueError(
                            f"{field_name} does not accept a dict entry; choices: {list(allowed)}"
                        )
                    if not v:
                        raise ValueError(f"{field_name} dict entry must not be empty")
                    unknown = set(v) - dict_keys
                    if unknown:
                        raise ValueError(
                            f"{field_name} dict has unknown keys {sorted(unknown)}; allowed: {sorted(dict_keys)}"
                        )
                    missing = COMPOSITE_REQUIRED_KEYS.get(
                        field_name, frozenset()
                    ) - set(v)
                    if missing:
                        raise ValueError(
                            f"{field_name} dict is missing required keys {sorted(missing)}; "
                            "a dict entry must be self-contained "
                            "(see docs/components/profiler/spica/search-space.md)"
                        )
                elif v not in allowed:
                    raise ValueError(
                        f"{field_name} has invalid choice {v!r}; allowed: {list(allowed)}"
                    )
        return self

    @model_validator(mode="after")
    def _validate_gpu_budget(self) -> SearchSpace:
        """When ``min_gpu_budget`` is set it must be a positive value not exceeding
        ``gpu_budget``. ``min_endpoint`` is carried into scaling candidates as a planner
        runtime floor; its detailed feasibility is validated by the planner."""
        if self.min_gpu_budget is not None and not (
            0 < self.min_gpu_budget <= self.gpu_budget
        ):
            raise ValueError(
                f"min_gpu_budget must satisfy 0 < min_gpu_budget <= gpu_budget "
                f"(got min_gpu_budget={self.min_gpu_budget}, gpu_budget={self.gpu_budget})"
            )
        return self

    @model_validator(mode="after")
    def _validate_kv_offload_replay_sizing(self) -> SearchSpace:
        """Require deterministic transfer sizing whenever G2 offload is enabled.

        Dynamo can try to infer this value from model metadata, but inference may
        fail for private, gated, or offline models and silently skip attaching the
        offload engine. Requiring the pinned value keeps replay and deployment
        artifacts on the same KVBM policy.
        """
        if self.num_g2_blocks > 0 and self.kv_bytes_per_token is None:
            raise ValueError(
                "kv_bytes_per_token is required when num_g2_blocks > 0 so replay can model KVBM offload"
            )
        return self

    @model_validator(mode="after")
    def _validate_router_admission_replay_support(self) -> SearchSpace:
        """Reject admission-control pins until Dynamo replay can model them.

        Generated deployments support these frontend flags, but the pinned replay
        API accepts only ``KvRouterConfig`` and would silently score a different
        router policy. Failing here keeps optimized and generated artifacts aligned.
        """
        if "kv_router" not in self.router_mode:
            return self

        admission_pins = {
            "active_decode_blocks_threshold": self.active_decode_blocks_threshold,
            "active_prefill_tokens_threshold": self.active_prefill_tokens_threshold,
            "active_prefill_tokens_threshold_frac": self.active_prefill_tokens_threshold_frac,
        }
        enabled = [name for name, value in admission_pins.items() if value is not None]
        if self.no_admission_control:
            enabled.append("no_admission_control")
        if enabled:
            raise ValueError(
                "router admission-control knobs are not supported by the Dynamo replay API; "
                f"remove {', '.join(enabled)} so replay and generated artifacts stay equivalent"
            )
        return self

    @model_validator(mode="after")
    def _validate_parallel_configs(self) -> SearchSpace:
        """A pinned ``parallel_configs`` (non-empty) must match a single deployment
        mode and have the right shape: an agg entry is a flat shape dict (needs
        ``tp``); a disagg entry nests ``prefill`` + ``decode`` shape dicts. Full
        legality (MoE width, KV feasibility, GPU budget) is checked in
        ``enumerate_branches`` against the model+hardware."""
        if not self.parallel_configs:
            return self

        def validate_shape_dict(value: Any, label: str) -> None:
            if not isinstance(value, dict):
                raise ValueError(f"{label} parallel_configs shape must be a dict")
            if "tp" not in value:
                raise ValueError(f"{label} parallel_configs shape needs a 'tp' field")

        if len(self.deployment_mode) != 1:
            raise ValueError(
                "pinning parallel_configs requires deployment_mode to list exactly one mode "
                f"(got {self.deployment_mode}); pin the mode too"
            )
        mode = self.deployment_mode[0]
        for entry in self.parallel_configs:
            if not isinstance(entry, dict):
                raise ValueError("each parallel_configs entry must be a dict")
            if mode == "agg":
                validate_shape_dict(entry, "an agg")
            else:
                if "prefill" not in entry or "decode" not in entry:
                    raise ValueError(
                        "a disagg parallel_configs entry needs 'prefill' and 'decode' sub-dicts"
                    )
                validate_shape_dict(entry["prefill"], "a disagg prefill")
                validate_shape_dict(entry["decode"], "a disagg decode")
        return self


class SweepConfig(BaseModel):
    """Sweep run-control."""

    model_config = ConfigDict(extra="forbid")

    max_rounds: int = Field(default=20, ge=1)  # total Vizier/replay barrier rounds
    parallel_evals: int = Field(
        default=16, ge=1
    )  # replay worker fan-out and default candidates per round
    # Successful unique replay configs per round; duplicate projections are told from
    # cache and replaced. Defaults to parallel_evals.
    candidates_per_round: int | None = Field(default=None, ge=1)
    # Per-candidate wall-clock cap for the replay. A candidate whose replay exceeds this is
    # killed and reported as infeasible ("exceed runtime") so the optimizer avoids that region
    # instead of hanging the sweep (e.g. an over-subscribed config that churns). Only enforced
    # on the worker-pool path (parallel_evals > 1); None disables the cap.
    max_eval_seconds: float | None = Field(default=600.0, gt=0)


class Candidate(BaseModel):
    """One evaluated configuration and its replay performance."""

    model_config = ConfigDict(extra="forbid")

    config: dict[str, Any]  # the decoded knob assignment (engine/router/planner)
    used_gpus: int
    score: float  # objective score, normalized so higher is better (pareto: the first objective's value)
    metrics: dict[str, float]  # replay performance: throughput, ttft, itl, e2e, goodput
    # Per-objective raw values (natural units/direction) under a pareto goal, keyed by
    # OptimizationTarget value (e.g. {"throughput_per_gpu": .., "throughput_per_user": ..});
    # None for a single-objective sweep. Drives Pareto dominance in score.pareto_front.
    objectives: dict[str, float] | None = None


class SmartSearchConfig(BaseModel):
    """Top-level config integrating every search input; one YAML maps to this."""

    model_config = ConfigDict(extra="forbid")

    search_space: SearchSpace
    workload: Workload
    goal: OptimizationGoal = Field(default_factory=OptimizationGoal)
    sweep: SweepConfig = Field(default_factory=SweepConfig)

    @model_validator(mode="before")
    @classmethod
    def _default_pareto_kv_load_ratio(cls, data: Any) -> Any:
        """A synthetic Pareto workload with no explicit load searches KV load in [0, 1]."""
        if not isinstance(data, dict):
            return data
        goal = data.get("goal")
        if isinstance(goal, OptimizationGoal):
            is_pareto = goal.is_pareto
        elif isinstance(goal, dict):
            is_pareto = goal.get("target", OptimizationTarget.THROUGHPUT) in {
                OptimizationTarget.PARETO,
                OptimizationTarget.PARETO.value,
            }
        else:
            is_pareto = False
        workload = data.get("workload")
        if (
            not is_pareto
            or not isinstance(workload, dict)
            or workload.get("trace_path") is not None
        ):
            return data
        if any(
            workload.get(name) is not None
            for name in ("request_rate", "concurrency", "kv_load_ratio")
        ):
            return data
        updated = dict(data)
        updated_workload = dict(workload)
        updated_workload["kv_load_ratio"] = [0.0, 1.0]
        updated["workload"] = updated_workload
        return updated

    @model_validator(mode="after")
    def _validate_kv_load_ratio_range(self) -> SmartSearchConfig:
        """Only a Pareto study may search a KV-load range; scalar ratios work for any goal."""
        if self.workload.kv_load_ratio_range is not None and not self.goal.is_pareto:
            raise ValueError(
                "a ranged workload.kv_load_ratio is only allowed when goal.target is 'pareto' "
                f"(got target={self.goal.target.value}); use one scalar kv_load_ratio"
            )
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> SmartSearchConfig:
        """Load + validate one YAML file into the nested config."""
        data = yaml.safe_load(Path(path).read_text())
        return cls.model_validate(data)
