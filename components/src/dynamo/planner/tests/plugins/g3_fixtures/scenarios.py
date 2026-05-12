# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Representative scenarios for G3 behavior parity matrix.

Each scenario defines:
- ``name``: filename slug (e.g. ``baseline_disagg_throughput_only_sla``)
- ``config_overrides``: dict for ``_make_config(**overrides)``
- ``caps``: ``WorkerCapabilities`` (default vs agg)
- ``bootstrap_fn``: optional ``Callable[[PSM], None]`` to call
  ``load_benchmark_fpms`` / ``warm_load_predictors`` before tick loop
- ``ticks``: list of ``TickInput``—fed to ``PSM.on_tick`` in order

Scenarios are intentionally **deterministic**:
- All times explicit (``now_s`` per tick); no wall-clock dependency.
- All FPM payloads constructed via ``_make_fpm`` with literal numbers.
- No randomness.

Adding a new scenario:
1. Define ``Scenario(...)`` instance in ``ALL_SCENARIOS``.
2. Re-run dump tool to regenerate fixture file.
3. The G3 matrix is documented in the design (mode × scaling toggle ×
   optimization_target = 36 combinations); the v1 fixture set covers
   representative cases—not every cell exhaustively (cells that are
   redundant or invalid are skipped, e.g. ``prefill`` mode +
   ``enable_load=False, enable_throughput=False`` is a no-op).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from dynamo.common.forward_pass_metrics import (
    ForwardPassMetrics,
    QueuedRequestMetrics,
    ScheduledRequestMetrics,
)
from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.state_machine import PlannerStateMachine
from dynamo.planner.core.types import (
    EngineCapabilities,
    FpmObservations,
    TickInput,
    TrafficObservation,
    WorkerCapabilities,
    WorkerCounts,
)


# ---------------------------------------------------------------------------
# Helpers (mirror tests/unit/test_state_machine.py to maximize parity)
# ---------------------------------------------------------------------------


def _make_fpm(
    *,
    sum_prefill_tokens: int = 0,
    num_prefill_requests: int = 0,
    sum_decode_kv_tokens: int = 0,
    num_decode_requests: int = 0,
    queued_prefill_tokens: int = 0,
    queued_decode_kv_tokens: int = 0,
    wall_time: float = 0.01,
    worker_id: str = "w1",
    dp_rank: int = 0,
) -> ForwardPassMetrics:
    return ForwardPassMetrics(
        worker_id=worker_id,
        dp_rank=dp_rank,
        wall_time=wall_time,
        scheduled_requests=ScheduledRequestMetrics(
            sum_prefill_tokens=sum_prefill_tokens,
            num_prefill_requests=num_prefill_requests,
            sum_decode_kv_tokens=sum_decode_kv_tokens,
            num_decode_requests=num_decode_requests,
        ),
        queued_requests=QueuedRequestMetrics(
            sum_prefill_tokens=queued_prefill_tokens,
            sum_decode_kv_tokens=queued_decode_kv_tokens,
        ),
    )


_DEFAULT_CONFIG = dict(
    mode="disagg",
    optimization_target="sla",
    ttft=500.0,
    itl=50.0,
    min_endpoint=1,
    max_gpu_budget=-1,
    throughput_adjustment_interval=60,
    load_adjustment_interval=5,
    load_scaling_down_sensitivity=80,
    max_num_fpm_samples=50,
    fpm_sample_bucket_size=16,
    load_min_observations=5,
    enable_load_scaling=True,
    enable_throughput_scaling=True,
    load_predictor="constant",
    backend="vllm",
    metric_pulling_prometheus_endpoint="http://localhost:9090",
    metric_reporting_prometheus_port=0,
)


def _make_config(**overrides: Any) -> PlannerConfig:
    cfg = dict(_DEFAULT_CONFIG)
    cfg.update(overrides)
    return PlannerConfig.model_construct(**cfg)


def _default_caps() -> WorkerCapabilities:
    return WorkerCapabilities(
        prefill=EngineCapabilities(num_gpu=1, max_num_batched_tokens=2048),
        decode=EngineCapabilities(num_gpu=1, max_num_batched_tokens=2048),
    )


def _agg_caps() -> WorkerCapabilities:
    return WorkerCapabilities(
        decode=EngineCapabilities(num_gpu=1, max_num_batched_tokens=2048),
    )


def _train_prefill_regression(core: PlannerStateMachine) -> None:
    fpms = [
        _make_fpm(
            sum_prefill_tokens=t, num_prefill_requests=1, wall_time=0.001 * t + 0.002
        )
        for t in [500, 1000, 1500, 2000, 2500]
    ]
    core.load_benchmark_fpms(prefill_fpms=fpms)


def _train_decode_regression(core: PlannerStateMachine) -> None:
    fpms = [
        _make_fpm(
            sum_decode_kv_tokens=kv,
            num_decode_requests=n,
            wall_time=0.00001 * kv + 0.001,
        )
        for n, kv in [(5, 5000), (10, 10000), (20, 20000), (30, 30000), (40, 40000)]
    ]
    core.load_benchmark_fpms(decode_fpms=fpms)


def _train_agg_regression(core: PlannerStateMachine) -> None:
    fpms = [
        _make_fpm(
            sum_decode_kv_tokens=kv,
            num_decode_requests=n,
            wall_time=0.00001 * kv + 0.001,
        )
        for n, kv in [(5, 5000), (10, 10000), (20, 20000), (30, 30000), (40, 40000)]
    ]
    core.load_benchmark_fpms(agg_fpms=fpms)


# ---------------------------------------------------------------------------
# Scenario data class
# ---------------------------------------------------------------------------


@dataclass
class Scenario:
    """A single G3 behavior-parity scenario."""

    name: str
    description: str
    config_overrides: dict[str, Any] = field(default_factory=dict)
    caps_factory: Callable[[], WorkerCapabilities] = _default_caps
    bootstrap_fn: Optional[Callable[[PlannerStateMachine], None]] = None
    initial_tick_at_s: float = 0.0
    ticks: list[TickInput] = field(default_factory=list)

    def make_config(self) -> PlannerConfig:
        return _make_config(**self.config_overrides)


# ---------------------------------------------------------------------------
# Scenario library
# ---------------------------------------------------------------------------


def _disagg_throughput_only_sla() -> Scenario:
    """Baseline disagg + throughput-only + sla.

    Default Dynamo configuration. Verifies throughput scaling decision
    pipeline (predict load -> compute desired -> apply budget -> scale).
    """
    ticks = [
        # tick 1 (t=5s): load tick (no FPM yet) - schedule advances
        TickInput(
            now_s=5.0,
            fpm_observations=FpmObservations(
                prefill={("w1", 0): _make_fpm()},
                decode={("w1", 0): _make_fpm()},
            ),
            worker_counts=WorkerCounts(
                ready_num_prefill=1, ready_num_decode=1
            ),
        ),
        # tick 2 (t=60s): throughput tick (with traffic)
        TickInput(
            now_s=60.0,
            traffic=TrafficObservation(
                duration_s=60, num_req=120, isl=1000, osl=200
            ),
            worker_counts=WorkerCounts(
                ready_num_prefill=1, ready_num_decode=1
            ),
        ),
        # tick 3 (t=120s): another throughput tick with higher load
        TickInput(
            now_s=120.0,
            traffic=TrafficObservation(
                duration_s=60, num_req=300, isl=1500, osl=300
            ),
            worker_counts=WorkerCounts(
                ready_num_prefill=2, ready_num_decode=3
            ),
        ),
    ]
    return Scenario(
        name="baseline_disagg_throughput_only_sla",
        description=(
            "Default Dynamo config: disagg mode, throughput-only scaling, "
            "sla optimization_target. Three ticks: bootstrap -> first throughput "
            "tick -> higher-load throughput tick."
        ),
        config_overrides=dict(
            mode="disagg",
            enable_load_scaling=False,
            enable_throughput_scaling=True,
            optimization_target="sla",
        ),
        bootstrap_fn=lambda core: (
            _train_prefill_regression(core),
            _train_decode_regression(core),
        ),
        initial_tick_at_s=0.0,
        ticks=ticks,
    )


def _disagg_load_throughput_sla() -> Scenario:
    """disagg + load + throughput + sla.

    Verifies load + throughput coexistence (throughput sets lower bound,
    load reads it). This is the core "load decision wins, throughput is
    fallback" behavior to preserve in G3.
    """
    ticks = [
        # tick 1 (t=5s): load tick with FPM
        TickInput(
            now_s=5.0,
            fpm_observations=FpmObservations(
                prefill={("w1", 0): _make_fpm(
                    sum_prefill_tokens=1500, num_prefill_requests=2, wall_time=0.6
                )},
                decode={("w1", 0): _make_fpm(
                    sum_decode_kv_tokens=8000, num_decode_requests=8, wall_time=0.05
                )},
            ),
            worker_counts=WorkerCounts(
                ready_num_prefill=1, ready_num_decode=1
            ),
        ),
        # tick 2 (t=60s): throughput tick
        TickInput(
            now_s=60.0,
            traffic=TrafficObservation(
                duration_s=60, num_req=120, isl=1000, osl=200
            ),
            worker_counts=WorkerCounts(
                ready_num_prefill=1, ready_num_decode=1
            ),
        ),
        # tick 3 (t=65s): another load tick
        TickInput(
            now_s=65.0,
            fpm_observations=FpmObservations(
                prefill={("w1", 0): _make_fpm(
                    sum_prefill_tokens=2000, num_prefill_requests=3, wall_time=0.8
                )},
                decode={("w1", 0): _make_fpm(
                    sum_decode_kv_tokens=15000, num_decode_requests=15, wall_time=0.10
                )},
            ),
            worker_counts=WorkerCounts(
                ready_num_prefill=2, ready_num_decode=2
            ),
        ),
    ]
    return Scenario(
        name="disagg_load_throughput_sla",
        description=(
            "disagg + load + throughput coexistence. Verifies "
            "throughput_lower_bound from throughput tick correctly affects "
            "load decision in next tick (key G3 invariant)."
        ),
        config_overrides=dict(
            mode="disagg",
            enable_load_scaling=True,
            enable_throughput_scaling=True,
            optimization_target="sla",
        ),
        bootstrap_fn=lambda core: (
            _train_prefill_regression(core),
            _train_decode_regression(core),
        ),
        initial_tick_at_s=0.0,
        ticks=ticks,
    )


def _disagg_load_only_latency_easy() -> Scenario:
    """disagg + load-only + latency (easy mode).

    Easy mode: optimization_target != "sla" -> static thresholds, no
    regression / predictor. Easy mode is **load-only by design**—throughput
    scaling needs SLA mode for regression-based predictions; PSM crashes
    if easy mode + enable_throughput_scaling=True (no predictor).
    """
    ticks = [
        # tick 1 (t=5s): load tick — easy mode uses static FPM thresholds
        # for scale-up decision (queued_prefill_tokens > context_length / 10
        # => latency scale-up trigger).
        TickInput(
            now_s=5.0,
            fpm_observations=FpmObservations(
                prefill={("w1", 0): _make_fpm(queued_prefill_tokens=400)},
                decode={("w1", 0): _make_fpm(
                    sum_decode_kv_tokens=1500, queued_decode_kv_tokens=300
                )},
            ),
            worker_counts=WorkerCounts(
                ready_num_prefill=1, ready_num_decode=1
            ),
        ),
        # tick 2 (t=10s): another load tick — verify cadence advances
        # (load_adjustment_interval=5s default, so next load tick at t=10).
        TickInput(
            now_s=10.0,
            fpm_observations=FpmObservations(
                prefill={("w1", 0): _make_fpm(queued_prefill_tokens=100)},
                decode={("w1", 0): _make_fpm(
                    sum_decode_kv_tokens=2000, queued_decode_kv_tokens=500
                )},
            ),
            worker_counts=WorkerCounts(
                ready_num_prefill=2, ready_num_decode=1
            ),
        ),
    ]
    return Scenario(
        name="disagg_load_only_latency_easy",
        description=(
            "Easy mode (optimization_target='latency'): no regression, "
            "uses static FPM thresholds. Load-only enabled (easy mode + "
            "throughput-scaling crashes—predictor not instantiated)."
        ),
        config_overrides=dict(
            mode="disagg",
            enable_load_scaling=True,
            enable_throughput_scaling=False,
            optimization_target="latency",
        ),
        bootstrap_fn=None,  # easy mode: no benchmark FPM bootstrap
        initial_tick_at_s=0.0,
        ticks=ticks,
    )


def _agg_throughput_only_sla() -> Scenario:
    """agg + throughput-only + sla.

    Single-engine aggregated mode (no separate prefill/decode). Verifies
    throughput scaling decision in agg path.
    """
    ticks = [
        TickInput(
            now_s=5.0,
            fpm_observations=FpmObservations(
                decode={("w1", 0): _make_fpm()},
            ),
            worker_counts=WorkerCounts(ready_num_decode=1),
        ),
        TickInput(
            now_s=60.0,
            traffic=TrafficObservation(
                duration_s=60, num_req=100, isl=1200, osl=200
            ),
            worker_counts=WorkerCounts(ready_num_decode=1),
        ),
    ]
    return Scenario(
        name="agg_throughput_only_sla",
        description=(
            "agg mode (single engine type): throughput-only + sla. "
            "Verifies throughput_agg path."
        ),
        config_overrides=dict(
            mode="agg",
            enable_load_scaling=False,
            enable_throughput_scaling=True,
            optimization_target="sla",
        ),
        caps_factory=_agg_caps,
        bootstrap_fn=lambda core: _train_agg_regression(core),
        initial_tick_at_s=0.0,
        ticks=ticks,
    )


def _prefill_throughput_only_sla() -> Scenario:
    """prefill (single-component) mode + throughput-only + sla.

    Verifies prefill-only mode: only prefill regression instantiated,
    only prefill scaling decisions emitted.
    """
    ticks = [
        TickInput(
            now_s=5.0,
            fpm_observations=FpmObservations(
                prefill={("w1", 0): _make_fpm()},
            ),
            worker_counts=WorkerCounts(ready_num_prefill=1),
        ),
        TickInput(
            now_s=60.0,
            traffic=TrafficObservation(
                duration_s=60, num_req=120, isl=1000, osl=100
            ),
            worker_counts=WorkerCounts(ready_num_prefill=1),
        ),
    ]
    return Scenario(
        name="prefill_throughput_only_sla",
        description="prefill-only mode + throughput-only + sla.",
        config_overrides=dict(
            mode="prefill",
            enable_load_scaling=False,
            enable_throughput_scaling=True,
            optimization_target="sla",
        ),
        bootstrap_fn=lambda core: _train_prefill_regression(core),
        initial_tick_at_s=0.0,
        ticks=ticks,
    )


def _decode_throughput_only_sla() -> Scenario:
    """decode (single-component) mode + throughput-only + sla."""
    ticks = [
        TickInput(
            now_s=5.0,
            fpm_observations=FpmObservations(
                decode={("w1", 0): _make_fpm()},
            ),
            worker_counts=WorkerCounts(ready_num_decode=1),
        ),
        TickInput(
            now_s=60.0,
            traffic=TrafficObservation(
                duration_s=60, num_req=100, isl=800, osl=200
            ),
            worker_counts=WorkerCounts(ready_num_decode=1),
        ),
    ]
    return Scenario(
        name="decode_throughput_only_sla",
        description="decode-only mode + throughput-only + sla.",
        config_overrides=dict(
            mode="decode",
            enable_load_scaling=False,
            enable_throughput_scaling=True,
            optimization_target="sla",
        ),
        bootstrap_fn=lambda core: _train_decode_regression(core),
        initial_tick_at_s=0.0,
        ticks=ticks,
    )


# ---------------------------------------------------------------------------
# Coverage gap scenarios
# ---------------------------------------------------------------------------


def _agg_load_throughput_sla() -> Scenario:
    """agg + load + throughput + sla.

    Agg variant of ``disagg_load_throughput_sla`` — exercises the
    throughput-lower-bound side-channel for the single-decode-engine
    agg shape. Key target for the plugin-decomposition because it
    hits ``_advance_load_agg`` (which has a different decision
    structure than ``_advance_load_single`` / ``_advance_load_disagg``).
    """
    ticks = [
        TickInput(
            now_s=5.0,
            fpm_observations=FpmObservations(
                decode={("w1", 0): _make_fpm(
                    sum_decode_kv_tokens=6000, queued_decode_kv_tokens=2000,
                    num_decode_requests=6, wall_time=0.05,
                )},
            ),
            worker_counts=WorkerCounts(ready_num_decode=1),
        ),
        TickInput(
            now_s=60.0,
            traffic=TrafficObservation(
                duration_s=60, num_req=100, isl=1000, osl=200
            ),
            worker_counts=WorkerCounts(ready_num_decode=1),
        ),
        TickInput(
            now_s=65.0,
            fpm_observations=FpmObservations(
                decode={("w1", 0): _make_fpm(
                    sum_decode_kv_tokens=12000, queued_decode_kv_tokens=4000,
                    num_decode_requests=12, wall_time=0.08,
                )},
            ),
            worker_counts=WorkerCounts(ready_num_decode=2),
        ),
    ]
    return Scenario(
        name="agg_load_throughput_sla",
        description=(
            "agg mode + load + throughput + sla. Exercises agg "
            "_advance_load_agg under coexistence (throughput writes "
            "lower bound, load reads it)."
        ),
        config_overrides=dict(
            mode="agg",
            enable_load_scaling=True,
            enable_throughput_scaling=True,
            optimization_target="sla",
        ),
        bootstrap_fn=lambda core: _train_agg_regression(core),
        initial_tick_at_s=0.0,
        ticks=ticks,
    )


def _prefill_load_only_sla() -> Scenario:
    """prefill (single-component) + load-only + sla.

    Exercises ``_advance_load_single`` with component='prefill' —
    previously only single-component mode with throughput-only was
    covered; this adds the SLA load path.
    """
    ticks = [
        TickInput(
            now_s=5.0,
            fpm_observations=FpmObservations(
                prefill={("w1", 0): _make_fpm(
                    sum_prefill_tokens=1500, num_prefill_requests=2, wall_time=0.6,
                )},
            ),
            worker_counts=WorkerCounts(ready_num_prefill=1),
        ),
        TickInput(
            now_s=10.0,
            fpm_observations=FpmObservations(
                prefill={("w1", 0): _make_fpm(
                    sum_prefill_tokens=3000, num_prefill_requests=4, wall_time=1.2,
                )},
            ),
            worker_counts=WorkerCounts(ready_num_prefill=1),
        ),
    ]
    return Scenario(
        name="prefill_load_only_sla",
        description=(
            "prefill-only mode + load-only + sla — exercises "
            "_advance_load_single with prefill component + SLA path "
            "(insufficient_data on early ticks, scale_up later)."
        ),
        config_overrides=dict(
            mode="prefill",
            enable_load_scaling=True,
            enable_throughput_scaling=False,
            optimization_target="sla",
        ),
        bootstrap_fn=lambda core: _train_prefill_regression(core),
        initial_tick_at_s=0.0,
        ticks=ticks,
    )


def _decode_load_only_sla() -> Scenario:
    """decode (single-component) + load-only + sla."""
    ticks = [
        TickInput(
            now_s=5.0,
            fpm_observations=FpmObservations(
                decode={("w1", 0): _make_fpm(
                    sum_decode_kv_tokens=8000, queued_decode_kv_tokens=1500,
                    num_decode_requests=8, wall_time=0.05,
                )},
            ),
            worker_counts=WorkerCounts(ready_num_decode=1),
        ),
        TickInput(
            now_s=10.0,
            fpm_observations=FpmObservations(
                decode={("w1", 0): _make_fpm(
                    sum_decode_kv_tokens=15000, queued_decode_kv_tokens=3000,
                    num_decode_requests=15, wall_time=0.10,
                )},
            ),
            worker_counts=WorkerCounts(ready_num_decode=1),
        ),
    ]
    return Scenario(
        name="decode_load_only_sla",
        description=(
            "decode-only mode + load-only + sla — exercises "
            "_advance_load_single with decode component + SLA path."
        ),
        config_overrides=dict(
            mode="decode",
            enable_load_scaling=True,
            enable_throughput_scaling=False,
            optimization_target="sla",
        ),
        bootstrap_fn=lambda core: _train_decode_regression(core),
        initial_tick_at_s=0.0,
        ticks=ticks,
    )


def _disagg_load_only_throughput_easy() -> Scenario:
    """disagg + load-only + throughput (easy mode).

    Easy mode variant with optimization_target='throughput' — exercises
    the static-threshold scale-up/down at different points on the
    _PREFILL_THROUGHPUT_* / _DECODE_THROUGHPUT_* thresholds than the
    existing 'latency' easy scenario.
    """
    ticks = [
        TickInput(
            now_s=5.0,
            fpm_observations=FpmObservations(
                prefill={("w1", 0): _make_fpm(queued_prefill_tokens=300)},
                decode={("w1", 0): _make_fpm(
                    sum_decode_kv_tokens=1000, queued_decode_kv_tokens=200
                )},
            ),
            worker_counts=WorkerCounts(
                ready_num_prefill=1, ready_num_decode=1
            ),
        ),
        TickInput(
            now_s=10.0,
            fpm_observations=FpmObservations(
                prefill={("w1", 0): _make_fpm(queued_prefill_tokens=5000)},
                decode={("w1", 0): _make_fpm(
                    sum_decode_kv_tokens=13000, queued_decode_kv_tokens=7000
                )},
            ),
            worker_counts=WorkerCounts(
                ready_num_prefill=1, ready_num_decode=1
            ),
        ),
    ]
    return Scenario(
        name="disagg_load_only_throughput_easy",
        description=(
            "disagg + load-only + throughput (easy mode). Static "
            "thresholds for throughput optimization_target."
        ),
        config_overrides=dict(
            mode="disagg",
            enable_load_scaling=True,
            enable_throughput_scaling=False,
            optimization_target="throughput",
        ),
        bootstrap_fn=None,  # easy mode: no benchmark FPM bootstrap
        initial_tick_at_s=0.0,
        ticks=ticks,
    )


# ---------------------------------------------------------------------------
# Public scenario list
# ---------------------------------------------------------------------------


ALL_SCENARIOS: list[Scenario] = [
    _disagg_throughput_only_sla(),
    _disagg_load_throughput_sla(),
    _disagg_load_only_latency_easy(),
    _agg_throughput_only_sla(),
    _prefill_throughput_only_sla(),
    _decode_throughput_only_sla(),
    # Coverage gaps
    _agg_load_throughput_sla(),
    _prefill_load_only_sla(),
    _decode_load_only_sla(),
    _disagg_load_only_throughput_easy(),
]


def find_scenario(name: str) -> Optional[Scenario]:
    for s in ALL_SCENARIOS:
        if s.name == name:
            return s
    return None
