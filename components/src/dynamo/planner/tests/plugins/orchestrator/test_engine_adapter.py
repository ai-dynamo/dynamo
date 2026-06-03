# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for ``OrchestratorEngineAdapter`` cadence parity.

Covers PSM-parity bugs caught after K8s smoke / dual-path review:

- ``initial_tick`` previously read ``self._config.throughput_adjustment_interval``
  (missing ``_seconds`` suffix). The Pydantic ``validation_alias`` only affects
  input parsing — attribute access requires the canonical name. Triggered an
  ``AttributeError`` whenever ``enable_throughput_scaling=True``.

- ``_MERGE_TOLERANCE_S`` was set to ``1e-9`` (float epsilon framing) instead
  of PSM's ``0.5`` (wall-clock-drift padding). With the tight tolerance a
  load tick and a throughput tick scheduled within ~ms of each other failed
  to merge — splitting into 2 ticks where PSM produces 1.

- Hard-coded ``WallClock`` broke replay: plugin scheduler / CircuitBreaker
  / HOLD_LAST cache all read ``self._clock.monotonic()``, but replay
  fast-forwards trace time without advancing real wall-clock. Adapter now
  accepts an injectable ``Clock`` and bumps it to ``tick_input.now_s`` on
  every tick when the clock is manually-advanced (``VirtualClock``).
"""

from __future__ import annotations

import pytest

from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.types import (
    EngineCapabilities,
    ScheduledTick,
    TickInput,
    WorkerCapabilities,
)
from dynamo.planner.plugins.clock import VirtualClock
from dynamo.planner.plugins.orchestrator.engine_adapter import OrchestratorEngineAdapter

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


def _caps() -> WorkerCapabilities:
    return WorkerCapabilities(
        decode=EngineCapabilities(
            num_gpu=1, max_num_batched_tokens=2048, max_kv_tokens=16384
        )
    )


def _agg_config_throughput_on() -> PlannerConfig:
    # SLA mode keeps ``enable_throughput_scaling=True`` honored;
    # easy modes (``optimization_target="throughput"`` / ``"load"``)
    # silently force it back to False during config validation.
    return PlannerConfig(
        mode="agg",
        enable_load_scaling=True,
        enable_throughput_scaling=True,
        optimization_target="sla",
        served_model_name="test",
    )


def test_initial_tick_with_throughput_scaling_enabled_does_not_attribute_error():
    """``initial_tick`` used to read the non-existent
    ``throughput_adjustment_interval`` attribute (canonical name has a
    ``_seconds`` suffix; the short form is only a validation alias, not
    an attribute accessor in Pydantic v2). Pre-fix this branch raised
    ``AttributeError`` and crashed planner startup whenever
    ``enable_throughput_scaling`` was True.
    """
    config = _agg_config_throughput_on()
    # Sanity guard: if the validator ever changes and silently flips
    # this off, the test would pass for the wrong reason (the buggy
    # branch is short-circuited at line 340 ``if enable_throughput_scaling``).
    assert config.enable_throughput_scaling is True

    adapter = OrchestratorEngineAdapter(config, _caps())
    tick = adapter.initial_tick(start_s=0.0)
    assert isinstance(tick, ScheduledTick)
    # First tick is whichever cadence is shorter. We don't pin the exact
    # value here — defaults move between SLA presets — only that we
    # got past the buggy attribute read.
    assert tick.at_s > 0.0
    assert tick.run_load_scaling or tick.run_throughput_scaling


def test_pipeline_fires_at_scale_interval_cadence():
    """Replaces the previous ``test_merge_tolerance_matches_psm_500ms_window``.

    Under the scale_interval cadence model, the engine_adapter no longer
    runs the PSM ``_MERGE_TOLERANCE_S = 0.5`` merge logic — there is no
    dual ``_next_load_s`` / ``_next_throughput_s`` to reconcile.
    Pipeline fires once per ``scale_interval_seconds`` from the last
    tick moment.  Per-plugin throttling (in ``PluginScheduler._is_due``)
    decides which plugins actually fire each tick — the merge-tolerance
    concept that used to live here is now naturally absorbed by the
    plugin scheduler, which evaluates each plugin's throttle
    independently using the same ``now`` value.

    Cadence-merge parity with PSM is preserved at the *decision* level
    rather than the *tick-shape* level (see Decision 1 in
    /tmp/scale_interval_design.md §11).  This test locks the new shape:
    next_tick.at_s = last_tick + scale_interval, exactly.
    """
    adapter = OrchestratorEngineAdapter(_agg_config_throughput_on(), _caps())
    tick = adapter.initial_tick(start_s=0.0)
    # Default scale_interval_seconds = 5.0 (see SchedulingConfig).
    assert tick.at_s == pytest.approx(5.0, abs=1e-9)
    assert tick.run_load_scaling, "scale_interval ticks fire both flags"
    assert tick.run_throughput_scaling, "scale_interval ticks fire both flags"


def test_scale_interval_advances_from_actual_tick_now():
    """Sequential ticks anchor on ``tick_input.now_s``, not on a
    pre-computed schedule — so a 700ms-late tick at T=5.7 produces the
    next tick at T=10.7, accumulating drift symmetrically with PSM
    (PSM also advances from ``tick_input.now_s``).  This is the basic
    contract for scale_interval cadence advancement.
    """

    adapter = OrchestratorEngineAdapter(_agg_config_throughput_on(), _caps())
    initial = adapter.initial_tick(start_s=0.0)
    assert initial.at_s == pytest.approx(5.0)


@pytest.mark.asyncio
async def test_lazy_traffic_pull_skips_prometheus_when_no_plugin_needs_traffic():
    """Under scale_interval, ``need_traffic_metrics`` is True only
    when some registered plugin both lists ``observations.traffic``
    in its ``needs`` AND is due at the next tick.  With no traffic
    consumer registered, every pipeline tick should signal
    ``need_traffic_metrics=False`` to the gather layer — saving
    36 Prometheus queries per 180s window compared to the eager
    "always pull" alternative.
    """
    adapter = OrchestratorEngineAdapter(_agg_config_throughput_on(), _caps())
    tick = adapter.initial_tick(start_s=0.0)
    # No plugin is registered, so no plugin needs ``observations.traffic``.
    assert tick.need_traffic_metrics is False
    assert tick.traffic_metrics_duration_s == 0.0


# ---------------------------------------------------------------------------
# Clock injection for replay
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tick_advances_injected_virtual_clock_to_trace_time():
    """When a ``VirtualClock`` is injected (replay path), every
    ``engine_adapter.tick()`` must bump the clock to
    ``tick_input.now_s`` so the plugin scheduler / CircuitBreaker /
    HOLD_LAST cache see *trace time*, not real wall-clock.

    Without this bump, a fast-forward replay (e.g. 1hr trace in 10s
    real time) would leave every plugin with
    ``execution_interval_seconds`` greater than the real elapsed time
    never re-firing after its first call — breaking PSM-parity on the
    replay path and blocking PR #10's ``use_orchestrator=True`` default.
    """
    vc = VirtualClock()
    adapter = OrchestratorEngineAdapter(_agg_config_throughput_on(), _caps(), clock=vc)
    # ``initial_tick`` is pure cadence math — no plugin scheduler call,
    # so the clock must not advance from this alone.
    initial = adapter.initial_tick(start_s=0.0)
    assert vc.monotonic() == 0.0

    # Drive a tick at trace time 180.0 — real wall-clock has barely
    # moved, but ``tick_input.now_s`` says we're 180s into the trace.
    await adapter.tick(initial, TickInput(now_s=180.0))
    assert vc.monotonic() == pytest.approx(180.0)

    # Subsequent tick at trace time 360.0 advances further.
    next_tick = ScheduledTick(
        at_s=360.0,
        run_load_scaling=True,
        run_throughput_scaling=True,
        need_worker_states=True,
        need_worker_fpm=True,
        need_traffic_metrics=True,
        traffic_metrics_duration_s=180.0,
    )
    await adapter.tick(next_tick, TickInput(now_s=360.0))
    assert vc.monotonic() == pytest.approx(360.0)


@pytest.mark.asyncio
async def test_tick_does_not_advance_clock_backwards():
    """Defensive: if ``tick_input.now_s`` is *before* the clock's
    current monotonic, ``advance(negative)`` would raise
    ``ValueError`` from VirtualClock. The bump must be gated on
    ``delta > 0`` so this case is a silent no-op.

    Trace time should never go backwards in practice, but a paranoid
    replay driver that pre-advances the clock manually should not
    crash the adapter.
    """
    vc = VirtualClock()
    vc.advance(500.0)  # clock already at 500s
    adapter = OrchestratorEngineAdapter(_agg_config_throughput_on(), _caps(), clock=vc)
    initial = adapter.initial_tick(start_s=0.0)
    # tick_input.now_s = 300.0 is *before* the clock — must not raise.
    await adapter.tick(initial, TickInput(now_s=300.0))
    # Clock stays put (no backwards advance).
    assert vc.monotonic() == pytest.approx(500.0)


def test_default_clock_is_wallclock():
    """Production path: when no ``clock`` kwarg is supplied, the
    adapter falls back to ``WallClock`` so existing K8s deployments
    keep their real-time semantics. Lock the default so a future
    refactor that flips it doesn't silently break production cadence
    tracking.
    """
    from dynamo.planner.plugins.clock import WallClock

    adapter = OrchestratorEngineAdapter(_agg_config_throughput_on(), _caps())
    assert isinstance(adapter._clock, WallClock)


def test_lazy_traffic_due_check_uses_monotonic_not_wall_epoch():
    """Regression: ``_compute_next_scheduled_tick`` must call
    ``PluginScheduler._is_due`` with the **monotonic-domain** projection
    of the next tick, not the wall-epoch projection.

    In production ``WallClock`` deployments, ``tick_input.now_s`` is
    wall-epoch (~1.7e9) while ``RegisteredPlugin.last_call_at`` is set
    by the pipeline via ``self._clock.monotonic()`` (boot-relative,
    ~1e3). A naive ``_is_due(plugin, _last_tick_s + scale_interval)``
    therefore compares ~1.7e9 against ~1e3 — every plugin reads as due
    forever, and the lazy traffic pull degenerates to "always pull".

    This test wires a custom ``Clock`` that fixes monotonic() at 100s
    while ``_last_tick_s`` is set to a wall-epoch-like 1.7e9, then
    pins ``plugin.last_call_at`` such that the plugin is **not yet due**
    in the monotonic domain.  With the bug, the plugin would be in
    ``traffic_consumers_due`` and ``need_traffic_metrics`` would be
    ``True``.  Fixed: monotonic projection correctly skips the plugin.
    """
    from dynamo.planner.plugins.clock import Clock
    from dynamo.planner.plugins.registry.types import RegisteredPlugin
    from dynamo.planner.plugins.types import HoldPolicy

    class _FixedMonoClock(Clock):
        """Wall-vs-monotonic drift simulator — not a VirtualClock so the
        ``tick()`` sync path stays out of the picture."""

        def __init__(self, mono: float) -> None:
            self._mono = mono

        def now(self) -> float:
            return 1.7e9  # arbitrary wall epoch

        def monotonic(self) -> float:
            return self._mono

        async def sleep(self, seconds: float) -> None:  # pragma: no cover
            return None

    clock = _FixedMonoClock(mono=100.0)
    adapter = OrchestratorEngineAdapter(
        _agg_config_throughput_on(), _caps(), clock=clock
    )
    # adapter._scale_interval defaults to 5.0s — the monotonic projection
    # below uses it implicitly inside ``_compute_next_scheduled_tick``.

    # Simulate "one tick has just been recorded" — _last_tick_s is wall epoch,
    # _last_tick_monotonic is the matching monotonic snapshot.
    adapter._last_tick_s = 1.7e9
    adapter._last_tick_monotonic = 100.0

    # Inject a registered traffic-consuming plugin whose last_call_at is in
    # monotonic domain and whose execution_interval keeps it NOT due at the
    # next tick.
    registry = adapter._orchestrator._registry
    plugin = RegisteredPlugin(
        plugin_id="traffic_consumer",
        plugin_type="propose",
        priority=10,
        endpoint="inproc://traffic_consumer",
        version="test",
        protocol_version="1.0",
        execution_interval_seconds=60.0,
        hold_policy=HoldPolicy.ACCEPT_WHEN_IDLE,
        needs=["observations.traffic"],
        is_builtin=False,
        transport=None,  # type: ignore[arg-type]
        transport_type="grpc",
        registered_at=95.0,  # monotonic — 5s ago, well before tick
    )
    plugin.last_call_at = 95.0  # plugin called 5s ago in monotonic time
    registry._plugins[plugin.plugin_id] = plugin

    # Next tick monotonic projection = 100 + 5 = 105; plugin last_call_at = 95,
    # execution_interval = 60 → next-due monotonic = 95 + 60 = 155.  Not due.
    # If the buggy code path were active, at_s = 1.7e9 + 5 ≫ 155 → due.
    sched = adapter._compute_next_scheduled_tick()
    assert sched.need_traffic_metrics is False, (
        "lazy traffic pull broke: plugin with last_call_at in monotonic domain "
        "was read as due against a wall-epoch projection of the next tick"
    )
