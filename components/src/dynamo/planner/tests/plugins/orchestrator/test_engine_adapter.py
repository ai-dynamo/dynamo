# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for ``OrchestratorEngineAdapter`` cadence.

Covers bugs caught after K8s smoke / plugin-path review:

- ``initial_tick`` previously read ``self._config.throughput_adjustment_interval``
  (missing ``_seconds`` suffix). The Pydantic ``validation_alias`` only affects
  input parsing — attribute access requires the canonical name. Triggered an
  ``AttributeError`` whenever ``enable_throughput_scaling=True``.

- The old dual-cadence merge used a too-tight tolerance, which split load
  and throughput work that should have been coordinated.

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
    FpmObservations,
    ScheduledTick,
    TickInput,
    TrafficObservation,
    WorkerCapabilities,
    WorkerCounts,
)
from dynamo.planner.plugins.clock import VirtualClock
from dynamo.planner.plugins.merge.types import ChainAugmentOutcome
from dynamo.planner.plugins.orchestrator.engine_adapter import OrchestratorEngineAdapter
from dynamo.planner.plugins.orchestrator.pipeline import PipelineOutcome
from dynamo.planner.plugins.types import ComponentTarget, ScalingProposal

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


def _agg_config_custom_intervals(
    *, load_interval: float, throughput_interval: float
) -> PlannerConfig:
    return PlannerConfig.model_validate(
        {
            "mode": "agg",
            "enable_load_scaling": True,
            "enable_throughput_scaling": True,
            "optimization_target": "sla",
            "served_model_name": "test",
            "load_adjustment_interval_seconds": load_interval,
            "throughput_adjustment_interval_seconds": throughput_interval,
        }
    )


def _agg_config_with_in_process_plugin() -> PlannerConfig:
    from dynamo.planner.plugins.registry.config import (
        InProcessPluginSpec,
        PluginRegistrationConfig,
    )

    return PlannerConfig(
        mode="agg",
        enable_load_scaling=True,
        enable_throughput_scaling=True,
        optimization_target="sla",
        served_model_name="test",
        plugin_registration=PluginRegistrationConfig(
            in_process_plugins=[
                InProcessPluginSpec.model_validate(
                    {
                        "module": "dynamo.planner.tests.plugins.orchestrator._fake_in_process_plugin",
                        "class": "FakePlugin",
                        "plugin_id": "configured_in_process",
                        "plugin_type": "propose",
                        "priority": 50,
                        "kwargs": {"tag": "from-config"},
                    }
                )
            ]
        ),
    )


def _disagg_caps() -> WorkerCapabilities:
    return WorkerCapabilities(
        prefill=EngineCapabilities(
            num_gpu=1, max_num_batched_tokens=2048, max_kv_tokens=16384
        ),
        decode=EngineCapabilities(
            num_gpu=1, max_num_batched_tokens=2048, max_kv_tokens=16384
        ),
    )


def _disagg_config_sla() -> PlannerConfig:
    return PlannerConfig(
        mode="disagg",
        enable_load_scaling=True,
        enable_throughput_scaling=True,
        optimization_target="sla",
        served_model_name="test",
    )


def _disagg_config_sla_no_budget() -> PlannerConfig:
    return PlannerConfig(
        mode="disagg",
        enable_load_scaling=True,
        enable_throughput_scaling=True,
        optimization_target="sla",
        served_model_name="test",
        max_gpu_budget=-1,
        min_gpu_budget=-1,
    )


def _make_fpm(worker_id: str = "w1", dp_rank: int = 0):
    from dynamo.common.forward_pass_metrics import (
        ForwardPassMetrics,
        QueuedRequestMetrics,
        ScheduledRequestMetrics,
    )

    return ForwardPassMetrics(
        worker_id=worker_id,
        dp_rank=dp_rank,
        wall_time=0.01,
        scheduled_requests=ScheduledRequestMetrics(
            sum_prefill_tokens=0,
            num_prefill_requests=0,
            sum_decode_kv_tokens=100,
            num_decode_requests=1,
        ),
        queued_requests=QueuedRequestMetrics(
            sum_prefill_tokens=0,
            sum_decode_kv_tokens=0,
        ),
    )


@pytest.mark.asyncio
async def test_builtin_load_propose_refreshes_throughput_bound_before_load():
    from dynamo.planner.plugins.builtins.local_planner import BuiltinLoadPropose
    from dynamo.planner.plugins.types import (
        ObservationData,
        PipelineContext,
        PredictionData,
        ProposeStageRequest,
        TrafficMetrics,
    )

    calls: list[str] = []

    class State:
        def advance_throughput_from_prediction(self, *args, **kwargs):
            calls.append("throughput")
            return None

        def advance_load(self, obs, **kwargs):
            calls.append("load")
            return None

    cfg = _agg_config_throughput_on()
    plugin = BuiltinLoadPropose(cfg, State())  # type: ignore[arg-type]

    response = await plugin.Propose(
        ProposeStageRequest(
            context=PipelineContext(
                observations=ObservationData(
                    traffic=TrafficMetrics(
                        duration_s=60.0,
                        num_req=100.0,
                        isl=512.0,
                        osl=128.0,
                    )
                ),
                predictions=PredictionData(
                    predicted_num_req=120.0,
                    predicted_isl=512.0,
                    predicted_osl=128.0,
                    predicted_kv_hit_rate=0.1,
                    source="builtin_load_predict",
                ),
            )
        )
    )

    assert calls == ["throughput", "load"]
    assert response.accept is not None


def _build_real_regression(cfg: PlannerConfig, caps: WorkerCapabilities, kind: str):
    """Build a regression the exact way production does.

    ``PlannerScalingState`` in SLA mode constructs ``PlannerEnginePerfModel``
    instances in its ``_{agg,prefill,decode}_regression`` slots — the same
    objects the orchestrator path installs via ``install_regressions``. We
    reuse that construction so the test exercises the real type (which only
    exposes ``add_observations``, never the singular ``add_observation``).
    """
    from dynamo.planner.core.state_machine import PlannerScalingState

    state = PlannerScalingState(cfg, caps)
    return getattr(state, f"_{kind}_regression")


def test_observe_fpm_feeds_installed_regression_without_crashing_agg():
    """Regression guard for the add_observation→add_observations P1.

    The orchestrator FPM-observation feed (``_observe_fpm``) is reached on
    every SLA-mode load tick when ``fpm_observations`` is non-empty and a
    regression is installed. The regression slots hold
    ``PlannerEnginePerfModel``, which exposes only ``add_observations(dict)``
    — the pre-fix singular ``add_observation(fpm)`` raised AttributeError
    and crashed the tick. No unit test or K8s smoke exercised this exact
    combination (SLA + live FPM + installed regression), so the crash
    shipped silently. Feed a *real* ``PlannerEnginePerfModel`` (built the
    same way builtin planner state builds it) and assert ``_observe_fpm`` does not raise.
    """
    cfg = _agg_config_throughput_on()  # agg, SLA
    caps = _caps()
    adapter = OrchestratorEngineAdapter(cfg, caps)
    adapter.install_regressions(agg=_build_real_regression(cfg, caps, "agg"))
    assert adapter._orchestrator.get_regression("agg") is not None

    # Pre-fix this raised:
    #   AttributeError: 'PlannerEnginePerfModel' object has no attribute
    #   'add_observation'
    adapter._observe_fpm(FpmObservations(decode={("w1", 0): _make_fpm()}))


def test_observe_fpm_feeds_installed_regression_without_crashing_disagg():
    """Same guard for the disagg prefill+decode branches of ``_observe_fpm``."""
    cfg = _disagg_config_sla()
    caps = _disagg_caps()
    adapter = OrchestratorEngineAdapter(cfg, caps)
    adapter.install_regressions(
        prefill=_build_real_regression(cfg, caps, "prefill"),
        decode=_build_real_regression(cfg, caps, "decode"),
    )
    assert adapter._orchestrator.get_regression("prefill") is not None
    assert adapter._orchestrator.get_regression("decode") is not None

    adapter._observe_fpm(
        FpmObservations(
            prefill={("p1", 0): _make_fpm("p1")},
            decode={("d1", 0): _make_fpm("d1")},
        )
    )


def _apply_outcome(targets):
    return PipelineOutcome(
        execute_action="apply",
        final_proposal=ScalingProposal(targets=targets),
    )


def test_project_scale_to_both_components_changed():
    # apply + prefill & decode both differ from current → full decision.
    adapter = OrchestratorEngineAdapter(_disagg_config_sla_no_budget(), _disagg_caps())
    wc = WorkerCounts(ready_num_prefill=2, ready_num_decode=4)
    outcome = _apply_outcome(
        [
            ComponentTarget(sub_component_type="prefill", replicas=6),
            ComponentTarget(sub_component_type="decode", replicas=8),
        ]
    )
    dec = adapter._project_scale_to(outcome, wc)
    assert dec is not None
    assert dec.num_prefill == 6
    assert dec.num_decode == 8


def test_project_scale_to_no_change_returns_none():
    # apply but both equal current -> planner "no change -> None".
    adapter = OrchestratorEngineAdapter(_disagg_config_sla_no_budget(), _disagg_caps())
    wc = WorkerCounts(ready_num_prefill=6, ready_num_decode=8)
    outcome = _apply_outcome(
        [
            ComponentTarget(sub_component_type="prefill", replicas=6),
            ComponentTarget(sub_component_type="decode", replicas=8),
        ]
    )
    assert adapter._project_scale_to(outcome, wc) is None


def test_project_scale_to_single_component_proposal():
    # Agg-mode proposals mention only decode; prefill stays None (no opinion),
    # decode changed → decision emitted.
    adapter = OrchestratorEngineAdapter(_agg_config_throughput_on(), _caps())
    wc = WorkerCounts(ready_num_prefill=2, ready_num_decode=4)
    outcome = _apply_outcome([ComponentTarget(sub_component_type="decode", replicas=6)])
    dec = adapter._project_scale_to(outcome, wc)
    assert dec is not None
    assert dec.num_prefill is None
    assert dec.num_decode == 6


def test_project_scale_to_non_apply_action_returns_none():
    adapter = OrchestratorEngineAdapter(_disagg_config_sla_no_budget(), _disagg_caps())
    wc = WorkerCounts(ready_num_prefill=2, ready_num_decode=4)
    for action in ("skip_short_circuit", "skip_no_targets", "skip_tick_timeout"):
        outcome = PipelineOutcome(execute_action=action, final_proposal=None)
        assert adapter._project_scale_to(outcome, wc) is None


def test_project_scale_to_applies_final_gpu_budget_to_external_proposal():
    cfg = PlannerConfig(
        mode="disagg",
        enable_load_scaling=True,
        enable_throughput_scaling=True,
        optimization_target="sla",
        served_model_name="test",
        max_gpu_budget=4,
        min_gpu_budget=-1,
    )
    adapter = OrchestratorEngineAdapter(cfg, _disagg_caps())
    wc = WorkerCounts(ready_num_prefill=1, ready_num_decode=1)
    outcome = _apply_outcome(
        [
            ComponentTarget(sub_component_type="prefill", replicas=5),
            ComponentTarget(sub_component_type="decode", replicas=5),
        ]
    )

    dec = adapter._project_scale_to(outcome, wc)

    assert dec is not None
    assert dec.num_prefill + dec.num_decode <= 4


def test_project_scale_to_budget_preserves_single_component_target_mask():
    cfg = PlannerConfig(
        mode="disagg",
        enable_load_scaling=True,
        enable_throughput_scaling=True,
        optimization_target="sla",
        served_model_name="test",
        max_gpu_budget=4,
        min_gpu_budget=-1,
    )
    adapter = OrchestratorEngineAdapter(cfg, _disagg_caps())
    wc = WorkerCounts(ready_num_prefill=6, ready_num_decode=1)
    outcome = _apply_outcome([ComponentTarget(sub_component_type="decode", replicas=4)])

    dec = adapter._project_scale_to(outcome, wc)

    assert dec is not None
    assert dec.num_prefill is None
    assert dec.num_decode == 2


def test_project_scale_to_does_not_apply_budget_on_baseline_only_noop():
    cfg = PlannerConfig(
        mode="disagg",
        enable_load_scaling=True,
        enable_throughput_scaling=True,
        optimization_target="sla",
        served_model_name="test",
        max_gpu_budget=4,
        min_gpu_budget=-1,
    )
    adapter = OrchestratorEngineAdapter(cfg, _disagg_caps())
    wc = WorkerCounts(ready_num_prefill=6, ready_num_decode=6)
    outcome = _apply_outcome(
        [
            ComponentTarget(sub_component_type="prefill", replicas=6),
            ComponentTarget(sub_component_type="decode", replicas=6),
        ]
    )

    assert adapter._project_scale_to(outcome, wc) is None


def test_tick_input_to_context_maps_observations_and_fpm():
    # The ingress glue: TickInput → PipelineContext.observations, including
    # the FPM msgpack encoding external plugins decode. Asserts field mapping
    # (traffic + worker scaling flags) and that the FPM bytes round-trip.
    from dynamo.common.forward_pass_metrics import decode as _fpm_decode

    adapter = OrchestratorEngineAdapter(_agg_config_throughput_on(), _caps())
    ti = TickInput(
        now_s=10.0,
        traffic=TrafficObservation(
            duration_s=60.0,
            num_req=100.0,
            isl=1000.0,
            osl=150.0,
            kv_hit_rate=0.4,
            accept_length=2.5,
        ),
        worker_counts=WorkerCounts(
            ready_num_prefill=2,
            ready_num_decode=4,
            prefill_scaling_in_progress=True,
            decode_scaling_in_progress=False,
        ),
        fpm_observations=FpmObservations(decode={("w1", 0): _make_fpm("w1")}),
    )
    ctx = adapter._tick_input_to_context(ti)

    assert ctx.observations.traffic.num_req == 100.0
    assert ctx.observations.traffic.kv_hit_rate == 0.4
    assert ctx.observations.traffic.accept_length == 2.5
    assert ctx.observations.workers.ready_decode == 4
    assert ctx.observations.workers.prefill_scaling_in_progress is True
    assert ctx.observations.workers.decode_scaling_in_progress is False

    # FPM key format is "<worker_id>/<dp_rank>" and the value is a
    # canonical-encoded ForwardPassMetrics the external plugin decodes.
    raw = ctx.observations.fpm.decode_engines["w1/0"]
    back = _fpm_decode(raw)
    assert back is not None
    assert back.worker_id == "w1"


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


def test_orchestrator_path_honours_configured_protocol_version_range():
    """``planner.plugin_registration.protocol_version_min/max`` must
    flow into the orchestrator-path registry server (previously dropped
    on the floor — server defaulted to ``("1.0", "1.0")`` regardless of
    config, making any non-default range silently ineffective on the
    gateway).
    """
    from dynamo.planner.config.planner_config import PluginRegistrationConfig

    config = PlannerConfig(
        mode="agg",
        enable_load_scaling=True,
        enable_throughput_scaling=True,
        optimization_target="sla",
        served_model_name="test",
        plugin_registration=PluginRegistrationConfig(
            protocol_version_min="1.0",
            protocol_version_max="1.5",
        ),
    )
    adapter = OrchestratorEngineAdapter(config, _caps())
    server = adapter._orchestrator._registry  # type: ignore[attr-defined]
    assert server._protocol_min == "1.0"
    assert server._protocol_max == "1.5"


@pytest.mark.asyncio
async def test_tick_propagates_pipeline_execute_action_to_diagnostics():
    """``PipelineOutcome.execute_action`` / ``short_circuit_reason`` /
    ``audit_events`` must surface on ``PlannerEffects.diagnostics`` so
    in-process consumers (replay adapter, diagnostics recorder) can
    tell ``apply`` from ``skip_short_circuit`` / ``skip_no_targets`` /
    ``skip_tick_timeout`` without scraping Prometheus.

    Pre-fix the adapter created a fresh ``TickDiagnostics()`` and only
    populated prediction / load / throughput fields — the three
    execute-action fields were silently dropped.
    """
    from dynamo.planner.plugins.orchestrator.pipeline import PipelineOutcome

    adapter = OrchestratorEngineAdapter(_agg_config_throughput_on(), _caps())

    canned_outcome = PipelineOutcome(
        execute_action="skip_short_circuit",
        final_proposal=None,
        short_circuit_reason="propose: my-plugin: over-capacity",
        audit_events=[
            "chain_break_warning: predict-A set final=true at non-lowest priority"
        ],
    )

    async def fake_tick(ctx, baseline):
        return canned_outcome

    adapter._orchestrator.tick = fake_tick  # type: ignore[method-assign]

    initial_tick = adapter.initial_tick(start_s=0.0)
    effects = await adapter.tick(initial_tick, TickInput(now_s=initial_tick.at_s))

    assert effects.diagnostics.execute_action == "skip_short_circuit"
    assert (
        effects.diagnostics.short_circuit_reason == "propose: my-plugin: over-capacity"
    )
    assert effects.diagnostics.audit_events == [
        "chain_break_warning: predict-A set final=true at non-lowest priority"
    ]


@pytest.mark.asyncio
async def test_bootstrap_registers_static_externals_before_bootstrap_fanout():
    """Static external plugins (``scheduling.external_plugins``) must be
    registered **before** the orchestrator-side Bootstrap fan-out so
    they receive the same ``historical_traffic`` warm + Bootstrap RPC
    pass as in-process / builtin plugins.

    Pre-fix order was bootstrap → register → gateway, which meant
    config-listed externals registered into an already-bootstrapped
    registry and silently missed the warm step.  This test pins the
    correct ordering via monkey-patched call records.
    """
    adapter = OrchestratorEngineAdapter(_agg_config_throughput_on(), _caps())

    call_order: list[str] = []

    orig_orchestrator_bootstrap = adapter._orchestrator.bootstrap_plugins

    async def record_orchestrator_bootstrap(*args, **kwargs):
        call_order.append("bootstrap")
        await orig_orchestrator_bootstrap(*args, **kwargs)

    orig_wire = adapter._wire_external_plugins_from_config

    async def record_wire():
        call_order.append("wire_externals")
        await orig_wire()

    orig_gateway = adapter._maybe_start_gateway

    async def record_gateway():
        call_order.append("gateway")
        await orig_gateway()

    adapter._orchestrator.bootstrap_plugins = (  # type: ignore[method-assign]
        record_orchestrator_bootstrap
    )
    adapter._wire_external_plugins_from_config = (  # type: ignore[method-assign]
        record_wire
    )
    adapter._maybe_start_gateway = record_gateway  # type: ignore[method-assign]

    await adapter.bootstrap_plugins()

    assert call_order == ["wire_externals", "bootstrap", "gateway"], call_order


@pytest.mark.asyncio
async def test_bootstrap_loads_configured_in_process_plugins():
    adapter = OrchestratorEngineAdapter(_agg_config_with_in_process_plugin(), _caps())
    assert adapter._orchestrator.registry.get_plugin("configured_in_process") is None

    await adapter.bootstrap_plugins()

    plugin = adapter._orchestrator.registry.get_plugin("configured_in_process")
    assert plugin is not None
    assert plugin.is_builtin is False
    assert plugin.transport._instance.tag == "from-config"  # noqa: SLF001


def test_pipeline_fires_at_scale_interval_cadence():
    """Under the scale_interval cadence model, the engine_adapter no longer
    runs dual-cadence merge logic.

    Pipeline fires once per ``scale_interval_seconds`` from the last
    tick moment.  Per-plugin throttling (in ``PluginScheduler._is_due``)
    decides which plugins actually fire each tick — the merge-tolerance
    concept that used to live here is now naturally absorbed by the
    plugin scheduler, which evaluates each plugin's throttle
    independently using the same ``now`` value.

    This test locks the shape:
    ``next_tick.at_s = last_tick + scale_interval``, exactly.
    """
    adapter = OrchestratorEngineAdapter(_agg_config_throughput_on(), _caps())
    tick = adapter.initial_tick(start_s=0.0)
    # Default scale_interval_seconds = 5.0 (see SchedulingConfig).
    assert tick.at_s == pytest.approx(5.0, abs=1e-9)
    assert tick.run_load_scaling
    assert not tick.run_throughput_scaling


def test_scale_interval_advances_from_actual_tick_now():
    """Sequential ticks anchor on ``tick_input.now_s``, not on a
    pre-computed schedule — so a 700ms-late tick at T=5.7 produces the
    next tick at T=10.7. This is the basic contract for scale_interval
    cadence advancement.
    """

    adapter = OrchestratorEngineAdapter(_agg_config_throughput_on(), _caps())
    initial = adapter.initial_tick(start_s=0.0)
    assert initial.at_s == pytest.approx(5.0)


def test_lazy_traffic_pull_skips_prometheus_when_no_plugin_needs_traffic():
    """Under scale_interval, ``need_traffic_metrics`` is True only
    when some registered plugin both lists ``observations.traffic``
    in its ``needs`` AND is due at the next tick. Builtin throughput
    plugins consume traffic, but they are not due on the first 5s
    pipeline tick when their execution interval is 180s.
    """
    vc = VirtualClock()
    adapter = OrchestratorEngineAdapter(_agg_config_throughput_on(), _caps(), clock=vc)
    tick = adapter.initial_tick(start_s=0.0)
    assert tick.need_traffic_metrics is False
    assert tick.use_full_traffic_metrics is False
    assert tick.traffic_metrics_duration_s == 0.0


def test_lazy_traffic_pull_fires_when_builtin_predict_is_due():
    """At the throughput interval boundary, builtin predict/propose plugins
    are due and the adapter requests the throughput-sized traffic window."""
    vc = VirtualClock()
    adapter = OrchestratorEngineAdapter(_agg_config_throughput_on(), _caps(), clock=vc)
    adapter.initial_tick(start_s=0.0)
    adapter._last_tick_s = 175.0
    adapter._last_tick_monotonic = 175.0

    tick = adapter._compute_next_scheduled_tick()

    assert tick.at_s == pytest.approx(180.0)
    assert tick.need_traffic_metrics is True
    assert tick.use_full_traffic_metrics is True
    assert tick.traffic_metrics_duration_s == pytest.approx(
        adapter._config.throughput_adjustment_interval_seconds
    )


def test_load_only_sla_uses_kv_hit_rate_traffic_scrape():
    cfg = PlannerConfig.model_validate(
        {
            "mode": "agg",
            "enable_load_scaling": True,
            "enable_throughput_scaling": False,
            "optimization_target": "sla",
            "served_model_name": "test",
            "load_adjustment_interval_seconds": 7.0,
            "throughput_adjustment_interval_seconds": 60.0,
        }
    )
    vc = VirtualClock()
    adapter = OrchestratorEngineAdapter(cfg, _caps(), clock=vc)

    tick = adapter.initial_tick(start_s=0.0)

    assert tick.at_s == pytest.approx(7.0)
    assert tick.run_load_scaling
    assert not tick.run_throughput_scaling
    assert tick.need_traffic_metrics is True
    assert tick.use_full_traffic_metrics is False
    assert tick.traffic_metrics_duration_s == pytest.approx(
        cfg.load_adjustment_interval_seconds
    )


def test_lazy_fpm_pull_only_when_load_builtin_is_due():
    cfg = _agg_config_custom_intervals(load_interval=7.0, throughput_interval=60.0)
    assert cfg.scheduling.scale_interval_seconds == pytest.approx(1.0)
    vc = VirtualClock()
    adapter = OrchestratorEngineAdapter(cfg, _caps(), clock=vc)

    first = adapter.initial_tick(start_s=0.0)
    assert first.at_s == pytest.approx(1.0)
    assert first.need_worker_fpm is False
    assert not first.run_load_scaling
    assert not first.run_throughput_scaling

    adapter._last_tick_s = 6.0
    adapter._last_tick_monotonic = 6.0
    load_tick = adapter._compute_next_scheduled_tick()
    assert load_tick.at_s == pytest.approx(7.0)
    assert load_tick.need_worker_fpm is True
    assert load_tick.run_load_scaling
    assert not load_tick.run_throughput_scaling


def test_throughput_only_sla_still_pulls_fpm_for_live_regression():
    cfg = PlannerConfig.model_validate(
        {
            "mode": "agg",
            "enable_load_scaling": False,
            "enable_throughput_scaling": True,
            "optimization_target": "sla",
            "served_model_name": "test",
            "load_adjustment_interval_seconds": 7.0,
            "throughput_adjustment_interval_seconds": 60.0,
        }
    )
    adapter = OrchestratorEngineAdapter(cfg, _caps(), clock=VirtualClock())
    assert adapter._orchestrator.registry.get_plugin("builtin_load_propose") is None

    first = adapter.initial_tick(start_s=0.0)
    assert first.need_worker_fpm is False

    adapter._last_tick_s = 6.0
    adapter._last_tick_monotonic = 6.0
    load_interval_tick = adapter._compute_next_scheduled_tick()
    assert load_interval_tick.at_s == pytest.approx(7.0)
    assert load_interval_tick.need_worker_fpm is True
    assert load_interval_tick.run_load_scaling
    assert not load_interval_tick.run_throughput_scaling


def test_builtin_first_fire_is_anchored_to_initial_tick_not_construction():
    cfg = _agg_config_custom_intervals(load_interval=7.0, throughput_interval=60.0)
    vc = VirtualClock()
    adapter = OrchestratorEngineAdapter(cfg, _caps(), clock=vc)
    vc.advance(30.0)

    first = adapter.initial_tick(start_s=0.0)

    assert first.at_s == pytest.approx(1.0)
    assert first.need_worker_fpm is False


@pytest.mark.asyncio
async def test_tick_skips_fpm_observation_when_fpm_was_not_requested():
    cfg = _agg_config_custom_intervals(load_interval=7.0, throughput_interval=60.0)
    adapter = OrchestratorEngineAdapter(cfg, _caps(), clock=VirtualClock())
    tick = adapter.initial_tick(start_s=0.0)
    assert tick.need_worker_fpm is False

    calls: list[FpmObservations] = []

    def record_fpm(obs):
        calls.append(obs)

    adapter._scaling_state.observe_fpm = record_fpm  # type: ignore[method-assign]

    await adapter.tick(
        tick,
        TickInput(
            now_s=tick.at_s,
            fpm_observations=FpmObservations(decode={("w1", 0): _make_fpm("w1")}),
        ),
    )

    assert calls == []


@pytest.mark.asyncio
async def test_external_fpm_request_does_not_feed_builtin_regression_early():
    cfg = _agg_config_custom_intervals(load_interval=7.0, throughput_interval=60.0)
    adapter = OrchestratorEngineAdapter(cfg, _caps(), clock=VirtualClock())
    adapter._orchestrator.register_internal(
        plugin_id="external_fpm_consumer",
        plugin_type="propose",
        priority=90,
        instance=object(),
        execution_interval_seconds=0.0,
        needs=["observations.fpm"],
        is_builtin=False,
    )
    tick = adapter.initial_tick(start_s=0.0)
    assert tick.at_s == pytest.approx(1.0)
    assert tick.need_worker_fpm is True
    assert not tick.run_load_scaling

    async def fake_orchestrator_tick(ctx, baseline):
        return PipelineOutcome(execute_action="skip_no_targets", final_proposal=None)

    adapter._orchestrator.tick = fake_orchestrator_tick  # type: ignore[method-assign]
    calls: list[FpmObservations] = []

    def record_fpm(obs):
        calls.append(obs)

    adapter._scaling_state.observe_fpm = record_fpm  # type: ignore[method-assign]

    await adapter.tick(
        tick,
        TickInput(
            now_s=tick.at_s,
            fpm_observations=FpmObservations(decode={("w1", 0): _make_fpm("w1")}),
        ),
    )

    assert calls == []


@pytest.mark.asyncio
async def test_internal_load_tick_feeds_builtin_regression():
    cfg = _agg_config_custom_intervals(load_interval=7.0, throughput_interval=60.0)
    adapter = OrchestratorEngineAdapter(cfg, _caps(), clock=VirtualClock())
    adapter.initial_tick(start_s=0.0)
    adapter._last_tick_s = 6.0
    adapter._last_tick_monotonic = 6.0
    tick = adapter._compute_next_scheduled_tick()
    assert tick.at_s == pytest.approx(7.0)
    assert tick.need_worker_fpm is True
    assert tick.run_load_scaling

    async def fake_orchestrator_tick(ctx, baseline):
        return PipelineOutcome(execute_action="skip_no_targets", final_proposal=None)

    adapter._orchestrator.tick = fake_orchestrator_tick  # type: ignore[method-assign]
    calls: list[FpmObservations] = []

    def record_fpm(obs):
        calls.append(obs)

    adapter._scaling_state.observe_fpm = record_fpm  # type: ignore[method-assign]

    await adapter.tick(
        tick,
        TickInput(
            now_s=tick.at_s,
            fpm_observations=FpmObservations(decode={("w1", 0): _make_fpm("w1")}),
        ),
    )

    assert len(calls) == 1


@pytest.mark.asyncio
async def test_disabled_load_scaling_reports_disabled_on_load_tick():
    cfg = PlannerConfig.model_validate(
        {
            "mode": "agg",
            "enable_load_scaling": False,
            "enable_throughput_scaling": True,
            "optimization_target": "sla",
            "served_model_name": "test",
            "load_adjustment_interval_seconds": 7.0,
            "throughput_adjustment_interval_seconds": 60.0,
        }
    )
    adapter = OrchestratorEngineAdapter(cfg, _caps(), clock=VirtualClock())
    adapter.initial_tick(start_s=0.0)
    adapter._last_tick_s = 6.0
    adapter._last_tick_monotonic = 6.0
    tick = adapter._compute_next_scheduled_tick()
    assert tick.run_load_scaling
    assert not tick.run_throughput_scaling

    async def fake_orchestrator_tick(ctx, baseline):
        return PipelineOutcome(execute_action="skip_no_targets", final_proposal=None)

    adapter._orchestrator.tick = fake_orchestrator_tick  # type: ignore[method-assign]

    effects = await adapter.tick(
        tick,
        TickInput(now_s=tick.at_s, worker_counts=WorkerCounts(ready_num_decode=1)),
    )

    assert effects.diagnostics.load_decision_reason == "disabled"


@pytest.mark.asyncio
async def test_predict_failed_reason_surfaces_on_throughput_tick():
    adapter = OrchestratorEngineAdapter(
        _agg_config_throughput_on(), _caps(), clock=VirtualClock()
    )
    adapter.initial_tick(start_s=0.0)
    adapter._last_tick_s = 175.0
    adapter._last_tick_monotonic = 175.0
    tick = adapter._compute_next_scheduled_tick()
    assert tick.run_throughput_scaling

    async def fake_orchestrator_tick(ctx, baseline):
        return PipelineOutcome(
            execute_action="skip_no_targets",
            final_proposal=None,
            predict_outcome=ChainAugmentOutcome(
                prediction=None,
                reasons=["predict_failed"],
            ),
        )

    adapter._orchestrator.tick = fake_orchestrator_tick  # type: ignore[method-assign]

    effects = await adapter.tick(
        tick,
        TickInput(now_s=tick.at_s, worker_counts=WorkerCounts(ready_num_decode=1)),
    )

    assert effects.diagnostics.throughput_decision_reason == "predict_failed"


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
    never re-firing after its first call on the replay path.
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


def test_lazy_traffic_pull_matches_dot_path_sub_paths_of_observations_traffic():
    """``needs`` are dot-paths into ``PipelineContext`` per the proto
    contract.  A plugin declaring a sub-path like
    ``"observations.traffic.num_req"`` (signalling "I only consume
    num_req — you may trim the rest from ctx") must still trigger the
    lazy Prometheus pull, because the sub-path can only resolve if
    its parent ``ctx.observations.traffic`` is populated.

    Sibling fields like ``"observations.traffic_legacy"`` must NOT
    trigger the pull — the trailing ``.`` in the prefix-match is
    load-bearing.  Locks both branches.
    """
    from dynamo.planner.plugins.registry.types import RegisteredPlugin
    from dynamo.planner.plugins.types import HoldPolicy

    def _make_traffic_plugin(plugin_id: str, needs: list[str]) -> RegisteredPlugin:
        plugin = RegisteredPlugin(
            plugin_id=plugin_id,
            plugin_type="propose",
            priority=10,
            endpoint=f"inproc://{plugin_id}",
            version="test",
            protocol_version="1.0",
            execution_interval_seconds=0.0,  # every tick, always due
            hold_policy=HoldPolicy.ACCEPT_WHEN_IDLE,
            needs=needs,
            is_builtin=False,
            transport=None,  # type: ignore[arg-type]
            transport_type="grpc",
            registered_at=0.0,
        )
        plugin.last_call_at = float("-inf")
        return plugin

    # --- Case 1: exact parent path matches (existing behavior, keep working) ---
    adapter = OrchestratorEngineAdapter(_agg_config_throughput_on(), _caps())
    adapter._orchestrator._registry._plugins["p_parent"] = _make_traffic_plugin(
        "p_parent", ["observations.traffic"]
    )
    sched = adapter._compute_next_scheduled_tick()
    assert sched.need_traffic_metrics is True, "parent path must trigger pull"
    del adapter._orchestrator._registry._plugins["p_parent"]

    # --- Case 2: sub-path matches (regression — was broken before fix) ---
    adapter._orchestrator._registry._plugins["p_child"] = _make_traffic_plugin(
        "p_child", ["observations.traffic.num_req"]
    )
    sched = adapter._compute_next_scheduled_tick()
    assert sched.need_traffic_metrics is True, (
        "sub-path of observations.traffic must trigger pull — without this "
        "the plugin would receive ctx.observations.traffic == None despite "
        "declaring a dot-path into it"
    )
    del adapter._orchestrator._registry._plugins["p_child"]

    # --- Case 3: sibling field must NOT match (prefix guard) ---
    adapter._orchestrator._registry._plugins["p_sibling"] = _make_traffic_plugin(
        "p_sibling", ["observations.traffic_legacy"]
    )
    sched = adapter._compute_next_scheduled_tick()
    assert (
        sched.need_traffic_metrics is False
    ), "prefix match must not over-fire on sibling field 'observations.traffic_legacy'"

    # --- Case 4: completely unrelated needs must NOT match ---
    adapter._orchestrator._registry._plugins["p_other"] = _make_traffic_plugin(
        "p_other", ["observations.fpm", "predictions"]
    )
    sched = adapter._compute_next_scheduled_tick()
    assert (
        sched.need_traffic_metrics is False
    ), "unrelated needs must not trigger the traffic pull"
