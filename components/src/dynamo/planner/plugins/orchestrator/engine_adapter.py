# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``OrchestratorEngineAdapter`` — production ``EngineProtocol`` adapter
for the plugin chain.

Wraps ``LocalPlannerOrchestrator`` behind the same ``initial_tick`` /
``tick`` / ``shutdown`` interface consumed by ``NativePlannerBase``.

Architecture invariant: PipelineContext is the only input channel
--------------------------------------------------------------------
All plugins — both in-process builtins and external gRPC plugins —
receive their per-tick inputs through
``PipelineContext.observations`` exclusively. There is **no**
``prime_tick(...)`` side-channel, ``self._last_fpm``-style stash,
or any other path that delivers observation data to a plugin instance
outside of the stage RPC.

This invariant ensures:
  * Plugin API is uniform across in-process and over-wire transports.
  * Adding a new observation field requires touching one schema
    (``ObservationData``), not two delivery paths.
  * Builtin plugins and external plugins receive byte-identical input,
    so public plugin behavior can be tested against the same context shape.

Internal responsibilities
-------------------------

1. **Tick lifecycle cadence tracking**:
   Owns the single ``scale_interval_seconds`` pipeline cadence. Plugin
   ``execution_interval_seconds`` values decide which builtins or external
   plugins fire on each pipeline tick.
2. **TickInput → PipelineContext bridge**:
   Extracts ``traffic`` into ``TrafficMetrics``, ``worker_counts``
   (counts + scaling-in-progress flags) into ``WorkerState``, and
   per-engine FPM observations into ``FpmData`` (msgspec/msgpack-
   encoded, keyed by ``"<worker_id>/<dp_rank>"``) on ``ObservationData``.
   External plugins declaring ``needs=["observations.fpm"]`` receive
   the FPM map; an empty/absent submap means "no FPM this tick".
3. **FPM regression observation**:
   Before load-cadence ticks, feeds FPM into the shared scaling-state
   regression models. This is a planner-internal regression-fit path,
   distinct from delivering FPM to plugins that request
   ``observations.fpm``.
4. **PipelineOutcome → PlannerEffects projection**:
   Reads the orchestrator's ``final_proposal.targets``, detects "no
   change" against ``worker_counts``, applies final min_endpoint / GPU
   budget invariants, and projects to
   ``PlannerEffects.scale_to`` and fills diagnostics from the shared
   scaling state.

Bootstrap API
-------------

``install_regressions`` + ``bootstrap_plugins`` mirror
``LocalPlannerOrchestrator``'s equivalents but are exposed on the
adapter so callers (mode subclasses) have a single entry point.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional, Sequence

from dynamo.common.forward_pass_metrics import encode as _encode_fpm_record

if TYPE_CHECKING:
    import grpc.aio

from dynamo.planner.core.budget import (
    proportional_clamp_pair,
    proportional_clamp_single,
)
from dynamo.planner.core.state_machine import PlannerScalingState
from dynamo.planner.core.types import (
    FpmObservations,
    PlannerEffects,
    ScalingDecision,
    ScheduledTick,
    TickDiagnostics,
    TickInput,
    TrafficObservation,
    WorkerCapabilities,
    WorkerCounts,
)
from dynamo.planner.plugins.builtins import (
    BuiltinLoadPredict,
    BuiltinLoadPropose,
    BuiltinThroughputPropose,
)
from dynamo.planner.plugins.builtins.observe import (
    EnvironmentObserver,
    ObserveStageRequest,
)
from dynamo.planner.plugins.clock import Clock, VirtualClock, WallClock
from dynamo.planner.plugins.merge.types import ComponentKey
from dynamo.planner.plugins.orchestrator.orchestrator import LocalPlannerOrchestrator
from dynamo.planner.plugins.registry.auth import AllowUnauthenticatedAuth
from dynamo.planner.plugins.registry.circuit_breaker import CircuitBreaker
from dynamo.planner.plugins.registry.config import build_auth_validator
from dynamo.planner.plugins.registry.server import PluginRegistryServer
from dynamo.planner.plugins.scheduler import PluginScheduler
from dynamo.planner.plugins.transport.config import make_transport_for_endpoint
from dynamo.planner.plugins.types import (
    FpmData,
    ObservationData,
    PipelineContext,
    TrafficMetrics,
    WorkerState,
)

log = logging.getLogger(__name__)


class OrchestratorEngineAdapter:
    """``EngineProtocol``-compatible wrapper around the builtin plugin chain.

    Lifecycle:

    1. ``OrchestratorEngineAdapter(config, capabilities)`` — builds
       orchestrator + local-planner builtins + registers them.
    2. ``install_regressions(prefill=, decode=, agg=)`` — fill the
       orchestrator's shared regression store.
    3. ``await bootstrap_plugins(historical_traffic=)`` — warm predictor
       + fire plugin Bootstrap RPCs.
    4. ``initial_tick(start_s)`` — get the first scheduled tick.
    5. ``await tick(scheduled_tick, tick_input)`` — repeatedly.
    6. ``await shutdown()`` — release plugin transports.
    """

    def __init__(
        self,
        config,  # PlannerConfig
        capabilities: WorkerCapabilities,
        *,
        observe_plugin: Optional[EnvironmentObserver] = None,
        clock: Optional[Clock] = None,
    ) -> None:
        self._config = config
        self._capabilities = capabilities
        self._observe_plugin = observe_plugin
        # Clock is shared with all sub-components (CircuitBreaker,
        # PluginRegistryServer, PluginScheduler, LocalPlannerOrchestrator).
        # Default ``WallClock`` is correct for production / K8s smoke
        # where ``tick_input.now_s`` already tracks wall-clock.
        # Replay paths pass a ``VirtualClock`` and call
        # ``advance_clock_to(tick_input.now_s)`` on each tick so plugin
        # scheduler ``is_due`` checks see trace time, not real time.
        # Without this hook a fast-forward replay (1hr trace in 10s real
        # time) would leave plugins with execution_interval >> 10s
        # never re-firing after the first tick.
        self._clock: Clock = clock if clock is not None else WallClock()

        # Scale_interval cadence model — pipeline fires once per
        # ``scale_interval_seconds`` regardless of individual plugin
        # cadences.  Per-plugin throttling (via
        # ``RegisteredPlugin.execution_interval_seconds``) handles
        # which plugins actually fire each tick.
        self._scale_interval: float = float(config.scheduling.scale_interval_seconds)
        # ``_last_tick_s`` is wall-epoch (matches ``tick_input.now_s`` /
        # ``ScheduledTick.at_s``).  ``_last_tick_monotonic`` is the
        # clock-domain twin used by the lazy-traffic-pull due-check
        # against ``RegisteredPlugin.last_call_at`` — which the
        # ``PluginScheduler.record_evaluation`` path stores in
        # ``self._clock.monotonic()`` domain.  Without the parallel
        # field, ``_is_due(p, _last_tick_s + scale_interval)`` would
        # compare wall-epoch against monotonic; in production
        # ``WallClock`` deployments that's ~1.7e9 vs ~1e3 — every
        # plugin reads as "due" and lazy-pull silently degenerates to
        # always-pull.  Replay path is unaffected because
        # ``VirtualClock.monotonic()`` is synchronised to
        # ``tick_input.now_s`` at the top of ``tick()``.
        self._last_tick_s: float = 0.0
        self._last_tick_monotonic: float = 0.0
        self._last_load_loop_monotonic: float = 0.0
        self._last_throughput_loop_monotonic: float = 0.0

        # Plugin-framework metrics live alongside the adapter so they
        # share the orchestrator's lifecycle.  Use the default global
        # ``prometheus_client.REGISTRY`` so planner's existing
        # ``start_http_server`` on ``metric_reporting_prometheus_port``
        # picks them up automatically.  Construction is lazy-guarded: if
        # an adapter is built a second time in the same Python process
        # (replay, tests), the first construction claims the metric
        # names on REGISTRY and the second would raise "Duplicated
        # timeseries" — we tolerate that by falling back to ``None`` so
        # emission becomes a no-op instead of crashing.
        from dynamo.planner.monitoring.planner_metrics import PluginFrameworkMetrics

        self._plugin_framework_metrics: Optional[PluginFrameworkMetrics]
        try:
            self._plugin_framework_metrics = PluginFrameworkMetrics()
        except ValueError:
            # Duplicate registration (only happens in tests / repeated
            # init in one process) — metrics emission disabled, but
            # the adapter still runs normally.
            self._plugin_framework_metrics = None

        # Build orchestrator + scheduler + circuit_breaker + registry.
        # All transport-shaped knobs (timeouts + wire security) live under
        # ``plugin_registration.transport``; we hand that subtree to the
        # transport factory verbatim. ``scheduling`` keeps only tick-level
        # knobs (``tick_max_duration_seconds`` etc).
        cb = CircuitBreaker(self._clock)
        transport_config = config.plugin_registration.transport

        def _factory(plugin_id, endpoint, *, in_process_instance=None):
            return make_transport_for_endpoint(
                plugin_id,
                endpoint,
                transport_config,
                in_process_instance=in_process_instance,
            )

        # Build the auth validator from config when ``trusted_sources`` is
        # set; otherwise fall back to ``AllowUnauthenticatedAuth`` so legacy
        # deployments that never configured plugin_registration still come
        # up (with the dev-mode WARN). Production manifests should populate
        # ``plugin_registration.auth.trusted_sources`` to opt in.
        auth_cfg = config.plugin_registration.auth
        if auth_cfg.trusted_sources:
            auth = build_auth_validator(auth_cfg)
        else:
            auth = AllowUnauthenticatedAuth()
        server = PluginRegistryServer(
            clock=self._clock,
            auth=auth,
            circuit_breaker=cb,
            transport_factory=_factory,
            # Honour the user-configured protocol-version range on the
            # orchestrator path.  Without this, ``planner.plugin_registration
            # .protocol_version_min/max`` was ineffective for plugins
            # registering through the gateway under the orchestrator path
            # (server defaulted to ``("1.0", "1.0")``).  Mirrors the
            # ``protocol_versions=...`` argument that
            # ``build_registry_from_config`` already passes on the
            # static-config path.
            protocol_versions=(
                config.plugin_registration.protocol_version_min,
                config.plugin_registration.protocol_version_max,
            ),
            # Phase-align ``registered_at`` to scale_interval boundary so
            # plugins with identical execution intervals fire on the same
            # pipeline tick irrespective of registration-time skew (see
            # design doc §4.3).  ``SchedulingConfig.scale_interval_seconds``
            # defaults to 5.0 — passes through unchanged on configs that
            # don't override it.
            scale_interval_seconds=config.scheduling.scale_interval_seconds,
        )
        scheduler = PluginScheduler(
            server, cb, self._clock, metrics=self._plugin_framework_metrics
        )
        self._orchestrator = LocalPlannerOrchestrator(
            registry=server,
            scheduler=scheduler,
            circuit_breaker=cb,
            clock=self._clock,
            tick_max_duration_seconds=config.scheduling.tick_max_duration_seconds,
            capabilities=capabilities,
            metrics=self._plugin_framework_metrics,
        )
        self._scaling_state = PlannerScalingState(config, capabilities)

        # Registration gateway lifecycle: populated lazily by
        # ``_maybe_start_gateway`` if config opts in; consumed by
        # ``shutdown``.  Default ``None`` keeps the typical (gateway
        # disabled) deployment path zero-cost.
        self._gateway_server: Optional[grpc.aio.Server] = None

        self._builtins: dict[str, object] = {}
        self._plugins_bootstrapped = False
        self._in_process_plugins_loaded = False
        self._register_builtin_plugins()

    @property
    def plugins_bootstrapped(self) -> bool:
        return self._plugins_bootstrapped

    def _register_builtin_plugins(self) -> None:
        """Register the in-process builtins that implement local planning."""
        cfg = self._config
        throughput_interval = float(cfg.throughput_adjustment_interval_seconds)
        load_interval = float(cfg.load_adjustment_interval_seconds)

        if cfg.enable_throughput_scaling:
            predict = BuiltinLoadPredict(cfg, self._capabilities)
            self._builtins["load_predict"] = predict
            self._orchestrator.register_internal(
                plugin_id=predict.plugin_id,
                plugin_type="predict",
                priority=0,
                instance=predict,
                execution_interval_seconds=throughput_interval,
                needs=["observations.traffic"],
                observation_window_seconds=throughput_interval,
            )

            throughput = BuiltinThroughputPropose(cfg, self._scaling_state)
            self._builtins["throughput_propose"] = throughput
            self._orchestrator.register_internal(
                plugin_id=throughput.plugin_id,
                plugin_type="propose",
                priority=20,
                instance=throughput,
                execution_interval_seconds=throughput_interval,
                needs=["observations.traffic", "predictions"],
                requires_produced_fields=["predictions"],
                observation_window_seconds=throughput_interval,
            )

        if cfg.enable_load_scaling:
            load = BuiltinLoadPropose(cfg, self._scaling_state)
            self._builtins["load_propose"] = load
            needs = ["observations.fpm", "observations.workers"]
            observation_window = 0.0
            if cfg.optimization_target == "sla" and not cfg.enable_throughput_scaling:
                needs.append("observations.traffic")
                observation_window = load_interval
            self._orchestrator.register_internal(
                plugin_id=load.plugin_id,
                plugin_type="propose",
                priority=10,
                instance=load,
                execution_interval_seconds=load_interval,
                needs=needs,
                observation_window_seconds=observation_window,
            )

    # ------------------------------------------------------------------
    # Bootstrap API (delegates to orchestrator)
    # ------------------------------------------------------------------

    def install_regressions(
        self,
        *,
        prefill: Optional[Any] = None,
        decode: Optional[Any] = None,
        agg: Optional[Any] = None,
    ) -> None:
        self._scaling_state.install_regressions(prefill=prefill, decode=decode, agg=agg)
        self._orchestrator.install_regressions(prefill=prefill, decode=decode, agg=agg)

    def update_capabilities(self, capabilities: WorkerCapabilities) -> None:
        """Refresh worker capabilities after late runtime discovery."""
        self._capabilities = capabilities
        self._scaling_state.update_capabilities(capabilities)
        self._orchestrator.update_capabilities(capabilities)
        predict = self._builtins.get("load_predict")
        update_predict_caps = getattr(predict, "update_capabilities", None)
        if callable(update_predict_caps):
            update_predict_caps(capabilities)

    async def bootstrap_plugins(
        self, *, historical_traffic: Optional[Sequence[TrafficObservation]] = None
    ) -> None:
        if self._plugins_bootstrapped:
            return
        self._load_in_process_plugins_from_config()
        # Order matters: register static external plugins from config
        # **before** dispatching Bootstrap so they receive the same
        # ``warm_from_observations`` / ``Bootstrap`` RPC fan-out as the
        # builtin in-process plugins.  The previous order
        # (bootstrap → register externals) silently denied static
        # external plugins access to ``historical_traffic`` — they
        # registered into an already-bootstrapped registry and never
        # saw the Bootstrap pass.
        #
        # Gateway opens last so a plugin trying to register via the
        # network can't race the bootstrap fan-out (the gateway's
        # plugins are intentionally a runtime-only path; they
        # cannot receive ``historical_traffic`` because Bootstrap has
        # already moved past them — that's a deliberate scope split
        # and is the documented contract for dynamic registration).
        await self._wire_external_plugins_from_config()
        await self._orchestrator.bootstrap_plugins(
            historical_traffic=historical_traffic
        )
        await self._maybe_start_gateway()
        self._plugins_bootstrapped = True

    def _load_in_process_plugins_from_config(self) -> None:
        if self._in_process_plugins_loaded:
            return
        specs = list(self._config.plugin_registration.in_process_plugins)
        if not specs:
            self._in_process_plugins_loaded = True
            return
        from dynamo.planner.plugins.orchestrator.in_process_loader import (
            load_in_process_plugins,
        )

        load_in_process_plugins(self._orchestrator, specs)
        self._in_process_plugins_loaded = True

    async def _wire_external_plugins_from_config(self) -> None:
        """Register the static-config external plugin list.

        Idempotent at the orchestrator level (registry rejects
        duplicates), but the adapter only ever calls this once per
        bootstrap. Per-entry failures are logged but don't raise — a
        bad ConfigMap entry must NOT prevent the planner from running.
        """
        entries = list(self._config.scheduling.external_plugins)
        if not entries:
            return
        accepted, failures = await self._orchestrator.register_external_from_config(
            entries
        )
        if failures:
            log.warning(
                "external plugin bootstrap: accepted=%d failed=%d failures=%s",
                accepted,
                len(failures),
                failures,
            )
        else:
            log.info(
                "external plugin bootstrap: accepted=%d (all entries OK)",
                accepted,
            )

    async def _maybe_start_gateway(self) -> None:
        """Stand up the gRPC registration gateway if configured.

        Stores the running ``grpc.aio.Server`` on ``self._gateway_server``
        so ``shutdown()`` can stop it cleanly. Failure to start the
        gateway IS fatal — it usually means a port collision or bad
        bind address, which the operator needs to know immediately
        rather than discovering later when plugins fail to register.
        """
        gw_cfg = self._config.scheduling.gateway
        if not gw_cfg.enabled:
            return
        # Local import keeps the gateway module out of the cold-start
        # import chain for deployments that never enable it.
        from dynamo.planner.plugins.registry.gateway import start_gateway_server

        grpc_server, actual_listen = await start_gateway_server(
            self._orchestrator.registry,
            listen=gw_cfg.listen,
            allow_insecure=gw_cfg.allow_insecure,
        )
        self._gateway_server = grpc_server
        log.info("plugin registration gateway listening at %s", actual_listen)

    async def bootstrap_from_fpms(
        self,
        *,
        prefill_fpms: Optional[Sequence[Any]] = None,
        decode_fpms: Optional[Sequence[Any]] = None,
        agg_fpms: Optional[Sequence[Any]] = None,
        historical_traffic: Optional[Sequence[TrafficObservation]] = None,
    ) -> None:
        """One-shot pre-first-tick bootstrap from benchmark FPMs.

        Loads benchmark FPMs and warms predictor plugins through the plugin
        chain:

        1. ``install_regressions_from_fpms`` — in SLA mode, build the
           regression models from benchmark FPMs and install them on the
           orchestrator's shared store (easy mode skips — no regressions).
        2. ``bootstrap_plugins`` — warm ``BuiltinLoadPredict`` from
           ``historical_traffic`` and fan out Bootstrap RPC.

        Replay uses these two steps separately (regressions are installed
        once benchmark FPMs are generated; plugins are bootstrapped at
        adapter construction), so the regression-install half lives in its
        own synchronous method.
        """
        self.install_regressions_from_fpms(
            prefill_fpms=prefill_fpms, decode_fpms=decode_fpms, agg_fpms=agg_fpms
        )
        await self.bootstrap_plugins(historical_traffic=historical_traffic)

    def install_regressions_from_fpms(
        self,
        *,
        prefill_fpms: Optional[Sequence[Any]] = None,
        decode_fpms: Optional[Sequence[Any]] = None,
        agg_fpms: Optional[Sequence[Any]] = None,
    ) -> None:
        """Build regression models from benchmark FPMs and install them on
        the orchestrator's shared store. Synchronous; does NOT bootstrap
        plugins. No-op in easy mode (no regression models are used)."""
        if self._config.optimization_target != "sla":
            return
        self._scaling_state.load_benchmark_fpms(
            prefill_fpms=list(prefill_fpms) if prefill_fpms else None,
            decode_fpms=list(decode_fpms) if decode_fpms else None,
            agg_fpms=list(agg_fpms) if agg_fpms else None,
        )
        self.install_regressions(
            prefill=getattr(self._scaling_state, "_prefill_regression", None),
            decode=getattr(self._scaling_state, "_decode_regression", None),
            agg=getattr(self._scaling_state, "_agg_regression", None),
        )

    # ------------------------------------------------------------------
    # EngineProtocol
    # ------------------------------------------------------------------

    def initial_tick(self, start_s: float) -> ScheduledTick:
        """First scheduled tick under the scale_interval cadence model.

        Pipeline fires at ``start_s + scale_interval`` regardless of
        load / throughput interval configuration — those intervals live
        on individual plugin
        ``execution_interval_seconds`` values rather than on the
        pipeline cadence.
        """
        self._last_tick_s = start_s
        self._last_tick_monotonic = self._clock.monotonic()
        self._last_load_loop_monotonic = self._last_tick_monotonic
        self._last_throughput_loop_monotonic = self._last_tick_monotonic
        for plugin in self._orchestrator._registry.all_plugins():
            if plugin.is_builtin and plugin.last_call_at == float("-inf"):
                plugin.registered_at = self._last_tick_monotonic
        return self._compute_next_scheduled_tick()

    async def tick(
        self,
        scheduled_tick: ScheduledTick,
        tick_input: TickInput,
    ) -> PlannerEffects:
        # NOTE: we intentionally do NOT gate plugins via ``plugin.enabled``
        # on top of ``ScheduledTick.run_*_scaling`` flags. Each plugin's
        # own config-toggle check (``if not self._config.enable_load_scaling:
        # return accept``) is already a per-tick no-op when the corresponding
        # toggle is off; those config toggles remain authoritative for
        # plugin self-gating.

        # 0. Sync the shared clock to ``tick_input.now_s`` when we hold a
        #    manually-advanced clock (replay / test). Plugin scheduler
        #    ``is_due``, CircuitBreaker cooldown, and HOLD_LAST cache age
        #    all read ``self._clock.monotonic()`` — without this bump the
        #    plugin layer would see real wall-clock instead of replay
        #    trace time, and any plugin with ``execution_interval`` >>
        #    fast-forward duration would never re-fire after the first
        #    tick.  ``WallClock`` ignores this path (no-op) — production
        #    K8s already runs in real time.
        if isinstance(self._clock, VirtualClock):
            delta = tick_input.now_s - self._clock.monotonic()
            if delta > 0:
                self._clock.advance(delta)

        self._scaling_state.begin_tick()
        if tick_input.worker_counts is not None:
            self._scaling_state.observe_worker_counts(tick_input.worker_counts)

        # 1. Observe FPM into regressions before load proposal consumes
        #    the fitted perf models.
        is_easy = self._config.optimization_target != "sla"
        if (
            scheduled_tick.run_load_scaling
            and not is_easy
            and tick_input.fpm_observations is not None
        ):
            self._scaling_state.observe_fpm(tick_input.fpm_observations)

        # 2. Advance the scale_interval cadence pointer. Under the
        #    model there is one base interval; pipeline tick fires every
        #    ``scale_interval`` seconds and individual plugin cadences
        #    are handled inside the orchestrator by per-plugin
        #    ``execution_interval_seconds`` throttling.
        self._last_tick_s = tick_input.now_s
        # Monotonic twin — see ``__init__`` for why we keep both.  Read
        # *after* the optional VirtualClock sync above so replay sees
        # ``last_tick_monotonic == tick_input.now_s`` and production
        # wall-clock deployments see the
        # boot-relative value that plugin ``last_call_at`` is recorded in.
        self._last_tick_monotonic = self._clock.monotonic()
        if scheduled_tick.run_load_scaling:
            self._last_load_loop_monotonic = self._last_tick_monotonic
        if scheduled_tick.run_throughput_scaling:
            self._last_throughput_loop_monotonic = self._last_tick_monotonic

        # 3. Build PipelineContext + baseline and drive the orchestrator.
        ctx = self._tick_input_to_context(tick_input)
        baseline = self._baseline_from_worker_counts(tick_input.worker_counts)
        outcome = await self._orchestrator.tick(
            ctx,
            baseline,
            tick_now=scheduled_tick.at_monotonic_s,
        )

        # 4. Project PipelineOutcome onto PlannerEffects.
        scale_to = self._project_scale_to(
            outcome, tick_input.worker_counts or WorkerCounts()
        )

        # 5. Populate diagnostics from the shared scaling state. Consumed by
        #    the diagnostics recorder for HTML reports + Prometheus gauges.
        diagnostics = self._scaling_state.diagnostics()
        if (
            outcome.predict_outcome is not None
            and outcome.predict_outcome.prediction is not None
        ):
            p = outcome.predict_outcome.prediction
            diagnostics.predicted_num_req = p.predicted_num_req
            diagnostics.predicted_isl = p.predicted_isl
            diagnostics.predicted_osl = p.predicted_osl
            diagnostics.predicted_kv_hit_rate = p.predicted_kv_hit_rate
        elif (
            scheduled_tick.run_throughput_scaling
            and outcome.predict_outcome is not None
            and diagnostics.throughput_decision_reason is None
        ):
            reasons = outcome.predict_outcome.reasons
            if "predict_failed" in reasons:
                diagnostics.throughput_decision_reason = "predict_failed"
            elif "no_traffic_data" in reasons:
                diagnostics.throughput_decision_reason = "no_traffic_data"

        if scheduled_tick.run_load_scaling and not self._config.enable_load_scaling:
            diagnostics.load_decision_reason = "disabled"

        # Surface pipeline execute_action / short_circuit_reason /
        # audit_events.  Same data is emitted as Prometheus
        # ``tick_skip_reasons_total`` etc., but exposing it on
        # ``TickDiagnostics`` lets in-process consumers (replay
        # adapter, diagnostics recorder) distinguish ``apply`` from
        # ``skip_short_circuit`` / ``skip_no_targets`` /
        # ``skip_tick_timeout`` without scraping metrics.
        diagnostics.execute_action = outcome.execute_action
        diagnostics.short_circuit_reason = outcome.short_circuit_reason
        diagnostics.audit_events = list(outcome.audit_events)

        return PlannerEffects(
            scale_to=scale_to,
            next_tick=self._compute_next_scheduled_tick(),
            diagnostics=diagnostics,
        )

    async def observe(self, scheduled_tick: ScheduledTick, now_s: float) -> TickInput:
        """Run the in-process OBSERVE plugin for native planner execution.

        Replay and tests can continue to bypass observation collection by
        calling ``tick(..., tick_input)`` directly.
        """
        if self._observe_plugin is None:
            raise RuntimeError("No observe plugin configured")
        response = await self._observe_plugin.Observe(
            ObserveStageRequest(scheduled_tick=scheduled_tick, now_s=now_s)
        )
        return response.tick_input

    def _project_load_diagnostics(self, diagnostics: TickDiagnostics) -> None:
        """Read ``BuiltinLoadPropose._last_load_diagnostics`` and write
        to ``diagnostics.load_decision_reason*`` + ``estimated_*_ms``.

        Mirrors the builtin planner diagnostic surface:
        - mode=agg → aggregate ``load_decision_reason``
        - mode=disagg → per-component ``load_decision_reason_prefill`` /
          ``_decode`` (and also the aggregate, set to whichever side
          has a stronger signal; see ``_aggregate_disagg_load_reason``)
        - mode=prefill/decode → aggregate reason from the single side
        """
        propose = self._builtins.get("load_propose")
        if propose is None:
            return
        d = getattr(propose, "_last_load_diagnostics", None)
        if d is None:
            return

        mode = self._config.mode
        if mode == "agg":
            diagnostics.load_decision_reason = d.get("agg")
        elif mode == "disagg":
            diagnostics.load_decision_reason_prefill = d.get("prefill")
            diagnostics.load_decision_reason_decode = d.get("decode")
            # Aggregate: prefer scale_up > scale_down > no_change >
            # <skip reason>. Lets a single dashboard widget show "what
            # did the load path do" without dropping into the per-
            # component detail.
            diagnostics.load_decision_reason = self._aggregate_disagg_load_reason(
                d.get("prefill"), d.get("decode")
            )
        elif mode in ("prefill", "decode"):
            diagnostics.load_decision_reason = d.get(mode)

        diagnostics.estimated_ttft_ms = d.get("estimated_ttft_ms")
        diagnostics.estimated_itl_ms = d.get("estimated_itl_ms")

    def _project_throughput_diagnostics(self, diagnostics: TickDiagnostics) -> None:
        """Read ``BuiltinThroughputPropose._last_throughput_diagnostics``
        and write to ``diagnostics.throughput_decision_reason*``.

        Symmetric with ``_project_load_diagnostics``: throughput proposal
        records per-component reasons and this helper projects them onto
        the public ``TickDiagnostics`` fields.

        Mode mapping:
        - mode=agg → aggregate ``throughput_decision_reason``
        - mode=disagg → per-component
          ``throughput_decision_reason_prefill``/``_decode`` plus the
          aggregate (precedence via ``_aggregate_disagg_throughput_reason``)
        - mode=prefill/decode → aggregate from the single side
        """
        propose = self._builtins.get("throughput_propose")
        if propose is None:
            return
        d = getattr(propose, "_last_throughput_diagnostics", None)
        if d is None:
            return

        mode = self._config.mode
        if mode == "agg":
            diagnostics.throughput_decision_reason = d.get("agg")
        elif mode == "disagg":
            diagnostics.throughput_decision_reason_prefill = d.get("prefill")
            diagnostics.throughput_decision_reason_decode = d.get("decode")
            diagnostics.throughput_decision_reason = (
                self._aggregate_disagg_throughput_reason(
                    d.get("prefill"), d.get("decode")
                )
            )
        elif mode in ("prefill", "decode"):
            diagnostics.throughput_decision_reason = d.get(mode)

    @staticmethod
    def _aggregate_disagg_load_reason(
        prefill_reason: Optional[str], decode_reason: Optional[str]
    ) -> Optional[str]:
        """Collapse two per-component reasons to a single aggregate
        string. Precedence keeps "a side scaled"
        wins over "both stable", "stable with data" wins over "no
        data"."""
        priority = [
            "scale_up",
            "scale_down_capped_by_throughput",
            "scale_down",
            "no_change",
            "insufficient_data",
            "worker_count_mismatch",
            "scaling_in_progress",
            "no_fpm_data",
            "disabled",
        ]
        pairs = [r for r in (prefill_reason, decode_reason) if r is not None]
        if not pairs:
            return None
        for p in priority:
            if p in pairs:
                return p
        return pairs[0]

    @staticmethod
    def _aggregate_disagg_throughput_reason(
        prefill_reason: Optional[str], decode_reason: Optional[str]
    ) -> Optional[str]:
        """Collapse two per-component throughput reasons. Vocabulary
        differs from load reasons (no scale_up/down enums on this
        side); ranking keeps "stronger action wins":
        ``scale`` > ``set_lower_bound`` > skip reasons."""
        priority = [
            "scale",
            "set_lower_bound",
            "model_not_ready",
            "no_traffic_data",
            "predict_failed",
            "disabled",
        ]
        pairs = [r for r in (prefill_reason, decode_reason) if r is not None]
        if not pairs:
            return None
        for p in priority:
            if p in pairs:
                return p
        return pairs[0]

    async def shutdown(self) -> None:
        # Stop the gateway BEFORE unregistering plugins so no new
        # external Register / Heartbeat call can race the teardown.
        if self._gateway_server is not None:
            try:
                await self._gateway_server.stop(grace=0.5)
            except Exception as exc:
                # Don't let a gateway shutdown error mask the real
                # planner shutdown work that follows.
                log.warning(
                    "gateway server stop raised %s: %s — continuing shutdown",
                    type(exc).__name__,
                    exc,
                )
            self._gateway_server = None
        await self._orchestrator.shutdown()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_next_scheduled_tick(self) -> ScheduledTick:
        """Next pipeline tick under the scale_interval cadence model.

        Pipeline fires at ``self._last_tick_s + scale_interval`` —
        a single base cadence with no dual-cadence merge. Per-plugin
        ``execution_interval_seconds`` throttling (in
        ``PluginScheduler._is_due``) handles which plugins actually
        fire each tick.

        Observation collection (``need_traffic_metrics``,
        ``traffic_metrics_duration_s``) is gated on whether any
        registered plugin both lists ``observations.traffic`` in its
        ``needs`` AND would be due at the next tick.  This recovers
        the lazy-pull cost profile (one Prometheus query per
        ``throughput_adjustment_interval_seconds`` in mixed mode)
        without leaking the cadence-type concept into the
        ScheduledTick API — plugins only see the window they
        themselves declared via ``observation_window_seconds``.
        """
        at_s = self._last_tick_s + self._scale_interval
        # Due-check operates in the monotonic domain that
        # ``RegisteredPlugin.last_call_at`` lives in — NOT wall-epoch.
        # ``at_s`` (wall-epoch) is for ``ScheduledTick.at_s`` only.
        at_monotonic = self._last_tick_monotonic + self._scale_interval
        load_loop_due = self._interval_due(
            self._last_load_loop_monotonic,
            float(self._config.load_adjustment_interval_seconds),
            at_monotonic,
        )
        throughput_loop_due = (
            self._config.enable_throughput_scaling
            and self._interval_due(
                self._last_throughput_loop_monotonic,
                float(self._config.throughput_adjustment_interval_seconds),
                at_monotonic,
            )
        )

        # Lazy traffic pull: only when some currently-registered,
        # currently-due plugin actually consumes
        # ``observations.traffic``.  Without any such plugin the
        # pipeline still ticks (e.g. for FPM-driven load decisions or
        # worker-state-only constrain logic), it just skips the
        # Prometheus query.
        #
        # ``needs`` are dot-paths into ``PipelineContext`` per the proto
        # contract.  Match the parent path ``"observations.traffic"``
        # AND any sub-path ``"observations.traffic.<field>"`` — both
        # require the traffic observation to be present, since the
        # sub-path can only resolve if its parent does.  The trailing
        # ``.`` in the prefix is load-bearing: it stops false-positives
        # on a sibling like ``"observations.traffic_legacy"`` (no such
        # field today but defensive against future schema additions).
        def due_consumers(path: str):
            prefix = f"{path}."
            return [
                p
                for p in self._orchestrator._registry.all_plugins()
                if any(n == path or n.startswith(prefix) for n in p.needs)
                and self._orchestrator._scheduler._is_due(p, at_monotonic)
            ]

        traffic_consumers_due = due_consumers("observations.traffic")
        if traffic_consumers_due:
            need_traffic = True
            use_full_traffic = any(
                not (
                    p.plugin_id == "builtin_load_propose"
                    and not self._config.enable_throughput_scaling
                )
                for p in traffic_consumers_due
            )
            # Aggregation window: max declared
            # ``observation_window_seconds`` across due consumers.
            # Declared 0.0 means "scale_interval freshness" — i.e. the
            # plugin doesn't need a longer window, so falls back to
            # the base cadence here.
            declared = [
                p.observation_window_seconds
                for p in traffic_consumers_due
                if p.observation_window_seconds > 0
            ]
            traffic_duration_s = (
                max(declared) if declared else float(self._scale_interval)
            )
        else:
            need_traffic = False
            use_full_traffic = False
            traffic_duration_s = 0.0

        fpm_consumers_due = due_consumers("observations.fpm")
        internal_fpm_due = self._config.optimization_target == "sla" and load_loop_due

        # ``run_load_scaling`` / ``run_throughput_scaling`` flags are
        # preserved on ScheduledTick for back-compat observability: they
        # mean the corresponding legacy builtin loop is due on this tick,
        # not merely that the plugin pipeline fired.  Input collection is
        # described by the separate need_* fields below.
        return ScheduledTick(
            at_s=at_s,
            at_monotonic_s=at_monotonic,
            run_load_scaling=load_loop_due,
            run_throughput_scaling=throughput_loop_due,
            need_worker_states=True,
            need_worker_fpm=bool(fpm_consumers_due) or internal_fpm_due,
            need_traffic_metrics=need_traffic,
            use_full_traffic_metrics=use_full_traffic,
            traffic_metrics_duration_s=traffic_duration_s,
        )

    @staticmethod
    def _interval_due(last_s: float, interval_s: float, at_s: float) -> bool:
        return at_s - last_s >= interval_s - 1e-9

    def _observe_fpm(self, obs: FpmObservations) -> None:
        """Mirror ``PlannerScalingState._observe_fpm`` — feeds observations
        into the orchestrator-owned regression models.

        ``obs.prefill`` / ``obs.decode`` are already
        ``dict[(worker_id, dp_rank) -> ForwardPassMetrics]`` — exactly the
        shape ``PlannerEnginePerfModel.add_observations`` consumes, so we
        hand the whole dict over in one call. The regression model only
        exposes ``add_observations`` (plural, dict-based); there is no
        singular ``add_observation`` on this class.
        """
        mode = self._config.mode
        if mode == "agg":
            if obs.decode:
                agg = self._orchestrator.get_regression("agg")
                if agg is not None:
                    agg.add_observations(obs.decode)
            return
        if obs.prefill:
            p_reg = self._orchestrator.get_regression("prefill")
            if p_reg is not None:
                p_reg.add_observations(obs.prefill)
        if obs.decode:
            d_reg = self._orchestrator.get_regression("decode")
            if d_reg is not None:
                d_reg.add_observations(obs.decode)

    def _tick_input_to_context(self, ti: TickInput) -> PipelineContext:
        traffic = None
        if ti.traffic is not None:
            traffic = TrafficMetrics(
                duration_s=ti.traffic.duration_s,
                num_req=ti.traffic.num_req,
                isl=ti.traffic.isl,
                osl=ti.traffic.osl,
                kv_hit_rate=ti.traffic.kv_hit_rate,
                accept_length=ti.traffic.accept_length,
            )
        workers = None
        if ti.worker_counts is not None:
            workers = WorkerState(
                ready_prefill=ti.worker_counts.ready_num_prefill,
                ready_decode=ti.worker_counts.ready_num_decode,
                expected_prefill=ti.worker_counts.expected_num_prefill,
                expected_decode=ti.worker_counts.expected_num_decode,
                prefill_scaling_in_progress=ti.worker_counts.prefill_scaling_in_progress,
                decode_scaling_in_progress=ti.worker_counts.decode_scaling_in_progress,
            )
        # FPM observations: encode per-engine ``ForwardPassMetrics`` to
        # msgpack bytes (the wire format the proto README + ``FpmData``
        # docstring already prescribe), keyed by "<worker_id>/<dp_rank>"
        # so cross-language plugins can decode without knowing about the
        # tuple key. Without this, external load-based plugins that
        # declare ``needs=["observations.fpm"]`` would always see None
        # and could not implement load-based decisions through
        # the public PipelineContext API.
        fpm = self._encode_fpm(ti.fpm_observations)
        return PipelineContext(
            request_id=f"tick-{ti.now_s}",
            decision_id=f"d-{ti.now_s}",
            observations=ObservationData(traffic=traffic, fpm=fpm, workers=workers),
        )

    @staticmethod
    def _encode_fpm(obs: Optional[FpmObservations]) -> Optional[FpmData]:
        """Encode ``FpmObservations`` for transport over the public
        ``PipelineContext.observations.fpm`` channel.

        Encoding contract (matches the proto README in
        ``plugins/proto/v1/README.md``):
        - per-engine map key = ``f"{worker_id}/{dp_rank}"`` (flat str
          since proto3 ``map<string, bytes>`` can't carry a tuple key)
        - per-engine map value = msgpack-encoded ``ForwardPassMetrics``
          via the canonical ``dynamo.common.forward_pass_metrics.encode``
          helper (shared module-level encoder) so cross-language plugins
          decode with any standard msgpack library and the wire format
          stays in lock-step with the rest of dynamo's FPM serialization.

        Returns None when ``obs`` is None (no FPM this tick) or when
        both prefill+decode submaps are empty.
        """
        if obs is None:
            return None
        if not obs.prefill and not obs.decode:
            return None
        prefill_engines: dict[str, bytes] = {}
        decode_engines: dict[str, bytes] = {}
        if obs.prefill:
            for (worker_id, dp_rank), fpm_obs in obs.prefill.items():
                prefill_engines[f"{worker_id}/{dp_rank}"] = _encode_fpm_record(fpm_obs)
        if obs.decode:
            for (worker_id, dp_rank), fpm_obs in obs.decode.items():
                decode_engines[f"{worker_id}/{dp_rank}"] = _encode_fpm_record(fpm_obs)
        return FpmData(
            prefill_engines=prefill_engines,
            decode_engines=decode_engines,
        )

    @staticmethod
    def _baseline_from_worker_counts(
        counts: Optional[WorkerCounts],
    ) -> dict[ComponentKey, int]:
        """Seed the PROPOSE-stage baseline with current worker counts so
        the merge chain has a reference point. Without this, when all
        PROPOSE plugins return Accept (e.g. FPM worker-count mismatch
        in load_propose), ``type_aware_merge`` produces empty targets
        → RECONCILE sees empty → CONSTRAIN's ``AT_LEAST(min_endpoint)``
        dominates with ``baseline.get(key, 0) == 0`` → result is
        ``min_endpoint`` instead of current.

        Projecting that back through ``_project_scale_to``'s no-change
        detection (``num_p == current_p``) would incorrectly report a
        scale-down; passing the worker counts as baseline lets the
        merge preserve the current value end-to-end so the projection
        returns ``None`` (matching the planner's scale_to semantic for the
        "load plugin had no opinion" case).
        """
        if counts is None:
            return {}
        out: dict[ComponentKey, int] = {}
        if counts.ready_num_prefill is not None:
            out[ComponentKey(sub_component_type="prefill")] = counts.ready_num_prefill
        if counts.ready_num_decode is not None:
            out[ComponentKey(sub_component_type="decode")] = counts.ready_num_decode
        return out

    def _project_scale_to(self, outcome, worker_counts: WorkerCounts):
        """Project the pipeline outcome onto ``PlannerEffects.scale_to``
        with planner "no change -> None" detection."""
        if outcome.execute_action != "apply" or outcome.final_proposal is None:
            return None

        by_comp = {
            t.sub_component_type: t.replicas for t in outcome.final_proposal.targets
        }
        num_p = by_comp.get("prefill")
        num_d = by_comp.get("decode")

        current_p = worker_counts.ready_num_prefill
        current_d = worker_counts.ready_num_decode

        p_unchanged = (num_p is None) or (num_p == current_p)
        d_unchanged = (num_d is None) or (num_d == current_d)
        if p_unchanged and d_unchanged:
            return None

        num_p, num_d = self._apply_final_budget(num_p, num_d, worker_counts)

        p_unchanged = (num_p is None) or (num_p == current_p)
        d_unchanged = (num_d is None) or (num_d == current_d)
        if p_unchanged and d_unchanged:
            return None

        return ScalingDecision(num_prefill=num_p, num_decode=num_d)

    def _apply_final_budget(
        self,
        num_p: Optional[int],
        num_d: Optional[int],
        worker_counts: WorkerCounts,
    ) -> tuple[Optional[int], Optional[int]]:
        min_endpoint = self._config.min_endpoint
        min_gpus = self._config.min_gpu_budget
        max_gpus = self._config.max_gpu_budget
        mode = self._config.mode

        def clamp_single(component: str, replicas: Optional[int]) -> Optional[int]:
            if replicas is None:
                return None
            caps = (
                self._capabilities.prefill
                if component == "prefill"
                else self._capabilities.decode
            )
            gpu = caps.num_gpu if caps else None
            if gpu is None:
                return max(replicas, min_endpoint)
            return proportional_clamp_single(
                max(replicas, min_endpoint),
                gpu,
                min_gpus,
                max_gpus,
                min_endpoint,
            )

        if mode == "prefill":
            return clamp_single("prefill", num_p), num_d
        if mode in ("decode", "agg"):
            return num_p, clamp_single("decode", num_d)
        if mode != "disagg":
            return num_p, num_d

        proposed_p = num_p is not None
        proposed_d = num_d is not None
        base_p = num_p if proposed_p else worker_counts.ready_num_prefill
        base_d = num_d if proposed_d else worker_counts.ready_num_decode
        if base_p is None or base_d is None:
            return clamp_single("prefill", num_p), clamp_single("decode", num_d)

        p_caps = self._capabilities.prefill
        d_caps = self._capabilities.decode
        p_gpu = p_caps.num_gpu if p_caps else None
        d_gpu = d_caps.num_gpu if d_caps else None
        if p_gpu is None or d_gpu is None:
            return (
                max(base_p, min_endpoint) if proposed_p else None,
                max(base_d, min_endpoint) if proposed_d else None,
            )

        clamped_p, clamped_d = proportional_clamp_pair(
            max(base_p, min_endpoint),
            max(base_d, min_endpoint),
            p_gpu,
            d_gpu,
            min_gpus,
            max_gpus,
            min_endpoint,
        )
        return clamped_p if proposed_p else None, clamped_d if proposed_d else None


__all__ = ["OrchestratorEngineAdapter"]
