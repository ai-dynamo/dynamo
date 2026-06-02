# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``OrchestratorEngineAdapter`` — production ``EngineProtocol`` adapter
for the plugin chain.

Wraps ``LocalPlannerOrchestrator`` behind the same ``initial_tick`` /
``tick`` / ``shutdown`` interface that the legacy ``_PSMEngineAdapter``
exposes. ``NativePlannerBase`` selects between the two via
``PlannerConfig.scheduling.use_orchestrator``.

Architecture invariant: PipelineContext is the only input channel
--------------------------------------------------------------------
All plugins — both in-process builtins (follow-up PR) and external
gRPC plugins — receive their per-tick inputs through
``PipelineContext.observations`` exclusively. There is **no**
``prime_tick(...)`` side-channel, ``self._last_fpm``-style stash,
or any other path that delivers observation data to a plugin instance
outside of the stage RPC.

This invariant ensures:
  * Plugin API is uniform across in-process and over-wire transports.
  * Adding a new observation field requires touching one schema
    (``ObservationData``), not two delivery paths.
  * Builtin plugins (follow-up PR) and external plugins receive
    byte-identical input, so dual-path parity tests are meaningful.

Internal responsibilities
-------------------------

1. **Tick lifecycle cadence tracking**:
   Owns ``_next_load_s`` / ``_next_throughput_s`` state and advances
   them at tick boundaries the same way
   ``PlannerStateMachine._next_scheduled_tick`` does, so the
   ``next_tick`` field in ``PlannerEffects`` matches PSM's legacy
   path bit-for-bit.
2. **TickInput → PipelineContext bridge**:
   Extracts ``traffic`` into ``TrafficMetrics`` and ``worker_counts``
   into ``WorkerState`` on ``ObservationData``. FPM ingestion to
   ``ObservationData.fpm`` lands in a follow-up PR (single
   msgspec/msgpack encoding; see plan).
3. **FPM regression observation**:
   Before the orchestrator tick, feeds FPM into the orchestrator-owned
   regression models (mirrors PSM's ``_observe_fpm``). This is a
   planner-internal regression-fit path, distinct from delivering FPM
   to plugins.
4. **PipelineOutcome → PlannerEffects projection**:
   Reads the orchestrator's ``final_proposal.targets``, detects "no
   change" against ``worker_counts``, and projects to
   ``PlannerEffects.scale_to``. ``diagnostics`` is empty — numeric
   fields moved to Prometheus.

Bootstrap API
-------------

``install_regressions`` + ``bootstrap_plugins`` mirror
``LocalPlannerOrchestrator``'s equivalents but are exposed on the
adapter so callers (mode subclasses) have a single entry point.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional, Sequence

if TYPE_CHECKING:
    import grpc.aio

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
from dynamo.planner.plugins.clock import Clock, VirtualClock, WallClock
from dynamo.planner.plugins.merge.types import ComponentKey
from dynamo.planner.plugins.orchestrator.orchestrator import LocalPlannerOrchestrator
from dynamo.planner.plugins.registry.auth import AllowUnauthenticatedAuth
from dynamo.planner.plugins.registry.config import build_auth_validator
from dynamo.planner.plugins.registry.circuit_breaker import CircuitBreaker
from dynamo.planner.plugins.registry.server import PluginRegistryServer
from dynamo.planner.plugins.scheduler import PluginScheduler
from dynamo.planner.plugins.transport.config import (
    TransportConfig,
    make_transport_for_endpoint,
)
from dynamo.planner.plugins.types import (
    ObservationData,
    PipelineContext,
    TrafficMetrics,
    WorkerState,
)

log = logging.getLogger(__name__)


class OrchestratorEngineAdapter:
    """``EngineProtocol``-compatible wrapper around the 5-builtin chain.

    Lifecycle:

    1. ``OrchestratorEngineAdapter(config, capabilities)`` — builds
       orchestrator + 5 plugins + registers them. No regression models
       installed yet.
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
        clock: Optional[Clock] = None,
    ) -> None:
        self._config = config
        self._capabilities = capabilities
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
        # which plugins actually fire each tick.  See design doc §4 and
        # ``test_decision_level_parity`` for how this matches PSM's
        # observable scaling decisions while collapsing the legacy
        # dual-cadence book-keeping into one base interval.
        self._scale_interval: float = float(
            config.scheduling.scale_interval_seconds
        )
        self._last_tick_s: float = 0.0

        # Legacy cadence fields preserved as a compatibility shim for
        # any existing test that still reads them.  Not consulted by the
        # scale_interval scheduling logic — pipeline tick selection runs
        # entirely off ``self._last_tick_s + self._scale_interval``.
        # Removed entirely once the PSM-parity test surface is rewritten
        # to its decision-level form (see same design doc §11).
        self._next_load_s: float = float("inf")
        self._next_throughput_s: float = float("inf")

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

        # Registration gateway lifecycle: populated lazily by
        # ``_maybe_start_gateway`` if config opts in; consumed by
        # ``shutdown``.  Default ``None`` keeps the typical (gateway
        # disabled) deployment path zero-cost.
        self._gateway_server: Optional[grpc.aio.Server] = None

        # Builtin plugins land in a follow-up PR. PR #1 ships only the
        # infrastructure (orchestrator + pipeline + transport + registry
        # + external-plugin wiring via both static config and the gRPC
        # registration gateway); the orchestrator path will produce
        # empty proposals on every tick until the follow-up adds builtin
        # load/throughput/reconcile/budget plugins, OR external plugins
        # fill the chain via either registration path.
        self._builtins: dict = {}
        self._plugin_ids: dict = {}

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
        self._orchestrator.install_regressions(
            prefill=prefill, decode=decode, agg=agg
        )

    async def bootstrap_plugins(
        self, *, historical_traffic: Optional[Sequence[TrafficObservation]] = None
    ) -> None:
        await self._orchestrator.bootstrap_plugins(historical_traffic=historical_traffic)
        await self._wire_external_plugins_from_config()
        await self._maybe_start_gateway()

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
            self._orchestrator.registry, listen=gw_cfg.listen
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

        Mirrors PSM's ``load_benchmark_fpms`` + ``warm_load_predictors``
        but through the plugin chain:

        1. In SLA mode, spin up a throwaway ``PlannerStateMachine`` that
           builds the regression model instances from benchmark FPMs the
           same way PSM does internally. Hand those instances to the
           orchestrator's shared store via ``install_regressions``.
           (Easy mode skips this — no regression models are used.)
        2. Call ``bootstrap_plugins`` to warm ``BuiltinLoadPredictor``
           from ``historical_traffic`` and fan out Bootstrap RPC.

        Using PSM as the regression factory is a shortcut — a future
        cleanup can extract regression-construction from PSM into a
        standalone helper so this can drop the throwaway instance.
        """
        # Import locally to avoid pulling PSM into module-level imports
        # (the adapter's own tick path shouldn't know about PSM).
        from dynamo.planner.core.state_machine import PlannerStateMachine

        if self._config.optimization_target == "sla":
            throwaway = PlannerStateMachine(self._config, self._capabilities)
            throwaway.load_benchmark_fpms(
                prefill_fpms=list(prefill_fpms) if prefill_fpms else None,
                decode_fpms=list(decode_fpms) if decode_fpms else None,
                agg_fpms=list(agg_fpms) if agg_fpms else None,
            )
            self.install_regressions(
                prefill=getattr(throwaway, "_prefill_regression", None),
                decode=getattr(throwaway, "_decode_regression", None),
                agg=getattr(throwaway, "_agg_regression", None),
            )

        await self.bootstrap_plugins(historical_traffic=historical_traffic)

    # ------------------------------------------------------------------
    # EngineProtocol
    # ------------------------------------------------------------------

    def initial_tick(self, start_s: float) -> ScheduledTick:
        """First scheduled tick under the scale_interval cadence model.

        Pipeline fires at ``start_s + scale_interval`` regardless of
        the legacy load / throughput interval configuration — those
        intervals now live on individual plugin
        ``execution_interval_seconds`` values rather than on the
        pipeline cadence.

        Legacy ``_next_load_s`` / ``_next_throughput_s`` still set for
        any compatibility code still reading them (those reads are
        scheduled for removal once decision-level parity is in place).
        """
        self._last_tick_s = start_s
        self._next_load_s = start_s + self._config.load_adjustment_interval_seconds
        if self._config.enable_throughput_scaling:
            self._next_throughput_s = (
                start_s + self._config.throughput_adjustment_interval_seconds
            )
        return self._compute_next_scheduled_tick()

    async def tick(
        self,
        scheduled_tick: ScheduledTick,
        tick_input: TickInput,
    ) -> PlannerEffects:
        # NOTE: we intentionally do NOT gate plugins via ``plugin.enabled``
        # per scheduled_tick flag. The plugins' own config-toggle checks
        # (``if not self._config.enable_load_scaling: return accept``) are
        # already per-tick no-ops when the corresponding toggle is off;
        # adding a secondary gate only introduces divergence risk. See
        # test_engine_adapter::test_g3_parity_via_adapter — equivalence
        # with PSM requires leaving the always-on plugins enabled.

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

        # 1. Observe FPM into regressions (mirror PSM ``_observe_fpm``
        #    before ``_advance_load``).
        is_easy = self._config.optimization_target != "sla"
        if (
            scheduled_tick.run_load_scaling
            and not is_easy
            and tick_input.fpm_observations is not None
        ):
            self._observe_fpm(tick_input.fpm_observations)

        # 2. Advance the scale_interval cadence pointer.  Under the new
        #    model there is one base interval; pipeline tick fires every
        #    ``scale_interval`` seconds and individual plugin cadences
        #    are handled inside the orchestrator by per-plugin
        #    ``execution_interval_seconds`` throttling.  The legacy
        #    ``_next_load_s`` / ``_next_throughput_s`` are kept current
        #    only for shim compatibility — they no longer drive next-
        #    tick selection.
        self._last_tick_s = tick_input.now_s
        self._next_load_s = (
            tick_input.now_s + self._config.load_adjustment_interval_seconds
        )
        if self._config.enable_throughput_scaling:
            self._next_throughput_s = (
                tick_input.now_s + self._config.throughput_adjustment_interval_seconds
            )

        # 3. Build PipelineContext + baseline and drive the orchestrator.
        ctx = self._tick_input_to_context(tick_input)
        baseline = self._baseline_from_worker_counts(tick_input.worker_counts)
        outcome = await self._orchestrator.tick(ctx, baseline)

        # 4. Project PipelineOutcome onto PlannerEffects.
        scale_to = self._project_scale_to(
            outcome, tick_input.worker_counts or WorkerCounts()
        )

        # 5. Populate prediction fields on diagnostics. Consumed by the
        #    diagnostics recorder for HTML reports + Prometheus
        #    ``predicted_*`` gauges (mirrors PSM's behaviour).
        diagnostics = TickDiagnostics()
        if (
            outcome.predict_outcome is not None
            and outcome.predict_outcome.prediction is not None
        ):
            p = outcome.predict_outcome.prediction
            diagnostics.predicted_num_req = p.predicted_num_req
            diagnostics.predicted_isl = p.predicted_isl
            diagnostics.predicted_osl = p.predicted_osl

        # Surface builtin_load_propose's per-tick reason + estimates
        # onto ``TickDiagnostics`` so orchestrator-path logs + Prometheus
        # enum match the semantic detail PSM path has carried since v0.
        # Plugin stores last decision on itself; we read
        # ``_last_load_diagnostics`` and project to the appropriate
        # legacy field (agg → aggregate ``load_decision_reason``;
        # disagg/prefill/decode → per-component fields).
        self._project_load_diagnostics(diagnostics)

        # Same shape for builtin_throughput_propose. Without this
        # projection, ``throughput_decision_reason`` stays None on the
        # orchestrator path while PSM path populated it from
        # ``_diag_throughput_reason`` — making it impossible to tell
        # accept-with-decision from accept-skipped on dashboards.
        self._project_throughput_diagnostics(diagnostics)

        return PlannerEffects(
            scale_to=scale_to,
            next_tick=self._compute_next_scheduled_tick(),
            diagnostics=diagnostics,
        )

    def _project_load_diagnostics(self, diagnostics: TickDiagnostics) -> None:
        """Read ``BuiltinLoadPropose._last_load_diagnostics`` and write
        to ``diagnostics.load_decision_reason*`` + ``estimated_*_ms``.

        Mirrors PSM's diagnostic surface:
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

    def _project_throughput_diagnostics(
        self, diagnostics: TickDiagnostics
    ) -> None:
        """Read ``BuiltinThroughputPropose._last_throughput_diagnostics``
        and write to ``diagnostics.throughput_decision_reason*``.

        Symmetric with ``_project_load_diagnostics``: PSM path populates
        these fields from ``_diag_throughput_reason*``; this helper
        keeps the orchestrator path's surface byte-equivalent at the
        observability layer (decision outputs are already
        byte-identical, locked by ``test_dual_path_parity.py``).

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
                self._aggregate_disagg_throughput_reason(d.get("prefill"), d.get("decode"))
            )
        elif mode in ("prefill", "decode"):
            diagnostics.throughput_decision_reason = d.get(mode)

    @staticmethod
    def _aggregate_disagg_load_reason(
        prefill_reason: Optional[str], decode_reason: Optional[str]
    ) -> Optional[str]:
        """Collapse two per-component reasons to a single aggregate
        string.  Precedence mirrors PSM's convention: "a side scaled"
        wins over "both stable", "stable with data" wins over "no
        data"."""
        priority = [
            "scale_up",
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
        side); ranking mirrors PSM convention "stronger action wins":
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

    def _set_enabled(self, slot: str, enabled: bool) -> None:
        reg = self._orchestrator._registry.get_plugin(self._plugin_ids[slot])
        if reg is not None:
            reg.enabled = enabled

    def _compute_next_scheduled_tick(self) -> ScheduledTick:
        """Next pipeline tick under the scale_interval cadence model.

        Pipeline fires at ``self._last_tick_s + scale_interval`` —
        a single base cadence, no more dual ``_next_load_s`` /
        ``_next_throughput_s`` merging.  Per-plugin
        ``execution_interval_seconds`` throttling (in
        ``PluginScheduler._is_due``) handles which plugins actually
        fire each tick.

        Observation collection (``need_traffic_metrics``,
        ``traffic_metrics_duration_s``) is gated on whether any
        registered plugin both lists ``observations.traffic`` in its
        ``needs`` AND would be due at the next tick.  This recovers
        PSM's lazy-pull cost profile (one Prometheus query per
        ``throughput_adjustment_interval_seconds`` in mixed mode)
        without leaking the cadence-type concept into the
        ScheduledTick API — plugins only see the window they
        themselves declared via ``observation_window_seconds``.
        """
        at_s = self._last_tick_s + self._scale_interval

        # Lazy traffic pull: only when some currently-registered,
        # currently-due plugin actually consumes
        # ``observations.traffic``.  Without any such plugin the
        # pipeline still ticks (e.g. for FPM-driven load decisions or
        # worker-state-only constrain logic), it just skips the
        # Prometheus query.
        traffic_consumers_due = [
            p
            for p in self._orchestrator._registry.all_plugins()
            if "observations.traffic" in p.needs
            and self._orchestrator._scheduler._is_due(p, at_s)
        ]
        if traffic_consumers_due:
            need_traffic = True
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
            traffic_duration_s = 0.0

        # ``run_load_scaling`` / ``run_throughput_scaling`` flags are
        # preserved on ScheduledTick for back-compat with PSM-path
        # tests and the diagnostics-projection methods below.  Under
        # scale_interval both are always True — every pipeline tick is
        # treated as an opportunity for either type of plugin to fire
        # (subject to its own throttle).
        return ScheduledTick(
            at_s=at_s,
            run_load_scaling=True,
            run_throughput_scaling=True,
            need_worker_states=True,
            need_worker_fpm=True,
            need_traffic_metrics=need_traffic,
            traffic_metrics_duration_s=traffic_duration_s,
        )

    def _observe_fpm(self, obs: FpmObservations) -> None:
        """Mirror ``PlannerStateMachine._observe_fpm`` — feeds observations
        into the orchestrator-owned regression models."""
        mode = self._config.mode
        if mode == "agg":
            if obs.decode:
                agg = self._orchestrator.get_regression("agg")
                if agg is not None:
                    for fpm in obs.decode.values():
                        agg.add_observation(fpm)
            return
        if obs.prefill:
            p_reg = self._orchestrator.get_regression("prefill")
            if p_reg is not None:
                for fpm in obs.prefill.values():
                    p_reg.add_observation(fpm)
        if obs.decode:
            d_reg = self._orchestrator.get_regression("decode")
            if d_reg is not None:
                for fpm in obs.decode.values():
                    d_reg.add_observation(fpm)

    def _tick_input_to_context(self, ti: TickInput) -> PipelineContext:
        traffic = None
        if ti.traffic is not None:
            traffic = TrafficMetrics(
                duration_s=ti.traffic.duration_s,
                num_req=ti.traffic.num_req,
                isl=ti.traffic.isl,
                osl=ti.traffic.osl,
            )
        workers = None
        if ti.worker_counts is not None:
            workers = WorkerState(
                ready_prefill=ti.worker_counts.ready_num_prefill,
                ready_decode=ti.worker_counts.ready_num_decode,
                expected_prefill=ti.worker_counts.expected_num_prefill,
                expected_decode=ti.worker_counts.expected_num_decode,
            )
        return PipelineContext(
            request_id=f"tick-{ti.now_s}",
            decision_id=f"d-{ti.now_s}",
            observations=ObservationData(traffic=traffic, workers=workers),
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
        returns ``None`` (matching PSM's scale_to semantic for the
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

    @staticmethod
    def _project_scale_to(outcome, worker_counts: WorkerCounts):
        """Project the pipeline outcome onto ``PlannerEffects.scale_to``
        with PSM-equivalent "no change → None" detection."""
        if outcome.execute_action != "apply" or outcome.final_proposal is None:
            return None

        by_comp = {
            t.sub_component_type: t.replicas
            for t in outcome.final_proposal.targets
        }
        num_p = by_comp.get("prefill")
        num_d = by_comp.get("decode")

        current_p = worker_counts.ready_num_prefill
        current_d = worker_counts.ready_num_decode

        p_unchanged = (num_p is None) or (num_p == current_p)
        d_unchanged = (num_d is None) or (num_d == current_d)
        if p_unchanged and d_unchanged:
            return None

        return ScalingDecision(num_prefill=num_p, num_decode=num_d)


__all__ = ["OrchestratorEngineAdapter"]
