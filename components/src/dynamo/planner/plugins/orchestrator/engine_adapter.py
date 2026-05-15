# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``OrchestratorEngineAdapter`` — production ``EngineProtocol`` adapter
for the plugin chain.

Wraps ``LocalPlannerOrchestrator`` + the 5 real builtin plugins
(BuiltinLoadPredictor / BuiltinThroughputPropose / BuiltinLoadPropose /
BuiltinReconcile / BuiltinBudgetConstrain) behind the same
``initial_tick`` / ``tick`` / ``shutdown`` interface that the legacy
``_PSMEngineAdapter`` exposes. ``NativePlannerBase`` selects between
the two via ``PlannerConfig.scheduling.use_orchestrator``.

Internal responsibilities
-------------------------

1. **Plugin construction + registration** (done once in ``__init__``):
   Builds the 5 builtin plugin instances and registers them in the
   orchestrator's in-process registry. Keeps direct references to the
   plugins' ``RegisteredPlugin`` entries so ``enabled`` can be toggled
   per scheduled_tick flag (mirrors PSM's per-tick "run_load /
   run_throughput" gating).
2. **Tick lifecycle cadence tracking**:
   Owns ``_next_load_s`` / ``_next_throughput_s`` state and advances
   them at tick boundaries the same way
   ``PlannerStateMachine._next_scheduled_tick`` does, so the
   ``next_tick`` field in ``PlannerEffects`` matches PSM's legacy
   path bit-for-bit.
3. **TickInput → PipelineContext bridge**:
   Extracts ``traffic`` into Pydantic ``TrafficMetrics``; FPM
   observations stay as Python objects on the side channel (no wire-
   format encoding in v1).
4. **FPM regression observation + plugin prime**:
   Before the orchestrator tick, feeds FPM into the orchestrator-owned
   regression models (mirrors PSM's ``_observe_fpm``) and primes
   ``BuiltinLoadPropose`` / ``BuiltinBudgetConstrain`` with the
   per-tick FPM + worker counts.
5. **PipelineOutcome → PlannerEffects projection**:
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
from typing import Any, Optional, Sequence

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
from dynamo.planner.plugins.clock import WallClock
from dynamo.planner.plugins.merge.types import ComponentKey
from dynamo.planner.plugins.orchestrator.orchestrator import LocalPlannerOrchestrator
from dynamo.planner.plugins.registry.auth import AllowUnauthenticatedAuth
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
)

log = logging.getLogger(__name__)

# Matches ``PlannerStateMachine._MERGE_TOLERANCE_S`` so adapter next_tick
# computation is bit-identical to PSM when both cadences are due.
_MERGE_TOLERANCE_S = 1e-9


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
    ) -> None:
        self._config = config
        self._capabilities = capabilities
        self._clock = WallClock()

        # Cadence tracking (mirrors PSM ``_next_load_s`` / ``_next_throughput_s``)
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

        try:
            self._plugin_framework_metrics = PluginFrameworkMetrics()
        except ValueError:
            # Duplicate registration (only happens in tests / repeated
            # init in one process) — metrics emission disabled, but
            # the adapter still runs normally.
            self._plugin_framework_metrics = None

        # Build orchestrator + scheduler + circuit_breaker + registry.
        cb = CircuitBreaker(self._clock)
        transport_config = TransportConfig(
            request_timeout_seconds=config.scheduling.request_timeout_seconds,
            allow_insecure_grpc=config.scheduling.allow_insecure_grpc,
        )

        def _factory(plugin_id, endpoint, *, in_process_instance=None):
            return make_transport_for_endpoint(
                plugin_id,
                endpoint,
                transport_config,
                in_process_instance=in_process_instance,
            )

        server = PluginRegistryServer(
            clock=self._clock,
            auth=AllowUnauthenticatedAuth(),
            circuit_breaker=cb,
            transport_factory=_factory,
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

        # W2 gateway lifecycle: populated lazily by
        # ``_maybe_start_gateway`` if config opts in; consumed by
        # ``shutdown``.  Default ``None`` keeps the typical (gateway
        # disabled) deployment path zero-cost.
        self._gateway_server = None

        # Builtin plugins land in a follow-up PR. PR #1 ships only the
        # infrastructure (orchestrator + pipeline + transport + registry
        # + W1/W2 external-plugin wiring); the orchestrator path will
        # produce empty proposals on every tick until the follow-up adds
        # builtin load/throughput/reconcile/budget plugins.
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
        """W1: register the static-config external plugin list.

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
        """W2: stand up the gRPC registration gateway if configured.

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
            self._orchestrator._registry, listen=gw_cfg.listen
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
        """Matches ``PlannerStateMachine.initial_tick``."""
        self._next_load_s = start_s + self._config.load_adjustment_interval
        if self._config.enable_throughput_scaling:
            self._next_throughput_s = (
                start_s + self._config.throughput_adjustment_interval
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

        # 1. Observe FPM into regressions (mirror PSM ``_observe_fpm``
        #    before ``_advance_load``).
        is_easy = self._config.optimization_target != "sla"
        if (
            scheduled_tick.run_load_scaling
            and not is_easy
            and tick_input.fpm_observations is not None
        ):
            self._observe_fpm(tick_input.fpm_observations)

        # 3. Prime side-channel-dependent plugins. PR #1 ships zero
        #    builtins so this is a no-op; the follow-up PR re-introduces
        #    the prime_tick calls when it adds builtin_load_propose +
        #    builtin_budget_constrain.
        for slot in ("load_propose", "budget"):
            plugin = self._builtins.get(slot)
            if plugin is not None and hasattr(plugin, "prime_tick"):
                if slot == "load_propose":
                    plugin.prime_tick(
                        tick_input.fpm_observations, tick_input.worker_counts
                    )
                else:
                    plugin.prime_tick(tick_input.worker_counts)

        # 4. Advance cadence BEFORE running the tick — PSM does this in
        #    on_tick too; doing it here keeps ``_next_scheduled_tick``
        #    output aligned when returning PlannerEffects.next_tick.
        if scheduled_tick.run_throughput_scaling:
            self._next_throughput_s = (
                tick_input.now_s + self._config.throughput_adjustment_interval
            )
        if scheduled_tick.run_load_scaling:
            self._next_load_s = (
                tick_input.now_s + self._config.load_adjustment_interval
            )

        # 5. Build PipelineContext + baseline and drive the orchestrator.
        ctx = self._tick_input_to_context(tick_input)
        baseline = self._baseline_from_worker_counts(tick_input.worker_counts)
        outcome = await self._orchestrator.tick(ctx, baseline)

        # 6. Project PipelineOutcome onto PlannerEffects.
        scale_to = self._project_scale_to(
            outcome, tick_input.worker_counts or WorkerCounts()
        )

        # 7. Populate prediction fields on diagnostics. Consumed by the
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
          has a stronger signal; see ``_aggregate_disagg_reason``)
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
            diagnostics.load_decision_reason = self._aggregate_disagg_reason(
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
          aggregate (precedence via ``_aggregate_throughput_reason``)
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
                self._aggregate_throughput_reason(d.get("prefill"), d.get("decode"))
            )
        elif mode in ("prefill", "decode"):
            diagnostics.throughput_decision_reason = d.get(mode)

    @staticmethod
    def _aggregate_disagg_reason(
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
    def _aggregate_throughput_reason(
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
        if getattr(self, "_gateway_server", None) is not None:
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
        """Mirror of ``PlannerStateMachine._next_scheduled_tick``.

        Tracks upstream PSM commit `c388483ae` (KV-reuse awareness in
        load + throughput scaling): in load-only deployments (no
        throughput tick) load ticks carry a traffic-metrics scrape
        over the load interval so the planner can discount prefill
        work by KV hit rate. Without this branch, dual-path parity
        diverges on easy-mode scenarios.
        """
        at_s = min(self._next_load_s, self._next_throughput_s)
        is_load = self._next_load_s <= at_s + _MERGE_TOLERANCE_S
        is_throughput = self._next_throughput_s <= at_s + _MERGE_TOLERANCE_S
        if is_throughput:
            need_traffic = True
            traffic_duration_s = float(self._config.throughput_adjustment_interval)
        elif is_load and not self._config.enable_throughput_scaling:
            need_traffic = True
            traffic_duration_s = float(self._config.load_adjustment_interval)
        else:
            need_traffic = False
            traffic_duration_s = 0.0
        return ScheduledTick(
            at_s=at_s,
            run_load_scaling=is_load,
            run_throughput_scaling=is_throughput,
            need_worker_states=True,
            need_worker_fpm=is_load,
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
        return PipelineContext(
            request_id=f"tick-{ti.now_s}",
            decision_id=f"d-{ti.now_s}",
            observations=ObservationData(traffic=traffic),
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
