# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``LocalPlannerOrchestrator`` — thin composition of the plugin pieces.

Owns:
  - the ``PluginRegistryServer`` (plugin lifecycle)
  - the ``CircuitBreaker`` (per-plugin failure-budget state)
  - the ``PluginScheduler`` (per-tick active-set + HOLD_LAST cache)
  - a regression-model dict consumed by the throughput-scaling builtin

Does **not** own:
  - the existing ``PlannerConnector`` — EXECUTE is returned as a
    ``PipelineOutcome`` decision; the caller (``NativePlannerBase``)
    translates ``apply`` into ``connector.add_component`` /
    ``remove_component`` calls.
  - any adapter between the proto ``PipelineContext`` and the existing
    ``core/types.py`` TickInput / PlannerEffects — that lives in the
    engine adapter.

Regression-model access:
  - ``get_regression(kind)`` returns the live reference; single-threaded
    asyncio means no locks are required.
  - ``update_regression(kind, fpm)`` mutates in place; callers (the
    throughput-propose builtin) invoke this serially on the event-loop
    main task.
  - Holding a returned reference across an ``await`` is unsafe — fetch
    a fresh reference after every await point.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence

from dynamo.planner.plugins.clock import Clock
from dynamo.planner.plugins.merge.types import ComponentKey
from dynamo.planner.plugins.orchestrator.pipeline import (
    PipelineOutcome,
    run_pipeline,
)
from dynamo.planner.plugins.registry.circuit_breaker import CircuitBreaker
from dynamo.planner.plugins.registry.server import PluginRegistryServer
from dynamo.planner.plugins.registry.types import RegisteredPlugin
from dynamo.planner.plugins.scheduler import PluginScheduler
from dynamo.planner.plugins.transport.errors import PluginUnknownMethodError
from dynamo.planner.plugins.types import (
    BootstrapRequest,
    HoldPolicy,
    ListPluginsRequest,
    PipelineContext,
    PluginInfo,
    RegisterRequest,
)

if TYPE_CHECKING:
    from dynamo.planner.core.types import TrafficObservation, WorkerCapabilities
    from dynamo.planner.monitoring.planner_metrics import PluginFrameworkMetrics

log = logging.getLogger(__name__)


class LocalPlannerOrchestrator:
    """Composes registry + scheduler + circuit breaker + merge into a
    single per-tick pipeline driver."""

    def __init__(
        self,
        *,
        registry: PluginRegistryServer,
        scheduler: PluginScheduler,
        circuit_breaker: CircuitBreaker,
        clock: Clock,
        tick_max_duration_seconds: float = 30.0,
        capabilities: Optional["WorkerCapabilities"] = None,
        metrics: Optional["PluginFrameworkMetrics"] = None,
    ) -> None:
        if tick_max_duration_seconds <= 0:
            raise ValueError("tick_max_duration_seconds must be > 0")
        self._registry = registry
        self._scheduler = scheduler
        self._circuit_breaker = circuit_breaker
        self._clock = clock
        self._tick_max_duration_seconds = tick_max_duration_seconds
        # Optional plugin-framework metrics. ``None`` = emission off
        # (test path, replay without scraping endpoint); every production
        # call path passes a populated ``PluginFrameworkMetrics``.
        self._metrics = metrics
        # Regression-model store keyed by component kind ("prefill" /
        # "decode" / "agg"). The throughput-propose / load-propose
        # builtins read these; ``NativePlannerBase`` wires the
        # mode-specific models in at startup.
        self._regression: dict[str, Any] = {}
        # Per-engine static capabilities (from WorkerInfo / MDC).
        # Builtins that compute engine throughput (throughput-propose /
        # load-propose) need these to clamp to max_num_batched_tokens /
        # max_kv_tokens / etc. ``None`` is allowed for early pipelines
        # that don't run those builtins.
        self._capabilities = capabilities
        # Cross-plugin shared state: PSM tracks ``_throughput_lower_bound_p/d``
        # on the state machine; in the plugin decomposition the throughput-
        # propose builtin writes these and the load-propose builtin reads
        # them. A later refactor can swap this for AT_LEAST-merge semantics.
        self._throughput_lower_bound: dict[str, int] = {"prefill": 1, "decode": 1}

    # ------------------------------------------------------------------
    # Regression model accessors
    # ------------------------------------------------------------------

    def get_regression(self, kind: str) -> Optional[Any]:
        """Live reference to the regression model for ``kind``.

        Callers MUST use synchronously on the event loop main task.
        Holding the reference across an ``await`` is unsafe — fetch a
        fresh reference after every await point."""
        return self._regression.get(kind)

    def update_regression(self, kind: str, model: Any) -> None:
        """Install / replace the regression model for ``kind``. Typically
        called by the throughput-propose builtin after adding a new FPM
        observation."""
        self._regression[kind] = model

    @property
    def capabilities(self) -> Optional["WorkerCapabilities"]:
        """Static per-engine capabilities. Builtins that need
        ``max_num_batched_tokens`` / ``max_kv_tokens`` etc. read this;
        ``None`` when the orchestrator was constructed without
        capabilities (e.g. early skeleton tests)."""
        return self._capabilities

    # Cross-plugin throughput lower bound (PSM's ``_throughput_lower_bound_p/d``).
    def set_throughput_lower_bound(self, component: str, value: int) -> None:
        self._throughput_lower_bound[component] = value

    def get_throughput_lower_bound(self, component: str) -> int:
        return self._throughput_lower_bound.get(component, 1)

    # ------------------------------------------------------------------
    # Plugin lifecycle (delegates)
    # ------------------------------------------------------------------

    def register_internal(
        self,
        plugin_id: str,
        plugin_type: str,
        priority: int,
        instance: Any,
        *,
        execution_interval_seconds: float = 0.0,
        hold_policy: HoldPolicy = HoldPolicy.ACCEPT_WHEN_IDLE,
        is_builtin: bool = True,
        version: str = "builtin",
        needs: Optional[list[str]] = None,
    ) -> RegisteredPlugin:
        """Register a plugin object that lives in this Python process.

        Thin wrapper around ``PluginRegistryServer.register_internal``;
        exists so callers (``NativePlannerBase``, tests) can interact
        with a single facade without reaching through to the registry.
        """
        return self._registry.register_internal(
            plugin_id=plugin_id,
            plugin_type=plugin_type,
            priority=priority,
            instance=instance,
            execution_interval_seconds=execution_interval_seconds,
            hold_policy=hold_policy,
            is_builtin=is_builtin,
            version=version,
            needs=needs,
        )

    def list_plugins(
        self, request: Optional[ListPluginsRequest] = None
    ) -> list[PluginInfo]:
        return self._registry.list_plugins(request or ListPluginsRequest())

    async def register_external_from_config(
        self, entries: Sequence[Any]
    ) -> tuple[int, list[tuple[str, str]]]:
        """Register a static list of out-of-process plugins.

        ``entries`` is typically ``PlannerConfig.scheduling.external_plugins``.
        Each entry is converted to a ``RegisterRequest`` and pushed
        through the same ``await registry.register(...)`` code path the
        gRPC gateway uses — so behaviour is identical between
        static-config and dynamic-self-register deployments.

        **Failure isolation**: a single bad entry (auth failure, bad
        endpoint scheme, plugin process unreachable) MUST NOT take down
        the planner. Each per-entry failure is logged with the reject
        reason; the function returns a per-entry status report so the
        caller (planner startup) can surface a summary line in
        operational logs.

        Returns ``(num_accepted, [(plugin_id, reject_reason_or_error), ...])``
        where the second element lists only the entries that failed.
        """
        accepted = 0
        failures: list[tuple[str, str]] = []
        for entry in entries:
            try:
                req = RegisterRequest(
                    plugin_id=entry.plugin_id,
                    plugin_type=entry.plugin_type,
                    priority=entry.priority,
                    endpoint=entry.endpoint,
                    auth_token=entry.auth_token,
                    protocol_version=entry.protocol_version,
                    version=entry.version,
                    execution_interval_seconds=entry.execution_interval_seconds,
                    hold_policy=entry.hold_policy,
                    needs=list(entry.needs),
                )
                resp = await self._registry.register(req)
            except Exception as exc:
                # Defensive: any unexpected exception (e.g. a transport
                # factory bug, a Pydantic validation slip) is logged
                # and the next entry is still attempted.
                log.warning(
                    "register_external_from_config: entry plugin_id=%s "
                    "raised %s: %s",
                    entry.plugin_id,
                    type(exc).__name__,
                    exc,
                )
                failures.append((entry.plugin_id, f"{type(exc).__name__}: {exc}"))
                continue
            if resp.accepted:
                accepted += 1
                log.info(
                    "register_external_from_config: accepted plugin_id=%s "
                    "type=%s endpoint=%s",
                    entry.plugin_id,
                    entry.plugin_type,
                    entry.endpoint,
                )
            else:
                log.warning(
                    "register_external_from_config: rejected plugin_id=%s "
                    "reason=%s",
                    entry.plugin_id,
                    resp.reject_reason,
                )
                failures.append((entry.plugin_id, resp.reject_reason))
        return accepted, failures

    # ------------------------------------------------------------------
    # Pipeline driver
    # ------------------------------------------------------------------

    async def tick(
        self,
        ctx: PipelineContext,
        baseline: Mapping[ComponentKey, int],
    ) -> PipelineOutcome:
        """Run one tick through PREDICT / PROPOSE / RECONCILE / CONSTRAIN.

        Returns a ``PipelineOutcome`` naming the EXECUTE decision
        (``apply`` / ``skip_no_targets`` / ``skip_short_circuit`` /
        ``skip_tick_timeout``) — the caller is responsible for
        projecting ``apply`` onto a ``PlannerConnector``.
        """
        tick_now = self._clock.monotonic()
        return await run_pipeline(
            ctx=ctx,
            scheduler=self._scheduler,
            circuit_breaker=self._circuit_breaker,
            baseline=baseline,
            clock=self._clock,
            tick_now=tick_now,
            tick_max_duration_seconds=self._tick_max_duration_seconds,
            metrics=self._metrics,
        )

    async def shutdown(self) -> None:
        """Unregister every plugin, closing their transports.

        Idempotent: subsequent calls find the registry empty and exit.
        """
        plugins = list(self._registry.all_plugins())
        for plugin in plugins:
            await self._registry.unregister(plugin.plugin_id, reason="shutdown")

    # ------------------------------------------------------------------
    # Pre-first-tick initialisation
    # ------------------------------------------------------------------

    def install_regressions(
        self,
        *,
        prefill: Optional[Any] = None,
        decode: Optional[Any] = None,
        agg: Optional[Any] = None,
    ) -> None:
        """Install regression models on the orchestrator's shared store
        so ``BuiltinThroughputPropose`` / ``BuiltinLoadPropose`` can read
        them via ``get_regression``.

        This is **orchestrator-owned state** (not a plugin concern) —
        the caller constructs regressions (typically via
        ``PSM.load_benchmark_fpms`` on a throwaway PSM) and hands them
        here for shared access across builtins. ``None`` for any kind
        skips that slot.

        Distinct from ``bootstrap_plugins`` on purpose — regressions
        must be installed **before** ``bootstrap_plugins`` is called
        because plugin Bootstrap implementations may read them via
        ``get_regression``.
        """
        if prefill is not None:
            self.update_regression("prefill", prefill)
        if decode is not None:
            self.update_regression("decode", decode)
        if agg is not None:
            self.update_regression("agg", agg)

    async def bootstrap_plugins(
        self,
        *,
        historical_traffic: Optional[Sequence["TrafficObservation"]] = None,
    ) -> None:
        """Fan out plugin Bootstrap lifecycle hooks.

        Two things happen in order:

        1. **Warm predictors** via Python-level helpers on any
           registered plugin exposing ``warm_from_observations``
           (currently ``BuiltinLoadPredictor``). Matches
           ``PSM.warm_load_predictors``.
        2. **Dispatch Bootstrap RPC** to every registered plugin so
           any side effects in concrete plugin implementations fire.
           Plugins that don't implement Bootstrap have the
           ``PluginUnknownMethodError`` caught and skipped.

        Regression installation is a **separate** concern — call
        ``install_regressions(...)`` before this if plugin Bootstrap
        implementations need to read regressions.

        The wire-format ``BootstrapRequest.bootstrap_data`` encoding
        for historical traffic is still TBD; in-process plugins get
        the data via the Python helpers above.
        """
        # 1. Python-level warm hook (primarily BuiltinLoadPredictor)
        if historical_traffic is not None:
            for plugin in self._registry.all_plugins():
                instance = getattr(plugin.transport, "_instance", None)
                warm = getattr(instance, "warm_from_observations", None)
                if callable(warm):
                    warm(historical_traffic)

        # 2. Bootstrap RPC fan-out (plugins that don't implement it are skipped)
        for plugin in self._registry.all_plugins():
            try:
                await plugin.transport.call("Bootstrap", BootstrapRequest())
            except PluginUnknownMethodError:
                continue
            except Exception as exc:  # noqa: BLE001 — defensive
                log.warning(
                    "bootstrap_plugins: Bootstrap RPC failed plugin_id=%s detail=%s",
                    plugin.plugin_id,
                    exc,
                )


__all__ = ["LocalPlannerOrchestrator"]
