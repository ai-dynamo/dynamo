# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PluginScheduler.

Single-threaded asyncio invariant
---------------------------------

**All** public methods MUST be called from the event loop main task.
``record_result`` / ``invalidate_cache`` / ``compute_active_set`` mutate
the same in-memory dict; concurrent invocations from multiple asyncio
tasks (e.g. inside ``asyncio.gather`` plugin coroutines) is undefined
behaviour.

The expected orchestrator pattern is::

    active = scheduler.compute_active_set(now, stage)
    results = await asyncio.gather(*[p.transport.call(...) for p in active.triggered])
    # Back on the main task — serialise record_result:
    for plugin, result in zip(active.triggered, results):
        scheduler.record_result(plugin.plugin_id, stage, result, now)

No locks needed by assumption.

Cache invalidation 6-row table
------------------------------

Row-by-row, the scheduler clears a plugin's HOLD_LAST cache when:

1. ``registry.unregister(plugin_id)`` is called → subscribed via
   ``registry.on_unregister``.
2. Heartbeat monitor evicts a plugin → same code path as row 1
   (heartbeat monitor calls ``registry.unregister``).
3. ``CircuitBreaker`` transitions any plugin to OPEN → subscribed via
   ``circuit_breaker.on_open``.
4. Client-driven version upgrade (Unregister old + Register new) →
   row 1 for the unregister; the fresh Register starts a new cache.
5. ``invalidate_cache(reason="config_reload")`` — explicit full clear.
6. Orchestrator / planner process restart — cache lives in memory only,
   so process exit drops everything. No code needed.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from dynamo.planner.plugins.clock import Clock
from dynamo.planner.plugins.registry.circuit_breaker import CircuitBreaker
from dynamo.planner.plugins.registry.types import RegisteredPlugin
from dynamo.planner.plugins.types import HoldPolicy, OverrideResult

if TYPE_CHECKING:
    from dynamo.planner.monitoring.planner_metrics import PluginFrameworkMetrics
    from dynamo.planner.plugins.registry.server import PluginRegistryServer

log = logging.getLogger(__name__)


@dataclass
class InheritedResult:
    """A cached OverrideResult injected into the active set when a plugin
    is not due this tick but has HOLD_LAST hold_policy.

    ``priority`` is read from the current ``RegisteredPlugin`` (not the
    cache entry) so priority changes via re-registration take effect even
    if cache carry-over is momentarily in play.
    """

    plugin_id: str
    priority: int
    result: OverrideResult
    cached_at: float


@dataclass
class ActiveSet:
    """Per-tick scheduling decision for one stage."""

    triggered: list[RegisteredPlugin]
    inherited: list[InheritedResult]


@dataclass
class _CacheEntry:
    stage: str
    result: OverrideResult
    cached_at: float


class PluginScheduler:
    def __init__(
        self,
        registry: "PluginRegistryServer",
        circuit_breaker: CircuitBreaker,
        clock: Clock,
        metrics: Optional["PluginFrameworkMetrics"] = None,
    ) -> None:
        self._registry = registry
        self._circuit_breaker = circuit_breaker
        self._clock = clock
        # Cache keyed by (plugin_id, stage) — a plugin of plugin_type="predict"
        # only ever caches "predict" stage results, but keying on both makes
        # future per-stage caching explicit + avoids stage coupling bugs.
        self._cache: dict[tuple[str, str], _CacheEntry] = {}
        # Subscribe to the two event sources that drive cache invalidation.
        registry.on_unregister(self._on_registry_unregister)
        circuit_breaker.on_open(self._on_circuit_open)
        # Expose cache_age to the server's list_plugins.
        registry.attach_cache_age_lookup(self.cache_age)
        # Per-plugin tick scheduling metrics.
        # None = emission off; production path passes the orchestrator's
        # shared PluginFrameworkMetrics instance.
        self._metrics = metrics

    # ------------------------------------------------------------------
    # Per-tick scheduling
    # ------------------------------------------------------------------

    def compute_active_set(self, now: float, stage: str) -> ActiveSet:
        """Return triggered plugins (orchestrator must call) and inherited
        results (use cached output in place of calling the plugin) for
        this stage at ``now``.

        Plugins skipped entirely:
          - different ``plugin_type`` than the stage
          - ``enabled=False``
          - ``CircuitBreaker.can_call`` returns False (includes OPEN)

        Among the remainder:
          - "due" (``now - last_call_at >= execution_interval`` or
            first-ever tick) → ``triggered``
          - "not due" + ``HoldPolicy.HOLD_LAST`` + cache hit → ``inherited``
          - otherwise → skipped (treat as ACCEPT)
        """
        triggered: list[RegisteredPlugin] = []
        inherited: list[InheritedResult] = []

        for plugin in self._registry.all_plugins():
            if plugin.plugin_type != stage:
                continue
            if not plugin.enabled:
                continue
            if not self._circuit_breaker.can_call(plugin.plugin_id):
                continue

            is_due = self._is_due(plugin, now)
            if is_due:
                triggered.append(plugin)
                # tick_lag_seconds = how far behind the scheduled
                # cadence this tick is.  For the first-ever
                # call (last_call_at == -inf) lag is undefined; pin at
                # 0.  For zero-interval plugins lag is also 0 ("every
                # tick" means "always on time").
                if self._metrics is not None:
                    lag = self._compute_tick_lag(plugin, now)
                    self._metrics.tick_lag_seconds.labels(
                        plugin_id=plugin.plugin_id
                    ).set(lag)
                continue

            # Not due — either inherit cache or skip entirely; in both
            # cases the plugin was scheduled-but-deferred this tick, so
            # the counter fires.
            if self._metrics is not None:
                self._metrics.tick_skipped_total.labels(
                    plugin_id=plugin.plugin_id
                ).inc()

            if plugin.hold_policy == HoldPolicy.HOLD_LAST:
                entry = self._cache.get((plugin.plugin_id, stage))
                if entry is not None:
                    inherited.append(
                        InheritedResult(
                            plugin_id=plugin.plugin_id,
                            priority=plugin.priority,
                            result=entry.result,
                            cached_at=entry.cached_at,
                        )
                    )
                    continue
            # ACCEPT_WHEN_IDLE or HOLD_LAST with empty cache → treat as ACCEPT (skip).

        return ActiveSet(triggered=triggered, inherited=inherited)

    @staticmethod
    def _compute_tick_lag(plugin: RegisteredPlugin, now: float) -> float:
        """Seconds elapsed past the plugin's next-scheduled moment.

        Returns 0 for first-ever calls and for zero-interval plugins
        (no "scheduled moment" to lag behind).
        """
        if plugin.last_call_at == -math.inf:
            return 0.0
        if plugin.execution_interval_seconds <= 0.0:
            return 0.0
        due_at = plugin.last_call_at + plugin.execution_interval_seconds
        return max(0.0, now - due_at)

    @staticmethod
    def _is_due(plugin: RegisteredPlugin, now: float) -> bool:
        # First-ever tick: last_call_at == -inf → due regardless of interval.
        if plugin.last_call_at == -math.inf:
            return True
        # Zero interval means "every tick".
        if plugin.execution_interval_seconds <= 0.0:
            return True
        return (now - plugin.last_call_at) >= plugin.execution_interval_seconds

    # ------------------------------------------------------------------
    # Result recording (HOLD_LAST cache)
    # ------------------------------------------------------------------

    def record_result(
        self,
        plugin_id: str,
        stage: str,
        result: OverrideResult,
        tick_now: float,
    ) -> None:
        """Cache an OverrideResult for HOLD_LAST inheritance + bump the
        plugin's ``last_call_at`` / ``evaluations_total`` counters.

        The orchestrator should only call this for OverrideResult outputs
        — AcceptResult / RejectResult produce no inheritable content
        (AcceptResult = no opinion, RejectResult = stage short-circuited).
        """
        plugin = self._registry.get_plugin(plugin_id)
        if plugin is None:
            # Plugin unregistered between dispatch and result — drop silently.
            return
        plugin.last_call_at = tick_now
        plugin.evaluations_total += 1
        if plugin.hold_policy == HoldPolicy.HOLD_LAST:
            self._cache[(plugin_id, stage)] = _CacheEntry(
                stage=stage, result=result, cached_at=tick_now
            )

    # ------------------------------------------------------------------
    # Explicit cache invalidation (rows 5 + 6 of the v11 table)
    # ------------------------------------------------------------------

    def invalidate_cache(
        self, plugin_id: Optional[str] = None, reason: str = ""
    ) -> None:
        """Clear cache for one plugin (``plugin_id`` set) or all plugins
        (``plugin_id=None``). The ``reason`` is logged for audit.

        Row 5 (config reload) passes ``plugin_id=None, reason="config_reload"``.
        """
        if plugin_id is None:
            count = len(self._cache)
            self._cache.clear()
            log.info(
                "scheduler.invalidate_cache: cleared ALL (%d entries) reason=%s",
                count,
                reason or "<unspecified>",
            )
            return
        cleared = [k for k in self._cache if k[0] == plugin_id]
        for key in cleared:
            del self._cache[key]
        if cleared:
            log.info(
                "scheduler.invalidate_cache: plugin_id=%s cleared %d entries reason=%s",
                plugin_id,
                len(cleared),
                reason or "<unspecified>",
            )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def cache_age(self, plugin_id: str) -> float:
        """Oldest cached entry age in seconds for ``plugin_id``; ``0.0``
        if no cache entry exists (matches the PluginInfo default for
        never-called plugins)."""
        ages = [
            self._clock.monotonic() - entry.cached_at
            for key, entry in self._cache.items()
            if key[0] == plugin_id
        ]
        return max(ages) if ages else 0.0

    def cache_entries_count(self) -> int:
        """Total cache entries (all plugins × stages); admin/metric use."""
        return len(self._cache)

    # ------------------------------------------------------------------
    # Event handlers (private)
    # ------------------------------------------------------------------

    def _on_registry_unregister(self, plugin_id: str, reason: str) -> None:
        # Rows 1 + 2 + 4: unregister (explicit, heartbeat_missed, or
        # version-upgrade) → drop this plugin's cache.
        self.invalidate_cache(plugin_id, reason=f"unregister:{reason}")

    def _on_circuit_open(self, plugin_id: str) -> None:
        # Row 3: circuit OPEN → drop this plugin's cache so stale results
        # don't leak as "inherited" while the plugin is failing.
        self.invalidate_cache(plugin_id, reason="circuit_open")


__all__ = ["PluginScheduler", "ActiveSet", "InheritedResult"]
