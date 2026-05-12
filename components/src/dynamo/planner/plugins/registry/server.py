# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PluginRegistryServer.

Hosts the four Register / Heartbeat / Unregister / ListPlugins operations.
Invokable both via gRPC (a generated gRPC servicer wires to these methods)
and in-process (orchestrator calls ``register_internal`` for builtin
plugins + ``unregister`` during shutdown).

Responsibilities
----------------
1. Gate every Register through the ``AuthValidator`` and a protocol
   version check; reject duplicates (clients must Unregister + Register
   for version upgrades, not upsert).
2. Build the appropriate ``PluginTransport`` via an injected factory
   (``functools.partial(make_transport_for_endpoint, config=...)`` or
   equivalent) — the server stays decoupled from ``TransportConfig``.
3. Maintain the in-memory ``dict[plugin_id -> RegisteredPlugin]`` and
   update ``last_heartbeat_at`` / ``last_call_at`` / ``evaluations_total``
   (the last two are written by the orchestrator via accessors).
4. On Unregister, close the transport, reset the plugin's circuit-breaker
   state, and fan out to any ``on_unregister`` subscriber
   (PluginScheduler uses this to drop the plugin's HOLD_LAST cache).

The class is **single-threaded asyncio** — all methods run on the event
loop main task; the internal dict is unlocked.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from dynamo.planner.plugins.clock import Clock
from dynamo.planner.plugins.registry.auth.base import AuthValidator
from dynamo.planner.plugins.registry.circuit_breaker import CircuitBreaker
from dynamo.planner.plugins.registry.errors import AuthError
from dynamo.planner.plugins.registry.types import (
    RegisteredPlugin,
    derive_transport_type,
)
from dynamo.planner.plugins.transport.base import PluginTransport
from dynamo.planner.plugins.types import (
    HoldPolicy,
    ListPluginsRequest,
    PluginInfo,
    RegisterRequest,
    RegisterResponse,
)

log = logging.getLogger(__name__)


# Callable[(plugin_id, endpoint, *, in_process_instance=None), PluginTransport]
TransportFactory = Callable[..., PluginTransport]
# Callback invoked on unregister with (plugin_id, reason).
UnregisterCallback = Callable[[str, str], None]


class PluginRegistryServer:
    """In-memory plugin registry + transport lifecycle manager."""

    def __init__(
        self,
        clock: Clock,
        auth: AuthValidator,
        circuit_breaker: CircuitBreaker,
        transport_factory: TransportFactory,
        protocol_versions: tuple[str, str] = ("1.0", "1.0"),
    ) -> None:
        self._clock = clock
        self._auth = auth
        self._circuit_breaker = circuit_breaker
        self._transport_factory = transport_factory
        self._protocol_min, self._protocol_max = protocol_versions
        self._plugins: dict[str, RegisteredPlugin] = {}
        self._unregister_callbacks: list[UnregisterCallback] = []
        # Scheduler reference lazy-attached so ``list_plugins`` can report
        # ``cache_age_seconds`` without making server construction depend
        # on the scheduler (which in turn already depends on the server).
        self._cache_age_lookup: Optional[Callable[[str], float]] = None

    # ------------------------------------------------------------------
    # Public RPC-shaped API
    # ------------------------------------------------------------------

    async def register(self, req: RegisterRequest) -> RegisterResponse:
        # 1. Auth — on failure, return generic reject_reason to avoid oracle.
        try:
            identity = await self._auth.validate(req.auth_token)
        except AuthError as exc:
            log.info(
                "register rejected plugin_id=%s reason=auth_failed detail=%s",
                req.plugin_id,
                exc,
            )
            return RegisterResponse(accepted=False, reject_reason="auth_failed")

        # 2. Protocol version (inclusive range check).
        if not (self._protocol_min <= req.protocol_version <= self._protocol_max):
            reason = (
                f"protocol_version_unsupported: requested={req.protocol_version}, "
                f"supported=[{self._protocol_min},{self._protocol_max}]"
            )
            log.info("register rejected plugin_id=%s reason=%s", req.plugin_id, reason)
            return RegisterResponse(accepted=False, reject_reason=reason)

        # 3. Duplicate plugin_id → reject.
        if req.plugin_id in self._plugins:
            reason = "duplicate_plugin_id: must Unregister before re-Register"
            log.info("register rejected plugin_id=%s reason=%s", req.plugin_id, reason)
            return RegisterResponse(accepted=False, reject_reason=reason)

        # 4. Build transport. ValueError from the factory (e.g. unknown scheme,
        # missing mTLS on grpc://) surfaces as a reject — don't crash the server.
        try:
            transport_type = derive_transport_type(req.endpoint)
            if transport_type == "in_process":
                # in_process endpoints arriving via the *network* RPC are a
                # client-side bug: in-process plugins MUST use register_internal.
                reason = (
                    "endpoint_rejected: inproc:// endpoints are for in-process "
                    "registration only; use register_internal() for builtin / "
                    "in-process user plugins"
                )
                log.warning(
                    "register rejected plugin_id=%s reason=%s", req.plugin_id, reason
                )
                return RegisterResponse(accepted=False, reject_reason=reason)
            transport = self._transport_factory(req.plugin_id, req.endpoint)
        except ValueError as exc:
            log.warning(
                "register rejected plugin_id=%s reason=transport_build_failed detail=%s",
                req.plugin_id,
                exc,
            )
            return RegisterResponse(
                accepted=False, reject_reason=f"transport_build_failed: {exc}"
            )

        # 5. Build record + add to dict.
        plugin = RegisteredPlugin(
            plugin_id=req.plugin_id,
            plugin_type=req.plugin_type,
            priority=req.priority,
            endpoint=req.endpoint,
            version=req.version,
            protocol_version=req.protocol_version,
            execution_interval_seconds=req.execution_interval_seconds,
            hold_policy=req.hold_policy,
            needs=list(req.needs),
            fpm_encoding=req.fpm_encoding,
            request_timeout_seconds=req.request_timeout_seconds,
            is_builtin=False,
            transport=transport,
            transport_type=transport_type,
            registered_at=self._clock.monotonic(),
        )
        self._plugins[req.plugin_id] = plugin
        self._circuit_breaker.reset(req.plugin_id)

        log.info(
            "register accepted plugin_id=%s type=%s priority=%d endpoint=%s "
            "subject=%s auth_source=%s",
            plugin.plugin_id,
            plugin.plugin_type,
            plugin.priority,
            plugin.endpoint,
            identity.subject,
            identity.source,
        )
        return RegisterResponse(
            accepted=True,
            negotiated_protocol_version=req.protocol_version,
        )

    async def heartbeat(self, plugin_id: str) -> bool:
        plugin = self._plugins.get(plugin_id)
        if plugin is None:
            return False
        plugin.last_heartbeat_at = self._clock.monotonic()
        return True

    async def unregister(self, plugin_id: str, reason: str = "") -> bool:
        plugin = self._plugins.pop(plugin_id, None)
        if plugin is None:
            return False  # idempotent — caller can retry without surprise

        try:
            await plugin.transport.close()
        except Exception as exc:  # noqa: BLE001 — defensive; close should not block unregister
            log.warning(
                "unregister: transport.close failed plugin_id=%s detail=%s",
                plugin_id,
                exc,
            )

        self._circuit_breaker.reset(plugin_id)
        for cb in list(self._unregister_callbacks):
            try:
                cb(plugin_id, reason)
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "unregister: on_unregister callback failed plugin_id=%s detail=%s",
                    plugin_id,
                    exc,
                )

        log.info(
            "unregister plugin_id=%s reason=%s", plugin_id, reason or "<unspecified>"
        )
        return True

    def list_plugins(self, req: ListPluginsRequest) -> list[PluginInfo]:
        """Return plugin metadata filtered by ``stage_filter`` and
        ``include_disabled``. Full observability fields
        (``last_call_at_seconds_ago`` / ``cache_age_seconds``) are stubbed
        to ``0.0`` here and wired through the scheduler via
        ``attach_cache_age_lookup``.
        """
        now = self._clock.monotonic()
        out: list[PluginInfo] = []
        for plugin in self._plugins.values():
            if req.stage_filter and plugin.plugin_type != req.stage_filter:
                continue
            if not req.include_disabled and not plugin.enabled:
                continue
            out.append(
                PluginInfo(
                    plugin_id=plugin.plugin_id,
                    plugin_type=plugin.plugin_type,
                    priority=plugin.priority,
                    version=plugin.version,
                    protocol_version=plugin.protocol_version,
                    enabled=plugin.enabled,
                    is_builtin=plugin.is_builtin,
                    transport=plugin.transport_type,
                    circuit_state=self._circuit_breaker.state(plugin.plugin_id),
                    evaluations_total=plugin.evaluations_total,
                    last_call_at_seconds_ago=(
                        0.0
                        if plugin.last_call_at == float("-inf")
                        else max(0.0, now - plugin.last_call_at)
                    ),
                    cache_age_seconds=(
                        self._cache_age_lookup(plugin.plugin_id)
                        if self._cache_age_lookup is not None
                        else 0.0
                    ),
                )
            )
        return out

    # ------------------------------------------------------------------
    # Internal accessors (orchestrator / scheduler / heartbeat monitor use)
    # ------------------------------------------------------------------

    def get_plugin(self, plugin_id: str) -> Optional[RegisteredPlugin]:
        return self._plugins.get(plugin_id)

    def all_plugins(self) -> list[RegisteredPlugin]:
        return list(self._plugins.values())

    def on_unregister(self, callback: UnregisterCallback) -> None:
        """Subscribe to unregister events; callback receives
        ``(plugin_id, reason)``. Called synchronously on the event loop
        main task — callbacks MUST NOT await."""
        self._unregister_callbacks.append(callback)

    def attach_cache_age_lookup(
        self, lookup: Callable[[str], float]
    ) -> None:
        """Wire a scheduler's ``cache_age(plugin_id)`` into
        ``list_plugins``. Scheduler calls this from its own constructor so
        the server-side view reports cache age without introducing a
        server→scheduler import cycle."""
        self._cache_age_lookup = lookup

    # ------------------------------------------------------------------
    # Internal register path (builtin + in_process user plugins)
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
        fpm_encoding: str = "",
        request_timeout_seconds: float = 0.0,
    ) -> RegisteredPlugin:
        """Register without auth / protocol checks; wrap ``instance`` in
        ``InProcessTransport`` via the factory.

        Used by the orchestrator at startup for builtin plugins and by
        ``NativePlannerBase`` for ``in_process_plugins`` config entries.
        ``is_builtin=False`` should be passed for user in-process plugins
        so they show up as such in ListPlugins + metrics; they still
        bypass auth (trust boundary is the Python process).
        """
        if plugin_id in self._plugins:
            raise ValueError(
                f"register_internal: plugin_id={plugin_id!r} already registered"
            )
        endpoint = f"inproc://{plugin_id}"
        transport = self._transport_factory(
            plugin_id, endpoint, in_process_instance=instance
        )
        plugin = RegisteredPlugin(
            plugin_id=plugin_id,
            plugin_type=plugin_type,  # type: ignore[arg-type]
            priority=priority,
            endpoint=endpoint,
            version=version,
            protocol_version=self._protocol_max,
            execution_interval_seconds=execution_interval_seconds,
            hold_policy=hold_policy,
            needs=list(needs or []),
            fpm_encoding=fpm_encoding,
            request_timeout_seconds=request_timeout_seconds,
            is_builtin=is_builtin,
            transport=transport,
            transport_type="in_process",
            registered_at=self._clock.monotonic(),
        )
        self._plugins[plugin_id] = plugin
        self._circuit_breaker.reset(plugin_id)
        log.info(
            "register_internal plugin_id=%s type=%s priority=%d is_builtin=%s",
            plugin_id,
            plugin_type,
            priority,
            is_builtin,
        )
        return plugin


__all__ = ["PluginRegistryServer", "TransportFactory", "UnregisterCallback"]
