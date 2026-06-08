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
import math
from typing import Any, Callable, Optional

from packaging.version import InvalidVersion, Version

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


# Floating-point tolerance for "is this a clean multiple of scale_interval?"
# Picked to absorb the float arithmetic noise that creeps in when a YAML
# author writes ``observation_window_seconds: 7.5`` and Pydantic /
# protobuf round-trip through float32: a 1e-6 tolerance is well below
# any meaningful scale_interval (defaults to 5.0s) yet large enough to
# accept ``2.0 * 5.0`` after a couple of float operations.
_WINDOW_ALIGN_TOLERANCE_S = 1e-6


def _check_observation_window(
    window_s: float, scale_interval_s: float
) -> Optional[str]:
    """Return a ``reject_reason`` string if ``window_s`` violates the
    proto contract, else ``None``.  Contract:

      0.0                                       -> accept (use scale_interval)
      N where (N > 0 and N is k * scale_interval) -> accept
      anything else                             -> reject

    ``scale_interval_s == 0.0`` disables the check (PSM path constructs
    the server without a scale_interval; the proto contract there is
    implicit and unverifiable).
    """
    if scale_interval_s <= 0.0:
        return None
    if window_s == 0.0:
        return None
    if window_s < 0.0:
        return (
            "observation_window_misaligned: "
            f"observation_window_seconds={window_s} must be >= 0"
        )
    ratio = window_s / scale_interval_s
    rounded = round(ratio)
    if rounded < 1 or abs(ratio - rounded) > _WINDOW_ALIGN_TOLERANCE_S:
        return (
            "observation_window_misaligned: "
            f"observation_window_seconds={window_s} must be 0 or a positive "
            f"integer multiple of scale_interval_seconds="
            f"{scale_interval_s}"
        )
    return None


class PluginRegistryServer:
    """In-memory plugin registry + transport lifecycle manager."""

    def __init__(
        self,
        clock: Clock,
        auth: AuthValidator,
        circuit_breaker: CircuitBreaker,
        transport_factory: TransportFactory,
        protocol_versions: tuple[str, str] = ("1.0", "1.0"),
        scale_interval_seconds: float = 0.0,
    ) -> None:
        self._clock = clock
        self._auth = auth
        self._circuit_breaker = circuit_breaker
        self._transport_factory = transport_factory
        self._protocol_min, self._protocol_max = protocol_versions
        # ``scale_interval_seconds`` enables phase alignment of
        # ``RegisteredPlugin.registered_at`` to the nearest tick boundary
        # (``floor(now / scale_interval) * scale_interval``).  This makes
        # plugins with the same ``execution_interval_seconds`` fire on
        # the same pipeline tick irrespective of small registration-time
        # skew between bootstrapped builtins (a few ms apart from
        # ``register_internal`` ordering in the orchestrator startup
        # sequence).  Without alignment, plugin A registered at T=0 and
        # plugin B at T=0.003 would forever fire on different ticks if
        # both declare the same interval, and any future
        # ``requires_produced_fields`` dependency between them would
        # silently deadlock.
        #
        # ``scale_interval_seconds == 0.0`` (the default) disables phase
        # alignment — preserves the legacy behaviour for tests and the
        # PSM path that constructs the server without a scale_interval.
        self._scale_interval_seconds = scale_interval_seconds
        self._plugins: dict[str, RegisteredPlugin] = {}
        self._unregister_callbacks: list[UnregisterCallback] = []
        # Scheduler reference lazy-attached so ``list_plugins`` can report
        # ``cache_age_seconds`` without making server construction depend
        # on the scheduler (which in turn already depends on the server).
        self._cache_age_lookup: Optional[Callable[[str], float]] = None

    def _aligned_anchor(self, raw_now: float) -> float:
        """Compute the registered_at anchor for a plugin registering at
        ``raw_now`` (monotonic clock).  When ``scale_interval_seconds``
        > 0, snaps to the nearest tick boundary below ``raw_now``.
        Otherwise returns ``raw_now`` unchanged (legacy / PSM path).
        """
        if self._scale_interval_seconds <= 0.0:
            return raw_now
        return (
            math.floor(raw_now / self._scale_interval_seconds)
            * self._scale_interval_seconds
        )

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

        # 2. Protocol version (inclusive range check). Use semantic
        # version compare via ``packaging.version.Version`` — a plain
        # string compare mis-orders "1.10" vs "1.2" (lexicographic puts
        # "1.10" < "1.2" because '1' < '2'), which could mistakenly
        # reject valid plugins once a component reaches 10.
        try:
            req_v = Version(req.protocol_version)
            min_v = Version(self._protocol_min)
            max_v = Version(self._protocol_max)
        except InvalidVersion as exc:
            reason = (
                f"protocol_version_malformed: requested={req.protocol_version!r} "
                f"(detail={exc!s})"
            )
            log.info("register rejected plugin_id=%s reason=%s", req.plugin_id, reason)
            return RegisterResponse(accepted=False, reject_reason=reason)
        if not (min_v <= req_v <= max_v):
            reason = (
                f"protocol_version_unsupported: requested={req.protocol_version}, "
                f"supported=[{self._protocol_min},{self._protocol_max}]"
            )
            log.info("register rejected plugin_id=%s reason=%s", req.plugin_id, reason)
            return RegisterResponse(accepted=False, reject_reason=reason)

        # 2.5. Observation window alignment.  Proto contract
        # (plugin.proto:107) requires ``observation_window_seconds`` to
        # be 0 OR a positive multiple of ``scale_interval_seconds`` so
        # the resulting Prometheus window aligns to tick boundaries —
        # otherwise the aggregated value crosses tick boundaries and the
        # plugin sees a moving target.  Only enforce when the server has
        # a known scale_interval (PSM path constructs without one, so
        # the constraint there is implicit / unverifiable).
        reject_window = _check_observation_window(
            req.observation_window_seconds, self._scale_interval_seconds
        )
        if reject_window is not None:
            log.info(
                "register rejected plugin_id=%s reason=%s",
                req.plugin_id,
                reject_window,
            )
            return RegisterResponse(accepted=False, reject_reason=reject_window)

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
            requires_produced_fields=list(req.requires_produced_fields),
            observation_window_seconds=req.observation_window_seconds,
            is_builtin=False,
            transport=transport,
            transport_type=transport_type,
            registered_at=self._aligned_anchor(self._clock.monotonic()),
            auth_subject=identity.subject,
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

    async def authenticated_heartbeat(
        self, plugin_id: str, auth_token: str
    ) -> tuple[bool, Optional[str]]:
        """Heartbeat for gateway-facing callers.

        Returns ``(ok, reject)`` where ``reject`` is one of:
          * ``None`` — auth succeeded AND plugin exists AND caller is its owner.
            ``ok`` is the underlying heartbeat result.
          * ``"auth_failed"`` — token did not validate.
          * ``"permission_denied"`` — caller does not own this plugin OR the
            plugin does not exist OR the plugin is in-process-only (registered
            via ``register_internal``, ``auth_subject == ""``).  The three
            cases collapse to a single gRPC status to avoid an existence-/
            transport-leak oracle: a token-holder could otherwise enumerate
            registered ``plugin_id``s by probing for differing status codes
            (``OK ok=false`` vs ``PERMISSION_DENIED``).
        """
        try:
            identity = await self._auth.validate(auth_token)
        except AuthError:
            return False, "auth_failed"
        plugin = self._plugins.get(plugin_id)
        # Collapse "unknown plugin" / "wrong subject" / "in-process plugin"
        # into a single response.  See docstring for the rationale.
        if (
            plugin is None
            or not plugin.auth_subject
            or plugin.auth_subject != identity.subject
        ):
            return False, "permission_denied"
        plugin.last_heartbeat_at = self._clock.monotonic()
        return True, None

    async def authenticated_unregister(
        self, plugin_id: str, auth_token: str, reason: str = ""
    ) -> tuple[bool, Optional[str]]:
        """Unregister for gateway-facing callers; same return contract as
        ``authenticated_heartbeat``.  Subject mismatch is rejected BEFORE the
        plugin is removed from the dict, so a forged Unregister cannot evict
        another caller's plugin even by accident.  Unknown plugin and
        in-process-only plugin (``auth_subject == ""``) also collapse to
        ``"permission_denied"`` to avoid an existence oracle — see
        ``authenticated_heartbeat`` docstring."""
        try:
            identity = await self._auth.validate(auth_token)
        except AuthError:
            return False, "auth_failed"
        plugin = self._plugins.get(plugin_id)
        if (
            plugin is None
            or not plugin.auth_subject
            or plugin.auth_subject != identity.subject
        ):
            return False, "permission_denied"
        ok = await self.unregister(plugin_id, reason=reason)
        return ok, None

    async def unregister(self, plugin_id: str, reason: str = "") -> bool:
        plugin = self._plugins.pop(plugin_id, None)
        if plugin is None:
            return False  # idempotent — caller can retry without surprise

        try:
            await plugin.transport.close()
        except (
            Exception
        ) as exc:  # noqa: BLE001 — defensive; close should not block unregister
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

    def attach_cache_age_lookup(self, lookup: Callable[[str], float]) -> None:
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
        requires_produced_fields: Optional[list[str]] = None,
        observation_window_seconds: float = 0.0,
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
            requires_produced_fields=list(requires_produced_fields or []),
            observation_window_seconds=observation_window_seconds,
            is_builtin=is_builtin,
            transport=transport,
            transport_type="in_process",
            registered_at=self._aligned_anchor(self._clock.monotonic()),
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
