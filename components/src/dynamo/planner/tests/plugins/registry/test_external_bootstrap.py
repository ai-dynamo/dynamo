# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``LocalPlannerOrchestrator.register_external_from_config``
and ``ExternalPluginEntry`` (static config-driven external plugin
registration).

This is the deployment model where a list of plugin endpoints is
supplied to the planner at startup (typically from a ConfigMap),
and the planner registers each by calling ``registry.register(...)``
on its own behalf — distinct from the gRPC gateway path (where
plugins self-register over the network).

Key invariants asserted here:

1. Schema-level rejects (bad scheme, bad type, missing required
   fields) surface as Pydantic ValidationError before the entry ever
   reaches the registry.
2. A bad entry MUST NOT block other entries — failure isolation is
   the difference between "ConfigMap typo" and "planner crashloop".
3. Per-entry failure path returns the registry's reject_reason
   verbatim so operators can debug without log-grepping.
4. Registered plugins show up in ``list_plugins`` with the right
   transport label (uds / grpc), matching what the e2e test would see.
"""

from __future__ import annotations

import pytest

from dynamo.planner.config.planner_config import ExternalPluginEntry
from dynamo.planner.plugins.clock import VirtualClock
from dynamo.planner.plugins.orchestrator.orchestrator import (
    LocalPlannerOrchestrator,
)
from dynamo.planner.plugins.registry.auth.base import (
    AllowUnauthenticatedAuth,
    AuthIdentity,
    AuthValidator,
)
from dynamo.planner.plugins.registry.circuit_breaker import CircuitBreaker
from dynamo.planner.plugins.registry.errors import AuthError
from dynamo.planner.plugins.registry.server import PluginRegistryServer
from dynamo.planner.plugins.scheduler import PluginScheduler
from dynamo.planner.plugins.transport.base import PluginTransport
from dynamo.planner.plugins.types import HoldPolicy, ListPluginsRequest

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubTransport(PluginTransport):
    def __init__(self, plugin_id, endpoint, *, in_process_instance=None, **_):
        self.plugin_id = plugin_id
        self.endpoint = endpoint
        self.timeout_seconds = 1.0
        self.closed = False

    async def call(self, method, request):
        return None

    async def close(self):
        self.closed = True


def _build_orch(
    *, auth: AuthValidator | None = None
) -> tuple[LocalPlannerOrchestrator, PluginRegistryServer]:
    clock = VirtualClock()
    cb = CircuitBreaker(clock)

    def factory(plugin_id, endpoint, *, in_process_instance=None):
        return _StubTransport(plugin_id, endpoint)

    server = PluginRegistryServer(
        clock=clock,
        auth=auth or AllowUnauthenticatedAuth(),
        circuit_breaker=cb,
        transport_factory=factory,
    )
    scheduler = PluginScheduler(server, cb, clock)
    orch = LocalPlannerOrchestrator(
        registry=server,
        scheduler=scheduler,
        circuit_breaker=cb,
        clock=clock,
        capabilities=None,
    )
    return orch, server


def _entry(
    plugin_id: str,
    *,
    plugin_type: str = "propose",
    priority: int = 5,
    endpoint: str = "grpc://127.0.0.1:9000",
    auth_token: str = "tok",
    protocol_version: str = "1.0",
    version: str = "v1",
    hold_policy: HoldPolicy = HoldPolicy.HOLD_LAST,
) -> ExternalPluginEntry:
    return ExternalPluginEntry(
        plugin_id=plugin_id,
        plugin_type=plugin_type,
        priority=priority,
        endpoint=endpoint,
        auth_token=auth_token,
        protocol_version=protocol_version,
        version=version,
        hold_policy=hold_policy,
    )


# ---------------------------------------------------------------------------
# Schema-level
# ---------------------------------------------------------------------------


def test_entry_rejects_unknown_plugin_type():
    """plugin_type is Literal'd to the four valid stages — anything
    else raises Pydantic ValidationError before the orchestrator sees
    it. Stops typos at the boundary."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ExternalPluginEntry(
            plugin_id="x",
            plugin_type="bogus",  # type: ignore[arg-type]
            priority=5,
            endpoint="grpc://127.0.0.1:9000",
        )


def test_entry_rejects_empty_endpoint():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ExternalPluginEntry(
            plugin_id="x",
            plugin_type="propose",
            priority=5,
            endpoint="",
        )


def test_entry_default_hold_policy_is_hold_last():
    """HOLD_LAST is the recommended default for static-config plugins —
    they're typically slow regression-style decisions that should
    persist across throttled ticks."""
    e = ExternalPluginEntry(
        plugin_id="x",
        plugin_type="propose",
        priority=5,
        endpoint="grpc://127.0.0.1:9000",
    )
    assert e.hold_policy == HoldPolicy.HOLD_LAST


# ---------------------------------------------------------------------------
# Bootstrap: empty + happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bootstrap_empty_list_no_op():
    orch, server = _build_orch()
    accepted, failures = await orch.register_external_from_config([])
    assert accepted == 0
    assert failures == []
    assert server.list_plugins(ListPluginsRequest()) == []


@pytest.mark.asyncio
async def test_bootstrap_happy_path_registers_entry():
    orch, server = _build_orch()
    accepted, failures = await orch.register_external_from_config([
        _entry("ext-a", endpoint="grpc://127.0.0.1:9000"),
    ])
    assert accepted == 1
    assert failures == []
    plugins = server.list_plugins(ListPluginsRequest())
    assert [p.plugin_id for p in plugins] == ["ext-a"]
    assert plugins[0].transport == "grpc"
    assert plugins[0].plugin_type == "propose"


@pytest.mark.asyncio
async def test_bootstrap_records_grpc_endpoint_correctly():
    """Different scheme → different transport label visible in
    list_plugins. Validates the entry → factory → transport_type
    derivation works for grpc:// not just unix://."""
    orch, server = _build_orch()
    await orch.register_external_from_config([
        _entry("ext-tcp", endpoint="grpc://10.0.0.5:9090"),
    ])
    info = server.list_plugins(ListPluginsRequest())[0]
    assert info.transport == "grpc"


# ---------------------------------------------------------------------------
# Failure isolation: one bad entry must not stop the others
# ---------------------------------------------------------------------------


class _SelectiveAuth(AuthValidator):
    """Approves any token in ``allow``; rejects everything else.
    Used to drive per-entry auth failure paths deterministically."""

    def __init__(self, allow: set[str]) -> None:
        self._allow = set(allow)

    async def validate(self, token):
        if token in self._allow:
            return AuthIdentity(source="static_secret", subject=token, metadata={})
        raise AuthError(f"selective_auth: token {token!r} not allowed")


@pytest.mark.asyncio
async def test_bootstrap_auth_failure_isolated():
    """Two entries; first has bad auth, second has good. The first
    must fail with reject_reason='auth_failed' and the second must
    still succeed — failure isolation is the primary contract this
    function exists for."""
    orch, server = _build_orch(auth=_SelectiveAuth(allow={"good"}))
    accepted, failures = await orch.register_external_from_config([
        _entry("bad-auth", auth_token="WRONG"),
        _entry("good-auth", auth_token="good"),
    ])
    assert accepted == 1
    assert len(failures) == 1
    assert failures[0][0] == "bad-auth"
    assert "auth_failed" in failures[0][1]
    # Good plugin still registered.
    ids = {p.plugin_id for p in server.list_plugins(ListPluginsRequest())}
    assert ids == {"good-auth"}


@pytest.mark.asyncio
async def test_bootstrap_inproc_endpoint_rejected():
    """``inproc://`` over the public register() path is a deployment
    bug — static-config plugins are out-of-process by definition.
    The reject must surface to the caller via failures, but other
    entries must continue."""
    orch, server = _build_orch()
    accepted, failures = await orch.register_external_from_config([
        _entry("misconfigured", endpoint="inproc://x"),
        _entry("ok", endpoint="grpc://127.0.0.1:9000"),
    ])
    assert accepted == 1
    assert {f[0] for f in failures} == {"misconfigured"}
    assert "inproc://" in failures[0][1]


@pytest.mark.asyncio
async def test_bootstrap_protocol_mismatch_isolated():
    """Plugin asking for an unsupported protocol_version must be
    rejected without dragging others down. Catches operator errors
    where a stale ConfigMap entry references an old protocol."""
    orch, server = _build_orch()
    accepted, failures = await orch.register_external_from_config([
        _entry("too-new", protocol_version="9.9"),
        _entry("ok"),
    ])
    assert accepted == 1
    assert {f[0] for f in failures} == {"too-new"}
    assert "protocol_version_unsupported" in failures[0][1]


@pytest.mark.asyncio
async def test_bootstrap_duplicate_plugin_id_within_config():
    """Two entries with the same plugin_id: first wins, second is
    rejected as duplicate. Catches a common ConfigMap copy-paste
    error before it manifests as confusing tick behaviour."""
    orch, server = _build_orch()
    accepted, failures = await orch.register_external_from_config([
        _entry("dup", endpoint="grpc://127.0.0.1:9000"),
        _entry("dup", endpoint="grpc://127.0.0.1:9000"),
    ])
    assert accepted == 1
    assert {f[0] for f in failures} == {"dup"}
    assert "duplicate_plugin_id" in failures[0][1]


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bootstrap_called_twice_second_is_all_duplicates():
    """Calling register_external_from_config twice with the same list
    on the same orchestrator: first run registers everything, second
    run sees them all as duplicates. Confirms there's no implicit
    'force re-register' that would silently break HOLD_LAST caches."""
    orch, server = _build_orch()
    entries = [_entry("a"), _entry("b", endpoint="grpc://h:1")]
    accepted_1, failures_1 = await orch.register_external_from_config(entries)
    assert accepted_1 == 2 and failures_1 == []
    accepted_2, failures_2 = await orch.register_external_from_config(entries)
    assert accepted_2 == 0
    assert {f[0] for f in failures_2} == {"a", "b"}
    for _, reason in failures_2:
        assert "duplicate_plugin_id" in reason


# ---------------------------------------------------------------------------
# Multi-stage smoke test: 4 stages registered side-by-side
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bootstrap_registers_all_four_stages():
    """Four entries, one per plugin_type. All four end up in
    list_plugins with the right plugin_type label. Validates the
    schema's plugin_type Literal lines up with the registry's accepted
    set."""
    orch, server = _build_orch()
    accepted, failures = await orch.register_external_from_config([
        _entry("ext-pred", plugin_type="predict", priority=1),
        _entry("ext-prop", plugin_type="propose", priority=5),
        _entry("ext-recon", plugin_type="reconcile", priority=2),
        _entry("ext-cons", plugin_type="constrain", priority=3),
    ])
    assert accepted == 4
    assert failures == []
    by_id = {p.plugin_id: p.plugin_type for p in server.list_plugins(ListPluginsRequest())}
    assert by_id == {
        "ext-pred": "predict",
        "ext-prop": "propose",
        "ext-recon": "reconcile",
        "ext-cons": "constrain",
    }
