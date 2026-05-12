# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``OrchestratorEngineAdapter`` startup wiring of the
external plugin path (static-config + gateway).

Locks the contracts:

1. ``bootstrap_plugins`` calls ``register_external_from_config`` when
   ``config.scheduling.external_plugins`` has entries — and is a
   no-op when it's empty (no surprise side-effects on the existing
   "no external plugins" deployment).
2. ``bootstrap_plugins`` opens the gRPC gateway iff
   ``scheduling.gateway.enabled`` is True; ``shutdown`` stops it.
3. Disabled-default contract: a fresh PlannerConfig must NOT cause
   any external registration or gateway port to open. Critical
   regression guard for existing deployments that don't know about
   W1/W2.

Uses the same ``allow_insecure_grpc=True`` + UDS pattern as
``test_external_plugin_e2e.py``; no subprocess or K8s needed —
this is the unit-level invariant guard. The full subprocess e2e
lives in ``tests/integration/test_external_plugin_subprocess_e2e.py``.
"""

from __future__ import annotations

from pathlib import Path

import grpc
import pytest

from dynamo.planner.config.planner_config import (
    ExternalPluginEntry,
    GatewayConfig,
    PlannerConfig,
)
from dynamo.planner.core.types import EngineCapabilities, WorkerCapabilities
from dynamo.planner.plugins.orchestrator.engine_adapter import (
    OrchestratorEngineAdapter,
)
from dynamo.planner.plugins.proto.v1 import plugin_pb2 as pb
from dynamo.planner.plugins.proto.v1 import plugin_pb2_grpc as pbg
from dynamo.planner.plugins.types import HoldPolicy, ListPluginsRequest

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


def _config(
    *,
    external_plugins: list[ExternalPluginEntry] | None = None,
    gateway: GatewayConfig | None = None,
    optimization_target: str = "throughput",
) -> PlannerConfig:
    cfg_kwargs: dict = dict(
        environment="kubernetes",
        mode="agg",
        enable_load_scaling=True,
        enable_throughput_scaling=False,
        optimization_target=optimization_target,
    )
    cfg = PlannerConfig(**cfg_kwargs)
    if external_plugins is not None:
        cfg.scheduling.external_plugins = external_plugins
    if gateway is not None:
        cfg.scheduling.gateway = gateway
    return cfg


def _caps() -> WorkerCapabilities:
    e = EngineCapabilities(num_gpu=1, max_num_batched_tokens=2048, max_kv_tokens=16384)
    return WorkerCapabilities(decode=e)


# ---------------------------------------------------------------------------
# Default behaviour: nothing extra happens
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_default_config_no_external_no_gateway():
    """A vanilla PlannerConfig must trigger zero W1/W2 side effects.
    Only the 5 builtins should be registered after bootstrap; the
    gateway port should remain closed."""
    adapter = OrchestratorEngineAdapter(_config(), _caps())
    await adapter.bootstrap_plugins()
    try:
        plugin_ids = {
            p.plugin_id
            for p in adapter._orchestrator.list_plugins(ListPluginsRequest())
        }
        assert plugin_ids == {
            "builtin_load_predictor",
            "builtin_load_propose",
            "builtin_throughput_propose",
            "builtin_reconcile",
            "builtin_budget_constrain",
        }
        assert adapter._gateway_server is None
    finally:
        await adapter.shutdown()


# ---------------------------------------------------------------------------
# W1: external_plugins entries get registered at bootstrap
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bootstrap_registers_external_plugins(tmp_path: Path):
    """Two ExternalPluginEntry → both visible in list_plugins after
    bootstrap. Endpoints don't need to be reachable at registration
    time — registration just records metadata + builds a transport
    handle; failure to connect surfaces on the first ``call()``."""
    sock_a = tmp_path / "ext-a.sock"
    sock_b = tmp_path / "ext-b.sock"
    cfg = _config(
        external_plugins=[
            ExternalPluginEntry(
                plugin_id="ext-a",
                plugin_type="propose",
                priority=4,
                endpoint=f"unix://{sock_a}",
                hold_policy=HoldPolicy.HOLD_LAST,
            ),
            ExternalPluginEntry(
                plugin_id="ext-b",
                plugin_type="reconcile",
                priority=2,
                endpoint=f"unix://{sock_b}",
                hold_policy=HoldPolicy.HOLD_LAST,
            ),
        ],
    )
    adapter = OrchestratorEngineAdapter(cfg, _caps())
    await adapter.bootstrap_plugins()
    try:
        ids = {
            p.plugin_id
            for p in adapter._orchestrator.list_plugins(ListPluginsRequest())
        }
        # 5 builtins + 2 external = 7
        assert "ext-a" in ids and "ext-b" in ids
        assert len(ids) == 7
    finally:
        await adapter.shutdown()


@pytest.mark.asyncio
async def test_bootstrap_handles_one_bad_entry(tmp_path: Path):
    """A protocol-version-mismatch entry must NOT prevent the good
    entry from being registered. Failure isolation lives in
    ``register_external_from_config``; this test asserts the
    adapter's bootstrap path actually invokes it (rather than
    swallowing or short-circuiting)."""
    cfg = _config(
        external_plugins=[
            ExternalPluginEntry(
                plugin_id="bad-version",
                plugin_type="propose",
                priority=5,
                endpoint=f"unix://{tmp_path}/x.sock",
                protocol_version="9.9",  # unsupported
            ),
            ExternalPluginEntry(
                plugin_id="ok",
                plugin_type="propose",
                priority=6,
                endpoint=f"unix://{tmp_path}/ok.sock",
            ),
        ],
    )
    adapter = OrchestratorEngineAdapter(cfg, _caps())
    await adapter.bootstrap_plugins()
    try:
        ids = {
            p.plugin_id
            for p in adapter._orchestrator.list_plugins(ListPluginsRequest())
        }
        assert "ok" in ids
        assert "bad-version" not in ids
    finally:
        await adapter.shutdown()


# ---------------------------------------------------------------------------
# W2: gateway opens iff config says so; shutdown closes it
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gateway_disabled_no_socket(tmp_path: Path):
    """``gateway.enabled=False`` (default) → no gRPC server; the
    socket path must not exist after bootstrap."""
    sock = tmp_path / "should-not-exist.sock"
    cfg = _config(gateway=GatewayConfig(enabled=False, listen=f"unix:{sock}"))
    adapter = OrchestratorEngineAdapter(cfg, _caps())
    await adapter.bootstrap_plugins()
    try:
        assert adapter._gateway_server is None
        assert not sock.exists()
    finally:
        await adapter.shutdown()


@pytest.mark.asyncio
async def test_gateway_enabled_listens_and_serves_register(tmp_path: Path):
    """``gateway.enabled=True`` → grpc.aio.Server is up and the
    PluginRegistry/Register RPC actually works through it.
    Validates the wiring (config → ``start_gateway_server``) and
    the bridge to the in-process registry in one shot."""
    sock = tmp_path / "gateway.sock"
    cfg = _config(gateway=GatewayConfig(enabled=True, listen=f"unix:{sock}"))
    adapter = OrchestratorEngineAdapter(cfg, _caps())
    await adapter.bootstrap_plugins()
    try:
        assert adapter._gateway_server is not None
        assert sock.exists(), f"gateway socket {sock} not created"

        # Connect to the gateway and call Register over the wire.
        async with grpc.aio.insecure_channel(f"unix:{sock}") as ch:
            stub = pbg.PluginRegistryStub(ch)
            resp = await stub.Register(
                pb.RegisterRequest(
                    plugin_id="dyn-plugin",
                    plugin_type="propose",
                    priority=5,
                    endpoint=f"unix://{tmp_path}/dyn.sock",
                    auth_token="anything",
                    protocol_version="1.0",
                    hold_policy=pb.HoldPolicy.HOLD_LAST,
                    version="v1",
                )
            )
        assert resp.accepted, f"Register rejected: {resp.reject_reason!r}"

        # And the registry now sees the plugin alongside the builtins.
        ids = {
            p.plugin_id
            for p in adapter._orchestrator.list_plugins(ListPluginsRequest())
        }
        assert "dyn-plugin" in ids
    finally:
        await adapter.shutdown()


@pytest.mark.asyncio
async def test_gateway_shutdown_closes_socket(tmp_path: Path):
    """After ``adapter.shutdown()`` the gateway socket is no longer
    accepting connections. Critical for clean restart in K8s — a
    leftover socket can wedge the next planner Pod."""
    sock = tmp_path / "shutdown_me.sock"
    cfg = _config(gateway=GatewayConfig(enabled=True, listen=f"unix:{sock}"))
    adapter = OrchestratorEngineAdapter(cfg, _caps())
    await adapter.bootstrap_plugins()
    assert adapter._gateway_server is not None
    await adapter.shutdown()

    # Connecting now must fail (gRPC raises UNAVAILABLE).
    async with grpc.aio.insecure_channel(f"unix:{sock}") as ch:
        stub = pbg.PluginRegistryStub(ch)
        with pytest.raises(grpc.aio.AioRpcError) as exc_info:
            await stub.Register(
                pb.RegisterRequest(
                    plugin_id="post-shutdown",
                    plugin_type="propose",
                    priority=5,
                    endpoint=f"unix://{tmp_path}/x.sock",
                    auth_token="x",
                    protocol_version="1.0",
                    hold_policy=pb.HoldPolicy.HOLD_LAST,
                    version="v1",
                )
            )
        assert exc_info.value.code() == grpc.StatusCode.UNAVAILABLE


# ---------------------------------------------------------------------------
# Combined W1 + W2: both wired together is the K8s-realistic scenario
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_w1_and_w2_compose(tmp_path: Path):
    """A typical K8s deployment: a static-config plugin (W1) is
    registered at startup AND the gateway (W2) is open for dynamic
    self-registration. Both populations must coexist correctly in
    the registry."""
    static_sock = tmp_path / "static.sock"
    gateway_sock = tmp_path / "gw.sock"
    cfg = _config(
        external_plugins=[
            ExternalPluginEntry(
                plugin_id="static-team-a",
                plugin_type="propose",
                priority=4,
                endpoint=f"unix://{static_sock}",
            ),
        ],
        gateway=GatewayConfig(enabled=True, listen=f"unix:{gateway_sock}"),
    )
    adapter = OrchestratorEngineAdapter(cfg, _caps())
    await adapter.bootstrap_plugins()
    try:
        # W1 entry is in.
        ids = {
            p.plugin_id
            for p in adapter._orchestrator.list_plugins(ListPluginsRequest())
        }
        assert "static-team-a" in ids

        # W2 self-register a second plugin.
        async with grpc.aio.insecure_channel(f"unix:{gateway_sock}") as ch:
            stub = pbg.PluginRegistryStub(ch)
            resp = await stub.Register(
                pb.RegisterRequest(
                    plugin_id="dynamic-team-b",
                    plugin_type="propose",
                    priority=6,
                    endpoint=f"unix://{tmp_path}/team-b.sock",
                    auth_token="x",
                    protocol_version="1.0",
                    hold_policy=pb.HoldPolicy.HOLD_LAST,
                    version="v1",
                )
            )
        assert resp.accepted

        ids = {
            p.plugin_id
            for p in adapter._orchestrator.list_plugins(ListPluginsRequest())
        }
        assert {"static-team-a", "dynamic-team-b"} <= ids
    finally:
        await adapter.shutdown()
