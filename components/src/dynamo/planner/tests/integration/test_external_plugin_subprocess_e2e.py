# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-process e2e test for the W2 gateway path (DEP-XXXX).

The planner-side gRPC gateway (``plugins/registry/gateway.py``) plus
the subprocess plugin runner (``external_plugin_subprocess_runner.py``)
together let an external plugin process self-register over the
network, just like a real K8s sidecar Deployment would.

What this proves vs the in-process e2e in
``test_external_plugin_e2e.py``:

- Plugin runs in its own Python interpreter — no shared imports, no
  shared event loop, no shared memory. If gRPC is doing an in-process
  shortcut anywhere, this test will catch it.
- Real fork / exec / signal lifecycle. ``kill(SIGKILL)`` reproduces a
  K8s Pod OOM more faithfully than ``cancel()`` on a coroutine.
- The plugin's only contract with the planner is the proto file —
  this is the contract a third-party plugin author actually writes
  against.

What this still doesn't cover (relative to real K8s):

- Cross-Pod networking (CNI, NetworkPolicy, service discovery)
- mTLS handshake (we use ``allow_insecure_grpc`` for both sides)
- ServiceAccount / TokenReview auth
- Pod restart policy + image pull

Those land in the K8s validation backlog.
"""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import AsyncIterator

import pytest

from dynamo.planner.plugins.clock import WallClock
from dynamo.planner.plugins.merge.types import ComponentKey
from dynamo.planner.plugins.orchestrator.orchestrator import (
    LocalPlannerOrchestrator,
)
from dynamo.planner.plugins.registry.auth.base import AllowUnauthenticatedAuth
from dynamo.planner.plugins.registry.circuit_breaker import CircuitBreaker
from dynamo.planner.plugins.registry.gateway import start_gateway_server
from dynamo.planner.plugins.registry.server import PluginRegistryServer
from dynamo.planner.plugins.scheduler import PluginScheduler
from dynamo.planner.plugins.transport.config import (
    TransportConfig,
    make_transport_for_endpoint,
)
from dynamo.planner.plugins.types import (
    CircuitState,
    ListPluginsRequest,
    PipelineContext,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
    pytest.mark.subprocess,  # opt-in marker — slow, spawns Python subprocesses
]


# ---------------------------------------------------------------------------
# Test harness: build planner stack with gateway running
# ---------------------------------------------------------------------------


def _build_planner_stack() -> tuple[
    LocalPlannerOrchestrator, PluginRegistryServer
]:
    clock = WallClock()
    cb = CircuitBreaker(clock)
    transport_config = TransportConfig(
        request_timeout_seconds=2.0,
        allow_insecure_grpc=True,
    )

    def factory(plugin_id: str, endpoint: str, *, in_process_instance=None):
        return make_transport_for_endpoint(
            plugin_id,
            endpoint,
            transport_config,
            in_process_instance=in_process_instance,
        )

    server = PluginRegistryServer(
        clock=clock,
        auth=AllowUnauthenticatedAuth(),
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


async def _spawn_subprocess_plugin(
    *,
    listen: str,
    gateway_endpoint: str,
    plugin_id: str = "external-subprocess-propose",
    prefill: int = 7,
    decode: int = 11,
    stage: str = "propose",
    priority: int = 5,
    extra_args: tuple[str, ...] = (),
    timeout_s: float = 8.0,
) -> tuple[subprocess.Popen, str]:
    """Spawn the runner as a separate Python process; wait for the
    ``LISTEN_READY`` line on stdout. Returns (process, plugin_endpoint).

    Reads stdout via ``asyncio.to_thread`` so we don't deadlock the
    event loop while the subprocess is initialising.
    """
    cmd = [
        sys.executable,
        "-m",
        "dynamo.planner.tests.integration.external_plugin_subprocess_runner",
        "--listen",
        listen,
        "--stage",
        stage,
        "--plugin-id",
        plugin_id,
        "--priority",
        str(priority),
        "--gateway-endpoint",
        gateway_endpoint,
        "--prefill",
        str(prefill),
        "--decode",
        str(decode),
    ]
    cmd.extend(extra_args)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    async def _readline_with_timeout() -> str:
        return await asyncio.wait_for(
            asyncio.to_thread(proc.stdout.readline),  # type: ignore[union-attr]
            timeout=timeout_s,
        )

    try:
        line = (await _readline_with_timeout()).decode().strip()
    except asyncio.TimeoutError:
        proc.kill()
        stderr = (proc.stderr.read() or b"").decode(errors="replace")  # type: ignore[union-attr]
        raise AssertionError(
            f"subprocess plugin failed to print LISTEN_READY in {timeout_s}s. "
            f"stderr:\n{stderr}"
        )
    if not line.startswith("LISTEN_READY "):
        proc.kill()
        stderr = (proc.stderr.read() or b"").decode(errors="replace")  # type: ignore[union-attr]
        raise AssertionError(
            f"subprocess plugin printed unexpected first line: {line!r}\n"
            f"stderr:\n{stderr}"
        )
    plugin_endpoint = line.split(" ", 1)[1]
    return proc, plugin_endpoint


def _terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def _make_baseline(prefill: int, decode: int) -> dict[ComponentKey, int]:
    return {
        ComponentKey(sub_component_type="prefill"): prefill,
        ComponentKey(sub_component_type="decode"): decode,
    }


def _ctx() -> PipelineContext:
    return PipelineContext(
        request_id="subproc-e2e-tick", decision_id="d-1"
    )


# ---------------------------------------------------------------------------
# Fixture: planner stack + gateway running on a UDS socket
# ---------------------------------------------------------------------------


@pytest.fixture
async def planner_with_gateway(
    tmp_path: Path,
) -> AsyncIterator[tuple[LocalPlannerOrchestrator, PluginRegistryServer, str]]:
    orch, server = _build_planner_stack()
    gateway_sock = tmp_path / "planner_gateway.sock"
    grpc_server, _ = await start_gateway_server(
        server, listen=f"unix:{gateway_sock}"
    )
    try:
        yield orch, server, f"unix://{gateway_sock}"
    finally:
        await grpc_server.stop(grace=0.1)
        await orch.shutdown()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_subprocess_plugin_self_registers_via_gateway(
    planner_with_gateway, tmp_path: Path
):
    """The full W2 happy path: plugin runs in its own process, calls
    ``Register`` through the gateway, gets accepted; planner then
    drives one tick and the plugin's decision lands in the final
    proposal."""
    orch, registry, gateway_endpoint = planner_with_gateway

    plugin_sock = tmp_path / "plugin.sock"
    proc, plugin_endpoint = await _spawn_subprocess_plugin(
        listen=f"unix:{plugin_sock}",
        gateway_endpoint=gateway_endpoint,
    )
    try:
        # Plugin is registered by the time LISTEN_READY printed —
        # registration happens before the print in the runner.
        plugins = registry.list_plugins(ListPluginsRequest())
        assert any(
            p.plugin_id == "external-subprocess-propose" for p in plugins
        ), f"plugin not registered; saw {[p.plugin_id for p in plugins]}"

        outcome = await orch.tick(
            _ctx(), _make_baseline(prefill=2, decode=2)
        )
        assert outcome.execute_action == "apply"
        assert outcome.final_proposal is not None
        targets = {
            t.sub_component_type: t.replicas
            for t in outcome.final_proposal.targets
            if t.replicas is not None
        }
        assert targets == {"prefill": 7, "decode": 11}, (
            f"plugin's decision didn't propagate; got targets={targets}"
        )
    finally:
        _terminate(proc)


@pytest.mark.asyncio
async def test_subprocess_plugin_crash_opens_circuit_breaker(
    planner_with_gateway, tmp_path: Path
):
    """``kill -9`` the plugin process; the next several ticks must
    each see ``PluginConnectionError`` from the dead UDS socket and
    increment the circuit breaker. After threshold failures, the
    circuit must show OPEN — proves the planner is robust to plugin
    crashes (the failure mode K8s Pod OOM produces).
    """
    orch, registry, gateway_endpoint = planner_with_gateway

    plugin_sock = tmp_path / "crash_plugin.sock"
    proc, _ = await _spawn_subprocess_plugin(
        listen=f"unix:{plugin_sock}",
        gateway_endpoint=gateway_endpoint,
        plugin_id="ext-crash",
    )
    try:
        # 1. First tick succeeds — proves the plugin was alive.
        out = await orch.tick(_ctx(), _make_baseline(prefill=2, decode=2))
        assert out.execute_action == "apply"

        # 2. Hard-kill the plugin. SIGKILL bypasses any graceful
        #    shutdown — closest thing to OOM-killer behaviour.
        proc.send_signal(signal.SIGKILL)
        proc.wait(timeout=5)

        # 3. Drive enough ticks to drive the circuit OPEN. The default
        #    threshold is 5 consecutive failures; do 7 to leave headroom.
        for _ in range(7):
            await orch.tick(_ctx(), _make_baseline(prefill=2, decode=2))

        info = next(
            p
            for p in registry.list_plugins(ListPluginsRequest())
            if p.plugin_id == "ext-crash"
        )
        assert info.circuit_state == CircuitState.OPEN, (
            f"circuit breaker did not open after plugin crash; "
            f"state={info.circuit_state}"
        )
    finally:
        _terminate(proc)


@pytest.mark.asyncio
async def test_gateway_unregister_via_grpc(
    planner_with_gateway, tmp_path: Path
):
    """An external client (here: another gRPC client in the test
    process — same code path as a plugin's graceful shutdown) calls
    ``Unregister`` through the gateway. Plugin must be removed from
    the registry's active set."""
    import grpc as _grpc

    from dynamo.planner.plugins.proto.v1 import plugin_pb2 as pb
    from dynamo.planner.plugins.proto.v1 import plugin_pb2_grpc as pbg

    orch, registry, gateway_endpoint = planner_with_gateway

    plugin_sock = tmp_path / "graceful_plugin.sock"
    proc, _ = await _spawn_subprocess_plugin(
        listen=f"unix:{plugin_sock}",
        gateway_endpoint=gateway_endpoint,
        plugin_id="ext-graceful",
    )
    try:
        target = gateway_endpoint.replace("unix://", "unix:")
        async with _grpc.aio.insecure_channel(target) as channel:
            stub = pbg.PluginRegistryStub(channel)
            resp = await stub.Unregister(
                pb.UnregisterRequest(
                    plugin_id="ext-graceful", reason="test_graceful"
                )
            )
            assert resp.ok is True

        ids = {p.plugin_id for p in registry.list_plugins(ListPluginsRequest())}
        assert "ext-graceful" not in ids
    finally:
        _terminate(proc)


# ---------------------------------------------------------------------------
# 4-stage subprocess coverage
#
# The first three tests above cover PROPOSE end-to-end (subprocess +
# gateway + tick invocation + crash isolation). PREDICT / RECONCILE /
# CONSTRAIN have in-process gRPC e2e in ``test_external_plugin_e2e.py``
# but no cross-process cover. These three tests close that gap by
# spawning the runner with ``--stage <name>`` and asserting the
# stage-specific output landed in the orchestrator's PipelineOutcome.
#
# What this catches that the in-process tests don't:
# - Stage-specific proto serialization across address spaces
#   (each stage has different request/response oneof shapes —
#   ReconcileStageRequest carries a repeated nested ``proposals``
#   field that PROPOSE/PREDICT don't, etc.)
# - The runner's ``--stage`` dispatch + per-stage register flow
# - That the plugin process exits cleanly under SIGTERM regardless
#   of which servicer it was hosting
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_subprocess_predict_plugin_threads_prediction(
    planner_with_gateway, tmp_path: Path
):
    """PREDICT subprocess plugin: run a tick and assert the planner's
    predict_outcome carries the values we hard-coded in the runner."""
    orch, _registry, gateway_endpoint = planner_with_gateway
    plugin_sock = tmp_path / "predict.sock"
    proc, _ = await _spawn_subprocess_plugin(
        listen=f"unix:{plugin_sock}",
        gateway_endpoint=gateway_endpoint,
        stage="predict",
        plugin_id="ext-sub-predict",
        priority=1,  # lowest = chain terminator
        extra_args=(
            "--predict-num-req", "4242",
            "--predict-isl", "1234",
            "--predict-osl", "789",
        ),
    )
    try:
        outcome = await orch.tick(_ctx(), _make_baseline(prefill=2, decode=2))
        assert outcome.predict_outcome is not None, "PREDICT chain produced no outcome"
        pred = outcome.predict_outcome.prediction
        assert pred is not None, "no prediction returned from chain"
        assert pred.predicted_num_req == 4242
        assert pred.predicted_isl == 1234
        assert pred.predicted_osl == 789
        # Lowest-priority terminator with final=True wins the chain.
        assert outcome.predict_outcome.final_from == "ext-sub-predict"
    finally:
        _terminate(proc)


@pytest.mark.asyncio
async def test_subprocess_reconcile_plugin_drives_final_proposal(
    planner_with_gateway, tmp_path: Path
):
    """RECONCILE subprocess plugin with no PROPOSE plugins registered:
    the RECONCILE override is what reaches scale_to. Locks the
    cross-process serialization of the RECONCILE request shape (which
    differs from PROPOSE — it carries a repeated ``proposals`` nested
    message even when empty)."""
    orch, _registry, gateway_endpoint = planner_with_gateway
    plugin_sock = tmp_path / "reconcile.sock"
    proc, _ = await _spawn_subprocess_plugin(
        listen=f"unix:{plugin_sock}",
        gateway_endpoint=gateway_endpoint,
        stage="reconcile",
        plugin_id="ext-sub-reconcile",
        priority=2,
        prefill=12,
        decode=15,
    )
    try:
        outcome = await orch.tick(_ctx(), _make_baseline(prefill=1, decode=1))
        assert outcome.execute_action == "apply"
        assert outcome.final_proposal is not None
        targets = {
            t.sub_component_type: t.replicas
            for t in outcome.final_proposal.targets
            if t.replicas is not None
        }
        assert targets == {"prefill": 12, "decode": 15}, (
            f"RECONCILE subprocess output didn't reach final_proposal; "
            f"got {targets}"
        )
    finally:
        _terminate(proc)


@pytest.mark.asyncio
async def test_subprocess_constrain_plugin_clamps_propose_overshoot(
    planner_with_gateway, tmp_path: Path
):
    """Combined PROPOSE+CONSTRAIN, both subprocess: PROPOSE asks for
    high replicas, subprocess CONSTRAIN's AT_MOST clamps. Validates
    OverrideType.AT_MOST encoding survives proto round-trip on the
    constrain path specifically (separate from PROPOSE's SET path
    which the existing tests already cover)."""
    orch, _registry, gateway_endpoint = planner_with_gateway

    # PROPOSE asks for big replicas.
    propose_sock = tmp_path / "propose_overshoot.sock"
    propose_proc, _ = await _spawn_subprocess_plugin(
        listen=f"unix:{propose_sock}",
        gateway_endpoint=gateway_endpoint,
        stage="propose",
        plugin_id="ext-sub-propose-big",
        priority=5,
        prefill=20,
        decode=25,
    )
    # CONSTRAIN caps at lower ceilings.
    constrain_sock = tmp_path / "constrain_cap.sock"
    constrain_proc, _ = await _spawn_subprocess_plugin(
        listen=f"unix:{constrain_sock}",
        gateway_endpoint=gateway_endpoint,
        stage="constrain",
        plugin_id="ext-sub-constrain-cap",
        priority=3,
        prefill=8,   # AT_MOST ceiling for prefill
        decode=10,   # AT_MOST ceiling for decode
    )
    try:
        outcome = await orch.tick(_ctx(), _make_baseline(prefill=2, decode=2))
        assert outcome.execute_action == "apply"
        targets = {
            t.sub_component_type: t.replicas
            for t in outcome.final_proposal.targets  # type: ignore[union-attr]
            if t.replicas is not None
        }
        # PROPOSE asked (20,25); CONSTRAIN clamps to (8,10).
        assert targets == {"prefill": 8, "decode": 10}
    finally:
        _terminate(propose_proc)
        _terminate(constrain_proc)
