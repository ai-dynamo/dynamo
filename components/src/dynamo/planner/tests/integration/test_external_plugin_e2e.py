# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end test for the **external plugin** path.

A third-party plugin running in its own gRPC server, registering with
the planner through the public ``register()`` RPC over a real socket,
and being invoked by the orchestrator during a tick.

Coverage gap before this file:
- transport contract test exercises every transport (in_process / uds /
  grpc / grpc_mtls) but only against an **echo** servicer — the plugin
  contract (ProposeStageRequest → ProposeStageResponse oneof) is never
  driven over a network socket
- registry integration test covers the full lifecycle but with a
  **stub** transport — no real gRPC channel ever opens
- orchestrator e2e tests register builtins via ``register_internal``
  (in-process) — no transport hop

What this file proves:
1. A user-authored ``ProposePluginServicer`` running in a real
   ``grpc.aio.Server`` accepts ``Propose`` calls over the wire.
2. ``PluginRegistryServer.register(RegisterRequest(endpoint='grpc://...'))``
   builds a real ``GrpcTransport`` and stores the plugin record.
3. ``LocalPlannerOrchestrator.tick(...)`` invokes the external plugin
   over the gRPC channel and threads its ``OverrideResult`` through
   merge → reconcile → constrain into the final ``ScalingProposal``.
4. Same path works for ``unix://`` (UDS) — exercising both production
   transport schemes.

The test deliberately **does not** stand up a gRPC gateway in front of
``PluginRegistryServer``. This file calls ``server.register()`` from
Python directly; a real external plugin would need either (a) a gateway
or (b) ``register_internal`` for in-process Python plugins. The
transport hop being exercised is the **plugin invocation** hop
(orchestrator → plugin), which is the path that matters for dual-path
correctness.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, AsyncIterator, Iterable

import grpc
import pytest

from dynamo.planner.plugins.clock import WallClock
from dynamo.planner.plugins.merge.types import ComponentKey
from dynamo.planner.plugins.orchestrator.orchestrator import (
    LocalPlannerOrchestrator,
)
from dynamo.planner.plugins.proto.v1 import plugin_pb2 as pb
from dynamo.planner.plugins.proto.v1 import plugin_pb2_grpc as pbg
from dynamo.planner.plugins.registry.auth.base import AllowUnauthenticatedAuth
from dynamo.planner.plugins.registry.circuit_breaker import CircuitBreaker
from dynamo.planner.plugins.registry.server import PluginRegistryServer
from dynamo.planner.plugins.scheduler import PluginScheduler
from dynamo.planner.plugins.transport.config import (
    TransportConfig,
    make_transport_for_endpoint,
)
from dynamo.planner.plugins.types import (
    HoldPolicy,
    ListPluginsRequest,
    PipelineContext,
    RegisterRequest,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


# ---------------------------------------------------------------------------
# External plugin under test: a deterministic ProposePluginServicer
# ---------------------------------------------------------------------------


class _RecordingProposePlugin(pbg.ProposePluginServicer):
    """A deliberately simple external plugin that returns a fixed
    OverrideResult on every call. The orchestrator hits this over the
    network; the test asserts that:

    1. ``Propose`` was actually invoked (vs. the orchestrator silently
       skipping the plugin) — tracked via ``self.calls``.
    2. The response we returned is what lands in the tick's final
       proposal.
    """

    def __init__(self, *, prefill: int = 7, decode: int = 11) -> None:
        self.prefill = prefill
        self.decode = decode
        self.calls: list[pb.ProposeStageRequest] = []

    async def Propose(
        self,
        request: pb.ProposeStageRequest,
        context: grpc.aio.ServicerContext,
    ) -> pb.ProposeStageResponse:
        self.calls.append(request)
        resp = pb.ProposeStageResponse()
        ovr = resp.override
        ovr.reason = "external_plugin_e2e"
        t1 = ovr.targets.add()
        t1.sub_component_type = "prefill"
        t1.replicas = self.prefill
        t1.type = pb.OverrideType.SET
        t2 = ovr.targets.add()
        t2.sub_component_type = "decode"
        t2.replicas = self.decode
        t2.type = pb.OverrideType.SET
        # final=False so the merge layer is exercised normally.
        return resp


# ---------------------------------------------------------------------------
# Test infrastructure: spin up a plugin gRPC server and tear it down
# ---------------------------------------------------------------------------


async def _start_plugin_grpc_server(
    plugin: _RecordingProposePlugin, listen: str
) -> tuple[grpc.aio.Server, str]:
    """Start a real gRPC server hosting ``plugin`` at ``listen``.

    Returns (server, actual_listen). For ``:0`` ports, the actual bound
    port replaces the placeholder so the caller can plug it back into
    a ``grpc://`` endpoint.
    """
    server = grpc.aio.server()
    pbg.add_ProposePluginServicer_to_server(plugin, server)
    if listen.startswith("unix:"):
        port = server.add_insecure_port(listen)
    else:
        port = server.add_insecure_port(listen)
    await server.start()
    if listen.endswith(":0"):
        host = listen.rsplit(":", 1)[0]
        return server, f"{host}:{port}"
    return server, listen


def _build_orchestrator() -> tuple[
    LocalPlannerOrchestrator, PluginRegistryServer, list[Any]
]:
    """Compose the registry + scheduler + circuit breaker + orchestrator
    with the **real** transport factory (so ``register()`` over
    ``grpc://`` / ``unix://`` actually opens a channel).

    Returns the orchestrator, the registry (so the test can call
    ``register()`` directly), and a list of cleanup callables.
    """
    clock = WallClock()
    cb = CircuitBreaker(clock)
    transport_config = TransportConfig(
        request_timeout_seconds=2.0,
        # External plugin uses plain grpc:// (no mTLS) — opt in
        # explicitly so the factory accepts the endpoint instead of
        # rejecting it as insecure.
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
        capabilities=None,  # this test doesn't depend on capabilities
    )
    return orch, server, []


async def _register_external_plugin(
    server: PluginRegistryServer, *, plugin_id: str, endpoint: str
) -> None:
    """Hit the public ``register()`` RPC the way an external client
    would — same code path as a future gRPC gateway would invoke."""
    resp = await server.register(
        RegisterRequest(
            plugin_id=plugin_id,
            plugin_type="propose",
            priority=5,
            endpoint=endpoint,
            auth_token="anything",  # AllowUnauthenticatedAuth ignores it
            protocol_version="1.0",
            execution_interval_seconds=0.0,  # always due
            hold_policy=HoldPolicy.HOLD_LAST,
            version="v1",
        )
    )
    assert resp.accepted, f"register() rejected: {resp.reject_reason!r}"


def _make_baseline(prefill: int, decode: int) -> dict[ComponentKey, int]:
    return {
        ComponentKey(sub_component_type="prefill"): prefill,
        ComponentKey(sub_component_type="decode"): decode,
    }


def _ctx() -> PipelineContext:
    return PipelineContext(request_id="external-e2e-tick", decision_id="d-1")


def _final_targets(outcome) -> dict[str, int]:
    """Project ``ScalingProposal.targets`` into a dict keyed by sub
    component for easy assertion."""
    assert outcome.final_proposal is not None, (
        f"expected final_proposal on outcome, got "
        f"execute_action={outcome.execute_action!r} "
        f"short_circuit_reason={outcome.short_circuit_reason!r}"
    )
    out: dict[str, int] = {}
    for t in outcome.final_proposal.targets:
        if t.replicas is not None:
            out[t.sub_component_type] = t.replicas
    return out


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def grpc_external_plugin(
    request,
) -> AsyncIterator[tuple[_RecordingProposePlugin, str]]:
    """Plugin reachable via ``grpc://127.0.0.1:<bound>``."""
    plugin = _RecordingProposePlugin(prefill=7, decode=11)
    server, listen = await _start_plugin_grpc_server(plugin, "127.0.0.1:0")
    try:
        yield plugin, f"grpc://{listen}"
    finally:
        await server.stop(grace=0.1)


@pytest.fixture
async def uds_external_plugin(
    tmp_path: Path,
) -> AsyncIterator[tuple[_RecordingProposePlugin, str]]:
    """Plugin reachable via ``unix://<sock>``."""
    plugin = _RecordingProposePlugin(prefill=3, decode=4)
    sock_path = tmp_path / "external_plugin.sock"
    listen = f"unix:{sock_path}"
    server, _ = await _start_plugin_grpc_server(plugin, listen)
    try:
        yield plugin, f"unix://{sock_path}"
    finally:
        await server.stop(grace=0.1)


# ---------------------------------------------------------------------------
# Tests — both transports drive the same flow end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_external_plugin_register_and_invoked_over_grpc(
    grpc_external_plugin,
):
    """grpc:// path: plugin server in its own coroutine, planner
    registers it, one tick → plugin's Propose fires over the wire,
    its decision lands in the final proposal."""
    plugin, endpoint = grpc_external_plugin
    orch, registry, _ = _build_orchestrator()

    await _register_external_plugin(
        registry, plugin_id="external-propose-grpc", endpoint=endpoint
    )

    # Sanity: registry list shows the plugin with the right transport.
    plugins = registry.list_plugins(ListPluginsRequest())
    info = next(p for p in plugins if p.plugin_id == "external-propose-grpc")
    assert info.transport == "grpc"
    assert info.plugin_type == "propose"

    # Drive a tick. baseline=(2,2) means the budget/reconcile passthrough
    # leaves the plugin's SET unchanged → final proposal == plugin's
    # OverrideResult after merge.
    outcome = await orch.tick(_ctx(), _make_baseline(prefill=2, decode=2))

    # 1. The plugin actually got called over the network.
    assert len(plugin.calls) == 1, (
        f"expected exactly one Propose() call, got {len(plugin.calls)}"
    )
    # 2. The decision propagated end-to-end into the final proposal.
    assert outcome.execute_action == "apply"
    assert _final_targets(outcome) == {"prefill": 7, "decode": 11}

    await orch.shutdown()


@pytest.mark.asyncio
async def test_external_plugin_register_and_invoked_over_uds(
    uds_external_plugin,
):
    """unix:// path: same flow, different transport — proves UDS
    doesn't have any divergence vs grpc:// in production wiring."""
    plugin, endpoint = uds_external_plugin
    orch, registry, _ = _build_orchestrator()

    await _register_external_plugin(
        registry, plugin_id="external-propose-uds", endpoint=endpoint
    )

    plugins = registry.list_plugins(ListPluginsRequest())
    info = next(p for p in plugins if p.plugin_id == "external-propose-uds")
    assert info.transport == "uds"

    outcome = await orch.tick(_ctx(), _make_baseline(prefill=1, decode=1))

    assert len(plugin.calls) == 1
    assert outcome.execute_action == "apply"
    assert _final_targets(outcome) == {"prefill": 3, "decode": 4}

    await orch.shutdown()


@pytest.mark.asyncio
async def test_external_plugin_unregister_stops_invocations(
    grpc_external_plugin,
):
    """After Unregister, subsequent ticks must NOT invoke the plugin —
    the registry contract: removing a plugin closes its transport and
    drops it from the active set."""
    plugin, endpoint = grpc_external_plugin
    orch, registry, _ = _build_orchestrator()

    await _register_external_plugin(
        registry, plugin_id="external-propose-bye", endpoint=endpoint
    )

    # First tick invokes the plugin.
    await orch.tick(_ctx(), _make_baseline(prefill=2, decode=2))
    assert len(plugin.calls) == 1

    # Unregister; second tick must not call the plugin.
    ok = await registry.unregister("external-propose-bye", reason="test")
    assert ok
    await orch.tick(_ctx(), _make_baseline(prefill=2, decode=2))
    assert len(plugin.calls) == 1, (
        "plugin received a call after Unregister — registry isn't honouring "
        "the unregister contract"
    )

    await orch.shutdown()


@pytest.mark.asyncio
async def test_external_plugin_register_rejects_inproc_endpoint():
    """``inproc://`` over the network RPC is a client-side bug; the
    server must reject it. Locks the contract that drives the
    distinction between ``register()`` and ``register_internal()``."""
    orch, registry, _ = _build_orchestrator()

    resp = await registry.register(
        RegisterRequest(
            plugin_id="should-not-register",
            plugin_type="propose",
            priority=5,
            endpoint="inproc://x",
            auth_token="anything",
            protocol_version="1.0",
            hold_policy=HoldPolicy.HOLD_LAST,
            version="v1",
        )
    )
    assert resp.accepted is False
    assert "inproc://" in resp.reject_reason

    await orch.shutdown()


@pytest.mark.asyncio
async def test_external_plugin_two_external_plugins_compose(
    tmp_path: Path,
):
    """Two external plugins on different transports, both registered;
    one tick → both invoked → merge picks the higher-priority winner.
    Sanity-checks the multi-plugin path under real transport hops."""
    plugin_a = _RecordingProposePlugin(prefill=10, decode=10)
    plugin_b = _RecordingProposePlugin(prefill=99, decode=99)

    server_a, listen_a = await _start_plugin_grpc_server(
        plugin_a, "127.0.0.1:0"
    )
    sock_path = tmp_path / "plugin_b.sock"
    server_b, _ = await _start_plugin_grpc_server(
        plugin_b, f"unix:{sock_path}"
    )
    try:
        orch, registry, _ = _build_orchestrator()

        # plugin_a priority=5 (wins); plugin_b priority=10 (loses).
        # type-aware merge: smallest priority number wins on
        # PROPOSE conflict.
        await registry.register(
            RegisterRequest(
                plugin_id="ext-a",
                plugin_type="propose",
                priority=5,
                endpoint=f"grpc://{listen_a}",
                auth_token="x",
                protocol_version="1.0",
                hold_policy=HoldPolicy.HOLD_LAST,
                version="v1",
            )
        )
        await registry.register(
            RegisterRequest(
                plugin_id="ext-b",
                plugin_type="propose",
                priority=10,
                endpoint=f"unix://{sock_path}",
                auth_token="x",
                protocol_version="1.0",
                hold_policy=HoldPolicy.HOLD_LAST,
                version="v1",
            )
        )

        outcome = await orch.tick(_ctx(), _make_baseline(prefill=2, decode=2))

        # Both plugins were called over their respective transports
        # (M-1 zip evaluates all PROPOSE plugins in parallel).
        assert len(plugin_a.calls) == 1
        assert len(plugin_b.calls) == 1
        # plugin_a's lower priority wins the merge.
        assert outcome.execute_action == "apply"
        assert _final_targets(outcome) == {"prefill": 10, "decode": 10}

        await orch.shutdown()
    finally:
        await server_a.stop(grace=0.1)
        await server_b.stop(grace=0.1)


# ---------------------------------------------------------------------------
# 4-stage coverage: PREDICT / RECONCILE / CONSTRAIN external plugins
#
# The PROPOSE tests above prove the wire path works for the most-used stage,
# but each stage has its own request/response shape and its own pipeline-
# adapter logic (``_PredictAdapter`` for PREDICT chain-augment, the
# ``ReconcileStageRequest.proposals`` carrier for RECONCILE, the silent
# SET-drop in CONSTRAIN). These tests drive each over a real grpc.aio
# socket so stage-specific bugs surface.
# ---------------------------------------------------------------------------


class _RecordingPredictPlugin(pbg.PredictPluginServicer):
    """External PREDICT plugin: returns a deterministic
    ``PredictionData`` so the test can confirm chain-augment threaded
    it through and the orchestrator surfaced it on the
    ``PipelineOutcome.predict_outcome.prediction``.
    """

    def __init__(
        self,
        *,
        predicted_num_req: float = 1234.0,
        predicted_isl: float = 567.0,
        predicted_osl: float = 89.0,
        source: str = "external_predict_e2e",
    ) -> None:
        self._num_req = predicted_num_req
        self._isl = predicted_isl
        self._osl = predicted_osl
        self._source = source
        self.calls: list[pb.PredictStageRequest] = []

    async def Predict(
        self,
        request: pb.PredictStageRequest,
        context: grpc.aio.ServicerContext,
    ) -> pb.PredictStageResponse:
        self.calls.append(request)
        resp = pb.PredictStageResponse()
        resp.predictions.predicted_num_req = self._num_req
        resp.predictions.predicted_isl = self._isl
        resp.predictions.predicted_osl = self._osl
        resp.predictions.source = self._source
        # final=True: lowest-priority terminator. We only register one
        # external PREDICT plugin in this test so it's the only one.
        resp.final = True
        return resp


class _RecordingReconcilePlugin(pbg.ReconcilePluginServicer):
    """External RECONCILE plugin: emits a SET that re-shapes whatever
    the PROPOSE merge produced. RECONCILE-stage merge runs after
    PROPOSE so any RECONCILE OverrideResult takes precedence in the
    final scaling proposal."""

    def __init__(self, *, prefill: int, decode: int) -> None:
        self._prefill = prefill
        self._decode = decode
        self.calls: list[pb.ReconcileStageRequest] = []

    async def Reconcile(
        self,
        request: pb.ReconcileStageRequest,
        context: grpc.aio.ServicerContext,
    ) -> pb.ReconcileStageResponse:
        self.calls.append(request)
        resp = pb.ReconcileStageResponse()
        ovr = resp.override
        ovr.reason = "external_reconcile_e2e"
        for sub, n in (("prefill", self._prefill), ("decode", self._decode)):
            t = ovr.targets.add()
            t.sub_component_type = sub
            t.replicas = n
            t.type = pb.OverrideType.SET
        return resp


class _RecordingConstrainPlugin(pbg.ConstrainPluginServicer):
    """External CONSTRAIN plugin: emits an AT_MOST ceiling that should
    clamp whatever PROPOSE+RECONCILE produced. ``SET`` from a
    CONSTRAIN plugin is silently dropped at runtime (v11 contract); we
    use AT_MOST so the test can assert clamping actually happens."""

    def __init__(self, *, ceiling_prefill: int, ceiling_decode: int) -> None:
        self._cp = ceiling_prefill
        self._cd = ceiling_decode
        self.calls: list[pb.ConstrainStageRequest] = []

    async def Constrain(
        self,
        request: pb.ConstrainStageRequest,
        context: grpc.aio.ServicerContext,
    ) -> pb.ConstrainStageResponse:
        self.calls.append(request)
        resp = pb.ConstrainStageResponse()
        ovr = resp.override
        ovr.reason = "external_constrain_e2e"
        for sub, n in (("prefill", self._cp), ("decode", self._cd)):
            t = ovr.targets.add()
            t.sub_component_type = sub
            t.replicas = n
            t.type = pb.OverrideType.AT_MOST
        return resp


async def _start_predict_grpc_server(
    plugin: _RecordingPredictPlugin,
) -> tuple[grpc.aio.Server, str]:
    server = grpc.aio.server()
    pbg.add_PredictPluginServicer_to_server(plugin, server)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    return server, f"127.0.0.1:{port}"


async def _start_reconcile_grpc_server(
    plugin: _RecordingReconcilePlugin,
) -> tuple[grpc.aio.Server, str]:
    server = grpc.aio.server()
    pbg.add_ReconcilePluginServicer_to_server(plugin, server)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    return server, f"127.0.0.1:{port}"


async def _start_constrain_grpc_server(
    plugin: _RecordingConstrainPlugin,
) -> tuple[grpc.aio.Server, str]:
    server = grpc.aio.server()
    pbg.add_ConstrainPluginServicer_to_server(plugin, server)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    return server, f"127.0.0.1:{port}"


async def _register_with_type(
    server, *, plugin_id, plugin_type, priority, endpoint
):
    resp = await server.register(
        RegisterRequest(
            plugin_id=plugin_id,
            plugin_type=plugin_type,
            priority=priority,
            endpoint=endpoint,
            auth_token="anything",
            protocol_version="1.0",
            execution_interval_seconds=0.0,
            hold_policy=HoldPolicy.HOLD_LAST,
            version="v1",
        )
    )
    assert resp.accepted, f"register({plugin_type}) rejected: {resp.reject_reason!r}"


@pytest.mark.asyncio
async def test_external_predict_plugin_threaded_through_chain_augment():
    """A real PREDICT plugin in its own gRPC server: its
    ``PredictionData`` must surface as the chain-augment final
    prediction (visible on ``PipelineOutcome.predict_outcome``).

    Stage-specific check beyond PROPOSE: the wire boundary correctly
    handles the ``optional float`` semantics (``HasField()``-driven
    partial-merge in ``chain_augment``) — pre-bridge-fix, every
    PREDICT plugin call would have failed at gRPC serialisation just
    like PROPOSE did, but only the PROPOSE tests would have caught it.
    """
    plugin = _RecordingPredictPlugin(
        predicted_num_req=1234.0, predicted_isl=567.0, predicted_osl=89.0
    )
    server, listen = await _start_predict_grpc_server(plugin)
    try:
        orch, registry, _ = _build_orchestrator()
        await _register_with_type(
            registry,
            plugin_id="ext-predict",
            plugin_type="predict",
            priority=1,  # lowest priority = chain terminator
            endpoint=f"grpc://{listen}",
        )

        outcome = await orch.tick(_ctx(), _make_baseline(prefill=2, decode=2))

        # 1. Plugin actually got called over the wire.
        assert len(plugin.calls) == 1
        # 2. Its PredictionData propagated to the chain outcome.
        assert outcome.predict_outcome is not None
        pred = outcome.predict_outcome.prediction
        assert pred is not None
        assert pred.predicted_num_req == 1234.0
        assert pred.predicted_isl == 567.0
        assert pred.predicted_osl == 89.0
        # 3. ``final=True`` correctly set the chain terminator.
        assert outcome.predict_outcome.final_from == "ext-predict"

        await orch.shutdown()
    finally:
        await server.stop(grace=0.1)


@pytest.mark.asyncio
async def test_external_reconcile_plugin_overrides_propose_decision():
    """A real RECONCILE plugin: even with no PROPOSE plugins
    registered (so PROPOSE merge is empty), the RECONCILE override
    must drive the final proposal — proves RECONCILE plugins can
    inject decisions, not just transform existing ones.

    Pre-bridge-fix, ``ReconcileStageRequest.proposals`` (a repeated
    nested message) would have failed conversion separately from
    PROPOSE's flatter shape, so this is a distinct serialisation
    path worth locking."""
    plugin = _RecordingReconcilePlugin(prefill=12, decode=15)
    server, listen = await _start_reconcile_grpc_server(plugin)
    try:
        orch, registry, _ = _build_orchestrator()
        await _register_with_type(
            registry,
            plugin_id="ext-reconcile",
            plugin_type="reconcile",
            priority=2,
            endpoint=f"grpc://{listen}",
        )

        outcome = await orch.tick(_ctx(), _make_baseline(prefill=1, decode=1))

        assert len(plugin.calls) == 1
        assert outcome.execute_action == "apply"
        assert _final_targets(outcome) == {"prefill": 12, "decode": 15}

        await orch.shutdown()
    finally:
        await server.stop(grace=0.1)


@pytest.mark.asyncio
async def test_external_constrain_plugin_clamps_with_at_most():
    """A real CONSTRAIN plugin emits AT_MOST ceilings; combined with
    a PROPOSE plugin's SET targets above the ceiling, the final
    proposal must reflect the clamp.

    CONSTRAIN's runtime SET-drop is locked elsewhere; this test
    verifies the *over-the-wire* AT_MOST path actually clamps. Tests
    the OverrideType enum encoding survives proto round-trip — a
    common breakage mode in proto schema evolution."""
    propose_plugin = _RecordingProposePlugin(prefill=20, decode=25)
    constrain_plugin = _RecordingConstrainPlugin(
        ceiling_prefill=8, ceiling_decode=10
    )
    s_propose, listen_p = await _start_plugin_grpc_server(propose_plugin, "127.0.0.1:0")
    s_constrain, listen_c = await _start_constrain_grpc_server(constrain_plugin)
    try:
        orch, registry, _ = _build_orchestrator()
        await _register_with_type(
            registry,
            plugin_id="ext-propose-overshoot",
            plugin_type="propose",
            priority=5,
            endpoint=f"grpc://{listen_p}",
        )
        await _register_with_type(
            registry,
            plugin_id="ext-constrain-cap",
            plugin_type="constrain",
            priority=3,
            endpoint=f"grpc://{listen_c}",
        )

        outcome = await orch.tick(_ctx(), _make_baseline(prefill=2, decode=2))

        assert len(propose_plugin.calls) == 1
        assert len(constrain_plugin.calls) == 1
        # PROPOSE asked for (20,25); CONSTRAIN ceiling clamps to (8,10).
        assert outcome.execute_action == "apply"
        assert _final_targets(outcome) == {"prefill": 8, "decode": 10}

        await orch.shutdown()
    finally:
        await s_propose.stop(grace=0.1)
        await s_constrain.stop(grace=0.1)


@pytest.mark.asyncio
async def test_external_constrain_plugin_set_silently_dropped():
    """v11 contract: SET-type targets from a CONSTRAIN plugin are
    **silently dropped at runtime** (register-time rejection is
    infeasible). The plugin is still called; its output just has no
    effect on scale_to.

    This is the regression guard against a constraint plugin
    accidentally taking over the scaling decision via SET — which it
    must NOT be allowed to do (that's PROPOSE/RECONCILE territory)."""

    class _SetEmittingConstrain(pbg.ConstrainPluginServicer):
        def __init__(self):
            self.calls = 0

        async def Constrain(self, request, context):
            self.calls += 1
            resp = pb.ConstrainStageResponse()
            ovr = resp.override
            t = ovr.targets.add()
            t.sub_component_type = "prefill"
            t.replicas = 999  # absurd value to make the SET-drop visible
            t.type = pb.OverrideType.SET
            return resp

    propose_plugin = _RecordingProposePlugin(prefill=4, decode=5)
    constrain_plugin = _SetEmittingConstrain()
    s_propose, listen_p = await _start_plugin_grpc_server(propose_plugin, "127.0.0.1:0")
    s_constrain = grpc.aio.server()
    pbg.add_ConstrainPluginServicer_to_server(constrain_plugin, s_constrain)
    port_c = s_constrain.add_insecure_port("127.0.0.1:0")
    await s_constrain.start()
    try:
        orch, registry, _ = _build_orchestrator()
        await _register_with_type(
            registry,
            plugin_id="ext-propose-good",
            plugin_type="propose",
            priority=5,
            endpoint=f"grpc://{listen_p}",
        )
        await _register_with_type(
            registry,
            plugin_id="ext-constrain-misuse",
            plugin_type="constrain",
            priority=3,
            endpoint=f"grpc://127.0.0.1:{port_c}",
        )

        outcome = await orch.tick(_ctx(), _make_baseline(prefill=2, decode=2))

        # Constrain WAS called over the wire (its SET wasn't silently
        # short-circuited at the transport layer).
        assert constrain_plugin.calls == 1
        # And the SET=999 was dropped — final proposal still reflects
        # PROPOSE's prefill=4.
        assert _final_targets(outcome)["prefill"] == 4

        await orch.shutdown()
    finally:
        await s_propose.stop(grace=0.1)
        await s_constrain.stop(grace=0.1)
