# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transport contract test — **core acceptance**.

For a single ``echo`` plugin (returns the request's PipelineContext as the
response's predictions field), assert that both PR #1 transports
(``in_process`` / ``grpc``) produce the **byte-equal serialized response**
for the same set of inputs.

This is the strongest guarantee that transport changes won't introduce
silent behavioral drift between deployment forms. The ``grpc_mtls``
variant lands alongside cert-manager wiring in a follow-up PR.
"""

from __future__ import annotations

from pathlib import Path
from typing import AsyncIterator

import grpc
import pytest

from dynamo.planner.plugins.proto.v1 import plugin_pb2 as pb
from dynamo.planner.plugins.proto.v1 import plugin_pb2_grpc as pbg
from dynamo.planner.plugins.transport import (
    GrpcTransport,
    InProcessTransport,
    PluginConnectionError,
    PluginTimeoutError,
    PluginTransport,
    PluginUnknownMethodError,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


# ----------------------------------------------------------------------------
# Echo plugin — Predict echoes context.observations.traffic into predictions
# ----------------------------------------------------------------------------


class EchoServicer(pbg.PredictPluginServicer):
    """gRPC servicer that echoes traffic.num_req into predictions.predicted_num_req
    so we can verify the request reached the plugin and round-tripped.
    """

    async def Predict(
        self, request: pb.PredictStageRequest, context
    ) -> pb.PredictStageResponse:
        ctx = request.context
        resp = pb.PredictStageResponse()
        if ctx.HasField("observations") and ctx.observations.HasField("traffic"):
            resp.predictions.predicted_num_req = ctx.observations.traffic.num_req
            resp.predictions.predicted_isl = ctx.observations.traffic.isl
            resp.predictions.predicted_osl = ctx.observations.traffic.osl
        resp.predictions.source = "echo-server"
        return resp


class EchoPluginInProcess:
    """Same logic as EchoServicer but as a Python in-process callable.

    The InProcessTransport calls ``Predict(req)`` directly — no servicer
    wrapper / context arg.
    """

    async def Predict(self, request: pb.PredictStageRequest) -> pb.PredictStageResponse:
        ctx = request.context
        resp = pb.PredictStageResponse()
        if ctx.HasField("observations") and ctx.observations.HasField("traffic"):
            resp.predictions.predicted_num_req = ctx.observations.traffic.num_req
            resp.predictions.predicted_isl = ctx.observations.traffic.isl
            resp.predictions.predicted_osl = ctx.observations.traffic.osl
        resp.predictions.source = "echo-server"
        return resp


# ----------------------------------------------------------------------------
# gRPC server fixtures (insecure TCP)
# ----------------------------------------------------------------------------


async def _start_grpc_server(listen: str) -> tuple[grpc.aio.Server, str]:
    """Start a gRPC server with EchoServicer at ``listen``.

    Returns (server, actual_listen) — for ":0" port, returns the bound port.
    """
    server = grpc.aio.server()
    pbg.add_PredictPluginServicer_to_server(EchoServicer(), server)
    port = server.add_insecure_port(listen)
    await server.start()
    # For TCP ":0", rebuild listen with actual port; for UDS, listen is unchanged
    if (
        listen.startswith("[::]:0")
        or listen.startswith("0.0.0.0:0")
        or listen.endswith(":0")
    ):
        host = listen.rsplit(":", 1)[0]
        actual_listen = f"{host}:{port}"
    else:
        actual_listen = listen
    return server, actual_listen


# ----------------------------------------------------------------------------
# Test data — 7 representative PipelineContext payloads
# ----------------------------------------------------------------------------


def _ctx_minimal() -> pb.PipelineContext:
    return pb.PipelineContext(request_id="req-min")


def _ctx_with_traffic() -> pb.PipelineContext:
    c = pb.PipelineContext(request_id="req-traffic")
    c.observations.traffic.duration_s = 60.0
    c.observations.traffic.num_req = 1500.0
    c.observations.traffic.isl = 3000.0
    c.observations.traffic.osl = 150.0
    return c


def _ctx_with_full_observations() -> pb.PipelineContext:
    c = pb.PipelineContext(request_id="req-full", decision_id="d-1")
    c.observations.traffic.num_req = 2000
    c.observations.traffic.isl = 2500
    c.observations.traffic.osl = 200
    c.observations.workers.ready_prefill = 4
    c.observations.workers.ready_decode = 8
    c.observations.workers.expected_prefill = 4
    c.observations.workers.expected_decode = 10
    c.observations.fpm.prefill_engines["e0"] = b"\x01\x02\x03"
    c.observations.fpm.decode_engines["e1"] = b"\xff\xfe"
    return c


def _ctx_with_predictions_proposal() -> pb.PipelineContext:
    c = pb.PipelineContext(request_id="req-pp")
    c.predictions.predicted_num_req = 1800.0
    c.predictions.source = "upstream-predictor"
    c.proposal.targets.add(sub_component_type="prefill", replicas=6)
    c.proposal.targets.add(sub_component_type="decode", replicas=12)
    return c


def _ctx_with_unicode_reason() -> pb.PipelineContext:
    c = pb.PipelineContext(request_id="req-unicode")
    c.proposal.reason = "测试中文 reason — 包括 emoji 🚀"
    c.proposal.targets.add(sub_component_type="prefill", replicas=8)
    return c


def _ctx_with_constrained() -> pb.PipelineContext:
    c = pb.PipelineContext(request_id="req-constrained", decision_id="d-2")
    c.observations.traffic.num_req = 500
    c.observations.traffic.isl = 1000
    c.observations.traffic.osl = 100
    c.constrained.targets.add(sub_component_type="prefill", replicas=2)
    c.constrained.reason = "budget-constrained"
    return c


def _ctx_zero_replicas_explicit() -> pb.PipelineContext:
    """ComponentTarget.replicas=0 explicitly set (not unset) — must round-trip."""
    c = pb.PipelineContext(request_id="req-zero")
    c.observations.traffic.num_req = 0.0  # also explicit zero
    c.observations.traffic.isl = 0
    c.observations.traffic.osl = 0
    target = c.constrained.targets.add(sub_component_type="decode")
    target.replicas = 0  # explicit zero
    return c


_INPUTS = [
    ("minimal", _ctx_minimal),
    ("with_traffic", _ctx_with_traffic),
    ("full_observations", _ctx_with_full_observations),
    ("predictions_proposal", _ctx_with_predictions_proposal),
    ("unicode_reason", _ctx_with_unicode_reason),
    ("constrained", _ctx_with_constrained),
    ("zero_replicas_explicit", _ctx_zero_replicas_explicit),
]


# ----------------------------------------------------------------------------
# Async fixture: 4 transports targeting the same Echo plugin
# ----------------------------------------------------------------------------


@pytest.fixture
def transport_kind(request):
    """Parametrized over 2 transport kinds (in_process, grpc)."""
    return request.param


@pytest.fixture
async def echo_transport(transport_kind) -> AsyncIterator[PluginTransport]:
    """Yield a PluginTransport pointed at an EchoServicer (or in-process Echo).

    Cleans up server + transport on teardown.
    """
    if transport_kind == "in_process":
        t = InProcessTransport("echo", EchoPluginInProcess(), timeout_seconds=2.0)
        try:
            yield t
        finally:
            await t.close()
        return

    if transport_kind == "grpc":
        server, listen = await _start_grpc_server("127.0.0.1:0")
        try:
            t = GrpcTransport(
                "echo", f"grpc://{listen}", allow_insecure=True, timeout_seconds=2.0
            )
            try:
                yield t
            finally:
                await t.close()
        finally:
            await server.stop(grace=0.1)
        return

    pytest.fail(f"unknown transport_kind: {transport_kind}")


_TRANSPORT_KINDS = ["in_process", "grpc"]


# ----------------------------------------------------------------------------
# Contract test: 7 inputs × 2 transports = 14 cases of byte-equality
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("transport_kind", _TRANSPORT_KINDS, indirect=True)
@pytest.mark.parametrize(
    "input_name,ctx_factory",
    _INPUTS,
    ids=[name for name, _ in _INPUTS],
)
@pytest.mark.asyncio
async def test_round_trip_equivalence(
    echo_transport: PluginTransport,
    input_name: str,
    ctx_factory,
    transport_kind: str,
):
    """For every (input × transport) pair, the response is byte-equal."""
    ctx = ctx_factory()
    request = pb.PredictStageRequest(context=ctx)
    response = await echo_transport.call("Predict", request)
    assert isinstance(response, pb.PredictStageResponse), f"got {type(response)}"

    # Echo plugin reflects traffic into predictions; verify
    if ctx.HasField("observations") and ctx.observations.HasField("traffic"):
        assert (
            response.predictions.predicted_num_req == ctx.observations.traffic.num_req
        )
        assert response.predictions.predicted_isl == ctx.observations.traffic.isl
        assert response.predictions.predicted_osl == ctx.observations.traffic.osl
    assert response.predictions.source == "echo-server"


@pytest.mark.parametrize(
    "input_name,ctx_factory",
    _INPUTS,
    ids=[name for name, _ in _INPUTS],
)
@pytest.mark.asyncio
async def test_byte_equal_response_across_transports(
    input_name: str,
    ctx_factory,
    tmp_path: Path,
):
    """Response bytes from in_process / grpc must be **byte-identical**.

    This is the strongest possible contract — any silent semantic drift
    between transport implementations is caught here.
    """
    request = pb.PredictStageRequest(context=ctx_factory())

    # In-process
    t_inp = InProcessTransport("echo", EchoPluginInProcess(), timeout_seconds=2.0)
    try:
        resp_inp = await t_inp.call("Predict", request)
        bytes_inp = resp_inp.SerializeToString()
    finally:
        await t_inp.close()

    # gRPC insecure
    server_grpc, listen = await _start_grpc_server("127.0.0.1:0")
    try:
        t_grpc = GrpcTransport(
            "echo", f"grpc://{listen}", allow_insecure=True, timeout_seconds=2.0
        )
        try:
            resp_grpc = await t_grpc.call("Predict", request)
            bytes_grpc = resp_grpc.SerializeToString()
        finally:
            await t_grpc.close()
    finally:
        await server_grpc.stop(grace=0.1)

    assert (
        bytes_inp == bytes_grpc
    ), f"in_process vs grpc bytes differ for input {input_name!r}"


# ----------------------------------------------------------------------------
# Error contract: each transport raises typed errors for common failures
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("transport_kind", _TRANSPORT_KINDS, indirect=True)
@pytest.mark.asyncio
async def test_unknown_method_typed_error(
    echo_transport: PluginTransport, transport_kind: str
):
    """All transports must raise PluginUnknownMethodError for unregistered methods."""
    request = pb.ProposeStageRequest()  # different stage's request
    with pytest.raises(PluginUnknownMethodError):
        # Echo plugin only implements Predict; Propose should be UnknownMethod
        await echo_transport.call("Propose", request)


@pytest.mark.asyncio
async def test_unreachable_endpoint_raises_connection_error(tmp_path: Path):
    """gRPC: pointing transport at non-existent endpoint -> PluginConnectionError."""
    # Port not bound
    t = GrpcTransport(
        "noplug", "grpc://127.0.0.1:1", allow_insecure=True, timeout_seconds=0.5
    )
    try:
        with pytest.raises((PluginConnectionError, PluginTimeoutError)):
            await t.call("Predict", pb.PredictStageRequest())
    finally:
        await t.close()


@pytest.mark.parametrize("transport_kind", _TRANSPORT_KINDS, indirect=True)
@pytest.mark.asyncio
async def test_close_idempotent_all_transports(
    echo_transport: PluginTransport, transport_kind: str
):
    """All transports must satisfy two close()-related invariants:
    1. ``close()`` is idempotent (multiple calls don't raise).
    2. Subsequent ``call()`` raises ``PluginConnectionError`` — uniform
       contract across in-process and gRPC so the orchestrator can
       handle post-close mistakes the same way regardless of transport.
    """
    await echo_transport.close()
    await echo_transport.close()  # idempotent
    with pytest.raises(PluginConnectionError):
        await echo_transport.call("Predict", pb.PredictStageRequest())
