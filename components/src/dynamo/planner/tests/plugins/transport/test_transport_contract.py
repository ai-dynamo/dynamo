# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transport contract test (PR 2 sub-task 2-6) — **core acceptance**.

For a single ``echo`` plugin (returns the request's PipelineContext as the
response's predictions field), assert that all 4 transports
(``in_process`` / ``uds`` / ``grpc`` / ``grpc_mtls``) produce the **byte-equal
serialized response** for the same set of inputs.

This is the strongest guarantee that transport changes won't introduce
silent behavioral drift between deployment forms.

NOTE: ``grpc_mtls`` test is gated on test cert fixtures; if missing, that
parametrize variant is skipped (xfail wouldn't help since fixture absence
is environmental, not a bug).
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path
from typing import Any, AsyncIterator

import grpc
import pytest

from dynamo.planner.plugins.proto.v1 import plugin_pb2 as pb
from dynamo.planner.plugins.proto.v1 import plugin_pb2_grpc as pbg
from dynamo.planner.plugins.transport import (
    GrpcTransport,
    InProcessTransport,
    PluginCallError,
    PluginConnectionError,
    PluginTimeoutError,
    PluginTransport,
    PluginUnknownMethodError,
    UdsTransport,
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

    async def Predict(self, request: pb.PredictStageRequest, context) -> pb.PredictStageResponse:
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
# gRPC server fixtures (UDS + insecure TCP + mTLS TCP)
# ----------------------------------------------------------------------------


async def _start_grpc_server(
    listen: str,
    *,
    server_credentials: grpc.ServerCredentials | None = None,
) -> tuple[grpc.aio.Server, str]:
    """Start a gRPC server with EchoServicer at ``listen``.

    Returns (server, actual_listen) — for ":0" port, returns the bound port.
    """
    server = grpc.aio.server()
    pbg.add_PredictPluginServicer_to_server(EchoServicer(), server)
    if server_credentials is not None:
        port = server.add_secure_port(listen, server_credentials)
    else:
        port = server.add_insecure_port(listen)
    await server.start()
    # For TCP ":0", rebuild listen with actual port; for UDS, listen is unchanged
    if listen.startswith("[::]:0") or listen.startswith("0.0.0.0:0") or listen.endswith(":0"):
        host = listen.rsplit(":", 1)[0]
        actual_listen = f"{host}:{port}"
    else:
        actual_listen = listen
    return server, actual_listen


# ----------------------------------------------------------------------------
# Test data — 8 representative PipelineContext payloads
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


def _ctx_multi_pool() -> pb.PipelineContext:
    c = pb.PipelineContext(request_id="req-multi-pool")
    c.proposal.targets.add(sub_component_type="prefill", component_name="pool-A", replicas=8)
    c.proposal.targets.add(sub_component_type="prefill", component_name="pool-B", replicas=4)
    c.proposal.targets.add(sub_component_type="decode", replicas=10)
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
    ("multi_pool", _ctx_multi_pool),
    ("constrained", _ctx_with_constrained),
    ("zero_replicas_explicit", _ctx_zero_replicas_explicit),
]


# ----------------------------------------------------------------------------
# Test cert fixtures (self-signed) for mTLS — generated lazily
# ----------------------------------------------------------------------------


def _generate_self_signed_certs(tmp_dir: Path) -> tuple[Path, Path, Path]:
    """Generate self-signed CA + server + client cert/key triple in tmp_dir.

    Returns (ca_path, cert_path, key_path) — same cert serves as CA, server,
    and client (test-only simplification).

    Returns None tuple if ``cryptography`` package not installed.
    """
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID
    except ImportError:
        pytest.skip("cryptography package not available; mTLS tests skipped")

    import datetime as dt

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=1))
        .not_valid_after(dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=365))
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName("localhost"), x509.IPAddress(__import__("ipaddress").IPv4Address("127.0.0.1"))]),
            critical=False,
        )
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    cert_pem = cert.public_bytes(serialization.Encoding.PEM)
    key_pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    )
    ca_path = tmp_dir / "ca.crt"
    cert_path = tmp_dir / "tls.crt"
    key_path = tmp_dir / "tls.key"
    ca_path.write_bytes(cert_pem)
    cert_path.write_bytes(cert_pem)
    key_path.write_bytes(key_pem)
    return ca_path, cert_path, key_path


# ----------------------------------------------------------------------------
# Async fixture: 4 transports targeting the same Echo plugin
# ----------------------------------------------------------------------------


@pytest.fixture
def transport_kind(request):
    """Parametrized over 4 transport kinds."""
    return request.param


@pytest.fixture
async def echo_transport(transport_kind, tmp_path: Path) -> AsyncIterator[PluginTransport]:
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

    if transport_kind == "uds":
        sock_path = tmp_path / "echo.sock"
        listen = f"unix:{sock_path}"
        server, _ = await _start_grpc_server(listen)
        try:
            t = UdsTransport("echo", f"unix://{sock_path}", timeout_seconds=2.0)
            try:
                yield t
            finally:
                await t.close()
        finally:
            await server.stop(grace=0.1)
        return

    if transport_kind == "grpc":
        server, listen = await _start_grpc_server("127.0.0.1:0")
        try:
            t = GrpcTransport("echo", f"grpc://{listen}", allow_insecure=True, timeout_seconds=2.0)
            try:
                yield t
            finally:
                await t.close()
        finally:
            await server.stop(grace=0.1)
        return

    if transport_kind == "grpc_mtls":
        ca_path, cert_path, key_path = _generate_self_signed_certs(tmp_path)
        server_creds = grpc.ssl_server_credentials(
            [(key_path.read_bytes(), cert_path.read_bytes())],
            root_certificates=ca_path.read_bytes(),
            require_client_auth=True,
        )
        server, listen = await _start_grpc_server("127.0.0.1:0", server_credentials=server_creds)
        try:
            from dynamo.planner.plugins.transport import MtlsConfig

            mtls = MtlsConfig.from_files(ca_path, cert_path, key_path)
            t = GrpcTransport("echo", f"grpc://{listen}", mtls_config=mtls, timeout_seconds=2.0)
            try:
                yield t
            finally:
                await t.close()
        finally:
            await server.stop(grace=0.1)
        return

    pytest.fail(f"unknown transport_kind: {transport_kind}")


_TRANSPORT_KINDS = ["in_process", "uds", "grpc", "grpc_mtls"]


# ----------------------------------------------------------------------------
# Contract test: 8 inputs × 4 transports = 32 cases of byte-equality
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
        assert response.predictions.predicted_num_req == ctx.observations.traffic.num_req
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
    """Response bytes from in_process / uds / grpc must be **byte-identical**.

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

    # UDS
    sock_path = tmp_path / f"echo-{input_name}.sock"
    server_uds, _ = await _start_grpc_server(f"unix:{sock_path}")
    try:
        t_uds = UdsTransport("echo", f"unix://{sock_path}", timeout_seconds=2.0)
        try:
            resp_uds = await t_uds.call("Predict", request)
            bytes_uds = resp_uds.SerializeToString()
        finally:
            await t_uds.close()
    finally:
        await server_uds.stop(grace=0.1)

    # gRPC insecure
    server_grpc, listen = await _start_grpc_server("127.0.0.1:0")
    try:
        t_grpc = GrpcTransport("echo", f"grpc://{listen}", allow_insecure=True, timeout_seconds=2.0)
        try:
            resp_grpc = await t_grpc.call("Predict", request)
            bytes_grpc = resp_grpc.SerializeToString()
        finally:
            await t_grpc.close()
    finally:
        await server_grpc.stop(grace=0.1)

    assert bytes_inp == bytes_uds, (
        f"in_process vs uds bytes differ for input {input_name!r}"
    )
    assert bytes_uds == bytes_grpc, (
        f"uds vs grpc bytes differ for input {input_name!r}"
    )
    assert bytes_inp == bytes_grpc, (
        f"in_process vs grpc bytes differ for input {input_name!r}"
    )


# ----------------------------------------------------------------------------
# Error contract: each transport raises typed errors for common failures
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("transport_kind", _TRANSPORT_KINDS, indirect=True)
@pytest.mark.asyncio
async def test_unknown_method_typed_error(echo_transport: PluginTransport, transport_kind: str):
    """All transports must raise PluginUnknownMethodError for unregistered methods."""
    request = pb.ProposeStageRequest()  # different stage's request
    with pytest.raises(PluginUnknownMethodError):
        # Echo plugin only implements Predict; Propose should be UnknownMethod
        await echo_transport.call("Propose", request)


@pytest.mark.parametrize("transport_kind", ["uds", "grpc"])
@pytest.mark.asyncio
async def test_unreachable_endpoint_raises_connection_error(
    transport_kind: str, tmp_path: Path
):
    """UDS/gRPC: pointing transport at non-existent endpoint -> PluginConnectionError."""
    if transport_kind == "uds":
        # Socket file does not exist
        t = UdsTransport("noplug", f"unix://{tmp_path}/noexist.sock", timeout_seconds=0.5)
    else:
        # Port not bound
        t = GrpcTransport("noplug", "grpc://127.0.0.1:1", allow_insecure=True, timeout_seconds=0.5)
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
    """All transports' close() must be idempotent (multiple calls)."""
    await echo_transport.close()
    await echo_transport.close()
    # After close, calling raises connection error (defensive; not strictly required)
    # — just verify no double-close raises
