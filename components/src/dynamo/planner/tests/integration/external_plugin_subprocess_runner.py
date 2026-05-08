# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone external plugin process — driven by
``test_external_plugin_subprocess_e2e.py``.

Runs as a separate Python process (different PID, different address
space, independent gRPC channel state) so the e2e test can exercise:

- real cross-process serialization
- subprocess crash isolation (the test ``kill -9``'s this process and
  asserts the planner's circuit breaker recovers)
- self-registration via the W2 gateway's ``Register`` RPC

Stage support
-------------

``--stage`` selects which plugin contract this process serves; the
e2e test driver spawns one runner per stage to exercise the full
pipeline over a real wire:

- ``predict``    — ``PredictPluginServicer.Predict`` returns a fixed
                   ``PredictionData`` (chain-augment terminator).
- ``propose``    — ``ProposePluginServicer.Propose`` returns a fixed
                   ``OverrideResult`` (default; original behaviour).
- ``reconcile``  — ``ReconcilePluginServicer.Reconcile`` returns a
                   fixed ``OverrideResult``.
- ``constrain``  — ``ConstrainPluginServicer.Constrain`` returns
                   ``AT_MOST`` ceilings.

Output discipline matters: the test reads the first stdout line as a
ready signal, so anything else (logs, warnings) goes to stderr.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys

import grpc

from dynamo.planner.plugins.proto.v1 import plugin_pb2 as pb
from dynamo.planner.plugins.proto.v1 import plugin_pb2_grpc as pbg


# ---------------------------------------------------------------------------
# Per-stage Servicer implementations
#
# Each servicer is deterministic so the e2e test can assert exact
# decision values landed in PipelineOutcome — no randomness, no
# context-dependent branches.
# ---------------------------------------------------------------------------


class _DeterministicPredictPlugin(pbg.PredictPluginServicer):
    """Returns a fixed ``PredictionData``. ``final=True`` to terminate
    the chain (lowest-priority plugin in the chain-augment order)."""

    def __init__(self, *, num_req: float, isl: float, osl: float) -> None:
        self._num_req = num_req
        self._isl = isl
        self._osl = osl

    async def Predict(
        self,
        request: pb.PredictStageRequest,
        context: grpc.aio.ServicerContext,
    ) -> pb.PredictStageResponse:
        resp = pb.PredictStageResponse()
        resp.predictions.predicted_num_req = self._num_req
        resp.predictions.predicted_isl = self._isl
        resp.predictions.predicted_osl = self._osl
        resp.predictions.source = "subprocess_external_predict"
        resp.final = True
        return resp


class _DeterministicProposePlugin(pbg.ProposePluginServicer):
    """Returns a fixed OverrideResult on every call so the e2e test
    can assert exact target replicas."""

    def __init__(self, *, prefill: int, decode: int) -> None:
        self._prefill = prefill
        self._decode = decode

    async def Propose(
        self,
        request: pb.ProposeStageRequest,
        context: grpc.aio.ServicerContext,
    ) -> pb.ProposeStageResponse:
        resp = pb.ProposeStageResponse()
        ovr = resp.override
        ovr.reason = "subprocess_external_propose"
        for sub, n in (("prefill", self._prefill), ("decode", self._decode)):
            t = ovr.targets.add()
            t.sub_component_type = sub
            t.replicas = n
            t.type = pb.OverrideType.SET
        return resp


class _DeterministicReconcilePlugin(pbg.ReconcilePluginServicer):
    """RECONCILE-stage fixed-decision plugin. Re-shapes whatever
    PROPOSE produced (or injects from scratch when no PROPOSE)."""

    def __init__(self, *, prefill: int, decode: int) -> None:
        self._prefill = prefill
        self._decode = decode

    async def Reconcile(
        self,
        request: pb.ReconcileStageRequest,
        context: grpc.aio.ServicerContext,
    ) -> pb.ReconcileStageResponse:
        resp = pb.ReconcileStageResponse()
        ovr = resp.override
        ovr.reason = "subprocess_external_reconcile"
        for sub, n in (("prefill", self._prefill), ("decode", self._decode)):
            t = ovr.targets.add()
            t.sub_component_type = sub
            t.replicas = n
            t.type = pb.OverrideType.SET
        return resp


class _DeterministicConstrainPlugin(pbg.ConstrainPluginServicer):
    """CONSTRAIN-stage fixed-ceiling plugin. Emits AT_MOST so the
    test can validate that ceilings clamp PROPOSE/RECONCILE outputs
    over the wire (SET would be silently dropped per v11 contract)."""

    def __init__(self, *, ceiling_prefill: int, ceiling_decode: int) -> None:
        self._cp = ceiling_prefill
        self._cd = ceiling_decode

    async def Constrain(
        self,
        request: pb.ConstrainStageRequest,
        context: grpc.aio.ServicerContext,
    ) -> pb.ConstrainStageResponse:
        resp = pb.ConstrainStageResponse()
        ovr = resp.override
        ovr.reason = "subprocess_external_constrain"
        for sub, n in (("prefill", self._cp), ("decode", self._cd)):
            t = ovr.targets.add()
            t.sub_component_type = sub
            t.replicas = n
            t.type = pb.OverrideType.AT_MOST
        return resp


# Stage-name → (servicer factory, register_to_server fn, default plugin_id) ---

_STAGE_TABLE = {
    "predict": (
        lambda args: _DeterministicPredictPlugin(
            num_req=args.predict_num_req,
            isl=args.predict_isl,
            osl=args.predict_osl,
        ),
        pbg.add_PredictPluginServicer_to_server,
        "external-subprocess-predict",
    ),
    "propose": (
        lambda args: _DeterministicProposePlugin(
            prefill=args.prefill, decode=args.decode
        ),
        pbg.add_ProposePluginServicer_to_server,
        "external-subprocess-propose",
    ),
    "reconcile": (
        lambda args: _DeterministicReconcilePlugin(
            prefill=args.prefill, decode=args.decode
        ),
        pbg.add_ReconcilePluginServicer_to_server,
        "external-subprocess-reconcile",
    ),
    "constrain": (
        lambda args: _DeterministicConstrainPlugin(
            ceiling_prefill=args.prefill,
            ceiling_decode=args.decode,
        ),
        pbg.add_ConstrainPluginServicer_to_server,
        "external-subprocess-constrain",
    ),
}


async def _self_register(
    *,
    gateway_endpoint: str,
    plugin_id: str,
    plugin_type: str,
    plugin_listen: str,
    auth_token: str,
    priority: int,
) -> None:
    """Open a gRPC client to the gateway and call Register so the
    planner picks us up. ``unix:`` and ``host:port`` both supported by
    the standard gRPC channel constructor."""
    if gateway_endpoint.startswith("unix://"):
        target = gateway_endpoint.replace("unix://", "unix:")
    elif gateway_endpoint.startswith("grpc://"):
        target = gateway_endpoint[len("grpc://"):]
    else:
        # Accept bare host:port as well — caller convenience.
        target = gateway_endpoint
    async with grpc.aio.insecure_channel(target) as channel:
        stub = pbg.PluginRegistryStub(channel)
        req = pb.RegisterRequest(
            plugin_id=plugin_id,
            plugin_type=plugin_type,
            priority=priority,
            endpoint=plugin_listen,
            auth_token=auth_token,
            protocol_version="1.0",
            execution_interval_seconds=0.0,
            hold_policy=pb.HoldPolicy.HOLD_LAST,
            version="v1",
        )
        resp = await stub.Register(req)
        if not resp.accepted:
            raise SystemExit(
                f"subprocess plugin self-register rejected: {resp.reject_reason!r}"
            )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--listen",
        required=True,
        help="bind address for this plugin's gRPC server "
        "(e.g. ``unix:/tmp/p.sock`` or ``127.0.0.1:0``)",
    )
    parser.add_argument(
        "--stage",
        choices=sorted(_STAGE_TABLE.keys()),
        default="propose",
        help="which plugin stage this runner serves",
    )
    parser.add_argument(
        "--plugin-id",
        default=None,
        help="plugin_id used during self-registration; defaults to a "
        "stage-specific name when omitted",
    )
    parser.add_argument(
        "--priority",
        type=int,
        default=5,
        help="plugin priority used during self-registration. PREDICT "
        "wants priority=1 (lowest = chain terminator); RECONCILE / "
        "CONSTRAIN typically use 1 too. PROPOSE merge picks smallest "
        "first, so for PROPOSE pick 4–10.",
    )
    parser.add_argument(
        "--gateway-endpoint",
        default="",
        help="if set, self-register via this gateway after starting "
        "(e.g. ``grpc://127.0.0.1:7777`` or ``unix:///var/run/dynamo.sock``)",
    )
    parser.add_argument("--auth-token", default="anything")
    # PROPOSE / RECONCILE / CONSTRAIN reuse these:
    parser.add_argument("--prefill", type=int, default=7)
    parser.add_argument("--decode", type=int, default=11)
    # PREDICT-only knobs:
    parser.add_argument("--predict-num-req", type=float, default=1234.0)
    parser.add_argument("--predict-isl", type=float, default=567.0)
    parser.add_argument("--predict-osl", type=float, default=89.0)
    args = parser.parse_args()

    factory, attach_to_server, default_id = _STAGE_TABLE[args.stage]
    if args.plugin_id is None:
        args.plugin_id = default_id

    # Send all logs to stderr — stdout is reserved for the
    # ``LISTEN_READY`` ready signal so the test driver can read it
    # synchronously without log noise.
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)

    server = grpc.aio.server()
    attach_to_server(factory(args), server)
    port = server.add_insecure_port(args.listen)
    await server.start()

    actual_listen = args.listen
    if args.listen.endswith(":0"):
        actual_listen = f"{args.listen.rsplit(':', 1)[0]}:{port}"

    # Plugin endpoint as the planner will see it (matches scheme
    # convention used by ``derive_transport_type``).
    if actual_listen.startswith("unix:"):
        plugin_endpoint_for_planner = "unix://" + actual_listen[len("unix:"):]
    else:
        plugin_endpoint_for_planner = "grpc://" + actual_listen

    if args.gateway_endpoint:
        await _self_register(
            gateway_endpoint=args.gateway_endpoint,
            plugin_id=args.plugin_id,
            plugin_type=args.stage,
            plugin_listen=plugin_endpoint_for_planner,
            auth_token=args.auth_token,
            priority=args.priority,
        )

    print(f"LISTEN_READY {plugin_endpoint_for_planner}", flush=True)

    stop = asyncio.Event()
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    await stop.wait()
    await server.stop(grace=0.5)


if __name__ == "__main__":
    asyncio.run(main())
