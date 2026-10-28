# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serve a GMS V1 server's committed weights to ModelExpress peers.

A restoring server fills its allocations from this one with the ``mx``
transfer backend. The weights stay owned by the server; this process holds a
read-only lease, maps them, registers them with ModelExpress's NIXL agent and
publishes them, one byte tensor per allocation in creation order, until it is
stopped.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading

from gpu_memory_service.cli.snapshot import run_per_device
from gpu_memory_service.common.locks import GrantedLockType, RequestedLockType
from gpu_memory_service.common.vmm import VMMDeviceType, get_vmm, init_vmm
from gpu_memory_service.snapshot.backends.mx import byte_tensors, mx_load_context
from gpu_memory_service.v1 import device as device_identity
from gpu_memory_service.v1.client.session import _GMSClientSession
from gpu_memory_service.v1.snapshot.weight_artifact import (
    _Cleanup,
    _list_allocations,
    _map_export,
    _release_mapping,
    _verify_session_device,
)

logger = logging.getLogger(__name__)


def serve(
    socket_path: str,
    device: int,
    rank: int,
    backend_config: dict,
    stop: threading.Event,
) -> None:
    import torch
    from modelexpress.load_strategy.base import publish_metadata, register_tensors

    vmm = get_vmm()
    vmm.ensure_initialized()
    vmm.runtime_set_device(device)
    torch.cuda.set_device(device)
    granularity = int(vmm.get_allocation_granularity(device))
    with _Cleanup() as resources:
        session = _GMSClientSession(socket_path, RequestedLockType.RO)
        resources.callback(session.close)
        _verify_session_device(session, device)
        records = _list_allocations(session)
        if not records:
            raise RuntimeError("GMS V1 weights server has no committed allocations")
        extents = []
        for record in records:
            mapping = _map_export(
                session, record, vmm, device, granularity, GrantedLockType.RO
            )
            resources.callback(_release_mapping, vmm, mapping)
            extents.append((mapping[0].base, record.aligned_size))
        ctx = mx_load_context(
            backend_config, rank, device, byte_tensors(extents, device)
        )
        register_tensors(None, ctx, reuse_discovered=True)
        if ctx.nixl_manager is None:
            raise RuntimeError("ModelExpress NIXL registration failed")
        resources.callback(ctx.nixl_manager.shutdown)
        publish_metadata(ctx)
        logger.info(
            "Serving rank %d from device %d: %d allocations, %.2f GiB",
            rank,
            device,
            len(records),
            sum(r.aligned_size for r in records) / 2**30,
        )
        stop.wait()


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(
        description="Serve committed GMS V1 weights to ModelExpress peers."
    )
    parser.add_argument(
        "--device",
        type=int,
        help="GMS device whose weights to serve. Default: every device, one process each.",
    )
    parser.add_argument(
        "--source-key",
        required=True,
        help="names these weights to restoring peers, e.g. their checkpoint",
    )
    parser.add_argument(
        "--mx-server-url",
        help=(
            "ModelExpress server host:port. Default: the ModelExpress client's "
            "own setting, the MX_SERVER_ADDRESS environment variable."
        ),
    )
    # python -m passes no argv, and run_per_device forwards argv verbatim to
    # the per-device processes, so substitute the real command line first.
    argv = argv if argv is not None else sys.argv[1:]
    args = parser.parse_args(argv)
    if args.device is None:
        run_per_device("gpu_memory_service.v1.snapshot.mx_source", argv)
        return
    init_vmm(VMMDeviceType.CUDA)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    gpu_uuid = device_identity.gpu_uuid(args.device)
    serve(
        device_identity.get_socket_path(gpu_uuid, "weights"),
        device_identity.cuda_ordinal(gpu_uuid),
        args.device,
        {"mx_source_key": args.source_key, "mx_server_url": args.mx_server_url},
        stop,
    )


if __name__ == "__main__":
    main()
