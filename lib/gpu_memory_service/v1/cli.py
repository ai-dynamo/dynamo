# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import logging
import signal
from collections.abc import Sequence
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import ExitStack
from threading import Event, Thread

from gpu_memory_service.common.vmm import VMMDeviceType, get_vmm, init_vmm
from gpu_memory_service.v1 import device as device_identity
from gpu_memory_service.v1 import isolation
from gpu_memory_service.v1.checkpoint import GMSCheckpointLifecycle
from gpu_memory_service.v1.device import get_socket_path
from gpu_memory_service.v1.server.rpc import GMSRPCServer, GMSServerMemoryManager

_DOMAINS = ("weights", "kv_cache")
logger = logging.getLogger(__name__)


def run_servers(
    servers: Sequence[GMSRPCServer],
    stop: Event | None = None,
) -> None:
    """Serve both domains; when either stops, stop the other and raise."""
    with ThreadPoolExecutor(
        max_workers=len(servers), thread_name_prefix="gms-v1"
    ) as executor:
        futures = [executor.submit(server.serve_forever) for server in servers]
        try:
            while True:
                done, _ = wait(
                    futures,
                    timeout=0.1 if stop is not None else None,
                    return_when=FIRST_COMPLETED,
                )
                if done:
                    for future in done:
                        future.result()
                    raise RuntimeError("GMS server stopped unexpectedly")
                if stop is not None and stop.is_set():
                    return
        finally:
            for server in servers:
                server.shutdown()


def _start_loader(
    loader_argv: list[str], device: int, rank: int, failed: Event, stop: Event
) -> Thread:
    """Fill this rank's weights server from inside the server process.

    Sharing the server's process saves the loader its own CUDA bring-up.
    Backend setup that needs no weights memory starts at once, overlapping
    the server's start.
    """
    from gpu_memory_service.v1.snapshot import loader

    args = loader.parse_args(
        [*loader_argv, "--device", str(device), "--rank", str(rank)]
    )
    prepared = None
    if args.transfer_backend == "mx":
        from gpu_memory_service.snapshot.backends.mx import prepare_nixl_manager

        prepared = ThreadPoolExecutor(1, thread_name_prefix="gms-v1-prepare").submit(
            prepare_nixl_manager, args.backend_params, device
        )

    def run() -> None:
        try:
            # The backend waits for the agent only when it transfers, so the
            # allocations happen while the agent is still coming up.
            extra = {} if prepared is None else {"mx_nixl_manager": prepared}
            loader.load(args, **extra)
            logger.info("GMS V1 loader complete rank=%d", rank)
        except BaseException:
            logger.exception("GMS V1 loader failed rank=%d", rank)
            failed.set()
            stop.set()

    thread = Thread(target=run, name="gms-v1-loader", daemon=True)
    thread.start()
    return thread


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="GMS V1 single-device server",
        allow_abbrev=False,
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help="Rank that names the sockets. Default: --device.",
    )
    parser.add_argument(
        "--enable-loader",
        nargs=argparse.REMAINDER,
        metavar="ARG",
        help="Load this rank's weights in-process; remaining args go to the loader.",
    )
    args = parser.parse_args(argv)
    rank = args.device if args.rank is None else args.rank
    # Before any thread starts, so every thread is confined.
    isolation.confine_from_env()

    stop = Event()
    loader_failed = Event()
    if args.enable_loader is not None:
        _start_loader(args.enable_loader, args.device, rank, loader_failed, stop)

    init_vmm(VMMDeviceType.CUDA)
    vmm = get_vmm()
    gpu_uuid = device_identity.get_device_uuid(args.device)
    with ExitStack() as stack:
        checkpoint_lifecycle = GMSCheckpointLifecycle()
        managers = {
            domain: GMSServerMemoryManager(
                gpu_uuid,
                vmm,
                args.device,
                checkpoint_lifecycle=checkpoint_lifecycle,
            )
            for domain in _DOMAINS
        }
        checkpoint_lifecycle.bind_domains(managers)
        servers = [
            stack.enter_context(
                GMSRPCServer(
                    get_socket_path(rank, domain),
                    managers[domain],
                )
            )
            for domain in _DOMAINS
        ]
        logger.info("GMS V1 server listening device=%d rank=%d", args.device, rank)

        def terminate(*_args) -> None:
            stop.set()

        signal.signal(signal.SIGTERM, terminate)
        signal.signal(signal.SIGINT, terminate)
        run_servers(servers, stop)
    if loader_failed.is_set():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
