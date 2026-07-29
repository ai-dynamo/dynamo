# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import signal
from collections.abc import Sequence
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import ExitStack
from threading import Event

import torch
from gpu_memory_service.common.utils import get_socket_path
from gpu_memory_service.common.vmm import get_vmm
from gpu_memory_service.core.server.gms import GMSServerMemoryManager
from gpu_memory_service.core.server.rpc import GMSRPCServer

_DOMAINS = ("weights", "kv_cache")


def _gpu_uuid(device: int) -> str:
    return str(torch.cuda.get_device_properties(device).uuid)


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


def main() -> None:
    parser = argparse.ArgumentParser(description="GMS V1 rank-local sidecar")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    vmm = get_vmm()
    gpu_uuid = _gpu_uuid(args.device)
    with ExitStack() as stack:
        servers = [
            stack.enter_context(
                GMSRPCServer(
                    get_socket_path(args.device, domain),
                    GMSServerMemoryManager(gpu_uuid, vmm, args.device),
                )
            )
            for domain in _DOMAINS
        ]
        stop = Event()

        def terminate(*_args) -> None:
            stop.set()

        signal.signal(signal.SIGTERM, terminate)
        signal.signal(signal.SIGINT, terminate)
        run_servers(servers, stop)


if __name__ == "__main__":
    main()
