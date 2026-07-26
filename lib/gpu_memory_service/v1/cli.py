# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import threading
from contextlib import ExitStack

import torch
from gpu_memory_service.common.utils import get_socket_path
from gpu_memory_service.common.vmm import get_vmm
from gpu_memory_service.core.server.allocations import GMSAllocationManager
from gpu_memory_service.core.server.gms import GMS
from gpu_memory_service.core.server.rpc import GMSRPCServer

_DOMAINS = ("weights", "kv_cache")


def _gpu_uuid(device: int) -> str:
    return str(torch.cuda.get_device_properties(device).uuid)


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
                    GMS(
                        gpu_uuid,
                        GMSAllocationManager(vmm, args.device),
                    ),
                )
            )
            for domain in _DOMAINS
        ]
        kv_thread = threading.Thread(
            target=servers[1].serve_forever,
            name="gms-v1-kv-cache",
            daemon=True,
        )
        kv_thread.start()
        try:
            servers[0].serve_forever()
        finally:
            servers[1].shutdown()
            kv_thread.join()


if __name__ == "__main__":
    main()
