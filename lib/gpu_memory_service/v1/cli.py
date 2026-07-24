# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse

from gpu_memory_service.common.utils import get_socket_path
from gpu_memory_service.common.vmm import get_vmm

from .server.allocations import AllocationStore
from .server.rpc import AllocationRPCServer


def _gpu_uuid(device: int) -> str:
    import torch

    return str(torch.cuda.get_device_properties(device).uuid)


def main() -> None:
    parser = argparse.ArgumentParser(description="snapshot-only GMS V1 sidecar")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--socket-path")
    args = parser.parse_args()
    path = args.socket_path or get_socket_path(args.device, "snapshot-v1")
    vmm = get_vmm()
    allocations = AllocationStore(_gpu_uuid(args.device), vmm, args.device)
    with AllocationRPCServer(path, allocations) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
