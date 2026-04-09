# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU Memory Service - out-of-process GPU memory manager.

The GPU Memory Service decouples ownership of GPU memory from the processes
that use it, enabling zero-copy sharing and data survival across process crashes.

Package structure:
- common/: Shared types and protocol (used by both server and client)
- server/: Allocation server daemon (no CUDA context required)
- client/: Client library for memory management
  - client/torch/: PyTorch integration (allocator, tensor, module, extensions)

Primary client API:
    from gpu_memory_service import (
        GMSClientMemoryManager,
        get_or_create_gms_client_memory_manager,
        get_gms_client_memory_manager,
    )

Server API:
    from gpu_memory_service.server import GMSRPCServer
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gpu_memory_service.client.memory_manager import (
        GMSClientMemoryManager,
        StaleMemoryLayoutError,
    )
    from gpu_memory_service.client.torch.allocator import (
        get_gms_client_memory_manager,
        get_gms_client_memory_managers,
        get_or_create_gms_client_memory_manager,
        gms_use_mem_pool,
    )

__all__ = [
    # Client
    "GMSClientMemoryManager",
    "StaleMemoryLayoutError",
    # GMS client memory manager
    "get_or_create_gms_client_memory_manager",
    "get_gms_client_memory_manager",
    "get_gms_client_memory_managers",
    "gms_use_mem_pool",
]


def __getattr__(name: str) -> Any:
    if name in {"GMSClientMemoryManager", "StaleMemoryLayoutError"}:
        from gpu_memory_service.client.memory_manager import (
            GMSClientMemoryManager,
            StaleMemoryLayoutError,
        )

        return {
            "GMSClientMemoryManager": GMSClientMemoryManager,
            "StaleMemoryLayoutError": StaleMemoryLayoutError,
        }[name]

    if name in {
        "get_or_create_gms_client_memory_manager",
        "get_gms_client_memory_manager",
        "get_gms_client_memory_managers",
        "gms_use_mem_pool",
    }:
        from gpu_memory_service.client.torch.allocator import (
            get_gms_client_memory_manager,
            get_gms_client_memory_managers,
            get_or_create_gms_client_memory_manager,
            gms_use_mem_pool,
        )

        return {
            "get_or_create_gms_client_memory_manager": get_or_create_gms_client_memory_manager,
            "get_gms_client_memory_manager": get_gms_client_memory_manager,
            "get_gms_client_memory_managers": get_gms_client_memory_managers,
            "gms_use_mem_pool": gms_use_mem_pool,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
