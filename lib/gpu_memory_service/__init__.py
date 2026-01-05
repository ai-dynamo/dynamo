# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU Memory Service - out-of-process GPU memory manager.

The GPU Memory Service decouples ownership of GPU memory from the processes
that use it, enabling zero-copy sharing and data survival across process crashes.

Package structure:
- common/: Shared types and protocol (used by both server and client)
- server/: Allocation server daemon (no CUDA context required)
- client/: Client library for memory management
  - client/torch/: PyTorch integration (lifecycle, tensor utilities, extensions)

Primary client API:
    from gpu_memory_service import (
        GMSClientMemoryManager,
        get_or_create_allocator,
        get_allocator,
    )

Server API:
    from gpu_memory_service.server import GMSRPCServer
"""

# Primary client exports
from gpu_memory_service.client.memory_manager import (
    GMSClientMemoryManager,
    StaleWeightsError,
)
from gpu_memory_service.client.rpc import GMSRPCClient

# PyTorch integration (lifecycle management)
from gpu_memory_service.client.torch.lifecycle import (
    clear_allocator,
    get_allocator,
    get_mem_pool,
    get_or_create_allocator,
    register_allocator,
)

# Common types
from gpu_memory_service.common.types import ConnectionMode, ServerState

__all__ = [
    # Client
    "GMSClientMemoryManager",
    "StaleWeightsError",
    "GMSRPCClient",
    # Lifecycle
    "get_or_create_allocator",
    "get_allocator",
    "get_mem_pool",
    "register_allocator",
    "clear_allocator",
    # Types
    "ConnectionMode",
    "ServerState",
]
