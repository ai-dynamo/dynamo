# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU Memory Service component for Dynamo.

This module provides the Dynamo component wrapper around the gpu_memory_service package.
The core functionality is in the gpu_memory package; this module provides:
- CLI entry point (python -m dynamo.gpu_memory_service)
- Re-exports for backwards compatibility
"""

# Re-export core functionality from gpu_memory_service package
from gpu_memory_service import (
    GMSClientMemoryManager,
    GMSRPCClient,
    StaleWeightsError,
    clear_allocator,
    get_allocator,
    get_mem_pool,
    get_or_create_allocator,
    register_allocator,
)

# Re-export extensions (built separately)
from gpu_memory_service.client.torch.extensions import (
    _allocator_ext,
    _tensor_from_pointer,
)

# Re-export tensor utilities
from gpu_memory_service.client.torch.tensor import (
    materialize_module_from_gms,
    register_module_tensors,
)

# Re-export server
from gpu_memory_service.server import GMSRPCServer

__all__ = [
    # Core allocator
    "GMSClientMemoryManager",
    "StaleWeightsError",
    "GMSRPCClient",
    # Lifecycle management
    "get_or_create_allocator",
    "get_allocator",
    "get_mem_pool",
    "register_allocator",
    "clear_allocator",
    # Server components
    "GMSRPCServer",
    # Tensor utilities
    "register_module_tensors",
    "materialize_module_from_gms",
    # Extensions
    "_allocator_ext",
    "_tensor_from_pointer",
]
