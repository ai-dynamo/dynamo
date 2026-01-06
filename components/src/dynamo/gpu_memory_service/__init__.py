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
    StaleWeightsError,
    get_allocator,
    get_or_create_allocator,
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

__all__ = [
    # Core allocator
    "GMSClientMemoryManager",
    "StaleWeightsError",
    # Lifecycle management
    "get_or_create_allocator",
    "get_allocator",
    # Tensor utilities
    "register_module_tensors",
    "materialize_module_from_gms",
    # Extensions
    "_allocator_ext",
    "_tensor_from_pointer",
]
