# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU Memory Service component for Dynamo.

This module provides the Dynamo component wrapper around the gpu_memory_service package.
The core functionality is in the gpu_memory_service package; this module provides:
- CLI entry point (python -m dynamo.gpu_memory_service)
- Re-exports for backwards compatibility
"""

# Re-export core functionality from gpu_memory_service package
from gpu_memory_service import (
    GMSClientMemoryManager,
    StaleMemoryLayoutError,
    get_gms_client_memory_manager,
    get_or_create_gms_client_memory_manager,
)

# Re-export extensions (built separately)
try:
    from gpu_memory_service.client.torch.extensions import _allocator_ext
except (ImportError, OSError):
    _allocator_ext = None

# Re-export module utilities
from gpu_memory_service.client.torch.module import (
    materialize_module_from_gms,
    register_module_tensors,
)

__all__ = [
    # Core
    "GMSClientMemoryManager",
    "StaleMemoryLayoutError",
    # GMS client memory manager
    "get_or_create_gms_client_memory_manager",
    "get_gms_client_memory_manager",
    # Tensor utilities
    "register_module_tensors",
    "materialize_module_from_gms",
    # Extensions
    "_allocator_ext",
]
