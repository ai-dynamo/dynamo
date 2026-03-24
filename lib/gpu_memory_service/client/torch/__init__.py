# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PyTorch integration for GPU Memory Service.

This module provides PyTorch-specific functionality:

- Memory manager singleton management
- Tensor utilities (metadata, registration, materialization)
- C++ extension for CUDAPluggableAllocator
"""

from gpu_memory_service.client.torch.allocator import (
    get_gms_client_memory_manager,
    get_or_create_gms_client_memory_manager,
)
from gpu_memory_service.client.torch.module import (
    materialize_module_from_gms,
    register_module_tensors,
)

__all__ = [
    # GMS client memory manager
    "get_or_create_gms_client_memory_manager",
    "get_gms_client_memory_manager",
    # Tensor operations (public API)
    "register_module_tensors",
    "materialize_module_from_gms",
]
