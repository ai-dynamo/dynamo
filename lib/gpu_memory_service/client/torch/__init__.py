# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PyTorch integration for GPU Memory Service.

This module provides PyTorch-specific functionality:

- Lifecycle management (singleton allocator, MemPool setup)
- Tensor utilities (metadata, registration, materialization)
- C++ extensions (CUDAPluggableAllocator, tensor_from_pointer)
"""

from gpu_memory_service.client.torch.lifecycle import (
    get_allocator,
    get_or_create_allocator,
)
from gpu_memory_service.client.torch.tensor import (
    materialize_module_from_gms,
    register_module_tensors,
)

__all__ = [
    # Lifecycle
    "get_or_create_allocator",
    "get_allocator",
    # Tensor operations (public API)
    "register_module_tensors",
    "materialize_module_from_gms",
]
