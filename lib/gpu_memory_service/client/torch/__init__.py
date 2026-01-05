# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PyTorch integration for GPU Memory Service.

This module provides PyTorch-specific functionality:

- Lifecycle management (singleton allocator, MemPool setup)
- Tensor utilities (metadata, registration, materialization)
- C++ extensions (CUDAPluggableAllocator, tensor_from_pointer)
"""

from gpu_memory_service.client.torch.lifecycle import (
    clear_allocator,
    get_allocator,
    get_mem_pool,
    get_or_create_allocator,
    register_allocator,
)
from gpu_memory_service.client.torch.tensor import (
    ModuleTreeNode,
    ParsedTensorMeta,
    TensorIPCInfo,
    TensorMeta,
    TensorSpec,
    extract_module_tree,
    iter_module_tree_tensors,
    load_tensor_specs,
    materialize_module_from_gms,
    register_module_tensors,
    register_tensor,
    resolve_module_attr,
    tensor_from_spec,
)

__all__ = [
    # Lifecycle
    "get_or_create_allocator",
    "get_allocator",
    "get_mem_pool",
    "register_allocator",
    "clear_allocator",
    # Tensor types
    "TensorMeta",
    "ParsedTensorMeta",
    "TensorIPCInfo",
    "ModuleTreeNode",
    "TensorSpec",
    # Tensor operations
    "extract_module_tree",
    "iter_module_tree_tensors",
    "resolve_module_attr",
    "load_tensor_specs",
    "register_tensor",
    "register_module_tensors",
    "tensor_from_spec",
    "materialize_module_from_gms",
]
