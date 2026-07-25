# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared Torch allocator construction and tensor-isolation primitives."""

from .allocator import (
    TorchAllocatorCallbacks,
    create_torch_allocator,
    create_torch_mem_pool,
)
from .tensor import isolate_tensors

__all__ = [
    "TorchAllocatorCallbacks",
    "create_torch_allocator",
    "create_torch_mem_pool",
    "isolate_tensors",
]
