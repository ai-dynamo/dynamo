# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU Memory Service - CUDA VMM-based GPU memory allocation and sharing."""

from gpu_memory_service.allocator import RPCCumemAllocator, StaleWeightsError
from gpu_memory_service.lifecycle import (
    clear_allocator,
    get_allocator,
    get_mem_pool,
    get_or_create_allocator,
    register_allocator,
)

__all__ = [
    "RPCCumemAllocator",
    "StaleWeightsError",
    "get_or_create_allocator",
    "get_allocator",
    "get_mem_pool",
    "register_allocator",
    "clear_allocator",
]
