# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU Memory Service client library.

This module provides the client-side components for interacting with the
GPU Memory Service:

- GMSClientMemoryManager: Manages local VA mappings of remote GPU memory

For PyTorch integration (MemPool, tensor utilities), see gpu_memory_service.client.torch.
"""

from gpu_memory_service.client.memory_manager import (
    GMSClientMemoryManager,
    StaleMemoryLayoutError,
)

__all__ = [
    "GMSClientMemoryManager",
    "StaleMemoryLayoutError",
]
