# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU Memory Service client library.

This module provides the client-side components for interacting with the
GPU Memory Service:

- GMSClientMemoryManager: Manages local VA mappings of remote GPU memory
- GMSStorageClient: Save and restore GMS state to disk

For PyTorch integration (MemPool, tensor utilities), see gpu_memory_service.client.torch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gpu_memory_service.client.gms_storage_client import GMSStorageClient, SaveManifest
    from gpu_memory_service.client.memory_manager import (
        GMSClientMemoryManager,
        StaleMemoryLayoutError,
    )

__all__ = [
    "GMSClientMemoryManager",
    "StaleMemoryLayoutError",
    "GMSStorageClient",
    "SaveManifest",
]


def __getattr__(name: str) -> Any:
    if name in {"GMSClientMemoryManager", "StaleMemoryLayoutError"}:
        from gpu_memory_service.client.memory_manager import (
            GMSClientMemoryManager,
            StaleMemoryLayoutError,
        )

        return {
            "GMSClientMemoryManager": GMSClientMemoryManager,
            "StaleMemoryLayoutError": StaleMemoryLayoutError,
        }[name]

    if name in {"GMSStorageClient", "SaveManifest"}:
        from gpu_memory_service.client.gms_storage_client import (
            GMSStorageClient,
            SaveManifest,
        )

        return {
            "GMSStorageClient": GMSStorageClient,
            "SaveManifest": SaveManifest,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
