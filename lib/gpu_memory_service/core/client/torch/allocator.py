# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Torch CUDAPluggableAllocator and MemPool construction."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any


class TorchAllocatorCallbacks:
    """Adapt Python allocation operations to Torch's callback ABI."""

    def __init__(
        self,
        malloc: Callable[[int, int, int], int],
        free: Callable[[int, int, int, int], None],
    ):
        self._malloc = malloc
        self._free = free
        self.failure: Exception | None = None
        self._failure_lock = threading.Lock()

    def malloc(self, size: int, device: int, stream: int) -> int:
        try:
            return self._malloc(size, device, stream)
        except Exception as exc:
            self._record_failure(exc)
            raise

    def free(self, base: int, size: int, device: int, stream: int) -> None:
        try:
            self._free(base, size, device, stream)
        except Exception as exc:
            self._record_failure(exc)

    def get_failure(self) -> Exception | None:
        with self._failure_lock:
            return self.failure

    def _record_failure(self, failure: Exception) -> None:
        with self._failure_lock:
            self.failure = self.failure or failure


def create_torch_allocator(
    callbacks: TorchAllocatorCallbacks,
) -> Any:
    """Register one callback owner and construct its Torch allocator."""
    import torch
    from gpu_memory_service.core.client.torch.extensions import _allocator_ext

    if _allocator_ext is None:
        raise RuntimeError("GPU Memory Service allocator extension is not built")
    _allocator_ext.init_module(callbacks.malloc, callbacks.free)
    return torch.cuda.CUDAPluggableAllocator(
        _allocator_ext.__file__, "my_malloc", "my_free"
    )


def create_torch_mem_pool(allocator: Any) -> Any:
    """Construct a Torch MemPool retaining the supplied allocator."""
    import torch

    return torch.cuda.MemPool(allocator=allocator.allocator())
