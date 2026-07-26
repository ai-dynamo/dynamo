# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Torch MemPools for V1 Parameter and KV-cache policies."""

from __future__ import annotations

import gc
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

import torch
from gpu_memory_service.core.client.torch.allocator import (
    TorchAllocatorCallbacks,
    create_torch_allocator,
    create_torch_mem_pool,
)
from gpu_memory_service.core.errors import GMSError
from gpu_memory_service.v1.memory_manager import (
    EphemeralKVCacheMemoryManager,
    PersistentParameterMemoryManager,
)
from gpu_memory_service.v1.tensor import normalize_captured_tensors

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

logger = logging.getLogger(__name__)

_WEIGHTS = "weights"
_KV_CACHE = "kv_cache"


class V1TorchPools:
    """Route two Torch MemPools through one pluggable allocator callback owner."""

    def __init__(
        self,
        weights: PersistentParameterMemoryManager,
        kv_cache: EphemeralKVCacheMemoryManager,
    ):
        if weights.device != kv_cache.device:
            raise ValueError("V1 weights and KV cache must use the same device")
        self.weights = weights
        self.kv_cache = kv_cache
        self.device = weights.device
        self._active_domain: ContextVar[str | None] = ContextVar(
            "gms_v1_active_domain",
            default=None,
        )
        self._allocator = TorchAllocatorCallbacks(self._malloc, self._free)
        self._pluggable_allocator = create_torch_allocator(self._allocator)
        with torch.cuda.device(self.device):
            self._weights_pool = create_torch_mem_pool(self._pluggable_allocator)
            self._kv_cache_pool = create_torch_mem_pool(self._pluggable_allocator)

    @contextmanager
    def capture_weights(self, model: "Callable[[], object]") -> "Iterator[None]":
        """Capture broad weight loading and commit its Parameter backing."""
        try:
            with self._use_pool(_WEIGHTS, self._weights_pool):
                yield
        except Exception as cause:
            self.weights.abort(cause)

        self._finalize_weights(model)

    @contextmanager
    def capture_kv_cache(self) -> "Iterator[None]":
        """Allocate vLLM KV tensors from the ephemeral GMS domain."""
        with self._use_pool(_KV_CACHE, self._kv_cache_pool):
            yield
        self.raise_if_allocator_failed()

    def raise_if_allocator_failed(self) -> None:
        failure = self._allocator.get_failure()
        if failure is not None:
            raise GMSError("allocator free callback failed") from failure

    @contextmanager
    def _use_pool(self, domain: str, pool: object) -> "Iterator[None]":
        token = self._active_domain.set(domain)
        try:
            with torch.cuda.device(self.device):
                with torch.cuda.use_mem_pool(pool, device=self.device):
                    yield
        finally:
            self._active_domain.reset(token)

    def _finalize_weights(self, model: "Callable[[], object]") -> None:
        try:
            (
                retained_gms_parameter_span_bytes,
                copied_out_bytes,
            ) = normalize_captured_tensors(model(), self.weights.mappings)
            torch.cuda.synchronize(self.device)
            self._destroy_weights_pool()
        except Exception as cause:
            self.weights.abort(cause)

        self.weights.commit()
        retained_gms_allocated_bytes = sum(
            mapping.aligned_size for mapping in self.weights.mappings
        )
        fragmentation_bytes = (
            retained_gms_allocated_bytes - retained_gms_parameter_span_bytes
        )
        logger.info(
            "GMS weights committed device=%d "
            "retained_gms_parameter_span_bytes=%d retained_gms_allocated_bytes=%d "
            "parameter_span_to_allocated_ratio=%.6f fragmentation_bytes=%d "
            "fragmentation_percent=%.2f copied_out_bytes=%d "
            "retained_gms_allocation_count=%d",
            self.device,
            retained_gms_parameter_span_bytes,
            retained_gms_allocated_bytes,
            retained_gms_parameter_span_bytes / retained_gms_allocated_bytes,
            fragmentation_bytes,
            fragmentation_bytes / retained_gms_allocated_bytes * 100,
            copied_out_bytes,
            self.weights.retained_gms_allocation_count,
        )

    def _destroy_weights_pool(self) -> None:
        gc.collect()
        weights_pool = self._weights_pool
        self._weights_pool = None
        del weights_pool
        gc.collect()
        self.raise_if_allocator_failed()

    def _malloc(self, size: int, device: int, _stream: int) -> int:
        if device != self.device:
            raise GMSError(f"allocator callback device {device} != {self.device}")
        domain = self._active_domain.get()
        if domain == _WEIGHTS:
            return self.weights.allocate(size)
        if domain == _KV_CACHE:
            return self.kv_cache.allocate(size)
        raise GMSError("GMS allocator callback has no active domain")

    def _free(self, base: int, size: int, device: int, _stream: int) -> None:
        if device != self.device:
            raise GMSError(f"allocator callback device {device} != {self.device}")
        if self.weights.owns(base):
            self.weights.free_from_allocator(base, size)
            return
        if self.kv_cache.owns(base):
            self.kv_cache.free_from_allocator(base, size)
            return
        raise GMSError(f"GMS allocator does not own VA 0x{base:x}")
