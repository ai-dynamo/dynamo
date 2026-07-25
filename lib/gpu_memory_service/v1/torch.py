# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Snapshot profile orchestration for a temporary GMS Torch pool."""

from __future__ import annotations

import gc
import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING

from gpu_memory_service.core.client.torch.allocator import (
    TorchAllocatorCallbacks,
    create_torch_allocator,
    create_torch_mem_pool,
)
from gpu_memory_service.core.errors import GMSError
from gpu_memory_service.v1.memory_manager import SnapshotMemoryManager
from gpu_memory_service.v1.tensor import normalize_captured_tensors

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

logger = logging.getLogger(__name__)


class SnapshotTorchPool:
    """Own the temporary GMS model-load pool."""

    def __init__(self, manager: SnapshotMemoryManager):
        import torch

        self._torch = torch
        self._manager = manager
        self._allocator = TorchAllocatorCallbacks(self._malloc, self._free)
        self.device = manager.device
        self._pluggable_allocator = create_torch_allocator(self._allocator)
        with torch.cuda.device(self.device):
            self.model_load = create_torch_mem_pool(self._pluggable_allocator)

    @contextmanager
    def _model_load_pool(self) -> "Iterator[None]":
        with self._torch.cuda.device(self.device):
            with self._torch.cuda.use_mem_pool(self.model_load, device=self.device):
                yield

    @contextmanager
    def capture_weights(self, model: "Callable[[], object]") -> "Iterator[None]":
        """Capture broad weight loading and commit its Parameter backing."""
        try:
            with self._model_load_pool():
                yield
        except Exception as cause:
            self._manager.abort(cause)

        self._finalize_model_load(model)

    def _finalize_model_load(self, model: "Callable[[], object]") -> None:
        try:
            (
                retained_gms_parameter_span_bytes,
                copied_out_bytes,
            ) = normalize_captured_tensors(model(), self._manager.mappings)
            self._torch.cuda.synchronize(self.device)
            self._destroy_model_load_pool()
        except Exception as cause:
            self._manager.abort(cause)

        self._manager.commit()
        retained_gms_allocated_bytes = sum(
            mapping.aligned_size for mapping in self._manager.mappings
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
            self._manager.retained_gms_allocation_count,
        )

    def _destroy_model_load_pool(self) -> None:
        gc.collect()
        model_load = self.model_load
        self.model_load = None
        del model_load
        gc.collect()
        failure = self._allocator.get_failure()
        if failure is not None:
            raise GMSError("allocator free callback failed") from failure

    def _malloc(self, size: int, device: int, _stream: int) -> int:
        if device != self._manager.device:
            raise GMSError(
                f"allocator callback device {device} != {self._manager.device}"
            )
        return self._manager.allocate(size)

    def _free(self, base: int, size: int, device: int, _stream: int) -> None:
        if device != self._manager.device:
            raise GMSError(
                f"allocator callback device {device} != {self._manager.device}"
            )
        self._manager.free_from_allocator(base, size)

    def sleep(self) -> None:
        self._manager.sleep()

    def wake(self) -> None:
        self._manager.wake()
