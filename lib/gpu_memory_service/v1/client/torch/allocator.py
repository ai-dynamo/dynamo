# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Temporary Torch model-load pool backed by GMS V1."""

from __future__ import annotations

import gc
import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING

from ...errors import GMSError
from ..memory_manager import SnapshotMemoryManager
from .module import normalize_model_storages, validate_model_storage_ownership

if TYPE_CHECKING:
    from collections.abc import Iterator


class _AllocatorCallbacks:
    def __init__(self, manager: SnapshotMemoryManager):
        self.manager = manager
        self.failure: Exception | None = None
        self._failure_lock = threading.Lock()

    def _record_failure(self, failure: Exception) -> None:
        with self._failure_lock:
            self.failure = self.failure or failure

    def malloc(self, size: int, device: int, _stream: int) -> int:
        try:
            if device != self.manager.device:
                raise GMSError(
                    f"allocator callback device {device} != {self.manager.device}"
                )
            return self.manager.allocate(size)
        except Exception as exc:
            self._record_failure(exc)
            raise

    def free(self, base: int, size: int, device: int, _stream: int) -> None:
        try:
            if device != self.manager.device:
                raise GMSError(
                    f"allocator callback device {device} != {self.manager.device}"
                )
            self.manager.free_from_allocator(base, size)
        except Exception as exc:
            # CUDAPluggableAllocator's free ABI returns void.
            self._record_failure(exc)


class SnapshotTorchPool:
    """Own one temporary GMS pool and one retained native workspace pool."""

    def __init__(self, manager: SnapshotMemoryManager):
        import torch
        from gpu_memory_service.client.torch.extensions import _allocator_ext

        self._torch = torch
        self._manager = manager
        self._allocator = _AllocatorCallbacks(manager)
        self._condition = threading.Condition()
        self._active_scope = False
        self._finalized = False
        self.device = manager.device
        if _allocator_ext is None:
            raise RuntimeError("GPU Memory Service allocator extension is not built")
        _allocator_ext.init_module_strict(self._allocator.malloc, self._allocator.free)
        # MemPools and live storages hold only a non-owning native pointer.
        self._pluggable_allocator = torch.cuda.CUDAPluggableAllocator(
            _allocator_ext.__file__, "my_malloc", "my_free"
        )
        with torch.cuda.device(self.device):
            self.model_load: torch.cuda.MemPool | None = torch.cuda.MemPool(
                allocator=self._pluggable_allocator.allocator()
            )
            self.native_workspace = torch.cuda.MemPool()

    @contextmanager
    def model_load_pool(self) -> "Iterator[None]":
        with self._condition:
            if self._active_scope:
                raise GMSError("GMS V1 model-load pool is already active")
            if self.model_load is None:
                raise GMSError("GMS V1 model-load pool has been destroyed")
            self._active_scope = True
        try:
            with self._torch.cuda.device(self.device):
                with self._torch.cuda.use_mem_pool(self.model_load, device=self.device):
                    yield
        finally:
            with self._condition:
                self._active_scope = False
                self._condition.notify_all()

    @contextmanager
    def native_workspace_pool(self) -> "Iterator[None]":
        with self._torch.cuda.device(self.device):
            with self._torch.cuda.use_mem_pool(
                self.native_workspace, device=self.device
            ):
                yield

    def finalize_model_load(self, model: object) -> None:
        """Normalize live storages and destroy the temporary GMS pool."""
        with self._condition:
            if self._active_scope:
                raise GMSError("GMS V1 model-load scope is still active")
            if self._finalized:
                raise GMSError("GMS V1 model load has already been finalized")
            self._finalized = True

        try:
            normalize_model_storages(model, self._manager.mappings)
            self._torch.cuda.synchronize(self.device)
            self._destroy_model_load_pool()
            validate_model_storage_ownership(model, self._manager.mappings)
        except Exception as cause:
            if self.model_load is not None:
                self.abort_model_load(cause)
            self._manager.abort(cause)

    def abort_model_load(self, cause: Exception) -> None:
        with self._condition:
            self._finalized = True
        try:
            self._torch.cuda.synchronize(self.device)
            self._destroy_model_load_pool()
        except Exception as cleanup:
            cause = cleanup
        self._manager.abort(cause)

    def _destroy_model_load_pool(self) -> None:
        gc.collect()
        model_load = self.model_load
        self.model_load = None
        del model_load
        gc.collect()
        with self._allocator._failure_lock:
            failure = self._allocator.failure
        if failure is not None:
            raise GMSError("allocator free callback failed") from failure

    def prepare_snapshot(self) -> None:
        if not self._finalized or self.model_load is not None:
            raise GMSError("GMS V1 model load has not been finalized")
        self._manager.sleep()
