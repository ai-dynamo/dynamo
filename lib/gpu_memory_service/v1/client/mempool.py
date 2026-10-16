# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process-local Torch MemPool ownership for a GMS V1 client."""

from __future__ import annotations

import gc
import logging
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from time import monotonic
from typing import TYPE_CHECKING

import torch
from gpu_memory_service.client.torch.extensions import _allocator_ext
from gpu_memory_service.common import utils as common_utils
from gpu_memory_service.common.locks import RequestedLockType
from gpu_memory_service.common.vmm import get_vmm
from gpu_memory_service.v1.client.memory_manager import GMSClientMemoryManager
from gpu_memory_service.v1.client.parameter_storage import (
    copy_non_parameter_tensors_to_default_allocator,
)
from gpu_memory_service.v1.device import get_socket_path

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

logger = logging.getLogger(__name__)
_WEIGHTS = "weights"
_KV_CACHE = "kv_cache"
_allocator_owner_lock = threading.Lock()
_allocator_owner: object | None = None
_allocator_initializing: object | None = None


def _reserve_allocator(owner: object) -> None:
    global _allocator_initializing
    with _allocator_owner_lock:
        if _allocator_owner is not None or _allocator_initializing is not None:
            raise RuntimeError(
                "GMS V1 supports exactly one Torch MemPool client per process"
            )
        _allocator_initializing = owner


def _claim_allocator(owner: object, malloc: object, free: object) -> None:
    global _allocator_initializing, _allocator_owner
    with _allocator_owner_lock:
        if _allocator_initializing is not owner:
            raise RuntimeError("GMS V1 allocator initialization was not reserved")
        _allocator_ext.init_module(malloc, free)
        _allocator_owner = owner
        _allocator_initializing = None


def _release_allocator_reservation(owner: object) -> None:
    global _allocator_initializing
    with _allocator_owner_lock:
        if _allocator_initializing is owner:
            _allocator_initializing = None


class TorchMempoolMemoryClient:
    """Own GMS weight and KV domains behind Torch MemPool contexts.

    Lifecycle operations are serialized by the inference engine. Only the
    allocator callbacks may execute concurrently.
    """

    def __init__(self) -> None:
        self._state = "RUNNING"
        self._weights_state = "OPEN"
        weights: GMSClientMemoryManager | None = None
        kv_cache: GMSClientMemoryManager | None = None
        _reserve_allocator(self)
        try:
            self._device = torch.cuda.current_device()
            vmm = get_vmm()
            weights = GMSClientMemoryManager(
                get_socket_path(self._device, _WEIGHTS),
                vmm,
                self._device,
            )
            kv_cache = GMSClientMemoryManager(
                get_socket_path(self._device, _KV_CACHE),
                vmm,
                self._device,
            )
            self._weights = weights
            self._kv_cache = kv_cache
            self._active_domain: ContextVar[str | None] = ContextVar(
                "gms_v1_active_domain",
                default=None,
            )
            self._allocator_failure: Exception | None = None
            self._allocator_failure_lock = threading.Lock()
            self._unmanaged_pool = None
            self._weights.connect(RequestedLockType.RW)
            self._kv_cache.connect(RequestedLockType.RW)
            self._pluggable_allocator = torch.cuda.CUDAPluggableAllocator(
                _allocator_ext.__file__,
                "my_malloc",
                "my_free",
            )
            with torch.cuda.device(self._device):
                self._weights_pool = torch.cuda.MemPool(
                    allocator=self._pluggable_allocator.allocator()
                )
                self._kv_cache_pool = torch.cuda.MemPool(
                    allocator=self._pluggable_allocator.allocator()
                )
            # The native shim retains these callback pointers for process life.
            self._malloc_callback = self._malloc
            self._free_callback = self._free
            _claim_allocator(self, self._malloc_callback, self._free_callback)
        except BaseException:
            self._disconnect_managers_after_failure(kv_cache, weights)
            _release_allocator_reservation(self)
            raise

    @contextmanager
    def weight_region(self) -> Iterator[None]:
        """Route one of possibly many model loads into the weight domain."""
        self._require_running()
        self._require_weights_open()
        try:
            with self._use_pool(_WEIGHTS, self._weights_pool):
                yield
            self._raise_if_allocator_failed()
        except BaseException:
            self._fail()
            raise

    @contextmanager
    def kv_cache_region(self) -> Iterator[None]:
        """Route a KV allocation region into the reusable KV domain."""
        self._require_running()
        try:
            with self._use_pool(_KV_CACHE, self._kv_cache_pool):
                yield
            self._raise_if_allocator_failed()
        except BaseException:
            self._fail()
            raise

    @contextmanager
    def unmanaged_region(self) -> Iterator[None]:
        """Route temporary allocations through an ordinary Torch MemPool."""
        self._require_running()
        if self._unmanaged_pool is None:
            with torch.cuda.device(self._device):
                self._unmanaged_pool = torch.cuda.MemPool()
        with (
            torch.cuda.device(self._device),
            torch.cuda.use_mem_pool(self._unmanaged_pool, device=self._device),
        ):
            yield

    def publish_weights(self, models: Iterable[object]) -> None:
        """Commit all model Parameters captured by preceding weight regions."""
        if self._weights_state == "PUBLISHED":
            return
        self._require_weights_open()
        self._weights_state = "PUBLISHING"
        try:
            models = tuple(models)
            if not models:
                raise RuntimeError("GMS V1 did not observe any loaded models")
            if any(model is None for model in models):
                raise TypeError("GMS V1 model must not be None")
            copy_non_parameter_tensors_to_default_allocator(
                models,
                self._weights.mappings,
            )
            torch.cuda.synchronize(self._device)
            self._destroy_weights_pool()
            self._raise_if_allocator_failed()
            self._weights.commit()
        except BaseException:
            self._fail()
            raise
        self._weights_state = "PUBLISHED"
        logger.info(
            "GMS weights committed device=%d models=%d allocations=%d",
            self._device,
            len(models),
            len(self._weights.mappings),
        )

    def suspend(self) -> None:
        """Unmap all client-owned GPU memory."""
        if self._state != "RUNNING":
            raise RuntimeError(f"cannot suspend GMS V1 from {self._state}")
        if self._weights_state != "PUBLISHED":
            raise RuntimeError(
                f"cannot suspend GMS V1 with weights in {self._weights_state}"
            )

        try:
            gc.collect()
            self._raise_if_allocator_failed()
            torch.cuda.empty_cache()
            self._raise_if_allocator_failed()
            self._weights.unmap_all_vas()
            self._weights.disconnect()
            self._kv_cache.unmap_all_vas()
            self._kv_cache.disconnect()
            self._state = "SUSPENDED"
        except Exception:  # noqa: BLE001
            common_utils.fail(
                "GMS V1 suspend failed; terminating the worker process",
                exc_info=True,
            )

    def resume(self) -> None:
        """Reconnect and remap all client-owned GPU memory."""
        if self._state != "SUSPENDED":
            raise RuntimeError(f"cannot resume GMS V1 from {self._state}")

        try:
            self._state = "RESUMING"
            wake_t0 = monotonic()
            self._kv_cache.connect(RequestedLockType.RW)
            self._kv_cache.reallocate_all_handles()
            self._kv_cache.remap_all_vas()
            self._weights.connect(RequestedLockType.RO)
            self._weights.remap_all_vas()
            self._state = "RUNNING"
            logger.info(
                "GMS V1 wake complete device=%d total_elapsed=%.3fs",
                self._device,
                monotonic() - wake_t0,
            )
        except Exception:  # noqa: BLE001
            common_utils.fail(
                "GMS V1 resume failed; terminating the worker process",
                exc_info=True,
            )

    @contextmanager
    def _use_pool(self, domain: str, pool: object) -> Iterator[None]:
        token = self._active_domain.set(domain)
        try:
            with (
                torch.cuda.device(self._device),
                torch.cuda.use_mem_pool(pool, device=self._device),
            ):
                yield
        finally:
            self._active_domain.reset(token)

    def _destroy_weights_pool(self) -> None:
        gc.collect()
        weights_pool = self._weights_pool
        self._weights_pool = None
        del weights_pool
        gc.collect()

    def _require_weights_open(self) -> None:
        if self._weights_state != "OPEN":
            raise RuntimeError(
                f"cannot capture or publish GMS V1 weights from {self._weights_state}"
            )

    def _require_running(self) -> None:
        if self._state != "RUNNING":
            raise RuntimeError(f"cannot allocate with GMS V1 from {self._state}")

    def _fail(self) -> None:
        if self._state == "FAILED" and self._weights_state == "FAILED":
            return
        self._state = "FAILED"
        self._weights_state = "FAILED"
        self._disconnect_managers_after_failure(self._kv_cache, self._weights)

    def _malloc(self, size: int, device: int, _stream: int) -> int:
        try:
            if device != self._device:
                raise RuntimeError(
                    f"allocator callback device {device} != {self._device}"
                )
            domain = self._active_domain.get()
            if domain == _WEIGHTS:
                return self._weights.create_mapping(size)
            if domain == _KV_CACHE:
                return self._kv_cache.create_mapping(size)
            raise RuntimeError("GMS allocator callback has no active domain")
        except Exception as exc:
            self._record_allocator_failure(exc)
            raise

    def _free(self, va: int, size: int, device: int, _stream: int) -> None:
        try:
            if device != self._device:
                raise RuntimeError(
                    f"allocator callback device {device} != {self._device}"
                )
            if self._weights.owns(va):
                self._weights.destroy_mapping(va, size)
                return
            if self._kv_cache.owns(va):
                self._kv_cache.destroy_mapping(va, size)
                return
            raise RuntimeError(f"GMS allocator does not own VA 0x{va:x}")
        except Exception as exc:  # noqa: BLE001
            self._record_allocator_failure(exc)

    def _record_allocator_failure(self, failure: Exception) -> None:
        with self._allocator_failure_lock:
            if self._allocator_failure is None:
                self._allocator_failure = failure

    def _raise_if_allocator_failed(self) -> None:
        with self._allocator_failure_lock:
            failure = self._allocator_failure
        if failure is not None:
            raise RuntimeError("allocator callback failed") from failure

    @staticmethod
    def _disconnect_managers_after_failure(
        *managers: GMSClientMemoryManager | None,
    ) -> None:
        for manager in managers:
            if manager is None:
                continue
            try:
                manager.disconnect()
            except BaseException:
                logger.exception(
                    "GMS V1 disconnect failed while preserving an earlier error"
                )
