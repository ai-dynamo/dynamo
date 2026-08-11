# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM routing, Torch allocation, and GMS V1 lifecycle ownership."""

from __future__ import annotations

import gc
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
from vllm.device_allocator.sleep_mode_backend import (
    SleepModeBackend,
    SleepModeBackendFactory,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

BACKEND_NAME = "gms-v1"
logger = init_logger("vllm.gpu_memory_service.v1")
_WEIGHTS = "weights"
_KV_CACHE = "kv_cache"
_allocator_owner_lock = threading.Lock()
_allocator_owner: object | None = None
_allocator_initializing: object | None = None


def _reserve_allocator(owner: object) -> None:
    """Reserve construction of the one supported V1 allocator backend."""
    global _allocator_initializing
    with _allocator_owner_lock:
        if _allocator_owner is not None or _allocator_initializing is not None:
            raise RuntimeError(
                "GMS V1 supports exactly one allocator backend per process; "
                "a V1 backend is already initialized"
            )
        _allocator_initializing = owner


def _claim_allocator(owner: object, malloc: object, free: object) -> None:
    """Install callbacks and publish a fully initialized V1 process owner."""
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


class GMSV1SleepModeBackend(SleepModeBackend):
    """Own the rank-local GMS Parameter and ephemeral KV domains."""

    def __init__(self) -> None:
        super().__init__()
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
            # The shared native shim keeps callback pointers for process lifetime.
            # Pin these exact bound-method objects on their V1 backend owner.
            self._malloc_callback = self._malloc
            self._free_callback = self._free
            _claim_allocator(self, self._malloc_callback, self._free_callback)
        except BaseException:
            self._disconnect_managers_after_failure(kv_cache, weights)
            _release_allocator_reservation(self)
            raise

    @contextmanager
    def capture_weights(self, model: Callable[[], object]) -> Iterator[None]:
        try:
            with self._use_pool(_WEIGHTS, self._weights_pool):
                yield

            copy_non_parameter_tensors_to_default_allocator(
                model(),
                self._weights.mappings,
            )
            torch.cuda.synchronize(self._device)
            self._destroy_weights_pool()
            self._raise_if_allocator_failed()
            self._weights.commit()
        except BaseException:
            self._disconnect_after_failure()
            raise

        logger.info(
            "GMS weights committed device=%d allocations=%d",
            self._device,
            len(self._weights.mappings),
        )

    @contextmanager
    def capture_kv_cache(self) -> Iterator[None]:
        try:
            with self._use_pool(_KV_CACHE, self._kv_cache_pool):
                yield
            self._raise_if_allocator_failed()
        except BaseException:
            self._disconnect_after_failure()
            raise

    def suspend(self, level: int = 1) -> None:
        if level != 1:
            raise ValueError("GMS V1 supports only whole-engine level 1 suspend")
        if self._state != "RUNNING":
            raise RuntimeError(f"cannot suspend GMS V1 from {self._state}")

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

    def resume(self, tags: list[str] | None = None) -> None:
        if tags is not None:
            raise ValueError("GMS V1 does not support partial-tag resume")
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

    @classmethod
    def preserves_communicators(cls) -> bool:
        return True

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
        """Prune GMS mappings not retained by live model Parameters.

        Deleting the temporary MemPool releases its cached and unreferenced
        blocks through the allocator free callback. Blocks still owned by live
        Parameter storage remain mapped and become the committed weight set.
        """
        gc.collect()
        weights_pool = self._weights_pool
        self._weights_pool = None
        del weights_pool
        gc.collect()

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

    def _disconnect_after_failure(self) -> None:
        self._disconnect_managers_after_failure(self._kv_cache, self._weights)

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


SleepModeBackendFactory.register_backend(
    BACKEND_NAME,
    "gpu_memory_service.v1.integrations.vllm.backend",
    "GMSV1SleepModeBackend",
)
