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
from gpu_memory_service.common.locks import RequestedLockType
from gpu_memory_service.common.utils import get_socket_path
from gpu_memory_service.common.vmm import get_vmm
from gpu_memory_service.core.client.torch.extensions import _allocator_ext
from gpu_memory_service.v1.memory_manager import GMSClientMemoryManager
from gpu_memory_service.v1.parameter_storage import (
    copy_non_parameter_tensors_to_default_allocator,
)
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


class GMSV1SleepModeBackend(SleepModeBackend):
    """Own the rank-local GMS Parameter and ephemeral KV domains."""

    def __init__(self) -> None:
        super().__init__()
        self._device = torch.cuda.current_device()
        vmm = get_vmm()
        self._weights = GMSClientMemoryManager(
            get_socket_path(self._device, _WEIGHTS),
            vmm,
            self._device,
        )
        self._kv_cache = GMSClientMemoryManager(
            get_socket_path(self._device, _KV_CACHE),
            vmm,
            self._device,
        )
        self._weights.connect(RequestedLockType.RW)
        self._kv_cache.connect(RequestedLockType.RW)
        self._active_domain: ContextVar[str | None] = ContextVar(
            "gms_v1_active_domain",
            default=None,
        )
        self._allocator_failure: Exception | None = None
        self._allocator_failure_lock = threading.Lock()
        _allocator_ext.init_module(self._malloc, self._free)
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

    @contextmanager
    def capture_weights(self, model: "Callable[[], object]") -> "Iterator[None]":
        with self._use_pool(_WEIGHTS, self._weights_pool):
            yield

        (
            parameter_span_bytes,
            copied_out_bytes,
        ) = copy_non_parameter_tensors_to_default_allocator(
            model(),
            self._weights.mappings,
        )
        torch.cuda.synchronize(self._device)
        self._destroy_weights_pool()
        self._raise_if_allocator_failed()
        self._weights.commit()

        retained_aligned_gms_bytes = sum(
            mapping.aligned_size for mapping in self._weights.mappings
        )
        uncovered_retained_gms_bytes = retained_aligned_gms_bytes - parameter_span_bytes
        logger.info(
            "GMS weights committed device=%d parameter_span_bytes=%d "
            "retained_aligned_gms_bytes=%d "
            "parameter_span_to_retained_ratio=%.6f "
            "uncovered_retained_gms_bytes=%d uncovered_retained_gms_percent=%.2f "
            "copied_out_bytes=%d retained_gms_allocation_count=%d",
            self._device,
            parameter_span_bytes,
            retained_aligned_gms_bytes,
            parameter_span_bytes / retained_aligned_gms_bytes,
            uncovered_retained_gms_bytes,
            uncovered_retained_gms_bytes / retained_aligned_gms_bytes * 100,
            copied_out_bytes,
            len(self._weights.mappings),
        )

    @contextmanager
    def capture_kv_cache(self) -> "Iterator[None]":
        with self._use_pool(_KV_CACHE, self._kv_cache_pool):
            yield
        self._raise_if_allocator_failed()

    def suspend(self, level: int = 1) -> None:
        if level != 1:
            raise ValueError("GMS V1 supports only whole-engine level 1 suspend")
        if self._state != "RUNNING":
            raise RuntimeError(f"cannot suspend GMS V1 from {self._state}")

        try:
            gc.collect()
            self._raise_if_allocator_failed()
            self._weights.unmap_all_vas()
            self._weights.disconnect()
            self._kv_cache.unmap_all_vas()
            self._kv_cache.disconnect()
            torch.cuda.empty_cache()
            self._state = "SUSPENDED"
        except Exception as cause:
            logger.exception("GMS V1 suspend failed; terminating the worker process")
            raise SystemExit(1) from cause

    def resume(self, tags: list[str] | None = None) -> None:
        if tags is not None:
            raise ValueError("GMS V1 does not support partial-tag resume")
        if self._state != "SUSPENDED":
            raise RuntimeError(f"cannot resume GMS V1 from {self._state}")

        try:
            self._state = "RESUMING"
            wake_t0 = monotonic()

            kv_t0 = monotonic()
            self._kv_cache.connect(RequestedLockType.RW)
            kv_connected_at = monotonic()
            self._kv_cache.reallocate_all_handles()
            kv_reallocated_at = monotonic()
            self._kv_cache.remap_all_vas()
            kv_done_at = monotonic()
            logger.info(
                "GMS V1 KV wake device=%d connect_elapsed=%.3fs "
                "reallocate_elapsed=%.3fs remap_elapsed=%.3fs total_elapsed=%.3fs",
                self._device,
                kv_connected_at - kv_t0,
                kv_reallocated_at - kv_connected_at,
                kv_done_at - kv_reallocated_at,
                kv_done_at - kv_t0,
            )

            weights_t0 = monotonic()
            self._weights.connect(RequestedLockType.RO)
            weights_connected_at = monotonic()
            self._weights.remap_all_vas()
            weights_done_at = monotonic()
            logger.info(
                "GMS V1 weights wake device=%d connect_elapsed=%.3fs "
                "remap_elapsed=%.3fs total_elapsed=%.3fs",
                self._device,
                weights_connected_at - weights_t0,
                weights_done_at - weights_connected_at,
                weights_done_at - weights_t0,
            )
            self._state = "RUNNING"
            logger.info(
                "GMS V1 wake complete device=%d total_elapsed=%.3fs",
                self._device,
                monotonic() - wake_t0,
            )
        except Exception as cause:
            logger.exception("GMS V1 resume failed; terminating the worker process")
            raise SystemExit(1) from cause

    @classmethod
    def preserves_communicators(cls) -> bool:
        return True

    @contextmanager
    def _use_pool(self, domain: str, pool: object) -> "Iterator[None]":
        token = self._active_domain.set(domain)
        try:
            with torch.cuda.device(self._device):
                with torch.cuda.use_mem_pool(pool, device=self._device):
                    yield
        finally:
            self._active_domain.reset(token)

    def _destroy_weights_pool(self) -> None:
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
        except Exception as exc:
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


SleepModeBackendFactory.register_backend(
    BACKEND_NAME,
    "gpu_memory_service.v1.integrations.vllm.backend",
    "GMSV1SleepModeBackend",
)
