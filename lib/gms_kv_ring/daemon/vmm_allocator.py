# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Daemon-side allocator that backs the KV cache pool via the CUDA VMM
driver API and exports POSIX file descriptors for cross-process
attach.

This module is **isolated from the rest of the daemon** so we can
exercise its surface area on hosts without CUDA via the no-op /
unsupported branch, and so the runtime cost of the CUDA driver API
is paid only when the feature is actually used (`allocate_kv_pool`
RPC).

End-to-end flow this implements the daemon side of:

  1. RPC: engine asks for an HBM pool of a given size on a given
     device.
  2. ``VmmAllocator.allocate(...)`` here:
       - ``cuMemCreate`` → physical allocation handle
       - ``cuMemAddressReserve`` → daemon-side VA
       - ``cuMemMap`` + ``cuMemSetAccess`` → daemon can read/write
         the bytes
       - ``cuMemExportToShareableHandle`` → POSIX FD ready to ship
         to the engine over SCM_RIGHTS
  3. The daemon's RPC handler ships the FD; the engine imports it
     (P3 + P4 of the design) and maps the same physical pages into
     its own VA.
  4. ``VmmAllocator.release(...)`` later: unmaps the daemon-side
     VA, releases the daemon's refcount on the handle, frees the
     VA reservation, closes the FD. Engine-side mappings continue
     to work (``cuMemRelease`` is refcounted).

See ``docs/ALLOCATE_KV_POOL_DESIGN.md`` for the full plan.
"""

from __future__ import annotations

import logging
import os
import threading
import uuid
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


class VmmUnsupportedError(RuntimeError):
    """Raised by ``allocate`` when the host driver doesn't support VMM
    or the cuda-python package isn't importable. Callers should turn
    this into the ``vmm_unsupported`` RPC error code."""


class VmmAllocateError(RuntimeError):
    """Raised when one of the cuMem* driver calls fails. The message
    carries the underlying CUDA error name. Callers should turn this
    into the appropriate RPC error code (``insufficient_memory``,
    ``fd_export_failed``, etc.) based on the failing step."""


@dataclass
class VmmAllocation:
    """Bookkeeping for one VMM-backed pool. Owned by the daemon;
    the engine receives only ``fd`` + ``size_aligned``."""

    pool_id: str
    cu_handle: int  # CUmemGenericAllocationHandle (opaque integer)
    va: int  # daemon-side mapped virtual address
    size_aligned: int  # bytes; multiple of granularity
    granularity: int  # the allocator granularity used
    fd: int  # cuMemExportToShareableHandle output
    device_index: int  # GPU id the allocation lives on


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


class VmmAllocator:
    """Per-GPU VMM allocator.

    Construction is cheap; the actual CUDA support probe happens on
    the first ``is_supported()`` call (or implicitly inside
    ``allocate()``). This lazy-probe pattern lets the daemon boot
    cleanly on hosts where cuda-python isn't installed; only the
    ``allocate_kv_pool`` RPC is affected.

    Thread-safety: ``allocate`` / ``release`` acquire an internal
    lock around the in-memory pool-id → allocation dict. The cuMem*
    calls themselves are not serialized — the CUDA driver is
    thread-safe and these are non-overlapping per-allocation operations.
    """

    def __init__(self, device_index: int) -> None:
        self.device_index = int(device_index)
        self._allocations: dict[str, VmmAllocation] = {}
        self._granularity: Optional[int] = None
        self._supported: Optional[bool] = None
        self._primary_ctx = None
        self._cu_device = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_supported(self) -> bool:
        """Probe whether VMM is usable on this host. Cached after the
        first call; subsequent calls return the cached result.

        Returns False (no exception) in these cases:
          * cuda-python isn't importable
          * the CUDA driver is too old / lacks
            ``CU_MEM_ALLOC_GRANULARITY_MINIMUM`` for our requested
            property
          * any other unexpected error during the probe
        """
        if self._supported is not None:
            return self._supported
        self._supported = self._probe_support()
        return self._supported

    def allocate(self, size_bytes: int) -> VmmAllocation:
        """Allocate an HBM region of at least ``size_bytes`` bytes,
        export a POSIX FD for cross-process attach, and return the
        bookkeeping record. The caller is responsible for shipping
        the FD to the engine via SCM_RIGHTS (P3 of the design)."""
        if size_bytes <= 0:
            raise ValueError(
                f"VmmAllocator.allocate: size_bytes must be > 0, got {size_bytes}"
            )
        if not self.is_supported():
            raise VmmUnsupportedError(
                "VMM not supported on this host (cuda-python missing or "
                "driver too old)."
            )
        return self._allocate_supported(size_bytes)

    def release(self, pool_id: str) -> bool:
        """Drop the daemon's refcount on ``pool_id``. Returns True if
        the allocation was found + released, False if the pool_id is
        unknown (idempotent on duplicate release).

        After ``release``, engine-side mappings of the same handle
        continue to work — physical pages are only reclaimed when ALL
        processes have dropped their refcount (``cuMemRelease``
        semantics)."""
        with self._lock:
            alloc = self._allocations.pop(pool_id, None)
        if alloc is None:
            return False
        self._release_locked(alloc)
        return True

    def known_pool_ids(self) -> "list[str]":
        """Snapshot of currently-allocated pool_ids (for tests +
        diagnostics)."""
        with self._lock:
            return list(self._allocations.keys())

    def shutdown(self) -> None:
        """Release every outstanding allocation. Called from the
        daemon's shutdown path."""
        with self._lock:
            pool_ids = list(self._allocations.keys())
        for pid in pool_ids:
            self.release(pid)

    # ------------------------------------------------------------------
    # Internal — cuda-python interaction
    # ------------------------------------------------------------------

    def _probe_support(self) -> bool:
        """Run a one-time check that VMM + POSIX-FD export are
        available on this host. The result is cached in
        ``self._supported`` and the granularity in
        ``self._granularity`` for later allocate() calls."""
        try:
            from cuda import cuda  # type: ignore[import-not-found]
        except ImportError:
            logger.warning(
                "[VmmAllocator] cuda-python not importable; VMM mode "
                "disabled. Install `cuda-python` to enable the "
                "allocate_kv_pool RPC.",
            )
            return False
        try:
            # Ensure the driver is initialized and a primary context is
            # current on the target device. The daemon process won't
            # have one set up unless something earlier (e.g. torch)
            # touched CUDA — for tests + standalone deployments we
            # need the allocator to be self-sufficient.
            (err_init,) = cuda.cuInit(0)
            if err_init != cuda.CUresult.CUDA_SUCCESS:
                logger.warning(
                    "[VmmAllocator] cuInit failed (%s); VMM mode disabled.",
                    self._err_name(cuda, err_init),
                )
                return False
            err_dev, dev = cuda.cuDeviceGet(self.device_index)
            if err_dev != cuda.CUresult.CUDA_SUCCESS:
                logger.warning(
                    "[VmmAllocator] cuDeviceGet(%d) failed (%s); " "VMM mode disabled.",
                    self.device_index,
                    self._err_name(cuda, err_dev),
                )
                return False
            err_ctx, ctx = cuda.cuDevicePrimaryCtxRetain(dev)
            if err_ctx != cuda.CUresult.CUDA_SUCCESS:
                logger.warning(
                    "[VmmAllocator] cuDevicePrimaryCtxRetain failed "
                    "(%s); VMM mode disabled.",
                    self._err_name(cuda, err_ctx),
                )
                return False
            (err_set,) = cuda.cuCtxSetCurrent(ctx)
            if err_set != cuda.CUresult.CUDA_SUCCESS:
                logger.warning(
                    "[VmmAllocator] cuCtxSetCurrent failed (%s); " "VMM mode disabled.",
                    self._err_name(cuda, err_set),
                )
                return False
            # Track the retained context so shutdown() can release it.
            self._primary_ctx = ctx
            self._cu_device = dev

            prop = self._build_alloc_prop(cuda)
            err, gran = cuda.cuMemGetAllocationGranularity(
                prop,
                cuda.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM,
            )
            if err != cuda.CUresult.CUDA_SUCCESS:
                logger.warning(
                    "[VmmAllocator] cuMemGetAllocationGranularity failed "
                    "(err=%s); VMM mode disabled.",
                    err,
                )
                return False
            self._granularity = int(gran)
            if self._granularity <= 0:
                logger.warning(
                    "[VmmAllocator] driver returned non-positive "
                    "granularity %d; VMM mode disabled.",
                    self._granularity,
                )
                return False
            logger.info(
                "[VmmAllocator] VMM mode available on device %d "
                "(granularity=%d bytes)",
                self.device_index,
                self._granularity,
            )
            return True
        except Exception:  # noqa: BLE001
            logger.warning(
                "[VmmAllocator] support probe raised; VMM mode disabled.",
                exc_info=True,
            )
            return False

    def _build_alloc_prop(self, cuda):
        """Build a ``CUmemAllocationProp`` for a pinned device
        allocation on ``self.device_index`` with POSIX-FD export
        requested. Used both by the support probe and by allocate()."""
        prop = cuda.CUmemAllocationProp()
        prop.type = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location = cuda.CUmemLocation()
        prop.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = self.device_index
        # Request POSIX FD as the shareable handle format so we can
        # SCM_RIGHTS-pass it to the engine. Without this, exported
        # handles would be opaque integers tied to the daemon process.
        prop.requestedHandleTypes = (
            cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
        )
        return prop

    def _allocate_supported(self, size_bytes: int) -> VmmAllocation:
        """Run the actual cuMem* sequence. Caller MUST have verified
        is_supported(); this method assumes self._granularity is set."""
        from cuda import cuda  # type: ignore[import-not-found]

        assert self._granularity is not None  # invariant from is_supported
        size_aligned = _round_up(size_bytes, self._granularity)
        prop = self._build_alloc_prop(cuda)

        # ---- cuMemCreate ----
        err, handle = cuda.cuMemCreate(size_aligned, prop, 0)
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise VmmAllocateError(
                f"cuMemCreate failed: {self._err_name(cuda, err)} "
                f"(size={size_aligned})"
            )

        # ---- cuMemAddressReserve ----
        try:
            err, va = cuda.cuMemAddressReserve(size_aligned, 0, 0, 0)
            if err != cuda.CUresult.CUDA_SUCCESS:
                raise VmmAllocateError(
                    f"cuMemAddressReserve failed: {self._err_name(cuda, err)}"
                )
        except VmmAllocateError:
            cuda.cuMemRelease(handle)
            raise

        # ---- cuMemMap ----
        try:
            (err,) = cuda.cuMemMap(va, size_aligned, 0, handle, 0)
            if err != cuda.CUresult.CUDA_SUCCESS:
                raise VmmAllocateError(f"cuMemMap failed: {self._err_name(cuda, err)}")
        except VmmAllocateError:
            cuda.cuMemAddressFree(va, size_aligned)
            cuda.cuMemRelease(handle)
            raise

        # ---- cuMemSetAccess ----
        try:
            access = cuda.CUmemAccessDesc()
            access.location = prop.location
            access.flags = cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
            (err,) = cuda.cuMemSetAccess(va, size_aligned, [access], 1)
            if err != cuda.CUresult.CUDA_SUCCESS:
                raise VmmAllocateError(
                    f"cuMemSetAccess failed: {self._err_name(cuda, err)}"
                )
        except VmmAllocateError:
            cuda.cuMemUnmap(va, size_aligned)
            cuda.cuMemAddressFree(va, size_aligned)
            cuda.cuMemRelease(handle)
            raise

        # ---- cuMemExportToShareableHandle ----
        try:
            err, fd = cuda.cuMemExportToShareableHandle(
                handle,
                cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
                0,
            )
            if err != cuda.CUresult.CUDA_SUCCESS:
                raise VmmAllocateError(
                    f"cuMemExportToShareableHandle failed: "
                    f"{self._err_name(cuda, err)}"
                )
        except VmmAllocateError:
            cuda.cuMemUnmap(va, size_aligned)
            cuda.cuMemAddressFree(va, size_aligned)
            cuda.cuMemRelease(handle)
            raise

        pool_id = f"pool-{uuid.uuid4().hex[:12]}"
        alloc = VmmAllocation(
            pool_id=pool_id,
            cu_handle=int(handle),
            va=int(va),
            size_aligned=int(size_aligned),
            granularity=int(self._granularity),
            fd=int(fd),
            device_index=self.device_index,
        )
        with self._lock:
            self._allocations[pool_id] = alloc
        logger.info(
            "[VmmAllocator] allocated pool %s: %d bytes (aligned), fd=%d, " "device=%d",
            pool_id,
            alloc.size_aligned,
            alloc.fd,
            alloc.device_index,
        )
        return alloc

    def _release_locked(self, alloc: VmmAllocation) -> None:
        """Unmap + release a single allocation. Already removed from
        the dict; this method only touches CUDA + FD resources.
        Best-effort: errors are logged but not raised, since release
        is typically called from shutdown / cleanup paths."""
        try:
            from cuda import cuda  # type: ignore[import-not-found]
        except ImportError:
            # If we got here we must have allocated successfully, so
            # cuda-python should be present. This branch is defensive
            # for tests that mock the import.
            return
        try:
            (err,) = cuda.cuMemUnmap(alloc.va, alloc.size_aligned)
            if err != cuda.CUresult.CUDA_SUCCESS:
                logger.warning(
                    "[VmmAllocator] cuMemUnmap(%s) failed: %s",
                    alloc.pool_id,
                    self._err_name(cuda, err),
                )
        except Exception:  # noqa: BLE001
            logger.warning(
                "[VmmAllocator] cuMemUnmap(%s) raised",
                alloc.pool_id,
                exc_info=True,
            )
        try:
            (err,) = cuda.cuMemRelease(alloc.cu_handle)
            if err != cuda.CUresult.CUDA_SUCCESS:
                logger.warning(
                    "[VmmAllocator] cuMemRelease(%s) failed: %s",
                    alloc.pool_id,
                    self._err_name(cuda, err),
                )
        except Exception:  # noqa: BLE001
            logger.warning(
                "[VmmAllocator] cuMemRelease(%s) raised",
                alloc.pool_id,
                exc_info=True,
            )
        try:
            (err,) = cuda.cuMemAddressFree(alloc.va, alloc.size_aligned)
            if err != cuda.CUresult.CUDA_SUCCESS:
                logger.warning(
                    "[VmmAllocator] cuMemAddressFree(%s) failed: %s",
                    alloc.pool_id,
                    self._err_name(cuda, err),
                )
        except Exception:  # noqa: BLE001
            logger.warning(
                "[VmmAllocator] cuMemAddressFree(%s) raised",
                alloc.pool_id,
                exc_info=True,
            )
        try:
            os.close(alloc.fd)
        except OSError:
            pass

    @staticmethod
    def _err_name(cuda, err) -> str:
        """Stringify a CUDA error code. Falls back to the integer
        repr if cuGetErrorName itself fails."""
        try:
            _, name = cuda.cuGetErrorName(err)
            if name is not None:
                return name.decode() if isinstance(name, bytes) else str(name)
        except Exception:  # noqa: BLE001
            pass
        return str(err)
