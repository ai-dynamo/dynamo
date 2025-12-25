"""RPCCumemAllocator - unified client-side CUDA VMM allocator.

This is the unified allocator for the GPU Memory Service architecture.

Key properties:
- Uses AllocationServerClient over a Unix-domain socket.
- The socket connection itself is the RW/RO lock.
- In write mode, the allocator can allocate + map RW and then publish via commit().
- In read mode, the allocator can import + map RO and hold the RO lock during inference.
- sleep()/wake() releases and reacquires the RO lock (and remaps allocations).

This module intentionally uses CUDA driver API calls via ctypes to:
- import FDs (cuMemImportFromShareableHandle)
- reserve VA (cuMemAddressReserve)
- map/unmap (cuMemMap/cuMemUnmap)
- enforce access (cuMemSetAccess)
"""

from __future__ import annotations

import ctypes
import logging
import os
from ctypes import byref, c_int, c_size_t, c_ulonglong, c_void_p
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import torch
from gpu_memory_service.server.client import AllocationServerClient

logger = logging.getLogger(__name__)


class StaleWeightsError(Exception):
    """Raised when weight structure was changed while sleeping.

    This error indicates that a writer acquired the RW lock and modified the
    allocations (different sizes, different tensor layouts) while this reader
    was sleeping. The caller should re-import the model from scratch.

    Note: This does NOT detect content-only changes. A writer can update tensor
    values (e.g., for post-training/fine-tuning) without triggering this error.
    The validation only detects structural changes that would invalidate pointers.
    """

    pass


# CUDA constants
CU_MEM_ALLOCATION_TYPE_PINNED = 0x1
CU_MEM_LOCATION_TYPE_DEVICE = 0x1
CU_MEM_ACCESS_FLAGS_PROT_READ = 0x1
CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 0x3
CU_MEM_ALLOC_GRANULARITY_MINIMUM = 0x0
CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 0x1

CUdeviceptr = c_ulonglong
CUmemGenericAllocationHandle = c_ulonglong


class CUmemLocation(ctypes.Structure):
    _fields_ = [("type", c_int), ("id", c_int)]


class CUmemAccessDesc(ctypes.Structure):
    _fields_ = [("location", CUmemLocation), ("flags", c_int)]


class CUmemAllocationProp(ctypes.Structure):
    _fields_ = [
        ("type", c_int),
        ("requestedHandleTypes", c_int),
        ("location", CUmemLocation),
        ("win32HandleMetaData", c_void_p),
        ("allocFlags_compressionType", ctypes.c_uint),
        ("allocFlags_gpuDirectRDMACapable", ctypes.c_uint),
        ("allocFlags_usage", ctypes.c_uint),
        ("allocFlags_reserved", ctypes.c_uint * 4),
    ]


# Load CUDA library (lazy)
_cuda = None


def _get_cuda():
    global _cuda
    if _cuda is None:
        _cuda = ctypes.CDLL("libcuda.so.1")
    return _cuda


def _check(result: int, name: str):
    """Check CUDA result and raise on error."""
    if result != 0:
        cuda = _get_cuda()
        err = ctypes.c_char_p()
        cuda.cuGetErrorString(result, byref(err))
        raise RuntimeError(f"{name}: {err.value.decode() if err.value else result}")


@dataclass(frozen=True)
class LocalMapping:
    """Immutable record of a local VA mapping."""

    allocation_id: str
    va: int
    size: int
    aligned_size: int
    handle: int  # 0 if unmapped but VA reserved
    tag: str
    access: Literal["ro", "rw"]


@dataclass(frozen=True)
class PreservedRegistrySpec:
    """Snapshot of registry tensor spec for validation on wake."""

    key: str
    allocation_id: str
    offset_bytes: int
    shape: Tuple[int, ...]
    dtype: str
    stride: Optional[Tuple[int, ...]]


class RPCCumemAllocator:
    """Unified allocator that can act as writer or reader.

    Modes:
    - mode="write": acquire RW lock, allocate/map RW, mutate registry, commit/publish.
    - mode="read": acquire RO lock (READY only), import/map RO, sleep/wake.
    - mode="auto": try RO immediately; if not READY, fall back to RW.
    """

    def __init__(
        self,
        socket_path: str,
        *,
        mode: Literal["write", "read", "auto"] = "read",
        device: int = 0,
        timeout_ms: Optional[int] = None,
    ) -> None:
        self.socket_path = socket_path
        self.device = device
        self._timeout_ms = timeout_ms

        self._client: Optional[AllocationServerClient] = None
        self._mappings: Dict[int, LocalMapping] = {}  # va -> mapping
        self._allocation_id_to_va: Dict[str, int] = {}

        self._sleeping = False
        self._closed = False
        self._preserved_allocation_ids: List[str] = []
        self._published = False
        self._mode: Literal["write", "read"] = "read"  # Updated by _connect

        # VA-stable sleep/wake state
        self._va_preserved = False
        self._preserved_registry_prefix: Optional[str] = None
        self._preserved_registry_specs: Dict[str, PreservedRegistrySpec] = {}

        # Ensure torch is on the right device for subsequent CUDA operations.
        if torch.cuda.is_available():
            torch.cuda.set_device(self.device)

        # Cache granularity for VA alignment
        self.granularity = self._get_granularity()

        if mode == "write":
            self._connect(lock_type="rw", timeout_ms=timeout_ms)
        elif mode == "read":
            self._connect(lock_type="ro", timeout_ms=timeout_ms)
        elif mode == "auto":
            try:
                # Fast-path: if READY, RO handshake succeeds immediately.
                self._connect(lock_type="ro", timeout_ms=0)
            except TimeoutError:
                self._connect(lock_type="rw", timeout_ms=timeout_ms)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _get_granularity(self) -> int:
        cuda = _get_cuda()
        prop = CUmemAllocationProp()
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = self.device
        prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
        gran = c_size_t()
        _check(
            cuda.cuMemGetAllocationGranularity(byref(gran), byref(prop), 0),
            "granularity",
        )
        return int(gran.value)

    def _connect(
        self, *, lock_type: Literal["rw", "ro"], timeout_ms: Optional[int]
    ) -> None:
        self._client = AllocationServerClient(
            self.socket_path, lock_type=lock_type, timeout_ms=timeout_ms
        )
        self._sleeping = False
        # Update mode based on lock type
        self._mode = "write" if lock_type == "rw" else "read"

    @property
    def mode(self) -> Literal["write", "read"]:
        """Current mode of the allocator."""
        return self._mode

    @property
    def lock_type(self) -> Optional[Literal["rw", "ro"]]:
        if self._client is None:
            return None
        return "rw" if self._client.lock_type == "rw" else "ro"

    @property
    def is_connected(self) -> bool:
        return self._client is not None and self._client.is_connected

    @property
    def is_sleeping(self) -> bool:
        return self._sleeping

    @property
    def mapping_count(self) -> int:
        """Count of currently active (mapped) allocations.

        Excludes preserved VA reservations that are unmapped during sleep.
        """
        return sum(1 for m in self._mappings.values() if m.handle != 0)

    @property
    def preserved_va_count(self) -> int:
        """Count of preserved VA reservations (mapped + unmapped)."""
        return len(self._mappings)

    # ==================== Registry convenience ====================

    def registry_put(
        self, key: str, allocation_id: str, offset_bytes: int, value: bytes
    ) -> bool:
        self._require_connected()
        assert self._client is not None
        return self._client.registry_put(key, allocation_id, offset_bytes, value)

    def registry_get(self, key: str) -> Optional[tuple[str, int, bytes]]:
        self._require_connected()
        assert self._client is not None
        return self._client.registry_get(key)

    def registry_list(self, prefix: str = "") -> List[str]:
        self._require_connected()
        assert self._client is not None
        return self._client.registry_list(prefix)

    def registry_delete(self, key: str) -> bool:
        self._require_connected()
        assert self._client is not None
        return self._client.registry_delete(key)

    def registry_delete_prefix(self, prefix: str) -> int:
        self._require_connected()
        assert self._client is not None
        return self._client.registry_delete_prefix(prefix)

    def registry_prune_allocation(self, allocation_id: str) -> int:
        self._require_connected()
        assert self._client is not None
        return self._client.registry_prune_allocation(allocation_id)

    def registry_prune_missing_allocations(
        self, valid_allocation_ids: List[str]
    ) -> int:
        self._require_connected()
        assert self._client is not None
        return self._client.registry_prune_missing_allocations(valid_allocation_ids)

    # ==================== Allocation operations ====================

    def list_allocations(self, tag: Optional[str] = None) -> List[Dict]:
        self._require_connected()
        assert self._client is not None
        return self._client.list_allocations(tag)

    def get_va(self, allocation_id: str) -> Optional[int]:
        return self._allocation_id_to_va.get(allocation_id)

    def allocate(self, size: int, tag: str = "default") -> Tuple[str, int]:
        """Allocate on server and map RW locally (write mode only)."""
        self._require_connected()
        if self.lock_type != "rw":
            raise RuntimeError("allocate() requires RW mode")
        assert self._client is not None

        cuda = _get_cuda()

        allocation_id, aligned_size = self._client.allocate(size, tag)
        fd = self._client.export(allocation_id)

        try:
            handle = CUmemGenericAllocationHandle()
            _check(
                cuda.cuMemImportFromShareableHandle(
                    byref(handle),
                    c_void_p(fd),
                    c_int(CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR),
                    c_ulonglong(0),
                ),
                "cuMemImportFromShareableHandle",
            )
        finally:
            os.close(fd)

        va = CUdeviceptr()
        _check(
            cuda.cuMemAddressReserve(
                byref(va),
                c_size_t(aligned_size),
                c_size_t(self.granularity),
                CUdeviceptr(0),
                c_ulonglong(0),
            ),
            "cuMemAddressReserve",
        )

        _check(
            cuda.cuMemMap(
                va, c_size_t(aligned_size), c_size_t(0), handle, c_ulonglong(0)
            ),
            "cuMemMap",
        )

        self._set_access(va.value, aligned_size, access="rw")

        self._track_mapping(
            LocalMapping(
                allocation_id=allocation_id,
                va=va.value,
                size=size,
                aligned_size=aligned_size,
                handle=int(handle.value),
                tag=tag,
                access="rw",
            )
        )

        return allocation_id, va.value

    def allocate_to_va(
        self,
        size: int,
        va: int,
        aligned_size: int,
        device: int,
        tag: str = "default",
    ) -> str:
        """Allocate on server and map to a pre-reserved VA.

        This method is used for integration with PyTorch's CUDAPluggableAllocator
        where the C++ extension reserves VA before calling this method.

        The C++ extension's my_malloc() reserves VA first, then calls the Python
        callback which should use this method to:
        1. Allocate physical memory on the server
        2. Export the FD
        3. Import and map to the pre-reserved VA via cumem.import_and_map()
        4. Track the mapping in Python for sleep/wake

        Args:
            size: Original requested size
            va: Pre-reserved virtual address from C++ extension
            aligned_size: Aligned size matching the VA reservation
            device: CUDA device index
            tag: Allocation tag

        Returns:
            allocation_id from the server
        """
        self._require_connected()
        if self.lock_type != "rw":
            raise RuntimeError("allocate_to_va() requires RW mode")
        assert self._client is not None

        # Import the C++ extension for import_and_map
        try:
            from gpu_memory_service.extensions import _rpc_cumem_ext as cumem
        except ImportError as e:
            raise RuntimeError(
                "Missing CUDA VMM pluggable allocator extension. "
                "Build gpu_memory_service/core/csrc/rpc_cumem.cpp first."
            ) from e

        # Allocate on server
        allocation_id, server_aligned = self._client.allocate(aligned_size, tag)
        if int(server_aligned) != int(aligned_size):
            raise RuntimeError(
                f"Alignment mismatch: client={aligned_size} server={server_aligned}"
            )

        # Export FD and map to the pre-reserved VA
        fd = self._client.export(allocation_id)
        try:
            # Map RW for writer
            cumem.import_and_map(
                int(va), int(fd), int(aligned_size), int(device), False
            )
        finally:
            os.close(fd)

        # Get the handle from the C++ extension for tracking
        # cumem stores handle internally, but we need it for LocalMapping
        # Query the allocation info from cumem
        alloc_infos = cumem.get_all_allocations()
        handle = 0
        for info in alloc_infos:
            if int(info[0]) == int(va):
                handle = int(info[3])
                break

        # Track in Python allocator for sleep/wake
        self._track_mapping(
            LocalMapping(
                allocation_id=allocation_id,
                va=va,
                size=size,
                aligned_size=aligned_size,
                handle=handle,
                tag=tag,
                access="rw",
            )
        )

        return allocation_id

    def import_allocation(self, allocation_id: str) -> int:
        """Import an existing allocation and map locally.

        In RO mode, maps read-only. In RW mode, maps read-write.
        """
        self._require_connected()
        assert self._client is not None

        if allocation_id in self._allocation_id_to_va:
            return self._allocation_id_to_va[allocation_id]

        cuda = _get_cuda()

        alloc_info = self._client.get_allocation(allocation_id)
        aligned_size = int(alloc_info["aligned_size"])
        size = int(alloc_info["size"])
        tag = str(alloc_info.get("tag", "default"))

        fd = self._client.export(allocation_id)
        try:
            handle = CUmemGenericAllocationHandle()
            _check(
                cuda.cuMemImportFromShareableHandle(
                    byref(handle),
                    c_void_p(fd),
                    c_int(CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR),
                    c_ulonglong(0),
                ),
                "cuMemImportFromShareableHandle",
            )
        finally:
            os.close(fd)

        va = CUdeviceptr()
        _check(
            cuda.cuMemAddressReserve(
                byref(va),
                c_size_t(aligned_size),
                c_size_t(self.granularity),
                CUdeviceptr(0),
                c_ulonglong(0),
            ),
            "cuMemAddressReserve",
        )
        _check(
            cuda.cuMemMap(
                va, c_size_t(aligned_size), c_size_t(0), handle, c_ulonglong(0)
            ),
            "cuMemMap",
        )

        access: Literal["ro", "rw"] = "rw" if self.lock_type == "rw" else "ro"
        self._set_access(va.value, aligned_size, access=access)

        self._track_mapping(
            LocalMapping(
                allocation_id=allocation_id,
                va=va.value,
                size=size,
                aligned_size=aligned_size,
                handle=int(handle.value),
                tag=tag,
                access=access,
            )
        )

        return va.value

    def import_all(self, tag: Optional[str] = None) -> Dict[str, int]:
        self._require_connected()
        allocs = self.list_allocations(tag)
        out: Dict[str, int] = {}
        for a in allocs:
            alloc_id = a["allocation_id"]
            out[alloc_id] = self.import_allocation(alloc_id)
        return out

    def clear_all(self) -> int:
        """Clear all allocations on the server (RW only).

        Local mappings are unmapped first.
        """
        self._require_connected()
        if self.lock_type != "rw":
            raise RuntimeError("clear_all() requires RW mode")
        assert self._client is not None

        self._unmap_all()
        return self._client.clear_all()

    # ==================== Publish / mode switching ====================

    def commit(self) -> bool:
        """Publish weights (RW only).

        Client responsibilities:
        - cudaDeviceSynchronize() before publishing
        - flip local mappings to RO before publishing

        Server responsibilities:
        - transition to COMMITTED
        - close the RW socket (publish + release)
        """
        self._require_connected()
        if self.lock_type != "rw":
            raise RuntimeError("commit() requires RW mode")
        assert self._client is not None

        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

        # After publishing, prevent further writes locally.
        for va, m in list(self._mappings.items()):
            if m.access != "ro":
                self._set_access(m.va, m.aligned_size, access="ro")
                # Create new immutable LocalMapping with updated access
                self._mappings[va] = LocalMapping(
                    allocation_id=m.allocation_id,
                    va=m.va,
                    size=m.size,
                    aligned_size=m.aligned_size,
                    handle=m.handle,
                    tag=m.tag,
                    access="ro",
                )

        ok = self._client.commit()
        self._published = bool(ok)
        # _client.commit() closes the socket on success; reflect that here.
        if ok:
            self._client = None
        return bool(ok)

    def switch_to_read(self, timeout_ms: Optional[int] = None) -> None:
        """Acquire an RO lock after publishing.

        This is intended for the common flow where a writer loads weights and then
        becomes a reader for inference.
        """
        if self._closed:
            raise RuntimeError("Allocator is closed")
        if self._sleeping:
            raise RuntimeError(
                "Cannot switch_to_read() while sleeping; call wake() first"
            )
        if self._client is not None:
            if self.lock_type == "ro":
                return
            raise RuntimeError(
                "switch_to_read() requires the RW connection to be released (call commit() first)"
            )

        eff_timeout = timeout_ms if timeout_ms is not None else self._timeout_ms
        self._connect(lock_type="ro", timeout_ms=eff_timeout)

    # ==================== Sleep / wake (read mode) ====================

    def sleep(self, *, registry_prefix: Optional[str] = None) -> None:
        """Release RO lock and unmap local allocations (VA-stable).

        VAs are preserved during sleep so tensor pointers remain stable.
        On wake, allocations are remapped to the same VAs.

        Args:
            registry_prefix: If provided, snapshot registry specs for validation on wake.
                            Typically the config_hash prefix (e.g., "abc123:").
        """
        if self._closed:
            raise RuntimeError("Allocator is closed")
        if self._sleeping:
            return
        self._require_connected()
        if self.lock_type != "ro":
            raise RuntimeError("sleep() requires RO mode")

        # Ensure all device reads complete before unmapping and releasing the lock.
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

        # Preserve allocation IDs for remapping on wake
        self._preserved_allocation_ids = list(self._allocation_id_to_va.keys())

        # Snapshot registry specs for validation on wake (if prefix provided)
        self._preserved_registry_prefix = registry_prefix
        self._preserved_registry_specs.clear()
        if registry_prefix is not None:
            self._snapshot_registry_specs(registry_prefix)

        # Unmap physical memory but keep VA reservations
        self._unmap_preserving_va()
        self._va_preserved = True

        assert self._client is not None
        self._client.close()
        self._client = None
        self._sleeping = True

    def _snapshot_registry_specs(self, prefix: str) -> None:
        """Snapshot registry tensor specs for validation on wake."""
        assert self._client is not None
        keys = self._client.registry_list(prefix)
        for key in keys:
            got = self._client.registry_get(key)
            if got is None:
                continue
            allocation_id, offset_bytes, value = got
            # Parse the value to extract shape, dtype, stride
            try:
                import json

                obj = json.loads(value.decode("utf-8"))
                shape = tuple(int(x) for x in obj.get("shape", []))
                dtype = str(obj.get("dtype", ""))
                stride = None
                if "stride" in obj and obj["stride"] is not None:
                    stride = tuple(int(x) for x in obj["stride"])
                self._preserved_registry_specs[key] = PreservedRegistrySpec(
                    key=key,
                    allocation_id=allocation_id,
                    offset_bytes=offset_bytes,
                    shape=shape,
                    dtype=dtype,
                    stride=stride,
                )
            except Exception as e:
                logger.debug(f"Failed to parse registry spec for {key}: {e}")

    def wake(self, timeout_ms: Optional[int] = None) -> bool:
        """Reacquire RO lock and remap preserved allocations (VA-stable).

        Allocations are remapped to the same VAs they had before sleep,
        ensuring tensor pointers remain valid.

        Args:
            timeout_ms: Timeout for RO lock acquisition.

        Returns:
            True on success.

        Raises:
            TimeoutError: If timeout_ms expires waiting for RO lock.
            StaleWeightsError: If weights were structurally changed while sleeping.
        """
        if self._closed:
            raise RuntimeError("Allocator is closed")
        if not self._sleeping:
            return True

        # Ensure we're in the correct CUDA context before any operations
        if torch.cuda.is_available():
            torch.cuda.set_device(self.device)
            logger.debug(
                f"[GPU Memory Service] Set CUDA device to {self.device} for wake"
            )

        eff_timeout = timeout_ms if timeout_ms is not None else self._timeout_ms
        self._connect(lock_type="ro", timeout_ms=eff_timeout)

        # Validate registry specs if we have preserved specs
        if self._preserved_registry_specs:
            self._validate_registry_specs()

        # Remap to preserved VAs
        remapped_count = 0
        failed_count = 0
        total_bytes = 0
        for alloc_id in self._preserved_allocation_ids:
            try:
                va = self._remap_preserved_va(alloc_id)
                mapping = self._mappings.get(va)
                if mapping:
                    total_bytes += mapping.aligned_size
                remapped_count += 1
            except StaleWeightsError:
                raise  # Let StaleWeightsError propagate
            except Exception as e:
                logger.warning(f"Failed to remap {alloc_id}: {e}")
                failed_count += 1

        logger.info(
            f"[GPU Memory Service] Wake complete on device {self.device}: "
            f"remapped {remapped_count} allocations ({total_bytes / (1 << 30):.2f} GiB), "
            f"{failed_count} failed"
        )

        self._sleeping = False
        self._va_preserved = False
        self._preserved_registry_specs.clear()
        self._preserved_registry_prefix = None
        return True

    def _validate_registry_specs(self) -> None:
        """Validate registry specs haven't changed structurally since sleep.

        Raises StaleWeightsError if tensor structure changed.
        """
        assert self._client is not None
        for key, preserved in self._preserved_registry_specs.items():
            got = self._client.registry_get(key)
            if got is None:
                raise StaleWeightsError(f"Registry key {key} no longer exists")

            allocation_id, offset_bytes, value = got

            # Check allocation_id matches
            if allocation_id != preserved.allocation_id:
                raise StaleWeightsError(
                    f"Registry key {key}: allocation_id changed from "
                    f"{preserved.allocation_id} to {allocation_id}"
                )

            # Check offset matches
            if offset_bytes != preserved.offset_bytes:
                raise StaleWeightsError(
                    f"Registry key {key}: offset changed from "
                    f"{preserved.offset_bytes} to {offset_bytes}"
                )

            # Parse and compare shape, dtype, stride
            try:
                import json

                obj = json.loads(value.decode("utf-8"))
                shape = tuple(int(x) for x in obj.get("shape", []))
                dtype = str(obj.get("dtype", ""))
                stride = None
                if "stride" in obj and obj["stride"] is not None:
                    stride = tuple(int(x) for x in obj["stride"])

                if shape != preserved.shape:
                    raise StaleWeightsError(
                        f"Registry key {key}: shape changed from "
                        f"{preserved.shape} to {shape}"
                    )
                if dtype != preserved.dtype:
                    raise StaleWeightsError(
                        f"Registry key {key}: dtype changed from "
                        f"{preserved.dtype} to {dtype}"
                    )
                if stride != preserved.stride:
                    raise StaleWeightsError(
                        f"Registry key {key}: stride changed from "
                        f"{preserved.stride} to {stride}"
                    )
            except StaleWeightsError:
                raise
            except Exception as e:
                logger.warning(f"Failed to validate registry spec for {key}: {e}")

    # ==================== Cleanup ====================

    def close(self) -> None:
        if self._closed:
            return

        # Ensure kernels are done before tearing down mappings.
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

        # Release all mappings including preserved VA reservations
        self._unmap_all()

        if self._client is not None:
            self._client.close()
            self._client = None
        self._closed = True
        self._sleeping = False
        self._va_preserved = False
        self._preserved_allocation_ids.clear()
        self._preserved_registry_specs.clear()
        self._preserved_registry_prefix = None

    def __enter__(self) -> "RPCCumemAllocator":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
        return False

    # ==================== Internals ====================

    def _require_connected(self) -> None:
        if self._client is None:
            if self._sleeping:
                raise RuntimeError("Allocator is sleeping")
            raise RuntimeError("Allocator is not connected")

    def _track_mapping(self, m: LocalMapping) -> None:
        self._mappings[m.va] = m
        self._allocation_id_to_va[m.allocation_id] = m.va

    def _set_access(self, va: int, size: int, *, access: Literal["ro", "rw"]) -> None:
        cuda = _get_cuda()
        acc = CUmemAccessDesc()
        acc.location.type = CU_MEM_LOCATION_TYPE_DEVICE
        acc.location.id = self.device
        acc.flags = (
            CU_MEM_ACCESS_FLAGS_PROT_READ
            if access == "ro"
            else CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        )
        _check(
            cuda.cuMemSetAccess(
                CUdeviceptr(va), c_size_t(size), byref(acc), c_size_t(1)
            ),
            "cuMemSetAccess",
        )

    def _unmap_preserving_va(self) -> None:
        """Unmap physical memory but PRESERVE VA reservations for sleep/wake.

        This keeps the VA reservation intact so tensors maintain stable pointers.
        On wake, we can remap to the same VAs.
        """
        cuda = _get_cuda()
        unmapped_count = 0
        total_bytes = 0
        for va, mapping in list(self._mappings.items()):
            if mapping.handle == 0:
                continue  # Already unmapped
            try:
                # Unmap physical memory from VA
                result = cuda.cuMemUnmap(
                    CUdeviceptr(va), c_size_t(mapping.aligned_size)
                )
                if result != 0:
                    logger.warning(f"cuMemUnmap failed for VA 0x{va:x}: error {result}")
                # Release the imported handle reference (we need to re-import on wake)
                result = cuda.cuMemRelease(CUmemGenericAllocationHandle(mapping.handle))
                if result != 0:
                    logger.warning(
                        f"cuMemRelease failed for handle {mapping.handle}: error {result}"
                    )
                # Create new LocalMapping with handle=0 to mark as unmapped but VA reserved
                self._mappings[va] = LocalMapping(
                    allocation_id=mapping.allocation_id,
                    va=mapping.va,
                    size=mapping.size,
                    aligned_size=mapping.aligned_size,
                    handle=0,  # Unmapped
                    tag=mapping.tag,
                    access=mapping.access,
                )
                unmapped_count += 1
                total_bytes += mapping.aligned_size
            except Exception as e:
                logger.warning(
                    f"Error unmapping VA 0x{va:x} (preserving reservation): {e}"
                )
        logger.info(
            f"[GPU Memory Service] Unmapped {unmapped_count} allocations ({total_bytes / (1 << 30):.2f} GiB), preserving {len(self._mappings)} VA reservations"
        )
        # Note: We do NOT clear _mappings or _allocation_id_to_va
        # The VA reservations remain valid

    def _remap_preserved_va(self, allocation_id: str) -> int:
        """Remap an allocation to its preserved VA.

        Requires the VA to already be reserved (from before sleep).
        Validates allocation still exists and size matches.

        Returns the VA.
        Raises StaleWeightsError if allocation is missing or size changed.
        """
        # Ensure we're in the correct CUDA context for this device
        # This is critical for driver API calls to work correctly
        if torch.cuda.is_available():
            torch.cuda.set_device(self.device)

        va = self._allocation_id_to_va.get(allocation_id)
        if va is None:
            raise RuntimeError(f"No preserved VA for allocation {allocation_id}")

        mapping = self._mappings.get(va)
        if mapping is None:
            raise RuntimeError(f"No mapping info for VA 0x{va:x}")

        if mapping.handle != 0:
            return va  # Already mapped

        assert self._client is not None
        cuda = _get_cuda()

        # Validate allocation still exists and size matches
        try:
            alloc_info = self._client.get_allocation(allocation_id)
        except Exception as e:
            raise StaleWeightsError(
                f"Allocation {allocation_id} no longer exists on server: {e}"
            ) from e

        server_aligned_size = int(alloc_info["aligned_size"])
        if server_aligned_size != mapping.aligned_size:
            raise StaleWeightsError(
                f"Allocation {allocation_id} size changed: "
                f"expected {mapping.aligned_size}, got {server_aligned_size}"
            )

        # Re-import the handle
        fd = self._client.export(allocation_id)
        try:
            handle = CUmemGenericAllocationHandle()
            _check(
                cuda.cuMemImportFromShareableHandle(
                    byref(handle),
                    c_void_p(fd),
                    c_int(CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR),
                    c_ulonglong(0),
                ),
                "cuMemImportFromShareableHandle",
            )
        finally:
            os.close(fd)

        # Map to the SAME VA (which is still reserved)
        _check(
            cuda.cuMemMap(
                CUdeviceptr(va),
                c_size_t(mapping.aligned_size),
                c_size_t(0),
                handle,
                c_ulonglong(0),
            ),
            "cuMemMap",
        )

        # Set access permissions based on current lock type
        access: Literal["ro", "rw"] = "rw" if self.lock_type == "rw" else "ro"
        self._set_access(va, mapping.aligned_size, access=access)

        # Synchronize to ensure mapping is complete before any access
        cuda.cuCtxSynchronize()

        # Validate the pointer is accessible (this is what Triton checks)
        CU_POINTER_ATTRIBUTE_DEVICE_POINTER = 2
        dev_ptr = CUdeviceptr()
        result = cuda.cuPointerGetAttribute(
            byref(dev_ptr), c_int(CU_POINTER_ATTRIBUTE_DEVICE_POINTER), CUdeviceptr(va)
        )
        if result != 0:
            err_str = ctypes.c_char_p()
            cuda.cuGetErrorString(result, byref(err_str))
            logger.warning(
                f"[GPU Memory Service] cuPointerGetAttribute failed for VA 0x{va:x} after remap: "
                f"error {result} ({err_str.value.decode() if err_str.value else 'unknown'})"
            )
        else:
            logger.debug(
                f"[GPU Memory Service] Remapped VA 0x{va:x} validated OK (device={self.device})"
            )

        # Update mapping with new handle
        self._mappings[va] = LocalMapping(
            allocation_id=mapping.allocation_id,
            va=mapping.va,
            size=mapping.size,
            aligned_size=mapping.aligned_size,
            handle=int(handle.value),
            tag=mapping.tag,
            access=access,
        )

        return va

    def _unmap_all(self) -> None:
        """Unmap and release all local mappings including VA reservations."""
        cuda = _get_cuda()
        for va, mapping in list(self._mappings.items()):
            try:
                if mapping.handle != 0:
                    cuda.cuMemUnmap(CUdeviceptr(va), c_size_t(mapping.aligned_size))
                    cuda.cuMemRelease(CUmemGenericAllocationHandle(mapping.handle))
                cuda.cuMemAddressFree(CUdeviceptr(va), c_size_t(mapping.aligned_size))
            except Exception as e:
                logger.warning(f"Error unmapping VA 0x{va:x}: {e}")
        self._mappings.clear()
        self._allocation_id_to_va.clear()
