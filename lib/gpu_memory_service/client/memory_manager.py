# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU Memory Service client-side memory manager.

This is the unified memory manager for the GPU Memory Service architecture.

Key properties:
- Uses GMSRPCClient over a Unix-domain socket.
- The socket connection itself is the RW/RO lock.
- In write mode, the manager can allocate + map RW and then publish via commit().
- In read mode, the manager can import + map RO and hold the RO lock during inference.
- unmap()/remap() releases and reacquires the RO lock (and remaps allocations).

This module uses cuda-python bindings for CUDA driver API calls:
- import FDs (cuMemImportFromShareableHandle)
- reserve VA (cuMemAddressReserve)
- map/unmap (cuMemMap/cuMemUnmap)
- enforce access (cuMemSetAccess)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from cuda.bindings import driver as cuda
from gpu_memory_service.client.cuda_vmm_utils import (
    free_va,
    import_handle_from_fd,
    map_to_va,
    release_handle,
    reserve_va,
    set_access,
    set_current_device,
    synchronize,
    unmap,
)
from gpu_memory_service.client.rpc import GMSRPCClient
from gpu_memory_service.common.cuda_vmm_utils import (
    align_to_granularity,
    get_allocation_granularity,
)
from gpu_memory_service.common.types import GrantedLockType, RequestedLockType

logger = logging.getLogger(__name__)


class StaleMemoryLayoutError(Exception):
    """Raised when memory layout was modified while unmapped.

    This error indicates that a writer acquired the RW lock and changed the
    allocation structure (different sizes, different tensor layouts) while this
    reader was unmapped. The caller should re-import the model from scratch.

    IMPORTANT: This is a LAYOUT check, NOT a CONTENT check.
    - Detected: Allocation sizes changed, tensors added/removed, metadata structure changed
    - NOT detected: Weight values modified in-place

    This design is intentional: unmap/remap enables use cases like RL training
    where another process can write to the same memory locations (e.g., updating
    weights) while preserving the structure. As long as the layout (allocation
    and metadata table hashes) remains identical, remap() succeeds.
    """

    pass


@dataclass(frozen=True)
class LocalMapping:
    """Immutable record of a local VA mapping."""

    allocation_id: str
    va: int
    size: int
    aligned_size: int
    handle: int  # 0 if unmapped but VA reserved
    tag: str
    access: GrantedLockType

    def with_handle(self, handle: int) -> "LocalMapping":
        return LocalMapping(
            self.allocation_id,
            self.va,
            self.size,
            self.aligned_size,
            handle,
            self.tag,
            self.access,
        )

    def with_access(self, access: GrantedLockType) -> "LocalMapping":
        return LocalMapping(
            self.allocation_id,
            self.va,
            self.size,
            self.aligned_size,
            self.handle,
            self.tag,
            access,
        )


class GMSClientMemoryManager:
    """Unified memory manager that can act as writer or reader.

    Modes:
    - mode=RequestedLockType.RW: acquire RW lock, allocate/map RW, mutate metadata, commit/publish.
    - mode=RequestedLockType.RO: acquire RO lock (READY only), import/map RO, unmap/remap.
    - mode=RequestedLockType.RW_OR_RO: try RW if available, else wait for RO.
    """

    def __init__(
        self,
        socket_path: str,
        *,
        mode: RequestedLockType,
        device: int = 0,
        timeout_ms: Optional[int] = None,
    ) -> None:
        self.socket_path = socket_path
        self.device = device
        self._timeout_ms = timeout_ms

        self._client: Optional[GMSRPCClient] = None
        self._mappings: Dict[int, LocalMapping] = {}  # va -> mapping
        self._allocation_id_to_va: Dict[str, int] = {}

        self._unmapped = False
        self._closed = False
        self._preserved_allocation_ids: List[str] = []
        self._published = False
        self._mode: Optional[GrantedLockType] = None  # Updated by _connect

        # VA-stable unmap/remap state
        self._va_preserved = False
        self._last_memory_layout_hash: str = (
            ""  # Hash from server, saved on connect/commit
        )

        # Set the current CUDA device for subsequent operations.
        set_current_device(self.device)

        # Cache granularity for VA alignment
        self.granularity = get_allocation_granularity(device)

        self._connect(lock_type=mode, timeout_ms=timeout_ms)

    def _connect(
        self,
        *,
        lock_type: RequestedLockType,
        timeout_ms: Optional[int],
        update_memory_layout_hash: bool = True,
    ) -> None:
        self._client = GMSRPCClient(
            self.socket_path, lock_type=lock_type, timeout_ms=timeout_ms
        )
        self._unmapped = False
        # Update mode based on granted lock type (may differ from requested for rw_or_ro)
        self._mode = self._client.lock_type
        # Save state hash for stale detection on remap (skip during remap itself)
        if update_memory_layout_hash and self._client.committed:
            self._last_memory_layout_hash = self._client.get_memory_layout_hash()

    @property
    def mode(self) -> Optional[GrantedLockType]:
        """Current mode of the memory manager."""
        return self._mode

    @property
    def lock_type(self) -> Optional[GrantedLockType]:
        """Get the lock type actually granted by the server."""
        if self._client is None:
            return None
        return self._client.lock_type

    @property
    def is_connected(self) -> bool:
        return self._client is not None and self._client.is_connected

    @property
    def is_unmapped(self) -> bool:
        return self._unmapped

    @property
    def mappings(self) -> Dict[int, LocalMapping]:
        """Read-only view of VA -> LocalMapping dictionary."""
        return self._mappings

    @property
    def total_bytes(self) -> int:
        """Total bytes allocated across all mappings."""
        return sum(m.aligned_size for m in self._mappings.values())

    # ==================== Metadata convenience ====================

    def metadata_put(
        self, key: str, allocation_id: str, offset_bytes: int, value: bytes
    ) -> bool:
        return self._client_rpc.metadata_put(key, allocation_id, offset_bytes, value)

    def metadata_get(self, key: str) -> Optional[tuple[str, int, bytes]]:
        return self._client_rpc.metadata_get(key)

    def metadata_list(self, prefix: str = "") -> List[str]:
        return self._client_rpc.metadata_list(prefix)

    def metadata_delete(self, key: str) -> bool:
        return self._client_rpc.metadata_delete(key)

    # ==================== Allocation operations ====================

    def list_allocations(self, tag: Optional[str] = None) -> List[Dict]:
        """List all allocations on the server."""
        return self._client_rpc.list_allocations(tag)

    def allocate_and_map(self, size: int, tag: str = "default") -> int:
        """Allocate on server, reserve VA, and map locally.

        Args:
            size: Requested allocation size in bytes.
            tag: Allocation tag for server tracking.

        Returns:
            Virtual address of the mapped allocation.
        """
        self._require_rw()
        client = self._client_rpc
        aligned_size = align_to_granularity(size, self.granularity)

        va = reserve_va(aligned_size, self.granularity)
        try:
            allocation_id, server_aligned = client.allocate(aligned_size, tag)
            if int(server_aligned) != aligned_size:
                raise RuntimeError(
                    f"Alignment mismatch: {aligned_size} vs {server_aligned}"
                )

            fd = client.export(allocation_id)
            handle = import_handle_from_fd(fd)
            map_to_va(va, aligned_size, handle)
            set_access(va, aligned_size, self.device, GrantedLockType.RW)

            self._track_mapping(
                LocalMapping(
                    allocation_id=allocation_id,
                    va=va,
                    size=size,
                    aligned_size=aligned_size,
                    handle=handle,
                    tag=tag,
                    access=GrantedLockType.RW,
                )
            )
            return va
        except Exception:
            free_va(va, aligned_size)
            raise

    def free_mapping(self, va: int) -> None:
        """Unmap and free a local mapping."""
        mapping = self._mappings.pop(va, None)
        if mapping is None:
            return

        self._allocation_id_to_va.pop(mapping.allocation_id, None)

        try:
            if mapping.handle != 0:
                unmap(va, mapping.aligned_size)
                release_handle(mapping.handle)
            free_va(va, mapping.aligned_size)
        except Exception as e:
            logger.warning(f"Error freeing VA 0x{va:x}: {e}")

        if self.lock_type == GrantedLockType.RW and not self._published:
            try:
                self._client_rpc.free(mapping.allocation_id)
            except Exception:
                pass

    def import_allocation(self, allocation_id: str) -> int:
        """Import an existing allocation and map locally.

        In RO mode, maps read-only. In RW mode, maps read-write.
        """
        if allocation_id in self._allocation_id_to_va:
            return self._allocation_id_to_va[allocation_id]

        client = self._client_rpc
        # lock_type is guaranteed non-None when connected (after _client_rpc succeeds)
        assert self.lock_type is not None
        current_access = self.lock_type
        alloc_info = client.get_allocation(allocation_id)
        aligned_size = int(alloc_info.aligned_size)
        size = int(alloc_info.size)
        tag = str(getattr(alloc_info, "tag", "default"))

        va = reserve_va(aligned_size, self.granularity)
        try:
            fd = client.export(allocation_id)
            handle = import_handle_from_fd(fd)
            map_to_va(va, aligned_size, handle)
            set_access(va, aligned_size, self.device, current_access)

            self._track_mapping(
                LocalMapping(
                    allocation_id=allocation_id,
                    va=va,
                    size=size,
                    aligned_size=aligned_size,
                    handle=handle,
                    tag=tag,
                    access=current_access,
                )
            )
            return va
        except Exception:
            free_va(va, aligned_size)
            raise

    def clear_all(self) -> int:
        """Clear all allocations on the server (RW only). Local mappings are unmapped first."""
        self._require_rw()
        self._unmap_all()
        return self._client_rpc.clear_all()

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
        self._require_rw()

        synchronize()

        # After publishing, prevent further writes locally.
        for va, m in list(self._mappings.items()):
            if m.access != GrantedLockType.RO:
                set_access(m.va, m.aligned_size, self.device, GrantedLockType.RO)
                self._mappings[va] = m.with_access(GrantedLockType.RO)

        ok = self._client_rpc.commit()
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
            raise RuntimeError("Memory manager is closed")
        if self._unmapped:
            raise RuntimeError(
                "Cannot switch_to_read() while unmapped; call remap() first"
            )
        if self._client is not None:
            if self.lock_type == GrantedLockType.RO:
                return
            raise RuntimeError(
                "switch_to_read() requires the RW connection to be released (call commit() first)"
            )

        eff_timeout = timeout_ms if timeout_ms is not None else self._timeout_ms
        self._connect(lock_type=RequestedLockType.RO, timeout_ms=eff_timeout)

    # ==================== Unmap / remap (read mode) ====================

    def unmap(self) -> None:
        """Release RO lock and unmap local allocations (VA-stable).

        VAs are preserved during unmap so tensor pointers remain stable.
        On remap, allocations are remapped to the same VAs.
        """
        if self._closed:
            raise RuntimeError("Memory manager is closed")
        if self._unmapped:
            return
        if self.lock_type != GrantedLockType.RO:
            raise RuntimeError("unmap() requires RO mode")

        synchronize()

        # Preserve allocation IDs for remapping on remap
        self._preserved_allocation_ids = list(self._allocation_id_to_va.keys())

        # Unmap physical memory but keep VA reservations
        self._unmap_preserving_va()
        self._va_preserved = True

        # Ensure all CUDA VMM unmap operations complete before releasing the lock.
        # This prevents race conditions where remap() may be called before
        # physical memory is fully released.
        synchronize()

        self._client_rpc.close()
        self._client = None
        self._unmapped = True

    def remap(self, timeout_ms: Optional[int] = None) -> bool:
        """Reacquire RO lock and remap preserved allocations (VA-stable).

        Allocations are remapped to the same VAs they had before unmap,
        ensuring tensor pointers remain valid.

        Args:
            timeout_ms: Timeout for RO lock acquisition.

        Returns:
            True on success.

        Raises:
            TimeoutError: If timeout_ms expires waiting for RO lock.
            StaleMemoryLayoutError: If weights were structurally changed while unmapped.
        """
        if self._closed:
            raise RuntimeError("Memory manager is closed")
        if not self._unmapped:
            return True

        set_current_device(self.device)

        eff_timeout = timeout_ms if timeout_ms is not None else self._timeout_ms
        self._connect(
            lock_type=RequestedLockType.RO,
            timeout_ms=eff_timeout,
            update_memory_layout_hash=False,
        )

        # Check if memory layout changed while unmapped
        current_hash = self._client_rpc.get_memory_layout_hash()
        if (
            self._last_memory_layout_hash
            and current_hash != self._last_memory_layout_hash
        ):
            raise StaleMemoryLayoutError(
                f"State changed while unmapped: hash {self._last_memory_layout_hash[:16]}... -> {current_hash[:16]}..."
            )

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
            except StaleMemoryLayoutError:
                raise  # Let StaleMemoryLayoutError propagate
            except Exception as e:
                logger.warning(f"Failed to remap {alloc_id}: {e}")
                failed_count += 1

        if failed_count > 0:
            raise RuntimeError(
                f"Remap failed: {failed_count} of {len(self._preserved_allocation_ids)} "
                f"allocations could not be remapped"
            )

        logger.info(
            f"[GPU Memory Service] Remap complete on device {self.device}: "
            f"remapped {remapped_count} allocations ({total_bytes / (1 << 30):.2f} GiB)"
        )

        self._unmapped = False
        self._va_preserved = False
        return True

    # ==================== Cleanup ====================

    def close(self) -> None:
        if self._closed:
            return

        # Ensure kernels are done before tearing down mappings.
        synchronize()

        # Release all mappings including preserved VA reservations
        self._unmap_all()

        if self._client is not None:
            self._client.close()
            self._client = None
        self._closed = True
        self._unmapped = False
        self._va_preserved = False
        self._preserved_allocation_ids.clear()

    def __enter__(self) -> "GMSClientMemoryManager":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ==================== Internals ====================

    @property
    def _client_rpc(self) -> GMSRPCClient:
        """Get connected client or raise. Use instead of _require_connected() + assert."""
        if self._client is None:
            if self._unmapped:
                raise RuntimeError("Memory manager is unmapped")
            raise RuntimeError("Memory manager is not connected")
        return self._client

    def _require_rw(self) -> None:
        """Raise if not in RW mode."""
        if self.lock_type != GrantedLockType.RW:
            raise RuntimeError("Operation requires RW mode")

    def _track_mapping(self, m: LocalMapping) -> None:
        self._mappings[m.va] = m
        self._allocation_id_to_va[m.allocation_id] = m.va

    def _unmap_preserving_va(self) -> None:
        """Unmap physical memory but PRESERVE VA reservations for unmap/remap.

        This keeps the VA reservation intact so tensors maintain stable pointers.
        On remap, we can remap to the same VAs.
        """
        unmapped_count = 0
        total_bytes = 0
        for va, mapping in list(self._mappings.items()):
            if mapping.handle == 0:
                continue  # Already unmapped
            try:
                unmap(va, mapping.aligned_size)
                release_handle(mapping.handle)
                self._mappings[va] = mapping.with_handle(
                    0
                )  # Mark unmapped, VA reserved
                unmapped_count += 1
                total_bytes += mapping.aligned_size
            except Exception as e:
                logger.warning(
                    f"Error unmapping VA 0x{va:x} (preserving reservation): {e}"
                )
        logger.info(
            f"[GPU Memory Service] Unmapped {unmapped_count} allocations ({total_bytes / (1 << 30):.2f} GiB), "
            f"preserving {len(self._mappings)} VA reservations"
        )

    def _remap_preserved_va(self, allocation_id: str) -> int:
        """Remap an allocation to its preserved VA.

        Requires the VA to already be reserved (from before unmap).
        Validates allocation still exists and size matches.

        Returns the VA.
        Raises StaleMemoryLayoutError if allocation is missing or size changed.
        """
        set_current_device(self.device)

        va = self._allocation_id_to_va.get(allocation_id)
        if va is None:
            raise RuntimeError(f"No preserved VA for allocation {allocation_id}")

        mapping = self._mappings.get(va)
        if mapping is None:
            raise RuntimeError(f"No mapping info for VA 0x{va:x}")

        if mapping.handle != 0:
            return va  # Already mapped

        client = self._client_rpc
        # lock_type is guaranteed non-None when connected (after _client_rpc succeeds)
        assert self.lock_type is not None
        current_access = self.lock_type

        # Validate allocation still exists and size matches
        try:
            alloc_info = client.get_allocation(allocation_id)
        except Exception as e:
            raise StaleMemoryLayoutError(
                f"Allocation {allocation_id} no longer exists on server: {e}"
            ) from e

        server_aligned_size = int(alloc_info.aligned_size)
        if server_aligned_size != mapping.aligned_size:
            raise StaleMemoryLayoutError(
                f"Allocation {allocation_id} size changed: expected {mapping.aligned_size}, got {server_aligned_size}"
            )

        # Re-import the handle and map to the SAME VA (which is still reserved)
        fd = client.export(allocation_id)
        handle = import_handle_from_fd(fd)
        map_to_va(va, mapping.aligned_size, handle)

        # Set access permissions based on current lock type
        set_access(va, mapping.aligned_size, self.device, current_access)

        # Synchronize to ensure mapping is complete before any access
        synchronize()

        # Validate the pointer is accessible (this is what Triton checks)
        result, _dev_ptr = cuda.cuPointerGetAttribute(
            cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_DEVICE_POINTER, va
        )
        if result != cuda.CUresult.CUDA_SUCCESS:
            err_result, err_str = cuda.cuGetErrorString(result)
            err_msg = ""
            if err_result == cuda.CUresult.CUDA_SUCCESS and err_str:
                err_msg = (
                    err_str.decode() if isinstance(err_str, bytes) else str(err_str)
                )
            logger.warning(
                f"[GPU Memory Service] cuPointerGetAttribute failed for VA 0x{va:x} after remap: "
                f"error {result} ({err_msg})"
            )
        else:
            logger.debug(
                f"[GPU Memory Service] Remapped VA 0x{va:x} validated OK (device={self.device})"
            )

        # Update mapping with new handle and access
        updated = mapping.with_handle(handle)
        self._mappings[va] = updated.with_access(current_access)

        return va

    def _unmap_all(self) -> None:
        """Unmap and release all local mappings including VA reservations."""
        for va, mapping in list(self._mappings.items()):
            try:
                if mapping.handle != 0:
                    unmap(va, mapping.aligned_size)
                    release_handle(mapping.handle)
                free_va(va, mapping.aligned_size)
            except Exception as e:
                logger.warning(f"Error unmapping VA 0x{va:x}: {e}")
        self._mappings.clear()
        self._allocation_id_to_va.clear()
