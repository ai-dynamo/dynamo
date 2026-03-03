# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU Memory Service client-side memory manager.

Two-tier API for GPU memory lifecycle management:

Tier 1 (Atomic Operations):
  - Connection: connect(), disconnect()
  - Handle ops (server-side cuMem allocations): allocate_handle, export_handle,
    get_handle_info, free_handle, clear_all_handles, commit, list_handles,
    get_memory_layout_hash
  - VA ops (local address space): reserve_va, map_va, unmap_va, free_va
  - Metadata: metadata_put, metadata_get, metadata_list, metadata_delete

Tier 2 (Convenience — compose Tier 1 with error handling + sync):
  - create_mapping, destroy_mapping
  - unmap_all_vas, remap_all_vas, reallocate_all_handles
  - close

Integrations (vLLM/SGLang) call Tier 2. Advanced callers (e.g., KV failover)
can compose Tier 1 atomics directly.

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

from gpu_memory_service.client.cuda_vmm_utils import free_va as _cuda_free_va
from gpu_memory_service.client.cuda_vmm_utils import (
    import_handle_from_fd,
    map_to_va,
    release_handle,
)
from gpu_memory_service.client.cuda_vmm_utils import reserve_va as _cuda_reserve_va
from gpu_memory_service.client.cuda_vmm_utils import (
    set_access,
    set_current_device,
    synchronize,
    unmap,
    validate_pointer,
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
    - NOT detected: Data values modified in-place

    This design is intentional: unmap/remap enables use cases like RL training
    where another process can write to the same memory locations (e.g., updating
    data) while preserving the structure. As long as the layout (allocation
    and metadata table hashes) remains identical, remap() succeeds.
    """

    pass


@dataclass(frozen=True)
class LocalMapping:
    """Immutable record of a local VA mapping.

    Fields:
      - allocation_id: Server-side allocation ID
      - va: Local virtual address
      - size: Original requested size
      - aligned_size: Size aligned to VMM granularity
      - handle: CUDA memory handle (0 if unmapped but VA reserved)
      - tag: Allocation tag for server tracking
    """

    allocation_id: str
    va: int
    size: int
    aligned_size: int
    handle: int  # 0 if unmapped but VA reserved
    tag: str

    def with_handle(self, handle: int) -> "LocalMapping":
        return LocalMapping(
            self.allocation_id,
            self.va,
            self.size,
            self.aligned_size,
            handle,
            self.tag,
        )

    def with_allocation_id(self, allocation_id: str) -> "LocalMapping":
        return LocalMapping(
            allocation_id,
            self.va,
            self.size,
            self.aligned_size,
            self.handle,
            self.tag,
        )


class GMSClientMemoryManager:
    """Unified memory manager for GPU Memory Service.

    Constructor does NOT connect — call connect() explicitly after construction.
    """

    def __init__(
        self,
        socket_path: str,
        *,
        device: int = 0,
    ) -> None:
        self.socket_path = socket_path
        self.device = device

        self._client: Optional[GMSRPCClient] = None
        self._mappings: Dict[int, LocalMapping] = {}  # va -> mapping
        self._inverse_mapping: Dict[str, int] = {}

        self._unmapped = False
        self._granted_lock_type: Optional[GrantedLockType] = None

        # VA-stable unmap/remap state
        self._va_preserved = False
        self._last_memory_layout_hash: str = ""

        set_current_device(self.device)
        self.granularity = get_allocation_granularity(device)

    # ==================== Properties ====================

    @property
    def granted_lock_type(self) -> Optional[GrantedLockType]:
        return self._granted_lock_type

    @property
    def is_connected(self) -> bool:
        return self._client is not None and self._client.is_connected

    @property
    def is_unmapped(self) -> bool:
        return self._unmapped

    @property
    def mappings(self) -> Dict[int, LocalMapping]:
        return self._mappings

    @property
    def total_bytes(self) -> int:
        return sum(m.aligned_size for m in self._mappings.values())

    # ==================== Tier 1: Connection ====================

    def connect(
        self, lock_type: RequestedLockType, timeout_ms: Optional[int] = None
    ) -> None:
        """Connect to GMS server and acquire lock.

        Updates self._granted_lock_type based on granted lock type. Saves memory layout hash
        for stale detection if server is in committed state.
        """
        self._client = GMSRPCClient(
            self.socket_path,
            lock_type=lock_type,
            timeout_ms=timeout_ms,
        )
        self._granted_lock_type = self._client.lock_type
        # Save layout hash for stale detection on future remap
        if self._client.committed:
            self._last_memory_layout_hash = self._client.get_memory_layout_hash()

    def disconnect(self) -> None:
        """Close connection and release lock."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None

    # ==================== Tier 1: Handle Operations (server-side) ====================

    def allocate_handle(self, size: int, tag: str = "default") -> str:
        """Allocate a cuMem handle on the server.

        Returns allocation_id. Size is aligned to VMM granularity before sending.
        """
        self._require_rw()
        aligned_size = align_to_granularity(size, self.granularity)
        allocation_id, server_aligned = self._client_rpc.allocate(aligned_size, tag)
        if int(server_aligned) != aligned_size:
            raise RuntimeError(
                f"Alignment mismatch: {aligned_size} vs {server_aligned}"
            )
        return allocation_id

    def export_handle(self, allocation_id: str) -> int:
        """Export allocation as POSIX FD."""
        return self._client_rpc.export(allocation_id)

    def get_handle_info(self, allocation_id: str):
        """Query allocation info from server."""
        return self._client_rpc.get_allocation(allocation_id)

    def free_handle(self, allocation_id: str) -> bool:
        """Release a cuMem allocation on the server."""
        return self._client_rpc.free(allocation_id)

    def clear_all_handles(self) -> int:
        """Clear all allocations on the server. NO local unmap.

        Safe at startup (no local mappings) and during failover
        (preserves local VA reservations).
        """
        self._require_rw()
        return self._client_rpc.clear_all()

    def commit(self) -> bool:
        """Server-only commit: transition to COMMITTED state.

        No synchronize(), no CUDA access flip. The caller is responsible for
        synchronizing before calling this. Server closes the RW socket on
        success, so self._client becomes None.
        """
        self._require_rw()
        ok = self._client_rpc.commit()
        if ok:
            self._client = None
        return bool(ok)

    def get_memory_layout_hash(self) -> str:
        return self._client_rpc.get_memory_layout_hash()

    def list_handles(self, tag: Optional[str] = None) -> List[Dict]:
        return self._client_rpc.list_allocations(tag)

    # ==================== Tier 1: Metadata ====================

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

    # ==================== Tier 1: VA Operations (local) ====================

    def reserve_va(self, size: int) -> int:
        """Reserve virtual address space (cuMemAddressReserve). No tracking."""
        aligned_size = align_to_granularity(size, self.granularity)
        return _cuda_reserve_va(aligned_size, self.granularity)

    def map_va(self, fd: int, va: int, size: int, allocation_id: str, tag: str) -> int:
        """Import FD + cuMemMap + set access + track.

        Access is set based on current lock_type. Returns the CUDA handle.
        """
        assert self._granted_lock_type is not None
        aligned_size = align_to_granularity(size, self.granularity)
        handle = import_handle_from_fd(fd)
        try:
            map_to_va(va, aligned_size, handle)
            set_access(va, aligned_size, self.device, self._granted_lock_type)
        except Exception:
            try:
                unmap(va, aligned_size)
            except Exception:
                pass
            release_handle(handle)
            raise
        self._track_mapping(
            LocalMapping(
                allocation_id=allocation_id,
                va=va,
                size=size,
                aligned_size=aligned_size,
                handle=handle,
                tag=tag,
            )
        )
        return handle

    def unmap_va(self, va: int) -> None:
        """Unmap a single VA: cuMemUnmap + release handle.

        Keeps the VA reservation and tracking entry (handle set to 0).
        Works in both RW and RO modes.
        """
        mapping = self._mappings.get(va)
        if mapping is None or mapping.handle == 0:
            return
        unmap(va, mapping.aligned_size)
        release_handle(mapping.handle)
        self._mappings[va] = mapping.with_handle(0)

    def free_va(self, va: int) -> None:
        """Release a VA reservation: cuMemAddressFree + untrack.

        Unmaps first if still mapped.
        """
        mapping = self._mappings.get(va)
        if mapping is None:
            return
        if mapping.handle != 0:
            self.unmap_va(va)
            mapping = self._mappings.get(va)
            if mapping is None:
                return
        _cuda_free_va(va, mapping.aligned_size)
        self._mappings.pop(va, None)
        self._inverse_mapping.pop(mapping.allocation_id, None)

    # ==================== Tier 2: Convenience ====================

    def create_mapping(
        self,
        allocation_id: Optional[str] = None,
        size: int = 0,
        tag: str = "default",
    ) -> int:
        """Allocate or import a handle and map to a new VA.

        If allocation_id is None (allocate path):
          allocate_handle -> export_handle -> reserve_va -> map_va

        If allocation_id given (import path, cached):
          Check cache -> get_handle_info -> export_handle -> reserve_va -> map_va
        """
        if allocation_id is not None:
            # Import path: check cache first
            cached_va = self._inverse_mapping.get(allocation_id)
            if cached_va is not None:
                mapping = self._mappings.get(cached_va)
                if mapping is not None and mapping.handle == 0:
                    raise RuntimeError(
                        f"Allocation {allocation_id} is cached but unmapped "
                        f"(VA 0x{cached_va:x}). Use remap_all_vas() to restore."
                    )
                return cached_va

            info = self.get_handle_info(allocation_id)
            alloc_size = int(info.size)
            aligned_size = int(info.aligned_size)
            alloc_tag = str(getattr(info, "tag", "default"))

            fd = self.export_handle(allocation_id)
            va = self.reserve_va(aligned_size)
            try:
                self.map_va(fd, va, alloc_size, allocation_id, alloc_tag)
            except Exception:
                _cuda_free_va(va, align_to_granularity(aligned_size, self.granularity))
                raise
            return va

        # Allocate path
        if size <= 0:
            raise ValueError("size must be > 0 when allocation_id is None")
        alloc_id = self.allocate_handle(size, tag)
        fd = self.export_handle(alloc_id)
        aligned_size = align_to_granularity(size, self.granularity)
        va = self.reserve_va(aligned_size)
        try:
            self.map_va(fd, va, size, alloc_id, tag)
        except Exception:
            _cuda_free_va(va, aligned_size)
            raise
        return va

    def destroy_mapping(self, va: int) -> None:
        """Unmap + free VA + free server handle for a single mapping."""
        mapping = self._mappings.get(va)
        if mapping is None:
            return

        alloc_id = mapping.allocation_id

        try:
            self.unmap_va(va)
        except Exception as e:
            logger.warning("Error in unmap_va for 0x%x: %s", va, e)

        try:
            self.free_va(va)
        except Exception as e:
            logger.warning("Error in free_va for 0x%x: %s", va, e)

        # Only free server handle if we're RW and haven't committed
        if self._granted_lock_type == GrantedLockType.RW:
            try:
                self.free_handle(alloc_id)
            except Exception:
                pass

    def unmap_all_vas(self) -> None:
        """Synchronize + unmap all VAs. Preserves VA reservations for remap."""
        synchronize()

        unmapped_count = 0
        total_bytes = 0
        for va, mapping in list(self._mappings.items()):
            if mapping.handle == 0:
                continue
            try:
                self.unmap_va(va)
                unmapped_count += 1
                total_bytes += mapping.aligned_size
            except Exception as e:
                logger.warning("Error unmapping VA 0x%x: %s", va, e)

        self._va_preserved = True
        self._unmapped = True
        logger.info(
            "[GPU Memory Service] Unmapped %d allocations (%.2f GiB), "
            "preserving %d VA reservations",
            unmapped_count,
            total_bytes / (1 << 30),
            len(self._mappings),
        )

    def remap_all_vas(self) -> None:
        """Re-import existing handles at preserved VAs.

        Checks layout hash for staleness. Validates each allocation still
        exists and size matches before remapping.
        """
        set_current_device(self.device)

        # Stale layout check
        current_hash = self.get_memory_layout_hash()
        if (
            self._last_memory_layout_hash
            and current_hash != self._last_memory_layout_hash
        ):
            raise StaleMemoryLayoutError(
                f"Layout changed: {self._last_memory_layout_hash[:16]}... -> {current_hash[:16]}..."
            )

        assert self._granted_lock_type is not None

        remapped_count = 0
        total_bytes = 0
        for va, mapping in list(self._mappings.items()):
            if mapping.handle != 0:
                continue  # Already mapped

            # Validate allocation still exists
            try:
                alloc_info = self.get_handle_info(mapping.allocation_id)
            except Exception as e:
                raise StaleMemoryLayoutError(
                    f"Allocation {mapping.allocation_id} no longer exists: {e}"
                ) from e

            if int(alloc_info.aligned_size) != mapping.aligned_size:
                raise StaleMemoryLayoutError(
                    f"Allocation {mapping.allocation_id} size changed: "
                    f"{mapping.aligned_size} vs {int(alloc_info.aligned_size)}"
                )

            # Re-import and map to preserved VA
            fd = self.export_handle(mapping.allocation_id)
            handle = import_handle_from_fd(fd)
            map_to_va(va, mapping.aligned_size, handle)
            set_access(va, mapping.aligned_size, self.device, self._granted_lock_type)

            synchronize()
            validate_pointer(va)

            self._mappings[va] = mapping.with_handle(handle)
            remapped_count += 1
            total_bytes += mapping.aligned_size

        self._va_preserved = False
        self._unmapped = False
        logger.info(
            "[GPU Memory Service] Remap complete on device %d: "
            "remapped %d allocations (%.2f GiB)",
            self.device,
            remapped_count,
            total_bytes / (1 << 30),
        )

    def reallocate_all_handles(self, tag: str = "default") -> None:
        """Allocate fresh server handles for all preserved VAs (no mapping).

        Used during failover: the shadow engine's VAs are still reserved,
        but the physical memory was freed. This allocates new server-side
        handles and updates tracking (handle stays 0 — call remap_all_vas()
        afterward to actually map them).
        """
        self._require_rw()
        if not self._va_preserved:
            raise RuntimeError(
                "reallocate_all_handles requires preserved VAs (call unmap_all_vas first)"
            )

        reallocated = 0
        for va, mapping in list(self._mappings.items()):
            if mapping.handle != 0:
                continue

            # Allocate fresh handle on server (uses raw RPC to avoid re-aligning)
            allocation_id, server_aligned = self._client_rpc.allocate(
                mapping.aligned_size, tag
            )
            if int(server_aligned) != mapping.aligned_size:
                raise RuntimeError(
                    f"Alignment mismatch during reallocation: "
                    f"{mapping.aligned_size} vs {server_aligned}"
                )

            # Update tracking: new allocation_id, handle stays 0
            old_alloc_id = mapping.allocation_id
            self._inverse_mapping.pop(old_alloc_id, None)
            self._mappings[va] = mapping.with_allocation_id(allocation_id)
            self._inverse_mapping[allocation_id] = va
            reallocated += 1

        logger.info(
            "[GPU Memory Service] Reallocated %d handles for preserved VAs",
            reallocated,
        )

    # ==================== Lifecycle ====================

    def close(self, free: bool = False) -> None:
        """Best-effort cleanup. NOT reliable in crash/signal paths.

        synchronize + unmap all + free all VAs + disconnect.
        free=True: also clear_all_handles() on server before disconnect.
        VAs are freed by CUDA context teardown on process exit anyway.
        """
        try:
            synchronize()
        except Exception:
            pass

        for va in list(self._mappings.keys()):
            try:
                self.unmap_va(va)
            except Exception as e:
                logger.warning("Error unmapping VA 0x%x during close: %s", va, e)

        for va in list(self._mappings.keys()):
            try:
                self.free_va(va)
            except Exception as e:
                logger.warning("Error freeing VA 0x%x during close: %s", va, e)

        if (
            free
            and self._client is not None
            and self._granted_lock_type == GrantedLockType.RW
        ):
            try:
                self.clear_all_handles()
            except Exception as e:
                logger.warning("Error clearing handles during close: %s", e)

        self.disconnect()
        self._unmapped = False
        self._va_preserved = False

    def __enter__(self) -> "GMSClientMemoryManager":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ==================== Internals ====================

    @property
    def _client_rpc(self) -> GMSRPCClient:
        """Get connected client or raise."""
        if self._client is None:
            if self._unmapped:
                raise RuntimeError("Memory manager is unmapped")
            raise RuntimeError("Memory manager is not connected")
        return self._client

    def _require_rw(self) -> None:
        if self._granted_lock_type != GrantedLockType.RW:
            raise RuntimeError("Operation requires RW mode")

    def _track_mapping(self, m: LocalMapping) -> None:
        self._mappings[m.va] = m
        self._inverse_mapping[m.allocation_id] = m.va
