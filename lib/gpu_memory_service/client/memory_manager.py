# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU Memory Service client-side memory manager.

Two-tier API for GPU memory lifecycle management:

Tier 1 (Atomic Operations):
  - Connection: connect(), disconnect()
  - Handle ops (server-side cuMem allocations): allocate_handle, export_handle,
    get_handle_info, free_handle, commit, list_handles,
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

from gpu_memory_service.client.session import _GMSClientSession
from gpu_memory_service.common.cuda_utils import (
    align_to_granularity,
    cuda_set_current_device,
    cuda_synchronize,
    cuda_validate_pointer,
    cumem_address_free,
    cumem_address_reserve,
    cumem_get_allocation_granularity,
    cumem_import_from_shareable_handle_close_fd,
    cumem_map,
    cumem_release,
    cumem_set_access,
    cumem_unmap,
)
from gpu_memory_service.common.protocol.messages import GetAllocationResponse
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
    layout_slot: int

    def with_handle(self, handle: int) -> "LocalMapping":
        return LocalMapping(
            self.allocation_id,
            self.va,
            self.size,
            self.aligned_size,
            handle,
            self.tag,
            self.layout_slot,
        )

    def with_server_identity(
        self,
        allocation_id: str,
        layout_slot: int,
    ) -> "LocalMapping":
        return LocalMapping(
            allocation_id,
            self.va,
            self.size,
            self.aligned_size,
            self.handle,
            self.tag,
            layout_slot,
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

        self._client: Optional[_GMSClientSession] = None
        self._mappings: Dict[int, LocalMapping] = {}  # va -> mapping
        self._inverse_mapping: Dict[str, int] = {}

        self._unmapped = False
        self._granted_lock_type: Optional[GrantedLockType] = None

        # VA-stable unmap/remap state
        self._va_preserved = False
        self._last_memory_layout_hash: str = ""

        cuda_set_current_device(self.device)
        self.granularity = cumem_get_allocation_granularity(device)

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
        if self._client is not None:
            raise RuntimeError("Memory manager is already connected")
        self._client = _GMSClientSession(
            self.socket_path,
            lock_type=lock_type,
            timeout_ms=timeout_ms,
        )
        self._granted_lock_type = self._client.lock_type
        if self._granted_lock_type == GrantedLockType.RW:
            self._last_memory_layout_hash = ""
            return
        # Preserve the pre-unmap hash across reconnects so remap_all_vas can
        # detect that another writer changed the committed layout while this
        # process was disconnected.
        if self._client.committed and (
            not self._va_preserved or not self._last_memory_layout_hash
        ):
            self._last_memory_layout_hash = self._client.get_memory_layout_hash()
        elif not self._va_preserved:
            self._last_memory_layout_hash = ""

    def abort(self) -> None:
        """Drop the GMS session.

        Clean callers should unmap first. This also supports abrupt session
        drop with live mappings still present.
        """
        if self._client is not None:
            try:
                self._client.close()
            finally:
                self._client = None
                self._granted_lock_type = None
            return
        self._granted_lock_type = None

    # ==================== Tier 1: Handle Operations (server-side) ====================

    def allocate_handle(self, size: int, tag: str = "default") -> tuple[str, int]:
        """Allocate a cuMem handle on the server.

        Returns allocation_id and layout_slot. Size is aligned to VMM granularity
        before sending.
        """
        self._require_rw()
        aligned_size = align_to_granularity(size, self.granularity)
        response = self._client_rpc.allocate_info(aligned_size, tag)
        if int(response.aligned_size) != aligned_size:
            raise RuntimeError(
                "GMS allocation alignment mismatch: "
                f"{aligned_size} vs {response.aligned_size}"
            )
        return response.allocation_id, int(response.layout_slot)

    def export_handle(self, allocation_id: str) -> int:
        """Export allocation as POSIX FD."""
        return self._client_rpc.export(allocation_id)

    def get_handle_info(self, allocation_id: str):
        """Query allocation info from server."""
        return self._client_rpc.get_allocation(allocation_id)

    def free_handle(self, allocation_id: str) -> bool:
        """Release a cuMem allocation on the server."""
        ok = self._client_rpc.free(allocation_id)
        if not ok:
            raise RuntimeError(
                f"GMS free_handle failed for allocation_id={allocation_id}"
            )
        return True

    def commit(self) -> bool:
        """Synchronize, unmap writer mappings, then commit.

        Commit is a publish barrier. It guarantees all prior GPU writes in the
        current context are complete before the server transitions state. After
        a successful commit, the former writer process no longer has any mapped
        access to the published allocations. Any failure after local unmap
        raises because the process cannot safely recover its CUDA VMM state.
        """
        self._require_rw()

        # Publish barrier: all writer-side GPU work must be visible before commit.
        cuda_synchronize()

        for mapping in list(self._mappings.values()):
            if mapping.handle != 0:
                self.unmap_va(mapping.va)

        self._va_preserved = True
        self._unmapped = True

        self._client_rpc.commit()
        self._client = None
        self._granted_lock_type = None
        return True

    def get_memory_layout_hash(self) -> str:
        return self._client_rpc.get_memory_layout_hash()

    def list_handles(self, tag: Optional[str] = None) -> List[GetAllocationResponse]:
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
        return cumem_address_reserve(aligned_size, self.granularity)

    def map_va(
        self,
        fd: int,
        va: int,
        size: int,
        allocation_id: str,
        tag: str,
        layout_slot: int,
    ) -> int:
        """Import FD + cuMemMap + set access + track.

        Access is set based on current lock_type. Returns the CUDA handle.
        """
        assert self._granted_lock_type is not None
        aligned_size = align_to_granularity(size, self.granularity)
        handle = cumem_import_from_shareable_handle_close_fd(fd)
        cumem_map(va, aligned_size, handle)
        cumem_set_access(va, aligned_size, self.device, self._granted_lock_type)
        self._track_mapping(
            LocalMapping(
                allocation_id=allocation_id,
                va=va,
                size=size,
                aligned_size=aligned_size,
                handle=handle,
                tag=tag,
                layout_slot=layout_slot,
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
        cumem_unmap(va, mapping.aligned_size)
        cumem_release(mapping.handle)
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
        cumem_address_free(va, mapping.aligned_size)
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
            layout_slot = int(info.layout_slot)

            fd = self.export_handle(allocation_id)
            va = self.reserve_va(aligned_size)
            self.map_va(fd, va, alloc_size, allocation_id, alloc_tag, layout_slot)
            return va

        # Allocate path
        if size <= 0:
            raise ValueError("size must be > 0 when allocation_id is None")
        alloc_id, layout_slot = self.allocate_handle(size, tag)
        fd = self.export_handle(alloc_id)
        aligned_size = align_to_granularity(size, self.granularity)
        va = self.reserve_va(aligned_size)
        self.map_va(fd, va, size, alloc_id, tag, layout_slot)
        return va

    def destroy_mapping(self, va: int) -> None:
        """Unmap + free VA + free server handle for a single mapping."""
        mapping = self._mappings.get(va)
        if mapping is None:
            return

        alloc_id = mapping.allocation_id

        # Only free server handle if we're RW and haven't committed
        if self._granted_lock_type == GrantedLockType.RW:
            self.free_handle(alloc_id)

        self.unmap_va(va)
        self.free_va(va)

    def unmap_all_vas(self) -> None:
        """Synchronize + unmap all VAs. Preserves VA reservations for remap."""
        cuda_synchronize()

        unmapped_count = 0
        total_bytes = 0
        for va, mapping in list(self._mappings.items()):
            if mapping.handle == 0:
                continue
            self.unmap_va(va)
            unmapped_count += 1
            total_bytes += mapping.aligned_size

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
        cuda_set_current_device(self.device)

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

        allocations_by_slot = {
            int(info.layout_slot): info for info in self.list_handles()
        }

        remapped_count = 0
        total_bytes = 0
        for va, mapping in sorted(
            self._mappings.items(), key=lambda item: item[1].layout_slot
        ):
            if mapping.handle != 0:
                continue

            alloc_info = allocations_by_slot.get(mapping.layout_slot)
            if alloc_info is None:
                raise StaleMemoryLayoutError(
                    f"Layout slot {mapping.layout_slot} is missing from the committed layout"
                )
            if int(alloc_info.aligned_size) != mapping.aligned_size:
                raise StaleMemoryLayoutError(
                    f"Layout slot {mapping.layout_slot} size changed: "
                    f"{mapping.aligned_size} vs {int(alloc_info.aligned_size)}"
                )
            if str(alloc_info.tag) != mapping.tag:
                raise StaleMemoryLayoutError(
                    f"Layout slot {mapping.layout_slot} tag changed: "
                    f"{mapping.tag} vs {alloc_info.tag}"
                )

            fd = self.export_handle(alloc_info.allocation_id)
            handle = cumem_import_from_shareable_handle_close_fd(fd)
            cumem_map(va, mapping.aligned_size, handle)
            cumem_set_access(
                va, mapping.aligned_size, self.device, self._granted_lock_type
            )
            cuda_synchronize()
            cuda_validate_pointer(va)

            if mapping.allocation_id != alloc_info.allocation_id:
                self._inverse_mapping.pop(mapping.allocation_id, None)
            self._mappings[va] = mapping.with_server_identity(
                alloc_info.allocation_id,
                int(alloc_info.layout_slot),
            ).with_handle(handle)
            self._inverse_mapping[alloc_info.allocation_id] = va
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
        for va, mapping in sorted(
            self._mappings.items(), key=lambda item: item[1].layout_slot
        ):
            if mapping.handle != 0:
                continue

            response = self._client_rpc.allocate_info(mapping.aligned_size, tag)
            if int(response.aligned_size) != mapping.aligned_size:
                raise RuntimeError(
                    "GMS reallocation alignment mismatch: "
                    f"{mapping.aligned_size} vs {response.aligned_size}"
                )
            allocation_id = response.allocation_id

            old_alloc_id = mapping.allocation_id
            self._inverse_mapping.pop(old_alloc_id, None)
            self._mappings[va] = mapping.with_server_identity(
                allocation_id,
                int(response.layout_slot),
            )
            self._inverse_mapping[allocation_id] = va
            reallocated += 1

        logger.info(
            "[GPU Memory Service] Reallocated %d handles for preserved VAs",
            reallocated,
        )

    # ==================== Lifecycle ====================

    def close(self) -> None:
        """Strict cleanup.

        synchronize + unmap all + free all VAs + abort.
        """
        cuda_synchronize()

        for va in list(self._mappings.keys()):
            self.unmap_va(va)
            self.free_va(va)

        self.abort()
        self._unmapped = False
        self._va_preserved = False
        from gpu_memory_service.client.torch.allocator import (
            evict_gms_client_memory_manager,
        )

        evict_gms_client_memory_manager(self)

    def __enter__(self) -> "GMSClientMemoryManager":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ==================== Internals ====================

    @property
    def _client_rpc(self) -> _GMSClientSession:
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
