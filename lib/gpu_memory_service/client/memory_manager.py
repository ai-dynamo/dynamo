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
  - Metadata: metadata_put, metadata_get, metadata_list, metadata_get_all,
    metadata_delete

Tier 2 (Convenience — compose Tier 1 with error handling + sync):
  - create_mapping, import_allocations, destroy_mapping
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

import bisect
import logging
import os
import resource
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from gpu_memory_service.client.session import _GMSClientSession
from gpu_memory_service.common.locks import GrantedLockType, RequestedLockType
from gpu_memory_service.common.protocol.messages import GetAllocationResponse
from gpu_memory_service.common.utils import align_to_granularity, is_bulk_import_enabled
from gpu_memory_service.common.vmm import VMMDeviceType, get_vmm, get_vmm_device_type

logger = logging.getLogger(__name__)


def _fd_headroom() -> Optional[int]:
    """File descriptors this process may still open; None if unknown or unlimited."""
    soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft == resource.RLIM_INFINITY:
        return None
    try:
        return soft - len(os.listdir("/proc/self/fd"))
    except OSError:
        return None


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


@dataclass
class _ScratchMapping:
    """Per-VA tracking for one scratch-aliased allocation.

    VA chunks all alias the same physical chunk (scratch_handle). One physical
    allocation, many virtual mappings. torch.zeros into the range succeeds;
    cudagraphs capture VAs so they survive the eventual swap to real.

    Lifecycle: install via create_scratch_mapping; tear down via
    unmap_all_vas (drops physical) and prepare_scratch_for_reallocation
    (moves the scratch bookkeeping into _mappings as preserved-VA metadata so
    reallocate_all_handles + remap_all_vas can install fresh server backing).
    """

    size: int
    aligned_size: int  # CUDA-granularity server allocation size
    va_reserved_size: int  # scratch-rounded local VA reservation
    tag: str
    scratch_handle: int  # 0 after unmap_all_vas drops the physical


@dataclass(frozen=True)
class LocalMapping:
    """Immutable record of a local VA mapping.

    Fields:
      - allocation_id: Server-side allocation ID
      - va: Local virtual address
      - size: Original requested size
      - aligned_size: Size aligned to VMM granularity
      - va_reserved_size: Size of the local VA reservation
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
    va_reserved_size: int = 0

    def __post_init__(self) -> None:
        if self.va_reserved_size == 0:
            object.__setattr__(self, "va_reserved_size", self.aligned_size)

    def with_handle(self, handle: int) -> "LocalMapping":
        return LocalMapping(
            self.allocation_id,
            self.va,
            self.size,
            self.aligned_size,
            handle,
            self.tag,
            self.layout_slot,
            self.va_reserved_size,
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
            self.va_reserved_size,
        )


class _FixedVAWindow:
    """Bookkeeping for one fixed-VA window; makes no driver calls.

    Imports sit at ``base + offset(layout slot)`` and grow up from ``base``.
    Client-created ranges grow down from the end and reuse released ranges
    first, so neither placement depends on the order the two kinds are created
    in. Overlap is refused here because cuMemMap onto a mapped range is fatal
    on CUDA.
    """

    def __init__(self, base: int, size: int) -> None:
        self.base = base
        self.size = size
        self.reserved = False
        self.imports = 0  # slot-placed ranges, all below _floor
        self._starts: List[int] = []  # sorted starts of carved ranges
        self._sizes: Dict[int, int] = {}
        # Client ranges live in [_floor, end), either carved or free in _holes.
        self._floor = self.end
        self._holes: Dict[int, int] = {}

    @property
    def end(self) -> int:
        return self.base + self.size

    @property
    def is_empty(self) -> bool:
        return not self._starts

    def _describe(self) -> str:
        return f"fixed VA window [0x{self.base:x}, 0x{self.end:x})"

    def _overlaps(self, va: int, size: int) -> bool:
        i = bisect.bisect_right(self._starts, va)
        if i and self._starts[i - 1] + self._sizes[self._starts[i - 1]] > va:
            return True
        return i < len(self._starts) and self._starts[i] < va + size

    def _insert(self, va: int, size: int) -> int:
        bisect.insort(self._starts, va)
        self._sizes[va] = size
        return va

    def carve(self, va: int, size: int, alignment: int) -> int:
        """Carve the range of a slot-placed import; it must stay below the
        client ranges."""
        if (
            va % alignment
            or va < self.base
            or va + size > self._floor
            or self._overlaps(va, size)
        ):
            raise RuntimeError(
                f"[GMS] 0x{size:x} bytes at 0x{va:x} do not fit the "
                f"{self._describe()} (client ranges from 0x{self._floor:x})"
            )
        self.imports += 1
        return self._insert(va, size)

    def carve_top_down(self, size: int, alignment: int) -> int:
        """Carve a client range: the highest fitting released range, else below
        the lowest client range."""
        for start, n in sorted(self._holes.items(), reverse=True):
            va = (start + n - size) // alignment * alignment
            if va >= start:
                del self._holes[start]
                if va > start:
                    self._holes[start] = va - start
                if va + size < start + n:
                    self._holes[va + size] = start + n - va - size
                return self._insert(va, size)
        va = (self._floor - size) // alignment * alignment
        # [va, _floor) becomes client space, alignment gap included.
        if va < self.base or self._overlaps(va, self._floor - va):
            raise RuntimeError(
                f"[GMS] the {self._describe()} has no free 0x{size:x}-byte range"
            )
        if va + size < self._floor:
            self._holes[va + size] = self._floor - va - size
        self._floor = va
        return self._insert(va, size)

    def release(self, va: int) -> None:
        size = self._sizes.pop(va)
        del self._starts[bisect.bisect_left(self._starts, va)]
        if va < self._floor:
            self.imports -= 1
            return
        for start, n in list(self._holes.items()):
            if start + n == va:
                del self._holes[start]
                va, size = start, n + size
            elif va + size == start:
                del self._holes[start]
                size += n
        if va == self._floor:
            self._floor = va + size
        else:
            self._holes[va] = size

    def clear(self) -> None:
        self.imports = 0
        self._starts.clear()
        self._sizes.clear()
        self._holes.clear()
        self._floor = self.end


@dataclass(frozen=True)
class LayoutCommit:
    """What :meth:`GMSClientMemoryManager.commit_layout` did to this session.

    Committing narrows the caller's grant, so the call hands back what it now holds.
    """

    memory_layout_hash: str
    granted_lock_type: GrantedLockType


class GMSClientMemoryManager:
    """Unified memory manager for GPU Memory Service.

    Constructor does NOT connect — call connect() explicitly after construction.
    """

    # Exported fds a bulk import holds open at once, at most. A chunk shrinks
    # when the process is close to RLIMIT_NOFILE, down to one fd at a time,
    # which is what a per-allocation import holds.
    _BULK_CHUNK = 256
    # Descriptors a chunk leaves free for the rest of the process.
    _BULK_FD_MARGIN = 32

    def __init__(
        self,
        socket_path: str,
        *,
        device: int = 0,
        tag: Optional[str] = None,
        scratch_size: int = 512 * 1024 * 1024,
        bulk_import: Optional[bool] = None,
        fixed_va_window: Optional[tuple[int, int]] = None,
    ) -> None:
        """``bulk_import``: whether ``remap_all_vas`` and the read-only attach
        map allocations in bulk (see ``import_allocations``) instead of one at
        a time. ``None`` follows ``DYN_GMS_BULK_IMPORT``, which defaults to on.

        ``fixed_va_window``: ``(base, size)`` of an address window reserved
        here and holding every mapping of this manager. An import lands at
        ``base`` plus the offset of its layout slot, so every process that
        imports the same committed layout maps it at the same addresses.
        Client-created ranges are carved down from the end of the window.
        ``None`` (default): the driver picks every address. CUDA only.
        """
        self.socket_path = socket_path
        self.device = device
        self.tag = tag
        self.scratch_size = scratch_size
        self._vmm = get_vmm()

        self._client: Optional[_GMSClientSession] = None

        # Two disjoint VA registries keyed by base VA:
        #   _mappings           — server-backed allocations (VA <-> LocalMapping).
        #   _scratch_mappings   — client-local scratch-aliased VAs awaiting
        #                         preserved-VA bookkeeping via
        #                         prepare_scratch_for_reallocation().
        self._mappings: Dict[int, LocalMapping] = {}
        self._inverse_mapping: Dict[str, int] = {}
        self._scratch_mappings: Dict[int, _ScratchMapping] = {}
        # All scratch mappings alias ONE shared physical granule (N KV layers ->
        # one scratch_size block, not N). Created lazily, released once.
        self._shared_scratch_handle: int = 0

        self._unmapped = False
        self._aborted = False
        self._granted_lock_type: Optional[GrantedLockType] = None

        self.bulk_import = (
            is_bulk_import_enabled() if bulk_import is None else bulk_import
        )

        self._fixed = (
            None if fixed_va_window is None else _FixedVAWindow(*fixed_va_window)
        )
        # Slot offsets of the layout the live imports were placed by; re-read
        # once no import is left, so a new layout is not placed by an old one.
        self._fixed_slot_offsets: Optional[Dict[int, int]] = None

        # VA-stable unmap/remap state
        self._va_preserved = False
        self._last_memory_layout_hash: str = ""

        self._vmm.ensure_initialized()
        self._vmm.runtime_set_device(device)
        self.granularity = self._vmm.get_allocation_granularity(device)
        if self._fixed is not None:
            # The driver honors the base only while the range is free, so take
            # the whole window before anything else in the process can.
            self._reserve_fixed_window(self._fixed)

    # ==================== Properties ====================

    @property
    def device_type(self) -> VMMDeviceType:
        return get_vmm_device_type()

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

    @property
    def fixed_va_window(self) -> Optional[tuple[int, int]]:
        if self._fixed is None:
            return None
        return self._fixed.base, self._fixed.size

    # ==================== Tier 1: Connection ====================

    def connect(
        self, lock_type: RequestedLockType, timeout_ms: Optional[int] = None
    ) -> None:
        """Connect to GMS server and acquire lock.

        Updates self._granted_lock_type based on granted lock type. Saves memory layout hash
        for stale detection if server is in committed state.

        On reconnect after abort (e.g. after CRIU restore on a different GPU),
        refreshes the socket path from the current GPU UUID so we connect to
        the correct GMS server.
        """
        if self._client is not None:
            raise RuntimeError("Memory manager is already connected")

        # After abort + CRIU restore the process may be on a different GPU.
        # Re-derive socket path from current UUID so we talk to the right server.
        if self._aborted and self.tag is not None:
            from gpu_memory_service.common.utils import (
                get_socket_path,
                invalidate_uuid_cache,
            )

            invalidate_uuid_cache()
            new_path = get_socket_path(self.device, self.tag)
            if new_path != self.socket_path:
                logger.info(
                    "Refreshed socket path for tag=%s: %s -> %s",
                    self.tag,
                    self.socket_path,
                    new_path,
                )
                self.socket_path = new_path
            self._aborted = False

        try:
            self._client = _GMSClientSession(
                self.socket_path,
                lock_type=lock_type,
                timeout_ms=timeout_ms,
            )
        except BaseException:
            # Callers drop a manager whose connect failed without closing it.
            # An empty window left reserved would make every later manager for
            # the same window fail; it is reserved again on first use.
            if self._fixed is not None and self._fixed.is_empty:
                self._release_fixed_window(self._fixed)
            raise
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
        self._aborted = True
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
        self._vmm.synchronize()

        for mapping in list(self._mappings.values()):
            if mapping.handle != 0:
                self.unmap_va(mapping.va)

        self._va_preserved = True
        self._unmapped = True

        self._client_rpc.commit()
        self._client = None
        self._granted_lock_type = None
        return True

    def commit_layout(self) -> "LayoutCommit":
        """Seal the allocation set: the shape is final, the pages outlive this session.

        The counterpart to :meth:`commit`, which publishes *contents*: it unmaps the
        writer, closes the session, and lets readers attach. This publishes only the
        *shape*, leaving mappings and session intact so the caller keeps writing.

        Call it once the pool is built. That call is the atomic boundary: die before it
        and the half-built pool is discarded, die after and a standby can adopt it.

        The session is narrowed to RW_DATA, so allocate and free now raise. Reconnect
        with RW to build a different layout.
        """
        self._require_rw()
        # Publish barrier, matching commit(): make this process's GPU writes visible
        # before the layout is advertised as reattachable.
        self._vmm.synchronize()
        response = self._client.commit_layout()
        # The server decides what we hold now; read it back rather than assuming.
        self._granted_lock_type = self._client.lock_type
        self._last_memory_layout_hash = response.memory_layout_hash
        return LayoutCommit(
            memory_layout_hash=response.memory_layout_hash,
            granted_lock_type=self._granted_lock_type,
        )

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

    def metadata_get_all(self, prefix: str = "") -> List[tuple[str, str, int, bytes]]:
        """Every (key, allocation_id, offset_bytes, value) under ``prefix``, in key order."""
        return self._client_rpc.metadata_get_all(prefix)

    def metadata_delete(self, key: str) -> bool:
        return self._client_rpc.metadata_delete(key)

    # ==================== Tier 1: VA Operations (local) ====================

    def reserve_va(self, size: int) -> int:
        """Reserve virtual address space (cuMemAddressReserve). No tracking.

        With a fixed-VA window, carves a client range from it instead.
        """
        aligned_size = align_to_granularity(size, self.granularity)
        return self._reserve_client_va(aligned_size, self.granularity)

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
        handle = self._vmm.import_shareable_handle_close_fd(
            fd, import_size=aligned_size
        )
        # All or nothing, so the caller can return the VA and retry.
        try:
            self._vmm.map(va, aligned_size, handle)
            try:
                self._vmm.set_access(
                    va, aligned_size, self.device, self._granted_lock_type
                )
            except BaseException:
                self._vmm.unmap(va, aligned_size)
                raise
        except BaseException:
            self._vmm.release(handle)
            raise
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
        self._vmm.unmap(va, mapping.aligned_size)
        self._vmm.release(mapping.handle)
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
        self._release_va(va, mapping.va_reserved_size)
        self._mappings.pop(va, None)
        self._inverse_mapping.pop(mapping.allocation_id, None)

    # ==================== Fixed-VA window ====================

    def _reserve_fixed_window(self, window: _FixedVAWindow) -> None:
        if window.reserved:
            return
        g = self.granularity
        if window.base <= 0 or window.size <= 0 or window.base % g or window.size % g:
            # An unaligned placement hint is a fatal driver error, not a miss.
            raise ValueError(
                f"[GMS] fixed VA window 0x{window.base:x} (+0x{window.size:x}) for "
                f"tag {self.tag!r} must be non-empty and aligned to 0x{g:x}"
            )
        va = int(self._vmm.address_reserve(window.size, g, addr=window.base))
        if va != window.base:
            # A hint that overlaps anything already reserved succeeds elsewhere.
            self._vmm.address_free(va, window.size)
            raise RuntimeError(
                f"[GMS] fixed VA window 0x{window.base:x} (+0x{window.size:x}) for "
                f"tag {self.tag!r} is not available; the driver returned 0x{va:x}"
            )
        window.reserved = True
        logger.info(
            "[GMS] Reserved fixed VA window 0x%x (+0x%x) for tag %r",
            window.base,
            window.size,
            self.tag,
        )

    def _release_fixed_window(self, window: _FixedVAWindow) -> None:
        if window.reserved:
            self._vmm.address_free(window.base, window.size)
            window.reserved = False
        window.clear()
        self._fixed_slot_offsets = None

    def _reserve_client_va(self, size: int, alignment: int) -> int:
        if self._fixed is None:
            return self._vmm.address_reserve(size, alignment)
        self._reserve_fixed_window(self._fixed)
        return self._fixed.carve_top_down(size, alignment)

    def _reserve_import_va(self, layout_slot: int, aligned_size: int) -> int:
        """Reserve the VA of an imported allocation.

        In a fixed-VA window it depends only on the committed layout, not on
        import order or on what else this process has mapped.
        """
        window = self._fixed
        if window is None:
            return self.reserve_va(aligned_size)
        self._reserve_fixed_window(window)
        if self._fixed_slot_offsets is None or not window.imports:
            offsets, cursor = {}, 0
            for info in sorted(self.list_handles(), key=lambda h: int(h.layout_slot)):
                offsets[int(info.layout_slot)] = cursor
                cursor += align_to_granularity(int(info.aligned_size), self.granularity)
            self._fixed_slot_offsets = offsets
        offset = self._fixed_slot_offsets.get(layout_slot)
        if offset is None:
            raise RuntimeError(
                f"[GMS] layout slot {layout_slot} is not in the committed layout"
            )
        return window.carve(window.base + offset, aligned_size, self.granularity)

    def _release_va(self, va: int, size: int) -> None:
        """Return a VA from _reserve_client_va or _reserve_import_va."""
        if self._fixed is None:
            self._vmm.address_free(va, size)
        else:
            self._fixed.release(va)

    # ==================== Tier 2: Convenience ====================

    def _bulk_map(self, entries: List[tuple[int, int, str]]) -> List[int]:
        """Map server allocations in phases and return their handles in order.

        ``entries``: (va, aligned_size, allocation_id). Each chunk is exported,
        then imported, then mapped; access is set last, once per mapping. One
        phase over many allocations avoids alternating client and server driver
        calls for every allocation.

        All or nothing for a failure that propagates, such as a failed export
        or an interrupt: every fd, handle and mapping made here is released
        first. A CUDA driver error does not propagate: ``cuda_check_result``
        ends the process.
        """
        if self._granted_lock_type is None:
            raise RuntimeError("Memory manager holds no lock")
        fds: List[int] = []
        handles: List[int] = []
        mapped: List[tuple[int, int]] = []
        try:
            lo = 0
            while lo < len(entries):
                chunk = entries[lo : lo + self._bulk_chunk_size()]
                lo += len(chunk)
                fds = []
                for _, _, allocation_id in chunk:
                    fds.append(self.export_handle(allocation_id))
                for i, (_, aligned_size, _) in enumerate(chunk):
                    # The import closes the fd even when it fails.
                    fd, fds[i] = fds[i], -1
                    handles.append(
                        self._vmm.import_shareable_handle_close_fd(
                            fd, import_size=aligned_size
                        )
                    )
                fds = []
                for va, aligned_size, _ in chunk:
                    self._vmm.map(va, aligned_size, handles[len(mapped)])
                    mapped.append((va, aligned_size))
            for va, aligned_size, _ in entries:
                self._vmm.set_access(
                    va, aligned_size, self.device, self._granted_lock_type
                )
        except BaseException:
            # A release that fails is logged and the unwind goes on: the caller
            # needs the first error, and the rest still has to be released.
            for fd in fds:
                if fd >= 0:
                    try:
                        os.close(fd)
                    except OSError:
                        logger.exception("[GMS] unwinding bulk import: close fd %d", fd)
            for va, size in reversed(mapped):
                try:
                    self._vmm.unmap(va, size)
                except Exception:
                    logger.exception("[GMS] unwinding bulk import: unmap 0x%x", va)
            for handle in handles:
                try:
                    self._vmm.release(handle)
                except Exception:
                    logger.exception("[GMS] unwinding bulk import: release %d", handle)
            raise
        return handles

    def _bulk_chunk_size(self) -> int:
        headroom = _fd_headroom()
        if headroom is None:
            return self._BULK_CHUNK
        return max(1, min(self._BULK_CHUNK, headroom - self._BULK_FD_MARGIN))

    def import_allocations(self, allocation_ids: Iterable[str]) -> None:
        """Import and map, in bulk, every listed allocation not tracked yet.

        Later ``create_mapping(allocation_id=...)`` calls return the cached VA.
        Each allocation gets the VA a per-allocation import would give it.
        All or nothing: on failure no reservation or mapping is left behind.
        """
        wanted = [
            a for a in dict.fromkeys(allocation_ids) if a not in self._inverse_mapping
        ]
        if not wanted:
            return
        infos = {str(info.allocation_id): info for info in self.list_handles()}
        entries: List[tuple[int, int, str]] = []
        try:
            for allocation_id in wanted:
                info = infos.get(allocation_id)
                if info is None:
                    raise RuntimeError(
                        f"Allocation {allocation_id} is not in the layout"
                    )
                aligned_size = int(info.aligned_size)
                va = self._reserve_import_va(int(info.layout_slot), aligned_size)
                entries.append((va, aligned_size, allocation_id))
            handles = self._bulk_map(entries)
        except BaseException:
            # As in _bulk_map: a failed release must not hide the first error.
            for va, aligned_size, _ in entries:
                try:
                    self._release_va(va, aligned_size)
                except Exception:
                    logger.exception("[GMS] unwinding bulk import: free 0x%x", va)
            raise
        for (va, aligned_size, allocation_id), handle in zip(entries, handles):
            info = infos[allocation_id]
            self._track_mapping(
                LocalMapping(
                    allocation_id=allocation_id,
                    va=va,
                    size=int(info.size),
                    aligned_size=aligned_size,
                    handle=handle,
                    tag=str(info.tag),
                    layout_slot=int(info.layout_slot),
                )
            )

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
            try:
                va = self._reserve_import_va(layout_slot, aligned_size)
            except BaseException:
                os.close(fd)
                raise
            try:
                self.map_va(fd, va, alloc_size, allocation_id, alloc_tag, layout_slot)
            except BaseException:
                self._release_va(va, aligned_size)
                raise
            return va

        # Allocate path
        if size <= 0:
            raise ValueError("size must be > 0 when allocation_id is None")
        alloc_id, layout_slot = self.allocate_handle(size, tag)
        aligned_size = align_to_granularity(size, self.granularity)
        try:
            fd = self.export_handle(alloc_id)
            try:
                va = self.reserve_va(aligned_size)
            except BaseException:
                os.close(fd)
                raise
            try:
                self.map_va(fd, va, size, alloc_id, tag, layout_slot)
            except BaseException:
                self._release_va(va, aligned_size)
                raise
        except BaseException:
            # A full window makes this reachable; an allocation left on the
            # server with no mapping here would be committed with the layout.
            self.free_handle(alloc_id)
            raise
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
        """Synchronize + unmap all VAs (real mappings AND scratch mappings).
        Preserves VA reservations for remap.
        """
        self._vmm.synchronize()

        unmapped_count = 0
        total_bytes = 0
        for va, mapping in list(self._mappings.items()):
            if mapping.handle == 0:
                continue
            self.unmap_va(va)
            unmapped_count += 1
            total_bytes += mapping.aligned_size

        # Scratch is 1 handle aliased N times across [base_va, +va_reserved_size).
        # cuMemUnmap over the whole range covers all aliases in one call.
        for base_va, scratch in self._scratch_mappings.items():
            if scratch.scratch_handle == 0:
                continue
            self._vmm.unmap(base_va, scratch.va_reserved_size)
            scratch.scratch_handle = 0
            unmapped_count += 1
            total_bytes += scratch.va_reserved_size
        # Every mapping aliased the one shared granule; release it once, after all
        # ranges are unmapped.
        if self._shared_scratch_handle != 0:
            self._vmm.release(self._shared_scratch_handle)
            self._shared_scratch_handle = 0

        self._va_preserved = True
        self._unmapped = True
        logger.info(
            "[GPU Memory Service] Unmapped %d allocations (%.2f GiB), "
            "preserving %d VA reservations",
            unmapped_count,
            total_bytes / (1 << 30),
            len(self._mappings) + len(self._scratch_mappings),
        )

    def remap_all_vas(self) -> None:
        """Re-import existing handles at preserved VAs.

        Checks layout hash for staleness. Validates each allocation still
        exists and size matches before remapping.
        """
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

        committed_allocations = sorted(
            self.list_handles(),
            key=lambda info: int(info.layout_slot),
        )
        local_mappings = sorted(
            self._mappings.items(), key=lambda item: item[1].layout_slot
        )
        if len(committed_allocations) != len(local_mappings):
            raise StaleMemoryLayoutError(
                "Layout allocation count changed: "
                f"{len(local_mappings)} vs {len(committed_allocations)}"
            )

        remapped_count = 0
        total_bytes = 0
        remapped_vas: list[int] = []
        pending: List[tuple[int, LocalMapping, GetAllocationResponse]] = []
        for rank, ((va, mapping), alloc_info) in enumerate(
            zip(local_mappings, committed_allocations)
        ):
            if mapping.handle != 0:
                continue

            if int(alloc_info.aligned_size) != mapping.aligned_size:
                raise StaleMemoryLayoutError(
                    f"Layout rank {rank} size changed: "
                    f"{mapping.aligned_size} vs {int(alloc_info.aligned_size)}"
                )
            if str(alloc_info.tag) != mapping.tag:
                raise StaleMemoryLayoutError(
                    f"Layout rank {rank} tag changed: {mapping.tag} vs {alloc_info.tag}"
                )

            if self.bulk_import:
                pending.append((va, mapping, alloc_info))
                continue
            fd = self.export_handle(alloc_info.allocation_id)
            handle = self._vmm.import_shareable_handle_close_fd(
                fd, import_size=mapping.aligned_size
            )
            self._vmm.map(va, mapping.aligned_size, handle)
            self._vmm.set_access(
                va, mapping.aligned_size, self.device, self._granted_lock_type
            )
            remapped_vas.append(va)

            self._record_remap(va, mapping, alloc_info, handle)
            remapped_count += 1
            total_bytes += mapping.aligned_size

        if pending:
            handles = self._bulk_map(
                [(va, m.aligned_size, str(i.allocation_id)) for va, m, i in pending]
            )
            for (va, mapping, alloc_info), handle in zip(pending, handles):
                remapped_vas.append(va)
                self._record_remap(va, mapping, alloc_info, handle)
                remapped_count += 1
                total_bytes += mapping.aligned_size

        if remapped_vas:
            self._vmm.synchronize()
            for va in remapped_vas:
                self._vmm.validate_pointer(va)

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

            response = self._client_rpc.allocate_info(mapping.size, tag)
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

    # ==================== Scratch-aliased mappings ====================

    def create_scratch_mapping(self, size: int, tag: str = "kv_cache") -> int:
        """Reserve VA range and back it with ONE aliased physical chunk.

        Purely client-local — does not require a GMS server connection.

        Used by the shadow engine at init so torch.zeros on the full kv_cache
        size succeeds without paying the real memory cost. The shadow then
        sleeps (unmap_all_vas drops scratch physical, preserves VAs) and
        wakes by moving scratch bookkeeping into _mappings via
        prepare_scratch_for_reallocation; reallocate_all_handles + remap_all_vas
        then install fresh server backing at the same VAs.

        Cudagraphs capture VAs, not physical, so the swap is invisible to
        replay.
        """
        # Coarse scratch aliases keep CUDA VMM map/access metadata bounded.
        # Committed GMS allocations still use CUDA's reported granularity.
        if self.scratch_size < self.granularity:
            raise ValueError(
                "Scratch size must be at least CUDA's allocation granularity: "
                f"{self.scratch_size} < {self.granularity}"
            )
        if self.scratch_size % self.granularity != 0:
            raise ValueError(
                "Scratch size must be a multiple of CUDA's allocation granularity: "
                f"{self.scratch_size} is not divisible by {self.granularity}"
            )
        if self.scratch_size & (self.scratch_size - 1):
            raise ValueError(
                "Scratch size must be a power of two because it is used as a "
                f"VA reservation alignment, got {self.scratch_size}"
            )
        aligned_size = align_to_granularity(size, self.granularity)
        va_reserved_size = align_to_granularity(size, self.scratch_size)

        if self._shared_scratch_handle != 0:
            # Reuse the one shared granule; every mapping aliases it.
            scratch_handle = self._shared_scratch_handle
        else:
            ok, scratch_handle = self._vmm.create_tolerate_oom(
                self.scratch_size, self.device
            )
            if not ok:
                raise RuntimeError(
                    f"VMM physical memory allocation failed "
                    f"({self.scratch_size // (1 << 20)} MiB) on "
                    f"{self.device_type.value} device {self.device}"
                )
            self._shared_scratch_handle = scratch_handle

        try:
            va = self._reserve_client_va(va_reserved_size, self.scratch_size)
        except BaseException:
            # The block is otherwise released with the last scratch mapping.
            if not self._scratch_mappings:
                self._vmm.release(self._shared_scratch_handle)
                self._shared_scratch_handle = 0
            raise
        for offset in range(0, va_reserved_size, self.scratch_size):
            self._vmm.map(va + offset, self.scratch_size, scratch_handle)
        self._vmm.set_access(va, va_reserved_size, self.device, GrantedLockType.RW)

        self._scratch_mappings[va] = _ScratchMapping(
            size=size,
            aligned_size=aligned_size,
            va_reserved_size=va_reserved_size,
            tag=tag,
            scratch_handle=scratch_handle,
        )
        logger.info(
            "[GMS] Reserved %d MiB VA at 0x%x, aliased a %d MiB scratch block across %d chunks",
            va_reserved_size // (1 << 20),
            va,
            self.scratch_size // (1 << 20),
            va_reserved_size // self.scratch_size,
        )
        return va

    def scratch_summary(self) -> tuple[int, int, int]:
        """Return (count, virtual_bytes, physical_bytes) of live scratch mappings.

        virtual_bytes is the VA range reserved; physical_bytes is the DRAM
        actually committed (distinct scratch blocks * scratch_size), far smaller
        since each mapping aliases one block across its whole range.
        """
        mappings = self._scratch_mappings.values()
        virtual = sum(m.size for m in mappings)
        live_blocks = {m.scratch_handle for m in mappings if m.scratch_handle}
        physical = len(live_blocks) * self.scratch_size
        return len(self._scratch_mappings), virtual, physical

    def prepare_scratch_for_reallocation(self) -> None:
        """Move scratch bookkeeping into _mappings as preserved-VA records.

        Pre-condition: scratch was already torn down by unmap_all_vas during
        sleep, so every entry's scratch_handle == 0. Each entry becomes a
        LocalMapping(handle=0) under its base_va so the standard
        reallocate_all_handles + remap_all_vas pipeline produces real backing
        at the preserved VA.

        No CUDA driver calls and no server RPCs. Does not write to
        _inverse_mapping; reallocate_all_handles populates it when it assigns
        the real allocation_id. If this manager is registered with the torch
        allocator, also flips future allocations for the tag to server-backed
        routing.
        """
        for base_va, scratch in self._scratch_mappings.items():
            if scratch.scratch_handle != 0:
                raise RuntimeError(
                    "prepare_scratch_for_reallocation requires scratch to be "
                    "unmapped first: "
                    f"base_va=0x{base_va:x} scratch_handle={scratch.scratch_handle}"
                )

        for base_va, scratch in list(self._scratch_mappings.items()):
            self._mappings[base_va] = LocalMapping(
                allocation_id="",
                va=base_va,
                size=scratch.size,
                aligned_size=scratch.aligned_size,
                handle=0,
                tag=scratch.tag,
                layout_slot=0,
                va_reserved_size=scratch.va_reserved_size,
            )
        moved = len(self._scratch_mappings)
        self._scratch_mappings.clear()
        if moved:
            logger.info(
                "[GMS] Moved %d scratch VA records into _mappings for reallocation",
                moved,
            )
        if self.tag is not None:
            from gpu_memory_service.client.torch.allocator import _tag_states

            state = _tag_states.get(self.tag)
            if state is not None and state.manager is self:
                # RW_DATA counts: an adopting standby must move the same bookkeeping
                # before it can remap, and this touches neither driver nor server. The
                # server still refuses the allocations that routing would request.
                if self.granted_lock_type not in (
                    GrantedLockType.RW,
                    GrantedLockType.RW_DATA,
                ):
                    raise RuntimeError(
                        "prepare_scratch_for_reallocation requires a writer grant "
                        "before disabling scratch routing: "
                        f"tag={self.tag!r} "
                        f"granted_lock_type={self.granted_lock_type}"
                    )
                state.is_scratch = False

    def destroy_scratch_mapping(self, base_va: int) -> bool:
        """Tear down a scratch entry.

        Called from _gms_free when freeing a VA tracked in _scratch_mappings.
        Returns True if the VA was a scratch entry and was destroyed, False if
        the VA was not tracked (caller falls through to destroy_mapping).
        """
        scratch = self._scratch_mappings.pop(base_va, None)
        if scratch is None:
            return False

        self._vmm.synchronize()
        if scratch.scratch_handle:
            self._vmm.unmap(base_va, scratch.va_reserved_size)
            # Shared granule: release only once the last mapping is gone
            # (this entry was already popped above).
            if self._shared_scratch_handle != 0 and not self._scratch_mappings:
                self._vmm.release(self._shared_scratch_handle)
                self._shared_scratch_handle = 0
        self._release_va(base_va, scratch.va_reserved_size)
        return True

    # ==================== Lifecycle ====================

    def close(self, *, best_effort: bool = False) -> None:
        """Cleanup mappings and abort.

        synchronize + unmap all + free all VAs + abort.

        Args:
            best_effort: If True, skip self._vmm.synchronize() and swallow
                errors during cleanup. Used after checkpoint where
                cuda-checkpoint may have torn down the device context
                (self._vmm.synchronize() calls os._exit via fail()).
        """
        if best_effort:
            try:
                self.abort()
            except Exception:
                pass
            self._mappings.clear()
            self._inverse_mapping.clear()
            self._scratch_mappings.clear()
            self._shared_scratch_handle = 0
        else:
            self._vmm.synchronize()
            for base_va in list(self._scratch_mappings.keys()):
                self.destroy_scratch_mapping(base_va)
            for va in list(self._mappings.keys()):
                self.unmap_va(va)
                self.free_va(va)
            if self._fixed is not None:
                self._release_fixed_window(self._fixed)
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

    def _record_remap(
        self,
        va: int,
        mapping: LocalMapping,
        alloc_info: GetAllocationResponse,
        handle: int,
    ) -> None:
        if mapping.allocation_id != alloc_info.allocation_id:
            self._inverse_mapping.pop(mapping.allocation_id, None)
        self._mappings[va] = mapping.with_server_identity(
            alloc_info.allocation_id,
            int(alloc_info.layout_slot),
        ).with_handle(handle)
        self._inverse_mapping[alloc_info.allocation_id] = va
