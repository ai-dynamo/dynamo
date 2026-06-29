# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persistent CUDA allocation store — namespace separate from the
RW/RO/COMMITTED layout machinery.

Persistent allocations are keyed by ``(engine_id, tag)``. They:
  - survive client disconnect (the engine process can crash and a new
    engine with the same ``engine_id`` reattaches to the same physical
    memory),
  - are NEVER cleared by ``RW_ABORT`` / ``RW_CONNECT`` (those reset the
    weights layout, not persistent allocations),
  - require explicit ``release(engine_id, tag)`` to free,
  - default to single-claimant exclusivity per ``(engine_id, tag)`` while
    a client is attached,
  - optionally allow shared claims for cooperating engines that coordinate
    writes through the KV lease table.

Used for VMM-IPC KV cache pools: the engine maps the FD into its own
address space via ``cuMemImportFromShareableHandle`` + ``cuMemMap`` and
writes KV bytes every forward pass.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Iterable, Optional
from uuid import uuid4

from gpu_memory_service.common.cuda_utils import (
    align_to_granularity,
    cuda_ensure_initialized,
    cumem_address_free,
    cumem_address_reserve,
    cumem_create_tolerate_oom,
    cumem_export_to_shareable_handle,
    cumem_get_allocation_granularity,
    cumem_map,
    cumem_release,
    cumem_set_access,
    cumem_unmap,
)
from gpu_memory_service.common.locks import GrantedLockType

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PersistentAllocation:
    allocation_id: str
    engine_id: str
    tag: str
    size: int
    aligned_size: int
    handle: int
    export_fd: int
    # Daemon-side mapped virtual address. Reading/writing through this
    # VA reads/writes the SAME PHYSICAL PAGES the engine sees (via the
    # exported FD). Sync against engine writes via the existing event
    # / counter protocol — concurrency is the caller's responsibility.
    va_daemon: int = 0


class PersistentClaimConflictError(Exception):
    """A second concurrent claimant tried to attach to a (engine_id,
    tag) key while another claimant holds it."""


class PersistentNotFoundError(Exception):
    """No persistent allocation exists for the requested
    (engine_id, tag) key."""


class PersistentAllocationManager:
    """Server-side store for persistent CUDA VMM allocations.

    Allocations are keyed by ``(engine_id, tag)``. Single-claimant
    exclusivity is enforced by default: while a client has an exclusive
    open claim, any other client claiming the same key is rejected. Shared
    claims are opt-in and intended only for cooperating engines that use
    KV block leases before writing into the mapped pool. On client
    disconnect the claim is released but the allocation persists; the
    next claimant with the matching key reattaches.
    """

    def __init__(self, device: int = 0):
        cuda_ensure_initialized()
        self._device = device
        self._granularity = cumem_get_allocation_granularity(device)
        self._allocations: dict[tuple[str, str], PersistentAllocation] = {}
        self._exclusive_claimed: set[tuple[str, str]] = set()
        self._shared_claim_counts: dict[tuple[str, str], int] = {}
        logger.info(
            "PersistentAllocationManager initialized: device=%d granularity=%d",
            device,
            self._granularity,
        )

    @property
    def device(self) -> int:
        return self._device

    @property
    def active_claim_count(self) -> int:
        """Number of distinct (engine_id, tag) keys currently claimed.

        A key is "active" while any claimant (exclusive or shared) holds it.
        The kv_cache daemon projects this onto its reported runtime-state so
        persistent KV — which bypasses the single-writer FSM — is observable to
        layout assertions: claims present => an active (RW) KV layout exists;
        claims fully released (engine pause / crash cleanup) => no layout.
        """
        shared = {key for key, count in self._shared_claim_counts.items() if count > 0}
        return len(self._exclusive_claimed | shared)

    @property
    def has_active_claims(self) -> bool:
        return self.active_claim_count > 0

    @property
    def granularity(self) -> int:
        return self._granularity

    def _check_claim_allowed(
        self,
        key: tuple[str, str],
        *,
        shared: bool,
    ) -> None:
        if shared:
            if key in self._exclusive_claimed:
                raise PersistentClaimConflictError(
                    f"persistent allocation {key!r} already claimed exclusively"
                )
            return
        if key in self._exclusive_claimed or self._shared_claim_counts.get(key, 0) > 0:
            raise PersistentClaimConflictError(
                f"persistent allocation {key!r} already claimed"
            )

    def _mark_claimed(self, key: tuple[str, str], *, shared: bool) -> None:
        if shared:
            self._shared_claim_counts[key] = self._shared_claim_counts.get(key, 0) + 1
        else:
            self._exclusive_claimed.add(key)

    def claim(
        self,
        engine_id: str,
        tag: str,
        size: int,
        *,
        shared: bool = False,
    ) -> tuple[PersistentAllocation, bool]:
        """Claim a persistent allocation.

        If an allocation already exists for ``(engine_id, tag)``:
          - reject if claim modes conflict,
          - else mark claimed and return existing (reattach).

        Exclusive claims preserve the historical single-writer behavior.
        Shared claims are for multi-engine KV pools and must be paired with
        KV block leases for write safety.

        Returns ``(allocation, reattached)``.
        """
        if not engine_id:
            raise ValueError("engine_id must be non-empty")
        if not tag:
            raise ValueError("tag must be non-empty")
        if size <= 0:
            raise ValueError(f"size must be > 0, got {size}")

        key = (engine_id, tag)
        self._check_claim_allowed(key, shared=shared)

        existing = self._allocations.get(key)
        if existing is not None:
            self._mark_claimed(key, shared=shared)
            logger.info(
                "Reattached persistent allocation %s engine_id=%s tag=%s shared=%s",
                existing.allocation_id,
                engine_id,
                tag,
                shared,
            )
            return existing, True

        aligned_size = align_to_granularity(size, self._granularity)
        allocated, handle = cumem_create_tolerate_oom(aligned_size, self._device)
        if not allocated:
            raise MemoryError(
                f"cuMemCreate OOM for persistent ({engine_id!r}, {tag!r}) "
                f"size={size} aligned_size={aligned_size}"
            )
        export_fd = int(cumem_export_to_shareable_handle(int(handle)))
        # Also map into the daemon's own VA so the daemon can read/write
        # the same physical pages the engine sees. Failure to map is
        # non-fatal — we still hand out the FD; va_daemon stays 0 and
        # direct-access ops will raise.
        va_daemon = 0
        try:
            va_daemon = int(cumem_address_reserve(aligned_size, self._granularity))
            cumem_map(va_daemon, aligned_size, int(handle))
            cumem_set_access(
                va_daemon,
                aligned_size,
                self._device,
                GrantedLockType.RW,
            )
        except Exception:  # noqa: BLE001
            logger.warning(
                "Failed to map daemon-side VA for persistent (%r, %r); "
                "direct-access ops will be unavailable",
                engine_id,
                tag,
                exc_info=True,
            )
            if va_daemon:
                try:
                    cumem_address_free(va_daemon, aligned_size)
                except Exception:  # noqa: BLE001
                    pass
                va_daemon = 0
        alloc = PersistentAllocation(
            allocation_id=str(uuid4()),
            engine_id=engine_id,
            tag=tag,
            size=size,
            aligned_size=aligned_size,
            handle=int(handle),
            export_fd=export_fd,
            va_daemon=va_daemon,
        )
        self._allocations[key] = alloc
        self._mark_claimed(key, shared=shared)
        logger.info(
            "Created persistent allocation %s engine_id=%s tag=%s size=%d "
            "aligned=%d shared=%s",
            alloc.allocation_id,
            engine_id,
            tag,
            size,
            aligned_size,
            shared,
        )
        return alloc, False

    def unclaim(self, engine_id: str, tag: str) -> bool:
        """Release the claim on (engine_id, tag) without freeing the
        allocation. Used on client disconnect."""
        key = (engine_id, tag)
        if key in self._exclusive_claimed:
            self._exclusive_claimed.discard(key)
            logger.debug(
                "Unclaimed persistent allocation engine_id=%s tag=%s",
                engine_id,
                tag,
            )
            return True
        count = self._shared_claim_counts.get(key, 0)
        if count > 1:
            self._shared_claim_counts[key] = count - 1
            return True
        if count == 1:
            self._shared_claim_counts.pop(key, None)
            logger.debug(
                "Unclaimed shared persistent allocation engine_id=%s tag=%s",
                engine_id,
                tag,
            )
            return True
        return False

    def release(self, engine_id: str, tag: str) -> bool:
        """Free a persistent allocation. Removes both the claim (if
        held) and the underlying CUDA handle. Returns True iff an
        allocation was found and freed."""
        key = (engine_id, tag)
        if self._shared_claim_counts.get(key, 0) > 1:
            raise PersistentClaimConflictError(
                f"persistent allocation {key!r} still has other shared claimants"
            )
        alloc = self._allocations.pop(key, None)
        if alloc is None:
            return False
        self._exclusive_claimed.discard(key)
        self._shared_claim_counts.pop(key, None)
        # Tear down daemon-side mapping (if it succeeded at claim time).
        if alloc.va_daemon:
            try:
                cumem_unmap(alloc.va_daemon, alloc.aligned_size)
            except Exception:  # noqa: BLE001
                logger.warning(
                    "cuMemUnmap failed for %s", alloc.allocation_id, exc_info=True
                )
            try:
                cumem_address_free(alloc.va_daemon, alloc.aligned_size)
            except Exception:  # noqa: BLE001
                logger.warning(
                    "cuMemAddressFree failed for %s", alloc.allocation_id, exc_info=True
                )
        os.close(alloc.export_fd)
        cumem_release(alloc.handle)
        logger.info(
            "Released persistent allocation %s engine_id=%s tag=%s",
            alloc.allocation_id,
            engine_id,
            tag,
        )
        return True

    def export(self, engine_id: str, tag: str) -> tuple[PersistentAllocation, int]:
        """Return ``(allocation, dup_fd)``. The caller owns ``dup_fd``
        and must close it after passing to the client."""
        key = (engine_id, tag)
        alloc = self._allocations.get(key)
        if alloc is None:
            raise PersistentNotFoundError(
                f"persistent allocation ({engine_id!r}, {tag!r}) does not exist"
            )
        return alloc, os.dup(alloc.export_fd)

    def get(self, engine_id: str, tag: str) -> PersistentAllocation:
        key = (engine_id, tag)
        alloc = self._allocations.get(key)
        if alloc is None:
            raise PersistentNotFoundError(
                f"persistent allocation ({engine_id!r}, {tag!r}) does not exist"
            )
        return alloc

    def is_claimed(self, engine_id: str, tag: str) -> bool:
        key = (engine_id, tag)
        return (
            key in self._exclusive_claimed or self._shared_claim_counts.get(key, 0) > 0
        )

    def shared_claim_count(self, engine_id: str, tag: str) -> int:
        return self._shared_claim_counts.get((engine_id, tag), 0)

    def list(
        self,
        engine_id: Optional[str] = None,
    ) -> Iterable[PersistentAllocation]:
        allocations = self._allocations.values()
        if engine_id is None:
            return list(allocations)
        return [a for a in allocations if a.engine_id == engine_id]

    def clear_all(self) -> int:
        """Free every persistent allocation. Test / shutdown only."""
        count = len(self._allocations)
        for alloc in list(self._allocations.values()):
            if alloc.va_daemon:
                try:
                    cumem_unmap(alloc.va_daemon, alloc.aligned_size)
                except Exception:  # noqa: BLE001
                    pass
                try:
                    cumem_address_free(alloc.va_daemon, alloc.aligned_size)
                except Exception:  # noqa: BLE001
                    pass
            try:
                os.close(alloc.export_fd)
            except OSError:
                pass
            try:
                cumem_release(alloc.handle)
            except Exception:  # noqa: BLE001
                logger.warning(
                    "cumem_release failed during clear_all for %s",
                    alloc.allocation_id,
                    exc_info=True,
                )
        self._allocations.clear()
        self._exclusive_claimed.clear()
        self._shared_claim_counts.clear()
        return count

    # ------------------------------------------------------------------
    # Daemon-side direct access
    # ------------------------------------------------------------------
    #
    # These methods read/write the SAME PHYSICAL PAGES the engine sees,
    # via the daemon's own VA mapping. They do NOT memcpy across
    # processes — the engine and daemon both see the same bytes.
    # Synchronization against engine writes is the CALLER's job (use
    # the existing CUDA event / ring-counter protocol).

    def daemon_va(self, engine_id: str, tag: str) -> int:
        """Return the daemon-side VA for ``(engine_id, tag)``. Raises
        if no mapping exists (claim() can fail to map even though the
        allocation succeeded; in that case direct access is unavailable
        and the caller should fall back to memcpy through the FD)."""
        alloc = self.get(engine_id, tag)
        if not alloc.va_daemon:
            raise RuntimeError(
                f"persistent allocation ({engine_id!r}, {tag!r}) has no "
                f"daemon-side VA mapping; direct access unavailable"
            )
        return alloc.va_daemon

    def read_block(
        self,
        engine_id: str,
        tag: str,
        offset: int,
        size: int,
    ) -> bytes:
        """Read ``size`` bytes at ``offset`` from the daemon's mapping
        of the persistent allocation. Synchronous D2H copy.

        Caller is responsible for any required CUDA event synchronization
        against concurrent engine writes."""
        if size <= 0:
            raise ValueError(f"size must be > 0, got {size}")
        alloc = self.get(engine_id, tag)
        if offset < 0 or offset + size > alloc.aligned_size:
            raise ValueError(
                f"read_block out of range: offset={offset} size={size} "
                f"aligned_size={alloc.aligned_size}"
            )
        if not alloc.va_daemon:
            raise RuntimeError(
                f"persistent allocation ({engine_id!r}, {tag!r}) has no "
                f"daemon-side VA mapping; direct access unavailable"
            )
        # Lazy import to keep the module loadable in test envs where
        # the cuda-python sync memcpy helpers may be patched away.
        # Use cuMemcpyDtoH for sync D2H. cuda-python returns a tuple
        # of (err,) for void-returning funcs.
        import ctypes
        from ctypes import addressof, c_char, sizeof  # noqa: F401

        from cuda.bindings import driver as drv

        host_buf = (ctypes.c_ubyte * size)()
        (err,) = drv.cuMemcpyDtoH(
            host_buf,
            drv.CUdeviceptr(alloc.va_daemon + offset),
            size,
        )
        if err != drv.CUresult.CUDA_SUCCESS:
            _, msg = drv.cuGetErrorString(err)
            raise RuntimeError(f"cuMemcpyDtoH failed: {msg.decode() if msg else err}")
        return bytes(host_buf)

    def write_block(
        self,
        engine_id: str,
        tag: str,
        offset: int,
        data: bytes,
    ) -> None:
        """Write ``data`` at ``offset`` into the daemon's mapping of
        the persistent allocation. Synchronous H2D copy."""
        if not data:
            raise ValueError("data must be non-empty")
        size = len(data)
        alloc = self.get(engine_id, tag)
        if offset < 0 or offset + size > alloc.aligned_size:
            raise ValueError(
                f"write_block out of range: offset={offset} size={size} "
                f"aligned_size={alloc.aligned_size}"
            )
        if not alloc.va_daemon:
            raise RuntimeError(
                f"persistent allocation ({engine_id!r}, {tag!r}) has no "
                f"daemon-side VA mapping; direct access unavailable"
            )
        import ctypes

        from cuda.bindings import driver as drv

        host_buf = (ctypes.c_ubyte * size).from_buffer_copy(data)
        (err,) = drv.cuMemcpyHtoD(
            drv.CUdeviceptr(alloc.va_daemon + offset),
            host_buf,
            size,
        )
        if err != drv.CUresult.CUDA_SUCCESS:
            _, msg = drv.cuGetErrorString(err)
            raise RuntimeError(f"cuMemcpyHtoD failed: {msg.decode() if msg else err}")
