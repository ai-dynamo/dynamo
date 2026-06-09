# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Receive-side buffer pool for cross-node staging transfers.

One pinned host buffer per daemon, allocated at startup, registered
once with the NIXL transport. The daemon hands out offsets into this
buffer for each inbound transfer; the sender writes RDMA-style into
the offset; the receiver's notif callback reads back from the offset
and bridges into StagingTier.commit_or_reject.

Why a single pool, not per-transfer alloc:

  - NIXL `register_memory` triggers a metadata snapshot bump. Peers
    don't see new registrations until they re-fetch (cross-host) or
    re-add_remote_agent (loopback). One-time registration of a pool
    sidesteps that entirely; senders see all offsets within the pool
    as soon as initial metadata exchange completes.

  - Fixed memory budget: capacity is bounded at daemon startup,
    matching the StagingTier's own capacity discipline.

  - Free-list-then-bump allocator is good-enough for typical KV
    sizes; per-block allocations are tens of KiB to MiB. Fragmentation
    is acceptable at these grain sizes.

The pool is concurrency-safe (single lock). All operations are O(N)
in the free-list size, which stays small in practice."""

from __future__ import annotations

import ctypes
import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)


class StagingReceiveBuffer:
    """Pinned host buffer with a simple free-list + bump allocator."""

    def __init__(self, capacity_bytes: int) -> None:
        if capacity_bytes <= 0:
            raise ValueError(
                f"capacity_bytes must be > 0, got {capacity_bytes}",
            )
        self._capacity = int(capacity_bytes)
        # Hold the ctypes array so refcount keeps the backing storage.
        self._buf = (ctypes.c_ubyte * self._capacity)()
        self._base_ptr = ctypes.addressof(self._buf)
        # Bump pointer for fresh-region allocs.
        self._used = 0
        # Free regions returned via `free()`. Each entry is (offset, size).
        # Coalescing happens on free() so the list stays compact.
        self._free_list: list[tuple[int, int]] = []
        self._lock = threading.Lock()

    def base_ptr(self) -> int:
        return self._base_ptr

    def capacity(self) -> int:
        return self._capacity

    def bytes_free(self) -> int:
        with self._lock:
            free_in_list = sum(sz for _, sz in self._free_list)
            return (self._capacity - self._used) + free_in_list

    def alloc_many(self, sizes: "list[int]") -> "list[Optional[int]]":
        """Vectorized alloc: takes the lock ONCE, allocates N regions
        contiguously when possible. Designed for `fetch_remote` which
        reserves N regions per call. Amortizes lock acquisition cost."""
        out: list = []
        with self._lock:
            for size in sizes:
                if size <= 0 or size > self._capacity:
                    out.append(None)
                    continue
                placed = False
                for i, (off, sz) in enumerate(self._free_list):
                    if sz < size:
                        continue
                    if sz == size:
                        self._free_list.pop(i)
                    else:
                        self._free_list[i] = (off + size, sz - size)
                    out.append(off)
                    placed = True
                    break
                if placed:
                    continue
                if self._used + size <= self._capacity:
                    off = self._used
                    self._used += size
                    out.append(off)
                else:
                    out.append(None)
        return out

    def alloc(self, size: int) -> Optional[int]:
        """Reserve `size` bytes; returns the offset, or None on
        out-of-memory."""
        if size <= 0:
            raise ValueError(f"size must be > 0, got {size}")
        if size > self._capacity:
            return None
        with self._lock:
            # First-fit on the free list.
            for i, (off, sz) in enumerate(self._free_list):
                if sz < size:
                    continue
                # Use this region.
                if sz == size:
                    self._free_list.pop(i)
                else:
                    self._free_list[i] = (off + size, sz - size)
                return off
            # Bump.
            if self._used + size <= self._capacity:
                off = self._used
                self._used += size
                return off
            return None

    def free(self, offset: int, size: int) -> None:
        """Return a previously-allocated region to the pool. Coalesces
        with adjacent free regions."""
        if size <= 0:
            return
        with self._lock:
            # Insert and coalesce.
            self._free_list.append((int(offset), int(size)))
            self._free_list.sort()
            coalesced: list[tuple[int, int]] = []
            for off, sz in self._free_list:
                if coalesced and coalesced[-1][0] + coalesced[-1][1] == off:
                    last_off, last_sz = coalesced[-1]
                    coalesced[-1] = (last_off, last_sz + sz)
                else:
                    coalesced.append((off, sz))
            self._free_list = coalesced
            # If the highest region is at the end, fold it back into bump.
            if self._free_list and (
                self._free_list[-1][0] + self._free_list[-1][1] == self._used
            ):
                tail_off, tail_sz = self._free_list.pop()
                self._used = tail_off

    def ptr_at(self, offset: int) -> int:
        """Absolute pointer for `offset`. Used by NIXL `remote_ptr`."""
        if offset < 0 or offset >= self._capacity:
            raise ValueError(
                f"offset {offset} out of range [0, {self._capacity})",
            )
        return self._base_ptr + int(offset)

    def read(self, offset: int, size: int) -> bytes:
        """Copy `size` bytes from `offset` into a fresh Python bytes
        object. Used by the inbound notif callback to extract the
        payload after a NIXL WRITE landed."""
        if offset < 0 or offset + size > self._capacity:
            raise ValueError(
                f"read [{offset}, {offset + size}) out of range "
                f"[0, {self._capacity})",
            )
        view = (ctypes.c_ubyte * size).from_address(self._base_ptr + offset)
        return bytes(view)


__all__ = ["StagingReceiveBuffer"]
