# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host tier — pinned RAM-backed offload storage.

Slot keying: (engine_id, layer_idx, byte_offset) → pinned host pointer.

Allocation strategy: per-spill allocation via `cudaHostAlloc`. Slots
are reclaimed on `release_slot` (engine-driven), daemon shutdown, or
the optional daemon-level bounded-capacity enforcement.

The default bounded policy is coarse slot-level LRU, matching the
original behavior. `lfu` and `tiny_lfu` add frequency-aware choices for
CPU-RAM pressure without changing engine ownership of HBM KV.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class _Slot:
    host_ptr: int
    size: int
    # True iff the D2H copy into this slot has been synchronized
    # — i.e., the bytes are committed. `HostTier.get` returns None
    # for not-ready slots so a restore consumer treats them as
    # "missing" rather than silently reading partial bytes.
    ready: bool = False
    # CRC32 of the slot bytes, captured by the evict consumer after
    # D2H sync. Restore consumer recomputes and compares before H2D;
    # mismatch means the host-tier bytes were corrupted between
    # offload and restore (DMA tear, bitrot, stale slot read).
    # Zero before the evict consumer has committed.
    crc: int = 0
    # Monotonic source-slot generation captured by the engine-side prefix
    # index when the block was mirrored. Restore callers can require an
    # expected generation so stale CPU mirrors fail closed.
    generation: int = 0
    # Policy metadata. HostTier updates these under `_lock` on
    # put/mark_ready/get/pin so bounded eviction can prefer older or
    # colder slots without consulting engine state.
    access_count: int = 0
    last_access_seq: int = 0
    created_seq: int = 0
    # Reader pins prevent cudaFreeHost while a restore, demote, scrub,
    # or bootstrap registration is reading from this pinned buffer.
    pins: int = 0
    retired: bool = False
    free_when_unpinned: bool = True
    writing: bool = False


@dataclass(frozen=True)
class HostTierWrite:
    """Identity token for one provisional host-tier write."""

    key: tuple[str, int, int]
    slot: _Slot

    @property
    def host_ptr(self) -> int:
        return self.slot.host_ptr


class _SlotLease:
    """Process-local pin for a ready host-tier slot.

    This is the host-tier analogue of Pegaflow's server-side block
    lease: eviction and explicit release make the key unavailable, but
    the underlying pinned pointer is not freed until active readers
    drop their lease.
    """

    def __init__(
        self,
        tier: "HostTier",
        key: tuple[str, int, int],
        slot: _Slot,
    ) -> None:
        self._tier = tier
        self._key = key
        self._slot = slot
        self._released = False

    def __enter__(self) -> _Slot:
        return self._slot

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()

    @property
    def key(self) -> tuple[str, int, int]:
        return self._key

    @property
    def slot(self) -> _Slot:
        return self._slot

    def release(self) -> None:
        to_free: Optional[_Slot] = None
        with self._tier._lock:
            if self._released:
                return
            self._released = True
            if self._slot.pins > 0:
                self._slot.pins -= 1
            if (
                self._slot.pins == 0
                and self._slot.retired
                and self._slot.free_when_unpinned
            ):
                self._slot.free_when_unpinned = False
                to_free = self._slot
        if to_free is not None:
            _free_host(to_free.host_ptr)


class HostTier:
    """Pinned RAM offload pool keyed by (engine_id, layer, offset)."""

    _VALID_EVICTION_POLICIES = frozenset(("lru", "lfu", "tiny_lfu"))

    def __init__(self, eviction_policy: str = "lru") -> None:
        self._eviction_policy = self.normalize_eviction_policy(
            eviction_policy,
        )
        self._slots: OrderedDict[tuple[str, int, int], _Slot] = OrderedDict()
        self._frequency: dict[tuple[str, int, int], int] = {}
        self._access_seq = 0
        self._frequency_events = 0
        self._bytes = 0
        self._lock = threading.Lock()

    @classmethod
    def normalize_eviction_policy(cls, policy: str) -> str:
        """Normalize host-tier policy names accepted from config/env."""
        normalized = str(policy or "lru").strip().lower().replace("-", "_")
        if normalized == "tinylfu":
            normalized = "tiny_lfu"
        if normalized not in cls._VALID_EVICTION_POLICIES:
            raise ValueError(
                "unknown host-tier eviction policy "
                f"{policy!r}; expected one of "
                f"{sorted(cls._VALID_EVICTION_POLICIES)}",
            )
        return normalized

    @property
    def eviction_policy(self) -> str:
        return self._eviction_policy

    def _touch_locked(
        self,
        key: tuple[str, int, int],
        slot: _Slot,
    ) -> None:
        self._access_seq += 1
        slot.access_count += 1
        slot.last_access_seq = self._access_seq
        if slot.created_seq <= 0:
            slot.created_seq = self._access_seq
        self._frequency[key] = min(
            self._frequency.get(key, 0) + 1,
            1 << 30,
        )
        self._frequency_events += 1
        if self._frequency_events >= 8192:
            self._age_frequency_sketch_locked()

    def _age_frequency_sketch_locked(self) -> None:
        """Age the tiny-LFU sketch so stale historical heat decays."""
        self._frequency_events = 0
        for key, count in list(self._frequency.items()):
            aged = count >> 1
            if aged <= 0 and key not in self._slots:
                self._frequency.pop(key, None)
            else:
                self._frequency[key] = max(1, aged)

    def _retire_locked(
        self,
        key: tuple[str, int, int],
        *,
        free_when_unpinned: bool,
    ) -> Optional[_Slot]:
        slot = self._slots.pop(key, None)
        if slot is None:
            return None
        self._bytes -= slot.size
        if slot.pins > 0:
            slot.retired = True
            slot.free_when_unpinned = bool(free_when_unpinned)
            return None
        return slot if free_when_unpinned else None

    def _eviction_order_locked(
        self,
        policy: str,
        protected: set[tuple[str, int, int]],
    ) -> list[tuple[tuple[str, int, int], _Slot]]:
        candidates = [
            (key, slot)
            for key, slot in self._slots.items()
            if key not in protected and slot.ready and slot.pins == 0
        ]
        if policy == "lru":
            return candidates
        if policy == "lfu":
            return sorted(
                candidates,
                key=lambda item: (
                    item[1].access_count,
                    item[1].last_access_seq,
                    item[1].created_seq,
                ),
            )
        # TinyLFU-style bounded host tier: use an aged frequency sketch
        # to bias eviction against historically hot KV, with resident LFU
        # and LRU as deterministic tie-breakers. This deliberately avoids
        # a two-phase admission protocol; the daemon still protects
        # in-flight writes/promotes from immediate self-eviction.
        return sorted(
            candidates,
            key=lambda item: (
                self._frequency.get(item[0], item[1].access_count),
                item[1].access_count,
                item[1].last_access_seq,
                item[1].created_seq,
            ),
        )

    def put(
        self,
        engine_id: str,
        layer: int,
        offset: int,
        size: int,
    ) -> int:
        """Compatibility wrapper for callers that commit via mark_ready."""
        return self.reserve(engine_id, layer, offset, size).host_ptr

    def reserve(
        self,
        engine_id: str,
        layer: int,
        offset: int,
        size: int,
    ) -> HostTierWrite:
        """Reserve an identity-pinned slot for DMA or backend I/O.

        Commit or abort with the returned token. A concurrent replacement
        cannot publish or delete this allocation by key accidentally.
        """
        key = (engine_id, int(layer), int(offset))
        old_to_free: Optional[_Slot] = None
        with self._lock:
            existing = self._slots.get(key)
            if existing is not None and existing.size == size and existing.pins == 0:
                existing.ready = False
                existing.crc = 0
                existing.generation = 0
                self._touch_locked(key, existing)
                self._slots.move_to_end(key)
                existing.pins = 1
                existing.writing = True
                return HostTierWrite(key, existing)
            host_ptr = _alloc_host(size)
            if existing is not None:
                old_to_free = self._retire_locked(
                    key,
                    free_when_unpinned=True,
                )
            slot = _Slot(host_ptr=host_ptr, size=int(size), pins=1, writing=True)
            self._slots[key] = slot
            self._touch_locked(key, slot)
            self._slots.move_to_end(key)
            self._bytes += int(size)
        if old_to_free is not None:
            _free_host(old_to_free.host_ptr)
        return HostTierWrite(key, slot)

    def commit(
        self,
        write: HostTierWrite,
        crc: int = 0,
        *,
        generation: int = 0,
    ) -> bool:
        """Publish a provisional write only if it is still current."""
        to_free: Optional[_Slot] = None
        with self._lock:
            slot = write.slot
            if not slot.writing:
                return False
            slot.writing = False
            if slot.pins > 0:
                slot.pins -= 1
            if self._slots.get(write.key) is not slot:
                if slot.retired and slot.pins == 0 and slot.free_when_unpinned:
                    slot.free_when_unpinned = False
                    to_free = slot
                committed = False
            else:
                slot.ready = True
                slot.crc = int(crc) & 0xFFFFFFFF
                slot.generation = int(generation or 0)
                self._touch_locked(write.key, slot)
                self._slots.move_to_end(write.key)
                committed = True
        if to_free is not None:
            _free_host(to_free.host_ptr)
        return committed

    def abort(self, write: HostTierWrite) -> bool:
        """Discard a provisional write without touching a replacement."""
        to_free: Optional[_Slot] = None
        with self._lock:
            slot = write.slot
            if not slot.writing:
                return False
            slot.writing = False
            if slot.pins > 0:
                slot.pins -= 1
            current = self._slots.get(write.key) is slot
            if current:
                to_free = self._retire_locked(write.key, free_when_unpinned=True)
            elif slot.retired and slot.pins == 0 and slot.free_when_unpinned:
                slot.free_when_unpinned = False
                to_free = slot
        if to_free is not None:
            _free_host(to_free.host_ptr)
        return current

    def get(
        self,
        engine_id: str,
        layer: int,
        offset: int,
    ) -> Optional[_Slot]:
        """Returns the slot ONLY if it's been marked ready (DMA committed).
        Returns None for slots that are mid-spill — the restore consumer
        treats this the same as "no such slot" and signals failure to
        the engine instead of reading partial bytes.

        `get` does not pin the slot. Use `pin` when the returned pointer
        may be read after this method returns.
        """
        key = (engine_id, int(layer), int(offset))
        with self._lock:
            slot = self._slots.get(key)
            if slot is not None and slot.ready:
                self._touch_locked(key, slot)
                self._slots.move_to_end(key)
        if slot is None or not slot.ready:
            return None
        return slot

    def pin(
        self,
        engine_id: str,
        layer: int,
        offset: int,
        *,
        expected_generation: int = 0,
    ) -> Optional[_SlotLease]:
        """Return a lease for a ready slot, or None if unavailable.

        While leased, eviction skips the slot and explicit release
        defers cudaFreeHost until the lease is released. The key can
        still be removed from the index immediately, so new readers do
        not discover retired bytes.
        """
        key = (engine_id, int(layer), int(offset))
        with self._lock:
            slot = self._slots.get(key)
            if slot is None or not slot.ready or slot.retired:
                return None
            expected = int(expected_generation or 0)
            if expected and int(slot.generation) != expected:
                return None
            slot.pins += 1
            self._touch_locked(key, slot)
            self._slots.move_to_end(key)
            return _SlotLease(self, key, slot)

    def mark_ready(
        self,
        engine_id: str,
        layer: int,
        offset: int,
        crc: int = 0,
        *,
        generation: int = 0,
    ) -> bool:
        """Flip a slot's ready flag to True and store its CRC. Called
        by the evict consumer after cuStreamSynchronize confirms the
        D2H landed and the CRC has been computed over the committed
        bytes. Returns True if the slot existed.

        Passing crc=0 is allowed for tests that don't care about
        integrity (restore will skip the CRC check when stored CRC
        is 0).
        """
        key = (engine_id, int(layer), int(offset))
        with self._lock:
            slot = self._slots.get(key)
            if slot is None:
                return False
            if slot.writing and slot.pins > 0:
                slot.writing = False
                slot.pins -= 1
            slot.ready = True
            slot.crc = int(crc) & 0xFFFFFFFF
            slot.generation = int(generation or 0)
            self._touch_locked(key, slot)
            self._slots.move_to_end(key)
        return True

    def release_slot(
        self,
        engine_id: str,
        layer: int,
        offset: int,
    ) -> bool:
        to_free: Optional[_Slot]
        with self._lock:
            key = (engine_id, int(layer), int(offset))
            present = key in self._slots
            to_free = self._retire_locked(key, free_when_unpinned=True)
        if to_free is not None:
            _free_host(to_free.host_ptr)
        return present

    def release_if_current(self, lease: _SlotLease) -> bool:
        """Release the leased slot only if no replacement owns its key."""
        to_free: Optional[_Slot] = None
        with self._lock:
            if self._slots.get(lease.key) is not lease.slot:
                return False
            to_free = self._retire_locked(lease.key, free_when_unpinned=True)
        if to_free is not None:
            _free_host(to_free.host_ptr)
        return True

    def release_engine(self, engine_id: str) -> int:
        """Free all slots belonging to `engine_id`. Returns count freed."""
        to_free: list[_Slot] = []
        with self._lock:
            keys = [k for k in self._slots if k[0] == engine_id]
            for key in keys:
                slot = self._retire_locked(key, free_when_unpinned=True)
                if slot is not None:
                    to_free.append(slot)
        for slot in to_free:
            _free_host(slot.host_ptr)
        return len(keys)

    def forget_engine(self, engine_id: str) -> int:
        """Drop the slot index entries for `engine_id` WITHOUT calling
        cudaFreeHost on the underlying buffers. Used only when we
        couldn't drain the daemon stream — freeing a buffer that's
        still the target of an in-flight cuMemcpyAsync is undefined
        behavior that can corrupt other CUDA state in the process.
        The buffers leak until process exit, which is the correct
        trade vs the use-after-free alternative.
        """
        with self._lock:
            keys = [k for k in self._slots if k[0] == engine_id]
            for key in keys:
                self._retire_locked(key, free_when_unpinned=False)
        return len(keys)

    def n_slots(self) -> int:
        with self._lock:
            return len(self._slots)

    def total_bytes(self) -> int:
        with self._lock:
            return int(self._bytes)

    def evict_lru_until_under(
        self,
        max_bytes: int,
        protected_keys: Optional[set[tuple[str, int, int]]] = None,
    ) -> list[tuple[str, int, int]]:
        """Drop oldest ready slots until total_bytes <= max_bytes.

        Kept as the compatibility API for older tests and callers.
        """
        return self.evict_until_under(
            max_bytes,
            protected_keys=protected_keys,
            policy="lru",
        )

    def evict_until_under(
        self,
        max_bytes: int,
        protected_keys: Optional[set[tuple[str, int, int]]] = None,
        *,
        policy: Optional[str] = None,
    ) -> list[tuple[str, int, int]]:
        """Drop ready slots under the configured CPU-RAM policy.

        Supported policies:
        - lru: evict the oldest ready slot first.
        - lfu: evict the least frequently accessed ready slot first.
        - tiny_lfu: aged-frequency eviction that approximates Pegaflow's
          TinyLFU preference for historically hot KV without adding a
          full admission/lease RPC to the request plane.

        Not-ready slots are never evicted here because a daemon stream
        may still be writing into them. Protected keys are skipped so
        a caller can avoid evicting slots from the spill/promote it
        just completed before it has registered metadata for them.
        Pinned slots are also skipped; release/eviction will retry on a
        later quota pass after active readers drop their leases.
        """
        if max_bytes < 0:
            max_bytes = 0
        protected = protected_keys or set()
        selected_policy = (
            self._eviction_policy
            if policy is None
            else self.normalize_eviction_policy(policy)
        )
        evicted: list[tuple[str, int, int]] = []
        to_free: list[_Slot] = []
        with self._lock:
            if self._bytes <= max_bytes:
                return evicted
            for key, _slot in self._eviction_order_locked(
                selected_policy,
                protected,
            ):
                if self._bytes <= max_bytes:
                    break
                popped = self._retire_locked(
                    key,
                    free_when_unpinned=True,
                )
                if popped is None:
                    continue
                evicted.append(key)
                to_free.append(popped)
        for slot in to_free:
            _free_host(slot.host_ptr)
        return evicted

    def snapshot_keys(self) -> list[tuple[str, int, int]]:
        """Return a copy of the current slot-key list. Used by the
        background scrub thread, which doesn't want to hold
        `self._lock` across CRC computation. Keys taken now may
        have been released or replaced by the time the scrubber
        reads them — that's fine; the scrubber re-checks via
        `pin()` before doing any work.
        """
        with self._lock:
            return list(self._slots.keys())


# ----- internal CUDA host alloc/free helpers -----


def _alloc_host(size: int) -> int:
    from cuda.bindings import runtime as rt

    err, ptr = rt.cudaHostAlloc(size, 0)
    if err != rt.cudaError_t.cudaSuccess:
        _, msg = rt.cudaGetErrorString(err)
        raise RuntimeError(
            f"cudaHostAlloc({size}) failed: {msg.decode() if msg else err}",
        )
    return int(ptr)


def _free_host(host_ptr: int) -> None:
    from cuda.bindings import runtime as rt

    err = rt.cudaFreeHost(host_ptr)[0]
    if err != rt.cudaError_t.cudaSuccess:
        _, msg = rt.cudaGetErrorString(err)
        logger.warning("cudaFreeHost failed: %s", msg.decode() if msg else err)
