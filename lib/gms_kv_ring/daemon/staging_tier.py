# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Staging tier — content-hash-keyed buffer for cross-node KV transfers.

External (router-orchestrated) byte transfers land in this tier BEFORE
being consumed by the connector through the existing restore ring. The
engine never sees this tier directly; the connector polls it by content
hash at `get_num_new_matched_tokens` time and emits a restore record
with the `SOURCE_STAGING` flag.

See `docs/CROSS_NODE_DESIGN.md` for the architecture and full race
analysis. This file enforces:

  I-8 — At most one inbound transfer per `(daemon, content_hash)` pair.
        Atomic reservation in `reserve_or_wait` under a single lock.
        Concurrent commands for the same hash coalesce as waiters on
        the original reservation. Solves the
        "two-peer-simultaneously-deliver" race.

  I-9 — Receiver hash-verifies on commit by default. Bytes whose
        content hash does not match the advertised hash transition the
        slot to CORRUPT and notify waiters with failure. Trusted
        prefix-hash paths can explicitly skip the byte-digest check and
        use the hash as a routing key.

  I-3' (staging) — Each staging slot carries a monotonic per-hash
        generation. `begin_consume` requires the caller-supplied
        `expected_generation` to match; mismatch → None → the connector
        treats it the same as a missing slot and falls through to its
        existing drop-on-failure (I-5) recompute path.

  I-6' (staging) — `scrub_step` re-CRC's READY slots on a slow cadence;
        drift drops the slot and bumps a corruption counter.

State machine (one slot per content hash):

    EMPTY ──reserve──> RESERVED ──commit_ok──> READY
                          │                      │
                          │  commit_corrupt      │  LRU evict
                          │  or fail             │  or scrub drift
                          ▼                      ▼
                       CORRUPT ───────────────> EMPTY
                          │                      ▲
                          │   reclaim            │
                          └──────────────────────┘

Reservation TTL handles client crashes — RESERVED slots whose owner
hasn't called commit_or_reject within `reservation_stale_s` get
reclaimed by `evict_stale_reservations`.
"""

from __future__ import annotations

import enum
import hashlib
import logging
import threading
import time
import uuid
import zlib
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Optional, Union

logger = logging.getLogger(__name__)


# ---- Pluggable allocator -----------------------------------------------------


class StagingAllocator:
    """Pluggable allocator for staging-slot bytes.

    Production uses `_CudaPinnedAllocator` (cudaHostAlloc-backed) so
    cuFile / NIXL can DMA directly. Tests use `_BytearrayAllocator` so
    the state machine can be exercised without CUDA."""

    def alloc(self, size: int) -> int:
        raise NotImplementedError

    def free(self, ptr: int) -> None:
        raise NotImplementedError

    def write(self, ptr: int, data: bytes) -> None:
        raise NotImplementedError

    def read(self, ptr: int, size: int) -> bytes:
        raise NotImplementedError


class _BytearrayAllocator(StagingAllocator):
    """Backed by Python bytearrays. Test-only; the "pointer" returned is
    just a stable id into a dict that maps to the underlying bytes."""

    def __init__(self) -> None:
        self._buffers: dict[int, bytearray] = {}
        self._next_id = 1
        self._lock = threading.Lock()

    def alloc(self, size: int) -> int:
        with self._lock:
            ptr = self._next_id
            self._next_id += 1
            self._buffers[ptr] = bytearray(size)
        return ptr

    def free(self, ptr: int) -> None:
        with self._lock:
            self._buffers.pop(ptr, None)

    def write(self, ptr: int, data: bytes) -> None:
        with self._lock:
            buf = self._buffers[ptr]
            if len(data) != len(buf):
                raise ValueError(
                    f"write size {len(data)} != buffer size {len(buf)}",
                )
            buf[:] = data

    def read(self, ptr: int, size: int) -> bytes:
        with self._lock:
            return bytes(self._buffers[ptr][:size])


class _CudaPinnedAllocator(StagingAllocator):
    """Pinned host allocator. Imports `cuda.bindings` lazily so the
    module can be imported in CUDA-free environments (tests)."""

    def alloc(self, size: int) -> int:
        from cuda.bindings import runtime as rt

        err, ptr = rt.cudaHostAlloc(size, 0)
        if err != rt.cudaError_t.cudaSuccess:
            _, msg = rt.cudaGetErrorString(err)
            raise RuntimeError(
                f"cudaHostAlloc({size}) failed: " f"{msg.decode() if msg else err}",
            )
        return int(ptr)

    def free(self, ptr: int) -> None:
        from cuda.bindings import runtime as rt

        err = rt.cudaFreeHost(ptr)[0]
        if err != rt.cudaError_t.cudaSuccess:
            _, msg = rt.cudaGetErrorString(err)
            logger.warning(
                "cudaFreeHost failed: %s",
                msg.decode() if msg else err,
            )

    def write(self, ptr: int, data: bytes) -> None:
        import ctypes

        ctypes.memmove(ptr, data, len(data))

    def read(self, ptr: int, size: int) -> bytes:
        import ctypes

        buf = (ctypes.c_char * size).from_address(ptr)
        return bytes(buf)


# ---- Hash helpers ------------------------------------------------------------


def _sha256_bytes(data: bytes) -> bytes:
    """Cryptographic-strength content hash: SHA-256 → 32-byte digest.
    Slowest but strongest. Backward-compatible default."""
    return hashlib.sha256(data).digest()


def _blake2b_256_bytes(data: bytes) -> bytes:
    """Full 256-bit BLAKE2b content identity.

    Content addresses remain collision resistant independently of transport
    checksums. CRC32 remains a separate, fast corruption check.
    """
    return hashlib.blake2b(data, digest_size=32).digest()


HASH_FN_BY_MODE: dict = {
    "sha256": _sha256_bytes,
    "blake2b_256": _blake2b_256_bytes,
}


def hash_fn_for_mode(mode: str):
    """Resolve a supported synchronous hash mode to its callable."""
    if mode not in HASH_FN_BY_MODE:
        raise KeyError(
            f"unknown hash mode {mode!r}; expected one of " f"{sorted(HASH_FN_BY_MODE)}"
        )
    return HASH_FN_BY_MODE[mode]


# ---- Slot state --------------------------------------------------------------


class _State(enum.Enum):
    EMPTY = "empty"  # placeholder; not stored — absence = EMPTY
    RESERVED = "reserved"  # transfer in flight
    READY = "ready"  # bytes verified, available to consume
    CORRUPT = "corrupt"  # commit verification failed; awaiting cleanup


@dataclass
class _Waiter:
    """Coalesced caller blocked on the active reservation for some hash.
    Notified atomically when the reservation resolves (READY / fail).
    `hit` set on success, `failure` set on error; exactly one is set
    before `event.set()`."""

    event: threading.Event = field(default_factory=threading.Event)
    hit: Optional["StagingHit"] = None
    failure: Optional[str] = None


@dataclass
class _Slot:
    state: _State
    content_hash: bytes
    reservation_id: Optional[str] = None
    source_daemon: Optional[str] = None
    reserved_at: float = 0.0
    bytes_ptr: Optional[int] = None
    bytes_size: int = 0
    crc32: int = 0
    generation: int = 0  # bumped on each successful commit
    refcount: int = 0  # nonzero while a consume is in flight
    waiters: list[_Waiter] = field(default_factory=list)
    last_accessed: float = 0.0


# ---- Result types ------------------------------------------------------------


@dataclass(frozen=True)
class Reservation:
    """Caller owns the slot. Must call `commit_or_reject(reservation_id,
    payload)` or `fail_reservation(reservation_id, reason)` within
    `reservation_stale_s` to avoid the slot being reclaimed."""

    reservation_id: str
    content_hash: bytes


@dataclass(frozen=True)
class Waiter:
    """Another caller is fetching this hash. Block on `waiter.event` and
    read `waiter.hit` (success) or `waiter.failure` (error)."""

    waiter: _Waiter
    content_hash: bytes


@dataclass(frozen=True)
class AlreadyReady:
    """Hash is already populated and ready to consume immediately."""

    hit: "StagingHit"


@dataclass(frozen=True)
class Rejected:
    """Reservation refused — typically capacity-driven backpressure."""

    reason: str


ReserveResult = Union[Reservation, Waiter, AlreadyReady, Rejected]


@dataclass(frozen=True)
class CommitOk:
    hit: "StagingHit"


@dataclass(frozen=True)
class CommitCorrupt:
    """Hash verify failed; slot transitioned to CORRUPT. Waiters were
    notified with failure."""

    reason: str


@dataclass(frozen=True)
class CommitRejected:
    """Reservation no longer valid (stale-reclaimed, mismatched id)."""

    reason: str


CommitResult = Union[CommitOk, CommitCorrupt, CommitRejected]


@dataclass(frozen=True)
class StagingHit:
    """Information about a READY staging slot."""

    content_hash: bytes
    bytes_ptr: int
    bytes_size: int
    crc32: int
    generation: int


@dataclass(frozen=True)
class ConsumeHandle:
    """Opaque token returned by `begin_consume`. Must be passed to
    `end_consume` to release the slot's refcount."""

    content_hash: bytes
    generation: int


# ---- Staging tier ------------------------------------------------------------


class StagingTier:
    """Content-hash-keyed buffer tier. Owned exclusively by the GMS daemon.
    The engine has no direct binding; the connector consumes via the
    existing restore ring with a `SOURCE_STAGING` flag."""

    def __init__(
        self,
        *,
        capacity_bytes: int,
        reservation_stale_s: float = 30.0,
        allocator: Optional[StagingAllocator] = None,
        hash_fn: Optional[Callable[[bytes], bytes]] = None,
        clock: Optional[Callable[[], float]] = None,
        verify_on_receive: bool = True,
    ) -> None:
        if capacity_bytes <= 0:
            raise ValueError(
                f"capacity_bytes must be > 0, got {capacity_bytes}",
            )
        self._capacity = int(capacity_bytes)
        self._reservation_stale_s = float(reservation_stale_s)
        self._alloc = allocator if allocator is not None else _CudaPinnedAllocator()
        self._hash_fn = hash_fn or _sha256_bytes
        self._verify_on_receive = bool(verify_on_receive)
        self._clock = clock or time.monotonic

        # All slots, keyed by content hash. Absence = EMPTY state.
        self._slots: dict[bytes, _Slot] = {}
        # LRU order for READY slots only. Most-recently-accessed at end.
        self._lru: "OrderedDict[bytes, None]" = OrderedDict()
        # reservation_id → content_hash (for commit/fail lookup).
        self._by_reservation: dict[str, bytes] = {}
        # Bytes used by READY + CORRUPT slots. RESERVED slots have no
        # allocation yet; they reserve the slot key, not memory.
        self._bytes_used = 0
        # Round-robin cursor for scrub.
        self._scrub_cursor = 0

        self._lock = threading.Lock()

        self._stats = {
            "reservations_created": 0,
            "waiters_coalesced": 0,
            "commits_ok": 0,
            "commits_corrupt": 0,
            "reservations_failed": 0,
            "stale_reservations_reclaimed": 0,
            "evictions_lru": 0,
            "scrub_corruptions": 0,
            "rejected_capacity": 0,
        }

    # ----- Reservation API -----

    def reserve_or_wait_many(
        self,
        content_hashes: "list[bytes]",
        source_daemon: str,
    ) -> "list[ReserveResult]":
        """Vectorized reservation: takes the lock ONCE and processes N
        hashes inside. ~100× faster than N separate `reserve_or_wait`
        calls for large N because we amortize lock acquisition + dict
        bookkeeping cost. Returns one result per input hash, in order.

        Used by `fetch_remote` which always reserves N hashes at once
        — eliminates the per-block Python overhead that dominated the
        GMS-vs-Dynamo benchmark gap."""
        now = self._clock()
        results: list = []
        with self._lock:
            for content_hash in content_hashes:
                slot = self._slots.get(content_hash)
                if slot is None:
                    results.append(
                        self._create_reservation_locked(
                            content_hash,
                            source_daemon,
                            now,
                        )
                    )
                    continue
                if slot.state is _State.READY:
                    self._lru.move_to_end(content_hash)
                    slot.last_accessed = now
                    results.append(AlreadyReady(hit=_hit_from_slot(slot)))
                    continue
                if slot.state is _State.RESERVED:
                    if (now - slot.reserved_at) > self._reservation_stale_s:
                        self._reclaim_stale_reservation_locked(slot, now)
                        results.append(
                            self._create_reservation_locked(
                                content_hash,
                                source_daemon,
                                now,
                            )
                        )
                        continue
                    w = _Waiter()
                    slot.waiters.append(w)
                    self._stats["waiters_coalesced"] += 1
                    results.append(Waiter(waiter=w, content_hash=content_hash))
                    continue
                # CORRUPT
                self._drop_slot_locked(content_hash)
                results.append(
                    self._create_reservation_locked(
                        content_hash,
                        source_daemon,
                        now,
                    )
                )
        return results

    def reserve_or_wait(
        self,
        content_hash: bytes,
        source_daemon: str,
    ) -> ReserveResult:
        """Atomic per-hash reservation (I-8).

        - EMPTY (absent) → create RESERVED, return Reservation.
        - RESERVED (active owner) → add to waiters, return Waiter.
        - RESERVED (stale owner) → reclaim, create new RESERVED, return Reservation.
        - READY → return AlreadyReady.
        - CORRUPT → reclaim, create new RESERVED, return Reservation.
        """
        now = self._clock()
        with self._lock:
            slot = self._slots.get(content_hash)

            if slot is None:
                return self._create_reservation_locked(
                    content_hash,
                    source_daemon,
                    now,
                )

            if slot.state is _State.READY:
                # Mark accessed for LRU
                self._lru.move_to_end(content_hash)
                slot.last_accessed = now
                return AlreadyReady(hit=_hit_from_slot(slot))

            if slot.state is _State.RESERVED:
                if (now - slot.reserved_at) > self._reservation_stale_s:
                    # Owner is stuck/dead; reclaim.
                    self._reclaim_stale_reservation_locked(slot, now)
                    return self._create_reservation_locked(
                        content_hash,
                        source_daemon,
                        now,
                    )
                # Active owner; coalesce.
                w = _Waiter()
                slot.waiters.append(w)
                self._stats["waiters_coalesced"] += 1
                return Waiter(waiter=w, content_hash=content_hash)

            # CORRUPT — reclaim and retry. (Waiters were already
            # notified when state transitioned to CORRUPT.)
            self._drop_slot_locked(content_hash)
            return self._create_reservation_locked(
                content_hash,
                source_daemon,
                now,
            )

    def commit_or_reject(
        self,
        reservation_id: str,
        payload: bytes,
        *,
        verify_content_hash: bool = True,
    ) -> CommitResult:
        """Transition RESERVED to READY or CORRUPT.

        By default this verifies that ``hash_fn(payload)`` equals the
        reservation key (I-9). Some engine integrations use the key as a
        deterministic prefix hash rather than a byte hash; those trusted paths
        pass ``verify_content_hash=False`` and rely on transport integrity plus
        the source daemon's prefix-to-address bookkeeping.
        """
        # Hash compute happens OUTSIDE the lock — it can be slow for
        # large payloads (~MB) and we don't want to block reserve_or_wait
        # callers during verify. The state-machine transition step does
        # the final critical-section work under the lock.
        #
        # When `verify_on_receive=False` (mode="none"), we skip the
        # content hash entirely and trust the wire — the receiver's
        # check_remote_metadata + NIXL/UCX CRC layer already protect
        # against random corruption. `computed_hash` falls back to the
        # registered hash so the equality check at line 488 trivially
        # passes. `computed_crc` is always computed because it's cheap
        # (zlib.crc32 is ~5ns/byte hardware-accelerated) and used as
        # the storage-tier integrity check on later reads.
        # Never publish READY before verification completes: consumers
        # must not observe bytes that may later be declared corrupt.
        if verify_content_hash and self._verify_on_receive:
            computed_hash = self._hash_fn(payload)
        else:
            computed_hash = None
        computed_crc = zlib.crc32(payload) & 0xFFFFFFFF
        now = self._clock()

        with self._lock:
            content_hash = self._by_reservation.get(reservation_id)
            if content_hash is None:
                return CommitRejected(
                    reason="reservation_id unknown (reclaimed?)",
                )
            slot = self._slots.get(content_hash)
            if slot is None or slot.reservation_id != reservation_id:
                # Stale reservation; somebody else owns this hash now.
                self._by_reservation.pop(reservation_id, None)
                return CommitRejected(
                    reason="reservation no longer matches slot",
                )
            if slot.state is not _State.RESERVED:
                self._by_reservation.pop(reservation_id, None)
                return CommitRejected(
                    reason=f"slot state is {slot.state.value}",
                )
            if computed_hash is None:
                # Trusted routing-key path; the key is not a byte digest.
                computed_hash = content_hash

            if computed_hash != content_hash:
                # I-9 violation: bytes don't match advertised hash.
                # Drop the slot to CORRUPT, notify waiters with failure.
                slot.state = _State.CORRUPT
                self._stats["commits_corrupt"] += 1
                reason = (
                    f"hash mismatch (expected {content_hash.hex()[:16]}…, "
                    f"got {computed_hash.hex()[:16]}…)"
                )
                self._notify_waiters_locked(slot, None, reason)
                self._by_reservation.pop(reservation_id, None)
                # Eagerly drop CORRUPT slots so the next reserver sees EMPTY.
                self._drop_slot_locked(content_hash)
                logger.warning(
                    "[StagingTier] commit rejected: %s",
                    reason,
                )
                return CommitCorrupt(reason=reason)

            # Make room before allocating.
            size = len(payload)
            if not self._make_room_locked(size, exclude_hash=content_hash):
                # Cannot fit even after eviction; reject.
                slot.state = _State.EMPTY  # transient; will be removed
                self._drop_slot_locked(content_hash)
                self._stats["rejected_capacity"] += 1
                self._notify_waiters_locked(slot, None, "capacity")
                self._by_reservation.pop(reservation_id, None)
                return CommitRejected(reason="capacity exhausted")

            # Allocate + copy outside the critical bytes path but still
            # under the lock (allocator state must be consistent with
            # _bytes_used). For pinned allocator this is a fast syscall.
            ptr = self._alloc.alloc(size)
            self._alloc.write(ptr, payload)
            self._bytes_used += size

            slot.state = _State.READY
            slot.bytes_ptr = ptr
            slot.bytes_size = size
            slot.crc32 = computed_crc
            slot.generation += 1
            slot.last_accessed = now
            self._lru[content_hash] = None
            self._lru.move_to_end(content_hash)
            self._stats["commits_ok"] += 1
            hit = _hit_from_slot(slot)
            self._notify_waiters_locked(slot, hit, None)
            self._by_reservation.pop(reservation_id, None)
            return CommitOk(hit=hit)

    def fail_reservation(
        self,
        reservation_id: str,
        reason: str,
    ) -> None:
        """Release a reservation without delivering bytes; notify
        waiters of failure."""
        with self._lock:
            content_hash = self._by_reservation.pop(reservation_id, None)
            if content_hash is None:
                return
            slot = self._slots.get(content_hash)
            if slot is None or slot.reservation_id != reservation_id:
                return
            if slot.state is not _State.RESERVED:
                return
            self._stats["reservations_failed"] += 1
            self._notify_waiters_locked(slot, None, reason)
            self._drop_slot_locked(content_hash)

    # ----- Lookup API -----

    def scan(
        self,
        content_hashes: list[bytes],
    ) -> dict[bytes, StagingHit]:
        """Batch lookup. Returns only currently-READY slots."""
        out: dict[bytes, StagingHit] = {}
        now = self._clock()
        with self._lock:
            for h in content_hashes:
                slot = self._slots.get(h)
                if slot is None or slot.state is not _State.READY:
                    continue
                out[h] = _hit_from_slot(slot)
                # Don't move-to-end on scan — only on actual consume.
                # Scanning is read-only from LRU's perspective; a poll
                # that finds nothing shouldn't keep cold slots warm.
                slot.last_accessed = now
        return out

    def begin_consume(
        self,
        content_hash: bytes,
        expected_generation: int,
    ) -> Optional[ConsumeHandle]:
        """Pin a READY slot for restore. Returns None if not READY or
        generation mismatch (I-3'). Increments refcount; caller MUST
        call end_consume to release."""
        with self._lock:
            slot = self._slots.get(content_hash)
            if slot is None or slot.state is not _State.READY:
                return None
            if slot.generation != expected_generation:
                return None
            slot.refcount += 1
            slot.last_accessed = self._clock()
            self._lru.move_to_end(content_hash)
            return ConsumeHandle(
                content_hash=content_hash,
                generation=slot.generation,
            )

    def consume_pointer(
        self,
        handle: ConsumeHandle,
    ) -> Optional[tuple[int, int, int]]:
        """Return ``(ptr, size, crc32)`` for a pinned READY slot.

        The caller must have obtained ``handle`` from ``begin_consume``
        and must call ``end_consume`` after it has copied the bytes.
        ``None`` means the slot disappeared or its generation changed,
        so the caller must fail the restore and let the engine
        recompute.
        """
        with self._lock:
            slot = self._slots.get(handle.content_hash)
            if slot is None or slot.state is not _State.READY:
                return None
            if slot.generation != handle.generation:
                return None
            if slot.bytes_ptr is None or slot.bytes_size <= 0:
                return None
            return (int(slot.bytes_ptr), int(slot.bytes_size), int(slot.crc32))

    def end_consume(self, handle: ConsumeHandle) -> None:
        with self._lock:
            slot = self._slots.get(handle.content_hash)
            if slot is None:
                return
            if slot.refcount > 0:
                slot.refcount -= 1

            if slot.refcount == 0 and slot.state is _State.CORRUPT:
                self._drop_slot_locked(handle.content_hash)
    # ----- Background sweeps -----

    def scrub_step(self) -> int:
        """Re-CRC one READY slot while holding a consumer pin."""
        with self._lock:
            ready_hashes = [
                h
                for h, slot in self._slots.items()
                if slot.state is _State.READY and slot.refcount == 0
            ]
            if not ready_hashes:
                return 0
            self._scrub_cursor = (self._scrub_cursor + 1) % len(ready_hashes)
            target_hash = ready_hashes[self._scrub_cursor]
            slot = self._slots[target_hash]
            ptr = slot.bytes_ptr
            size = slot.bytes_size
            expected_crc = slot.crc32
            slot.refcount += 1

        try:
            actual_crc = zlib.crc32(self._alloc.read(ptr, size)) & 0xFFFFFFFF
        except Exception:
            logger.exception("[StagingTier] scrub read failed")
            with self._lock:
                current = self._slots.get(target_hash)
                if current is slot and current.refcount > 0:
                    current.refcount -= 1
            return 0

        with self._lock:
            current = self._slots.get(target_hash)
            if current is not slot:
                return 1
            if current.refcount > 0:
                current.refcount -= 1
            if actual_crc == expected_crc:
                return 1
            logger.warning(
                "[StagingTier] scrub drift: hash=%s expected_crc=%08x "
                "actual_crc=%08x — dropping slot",
                target_hash.hex()[:16],
                expected_crc,
                actual_crc,
            )
            self._stats["scrub_corruptions"] += 1
            current.state = _State.CORRUPT
            if current.refcount == 0:
                self._drop_slot_locked(target_hash)
        return 1

    def evict_stale_reservations(self) -> int:
        """Reclaim RESERVED slots whose owners have been silent past
        `reservation_stale_s`. Returns count reclaimed."""
        now = self._clock()
        reclaimed = 0
        with self._lock:
            stale_hashes = [
                h
                for h, s in self._slots.items()
                if s.state is _State.RESERVED
                and (now - s.reserved_at) > self._reservation_stale_s
            ]
            for h in stale_hashes:
                slot = self._slots[h]
                self._reclaim_stale_reservation_locked(slot, now)
                reclaimed += 1
        return reclaimed

    # ----- Lifecycle / inspection -----

    def shutdown(self) -> None:
        """Free all allocations and notify any waiters with failure."""
        with self._lock:
            for h, slot in list(self._slots.items()):
                if slot.waiters:
                    self._notify_waiters_locked(slot, None, "daemon shutdown")
                if slot.bytes_ptr is not None:
                    try:
                        self._alloc.free(slot.bytes_ptr)
                    except Exception:
                        logger.exception(
                            "[StagingTier] alloc.free during shutdown",
                        )
            self._slots.clear()
            self._lru.clear()
            self._by_reservation.clear()
            self._bytes_used = 0

    def stats(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)

    def n_slots(self) -> int:
        with self._lock:
            return len(self._slots)

    def bytes_used(self) -> int:
        with self._lock:
            return self._bytes_used

    # ----- Internal helpers (must be called with self._lock held) -----

    def _create_reservation_locked(
        self,
        content_hash: bytes,
        source_daemon: str,
        now: float,
    ) -> Reservation:
        reservation_id = uuid.uuid4().hex
        # Preserve generation across reservations so a freshly-reclaimed
        # slot's new generation differs from its previous incarnation
        # (defense in depth on top of I-3').
        prev_gen = (
            self._slots[content_hash].generation if content_hash in self._slots else 0
        )
        self._slots[content_hash] = _Slot(
            state=_State.RESERVED,
            content_hash=content_hash,
            reservation_id=reservation_id,
            source_daemon=source_daemon,
            reserved_at=now,
            last_accessed=now,
            generation=prev_gen,
        )
        self._by_reservation[reservation_id] = content_hash
        self._stats["reservations_created"] += 1
        return Reservation(
            reservation_id=reservation_id,
            content_hash=content_hash,
        )

    def _reclaim_stale_reservation_locked(
        self,
        slot: _Slot,
        now: float,
    ) -> None:
        logger.warning(
            "[StagingTier] stale reservation reclaimed: hash=%s " "owner=%s age=%.1fs",
            slot.content_hash.hex()[:16],
            slot.source_daemon,
            now - slot.reserved_at,
        )
        self._stats["stale_reservations_reclaimed"] += 1
        self._notify_waiters_locked(slot, None, "reservation stale")
        if slot.reservation_id is not None:
            self._by_reservation.pop(slot.reservation_id, None)
        self._drop_slot_locked(slot.content_hash)

    def _notify_waiters_locked(
        self,
        slot: _Slot,
        hit: Optional[StagingHit],
        failure: Optional[str],
    ) -> None:
        for w in slot.waiters:
            w.hit = hit
            w.failure = failure
            w.event.set()
        slot.waiters = []

    def _drop_slot_locked(self, content_hash: bytes) -> None:
        slot = self._slots.pop(content_hash, None)
        if slot is None:
            return
        self._lru.pop(content_hash, None)
        if slot.reservation_id is not None:
            self._by_reservation.pop(slot.reservation_id, None)
        if slot.bytes_ptr is not None:
            try:
                self._alloc.free(slot.bytes_ptr)
            except Exception:
                logger.exception("[StagingTier] alloc.free during drop")
            self._bytes_used -= slot.bytes_size

    def _make_room_locked(
        self,
        needed: int,
        exclude_hash: bytes,
    ) -> bool:
        """Try to evict LRU READY slots (with refcount==0) until
        `_bytes_used + needed <= _capacity`. Returns True on success,
        False if we couldn't fit even after evicting everything
        evictable."""
        if self._bytes_used + needed <= self._capacity:
            return True
        # Iterate LRU front (coldest) → back. Skip exclude_hash and
        # any slot with refcount > 0.
        for h in list(self._lru.keys()):
            if self._bytes_used + needed <= self._capacity:
                return True
            if h == exclude_hash:
                continue
            slot = self._slots.get(h)
            if slot is None or slot.state is not _State.READY:
                continue
            if slot.refcount > 0:
                continue
            self._stats["evictions_lru"] += 1
            self._drop_slot_locked(h)
        return self._bytes_used + needed <= self._capacity


def _hit_from_slot(slot: _Slot) -> StagingHit:
    assert slot.state is _State.READY
    assert slot.bytes_ptr is not None
    return StagingHit(
        content_hash=slot.content_hash,
        bytes_ptr=slot.bytes_ptr,
        bytes_size=slot.bytes_size,
        crc32=slot.crc32,
        generation=slot.generation,
    )


__all__ = [
    "StagingTier",
    "StagingAllocator",
    "StagingHit",
    "Reservation",
    "Waiter",
    "AlreadyReady",
    "Rejected",
    "CommitOk",
    "CommitCorrupt",
    "CommitRejected",
    "ConsumeHandle",
    "ReserveResult",
    "CommitResult",
]
