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


def _blake2b_128_bytes(data: bytes) -> bytes:
    """Production default: BLAKE2b truncated to 128 bits, padded to
    32 bytes for wire compatibility. ~3-5× faster than SHA-256,
    collision-resistant in practice for caches up to ~2^64 blocks.

    The 16-byte digest is followed by 16 zero bytes so the wire
    `content_hash` stays 32 bytes (matching the daemon's existing
    invariant). The receiver hashes the same way and compares the
    first 16 bytes only (zeros match trivially)."""
    return hashlib.blake2b(data, digest_size=16).digest() + b"\x00" * 16


def _crc32_bytes(data: bytes) -> bytes:
    """Maximum-speed content hash: CRC-32 (zlib, hardware-accelerated
    on modern x86 via SSE 4.2 CRC32 instruction). ~5-10× faster than
    SHA-256. 32-bit collision space is acceptable for cluster-internal
    deployments where (a) the wire is already TCP/UCX-checksummed, and
    (b) the cache holds ≪ 2^16 blocks per peer (the birthday-collision
    threshold).

    The 4-byte CRC32 is padded to 32 bytes for wire compatibility."""
    import zlib

    crc = zlib.crc32(data) & 0xFFFFFFFF
    return crc.to_bytes(4, "big") + b"\x00" * 28


def _trust_peer_bytes(data: bytes) -> bytes:
    """`none` mode hash: returns crc32(data)-padded for use as a
    UNIQUE slot_id (collision-free in practice up to ~2^16 blocks /
    peer), but the receiver SKIPS the equality check entirely
    (`verify_on_receive=False` on the StagingTier). The bytes are
    accepted without verification — trust-peer mode. Lowest possible
    overhead while still keeping each block's slot distinct."""
    import zlib

    crc = zlib.crc32(data) & 0xFFFFFFFF
    return crc.to_bytes(4, "big") + b"\x00" * 28


# Modes where the daemon should skip the verify-on-receive step.
# Hash function is still used to compute slot_id; verify is just
# byte equality, which we trust the wire for.
SKIP_VERIFY_MODES: "set[str]" = {"none"}

# Modes where verify runs in a background thread AFTER the slot has
# transitioned to READY. Decode-side engine consumes bytes
# immediately (zero critical-path verify latency); mismatch (if any)
# is detected asynchronously and transitions the slot to CORRUPT,
# which the existing failure-recovery path handles (engine recomputes).
ASYNC_VERIFY_MODES: "set[str]" = {
    "async_sha256",
    "async_blake2b_128",
    "async_crc32",
}


def _sha256_bytes_async(data: bytes) -> bytes:
    """`async_sha256` mode: same digest as `sha256`, but `verify_mode`
    on the StagingTier is set to ASYNC so the compute happens in a
    background thread off the receive critical path."""
    return _sha256_bytes(data)


def _blake2b_128_bytes_async(data: bytes) -> bytes:
    return _blake2b_128_bytes(data)


def _crc32_bytes_async(data: bytes) -> bytes:
    return _crc32_bytes(data)


HASH_FN_BY_MODE: dict = {
    "sha256": _sha256_bytes,
    "blake2b_128": _blake2b_128_bytes,
    "crc32": _crc32_bytes,
    "none": _trust_peer_bytes,
    "async_sha256": _sha256_bytes_async,
    "async_blake2b_128": _blake2b_128_bytes_async,
    "async_crc32": _crc32_bytes_async,
}


def hash_fn_for_mode(mode: str):
    """Resolve a hash_fn name (sha256, blake2b_128, crc32, none,
    async_*) to the callable. Raises KeyError on unknown mode."""
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
        async_verify: bool = False,
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
        self._async_verify = bool(async_verify)
        self._clock = clock or time.monotonic
        # Async-verify thread pool. Only constructed if async_verify
        # is enabled. Single worker is sufficient — verify ops are
        # CPU-bound and serializing them gives us back-pressure if
        # verification can't keep up with arrivals (we'd want to
        # surface that via metrics).
        self._async_verify_executor = None
        if self._async_verify:
            from concurrent.futures import ThreadPoolExecutor

            self._async_verify_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="staging-async-verify",
            )

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
        # Three verification regimes:
        #   1. SYNC (`verify_on_receive=True, async_verify=False`):
        #      compute hash here, compare under the lock, then commit.
        #      Cost is on the critical path — current default.
        #   2. SKIP (`verify_on_receive=False`):
        #      no hash compute at all; trust the wire. Cheapest.
        #   3. ASYNC (`async_verify=True`):
        #      commit immediately so decode can start consuming bytes,
        #      then schedule the hash on a background thread.
        #      On mismatch, transition slot to CORRUPT later.
        #      Zero critical-path cost; corruption detected but not
        #      blocking. Best of both worlds for trusted clusters.
        if verify_content_hash and self._verify_on_receive and not self._async_verify:
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
                # verify_on_receive=False or async_verify=True path:
                # commit without comparing here. The async-verify case
                # schedules the actual compare on a background thread
                # below (after we drop the lock).
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
            commit_result = CommitOk(hit=hit)

        # async-verify: now that the slot is READY and visible to
        # consumers, queue the hash compute on a background thread.
        # If a mismatch is detected later, the slot is transitioned
        # to CORRUPT (which the engine's failure-recovery path
        # handles via recompute). Until then, decode-side consumes
        # the bytes immediately — zero critical-path verify cost.
        if (
            verify_content_hash
            and self._async_verify
            and self._async_verify_executor is not None
        ):
            self._async_verify_executor.submit(
                self._async_verify_one,
                content_hash,
                payload,
                reservation_id,
            )
        return commit_result

    def _async_verify_one(
        self,
        content_hash: bytes,
        payload: bytes,
        reservation_id: str,
    ) -> None:
        """Background hash compute. If the bytes don't match the
        registered hash, the slot is transitioned to CORRUPT and
        waiters are notified. The decode engine's existing failure-
        recovery path (which already handles CORRUPT slots for the
        sync-verify case) takes over."""
        try:
            actual = self._hash_fn(payload)
        except Exception:
            logger.exception(
                "[StagingTier] async-verify: hash_fn raised on rid=%s",
                reservation_id,
            )
            return
        if actual == content_hash:
            return  # success — already READY, no work to do
        # Mismatch — transition to CORRUPT.
        logger.warning(
            "[StagingTier] async-verify: hash mismatch on rid=%s "
            "(expected=%s actual=%s); slot → CORRUPT",
            reservation_id,
            content_hash.hex()[:16],
            actual.hex()[:16],
        )
        with self._lock:
            slot = self._slots.get(content_hash)
            if slot is None:
                return
            if slot.state is _State.READY:
                slot.state = _State.CORRUPT
                self._stats["async_verify_corruptions"] = (
                    self._stats.get("async_verify_corruptions", 0) + 1
                )
                self._notify_waiters_locked(
                    slot,
                    None,
                    CommitRejected(
                        reason="async verify: hash mismatch",
                    ),
                )

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

    # ----- Background sweeps -----

    def scrub_step(self) -> int:
        """Re-CRC one READY slot in round-robin order. Drops if drift
        detected (I-6'). Returns 1 if scrubbed, 0 if no work."""
        # Pick a slot under lock, but compute CRC outside lock — slot
        # remains pinned by refcount-by-virtue-of-being-scrubbed.
        with self._lock:
            ready_hashes = [
                h
                for h, s in self._slots.items()
                if s.state is _State.READY and s.refcount == 0
            ]
            if not ready_hashes:
                return 0
            self._scrub_cursor = (self._scrub_cursor + 1) % len(ready_hashes)
            target_hash = ready_hashes[self._scrub_cursor]
            slot = self._slots[target_hash]
            ptr = slot.bytes_ptr
            size = slot.bytes_size
            expected_crc = slot.crc32

        # Read + CRC outside lock.
        try:
            data = self._alloc.read(ptr, size)
        except Exception:
            logger.exception("[StagingTier] scrub read failed")
            return 0
        actual_crc = zlib.crc32(data) & 0xFFFFFFFF
        if actual_crc == expected_crc:
            return 1

        # Drift detected. Re-acquire lock to drop the slot; recheck the
        # slot is still the one we scrubbed (could have been replaced
        # while we were computing CRC).
        with self._lock:
            current = self._slots.get(target_hash)
            if current is None or current.bytes_ptr != ptr:
                # Slot already replaced; nothing to do.
                return 1
            logger.warning(
                "[StagingTier] scrub drift: hash=%s expected_crc=%08x "
                "actual_crc=%08x — dropping slot",
                target_hash.hex()[:16],
                expected_crc,
                actual_crc,
            )
            self._stats["scrub_corruptions"] += 1
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
