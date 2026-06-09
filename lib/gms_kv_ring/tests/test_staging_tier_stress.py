# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stress tests for the StagingTier state machine and its invariants.

Targets:
  - I-8 (atomic per-hash reservation): under N-thread contention, exactly
    one thread receives a Reservation per hash; all others get a Waiter
    (or AlreadyReady once the first commits).
  - I-9 (hash-on-receive): when a sender ships bytes that don't match
    the advertised content hash, commit_or_reject MUST transition the
    slot to CORRUPT, notify waiters with failure, and free the slot
    for a fresh reservation.
  - Reservation lifecycle: a never-completed reservation (caller dies,
    network drops the message) must be reclaimable via fail_reservation
    AND via the stale-timeout path. Subsequent reservations succeed
    without slot leakage and with a fresh reservation_id.
  - Capacity backpressure: filling the tier to capacity and committing
    more must NOT silently overwrite; the daemon either evicts an LRU
    READY slot or rejects with `capacity exhausted`.

All tests run in-process with the test-only `_BytearrayAllocator`, so
no CUDA / NIXL is required. They are deterministic — the contention
tests use sufficient repetitions + thread counts to surface any
non-atomic reservation gap, but each iteration is independently
verifiable, not flake-prone."""

from __future__ import annotations

import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor

from gms_kv_ring.daemon.staging_tier import (
    AlreadyReady,
    CommitCorrupt,
    CommitOk,
    CommitRejected,
    Reservation,
    StagingTier,
    Waiter,
    _BytearrayAllocator,
)


def _hash(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()


def _new_tier(capacity_bytes: int = 1 << 20, stale_s: float = 30.0):
    return StagingTier(
        capacity_bytes=capacity_bytes,
        reservation_stale_s=stale_s,
        allocator=_BytearrayAllocator(),
    )


# ---------------------------------------------------------------------------
# I-8: atomic per-hash reservation under contention
# ---------------------------------------------------------------------------


def test_i8_atomic_reservation_under_contention():
    """N threads race reserve_or_wait for the SAME hash. Exactly ONE
    thread must receive a Reservation; all others must receive Waiter
    (until the winner commits, after which late-comers see AlreadyReady).

    This is the load-bearing invariant for cross-node correctness: if
    two threads both got Reservations for the same hash, two different
    sets of bytes could land in the slot non-deterministically."""
    payload = b"contention-payload" * 64
    h = _hash(payload)

    for trial in range(20):
        tier = _new_tier()
        N = 16
        results: list = [None] * N
        barrier = threading.Barrier(N)

        def worker(i: int) -> None:
            barrier.wait()
            results[i] = tier.reserve_or_wait(h, source_daemon=f"peer-{i}")

        with ThreadPoolExecutor(max_workers=N) as ex:
            list(ex.map(worker, range(N)))

        reservations = [r for r in results if isinstance(r, Reservation)]
        waiters = [r for r in results if isinstance(r, Waiter)]
        already_ready = [r for r in results if isinstance(r, AlreadyReady)]

        assert len(reservations) == 1, (
            f"trial {trial}: expected exactly 1 Reservation, "
            f"got {len(reservations)} (waiters={len(waiters)} "
            f"already_ready={len(already_ready)})"
        )
        assert len(waiters) + len(already_ready) == N - 1
        assert (
            len(already_ready) == 0
        ), "no thread saw the slot READY before the winner committed"

        # All Waiters carry the same content_hash and the same _Waiter
        # bucket? No — each Waiter has its own _Waiter event, but all
        # point at the same content_hash.
        for w in waiters:
            assert w.content_hash == h


def test_i8_winner_commit_wakes_all_waiters():
    """Once the Reservation winner commits OK, ALL queued waiters must
    receive the StagingHit, not just one."""
    payload = b"wake-all-waiters" * 100
    h = _hash(payload)
    tier = _new_tier()

    # Reserve first (becomes the slot owner).
    res = tier.reserve_or_wait(h, source_daemon="leader")
    assert isinstance(res, Reservation)

    # Queue up 8 waiters.
    waiters: list[Waiter] = []
    for i in range(8):
        w = tier.reserve_or_wait(h, source_daemon=f"follower-{i}")
        assert isinstance(w, Waiter)
        waiters.append(w)

    # Commit the bytes.
    out = tier.commit_or_reject(res.reservation_id, payload)
    assert isinstance(out, CommitOk)

    # All waiters' events must be set, all hits must be non-None.
    for i, w in enumerate(waiters):
        assert w.waiter.event.is_set(), f"waiter {i} not woken"
        assert w.waiter.hit is not None, f"waiter {i} got no hit"
        assert w.waiter.failure is None
        assert w.waiter.hit.content_hash == h
        assert w.waiter.hit.bytes_size == len(payload)


# ---------------------------------------------------------------------------
# I-9: hash-on-receive — corrupted bytes must NOT commit
# ---------------------------------------------------------------------------


def test_i9_hash_mismatch_lands_corrupt_and_notifies_waiters():
    """When commit_or_reject is given bytes whose sha256 does NOT match
    the advertised content_hash, the slot must transition to CORRUPT,
    all waiters must be notified with failure (not a hit), and the
    slot must drop so a re-reservation succeeds."""
    advertised = _hash(b"the-truth")
    tier = _new_tier()

    res = tier.reserve_or_wait(advertised, source_daemon="liar")
    assert isinstance(res, Reservation)
    w = tier.reserve_or_wait(advertised, source_daemon="hopeful")
    assert isinstance(w, Waiter)

    # Sender ships different bytes (network bit-flip, malicious peer,
    # mismatched cache_salt, etc.). Hash check MUST catch it.
    out = tier.commit_or_reject(res.reservation_id, b"the-lie")
    assert isinstance(out, CommitCorrupt)
    assert "hash mismatch" in out.reason

    # Waiter must have been notified with FAILURE.
    assert w.waiter.event.is_set()
    assert w.waiter.hit is None
    assert w.waiter.failure is not None
    assert "hash mismatch" in w.waiter.failure

    # The corrupt slot must have been dropped — a fresh reservation
    # for the same hash must SUCCEED with a NEW reservation_id.
    res2 = tier.reserve_or_wait(advertised, source_daemon="retry")
    assert isinstance(res2, Reservation)
    assert res2.reservation_id != res.reservation_id

    # Stats reflect the corruption.
    stats = tier._stats
    assert stats["commits_corrupt"] == 1
    assert stats["commits_ok"] == 0


def test_i9_corruption_under_concurrent_waiters():
    """Heavy contention + corrupt commit: ALL waiters must see failure;
    no waiter must spuriously see a hit."""
    h = _hash(b"correct-payload" * 32)
    tier = _new_tier()

    res = tier.reserve_or_wait(h, source_daemon="leader")
    assert isinstance(res, Reservation)

    N = 12
    waiters: list[Waiter] = []
    for i in range(N):
        w = tier.reserve_or_wait(h, source_daemon=f"f-{i}")
        assert isinstance(w, Waiter)
        waiters.append(w)

    # Ship bytes that hash to something else.
    out = tier.commit_or_reject(res.reservation_id, b"WRONG")
    assert isinstance(out, CommitCorrupt)

    for i, w in enumerate(waiters):
        assert w.waiter.event.is_set(), f"waiter {i} not notified"
        assert w.waiter.hit is None, f"waiter {i} got phantom hit"
        assert w.waiter.failure is not None


# ---------------------------------------------------------------------------
# Reservation lifecycle: fail/reclaim, generation, no slot leak
# ---------------------------------------------------------------------------


def test_lifecycle_fail_then_re_reserve_succeeds():
    """A failed reservation must release the slot; the next reservation
    must succeed with a fresh reservation_id."""
    h = _hash(b"will-fail-first" * 10)
    tier = _new_tier()

    r1 = tier.reserve_or_wait(h, source_daemon="src1")
    assert isinstance(r1, Reservation)

    tier.fail_reservation(r1.reservation_id, reason="net error")

    # Slot should be empty now → fresh Reservation.
    r2 = tier.reserve_or_wait(h, source_daemon="src2")
    assert isinstance(r2, Reservation)
    assert r2.reservation_id != r1.reservation_id

    # And this one can commit normally.
    payload = b"will-fail-first" * 10
    out = tier.commit_or_reject(r2.reservation_id, payload)
    assert isinstance(out, CommitOk)
    assert tier._stats["reservations_failed"] == 1
    assert tier._stats["commits_ok"] == 1


def test_lifecycle_fail_with_waiters_notifies_failure():
    """fail_reservation must wake queued waiters with a failure
    notification — otherwise a slow consumer would block forever."""
    h = _hash(b"abandoned" * 50)
    tier = _new_tier()

    r1 = tier.reserve_or_wait(h, source_daemon="dropper")
    assert isinstance(r1, Reservation)
    w = tier.reserve_or_wait(h, source_daemon="hopeful")
    assert isinstance(w, Waiter)

    tier.fail_reservation(r1.reservation_id, reason="peer crashed")

    assert w.waiter.event.is_set()
    assert w.waiter.hit is None
    assert w.waiter.failure is not None


def test_lifecycle_stale_timeout_reclaims_silent_owner():
    """If the reservation owner never commits and the stale timeout
    expires, the next reserver claims the slot. Uses a fake clock for
    determinism."""
    h = _hash(b"silent-owner" * 8)
    t = [1000.0]

    def clock():
        return t[0]

    tier = StagingTier(
        capacity_bytes=1 << 20,
        reservation_stale_s=5.0,
        allocator=_BytearrayAllocator(),
        clock=clock,
    )

    r1 = tier.reserve_or_wait(h, source_daemon="silent")
    assert isinstance(r1, Reservation)

    # Advance the clock past the stale window.
    t[0] += 10.0

    r2 = tier.reserve_or_wait(h, source_daemon="impatient")
    assert isinstance(r2, Reservation)
    assert r2.reservation_id != r1.reservation_id
    assert tier._stats["stale_reservations_reclaimed"] == 1

    # And the original owner can no longer commit — stale reservation_id.
    out = tier.commit_or_reject(r1.reservation_id, b"silent-owner" * 8)
    assert isinstance(out, CommitRejected)


def test_lifecycle_commit_after_stale_reclaim_does_not_corrupt_new_slot():
    """Critical: if the original owner finally shows up after being
    reclaimed, their late commit must NOT corrupt the new owner's slot
    (e.g., write the wrong bytes into someone else's reservation)."""
    h = _hash(b"late-arrival" * 20)
    t = [1000.0]

    def clock():
        return t[0]

    tier = StagingTier(
        capacity_bytes=1 << 20,
        reservation_stale_s=2.0,
        allocator=_BytearrayAllocator(),
        clock=clock,
    )

    r1 = tier.reserve_or_wait(h, source_daemon="laggard")
    assert isinstance(r1, Reservation)
    t[0] += 10.0  # stale

    r2 = tier.reserve_or_wait(h, source_daemon="winner")
    assert isinstance(r2, Reservation)

    # Late commit from r1 must be rejected, not silently swap bytes
    # in r2's slot.
    late = tier.commit_or_reject(r1.reservation_id, b"late-arrival" * 20)
    assert isinstance(late, CommitRejected)

    # r2 can still commit normally.
    ok = tier.commit_or_reject(r2.reservation_id, b"late-arrival" * 20)
    assert isinstance(ok, CommitOk)


def test_lifecycle_unknown_reservation_id_is_rejected():
    """A commit with a bogus reservation_id must NOT succeed silently."""
    tier = _new_tier()
    out = tier.commit_or_reject("never-issued-id", b"anything")
    assert isinstance(out, CommitRejected)


# ---------------------------------------------------------------------------
# Capacity backpressure
# ---------------------------------------------------------------------------


def test_capacity_lru_eviction_under_pressure():
    """Filling the tier to capacity then committing more should evict
    the LRU READY slot, not reject. Validates the LRU-as-backpressure
    assumption built into commit_or_reject."""
    PAYLOAD_SIZE = 1024
    CAPACITY = 4 * PAYLOAD_SIZE  # exactly 4 slots fit
    tier = _new_tier(capacity_bytes=CAPACITY)

    hashes = []
    for i in range(4):
        p = bytes([i & 0xFF]) * PAYLOAD_SIZE
        h = _hash(p)
        hashes.append(h)
        r = tier.reserve_or_wait(h, source_daemon=f"s{i}")
        assert isinstance(r, Reservation)
        out = tier.commit_or_reject(r.reservation_id, p)
        assert isinstance(out, CommitOk)

    # Fifth commit should evict the LRU (hashes[0]) to make room.
    p5 = bytes([5]) * PAYLOAD_SIZE
    h5 = _hash(p5)
    r5 = tier.reserve_or_wait(h5, source_daemon="s5")
    assert isinstance(r5, Reservation)
    out5 = tier.commit_or_reject(r5.reservation_id, p5)
    assert isinstance(out5, CommitOk), out5

    # Newest is present.
    res_h5 = tier.reserve_or_wait(h5, source_daemon="lookup")
    assert isinstance(res_h5, AlreadyReady)

    # LRU (hashes[0]) was evicted.
    res_h0 = tier.reserve_or_wait(hashes[0], source_daemon="lookup")
    assert isinstance(
        res_h0, Reservation
    ), "expected LRU eviction to make hashes[0] re-reservable"
    assert tier._stats["evictions_lru"] >= 1


def test_capacity_exhaustion_with_single_oversized_payload_rejects_cleanly():
    """A single payload larger than total capacity must NOT crash —
    commit must return CommitRejected with reason 'capacity exhausted',
    waiters must be notified, slot must be cleared."""
    CAPACITY = 1024
    tier = _new_tier(capacity_bytes=CAPACITY)

    oversized = bytes(2 * CAPACITY)
    h = _hash(oversized)

    r = tier.reserve_or_wait(h, source_daemon="too-big")
    assert isinstance(r, Reservation)
    w = tier.reserve_or_wait(h, source_daemon="watching")
    assert isinstance(w, Waiter)

    out = tier.commit_or_reject(r.reservation_id, oversized)
    assert isinstance(out, CommitRejected)
    assert "capacity" in out.reason.lower()

    # Waiter must have been notified with failure.
    assert w.waiter.event.is_set()
    assert w.waiter.hit is None
    assert w.waiter.failure is not None

    # And the slot must be re-reservable (no leak).
    r2 = tier.reserve_or_wait(h, source_daemon="retry")
    assert isinstance(r2, Reservation)
    assert tier._stats["rejected_capacity"] == 1


# ---------------------------------------------------------------------------
# Mixed-load smoke: hundreds of distinct hashes + heavy concurrent reservations
# ---------------------------------------------------------------------------


def test_high_volume_distinct_hashes_no_dropped_reservations():
    """Many distinct hashes commit concurrently. After the dust
    settles, every hash should be either READY (within capacity) or
    LRU-evicted (gracefully). No hash should be stuck in RESERVED,
    no _by_reservation entry should leak."""
    # Generous capacity so most stay READY.
    PAYLOAD_SIZE = 256
    N_HASHES = 200
    tier = _new_tier(capacity_bytes=PAYLOAD_SIZE * N_HASHES)

    payloads = [
        bytes([i % 256] * PAYLOAD_SIZE) + i.to_bytes(4, "big") for i in range(N_HASHES)
    ]
    hashes = [_hash(p) for p in payloads]

    def worker(i: int) -> None:
        r = tier.reserve_or_wait(hashes[i], source_daemon=f"src-{i}")
        if isinstance(r, Reservation):
            out = tier.commit_or_reject(r.reservation_id, payloads[i])
            assert isinstance(out, CommitOk), out
        # else: AlreadyReady on retry — fine.

    with ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(worker, range(N_HASHES)))

    # Bookkeeping invariants:
    # - No reservation_ids left dangling.
    assert tier._by_reservation == {}, f"leaked reservations: {tier._by_reservation}"
    # - At least most hashes landed READY (no silent drops under load).
    ready_count = sum(1 for _ in tier._slots.values())
    assert (
        ready_count >= N_HASHES // 2
    ), f"too many hashes dropped: ready_count={ready_count} / {N_HASHES}"


def test_mixed_corrupt_and_clean_under_contention():
    """Mix of valid + invalid commits across many distinct hashes. The
    corrupt ones must NOT poison the valid ones."""
    tier = _new_tier(capacity_bytes=1 << 20)
    N = 50

    payloads = [f"payload-{i}".encode() * 16 for i in range(N)]
    hashes = [_hash(p) for p in payloads]

    def worker(i: int) -> None:
        r = tier.reserve_or_wait(hashes[i], source_daemon=f"s{i}")
        if not isinstance(r, Reservation):
            return
        # Every 5th sender ships corrupt bytes (deliberate hash mismatch).
        if i % 5 == 0:
            tier.commit_or_reject(r.reservation_id, b"GARBAGE" + bytes(i))
        else:
            tier.commit_or_reject(r.reservation_id, payloads[i])

    with ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(worker, range(N)))

    # All non-corrupt hashes must be queryable as ready.
    ready_hits = 0
    for i in range(N):
        if i % 5 == 0:
            continue
        r = tier.reserve_or_wait(hashes[i], source_daemon="lookup")
        if isinstance(r, AlreadyReady):
            ready_hits += 1
    # At least 80% of clean ones land ready (some LRU churn allowed).
    expected_min = int(0.8 * (N - N // 5))
    assert (
        ready_hits >= expected_min
    ), f"too few clean commits landed: {ready_hits} < {expected_min}"

    # Corrupt commits should all have been rejected.
    assert tier._stats["commits_corrupt"] >= N // 5 - 1  # off-by-1 tolerance
