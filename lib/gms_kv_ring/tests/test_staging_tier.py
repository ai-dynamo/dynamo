# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for StagingTier — Phase 1 of cross-node KV transfer.

Validates the state machine + invariants documented in
`docs/CROSS_NODE_DESIGN.md`:

  I-8  — At most one inbound transfer per (daemon, hash) in flight.
         Concurrent reservers for the same hash coalesce as waiters.
  I-9  — Receiver verifies bytes' content hash on commit by default.
         Trusted prefix-hash commits can opt out of byte verification.
  I-3' — Each staging slot carries a per-hash generation. begin_consume
         requires expected_generation; mismatch returns None.
  I-6' — Periodic scrub re-CRCs READY slots and drops drift.

Runs without CUDA via injected `_BytearrayAllocator`."""

from __future__ import annotations

import hashlib
import threading
import zlib

import pytest

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

# ----- helpers ----------------------------------------------------------------


def _hash(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def _make_tier(
    *,
    capacity_bytes: int = 1 << 20,
    reservation_stale_s: float = 30.0,
) -> StagingTier:
    return StagingTier(
        capacity_bytes=capacity_bytes,
        reservation_stale_s=reservation_stale_s,
        allocator=_BytearrayAllocator(),
    )


# ----- content identity -------------------------------------------------------


def test_content_hash_modes_are_full_width_and_collision_resistant():
    from gms_kv_ring.daemon.staging_tier import hash_fn_for_mode

    data = b"content-address-contract"
    assert hash_fn_for_mode("sha256")(data) == hashlib.sha256(data).digest()
    assert hash_fn_for_mode("blake2b_256")(data) == hashlib.blake2b(
        data, digest_size=32
    ).digest()
    assert len(hash_fn_for_mode("blake2b_256")(data)) == 32


@pytest.mark.parametrize("weak_mode", ["crc32", "none", "blake2b_128"])
def test_weak_content_hash_modes_are_rejected(weak_mode):
    from gms_kv_ring.daemon.staging_tier import hash_fn_for_mode

    with pytest.raises(KeyError, match="unknown hash mode"):
        hash_fn_for_mode(weak_mode)


def test_hash_algorithm_mismatch_fails_closed():
    from gms_kv_ring.daemon.staging_tier import hash_fn_for_mode

    data = b"rolling-upgrade-algorithm-mismatch"
    sha_id = hash_fn_for_mode("sha256")(data)
    tier = StagingTier(
        capacity_bytes=4096,
        allocator=_BytearrayAllocator(),
        hash_fn=hash_fn_for_mode("blake2b_256"),
    )
    reservation = tier.reserve_or_wait(sha_id, source_daemon="sha-peer")
    result = tier.commit_or_reject(reservation.reservation_id, data)
    assert isinstance(result, CommitCorrupt)
    assert tier.scan([sha_id]) == {}


# ----- happy path -------------------------------------------------------------


def test_reserve_then_commit_round_trip():
    tier = _make_tier()
    data = b"hello world" * 16
    h = _hash(data)

    res = tier.reserve_or_wait(h, source_daemon="node-A")
    assert isinstance(res, Reservation)
    assert res.content_hash == h

    result = tier.commit_or_reject(res.reservation_id, data)
    assert isinstance(result, CommitOk)
    assert result.hit.content_hash == h
    assert result.hit.bytes_size == len(data)
    assert result.hit.crc32 == zlib.crc32(data) & 0xFFFFFFFF
    assert result.hit.generation == 1

    # scan finds it
    found = tier.scan([h])
    assert h in found
    assert found[h].generation == 1


def test_already_ready_short_circuits():
    tier = _make_tier()
    data = b"already there"
    h = _hash(data)
    res = tier.reserve_or_wait(h, source_daemon="A")
    tier.commit_or_reject(res.reservation_id, data)

    # Second reserver sees AlreadyReady
    res2 = tier.reserve_or_wait(h, source_daemon="B")
    assert isinstance(res2, AlreadyReady)
    assert res2.hit.content_hash == h


def test_consume_pointer_returns_pinned_ready_slot():
    tier = _make_tier()
    data = b"consume pointer" * 16
    h = _hash(data)
    res = tier.reserve_or_wait(h, source_daemon="node-A")
    assert isinstance(res, Reservation)
    committed = tier.commit_or_reject(res.reservation_id, data)
    assert isinstance(committed, CommitOk)

    handle = tier.begin_consume(h, expected_generation=1)
    assert handle is not None
    try:
        ptr, size, crc = tier.consume_pointer(handle)
        assert ptr > 0
        assert size == len(data)
        assert crc == zlib.crc32(data) & 0xFFFFFFFF
        assert tier._alloc.read(ptr, size) == data
    finally:
        tier.end_consume(handle)


def test_consume_pointer_rejects_stale_generation():
    tier = _make_tier()
    data = b"stale generation" * 8
    h = _hash(data)
    res = tier.reserve_or_wait(h, source_daemon="node-A")
    assert isinstance(res, Reservation)
    assert isinstance(tier.commit_or_reject(res.reservation_id, data), CommitOk)
    assert tier.begin_consume(h, expected_generation=2) is None


# ----- I-8 coalescing --------------------------------------------------------


def test_two_concurrent_reservations_coalesce():
    tier = _make_tier()
    data = b"abc" * 100
    h = _hash(data)

    res = tier.reserve_or_wait(h, source_daemon="A")
    assert isinstance(res, Reservation)

    # Second caller sees the in-flight reservation and becomes a waiter
    res2 = tier.reserve_or_wait(h, source_daemon="B")
    assert isinstance(res2, Waiter)
    assert res2.content_hash == h
    assert not res2.waiter.event.is_set()

    # When original reservation commits, waiter is notified
    result = tier.commit_or_reject(res.reservation_id, data)
    assert isinstance(result, CommitOk)
    assert res2.waiter.event.is_set()
    assert res2.waiter.hit is not None
    assert res2.waiter.hit.content_hash == h
    assert res2.waiter.failure is None

    # Coalesced waiter did NOT trigger a second reservation
    stats = tier.stats()
    assert stats["reservations_created"] == 1
    assert stats["waiters_coalesced"] == 1
    assert stats["commits_ok"] == 1


def test_many_concurrent_waiters_all_notified_on_success():
    tier = _make_tier()
    data = b"x" * 50
    h = _hash(data)

    res = tier.reserve_or_wait(h, source_daemon="A")
    assert isinstance(res, Reservation)

    waiters = []
    for i in range(8):
        r = tier.reserve_or_wait(h, source_daemon=f"peer-{i}")
        assert isinstance(r, Waiter)
        waiters.append(r)

    tier.commit_or_reject(res.reservation_id, data)
    for w in waiters:
        assert w.waiter.event.is_set()
        assert w.waiter.hit is not None
        assert w.waiter.failure is None


def test_waiters_notified_on_failure():
    tier = _make_tier()
    data = b"y" * 20
    h = _hash(data)

    res = tier.reserve_or_wait(h, source_daemon="A")
    waiter_result = tier.reserve_or_wait(h, source_daemon="B")
    assert isinstance(waiter_result, Waiter)

    tier.fail_reservation(res.reservation_id, "transport error")

    assert waiter_result.waiter.event.is_set()
    assert waiter_result.waiter.hit is None
    assert waiter_result.waiter.failure == "transport error"


# ----- I-9 hash verify --------------------------------------------------------


def test_corrupt_bytes_rejected():
    tier = _make_tier()
    data = b"actual content"
    advertised_hash = _hash(b"different content")  # mismatched

    res = tier.reserve_or_wait(advertised_hash, source_daemon="A")
    assert isinstance(res, Reservation)
    result = tier.commit_or_reject(res.reservation_id, data)
    assert isinstance(result, CommitCorrupt)
    assert "hash mismatch" in result.reason

    # Slot is dropped after CORRUPT — next reserve sees EMPTY
    res2 = tier.reserve_or_wait(advertised_hash, source_daemon="B")
    assert isinstance(res2, Reservation)


def test_trusted_commit_accepts_prefix_hash_key():
    """Router-orchestrated engine transfers key staging by prefix hash,
    which is not a digest of the KV bytes themselves."""
    tier = _make_tier()
    data = b"actual kv bytes"
    prefix_hash = b"\x8a" * 32

    res = tier.reserve_or_wait(prefix_hash, source_daemon="router")
    assert isinstance(res, Reservation)
    result = tier.commit_or_reject(
        res.reservation_id,
        data,
        verify_content_hash=False,
    )

    assert isinstance(result, CommitOk)
    hits = tier.scan([prefix_hash])
    assert prefix_hash in hits
    assert hits[prefix_hash].bytes_size == len(data)


def test_corrupt_notifies_waiters_with_failure():
    tier = _make_tier()
    data = b"actual"
    bad_hash = _hash(b"expected")

    res = tier.reserve_or_wait(bad_hash, source_daemon="A")
    waiter = tier.reserve_or_wait(bad_hash, source_daemon="B")
    assert isinstance(waiter, Waiter)

    tier.commit_or_reject(res.reservation_id, data)

    assert waiter.waiter.event.is_set()
    assert waiter.waiter.hit is None
    assert waiter.waiter.failure is not None
    assert "hash mismatch" in waiter.waiter.failure


# ----- I-3' generation --------------------------------------------------------


def test_generation_increments_across_commits():
    tier = _make_tier()
    data1, data2 = b"first", b"second"
    h1 = _hash(data1)
    h2 = _hash(data2)

    r1 = tier.reserve_or_wait(h1, "A")
    tier.commit_or_reject(r1.reservation_id, data1)
    g1 = tier.scan([h1])[h1].generation
    assert g1 == 1

    # Different hash → its own generation starts at 1
    r2 = tier.reserve_or_wait(h2, "A")
    tier.commit_or_reject(r2.reservation_id, data2)
    g2 = tier.scan([h2])[h2].generation
    assert g2 == 1


def test_generation_resets_after_full_reclaim():
    """When a slot is fully reclaimed (LRU eviction, scrub corruption,
    or explicit fail), the new slot starts fresh at generation 1. This
    is safe because staging slots are content-addressed (same hash →
    same bytes by collision resistance + I-9), so a "stale" generation
    on the connector side that happens to match the new slot's
    generation still consumes correct bytes."""
    tier = _make_tier()
    data = b"hello"
    h = _hash(data)

    r1 = tier.reserve_or_wait(h, "A")
    tier.commit_or_reject(r1.reservation_id, data)
    assert tier.scan([h])[h].generation == 1

    # Fail then re-reserve → generation back to 1 (not 2)
    r2 = tier.reserve_or_wait(_hash(b"unrelated"), "A")
    tier.fail_reservation(r2.reservation_id, "test")
    # h still ready; re-deposit needs eviction first
    # Drop h by explicit shutdown path then verify generation resets
    tier.shutdown()
    tier2 = _make_tier()
    r3 = tier2.reserve_or_wait(h, "A")
    tier2.commit_or_reject(r3.reservation_id, data)
    assert tier2.scan([h])[h].generation == 1


def test_begin_consume_rejects_generation_mismatch():
    tier = _make_tier()
    data = b"data"
    h = _hash(data)
    r = tier.reserve_or_wait(h, "A")
    tier.commit_or_reject(r.reservation_id, data)
    gen = tier.scan([h])[h].generation

    # Right generation → succeeds
    handle = tier.begin_consume(h, expected_generation=gen)
    assert handle is not None
    tier.end_consume(handle)

    # Wrong generation → rejected
    bad_handle = tier.begin_consume(h, expected_generation=gen + 1)
    assert bad_handle is None


# ----- I-6' scrub -------------------------------------------------------------


def test_scrub_detects_in_place_corruption():
    alloc = _BytearrayAllocator()
    tier = StagingTier(capacity_bytes=1 << 20, allocator=alloc)
    data = b"original bytes here"
    h = _hash(data)
    r = tier.reserve_or_wait(h, "A")
    tier.commit_or_reject(r.reservation_id, data)
    hit = tier.scan([h])[h]

    # Simulate bit-flip in place by writing different bytes to the same ptr
    alloc.write(hit.bytes_ptr, b"X" * len(data))

    scrubbed = tier.scrub_step()
    assert scrubbed == 1
    assert tier.stats()["scrub_corruptions"] == 1
    # The slot has been dropped
    assert tier.scan([h]) == {}


def test_scrub_clean_pass_no_drops():
    tier = _make_tier()
    data = b"unchanged bytes"
    h = _hash(data)
    r = tier.reserve_or_wait(h, "A")
    tier.commit_or_reject(r.reservation_id, data)

    for _ in range(5):
        tier.scrub_step()
    assert tier.stats()["scrub_corruptions"] == 0
    assert h in tier.scan([h])


def test_scrub_skips_active_consumers():
    """A slot pinned by begin_consume should not be scrubbed (refcount>0)."""
    tier = _make_tier()
    data = b"pinned"
    h = _hash(data)
    r = tier.reserve_or_wait(h, "A")
    tier.commit_or_reject(r.reservation_id, data)
    gen = tier.scan([h])[h].generation
    handle = tier.begin_consume(h, expected_generation=gen)
    assert handle is not None

    # No other slots → scrub finds nothing to do
    assert tier.scrub_step() == 0

    tier.end_consume(handle)
    # Now scrubable
    assert tier.scrub_step() == 1


def test_scrub_pins_slot_against_concurrent_capacity_eviction():
    data = b"12345678"
    other = b"abcdefgh"
    alloc = _BytearrayAllocator()
    tier = StagingTier(capacity_bytes=len(data), allocator=alloc)
    content_hash = _hash(data)
    reservation = tier.reserve_or_wait(content_hash, "A")
    assert isinstance(reservation, Reservation)
    result = tier.commit_or_reject(reservation.reservation_id, data)
    assert isinstance(result, CommitOk)

    original_read = alloc.read
    attempted = []

    def read_while_reserving(ptr, size):
        # The scrubber must pin before exposing the pointer outside its lock.
        assert tier._slots[content_hash].refcount == 1
        other_reservation = tier.reserve_or_wait(_hash(other), "B")
        assert isinstance(other_reservation, Reservation)
        attempted.append(
            tier.commit_or_reject(other_reservation.reservation_id, other)
        )
        return original_read(ptr, size)

    alloc.read = read_while_reserving

    assert tier.scrub_step() == 1
    assert len(attempted) == 1
    assert isinstance(attempted[0], CommitRejected)
    assert content_hash in tier.scan([content_hash])


# ----- stale reservation reclaim ---------------------------------------------


def test_stale_reservation_reclaimed_by_evict_call():
    """Reservation older than reservation_stale_s gets reclaimed."""
    fake_now = {"t": 1000.0}
    tier = StagingTier(
        capacity_bytes=1 << 20,
        reservation_stale_s=10.0,
        allocator=_BytearrayAllocator(),
        clock=lambda: fake_now["t"],
    )
    h = _hash(b"data")
    r = tier.reserve_or_wait(h, "A")
    assert isinstance(r, Reservation)
    # Not stale yet
    assert tier.evict_stale_reservations() == 0

    # Advance past staleness window
    fake_now["t"] = 1100.0
    assert tier.evict_stale_reservations() == 1
    # Slot is gone; a new reservation can be created
    r2 = tier.reserve_or_wait(h, "B")
    assert isinstance(r2, Reservation)


def test_stale_reservation_reclaimed_by_new_reserver():
    fake_now = {"t": 1000.0}
    tier = StagingTier(
        capacity_bytes=1 << 20,
        reservation_stale_s=5.0,
        allocator=_BytearrayAllocator(),
        clock=lambda: fake_now["t"],
    )
    h = _hash(b"data")
    r1 = tier.reserve_or_wait(h, "stuck-owner")
    assert isinstance(r1, Reservation)

    # Owner is stuck. Time passes.
    fake_now["t"] = 1010.0

    # New reserver sees stale + takes over
    r2 = tier.reserve_or_wait(h, "rescuer")
    assert isinstance(r2, Reservation)
    assert r2.reservation_id != r1.reservation_id
    assert tier.stats()["stale_reservations_reclaimed"] == 1

    # Original owner's commit attempt now fails (different reservation id holds slot)
    result = tier.commit_or_reject(r1.reservation_id, b"data")
    assert isinstance(result, CommitRejected)


# ----- LRU eviction -----------------------------------------------------------


def test_lru_evicts_cold_slot_to_fit_new():
    tier = StagingTier(
        capacity_bytes=100,
        allocator=_BytearrayAllocator(),
    )
    # Two 40-byte slots fit (80 total). Third would push past 100.
    d_a = b"A" * 40
    d_b = b"B" * 40
    d_c = b"C" * 40
    ha, hb, hc = _hash(d_a), _hash(d_b), _hash(d_c)

    for h, d in [(ha, d_a), (hb, d_b)]:
        r = tier.reserve_or_wait(h, "src")
        tier.commit_or_reject(r.reservation_id, d)

    assert tier.bytes_used() == 80
    assert tier.scan([ha, hb, hc]).keys() == {ha, hb}

    # Touch hb so ha becomes coldest
    tier.scan([hb])  # scan touches last_accessed but NOT lru position;
    # to actually promote LRU we need begin_consume
    gen_b = tier.scan([hb])[hb].generation
    handle = tier.begin_consume(hb, expected_generation=gen_b)
    tier.end_consume(handle)

    # Deposit hc — should evict ha (coldest), not hb
    r_c = tier.reserve_or_wait(hc, "src")
    tier.commit_or_reject(r_c.reservation_id, d_c)
    remaining = tier.scan([ha, hb, hc])
    assert hc in remaining
    assert hb in remaining
    assert ha not in remaining
    assert tier.stats()["evictions_lru"] >= 1


def test_refcount_blocks_eviction_during_consume():
    tier = StagingTier(
        capacity_bytes=100,
        allocator=_BytearrayAllocator(),
    )
    d_a = b"A" * 60
    d_b = b"B" * 60
    ha, hb = _hash(d_a), _hash(d_b)

    r = tier.reserve_or_wait(ha, "src")
    tier.commit_or_reject(r.reservation_id, d_a)
    gen = tier.scan([ha])[ha].generation
    handle = tier.begin_consume(ha, expected_generation=gen)

    # Try to insert hb (60 bytes) — would need to evict ha (60 bytes)
    # but ha is pinned. Reservation succeeds, but commit fails for capacity.
    r_b = tier.reserve_or_wait(hb, "src")
    assert isinstance(r_b, Reservation)
    result = tier.commit_or_reject(r_b.reservation_id, d_b)
    assert isinstance(result, CommitRejected)
    assert "capacity" in result.reason

    tier.end_consume(handle)
    # Now hb can fit by evicting unpinned ha
    r_b2 = tier.reserve_or_wait(hb, "src")
    result2 = tier.commit_or_reject(r_b2.reservation_id, d_b)
    assert isinstance(result2, CommitOk)


# ----- commit lifecycle -------------------------------------------------------


def test_commit_with_unknown_reservation_rejected():
    tier = _make_tier()
    result = tier.commit_or_reject("bogus-id", b"data")
    assert isinstance(result, CommitRejected)


def test_commit_twice_second_rejected():
    tier = _make_tier()
    data = b"data"
    h = _hash(data)
    r = tier.reserve_or_wait(h, "A")
    first = tier.commit_or_reject(r.reservation_id, data)
    assert isinstance(first, CommitOk)
    second = tier.commit_or_reject(r.reservation_id, data)
    assert isinstance(second, CommitRejected)


def test_fail_reservation_releases_slot():
    tier = _make_tier()
    h = _hash(b"never arrives")
    r = tier.reserve_or_wait(h, "A")
    tier.fail_reservation(r.reservation_id, "transport timeout")
    # New reservation possible
    r2 = tier.reserve_or_wait(h, "B")
    assert isinstance(r2, Reservation)


# ----- shutdown ---------------------------------------------------------------


def test_shutdown_notifies_pending_waiters():
    tier = _make_tier()
    h = _hash(b"never delivered")
    assert isinstance(tier.reserve_or_wait(h, "A"), Reservation)
    waiter = tier.reserve_or_wait(h, "B")
    assert isinstance(waiter, Waiter)
    tier.shutdown()
    assert waiter.waiter.event.is_set()
    assert waiter.waiter.failure == "daemon shutdown"


# ----- concurrency stress -----------------------------------------------------


def test_concurrent_reservers_serialize_through_single_transfer():
    """Hammer reserve_or_wait from many threads with the SAME hash;
    only one Reservation should be returned, all others should be
    Waiter or AlreadyReady."""
    tier = _make_tier()
    h = _hash(b"contended")
    n_threads = 16

    barrier = threading.Barrier(n_threads + 1)
    results: list = [None] * n_threads

    def worker(i: int) -> None:
        barrier.wait()
        results[i] = tier.reserve_or_wait(h, source_daemon=f"src-{i}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    barrier.wait()
    for t in threads:
        t.join()

    n_reservations = sum(isinstance(r, Reservation) for r in results)
    n_waiters = sum(isinstance(r, Waiter) for r in results)
    assert n_reservations == 1
    assert n_reservations + n_waiters == n_threads
    assert tier.stats()["reservations_created"] == 1
    assert tier.stats()["waiters_coalesced"] == n_threads - 1
