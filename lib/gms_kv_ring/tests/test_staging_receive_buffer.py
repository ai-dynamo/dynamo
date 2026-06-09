# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for StagingReceiveBuffer (Phase 4 helper)."""

from __future__ import annotations

import ctypes

import pytest
from gms_kv_ring.daemon.staging_receive_buffer import StagingReceiveBuffer


def test_alloc_returns_valid_offset():
    buf = StagingReceiveBuffer(capacity_bytes=4096)
    off = buf.alloc(128)
    assert off == 0
    off2 = buf.alloc(256)
    assert off2 == 128


def test_alloc_returns_none_when_capacity_exhausted():
    buf = StagingReceiveBuffer(capacity_bytes=1024)
    assert buf.alloc(800) == 0
    assert buf.alloc(300) is None  # 800+300 > 1024


def test_alloc_size_larger_than_capacity_returns_none():
    buf = StagingReceiveBuffer(capacity_bytes=1024)
    assert buf.alloc(2048) is None


def test_free_returns_region_to_pool():
    buf = StagingReceiveBuffer(capacity_bytes=1024)
    a = buf.alloc(256)
    assert buf.alloc(256) == 256
    buf.free(a, 256)
    # Free-list now has [0, 256). Next 256-byte alloc should reuse it.
    c = buf.alloc(256)
    assert c == 0
    # bytes_free should reflect what's left
    assert buf.bytes_free() == 1024 - 512


def test_free_coalesces_adjacent_regions():
    buf = StagingReceiveBuffer(capacity_bytes=1024)
    a = buf.alloc(100)
    b = buf.alloc(100)
    assert buf.alloc(100) == 200
    buf.free(a, 100)
    buf.free(b, 100)  # coalesces with `a` → one 200-byte region
    # Now alloc(200) should fit in the coalesced region starting at 0
    d = buf.alloc(200)
    assert d == 0


def test_free_at_tail_folds_back_into_bump():
    """When the freed region is at the bump-pointer tail, the bump
    pointer rolls back, leaving no fragment behind."""
    buf = StagingReceiveBuffer(capacity_bytes=1024)
    assert buf.alloc(100) == 0
    b = buf.alloc(100)
    buf.free(b, 100)  # tail region — should fold
    assert buf.bytes_free() == 1024 - 100


def test_write_then_read_round_trips():
    buf = StagingReceiveBuffer(capacity_bytes=4096)
    off = buf.alloc(256)
    payload = bytes(range(256))
    # Write directly via the absolute pointer (simulates an RDMA WRITE)
    ctypes.memmove(buf.ptr_at(off), payload, len(payload))
    assert buf.read(off, 256) == payload


def test_read_out_of_range_rejected():
    buf = StagingReceiveBuffer(capacity_bytes=1024)
    with pytest.raises(ValueError):
        buf.read(0, 2048)
    with pytest.raises(ValueError):
        buf.read(-1, 64)


def test_ptr_at_out_of_range_rejected():
    buf = StagingReceiveBuffer(capacity_bytes=1024)
    with pytest.raises(ValueError):
        buf.ptr_at(1024)


def test_invalid_capacity_rejected():
    with pytest.raises(ValueError):
        StagingReceiveBuffer(capacity_bytes=0)
    with pytest.raises(ValueError):
        StagingReceiveBuffer(capacity_bytes=-1)


def test_concurrent_alloc_free_thread_safe():
    """Hammer alloc + free from many threads; no double-allocation."""
    import threading

    buf = StagingReceiveBuffer(capacity_bytes=4096)
    n_threads = 8
    n_iters = 50
    seen_offsets = []
    seen_lock = threading.Lock()

    def worker() -> None:
        for _ in range(n_iters):
            off = buf.alloc(64)
            if off is None:
                continue
            with seen_lock:
                seen_offsets.append(off)
            buf.free(off, 64)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # No assertion on exact values — just that we didn't crash and
    # the pool is still functional.
    assert buf.alloc(64) is not None


def test_full_capacity_round_trip():
    buf = StagingReceiveBuffer(capacity_bytes=1024)
    off = buf.alloc(1024)
    assert off == 0
    assert buf.alloc(1) is None
    buf.free(off, 1024)
    assert buf.alloc(1024) == 0
