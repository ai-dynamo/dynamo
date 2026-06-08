# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the SHM evict ring buffer.

Covers:
  - basic enqueue/drain round-trip preserving payload
  - head/tail / queue_depth accounting
  - backpressure (full ring → drops counter bumped, enqueue returns False)
  - cross-process attach (write in parent, read via fresh attach)
  - eventfd wakeup
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import pytest
from gpu_memory_service.common.evict_ring import (
    IPC_EVENT_HANDLE_LEN,
    MAX_RANGES_PER_RECORD,
    OP_EVICT,
    attach_ring_reader,
    create_ring,
    eventfd_read_nonblock,
    eventfd_write,
    make_eventfd,
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.none,
    pytest.mark.gpu_0,
]


def _ring_path(tmp_path: Path) -> str:
    return str(tmp_path / "evict_ring")


def test_create_round_trip(tmp_path):
    """A single enqueue → drain preserves all payload fields."""
    w = create_ring(_ring_path(tmp_path), capacity=8)
    r = attach_ring_reader(w.path)
    try:
        ipc = bytes(range(IPC_EVENT_HANDLE_LEN))
        ranges = [(3, 1024, 7 * 1024), (5, 4096, 0)]
        assert w.enqueue_evict(block_id=42, ipc_event_handle=ipc, ranges=ranges)
        assert w.queue_depth() == 1
        rec = r.drain_one()
        assert rec is not None
        assert rec["op"] == OP_EVICT
        assert rec["block_id"] == 42
        assert rec["ipc_event"] == ipc
        assert rec["ranges"] == ranges
        assert r.queue_depth() == 0
        # Empty drain returns None.
        assert r.drain_one() is None
    finally:
        r.close()
        w.close()


def test_drain_all_in_order(tmp_path):
    """Many enqueues drain in FIFO order."""
    w = create_ring(_ring_path(tmp_path), capacity=16)
    r = attach_ring_reader(w.path)
    try:
        N = 12
        for i in range(N):
            assert w.enqueue_evict(
                block_id=100 + i,
                ipc_event_handle=bytes([i]) * IPC_EVENT_HANDLE_LEN,
                ranges=[(0, i + 1, i * 64)],
            )
        seen = []
        r.drain_all(lambda rec: seen.append(rec["block_id"]))
        assert seen == list(range(100, 100 + N))
    finally:
        r.close()
        w.close()


def test_backpressure_drops_when_full(tmp_path):
    """When the ring is full, enqueue returns False and bumps drops."""
    w = create_ring(_ring_path(tmp_path), capacity=4)
    try:
        # Fill it.
        for i in range(4):
            assert w.enqueue_evict(block_id=i, ipc_event_handle=b"", ranges=[])
        assert w.queue_depth() == 4
        # Next enqueue must fail.
        assert w.drops() == 0
        assert not w.enqueue_evict(block_id=99, ipc_event_handle=b"", ranges=[])
        assert w.drops() == 1
        # After draining one, enqueue succeeds again.
        r = attach_ring_reader(w.path)
        try:
            assert r.drain_one() is not None
            assert w.enqueue_evict(block_id=99, ipc_event_handle=b"", ranges=[])
        finally:
            r.close()
    finally:
        w.close()


def test_attach_after_create_sees_records(tmp_path):
    """Writer creates and writes; a fresh reader attach picks them up."""
    path = _ring_path(tmp_path)
    w = create_ring(path, capacity=8)
    try:
        w.enqueue_evict(block_id=1, ipc_event_handle=b"", ranges=[(0, 128, 0)])
        w.enqueue_evict(block_id=2, ipc_event_handle=b"", ranges=[(1, 256, 128)])
    finally:
        w.close()
    # Fresh attach.
    r = attach_ring_reader(path)
    try:
        a = r.drain_one()
        b = r.drain_one()
        c = r.drain_one()
        assert a["block_id"] == 1
        assert b["block_id"] == 2
        assert c is None
    finally:
        r.close()


def test_too_many_ranges_drops(tmp_path):
    """Ranges > MAX_RANGES_PER_RECORD → drop + drops counter bump."""
    w = create_ring(_ring_path(tmp_path), capacity=8)
    try:
        ranges = [(i, 64, i * 64) for i in range(MAX_RANGES_PER_RECORD + 1)]
        assert w.drops() == 0
        assert not w.enqueue_evict(block_id=7, ipc_event_handle=b"", ranges=ranges)
        assert w.drops() == 1
        assert w.queue_depth() == 0
    finally:
        w.close()


def test_concurrent_single_producer_single_consumer(tmp_path):
    """One producer thread, one consumer thread; total ordering preserved
    and no records lost."""
    w = create_ring(_ring_path(tmp_path), capacity=128)
    r = attach_ring_reader(w.path)
    received: list[int] = []
    stop = threading.Event()
    N = 5000

    def producer():
        for i in range(N):
            # Spin until accepted (ring is large enough).
            while not w.enqueue_evict(
                block_id=i,
                ipc_event_handle=b"",
                ranges=[],
            ):
                time.sleep(0)

    def consumer():
        while not stop.is_set() or len(received) < N:
            rec = r.drain_one()
            if rec is None:
                time.sleep(1e-4)
                continue
            received.append(rec["block_id"])

    t_prod = threading.Thread(target=producer, daemon=True)
    t_cons = threading.Thread(target=consumer, daemon=True)
    t_cons.start()
    t_prod.start()
    t_prod.join(timeout=15)
    deadline = time.monotonic() + 10
    while len(received) < N and time.monotonic() < deadline:
        time.sleep(1e-3)
    stop.set()
    t_cons.join(timeout=3)

    assert received == list(range(N)), (
        f"expected 0..{N - 1} in order; got {received[:5]}...{received[-5:]} "
        f"(len={len(received)})"
    )
    r.close()
    w.close()


def test_eventfd_signal_and_consume(tmp_path):
    """eventfd_write delivers a signal; eventfd_read_nonblock retrieves
    the counter and clears it."""
    fd = make_eventfd()
    if fd < 0:
        pytest.skip("eventfd not available")
    try:
        # Initially empty.
        assert eventfd_read_nonblock(fd) == 0
        eventfd_write(fd, 1)
        eventfd_write(fd, 1)
        eventfd_write(fd, 1)
        # eventfd accumulates: reading once returns the sum.
        n = eventfd_read_nonblock(fd)
        assert n == 3
        # And is then empty again.
        assert eventfd_read_nonblock(fd) == 0
    finally:
        os.close(fd)
