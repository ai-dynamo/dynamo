# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The prefix cache's SSD tier on numpy rows: slots, round trips through the
file, backpressure, errors and completion tickets."""

import errno
import os
import sys
import threading
import types

import numpy as np
import pytest

from dynamo.deepinfer.kv_cache.ssd import SSDPool

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]

# DeepSeek-V4-Flash's packed KV block row: not a multiple of 4096.
ROW_BYTES = 114048
STRIDE = 114688


def rows_of(n, row_bytes=ROW_BYTES, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 256, row_bytes, dtype=np.uint8) for _ in range(n)]


def drain(pool, n):
    """Wait for n tickets to complete and poll them."""
    out = []
    while len(out) < n:
        ticket = pool._done.get(timeout=10)
        pool._done.put(ticket)  # leave it for poll()
        out += pool.poll()
    return out


@pytest.fixture
def pool(tmp_path):
    p = SSDPool(str(tmp_path / "rank0.kv"), ROW_BYTES, 32 * STRIDE)
    yield p
    p.close()


def test_rows_round_trip_byte_for_byte(pool):
    rows = rows_of(20)
    slots = [pool.alloc() for _ in rows]
    assert len(set(slots)) == 20
    ticket = pool.store(rows, slots)
    (done,) = drain(pool, 1)
    assert done is ticket and done.kind == "store" and done.error is None
    out = [np.zeros(ROW_BYTES, dtype=np.uint8) for _ in rows]
    # Read back in a different order than written.
    order = list(reversed(range(20)))
    pool.load([slots[i] for i in order], [out[i] for i in order])
    (done,) = drain(pool, 1)
    assert done.kind == "load" and done.error is None
    for want, got in zip(rows, out):
        assert np.array_equal(want, got)
    # Slots are padded to 4 KiB, so O_DIRECT stays possible later.
    assert pool.stride == STRIDE and pool.stride % 4096 == 0
    assert os.path.getsize(pool.path) == pool.nbytes
    with open(pool.path, "rb") as f:
        for slot, row in zip(slots, rows):
            f.seek(slot * pool.stride)
            data = f.read(pool.stride)
            assert np.array_equal(np.frombuffer(data[:ROW_BYTES], np.uint8), row)
            assert not any(data[ROW_BYTES:])  # padding stays zero


def test_inflight_writes_are_counted_until_they_finish(tmp_path):
    gate = threading.Event()

    class Event:  # a CUDA event stand-in that holds the write back
        def synchronize(self):
            assert gate.wait(10)

    pool = SSDPool(str(tmp_path / "k"), 4096, 8 * 4096, max_inflight_writes=2)
    try:
        rows = rows_of(2, 4096)
        assert pool.inflight_writes == 0 and pool.max_inflight_writes == 2
        pool.store(rows[:1], [0], after=Event())
        assert pool.inflight_writes == 1
        pool.store(rows[1:2], [1], after=Event())
        assert pool.inflight_writes == 2
        gate.set()
        drain(pool, 2)
        assert pool.inflight_writes == 0
    finally:
        gate.set()
        pool.close()


def test_store_waits_for_the_event(tmp_path):
    seen = []

    class Event:
        def synchronize(self):
            seen.append("sync")

    pool = SSDPool(str(tmp_path / "k"), 4096, 2 * 4096)
    try:
        pool.store(rows_of(1, 4096), [0], after=Event())
        drain(pool, 1)
        assert seen == ["sync"]
    finally:
        pool.close()


def test_load_waits_for_the_event(tmp_path):
    seen = []

    class Event:
        def synchronize(self):
            seen.append("sync")

    pool = SSDPool(str(tmp_path / "k"), 4096, 2 * 4096)
    try:
        (row,) = rows_of(1, 4096)
        pool.store([row], [0])
        drain(pool, 1)
        out = np.zeros(4096, dtype=np.uint8)
        pool.load([0], [out], after=Event(), tag="t")
        (done,) = drain(pool, 1)
        assert seen == ["sync"] and done.tag == "t" and np.array_equal(out, row)
    finally:
        pool.close()


def test_torch_rows_without_a_numpy_dtype_are_read_as_bytes(tmp_path, monkeypatch):
    # bfloat16 and float8 tensors have no numpy dtype: .numpy() refuses them.
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(uint8="uint8"))

    class Row:
        def __init__(self, data):
            self.data = data

        def numpy(self):
            raise TypeError("Got unsupported ScalarType BFloat16")

        def view(self, dtype):
            assert dtype == "uint8"
            return types.SimpleNamespace(numpy=lambda: self.data)

    pool = SSDPool(str(tmp_path / "k"), 4096, 2 * 4096)
    try:
        (want,) = rows_of(1, 4096)
        pool.store([Row(want.copy())], [1])
        (done,) = drain(pool, 1)
        assert done.error is None
        out = np.zeros(4096, dtype=np.uint8)
        pool.load([1], [Row(out)])
        (done,) = drain(pool, 1)
        assert done.error is None and np.array_equal(out, want)
    finally:
        pool.close()


def test_io_error_is_captured_in_the_ticket(pool, monkeypatch):
    def boom(*args):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "pwritev", boom)
    ticket = pool.store(rows_of(1), [pool.alloc()])
    (done,) = drain(pool, 1)
    assert done is ticket
    assert isinstance(done.error, OSError) and done.error.errno == 28
    assert pool.inflight_writes == 0


def test_failed_event_is_captured_in_the_ticket(pool):
    class Event:
        def synchronize(self):
            raise RuntimeError("CUDA error")

    pool.store(rows_of(1), [0], after=Event())
    (done,) = drain(pool, 1)
    assert isinstance(done.error, RuntimeError)


def test_bad_row_size_is_an_error_not_a_raise(pool):
    ticket = pool.load([0], [np.zeros(10, dtype=np.uint8)])
    (done,) = drain(pool, 1)
    assert done is ticket and isinstance(done.error, ValueError)


def test_short_reads_and_writes_are_looped(pool, monkeypatch):
    real_w, real_r = os.pwritev, os.preadv
    monkeypatch.setattr(
        os, "pwritev", lambda fd, bufs, off: real_w(fd, [bufs[0][:1000]], off)
    )
    monkeypatch.setattr(
        os, "preadv", lambda fd, bufs, off: real_r(fd, [bufs[0][:777]], off)
    )
    (row,) = rows_of(1)
    pool.store([row], [3])
    drain(pool, 1)
    out = np.zeros(ROW_BYTES, dtype=np.uint8)
    pool.load([3], [out])
    (done,) = drain(pool, 1)
    assert done.error is None and np.array_equal(out, row)


def test_read_past_the_end_is_an_error(tmp_path):
    pool = SSDPool(str(tmp_path / "k"), 4096, 2 * 4096)
    try:
        os.ftruncate(pool._fd, 0)
        pool.load([1], [np.zeros(4096, dtype=np.uint8)])
        (done,) = drain(pool, 1)
        assert isinstance(done.error, EOFError)
    finally:
        pool.close()


def test_file_is_recreated_empty(tmp_path):
    path = str(tmp_path / "rank0.kv")
    first = SSDPool(path, 4096, 4 * 4096)
    first.store([np.full(4096, 7, dtype=np.uint8)], [2])
    drain(first, 1)
    second = SSDPool(path, 4096, 4 * 4096)
    try:
        assert os.path.getsize(path) == 4 * 4096
        with open(path, "rb") as f:
            assert not any(f.read())
        assert second.num_free == 4
    finally:
        second.close()
        first._executor.shutdown()
        os.close(first._fd)


def test_poll_returns_each_ticket_once(pool):
    assert pool.poll() == []
    tickets = [pool.store(rows_of(1, seed=i), [i]) for i in range(5)]
    got = drain(pool, 5)
    assert sorted(id(t) for t in got) == sorted(id(t) for t in tickets)
    assert pool.poll() == []


def test_tag_travels_with_the_ticket(pool):
    pool.store(rows_of(1), [0], tag=("block", 42))
    (done,) = drain(pool, 1)
    assert done.tag == ("block", 42) and done.slots == [0]


def test_close_waits_deletes_and_is_idempotent(tmp_path):
    path = str(tmp_path / "k")
    pool = SSDPool(path, 4096, 8 * 4096)
    for i in range(8):
        pool.store(rows_of(1, 4096, i), [i])
    pool.close()
    # Everything in flight finished before close returned.
    done = pool.poll()
    assert len(done) == 8 and all(t.error is None for t in done)
    assert not os.path.exists(path)
    pool.close()
    late = pool.store(rows_of(1, 4096), [0])
    assert isinstance(late.error, RuntimeError)


@pytest.mark.parametrize("code", [errno.ENOSPC, errno.EOPNOTSUPP])
def test_failing_to_preallocate_turns_the_tier_off(tmp_path, monkeypatch, code):
    def refuse(fd, offset, length):
        raise OSError(code, os.strerror(code))

    monkeypatch.setattr(os, "posix_fallocate", refuse)
    path = tmp_path / "rank0.kv"
    pool = SSDPool(str(path), ROW_BYTES, 8 * STRIDE, reserve_fraction=0)
    assert pool.capacity == 0 and pool.alloc() is None
    assert not path.exists()
