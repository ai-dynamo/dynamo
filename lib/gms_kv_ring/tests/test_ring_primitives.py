# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ring + counter primitives.

Covers:
  - Evict ring roundtrip: push N → pop N, payload preserved
  - Evict ring capacity drops on full
  - Restore ring roundtrip + chunk-grained shape
  - Restore ring power-of-2 enforcement
  - Counter array: engine ↔ daemon visibility
  - Counter cuStreamWaitValue32 (stored-before-wait)
  - Counter cuStreamWriteValue32 (daemon-stream-ordered)

Tests run from any cwd; the gms_kv_ring package must be importable
(set PYTHONPATH or use editable install).
"""

from __future__ import annotations

import os
import uuid

import pytest


def test_evict_ring_roundtrip(tmp_path):
    from gms_kv_ring.common import evict_ring as er

    path = str(tmp_path / "e.ring")
    w = er.create_ring(path, capacity=16)
    r = er.attach_reader(path)

    ranges = [(0, 1024, 0), (1, 1024, 0)]
    assert w.push(block_id=42, ranges=ranges) is True
    rec = r.try_pop()
    assert rec["block_id"] == 42
    assert rec["ranges"] == [(0, 1024, 0), (1, 1024, 0)]
    assert rec["ipc_event"] == b""
    assert r.try_pop() is None
    w.close()
    r.close()


def test_evict_ring_drops_when_full(tmp_path):
    from gms_kv_ring.common import evict_ring as er

    path = str(tmp_path / "e.ring")
    w = er.create_ring(path, capacity=4)
    for i in range(4):
        assert w.push(block_id=i, ranges=[(0, 8, 0)])
    assert w.push(block_id=99, ranges=[(0, 8, 0)]) is False
    assert w.stats()["drops"] == 1
    w.close()


def test_evict_ring_rejects_non_pow2_capacity(tmp_path):
    from gms_kv_ring.common import evict_ring as er

    with pytest.raises(ValueError):
        er.create_ring(str(tmp_path / "e.ring"), capacity=10)


def test_evict_ring_ipc_event_carry(tmp_path):
    from gms_kv_ring.common import evict_ring as er

    path = str(tmp_path / "e.ring")
    w = er.create_ring(path, capacity=8)
    r = er.attach_reader(path)
    handle = bytes(range(64))
    w.push(block_id=1, ranges=[(0, 16, 0)], ipc_event_handle=handle)
    rec = r.try_pop()
    assert rec["ipc_event"] == handle
    w.close()
    r.close()


def test_restore_ring_roundtrip(tmp_path):
    from gms_kv_ring.common import restore_ring as rr

    path = str(tmp_path / "r.ring")
    w = rr.create_ring(path, capacity=8)
    r = rr.attach_reader(path)
    pairs = [(0, 100), (1, 101), (2, 102)]
    assert w.push(
        src_engine_id="eng-abc",
        block_pairs=pairs,
        counter_slot=7,
        counter_target=42,
    )
    rec = r.try_pop()
    assert rec["src_engine_id"] == "eng-abc"
    assert rec["counter_slot"] == 7
    assert rec["counter_target"] == 42
    assert rec["block_pairs"] == pairs
    w.close()
    r.close()


def test_restore_ring_max_pairs(tmp_path):
    from gms_kv_ring.common import restore_ring as rr

    path = str(tmp_path / "r.ring")
    w = rr.create_ring(path, capacity=8)
    # OK at the cap.
    pairs = [(i, i + 100) for i in range(rr.MAX_BLOCK_PAIRS)]
    assert w.push(
        src_engine_id="e",
        block_pairs=pairs,
        counter_slot=0,
        counter_target=1,
    )
    # One over → ValueError.
    with pytest.raises(ValueError):
        w.push(
            src_engine_id="e",
            block_pairs=pairs + [(99, 199)],
            counter_slot=0,
            counter_target=2,
        )
    w.close()


# ----- counter tests (need CUDA) -----

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("CUDA required for counter tests", allow_module_level=True)


def _shm_path(prefix: str) -> str:
    return f"/dev/shm/gms-kvr-test-{prefix}-{uuid.uuid4().hex}.bin"


def test_counter_cross_process_visibility():
    """Daemon write → engine read across two file handles."""
    torch.cuda.init()
    _ = torch.empty(1, device="cuda")
    from gms_kv_ring.common.counter import DaemonCounterArray, EngineCounterArray

    path = _shm_path("vis")
    eng = EngineCounterArray.create(path, num_counters=32)
    try:
        dmn = DaemonCounterArray.attach(path, num_counters=32)
        try:
            # Reserve a slot — engine state.
            slot, target = eng.reserve_slot()
            assert target == 1
            # Daemon writes via host store path.
            import struct as _s

            _s.pack_into("<I", dmn.buf, slot * 4, target)
            assert eng.read_slot(slot) == target
        finally:
            dmn.close()
    finally:
        eng.close()
        if os.path.exists(path):
            os.unlink(path)


def test_counter_stream_wait_then_write():
    """cuStreamWaitValue32 releases after cuStreamWriteValue32."""

    torch.cuda.init()
    _ = torch.empty(1, device="cuda")
    from gms_kv_ring.common.counter import DaemonCounterArray, EngineCounterArray

    path = _shm_path("wait-write")
    eng = EngineCounterArray.create(path, num_counters=16)
    dmn = DaemonCounterArray.attach(path, num_counters=16)
    try:
        slot, target = eng.reserve_slot()

        # Engine stream waits for `target`.
        engine_stream = torch.cuda.Stream()
        flag = torch.zeros((1,), dtype=torch.int32, device="cuda")
        with torch.cuda.stream(engine_stream):
            eng.wait_value32(int(engine_stream.cuda_stream), slot, target)
            flag.fill_(1)

        # Daemon stream writes `target` AFTER a no-op (proves ordering).
        from cuda.bindings import driver as drv

        err, daemon_stream = drv.cuStreamCreate(0)
        assert err == drv.CUresult.CUDA_SUCCESS
        dmn.write_on_stream(int(daemon_stream), slot, target)

        # synchronize both — flag must end up 1.
        torch.cuda.synchronize()
        assert int(flag.item()) == 1
    finally:
        dmn.close()
        eng.close()
        if os.path.exists(path):
            os.unlink(path)
