# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-engine throughput microbench (1000 spills + 1000 restores).

Sanity-check perf datapoint for the clean-branch tree. Not a
regression assertion — just a printed number. Numbers should be
comparable to (or better than) the legacy tree's ring-perf bench.

Previously this module also contained two-engine concurrent-spill
and cross-engine-restore tests; both were removed 2026-05-15 when
the project adopted the single-active-engine model (see memory:
project_single_active_engine.md). Multi-engine activity on one
daemon is OUT OF SCOPE.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
import uuid

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("CUDA required", allow_module_level=True)


def _spawn_daemon(socket_path: str):
    from gms_kv_ring.daemon.server import Daemon

    d = Daemon(socket_path)
    loop_holder = {}

    def _runner():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop_holder["loop"] = loop
        try:
            loop.run_until_complete(d.serve())
        finally:
            loop.close()

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not os.path.exists(socket_path):
        time.sleep(0.02)
    if not os.path.exists(socket_path):
        raise RuntimeError(f"daemon did not create {socket_path} within 5s")
    return d, t, loop_holder


def _wait_for_counter(handle, slot: int, target: int, deadline: float) -> bool:
    """Bounded host-side wait for a ring acknowledgement.

    Polling is intentional here: enqueueing thousands of CUDA stream waits and
    synchronizing once can deadlock with ``cudaFreeHost``'s context-wide sync
    when the benchmark repeatedly replaces the same host slots. Real engine
    integrations also consume each acknowledgement before reusing a counter.
    """
    while time.monotonic() < deadline:
        value = handle.counters.read_slot(slot)
        if value >= target:
            return value == target
        time.sleep(0.0001)
    return False


def _build_pool(n_layers, layer_size):
    """Allocate a device buffer, fill with a pattern, return (tensor, layer_descs)."""
    dev = torch.zeros(n_layers * layer_size, dtype=torch.uint8, device="cuda")
    host = torch.arange(n_layers * layer_size, dtype=torch.uint8)
    dev.copy_(host)
    torch.cuda.synchronize()
    base = int(dev.data_ptr())
    layers = [
        {
            "layer_idx": i,
            "va": base + i * layer_size,
            "size": layer_size,
            "stride": 1024,
        }
        for i in range(n_layers)
    ]
    return dev, layers


def test_sustained_throughput_microbench(tmp_path, capsys):
    """Push 1000 spills + 1000 restores on one engine; report wall time.

    This is a sanity-check perf number for the clean-branch tree.
    Not a regression assertion — just a printed datapoint for the
    review reader. The numbers should be comparable to (or better
    than) the legacy tree's ring-perf bench."""
    torch.cuda.init()
    _ = torch.empty(1, device="cuda")
    from gms_kv_ring.engines.handle import GMSKvRing

    sock = str(tmp_path / "d.sock")
    d, t, lh = _spawn_daemon(sock)
    try:
        n_layers = 4
        layer_size = 64 * 1024  # 64 KB
        stride = 1024
        dev, layers = _build_pool(n_layers, layer_size)
        n_blocks = layer_size // stride  # 64

        eid = f"perf-{uuid.uuid4().hex[:6]}"
        # Bump ring capacities so the producer doesn't block on the
        # consumer in this microbench.
        h = GMSKvRing(
            engine_id=eid,
            daemon_socket=sock,
            layers=layers,
            evict_ring_capacity=4096,
            restore_ring_capacity=4096,
            num_counters=4096,
        )
        try:
            # ---- spill bench ----
            N_SPILLS = 1000
            t0 = time.monotonic()
            pending_spills = []
            for i in range(N_SPILLS):
                blk = i % n_blocks
                ranges = [(li, stride, blk * stride) for li in range(n_layers)]
                # First pass populates distinct slots; later passes exercise
                # replacement and retirement of the same host-tier keys.
                ack = h.record_evict(block_id=blk, ranges=ranges)
                if ack is None:
                    # Producer outran consumer — let it drain a bit.
                    time.sleep(0.001)
                    continue
                pending_spills.append(ack)
            assert len(pending_spills) == N_SPILLS, "spill ring rejected a record"
            spill_dt = time.monotonic() - t0

            # Drain every accepted spill before measuring restores. Merely
            # waiting for the number of distinct host slots is insufficient:
            # repeated blocks may still be queued behind the first 64.
            deadline = time.monotonic() + 15
            assert all(
                _wait_for_counter(h, slot, target, deadline)
                for slot, target in pending_spills
            ), "spill consumer did not drain successfully"
            assert d.host_tier.n_slots() == n_blocks * n_layers

            # ---- restore bench ----
            cs = torch.cuda.Stream()
            N_RESTORES = 1000
            t0 = time.monotonic()
            for i in range(N_RESTORES):
                blk = i % n_blocks
                # restore block `blk` into the same slot — simplest case.
                res = h.record_restore(
                    src_engine_id=eid,
                    block_pairs=[(blk, blk)],
                )
                if res is None:
                    time.sleep(0.001)
                    continue
                slot, target = res
                with torch.cuda.stream(cs):
                    h.wait_restore(int(cs.cuda_stream), slot, target)
                # Consume each acknowledgement before reserving another
                # counter or replacing another source slot. This keeps the
                # benchmark faithful to engine lifecycle semantics and makes
                # failure bounded and attributable to a specific operation.
                cs.synchronize()
                assert h.restore_succeeded(slot, target)
            restore_dt = time.monotonic() - t0

            spill_us = spill_dt / N_SPILLS * 1e6
            restore_us = restore_dt / N_RESTORES * 1e6
            with capsys.disabled():
                print(
                    f"\n  spill   producer:   {spill_us:7.2f} µs/op "
                    f"({N_SPILLS} ops, {spill_dt:.3f} s)"
                )
                print(
                    f"  restore round-trip: {restore_us:7.2f} µs/op "
                    f"({N_RESTORES} ops, {restore_dt:.3f} s)"
                )
            # Loose bounds. Restore is a sequential round-trip now, so allow
            # scheduler variance under xdist while still catching a stall.
            assert spill_us < 1000, f"spill too slow: {spill_us:.2f} µs"
            assert restore_us < 5000, f"restore too slow: {restore_us:.2f} µs"
        finally:
            h.close()
    finally:
        lh["loop"].call_soon_threadsafe(d.stop)
        t.join(timeout=3)
        assert not t.is_alive(), "daemon did not stop within 3s"
