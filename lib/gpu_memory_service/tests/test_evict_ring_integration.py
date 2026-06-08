# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end async-eviction ring integration test.

Decoupled from CUDA so it can run on any host:

  • Producer thread (simulated engine) enqueues N EVICT records into a
    SHM ring at a known path.
  • Consumer thread (simulated daemon) drains the ring and forwards
    each record to a Python callback.
  • We assert: every record is delivered, in order, with payload
    intact, and the producer's per-call cost is sub-µs (or at least
    not dominated by socket-RPC-scale work).

This proves the substrate works end-to-end. Real-engine perf
measurement still depends on the daemon being able to attach +
cudaMemcpyAsync, which on the current dev host is blocked by the
cuMemMap-during-engine-runs constraint (see test_real_vllm_evict_hook
docstring). The substrate piece is what this test validates.
"""

from __future__ import annotations

import threading
import time

import pytest
from gpu_memory_service.common.evict_ring import (
    IPC_EVENT_HANDLE_LEN,
    OP_EVICT,
    attach_ring_reader,
    create_ring,
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.integration,
    pytest.mark.none,
    pytest.mark.gpu_0,
]


def test_async_ring_e2e_throughput(tmp_path):
    """N=5000 eviction records traverse the ring end-to-end. Asserts
    delivery + ordering + producer per-call cost ~ ring enqueue cost."""
    ring_path = str(tmp_path / "evict_ring")
    writer = create_ring(ring_path, capacity=512)
    reader = attach_ring_reader(ring_path)

    N = 5000
    received: list[dict] = []
    stop = threading.Event()

    def consumer():
        while not stop.is_set() or reader.queue_depth() > 0:
            n = reader.drain_all(lambda rec: received.append(rec))
            if n == 0:
                time.sleep(1e-4)

    t = threading.Thread(target=consumer, daemon=True)
    t.start()

    ipc = bytes(range(IPC_EVENT_HANDLE_LEN))
    ranges = [(i, 16384, i * 16384) for i in range(8)]  # 8 layers per block
    t0 = time.perf_counter()
    for i in range(N):
        # Spin until accepted — ring is small so this exercises
        # backpressure too. Wait shouldn't be long because consumer is
        # draining concurrently.
        while not writer.enqueue_evict(
            block_id=i,
            ipc_event_handle=ipc,
            ranges=ranges,
        ):
            time.sleep(0)
    e2e_enqueue_elapsed = time.perf_counter() - t0
    e2e_per_call_us = (e2e_enqueue_elapsed / N) * 1e6

    # Wait for consumer to drain.
    deadline = time.monotonic() + 5
    while len(received) < N and time.monotonic() < deadline:
        time.sleep(1e-3)
    stop.set()
    t.join(timeout=2)

    assert len(received) == N, f"only {len(received)}/{N} delivered"

    # Order preservation: every record's block_id is the loop index it
    # was enqueued at.
    for i, rec in enumerate(received):
        assert rec["op"] == OP_EVICT
        assert (
            rec["block_id"] == i
        ), f"out-of-order at index {i}: got block_id={rec['block_id']}"
    # Payload preservation: ipc + ranges intact.
    assert received[0]["ipc_event"] == ipc
    assert received[0]["ranges"] == ranges

    print(
        f"\n[evict-ring e2e] N={N} enqueue_time={e2e_enqueue_elapsed * 1000:.1f}ms "
        f"per_call={e2e_per_call_us:.2f}µs drops={writer.drops()}",
    )

    reader.close()
    writer.close()

    # Producer cost target: measure the enqueue path separately from
    # consumer scheduling and intentional backpressure in the e2e section.
    perf_path = str(tmp_path / "evict_ring_perf")
    perf_writer = create_ring(perf_path, capacity=8192)
    try:
        t0 = time.perf_counter()
        for i in range(N):
            assert perf_writer.enqueue_evict(
                block_id=i,
                ipc_event_handle=ipc,
                ranges=ranges,
            )
        enqueue_elapsed = time.perf_counter() - t0
        per_call_us = (enqueue_elapsed / N) * 1e6
        print(
            f"\n[evict-ring enqueue] N={N} enqueue_time={enqueue_elapsed * 1000:.1f}ms "
            f"per_call={per_call_us:.2f}µs drops={perf_writer.drops()}",
        )
    finally:
        perf_writer.close()

    # Per-call cost target: well under typical RPC roundtrip (~30-50µs).
    # Generous bound to absorb CI noise.
    assert per_call_us < 20.0, (
        f"per-call producer cost {per_call_us:.2f}µs exceeds 20µs budget — "
        f"likely a regression in the ring's enqueue path"
    )


def test_async_ring_backpressure_falls_back_gracefully(tmp_path):
    """When the ring is full, enqueue returns False and the drops
    counter bumps. The caller is responsible for falling back (sync
    RPC, drop, retry) — the ring's contract is just to signal."""
    ring_path = str(tmp_path / "evict_ring")
    writer = create_ring(ring_path, capacity=4)
    try:
        for i in range(4):
            assert writer.enqueue_evict(
                block_id=i,
                ipc_event_handle=b"",
                ranges=[],
            )
        assert writer.queue_depth() == 4
        # 5th must drop.
        assert not writer.enqueue_evict(
            block_id=99,
            ipc_event_handle=b"",
            ranges=[],
        )
        assert writer.drops() == 1
    finally:
        writer.close()
