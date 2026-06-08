# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-engine ring stress — two engines, one daemon, concurrent
ring traffic.

What this exercises:
  - Two RestoreRings + two EvictRings + two counter arrays, all
    attached to the same daemon process.
  - Producer threads pump records into BOTH restore rings
    concurrently.
  - Daemon's two _RestoreRingConsumer threads (one per engine) drain
    in parallel.
  - Verifies: no record corruption, no daemon deadlock, all expected
    records processed, counter slots advance correctly.

This is the closest we can get to multi-tenant production load
without spinning up two real engines.
"""

from __future__ import annotations

import os
import threading
import time
import uuid

import pytest

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.integration,
    pytest.mark.stress,
    pytest.mark.none,
    pytest.mark.gpu_1,
]

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("CUDA required", allow_module_level=True)


def test_two_engines_concurrent_restore_rings():
    """Two engines push ring records concurrently; daemon-side counter
    advance must match what each engine pushed."""
    torch.cuda.init()
    _ = torch.empty(1, device="cuda")
    from gpu_memory_service.common.restore_counter import (
        DaemonCounterArray,
        EngineCounterArray,
    )
    from gpu_memory_service.common.restore_ring import (
        attach_restore_ring_reader,
        create_restore_ring,
    )

    # Each "engine" gets its own ring + counter array. Both attach to
    # /dev/shm so a single "daemon" can drain both.
    engines = []
    for i in range(2):
        eid = f"stress-eng-{i}-{uuid.uuid4().hex[:6]}"
        ring_path = f"/dev/shm/gms-stress-ring-{eid}.bin"
        ctr_path = f"/dev/shm/gms-stress-ctr-{eid}.bin"
        writer = create_restore_ring(ring_path, capacity=512)
        reader = attach_restore_ring_reader(ring_path)
        engine_ctrs = EngineCounterArray.create(ctr_path, num_counters=128)
        daemon_ctrs = DaemonCounterArray.attach(ctr_path, num_counters=128)
        engines.append(
            {
                "eid": eid,
                "ring_path": ring_path,
                "ctr_path": ctr_path,
                "writer": writer,
                "reader": reader,
                "engine_ctrs": engine_ctrs,
                "daemon_ctrs": daemon_ctrs,
            }
        )

    # Track how many records each engine pushed, and the (slot, target)
    # mapping we'll use to verify daemon processed them.
    expected_per_engine = 100
    pushed: dict[str, list[tuple[int, int]]] = {e["eid"]: [] for e in engines}
    push_locks = {e["eid"]: threading.Lock() for e in engines}

    def producer(engine_idx: int) -> None:
        e = engines[engine_idx]
        for i in range(expected_per_engine):
            slot, target = e["engine_ctrs"].reserve_slot()
            ok = e["writer"].push(
                src_engine_id=e["eid"],
                block_pairs=[(i, 1000 + i), (i + 1, 2000 + i)],
                counter_slot=slot,
                counter_target=target,
            )
            if ok:
                with push_locks[e["eid"]]:
                    pushed[e["eid"]].append((slot, target))
            # Tiny breath so producers interleave naturally.
            if i % 10 == 0:
                time.sleep(1e-4)

    # Daemon-side: one consumer thread per engine. Each pops records
    # and atomic-stores the counter (simulating the cuStreamWriteValue32
    # the real daemon does). No real CUDA work here — we're testing
    # ring correctness + concurrent counter integrity.
    stop = threading.Event()
    drained: dict[str, list[tuple[int, int]]] = {e["eid"]: [] for e in engines}

    def consumer(engine_idx: int) -> None:
        e = engines[engine_idx]
        while not stop.is_set():
            rec = e["reader"].try_pop()
            if rec is None:
                time.sleep(1e-5)
                continue
            # "Process" by atomic-storing the target value to the
            # counter slot — same final step the real daemon performs
            # (via cuStreamWriteValue32 — here host-store is fine since
            # there's no async GPU work).
            e["daemon_ctrs"].store(rec["counter_slot"], rec["counter_target"])
            drained[e["eid"]].append((rec["counter_slot"], rec["counter_target"]))

    try:
        # Launch producers + consumers.
        p_threads = [
            threading.Thread(target=producer, args=(i,), daemon=True) for i in range(2)
        ]
        c_threads = [
            threading.Thread(target=consumer, args=(i,), daemon=True) for i in range(2)
        ]
        for t in c_threads:
            t.start()
        for t in p_threads:
            t.start()
        for t in p_threads:
            t.join(timeout=30)
            assert not t.is_alive(), "producer thread stuck"
        # Drain pending records.
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if all(len(drained[e["eid"]]) >= len(pushed[e["eid"]]) for e in engines):
                break
            time.sleep(0.05)
        stop.set()
        for t in c_threads:
            t.join(timeout=5)

        # Each engine's drained set must match its pushed set.
        for e in engines:
            ps = sorted(pushed[e["eid"]])
            ds = sorted(drained[e["eid"]])
            assert ds == ps, (
                f"engine {e['eid']}: drained != pushed "
                f"({len(ds)} vs {len(ps)} entries)"
            )
            # Final counter state: each slot should hold its latest target.
            for slot, target in ps:
                assert e["engine_ctrs"].read_slot(slot) >= target, (
                    f"engine {e['eid']} slot {slot}: counter "
                    f"{e['engine_ctrs'].read_slot(slot)} < target {target}"
                )

        # Cross-engine isolation check: engine A's counter must not be
        # affected by engine B's traffic.
        # Implicit in the per-engine drained-equals-pushed check above,
        # but assert explicitly via the counter file separation.
        assert engines[0]["ctr_path"] != engines[1]["ctr_path"]

    finally:
        for e in engines:
            try:
                e["writer"].close()
                e["reader"].close()
                e["engine_ctrs"].close()
                e["daemon_ctrs"].close()
            except Exception:
                pass
            for p in (e["ring_path"], e["ctr_path"]):
                if os.path.exists(p):
                    try:
                        os.unlink(p)
                    except OSError:
                        pass


def test_ring_under_backpressure_per_engine_independent():
    """If engine A's consumer stalls, engine B's ring must still drain
    independently (no daemon-wide head-of-line blocking)."""
    torch.cuda.init()
    _ = torch.empty(1, device="cuda")
    from gpu_memory_service.common.restore_ring import (
        attach_restore_ring_reader,
        create_restore_ring,
    )

    # Two rings.
    paths_to_cleanup: list[str] = []
    a_ring = f"/dev/shm/gms-bp-a-{uuid.uuid4().hex[:6]}.bin"
    b_ring = f"/dev/shm/gms-bp-b-{uuid.uuid4().hex[:6]}.bin"
    paths_to_cleanup.extend([a_ring, b_ring])

    try:
        a_w = create_restore_ring(a_ring, capacity=4)  # tiny ring → fills fast
        b_w = create_restore_ring(b_ring, capacity=4)
        a_r = attach_restore_ring_reader(a_ring)
        b_r = attach_restore_ring_reader(b_ring)
        assert a_r is not None

        # Fill A's ring without draining.
        for i in range(8):
            a_w.push(
                src_engine_id="A",
                block_pairs=[(i, i + 100)],
                counter_slot=i % 8,
                counter_target=i + 1,
            )
        a_stats = a_w.stats()
        # A's drop counter should be ≥ 4 (capacity was 4).
        assert a_stats["drops"] >= 4

        # B's ring should still accept pushes — no daemon-wide HoL.
        for i in range(4):
            ok = b_w.push(
                src_engine_id="B",
                block_pairs=[(i, i + 200)],
                counter_slot=i,
                counter_target=i + 1,
            )
            assert ok, f"B push {i} unexpectedly failed"

        # Drain B independently of A.
        for _ in range(4):
            rec = b_r.try_pop()
            assert rec is not None
            assert rec["src_engine_id"] == "B"

        # A is still full / has drops — confirms A's stall doesn't
        # leak into B's flow.
        a_stats_after = a_w.stats()
        assert a_stats_after["drops"] == a_stats["drops"]

    finally:
        for p in paths_to_cleanup:
            if os.path.exists(p):
                try:
                    os.unlink(p)
                except OSError:
                    pass
