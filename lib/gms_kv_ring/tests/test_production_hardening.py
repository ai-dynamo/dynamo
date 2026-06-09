# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the pre-production hardening fixes:

R1. Re-attaching an engine_id wipes its prior host-tier slots — silent
    data corruption guard for engine respawn / pid recycling (this is
    also the failover path under the single-active-engine model).
R2 (removed). Cross-engine restore source-stream sync — this fix and
    its tests were removed 2026-05-15 when concurrent multi-active-
    engine on one daemon was declared out of scope. See memory:
    project_single_active_engine.md.
R3 (consumer cleanup). consumer.stop() doesn't munmap mid-flight —
    even if join times out the consumer thread owns its buffer lifetime.

Second-round hardening (under "deployment restart" assumption):

R4 (counter on error). Consumer exception path still signals the
    counter so engine's cuStreamWaitValue32 unblocks instead of hanging.
R5 (sync-fail safety). detach_engine_pool skips cudaFreeHost when
    cuStreamSynchronize errors — leak buffers vs use-after-free.
R6 (SPSC under multi-thread). record_evict / record_restore are
    serialized by per-ring producer locks so misuse from multiple
    engine threads doesn't corrupt the ring.
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
    holder = {}

    def _runner():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        holder["loop"] = loop
        try:
            loop.run_until_complete(d.serve())
        finally:
            loop.close()

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not os.path.exists(socket_path):
        time.sleep(0.02)
    return d, t, holder


def _build_pool(n_layers, layer_size, fill=None):
    dev = torch.zeros(n_layers * layer_size, dtype=torch.uint8, device="cuda")
    if fill is not None:
        dev.fill_(fill)
    else:
        dev.copy_(torch.arange(n_layers * layer_size, dtype=torch.uint8))
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


# ---------------------------------------------------------------------------
# Fix #1: re-attach wipes prior pool
# ---------------------------------------------------------------------------


def test_reattach_engine_id_wipes_prior_host_tier_slots(tmp_path):
    """Simulate engine respawn: a pool exists in _pools (prior engine
    didn't get to detach cleanly), host_tier has slots keyed by that
    engine_id, and a new attach with the same engine_id arrives. The
    new attach must wipe those slots — otherwise a respawned engine
    inherits stale HBM VA references from the dead process."""
    torch.cuda.init()
    _ = torch.empty(1, device="cuda")
    from gms_kv_ring.daemon.client import DaemonClient

    sock = str(tmp_path / "d.sock")
    d, t, lh = _spawn_daemon(sock)
    try:
        eid = f"respawn-{uuid.uuid4().hex[:6]}"
        c = DaemonClient(sock)
        try:
            _, layers1 = _build_pool(1, 4096)

            # First attach establishes a pool in _pools.
            c.attach_engine_pool(eid, layers1)
            assert eid in d._pools, "first attach didn't register pool"

            # Simulate the dead engine having committed-spill bytes —
            # slot keyed by THIS engine_id, pointing at the (now-stale)
            # HBM layout from before the "respawn". Mark ready so
            # host_tier.get returns it.
            d.host_tier.put(engine_id=eid, layer=99, offset=0, size=1024)
            d.host_tier.mark_ready(engine_id=eid, layer=99, offset=0)
            assert d.host_tier.get(eid, 99, 0) is not None

            # Now the "respawned" engine attaches — same engine_id,
            # different HBM addresses (would be a new tensor data_ptr
            # in real life).
            _, layers2 = _build_pool(1, 4096)
            c.attach_engine_pool(eid, layers2)

            assert d.host_tier.get(eid, 99, 0) is None, (
                "re-attach must wipe the prior incarnation's host-tier "
                "slots — otherwise respawned engine inherits stale state"
            )
            assert eid in d._pools, "re-attach should still leave a pool"
        finally:
            c.detach_engine_pool(eid)
            c.close()
    finally:
        lh["loop"].call_soon_threadsafe(d.stop)
        t.join(timeout=3)


# ---------------------------------------------------------------------------
# (former Fix #2 — cross-engine restore source-stream sync — was removed
# 2026-05-15 when concurrent multi-active-engine on one daemon was
# declared out of scope. See memory project_single_active_engine.md.
# The daemon's restore consumer now enforces
# src_engine_id == self.engine_id and signals failure on mismatch.)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Fix #3: consumer.stop owns reader cleanup; no munmap-during-run
# ---------------------------------------------------------------------------


def test_consumer_stop_does_not_munmap_while_consumer_alive(tmp_path):
    """If consumer is mid-processing when stop() is called, the reader
    mmap must stay alive — the consumer thread cleans it up on its own
    way out via its finally clause.

    Tested at the unit level on _EvictConsumer to avoid the cross-process
    counter setup complexity. Same invariant holds for _RestoreConsumer
    by code-share — they have identical stop()/run() shape."""
    torch.cuda.init()
    _ = torch.empty(1, device="cuda")
    from gms_kv_ring.daemon.server import _EvictConsumer

    # A minimal fake daemon so the consumer can look up its pool.
    class _FakeDaemon:
        _pools = {}

        class _FakeHostTier:
            def put(self, *a, **kw):
                return 0

        host_tier = _FakeHostTier()

    blocked = threading.Event()
    release = threading.Event()

    # Patch _process on the class so the consumer blocks on entry.
    orig_process = _EvictConsumer._process

    def slow_process(self, rec):
        blocked.set()
        release.wait(timeout=10)

    _EvictConsumer._process = slow_process
    try:
        ring_path = f"/dev/shm/gms-kvr-stoptest-{uuid.uuid4().hex}.ring"
        try:
            from gms_kv_ring.common.evict_ring import create_ring

            w = create_ring(ring_path, capacity=16)
            consumer = _EvictConsumer(_FakeDaemon(), "stop-test", ring_path)
            consumer.start()
            try:
                # Push a record so consumer enters _process and blocks.
                w.push(block_id=0, ranges=[(0, 1024, 0)])
                assert blocked.wait(timeout=3), "consumer never entered _process"

                # Reader must be alive — touch it.
                pre = consumer._reader.stats()
                assert pre["head"] >= 1

                # Issue stop in a background thread (it will block on
                # join with the 5s timeout).
                t0 = time.monotonic()
                stop_thread = threading.Thread(target=consumer.stop)
                stop_thread.start()

                # While stop is waiting, the reader must still be alive.
                time.sleep(0.3)
                try:
                    consumer._reader.stats()
                except Exception as e:
                    release.set()
                    stop_thread.join(timeout=10)
                    raise AssertionError(
                        f"reader was closed/munmap'd while consumer "
                        f"thread still alive: {type(e).__name__}: {e}"
                    )

                # Release the consumer so it can finish + exit.
                release.set()
                stop_thread.join(timeout=10)
                assert not stop_thread.is_alive(), "stop() never returned"
                elapsed = time.monotonic() - t0
                assert elapsed < 8, f"stop took too long: {elapsed:.1f}s"
            finally:
                release.set()
                try:
                    w.close()
                except Exception:
                    pass
        finally:
            try:
                os.unlink(ring_path)
            except OSError:
                pass
    finally:
        _EvictConsumer._process = orig_process


# ---------------------------------------------------------------------------
# F1: evict-ack contract + host_tier ready flag
#     Engine learns when spill is committed before reusing the HBM slot.
#     Daemon never exposes host_tier slots until DMA actually completes.
# ---------------------------------------------------------------------------


def test_evict_succeeded_returns_true_after_dma_committed(tmp_path):
    """Happy path: engine pushes evict, daemon completes D2H, signals
    ack. Engine's evict_succeeded() returns True."""
    torch.cuda.init()
    _ = torch.empty(1, device="cuda")
    from gms_kv_ring.engines.handle import GMSKvRing

    sock = str(tmp_path / "d.sock")
    d, t, lh = _spawn_daemon(sock)
    try:
        dev, layers = _build_pool(2, 4096)
        eid = f"evict-ok-{uuid.uuid4().hex[:6]}"
        h = GMSKvRing(engine_id=eid, daemon_socket=sock, layers=layers)
        try:
            res = h.record_evict(
                block_id=0,
                ranges=[(0, 1024, 0), (1, 1024, 0)],
            )
            assert res is not None, "evict push must succeed"
            slot, target = res

            deadline = time.monotonic() + 3
            while time.monotonic() < deadline and d.host_tier.n_slots() < 2:
                time.sleep(0.02)
            time.sleep(0.05)

            assert h.evict_succeeded(slot, target), (
                f"committed evict should report success; "
                f"counter={h.counters.read_slot(slot)} target={target}"
            )
        finally:
            h.close()
    finally:
        lh["loop"].call_soon_threadsafe(d.stop)
        t.join(timeout=3)


def test_evict_succeeded_returns_false_when_consumer_fails(tmp_path):
    """If evict consumer raises, engine's evict_succeeded() must
    return False — otherwise engine reuses HBM believing the spill
    is durable when it isn't."""
    torch.cuda.init()
    _ = torch.empty(1, device="cuda")
    from gms_kv_ring.daemon.server import _EvictConsumer
    from gms_kv_ring.engines.handle import GMSKvRing

    sock = str(tmp_path / "d.sock")
    d, t, lh = _spawn_daemon(sock)
    try:
        orig_process = _EvictConsumer._process

        def boom(self, rec):
            raise RuntimeError("synthetic")

        _EvictConsumer._process = boom
        try:
            dev, layers = _build_pool(1, 4096)
            eid = f"evict-fail-{uuid.uuid4().hex[:6]}"
            h = GMSKvRing(engine_id=eid, daemon_socket=sock, layers=layers)
            try:
                res = h.record_evict(block_id=0, ranges=[(0, 1024, 0)])
                assert res is not None
                slot, target = res

                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    if h.counters.read_slot(slot) != 0:
                        break
                    time.sleep(0.02)

                assert not h.evict_succeeded(slot, target), (
                    f"failed evict must report failure; "
                    f"counter={h.counters.read_slot(slot)} target={target}"
                )
            finally:
                h.close()
        finally:
            _EvictConsumer._process = orig_process
    finally:
        lh["loop"].call_soon_threadsafe(d.stop)
        t.join(timeout=3)


def test_host_tier_get_returns_none_until_evict_committed(tmp_path):
    """host_tier slot must be invisible until mark_ready is called."""
    from gms_kv_ring.daemon.host_tier import HostTier

    torch.cuda.init()
    _ = torch.empty(1, device="cuda")

    ht = HostTier()
    eid = "test-engine"
    ht.put(engine_id=eid, layer=0, offset=0, size=1024)
    assert ht.get(eid, 0, 0) is None, (
        "slot must be invisible until mark_ready — otherwise restore "
        "consumer reads partial DMA bytes"
    )
    assert ht.mark_ready(eid, 0, 0)
    slot = ht.get(eid, 0, 0)
    assert slot is not None and slot.ready
    ht.release_engine(eid)


# (former test_cross_engine_restore_signals_failure_for_not_ready_slot —
# removed 2026-05-15 with the rest of the multi-active-engine logic.)


# ---------------------------------------------------------------------------
# R4: consumer exception path signals counter, doesn't hang engine
# ---------------------------------------------------------------------------


def test_restore_consumer_signals_counter_on_process_exception(tmp_path):
    """If _process raises, the consumer must still write the counter
    to its target so the engine's cuStreamWaitValue32 unblocks.
    Without this, any CUDA error in the consumer wedges the engine."""
    torch.cuda.init()
    _ = torch.empty(1, device="cuda")
    from gms_kv_ring.daemon.server import _RestoreConsumer
    from gms_kv_ring.engines.handle import GMSKvRing

    sock = str(tmp_path / "d.sock")
    d, t, lh = _spawn_daemon(sock)
    try:
        # Hijack _process to raise — simulates any in-process failure
        # (CUDA OOM, host_tier slot missing, batch_h2d error, etc).
        orig_process = _RestoreConsumer._process

        def boom(self, rec):
            raise RuntimeError("synthetic: simulate process failure")

        _RestoreConsumer._process = boom
        try:
            dev, layers = _build_pool(1, 4096)
            eid = f"err-{uuid.uuid4().hex[:6]}"
            h = GMSKvRing(engine_id=eid, daemon_socket=sock, layers=layers)
            try:
                res = h.record_restore(src_engine_id=eid, block_pairs=[(0, 0)])
                assert res is not None
                slot, target = res

                cs = torch.cuda.Stream()
                with torch.cuda.stream(cs):
                    h.wait_restore(int(cs.cuda_stream), slot, target)
                # Critical: this would hang forever before the fix.
                # With the fix, the consumer's exception handler writes
                # the counter, releasing the wait. Wrap in a deadline
                # check just to be sure.
                t0 = time.monotonic()
                torch.cuda.synchronize()
                elapsed = time.monotonic() - t0
                assert elapsed < 5, (
                    f"synchronize took {elapsed:.1f}s — consumer error "
                    "path didn't unblock the engine wait"
                )
            finally:
                h.close()
        finally:
            _RestoreConsumer._process = orig_process
    finally:
        lh["loop"].call_soon_threadsafe(d.stop)
        t.join(timeout=3)


# ---------------------------------------------------------------------------
# C1+C4: restore_succeeded contract — engine learns success vs failure
# ---------------------------------------------------------------------------


def test_restore_succeeded_returns_true_on_success(tmp_path):
    """Happy path: daemon completes restore → engine's
    restore_succeeded() returns True."""
    torch.cuda.init()
    _ = torch.empty(1, device="cuda")
    from gms_kv_ring.engines.handle import GMSKvRing

    sock = str(tmp_path / "d.sock")
    d, t, lh = _spawn_daemon(sock)
    try:
        dev, layers = _build_pool(2, 4096)
        eid = f"ok-{uuid.uuid4().hex[:6]}"
        h = GMSKvRing(engine_id=eid, daemon_socket=sock, layers=layers)
        try:
            # Spill block 0 first so the restore has host_tier bytes.
            assert h.record_evict(
                block_id=0,
                ranges=[(0, 1024, 0), (1, 1024, 0)],
            )
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline and d.host_tier.n_slots() < 2:
                time.sleep(0.02)

            res = h.record_restore(src_engine_id=eid, block_pairs=[(0, 1)])
            assert res is not None
            slot, target = res

            cs = torch.cuda.Stream()
            with torch.cuda.stream(cs):
                h.wait_restore(int(cs.cuda_stream), slot, target)
            torch.cuda.synchronize()

            assert h.restore_succeeded(slot, target), (
                f"happy-path restore should report success; "
                f"counter={h.counters.read_slot(slot)} target={target}"
            )
        finally:
            h.close()
    finally:
        lh["loop"].call_soon_threadsafe(d.stop)
        t.join(timeout=3)


def test_restore_succeeded_returns_false_when_no_spill_exists(tmp_path):
    """Restore for blocks that were never spilled to host_tier — the
    daemon detects expected-vs-found mismatch and signals failure
    (target+1) instead of silently writing success and letting the
    engine read stale HBM."""
    torch.cuda.init()
    _ = torch.empty(1, device="cuda")
    from gms_kv_ring.common import metrics
    from gms_kv_ring.engines.handle import GMSKvRing

    sock = str(tmp_path / "d.sock")
    d, t, lh = _spawn_daemon(sock)
    try:
        dev, layers = _build_pool(2, 4096)
        eid = f"miss-{uuid.uuid4().hex[:6]}"
        before_failures = metrics.restore_failures.get(engine_id=eid)
        h = GMSKvRing(engine_id=eid, daemon_socket=sock, layers=layers)
        try:
            # No record_evict — host_tier is empty for this engine.
            res = h.record_restore(src_engine_id=eid, block_pairs=[(0, 1)])
            assert res is not None
            slot, target = res
            cs = torch.cuda.Stream()
            with torch.cuda.stream(cs):
                h.wait_restore(int(cs.cuda_stream), slot, target)
            torch.cuda.synchronize()
            # Engine wait must have unblocked (GEQ target satisfies on target+1)
            # but restore_succeeded must report False.
            assert not h.restore_succeeded(slot, target), (
                f"restore over a never-spilled block must report failure; "
                f"counter={h.counters.read_slot(slot)} target={target}"
            )
            assert (
                metrics.restore_failures.get(engine_id=eid) > before_failures
            ), "failure metric must increment"
        finally:
            h.close()
    finally:
        lh["loop"].call_soon_threadsafe(d.stop)
        t.join(timeout=3)


def test_restore_succeeded_returns_false_when_consumer_raises(tmp_path):
    """Consumer's _process throws (CUDA OOM, etc.) → engine's
    restore_succeeded() returns False instead of silently accepting
    garbage bytes."""
    torch.cuda.init()
    _ = torch.empty(1, device="cuda")
    from gms_kv_ring.common import metrics
    from gms_kv_ring.daemon.server import _RestoreConsumer
    from gms_kv_ring.engines.handle import GMSKvRing

    sock = str(tmp_path / "d.sock")
    d, t, lh = _spawn_daemon(sock)
    try:
        orig_process = _RestoreConsumer._process

        def boom(self, rec):
            raise RuntimeError("synthetic")

        _RestoreConsumer._process = boom
        try:
            dev, layers = _build_pool(1, 4096)
            eid = f"throw-{uuid.uuid4().hex[:6]}"
            before_failures = metrics.restore_failures.get(engine_id=eid)
            h = GMSKvRing(engine_id=eid, daemon_socket=sock, layers=layers)
            try:
                res = h.record_restore(src_engine_id=eid, block_pairs=[(0, 0)])
                assert res is not None
                slot, target = res
                cs = torch.cuda.Stream()
                with torch.cuda.stream(cs):
                    h.wait_restore(int(cs.cuda_stream), slot, target)
                # Engine sync must unblock (not hang).
                t0 = time.monotonic()
                torch.cuda.synchronize()
                assert time.monotonic() - t0 < 5, "engine wait wasn't released"
                # And restore must report failure, not silent success.
                assert not h.restore_succeeded(slot, target)
                assert metrics.restore_failures.get(engine_id=eid) > before_failures
            finally:
                h.close()
        finally:
            _RestoreConsumer._process = orig_process
    finally:
        lh["loop"].call_soon_threadsafe(d.stop)
        t.join(timeout=3)


# ---------------------------------------------------------------------------
# R5: sync-fail in detach skips cudaFreeHost (avoid use-after-free)
# ---------------------------------------------------------------------------


def test_detach_skips_free_when_stream_sync_fails(tmp_path):
    """If cuStreamSynchronize during detach fails, the host_tier slots
    must be forgotten (entries dropped) but NOT freed — freeing a
    pinned host buffer that's still the target of an in-flight
    cuMemcpyAsync is undefined behavior."""
    torch.cuda.init()
    _ = torch.empty(1, device="cuda")
    from gms_kv_ring.daemon import host_tier as ht_mod
    from gms_kv_ring.engines.handle import GMSKvRing

    sock = str(tmp_path / "d.sock")
    d, t, lh = _spawn_daemon(sock)
    try:
        dev, layers = _build_pool(1, 4096)
        eid = f"syncfail-{uuid.uuid4().hex[:6]}"
        h = GMSKvRing(engine_id=eid, daemon_socket=sock, layers=layers)
        try:
            # Land at least one host_tier slot.
            assert h.record_evict(block_id=0, ranges=[(0, 1024, 0)])
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline and d.host_tier.n_slots() < 1:
                time.sleep(0.02)
            n_before = d.host_tier.n_slots()
            assert n_before >= 1

            # Patch _free_host to track calls. Then trigger detach with
            # a poisoned cuStreamSynchronize.
            free_calls = []
            orig_free = ht_mod._free_host

            def tracking_free(ptr):
                free_calls.append(ptr)
                return orig_free(ptr)

            ht_mod._free_host = tracking_free

            from cuda.bindings import driver as drv

            orig_sync = drv.cuStreamSynchronize

            def bad_sync(stream):
                # Return CUDA error code instead of success
                raise RuntimeError("synthetic: simulate sync failure")

            drv.cuStreamSynchronize = bad_sync
            try:
                # Trigger detach via direct API (h.close → detach).
                # Note: detach catches the exception inside its try.
                h._client.detach_engine_pool(eid)

                # After detach: slots dropped from index but _free_host
                # NOT called for them.
                assert d.host_tier.n_slots() == 0, "slots should be dropped from index"
                assert len(free_calls) == 0, (
                    f"cudaFreeHost called {len(free_calls)} times despite "
                    f"sync failure — would be use-after-free under in-flight DMA"
                )
            finally:
                drv.cuStreamSynchronize = orig_sync
                ht_mod._free_host = orig_free
        finally:
            try:
                h.close()
            except Exception:
                pass
    finally:
        lh["loop"].call_soon_threadsafe(d.stop)
        t.join(timeout=3)


# ---------------------------------------------------------------------------
# R6: producer-side lock keeps SPSC invariant under multi-thread misuse
# ---------------------------------------------------------------------------


def test_concurrent_record_evict_from_multiple_threads(tmp_path):
    """N engine threads push to the same evict ring concurrently —
    without the producer lock the SPSC ring would corrupt. With it,
    every record lands and the daemon processes the right count."""
    torch.cuda.init()
    _ = torch.empty(1, device="cuda")
    from gms_kv_ring.engines.handle import GMSKvRing

    sock = str(tmp_path / "d.sock")
    d, t, lh = _spawn_daemon(sock)
    try:
        # Bigger pool so each thread pushes to a different block.
        n_layers = 1
        layer_size = 64 * 1024
        stride = 1024
        dev = torch.zeros(n_layers * layer_size, dtype=torch.uint8, device="cuda")
        torch.cuda.synchronize()
        layers = [
            {
                "layer_idx": 0,
                "va": int(dev.data_ptr()),
                "size": layer_size,
                "stride": stride,
            }
        ]

        eid = f"spsc-{uuid.uuid4().hex[:6]}"
        h = GMSKvRing(
            engine_id=eid,
            daemon_socket=sock,
            layers=layers,
            evict_ring_capacity=4096,
        )
        try:
            N_THREADS = 8
            PER_THREAD = 50
            errors = []

            def producer(tid: int):
                try:
                    for i in range(PER_THREAD):
                        blk = tid * PER_THREAD + i
                        # offset stays inside layer_size
                        offset = (blk * stride) % (layer_size - stride)
                        ok = h.record_evict(
                            block_id=blk,
                            ranges=[(0, stride, offset)],
                        )
                        if not ok:
                            errors.append(f"thread {tid} push #{i} dropped")
                except Exception as e:  # noqa: BLE001
                    errors.append(f"thread {tid}: {type(e).__name__}: {e}")

            ts = [
                threading.Thread(target=producer, args=(i,)) for i in range(N_THREADS)
            ]
            for thr in ts:
                thr.start()
            for thr in ts:
                thr.join(timeout=15)

            assert not errors, f"producer errors: {errors}"

            # Wait for daemon to drain everything. Each push allocates
            # a fresh host_tier slot (different (layer, offset) per blk
            # only when offset wraps; here offset collides every
            # layer_size/stride blocks — that's fine, host_tier reuses
            # the slot).
            expected_total = N_THREADS * PER_THREAD
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                from gms_kv_ring.common import metrics

                if metrics.evict_records.get(engine_id=eid) >= expected_total:
                    break
                time.sleep(0.05)

            from gms_kv_ring.common import metrics

            n_processed = metrics.evict_records.get(engine_id=eid)
            assert n_processed == expected_total, (
                f"expected daemon to consume {expected_total} records, "
                f"got {n_processed} — ring corruption under multi-thread "
                f"push?"
            )
        finally:
            h.close()
    finally:
        lh["loop"].call_soon_threadsafe(d.stop)
        t.join(timeout=3)


# ---------------------------------------------------------------------------
# Integrity: CRC32 on host-tier slots
# ---------------------------------------------------------------------------


def test_checksum_helper_matches_zlib_reference():
    """The common.checksum.crc32_at_ptr function (whichever backend it
    picked — Rust SIMD or zlib fallback) must produce IEEE CRC32
    values matching the stdlib zlib reference. Bit-for-bit identical
    output is what lets the evict + restore sides agree."""
    import ctypes
    import zlib

    from gms_kv_ring.common.checksum import crc32_at_ptr

    payload = b"the quick brown fox jumps over the lazy dog" * 32
    buf = (ctypes.c_ubyte * len(payload)).from_buffer_copy(payload)
    got = crc32_at_ptr(ctypes.addressof(buf), len(payload))
    want = zlib.crc32(payload) & 0xFFFFFFFF
    assert got == want, f"crc32_at_ptr={got:#x} != zlib={want:#x}"

    # Zero-length is well-defined as 0.
    assert crc32_at_ptr(0, 0) == 0


def test_host_tier_slot_records_crc_after_evict_commits(tmp_path):
    """The evict consumer must capture a non-zero CRC for each
    committed slot, computed over the bytes that landed in pinned
    host RAM. Without this, the restore consumer has no way to
    detect mid-flight corruption between evict and restore."""
    import ctypes
    import zlib

    torch.cuda.init()
    _ = torch.empty(1, device="cuda")
    from gms_kv_ring.engines.handle import GMSKvRing

    sock = str(tmp_path / "d.sock")
    d, t, lh = _spawn_daemon(sock)
    try:
        # Deterministic pattern bytes so we can reproduce the CRC
        # from the engine side.
        n_layers = 1
        layer_size = 4096
        stride = 1024
        dev = torch.zeros(n_layers * layer_size, dtype=torch.uint8, device="cuda")
        pattern = torch.arange(stride, dtype=torch.uint8, device="cuda")
        dev[0:stride].copy_(pattern)
        torch.cuda.synchronize()
        layers = [
            {
                "layer_idx": 0,
                "va": int(dev.data_ptr()),
                "size": layer_size,
                "stride": stride,
            }
        ]
        eid = f"crc-ok-{uuid.uuid4().hex[:6]}"
        h = GMSKvRing(engine_id=eid, daemon_socket=sock, layers=layers)
        try:
            res = h.record_evict(block_id=0, ranges=[(0, stride, 0)])
            assert res is not None
            slot_id, target = res

            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                if h.evict_succeeded(slot_id, target):
                    break
                time.sleep(0.02)
            assert h.evict_succeeded(
                slot_id, target
            ), "evict must commit so we can inspect the captured CRC"

            slot = d.host_tier._slots[(eid, 0, 0)]
            assert slot.ready, "slot must be ready after evict_succeeded"
            assert slot.crc != 0, (
                "evict consumer must store a non-zero CRC for committed "
                "slots — otherwise restore has nothing to verify against"
            )

            # Recompute reference CRC from the actual host-tier bytes.
            # The host_ptr is a pinned RAM buffer; we can read it via
            # ctypes. The stored CRC must match what's currently there.
            buf = (ctypes.c_ubyte * slot.size).from_address(slot.host_ptr)
            actual_bytes = bytes(buf)
            expected_crc = zlib.crc32(actual_bytes) & 0xFFFFFFFF
            assert slot.crc == expected_crc, (
                f"stored CRC {slot.crc:#x} != recomputed {expected_crc:#x} "
                "— evict consumer captured a CRC that doesn't match the "
                "bytes it ostensibly committed"
            )
        finally:
            h.close()
    finally:
        lh["loop"].call_soon_threadsafe(d.stop)
        t.join(timeout=3)


def test_restore_signals_failure_on_host_tier_byte_corruption(tmp_path):
    """End-to-end corruption detection: spill block, mutate the
    host-tier bytes (simulating bitrot or a torn DMA the evict path
    didn't catch), then issue a restore. The restore consumer's CRC
    re-check must catch the mismatch and signal failure — the engine
    must NOT consume the corrupted bytes silently."""
    import ctypes

    torch.cuda.init()
    _ = torch.empty(1, device="cuda")
    from gms_kv_ring.common import metrics
    from gms_kv_ring.engines.handle import GMSKvRing

    sock = str(tmp_path / "d.sock")
    d, t, lh = _spawn_daemon(sock)
    try:
        n_layers = 1
        layer_size = 4096
        stride = 1024
        dev = torch.zeros(n_layers * layer_size, dtype=torch.uint8, device="cuda")
        pattern = torch.arange(stride, dtype=torch.uint8, device="cuda")
        dev[0:stride].copy_(pattern)
        torch.cuda.synchronize()
        layers = [
            {
                "layer_idx": 0,
                "va": int(dev.data_ptr()),
                "size": layer_size,
                "stride": stride,
            }
        ]
        eid = f"crc-bad-{uuid.uuid4().hex[:6]}"
        h = GMSKvRing(engine_id=eid, daemon_socket=sock, layers=layers)

        # Snapshot the mismatch counter so we can attribute the bump.
        before = metrics.checksum_mismatches.get(engine_id=eid)
        try:
            # 1) Spill block 0 → host_tier with CRC captured.
            res = h.record_evict(block_id=0, ranges=[(0, stride, 0)])
            assert res is not None
            slot_id, target = res
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline and not h.evict_succeeded(
                slot_id, target
            ):
                time.sleep(0.02)
            assert h.evict_succeeded(slot_id, target)

            # 2) Corrupt the host-tier slot bytes WITHOUT updating its
            # stored CRC. This simulates bitrot / cosmic-ray / DMA
            # tearing the evict path missed.
            slot = d.host_tier._slots[(eid, 0, 0)]
            buf = (ctypes.c_ubyte * slot.size).from_address(slot.host_ptr)
            buf[0] = (buf[0] + 1) & 0xFF  # flip one byte

            # 3) Issue a restore. Source block 0 → dest block 1 (so the
            # H2D destination is the same engine's HBM, different slot).
            r_slot, r_target = h.record_restore(
                src_engine_id=eid,
                block_pairs=[(0, 1)],
            )
            cs = torch.cuda.Stream()
            with torch.cuda.stream(cs):
                h.wait_restore(int(cs.cuda_stream), r_slot, r_target)
            torch.cuda.synchronize()

            # 4) Engine must observe failure — restore_succeeded false.
            assert not h.restore_succeeded(r_slot, r_target), (
                "engine consumed corrupted bytes! restore_succeeded "
                f"must return False on CRC mismatch, "
                f"counter={h.counters.read_slot(r_slot)} "
                f"target={r_target}"
            )

            # 5) checksum_mismatches metric bumped.
            after = metrics.checksum_mismatches.get(engine_id=eid)
            assert after == before + 1, (
                f"checksum_mismatches metric should bump by 1; "
                f"before={before} after={after}"
            )
        finally:
            h.close()
    finally:
        lh["loop"].call_soon_threadsafe(d.stop)
        t.join(timeout=3)
