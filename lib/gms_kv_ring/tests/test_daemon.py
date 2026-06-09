# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Daemon end-to-end: ring push → consumer drains → bytes land in HBM.

Tests use a pre-allocated CUDA buffer as the "engine pool" so we
don't need full VMM-IPC engine integration here. That's the engine
hook's job (C6/C7).
"""

from __future__ import annotations

import asyncio
import os
import queue
import threading
import time
import uuid

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("CUDA required", allow_module_level=True)


def _spawn_daemon(socket_path: str):
    """Run the daemon's asyncio server in a background thread."""
    from gms_kv_ring.daemon.server import Daemon

    d = Daemon(socket_path)

    started = threading.Event()
    loop_holder = {}

    def _runner():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop_holder["loop"] = loop
        loop_holder["daemon"] = d

        async def _main():
            started.set()
            await d.serve()

        try:
            loop.run_until_complete(_main())
        finally:
            loop.close()

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    # Wait for the listen socket to exist.
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not os.path.exists(socket_path):
        time.sleep(0.02)
    assert os.path.exists(socket_path), "daemon socket never appeared"
    return d, t, loop_holder


def test_daemon_attach_and_ping(tmp_path):
    sock = str(tmp_path / "d.sock")
    d, t, lh = _spawn_daemon(sock)
    try:
        from gms_kv_ring.daemon.client import DaemonClient

        c = DaemonClient(sock)
        try:
            # ping was already issued by the constructor; success
            # implies the socket loop is healthy.
            assert c._ok({"op": "ping"})["ok"]
        finally:
            c.close()
    finally:
        # Stop daemon.
        lh["loop"].call_soon_threadsafe(d.stop)
        t.join(timeout=3)


def test_daemon_evict_ring_drains_to_host_tier(tmp_path):
    """Push 4 records into evict ring → daemon consumer pops them and
    does cuMemcpyAsync D2H → bytes land in the host tier."""
    torch.cuda.init()
    _ = torch.empty(1, device="cuda")
    from gms_kv_ring.common.evict_ring import create_ring as create_evict
    from gms_kv_ring.daemon.client import DaemonClient

    sock = str(tmp_path / "d.sock")
    d, t, lh = _spawn_daemon(sock)
    try:
        c = DaemonClient(sock)
        try:
            # Build a pre-allocated CUDA buffer as our "engine pool".
            # 2 layers × 4 KB each.
            n_layers = 2
            layer_size = 4096
            stride = 1024  # 4 blocks per layer
            dev = torch.zeros(n_layers * layer_size, dtype=torch.uint8, device="cuda")
            base = int(dev.data_ptr())
            # Fill with recognizable pattern.
            for i in range(n_layers * layer_size):
                pass  # bytes are 0 from torch.zeros
            # Actually patch some bytes via host-side write through D2H:
            host_src = torch.arange(n_layers * layer_size, dtype=torch.uint8)
            dev.copy_(host_src)
            torch.cuda.synchronize()

            engine_id = f"test-eng-{uuid.uuid4().hex[:6]}"
            layers = [
                {
                    "layer_idx": i,
                    "va": base + i * layer_size,
                    "size": layer_size,
                    "stride": stride,
                }
                for i in range(n_layers)
            ]
            c.attach_engine_pool(engine_id, layers)

            # Create + attach the evict ring.
            ring_path = f"/dev/shm/gms-kvr-evict-{uuid.uuid4().hex}.ring"
            try:
                w = create_evict(ring_path, capacity=64)
                c.attach_evict_ring(engine_id, ring_path)

                # Push 4 spill records (one per block, both layers).
                for blk in range(4):
                    ranges = [(li, stride, blk * stride) for li in range(n_layers)]
                    assert w.push(block_id=blk, ranges=ranges)

                # Wait for daemon to drain + D2H to complete. The
                # daemon's CUDA stream is private; we sync via the
                # default CUDA context.
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    if d.host_tier.n_slots() >= 4 * n_layers:
                        break
                    time.sleep(0.05)
                torch.cuda.synchronize()

                # Verify n_slots and at least one slot's bytes.
                assert (
                    d.host_tier.n_slots() >= 4 * n_layers
                ), f"expected ≥{4*n_layers} slots, got {d.host_tier.n_slots()}"
                slot = d.host_tier.get(engine_id, 0, 0)
                assert slot is not None
                # Read host bytes and compare to what was on device.
                import ctypes

                host_bytes = (ctypes.c_ubyte * stride).from_address(slot.host_ptr)
                # Device bytes [0:stride] should be 0..stride-1 (mod 256).
                expected = bytes((i & 0xFF) for i in range(stride))
                got = bytes(host_bytes)
                assert got == expected, "D2H bytes don't match source"
            finally:
                try:
                    os.unlink(ring_path)
                except OSError:
                    pass
                c.detach_engine_pool(engine_id)
        finally:
            c.close()
    finally:
        lh["loop"].call_soon_threadsafe(d.stop)
        t.join(timeout=3)


def test_daemon_restore_ring_signals_counter(tmp_path):
    """Push a restore record → daemon does (fake) H2D + cuStreamWriteValue32
    → engine cuStreamWaitValue32 releases."""
    torch.cuda.init()
    _ = torch.empty(1, device="cuda")
    from gms_kv_ring.common.counter import EngineCounterArray
    from gms_kv_ring.common.evict_ring import create_ring as create_evict
    from gms_kv_ring.common.restore_ring import create_ring as create_restore
    from gms_kv_ring.daemon.client import DaemonClient

    sock = str(tmp_path / "d.sock")
    d, t, lh = _spawn_daemon(sock)
    try:
        c = DaemonClient(sock)
        try:
            # Pre-allocate device buffer.
            n_layers = 2
            layer_size = 4096
            stride = 1024
            dev = torch.zeros(n_layers * layer_size, dtype=torch.uint8, device="cuda")
            base = int(dev.data_ptr())
            host_pattern = torch.arange(n_layers * layer_size, dtype=torch.uint8)
            dev.copy_(host_pattern)
            torch.cuda.synchronize()

            engine_id = f"test-eng-{uuid.uuid4().hex[:6]}"
            layers = [
                {
                    "layer_idx": i,
                    "va": base + i * layer_size,
                    "size": layer_size,
                    "stride": stride,
                }
                for i in range(n_layers)
            ]
            c.attach_engine_pool(engine_id, layers)

            evict_path = f"/dev/shm/gms-kvr-e-{uuid.uuid4().hex}.ring"
            restore_path = f"/dev/shm/gms-kvr-r-{uuid.uuid4().hex}.ring"
            counter_path = f"/dev/shm/gms-kvr-c-{uuid.uuid4().hex}.bin"
            try:
                # Spill block 0 first so the host tier has its bytes.
                we = create_evict(evict_path, capacity=16)
                c.attach_evict_ring(engine_id, evict_path)
                we.push(block_id=0, ranges=[(0, stride, 0), (1, stride, 0)])
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline and d.host_tier.n_slots() < n_layers:
                    time.sleep(0.05)

                # Now restore block 0 → block 1 (different dest block,
                # same engine).
                wr = create_restore(restore_path, capacity=16)
                eng_ctrs = EngineCounterArray.create(counter_path, num_counters=32)
                try:
                    # Pass counter_host_addr so the same-process daemon
                    # reuses the engine's already-pinned VA — avoids
                    # double-register + leaks ALREADY_REGISTERED never
                    # has to be tolerated.
                    c.attach_restore_ring(
                        engine_id,
                        restore_path,
                        counter_path,
                        32,
                        counter_host_addr=int(eng_ctrs.device_ptr),
                    )

                    slot, target = eng_ctrs.reserve_slot()
                    assert wr.push(
                        src_engine_id=engine_id,
                        block_pairs=[(0, 1)],
                        counter_slot=slot,
                        counter_target=target,
                    )

                    # Engine stream waits on the counter.
                    engine_stream = torch.cuda.Stream()
                    with torch.cuda.stream(engine_stream):
                        eng_ctrs.wait_value32(
                            int(engine_stream.cuda_stream), slot, target
                        )
                    torch.cuda.synchronize()  # would deadlock if daemon never signals

                    # Verify byte correctness: dev[block 1] should now hold
                    # the same bytes as dev[block 0] (which we spilled).
                    buf = dev.cpu().numpy()
                    assert bytes(buf[stride : 2 * stride]) == bytes(
                        buf[0:stride]
                    ), "restore H2D didn't propagate bytes"
                finally:
                    eng_ctrs.close()
            finally:
                for p in (evict_path, restore_path, counter_path):
                    if os.path.exists(p):
                        try:
                            os.unlink(p)
                        except OSError:
                            pass
                c.detach_engine_pool(engine_id)
        finally:
            c.close()
    finally:
        lh["loop"].call_soon_threadsafe(d.stop)
        t.join(timeout=3)


def test_daemon_host_tier_restore_primary_to_shadow(tmp_path):
    """A shadow pool can restore a CPU-mirrored block owned by a
    different attached primary pool. This is the daemon primitive used by
    planned primary→shadow transition before traffic is shifted."""
    torch.cuda.init()
    _ = torch.empty(1, device="cuda")
    from gms_kv_ring.common.counter import EngineCounterArray
    from gms_kv_ring.common.evict_ring import create_ring as create_evict
    from gms_kv_ring.common.restore_ring import create_ring as create_restore
    from gms_kv_ring.daemon.client import DaemonClient

    sock = str(tmp_path / "d.sock")
    d, t, lh = _spawn_daemon(sock)
    try:
        c = DaemonClient(sock)
        try:
            n_layers = 2
            layer_size = 4096
            stride = 1024
            primary_dev = torch.arange(
                n_layers * layer_size, dtype=torch.uint8, device="cuda"
            )
            shadow_dev = torch.zeros(
                n_layers * layer_size, dtype=torch.uint8, device="cuda"
            )
            torch.cuda.synchronize()

            primary_id = f"primary-{uuid.uuid4().hex[:6]}"
            shadow_id = f"shadow-{uuid.uuid4().hex[:6]}"
            primary_base = int(primary_dev.data_ptr())
            shadow_base = int(shadow_dev.data_ptr())
            primary_layers = [
                {
                    "layer_idx": i,
                    "va": primary_base + i * layer_size,
                    "size": layer_size,
                    "stride": stride,
                }
                for i in range(n_layers)
            ]
            shadow_layers = [
                {
                    "layer_idx": i,
                    "va": shadow_base + i * layer_size,
                    "size": layer_size,
                    "stride": stride,
                }
                for i in range(n_layers)
            ]
            c.attach_engine_pool(primary_id, primary_layers)
            c.attach_engine_pool(shadow_id, shadow_layers)

            evict_path = f"/dev/shm/gms-kvr-primary-e-{uuid.uuid4().hex}.ring"
            restore_path = f"/dev/shm/gms-kvr-shadow-r-{uuid.uuid4().hex}.ring"
            counter_path = f"/dev/shm/gms-kvr-shadow-c-{uuid.uuid4().hex}.bin"
            try:
                we = create_evict(evict_path, capacity=16)
                c.attach_evict_ring(primary_id, evict_path)
                assert we.push(
                    block_id=0,
                    ranges=[(layer, stride, 0) for layer in range(n_layers)],
                )
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline and d.host_tier.n_slots() < n_layers:
                    time.sleep(0.05)
                assert d.host_tier.n_slots() >= n_layers

                wr = create_restore(restore_path, capacity=16)
                ctrs = EngineCounterArray.create(counter_path, num_counters=32)
                try:
                    c.attach_restore_ring(
                        shadow_id,
                        restore_path,
                        counter_path,
                        32,
                        counter_host_addr=int(ctrs.device_ptr),
                    )
                    slot, target = ctrs.reserve_slot()
                    assert wr.push(
                        src_engine_id=primary_id,
                        block_pairs=[(0, 1)],
                        counter_slot=slot,
                        counter_target=target,
                    )
                    engine_stream = torch.cuda.Stream()
                    with torch.cuda.stream(engine_stream):
                        ctrs.wait_value32(int(engine_stream.cuda_stream), slot, target)
                    torch.cuda.synchronize()
                    assert ctrs.read_slot(slot) == target

                    src = primary_dev.cpu().numpy()
                    dst = shadow_dev.cpu().numpy()
                    for layer in range(n_layers):
                        src_off = layer * layer_size
                        dst_off = layer * layer_size + stride
                        assert bytes(dst[dst_off : dst_off + stride]) == bytes(
                            src[src_off : src_off + stride]
                        )
                finally:
                    ctrs.close()
            finally:
                for path in (evict_path, restore_path, counter_path):
                    if os.path.exists(path):
                        try:
                            os.unlink(path)
                        except OSError:
                            pass
                c.detach_engine_pool(shadow_id)
                c.detach_engine_pool(primary_id)
        finally:
            c.close()
    finally:
        lh["loop"].call_soon_threadsafe(d.stop)
        t.join(timeout=3)


def test_daemon_host_tier_shadow_serves_cpu_kv_under_sustained_load(tmp_path):
    """Primary keeps mirroring KV into CPU RAM while a shadow serves a
    sustained stream of restore requests from that CPU mirror.

    The producer poisons/reuses the primary HBM slot immediately after each
    host-tier mirror. The shadow must still observe the pre-poison bytes, so a
    passing test proves the served request came from CPU KV, not primary HBM.
    """
    torch.cuda.init()
    _ = torch.empty(1, device="cuda")
    from gms_kv_ring.engines.gds_block_connector import BlockGdsConnector
    from gms_kv_ring.engines.handle import GMSKvRing

    sock = str(tmp_path / "d.sock")
    d, t, lh = _spawn_daemon(sock)
    primary_handle = None
    shadow_handle = None
    try:
        n_layers = 3
        num_blocks = 16
        stride = 512
        layer_size = num_blocks * stride
        iterations = 128
        primary_dev = torch.zeros(
            n_layers * layer_size, dtype=torch.uint8, device="cuda"
        )
        shadow_dev = torch.zeros(
            n_layers * layer_size, dtype=torch.uint8, device="cuda"
        )
        primary_id = f"primary-live-{uuid.uuid4().hex[:6]}"
        shadow_id = f"shadow-live-{uuid.uuid4().hex[:6]}"
        primary_base = int(primary_dev.data_ptr())
        shadow_base = int(shadow_dev.data_ptr())
        primary_layers = [
            {
                "layer_idx": layer,
                "va": primary_base + layer * layer_size,
                "size": layer_size,
                "stride": stride,
            }
            for layer in range(n_layers)
        ]
        shadow_layers = [
            {
                "layer_idx": layer,
                "va": shadow_base + layer * layer_size,
                "size": layer_size,
                "stride": stride,
            }
            for layer in range(n_layers)
        ]

        primary_handle = GMSKvRing(
            engine_id=primary_id,
            daemon_socket=sock,
            layers=primary_layers,
            evict_ring_capacity=256,
            restore_ring_capacity=256,
            num_counters=256,
        )
        shadow_handle = GMSKvRing(
            engine_id=shadow_id,
            daemon_socket=sock,
            layers=shadow_layers,
            evict_ring_capacity=256,
            restore_ring_capacity=256,
            num_counters=256,
        )

        def layout(block_id):
            return [
                (layer, int(block_id) * stride, stride) for layer in range(n_layers)
            ]

        primary = BlockGdsConnector(primary_handle, layout)
        shadow = BlockGdsConnector(shadow_handle, layout)
        work: "queue.Queue[tuple[int, int, int, int] | None]" = queue.Queue(
            maxsize=num_blocks // 2,
        )
        errors: list[BaseException] = []
        served: list[int] = []

        def fill_block(dev, block_id: int, value: int) -> None:
            for layer in range(n_layers):
                off = layer * layer_size + int(block_id) * stride
                dev[off : off + stride].fill_(int(value) & 0xFF)
            torch.cuda.synchronize()

        def expect_shadow_block(block_id: int, value: int) -> None:
            torch.cuda.synchronize()
            data = shadow_dev.cpu().numpy()
            expected = bytes([int(value) & 0xFF]) * stride
            for layer in range(n_layers):
                off = layer * layer_size + int(block_id) * stride
                got = bytes(data[off : off + stride])
                assert got == expected, (
                    f"shadow block={block_id} layer={layer} served stale/corrupt "
                    f"bytes got={got[:8].hex()} expected={expected[:8].hex()}"
                )

        def producer() -> None:
            try:
                for i in range(iterations):
                    src = i % num_blocks
                    dst = (i * 5 + 3) % num_blocks
                    gen = i + 1
                    value = (i * 29 + 7) % 251 + 1
                    poison = value ^ 0xFF

                    fill_block(primary_dev, src, value)
                    evicted = primary.evict_blocks_to_host(
                        [src],
                        generations={src: gen},
                        timeout_s=10.0,
                    )
                    assert evicted.failed == 0, evicted

                    # Simulate continued primary traffic immediately reusing the
                    # HBM block. Shadow correctness must come from the CPU mirror.
                    fill_block(primary_dev, src, poison)
                    work.put((src, dst, gen, value), timeout=10.0)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
            finally:
                work.put(None, timeout=10.0)

        def consumer() -> None:
            try:
                while True:
                    item = work.get(timeout=15.0)
                    if item is None:
                        return
                    src, dst, gen, value = item
                    restored = shadow.restore_blocks_remap_from_host(
                        primary_id,
                        [(src, dst, gen)],
                        timeout_s=10.0,
                    )
                    assert restored == {dst: True}
                    expect_shadow_block(dst, value)
                    served.append(gen)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        producer_thread = threading.Thread(target=producer, name="gms-primary-load")
        consumer_thread = threading.Thread(target=consumer, name="gms-shadow-serve")
        producer_thread.start()
        consumer_thread.start()
        producer_thread.join(timeout=60.0)
        consumer_thread.join(timeout=60.0)

        assert not producer_thread.is_alive(), "primary load thread hung"
        assert not consumer_thread.is_alive(), "shadow serve thread hung"
        if errors:
            raise AssertionError("transition stress thread failed") from errors[0]
        assert len(served) == iterations
        assert served == list(range(1, iterations + 1))
        assert d.host_tier.n_slots() >= n_layers * min(num_blocks, iterations)
    finally:
        if shadow_handle is not None:
            shadow_handle.close()
        if primary_handle is not None:
            primary_handle.close()
        lh["loop"].call_soon_threadsafe(d.stop)
        t.join(timeout=3)


def test_daemon_host_tier_primary_shadow_generation_stress(tmp_path):
    """Sustained planned-transition primitive: primary mirrors KV into
    CPU RAM while shadow repeatedly restores from that CPU copy. Reusing the
    same source block with a newer generation must make older restore claims
    fail closed, and overwriting primary HBM after the mirror must not change
    what shadow receives from CPU RAM.
    """
    torch.cuda.init()
    _ = torch.empty(1, device="cuda")
    from gms_kv_ring.engines.gds_block_connector import BlockGdsConnector
    from gms_kv_ring.engines.handle import GMSKvRing

    sock = str(tmp_path / "d.sock")
    d, t, lh = _spawn_daemon(sock)
    primary_handle = None
    shadow_handle = None
    try:
        n_layers = 2
        num_blocks = 4
        stride = 256
        layer_size = num_blocks * stride
        primary_dev = torch.zeros(
            n_layers * layer_size, dtype=torch.uint8, device="cuda"
        )
        shadow_dev = torch.zeros(
            n_layers * layer_size, dtype=torch.uint8, device="cuda"
        )
        primary_id = f"primary-stress-{uuid.uuid4().hex[:6]}"
        shadow_id = f"shadow-stress-{uuid.uuid4().hex[:6]}"
        primary_base = int(primary_dev.data_ptr())
        shadow_base = int(shadow_dev.data_ptr())
        primary_layers = [
            {
                "layer_idx": i,
                "va": primary_base + i * layer_size,
                "size": layer_size,
                "stride": stride,
            }
            for i in range(n_layers)
        ]
        shadow_layers = [
            {
                "layer_idx": i,
                "va": shadow_base + i * layer_size,
                "size": layer_size,
                "stride": stride,
            }
            for i in range(n_layers)
        ]

        primary_handle = GMSKvRing(
            engine_id=primary_id,
            daemon_socket=sock,
            layers=primary_layers,
            evict_ring_capacity=64,
            restore_ring_capacity=64,
            num_counters=128,
        )
        shadow_handle = GMSKvRing(
            engine_id=shadow_id,
            daemon_socket=sock,
            layers=shadow_layers,
            evict_ring_capacity=64,
            restore_ring_capacity=64,
            num_counters=128,
        )

        def layout(block_id):
            return [
                (layer, int(block_id) * stride, stride) for layer in range(n_layers)
            ]

        primary = BlockGdsConnector(primary_handle, layout)
        shadow = BlockGdsConnector(shadow_handle, layout)

        def fill_block(dev, block_id: int, value: int) -> None:
            for layer in range(n_layers):
                off = layer * layer_size + int(block_id) * stride
                dev[off : off + stride].fill_(int(value) & 0xFF)
            torch.cuda.synchronize()

        def expect_shadow_block(block_id: int, value: int) -> None:
            data = shadow_dev.cpu().numpy()
            expected = bytes([int(value) & 0xFF]) * stride
            for layer in range(n_layers):
                off = layer * layer_size + int(block_id) * stride
                assert bytes(data[off : off + stride]) == expected

        last_gen_by_src: dict[int, int] = {}
        iterations = 64
        for i in range(iterations):
            src = i % num_blocks
            dst = (i * 3 + 1) % num_blocks
            gen = i + 1
            value = (i * 17 + 11) % 251 + 1
            poison = value ^ 0xFF

            fill_block(primary_dev, src, value)
            evicted = primary.evict_blocks_to_host(
                [src],
                generations={src: gen},
                timeout_s=10.0,
            )
            assert evicted.failed == 0

            # Reusing the primary HBM slot after the CPU mirror must not alter
            # the shadow restore; shadow reads from pinned CPU RAM.
            fill_block(primary_dev, src, poison)

            stale_gen = last_gen_by_src.get(src)
            if stale_gen is not None:
                stale = shadow.restore_blocks_remap_from_host(
                    primary_id,
                    [(src, dst, stale_gen)],
                    timeout_s=10.0,
                )
                assert stale == {dst: False}

            restored = shadow.restore_blocks_remap_from_host(
                primary_id,
                [(src, dst, gen)],
                timeout_s=10.0,
            )
            assert restored == {dst: True}
            expect_shadow_block(dst, value)
            last_gen_by_src[src] = gen

        assert d.host_tier.n_slots() >= n_layers
    finally:
        if shadow_handle is not None:
            shadow_handle.close()
        if primary_handle is not None:
            primary_handle.close()
        lh["loop"].call_soon_threadsafe(d.stop)
        t.join(timeout=3)
