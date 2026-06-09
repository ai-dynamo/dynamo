# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GMSKvRing handle end-to-end against a real daemon."""

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
    return d, t, loop_holder


def test_handle_evict_and_restore_roundtrip(tmp_path):
    """Build a handle; spill block 0 → restore into block 1; verify
    bytes propagate."""
    torch.cuda.init()
    _ = torch.empty(1, device="cuda")
    from gms_kv_ring.engines.handle import GMSKvRing

    sock = str(tmp_path / "d.sock")
    d, t, lh = _spawn_daemon(sock)
    try:
        n_layers = 2
        layer_size = 4096
        stride = 1024
        dev = torch.zeros(n_layers * layer_size, dtype=torch.uint8, device="cuda")
        base = int(dev.data_ptr())
        host_pattern = torch.arange(n_layers * layer_size, dtype=torch.uint8)
        dev.copy_(host_pattern)
        torch.cuda.synchronize()

        engine_id = f"eng-{uuid.uuid4().hex[:6]}"
        layers = [
            {
                "layer_idx": i,
                "va": base + i * layer_size,
                "size": layer_size,
                "stride": stride,
            }
            for i in range(n_layers)
        ]
        h = GMSKvRing(
            engine_id=engine_id,
            daemon_socket=sock,
            layers=layers,
        )
        try:
            # Spill block 0 (both layers).
            assert h.record_evict(
                block_id=0,
                ranges=[(0, stride, 0), (1, stride, 0)],
            )
            # Wait for daemon to drain D2H.
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline and d.host_tier.n_slots() < 2:
                time.sleep(0.02)
            assert d.host_tier.n_slots() >= 2

            # Restore block 0 → block 1.
            res = h.record_restore(
                src_engine_id=engine_id,
                block_pairs=[(0, 1)],
            )
            assert res is not None
            slot, target = res

            compute_stream = torch.cuda.Stream()
            with torch.cuda.stream(compute_stream):
                h.wait_restore(int(compute_stream.cuda_stream), slot, target)
            torch.cuda.synchronize()

            # Verify bytes.
            buf = dev.cpu().numpy()
            assert bytes(buf[stride : 2 * stride]) == bytes(
                buf[0:stride]
            ), "restore via handle didn't propagate bytes"
        finally:
            h.close()
    finally:
        lh["loop"].call_soon_threadsafe(d.stop)
        t.join(timeout=3)


def test_handle_close_cleans_up_files(tmp_path):
    """After handle.close(), the /dev/shm files are gone."""
    torch.cuda.init()
    _ = torch.empty(1, device="cuda")
    from gms_kv_ring.engines.handle import GMSKvRing

    sock = str(tmp_path / "d.sock")
    d, t, lh = _spawn_daemon(sock)
    try:
        dev = torch.zeros(4096, dtype=torch.uint8, device="cuda")
        layers = [
            {
                "layer_idx": 0,
                "va": int(dev.data_ptr()),
                "size": 4096,
                "stride": 1024,
            }
        ]
        h = GMSKvRing(
            engine_id="cleanup-test",
            daemon_socket=sock,
            layers=layers,
        )
        paths = (h._paths.evict_ring, h._paths.restore_ring, h._paths.counter)
        for p in paths:
            assert os.path.exists(p), f"{p} missing pre-close"
        h.close()
        for p in paths:
            assert not os.path.exists(p), f"{p} not cleaned up"
    finally:
        lh["loop"].call_soon_threadsafe(d.stop)
        t.join(timeout=3)
