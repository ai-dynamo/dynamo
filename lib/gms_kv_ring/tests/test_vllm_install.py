# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""install_for_vllm round-trip + on_evict/on_hit callback wiring.

We don't import vLLM here — the install entry point is engine-agnostic.
The shape mirrors test_engine_handle.py but adds callback assertions
specific to the vLLM adapter contract."""

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


def test_install_for_vllm_callbacks_fire(tmp_path):
    """on_evict / on_hit must each be invoked exactly once with the
    handle, before install_for_vllm returns."""
    torch.cuda.init()
    _ = torch.empty(1, device="cuda")
    from gms_kv_ring.engines.vllm import install_for_vllm

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
        seen = {"evict": 0, "hit": 0}

        def _on_evict(h):
            seen["evict"] += 1
            assert hasattr(h, "record_evict")

        def _on_hit(h):
            seen["hit"] += 1
            assert hasattr(h, "record_restore")
            assert hasattr(h, "wait_restore")

        h = install_for_vllm(
            engine_id="vllm-cb-test",
            daemon_socket=sock,
            layers=layers,
            on_evict=_on_evict,
            on_hit=_on_hit,
        )
        try:
            assert seen == {"evict": 1, "hit": 1}
        finally:
            h.close()
    finally:
        lh["loop"].call_soon_threadsafe(d.stop)
        t.join(timeout=3)


def test_install_for_vllm_roundtrip(tmp_path):
    """Full evict → restore → byte-correctness through the vLLM entry point."""
    torch.cuda.init()
    _ = torch.empty(1, device="cuda")
    from gms_kv_ring.engines.vllm import install_for_vllm

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

        engine_id = f"vllm-{uuid.uuid4().hex[:6]}"
        layers = [
            {
                "layer_idx": i,
                "va": base + i * layer_size,
                "size": layer_size,
                "stride": stride,
            }
            for i in range(n_layers)
        ]
        h = install_for_vllm(
            engine_id=engine_id,
            daemon_socket=sock,
            layers=layers,
        )
        try:
            assert h.record_evict(
                block_id=0,
                ranges=[(0, stride, 0), (1, stride, 0)],
            )
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline and d.host_tier.n_slots() < 2:
                time.sleep(0.02)
            assert d.host_tier.n_slots() >= 2

            res = h.record_restore(
                src_engine_id=engine_id,
                block_pairs=[(0, 1)],
            )
            assert res is not None
            slot, target = res
            cs = torch.cuda.Stream()
            with torch.cuda.stream(cs):
                h.wait_restore(int(cs.cuda_stream), slot, target)
            torch.cuda.synchronize()

            buf = dev.cpu().numpy()
            assert bytes(buf[stride : 2 * stride]) == bytes(buf[0:stride])
        finally:
            h.close()
    finally:
        lh["loop"].call_soon_threadsafe(d.stop)
        t.join(timeout=3)
