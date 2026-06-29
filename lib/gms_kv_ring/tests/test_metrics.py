# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""metrics: counter primitives + Prometheus render + HTTP scrape +
end-to-end (daemon consumes records → counters tick)."""

from __future__ import annotations

import asyncio
import http.client
import os
import threading
import time
import uuid

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("CUDA required", allow_module_level=True)


def test_counter_basic_inc_and_render():
    from gms_kv_ring.common.metrics import Counter

    c = Counter("test_unlabeled", "help text")
    c.inc()
    c.inc(n=5)
    assert c.get() == 6
    txt = c.render()
    assert "# HELP test_unlabeled help text" in txt
    assert "# TYPE test_unlabeled counter" in txt
    assert "test_unlabeled 6" in txt

    labeled = Counter("test_labeled", "h", label_name="engine_id")
    labeled.inc(engine_id="A")
    labeled.inc(engine_id="A", n=2)
    labeled.inc(engine_id="B")
    assert labeled.get(engine_id="A") == 3
    assert labeled.get(engine_id="B") == 1
    rendered = labeled.render()
    assert 'test_labeled{engine_id="A"} 3' in rendered
    assert 'test_labeled{engine_id="B"} 1' in rendered


def test_counter_label_validation():
    from gms_kv_ring.common.metrics import Counter

    c = Counter("foo", "h", label_name="engine_id")
    with pytest.raises(ValueError):
        c.inc()  # missing label
    with pytest.raises(ValueError):
        c.inc(wrong_label="x")
    unlabeled = Counter("bar", "h")
    with pytest.raises(ValueError):
        unlabeled.inc(engine_id="x")


def test_http_scrape_returns_prometheus_text():
    from gms_kv_ring.common import metrics

    server = metrics.serve_http("127.0.0.1", 0)
    try:
        port = server.server_address[1]
        metrics.evict_records.inc(engine_id="scrape-test")
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        conn.request("GET", "/metrics")
        resp = conn.getresponse()
        assert resp.status == 200
        body = resp.read().decode("utf-8")
        assert 'gms_kvr_evict_records_total{engine_id="scrape-test"}' in body
        # 404 path:
        conn.request("GET", "/nope")
        resp2 = conn.getresponse()
        assert resp2.status == 404
    finally:
        server.shutdown()
        server.server_close()


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


def test_daemon_consumers_tick_counters(tmp_path):
    """Push 4 spills + 4 restores; verify evict_records / restore_records
    counters tick to 4 each for this engine."""
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
        dev.copy_(torch.arange(n_layers * layer_size, dtype=torch.uint8))
        torch.cuda.synchronize()
        layers = [
            {
                "layer_idx": 0,
                "va": int(dev.data_ptr()),
                "size": layer_size,
                "stride": stride,
            }
        ]

        eid = f"metrics-{uuid.uuid4().hex[:6]}"
        before_e = metrics.evict_records.get(engine_id=eid)
        before_r = metrics.restore_records.get(engine_id=eid)
        before_b = metrics.evict_d2h_bytes.get(engine_id=eid)

        h = GMSKvRing(engine_id=eid, daemon_socket=sock, layers=layers)
        try:
            # 4 spills.
            for blk in range(4):
                assert h.record_evict(
                    block_id=blk,
                    ranges=[(0, stride, blk * stride)],
                )
            deadline = time.monotonic() + 3
            while (
                time.monotonic() < deadline
                and metrics.evict_records.get(engine_id=eid) < before_e + 4
            ):
                time.sleep(0.02)
            assert metrics.evict_records.get(engine_id=eid) == before_e + 4
            assert metrics.evict_d2h_bytes.get(engine_id=eid) >= before_b + 4 * stride

            # 4 restores.
            cs = torch.cuda.Stream()
            for blk in range(4):
                res = h.record_restore(
                    src_engine_id=eid,
                    block_pairs=[(blk, blk)],
                )
                assert res is not None
                slot, target = res
                with torch.cuda.stream(cs):
                    h.wait_restore(int(cs.cuda_stream), slot, target)
            torch.cuda.synchronize()

            deadline = time.monotonic() + 3
            while (
                time.monotonic() < deadline
                and metrics.restore_records.get(engine_id=eid) < before_r + 4
            ):
                time.sleep(0.02)
            assert metrics.restore_records.get(engine_id=eid) == before_r + 4

            # Prometheus render contains this engine's labels.
            text = metrics.render_prometheus()
            assert f'engine_id="{eid}"' in text
        finally:
            h.close()
    finally:
        lh["loop"].call_soon_threadsafe(d.stop)
        t.join(timeout=3)
