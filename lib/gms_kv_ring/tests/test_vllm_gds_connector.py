# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the vLLM GDS-direct connector glue.

The connector wraps GMSKvRing's layer-centric API behind a
block-id-centric interface that matches vLLM's KVCacheConnectorV1
hook shape. Tests use the same `_FakeGpuDirectBackend` from
test_demote_hbm_to_storage to exercise the routing without
needing real vLLM installed."""

from __future__ import annotations

import uuid

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("CUDA required", allow_module_level=True)


# Reuse the fixture + fake backend from the demote_hbm_to_storage suite.
from tests.test_demote_hbm_to_storage import (  # noqa: E402
    _FakeGpuDirectBackend,
    _spawn_daemon,
)


def _make_handle_and_dev(sock, *, n_layers=4, block_bytes=1024, n_blocks=8):
    """Allocate `n_layers` × `n_blocks × block_bytes` HBM and
    return (GMSKvRing, dev_tensor, layout_fn).

    Layout: contiguous per-layer regions; block_id b at layer L
    starts at `base_va + L*layer_stride + b*block_bytes`."""
    from gms_kv_ring.engines.handle import GMSKvRing

    layer_stride = n_blocks * block_bytes
    total = n_layers * layer_stride
    dev = torch.zeros(total, dtype=torch.uint8, device="cuda")
    torch.cuda.synchronize()
    base = int(dev.data_ptr())
    layers = [
        {
            "layer_idx": L,
            "va": base + L * layer_stride,
            "size": layer_stride,
            "stride": block_bytes,
        }
        for L in range(n_layers)
    ]
    eid = f"v-{uuid.uuid4().hex[:6]}"
    handle = GMSKvRing(
        engine_id=eid,
        daemon_socket=sock,
        layers=layers,
    )

    def layout(block_id):
        return [(L, block_id * block_bytes, block_bytes) for L in range(n_layers)]

    return handle, dev, layout, n_layers, block_bytes


def test_vllm_gds_connector_is_available_reflects_backend(tmp_path):
    from gms_kv_ring.engines.vllm.gds_connector import VllmGdsConnector

    # GPU-direct backend → True.
    fake = _FakeGpuDirectBackend()
    sock = str(tmp_path / "d-gpu.sock")
    d, t, lh = _spawn_daemon(sock, storage_backend=fake)
    try:
        h, _dev, layout, _nL, _bs = _make_handle_and_dev(sock)
        try:
            assert VllmGdsConnector(h, layout).is_available() is True
        finally:
            h.close()
    finally:
        lh["loop"].call_soon_threadsafe(d.stop)
        t.join(timeout=3)

    # Default StorageTier (no GPU-direct) → False.
    sock2 = str(tmp_path / "d-local.sock")
    d2, t2, lh2 = _spawn_daemon(sock2)
    try:
        h2, _dev2, layout2, _, _ = _make_handle_and_dev(sock2)
        try:
            assert VllmGdsConnector(h2, layout2).is_available() is False
        finally:
            h2.close()
    finally:
        lh2["loop"].call_soon_threadsafe(d2.stop)
        t2.join(timeout=3)


def test_vllm_gds_connector_evict_blocks_to_storage(tmp_path):
    """evict_blocks_to_storage calls demote_hbm_to_storage for every
    (layer, offset, size) of every block. With all layers succeeding,
    the EvictResult reports all blocks as succeeded."""
    from gms_kv_ring.engines.vllm.gds_connector import VllmGdsConnector

    fake = _FakeGpuDirectBackend()
    sock = str(tmp_path / "d.sock")
    d, t, lh = _spawn_daemon(sock, storage_backend=fake)
    try:
        h, dev, layout, n_layers, block_bytes = _make_handle_and_dev(sock)
        try:
            # Pre-fill HBM with a per-block-per-layer pattern so we
            # can assert each demote saw the right bytes.
            for L in range(n_layers):
                for b in range(8):
                    start = L * 8 * block_bytes + b * block_bytes
                    pat = bytes(
                        (i * (L + 1) * (b + 1) & 0xFF) for i in range(block_bytes)
                    )
                    dev[start : start + block_bytes].copy_(
                        torch.frombuffer(bytearray(pat), dtype=torch.uint8)
                    )
            torch.cuda.synchronize()

            conn = VllmGdsConnector(h, layout)
            res = conn.evict_blocks_to_storage([0, 3, 7])
            assert res.succeeded == 3
            assert res.failed == 0
            assert res.failed_ids == []

            # 3 blocks × n_layers demote_from_gpu calls landed.
            assert len(fake.demote_calls) == 3 * n_layers

            # Spot-check one block's bytes round-tripped correctly.
            expected_block0_layer0 = bytes(
                (i * 1 * 1 & 0xFF) for i in range(block_bytes)
            )
            calls_block0_layer0 = [
                c
                for c in fake.demote_calls
                if c["engine_id"] == h.engine_id
                and c["layer"] == 0
                and c["offset"] == 0
            ]
            assert len(calls_block0_layer0) == 1
            assert calls_block0_layer0[0]["observed"] == expected_block0_layer0

            # host_tier untouched.
            assert d.host_tier.n_slots() == 0
        finally:
            h.close()
    finally:
        lh["loop"].call_soon_threadsafe(d.stop)
        t.join(timeout=3)


def test_vllm_gds_connector_restore_blocks_from_storage(tmp_path):
    """Round-trip: evict 2 blocks, zero HBM, restore them via the
    connector, verify the HBM bytes round-tripped layer-by-layer."""
    from gms_kv_ring.engines.vllm.gds_connector import VllmGdsConnector

    fake = _FakeGpuDirectBackend()
    sock = str(tmp_path / "d.sock")
    d, t, lh = _spawn_daemon(sock, storage_backend=fake)
    try:
        h, dev, layout, n_layers, block_bytes = _make_handle_and_dev(sock)
        try:
            # Pattern in blocks 0 and 5.
            patterns = {}
            for b in (0, 5):
                for L in range(n_layers):
                    start = L * 8 * block_bytes + b * block_bytes
                    pat = bytes(
                        ((i * (L + 1)) ^ (b * 11)) & 0xFF for i in range(block_bytes)
                    )
                    patterns[(b, L)] = pat
                    dev[start : start + block_bytes].copy_(
                        torch.frombuffer(bytearray(pat), dtype=torch.uint8)
                    )
            torch.cuda.synchronize()

            conn = VllmGdsConnector(h, layout)
            res = conn.evict_blocks_to_storage([0, 5])
            assert res.succeeded == 2

            # Zero HBM so restore must rewrite the bytes.
            dev.zero_()
            torch.cuda.synchronize()

            out = conn.restore_blocks_from_storage([0, 5])
            assert out == {0: True, 5: True}

            # Verify HBM has the expected per-layer bytes back.
            for b in (0, 5):
                for L in range(n_layers):
                    start = L * 8 * block_bytes + b * block_bytes
                    got = bytes(
                        dev[start : start + block_bytes].cpu().numpy().tobytes()
                    )
                    assert got == patterns[(b, L)], f"mismatch at block={b} layer={L}"
        finally:
            h.close()
    finally:
        lh["loop"].call_soon_threadsafe(d.stop)
        t.join(timeout=3)


def test_vllm_gds_connector_restore_returns_false_for_missing(tmp_path):
    """Calling restore for a block that was never evicted returns
    False — engine falls to cold compute. Other blocks in the same
    call are unaffected."""
    from gms_kv_ring.engines.vllm.gds_connector import VllmGdsConnector

    fake = _FakeGpuDirectBackend()
    sock = str(tmp_path / "d.sock")
    d, t, lh = _spawn_daemon(sock, storage_backend=fake)
    try:
        h, _dev, layout, _, block_bytes = _make_handle_and_dev(sock)
        try:
            conn = VllmGdsConnector(h, layout)
            # Evict only block 2.
            conn.evict_blocks_to_storage([2])
            # Restore 2 (should work) + 99 (never evicted, should fail).
            out = conn.restore_blocks_from_storage([2, 99])
            assert out[2] is True
            assert out[99] is False
        finally:
            h.close()
    finally:
        lh["loop"].call_soon_threadsafe(d.stop)
        t.join(timeout=3)


def test_vllm_gds_connector_evict_partial_layer_failure_is_failed(
    tmp_path,
    monkeypatch,
):
    """If even one layer of a block fails to demote, the block is
    reported as failed (engine should retry or fall back). Other
    blocks in the same call are unaffected."""
    from gms_kv_ring.engines.vllm.gds_connector import VllmGdsConnector

    fake = _FakeGpuDirectBackend()
    sock = str(tmp_path / "d.sock")
    d, t, lh = _spawn_daemon(sock, storage_backend=fake)
    try:
        h, _dev, layout, n_layers, _bs = _make_handle_and_dev(sock)
        try:
            conn = VllmGdsConnector(h, layout)
            # Patch the handle to fail on layer 2 specifically.
            orig = h.demote_hbm_to_storage

            def selective(layer, offset, size):
                if layer == 2:
                    return False
                return orig(layer, offset, size)

            monkeypatch.setattr(h, "demote_hbm_to_storage", selective)

            res = conn.evict_blocks_to_storage([0, 1])
            assert res.succeeded == 0
            assert res.failed == 2
            assert sorted(res.failed_ids) == [0, 1]
        finally:
            h.close()
    finally:
        lh["loop"].call_soon_threadsafe(d.stop)
        t.join(timeout=3)
