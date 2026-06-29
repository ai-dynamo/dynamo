# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the SGLang GDS-direct connector glue.

`SGLangGdsConnector` is an alias for the engine-agnostic
`BlockGdsConnector`; what these tests exercise is the SGLang-
flavored layout convention (page_id × n_layers → per-page-per-
layer ranges) and that the existing logic handles it correctly.

The byte layouts differ from the vLLM tests:
  - vLLM: contiguous per-layer regions, block_id maps directly to
    a contiguous (offset = block_id * block_bytes, size =
    block_bytes) range per layer.
  - SGLang: per-page slots. A "page" here = n_tokens × bytes_per_token
    per layer. We still get one (layer, offset, size) range per
    layer per page — the connector is structurally identical."""

from __future__ import annotations

import uuid

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("CUDA required", allow_module_level=True)

from tests.test_demote_hbm_to_storage import (  # noqa: E402
    _FakeGpuDirectBackend,
    _spawn_daemon,
)


def _build_sglang_handle(
    sock, *, n_layers=3, n_pages=6, tokens_per_page=8, bytes_per_token=16
):
    """SGLang-shaped HBM layout: per-layer regions, each holding
    `n_pages * tokens_per_page * bytes_per_token` bytes. Returns
    (handle, dev_tensor, page_layout_fn, dims)."""
    from gms_kv_ring.engines.handle import GMSKvRing

    page_bytes = tokens_per_page * bytes_per_token
    layer_stride = n_pages * page_bytes
    total = n_layers * layer_stride
    dev = torch.zeros(total, dtype=torch.uint8, device="cuda")
    torch.cuda.synchronize()
    base = int(dev.data_ptr())
    layers = [
        {
            "layer_idx": L,
            "va": base + L * layer_stride,
            "size": layer_stride,
            "stride": page_bytes,
        }
        for L in range(n_layers)
    ]
    eid = f"sgl-{uuid.uuid4().hex[:6]}"
    handle = GMSKvRing(
        engine_id=eid,
        daemon_socket=sock,
        layers=layers,
    )

    def page_layout(page_id):
        # One range per layer; offset within layer = page_id * page_bytes.
        return [(L, page_id * page_bytes, page_bytes) for L in range(n_layers)]

    dims = {
        "n_layers": n_layers,
        "n_pages": n_pages,
        "page_bytes": page_bytes,
        "layer_stride": layer_stride,
    }
    return handle, dev, page_layout, dims


def test_sglang_gds_connector_is_available(tmp_path):
    from gms_kv_ring.engines.sglang.gds_connector import SGLangGdsConnector

    fake = _FakeGpuDirectBackend()
    sock = str(tmp_path / "d.sock")
    d, t, lh = _spawn_daemon(sock, storage_backend=fake)
    try:
        h, _dev, layout, _ = _build_sglang_handle(sock)
        try:
            assert SGLangGdsConnector(h, layout).is_available() is True
        finally:
            h.close()
    finally:
        lh["loop"].call_soon_threadsafe(d.stop)
        t.join(timeout=3)


def test_sglang_gds_connector_page_round_trip(tmp_path):
    """Spill 2 SGLang pages, zero HBM, restore them. Verify each
    per-layer slice round-tripped correctly. Catches off-by-one
    bugs in the page_id × n_layers → (layer, offset, size) mapping."""
    from gms_kv_ring.engines.sglang.gds_connector import SGLangGdsConnector

    fake = _FakeGpuDirectBackend()
    sock = str(tmp_path / "d.sock")
    d, t, lh = _spawn_daemon(sock, storage_backend=fake)
    try:
        h, dev, layout, dims = _build_sglang_handle(sock)
        n_layers = dims["n_layers"]
        page_bytes = dims["page_bytes"]
        layer_stride = dims["layer_stride"]
        try:
            # Pattern in pages 1 and 4.
            patterns = {}
            for pid in (1, 4):
                for L in range(n_layers):
                    start = L * layer_stride + pid * page_bytes
                    pat = bytes(
                        ((i * (L + 3)) ^ (pid * 7)) & 0xFF for i in range(page_bytes)
                    )
                    patterns[(pid, L)] = pat
                    dev[start : start + page_bytes].copy_(
                        torch.frombuffer(bytearray(pat), dtype=torch.uint8)
                    )
            torch.cuda.synchronize()

            conn = SGLangGdsConnector(h, layout)
            assert conn.is_available()
            res = conn.evict_blocks_to_storage([1, 4])
            assert res.succeeded == 2
            assert res.failed == 0

            # Zero so restore must rewrite.
            dev.zero_()
            torch.cuda.synchronize()

            out = conn.restore_blocks_from_storage([1, 4])
            assert out == {1: True, 4: True}

            for pid in (1, 4):
                for L in range(n_layers):
                    start = L * layer_stride + pid * page_bytes
                    got = bytes(dev[start : start + page_bytes].cpu().numpy().tobytes())
                    assert got == patterns[(pid, L)], f"mismatch page={pid} layer={L}"

            # Pages NOT in the evict list stay zero (no spurious writes).
            for pid in (0, 2, 3, 5):
                for L in range(n_layers):
                    start = L * layer_stride + pid * page_bytes
                    region = dev[start : start + page_bytes].cpu().numpy()
                    assert region.tobytes() == b"\x00" * page_bytes
        finally:
            h.close()
    finally:
        lh["loop"].call_soon_threadsafe(d.stop)
        t.join(timeout=3)


def test_sglang_gds_connector_reexport_is_same_class():
    """Confirm SGLangGdsConnector and VllmGdsConnector are the same
    underlying class — they're aliases for the engine-agnostic
    BlockGdsConnector. This pins the contract so a refactor that
    splits them later is an intentional break."""
    from gms_kv_ring.engines.gds_block_connector import BlockGdsConnector
    from gms_kv_ring.engines.sglang.gds_connector import SGLangGdsConnector
    from gms_kv_ring.engines.vllm.gds_connector import VllmGdsConnector

    assert SGLangGdsConnector is BlockGdsConnector
    assert VllmGdsConnector is BlockGdsConnector
    assert SGLangGdsConnector is VllmGdsConnector
