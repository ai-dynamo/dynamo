# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the GPU CRC32 kernel + Rust-side combine helper.

The kernel verifies GDS-restored payloads directly in HBM, preserving
the GPU-direct data path. The native GF(2) reduction combines parallel
chunk CRCs without copying payload bytes to host memory.

These tests are CUDA-gated. The kernel falls back gracefully when
NVRTC compile fails — `is_available()` returns False — but in this
host we expect a working CUDA + NVRTC, so we hard-skip on CUDA
absence."""

from __future__ import annotations

import zlib

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("gms_rust_ring")
if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("CUDA required", allow_module_level=True)


@pytest.fixture(scope="module", autouse=True)
def _ensure_cuda_inited():
    torch.cuda.init()
    _ = torch.empty(1, device="cuda")  # warm CUDA context


def test_crc32_combine_chunks_matches_zlib_on_known_data():
    """Pure-Rust math test: combine independently-computed chunk
    CRCs into the CRC of the whole. Verifies the GF(2) matrix
    machinery vs zlib for various chunk counts."""
    from gms_rust_ring import crc32_combine_chunks

    data = bytes((i * 13 & 0xFF) for i in range(12_345))
    expected = zlib.crc32(data) & 0xFFFFFFFF
    for chunk in (1, 7, 256, 4096, len(data), len(data) + 1):
        crcs = []
        for off in range(0, len(data), chunk):
            crcs.append(zlib.crc32(data[off : off + chunk]) & 0xFFFFFFFF)
        got = crc32_combine_chunks(crcs, chunk, len(data))
        assert got == expected, (
            f"combine mismatch with chunk={chunk}: "
            f"expected {expected:#x} got {got:#x}"
        )


def test_crc32_gpu_is_available():
    """On a normal CUDA host, NVRTC compile should succeed and the
    kernel should be ready to use. If it ever does not, the failure
    is logged and GDS restore fails closed. We assert availability
    here as the expected production case."""
    from gms_kv_ring.common.crc32_gpu import is_available

    assert is_available(), "expected NVRTC kernel to compile on a working CUDA install"


@pytest.mark.parametrize(
    "size",
    [
        0,
        1,
        16,
        256,
        4096,
        100_000,
        1_800_000,
    ],
)
def test_crc32_gpu_matches_zlib(size):
    """GPU CRC over a torch CUDA tensor matches zlib.crc32 of the
    same bytes. Spans single byte → ~2MB to exercise both small
    (single-thread useful work) and large (multi-thread) regimes."""
    from gms_kv_ring.common.crc32_gpu import crc32_gpu

    payload = bytes((i * 17 & 0xFF) for i in range(size))
    expected = zlib.crc32(payload) & 0xFFFFFFFF
    if size == 0:
        # GPU pointer doesn't matter for empty; pass 0.
        assert crc32_gpu(0, 0) == expected
        return
    t = torch.frombuffer(
        bytearray(payload),
        dtype=torch.uint8,
    ).cuda()
    got = crc32_gpu(int(t.data_ptr()), size)
    assert got == expected, f"size={size}: expected {expected:#x} got {got:#x}"


def test_crc32_gpu_handles_unaligned_size():
    """Sizes not divisible by chunk_bytes still produce the correct
    CRC. The last thread handles the partial tail."""
    from gms_kv_ring.common.crc32_gpu import crc32_gpu

    # 64 threads × chunk = ceil(n / 64). With n = 100003 prime,
    # chunk = ceil(100003/64) = 1563 bytes; last thread sees 100003
    # - 63*1563 = 1534 bytes (a shorter tail).
    n = 100_003
    payload = bytes((i & 0xFF) for i in range(n))
    expected = zlib.crc32(payload) & 0xFFFFFFFF
    t = torch.frombuffer(bytearray(payload), dtype=torch.uint8).cuda()
    assert crc32_gpu(int(t.data_ptr()), n) == expected


def test_crc32_gpu_detects_device_corruption():
    """A mutation in HBM must not verify against the stored CRC."""
    from gms_kv_ring.common.crc32_gpu import crc32_gpu

    payload = bytearray((i * 29 & 0xFF) for i in range(1_800_003))
    expected = zlib.crc32(payload) & 0xFFFFFFFF
    tensor = torch.frombuffer(payload, dtype=torch.uint8).cuda()
    assert crc32_gpu(int(tensor.data_ptr()), len(payload)) == expected
    tensor[len(payload) // 2] ^= 0x80
    assert crc32_gpu(int(tensor.data_ptr()), len(payload)) != expected
