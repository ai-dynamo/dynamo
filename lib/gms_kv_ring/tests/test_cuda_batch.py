# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cuMemcpyBatchAsync correctness + speedup vs loop."""

from __future__ import annotations

import time

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("CUDA required", allow_module_level=True)


def test_batch_h2d_byte_correctness():
    """Bytes batched H2D'd must equal what a loop would produce."""
    import ctypes

    torch.cuda.init()
    _ = torch.empty(1, device="cuda")
    from cuda.bindings import driver as drv
    from cuda.bindings import runtime as rt
    from gms_kv_ring.common.cuda_batch import batch_h2d, has_batch_h2d

    if not has_batch_h2d():
        pytest.skip("cuMemcpyBatchAsync not available")

    N, SZ = 16, 4096
    # Pinned host source with a recognizable per-region pattern.
    err, host_ptr = rt.cudaHostAlloc(N * SZ, 0)
    assert err == rt.cudaError_t.cudaSuccess
    try:
        host_bytes = (ctypes.c_ubyte * (N * SZ)).from_address(int(host_ptr))
        for i in range(N):
            for j in range(SZ):
                host_bytes[i * SZ + j] = (i * 7 + j * 13) & 0xFF

        dev = torch.zeros(N * SZ, dtype=torch.uint8, device="cuda")
        dest_vas = [int(dev.data_ptr()) + i * SZ for i in range(N)]
        src_ptrs = [int(host_ptr) + i * SZ for i in range(N)]

        err, stream = drv.cuStreamCreate(0)
        assert err == drv.CUresult.CUDA_SUCCESS
        batch_h2d(dest_vas, src_ptrs, [SZ] * N, int(stream))
        drv.cuStreamSynchronize(int(stream))

        out = dev.cpu().numpy()
        for i in range(N):
            for j in range(0, SZ, 257):  # sparse check
                expected = (i * 7 + j * 13) & 0xFF
                assert (
                    int(out[i * SZ + j]) == expected
                ), f"mismatch region {i} offset {j}"
    finally:
        rt.cudaFreeHost(host_ptr)


def test_batch_h2d_speedup_vs_loop():
    """Batched H2D should be meaningfully faster than N×cuMemcpyAsync."""
    torch.cuda.init()
    _ = torch.empty(1, device="cuda")
    from cuda.bindings import driver as drv
    from cuda.bindings import runtime as rt
    from gms_kv_ring.common.cuda_batch import batch_h2d, has_batch_h2d

    if not has_batch_h2d():
        pytest.skip("cuMemcpyBatchAsync not available")

    N, SZ = 128, 1024
    err, host_ptr = rt.cudaHostAlloc(N * SZ, 0)
    assert err == rt.cudaError_t.cudaSuccess
    try:
        dev = torch.zeros(N * SZ, dtype=torch.uint8, device="cuda")
        dsts = [int(dev.data_ptr()) + i * SZ for i in range(N)]
        srcs = [int(host_ptr) + i * SZ for i in range(N)]

        err, stream = drv.cuStreamCreate(0)
        assert err == drv.CUresult.CUDA_SUCCESS

        ITERS = 50
        # Loop baseline
        t0 = time.perf_counter()
        for _ in range(ITERS):
            for i in range(N):
                drv.cuMemcpyAsync(
                    drv.CUdeviceptr(dsts[i]),
                    drv.CUdeviceptr(srcs[i]),
                    SZ,
                    int(stream),
                )
        drv.cuStreamSynchronize(int(stream))
        loop_us = (time.perf_counter() - t0) / ITERS * 1e6

        t0 = time.perf_counter()
        for _ in range(ITERS):
            batch_h2d(dsts, srcs, [SZ] * N, int(stream))
        drv.cuStreamSynchronize(int(stream))
        batch_us = (time.perf_counter() - t0) / ITERS * 1e6

        speedup = loop_us / max(batch_us, 1e-6)
        print(
            f"\n[cuda_batch bench] N={N} size={SZ}B\n"
            f"  loop:  {loop_us:6.2f} µs/iter\n"
            f"  batch: {batch_us:6.2f} µs/iter\n"
            f"  speedup: {speedup:.2f}×"
        )
        assert speedup >= 2.0
    finally:
        rt.cudaFreeHost(host_ptr)
