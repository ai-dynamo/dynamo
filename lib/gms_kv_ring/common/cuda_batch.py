# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Batched H2D via cuMemcpyBatchAsync (CUDA 12.6+).

The daemon does N×L cuMemcpyAsync per chunk restore. Looping them
costs ~2 µs per call in CUDA driver overhead. Batching them via
cuMemcpyBatchAsync drops that to one call total — measured ~4×
speedup at N=128.

Falls back gracefully on older CUDA where the API isn't available
(caller checks `has_batch_h2d()` first).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def has_batch_h2d() -> bool:
    """True iff cuMemcpyBatchAsync is available in the linked CUDA."""
    from cuda.bindings import driver as drv

    return hasattr(drv, "cuMemcpyBatchAsync")


def batch_h2d(
    dest_vas: list[int],
    src_host_ptrs: list[int],
    sizes: list[int],
    stream: int,
) -> None:
    """Issue N H2D copies in ONE driver call. All ops queue on `stream`.

    Requires CUDA 12.6+. Caller is responsible for sync (typically via
    cuStreamWriteValue32 enqueued after this returns).

    Args lists must be same length. dest_vas are CUdeviceptr-castable
    device addresses; src_host_ptrs are pinned-host pointers.
    """
    n = len(dest_vas)
    if n == 0:
        return
    if len(src_host_ptrs) != n or len(sizes) != n:
        raise ValueError(
            f"batch_h2d length mismatch: "
            f"dsts={n} srcs={len(src_host_ptrs)} sizes={len(sizes)}",
        )

    from cuda.bindings import driver as drv

    dsts = [drv.CUdeviceptr(int(v)) for v in dest_vas]
    srcs = [drv.CUdeviceptr(int(v)) for v in src_host_ptrs]
    # CUmemcpyAttributes with STREAM access order — required for
    # mixed host/device source pointers under UVA.
    attrs = drv.CUmemcpyAttributes()
    attrs.srcAccessOrder = drv.CUmemcpySrcAccessOrder.CU_MEMCPY_SRC_ACCESS_ORDER_STREAM
    err = drv.cuMemcpyBatchAsync(
        dsts,
        srcs,
        sizes,
        n,
        [attrs],
        [0] * n,
        1,
        int(stream),
    )[0]
    if err != drv.CUresult.CUDA_SUCCESS:
        _, msg = drv.cuGetErrorString(err)
        raise RuntimeError(
            f"cuMemcpyBatchAsync({n}) failed: " f"{msg.decode() if msg else err}",
        )
