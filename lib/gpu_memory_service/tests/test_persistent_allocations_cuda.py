# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real-CUDA integration: persistent allocations are the SAME physical
pages on both sides.

Single process exercises both the daemon role (PersistentAllocationManager
that does cuMemCreate + maps locally) and the engine role (imports the
exported FD into its own VA + cuMemMap). Then asserts that writes from
one side are visible on the other side — proving zero-copy.

Skipped unless CUDA is available + cuda-python is importable.
"""

from __future__ import annotations

import ctypes
import os

import pytest
from gpu_memory_service.common.cuda_utils import (
    cumem_address_free,
    cumem_address_reserve,
    cumem_import_from_shareable_handle_close_fd,
    cumem_map,
    cumem_release,
    cumem_set_access,
    cumem_unmap,
)
from gpu_memory_service.common.locks import GrantedLockType
from gpu_memory_service.server.persistent_allocations import PersistentAllocationManager

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.integration,
    pytest.mark.none,
    pytest.mark.gpu_1,
]

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("CUDA required", allow_module_level=True)

pytest.importorskip("cuda.bindings.driver")


def _engine_side_adopt(
    fd: int,
    aligned_size: int,
    granularity: int,
    device: int,
) -> tuple[int, int]:
    """Engine-side: import FD → reserve VA → cuMemMap → cuMemSetAccess.
    Returns (va_engine, handle_engine) so the test can tear down.
    Mirrors what the per-engine installer will do in P3-P5."""
    handle_engine = cumem_import_from_shareable_handle_close_fd(fd)
    va_engine = cumem_address_reserve(aligned_size, granularity)
    cumem_map(va_engine, aligned_size, handle_engine)
    cumem_set_access(
        va_engine,
        aligned_size,
        device,
        GrantedLockType.RW,
    )
    return va_engine, handle_engine


def test_daemon_write_visible_to_engine_via_same_physical_pages():
    """Daemon writes a pattern via its own VA. Engine reads via its
    own (different) VA. Bytes match → same physical pages."""
    device = int(os.environ.get("GMS_TEST_CUDA_DEVICE", "0"))
    if torch.cuda.device_count() == 0:
        pytest.skip("no CUDA devices")
    torch.cuda.set_device(device)
    # Force CUDA context init before the manager probes granularity.
    _ = torch.empty(1, device=f"cuda:{device}")

    m = PersistentAllocationManager(device=device)
    size = 2 * 1024 * 1024  # 2 MiB — matches the default VMM granularity
    alloc, reattached = m.claim("eng-test", "kv_pool", size)
    assert reattached is False
    assert alloc.va_daemon != 0

    # Engine side: re-export the FD (daemon's stored FD is its own to
    # keep; we dup so we own the engine-side FD for cuMemImport).
    _, engine_fd = m.export("eng-test", "kv_pool")
    granularity = m.granularity
    va_engine, handle_engine = _engine_side_adopt(
        engine_fd,
        alloc.aligned_size,
        granularity,
        device,
    )

    try:
        pattern = bytes((i * 7 + 13) & 0xFF for i in range(256))
        # Daemon writes via daemon VA.
        m.write_block("eng-test", "kv_pool", offset=0, data=pattern)
        # Engine reads via its own VA.
        from cuda.bindings import driver as drv

        host_buf = (ctypes.c_ubyte * 256)()
        (err,) = drv.cuMemcpyDtoH(host_buf, drv.CUdeviceptr(va_engine), 256)
        assert err == drv.CUresult.CUDA_SUCCESS
        engine_bytes = bytes(host_buf)
        assert engine_bytes == pattern, (
            "engine VA must see the same bytes daemon wrote — "
            "proves same physical pages"
        )

        # Reverse direction: engine writes via its VA, daemon reads via
        # its VA.
        new_pattern = bytes(((255 - i) * 11) & 0xFF for i in range(256))
        host_src = (ctypes.c_ubyte * 256).from_buffer_copy(new_pattern)
        (err,) = drv.cuMemcpyHtoD(
            drv.CUdeviceptr(va_engine),
            host_src,
            256,
        )
        assert err == drv.CUresult.CUDA_SUCCESS
        daemon_bytes = m.read_block("eng-test", "kv_pool", offset=0, size=256)
        assert daemon_bytes == new_pattern, "daemon VA must see the bytes engine wrote"
    finally:
        cumem_unmap(va_engine, alloc.aligned_size)
        cumem_address_free(va_engine, alloc.aligned_size)
        cumem_release(handle_engine)
        m.release("eng-test", "kv_pool")


def test_reattach_returns_same_physical_pages():
    """Unclaim + reclaim must return a mapping that sees the SAME
    bytes written before unclaim. Proves the engine-restart-survival
    primitive at the allocation layer."""
    device = int(os.environ.get("GMS_TEST_CUDA_DEVICE", "0"))
    torch.cuda.set_device(device)
    _ = torch.empty(1, device=f"cuda:{device}")

    m = PersistentAllocationManager(device=device)
    size = 2 * 1024 * 1024
    alloc1, _ = m.claim("eng-restart", "kv_pool", size)
    # Stamp a known pattern via the daemon's VA.
    pattern = bytes((i * 17 + 5) & 0xFF for i in range(512))
    m.write_block("eng-restart", "kv_pool", offset=0, data=pattern)

    # Simulate engine disconnect (claim released; allocation persists).
    m.unclaim("eng-restart", "kv_pool")

    # Reattach with same key.
    alloc2, reattached = m.claim("eng-restart", "kv_pool", size)
    try:
        assert reattached is True
        assert alloc2.allocation_id == alloc1.allocation_id
        assert alloc2.va_daemon == alloc1.va_daemon  # unchanged VA
        # Bytes still there.
        got = m.read_block("eng-restart", "kv_pool", offset=0, size=512)
        assert got == pattern, (
            "reattached persistent allocation must contain the bytes "
            "written before unclaim — engine-restart survival proven"
        )
    finally:
        m.release("eng-restart", "kv_pool")
