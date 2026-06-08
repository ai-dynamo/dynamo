# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end: torch.empty() inside gms_use_persistent_pool routes
through the persistent namespace, daemon owns the bytes, engine
restart re-attaches to the same physical pages.

This is the foundation real-engine installers (P3-P5) will use.
Engine code wrapping its KV-pool allocation in
``with gms_use_persistent_pool("kv_pool", device): ...`` gets
GMS-owned VMM-IPC KV pools transparently.

Run inside one of the engine dev shells with real CUDA.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("CUDA required", allow_module_level=True)


pytestmark = pytest.mark.filterwarnings(
    "ignore::pytest.PytestUnhandledThreadExceptionWarning",
)


def _start_daemon(socket_path: str, device: int):
    """Start a GMS daemon serving the weights-style server in an
    async thread. Returns (server, thread, loop_holder)."""
    from gpu_memory_service.server.rpc import GMSRPCServer

    server = GMSRPCServer(socket_path, device=device, allocation_retry_interval=0.05)
    holder: dict = {}

    def _runner():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        holder["loop"] = loop
        try:
            loop.run_until_complete(server.serve())
        finally:
            loop.close()

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if server._server is not None and os.path.exists(socket_path):
            break
        time.sleep(0.05)
    assert os.path.exists(socket_path), "daemon did not bind socket"
    return server, t, holder


def _stop_daemon(server, t, holder):
    loop = holder.get("loop")
    if loop is not None:

        def _cancel():
            if server._server is not None:
                server._server.close()

        loop.call_soon_threadsafe(_cancel)
    t.join(timeout=5)


def test_torch_empty_inside_persistent_pool_routes_to_daemon(tmp_path):
    """A torch.empty() call inside gms_use_persistent_pool returns a
    tensor whose data_ptr is a daemon-issued VMM-IPC VA. Writes from
    the engine side are visible to the daemon via its own VA."""
    from gpu_memory_service.client.torch.allocator import (
        get_or_create_persistent_allocator,
        gms_use_persistent_pool,
    )

    device = int(os.environ.get("GMS_TEST_CUDA_DEVICE", "0"))
    if torch.cuda.device_count() == 0:
        pytest.skip("no CUDA devices")
    torch.cuda.set_device(device)
    _ = torch.empty(1, device=f"cuda:{device}")  # force CUDA ctx init

    sock = str(tmp_path / "pool.sock")
    server, t, holder = _start_daemon(sock, device)
    try:
        engine_id = "torch-pool-test"
        get_or_create_persistent_allocator(
            sock,
            device,
            engine_id,
            tag="kv_pool",
        )

        # Allocate a 4 MiB tensor inside the persistent pool.
        size_bytes = 4 * 1024 * 1024
        n = size_bytes // 4  # float32
        with gms_use_persistent_pool("kv_pool", device):
            tensor = torch.empty(n, dtype=torch.float32, device=f"cuda:{device}")

        # The tensor's storage should be backed by a GMS-issued
        # persistent allocation. Find it via the daemon's manager.
        gms = server._gms
        persistent = gms.persistent
        # We don't know the exact aligned size from this side without
        # asking the daemon — but a single persistent claim should
        # exist under engine_id=torch-pool-test with sub_tag kv_pool#0.
        allocs = list(persistent.list(engine_id))
        assert len(allocs) == 1, (
            f"expected exactly 1 persistent allocation for {engine_id}, "
            f"got {len(allocs)}"
        )
        alloc = allocs[0]
        assert (
            alloc.va_daemon != 0
        ), "daemon must have its own VA into the same physical pages"

        # Engine side: write a pattern via the tensor.
        pattern_int = 0x12345678  # fits in signed int32
        tensor.fill_(0)
        torch.cuda.synchronize(device)
        as_int = tensor.view(torch.int32)
        as_int[:64] = pattern_int
        torch.cuda.synchronize(device)

        # Daemon side: read via va_daemon — must see the same bytes.
        bytes_read = persistent.read_block(
            engine_id,
            alloc.tag,
            offset=0,
            size=64 * 4,
        )
        # 0x12345678 is little-endian: 78 56 34 12.
        expected = b"\x78\x56\x34\x12" * 64
        assert bytes_read == expected, (
            "daemon read via va_daemon must match what engine wrote "
            "via its tensor.data_ptr — same physical pages"
        )

        # Reverse: daemon writes; engine reads.
        new_pattern = bytes((i & 0xFF) for i in range(256))
        persistent.write_block(engine_id, alloc.tag, offset=0, data=new_pattern)
        torch.cuda.synchronize(device)
        as_u8 = tensor.view(torch.uint8)
        engine_reads = bytes(as_u8[:256].cpu().numpy())
        assert engine_reads == new_pattern, (
            "engine tensor must observe daemon's writes via the " "same physical pages"
        )
    finally:
        _stop_daemon(server, t, holder)
