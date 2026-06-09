# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared test helpers for the P5 KV-transfer test suite.

Every test file in this suite needs the same basic shape: a port-pair
allocator (UCX/NIXL listen ports), a `Daemon` spawner backed by a
thread + asyncio loop, a teardown helper, an `_inject_block` helper
that puts content directly into `host_tier`, and a poll loop that
waits for blocks to land at a destination's staging tier.

These were inlined in each test file (5 copies, slightly diverging
names — `_spawn` vs `_spawn_daemon`, `_inject` vs `_inject_block`).
Moved here to keep the test files focused on what they assert."""

from __future__ import annotations

import asyncio
import ctypes
import os
import tempfile
import threading
import time

# Port allocator: each test spawn picks a fresh (a, b) pair. Process-
# unique base keeps parallel pytest workers from colliding on UCX
# listen ports. Each test file imports this module so the lock and
# counter are shared process-wide.
_PORT_BASE = 0x4000 + (os.getpid() % 100) * 200
_port_lock = threading.Lock()
_next_port = [_PORT_BASE]


def next_port_pair() -> tuple[int, int]:
    """Return two consecutive free ports for a daemon pair."""
    with _port_lock:
        a, b = _next_port[0], _next_port[0] + 1
        _next_port[0] += 2
    return a, b


def spawn_daemon(
    uds: str,
    nixl_port: int,
    agent_name: str,
    *,
    cap: int = 64 << 20,
    send_buf: int = 32 << 20,
    recv_buf: int = 32 << 20,
    storage_dir_prefix: str = "gms-p5-",
    use_bytearray_allocator: bool = True,
):
    """Spawn one `Daemon` on a background thread and wait until its
    UDS socket is bound.

    Returns `(daemon, thread, holder)` — `holder["loop"]` is the
    daemon's asyncio loop, used by `stop_daemon` for clean shutdown.

    `use_bytearray_allocator=True` swaps in the test-only
    `_BytearrayAllocator` so the staging tier doesn't need real
    cudaHostAlloc / pinned memory."""
    from gms_kv_ring.daemon.server import Daemon

    daemon = Daemon(
        listen_socket=uds,
        storage_dir=tempfile.mkdtemp(prefix=storage_dir_prefix),
        supervise_backend=False,
        staging_capacity_bytes=cap,
        transport_listen_port=nixl_port,
        transport_agent_name=agent_name,
        staging_receive_buffer_bytes=recv_buf,
        staging_send_buffer_bytes=send_buf,
    )
    if use_bytearray_allocator:
        from gms_kv_ring.daemon.staging_tier import _BytearrayAllocator

        daemon.staging_tier._alloc = _BytearrayAllocator()
    holder: dict = {}

    def _runner():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        holder["loop"] = loop
        try:
            loop.run_until_complete(daemon.serve())
        finally:
            loop.close()

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not os.path.exists(uds):
        time.sleep(0.02)
    assert os.path.exists(uds), f"daemon never bound socket {uds}"
    return daemon, t, holder


def stop_daemon(daemon, t, holder) -> None:
    """Tear down a daemon spawned by `spawn_daemon`. Idempotent."""
    if "loop" in holder:
        try:
            holder["loop"].call_soon_threadsafe(daemon.stop)
        except Exception:  # noqa: BLE001
            pass
    t.join(timeout=5)


def inject_block(daemon, engine_id: str, layer: int, offset: int, data: bytes) -> None:
    """Place a block directly into a daemon's `host_tier`, mimicking
    what an engine's evict/spill path would do. Used to set up test
    scenarios where the daemon has KV bytes available without
    actually running an engine."""
    ptr = daemon.host_tier.put(engine_id, layer, offset, len(data))
    ctypes.memmove(ptr, data, len(data))
    daemon.host_tier.mark_ready(engine_id, layer, offset, crc=0)


def wait_for_arrival(
    dest_daemon,
    hex_hashes: "list[str]",
    sla: float,
    poll: float = 0.005,
) -> int:
    """Poll `dest_daemon.staging_tier._slots` until all `hex_hashes`
    are in the READY state, or `sla` seconds pass. Returns the count
    of ready blocks at exit (== len(hex_hashes) on success)."""
    needed = {bytes.fromhex(h) for h in hex_hashes}
    deadline = time.monotonic() + sla
    last_ready = 0
    while time.monotonic() < deadline:
        ready = sum(
            1
            for k in needed
            if k in dest_daemon.staging_tier._slots
            and dest_daemon.staging_tier._slots[k].state.value == "ready"
        )
        last_ready = ready
        if ready == len(needed):
            return ready
        time.sleep(poll)
    return last_ready
