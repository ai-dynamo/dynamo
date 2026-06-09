# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end failover proof in VMM-IPC mode.

Engine A allocates KV tensors via gms_use_persistent_pool and stamps
known patterns. Engine A "dies" (process simulation: tear down its
client manager + close its handles to the persistent allocations).
Daemon retains the physical pages. Engine B starts fresh, registers
under the SAME engine_id, calls gms_use_persistent_pool, gets
tensors that wrap the SAME PHYSICAL PAGES, and reads the patterns
back byte-identically.

This is the equivalent of test_failover_at_scale but for the new
VMM-IPC ownership model — proves that the layer that the per-engine
installers (P3-P5) sit on top of correctly preserves KV across
engine restart.
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


# The async daemon thread raises CancelledError on teardown — that's
# the expected shutdown path. Silence pytest's strict thread-exception
# warning for this file.
pytestmark = pytest.mark.filterwarnings(
    "ignore::pytest.PytestUnhandledThreadExceptionWarning",
)


def _start_daemon(socket_path: str, device: int):
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


def _simulate_engine_restart():
    """Tear down whatever the engine-side allocator left behind so the
    next get_or_create_persistent_allocator() registers a fresh state.
    This mimics what happens when the engine subprocess actually dies."""
    from gpu_memory_service.client.torch import allocator as A

    for tag, state in list(A._tag_states.items()):
        try:
            state.manager.abort()
        except Exception:  # noqa: BLE001
            pass
        A._tag_states.pop(tag, None)


@pytest.fixture(autouse=True)
def _clean_allocator_registry():
    """Wipe the process-global allocator registry around each test
    to keep state from previous test modules from leaking in. Real
    engine processes don't share state with each other; this fixture
    makes the in-process tests match that invariant."""
    _simulate_engine_restart()
    yield
    _simulate_engine_restart()


def test_vmm_ipc_engine_restart_preserves_kv_bytes(tmp_path):
    from gpu_memory_service.client.torch.allocator import (
        get_or_create_persistent_allocator,
        gms_use_persistent_pool,
    )

    device = int(os.environ.get("GMS_TEST_CUDA_DEVICE", "0"))
    if torch.cuda.device_count() == 0:
        pytest.skip("no CUDA devices")
    torch.cuda.set_device(device)
    _ = torch.empty(1, device=f"cuda:{device}")

    sock = str(tmp_path / "failover-vmm-ipc.sock")
    server, daemon_thread, holder = _start_daemon(sock, device)
    try:
        engine_id = "failover-test-engine"
        N_LAYERS = 8
        N_ELEMS = 64 * 1024  # 256 KiB per layer at int32
        # The first allocation will be rounded up to the VMM
        # granularity (typically 2 MiB) but Torch's mempool block
        # size is much larger. To keep this test fast we use a
        # modest per-layer size and let Torch coalesce.

        # === Engine A: allocate + write known patterns ===
        get_or_create_persistent_allocator(
            sock,
            device,
            engine_id,
            tag="kv_pool",
        )
        patterns: list[int] = []
        tensors_a: list[torch.Tensor] = []
        with gms_use_persistent_pool("kv_pool", device):
            for layer in range(N_LAYERS):
                tensors_a.append(
                    torch.empty(N_ELEMS, dtype=torch.int32, device=f"cuda:{device}")
                )
        for layer, t_ in enumerate(tensors_a):
            pat = 0x10000000 + (layer * 0x100)
            patterns.append(pat)
            t_.fill_(pat)
        torch.cuda.synchronize(device)

        # Verify the daemon sees the same bytes. The daemon's
        # PersistentAllocationManager will have one persistent
        # allocation (probably — Torch coalesces into one big block).
        # Either way, every (engine_id, sub_tag) pair must be present
        # in the daemon's claimed set.
        persistent = server._gms.persistent
        engine_a_allocs = sorted(
            persistent.list(engine_id),
            key=lambda a: a.tag,
        )
        assert (
            engine_a_allocs
        ), "Engine A should have at least one persistent allocation"
        for alloc in engine_a_allocs:
            assert persistent.is_claimed(alloc.engine_id, alloc.tag)

        # Snapshot what's in HBM via the engine tensors (for ground truth).
        ground_truth = [t_.clone().cpu() for t_ in tensors_a]

        # === Engine A "dies" ===
        del tensors_a
        _simulate_engine_restart()

        # After the simulated restart the daemon's allocations
        # persist but the claims are released (cleanup_connection
        # already ran on the closed connection).
        for alloc in engine_a_allocs:
            # Will become unclaimed when the daemon processes the
            # disconnect. Allow a brief window for cleanup to settle.
            for _ in range(50):
                if not persistent.is_claimed(alloc.engine_id, alloc.tag):
                    break
                time.sleep(0.02)
            assert not persistent.is_claimed(alloc.engine_id, alloc.tag), (
                f"claim should be released after engine A disconnect: "
                f"{alloc.engine_id}/{alloc.tag}"
            )

        # === Engine B: same engine_id, fresh client ===
        get_or_create_persistent_allocator(
            sock,
            device,
            engine_id,
            tag="kv_pool",
        )
        tensors_b: list[torch.Tensor] = []
        # IMPORTANT: For Engine B's tensors to land at the SAME
        # PHYSICAL PAGES, the allocation requests have to follow the
        # same order/shape. The daemon's auto-tag counter resets per
        # process (because _TagState is per-process), so when engine
        # B's pluggable allocator calls `kv_pool#0`, `kv_pool#1`, ...
        # those keys match what engine A used → reattach.
        with gms_use_persistent_pool("kv_pool", device):
            for layer in range(N_LAYERS):
                tensors_b.append(
                    torch.empty(N_ELEMS, dtype=torch.int32, device=f"cuda:{device}")
                )

        # Verify Engine B's tensors contain Engine A's data — proves
        # the same physical pages.
        for layer, t_ in enumerate(tensors_b):
            got = t_.cpu()
            want = ground_truth[layer]
            assert torch.equal(got, want), (
                f"layer {layer}: Engine B's tensor does NOT match "
                f"Engine A's pre-restart contents (first 4 ints: "
                f"got={got[:4].tolist()} want={want[:4].tolist()})"
            )
        print(
            f"[vmm-ipc failover] {N_LAYERS} layers × {N_ELEMS*4//1024} KiB "
            f"survived engine restart byte-identically"
        )
    finally:
        _simulate_engine_restart()
        _stop_daemon(server, daemon_thread, holder)


def test_vmm_ipc_sleep_wake_remaps_persistent_kv_bytes(tmp_path):
    from gpu_memory_service.client.torch.allocator import (
        get_gms_client_memory_manager,
        get_or_create_persistent_allocator,
        gms_use_persistent_pool,
    )
    from gpu_memory_service.common.locks import RequestedLockType

    device = int(os.environ.get("GMS_TEST_CUDA_DEVICE", "0"))
    if torch.cuda.device_count() == 0:
        pytest.skip("no CUDA devices")
    torch.cuda.set_device(device)
    _ = torch.empty(1, device=f"cuda:{device}")

    sock = str(tmp_path / "sleep-wake-vmm-ipc.sock")
    server, daemon_thread, holder = _start_daemon(sock, device)
    try:
        engine_id = "sleep-wake-test-engine"
        get_or_create_persistent_allocator(
            sock,
            device,
            engine_id,
            tag="kv_pool",
        )
        with gms_use_persistent_pool("kv_pool", device):
            tensor = torch.empty(128 * 1024, dtype=torch.int32, device=f"cuda:{device}")
        tensor.fill_(0x51525354)
        torch.cuda.synchronize(device)
        expected = tensor.clone().cpu()

        manager = get_gms_client_memory_manager("kv_pool")
        assert manager is not None
        assert manager.socket_path == sock
        manager.unmap_all_vas()
        manager.abort()

        manager.connect(RequestedLockType.RW)
        assert manager.socket_path == sock
        manager.remap_persistent_vas(engine_id)

        observed = tensor.cpu()
        assert torch.equal(observed, expected)
    finally:
        _simulate_engine_restart()
        _stop_daemon(server, daemon_thread, holder)
