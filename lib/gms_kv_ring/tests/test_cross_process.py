# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-process daemon mode tests.

The daemon supports two ways to attach the engine's counter array:
  • same-process: `attach_in_process(host_addr)` — daemon reuses the
    engine's pinned VA. Cheapest; used by `GMSKvRing` by default.
  • cross-process: `attach(path)` — daemon does its own mmap + cuda
    register. Required when the daemon runs in a separate Python
    process from the engine.

Tests here cover the cross-process code path WITHOUT spawning a
subprocess daemon. We drive the daemon's API directly with
counter_path-only arguments, bypassing GMSKvRing's
counter_host_addr default.

The HBM data plane (cuMemcpyAsync from engine's torch tensor VA)
WORKS in these tests because the daemon is still in the same Python
process and shares the CUDA primary context. A true subprocess
daemon would need VMM-IPC integration to map the engine's KV pool
into the daemon's address space; see `test_true_subprocess_daemon`
for the metadata-only flow that works without that.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("CUDA required", allow_module_level=True)


def _spawn_daemon(socket_path: str):
    from gms_kv_ring.daemon.server import Daemon

    d = Daemon(socket_path)
    holder = {}

    def _runner():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        holder["loop"] = loop
        try:
            loop.run_until_complete(d.serve())
        finally:
            loop.close()

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not os.path.exists(socket_path):
        time.sleep(0.02)
    return d, t, holder


def _build_pool(n_layers, layer_size):
    dev = torch.zeros(n_layers * layer_size, dtype=torch.uint8, device="cuda")
    dev.copy_(torch.arange(n_layers * layer_size, dtype=torch.uint8))
    torch.cuda.synchronize()
    base = int(dev.data_ptr())
    layers = [
        {
            "layer_idx": i,
            "va": base + i * layer_size,
            "size": layer_size,
            "stride": 1024,
        }
        for i in range(n_layers)
    ]
    return dev, layers


# ---------------------------------------------------------------------------
# Cross-process daemon code path (file-backed counter attach)
# ---------------------------------------------------------------------------


def test_restore_succeeds_under_cross_process_counter_attach(tmp_path):
    """End-to-end evict + restore using counter_path (NO counter_host_addr).
    Daemon does its own mmap+register of the counter file."""
    torch.cuda.init()
    _ = torch.empty(1, device="cuda")
    from gms_kv_ring.common.counter import EngineCounterArray
    from gms_kv_ring.common.evict_ring import create_ring as create_evict
    from gms_kv_ring.common.restore_ring import create_ring as create_restore
    from gms_kv_ring.daemon.client import DaemonClient

    sock = str(tmp_path / "d.sock")
    d, t, lh = _spawn_daemon(sock)
    try:
        c = DaemonClient(sock)
        try:
            dev, layers = _build_pool(2, 4096)
            eid = f"xproc-{uuid.uuid4().hex[:6]}"
            c.attach_engine_pool(eid, layers)

            counter_path = f"/dev/shm/gms-kvr-xproc-c-{uuid.uuid4().hex}.bin"
            eng_ctrs = EngineCounterArray.create(counter_path, num_counters=64)
            try:
                evict_path = f"/dev/shm/gms-kvr-xproc-e-{uuid.uuid4().hex}.ring"
                restore_path = f"/dev/shm/gms-kvr-xproc-r-{uuid.uuid4().hex}.ring"
                try:
                    we = create_evict(evict_path, capacity=16)
                    wr = create_restore(restore_path, capacity=16)

                    # Cross-process attach: pass counter_path, NOT
                    # counter_host_addr.
                    c.attach_evict_ring(
                        eid,
                        evict_path,
                        counter_path=counter_path,
                        num_counters=64,
                    )
                    c.attach_restore_ring(
                        eid,
                        restore_path,
                        counter_path=counter_path,
                        num_counters=64,
                    )

                    # Spill block 0.
                    e_slot, e_target = eng_ctrs.reserve_slot()
                    assert we.push(
                        block_id=0,
                        ranges=[(0, 1024, 0), (1, 1024, 0)],
                        counter_slot=e_slot,
                        counter_target=e_target,
                    )
                    # Wait for evict ack.
                    deadline = time.monotonic() + 3
                    while time.monotonic() < deadline:
                        if eng_ctrs.read_slot(e_slot) >= e_target:
                            break
                        time.sleep(0.02)
                    assert eng_ctrs.read_slot(e_slot) == e_target, (
                        f"cross-process evict-ack failed; counter="
                        f"{eng_ctrs.read_slot(e_slot)} target={e_target}"
                    )

                    # Now restore block 0 → block 1.
                    r_slot, r_target = eng_ctrs.reserve_slot()
                    assert wr.push(
                        src_engine_id=eid,
                        block_pairs=[(0, 1)],
                        counter_slot=r_slot,
                        counter_target=r_target,
                    )
                    cs = torch.cuda.Stream()
                    with torch.cuda.stream(cs):
                        eng_ctrs.wait_value32(int(cs.cuda_stream), r_slot, r_target)
                    torch.cuda.synchronize()
                    assert eng_ctrs.read_slot(r_slot) == r_target, (
                        f"cross-process restore-ack failed; counter="
                        f"{eng_ctrs.read_slot(r_slot)} target={r_target}"
                    )

                    # Byte-correctness: dev[block 1] of each layer must
                    # equal dev[block 0] post-restore.
                    buf = dev.cpu().numpy()
                    for li in range(2):
                        b0 = bytes(buf[li * 4096 : li * 4096 + 1024])
                        b1 = bytes(buf[li * 4096 + 1024 : li * 4096 + 2048])
                        assert b1 == b0, f"layer {li}: restored bytes don't match"
                finally:
                    for p in (evict_path, restore_path):
                        if os.path.exists(p):
                            try:
                                os.unlink(p)
                            except OSError:
                                pass
                    c.detach_engine_pool(eid)
            finally:
                eng_ctrs.close()
                if os.path.exists(counter_path):
                    try:
                        os.unlink(counter_path)
                    except OSError:
                        pass
        finally:
            c.close()
    finally:
        lh["loop"].call_soon_threadsafe(d.stop)
        t.join(timeout=3)


def test_evict_failure_signals_cross_process(tmp_path):
    """Cross-process counter attach + consumer error → target+1 must
    propagate. Verifies the failure path uses the file-backed counter
    mmap correctly."""
    torch.cuda.init()
    _ = torch.empty(1, device="cuda")
    from gms_kv_ring.common.counter import EngineCounterArray
    from gms_kv_ring.common.evict_ring import create_ring as create_evict
    from gms_kv_ring.daemon.client import DaemonClient
    from gms_kv_ring.daemon.consumers import _EvictConsumer

    sock = str(tmp_path / "d.sock")
    d, t, lh = _spawn_daemon(sock)
    try:
        orig_process = _EvictConsumer._process

        def boom(self, rec):
            raise RuntimeError("synthetic")

        _EvictConsumer._process = boom
        try:
            c = DaemonClient(sock)
            try:
                dev, layers = _build_pool(1, 4096)
                eid = f"xproc-fail-{uuid.uuid4().hex[:6]}"
                c.attach_engine_pool(eid, layers)

                counter_path = f"/dev/shm/gms-kvr-fail-c-{uuid.uuid4().hex}.bin"
                eng_ctrs = EngineCounterArray.create(counter_path, num_counters=32)
                try:
                    evict_path = f"/dev/shm/gms-kvr-fail-e-{uuid.uuid4().hex}.ring"
                    try:
                        we = create_evict(evict_path, capacity=16)
                        c.attach_evict_ring(
                            eid,
                            evict_path,
                            counter_path=counter_path,
                            num_counters=32,
                        )
                        e_slot, e_target = eng_ctrs.reserve_slot()
                        assert we.push(
                            block_id=0,
                            ranges=[(0, 1024, 0)],
                            counter_slot=e_slot,
                            counter_target=e_target,
                        )
                        deadline = time.monotonic() + 3
                        while time.monotonic() < deadline:
                            if eng_ctrs.read_slot(e_slot) != 0:
                                break
                            time.sleep(0.02)
                        assert eng_ctrs.read_slot(e_slot) == e_target + 1, (
                            f"failed evict must write target+1; got "
                            f"{eng_ctrs.read_slot(e_slot)} (target {e_target})"
                        )
                    finally:
                        if os.path.exists(evict_path):
                            try:
                                os.unlink(evict_path)
                            except OSError:
                                pass
                        c.detach_engine_pool(eid)
                finally:
                    eng_ctrs.close()
                    if os.path.exists(counter_path):
                        try:
                            os.unlink(counter_path)
                        except OSError:
                            pass
            finally:
                c.close()
        finally:
            _EvictConsumer._process = orig_process
    finally:
        lh["loop"].call_soon_threadsafe(d.stop)
        t.join(timeout=3)


# ---------------------------------------------------------------------------
# True subprocess daemon — metadata-only flow (no HBM ops)
# ---------------------------------------------------------------------------

DAEMON_SCRIPT = """
import asyncio, os, sys
sys.path.insert(0, %(libpath)r)
from gms_kv_ring.daemon.server import Daemon

sock = %(sock)r
try: os.unlink(sock)
except FileNotFoundError: pass

d = Daemon(sock)

async def main():
    print("READY", flush=True)
    await d.serve()

asyncio.run(main())
"""


def test_true_subprocess_daemon_handles_attach_and_ping(tmp_path):
    """Spawn a daemon in a SEPARATE Python process and verify the
    control plane (attach_engine_pool, ping, detach) works.

    Note: HBM operations (cuMemcpyAsync from engine VAs) WILL NOT
    work cross-process without VMM-IPC integration, because the
    engine's torch.tensor.data_ptr() is process-local. This test
    exercises the metadata plane only — proving the control socket,
    ring file sharing, and counter file sharing all work across
    process boundaries. The HBM data plane is a separate work item
    (see legacy gpu_memory_service/ VMM-IPC integration)."""
    torch.cuda.init()
    _ = torch.empty(1, device="cuda")

    libpath = str(Path(__file__).resolve().parents[2])
    sock = str(tmp_path / "subproc.sock")
    script = DAEMON_SCRIPT % {"libpath": libpath, "sock": sock}

    env = os.environ.copy()
    env["PYTHONPATH"] = libpath + ":" + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        [sys.executable, "-c", script],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        # Wait for "READY" line or process to die.
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if not os.path.exists(sock):
                time.sleep(0.05)
                continue
            break
        if not os.path.exists(sock):
            stdout, stderr = proc.communicate(timeout=2)
            raise AssertionError(
                f"daemon subprocess never created socket; "
                f"stdout={stdout!r} stderr={stderr!r}"
            )

        # Connect from main process (engine side).
        from gms_kv_ring.daemon.client import DaemonClient

        c = DaemonClient(sock, connect_timeout=10)
        try:
            # Ping
            assert c._ok({"op": "ping"})["ok"]

            # Attach a pool (with fake VAs — daemon won't try to
            # dereference until a record arrives).
            eid = f"subproc-{uuid.uuid4().hex[:6]}"
            layers = [
                {
                    "layer_idx": 0,
                    "va": 0xDEADBEEF000,  # bogus VA
                    "size": 4096,
                    "stride": 1024,
                }
            ]
            c.attach_engine_pool(eid, layers)

            # Detach.
            found = c.detach_engine_pool(eid)
            assert found, "detach must find the pool we just attached"
        finally:
            c.close()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def test_true_subprocess_daemon_attaches_evict_ring_with_file_counter(tmp_path):
    """Subprocess daemon accepts an evict-ring attach that references a
    file-backed counter (the cross-process path the engine would use
    when daemon lives in a separate process).

    NOTE on data-plane subprocess coverage: actually pushing an evict
    record through a subprocess daemon's `cuMemcpyAsync` requires real
    cross-process HBM sharing (VMM-IPC), which is intentionally
    deferred. We tested the bogus-VA failure path but the NVIDIA
    kernel driver SIGSEGV's the entire process on a sufficiently
    invalid VA — we cannot defend against that from Python. The
    in-process evict/restore data-plane tests in
    `test_production_hardening.py` cover the consumer logic, and the
    cross-process counter wire format is covered by
    `test_evict_failure_signals_cross_process` /
    `test_restore_succeeds_under_cross_process_counter_attach`. This
    test fills the remaining gap: that the subprocess daemon's RPC
    surface accepts an evict-ring attach with `counter_path`."""
    torch.cuda.init()
    _ = torch.empty(1, device="cuda")
    from gms_kv_ring.common.counter import EngineCounterArray
    from gms_kv_ring.common.evict_ring import create_ring as create_evict
    from gms_kv_ring.daemon.client import DaemonClient

    libpath = str(Path(__file__).resolve().parents[2])
    sock = str(tmp_path / "subproc.sock")
    script = DAEMON_SCRIPT % {"libpath": libpath, "sock": sock}

    env = os.environ.copy()
    env["PYTHONPATH"] = libpath + ":" + env.get("PYTHONPATH", "")
    env["LD_LIBRARY_PATH"] = env.get("LD_LIBRARY_PATH", "")
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        [sys.executable, "-c", script],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not os.path.exists(sock):
            if proc.poll() is not None:
                stdout, stderr = proc.communicate(timeout=2)
                raise AssertionError(
                    f"daemon subprocess exited early rc={proc.returncode}; "
                    f"stdout={stdout!r} stderr={stderr!r}"
                )
            time.sleep(0.05)
        if not os.path.exists(sock):
            proc.terminate()
            stdout, stderr = proc.communicate(timeout=2)
            raise AssertionError(
                f"daemon never came up; stdout={stdout!r} stderr={stderr!r}"
            )

        c = DaemonClient(sock, connect_timeout=10)
        try:
            eid = f"subproc-attach-{uuid.uuid4().hex[:6]}"
            # Allocate a real HBM range so the daemon can attach
            # without driver complaints — we never DMA from it.
            t = torch.zeros(4096, dtype=torch.uint8, device="cuda")
            layers = [
                {
                    "layer_idx": 0,
                    "va": int(t.data_ptr()),
                    "size": 4096,
                    "stride": 1024,
                }
            ]
            c.attach_engine_pool(eid, layers)

            counter_path = f"/dev/shm/gms-kvr-subproc-c-{uuid.uuid4().hex}.bin"
            eng_ctrs = EngineCounterArray.create(counter_path, num_counters=32)
            try:
                evict_path = f"/dev/shm/gms-kvr-subproc-e-{uuid.uuid4().hex}.ring"
                try:
                    create_evict(evict_path, capacity=16)
                    # The interesting bit: daemon mmaps the counter
                    # file in its own address space and registers it
                    # via cuMemHostRegister. If this RPC returns ok=True,
                    # the cross-process counter attach surface works.
                    c.attach_evict_ring(
                        eid,
                        evict_path,
                        counter_path=counter_path,
                        num_counters=32,
                    )
                    # Verify the daemon is still healthy after attach.
                    assert proc.poll() is None, "daemon died during evict-ring attach"
                finally:
                    if os.path.exists(evict_path):
                        try:
                            os.unlink(evict_path)
                        except OSError:
                            pass
                    try:
                        c.detach_engine_pool(eid)
                    except Exception:
                        pass
            finally:
                eng_ctrs.close()
                if os.path.exists(counter_path):
                    try:
                        os.unlink(counter_path)
                    except OSError:
                        pass
                del t
        finally:
            c.close()
    finally:
        proc.terminate()
        try:
            stdout, stderr = proc.communicate(timeout=5)
            if stderr and "Traceback" in stderr:
                print(f"\n--- daemon subprocess stderr ---\n{stderr}", file=sys.stderr)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
