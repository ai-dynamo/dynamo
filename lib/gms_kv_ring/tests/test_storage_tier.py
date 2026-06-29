# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the storage tier — filesystem-backed offload below
host_tier — and the daemon's demote/promote state transitions.

Covered:
  - Unit: round-trip demote + promote with CRC carrying through
  - Unit: at-rest corruption (file byte flip) is caught by promote
  - Unit: bad magic / wrong size / missing file all fail safely
  - Unit: release_engine cleans up files
  - Integration via daemon RPC: spill -> demote -> promote -> verify
"""

from __future__ import annotations

import asyncio
import ctypes
import os
import struct
import threading
import time
import uuid
import zlib

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("CUDA required", allow_module_level=True)


def _spawn_daemon(socket_path: str, storage_dir: str):
    from gms_kv_ring.daemon.server import Daemon

    d = Daemon(socket_path, storage_dir=storage_dir)
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


# ---------------------------------------------------------------------------
# Unit: StorageTier directly (no daemon, no CUDA — pure host bytes)
# ---------------------------------------------------------------------------


def test_storage_tier_round_trip_with_crc(tmp_path):
    """Write a pattern + CRC into the storage tier, read it back into
    a fresh buffer, verify bytes and CRC match."""
    from gms_kv_ring.daemon.storage_tier import StorageTier

    st = StorageTier(str(tmp_path / "storage"))
    size = 4096
    payload = bytes((i & 0xFF) for i in range(size))
    src = (ctypes.c_ubyte * size).from_buffer_copy(payload)
    crc = zlib.crc32(payload) & 0xFFFFFFFF

    slot = st.demote(
        engine_id="eng",
        layer=0,
        offset=0,
        host_ptr=ctypes.addressof(src),
        size=size,
        crc=crc,
    )
    assert os.path.exists(slot.path)
    assert slot.size == size
    assert slot.crc == crc
    assert st.n_slots() == 1

    # Read into a fresh buffer.
    dst = (ctypes.c_ubyte * size)()
    verified = st.promote(
        engine_id="eng",
        layer=0,
        offset=0,
        dest_host_ptr=ctypes.addressof(dst),
        max_size=size,
    )
    assert verified == crc, "promote must return the verified CRC"
    assert bytes(dst) == payload, "promoted bytes must match original"


def test_storage_tier_promote_detects_at_rest_corruption(tmp_path):
    """Flip a single byte in the storage file (after the header) so
    the on-disk CRC stamp no longer matches the bytes. Promote must
    return None — engine must not get corrupted data."""
    from gms_kv_ring.daemon.storage_tier import StorageTier

    st = StorageTier(str(tmp_path / "storage"))
    size = 2048
    payload = bytes((i * 7 & 0xFF) for i in range(size))
    src = (ctypes.c_ubyte * size).from_buffer_copy(payload)
    crc = zlib.crc32(payload) & 0xFFFFFFFF

    slot = st.demote("eng", 0, 0, ctypes.addressof(src), size, crc)

    # Header is 24 bytes (magic+version+crc+size+_pad).
    # Flip a byte well inside the payload.
    with open(slot.path, "r+b") as f:
        f.seek(100)
        byte = f.read(1)
        f.seek(100)
        f.write(bytes([byte[0] ^ 0xFF]))

    dst = (ctypes.c_ubyte * size)()
    verified = st.promote(
        "eng",
        0,
        0,
        ctypes.addressof(dst),
        max_size=size,
    )
    assert verified is None, (
        "corrupted file must fail promote — otherwise we'd hand "
        "corrupted bytes to the engine"
    )


def test_storage_tier_promote_handles_missing_file(tmp_path):
    """Slot in the index but the file got unlinked underneath us.
    Promote must return None rather than raising."""
    from gms_kv_ring.daemon.storage_tier import StorageTier

    st = StorageTier(str(tmp_path / "storage"))
    size = 1024
    payload = b"x" * size
    src = (ctypes.c_ubyte * size).from_buffer_copy(payload)
    crc = zlib.crc32(payload) & 0xFFFFFFFF

    slot = st.demote("eng", 0, 0, ctypes.addressof(src), size, crc)
    os.unlink(slot.path)  # disk file gone, but index still has slot

    dst = (ctypes.c_ubyte * size)()
    assert (
        st.promote(
            "eng",
            0,
            0,
            ctypes.addressof(dst),
            max_size=size,
        )
        is None
    )


def test_storage_tier_promote_rejects_undersize_dest(tmp_path):
    """If the caller's destination buffer is smaller than the file's
    payload, promote must refuse — otherwise we'd write OOB."""
    from gms_kv_ring.daemon.storage_tier import StorageTier

    st = StorageTier(str(tmp_path / "storage"))
    size = 4096
    payload = b"a" * size
    src = (ctypes.c_ubyte * size).from_buffer_copy(payload)
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    st.demote("eng", 0, 0, ctypes.addressof(src), size, crc)

    dst = (ctypes.c_ubyte * 1024)()  # too small
    assert (
        st.promote(
            "eng",
            0,
            0,
            ctypes.addressof(dst),
            max_size=1024,
        )
        is None
    )


def test_storage_tier_release_engine_unlinks_files(tmp_path):
    """release_engine must delete every slot's file and clean up the
    per-engine directory tree."""
    from gms_kv_ring.daemon.storage_tier import StorageTier

    st = StorageTier(str(tmp_path / "storage"))
    payload = b"y" * 256
    src = (ctypes.c_ubyte * 256).from_buffer_copy(payload)
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    paths = []
    for off in (0, 256, 512):
        slot = st.demote(
            "eng-A",
            layer=0,
            offset=off,
            host_ptr=ctypes.addressof(src),
            size=256,
            crc=crc,
        )
        paths.append(slot.path)
    # Also drop one slot for a different engine — must NOT be touched.
    other = st.demote(
        "eng-B",
        0,
        0,
        ctypes.addressof(src),
        256,
        crc,
    )

    n = st.release_engine("eng-A")
    assert n == 3
    for p in paths:
        assert not os.path.exists(p), f"file should be unlinked: {p}"
    assert os.path.exists(other.path), "other engine's slot must survive"
    assert st.n_slots() == 1


# ---------------------------------------------------------------------------
# Integration via Daemon RPC: spill -> demote -> promote -> verify
# ---------------------------------------------------------------------------


def test_demote_and_promote_round_trip_via_daemon(tmp_path):
    """Full state-machine round-trip:
       HBM_FRESH -> AT_REST_HOST -> AT_REST_STORAGE -> AT_REST_HOST.
    Verify the CRC stored on the host_tier slot after promote matches
    what was captured on the original evict — proving the integrity
    stamp survived the tier hop."""
    torch.cuda.init()
    _ = torch.empty(1, device="cuda")
    from gms_kv_ring.daemon.client import DaemonClient
    from gms_kv_ring.engines.handle import GMSKvRing

    sock = str(tmp_path / "d.sock")
    storage_dir = str(tmp_path / "storage")
    d, t, lh = _spawn_daemon(sock, storage_dir)
    try:
        n_layers = 1
        layer_size = 4096
        stride = 1024
        dev = torch.zeros(n_layers * layer_size, dtype=torch.uint8, device="cuda")
        pattern = torch.arange(stride, dtype=torch.uint8, device="cuda")
        dev[0:stride].copy_(pattern)
        torch.cuda.synchronize()
        layers = [
            {
                "layer_idx": 0,
                "va": int(dev.data_ptr()),
                "size": layer_size,
                "stride": stride,
            }
        ]

        eid = f"tier-{uuid.uuid4().hex[:6]}"
        h = GMSKvRing(engine_id=eid, daemon_socket=sock, layers=layers)
        c = DaemonClient(sock)
        try:
            # 1) AT_REST_HOST: spill block 0 into host_tier.
            res = h.record_evict(block_id=0, ranges=[(0, stride, 0)])
            assert res is not None
            slot_id, target = res
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline and not h.evict_succeeded(
                slot_id, target
            ):
                time.sleep(0.02)
            assert h.evict_succeeded(slot_id, target)

            host_slot_pre = d.host_tier._slots[(eid, 0, 0)]
            original_crc = host_slot_pre.crc
            assert original_crc != 0

            # 2) AT_REST_STORAGE: demote.
            assert c.demote_to_storage(eid, layer=0, offset=0)
            assert (
                d.host_tier.get(eid, 0, 0) is None
            ), "host_tier slot should be freed after demote"
            assert d.storage_tier.n_slots() == 1
            ss = d.storage_tier.get(eid, 0, 0)
            assert ss is not None
            assert ss.crc == original_crc, "storage tier must preserve the original CRC"

            # 3) Back to AT_REST_HOST: promote.
            assert c.promote_from_storage(eid, layer=0, offset=0)
            assert (
                d.storage_tier.n_slots() == 0
            ), "storage slot should be released after successful promote"
            host_slot_post = d.host_tier.get(eid, 0, 0)
            assert host_slot_post is not None and host_slot_post.ready
            assert (
                host_slot_post.crc == original_crc
            ), "CRC must round-trip unchanged through demote/promote"
            # And the actual bytes must match the original CRC.
            from gms_kv_ring.common.checksum import crc32_at_ptr

            actual = crc32_at_ptr(host_slot_post.host_ptr, host_slot_post.size)
            assert actual == original_crc, (
                f"bytes after promote must hash to original CRC; "
                f"got {actual:#x} want {original_crc:#x}"
            )
        finally:
            c.close()
            h.close()
    finally:
        lh["loop"].call_soon_threadsafe(d.stop)
        t.join(timeout=3)


def test_promote_fails_loudly_on_at_rest_corruption_via_daemon(tmp_path):
    """Same lifecycle as above, but flip a byte in the storage file
    while it's AT_REST_STORAGE. The promote RPC must return False
    AND bump the storage_tier_promote_failures metric, AND leave
    host_tier slot absent (engine falls to cold compute)."""
    torch.cuda.init()
    _ = torch.empty(1, device="cuda")
    from gms_kv_ring.common import metrics
    from gms_kv_ring.daemon.client import DaemonClient
    from gms_kv_ring.engines.handle import GMSKvRing

    sock = str(tmp_path / "d.sock")
    storage_dir = str(tmp_path / "storage")
    d, t, lh = _spawn_daemon(sock, storage_dir)
    try:
        layer_size, stride = 4096, 1024
        dev = torch.arange(layer_size, dtype=torch.uint8, device="cuda")
        torch.cuda.synchronize()
        layers = [
            {
                "layer_idx": 0,
                "va": int(dev.data_ptr()),
                "size": layer_size,
                "stride": stride,
            }
        ]
        eid = f"tier-bad-{uuid.uuid4().hex[:6]}"
        h = GMSKvRing(engine_id=eid, daemon_socket=sock, layers=layers)
        c = DaemonClient(sock)
        try:
            res = h.record_evict(block_id=0, ranges=[(0, stride, 0)])
            assert res is not None
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline and not h.evict_succeeded(*res):
                time.sleep(0.02)
            assert h.evict_succeeded(*res)

            assert c.demote_to_storage(eid, 0, 0)
            ss = d.storage_tier.get(eid, 0, 0)
            assert ss is not None

            # Tamper with the file at rest.
            with open(ss.path, "r+b") as f:
                f.seek(50)
                b = f.read(1)
                f.seek(50)
                f.write(bytes([b[0] ^ 0x55]))

            before = metrics.storage_tier_promote_failures.get(
                engine_id=eid,
            )
            assert not c.promote_from_storage(
                eid, 0, 0
            ), "tampered storage file must fail promote"
            after = metrics.storage_tier_promote_failures.get(
                engine_id=eid,
            )
            assert after == before + 1, "failure metric must bump"
            assert (
                d.host_tier.get(eid, 0, 0) is None
            ), "failed promote must release the provisional host slot"
        finally:
            c.close()
            h.close()
    finally:
        lh["loop"].call_soon_threadsafe(d.stop)
        t.join(timeout=3)


# ---------------------------------------------------------------------------
# Crash-recovery: restart-replay invariants
# ---------------------------------------------------------------------------


def test_storage_tier_reconciles_index_on_startup(tmp_path):
    """First StorageTier writes 3 slots. We DROP the instance (Python
    GC = process exit, the in-memory index vanishes). A fresh
    StorageTier rooted at the same dir must rebuild the index from
    disk so promotes still work — that's the whole point of having
    a persistent tier."""
    from gms_kv_ring.daemon.storage_tier import StorageTier

    storage_dir = str(tmp_path / "storage")
    st1 = StorageTier(storage_dir)
    payload = b"abc123" * 100
    size = len(payload)
    src = (ctypes.c_ubyte * size).from_buffer_copy(payload)
    crc = zlib.crc32(payload) & 0xFFFFFFFF

    paths = []
    for layer, offset in [(0, 0), (0, 1024), (1, 0)]:
        slot = st1.demote(
            "eng-A",
            layer=layer,
            offset=offset,
            host_ptr=ctypes.addressof(src),
            size=size,
            crc=crc,
        )
        paths.append(slot.path)
    assert st1.n_slots() == 3

    # "Daemon dies." Drop the instance — index is gone from RAM.
    del st1

    # Fresh instance rebuilds from disk.
    st2 = StorageTier(storage_dir)
    assert st2.n_slots() == 3, "reconcile must rebuild every valid slot found on disk"

    # And promote still works — header CRC matches payload bytes.
    dst = (ctypes.c_ubyte * size)()
    verified = st2.promote(
        "eng-A",
        0,
        0,
        ctypes.addressof(dst),
        max_size=size,
    )
    assert verified == crc, "promote after reconcile must verify CRC"
    assert bytes(dst) == payload


def test_storage_tier_reconcile_cleans_torn_temp_files(tmp_path):
    """If the daemon dies between mkstemp and rename, a .tmp-* file
    is left in the engine subtree. On restart, reconcile must unlink
    it (the atomic rename never ran, so the file is by definition
    incomplete / suspect)."""
    from gms_kv_ring.daemon.storage_tier import StorageTier

    storage_dir = str(tmp_path / "storage")
    # Prime the directory tree with a real file first so the engine
    # subdir exists.
    st1 = StorageTier(storage_dir)
    payload = b"x" * 1024
    src = (ctypes.c_ubyte * 1024).from_buffer_copy(payload)
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    real = st1.demote("eng", 0, 0, ctypes.addressof(src), 1024, crc)
    del st1

    # Drop a torn temp file alongside the real one. (This is the
    # state mkstemp leaves the disk in if we die before rename.)
    torn_path = os.path.join(os.path.dirname(real.path), ".tmp-fake.bin")
    with open(torn_path, "wb") as f:
        f.write(b"partial junk that never reached rename()")
    assert os.path.exists(torn_path)

    # Reconcile via fresh StorageTier.
    st2 = StorageTier(storage_dir)
    assert not os.path.exists(
        torn_path
    ), "torn .tmp-* file must be cleaned up on reconcile"
    assert st2.n_slots() == 1, "real slot must still be indexed"


def test_storage_tier_reconcile_skips_files_with_bad_header(tmp_path):
    """A file with a bad magic / version (e.g., from an older daemon
    version or corrupted on disk) must NOT enter the index. Leaving
    it on disk for forensics is fine; indexing it would mean a future
    promote tries to read a file we already know is suspect."""
    from gms_kv_ring.daemon.storage_tier import StorageTier

    storage_dir = str(tmp_path / "storage")
    # Lay down a real file so the engine subdir exists.
    st1 = StorageTier(storage_dir)
    payload = b"y" * 512
    src = (ctypes.c_ubyte * 512).from_buffer_copy(payload)
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    st1.demote("eng", 0, 0, ctypes.addressof(src), 512, crc)
    del st1

    # Drop a file with the right name but wrong header.
    bad_path = os.path.join(storage_dir, "eng", "0", "9999.bin")
    with open(bad_path, "wb") as f:
        f.write(b"NOT_A_VALID_HEADER" + b"\x00" * 100)

    st2 = StorageTier(storage_dir)
    assert (
        st2.n_slots() == 1
    ), f"only the valid slot must be indexed; got {st2.n_slots()}"
    # And get() for the bad slot's key returns None.
    assert st2.get("eng", 0, 9999) is None
    # File is still on disk (forensics) — we don't auto-delete bad
    # headers.
    assert os.path.exists(bad_path)


def test_storage_tier_reconcile_skips_truncated_files(tmp_path):
    """Header says payload is N bytes but the file on disk has <N
    bytes after the header (write killed between fsync and dirent
    commit, or filesystem corruption). Reconcile must not index it —
    promote would short-read and we'd waste cycles before failing."""
    from gms_kv_ring.daemon.storage_tier import StorageTier

    storage_dir = str(tmp_path / "storage")
    os.makedirs(os.path.join(storage_dir, "eng", "0"), exist_ok=True)
    bad_path = os.path.join(storage_dir, "eng", "0", "0.bin")

    # Craft a valid-looking header that claims 4096 payload bytes,
    # but write only 100 of them.
    declared = 4096
    header = struct.pack(
        "<IIIIQ",
        0x47535453,
        1,
        0xDEADBEEF,
        declared,
        0,
    )
    with open(bad_path, "wb") as f:
        f.write(header)
        f.write(b"\x00" * 100)  # truncated payload

    st = StorageTier(storage_dir)
    assert st.n_slots() == 0, (
        "truncated file must not be indexed; got n_slots=" f"{st.n_slots()}"
    )


def test_host_tier_is_empty_after_daemon_restart(tmp_path):
    """host_tier lives in pinned RAM. When the daemon process dies,
    those pages go back to the OS — there is no "torn host slot" to
    observe across restart. We prove this with two daemons sharing
    a storage_dir: daemon A spills, daemon B starts cleanly, and
    daemon B's host_tier index is empty even though daemon A's was
    populated. This is the resiliency invariant in its strongest
    form for the host tier."""
    torch.cuda.init()
    _ = torch.empty(1, device="cuda")
    from gms_kv_ring.engines.handle import GMSKvRing

    sock_a = str(tmp_path / "a.sock")
    sock_b = str(tmp_path / "b.sock")
    storage_dir = str(tmp_path / "storage")

    # Daemon A
    da, ta, lha = _spawn_daemon(sock_a, storage_dir)
    try:
        layer_size, stride = 4096, 1024
        dev_a = torch.arange(layer_size, dtype=torch.uint8, device="cuda")
        torch.cuda.synchronize()
        layers_a = [
            {
                "layer_idx": 0,
                "va": int(dev_a.data_ptr()),
                "size": layer_size,
                "stride": stride,
            }
        ]
        eid = f"restart-{uuid.uuid4().hex[:6]}"
        ha = GMSKvRing(engine_id=eid, daemon_socket=sock_a, layers=layers_a)
        try:
            res = ha.record_evict(block_id=0, ranges=[(0, stride, 0)])
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline and not ha.evict_succeeded(*res):
                time.sleep(0.02)
            assert ha.evict_succeeded(*res)
            assert da.host_tier.n_slots() == 1
        finally:
            # Detach engine via close().
            ha.close()
    finally:
        # NOTE: NOT a graceful shutdown of pools — we just stop the
        # asyncio loop and drop the Daemon. Pinned RAM goes back to
        # the OS along with the Python process state.
        lha["loop"].call_soon_threadsafe(da.stop)
        ta.join(timeout=3)
    del da

    # Daemon B — same storage_dir, fresh process semantically.
    db, tb, lhb = _spawn_daemon(sock_b, storage_dir)
    try:
        assert db.host_tier.n_slots() == 0, (
            "host_tier must be empty after restart — pinned RAM is "
            "reclaimed by the OS, no torn slots can survive"
        )
    finally:
        lhb["loop"].call_soon_threadsafe(db.stop)
        tb.join(timeout=3)


def test_daemon_restart_preserves_storage_files_for_engine_reattach(tmp_path):
    """End-to-end resiliency: daemon A demotes a block to storage,
    "dies" without graceful shutdown, daemon B starts on the same
    storage_dir, engine re-attaches with the same engine_id, promote
    succeeds (CRC verifies) and returns the original bytes.

    This is the invariant the user asked us to prove: 'GMS is
    persisted, if it dies the whole deployment is restarted, evicted
    blocks must not be corrupted.'"""
    torch.cuda.init()
    _ = torch.empty(1, device="cuda")
    from gms_kv_ring.common.checksum import crc32_at_ptr
    from gms_kv_ring.daemon.client import DaemonClient
    from gms_kv_ring.engines.handle import GMSKvRing

    sock_a = str(tmp_path / "a.sock")
    sock_b = str(tmp_path / "b.sock")
    storage_dir = str(tmp_path / "storage")
    layer_size, stride = 4096, 1024

    # Use a stable, deterministic engine_id so both daemon
    # incarnations key the same files.
    eid = f"resil-{uuid.uuid4().hex[:6]}"
    original_crc = None

    # Daemon A: spill -> demote -> "die"
    da, ta, lha = _spawn_daemon(sock_a, storage_dir)
    try:
        dev_a = torch.arange(layer_size, dtype=torch.uint8, device="cuda")
        torch.cuda.synchronize()
        layers_a = [
            {
                "layer_idx": 0,
                "va": int(dev_a.data_ptr()),
                "size": layer_size,
                "stride": stride,
            }
        ]
        ha = GMSKvRing(engine_id=eid, daemon_socket=sock_a, layers=layers_a)
        ca = DaemonClient(sock_a)
        try:
            res = ha.record_evict(block_id=0, ranges=[(0, stride, 0)])
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline and not ha.evict_succeeded(*res):
                time.sleep(0.02)
            assert ha.evict_succeeded(*res)
            original_crc = da.host_tier._slots[(eid, 0, 0)].crc

            assert ca.demote_to_storage(eid, 0, 0)
            assert da.storage_tier.n_slots() == 1
        finally:
            ca.close()
            ha.close()
    finally:
        # Same as above — not graceful, simulating crash.
        lha["loop"].call_soon_threadsafe(da.stop)
        ta.join(timeout=3)
    del da

    # Daemon B: reconcile picks up the file
    db, tb, lhb = _spawn_daemon(sock_b, storage_dir)
    try:
        assert (
            db.storage_tier.n_slots() == 1
        ), "reconcile must rebuild storage index from disk"
        # Engine re-attaches with same engine_id. A fresh HBM tensor
        # (different data_ptr) — that's expected; restart means new
        # HBM layout.
        dev_b = torch.zeros(layer_size, dtype=torch.uint8, device="cuda")
        torch.cuda.synchronize()
        layers_b = [
            {
                "layer_idx": 0,
                "va": int(dev_b.data_ptr()),
                "size": layer_size,
                "stride": stride,
            }
        ]
        hb = GMSKvRing(engine_id=eid, daemon_socket=sock_b, layers=layers_b)
        cb = DaemonClient(sock_b)
        try:
            assert cb.promote_from_storage(
                eid, 0, 0
            ), "engine restart must be able to promote durable blocks"
            slot_b = db.host_tier.get(eid, 0, 0)
            assert slot_b is not None and slot_b.ready
            assert slot_b.crc == original_crc, (
                f"CRC must survive daemon restart: "
                f"original={original_crc:#x} after_restart={slot_b.crc:#x}"
            )
            actual = crc32_at_ptr(slot_b.host_ptr, slot_b.size)
            assert (
                actual == original_crc
            ), "actual bytes after promote must hash to the original CRC"
        finally:
            cb.close()
            hb.close()
    finally:
        lhb["loop"].call_soon_threadsafe(db.stop)
        tb.join(timeout=3)


# ---------------------------------------------------------------------------
# Operator-driven cleanup: TTL / quota / explicit release
# ---------------------------------------------------------------------------


def _write_slot(st, eid, layer, offset, n_bytes):
    """Test helper: write a deterministic payload into a storage slot."""
    payload = bytes((i & 0xFF) for i in range(n_bytes))
    src = (ctypes.c_ubyte * n_bytes).from_buffer_copy(payload)
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    return st.demote(eid, layer, offset, ctypes.addressof(src), n_bytes, crc)


def test_storage_tier_prune_older_than_uses_mtime(tmp_path):
    """TTL eviction: slots whose mtime is older than (now - max_age)
    are unlinked and dropped from the index. Fresh slots survive."""
    from gms_kv_ring.daemon.storage_tier import StorageTier

    st = StorageTier(str(tmp_path / "storage"))

    old_a = _write_slot(st, "eng", 0, 0, 256)
    old_b = _write_slot(st, "eng", 0, 256, 256)
    fresh = _write_slot(st, "eng", 0, 512, 256)
    # Manually backdate two slots so we don't need to sleep.
    st._slots[("eng", 0, 0)].mtime = 1000.0
    st._slots[("eng", 0, 256)].mtime = 1000.0
    st._slots[("eng", 0, 512)].mtime = 2000.0

    # `now=2100` and max_age=300 → threshold=1800, so old_a/old_b
    # (mtime=1000) are pruned; fresh (mtime=2000) survives.
    n = st.prune_older_than(max_age_seconds=300, now=2100.0)
    assert n == 2
    assert st.n_slots() == 1
    assert not os.path.exists(old_a.path)
    assert not os.path.exists(old_b.path)
    assert os.path.exists(fresh.path)
    assert st.get("eng", 0, 512) is not None


def test_storage_tier_enforce_byte_quota_evicts_lru_first(tmp_path):
    """LRU eviction: when total bytes exceed the quota, evict
    oldest-mtime slots until under. Sizes are tracked exactly."""
    from gms_kv_ring.daemon.storage_tier import StorageTier

    st = StorageTier(str(tmp_path / "storage"))

    # Three 1 KB slots, total = 3 KB.
    s1 = _write_slot(st, "eng", 0, 0, 1024)
    s2 = _write_slot(st, "eng", 0, 1024, 1024)
    s3 = _write_slot(st, "eng", 0, 2048, 1024)
    st._slots[("eng", 0, 0)].mtime = 100.0  # oldest
    st._slots[("eng", 0, 1024)].mtime = 200.0
    st._slots[("eng", 0, 2048)].mtime = 300.0  # newest
    assert st.total_bytes() == 3072

    # Quota = 1500 bytes → must drop until <= 1500.
    # After dropping the 100.0 slot: 2048 bytes left (still > 1500).
    # After dropping the 200.0 slot: 1024 bytes left (<= 1500, stop).
    n = st.enforce_byte_quota(max_bytes=1500)
    assert n == 2, f"should evict 2 slots; got {n}"
    assert st.total_bytes() == 1024
    assert not os.path.exists(s1.path)
    assert not os.path.exists(s2.path)
    assert os.path.exists(s3.path), "newest must survive"


def test_storage_tier_quota_noop_when_under(tmp_path):
    """If total bytes already fit under the quota, no eviction
    happens — even if files exist."""
    from gms_kv_ring.daemon.storage_tier import StorageTier

    st = StorageTier(str(tmp_path / "storage"))
    _write_slot(st, "eng", 0, 0, 1024)
    assert st.enforce_byte_quota(max_bytes=10 * 1024) == 0
    assert st.n_slots() == 1


def test_storage_tier_stats_reflects_state(tmp_path):
    """stats() returns total bytes, slot count, and mtime extrema."""
    from gms_kv_ring.daemon.storage_tier import StorageTier

    st = StorageTier(str(tmp_path / "storage"))

    empty = st.stats()
    assert empty["n_slots"] == 0
    assert empty["total_bytes"] == 0

    _write_slot(st, "eng", 0, 0, 512)
    _write_slot(st, "eng", 0, 512, 256)
    st._slots[("eng", 0, 0)].mtime = 100.0
    st._slots[("eng", 0, 512)].mtime = 200.0

    s = st.stats()
    assert s["n_slots"] == 2
    assert s["total_bytes"] == 768
    assert s["oldest_mtime"] == 100.0
    assert s["newest_mtime"] == 200.0


def test_storage_tier_operator_cleanup_via_daemon_rpc(tmp_path):
    """End-to-end via DaemonClient: write some slots, prune via TTL,
    enforce quota, explicit release. Verifies the RPC wiring + the
    storage_stats observability surface."""
    torch.cuda.init()
    _ = torch.empty(1, device="cuda")
    from gms_kv_ring.common import metrics
    from gms_kv_ring.daemon.client import DaemonClient
    from gms_kv_ring.engines.handle import GMSKvRing

    sock = str(tmp_path / "d.sock")
    storage_dir = str(tmp_path / "storage")
    d, t, lh = _spawn_daemon(sock, storage_dir)
    try:
        layer_size, stride = 4096, 1024
        dev = torch.arange(layer_size, dtype=torch.uint8, device="cuda")
        torch.cuda.synchronize()
        layers = [
            {
                "layer_idx": 0,
                "va": int(dev.data_ptr()),
                "size": layer_size,
                "stride": stride,
            }
        ]
        eid = f"ops-{uuid.uuid4().hex[:6]}"
        h = GMSKvRing(engine_id=eid, daemon_socket=sock, layers=layers)
        c = DaemonClient(sock)
        try:
            # Spill + demote 3 blocks at different offsets.
            for off in (0, 1024, 2048):
                res = h.record_evict(
                    block_id=off // stride,
                    ranges=[(0, stride, off)],
                )
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline and not h.evict_succeeded(*res):
                    time.sleep(0.02)
                assert h.evict_succeeded(*res)
                assert c.demote_to_storage(eid, 0, off)

            stats = c.storage_stats()
            assert stats["n_slots"] == 3
            assert stats["total_bytes"] == 3 * stride

            # Backdate one slot so TTL prune evicts exactly it.
            d.storage_tier._slots[(eid, 0, 0)].mtime = 1.0  # ancient

            evicted_metric_before = metrics.storage_tier_evictions.get(
                reason="ttl",
            )
            n = c.prune_storage(max_age_seconds=60.0)
            assert n == 1, f"TTL should drop the ancient slot; got {n}"
            assert (
                metrics.storage_tier_evictions.get(reason="ttl")
                == evicted_metric_before + 1
            )
            stats = c.storage_stats()
            assert stats["n_slots"] == 2

            # Now enforce a quota tighter than what's left → LRU drop.
            quota_before = metrics.storage_tier_evictions.get(
                reason="quota",
            )
            n = c.prune_storage(max_bytes=stride)  # room for 1 slot only
            assert n == 1
            assert (
                metrics.storage_tier_evictions.get(reason="quota") == quota_before + 1
            )
            assert c.storage_stats()["n_slots"] == 1

            # Explicit release zeros out the engine's storage.
            explicit_before = metrics.storage_tier_evictions.get(
                reason="explicit",
            )
            n = c.release_engine_storage(eid)
            assert n == 1
            assert (
                metrics.storage_tier_evictions.get(reason="explicit")
                == explicit_before + 1
            )
            assert c.storage_stats()["n_slots"] == 0
        finally:
            c.close()
            h.close()
    finally:
        lh["loop"].call_soon_threadsafe(d.stop)
        t.join(timeout=3)


def test_detach_engine_preserves_storage_files(tmp_path):
    """Detaching an engine must NOT release its storage-tier slots.
    Storage is the persistent tier — files survive engine detach so
    that engine restart / daemon restart can promote them back. Stale
    cleanup is a separate operator-driven concern (quota / TTL /
    explicit release RPC), not coupled to detach lifecycle. This is
    the resiliency invariant: 'GMS is persisted, if it dies the whole
    deployment is restarted, evicted blocks must not be corrupted'
    — which requires the bytes to still be there after restart."""
    torch.cuda.init()
    _ = torch.empty(1, device="cuda")
    from gms_kv_ring.daemon.client import DaemonClient
    from gms_kv_ring.engines.handle import GMSKvRing

    sock = str(tmp_path / "d.sock")
    storage_dir = str(tmp_path / "storage")
    d, t, lh = _spawn_daemon(sock, storage_dir)
    try:
        layer_size, stride = 4096, 1024
        dev = torch.arange(layer_size, dtype=torch.uint8, device="cuda")
        torch.cuda.synchronize()
        layers = [
            {
                "layer_idx": 0,
                "va": int(dev.data_ptr()),
                "size": layer_size,
                "stride": stride,
            }
        ]
        eid = f"tier-detach-{uuid.uuid4().hex[:6]}"
        h = GMSKvRing(engine_id=eid, daemon_socket=sock, layers=layers)
        c = DaemonClient(sock)
        try:
            res = h.record_evict(block_id=0, ranges=[(0, stride, 0)])
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline and not h.evict_succeeded(*res):
                time.sleep(0.02)
            assert h.evict_succeeded(*res)
            assert c.demote_to_storage(eid, 0, 0)

            stored_path = d.storage_tier.get(eid, 0, 0).path
            assert os.path.exists(stored_path)
        finally:
            c.close()
            h.close()  # triggers detach_engine_pool

        # After detach: storage file PERSISTS (this is the invariant
        # we want for restart-replay correctness).
        assert os.path.exists(stored_path), (
            "detach must NOT unlink storage-tier files; otherwise "
            "daemon/engine restart can't promote durable blocks"
        )
        assert d.storage_tier.n_slots() == 1
    finally:
        lh["loop"].call_soon_threadsafe(d.stop)
        t.join(timeout=3)


# ---------------------------------------------------------------------------
# Background sweeper thread
# ---------------------------------------------------------------------------


def _spawn_daemon_with_sweeper(socket_path, storage_dir, **sweeper_kw):
    """Same as _spawn_daemon but passes sweeper config to Daemon()."""
    from gms_kv_ring.daemon.server import Daemon

    d = Daemon(socket_path, storage_dir=storage_dir, **sweeper_kw)
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


def test_sweeper_rejects_invalid_config(tmp_path):
    """Sweeper must refuse interval <= 0 or no max_age/max_bytes —
    those configurations would do nothing or loop hot."""
    from gms_kv_ring.daemon.storage_tier import StorageSweeper, StorageTier

    st = StorageTier(str(tmp_path / "storage"))
    with pytest.raises(ValueError):
        StorageSweeper(st, interval_s=0.0, max_age_s=10)
    with pytest.raises(ValueError):
        StorageSweeper(st, interval_s=-1, max_age_s=10)
    with pytest.raises(ValueError):
        StorageSweeper(st, interval_s=1.0)  # no policy


def test_sweeper_evicts_old_slots_on_schedule(tmp_path):
    """With a short interval and a low TTL, the sweeper must evict
    old slots without explicit operator intervention."""
    from gms_kv_ring.daemon.storage_tier import StorageSweeper, StorageTier

    st = StorageTier(str(tmp_path / "storage"))
    payload = b"z" * 256
    src = (ctypes.c_ubyte * 256).from_buffer_copy(payload)
    crc = zlib.crc32(payload) & 0xFFFFFFFF

    # Three slots; backdate two of them so they're TTL-eligible.
    paths = []
    for offset in (0, 256, 512):
        paths.append(st.demote("eng", 0, offset, ctypes.addressof(src), 256, crc).path)
    now = time.time()
    st._slots[("eng", 0, 0)].mtime = now - 1000
    st._slots[("eng", 0, 256)].mtime = now - 1000

    swept = threading.Event()

    def on_sweep(ttl_n, per_eng_n, quota_n):
        if ttl_n > 0:
            swept.set()

    sweeper = StorageSweeper(
        st,
        interval_s=0.05,
        max_age_s=10,
        on_sweep=on_sweep,
    )
    sweeper.start()
    try:
        assert swept.wait(timeout=3.0), "sweeper should have run TTL eviction by now"
    finally:
        sweeper.stop()

    assert st.n_slots() == 1
    assert not os.path.exists(paths[0])
    assert not os.path.exists(paths[1])
    assert os.path.exists(paths[2])
    assert sweeper.sweeps_run >= 1
    assert sweeper.last_ttl_evicted == 2


def test_sweeper_enforces_byte_quota_lru(tmp_path):
    """Background sweeper should enforce a byte budget by evicting
    LRU slots until under."""
    from gms_kv_ring.daemon.storage_tier import StorageSweeper, StorageTier

    st = StorageTier(str(tmp_path / "storage"))
    payload = b"q" * 1024
    src = (ctypes.c_ubyte * 1024).from_buffer_copy(payload)
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    for off in (0, 1024, 2048, 3072):
        st.demote("eng", 0, off, ctypes.addressof(src), 1024, crc)
    st._slots[("eng", 0, 0)].mtime = 1.0  # oldest
    st._slots[("eng", 0, 1024)].mtime = 2.0
    st._slots[("eng", 0, 2048)].mtime = 3.0
    st._slots[("eng", 0, 3072)].mtime = 4.0  # newest
    assert st.total_bytes() == 4096

    seen = threading.Event()

    def on_sweep(_ttl, _per_eng, quota_n):
        if quota_n > 0:
            seen.set()

    # Quota of 2 KB => 2 evictions expected.
    sweeper = StorageSweeper(
        st,
        interval_s=0.05,
        max_bytes=2048,
        on_sweep=on_sweep,
    )
    sweeper.start()
    try:
        assert seen.wait(timeout=3.0)
    finally:
        sweeper.stop()
    assert st.total_bytes() <= 2048
    # The two oldest must be gone.
    assert st.get("eng", 0, 0) is None
    assert st.get("eng", 0, 1024) is None
    # The two newest must remain.
    assert st.get("eng", 0, 2048) is not None
    assert st.get("eng", 0, 3072) is not None


def test_sweeper_stops_promptly_on_signal(tmp_path):
    """stop() must return well within one sweep interval — we use
    the stop_event in the wait, so signaling is immediate."""
    from gms_kv_ring.daemon.storage_tier import StorageSweeper, StorageTier

    st = StorageTier(str(tmp_path / "storage"))
    # Long interval to verify we don't sit on it.
    sweeper = StorageSweeper(
        st,
        interval_s=60.0,
        max_age_s=1.0,
    )
    sweeper.start()
    time.sleep(0.05)  # let the thread enter wait()
    t0 = time.monotonic()
    sweeper.stop(timeout=2.0)
    elapsed = time.monotonic() - t0
    assert elapsed < 1.0, (
        f"stop() should return immediately on event signal; " f"took {elapsed:.2f}s"
    )


def test_sweeper_survives_sweep_failure(tmp_path):
    """If the underlying StorageTier raises during a sweep (e.g.,
    transient disk issue), the sweeper must not crash — it should
    log and try again next interval. Otherwise a one-off glitch
    leaves storage growing silently forever."""
    from gms_kv_ring.daemon.storage_tier import StorageSweeper, StorageTier

    st = StorageTier(str(tmp_path / "storage"))
    payload = b"r" * 128
    src = (ctypes.c_ubyte * 128).from_buffer_copy(payload)
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    st.demote("eng", 0, 0, ctypes.addressof(src), 128, crc)
    st._slots[("eng", 0, 0)].mtime = 0.0  # ancient → TTL-eligible

    # Patch _sweep_once to raise on the first call, succeed after.
    orig = st._sweep_once
    calls = {"n": 0}

    def flaky(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("synthetic disk hiccup")
        return orig(*a, **kw)

    st._sweep_once = flaky

    sweeper = StorageSweeper(
        st,
        interval_s=0.05,
        max_age_s=1.0,
    )
    sweeper.start()
    try:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if st.n_slots() == 0:
                break
            time.sleep(0.02)
    finally:
        sweeper.stop()
    assert st.n_slots() == 0, (
        "sweeper should recover from the first sweep's exception "
        "and successfully evict on a later sweep"
    )
    assert calls["n"] >= 2


def test_sweeper_disabled_when_interval_zero(tmp_path):
    """Daemon with default sweeper_interval_s=0 must not start a
    sweeper thread — opt-in feature, not on by default."""
    from gms_kv_ring.daemon.server import Daemon

    d = Daemon(str(tmp_path / "x.sock"), storage_dir=str(tmp_path / "storage"))
    assert d._sweeper is None


def test_sweeper_disabled_when_no_policy(tmp_path):
    """Interval set but no policy => no sweeper. Otherwise we'd
    burn a thread doing nothing."""
    from gms_kv_ring.daemon.server import Daemon

    d = Daemon(
        str(tmp_path / "x.sock"),
        storage_dir=str(tmp_path / "storage"),
        sweeper_interval_s=0.05,
    )
    assert d._sweeper is None


def test_sweeper_integration_via_daemon_serve(tmp_path):
    """Full Daemon-with-sweeper lifecycle: configure, serve, observe
    eviction happen, stop cleanly. Also verifies that metrics
    increment from background sweeps (not just explicit prune_storage
    calls)."""
    torch.cuda.init()
    _ = torch.empty(1, device="cuda")
    from gms_kv_ring.common import metrics
    from gms_kv_ring.daemon.client import DaemonClient
    from gms_kv_ring.engines.handle import GMSKvRing

    sock = str(tmp_path / "d.sock")
    storage_dir = str(tmp_path / "storage")
    d, t, lh = _spawn_daemon_with_sweeper(
        sock,
        storage_dir,
        sweeper_interval_s=0.05,
        sweeper_max_age_s=10.0,  # TTL active
    )
    try:
        assert (
            d._sweeper is not None and d._sweeper._thread is not None
        ), "sweeper should be running"

        layer_size, stride = 4096, 1024
        dev = torch.arange(layer_size, dtype=torch.uint8, device="cuda")
        torch.cuda.synchronize()
        layers = [
            {
                "layer_idx": 0,
                "va": int(dev.data_ptr()),
                "size": layer_size,
                "stride": stride,
            }
        ]
        eid = f"sweep-{uuid.uuid4().hex[:6]}"
        h = GMSKvRing(engine_id=eid, daemon_socket=sock, layers=layers)
        c = DaemonClient(sock)
        try:
            # Spill + demote one block.
            res = h.record_evict(block_id=0, ranges=[(0, stride, 0)])
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline and not h.evict_succeeded(*res):
                time.sleep(0.02)
            assert h.evict_succeeded(*res)
            assert c.demote_to_storage(eid, 0, 0)
            assert d.storage_tier.n_slots() == 1

            # Backdate the slot so the sweeper's TTL hits it.
            d.storage_tier._slots[(eid, 0, 0)].mtime = 0.0

            before = metrics.storage_tier_evictions.get(reason="ttl")
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                if d.storage_tier.n_slots() == 0:
                    break
                time.sleep(0.02)
            assert (
                d.storage_tier.n_slots() == 0
            ), "background sweeper should have evicted the stale slot"
            after = metrics.storage_tier_evictions.get(reason="ttl")
            assert after >= before + 1, (
                "TTL-eviction metric should bump from the sweep, "
                f"before={before} after={after}"
            )
        finally:
            c.close()
            h.close()
    finally:
        lh["loop"].call_soon_threadsafe(d.stop)
        t.join(timeout=3)

    # Sweeper thread must be gone after daemon shutdown.
    assert d._sweeper is not None
    assert d._sweeper._thread is None, "daemon shutdown must stop the sweeper thread"


# ---------------------------------------------------------------------------
# Per-engine quotas
# ---------------------------------------------------------------------------


def test_storage_tier_per_engine_quota_isolates_noisy_engine(tmp_path):
    """One engine fills storage with many slots; another has just one
    small slot. With a per-engine quota, only the noisy engine's
    oldest slots are evicted — the quiet engine is untouched even
    though some of its data is older than the noisy engine's
    surviving data. This is the noisy-neighbor isolation invariant."""
    from gms_kv_ring.daemon.storage_tier import StorageTier

    st = StorageTier(str(tmp_path / "storage"))
    # Quiet engine: one OLD slot, 1 KB.
    quiet = _write_slot(st, "quiet", 0, 0, 1024)
    st._slots[("quiet", 0, 0)].mtime = 1.0
    # Noisy engine: four newer slots, 1 KB each.
    noisy = []
    for off in (0, 1024, 2048, 3072):
        noisy.append(_write_slot(st, "noisy", 0, off, 1024))
    st._slots[("noisy", 0, 0)].mtime = 100.0
    st._slots[("noisy", 0, 1024)].mtime = 200.0
    st._slots[("noisy", 0, 2048)].mtime = 300.0
    st._slots[("noisy", 0, 3072)].mtime = 400.0

    # Per-engine cap = 2 KB. Noisy engine has 4 KB → evict 2 of its 4.
    # Quiet engine has 1 KB → untouched (even though it's the oldest).
    n = st.enforce_per_engine_byte_quota(max_bytes_per_engine=2048)
    assert n == 2, f"expected 2 evictions in noisy engine; got {n}"

    assert os.path.exists(
        quiet.path
    ), "quiet engine's old slot must survive — per-engine isolation"
    assert not os.path.exists(noisy[0].path), "noisy oldest must go"
    assert not os.path.exists(noisy[1].path), "noisy 2nd-oldest must go"
    assert os.path.exists(noisy[2].path), "noisy 3rd-oldest must survive"
    assert os.path.exists(noisy[3].path), "noisy newest must survive"


def test_storage_tier_per_engine_quota_noop_when_under(tmp_path):
    """If every engine is already under its cap, no eviction."""
    from gms_kv_ring.daemon.storage_tier import StorageTier

    st = StorageTier(str(tmp_path / "storage"))
    _write_slot(st, "a", 0, 0, 512)
    _write_slot(st, "b", 0, 0, 512)
    assert st.enforce_per_engine_byte_quota(max_bytes_per_engine=4096) == 0
    assert st.n_slots() == 2


def test_storage_tier_stats_includes_bytes_by_engine(tmp_path):
    """stats() must surface per-engine bytes so an operator can size
    the cap intelligently."""
    from gms_kv_ring.daemon.storage_tier import StorageTier

    st = StorageTier(str(tmp_path / "storage"))
    _write_slot(st, "a", 0, 0, 1024)
    _write_slot(st, "a", 0, 1024, 256)
    _write_slot(st, "b", 0, 0, 512)
    s = st.stats()
    assert s["bytes_by_engine"] == {"a": 1280, "b": 512}


def test_sweeper_enforces_per_engine_quota(tmp_path):
    """Background sweeper drives per-engine eviction. Independent of
    global quota — even if no global cap is set, the per-engine cap
    still fires."""
    from gms_kv_ring.daemon.storage_tier import StorageSweeper, StorageTier

    st = StorageTier(str(tmp_path / "storage"))
    # Two engines: A is fine, B is over the cap.
    _write_slot(st, "A", 0, 0, 1024)
    st._slots[("A", 0, 0)].mtime = 50.0  # ancient but small
    for off in (0, 1024, 2048):
        _write_slot(st, "B", 0, off, 1024)
    st._slots[("B", 0, 0)].mtime = 100.0
    st._slots[("B", 0, 1024)].mtime = 200.0
    st._slots[("B", 0, 2048)].mtime = 300.0

    seen = threading.Event()

    def on_sweep(_ttl, per_eng_n, _quota):
        if per_eng_n > 0:
            seen.set()

    sweeper = StorageSweeper(
        st,
        interval_s=0.05,
        max_bytes_per_engine=1500,  # B has 3 KB → drop 2 of its 3
        on_sweep=on_sweep,
    )
    sweeper.start()
    try:
        assert seen.wait(timeout=3.0)
    finally:
        sweeper.stop()
    assert (
        st.get("A", 0, 0) is not None
    ), "A is under the per-engine cap — untouched even though oldest"
    assert st.get("B", 0, 0) is None
    assert st.get("B", 0, 1024) is None
    assert st.get("B", 0, 2048) is not None


def test_prune_storage_rpc_supports_per_engine_quota(tmp_path):
    """RPC plumbing: max_bytes_per_engine flows through DaemonClient
    → server.prune_storage → StorageTier.enforce_per_engine_byte_quota,
    and metrics tag the eviction with reason=per_engine_quota."""
    torch.cuda.init()
    _ = torch.empty(1, device="cuda")
    from gms_kv_ring.common import metrics
    from gms_kv_ring.daemon.client import DaemonClient
    from gms_kv_ring.engines.handle import GMSKvRing

    sock = str(tmp_path / "d.sock")
    storage_dir = str(tmp_path / "storage")
    d, t, lh = _spawn_daemon(sock, storage_dir)
    try:
        layer_size, stride = 4096, 1024
        dev = torch.arange(layer_size, dtype=torch.uint8, device="cuda")
        torch.cuda.synchronize()
        layers = [
            {
                "layer_idx": 0,
                "va": int(dev.data_ptr()),
                "size": layer_size,
                "stride": stride,
            }
        ]
        eid = f"pe-{uuid.uuid4().hex[:6]}"
        h = GMSKvRing(engine_id=eid, daemon_socket=sock, layers=layers)
        c = DaemonClient(sock)
        try:
            # Spill + demote 3 blocks for the engine.
            for off in (0, 1024, 2048):
                res = h.record_evict(
                    block_id=off // stride,
                    ranges=[(0, stride, off)],
                )
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline and not h.evict_succeeded(*res):
                    time.sleep(0.02)
                assert h.evict_succeeded(*res)
                assert c.demote_to_storage(eid, 0, off)

            # Backdate mtimes to make LRU order deterministic.
            d.storage_tier._slots[(eid, 0, 0)].mtime = 1.0
            d.storage_tier._slots[(eid, 0, 1024)].mtime = 2.0
            d.storage_tier._slots[(eid, 0, 2048)].mtime = 3.0

            stats = c.storage_stats()
            assert stats["bytes_by_engine"][eid] == 3 * stride

            before = metrics.storage_tier_evictions.get(
                reason="per_engine_quota",
            )
            # Cap at exactly 1 slot's worth → evict 2.
            n = c.prune_storage(max_bytes_per_engine=stride)
            assert n == 2, f"expected 2 evictions; got {n}"
            after = metrics.storage_tier_evictions.get(
                reason="per_engine_quota",
            )
            assert after == before + 2
            stats = c.storage_stats()
            assert stats["n_slots"] == 1
            assert stats["bytes_by_engine"][eid] == stride
        finally:
            c.close()
            h.close()
    finally:
        lh["loop"].call_soon_threadsafe(d.stop)
        t.join(timeout=3)
