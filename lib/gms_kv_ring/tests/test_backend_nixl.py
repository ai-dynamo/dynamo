# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real-NIXL tests for NixlBackend (POSIX plugin).

These tests exercise the actual `nixl_agent` data plane:
register memory → initialize_xfer → transfer → poll → release.
The whole suite is skipped on machines without `nixl` importable
so the gms_kv_ring suite still passes on a bare host.

Covered:
  - End-to-end round trip: demote pinned host bytes → file →
    promote → verify byte equality and CRC
  - CRC mismatch detected if the file is tampered after demote
  - Reconcile reads back files written by a previous instance
  - Cleanup primitives (TTL prune, LRU quota, per-engine quota)
    behave the same as the Local backend
  - Cross-backend compatibility: a file written by `StorageTier`
    is readable by `NixlBackend` (same 24-byte CRC header)
"""

from __future__ import annotations

import ctypes
import os
import time
import zlib

import pytest
from gms_kv_ring.daemon.backends_nixl import NixlBackend

if not NixlBackend.is_available():
    pytest.skip(
        "nixl not importable in this environment",
        allow_module_level=True,
    )


def _pinned(payload: bytes) -> tuple[int, "ctypes.Array"]:
    """Allocate a ctypes buffer holding `payload`, return (addr, buf).
    Caller MUST keep the buf reference alive for the test's duration."""
    buf = (ctypes.c_ubyte * len(payload)).from_buffer_copy(payload)
    return ctypes.addressof(buf), buf


def test_nixl_backend_demote_promote_round_trip(tmp_path):
    """Write a payload via NIXL, read it back, verify bytes + CRC."""
    nb = NixlBackend(str(tmp_path / "store"), plugin="POSIX")
    payload = bytes((i * 31 & 0xFF) for i in range(4096))
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    src_addr, _src_buf = _pinned(payload)

    slot = nb.demote("eng-A", 0, 0, host_ptr=src_addr, size=len(payload), crc=crc)
    assert slot.size == len(payload)
    assert slot.crc == crc
    assert os.path.exists(slot.path)

    dst_buf = (ctypes.c_ubyte * len(payload))()
    dst_addr = ctypes.addressof(dst_buf)
    verified = nb.promote("eng-A", 0, 0, dest_host_ptr=dst_addr, max_size=len(payload))
    assert verified == crc
    assert bytes(dst_buf) == payload


def test_nixl_backend_promote_detects_byte_corruption(tmp_path):
    """If someone tampers with the file after demote, promote must
    detect the CRC mismatch and return None — engine must not get
    corrupted data."""
    nb = NixlBackend(str(tmp_path / "store"), plugin="POSIX")
    payload = b"hello, world" * 200
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    src_addr, _src_buf = _pinned(payload)
    slot = nb.demote("eng", 0, 0, src_addr, len(payload), crc)

    # Flip a byte well inside the payload (past the 24-byte header).
    with open(slot.path, "r+b") as f:
        f.seek(100)
        b = f.read(1)
        f.seek(100)
        f.write(bytes([b[0] ^ 0xA5]))

    dst_buf = (ctypes.c_ubyte * len(payload))()
    assert (
        nb.promote(
            "eng",
            0,
            0,
            ctypes.addressof(dst_buf),
            len(payload),
        )
        is None
    ), "CRC mismatch must surface as promote failure"


def test_nixl_backend_reconciles_index_on_startup(tmp_path):
    """Write some files via instance #1, drop the instance,
    instance #2 picks them up from disk via reconcile."""
    storage_dir = str(tmp_path / "store")
    nb1 = NixlBackend(storage_dir, plugin="POSIX")
    payload = b"r" * 1024
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    src_addr, _src_buf = _pinned(payload)
    for off in (0, 1024, 2048):
        nb1.demote("eng", 0, off, src_addr, len(payload), crc)
    assert nb1.n_slots() == 3
    del nb1

    nb2 = NixlBackend(storage_dir, plugin="POSIX")
    assert nb2.n_slots() == 3
    # Promote still works post-reconcile.
    dst = (ctypes.c_ubyte * len(payload))()
    assert (
        nb2.promote(
            "eng",
            0,
            0,
            ctypes.addressof(dst),
            len(payload),
        )
        == crc
    )


def test_nixl_backend_files_interoperable_with_local(tmp_path):
    """File format is shared: a file written by StorageTier (Local)
    must be readable by NixlBackend. This lets operators migrate
    backends without re-spilling."""
    from gms_kv_ring.daemon.storage_tier import StorageTier

    storage_dir = str(tmp_path / "store")
    payload = bytes((i * 17 & 0xFF) for i in range(2048))
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    src_addr, _src_buf = _pinned(payload)

    # Write with Local.
    local = StorageTier(storage_dir)
    local.demote("eng", 0, 0, src_addr, len(payload), crc)
    del local

    # Read with NIXL.
    nb = NixlBackend(storage_dir, plugin="POSIX")
    assert nb.n_slots() == 1
    dst = (ctypes.c_ubyte * len(payload))()
    assert (
        nb.promote(
            "eng",
            0,
            0,
            ctypes.addressof(dst),
            len(payload),
        )
        == crc
    )
    assert bytes(dst) == payload


def test_nixl_backend_release_slot_unlinks(tmp_path):
    nb = NixlBackend(str(tmp_path / "store"), plugin="POSIX")
    payload = b"x" * 512
    src_addr, _src_buf = _pinned(payload)
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    slot = nb.demote("eng", 0, 0, src_addr, len(payload), crc)
    assert os.path.exists(slot.path)
    assert nb.release_slot("eng", 0, 0) is True
    assert not os.path.exists(slot.path)
    assert nb.n_slots() == 0


def test_nixl_release_keeps_slot_when_tombstone_fails(tmp_path, monkeypatch):
    manifest = str(tmp_path / "manifest.jsonl")
    nb = NixlBackend(str(tmp_path / "store"), plugin="POSIX", manifest_path=manifest)
    payload = b"durable" * 128
    src_addr, _src_buf = _pinned(payload)
    slot = nb.demote("eng", 0, 0, src_addr, len(payload), zlib.crc32(payload))

    def fail_tombstone(*_args):
        raise OSError("manifest fsync failed")

    monkeypatch.setattr(nb, "_append_tombstone", fail_tombstone)
    with pytest.raises(OSError, match="manifest fsync"):
        nb.release_slot("eng", 0, 0)
    assert nb.get("eng", 0, 0) is slot
    assert os.path.exists(slot.path)
    nb.close()


def test_nixl_backend_manifest_tombstone_prevents_release_resurrection(tmp_path):
    storage_dir = str(tmp_path / "store")
    manifest = str(tmp_path / "manifest.jsonl")
    payload = b"m" * 512
    src_addr, _src_buf = _pinned(payload)
    crc = zlib.crc32(payload) & 0xFFFFFFFF

    nb1 = NixlBackend(storage_dir, plugin="POSIX", manifest_path=manifest)
    nb1.demote("eng-M", 0, 0, src_addr, len(payload), crc)
    assert nb1.release_slot("eng-M", 0, 0) is True
    nb1.close()

    nb2 = NixlBackend(storage_dir, plugin="POSIX", manifest_path=manifest)
    try:
        assert nb2.n_slots() == 0
        assert nb2.get("eng-M", 0, 0) is None
    finally:
        nb2.close()


def test_nixl_backend_manifest_drops_missing_file_records(tmp_path):
    storage_dir = str(tmp_path / "store")
    manifest = tmp_path / "manifest.jsonl"
    missing_path = tmp_path / "store" / "eng" / "0" / "0.bin"
    manifest.write_text(
        '{"eid":"eng","layer":0,"offset":0,'
        f'"path":"{missing_path}","size":512,'
        '"crc":1,"mtime":1700000000.0}\n'
    )

    nb = NixlBackend(storage_dir, plugin="POSIX", manifest_path=str(manifest))
    try:
        assert nb.n_slots() == 0
        assert nb.get("eng", 0, 0) is None
    finally:
        nb.close()


def test_nixl_backend_manifest_records_quota_tombstones(tmp_path):
    storage_dir = str(tmp_path / "store")
    manifest = str(tmp_path / "manifest.jsonl")
    payload = b"q" * 512
    src_addr, _src_buf = _pinned(payload)
    crc = zlib.crc32(payload) & 0xFFFFFFFF

    nb1 = NixlBackend(storage_dir, plugin="POSIX", manifest_path=manifest)
    for off in (0, 512, 1024):
        nb1.demote("eng-Q", 0, off, src_addr, len(payload), crc)
    nb1._slots[("eng-Q", 0, 0)].mtime = 1.0
    nb1._slots[("eng-Q", 0, 512)].mtime = 2.0
    nb1._slots[("eng-Q", 0, 1024)].mtime = 3.0
    assert nb1.enforce_byte_quota(512) == 2
    nb1.close()

    nb2 = NixlBackend(storage_dir, plugin="POSIX", manifest_path=manifest)
    try:
        assert nb2.n_slots() == 1
        assert nb2.get("eng-Q", 0, 0) is None
        assert nb2.get("eng-Q", 0, 512) is None
        assert nb2.get("eng-Q", 0, 1024) is not None
    finally:
        nb2.release_engine("eng-Q")
        nb2.close()


def test_nixl_backend_cleanup_primitives(tmp_path):
    """TTL + global + per-engine quota all work on the NIXL backend
    with the same semantics as Local."""
    nb = NixlBackend(str(tmp_path / "store"), plugin="POSIX")
    payload = b"y" * 1024
    src_addr, _src_buf = _pinned(payload)
    crc = zlib.crc32(payload) & 0xFFFFFFFF

    # Four slots for engine A.
    for off in (0, 1024, 2048, 3072):
        nb.demote("A", 0, off, src_addr, len(payload), crc)
    # Backdate two of them.
    nb._slots[("A", 0, 0)].mtime = 100.0
    nb._slots[("A", 0, 1024)].mtime = 200.0
    nb._slots[("A", 0, 2048)].mtime = 300.0
    nb._slots[("A", 0, 3072)].mtime = 400.0

    # Global quota at 2KB → 2 evictions (oldest first).
    n = nb.enforce_byte_quota(2048)
    assert n == 2
    assert nb.get("A", 0, 0) is None
    assert nb.get("A", 0, 1024) is None
    assert nb.get("A", 0, 2048) is not None
    assert nb.get("A", 0, 3072) is not None

    # TTL prune: reset surviving slots to "now" so they're fresh,
    # add an ancient one, prune. (Backdated mtimes from the quota
    # step were absolute values — relative to time.time() they're
    # ancient, so we re-fresh them here to isolate the TTL test.)
    now = time.time()
    nb._slots[("A", 0, 2048)].mtime = now
    nb._slots[("A", 0, 3072)].mtime = now
    nb.demote("A", 0, 4096, src_addr, len(payload), crc)
    nb._slots[("A", 0, 4096)].mtime = 1.0  # ancient
    n_pruned = nb.prune_older_than(max_age_seconds=60.0)
    assert n_pruned == 1, f"only the ancient slot should be pruned; got {n_pruned}"
    assert nb.get("A", 0, 4096) is None
    assert nb.get("A", 0, 2048) is not None
    assert nb.get("A", 0, 3072) is not None


def test_nixl_backend_with_supervisor(tmp_path):
    """NixlBackend works behind BackendSupervisor exactly like the
    Local backend — the supervisor cares about the protocol, not the
    transport."""
    from gms_kv_ring.daemon.supervisor import BackendSupervisor

    nb = NixlBackend(str(tmp_path / "store"), plugin="POSIX")
    sup = BackendSupervisor(
        nb,
        factory=lambda: NixlBackend(
            str(tmp_path / "store"),
            plugin="POSIX",
        ),
    )
    payload = b"z" * 256
    src_addr, _src_buf = _pinned(payload)
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    slot = sup.demote("eng", 0, 0, src_addr, len(payload), crc)
    assert slot is not None
    assert sup.n_slots() == 1


def test_nixl_backend_rejects_unknown_plugin(tmp_path):
    """Specifying a plugin NIXL doesn't have must raise at
    construction with a clear list of what IS available."""
    with pytest.raises(RuntimeError, match="not in available plugins"):
        NixlBackend(
            str(tmp_path / "store"),
            plugin="NOT_A_REAL_PLUGIN",
        )


# ---------------------------------------------------------------------------
# OBJ plugin (S3-compatible object storage)
# ---------------------------------------------------------------------------


def test_nixl_obj_rejects_construction_without_bucket(tmp_path, monkeypatch):
    """OBJ plugin REQUIRES a bucket. Without one (no kwarg, no env)
    the constructor must raise a clear error pointing at the three
    knobs operators can set."""
    monkeypatch.delenv("AWS_DEFAULT_BUCKET", raising=False)
    with pytest.raises(RuntimeError, match="bucket"):
        NixlBackend(
            str(tmp_path / "store"),
            plugin="OBJ",
            manifest_path=str(tmp_path / "m.jsonl"),
        )


def test_nixl_obj_rejects_construction_without_manifest(
    tmp_path,
    monkeypatch,
):
    """OBJ has no list-objects API → restart-replay needs a
    manifest. Constructing without one must fail loudly."""
    monkeypatch.setenv("AWS_DEFAULT_BUCKET", "test-bucket")
    with pytest.raises(RuntimeError, match="manifest_path"):
        NixlBackend(
            str(tmp_path / "store"),
            plugin="OBJ",
            # missing manifest_path
        )


def test_nixl_obj_picks_up_bucket_from_env(tmp_path, monkeypatch):
    """If AWS_DEFAULT_BUCKET is set in the env AND we don't pass an
    explicit bucket, construction proceeds. (We can't test the full
    construction here without a live S3 endpoint — NIXL will try to
    talk to it and fail. We test up to the point where it'd try.)

    Concretely: validate the error message changes from 'bucket'
    (our pre-flight) to a NIXL-side error when env is set."""
    monkeypatch.setenv("AWS_DEFAULT_BUCKET", "test-bucket")
    try:
        NixlBackend(
            str(tmp_path / "store"),
            plugin="OBJ",
            manifest_path=str(tmp_path / "m.jsonl"),
        )
    except RuntimeError as exc:
        # NIXL error from the actual OBJ backend trying to connect
        # to S3 is fine — what we care about is NOT seeing our
        # pre-flight bucket check fail.
        assert "bucket" not in str(exc).lower(), (
            f"expected bucket pre-flight to pass with env set; " f"got error: {exc}"
        )
    except Exception:  # noqa: BLE001
        # NIXL plugin-level errors (no S3 endpoint, etc.) — also fine.
        pass


def test_nixl_obj_explicit_bucket_overrides_env(tmp_path, monkeypatch):
    """An explicit `bucket=` kwarg overrides AWS_DEFAULT_BUCKET.
    Verify by checking that without env BUT WITH the kwarg, our
    pre-flight passes."""
    monkeypatch.delenv("AWS_DEFAULT_BUCKET", raising=False)
    try:
        NixlBackend(
            str(tmp_path / "store"),
            plugin="OBJ",
            bucket="explicit-bucket-xyz",
            manifest_path=str(tmp_path / "m.jsonl"),
        )
    except RuntimeError as exc:
        assert "bucket" not in str(exc).lower(), (
            f"explicit bucket kwarg should satisfy pre-flight; " f"got: {exc}"
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# GDS plugin (GPUDirect Storage)
# ---------------------------------------------------------------------------


def test_nixl_gds_construction_succeeds(tmp_path):
    """GDS plugin instantiates without errors on any host where
    NIXL is installed — the plugin's libcufile dependency comes
    along with the NIXL Python install. Actual GDS data plane
    requires a cuFile-enabled mount; we test construction here,
    data plane separately env-gated."""
    try:
        nb = NixlBackend(str(tmp_path / "store"), plugin="GDS")
    except Exception as exc:  # noqa: BLE001
        if exc.__class__.__name__ == "nixlBackendError":
            pytest.skip(f"NIXL GDS backend unavailable: {exc}")
        raise
    assert nb.plugin == "GDS"
    # File-walk reconcile runs (same shape as POSIX); empty dir → 0.
    assert nb.n_slots() == 0


def test_nixl_gds_demote_from_gpu_rejects_wrong_plugin(tmp_path):
    """demote_from_gpu / promote_into_gpu only work for GDS plugins
    — POSIX/OBJ raise immediately with a clear error. Same for the
    deferred-CRC pair."""
    nb = NixlBackend(str(tmp_path / "store"), plugin="POSIX")
    with pytest.raises(NotImplementedError, match="GDS"):
        nb.demote_from_gpu("eng", 0, 0, gpu_ptr=0, size=0, crc=0)
    with pytest.raises(NotImplementedError, match="GDS"):
        nb.promote_into_gpu("eng", 0, 0, dest_gpu_ptr=0, max_size=0)
    with pytest.raises(NotImplementedError, match="GDS"):
        nb.demote_from_gpu_deferred_crc(
            "eng",
            0,
            0,
            gpu_ptr=0,
            size=0,
        )


@pytest.mark.skipif(
    not os.environ.get("GMS_KVR_TEST_GDS_FAILS_LOUDLY"),
    reason="Triggering cuFile registration failure leaves the "
    "process-global cuFile state in a way that breaks later "
    "tests (Mooncake master fixture in the same process "
    "starts dumping core). Opt in via "
    "GMS_KVR_TEST_GDS_FAILS_LOUDLY=1 when investigating GDS "
    "configuration; not safe to run in the default suite.",
)
def test_nixl_gds_demote_from_gpu_fails_loudly_on_non_gds_mount(
    tmp_path,
):
    """On a non-GDS mount (the typical test path), cuFile's
    `file register` step fails — we want this to propagate as a
    clear NIXL_ERR_BACKEND, not silently route through a slower
    fallback path that masks the configuration error.

    This test documents the runtime contract: if your storage
    isn't GDS-capable, GDS demote fails fast.

    CAVEAT: triggering this failure once leaves cuFile in a state
    where other CUDA libraries (Mooncake's transfer engine) abort
    when initialized later in the same process. We've reproduced
    this with the Mooncake master fixture — the master subprocess
    is fine, but any in-process cuFile-touching code that runs
    after this test crashes. So the test is opt-in only."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    nb = NixlBackend(str(tmp_path / "store"), plugin="GDS")
    t = torch.zeros(1024, dtype=torch.uint8, device="cuda")
    import zlib

    crc = zlib.crc32(b"\x00" * 1024) & 0xFFFFFFFF
    with pytest.raises(Exception):
        nb.demote_from_gpu(
            "eng",
            0,
            0,
            gpu_ptr=int(t.data_ptr()),
            size=1024,
            crc=crc,
        )


# Real GDS data-plane tests need a cuFile-enabled mount
# (NVMe with proper kernel/driver, BeeGFS, Lustre, etc.). Gate
# on an env var pointing to such a path.
_GDS_PATH = os.environ.get("GMS_KVR_NIXL_GDS_PATH")
_GDS_AVAILABLE = bool(_GDS_PATH) and os.path.isdir(_GDS_PATH or "/")


@pytest.mark.skipif(
    not _GDS_AVAILABLE,
    reason="Set GMS_KVR_NIXL_GDS_PATH to a cuFile-capable directory "
    "to exercise real GDS data-plane tests",
)
def test_nixl_gds_round_trip_real_mount():
    """End-to-end GPU-direct write+read on a real GDS-capable
    mount. Skipped without the env var."""
    import zlib

    import torch

    nb = NixlBackend(_GDS_PATH, plugin="GDS")
    try:
        payload_bytes = bytes((i * 53 & 0xFF) for i in range(4096))
        crc = zlib.crc32(payload_bytes) & 0xFFFFFFFF
        src_t = torch.frombuffer(
            bytearray(payload_bytes),
            dtype=torch.uint8,
        ).cuda()
        nb.demote_from_gpu(
            "eng-GDS",
            0,
            0,
            gpu_ptr=int(src_t.data_ptr()),
            size=len(payload_bytes),
            crc=crc,
        )
        dst_t = torch.zeros(
            len(payload_bytes),
            dtype=torch.uint8,
            device="cuda",
        )
        verified = nb.promote_into_gpu(
            "eng-GDS",
            0,
            0,
            dest_gpu_ptr=int(dst_t.data_ptr()),
            max_size=len(payload_bytes),
        )
        assert verified == crc
        assert torch.equal(src_t, dst_t)
    finally:
        nb.release_engine("eng-GDS")


@pytest.mark.skipif(
    not _GDS_AVAILABLE,
    reason="Set GMS_KVR_NIXL_GDS_PATH to exercise the daemon's "
    "promote_storage_to_hbm dest_offset path against real GDS",
)
def test_nixl_gds_promote_to_hbm_with_separate_dest_offset(tmp_path):
    """End-to-end check that the daemon's `dest_offset` plumbing
    works against a real GDS mount.

    The connector's restore-on-hit path issues
    `promote_storage_to_hbm(layer, src_offset, size,
    dest_offset=dst_offset)` with src_offset != dst_offset. The
    backend's `promote_into_gpu` already separates the storage key
    from the dest GPU pointer (see daemon/server.py:1048), but the
    fake-backend tests don't exercise the real cuFile write into
    an arbitrary dest VA. This test does.
    """
    import asyncio
    import os
    import threading
    import time

    import torch
    from gms_kv_ring.daemon.server import Daemon
    from gms_kv_ring.engines.handle import GMSKvRing

    backend = NixlBackend(_GDS_PATH, plugin="GDS")
    sock = str(tmp_path / "gds-remap.sock")

    daemon = Daemon(
        sock,
        storage_backend=backend,
        supervise_backend=False,
    )
    holder = {}

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
    while time.monotonic() < deadline and not os.path.exists(sock):
        time.sleep(0.02)

    try:
        block_bytes = 4096
        n_blocks = 4
        dev = torch.zeros(
            n_blocks * block_bytes,
            dtype=torch.uint8,
            device="cuda",
        )
        torch.cuda.synchronize()
        base = int(dev.data_ptr())
        h = GMSKvRing(
            engine_id="eng-GDS-remap",
            daemon_socket=sock,
            layers=[
                {
                    "layer_idx": 0,
                    "va": base,
                    "size": n_blocks * block_bytes,
                    "stride": block_bytes,
                }
            ],
        )
        try:
            # Demote block 1: src_offset = 1 * 4096.
            src_offset = 1 * block_bytes
            payload = bytes((i * 37 & 0xFF) for i in range(block_bytes))
            src_tensor = torch.frombuffer(
                bytearray(payload),
                dtype=torch.uint8,
            ).cuda()
            dev[src_offset : src_offset + block_bytes].copy_(src_tensor)
            torch.cuda.synchronize()
            assert h.demote_hbm_to_storage(0, src_offset, block_bytes)

            # Zero the destination region so the restore must
            # actually write bytes there.
            dev.zero_()
            torch.cuda.synchronize()

            # Promote with dest_offset != offset: storage key uses
            # src_offset, HBM dest uses 3 * block_bytes.
            dst_offset = 3 * block_bytes
            ok = h.promote_storage_to_hbm(
                0,
                src_offset,
                block_bytes,
                dest_offset=dst_offset,
            )
            assert ok is True

            got_at_dst = bytes(
                dev[dst_offset : dst_offset + block_bytes].cpu().numpy().tobytes()
            )
            assert got_at_dst == payload
            # Source region must NOT have been touched (it's the
            # cuFile READ destination that matters, not the
            # original demote source).
            got_at_src = bytes(
                dev[src_offset : src_offset + block_bytes].cpu().numpy().tobytes()
            )
            assert got_at_src == b"\x00" * block_bytes
        finally:
            h.close()
    finally:
        holder["loop"].call_soon_threadsafe(daemon.stop)
        t.join(timeout=3)
        backend.release_engine("eng-GDS-remap")


# Real OBJ data-plane tests need an S3-compatible endpoint
# (MinIO / AWS / Azure Blob). Gate on a runtime env var.

_OBJ_BUCKET = os.environ.get("GMS_KVR_NIXL_OBJ_BUCKET")
_OBJ_AVAILABLE = bool(_OBJ_BUCKET) and bool(os.environ.get("AWS_ACCESS_KEY_ID"))


@pytest.mark.skipif(
    not _OBJ_AVAILABLE,
    reason="Set GMS_KVR_NIXL_OBJ_BUCKET + AWS_* env to exercise S3 "
    "data-plane tests against a real endpoint",
)
def test_nixl_obj_demote_promote_round_trip(tmp_path):
    """End-to-end against a real S3-compatible bucket: demote a
    payload, promote it back, verify bytes + CRC. Skipped without
    a configured bucket — the bucket env var is what gates this."""
    manifest = str(tmp_path / "m.jsonl")
    nb = NixlBackend(
        str(tmp_path / "store"),
        plugin="OBJ",
        bucket=_OBJ_BUCKET,
        manifest_path=manifest,
    )
    try:
        import zlib

        payload = bytes((i * 47 & 0xFF) for i in range(2048))
        crc = zlib.crc32(payload) & 0xFFFFFFFF
        src_addr, _src_buf = _pinned(payload)
        slot = nb.demote(
            "eng-OBJ",
            0,
            0,
            src_addr,
            len(payload),
            crc,
        )
        assert slot.size == len(payload)

        dst_buf = (ctypes.c_ubyte * len(payload))()
        verified = nb.promote(
            "eng-OBJ",
            0,
            0,
            ctypes.addressof(dst_buf),
            len(payload),
        )
        assert verified == crc
        assert bytes(dst_buf) == payload
    finally:
        nb.release_engine("eng-OBJ")
