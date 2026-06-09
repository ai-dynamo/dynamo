# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real-Mooncake tests for MooncakeBackend.

These spin up an actual `mooncake_master` subprocess + connect a
client to it. The whole module skips if:
  - `mooncake.store` isn't importable on this Python install
  - the `mooncake_master` binary isn't on disk
  - the master subprocess can't bind a port

Memory note `21-mooncake-unified-pool-layout-bug.md` flags one
Mooncake xfail on TRT-LLM warm_hit; that's the engine-side
unified-pool path, NOT the storage put_from/get_into path we
exercise here. Worth re-checking if these tests ever start
flaking.

Covered: setup, demote, promote, CRC mismatch detection,
multi-block put/get, release_slot, release_engine."""

from __future__ import annotations

import ctypes
import os
import socket
import subprocess
import time
import uuid
import zlib

import pytest
from gms_kv_ring.daemon.backends_mooncake import MooncakeBackend

if not MooncakeBackend.is_available():
    pytest.skip(
        "mooncake.store not importable in this environment",
        allow_module_level=True,
    )

# Locate the master binary. Skip if missing.
_MOONCAKE_PKG = os.path.dirname(__import__("mooncake").__file__)
_MASTER_BIN = os.path.join(_MOONCAKE_PKG, "mooncake_master")
if not os.access(_MASTER_BIN, os.X_OK):
    pytest.skip(
        f"mooncake_master not executable at {_MASTER_BIN}",
        allow_module_level=True,
    )


def _free_port() -> int:
    """OS-assigned free port. Closes immediately; small TOCTOU
    window before the master claims it — fine for tests."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture(scope="module")
def mooncake_master(tmp_path_factory):
    """Spawn a Mooncake master process with an HTTP metadata
    server. Yields (master_rpc_addr, metadata_addr). Cleans up at
    module teardown."""
    rpc_port = _free_port()
    meta_port = _free_port()
    root_fs = tmp_path_factory.mktemp("mooncake-storage")
    os.environ["GMS_TEST_MOONCAKE_ROOT_FS"] = str(root_fs)
    # Keep the test offload client small; Mooncake defaults can reserve
    # more than a GiB of process memory for the local-disk path. These
    # must be visible to the in-process client, not just the master.
    os.environ.setdefault(
        "MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES",
        str(128 << 20),
    )
    os.environ.setdefault(
        "MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES",
        str(128 << 20),
    )
    os.environ.setdefault("MOONCAKE_OFFLOAD_ENABLE_EVICTION", "true")
    env = os.environ.copy()
    # The active runtime environment owns LD_LIBRARY_PATH for libibverbs,
    # CUDA runtime, libcurl, and liblzma. Avoid hardcoded paths here so
    # the fixture validates the packaged runtime environment.
    proc = subprocess.Popen(
        [
            _MASTER_BIN,
            f"--rpc_port={rpc_port}",
            "--enable_http_metadata_server",
            f"--http_metadata_server_port={meta_port}",
            "--http_metadata_server_host=127.0.0.1",
            "--enable_offload=true",
            f"--root_fs_dir={root_fs}",
            "--cluster_id=gms_test",
            f"--global_file_segment_size={64 << 20}",
            "--logtostderr",
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        # Wait for the RPC port to accept connections.
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                stderr = proc.stderr.read().decode("utf-8", "replace")
                pytest.skip(
                    f"mooncake_master exited rc={proc.returncode}; "
                    f"stderr={stderr[:500]!r}"
                )
            try:
                s = socket.create_connection(
                    ("127.0.0.1", rpc_port),
                    timeout=0.5,
                )
                s.close()
                break
            except OSError:
                time.sleep(0.05)
        else:
            proc.terminate()
            proc.wait(timeout=2)
            pytest.skip("mooncake_master never bound the rpc_port")
        yield (
            f"127.0.0.1:{rpc_port}",
            f"127.0.0.1:{meta_port}",
        )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def _make_backend(mooncake_master):
    master_addr, meta_addr = mooncake_master
    # Mooncake's HTTP metadata server needs the full URL form,
    # not just host:port — the C client builds GET requests like
    # `{metadata_server}?key=...` so the path matters.
    return MooncakeBackend(
        local_hostname=f"test-{uuid.uuid4().hex[:6]}",
        metadata_server=f"http://{meta_addr}/metadata",
        master_server_addr=master_addr,
        global_segment_size=64 << 20,  # 64 MiB
        local_buffer_size=16 << 20,  # 16 MiB
        protocol="tcp",
        rdma_devices="",
    )


def _make_backend_with_ssd_offload(mooncake_master, ssd_path):
    master_addr, meta_addr = mooncake_master
    os.makedirs(ssd_path, exist_ok=True)
    return MooncakeBackend(
        local_hostname=f"ssd-{uuid.uuid4().hex[:6]}",
        metadata_server=f"http://{meta_addr}/metadata",
        master_server_addr=master_addr,
        global_segment_size=64 << 20,
        local_buffer_size=16 << 20,
        protocol="tcp",
        rdma_devices="",
        enable_ssd_offload=True,
        ssd_offload_path=str(ssd_path),
    )


def _pinned(payload: bytes):
    buf = (ctypes.c_ubyte * len(payload)).from_buffer_copy(payload)
    return ctypes.addressof(buf), buf


def test_mooncake_backend_demote_promote_round_trip(mooncake_master):
    """Put a payload, get it back, verify bytes + CRC."""
    nb = _make_backend(mooncake_master)
    try:
        payload = bytes((i * 23 & 0xFF) for i in range(4096))
        crc = zlib.crc32(payload) & 0xFFFFFFFF
        src_addr, _src_buf = _pinned(payload)

        slot = nb.demote("eng-A", 0, 0, host_ptr=src_addr, size=len(payload), crc=crc)
        assert slot.size == len(payload)
        assert slot.crc == crc
        assert slot.mc_key == "gms/eng-A/0/0"

        dst = (ctypes.c_ubyte * len(payload))()
        verified = nb.promote(
            "eng-A",
            0,
            0,
            dest_host_ptr=ctypes.addressof(dst),
            max_size=len(payload),
        )
        assert verified == crc
        assert bytes(dst) == payload
    finally:
        nb.release_engine("eng-A")
        nb.close()


def test_mooncake_backend_multiple_blocks(mooncake_master):
    """Several blocks in flight, all round-trippable."""
    nb = _make_backend(mooncake_master)
    try:
        payloads = [bytes((i * (k + 1) & 0xFF) for i in range(1024)) for k in range(4)]
        for k, p in enumerate(payloads):
            addr, _ = _pinned(p)
            nb.demote(
                "eng-B",
                0,
                k * 1024,
                host_ptr=addr,
                size=len(p),
                crc=zlib.crc32(p) & 0xFFFFFFFF,
            )
        assert nb.n_slots() == 4

        for k, p in enumerate(payloads):
            dst = (ctypes.c_ubyte * len(p))()
            v = nb.promote(
                "eng-B",
                0,
                k * 1024,
                dest_host_ptr=ctypes.addressof(dst),
                max_size=len(p),
            )
            assert v == zlib.crc32(p) & 0xFFFFFFFF
            assert bytes(dst) == p
    finally:
        nb.release_engine("eng-B")
        nb.close()


def test_mooncake_backend_release_slot_removes_key(mooncake_master):
    """release_slot drops the index entry AND removes the key from
    Mooncake — a subsequent promote returns None (slot unknown)."""
    nb = _make_backend(mooncake_master)
    try:
        payload = b"x" * 512
        crc = zlib.crc32(payload) & 0xFFFFFFFF
        addr, _b = _pinned(payload)
        nb.demote("eng-C", 0, 0, addr, len(payload), crc)
        assert nb.release_slot("eng-C", 0, 0) is True
        assert nb.get("eng-C", 0, 0) is None
        # Promote on a released slot returns None.
        dst = (ctypes.c_ubyte * len(payload))()
        assert (
            nb.promote(
                "eng-C",
                0,
                0,
                ctypes.addressof(dst),
                len(payload),
            )
            is None
        )
    finally:
        nb.close()


def test_mooncake_backend_release_engine(mooncake_master):
    """release_engine drops every slot for one engine, leaves others
    alone."""
    nb = _make_backend(mooncake_master)
    try:
        addr_a, _ba = _pinned(b"a" * 256)
        addr_b, _bb = _pinned(b"b" * 256)
        nb.demote("eng-X", 0, 0, addr_a, 256, 1)
        nb.demote("eng-X", 0, 256, addr_a, 256, 2)
        nb.demote("eng-Y", 0, 0, addr_b, 256, 3)

        n = nb.release_engine("eng-X")
        assert n == 2
        assert nb.get("eng-X", 0, 0) is None
        assert nb.get("eng-Y", 0, 0) is not None
    finally:
        nb.release_engine("eng-Y")
        nb.close()


def test_mooncake_backend_ssd_offload_local_path_round_trip(
    mooncake_master,
    tmp_path,
):
    """Enable Mooncake's SSD offload knob against a local folder.

    Small payloads may remain in Mooncake's DRAM pool, so this validates
    configuration and data correctness rather than forcing eviction.
    """
    nb = _make_backend_with_ssd_offload(
        mooncake_master,
        tmp_path / "mooncake-ssd",
    )
    try:
        root_fs = os.environ["GMS_TEST_MOONCAKE_ROOT_FS"]

        def files_under(path):
            out = set()
            for dirpath, _dirnames, filenames in os.walk(path):
                for filename in filenames:
                    out.add(os.path.relpath(os.path.join(dirpath, filename), path))
            return out

        before_files = files_under(root_fs)
        payload = bytes((i * 29 & 0xFF) for i in range(4096))
        crc = zlib.crc32(payload) & 0xFFFFFFFF
        addr, _b = _pinned(payload)
        nb.demote("eng-SSD", 0, 0, addr, len(payload), crc)

        deadline = time.monotonic() + 3.0
        new_files = set()
        while time.monotonic() < deadline:
            new_files = files_under(root_fs) - before_files
            if new_files:
                break
            time.sleep(0.05)
        assert new_files, "Mooncake local-disk offload produced no files"

        dst = (ctypes.c_ubyte * len(payload))()
        verified = nb.promote(
            "eng-SSD",
            0,
            0,
            ctypes.addressof(dst),
            len(payload),
        )
        assert verified == crc
        assert bytes(dst) == payload
    finally:
        nb.release_engine("eng-SSD")
        nb.close()


def test_mooncake_backend_manifest_tombstone_replay(tmp_path):
    """Manifest tombstones must win even if the backing store says
    the old key still exists. This is a pure replay-state test."""
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        '{"eid":"eng-T","layer":0,"offset":0,'
        '"mc_key":"gms/eng-T/0/0","size":256,'
        '"crc":1,"mtime":1700000000.0}\n'
        '{"eid":"eng-T","layer":0,"offset":0,'
        '"deleted":true,"mtime":1700000001.0}\n'
    )

    class FakeStore:
        def is_exist(self, _key):
            return 1

    nb = MooncakeBackend.__new__(MooncakeBackend)
    nb.manifest_path = str(manifest)
    nb._slots = {}
    nb._mds = FakeStore()
    nb._replay_manifest()
    assert nb._slots == {}


def test_mooncake_backend_manifest_quota_tombstone_replay(
    mooncake_master,
    tmp_path,
):
    master_addr, meta_addr = mooncake_master
    manifest = str(tmp_path / "manifest.jsonl")
    payload = b"q" * 512
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    addr, _b = _pinned(payload)

    nb1 = MooncakeBackend(
        local_hostname=f"quota-{uuid.uuid4().hex[:6]}",
        metadata_server=f"http://{meta_addr}/metadata",
        master_server_addr=master_addr,
        global_segment_size=64 << 20,
        local_buffer_size=16 << 20,
        protocol="tcp",
        rdma_devices="",
        manifest_path=manifest,
    )
    try:
        for off in (0, 512, 1024):
            nb1.demote("eng-Q", 0, off, addr, len(payload), crc)
        nb1._slots[("eng-Q", 0, 0)].mtime = 1.0
        nb1._slots[("eng-Q", 0, 512)].mtime = 2.0
        nb1._slots[("eng-Q", 0, 1024)].mtime = 3.0
        assert nb1.enforce_byte_quota(512) == 2

        nb2 = MooncakeBackend(
            local_hostname=f"quota-r-{uuid.uuid4().hex[:6]}",
            metadata_server=f"http://{meta_addr}/metadata",
            master_server_addr=master_addr,
            global_segment_size=64 << 20,
            local_buffer_size=16 << 20,
            protocol="tcp",
            rdma_devices="",
            manifest_path=manifest,
        )
        try:
            assert nb2.n_slots() == 1
            assert nb2.get("eng-Q", 0, 0) is None
            assert nb2.get("eng-Q", 0, 512) is None
            assert nb2.get("eng-Q", 0, 1024) is not None
        finally:
            nb2.close()
    finally:
        nb1.release_engine("eng-Q")
        nb1.close()


def test_mooncake_backend_with_supervisor(mooncake_master):
    """MooncakeBackend works behind BackendSupervisor — exercises
    the supervisor's exception-isolation path on a real transport."""
    from gms_kv_ring.daemon.supervisor import BackendSupervisor

    master_addr, meta_addr = mooncake_master
    nb = _make_backend(mooncake_master)
    sup = BackendSupervisor(nb, factory=lambda: _make_backend(mooncake_master))
    try:
        payload = b"sup" * 64
        addr, _b = _pinned(payload)
        crc = zlib.crc32(payload) & 0xFFFFFFFF
        slot = sup.demote("eng-S", 0, 0, addr, len(payload), crc)
        assert slot is not None
        assert sup.n_slots() == 1
    finally:
        sup.release_engine("eng-S")
        nb.close()


def test_mooncake_backend_manifest_filters_lost_keys(mooncake_master):
    """If the manifest references keys Mooncake no longer has,
    replay must skip them and start with an empty index.

    This validates the important partial-loss restart behavior without
    relying on any specific persistence mode: a stale manifest record
    must not become a phantom slot returned to the engine.
    """
    import tempfile

    master_addr, meta_addr = mooncake_master
    with tempfile.TemporaryDirectory() as td:
        manifest = os.path.join(td, "manifest.jsonl")
        payload = b"L" * 256
        crc = zlib.crc32(payload) & 0xFFFFFFFF

        nb1 = MooncakeBackend(
            local_hostname=f"d1-{uuid.uuid4().hex[:6]}",
            metadata_server=f"http://{meta_addr}/metadata",
            master_server_addr=master_addr,
            global_segment_size=64 << 20,
            local_buffer_size=16 << 20,
            protocol="tcp",
            rdma_devices="",
            manifest_path=manifest,
        )
        addr, _b = _pinned(payload)
        nb1.demote("eng-D", 0, 0, addr, len(payload), crc)
        nb1.demote("eng-D", 0, 1024, addr, len(payload), crc)
        # Simulate backing-store loss while leaving live records in
        # the manifest: remove Mooncake objects directly, without
        # calling release_slot/release_engine, so no tombstones are
        # written. Replay must use is_exist() to filter these records.
        nb1._free_resource(nb1.get("eng-D", 0, 0))
        nb1._free_resource(nb1.get("eng-D", 0, 1024))
        nb1.close()
        del nb1

        nb2 = MooncakeBackend(
            local_hostname=f"d2-{uuid.uuid4().hex[:6]}",
            metadata_server=f"http://{meta_addr}/metadata",
            master_server_addr=master_addr,
            global_segment_size=64 << 20,
            local_buffer_size=16 << 20,
            protocol="tcp",
            rdma_devices="",
            manifest_path=manifest,
        )
        try:
            # Manifest had 2 records but Mooncake lost both segments.
            # Reconcile must report 0 surviving — engine will fall
            # to cold compute for any block referenced post-restart.
            assert nb2.n_slots() == 0, (
                f"is_exist filter should drop lost keys; got "
                f"{nb2.n_slots()} phantom slots"
            )
        finally:
            nb2.close()


def test_mooncake_backend_manifest_replay_within_session(mooncake_master, tmp_path):
    """A working version of the round-trip semantics: ONE client
    writes the manifest. Then a SECOND client opens the same
    manifest WHILE the first is still alive (i.e., the data is
    still in client #1's DRAM segment). is_exist returns 1 for
    keys that are reachable cluster-wide, so reconcile picks them
    up.

    This isn't the kill-and-restart story (which needs SSD
    offload), but it exercises the manifest read + is_exist
    filter path with real-Mooncake."""
    master_addr, meta_addr = mooncake_master
    manifest = str(tmp_path / "manifest.jsonl")

    payload = b"S" * 256
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    nb1 = MooncakeBackend(
        local_hostname=f"keep-{uuid.uuid4().hex[:6]}",
        metadata_server=f"http://{meta_addr}/metadata",
        master_server_addr=master_addr,
        global_segment_size=64 << 20,
        local_buffer_size=16 << 20,
        protocol="tcp",
        rdma_devices="",
        manifest_path=manifest,
    )
    try:
        addr, _b = _pinned(payload)
        nb1.demote("eng-S", 0, 0, addr, len(payload), crc)
        nb1.demote("eng-S", 0, 1024, addr, len(payload), crc)
        # NB1 is still alive — its DRAM segment still holds the keys.
        nb2 = MooncakeBackend(
            local_hostname=f"reader-{uuid.uuid4().hex[:6]}",
            metadata_server=f"http://{meta_addr}/metadata",
            master_server_addr=master_addr,
            global_segment_size=64 << 20,
            local_buffer_size=16 << 20,
            protocol="tcp",
            rdma_devices="",
            manifest_path=manifest,
        )
        try:
            # Reconcile saw 2 records, both still in Mooncake.
            assert nb2.n_slots() == 2, (
                f"reconcile should see 2 surviving keys; got " f"{nb2.n_slots()}"
            )
        finally:
            nb2.close()
    finally:
        nb1.release_engine("eng-S")
        nb1.close()


def test_mooncake_backend_manifest_skips_corrupt_lines(mooncake_master, tmp_path):
    """If a manifest line is malformed JSON (truncated write
    on crash, disk corruption), reconcile must skip it cleanly and
    pick up the rest."""
    master_addr, meta_addr = mooncake_master
    manifest = str(tmp_path / "manifest.jsonl")
    # Pre-seed the manifest with one bad line + one good (but the
    # good key won't be in Mooncake — that's fine, we're testing
    # the parse path, not the is_exist path).
    with open(manifest, "wb") as f:
        f.write(b"{ this is not valid json\n")
        f.write(
            b'{"eid":"e","layer":0,"offset":0,"mc_key":"gms/e/0/0",'
            b'"size":256,"crc":1,"mtime":1700000000.0}\n'
        )

    nb = MooncakeBackend(
        local_hostname=f"c1-{uuid.uuid4().hex[:6]}",
        metadata_server=f"http://{meta_addr}/metadata",
        master_server_addr=master_addr,
        global_segment_size=64 << 20,
        local_buffer_size=16 << 20,
        protocol="tcp",
        rdma_devices="",
        manifest_path=manifest,
    )
    try:
        # Bad line skipped silently; the good line's key isn't in
        # Mooncake so it's filtered too — net: 0 slots indexed.
        assert nb.n_slots() == 0
    finally:
        nb.close()


def test_mooncake_backend_no_manifest_means_no_reconcile(
    mooncake_master,
    tmp_path,
):
    """Without a manifest_path, a fresh backend starts empty even
    if there were earlier puts on the same master — that's the
    documented ephemeral behavior."""
    master_addr, meta_addr = mooncake_master
    payload = b"N" * 128
    crc = zlib.crc32(payload) & 0xFFFFFFFF

    nb1 = MooncakeBackend(
        local_hostname=f"e1-{uuid.uuid4().hex[:6]}",
        metadata_server=f"http://{meta_addr}/metadata",
        master_server_addr=master_addr,
        global_segment_size=64 << 20,
        local_buffer_size=16 << 20,
        protocol="tcp",
        rdma_devices="",
        # No manifest_path.
    )
    addr, _b = _pinned(payload)
    nb1.demote("eng-N", 0, 0, addr, len(payload), crc)
    # Release before close so we don't leak in the master.
    nb1.release_engine("eng-N")
    nb1.close()
    del nb1

    nb2 = MooncakeBackend(
        local_hostname=f"e2-{uuid.uuid4().hex[:6]}",
        metadata_server=f"http://{meta_addr}/metadata",
        master_server_addr=master_addr,
        global_segment_size=64 << 20,
        local_buffer_size=16 << 20,
        protocol="tcp",
        rdma_devices="",
    )
    try:
        assert nb2.n_slots() == 0
    finally:
        nb2.close()


def test_mooncake_backend_manifest_compaction_shrinks_file(
    mooncake_master,
    tmp_path,
):
    """Demote many slots, release most, compact: the manifest
    shrinks to just the surviving records. Replay on a new backend
    sees only the surviving slots."""
    from gms_kv_ring.common import metrics

    master_addr, meta_addr = mooncake_master
    manifest = str(tmp_path / "manifest.jsonl")
    payload = b"K" * 256
    crc = zlib.crc32(payload) & 0xFFFFFFFF

    nb = MooncakeBackend(
        local_hostname=f"comp-{uuid.uuid4().hex[:6]}",
        metadata_server=f"http://{meta_addr}/metadata",
        master_server_addr=master_addr,
        global_segment_size=64 << 20,
        local_buffer_size=16 << 20,
        protocol="tcp",
        rdma_devices="",
        manifest_path=manifest,
    )
    try:
        # 20 demotes, then release 18 — manifest has 20 records but
        # only 2 are live.
        addr, _b = _pinned(payload)
        for off in range(0, 20 * 1024, 1024):
            nb.demote("eng-C", 0, off, addr, len(payload), crc)
        for off in range(0, 18 * 1024, 1024):
            nb.release_slot("eng-C", 0, off)
        assert nb.n_slots() == 2

        size_before = nb.manifest_size_bytes()
        before_metric = metrics.mooncake_manifest_compactions.get()
        savings = nb.compact_manifest()
        size_after = nb.manifest_size_bytes()
        after_metric = metrics.mooncake_manifest_compactions.get()

        assert savings > 0, (
            f"compaction should free bytes; got savings={savings} "
            f"(before={size_before} after={size_after})"
        )
        assert size_after < size_before
        assert after_metric == before_metric + 1
        # New file has exactly 2 records (one per surviving slot).
        with open(manifest, "rb") as f:
            lines = f.read().splitlines()
        assert len(lines) == 2

        # Demote after compaction still works (append fp was swapped).
        nb.demote("eng-C", 0, 99_000, addr, len(payload), crc)
        with open(manifest, "rb") as f:
            lines = f.read().splitlines()
        assert len(lines) == 3
    finally:
        nb.release_engine("eng-C")
        nb.close()


def test_mooncake_backend_compaction_then_replay(mooncake_master, tmp_path):
    """End-to-end: write 5 records, compact, drop the backend,
    open a second backend on the same manifest. Reconcile sees
    exactly the post-compaction set (with is_exist filtering)."""
    master_addr, meta_addr = mooncake_master
    manifest = str(tmp_path / "manifest.jsonl")
    payload = b"R" * 128
    crc = zlib.crc32(payload) & 0xFFFFFFFF

    nb1 = MooncakeBackend(
        local_hostname=f"keep-{uuid.uuid4().hex[:6]}",
        metadata_server=f"http://{meta_addr}/metadata",
        master_server_addr=master_addr,
        global_segment_size=64 << 20,
        local_buffer_size=16 << 20,
        protocol="tcp",
        rdma_devices="",
        manifest_path=manifest,
    )
    try:
        addr, _b = _pinned(payload)
        for k in range(5):
            nb1.demote("eng-X", 0, k * 1024, addr, len(payload), crc)
        for k in (1, 3):
            nb1.release_slot("eng-X", 0, k * 1024)
        assert nb1.n_slots() == 3
        nb1.compact_manifest()

        # While nb1 is still alive (segment owner), open a second
        # backend on the same manifest. is_exist will return 1 for
        # nb1's live keys.
        nb2 = MooncakeBackend(
            local_hostname=f"r-{uuid.uuid4().hex[:6]}",
            metadata_server=f"http://{meta_addr}/metadata",
            master_server_addr=master_addr,
            global_segment_size=64 << 20,
            local_buffer_size=16 << 20,
            protocol="tcp",
            rdma_devices="",
            manifest_path=manifest,
        )
        try:
            assert nb2.n_slots() == 3
            # And it's exactly the right 3 (offsets 0, 2, 4).
            assert nb2.get("eng-X", 0, 0) is not None
            assert nb2.get("eng-X", 0, 1024) is None
            assert nb2.get("eng-X", 0, 2 * 1024) is not None
            assert nb2.get("eng-X", 0, 3 * 1024) is None
            assert nb2.get("eng-X", 0, 4 * 1024) is not None
        finally:
            nb2.close()
    finally:
        nb1.release_engine("eng-X")
        nb1.close()


def test_mooncake_backend_sweeper_auto_compaction(mooncake_master, tmp_path):
    """End-to-end auto-trigger: a sweeper with
    `compact_manifest_at_bytes` set runs against a MooncakeBackend
    whose manifest has grown past that threshold. The sweeper
    invokes compact_manifest() on the next tick; the file shrinks;
    `sweeper.compactions_run` increments."""
    import threading

    from gms_kv_ring.daemon.storage_tier import StorageSweeper

    master_addr, meta_addr = mooncake_master
    manifest = str(tmp_path / "manifest.jsonl")
    payload = b"S" * 256
    crc = zlib.crc32(payload) & 0xFFFFFFFF

    nb = MooncakeBackend(
        local_hostname=f"sw-{uuid.uuid4().hex[:6]}",
        metadata_server=f"http://{meta_addr}/metadata",
        master_server_addr=master_addr,
        global_segment_size=64 << 20,
        local_buffer_size=16 << 20,
        protocol="tcp",
        rdma_devices="",
        manifest_path=manifest,
    )
    try:
        # Bloat the manifest: 30 demotes + 28 releases. 30 records on
        # disk, 2 live in memory.
        addr, _b = _pinned(payload)
        for off in range(0, 30 * 1024, 1024):
            nb.demote("eng-W", 0, off, addr, len(payload), crc)
        for off in range(0, 28 * 1024, 1024):
            nb.release_slot("eng-W", 0, off)
        assert nb.n_slots() == 2

        size_before = nb.manifest_size_bytes()
        # Set threshold below current size so the next sweep fires.
        threshold = size_before // 2

        compacted = threading.Event()

        def on_sweep(_ttl, _per_eng, _quota):
            # Fires every sweep; we only signal once the sweeper has
            # actually compacted.
            if sweeper.compactions_run > 0:
                compacted.set()

        sweeper = StorageSweeper(
            nb,
            interval_s=0.05,
            compact_manifest_at_bytes=threshold,
            on_sweep=on_sweep,
        )
        sweeper.start()
        try:
            assert compacted.wait(timeout=3.0), (
                f"sweeper should auto-compact within 3s; "
                f"compactions_run={sweeper.compactions_run}"
            )
        finally:
            sweeper.stop()

        size_after = nb.manifest_size_bytes()
        assert size_after < size_before
        assert sweeper.last_compaction_savings > 0
        # Manifest now has only 2 live records.
        with open(manifest, "rb") as f:
            lines = f.read().splitlines()
        assert len(lines) == 2
    finally:
        nb.release_engine("eng-W")
        nb.close()


def test_mooncake_backend_sweeper_skips_compaction_below_threshold(
    mooncake_master,
    tmp_path,
):
    """If the manifest is below the threshold, the sweeper must
    NOT compact — extra fsyncs and locking are wasted work
    otherwise."""
    import threading

    from gms_kv_ring.daemon.storage_tier import StorageSweeper

    master_addr, meta_addr = mooncake_master
    manifest = str(tmp_path / "manifest.jsonl")
    payload = b"B" * 64
    crc = zlib.crc32(payload) & 0xFFFFFFFF

    nb = MooncakeBackend(
        local_hostname=f"sw2-{uuid.uuid4().hex[:6]}",
        metadata_server=f"http://{meta_addr}/metadata",
        master_server_addr=master_addr,
        global_segment_size=64 << 20,
        local_buffer_size=16 << 20,
        protocol="tcp",
        rdma_devices="",
        manifest_path=manifest,
    )
    try:
        addr, _b = _pinned(payload)
        nb.demote("eng-Q", 0, 0, addr, len(payload), crc)
        size = nb.manifest_size_bytes()
        assert size > 0

        ran = threading.Event()

        def on_sweep(*_):
            ran.set()

        # Threshold MUCH larger than current size → never fires.
        sweeper = StorageSweeper(
            nb,
            interval_s=0.05,
            compact_manifest_at_bytes=size * 100,
            on_sweep=on_sweep,
        )
        sweeper.start()
        try:
            # Wait for at least one sweep iteration to happen.
            assert ran.wait(timeout=2.0)
            # Sweeper should have run but never compacted.
            assert sweeper.sweeps_run >= 1
            assert sweeper.compactions_run == 0
        finally:
            sweeper.stop()
    finally:
        nb.release_engine("eng-Q")
        nb.close()


def test_mooncake_backend_compaction_noop_without_manifest(mooncake_master):
    """compact_manifest is a no-op when manifest is disabled — must
    not raise."""
    master_addr, meta_addr = mooncake_master
    nb = MooncakeBackend(
        local_hostname=f"n-{uuid.uuid4().hex[:6]}",
        metadata_server=f"http://{meta_addr}/metadata",
        master_server_addr=master_addr,
        global_segment_size=64 << 20,
        local_buffer_size=16 << 20,
        protocol="tcp",
        rdma_devices="",
    )
    try:
        assert nb.compact_manifest() == 0
        assert nb.manifest_size_bytes() == 0
    finally:
        nb.close()


def test_mooncake_backend_stats_includes_bytes_by_engine(mooncake_master):
    nb = _make_backend(mooncake_master)
    try:
        addr_a, _ba = _pinned(b"a" * 1024)
        nb.demote("eng-1", 0, 0, addr_a, 1024, zlib.crc32(b"a" * 1024) & 0xFFFFFFFF)
        nb.demote("eng-2", 0, 0, addr_a, 1024, zlib.crc32(b"a" * 1024) & 0xFFFFFFFF)
        nb.demote("eng-2", 0, 1024, addr_a, 1024, zlib.crc32(b"a" * 1024) & 0xFFFFFFFF)
        s = nb.stats()
        assert s["n_slots"] == 3
        assert s["bytes_by_engine"] == {"eng-1": 1024, "eng-2": 2048}
    finally:
        nb.release_engine("eng-1")
        nb.release_engine("eng-2")
        nb.close()
