# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phase 4b end-to-end test: the cross-node RECEIVE pipeline.

Two daemons (sender + receiver) on the same host, talking via real
NIXL UCX (over SHM transport since they're local). The test plays
the role of the future router:

  1. Discovers each daemon's transport identity.
  2. Wires them as peers via `transport_add_peer`.
  3. Calls `staging_reserve` on the receiver to allocate a slot +
     receive-buffer offset for an incoming hash.
  4. Plays the role of the sender: registers source bytes with the
     sender daemon's NIXL agent, issues a WRITE to the receiver's
     returned `remote_ptr` with the reservation_id + content_hash
     in the notif payload.
  5. Waits for the receiver's `_on_nixl_inbound` to commit the
     payload into StagingTier and publish a Stored event.
  6. Verifies `staging_scan` from a third client shows the hash
     READY on the receiver.

Env-gated behind GMS_KVR_TEST_NIXL_TRANSPORT=1 — same gate as the
P3 NIXL transport tests, for the same reason (UCX-loopback notif
latency variance under pytest)."""

from __future__ import annotations

import asyncio
import ctypes
import hashlib
import os
import tempfile
import threading
import time

import pytest

# Skip whole module if NIXL or test gate are missing.
pytest.importorskip("nixl")
if not os.environ.get("GMS_KVR_TEST_NIXL_TRANSPORT"):
    pytest.skip(
        "Set GMS_KVR_TEST_NIXL_TRANSPORT=1 to run cross-node receive tests",
        allow_module_level=True,
    )

from gms_kv_ring.daemon.client import DaemonClient  # noqa: E402
from gms_kv_ring.daemon.server import Daemon  # noqa: E402

_PORT_BASE = 0x6000 + (os.getpid() % 100) * 200
_port_lock = threading.Lock()
_next_port = [_PORT_BASE]


def _next_quartet():
    """Reserve 4 ports: 2 daemons × (UDS not needed; just NIXL listen)."""
    with _port_lock:
        ports = list(range(_next_port[0], _next_port[0] + 4))
        _next_port[0] += 4
    return ports


def _spawn_daemon(
    listen_uds: str,
    nixl_port: int,
    agent_name: str,
    staging_cap: int = 1 << 20,
    recv_buf: int = 1 << 20,
):
    daemon = Daemon(
        listen_socket=listen_uds,
        storage_dir=tempfile.mkdtemp(prefix="gms-p4b-"),
        supervise_backend=False,
        staging_capacity_bytes=staging_cap,
        staging_allocator=None,  # use default — but tests run without CUDA
        transport_listen_port=nixl_port,
        transport_agent_name=agent_name,
        staging_receive_buffer_bytes=recv_buf,
    )
    # Patch in the test allocator so we don't need CUDA
    from gms_kv_ring.daemon.staging_tier import _BytearrayAllocator

    daemon.staging_tier._alloc = _BytearrayAllocator()

    lh: dict = {}

    def _run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        lh["loop"] = loop
        try:
            loop.run_until_complete(daemon.serve())
        finally:
            loop.close()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and not os.path.exists(listen_uds):
        time.sleep(0.02)
    if not os.path.exists(listen_uds):
        raise RuntimeError("daemon never bound socket")
    return daemon, t, lh


def _stop(daemon, t, lh) -> None:
    if "loop" in lh:
        lh["loop"].call_soon_threadsafe(daemon.stop)
    t.join(timeout=3)


def _hash(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def test_cross_node_receive_pipeline():
    """End-to-end: sender daemon WRITEs bytes to receiver daemon's
    staging slot; receiver commits via _on_nixl_inbound."""
    tmpdir_s = tempfile.mkdtemp(prefix="gms-p4b-s-")
    tmpdir_r = tempfile.mkdtemp(prefix="gms-p4b-r-")
    sock_s = os.path.join(tmpdir_s, "d.sock")
    sock_r = os.path.join(tmpdir_r, "d.sock")
    p_s, p_r, p3, p4 = _next_quartet()
    name_s = f"sender-{p_s}"
    name_r = f"receiver-{p_r}"

    sender, ts, lhs = _spawn_daemon(sock_s, p_s, name_s)
    receiver, tr, lhr = _spawn_daemon(sock_r, p_r, name_r)

    try:
        # The "router" wires the peers. add_peer_inproc would work
        # since both transports are in the same process; in production
        # the router calls `transport_add_peer` on each daemon. We
        # use the in-process path here because the listen-thread
        # exchange is racy under pytest (see test_nixl_transport.py).
        sender.transport.add_peer_inproc(receiver.transport)

        # Test plays the role of the source: registers its OWN bytes,
        # then asks the receiver to reserve a slot, then writes.
        client_r = DaemonClient(sock_r)
        try:
            info_r = client_r.transport_info()
            assert info_r["agent_name"] == name_r
            assert info_r["listen_port"] == p_r
            assert info_r["receive_buffer_bytes"] == (1 << 20)

            # Build sender-side source memory + register with sender's
            # transport. ctypes-owned so refcount keeps it alive.
            payload = b"cross-node receive pipeline e2e" * 8  # 248 bytes
            src_buf = (ctypes.c_ubyte * 4096)()
            src_ptr = ctypes.addressof(src_buf)
            ctypes.memmove(src_ptr, payload, len(payload))
            sender.transport.register_buffer(src_ptr, 4096, label="src")
            # Re-snapshot peer metadata so the receiver sees src
            sender.transport.add_peer_inproc(receiver.transport)

            content_hash = _hash(payload)

            # Receiver reserves
            r = client_r.staging_reserve(
                content_hash=content_hash,
                size=len(payload),
                source_daemon=name_s,
            )
            assert r["outcome"] == "reserved", r
            assert r["reservation_id"]
            remote_ptr = r["remote_ptr"]
            assert remote_ptr != 0

            # Source pushes via NIXL. Build the PeerHandle from the
            # already-registered peer (add_peer_inproc was called above).
            from gms_kv_ring.daemon.transport import PeerHandle

            peer = PeerHandle(
                nixl_name=name_r,
                ip_addr="inproc",
                port=p_r,
            )
            sender.transport.send(
                peer=peer,
                local_ptr=src_ptr,
                size=len(payload),
                remote_ptr=remote_ptr,
                reservation_id=r["reservation_id"],
                content_hash=content_hash,
                timeout_s=10.0,
            )

            # Wait for the receiver's _on_nixl_inbound to commit
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                hits = client_r.staging_scan([content_hash])
                if content_hash in hits:
                    break
                time.sleep(0.05)
            hits = client_r.staging_scan([content_hash])
            assert content_hash in hits, "receiver never committed"
            assert hits[content_hash]["bytes_size"] == len(payload)

            # Publisher emitted a Stored event
            stats = receiver.placement_publisher.stats()
            assert stats["stored"] >= 1
        finally:
            client_r.close()
    finally:
        _stop(receiver, tr, lhr)
        _stop(sender, ts, lhs)


def test_staging_reserve_already_ready_short_circuits():
    """If the destination already has the hash, staging_reserve
    returns `already_ready` and the router skips the transfer."""
    tmpdir = tempfile.mkdtemp(prefix="gms-p4b-r2-")
    sock = os.path.join(tmpdir, "d.sock")
    p_r, _, _, _ = _next_quartet()
    name = f"receiver-{p_r}"

    daemon, t, lh = _spawn_daemon(sock, p_r, name)
    try:
        # Pre-populate StagingTier directly (bypass NIXL)
        from gms_kv_ring.daemon.staging_tier import Reservation

        h = _hash(b"already there")
        res = daemon.staging_tier.reserve_or_wait(h, source_daemon="test")
        assert isinstance(res, Reservation)
        daemon.staging_tier.commit_or_reject(res.reservation_id, b"already there")

        client = DaemonClient(sock)
        try:
            r = client.staging_reserve(
                content_hash=h,
                size=13,
                source_daemon="some-peer",
            )
            assert r["outcome"] == "already_ready", r
            assert r["bytes_size"] == 13
        finally:
            client.close()
    finally:
        _stop(daemon, t, lh)


def test_staging_fail_releases_reservation():
    """If the router calls staging_fail, the slot is released cleanly
    and subsequent reservations for the same hash succeed."""
    tmpdir = tempfile.mkdtemp(prefix="gms-p4b-r3-")
    sock = os.path.join(tmpdir, "d.sock")
    p_r, _, _, _ = _next_quartet()
    name = f"receiver-{p_r}"
    daemon, t, lh = _spawn_daemon(sock, p_r, name)
    try:
        client = DaemonClient(sock)
        try:
            h = _hash(b"will fail")
            r1 = client.staging_reserve(content_hash=h, size=128)
            assert r1["outcome"] == "reserved"
            rid1 = r1["reservation_id"]

            client.staging_fail(rid1, reason="transport error")

            # New reservation for the same hash should succeed
            r2 = client.staging_reserve(content_hash=h, size=128)
            assert r2["outcome"] == "reserved"
            assert r2["reservation_id"] != rid1
        finally:
            client.close()
    finally:
        _stop(daemon, t, lh)
