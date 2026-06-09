# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for the `staging_scan` RPC — Phase 2 of cross-node.

Validates that the daemon's StagingTier is reachable from a remote
client over UDS. The DaemonClient is the same wrapper engine adapters
use through `GMSKvRing.staging_scan`; both code paths get the same
JSON RPC.

Runs without CUDA: the daemon's StagingTier is constructed with a
test-only `_BytearrayAllocator`."""

from __future__ import annotations

import asyncio
import hashlib
import os
import tempfile
import threading
import time

from gms_kv_ring.daemon.client import DaemonClient
from gms_kv_ring.daemon.server import Daemon
from gms_kv_ring.daemon.staging_tier import _BytearrayAllocator


def _hash(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def _spawn_daemon(staging_capacity_bytes: int):
    """Spawn a daemon on a unique UDS in a background asyncio loop.
    Returns (daemon, sock_path, thread, loop_holder). Caller stops via
    `loop_holder["loop"].call_soon_threadsafe(daemon.stop)`."""
    tmpdir = tempfile.mkdtemp(prefix="gms-kvr-staging-rpc-")
    sock = os.path.join(tmpdir, "daemon.sock")
    daemon = Daemon(
        listen_socket=sock,
        storage_dir=tmpdir,
        supervise_backend=False,
        staging_capacity_bytes=staging_capacity_bytes,
        staging_allocator=_BytearrayAllocator() if staging_capacity_bytes > 0 else None,
    )
    loop_holder: dict = {}

    def _run() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop_holder["loop"] = loop
        try:
            loop.run_until_complete(daemon.serve())
        finally:
            loop.close()

    th = threading.Thread(target=_run, daemon=True)
    th.start()
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and not os.path.exists(sock):
        time.sleep(0.02)
    assert os.path.exists(sock), "daemon socket never appeared"
    return daemon, sock, th, loop_holder


def _stop(daemon, thread, loop_holder) -> None:
    if "loop" in loop_holder:
        loop_holder["loop"].call_soon_threadsafe(daemon.stop)
    thread.join(timeout=3)


# ----- empty staging tier ----------------------------------------------------


def test_staging_scan_disabled_returns_empty():
    """With staging_capacity_bytes=0, the staging tier isn't constructed.
    The RPC should return an empty dict without erroring."""
    daemon, sock, th, lh = _spawn_daemon(staging_capacity_bytes=0)
    try:
        with DaemonClient(sock) as client:
            hits = client.staging_scan([_hash(b"anything")])
            assert hits == {}
    finally:
        _stop(daemon, th, lh)


def test_staging_scan_empty_query_short_circuits():
    """Empty hash list returns empty dict without any RPC."""
    daemon, sock, th, lh = _spawn_daemon(staging_capacity_bytes=1 << 20)
    try:
        with DaemonClient(sock) as client:
            hits = client.staging_scan([])
            assert hits == {}
    finally:
        _stop(daemon, th, lh)


# ----- populated staging tier ------------------------------------------------


def test_staging_scan_returns_only_ready_hashes():
    """Populate staging via direct reserve+commit, scan via RPC."""
    daemon, sock, th, lh = _spawn_daemon(staging_capacity_bytes=1 << 20)
    data_a = b"alpha bytes" * 8
    data_b = b"beta bytes" * 8
    ha = _hash(data_a)
    hb = _hash(data_b)
    hc = _hash(b"never deposited")

    # Deposit ha and hb directly into the daemon's staging tier
    from gms_kv_ring.daemon.staging_tier import Reservation

    r_a = daemon.staging_tier.reserve_or_wait(ha, "test")
    assert isinstance(r_a, Reservation)
    daemon.staging_tier.commit_or_reject(r_a.reservation_id, data_a)
    r_b = daemon.staging_tier.reserve_or_wait(hb, "test")
    assert isinstance(r_b, Reservation)
    daemon.staging_tier.commit_or_reject(r_b.reservation_id, data_b)

    try:
        with DaemonClient(sock) as client:
            hits = client.staging_scan([ha, hb, hc])
            assert set(hits.keys()) == {ha, hb}
            assert hits[ha]["bytes_size"] == len(data_a)
            assert hits[hb]["bytes_size"] == len(data_b)
            assert hits[ha]["generation"] == 1
            assert hits[hb]["generation"] == 1
            # CRC32 round-trips through the JSON int channel
            import zlib

            assert hits[ha]["crc32"] == zlib.crc32(data_a) & 0xFFFFFFFF
    finally:
        _stop(daemon, th, lh)


def test_staging_scan_skips_reserved_slot():
    """A slot in RESERVED state (transfer in flight) must NOT appear in
    scan results — only READY slots are visible to consumers."""
    daemon, sock, th, lh = _spawn_daemon(staging_capacity_bytes=1 << 20)
    h = _hash(b"in flight")

    from gms_kv_ring.daemon.staging_tier import Reservation

    r = daemon.staging_tier.reserve_or_wait(h, "test")
    assert isinstance(r, Reservation)
    # Do NOT commit — slot stays RESERVED.

    try:
        with DaemonClient(sock) as client:
            hits = client.staging_scan([h])
            assert hits == {}
    finally:
        _stop(daemon, th, lh)


def test_staging_scan_hex_round_trip_is_lossless():
    """The RPC encodes hashes as hex on the wire; verify bytes
    round-trip without corruption."""
    daemon, sock, th, lh = _spawn_daemon(staging_capacity_bytes=1 << 20)
    payload = bytes(range(256))  # forces every byte value
    h = _hash(payload)

    from gms_kv_ring.daemon.staging_tier import Reservation

    r = daemon.staging_tier.reserve_or_wait(h, "test")
    assert isinstance(r, Reservation)
    daemon.staging_tier.commit_or_reject(r.reservation_id, payload)

    try:
        with DaemonClient(sock) as client:
            hits = client.staging_scan([h])
            assert h in hits  # exact byte-for-byte match
    finally:
        _stop(daemon, th, lh)
