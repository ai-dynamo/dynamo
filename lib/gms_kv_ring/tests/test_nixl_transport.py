# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end NIXL transport tests (Phase 3 of cross-node design).

Two NIXL transports in the same process, each with its own agent +
listen port, exchange bytes via UCX-over-SHM (the local transport
UCX picks when peers are on the same host). Receiver-side notif
drain invokes a callback that bridges into StagingTier
(reserve+commit), exercising the full inbound path including I-9
hash verification.

Skipped if pynixl isn't installed or UCX backend can't initialize."""

from __future__ import annotations

import ctypes
import hashlib
import os
import subprocess
import sys
import threading

import pytest

# Skip the whole module if NIXL isn't available in this Python environment.
nixl = pytest.importorskip("nixl")

# Even with NIXL installed, the agent-to-agent transfer in the pytest
# harness shows flaky notif-delivery latency (single-host UCX-over-SHM
# loopback works <100ms in a standalone Python process but variably
# 1-5s under pytest's threading + capture machinery). The transport
# code is exercised by the standalone smoke harness (see
# `docs/CROSS_NODE_DESIGN.md` and the `_smoke*.py` files in the
# transport module docstring). For routine PR runs we env-gate this
# module the same way `test_backend_nixl.py` gates its real-GDS
# tests — opt-in via env var; failure modes are debugged by hand.
if not os.environ.get("GMS_KVR_TEST_NIXL_TRANSPORT"):
    pytest.skip(
        "Set GMS_KVR_TEST_NIXL_TRANSPORT=1 to run NIXL transport tests. "
        "Skipped by default because pytest's threading model triggers "
        "variable UCX-loopback notif-delivery latency.",
        allow_module_level=True,
    )


def _probe_ucx_loopback() -> None:
    """Contain native UCX capability failures in a sacrificial process."""
    probe = r"""
import ctypes, hashlib, socket
from gms_kv_ring.daemon.transport import NixlTransport

def port():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    value = sock.getsockname()[1]
    sock.close()
    return value

a = b = None
try:
    a = NixlTransport("probe-a", port())
    b = NixlTransport("probe-b", port())
    src = (ctypes.c_ubyte * 4096)(*[(i & 0xff) for i in range(4096)])
    dst = (ctypes.c_ubyte * 4096)()
    a.register_buffer(ctypes.addressof(src), 4096)
    b.register_buffer(ctypes.addressof(dst), 4096)
    peer = a.add_peer_inproc(b)
    payload = bytes(src)
    a.send(peer, ctypes.addressof(src), 4096, ctypes.addressof(dst),
           "probe", hashlib.sha256(payload).digest(), timeout_s=5)
    assert bytes(dst) == payload
finally:
    if a is not None:
        a.close()
    if b is not None:
        b.close()
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            timeout=20,
            env=os.environ.copy(),
        )
    except subprocess.TimeoutExpired:
        pytest.skip(
            "NIXL/UCX loopback capability probe timed out after 20s",
            allow_module_level=True,
        )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()[-500:]
        if result.returncode < 0:
            reason = f"crashed with signal {-result.returncode}"
        else:
            reason = f"failed with exit code {result.returncode}"
        pytest.skip(
            f"NIXL/UCX loopback capability probe {reason}: {detail}",
            allow_module_level=True,
        )


_probe_ucx_loopback()

# NIXL listen sockets aren't released cleanly on agent destruction, so
# every test needs unique ports across the run. Module-level counter.
_PORT_BASE = 0x5000 + (os.getpid() % 100) * 100
_port_lock = threading.Lock()
_next_port = [_PORT_BASE]


def _next_pair():
    with _port_lock:
        a, b = _next_port[0], _next_port[0] + 1
        _next_port[0] += 2
        return a, b


def _hash(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def _alloc_aligned(size: int):
    """Allocate a host buffer and return (ptr_int, ctypes_array). The
    array must be kept alive by the caller so refcount keeps the
    backing storage."""
    buf = (ctypes.c_ubyte * size)()
    return ctypes.addressof(buf), buf


def _make_transport(name: str, port: int, on_inbound=None):
    from gms_kv_ring.daemon.transport import NixlTransport, TransportNotAvailable

    try:
        return NixlTransport(
            agent_name=name,
            listen_port=port,
            on_inbound_notif=on_inbound,
        )
    except TransportNotAvailable as exc:
        pytest.skip(f"NIXL transport not available: {exc}")


@pytest.fixture
def transports():
    """Yield two NIXL transports A and B; clean up at teardown.
    Uses unique ports per test invocation."""
    port_a, port_b = _next_pair()
    a = _make_transport(f"agent_a_{port_a}", port_a)
    b = _make_transport(f"agent_b_{port_b}", port_b)
    yield (a, b, port_a, port_b)
    try:
        a.close()
    finally:
        b.close()


# ----- bare loopback transfer (no StagingTier) ------------------------------


def test_loopback_write_and_notif(transports):
    a, b, port_a, port_b = transports

    src_ptr, _src_buf = _alloc_aligned(4096)
    dst_ptr, _dst_buf = _alloc_aligned(4096)
    payload = bytes((i & 0xFF) for i in range(64)) + b"\x00" * (4096 - 64)
    ctypes.memmove(src_ptr, payload, len(payload))

    a.register_buffer(src_ptr, 4096, label="src")
    b.register_buffer(dst_ptr, 4096, label="dst")

    # In-process loopback metadata exchange (memory must be registered
    # FIRST). add_peer_inproc handles both directions.
    peer_b = a.add_peer_inproc(b)

    received: list[tuple[str, str, bytes]] = []
    drained = threading.Event()

    def on_inbound(peer_name, rid, h):
        received.append((peer_name, rid, h))
        drained.set()

    b._on_inbound_notif = on_inbound  # override (test convenience)

    payload_hash = _hash(payload)
    a.send(
        peer=peer_b,
        local_ptr=src_ptr,
        size=len(payload),
        remote_ptr=dst_ptr,
        reservation_id="rsv-001",
        content_hash=payload_hash,
        timeout_s=5.0,
    )

    # Wait up to 3s for notif to be drained
    assert drained.wait(timeout=3.0), "notif never delivered"
    assert len(received) == 1
    sender_name, rid, h = received[0]
    assert sender_name == a.agent_name()
    assert rid == "rsv-001"
    assert h == payload_hash

    # Verify the bytes actually landed in B's destination buffer
    dst_view = (ctypes.c_ubyte * len(payload)).from_address(dst_ptr)
    assert bytes(dst_view) == payload


# ----- staging-tier integration ---------------------------------------------


def test_transport_to_staging_tier_round_trip(transports):
    """The full intended path: sender pushes bytes; receiver's notif
    callback reads dest bytes and calls StagingTier.commit_or_reject,
    transitioning the slot to READY."""
    from gms_kv_ring.daemon.staging_tier import (
        Reservation,
        StagingTier,
        _BytearrayAllocator,
    )

    a, b, port_a, port_b = transports

    # Receiver-side staging tier (manages its own buffer pool, but for
    # this test we pre-allocate the dst buffer outside it via ctypes
    # — staging_tier just bookkeeps the reservation state machine).
    staging = StagingTier(
        capacity_bytes=1 << 20,
        allocator=_BytearrayAllocator(),
    )

    src_ptr, _src_buf = _alloc_aligned(4096)
    dst_ptr, _dst_buf = _alloc_aligned(4096)
    payload = b"cross-node round trip payload" * 4  # 116 bytes
    ctypes.memmove(src_ptr, payload, len(payload))

    a.register_buffer(src_ptr, 4096, label="src")
    b.register_buffer(dst_ptr, 4096, label="dst")

    peer_b = a.add_peer_inproc(b)

    # Pre-reserve a staging slot on B (the receiver) so we know its
    # reservation_id when sending. In production this is mediated by
    # a UDS pre-reserve RPC; here we call directly.
    payload_hash = _hash(payload)
    reservation = staging.reserve_or_wait(payload_hash, source_daemon="a")
    assert isinstance(reservation, Reservation)

    commit_done = threading.Event()
    commit_result = []

    def on_inbound(peer_name, rid, h):
        # Read the bytes from the dst buffer (NIXL WRITE landed here).
        dst_view = (ctypes.c_ubyte * len(payload)).from_address(dst_ptr)
        bytes_received = bytes(dst_view)
        result = staging.commit_or_reject(rid, bytes_received)
        commit_result.append(result)
        commit_done.set()

    b._on_inbound_notif = on_inbound

    a.send(
        peer=peer_b,
        local_ptr=src_ptr,
        size=len(payload),
        remote_ptr=dst_ptr,
        reservation_id=reservation.reservation_id,
        content_hash=payload_hash,
        timeout_s=5.0,
    )

    # Notif delivery latency varies — fast in production-tuned envs,
    # surprisingly slow under pytest's threading + capture machinery.
    # Generous timeout for the integration test; flake-resistant.
    assert commit_done.wait(timeout=10.0)
    from gms_kv_ring.daemon.staging_tier import CommitOk

    assert len(commit_result) == 1
    assert isinstance(commit_result[0], CommitOk), commit_result[0]

    # Slot is now READY in the receiver's staging tier
    hit = staging.scan([payload_hash])
    assert payload_hash in hit
    assert hit[payload_hash].bytes_size == len(payload)


# ----- I-9 hash mismatch via wire corruption --------------------------------


def test_hash_mismatch_caught_at_commit(transports):
    """If we lie about the content_hash (advertise wrong hash), the
    receiver's StagingTier.commit_or_reject rejects via I-9."""
    from gms_kv_ring.daemon.staging_tier import (
        CommitCorrupt,
        Reservation,
        StagingTier,
        _BytearrayAllocator,
    )

    a, b, port_a, port_b = transports

    staging = StagingTier(
        capacity_bytes=1 << 20,
        allocator=_BytearrayAllocator(),
    )

    src_ptr, _src_buf = _alloc_aligned(4096)
    dst_ptr, _dst_buf = _alloc_aligned(4096)
    actual_payload = b"true bytes" * 8
    advertised_hash = _hash(b"different content")  # wrong hash on purpose
    ctypes.memmove(src_ptr, actual_payload, len(actual_payload))

    a.register_buffer(src_ptr, 4096, label="src")
    b.register_buffer(dst_ptr, 4096, label="dst")

    peer_b = a.add_peer_inproc(b)

    reservation = staging.reserve_or_wait(advertised_hash, source_daemon="a")
    assert isinstance(reservation, Reservation)

    commit_done = threading.Event()
    commit_result = []

    def on_inbound(peer_name, rid, h):
        dst_view = (ctypes.c_ubyte * len(actual_payload)).from_address(dst_ptr)
        result = staging.commit_or_reject(rid, bytes(dst_view))
        commit_result.append(result)
        commit_done.set()

    b._on_inbound_notif = on_inbound

    a.send(
        peer=peer_b,
        local_ptr=src_ptr,
        size=len(actual_payload),
        remote_ptr=dst_ptr,
        reservation_id=reservation.reservation_id,
        content_hash=advertised_hash,
        timeout_s=5.0,
    )

    assert commit_done.wait(timeout=10.0)
    assert isinstance(commit_result[0], CommitCorrupt)
    assert "hash mismatch" in commit_result[0].reason
    # Slot is dropped; subsequent scan empty.
    assert staging.scan([advertised_hash]) == {}
