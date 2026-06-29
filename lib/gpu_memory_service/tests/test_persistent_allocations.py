# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for PersistentAllocationManager + RPC surface.

Two flavors:
  - Unit: PersistentAllocationManager directly, cuMem* monkey-patched
    to fakes (no real CUDA needed).
  - RPC contract: through GMS.handle_request, exercising the new
    Claim/Release/Export/List RPC messages and the session-bypass
    that lets persistent ops run without an RW lock.
"""

from __future__ import annotations

import asyncio
import itertools
import os
from types import SimpleNamespace

import pytest
from gpu_memory_service.client.memory_manager import GMSClientMemoryManager
from gpu_memory_service.common.locks import GrantedLockType
from gpu_memory_service.common.protocol.messages import (
    ClaimPersistentAllocationRequest,
    ClaimPersistentAllocationResponse,
    ErrorResponse,
    ExportPersistentAllocationRequest,
    ExportPersistentAllocationResponse,
    ListPersistentAllocationsRequest,
    ListPersistentAllocationsResponse,
    ReleasePersistentAllocationRequest,
    ReleasePersistentAllocationResponse,
)
from gpu_memory_service.server import allocations as server_allocations
from gpu_memory_service.server import persistent_allocations as server_persistent
from gpu_memory_service.server.fsm import Connection
from gpu_memory_service.server.gms import GMS
from gpu_memory_service.server.persistent_allocations import (
    PersistentAllocationManager,
    PersistentClaimConflictError,
    PersistentNotFoundError,
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.none,
    pytest.mark.gpu_0,
]


# ---------------------------------------------------------------------
# Shared cuMem* monkey-patching
# ---------------------------------------------------------------------


@pytest.fixture
def fake_cuda(monkeypatch):
    """Stub the CUDA driver calls used by both managers so tests can
    run on hosts without CUDA + don't allocate real GPU memory."""
    handles = itertools.count(2000)
    vas = itertools.count(0x80000000, 0x10000)

    def export_fd(handle: int) -> int:
        # Use a real pipe so the FD is valid (we close + dup it in code
        # under test).
        read_fd, write_fd = os.pipe()
        os.close(write_fd)
        return read_fd

    for mod in (server_allocations, server_persistent):
        monkeypatch.setattr(mod, "cuda_ensure_initialized", lambda: None)
        monkeypatch.setattr(
            mod,
            "cumem_get_allocation_granularity",
            lambda device: 4096,
        )
        monkeypatch.setattr(
            mod,
            "cumem_create_tolerate_oom",
            lambda size, device: (True, next(handles)),
        )
        monkeypatch.setattr(mod, "cumem_release", lambda handle: None)
        monkeypatch.setattr(
            mod,
            "cumem_export_to_shareable_handle",
            export_fd,
        )
    # Daemon-side VA mapping helpers — only used by persistent_allocations.
    monkeypatch.setattr(
        server_persistent,
        "cumem_address_reserve",
        lambda size, gran: next(vas),
    )
    monkeypatch.setattr(
        server_persistent,
        "cumem_address_free",
        lambda va, size: None,
    )
    monkeypatch.setattr(
        server_persistent,
        "cumem_map",
        lambda va, size, handle: None,
    )
    monkeypatch.setattr(
        server_persistent,
        "cumem_set_access",
        lambda va, size, device, access: None,
    )
    monkeypatch.setattr(
        server_persistent,
        "cumem_unmap",
        lambda va, size: None,
    )


# ---------------------------------------------------------------------
# Unit tests: PersistentAllocationManager direct
# ---------------------------------------------------------------------


class _FakePersistentClient:
    def __init__(self, *, aligned_size: int, reattached: bool):
        self.aligned_size = aligned_size
        self.reattached = reattached

    def claim_persistent(self, **_kwargs):
        return SimpleNamespace(
            allocation_id="alloc-1",
            aligned_size=self.aligned_size,
            reattached=self.reattached,
        )


def _memory_manager_with_client(client) -> GMSClientMemoryManager:
    manager = GMSClientMemoryManager.__new__(GMSClientMemoryManager)
    manager.granularity = 4096
    manager._client = client
    return manager


def test_client_allows_shared_reattach_to_larger_existing_allocation():
    manager = _memory_manager_with_client(
        _FakePersistentClient(aligned_size=8192, reattached=True)
    )

    allocation_id, aligned_size, reattached = manager.claim_persistent(
        "eng-A", "kv_pool", 4096, shared=True
    )

    assert allocation_id == "alloc-1"
    assert aligned_size == 8192
    assert reattached is True


def test_client_rejects_undersized_or_fresh_persistent_mismatch():
    undersized = _memory_manager_with_client(
        _FakePersistentClient(aligned_size=4096, reattached=True)
    )
    with pytest.raises(RuntimeError, match="alignment mismatch"):
        undersized.claim_persistent("eng-A", "kv_pool", 8192, shared=True)

    fresh_mismatch = _memory_manager_with_client(
        _FakePersistentClient(aligned_size=8192, reattached=False)
    )
    with pytest.raises(RuntimeError, match="alignment mismatch"):
        fresh_mismatch.claim_persistent("eng-A", "kv_pool", 4096, shared=True)


def test_client_rejects_larger_exclusive_persistent_reattach():
    manager = _memory_manager_with_client(
        _FakePersistentClient(aligned_size=8192, reattached=True)
    )

    with pytest.raises(RuntimeError, match="alignment mismatch"):
        manager.claim_persistent("eng-A", "kv_pool", 4096)


def test_claim_creates_fresh_allocation(fake_cuda):
    m = PersistentAllocationManager(device=0)
    alloc, reattached = m.claim(
        engine_id="eng-A",
        tag="kv_pool",
        size=4096,
    )
    assert reattached is False
    assert alloc.engine_id == "eng-A"
    assert alloc.tag == "kv_pool"
    assert alloc.size == 4096
    assert alloc.aligned_size >= 4096
    assert m.is_claimed("eng-A", "kv_pool") is True


def test_reclaim_after_unclaim_returns_existing(fake_cuda):
    """Engine disconnects (unclaim) but doesn't release. New claim
    with same key reattaches to the same allocation."""
    m = PersistentAllocationManager(device=0)
    alloc1, r1 = m.claim("eng-A", "kv_pool", 4096)
    assert r1 is False

    # Simulate engine disconnect.
    assert m.unclaim("eng-A", "kv_pool") is True

    alloc2, r2 = m.claim("eng-A", "kv_pool", 4096)
    assert r2 is True, "second claim should reattach"
    assert (
        alloc1.allocation_id == alloc2.allocation_id
    ), "reattach must return the SAME underlying allocation"


def test_concurrent_claim_rejected(fake_cuda):
    """Second engine claiming the same (engine_id, tag) while another
    holds it must be rejected."""
    m = PersistentAllocationManager(device=0)
    m.claim("eng-A", "kv_pool", 4096)
    with pytest.raises(PersistentClaimConflictError):
        m.claim("eng-A", "kv_pool", 4096)


def test_shared_claim_allows_multiple_attached_engines(fake_cuda):
    m = PersistentAllocationManager(device=0)
    alloc1, reattached1 = m.claim("eng-A", "kv_pool", 4096, shared=True)
    alloc2, reattached2 = m.claim("eng-A", "kv_pool", 4096, shared=True)

    assert reattached1 is False
    assert reattached2 is True
    assert alloc1.allocation_id == alloc2.allocation_id
    assert m.shared_claim_count("eng-A", "kv_pool") == 2
    assert m.is_claimed("eng-A", "kv_pool") is True

    assert m.unclaim("eng-A", "kv_pool") is True
    assert m.shared_claim_count("eng-A", "kv_pool") == 1
    assert m.is_claimed("eng-A", "kv_pool") is True

    assert m.unclaim("eng-A", "kv_pool") is True
    assert m.shared_claim_count("eng-A", "kv_pool") == 0
    assert m.is_claimed("eng-A", "kv_pool") is False

def test_shared_claimant_cannot_release_allocation_used_by_peer(fake_cuda):
    m = PersistentAllocationManager(device=0)
    m.claim("eng-A", "kv_pool", 4096, shared=True)
    m.claim("eng-A", "kv_pool", 4096, shared=True)

    with pytest.raises(PersistentClaimConflictError, match="other shared claimants"):
        m.release("eng-A", "kv_pool")
    assert m.get("eng-A", "kv_pool")

    assert m.unclaim("eng-A", "kv_pool") is True
    assert m.release("eng-A", "kv_pool") is True




def test_shared_claim_can_reattach_to_larger_existing_allocation(fake_cuda):
    m = PersistentAllocationManager(device=0)
    alloc1, reattached1 = m.claim("eng-A", "kv_pool", 8192, shared=True)
    alloc2, reattached2 = m.claim("eng-A", "kv_pool", 4096, shared=True)

    assert reattached1 is False
    assert reattached2 is True
    assert alloc1.allocation_id == alloc2.allocation_id
    assert alloc2.aligned_size == alloc1.aligned_size


def test_shared_and_exclusive_claims_conflict(fake_cuda):
    m = PersistentAllocationManager(device=0)
    m.claim("eng-A", "kv_pool", 4096)
    with pytest.raises(PersistentClaimConflictError):
        m.claim("eng-A", "kv_pool", 4096, shared=True)

    assert m.unclaim("eng-A", "kv_pool") is True
    m.claim("eng-A", "kv_pool", 4096, shared=True)
    with pytest.raises(PersistentClaimConflictError):
        m.claim("eng-A", "kv_pool", 4096)


def test_release_frees_and_removes(fake_cuda):
    m = PersistentAllocationManager(device=0)
    alloc, _ = m.claim("eng-A", "kv_pool", 4096)
    assert m.release("eng-A", "kv_pool") is True
    assert m.is_claimed("eng-A", "kv_pool") is False
    # Subsequent claim with the same key creates a NEW allocation
    # (not a reattach).
    alloc2, reattached = m.claim("eng-A", "kv_pool", 4096)
    assert reattached is False
    assert alloc.allocation_id != alloc2.allocation_id


def test_release_unknown_is_idempotent(fake_cuda):
    m = PersistentAllocationManager(device=0)
    assert m.release("nope", "nope") is False
    assert m.release("nope", "nope") is False


def test_export_returns_dup_fd(fake_cuda):
    """export() returns a duplicated FD; closing it must not invalidate
    the canonical export_fd in the allocation record."""
    m = PersistentAllocationManager(device=0)
    alloc1, _ = m.claim("eng-A", "kv_pool", 4096)
    _, fd1 = m.export("eng-A", "kv_pool")
    _, fd2 = m.export("eng-A", "kv_pool")
    try:
        assert fd1 != alloc1.export_fd
        assert fd2 != alloc1.export_fd
        assert fd1 != fd2
    finally:
        os.close(fd1)
        os.close(fd2)


def test_export_missing_raises(fake_cuda):
    m = PersistentAllocationManager(device=0)
    with pytest.raises(PersistentNotFoundError):
        m.export("nobody", "nothing")


def test_list_filters_by_engine(fake_cuda):
    m = PersistentAllocationManager(device=0)
    m.claim("eng-A", "kv_pool", 4096)
    m.claim("eng-B", "kv_pool", 4096)
    a_list = list(m.list("eng-A"))
    b_list = list(m.list("eng-B"))
    all_list = list(m.list(None))
    assert len(a_list) == 1 and a_list[0].engine_id == "eng-A"
    assert len(b_list) == 1 and b_list[0].engine_id == "eng-B"
    assert len(all_list) == 2


def test_claim_validation(fake_cuda):
    m = PersistentAllocationManager(device=0)
    with pytest.raises(ValueError):
        m.claim("", "tag", 4096)
    with pytest.raises(ValueError):
        m.claim("eng", "", 4096)
    with pytest.raises(ValueError):
        m.claim("eng", "tag", 0)


# ---------------------------------------------------------------------
# RPC contract: GMS.handle_request integration
# ---------------------------------------------------------------------


class _DummyWriter:
    """Minimal stream-writer stand-in. cleanup_connection eventually
    calls conn.close() which calls writer.close() + wait_closed()."""

    def close(self) -> None:
        pass

    async def wait_closed(self) -> None:
        return None


_dummy_session_seq = 0


def _make_dummy_conn(mode=GrantedLockType.RW) -> Connection:
    global _dummy_session_seq
    _dummy_session_seq += 1
    # Give each connection a unique session_id so per-session claim
    # tracking isn't aliased across tests in the same fixture.
    return Connection(
        reader=None,
        writer=_DummyWriter(),
        mode=mode,
        session_id=f"test-session-{_dummy_session_seq}",
    )


@pytest.fixture
def gms(fake_cuda) -> GMS:
    return GMS(device=0)


def _run(coro):
    return (
        asyncio.get_event_loop().run_until_complete(coro)
        if (asyncio.get_event_loop().is_running() is False)
        else asyncio.run(coro)
    )


def test_claim_release_round_trip_via_rpc(gms):
    """End-to-end: ClaimPersistentAllocationRequest → response →
    ReleasePersistentAllocationRequest → response. Uses a connection
    with no active lock, proving the persistent allow-list bypass
    works."""
    conn = _make_dummy_conn()
    # Connection isn't registered with the FSM, so check_operation
    # will only let through messages in PERSISTENT_ALLOWED. Perfect
    # for what we want to test.
    claim_req = ClaimPersistentAllocationRequest(
        engine_id="eng-X",
        tag="kv_pool",
        size=8192,
    )
    resp, fd, _ = asyncio.run(gms.handle_request(conn, claim_req, lambda: True))
    assert isinstance(resp, ClaimPersistentAllocationResponse)
    assert resp.reattached is False
    assert resp.size == 8192
    assert fd == -1  # claim doesn't pass an FD

    rel_req = ReleasePersistentAllocationRequest(
        engine_id="eng-X",
        tag="kv_pool",
    )
    resp2, _, _ = asyncio.run(gms.handle_request(conn, rel_req, lambda: True))
    assert isinstance(resp2, ReleasePersistentAllocationResponse)
    assert resp2.released is True

def test_repeated_shared_claim_from_one_session_is_idempotent(gms):
    conn = _make_dummy_conn()
    request = ClaimPersistentAllocationRequest(
        engine_id="eng-X", tag="kv_pool", size=8192, shared=True
    )

    first, _, _ = asyncio.run(gms.handle_request(conn, request, lambda: True))
    second, _, _ = asyncio.run(gms.handle_request(conn, request, lambda: True))

    assert isinstance(first, ClaimPersistentAllocationResponse)
    assert isinstance(second, ClaimPersistentAllocationResponse)
    assert second.reattached is True
    assert gms._persistent.shared_claim_count("eng-X", "kv_pool") == 1

    asyncio.run(gms.cleanup_connection(conn))
    assert not gms._persistent.is_claimed("eng-X", "kv_pool")


def test_shared_claimant_cannot_release_peer_allocation_via_rpc(gms):
    first = _make_dummy_conn()
    second = _make_dummy_conn()
    request = ClaimPersistentAllocationRequest(
        engine_id="eng-X", tag="kv_pool", size=8192, shared=True
    )
    asyncio.run(gms.handle_request(first, request, lambda: True))
    asyncio.run(gms.handle_request(second, request, lambda: True))

    response, _, _ = asyncio.run(
        gms.handle_request(
            first,
            ReleasePersistentAllocationRequest("eng-X", "kv_pool"),
            lambda: True,
        )
    )
    assert isinstance(response, ErrorResponse)
    assert "other shared claimants" in response.error
    assert gms._persistent.shared_claim_count("eng-X", "kv_pool") == 2

    asyncio.run(gms.cleanup_connection(first))
    response, _, _ = asyncio.run(
        gms.handle_request(
            second,
            ReleasePersistentAllocationRequest("eng-X", "kv_pool"),
            lambda: True,
        )
    )
    assert isinstance(response, ReleasePersistentAllocationResponse)
    assert response.released is True

def test_export_returns_fd_via_rpc(gms):
    conn = _make_dummy_conn()
    asyncio.run(
        gms.handle_request(
            conn,
            ClaimPersistentAllocationRequest(
                engine_id="eng-X",
                tag="kv_pool",
                size=8192,
            ),
            lambda: True,
        )
    )
    export_req = ExportPersistentAllocationRequest(
        engine_id="eng-X",
        tag="kv_pool",
    )
    resp, fd, _ = asyncio.run(gms.handle_request(conn, export_req, lambda: True))
    try:
        assert isinstance(resp, ExportPersistentAllocationResponse)
        assert fd > 0, "export must pass back a valid FD"
        assert resp.size == 8192
    finally:
        if fd > 0:
            os.close(fd)


def test_conflict_returns_error_response(gms):
    conn = _make_dummy_conn()
    claim = ClaimPersistentAllocationRequest(
        engine_id="eng-X",
        tag="kv_pool",
        size=8192,
    )
    asyncio.run(gms.handle_request(conn, claim, lambda: True))
    # Don't unclaim. Second claim with same key → conflict.
    resp, _, _ = asyncio.run(gms.handle_request(conn, claim, lambda: True))
    assert isinstance(resp, ErrorResponse)
    assert "already claimed" in resp.error


def test_list_via_rpc(gms):
    conn = _make_dummy_conn()
    asyncio.run(
        gms.handle_request(
            conn,
            ClaimPersistentAllocationRequest("eng-A", "kv_pool", 8192),
            lambda: True,
        )
    )
    asyncio.run(
        gms.handle_request(
            conn,
            ClaimPersistentAllocationRequest("eng-B", "kv_pool", 8192),
            lambda: True,
        )
    )
    resp, _, _ = asyncio.run(
        gms.handle_request(
            conn,
            ListPersistentAllocationsRequest(engine_id=None),
            lambda: True,
        )
    )
    assert isinstance(resp, ListPersistentAllocationsResponse)
    engine_ids = sorted(a.engine_id for a in resp.allocations)
    assert engine_ids == ["eng-A", "eng-B"]
    # All currently claimed
    for a in resp.allocations:
        assert a.claimed is True


def test_persistent_rpc_requires_session_claim_for_release_and_export(gms):
    owner = _make_dummy_conn()
    other = _make_dummy_conn()
    asyncio.run(
        gms.handle_request(
            owner,
            ClaimPersistentAllocationRequest("eng-X", "kv_pool", 8192),
            lambda: True,
        )
    )

    resp, fd, _ = asyncio.run(
        gms.handle_request(
            other,
            ExportPersistentAllocationRequest("eng-X", "kv_pool"),
            lambda: True,
        )
    )
    assert isinstance(resp, ErrorResponse)
    assert "not claimed by session" in resp.error
    assert fd == -1

    resp, _, _ = asyncio.run(
        gms.handle_request(
            other,
            ReleasePersistentAllocationRequest("eng-X", "kv_pool"),
            lambda: True,
        )
    )
    assert isinstance(resp, ErrorResponse)
    assert "not claimed by session" in resp.error
    assert gms._persistent.is_claimed("eng-X", "kv_pool")

    resp, _, _ = asyncio.run(
        gms.handle_request(
            owner,
            ReleasePersistentAllocationRequest("eng-X", "kv_pool"),
            lambda: True,
        )
    )
    assert isinstance(resp, ReleasePersistentAllocationResponse)
    assert resp.released is True


def test_list_via_rpc_only_returns_current_session_claims(gms):
    conn1 = _make_dummy_conn()
    conn2 = _make_dummy_conn()
    asyncio.run(
        gms.handle_request(
            conn1,
            ClaimPersistentAllocationRequest("eng-A", "kv_pool", 8192),
            lambda: True,
        )
    )
    asyncio.run(
        gms.handle_request(
            conn2,
            ClaimPersistentAllocationRequest("eng-B", "kv_pool", 8192),
            lambda: True,
        )
    )

    resp1, _, _ = asyncio.run(
        gms.handle_request(
            conn1,
            ListPersistentAllocationsRequest(engine_id=None),
            lambda: True,
        )
    )
    assert isinstance(resp1, ListPersistentAllocationsResponse)
    assert [(a.engine_id, a.tag) for a in resp1.allocations] == [("eng-A", "kv_pool")]

    resp2, _, _ = asyncio.run(
        gms.handle_request(
            conn2,
            ListPersistentAllocationsRequest(engine_id=None),
            lambda: True,
        )
    )
    assert isinstance(resp2, ListPersistentAllocationsResponse)
    assert [(a.engine_id, a.tag) for a in resp2.allocations] == [("eng-B", "kv_pool")]


# ---------------------------------------------------------------------
# Coexistence with the weights flow
# ---------------------------------------------------------------------


def test_persistent_allocations_unaffected_by_rw_abort(gms):
    """A persistent allocation must survive a simulated RW_ABORT
    cleanup of the weights namespace."""
    conn = _make_dummy_conn()
    asyncio.run(
        gms.handle_request(
            conn,
            ClaimPersistentAllocationRequest("eng-X", "kv_pool", 8192),
            lambda: True,
        )
    )
    # Simulate the weights-side cleanup (what would happen on RW_ABORT).
    gms._clear_layout_state()

    # Persistent allocation must still be there.
    resp, _, _ = asyncio.run(
        gms.handle_request(
            conn,
            ListPersistentAllocationsRequest(engine_id="eng-X"),
            lambda: True,
        )
    )
    assert isinstance(resp, ListPersistentAllocationsResponse)
    assert len(resp.allocations) == 1
    assert resp.allocations[0].engine_id == "eng-X"


# ---------------------------------------------------------------------
# Cleanup-on-disconnect: claims are released, allocations persist
# ---------------------------------------------------------------------


def test_cleanup_releases_claims_keeps_allocation(gms):
    """When a session disconnects, its persistent CLAIMS are released
    (so another client can attach next) but the underlying ALLOCATIONS
    persist."""
    conn1 = _make_dummy_conn()
    asyncio.run(
        gms.handle_request(
            conn1,
            ClaimPersistentAllocationRequest("eng-X", "kv_pool", 8192),
            lambda: True,
        )
    )
    assert gms._persistent.is_claimed("eng-X", "kv_pool") is True

    # Simulate disconnect of session 1.
    asyncio.run(gms.cleanup_connection(conn1))
    assert (
        gms._persistent.is_claimed("eng-X", "kv_pool") is False
    ), "claim should be released on disconnect"
    assert (
        gms._persistent.get("eng-X", "kv_pool") is not None
    ), "allocation must persist across disconnect"

    # New session can now attach to the same key — re-attach.
    conn2 = _make_dummy_conn()
    resp, _, _ = asyncio.run(
        gms.handle_request(
            conn2,
            ClaimPersistentAllocationRequest("eng-X", "kv_pool", 8192),
            lambda: True,
        )
    )
    assert isinstance(resp, ClaimPersistentAllocationResponse)
    assert resp.reattached is True


def test_cleanup_releases_multiple_claims(gms):
    """A session that claimed several persistent allocations should
    have ALL its claims released on disconnect."""
    conn = _make_dummy_conn()
    asyncio.run(
        gms.handle_request(
            conn,
            ClaimPersistentAllocationRequest("eng-A", "kv_pool", 4096),
            lambda: True,
        )
    )
    asyncio.run(
        gms.handle_request(
            conn,
            ClaimPersistentAllocationRequest("eng-A", "scratch", 4096),
            lambda: True,
        )
    )
    asyncio.run(
        gms.handle_request(
            conn,
            ClaimPersistentAllocationRequest("eng-B", "kv_pool", 4096),
            lambda: True,
        )
    )
    assert gms._persistent.is_claimed("eng-A", "kv_pool")
    assert gms._persistent.is_claimed("eng-A", "scratch")
    assert gms._persistent.is_claimed("eng-B", "kv_pool")

    asyncio.run(gms.cleanup_connection(conn))
    assert not gms._persistent.is_claimed("eng-A", "kv_pool")
    assert not gms._persistent.is_claimed("eng-A", "scratch")
    assert not gms._persistent.is_claimed("eng-B", "kv_pool")


def test_release_dedupes_session_claim_record(gms):
    """Explicit release_persistent must also drop the session's claim
    record so we don't double-unclaim on subsequent disconnect."""
    conn = _make_dummy_conn()
    asyncio.run(
        gms.handle_request(
            conn,
            ClaimPersistentAllocationRequest("eng-X", "kv_pool", 4096),
            lambda: True,
        )
    )
    # Explicit release.
    asyncio.run(
        gms.handle_request(
            conn,
            ReleasePersistentAllocationRequest("eng-X", "kv_pool"),
            lambda: True,
        )
    )
    assert not gms._persistent.is_claimed("eng-X", "kv_pool")
    # Session record should be empty (or no entry at all).
    record = gms._persistent_claims_by_session.get(conn.session_id, set())
    assert ("eng-X", "kv_pool") not in record

    # cleanup_connection still works without errors.
    asyncio.run(gms.cleanup_connection(conn))


# ---------------------------------------------------------------------
# Daemon-side direct access (VA mapping, read/write contracts)
# ---------------------------------------------------------------------


def test_claim_records_daemon_va(fake_cuda):
    """After a successful claim, va_daemon must be populated so that
    daemon-side direct access is available."""
    m = PersistentAllocationManager(device=0)
    alloc, _ = m.claim("eng-X", "kv_pool", 4096)
    assert (
        alloc.va_daemon != 0
    ), "claim must record a daemon-side VA when cuMemMap succeeds"
    assert m.daemon_va("eng-X", "kv_pool") == alloc.va_daemon


def test_daemon_va_missing_raises(fake_cuda):
    m = PersistentAllocationManager(device=0)
    with pytest.raises(PersistentNotFoundError):
        m.daemon_va("nope", "nope")


def test_read_block_out_of_range_raises(fake_cuda):
    """Range checks must run BEFORE any CUDA op. They run regardless
    of whether va_daemon is real or fake."""
    m = PersistentAllocationManager(device=0)
    m.claim("eng-X", "kv_pool", 4096)
    with pytest.raises(ValueError, match="size must be > 0"):
        m.read_block("eng-X", "kv_pool", offset=0, size=0)
    with pytest.raises(ValueError, match="out of range"):
        m.read_block("eng-X", "kv_pool", offset=0, size=5000)
    with pytest.raises(ValueError, match="out of range"):
        m.read_block("eng-X", "kv_pool", offset=-1, size=8)


def test_write_block_out_of_range_raises(fake_cuda):
    m = PersistentAllocationManager(device=0)
    m.claim("eng-X", "kv_pool", 4096)
    with pytest.raises(ValueError, match="non-empty"):
        m.write_block("eng-X", "kv_pool", offset=0, data=b"")
    with pytest.raises(ValueError, match="out of range"):
        m.write_block("eng-X", "kv_pool", offset=0, data=b"x" * 5000)


def test_read_block_unmapped_raises(monkeypatch, fake_cuda):
    """If cuMemMap failed at claim time, va_daemon is 0 and
    read/write_block must raise rather than dereference 0."""

    # Force the daemon-side mapping to fail.
    def fail_map(va, size, handle):
        raise RuntimeError("simulated cuMemMap failure")

    monkeypatch.setattr(server_persistent, "cumem_map", fail_map)

    m = PersistentAllocationManager(device=0)
    alloc, _ = m.claim("eng-X", "kv_pool", 4096)
    assert alloc.va_daemon == 0, "va_daemon must be 0 when map failed"

    with pytest.raises(RuntimeError, match="no daemon-side VA"):
        m.read_block("eng-X", "kv_pool", offset=0, size=64)
    with pytest.raises(RuntimeError, match="no daemon-side VA"):
        m.write_block("eng-X", "kv_pool", offset=0, data=b"x" * 64)
    with pytest.raises(RuntimeError, match="no daemon-side VA"):
        m.daemon_va("eng-X", "kv_pool")


def test_release_tears_down_daemon_va(fake_cuda, monkeypatch):
    """release() must unmap the daemon-side VA. Track that cumem_unmap
    + cumem_address_free are called with the recorded VA."""
    unmap_calls = []
    free_calls = []
    monkeypatch.setattr(
        server_persistent,
        "cumem_unmap",
        lambda va, size: unmap_calls.append((va, size)),
    )
    monkeypatch.setattr(
        server_persistent,
        "cumem_address_free",
        lambda va, size: free_calls.append((va, size)),
    )
    m = PersistentAllocationManager(device=0)
    alloc, _ = m.claim("eng-X", "kv_pool", 4096)
    expected_va = alloc.va_daemon
    expected_size = alloc.aligned_size

    assert m.release("eng-X", "kv_pool") is True
    assert (expected_va, expected_size) in unmap_calls
    assert (expected_va, expected_size) in free_calls
