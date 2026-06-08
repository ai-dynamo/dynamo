# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import random
import threading
import time

import pytest
from gpu_memory_service.common.locks import GrantedLockType
from gpu_memory_service.common.protocol.messages import (
    AcquireKVBlockLeasesRequest,
    AcquireKVBlockLeasesResponse,
    ErrorResponse,
    InitKVLeaseNamespaceRequest,
    InitKVLeaseNamespaceResponse,
    ListKVBlockLeasesRequest,
    ListKVBlockLeasesResponse,
    PinKVBlockLeasesRequest,
    ReleaseKVBlockLeasesRequest,
    ReleaseKVBlockLeasesResponse,
    SealKVBlockLeasesRequest,
    SealKVBlockLeasesResponse,
    UnpinKVBlockLeasesRequest,
)
from gpu_memory_service.server.fsm import Connection
from gpu_memory_service.server.gms import GMS
from gpu_memory_service.server.kv_leases import (
    STATE_FREE,
    STATE_RETIRING,
    STATE_SEALED,
    STATE_WRITING,
    KVLeaseConflictError,
    KVLeaseManager,
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.none,
    pytest.mark.gpu_0,
]


class _DummyWriter:
    def close(self) -> None:
        pass

    async def wait_closed(self) -> None:
        return None


def _make_conn(session_id: str = "kv-lease-test") -> Connection:
    return Connection(
        reader=None,
        writer=_DummyWriter(),
        mode=GrantedLockType.RW_PERSISTENT,
        session_id=session_id,
    )


def _run(coro):
    return asyncio.run(coro)


def test_strict_preferred_keeps_unselected_free_blocks():
    manager = KVLeaseManager()
    manager.init_namespace("ns", 5, reserved_blocks=[0])

    first = manager.acquire(
        "ns", "engine-a", 1, preferred_blocks=[1, 2, 3], strict_preferred=True
    )
    assert [lease.block_id for lease in first] == [1]
    assert manager.count_free("ns") == 3

    second = manager.acquire(
        "ns", "engine-b", 2, preferred_blocks=[2, 3], strict_preferred=True
    )
    assert [lease.block_id for lease in second] == [2, 3]
    assert manager.count_free("ns") == 1


def test_existing_namespace_accepts_smaller_local_attach():
    manager = KVLeaseManager()
    assert manager.init_namespace("shared", 8, reserved_blocks=[0]) == 8

    # A second engine may have a smaller local allocator after attaching to an
    # existing shared persistent KV pool. The existing larger namespace remains
    # authoritative; the smaller engine will acquire only its local block IDs.
    assert manager.init_namespace("shared", 4, reserved_blocks=[0]) == 8

    with pytest.raises(KVLeaseConflictError):
        manager.init_namespace("shared", 9)


def test_pin_release_retiring_and_generation_checks():
    manager = KVLeaseManager()
    manager.init_namespace("ns", 4, reserved_blocks=[0])

    lease = manager.acquire("ns", "writer", 1, preferred_blocks=[1])[0]
    sealed = manager.seal("ns", "writer", [lease.block_id], [lease.generation])[0]
    assert sealed.state == STATE_SEALED

    pinned = manager.pin("ns", "reader", [lease.block_id], [lease.generation])[0]
    assert pinned.read_pins == 1

    retiring = manager.release("ns", "writer", [lease.block_id], [lease.generation])[0]
    assert retiring.state == STATE_RETIRING
    assert manager.count_free("ns") == 2

    with pytest.raises(KVLeaseConflictError):
        manager.acquire(
            "ns", "other", 1, preferred_blocks=[lease.block_id], strict_preferred=True
        )

    unpinned = manager.unpin("ns", "reader", [lease.block_id], [lease.generation])[0]
    assert unpinned.state == STATE_FREE

    replacement = manager.acquire(
        "ns", "other", 1, preferred_blocks=[lease.block_id], strict_preferred=True
    )[0]
    assert replacement.block_id == lease.block_id
    assert replacement.generation == lease.generation + 1

    with pytest.raises(KVLeaseConflictError):
        manager.release("ns", "writer", [lease.block_id], [lease.generation])


def test_manager_prevents_duplicate_writers_under_stress():
    manager = KVLeaseManager()
    manager.init_namespace("stress", 17, reserved_blocks=[0])
    active: set[int] = set()
    active_lock = threading.Lock()
    errors: list[BaseException] = []

    def worker(worker_id: int) -> None:
        rng = random.Random(worker_id)
        owner = f"engine-{worker_id}"
        for _ in range(300):
            try:
                count = rng.randint(1, 3)
                leases = manager.acquire(
                    "stress",
                    owner,
                    count,
                    preferred_blocks=list(range(1, 17)),
                    strict_preferred=True,
                )
            except KVLeaseConflictError:
                time.sleep(rng.random() * 0.0002)
                continue
            try:
                with active_lock:
                    for lease in leases:
                        if lease.block_id in active:
                            raise AssertionError(
                                f"duplicate writer for block {lease.block_id}"
                            )
                        active.add(lease.block_id)
                time.sleep(rng.random() * 0.0002)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)
            finally:
                with active_lock:
                    for lease in leases:
                        active.discard(lease.block_id)
                manager.release(
                    "stress",
                    owner,
                    [lease.block_id for lease in leases],
                    [lease.generation for lease in leases],
                )

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert active == set()
    assert manager.count_free("stress") == 16


def test_lease_rpc_round_trip_through_persistent_session():
    gms = GMS(device=0)
    conn = _make_conn()

    resp, fd, _ = _run(
        gms.handle_request(
            conn,
            InitKVLeaseNamespaceRequest("rpc", 4, reserved_blocks=[0]),
            lambda: True,
        )
    )
    assert isinstance(resp, InitKVLeaseNamespaceResponse)
    assert resp.total_blocks == 4
    assert fd == -1

    resp, _, _ = _run(
        gms.handle_request(
            conn,
            AcquireKVBlockLeasesRequest(
                "rpc", "engine-a", 2, preferred_blocks=[1, 2, 3], strict_preferred=True
            ),
            lambda: True,
        )
    )
    assert isinstance(resp, AcquireKVBlockLeasesResponse)
    assert [block.block_id for block in resp.blocks] == [1, 2]

    resp, _, _ = _run(
        gms.handle_request(
            conn,
            SealKVBlockLeasesRequest(
                "rpc",
                "engine-a",
                [block.block_id for block in resp.blocks],
                [block.generation for block in resp.blocks],
            ),
            lambda: True,
        )
    )
    assert isinstance(resp, SealKVBlockLeasesResponse)
    assert {block.state for block in resp.blocks} == {STATE_SEALED}

    resp, _, _ = _run(
        gms.handle_request(
            conn,
            ListKVBlockLeasesRequest("rpc"),
            lambda: True,
        )
    )
    assert isinstance(resp, ListKVBlockLeasesResponse)
    states = {block.block_id: block.state for block in resp.blocks}
    assert states[1] == STATE_SEALED
    assert states[2] == STATE_SEALED
    assert states[3] == STATE_FREE

    resp, _, _ = _run(
        gms.handle_request(
            conn,
            ReleaseKVBlockLeasesRequest(
                "rpc",
                "engine-a",
                [1, 2],
                [block.generation for block in resp.blocks if block.block_id in (1, 2)],
            ),
            lambda: True,
        )
    )
    assert isinstance(resp, ReleaseKVBlockLeasesResponse)
    assert {block.state for block in resp.blocks} == {STATE_FREE}


def test_lease_rpc_conflicts_return_error_response():
    gms = GMS(device=0)
    conn = _make_conn()

    _run(
        gms.handle_request(
            conn,
            InitKVLeaseNamespaceRequest("rpc-conflict", 2),
            lambda: True,
        )
    )
    _run(
        gms.handle_request(
            conn,
            AcquireKVBlockLeasesRequest("rpc-conflict", "engine-a", 1),
            lambda: True,
        )
    )
    resp, _, _ = _run(
        gms.handle_request(
            conn,
            AcquireKVBlockLeasesRequest("rpc-conflict", "engine-b", 2),
            lambda: True,
        )
    )
    assert isinstance(resp, ErrorResponse)
    assert "leaseable" in resp.error
    assert STATE_WRITING in {
        block.state for block in gms.kv_leases.list("rpc-conflict")
    }


def test_lease_rpc_rejects_owner_spoofing_across_sessions():
    gms = GMS(device=0)
    owner = _make_conn("owner-session")
    other = _make_conn("other-session")

    _run(
        gms.handle_request(
            owner,
            InitKVLeaseNamespaceRequest("rpc-owner-bind", 2),
            lambda: True,
        )
    )
    resp, _, _ = _run(
        gms.handle_request(
            owner,
            AcquireKVBlockLeasesRequest("rpc-owner-bind", "engine-a", 1),
            lambda: True,
        )
    )
    assert isinstance(resp, AcquireKVBlockLeasesResponse)

    resp, _, _ = _run(
        gms.handle_request(
            other,
            ReleaseKVBlockLeasesRequest(
                "rpc-owner-bind",
                "engine-a",
                [resp.blocks[0].block_id],
                [resp.blocks[0].generation],
            ),
            lambda: True,
        )
    )
    assert isinstance(resp, ErrorResponse)
    assert "owner not bound" in resp.error
    assert gms.kv_leases.count_free("rpc-owner-bind") == 1


def test_lease_rpc_rejects_reader_spoofing_across_sessions():
    gms = GMS(device=0)
    owner = _make_conn("owner-session")
    reader = _make_conn("reader-session")
    other = _make_conn("other-session")

    _run(
        gms.handle_request(
            owner,
            InitKVLeaseNamespaceRequest("rpc-reader-bind", 2),
            lambda: True,
        )
    )
    acquired, _, _ = _run(
        gms.handle_request(
            owner,
            AcquireKVBlockLeasesRequest("rpc-reader-bind", "engine-a", 1),
            lambda: True,
        )
    )
    assert isinstance(acquired, AcquireKVBlockLeasesResponse)
    sealed, _, _ = _run(
        gms.handle_request(
            owner,
            SealKVBlockLeasesRequest(
                "rpc-reader-bind",
                "engine-a",
                [acquired.blocks[0].block_id],
                [acquired.blocks[0].generation],
            ),
            lambda: True,
        )
    )
    assert isinstance(sealed, SealKVBlockLeasesResponse)

    pinned, _, _ = _run(
        gms.handle_request(
            reader,
            PinKVBlockLeasesRequest(
                "rpc-reader-bind",
                "reader-a",
                [sealed.blocks[0].block_id],
                [sealed.blocks[0].generation],
            ),
            lambda: True,
        )
    )
    assert pinned.blocks[0].read_pins == 1

    resp, _, _ = _run(
        gms.handle_request(
            other,
            UnpinKVBlockLeasesRequest(
                "rpc-reader-bind",
                "reader-a",
                [sealed.blocks[0].block_id],
                [sealed.blocks[0].generation],
            ),
            lambda: True,
        )
    )
    assert isinstance(resp, ErrorResponse)
    assert "reader not bound" in resp.error
    assert gms.kv_leases.list("rpc-reader-bind")[0].read_pins == 1
