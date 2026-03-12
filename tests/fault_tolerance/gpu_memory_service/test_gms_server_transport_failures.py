# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Targeted GMS fault-tolerance unit tests."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest
from cuda.bindings import driver as cuda

from gpu_memory_service.client.memory_manager import (
    GMSClientMemoryManager,
    LocalMapping,
)
from gpu_memory_service.common import cuda_vmm_utils
from gpu_memory_service.common.protocol.messages import (
    CommitRequest,
    CommitResponse,
    GetLockStateRequest,
    GetLockStateResponse,
    HandshakeRequest,
)
from gpu_memory_service.common.types import (
    GrantedLockType,
    RequestedLockType,
    ServerState,
    StateEvent,
)
from gpu_memory_service.server.allocations import GMSAllocationManager
from gpu_memory_service.server.gms import GMS
from gpu_memory_service.server.rpc import GMSRPCServer
from gpu_memory_service.server.session import GMSSessionManager
from gpu_memory_service.server.session import Connection

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.fault_tolerance,
]


def test_cumem_create_tolerate_oom_returns_handle_on_success(monkeypatch):
    monkeypatch.setattr(
        cuda_vmm_utils.cuda,
        "cuMemCreate",
        lambda size, prop, flags: (cuda.CUresult.CUDA_SUCCESS, 1234),
    )

    allocated, handle = cuda_vmm_utils.cumem_create_tolerate_oom(4096, 0)

    assert allocated
    assert handle == 1234


def test_cumem_create_tolerate_oom_returns_false_on_oom(monkeypatch):
    monkeypatch.setattr(
        cuda_vmm_utils.cuda,
        "cuMemCreate",
        lambda size, prop, flags: (cuda.CUresult.CUDA_ERROR_OUT_OF_MEMORY, 0),
    )

    allocated, handle = cuda_vmm_utils.cumem_create_tolerate_oom(4096, 0)

    assert not allocated
    assert handle == 0


def test_cumem_export_to_shareable_handle_returns_fd(monkeypatch):
    monkeypatch.setattr(
        cuda_vmm_utils.cuda,
        "cuMemExportToShareableHandle",
        lambda handle, handle_type, flags: (cuda.CUresult.CUDA_SUCCESS, 77),
    )

    fd = cuda_vmm_utils.cumem_export_to_shareable_handle(1234)

    assert fd == 77


class _DummyReader:
    pass


class _DummyWriter:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        return None

    def get_extra_info(self, _name: str):
        return None


@dataclass
class _FakeHandler:
    committed_epoch_id: int | None = None
    active_rw_epoch_id: int | None = None
    rw_connect_calls: int = 0
    rw_abort_calls: int = 0
    commit_calls: int = 0

    def on_rw_connect(self) -> None:
        self.rw_connect_calls += 1
        self.active_rw_epoch_id = 2
        self.committed_epoch_id = None

    def on_rw_abort(self) -> None:
        self.rw_abort_calls += 1
        self.active_rw_epoch_id = None

    def on_commit(self) -> None:
        self.commit_calls += 1
        self.committed_epoch_id = self.active_rw_epoch_id or 1
        self.active_rw_epoch_id = None

    def handle_get_lock_state(
        self,
        has_rw: bool,
        ro_count: int,
        waiting_writers: int,
        committed: bool,
    ) -> GetLockStateResponse:
        return GetLockStateResponse(
            state=(
                "RW"
                if has_rw
                else "RO" if ro_count else "COMMITTED" if committed else "EMPTY"
            ),
            has_rw_session=has_rw,
            ro_session_count=ro_count,
            waiting_writers=waiting_writers,
            committed=committed,
            is_ready=committed and not has_rw,
        )


class _FakeGMS:
    def __init__(self, handler: _FakeHandler | None = None):
        self.handler = handler or _FakeHandler()
        self._sessions = GMSSessionManager()
        if self.handler.committed_epoch_id is not None:
            self._sessions._locking._committed = True

    @property
    def committed(self) -> bool:
        return self._sessions.snapshot().committed

    async def acquire_lock(self, mode, timeout_ms, session_id):
        return await self._sessions.acquire_lock(mode, timeout_ms, session_id)

    async def cancel_connect(self, session_id, mode):
        await self._sessions.cancel_connect(session_id, mode)

    def on_connect(self, conn: Connection) -> None:
        if conn.mode == GrantedLockType.RW:
            self.handler.on_rw_connect()
        self._sessions.on_connect(conn)

    async def cleanup_connection(self, conn: Connection | None) -> None:
        event = self._sessions.begin_cleanup(conn)
        if event == StateEvent.RW_ABORT:
            self.handler.on_rw_abort()
        await self._sessions.finish_cleanup(conn)

    async def handle_request(self, conn: Connection, msg, _is_connected):
        if isinstance(msg, GetLockStateRequest):
            snapshot = self._sessions.snapshot()
            return (
                self.handler.handle_get_lock_state(
                    snapshot.has_rw_session,
                    snapshot.ro_session_count,
                    snapshot.waiting_writers,
                    snapshot.committed,
                ),
                -1,
                False,
            )
        if isinstance(msg, CommitRequest):
            self.handler.on_commit()
            self._sessions.on_commit(conn)
            return CommitResponse(success=True), -1, True
        raise AssertionError(f"Unexpected request type in test: {type(msg)}")


def _make_server(handler: _FakeHandler | None = None) -> GMSRPCServer:
    server = object.__new__(GMSRPCServer)
    server.socket_path = "/tmp/gms-test.sock"
    server.device = 0
    server._gms = _FakeGMS(handler)
    server._server = None
    return server


@pytest.mark.asyncio
async def test_handshake_success_send_failure_cleans_up_rw_state(monkeypatch):
    server = _make_server()
    reader = _DummyReader()
    writer = _DummyWriter()

    async def fake_recv_message(_reader, _buffer):
        return HandshakeRequest(lock_type=RequestedLockType.RW), -1, bytearray()

    async def fake_acquire_lock(_mode, _timeout_ms, _session_id):
        server._gms._sessions._reserved_rw_session_id = _session_id
        return GrantedLockType.RW

    async def fake_send_message(_writer, _msg, _fd=-1):
        raise BrokenPipeError("handshake reply failed")

    monkeypatch.setattr("gpu_memory_service.server.rpc.recv_message", fake_recv_message)
    monkeypatch.setattr(server._gms, "acquire_lock", fake_acquire_lock)
    monkeypatch.setattr("gpu_memory_service.server.rpc.send_message", fake_send_message)

    conn = await server._do_handshake(reader, writer, "session_1")

    assert conn is None
    assert server._gms._sessions._locking.rw_conn is None
    assert server._gms._sessions.state == ServerState.EMPTY
    assert server._gms.handler.rw_connect_calls == 1
    assert server._gms.handler.rw_abort_calls == 1
    assert writer.closed


@pytest.mark.asyncio
async def test_rw_lock_is_reserved_until_connect():
    sessions = GMSSessionManager()

    first = await sessions.acquire_lock(
        RequestedLockType.RW,
        timeout_ms=50,
        session_id="session_1",
    )
    second = await sessions.acquire_lock(
        RequestedLockType.RW,
        timeout_ms=50,
        session_id="session_2",
    )

    assert first == GrantedLockType.RW
    assert second is None

    await sessions.cancel_connect("session_1", GrantedLockType.RW)


@pytest.mark.asyncio
async def test_reader_cannot_acquire_while_rw_is_reserved_before_connect():
    sessions = GMSSessionManager()
    sessions._locking._committed = True

    granted = await sessions.acquire_lock(
        RequestedLockType.RW,
        timeout_ms=50,
        session_id="writer_1",
    )
    assert granted == GrantedLockType.RW

    reader = await sessions.acquire_lock(
        RequestedLockType.RO,
        timeout_ms=50,
        session_id="reader_1",
    )
    assert reader is None

    await sessions.cancel_connect("writer_1", GrantedLockType.RW)


@pytest.mark.asyncio
async def test_reader_waiter_wakes_when_waiting_writer_times_out():
    sessions = GMSSessionManager()
    sessions._locking._committed = True

    existing_reader = Connection(
        reader=_DummyReader(),
        writer=_DummyWriter(),
        mode=GrantedLockType.RO,
        session_id="reader_1",
        recv_buffer=bytearray(),
    )
    sessions.on_connect(existing_reader)

    writer_task = asyncio.create_task(
        sessions.acquire_lock(
            RequestedLockType.RW,
            timeout_ms=50,
            session_id="writer_1",
        )
    )
    await asyncio.sleep(0)
    reader_task = asyncio.create_task(
        sessions.acquire_lock(
            RequestedLockType.RO,
            timeout_ms=200,
            session_id="reader_2",
        )
    )

    assert await writer_task is None
    assert await reader_task == GrantedLockType.RO

    event = sessions.begin_cleanup(existing_reader)
    assert event == StateEvent.RO_DISCONNECT
    await sessions.finish_cleanup(existing_reader)


@pytest.mark.asyncio
async def test_rw_or_ro_waiter_becomes_rw_after_writer_abort():
    sessions = GMSSessionManager()

    writer_mode = await sessions.acquire_lock(
        RequestedLockType.RW,
        timeout_ms=50,
        session_id="writer_1",
    )
    assert writer_mode == GrantedLockType.RW

    writer = Connection(
        reader=_DummyReader(),
        writer=_DummyWriter(),
        mode=GrantedLockType.RW,
        session_id="writer_1",
        recv_buffer=bytearray(),
    )
    sessions.on_connect(writer)

    waiter = asyncio.create_task(
        sessions.acquire_lock(
            RequestedLockType.RW_OR_RO,
            timeout_ms=200,
            session_id="waiter_1",
        )
    )
    await asyncio.sleep(0)
    assert not waiter.done()

    event = sessions.begin_cleanup(writer)
    assert event == StateEvent.RW_ABORT
    await sessions.finish_cleanup(writer)

    assert await waiter == GrantedLockType.RW
    await sessions.cancel_connect("waiter_1", GrantedLockType.RW)


@pytest.mark.asyncio
async def test_gms_clears_aborted_rw_epoch_before_waking_waiters():
    gms = object.__new__(GMS)
    cleanup_order: list[str] = []
    conn = Connection(
        reader=_DummyReader(),
        writer=_DummyWriter(),
        mode=GrantedLockType.RW,
        session_id="session_1",
        recv_buffer=bytearray(),
    )

    def begin_cleanup(self, cleanup_conn):
        cleanup_order.append("begin_cleanup")
        return StateEvent.RW_ABORT

    async def finish_cleanup(self, cleanup_conn):
        cleanup_order.append("finish_cleanup")

    def on_rw_abort(self):
        cleanup_order.append("on_rw_abort")
        return 7

    def clear_all_allocations(self, epoch_id):
        cleanup_order.append(f"clear_{epoch_id}")

    gms._sessions = type(
        "_Sessions",
        (),
        {
            "begin_cleanup": begin_cleanup,
            "finish_cleanup": finish_cleanup,
        },
    )()
    gms._epochs = type(
        "_Epochs",
        (),
        {"on_rw_abort": on_rw_abort},
    )()
    gms._allocations = type(
        "_Allocations",
        (),
        {"clear_all_allocations": clear_all_allocations},
    )()

    await gms.cleanup_connection(conn)

    assert cleanup_order == [
        "begin_cleanup",
        "on_rw_abort",
        "clear_7",
        "finish_cleanup",
    ]


@pytest.mark.asyncio
async def test_request_response_send_failure_disconnects_without_error_response(
    monkeypatch,
):
    handler = _FakeHandler(committed_epoch_id=1)
    server = _make_server(handler)
    server._gms._sessions._locking._committed = True

    reader = _DummyReader()
    writer = _DummyWriter()
    conn = Connection(
        reader=reader,
        writer=writer,
        mode=GrantedLockType.RO,
        session_id="session_2",
        recv_buffer=bytearray(),
    )
    server._gms._sessions._locking.transition(StateEvent.RO_CONNECT, conn)

    recv_calls = 0
    sent_messages: list[object] = []

    async def fake_recv_message(_reader, _buffer):
        nonlocal recv_calls
        recv_calls += 1
        return GetLockStateRequest(), -1, bytearray()

    async def fake_send_message(_writer, msg, _fd=-1):
        sent_messages.append(msg)
        raise BrokenPipeError("response send failed")

    monkeypatch.setattr("gpu_memory_service.server.rpc.recv_message", fake_recv_message)
    monkeypatch.setattr("gpu_memory_service.server.rpc.send_message", fake_send_message)

    await server._request_loop(conn)
    await server._gms.cleanup_connection(conn)

    assert recv_calls == 1
    assert len(sent_messages) == 1
    assert isinstance(sent_messages[0], GetLockStateResponse)
    assert conn not in server._gms._sessions._locking.ro_conns
    assert server._gms._sessions.state == ServerState.COMMITTED
    assert writer.closed


@pytest.mark.asyncio
async def test_post_commit_response_send_failure_stays_committed(monkeypatch):
    handler = _FakeHandler(active_rw_epoch_id=2)
    server = _make_server(handler)

    reader = _DummyReader()
    writer = _DummyWriter()
    conn = Connection(
        reader=reader,
        writer=writer,
        mode=GrantedLockType.RW,
        session_id="session_3",
        recv_buffer=bytearray(),
    )
    server._gms._sessions._locking.transition(StateEvent.RW_CONNECT, conn)

    recv_calls = 0
    sent_messages: list[object] = []

    async def fake_recv_message(_reader, _buffer):
        nonlocal recv_calls
        recv_calls += 1
        return CommitRequest(), -1, bytearray()

    async def fake_send_message(_writer, msg, _fd=-1):
        sent_messages.append(msg)
        raise BrokenPipeError("commit reply failed")

    monkeypatch.setattr("gpu_memory_service.server.rpc.recv_message", fake_recv_message)
    monkeypatch.setattr("gpu_memory_service.server.rpc.send_message", fake_send_message)

    await server._request_loop(conn)
    await server._gms.cleanup_connection(conn)

    assert recv_calls == 1
    assert len(sent_messages) == 1
    assert handler.commit_calls == 1
    assert server._gms._sessions._locking.rw_conn is None
    assert server._gms._sessions.snapshot().committed
    assert server._gms._sessions.state == ServerState.COMMITTED
    assert writer.closed


@pytest.mark.asyncio
async def test_allocate_rejects_non_positive_size_before_cuda():
    manager = object.__new__(GMSAllocationManager)
    manager._device = 0
    manager._allocations = {}
    manager._granularity = 1
    manager._allocation_retry_interval = 0.0
    manager._allocation_retry_timeout = None

    with pytest.raises(ValueError, match="size must be > 0"):
        await manager.allocate(0, epoch_id=1, tag="weights", is_connected=None)


@pytest.mark.asyncio
async def test_allocate_aborts_retry_when_writer_disconnects(monkeypatch):
    manager = object.__new__(GMSAllocationManager)
    manager._device = 0
    manager._allocations = {}
    manager._granularity = 1
    manager._allocation_retry_interval = 0.0
    manager._allocation_retry_timeout = None

    checks = 0

    def is_connected() -> bool:
        nonlocal checks
        checks += 1
        return checks < 2

    monkeypatch.setattr(
        "gpu_memory_service.server.allocations.cumem_create_tolerate_oom",
        lambda size, device: (False, 0),
    )

    with pytest.raises(
        ConnectionAbortedError, match="RW client disconnected during allocation retry"
    ):
        await manager.allocate(1, epoch_id=1, tag="weights", is_connected=is_connected)


def test_commit_failure_after_local_unmap_keeps_preserved_unmapped_state(monkeypatch):
    manager = object.__new__(GMSClientMemoryManager)
    manager.socket_path = "/tmp/gms-test.sock"
    manager.device = 0
    manager._client = type(
        "_FailingClient",
        (),
        {
            "commit": lambda self: (_ for _ in ()).throw(
                ConnectionError("commit failed after local unmap")
            ),
            "is_connected": True,
        },
    )()
    manager._mappings = {
        0x1000: LocalMapping(
            allocation_id="alloc_1",
            va=0x1000,
            size=4096,
            aligned_size=4096,
            handle=11,
            tag="weights",
        ),
        0x2000: LocalMapping(
            allocation_id="alloc_2",
            va=0x2000,
            size=4096,
            aligned_size=4096,
            handle=0,
            tag="weights",
        ),
    }
    manager._inverse_mapping = {"alloc_1": 0x1000, "alloc_2": 0x2000}
    manager._unmapped = False
    manager._granted_lock_type = GrantedLockType.RW
    manager._va_preserved = False
    manager._last_memory_layout_hash = ""
    manager.granularity = 4096

    unmapped_vas: list[int] = []

    monkeypatch.setattr(
        "gpu_memory_service.client.memory_manager.cuda_synchronize", lambda: None
    )

    def fake_unmap_va(self, va: int) -> None:
        unmapped_vas.append(va)
        self._mappings[va] = self._mappings[va].with_handle(0)

    monkeypatch.setattr(GMSClientMemoryManager, "unmap_va", fake_unmap_va)

    with pytest.raises(ConnectionError, match="commit failed after local unmap"):
        manager.commit()

    assert unmapped_vas == [0x1000]
    assert manager._mappings[0x1000].handle == 0
    assert manager._mappings[0x2000].handle == 0
    assert manager._va_preserved
    assert manager._unmapped
    assert manager._client is not None


def test_successful_commit_clears_rw_mode_before_local_cleanup(monkeypatch):
    manager = object.__new__(GMSClientMemoryManager)
    manager.socket_path = "/tmp/gms-test.sock"
    manager.device = 0
    manager._client = type(
        "_CommittingClient",
        (),
        {
            "commit": lambda self: True,
            "is_connected": True,
        },
    )()
    manager._mappings = {
        0x1000: LocalMapping(
            allocation_id="alloc_1",
            va=0x1000,
            size=4096,
            aligned_size=4096,
            handle=11,
            tag="weights",
        ),
    }
    manager._inverse_mapping = {"alloc_1": 0x1000}
    manager._unmapped = False
    manager._granted_lock_type = GrantedLockType.RW
    manager._va_preserved = False
    manager._last_memory_layout_hash = ""
    manager.granularity = 4096

    free_handle_calls: list[str] = []

    monkeypatch.setattr(
        "gpu_memory_service.client.memory_manager.cuda_synchronize", lambda: None
    )

    def fake_unmap_va(self, va: int) -> None:
        self._mappings[va] = self._mappings[va].with_handle(0)

    def fake_free_va(self, va: int) -> None:
        mapping = self._mappings.pop(va)
        self._inverse_mapping.pop(mapping.allocation_id, None)

    def fake_free_handle(self, allocation_id: str) -> bool:
        free_handle_calls.append(allocation_id)
        return True

    monkeypatch.setattr(GMSClientMemoryManager, "unmap_va", fake_unmap_va)
    monkeypatch.setattr(GMSClientMemoryManager, "free_va", fake_free_va)
    monkeypatch.setattr(GMSClientMemoryManager, "free_handle", fake_free_handle)

    assert manager.commit()
    assert manager._client is None
    assert manager._granted_lock_type is None
    assert manager._unmapped
    assert manager._va_preserved

    manager.destroy_mapping(0x1000)
    assert free_handle_calls == []


def test_destroy_mapping_keeps_local_state_when_server_free_fails(monkeypatch):
    manager = object.__new__(GMSClientMemoryManager)
    manager.socket_path = "/tmp/gms-test.sock"
    manager.device = 0
    manager._client = None
    manager._mappings = {}
    manager._inverse_mapping = {}
    manager._granted_lock_type = GrantedLockType.RW
    manager._unmapped = False
    manager._va_preserved = False
    manager._last_memory_layout_hash = ""
    mapping = LocalMapping(
        allocation_id="alloc_1",
        va=0x1000,
        size=4096,
        aligned_size=4096,
        handle=77,
        tag="weights",
    )
    manager._track_mapping(mapping)

    monkeypatch.setattr(
        manager,
        "free_handle",
        lambda allocation_id: (_ for _ in ()).throw(RuntimeError("server free failed")),
    )

    with pytest.raises(RuntimeError, match="server free failed"):
        manager.destroy_mapping(mapping.va)

    assert manager.mappings[mapping.va] == mapping


def test_remap_all_vas_releases_imported_handles_before_local_state_changes(
    monkeypatch,
):
    manager = object.__new__(GMSClientMemoryManager)
    manager.socket_path = "/tmp/gms-test.sock"
    manager.device = 0
    manager._client = None
    manager._mappings = {}
    manager._inverse_mapping = {}
    manager.granularity = 4096
    manager._granted_lock_type = GrantedLockType.RO
    manager._last_memory_layout_hash = "stable"
    manager._unmapped = True
    manager._va_preserved = True
    monkeypatch.setattr(
        "gpu_memory_service.client.memory_manager.cuda_set_current_device",
        lambda device: None,
    )
    manager._track_mapping(
        LocalMapping(
            allocation_id="alloc_1",
            va=0x1000,
            size=4096,
            aligned_size=4096,
            handle=0,
            tag="weights",
        )
    )
    manager._track_mapping(
        LocalMapping(
            allocation_id="alloc_2",
            va=0x2000,
            size=4096,
            aligned_size=4096,
            handle=0,
            tag="weights",
        )
    )

    monkeypatch.setattr(manager, "get_memory_layout_hash", lambda: "stable")
    monkeypatch.setattr(
        manager,
        "get_handle_info",
        lambda allocation_id: type("_Info", (), {"aligned_size": 4096})(),
    )
    export_fds = iter([11])
    monkeypatch.setattr(
        manager, "export_handle", lambda allocation_id: next(export_fds)
    )
    monkeypatch.setattr(
        "gpu_memory_service.client.memory_manager.cumem_import_from_shareable_handle_close_fd",
        lambda fd: 71,
    )
    released_handles: list[int] = []
    monkeypatch.setattr(
        "gpu_memory_service.client.memory_manager.cumem_release",
        released_handles.append,
    )

    with pytest.raises(StopIteration):
        manager.remap_all_vas()

    assert released_handles == [71]
    assert manager.mappings[0x1000].handle == 0
    assert manager.mappings[0x2000].handle == 0
    assert manager.is_unmapped


def test_disconnect_clears_local_state_even_if_close_fails():
    manager = object.__new__(GMSClientMemoryManager)
    manager.socket_path = "/tmp/gms-test.sock"
    manager.device = 0
    manager._granted_lock_type = GrantedLockType.RO
    manager._client = type(
        "_Client",
        (),
        {"close": lambda self: (_ for _ in ()).throw(ConnectionError("close failed"))},
    )()
    manager._mappings = {}
    manager._inverse_mapping = {}
    manager._unmapped = False
    manager._va_preserved = False
    manager._last_memory_layout_hash = ""

    with pytest.raises(ConnectionError, match="close failed"):
        manager.disconnect()

    assert manager._client is None
    assert manager._granted_lock_type is None


def test_create_mapping_frees_server_allocation_if_export_fails(monkeypatch):
    manager = object.__new__(GMSClientMemoryManager)
    manager.socket_path = "/tmp/gms-test.sock"
    manager.device = 0
    manager.granularity = 4096
    manager._client = object()
    manager._granted_lock_type = GrantedLockType.RW
    manager._mappings = {}
    manager._inverse_mapping = {}
    manager._unmapped = False
    manager._va_preserved = False
    manager._last_memory_layout_hash = ""

    freed_allocations: list[str] = []
    monkeypatch.setattr(manager, "allocate_handle", lambda size, tag: "alloc_1")
    monkeypatch.setattr(
        manager,
        "export_handle",
        lambda allocation_id: (_ for _ in ()).throw(RuntimeError("export failed")),
    )
    monkeypatch.setattr(manager, "free_handle", freed_allocations.append)

    with pytest.raises(RuntimeError, match="export failed"):
        manager.create_mapping(size=4096, tag="kv_cache")

    assert freed_allocations == ["alloc_1"]
