# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Targeted GMS fault-tolerance unit tests."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import os
import socket
import subprocess
import sys
import time

import pytest
from cuda.bindings import driver as cuda
from gpu_memory_service.common import cuda_utils
from gpu_memory_service.common.protocol.messages import (
    CommitRequest,
    CommitResponse,
    GetEventHistoryRequest,
    GetLockStateRequest,
    GetLockStateResponse,
    GetRuntimeStateRequest,
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
from gpu_memory_service.server.rpc import GMSRPCServer, _is_connection_alive
from gpu_memory_service.server.session import (
    Connection,
    GMSSessionManager,
    OperationNotAllowed,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
]


def test_cumem_create_tolerate_oom_returns_handle_on_success(monkeypatch):
    monkeypatch.setattr(
        cuda_utils.cuda,
        "cuMemCreate",
        lambda size, prop, flags: (cuda.CUresult.CUDA_SUCCESS, 1234),
    )

    allocated, handle = cuda_utils.cumem_create_tolerate_oom(4096, 0)

    assert allocated
    assert handle == 1234


def test_cumem_create_tolerate_oom_returns_false_on_oom(monkeypatch):
    monkeypatch.setattr(
        cuda_utils.cuda,
        "cuMemCreate",
        lambda size, prop, flags: (cuda.CUresult.CUDA_ERROR_OUT_OF_MEMORY, 0),
    )

    allocated, handle = cuda_utils.cumem_create_tolerate_oom(4096, 0)

    assert not allocated
    assert handle == 0


def test_cumem_export_to_shareable_handle_returns_fd(monkeypatch):
    monkeypatch.setattr(
        cuda_utils.cuda,
        "cuMemExportToShareableHandle",
        lambda handle, handle_type, flags: (cuda.CUresult.CUDA_SUCCESS, 77),
    )

    fd = cuda_utils.cumem_export_to_shareable_handle(1234)

    assert fd == 77


def test_non_oom_cuda_error_exits_process() -> None:
    script = """
from cuda.bindings import driver as cuda
from gpu_memory_service.common import cuda_utils

cuda_utils.cuda_check_result(cuda.CUresult.CUDA_ERROR_INVALID_VALUE, "synthetic")
"""
    result = subprocess.run([sys.executable, "-c", script], check=False)
    assert result.returncode == 1


class _DummyReader:
    def at_eof(self) -> bool:
        return False

    def exception(self):
        return None


class _DummyWriter:
    def __init__(self, sock: socket.socket | None = None) -> None:
        self.closed = False
        self._socket = sock

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        return None

    def is_closing(self) -> bool:
        return self.closed

    def get_extra_info(self, _name: str):
        return self._socket


def test_is_connection_alive_detects_dead_peer() -> None:
    local_sock, peer_sock = socket.socketpair()
    local_sock.setblocking(False)
    try:
        conn = Connection(
            reader=_DummyReader(),
            writer=_DummyWriter(local_sock),
            mode=GrantedLockType.RW,
            session_id="session_1",
            recv_buffer=bytearray(),
        )
        assert _is_connection_alive(conn)

        peer_sock.close()
        deadline = time.monotonic() + 1.0
        while _is_connection_alive(conn):
            if time.monotonic() > deadline:
                raise TimeoutError("peer disconnect was not detected")
            time.sleep(0.01)
    finally:
        peer_sock.close()
        local_sock.close()


def _make_connection(
    mode: GrantedLockType,
    session_id: str,
) -> tuple[Connection, _DummyWriter]:
    writer = _DummyWriter()
    return (
        Connection(
            reader=_DummyReader(),
            writer=writer,
            mode=mode,
            session_id=session_id,
            recv_buffer=bytearray(),
        ),
        writer,
    )


def _make_allocation_manager() -> GMSAllocationManager:
    manager = object.__new__(GMSAllocationManager)
    manager._device = 0
    manager._allocations = {}
    manager._granularity = 1
    manager._allocation_retry_interval = 0.0
    manager._allocation_retry_timeout = None
    return manager


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
    conn, _ = _make_connection(GrantedLockType.RW, "session_1")
    gms._events = []

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

    conn, writer = _make_connection(GrantedLockType.RO, "session_2")
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

    conn, writer = _make_connection(GrantedLockType.RW, "session_3")
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
async def test_runtime_state_handshake_send_failure_does_not_fail_server(monkeypatch):
    server = _make_server()
    reader = _DummyReader()
    writer = _DummyWriter()

    async def fake_recv_message(_reader, _buffer):
        return GetRuntimeStateRequest(), -1, bytearray()

    async def fake_send_message(_writer, _msg, _fd=-1):
        raise BrokenPipeError("runtime-state send failed")

    monkeypatch.setattr("gpu_memory_service.server.rpc.recv_message", fake_recv_message)
    monkeypatch.setattr("gpu_memory_service.server.rpc.send_message", fake_send_message)

    conn = await server._do_handshake(reader, writer, "session_diag")

    assert conn is None
    assert server._gms._sessions.state == ServerState.EMPTY
    assert writer.closed


@pytest.mark.asyncio
async def test_runtime_state_request_is_rejected_on_live_session():
    gms = GMS()
    conn, _ = _make_connection(GrantedLockType.RW, "session_4")

    gms._sessions._reserved_rw_session_id = conn.session_id
    gms.on_connect(conn)

    with pytest.raises(OperationNotAllowed):
        await gms.handle_request(
            conn,
            GetRuntimeStateRequest(),
            lambda: True,
        )


@pytest.mark.asyncio
async def test_event_history_request_is_rejected_on_live_session():
    gms = GMS()
    conn, _ = _make_connection(GrantedLockType.RW, "session_5")

    gms._sessions._reserved_rw_session_id = conn.session_id
    gms.on_connect(conn)

    with pytest.raises(OperationNotAllowed):
        await gms.handle_request(
            conn,
            GetEventHistoryRequest(),
            lambda: True,
        )


@pytest.mark.asyncio
async def test_server_refuses_to_bind_over_live_socket(monkeypatch, tmp_path):
    socket_path = str(tmp_path / "gms.sock")
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(socket_path)
    listener.listen(1)

    monkeypatch.setattr(
        "gpu_memory_service.server.allocations.cuda_ensure_initialized",
        lambda: None,
    )
    monkeypatch.setattr(
        "gpu_memory_service.server.allocations.cumem_get_allocation_granularity",
        lambda device: 4096,
    )

    server = GMSRPCServer(socket_path, device=0)
    try:
        with pytest.raises(RuntimeError, match="already running"):
            await asyncio.wait_for(server.serve(), timeout=0.1)
    finally:
        listener.close()
        if os.path.exists(socket_path):
            os.unlink(socket_path)


@pytest.mark.asyncio
async def test_allocate_rejects_non_positive_size_before_cuda():
    manager = _make_allocation_manager()

    with pytest.raises(ValueError, match="size must be > 0"):
        await manager.allocate(0, epoch_id=1, tag="weights", is_connected=None)


@pytest.mark.asyncio
async def test_allocate_aborts_retry_when_writer_disconnects(monkeypatch):
    manager = _make_allocation_manager()

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
