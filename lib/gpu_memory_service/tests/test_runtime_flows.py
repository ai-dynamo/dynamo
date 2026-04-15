# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GMS runtime flow coverage.

This module exercises lock handoff, committed-layout publication and remap,
reader/writer state transitions, and allocation retry behavior against one
in-process GMS server.
"""

from __future__ import annotations

import asyncio
import itertools
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import TimeoutError as FutureTimeoutError

import pynvml
import pytest

try:
    from gpu_memory_service.client import memory_manager as client_memory_manager
    from gpu_memory_service.client.memory_manager import (
        GMSClientMemoryManager,
        StaleMemoryLayoutError,
    )
    from gpu_memory_service.client.rpc import _GMSRPCTransport
    from gpu_memory_service.client.session import _GMSClientSession
    from gpu_memory_service.common.locks import GrantedLockType, RequestedLockType
    from gpu_memory_service.common.protocol.messages import (
        GetEventHistoryRequest,
        GetEventHistoryResponse,
        GetRuntimeStateRequest,
        GetRuntimeStateResponse,
    )
    from gpu_memory_service.server import allocations as server_allocations
    from gpu_memory_service.server.allocations import GMSAllocationManager
    from gpu_memory_service.server.fsm import ServerState
    from gpu_memory_service.server.rpc import GMSRPCServer
except ModuleNotFoundError:
    pytest.skip(
        "gpu_memory_service package is not available in this test image",
        allow_module_level=True,
    )

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.integration,
    pytest.mark.none,
    pytest.mark.gpu_1,
]

_SOCKET_TEST_TIMEOUT_SECONDS = 60
_DEFAULT_WAIT_TIMEOUT_SECONDS = 2.0
_SERVER_START_TIMEOUT_SECONDS = 5.0
_SERVER_STOP_TIMEOUT_SECONDS = 5.0
_RW_DISCONNECT_TIMEOUT_SECONDS = 5.0
_BLOCKED_WRITER_JOIN_TIMEOUT_SECONDS = 2.0
_ALLOCATION_RETRY_INTERVAL_SECONDS = 0.1
_ALLOCATION_RETRY_TIMEOUT_SECONDS = 120.0
_EXPORT_HOLDER_READY_TIMEOUT_SECONDS = 30.0
_ALLOCATION_BLOCK_ASSERTION_SECONDS = 5.0
_GPU_MEMORY_RECOVERY_TIMEOUT_SECONDS = 30.0
_FAST_POLL_INTERVAL_SECONDS = 0.01
_SLOW_POLL_INTERVAL_SECONDS = 0.1


def _gpu_memory_free_bytes(device: int = 0) -> int:
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        return int(pynvml.nvmlDeviceGetMemoryInfo(handle).free)
    finally:
        pynvml.nvmlShutdown()


def _drop_connection(session: _GMSClientSession) -> None:
    # Use a raw transport break here, not abort(), because these tests need to
    # simulate an unexpected socket loss while a request is still in flight.
    sock = session._transport._socket
    assert sock is not None
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    sock.close()
    session._transport._socket = None


def _wait_for_server_state(
    server: GMSRPCServer,
    expected: ServerState,
    timeout: float = _DEFAULT_WAIT_TIMEOUT_SECONDS,
) -> None:
    deadline = time.monotonic() + timeout
    while server.state != expected:
        if time.monotonic() > deadline:
            raise TimeoutError(f"server did not reach {expected.name}")
        time.sleep(_FAST_POLL_INTERVAL_SECONDS)


def _wait_for_waiting_writers(
    server: GMSRPCServer,
    expected: int,
    timeout: float = _DEFAULT_WAIT_TIMEOUT_SECONDS,
) -> None:
    deadline = time.monotonic() + timeout
    while server._gms._sessions.snapshot().waiting_writers != expected:
        if time.monotonic() > deadline:
            raise TimeoutError(f"waiting writers did not reach {expected}")
        time.sleep(_FAST_POLL_INTERVAL_SECONDS)


def _wait_for_ro_session_count(
    server: GMSRPCServer,
    expected: int,
    timeout: float = _DEFAULT_WAIT_TIMEOUT_SECONDS,
) -> None:
    deadline = time.monotonic() + timeout
    while server._gms._sessions.snapshot().ro_session_count != expected:
        if time.monotonic() > deadline:
            raise TimeoutError(f"RO session count did not reach {expected}")
        time.sleep(_FAST_POLL_INTERVAL_SECONDS)


class _WhiteBoxServerThread:
    """Threaded in-process server helper."""

    def __init__(self, server, socket_path: str):
        self.server = server
        self.socket_path = socket_path
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task[None] | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._exception: BaseException | None = None

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        self._task = loop.create_task(self.server.serve())
        try:
            loop.run_until_complete(self._task)
        except asyncio.CancelledError:
            pass
        except BaseException as exc:
            self._exception = exc
        finally:
            pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            loop.close()

    def start(self) -> None:
        self._thread.start()
        deadline = time.monotonic() + _SERVER_START_TIMEOUT_SECONDS
        last_probe_error: Exception | None = None
        while True:
            if self._exception is not None:
                raise self._exception
            if self.server._server is not None and os.path.exists(self.socket_path):
                try:
                    self.server._gms.get_runtime_state()
                    return
                except Exception as exc:
                    last_probe_error = exc
            if time.monotonic() > deadline:
                timeout_error = TimeoutError(
                    f"GMS socket did not appear at {self.socket_path}"
                )
                if last_probe_error is not None:
                    raise timeout_error from last_probe_error
                raise timeout_error
            time.sleep(_FAST_POLL_INTERVAL_SECONDS)

    def stop(self) -> None:
        if self._loop is not None:

            def cancel() -> None:
                if self.server._server is not None:
                    self.server._server.close()
                if self._task is not None:
                    self._task.cancel()

            self._loop.call_soon_threadsafe(cancel)
        self._thread.join(timeout=_SERVER_STOP_TIMEOUT_SECONDS)
        if self._thread.is_alive():
            raise RuntimeError(
                f"GMS server thread failed to stop for {self.socket_path}"
            )
        if self._exception is not None:
            raise self._exception
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

    def disconnect_rw_session(
        self,
        timeout: float = _RW_DISCONNECT_TIMEOUT_SECONDS,
    ) -> None:
        if self._loop is None:
            raise RuntimeError("GMS server thread is not running")
        future = asyncio.run_coroutine_threadsafe(
            self._disconnect_rw_session(), self._loop
        )
        try:
            future.result(timeout=timeout)
        except FutureTimeoutError as exc:
            raise TimeoutError("Timed out disconnecting RW session") from exc

    async def _disconnect_rw_session(self) -> None:
        conn = self.server._gms._sessions._locking.rw_conn
        if conn is None:
            raise RuntimeError("No active RW session to disconnect")
        await self.server._gms.cleanup_connection(conn)


@pytest.fixture
def running_gms(monkeypatch, tmp_path):
    server_handles = itertools.count(1000)
    client_handles = itertools.count(10000)
    next_va = itertools.count(0x100000, 0x10000)

    monkeypatch.setattr(server_allocations, "cuda_ensure_initialized", lambda: None)
    monkeypatch.setattr(
        server_allocations,
        "cumem_get_allocation_granularity",
        lambda device: 4096,
    )
    monkeypatch.setattr(
        server_allocations,
        "cumem_create_tolerate_oom",
        lambda size, device: (True, next(server_handles)),
    )
    monkeypatch.setattr(server_allocations, "cumem_release", lambda handle: None)

    def export_fd(handle: int) -> int:
        read_fd, write_fd = os.pipe()
        os.close(write_fd)
        return read_fd

    monkeypatch.setattr(
        server_allocations, "cumem_export_to_shareable_handle", export_fd
    )

    monkeypatch.setattr(
        client_memory_manager,
        "cumem_get_allocation_granularity",
        lambda device: 4096,
    )
    monkeypatch.setattr(client_memory_manager, "cuda_synchronize", lambda: None)
    monkeypatch.setattr(
        client_memory_manager,
        "cumem_address_reserve",
        lambda size, granularity: next(next_va),
    )
    monkeypatch.setattr(
        client_memory_manager,
        "cumem_address_free",
        lambda va, size: None,
    )
    monkeypatch.setattr(
        client_memory_manager, "cumem_map", lambda va, size, handle: None
    )
    monkeypatch.setattr(
        client_memory_manager,
        "cumem_set_access",
        lambda va, size, device, mode: None,
    )
    monkeypatch.setattr(client_memory_manager, "cumem_unmap", lambda va, size: None)
    monkeypatch.setattr(client_memory_manager, "cumem_release", lambda handle: None)
    monkeypatch.setattr(client_memory_manager, "cuda_validate_pointer", lambda va: True)

    def import_fd(fd: int) -> int:
        os.close(fd)
        return next(client_handles)

    monkeypatch.setattr(
        client_memory_manager,
        "cumem_import_from_shareable_handle_close_fd",
        import_fd,
    )

    socket_path = str(tmp_path / "gms.sock")
    server = GMSRPCServer(socket_path, device=0, allocation_retry_interval=0.01)
    thread = _WhiteBoxServerThread(server, socket_path)
    thread.start()
    try:
        yield server, socket_path
    finally:
        thread.stop()


@pytest.mark.timeout(_SOCKET_TEST_TIMEOUT_SECONDS)
def test_rw_commit_publishes_allocations_metadata_and_layout_hash(running_gms):
    server, socket_path = running_gms

    writer = GMSClientMemoryManager(socket_path, device=0)
    try:
        writer.connect(RequestedLockType.RW)
        va = writer.create_mapping(size=4096, tag="weights")
        allocation_id = writer.mappings[va].allocation_id
        writer.metadata_put("tensor.0", allocation_id, 0, b"weights")
        assert writer.commit()

        reader = _GMSClientSession(socket_path, RequestedLockType.RO, None)
        try:
            assert reader.lock_type == GrantedLockType.RO
            assert reader.committed
            assert len(reader.list_allocations()) == 1
            assert reader.metadata_get("tensor.0") == (allocation_id, 0, b"weights")
            assert reader.get_memory_layout_hash()
        finally:
            reader.close()

        assert writer.is_unmapped
        assert not writer.is_connected
        _wait_for_server_state(server, ServerState.COMMITTED)
    finally:
        writer.close()


@pytest.mark.timeout(_SOCKET_TEST_TIMEOUT_SECONDS)
def test_rw_disconnect_aborts_layout_and_next_writer_starts_clean(running_gms):
    server, socket_path = running_gms

    writer = _GMSClientSession(socket_path, RequestedLockType.RW, None)
    allocation_id, _ = writer.allocate(4096, "weights")
    writer.metadata_put("stale", allocation_id, 0, b"value")
    _drop_connection(writer)

    _wait_for_server_state(server, ServerState.EMPTY)

    next_writer = _GMSClientSession(socket_path, RequestedLockType.RW, None)
    try:
        assert next_writer.list_allocations() == []
        assert next_writer.metadata_list() == []
    finally:
        next_writer.close()


@pytest.mark.timeout(_SOCKET_TEST_TIMEOUT_SECONDS)
def test_rw_or_ro_grants_rw_from_empty_and_ro_from_committed(running_gms):
    server, socket_path = running_gms

    session = _GMSClientSession(socket_path, RequestedLockType.RW_OR_RO, 100)
    assert session.lock_type == GrantedLockType.RW
    session.commit()

    _wait_for_server_state(server, ServerState.COMMITTED)

    session = _GMSClientSession(socket_path, RequestedLockType.RW_OR_RO, 100)
    try:
        assert session.lock_type == GrantedLockType.RO
        assert session.committed
    finally:
        session.close()


@pytest.mark.timeout(_SOCKET_TEST_TIMEOUT_SECONDS)
def test_runtime_state_and_event_history_are_side_effect_free(running_gms):
    server, socket_path = running_gms

    writer = GMSClientMemoryManager(socket_path, device=0)
    try:
        writer.connect(RequestedLockType.RW)
        writer.create_mapping(size=4096, tag="weights")
        assert writer.commit()

        assert server._gms._sessions.snapshot().ro_session_count == 0

        with _GMSRPCTransport(socket_path) as transport:
            transport.connect()
            state = transport.request(
                GetRuntimeStateRequest(),
                GetRuntimeStateResponse,
            )

        with _GMSRPCTransport(socket_path) as transport:
            transport.connect()
            history = transport.request(
                GetEventHistoryRequest(),
                GetEventHistoryResponse,
            )

        assert state.state == ServerState.COMMITTED.name
        assert state.committed
        assert state.is_ready
        assert state.ro_session_count == 0
        assert state.waiting_writers == 0
        assert state.allocation_count == 1
        assert state.memory_layout_hash
        assert [event.kind for event in history.events] == ["rw_connected", "committed"]
        assert server._gms._sessions.snapshot().ro_session_count == 0
    finally:
        writer.close()


@pytest.mark.timeout(_SOCKET_TEST_TIMEOUT_SECONDS)
def test_committed_layout_is_replaced_when_new_writer_connects(running_gms):
    server, socket_path = running_gms

    first_writer = GMSClientMemoryManager(socket_path, device=0)
    try:
        first_writer.connect(RequestedLockType.RW)
        first_writer.create_mapping(size=4096, tag="weights")
        assert first_writer.commit()

        _wait_for_server_state(server, ServerState.COMMITTED)
        assert server._gms.allocation_count == 1

        second_writer = _GMSClientSession(socket_path, RequestedLockType.RW, None)
        try:
            assert second_writer.lock_type == GrantedLockType.RW
            assert second_writer.list_allocations() == []
            assert second_writer.metadata_list() == []
            assert server._gms.allocation_count == 0
            assert server.state == ServerState.RW
            assert not server._gms.committed
        finally:
            second_writer.close()
    finally:
        first_writer.close()


@pytest.mark.timeout(_SOCKET_TEST_TIMEOUT_SECONDS)
def test_reader_mapping_disconnect_then_next_writer_clears_old_layout(
    running_gms,
):
    server, socket_path = running_gms

    writer = GMSClientMemoryManager(socket_path, device=0)
    reader = GMSClientMemoryManager(socket_path, device=0)
    try:
        writer.connect(RequestedLockType.RW)
        va = writer.create_mapping(size=4096, tag="weights")
        allocation_id = writer.mappings[va].allocation_id
        assert writer.commit()

        reader.connect(RequestedLockType.RO)
        imported_va = reader.create_mapping(allocation_id=allocation_id)
        assert reader.mappings[imported_va].handle != 0

        next_writer_result: dict[str, object] = {}

        def open_writer() -> None:
            try:
                next_writer_result["session"] = _GMSClientSession(
                    socket_path,
                    RequestedLockType.RW,
                    500,
                )
            except Exception as exc:
                next_writer_result["error"] = exc

        thread = threading.Thread(target=open_writer)
        thread.start()
        _wait_for_waiting_writers(server, 1)

        assert thread.is_alive()
        assert server.state == ServerState.RO
        assert server._gms.allocation_count == 1

        reader.unmap_all_vas()
        reader.abort()
        thread.join(timeout=_BLOCKED_WRITER_JOIN_TIMEOUT_SECONDS)

        next_writer = next_writer_result.get("session")
        assert isinstance(next_writer, _GMSClientSession)
        try:
            assert next_writer.lock_type == GrantedLockType.RW
            assert next_writer.list_allocations() == []
            assert server._gms.allocation_count == 0
            assert server.state == ServerState.RW
            assert not server._gms.committed
        finally:
            next_writer.close()
    finally:
        reader.close()
        writer.close()


@pytest.mark.timeout(_SOCKET_TEST_TIMEOUT_SECONDS)
def test_waiting_writer_blocks_new_readers_until_last_reader_disconnects(
    running_gms,
):
    server, socket_path = running_gms

    writer = _GMSClientSession(socket_path, RequestedLockType.RW, None)
    writer.commit()

    reader = _GMSClientSession(socket_path, RequestedLockType.RO, None)
    writer_result: dict[str, object] = {}

    def open_writer() -> None:
        try:
            writer_result["session"] = _GMSClientSession(
                socket_path,
                RequestedLockType.RW,
                500,
            )
        except Exception as exc:
            writer_result["error"] = exc

    thread = threading.Thread(target=open_writer)
    thread.start()
    _wait_for_waiting_writers(server, 1)

    with pytest.raises(TimeoutError, match="Timeout waiting for lock"):
        _GMSClientSession(socket_path, RequestedLockType.RO, 100)

    reader.close()
    thread.join(timeout=_BLOCKED_WRITER_JOIN_TIMEOUT_SECONDS)

    waiting_writer = writer_result.get("session")
    assert isinstance(waiting_writer, _GMSClientSession)
    try:
        assert waiting_writer.lock_type == GrantedLockType.RW
    finally:
        waiting_writer.close()


@pytest.mark.timeout(_SOCKET_TEST_TIMEOUT_SECONDS)
def test_rw_or_ro_times_out_while_writer_waits_behind_reader(running_gms):
    server, socket_path = running_gms

    writer = _GMSClientSession(socket_path, RequestedLockType.RW, None)
    writer.commit()

    reader = _GMSClientSession(socket_path, RequestedLockType.RO, None)
    waiting_writer: dict[str, object] = {}

    def block_writer() -> None:
        try:
            waiting_writer["session"] = _GMSClientSession(
                socket_path,
                RequestedLockType.RW,
                500,
            )
        except Exception as exc:
            waiting_writer["error"] = exc

    thread = threading.Thread(target=block_writer)
    thread.start()
    _wait_for_waiting_writers(server, 1)

    with pytest.raises(TimeoutError, match="Timeout waiting for lock"):
        _GMSClientSession(socket_path, RequestedLockType.RW_OR_RO, 100)

    reader.close()
    thread.join(timeout=_BLOCKED_WRITER_JOIN_TIMEOUT_SECONDS)

    granted_writer = waiting_writer.get("session")
    assert isinstance(granted_writer, _GMSClientSession)
    granted_writer.close()


@pytest.mark.timeout(_SOCKET_TEST_TIMEOUT_SECONDS)
def test_reader_can_acquire_after_waiting_writer_times_out(running_gms):
    server, socket_path = running_gms

    writer = _GMSClientSession(socket_path, RequestedLockType.RW, None)
    writer.commit()

    reader = _GMSClientSession(socket_path, RequestedLockType.RO, None)
    writer_result: dict[str, BaseException | None] = {"error": None}

    def timeout_writer() -> None:
        try:
            _GMSClientSession(socket_path, RequestedLockType.RW, 100)
        except BaseException as exc:
            writer_result["error"] = exc

    thread = threading.Thread(target=timeout_writer)
    thread.start()
    _wait_for_waiting_writers(server, 1)
    thread.join(timeout=_BLOCKED_WRITER_JOIN_TIMEOUT_SECONDS)

    assert isinstance(writer_result["error"], TimeoutError)
    _wait_for_waiting_writers(server, 0)

    second_reader = _GMSClientSession(socket_path, RequestedLockType.RO, 200)
    try:
        assert second_reader.lock_type == GrantedLockType.RO
    finally:
        second_reader.close()
        reader.close()


@pytest.mark.timeout(_SOCKET_TEST_TIMEOUT_SECONDS)
def test_multiple_readers_hold_committed_state_until_last_disconnect(
    running_gms,
):
    server, socket_path = running_gms

    writer = _GMSClientSession(socket_path, RequestedLockType.RW, None)
    writer.commit()

    reader_a = _GMSClientSession(socket_path, RequestedLockType.RO, None)
    reader_b = _GMSClientSession(socket_path, RequestedLockType.RO, None)

    _wait_for_server_state(server, ServerState.RO)
    assert server._gms._sessions.snapshot().ro_session_count == 2

    reader_a.close()
    _wait_for_ro_session_count(server, 1)
    assert server.state == ServerState.RO

    reader_b.close()
    _wait_for_server_state(server, ServerState.COMMITTED)


@pytest.mark.timeout(_SOCKET_TEST_TIMEOUT_SECONDS)
def test_ro_session_rejects_rw_only_requests(running_gms):
    _, socket_path = running_gms

    writer = _GMSClientSession(socket_path, RequestedLockType.RW, None)
    writer.commit()

    reader = _GMSClientSession(socket_path, RequestedLockType.RO, None)
    try:
        with pytest.raises(RuntimeError, match="not allowed for RO session"):
            reader.allocate(4096, "weights")
        with pytest.raises(RuntimeError, match="not allowed for RO session"):
            reader.commit()
    finally:
        reader.close()


@pytest.mark.timeout(_SOCKET_TEST_TIMEOUT_SECONDS)
def test_lock_and_allocation_state_requests_reflect_real_server_state(
    running_gms,
):
    _, socket_path = running_gms

    writer = _GMSClientSession(socket_path, RequestedLockType.RW, None)
    allocation_id, _ = writer.allocate(4096, "weights")

    lock_state = writer.get_lock_state()
    allocation_state = writer.get_allocation_state()

    assert lock_state.state == ServerState.RW.name
    assert lock_state.has_rw_session
    assert lock_state.ro_session_count == 0
    assert allocation_state.allocation_count == 1

    writer.metadata_put("tensor.0", allocation_id, 0, b"x")
    writer.commit()

    reader = _GMSClientSession(socket_path, RequestedLockType.RO, None)
    try:
        lock_state = reader.get_lock_state()
        allocation_state = reader.get_allocation_state()
        assert lock_state.state == ServerState.RO.name
        assert not lock_state.has_rw_session
        assert lock_state.ro_session_count == 1
        assert allocation_state.allocation_count == 1
    finally:
        reader.close()


@pytest.mark.timeout(_SOCKET_TEST_TIMEOUT_SECONDS)
def test_invalid_metadata_offset_is_rejected_without_mutating_state(
    running_gms,
):
    _, socket_path = running_gms

    writer = _GMSClientSession(socket_path, RequestedLockType.RW, None)
    try:
        allocation_id, aligned_size = writer.allocate(4096, "weights")
        with pytest.raises(RuntimeError, match="out of range"):
            writer.metadata_put("tensor.bad", allocation_id, aligned_size, b"x")
        assert writer.metadata_list() == []
    finally:
        writer.close()


@pytest.mark.timeout(_SOCKET_TEST_TIMEOUT_SECONDS)
def test_destroy_mapping_frees_allocation_and_metadata(running_gms):
    _, socket_path = running_gms

    writer = GMSClientMemoryManager(socket_path, device=0)
    try:
        writer.connect(RequestedLockType.RW)
        va = writer.create_mapping(size=4096, tag="weights")
        allocation_id = writer.mappings[va].allocation_id
        writer.metadata_put("tensor.0", allocation_id, 0, b"payload")

        writer.destroy_mapping(va)

        assert writer.list_handles() == []
        assert writer.metadata_list() == []
    finally:
        writer.close()


@pytest.mark.timeout(_SOCKET_TEST_TIMEOUT_SECONDS)
def test_remap_all_vas_succeeds_when_committed_layout_is_unchanged(
    running_gms,
):
    _, socket_path = running_gms

    writer = GMSClientMemoryManager(socket_path, device=0)
    reader = GMSClientMemoryManager(socket_path, device=0)
    try:
        writer.connect(RequestedLockType.RW)
        va = writer.create_mapping(size=4096, tag="weights")
        allocation_id = writer.mappings[va].allocation_id
        assert writer.commit()

        reader.connect(RequestedLockType.RO)
        imported_va = reader.create_mapping(allocation_id=allocation_id)
        imported_mapping = reader.mappings[imported_va]
        reader.unmap_all_vas()
        reader.abort()

        reader.connect(RequestedLockType.RO)
        reader.remap_all_vas()

        assert reader.mappings[imported_va].handle != 0
        assert (
            reader.mappings[imported_va].allocation_id == imported_mapping.allocation_id
        )
    finally:
        reader.close()
        writer.close()


@pytest.mark.timeout(_SOCKET_TEST_TIMEOUT_SECONDS)
def test_remap_all_vas_rejects_stale_layout_after_new_layout_commit(
    running_gms,
):
    _, socket_path = running_gms

    writer = GMSClientMemoryManager(socket_path, device=0)
    reader = GMSClientMemoryManager(socket_path, device=0)
    next_writer = GMSClientMemoryManager(socket_path, device=0)
    try:
        writer.connect(RequestedLockType.RW)
        va = writer.create_mapping(size=4096, tag="weights")
        allocation_id = writer.mappings[va].allocation_id
        assert writer.commit()

        reader.connect(RequestedLockType.RO)
        reader.create_mapping(allocation_id=allocation_id)
        reader.unmap_all_vas()
        reader.abort()

        next_writer.connect(RequestedLockType.RW)
        next_writer.create_mapping(size=writer.granularity + 4096, tag="weights")
        assert next_writer.commit()

        reader.connect(RequestedLockType.RO)
        with pytest.raises(StaleMemoryLayoutError, match="Layout changed"):
            reader.remap_all_vas()
        reader.abort()
    finally:
        next_writer.close()
        reader.close()
        writer.close()


@pytest.mark.timeout(_SOCKET_TEST_TIMEOUT_SECONDS)
def test_remap_all_vas_accepts_new_layout_with_same_structural_layout(
    running_gms,
):
    _, socket_path = running_gms

    first_writer = GMSClientMemoryManager(socket_path, device=0)
    reader = GMSClientMemoryManager(socket_path, device=0)
    second_writer = GMSClientMemoryManager(socket_path, device=0)
    try:
        first_writer.connect(RequestedLockType.RW)
        va = first_writer.create_mapping(size=4096, tag="weights")
        first_allocation_id = first_writer.mappings[va].allocation_id
        first_writer.metadata_put("tensor.0", first_allocation_id, 0, b"shape")
        assert first_writer.commit()

        reader.connect(RequestedLockType.RO)
        imported_va = reader.create_mapping(allocation_id=first_allocation_id)
        reader.unmap_all_vas()
        reader.abort()

        second_writer.connect(RequestedLockType.RW)
        second_va = second_writer.create_mapping(size=4096, tag="weights")
        second_allocation_id = second_writer.mappings[second_va].allocation_id
        assert second_allocation_id != first_allocation_id
        second_writer.metadata_put("tensor.0", second_allocation_id, 0, b"shape")
        assert second_writer.commit()

        reader.connect(RequestedLockType.RO)
        reader.remap_all_vas()

        assert reader.mappings[imported_va].va == imported_va
        assert reader.mappings[imported_va].allocation_id == second_allocation_id
        assert reader.metadata_get("tensor.0") == (second_allocation_id, 0, b"shape")
    finally:
        second_writer.close()
        reader.close()
        first_writer.close()


@pytest.mark.timeout(_SOCKET_TEST_TIMEOUT_SECONDS)
def test_reallocate_all_handles_reuses_preserved_vas_in_new_layout(
    running_gms,
):
    server, socket_path = running_gms

    manager = GMSClientMemoryManager(socket_path, device=0)
    manager.connect(RequestedLockType.RW)
    va = manager.create_mapping(size=4096, tag="weights")
    old_allocation_id = manager.mappings[va].allocation_id
    assert manager.commit()

    _wait_for_server_state(server, ServerState.COMMITTED)
    manager.connect(RequestedLockType.RW)
    manager.reallocate_all_handles(tag="weights")

    assert manager.mappings[va].allocation_id != old_allocation_id
    assert manager.mappings[va].handle == 0

    manager.remap_all_vas()

    assert manager.mappings[va].va == va
    assert manager.mappings[va].handle != 0
    manager.close()
    _wait_for_server_state(server, ServerState.EMPTY)


@pytest.mark.asyncio
async def test_allocation_manager_caches_exported_fd(monkeypatch):
    export_calls = 0

    monkeypatch.setattr(server_allocations, "cuda_ensure_initialized", lambda: None)
    monkeypatch.setattr(
        server_allocations,
        "cumem_get_allocation_granularity",
        lambda device: 4096,
    )
    monkeypatch.setattr(
        server_allocations,
        "cumem_create_tolerate_oom",
        lambda size, device: (True, 4242),
    )
    monkeypatch.setattr(server_allocations, "cumem_release", lambda handle: None)

    def export_fd(handle: int) -> int:
        nonlocal export_calls
        export_calls += 1
        read_fd, write_fd = os.pipe()
        os.close(write_fd)
        return read_fd

    monkeypatch.setattr(
        server_allocations, "cumem_export_to_shareable_handle", export_fd
    )

    allocations = GMSAllocationManager(device=0)
    info = await allocations.allocate(size=4096, tag="weights")

    first_fd = allocations.export_allocation(info.allocation_id)
    second_fd = allocations.export_allocation(info.allocation_id)

    try:
        assert export_calls == 1
        assert info.export_fd >= 0
        assert first_fd != info.export_fd
        assert second_fd != info.export_fd
        assert first_fd != second_fd
        os.fstat(first_fd)
        os.fstat(second_fd)
    finally:
        os.close(first_fd)
        os.close(second_fd)

    assert allocations.free_allocation(info.allocation_id)


@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_large_allocation_unblocks_after_export_fd_holder_dies(
    tmp_path,
):
    """Allocation retries should unblock after the last exported FD holder dies."""

    _HOLD_EXPORT_FD_PROGRAM = (
        "import sys\n"
        "import time\n"
        "from pathlib import Path\n"
        "\n"
        "fd = int(sys.argv[1])\n"
        "ready_path = Path(sys.argv[2])\n"
        "ready_path.write_text(str(fd), encoding='utf-8')\n"
        "while True:\n"
        "    time.sleep(1.0)\n"
    )

    free_before = _gpu_memory_free_bytes()
    size = int(free_before * 0.9)
    assert size > 0

    allocations = GMSAllocationManager(
        device=0,
        allocation_retry_interval=_ALLOCATION_RETRY_INTERVAL_SECONDS,
        allocation_retry_timeout=_ALLOCATION_RETRY_TIMEOUT_SECONDS,
    )
    holder = None
    allocation_task = None

    try:
        first = await allocations.allocate(
            size=size,
            tag="weights",
            is_connected=lambda: True,
        )
        assert first.layout_slot == 0

        free_after_first = _gpu_memory_free_bytes()
        assert free_after_first < free_before - (size // 2)

        exported_fd = allocations.export_allocation(first.allocation_id)
        holder_ready = tmp_path / "holder.ready"
        holder_log = tmp_path / "holder.log"
        # Hold the exported FD in another process so `clear_all()` drops the
        # server's bookkeeping first, then the allocator retry path waits for
        # the last external reference to die before reusing GPU memory.
        with holder_log.open("w", encoding="utf-8") as log_file:
            holder = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    _HOLD_EXPORT_FD_PROGRAM,
                    str(exported_fd),
                    str(holder_ready),
                ],
                pass_fds=[exported_fd],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        os.close(exported_fd)

        deadline = time.monotonic() + _EXPORT_HOLDER_READY_TIMEOUT_SECONDS
        while not holder_ready.exists():
            assert holder.poll() is None, holder_log.read_text(encoding="utf-8")
            assert time.monotonic() < deadline, holder_log.read_text(encoding="utf-8")
            await asyncio.sleep(_SLOW_POLL_INTERVAL_SECONDS)

        allocations.clear_all()
        assert allocations.allocation_count == 0

        allocation_task = asyncio.create_task(
            allocations.allocate(
                size=size,
                tag="weights",
                is_connected=lambda: True,
            )
        )

        deadline = time.monotonic() + _ALLOCATION_BLOCK_ASSERTION_SECONDS
        while time.monotonic() < deadline:
            assert holder.poll() is None, holder_log.read_text(encoding="utf-8")
            assert not allocation_task.done()
            await asyncio.sleep(_SLOW_POLL_INTERVAL_SECONDS)

        assert not allocation_task.done()

        os.killpg(os.getpgid(holder.pid), signal.SIGKILL)
        holder.wait(timeout=_EXPORT_HOLDER_READY_TIMEOUT_SECONDS)

        second = await asyncio.wait_for(
            allocation_task,
            timeout=_ALLOCATION_RETRY_TIMEOUT_SECONDS,
        )
        assert second.layout_slot == 0
        assert allocations.allocation_count == 1

        allocations.clear_all()
        assert allocations.allocation_count == 0

        deadline = time.monotonic() + _GPU_MEMORY_RECOVERY_TIMEOUT_SECONDS
        while _gpu_memory_free_bytes() < free_before - (1 << 30):
            assert time.monotonic() < deadline
            await asyncio.sleep(_SLOW_POLL_INTERVAL_SECONDS)
    finally:
        if allocation_task is not None and not allocation_task.done():
            allocation_task.cancel()
            try:
                await allocation_task
            except asyncio.CancelledError:
                pass
        if allocations.allocation_count > 0:
            allocations.clear_all()
        if holder is not None and holder.poll() is None:
            os.killpg(os.getpgid(holder.pid), signal.SIGKILL)
            holder.wait(timeout=_EXPORT_HOLDER_READY_TIMEOUT_SECONDS)
