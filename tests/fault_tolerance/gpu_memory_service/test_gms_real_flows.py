# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import itertools
import os
import socket
import threading
import time

import pytest
from gpu_memory_service.client.memory_manager import (
    GMSClientMemoryManager,
    StaleMemoryLayoutError,
)
from gpu_memory_service.client.session import _GMSClientSession
from gpu_memory_service.common.types import (
    GrantedLockType,
    RequestedLockType,
    ServerState,
)
from gpu_memory_service.server.rpc import GMSRPCServer

pytestmark = [
    pytest.mark.unit,
    pytest.mark.fault_tolerance,
]


class _ServerThread:
    def __init__(self, server: GMSRPCServer, socket_path: str) -> None:
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
        deadline = time.time() + 5
        while not os.path.exists(self.socket_path):
            if self._exception is not None:
                raise self._exception
            if time.time() > deadline:
                raise TimeoutError(f"GMS socket did not appear at {self.socket_path}")
            time.sleep(0.01)

    def stop(self) -> None:
        if self._loop is not None:

            def cancel() -> None:
                if self.server._server is not None:
                    self.server._server.close()
                if self._task is not None:
                    self._task.cancel()

            self._loop.call_soon_threadsafe(cancel)
        self._thread.join(timeout=5)
        if self._exception is not None:
            raise self._exception
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)


def _wait_for(predicate, *, timeout: float = 2.0) -> None:
    deadline = time.time() + timeout
    while not predicate():
        if time.time() > deadline:
            raise TimeoutError("condition was not satisfied before timeout")
        time.sleep(0.01)


def _drop_connection(session: _GMSClientSession) -> None:
    sock = session._transport._socket
    assert sock is not None
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    sock.close()
    session._transport._socket = None


@pytest.fixture
def real_gms(monkeypatch, tmp_path):
    server_handles = itertools.count(1000)
    client_handles = itertools.count(10000)
    next_va = itertools.count(0x100000, 0x10000)

    monkeypatch.setattr(
        "gpu_memory_service.server.allocations.cuda_ensure_initialized",
        lambda: None,
    )
    monkeypatch.setattr(
        "gpu_memory_service.server.allocations.cumem_get_allocation_granularity",
        lambda device: 4096,
    )
    monkeypatch.setattr(
        "gpu_memory_service.server.allocations.cumem_create_tolerate_oom",
        lambda size, device: (True, next(server_handles)),
    )
    monkeypatch.setattr(
        "gpu_memory_service.server.allocations.cumem_release",
        lambda handle: None,
    )

    def export_fd(handle: int) -> int:
        read_fd, write_fd = os.pipe()
        os.close(write_fd)
        return read_fd

    monkeypatch.setattr(
        "gpu_memory_service.server.allocations.cumem_export_to_shareable_handle",
        export_fd,
    )

    monkeypatch.setattr(
        "gpu_memory_service.client.memory_manager.cuda_set_current_device",
        lambda device: None,
    )
    monkeypatch.setattr(
        "gpu_memory_service.client.memory_manager.cumem_get_allocation_granularity",
        lambda device: 4096,
    )
    monkeypatch.setattr(
        "gpu_memory_service.client.memory_manager.cuda_synchronize",
        lambda: None,
    )
    monkeypatch.setattr(
        "gpu_memory_service.client.memory_manager.cumem_address_reserve",
        lambda size, granularity: next(next_va),
    )
    monkeypatch.setattr(
        "gpu_memory_service.client.memory_manager.cumem_address_free",
        lambda va, size: None,
    )
    monkeypatch.setattr(
        "gpu_memory_service.client.memory_manager.cumem_map",
        lambda va, size, handle: None,
    )
    monkeypatch.setattr(
        "gpu_memory_service.client.memory_manager.cumem_set_access",
        lambda va, size, device, mode: None,
    )
    monkeypatch.setattr(
        "gpu_memory_service.client.memory_manager.cumem_unmap",
        lambda va, size: None,
    )
    monkeypatch.setattr(
        "gpu_memory_service.client.memory_manager.cumem_release",
        lambda handle: None,
    )
    monkeypatch.setattr(
        "gpu_memory_service.client.memory_manager.cuda_validate_pointer",
        lambda va: True,
    )

    def import_fd(fd: int) -> int:
        os.close(fd)
        return next(client_handles)

    monkeypatch.setattr(
        "gpu_memory_service.client.memory_manager.cumem_import_from_shareable_handle_close_fd",
        import_fd,
    )

    socket_path = str(tmp_path / "gms.sock")
    server = GMSRPCServer(socket_path, device=0, allocation_retry_interval=0.01)
    thread = _ServerThread(server, socket_path)
    thread.start()
    try:
        yield server, socket_path
    finally:
        thread.stop()


def test_rw_commit_publishes_allocations_metadata_and_layout_hash(real_gms):
    server, socket_path = real_gms

    writer = GMSClientMemoryManager(socket_path, device=0)
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
    _wait_for(lambda: server.state == ServerState.COMMITTED)


def test_rw_disconnect_aborts_epoch_and_next_writer_starts_clean(real_gms):
    server, socket_path = real_gms

    writer = _GMSClientSession(socket_path, RequestedLockType.RW, None)
    allocation_id, _ = writer.allocate(4096, "weights")
    writer.metadata_put("stale", allocation_id, 0, b"value")
    _drop_connection(writer)

    _wait_for(lambda: server.state == ServerState.EMPTY)

    next_writer = _GMSClientSession(socket_path, RequestedLockType.RW, None)
    try:
        assert next_writer.list_allocations() == []
        assert next_writer.metadata_list() == []
    finally:
        next_writer.close()


def test_rw_or_ro_grants_rw_from_empty_and_ro_from_committed(real_gms):
    server, socket_path = real_gms

    session = _GMSClientSession(socket_path, RequestedLockType.RW_OR_RO, 100)
    assert session.lock_type == GrantedLockType.RW
    session.commit()

    _wait_for(lambda: server.state == ServerState.COMMITTED)

    session = _GMSClientSession(socket_path, RequestedLockType.RW_OR_RO, 100)
    try:
        assert session.lock_type == GrantedLockType.RO
        assert session.committed
    finally:
        session.close()


def test_committed_epoch_is_replaced_when_new_writer_connects(real_gms):
    server, socket_path = real_gms

    first_writer = GMSClientMemoryManager(socket_path, device=0)
    first_writer.connect(RequestedLockType.RW)
    first_writer.create_mapping(size=4096, tag="weights")
    assert first_writer.commit()

    _wait_for(lambda: server.state == ServerState.COMMITTED)
    assert server._gms.allocation_count == 1

    second_writer = _GMSClientSession(socket_path, RequestedLockType.RW, None)
    try:
        assert second_writer.lock_type == GrantedLockType.RW
        assert second_writer.list_allocations() == []
        assert second_writer.metadata_list() == []
        assert server._gms.allocation_count == 0
        assert server._gms.committed_epoch_id is None
        assert server._gms.active_rw_epoch_id is not None
    finally:
        second_writer.close()


def test_waiting_writer_blocks_new_readers_until_last_reader_disconnects(real_gms):
    server, socket_path = real_gms

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
    _wait_for(lambda: server._gms._sessions.snapshot().waiting_writers == 1)

    with pytest.raises(TimeoutError, match="Timeout waiting for lock"):
        _GMSClientSession(socket_path, RequestedLockType.RO, 100)

    reader.close()
    thread.join(timeout=2)

    waiting_writer = writer_result.get("session")
    assert isinstance(waiting_writer, _GMSClientSession)
    try:
        assert waiting_writer.lock_type == GrantedLockType.RW
    finally:
        waiting_writer.close()


def test_rw_or_ro_times_out_while_writer_waits_behind_reader(real_gms):
    server, socket_path = real_gms

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
    _wait_for(lambda: server._gms._sessions.snapshot().waiting_writers == 1)

    with pytest.raises(TimeoutError, match="Timeout waiting for lock"):
        _GMSClientSession(socket_path, RequestedLockType.RW_OR_RO, 100)

    reader.close()
    thread.join(timeout=2)
    granted_writer = waiting_writer.get("session")
    assert isinstance(granted_writer, _GMSClientSession)
    granted_writer.close()


def test_reader_can_acquire_after_waiting_writer_times_out(real_gms):
    server, socket_path = real_gms

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
    _wait_for(lambda: server._gms._sessions.snapshot().waiting_writers == 1)
    thread.join(timeout=2)

    assert isinstance(writer_result["error"], TimeoutError)
    _wait_for(lambda: server._gms._sessions.snapshot().waiting_writers == 0)

    second_reader = _GMSClientSession(socket_path, RequestedLockType.RO, 200)
    try:
        assert second_reader.lock_type == GrantedLockType.RO
    finally:
        second_reader.close()
        reader.close()


def test_multiple_readers_hold_committed_state_until_last_disconnect(real_gms):
    server, socket_path = real_gms

    writer = _GMSClientSession(socket_path, RequestedLockType.RW, None)
    writer.commit()

    reader_a = _GMSClientSession(socket_path, RequestedLockType.RO, None)
    reader_b = _GMSClientSession(socket_path, RequestedLockType.RO, None)

    _wait_for(lambda: server.state == ServerState.RO)
    assert server._gms._sessions.snapshot().ro_session_count == 2

    reader_a.close()
    _wait_for(lambda: server._gms._sessions.snapshot().ro_session_count == 1)
    assert server.state == ServerState.RO

    reader_b.close()
    _wait_for(lambda: server.state == ServerState.COMMITTED)


def test_ro_session_rejects_rw_only_requests(real_gms):
    _, socket_path = real_gms

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


def test_lock_and_allocation_state_requests_reflect_real_server_state(real_gms):
    _, socket_path = real_gms

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


def test_invalid_metadata_offset_is_rejected_without_mutating_state(real_gms):
    _, socket_path = real_gms

    writer = _GMSClientSession(socket_path, RequestedLockType.RW, None)
    try:
        allocation_id, _ = writer.allocate(4096, "weights")
        with pytest.raises(RuntimeError, match="out of range"):
            writer.metadata_put("tensor.bad", allocation_id, 4096, b"x")
        assert writer.metadata_list() == []
    finally:
        writer.close()


def test_destroy_mapping_frees_allocation_and_metadata(real_gms):
    _, socket_path = real_gms

    writer = GMSClientMemoryManager(socket_path, device=0)
    writer.connect(RequestedLockType.RW)
    va = writer.create_mapping(size=4096, tag="weights")
    allocation_id = writer.mappings[va].allocation_id
    writer.metadata_put("tensor.0", allocation_id, 0, b"payload")

    writer.destroy_mapping(va)

    assert writer.list_handles() == []
    assert writer.metadata_list() == []
    writer.disconnect()


def test_remap_all_vas_succeeds_when_committed_layout_is_unchanged(real_gms):
    _, socket_path = real_gms

    writer = GMSClientMemoryManager(socket_path, device=0)
    writer.connect(RequestedLockType.RW)
    va = writer.create_mapping(size=4096, tag="weights")
    allocation_id = writer.mappings[va].allocation_id
    assert writer.commit()

    reader = GMSClientMemoryManager(socket_path, device=0)
    reader.connect(RequestedLockType.RO)
    imported_va = reader.create_mapping(allocation_id=allocation_id)
    imported_mapping = reader.mappings[imported_va]
    reader.unmap_all_vas()
    reader.disconnect()

    reader.connect(RequestedLockType.RO)
    reader.remap_all_vas()

    assert reader.mappings[imported_va].handle != 0
    assert reader.mappings[imported_va].allocation_id == imported_mapping.allocation_id
    reader.close()


def test_remap_all_vas_rejects_stale_layout_after_new_epoch_commit(real_gms):
    _, socket_path = real_gms

    writer = GMSClientMemoryManager(socket_path, device=0)
    writer.connect(RequestedLockType.RW)
    va = writer.create_mapping(size=4096, tag="weights")
    allocation_id = writer.mappings[va].allocation_id
    assert writer.commit()

    reader = GMSClientMemoryManager(socket_path, device=0)
    reader.connect(RequestedLockType.RO)
    reader.create_mapping(allocation_id=allocation_id)
    reader.unmap_all_vas()
    reader.disconnect()

    next_writer = GMSClientMemoryManager(socket_path, device=0)
    next_writer.connect(RequestedLockType.RW)
    next_writer.create_mapping(size=8192, tag="weights")
    assert next_writer.commit()

    reader.connect(RequestedLockType.RO)
    with pytest.raises(StaleMemoryLayoutError, match="Layout changed"):
        reader.remap_all_vas()
    reader.disconnect()


def test_remap_all_vas_rolls_back_partial_failure(real_gms, monkeypatch):
    _, socket_path = real_gms

    writer = GMSClientMemoryManager(socket_path, device=0)
    writer.connect(RequestedLockType.RW)
    first_va = writer.create_mapping(size=4096, tag="weights")
    second_va = writer.create_mapping(size=4096, tag="weights")
    first_allocation_id = writer.mappings[first_va].allocation_id
    second_allocation_id = writer.mappings[second_va].allocation_id
    assert writer.commit()

    reader = GMSClientMemoryManager(socket_path, device=0)
    reader.connect(RequestedLockType.RO)
    remap_first_va = reader.create_mapping(allocation_id=first_allocation_id)
    remap_second_va = reader.create_mapping(allocation_id=second_allocation_id)
    reader.unmap_all_vas()
    reader.disconnect()
    reader.connect(RequestedLockType.RO)

    map_calls = 0
    released_handles: list[int] = []
    unmapped_vas: list[int] = []

    def fail_on_second_map(va: int, size: int, handle: int) -> None:
        nonlocal map_calls
        map_calls += 1
        if map_calls == 2:
            raise RuntimeError("synthetic remap failure")

    monkeypatch.setattr(
        "gpu_memory_service.client.memory_manager.cumem_map",
        fail_on_second_map,
    )
    monkeypatch.setattr(
        "gpu_memory_service.client.memory_manager.cumem_release",
        released_handles.append,
    )
    monkeypatch.setattr(
        "gpu_memory_service.client.memory_manager.cumem_unmap",
        lambda va, size: unmapped_vas.append(va),
    )

    with pytest.raises(RuntimeError, match="synthetic remap failure"):
        reader.remap_all_vas()

    assert reader.is_unmapped
    assert reader._va_preserved
    assert reader.mappings[remap_first_va].handle == 0
    assert reader.mappings[remap_second_va].handle == 0
    assert released_handles and len(released_handles) == 2
    assert unmapped_vas == [remap_first_va]
    reader.disconnect()


def test_reallocate_all_handles_reuses_preserved_vas_in_new_epoch(real_gms):
    server, socket_path = real_gms

    manager = GMSClientMemoryManager(socket_path, device=0)
    manager.connect(RequestedLockType.RW)
    va = manager.create_mapping(size=4096, tag="weights")
    old_allocation_id = manager.mappings[va].allocation_id
    assert manager.commit()

    _wait_for(lambda: server.state == ServerState.COMMITTED)
    manager.connect(RequestedLockType.RW)
    manager.reallocate_all_handles(tag="weights")

    assert manager.mappings[va].allocation_id != old_allocation_id
    assert manager.mappings[va].handle == 0

    manager.remap_all_vas()

    assert manager.mappings[va].va == va
    assert manager.mappings[va].handle != 0
    manager.close()
    _wait_for(lambda: server.state == ServerState.EMPTY)


def test_disconnect_during_allocation_retry_aborts_writer_and_unblocks_next_writer(
    real_gms,
    monkeypatch,
):
    server, socket_path = real_gms
    oom_attempts = 0
    allow_allocation = False

    def always_oom(size: int, device: int) -> tuple[bool, int]:
        nonlocal oom_attempts
        nonlocal allow_allocation
        if allow_allocation:
            return True, 4242
        oom_attempts += 1
        return False, 0

    monkeypatch.setattr(
        "gpu_memory_service.server.allocations.cumem_create_tolerate_oom",
        always_oom,
    )

    writer = _GMSClientSession(socket_path, RequestedLockType.RW, None)
    result: dict[str, BaseException] = {}

    def allocate() -> None:
        try:
            writer.allocate(4096, "weights")
        except BaseException as exc:
            result["error"] = exc

    thread = threading.Thread(target=allocate)
    thread.start()
    _wait_for(lambda: oom_attempts > 0)

    _drop_connection(writer)

    thread.join(timeout=2)
    _wait_for(lambda: server.state == ServerState.EMPTY)

    allow_allocation = True
    next_writer = _GMSClientSession(socket_path, RequestedLockType.RW, 200)
    try:
        assert next_writer.lock_type == GrantedLockType.RW
        allocation_id, aligned_size = next_writer.allocate(4096, "weights")
        assert allocation_id
        assert aligned_size == 4096
    finally:
        next_writer.close()

    assert isinstance(result.get("error"), ConnectionError)
