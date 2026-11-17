# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GMS runtime flow coverage.

This module exercises lock handoff, committed-layout publication and remap,
reader/writer state transitions, and allocation retry behavior against one
in-process GMS server.
"""

from __future__ import annotations

import asyncio
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import TimeoutError as FutureTimeoutError

import pytest
from _deps import HAS_CUDA, HAS_GMS, HAS_PYNVML, HAS_XPU

if not HAS_GMS:
    pytest.skip(
        "gpu_memory_service package is not available in this test image",
        allow_module_level=True,
    )

if HAS_PYNVML:
    import pynvml

import gpu_memory_service.common.vmm as _vmm_module
from _fake_vmm import FakeVMM
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
from gpu_memory_service.common.vmm import VMMDeviceType
from gpu_memory_service.server.allocations import GMSAllocationManager
from gpu_memory_service.server.fsm import ServerState
from gpu_memory_service.server.rpc import GMSRPCServer

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
    """Query free GPU memory — works on both CUDA (pynvml) and XPU (torch.xpu)."""
    if HAS_XPU:
        import torch

        free_bytes, _total = torch.xpu.mem_get_info(device)
        # Workaround: GPU runtime does not reflect VMM allocations in
        # free-memory queries. Subtract internally tracked VMM usage.
        try:
            from gpu_memory_service.common.vmm import _sycl_vmm

            free_bytes = max(0, free_bytes - _sycl_vmm.total_allocated_bytes())
        except Exception:
            pass
        return int(free_bytes)
    else:
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
    fake_vmm = FakeVMM()

    # Inject fake VMM into the process-global singleton so that
    # GMSAllocationManager and GMSClientMemoryManager both use it.
    monkeypatch.setattr(_vmm_module, "_vmm_instance", fake_vmm)
    monkeypatch.setattr(
        _vmm_module,
        "_vmm_device_type",
        VMMDeviceType.XPU if HAS_XPU else VMMDeviceType.CUDA,
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
def test_prune_allocations_frees_unreferenced_mappings(running_gms):
    _, socket_path = running_gms

    from gpu_memory_service.client.torch.allocator import prune_allocations

    writer = GMSClientMemoryManager(socket_path, device=0)
    try:
        writer.connect(RequestedLockType.RW)
        keep_va = writer.create_mapping(size=4096, tag="weights")
        prune_va = writer.create_mapping(size=8192, tag="weights")

        keep_allocation_id = writer.mappings[keep_va].allocation_id
        prune_allocation_id = writer.mappings[prune_va].allocation_id

        prune_allocations(
            writer,
            referenced_allocation_ids={keep_allocation_id},
            synchronize=False,
        )

        assert keep_va in writer.mappings
        assert prune_va not in writer.mappings
        assert [info.allocation_id for info in writer.list_handles()] == [
            keep_allocation_id
        ]
        with pytest.raises(RuntimeError, match="Unknown allocation"):
            writer.get_handle_info(prune_allocation_id)
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
def test_remap_all_vas_accepts_restored_layout_after_pruned_slot_gap(
    running_gms,
):
    """Pruned layouts may have sparse source slots but compact restore slots."""

    _, socket_path = running_gms

    from gpu_memory_service.client.torch.allocator import prune_allocations

    first_writer = GMSClientMemoryManager(socket_path, device=0)
    reader = GMSClientMemoryManager(socket_path, device=0)
    second_writer = GMSClientMemoryManager(socket_path, device=0)
    try:
        first_writer.connect(RequestedLockType.RW)
        kept_a_va = first_writer.create_mapping(size=4096, tag="weights")
        pruned_va = first_writer.create_mapping(size=8192, tag="weights")
        kept_b_va = first_writer.create_mapping(
            size=first_writer.granularity + 4096,
            tag="weights",
        )

        kept_a = first_writer.mappings[kept_a_va]
        kept_b = first_writer.mappings[kept_b_va]
        assert first_writer.mappings[pruned_va].layout_slot == 1
        assert kept_b.layout_slot == 2

        prune_allocations(
            first_writer,
            referenced_allocation_ids={kept_a.allocation_id, kept_b.allocation_id},
            synchronize=False,
        )
        first_writer.metadata_put("tensor.0", kept_a.allocation_id, 0, b"a")
        first_writer.metadata_put("tensor.1", kept_b.allocation_id, 0, b"b")
        assert first_writer.commit()

        reader.connect(RequestedLockType.RO)
        source_hash = reader.get_memory_layout_hash()
        assert source_hash
        imported_a_va = reader.create_mapping(allocation_id=kept_a.allocation_id)
        imported_b_va = reader.create_mapping(allocation_id=kept_b.allocation_id)
        reader.unmap_all_vas()
        reader.abort()

        second_writer.connect(RequestedLockType.RW)
        restored_a_va = second_writer.create_mapping(size=kept_a.size, tag=kept_a.tag)
        restored_b_va = second_writer.create_mapping(size=kept_b.size, tag=kept_b.tag)
        restored_a = second_writer.mappings[restored_a_va]
        restored_b = second_writer.mappings[restored_b_va]
        assert restored_a.layout_slot == 0
        assert restored_b.layout_slot == 1
        second_writer.metadata_put("tensor.0", restored_a.allocation_id, 0, b"a")
        second_writer.metadata_put("tensor.1", restored_b.allocation_id, 0, b"b")
        assert second_writer.commit()

        verifier = _GMSClientSession(socket_path, RequestedLockType.RO, None)
        try:
            assert verifier.get_memory_layout_hash() == source_hash
        finally:
            verifier.close()

        reader.connect(RequestedLockType.RO)
        reader.remap_all_vas()

        assert reader.mappings[imported_a_va].allocation_id == restored_a.allocation_id
        assert reader.mappings[imported_b_va].allocation_id == restored_b.allocation_id
        assert reader.metadata_get("tensor.0") == (restored_a.allocation_id, 0, b"a")
        assert reader.metadata_get("tensor.1") == (restored_b.allocation_id, 0, b"b")
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


@pytest.mark.timeout(_SOCKET_TEST_TIMEOUT_SECONDS)
def test_scratch_reallocation_keeps_committed_allocation_on_vmm_granularity(
    running_gms,
):
    _, socket_path = running_gms

    scratch_size = 1 << 20
    requested_size = 4097
    manager = GMSClientMemoryManager(socket_path, device=0, scratch_size=scratch_size)
    try:
        va = manager.create_scratch_mapping(size=requested_size, tag="kv_cache")
        manager.unmap_all_vas()
        manager.prepare_scratch_for_reallocation()

        manager.connect(RequestedLockType.RW)
        manager.reallocate_all_handles(tag="kv_cache")
        allocation = manager.get_handle_info(manager.mappings[va].allocation_id)
        assert allocation.size == requested_size
        assert allocation.aligned_size == 8192

        manager.remap_all_vas()
    finally:
        manager.close()


@pytest.mark.asyncio
async def test_allocation_manager_lazily_exports_fresh_fds(monkeypatch):
    export_calls = 0

    class _CountingVMM(FakeVMM):
        def create_tolerate_oom(self, size, device):
            return (True, 4242)

        def export_to_shareable_handle(self, handle):
            nonlocal export_calls
            export_calls += 1
            read_fd, write_fd = os.pipe()
            os.close(write_fd)
            return read_fd

    fake_vmm = _CountingVMM()
    monkeypatch.setattr(_vmm_module, "_vmm_instance", fake_vmm)
    monkeypatch.setattr(
        _vmm_module,
        "_vmm_device_type",
        VMMDeviceType.XPU if HAS_XPU else VMMDeviceType.CUDA,
    )

    allocations = GMSAllocationManager(device=0)
    info = await allocations.allocate(size=4096, tag="weights")

    assert export_calls == 0

    first_fd = allocations.export_allocation(info.allocation_id)
    second_fd = allocations.export_allocation(info.allocation_id)

    try:
        assert export_calls == 2
        assert first_fd != second_fd
        os.fstat(first_fd)
        os.fstat(second_fd)
    finally:
        os.close(first_fd)
        os.close(second_fd)

    assert allocations.free_allocation(info.allocation_id)


@pytest.mark.asyncio
@pytest.mark.timeout(180)
@pytest.mark.skipif(
    not (HAS_CUDA and HAS_PYNVML),
    reason="CUDA+pynvml only. Caveat: Unsupported on XPU."
    "Re-enable when VMM OOM + free-memory tracking ready for XPU.",
)
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


# ---------------------------------------------------------------------------
# Committed layouts (LAYOUT_COMMITTED / RW_DATA)
#
# `commit()` publishes contents: it unmaps the writer, closes the session, and admits
# RO readers. `commit_layout()` publishes only the *shape*: the writer keeps its session
# and mappings and goes on writing, but may no longer reshape what it froze. The pages
# then outlive the session, which is what lets a standby adopt them after a crash.
# ---------------------------------------------------------------------------


@pytest.fixture
def gms_thread(monkeypatch, tmp_path):
    """Start a GMS server over a FakeVMM and hand back (server, socket_path, thread)."""
    fake_vmm = FakeVMM()
    monkeypatch.setattr(_vmm_module, "_vmm_instance", fake_vmm)
    monkeypatch.setattr(_vmm_module, "_vmm_device_type", VMMDeviceType.CUDA)

    socket_path = str(tmp_path / "gms_layout.sock")
    server = GMSRPCServer(socket_path, device=0, allocation_retry_interval=0.01)
    thread = _WhiteBoxServerThread(server, socket_path)
    thread.start()
    try:
        yield server, socket_path, thread
    finally:
        thread.stop()


def _build_sealed_layout(socket_path, size=4096, tag="kv_cache"):
    """Connect, allocate one handle, seal the shape. Returns (manager, allocation_id)."""
    writer = GMSClientMemoryManager(socket_path, device=0)
    writer.connect(RequestedLockType.RW_DATA_OR_RW)
    assert writer.granted_lock_type == GrantedLockType.RW
    va = writer.create_mapping(size=size, tag=tag)
    allocation_id = writer.mappings[va].allocation_id
    commit = writer.commit_layout()
    # The server is the authority on the new grant, and the caller is handed it back
    # rather than having to know that committing narrows them.
    assert commit.granted_lock_type == GrantedLockType.RW_DATA
    assert commit.memory_layout_hash
    assert writer.granted_lock_type == GrantedLockType.RW_DATA
    return writer, allocation_id


@pytest.mark.timeout(_SOCKET_TEST_TIMEOUT_SECONDS)
def test_commit_layout_seals_shape_and_narrows_writer_to_rw_data(gms_thread):
    server, socket_path, _thread = gms_thread
    writer, _ = _build_sealed_layout(socket_path)

    # Sealing does not publish contents, so nothing became readable...
    assert not server._gms.committed
    # ...but the shape is now frozen, and the writer is the one holding the restriction.
    assert server._gms._sessions.layout_committed
    with pytest.raises(Exception):
        writer.allocate_handle(size=4096, tag="kv_cache")


@pytest.mark.timeout(_SOCKET_TEST_TIMEOUT_SECONDS)
@pytest.mark.parametrize(
    "seal, allocations_after_crash, state_after_crash",
    [(False, 0, ServerState.EMPTY), (True, 1, ServerState.LAYOUT_COMMITTED)],
    ids=["unsealed_is_discarded", "sealed_survives"],
)
def test_only_a_committed_layout_survives_the_writer(
    gms_thread, seal, allocations_after_crash, state_after_crash
):
    """Atomicity: a writer that dies part-way through building leaves nothing behind."""
    server, socket_path, thread = gms_thread

    writer = GMSClientMemoryManager(socket_path, device=0)
    writer.connect(RequestedLockType.RW_DATA_OR_RW)
    writer.create_mapping(size=4096, tag="kv_cache")
    if seal:
        writer.commit_layout()
    assert server._gms._allocations.allocation_count == 1

    thread.disconnect_rw_session()  # the writer dies

    assert server._gms._allocations.allocation_count == allocations_after_crash
    assert server._gms.state is state_after_crash


@pytest.mark.timeout(_SOCKET_TEST_TIMEOUT_SECONDS)
def test_standby_adopts_a_committed_layout_and_replays_across_takeovers(gms_thread):
    """The failover path: adopt the same allocation, repeatedly."""
    server, socket_path, thread = gms_thread
    writer, allocation_id = _build_sealed_layout(socket_path)
    thread.disconnect_rw_session()

    for _takeover in range(2):
        standby = GMSClientMemoryManager(socket_path, device=0)
        standby.connect(RequestedLockType.RW_DATA_OR_RW)
        # Granted RW_DATA because a committed layout was there to adopt.
        assert standby.granted_lock_type == GrantedLockType.RW_DATA
        assert [h.allocation_id for h in standby.list_handles(tag="kv_cache")] == [
            allocation_id
        ]
        # Adopting does not re-seal anything, so the next standby inherits it too.
        thread.disconnect_rw_session()
        assert server._gms.state is ServerState.LAYOUT_COMMITTED


@pytest.mark.timeout(_SOCKET_TEST_TIMEOUT_SECONDS)
def test_requesting_rw_replaces_a_committed_layout(gms_thread):
    """Adopt-vs-replace is chosen by the requested mode, not by server policy."""
    server, socket_path, thread = gms_thread
    _writer, _ = _build_sealed_layout(socket_path)
    thread.disconnect_rw_session()
    assert server._gms._allocations.allocation_count == 1

    replacer = GMSClientMemoryManager(socket_path, device=0)
    replacer.connect(RequestedLockType.RW)  # "wipe it, I'll build my own"

    assert replacer.granted_lock_type == GrantedLockType.RW
    assert server._gms._allocations.allocation_count == 0
    assert server._gms.state is ServerState.RW


@pytest.mark.timeout(_SOCKET_TEST_TIMEOUT_SECONDS)
def test_readers_are_refused_while_contents_are_unspecified(gms_thread):
    """A reader must never attach to a live, mutating pool.

    This needs no new code: can_acquire_ro already requires a content commit, which
    LAYOUT_COMMITTED does not have. The reader waits exactly as it would for a loader that has
    not committed yet, and times out.
    """
    server, socket_path, thread = gms_thread
    _writer, _ = _build_sealed_layout(socket_path)
    thread.disconnect_rw_session()

    reader = GMSClientMemoryManager(socket_path, device=0)
    with pytest.raises(TimeoutError):
        reader.connect(RequestedLockType.RO, timeout_ms=200)
    assert server._gms.state is ServerState.LAYOUT_COMMITTED


@pytest.mark.timeout(_SOCKET_TEST_TIMEOUT_SECONDS)
def test_content_commit_path_is_unchanged(gms_thread):
    """Regression guard: the weights lifecycle must not notice any of this."""
    server, socket_path, _thread = gms_thread

    writer = GMSClientMemoryManager(socket_path, device=0)
    writer.connect(RequestedLockType.RW)
    va = writer.create_mapping(size=4096, tag="weights")
    writer.metadata_put("tensor.0", writer.mappings[va].allocation_id, 0, b"weights")
    assert writer.commit()

    assert server._gms.state is ServerState.COMMITTED
    assert server._gms.is_ready()
    reader = _GMSClientSession(socket_path, RequestedLockType.RO, None)
    try:
        assert reader.lock_type == GrantedLockType.RO
        assert len(reader.list_allocations()) == 1
    finally:
        reader.close()


# ---- bulk import: lifetime, partial failure ----


def _publish(socket_path, sizes, tag="weights"):
    """Commit one layout with allocations of ``sizes``; return allocation ids by slot."""
    writer = GMSClientMemoryManager(socket_path, device=0)
    writer.connect(RequestedLockType.RW)
    ids = [
        writer.mappings[writer.create_mapping(size=s, tag=tag)].allocation_id
        for s in sizes
    ]
    assert writer.commit()
    writer.close()
    return ids


def _fake_vmm():
    return _vmm_module._vmm_instance


@pytest.mark.timeout(_SOCKET_TEST_TIMEOUT_SECONDS)
@pytest.mark.parametrize("fail_at", ["export", "import", "map"])
def test_bulk_import_unwinds_a_partial_failure(running_gms, monkeypatch, fail_at):
    """A failure in the middle of import_allocations leaves nothing behind: no
    reservations, no imported handles, no mappings, and no fd closed twice."""
    _, socket_path = running_gms
    ids = _publish(socket_path, (4096, 4096, 4096))
    vmm = _fake_vmm()
    reader = GMSClientMemoryManager(socket_path, device=0, tag="weights")
    reader.connect(RequestedLockType.RO)
    reservations_before = dict(vmm.reservations)
    decoy = []
    calls = {"n": 0}

    def fail_second(real):
        def wrapped(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 2:
                if fail_at == "import":
                    # Contract: the fd is consumed even on failure. Another
                    # thread may get the same fd number right away.
                    os.close(args[0])
                    decoy.extend(os.pipe())
                raise RuntimeError(f"injected {fail_at} failure")
            return real(*args, **kwargs)

        return wrapped

    if fail_at == "export":
        monkeypatch.setattr(reader, "export_handle", fail_second(reader.export_handle))
    elif fail_at == "import":
        monkeypatch.setattr(
            vmm,
            "import_shareable_handle_close_fd",
            fail_second(vmm.import_shareable_handle_close_fd),
        )
    else:
        monkeypatch.setattr(vmm, "map", fail_second(vmm.map))
    try:
        with pytest.raises(RuntimeError, match="injected"):
            reader.import_allocations(ids)
        assert reader.mappings == {}
        assert vmm.mapped == {}
        assert vmm.imports == set()
        assert vmm.reservations == reservations_before
        for fd in decoy:
            os.fstat(fd)  # still open: not closed a second time by the unwind
        monkeypatch.undo()
        reader.import_allocations(ids)
        assert len(reader.mappings) == 3
    finally:
        for fd in decoy:
            os.close(fd)
        reader.close()


@pytest.mark.timeout(_SOCKET_TEST_TIMEOUT_SECONDS)
def test_bulk_remap_failure_leaves_the_manager_remappable(running_gms, monkeypatch):
    _, socket_path = running_gms
    ids = _publish(socket_path, (4096, 4096, 4096))
    vmm = _fake_vmm()
    reader = GMSClientMemoryManager(socket_path, device=0)
    try:
        reader.connect(RequestedLockType.RO)
        reader.import_allocations(ids)
        reader.unmap_all_vas()
        reader.abort()
        reader.connect(RequestedLockType.RO)
        real_map, calls = vmm.map, {"n": 0}

        def flaky_map(*args):
            calls["n"] += 1
            if calls["n"] == 3:
                raise RuntimeError("injected map failure")
            return real_map(*args)

        monkeypatch.setattr(vmm, "map", flaky_map)
        with pytest.raises(RuntimeError, match="injected"):
            reader.remap_all_vas()
        assert vmm.mapped == {} and vmm.imports == set()
        assert all(m.handle == 0 for m in reader.mappings.values())
        monkeypatch.setattr(vmm, "map", real_map)
        reader.remap_all_vas()
        assert all(m.handle != 0 for m in reader.mappings.values())
    finally:
        reader.close()


@pytest.mark.timeout(_SOCKET_TEST_TIMEOUT_SECONDS)
def test_bulk_import_default_mode_sets_access_per_reservation(running_gms, monkeypatch):
    """Without fixed VA every import is its own cuMemAddressReserve; access is set
    per mapping (as the per-allocation path does) even when the driver happens to
    hand out adjacent reservations."""
    _, socket_path = running_gms
    ids = _publish(socket_path, (4096, 4096, 4096))
    vmm = _fake_vmm()
    cursor = {"va": 0x7000000000}

    def adjacent_reserve(size, granularity, addr=0):
        va = cursor["va"]
        cursor["va"] += size
        vmm.reservations[va] = size
        return va

    monkeypatch.setattr(vmm, "address_reserve", adjacent_reserve)
    reader = GMSClientMemoryManager(socket_path, device=0)
    try:
        reader.connect(RequestedLockType.RO)
        vmm.calls.clear()
        reader.import_allocations(ids)
        access = [c for c in vmm.calls if c[0] == "set_access"]
        assert sorted(access) == [
            ("set_access", va, 4096) for va in sorted(reader.mappings)
        ]
    finally:
        reader.close()


@pytest.mark.timeout(_SOCKET_TEST_TIMEOUT_SECONDS)
def test_bulk_import_bounds_the_number_of_open_fds(running_gms, monkeypatch):
    """Exports are imported in chunks so a large layout cannot hit RLIMIT_NOFILE."""
    _, socket_path = running_gms
    ids = _publish(socket_path, [4096] * 7)
    vmm = _fake_vmm()
    monkeypatch.setattr(GMSClientMemoryManager, "_BULK_CHUNK", 3, raising=False)
    reader = GMSClientMemoryManager(socket_path, device=0)
    outstanding = {"now": 0, "max": 0}
    real_export, real_import = (
        reader.export_handle,
        vmm.import_shareable_handle_close_fd,
    )

    def export(allocation_id):
        outstanding["now"] += 1
        outstanding["max"] = max(outstanding["max"], outstanding["now"])
        return real_export(allocation_id)

    def import_(fd, import_size=0):
        outstanding["now"] -= 1
        return real_import(fd, import_size=import_size)

    monkeypatch.setattr(reader, "export_handle", export)
    monkeypatch.setattr(vmm, "import_shareable_handle_close_fd", import_)
    try:
        reader.connect(RequestedLockType.RO)
        reader.import_allocations(ids)
        assert len(reader.mappings) == 7
        assert outstanding["max"] == 3
    finally:
        reader.close()


@pytest.mark.timeout(_SOCKET_TEST_TIMEOUT_SECONDS)
def test_bulk_import_option_and_per_manager_import_stats(running_gms, monkeypatch):
    """The constructor argument wins over DYN_GMS_BULK_IMPORT, which is only a
    fallback, and import counters belong to the manager that imported."""
    _, socket_path = running_gms
    ids = _publish(socket_path, (4096, 4096))
    monkeypatch.delenv("DYN_GMS_BULK_IMPORT", raising=False)
    assert GMSClientMemoryManager(socket_path, device=0).bulk_import
    monkeypatch.setenv("DYN_GMS_BULK_IMPORT", "0")
    assert not GMSClientMemoryManager(socket_path, device=0).bulk_import
    bulk = GMSClientMemoryManager(socket_path, device=0, bulk_import=True)
    single = GMSClientMemoryManager(socket_path, device=0, bulk_import=False)
    assert bulk.bulk_import and not single.bulk_import
    try:
        bulk.connect(RequestedLockType.RO)
        single.connect(RequestedLockType.RO)
        bulk.import_allocations(ids)
        single.create_mapping(allocation_id=ids[0])
        assert bulk.take_import_stats()["imports"] == 2
        assert bulk.take_import_stats() == {}
        assert single.take_import_stats()["imports"] == 1
    finally:
        bulk.close()
        single.close()


# ---- default read-only attach (materialize_module_from_gms) with bulk import ----


def _publish_module_layout(socket_path):
    """Publish four allocations: two parameters at different offsets of one, a
    parameter and its alias on another, a buffer and a tensor attribute on a
    third, and one allocation no tensor references. Returns allocation ids."""
    import torch
    from gpu_memory_service.client.torch.tensor import TensorMetadata

    writer = GMSClientMemoryManager(socket_path, device=0)
    writer.connect(RequestedLockType.RW)
    ids = [
        writer.mappings[writer.create_mapping(size=s, tag="weights")].allocation_id
        for s in (2 * 4096, 4096, 4096, 4096)
    ]

    def put(key, allocation, offset, shape, tensor_type="parameter"):
        stride = (shape[1], 1) if len(shape) == 2 else (1,)
        meta = TensorMetadata(tuple(shape), torch.float32, stride, tensor_type)
        assert writer.metadata_put(key, allocation, offset, meta.to_bytes())

    put("a.weight", ids[0], 0, (4, 4))
    put("a.bias", ids[0], 4096, (4,))
    put("b.weight", ids[1], 0, (4, 4))
    put("b.tied", ids[1], 0, (4, 4))
    put("b.scale", ids[2], 0, (4,), "buffer")
    put("b.extra", ids[2], 64, (4,), "tensor_attr")
    assert writer.commit()
    writer.close()
    return ids


def _meta_module():
    import torch

    model = torch.nn.Module()
    with torch.device("meta"):
        model.a = torch.nn.Linear(4, 4)
        model.b = torch.nn.Linear(4, 4, bias=False)
        model.b.tied = torch.nn.Parameter(torch.empty(4, 4))
        model.b.register_buffer("scale", torch.empty(4))
    return model


def _attach(socket_path, monkeypatch, **manager_kwargs):
    """Run the default RO attach with GPU tensors faked by the pointer they
    alias. Returns (reader, model, per-tensor pointers, server RPC counts)."""
    import torch
    from gpu_memory_service.client.torch import tensor as gms_tensor
    from gpu_memory_service.client.torch.module import materialize_module_from_gms

    pointers = {}

    def fake_tensor(ptr, shape, stride, dtype, device_index):
        t = torch.zeros(shape, dtype=dtype)
        pointers[id(t)] = ptr
        return t

    monkeypatch.setattr(gms_tensor, "_tensor_from_pointer", fake_tensor)
    reader = GMSClientMemoryManager(
        socket_path, device=0, tag="weights", **manager_kwargs
    )
    reader.connect(RequestedLockType.RO)
    rpcs = {"get_handle_info": 0, "export_handle": 0}
    for name in rpcs:
        real = getattr(reader, name)

        def counted(*args, _real=real, _name=name):
            rpcs[_name] += 1
            return _real(*args)

        monkeypatch.setattr(reader, name, counted)
    model = _meta_module()
    by_name = {}
    real_materialize = gms_tensor.GMSTensorSpec.materialize

    def materialize(spec, manager, device_index):
        t = real_materialize(spec, manager, device_index)
        by_name[spec.name] = pointers[id(t)]
        return t

    monkeypatch.setattr(gms_tensor.GMSTensorSpec, "materialize", materialize)
    try:
        materialize_module_from_gms(reader, model, device_index=0)
    finally:
        monkeypatch.setattr(gms_tensor.GMSTensorSpec, "materialize", real_materialize)
    return reader, model, by_name, rpcs


def _attach_view(reader, by_name):
    """Mappings and tensor placement independent of where the driver put VAs."""
    base = {va: m.allocation_id for va, m in reader.mappings.items()}

    def where(ptr):
        for va, m in reader.mappings.items():
            if va <= ptr < va + m.aligned_size:
                return base[va], ptr - va
        raise AssertionError(f"0x{ptr:x} is not in any mapping")

    mappings = sorted(
        (m.allocation_id, m.size, m.aligned_size, m.tag, m.layout_slot)
        for m in reader.mappings.values()
    )
    return mappings, {name: where(ptr) for name, ptr in by_name.items()}


@pytest.mark.timeout(_SOCKET_TEST_TIMEOUT_SECONDS)
def test_default_ro_attach_maps_the_same_with_bulk_import_on_and_off(
    running_gms, monkeypatch
):
    """materialize_module_from_gms imports every referenced allocation in one
    bulk call when the manager's bulk_import is on (no per-tensor handle-info
    round trips) and gives the same mappings, tensor placement, aliases and
    read-only access as one create_mapping per allocation."""
    _, socket_path = running_gms
    ids = _publish_module_layout(socket_path)
    vmm = _fake_vmm()
    views = {}
    for bulk in (False, True):
        vmm.calls.clear()
        reader, model, by_name, rpcs = _attach(
            socket_path, monkeypatch, bulk_import=bulk
        )
        try:
            assert reader.bulk_import is bulk
            # Unreferenced allocations stay unmapped on both paths.
            assert {m.allocation_id for m in reader.mappings.values()} == set(ids[:3])
            assert rpcs == {"get_handle_info": 0 if bulk else 3, "export_handle": 3}
            assert set(vmm.access.values()) == {GrantedLockType.RO}
            assert set(vmm.access) == set(reader.mappings)
            assert by_name["b.tied"] == by_name["b.weight"]
            assert not any(p.is_meta for p in model.parameters())
            assert not any(b.is_meta for b in model.buffers())
            views[bulk] = _attach_view(reader, by_name)
        finally:
            reader.close()
    assert views[True] == views[False]
    mappings, placement = views[True]
    assert placement["a.bias"] == (ids[0], 4096)
    assert placement["b.extra"] == (ids[2], 64)


@pytest.mark.timeout(_SOCKET_TEST_TIMEOUT_SECONDS)
def test_default_ro_attach_follows_the_bulk_import_env_fallback(
    running_gms, monkeypatch
):
    _, socket_path = running_gms
    _publish_module_layout(socket_path)
    monkeypatch.setenv("DYN_GMS_BULK_IMPORT", "0")
    reader, _, _, rpcs = _attach(socket_path, monkeypatch)
    reader.close()
    assert rpcs["get_handle_info"] == 3  # per-allocation path
    monkeypatch.delenv("DYN_GMS_BULK_IMPORT")
    reader, _, _, rpcs = _attach(socket_path, monkeypatch)
    reader.close()
    assert rpcs["get_handle_info"] == 0  # bulk by default


@pytest.mark.timeout(_SOCKET_TEST_TIMEOUT_SECONDS)
@pytest.mark.parametrize("fail_at", ["export", "import", "map", "set_access"])
def test_default_ro_attach_bulk_failure_is_all_or_nothing(
    running_gms, monkeypatch, fail_at
):
    """A failure partway through the bulk attach leaves no reservation, handle
    or mapping behind and the model untouched; the same manager can attach
    again afterwards."""
    import torch
    from gpu_memory_service.client.torch import tensor as gms_tensor
    from gpu_memory_service.client.torch.module import materialize_module_from_gms

    _, socket_path = running_gms
    _publish_module_layout(socket_path)
    vmm = _fake_vmm()
    reservations_before = dict(vmm.reservations)
    reader = GMSClientMemoryManager(
        socket_path, device=0, tag="weights", bulk_import=True
    )
    reader.connect(RequestedLockType.RO)
    window = dict(vmm.reservations)
    fail_on = 1 if fail_at == "set_access" else 2
    calls = {"n": 0}

    def failing(real):
        def wrapped(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == fail_on:
                if fail_at == "import":
                    os.close(args[0])  # the fd is consumed even on failure
                raise RuntimeError(f"injected {fail_at} failure")
            return real(*args, **kwargs)

        return wrapped

    owner = reader if fail_at == "export" else vmm
    name = {
        "export": "export_handle",
        "import": "import_shareable_handle_close_fd",
    }.get(fail_at, fail_at)
    real = getattr(owner, name)
    monkeypatch.setattr(owner, name, failing(real))
    materialized = []
    real_materialize = gms_tensor.GMSTensorSpec.materialize

    def materialize(*args):
        materialized.append(args[0].name)
        return real_materialize(*args)

    monkeypatch.setattr(gms_tensor.GMSTensorSpec, "materialize", materialize)
    monkeypatch.setattr(
        gms_tensor,
        "_tensor_from_pointer",
        lambda ptr, shape, stride, dtype, device_index: torch.zeros(shape, dtype=dtype),
    )
    model = _meta_module()
    try:
        with pytest.raises(RuntimeError, match="injected"):
            materialize_module_from_gms(reader, model, device_index=0)
        assert materialized == []  # failed before any tensor was bound
        assert reader.mappings == {}
        assert vmm.mapped == {} and vmm.imports == set()
        assert vmm.reservations == window
        assert all(p.is_meta for p in model.parameters())
        assert all(b.is_meta for b in model.buffers())

        monkeypatch.setattr(owner, name, real)
        materialize_module_from_gms(reader, model, device_index=0)
        assert len(reader.mappings) == 3
        assert set(vmm.access) == set(reader.mappings)
        assert not any(p.is_meta for p in model.parameters())
        assert not any(b.is_meta for b in model.buffers())
    finally:
        reader.close()
    assert vmm.reservations == reservations_before
