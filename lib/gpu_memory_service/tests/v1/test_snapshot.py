# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import socket
import threading
from contextlib import contextmanager
from pathlib import Path

import msgspec
import pytest
from _fake_vmm import FakeVMM
from gpu_memory_service.common.locks import GrantedLockType, RequestedLockType
from gpu_memory_service.core.client.memory_manager import (
    LocalMapping,
    install_mapping,
    release_mapping,
    reserve_and_install_mapping,
    unmap_mapping,
)
from gpu_memory_service.core.client.session import _GMSClientSession
from gpu_memory_service.core.protocol import (
    AllocationRecord,
    CommitRequest,
    SuccessResponse,
)
from gpu_memory_service.core.server import rpc as server_rpc
from gpu_memory_service.core.server.gms import GMSServerMemoryManager
from gpu_memory_service.core.server.rpc import GMSRPCServer
from gpu_memory_service.v1 import snapshot

pytestmark = [pytest.mark.pre_merge, pytest.mark.integration, pytest.mark.gpu_0]


@contextmanager
def _server(
    path: str,
    vmm: FakeVMM,
    manager: GMSServerMemoryManager | None = None,
):
    server = GMSRPCServer(
        path,
        manager or GMSServerMemoryManager("GPU-0", vmm, 0),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)
        assert not thread.is_alive()


class _Transfer:
    def __init__(self, sources, events, failure=None):
        self._sources = sources
        self._events = events
        self._failure = failure

    def restore(self, targets):
        assert set(targets) == {source.allocation_id for source in self._sources}
        assert {
            allocation_id: target.byte_count
            for allocation_id, target in targets.items()
        } == {source.allocation_id: source.byte_count for source in self._sources}
        if self._failure:
            raise RuntimeError("restore failed")

    def close(self):
        self._events.append("transfer_close")
        if self._failure:
            raise RuntimeError("transfer close failed")


class _Backend:
    def __init__(self, events, failure=None):
        self._events = events
        self._failure = failure
        self.closed = False

    def start_restore(self, sources):
        return _Transfer(sources, self._events, self._failure)

    def close(self):
        self.closed = True
        if self._failure:
            raise RuntimeError("backend close failed")


class _Writer:
    def __init__(self, path, *, device):
        self._path = Path(path)
        self._device = device
        self._data = bytearray()

    def __enter__(self):
        return self

    def write_device(self, src_ptr, byte_count):
        assert self._device == 0
        assert src_ptr > 0
        self._data.extend(bytes(byte_count))

    def __exit__(self, *_args):
        self._path.write_bytes(self._data)


@pytest.mark.timeout(10)
def test_exact_ids_roundtrip_through_fresh_server_hydration(
    tmp_path,
    monkeypatch,
) -> None:
    source_path = str(tmp_path / "source.sock")
    source_vmm = FakeVMM(granularity=64)
    checkpoint_dir = tmp_path / "checkpoint"
    artifact_dir = str(checkpoint_dir / "device-0")
    monkeypatch.setattr(snapshot, "DeviceToFileWriter", _Writer)

    with _server(source_path, source_vmm):
        writer = _GMSClientSession(source_path, RequestedLockType.RW)
        records = (
            AllocationRecord("weight-0", 64),
            AllocationRecord("weight-1", 128),
        )
        engine_mappings = []
        for record in records:
            writer.allocate(record.allocation_id, record.aligned_size)
            engine_mappings.append(
                _map_record(writer, record, source_vmm, GrantedLockType.RW)
            )
        saved_mappings = tuple(
            (mapping.allocation_id, mapping.aligned_size, mapping.base)
            for mapping, _handle in engine_mappings
        )
        writer.commit()
        monkeypatch.setattr(snapshot, "get_vmm", lambda: source_vmm)
        manifest = snapshot.save_weights(
            artifact_dir,
            source_path,
            0,
            shard_size_bytes=64,
        )
        for mapping, handle in reversed(engine_mappings):
            unmap_mapping(source_vmm, mapping, handle)
        writer.close()

    assert [
        (allocation.allocation_id, allocation.aligned_size)
        for allocation in manifest.allocations
    ] == [(allocation_id, size) for allocation_id, size, _va in saved_mappings]

    target_vmm = FakeVMM(granularity=64)
    events = []
    backend = _Backend(events)
    monkeypatch.setattr(snapshot, "get_vmm", lambda: target_vmm)
    monkeypatch.setattr(
        snapshot,
        "create_transfer_backend",
        lambda *_args, **_kwargs: backend,
    )

    with _server(source_path, target_vmm):
        assert snapshot.hydrate_weights(artifact_dir, source_path, 0) is None
        assert not target_vmm.imports
        assert not target_vmm.reservations

        restored = _GMSClientSession(source_path, RequestedLockType.RO)
        assert [
            (record.allocation_id, record.aligned_size)
            for record in restored.list_allocations()
        ] == [
            (allocation_id, aligned_size)
            for allocation_id, aligned_size, _va in saved_mappings
        ]
        restored_handles = []
        for mapping, _old_handle in engine_mappings:
            restored_handles.append(
                install_mapping(
                    source_vmm,
                    mapping,
                    restored.export(mapping.allocation_id),
                    0,
                    GrantedLockType.RO,
                )
            )
        assert source_vmm.access == {
            mapping.base: GrantedLockType.RO for mapping, _handle in engine_mappings
        }
        for (mapping, _old_handle), handle in reversed(
            tuple(zip(engine_mappings, restored_handles))
        ):
            unmap_mapping(source_vmm, mapping, handle)
            release_mapping(source_vmm, mapping)
        restored.close()

    assert backend.closed


class _FailingCleanupVMM(FakeVMM):
    def __init__(self, events):
        super().__init__(granularity=64)
        self._events = events

    def unmap(self, va, size):
        self._events.append("unmap")
        super().unmap(va, size)
        raise RuntimeError("unmap cleanup failed")

    def release(self, handle):
        imported = handle in self.imports
        super().release(handle)
        if imported:
            raise RuntimeError("handle cleanup failed")

    def address_free(self, va, size):
        super().address_free(va, size)
        raise RuntimeError("VA cleanup failed")


def _write_manifest(artifact_dir: Path) -> None:
    artifact_dir.mkdir()
    manifest = snapshot.SnapshotManifest(
        1,
        (
            snapshot.SnapshotAllocation("weight-0", 64, "shard.bin", 0),
            snapshot.SnapshotAllocation("weight-1", 64, "shard.bin", 64),
        ),
    )
    (artifact_dir / "manifest.json").write_bytes(msgspec.json.encode(manifest))
    (artifact_dir / "shard.bin").write_bytes(bytes(128))


def _map_record(
    session: _GMSClientSession,
    record: AllocationRecord,
    vmm: FakeVMM,
    access: GrantedLockType,
) -> tuple[LocalMapping, int]:
    return reserve_and_install_mapping(
        vmm,
        session.export(record.allocation_id),
        record.allocation_id,
        record.aligned_size,
        record.aligned_size,
        record.aligned_size,
        64,
        0,
        access,
    )


@pytest.mark.timeout(10)
def test_save_cleanup_failure_blocks_manifest(
    tmp_path,
    monkeypatch,
) -> None:
    socket_path = str(tmp_path / "weights.sock")
    events = []
    vmm = _FailingCleanupVMM(events)
    artifact_dir = tmp_path / "artifact"
    monkeypatch.setattr(snapshot, "get_vmm", lambda: vmm)
    monkeypatch.setattr(snapshot, "DeviceToFileWriter", _Writer)

    with _server(socket_path, vmm):
        writer = _GMSClientSession(socket_path, RequestedLockType.RW)
        writer.allocate("weight-0", 64)
        writer.allocate("weight-1", 64)
        writer.commit()
        writer.close()

        with pytest.raises(RuntimeError, match="unmap cleanup failed"):
            snapshot.save_weights(str(artifact_dir), socket_path, 0)

        assert not (artifact_dir / "manifest.json").exists()
        assert not vmm.imports
        assert not vmm.mapped
        assert not vmm.reservations

        fresh_writer = _GMSClientSession(socket_path, RequestedLockType.RW)
        fresh_writer.close()
        assert not vmm.server_handles


@pytest.mark.timeout(10)
def test_hydrate_preserves_transfer_error_and_attempts_all_cleanup(
    tmp_path,
    monkeypatch,
    caplog,
) -> None:
    socket_path = str(tmp_path / "weights.sock")
    artifact_dir = tmp_path / "artifact"
    _write_manifest(artifact_dir)
    events = []
    vmm = _FailingCleanupVMM(events)
    caplog.set_level("ERROR")
    monkeypatch.setattr(snapshot, "get_vmm", lambda: vmm)
    monkeypatch.setattr(
        snapshot,
        "create_transfer_backend",
        lambda *_args, **_kwargs: _Backend(events, failure=True),
    )

    with _server(socket_path, vmm):
        with pytest.raises(RuntimeError, match="restore failed"):
            snapshot.hydrate_weights(str(artifact_dir), socket_path, 0)

        assert not vmm.imports
        assert not vmm.mapped
        assert not vmm.reservations
        assert not vmm.server_handles

        fresh_writer = _GMSClientSession(socket_path, RequestedLockType.RW)
        fresh_writer.close()

    assert events.index("transfer_close") < events.index("unmap")
    assert events.count("unmap") == 2
    assert "resource cleanup failed" in caplog.text


class _CommitResponseLossManager(GMSServerMemoryManager):
    def __init__(self, vmm):
        super().__init__("GPU-0", vmm, 0)
        self.drop_commit_response = threading.Event()
        self.commit_published = threading.Event()

    def handle_request(self, session, request):
        response = super().handle_request(session, request)
        if isinstance(request, CommitRequest):
            self.commit_published.set()
            self.drop_commit_response.set()
        return response


@pytest.mark.timeout(10)
def test_hydrate_verifies_publication_after_commit_response_loss(
    tmp_path,
    monkeypatch,
    caplog,
) -> None:
    socket_path = str(tmp_path / "weights.sock")
    artifact_dir = tmp_path / "artifact"
    _write_manifest(artifact_dir)
    vmm = FakeVMM(granularity=64)
    manager = _CommitResponseLossManager(vmm)
    events = []
    backend = _Backend(events)
    original_send_message = server_rpc.send_message

    def drop_commit_response(sock, message, fd=-1):
        if (
            isinstance(message, SuccessResponse)
            and manager.drop_commit_response.is_set()
        ):
            manager.drop_commit_response.clear()
            sock.shutdown(socket.SHUT_RDWR)
            return
        original_send_message(sock, message, fd)

    caplog.set_level("WARNING")
    monkeypatch.setattr(snapshot, "get_vmm", lambda: vmm)
    monkeypatch.setattr(
        snapshot,
        "create_transfer_backend",
        lambda *_args, **_kwargs: backend,
    )
    monkeypatch.setattr(server_rpc, "send_message", drop_commit_response)

    with _server(socket_path, vmm, manager):
        assert snapshot.hydrate_weights(str(artifact_dir), socket_path, 0) is None
        assert manager.commit_published.is_set()

        reader = _GMSClientSession(socket_path, RequestedLockType.RO)
        assert reader.list_allocations() == (
            AllocationRecord("weight-0", 64),
            AllocationRecord("weight-1", 64),
        )
        reader.close()

    assert not vmm.imports
    assert not vmm.mapped
    assert not vmm.reservations
    assert backend.closed
    assert "commit response was lost" in caplog.text
