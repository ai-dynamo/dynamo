# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from pathlib import Path

import msgspec
import pytest
from _fake_vmm import FakeVMM
from gpu_memory_service.common.locks import GrantedLockType, RequestedLockType
from gpu_memory_service.snapshot import disk as snapshot_disk
from gpu_memory_service.v1.client.session import _GMSClientSession
from gpu_memory_service.v1.protocol import AllocationRecord
from gpu_memory_service.v1.server.rpc import GMSRPCServer, GMSServerMemoryManager
from gpu_memory_service.v1.snapshot import weight_artifact

pytestmark = [pytest.mark.pre_merge, pytest.mark.integration, pytest.mark.gpu_0]


class _ByteVMM(FakeVMM):
    def __init__(self) -> None:
        super().__init__(granularity=64)
        self._physical: dict[int, bytearray] = {}
        self._exported: dict[int, int] = {}
        self._imported: dict[int, int] = {}

    def create_tolerate_oom(self, size, device):
        allocated, handle = super().create_tolerate_oom(size, device)
        self._physical[handle] = bytearray(size)
        return allocated, handle

    def release(self, handle):
        if handle in self.server_handles:
            self._physical.pop(handle)
        self._imported.pop(handle, None)
        super().release(handle)

    def export_to_shareable_handle(self, handle):
        fd = super().export_to_shareable_handle(handle)
        self._exported[os.fstat(fd).st_ino] = handle
        return fd

    def import_shareable_handle_close_fd(self, fd):
        physical_handle = self._exported[os.fstat(fd).st_ino]
        imported_handle = super().import_shareable_handle_close_fd(fd)
        self._imported[imported_handle] = physical_handle
        return imported_handle

    def _allocation(self, va, size):
        mapped_size, imported_handle = self.mapped[va]
        assert size <= mapped_size
        return self._physical[self._imported[imported_handle]]

    def read(self, va, size):
        return bytes(self._allocation(va, size)[:size])

    def write(self, va, data):
        self._allocation(va, len(data))[: len(data)] = data


class _Writer:
    vmm: _ByteVMM

    def __init__(self, path, *, device):
        assert device == 0
        self._path = Path(path)
        self._data = bytearray()

    def __enter__(self):
        return self

    def write_device(self, src_ptr, byte_count):
        self._data.extend(self.vmm.read(src_ptr, byte_count))

    def __exit__(self, *_args):
        self._path.write_bytes(self._data)


class _Transfer:
    def __init__(self, vmm, sources, *, fail=False):
        self._vmm = vmm
        self._sources = sources
        self._fail = fail

    def restore(self, targets):
        if self._fail:
            raise RuntimeError("restore failed")
        for source in self._sources:
            target = targets[source.allocation_id]
            shard = Path(source.file_path).read_bytes()
            data = shard[source.file_offset : source.file_offset + source.byte_count]
            assert len(data) == target.byte_count
            self._vmm.write(target.va, data)

    def close(self):
        pass


class _Backend:
    def __init__(self, vmm, *, fail=False):
        self._vmm = vmm
        self._fail = fail
        self.closed = False

    def start_restore(self, sources):
        return _Transfer(self._vmm, sources, fail=self._fail)

    def close(self):
        self.closed = True


@contextmanager
def _server(path: str, vmm: FakeVMM):
    manager = GMSServerMemoryManager("GPU-0", vmm, 0)
    server = GMSRPCServer(path, manager)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield manager
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)
        assert not thread.is_alive()


def _configure_device(monkeypatch, vmm: _ByteVMM) -> None:
    monkeypatch.setattr(weight_artifact, "get_vmm", lambda: vmm)
    monkeypatch.setattr(
        weight_artifact.device_identity,
        "invalidate_device_uuid_cache",
        lambda: None,
    )
    monkeypatch.setattr(
        weight_artifact.device_identity,
        "get_device_uuid",
        lambda _device: "GPU-0",
    )


def _map(session, record, vmm, access):
    return weight_artifact._map_export(
        session,
        record,
        vmm,
        0,
        64,
        access,
    )


@pytest.mark.timeout(10)
def test_weight_artifact_round_trip_preserves_exact_ids_and_bytes(
    tmp_path,
    monkeypatch,
) -> None:
    socket_path = str(tmp_path / "weights.sock")
    artifact = tmp_path / "artifact"
    records = (
        AllocationRecord("weight-0", 64),
        AllocationRecord("weight-1", 128),
    )
    expected = {
        "weight-0": bytes(range(64)),
        "weight-1": bytes((255 - index) % 256 for index in range(128)),
    }

    source_vmm = _ByteVMM()
    _Writer.vmm = source_vmm
    _configure_device(monkeypatch, source_vmm)
    monkeypatch.setattr(snapshot_disk, "DeviceToFileWriter", _Writer)
    with _server(socket_path, source_vmm):
        writer = _GMSClientSession(socket_path, RequestedLockType.RW)
        mappings = []
        for record in records:
            writer.allocate(record.allocation_id, record.aligned_size)
            mapping = _map(writer, record, source_vmm, GrantedLockType.RW)
            source_vmm.write(mapping[0].base, expected[record.allocation_id])
            mappings.append(mapping)
        writer.commit()
        manifest = weight_artifact.save_weights(
            str(artifact),
            socket_path,
            0,
            shard_size_bytes=64,
        )
        for mapping in reversed(mappings):
            weight_artifact._release_mapping(source_vmm, mapping)
        writer.close()

    assert [
        (allocation.allocation_id, allocation.aligned_size)
        for allocation in manifest.allocations
    ] == [(record.allocation_id, record.aligned_size) for record in records]

    target_vmm = _ByteVMM()
    backend = _Backend(target_vmm)
    _configure_device(monkeypatch, target_vmm)
    monkeypatch.setattr(
        weight_artifact,
        "create_transfer_backend",
        lambda *_args, **_kwargs: backend,
    )
    with _server(socket_path, target_vmm):
        weight_artifact.load_weights(str(artifact), socket_path, 0)
        reader = _GMSClientSession(socket_path, RequestedLockType.RO)
        assert weight_artifact._list_allocations(reader) == records
        restored = [
            _map(reader, record, target_vmm, GrantedLockType.RO) for record in records
        ]
        assert {
            mapping.allocation_id: target_vmm.read(mapping.base, mapping.aligned_size)
            for mapping, _handle in restored
        } == expected
        for mapping in reversed(restored):
            weight_artifact._release_mapping(target_vmm, mapping)
        reader.close()
    assert backend.closed


@pytest.mark.timeout(10)
def test_failed_weight_load_does_not_publish_partial_epoch(
    tmp_path,
    monkeypatch,
) -> None:
    socket_path = str(tmp_path / "weights.sock")
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    allocation = weight_artifact.WeightArtifactAllocation(
        "weight",
        64,
        "shard.bin",
        0,
    )
    (artifact / "manifest.json").write_bytes(
        msgspec.json.encode(weight_artifact.WeightArtifactManifest(1, (allocation,)))
    )
    (artifact / "shard.bin").write_bytes(bytes(64))

    vmm = _ByteVMM()
    backend = _Backend(vmm, fail=True)
    _configure_device(monkeypatch, vmm)
    monkeypatch.setattr(
        weight_artifact,
        "create_transfer_backend",
        lambda *_args, **_kwargs: backend,
    )
    with _server(socket_path, vmm) as manager:
        with pytest.raises(RuntimeError, match="restore failed"):
            weight_artifact.load_weights(str(artifact), socket_path, 0)
        assert not manager._sessions._committed
        assert not vmm.server_handles
        replacement = _GMSClientSession(socket_path, RequestedLockType.RW)
        replacement.close()
    assert backend.closed
