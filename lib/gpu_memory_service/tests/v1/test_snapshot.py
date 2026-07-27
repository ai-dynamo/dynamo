# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from pathlib import Path

import pytest
from _fake_vmm import FakeVMM
from gpu_memory_service.common.locks import GrantedLockType, RequestedLockType
from gpu_memory_service.core.client.session import _GMSClientSession
from gpu_memory_service.core.server.gms import GMSServerMemoryManager
from gpu_memory_service.core.server.rpc import GMSRPCServer
from gpu_memory_service.v1 import loader, snapshot
from gpu_memory_service.v1.memory_manager import GMSClientMemoryManager

pytestmark = [pytest.mark.pre_merge, pytest.mark.integration, pytest.mark.gpu_0]


@contextmanager
def _server(path: str, vmm: FakeVMM):
    server = GMSRPCServer(
        path,
        GMSServerMemoryManager("GPU-0", vmm, 0),
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
    def __init__(self, sources, events):
        self._sources = sources
        self._events = events

    def restore(self, targets):
        self._events.append("restore")
        assert set(targets) == {source.allocation_id for source in self._sources}
        assert {
            allocation_id: target.byte_count
            for allocation_id, target in targets.items()
        } == {source.allocation_id: source.byte_count for source in self._sources}

    def close(self):
        self._events.append("transfer_close")


class _Backend:
    def __init__(self, events):
        self._events = events

    def start_restore(self, sources):
        self._events.append("start_restore")
        return _Transfer(sources, self._events)

    def close(self):
        self._events.append("backend_close")


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
def test_exact_ids_roundtrip_after_fresh_server_loader_returns(
    tmp_path,
    monkeypatch,
    caplog,
) -> None:
    source_path = str(tmp_path / "source.sock")
    source_vmm = FakeVMM(granularity=64)
    checkpoint_dir = tmp_path / "checkpoint"
    artifact_dir = str(checkpoint_dir / "device-0")
    caplog.set_level("INFO")
    monkeypatch.setattr(
        GMSClientMemoryManager,
        "_gpu_identity",
        lambda self: "GPU-0",
    )
    monkeypatch.setattr(snapshot, "DeviceToFileWriter", _Writer)

    with _server(source_path, source_vmm):
        restored_engine = GMSClientMemoryManager(source_path, source_vmm, 0)
        restored_engine.connect(RequestedLockType.RW)
        first_va = restored_engine.create_mapping(64)
        second_va = restored_engine.create_mapping(65)
        saved_mappings = tuple(
            (mapping.allocation_id, mapping.aligned_size, mapping.base)
            for mapping in restored_engine.mappings
        )
        restored_engine.commit()
        monkeypatch.setattr(snapshot, "get_vmm", lambda: source_vmm)
        manifest = snapshot.save_weights(
            artifact_dir,
            source_path,
            0,
            shard_size_bytes=64,
        )
        restored_engine.unmap_all_vas()
        restored_engine.disconnect()

    assert [
        (allocation.allocation_id, allocation.aligned_size)
        for allocation in manifest.allocations
    ] == [(allocation_id, size) for allocation_id, size, _va in saved_mappings]
    assert [
        (allocation.shard, allocation.offset) for allocation in manifest.allocations
    ] == [
        (os.path.join("shards", "shard_0000.bin"), 0),
        (os.path.join("shards", "shard_0001.bin"), 0),
    ]

    target_vmm = FakeVMM(granularity=64)
    events = []
    monkeypatch.setattr(snapshot, "get_vmm", lambda: target_vmm)
    monkeypatch.setattr(
        snapshot,
        "create_transfer_backend",
        lambda *_args, **_kwargs: _Backend(events),
    )
    monkeypatch.setattr(loader, "get_socket_path", lambda *_args: source_path)

    with _server(source_path, target_vmm):
        assert (
            loader.main(
                [
                    "--checkpoint-dir",
                    str(checkpoint_dir),
                    "--device",
                    "0",
                ]
            )
            is None
        )
        assert not target_vmm.imports
        assert not target_vmm.reservations

        restored_engine.connect(RequestedLockType.RO)
        restored_engine.remap_all_vas()
        assert (
            tuple(
                (mapping.allocation_id, mapping.aligned_size, mapping.base)
                for mapping in restored_engine.mappings
            )
            == saved_mappings
        )
        assert {mapping.base for mapping in restored_engine.mappings} == {
            first_va,
            second_va,
        }
        assert source_vmm.access == {
            first_va: GrantedLockType.RO,
            second_va: GrantedLockType.RO,
        }

        restored = _GMSClientSession(source_path, RequestedLockType.RO)
        assert [
            (record.allocation_id, record.aligned_size)
            for record in restored.list_allocations()
        ] == [
            (allocation_id, aligned_size)
            for allocation_id, aligned_size, _va in saved_mappings
        ]
        for allocation_id, _aligned_size, _va in saved_mappings:
            os.close(restored.export(allocation_id))
        restored.close()
        restored_engine.unmap_all_vas()
        restored_engine.disconnect()

    assert events == [
        "start_restore",
        "restore",
        "transfer_close",
        "backend_close",
    ]
    messages = [record.message for record in caplog.records]
    for phase in (
        "enumerate/map/import setup",
        "device-to-file shard write",
        "release",
        "total",
    ):
        assert any(
            f"GMS V1 saver {phase} device=0 allocations=2 bytes=192 elapsed=" in message
            for message in messages
        )
    assert any("GMS V1 loader target allocation" in message for message in messages)
    assert any("GMS V1 loader NIXL transfer" in message for message in messages)
    assert any("GMS V1 loader commit/publish" in message for message in messages)
    assert any("GMS V1 loader total" in message for message in messages)
    assert any("GMS V1 loader complete; exiting" in message for message in messages)
