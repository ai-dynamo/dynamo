# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from gpu_memory_service.common import utils as common_utils
from gpu_memory_service.snapshot import storage_client, transfer
from gpu_memory_service.snapshot.model import (
    CURRENT_VERSION,
    AllocationEntry,
    SaveManifest,
)


def _write_dump_dir(tmp_path) -> list[AllocationEntry]:
    shards_dir = tmp_path / "shards"
    shards_dir.mkdir()
    (shards_dir / "shard_0000.bin").write_bytes(b"a" * 24)
    entries = [
        AllocationEntry(
            allocation_id="old-0",
            size=8,
            aligned_size=8,
            tag="weights",
            tensor_file="shards/shard_0000.bin",
            tensor_offset=0,
        ),
        AllocationEntry(
            allocation_id="old-1",
            size=16,
            aligned_size=16,
            tag="weights",
            tensor_file="shards/shard_0000.bin",
            tensor_offset=8,
        ),
    ]
    manifest = SaveManifest(
        version=CURRENT_VERSION,
        timestamp=1.0,
        layout_hash="layout",
        device=0,
        allocations=entries,
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest.to_dict()),
        encoding="utf-8",
    )
    (tmp_path / "gms_metadata.json").write_text(
        json.dumps(
            {
                "meta0": {
                    "allocation_id": "old-0",
                    "offset_bytes": 0,
                    "value": base64.b64encode(b"value0").decode("ascii"),
                },
                "meta1": {
                    "allocation_id": "old-1",
                    "offset_bytes": 4,
                    "value": base64.b64encode(b"value1").decode("ascii"),
                },
            }
        ),
        encoding="utf-8",
    )
    return entries


class _FakeMemoryManager:
    def __init__(self, events=None) -> None:
        self.mappings = {}
        self.metadata_writes = []
        self.connect_calls = []
        self.committed = False
        self.events = events

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def connect(self, *args, **kwargs) -> None:
        if self.events is not None:
            self.events.append("connect")
        self.connect_calls.append((args, kwargs))

    def create_mapping(self, *, size: int, tag: str) -> int:
        index = len(self.mappings)
        va = 0x1000_0000 + index * 0x1000
        self.mappings[va] = SimpleNamespace(allocation_id=f"new-{index}")
        return va

    def metadata_put(
        self,
        key: str,
        allocation_id: str,
        offset_bytes: int,
        value: bytes,
    ) -> bool:
        self.metadata_writes.append((key, allocation_id, offset_bytes, value))
        return True

    def commit(self) -> bool:
        self.committed = True
        return True


class _FakeSession:
    def __init__(self) -> None:
        self.targets = None
        self.closed = False

    def restore(self, targets) -> None:
        self.targets = dict(targets)

    def close(self) -> None:
        self.closed = True


class _FakeBackend:
    name = "fake"

    def __init__(self, session: _FakeSession, events=None) -> None:
        self.session = session
        self.sources = None
        self.closed = False
        self.events = events

    def start_restore(self, sources):
        if self.events is not None:
            self.events.append("start_restore")
        self.sources = list(sources)
        return self.session

    def close(self) -> None:
        self.closed = True


def test_storage_client_delegates_bytes_to_transfer_backend(tmp_path, monkeypatch):
    _write_dump_dir(tmp_path)
    events = []
    mm = _FakeMemoryManager(events)
    session = _FakeSession()
    backend = _FakeBackend(session, events)
    factory = MagicMock(return_value=backend)

    monkeypatch.setattr(storage_client, "_GMS_CORE_IMPORTS_AVAILABLE", True)
    monkeypatch.setattr(storage_client, "_GMS_IMPORTS_AVAILABLE", False)
    monkeypatch.setattr(storage_client, "_GMS_TENSOR_IMPORTS_AVAILABLE", False)
    monkeypatch.setattr(storage_client, "_TORCH_AVAILABLE", False)
    monkeypatch.setattr(storage_client, "torch", None)
    monkeypatch.setattr(storage_client, "_tensor_from_pointer", None)
    monkeypatch.setattr(
        storage_client,
        "GMSClientMemoryManager",
        lambda *_, **__: mm,
    )
    monkeypatch.setattr(
        storage_client,
        "RequestedLockType",
        SimpleNamespace(RW="RW"),
    )
    monkeypatch.setattr(storage_client, "create_transfer_backend", factory)
    monkeypatch.setattr(
        common_utils,
        "wait_for_weights_socket",
        lambda device: events.append(f"wait_for_socket:{device}"),
    )

    client = storage_client.GMSStorageClient(
        socket_path="/tmp/fake.sock",
        device=0,
        transfer_backend=transfer.NIXL_GDS_TRANSFER_BACKEND,
    )

    id_map = client.load_to_gms(str(tmp_path), max_workers=8, wait_for_socket=True)

    assert id_map == {"old-0": "new-0", "old-1": "new-1"}
    assert events[:3] == ["start_restore", "wait_for_socket:0", "connect"]
    factory.assert_called_once_with(
        transfer.NIXL_GDS_TRANSFER_BACKEND,
        device=0,
        max_workers=8,
        torch_module=None,
        tensor_from_pointer=None,
    )
    assert [source.allocation_id for source in backend.sources] == ["old-0", "old-1"]
    assert [source.file_offset for source in backend.sources] == [0, 8]
    assert session.targets["old-0"].va == 0x1000_0000
    assert session.targets["old-0"].byte_count == 8
    assert session.targets["old-1"].va == 0x1000_1000
    assert session.closed
    assert backend.closed
    assert mm.metadata_writes == [
        ("meta0", "new-0", 0, b"value0"),
        ("meta1", "new-1", 4, b"value1"),
    ]
    assert mm.committed


class _FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return False


class _FakeTorch:
    uint8 = object()
    cuda = _FakeCuda()


class _FakeSrc:
    pass


class _FakeDst:
    def __init__(self, copied) -> None:
        self._copied = copied

    def copy_(self, src) -> None:
        self._copied.append(src)


class _FakeCufileBindings:
    def __init__(self) -> None:
        self.driver_open_count = 0
        self.driver_close_count = 0
        self.handle_registers = []
        self.handle_deregisters = []
        self.buf_registers = []
        self.buf_deregisters = []
        self.read_calls = []

    def cuFileDriverOpen(self) -> None:
        self.driver_open_count += 1

    def cuFileDriverClose(self) -> None:
        self.driver_close_count += 1

    def cuFileHandleRegister(self, fd: int) -> str:
        self.handle_registers.append(fd)
        return f"handle-{fd}"

    def cuFileHandleDeregister(self, handle: str) -> None:
        self.handle_deregisters.append(handle)

    def cuFileBufRegister(self, ptr, size: int, flags: int) -> None:
        self.buf_registers.append((ptr.value, size, flags))

    def cuFileBufDeregister(self, ptr) -> None:
        self.buf_deregisters.append(ptr.value)

    def cuFileRead(
        self,
        handle: str,
        ptr,
        size: int,
        file_offset: int,
        dev_offset: int,
    ) -> int:
        self.read_calls.append((handle, ptr.value, size, file_offset, dev_offset))
        return min(size, 7)


def test_default_transfer_backend_copies_sources_to_gms_targets(monkeypatch):
    read_calls = []
    copied = []

    def fake_read_shard_streaming_to_queue(
        abs_path,
        sorted_entries,
        work_q,
        *,
        pin_memory,
        cancel_event,
        os_module,
        np_module,
        torch_module,
        logger,
        stats,
    ):
        read_calls.append((abs_path, sorted_entries, pin_memory))
        for entry in sorted_entries:
            work_q.put((entry, _FakeSrc()))
        return len(sorted_entries)

    def fake_tensor_from_pointer(va, shape, stride, dtype, device):
        copied.append((va, shape, stride, dtype, device))
        return _FakeDst(copied)

    monkeypatch.setattr(transfer, "_get_numpy_module", lambda: object())
    monkeypatch.setattr(
        transfer,
        "read_shard_streaming_to_queue",
        fake_read_shard_streaming_to_queue,
    )
    backend = transfer.DefaultTransferBackend(
        device=3,
        max_workers=4,
        torch_module=_FakeTorch,
        tensor_from_pointer=fake_tensor_from_pointer,
    )
    source = transfer.FileTransferSource(
        allocation_id="old-0",
        file_path="/tmp/shard.bin",
        file_offset=0,
        byte_count=32,
    )
    target = transfer.GMSTransferTarget(
        allocation_id="old-0",
        va=0xCAFE,
        device=3,
        byte_count=32,
    )

    session = backend.start_restore([source])
    try:
        session.restore({"old-0": target})
    finally:
        session.close()

    assert read_calls[0][0] == "/tmp/shard.bin"
    assert read_calls[0][1][0].allocation_id == "old-0"
    assert read_calls[0][2] is False
    assert copied[0] == (0xCAFE, [32], [1], _FakeTorch.uint8, 3)
    assert isinstance(copied[1], _FakeSrc)


def test_aio_transfer_backend_uses_aio_reader(monkeypatch):
    read_calls = []
    copied = []

    def fake_read_shard_aio_to_queue(
        abs_path,
        sorted_entries,
        work_q,
        *,
        pin_memory,
        cancel_event,
        os_module,
        np_module,
        torch_module,
        logger,
        stats,
    ):
        read_calls.append((abs_path, sorted_entries, pin_memory, stats.backend_name))
        for entry in sorted_entries:
            work_q.put((entry, _FakeSrc()))
        return len(sorted_entries)

    def fake_tensor_from_pointer(va, shape, stride, dtype, device):
        copied.append((va, shape, stride, dtype, device))
        return _FakeDst(copied)

    monkeypatch.setattr(transfer, "_get_numpy_module", lambda: object())
    monkeypatch.setattr(
        transfer,
        "read_shard_aio_to_queue",
        fake_read_shard_aio_to_queue,
    )
    backend = transfer.AioTransferBackend(
        device=3,
        max_workers=4,
        torch_module=_FakeTorch,
        tensor_from_pointer=fake_tensor_from_pointer,
    )
    source = transfer.FileTransferSource(
        allocation_id="old-0",
        file_path="/tmp/shard.bin",
        file_offset=0,
        byte_count=32,
    )
    target = transfer.GMSTransferTarget(
        allocation_id="old-0",
        va=0xCAFE,
        device=3,
        byte_count=32,
    )

    session = backend.start_restore([source])
    try:
        session.restore({"old-0": target})
    finally:
        session.close()

    assert read_calls[0][0] == "/tmp/shard.bin"
    assert read_calls[0][1][0].allocation_id == "old-0"
    assert read_calls[0][2] is False
    assert read_calls[0][3] == transfer.AIO_TRANSFER_BACKEND
    assert copied[0] == (0xCAFE, [32], [1], _FakeTorch.uint8, 3)
    assert isinstance(copied[1], _FakeSrc)


def test_cufile_gds_backend_reads_file_extent_to_target(tmp_path, monkeypatch):
    shard_path = tmp_path / "shard.bin"
    shard_path.write_bytes(b"x" * 64)
    bindings = _FakeCufileBindings()
    monkeypatch.setattr(transfer, "_load_cufile_bindings", lambda: bindings)
    monkeypatch.setattr(transfer, "_set_current_cuda_device", lambda device: None)

    backend = transfer.CufileGDSTransferBackend(device=2, max_workers=4)
    source = transfer.FileTransferSource(
        allocation_id="old-0",
        file_path=str(shard_path),
        file_offset=5,
        byte_count=16,
    )
    target = transfer.GMSTransferTarget(
        allocation_id="old-0",
        va=0xBEEF,
        device=2,
        byte_count=16,
    )

    session = backend.start_restore([source])
    session.restore({"old-0": target})
    backend.close()

    assert bindings.driver_open_count == 1
    assert bindings.driver_close_count == 1
    assert bindings.buf_registers == [(0xBEEF, 16, 0)]
    assert bindings.buf_deregisters == [0xBEEF]
    assert len(bindings.handle_registers) == 1
    assert bindings.handle_deregisters == [f"handle-{bindings.handle_registers[0]}"]
    assert bindings.read_calls == [
        (bindings.handle_deregisters[0], 0xBEEF, 16, 5, 0),
        (bindings.handle_deregisters[0], 0xBEEF, 9, 12, 7),
        (bindings.handle_deregisters[0], 0xBEEF, 2, 19, 14),
    ]


def test_nixl_gds_backend_registers_file_and_vram_extents(tmp_path, monkeypatch):
    shard_path = tmp_path / "shard.bin"
    shard_path.write_bytes(b"x" * 64)
    agent = MagicMock()
    file_reg = MagicMock()
    vram_reg = MagicMock()
    file_reg.trim.return_value = "file-xfer"
    vram_reg.trim.return_value = "vram-xfer"
    agent.register_memory.side_effect = [file_reg, vram_reg]
    agent.initialize_xfer.return_value = "handle"
    agent.transfer.return_value = "PROC"
    agent.check_xfer_state.return_value = "DONE"

    monkeypatch.setattr(transfer, "_NIXL_AVAILABLE", True)
    monkeypatch.setattr(transfer, "nixl_agent", MagicMock(return_value=agent))
    monkeypatch.setattr(transfer, "nixl_agent_config", MagicMock(return_value="cfg"))

    backend = transfer.NixlGDSTransferBackend(device=2)
    source = transfer.FileTransferSource(
        allocation_id="old-0",
        file_path=str(shard_path),
        file_offset=4,
        byte_count=32,
    )
    target = transfer.GMSTransferTarget(
        allocation_id="old-0",
        va=0xBEEF,
        device=2,
        byte_count=32,
    )

    session = backend.start_restore([source])
    session.restore({"old-0": target})
    backend.close()

    agent.create_backend.assert_called_once_with("GDS_MT")
    file_descs = agent.register_memory.call_args_list[0].args[0]
    vram_descs = agent.register_memory.call_args_list[1].args[0]
    assert [(desc[0], desc[1], desc[3]) for desc in file_descs] == [(4, 32, "")]
    assert vram_descs == [(0xBEEF, 32, 2, "")]
    agent.initialize_xfer.assert_called_once()
    assert agent.initialize_xfer.call_args.args[:3] == (
        "READ",
        "vram-xfer",
        "file-xfer",
    )
    agent.release_xfer_handle.assert_called_once_with("handle")
    agent.deregister_memory.assert_any_call(vram_reg)
    agent.deregister_memory.assert_any_call(file_reg)
