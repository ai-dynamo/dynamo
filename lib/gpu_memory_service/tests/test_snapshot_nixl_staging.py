# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for NIXL/POSIX staging restore planning."""

import os
import threading

import pytest
from _deps import HAS_GMS

if not HAS_GMS:
    pytest.skip(
        "gpu_memory_service package is not available in this test image",
        allow_module_level=True,
    )

from gpu_memory_service.snapshot.backends.pinned_host import PinnedCopySlot
from _fake_vmm import FakeVMM
from blake3 import blake3
from gpu_memory_service.snapshot.backends.nixl import NixlTransferBackend
from gpu_memory_service.snapshot.backends.nixl_common import (
    NixlFileGroup,
    NixlWorkGroup,
    split_work_groups,
)
from gpu_memory_service.snapshot.backends.nixl_staging import (
    _NixlPosixStagingTransferSession,
    restore_file_groups_with_nixl_staging,
)
from gpu_memory_service.snapshot.transfer import (
    FileIntegrity,
    FileTransferSource,
    GMSSnapshotConfig,
    GMSTransferTarget,
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.none,
    pytest.mark.gpu_0,
]


def _work_group(name: str, byte_count: int) -> NixlWorkGroup:
    source = FileTransferSource(
        allocation_id=name,
        file_path=f"/checkpoint/{name}.bin",
        file_offset=0,
        byte_count=byte_count,
    )
    file_group: NixlFileGroup = (source.file_path, [source])
    return name, [file_group]


def test_split_work_groups_limits_bucket_count():
    work_groups = [_work_group(f"shard-{idx}", 1) for idx in range(142)]

    buckets = split_work_groups(work_groups, worker_count=8)

    assert len(buckets) == 8
    assert sum(len(file_groups) for _name, file_groups in buckets) == 142
    expected = sorted(f"shard-{idx}" for idx in range(142))
    actual = sorted(
        source.allocation_id
        for _bucket_name, file_groups in buckets
        for _file_path, sources in file_groups
        for source in sources
    )
    assert actual == expected


def test_split_work_groups_balances_by_bytes():
    work_groups = [
        _work_group("large", 10),
        _work_group("medium-a", 6),
        _work_group("medium-b", 6),
        _work_group("small-a", 1),
        _work_group("small-b", 1),
    ]

    buckets = split_work_groups(work_groups, worker_count=2)
    bucket_sizes = sorted(
        sum(source.byte_count for _path, sources in file_groups for source in sources)
        for _name, file_groups in buckets
    )

    assert bucket_sizes == [12, 12]


def test_staging_prep_starts_before_restore(monkeypatch):
    from gpu_memory_service.snapshot.backends import nixl_staging

    source = FileTransferSource(
        allocation_id="alloc-0",
        file_path="/checkpoint/shard.bin",
        file_offset=0,
        byte_count=4096,
    )
    prep_started = threading.Event()
    allow_finish = threading.Event()

    class FakeApi:
        @staticmethod
        def agent_config_type(*, backends):
            return {"backends": backends}

        @staticmethod
        def agent_type(_agent_name, _config):
            return FakeAgent()

    class FakeAgent:
        def create_backend(self, _backend_name, backend_params=None):
            self.backend_params = backend_params

    def group_sources(_sources):
        return {"file": [(source.file_path, [source])]}

    def fake_load_nixl_api():
        prep_started.set()
        assert allow_finish.wait(timeout=1.0)
        return FakeApi()

    fake_vmm = FakeVMM()
    monkeypatch.setattr(nixl_staging, "get_vmm", lambda: fake_vmm)
    monkeypatch.setattr(nixl_staging, "load_nixl_api", fake_load_nixl_api)
    monkeypatch.setattr(nixl_staging, "make_pinned_copy_slots", lambda _vmm, _count: [])

    session = _NixlPosixStagingTransferSession(
        backend_name="test-backend",
        device=0,
        max_workers=1,
        group_sources=group_sources,
        group_kind="file",
        warn_under_parallelized=False,
        posix_backend_params={"ios_pool_size": "64", "kernel_queue_size": "16"},
        sources=[source],
    )
    try:
        assert prep_started.wait(timeout=1.0)
    finally:
        allow_finish.set()
        session.close()


def _restore_staged_bytes(
    monkeypatch,
    data: bytes,
    expected_digest: str,
):
    from gpu_memory_service.snapshot.backends import nixl_staging

    class Slot:
        def __init__(self):
            self._buffer = bytearray(len(data))
            self.view = memoryview(self._buffer)
            self.ptr = 123
            self.copies = []

        def wait(self):
            pass

        def copy_to_device_async(self, dst_ptr, size):
            self.copies.append((dst_ptr, bytes(self.view[:size])))

    slot = Slot()
    sources = [
        FileTransferSource(
            allocation_id="weight-0",
            file_path="/checkpoint/shard.bin",
            file_offset=0,
            byte_count=4,
        ),
        FileTransferSource(
            allocation_id="weight-1",
            file_path="/checkpoint/shard.bin",
            file_offset=4,
            byte_count=4,
        ),
    ]
    targets = {
        "weight-0": GMSTransferTarget("weight-0", 100, 0, 4),
        "weight-1": GMSTransferTarget("weight-1", 200, 0, 4),
    }

    def fake_read(
        _agent,
        _agent_name,
        _fd,
        _file_path,
        file_offset,
        _host_ptr,
        size,
        _backend_name,
    ):
        slot._buffer[:size] = data[file_offset : file_offset + size]

    monkeypatch.setattr(
        nixl_staging, "open_direct_read_fd", lambda *_args, **_kwargs: 9
    )
    monkeypatch.setattr(nixl_staging, "_read_file_to_dram", fake_read)
    monkeypatch.setattr(nixl_staging.os, "close", lambda _fd: None)

    transferred = restore_file_groups_with_nixl_staging(
        backend_name="nixl",
        agent=object(),
        agent_name="test-agent",
        file_groups=[("/checkpoint/shard.bin", sources)],
        integrity_by_path={"/checkpoint/shard.bin": FileIntegrity(8, expected_digest)},
        targets=targets,
        slots=[slot],
    )
    return transferred, slot.copies


def test_staging_restore_hashes_the_exact_transferred_bytes(monkeypatch):
    data = b"abcdefgh"

    transferred, copies = _restore_staged_bytes(
        monkeypatch,
        data,
        blake3(data).hexdigest(),
    )

    assert transferred == len(data)
    assert copies == [(100, b"abcd"), (200, b"efgh")]


def test_staging_restore_rejects_same_length_mutation(monkeypatch):
    with pytest.raises(RuntimeError, match="BLAKE3 mismatch"):
        _restore_staged_bytes(
            monkeypatch,
            b"xbcdefgh",
            blake3(b"abcdefgh").hexdigest(),
        )


def _restore_with_staging_backend(monkeypatch, tmp_path, saved, on_disk):
    from gpu_memory_service.snapshot.backends import nixl_staging

    class FakeApi:
        @staticmethod
        def agent_config_type(*, backends):
            return {"backends": backends}

        @staticmethod
        def agent_type(_agent_name, _config):
            return FakeAgent()

    class FakeAgent:
        def create_backend(self, _backend_name, backend_params=None):
            self.backend_params = backend_params

    class Slot:
        def __init__(self):
            self._buffer = bytearray(len(saved))
            self.view = memoryview(self._buffer)
            self.ptr = 123

        def wait(self):
            pass

        def copy_to_device_async(self, _dst_ptr, _size):
            pass

    def fake_read(_agent, _name, fd, _path, file_offset, _ptr, size, _backend):
        slot._buffer[:size] = os.pread(fd, size, file_offset)

    shard_path = tmp_path / "shard.bin"
    shard_path.write_bytes(on_disk)
    slot = Slot()
    fake_vmm = FakeVMM()
    monkeypatch.setattr(nixl_staging, "get_vmm", lambda: fake_vmm)
    monkeypatch.setattr(nixl_staging, "load_nixl_api", lambda: FakeApi())
    monkeypatch.setattr(
        nixl_staging, "make_pinned_copy_slots", lambda _vmm, _count: [slot]
    )
    monkeypatch.setattr(nixl_staging, "close_pinned_copy_slots", lambda *_args: None)
    monkeypatch.setattr(
        nixl_staging,
        "open_direct_read_fd",
        lambda path, **_kwargs: os.open(path, os.O_RDONLY),
    )
    monkeypatch.setattr(nixl_staging, "_read_file_to_dram", fake_read)

    backend = NixlTransferBackend(config=GMSSnapshotConfig(device=0, max_workers=1))
    session = backend.start_restore(
        [FileTransferSource("weight", str(shard_path), 0, len(saved))],
        {str(shard_path): FileIntegrity(len(saved), blake3(saved).hexdigest())},
    )
    try:
        session.restore({"weight": GMSTransferTarget("weight", 100, 0, len(saved))})
    finally:
        session.close()
        backend.close()


def test_staging_backend_restores_an_unchanged_shard(monkeypatch, tmp_path):
    _restore_with_staging_backend(monkeypatch, tmp_path, b"abcdefgh", b"abcdefgh")


def test_staging_backend_rejects_same_length_mutation(monkeypatch, tmp_path):
    with pytest.raises(RuntimeError, match="BLAKE3 mismatch"):
        _restore_with_staging_backend(monkeypatch, tmp_path, b"abcdefgh", b"xbcdefgh")


def test_pinned_copy_slot_skips_unsupported_host_register():
    class HostRegisterUnsupportedVMM(FakeVMM):
        def host_register(self, ptr, size):
            raise RuntimeError(
                "_sycl_vmm: register_host_memory not available (oneAPI too old)"
            )

        def host_unregister(self, ptr):
            raise AssertionError("host_unregister should be skipped when registration is unsupported")

    slot = PinnedCopySlot(HostRegisterUnsupportedVMM())

    assert slot._registered is False

    slot.close()
