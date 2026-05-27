# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

import pytest

from gpu_memory_service.common.protocol.messages import AllocateResponse
from gpu_memory_service.server import allocations as server_allocations
from gpu_memory_service.server.allocations import GMSAllocationManager
from gpu_memory_service.snapshot.model import AllocationEntry, SaveManifest
from gpu_memory_service.snapshot.storage_client import GMSStorageClient


@pytest.mark.asyncio
async def test_allocation_manager_allocate_many_creates_independent_handles(
    monkeypatch,
):
    monkeypatch.setattr(server_allocations, "cuda_ensure_initialized", lambda: None)
    monkeypatch.setattr(
        server_allocations,
        "cumem_get_allocation_granularity",
        lambda device: 4096,
    )

    next_handle = 1000

    def create_handle(size: int, device: int) -> tuple[bool, int]:
        nonlocal next_handle
        next_handle += 1
        return True, next_handle

    monkeypatch.setattr(
        server_allocations, "cumem_create_tolerate_oom", create_handle
    )
    monkeypatch.setattr(server_allocations, "cumem_release", lambda handle: None)

    export_calls: list[int] = []

    def export_fd(handle: int) -> int:
        export_calls.append(handle)
        read_fd, write_fd = os.pipe()
        os.close(write_fd)
        return read_fd

    monkeypatch.setattr(
        server_allocations, "cumem_export_to_shareable_handle", export_fd
    )

    allocations = GMSAllocationManager(device=0)
    infos = await allocations.allocate_many(
        [(4096, "weights"), (8192, "weights")]
    )

    try:
        assert [info.size for info in infos] == [4096, 8192]
        assert [info.aligned_size for info in infos] == [4096, 8192]
        assert [info.layout_slot for info in infos] == [0, 1]
        assert infos[0].allocation_id != infos[1].allocation_id
        assert [info.handle for info in infos] == export_calls
        assert allocations.allocation_count == 2
        first_fd = allocations.export_allocation(infos[0].allocation_id)
        second_fd = allocations.export_allocation(infos[1].allocation_id)
        try:
            os.fstat(first_fd)
            os.fstat(second_fd)
            assert first_fd != second_fd
        finally:
            os.close(first_fd)
            os.close(second_fd)
    finally:
        allocations.clear_all()


def test_restore_phase_a_uses_batched_allocation_api():
    class FakeMemoryManager:
        def __init__(self):
            self.allocate_handle_calls = 0
            self.create_mapping_calls = 0
            self.mapped: list[tuple[int, int, int, str, str, int]] = []
            self._next_va = 0x100000

        def allocate_handles(self, specs):
            assert specs == [(4096, "weights"), (8192, "weights")]
            return [
                AllocateResponse("new-0", 4096, 4096, 0),
                AllocateResponse("new-1", 8192, 8192, 1),
            ]

        def allocate_handle(self, *args, **kwargs):
            self.allocate_handle_calls += 1
            raise AssertionError("restore should use allocate_handles()")

        def create_mapping(self, *args, **kwargs):
            self.create_mapping_calls += 1
            raise AssertionError("restore should not allocate through create_mapping()")

        def export_handle(self, allocation_id):
            raise AssertionError("restore should use export_handles()")

        def export_handles(self, allocation_ids):
            fds = []
            for allocation_id in allocation_ids:
                assert allocation_id in {"new-0", "new-1"}
                read_fd, write_fd = os.pipe()
                os.close(write_fd)
                fds.append(read_fd)
            return fds

        def reserve_va(self, size):
            raise AssertionError("restore should use reserve_va_arena()")

        def reserve_va_arena(self, sizes):
            assert sizes == [4096, 8192]
            va = self._next_va
            self._next_va += sum(sizes)
            return va, [va, va + 4096]

        def map_va(self, fd, va, size, allocation_id, tag, layout_slot):
            raise AssertionError("restore should use map_va_at_reserved()")

        def map_va_at_reserved(
            self,
            fd,
            va,
            size,
            allocation_id,
            tag,
            layout_slot,
            *,
            set_access=True,
        ):
            assert set_access is False
            os.close(fd)
            self.mapped.append((va, size, 0, allocation_id, tag, layout_slot))
            return 0

        def set_access_all_vas(self):
            pass

    manifest = SaveManifest(
        version="1.0",
        timestamp=0.0,
        layout_hash="",
        device=0,
        allocations=[
            AllocationEntry("old-0", 4096, 4096, "weights", "shard_0000.bin"),
            AllocationEntry("old-1", 8192, 8192, "weights", "shard_0000.bin"),
        ],
    )

    manager = FakeMemoryManager()
    id_map, targets = GMSStorageClient(
        socket_path="/tmp/gms.sock",
        device=0,
    )._allocate_restore_targets(manager, manifest)

    assert manager.allocate_handle_calls == 0
    assert manager.create_mapping_calls == 0
    assert [mapped[3:] for mapped in manager.mapped] == [
        ("new-0", "weights", 0),
        ("new-1", "weights", 1),
    ]
    assert set(id_map) == {"old-0", "old-1"}
    assert set(targets) == {"old-0", "old-1"}
    assert id_map == {"old-0": "new-0", "old-1": "new-1"}
    assert targets["old-0"].va == 0x100000
    assert targets["old-0"].byte_count == 4096
    assert targets["old-1"].byte_count == 8192

