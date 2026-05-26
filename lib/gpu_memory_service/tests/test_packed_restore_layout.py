# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

import pytest

from gpu_memory_service.server import allocations as server_allocations
from gpu_memory_service.server.allocations import GMSAllocationManager


@pytest.mark.asyncio
async def test_packed_layout_publishes_allocations_over_hidden_backings(monkeypatch):
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
    infos = await allocations.create_packed_layout(
        backing_sizes=[8192],
        placements=[
            (4096, 4096, "weights", 0, 0),
            (4096, 4096, "weights", 0, 4096),
        ],
    )

    assert len(infos) == 2
    assert allocations.allocation_count == 2
    assert len(export_calls) == 1
    assert infos[0].allocation_id != infos[1].allocation_id
    assert infos[0].backing_id == infos[1].backing_id
    assert infos[0].backing_offset == 0
    assert infos[1].backing_offset == 4096
    assert [info.layout_slot for info in allocations.list_allocations()] == [0, 1]

    first_fd = allocations.export_allocation(infos[0].allocation_id)
    second_fd = allocations.export_allocation(infos[1].allocation_id)
    try:
        os.fstat(first_fd)
        os.fstat(second_fd)
        assert first_fd != second_fd
    finally:
        os.close(first_fd)
        os.close(second_fd)

    assert allocations.free_allocation(infos[0].allocation_id)
    assert allocations.allocation_count == 1
    assert allocations.free_allocation(infos[1].allocation_id)
    assert allocations.allocation_count == 0
