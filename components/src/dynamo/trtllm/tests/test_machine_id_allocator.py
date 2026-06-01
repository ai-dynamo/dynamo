# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for DisaggMachineIdAllocator."""

from __future__ import annotations

import tempfile

import pytest

from dynamo.trtllm.machine_id_allocator import DisaggMachineIdAllocator

pytestmark = [
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


class TestDisaggMachineIdAllocator:
    def test_allocate_unique_slots(self):
        """All allocated slots are distinct."""
        with tempfile.TemporaryDirectory() as tmpdir:
            alloc = DisaggMachineIdAllocator(tmpdir)
            ids = [alloc.allocate(i) for i in range(100)]
            assert len(set(ids)) == 100
            for slot in ids:
                assert 1 <= slot <= 1024

    def test_allocate_recovers_orphaned_file(self):
        """If connection_id file already exists, allocate() returns that slot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            alloc = DisaggMachineIdAllocator(tmpdir)
            slot = alloc.allocate(42)
            # Simulate process death: re-allocate same connection_id
            slot2 = alloc.allocate(42)
            assert slot == slot2

    def test_free_makes_slot_available(self):
        """Free releases the slot for reuse by another connection_id."""
        with tempfile.TemporaryDirectory() as tmpdir:
            alloc = DisaggMachineIdAllocator(tmpdir)
            slot1 = alloc.allocate(1)
            alloc.free(1)
            slot2 = alloc.allocate(2)
            assert slot1 == slot2  # slot was reclaimed

    def test_exhaustion_raises(self):
        """Allocate all 1024 slots; next allocate raises RuntimeError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            alloc = DisaggMachineIdAllocator(tmpdir)
            allocated = []
            for cid in range(1, 1025):
                allocated.append(alloc.allocate(cid))
            assert len(set(allocated)) == 1024
            with pytest.raises(RuntimeError, match="pool exhausted"):
                alloc.allocate(9999)

    def test_zero_slot_never_allocated(self):
        """Slot 0 is never allocated; it is the HandlerBase sentinel."""
        with tempfile.TemporaryDirectory() as tmpdir:
            alloc = DisaggMachineIdAllocator(tmpdir)
            for cid in range(500):
                slot = alloc.allocate(cid)
                assert slot != 0
