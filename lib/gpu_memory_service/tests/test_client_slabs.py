# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""First-fit slab packing in the V0 client memory manager.

Exercises the allocator through GMSClientMemoryManager with the CUDA VMM and
the GMS session faked out, so the packing arithmetic is covered without a GPU.
"""

from __future__ import annotations

import pytest

from gpu_memory_service.client.memory_manager import GMSClientMemoryManager
from gpu_memory_service.common.locks import GrantedLockType

GRANULARITY = 2 * 1024 * 1024
SLAB = 16 * GRANULARITY


class _FakeVMM:
    """Hands out non-overlapping VA reservations and records what is mapped."""

    def __init__(self) -> None:
        self.next_va = 1 << 40
        self.reservations: dict[int, int] = {}
        self.mapped: dict[int, int] = {}
        self.released: list[int] = []
        self.next_handle = 1

    def ensure_initialized(self) -> None:
        pass

    def get_allocation_granularity(self, device: int) -> int:
        return GRANULARITY

    def address_reserve(self, size: int, granularity: int) -> int:
        va = self.next_va
        self.next_va += size + granularity
        self.reservations[va] = size
        return va

    def address_free(self, va: int, size: int) -> None:
        assert self.reservations.pop(va, None) == size, f"bad free of 0x{va:x}"

    def import_shareable_handle_close_fd(self, fd: int) -> int:
        handle = self.next_handle
        self.next_handle += 1
        return handle

    def map(self, va: int, size: int, handle: int) -> None:
        self.mapped[va] = size

    def unmap(self, va: int, size: int) -> None:
        self.mapped.pop(va, None)

    def release(self, handle: int) -> None:
        self.released.append(handle)

    def set_access(self, va, size, device, lock_type) -> None:
        pass

    def synchronize(self) -> None:
        pass

    def validate_pointer(self, va: int) -> None:
        pass


class _FakeSession:
    """Minimal stand-in for the server RPC surface the allocator touches."""

    def __init__(self) -> None:
        self.live: dict[str, int] = {}
        self.slot = 0
        self.next_id = 0

    def allocate_info(self, aligned_size: int, tag: str):
        self.next_id += 1
        allocation_id = f"alloc-{self.next_id}"
        self.live[allocation_id] = aligned_size
        slot, self.slot = self.slot, self.slot + 1
        return type(
            "AllocateResponse",
            (),
            {
                "allocation_id": allocation_id,
                "aligned_size": aligned_size,
                "layout_slot": slot,
            },
        )()

    def export(self, allocation_id: str) -> int:
        assert allocation_id in self.live, f"export of freed {allocation_id}"
        return 7

    def free(self, allocation_id: str) -> bool:
        return self.live.pop(allocation_id, None) is not None


@pytest.fixture()
def manager(monkeypatch):
    vmm = _FakeVMM()
    monkeypatch.setattr(
        "gpu_memory_service.client.memory_manager.get_vmm", lambda: vmm
    )

    def build(slab_size=SLAB):
        mgr = GMSClientMemoryManager(
            "/tmp/does-not-exist.sock", device=0, tag="weights", slab_size=slab_size
        )
        mgr._client = _FakeSession()
        mgr._granted_lock_type = GrantedLockType.RW
        mgr.vmm = vmm
        return mgr

    build.vmm = vmm
    return build


def test_many_allocations_share_one_slab(manager):
    mgr = manager()
    vas = [mgr.create_mapping(size=GRANULARITY, tag="weights") for _ in range(8)]

    assert len(set(vas)) == 8
    # One server allocation, one VA reservation, one cuMemMap for all 8.
    assert len(mgr.mappings) == 1
    assert len(manager.vmm.reservations) == 1
    assert all(mgr.owns(va) for va in vas)


def test_total_bytes_reports_carved_not_reserved(manager):
    mgr = manager()
    mgr.create_mapping(size=GRANULARITY, tag="weights")

    # A slab is mapped whole; only the carved byte counts as weight memory.
    assert mgr.total_bytes == GRANULARITY
    assert mgr.reserved_bytes == SLAB


def test_slab_grows_when_full_and_retires_when_empty(manager):
    mgr = manager()
    vas = [mgr.create_mapping(size=4 * GRANULARITY, tag="weights") for _ in range(6)]
    assert len(mgr.mappings) == 2, "16 granules per slab, 4 per carve -> 2 slabs"

    for va in vas:
        mgr.destroy_mapping(va, 4 * GRANULARITY)

    assert mgr.mappings == {}
    assert manager.vmm.reservations == {}
    assert mgr.total_bytes == 0


def test_freed_region_is_reused_and_holes_coalesce(manager):
    mgr = manager()
    first = mgr.create_mapping(size=GRANULARITY, tag="weights")
    second = mgr.create_mapping(size=GRANULARITY, tag="weights")
    mgr.create_mapping(size=GRANULARITY, tag="weights")

    assert len(mgr.mappings) == 1
    mgr.destroy_mapping(first, GRANULARITY)
    assert mgr.create_mapping(size=GRANULARITY, tag="weights") == first

    # Free two adjacent regions out of order; they must merge into one hole
    # big enough for a carve neither could satisfy alone.
    mgr.destroy_mapping(second, GRANULARITY)
    mgr.destroy_mapping(first, GRANULARITY)
    merged = mgr.create_mapping(size=2 * GRANULARITY, tag="weights")
    assert merged == min(first, second)
    assert len(mgr.mappings) == 1


def test_oversized_allocation_gets_its_own_slab(manager):
    mgr = manager()
    big = mgr.create_mapping(size=SLAB * 2, tag="weights")

    assert len(mgr.mappings) == 1
    assert mgr.mappings[big].aligned_size == SLAB * 2
    # A slab whose single region spans it entirely starts with no holes; it
    # must still be recognised as empty on free.
    mgr.destroy_mapping(big, SLAB * 2)
    assert mgr.mappings == {}
    assert manager.vmm.reservations == {}


def test_tags_never_share_a_slab(manager):
    mgr = manager()
    mgr.create_mapping(size=GRANULARITY, tag="weights")
    mgr.create_mapping(size=GRANULARITY, tag="kv_cache")

    assert len(mgr.mappings) == 2
    tags = {m.tag for m in mgr.mappings.values()}
    assert tags == {"weights", "kv_cache"}


def test_unaligned_sizes_round_up_and_free_symmetrically(manager):
    mgr = manager()
    va = mgr.create_mapping(size=GRANULARITY + 1, tag="weights")
    other = mgr.create_mapping(size=1, tag="weights")

    # The carve is granularity-aligned, so the neighbour cannot overlap it.
    assert other >= va + 2 * GRANULARITY
    mgr.destroy_mapping(va, GRANULARITY + 1)
    mgr.destroy_mapping(other, 1)
    assert mgr.mappings == {}


def test_free_with_mismatched_size_is_rejected(manager):
    mgr = manager()
    va = mgr.create_mapping(size=GRANULARITY, tag="weights")
    with pytest.raises(RuntimeError, match="does not match"):
        mgr.destroy_mapping(va, 4 * GRANULARITY)


def test_disabled_slabbing_allocates_one_mapping_per_call(manager):
    mgr = manager(slab_size=0)
    vas = [mgr.create_mapping(size=GRANULARITY, tag="weights") for _ in range(3)]

    assert len(mgr.mappings) == 3
    assert all(va in mgr.mappings for va in vas)
    assert mgr.total_bytes == 3 * GRANULARITY


def test_dedicated_bypasses_slabs_while_packing_stays_on(manager):
    mgr = manager()
    packed = mgr.create_mapping(size=GRANULARITY, tag="weights")
    alone = mgr.create_mapping(size=GRANULARITY, tag="weights", dedicated=True)

    # The snapshot restore path needs its own allocation per manifest entry so
    # saved allocation-relative offsets replay unchanged: its mapping spans
    # exactly the request, not a slab.
    assert mgr.mappings[alone].aligned_size == GRANULARITY
    assert alone not in mgr._regions

    # The first carve of a slab lands on the slab base, so it shares that VA
    # with the slab's own mapping; packing continues in it regardless.
    assert packed in mgr._regions
    assert mgr.mappings[packed].aligned_size == SLAB
    assert mgr.create_mapping(size=GRANULARITY, tag="weights") == packed + GRANULARITY
    assert len(mgr.mappings) == 2


def test_freeing_the_first_carve_does_not_destroy_a_live_slab(manager):
    # The first carve of a slab lands on the slab's own base VA. Freeing it
    # while siblings are live must return only that region, not tear down the
    # slab underneath them.
    mgr = manager()
    first = mgr.create_mapping(size=GRANULARITY, tag="weights")
    second = mgr.create_mapping(size=GRANULARITY, tag="weights")
    assert first in mgr.mappings, "first carve shares the slab base VA"

    mgr.destroy_mapping(first, GRANULARITY)

    assert len(mgr.mappings) == 1, "slab must survive; a sibling is still live"
    assert mgr.owns(second)
    assert manager.vmm.mapped, "slab must still be mapped"
    assert mgr.total_bytes == GRANULARITY

    mgr.destroy_mapping(second, GRANULARITY)
    assert mgr.mappings == {}
    assert manager.vmm.reservations == {}


def test_failed_slab_grow_leaves_no_orphaned_state(manager, monkeypatch):
    mgr = manager()

    def boom(allocation_id):
        raise RuntimeError("export failed")

    monkeypatch.setattr(mgr, "export_handle", boom)
    with pytest.raises(RuntimeError, match="export failed"):
        mgr.create_mapping(size=GRANULARITY, tag="weights")

    assert mgr.mappings == {}
    assert manager.vmm.reservations == {}
    assert mgr._client.live == {}, "server allocation must be rolled back"
