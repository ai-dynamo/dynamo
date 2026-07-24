# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading

import pytest
from _v1_fakes import V1FakeService, V1FakeVMM
from gpu_memory_service.common.locks import GrantedLockType
from gpu_memory_service.v1.client.memory_manager import SnapshotMemoryManager
from gpu_memory_service.v1.errors import FatalGMSError, GMSError
from gpu_memory_service.v1.server.allocations import AllocationStore

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def _manager(monkeypatch):
    vmm = V1FakeVMM()
    store = AllocationStore("GPU-0", vmm, 3)
    service = V1FakeService(store)
    monkeypatch.setattr(SnapshotMemoryManager, "_gpu_identity", lambda self: "GPU-0")
    return vmm, store, service, SnapshotMemoryManager(service, vmm, 3)


def test_sleep_wake_preserves_ids_and_reconnects_before_export(monkeypatch) -> None:
    vmm, store, service, manager = _manager(monkeypatch)
    first = manager.allocate(65)
    second = manager.allocate(33)
    before = {mapping.base: mapping.allocation_id for mapping in manager.mappings}
    handles = set(vmm.server_handles)

    manager.sleep()

    assert set(vmm.reservations) == {first, second}
    assert not vmm.mapped
    assert vmm.server_handles == handles
    assert not service.connected
    disconnect_index = service.events.index(("disconnect",))
    assert service.events[disconnect_index - 1][0] == "export"

    manager.wake()

    after = {mapping.base: mapping.allocation_id for mapping in manager.mappings}
    assert after == before
    assert set(vmm.mapped) == {first, second}
    assert set(vmm.access.values()) == {GrantedLockType.RO}
    reconnect_index = service.events.index(("reconnect",))
    assert all(event[0] == "export" for event in service.events[reconnect_index + 1 :])

    manager.retire()
    assert not vmm.server_handles
    assert not vmm.imports
    assert not vmm.reservations
    for allocation_id in after.values():
        with pytest.raises(Exception, match="unknown allocation"):
            store.export(allocation_id)


@pytest.mark.parametrize("failure", ["ro_transition", "sleep_unmap", "wake_map"])
def test_lifecycle_failures_are_atomic_and_fatal(monkeypatch, failure) -> None:
    vmm, _store, _service, manager = _manager(monkeypatch)
    first = manager.allocate(64)
    second = manager.allocate(64)

    if failure == "ro_transition":
        vmm.fail_access_call = vmm.access_calls + 2
    elif failure == "sleep_unmap":
        vmm.fail_unmap.add(second)
    else:
        manager.sleep()
        vmm.fail_map_call = vmm.map_calls + 2

    with pytest.raises(FatalGMSError) as first_failure:
        manager.wake() if failure == "wake_map" else manager.sleep()
    with pytest.raises(FatalGMSError) as replay:
        manager.retire()

    assert replay.value is first_failure.value
    attempted_unmaps = {event[1] for event in vmm.events if event[0] == "unmap"}
    assert {first, second} <= attempted_unmaps
    assert not vmm.server_handles
    assert not vmm.imports
    assert not vmm.reservations


def test_allocation_cannot_commit_during_sleep(monkeypatch) -> None:
    vmm, _store, _service, manager = _manager(monkeypatch)
    manager.allocate(64)

    synchronize_entered = threading.Event()
    finish_synchronize = threading.Event()
    original_synchronize = vmm.synchronize

    def blocked_synchronize():
        synchronize_entered.set()
        assert finish_synchronize.wait(10)
        original_synchronize()

    vmm.synchronize = blocked_synchronize
    failures: list[Exception] = []

    def sleep():
        try:
            manager.sleep()
        except Exception as exc:
            failures.append(exc)

    def allocate():
        try:
            manager.allocate(64)
        except Exception as exc:
            failures.append(exc)

    sleeping = threading.Thread(target=sleep)
    sleeping.start()
    assert synchronize_entered.wait(10)
    allocating = threading.Thread(target=allocate)
    allocating.start()
    finish_synchronize.set()
    sleeping.join(timeout=10)
    allocating.join(timeout=10)

    assert not sleeping.is_alive()
    assert not allocating.is_alive()
    assert len(failures) == 1
    assert isinstance(failures[0], GMSError)
    assert "not awake" in str(failures[0])

    manager.wake()
    assert set(vmm.mapped) == {manager.mappings[0].base}
