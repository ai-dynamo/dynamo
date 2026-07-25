# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

import pytest
from _v1_fakes import V1FakeSessionFactory, V1FakeVMM
from gpu_memory_service.common.locks import GrantedLockType, RequestedLockType
from gpu_memory_service.core.client.memory_manager import LocalMapping, install_mapping
from gpu_memory_service.core.errors import FatalGMSError, GMSError
from gpu_memory_service.core.server.allocations import GMSAllocationManager
from gpu_memory_service.core.server.gms import GMS
from gpu_memory_service.v1.memory_manager import SnapshotMemoryManager

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def test_install_mapping_releases_import_when_map_fails() -> None:
    vmm = V1FakeVMM()
    vmm.fail_map_call = 1
    mapping = LocalMapping("allocation-1", 64, 64, 0x100000, 64)

    with pytest.raises(RuntimeError, match="map failed"):
        install_mapping(
            vmm,
            mapping,
            os.open("/dev/null", os.O_RDONLY),
            0,
            GrantedLockType.RW,
        )

    assert not vmm.imports


def test_commit_sleep_wake_holds_ro_and_preserves_ids_and_vas(monkeypatch) -> None:
    vmm = V1FakeVMM()
    allocations = GMSAllocationManager(vmm, 3)
    gms = GMS("GPU-0", allocations)
    events = vmm.events
    sessions = V1FakeSessionFactory(gms, events)
    monkeypatch.setattr(
        SnapshotMemoryManager,
        "_gpu_identity",
        lambda self: "GPU-0",
    )
    manager = SnapshotMemoryManager(
        "unused",
        vmm,
        3,
        session_factory=sessions,
    )
    first = manager.allocate(65)
    second = manager.allocate(33)
    before = {mapping.base: mapping.allocation_id for mapping in manager.mappings}
    handles = set(vmm.server_handles)

    manager.commit()

    assert sessions.sessions[0].is_connected
    assert sessions.sessions[0].lock_type == GrantedLockType.RO
    assert gms.snapshot().ro_session_count == 1
    assert set(vmm.mapped) == {first, second}
    assert set(vmm.access.values()) == {GrantedLockType.RO}
    assert events.index(("commit",)) > max(
        index
        for index, event in enumerate(events)
        if event[-1:] == (GrantedLockType.RO,) and event[0] == "access"
    )

    manager.sleep()

    close_index = events.index(("close", GrantedLockType.RO))
    assert close_index > max(
        index for index, event in enumerate(events) if event[0] == "unmap"
    )
    assert gms.snapshot().ro_session_count == 0
    assert set(vmm.reservations) == {first, second}
    assert not vmm.mapped
    assert vmm.server_handles == handles

    manager.wake()

    assert events[close_index + 1] == ("handshake", RequestedLockType.RO)
    assert {
        mapping.base: mapping.allocation_id for mapping in manager.mappings
    } == before
    assert set(vmm.mapped) == {first, second}
    assert set(vmm.access.values()) == {GrantedLockType.RO}
    assert sessions.sessions[-1].is_connected
    assert gms.snapshot().ro_session_count == 1

    manager.retire()
    assert allocations.allocation_count == 2
    assert vmm.server_handles == handles
    assert not vmm.imports
    assert not vmm.reservations
    assert gms.snapshot().ro_session_count == 0

    replacement = sessions("unused", RequestedLockType.RW, None)
    assert allocations.allocation_count == 0
    assert not vmm.server_handles
    with pytest.raises(GMSError, match="unknown allocation ID"):
        replacement.export(next(iter(before.values())))
    replacement.close()


def test_wake_rejects_another_server_incarnation(monkeypatch) -> None:
    vmm = V1FakeVMM()
    allocations = GMSAllocationManager(vmm, 0)
    gms = GMS("GPU-0", allocations)
    sessions = V1FakeSessionFactory(gms, vmm.events)
    monkeypatch.setattr(
        SnapshotMemoryManager,
        "_gpu_identity",
        lambda self: "GPU-0",
    )
    manager = SnapshotMemoryManager(
        "unused",
        vmm,
        0,
        session_factory=sessions,
    )
    manager.allocate(64)
    manager.commit()
    manager.sleep()

    replacement = GMS("GPU-0", GMSAllocationManager(vmm, 0))
    sessions.gms = replacement
    with pytest.raises(FatalGMSError, match="incarnation"):
        manager.wake()

    assert replacement.snapshot().ro_session_count == 0
    assert not vmm.imports
    assert allocations.allocation_count == 1
