# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from contextlib import ExitStack

import pytest
from _fake_vmm import FakeVMM
from gpu_memory_service.common.locks import GrantedLockType, RequestedLockType
from gpu_memory_service.core.server.gms import GMSServerMemoryManager
from gpu_memory_service.core.server.rpc import GMSRPCServer
from gpu_memory_service.v1.memory_manager import GMSClientMemoryManager

pytestmark = [pytest.mark.pre_merge, pytest.mark.integration, pytest.mark.gpu_0]


def test_latch_preserves_operational_failure_when_disconnect_fails() -> None:
    class FailingSession:
        lock_type = GrantedLockType.RW

        def allocate(self, allocation_id: str, aligned_size: int) -> None:
            raise RuntimeError("allocation failed")

        def close(self) -> None:
            raise RuntimeError("close failed")

    manager = GMSClientMemoryManager("unused", FakeVMM(granularity=64), 0)
    manager._session = FailingSession()

    with pytest.raises(
        RuntimeError,
        match="GMS mapping creation failed: allocation failed",
    ) as original:
        manager.create_mapping(1)

    assert manager._session is None
    with pytest.raises(RuntimeError) as latched:
        manager.connect(RequestedLockType.RW)
    assert latched.value is original.value


@pytest.mark.timeout(10)
def test_same_client_manager_preserves_weights_and_recreates_kv(
    tmp_path,
    monkeypatch,
) -> None:
    paths = {
        domain: str(tmp_path / f"{domain}.sock") for domain in ("weights", "kv_cache")
    }
    vmms = {domain: FakeVMM(granularity=64) for domain in paths}
    with ExitStack() as stack:
        servers = {}
        threads = {}
        for domain, path in paths.items():
            manager = GMSServerMemoryManager("GPU-0", vmms[domain], 0)
            server = GMSRPCServer(path, manager)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            servers[domain] = stack.enter_context(server)
            threads[domain] = thread

        monkeypatch.setattr(
            GMSClientMemoryManager,
            "_gpu_identity",
            lambda self: "GPU-0",
        )
        weights = GMSClientMemoryManager(paths["weights"], vmms["weights"], 0)
        kv_cache = GMSClientMemoryManager(paths["kv_cache"], vmms["kv_cache"], 0)
        weights.connect(RequestedLockType.RW)
        kv_cache.connect(RequestedLockType.RW)

        weights_va = weights.create_mapping(65)
        kv_va = kv_cache.create_mapping(33)
        weights_saved = [
            (mapping.allocation_id, mapping.base) for mapping in weights.mappings
        ]
        kv_saved = [
            (mapping.allocation_id, mapping.base) for mapping in kv_cache.mappings
        ]
        weights_handles = set(vmms["weights"].server_handles)
        old_kv_handles = set(vmms["kv_cache"].server_handles)
        weights.commit()

        weights.unmap_all_vas()
        weights.disconnect()
        kv_cache.unmap_all_vas()
        kv_cache.disconnect()
        assert not vmms["kv_cache"].server_handles
        assert vmms["weights"].server_handles == weights_handles
        assert set(vmms["weights"].reservations) == {weights_va}
        assert set(vmms["kv_cache"].reservations) == {kv_va}

        kv_cache.connect(RequestedLockType.RW)
        kv_cache.reallocate_all_handles()
        kv_cache.remap_all_vas()
        weights.connect(RequestedLockType.RO)
        weights.remap_all_vas()

        assert [
            (mapping.allocation_id, mapping.base) for mapping in weights.mappings
        ] == weights_saved
        assert [
            (mapping.allocation_id, mapping.base) for mapping in kv_cache.mappings
        ] == kv_saved
        assert vmms["weights"].server_handles == weights_handles
        assert vmms["kv_cache"].server_handles.isdisjoint(old_kv_handles)
        assert vmms["weights"].access == {weights_va: GrantedLockType.RO}
        assert vmms["kv_cache"].access == {kv_va: GrantedLockType.RW}

        weights.unmap_all_vas()
        weights.disconnect()
        kv_cache.close()
        for server in servers.values():
            server.shutdown()
        for thread in threads.values():
            thread.join(timeout=10)
            assert not thread.is_alive()
