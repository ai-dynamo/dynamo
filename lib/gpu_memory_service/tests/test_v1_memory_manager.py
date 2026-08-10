# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from contextlib import ExitStack

import pytest
from _fake_vmm import FakeVMM
from gpu_memory_service.common.locks import GrantedLockType, RequestedLockType
from gpu_memory_service.v1 import device as device_identity
from gpu_memory_service.v1.memory_manager import GMSClientMemoryManager
from gpu_memory_service.v1.server import GMSRPCServer, GMSServerMemoryManager
from gpu_memory_service.v1.session import _GMSClientSession

pytestmark = [pytest.mark.pre_merge, pytest.mark.integration, pytest.mark.gpu_0]


@pytest.mark.timeout(10)
def test_same_client_manager_preserves_weights_and_recreates_kv(
    tmp_path,
    monkeypatch,
) -> None:
    domains = ("weights", "kv_cache")
    physical_uuid = ["GPU-source"]
    cached_uuid = [None]

    def invalidate_device_uuid_cache():
        cached_uuid[0] = None

    def get_device_uuid(device):
        if cached_uuid[0] is None:
            cached_uuid[0] = physical_uuid[0]
        return cached_uuid[0]

    monkeypatch.setenv("GMS_SOCKET_DIR", str(tmp_path))
    monkeypatch.setattr(
        device_identity,
        "invalidate_device_uuid_cache",
        invalidate_device_uuid_cache,
    )
    monkeypatch.setattr(device_identity, "get_device_uuid", get_device_uuid)

    source_paths = {
        domain: device_identity.get_socket_path(0, domain) for domain in domains
    }
    client_vmms = {domain: FakeVMM(granularity=64) for domain in domains}
    source_vmms = {domain: FakeVMM(granularity=64) for domain in domains}
    with ExitStack() as stack:
        servers = {}
        threads = {}
        for domain, path in source_paths.items():
            manager = GMSServerMemoryManager("GPU-source", source_vmms[domain], 0)
            server = GMSRPCServer(path, manager)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            servers[f"source-{domain}"] = stack.enter_context(server)
            threads[f"source-{domain}"] = thread

        weights = GMSClientMemoryManager(
            source_paths["weights"], client_vmms["weights"], 0
        )
        kv_cache = GMSClientMemoryManager(
            source_paths["kv_cache"], client_vmms["kv_cache"], 0
        )
        weights.connect(RequestedLockType.RW)
        kv_cache.connect(RequestedLockType.RW)

        weights_va = weights.create_mapping(65)
        kv_va = kv_cache.create_mapping(33)
        weights_saved = [
            (mapping.allocation_id, mapping.base, mapping.aligned_size)
            for mapping in weights.mappings
        ]
        kv_saved = [
            (mapping.allocation_id, mapping.base) for mapping in kv_cache.mappings
        ]
        weights_handles = set(source_vmms["weights"].server_handles)
        weights.commit()

        with pytest.raises(RuntimeError, match="while connected"):
            weights.refresh_socket_path(str(tmp_path / "invalid.sock"))

        weights.unmap_all_vas()
        weights.disconnect()
        kv_cache.unmap_all_vas()
        kv_cache.disconnect()
        assert not source_vmms["kv_cache"].server_handles
        assert source_vmms["weights"].server_handles == weights_handles
        assert set(client_vmms["weights"].reservations) == {weights_va}
        assert set(client_vmms["kv_cache"].reservations) == {kv_va}

        physical_uuid[0] = "GPU-target"
        device_identity.invalidate_device_uuid_cache()
        target_paths = {
            domain: device_identity.get_socket_path(0, domain) for domain in domains
        }
        assert target_paths != source_paths

        target_vmms = {domain: FakeVMM(granularity=64) for domain in domains}
        for domain, path in target_paths.items():
            manager = GMSServerMemoryManager("GPU-target", target_vmms[domain], 0)
            server = GMSRPCServer(path, manager)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            servers[f"target-{domain}"] = stack.enter_context(server)
            threads[f"target-{domain}"] = thread

        target_weights_writer = _GMSClientSession(
            target_paths["weights"],
            RequestedLockType.RW,
        )
        for allocation_id, _base, aligned_size in weights_saved:
            target_weights_writer.allocate(allocation_id, aligned_size)
        target_weights_writer.commit()
        target_weights_writer.close()

        weights.refresh_socket_path(target_paths["weights"])
        kv_cache.refresh_socket_path(target_paths["kv_cache"])

        kv_cache.connect(RequestedLockType.RW)
        kv_cache.reallocate_all_handles()
        kv_cache.remap_all_vas()
        weights.connect(RequestedLockType.RO)
        weights.remap_all_vas()

        assert [
            (mapping.allocation_id, mapping.base, mapping.aligned_size)
            for mapping in weights.mappings
        ] == weights_saved
        assert [
            (mapping.allocation_id, mapping.base) for mapping in kv_cache.mappings
        ] == kv_saved
        assert target_vmms["weights"].server_handles
        assert target_vmms["kv_cache"].server_handles
        assert client_vmms["weights"].access == {weights_va: GrantedLockType.RO}
        assert client_vmms["kv_cache"].access == {kv_va: GrantedLockType.RW}

        weights.close()
        kv_cache.close()
        for server in servers.values():
            server.shutdown()
        for thread in threads.values():
            thread.join(timeout=10)
            assert not thread.is_alive()
