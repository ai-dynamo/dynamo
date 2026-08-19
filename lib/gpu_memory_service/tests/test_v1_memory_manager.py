# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from contextlib import ExitStack

import pytest
from _deps import HAS_GMS

if not HAS_GMS:
    pytest.skip(
        "gpu_memory_service package is not available in this test image",
        allow_module_level=True,
    )

from _fake_vmm import FakeVMM
from gpu_memory_service.common.locks import GrantedLockType, RequestedLockType
from gpu_memory_service.v1 import device as device_identity
from gpu_memory_service.v1.client.memory_manager import GMSClientMemoryManager
from gpu_memory_service.v1.server.rpc import GMSRPCServer, GMSServerMemoryManager

pytestmark = [pytest.mark.pre_merge, pytest.mark.integration, pytest.mark.gpu_0]


def _stop(server: GMSRPCServer, thread: threading.Thread) -> None:
    server.shutdown()
    thread.join(timeout=5)
    assert not thread.is_alive()


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
        for domain, path in paths.items():
            manager = GMSServerMemoryManager("GPU-0", vmms[domain], 0)
            server = stack.enter_context(GMSRPCServer(path, manager))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            stack.callback(_stop, server, thread)

        identity_events = []
        physical_uuid = ["GPU-0"]
        cached_uuid = [None]

        def invalidate_device_uuid_cache():
            identity_events.append("invalidate")
            cached_uuid[0] = None

        def get_device_uuid(device):
            identity_events.append("lookup")
            if cached_uuid[0] is None:
                cached_uuid[0] = physical_uuid[0]
            return cached_uuid[0]

        monkeypatch.setattr(
            device_identity,
            "invalidate_device_uuid_cache",
            invalidate_device_uuid_cache,
        )
        monkeypatch.setattr(device_identity, "get_device_uuid", get_device_uuid)
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
        assert identity_events == ["invalidate", "lookup"] * 4

        weights.unmap_all_vas()
        weights.disconnect()
        physical_uuid[0] = "GPU-1"
        with pytest.raises(RuntimeError, match="sidecar is on another physical GPU"):
            weights.connect(RequestedLockType.RO)
        kv_cache.close()


def test_identity_mismatch_wins_when_session_close_fails(monkeypatch) -> None:
    class _Session:
        identity = ("nonce", "GPU-1")

        def close(self) -> None:
            raise ConnectionError("close failed")

    monkeypatch.setattr(device_identity, "invalidate_device_uuid_cache", lambda: None)
    monkeypatch.setattr(device_identity, "get_device_uuid", lambda _device: "GPU-0")
    manager = GMSClientMemoryManager(
        "/unused.sock",
        FakeVMM(granularity=64),
        0,
        session_factory=lambda *_args: _Session(),
    )

    with pytest.raises(RuntimeError, match="sidecar is on another physical GPU"):
        manager.connect(RequestedLockType.RO)


def test_latched_failure_raises_fresh_exceptions(monkeypatch) -> None:
    monkeypatch.setattr(device_identity, "invalidate_device_uuid_cache", lambda: None)
    monkeypatch.setattr(device_identity, "get_device_uuid", lambda _device: "GPU-0")

    def fail_session(*_args):
        raise ConnectionError("boom")

    manager = GMSClientMemoryManager(
        "/unused.sock",
        FakeVMM(granularity=64),
        0,
        session_factory=fail_session,
    )

    with pytest.raises(RuntimeError) as first:
        manager.connect(RequestedLockType.RO)
    with pytest.raises(RuntimeError) as second:
        manager.connect(RequestedLockType.RO)
    assert first.value is not second.value
    assert str(first.value) == str(second.value)


def _connected_client(tmp_path, monkeypatch, *, slab_size: int):
    path = str(tmp_path / "weights.sock")
    vmm = FakeVMM(granularity=64)
    server_manager = GMSServerMemoryManager("GPU-0", vmm, 0)
    server = GMSRPCServer(path, server_manager)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(device_identity, "invalidate_device_uuid_cache", lambda: None)
    monkeypatch.setattr(device_identity, "get_device_uuid", lambda _device: "GPU-0")
    client = GMSClientMemoryManager(path, vmm, 0, slab_size=slab_size)
    client.connect(RequestedLockType.RW)
    return client, vmm, server, thread


def _stop_client(client, server, thread) -> None:
    client.close()
    _stop(server, thread)


@pytest.mark.timeout(10)
def test_small_mappings_share_one_slab_and_reuse_freed_holes(
    tmp_path,
    monkeypatch,
) -> None:
    client, vmm, server, thread = _connected_client(tmp_path, monkeypatch, slab_size=256)
    try:
        first = client.create_mapping(65)
        second = client.create_mapping(65)
        assert second == first + 128
        assert len(client.mappings) == 1
        assert client.owns(first)
        assert client.owns(second)
        assert set(vmm.reservations) == {client.mappings[0].base}

        client.destroy_mapping(first, 65)
        reused = client.create_mapping(65)
        assert reused == first
        assert len(client.mappings) == 1

        client.unmap_all_vas()
        client.disconnect()
        client.connect(RequestedLockType.RW)
        client.reallocate_all_handles()
        client.remap_all_vas()
        assert len(client.mappings) == 1
        assert client.owns(reused)
        assert client.owns(second)
    finally:
        _stop_client(client, server, thread)


@pytest.mark.timeout(10)
def test_mapping_larger_than_slab_gets_its_own_reservation(
    tmp_path,
    monkeypatch,
) -> None:
    client, vmm, server, thread = _connected_client(tmp_path, monkeypatch, slab_size=256)
    try:
        small = client.create_mapping(65)
        large = client.create_mapping(320)
        assert len(client.mappings) == 2
        assert large != small
        assert set(vmm.reservations) == {mapping.base for mapping in client.mappings}
        client.destroy_mapping(large, 320)
        assert len(client.mappings) == 1
        assert client.owns(small)
    finally:
        _stop_client(client, server, thread)
