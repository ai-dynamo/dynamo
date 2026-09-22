# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from gpu_memory_service.common.locks import GrantedLockType

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.none,
    pytest.mark.gpu_0,
]


def test_gms_materialization_uses_vllm_preprocessed_weight_protocol(monkeypatch):
    from contextlib import contextmanager

    from gpu_memory_service.integrations.vllm import model_loader
    from vllm.model_executor import utils as model_executor_utils
    from vllm.model_executor.model_loader import utils as model_loader_utils

    events = []

    @contextmanager
    def already_processed():
        events.append("enter")
        yield
        events.append("exit")

    def process(model, model_config, target_device):
        events.append((model, model_config, target_device))

    monkeypatch.setattr(
        model_executor_utils,
        "weights_already_processed",
        already_processed,
    )
    monkeypatch.setattr(
        model_loader_utils,
        "process_weights_after_loading",
        process,
    )
    model = object()
    model_config = object()
    target_device = object()

    model_loader._process_weights_after_gms_materialization(
        model,
        model_config,
        target_device,
    )

    assert events == [
        "enter",
        (model, model_config, target_device),
        "exit",
    ]


def test_vllm_gms_early_device_resolution_matches_upstream_mapping(monkeypatch):
    from gpu_memory_service.integrations.vllm import worker as worker_module
    from vllm.platforms import interface

    installed = []
    monkeypatch.setattr(
        interface,
        "set_assigned_physical_gpu_ids",
        lambda ids: installed.append(list(ids)),
    )

    class Platform:
        @staticmethod
        def logical_device_id_to_visible_device_id(device_id):
            assert installed == [[5, 3]]
            assert device_id == 1
            return 0

    parallel_config = SimpleNamespace(
        distributed_executor_backend="mp",
        data_parallel_backend="mp",
        nnodes_within_dp=1,
        data_parallel_rank_local=1,
        data_parallel_index=1,
        pipeline_parallel_size=1,
        tensor_parallel_size=1,
        assigned_physical_gpu_ids=[5, 3],
    )

    assert (
        worker_module._resolve_gms_visible_device(0, parallel_config, Platform()) == 0
    )


def test_vllm_gms_model_loader_base_worker_reserves_ro_imported_weights(monkeypatch):
    from gpu_memory_service.integrations.vllm import model_loader
    from vllm.v1.worker.gpu_worker import Worker

    original = Worker.determine_available_memory
    calls = []

    def fake_determine_available_memory(self):
        calls.append(self.model_runner.model_memory_usage)
        return 42

    monkeypatch.setattr(
        Worker, "determine_available_memory", fake_determine_available_memory
    )
    monkeypatch.setattr(
        model_loader,
        "get_gms_client_memory_manager",
        lambda tag: SimpleNamespace(granted_lock_type=GrantedLockType.RO),
    )
    monkeypatch.setattr(model_loader, "get_imported_weights_bytes", lambda: 13)

    model_loader.patch_vllm_worker_memory_accounting()
    worker = SimpleNamespace(model_runner=SimpleNamespace(model_memory_usage=99))

    try:
        assert Worker.determine_available_memory(worker) == 29
        assert calls == [99]
        assert worker.model_runner.model_memory_usage == 99
    finally:
        Worker.determine_available_memory = original


def test_vllm_gms_model_loader_preserves_explicit_kv_capacity(monkeypatch):
    from gpu_memory_service.integrations.vllm import model_loader
    from vllm.v1.worker.gpu_worker import Worker

    original = Worker.determine_available_memory
    monkeypatch.setattr(Worker, "determine_available_memory", lambda self: 42)
    monkeypatch.setattr(
        model_loader,
        "get_gms_client_memory_manager",
        lambda tag: SimpleNamespace(granted_lock_type=GrantedLockType.RO),
    )
    monkeypatch.setattr(model_loader, "get_imported_weights_bytes", lambda: 13)
    model_loader.patch_vllm_worker_memory_accounting()
    worker = SimpleNamespace(cache_config=SimpleNamespace(kv_cache_memory_bytes=42))

    try:
        assert Worker.determine_available_memory(worker) == 42
    finally:
        Worker.determine_available_memory = original


@pytest.mark.parametrize(
    ("imported", "match"), [(0, "no imported"), (43, "no positive")]
)
def test_vllm_gms_model_loader_rejects_invalid_ro_capacity(
    monkeypatch, imported, match
):
    from gpu_memory_service.integrations.vllm import model_loader
    from vllm.v1.worker.gpu_worker import Worker

    original = Worker.determine_available_memory
    monkeypatch.setattr(Worker, "determine_available_memory", lambda self: 42)
    monkeypatch.setattr(
        model_loader,
        "get_gms_client_memory_manager",
        lambda tag: SimpleNamespace(granted_lock_type=GrantedLockType.RO),
    )
    monkeypatch.setattr(model_loader, "get_imported_weights_bytes", lambda: imported)
    model_loader.patch_vllm_worker_memory_accounting()

    try:
        with pytest.raises(RuntimeError, match=match):
            Worker.determine_available_memory(SimpleNamespace())
    finally:
        Worker.determine_available_memory = original


def test_vllm_gms_ro_snapshot_accounts_for_unclaimed_persistent_kv(monkeypatch):
    from gpu_memory_service.integrations.vllm import patches
    from vllm.utils.mem_utils import MemorySnapshot

    class KVManager:
        is_connected = True
        device = 0

        def list_persistent(self, engine_id=None, *, include_unclaimed=False):
            assert engine_id == "vllm-test"
            assert include_unclaimed is True
            return [SimpleNamespace(aligned_size=50)]

    managers = {
        "weights": SimpleNamespace(
            granted_lock_type=GrantedLockType.RO,
            list_handles=lambda: [SimpleNamespace(aligned_size=100)],
        ),
        "kv_pool": KVManager(),
    }
    monkeypatch.setattr(
        patches, "get_gms_client_memory_manager", lambda tag: managers[tag]
    )
    monkeypatch.setattr(patches, "allocation_engine_id", lambda _device: "vllm-test")
    monkeypatch.setattr(patches, "_memory_snapshot_patched", False)
    monkeypatch.setattr(
        MemorySnapshot, "measure", lambda snapshot: setattr(snapshot, "free_memory", 10)
    )

    patches.patch_memory_snapshot()
    snapshot = SimpleNamespace(device_=SimpleNamespace(index=0))
    MemorySnapshot.measure(snapshot)

    assert snapshot.free_memory == 160


def test_vllm_shared_snapshot_fails_closed_when_kv_inventory_fails(monkeypatch):
    from gpu_memory_service.integrations.vllm import patches
    from vllm.utils.mem_utils import MemorySnapshot

    class KVManager:
        is_connected = True
        device = 0

        @staticmethod
        def list_persistent(*args, **kwargs):
            raise ConnectionError("daemon unavailable")

    managers = {
        "weights": SimpleNamespace(
            granted_lock_type=GrantedLockType.RO,
            list_handles=lambda: [SimpleNamespace(aligned_size=100)],
        ),
        "kv_pool": KVManager(),
    }
    monkeypatch.setattr(
        patches, "get_gms_client_memory_manager", lambda tag: managers[tag]
    )
    monkeypatch.setattr(patches, "failover_hooks_required", lambda: True)
    monkeypatch.setattr(patches, "_memory_snapshot_patched", False)
    monkeypatch.setattr(
        MemorySnapshot, "measure", lambda snapshot: setattr(snapshot, "free_memory", 10)
    )

    patches.patch_memory_snapshot()
    with pytest.raises(RuntimeError, match="persistent KV accounting failed"):
        MemorySnapshot.measure(SimpleNamespace(device_=SimpleNamespace(index=0)))


def test_vllm_kv_disable_flag_uses_base_worker_paths(monkeypatch):
    from gpu_memory_service.integrations.vllm import worker as worker_module
    from vllm.v1.worker.gpu_worker import Worker

    sentinel = object()
    calls = []
    monkeypatch.setenv("GMS_VLLM_VMM_IPC_KV", "0")
    monkeypatch.setattr(
        Worker,
        "initialize_from_config",
        lambda _self, config: calls.append(config) or sentinel,
    )
    monkeypatch.setattr(
        Worker,
        "_maybe_get_memory_pool_context",
        lambda _self, tag: calls.append(tag) or sentinel,
    )
    instance = object.__new__(worker_module.GMSWorker)
    config = object()

    assert instance.initialize_from_config(config) is sentinel
    assert instance._maybe_get_memory_pool_context("kv_cache") is sentinel
    assert calls == [config, "kv_cache"]
    assert not hasattr(instance, "_gms_kv_manager")


def test_vllm_kv_disable_flag_skips_sleep_wake_kv_lifecycle(monkeypatch):
    from gpu_memory_service.integrations.vllm import worker as worker_module

    weights_manager = MagicMock(is_unmapped=False)
    requested_tags = []

    def get_manager(tag):
        requested_tags.append(tag)
        assert tag == "weights"
        return weights_manager

    device = MagicMock()
    device.mem_get_info.side_effect = [(100, 200), (150, 200)]
    monkeypatch.setenv("GMS_VLLM_VMM_IPC_KV", "0")
    monkeypatch.setattr(worker_module, "get_gms_client_memory_manager", get_manager)
    monkeypatch.setattr(worker_module, "get_mx_load_context", lambda: None)
    monkeypatch.setattr(worker_module, "torch_device", lambda: device)
    monkeypatch.setattr(worker_module.gc, "collect", lambda: None)

    instance = object.__new__(worker_module.GMSWorker)
    instance.sleep()

    weights_manager.unmap_all_vas.assert_called_once_with()
    weights_manager.abort.assert_called_once_with()
    assert requested_tags == ["weights"]

    instance.wake_up(tags=["kv_cache"])
    assert requested_tags == ["weights"]
