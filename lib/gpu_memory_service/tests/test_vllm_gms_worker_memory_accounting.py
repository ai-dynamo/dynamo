# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from gpu_memory_service.common.locks import GrantedLockType

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.none,
    pytest.mark.gpu_0,
]


def _make_worker(worker_module, usage: int):
    worker = object.__new__(worker_module.GMSWorker)
    worker.model_runner = SimpleNamespace(model_memory_usage=usage)
    return worker


def test_vllm_gms_rw_weight_memory_not_double_counted(monkeypatch):
    from gpu_memory_service.integrations.vllm import worker as worker_module

    calls = []
    worker = _make_worker(worker_module, 1234)
    monkeypatch.setattr(
        worker_module,
        "get_gms_client_memory_manager",
        lambda tag: SimpleNamespace(granted_lock_type=GrantedLockType.RW),
    )

    def fake_determine_available_memory(self):
        calls.append(self.model_runner.model_memory_usage)
        return 5678

    monkeypatch.setattr(
        worker_module.Worker,
        "determine_available_memory",
        fake_determine_available_memory,
    )

    assert worker._determine_available_memory_with_gms_weight_accounting() == 5678
    assert calls == [0]
    assert worker.model_runner.model_memory_usage == 1234


def test_vllm_gms_ro_weight_memory_preserved(monkeypatch):
    from gpu_memory_service.integrations.vllm import worker as worker_module

    calls = []
    worker = _make_worker(worker_module, 1234)
    monkeypatch.setattr(
        worker_module,
        "get_gms_client_memory_manager",
        lambda tag: SimpleNamespace(granted_lock_type=GrantedLockType.RO),
    )

    def fake_determine_available_memory(self):
        calls.append(self.model_runner.model_memory_usage)
        return 5678

    monkeypatch.setattr(
        worker_module.Worker,
        "determine_available_memory",
        fake_determine_available_memory,
    )

    assert worker._determine_available_memory_with_gms_weight_accounting() == 5678
    assert calls == [1234]
    assert worker.model_runner.model_memory_usage == 1234


def test_vllm_gms_model_loader_patches_base_worker_memory_accounting(monkeypatch):
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
        lambda tag: SimpleNamespace(granted_lock_type=GrantedLockType.RW),
    )

    model_loader.patch_vllm_worker_memory_accounting()
    worker = SimpleNamespace(model_runner=SimpleNamespace(model_memory_usage=99))

    try:
        assert Worker.determine_available_memory(worker) == 42
        assert calls == [0]
        assert worker.model_runner.model_memory_usage == 99
    finally:
        Worker.determine_available_memory = original


def test_vllm_gms_model_loader_base_worker_patch_preserves_ro(monkeypatch):
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

    model_loader.patch_vllm_worker_memory_accounting()
    worker = SimpleNamespace(model_runner=SimpleNamespace(model_memory_usage=99))

    try:
        assert Worker.determine_available_memory(worker) == 42
        assert calls == [99]
        assert worker.model_runner.model_memory_usage == 99
    finally:
        Worker.determine_available_memory = original
