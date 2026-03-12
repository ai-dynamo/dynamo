# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
]


class _FakeManager:
    def __init__(self, *, is_unmapped: bool = False) -> None:
        self.is_unmapped = is_unmapped
        self.calls: list[object] = []

    def unmap_all_vas(self) -> None:
        self.calls.append("unmap_all_vas")
        self.is_unmapped = True

    def disconnect(self) -> None:
        self.calls.append("disconnect")

    def connect(self, lock_type) -> None:
        self.calls.append(("connect", lock_type.value))
        self.is_unmapped = False

    def reallocate_all_handles(self, *, tag: str) -> None:
        self.calls.append(("reallocate_all_handles", tag))

    def remap_all_vas(self) -> None:
        self.calls.append("remap_all_vas")
        self.is_unmapped = False


def test_initialize_from_config_uses_kv_cache_gms_scope(monkeypatch):
    from gpu_memory_service.integrations.vllm.worker import GMSWorker
    import gpu_memory_service.integrations.vllm.worker as worker_module
    import vllm.distributed.kv_transfer as kv_transfer

    create_calls: list[tuple[object, ...]] = []
    pool_calls: list[tuple[str, str]] = []
    kv_transfer_calls: list[object] = []
    kv_init_calls: list[object] = []

    @contextmanager
    def fake_use_mem_pool(scope, device):
        pool_calls.append((scope, str(device)))
        yield

    def fake_get_or_create(socket_path, device, mode, *, scope, tag, timeout_ms=None):
        create_calls.append((socket_path, device, mode.value, scope, tag, timeout_ms))
        return object()

    monkeypatch.setattr(worker_module, "gms_use_mem_pool", fake_use_mem_pool)
    monkeypatch.setattr(
        worker_module,
        "get_or_create_gms_client_memory_manager",
        fake_get_or_create,
    )
    monkeypatch.setattr(
        worker_module,
        "get_socket_path",
        lambda device, scope: f"/tmp/{scope}-{device}.sock",
    )
    monkeypatch.setattr(
        kv_transfer,
        "ensure_kv_transfer_initialized",
        lambda vllm_config, kv_cache_config: kv_transfer_calls.append(kv_cache_config),
    )

    worker = object.__new__(GMSWorker)
    worker.local_rank = 3
    worker.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(enable_sleep_mode=True)
    )
    worker.model_runner = SimpleNamespace(
        initialize_kv_cache=lambda kv_cache_config: kv_init_calls.append(
            kv_cache_config
        )
    )

    worker.initialize_from_config("kv-config")

    assert create_calls == [
        ("/tmp/kv_cache-3.sock", 3, "rw", "kv_cache", "kv_cache", None)
    ]
    assert pool_calls == [("kv_cache", "cuda:3")]
    assert kv_transfer_calls == ["kv-config"]
    assert kv_init_calls == ["kv-config"]


def test_sleep_level_2_unmaps_weights_and_kv_cache(monkeypatch):
    from gpu_memory_service.integrations.vllm.worker import GMSWorker
    import gpu_memory_service.integrations.vllm.worker as worker_module

    weights = _FakeManager()
    kv_cache = _FakeManager()

    monkeypatch.setattr(
        worker_module,
        "get_gms_client_memory_manager",
        lambda scope: weights if scope == "weights" else kv_cache,
    )
    monkeypatch.setattr(
        worker_module.torch.cuda,
        "mem_get_info",
        lambda: (2 << 30, 8 << 30),
    )

    worker = object.__new__(GMSWorker)
    worker.sleep(level=2)

    assert weights.calls == ["unmap_all_vas", "disconnect"]
    assert kv_cache.calls == ["unmap_all_vas", "disconnect"]


def test_wake_up_remaps_weights_and_reallocates_kv_cache(monkeypatch):
    from gpu_memory_service.integrations.vllm.worker import GMSWorker
    import gpu_memory_service.integrations.vllm.worker as worker_module

    weights = _FakeManager(is_unmapped=True)
    kv_cache = _FakeManager(is_unmapped=True)
    fp8_calls: list[str] = []

    monkeypatch.setattr(
        worker_module,
        "get_gms_client_memory_manager",
        lambda scope: weights if scope == "weights" else kv_cache,
    )

    worker = object.__new__(GMSWorker)
    worker.cache_config = SimpleNamespace(cache_dtype="fp8_e4m3")
    worker.model_runner = SimpleNamespace(
        init_fp8_kv_scales=lambda: fp8_calls.append("fp8")
    )

    worker.wake_up(["weights", "kv_cache"])

    assert weights.calls == [
        ("connect", "ro"),
        "remap_all_vas",
    ]
    assert kv_cache.calls == [
        ("connect", "rw"),
        ("reallocate_all_handles", "kv_cache"),
        "remap_all_vas",
    ]
    assert fp8_calls == ["fp8"]
