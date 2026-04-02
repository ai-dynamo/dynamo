# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.vllm,
]


class _FakeManager:
    def __init__(self, *, is_unmapped: bool = False) -> None:
        self.is_unmapped = is_unmapped
        self.calls: list[object] = []

    def unmap_all_vas(self) -> None:
        self.calls.append("unmap_all_vas")
        self.is_unmapped = True

    def abort(self) -> None:
        self.calls.append("abort")

    def connect(self, lock_type, timeout_ms=None) -> None:
        self.calls.append(("connect", lock_type.value))
        self.is_unmapped = False

    def reallocate_all_handles(self, *, tag: str) -> None:
        self.calls.append(("reallocate_all_handles", tag))

    def remap_all_vas(self) -> None:
        self.calls.append("remap_all_vas")
        self.is_unmapped = False


def test_initialize_from_config_uses_kv_cache_gms_tag(monkeypatch):
    import gpu_memory_service.integrations.vllm.worker as worker_module
    import vllm.distributed.kv_transfer as kv_transfer
    from gpu_memory_service.integrations.vllm.worker import GMSWorker

    create_calls: list[tuple[object, ...]] = []
    pool_calls: list[tuple[str, str]] = []
    kv_transfer_calls: list[object] = []
    kv_init_calls: list[object] = []

    @contextmanager
    def fake_use_mem_pool(tag, device):
        pool_calls.append((tag, str(device)))
        yield

    def fake_get_or_create(socket_path, device, mode, *, tag, timeout_ms=None):
        create_calls.append((socket_path, device, mode.value, tag, timeout_ms))
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
        lambda device, tag: f"/tmp/{tag}-{device}.sock",
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

    assert create_calls == [("/tmp/kv_cache-3.sock", 3, "rw", "kv_cache", None)]
    assert pool_calls == [("kv_cache", "cuda:3")]
    assert kv_transfer_calls == ["kv-config"]
    assert kv_init_calls == ["kv-config"]


def test_sleep_level_2_unmaps_weights_and_kv_cache(monkeypatch):
    import gpu_memory_service.integrations.vllm.worker as worker_module
    from gpu_memory_service.integrations.vllm.worker import GMSWorker

    weights = _FakeManager()
    kv_cache = _FakeManager()

    monkeypatch.setattr(
        worker_module,
        "get_gms_client_memory_manager",
        lambda tag: weights if tag == "weights" else kv_cache,
    )
    monkeypatch.setattr(
        worker_module.torch.cuda,
        "mem_get_info",
        lambda: (2 << 30, 8 << 30),
    )

    worker = object.__new__(GMSWorker)
    worker.sleep(level=2)

    assert weights.calls == ["unmap_all_vas", "abort"]
    assert kv_cache.calls == ["unmap_all_vas", "abort"]


def test_wake_up_remaps_weights_and_reallocates_kv_cache(monkeypatch):
    import gpu_memory_service.integrations.vllm.worker as worker_module
    from gpu_memory_service.integrations.vllm.worker import GMSWorker

    weights = _FakeManager(is_unmapped=True)
    kv_cache = _FakeManager(is_unmapped=True)
    fp8_calls: list[str] = []

    monkeypatch.setattr(
        worker_module,
        "get_gms_client_memory_manager",
        lambda tag: weights if tag == "weights" else kv_cache,
    )

    worker = object.__new__(GMSWorker)
    worker.local_rank = 0
    worker.cache_config = SimpleNamespace(cache_dtype="fp8_e4m3")
    worker.model_runner = SimpleNamespace(
        kv_caches={"layer_0": True},
        init_fp8_kv_scales=lambda: fp8_calls.append("fp8"),
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


def test_maybe_get_memory_pool_context_routes_tags(monkeypatch):
    import gpu_memory_service.integrations.vllm.worker as worker_module
    from gpu_memory_service.integrations.vllm.worker import GMSWorker, Worker

    kv_cache_context = object()
    super_calls: list[str] = []
    mem_pool_calls: list[tuple[str, str]] = []

    def fake_use_mem_pool(tag, device):
        mem_pool_calls.append((tag, str(device)))
        return kv_cache_context

    def fake_super_context(self, tag):
        del self
        super_calls.append(tag)
        return f"super:{tag}"

    monkeypatch.setattr(worker_module, "gms_use_mem_pool", fake_use_mem_pool)
    monkeypatch.setattr(Worker, "_maybe_get_memory_pool_context", fake_super_context)

    worker = object.__new__(GMSWorker)
    worker.local_rank = 2

    weights_context = worker._maybe_get_memory_pool_context("weights")
    with weights_context:
        pass
    assert mem_pool_calls == []
    assert super_calls == []

    assert worker._maybe_get_memory_pool_context("kv_cache") is kv_cache_context
    assert mem_pool_calls == [("kv_cache", "cuda:2")]
    assert super_calls == []

    assert worker._maybe_get_memory_pool_context("other") == "super:other"
    assert super_calls == ["other"]
