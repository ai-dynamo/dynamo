# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import contextmanager

import pytest
import torch

from gpu_memory_service.common.types import RequestedLockType
from gpu_memory_service.integrations.sglang.memory_saver import GMSMemorySaverImpl

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.fault_tolerance,
]


class _FakeManager:
    def __init__(self, *, is_unmapped: bool = False):
        self.is_unmapped = is_unmapped
        self.calls: list[object] = []

    def unmap_all_vas(self) -> None:
        self.calls.append("unmap_all_vas")
        self.is_unmapped = True

    def disconnect(self) -> None:
        self.calls.append("disconnect")

    def connect(self, lock_type) -> None:
        self.calls.append(("connect", lock_type))
        self.is_unmapped = False

    def reallocate_all_handles(self, *, tag: str) -> None:
        self.calls.append(("reallocate_all_handles", tag))

    def remap_all_vas(self) -> None:
        self.calls.append("remap_all_vas")
        self.is_unmapped = False


class _FakeTorchImpl:
    def __init__(self):
        self.region_calls: list[tuple[str, bool]] = []
        self.pause_calls: list[object] = []
        self.resume_calls: list[object] = []

    @contextmanager
    def region(self, tag: str, enable_cpu_backup: bool):
        self.region_calls.append((tag, enable_cpu_backup))
        yield

    def pause(self, tag=None) -> None:
        self.pause_calls.append(tag)

    def resume(self, tag=None) -> None:
        self.resume_calls.append(tag)


def test_region_routes_weights_and_kv_cache_to_gms(monkeypatch):
    weights = _FakeManager()
    kv_cache = _FakeManager()
    torch_impl = _FakeTorchImpl()
    pool_calls: list[tuple[str, torch.device]] = []

    @contextmanager
    def fake_use_mem_pool(scope: str, device: torch.device):
        pool_calls.append((scope, device))
        yield

    monkeypatch.setattr(
        GMSMemorySaverImpl,
        "_init_allocators",
        lambda self: (weights, kv_cache, "write"),
    )
    monkeypatch.setattr(
        "gpu_memory_service.integrations.sglang.memory_saver.gms_use_mem_pool",
        fake_use_mem_pool,
    )

    impl = GMSMemorySaverImpl(torch_impl=torch_impl, device_index=2, mode=None)

    with impl.region("weights", enable_cpu_backup=False):
        pass
    with impl.region("kv_cache", enable_cpu_backup=False):
        pass
    with impl.region("cuda_graph", enable_cpu_backup=True):
        pass

    assert pool_calls == [
        ("weights", torch.device("cuda", 2)),
        ("kv_cache", torch.device("cuda", 2)),
    ]
    assert torch_impl.region_calls == [("cuda_graph", True)]


def test_pause_resume_routes_kv_cache_to_gms(monkeypatch):
    weights = _FakeManager()
    kv_cache = _FakeManager()
    torch_impl = _FakeTorchImpl()

    monkeypatch.setattr(
        GMSMemorySaverImpl,
        "_init_allocators",
        lambda self: (weights, kv_cache, "read"),
    )

    impl = GMSMemorySaverImpl(torch_impl=torch_impl, device_index=0, mode=None)

    impl.pause()
    impl.resume()

    assert weights.calls == [
        "unmap_all_vas",
        "disconnect",
        ("connect", RequestedLockType.RO),
        "remap_all_vas",
    ]
    assert kv_cache.calls == [
        "unmap_all_vas",
        "disconnect",
        ("connect", RequestedLockType.RW),
        ("reallocate_all_handles", "kv_cache"),
        "remap_all_vas",
    ]
    assert torch_impl.pause_calls == [None]
    assert torch_impl.resume_calls == [None]
