# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import contextmanager

import pytest

pytest.importorskip("gpu_memory_service", reason="gpu_memory_service is required")
torch = pytest.importorskip("torch", reason="torch is required")

import gpu_memory_service.integrations.sglang.memory_saver as gms_memory_saver  # noqa: E402
from gpu_memory_service.common.locks import (  # noqa: E402
    GrantedLockType,
    RequestedLockType,
)
from gpu_memory_service.integrations.sglang.memory_saver import (  # noqa: E402
    GMSMemorySaverImpl,
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.sglang,
]


class _FakeManager:
    def __init__(
        self,
        *,
        is_unmapped: bool = False,
        granted_lock_type: GrantedLockType | None = None,
    ):
        self.is_unmapped = is_unmapped
        self.granted_lock_type = granted_lock_type
        self.calls: list[object] = []

    def unmap_all_vas(self) -> None:
        self.calls.append("unmap_all_vas")
        self.is_unmapped = True

    def abort(self) -> None:
        self.calls.append("abort")
        self.granted_lock_type = None

    def connect(self, lock_type) -> None:
        self.calls.append(("connect", lock_type))
        self.granted_lock_type = GrantedLockType(lock_type.value)
        self.is_unmapped = False

    def reallocate_all_handles(self, *, tag: str) -> None:
        self.calls.append(("reallocate_all_handles", tag))

    def remap_all_vas(self) -> None:
        self.calls.append("remap_all_vas")
        self.is_unmapped = False


@pytest.fixture
def build_impl(monkeypatch, tmp_path):
    monkeypatch.setattr(
        gms_memory_saver,
        "get_socket_path",
        lambda device_index, tag: str(tmp_path / f"gms-test-{device_index}-{tag}.sock"),
    )

    def build(
        *,
        weights_lock: GrantedLockType = GrantedLockType.RW,
        kv_cache_lock: GrantedLockType = GrantedLockType.RW,
    ):
        weights = _FakeManager(granted_lock_type=weights_lock)
        kv_cache = _FakeManager(granted_lock_type=kv_cache_lock)
        pool_calls: list[tuple[str, torch.device]] = []

        @contextmanager
        def fake_use_mem_pool(tag: str, device: torch.device):
            pool_calls.append((tag, device))
            yield

        monkeypatch.setattr(
            gms_memory_saver,
            "get_or_create_gms_client_memory_manager",
            lambda socket_path, device, mode, tag: {
                "weights": weights,
                "kv_cache": kv_cache,
            }[tag],
        )
        monkeypatch.setattr(gms_memory_saver, "gms_use_mem_pool", fake_use_mem_pool)
        return (
            GMSMemorySaverImpl(device_index=0, mode=None),
            weights,
            kv_cache,
            pool_calls,
        )

    return build


@pytest.mark.parametrize(
    ("tag", "weights_lock", "expected_pool_calls"),
    [
        ("weights", GrantedLockType.RW, [("weights", torch.device("cuda", 0))]),
        ("weights", GrantedLockType.RO, []),
        ("kv_cache", GrantedLockType.RW, [("kv_cache", torch.device("cuda", 0))]),
        ("cuda_graph", GrantedLockType.RW, []),
    ],
)
def test_region_uses_gms_pool_only_for_rw_managed_tags(
    build_impl,
    tag,
    weights_lock,
    expected_pool_calls,
):
    impl, _, _, pool_calls = build_impl(
        weights_lock=weights_lock,
        kv_cache_lock=GrantedLockType.RW,
    )

    with impl.region(tag, enable_cpu_backup=False):
        pass

    assert pool_calls == expected_pool_calls


def test_pause_resume_routes_only_managed_tags(build_impl):
    impl, weights, kv_cache, _ = build_impl(
        weights_lock=GrantedLockType.RO,
        kv_cache_lock=GrantedLockType.RW,
    )

    impl.pause("model_weights")
    impl.resume("anything_else")

    impl.pause()
    impl.resume()

    assert weights.calls == [
        "unmap_all_vas",
        "abort",
        ("connect", RequestedLockType.RO),
        "remap_all_vas",
    ]
    assert kv_cache.calls == [
        "unmap_all_vas",
        "abort",
        ("connect", RequestedLockType.RW),
        ("reallocate_all_handles", "kv_cache"),
        "remap_all_vas",
    ]


@pytest.mark.parametrize("tag", ["weights", "kv_cache"])
def test_region_requires_rw_allocator(build_impl, tag):
    impl, _, _, _ = build_impl()
    impl.allocators[tag].abort()

    with pytest.raises(RuntimeError, match=rf"requires '{tag}' to be RW"):
        with impl.region(tag, enable_cpu_backup=False):
            pass
