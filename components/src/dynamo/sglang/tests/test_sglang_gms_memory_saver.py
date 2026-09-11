# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: E402
from __future__ import annotations

import sys
from contextlib import contextmanager, nullcontext
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("gpu_memory_service", reason="gpu_memory_service is required")
torch = pytest.importorskip("torch", reason="torch is required")

import gpu_memory_service.integrations.sglang as gms_sglang
import gpu_memory_service.integrations.sglang.memory_saver as gms_memory_saver
from gpu_memory_service.common.locks import GrantedLockType, RequestedLockType
from gpu_memory_service.integrations.sglang.memory_saver import GMSMemorySaverImpl

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.sglang,
    pytest.mark.core,
]


@pytest.fixture
def fake_gms_model_loader(monkeypatch):
    model_loader = ModuleType("gpu_memory_service.integrations.sglang.model_loader")
    model_loader.GMSModelLoader = object
    monkeypatch.setitem(
        sys.modules,
        "gpu_memory_service.integrations.sglang.model_loader",
        model_loader,
    )
    monkeypatch.setattr(gms_sglang, "_gms_initialized", False)
    monkeypatch.setattr(gms_sglang, "_gms_lock_mode", None)
    monkeypatch.setattr(gms_sglang, "_gms_ro_connect_timeout_ms", None)


def test_setup_gms_declares_memory_saver(monkeypatch, fake_gms_model_loader):
    class ReadOnlyServerArgs:
        def __setattr__(self, name, value):
            raise AssertionError(f"unexpected direct assignment: {name}={value!r}")

    server_args = ReadOnlyServerArgs()
    declare_late_resolution = Mock()
    monkeypatch.setattr(gms_sglang, "declare_late_resolution", declare_late_resolution)

    loader = gms_sglang.setup_gms(server_args)

    declare_late_resolution.assert_called_once_with(
        server_args,
        "dynamo.gms",
        enable_memory_saver=True,
    )
    assert loader is object
    assert gms_sglang.is_gms_active()


def test_setup_gms_uses_override_when_declaration_is_unavailable(
    monkeypatch,
    fake_gms_model_loader,
):
    override = Mock()
    server_args = SimpleNamespace(override=override)
    monkeypatch.setattr(gms_sglang, "declare_late_resolution", None)

    gms_sglang.setup_gms(server_args)

    override.assert_called_once_with("dynamo.gms", enable_memory_saver=True)


def test_setup_gms_assigns_memory_saver_for_legacy_args(
    monkeypatch,
    fake_gms_model_loader,
):
    server_args = SimpleNamespace(enable_memory_saver=False)
    monkeypatch.setattr(gms_sglang, "declare_late_resolution", None)

    gms_sglang.setup_gms(server_args)

    assert server_args.enable_memory_saver is True


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

    def connect(self, lock_type, timeout_ms=None) -> None:
        self.calls.append(("connect", lock_type, timeout_ms))
        self.granted_lock_type = GrantedLockType(lock_type.value)
        self.is_unmapped = False

    def reallocate_all_handles(self, *, tag: str) -> None:
        self.calls.append(("reallocate_all_handles", tag))

    def remap_persistent_vas(
        self,
        engine_id: str,
        *,
        shared: bool,
        synchronize_per_mapping: bool,
    ) -> None:
        self.calls.append(
            (
                "remap_persistent_vas",
                engine_id,
                shared,
                synchronize_per_mapping,
            )
        )
        self.is_unmapped = False

    def remap_all_vas(self) -> None:
        self.calls.append("remap_all_vas")
        self.is_unmapped = False

    def list_persistent(self, engine_id=None, *, include_unclaimed=False):
        self.calls.append(("list_persistent", engine_id, include_unclaimed))
        return []


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
        kv_cache_lock: GrantedLockType = GrantedLockType.RW_PERSISTENT,
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
            lambda socket_path, device, mode, tag: weights,
        )
        monkeypatch.setattr(
            gms_memory_saver,
            "get_or_create_persistent_allocator",
            lambda socket_path, device, engine_id, tag, shared: kv_cache,
        )
        monkeypatch.setattr(
            gms_memory_saver,
            "get_gms_persistent_kv_socket",
            lambda device, env_name: str(tmp_path / f"gms-kv-{device}.sock"),
        )
        monkeypatch.setattr(
            gms_memory_saver, "allocation_engine_id", lambda device: "sglang-test"
        )
        monkeypatch.setattr(gms_memory_saver, "allocation_shared", lambda: True)
        monkeypatch.setattr(
            gms_memory_saver, "allocator_tag", lambda device: f"kv_pool:cuda{device}"
        )
        monkeypatch.setattr(gms_memory_saver, "gms_use_mem_pool", fake_use_mem_pool)
        monkeypatch.setattr(
            gms_memory_saver,
            "gms_use_persistent_pool",
            fake_use_mem_pool,
        )
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
        (
            "kv_cache",
            GrantedLockType.RW_PERSISTENT,
            [("kv_pool:cuda0", torch.device("cuda", 0))],
        ),
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
        kv_cache_lock=GrantedLockType.RW_PERSISTENT,
    )

    scope = (
        gms_memory_saver.persistent_kv_pool_scope(reattaching=False)
        if tag == "kv_cache"
        else nullcontext()
    )
    with scope, impl.region(tag, enable_cpu_backup=False):
        pass

    assert pool_calls == expected_pool_calls


def test_unscoped_kv_region_keeps_scheduler_metadata_process_local(build_impl):
    impl, _, _, pool_calls = build_impl()

    with impl.region("kv_cache", enable_cpu_backup=False):
        pass

    assert pool_calls == []


def test_reattach_kv_region_replaces_zero_fill_only_inside_scope(
    build_impl, monkeypatch
):
    impl, _, _, _ = build_impl()
    zeros = Mock(return_value="zero")
    empty = Mock(return_value="empty")
    monkeypatch.setattr(torch, "zeros", zeros)
    monkeypatch.setattr(torch, "empty", empty)

    with gms_memory_saver.persistent_kv_pool_scope(reattaching=True):
        with impl.region("kv_cache", enable_cpu_backup=False):
            assert torch.zeros((2,), device="cuda") == "empty"
        assert torch.zeros((2,), device="cuda") == "zero"

    assert empty.call_count == 1


def test_region_accepts_current_torch_memory_saver_protocol(build_impl):
    impl, _, _, pool_calls = build_impl()

    with impl.region(
        "weights",
        enable_cpu_backup=False,
        enable_disk_backup=False,
        cpu_backup_backend=None,
    ):
        pass

    assert pool_calls == [("weights", torch.device("cuda", 0))]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"enable_cpu_backup": True},
        {"enable_cpu_backup": False, "enable_disk_backup": True},
        {"enable_cpu_backup": False, "cpu_backup_backend": "pinned"},
    ],
)
def test_region_rejects_unsupported_backup_modes(build_impl, kwargs):
    impl, _, _, _ = build_impl()

    with (
        pytest.raises(ValueError, match="does not support"),
        impl.region("weights", **kwargs),
    ):
        pass


def test_pause_resume_routes_only_managed_tags(build_impl):
    impl, weights, kv_cache, _ = build_impl(
        weights_lock=GrantedLockType.RO,
        kv_cache_lock=GrantedLockType.RW_PERSISTENT,
    )

    impl.pause("model_weights")
    impl.resume("anything_else")

    impl.pause()
    impl.resume()

    assert weights.calls == [
        "unmap_all_vas",
        "abort",
        ("connect", RequestedLockType.RO, None),
        "remap_all_vas",
    ]
    assert kv_cache.calls == [
        "unmap_all_vas",
        "abort",
        ("connect", RequestedLockType.RW_PERSISTENT, None),
        ("remap_persistent_vas", "sglang-test", True, False),
    ]


def test_region_requires_rw_weights_allocator(build_impl):
    impl, _, _, _ = build_impl()
    impl.allocators["weights"].abort()

    with (
        pytest.raises(RuntimeError, match="requires 'weights' to be RW"),
        impl.region("weights", enable_cpu_backup=False),
    ):
        pass


def test_region_requires_rw_persistent_kv_allocator(build_impl):
    impl, _, _, _ = build_impl(kv_cache_lock=GrantedLockType.RW)

    with (
        pytest.raises(RuntimeError, match="requires 'kv_cache' to be RW_PERSISTENT"),
        gms_memory_saver.persistent_kv_pool_scope(reattaching=False),
        impl.region("kv_cache", enable_cpu_backup=False),
    ):
        pass


def test_write_publication_follows_outermost_region_and_switches_weights_ro(
    build_impl, monkeypatch
):
    impl, weights, _, _ = build_impl()
    model = torch.nn.Module()
    impl.preloaded_weights_bytes = 456
    finalized_models = []
    events = []

    @contextmanager
    def traced_pool(tag, device):
        events.append(f"pool_enter:{tag}")
        try:
            yield
        finally:
            events.append(f"pool_exit:{tag}")

    def finalize(allocator, pending_model):
        events.append("finalize")
        finalized_models.append(pending_model)
        allocator.granted_lock_type = GrantedLockType.RO
        return SimpleNamespace(committed_bytes=123)

    monkeypatch.setattr(gms_memory_saver, "gms_use_mem_pool", traced_pool)
    monkeypatch.setattr(gms_memory_saver, "gms_use_persistent_pool", traced_pool)
    monkeypatch.setattr(gms_memory_saver, "finalize_gms_write", finalize)

    with (
        impl.region("weights", enable_cpu_backup=False),
        gms_memory_saver.persistent_kv_pool_scope(reattaching=False),
        impl.region("kv_cache", enable_cpu_backup=False),
    ):
        impl.finalize_write_mode(model)
        events.append("defer")

    assert events == [
        "pool_enter:weights",
        "pool_enter:kv_pool:cuda0",
        "defer",
        "pool_exit:kv_pool:cuda0",
        "pool_exit:weights",
        "finalize",
    ]
    assert finalized_models == [model]
    assert weights.granted_lock_type == GrantedLockType.RO
    assert impl.imported_weights_bytes == 123
    assert impl.preloaded_weights_bytes == 0

    with impl.region("weights", enable_cpu_backup=False):
        impl.finalize_write_mode(torch.nn.Module())

    assert events[-1] == "finalize"
    assert finalized_models == [model]


@pytest.mark.parametrize("failure_source", ["body", "pool_exit"])
def test_nested_region_failure_discards_pending_publication(
    build_impl, monkeypatch, failure_source
):
    impl, weights, _, _ = build_impl()
    stale_model = torch.nn.Module()
    fresh_model = torch.nn.Module()
    finalize = Mock(return_value=SimpleNamespace(committed_bytes=1))
    monkeypatch.setattr(gms_memory_saver, "finalize_gms_write", finalize)

    @contextmanager
    def maybe_failing_pool(tag, device):
        yield
        if failure_source == "pool_exit" and tag == "kv_pool:cuda0":
            raise ValueError("pool exit failed")

    monkeypatch.setattr(gms_memory_saver, "gms_use_mem_pool", maybe_failing_pool)
    monkeypatch.setattr(
        gms_memory_saver,
        "gms_use_persistent_pool",
        maybe_failing_pool,
    )

    failure_message = failure_source.replace("_", " ")
    with impl.region("weights", enable_cpu_backup=False):
        impl.finalize_write_mode(stale_model)
        with (
            pytest.raises(ValueError, match=f"{failure_message} failed"),
            gms_memory_saver.persistent_kv_pool_scope(reattaching=False),
            impl.region("kv_cache", enable_cpu_backup=False),
        ):
            if failure_source == "body":
                raise ValueError("body failed")

    finalize.assert_not_called()

    with impl.region("weights", enable_cpu_backup=False):
        impl.finalize_write_mode(fresh_model)
    finalize.assert_called_once_with(weights, fresh_model)


def test_failed_finalization_is_cleared_and_not_retried(build_impl, monkeypatch):
    impl, weights, _, _ = build_impl()
    model = torch.nn.Module()
    finalize = Mock(side_effect=ValueError("finalization failed"))
    monkeypatch.setattr(gms_memory_saver, "finalize_gms_write", finalize)

    with (
        pytest.raises(ValueError, match="finalization failed"),
        impl.region("weights", enable_cpu_backup=False),
    ):
        impl.finalize_write_mode(model)

    with impl.region("weights", enable_cpu_backup=False):
        pass
    finalize.assert_called_once_with(weights, model)


def test_invalid_or_duplicate_publication_preserves_first_model(
    build_impl, monkeypatch
):
    impl, weights, _, _ = build_impl()
    first_model = torch.nn.Module()
    finalize = Mock(return_value=SimpleNamespace(committed_bytes=1))
    monkeypatch.setattr(gms_memory_saver, "finalize_gms_write", finalize)

    with impl.region("weights", enable_cpu_backup=False):
        impl.finalize_write_mode(first_model)
        with pytest.raises(TypeError, match="must not be None"):
            impl.finalize_write_mode(None)
        with pytest.raises(RuntimeError, match="publication is already pending"):
            impl.finalize_write_mode(torch.nn.Module())

    finalize.assert_called_once_with(weights, first_model)
