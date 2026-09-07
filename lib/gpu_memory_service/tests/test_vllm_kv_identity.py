# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from gpu_memory_service.integrations.vllm import install_vmm_ipc_kv, kv_identity

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.none,
    pytest.mark.gpu_0,
]


@pytest.fixture(autouse=True)
def _clear_dynamic_gms_role_env(monkeypatch):
    monkeypatch.delenv("DYN_VLLM_GMS_ACTIVE_LOCK_HELD", raising=False)
    yield
    monkeypatch.delenv("DYN_VLLM_GMS_ACTIVE_LOCK_HELD", raising=False)


def test_v2_semantic_kv_tags_follow_layer_identity():
    tensor_a = SimpleNamespace(
        shared_by=["model.layers.1.self_attn", "model.layers.0.self_attn"], size=123
    )
    tensor_b = SimpleNamespace(shared_by=["model.layers.0.mla"], size=456)
    same_a_different_order = SimpleNamespace(
        shared_by=["model.layers.0.self_attn", "model.layers.1.self_attn"],
        size=789,
    )

    tag_a, tag_b = install_vmm_ipc_kv._semantic_kv_tensor_tag_plan(
        SimpleNamespace(kv_cache_tensors=[tensor_a, tensor_b])
    )
    tag_b2, tag_a2 = install_vmm_ipc_kv._semantic_kv_tensor_tag_plan(
        SimpleNamespace(kv_cache_tensors=[tensor_b, same_a_different_order])
    )

    assert tag_a.startswith("kv_pool:v2:")
    assert tag_b.startswith("kv_pool:v2:")
    assert tag_a == tag_a2
    assert tag_b == tag_b2
    assert tag_a != tag_b


def test_semantic_kv_tags_disambiguate_duplicate_layer_identity():
    tensor_a = SimpleNamespace(shared_by=["model.layers.0.self_attn"], size=123)
    tensor_b = SimpleNamespace(shared_by=["model.layers.0.self_attn"], size=123)

    tag_a, tag_b = install_vmm_ipc_kv._semantic_kv_tensor_tag_plan(
        SimpleNamespace(kv_cache_tensors=[tensor_a, tensor_b])
    )

    assert tag_a.startswith("kv_pool:v2:")
    assert tag_b.startswith("kv_pool:v2:")
    assert tag_a != tag_b
    assert tag_a.endswith(":dup0")
    assert tag_b.endswith(":dup1")


@pytest.mark.parametrize(
    ("existing_tags", "expected"),
    [
        ([], False),
        (["kv:a", "kv:b"], True),
    ],
)
def test_persistent_tag_plan_distinguishes_new_and_complete_reattach(
    existing_tags, expected
):
    class Manager:
        def list_persistent(self, engine_id=None, *, include_unclaimed=False):
            assert engine_id == "engine"
            assert include_unclaimed is True
            return [SimpleNamespace(tag=tag) for tag in existing_tags]

    assert (
        install_vmm_ipc_kv._persistent_tag_plan_reattaches(
            Manager(), "engine", ["kv:a", "kv:b"]
        )
        is expected
    )


def test_persistent_tag_plan_rejects_partial_reattach():
    manager = SimpleNamespace(
        list_persistent=lambda engine_id=None, include_unclaimed=False: [
            SimpleNamespace(tag="kv:a")
        ]
    )

    with pytest.raises(RuntimeError, match="only partially present"):
        install_vmm_ipc_kv._persistent_tag_plan_reattaches(
            manager, "engine", ["kv:a", "kv:b"]
        )


def test_persistent_kv_zeros_as_empty_only_rewrites_int8(monkeypatch):
    import sys
    from types import SimpleNamespace

    int8_marker = object()
    fp16_marker = object()
    calls = []

    def fake_zeros(*args, **kwargs):
        calls.append(("zeros", args, dict(kwargs)))
        return ("zeros", kwargs.get("dtype"))

    def fake_empty(*args, **kwargs):
        calls.append(("empty", args, dict(kwargs)))
        return ("empty", kwargs.get("dtype"))

    fake_torch = SimpleNamespace(
        int8=int8_marker,
        float16=fp16_marker,
        zeros=fake_zeros,
        empty=fake_empty,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    with install_vmm_ipc_kv._persistent_kv_zeros_as_empty(True):
        assert fake_torch.zeros((16,), dtype=int8_marker, device="cuda") == (
            "empty",
            int8_marker,
        )
        assert fake_torch.zeros((16,), dtype=fp16_marker) == (
            "zeros",
            fp16_marker,
        )

    assert fake_torch.zeros is fake_zeros
    assert calls[0][0] == "empty"
    assert calls[1][0] == "zeros"

    calls.clear()
    with install_vmm_ipc_kv._persistent_kv_zeros_as_empty(False):
        assert fake_torch.zeros((16,), dtype=int8_marker) == ("zeros", int8_marker)
    assert calls == [("zeros", ((16,),), {"dtype": int8_marker})]


def test_generic_failover_shadow_mode_enables_shared_geometry(monkeypatch):
    monkeypatch.delenv("DYN_VLLM_GMS_SHADOW_MODE", raising=False)
    monkeypatch.delenv("GMS_VLLM_SHARED_KV", raising=False)
    monkeypatch.setenv("DYN_GMS_FAILOVER_SHADOW_MODE", "true")

    assert kv_identity.shared_kv_enabled()
    assert kv_identity.allocation_shared()
    assert kv_identity.use_existing_shared_geometry()


def test_vllm_v2_device_index_uses_current_cuda_device_for_unindexed_cuda(
    monkeypatch,
):
    from types import SimpleNamespace

    from gpu_memory_service.integrations.vllm import install_vmm_ipc_kv

    monkeypatch.setattr(install_vmm_ipc_kv, "_current_cuda_device", lambda: 3)

    assert install_vmm_ipc_kv._device_index(SimpleNamespace(index=None)) == 3


def test_geometry_wait_honors_vllm_specific_timeout(monkeypatch):
    from gpu_memory_service.integrations.vllm import install_vmm_ipc_kv

    monkeypatch.setenv("GMS_KV_LEASE_GEOMETRY_WAIT_MS", "300000")
    monkeypatch.setenv("GMS_VLLM_KV_GEOMETRY_WAIT_MS", "42")

    assert install_vmm_ipc_kv._geometry_wait_ms(-1) == 42


def test_vllm_geometry_patch_updates_late_engine_core_alias(monkeypatch):
    import sys
    import types

    from gpu_memory_service.integrations.vllm import install_vmm_ipc_kv

    def original(_vllm_config, _kv_cache_specs, _available_memory):
        return "original"

    vllm_mod = types.ModuleType("vllm")
    v1_mod = types.ModuleType("vllm.v1")
    core_pkg = types.ModuleType("vllm.v1.core")
    kv_cache_utils = types.ModuleType("vllm.v1.core.kv_cache_utils")
    kv_cache_utils.get_kv_cache_configs = original
    core_pkg.kv_cache_utils = kv_cache_utils

    monkeypatch.setitem(sys.modules, "vllm", vllm_mod)
    monkeypatch.setitem(sys.modules, "vllm.v1", v1_mod)
    monkeypatch.setitem(sys.modules, "vllm.v1.core", core_pkg)
    monkeypatch.setitem(sys.modules, "vllm.v1.core.kv_cache_utils", kv_cache_utils)
    monkeypatch.delitem(sys.modules, "vllm.v1.engine.core", raising=False)
    monkeypatch.setattr(install_vmm_ipc_kv, "_GEOMETRY_PATCH_INSTALLED", False)

    assert install_vmm_ipc_kv.install_geometry_patch()
    patched = kv_cache_utils.get_kv_cache_configs
    assert getattr(patched, "_gms_geometry_patched", False)

    engine_core = types.ModuleType("vllm.v1.engine.core")
    engine_core.get_kv_cache_configs = original
    monkeypatch.setitem(sys.modules, "vllm.v1.engine.core", engine_core)

    assert install_vmm_ipc_kv.install_geometry_patch()
    assert engine_core.get_kv_cache_configs is patched
