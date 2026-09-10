# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import threading
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


def test_pre_vllm_import_arms_lazy_installer(monkeypatch):
    calls = []
    monkeypatch.delitem(
        install_vmm_ipc_kv.sys.modules,
        "vllm.v1.worker.gpu_model_runner",
        raising=False,
    )
    monkeypatch.setattr(install_vmm_ipc_kv, "_is_enabled", lambda: True)
    monkeypatch.setattr(
        install_vmm_ipc_kv, "install_lazy", lambda: calls.append("lazy")
    )

    install_vmm_ipc_kv._install_or_arm()

    assert calls == ["lazy"]


def test_v3_semantic_kv_tags_include_model_layers_and_size():
    tensor_a = SimpleNamespace(
        shared_by=["model.layers.1.self_attn", "model.layers.0.self_attn"],
        size=123,
    )
    same_a_different_order = SimpleNamespace(
        shared_by=["model.layers.0.self_attn", "model.layers.1.self_attn"],
        size=123,
    )
    resized_a = SimpleNamespace(shared_by=tensor_a.shared_by, size=789)

    def config(tensor):
        return SimpleNamespace(kv_cache_tensors=[tensor])

    tag_a = install_vmm_ipc_kv._semantic_kv_tensor_tag_plan(
        config(tensor_a), "model=org/model\0revision=abc"
    )[0]
    same_tag = install_vmm_ipc_kv._semantic_kv_tensor_tag_plan(
        config(same_a_different_order), "model=org/model\0revision=abc"
    )[0]
    resized_tag = install_vmm_ipc_kv._semantic_kv_tensor_tag_plan(
        config(resized_a), "model=org/model\0revision=abc"
    )[0]
    different_model_tag = install_vmm_ipc_kv._semantic_kv_tensor_tag_plan(
        config(tensor_a), "model=org/model\0revision=def"
    )[0]

    assert tag_a.startswith("kv_pool:v3:")
    assert tag_a == same_tag
    assert tag_a != resized_tag
    assert tag_a != different_model_tag


def test_model_identity_includes_resolved_model_revision():
    commit = "a" * 40
    identity = install_vmm_ipc_kv._model_identity(
        SimpleNamespace(
            model="org/model",
            revision="main",
            code_revision=None,
            quantization="fp8",
            hf_config=SimpleNamespace(_commit_hash=commit),
        )
    )

    assert identity == f"model=org/model\0artifact={commit}\0quantization=fp8"


def test_model_identity_accepts_explicit_artifact_digest(monkeypatch):
    monkeypatch.setenv("GMS_VLLM_MODEL_ARTIFACT_DIGEST", "image-sha256:abc")
    identity = install_vmm_ipc_kv._model_identity(
        SimpleNamespace(model="/models/current", revision=None, code_revision="main")
    )

    assert identity == "model=/models/current\0artifact=image-sha256:abc"


@pytest.mark.parametrize("revision", [None, "main", "refs/pr/1", "/models/current"])
def test_model_identity_rejects_mutable_revision(monkeypatch, revision):
    monkeypatch.delenv("GMS_VLLM_MODEL_ARTIFACT_DIGEST", raising=False)
    with pytest.raises(RuntimeError, match="immutable resolved model revision"):
        install_vmm_ipc_kv._model_identity(
            SimpleNamespace(model="org/model", revision=revision)
        )


def test_model_identity_is_derived_from_runner_config():
    commit = "b" * 40
    runner = SimpleNamespace(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(
                model="org/model",
                revision="main",
                hf_config=SimpleNamespace(_commit_hash=commit),
            )
        )
    )

    assert install_vmm_ipc_kv._model_identity_from_runner(runner) == (
        f"model=org/model\0artifact={commit}"
    )


def test_model_identity_fails_closed_when_unavailable():
    with pytest.raises(RuntimeError, match="immutable resolved model revision"):
        install_vmm_ipc_kv._model_identity(SimpleNamespace())


def test_semantic_kv_tags_fail_closed_without_model_identity():
    config = SimpleNamespace(
        kv_cache_tensors=[SimpleNamespace(shared_by=["layer.0"], size=123)]
    )

    with pytest.raises(RuntimeError, match="stable model identity"):
        install_vmm_ipc_kv._semantic_kv_tensor_tag_plan(config)


def test_semantic_kv_tags_disambiguate_duplicate_layer_identity():
    tensor_a = SimpleNamespace(shared_by=["model.layers.0.self_attn"], size=123)
    tensor_b = SimpleNamespace(shared_by=["model.layers.0.self_attn"], size=123)

    tag_a, tag_b = install_vmm_ipc_kv._semantic_kv_tensor_tag_plan(
        SimpleNamespace(kv_cache_tensors=[tensor_a, tensor_b]), "model=org/model"
    )

    assert tag_a.startswith("kv_pool:v3:")
    assert tag_b.startswith("kv_pool:v3:")
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


def test_persistent_kv_zeros_as_empty_is_context_local(monkeypatch):
    import sys

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

    other_thread_result = []
    with install_vmm_ipc_kv._persistent_kv_zeros_as_empty(True):
        assert fake_torch.zeros((16,), dtype=int8_marker, device="cuda") == (
            "empty",
            int8_marker,
        )
        thread = threading.Thread(
            target=lambda: other_thread_result.append(
                fake_torch.zeros((16,), dtype=int8_marker)
            )
        )
        thread.start()
        thread.join()
        assert fake_torch.zeros((16,), dtype=fp16_marker) == (
            "zeros",
            fp16_marker,
        )

    assert other_thread_result == [("zeros", int8_marker)]
    assert fake_torch.zeros((16,), dtype=int8_marker) == ("zeros", int8_marker)
    assert [kind for kind, _, _ in calls] == ["empty", "zeros", "zeros", "zeros"]


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
