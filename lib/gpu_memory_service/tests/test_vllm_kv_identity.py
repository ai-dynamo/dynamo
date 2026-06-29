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
    for name in (
        "DYN_VLLM_GMS_ACTIVE_LOCK_HELD",
        "DYN_VLLM_GMS_FORCE_PRIVATE_BOOTSTRAP_KV",
        "GMS_PERSISTENT_DEFER_PHYSICAL_SCRATCH_BACKED",
    ):
        monkeypatch.delenv(name, raising=False)
    yield
    for name in (
        "DYN_VLLM_GMS_ACTIVE_LOCK_HELD",
        "DYN_VLLM_GMS_FORCE_PRIVATE_BOOTSTRAP_KV",
        "GMS_PERSISTENT_DEFER_PHYSICAL_SCRATCH_BACKED",
    ):
        monkeypatch.delenv(name, raising=False)


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


def test_private_bootstrap_kv_zeros_as_empty_only_rewrites_int8(monkeypatch):
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

    with install_vmm_ipc_kv._private_bootstrap_kv_zeros_as_empty(True):
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
    with install_vmm_ipc_kv._private_bootstrap_kv_zeros_as_empty(False):
        assert fake_torch.zeros((16,), dtype=int8_marker) == ("zeros", int8_marker)
    assert calls == [("zeros", ((16,),), {"dtype": int8_marker})]


def test_v1_profiling_kv_tensors_bypass_persistent_pool(monkeypatch):
    import sys
    import types
    from contextlib import contextmanager

    from gpu_memory_service.client.torch import allocator as torch_allocator

    class FakeGPUModelRunner:
        device = 0

        def initialize_kv_cache(self, kv_cache_config, is_profiling=False):
            return self.initialize_kv_cache_tensors(kv_cache_config, [16])

        def initialize_kv_cache_tensors(self, _kv_cache_config, _kernel_block_sizes):
            return "original"

    calls = []

    def fake_get_or_create(*args, **kwargs):
        calls.append(("register", args, kwargs))
        return object()

    @contextmanager
    def fake_pool(tag, device):
        calls.append(("pool", tag, device))
        yield

    monkeypatch.setattr(install_vmm_ipc_kv, "_INSTALLED", False)
    monkeypatch.setattr(install_vmm_ipc_kv, "_install_kv_leases", lambda: False)
    monkeypatch.setattr(
        install_vmm_ipc_kv, "_resolve_socket", lambda device: f"sock-{device}"
    )
    monkeypatch.setattr(
        install_vmm_ipc_kv, "_engine_id", lambda device=0: f"engine-{device}"
    )
    monkeypatch.setattr(install_vmm_ipc_kv, "_shared_kv_enabled", lambda: True)
    monkeypatch.setattr(install_vmm_ipc_kv, "_defer_physical_enabled", lambda: False)
    monkeypatch.setattr(
        torch_allocator, "get_or_create_persistent_allocator", fake_get_or_create
    )
    monkeypatch.setattr(torch_allocator, "gms_use_persistent_pool", fake_pool)
    monkeypatch.setattr(
        torch_allocator,
        "set_persistent_allocator_tag_plan",
        lambda tag, plan: calls.append(("plan", tag, tuple(plan))),
    )
    monkeypatch.setattr(
        torch_allocator,
        "clear_persistent_allocator_tag_plan",
        lambda tag: calls.append(("clear", tag)),
    )

    vllm_mod = types.ModuleType("vllm")
    v1_mod = types.ModuleType("vllm.v1")
    core_pkg = types.ModuleType("vllm.v1.core")
    kv_cache_utils = types.ModuleType("vllm.v1.core.kv_cache_utils")
    kv_cache_utils.get_kv_cache_configs = lambda *args, **kwargs: None
    worker_pkg = types.ModuleType("vllm.v1.worker")
    gpu_pkg = types.ModuleType("vllm.v1.worker.gpu")
    attn_utils = types.ModuleType("vllm.v1.worker.gpu.attn_utils")
    attn_utils._allocate_kv_cache = lambda *args, **kwargs: "v2"
    gpu_pkg.attn_utils = attn_utils
    gpu_model_runner = types.ModuleType("vllm.v1.worker.gpu_model_runner")
    gpu_model_runner.GPUModelRunner = FakeGPUModelRunner

    for name, module in {
        "vllm": vllm_mod,
        "vllm.v1": v1_mod,
        "vllm.v1.core": core_pkg,
        "vllm.v1.core.kv_cache_utils": kv_cache_utils,
        "vllm.v1.worker": worker_pkg,
        "vllm.v1.worker.gpu": gpu_pkg,
        "vllm.v1.worker.gpu.attn_utils": attn_utils,
        "vllm.v1.worker.gpu_model_runner": gpu_model_runner,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    assert install_vmm_ipc_kv.install()

    cfg = SimpleNamespace(
        kv_cache_tensors=[SimpleNamespace(shared_by=["layer.0"], size=128)]
    )
    runner = FakeGPUModelRunner()

    assert runner.initialize_kv_cache(cfg, is_profiling=True) == "original"
    assert calls == []

    assert runner.initialize_kv_cache(cfg) == "original"
    assert calls[0][0] == "register"
    assert calls[1][0] == "plan"
    assert calls[2][0] == "pool"
    assert calls[3][0] == "clear"


def test_default_shared_shadow_uses_stable_shared_id(monkeypatch):
    monkeypatch.setenv("DYN_VLLM_GMS_SHADOW_MODE", "true")
    monkeypatch.setenv("DYN_NAMESPACE", "ns")
    monkeypatch.setenv("DYN_COMPONENT", "worker")
    monkeypatch.delenv("DYN_VLLM_GMS_PRIVATE_BOOTSTRAP_KV", raising=False)
    monkeypatch.delenv("GMS_VLLM_PRIVATE_BOOTSTRAP_KV", raising=False)
    monkeypatch.delenv("GMS_VLLM_VMM_IPC_ENGINE_ID", raising=False)

    stable = kv_identity.stable_engine_id(0)

    assert kv_identity.shared_kv_enabled()
    assert not kv_identity.private_bootstrap_kv_enabled()
    assert kv_identity.allocation_engine_id(0) == stable
    assert kv_identity.promotion_engine_id(0) == stable
    assert kv_identity.allocation_shared()
    assert kv_identity.use_existing_shared_geometry()


def test_private_bootstrap_uses_member_scoped_init_id(monkeypatch):
    monkeypatch.setenv("DYN_VLLM_GMS_SHADOW_MODE", "true")
    monkeypatch.setenv("DYN_VLLM_GMS_PRIVATE_BOOTSTRAP_KV", "1")
    monkeypatch.setenv("GMS_VLLM_VMM_IPC_ENGINE_ID", "shared-engine")
    monkeypatch.setenv("ENGINE_ID", "shadow-1")

    assert kv_identity.private_bootstrap_kv_enabled()
    assert kv_identity.stable_engine_id(0) == "shared-engine"
    assert kv_identity.allocation_engine_id(0) == "shared-engine|bootstrap=shadow-1"
    assert kv_identity.promotion_engine_id(0) == "shared-engine"
    assert not kv_identity.allocation_shared()
    assert kv_identity.use_existing_shared_geometry()


def test_private_bootstrap_is_shadow_only_in_failover(monkeypatch):
    monkeypatch.setenv("DYN_GMS_FAILOVER_SHADOW_MODE", "true")
    monkeypatch.setenv("DYN_VLLM_GMS_SHADOW_MODE", "true")
    monkeypatch.setenv("DYN_VLLM_GMS_PRIVATE_BOOTSTRAP_KV", "1")
    monkeypatch.setenv("GMS_VLLM_VMM_IPC_ENGINE_ID", "shared-engine")
    monkeypatch.delenv("DYN_VLLM_GMS_FORCE_PRIVATE_BOOTSTRAP_KV", raising=False)
    monkeypatch.delenv("DYN_VLLM_GMS_ACTIVE_LOCK_HELD", raising=False)
    monkeypatch.setenv("DYN_GMS_FAILOVER_PRIMARY_ENGINE_ID", "0")
    monkeypatch.setenv("ENGINE_ID", "0")

    assert not kv_identity.private_bootstrap_kv_enabled()
    assert kv_identity.allocation_engine_id(0) == "shared-engine"
    assert kv_identity.allocation_shared()

    monkeypatch.setenv("ENGINE_ID", "1")

    assert kv_identity.private_bootstrap_kv_enabled()
    assert kv_identity.allocation_engine_id(0) == "shared-engine|bootstrap=1"
    assert not kv_identity.allocation_shared()


def test_dynamic_preinit_role_overrides_static_primary(monkeypatch):
    monkeypatch.setenv("DYN_GMS_FAILOVER_SHADOW_MODE", "true")
    monkeypatch.setenv("DYN_VLLM_GMS_SHADOW_MODE", "true")
    monkeypatch.setenv("DYN_VLLM_GMS_PRIVATE_BOOTSTRAP_KV", "1")
    monkeypatch.setenv("GMS_VLLM_VMM_IPC_ENGINE_ID", "shared-engine")
    monkeypatch.setenv("DYN_GMS_FAILOVER_PRIMARY_ENGINE_ID", "0")
    monkeypatch.setenv("ENGINE_ID", "0")
    monkeypatch.setenv("DYN_VLLM_GMS_FORCE_PRIVATE_BOOTSTRAP_KV", "1")

    assert kv_identity.private_bootstrap_kv_enabled()
    assert kv_identity.allocation_engine_id(0) == "shared-engine|bootstrap=0"
    assert not kv_identity.allocation_shared()


def test_dynamic_preinit_lock_holder_uses_shared_kv(monkeypatch):
    monkeypatch.setenv("DYN_GMS_FAILOVER_SHADOW_MODE", "true")
    monkeypatch.setenv("DYN_VLLM_GMS_SHADOW_MODE", "true")
    monkeypatch.setenv("DYN_VLLM_GMS_PRIVATE_BOOTSTRAP_KV", "1")
    monkeypatch.setenv("GMS_VLLM_VMM_IPC_ENGINE_ID", "shared-engine")
    monkeypatch.setenv("DYN_GMS_FAILOVER_PRIMARY_ENGINE_ID", "0")
    monkeypatch.setenv("ENGINE_ID", "1")
    monkeypatch.setenv("DYN_VLLM_GMS_ACTIVE_LOCK_HELD", "1")
    monkeypatch.setenv("DYN_VLLM_GMS_FORCE_PRIVATE_BOOTSTRAP_KV", "1")

    assert not kv_identity.private_bootstrap_kv_enabled()
    assert kv_identity.allocation_engine_id(0) == "shared-engine"
    assert kv_identity.allocation_shared()


def test_private_bootstrap_can_be_disabled(monkeypatch):
    monkeypatch.setenv("DYN_VLLM_GMS_SHADOW_MODE", "true")
    monkeypatch.setenv("DYN_VLLM_GMS_PRIVATE_BOOTSTRAP_KV", "0")
    monkeypatch.setenv("GMS_VLLM_PRIVATE_BOOTSTRAP_KV", "1")
    monkeypatch.setenv("GMS_VLLM_VMM_IPC_ENGINE_ID", "shared-engine")

    assert not kv_identity.private_bootstrap_kv_enabled()
    assert kv_identity.allocation_engine_id(0) == "shared-engine"
    assert kv_identity.allocation_shared()


def test_private_bootstrap_scratch_warmup_requires_private_bootstrap(monkeypatch):
    monkeypatch.setenv("DYN_VLLM_GMS_SHADOW_MODE", "true")
    monkeypatch.setenv("DYN_VLLM_GMS_PRIVATE_BOOTSTRAP_KV", "0")
    monkeypatch.setenv("DYN_VLLM_GMS_PRIVATE_BOOTSTRAP_SCRATCH_WARMUP", "1")

    assert not kv_identity.private_bootstrap_kv_enabled()
    assert not kv_identity.private_bootstrap_scratch_warmup_enabled()


def test_private_bootstrap_scratch_warmup_is_shadow_only(monkeypatch):
    monkeypatch.setenv("DYN_GMS_FAILOVER_SHADOW_MODE", "true")
    monkeypatch.setenv("DYN_VLLM_GMS_SHADOW_MODE", "true")
    monkeypatch.setenv("DYN_VLLM_GMS_PRIVATE_BOOTSTRAP_KV", "1")
    monkeypatch.setenv("DYN_VLLM_GMS_PRIVATE_BOOTSTRAP_SCRATCH_WARMUP", "1")
    monkeypatch.setenv("DYN_GMS_FAILOVER_PRIMARY_ENGINE_ID", "0")
    monkeypatch.setenv("ENGINE_ID", "0")

    assert not kv_identity.private_bootstrap_scratch_warmup_enabled()

    monkeypatch.setenv("ENGINE_ID", "1")

    assert kv_identity.private_bootstrap_kv_enabled()
    assert kv_identity.private_bootstrap_scratch_warmup_enabled()


def test_vllm_v2_device_index_uses_current_cuda_device_for_unindexed_cuda(
    monkeypatch,
):
    from types import SimpleNamespace

    from gpu_memory_service.integrations.vllm import install_vmm_ipc_kv

    monkeypatch.setattr(install_vmm_ipc_kv, "_current_cuda_device", lambda: 3)

    assert install_vmm_ipc_kv._device_index(SimpleNamespace(index=None)) == 3


def test_private_bootstrap_caps_startup_memory_for_existing_shared_geometry(
    monkeypatch,
):
    from types import SimpleNamespace

    from gpu_memory_service.integrations.vllm import worker

    monkeypatch.setenv("DYN_VLLM_GMS_SHADOW_MODE", "true")
    monkeypatch.setenv("DYN_VLLM_GMS_PRIVATE_BOOTSTRAP_KV", "1")
    monkeypatch.setenv("DYN_VLLM_GMS_BOOTSTRAP_GPU_MEMORY_UTILIZATION", "0.125")
    monkeypatch.setattr(worker, "_existing_shared_kv_blocks", lambda device: 1024)

    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(gpu_memory_utilization=0.7)
    )

    worker._maybe_cap_private_bootstrap_memory(vllm_config, 0)

    assert vllm_config.cache_config.gpu_memory_utilization == 0.125


def test_existing_shared_kv_blocks_falls_back_to_any_rank_geometry(monkeypatch):
    from gpu_memory_service.integrations.vllm import worker

    monkeypatch.setenv("GMS_KV_LEASES", "1")
    monkeypatch.setattr(
        "gpu_memory_service.integrations.common.kv_lease_client.read_kv_lease_namespace_total_blocks",
        lambda engine, device, *, namespace_suffix="kv": ("vllm:gpu7:block-pool", None),
    )
    monkeypatch.setattr(
        "gpu_memory_service.integrations.common.kv_lease_client.read_any_kv_lease_namespace_total_blocks",
        lambda engine: ("/pod/gms-kv-lease-rank0.shm", 2048),
    )

    assert worker._existing_shared_kv_blocks(7) == 2048


def test_private_bootstrap_caps_startup_memory_before_geometry(monkeypatch):
    from types import SimpleNamespace

    from gpu_memory_service.integrations.vllm import worker

    monkeypatch.setenv("DYN_GMS_FAILOVER_SHADOW_MODE", "true")
    monkeypatch.setenv("DYN_GMS_FAILOVER_PRIMARY_ENGINE_ID", "0")
    monkeypatch.setenv("ENGINE_ID", "1")
    monkeypatch.setenv("DYN_VLLM_GMS_SHADOW_MODE", "true")
    monkeypatch.setenv("DYN_VLLM_GMS_PRIVATE_BOOTSTRAP_KV", "1")
    monkeypatch.setenv("DYN_VLLM_GMS_BOOTSTRAP_GPU_MEMORY_UTILIZATION", "0.125")
    monkeypatch.setattr(worker, "_existing_shared_kv_blocks", lambda device: None)

    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(gpu_memory_utilization=0.7)
    )

    worker._maybe_cap_private_bootstrap_memory(vllm_config, 0)

    assert vllm_config.cache_config.gpu_memory_utilization == 0.125


def test_private_bootstrap_memory_cap_ignores_first_primary(monkeypatch):
    from types import SimpleNamespace

    from gpu_memory_service.integrations.vllm import worker

    monkeypatch.setenv("DYN_GMS_FAILOVER_SHADOW_MODE", "true")
    monkeypatch.setenv("DYN_GMS_FAILOVER_PRIMARY_ENGINE_ID", "0")
    monkeypatch.setenv("ENGINE_ID", "0")
    monkeypatch.setenv("DYN_VLLM_GMS_SHADOW_MODE", "true")
    monkeypatch.setenv("DYN_VLLM_GMS_PRIVATE_BOOTSTRAP_KV", "1")
    monkeypatch.setattr(worker, "_existing_shared_kv_blocks", lambda device: None)

    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(gpu_memory_utilization=0.7)
    )

    worker._maybe_cap_private_bootstrap_memory(vllm_config, 0)

    assert vllm_config.cache_config.gpu_memory_utilization == 0.7


def test_vllm_geometry_reader_uses_block_pool_lease_namespace(monkeypatch):
    from gpu_memory_service.integrations.vllm import install_vmm_ipc_kv

    calls = []

    def fake_read(engine, device, *, namespace_suffix="kv"):
        calls.append((engine, device, namespace_suffix))
        if namespace_suffix == "block-pool":
            return "vllm:gpu2:block-pool", 123
        return "vllm:gpu2:kv", None

    monkeypatch.setenv("GMS_KV_LEASES", "1")
    monkeypatch.setenv("GMS_VLLM_SHARED_KV", "1")
    monkeypatch.setenv("LOCAL_RANK", "2")
    monkeypatch.setattr(
        "gpu_memory_service.integrations.common.kv_lease_client.read_kv_lease_namespace_total_blocks",
        fake_read,
    )

    assert install_vmm_ipc_kv._existing_shared_kv_blocks() == 123
    assert calls == [("vllm", 2, "block-pool")]


def test_private_bootstrap_geometry_wait_extends_generic_timeout(monkeypatch):
    from gpu_memory_service.integrations.vllm import install_vmm_ipc_kv

    monkeypatch.setenv("DYN_GMS_FAILOVER_SHADOW_MODE", "true")
    monkeypatch.setenv("DYN_GMS_FAILOVER_PRIMARY_ENGINE_ID", "0")
    monkeypatch.setenv("ENGINE_ID", "1")
    monkeypatch.setenv("DYN_VLLM_GMS_SHADOW_MODE", "true")
    monkeypatch.setenv("DYN_VLLM_GMS_PRIVATE_BOOTSTRAP_KV", "1")
    monkeypatch.setenv("GMS_KV_LEASE_GEOMETRY_WAIT_MS", "300000")
    monkeypatch.delenv("GMS_VLLM_KV_GEOMETRY_WAIT_MS", raising=False)

    assert install_vmm_ipc_kv._geometry_wait_ms(-1) == 1_800_000


def test_private_bootstrap_geometry_wait_honors_vllm_specific_timeout(monkeypatch):
    from gpu_memory_service.integrations.vllm import install_vmm_ipc_kv

    monkeypatch.setenv("DYN_GMS_FAILOVER_SHADOW_MODE", "true")
    monkeypatch.setenv("DYN_GMS_FAILOVER_PRIMARY_ENGINE_ID", "0")
    monkeypatch.setenv("ENGINE_ID", "1")
    monkeypatch.setenv("DYN_VLLM_GMS_SHADOW_MODE", "true")
    monkeypatch.setenv("DYN_VLLM_GMS_PRIVATE_BOOTSTRAP_KV", "1")
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


def test_release_private_bootstrap_kv_pool_releases_only_kv_tags():
    from types import SimpleNamespace

    from gpu_memory_service.integrations.vllm.kv_identity import (
        release_private_bootstrap_kv_pool,
    )

    class FakeManager:
        def __init__(self):
            self.released = []

        def list_persistent(self, engine_id=None):
            assert engine_id == "stable|bootstrap=1"
            return [
                SimpleNamespace(tag="kv_pool#0"),
                SimpleNamespace(tag="kv_pool#1"),
                SimpleNamespace(tag="weights#0"),
            ]

        def release_persistent(self, engine_id, tag):
            self.released.append((engine_id, tag))
            return True

    manager = FakeManager()

    released = release_private_bootstrap_kv_pool(manager, "stable|bootstrap=1")

    assert released == 2
    assert manager.released == [
        ("stable|bootstrap=1", "kv_pool#0"),
        ("stable|bootstrap=1", "kv_pool#1"),
    ]
