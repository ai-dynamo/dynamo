# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""No-native-L2 and cleanup tests for the SGLang KVBM cache factory."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import SimpleNamespace

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.kvbm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@pytest.fixture
def factory_contract(install_module, load_source):
    class PoolName(Enum):
        KV = "kv"

    class ComponentType(Enum):
        FULL = "full"

    class UnifiedTreeCore:
        pass

    class UnifiedRadixCache:
        def __init__(self):
            self.tree_core = UnifiedTreeCore()
            self.tree_components = [ComponentType.FULL]
            self.cache_controller = None
            self.host_pool_group = None
            self._storage_attachment = None
            self.buffer_pipeline = None
            self.enable_storage = False
            self.linker = None

        def init_cache_linker(self, linker):
            self.linker = linker

    class Tensor:
        def __init__(self, nbytes=64, device_index=3):
            self.nbytes = nbytes
            self.shape = (8, 4)
            self.dtype = "torch.float16"
            self.device = SimpleNamespace(type="cuda", index=device_index)

        def stride(self):
            return (4, 1)

    class Region:
        def __init__(self):
            self.activated = False
            self.aborted = False
            self.closed = False
            self.activation_error = None

        def data_ptr(self):
            return 512

        def nbytes(self):
            return 96

        def activate(self):
            if self.activation_error is not None:
                raise self.activation_error
            self.activated = True

        def abort(self):
            self.aborted = True

        def close(self):
            self.closed = True

    @dataclass(frozen=True)
    class HostRegionRequest:
        requested_bytes: int
        bytes_per_block: int
        alignment: int
        manager_namespace: bytes
        tp_rank: int
        dp_rank: int | None
        attn_dp_rank: int
        attn_cp_rank: int

    memory = SimpleNamespace(
        enable_hierarchical_cache=False,
        hicache_size=0,
        hicache_ratio=None,
        hicache_storage_backend=None,
        enable_lmcache=False,
        enable_flexkv=False,
    )
    disagg = SimpleNamespace(disaggregation_decode_retraction_backup="none")
    cache = UnifiedRadixCache()
    tensors = [Tensor(), Tensor()]
    pool_entry = SimpleNamespace(get_hybrid_pool_buffer=lambda: tensors)
    pool_group = SimpleNamespace(
        entry_map={PoolName.KV: pool_entry},
        num_layers=1,
    )
    region = Region()
    provider_calls = []
    create_calls = []
    core_instances = []
    linker_instances = []

    class Provider:
        def attach(self, request, cuda_device):
            provider_calls.append((request, cuda_device))
            return region

    provider = Provider()

    class SglangLocalKvStore:
        def __init__(self, *args):
            self.args = args
            self.closed = False
            core_instances.append(self)

        def close(self):
            self.closed = True

    class DynamoKvbmLinker:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.layer_done_counter = object()
            self.closed = False
            linker_instances.append(self)

        def close(self):
            self.closed = True
            self.kwargs["core"].close()
            self.kwargs["host_region"].close()

    def create_unified_cache(ctx):
        create_calls.append(ctx)
        return cache

    def resolve_pool_group(**kwargs):
        return pool_group

    install_module(
        "sglang.srt.mem_cache.hicache_storage",
        PoolName=PoolName,
    )
    install_module(
        "sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler",
        resolve_hybrid_device_pool_group=resolve_pool_group,
    )
    install_module(
        "sglang.srt.mem_cache.registry",
        TreeCacheBuildContext=object,
        create_unified_radix_cache_without_hicache=create_unified_cache,
    )
    install_module(
        "sglang.srt.mem_cache.unified_cache.component_type",
        ComponentType=ComponentType,
    )
    install_module(
        "sglang.srt.mem_cache.unified_cache.unified_tree_core",
        UnifiedTreeCore=UnifiedTreeCore,
    )
    install_module(
        "sglang.srt.mem_cache.unified_radix_cache",
        UnifiedRadixCache=UnifiedRadixCache,
    )
    install_module(
        "sglang.srt.runtime_context",
        get_disagg=lambda: disagg,
        get_memory=lambda: memory,
    )
    install_module("kvbm._core", SglangLocalKvStore=SglangLocalKvStore)
    install_module("kvbm.sglang_integration.linker", DynamoKvbmLinker=DynamoKvbmLinker)
    install_module(
        "kvbm.sglang_integration.provider",
        HostRegionRequest=HostRegionRequest,
        get_host_memory_provider=lambda: provider,
    )
    module = load_source("test_kvbm_sglang_factory", "factory.py")

    kvcache = SimpleNamespace(size=6, registered_counters=[])
    kvcache.register_layer_transfer_counter = kvcache.registered_counters.append
    allocator = SimpleNamespace(get_kvcache=lambda: kvcache)
    params = SimpleNamespace(
        page_size=2,
        pp_size=1,
        pp_rank=0,
        is_eagle=False,
        token_to_kv_pool_allocator=allocator,
    )
    tp_worker = SimpleNamespace(registered_counters=[])
    tp_worker.register_hicache_layer_transfer_counter = (
        tp_worker.registered_counters.append
    )
    ctx = SimpleNamespace(
        server_args=SimpleNamespace(model_path="model/revision"),
        params=params,
        tp_worker=tp_worker,
        tp_rank=1,
        dp_rank=2,
        attn_dp_rank=3,
        attn_cp_rank=4,
        enable_hierarchical_cache=False,
        disable_radix_cache=False,
        is_hybrid_swa=False,
        is_hybrid_ssm=False,
        is_dsa=False,
    )
    return SimpleNamespace(
        module=module,
        ctx=ctx,
        memory=memory,
        cache=cache,
        tensors=tensors,
        region=region,
        provider_calls=provider_calls,
        create_calls=create_calls,
        core_instances=core_instances,
        linker_instances=linker_instances,
        kvcache=kvcache,
        tp_worker=tp_worker,
    )


def test_factory_builds_direct_cache_without_native_l2(monkeypatch, factory_contract):
    contract = factory_contract
    monkeypatch.setenv("DYN_KVBM_MANAGER_NAMESPACE", "deployment-a")
    monkeypatch.setenv("DYN_KVBM_G2_CAPACITY_BYTES", "100")

    cache = contract.module.build_dynamo_kvbm_cache(contract.ctx)

    assert cache is contract.cache
    assert contract.create_calls == [contract.ctx]
    assert cache.cache_controller is None
    assert cache.host_pool_group is None
    assert cache._storage_attachment is None
    assert cache.buffer_pipeline is None
    assert cache.enable_storage is False
    assert contract.region.activated is True
    request, cuda_device = contract.provider_calls[0]
    assert request.requested_bytes == 96
    assert request.bytes_per_block == 32
    assert request.alignment == 512
    assert request.tp_rank == 1
    assert request.dp_rank == 2
    assert request.attn_dp_rank == 3
    assert request.attn_cp_rank == 4
    assert len(request.manager_namespace) == 32
    assert cuda_device == 3
    core = contract.core_instances[0]
    assert core.args[:3] == (2, 4, contract.tensors)
    assert core.args[3:5] == (512, 96)
    linker = contract.linker_instances[0]
    assert cache.linker is linker
    assert contract.kvcache.registered_counters == [linker.layer_done_counter]
    assert contract.tp_worker.registered_counters == [linker.layer_done_counter]


@pytest.mark.parametrize(
    ("field", "value", "option"),
    [
        ("enable_hierarchical_cache", True, "--enable-hierarchical-cache"),
        ("hicache_size", 1, "--hicache-size"),
        ("hicache_ratio", 1.0, "--hicache-ratio"),
        ("hicache_storage_backend", "file", "--hicache-storage-backend"),
        ("enable_lmcache", True, "--enable-lmcache"),
        ("enable_flexkv", True, "--enable-flexkv"),
    ],
)
def test_factory_rejects_conflicting_cache_modes_before_allocating(
    factory_contract, field, value, option
):
    contract = factory_contract
    setattr(contract.memory, field, value)

    with pytest.raises(ValueError, match=option):
        contract.module.build_dynamo_kvbm_cache(contract.ctx)

    assert contract.create_calls == []
    assert contract.provider_calls == []


def test_factory_aborts_owner_region_if_activation_fails(monkeypatch, factory_contract):
    contract = factory_contract
    monkeypatch.setenv("DYN_KVBM_MANAGER_NAMESPACE", "deployment-a")
    monkeypatch.setenv("DYN_KVBM_G2_CAPACITY_BYTES", "96")
    contract.region.activation_error = RuntimeError("owner activation failed")

    with pytest.raises(RuntimeError, match="owner activation failed"):
        contract.module.build_dynamo_kvbm_cache(contract.ctx)

    assert contract.core_instances[0].closed is True
    assert contract.region.aborted is True
    assert contract.region.closed is False


def test_manager_namespace_is_deterministic_and_rank_isolated(
    monkeypatch, factory_contract
):
    contract = factory_contract
    monkeypatch.setenv("DYN_KVBM_MANAGER_NAMESPACE", "deployment-a")

    first = contract.module._manager_namespace(contract.ctx, contract.tensors)
    second = contract.module._manager_namespace(contract.ctx, contract.tensors)
    contract.ctx.attn_cp_rank += 1
    different_rank = contract.module._manager_namespace(contract.ctx, contract.tensors)

    assert first == second
    assert first != different_rank
    assert len(first) == 32
