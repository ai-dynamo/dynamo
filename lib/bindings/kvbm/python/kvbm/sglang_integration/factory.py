# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Factory for SGLang's no-native-L2 Dynamo KVBM mode."""

from __future__ import annotations

import hashlib
import os

from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
    resolve_hybrid_device_pool_group,
)
from sglang.srt.mem_cache.registry import (
    TreeCacheBuildContext,
    create_unified_radix_cache_without_hicache,
)
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import get_disagg, get_memory

# isort: split

from kvbm._core import SglangLocalKvStore
from kvbm.sglang_integration.linker import DynamoKvbmLinker
from kvbm.sglang_integration.provider import HostRegionRequest, get_host_memory_provider


def _manager_namespace(ctx: TreeCacheBuildContext, tensors: list) -> bytes:
    deployment_namespace = os.environ.get("DYN_KVBM_MANAGER_NAMESPACE")
    if not deployment_namespace:
        raise ValueError(
            "DYN_KVBM_MANAGER_NAMESPACE is required and must uniquely identify "
            "the deployment/model revision."
        )
    digest = hashlib.sha256()
    digest.update(b"dynamo-kvbm-manager-v1\0")
    for text in (deployment_namespace, ctx.server_args.model_path):
        encoded = text.encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    digest.update(ctx.params.page_size.to_bytes(8, "big"))
    for rank in (
        ctx.tp_rank,
        -1 if ctx.dp_rank is None else ctx.dp_rank,
        ctx.attn_dp_rank,
        ctx.attn_cp_rank,
        ctx.params.pp_rank,
    ):
        digest.update(rank.to_bytes(4, "big", signed=True))
    for tensor in tensors:
        for values in (tensor.shape, tensor.stride()):
            digest.update(len(values).to_bytes(4, "big"))
            for value in values:
                digest.update(value.to_bytes(8, "big", signed=True))
        dtype = str(tensor.dtype).encode()
        digest.update(len(dtype).to_bytes(4, "big"))
        digest.update(dtype)
        digest.update(tensor.nbytes.to_bytes(8, "big"))
    return digest.digest()


def _validate_mode(ctx: TreeCacheBuildContext) -> None:
    memory = get_memory()
    conflicts = []
    if ctx.enable_hierarchical_cache or memory.enable_hierarchical_cache:
        conflicts.append("--enable-hierarchical-cache")
    if memory.hicache_size > 0:
        conflicts.append("--hicache-size")
    if memory.hicache_ratio is not None:
        conflicts.append("--hicache-ratio")
    if memory.hicache_storage_backend is not None:
        conflicts.append("--hicache-storage-backend")
    if memory.enable_lmcache:
        conflicts.append("--enable-lmcache")
    if memory.enable_flexkv:
        conflicts.append("--enable-flexkv")
    if get_disagg().disaggregation_decode_retraction_backup == "host_pool":
        conflicts.append("--disaggregation-decode-retraction-backup=host_pool")
    if conflicts:
        raise ValueError(
            "dynamo_kvbm is mutually exclusive with native/external cache modes: "
            + ", ".join(conflicts)
        )
    if ctx.disable_radix_cache:
        raise ValueError("dynamo_kvbm requires SGLang's GPU radix cache.")
    if ctx.is_hybrid_swa or ctx.is_hybrid_ssm or ctx.is_dsa:
        raise ValueError("Dynamo KVBM V1 supports only homogeneous FULL attention.")
    if ctx.params.pp_size != 1:
        raise ValueError("Dynamo KVBM V1 requires PP=1.")
    if ctx.params.is_eagle:
        raise ValueError(
            "Dynamo KVBM V1 does not support bigram/speculative tree keys."
        )


def _assert_no_native_l2(cache: UnifiedRadixCache) -> None:
    if (
        any(
            value is not None
            for value in (
                cache.cache_controller,
                cache.host_pool_group,
                cache._storage_attachment,
                cache.buffer_pipeline,
            )
        )
        or cache.enable_storage
    ):
        raise RuntimeError("dynamo_kvbm unexpectedly initialized native HiCache state.")


def build_dynamo_kvbm_cache(ctx: TreeCacheBuildContext) -> UnifiedRadixCache:
    _validate_mode(ctx)
    cache = create_unified_radix_cache_without_hicache(ctx)
    if not isinstance(cache, UnifiedRadixCache):
        raise TypeError("dynamo_kvbm requires UnifiedRadixCache.")
    if not isinstance(cache.tree_core, UnifiedTreeCore):
        raise TypeError("dynamo_kvbm V1 requires SGLang's Python unified tree core.")
    if set(cache.tree_components) != {ComponentType.FULL}:
        raise ValueError("dynamo_kvbm V1 requires exactly the FULL tree component.")

    kvcache = ctx.params.token_to_kv_pool_allocator.get_kvcache()
    pool_group = resolve_hybrid_device_pool_group(
        kvcache=kvcache,
        page_size=ctx.params.page_size,
        params=ctx.params,
        components={ComponentType.FULL},
    )
    if set(pool_group.entry_map) != {PoolName.KV}:
        raise ValueError("dynamo_kvbm V1 requires exactly one physical KV pool.")
    tensors = pool_group.entry_map[PoolName.KV].get_hybrid_pool_buffer()
    if len(tensors) != 2 * pool_group.num_layers:
        raise ValueError("dynamo_kvbm V1 requires one K/V tensor pair per layer.")
    total_slots = kvcache.size + ctx.params.page_size
    if total_slots % ctx.params.page_size:
        raise ValueError("SGLang KV pool capacity is not page aligned.")
    num_device_blocks = total_slots // ctx.params.page_size
    if not tensors or any(tensor.nbytes != tensors[0].nbytes for tensor in tensors):
        raise ValueError("dynamo_kvbm V1 requires homogeneous K/V tensor sizes.")
    total_tensor_bytes = sum(tensor.nbytes for tensor in tensors)
    if total_tensor_bytes % num_device_blocks:
        raise ValueError("K/V tensor bytes are not divisible into whole device blocks.")
    cuda_device = tensors[0].device.index
    if cuda_device is None or any(
        tensor.device.type != "cuda" or tensor.device.index != cuda_device
        for tensor in tensors
    ):
        raise ValueError("dynamo_kvbm V1 requires one CUDA device for all KV tensors.")
    bytes_per_block = total_tensor_bytes // num_device_blocks
    manager_namespace = _manager_namespace(ctx, tensors)
    requested_bytes = int(
        os.environ.get("DYN_KVBM_G2_CAPACITY_BYTES", str(100 * 1024**3))
    )
    requested_bytes -= requested_bytes % bytes_per_block
    if requested_bytes <= 0:
        raise ValueError("DYN_KVBM_G2_CAPACITY_BYTES cannot hold one KV block.")

    provider = get_host_memory_provider()
    region = provider.attach(
        HostRegionRequest(
            requested_bytes=requested_bytes,
            bytes_per_block=bytes_per_block,
            alignment=512,
            manager_namespace=manager_namespace,
            tp_rank=ctx.tp_rank,
            dp_rank=ctx.dp_rank,
            attn_dp_rank=ctx.attn_dp_rank,
            attn_cp_rank=ctx.attn_cp_rank,
        ),
        cuda_device,
    )
    try:
        region_ptr = region.data_ptr()
        region_nbytes = region.nbytes()
        if (
            region_ptr <= 0
            or region_ptr % 512
            or region_nbytes < bytes_per_block
            or region_nbytes % bytes_per_block
        ):
            raise ValueError(
                "The owner-backed KVBM region does not satisfy its "
                "pointer/block geometry."
            )
    except Exception:
        region.abort()
        raise
    core = None
    try:
        core = SglangLocalKvStore(
            ctx.params.page_size,
            num_device_blocks,
            tensors,
            region_ptr,
            region_nbytes,
            manager_namespace,
            region,
            cuda_device,
        )
        region.activate()
    except Exception:
        try:
            if core is not None:
                core.close()
        finally:
            region.abort()
        raise

    linker = None
    try:
        linker = DynamoKvbmLinker(
            core=core,
            manager_namespace=manager_namespace,
            page_size=ctx.params.page_size,
            num_device_blocks=num_device_blocks,
            num_layers=pool_group.num_layers,
            host_region=region,
        )
        cache.init_cache_linker(linker)
        ctx.tp_worker.register_hicache_layer_transfer_counter(linker.layer_done_counter)
        kvcache.register_layer_transfer_counter(linker.layer_done_counter)
        _assert_no_native_l2(cache)
    except Exception:
        if linker is not None:
            linker.close()
        else:
            core.close()
            region.close()
        raise
    return cache
