# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from vllm.config import VllmConfig
from vllm.v1.engine import KVCacheGroupMetadata
from vllm.v1.engine.async_llm import AsyncLLM

DYNAMO_KV_EVENT_BLOCK_SIZE_KEY = "dynamo_kv_event_block_size"
MAIN_ATTENTION_KV_CACHE_KINDS = {
    "full_attention",
    "mla_attention",
    "sink_full_attention",
}


def get_configured_kv_event_block_size(vllm_config: VllmConfig) -> int:
    """Return the configured KV event block size, falling back to vLLM's cache block size."""
    additional_config = vllm_config.additional_config or {}
    return additional_config.get(
        DYNAMO_KV_EVENT_BLOCK_SIZE_KEY,
        vllm_config.cache_config.block_size,
    )


def select_main_attention_block_size(
    group_metadata: list[KVCacheGroupMetadata] | None,
    fallback_block_size: int,
) -> int:
    """Select the main-attention KV block size from engine cache-group metadata."""
    if group_metadata is None:
        raise ValueError("vLLM does not provide initialized KV cache group metadata")
    if not group_metadata:
        return fallback_block_size

    block_sizes = set()
    for group in group_metadata:
        if group.get("kind") in MAIN_ATTENTION_KV_CACHE_KINDS:
            block_size = group.get("logical_block_size")
            if type(block_size) is not int or block_size <= 0:
                raise ValueError(
                    "vLLM main-attention KV cache group requires a positive "
                    "integer logical_block_size"
                )
            block_sizes.add(block_size)

    if len(block_sizes) != 1:
        raise ValueError(
            "vLLM KV cache metadata must report one shared main-attention "
            "logical_block_size"
        )
    return block_sizes.pop()


async def configure_kv_event_block_size(
    engine: AsyncLLM,
    vllm_config: VllmConfig,
) -> int:
    """Fetch engine cache-group metadata and cache the KV event block size on vLLM config."""
    group_metadata = await engine.get_kv_cache_group_metadata()
    kv_event_block_size = select_main_attention_block_size(
        group_metadata,
        vllm_config.cache_config.block_size,
    )

    if vllm_config.additional_config is None:
        vllm_config.additional_config = {}
    vllm_config.additional_config[DYNAMO_KV_EVENT_BLOCK_SIZE_KEY] = kv_event_block_size
    return kv_event_block_size
