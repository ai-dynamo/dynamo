# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.engine.async_llm import AsyncLLM

logger = logging.getLogger(__name__)

DYNAMO_KV_CACHE_GROUPS_KEY = "dynamo_kv_cache_groups"
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
    group_metadata: list[dict[str, Any]],
    fallback_block_size: int,
) -> int:
    """Select the main-attention KV block size from engine cache-group metadata."""
    if not group_metadata:
        return fallback_block_size

    for group in group_metadata:
        if group.get("kind") in MAIN_ATTENTION_KV_CACHE_KINDS:
            return group.get("block_size", fallback_block_size)

    return fallback_block_size


async def configure_kv_event_block_size(
    engine: AsyncLLM,
    vllm_config: VllmConfig,
) -> int:
    """Fetch engine cache-group metadata and cache the KV event block size on vLLM config."""
    fallback_block_size = vllm_config.cache_config.block_size
    group_metadata = None
    try:
        group_metadata = await engine.engine_core.call_utility_async(
            "get_kv_cache_group_metadata"
        )
    except Exception as e:
        logger.warning(
            "Failed to fetch KV cache group metadata; falling back to "
            "vLLM cache_config.block_size: %s",
            e,
        )
        kv_event_block_size = fallback_block_size
    else:
        kv_event_block_size = select_main_attention_block_size(
            group_metadata,
            fallback_block_size,
        )

    if vllm_config.additional_config is None:
        vllm_config.additional_config = {}
    vllm_config.additional_config[DYNAMO_KV_EVENT_BLOCK_SIZE_KEY] = kv_event_block_size
    vllm_config.additional_config[DYNAMO_KV_CACHE_GROUPS_KEY] = group_metadata
    return kv_event_block_size


def recurrent_kv_groups(
    vllm_config: VllmConfig, *, enabled: bool = False
) -> list[int] | None:
    """Return the complete recurrent layout for an explicitly enabled publisher."""
    if not enabled:
        return None
    metadata = (vllm_config.additional_config or {}).get(DYNAMO_KV_CACHE_GROUPS_KEY)
    if not isinstance(metadata, list) or not metadata:
        raise ValueError("Recurrent KV events require engine KV cache group metadata")
    groups = [group["group_idx"] for group in metadata if group.get("kind") == "mamba"]
    if any(
        type(group) is not int or not 0 <= group <= 0xFFFFFFFF for group in groups
    ) or len(set(groups)) != len(groups):
        raise ValueError(
            "Recurrent KV cache group ids must be unique unsigned integers"
        )
    return groups


def recurrent_alignment_blocks(vllm_config: VllmConfig) -> int:
    """Align reuse to the least common multiple of the engine's group block sizes."""
    from math import lcm

    metadata = (vllm_config.additional_config or {}).get(
        DYNAMO_KV_CACHE_GROUPS_KEY
    ) or []
    sizes = [group["block_size"] for group in metadata]
    if not sizes or any(type(size) is not int or size <= 0 for size in sizes):
        raise ValueError("Recurrent KV events require positive group block sizes")
    unit = get_configured_kv_event_block_size(vllm_config)
    alignment = lcm(*sizes)
    if alignment % unit:
        raise ValueError(
            "KV cache group alignment must be divisible by the event block size"
        )
    return alignment // unit
