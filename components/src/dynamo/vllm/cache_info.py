# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.engine.async_llm import AsyncLLM

logger = logging.getLogger(__name__)

DYNAMO_KV_EVENT_BLOCK_SIZE_KEY = "dynamo_kv_event_block_size"
DYNAMO_KV_CACHE_GROUP_COUNT_KEY = "dynamo_kv_cache_group_count"
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
    if not isinstance(group_metadata, list):
        return fallback_block_size

    for group in group_metadata:
        kind = group.get("kind") if isinstance(group, dict) else None
        if isinstance(kind, str) and kind in MAIN_ATTENTION_KV_CACHE_KINDS:
            block_size = group.get("block_size")
            if type(block_size) is int and block_size > 0:
                return block_size

    return fallback_block_size


def _validated_group_count(group_metadata: Any) -> int | None:
    if not isinstance(group_metadata, list) or not 0 < len(group_metadata) <= 256:
        return None
    for index, group in enumerate(group_metadata):
        if (
            not isinstance(group, dict)
            or type(group.get("group_idx")) is not int
            or group["group_idx"] != index
            or not isinstance(group.get("kind"), str)
            or not group["kind"]
            or type(group.get("block_size")) is not int
            or group["block_size"] <= 0
        ):
            return None
    return len(group_metadata)


async def configure_kv_event_block_size(
    engine: AsyncLLM,
    vllm_config: VllmConfig,
) -> int:
    """Cache engine-derived block size and group count before worker registration."""
    fallback_block_size = vllm_config.cache_config.block_size
    if vllm_config.additional_config is None:
        vllm_config.additional_config = {}
    # This value authorizes optional reuse; configuration input is not evidence.
    vllm_config.additional_config.pop(DYNAMO_KV_CACHE_GROUP_COUNT_KEY, None)
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
        group_count = _validated_group_count(group_metadata)
        if group_count is not None:
            vllm_config.additional_config[DYNAMO_KV_CACHE_GROUP_COUNT_KEY] = group_count
        kv_event_block_size = select_main_attention_block_size(
            group_metadata,
            fallback_block_size,
        )

    vllm_config.additional_config[DYNAMO_KV_EVENT_BLOCK_SIZE_KEY] = kv_event_block_size
    return kv_event_block_size
