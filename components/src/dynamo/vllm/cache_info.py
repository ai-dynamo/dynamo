# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any

from vllm.config import VllmConfig
from vllm.v1.engine.async_llm import AsyncLLM

logger = logging.getLogger(__name__)

DYNAMO_KV_EVENT_BLOCK_SIZE_KEY = "dynamo_kv_event_block_size"
MAIN_ATTENTION_KV_CACHE_KINDS = {
    "full_attention",
    "mla_attention",
    "sink_full_attention",
}


def get_configured_kv_event_block_size(vllm_config: VllmConfig) -> int:
    additional_config = vllm_config.additional_config or {}
    return additional_config.get(
        DYNAMO_KV_EVENT_BLOCK_SIZE_KEY,
        vllm_config.cache_config.block_size,
    )


def select_main_attention_block_size(
    group_metadata: list[dict[str, Any]],
    fallback_block_size: int,
) -> int:
    if not group_metadata:
        return fallback_block_size

    fallback = group_metadata[0].get("block_size", fallback_block_size)
    for group in group_metadata:
        if group.get("kind") in MAIN_ATTENTION_KV_CACHE_KINDS:
            return group.get("block_size", fallback_block_size)

    return fallback


async def configure_kv_event_block_size(
    engine: AsyncLLM,
    vllm_config: VllmConfig,
) -> int:
    fallback_block_size = vllm_config.cache_config.block_size
    try:
        group_metadata = await engine.engine_core.call_utility_async(
            "get_kv_cache_group_metadata"
        )
        kv_event_block_size = select_main_attention_block_size(
            group_metadata,
            fallback_block_size,
        )
    except Exception:
        logger.exception(
            "Failed to resolve main-attention KV event block size; "
            "falling back to vLLM cache_config.block_size"
        )
        kv_event_block_size = fallback_block_size

    vllm_config.additional_config[DYNAMO_KV_EVENT_BLOCK_SIZE_KEY] = kv_event_block_size
    logger.info(
        "Resolved Dynamo KV event block size: scheduler_block_size=%s, "
        "kv_event_block_size=%s",
        fallback_block_size,
        kv_event_block_size,
    )
    return kv_event_block_size
