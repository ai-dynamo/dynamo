# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any

from vllm.config import VllmConfig
from vllm.v1.engine.async_llm import AsyncLLM

logger = logging.getLogger(__name__)

DYNAMO_KV_EVENT_BLOCK_SIZE_KEY = "dynamo_kv_event_block_size"
PREFILL_KV_ALLOCATION_KEY = "prefill_kv_allocation"
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


def _prefill_kv_allocation(
    groups: list[dict[str, Any]], vllm_config: VllmConfig, block_size: int
) -> dict[str, int | None] | None:
    # Block counts are only comparable to the advertised pool for a single
    # cache group with matching geometry.
    if len(groups) != 1 or groups[0].get("block_size") != block_size:
        return None
    group = groups[0]
    kind = group.get("kind")
    window = group.get("sliding_window")
    if kind == "full_attention" and window is None:
        return {
            "block_size": block_size,
            "max_admission_blocks": None,
            "reserved_blocks": 1,
        }
    if (
        kind != "sliding_window"
        or type(window) is not int
        or window <= 0
        or vllm_config.speculative_config is not None
    ):
        return None
    # vLLM's sliding-window admission needs the live window, in-flight
    # tokens and a block for unaligned window placement. Speculative layouts
    # can retain extra tokens not included in the group metadata.
    pending_tokens = getattr(vllm_config, "max_in_flight_tokens", None)
    if type(pending_tokens) is not int or pending_tokens <= 0:
        return None
    resident = min(window - 1 + pending_tokens, vllm_config.model_config.max_model_len)
    return {
        "block_size": block_size,
        # vLLM excludes the permanently allocated null block from usable KV.
        "reserved_blocks": 1,
        "max_admission_blocks": (resident + block_size - 1) // block_size + 1,
    }


async def configure_kv_event_block_size(
    engine: AsyncLLM,
    vllm_config: VllmConfig,
) -> int:
    """Fetch engine cache-group metadata and cache the KV event block size on vLLM config."""
    fallback_block_size = vllm_config.cache_config.block_size
    allocation = None
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
        allocation = _prefill_kv_allocation(
            group_metadata, vllm_config, kv_event_block_size
        )

    if vllm_config.additional_config is None:
        vllm_config.additional_config = {}
    vllm_config.additional_config[DYNAMO_KV_EVENT_BLOCK_SIZE_KEY] = kv_event_block_size
    vllm_config.additional_config.pop(PREFILL_KV_ALLOCATION_KEY, None)
    if allocation is not None:
        vllm_config.additional_config[PREFILL_KV_ALLOCATION_KEY] = allocation
    return kv_event_block_size
