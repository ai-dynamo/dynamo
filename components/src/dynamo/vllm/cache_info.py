# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.engine.async_llm import AsyncLLM

logger = logging.getLogger(__name__)

DYNAMO_KV_EVENT_BLOCK_SIZE_KEY = "dynamo_kv_event_block_size"
MAIN_ATTENTION_KV_CACHE_KINDS = {
    "full_attention",
    "mla_attention",
    "sink_full_attention",
}


def _find_main_attention_block_size(group_metadata: list[dict[str, Any]]) -> int | None:
    """Return the block_size of the first main-attention cache group, or None if absent."""
    for group in group_metadata:
        if group.get("kind") in MAIN_ATTENTION_KV_CACHE_KINDS:
            return group.get("block_size")
    return None


def _validate_kv_event_block_size(value: Any) -> int:
    """Validate an operator-supplied dynamo_kv_event_block_size override.

    Preserves the strict, fail-at-startup validation that the removed
    DYN_VLLM_KV_EVENT_BLOCK_SIZE env var provided, so a bad override fails
    loudly instead of silently falling back to cache_config.block_size.
    """
    # bool is a subclass of int, so reject it explicitly.
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            f"{DYNAMO_KV_EVENT_BLOCK_SIZE_KEY} must be a positive integer, "
            f"got {value!r}."
        )
    return value


def get_configured_kv_event_block_size(vllm_config: "VllmConfig") -> int:
    """Return the configured KV event block size, falling back to vLLM's cache block size."""
    additional_config = vllm_config.additional_config or {}
    if DYNAMO_KV_EVENT_BLOCK_SIZE_KEY in additional_config:
        return _validate_kv_event_block_size(
            additional_config[DYNAMO_KV_EVENT_BLOCK_SIZE_KEY]
        )
    return vllm_config.cache_config.block_size


def select_main_attention_block_size(
    group_metadata: list[dict[str, Any]],
    fallback_block_size: int,
) -> int:
    """Select the main-attention KV block size from engine cache-group metadata."""
    block_size = _find_main_attention_block_size(group_metadata)
    return fallback_block_size if block_size is None else block_size


async def configure_kv_event_block_size(
    engine: "AsyncLLM",
    vllm_config: "VllmConfig",
    *,
    require_exact_match: bool = False,
) -> int:
    """Fetch engine cache-group metadata and cache the KV event block size on vLLM config."""
    additional_config = vllm_config.additional_config or {}
    if DYNAMO_KV_EVENT_BLOCK_SIZE_KEY in additional_config:
        kv_event_block_size = _validate_kv_event_block_size(
            additional_config[DYNAMO_KV_EVENT_BLOCK_SIZE_KEY]
        )
    else:
        fallback_block_size = vllm_config.cache_config.block_size
        try:
            group_metadata = await engine.engine_core.call_utility_async(
                "get_kv_cache_group_metadata"
            )
        except Exception as e:
            # Metadata API unavailable means an older vLLM build where
            # block_size is always the correct fallback. Warn so users can
            # set dynamo_kv_event_block_size in --additional-config if the
            # fallback is wrong.
            logger.warning(
                "Failed to fetch KV cache group metadata; falling back to "
                "vLLM cache_config.block_size=%d. If this is incorrect, set "
                "dynamo_kv_event_block_size in --additional-config: %s",
                fallback_block_size,
                e,
            )
            kv_event_block_size = fallback_block_size
        else:
            matched_block_size = _find_main_attention_block_size(group_metadata)
            if require_exact_match and matched_block_size is None:
                raise RuntimeError(
                    "Failed to determine the vLLM KV event block size from cache "
                    "group metadata. Set dynamo_kv_event_block_size in "
                    "--additional-config to an explicit value or run with a vLLM "
                    "build that reports a main-attention cache group block_size."
                )
            kv_event_block_size = select_main_attention_block_size(
                group_metadata,
                fallback_block_size,
            )

    if vllm_config.additional_config is None:
        vllm_config.additional_config = {}
    vllm_config.additional_config[DYNAMO_KV_EVENT_BLOCK_SIZE_KEY] = kv_event_block_size
    return kv_event_block_size
