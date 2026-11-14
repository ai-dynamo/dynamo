# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from dataclasses import dataclass
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


@dataclass(frozen=True)
class RecurrentKvLayout:
    """The fixed-common-unit vLLM resume contract supported by this publisher."""

    recurrent_groups: list[int]
    cache_groups: list[dict[str, Any]]
    indexer_block_size: int
    hash_block_size: int
    scheduler_block_size: int
    cache_hit_alignment_tokens: int

    @property
    def alignment_blocks(self) -> int:
        return self.cache_hit_alignment_tokens // self.indexer_block_size

    def runtime_metadata(self) -> dict[str, Any]:
        return {
            "contract": "vllm_fixed_common_unit_v1",
            "geometry_source": "validated_common_unit",
            "recurrent_groups": self.recurrent_groups,
            "alignment_blocks": self.alignment_blocks,
            "indexer_block_size": self.indexer_block_size,
            "hash_block_size": self.hash_block_size,
            "scheduler_block_size": self.scheduler_block_size,
            "cache_hit_alignment_tokens": self.cache_hit_alignment_tokens,
            "cache_groups": self.cache_groups,
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
    layout = recurrent_kv_layout(vllm_config, enabled=enabled)
    return layout.recurrent_groups if layout is not None else None


def recurrent_kv_layout(
    vllm_config: VllmConfig, *, enabled: bool = False
) -> RecurrentKvLayout | None:
    """Validate the supported engine resume geometry before enabling publication.

    The group utility reports physical block sizes, not the coordinator's resolved
    hash and hit units. Equal group sizes plus the explicit configuration checks
    below make those units identical; unsupported or unknown geometries fail closed.
    """
    if not enabled:
        return None
    metadata = (vllm_config.additional_config or {}).get(DYNAMO_KV_CACHE_GROUPS_KEY)
    if not isinstance(metadata, list) or not metadata:
        raise ValueError("Recurrent KV events require engine KV cache group metadata")
    if any(not isinstance(group, dict) for group in metadata):
        raise ValueError("Recurrent KV events require engine KV cache group metadata")
    groups = [group.get("group_idx") for group in metadata]
    if any(
        type(group) is not int or not 0 <= group <= 0xFFFFFFFF for group in groups
    ) or len(set(groups)) != len(groups):
        raise ValueError("KV cache group ids must be unique unsigned integers")
    supported_kinds = MAIN_ATTENTION_KV_CACHE_KINDS | {"mamba"}
    if any(
        not isinstance(group.get("kind"), str) or group["kind"] not in supported_kinds
        for group in metadata
    ):
        raise ValueError(
            "Recurrent KV events support only full-attention and Mamba groups; "
            "sliding-window, chunked-local, and unknown groups are unsupported"
        )
    recurrent_groups = [
        group["group_idx"] for group in metadata if group["kind"] == "mamba"
    ]
    if not recurrent_groups:
        return None
    if not any(group["kind"] in MAIN_ATTENTION_KV_CACHE_KINDS for group in metadata):
        raise ValueError("Recurrent KV events require a full-attention cache group")

    sizes = [group.get("block_size") for group in metadata]
    if any(type(size) is not int or size <= 0 for size in sizes):
        raise ValueError("Recurrent KV events require positive group block sizes")
    unit = get_configured_kv_event_block_size(vllm_config)
    if type(unit) is not int or unit <= 0 or any(size != unit for size in sizes):
        raise ValueError(
            "Recurrent KV events require every cache group and the indexer to use "
            "one common block size; mixed or finer-than-indexer geometry is unsupported"
        )

    # These opt-in capabilities must be known explicitly. Older engine versions
    # without them cannot establish the resume contract from physical sizes alone.
    try:
        cache = vllm_config.cache_config
        prefix_match_unit = cache.prefix_match_unit
        prefix_caching = cache.enable_prefix_caching
        mamba_cache_mode = cache.mamba_cache_mode
        parallel = vllm_config.parallel_config
        dcp_size = parallel.decode_context_parallel_size
        pcp_size = parallel.prefill_context_parallel_size
        speculative = vllm_config.speculative_config
        transfer = vllm_config.kv_transfer_config
    except AttributeError as error:
        raise ValueError(
            "Recurrent KV events require vLLM prefix-match, cache-mode, parallel, "
            "speculation, and connector capability fields"
        ) from error
    if prefix_caching is not True or mamba_cache_mode != "align":
        raise ValueError(
            "Recurrent KV events require prefix caching and mamba_cache_mode='align'"
        )
    if prefix_match_unit is not None and (
        type(prefix_match_unit) is not int or prefix_match_unit != unit
    ):
        raise ValueError(
            "Recurrent KV events require prefix_match_unit to equal the indexer "
            "block size; finer hash geometry is unsupported"
        )
    if dcp_size != 1 or pcp_size != 1:
        raise ValueError("Recurrent KV events do not yet support context parallelism")
    if speculative is not None:
        raise ValueError(
            "Recurrent KV events do not yet support EAGLE or speculative decoding"
        )
    if transfer is not None:
        extra = transfer.kv_connector_extra_config
        if (
            transfer.kv_connector != "OffloadingConnector"
            or extra.get("self_describing_kv_events") is not True
            or type(extra.get("blocks_per_chunk")) is not int
            or extra["blocks_per_chunk"] != 1
            or "block_size" in extra
            or extra.get("spec_name") not in (None, "CPUOffloadingSpec")
        ):
            raise ValueError(
                "Recurrent KV events support only OffloadingConnector with "
                "self_describing_kv_events=true, blocks_per_chunk=1, and the "
                "default CPU offloading spec"
            )

    # With no CP, no speculative tail, equal physical group sizes and no finer
    # prefix-match override, vLLM's scheduler LCM, hash GCD and hit alignment
    # are all this same unit. This is intentionally narrower than general HMA.
    return RecurrentKvLayout(
        recurrent_groups=recurrent_groups,
        cache_groups=metadata,
        indexer_block_size=unit,
        hash_block_size=unit,
        scheduler_block_size=unit,
        cache_hit_alignment_tokens=unit,
    )


def recurrent_alignment_blocks(vllm_config: VllmConfig) -> int:
    """Return alignment only after validating the fixed-common-unit contract."""
    layout = recurrent_kv_layout(vllm_config, enabled=True)
    return layout.alignment_blocks if layout is not None else 1
