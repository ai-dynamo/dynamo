# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Utilities for building multimodal routing metadata from vLLM's mm_features.

Converts vLLM's EngineCoreRequest.mm_features (mm_hashes + mm_placeholders)
into the block_mm_infos format expected by dynamo's KvRouter.
"""

from __future__ import annotations

from typing import Any


def build_block_mm_infos(
    num_tokens: int,
    block_size: int,
    mm_hashes: list[int] | None,
    image_ranges: list[tuple[int, int]] | None,
) -> list[dict | None] | None:
    """Build per-block mm_info for routing.

    For each KV cache block, identifies which multimodal items overlap with it
    and records their mm_hash. This is model-agnostic — it only uses token
    offsets and block boundaries.

    Args:
        num_tokens: Total token count in the sequence.
        block_size: Number of tokens per KV cache block.
        mm_hashes: Integer hash per multimodal item (truncated from hex).
        image_ranges: (start, end) token index per multimodal item.

    Returns:
        List of per-block dicts (or None for blocks without MM content).
    """
    if not mm_hashes or not image_ranges or len(mm_hashes) != len(image_ranges):
        return None

    num_blocks = (num_tokens + block_size - 1) // block_size
    result = []

    for block_idx in range(num_blocks):
        block_start = block_idx * block_size
        block_end = block_start + block_size

        # Find images overlapping this block
        mm_objects = [
            {"mm_hash": mm_hash, "offsets": []}
            for mm_hash, (img_start, img_end) in zip(mm_hashes, image_ranges)
            # FIXME: Revisit the bounds checks here
            # https://github.com/ai-dynamo/dynamo/issues/6588
            if block_end > img_start and block_start <= img_end
        ]

        result.append({"mm_objects": mm_objects} if mm_objects else None)

    return result


def build_mm_routing_info_from_features(
    mm_features: list[Any],
    prompt_token_ids: list[int],
    block_size: int,
) -> dict | None:
    """Convert vLLM's mm_features to KvRouter mm_routing_info.

    Extracts mm_hashes and placeholder ranges from MultiModalFeatureSpec
    objects (produced by vLLM's InputProcessor.process_inputs()) and builds
    the block-level multimodal metadata needed for KV-cache-aware routing.

    Args:
        mm_features: List of MultiModalFeatureSpec from EngineCoreRequest.
        prompt_token_ids: The expanded prompt token IDs.
        block_size: KV cache block size (must match backend).

    Returns:
        Dict with ``routing_token_ids`` and ``block_mm_infos``, or None
        if no valid MM features are present.
    """
    if not mm_features:
        return None

    mm_hashes: list[int] = []
    image_ranges: list[tuple[int, int]] = []

    for f in mm_features:
        if f.mm_hash is None:
            continue
        # Convert hex hash string to 64-bit integer (matching KvRouter's u64)
        mm_hashes.append(int(f.mm_hash[:16], 16))
        offset = f.mm_position.offset
        length = f.mm_position.length
        image_ranges.append((offset, offset + length))

    if not mm_hashes:
        return None

    num_tokens = len(prompt_token_ids)
    block_mm_infos = build_block_mm_infos(
        num_tokens, block_size, mm_hashes, image_ranges
    )

    if block_mm_infos is None:
        return None

    return {
        "routing_token_ids": prompt_token_ids,
        "block_mm_infos": block_mm_infos,
    }
