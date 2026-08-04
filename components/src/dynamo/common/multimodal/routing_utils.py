# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Utilities for building multimodal routing metadata from vLLM's mm_features.

Converts vLLM's EngineCoreRequest.mm_features (mm_hashes + mm_placeholders)
into the canonical pad_value routing scheme used by dynamo's KvRouter: image
positions in the routing token stream are replaced with pad_value(mm_hash), and
no block_mm_infos side channel is emitted. This matches the Rust frontend
(`gather_mm_exact_routing_info`) and the kv-router's vLLM-event normalization,
so request-side and event-side block hashes agree across backends.
"""

from __future__ import annotations

from typing import Any

# Mirror of sglang's MultimodalItem._compute_pad_value, also pinned in Rust
# (dynamo_kv_router::protocols::pad_value_for_mm_hash). Must stay in lockstep.
MM_PAD_SHIFT_VALUE = 1_000_000
MM_PAD_HASH_MASK = (1 << 30) - 1


def pad_value_for_mm_hash(mm_hash: int) -> int:
    """Canonical per-image pad_value from a routing-side mm_hash (low 30 bits)."""
    return MM_PAD_SHIFT_VALUE + (mm_hash & MM_PAD_HASH_MASK)


def _is_embed_mask(mm_position: Any) -> list[bool] | None:
    is_embed = getattr(mm_position, "is_embed", None)
    if is_embed is None:
        return None
    if hasattr(is_embed, "detach"):
        is_embed = is_embed.detach().cpu().tolist()
    elif hasattr(is_embed, "tolist"):
        is_embed = is_embed.tolist()
    return [bool(value) for value in is_embed]


def build_mm_routing_info_from_features(
    mm_features: list[Any],
    prompt_token_ids: list[int],
) -> dict | None:
    """Convert vLLM's mm_features to KvRouter mm_routing_info (pad_value scheme).

    Extracts mm_hashes and placeholder ranges from MultiModalFeatureSpec
    objects (produced by vLLM's InputProcessor.process_inputs()) and rewrites
    the routing token stream so each image's positions hold pad_value(mm_hash).
    No block_mm_infos is emitted — mm identity rides in the tokens, matching the
    Rust frontend and the kv-router's event normalization.

    Args:
        mm_features: List of MultiModalFeatureSpec from EngineCoreRequest.
        prompt_token_ids: The expanded prompt token IDs.

    Returns:
        Dict with ``routing_token_ids`` (pad_value-substituted) and an empty
        ``block_mm_infos``, or None if no valid MM features are present.
    """
    if not mm_features:
        return None

    routing_token_ids = list(prompt_token_ids)
    n_tokens = len(routing_token_ids)
    found = False

    for f in mm_features:
        if f.mm_hash is None:
            continue
        # Hex hash string -> 64-bit int (matching KvRouter's u64), then pad_value.
        mm_hash = int(f.mm_hash[:16], 16)
        pad = pad_value_for_mm_hash(mm_hash)
        start = f.mm_position.offset
        end = min(start + f.mm_position.length, n_tokens)
        is_embed = _is_embed_mask(f.mm_position)
        if is_embed is None:
            for i in range(start, end):
                routing_token_ids[i] = pad
                found = True
        else:
            for rel_idx, use_pad in enumerate(is_embed[: end - start]):
                if use_pad:
                    routing_token_ids[start + rel_idx] = pad
                    found = True

    if not found:
        return None

    return {
        "routing_token_ids": routing_token_ids,
        "block_mm_infos": [],
        "expanded_prompt_len": n_tokens,
    }
