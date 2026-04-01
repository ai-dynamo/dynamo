# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Helper to compute per-block extra hashes for multimodal (image+text) requests.

vLLM's native prefix caching shares text-only prefix blocks across requests and
uniquifies blocks that contain multimodal (e.g. image) tokens. KVBM replicates
this by attaching a stable per-image hash to every block that overlaps multimodal
token positions.  Text-only blocks get None (sharable); multimodal blocks get a
u64 derived from SHA-256 of the image features.
"""

from __future__ import annotations

import hashlib
import math
import struct
from typing import Any, Optional


def _hash_features_to_u64(features: Any) -> int:
    """Return a stable u64 hash for a multimodal feature.

    Aligned with vLLM's native prefix cache which uses
    ``request.mm_hashes[i]`` (== ``feature.identifier``) as the sole
    extra key for block-hash computation.  ``identifier`` is already a
    content-addressable blake3 hex digest of the raw image pixels,
    model id, and processor kwargs, so re-hashing the tensor data is
    redundant.

    Falls back to ``str(features)`` for objects without ``identifier``.
    """
    ident = getattr(features, "identifier", None) if features is not None else None
    if ident is not None:
        h = hashlib.sha256(str(ident).encode())
        return struct.unpack("<Q", h.digest()[:8])[0]

    # Fallback for non-standard feature objects.
    h = hashlib.sha256(str(features).encode())
    return struct.unpack("<Q", h.digest()[:8])[0]


def compute_mm_block_hashes(
    mm_positions: Any,
    mm_features: Any,
    num_tokens: int,
    block_size: int,
) -> list[Optional[int]]:
    """
    Return a list of length ``ceil(num_tokens / block_size)``.

    Each entry is:
    - ``None``  – the block contains only text tokens (prefix-sharable).
    - ``int``   – a stable u64 hash of the image(s) whose tokens overlap this
                  block (prevents cross-request sharing for different images).

    Parameters
    ----------
    mm_positions:
        Optional list of ``PlaceholderRange`` (``offset`` / ``length``). Older vLLM
        exposes this on ``request.mm_positions``; vLLM 0.18+ embeds the same
        range on each ``MultiModalFeatureSpec`` as ``mm_position`` — this
        function derives positions from ``mm_features`` when the top-level list
        is missing.
    mm_features:
        ``request.mm_features`` – a sequence of feature objects (one per
        multimodal item). Prefer ``MultiModalFeatureSpec.identifier`` (and
        ``data`` when present) for hashing.
    num_tokens:
        Total number of tokens in the request (``len(request.all_token_ids)``).
    block_size:
        KV-cache block size in tokens.
    """
    num_blocks = math.ceil(num_tokens / block_size) if num_tokens > 0 else 0
    result: list[Optional[int]] = [None] * num_blocks

    # mm_features is always present for multimodal requests.
    # mm_positions is a legacy field that older vLLM provides as a top-level
    # list; vLLM 0.18+ embeds the position on each MultiModalFeatureSpec
    # instead.  Derive mm_positions from features when not provided.
    features_list = list(mm_features) if mm_features else []

    if not mm_positions:
        mm_positions = [getattr(f, "mm_position", None) for f in features_list]

    if not mm_positions:
        return result

    for i, pos_range in enumerate(mm_positions):
        if pos_range is None:
            continue
        feature = features_list[i] if i < len(features_list) else None
        feature_hash = _hash_features_to_u64(feature) if feature is not None else 0

        start_token: int = pos_range.offset
        end_token: int = pos_range.offset + pos_range.length
        if end_token <= start_token:
            continue

        start_block = start_token // block_size
        end_block = (end_token - 1) // block_size + 1

        for block_idx in range(start_block, min(end_block, num_blocks)):
            existing = result[block_idx]
            if existing is None:
                result[block_idx] = feature_hash
            else:
                # Multiple images in the same block – combine hashes.
                combined = hashlib.sha256(
                    struct.pack("<QQ", existing, feature_hash)
                ).digest()
                result[block_idx] = struct.unpack("<Q", combined[:8])[0]

    return result
