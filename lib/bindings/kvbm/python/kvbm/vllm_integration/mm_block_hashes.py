# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    """Return a stable u64 hash of arbitrary feature data."""
    # vLLM MultiModalFeatureSpec: prefer identifier (per-item cache key) and
    # tensor data; avoid str(feature) which is not a stable unique key for VLMs.
    if features is not None and hasattr(features, "identifier"):
        ident = getattr(features, "identifier", None)
        data = getattr(features, "data", None)
        if ident is not None or data is not None:
            h = hashlib.sha256()
            if ident is not None:
                h.update(str(ident).encode())
            if data is not None:
                _update_hash(h, data)
            return struct.unpack("<Q", h.digest()[:8])[0]

    h = hashlib.sha256()
    _update_hash(h, features)
    return struct.unpack("<Q", h.digest()[:8])[0]


def _update_hash(h: "hashlib._Hash", obj: Any) -> None:
    """Recursively feed obj into the running SHA-256 digest."""
    try:
        import torch

        if isinstance(obj, torch.Tensor):
            h.update(obj.cpu().contiguous().numpy().tobytes())
            return
    except ImportError:
        pass

    if isinstance(obj, (bytes, bytearray)):
        h.update(obj)
    elif isinstance(obj, dict):
        for k in sorted(obj.keys(), key=str):
            h.update(str(k).encode())
            _update_hash(h, obj[k])
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            _update_hash(h, item)
    else:
        h.update(str(obj).encode())


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

    # vLLM 0.18+: Request no longer has top-level mm_positions; each
    # MultiModalFeatureSpec carries PlaceholderRange as mm_position.
    if not mm_positions and mm_features:
        derived = [getattr(f, "mm_position", None) for f in mm_features]
        mm_positions = [p for p in derived if p is not None]

    if not mm_positions:
        return result

    features_list = list(mm_features) if mm_features else []

    for i, pos_range in enumerate(mm_positions):
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
