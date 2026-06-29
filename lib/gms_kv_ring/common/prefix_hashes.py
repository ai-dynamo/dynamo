# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-engine prefix-block hash function.

A `content_hash` in GMS uniquely identifies "the bytes that should
be in a KV slot for a given prefix of tokens." It is the only
identifier the daemon uses to route cross-node transfers — the
sender's daemon and the receiver's daemon must agree on a hash for
the same prefix regardless of which engine produced it.

To make that hash stable across engines, both vLLM and SGLang
connectors compute it via this single shared function. Changing
it requires bumping a version byte AND ensuring both connectors
upgrade in lockstep.

Hash construction (version `GMSKVRv1`):

    h_0 = sha256("GMSKVRv1\\x00" || BE_u32(len(salt)) || salt
                 || BE_u32(block_size))
    h_i = sha256(h_{i-1}_state || BE_u64*block_size(token_ids[i*B:(i+1)*B]))

i.e., a cumulative running sha256 over (salt, block_size, prefix
tokens) chunked into blocks. The returned list contains one digest
per FULL block in the prefix — partial trailing blocks are not
emitted because they cannot be cache-hit externally (vLLM and
SGLang both match whole blocks).
"""

from __future__ import annotations

import hashlib
import struct
from typing import Optional


def prefix_block_hashes(
    token_ids: "list[int]",
    block_size: int,
    cache_salt: Optional[str],
) -> "list[bytes]":
    """Per-block cumulative prefix sha256 hashes.

    Returns one hash per FULL block (partial trailing blocks
    excluded). Each hash covers `(cache_salt, token_ids[:i*block_size])`
    for `i in 1..n_full_blocks`.

    The salt is mixed in so two requests with the same tokens but
    different `cache_salt` won't cross-hit (matches vLLM's
    prefix-cache isolation semantics).

    Determinism: the function is pure over `(token_ids, block_size,
    cache_salt)`. Use the exact same inputs on the receiver side to
    look up bytes by hash."""
    if block_size <= 0:
        return []
    salt = (cache_salt or "").encode("utf-8")
    n_full = len(token_ids) // block_size
    if n_full == 0:
        return []
    hashes: list[bytes] = []
    h = hashlib.sha256()
    h.update(b"GMSKVRv1\x00")
    h.update(struct.pack(">I", len(salt)))
    h.update(salt)
    h.update(struct.pack(">I", int(block_size)))
    for i in range(n_full):
        chunk = token_ids[i * block_size : (i + 1) * block_size]
        h.update(struct.pack(f">{block_size}Q", *chunk))
        hashes.append(h.digest())
    return hashes
