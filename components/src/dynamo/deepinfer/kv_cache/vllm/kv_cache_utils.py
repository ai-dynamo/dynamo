# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Adapted from vllm vllm/v1/core/kv_cache_utils.py
# @ 82a24bb82dd9344cce7a43195bb51a2390adf00a.
# Excerpt: block hash types, NONE_HASH seeding and hash_block_tokens, verbatim;
# vllm.* imports are replaced by the local stubs below.
"""KV-Cache Utilities."""

import logging
import os
from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Any, NewType, TypeAlias

from .hashing import xxhash, xxhash_cbor

# Stubs for vllm.envs and vllm.logger.init_logger.
envs = SimpleNamespace(VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=True)
init_logger = logging.getLogger

# BlockHash represents the hash of a single KV-cache block used for
# prefix caching.  Treating it as a distinct type from `bytes` helps
# catch accidental misuse when passing around raw byte strings.
BlockHash = NewType("BlockHash", bytes)

# `BlockHashWithGroupId` combines a `BlockHash` with its KV cache group ID.
# It is represented as raw bytes for compactness and efficiency. The helper
# functions below pack/unpack the `BlockHash` and group id into/from the key.
BlockHashWithGroupId = NewType("BlockHashWithGroupId", bytes)

# ExternalBlockHash is used for reproducible prefix-cache block hashing.
# It's a union of `bytes` and `int` to keep backward compatibility
# after we default block hashing to use sha256 bytes.
ExternalBlockHash: TypeAlias = bytes | int


def make_block_hash_with_group_id(
    block_hash: BlockHash, group_id: int
) -> BlockHashWithGroupId:
    """Pack a `BlockHash` and group id into a `BlockHashWithGroupId`.

    The group id is encoded using 4 bytes in big-endian order and appended to
    the block hash bytes.  This representation avoids creating tuples while
    still allowing us to recover both components when needed.
    """
    return BlockHashWithGroupId(block_hash + group_id.to_bytes(4, "big", signed=False))


def get_block_hash(key: BlockHashWithGroupId) -> BlockHash:
    """Extract the `BlockHash` from a `BlockHashWithGroupId`."""
    return BlockHash(key[:-4])


def get_group_id(key: BlockHashWithGroupId) -> int:
    """Extract the group id from a `BlockHashWithGroupId`."""
    return int.from_bytes(key[-4:], "big", signed=False)


def maybe_convert_block_hash(hash_bytes: BlockHash) -> ExternalBlockHash:
    if not envs.VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES:
        return hash_bytes
    return int.from_bytes(hash_bytes, byteorder="big") & ((1 << 64) - 1)


logger = init_logger(__name__)

# The hash seed for the first block of any prefix block sequence.
#
# For cryptographic hash algorithms it is derived deterministically from a fixed
# default seed, so independent vLLM processes compute identical block hashes for
# identical content and can share a prefix cache (e.g. KV cache reuse across
# nodes) without extra configuration. This does not weaken collision resistance,
# which for SHA-256 does not depend on keeping the seed secret; ``cache_salt``
# remains the mechanism for intentional cache isolation.
#
# Non-cryptographic algorithms keep a per-process random seed, because a
# predictable seed would let an attacker precompute colliding blocks offline
# (see #12621). Setting PYTHONHASHSEED overrides the seed in both cases.
#
# The function `init_none_hash` initializes this variable globally.
NONE_HASH: BlockHash

# Fixed seed used when the PYTHONHASHSEED environment variable is not set and
# the hash algorithm is cryptographic.
DEFAULT_NONE_HASH_SEED = "vllm-none-hash"

# Algorithms that are not collision resistant, so the seed must stay secret.
_NON_CRYPTO_HASH_FUNCTIONS = frozenset({xxhash, xxhash_cbor})

# The seed NONE_HASH was derived from, set by init_none_hash.
_NONE_HASH_SEED: str | None = None


def resolve_none_hash_seed(hash_fn: Callable[[Any], bytes]) -> str:
    """Resolve the seed to derive NONE_HASH from.

    PYTHONHASHSEED wins if set. Otherwise cryptographic algorithms get the
    fixed default (shareable across processes) and non-cryptographic ones get
    fresh random bytes, keeping the seed unpredictable where collision
    resistance depends on it.
    """
    hash_seed = os.getenv("PYTHONHASHSEED")
    if hash_seed is not None:
        return hash_seed
    if hash_fn in _NON_CRYPTO_HASH_FUNCTIONS:
        return os.urandom(32).hex()
    return DEFAULT_NONE_HASH_SEED


def get_none_hash_seed() -> str:
    """Return the seed NONE_HASH was derived from.

    Components that must agree on NONE_HASH across processes (the P2P tier
    advertises this during its connect handshake) read the resolved seed here
    instead of re-deriving it, so they observe the random seed too. Falls back
    to the deterministic seed before ``init_none_hash`` has run.
    """
    if _NONE_HASH_SEED is None:
        return DEFAULT_NONE_HASH_SEED
    return _NONE_HASH_SEED


def init_none_hash(hash_fn: Callable[[Any], bytes]):
    global NONE_HASH, _NONE_HASH_SEED

    _NONE_HASH_SEED = resolve_none_hash_seed(hash_fn)
    if hash_fn in _NON_CRYPTO_HASH_FUNCTIONS and os.getenv("PYTHONHASHSEED") is None:
        logger.warning(
            "Using a random per-process NONE_HASH seed because %s is not "
            "collision resistant. Block hashes are therefore not reproducible "
            "across processes; set PYTHONHASHSEED to a shared value to reuse "
            "the prefix cache across instances, or use sha256.",
            hash_fn.__name__,
        )
    NONE_HASH = BlockHash(hash_fn(_NONE_HASH_SEED))


def hash_block_tokens(
    hash_function: Callable[[Any], bytes],
    parent_block_hash: BlockHash | None,
    curr_block_token_ids: Sequence[int],
    extra_keys: tuple[Any, ...] | None = None,
) -> BlockHash:
    """Computes a hash value corresponding to the contents of a block and
    the contents of the preceding block(s). The hash value is used for
    prefix caching. We use LRU cache for this function to avoid recomputing
    hash values for the same block contents.

    Args:
        hash_function: The hash function used to compute block hash.
        parent_block_hash: The hash of the parent block. None
            if this is the first block.
        curr_block_token_ids: A list of token ids in the current
            block. The current block is assumed to be full.
        extra_keys: Extra keys for the block.

    Returns:
        The hash value of the block and the token ids in the block.
        The entire tuple is used as the hash key of the block.

    """
    if not parent_block_hash:
        parent_block_hash = NONE_HASH

    curr_block_token_ids_tuple = tuple(curr_block_token_ids)
    return BlockHash(
        hash_function((parent_block_hash, curr_block_token_ids_tuple, extra_keys))
    )
