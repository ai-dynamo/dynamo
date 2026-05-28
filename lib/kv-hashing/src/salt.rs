// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Canonical [`SaltHash`] derivation from a [`crate::Request`].
//!
//! The salt is the *prefix-cache isolation key*. Two requests that should not share
//! cache prefixes must produce different salt hashes; two requests that should share
//! prefixes must produce identical salt hashes.
//!
//! # Router parity
//!
//! For requests with no extra `salt`, this function reproduces the seed used by
//! `dynamo_kv_router::protocols::compute_block_hash_for_seq`
//! (`lib/kv-router/src/protocols.rs:79-82`):
//!
//! ```text
//!   (salt=None, lora=None)         → CHAIN_XXH3_SEED
//!   (salt=None, lora=Some(name))   → CHAIN_XXH3_SEED.wrapping_add(xxh3_64(name))
//! ```
//!
//! Producer events whose `block_hash` is `compute_block_hash(tokens, salt_hash)` therefore
//! match the router's `compute_block_hash_for_seq(tokens, _, BlockHashOptions { lora_name })`
//! byte-for-byte on the no-salt path — required for kv-router's indexers (which key on
//! both `tokens_hash` and the `seq_hash` chain) to find matches against consolidator-emitted
//! events.
//!
//! Multimodal data is **not** part of the salt — it is folded into per-block hashing
//! so that requests with the same image at the same global token position can still
//! share the prefix blocks before the image.

use dynamo_tokens::{CHAIN_XXH3_SEED, SaltHash, compute_hash_v2};

use crate::error::KvHashingError;

/// Computes the canonical [`SaltHash`] from `(salt, lora_name)`.
///
/// Empty strings (`Some("")`) are normalized to `None` for both `salt` and `lora_name`
/// before hashing. This matches the router's existing behavior at
/// `lib/kv-router/src/protocols.rs:79` (`options.lora_name.filter(|n| !n.is_empty())`)
/// so a client that sends `lora_name = ""` shares the cache with a client that sends
/// `lora_name = None`.
pub(crate) fn compute_salt_hash(
    salt: Option<&str>,
    lora_name: Option<&str>,
) -> Result<SaltHash, KvHashingError> {
    let salt = salt.filter(|s| !s.is_empty());
    let lora_name = lora_name.filter(|s| !s.is_empty());

    let mut seed = CHAIN_XXH3_SEED;
    if let Some(name) = lora_name {
        seed = seed.wrapping_add(compute_hash_v2(name.as_bytes(), 0));
    }
    if let Some(s) = salt {
        // Router has no concept of caller-supplied salt; mix orthogonally to lora so
        // salt-isolated requests stay distinct from both no-salt and lora-only requests.
        // The 1 vs 0 inner seed (and outer wrapping_add) keeps every (salt, lora) pair
        // separable: same lora + different salts diverge, and a future `(salt=lora_bytes,
        // lora=None)` request will not collide with `(salt=None, lora=lora_bytes)`.
        seed = seed.wrapping_add(compute_hash_v2(s.as_bytes(), 1));
    }
    Ok(seed)
}
