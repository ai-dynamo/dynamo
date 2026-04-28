// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Canonical salt payload used to derive a [`SaltHash`] from a [`crate::Request`].
//!
//! The salt is the *prefix-cache isolation key*. Two requests that should not share
//! cache prefixes must produce different salt hashes; two requests that should share
//! prefixes must produce identical salt hashes.
//!
//! Multimodal data is **not** part of the salt — it is folded into per-block hashing
//! so that requests with the same image at the same global token position can still
//! share the prefix blocks before the image.

use dynamo_tokens::{SaltHash, compute_salt_hash_from_bytes};
use serde::Serialize;

use crate::error::KvHashingError;

/// Production block-size range mixed into the salt. Power-of-two only, in `16..=1024`.
pub const MIN_BLOCK_SIZE: u32 = 16;
/// Maximum production block size mixed into the salt.
pub const MAX_BLOCK_SIZE: u32 = 1024;

/// Stable JSON shape of the salt payload.
///
/// The wire layout is intentionally minimal so future additions (e.g., model_arch tag,
/// quantization scheme) can be appended as new optional fields without invalidating
/// existing salt hashes when those fields are absent.
///
/// `block_size` is mixed into the salt so two requests with different block_sizes
/// silently cannot collide, even if their tokens are identical. The accompanying
/// [`PositionalLineageHash`](dynamo_tokens::PositionalLineageHash) also carries the
/// `block_size` as plaintext metadata, but the salt-mix is the load-bearing safety
/// mechanism — runtime equality checks would not catch dropped/forged metadata.
#[derive(Debug, Serialize)]
pub(crate) struct SaltPayload<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) salt: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) lora_name: Option<&'a str>,
    pub(crate) block_size: u32,
}

/// Computes the canonical [`SaltHash`] from `(salt, lora_name, block_size)`.
///
/// `compute_hash_v2(json_bytes, 0)` over the canonical [`SaltPayload`] JSON.
///
/// Empty strings (`Some("")`) are normalized to `None` for both `salt` and `lora_name`
/// before hashing. This matches the router's existing behavior at
/// `lib/kv-router/src/protocols.rs:84` (`options.lora_name.filter(|n| !n.is_empty())`)
/// so a client that sends `lora_name = ""` shares the cache with a client that sends
/// `lora_name = None`.
///
/// `block_size` must be a power of two in [`MIN_BLOCK_SIZE`]..=[`MAX_BLOCK_SIZE`]; this
/// is the cache-safety check ensuring different block sizes never collide.
pub(crate) fn compute_salt_hash(
    salt: Option<&str>,
    lora_name: Option<&str>,
    block_size: u32,
) -> Result<SaltHash, KvHashingError> {
    if !(block_size.is_power_of_two() && (MIN_BLOCK_SIZE..=MAX_BLOCK_SIZE).contains(&block_size)) {
        return Err(KvHashingError::InvalidBlockSize(block_size));
    }
    let salt = salt.filter(|s| !s.is_empty());
    let lora_name = lora_name.filter(|s| !s.is_empty());
    let payload = SaltPayload {
        salt,
        lora_name,
        block_size,
    };
    let bytes = serde_json::to_vec(&payload)?;
    Ok(compute_salt_hash_from_bytes(&bytes))
}
