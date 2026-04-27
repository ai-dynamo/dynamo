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

/// Stable JSON shape of the salt payload.
///
/// The wire layout is intentionally minimal so future additions (e.g., model_arch tag,
/// quantization scheme) can be appended as new optional fields without invalidating
/// existing salt hashes when those fields are absent.
#[derive(Debug, Serialize)]
pub(crate) struct SaltPayload<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) salt: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) lora_name: Option<&'a str>,
}

/// Computes the canonical [`SaltHash`] from `(salt, lora_name)`.
///
/// `compute_hash_v2(json_bytes, 0)` over the canonical [`SaltPayload`] JSON.
///
/// Empty strings (`Some("")`) are normalized to `None` for both `salt` and `lora_name`
/// before hashing. This matches the router's existing behavior at
/// `lib/kv-router/src/protocols.rs:84` (`options.lora_name.filter(|n| !n.is_empty())`)
/// so a client that sends `lora_name = ""` shares the cache with a client that sends
/// `lora_name = None`.
pub(crate) fn compute_salt_hash(
    salt: Option<&str>,
    lora_name: Option<&str>,
) -> Result<SaltHash, KvHashingError> {
    let salt = salt.filter(|s| !s.is_empty());
    let lora_name = lora_name.filter(|s| !s.is_empty());
    let payload = SaltPayload { salt, lora_name };
    let bytes = serde_json::to_vec(&payload)?;
    Ok(compute_salt_hash_from_bytes(&bytes))
}
