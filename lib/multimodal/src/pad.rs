// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The pad and identifier rules: how a multimodal object's identifier becomes Dynamo's routing
//! hash, and how that hash becomes the placeholder token value the frontend writes and the
//! engine derives, so request-side and event-side block hashes agree.

use xxhash_rust::xxh3;

/// sglang's `MultimodalItem._compute_pad_value` constants — must track upstream;
/// if they drift, MM routing silently degrades to text-prefix. Pinned by
/// `pad_value_matches_sglang_protocol`.
/// Offset added to a multimodal hash before masking, matching the engine-side padding rule.
pub const MM_PAD_SHIFT_VALUE: u64 = 1_000_000;
/// Mask applied to a shifted multimodal hash to produce a padding token value.
pub const MM_PAD_HASH_MASK: u64 = (1 << 30) - 1;

/// Canonical per-image pad_value from a routing-side `mm_hash`, called by both
/// the frontend and the kv-router so request- and event-side hashes agree.
/// Keeps the low 30 bits only (sglang's limit).
pub fn pad_value_for_mm_hash(mm_hash: u64) -> u32 {
    (MM_PAD_SHIFT_VALUE + (mm_hash & MM_PAD_HASH_MASK)) as u32
}

/// Map a non-empty multimodal identifier to Dynamo's routing hash.
///
/// Preserve vLLM's canonical 64-character hex-digest mapping for compatibility,
/// and hash shorter opaque identifiers emitted by external renderers with XXH3.
pub fn hash_mm_identifier(identifier: &str) -> Option<u64> {
    if identifier.is_empty() {
        return None;
    }
    if identifier.len() == 64 && identifier.chars().all(|c| c.is_ascii_hexdigit()) {
        return u64::from_str_radix(&identifier[..16], 16).ok();
    }
    Some(xxh3::xxh3_64(identifier.as_bytes()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pad_value_matches_sglang_protocol() {
        assert_eq!(MM_PAD_SHIFT_VALUE, 1_000_000);
        assert_eq!(MM_PAD_HASH_MASK, (1u64 << 30) - 1);
        assert_eq!(pad_value_for_mm_hash(0), MM_PAD_SHIFT_VALUE as u32);
        let fits = (1u64 << 30) - 1;
        assert_eq!(
            pad_value_for_mm_hash(fits),
            (MM_PAD_SHIFT_VALUE + fits) as u32
        );
        let overflow = (1u64 << 30) | 0xCAFE;
        assert_eq!(
            pad_value_for_mm_hash(overflow),
            (MM_PAD_SHIFT_VALUE + 0xCAFE) as u32,
            "high bits above the 30-bit mask must be discarded"
        );
    }
    #[test]
    fn mm_identifier_hash_preserves_canonical_vllm_digest_mapping() {
        let identifier = "0123456789abcdef".repeat(4);
        assert_eq!(hash_mm_identifier(&identifier), Some(0x0123_4567_89ab_cdef));
    }
    #[test]
    fn mm_identifier_hash_supports_opaque_identifiers() {
        let identifier = "opaque-renderer-image-0";
        assert_eq!(
            hash_mm_identifier(identifier),
            Some(xxh3::xxh3_64(identifier.as_bytes()))
        );
        assert_eq!(hash_mm_identifier(""), None);
    }
}
