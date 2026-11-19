// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The vLLM extra-key marker: the frontend writes a routing hash into `extra_keys` as a
//! 64-character key with a recognizable zero padding, and the router parses only keys in that
//! form back, so native vLLM hashes never enable exact multimodal routing by accident.

/// Encode a frontend-approved routing hash in the marker form carried through
/// vLLM `extra_keys`.
pub fn mark_mm_hash_for_extra_key(mm_hash: u64) -> String {
    format!("{mm_hash:016x}{}", "0".repeat(48))
}

/// Parse a frontend-issued Dynamo MM hash from a vLLM extra key.
///
/// The worker canonicalizes a frontend-approved routing hash to its first 16
/// hex characters plus 48 trailing zeroes before submitting it to vLLM. Treat
/// that padding as an enablement marker: native 64-character vLLM hashes must
/// not enable worker-side key rewriting when the frontend disabled exact MM
/// routing for this request.
pub fn parse_mm_hash_from_extra_key(s: &str) -> Option<u64> {
    // extra_keys mixes MM identifiers with LoRA/cache_salt/prompt-embed metadata.
    // Only MM identifiers should be mapped into BlockExtraInfo.
    let bytes = s.as_bytes();
    if bytes.len() == 64
        && bytes[..16].iter().all(u8::is_ascii_hexdigit)
        && bytes[16..].iter().all(|&c| c == b'0')
    {
        return u64::from_str_radix(&s[..16], 16).ok();
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_frontend_padded_mm_hashes_are_parsed() {
        assert_eq!(
            parse_mm_hash_from_extra_key(&mark_mm_hash_for_extra_key(0x0123_4567_89ab_cdef)),
            Some(0x0123_4567_89ab_cdef)
        );
        assert_eq!(
            parse_mm_hash_from_extra_key(
                "0123456789abcdef000000000000000000000000000000000000000000000000"
            ),
            Some(0x0123_4567_89ab_cdef)
        );
        assert_eq!(
            parse_mm_hash_from_extra_key(
                "0123456789abcdef00112233445566778899aabbccddeefffedcba9876543210"
            ),
            None
        );
        assert_eq!(parse_mm_hash_from_extra_key("123"), None);
        assert_eq!(parse_mm_hash_from_extra_key("not_a_hash"), None);
    }
}
