// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Source-derived native-server vocabulary, distinct from support declarations.
//!
//! Protect the union of known endpoint fields: a chat-only directive sent to the
//! completions endpoint is not a harmless unknown. The parser's typed fields and
//! explicit passthrough policy determine handling; this inventory grants no
//! forwarding permission and does not claim parity with a native server.

pub(crate) mod admission;
pub(crate) mod profile;
pub(crate) mod rejection;
pub(crate) mod telemetry;
mod vllm_fields;

pub(super) const VLLM_VERSIONS: &str = vllm_fields::VLLM_VERSIONS;

/// Return the bounded inventory identity, never the caller's string storage.
pub(super) fn known_native_field(field: &str) -> Option<&'static str> {
    vllm_fields::VLLM_REQUEST_FIELDS
        .binary_search(&field)
        .ok()
        .map(|index| vllm_fields::VLLM_REQUEST_FIELDS[index])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_vocabulary_is_sorted_unique_and_excludes_python_metadata() {
        assert!(
            vllm_fields::VLLM_REQUEST_FIELDS
                .windows(2)
                .all(|p| p[0] < p[1])
        );
        assert_eq!(known_native_field("watermarking"), Some("watermarking"));
        assert_eq!(
            known_native_field("continue_final_message"),
            Some("continue_final_message")
        );
        assert_eq!(known_native_field("field_names"), None);
        assert_eq!(known_native_field("_DEFAULT_SAMPLING_PARAMS"), None);
        assert_eq!(known_native_field("experimental_unknown_field"), None);
    }
}
