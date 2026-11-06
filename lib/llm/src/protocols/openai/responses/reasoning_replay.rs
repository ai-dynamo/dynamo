// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Portable reasoning replay compatible with the TRT-LLM serving profile.
//! The payload is encoded UTF-8, not encrypted, and carries no authentication.

use base64::{Engine, engine::general_purpose::STANDARD};

const PREFIX: &str = "trtllm-reasoning-v1:";

pub(super) fn encode(text: &str) -> String {
    format!("{PREFIX}{}", STANDARD.encode(text))
}

pub(super) fn decode(value: &str) -> anyhow::Result<Option<String>> {
    let Some(payload) = value.strip_prefix(PREFIX) else {
        return Ok(None);
    };
    let bytes = STANDARD.decode(payload).map_err(|_| {
        super::ResponsesConversionError::InvalidArgument(
            "Invalid base64 in trtllm-reasoning-v1 replay payload".into(),
        )
    })?;
    String::from_utf8(bytes).map(Some).map_err(|_| {
        super::ResponsesConversionError::InvalidArgument(
            "Invalid UTF-8 in trtllm-reasoning-v1 replay payload".into(),
        )
        .into()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_unicode_and_whitespace_replay() {
        let text = "  think\n工具\n";
        assert_eq!(decode(&encode(text)).unwrap().as_deref(), Some(text));
        assert_eq!(encode("hello"), "trtllm-reasoning-v1:aGVsbG8=");
    }

    #[test]
    fn foreign_encodings_are_not_interpreted() {
        assert_eq!(decode("opaque-provider-payload").unwrap(), None);
    }

    #[test]
    fn malformed_owned_payloads_fail_instead_of_losing_history() {
        assert!(decode("trtllm-reasoning-v1:!!!").is_err());
        assert!(decode("trtllm-reasoning-v1:/w==").is_err());
    }
}
