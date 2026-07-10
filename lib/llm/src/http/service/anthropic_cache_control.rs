// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::protocols::anthropic::types::{
    AnthropicContentBlock, AnthropicCreateMessageRequest, AnthropicMessageContent,
};

fn has_explicit_cache_control(request: &AnthropicCreateMessageRequest) -> bool {
    request
        .system
        .as_ref()
        .is_some_and(|system| system.cache_control.is_some())
        || request
            .tools
            .as_ref()
            .is_some_and(|tools| tools.iter().any(|tool| tool.cache_control.is_some()))
        || request.messages.iter().any(|message| {
            let AnthropicMessageContent::Blocks { content } = &message.content else {
                return false;
            };
            content.iter().any(|block| {
                matches!(
                    block,
                    AnthropicContentBlock::Text {
                        cache_control: Some(_),
                        ..
                    } | AnthropicContentBlock::Image {
                        cache_control: Some(_),
                        ..
                    } | AnthropicContentBlock::ToolUse {
                        cache_control: Some(_),
                        ..
                    } | AnthropicContentBlock::ToolResult {
                        cache_control: Some(_),
                        ..
                    } | AnthropicContentBlock::Thinking {
                        cache_control: Some(_),
                        ..
                    }
                )
            })
        })
}

pub(super) fn cache_retention_ttl_seconds(
    request: &AnthropicCreateMessageRequest,
) -> Result<Option<u32>, &'static str> {
    if has_explicit_cache_control(request) {
        return Err(
            "Block-level `cache_control` is not supported; use the top-level automatic `cache_control` field.",
        );
    }
    Ok(request
        .cache_control
        .as_ref()
        .map(|cache_control| cache_control.ttl_seconds() as u32))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(value: serde_json::Value) -> AnthropicCreateMessageRequest {
        serde_json::from_value(value).unwrap()
    }

    #[test]
    fn top_level_cache_control_maps_to_full_prompt_ttl() {
        let one_hour = request(serde_json::json!({
            "model": "test-model",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
            "cache_control": {"type": "ephemeral", "ttl": "1h"}
        }));
        let default = request(serde_json::json!({
            "model": "test-model",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
            "cache_control": {"type": "ephemeral"}
        }));

        assert_eq!(cache_retention_ttl_seconds(&one_hour), Ok(Some(3600)));
        assert_eq!(cache_retention_ttl_seconds(&default), Ok(Some(300)));
    }

    #[test]
    fn block_cache_controls_are_rejected() {
        for content in [
            serde_json::json!([
                {"type": "text", "text": "first", "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "last"}
            ]),
            serde_json::json!([
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": "AA=="},
                    "cache_control": {"type": "ephemeral"}
                }
            ]),
        ] {
            let request = request(serde_json::json!({
                "model": "test-model",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": content}]
            }));
            assert!(cache_retention_ttl_seconds(&request).is_err());
        }
    }
}
