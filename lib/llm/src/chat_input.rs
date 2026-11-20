// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Model input adaptation, separate from output parsing and API conversion.

use crate::model_card::{ModelDeploymentCard, TokenizerKind};
use crate::protocols::anthropic::types::{
    AnthropicCreateMessageRequest, AnthropicRole, SystemContent, system_message_content,
};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) enum ChatInputPolicy {
    #[default]
    Standard,
    Mistral,
}

impl ChatInputPolicy {
    /// Resolve once from downloaded artifacts, including factory-owned renderers.
    pub(crate) fn from_card(card: &ModelDeploymentCard) -> Self {
        if card
            .chat_template_file
            .as_ref()
            .is_some_and(|template| template.is_custom())
        {
            return Self::Standard;
        }
        if card
            .tokenizer
            .as_ref()
            .and_then(TokenizerKind::tekken_file)
            .is_some()
        {
            Self::Mistral
        } else {
            Self::Standard
        }
    }

    pub(crate) fn prepare_anthropic(&self, request: &mut AnthropicCreateMessageRequest) {
        if matches!(self, Self::Mistral) && merge_inline_system_messages(request) {
            static WARN_ONCE: std::sync::Once = std::sync::Once::new();
            WARN_ONCE.call_once(|| {
                tracing::warn!(
                    model = request.model,
                    "Inline system messages in /v1/messages are merged into the leading \
                     system prompt for the native Mistral renderer; each new one changes the \
                     prompt prefix, so the conversation misses the prefix cache"
                );
            });
        }
    }
}

/// Fold `role: "system"` messages from `messages[]` into the leading system prompt,
/// and report whether any were folded.
///
/// mistral-common's message order rejects a system message after an assistant or tool
/// turn, and Claude Code sends one after each tool result while its `<total_tokens>`
/// reminder is on. Upstream vLLM merges them similarly for templates that reject them.
/// The leading system prompt precedes the tools and the history, so each new inline
/// message changes the whole prompt prefix and the next turn misses the prefix cache.
fn merge_inline_system_messages(req: &mut AnthropicCreateMessageRequest) -> bool {
    let mut inline = Vec::new();
    req.messages.retain(|msg| {
        if !matches!(msg.role, AnthropicRole::System) {
            return true;
        }
        let text = system_message_content(&msg.content);
        if !text.is_empty() {
            inline.push(text);
        }
        false
    });
    if inline.is_empty() {
        return false;
    }
    let (leading, cache_control) = match req.system.take() {
        Some(SystemContent {
            text,
            cache_control,
        }) => (Some(text), cache_control),
        None => (None, None),
    };
    let text = leading
        .into_iter()
        .chain(inline)
        .collect::<Vec<_>>()
        .join("\n");
    req.system = Some(SystemContent {
        text,
        cache_control,
    });
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::openai::chat_completions::NvCreateChatCompletionRequest;
    use dynamo_protocols::types::ChatCompletionRequestMessage;

    #[test]
    fn native_policy_follows_the_card_for_token_and_text_input() {
        let mut card = ModelDeploymentCard::load_from_disk(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/data/sample-models/mock-mistral-tekken"),
            None,
        )
        .unwrap();
        for input in [
            crate::model_type::ModelInput::Tokens,
            crate::model_type::ModelInput::Text,
        ] {
            card.model_input = input;
            assert_eq!(ChatInputPolicy::from_card(&card), ChatInputPolicy::Mistral);
        }
    }

    #[test]
    fn test_merge_inline_system_messages_after_tool_result() {
        let json = r#"{
            "model": "test-model",
            "max_tokens": 100,
            "system": "You are a coding agent.",
            "messages": [
                {"role": "user", "content": "Run cat notes.txt"},
                {"role": "system", "content": "Environment: macOS"},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "toolu_01", "name": "Bash", "input": {"command": "cat notes.txt"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_01", "content": "hello"}
                ]},
                {"role": "system", "content": [{"type": "text", "text": "1000 tokens left"}]}
            ]
        }"#;
        let mut req: AnthropicCreateMessageRequest = serde_json::from_str(json).unwrap();
        assert!(merge_inline_system_messages(&mut req));

        assert_eq!(
            req.system.as_ref().unwrap().text,
            "You are a coding agent.\nEnvironment: macOS\n1000 tokens left"
        );
        let chat_req: NvCreateChatCompletionRequest = req.try_into().unwrap();
        let roles: Vec<_> = chat_req
            .inner
            .messages
            .iter()
            .map(|message| match message {
                ChatCompletionRequestMessage::System(_) => "system",
                ChatCompletionRequestMessage::User(_) => "user",
                ChatCompletionRequestMessage::Assistant(_) => "assistant",
                ChatCompletionRequestMessage::Tool(_) => "tool",
                _ => "other",
            })
            .collect();
        assert_eq!(roles, ["system", "user", "assistant", "tool"]);
    }

    #[test]
    fn test_merge_inline_system_messages_without_inline_keeps_system() {
        let json = r#"{
            "model": "test-model",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hi"}]
        }"#;
        let mut req: AnthropicCreateMessageRequest = serde_json::from_str(json).unwrap();
        assert!(!merge_inline_system_messages(&mut req));
        assert!(req.system.is_none());
        assert_eq!(req.messages.len(), 1);
    }
}
