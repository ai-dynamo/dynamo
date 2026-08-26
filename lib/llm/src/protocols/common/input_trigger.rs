// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Ingress request classification for agentic turn reconstruction.
//!
//! Each classifier inspects the *incoming* request before it is converted into the
//! internal chat-completions representation. This preserves provider-specific shape
//! information (e.g. Anthropic `tool_result` blocks nested inside a user message,
//! Responses API `function_call_output` items) that is lost or split during protocol
//! normalization.

use super::extensions::InputTrigger;
use crate::protocols::{
    anthropic::{
        AnthropicContentBlock, AnthropicCreateMessageRequest, AnthropicMessageContent,
        AnthropicRole,
    },
    openai::{
        chat_completions::NvCreateChatCompletionRequest, completions::NvCreateCompletionRequest,
        responses::NvCreateResponse,
    },
};
use dynamo_protocols::types::{
    ChatCompletionRequestMessage, CreateChatCompletionRequest,
    responses::{InputItem, InputParam, InputRole, Item, MessageItem, Role as ResponseRole},
};

/// Classify an OpenAI Chat Completions request by its last causal message.
pub fn classify_chat_request(request: &NvCreateChatCompletionRequest) -> InputTrigger {
    classify_create_chat_completion_request(&request.inner)
}

fn classify_create_chat_completion_request(request: &CreateChatCompletionRequest) -> InputTrigger {
    let Some(last) = request.messages.last() else {
        return InputTrigger::Other;
    };

    match last {
        ChatCompletionRequestMessage::User(_) => InputTrigger::UserMessage,
        ChatCompletionRequestMessage::Tool(_) | ChatCompletionRequestMessage::Function(_) => {
            InputTrigger::ToolResult
        }
        ChatCompletionRequestMessage::Assistant(_) => InputTrigger::Other,
        ChatCompletionRequestMessage::System(_) | ChatCompletionRequestMessage::Developer(_) => {
            InputTrigger::Other
        }
    }
}

/// Classify an OpenAI Responses request by its `input` parameter.
pub fn classify_response_request(request: &NvCreateResponse) -> InputTrigger {
    match &request.inner.input {
        InputParam::Text(_) => InputTrigger::UserMessage,
        InputParam::Items(items) => {
            let Some(last) = items.last() else {
                return InputTrigger::Other;
            };
            match last {
                InputItem::Item(Item::FunctionCallOutput(_)) => InputTrigger::ToolResult,
                InputItem::Item(Item::Message(MessageItem::Input(msg))) => {
                    if msg.role == InputRole::User {
                        InputTrigger::UserMessage
                    } else {
                        // System/Developer input messages have no clear trigger.
                        InputTrigger::Other
                    }
                }
                InputItem::Item(Item::Message(MessageItem::Output(_)))
                | InputItem::Item(Item::FunctionCall(_))
                | InputItem::Item(Item::Reasoning(_)) => InputTrigger::Other,
                InputItem::EasyMessage(easy) => match easy.role {
                    ResponseRole::User => InputTrigger::UserMessage,
                    _ => InputTrigger::Other,
                },
                // TODO: Classify non-function tool output variants as ToolResult.
                _ => InputTrigger::Other,
            }
        }
    }
}

/// Classify an Anthropic Messages request by its last causal message.
///
/// Anthropic tool results arrive inside an outer `role: user` message as
/// `tool_result` content blocks, so we inspect content blocks and not just role.
pub fn classify_anthropic_request(request: &AnthropicCreateMessageRequest) -> InputTrigger {
    let Some(last) = request.messages.last() else {
        return InputTrigger::Other;
    };

    match last.role {
        AnthropicRole::User => match &last.content {
            AnthropicMessageContent::Text { .. } => InputTrigger::UserMessage,
            AnthropicMessageContent::Blocks { content: blocks } => {
                if blocks
                    .iter()
                    .any(|b| matches!(b, AnthropicContentBlock::ToolResult { .. }))
                {
                    InputTrigger::ToolResult
                } else {
                    InputTrigger::UserMessage
                }
            }
        },
        AnthropicRole::Assistant | AnthropicRole::System => InputTrigger::Other,
    }
}

/// Classify an OpenAI Completions request.
///
/// Completions prompts have no user/tool role information.
pub fn classify_completion_request(_request: &NvCreateCompletionRequest) -> InputTrigger {
    InputTrigger::Other
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::openai::{
        chat_completions::NvCreateChatCompletionRequest, common_ext::CommonExt,
    };
    use dynamo_protocols::types::responses::{CreateResponse, EasyInputContent, EasyInputMessage};
    use dynamo_protocols::types::{
        ChatCompletionRequestMessage, ChatCompletionRequestToolMessage,
        ChatCompletionRequestToolMessageContent, ChatCompletionRequestUserMessage,
        ChatCompletionRequestUserMessageContent,
    };

    fn chat_request_with_messages(
        messages: Vec<ChatCompletionRequestMessage>,
    ) -> NvCreateChatCompletionRequest {
        NvCreateChatCompletionRequest {
            inner: CreateChatCompletionRequest {
                messages,
                model: "test".to_string(),
                ..Default::default()
            },
            common: CommonExt::default(),
            nvext: None,
            chat_template_args: None,
            thinking: None,
            media_io_kwargs: None,
            return_tokens_as_token_ids: None,
            return_token_ids: None,
            unsupported_fields: Default::default(),
        }
    }

    #[test]
    fn chat_user_message() {
        let req = chat_request_with_messages(vec![ChatCompletionRequestMessage::User(
            ChatCompletionRequestUserMessage {
                content: ChatCompletionRequestUserMessageContent::Text("hello".into()),
                name: None,
            },
        )]);
        assert_eq!(classify_chat_request(&req), InputTrigger::UserMessage);
    }

    #[test]
    fn chat_tool_result() {
        let req = chat_request_with_messages(vec![ChatCompletionRequestMessage::Tool(
            ChatCompletionRequestToolMessage {
                content: ChatCompletionRequestToolMessageContent::Text("42".into()),
                tool_call_id: "call-1".into(),
            },
        )]);
        assert_eq!(classify_chat_request(&req), InputTrigger::ToolResult);
    }

    #[test]
    fn chat_other_empty_messages() {
        let req = chat_request_with_messages(vec![]);
        assert_eq!(classify_chat_request(&req), InputTrigger::Other);
    }

    fn response_request_with_easy_messages(role: ResponseRole) -> NvCreateResponse {
        NvCreateResponse {
            inner: CreateResponse {
                input: InputParam::Items(vec![InputItem::EasyMessage(EasyInputMessage {
                    role,
                    content: EasyInputContent::Text("hi".into()),
                    ..Default::default()
                })]),
                model: Some("test".into()),
                ..Default::default()
            },
            nvext: None,
            chat_template_args: None,
        }
    }

    #[test]
    fn responses_easy_message_roles() {
        for (role, expected) in [
            (ResponseRole::User, InputTrigger::UserMessage),
            (ResponseRole::Assistant, InputTrigger::Other),
            (ResponseRole::System, InputTrigger::Other),
        ] {
            assert_eq!(
                classify_response_request(&response_request_with_easy_messages(role)),
                expected
            );
        }
    }
}
