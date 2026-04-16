// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Anthropic Messages API conversion logic.
//!
//! Pure protocol types live in `dynamo_protocols::types::anthropic`.
//! This module provides bidirectional conversion to/from the internal
//! chat completions format used by the Dynamo engine.

// Re-export all pure Anthropic protocol types so existing `use crate::protocols::anthropic::*`
// continues to work throughout dynamo-llm.
pub use dynamo_protocols::types::anthropic::*;

use dynamo_protocols::types::{
    ChatCompletionMessageToolCall, ChatCompletionNamedToolChoice,
    ChatCompletionRequestAssistantMessage, ChatCompletionRequestAssistantMessageContent,
    ChatCompletionRequestMessage, ChatCompletionRequestMessageContentPartImage,
    ChatCompletionRequestMessageContentPartText, ChatCompletionRequestSystemMessage,
    ChatCompletionRequestSystemMessageContent, ChatCompletionRequestToolMessage,
    ChatCompletionRequestToolMessageContent, ChatCompletionRequestUserMessage,
    ChatCompletionRequestUserMessageContent, ChatCompletionRequestUserMessageContentPart,
    ChatCompletionTool, ChatCompletionToolChoiceOption, ChatCompletionToolType, FunctionName,
    FunctionObject, FunctionType, ImageUrl, ReasoningContent,
};
use uuid::Uuid;

use crate::protocols::openai::chat_completions::{
    NvCreateChatCompletionRequest, NvCreateChatCompletionResponse,
};
use crate::protocols::openai::common_ext::CommonExt;

// ---------------------------------------------------------------------------
// Conversion: AnthropicCreateMessageRequest -> NvCreateChatCompletionRequest
// ---------------------------------------------------------------------------
impl TryFrom<AnthropicCreateMessageRequest> for NvCreateChatCompletionRequest {
    type Error = anyhow::Error;

    fn try_from(req: AnthropicCreateMessageRequest) -> Result<Self, Self::Error> {
        let mut messages = Vec::new();

        // Prepend system message if present
        if let Some(system_content) = &req.system {
            messages.push(ChatCompletionRequestMessage::System(
                ChatCompletionRequestSystemMessage {
                    content: ChatCompletionRequestSystemMessageContent::Text(
                        system_content.text.clone(),
                    ),
                    name: None,
                },
            ));
        }

        // Convert each Anthropic message
        for msg in &req.messages {
            match (&msg.role, &msg.content) {
                // User with plain text
                (AnthropicRole::User, AnthropicMessageContent::Text { content }) => {
                    messages.push(ChatCompletionRequestMessage::User(
                        ChatCompletionRequestUserMessage {
                            content: ChatCompletionRequestUserMessageContent::Text(content.clone()),
                            name: None,
                        },
                    ));
                }
                // User with content blocks
                (AnthropicRole::User, AnthropicMessageContent::Blocks { content: blocks }) => {
                    convert_user_blocks(blocks, &mut messages)?;
                }
                // Assistant with plain text
                (AnthropicRole::Assistant, AnthropicMessageContent::Text { content }) => {
                    messages.push(ChatCompletionRequestMessage::Assistant(
                        #[allow(deprecated)]
                        ChatCompletionRequestAssistantMessage {
                            content: Some(ChatCompletionRequestAssistantMessageContent::Text(
                                content.clone(),
                            )),
                            reasoning_content: None,
                            refusal: None,
                            name: None,
                            audio: None,
                            tool_calls: None,
                            function_call: None,
                        },
                    ));
                }
                // Assistant with content blocks (may contain tool_use)
                (AnthropicRole::Assistant, AnthropicMessageContent::Blocks { content: blocks }) => {
                    convert_assistant_blocks(blocks, &mut messages);
                }
            }
        }

        // Convert tools
        let tools = req.tools.as_ref().map(|t| convert_anthropic_tools(t));

        // Convert tool_choice
        let tool_choice = req.tool_choice.as_ref().map(convert_anthropic_tool_choice);

        // Convert stop_sequences -> stop
        let stop = req
            .stop_sequences
            .map(dynamo_protocols::types::Stop::StringArray);

        Ok(NvCreateChatCompletionRequest {
            inner: dynamo_protocols::types::CreateChatCompletionRequest {
                messages,
                model: req.model,
                temperature: req.temperature,
                top_p: req.top_p,
                max_completion_tokens: Some(req.max_tokens),
                stop,
                tools,
                tool_choice,
                stream: Some(true), // Always stream internally
                stream_options: Some(dynamo_protocols::types::ChatCompletionStreamOptions {
                    include_usage: true,
                    continuous_usage_stats: false,
                }),
                ..Default::default()
            },
            common: CommonExt {
                top_k: req.top_k.map(|k| k as i32),
                ..Default::default()
            },
            nvext: None,
            // chat_template_args may be augmented by the Anthropic handler
            // (anthropic.rs) after conversion — e.g., setting enable_thinking=true
            // when a reasoning parser is configured. The conversion layer only
            // forwards the client's explicit thinking preference here; the handler
            // has access to parsing_options and makes the final decision.
            chat_template_args: if req
                .thinking
                .as_ref()
                .is_some_and(|t| t.thinking_type == "enabled")
            {
                let mut args = std::collections::HashMap::new();
                args.insert("enable_thinking".to_string(), serde_json::Value::Bool(true));
                Some(args)
            } else {
                None
            },
            media_io_kwargs: None,
            return_tokens_as_token_ids: None,
            unsupported_fields: Default::default(),
        })
    }
}

/// Convert user-role content blocks into chat completion messages.
/// Tool results become separate Tool messages; text/image blocks become user messages.
fn convert_user_blocks(
    blocks: &[AnthropicContentBlock],
    messages: &mut Vec<ChatCompletionRequestMessage>,
) -> Result<(), anyhow::Error> {
    // Accumulate content parts (text + image). When the message contains images,
    // we emit `ChatCompletionRequestUserMessageContent::Array` (multimodal format).
    // For pure-text messages we keep `::Text` for backwards compatibility.
    let mut content_parts: Vec<ChatCompletionRequestUserMessageContentPart> = Vec::new();
    let mut has_image = false;

    for block in blocks {
        match block {
            AnthropicContentBlock::Text { text, .. } => {
                content_parts.push(ChatCompletionRequestUserMessageContentPart::Text(
                    ChatCompletionRequestMessageContentPartText { text: text.clone() },
                ));
            }
            AnthropicContentBlock::Image { source } => {
                if source.source_type != "base64" {
                    anyhow::bail!(
                        "unsupported image source type {:?}; only base64 is supported",
                        source.source_type
                    );
                }
                has_image = true;
                let data_uri = format!("data:{};base64,{}", source.media_type, source.data);
                let url = url::Url::parse(&data_uri)
                    .map_err(|e| anyhow::anyhow!("invalid image data URI: {e}"))?;
                content_parts.push(ChatCompletionRequestUserMessageContentPart::ImageUrl(
                    ChatCompletionRequestMessageContentPartImage {
                        image_url: ImageUrl {
                            url,
                            detail: None,
                            uuid: None,
                        },
                    },
                ));
            }
            AnthropicContentBlock::ToolResult {
                tool_use_id,
                content,
                ..
            } => {
                // Flush any accumulated content parts before the tool result message.
                flush_user_content_parts(&mut content_parts, has_image, messages);
                has_image = false;

                let text = content.clone().map(|c| c.into_text()).unwrap_or_default();
                messages.push(ChatCompletionRequestMessage::Tool(
                    ChatCompletionRequestToolMessage {
                        content: ChatCompletionRequestToolMessageContent::Text(text),
                        tool_call_id: tool_use_id.clone(),
                    },
                ));
            }
            AnthropicContentBlock::ToolUse { .. }
            | AnthropicContentBlock::Thinking { .. }
            | AnthropicContentBlock::RedactedThinking { .. }
            | AnthropicContentBlock::ServerToolUse { .. }
            | AnthropicContentBlock::WebSearchToolResult { .. }
            | AnthropicContentBlock::Other(_) => {
                // tool_use/thinking/server-side blocks/unknown in a user message: skip
            }
        }
    }

    // Flush remaining content parts.
    flush_user_content_parts(&mut content_parts, has_image, messages);

    Ok(())
}

/// Flush accumulated user content parts into a user message.
///
/// If the parts are pure text, joins them into a single `Text` message
/// (backwards-compatible with non-multimodal backends). If any images are
/// present, emits an `Array` message (OpenAI multimodal format).
fn flush_user_content_parts(
    parts: &mut Vec<ChatCompletionRequestUserMessageContentPart>,
    has_image: bool,
    messages: &mut Vec<ChatCompletionRequestMessage>,
) {
    if parts.is_empty() {
        return;
    }

    let content = if has_image {
        // Multimodal: emit as Array so images are preserved.
        ChatCompletionRequestUserMessageContent::Array(std::mem::take(parts))
    } else {
        // Pure text: join into a single string for backwards compatibility.
        let combined = parts
            .drain(..)
            .filter_map(|p| match p {
                ChatCompletionRequestUserMessageContentPart::Text(t) => Some(t.text),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("");
        ChatCompletionRequestUserMessageContent::Text(combined)
    };

    messages.push(ChatCompletionRequestMessage::User(
        ChatCompletionRequestUserMessage {
            content,
            name: None,
        },
    ));
}

/// Convert assistant-role content blocks into chat completion messages.
///
/// Text blocks become an assistant message; tool_use blocks become tool_calls on an assistant
/// message. Thinking blocks are preserved via `reasoning_content: Option<ReasoningContent>`:
///
/// - `ReasoningContent::Text(s)`: flat reasoning string (no tool calls present).
/// - `ReasoningContent::Segments(segs)`: one entry **per position** in the interleaved sequence,
///   enabling chat templates to reconstruct the exact token order:
///   `<think>segments[0]</think><call>tc[0]</call><think>segments[1]</think><call>tc[1]</call>…<think>segments[N]</think>`
///   - `segments[i]` is the thinking that immediately preceded `tool_calls[i]`
///   - `segments[tool_calls.len()]` is any trailing thinking after the last tool call
///   - `segments.len() == tool_calls.len() + 1` always
///   - Individual entries may be empty strings (no reasoning at that position)
/// - `None` when there is no reasoning content at all.
///
/// Preserving the original interleaved order is required for KV cache correctness: a prompt
/// reconstructed from a flattened `reasoning_content` will differ token-by-token from the
/// original assistant turn, causing a cache miss on every multi-tool exchange.
fn convert_assistant_blocks(
    blocks: &[AnthropicContentBlock],
    messages: &mut Vec<ChatCompletionRequestMessage>,
) {
    let mut text_content = String::new();
    let mut tool_calls = Vec::new();
    // One reasoning segment per tool call — segments[i] precedes tool_calls[i].
    let mut segments: Vec<String> = Vec::new();
    // Accumulates thinking text until the next tool_use block (or end of blocks).
    let mut pending_reasoning = String::new();

    for block in blocks {
        match block {
            AnthropicContentBlock::Text { text, .. } => {
                text_content.push_str(text);
            }
            AnthropicContentBlock::Thinking { thinking, .. } => {
                if !pending_reasoning.is_empty() {
                    pending_reasoning.push('\n');
                }
                pending_reasoning.push_str(thinking);
            }
            AnthropicContentBlock::RedactedThinking { .. } => {
                // Redacted thinking is encrypted model reasoning. We can't read
                // it but we preserve its position so it's not silently dropped.
                // The actual encrypted data would need to be passed back to the
                // model in multi-turn conversations for context continuity.
            }
            AnthropicContentBlock::ToolUse {
                id, name, input, ..
            }
            | AnthropicContentBlock::ServerToolUse {
                id, name, input, ..
            } => {
                // Snapshot the reasoning that preceded this tool call.
                // Server-initiated tool use (e.g. web search) is treated the
                // same as client tool use for conversion purposes.
                segments.push(std::mem::take(&mut pending_reasoning));
                tool_calls.push(ChatCompletionMessageToolCall {
                    id: id.clone(),
                    r#type: FunctionType::Function,
                    function: dynamo_protocols::types::FunctionCall {
                        name: name.clone(),
                        arguments: serde_json::to_string(input).unwrap_or_default(),
                    },
                });
            }
            _ => {}
        }
    }

    // Append any trailing reasoning (after the last tool call) as the final segment.
    // This makes segments.len() == tool_calls.len() + 1, preserving the full interleaved
    // order including reasoning that follows the last tool call.
    segments.push(std::mem::take(&mut pending_reasoning));

    let content = if text_content.is_empty() {
        None
    } else {
        Some(ChatCompletionRequestAssistantMessageContent::Text(
            text_content,
        ))
    };

    // Produce a single ReasoningContent value:
    // - Segments variant when there are tool calls and at least one segment is non-empty
    //   (genuine interleaving present).
    // - Text variant when there's reasoning but no tool calls (flat form).
    // - None when there's no reasoning at all.
    let reasoning_content = if !tool_calls.is_empty() && segments.iter().any(|s| !s.is_empty()) {
        Some(ReasoningContent::Segments(segments))
    } else {
        let flat: String = segments
            .iter()
            .filter(|s| !s.is_empty())
            .cloned()
            .collect::<Vec<_>>()
            .join("\n");
        if flat.is_empty() {
            None
        } else {
            Some(ReasoningContent::Text(flat))
        }
    };

    let tc = if tool_calls.is_empty() {
        None
    } else {
        Some(tool_calls)
    };

    messages.push(ChatCompletionRequestMessage::Assistant(
        ChatCompletionRequestAssistantMessage {
            content,
            reasoning_content,
            refusal: None,
            name: None,
            audio: None,
            tool_calls: tc,
            #[allow(deprecated)]
            function_call: None,
        },
    ));
}

/// Convert Anthropic tools to ChatCompletionTools.
fn convert_anthropic_tools(tools: &[AnthropicTool]) -> Vec<ChatCompletionTool> {
    tools
        .iter()
        .filter_map(|tool| {
            // Server tools (web_search, bash, etc.) don't have input_schema
            // and can't be meaningfully converted to OpenAI function tools.
            // They are backend-specific and handled separately.
            let schema = tool.input_schema.clone().or_else(|| {
                tracing::debug!(
                    tool_name = %tool.name,
                    tool_type = ?tool.tool_type,
                    "Skipping server tool in OpenAI conversion (no input_schema)"
                );
                None
            })?;
            Some(ChatCompletionTool {
                r#type: ChatCompletionToolType::Function,
                function: FunctionObject {
                    name: tool.name.clone(),
                    description: tool.description.clone(),
                    parameters: Some(schema),
                    strict: None,
                },
            })
        })
        .collect()
}

/// Convert Anthropic tool_choice to ChatCompletionToolChoiceOption.
fn convert_anthropic_tool_choice(tc: &AnthropicToolChoice) -> ChatCompletionToolChoiceOption {
    match tc {
        AnthropicToolChoice::Simple(simple) => match simple.choice_type {
            AnthropicToolChoiceMode::Auto => ChatCompletionToolChoiceOption::Auto,
            AnthropicToolChoiceMode::Any => ChatCompletionToolChoiceOption::Required,
            AnthropicToolChoiceMode::None => ChatCompletionToolChoiceOption::None,
            AnthropicToolChoiceMode::Tool => {
                // {"type": "tool"} without a "name" field is invalid per the Anthropic spec.
                // It deserialized as Simple because Named requires the name field.
                // Treat as "any" (required) since the caller wants a specific tool but
                // didn't specify which — this is the closest semantic match.
                tracing::warn!(
                    "tool_choice has type 'tool' without a 'name' field; treating as 'any' (required)"
                );
                ChatCompletionToolChoiceOption::Required
            }
        },
        AnthropicToolChoice::Named(named) => {
            ChatCompletionToolChoiceOption::Named(ChatCompletionNamedToolChoice {
                r#type: ChatCompletionToolType::Function,
                function: FunctionName {
                    name: named.name.clone(),
                },
            })
        }
    }
}
/// Convert a completed chat completion response into an Anthropic Messages response.
pub fn chat_completion_to_anthropic_response(
    chat_resp: NvCreateChatCompletionResponse,
    model: &str,
    api_context: Option<&crate::protocols::unified::AnthropicContext>,
) -> AnthropicMessageResponse {
    let _ = api_context; // Available for future enrichment (service_tier, etc.)
    let msg_id = format!("msg_{}", Uuid::new_v4().simple());

    let choice = chat_resp.inner.choices.into_iter().next();
    let mut content = Vec::new();
    let mut stop_reason = None;

    if let Some(choice) = choice {
        // Map finish_reason
        stop_reason = choice.finish_reason.map(|fr| match fr {
            dynamo_protocols::types::FinishReason::Stop => AnthropicStopReason::EndTurn,
            dynamo_protocols::types::FinishReason::Length => AnthropicStopReason::MaxTokens,
            dynamo_protocols::types::FinishReason::ToolCalls => AnthropicStopReason::ToolUse,
            dynamo_protocols::types::FinishReason::ContentFilter => AnthropicStopReason::EndTurn,
            dynamo_protocols::types::FinishReason::FunctionCall => AnthropicStopReason::ToolUse,
        });

        // Extract tool calls
        if let Some(tool_calls) = choice.message.tool_calls {
            for tc in tool_calls {
                let input: serde_json::Value =
                    serde_json::from_str(&tc.function.arguments).unwrap_or(serde_json::json!({}));
                content.push(AnthropicResponseContentBlock::ToolUse {
                    id: tc.id,
                    name: tc.function.name,
                    input,
                });
            }
        }

        // Extract reasoning content (from --dyn-reasoning-parser, e.g. qwen3).
        // The backend strips <think>...</think> from the text and surfaces it
        // as reasoning_content on the message. Map this to a Thinking block
        // so clients see proper extended thinking in the Anthropic response.
        if let Some(thinking) = choice.message.reasoning_content.filter(|t| !t.is_empty()) {
            content.insert(
                0,
                AnthropicResponseContentBlock::Thinking {
                    thinking,
                    signature: String::new(),
                },
            );
        }

        // Extract text content
        let text = match choice.message.content {
            Some(dynamo_protocols::types::ChatCompletionMessageContent::Text(t)) => Some(t),
            Some(dynamo_protocols::types::ChatCompletionMessageContent::Parts(_)) => {
                tracing::warn!(
                    "Multimodal (Parts) content in chat completion response replaced with placeholder text in Anthropic conversion."
                );
                Some("[multimodal content]".to_string())
            }
            None => None,
        };
        if let Some(text) = text {
            // Text goes after thinking block (if any)
            content.push(AnthropicResponseContentBlock::Text {
                text,
                citations: None,
            });
        }
    }

    // Ensure there's at least one content block
    if content.is_empty() {
        content.push(AnthropicResponseContentBlock::Text {
            text: String::new(),
            citations: None,
        });
    }

    // Map usage
    let usage = chat_resp
        .inner
        .usage
        .map(|u| {
            let cache_read_input_tokens = u
                .prompt_tokens_details
                .and_then(|d| d.cached_tokens)
                .filter(|&n| n > 0);
            AnthropicUsage {
                input_tokens: u.prompt_tokens,
                output_tokens: u.completion_tokens,
                cache_creation_input_tokens: None, // Not available from OpenAI format
                cache_read_input_tokens,
            }
        })
        .unwrap_or_default();

    AnthropicMessageResponse {
        id: msg_id,
        object_type: "message".to_string(),
        role: "assistant".to_string(),
        content,
        model: model.to_string(),
        stop_reason,
        stop_sequence: None,
        usage,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_user_message_conversion() {
        let req = AnthropicCreateMessageRequest {
            model: "test-model".into(),
            max_tokens: 100,
            messages: vec![AnthropicMessage {
                role: AnthropicRole::User,
                content: AnthropicMessageContent::Text {
                    content: "Hello!".into(),
                },
            }],
            system: None,
            temperature: Some(0.7),
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: false,
            metadata: None,
            tools: None,
            tool_choice: None,
            cache_control: None,
            thinking: None,
            service_tier: None,
            container: None,
            output_config: None,
        };

        let chat_req: NvCreateChatCompletionRequest = req.try_into().unwrap();
        assert_eq!(chat_req.inner.model, "test-model");
        assert_eq!(chat_req.inner.max_completion_tokens, Some(100));
        assert_eq!(chat_req.inner.temperature, Some(0.7));
        assert_eq!(chat_req.inner.messages.len(), 1);

        match &chat_req.inner.messages[0] {
            ChatCompletionRequestMessage::User(u) => match &u.content {
                ChatCompletionRequestUserMessageContent::Text(t) => {
                    assert_eq!(t, "Hello!");
                }
                _ => panic!("expected text content"),
            },
            _ => panic!("expected user message"),
        }
    }

    #[test]
    fn test_system_message_prepended() {
        let req = AnthropicCreateMessageRequest {
            model: "test-model".into(),
            max_tokens: 100,
            messages: vec![AnthropicMessage {
                role: AnthropicRole::User,
                content: AnthropicMessageContent::Text {
                    content: "Hi".into(),
                },
            }],
            system: Some(SystemContent {
                text: "You are helpful.".into(),
                cache_control: None,
            }),
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: false,
            metadata: None,
            tools: None,
            tool_choice: None,
            cache_control: None,
            thinking: None,
            service_tier: None,
            container: None,
            output_config: None,
        };

        let chat_req: NvCreateChatCompletionRequest = req.try_into().unwrap();
        assert_eq!(chat_req.inner.messages.len(), 2);
        assert!(matches!(
            &chat_req.inner.messages[0],
            ChatCompletionRequestMessage::System(_)
        ));
        assert!(matches!(
            &chat_req.inner.messages[1],
            ChatCompletionRequestMessage::User(_)
        ));
    }

    #[test]
    fn test_tool_use_blocks_conversion() {
        let req = AnthropicCreateMessageRequest {
            model: "test-model".into(),
            max_tokens: 100,
            messages: vec![
                AnthropicMessage {
                    role: AnthropicRole::User,
                    content: AnthropicMessageContent::Text {
                        content: "What's the weather?".into(),
                    },
                },
                AnthropicMessage {
                    role: AnthropicRole::Assistant,
                    content: AnthropicMessageContent::Blocks {
                        content: vec![AnthropicContentBlock::ToolUse {
                            id: "tool_123".into(),
                            name: "get_weather".into(),
                            input: serde_json::json!({"location": "SF"}),
                            cache_control: None,
                        }],
                    },
                },
                AnthropicMessage {
                    role: AnthropicRole::User,
                    content: AnthropicMessageContent::Blocks {
                        content: vec![AnthropicContentBlock::ToolResult {
                            tool_use_id: "tool_123".into(),
                            content: Some(ToolResultContent::Text("72F and sunny".into())),
                            is_error: None,
                            cache_control: None,
                        }],
                    },
                },
            ],
            system: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: false,
            metadata: None,
            tools: None,
            tool_choice: None,
            cache_control: None,
            thinking: None,
            service_tier: None,
            container: None,
            output_config: None,
        };

        let chat_req: NvCreateChatCompletionRequest = req.try_into().unwrap();
        assert_eq!(chat_req.inner.messages.len(), 3);
        assert!(matches!(
            &chat_req.inner.messages[0],
            ChatCompletionRequestMessage::User(_)
        ));
        assert!(matches!(
            &chat_req.inner.messages[1],
            ChatCompletionRequestMessage::Assistant(_)
        ));
        assert!(matches!(
            &chat_req.inner.messages[2],
            ChatCompletionRequestMessage::Tool(_)
        ));
    }

    #[test]
    fn test_stop_sequences_conversion() {
        let req = AnthropicCreateMessageRequest {
            model: "test-model".into(),
            max_tokens: 100,
            messages: vec![AnthropicMessage {
                role: AnthropicRole::User,
                content: AnthropicMessageContent::Text {
                    content: "Hi".into(),
                },
            }],
            system: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: Some(vec!["STOP".into(), "END".into()]),
            stream: false,
            metadata: None,
            tools: None,
            tool_choice: None,
            cache_control: None,
            thinking: None,
            service_tier: None,
            container: None,
            output_config: None,
        };

        let chat_req: NvCreateChatCompletionRequest = req.try_into().unwrap();
        assert!(chat_req.inner.stop.is_some());
    }

    #[test]
    fn test_tools_conversion() {
        let req = AnthropicCreateMessageRequest {
            model: "test-model".into(),
            max_tokens: 100,
            messages: vec![AnthropicMessage {
                role: AnthropicRole::User,
                content: AnthropicMessageContent::Text {
                    content: "Hi".into(),
                },
            }],
            system: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: false,
            metadata: None,
            tools: Some(vec![AnthropicTool {
                name: "get_weather".into(),
                tool_type: None,
                description: Some("Get weather info".into()),
                input_schema: Some(serde_json::json!({
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"]
                })),
                cache_control: None,
            }]),
            tool_choice: Some(AnthropicToolChoice::Simple(AnthropicToolChoiceSimple {
                choice_type: AnthropicToolChoiceMode::Auto,
                disable_parallel_tool_use: None,
            })),
            cache_control: None,
            thinking: None,
            service_tier: None,
            container: None,
            output_config: None,
        };

        let chat_req: NvCreateChatCompletionRequest = req.try_into().unwrap();
        assert!(chat_req.inner.tools.is_some());
        let tools = chat_req.inner.tools.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].function.name, "get_weather");
        assert!(matches!(
            chat_req.inner.tool_choice,
            Some(ChatCompletionToolChoiceOption::Auto)
        ));
    }

    #[allow(deprecated)]
    #[test]
    fn test_chat_completion_to_anthropic_response() {
        let chat_resp = NvCreateChatCompletionResponse {
            inner: dynamo_protocols::types::CreateChatCompletionResponse {
                id: "chatcmpl-xyz".into(),
                choices: vec![dynamo_protocols::types::ChatChoice {
                    index: 0,
                    message: dynamo_protocols::types::ChatCompletionResponseMessage {
                        content: Some(dynamo_protocols::types::ChatCompletionMessageContent::Text(
                            "Hello!".to_string(),
                        )),
                        refusal: None,
                        tool_calls: None,
                        role: dynamo_protocols::types::Role::Assistant,
                        function_call: None,
                        audio: None,
                        reasoning_content: None,
                    },
                    finish_reason: Some(dynamo_protocols::types::FinishReason::Stop),
                    stop_reason: None,
                    logprobs: None,
                }],
                created: 1726000000,
                model: "test-model".into(),
                service_tier: None,
                system_fingerprint: None,
                object: "chat.completion".to_string(),
                usage: Some(dynamo_protocols::types::CompletionUsage {
                    prompt_tokens: 10,
                    completion_tokens: 5,
                    total_tokens: 15,
                    prompt_tokens_details: None,
                    completion_tokens_details: None,
                }),
            },
            nvext: None,
        };

        let response = chat_completion_to_anthropic_response(chat_resp, "test-model", None);
        assert!(response.id.starts_with("msg_"));
        assert_eq!(response.object_type, "message");
        assert_eq!(response.role, "assistant");
        assert_eq!(response.model, "test-model");
        assert_eq!(response.stop_reason, Some(AnthropicStopReason::EndTurn));
        assert_eq!(response.usage.input_tokens, 10);
        assert_eq!(response.usage.output_tokens, 5);
        assert_eq!(response.content.len(), 1);
        match &response.content[0] {
            AnthropicResponseContentBlock::Text { text, .. } => {
                assert_eq!(text, "Hello!");
            }
            _ => panic!("expected text block"),
        }
    }

    #[test]
    fn test_deserialize_simple_message() {
        let json =
            r#"{"model":"test","max_tokens":100,"messages":[{"role":"user","content":"Hello"}]}"#;
        let req: AnthropicCreateMessageRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "test");
        assert_eq!(req.max_tokens, 100);
        assert_eq!(req.messages.len(), 1);
    }

    #[test]
    fn test_deserialize_content_blocks() {
        let json = r#"{
            "model": "test",
            "max_tokens": 100,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "tool_result", "tool_use_id": "tool_1", "content": "result text"}
                ]
            }]
        }"#;
        let req: AnthropicCreateMessageRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.messages.len(), 1);
        match &req.messages[0].content {
            AnthropicMessageContent::Blocks { content } => {
                assert_eq!(content.len(), 2);
            }
            _ => panic!("expected blocks content"),
        }
    }

    #[test]
    fn test_deserialize_thinking_block() {
        let json = r#"{
            "model": "test",
            "max_tokens": 100,
            "messages": [{
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Let me reason about this...", "signature": "sig123"},
                    {"type": "text", "text": "Here is my answer."}
                ]
            }]
        }"#;
        let req: AnthropicCreateMessageRequest = serde_json::from_str(json).unwrap();
        match &req.messages[0].content {
            AnthropicMessageContent::Blocks { content } => {
                assert_eq!(content.len(), 2);
                match &content[0] {
                    AnthropicContentBlock::Thinking {
                        thinking,
                        signature,
                        ..
                    } => {
                        assert_eq!(thinking, "Let me reason about this...");
                        assert_eq!(signature, "sig123");
                    }
                    other => panic!("expected Thinking, got {other:?}"),
                }
            }
            _ => panic!("expected blocks content"),
        }
    }

    #[test]
    fn test_thinking_block_becomes_reasoning_content() {
        let req = AnthropicCreateMessageRequest {
            model: "test-model".into(),
            max_tokens: 100,
            messages: vec![AnthropicMessage {
                role: AnthropicRole::Assistant,
                content: AnthropicMessageContent::Blocks {
                    content: vec![
                        AnthropicContentBlock::Thinking {
                            thinking: "I should think...".into(),
                            signature: "sig".into(),
                            cache_control: None,
                        },
                        AnthropicContentBlock::Text {
                            text: "Answer".into(),
                            citations: None,
                            cache_control: None,
                        },
                    ],
                },
            }],
            system: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: false,
            metadata: None,
            tools: None,
            tool_choice: None,
            cache_control: None,
            thinking: None,
            service_tier: None,
            container: None,
            output_config: None,
        };

        let chat_req: NvCreateChatCompletionRequest = req.try_into().unwrap();
        match &chat_req.inner.messages[0] {
            ChatCompletionRequestMessage::Assistant(a) => {
                assert_eq!(
                    a.reasoning_content,
                    Some(ReasoningContent::Text("I should think...".into()))
                );
                match &a.content {
                    Some(ChatCompletionRequestAssistantMessageContent::Text(t)) => {
                        assert_eq!(t, "Answer");
                    }
                    other => panic!("expected text content, got {other:?}"),
                }
            }
            other => panic!("expected assistant message, got {other:?}"),
        }
    }

    #[test]
    fn test_known_and_unknown_block_types() {
        let json = r#"{
            "model": "test",
            "max_tokens": 100,
            "messages": [{
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "hello"},
                    {"type": "server_tool_use", "id": "stu_1", "name": "web_search", "input": {}},
                    {"type": "redacted_thinking", "data": "encrypted"},
                    {"type": "web_search_tool_result", "tool_use_id": "stu_1", "content": [{"type": "web_search_result", "url": "https://example.com"}]},
                    {"type": "future_block_type", "some_field": 42},
                    {"type": "text", "text": "world"}
                ]
            }]
        }"#;
        let req: AnthropicCreateMessageRequest = serde_json::from_str(json).unwrap();
        match &req.messages[0].content {
            AnthropicMessageContent::Blocks { content } => {
                assert_eq!(content.len(), 6);
                assert!(matches!(&content[0], AnthropicContentBlock::Text { .. }));
                assert!(matches!(
                    &content[1],
                    AnthropicContentBlock::ServerToolUse { name, .. } if name == "web_search"
                ));
                assert!(matches!(
                    &content[2],
                    AnthropicContentBlock::RedactedThinking { data } if data == "encrypted"
                ));
                assert!(matches!(
                    &content[3],
                    AnthropicContentBlock::WebSearchToolResult { tool_use_id, .. } if tool_use_id == "stu_1"
                ));
                // Truly unknown types still fall through to Other with full JSON preserved
                assert!(matches!(
                    &content[4],
                    AnthropicContentBlock::Other(v) if v.get("type").and_then(|t| t.as_str()) == Some("future_block_type")
                ));
                assert!(matches!(&content[5], AnthropicContentBlock::Text { .. }));
            }
            _ => panic!("expected blocks content"),
        }

        // Conversion should succeed — server_tool_use becomes a tool call,
        // redacted_thinking and web_search_tool_result are preserved gracefully
        let chat_req: NvCreateChatCompletionRequest = AnthropicCreateMessageRequest {
            model: "test".into(),
            max_tokens: 100,
            messages: req.messages,
            system: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: false,
            metadata: None,
            tools: None,
            tool_choice: None,
            cache_control: None,
            thinking: None,
            service_tier: None,
            container: None,
            output_config: None,
        }
        .try_into()
        .unwrap();
        // server_tool_use becomes a tool call on the assistant message
        assert_eq!(chat_req.inner.messages.len(), 1);
        match &chat_req.inner.messages[0] {
            ChatCompletionRequestMessage::Assistant(a) => {
                assert!(a.tool_calls.is_some());
                let tc = a.tool_calls.as_ref().unwrap();
                assert_eq!(tc.len(), 1);
                assert_eq!(tc[0].function.name, "web_search");
            }
            other => panic!("expected assistant, got {other:?}"),
        }
    }

    #[test]
    fn test_tool_result_string_content() {
        let json = r#"{
            "model": "test",
            "max_tokens": 100,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "simple text"}
                ]
            }]
        }"#;
        let req: AnthropicCreateMessageRequest = serde_json::from_str(json).unwrap();
        match &req.messages[0].content {
            AnthropicMessageContent::Blocks { content } => match &content[0] {
                AnthropicContentBlock::ToolResult { content, .. } => {
                    let text = content.clone().unwrap().into_text();
                    assert_eq!(text, "simple text");
                }
                other => panic!("expected ToolResult, got {other:?}"),
            },
            _ => panic!("expected blocks"),
        }
    }

    #[test]
    fn test_tool_result_array_content() {
        let json = r#"{
            "model": "test",
            "max_tokens": 100,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": [
                        {"type": "text", "text": "line 1"},
                        {"type": "text", "text": "line 2"}
                    ]}
                ]
            }]
        }"#;
        let req: AnthropicCreateMessageRequest = serde_json::from_str(json).unwrap();
        match &req.messages[0].content {
            AnthropicMessageContent::Blocks { content } => match &content[0] {
                AnthropicContentBlock::ToolResult { content, .. } => {
                    let text = content.clone().unwrap().into_text();
                    assert_eq!(text, "line 1line 2");
                }
                other => panic!("expected ToolResult, got {other:?}"),
            },
            _ => panic!("expected blocks"),
        }
    }

    #[test]
    fn test_count_tokens_estimate() {
        let req = AnthropicCountTokensRequest {
            model: "test".into(),
            messages: vec![AnthropicMessage {
                role: AnthropicRole::User,
                content: AnthropicMessageContent::Text {
                    content: "Hello, world! This is a test message.".into(),
                },
            }],
            system: Some(SystemContent {
                text: "You are helpful.".into(),
                cache_control: None,
            }),
            tools: None,
        };

        let tokens = req.estimate_tokens();
        assert!(tokens > 0, "should estimate non-zero tokens");
        // "Hello, world! This is a test message." (37) + "You are helpful." (16) + role (4) = 57 / 3 = 19
        assert_eq!(tokens, 19);
    }

    // --- ReasoningContent enum tests ---

    fn make_req(blocks: Vec<AnthropicContentBlock>) -> ChatCompletionRequestAssistantMessage {
        let req = AnthropicCreateMessageRequest {
            model: "test-model".into(),
            max_tokens: 100,
            messages: vec![AnthropicMessage {
                role: AnthropicRole::Assistant,
                content: AnthropicMessageContent::Blocks { content: blocks },
            }],
            system: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: false,
            metadata: None,
            tools: None,
            tool_choice: None,
            cache_control: None,
            thinking: None,
            service_tier: None,
            container: None,
            output_config: None,
        };
        let chat_req: NvCreateChatCompletionRequest = req.try_into().unwrap();
        match chat_req.inner.messages.into_iter().next().unwrap() {
            ChatCompletionRequestMessage::Assistant(a) => a,
            other => panic!("expected assistant, got {other:?}"),
        }
    }

    fn tool_use(id: &str) -> AnthropicContentBlock {
        AnthropicContentBlock::ToolUse {
            id: id.into(),
            name: "fn".into(),
            input: serde_json::json!({}),
            cache_control: None,
        }
    }

    fn thinking(text: &str) -> AnthropicContentBlock {
        AnthropicContentBlock::Thinking {
            thinking: text.into(),
            signature: "sig".into(),
            cache_control: None,
        }
    }

    #[test]
    fn test_interleaved_thinking_and_tool_calls() {
        // [Thinking("A"), ToolUse("t1"), Thinking("B"), ToolUse("t2")]
        // segments = ["A", "B", ""] (trailing empty), tool_calls = [t1, t2]
        let msg = make_req(vec![
            thinking("A"),
            tool_use("t1"),
            thinking("B"),
            tool_use("t2"),
        ]);

        let segs = msg
            .reasoning_content
            .as_ref()
            .expect("reasoning_content should be set")
            .segments()
            .expect("should be Segments variant");
        assert_eq!(segs.len(), 3); // tool_calls.len() + 1
        assert_eq!(segs[0], "A");
        assert_eq!(segs[1], "B");
        assert_eq!(segs[2], ""); // no trailing reasoning

        assert_eq!(
            msg.reasoning_content.as_ref().unwrap().to_flat_string(),
            "A\nB"
        );

        let tcs = msg.tool_calls.as_ref().expect("tool_calls should be set");
        assert_eq!(tcs.len(), 2);
        assert_eq!(tcs[0].id, "t1");
        assert_eq!(tcs[1].id, "t2");
    }

    #[test]
    fn test_trailing_reasoning_preserved_in_segments() {
        // [Thinking("A"), ToolUse("t1"), Thinking("B")]
        // segments = ["A", "B"], trailing reasoning "B" must appear in segments[1]
        let msg = make_req(vec![thinking("A"), tool_use("t1"), thinking("B")]);

        let segs = msg
            .reasoning_content
            .as_ref()
            .expect("reasoning_content should be set")
            .segments()
            .expect("should be Segments variant");
        assert_eq!(segs.len(), 2); // 1 tool call + 1 trailing
        assert_eq!(segs[0], "A");
        assert_eq!(segs[1], "B"); // trailing reasoning preserved

        assert_eq!(
            msg.reasoning_content.as_ref().unwrap().to_flat_string(),
            "A\nB"
        );
    }

    #[test]
    fn test_tool_use_before_thinking() {
        // [ToolUse("t1"), Thinking("A"), ToolUse("t2")]
        // segments = ["", "A", ""] — empty first segment, reasoning before t2
        let msg = make_req(vec![tool_use("t1"), thinking("A"), tool_use("t2")]);

        let segs = msg
            .reasoning_content
            .as_ref()
            .expect("reasoning_content should be set")
            .segments()
            .expect("should be Segments variant");
        assert_eq!(segs.len(), 3);
        assert_eq!(segs[0], ""); // no reasoning before t1
        assert_eq!(segs[1], "A");
        assert_eq!(segs[2], ""); // no trailing

        assert_eq!(
            msg.reasoning_content.as_ref().unwrap().to_flat_string(),
            "A"
        );
    }

    #[test]
    fn test_all_thinking_then_all_tools() {
        // [Thinking("A"), Thinking("B"), ToolUse("t1"), ToolUse("t2")]
        // segments = ["A\nB", "", ""] — all reasoning before first tool
        let msg = make_req(vec![
            thinking("A"),
            thinking("B"),
            tool_use("t1"),
            tool_use("t2"),
        ]);

        let segs = msg
            .reasoning_content
            .as_ref()
            .expect("reasoning_content should be set")
            .segments()
            .expect("should be Segments variant");
        assert_eq!(segs.len(), 3);
        assert_eq!(segs[0], "A\nB");
        assert_eq!(segs[1], "");
        assert_eq!(segs[2], "");

        assert_eq!(
            msg.reasoning_content.as_ref().unwrap().to_flat_string(),
            "A\nB"
        );
    }

    #[test]
    fn test_tool_calls_no_thinking_produces_no_segments() {
        // [ToolUse("t1"), ToolUse("t2")] — all empty segments → reasoning_content = None
        let msg = make_req(vec![tool_use("t1"), tool_use("t2")]);

        assert!(
            msg.reasoning_content.is_none(),
            "no reasoning means no reasoning_content"
        );
    }

    #[test]
    fn test_thinking_only_no_tools_produces_text_variant() {
        // [Thinking("A"), Text("answer")] — no tool calls → ReasoningContent::Text
        let msg = make_req(vec![
            thinking("A"),
            AnthropicContentBlock::Text {
                text: "answer".into(),
                citations: None,
                cache_control: None,
            },
        ]);

        assert_eq!(
            msg.reasoning_content,
            Some(ReasoningContent::Text("A".into()))
        );
        assert!(msg.reasoning_content.as_ref().unwrap().segments().is_none());
        assert!(matches!(
            msg.content,
            Some(ChatCompletionRequestAssistantMessageContent::Text(ref t)) if t == "answer"
        ));
    }

    #[test]
    fn test_single_thinking_then_single_tool() {
        // [Thinking("reason"), ToolUse("t1")] → Segments(["reason", ""])
        let msg = make_req(vec![thinking("reason"), tool_use("t1")]);

        let segs = msg
            .reasoning_content
            .as_ref()
            .expect("reasoning_content should be set")
            .segments()
            .expect("should be Segments variant");
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0], "reason");
        assert_eq!(segs[1], "");

        assert_eq!(
            msg.reasoning_content.as_ref().unwrap().to_flat_string(),
            "reason"
        );
    }

    // Regression test for the KV-cache flattening bug.
    //
    // OLD CODE: `convert_assistant_blocks` concatenated all thinking blocks into a
    // single flat string — `reasoning_content = Text("A\nB")`.  A chat template
    // given only that string can only reconstruct:
    //
    //     <think>A\nB</think> <call>t1</call> <call>t2</call>
    //
    // That token sequence diverges from what the model originally generated at the
    // very first `</think>`, so the KV cache misses on every multi-tool exchange.
    //
    // NEW CODE: `convert_assistant_blocks` produces `Segments(["A", "B", ""])` so a
    // template that understands segments can reconstruct byte-for-byte:
    //
    //     <think>A</think> <call>t1</call> <think>B</think> <call>t2</call>
    //
    // This test fails on the old code because the old code returns `Text("A\nB")` and
    // `.segments()` returns `None`, causing the `expect` below to panic.
    #[test]
    fn test_interleaved_reasoning_not_flattened_regression() {
        let msg = make_req(vec![
            thinking("A"),
            tool_use("t1"),
            thinking("B"),
            tool_use("t2"),
        ]);

        // Must be Segments, not Text.  Text("A\nB") is the old (broken) behaviour:
        // it loses which reasoning block preceded which tool call.
        assert!(
            !matches!(msg.reasoning_content, Some(ReasoningContent::Text(_))),
            "reasoning_content must NOT be flat Text when tool calls are interleaved; \
             Text loses positional info and forces a KV cache miss on every multi-tool turn"
        );

        let segs = msg
            .reasoning_content
            .as_ref()
            .expect("reasoning_content should be set")
            .segments()
            .expect(
                "must be Segments so a chat template can reconstruct \
                 <think>A</think><call>t1</call><think>B</think><call>t2</call> \
                 rather than front-loading all reasoning before all calls",
            );

        // segs[i] precedes tool_calls[i] — the invariant a template relies on
        assert_eq!(segs[0], "A", "reasoning before t1");
        assert_eq!(segs[1], "B", "reasoning before t2");
        assert_eq!(segs[2], "", "no trailing reasoning");

        let tools = msg.tool_calls.as_ref().unwrap();
        assert_eq!(tools[0].id, "t1");
        assert_eq!(tools[1].id, "t2");
    }

    #[test]
    fn test_per_block_cache_control_deserialization() {
        let json = r#"{
            "model": "test",
            "max_tokens": 100,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello", "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": "World"}
                ]
            }]
        }"#;
        let req: AnthropicCreateMessageRequest = serde_json::from_str(json).unwrap();
        match &req.messages[0].content {
            AnthropicMessageContent::Blocks { content } => {
                match &content[0] {
                    AnthropicContentBlock::Text { cache_control, .. } => {
                        assert!(cache_control.is_some());
                    }
                    other => panic!("expected Text, got {other:?}"),
                }
                match &content[1] {
                    AnthropicContentBlock::Text { cache_control, .. } => {
                        assert!(cache_control.is_none());
                    }
                    other => panic!("expected Text, got {other:?}"),
                }
            }
            _ => panic!("expected blocks"),
        }
    }

    #[test]
    fn test_system_string_no_cache_control() {
        let json = r#"{
            "model": "test",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hello"}],
            "system": "You are helpful."
        }"#;
        let req: AnthropicCreateMessageRequest = serde_json::from_str(json).unwrap();
        let system = req.system.as_ref().unwrap();
        assert_eq!(system.text, "You are helpful.");
        assert!(system.cache_control.is_none());
    }

    #[test]
    fn test_text_block_with_citations() {
        let json = r#"{
            "model": "test",
            "max_tokens": 100,
            "messages": [{
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "According to the document...",
                        "citations": [
                            {"type": "char_location", "cited_text": "relevant text", "document_index": 0, "start_char_index": 0, "end_char_index": 13}
                        ]
                    }
                ]
            }]
        }"#;
        let req: AnthropicCreateMessageRequest = serde_json::from_str(json).unwrap();
        match &req.messages[0].content {
            AnthropicMessageContent::Blocks { content } => match &content[0] {
                AnthropicContentBlock::Text { citations, .. } => {
                    assert!(citations.is_some());
                    let cites = citations.as_ref().unwrap();
                    assert_eq!(cites.len(), 1);
                    assert_eq!(cites[0]["type"], "char_location");
                }
                other => panic!("expected Text, got {other:?}"),
            },
            _ => panic!("expected blocks"),
        }
    }

    #[test]
    fn test_redacted_thinking_block() {
        let json = r#"{
            "model": "test",
            "max_tokens": 100,
            "messages": [{
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "visible reasoning", "signature": "sig1"},
                    {"type": "redacted_thinking", "data": "base64-encrypted-data"},
                    {"type": "text", "text": "Final answer"}
                ]
            }]
        }"#;
        let req: AnthropicCreateMessageRequest = serde_json::from_str(json).unwrap();
        match &req.messages[0].content {
            AnthropicMessageContent::Blocks { content } => {
                assert_eq!(content.len(), 3);
                assert!(matches!(
                    &content[0],
                    AnthropicContentBlock::Thinking { .. }
                ));
                match &content[1] {
                    AnthropicContentBlock::RedactedThinking { data } => {
                        assert_eq!(data, "base64-encrypted-data");
                    }
                    other => panic!("expected RedactedThinking, got {other:?}"),
                }
                assert!(matches!(&content[2], AnthropicContentBlock::Text { .. }));
            }
            _ => panic!("expected blocks"),
        }
    }

    #[test]
    fn test_server_tool_use_and_web_search_result() {
        let json = r#"{
            "model": "test",
            "max_tokens": 100,
            "messages": [{
                "role": "assistant",
                "content": [
                    {"type": "server_tool_use", "id": "stu_1", "name": "web_search", "input": {"query": "rust programming"}},
                    {"type": "web_search_tool_result", "tool_use_id": "stu_1", "content": [{"type": "web_search_result", "url": "https://www.rust-lang.org", "title": "Rust"}]},
                    {"type": "text", "text": "Based on my search..."}
                ]
            }]
        }"#;
        let req: AnthropicCreateMessageRequest = serde_json::from_str(json).unwrap();
        match &req.messages[0].content {
            AnthropicMessageContent::Blocks { content } => {
                assert_eq!(content.len(), 3);
                match &content[0] {
                    AnthropicContentBlock::ServerToolUse { id, name, input } => {
                        assert_eq!(id, "stu_1");
                        assert_eq!(name, "web_search");
                        assert_eq!(input["query"], "rust programming");
                    }
                    other => panic!("expected ServerToolUse, got {other:?}"),
                }
                match &content[1] {
                    AnthropicContentBlock::WebSearchToolResult {
                        tool_use_id,
                        content,
                    } => {
                        assert_eq!(tool_use_id, "stu_1");
                        assert!(content.is_array());
                    }
                    other => panic!("expected WebSearchToolResult, got {other:?}"),
                }
            }
            _ => panic!("expected blocks"),
        }

        // ServerToolUse should convert to a tool call
        let chat_req: NvCreateChatCompletionRequest = req.try_into().unwrap();
        match &chat_req.inner.messages[0] {
            ChatCompletionRequestMessage::Assistant(a) => {
                let tc = a.tool_calls.as_ref().expect("should have tool calls");
                assert_eq!(tc.len(), 1);
                assert_eq!(tc[0].id, "stu_1");
                assert_eq!(tc[0].function.name, "web_search");
            }
            other => panic!("expected assistant, got {other:?}"),
        }
    }

    #[test]
    fn test_thinking_config_deserialization() {
        let json = r#"{
            "model": "test",
            "max_tokens": 16000,
            "messages": [{"role": "user", "content": "Solve this step by step"}],
            "thinking": {"type": "enabled", "budget_tokens": 10000}
        }"#;
        let req: AnthropicCreateMessageRequest = serde_json::from_str(json).unwrap();
        let thinking = req.thinking.as_ref().expect("thinking should be set");
        assert_eq!(thinking.thinking_type, "enabled");
        assert_eq!(thinking.budget_tokens, Some(10000));
    }

    #[test]
    fn test_thinking_config_disabled() {
        let json = r#"{
            "model": "test",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hello"}],
            "thinking": {"type": "disabled"}
        }"#;
        let req: AnthropicCreateMessageRequest = serde_json::from_str(json).unwrap();
        let thinking = req.thinking.as_ref().expect("thinking should be set");
        assert_eq!(thinking.thinking_type, "disabled");
        assert!(thinking.budget_tokens.is_none());
    }

    #[test]
    fn test_disable_parallel_tool_use() {
        let json = r#"{
            "model": "test",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hello"}],
            "tools": [{"name": "get_weather", "description": "Get weather", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "auto", "disable_parallel_tool_use": true}
        }"#;
        let req: AnthropicCreateMessageRequest = serde_json::from_str(json).unwrap();
        match &req.tool_choice {
            Some(AnthropicToolChoice::Simple(s)) => {
                assert_eq!(s.choice_type, AnthropicToolChoiceMode::Auto);
                assert_eq!(s.disable_parallel_tool_use, Some(true));
            }
            other => panic!("expected Simple tool choice, got {other:?}"),
        }
    }

    // --- Image passthrough tests ---

    #[test]
    fn test_image_block_becomes_multimodal_content() {
        let req = AnthropicCreateMessageRequest {
            model: "test-model".into(),
            max_tokens: 100,
            messages: vec![AnthropicMessage {
                role: AnthropicRole::User,
                content: AnthropicMessageContent::Blocks {
                    content: vec![
                        AnthropicContentBlock::Text {
                            text: "What is in this image?".into(),
                            citations: None,
                            cache_control: None,
                        },
                        AnthropicContentBlock::Image {
                            source: AnthropicImageSource {
                                source_type: "base64".into(),
                                media_type: "image/png".into(),
                                data: "iVBORw0KGgo=".into(), // tiny valid-ish base64
                            },
                        },
                    ],
                },
            }],
            system: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: false,
            metadata: None,
            tools: None,
            tool_choice: None,
            cache_control: None,
            thinking: None,
            service_tier: None,
            container: None,
            output_config: None,
        };

        let chat_req: NvCreateChatCompletionRequest = req.try_into().unwrap();
        assert_eq!(chat_req.inner.messages.len(), 1);

        match &chat_req.inner.messages[0] {
            ChatCompletionRequestMessage::User(u) => match &u.content {
                ChatCompletionRequestUserMessageContent::Array(parts) => {
                    assert_eq!(parts.len(), 2);
                    // First part: text
                    match &parts[0] {
                        ChatCompletionRequestUserMessageContentPart::Text(t) => {
                            assert_eq!(t.text, "What is in this image?");
                        }
                        other => panic!("expected text part, got {other:?}"),
                    }
                    // Second part: image with data URI
                    match &parts[1] {
                        ChatCompletionRequestUserMessageContentPart::ImageUrl(img) => {
                            let url_str = img.image_url.url.to_string();
                            assert!(
                                url_str.starts_with("data:image/png;base64,"),
                                "expected data URI, got: {url_str}"
                            );
                            assert!(url_str.contains("iVBORw0KGgo="));
                        }
                        other => panic!("expected image_url part, got {other:?}"),
                    }
                }
                other => panic!("expected Array content, got {other:?}"),
            },
            other => panic!("expected user message, got {other:?}"),
        }
    }

    #[test]
    fn test_pure_text_stays_text_format() {
        // Verify backwards compatibility: pure text messages don't use Array format.
        let req = AnthropicCreateMessageRequest {
            model: "test-model".into(),
            max_tokens: 100,
            messages: vec![AnthropicMessage {
                role: AnthropicRole::User,
                content: AnthropicMessageContent::Blocks {
                    content: vec![
                        AnthropicContentBlock::Text {
                            text: "Hello ".into(),
                            citations: None,
                            cache_control: None,
                        },
                        AnthropicContentBlock::Text {
                            text: "world".into(),
                            citations: None,
                            cache_control: None,
                        },
                    ],
                },
            }],
            system: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: false,
            metadata: None,
            tools: None,
            tool_choice: None,
            cache_control: None,
            thinking: None,
            service_tier: None,
            container: None,
            output_config: None,
        };

        let chat_req: NvCreateChatCompletionRequest = req.try_into().unwrap();
        match &chat_req.inner.messages[0] {
            ChatCompletionRequestMessage::User(u) => match &u.content {
                ChatCompletionRequestUserMessageContent::Text(t) => {
                    assert_eq!(t, "Hello world");
                }
                other => panic!("expected Text content (not Array), got {other:?}"),
            },
            other => panic!("expected user message, got {other:?}"),
        }
    }

    #[test]
    fn test_image_with_tool_result_flush() {
        // Image + text should flush as Array before tool_result becomes a Tool message.
        let req = AnthropicCreateMessageRequest {
            model: "test-model".into(),
            max_tokens: 100,
            messages: vec![
                AnthropicMessage {
                    role: AnthropicRole::User,
                    content: AnthropicMessageContent::Text {
                        content: "What's the weather?".into(),
                    },
                },
                AnthropicMessage {
                    role: AnthropicRole::Assistant,
                    content: AnthropicMessageContent::Blocks {
                        content: vec![AnthropicContentBlock::ToolUse {
                            id: "tool_1".into(),
                            name: "screenshot".into(),
                            input: serde_json::json!({}),
                            cache_control: None,
                        }],
                    },
                },
                AnthropicMessage {
                    role: AnthropicRole::User,
                    content: AnthropicMessageContent::Blocks {
                        content: vec![
                            AnthropicContentBlock::Image {
                                source: AnthropicImageSource {
                                    source_type: "base64".into(),
                                    media_type: "image/jpeg".into(),
                                    data: "/9j/4AAQ".into(),
                                },
                            },
                            AnthropicContentBlock::ToolResult {
                                tool_use_id: "tool_1".into(),
                                content: Some(ToolResultContent::Text("screenshot taken".into())),
                                is_error: None,
                                cache_control: None,
                            },
                        ],
                    },
                },
            ],
            system: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: false,
            metadata: None,
            tools: None,
            tool_choice: None,
            cache_control: None,
            thinking: None,
            service_tier: None,
            container: None,
            output_config: None,
        };

        let chat_req: NvCreateChatCompletionRequest = req.try_into().unwrap();
        // user("What's the weather?"), assistant(tool_use), user(image), tool("screenshot taken")
        assert_eq!(chat_req.inner.messages.len(), 4);

        // Third message: user with image (Array format, flushed before tool_result)
        match &chat_req.inner.messages[2] {
            ChatCompletionRequestMessage::User(u) => match &u.content {
                ChatCompletionRequestUserMessageContent::Array(parts) => {
                    assert_eq!(parts.len(), 1);
                    assert!(matches!(
                        &parts[0],
                        ChatCompletionRequestUserMessageContentPart::ImageUrl(_)
                    ));
                }
                other => panic!("expected Array content for image, got {other:?}"),
            },
            other => panic!("expected user message, got {other:?}"),
        }

        // Fourth message: tool result
        assert!(matches!(
            &chat_req.inner.messages[3],
            ChatCompletionRequestMessage::Tool(_)
        ));
    }
}
