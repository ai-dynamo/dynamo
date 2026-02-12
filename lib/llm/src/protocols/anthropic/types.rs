// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Anthropic Messages API types and conversion logic.
//!
//! All request/response types for the `/v1/messages` endpoint, plus
//! bidirectional conversion to/from the internal chat completions format.

use dynamo_async_openai::types::{
    ChatCompletionMessageToolCall, ChatCompletionNamedToolChoice,
    ChatCompletionRequestAssistantMessage, ChatCompletionRequestAssistantMessageContent,
    ChatCompletionRequestMessage, ChatCompletionRequestSystemMessage,
    ChatCompletionRequestSystemMessageContent, ChatCompletionRequestToolMessage,
    ChatCompletionRequestToolMessageContent, ChatCompletionRequestUserMessage,
    ChatCompletionRequestUserMessageContent, ChatCompletionTool, ChatCompletionToolChoiceOption,
    ChatCompletionToolType, FunctionName, FunctionObject,
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::protocols::openai::chat_completions::{
    NvCreateChatCompletionRequest, NvCreateChatCompletionResponse,
};

// ---------------------------------------------------------------------------
// Request types
// ---------------------------------------------------------------------------

/// Top-level request body for `POST /v1/messages`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicCreateMessageRequest {
    /// The model to use (e.g. "claude-sonnet-4-20250514").
    pub model: String,

    /// The maximum number of tokens to generate.
    pub max_tokens: u32,

    /// The conversation messages.
    pub messages: Vec<AnthropicMessage>,

    /// Optional system prompt.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,

    /// Sampling temperature (0.0 - 1.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// Nucleus sampling parameter.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    /// Top-K sampling parameter.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,

    /// Custom stop sequences.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,

    /// Whether to stream the response.
    #[serde(default)]
    pub stream: bool,

    /// Optional metadata (e.g. user_id).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,

    /// Tools the model may call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<AnthropicTool>>,

    /// How the model should choose which tool to call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<AnthropicToolChoice>,
}

/// A single message in the conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicMessage {
    pub role: AnthropicRole,
    #[serde(flatten)]
    pub content: AnthropicMessageContent,
}

/// The role of a message sender.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum AnthropicRole {
    User,
    Assistant,
}

/// Message content — either a plain string or an array of content blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AnthropicMessageContent {
    /// Plain text content.
    Text { content: String },
    /// Array of structured content blocks.
    Blocks { content: Vec<AnthropicContentBlock> },
}

/// A single content block within a message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AnthropicContentBlock {
    /// Text content block.
    #[serde(rename = "text")]
    Text { text: String },
    /// Image content block.
    #[serde(rename = "image")]
    Image { source: AnthropicImageSource },
    /// Tool use request from assistant.
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    /// Tool result from user.
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        content: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },
}

/// Image source for image content blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicImageSource {
    #[serde(rename = "type")]
    pub source_type: String,
    pub media_type: String,
    pub data: String,
}

/// A tool definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicTool {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub input_schema: serde_json::Value,
}

/// Tool choice specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AnthropicToolChoice {
    /// Simple mode: "auto", "any", or "none".
    Simple(AnthropicToolChoiceSimple),
    /// Named tool: `{type: "tool", name: "..."}`
    Named(AnthropicToolChoiceNamed),
}

/// Simple tool choice modes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicToolChoiceSimple {
    #[serde(rename = "type")]
    pub choice_type: AnthropicToolChoiceMode,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum AnthropicToolChoiceMode {
    Auto,
    Any,
    None,
    Tool,
}

/// Named tool choice.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicToolChoiceNamed {
    #[serde(rename = "type")]
    pub choice_type: AnthropicToolChoiceMode,
    pub name: String,
}

// ---------------------------------------------------------------------------
// Response types
// ---------------------------------------------------------------------------

/// Response body for `POST /v1/messages` (non-streaming).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicMessageResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub object_type: String,
    pub role: String,
    pub content: Vec<AnthropicResponseContentBlock>,
    pub model: String,
    pub stop_reason: Option<AnthropicStopReason>,
    pub stop_sequence: Option<String>,
    pub usage: AnthropicUsage,
}

/// A content block in the response.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AnthropicResponseContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
}

/// Token usage information.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AnthropicUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

/// Reason the model stopped generating.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AnthropicStopReason {
    EndTurn,
    MaxTokens,
    StopSequence,
    ToolUse,
}

// ---------------------------------------------------------------------------
// Streaming types
// ---------------------------------------------------------------------------

/// SSE event types for the Anthropic streaming API.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AnthropicStreamEvent {
    #[serde(rename = "message_start")]
    MessageStart { message: AnthropicMessageResponse },

    #[serde(rename = "content_block_start")]
    ContentBlockStart {
        index: u32,
        content_block: AnthropicResponseContentBlock,
    },

    #[serde(rename = "content_block_delta")]
    ContentBlockDelta { index: u32, delta: AnthropicDelta },

    #[serde(rename = "content_block_stop")]
    ContentBlockStop { index: u32 },

    #[serde(rename = "message_delta")]
    MessageDelta {
        delta: AnthropicMessageDeltaBody,
        usage: AnthropicUsage,
    },

    #[serde(rename = "message_stop")]
    MessageStop {},

    #[serde(rename = "ping")]
    Ping {},

    #[serde(rename = "error")]
    Error { error: AnthropicErrorBody },
}

/// Delta content in a streaming content_block_delta event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AnthropicDelta {
    #[serde(rename = "text_delta")]
    TextDelta { text: String },
    #[serde(rename = "input_json_delta")]
    InputJsonDelta { partial_json: String },
}

/// The delta body in a message_delta event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicMessageDeltaBody {
    pub stop_reason: Option<AnthropicStopReason>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequence: Option<String>,
}

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Anthropic API error response wrapper.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicErrorResponse {
    #[serde(rename = "type")]
    pub object_type: String,
    pub error: AnthropicErrorBody,
}

/// Error body within an error response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicErrorBody {
    #[serde(rename = "type")]
    pub error_type: String,
    pub message: String,
}

impl AnthropicErrorResponse {
    /// Create an `invalid_request_error` response.
    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self {
            object_type: "error".to_string(),
            error: AnthropicErrorBody {
                error_type: "invalid_request_error".to_string(),
                message: message.into(),
            },
        }
    }

    /// Create an `api_error` (internal server error) response.
    pub fn api_error(message: impl Into<String>) -> Self {
        Self {
            object_type: "error".to_string(),
            error: AnthropicErrorBody {
                error_type: "api_error".to_string(),
                message: message.into(),
            },
        }
    }

    /// Create a `not_found_error` response.
    pub fn not_found(message: impl Into<String>) -> Self {
        Self {
            object_type: "error".to_string(),
            error: AnthropicErrorBody {
                error_type: "not_found_error".to_string(),
                message: message.into(),
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Conversion: AnthropicCreateMessageRequest -> NvCreateChatCompletionRequest
// ---------------------------------------------------------------------------

impl TryFrom<AnthropicCreateMessageRequest> for NvCreateChatCompletionRequest {
    type Error = anyhow::Error;

    fn try_from(req: AnthropicCreateMessageRequest) -> Result<Self, Self::Error> {
        let mut messages = Vec::new();

        // Prepend system message if present
        if let Some(system_text) = &req.system {
            messages.push(ChatCompletionRequestMessage::System(
                ChatCompletionRequestSystemMessage {
                    content: ChatCompletionRequestSystemMessageContent::Text(system_text.clone()),
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
                        ChatCompletionRequestAssistantMessage {
                            content: Some(ChatCompletionRequestAssistantMessageContent::Text(
                                content.clone(),
                            )),
                            reasoning_content: None,
                            refusal: None,
                            name: None,
                            audio: None,
                            tool_calls: None,
                            #[allow(deprecated)]
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
        let stop = req.stop_sequences.map(|seqs| {
            dynamo_async_openai::types::Stop::StringArray(seqs)
        });

        Ok(NvCreateChatCompletionRequest {
            inner: dynamo_async_openai::types::CreateChatCompletionRequest {
                messages,
                model: req.model,
                temperature: req.temperature,
                top_p: req.top_p,
                max_completion_tokens: Some(req.max_tokens),
                stop,
                tools,
                tool_choice,
                stream: Some(true), // Always stream internally
                ..Default::default()
            },
            common: Default::default(),
            nvext: None,
            chat_template_args: None,
            media_io_kwargs: None,
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
    // Gather text blocks for a single user message, emit tool_result blocks as Tool messages.
    let mut text_parts = Vec::new();

    for block in blocks {
        match block {
            AnthropicContentBlock::Text { text } => {
                text_parts.push(text.clone());
            }
            AnthropicContentBlock::ToolResult {
                tool_use_id,
                content,
                ..
            } => {
                // Flush any accumulated text first
                if !text_parts.is_empty() {
                    let combined = text_parts.join("");
                    messages.push(ChatCompletionRequestMessage::User(
                        ChatCompletionRequestUserMessage {
                            content: ChatCompletionRequestUserMessageContent::Text(combined),
                            name: None,
                        },
                    ));
                    text_parts.clear();
                }
                messages.push(ChatCompletionRequestMessage::Tool(
                    ChatCompletionRequestToolMessage {
                        content: ChatCompletionRequestToolMessageContent::Text(
                            content.clone().unwrap_or_default(),
                        ),
                        tool_call_id: tool_use_id.clone(),
                    },
                ));
            }
            AnthropicContentBlock::Image { .. } => {
                // Image blocks are not yet supported in this conversion
                text_parts.push("[image]".to_string());
            }
            AnthropicContentBlock::ToolUse { .. } => {
                // tool_use in a user message is unexpected, skip
            }
        }
    }

    // Flush remaining text
    if !text_parts.is_empty() {
        let combined = text_parts.join("");
        messages.push(ChatCompletionRequestMessage::User(
            ChatCompletionRequestUserMessage {
                content: ChatCompletionRequestUserMessageContent::Text(combined),
                name: None,
            },
        ));
    }

    Ok(())
}

/// Convert assistant-role content blocks into chat completion messages.
/// Text blocks become an assistant message; tool_use blocks become tool_calls on an assistant message.
fn convert_assistant_blocks(
    blocks: &[AnthropicContentBlock],
    messages: &mut Vec<ChatCompletionRequestMessage>,
) {
    let mut text_content = String::new();
    let mut tool_calls = Vec::new();

    for block in blocks {
        match block {
            AnthropicContentBlock::Text { text } => {
                text_content.push_str(text);
            }
            AnthropicContentBlock::ToolUse { id, name, input } => {
                tool_calls.push(ChatCompletionMessageToolCall {
                    id: id.clone(),
                    r#type: ChatCompletionToolType::Function,
                    function: dynamo_async_openai::types::FunctionCall {
                        name: name.clone(),
                        arguments: serde_json::to_string(input).unwrap_or_default(),
                    },
                });
            }
            _ => {}
        }
    }

    let content = if text_content.is_empty() {
        None
    } else {
        Some(ChatCompletionRequestAssistantMessageContent::Text(
            text_content,
        ))
    };

    let tc = if tool_calls.is_empty() {
        None
    } else {
        Some(tool_calls)
    };

    messages.push(ChatCompletionRequestMessage::Assistant(
        ChatCompletionRequestAssistantMessage {
            content,
            reasoning_content: None,
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
        .map(|tool| ChatCompletionTool {
            r#type: ChatCompletionToolType::Function,
            function: FunctionObject {
                name: tool.name.clone(),
                description: tool.description.clone(),
                parameters: Some(tool.input_schema.clone()),
                strict: None,
            },
        })
        .collect()
}

/// Convert Anthropic tool_choice to ChatCompletionToolChoiceOption.
fn convert_anthropic_tool_choice(
    tc: &AnthropicToolChoice,
) -> ChatCompletionToolChoiceOption {
    match tc {
        AnthropicToolChoice::Simple(simple) => match simple.choice_type {
            AnthropicToolChoiceMode::Auto => ChatCompletionToolChoiceOption::Auto,
            AnthropicToolChoiceMode::Any => ChatCompletionToolChoiceOption::Required,
            AnthropicToolChoiceMode::None => ChatCompletionToolChoiceOption::None,
            AnthropicToolChoiceMode::Tool => ChatCompletionToolChoiceOption::Auto,
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

// ---------------------------------------------------------------------------
// Conversion: NvCreateChatCompletionResponse -> AnthropicMessageResponse
// ---------------------------------------------------------------------------

/// Convert a completed chat completion response into an Anthropic Messages response.
pub fn chat_completion_to_anthropic_response(
    chat_resp: NvCreateChatCompletionResponse,
    model: &str,
) -> AnthropicMessageResponse {
    let msg_id = format!("msg_{}", Uuid::new_v4().simple());

    let choice = chat_resp.choices.into_iter().next();
    let mut content = Vec::new();
    let mut stop_reason = None;

    if let Some(choice) = choice {
        // Map finish_reason
        stop_reason = choice.finish_reason.map(|fr| match fr {
            dynamo_async_openai::types::FinishReason::Stop => AnthropicStopReason::EndTurn,
            dynamo_async_openai::types::FinishReason::Length => AnthropicStopReason::MaxTokens,
            dynamo_async_openai::types::FinishReason::ToolCalls => AnthropicStopReason::ToolUse,
            dynamo_async_openai::types::FinishReason::ContentFilter => {
                AnthropicStopReason::EndTurn
            }
            dynamo_async_openai::types::FinishReason::FunctionCall => {
                AnthropicStopReason::ToolUse
            }
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

        // Extract text content
        let text = match choice.message.content {
            Some(dynamo_async_openai::types::ChatCompletionMessageContent::Text(t)) => Some(t),
            Some(dynamo_async_openai::types::ChatCompletionMessageContent::Parts(_)) => {
                Some("[multimodal content]".to_string())
            }
            None => None,
        };
        if let Some(text) = text {
            // Text goes first in the content array
            content.insert(0, AnthropicResponseContentBlock::Text { text });
        }
    }

    // Ensure there's at least one content block
    if content.is_empty() {
        content.push(AnthropicResponseContentBlock::Text {
            text: String::new(),
        });
    }

    // Map usage
    let usage = chat_resp
        .usage
        .map(|u| AnthropicUsage {
            input_tokens: u.prompt_tokens,
            output_tokens: u.completion_tokens,
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

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
            system: Some("You are helpful.".into()),
            temperature: None,
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: false,
            metadata: None,
            tools: None,
            tool_choice: None,
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
                        }],
                    },
                },
                AnthropicMessage {
                    role: AnthropicRole::User,
                    content: AnthropicMessageContent::Blocks {
                        content: vec![AnthropicContentBlock::ToolResult {
                            tool_use_id: "tool_123".into(),
                            content: Some("72F and sunny".into()),
                            is_error: None,
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
                description: Some("Get weather info".into()),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"]
                }),
            }]),
            tool_choice: Some(AnthropicToolChoice::Simple(AnthropicToolChoiceSimple {
                choice_type: AnthropicToolChoiceMode::Auto,
            })),
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
            id: "chatcmpl-xyz".into(),
            choices: vec![dynamo_async_openai::types::ChatChoice {
                index: 0,
                message: dynamo_async_openai::types::ChatCompletionResponseMessage {
                    content: Some(
                        dynamo_async_openai::types::ChatCompletionMessageContent::Text(
                            "Hello!".to_string(),
                        ),
                    ),
                    refusal: None,
                    tool_calls: None,
                    role: dynamo_async_openai::types::Role::Assistant,
                    function_call: None,
                    audio: None,
                    reasoning_content: None,
                },
                finish_reason: Some(dynamo_async_openai::types::FinishReason::Stop),
                stop_reason: None,
                logprobs: None,
            }],
            created: 1726000000,
            model: "test-model".into(),
            service_tier: None,
            system_fingerprint: None,
            object: "chat.completion".to_string(),
            usage: Some(dynamo_async_openai::types::CompletionUsage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
                prompt_tokens_details: None,
                completion_tokens_details: None,
            }),
            nvext: None,
        };

        let response = chat_completion_to_anthropic_response(chat_resp, "test-model");
        assert!(response.id.starts_with("msg_"));
        assert_eq!(response.object_type, "message");
        assert_eq!(response.role, "assistant");
        assert_eq!(response.model, "test-model");
        assert_eq!(response.stop_reason, Some(AnthropicStopReason::EndTurn));
        assert_eq!(response.usage.input_tokens, 10);
        assert_eq!(response.usage.output_tokens, 5);
        assert_eq!(response.content.len(), 1);
        match &response.content[0] {
            AnthropicResponseContentBlock::Text { text } => {
                assert_eq!(text, "Hello!");
            }
            _ => panic!("expected text block"),
        }
    }

    #[test]
    fn test_deserialize_simple_message() {
        let json = r#"{"model":"test","max_tokens":100,"messages":[{"role":"user","content":"Hello"}]}"#;
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
}
