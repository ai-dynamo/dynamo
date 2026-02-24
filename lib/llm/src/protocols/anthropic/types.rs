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
    ChatCompletionToolType, FunctionName, FunctionObject, ReasoningContent,
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::protocols::openai::chat_completions::{
    NvCreateChatCompletionRequest, NvCreateChatCompletionResponse,
};
use crate::protocols::openai::common_ext::CommonExt;

// ---------------------------------------------------------------------------
// Custom deserializers
// ---------------------------------------------------------------------------

/// Deserialize `system` from either a plain string or an array of text blocks.
/// The Anthropic API accepts both `"system": "text"` and
/// `"system": [{"type": "text", "text": "..."}]`.
fn deserialize_system_prompt<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum SystemPrompt {
        Text(String),
        Blocks(Vec<SystemBlock>),
    }

    #[derive(Deserialize)]
    struct SystemBlock {
        text: String,
    }

    let maybe: Option<SystemPrompt> = Option::deserialize(deserializer)?;
    Ok(maybe.map(|sp| match sp {
        SystemPrompt::Text(s) => s,
        SystemPrompt::Blocks(blocks) => blocks
            .into_iter()
            .map(|b| b.text)
            .collect::<Vec<_>>()
            .join("\n"),
    }))
}

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

    /// Optional system prompt (string or array of `{"type":"text","text":"..."}` blocks).
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_system_prompt"
    )]
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
///
/// Uses a custom deserializer so that unknown block types (e.g. `citations`,
/// `server_tool_use`, `redacted_thinking`) are captured as `Unknown` instead
/// of causing a hard deserialization failure. This is important because Claude
/// Code may send block types that we don't yet handle.
#[derive(Debug, Clone, Serialize)]
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
        #[serde(default, skip_serializing_if = "Option::is_none")]
        content: Option<ToolResultContent>,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },
    /// Thinking content block from assistant (extended thinking / reasoning).
    #[serde(rename = "thinking")]
    Thinking { thinking: String, signature: String },
    /// Catch-all for unrecognized block types. Silently accepted and skipped
    /// during conversion so that new Anthropic features don't break the endpoint.
    #[serde(skip)]
    Unknown { block_type: String },
}

/// Content of a `tool_result` block — either a plain string or an array of
/// content blocks (the Anthropic API accepts both).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolResultContent {
    Text(String),
    Blocks(Vec<ToolResultContentBlock>),
}

impl ToolResultContent {
    /// Extract the text content, concatenating array blocks if needed.
    pub fn into_text(self) -> String {
        match self {
            ToolResultContent::Text(s) => s,
            ToolResultContent::Blocks(blocks) => blocks
                .into_iter()
                .filter_map(|b| match b {
                    ToolResultContentBlock::Text { text } => Some(text),
                    ToolResultContentBlock::Other(_) => None,
                })
                .collect::<Vec<_>>()
                .join(""),
        }
    }
}

/// A content block within a `tool_result.content` array.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolResultContentBlock {
    Text {
        text: String,
    },
    /// Catch-all for non-text blocks (images, etc.) in tool results.
    Other(serde_json::Value),
}

/// Custom deserializer for `AnthropicContentBlock` that handles unknown types
/// gracefully. Since serde's `#[serde(other)]` is not supported on internally
/// tagged enums, we deserialize as `Value` first and dispatch manually.
impl<'de> Deserialize<'de> for AnthropicContentBlock {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;
        let block_type = value
            .get("type")
            .and_then(|t| t.as_str())
            .unwrap_or("")
            .to_string();

        match block_type.as_str() {
            "text" => {
                let text = value
                    .get("text")
                    .and_then(|t| t.as_str())
                    .unwrap_or("")
                    .to_string();
                Ok(AnthropicContentBlock::Text { text })
            }
            "image" => {
                let source: AnthropicImageSource =
                    serde_json::from_value(value.get("source").cloned().unwrap_or_default())
                        .map_err(serde::de::Error::custom)?;
                Ok(AnthropicContentBlock::Image { source })
            }
            "tool_use" => {
                let id = value
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let name = value
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let input = value.get("input").cloned().unwrap_or(serde_json::json!({}));
                Ok(AnthropicContentBlock::ToolUse { id, name, input })
            }
            "tool_result" => {
                let tool_use_id = value
                    .get("tool_use_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let content: Option<ToolResultContent> = value
                    .get("content")
                    .cloned()
                    .and_then(|v| serde_json::from_value(v).ok());
                let is_error = value.get("is_error").and_then(|v| v.as_bool());
                Ok(AnthropicContentBlock::ToolResult {
                    tool_use_id,
                    content,
                    is_error,
                })
            }
            "thinking" => {
                let thinking = value
                    .get("thinking")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let signature = value
                    .get("signature")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                Ok(AnthropicContentBlock::Thinking {
                    thinking,
                    signature,
                })
            }
            other => {
                tracing::debug!("Unknown Anthropic content block type '{}', skipping", other);
                Ok(AnthropicContentBlock::Unknown {
                    block_type: other.to_string(),
                })
            }
        }
    }
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
    /// Named tool: `{type: "tool", name: "..."}`
    /// Must be listed before Simple so serde tries the stricter shape first.
    Named(AnthropicToolChoiceNamed),
    /// Simple mode: "auto", "any", or "none".
    Simple(AnthropicToolChoiceSimple),
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
            .map(dynamo_async_openai::types::Stop::StringArray);

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
                stream_options: Some(dynamo_async_openai::types::ChatCompletionStreamOptions {
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
                let text = content.clone().map(|c| c.into_text()).unwrap_or_default();
                messages.push(ChatCompletionRequestMessage::Tool(
                    ChatCompletionRequestToolMessage {
                        content: ChatCompletionRequestToolMessageContent::Text(text),
                        tool_call_id: tool_use_id.clone(),
                    },
                ));
            }
            AnthropicContentBlock::Image { .. } => {
                tracing::warn!(
                    "Image content blocks are not supported in the Anthropic-to-chat-completions conversion; replaced with placeholder text."
                );
                text_parts.push("[image]".to_string());
            }
            AnthropicContentBlock::ToolUse { .. }
            | AnthropicContentBlock::Thinking { .. }
            | AnthropicContentBlock::Unknown { .. } => {
                // tool_use/thinking/unknown in a user message: skip
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
            AnthropicContentBlock::Text { text } => {
                text_content.push_str(text);
            }
            AnthropicContentBlock::Thinking { thinking, .. } => {
                if !pending_reasoning.is_empty() {
                    pending_reasoning.push('\n');
                }
                pending_reasoning.push_str(thinking);
            }
            AnthropicContentBlock::ToolUse { id, name, input } => {
                // Snapshot the reasoning that preceded this tool call.
                segments.push(std::mem::take(&mut pending_reasoning));
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
            dynamo_async_openai::types::FinishReason::ContentFilter => AnthropicStopReason::EndTurn,
            dynamo_async_openai::types::FinishReason::FunctionCall => AnthropicStopReason::ToolUse,
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
                tracing::warn!(
                    "Multimodal (Parts) content in chat completion response replaced with placeholder text in Anthropic conversion."
                );
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
// Count tokens
// ---------------------------------------------------------------------------

/// Request body for `POST /v1/messages/count_tokens`.
#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicCountTokensRequest {
    pub model: String,
    pub messages: Vec<AnthropicMessage>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_system_prompt"
    )]
    pub system: Option<String>,
    #[serde(default)]
    pub tools: Option<Vec<AnthropicTool>>,
}

/// Response body for `POST /v1/messages/count_tokens`.
#[derive(Debug, Clone, Serialize)]
pub struct AnthropicCountTokensResponse {
    pub input_tokens: u32,
}

impl AnthropicCountTokensRequest {
    /// Estimate input token count using a `len/3` heuristic.
    pub fn estimate_tokens(&self) -> u32 {
        let mut total_len: usize = 0;

        if let Some(system) = &self.system {
            total_len += system.len();
        }

        for msg in &self.messages {
            // Count role
            total_len += match msg.role {
                AnthropicRole::User => 4,
                AnthropicRole::Assistant => 9,
            };
            // Count content
            match &msg.content {
                AnthropicMessageContent::Text { content } => total_len += content.len(),
                AnthropicMessageContent::Blocks { content } => {
                    for block in content {
                        total_len += estimate_block_len(block);
                    }
                }
            }
        }

        if let Some(tools) = &self.tools {
            for tool in tools {
                total_len += tool.name.len();
                if let Some(desc) = &tool.description {
                    total_len += desc.len();
                }
                total_len += tool.input_schema.to_string().len();
            }
        }

        let tokens = total_len / 3;
        if tokens == 0 && total_len > 0 {
            1
        } else {
            tokens as u32
        }
    }
}

fn estimate_block_len(block: &AnthropicContentBlock) -> usize {
    match block {
        AnthropicContentBlock::Text { text } => text.len(),
        AnthropicContentBlock::ToolUse { name, input, .. } => name.len() + input.to_string().len(),
        AnthropicContentBlock::ToolResult { content, .. } => content
            .as_ref()
            .map(|c| match c {
                ToolResultContent::Text(s) => s.len(),
                ToolResultContent::Blocks(blocks) => blocks
                    .iter()
                    .map(|b| match b {
                        ToolResultContentBlock::Text { text } => text.len(),
                        ToolResultContentBlock::Other(v) => v.to_string().len(),
                    })
                    .sum(),
            })
            .unwrap_or(0),
        AnthropicContentBlock::Thinking { thinking, .. } => thinking.len(),
        AnthropicContentBlock::Image { .. } => 256, // rough estimate for image metadata
        AnthropicContentBlock::Unknown { .. } => 0,
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
                            content: Some(ToolResultContent::Text("72F and sunny".into())),
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
                        },
                        AnthropicContentBlock::Text {
                            text: "Answer".into(),
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
    fn test_unknown_block_type_does_not_fail() {
        let json = r#"{
            "model": "test",
            "max_tokens": 100,
            "messages": [{
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "hello"},
                    {"type": "server_tool_use", "id": "stu_1", "name": "web_search", "input": {}},
                    {"type": "redacted_thinking", "data": "encrypted"},
                    {"type": "text", "text": "world"}
                ]
            }]
        }"#;
        let req: AnthropicCreateMessageRequest = serde_json::from_str(json).unwrap();
        match &req.messages[0].content {
            AnthropicMessageContent::Blocks { content } => {
                assert_eq!(content.len(), 4);
                assert!(matches!(&content[0], AnthropicContentBlock::Text { .. }));
                assert!(matches!(
                    &content[1],
                    AnthropicContentBlock::Unknown { block_type } if block_type == "server_tool_use"
                ));
                assert!(matches!(
                    &content[2],
                    AnthropicContentBlock::Unknown { block_type } if block_type == "redacted_thinking"
                ));
                assert!(matches!(&content[3], AnthropicContentBlock::Text { .. }));
            }
            _ => panic!("expected blocks content"),
        }

        // Conversion should succeed, skipping unknown blocks
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
        }
        .try_into()
        .unwrap();
        assert_eq!(chat_req.inner.messages.len(), 1);
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
            system: Some("You are helpful.".into()),
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
        }
    }

    fn thinking(text: &str) -> AnthropicContentBlock {
        AnthropicContentBlock::Thinking {
            thinking: text.into(),
            signature: "sig".into(),
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
}
