// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Anthropic Messages API types.
//!
//! Pure protocol types for the `/v1/messages` endpoint -- request, response,
//! streaming events, error shapes, and count-tokens types.

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
/// Anthropic-style cache control hint for prefix pinning with TTL.
#[derive(ToSchema, Serialize, Deserialize, Debug, Clone, Default, PartialEq)]
pub struct CacheControl {
    #[serde(rename = "type")]
    pub control_type: CacheControlType,
    /// TTL as seconds (integer) or shorthand ("5m" = 300s, "1h" = 3600s). Clamped to [300, 3600].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttl: Option<String>,
}

#[derive(ToSchema, Serialize, Deserialize, Debug, Clone, Default, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum CacheControlType {
    #[default]
    Ephemeral,
    #[serde(other)]
    Unknown,
}

const MIN_TTL_SECONDS: u64 = 300;
const MAX_TTL_SECONDS: u64 = 3600;

impl CacheControl {
    /// Parse TTL string to seconds, clamped to [300, 3600].
    ///
    /// Accepts integer seconds ("120", "600") or shorthand ("5m", "1h").
    /// Values below 300 are clamped to 300; values above 3600 are clamped to 3600.
    /// Unrecognized strings default to 300s.
    pub fn ttl_seconds(&self) -> u64 {
        let raw = match self.ttl.as_deref() {
            None => return MIN_TTL_SECONDS,
            Some("5m") => 300,
            Some("1h") => 3600,
            Some(other) => match other.parse::<u64>() {
                Ok(secs) => secs,
                Err(_) => {
                    tracing::warn!("Unrecognized TTL '{}', defaulting to 300s", other);
                    return MIN_TTL_SECONDS;
                }
            },
        };
        raw.clamp(MIN_TTL_SECONDS, MAX_TTL_SECONDS)
    }
}
/// Parsed system prompt content, preserving cache_control from block arrays.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemContent {
    /// The concatenated text from all system blocks (or the plain string).
    pub text: String,
    /// Cache control from the last system block that had one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
}

/// Deserialize `system` from either a plain string or an array of text blocks.
/// The Anthropic API accepts both `"system": "text"` and
/// `"system": [{"type": "text", "text": "...", "cache_control": {...}}]`.
fn deserialize_system_prompt<'de, D>(deserializer: D) -> Result<Option<SystemContent>, D::Error>
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
        #[serde(default)]
        cache_control: Option<CacheControl>,
    }

    let maybe: Option<SystemPrompt> = Option::deserialize(deserializer)?;
    Ok(maybe.map(|sp| match sp {
        SystemPrompt::Text(s) => SystemContent {
            text: s,
            cache_control: None,
        },
        SystemPrompt::Blocks(blocks) => {
            let cache_control = blocks.iter().rev().find_map(|b| b.cache_control.clone());
            let text = blocks
                .into_iter()
                .map(|b| b.text)
                .collect::<Vec<_>>()
                .join("\n");
            SystemContent {
                text,
                cache_control,
            }
        }
    }))
}
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
    pub system: Option<SystemContent>,

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

    /// Top-level cache control for automatic prompt prefix caching.
    /// When present, the system caches all content up to the last cacheable block.
    /// Matches the Anthropic Messages API automatic caching mode.
    /// See: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching#automatic-caching
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,

    /// Extended thinking configuration. When enabled, the model produces
    /// `thinking` content blocks containing its internal reasoning before
    /// the final response. The `budget_tokens` field controls how many tokens
    /// the model may use for thinking (must be >= 1024 and < max_tokens).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingConfig>,

    /// Service tier selection: `"auto"` or `"standard_only"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,

    /// Container identifier for stateful sandbox sessions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub container: Option<String>,

    /// Output configuration: effort level and optional JSON schema format.
    /// `effort` can be `"low"`, `"medium"`, `"high"`, or `"max"`.
    /// `format` specifies structured JSON output constraints.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_config: Option<serde_json::Value>,
}

/// Extended thinking configuration for the request.
///
/// When `type` is `"enabled"`, the model will produce `thinking` content blocks
/// with its internal reasoning. `budget_tokens` controls the maximum tokens
/// available for thinking (minimum 1024, must be less than `max_tokens`).
/// When `type` is `"disabled"`, no thinking blocks are produced.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkingConfig {
    /// Either `"enabled"` or `"disabled"`.
    #[serde(rename = "type")]
    pub thinking_type: String,
    /// Maximum tokens for internal reasoning. Only relevant when type is "enabled".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub budget_tokens: Option<u32>,
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

/// Message content -- either a plain string or an array of content blocks.
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
/// `server_tool_use`, `redacted_thinking`) are captured as `Other(Value)` instead
/// of causing a hard deserialization failure. This is important because Claude
/// Code may send block types that we don't yet handle.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum AnthropicContentBlock {
    /// Text content block. May optionally include `citations` -- references to
    /// source documents that support the text content. Citations are generated
    /// by the model when document/PDF content is provided and citation mode is enabled.
    #[serde(rename = "text")]
    Text {
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        citations: Option<Vec<serde_json::Value>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    /// Image content block.
    #[serde(rename = "image")]
    Image { source: AnthropicImageSource },
    /// Tool use request from assistant.
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    /// Tool result from user.
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        content: Option<ToolResultContent>,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    /// Thinking content block from assistant (extended thinking / reasoning).
    #[serde(rename = "thinking")]
    Thinking {
        thinking: String,
        signature: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    /// Redacted thinking block from assistant. Contains encrypted reasoning data
    /// that is opaque to the client but must be passed back verbatim in multi-turn
    /// conversations so the model can maintain its chain of thought.
    #[serde(rename = "redacted_thinking")]
    RedactedThinking { data: String },
    /// Server-initiated tool use block. Represents a tool call that the API
    /// executes server-side (e.g., web search). The client receives the result
    /// via a corresponding `web_search_tool_result` or similar block.
    #[serde(rename = "server_tool_use")]
    ServerToolUse {
        id: String,
        name: String,
        #[serde(default)]
        input: serde_json::Value,
    },
    /// Result from a server-initiated tool (e.g., web search results).
    /// Contains structured content returned by the server-side tool execution.
    #[serde(rename = "web_search_tool_result")]
    WebSearchToolResult {
        tool_use_id: String,
        #[serde(default)]
        content: serde_json::Value,
    },
    /// Catch-all for unrecognized block types. Preserves the full JSON value
    /// so that new Anthropic features don't break the endpoint and can be
    /// round-tripped or inspected.
    #[serde(untagged)]
    Other(serde_json::Value),
}

/// Content of a `tool_result` block -- either a plain string or an array of
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
                    .ok_or_else(|| serde::de::Error::missing_field("text"))?
                    .to_string();
                let citations: Option<Vec<serde_json::Value>> = value
                    .get("citations")
                    .cloned()
                    .and_then(|v| serde_json::from_value(v).ok());
                let cache_control: Option<CacheControl> = value
                    .get("cache_control")
                    .cloned()
                    .and_then(|v| serde_json::from_value(v).ok());
                Ok(AnthropicContentBlock::Text {
                    text,
                    citations,
                    cache_control,
                })
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
                    .ok_or_else(|| serde::de::Error::missing_field("id"))?
                    .to_string();
                let name = value
                    .get("name")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| serde::de::Error::missing_field("name"))?
                    .to_string();
                let input = value.get("input").cloned().unwrap_or(serde_json::json!({}));
                let cache_control: Option<CacheControl> = value
                    .get("cache_control")
                    .cloned()
                    .and_then(|v| serde_json::from_value(v).ok());
                Ok(AnthropicContentBlock::ToolUse {
                    id,
                    name,
                    input,
                    cache_control,
                })
            }
            "tool_result" => {
                let tool_use_id = value
                    .get("tool_use_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| serde::de::Error::missing_field("tool_use_id"))?
                    .to_string();
                let content: Option<ToolResultContent> = value
                    .get("content")
                    .cloned()
                    .and_then(|v| serde_json::from_value(v).ok());
                let is_error = value.get("is_error").and_then(|v| v.as_bool());
                let cache_control: Option<CacheControl> = value
                    .get("cache_control")
                    .cloned()
                    .and_then(|v| serde_json::from_value(v).ok());
                Ok(AnthropicContentBlock::ToolResult {
                    tool_use_id,
                    content,
                    is_error,
                    cache_control,
                })
            }
            "thinking" => {
                let thinking = value
                    .get("thinking")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| serde::de::Error::missing_field("thinking"))?
                    .to_string();
                let signature = value
                    .get("signature")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| serde::de::Error::missing_field("signature"))?
                    .to_string();
                let cache_control: Option<CacheControl> = value
                    .get("cache_control")
                    .cloned()
                    .and_then(|v| serde_json::from_value(v).ok());
                Ok(AnthropicContentBlock::Thinking {
                    thinking,
                    signature,
                    cache_control,
                })
            }
            "redacted_thinking" => {
                let data = value
                    .get("data")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| serde::de::Error::missing_field("data"))?
                    .to_string();
                Ok(AnthropicContentBlock::RedactedThinking { data })
            }
            "server_tool_use" => {
                let id = value
                    .get("id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| serde::de::Error::missing_field("id"))?
                    .to_string();
                let name = value
                    .get("name")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| serde::de::Error::missing_field("name"))?
                    .to_string();
                let input = value.get("input").cloned().unwrap_or(serde_json::json!({}));
                Ok(AnthropicContentBlock::ServerToolUse { id, name, input })
            }
            "web_search_tool_result" => {
                let tool_use_id = value
                    .get("tool_use_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| serde::de::Error::missing_field("tool_use_id"))?
                    .to_string();
                let content = value
                    .get("content")
                    .cloned()
                    .unwrap_or(serde_json::json!([]));
                Ok(AnthropicContentBlock::WebSearchToolResult {
                    tool_use_id,
                    content,
                })
            }
            other => {
                tracing::debug!(
                    "Unrecognized Anthropic content block type '{}', preserving as Other",
                    other
                );
                Ok(AnthropicContentBlock::Other(value))
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
///
/// Client tools (custom) require `name` + `input_schema`. Server tools
/// (web_search, bash, text_editor, code_execution, etc.) are discriminated
/// by their `type` field (e.g. `"web_search_20260209"`) and may not have
/// `input_schema`. We keep all fields optional beyond `name` so both
/// kinds deserialize successfully and pass through to the backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicTool {
    /// Tool name (required for client tools, present on server tools too).
    pub name: String,
    /// Tool type discriminator. Client tools use `"custom"` (or omit).
    /// Server tools use versioned types like `"web_search_20260209"`.
    #[serde(default, rename = "type", skip_serializing_if = "Option::is_none")]
    pub tool_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// JSON Schema for the tool input. Required for client tools, absent on
    /// server tools (which define their own input shape server-side).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_schema: Option<serde_json::Value>,
    /// Cache control breakpoint on this tool definition.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<CacheControl>,
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
    /// When true, the model will call tools one at a time instead of
    /// potentially issuing multiple tool calls in a single response.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub disable_parallel_tool_use: Option<bool>,
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
    /// When true, the model will call tools one at a time instead of
    /// potentially issuing multiple tool calls in a single response.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub disable_parallel_tool_use: Option<bool>,
}
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
///
/// The Anthropic API returns up to 12 different block types. We model the
/// common ones explicitly and catch the rest as `Other` so the proxy can
/// forward them without losing data.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AnthropicResponseContentBlock {
    #[serde(rename = "thinking")]
    Thinking { thinking: String, signature: String },
    #[serde(rename = "text")]
    Text {
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        citations: Option<Vec<serde_json::Value>>,
    },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    #[serde(rename = "redacted_thinking")]
    RedactedThinking { data: String },
    #[serde(rename = "server_tool_use")]
    ServerToolUse {
        id: String,
        name: String,
        #[serde(default)]
        input: serde_json::Value,
    },
    #[serde(rename = "web_search_tool_result")]
    WebSearchToolResult {
        tool_use_id: String,
        #[serde(default)]
        content: serde_json::Value,
    },
    /// Catch-all for new/uncommon block types (web_fetch_tool_result,
    /// code_execution_tool_result, container_upload, etc.) so the proxy
    /// can serialize them back without data loss.
    #[serde(untagged)]
    Other(serde_json::Value),
}

/// Token usage information.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AnthropicUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    /// Number of input tokens used to create a new cache entry.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<u32>,
    /// Number of input tokens read from the prompt cache (prefix cache hits).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<u32>,
}

/// Reason the model stopped generating.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AnthropicStopReason {
    EndTurn,
    MaxTokens,
    StopSequence,
    ToolUse,
    /// The model paused to yield control in an agentic loop, intending to
    /// continue in a subsequent turn. Used with extended thinking / tool use.
    PauseTurn,
    /// The model refused to generate content (safety refusal).
    Refusal,
}
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
    #[serde(rename = "thinking_delta")]
    ThinkingDelta { thinking: String },
    #[serde(rename = "text_delta")]
    TextDelta { text: String },
    #[serde(rename = "input_json_delta")]
    InputJsonDelta { partial_json: String },
    /// Incremental signature for a thinking block (sent at the end).
    #[serde(rename = "signature_delta")]
    SignatureDelta { signature: String },
    /// Incremental citation attached to a text block.
    #[serde(rename = "citations_delta")]
    CitationsDelta { citation: serde_json::Value },
}

/// The delta body in a message_delta event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicMessageDeltaBody {
    pub stop_reason: Option<AnthropicStopReason>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequence: Option<String>,
}
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
    pub system: Option<SystemContent>,
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
            total_len += system.text.len();
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
                if let Some(schema) = &tool.input_schema {
                    total_len += schema.to_string().len();
                }
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
        AnthropicContentBlock::Text { text, .. } => text.len(),
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
        AnthropicContentBlock::RedactedThinking { data, .. } => data.len(),
        AnthropicContentBlock::ServerToolUse { name, input, .. } => {
            name.len() + input.to_string().len()
        }
        AnthropicContentBlock::WebSearchToolResult { content, .. } => content.to_string().len(),
        AnthropicContentBlock::Image { .. } => 256, // rough estimate for image metadata
        AnthropicContentBlock::Other(v) => v.to_string().len(),
    }
}
