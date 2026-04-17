// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unified internal request representation.
//!
//! `UnifiedRequest` is an API-agnostic wrapper that carries a fully-converted
//! `NvCreateChatCompletionRequest` alongside the API-specific context that
//! would otherwise be lost during the fan-in conversion.
//!
//! # Motivation
//!
//! Dynamo's HTTP frontend uses an hourglass architecture: multiple API surfaces
//! (Chat Completions, Anthropic Messages, Responses) fan in through `TryFrom`
//! to `NvCreateChatCompletionRequest`. Non-OpenAI features are lossy-compressed
//! or silently dropped during this conversion. `UnifiedRequest` preserves that
//! context so it can flow through the preprocessor and be used on the response
//! path for faithful reconstruction.
//!
//! # Architecture
//!
//! ```text
//! Anthropic Messages ──┐
//! OpenAI Responses ────┼──→ UnifiedRequest { inner: NvCreateChatCompletion, api_context, ... }
//! OpenAI Chat ─────────┘            │
//!                                   ↓
//!                          PreprocessedRequest ──→ Backend
//! ```
//!
//! The existing preprocessor pipeline is unchanged — `UnifiedRequest` implements
//! all the same traits (`OAIChatLikeRequest`, `SamplingOptionsProvider`, etc.)
//! by delegating to the inner `NvCreateChatCompletionRequest`. The additional
//! context fields are carried through for response-path use.

use std::collections::HashMap;

use dynamo_protocols::types::anthropic::CacheControl;
use dynamo_runtime::protocols::annotated::AnnotationsProvider;
use serde::{Deserialize, Serialize};

use crate::preprocessor::media::MediaDecoder;
use crate::preprocessor::prompt::{OAIChatLikeRequest, TextInput};

use crate::protocols::openai::chat_completions::NvCreateChatCompletionRequest;
use crate::protocols::openai::common_ext::{CommonExt, CommonExtProvider};
use crate::protocols::openai::nvext::{NvExt, NvExtProvider};
use crate::protocols::openai::{
    OpenAIOutputOptionsProvider, OpenAISamplingOptionsProvider, OpenAIStopConditionsProvider,
};

use dynamo_protocols::types::responses::{IncludeEnum, Reasoning, Truncation};

use super::anthropic::types::{AnthropicCreateMessageRequest, ThinkingConfig};
use super::openai::responses::NvCreateResponse;

/// Identifies which API surface originated the request and carries
/// fields specific to that API that cannot be represented in the
/// OpenAI Chat Completions format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ApiContext {
    /// Request came from the OpenAI Chat Completions API.
    /// All fields are natively represented in `NvCreateChatCompletionRequest`.
    ChatCompletions,

    /// Request came from the Anthropic Messages API.
    Anthropic(AnthropicContext),

    /// Request came from the OpenAI Responses API.
    Responses(ResponsesContext),
}

/// Anthropic-specific fields preserved from `AnthropicCreateMessageRequest`.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AnthropicContext {
    /// Extended thinking configuration (`type` + `budget_tokens`).
    /// Dropped during conversion because `NvCreateChatCompletionRequest` has
    /// no equivalent — only `reasoning_effort` (a string) exists.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingConfig>,

    /// Per-block cache control breakpoints with their position in the
    /// message array. These remain available in the API sidecar even when
    /// the request conversion does not forward cache control into `nvext`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub cache_breakpoints: Vec<CacheBreakpoint>,

    /// When true, the model should not issue parallel tool calls.
    /// The Anthropic API supports `disable_parallel_tool_use` on the tool_choice
    /// object but there is no OpenAI equivalent field.
    #[serde(default)]
    pub disable_parallel_tool_use: bool,

    /// Anthropic request metadata (e.g. `user_id`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,

    /// Service tier selection from the Anthropic request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,

    /// Container identifier for stateful sandbox sessions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub container: Option<String>,

    /// Output configuration (effort level, JSON schema format).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_config: Option<serde_json::Value>,
}

/// Responses API-specific fields preserved from `NvCreateResponse`.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ResponsesContext {
    /// Conversation continuation identifier.
    /// Dropped during conversion — no OpenAI Chat equivalent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,

    /// Context truncation strategy.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub truncation: Option<Truncation>,

    /// Reasoning configuration (effort + optional summary generation).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<Reasoning>,

    /// Output items to include in the response.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub include: Option<Vec<IncludeEnum>>,

    /// Whether responses should be stored server-side.
    #[serde(default)]
    pub store: bool,
}

/// A cache breakpoint records the position (message index, block index)
/// and the cache control directive from the original Anthropic request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheBreakpoint {
    /// Index of the message in the original messages array.
    pub message_index: usize,
    /// Index of the content block within the message (0 for plain-text messages).
    pub block_index: usize,
    /// The cache control directive.
    pub cache_control: CacheControl,
}

/// API-agnostic request wrapper that preserves the full context from any
/// API surface while remaining compatible with the existing preprocessor.
#[derive(Debug, Clone)]
pub struct UnifiedRequest {
    /// The core request in OpenAI Chat Completions format.
    /// This is what the preprocessor already knows how to handle.
    pub inner: NvCreateChatCompletionRequest,

    /// Which API surface originated this request, plus API-specific fields
    /// that were dropped during conversion to `NvCreateChatCompletionRequest`.
    pub api_context: ApiContext,
}

impl From<NvCreateChatCompletionRequest> for UnifiedRequest {
    fn from(req: NvCreateChatCompletionRequest) -> Self {
        Self {
            inner: req,
            api_context: ApiContext::ChatCompletions,
        }
    }
}

impl TryFrom<AnthropicCreateMessageRequest> for UnifiedRequest {
    type Error = anyhow::Error;

    fn try_from(req: AnthropicCreateMessageRequest) -> Result<Self, Self::Error> {
        // Capture API-specific fields BEFORE the lossy conversion
        let anthropic_ctx = AnthropicContext {
            thinking: req.thinking.clone(),
            cache_breakpoints: extract_cache_breakpoints(&req),
            disable_parallel_tool_use: extract_disable_parallel_tool_use(&req),
            metadata: req.metadata.clone(),
            service_tier: req.service_tier.clone(),
            container: req.container.clone(),
            output_config: req.output_config.clone(),
        };

        // Perform the existing lossy conversion
        let inner: NvCreateChatCompletionRequest = req.try_into()?;

        Ok(Self {
            inner,
            api_context: ApiContext::Anthropic(anthropic_ctx),
        })
    }
}

impl TryFrom<NvCreateResponse> for UnifiedRequest {
    type Error = anyhow::Error;

    fn try_from(req: NvCreateResponse) -> Result<Self, Self::Error> {
        // Capture API-specific fields BEFORE the lossy conversion
        let responses_ctx = ResponsesContext {
            previous_response_id: req.inner.previous_response_id.clone(),
            truncation: req.inner.truncation,
            reasoning: req.inner.reasoning.clone(),
            include: req.inner.include.clone(),
            store: req.inner.store.unwrap_or(false),
        };

        // Perform the existing lossy conversion
        let inner: NvCreateChatCompletionRequest = req.try_into()?;

        Ok(Self {
            inner,
            api_context: ApiContext::Responses(responses_ctx),
        })
    }
}

/// Walk the Anthropic message array and collect per-block cache_control
/// annotations with their (message_index, block_index) positions.
fn extract_cache_breakpoints(req: &AnthropicCreateMessageRequest) -> Vec<CacheBreakpoint> {
    use super::anthropic::types::{AnthropicContentBlock, AnthropicMessageContent};

    let mut breakpoints = Vec::new();

    // System-level cache control
    if let Some(system) = &req.system
        && let Some(cc) = &system.cache_control
    {
        breakpoints.push(CacheBreakpoint {
            message_index: 0, // system is logically position 0
            block_index: 0,
            cache_control: cc.clone(),
        });
    }

    let offset = if req.system.is_some() { 1 } else { 0 };

    for (msg_idx, msg) in req.messages.iter().enumerate() {
        if let AnthropicMessageContent::Blocks { content } = &msg.content {
            for (block_idx, block) in content.iter().enumerate() {
                let cc = match block {
                    AnthropicContentBlock::Text { cache_control, .. } => cache_control.as_ref(),
                    AnthropicContentBlock::ToolUse { cache_control, .. } => cache_control.as_ref(),
                    AnthropicContentBlock::ToolResult { cache_control, .. } => {
                        cache_control.as_ref()
                    }
                    AnthropicContentBlock::Thinking { cache_control, .. } => cache_control.as_ref(),
                    _ => None,
                };
                if let Some(cc) = cc {
                    breakpoints.push(CacheBreakpoint {
                        message_index: msg_idx + offset,
                        block_index: block_idx,
                        cache_control: cc.clone(),
                    });
                }
            }
        }
    }

    breakpoints
}

/// Extract `disable_parallel_tool_use` from the Anthropic tool_choice.
/// The Anthropic API allows `{"type": "auto", "disable_parallel_tool_use": true}`
/// but there's no OpenAI Chat equivalent.
fn extract_disable_parallel_tool_use(req: &AnthropicCreateMessageRequest) -> bool {
    use super::anthropic::types::AnthropicToolChoice;

    match &req.tool_choice {
        Some(AnthropicToolChoice::Simple(simple)) => {
            simple.disable_parallel_tool_use.unwrap_or(false)
        }
        Some(AnthropicToolChoice::Named(named)) => named.disable_parallel_tool_use.unwrap_or(false),
        None => false,
    }
}

// Trait implementations — delegate to inner NvCreateChatCompletionRequest

impl NvExtProvider for UnifiedRequest {
    fn nvext(&self) -> Option<&NvExt> {
        self.inner.nvext.as_ref()
    }

    fn raw_prompt(&self) -> Option<String> {
        None
    }
}

impl AnnotationsProvider for UnifiedRequest {
    fn annotations(&self) -> Option<Vec<String>> {
        self.inner
            .nvext
            .as_ref()
            .and_then(|nvext| nvext.annotations.clone())
    }

    fn has_annotation(&self, annotation: &str) -> bool {
        self.inner
            .nvext
            .as_ref()
            .and_then(|nvext| nvext.annotations.as_ref())
            .map(|annotations| annotations.contains(&annotation.to_string()))
            .unwrap_or(false)
    }
}

impl OpenAISamplingOptionsProvider for UnifiedRequest {
    fn get_temperature(&self) -> Option<f32> {
        self.inner.inner.temperature
    }

    fn get_top_p(&self) -> Option<f32> {
        self.inner.inner.top_p
    }

    fn get_frequency_penalty(&self) -> Option<f32> {
        self.inner.inner.frequency_penalty
    }

    fn get_presence_penalty(&self) -> Option<f32> {
        self.inner.inner.presence_penalty
    }

    fn nvext(&self) -> Option<&NvExt> {
        self.inner.nvext.as_ref()
    }

    fn get_seed(&self) -> Option<i64> {
        self.inner.inner.seed
    }

    fn get_n(&self) -> Option<u8> {
        self.inner.inner.n
    }

    fn get_best_of(&self) -> Option<u8> {
        OpenAISamplingOptionsProvider::get_best_of(&self.inner)
    }
}

impl CommonExtProvider for UnifiedRequest {
    fn common_ext(&self) -> Option<&CommonExt> {
        Some(&self.inner.common)
    }

    fn get_guided_json(&self) -> Option<serde_json::Value> {
        // Delegate to the inner impl which handles tool_choice → guided_json
        // and response_format → guided_json derivation.
        CommonExtProvider::get_guided_json(&self.inner)
    }

    fn get_guided_regex(&self) -> Option<String> {
        self.inner.common.guided_regex.clone()
    }

    fn get_guided_grammar(&self) -> Option<String> {
        self.inner.common.guided_grammar.clone()
    }

    fn get_guided_choice(&self) -> Option<Vec<String>> {
        self.inner.common.guided_choice.clone()
    }

    fn get_guided_decoding_backend(&self) -> Option<String> {
        self.inner.common.guided_decoding_backend.clone()
    }

    fn get_guided_whitespace_pattern(&self) -> Option<String> {
        self.inner.common.guided_whitespace_pattern.clone()
    }

    fn get_top_k(&self) -> Option<i32> {
        self.inner.common.top_k
    }

    fn get_min_p(&self) -> Option<f32> {
        self.inner.common.min_p
    }

    fn get_repetition_penalty(&self) -> Option<f32> {
        self.inner.common.repetition_penalty
    }

    fn get_include_stop_str_in_output(&self) -> Option<bool> {
        self.inner.common.include_stop_str_in_output
    }

    fn get_skip_special_tokens(&self) -> Option<bool> {
        self.inner.common.skip_special_tokens
    }
}

impl OpenAIStopConditionsProvider for UnifiedRequest {
    #[allow(deprecated)]
    fn get_max_tokens(&self) -> Option<u32> {
        self.inner
            .inner
            .max_completion_tokens
            .or(self.inner.inner.max_tokens)
    }

    fn get_min_tokens(&self) -> Option<u32> {
        self.inner.common.min_tokens
    }

    fn get_stop(&self) -> Option<Vec<String>> {
        self.inner.inner.stop.as_ref().map(|stop| match stop {
            dynamo_protocols::types::Stop::String(s) => vec![s.clone()],
            dynamo_protocols::types::Stop::StringArray(arr) => arr.clone(),
        })
    }

    fn nvext(&self) -> Option<&NvExt> {
        self.inner.nvext.as_ref()
    }

    fn get_common_ignore_eos(&self) -> Option<bool> {
        self.inner.common.ignore_eos
    }
}

impl OpenAIOutputOptionsProvider for UnifiedRequest {
    fn get_logprobs(&self) -> Option<u32> {
        match self.inner.inner.logprobs {
            Some(true) => match self.inner.inner.top_logprobs {
                Some(top_logprobs) => Some(top_logprobs as u32),
                None => Some(1_u32),
            },
            Some(false) => None,
            None => None,
        }
    }

    fn get_prompt_logprobs(&self) -> Option<u32> {
        OpenAIOutputOptionsProvider::get_prompt_logprobs(&self.inner)
    }

    fn get_skip_special_tokens(&self) -> Option<bool> {
        OpenAIOutputOptionsProvider::get_skip_special_tokens(&self.inner)
    }

    fn get_formatted_prompt(&self) -> Option<bool> {
        OpenAIOutputOptionsProvider::get_formatted_prompt(&self.inner)
    }
}

impl OAIChatLikeRequest for UnifiedRequest {
    fn model(&self) -> String {
        self.inner.inner.model.clone()
    }

    fn messages(&self) -> minijinja::value::Value {
        let messages_json = serde_json::to_value(&self.inner.inner.messages).unwrap();
        minijinja::value::Value::from_serialize(&messages_json)
    }

    fn typed_messages(&self) -> Option<&[dynamo_protocols::types::ChatCompletionRequestMessage]> {
        Some(self.inner.inner.messages.as_slice())
    }

    fn tools(&self) -> Option<minijinja::value::Value> {
        OAIChatLikeRequest::tools(&self.inner)
    }

    fn tool_choice(&self) -> Option<minijinja::value::Value> {
        OAIChatLikeRequest::tool_choice(&self.inner)
    }

    fn response_format(&self) -> Option<minijinja::value::Value> {
        OAIChatLikeRequest::response_format(&self.inner)
    }

    fn should_add_generation_prompt(&self) -> bool {
        OAIChatLikeRequest::should_add_generation_prompt(&self.inner)
    }

    fn extract_text(&self) -> Option<TextInput> {
        OAIChatLikeRequest::extract_text(&self.inner)
    }

    fn chat_template_args(&self) -> Option<&HashMap<String, serde_json::Value>> {
        self.inner.chat_template_args.as_ref()
    }

    fn media_io_kwargs(&self) -> Option<&MediaDecoder> {
        self.inner.media_io_kwargs.as_ref()
    }

    fn mm_processor_kwargs(&self) -> Option<&serde_json::Value> {
        self.inner.inner.mm_processor_kwargs.as_ref()
    }
}

impl UnifiedRequest {
    /// Returns the Anthropic context if this request originated from the
    /// Anthropic Messages API.
    pub fn anthropic_context(&self) -> Option<&AnthropicContext> {
        match &self.api_context {
            ApiContext::Anthropic(ctx) => Some(ctx),
            _ => None,
        }
    }

    /// Returns the Responses context if this request originated from the
    /// OpenAI Responses API.
    pub fn responses_context(&self) -> Option<&ResponsesContext> {
        match &self.api_context {
            ApiContext::Responses(ctx) => Some(ctx),
            _ => None,
        }
    }

    /// Unwrap back to the inner `NvCreateChatCompletionRequest`.
    /// Useful for gradual migration — callers that don't need the extra
    /// context can unwrap and use the existing code paths unchanged.
    pub fn into_inner(self) -> NvCreateChatCompletionRequest {
        self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chat_completions_roundtrip() {
        let req = NvCreateChatCompletionRequest {
            inner: dynamo_protocols::types::CreateChatCompletionRequest {
                model: "test-model".to_string(),
                messages: vec![],
                ..Default::default()
            },
            common: CommonExt::default(),
            nvext: None,
            chat_template_args: None,
            media_io_kwargs: None,
            return_tokens_as_token_ids: None,
            unsupported_fields: Default::default(),
        };

        let unified = UnifiedRequest::from(req.clone());
        assert!(matches!(unified.api_context, ApiContext::ChatCompletions));
        assert_eq!(unified.model(), "test-model");
    }

    #[test]
    fn test_anthropic_context_preserved() {
        use super::super::anthropic::types::*;

        let req = AnthropicCreateMessageRequest {
            model: "claude-sonnet-4-20250514".to_string(),
            max_tokens: 1024,
            messages: vec![AnthropicMessage {
                role: AnthropicRole::User,
                content: AnthropicMessageContent::Text {
                    content: "Hello".to_string(),
                },
            }],
            system: None,
            temperature: Some(0.7),
            top_p: None,
            top_k: None,
            stop_sequences: None,
            stream: true,
            metadata: Some(serde_json::json!({"user_id": "test"})),
            tools: None,
            tool_choice: None,
            cache_control: None,
            thinking: Some(ThinkingConfig {
                thinking_type: "enabled".to_string(),
                budget_tokens: Some(4096),
            }),
            service_tier: None,
            container: None,
            output_config: None,
        };

        let unified = UnifiedRequest::try_from(req).unwrap();

        // Verify the context was preserved
        let ctx = unified.anthropic_context().unwrap();
        assert!(ctx.thinking.is_some());
        assert_eq!(ctx.thinking.as_ref().unwrap().thinking_type, "enabled");
        assert_eq!(ctx.thinking.as_ref().unwrap().budget_tokens, Some(4096));
        assert!(ctx.metadata.is_some());

        // Verify it still works as a preprocessor input
        assert_eq!(unified.model(), "claude-sonnet-4-20250514");
        assert!(unified.extract_text().is_some());
    }

    #[test]
    fn test_responses_context_preserved() {
        // Construct an NvCreateResponse via JSON to satisfy all required fields
        let json = serde_json::json!({
            "model": "gpt-4o",
            "input": "What is the capital of France?",
            "previous_response_id": "resp_abc123",
            "store": true,
            "truncation": "auto",
            "reasoning": {
                "effort": "medium"
            },
            "include": ["message.output_text.logprobs"]
        });
        let req: NvCreateResponse = serde_json::from_value(json).unwrap();

        let unified = UnifiedRequest::try_from(req).unwrap();

        let ctx = unified.responses_context().unwrap();
        assert_eq!(ctx.previous_response_id.as_deref(), Some("resp_abc123"));
        assert!(ctx.store);
        assert!(ctx.truncation.is_some());
        assert!(ctx.reasoning.is_some());
        assert!(ctx.include.is_some());
        assert_eq!(ctx.include.as_ref().unwrap().len(), 1);

        // Verify it still works as a preprocessor input
        assert_eq!(unified.model(), "gpt-4o");
    }
}
