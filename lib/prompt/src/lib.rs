// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Prompt Formatting
//!
//! Standalone, runtime-free chat-template / prompt formatting for
//! OpenAI-compatible inference frontends. Renders HuggingFace `chat_template`
//! jinja2 (via `minijinja` + `minijinja-contrib` pycompat), handles tool
//! usage formatting and generation-prompt handling.
//!
//! Consumers implement [`OAIChatLikeRequest`] for their request type (or use
//! the ready-made impl for `dynamo-protocols`' OpenAI chat request) and render
//! with a [`PromptFormatter`] built from a HuggingFace `tokenizer_config.json`
//! ([`ChatTemplate`]).
//!
//! This crate is a *bridge* between OpenAI request types ([`dynamo_protocols`])
//! and the HF chat-template engine ([`minijinja`]); it does not depend on
//! tokenizer internals. `dynamo-tokenizers` is re-exported for convenience.

// TODO:
// 1. Query if `add_generation_prompt` is present in the prompt template
// 2. Support for models with add_generation_prompt:
//    - PALS (Prefix-Assisted Language Sampling)
//    - Continuation - Detected on user turns, where we can return
//      partial assistant responses without add_generation_prompt

use anyhow::Result;
use minijinja::value::Value;
use std::collections::HashMap;
use std::sync::Arc;

/// Re-export of `dynamo-tokenizers` as a one-import convenience: consumers that
/// want both tokenization and chat templating can reach the tokenizer types via
/// `dynamo_prompt::dynamo_tokenizers::*` without adding a second dependency.
/// This crate does not otherwise use the tokenizer internals.
pub use dynamo_tokenizers;

pub mod deepseek_common;
pub mod deepseek_v32;
pub mod deepseek_v4;
mod template;

pub use template::{
    ChatTemplate, ChatTemplateValue, ContextMixins, deepseek_formatter_for, may_be_fix_tool_schema,
};

/// Selects which context-mixin behaviors a template renders with.
///
/// Carried on the model deployment card (`prompt_context`) and consumed by the
/// chat-template renderer via [`ContextMixins`].
#[derive(serde::Serialize, serde::Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum PromptContextMixin {
    /// Support OAI Chat Messages and Tools
    OaiChat,

    /// Enables templates with `{{datetime}}` to be rendered with the current date and time.
    Llama3DateTime,
}

/// Shared helper: extract a boolean thinking toggle from `chat_template_args`.
///
/// Reads the two equivalent keys (`thinking`, `enable_thinking` — vLLM's
/// canonical kwarg) in order and returns the first bool value found, or `None`
/// if neither key is present (or neither carries a bool). Used by the V4
/// formatter's `resolve_thinking_mode` and by reasoning-parser gating in
/// consumers so both paths agree on the signal interpretation.
pub fn thinking_bool_from_args(args: Option<&HashMap<String, serde_json::Value>>) -> Option<bool> {
    let args = args?;
    for key in ["thinking", "enable_thinking"] {
        if let Some(v) = args.get(key).and_then(|x| x.as_bool()) {
            return Some(v);
        }
    }
    None
}

#[derive(Debug)]
pub enum TokenInput {
    Single(Vec<u32>),
    Batch(Vec<Vec<u32>>),
}

#[derive(Debug)]
pub enum TextInput {
    Single(String),
    Batch(Vec<String>),
}

#[derive(Debug)]
pub enum PromptInput {
    Tokens(TokenInput),
    Text(TextInput),
}

/// Trait that defines a request that can map to an OpenAI-like request.
///
/// Implement this for your request type to render it through a
/// [`PromptFormatter`]. Media/multimodal IO config is intentionally *not* part
/// of this trait — it is a preprocessing concern owned by the consumer, kept
/// off the rendering surface so this crate stays runtime-free.
pub trait OAIChatLikeRequest {
    fn model(&self) -> String;
    fn messages(&self) -> Value;
    fn typed_messages(&self) -> Option<&[dynamo_protocols::types::ChatCompletionRequestMessage]> {
        None
    }
    fn tools(&self) -> Option<Value> {
        None
    }
    fn tool_choice(&self) -> Option<Value> {
        None
    }
    fn response_format(&self) -> Option<Value> {
        None
    }

    fn should_add_generation_prompt(&self) -> bool;

    /// Optional additional args to merge into the chat template context
    fn chat_template_args(&self) -> Option<&HashMap<String, serde_json::Value>> {
        None
    }

    /// Returns the type of input for the prompt. Default is Text.
    fn prompt_input_type(&self) -> PromptInput {
        PromptInput::Text(TextInput::Single(String::new()))
    }

    /// Extract tokens if the input is pre-tokenized
    fn extract_tokens(&self) -> Option<TokenInput> {
        None
    }

    fn extract_text(&self) -> Option<TextInput> {
        None
    }

    fn mm_processor_kwargs(&self) -> Option<&serde_json::Value> {
        None
    }
}

pub trait OAIPromptFormatter: Send + Sync + 'static {
    fn supports_add_generation_prompt(&self) -> bool;
    fn render(&self, req: &dyn OAIChatLikeRequest) -> Result<String>;

    /// Per-family image-placeholder template used when the chat template
    /// requires string content and the request contains images. `{n}` in
    /// the template is the 1-based image index. `None` when the
    /// formatter has no flatten strategy — MM-aware routing falls back
    /// to text-prefix routing for those families.
    fn image_placeholder_template(&self) -> Option<&'static str> {
        None
    }
}

#[derive(Clone)]
pub enum PromptFormatter {
    OAI(Arc<dyn OAIPromptFormatter>),
}

// No-op formatter: used for models without chat_template
#[derive(Debug, Default)]
pub struct NoOpFormatter;

impl OAIPromptFormatter for NoOpFormatter {
    fn supports_add_generation_prompt(&self) -> bool {
        false
    }

    fn render(&self, req: &dyn OAIChatLikeRequest) -> Result<String> {
        let messages = req.messages();

        let first_message = messages
            .get_item_by_index(0)
            .map_err(|_| anyhow::Error::msg("No message at index 0 or messages array is empty"))?;

        let content = first_message
            .get_attr("content")
            .map_err(|_| anyhow::Error::msg("First message has no 'content' field"))?;

        let content_str = content
            .as_str()
            .ok_or_else(|| anyhow::Error::msg("Message content is not a string"))?
            .to_string();
        Ok(content_str)
    }
}

impl PromptFormatter {
    pub fn no_op() -> Self {
        Self::OAI(Arc::new(NoOpFormatter))
    }
}
