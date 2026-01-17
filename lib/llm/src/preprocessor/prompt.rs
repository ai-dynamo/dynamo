// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Prompt Formatting Module
//!
//! Handles formatting of LLM request prompts, including:
//! - Chat template rendering
//! - Tool usage formatting
//! - Generation prompt handling
//!
//! The module supports different prompt formatting strategies through the
//! PromptFormatter

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

use crate::preprocessor::media::MediaDecoder;

pub mod deepseek_v32;
mod template;

pub use template::ContextMixins;

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
pub trait OAIChatLikeRequest {
    fn model(&self) -> String;
    fn messages(&self) -> Value;
    fn tools(&self) -> Option<Value> {
        None
    }
    fn tool_choice(&self) -> Option<Value> {
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

    fn media_io_kwargs(&self) -> Option<&MediaDecoder> {
        None
    }
}

pub trait OAIPromptFormatter: Send + Sync + 'static {
    fn supports_add_generation_prompt(&self) -> bool;
    fn render(&self, req: &dyn OAIChatLikeRequest) -> Result<String>;
}

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
