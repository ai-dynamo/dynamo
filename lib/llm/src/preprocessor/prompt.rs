// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
use std::sync::Arc;

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
}

pub trait OAIPromptFormatter: Send + Sync + 'static {
    fn supports_add_generation_prompt(&self) -> bool;
    fn render(&self, req: &dyn OAIChatLikeRequest) -> Result<String>;
}

pub enum PromptFormatter {
    OAI(Arc<dyn OAIPromptFormatter>),
}
