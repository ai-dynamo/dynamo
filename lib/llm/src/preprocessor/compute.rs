// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CPU-intensive preprocessing operations batched for efficient offload.
//!
//! This module provides "super-functions" that combine multiple sync operations
//! into a single offloadable unit, amortizing scheduling overhead when any
//! operation is slow.
//!
//! # Motivation
//!
//! When offloading work to a compute pool (rayon), there's scheduling overhead
//! of ~10-50µs per offload. If we have two sequential operations that each take
//! ~100µs, offloading them separately costs ~20-100µs in scheduling overhead.
//! By combining them into a single super-function, we pay the overhead once.
//!
//! # Architecture
//!
//! The key types are:
//! - [`PreprocessInput`]: Extracted request data that can be sent to compute pool
//! - [`PreprocessOutput`]: Results from the offloaded operation
//! - [`preprocess_sync`]: The super-function combining template + tokenization
//!
//! # Example
//!
//! ```ignore
//! use dynamo_runtime::compute::spawn_compute;
//! use dynamo_llm::preprocessor::compute::{PreprocessInput, preprocess_sync};
//!
//! // Extract data from request (cheap, on async thread)
//! let input = PreprocessInput::from_request(request, use_raw_prompt);
//! let formatter = self.formatter.clone();
//! let tokenizer = self.tokenizer.clone();
//!
//! // Single offload for template rendering + tokenization
//! let output = spawn_compute(move || {
//!     preprocess_sync(&*formatter, &*tokenizer, &input)
//! }).await?;
//!
//! // Apply results (cheap, on async thread)
//! builder.token_ids(output.token_ids);
//! ```

use crate::preprocessor::media::MediaDecoder;
use crate::preprocessor::prompt::{
    OAIChatLikeRequest, OAIPromptFormatter, PromptInput, TextInput, TokenInput,
};
use crate::protocols::openai::nvext::NvExtProvider;
use crate::tokenizers::traits::Encoder;
use anyhow::Result;
use minijinja::value::Value;
use std::collections::HashMap;

/// Request data extracted for compute offload.
///
/// This struct contains all data needed for template rendering and tokenization,
/// extracted from the original request so it can be sent to the compute pool
/// without borrowing the request.
///
/// Implements [`OAIChatLikeRequest`] so it can be used directly with formatters.
#[derive(Debug, Clone)]
pub struct PreprocessInput {
    /// Model name
    pub model: String,
    /// Messages for template rendering (as minijinja Value)
    pub messages: Value,
    /// Tools definition for template rendering
    pub tools: Option<Value>,
    /// Tool choice for template rendering
    pub tool_choice: Option<Value>,
    /// Whether to add generation prompt in template
    pub should_add_generation_prompt: bool,
    /// Extra args for chat template context
    pub chat_template_args: Option<HashMap<String, serde_json::Value>>,
    /// Raw prompt string (for completions or use_raw_prompt mode)
    pub raw_prompt: Option<String>,
    /// Pre-provided token IDs (skip tokenization if present with backend_instance_id)
    pub pre_provided_tokens: Option<Vec<u32>>,
    /// Whether backend_instance_id is present (enables token bypass)
    pub has_backend_instance_id: bool,
    /// Whether to use raw prompt instead of template rendering
    pub use_raw_prompt: bool,
}

impl PreprocessInput {
    /// Extract data from any OAIChatLikeRequest + NvExtProvider for compute offload.
    ///
    /// This is a cheap operation (clones Arc'd data and copies small fields)
    /// that should be called on the async thread before offloading.
    pub fn from_request<R>(request: &R) -> Self
    where
        R: OAIChatLikeRequest + NvExtProvider,
    {
        let nvext = request.nvext();
        let use_raw_prompt = nvext.is_some_and(|ext| ext.use_raw_prompt.unwrap_or(false));

        // Extract raw_prompt from text input if available
        let raw_prompt = match request.extract_text() {
            Some(TextInput::Single(s)) if !s.is_empty() => Some(s),
            _ => request.raw_prompt(),
        };

        Self {
            model: request.model(),
            messages: request.messages(),
            tools: request.tools(),
            tool_choice: request.tool_choice(),
            should_add_generation_prompt: request.should_add_generation_prompt(),
            chat_template_args: request.chat_template_args().cloned(),
            raw_prompt,
            pre_provided_tokens: nvext.and_then(|e| e.token_data.clone()),
            has_backend_instance_id: nvext.and_then(|e| e.backend_instance_id).is_some(),
            use_raw_prompt,
        }
    }
}

impl OAIChatLikeRequest for PreprocessInput {
    fn model(&self) -> String {
        self.model.clone()
    }

    fn messages(&self) -> Value {
        self.messages.clone()
    }

    fn tools(&self) -> Option<Value> {
        self.tools.clone()
    }

    fn tool_choice(&self) -> Option<Value> {
        self.tool_choice.clone()
    }

    fn should_add_generation_prompt(&self) -> bool {
        self.should_add_generation_prompt
    }

    fn chat_template_args(&self) -> Option<&HashMap<String, serde_json::Value>> {
        self.chat_template_args.as_ref()
    }

    fn prompt_input_type(&self) -> PromptInput {
        PromptInput::Text(TextInput::Single(String::new()))
    }

    fn extract_text(&self) -> Option<TextInput> {
        self.raw_prompt
            .as_ref()
            .map(|s| TextInput::Single(s.clone()))
    }

    fn extract_tokens(&self) -> Option<TokenInput> {
        None
    }

    fn media_io_kwargs(&self) -> Option<&MediaDecoder> {
        None
    }
}

/// Output from the preprocessing super-function.
///
/// Contains both the formatted prompt (if applicable) and the token IDs
/// from the combined template rendering + tokenization operation.
#[derive(Debug, Clone)]
pub struct PreprocessOutput {
    /// The rendered prompt string, if template rendering was performed.
    /// Will be `None` if pre-provided tokens were used.
    pub formatted_prompt: Option<String>,
    /// The token IDs for the prompt.
    pub token_ids: Vec<u32>,
    /// Whether pre-provided tokens were used (skipped tokenization).
    pub used_pre_provided_tokens: bool,
}

/// Sync super-function: template rendering + tokenization.
///
/// This combines both CPU-intensive operations into a single offloadable unit,
/// amortizing scheduling overhead. Call this from `spawn_compute` for offloading.
///
/// # Arguments
///
/// * `formatter` - The prompt formatter for template rendering
/// * `tokenizer` - The tokenizer for encoding the prompt
/// * `input` - The extracted request data
///
/// # Returns
///
/// A `PreprocessOutput` containing the formatted prompt and token IDs.
///
/// # Example
///
/// ```ignore
/// let input = PreprocessInput::from_request(request);
/// let output = spawn_compute(move || {
///     preprocess_sync(&*formatter, &*tokenizer, &input)
/// }).await?;
/// ```
pub fn preprocess_sync(
    formatter: &dyn OAIPromptFormatter,
    tokenizer: &dyn Encoder,
    input: &PreprocessInput,
) -> Result<PreprocessOutput> {
    // Check for pre-provided tokens first (EPP flow)
    if input.has_backend_instance_id {
        if let Some(ref tokens) = input.pre_provided_tokens {
            tracing::trace!(
                "Using provided tokens from EPP: {} ids",
                tokens.len()
            );
            return Ok(PreprocessOutput {
                formatted_prompt: None,
                token_ids: tokens.clone(),
                used_pre_provided_tokens: true,
            });
        } else {
            tracing::warn!(
                "backend_instance_id provided but no token_data; tokenizing prompt"
            );
        }
    }

    // Step 1: Template rendering
    let formatted_prompt = if input.use_raw_prompt {
        match &input.raw_prompt {
            Some(prompt) => prompt.clone(),
            None => {
                tracing::warn!("Raw prompt requested but not available");
                formatter.render(input)?
            }
        }
    } else {
        formatter.render(input)?
    };

    // Step 2: Tokenization
    let encoding = tokenizer.encode(&formatted_prompt)?;
    let token_ids = encoding.token_ids().to_vec();

    Ok(PreprocessOutput {
        formatted_prompt: Some(formatted_prompt),
        token_ids,
        used_pre_provided_tokens: false,
    })
}

/// Result of combined template rendering and tokenization.
///
/// This struct holds both the formatted prompt string and the resulting
/// token IDs from a single offloaded operation.
#[derive(Debug, Clone)]
pub struct RenderTokenizeResult {
    /// The rendered prompt string, if applicable.
    /// Will be `None` if `use_raw_prompt` was true and the raw prompt was used directly.
    pub formatted_prompt: Option<String>,

    /// The token IDs resulting from tokenization.
    pub token_ids: Vec<u32>,
}

/// Combines template rendering + tokenization into a single offloadable unit.
///
/// This super-function amortizes scheduling overhead (~10-50µs) across both
/// operations. Template rendering typically takes ~50-500µs and tokenization
/// ~50-200µs, making this combination worthwhile.
///
/// # Arguments
///
/// * `formatter` - The prompt formatter to use for template rendering
/// * `tokenizer` - The tokenizer to encode the rendered prompt
/// * `request` - The chat-like request containing messages to render
/// * `use_raw_prompt` - If true, skip template rendering and use the raw prompt directly
///
/// # Returns
///
/// A `RenderTokenizeResult` containing both the formatted prompt and token IDs.
///
/// # Example
///
/// ```ignore
/// // For loom-runtime: use spawn_adaptive for MAB learning
/// #[cfg(feature = "loom-runtime")]
/// let result = dynamo_runtime::compute::spawn_adaptive(|| {
///     render_and_tokenize(&formatter, &tokenizer, &request, false)
/// }).await?;
///
/// // For standard runtime: use spawn_compute
/// #[cfg(not(feature = "loom-runtime"))]
/// let result = dynamo_runtime::compute::spawn_compute(|| {
///     render_and_tokenize(&formatter, &tokenizer, &request, false)
/// }).await?;
/// ```
pub fn render_and_tokenize<R: OAIChatLikeRequest>(
    formatter: &dyn OAIPromptFormatter,
    tokenizer: &dyn Encoder,
    request: &R,
    use_raw_prompt: bool,
) -> Result<RenderTokenizeResult> {
    // Step 1: Template rendering (Chain A or B from the analysis)
    let formatted = if use_raw_prompt {
        // Skip rendering, use raw prompt if available
        None
    } else {
        Some(formatter.render(request)?)
    };

    // Determine the actual prompt to tokenize
    let prompt_to_tokenize = formatted.as_ref().map(|s| s.as_str());

    // If we don't have a formatted prompt, we can't tokenize
    let prompt = prompt_to_tokenize
        .ok_or_else(|| anyhow::anyhow!("No prompt available for tokenization"))?;

    // Step 2: Tokenization
    let encoding = tokenizer.encode(prompt)?;
    let token_ids = encoding.token_ids().to_vec();

    Ok(RenderTokenizeResult {
        formatted_prompt: formatted,
        token_ids,
    })
}

/// Batch render and tokenize multiple inputs.
///
/// This is useful for embedding requests where multiple strings need to be
/// tokenized. By batching the operation, we amortize scheduling overhead
/// across all inputs.
///
/// # Arguments
///
/// * `tokenizer` - The tokenizer to encode the inputs
/// * `inputs` - The strings to tokenize
///
/// # Returns
///
/// A vector of token ID vectors, one for each input.
pub fn batch_tokenize(tokenizer: &dyn Encoder, inputs: &[&str]) -> Result<Vec<Vec<u32>>> {
    let encodings = tokenizer.encode_batch(inputs)?;
    Ok(encodings
        .into_iter()
        .map(|e| e.token_ids().to_vec())
        .collect())
}

/// Tokenize a single prompt string.
///
/// This is the CPU-intensive operation that should be offloaded via
/// `spawn_compute()` or `spawn_adaptive()`.
///
/// # Example
///
/// ```ignore
/// use dynamo_runtime::compute::spawn_compute;
/// use dynamo_llm::preprocessor::compute::tokenize_prompt;
///
/// // Offload tokenization to compute pool
/// let tokenizer = self.tokenizer.clone();
/// let prompt = prompt.to_owned();
/// let token_ids = spawn_compute(move || {
///     tokenize_prompt(&*tokenizer, &prompt)
/// }).await?;
/// ```
#[inline]
pub fn tokenize_prompt(tokenizer: &dyn Encoder, prompt: &str) -> Result<Vec<u32>> {
    let encoding = tokenizer.encode(prompt)?;
    Ok(encoding.token_ids().to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: Full integration tests require a tokenizer and formatter,
    // which are tested in the preprocessor integration tests.

    #[test]
    fn test_render_tokenize_result_debug() {
        let result = RenderTokenizeResult {
            formatted_prompt: Some("Hello, world!".to_string()),
            token_ids: vec![1, 2, 3],
        };
        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("Hello, world!"));
        assert!(debug_str.contains("[1, 2, 3]"));
    }

    #[test]
    fn test_render_tokenize_result_clone() {
        let result = RenderTokenizeResult {
            formatted_prompt: Some("test".to_string()),
            token_ids: vec![42],
        };
        let cloned = result.clone();
        assert_eq!(result.formatted_prompt, cloned.formatted_prompt);
        assert_eq!(result.token_ids, cloned.token_ids);
    }

    #[test]
    fn test_preprocess_input_debug() {
        let input = PreprocessInput {
            model: "test-model".to_string(),
            messages: Value::from_serialize(&vec!["hello"]),
            tools: None,
            tool_choice: None,
            should_add_generation_prompt: true,
            chat_template_args: None,
            raw_prompt: Some("Hello, world!".to_string()),
            pre_provided_tokens: None,
            has_backend_instance_id: false,
            use_raw_prompt: false,
        };
        let debug_str = format!("{:?}", input);
        assert!(debug_str.contains("test-model"));
        assert!(debug_str.contains("Hello, world!"));
    }

    #[test]
    fn test_preprocess_input_clone() {
        let input = PreprocessInput {
            model: "test-model".to_string(),
            messages: Value::from_serialize(&vec!["hello"]),
            tools: None,
            tool_choice: None,
            should_add_generation_prompt: true,
            chat_template_args: None,
            raw_prompt: Some("test".to_string()),
            pre_provided_tokens: Some(vec![1, 2, 3]),
            has_backend_instance_id: true,
            use_raw_prompt: false,
        };
        let cloned = input.clone();
        assert_eq!(input.model, cloned.model);
        assert_eq!(input.raw_prompt, cloned.raw_prompt);
        assert_eq!(input.pre_provided_tokens, cloned.pre_provided_tokens);
        assert_eq!(input.has_backend_instance_id, cloned.has_backend_instance_id);
    }

    #[test]
    fn test_preprocess_input_oai_trait() {
        let input = PreprocessInput {
            model: "test-model".to_string(),
            messages: Value::from_serialize(&vec!["hello"]),
            tools: Some(Value::from_serialize(&vec!["tool1"])),
            tool_choice: Some(Value::from_serialize(&"auto")),
            should_add_generation_prompt: true,
            chat_template_args: Some({
                let mut map = HashMap::new();
                map.insert("key".to_string(), serde_json::json!("value"));
                map
            }),
            raw_prompt: Some("raw prompt".to_string()),
            pre_provided_tokens: None,
            has_backend_instance_id: false,
            use_raw_prompt: false,
        };

        // Test OAIChatLikeRequest trait implementation
        assert_eq!(input.model(), "test-model");
        assert!(input.tools().is_some());
        assert!(input.tool_choice().is_some());
        assert!(input.should_add_generation_prompt());
        assert!(input.chat_template_args().is_some());
        assert!(matches!(input.prompt_input_type(), PromptInput::Text(_)));
        assert!(matches!(input.extract_text(), Some(TextInput::Single(_))));
        assert!(input.extract_tokens().is_none());
        assert!(input.media_io_kwargs().is_none());
    }

    #[test]
    fn test_preprocess_output_debug() {
        let output = PreprocessOutput {
            formatted_prompt: Some("Hello, world!".to_string()),
            token_ids: vec![1, 2, 3],
            used_pre_provided_tokens: false,
        };
        let debug_str = format!("{:?}", output);
        assert!(debug_str.contains("Hello, world!"));
        assert!(debug_str.contains("[1, 2, 3]"));
        assert!(debug_str.contains("false"));
    }

    #[test]
    fn test_preprocess_output_clone() {
        let output = PreprocessOutput {
            formatted_prompt: Some("test".to_string()),
            token_ids: vec![42],
            used_pre_provided_tokens: true,
        };
        let cloned = output.clone();
        assert_eq!(output.formatted_prompt, cloned.formatted_prompt);
        assert_eq!(output.token_ids, cloned.token_ids);
        assert_eq!(output.used_pre_provided_tokens, cloned.used_pre_provided_tokens);
    }
}
