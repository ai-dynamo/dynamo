// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SGLang tokenizer wrapper using PyO3.
//!
//! This module provides a tokenizer implementation that calls SGLang's Python tokenizer
//! via PyO3. It supports HuggingFace tokenizers, mistral-common, and other tokenizers
//! that SGLang handles.
//!
//! # Usage
//!
//! ```ignore
//! use dynamo_llm::tokenizers::sglang::SglangTokenizer;
//!
//! // Create tokenizer for a model
//! let tokenizer = SglangTokenizer::new("meta-llama/Llama-3.2-1B", None)?;
//!
//! // For mistral-common tokenizers
//! let tokenizer = SglangTokenizer::new("mistralai/Mistral-7B-v0.1", Some("mistral"))?;
//!
//! // Encode text to token IDs
//! let encoding = tokenizer.encode("Hello, world!")?;
//! let token_ids = encoding.token_ids();
//! ```

use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::{
    Encoding, Error, Result, TokenIdType,
    traits::{Decoder, Encoder, Tokenizer},
};

/// SGLang tokenizer wrapper that calls Python tokenizer via PyO3.
///
/// This tokenizer supports:
/// - HuggingFace transformers tokenizers (default)
/// - mistral-common tokenizers (via `tokenizer_mode="mistral"`)
/// - Other custom tokenizers that SGLang supports
pub struct SglangTokenizer {
    /// Python tokenizer object
    tokenizer: PyObject,
    /// Model path or HuggingFace repo ID
    #[allow(dead_code)]
    model_path: String,
    /// Tokenizer mode (e.g., "auto", "mistral")
    #[allow(dead_code)]
    tokenizer_mode: Option<String>,
}

impl SglangTokenizer {
    /// Create a new SGLang tokenizer.
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to model directory or HuggingFace repo ID
    /// * `tokenizer_mode` - Optional tokenizer mode:
    ///   - `None` or `"auto"` - Auto-detect tokenizer type (default)
    ///   - `"mistral"` - Use mistral-common tokenizer
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Python interpreter cannot be acquired
    /// - SGLang module cannot be imported
    /// - Tokenizer fails to load
    pub fn new(model_path: &str, tokenizer_mode: Option<&str>) -> Result<Self> {
        Python::with_gil(|py| {
            // Try to import SGLang's tokenizer utilities
            let sglang_utils = py.import("sglang.srt.utils.hf_transformers_utils").map_err(|e| {
                Error::msg(format!(
                    "Failed to import sglang.srt.utils.hf_transformers_utils: {}. \
                     Make sure SGLang is installed.",
                    e
                ))
            })?;

            // Build kwargs for get_tokenizer call
            let kwargs = PyDict::new(py);
            if let Some(mode) = tokenizer_mode {
                kwargs
                    .set_item("tokenizer_mode", mode)
                    .map_err(|e| Error::msg(format!("Failed to set tokenizer_mode: {}", e)))?;
            }

            // Call get_tokenizer(model_path, tokenizer_mode=...)
            let tokenizer = sglang_utils
                .getattr("get_tokenizer")
                .map_err(|e| Error::msg(format!("Failed to get get_tokenizer function: {}", e)))?
                .call((model_path,), Some(&kwargs))
                .map_err(|e| {
                    Error::msg(format!(
                        "Failed to load tokenizer for '{}': {}",
                        model_path, e
                    ))
                })?
                .into();

            Ok(Self {
                tokenizer,
                model_path: model_path.to_string(),
                tokenizer_mode: tokenizer_mode.map(String::from),
            })
        })
    }

    /// Apply chat template to messages.
    ///
    /// # Arguments
    ///
    /// * `messages` - Chat messages in OpenAI format
    /// * `tokenize` - If true, return token IDs; if false, return formatted text
    /// * `add_generation_prompt` - If true, add generation prompt to the end
    ///
    /// # Returns
    ///
    /// Either token IDs (if tokenize=true) or formatted text (if tokenize=false)
    pub fn apply_chat_template(
        &self,
        messages: &serde_json::Value,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> Result<ChatTemplateResult> {
        Python::with_gil(|py| {
            // Convert messages to Python object
            let py_messages = pythonize::pythonize(py, messages)
                .map_err(|e| Error::msg(format!("Failed to convert messages to Python: {}", e)))?;

            // Build kwargs
            let kwargs = PyDict::new(py);
            kwargs
                .set_item("tokenize", tokenize)
                .map_err(|e| Error::msg(format!("Failed to set tokenize kwarg: {}", e)))?;
            kwargs
                .set_item("add_generation_prompt", add_generation_prompt)
                .map_err(|e| {
                    Error::msg(format!("Failed to set add_generation_prompt kwarg: {}", e))
                })?;

            // Call apply_chat_template
            let result = self
                .tokenizer
                .call_method(py, "apply_chat_template", (py_messages,), Some(&kwargs))
                .map_err(|e| Error::msg(format!("Failed to apply chat template: {}", e)))?;

            if tokenize {
                // Extract as Vec<u32>
                let tokens: Vec<u32> = result
                    .extract(py)
                    .map_err(|e| Error::msg(format!("Failed to extract token IDs: {}", e)))?;
                Ok(ChatTemplateResult::Tokens(tokens))
            } else {
                // Extract as String
                let text: String = result
                    .extract(py)
                    .map_err(|e| Error::msg(format!("Failed to extract text: {}", e)))?;
                Ok(ChatTemplateResult::Text(text))
            }
        })
    }
}

/// Result of applying chat template
pub enum ChatTemplateResult {
    /// Token IDs (when tokenize=true)
    Tokens(Vec<u32>),
    /// Formatted text (when tokenize=false)
    Text(String),
}

impl Encoder for SglangTokenizer {
    fn encode(&self, input: &str) -> Result<Encoding> {
        Python::with_gil(|py| {
            let input_len = input.len();
            let start = std::time::Instant::now();

            // Call tokenizer.encode(input)
            let result = self
                .tokenizer
                .call_method1(py, "encode", (input,))
                .map_err(|e| Error::msg(format!("Failed to encode text: {}", e)))?;

            // Extract token IDs as Vec<u32>
            let token_ids: Vec<TokenIdType> = result
                .extract(py)
                .map_err(|e| Error::msg(format!("Failed to extract token IDs: {}", e)))?;

            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            tracing::info!(
                input_chars = input_len,
                num_tokens = token_ids.len(),
                elapsed_ms = format!("{:.3}", elapsed_ms),
                "SGLang tokenizer encode via PyO3"
            );

            Ok(Encoding::Sp(token_ids))
        })
    }

    fn encode_batch(&self, inputs: &[&str]) -> Result<Vec<Encoding>> {
        // For now, encode one at a time
        // TODO: Optimize by batching GIL acquisitions
        inputs.iter().map(|input| self.encode(input)).collect()
    }
}

impl Decoder for SglangTokenizer {
    fn decode(&self, token_ids: &[TokenIdType], skip_special_tokens: bool) -> Result<String> {
        Python::with_gil(|py| {
            // Convert token_ids to Python list
            let py_token_ids = token_ids.to_vec();

            // Build kwargs
            let kwargs = PyDict::new(py);
            kwargs
                .set_item("skip_special_tokens", skip_special_tokens)
                .map_err(|e| Error::msg(format!("Failed to set skip_special_tokens: {}", e)))?;

            // Call tokenizer.decode(token_ids, skip_special_tokens=...)
            let result = self
                .tokenizer
                .call_method(py, "decode", (py_token_ids,), Some(&kwargs))
                .map_err(|e| Error::msg(format!("Failed to decode tokens: {}", e)))?;

            // Extract as String
            let text: String = result
                .extract(py)
                .map_err(|e| Error::msg(format!("Failed to extract decoded text: {}", e)))?;

            Ok(text)
        })
    }
}

impl Tokenizer for SglangTokenizer {}

#[cfg(test)]
mod tests {
    // Tests would require a Python environment with SGLang installed
    // TODO: Add integration tests that run in appropriate CI environment
}
