// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request Tokenizer for EPP integration
//!
//! This module provides standalone tokenization and prompt template functionality
//! for use by the Inference Gateway EPP. It applies the same preprocessing logic
//! as the FrontEnd pipeline but returns token IDs that can be passed to the
//! QueryRouter for worker selection.

use std::sync::Arc;

use anyhow::{Context, Result};

use crate::{
    model_card::ModelDeploymentCard,
    preprocessor::prompt::{OAIPromptFormatter, PromptFormatter},
    tokenizers::{HuggingFaceTokenizer, traits::Tokenizer},
    types::openai::chat_completions::NvCreateChatCompletionRequest,
};

/// Result of tokenizing a request
#[derive(Debug, Clone)]
pub struct TokenizedRequest {
    /// Token IDs from the tokenized prompt
    pub token_ids: Vec<u32>,
    /// The formatted prompt string (before tokenization)
    pub formatted_prompt: String,
}

/// Request tokenizer for EPP integration
///
/// Applies the same preprocessing logic as the FrontEnd pipeline:
/// - Prompt template application (chat template)
/// - Tokenization
///
/// This allows the EPP to get token IDs for routing decisions without
/// building the full pipeline.
pub struct RequestTokenizer {
    formatter: Arc<dyn OAIPromptFormatter>,
    tokenizer: Arc<dyn Tokenizer>,
}

impl RequestTokenizer {
    /// Create a new RequestTokenizer from a ModelDeploymentCard
    pub fn from_card(card: &ModelDeploymentCard) -> Result<Self> {
        let formatter = PromptFormatter::from_mdc(card)
            .context("Failed to create prompt formatter")?;
        let hf_tokenizer = card.tokenizer_hf()
            .context("Failed to load tokenizer")?;

        let PromptFormatter::OAI(formatter) = formatter;
        let tokenizer = Arc::new(HuggingFaceTokenizer::from_tokenizer(hf_tokenizer));

        Ok(Self { formatter, tokenizer })
    }

    /// Create a new RequestTokenizer with explicit components
    pub fn new(
        formatter: Arc<dyn OAIPromptFormatter>,
        hf_tokenizer: tokenizers::Tokenizer,
    ) -> Self {
        let tokenizer = Arc::new(HuggingFaceTokenizer::from_tokenizer(hf_tokenizer));
        Self { formatter, tokenizer }
    }

    /// Tokenize a chat completion request
    ///
    /// Applies the prompt template and tokenizes the result.
    /// Returns the token IDs that can be passed to QueryRouter.
    pub fn tokenize_chat_request(
        &self,
        request: &NvCreateChatCompletionRequest,
    ) -> Result<TokenizedRequest> {
        // Apply prompt template
        let formatted_prompt = self.formatter.render(request)
            .context("Failed to apply chat template")?;

        // Tokenize
        let encoding = self.tokenizer.encode(&formatted_prompt)
            .context("Failed to tokenize prompt")?;

        Ok(TokenizedRequest {
            token_ids: encoding.token_ids().to_vec(),
            formatted_prompt,
        })
    }

    /// Tokenize a raw prompt string (for completions API)
    ///
    /// Directly tokenizes the prompt without template application.
    pub fn tokenize_raw(&self, prompt: &str) -> Result<TokenizedRequest> {
        let encoding = self.tokenizer.encode(prompt)
            .context("Failed to tokenize prompt")?;

        Ok(TokenizedRequest {
            token_ids: encoding.token_ids().to_vec(),
            formatted_prompt: prompt.to_string(),
        })
    }

    /// Get the underlying tokenizer for direct access
    pub fn tokenizer(&self) -> &Arc<dyn Tokenizer> {
        &self.tokenizer
    }
}
