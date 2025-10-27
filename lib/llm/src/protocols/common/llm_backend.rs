// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};

pub use super::FinishReason;
pub use super::preprocessor::PreprocessedRequest;
use crate::protocols::TokenIdType;
use dynamo_runtime::protocols::maybe_error::MaybeError;

pub type TokenType = Option<String>;
pub type LogProbs = Vec<f64>;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct TopLogprob {
    pub rank: u32,
    pub token_id: TokenIdType,
    pub token: TokenType,
    pub logprob: f64,
}
pub type TopLogprobs = Vec<Vec<TopLogprob>>; // num_tokens x top_logprobs

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct BackendOutput {
    /// New token_ids generated from the LLM Engine
    pub token_ids: Vec<TokenIdType>,

    /// Unlike [`LLMEngineOutput::tokens`], this is a vector of tokens, not an optional.
    /// The size of this vector should be the same as the size of `token_ids`.
    pub tokens: Vec<TokenType>,

    /// Decoded text from the list tokens.
    pub text: Option<String>,

    /// Optional cumulative log probabilities
    pub cum_log_probs: Option<f64>,

    /// Optional log probabilities
    pub log_probs: Option<LogProbs>,

    pub top_logprobs: Option<TopLogprobs>,

    // TODO: Enrich this with more information as can apply our first-level postprocessing
    // logic and return more detailed information
    pub finish_reason: Option<FinishReason>,
    // Model Deployment Card checksum
    //pub mdcsum: String,

    // Index field for batch requests to match OpenAI format
    pub index: Option<u32>,
}

/// The LLM engine and backnd with manage it's own state, specifically translating how a
/// given request/slot is managed on that particular backend.
///
/// For nvLLM's purpose, it has a single tracable request_id as part of it's context that
/// has propaged through the service pipeline to the backend.
///
/// This is the minimal raw output from the LLM engine. The Backend may then apply multiple
/// levels of post-processing before the BackendOutput is returns
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct LLMEngineOutput {
    // new token_ids
    pub token_ids: Vec<TokenIdType>,

    /// If the LLM Engine performs the detokenization, then this will have a Some of the detokenized
    /// text/tokens. If this value is None, then the Backend is responsible for detokenization.
    pub tokens: Option<Vec<TokenType>>,

    // decoded text -
    pub text: Option<String>,

    /// cumulative log probabilities
    pub cum_log_probs: Option<f64>,

    /// Optional log probabilities
    pub log_probs: Option<LogProbs>,

    pub top_logprobs: Option<TopLogprobs>,

    // TODO: Enrich this with more information as can apply our first-level postprocessing
    // logic and return more detailed information
    pub finish_reason: Option<FinishReason>,

    // Index field for batch requests to match OpenAI format
    pub index: Option<u32>,

    /// Disaggregated execution parameters (for prefill/decode separation)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub disaggregated_params: Option<serde_json::Value>,

    /// Additional arguments for extensibility
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra_args: Option<serde_json::Value>,
}

impl LLMEngineOutput {
    pub fn cancelled() -> Self {
        LLMEngineOutput {
            token_ids: vec![],
            tokens: None,
            text: None,
            cum_log_probs: None,
            log_probs: None,
            top_logprobs: None,
            finish_reason: Some(FinishReason::Cancelled),
            index: None,
            disaggregated_params: None,
            extra_args: None,
        }
    }

    pub fn stop() -> Self {
        LLMEngineOutput {
            token_ids: vec![],
            tokens: None,
            text: None,
            cum_log_probs: None,
            log_probs: None,
            finish_reason: Some(FinishReason::Stop),
            top_logprobs: None,
            index: None,
            disaggregated_params: None,
            extra_args: None,
        }
    }

    pub fn length() -> Self {
        LLMEngineOutput {
            token_ids: vec![],
            tokens: None,
            text: None,
            cum_log_probs: None,
            log_probs: None,
            top_logprobs: None,
            finish_reason: Some(FinishReason::Length),
            index: None,
            disaggregated_params: None,
            extra_args: None,
        }
    }

    pub fn error(err_msg: String) -> Self {
        LLMEngineOutput {
            token_ids: vec![],
            tokens: None,
            text: None,
            cum_log_probs: None,
            log_probs: None,
            top_logprobs: None,
            finish_reason: Some(FinishReason::Error(err_msg)),
            index: None,
            disaggregated_params: None,
            extra_args: None,
        }
    }
}

impl MaybeError for LLMEngineOutput {
    fn from_err(err: Box<dyn std::error::Error + Send + Sync>) -> Self {
        LLMEngineOutput::error(format!("{:?}", err))
    }

    fn err(&self) -> Option<anyhow::Error> {
        if let Some(FinishReason::Error(err_msg)) = &self.finish_reason {
            Some(anyhow::Error::msg(err_msg.clone()))
        } else {
            None
        }
    }
}

/// Raw output from embedding engines containing embedding vectors
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct EmbeddingsEngineOutput {
    /// Generated embedding vectors (one per input text)
    pub embeddings: Vec<Vec<f64>>,

    /// Token usage information
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_maybe_error() {
        let output = LLMEngineOutput::stop();
        assert!(output.err().is_none());
        assert!(output.is_ok());
        assert!(!output.is_err());

        let output = LLMEngineOutput::error("Test error".to_string());
        assert_eq!(format!("{}", output.err().unwrap()), "Test error");
        assert!(!output.is_ok());
        assert!(output.is_err());
    }
}
