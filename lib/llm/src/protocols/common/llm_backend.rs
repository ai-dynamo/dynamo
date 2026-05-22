// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};

pub use super::FinishReason;
pub use super::preprocessor::PreprocessedRequest;
use crate::protocols::TokenIdType;
use dynamo_protocols::types::CompletionUsage;
use dynamo_protocols::types::StopReason;
use dynamo_runtime::error::DynamoError;
use dynamo_runtime::protocols::maybe_error::MaybeError;

pub type TokenType = Option<String>;
pub type LogProbs = Vec<f64>;

/// Output type discriminator for different modalities
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum OutputType {
    #[default]
    Text,
    Image,
    Video,
    Audio,
}

/// Image URL data for responses
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ImageUrlData {
    pub url: String,
}

/// Video URL data for responses
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct VideoUrlData {
    pub url: String,
}

/// Audio URL data for responses
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct AudioUrlData {
    pub url: String,
}

/// Content part for multimodal outputs (internal representation)
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text { text: String },
    ImageUrl { image_url: ImageUrlData },
    VideoUrl { video_url: VideoUrlData },
    AudioUrl { audio_url: AudioUrlData },
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct TopLogprob {
    pub rank: u32,
    pub token_id: TokenIdType,
    pub token: TokenType,
    pub logprob: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bytes: Option<Vec<u8>>,
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

    /// The stop string or token that triggered the stop condition.
    /// This is set when finish_reason is Stop and identifies what triggered it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<StopReason>,

    // Model Deployment Card checksum
    //pub mdcsum: String,

    // Index field for batch requests to match OpenAI format
    pub index: Option<u32>,

    // Token usage information
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_usage: Option<CompletionUsage>,

    /// Disaggregated execution parameters (for prefill/decode separation).
    /// Engine-owned payload — backends pack their own KV-transfer format
    /// here (vLLM `kv_transfer_params`, SGLang bootstrap triple, TRT-LLM
    /// encoded `LlmDisaggregatedParams`). Dynamo does NOT inject framework
    /// metadata into this field — use `worker_trace_link` instead.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub disaggregated_params: Option<serde_json::Value>,

    /// Multimodal encoder handoff payload (object-only by contract).
    /// Set by Encode workers on their terminal chunk; consumed by the
    /// frontend's EncoderRouter and threaded onto the downstream
    /// PreprocessedRequest. Engine-opaque; framework does not inspect.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encoder_result: Option<serde_json::Value>,

    /// Framework-owned link to the prefill worker's span. Propagated
    /// alongside the engine's `disaggregated_params` so the decode worker
    /// can record an OTel `Link` on its `engine.generate` span. Engines
    /// should NOT read or write this field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub worker_trace_link: Option<crate::protocols::common::preprocessor::TraceLink>,

    /// Opaque engine data passed through from the backend worker to the response.
    /// Dynamo does not inspect this field; it is serialized as-is into `nvext.engine_data`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub engine_data: Option<serde_json::Value>,
}

/// The LLM engine and backnd with manage it's own state, specifically translating how a
/// given request/slot is managed on that particular backend.
///
/// For nvLLM's purpose, it has a single tracable request_id as part of it's context that
/// has propaged through the service pipeline to the backend.
///
/// This is the minimal raw output from the LLM engine. The Backend may then apply multiple
/// levels of post-processing before the BackendOutput is returns
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct LLMEngineOutput {
    // new token_ids
    pub token_ids: Vec<TokenIdType>,

    /// If the LLM Engine performs the detokenization, then this will have a Some of the detokenized
    /// text/tokens. If this value is None, then the Backend is responsible for detokenization.
    pub tokens: Option<Vec<TokenType>>,

    // decoded text -
    pub text: Option<String>,

    /// Output type discriminator (text, image, video, audio)
    #[serde(default)]
    pub output_type: OutputType,

    /// Multimodal content parts (for non-text outputs)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_parts: Option<Vec<ContentPart>>,

    /// cumulative log probabilities
    pub cum_log_probs: Option<f64>,

    /// Optional log probabilities
    pub log_probs: Option<LogProbs>,

    pub top_logprobs: Option<TopLogprobs>,

    // TODO: Enrich this with more information as can apply our first-level postprocessing
    // logic and return more detailed information
    pub finish_reason: Option<FinishReason>,

    /// The stop string or token that triggered the stop condition.
    /// This is set when finish_reason is Stop and identifies what triggered it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<StopReason>,

    // Index field for batch requests to match OpenAI format
    pub index: Option<u32>,

    /// Disaggregated execution parameters (for prefill/decode separation)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub disaggregated_params: Option<serde_json::Value>,

    /// Multimodal encoder handoff payload (object-only by contract).
    /// Set by Encode workers on their terminal chunk via
    /// `LLMEngineOutput::encode_terminal`; the post-processor copies it
    /// through to `BackendOutput.encoder_result`. Engine-opaque payload;
    /// framework does not inspect or mutate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encoder_result: Option<serde_json::Value>,

    /// Framework-owned link to the prefill worker's span (cross-process
    /// trace linking on disagg requests). Set by the framework on the
    /// prefill terminal chunk; consumed by the decode adapter. Engines
    /// should NOT read or write this field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub worker_trace_link: Option<crate::protocols::common::preprocessor::TraceLink>,

    /// Additional arguments for extensibility
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra_args: Option<serde_json::Value>,

    // Token usage information
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_usage: Option<CompletionUsage>,

    /// Opaque engine data passed through from the backend worker to the response.
    /// Dynamo does not inspect this field; it is serialized as-is into `nvext.engine_data`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub engine_data: Option<serde_json::Value>,
}

impl LLMEngineOutput {
    pub fn cancelled() -> Self {
        LLMEngineOutput {
            token_ids: vec![],
            tokens: None,
            text: None,
            output_type: OutputType::default(),
            content_parts: None,
            cum_log_probs: None,
            log_probs: None,
            top_logprobs: None,
            finish_reason: Some(FinishReason::Cancelled),
            stop_reason: None,
            index: None,
            disaggregated_params: None,
            encoder_result: None,
            worker_trace_link: None,
            extra_args: None,
            completion_usage: None,
            engine_data: None,
        }
    }

    pub fn stop() -> Self {
        LLMEngineOutput {
            token_ids: vec![],
            tokens: None,
            text: None,
            output_type: OutputType::default(),
            content_parts: None,
            cum_log_probs: None,
            log_probs: None,
            finish_reason: Some(FinishReason::Stop),
            stop_reason: None,
            top_logprobs: None,
            index: None,
            disaggregated_params: None,
            encoder_result: None,
            worker_trace_link: None,
            extra_args: None,
            completion_usage: None,
            engine_data: None,
        }
    }

    pub fn length() -> Self {
        LLMEngineOutput {
            token_ids: vec![],
            tokens: None,
            text: None,
            output_type: OutputType::default(),
            content_parts: None,
            cum_log_probs: None,
            log_probs: None,
            top_logprobs: None,
            finish_reason: Some(FinishReason::Length),
            stop_reason: None,
            index: None,
            disaggregated_params: None,
            encoder_result: None,
            worker_trace_link: None,
            extra_args: None,
            completion_usage: None,
            engine_data: None,
        }
    }

    pub fn error(err_msg: String) -> Self {
        LLMEngineOutput {
            token_ids: vec![],
            tokens: None,
            text: None,
            output_type: OutputType::default(),
            content_parts: None,
            cum_log_probs: None,
            log_probs: None,
            top_logprobs: None,
            finish_reason: Some(FinishReason::Error(err_msg)),
            stop_reason: None,
            index: None,
            disaggregated_params: None,
            encoder_result: None,
            worker_trace_link: None,
            extra_args: None,
            completion_usage: None,
            engine_data: None,
        }
    }

    /// Terminal chunk for an Encode-mode stream. The `encoder_result`
    /// payload is the engine-opaque handoff dict the downstream
    /// Prefill/Aggregated worker will receive on its
    /// `PreprocessedRequest.encoder_result`.
    ///
    /// Signature takes `serde_json::Map<String, serde_json::Value>` (not
    /// bare `Value`) so the object-only Wire Shape invariant is
    /// type-enforced and the constructor stays infallible -- there is no
    /// way to pass an array or scalar through this API. The constructor
    /// wraps the `Map` in `Value::Object(...)` internally.
    ///
    /// `index: Some(0)` matches the Python helper `encoder_terminal_chunk`
    /// so Rust and Python producers emit byte-identical terminals for the
    /// same `encoder_result`.
    pub fn encode_terminal(encoder_result: serde_json::Map<String, serde_json::Value>) -> Self {
        LLMEngineOutput {
            token_ids: vec![],
            tokens: None,
            text: None,
            output_type: OutputType::default(),
            content_parts: None,
            cum_log_probs: None,
            log_probs: None,
            top_logprobs: None,
            finish_reason: Some(FinishReason::Stop),
            stop_reason: None,
            index: Some(0),
            disaggregated_params: None,
            encoder_result: Some(serde_json::Value::Object(encoder_result)),
            worker_trace_link: None,
            extra_args: None,
            completion_usage: None,
            engine_data: None,
        }
    }
}

impl MaybeError for LLMEngineOutput {
    fn from_err(err: impl std::error::Error + 'static) -> Self {
        LLMEngineOutput::error(err.to_string())
    }

    fn err(&self) -> Option<DynamoError> {
        if let Some(FinishReason::Error(err_msg)) = &self.finish_reason {
            Some(DynamoError::msg(err_msg.clone()))
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
        assert!(format!("{}", output.err().unwrap()).contains("Test error"));
        assert!(!output.is_ok());
        assert!(output.is_err());
    }

    /// `encode_terminal` produces an Encode-mode terminal chunk with the
    /// exact field shape required by the Wire Shape contract:
    /// empty token_ids, FinishReason::Stop, index = Some(0), and the
    /// encoder_result wrapped as `Value::Object(_)` (object-only by type).
    #[test]
    fn encode_terminal_pins_terminal_chunk_shape() {
        let payload = serde_json::json!({
            "embedding_handle": {"uri": "nixl://encoder/0", "shape": [1, 1024]},
        });
        let map = payload.as_object().unwrap().clone();
        let chunk = LLMEngineOutput::encode_terminal(map);

        assert!(chunk.token_ids.is_empty(), "encode terminal has no tokens");
        assert_eq!(chunk.finish_reason, Some(FinishReason::Stop));
        assert_eq!(chunk.index, Some(0));
        let result = chunk
            .encoder_result
            .as_ref()
            .expect("encoder_result must be set");
        assert!(result.is_object(), "encoder_result must be a JSON object");
        assert_eq!(result, &payload);
        // Sibling Option fields stay None on the producer terminal.
        assert!(chunk.disaggregated_params.is_none());
        assert!(chunk.worker_trace_link.is_none());
        assert!(chunk.completion_usage.is_none());
    }
}
