// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use derive_builder::Builder;
use serde::{Deserialize, Serialize};

use super::{OutputOptions, SamplingOptions, StopConditions};
use crate::kv_router::RouterConfigOverride;
use crate::protocols::TokenIdType;

/// [`PreprocessedRequest`] is the internal representation of an LLM request. The [`dynamo.llm-preprocessor`]
/// crate is responsible for converting request from the public APIs to this internal representation.
#[derive(Serialize, Deserialize, Debug, Clone, Builder)]
pub struct PreprocessedRequest {
    /// ID of the model to use
    pub model: String,

    /// Type of prompt
    pub token_ids: Vec<TokenIdType>,

    /// StopConditions are conditions that the inference engine will use to stop generation.
    pub stop_conditions: StopConditions,

    /// SamplingOptions directs the inference engine to use sampling instead of greedy decoding.
    /// More documentation on how and on the order in which sampling options are applied
    /// are needed.
    pub sampling_options: SamplingOptions,

    /// OutputOptions are options that control the output of the inference engine such as whether
    /// to return log probabilities, or whether to skip special tokens in output.
    pub output_options: OutputOptions,

    /// The EOS token ID(s) for the Model
    /// Not every backend needs this, but those that do can find it here.
    /// TODO - refactor this to a better location
    #[builder(default)]
    pub eos_token_ids: Vec<TokenIdType>,

    /// The computed checksum of the Model Deployment Card (MDC).
    #[builder(default)]
    pub mdc_sum: Option<String>,

    /// User requested annotations for the request
    #[builder(default)]
    pub annotations: Vec<String>,

    /// Estimated number of prefix hit tokens (only used in kv aware routing)
    #[builder(default)]
    pub estimated_prefix_hit_num_blocks: Option<u32>,

    /// Targeted backend instance ID for the request
    #[builder(default)]
    pub backend_instance_id: Option<u64>,

    /// Router configuration overrides for this specific request
    #[builder(default)]
    pub router_config_override: Option<RouterConfigOverride>,

    /// Disaggregated execution parameters (for prefill/decode separation)
    #[builder(default)]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub disaggregated_params: Option<serde_json::Value>,

    /// Data parallel rank for the request (used with data parallelism)
    #[builder(default)]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dp_rank: Option<u32>,

    /// Additional arguments for extensibility
    #[builder(default)]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra_args: Option<serde_json::Value>,
}

impl PreprocessedRequest {
    pub fn has_annotation(&self, annotation: &str) -> bool {
        self.annotations.contains(&annotation.to_string())
    }
}

impl PreprocessedRequest {
    pub fn builder() -> PreprocessedRequestBuilder {
        PreprocessedRequestBuilder::default()
    }
}

/// [`PreprocessedEmbeddingRequest`] is the internal representation of an embedding request
/// after preprocessing. Contains tokenized input ready for embedding engines.
#[derive(Serialize, Deserialize, Debug, Clone, Builder)]
pub struct PreprocessedEmbeddingRequest {
    /// Tokenized input text as token IDs (one Vec per input text)
    pub token_ids: Vec<Vec<TokenIdType>>,

    /// Model to use for embedding
    pub model: String,

    /// Encoding format preference
    pub encoding_format: Option<String>,

    /// Number of dimensions for output embeddings (if supported)
    pub dimensions: Option<u32>,

    /// The computed checksum of the Model Deployment Card (MDC)
    #[builder(default)]
    pub mdc_sum: Option<String>,

    /// User requested annotations for the request
    #[builder(default)]
    pub annotations: Vec<String>,
}

impl PreprocessedEmbeddingRequest {
    pub fn has_annotation(&self, annotation: &str) -> bool {
        self.annotations.contains(&annotation.to_string())
    }
}

impl PreprocessedEmbeddingRequest {
    pub fn builder() -> PreprocessedEmbeddingRequestBuilder {
        PreprocessedEmbeddingRequestBuilder::default()
    }
}
