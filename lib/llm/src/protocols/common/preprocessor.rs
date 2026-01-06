// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use derive_builder::Builder;
use serde::{Deserialize, Serialize};

use super::timing::RequestTracker;
use super::{OutputOptions, SamplingOptions, StopConditions};
use crate::kv_router::RouterConfigOverride;
#[cfg(feature = "media-nixl")]
use crate::preprocessor::media::RdmaMediaDataDescriptor;
use crate::protocols::TokenIdType;

/// Routing hints for directing requests to specific workers.
/// These fields are extracted from nvext and used by the router to determine
/// which worker(s) should handle the request.
#[derive(Serialize, Deserialize, Debug, Clone, Default, Builder)]
#[builder(default)]
pub struct RoutingHints {
    /// General backend instance ID for direct routing (aggregated mode)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend_instance_id: Option<u64>,

    /// Targeted prefill worker ID for disaggregated serving (GAIE Stage 2)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefill_worker_id: Option<u64>,

    /// Targeted decode worker ID for disaggregated serving (GAIE Stage 2)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode_worker_id: Option<u64>,

    /// Data parallel rank for the request
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dp_rank: Option<u32>,

    /// Controls whether the router should manage local bookkeeping (add_request,
    /// mark_prefill_completed, free) for this request.
    ///
    /// - `None` or `Some(true)`: Router handles bookkeeping locally (default behavior)
    /// - `Some(false)`: External caller (e.g., GAIE sidecar) handles bookkeeping via C FFI
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enable_local_updates: Option<bool>,

    /// Expected number of output tokens for this request.
    /// Used as a hint for routing decisions to estimate resource requirements.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_output_tokens: Option<u32>,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct BootstrapInfo {
    /// The host address for bootstrap connection
    pub bootstrap_host: String,

    /// The port for bootstrap connection
    pub bootstrap_port: u16,

    /// Unique room ID for this request's KV transfer session
    pub bootstrap_room: u64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PrefillResult {
    /// Disaggregated execution parameters
    pub disaggregated_params: serde_json::Value,
    /// Prompt token details produced during prefill
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<dynamo_async_openai::types::PromptTokensDetails>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum MultimodalData {
    Url(url::Url),
    #[cfg(feature = "media-nixl")]
    Decoded(RdmaMediaDataDescriptor),
}

// multimodal map containing {mm_part_type: [data...]}
pub type MultimodalDataMap = std::collections::HashMap<String, Vec<MultimodalData>>;

/// [`PreprocessedRequest`] is the internal representation of an LLM request. The [`dynamo.llm-preprocessor`]
/// crate is responsible for converting request from the public APIs to this internal representation.
#[derive(Serialize, Deserialize, Debug, Clone, Builder)]
pub struct PreprocessedRequest {
    /// ID of the model to use
    pub model: String,

    /// Type of prompt
    pub token_ids: Vec<TokenIdType>,

    // Multimodal data
    #[builder(default)]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub multi_modal_data: Option<MultimodalDataMap>,

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

    /// Routing hints for worker targeting (backend_instance_id, prefill/decode worker IDs, dp_rank)
    #[builder(default)]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub routing: Option<RoutingHints>,

    /// Router configuration overrides for this specific request
    #[builder(default)]
    pub router_config_override: Option<RouterConfigOverride>,

    /// Structured prefill result
    #[builder(default)]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefill_result: Option<PrefillResult>,

    /// Bootstrap info for disaggregated serving
    #[builder(default)]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bootstrap_info: Option<BootstrapInfo>,

    /// Additional arguments for extensibility
    #[builder(default)]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra_args: Option<serde_json::Value>,

    /// Optional request tracker for per-request metrics (shared with DeltaGenerator)
    #[builder(default)]
    #[serde(skip)]
    pub tracker: Option<Arc<RequestTracker>>,
}

impl PreprocessedRequest {
    pub fn has_annotation(&self, annotation: &str) -> bool {
        self.annotations.contains(&annotation.to_string())
    }

    /// Get the value of an annotation in the format "key:value"
    /// Returns None if the annotation is not found or has no value
    pub fn get_annotation_value(&self, key: &str) -> Option<String> {
        let prefix = format!("{}:", key);
        self.annotations
            .iter()
            .find(|a| a.starts_with(&prefix))
            .map(|a| a[prefix.len()..].to_string())
    }

    pub fn builder() -> PreprocessedRequestBuilder {
        PreprocessedRequestBuilder::default()
    }

    /// Get mutable access to routing hints, creating default if None
    pub fn routing_mut(&mut self) -> &mut RoutingHints {
        self.routing.get_or_insert_with(RoutingHints::default)
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
