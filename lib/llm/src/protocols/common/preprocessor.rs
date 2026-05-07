// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
use std::sync::Arc;

use derive_builder::Builder;
use dynamo_kv_router::{
    config::RouterConfigOverride,
    protocols::{BlockExtraInfo, WorkerId},
    remote_g2_plan::{
        REMOTE_KV_REUSE_NO_PLAN_REASON_EXTRA_ARGS_KEY, REMOTE_KV_REUSE_PLAN_EXTRA_ARGS_KEY,
        RemoteKvReuseNoPlanReason, RemoteKvReusePlan,
    },
};
use serde::{Deserialize, Serialize};

use super::timing::RequestTracker;
use super::{OutputOptions, SamplingOptions, StopConditions};
use crate::agents::context::AgentContext;
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

    /// Data parallel rank for the decode worker
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dp_rank: Option<u32>,

    /// Data parallel rank for the prefill worker in disaggregated serving
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefill_dp_rank: Option<u32>,

    /// Expected number of output tokens for this request.
    /// Used as a hint for routing decisions to estimate resource requirements.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_output_tokens: Option<u32>,

    /// LORA adapter name for this request.
    /// Used for LORA-aware routing and tracking.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lora_name: Option<String>,

    /// Priority jump in seconds for queue ordering.
    /// A positive value decreases the effective arrival time, moving the request
    /// ahead in the scheduler queue.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub priority_jump: Option<f64>,

    /// Backend engine scheduling priority forwarded to the generate call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub priority: Option<i32>,

    /// Worker IDs provided externally and not discovered by the router.
    /// When set, only workers in this set are considered during scoring.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allowed_worker_ids: Option<HashSet<WorkerId>>,

    /// Session control for subagent KV isolation and sticky routing.
    /// Contains session_id (for affinity) and optional action (open/close).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_control: Option<crate::protocols::openai::nvext::SessionControl>,
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
    pub prompt_tokens_details: Option<dynamo_protocols::types::PromptTokensDetails>,
}

/// Optional multimodal routing-only data.
/// This is used by the router to compute overlaps on an alternate token sequence
/// (for example, MM-expanded tokens) without changing execution token_ids.
#[derive(Serialize, Deserialize, Debug, Clone, Default, Builder)]
#[builder(default)]
pub struct MmRoutingInfo {
    /// Token IDs to use for routing overlap computation.
    pub routing_token_ids: Vec<TokenIdType>,

    /// Block-level multimodal metadata aligned with routing_token_ids blocks.
    /// Use `None` entries for blocks without multimodal objects.
    pub block_mm_infos: Vec<Option<BlockExtraInfo>>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum MultimodalData {
    Url(url::Url),
    #[serde(rename(serialize = "Url"))]
    RawUrl(String),
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

    /// Base64-encoded PyTorch tensor containing pre-computed embeddings
    /// If provided, this takes precedence over token_ids for inference
    #[builder(default)]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_embeds: Option<String>,

    // Multimodal data
    #[builder(default)]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub multi_modal_data: Option<MultimodalDataMap>,

    /// Optional multimodal routing-only fields (separate from execution payload).
    #[builder(default)]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mm_routing_info: Option<MmRoutingInfo>,

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

    /// Optional agent identity metadata forwarded from nvext.
    #[builder(default)]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_context: Option<AgentContext>,

    /// Multimodal processor kwargs forwarded to the backend engine
    /// (e.g. `{"use_audio_in_video": true}` for omni models).
    #[builder(default)]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mm_processor_kwargs: Option<serde_json::Value>,

    /// Optional request timestamp in milliseconds forwarded from nvext.
    #[builder(default)]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_timestamp_ms: Option<f64>,

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

    /// Extract the token IDs and optional block MM info used for KV cache overlap computation.
    /// Falls back to the request's primary `token_ids` when no multimodal routing info is present.
    pub fn block_mm_routing_info(&self) -> (&[TokenIdType], Option<&[Option<BlockExtraInfo>]>) {
        let Some(mm) = self.mm_routing_info.as_ref() else {
            return (&self.token_ids, None);
        };
        let tokens = mm.routing_token_ids.as_slice();
        if tokens.is_empty() {
            return (&self.token_ids, None);
        }
        (tokens, Some(mm.block_mm_infos.as_slice()))
    }

    // TODO: replace extra_args transport with a typed routed request field after v1 hardening.
    pub fn attach_remote_kv_reuse_plan(
        &mut self,
        plan: &RemoteKvReusePlan,
    ) -> serde_json::Result<()> {
        let plan_value = serde_json::to_value(plan)?;
        let mut map = extra_args_object(self.extra_args.take());
        map.remove(REMOTE_KV_REUSE_NO_PLAN_REASON_EXTRA_ARGS_KEY);
        map.insert(
            REMOTE_KV_REUSE_PLAN_EXTRA_ARGS_KEY.to_string(),
            plan_value,
        );
        self.extra_args = Some(serde_json::Value::Object(map));
        Ok(())
    }

    pub fn attach_remote_kv_reuse_no_plan_reason(
        &mut self,
        reason: RemoteKvReuseNoPlanReason,
    ) -> serde_json::Result<()> {
        let reason_value = serde_json::to_value(reason)?;
        let mut map = extra_args_object(self.extra_args.take());
        map.remove(REMOTE_KV_REUSE_PLAN_EXTRA_ARGS_KEY);
        map.insert(
            REMOTE_KV_REUSE_NO_PLAN_REASON_EXTRA_ARGS_KEY.to_string(),
            reason_value,
        );
        self.extra_args = Some(serde_json::Value::Object(map));
        Ok(())
    }
}

fn extra_args_object(
    extra_args: Option<serde_json::Value>,
) -> serde_json::Map<String, serde_json::Value> {
    match extra_args {
        Some(serde_json::Value::Object(map)) => map,
        _ => serde_json::Map::new(),
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

#[cfg(test)]
mod tests {
    use dynamo_kv_router::{
        protocols::{LocalBlockHash, StorageTier},
        remote_g2_plan::{
            REMOTE_KV_REUSE_NO_PLAN_REASON_EXTRA_ARGS_KEY, REMOTE_KV_REUSE_PLAN_EXTRA_ARGS_KEY,
            REMOTE_KV_REUSE_PLAN_VERSION, RemoteKvReuseNoPlanReason, RemoteKvReusePlan,
        },
    };
    use serde_json::json;

    use super::PreprocessedRequest;

    fn test_request() -> PreprocessedRequest {
        PreprocessedRequest::builder()
            .model("test".to_string())
            .token_ids(vec![1, 2, 3, 4])
            .stop_conditions(Default::default())
            .sampling_options(Default::default())
            .output_options(Default::default())
            .extra_args(Some(json!({ "existing_key": "existing_value" })))
            .build()
            .unwrap()
    }

    fn test_plan() -> RemoteKvReusePlan {
        RemoteKvReusePlan {
            plan_id: "plan-1".to_string(),
            request_id: "request-1".to_string(),
            target_worker_id: 42,
            target_dp_rank: 2,
            source_worker_id: 7,
            source_dp_rank: 0,
            source_tier: StorageTier::HostPinned,
            block_hashes: vec![LocalBlockHash(11), LocalBlockHash(22)],
            planned_prefix_blocks: 2,
            block_size_tokens: 16,
            created_at_ms: 1000,
            expires_at_ms: 2000,
            plan_version: REMOTE_KV_REUSE_PLAN_VERSION,
        }
    }

    #[test]
    fn extra_args_preserves_existing_keys_when_attaching_remote_plan() {
        let mut request = test_request();
        request.attach_remote_kv_reuse_plan(&test_plan()).unwrap();

        let extra_args = request.extra_args.unwrap();
        assert_eq!(extra_args["existing_key"], "existing_value");
        assert_eq!(
            extra_args[REMOTE_KV_REUSE_PLAN_EXTRA_ARGS_KEY]["source_tier"],
            "host_pinned"
        );
    }

    #[test]
    fn extra_args_no_plan_reason_is_structured_only() {
        let mut request = test_request();
        request
            .attach_remote_kv_reuse_no_plan_reason(
                RemoteKvReuseNoPlanReason::NoRemoteG2Candidate,
            )
            .unwrap();

        let extra_args = request.extra_args.unwrap();
        assert_eq!(extra_args["existing_key"], "existing_value");
        assert_eq!(
            extra_args[REMOTE_KV_REUSE_NO_PLAN_REASON_EXTRA_ARGS_KEY],
            "no_remote_g2_candidate"
        );
        let serialized = serde_json::to_string(&extra_args).unwrap();
        for forbidden in [
            "virtual_address",
            "physical_address",
            "nixl_descriptor",
            "descriptor",
            "source_block_id",
            "target_g1_block_id",
            "prompt text",
            "11",
            "22",
        ] {
            assert!(!serialized.contains(forbidden));
        }
    }
}
