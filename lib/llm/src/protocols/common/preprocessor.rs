// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
use std::sync::Arc;

use derive_builder::Builder;
use dynamo_kv_router::{
    config::RouterConfigOverride,
    protocols::{BlockExtraInfo, RoutingConstraints, WorkerId},
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::extensions::{AgentContext, RouterParams};
use super::timing::RequestTracker;
use super::{OutputOptions, SamplingOptions, StopConditions};
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

    /// Cache namespace for request-scoped KV cache isolation.
    #[serde(
        default,
        rename = "cache_salt",
        skip_serializing_if = "Option::is_none"
    )]
    pub cache_namespace: Option<String>,

    /// Priority jump in seconds for queue ordering.
    /// A positive value decreases the effective arrival time, moving the request
    /// ahead in the scheduler queue.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub priority_jump: Option<f64>,

    /// Strict router pending-queue priority tier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strict_priority: Option<u32>,

    /// Backend engine scheduling priority forwarded to the generate call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub priority: Option<i32>,

    /// Worker IDs provided externally and not discovered by the router.
    /// When set, only workers in this set are considered during scoring.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allowed_worker_ids: Option<HashSet<WorkerId>>,

    /// Request routing constraints used for worker compatibility and soft preference.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub routing_constraints: Option<RoutingConstraints>,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default)]
pub struct BootstrapInfo {
    /// The host address for bootstrap connection
    pub bootstrap_host: String,

    /// The port for bootstrap connection
    pub bootstrap_port: u16,

    /// Unique room ID for this request's KV transfer session
    pub bootstrap_room: u64,

    /// Stable mocker lifecycle identity. Role, backend, and wire version are
    /// validated by the bootstrap registration and framing protocol.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub handoff_id: Option<Uuid>,
}

/// Directional pointer to a predecessor worker's `engine.generate` span.
/// Used for prefill→decode handoff, migration retries, and multi-modal
/// pipelines — wherever a downstream worker should render an OTel `Link`
/// back to a previous worker that handled (or attempted) the same
/// request. Framework-owned; engines do not read or write this.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct TraceLink {
    /// W3C trace_id of the predecessor span (32 hex chars).
    pub trace_id: String,
    /// W3C span_id of the predecessor span (16 hex chars).
    pub span_id: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PrefillResult {
    /// Disaggregated execution parameters. Engine-owned; the framework
    /// reads this through to the underlying inference engine without
    /// interpretation.
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

    /// Unpadded expanded prompt length. Use instead of `routing_token_ids.len()`
    /// (which includes block-padding) when a real token count is needed.
    #[serde(default)]
    pub expanded_prompt_len: usize,
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
    /// ID of the model to use.
    ///
    /// `serde(default)` so canary payloads from the runtime's
    /// `HealthCheckManager` deserialize without carrying a model name —
    /// real traffic always has this set by the preprocessor; only the
    /// in-process canary path is allowed to omit it.
    #[serde(default)]
    pub model: String,

    /// Type of prompt
    pub token_ids: Vec<TokenIdType>,

    /// Complete public token-native request. Present only for the engine-native
    /// generate API; routing reads the duplicated model/token fields above but
    /// must not mutate this payload before the worker adapter consumes it.
    #[builder(default)]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generate_request: Option<crate::protocols::inference::generate::GenerateWorkerRequest>,

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
    #[serde(default)]
    pub stop_conditions: StopConditions,

    /// SamplingOptions directs the inference engine to use sampling instead of greedy decoding.
    /// More documentation on how and on the order in which sampling options are applied
    /// are needed.
    #[serde(default)]
    pub sampling_options: SamplingOptions,

    /// OutputOptions are options that control the output of the inference engine such as whether
    /// to return log probabilities, or whether to skip special tokens in output.
    #[serde(default)]
    pub output_options: OutputOptions,

    /// The EOS token ID(s) for the Model
    /// Not every backend needs this, but those that do can find it here.
    /// TODO - refactor this to a better location
    #[builder(default)]
    #[serde(default)]
    pub eos_token_ids: Vec<TokenIdType>,

    /// The computed checksum of the Model Deployment Card (MDC).
    #[builder(default)]
    pub mdc_sum: Option<String>,

    /// User requested annotations for the request
    #[builder(default)]
    #[serde(default)]
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

    /// Multimodal encoder handoff payload, set by the frontend when
    /// forwarding a request from an Encode worker to a downstream
    /// Prefill/Aggregated peer. Engine-opaque JSON object;
    /// the framework neither inspects nor mutates the contents. Object-
    /// only by contract (see Python `require_encoder_result` and the
    /// Rust `LLMEngineOutput::encode_terminal` constructor).
    #[builder(default)]
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_optional_object"
    )]
    pub encoder_result: Option<serde_json::Value>,

    /// Directional link to a predecessor worker's `engine.generate` span.
    /// Set by `PrefillRouter` on the decode side (prefill→decode handoff)
    /// and by the migration `RetryManager` on retry attempts. Framework-
    /// owned — engines must not read or write. Consumed by `EngineAdapter`
    /// at request start to record an OTel `Link` on its `engine.generate`.
    #[builder(default)]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub migration_link: Option<TraceLink>,

    /// Bootstrap info for disaggregated serving
    #[builder(default)]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bootstrap_info: Option<BootstrapInfo>,

    /// Additional arguments for extensibility
    #[builder(default)]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra_args: Option<serde_json::Value>,

    /// Whether the backend should allow a reasoning phase before enforcing
    /// guided output. SGLang consumes this as its per-request
    /// `require_reasoning` engine argument.
    #[builder(default)]
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub require_reasoning: bool,

    /// Router-specific parameters forwarded from `nvext.router`.
    /// Consumed by router implementations (e.g. the global router) and ignored
    /// by engines/backends.
    #[builder(default)]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub router: Option<RouterParams>,

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

    /// Set by the runtime's `HealthCheckManager` when this request originated
    /// from a canary probe. Engines may use it in `generate()` to bypass
    /// cross-worker coordination (KV transfer, bootstrap handshake,
    /// `require_prefill_result`) and run local-only. The wire-format key
    /// is `_HEALTH_CHECK` so the canary payload built by
    /// `dynamo.common.backend.health_check.build_health_check_payload`
    /// (and the legacy `HealthCheckPayload` base class) round-trips through
    /// this field. Skipped from serialization when false so normal traffic
    /// doesn't carry the marker.
    #[builder(default)]
    #[serde(
        default,
        rename = "_HEALTH_CHECK",
        skip_serializing_if = "std::ops::Not::not"
    )]
    pub is_probe: bool,
}

/// Enforce the object-only `encoder_result` contract at the serde boundary.
/// The handoff payload is engine-opaque but must be a JSON object at every hop;
/// reject arrays/scalars here so a non-conforming (e.g. cross-language)
/// producer fails fast instead of leaking a malformed shape downstream.
fn deserialize_optional_object<'de, D>(
    deserializer: D,
) -> Result<Option<serde_json::Value>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Option::<serde_json::Value>::deserialize(deserializer)?;
    if let Some(v) = &value
        && !v.is_object()
    {
        return Err(serde::de::Error::custom(
            "encoder_result must be a JSON object",
        ));
    }
    Ok(value)
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
    use super::*;

    #[test]
    fn bootstrap_info_carries_only_stable_handoff_identity() {
        let handoff_id = Uuid::from_u128(42);
        let info = BootstrapInfo {
            bootstrap_host: "127.0.0.1".to_string(),
            bootstrap_port: 1234,
            bootstrap_room: 7,
            handoff_id: Some(handoff_id),
        };

        let value = serde_json::to_value(&info).unwrap();
        assert_eq!(value["handoff_id"], handoff_id.to_string());
        assert!(value.get("mocker_handoff_protocol_version").is_none());
        assert!(value.get("mocker_handoff_role").is_none());
        assert!(value.get("mocker_handoff_engine_type").is_none());
        assert_eq!(
            serde_json::from_value::<BootstrapInfo>(value)
                .unwrap()
                .handoff_id,
            Some(handoff_id)
        );
    }

    /// Covers the `is_probe` serde contract end-to-end: `rename = "_HEALTH_CHECK"`,
    /// `default`, and `skip_serializing_if`. Each assertion targets a distinct
    /// attribute; if any is removed the test fails.
    #[test]
    fn is_probe_serde_round_trip() {
        let mut req = PreprocessedRequest::builder()
            .model("t".to_string())
            .token_ids(vec![1])
            .stop_conditions(StopConditions::default())
            .sampling_options(SamplingOptions::default())
            .output_options(OutputOptions::default())
            .build()
            .unwrap();

        // skip_serializing_if: default (false) is omitted.
        assert!(!req.is_probe);
        let normal = serde_json::to_string(&req).unwrap();
        assert!(!normal.contains("_HEALTH_CHECK"), "got: {normal}");
        // default: absent marker round-trips to false.
        let back: PreprocessedRequest = serde_json::from_str(&normal).unwrap();
        assert!(!back.is_probe);

        // rename: true serializes as `_HEALTH_CHECK` and round-trips.
        req.is_probe = true;
        let probe = serde_json::to_string(&req).unwrap();
        assert!(probe.contains("\"_HEALTH_CHECK\":true"), "got: {probe}");
        let back: PreprocessedRequest = serde_json::from_str(&probe).unwrap();
        assert!(back.is_probe);
    }

    /// Covers the wire contract for the backend reasoning gate: old payloads
    /// default to false, false is omitted, and true survives serialization.
    #[test]
    fn require_reasoning_serde_round_trip() {
        let mut req = PreprocessedRequest::builder()
            .model("t".to_string())
            .token_ids(vec![1])
            .stop_conditions(StopConditions::default())
            .sampling_options(SamplingOptions::default())
            .output_options(OutputOptions::default())
            .build()
            .unwrap();

        let normal = serde_json::to_value(&req).unwrap();
        assert!(
            !normal
                .as_object()
                .unwrap()
                .contains_key("require_reasoning")
        );
        let back: PreprocessedRequest = serde_json::from_value(normal).unwrap();
        assert!(!back.require_reasoning);

        req.require_reasoning = true;
        let guided = serde_json::to_value(&req).unwrap();
        assert_eq!(guided["require_reasoning"], true);
        let back: PreprocessedRequest = serde_json::from_value(guided).unwrap();
        assert!(back.require_reasoning);
    }

    /// Canary payloads carry only engine-relevant fields. All other required
    /// fields (`model`, `stop_conditions`, `sampling_options`, etc.) must
    /// pick up `serde(default)` so the runtime's `JsonProbeAdapter` can
    /// deserialize without rewriting the JSON. Regression guard against the
    /// "missing field" failures the smoke tests hit.
    #[test]
    fn minimal_canary_payload_deserializes() {
        let req: PreprocessedRequest = serde_json::from_value(serde_json::json!({
            "token_ids": [1],
            "_HEALTH_CHECK": true,
        }))
        .unwrap();
        assert_eq!(req.token_ids, vec![1]);
        assert!(req.is_probe);
        assert_eq!(req.model, "");
    }

    /// `encoder_result` is the multimodal encoder handoff payload set by
    /// the frontend when forwarding a request to a downstream
    /// Prefill/Aggregated worker. The wire shape is engine-opaque -- the
    /// framework must round-trip the value byte-identical without
    /// inspecting or wrapping it.
    #[test]
    fn encoder_result_round_trips_through_serde() {
        let payload = serde_json::json!({
            "embedding_handle": {
                "shape": [1, 1024],
                "dtype": "fp16",
                "uri": "nixl://encoder-0/embedding-42",
            },
            "processed_token_ids": [128_000_u32, 200_001_u32, 200_002_u32],
        });
        let req = PreprocessedRequest::builder()
            .model("test/model".to_string())
            .token_ids(vec![1, 2, 3])
            .stop_conditions(StopConditions::default())
            .sampling_options(SamplingOptions::default())
            .output_options(OutputOptions::default())
            .encoder_result(Some(payload.clone()))
            .build()
            .unwrap();
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["encoder_result"], payload);

        let back: PreprocessedRequest = serde_json::from_value(json).unwrap();
        assert_eq!(back.encoder_result, Some(payload));
    }

    /// `encoder_result` defaults to `None` and is absent from the
    /// serialized payload when unset (via `skip_serializing_if`), matching
    /// the convention used by sibling optional fields like `prefill_result`.
    #[test]
    fn encoder_result_is_absent_when_none() {
        let req = PreprocessedRequest::builder()
            .model("test/model".to_string())
            .token_ids(vec![1, 2, 3])
            .stop_conditions(StopConditions::default())
            .sampling_options(SamplingOptions::default())
            .output_options(OutputOptions::default())
            .build()
            .unwrap();
        assert!(req.encoder_result.is_none());
        let json = serde_json::to_value(&req).unwrap();
        assert!(
            !json.as_object().unwrap().contains_key("encoder_result"),
            "encoder_result must be absent from wire when None; got {json}"
        );
    }

    #[test]
    fn routing_hints_cache_namespace_serializes_as_cache_salt() {
        let hints = RoutingHints {
            cache_namespace: Some("tenant-a".to_string()),
            ..Default::default()
        };

        let value = serde_json::to_value(&hints).unwrap();

        assert_eq!(value["cache_salt"], "tenant-a");
        assert!(value.get("cache_namespace").is_none());

        let decoded: RoutingHints = serde_json::from_value(value).unwrap();
        assert_eq!(decoded.cache_namespace.as_deref(), Some("tenant-a"));
    }
}
