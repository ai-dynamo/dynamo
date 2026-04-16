// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use axum::http::HeaderMap;
use derive_builder::Builder;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
use validator::{Validate, ValidationError};

pub use crate::protocols::common::timing::TimingInfo;

pub const HEADER_WORKER_INSTANCE_ID: &str = "x-worker-instance-id";
pub const HEADER_PREFILL_INSTANCE_ID: &str = "x-prefill-instance-id";
pub const HEADER_DP_RANK: &str = "x-dp-rank";
pub const HEADER_PREFILL_DP_RANK: &str = "x-prefill-dp-rank";
const UNSET_DP_RANK_SENTINEL: u32 = u32::MAX;

/// Apply routing overrides from HTTP headers to nvext.
///
/// Header mappings:
/// - `x-worker-instance-id` -> `backend_instance_id` and `decode_worker_id`
/// - `x-prefill-instance-id` -> `prefill_worker_id`
/// - `x-dp-rank` -> `dp_rank` (decode worker's DP rank)
/// - `x-prefill-dp-rank` -> `prefill_dp_rank`
///
/// Headers take priority over existing nvext values when present.
/// If no headers are present, returns the original nvext unchanged.
pub fn apply_header_routing_overrides(nvext: Option<NvExt>, headers: &HeaderMap) -> Option<NvExt> {
    let worker_id = headers
        .get(HEADER_WORKER_INSTANCE_ID)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok());

    let prefill_id = headers
        .get(HEADER_PREFILL_INSTANCE_ID)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok());

    let dp_rank = headers
        .get(HEADER_DP_RANK)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u32>().ok());

    let prefill_dp_rank = headers
        .get(HEADER_PREFILL_DP_RANK)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u32>().ok());
    let prefill_dp_rank = prefill_dp_rank.filter(|rank| *rank != UNSET_DP_RANK_SENTINEL);

    if worker_id.is_none() && prefill_id.is_none() && dp_rank.is_none() && prefill_dp_rank.is_none()
    {
        return nvext;
    }

    let mut ext = nvext.unwrap_or_default();
    if let Some(id) = worker_id {
        ext.backend_instance_id = Some(id);
        ext.decode_worker_id = Some(id);
    }
    if let Some(id) = prefill_id {
        ext.prefill_worker_id = Some(id);
    }
    if let Some(rank) = dp_rank {
        ext.dp_rank = Some(rank);
    }
    if let Some(rank) = prefill_dp_rank {
        ext.prefill_dp_rank = Some(rank);
    }
    Some(ext)
}

pub trait NvExtProvider {
    fn nvext(&self) -> Option<&NvExt>;
    fn raw_prompt(&self) -> Option<String>;
}

/// Worker ID information for disaggregated serving
#[derive(ToSchema, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct WorkerIdInfo {
    /// The prefill worker ID that processed this request
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill_worker_id: Option<u64>,

    /// The prefill worker's data parallel rank
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill_dp_rank: Option<u32>,

    /// The decode worker ID that processed this request
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decode_worker_id: Option<u64>,

    /// The decode worker's data parallel rank
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decode_dp_rank: Option<u32>,
}

/// NVIDIA LLM response extensions
#[derive(ToSchema, Serialize, Deserialize, Debug, Clone)]
pub struct NvExtResponse {
    /// Worker ID information (prefill and decode worker IDs)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub worker_id: Option<WorkerIdInfo>,

    /// Per-request timing information
    /// Populated when client requests `extra_fields: ["timing"]`
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timing: Option<TimingInfo>,

    /// Token IDs for GAIE Stage 1 query-only mode
    /// Contains the tokenized prompt for reuse in Stage 2
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_ids: Option<Vec<u32>>,

    /// Routed expert capture payload (SGLang-specific)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub routed_experts: Option<serde_json::Value>,
}

/// Response nvext fields requested for a given request.
///
/// The OpenAI-compatible API should only include `nvext` response fields when the
/// client explicitly opts in via `nvext.extra_fields`, except for the GAIE
/// `query_instance_id` flow which automatically returns `worker_id` and
/// `token_ids`. Note: timing is NOT auto-enabled for `query_instance_id`
/// because the query-only fast path returns no `finish_reason`, and timing
/// is only emitted on the final response chunk.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NvExtResponseFieldSelection {
    pub worker_id: bool,
    pub timing: bool,
    pub token_ids: bool,
    pub routed_experts: bool,
}

impl NvExtResponseFieldSelection {
    pub fn from_nvext(nvext: Option<&NvExt>) -> Self {
        let Some(ext) = nvext else {
            return Self::default();
        };

        let mut selection = Self::default();
        if let Some(fields) = ext.extra_fields.as_ref() {
            for field in fields {
                match field.as_str() {
                    "worker_id" => selection.worker_id = true,
                    "timing" => selection.timing = true,
                    "routed_experts" => selection.routed_experts = true,
                    _ => {}
                }
            }
        }
        if ext.has_query_instance_id_annotation() {
            selection.worker_id = true;
            selection.token_ids = true;
        }
        selection
    }
}

/// NVIDIA LLM extensions to the OpenAI API
#[derive(ToSchema, Serialize, Deserialize, Builder, Validate, Debug, Clone)]
#[validate(schema(function = "validate_nv_ext"))]
pub struct NvExt {
    /// If true, sampling will be forced to be greedy.
    /// The backend is responsible for selecting the correct backend-specific options to
    /// implement this.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub greed_sampling: Option<bool>,

    /// If true, the preproessor will try to bypass the prompt template and pass the prompt directly to
    /// to the tokenizer.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub use_raw_prompt: Option<bool>,

    /// Annotations
    /// User requests triggers which result in the request issue back out-of-band information in the SSE
    /// stream using the `event:` field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub annotations: Option<Vec<String>>,

    /// Targeted backend instance ID for the request
    /// If set, the request will be routed to backend instance with the given ID.
    /// If not set, the request will be routed to the best matching instance.
    #[builder(default, setter(strip_option))]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend_instance_id: Option<u64>,

    /// Pre-tokenized data to use instead of tokenizing the prompt
    /// If provided along with backend_instance_id, these tokens will be used directly
    /// and tokenization will be skipped.
    #[builder(default, setter(strip_option))]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token_data: Option<Vec<u32>>,

    /// Maximum number of thinking tokens allowed
    /// NOTE: Currently passed through to backends as a no-op for future implementation
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub max_thinking_tokens: Option<u32>,

    /// Extra fields to be included in the response's nvext
    /// This is a list of field names that should be populated in the response
    /// Supported fields include "worker_id", "timing", "routed_experts",
    /// which map to fields in NvExtResponse.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub extra_fields: Option<Vec<String>>,

    /// Targeted prefill worker ID for disaggregated serving (GAIE Stage 2)
    /// When set, the request will be routed to this specific prefill worker.
    #[builder(default, setter(strip_option))]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefill_worker_id: Option<u64>,

    /// Targeted decode worker ID for disaggregated serving (GAIE Stage 2)
    /// When set, the request will be routed to this specific decode worker.
    #[builder(default, setter(strip_option))]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode_worker_id: Option<u64>,

    /// Data parallel rank for the decode worker, set by the EPP via the
    /// `x-dp-rank` header. When a worker hosts multiple DP engines,
    /// this steers the request to the correct engine instance.
    #[builder(default, setter(strip_option))]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dp_rank: Option<u32>,

    /// Data parallel rank for the prefill worker in disaggregated serving,
    /// set by the EPP via the `x-prefill-dp-rank` header.
    #[builder(default, setter(strip_option))]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefill_dp_rank: Option<u32>,

    /// Agent-provided hints for request handling.
    #[builder(default, setter(strip_option))]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_hints: Option<AgentHints>,

    /// Optional request timestamp in milliseconds for trace replay / virtual-time simulation.
    #[builder(default, setter(strip_option))]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_timestamp_ms: Option<f64>,

    /// Session control for subagent KV isolation and sticky routing.
    /// When present, the router uses `session_id` for worker affinity.
    /// When `action` is set to `open` or `close`, the router also fires
    /// session lifecycle RPCs to the worker.
    #[builder(default, setter(strip_option))]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_control: Option<SessionControl>,
}

/// Hints from the agent/caller about request characteristics.
#[derive(ToSchema, Serialize, Deserialize, Builder, Debug, Clone, Default, PartialEq)]
pub struct AgentHints {
    /// Unified request priority.
    /// Higher values mean "more important" at the Dynamo API level.
    /// Dynamo uses this for router queue ordering and normalizes it per backend
    /// before forwarding engine priority values.
    #[builder(default, setter(strip_option))]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub priority: Option<i32>,

    /// Expected output sequence length (number of output tokens).
    /// Used as a hint for routing decisions to estimate resource requirements
    /// and for output block tracking decay.
    #[builder(default, setter(strip_option))]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub osl: Option<u32>,

    /// When true, after the assistant turn completes, the system will speculatively
    /// prefill the predicted next-turn prefix (conversation history with thinking
    /// content stripped) on a worker to warm the KV cache for the next request.
    #[builder(default, setter(strip_option))]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speculative_prefill: Option<bool>,

    /// Deprecated alias for router-only priority.
    /// Kept as an undocumented fallback while callers migrate to `priority`.
    #[builder(default, setter(strip_option))]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(ignore)]
    pub latency_sensitivity: Option<f64>,
}

fn default_session_timeout() -> u64 {
    300
}

/// Session control for subagent KV isolation and sticky routing.
///
/// Always requires `session_id`. The `action` field is optional:
/// - `action: "open"` on the first turn creates a streaming session on the worker
/// - `action: "close"` on the last turn frees session KV after generation
/// - No `action` on intermediate turns -- just provides `session_id` for sticky routing
#[derive(ToSchema, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct SessionControl {
    /// Unique session identifier. Present on every turn for sticky routing.
    pub session_id: String,
    /// Lifecycle action: `"open"` or `"close"`. Omit on intermediate turns.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub action: Option<SessionAction>,
    /// Inactivity timeout in seconds (default 300, only used with `action: "open"`).
    #[serde(default = "default_session_timeout")]
    pub timeout: u64,
}

/// Session lifecycle actions.
#[derive(ToSchema, Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum SessionAction {
    Open,
    Close,
}

impl Default for NvExt {
    fn default() -> Self {
        NvExt::builder().build().unwrap()
    }
}

impl NvExt {
    pub fn builder() -> NvExtBuilder {
        NvExtBuilder::default()
    }

    /// Check for a `query_instance_id:<value>` annotation (GAIE Stage 1).
    ///
    /// Must match the exact `"query_instance_id:"` key prefix used by
    /// `PreprocessedRequest::get_annotation_value` and the KvPushRouter
    /// query-only detection, so that stray annotations like
    /// `query_instance_id_extra:...` do not accidentally enable response
    /// metadata.
    pub fn has_query_instance_id_annotation(&self) -> bool {
        self.annotations.as_ref().is_some_and(|annotations| {
            annotations
                .iter()
                .any(|annotation| annotation.starts_with("query_instance_id:"))
        })
    }
}

fn validate_nv_ext(_nv_ext: &NvExt) -> Result<(), ValidationError> {
    Ok(())
}

impl NvExtBuilder {
    pub fn add_annotation(&mut self, annotation: impl Into<String>) -> &mut Self {
        self.annotations
            .get_or_insert_with(|| Some(vec![]))
            .as_mut()
            .expect("stop should always be Some(Vec)")
            .push(annotation.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use validator::Validate;

    use super::*;

    // Test default builder configuration
    #[test]
    fn test_nv_ext_builder_default() {
        let nv_ext = NvExt::builder().build().unwrap();
        assert_eq!(nv_ext.greed_sampling, None);
        assert_eq!(nv_ext.use_raw_prompt, None);
        assert_eq!(nv_ext.annotations, None);
        assert_eq!(nv_ext.backend_instance_id, None);
        assert_eq!(nv_ext.token_data, None);
        assert_eq!(nv_ext.max_thinking_tokens, None);
        assert_eq!(nv_ext.extra_fields, None);
        assert_eq!(nv_ext.prefill_worker_id, None);
        assert_eq!(nv_ext.decode_worker_id, None);
        assert_eq!(nv_ext.agent_hints, None);
        assert_eq!(nv_ext.request_timestamp_ms, None);
        assert_eq!(nv_ext.session_control, None);
    }

    // Test valid builder configurations
    #[test]
    fn test_nv_ext_builder_custom() {
        let nv_ext = NvExt::builder()
            .greed_sampling(true)
            .use_raw_prompt(true)
            .backend_instance_id(42)
            .token_data(vec![1, 2, 3, 4])
            .max_thinking_tokens(1024)
            .extra_fields(vec!["worker_id".to_string()])
            .build()
            .unwrap();

        assert_eq!(nv_ext.greed_sampling, Some(true));
        assert_eq!(nv_ext.use_raw_prompt, Some(true));
        assert_eq!(nv_ext.backend_instance_id, Some(42));
        assert_eq!(nv_ext.token_data, Some(vec![1, 2, 3, 4]));
        assert_eq!(nv_ext.max_thinking_tokens, Some(1024));
        assert_eq!(nv_ext.extra_fields, Some(vec!["worker_id".to_string()]));
        // Validate the built struct
        assert!(nv_ext.validate().is_ok());
    }

    // Test GAIE Stage 2 disaggregated worker IDs
    #[test]
    fn test_nv_ext_disagg_worker_ids() {
        let nv_ext = NvExt::builder()
            .prefill_worker_id(100)
            .decode_worker_id(200)
            .build()
            .unwrap();

        assert_eq!(nv_ext.prefill_worker_id, Some(100));
        assert_eq!(nv_ext.decode_worker_id, Some(200));
        assert!(nv_ext.validate().is_ok());
    }

    #[test]
    fn test_session_control_serde() {
        // Open action with timeout
        let sc_json = r#"{"session_id": "sub-1", "action": "open", "timeout": 60}"#;
        let sc: SessionControl = serde_json::from_str(sc_json).unwrap();
        assert_eq!(sc.action, Some(SessionAction::Open));
        assert_eq!(sc.session_id, "sub-1");
        assert_eq!(sc.timeout, 60);

        // Close action (timeout defaults to 300)
        let sc_close = r#"{"session_id": "sub-1", "action": "close"}"#;
        let sc: SessionControl = serde_json::from_str(sc_close).unwrap();
        assert_eq!(sc.action, Some(SessionAction::Close));
        assert_eq!(sc.timeout, 300);

        // Continue (no action, just session_id for sticky routing)
        let sc_continue = r#"{"session_id": "sub-1"}"#;
        let sc: SessionControl = serde_json::from_str(sc_continue).unwrap();
        assert_eq!(sc.action, None);
        assert_eq!(sc.session_id, "sub-1");

        // NvExt with session_control
        let nvext_json =
            r#"{"session_control": {"session_id": "sub-2", "action": "open", "timeout": 300}}"#;
        let nvext: NvExt = serde_json::from_str(nvext_json).unwrap();
        assert!(nvext.session_control.is_some());
        let sc = nvext.session_control.unwrap();
        assert_eq!(sc.action, Some(SessionAction::Open));
        assert_eq!(sc.session_id, "sub-2");

        // Roundtrip
        let original = SessionControl {
            session_id: "test-session".to_string(),
            action: Some(SessionAction::Close),
            timeout: 90,
        };
        let json = serde_json::to_string(&original).unwrap();
        let deser: SessionControl = serde_json::from_str(&json).unwrap();
        assert_eq!(deser, original);
    }

    #[test]
    fn test_apply_header_routing_overrides() {
        use axum::http::HeaderMap;

        let mut headers = HeaderMap::new();
        headers.insert(HEADER_WORKER_INSTANCE_ID, "123".parse().unwrap());
        headers.insert(HEADER_PREFILL_INSTANCE_ID, "456".parse().unwrap());
        headers.insert(HEADER_DP_RANK, "3".parse().unwrap());
        headers.insert(HEADER_PREFILL_DP_RANK, "5".parse().unwrap());

        let result = apply_header_routing_overrides(None, &headers).unwrap();

        assert_eq!(result.backend_instance_id, Some(123));
        assert_eq!(result.decode_worker_id, Some(123));
        assert_eq!(result.prefill_worker_id, Some(456));
        assert_eq!(result.dp_rank, Some(3));
        assert_eq!(result.prefill_dp_rank, Some(5));
    }

    #[test]
    fn test_nvext_response_field_selection_defaults_to_none() {
        let selection = NvExtResponseFieldSelection::from_nvext(None);

        assert_eq!(selection, NvExtResponseFieldSelection::default());
    }

    #[test]
    fn test_nvext_response_field_selection_respects_extra_fields() {
        let nvext = NvExt::builder()
            .extra_fields(vec!["worker_id".to_string(), "routed_experts".to_string()])
            .build()
            .unwrap();

        let selection = NvExtResponseFieldSelection::from_nvext(Some(&nvext));

        assert!(selection.worker_id);
        assert!(!selection.timing);
        assert!(!selection.token_ids);
        assert!(selection.routed_experts);
    }

    #[test]
    fn test_nvext_response_field_selection_query_instance_id_exception() {
        let nvext = NvExt::builder()
            .annotations(vec!["query_instance_id:".to_string()])
            .build()
            .unwrap();

        let selection = NvExtResponseFieldSelection::from_nvext(Some(&nvext));

        assert!(selection.worker_id);
        assert!(!selection.timing); // timing NOT auto-enabled: query-only fast path has no finish_reason
        assert!(selection.token_ids);
        assert!(!selection.routed_experts);
    }

    #[test]
    fn test_nvext_response_field_selection_rejects_stray_annotation() {
        // An annotation like "query_instance_id_extra:foo" must NOT trigger the
        // query_instance_id exception — only the exact "query_instance_id:" key
        // prefix should match, consistent with PreprocessedRequest::get_annotation_value.
        let nvext = NvExt::builder()
            .annotations(vec!["query_instance_id_extra:foo".to_string()])
            .build()
            .unwrap();

        assert_eq!(
            NvExtResponseFieldSelection::from_nvext(Some(&nvext)),
            NvExtResponseFieldSelection::default(),
        );
    }

    #[test]
    fn test_nvext_response_field_selection_worker_id_only() {
        let nvext = NvExt::builder()
            .extra_fields(vec!["worker_id".to_string()])
            .build()
            .unwrap();

        assert_eq!(
            NvExtResponseFieldSelection::from_nvext(Some(&nvext)),
            NvExtResponseFieldSelection {
                worker_id: true,
                ..Default::default()
            }
        );
    }

    #[test]
    fn test_nvext_response_field_selection_timing_only() {
        let nvext = NvExt::builder()
            .extra_fields(vec!["timing".to_string()])
            .build()
            .unwrap();

        assert_eq!(
            NvExtResponseFieldSelection::from_nvext(Some(&nvext)),
            NvExtResponseFieldSelection {
                timing: true,
                ..Default::default()
            }
        );
    }

    #[test]
    fn test_nvext_response_field_selection_routed_experts_only() {
        let nvext = NvExt::builder()
            .extra_fields(vec!["routed_experts".to_string()])
            .build()
            .unwrap();

        assert_eq!(
            NvExtResponseFieldSelection::from_nvext(Some(&nvext)),
            NvExtResponseFieldSelection {
                routed_experts: true,
                ..Default::default()
            }
        );
    }

    #[test]
    fn test_nvext_response_field_selection_multiple_extra_fields() {
        let nvext = NvExt::builder()
            .extra_fields(vec![
                "worker_id".to_string(),
                "timing".to_string(),
                "routed_experts".to_string(),
            ])
            .build()
            .unwrap();

        assert_eq!(
            NvExtResponseFieldSelection::from_nvext(Some(&nvext)),
            NvExtResponseFieldSelection {
                worker_id: true,
                timing: true,
                token_ids: false, // only enabled via query_instance_id
                routed_experts: true,
            }
        );
    }
}
