// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};

use axum::http::{HeaderMap, header::HeaderName};
use derive_builder::Builder;
use dynamo_protocols::types::StopReason;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::protocols::TokenIdType;
use crate::protocols::agents::{
    AgentContextHeaderValues, agent_context_header_values, session_affinity_header_value,
};
use crate::protocols::common::llm_backend::PromptLogprobs;
use crate::protocols::common::timing::TimingInfo;

/// Request-level taint constraints carried by `nvext.routing_constraints`.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
#[serde(deny_unknown_fields)]
pub struct RoutingConstraints {
    #[serde(default, skip_serializing_if = "HashSet::is_empty")]
    pub required_taints: HashSet<String>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub preferred_taints: HashMap<String, f32>,
}

/// Router-specific parameters carried via `nvext.router`.
#[derive(Serialize, Deserialize, Debug, Clone, Default, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RouterParams {
    /// Target time-to-first-token in milliseconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttft_target: Option<f64>,

    /// Target inter-token latency in milliseconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub itl_target: Option<f64>,
}

/// Destination for large backend metadata uploaded out of band.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MetadataUpload {
    #[serde(deserialize_with = "deserialize_metadata_upload_url")]
    pub url: String,
}

fn deserialize_metadata_upload_url<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let url = String::deserialize(deserializer)?;
    let url = url.trim();
    if url.is_empty() {
        return Err(serde::de::Error::custom(
            "metadata_upload.url must not be empty",
        ));
    }
    Ok(url.to_string())
}

/// Internal KV cache hints derived from agent lifecycle metadata.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct KvHints {
    pub evict_session: bool,
}

/// Identity metadata for agentic workloads.
#[derive(Serialize, Deserialize, Builder, Debug, Clone, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AgentContext {
    /// Stable reasoning/tool session identifier.
    pub session_id: String,

    /// Optional parent session for subagents.
    #[builder(default, setter(strip_option))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_session_id: Option<String>,

    /// Optional terminal marker for lifecycle-aware internal consumers.
    #[builder(default, setter(strip_option))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_final: Option<bool>,

    #[builder(default, setter(strip_option))]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_hints: Option<KvHints>,
}

impl AgentContext {
    pub fn builder() -> AgentContextBuilder {
        AgentContextBuilder::default()
    }
}

/// Hints from the agent/caller about request characteristics.
#[derive(Serialize, Deserialize, Builder, Debug, Clone, Default, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct AgentHints {
    /// Unified request priority.
    #[builder(default, setter(strip_option))]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub priority: Option<i32>,

    /// Strict router pending-queue priority tier.
    #[builder(default, setter(strip_option))]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strict_priority: Option<u32>,

    /// Expected output sequence length.
    #[builder(default, setter(strip_option))]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub osl: Option<u32>,

    /// Request-path speculative prefill hint.
    #[builder(default, setter(strip_option))]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speculative_prefill: Option<bool>,

    /// Deprecated alias for router-only priority.
    #[builder(default, setter(strip_option))]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latency_sensitivity: Option<f64>,
}

/// Dynamo's LLM request extension envelope.
#[derive(Serialize, Deserialize, Builder, Debug, Clone, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NvExt {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub greed_sampling: Option<bool>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub use_raw_prompt: Option<bool>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub annotations: Option<Vec<String>>,

    #[builder(default, setter(strip_option))]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend_instance_id: Option<u64>,

    #[builder(default, setter(strip_option))]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token_data: Option<Vec<u32>>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub max_thinking_tokens: Option<u32>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub cache_salt: Option<String>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub extra_fields: Option<Vec<String>>,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub metadata_upload: Option<MetadataUpload>,

    #[builder(default, setter(strip_option))]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefill_worker_id: Option<u64>,

    #[builder(default, setter(strip_option))]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode_worker_id: Option<u64>,

    #[builder(default, setter(strip_option))]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dp_rank: Option<u32>,

    #[builder(default, setter(strip_option))]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefill_dp_rank: Option<u32>,

    #[builder(default, setter(strip_option))]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_hints: Option<AgentHints>,

    #[builder(default, setter(strip_option))]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_timestamp_ms: Option<f64>,

    #[builder(default, setter(strip_option))]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub routing_constraints: Option<RoutingConstraints>,

    #[builder(default, setter(strip_option))]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub router: Option<RouterParams>,
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

    pub fn has_query_instance_id_annotation(&self) -> bool {
        self.annotations.as_ref().is_some_and(|annotations| {
            annotations
                .iter()
                .any(|annotation| annotation.starts_with("query_instance_id:"))
        })
    }
}

impl NvExtBuilder {
    pub fn add_annotation(&mut self, annotation: impl Into<String>) -> &mut Self {
        self.annotations
            .get_or_insert_with(|| Some(vec![]))
            .as_mut()
            .expect("annotations should always be Some(Vec)")
            .push(annotation.into());
        self
    }
}

pub fn parse_nvext(raw: Option<serde_json::Value>) -> anyhow::Result<Option<NvExt>> {
    raw.map(serde_json::from_value)
        .transpose()
        .map_err(|err| anyhow::anyhow!("invalid nvext: {err}"))
}

pub const HEADER_WORKER_INSTANCE_ID: &str = "x-dynamo-worker-instance-id";
pub const HEADER_PREFILL_INSTANCE_ID: &str = "x-dynamo-prefill-instance-id";
pub const HEADER_DP_RANK: &str = "x-dynamo-dp-rank";
pub const HEADER_PREFILL_DP_RANK: &str = "x-dynamo-prefill-dp-rank";
pub const HEADER_REQUEST_PRIORITY: &str = "x-dynamo-request-priority";
pub const HEADER_REQUEST_STRICT_PRIORITY: &str = "x-dynamo-request-strict-priority";
pub const HEADER_TENANT_ID: &str = "x-tenant-id";
// Compatibility aliases for the original unprefixed names. Future agents may remove these after
// the deprecation window.
pub const HEADER_WORKER_INSTANCE_ID_ALIAS: &str = "x-worker-instance-id";
pub const HEADER_PREFILL_INSTANCE_ID_ALIAS: &str = "x-prefill-instance-id";
pub const HEADER_DP_RANK_ALIAS: &str = "x-dp-rank";
pub const HEADER_DATA_PARALLEL_RANK_ALIAS: &str = "x-data-parallel-rank";
pub const HEADER_PREFILL_DP_RANK_ALIAS: &str = "x-prefill-dp-rank";
const UNSET_DP_RANK_SENTINEL: u32 = u32::MAX;

impl From<AgentContextHeaderValues> for AgentContext {
    fn from(values: AgentContextHeaderValues) -> Self {
        let kv_hints = (values.session_final == Some(true)).then_some(KvHints {
            evict_session: true,
        });
        Self {
            session_id: values.session_id,
            parent_session_id: values.parent_session_id,
            session_final: values.session_final,
            kv_hints,
        }
    }
}

pub const AGENT_CONTEXT_CONTEXT_KEY: &str = "dynamo.llm.agent_context";

pub const SESSION_AFFINITY_CONTEXT_KEY: &str = "dynamo.llm.session_affinity";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionAffinityId(String);

impl SessionAffinityId {
    pub(crate) fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

pub fn agent_context_from_headers(headers: &HeaderMap) -> Option<AgentContext> {
    agent_context_header_values(headers).map(AgentContext::from)
}

pub fn session_affinity_from_headers(
    headers: &HeaderMap,
    header_name: &HeaderName,
) -> Option<SessionAffinityId> {
    session_affinity_header_value(headers, header_name).map(SessionAffinityId::new)
}

/// Apply HTTP routing header overrides to nvext.
///
/// Header mappings:
/// - `x-dynamo-worker-instance-id` -> `backend_instance_id` and `decode_worker_id`
/// - `x-dynamo-prefill-instance-id` -> `prefill_worker_id`
/// - `x-dynamo-dp-rank` -> `dp_rank` (decode worker's DP rank)
/// - `x-dynamo-prefill-dp-rank` -> `prefill_dp_rank`
/// - `x-dynamo-request-priority` -> `agent_hints.priority`
/// - `x-dynamo-request-strict-priority` -> `agent_hints.strict_priority`
/// - `x-tenant-id` -> `cache_salt`
///
/// Routing headers take priority over existing nvext values when present.
/// If no headers are present, returns the original nvext unchanged.
pub fn apply_header_routing_overrides(nvext: Option<NvExt>, headers: &HeaderMap) -> Option<NvExt> {
    let worker_id = headers
        .get(HEADER_WORKER_INSTANCE_ID)
        .or_else(|| headers.get(HEADER_WORKER_INSTANCE_ID_ALIAS))
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok());

    let prefill_id = headers
        .get(HEADER_PREFILL_INSTANCE_ID)
        .or_else(|| headers.get(HEADER_PREFILL_INSTANCE_ID_ALIAS))
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok());

    let dp_rank = headers
        .get(HEADER_DP_RANK)
        .or_else(|| headers.get(HEADER_DP_RANK_ALIAS))
        .or_else(|| headers.get(HEADER_DATA_PARALLEL_RANK_ALIAS))
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u32>().ok());

    let prefill_dp_rank = headers
        .get(HEADER_PREFILL_DP_RANK)
        .or_else(|| headers.get(HEADER_PREFILL_DP_RANK_ALIAS))
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u32>().ok());
    let prefill_dp_rank = prefill_dp_rank.filter(|rank| *rank != UNSET_DP_RANK_SENTINEL);

    let priority = headers
        .get(HEADER_REQUEST_PRIORITY)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<i32>().ok());

    let strict_priority = headers
        .get(HEADER_REQUEST_STRICT_PRIORITY)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u32>().ok());
    let tenant_id = headers
        .get(HEADER_TENANT_ID)
        .and_then(|v| v.to_str().ok())
        .filter(|s| !s.is_empty())
        .map(str::to_owned);

    if worker_id.is_none()
        && prefill_id.is_none()
        && dp_rank.is_none()
        && prefill_dp_rank.is_none()
        && priority.is_none()
        && strict_priority.is_none()
        && tenant_id.is_none()
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
    if priority.is_some() || strict_priority.is_some() {
        let hints = ext.agent_hints.get_or_insert_with(AgentHints::default);
        if let Some(priority) = priority {
            hints.priority = Some(priority);
        }
        if let Some(strict_priority) = strict_priority {
            hints.strict_priority = Some(strict_priority);
        }
    }
    if let Some(salt) = tenant_id {
        ext.cache_salt = Some(salt);
    }
    Some(ext)
}

pub trait NvExtProvider {
    fn nvext(&self) -> Option<&NvExt>;
    fn raw_prompt(&self) -> Option<String>;
    fn unsupported_fields(&self) -> Option<&std::collections::HashMap<String, serde_json::Value>> {
        None
    }
}

/// Return the request's non-empty cache salt using Dynamo's public precedence rules.
///
/// `nvext.cache_salt` is the canonical input. The top-level `cache_salt` field remains a
/// compatibility fallback for request types that retain unsupported OpenAI fields. Empty strings
/// are treated as absent so an empty canonical value can still fall back to a non-empty legacy
/// value.
pub fn request_cache_salt<R: NvExtProvider>(request: &R) -> Option<&str> {
    request
        .nvext()
        .and_then(|nvext| nvext.cache_salt.as_deref())
        .filter(|salt| !salt.is_empty())
        .or_else(|| {
            request
                .unsupported_fields()
                .and_then(|fields| fields.get("cache_salt"))
                .and_then(|value| value.as_str())
                .filter(|salt| !salt.is_empty())
        })
}

pub fn routing_constraints_to_kv(
    constraints: RoutingConstraints,
) -> dynamo_kv_router::protocols::RoutingConstraints {
    dynamo_kv_router::protocols::RoutingConstraints {
        required_taints: constraints.required_taints,
        preferred_taints: constraints.preferred_taints,
    }
}

/// Worker ID information for disaggregated serving.
#[derive(ToSchema, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct WorkerIdInfo {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill_worker_id: Option<u64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill_dp_rank: Option<u32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub decode_worker_id: Option<u64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub decode_dp_rank: Option<u32>,
}

/// NVIDIA LLM response extensions.
#[derive(ToSchema, Serialize, Deserialize, Debug, Clone)]
pub struct NvExtResponse {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub worker_id: Option<WorkerIdInfo>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub timing: Option<TimingInfo>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_ids: Option<Vec<u32>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub routed_experts: Option<serde_json::Value>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub engine_data: Option<serde_json::Value>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<serde_json::Value>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_token_ids: Option<Vec<TokenIdType>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_logprobs: Option<PromptLogprobs>,
}

pub(crate) fn merge_response_nvext(
    target: &mut Option<serde_json::Value>,
    incoming: Option<serde_json::Value>,
) {
    let Some(incoming) = incoming else {
        return;
    };

    match (target.as_mut(), incoming) {
        (Some(serde_json::Value::Object(target_obj)), serde_json::Value::Object(incoming_obj)) => {
            for (key, value) in incoming_obj {
                match key.as_str() {
                    "completion_token_ids" => {
                        let entry = target_obj
                            .entry(&key)
                            .or_insert_with(|| serde_json::Value::Array(Vec::new()));
                        if let (serde_json::Value::Array(acc), serde_json::Value::Array(new)) =
                            (entry, value)
                        {
                            acc.extend(new);
                        }
                    }
                    "prompt_logprobs" => {
                        target_obj.insert(key, value);
                    }
                    _ => {
                        target_obj.insert(key, value);
                    }
                }
            }
        }
        (_, incoming) => {
            *target = Some(incoming);
        }
    }
}

/// Response nvext fields requested for a given request.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NvExtResponseFieldSelection {
    pub worker_id: bool,
    pub timing: bool,
    pub token_ids: bool,
    pub routed_experts: bool,
    pub engine_data: bool,
    pub stop_reason: bool,
    pub completion_token_ids: bool,
    pub prompt_logprobs: bool,
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
                    "engine_data" => selection.engine_data = true,
                    "stop_reason" => selection.stop_reason = true,
                    "completion_token_ids" => selection.completion_token_ids = true,
                    "prompt_logprobs" => selection.prompt_logprobs = true,
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

    #[allow(clippy::too_many_arguments)]
    pub fn build_response_nvext(
        &self,
        tracker: Option<&std::sync::Arc<crate::protocols::common::timing::RequestTracker>>,
        finish_reason_present: bool,
        engine_data_from_backend: Option<serde_json::Value>,
        stop_reason_from_backend: Option<StopReason>,
        completion_token_ids_from_backend: Option<&[TokenIdType]>,
        prompt_logprobs_from_backend: Option<PromptLogprobs>,
    ) -> Option<NvExtResponse> {
        let worker_id = if self.worker_id {
            tracker.and_then(|t| t.get_worker_info())
        } else {
            None
        };

        let token_ids = if self.token_ids {
            tracker.and_then(|t| t.query_token_ids().map(<[u32]>::to_vec))
        } else {
            None
        };

        let routed_experts = if self.routed_experts {
            engine_data_from_backend
                .as_ref()
                .and_then(|data| data.get("routed_experts"))
                .cloned()
        } else {
            None
        };

        let timing = if finish_reason_present && self.timing {
            tracker.map(|t| t.get_timing_info())
        } else {
            None
        };

        let engine_data = if self.engine_data {
            engine_data_from_backend
        } else {
            None
        };

        let stop_reason = if self.stop_reason {
            stop_reason_from_backend.and_then(|reason| serde_json::to_value(reason).ok())
        } else {
            None
        };

        let completion_token_ids = if self.completion_token_ids {
            completion_token_ids_from_backend.map(<[u32]>::to_vec)
        } else {
            None
        };

        let prompt_logprobs = if self.prompt_logprobs && finish_reason_present {
            prompt_logprobs_from_backend
        } else {
            None
        };

        if worker_id.is_none()
            && token_ids.is_none()
            && routed_experts.is_none()
            && timing.is_none()
            && engine_data.is_none()
            && stop_reason.is_none()
            && completion_token_ids.is_none()
            && prompt_logprobs.is_none()
        {
            return None;
        }

        Some(NvExtResponse {
            worker_id,
            timing,
            token_ids,
            routed_experts,
            engine_data,
            stop_reason,
            completion_token_ids,
            prompt_logprobs,
        })
    }
}

pub(crate) fn validate_completion_token_ids_single_choice(
    total_choices: usize,
    nvext: Option<&NvExt>,
) -> anyhow::Result<()> {
    let requested = nvext
        .and_then(|ext| ext.extra_fields.as_ref())
        .is_some_and(|fields| fields.iter().any(|field| field == "completion_token_ids"));

    if requested && total_choices > 1 {
        anyhow::bail!(
            "`nvext.extra_fields=[\"completion_token_ids\"]` requires exactly one generated choice"
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::agents::{
        HEADER_CLAUDE_CODE_AGENT_ID, HEADER_CLAUDE_CODE_PARENT_AGENT_ID,
        HEADER_CLAUDE_CODE_SESSION_ID, HEADER_CODEX_SESSION_ID, HEADER_DYNAMO_PARENT_SESSION_ID,
        HEADER_DYNAMO_SESSION_FINAL, HEADER_DYNAMO_SESSION_ID, HEADER_OPENCODE_PARENT_SESSION_ID,
        HEADER_OPENCODE_SESSION_ID,
    };

    #[derive(Default)]
    struct CacheSaltRequest {
        nvext: Option<NvExt>,
        unsupported_fields: HashMap<String, serde_json::Value>,
    }

    impl NvExtProvider for CacheSaltRequest {
        fn nvext(&self) -> Option<&NvExt> {
            self.nvext.as_ref()
        }

        fn raw_prompt(&self) -> Option<String> {
            None
        }

        fn unsupported_fields(&self) -> Option<&HashMap<String, serde_json::Value>> {
            Some(&self.unsupported_fields)
        }
    }

    #[test]
    fn request_cache_salt_uses_canonical_precedence_and_empty_fallbacks() {
        let mut request = CacheSaltRequest::default();
        assert_eq!(request_cache_salt(&request), None);

        request
            .unsupported_fields
            .insert("cache_salt".to_string(), serde_json::json!("tenant-legacy"));
        assert_eq!(request_cache_salt(&request), Some("tenant-legacy"));

        request.nvext = Some(NvExt {
            cache_salt: Some("tenant-nvext".to_string()),
            ..Default::default()
        });
        assert_eq!(request_cache_salt(&request), Some("tenant-nvext"));

        request.nvext.as_mut().unwrap().cache_salt = Some(String::new());
        assert_eq!(request_cache_salt(&request), Some("tenant-legacy"));

        request
            .unsupported_fields
            .insert("cache_salt".to_string(), serde_json::json!(""));
        assert_eq!(request_cache_salt(&request), None);
    }

    #[test]
    fn shared_nvext_builder_default() {
        let nv_ext = NvExt::builder().build().unwrap();
        assert_eq!(nv_ext.greed_sampling, None);
        assert_eq!(nv_ext.use_raw_prompt, None);
        assert_eq!(nv_ext.annotations, None);
        assert_eq!(nv_ext.backend_instance_id, None);
        assert_eq!(nv_ext.token_data, None);
        assert_eq!(nv_ext.max_thinking_tokens, None);
        assert_eq!(nv_ext.cache_salt, None);
        assert_eq!(nv_ext.extra_fields, None);
        assert_eq!(nv_ext.metadata_upload, None);
        assert_eq!(nv_ext.prefill_worker_id, None);
        assert_eq!(nv_ext.decode_worker_id, None);
        assert_eq!(nv_ext.agent_hints, None);
        assert_eq!(nv_ext.request_timestamp_ms, None);
        assert_eq!(nv_ext.routing_constraints, None);
    }

    #[test]
    fn shared_nvext_builder_custom() {
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
    }

    #[test]
    fn parse_nvext_rejects_unknown_fields_in_llm_layer() {
        let err = parse_nvext(Some(serde_json::json!({
            "unsupported_future_field": true
        })))
        .unwrap_err();

        assert!(err.to_string().contains("invalid nvext"));
        assert!(err.to_string().contains("unknown field"));
    }

    #[test]
    fn agent_hints_strict_priority_serde() {
        let hints: AgentHints = serde_json::from_str(r#"{"strict_priority":3}"#).unwrap();
        assert_eq!(hints.strict_priority, Some(3));
        assert_eq!(
            serde_json::to_string(&hints).unwrap(),
            r#"{"strict_priority":3}"#
        );

        assert!(serde_json::from_str::<AgentHints>(r#"{"strict_priority":-1}"#).is_err());
    }

    #[test]
    fn shared_nvext_disagg_worker_ids() {
        let nv_ext = NvExt::builder()
            .prefill_worker_id(100)
            .decode_worker_id(200)
            .build()
            .unwrap();

        assert_eq!(nv_ext.prefill_worker_id, Some(100));
        assert_eq!(nv_ext.decode_worker_id, Some(200));
    }

    #[test]
    fn nvext_agent_context_is_rejected() {
        for json in [
            r#"{"agent_context":{"session_id":"run-123"}}"#,
            r#"{"agent_context":{"session_id":"run-123","parent_session_id":"root-1"}}"#,
            r#"{"agent_context":{"session_id":"run-123","session_final":true}}"#,
        ] {
            let err = serde_json::from_str::<NvExt>(json).unwrap_err();
            assert!(err.to_string().contains("unknown field `agent_context`"));
        }
    }

    #[test]
    fn metadata_upload_parses_url() {
        let nvext: NvExt = serde_json::from_value(serde_json::json!({
            "metadata_upload": {
                "url": " s3://bucket/root/rollouts "
            }
        }))
        .unwrap();

        let upload = nvext.metadata_upload.as_ref().unwrap();
        assert_eq!(upload.url, "s3://bucket/root/rollouts");
        assert!(!NvExtResponseFieldSelection::from_nvext(Some(&nvext)).engine_data);

        assert!(
            serde_json::from_value::<NvExt>(serde_json::json!({
                "metadata_upload": {}
            }))
            .is_err()
        );
        assert!(
            serde_json::from_value::<NvExt>(serde_json::json!({
                "metadata_upload": {
                    "url": ""
                }
            }))
            .is_err()
        );
        assert!(
            serde_json::from_value::<NvExt>(serde_json::json!({
                "metadata_upload": {
                    "url": "s3://bucket/root/rollouts",
                    "format": "json"
                }
            }))
            .is_err()
        );
    }

    #[test]
    fn apply_header_routing_overrides_sets_worker_fields() {
        let mut headers = HeaderMap::new();
        headers.insert(HEADER_WORKER_INSTANCE_ID, "123".parse().unwrap());
        headers.insert(HEADER_PREFILL_INSTANCE_ID, "456".parse().unwrap());
        headers.insert(HEADER_DP_RANK, "3".parse().unwrap());
        headers.insert(HEADER_PREFILL_DP_RANK, "5".parse().unwrap());
        headers.insert(HEADER_WORKER_INSTANCE_ID_ALIAS, "1".parse().unwrap());
        headers.insert(HEADER_PREFILL_INSTANCE_ID_ALIAS, "2".parse().unwrap());
        headers.insert(HEADER_DP_RANK_ALIAS, "4".parse().unwrap());
        headers.insert(HEADER_PREFILL_DP_RANK_ALIAS, "6".parse().unwrap());

        let result = apply_header_routing_overrides(None, &headers).unwrap();

        assert_eq!(result.backend_instance_id, Some(123));
        assert_eq!(result.decode_worker_id, Some(123));
        assert_eq!(result.prefill_worker_id, Some(456));
        assert_eq!(result.dp_rank, Some(3));
        assert_eq!(result.prefill_dp_rank, Some(5));
    }

    #[test]
    fn apply_header_routing_overrides_sets_priorities() {
        let mut headers = HeaderMap::new();
        headers.insert(HEADER_REQUEST_PRIORITY, "-3".parse().unwrap());
        headers.insert(HEADER_REQUEST_STRICT_PRIORITY, "7".parse().unwrap());

        let hints = apply_header_routing_overrides(None, &headers)
            .unwrap()
            .agent_hints
            .unwrap();

        assert_eq!(hints.priority, Some(-3));
        assert_eq!(hints.strict_priority, Some(7));

        headers.remove(HEADER_REQUEST_STRICT_PRIORITY);
        let nvext = NvExt {
            agent_hints: Some(AgentHints {
                priority: Some(1),
                strict_priority: Some(2),
                osl: Some(99),
                ..Default::default()
            }),
            ..Default::default()
        };
        let hints = apply_header_routing_overrides(Some(nvext), &headers)
            .unwrap()
            .agent_hints
            .unwrap();

        assert_eq!(hints.priority, Some(-3));
        assert_eq!(hints.strict_priority, Some(2));
        assert_eq!(hints.osl, Some(99));
    }

    #[test]
    fn apply_header_routing_overrides_sets_cache_salt_from_tenant_header() {
        let mut headers = HeaderMap::new();
        headers.insert(HEADER_TENANT_ID, "tenant-a".parse().unwrap());

        let nvext = apply_header_routing_overrides(None, &headers).unwrap();
        assert_eq!(nvext.cache_salt.as_deref(), Some("tenant-a"));

        let mut headers = HeaderMap::new();
        headers.insert(HEADER_TENANT_ID, "tenant-header".parse().unwrap());
        let nvext = NvExt {
            cache_salt: Some("tenant-body".to_string()),
            ..Default::default()
        };

        let nvext = apply_header_routing_overrides(Some(nvext), &headers).unwrap();
        assert_eq!(nvext.cache_salt.as_deref(), Some("tenant-header"));
    }

    #[test]
    fn apply_header_routing_overrides_supports_unprefixed_aliases() {
        let mut headers = HeaderMap::new();
        headers.insert(HEADER_WORKER_INSTANCE_ID_ALIAS, "123".parse().unwrap());
        headers.insert(HEADER_PREFILL_INSTANCE_ID_ALIAS, "456".parse().unwrap());
        headers.insert(HEADER_DP_RANK_ALIAS, "3".parse().unwrap());
        headers.insert(HEADER_PREFILL_DP_RANK_ALIAS, "5".parse().unwrap());

        let result = apply_header_routing_overrides(None, &headers).unwrap();

        assert_eq!(result.backend_instance_id, Some(123));
        assert_eq!(result.decode_worker_id, Some(123));
        assert_eq!(result.prefill_worker_id, Some(456));
        assert_eq!(result.dp_rank, Some(3));
        assert_eq!(result.prefill_dp_rank, Some(5));

        headers.remove(HEADER_DP_RANK_ALIAS);
        headers.insert(HEADER_DATA_PARALLEL_RANK_ALIAS, "4".parse().unwrap());
        assert_eq!(
            apply_header_routing_overrides(None, &headers)
                .unwrap()
                .dp_rank,
            Some(4)
        );
    }

    #[test]
    fn agent_context_from_headers_derives_agent_context_table() {
        use axum::http::{HeaderMap, HeaderName};

        let cases = [
            (HEADER_CLAUDE_CODE_SESSION_ID, "claude-run-1", None, None),
            ("Session-ID", "codex-run-1", None, None),
            (
                HEADER_CODEX_SESSION_ID,
                "codex-run-2",
                Some("opencode-parent"),
                None,
            ),
            (
                HEADER_OPENCODE_SESSION_ID,
                "opencode-run-1",
                Some("parent-run-1"),
                Some("parent-run-1"),
            ),
            (HEADER_DYNAMO_SESSION_ID, "generic-run-1", None, None),
        ];

        for (header_name, header_value, parent_header_value, expected_parent_session_id) in cases {
            let mut headers = HeaderMap::new();
            headers.insert(
                header_name.parse::<HeaderName>().unwrap(),
                header_value.parse().unwrap(),
            );
            if let Some(parent) = parent_header_value {
                headers.insert(HEADER_OPENCODE_PARENT_SESSION_ID, parent.parse().unwrap());
            }

            let agent_context = agent_context_from_headers(&headers).unwrap();
            assert_eq!(agent_context.session_id.as_str(), header_value);
            assert_eq!(
                agent_context.parent_session_id.as_deref(),
                expected_parent_session_id
            );
            assert_eq!(agent_context.session_final, None);
            assert_eq!(agent_context.kv_hints, None);
        }
    }

    #[test]
    fn session_affinity_uses_configured_header() {
        let mut headers = HeaderMap::new();
        let header_name = HeaderName::from_static(HEADER_DYNAMO_SESSION_ID);
        headers.insert(
            HEADER_CLAUDE_CODE_SESSION_ID,
            "claude-session".parse().unwrap(),
        );
        headers.insert(HEADER_CODEX_SESSION_ID, "codex-session".parse().unwrap());
        headers.insert(
            HEADER_OPENCODE_SESSION_ID,
            "opencode-session".parse().unwrap(),
        );
        assert!(session_affinity_from_headers(&headers, &header_name).is_none());

        headers.insert(HEADER_DYNAMO_SESSION_ID, "canonical".parse().unwrap());
        assert_eq!(
            session_affinity_from_headers(&headers, &header_name)
                .unwrap()
                .as_str(),
            "canonical"
        );

        headers.insert(HEADER_DYNAMO_SESSION_ID, "   ".parse().unwrap());
        assert!(session_affinity_from_headers(&headers, &header_name).is_none());
    }

    #[test]
    fn native_agent_session_does_not_enable_affinity() {
        let mut headers = HeaderMap::new();
        let header_name = HeaderName::from_static(HEADER_DYNAMO_SESSION_ID);
        headers.insert(
            HEADER_CLAUDE_CODE_SESSION_ID,
            "claude-session".parse().unwrap(),
        );
        headers.insert(HEADER_CLAUDE_CODE_AGENT_ID, "claude-agent".parse().unwrap());

        let agent_context = agent_context_from_headers(&headers).unwrap();
        assert_eq!(agent_context.session_id, "claude-agent");
        assert!(session_affinity_from_headers(&headers, &header_name).is_none());

        headers.insert(
            HEADER_DYNAMO_SESSION_ID,
            "affinity-session".parse().unwrap(),
        );
        let agent_context = agent_context_from_headers(&headers).unwrap();
        assert_eq!(agent_context.session_id, "affinity-session");
        assert_eq!(
            session_affinity_from_headers(&headers, &header_name)
                .unwrap()
                .as_str(),
            "affinity-session"
        );
    }

    #[test]
    fn agent_context_from_headers_uses_claude_agent_lineage() {
        let mut headers = HeaderMap::new();
        headers.insert(
            HEADER_CLAUDE_CODE_SESSION_ID,
            "claude-session".parse().unwrap(),
        );
        headers.insert(HEADER_CLAUDE_CODE_AGENT_ID, "claude-agent".parse().unwrap());

        let agent_context = agent_context_from_headers(&headers).unwrap();

        assert_eq!(agent_context.session_id, "claude-agent");
        assert_eq!(
            agent_context.parent_session_id.as_deref(),
            Some("claude-session")
        );

        headers.insert(
            HEADER_CLAUDE_CODE_PARENT_AGENT_ID,
            "claude-parent-agent".parse().unwrap(),
        );
        assert_eq!(
            agent_context_from_headers(&headers)
                .unwrap()
                .parent_session_id
                .as_deref(),
            Some("claude-parent-agent")
        );

        headers.remove(HEADER_CLAUDE_CODE_AGENT_ID);
        let root_context = agent_context_from_headers(&headers).unwrap();
        assert_eq!(root_context.session_id, "claude-session");
        assert_eq!(root_context.parent_session_id, None);
    }

    #[test]
    fn agent_context_from_headers_reads_dynamo_parent_and_final() {
        let mut headers = HeaderMap::new();
        headers.insert(HEADER_DYNAMO_SESSION_ID, "generic-run".parse().unwrap());
        headers.insert(
            HEADER_DYNAMO_PARENT_SESSION_ID,
            "generic-parent".parse().unwrap(),
        );
        headers.insert(HEADER_DYNAMO_SESSION_FINAL, "true".parse().unwrap());

        let agent_context = agent_context_from_headers(&headers).unwrap();

        assert_eq!(agent_context.session_id, "generic-run");
        assert_eq!(
            agent_context.parent_session_id.as_deref(),
            Some("generic-parent")
        );
        assert_eq!(agent_context.session_final, Some(true));
        assert_eq!(
            agent_context.kv_hints,
            Some(KvHints {
                evict_session: true
            })
        );

        headers.insert(HEADER_DYNAMO_SESSION_FINAL, "false".parse().unwrap());
        assert_eq!(agent_context_from_headers(&headers).unwrap().kv_hints, None);
    }

    #[test]
    fn dynamo_session_headers_override_agent_native_headers() {
        let mut headers = HeaderMap::new();
        headers.insert(
            HEADER_CLAUDE_CODE_SESSION_ID,
            "claude-session".parse().unwrap(),
        );
        headers.insert(HEADER_CLAUDE_CODE_AGENT_ID, "claude-agent".parse().unwrap());
        headers.insert(HEADER_DYNAMO_SESSION_ID, "dynamo-session".parse().unwrap());
        headers.insert(
            HEADER_DYNAMO_PARENT_SESSION_ID,
            "dynamo-parent".parse().unwrap(),
        );

        let agent_context = agent_context_from_headers(&headers).unwrap();

        assert_eq!(agent_context.session_id, "dynamo-session");
        assert_eq!(
            agent_context.parent_session_id.as_deref(),
            Some("dynamo-parent")
        );
    }

    #[test]
    fn apply_header_routing_overrides_ignores_session_identity_headers() {
        use axum::http::{HeaderMap, HeaderName};

        for header_name in [
            HEADER_DYNAMO_SESSION_ID,
            HEADER_DYNAMO_PARENT_SESSION_ID,
            HEADER_DYNAMO_SESSION_FINAL,
        ] {
            let mut headers = HeaderMap::new();
            headers.insert(
                header_name.parse::<HeaderName>().unwrap(),
                "session-value".parse().unwrap(),
            );

            assert!(apply_header_routing_overrides(None, &headers).is_none());
        }
    }

    #[test]
    fn query_instance_annotation_detection_is_exact_prefix() {
        let nvext = NvExt::builder()
            .annotations(vec![
                "query_instance_id_extra:bad".to_string(),
                "query_instance_id:good".to_string(),
            ])
            .build()
            .unwrap();
        assert!(nvext.has_query_instance_id_annotation());

        let nvext = NvExt::builder()
            .annotations(vec!["query_instance_id_extra:bad".to_string()])
            .build()
            .unwrap();
        assert!(!nvext.has_query_instance_id_annotation());
    }

    #[test]
    fn response_field_selection_respects_extra_fields() {
        let nvext = NvExt::builder()
            .extra_fields(vec!["worker_id".to_string(), "routed_experts".to_string()])
            .build()
            .unwrap();

        assert_eq!(
            NvExtResponseFieldSelection::from_nvext(Some(&nvext)),
            NvExtResponseFieldSelection {
                worker_id: true,
                routed_experts: true,
                ..Default::default()
            }
        );
    }

    #[test]
    fn response_field_selection_query_instance_id_exception() {
        let nvext = NvExt::builder()
            .annotations(vec!["query_instance_id:".to_string()])
            .build()
            .unwrap();

        let selection = NvExtResponseFieldSelection::from_nvext(Some(&nvext));

        assert!(selection.worker_id);
        assert!(selection.token_ids);
        assert!(!selection.timing);
        assert!(!selection.routed_experts);
    }

    #[test]
    fn response_field_selection_multiple_extra_fields() {
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
                routed_experts: true,
                ..Default::default()
            }
        );
    }

    fn tracker_with_prefill_worker()
    -> std::sync::Arc<crate::protocols::common::timing::RequestTracker> {
        use crate::protocols::common::timing::{RequestTracker, WORKER_TYPE_PREFILL};
        let tracker = std::sync::Arc::new(RequestTracker::new());
        tracker.record_worker(42, Some(0), WORKER_TYPE_PREFILL);
        tracker
    }

    fn tracker_with_query_token_ids()
    -> std::sync::Arc<crate::protocols::common::timing::RequestTracker> {
        use crate::protocols::common::timing::RequestTracker;
        let tracker = std::sync::Arc::new(RequestTracker::new());
        tracker.set_external_query_token_ids(vec![11u32, 22, 33]);
        tracker
    }

    fn tracker_with_forwarded_worker_info()
    -> std::sync::Arc<crate::protocols::common::timing::RequestTracker> {
        use crate::protocols::common::timing::RequestTracker;
        let tracker = std::sync::Arc::new(RequestTracker::new());
        tracker.set_external_worker_info(WorkerIdInfo {
            prefill_worker_id: Some(7),
            prefill_dp_rank: Some(1),
            decode_worker_id: Some(9),
            decode_dp_rank: Some(2),
        });
        tracker
    }

    #[test]
    fn build_response_nvext_all_false_returns_none() {
        assert!(
            NvExtResponseFieldSelection::default()
                .build_response_nvext(None, false, None, None, None, None)
                .is_none()
        );
    }

    #[test]
    fn build_response_nvext_worker_id_only_without_finish() {
        let selection = NvExtResponseFieldSelection {
            worker_id: true,
            ..Default::default()
        };
        let tracker = tracker_with_prefill_worker();

        let out = selection
            .build_response_nvext(Some(&tracker), false, None, None, None, None)
            .expect("worker_id should emit regardless of finish_reason");

        assert!(out.worker_id.is_some());
        assert!(out.timing.is_none());
        assert!(out.token_ids.is_none());
        assert!(out.routed_experts.is_none());
    }

    #[test]
    fn build_response_nvext_surfaces_forwarded_split_router_worker_id() {
        let selection = NvExtResponseFieldSelection {
            worker_id: true,
            ..Default::default()
        };
        let tracker = tracker_with_forwarded_worker_info();

        let out = selection
            .build_response_nvext(Some(&tracker), false, None, None, None, None)
            .expect("forwarded worker_id should surface in nvext");

        assert_eq!(
            out.worker_id,
            Some(WorkerIdInfo {
                prefill_worker_id: Some(7),
                prefill_dp_rank: Some(1),
                decode_worker_id: Some(9),
                decode_dp_rank: Some(2),
            })
        );
    }

    #[test]
    fn build_response_nvext_timing_is_final_only() {
        let selection = NvExtResponseFieldSelection {
            timing: true,
            ..Default::default()
        };
        let tracker = tracker_with_prefill_worker();

        assert!(
            selection
                .build_response_nvext(Some(&tracker), false, None, None, None, None)
                .is_none()
        );

        let out = selection
            .build_response_nvext(Some(&tracker), true, None, None, None, None)
            .expect("timing should emit on finish");
        assert!(out.timing.is_some());
    }

    #[test]
    fn build_response_nvext_token_ids_from_tracker() {
        let selection = NvExtResponseFieldSelection {
            token_ids: true,
            ..Default::default()
        };
        let tracker = tracker_with_query_token_ids();

        let out = selection
            .build_response_nvext(Some(&tracker), false, None, None, None, None)
            .expect("token_ids should emit when present");

        assert_eq!(out.token_ids, Some(vec![11u32, 22, 33]));
    }

    #[test]
    fn build_response_nvext_routed_experts_from_engine_data() {
        let selection = NvExtResponseFieldSelection {
            routed_experts: true,
            ..Default::default()
        };
        let engine_data = serde_json::json!({ "routed_experts": {"layer_0": [1, 3]} });

        let out = selection
            .build_response_nvext(None, false, Some(engine_data), None, None, None)
            .expect("routed_experts should emit when present");

        assert_eq!(
            out.routed_experts,
            Some(serde_json::json!({"layer_0": [1, 3]}))
        );
    }

    #[test]
    fn build_response_nvext_completion_token_ids_pass_through() {
        let selection = NvExtResponseFieldSelection {
            completion_token_ids: true,
            ..Default::default()
        };

        let out = selection
            .build_response_nvext(None, false, None, None, Some(&[101u32, 102, 103]), None)
            .expect("completion_token_ids should emit when requested and present");

        assert_eq!(out.completion_token_ids, Some(vec![101u32, 102, 103]));
        assert!(out.prompt_logprobs.is_none());
    }

    #[test]
    fn build_response_nvext_prompt_logprobs_final_chunk_only() {
        let selection = NvExtResponseFieldSelection {
            prompt_logprobs: true,
            ..Default::default()
        };
        let mut entry = std::collections::HashMap::new();
        entry.insert(
            42u32,
            crate::protocols::common::llm_backend::PromptLogprobEntry {
                logprob: -1.234,
                rank: Some(1),
                decoded_token: None,
            },
        );
        let payload: PromptLogprobs = vec![None, Some(entry)];

        assert!(
            selection
                .build_response_nvext(None, false, None, None, None, Some(payload.clone()))
                .is_none()
        );

        let out = selection
            .build_response_nvext(None, true, None, None, None, Some(payload))
            .expect("prompt_logprobs should emit on the final chunk");
        let got = out.prompt_logprobs.expect("prompt_logprobs payload");
        assert_eq!(got.len(), 2);
        assert!(got[0].is_none());
        assert_eq!(
            got[1].as_ref().unwrap().get(&42u32).unwrap().logprob,
            -1.234
        );
    }

    #[test]
    fn merge_response_nvext_concatenates_completion_token_ids() {
        let mut target: Option<serde_json::Value> = None;

        merge_response_nvext(
            &mut target,
            Some(serde_json::json!({ "completion_token_ids": [10, 11, 12] })),
        );
        merge_response_nvext(
            &mut target,
            Some(serde_json::json!({ "completion_token_ids": [13, 14] })),
        );
        merge_response_nvext(
            &mut target,
            Some(serde_json::json!({
                "completion_token_ids": [15],
                "worker_id": { "decode_worker_id": 7 }
            })),
        );

        let aggregated = target.expect("aggregator state");
        assert_eq!(
            aggregated["completion_token_ids"],
            serde_json::json!([10, 11, 12, 13, 14, 15])
        );
        assert_eq!(aggregated["worker_id"]["decode_worker_id"], 7);
    }
}
