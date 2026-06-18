// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};

use axum::http::HeaderMap;
use derive_builder::Builder;
use dynamo_protocols::types::StopReason;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

use crate::protocols::TokenIdType;
use crate::protocols::agents::{AgentContextHeaderValues, agent_context_header_values};
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

/// Identity metadata for agentic workloads.
#[derive(Serialize, Deserialize, Builder, Debug, Clone, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AgentContext {
    /// Reusable session/profile class.
    pub session_type_id: String,

    /// Top-level agent run/session identifier.
    pub session_id: String,

    /// Schedulable reasoning/tool trajectory identifier.
    pub trajectory_id: String,

    /// Optional parent trajectory for subagents.
    #[builder(default, setter(strip_option))]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_trajectory_id: Option<String>,

    /// Optional terminal marker: when true, this request signals that the
    /// trajectory is complete.
    #[builder(default, setter(strip_option))]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trajectory_final: Option<bool>,
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

fn default_session_timeout() -> u64 {
    300
}

/// Session control for subagent KV isolation and sticky routing.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SessionControl {
    /// Unique session identifier. Present on every turn for sticky routing.
    pub session_id: String,
    /// Lifecycle action: `"open"`, `"bind"`, or `"close"`. Omit on intermediate turns.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub action: Option<SessionAction>,
    /// Inactivity timeout in seconds.
    #[serde(default = "default_session_timeout")]
    pub timeout: u64,
}

/// Session lifecycle actions.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum SessionAction {
    Open,
    Bind,
    Close,
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
    pub agent_context: Option<AgentContext>,

    #[builder(default, setter(strip_option))]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_timestamp_ms: Option<f64>,

    #[builder(default, setter(strip_option))]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_control: Option<SessionControl>,

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

pub const HEADER_WORKER_INSTANCE_ID: &str = "x-worker-instance-id";
pub const HEADER_PREFILL_INSTANCE_ID: &str = "x-prefill-instance-id";
pub const HEADER_DP_RANK: &str = "x-dp-rank";
/// Alias for data-parallel rank routing.
pub const HEADER_DP_RANK_ALIAS: &str = "x-data-parallel-rank";
pub const HEADER_PREFILL_DP_RANK: &str = "x-prefill-dp-rank";
const UNSET_DP_RANK_SENTINEL: u32 = u32::MAX;

impl From<AgentContextHeaderValues> for AgentContext {
    fn from(values: AgentContextHeaderValues) -> Self {
        Self {
            session_type_id: values.session_type_id.to_string(),
            session_id: values.session_id.clone(),
            trajectory_id: values.trajectory_id,
            parent_trajectory_id: values.parent_trajectory_id,
            trajectory_final: None,
        }
    }
}

/// Apply HTTP header overrides to nvext.
///
/// Header mappings:
/// - `x-worker-instance-id` -> `backend_instance_id` and `decode_worker_id`
/// - `x-prefill-instance-id` -> `prefill_worker_id`
/// - `x-dp-rank` -> `dp_rank` (decode worker's DP rank)
/// - `x-prefill-dp-rank` -> `prefill_dp_rank`
/// - coding-agent session headers -> `agent_context.session_id` and
///   `agent_context.trajectory_id`
/// - coding-agent child trajectory headers -> `agent_context.trajectory_id`
/// - coding-agent source -> `agent_context.session_type_id`
/// - configured coding-agent parent session headers ->
///   `agent_context.parent_trajectory_id`
///
/// Routing headers take priority over existing nvext values when present.
/// Explicit `nvext.agent_context` takes priority over coding-agent headers.
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
        .or_else(|| headers.get(HEADER_DP_RANK_ALIAS))
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u32>().ok());

    let prefill_dp_rank = headers
        .get(HEADER_PREFILL_DP_RANK)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u32>().ok());
    let prefill_dp_rank = prefill_dp_rank.filter(|rank| *rank != UNSET_DP_RANK_SENTINEL);

    let has_agent_context = nvext
        .as_ref()
        .is_some_and(|ext| ext.agent_context.is_some());
    let header_agent_context = if has_agent_context {
        None
    } else {
        agent_context_header_values(headers).map(AgentContext::from)
    };

    if worker_id.is_none()
        && prefill_id.is_none()
        && dp_rank.is_none()
        && prefill_dp_rank.is_none()
        && header_agent_context.is_none()
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
    if ext.agent_context.is_none() {
        ext.agent_context = header_agent_context;
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

/// Validate Dynamo extension semantics after protocol parsing.
pub fn validate_nvext_semantics(nvext: Option<&NvExt>) -> anyhow::Result<()> {
    let Some(nvext) = nvext else {
        return Ok(());
    };

    if let Some(agent_context) = nvext.agent_context.as_ref() {
        validate_non_empty(
            &agent_context.session_type_id,
            "nvext.agent_context.session_type_id",
        )?;
        validate_non_empty(&agent_context.session_id, "nvext.agent_context.session_id")?;
        validate_non_empty(
            &agent_context.trajectory_id,
            "nvext.agent_context.trajectory_id",
        )?;
        if let Some(parent_trajectory_id) = agent_context.parent_trajectory_id.as_ref() {
            validate_non_empty(
                parent_trajectory_id,
                "nvext.agent_context.parent_trajectory_id",
            )?;
        }
    }

    if let Some(session_control) = nvext.session_control.as_ref() {
        validate_non_empty(
            &session_control.session_id,
            "nvext.session_control.session_id",
        )?;
    }

    Ok(())
}

fn validate_non_empty(value: &str, field: &str) -> anyhow::Result<()> {
    if value.trim().is_empty() {
        anyhow::bail!("{field} must not be empty");
    }
    Ok(())
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
        HEADER_CLAUDE_CODE_AGENT_ID, HEADER_CLAUDE_CODE_SESSION_ID, HEADER_CODEX_SESSION_ID,
        HEADER_OPENCODE_PARENT_SESSION_ID, HEADER_OPENCODE_SESSION_ID,
    };

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
        assert_eq!(nv_ext.agent_context, None);
        assert_eq!(nv_ext.request_timestamp_ms, None);
        assert_eq!(nv_ext.session_control, None);
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
    fn shared_agent_context_semantics_reject_empty_required_ids() {
        let nvext = NvExt::builder()
            .agent_context(AgentContext {
                session_type_id: "deep_research:v1".to_string(),
                session_id: "run-123".to_string(),
                trajectory_id: " ".to_string(),
                parent_trajectory_id: None,
                trajectory_final: None,
            })
            .build()
            .unwrap();

        let err = validate_nvext_semantics(Some(&nvext)).unwrap_err();
        assert!(
            err.to_string()
                .contains("nvext.agent_context.trajectory_id must not be empty")
        );
    }

    #[test]
    fn agent_context_serde_missing_required_field_fails() {
        let json = r#"{
            "agent_context": {
                "session_type_id": "deep_research:v1",
                "trajectory_id": "run-123:researcher-0"
            }
        }"#;

        assert!(serde_json::from_str::<NvExt>(json).is_err());
    }

    #[test]
    fn session_control_defaults_timeout() {
        let sc: SessionControl =
            serde_json::from_str(r#"{"session_id":"s","action":"open"}"#).expect("session_control");
        assert_eq!(sc.action, Some(SessionAction::Open));
        assert_eq!(sc.timeout, 300);
    }

    #[test]
    fn session_control_round_trips_actions() {
        let sc: SessionControl =
            serde_json::from_str(r#"{"session_id":"sub-1","action":"bind"}"#).unwrap();
        assert_eq!(sc.action, Some(SessionAction::Bind));
        assert_eq!(sc.timeout, 300);

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

        let result = apply_header_routing_overrides(None, &headers).unwrap();

        assert_eq!(result.backend_instance_id, Some(123));
        assert_eq!(result.decode_worker_id, Some(123));
        assert_eq!(result.prefill_worker_id, Some(456));
        assert_eq!(result.dp_rank, Some(3));
        assert_eq!(result.prefill_dp_rank, Some(5));
    }

    #[test]
    fn apply_header_routing_overrides_derives_agent_context_table() {
        use axum::http::{HeaderMap, HeaderName};

        let cases = [
            (
                HEADER_CLAUDE_CODE_SESSION_ID,
                "claude-run-1",
                None,
                "claude_code",
                None,
            ),
            ("Session-ID", "codex-run-1", None, "codex", None),
            (
                HEADER_CODEX_SESSION_ID,
                "codex-run-2",
                Some("opencode-parent"),
                "codex",
                None,
            ),
            (
                HEADER_OPENCODE_SESSION_ID,
                "opencode-run-1",
                Some("parent-run-1"),
                "opencode",
                Some("parent-run-1"),
            ),
        ];

        for (
            header_name,
            header_value,
            parent_header_value,
            expected_session_type_id,
            expected_parent_trajectory_id,
        ) in cases
        {
            let mut headers = HeaderMap::new();
            headers.insert(
                header_name.parse::<HeaderName>().unwrap(),
                header_value.parse().unwrap(),
            );
            if let Some(parent) = parent_header_value {
                headers.insert(HEADER_OPENCODE_PARENT_SESSION_ID, parent.parse().unwrap());
            }

            let agent_context = apply_header_routing_overrides(None, &headers)
                .unwrap()
                .agent_context
                .unwrap();
            assert_eq!(
                (
                    agent_context.session_type_id.as_str(),
                    agent_context.session_id.as_str(),
                    agent_context.trajectory_id.as_str(),
                    agent_context.parent_trajectory_id.as_deref(),
                ),
                (
                    expected_session_type_id,
                    header_value,
                    header_value,
                    expected_parent_trajectory_id,
                )
            );
        }
    }

    #[test]
    fn apply_header_routing_overrides_uses_claude_agent_id_as_child_trajectory() {
        let mut headers = HeaderMap::new();
        headers.insert(
            HEADER_CLAUDE_CODE_SESSION_ID,
            "claude-session".parse().unwrap(),
        );
        headers.insert(HEADER_CLAUDE_CODE_AGENT_ID, "claude-agent".parse().unwrap());

        let agent_context = apply_header_routing_overrides(None, &headers)
            .unwrap()
            .agent_context
            .unwrap();

        assert_eq!(agent_context.session_type_id, "claude_code");
        assert_eq!(agent_context.session_id, "claude-session");
        assert_eq!(agent_context.trajectory_id, "claude-agent");
        assert_eq!(
            agent_context.parent_trajectory_id.as_deref(),
            Some("claude-session")
        );
    }

    #[test]
    fn apply_header_routing_overrides_ignores_non_identity_session_headers() {
        use axum::http::{HeaderMap, HeaderName};

        for header_name in ["x-session-affinity", "session_id"] {
            let mut headers = HeaderMap::new();
            headers.insert(
                header_name.parse::<HeaderName>().unwrap(),
                "session-value".parse().unwrap(),
            );

            assert!(apply_header_routing_overrides(None, &headers).is_none());
        }
    }

    #[test]
    fn apply_header_routing_overrides_explicit_agent_context_wins() {
        let mut headers = HeaderMap::new();
        headers.insert(HEADER_CODEX_SESSION_ID, "header-run".parse().unwrap());
        headers.insert(
            HEADER_OPENCODE_PARENT_SESSION_ID,
            "header-parent".parse().unwrap(),
        );
        let explicit = AgentContext::builder()
            .session_type_id("native".to_string())
            .session_id("native-session".to_string())
            .trajectory_id("native-trajectory".to_string())
            .parent_trajectory_id("native-parent".to_string())
            .build()
            .unwrap();
        let nvext = NvExt::builder()
            .agent_context(explicit.clone())
            .build()
            .unwrap();

        let result = apply_header_routing_overrides(Some(nvext), &headers).unwrap();

        assert_eq!(result.agent_context, Some(explicit));
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
