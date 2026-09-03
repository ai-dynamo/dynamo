// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::identity::{RoutingPartitionId, default_routing_group};
use crate::kv_hints::KvHint;
use crate::protocols::{
    DpRank, KvHintTransferWorkerMetadata, KvTransferEnforcement, RoutingConstraints,
    WorkerAffinityTarget, WorkerConfigLike, WorkerId, WorkerWithDpRank,
};
use crate::scheduling::config::RouterConfigOverride;
pub use crate::scheduling::{OverlapScoresResponse, SharedCacheOverlapScore, WorkerOverlapScore};
use crate::scheduling::{PotentialLoad, SessionContext, WorkerSelectionInputTrigger};
use crate::services::overlap::MooncakeOverlapSummary;

use super::input::PromptRequest;

const DEFAULT_MODEL_NAME: &str = "default";
pub(super) const REQUEST_BODY_LIMIT_BYTES: usize = 8 * 1024 * 1024;

fn default_model_name() -> String {
    DEFAULT_MODEL_NAME.to_string()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerLifecycle {
    Incomplete,
    Schedulable,
    Draining,
    Unschedulable,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SelectionWorkerConfig {
    pub endpoint: String,
    pub data_parallel_start_rank: u32,
    pub data_parallel_size: u32,
    pub max_num_batched_tokens: Option<u64>,
    pub total_kv_blocks: Option<u64>,
    pub stable_routing_id: Option<String>,
    pub is_eagle: Option<bool>,
    #[serde(default)]
    pub taints: HashSet<String>,
    #[serde(default)]
    pub topology_domains: HashMap<String, String>,
    pub kv_transfer_domain: Option<String>,
    pub kv_transfer_enforcement: Option<KvTransferEnforcement>,
    pub kv_transfer_preferred_weight: Option<f32>,
    /// Backend role used to match router-hint sources to targets. Presence
    /// means the worker can consume router hints.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub router_hint_worker_type: Option<String>,
    /// Per-global-DP-rank KV control endpoints a hint target fetches from.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub router_hint_source_control_endpoints: HashMap<u32, String>,
}

impl WorkerConfigLike for SelectionWorkerConfig {
    fn data_parallel_start_rank(&self) -> u32 {
        self.data_parallel_start_rank
    }

    fn data_parallel_size(&self) -> u32 {
        self.data_parallel_size
    }

    fn max_num_batched_tokens(&self) -> Option<u64> {
        self.max_num_batched_tokens
    }

    fn total_kv_blocks(&self) -> Option<u64> {
        self.total_kv_blocks
    }

    fn taints(&self) -> &HashSet<String> {
        &self.taints
    }

    fn stable_routing_id(&self) -> Option<&str> {
        self.stable_routing_id.as_deref()
    }

    fn topology_domains(&self) -> Option<&HashMap<String, String>> {
        Some(&self.topology_domains)
    }

    fn kv_transfer_domain(&self) -> Option<&str> {
        self.kv_transfer_domain.as_deref()
    }

    fn kv_transfer_enforcement(&self) -> Option<KvTransferEnforcement> {
        self.kv_transfer_enforcement
    }

    fn kv_transfer_preferred_weight(&self) -> Option<f32> {
        self.kv_transfer_preferred_weight
    }

    fn kv_hint_transfer_metadata_for_dp_rank(
        &self,
        dp_rank: DpRank,
    ) -> Option<KvHintTransferWorkerMetadata<'_>> {
        let worker_type = self.router_hint_worker_type.as_deref()?;
        if worker_type.is_empty() {
            return None;
        }
        Some(KvHintTransferWorkerMetadata {
            worker_type,
            source_control_endpoint: self
                .router_hint_source_control_endpoints
                .get(&dp_rank)
                .map(String::as_str)
                .filter(|endpoint| !endpoint.is_empty()),
        })
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct WorkerCatalogRecord {
    pub worker_id: WorkerId,
    pub model_name: String,
    pub routing_group: String,
    pub lifecycle: WorkerLifecycle,
    pub endpoint: Option<String>,
    pub kv_events_endpoint: Option<String>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub kv_events_endpoints: HashMap<u32, String>,
    pub replay_endpoint: Option<String>,
    pub block_size: Option<u32>,
    pub data_parallel_start_rank: Option<u32>,
    pub data_parallel_size: Option<u32>,
    pub max_num_batched_tokens: Option<u64>,
    pub total_kv_blocks: Option<u64>,
    pub stable_routing_id: Option<String>,
    pub is_eagle: Option<bool>,
    #[serde(default)]
    pub taints: HashSet<String>,
    #[serde(default)]
    pub topology_domains: HashMap<String, String>,
    pub kv_transfer_domain: Option<String>,
    pub kv_transfer_enforcement: Option<KvTransferEnforcement>,
    pub kv_transfer_preferred_weight: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub router_hint_worker_type: Option<String>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub router_hint_source_control_endpoints: HashMap<u32, String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub not_schedulable_reasons: Vec<String>,
}

impl WorkerCatalogRecord {
    pub(super) fn new(req: WorkerRequest) -> Self {
        Self {
            worker_id: req.worker_id,
            model_name: req.model_name,
            routing_group: req.routing_group,
            lifecycle: WorkerLifecycle::Incomplete,
            endpoint: req.endpoint,
            kv_events_endpoint: req.kv_events_endpoint,
            kv_events_endpoints: req.kv_events_endpoints,
            replay_endpoint: req.replay_endpoint,
            block_size: req.block_size,
            data_parallel_start_rank: req.data_parallel_start_rank,
            data_parallel_size: req.data_parallel_size,
            max_num_batched_tokens: req.max_num_batched_tokens,
            total_kv_blocks: req.total_kv_blocks,
            stable_routing_id: req.stable_routing_id,
            is_eagle: req.is_eagle,
            taints: req.taints,
            topology_domains: req.topology_domains,
            kv_transfer_domain: req.kv_transfer_domain,
            kv_transfer_enforcement: req.kv_transfer_enforcement,
            kv_transfer_preferred_weight: req.kv_transfer_preferred_weight,
            router_hint_worker_type: req.router_hint_worker_type,
            router_hint_source_control_endpoints: req.router_hint_source_control_endpoints,
            not_schedulable_reasons: Vec::new(),
        }
    }

    pub(super) fn key(&self) -> RoutingPartitionId {
        RoutingPartitionId::new(self.model_name.clone(), self.routing_group.clone())
    }

    pub(super) fn dp_start(&self) -> u32 {
        self.data_parallel_start_rank.unwrap_or(0)
    }

    pub(super) fn dp_size(&self) -> u32 {
        self.data_parallel_size.unwrap_or(1)
    }

    pub(super) fn dp_ranks(&self) -> impl Iterator<Item = u32> {
        let start = self.dp_start();
        let size = self.dp_size();
        start..start.saturating_add(size)
    }

    pub(super) fn scheduler_config(&self) -> Option<SelectionWorkerConfig> {
        Some(SelectionWorkerConfig {
            endpoint: self.endpoint.clone()?,
            data_parallel_start_rank: self.dp_start(),
            data_parallel_size: self.dp_size(),
            max_num_batched_tokens: self.max_num_batched_tokens,
            total_kv_blocks: self.total_kv_blocks,
            stable_routing_id: self.stable_routing_id.clone(),
            is_eagle: self.is_eagle,
            taints: self.taints.clone(),
            topology_domains: self.topology_domains.clone(),
            kv_transfer_domain: self.kv_transfer_domain.clone(),
            kv_transfer_enforcement: self.kv_transfer_enforcement,
            kv_transfer_preferred_weight: self.kv_transfer_preferred_weight,
            router_hint_worker_type: self.router_hint_worker_type.clone(),
            router_hint_source_control_endpoints: self.router_hint_source_control_endpoints.clone(),
        })
    }

    pub(super) fn listener_endpoints(&self) -> HashMap<u32, String> {
        if !self.kv_events_endpoints.is_empty() {
            return self.kv_events_endpoints.clone();
        }

        match (self.dp_size(), self.kv_events_endpoint.clone()) {
            (1, Some(endpoint)) => HashMap::from([(self.dp_start(), endpoint)]),
            _ => HashMap::new(),
        }
    }

    pub(super) fn missing_schedulable_metadata(
        &self,
        queueing_enabled: bool,
        kv_events_enabled: bool,
    ) -> Vec<String> {
        let mut missing = Vec::new();

        if self.endpoint.as_deref().is_none_or(str::is_empty) {
            missing.push("endpoint is required".to_string());
        }
        if self.block_size.is_none_or(|block_size| block_size == 0) {
            missing.push("block_size must be greater than 0".to_string());
        }
        if self.dp_size() == 0 {
            missing.push("data_parallel_size must be greater than 0".to_string());
        }
        if queueing_enabled && self.max_num_batched_tokens.is_none() {
            missing
                .push("max_num_batched_tokens is required while queueing is enabled".to_string());
        }
        if kv_events_enabled {
            let endpoints = self.listener_endpoints();
            for rank in self.dp_ranks() {
                if endpoints
                    .get(&rank)
                    .is_none_or(|endpoint| endpoint.is_empty())
                {
                    missing.push(format!("kv_events endpoint is required for dp_rank {rank}"));
                }
            }
        }

        missing
    }
}

// Implemented manually because `model_name` and `routing_group` have custom
// default values.
impl Default for WorkerRequest {
    fn default() -> Self {
        Self {
            worker_id: 0,
            model_name: default_model_name(),
            routing_group: default_routing_group(),
            endpoint: None,
            kv_events_endpoint: None,
            kv_events_endpoints: HashMap::new(),
            replay_endpoint: None,
            block_size: None,
            data_parallel_start_rank: None,
            data_parallel_size: None,
            max_num_batched_tokens: None,
            total_kv_blocks: None,
            stable_routing_id: None,
            is_eagle: None,
            taints: HashSet::new(),
            topology_domains: HashMap::new(),
            kv_transfer_domain: None,
            kv_transfer_enforcement: None,
            kv_transfer_preferred_weight: None,
            router_hint_worker_type: None,
            router_hint_source_control_endpoints: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct WorkerRequest {
    pub worker_id: WorkerId,
    #[serde(default = "default_model_name")]
    pub model_name: String,
    #[serde(default = "default_routing_group")]
    pub routing_group: String,
    pub endpoint: Option<String>,
    pub kv_events_endpoint: Option<String>,
    #[serde(default)]
    pub kv_events_endpoints: HashMap<u32, String>,
    pub replay_endpoint: Option<String>,
    pub block_size: Option<u32>,
    pub data_parallel_start_rank: Option<u32>,
    pub data_parallel_size: Option<u32>,
    pub max_num_batched_tokens: Option<u64>,
    pub total_kv_blocks: Option<u64>,
    pub stable_routing_id: Option<String>,
    pub is_eagle: Option<bool>,
    #[serde(default)]
    pub taints: HashSet<String>,
    #[serde(default)]
    pub topology_domains: HashMap<String, String>,
    pub kv_transfer_domain: Option<String>,
    pub kv_transfer_enforcement: Option<KvTransferEnforcement>,
    pub kv_transfer_preferred_weight: Option<f32>,
    /// Backend role for router-hint source/target matching. Set it to mark the
    /// worker as able to consume router hints.
    #[serde(default)]
    pub router_hint_worker_type: Option<String>,
    /// Per-global-DP-rank KV control endpoints this worker can serve hints from.
    #[serde(default)]
    pub router_hint_source_control_endpoints: HashMap<u32, String>,
}

#[derive(Debug, Deserialize)]
pub struct WorkerPatchRequest {
    pub endpoint: Option<String>,
    pub kv_events_endpoint: Option<String>,
    pub kv_events_endpoints: Option<HashMap<u32, String>>,
    pub replay_endpoint: Option<String>,
    pub block_size: Option<u32>,
    pub data_parallel_start_rank: Option<u32>,
    pub data_parallel_size: Option<u32>,
    pub max_num_batched_tokens: Option<u64>,
    pub total_kv_blocks: Option<u64>,
    pub stable_routing_id: Option<String>,
    pub is_eagle: Option<bool>,
    pub taints: Option<HashSet<String>>,
    pub topology_domains: Option<HashMap<String, String>>,
    pub kv_transfer_domain: Option<String>,
    pub kv_transfer_enforcement: Option<KvTransferEnforcement>,
    pub kv_transfer_preferred_weight: Option<f32>,
    #[serde(default)]
    pub router_hint_worker_type: Option<String>,
    #[serde(default)]
    pub router_hint_source_control_endpoints: Option<HashMap<u32, String>>,
}

impl WorkerCatalogRecord {
    pub(super) fn apply_patch(&mut self, patch: WorkerPatchRequest) {
        // TODO(rank-aware-kv-capacity): when the rank map is added, treat rank range, map,
        // scalar fallback, and provenance as one replace-only snapshot. A legacy scalar/range
        // patch must clear stale exact data rather than leave it winning lookup precedence.
        if patch.endpoint.is_some() {
            self.endpoint = patch.endpoint;
        }
        if patch.kv_events_endpoint.is_some() {
            self.kv_events_endpoint = patch.kv_events_endpoint;
        }
        if let Some(endpoints) = patch.kv_events_endpoints {
            self.kv_events_endpoints = endpoints;
        }
        if patch.replay_endpoint.is_some() {
            self.replay_endpoint = patch.replay_endpoint;
        }
        if patch.block_size.is_some() {
            self.block_size = patch.block_size;
        }
        if patch.data_parallel_start_rank.is_some() {
            self.data_parallel_start_rank = patch.data_parallel_start_rank;
        }
        if patch.data_parallel_size.is_some() {
            self.data_parallel_size = patch.data_parallel_size;
        }
        if patch.max_num_batched_tokens.is_some() {
            self.max_num_batched_tokens = patch.max_num_batched_tokens;
        }
        if patch.total_kv_blocks.is_some() {
            self.total_kv_blocks = patch.total_kv_blocks;
        }
        if patch.stable_routing_id.is_some() {
            self.stable_routing_id = patch.stable_routing_id;
        }
        if patch.is_eagle.is_some() {
            self.is_eagle = patch.is_eagle;
        }
        if let Some(taints) = patch.taints {
            self.taints = taints;
        }
        if let Some(topology_domains) = patch.topology_domains {
            self.topology_domains = topology_domains;
        }
        if patch.kv_transfer_domain.is_some() {
            self.kv_transfer_domain = patch.kv_transfer_domain;
        }
        if patch.kv_transfer_enforcement.is_some() {
            self.kv_transfer_enforcement = patch.kv_transfer_enforcement;
        }
        if patch.kv_transfer_preferred_weight.is_some() {
            self.kv_transfer_preferred_weight = patch.kv_transfer_preferred_weight;
        }
        if patch.router_hint_worker_type.is_some() {
            self.router_hint_worker_type = patch.router_hint_worker_type;
        }
        if let Some(endpoints) = patch.router_hint_source_control_endpoints {
            self.router_hint_source_control_endpoints = endpoints;
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct SelectRequest {
    #[serde(default = "default_model_name")]
    pub model_name: String,
    #[serde(default = "default_routing_group")]
    pub routing_group: String,
    pub selection_id: Option<String>,
    #[serde(flatten)]
    pub prompt: PromptRequest,
    pub router_config_override: Option<RouterConfigOverride>,
    pub expected_output_tokens: Option<u32>,
    pub priority_jump: Option<f64>,
    pub strict_priority: Option<u32>,
    /// Legacy session identity. Ignored when `session_context` is present.
    pub session_id: Option<String>,
    pub session_context: Option<SelectionSessionContext>,
    pub affinity_target: Option<WorkerAffinityTarget>,
    pub pinned_worker: Option<WorkerWithDpRank>,
    pub allowed_worker_ids: Option<HashSet<WorkerId>>,
    #[serde(default)]
    pub routing_constraints: RoutingConstraints,
    /// Select from current scheduler state without queue admission.
    ///
    /// The response then carries the chosen worker's `worker_load` snapshot
    /// and `prefill_busy` evaluation. The request never waits in the router
    /// queue; when the queue would have rejected it the call fails the same
    /// way an admitted selection does. Rejected on `select_and_reserve`.
    #[serde(default)]
    pub advisory: bool,
}

impl SelectRequest {
    pub(super) fn take_session_context(&mut self) -> Option<SessionContext> {
        resolve_session_context(self.session_context.take(), self.session_id.take())
    }
}

#[derive(Debug, Deserialize)]
pub struct SelectAndReserveRequest {
    #[serde(default = "default_model_name")]
    pub model_name: String,
    #[serde(default = "default_routing_group")]
    pub routing_group: String,
    pub selection_id: Option<String>,
    #[serde(flatten)]
    pub prompt: PromptRequest,
    pub router_config_override: Option<RouterConfigOverride>,
    pub expected_output_tokens: Option<u32>,
    pub priority_jump: Option<f64>,
    pub strict_priority: Option<u32>,
    /// Legacy session identity. Ignored when `session_context` is present.
    pub session_id: Option<String>,
    pub session_context: Option<SelectionSessionContext>,
    pub affinity_target: Option<WorkerAffinityTarget>,
    pub pinned_worker: Option<WorkerWithDpRank>,
    pub allowed_worker_ids: Option<HashSet<WorkerId>>,
    #[serde(default)]
    pub routing_constraints: RoutingConstraints,
}

impl SelectAndReserveRequest {
    pub(super) fn take_session_context(&mut self) -> Option<SessionContext> {
        resolve_session_context(self.session_context.take(), self.session_id.take())
    }
}

/// Session metadata handed to worker selection, mirroring the frontend's
/// agent-context extension field for field.
///
/// `session_context` supersedes the flat `session_id`: when both are present
/// the structured form wins.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct SelectionSessionContext {
    pub session_id: String,
    #[serde(default)]
    pub parent_session_id: Option<String>,
    #[serde(default)]
    pub session_final: Option<bool>,
    #[serde(default)]
    pub input_trigger: Option<SelectionInputTrigger>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SelectionInputTrigger {
    UserMessage,
    ToolResult,
    Other,
}

impl From<SelectionSessionContext> for SessionContext {
    fn from(context: SelectionSessionContext) -> Self {
        // Exhaustive so a new wire field must be mapped here.
        let SelectionSessionContext {
            session_id,
            parent_session_id,
            session_final,
            input_trigger,
        } = context;
        SessionContext::new(
            session_id,
            parent_session_id,
            session_final,
            input_trigger.map(|trigger| match trigger {
                SelectionInputTrigger::UserMessage => WorkerSelectionInputTrigger::UserMessage,
                SelectionInputTrigger::ToolResult => WorkerSelectionInputTrigger::ToolResult,
                SelectionInputTrigger::Other => WorkerSelectionInputTrigger::Other,
            }),
        )
    }
}

fn resolve_session_context(
    session_context: Option<SelectionSessionContext>,
    session_id: Option<String>,
) -> Option<SessionContext> {
    session_context
        .map(SessionContext::from)
        .or_else(|| session_id.map(|session_id| SessionContext::new(session_id, None, None, None)))
}

/// Booking request: replay the selection cached under `selection_id`, or book
/// self-contained with `worker_id`. The replay books exactly what `select` captured;
/// request fields other than the ids and model/routing-group are ignored.
#[derive(Debug, Deserialize)]
pub struct ReservationRequest {
    #[serde(default = "default_model_name")]
    pub model_name: String,
    #[serde(default = "default_routing_group")]
    pub routing_group: String,
    /// The single booking id: the cache key to replay and the scheduler request
    /// id the booking lands under (the `selection_id` from the matching `select`).
    pub selection_id: String,
    /// Explicit, self-contained form: books under `selection_id` on this worker
    /// without a cached select. Omit to replay the cached `selection_id`.
    pub worker_id: Option<WorkerId>,
    pub dp_rank: Option<DpRank>,
    #[serde(flatten)]
    pub prompt: PromptRequest,
    pub router_config_override: Option<RouterConfigOverride>,
    pub expected_output_tokens: Option<u32>,
    pub effective_prefill_tokens: Option<usize>,
    pub track_prefill_tokens: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct OutputBlockRequest {
    pub decay_fraction: Option<f64>,
}

#[derive(Debug, Deserialize)]
pub struct PotentialLoadsRequest {
    #[serde(default = "default_model_name")]
    pub model_name: String,
    #[serde(default = "default_routing_group")]
    pub routing_group: String,
    #[serde(flatten)]
    pub prompt: PromptRequest,
    pub router_config_override: Option<RouterConfigOverride>,
}

#[derive(Debug, Deserialize)]
pub struct OverlapScoresRequest {
    #[serde(default = "default_model_name")]
    pub model_name: String,
    #[serde(default = "default_routing_group")]
    pub routing_group: String,
    #[serde(flatten)]
    pub prompt: PromptRequest,
    pub router_config_override: Option<RouterConfigOverride>,
}

#[derive(Debug, Serialize)]
pub struct SelectResponse {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub selection_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sequence_hashes: Option<Vec<i64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub isl_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub track_prefill_tokens: Option<bool>,
    pub model_name: String,
    pub routing_group: String,
    pub worker_id: WorkerId,
    pub dp_rank: DpRank,
    pub endpoint: String,
    pub block_size: u32,
    pub overlap: MooncakeOverlapSummary,
    pub effective_prefill_tokens: usize,
    /// Projected KV blocks on the chosen worker once this request decodes,
    /// including its own blocks: the scheduler's `potential_decode_blocks`.
    pub potential_decode_blocks: u64,
    /// `potential_decode_blocks` against the chosen worker's `total_kv_blocks`
    /// at `conditional_disagg_decode_busy_threshold`. Absent when either the
    /// threshold or the worker's capacity is unknown.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decode_busy: Option<bool>,
    /// Chosen worker's load at selection time. Present only for advisory
    /// selections (`SelectRequest::advisory`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub worker_load: Option<SelectionWorkerLoad>,
    /// Source the chosen worker can fetch a longer cached prefix from. Present
    /// only for bookings when the partition has router-hint-capable workers,
    /// the indexer can retain the matched chain, and a better source exists.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_hint: Option<KvHint>,
}

/// Load snapshot of the chosen worker, as the scheduler projected it for this
/// request. Mirrors the frontend's advisory selection load.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct SelectionWorkerLoad {
    pub active_prefill_tokens: usize,
    pub prefill_token_capacity: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_kv_blocks: Option<u64>,
    /// `active_prefill_tokens` against `prefill_token_capacity` at
    /// `conditional_disagg_prefill_busy_threshold`. Absent when the threshold
    /// is unset.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill_busy: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct ReservationResponse {
    pub selection_id: String,
    pub model_name: String,
    pub routing_group: String,
    pub worker_id: WorkerId,
    pub dp_rank: DpRank,
    pub endpoint: String,
}

#[derive(Debug, Serialize)]
pub struct ReadyResponse {
    pub ready: bool,
    pub schedulable_workers: usize,
    pub workers: Vec<WorkerCatalogRecord>,
}

#[derive(Debug, Serialize)]
pub struct ModelLoadResponse {
    pub model_name: String,
    pub routing_group: String,
    pub loads: Vec<PotentialLoad>,
    pub pending_count: usize,
    pub pending_isl_tokens: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn select_request_deserializes_structured_session_context() {
        let mut request: SelectRequest = serde_json::from_value(serde_json::json!({
            "token_ids": [1, 2, 3, 4],
            "session_id": "legacy",
            "session_context": {
                "session_id": "child",
                "parent_session_id": "root",
                "session_final": false,
                "input_trigger": "user_message"
            }
        }))
        .expect("valid select request");

        let context = request
            .take_session_context()
            .expect("structured context wins over legacy session_id");
        assert_eq!(context.session_id(), "child");
        assert_eq!(context.parent_session_id(), Some("root"));
        assert_eq!(context.session_final(), Some(false));
        assert_eq!(
            context.input_trigger(),
            Some(WorkerSelectionInputTrigger::UserMessage)
        );
    }

    #[test]
    fn select_request_falls_back_to_legacy_session_id() {
        let mut request: SelectRequest = serde_json::from_value(serde_json::json!({
            "token_ids": [1, 2, 3, 4],
            "session_id": "legacy"
        }))
        .expect("valid select request");
        let context = request.take_session_context().expect("legacy context");
        assert_eq!(context.session_id(), "legacy");
        assert_eq!(context.parent_session_id(), None);

        let mut request: SelectAndReserveRequest =
            serde_json::from_value(serde_json::json!({ "token_ids": [1, 2, 3, 4] }))
                .expect("valid reserve request");
        assert!(request.take_session_context().is_none());
    }
}
