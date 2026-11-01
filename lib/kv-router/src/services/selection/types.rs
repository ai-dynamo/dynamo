// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::sync::Arc;

use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize};

use crate::identity::{RoutingPartitionId, default_routing_group};
use crate::kv_hints::{KvHint, WorkerKvTransferKeys, WorkerKvTransferKeysCache};
use crate::protocols::{
    BlockExtraInfo, DpRank, KvHintTransferWorkerMetadata, KvTransferEnforcement,
    RoutingConstraints, WorkerAffinityTarget, WorkerConfigLike, WorkerId, WorkerWithDpRank,
};
use crate::scheduling::config::RouterConfigOverride;
pub use crate::scheduling::{OverlapScoresResponse, SharedCacheOverlapScore, WorkerOverlapScore};
use crate::scheduling::{PotentialLoad, SessionContext, WorkerSelectionInputTrigger};
use crate::services::overlap::MooncakeOverlapSummary;

use super::input::{MmRoutingInfoRequest, PromptRequest};

pub const DEFAULT_MODEL_NAME: &str = "default";

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
    /// How the worker publishes KV events; a `state_agent_v2` worker is never
    /// a router-hint source.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_event_source_mode: Option<String>,
    /// Derived from the fields above on first use; published configs are
    /// replaced, never mutated.
    #[serde(skip)]
    pub kv_transfer_keys: WorkerKvTransferKeysCache,
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
        // A state-agent worker's blocks are served by its cache owner.
        let is_source = self.kv_event_source_mode.as_deref() != Some("state_agent_v2");
        Some(KvHintTransferWorkerMetadata {
            worker_type,
            source_control_endpoint: self
                .router_hint_source_control_endpoints
                .get(&dp_rank)
                .map(String::as_str)
                .filter(|endpoint| is_source && !endpoint.is_empty()),
        })
    }

    fn kv_transfer_keys(&self) -> Cow<'_, WorkerKvTransferKeys> {
        Cow::Borrowed(self.kv_transfer_keys.get_or_init(self))
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_event_source_mode: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub not_schedulable_reasons: Vec<String>,
}

impl WorkerCatalogRecord {
    pub fn new(req: WorkerRequest) -> Self {
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
            kv_event_source_mode: req.kv_event_source_mode,
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

    pub fn dp_ranks(&self) -> std::ops::Range<u32> {
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
            kv_event_source_mode: self.kv_event_source_mode.clone(),
            kv_transfer_keys: Default::default(),
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

    pub(super) fn missing_schedulable_metadata(&self, queueing_enabled: bool) -> Vec<String> {
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
            kv_event_source_mode: None,
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
    #[serde(default)]
    pub kv_event_source_mode: Option<String>,
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
    #[serde(default)]
    pub kv_event_source_mode: Option<String>,
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
        if patch.kv_event_source_mode.is_some() {
            self.kv_event_source_mode = patch.kv_event_source_mode;
        }
    }
}

/// Request bodies carry `PromptRequest`'s fields at the top level. `#[serde(flatten)]`
/// would buffer the whole body through serde's `Content` first: every token id becomes a
/// boxed `Content` value, and self-describing-only inputs (pythonize) fall back to untyped
/// integer extraction. Each request type instead deserializes through a flat wire struct
/// with the prompt fields inlined.
macro_rules! inline_prompt_wire {
    ($wire:ident => $target:ident {
        $($(#[$meta:meta])* $field:ident: $ty:ty,)*
    }) => {
        #[derive(Deserialize)]
        struct $wire {
            $($(#[$meta])* $field: $ty,)*
            token_ids: Option<Vec<u32>>,
            mm_routing_info: Option<MmRoutingInfoRequest>,
            block_mm_infos: Option<Vec<Option<BlockExtraInfo>>>,
            block_hashes: Option<Vec<i64>>,
            sequence_hashes: Option<Vec<i64>>,
            isl_tokens: Option<usize>,
            lora_name: Option<String>,
            #[serde(default, rename = "cache_salt")]
            cache_namespace: Option<String>,
            #[serde(default, deserialize_with = "strict_bool")]
            is_eagle: Option<bool>,
        }

        impl From<$wire> for $target {
            fn from(wire: $wire) -> Self {
                Self {
                    $($field: wire.$field,)*
                    prompt: PromptRequest {
                        token_ids: wire.token_ids,
                        mm_routing_info: wire.mm_routing_info,
                        block_mm_infos: wire.block_mm_infos,
                        block_hashes: wire.block_hashes,
                        sequence_hashes: wire.sequence_hashes,
                        isl_tokens: wire.isl_tokens,
                        lora_name: wire.lora_name,
                        cache_namespace: wire.cache_namespace,
                        is_eagle: wire.is_eagle,
                    },
                }
            }
        }
    };
}

/// A boolean or null, nothing else. A typed `bool` field would let pythonize coerce any
/// truthy object, so `"is_eagle": "false"` would enable Eagle hashing.
fn strict_bool<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Option<bool>, D::Error> {
    struct StrictBool;

    impl Visitor<'_> for StrictBool {
        type Value = Option<bool>;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("a boolean or null")
        }

        fn visit_bool<E: de::Error>(self, value: bool) -> Result<Self::Value, E> {
            Ok(Some(value))
        }

        fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
            Ok(None)
        }

        fn visit_none<E: de::Error>(self) -> Result<Self::Value, E> {
            Ok(None)
        }
    }

    deserializer.deserialize_any(StrictBool)
}

#[derive(Debug, Deserialize)]
#[serde(from = "SelectRequestWire")]
pub struct SelectRequest {
    pub model_name: String,
    pub routing_group: String,
    pub selection_id: Option<String>,
    /// Inlined at the top level of the body.
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
    pub routing_constraints: RoutingConstraints,
    /// Select from current scheduler state without queue admission.
    ///
    /// The response then carries the chosen worker's `worker_load` snapshot
    /// and `prefill_busy` evaluation. The request never waits in the router
    /// queue and is not subject to its admission checks, so an advisory
    /// selection can succeed where an admitted one would have been rejected.
    /// A `selection_id` still caches the booking inputs for a follow-up
    /// `create_reservation`. Ignored on `select_and_reserve`, which always
    /// books.
    pub advisory: bool,
}

inline_prompt_wire!(SelectRequestWire => SelectRequest {
    #[serde(default = "default_model_name")]
    model_name: String,
    #[serde(default = "default_routing_group")]
    routing_group: String,
    selection_id: Option<String>,
    router_config_override: Option<RouterConfigOverride>,
    expected_output_tokens: Option<u32>,
    priority_jump: Option<f64>,
    strict_priority: Option<u32>,
    session_id: Option<String>,
    session_context: Option<SelectionSessionContext>,
    affinity_target: Option<WorkerAffinityTarget>,
    pinned_worker: Option<WorkerWithDpRank>,
    allowed_worker_ids: Option<HashSet<WorkerId>>,
    #[serde(default)]
    routing_constraints: RoutingConstraints,
    #[serde(default)]
    advisory: bool,
});

impl SelectRequest {
    pub(super) fn take_session_context(&mut self) -> Option<SessionContext> {
        resolve_session_context(self.session_context.take(), self.session_id.take())
    }
}

#[derive(Debug, Deserialize)]
#[serde(from = "SelectAndReserveRequestWire")]
pub struct SelectAndReserveRequest {
    pub model_name: String,
    pub routing_group: String,
    pub selection_id: Option<String>,
    /// Inlined at the top level of the body.
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
    pub routing_constraints: RoutingConstraints,
}

inline_prompt_wire!(SelectAndReserveRequestWire => SelectAndReserveRequest {
    #[serde(default = "default_model_name")]
    model_name: String,
    #[serde(default = "default_routing_group")]
    routing_group: String,
    selection_id: Option<String>,
    router_config_override: Option<RouterConfigOverride>,
    expected_output_tokens: Option<u32>,
    priority_jump: Option<f64>,
    strict_priority: Option<u32>,
    session_id: Option<String>,
    session_context: Option<SelectionSessionContext>,
    affinity_target: Option<WorkerAffinityTarget>,
    pinned_worker: Option<WorkerWithDpRank>,
    allowed_worker_ids: Option<HashSet<WorkerId>>,
    #[serde(default)]
    routing_constraints: RoutingConstraints,
});

impl SelectAndReserveRequest {
    pub(super) fn take_session_context(&mut self) -> Option<SessionContext> {
        resolve_session_context(self.session_context.take(), self.session_id.take())
    }
}

/// Session metadata handed to worker selection.
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
    /// Opaque agent headers supplied by the caller's ingress.
    #[serde(default)]
    pub agent_headers: Option<Arc<BTreeMap<String, Vec<String>>>>,
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
            agent_headers,
        } = context;
        let session = SessionContext::new(
            session_id,
            parent_session_id,
            session_final,
            input_trigger.map(|trigger| match trigger {
                SelectionInputTrigger::UserMessage => WorkerSelectionInputTrigger::UserMessage,
                SelectionInputTrigger::ToolResult => WorkerSelectionInputTrigger::ToolResult,
                SelectionInputTrigger::Other => WorkerSelectionInputTrigger::Other,
            }),
        );
        if let Some(headers) = agent_headers {
            session.with_agent_headers(headers)
        } else {
            session
        }
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
#[serde(from = "ReservationRequestWire")]
pub struct ReservationRequest {
    pub model_name: String,
    pub routing_group: String,
    /// The single booking id: the cache key to replay and the scheduler request
    /// id the booking lands under (the `selection_id` from the matching `select`).
    pub selection_id: String,
    /// Explicit, self-contained form: books under `selection_id` on this worker
    /// without a cached select. Omit to replay the cached `selection_id`.
    pub worker_id: Option<WorkerId>,
    pub dp_rank: Option<DpRank>,
    /// Inlined at the top level of the body.
    pub prompt: PromptRequest,
    pub router_config_override: Option<RouterConfigOverride>,
    pub expected_output_tokens: Option<u32>,
    pub effective_prefill_tokens: Option<usize>,
    pub track_prefill_tokens: Option<bool>,
}

inline_prompt_wire!(ReservationRequestWire => ReservationRequest {
    #[serde(default = "default_model_name")]
    model_name: String,
    #[serde(default = "default_routing_group")]
    routing_group: String,
    selection_id: String,
    worker_id: Option<WorkerId>,
    dp_rank: Option<DpRank>,
    router_config_override: Option<RouterConfigOverride>,
    expected_output_tokens: Option<u32>,
    effective_prefill_tokens: Option<usize>,
    track_prefill_tokens: Option<bool>,
});

#[derive(Debug, Deserialize)]
pub struct OutputBlockRequest {
    pub decay_fraction: Option<f64>,
}

#[derive(Debug, Deserialize)]
#[serde(from = "PotentialLoadsRequestWire")]
pub struct PotentialLoadsRequest {
    pub model_name: String,
    pub routing_group: String,
    /// Inlined at the top level of the body.
    pub prompt: PromptRequest,
    pub router_config_override: Option<RouterConfigOverride>,
}

inline_prompt_wire!(PotentialLoadsRequestWire => PotentialLoadsRequest {
    #[serde(default = "default_model_name")]
    model_name: String,
    #[serde(default = "default_routing_group")]
    routing_group: String,
    router_config_override: Option<RouterConfigOverride>,
});

#[derive(Debug, Deserialize)]
#[serde(from = "OverlapScoresRequestWire")]
pub struct OverlapScoresRequest {
    pub model_name: String,
    pub routing_group: String,
    /// Inlined at the top level of the body.
    pub prompt: PromptRequest,
    pub router_config_override: Option<RouterConfigOverride>,
}

inline_prompt_wire!(OverlapScoresRequestWire => OverlapScoresRequest {
    #[serde(default = "default_model_name")]
    model_name: String,
    #[serde(default = "default_routing_group")]
    routing_group: String,
    router_config_override: Option<RouterConfigOverride>,
});

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
/// request.
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
        assert!(context.agent_headers().is_empty());
    }

    #[test]
    fn selection_requests_preserve_agent_headers() {
        let payload = serde_json::json!({
            "token_ids": [1, 2, 3, 4],
            "session_context": {
                "session_id": "child",
                "agent_headers": {
                    "x-claude-code-request-class": ["subagent", "future-class"],
                    "x-codex-future": ["{invalid json", ""]
                }
            }
        });
        let mut select: SelectRequest =
            serde_json::from_value(payload.clone()).expect("valid select request");
        let mut reserve: SelectAndReserveRequest =
            serde_json::from_value(payload).expect("valid reserve request");
        for context in [
            select.take_session_context(),
            reserve.take_session_context(),
        ] {
            let context = context.expect("structured session context");
            assert_eq!(
                context.agent_headers()["x-claude-code-request-class"],
                ["subagent", "future-class"]
            );
            assert_eq!(
                context.agent_headers()["x-codex-future"],
                ["{invalid json", ""]
            );
        }
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
        assert!(context.agent_headers().is_empty());

        let mut request: SelectAndReserveRequest =
            serde_json::from_value(serde_json::json!({ "token_ids": [1, 2, 3, 4] }))
                .expect("valid reserve request");
        assert!(request.take_session_context().is_none());
    }
}
