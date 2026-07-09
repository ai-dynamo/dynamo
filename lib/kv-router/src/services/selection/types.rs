// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::protocols::{
    DpRank, KvTransferEnforcement, RoutingConstraints, WorkerConfigLike, WorkerId, WorkerWithDpRank,
};
use crate::scheduling::PotentialLoad;
use crate::scheduling::config::RouterConfigOverride;
pub use crate::scheduling::{OverlapScoresResponse, SharedCacheOverlapScore, WorkerOverlapScore};
use crate::services::indexer::registry::IndexerKey;
use crate::services::overlap::MooncakeOverlapSummary;

use super::input::PromptRequest;

const DEFAULT_MODEL_NAME: &str = "default";
const DEFAULT_ROUTING_GROUP: &str = "default";
pub(super) const WORKER_TYPE: &str = "select";
pub(super) const REQUEST_BODY_LIMIT_BYTES: usize = 8 * 1024 * 1024;

fn default_model_name() -> String {
    DEFAULT_MODEL_NAME.to_string()
}

fn default_routing_group() -> String {
    DEFAULT_ROUTING_GROUP.to_string()
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize)]
pub struct SelectionKey {
    pub model_name: String,
    pub routing_group: String,
}

impl SelectionKey {
    pub(super) fn new(model_name: impl Into<String>, routing_group: impl Into<String>) -> Self {
        Self {
            model_name: model_name.into(),
            routing_group: routing_group.into(),
        }
    }

    pub(super) fn indexer_key(&self) -> IndexerKey {
        IndexerKey {
            model_name: self.model_name.clone(),
            routing_group: self.routing_group.clone(),
        }
    }
}

impl fmt::Display for SelectionKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "model={} routing_group={}",
            self.model_name, self.routing_group
        )
    }
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
            not_schedulable_reasons: Vec::new(),
        }
    }

    pub(super) fn key(&self) -> SelectionKey {
        SelectionKey::new(self.model_name.clone(), self.routing_group.clone())
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

#[derive(Debug, Deserialize)]
pub struct WorkerRequest {
    pub worker_id: WorkerId,
    #[serde(default = "default_model_name")]
    pub model_name: String,
    #[serde(default = "default_routing_group")]
    pub routing_group: String,
    #[serde(default)]
    pub endpoint: Option<String>,
    #[serde(default)]
    pub kv_events_endpoint: Option<String>,
    #[serde(default)]
    pub kv_events_endpoints: HashMap<u32, String>,
    #[serde(default)]
    pub replay_endpoint: Option<String>,
    #[serde(default)]
    pub block_size: Option<u32>,
    #[serde(default)]
    pub data_parallel_start_rank: Option<u32>,
    #[serde(default)]
    pub data_parallel_size: Option<u32>,
    #[serde(default)]
    pub max_num_batched_tokens: Option<u64>,
    #[serde(default)]
    pub total_kv_blocks: Option<u64>,
    #[serde(default)]
    pub stable_routing_id: Option<String>,
    #[serde(default)]
    pub is_eagle: Option<bool>,
    #[serde(default)]
    pub taints: HashSet<String>,
    #[serde(default)]
    pub topology_domains: HashMap<String, String>,
    #[serde(default)]
    pub kv_transfer_domain: Option<String>,
    #[serde(default)]
    pub kv_transfer_enforcement: Option<KvTransferEnforcement>,
    #[serde(default)]
    pub kv_transfer_preferred_weight: Option<f32>,
}

#[derive(Debug, Deserialize)]
pub struct WorkerPatchRequest {
    #[serde(default)]
    pub endpoint: Option<String>,
    #[serde(default)]
    pub kv_events_endpoint: Option<String>,
    #[serde(default)]
    pub kv_events_endpoints: Option<HashMap<u32, String>>,
    #[serde(default)]
    pub replay_endpoint: Option<String>,
    #[serde(default)]
    pub block_size: Option<u32>,
    #[serde(default)]
    pub data_parallel_start_rank: Option<u32>,
    #[serde(default)]
    pub data_parallel_size: Option<u32>,
    #[serde(default)]
    pub max_num_batched_tokens: Option<u64>,
    #[serde(default)]
    pub total_kv_blocks: Option<u64>,
    #[serde(default)]
    pub stable_routing_id: Option<String>,
    #[serde(default)]
    pub is_eagle: Option<bool>,
    #[serde(default)]
    pub taints: Option<HashSet<String>>,
    #[serde(default)]
    pub topology_domains: Option<HashMap<String, String>>,
    #[serde(default)]
    pub kv_transfer_domain: Option<String>,
    #[serde(default)]
    pub kv_transfer_enforcement: Option<KvTransferEnforcement>,
    #[serde(default)]
    pub kv_transfer_preferred_weight: Option<f32>,
}

impl WorkerCatalogRecord {
    pub(super) fn apply_patch(&mut self, patch: WorkerPatchRequest) {
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
    }
}

#[derive(Debug, Deserialize)]
pub struct SelectRequest {
    #[serde(default = "default_model_name")]
    pub model_name: String,
    #[serde(default = "default_routing_group")]
    pub routing_group: String,
    #[serde(default)]
    pub selection_id: Option<String>,
    #[serde(flatten)]
    pub prompt: PromptRequest,
    #[serde(default)]
    pub router_config_override: Option<RouterConfigOverride>,
    #[serde(default)]
    pub expected_output_tokens: Option<u32>,
    #[serde(default)]
    pub priority_jump: Option<f64>,
    #[serde(default)]
    pub strict_priority: Option<u32>,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub pinned_worker: Option<WorkerWithDpRank>,
    #[serde(default)]
    pub allowed_worker_ids: Option<HashSet<WorkerId>>,
    #[serde(default)]
    pub routing_constraints: RoutingConstraints,
}

#[derive(Debug, Deserialize)]
pub struct SelectAndReserveRequest {
    #[serde(default = "default_model_name")]
    pub model_name: String,
    #[serde(default = "default_routing_group")]
    pub routing_group: String,
    #[serde(default)]
    pub selection_id: Option<String>,
    #[serde(default)]
    pub reservation_id: Option<String>,
    #[serde(flatten)]
    pub prompt: PromptRequest,
    #[serde(default)]
    pub router_config_override: Option<RouterConfigOverride>,
    #[serde(default)]
    pub expected_output_tokens: Option<u32>,
    #[serde(default)]
    pub priority_jump: Option<f64>,
    #[serde(default)]
    pub strict_priority: Option<u32>,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub pinned_worker: Option<WorkerWithDpRank>,
    #[serde(default)]
    pub allowed_worker_ids: Option<HashSet<WorkerId>>,
    #[serde(default)]
    pub routing_constraints: RoutingConstraints,
}

#[derive(Debug, Deserialize)]
pub struct ReservationRequest {
    #[serde(default = "default_model_name")]
    pub model_name: String,
    #[serde(default = "default_routing_group")]
    pub routing_group: String,
    pub reservation_id: String,
    pub worker_id: WorkerId,
    #[serde(default)]
    pub dp_rank: Option<DpRank>,
    #[serde(flatten)]
    pub prompt: PromptRequest,
    #[serde(default)]
    pub router_config_override: Option<RouterConfigOverride>,
    #[serde(default)]
    pub expected_output_tokens: Option<u32>,
    #[serde(default)]
    pub effective_prefill_tokens: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct OutputBlockRequest {
    #[serde(default)]
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
    #[serde(default)]
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
    #[serde(default)]
    pub router_config_override: Option<RouterConfigOverride>,
}

#[derive(Debug, Serialize)]
pub struct SelectResponse {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub selection_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reservation_id: Option<String>,
    pub model_name: String,
    pub routing_group: String,
    pub worker_id: WorkerId,
    pub dp_rank: DpRank,
    pub endpoint: String,
    pub block_size: u32,
    pub overlap: MooncakeOverlapSummary,
    pub effective_prefill_tokens: usize,
}

#[derive(Debug, Serialize)]
pub struct ReservationResponse {
    pub reservation_id: String,
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
