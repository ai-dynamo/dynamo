// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::{HashMap, HashSet},
    env,
    sync::Arc,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::Result;
use dynamo_kv_router::{
    PrefillLoadEstimator, SharedKvCache,
    config::{KvRouterConfig, RouterConfigOverride, min_initial_workers_from_env},
    indexer::KvRouterError,
    protocols::KV_EVENT_SUBJECT,
    protocols::{
        BlockExtraInfo, BlockHashOptions, DpRank, LocalBlockHash, PrefillLoadHint, RouterEvent,
        RouterRequest, RouterResponse, TokensWithHashes, WorkerConfigLike, WorkerId,
        WorkerWithDpRank, compute_block_hash_for_seq,
    },
    remote_g2_plan::{
        RemoteG2CandidateDecision, RemoteG2CandidateScore, RemoteG2CostModel,
        RemoteKvReuseDecision, RemoteKvReuseNoPlanReason, RemoteKvReuseSelectionInput,
        RemoteKvReuseSelectionStats, RemoteKvReuseSourceRoute, materialize_remote_g2_reuse_plan,
        select_remote_g2_candidate,
    },
    scheduling::TierOverlapBlocks,
};
use dynamo_runtime::{
    component::{Client, Endpoint},
    discovery::DiscoveryQuery,
    pipeline::{
        AsyncEngine, AsyncEngineContextProvider, Error, ManyOut, ResponseStream, SingleIn,
        async_trait,
    },
    protocols::EndpointId,
    protocols::annotated::Annotated,
    traits::DistributedRuntimeProvider,
};
use futures::stream;
use serde::Deserialize;
use tracing::Instrument;
use validator::Validate;

// Re-export from dynamo-kv-router crate
pub use dynamo_kv_router::approx;
pub use dynamo_kv_router::protocols;
pub use dynamo_kv_router::scheduling;
pub use dynamo_kv_router::selector;

pub mod agent_controller;
pub mod indexer;
pub mod metrics;
pub mod prefill_router;
pub mod publisher;
pub mod push_router;
pub mod scheduler;
pub mod sequence;
pub mod shared_cache;
pub mod sticky_sessions;

pub use agent_controller::AgentController;
pub use indexer::{Indexer, ServedIndexerHandle, ServedIndexerMode, ensure_served_indexer_service};
pub use prefill_router::PrefillRouter;
pub use push_router::{DirectRoutingRouter, KvPushRouter};
pub use sticky_sessions::StickySessionRouter;

use crate::{
    discovery::RuntimeConfigWatch,
    kv_router::{
        scheduler::{DefaultWorkerSelector, KvScheduler, PotentialLoad},
        sequence::{SequenceError, SequenceRequest},
    },
    local_model::runtime_config::ModelRuntimeConfig,
    protocols::common::preprocessor::PreprocessedRequest,
};

// [gluo TODO] shouldn't need to be public
// this should be discovered from the component

// for metric scraping (pull-based)
pub const KV_METRICS_ENDPOINT: &str = "load_metrics";

// for metric publishing (push-based)
pub const KV_METRICS_SUBJECT: &str = "kv_metrics";

// for inter-router comms
pub const PREFILL_SUBJECT: &str = "prefill_events";
pub const ACTIVE_SEQUENCES_SUBJECT: &str = "active_sequences_events";

// for radix tree snapshot storage
pub const RADIX_STATE_BUCKET: &str = "radix-bucket";
pub const RADIX_STATE_FILE: &str = "radix-state";

// for worker-local kvindexer query
pub const WORKER_KV_INDEXER_BUFFER_SIZE: usize = 1024; // store 1024 most recent events in worker buffer

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct WorkerCacheHitEstimate {
    pub effective_overlap_blocks: f64,
    pub cached_tokens: usize,
}

impl WorkerCacheHitEstimate {
    pub fn rounded_overlap_blocks(self) -> u32 {
        self.effective_overlap_blocks.round() as u32
    }
}

#[derive(Debug, Clone, Default)]
struct CacheHitEstimates {
    effective_overlap_blocks: HashMap<WorkerWithDpRank, f64>,
    cached_tokens: HashMap<WorkerWithDpRank, usize>,
}

const REMOTE_KV_REUSE_PLAN_TTL_MS: u64 = 30_000;
const REMOTE_G2_TRACE_ENV: &str = "DYN_REMOTE_G2_TRACE";
const SGLANG_SHARED_HICACHE_RUNTIME_KEY: &str = "sglang_shared_hicache";

#[derive(Debug, Clone, Deserialize)]
struct SglangSharedHiCacheRuntimeData {
    source_host: String,
    source_bootstrap_port: u16,
}

fn shared_hicache_source_route_from_config(
    config: &ModelRuntimeConfig,
) -> Option<SglangSharedHiCacheRuntimeData> {
    let route = match config
        .get_engine_specific::<SglangSharedHiCacheRuntimeData>(SGLANG_SHARED_HICACHE_RUNTIME_KEY)
    {
        Ok(Some(route)) => route,
        Ok(None) => return None,
        Err(error) => {
            tracing::debug!(
                runtime_key = SGLANG_SHARED_HICACHE_RUNTIME_KEY,
                %error,
                "failed to read Shared HiCache runtime metadata"
            );
            return None;
        }
    };
    if route.source_host.trim().is_empty() || route.source_bootstrap_port == 0 {
        return None;
    }
    Some(route)
}

#[derive(Debug, Clone)]
pub(crate) struct BestMatchDetails {
    pub worker: WorkerWithDpRank,
    pub cache_hit: WorkerCacheHitEstimate,
    pub remote_kv_reuse: RemoteKvReuseDecision,
}

fn unix_epoch_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis().try_into().unwrap_or(u64::MAX))
        .unwrap_or_default()
}

fn remote_g2_trace_enabled() -> bool {
    env::var(REMOTE_G2_TRACE_ENV)
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            !matches!(normalized.as_str(), "0" | "false" | "no" | "off")
        })
        .unwrap_or(false)
}

pub(crate) fn attach_remote_kv_reuse_decision(
    request: &mut PreprocessedRequest,
    decision: &RemoteKvReuseDecision,
) -> serde_json::Result<()> {
    match decision {
        RemoteKvReuseDecision::Plan {
            plan, source_route, ..
        } => {
            if source_route.is_none() {
                return request.attach_remote_kv_reuse_no_plan_reason(
                    RemoteKvReuseNoPlanReason::NoSourceBootstrapEndpoint,
                );
            }
            if request
                .attach_remote_kv_reuse_plan(plan, source_route.as_ref())
                .is_ok()
            {
                return Ok(());
            }
            request.attach_remote_kv_reuse_no_plan_reason(
                RemoteKvReuseNoPlanReason::SerializationFailed,
            )
        }
        RemoteKvReuseDecision::NoPlan { reason, .. } => {
            request.attach_remote_kv_reuse_no_plan_reason(reason.clone())
        }
    }
}

fn cache_hit_weight_for_tier(
    kv_router_config: &KvRouterConfig,
    storage_tier: dynamo_kv_router::protocols::StorageTier,
) -> f64 {
    match storage_tier {
        dynamo_kv_router::protocols::StorageTier::Device => 1.0,
        dynamo_kv_router::protocols::StorageTier::HostPinned => {
            kv_router_config.host_cache_hit_weight
        }
        dynamo_kv_router::protocols::StorageTier::Disk
        | dynamo_kv_router::protocols::StorageTier::External => {
            kv_router_config.disk_cache_hit_weight
        }
    }
}

fn cached_tokens_from_effective_overlap(block_size: u32, effective_overlap_blocks: f64) -> usize {
    (effective_overlap_blocks * block_size as f64)
        .round()
        .max(0.0) as usize
}

fn remote_g2_target_local_prefix_blocks(
    target: WorkerWithDpRank,
    tiered_matches: &indexer::TieredMatchDetails,
) -> u32 {
    let device_blocks = tiered_matches
        .device
        .overlap_scores
        .scores
        .get(&target)
        .copied()
        .unwrap_or(0);
    let host_pinned_blocks = tiered_matches
        .lower_tier
        .get(&dynamo_kv_router::protocols::StorageTier::HostPinned)
        .and_then(|matches| matches.hits.get(&target))
        .copied()
        .unwrap_or(0)
        .min(u32::MAX as usize) as u32;
    device_blocks.saturating_add(host_pinned_blocks)
}

fn remote_g2_best_local_prefix_blocks(
    workers: &HashMap<WorkerId, ModelRuntimeConfig>,
    allowed_worker_ids: Option<&HashSet<WorkerId>>,
    tiered_matches: &indexer::TieredMatchDetails,
) -> u32 {
    workers
        .iter()
        .filter(|(worker_id, _)| {
            allowed_worker_ids
                .map(|allowed| allowed.contains(*worker_id))
                .unwrap_or(true)
        })
        .flat_map(|(worker_id, config)| {
            let start = config.data_parallel_start_rank();
            let end = start + config.data_parallel_size();
            (start..end).map(move |dp_rank| WorkerWithDpRank::new(*worker_id, dp_rank))
        })
        .map(|target| remote_g2_target_local_prefix_blocks(target, tiered_matches))
        .max()
        .unwrap_or(0)
}

#[allow(clippy::too_many_arguments)]
fn remote_g2_candidates_from_tiered_matches(
    kv_router_config: &KvRouterConfig,
    router_config_override: Option<&RouterConfigOverride>,
    workers: &HashMap<WorkerId, ModelRuntimeConfig>,
    block_hashes: &[LocalBlockHash],
    block_size: u32,
    tiered_matches: &indexer::TieredMatchDetails,
    allowed_worker_ids: Option<&HashSet<WorkerId>>,
    best_local_prefix_blocks: u32,
    request_id: &str,
    x_request_id: Option<&str>,
    created_at_ms: u64,
    expires_at_ms: u64,
) -> HashMap<WorkerWithDpRank, RemoteG2CandidateScore> {
    if !kv_router_config.remote_g2_reuse_enabled {
        return HashMap::new();
    }
    let cost_model = kv_router_config.remote_g2_cost_model(router_config_override);
    let mut candidates = HashMap::new();
    for (worker_id, config) in workers.iter().filter(|(worker_id, _)| {
        allowed_worker_ids
            .map(|allowed| allowed.contains(*worker_id))
            .unwrap_or(true)
    }) {
        let data_parallel_size = config.data_parallel_size();
        let data_parallel_start_rank = config.data_parallel_start_rank();
        for dp_rank in data_parallel_start_rank..(data_parallel_start_rank + data_parallel_size) {
            let target = WorkerWithDpRank::new(*worker_id, dp_rank);
            let target_local_prefix_blocks =
                remote_g2_target_local_prefix_blocks(target, tiered_matches);
            let decision = select_remote_g2_candidate(RemoteKvReuseSelectionInput {
                request_id,
                x_request_id,
                target,
                target_local_prefix_blocks,
                best_local_prefix_blocks,
                block_hashes,
                block_size_tokens: block_size,
                tiered_matches,
                created_at_ms,
                expires_at_ms,
                cost_model: Some(cost_model),
            });
            if let RemoteG2CandidateDecision::Candidate { candidate, .. } = decision {
                candidates.insert(target, candidate);
            }
        }
    }
    candidates
}

fn cache_hit_estimates_from_tiered_matches(
    kv_router_config: &KvRouterConfig,
    block_size: u32,
    tiered_matches: &indexer::TieredMatchDetails,
) -> CacheHitEstimates {
    let mut effective_overlap_blocks = HashMap::new();

    for (worker, overlap) in &tiered_matches.device.overlap_scores.scores {
        effective_overlap_blocks.insert(*worker, *overlap as f64);
    }

    for (storage_tier, tier_matches) in &tiered_matches.lower_tier {
        let weight = cache_hit_weight_for_tier(kv_router_config, *storage_tier);
        if weight == 0.0 {
            continue;
        }

        for (worker, hits) in &tier_matches.hits {
            if *hits == 0 {
                continue;
            }
            *effective_overlap_blocks.entry(*worker).or_insert(0.0) += *hits as f64 * weight;
        }
    }

    let cached_tokens = effective_overlap_blocks
        .iter()
        .map(|(worker, overlap)| {
            (
                *worker,
                cached_tokens_from_effective_overlap(block_size, *overlap),
            )
        })
        .collect();

    CacheHitEstimates {
        effective_overlap_blocks,
        cached_tokens,
    }
}

fn cache_hit_for_worker(
    cache_hit_estimates: &CacheHitEstimates,
    worker: WorkerWithDpRank,
) -> WorkerCacheHitEstimate {
    WorkerCacheHitEstimate {
        effective_overlap_blocks: cache_hit_estimates
            .effective_overlap_blocks
            .get(&worker)
            .copied()
            .unwrap_or(0.0),
        cached_tokens: cache_hit_estimates
            .cached_tokens
            .get(&worker)
            .copied()
            .unwrap_or(0),
    }
}

fn tier_overlap_blocks_from_tiered_matches(
    tiered_matches: &indexer::TieredMatchDetails,
) -> TierOverlapBlocks {
    let mut tier_overlap_blocks = TierOverlapBlocks::default();

    if let Some(host_matches) = tiered_matches
        .lower_tier
        .get(&dynamo_kv_router::protocols::StorageTier::HostPinned)
    {
        tier_overlap_blocks.host_pinned.extend(
            host_matches
                .hits
                .iter()
                .map(|(worker, hits)| (*worker, *hits)),
        );
    }

    // Disk and External share the same weighting (see `storage_tier_weight`),
    // so accumulate both into the disk bucket.
    for tier in [
        dynamo_kv_router::protocols::StorageTier::Disk,
        dynamo_kv_router::protocols::StorageTier::External,
    ] {
        if let Some(matches) = tiered_matches.lower_tier.get(&tier) {
            for (worker, hits) in &matches.hits {
                *tier_overlap_blocks.disk.entry(*worker).or_default() += *hits;
            }
        }
    }

    tier_overlap_blocks
}

/// Generates a worker/dp_rank-specific endpoint name for the worker KV indexer query service.
/// Each dp_rank has its own LocalKvIndexer and query endpoint to ensure per-dp_rank monotonicity.
pub fn worker_kv_indexer_query_endpoint(worker_id: WorkerId, dp_rank: DpRank) -> String {
    format!("worker_kv_indexer_query_worker{worker_id}_dp{dp_rank}")
}

fn log_routing_input_hashes(
    request_id: Option<&str>,
    block_size: u32,
    tokens: &[u32],
    local_hashes: &[LocalBlockHash],
) {
    if !tracing::enabled!(tracing::Level::DEBUG) {
        return;
    }

    let local_hash_ids: Vec<u64> = local_hashes.iter().map(|hash| hash.0).collect();

    tracing::debug!(
        request_id = request_id.unwrap_or(""),
        isl_tokens = tokens.len(),
        block_size,
        num_blocks = local_hashes.len(),
        local_hashes = ?local_hash_ids,
        "[ROUTING_INPUT] request local hashes"
    );
}

// for router discovery registration
pub const KV_ROUTER_ENDPOINT: &str = "router-discovery";

/// Creates an EndpointId for the KV router in the given namespace.
pub fn router_endpoint_id(namespace: String, component: String) -> EndpointId {
    EndpointId {
        namespace,
        component,
        name: KV_ROUTER_ENDPOINT.to_string(),
    }
}

/// Creates a DiscoveryQuery for the KV router in the given namespace.
pub fn router_discovery_query(namespace: String, component: String) -> DiscoveryQuery {
    DiscoveryQuery::Endpoint {
        namespace,
        component,
        endpoint: KV_ROUTER_ENDPOINT.to_string(),
    }
}

/// A KvRouter only decides which worker you should use. It doesn't send you there.
/// TODO: Rename this to indicate it only selects a worker, it does not route.
pub struct KvRouter<Sel = DefaultWorkerSelector>
where
    Sel: dynamo_kv_router::selector::WorkerSelector<ModelRuntimeConfig>,
{
    indexer: Indexer,
    scheduler: KvScheduler<Sel>,
    workers_with_configs: RuntimeConfigWatch,
    block_size: u32,
    kv_router_config: KvRouterConfig,
    prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
    cancellation_token: tokio_util::sync::CancellationToken,
    client: Client,
    is_eagle: bool,
    _served_indexer_handle: Option<ServedIndexerHandle>,
    /// Optional external shared KV cache pool. When present, `find_best_match`
    /// queries it in parallel with the indexer and factors shared hits into scoring.
    shared_cache: Option<Box<dyn SharedKvCache>>,
}

impl<Sel> KvRouter<Sel>
where
    Sel: dynamo_kv_router::selector::WorkerSelector<ModelRuntimeConfig> + Send + Sync + 'static,
{
    #[allow(clippy::too_many_arguments)]
    pub async fn new(
        endpoint: Endpoint,
        client: Client,
        workers_with_configs: RuntimeConfigWatch,
        block_size: u32,
        selector: Sel,
        kv_router_config: Option<KvRouterConfig>,
        prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
        worker_type: &'static str,
        model_name: Option<String>,
        is_eagle: bool,
        shared_cache: Option<Box<dyn SharedKvCache>>,
    ) -> Result<Self> {
        let kv_router_config = kv_router_config.unwrap_or_default();
        kv_router_config.validate()?;
        let component = endpoint.component();
        let cancellation_token = component.drt().primary_token();
        let min_initial_workers = min_initial_workers_from_env()?;

        let indexer = Indexer::new(
            component,
            &kv_router_config,
            block_size,
            model_name.as_deref(),
        )
        .await?;

        if min_initial_workers > 0 && !kv_router_config.skip_initial_worker_wait {
            let mut startup_watch = workers_with_configs.clone();
            let _ = startup_watch
                .wait_for(|m| m.len() >= min_initial_workers)
                .await
                .map_err(|_| {
                    anyhow::anyhow!(
                        "runtime config watch closed before {} workers appeared",
                        min_initial_workers
                    )
                })?;
        }

        let scheduler = KvScheduler::start(
            component.clone(),
            block_size,
            workers_with_configs.clone(),
            selector,
            &kv_router_config,
            prefill_load_estimator.clone(),
            worker_type,
        )
        .await?;

        // Start KV event subscription if needed — skip when using a remote indexer.
        if kv_router_config.use_remote_indexer {
            tracing::info!("Skipping KV event subscription (using remote indexer)");
        } else if kv_router_config.should_subscribe_to_kv_events() {
            indexer::start_subscriber(component.clone(), &kv_router_config, indexer.clone())
                .await?;
        } else {
            tracing::info!(
                "Skipping KV event subscription (use_kv_events={}, overlap_score_weight={})",
                kv_router_config.use_kv_events,
                kv_router_config.overlap_score_weight,
            );
        }

        let served_indexer_handle = if kv_router_config.serve_indexer {
            let model_name = model_name.clone().ok_or_else(|| {
                anyhow::anyhow!("model_name is required when serve_indexer is configured")
            })?;
            Some(
                ensure_served_indexer_service(
                    component.clone(),
                    ServedIndexerMode::from_use_kv_events(kv_router_config.use_kv_events),
                    model_name,
                    indexer.clone(),
                )
                .await?,
            )
        } else {
            None
        };

        tracing::info!("KV Routing initialized");
        Ok(Self {
            indexer,
            scheduler,
            workers_with_configs,
            block_size,
            kv_router_config,
            prefill_load_estimator,
            cancellation_token,
            client,
            is_eagle,
            _served_indexer_handle: served_indexer_handle,
            shared_cache,
        })
    }

    /// Get a reference to the client used by this KvRouter
    pub fn client(&self) -> &Client {
        &self.client
    }

    pub fn indexer(&self) -> &Indexer {
        &self.indexer
    }

    pub fn kv_router_config(&self) -> &KvRouterConfig {
        &self.kv_router_config
    }

    pub fn is_eagle(&self) -> bool {
        self.is_eagle
    }

    fn shared_hicache_source_route(
        &self,
        worker_id: WorkerId,
    ) -> Option<SglangSharedHiCacheRuntimeData> {
        let workers = self.workers_with_configs.borrow();
        let config = workers.get(&worker_id)?;
        shared_hicache_source_route_from_config(config).or_else(|| {
            tracing::debug!(worker_id, "missing Shared HiCache runtime metadata");
            None
        })
    }

    fn cache_hit_estimates_from_tiered_matches(
        &self,
        tiered_matches: &indexer::TieredMatchDetails,
    ) -> CacheHitEstimates {
        cache_hit_estimates_from_tiered_matches(
            &self.kv_router_config,
            self.block_size,
            tiered_matches,
        )
    }

    fn remote_g2_transferable_candidate(
        &self,
        candidate: RemoteG2CandidateScore,
        block_hashes: &[LocalBlockHash],
        tiered_matches: &indexer::TieredMatchDetails,
        cost_model: RemoteG2CostModel,
    ) -> Option<RemoteG2CandidateScore> {
        let start = candidate.start_block_index as usize;
        let end = start.checked_add(candidate.planned_blocks as usize)?;
        let candidate_hashes = block_hashes.get(start..end)?;

        let parent_hash = if start == 0 {
            None
        } else {
            tiered_matches
                .device
                .last_matched_hashes
                .get(&candidate.source)
                .copied()
        };
        let transferable_blocks = self
            .indexer
            .chain_block_hashes_for_host_pinned(candidate.source, parent_hash, candidate_hashes)
            .len()
            .min(candidate.planned_blocks as usize) as u32;
        if transferable_blocks == 0 {
            return None;
        }
        if transferable_blocks == candidate.planned_blocks {
            return Some(candidate);
        }

        let target_local_prefix_blocks =
            remote_g2_target_local_prefix_blocks(candidate.target, tiered_matches);
        let incremental_blocks = candidate
            .start_block_index
            .saturating_add(transferable_blocks)
            .saturating_sub(target_local_prefix_blocks.max(candidate.start_block_index));
        if incremental_blocks == 0 {
            return None;
        }
        let cost_blocks = cost_model.estimated_cost_blocks(transferable_blocks);
        let score_blocks = cost_model.score_blocks(incremental_blocks, transferable_blocks);
        if score_blocks <= 0.0 {
            return None;
        }

        Some(RemoteG2CandidateScore {
            planned_blocks: transferable_blocks,
            incremental_blocks,
            cost_blocks,
            score_blocks,
            ..candidate
        })
    }

    fn remote_g2_transferable_candidates(
        &self,
        candidates: HashMap<WorkerWithDpRank, RemoteG2CandidateScore>,
        block_hashes: &[LocalBlockHash],
        tiered_matches: &indexer::TieredMatchDetails,
        cost_model: RemoteG2CostModel,
    ) -> HashMap<WorkerWithDpRank, RemoteG2CandidateScore> {
        candidates
            .into_iter()
            .filter_map(|(target, candidate)| {
                self.remote_g2_transferable_candidate(
                    candidate,
                    block_hashes,
                    tiered_matches,
                    cost_model,
                )
                .map(|candidate| (target, candidate))
            })
            .collect()
    }

    fn cache_hit_for_worker(
        &self,
        cache_hit_estimates: &CacheHitEstimates,
        worker: WorkerWithDpRank,
    ) -> WorkerCacheHitEstimate {
        cache_hit_for_worker(cache_hit_estimates, worker)
    }

    pub async fn record_routing_decision(
        &self,
        mut tokens_with_hashes: TokensWithHashes,
        worker: WorkerWithDpRank,
    ) -> Result<(), KvRouterError> {
        self.indexer
            .process_routing_decision_for_request(&mut tokens_with_hashes, worker)
            .await
    }

    /// Give these tokens, find the worker with the best weighted cache hit.
    /// Returns the full match details for the selected worker.
    ///
    /// When `pinned_worker` is Some, scheduling and queueing are constrained to
    /// that exact worker/rank.
    ///
    /// When `allowed_worker_ids` is Some, only workers in that set are considered for selection.
    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn find_best_match_details(
        &self,
        context_id: Option<&str>,
        x_request_id: Option<&str>,
        tokens: &[u32],
        block_mm_infos: Option<&[Option<BlockExtraInfo>]>,
        router_config_override: Option<&RouterConfigOverride>,
        update_states: bool,
        lora_name: Option<String>,
        priority_jump: f64,
        expected_output_tokens: Option<u32>,
        pinned_worker: Option<WorkerWithDpRank>,
        allowed_worker_ids: Option<HashSet<WorkerId>>,
    ) -> anyhow::Result<BestMatchDetails> {
        let start = Instant::now();

        if update_states && context_id.is_none() {
            anyhow::bail!("context_id must be provided when update_states is true");
        }

        let isl_tokens = tokens.len();
        let hash_options = BlockHashOptions {
            block_mm_infos,
            lora_name: lora_name.as_deref(),
            is_eagle: Some(self.is_eagle),
        };

        let block_hashes = tracing::info_span!("kv_router.compute_block_hashes")
            .in_scope(|| compute_block_hash_for_seq(tokens, self.block_size, hash_options));
        log_routing_input_hashes(context_id, self.block_size, tokens, &block_hashes);
        let hash_elapsed = start.elapsed();
        // Compute seq_hashes only if scheduler needs it for active blocks tracking
        let maybe_seq_hashes = tracing::info_span!("kv_router.compute_seq_hashes").in_scope(|| {
            self.kv_router_config.compute_seq_hashes_for_tracking(
                tokens,
                self.block_size,
                router_config_override,
                hash_options,
                Some(&block_hashes),
            )
        });
        let seq_hash_elapsed = start.elapsed();

        // Query indexer (tiered) and shared cache in parallel when shared cache is configured.
        // Time each independently so metrics can separate indexer vs shared cache latency.
        let (tiered_matches, shared_cache_hits, indexer_duration, shared_cache_duration) =
            if let Some(ref shared_cache) = self.shared_cache {
                let indexer_fut = self
                    .indexer
                    .find_matches_by_tier(block_hashes.clone())
                    .instrument(tracing::info_span!("kv_router.find_matches"));
                let shared_fut = shared_cache
                    .check_blocks(tokens, self.block_size)
                    .instrument(tracing::info_span!("kv_router.shared_cache_check"));

                let indexer_timed = async {
                    let t = Instant::now();
                    let r = indexer_fut.await;
                    (r, t.elapsed())
                };
                let shared_timed = async {
                    let t = Instant::now();
                    let r = shared_fut.await;
                    (r, t.elapsed())
                };

                let ((indexer_result, idx_dur), (shared_result, sc_dur)) =
                    tokio::join!(indexer_timed, shared_timed);
                let tiered = indexer_result?;
                // Shared cache failure is non-fatal: log warning and fall back to empty hits.
                let hits = match shared_result {
                    Ok(hits) => Some(hits),
                    Err(e) => {
                        tracing::warn!(error = %e, "Shared cache query failed, ignoring");
                        if let Some(m) = metrics::RoutingOverheadMetrics::get() {
                            m.inc_shared_cache_errors();
                        }
                        None
                    }
                };
                (tiered, hits, idx_dur, Some(sc_dur))
            } else {
                let t = Instant::now();
                let tiered = self
                    .indexer
                    .find_matches_by_tier(block_hashes.clone())
                    .instrument(tracing::info_span!("kv_router.find_matches"))
                    .await?;
                (tiered, None, t.elapsed(), None)
            };

        let tier_overlap_blocks = tier_overlap_blocks_from_tiered_matches(&tiered_matches);
        let cache_hit_estimates = self.cache_hit_estimates_from_tiered_matches(&tiered_matches);
        let find_matches_elapsed = start.elapsed();
        let created_at_ms = unix_epoch_ms();
        let expires_at_ms = created_at_ms.saturating_add(REMOTE_KV_REUSE_PLAN_TTL_MS);
        let remote_g2_cost_model = self
            .kv_router_config
            .remote_g2_cost_model(router_config_override);
        let best_local_prefix_blocks = {
            let workers = self.workers_with_configs.borrow();
            if let Some(pinned_worker) = pinned_worker {
                // Exact pins make the global best-local worker infeasible for
                // optional local-gap admission.
                remote_g2_target_local_prefix_blocks(pinned_worker, &tiered_matches)
            } else {
                remote_g2_best_local_prefix_blocks(
                    &workers,
                    allowed_worker_ids.as_ref(),
                    &tiered_matches,
                )
            }
        };
        let remote_g2_candidates = {
            let workers = self.workers_with_configs.borrow();
            remote_g2_candidates_from_tiered_matches(
                &self.kv_router_config,
                router_config_override,
                &workers,
                &block_hashes,
                self.block_size,
                &tiered_matches,
                allowed_worker_ids.as_ref(),
                best_local_prefix_blocks,
                context_id.unwrap_or_default(),
                x_request_id,
                created_at_ms,
                expires_at_ms,
            )
        };
        let remote_g2_candidates = self.remote_g2_transferable_candidates(
            remote_g2_candidates,
            &block_hashes,
            &tiered_matches,
            remote_g2_cost_model,
        );

        // Capture shared cache info for metrics before moving into schedule().
        // Clone the hits so we can compute `hits_beyond(overlap_blocks)` after
        // scheduling returns, since `overlap_blocks` isn't known until then.
        let num_blocks = isl_tokens / self.block_size as usize;
        let sc_hits_for_metrics = shared_cache_hits.clone();

        let response = self
            .scheduler
            .schedule(
                context_id.map(|s| s.to_string()),
                isl_tokens,
                maybe_seq_hashes,
                tier_overlap_blocks.clone(),
                cache_hit_estimates.effective_overlap_blocks.clone(),
                cache_hit_estimates.cached_tokens.clone(),
                router_config_override,
                update_states,
                lora_name,
                priority_jump,
                expected_output_tokens,
                pinned_worker,
                allowed_worker_ids,
                shared_cache_hits,
                remote_g2_candidates.clone(),
            )
            .instrument(tracing::info_span!("kv_router.schedule"))
            .await?;
        let selected_target_local_prefix_blocks =
            remote_g2_target_local_prefix_blocks(response.best_worker, &tiered_matches);
        let selected_remote_g2_input = RemoteKvReuseSelectionInput {
            request_id: context_id.unwrap_or_default(),
            x_request_id,
            target: response.best_worker,
            target_local_prefix_blocks: selected_target_local_prefix_blocks,
            best_local_prefix_blocks,
            block_hashes: &block_hashes,
            block_size_tokens: self.block_size,
            tiered_matches: &tiered_matches,
            created_at_ms,
            expires_at_ms,
            cost_model: Some(remote_g2_cost_model),
        };
        let mut selected_remote_g2_candidate = None;
        let mut remote_kv_reuse = if self.kv_router_config.remote_g2_reuse_enabled {
            if let Some(candidate) = remote_g2_candidates.get(&response.best_worker).copied() {
                selected_remote_g2_candidate = Some(candidate);
                materialize_remote_g2_reuse_plan(selected_remote_g2_input, candidate)
            } else {
                match select_remote_g2_candidate(selected_remote_g2_input) {
                    RemoteG2CandidateDecision::Candidate { stats, .. } => {
                        RemoteKvReuseDecision::NoPlan {
                            reason: RemoteKvReuseNoPlanReason::NoContiguousPrefix,
                            stats,
                        }
                    }
                    RemoteG2CandidateDecision::NoCandidate { reason, stats } => {
                        RemoteKvReuseDecision::NoPlan { reason, stats }
                    }
                }
            }
        } else {
            RemoteKvReuseDecision::NoPlan {
                reason: RemoteKvReuseNoPlanReason::Disabled,
                stats: RemoteKvReuseSelectionStats::default(),
            }
        };

        // Post-selection: extract the chosen source's host-pinned chain of
        // framework block hashes and populate the plan. The chain may
        // come back shorter than `planned_prefix_blocks` if eviction races
        // with our walk; in that case shrink the plan to the chain length,
        // or demote to NoPlan if the chain is empty entirely.
        if let RemoteKvReuseDecision::Plan {
            plan,
            source_route,
            stats: plan_stats,
        } = &mut remote_kv_reuse
        {
            let source = dynamo_kv_router::protocols::WorkerWithDpRank::new(
                plan.source_worker_id,
                plan.source_dp_rank,
            );
            if let Some(route) = self.shared_hicache_source_route(plan.source_worker_id) {
                *source_route = Some(RemoteKvReuseSourceRoute {
                    source_worker_id: plan.source_worker_id,
                    source_host: route.source_host,
                    source_bootstrap_port: route.source_bootstrap_port,
                });

                let start = plan.start_block_index as usize;
                let end = start + plan.planned_prefix_blocks as usize;
                // Starting parent_hash for the host-pinned chain: None when the
                // source had no device-tier matches (start == 0), otherwise the
                // device tier's last matched block_hash for this source.
                let parent_hash = if start == 0 {
                    None
                } else {
                    tiered_matches
                        .device
                        .last_matched_hashes
                        .get(&source)
                        .copied()
                };

                let chain = self.indexer.chain_block_hashes_for_host_pinned(
                    source,
                    parent_hash,
                    &block_hashes[start..end],
                );

                if chain.is_empty() {
                    let stats_copy = *plan_stats;
                    remote_kv_reuse = RemoteKvReuseDecision::NoPlan {
                        reason: RemoteKvReuseNoPlanReason::NoContiguousPrefix,
                        stats: stats_copy,
                    };
                } else {
                    if chain.len() < end - start {
                        let new_len = chain.len();
                        plan.planned_prefix_blocks = new_len as u32;
                        plan.router_block_hashes.truncate(new_len);
                    }
                    plan.engine_block_hashes = chain.into_iter().map(|h| h.0).collect();
                }
            } else {
                let stats_copy = *plan_stats;
                remote_kv_reuse = RemoteKvReuseDecision::NoPlan {
                    reason: RemoteKvReuseNoPlanReason::NoSourceBootstrapEndpoint,
                    stats: stats_copy,
                };
            }
        }
        let total_elapsed = start.elapsed();

        if remote_g2_trace_enabled() {
            let device_hits: Vec<_> = tiered_matches
                .device
                .overlap_scores
                .scores
                .iter()
                .map(|(worker, hits)| (worker.worker_id, worker.dp_rank, *hits))
                .collect();
            let host_pinned_hits: Vec<_> = tiered_matches
                .lower_tier
                .get(&dynamo_kv_router::protocols::StorageTier::HostPinned)
                .map(|matches| {
                    matches
                        .hits
                        .iter()
                        .map(|(worker, hits)| (worker.worker_id, worker.dp_rank, *hits))
                        .collect()
                })
                .unwrap_or_else(Vec::new);
            tracing::warn!(
                request_id = context_id.unwrap_or_default(),
                x_request_id = x_request_id.unwrap_or_default(),
                target_worker_id = response.best_worker.worker_id,
                target_dp_rank = response.best_worker.dp_rank,
                request_blocks = block_hashes.len(),
                effective_overlap_blocks = response.effective_overlap_blocks,
                cached_tokens = response.cached_tokens,
                ?device_hits,
                ?host_pinned_hits,
                ?remote_g2_candidates,
                decision = ?remote_kv_reuse,
                "REMOTE_G2_TRACE planner decision"
            );
        }

        if let Some(m) = metrics::RoutingOverheadMetrics::get() {
            m.observe(
                hash_elapsed,
                seq_hash_elapsed,
                indexer_duration,
                shared_cache_duration,
                find_matches_elapsed,
                total_elapsed,
            );
        }

        // Observe per-request shared cache metrics.
        if let Some(hits) = sc_hits_for_metrics
            && let Some(m) = metrics::RouterRequestMetrics::get()
        {
            if num_blocks > 0 {
                m.shared_cache_hit_rate
                    .observe(hits.total_hits as f64 / num_blocks as f64);
            }
            let beyond = hits.hits_beyond(response.effective_overlap_blocks.round() as u32);
            m.shared_cache_beyond_blocks.observe(beyond as f64);
        }

        if let Some(m) = metrics::RouterRequestMetrics::get() {
            let zero_score_candidate_count = remote_g2_candidates
                .values()
                .filter(|candidate| candidate.score_blocks <= 0.0)
                .count();
            let zero_score_incremental_blocks = remote_g2_candidates
                .values()
                .filter(|candidate| candidate.score_blocks <= 0.0)
                .map(|candidate| u64::from(candidate.incremental_blocks))
                .sum();
            m.observe_remote_g2_candidates(
                remote_g2_candidates.len(),
                zero_score_candidate_count,
                zero_score_incremental_blocks,
                selected_remote_g2_candidate.as_ref(),
            );
            m.observe_remote_g2_decision(&remote_kv_reuse, self.block_size);
        }

        #[cfg(feature = "bench")]
        tracing::info!(
            isl_tokens,
            hash_us = hash_elapsed.as_micros() as u64,
            seq_hash_us = (seq_hash_elapsed - hash_elapsed).as_micros() as u64,
            find_matches_us = (find_matches_elapsed - seq_hash_elapsed).as_micros() as u64,
            schedule_us = (total_elapsed - find_matches_elapsed).as_micros() as u64,
            total_us = total_elapsed.as_micros() as u64,
            "find_best_match completed"
        );

        Ok(BestMatchDetails {
            worker: response.best_worker,
            cache_hit: WorkerCacheHitEstimate {
                effective_overlap_blocks: response.effective_overlap_blocks,
                cached_tokens: response.cached_tokens,
            },
            remote_kv_reuse,
        })
    }

    /// Give these tokens, find the worker with the best match in its KV cache.
    /// Returns the best worker (with dp_rank) and approximate effective overlap in blocks.
    #[allow(clippy::too_many_arguments)]
    pub async fn find_best_match(
        &self,
        context_id: Option<&str>,
        tokens: &[u32],
        block_mm_infos: Option<&[Option<BlockExtraInfo>]>,
        router_config_override: Option<&RouterConfigOverride>,
        update_states: bool,
        lora_name: Option<String>,
        priority_jump: f64,
        expected_output_tokens: Option<u32>,
        allowed_worker_ids: Option<HashSet<WorkerId>>,
    ) -> anyhow::Result<(WorkerWithDpRank, u32)> {
        let result = self
            .find_best_match_details(
                context_id,
                None,
                tokens,
                block_mm_infos,
                router_config_override,
                update_states,
                lora_name,
                priority_jump,
                expected_output_tokens,
                None,
                allowed_worker_ids,
            )
            .await?;
        Ok((result.worker, result.cache_hit.rounded_overlap_blocks()))
    }

    /// Register externally-provided workers in the slot tracker.
    pub fn register_workers(&self, worker_ids: &HashSet<WorkerId>) {
        self.scheduler.register_workers(worker_ids);
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn add_request(
        &self,
        request_id: String,
        tokens: &[u32],
        block_mm_infos: Option<&[Option<BlockExtraInfo>]>,
        cached_tokens: usize,
        expected_output_tokens: Option<u32>,
        worker: WorkerWithDpRank,
        lora_name: Option<String>,
        router_config_override: Option<&RouterConfigOverride>,
    ) {
        let isl_tokens = tokens.len();
        let hash_options = BlockHashOptions {
            block_mm_infos,
            lora_name: lora_name.as_deref(),
            is_eagle: Some(self.is_eagle),
        };

        let maybe_seq_hashes = self.kv_router_config.compute_seq_hashes_for_tracking(
            tokens,
            self.block_size,
            router_config_override,
            hash_options,
            None,
        );
        let track_prefill_tokens = self
            .kv_router_config
            .track_prefill_tokens(router_config_override);
        let prefill_load_hint =
            self.prefill_load_hint_for(isl_tokens, cached_tokens, track_prefill_tokens);

        if let Err(e) = self
            .scheduler
            .add_request(SequenceRequest {
                request_id: request_id.clone(),
                token_sequence: maybe_seq_hashes,
                track_prefill_tokens,
                expected_output_tokens,
                prefill_load_hint,
                worker,
                lora_name,
            })
            .await
        {
            tracing::warn!("Failed to add request {request_id}: {e}");
        }
    }

    pub async fn mark_prefill_completed(&self, request_id: &str) -> Result<(), SequenceError> {
        self.scheduler.mark_prefill_completed(request_id).await
    }

    pub async fn free(&self, request_id: &str) -> Result<(), SequenceError> {
        self.scheduler.free(request_id).await
    }

    /// Number of requests currently parked in the scheduler queue.
    pub fn pending_count(&self) -> usize {
        self.scheduler.pending_count()
    }

    fn prefill_load_hint_for(
        &self,
        isl_tokens: usize,
        cached_tokens: usize,
        track_prefill_tokens: bool,
    ) -> Option<PrefillLoadHint> {
        if !track_prefill_tokens {
            return None;
        }

        let prefix = cached_tokens.min(isl_tokens);
        let effective_isl = isl_tokens.saturating_sub(prefix);
        if effective_isl == 0 {
            return None;
        }

        let expected_prefill_duration = match &self.prefill_load_estimator {
            Some(estimator) => match estimator.predict_prefill_duration(1, effective_isl, prefix) {
                Ok(expected_prefill_duration) => Some(expected_prefill_duration),
                Err(error) => {
                    tracing::warn!(
                        effective_isl,
                        prefix,
                        "failed to predict prefill duration for direct add_request path: {error}"
                    );
                    None
                }
            },
            None => None,
        };

        Some(PrefillLoadHint {
            initial_effective_prefill_tokens: effective_isl,
            expected_prefill_duration,
        })
    }

    /// Get the worker type for this router ("prefill" or "decode").
    /// Used for Prometheus metric labeling.
    pub fn worker_type(&self) -> &'static str {
        self.scheduler.worker_type()
    }

    /// Return the worker's unique global DP rank when it owns exactly one rank.
    pub fn unique_dp_rank_for_worker(&self, worker_id: WorkerId) -> Option<u32> {
        let configs = self.workers_with_configs.borrow();
        let config = configs.get(&worker_id)?;
        (config.data_parallel_size == 1).then_some(config.data_parallel_start_rank)
    }

    pub fn add_output_block(
        &self,
        request_id: &str,
        decay_fraction: Option<f64>,
    ) -> Result<(), SequenceError> {
        self.scheduler.add_output_block(request_id, decay_fraction)
    }

    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    /// Compute the overlap blocks for a given token sequence and worker.
    /// This queries the indexer to find the effective weighted cache hit.
    pub async fn get_overlap_blocks(
        &self,
        tokens: &[u32],
        block_mm_infos: Option<&[Option<BlockExtraInfo>]>,
        worker: WorkerWithDpRank,
        lora_name: Option<&str>,
    ) -> Result<u32, KvRouterError> {
        Ok(self
            .get_cache_hit_estimate(tokens, block_mm_infos, worker, lora_name)
            .await?
            .rounded_overlap_blocks())
    }

    pub(crate) async fn get_cache_hit_estimate(
        &self,
        tokens: &[u32],
        block_mm_infos: Option<&[Option<BlockExtraInfo>]>,
        worker: WorkerWithDpRank,
        lora_name: Option<&str>,
    ) -> Result<WorkerCacheHitEstimate, KvRouterError> {
        let block_hashes = compute_block_hash_for_seq(
            tokens,
            self.block_size,
            BlockHashOptions {
                block_mm_infos,
                lora_name,
                is_eagle: Some(self.is_eagle),
            },
        );
        let tiered_matches = self.indexer.find_matches_by_tier(block_hashes).await?;
        let cache_hit_estimates = self.cache_hit_estimates_from_tiered_matches(&tiered_matches);
        Ok(self.cache_hit_for_worker(&cache_hit_estimates, worker))
    }

    /// Get potential prefill and decode loads for all workers
    pub async fn get_potential_loads(
        &self,
        tokens: &[u32],
        router_config_override: Option<&RouterConfigOverride>,
        block_mm_infos: Option<&[Option<BlockExtraInfo>]>,
        lora_name: Option<&str>,
    ) -> Result<Vec<PotentialLoad>> {
        let isl_tokens = tokens.len();
        let hash_options = BlockHashOptions {
            block_mm_infos,
            lora_name,
            is_eagle: Some(self.is_eagle),
        };
        let block_hashes = compute_block_hash_for_seq(tokens, self.block_size, hash_options);

        let maybe_seq_hashes = self.kv_router_config.compute_seq_hashes_for_tracking(
            tokens,
            self.block_size,
            router_config_override,
            hash_options,
            Some(&block_hashes),
        );
        let track_prefill_tokens = self
            .kv_router_config
            .track_prefill_tokens(router_config_override);
        let tiered_matches = self.indexer.find_matches_by_tier(block_hashes).await?;
        let cache_hit_estimates = self.cache_hit_estimates_from_tiered_matches(&tiered_matches);

        Ok(self.scheduler.get_potential_loads(
            maybe_seq_hashes,
            isl_tokens,
            cache_hit_estimates.cached_tokens,
            track_prefill_tokens,
        ))
    }

    /// Dump all events from the indexer
    pub async fn dump_events(&self) -> Result<Vec<RouterEvent>, KvRouterError> {
        self.indexer.dump_events().await
    }
}

// NOTE: KVRouter works like a PushRouter,
// but without the reverse proxy functionality, but based on contract of 3 request types
#[async_trait]
impl<Sel> AsyncEngine<SingleIn<RouterRequest>, ManyOut<Annotated<RouterResponse>>, Error>
    for KvRouter<Sel>
where
    Sel: dynamo_kv_router::selector::WorkerSelector<ModelRuntimeConfig> + Send + Sync + 'static,
{
    async fn generate(
        &self,
        request: SingleIn<RouterRequest>,
    ) -> Result<ManyOut<Annotated<RouterResponse>>> {
        let (request, ctx) = request.into_parts();
        let context_id = ctx.context().id().to_string();
        // Handle different request types
        let response = match request {
            RouterRequest::New {
                tokens,
                block_mm_infos,
            } => {
                let (best_worker, overlap_blocks) = self
                    .find_best_match(
                        Some(&context_id),
                        &tokens,
                        block_mm_infos.as_deref(),
                        None,
                        true,
                        None,
                        0.0,
                        None,
                        None,
                    )
                    .await?;

                RouterResponse::New {
                    worker_id: best_worker.worker_id,
                    dp_rank: best_worker.dp_rank,
                    overlap_blocks,
                }
            }
            RouterRequest::MarkPrefill => RouterResponse::PrefillMarked {
                success: self.mark_prefill_completed(&context_id).await.is_ok(),
            },
            RouterRequest::MarkFree { request_id } => {
                let request_id = match request_id.as_deref() {
                    Some(request_id) if !request_id.trim().is_empty() => request_id,
                    _ => &context_id,
                };
                RouterResponse::FreeMarked {
                    success: self.free(request_id).await.is_ok(),
                }
            }
        };

        let response = Annotated::from_data(response);
        let stream = stream::iter(vec![response]);
        Ok(ResponseStream::new(Box::pin(stream), ctx.context()))
    }
}

impl<Sel> Drop for KvRouter<Sel>
where
    Sel: dynamo_kv_router::selector::WorkerSelector<ModelRuntimeConfig>,
{
    fn drop(&mut self) {
        tracing::info!("Dropping KvRouter - cancelling background tasks");
        self.cancellation_token.cancel();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, HashSet};

    use async_trait::async_trait;
    use dynamo_kv_router::{
        indexer::{LowerTierMatchDetails, MatchDetails},
        protocols::{LocalBlockHash, OverlapScores, StorageTier},
        remote_g2_plan::{
            REMOTE_KV_REUSE_NO_PLAN_REASON_EXTRA_ARGS_KEY, REMOTE_KV_REUSE_PLAN_EXTRA_ARGS_KEY,
            REMOTE_KV_REUSE_PLAN_VERSION, REMOTE_KV_REUSE_SOURCE_ROUTE_EXTRA_ARGS_KEY,
            RemoteKvReuseNoPlanReason, RemoteKvReusePlan, RemoteKvReuseSelectionStats,
            RemoteKvReuseSourceRoute,
        },
    };
    use dynamo_runtime::{DistributedRuntime, Runtime, distributed::DistributedConfig};
    use tokio::sync::watch;

    use crate::kv_router::scheduler::KvSchedulerError;
    use crate::local_model::runtime_config::ModelRuntimeConfig;

    #[test]
    fn shared_hicache_source_route_uses_runtime_config() {
        let mut config = ModelRuntimeConfig::new();
        config
            .set_engine_specific(
                SGLANG_SHARED_HICACHE_RUNTIME_KEY,
                serde_json::json!({
                    "source_host": "10.0.0.7",
                    "source_bootstrap_port": 41000,
                }),
            )
            .unwrap();

        let route = shared_hicache_source_route_from_config(&config).unwrap();
        assert_eq!(route.source_host, "10.0.0.7");
        assert_eq!(route.source_bootstrap_port, 41000);
    }

    #[test]
    fn weighted_cache_hit_estimates_include_lower_tiers() {
        let worker_1 = WorkerWithDpRank::new(1, 0);
        let worker_2 = WorkerWithDpRank::new(2, 0);
        let mut device_overlap_scores = OverlapScores::new();
        device_overlap_scores.scores.insert(worker_1, 2);
        let mut host_match_details = LowerTierMatchDetails::default();
        host_match_details.hits.insert(worker_1, 1);
        host_match_details.hits.insert(worker_2, 1);
        let mut disk_match_details = LowerTierMatchDetails::default();
        disk_match_details.hits.insert(worker_1, 2);

        let tiered_matches = indexer::TieredMatchDetails {
            device: MatchDetails {
                overlap_scores: device_overlap_scores,
                ..Default::default()
            },
            lower_tier: HashMap::from([
                (StorageTier::HostPinned, host_match_details),
                (StorageTier::Disk, disk_match_details),
            ]),
        };

        let estimates = cache_hit_estimates_from_tiered_matches(
            &KvRouterConfig::default(),
            16,
            &tiered_matches,
        );

        assert_eq!(
            estimates.effective_overlap_blocks.get(&worker_1),
            Some(&3.25)
        );
        assert_eq!(estimates.cached_tokens.get(&worker_1), Some(&52));
        assert_eq!(
            estimates.effective_overlap_blocks.get(&worker_2),
            Some(&0.75)
        );
        assert_eq!(estimates.cached_tokens.get(&worker_2), Some(&12));
    }

    #[test]
    fn remote_g2_candidates_skip_suffix_already_covered_by_target_host_pinned() {
        let source = WorkerWithDpRank::new(1, 0);
        let target = WorkerWithDpRank::new(2, 0);
        let mut device_overlap_scores = OverlapScores::new();
        device_overlap_scores.scores.insert(source, 1537);
        device_overlap_scores.scores.insert(target, 1499);
        let mut host_match_details = LowerTierMatchDetails::default();
        host_match_details.hits.insert(source, 2490);
        host_match_details.hits.insert(target, 2528);

        let tiered_matches = indexer::TieredMatchDetails {
            device: MatchDetails {
                overlap_scores: device_overlap_scores,
                ..Default::default()
            },
            lower_tier: HashMap::from([(StorageTier::HostPinned, host_match_details)]),
        };
        let mut config = KvRouterConfig {
            remote_g2_reuse_enabled: true,
            shared_cache_multiplier: 0.5,
            remote_g2_cost_blocks: 16.0,
            remote_g2_cost_per_block: 0.02,
            ..Default::default()
        };
        config.host_cache_hit_weight = 0.75;
        let cache_hit_estimates =
            cache_hit_estimates_from_tiered_matches(&config, 16, &tiered_matches);
        assert_eq!(
            cache_hit_estimates.effective_overlap_blocks.get(&target),
            Some(&3395.0)
        );
        assert_eq!(
            remote_g2_target_local_prefix_blocks(target, &tiered_matches),
            4027
        );

        let workers = HashMap::from([
            (source.worker_id, ModelRuntimeConfig::new()),
            (target.worker_id, ModelRuntimeConfig::new()),
        ]);
        let allowed_worker_ids = HashSet::from([target.worker_id]);
        let block_hashes = (0..4168).map(LocalBlockHash).collect::<Vec<_>>();
        let candidates = remote_g2_candidates_from_tiered_matches(
            &config,
            None,
            &workers,
            &block_hashes,
            16,
            &tiered_matches,
            Some(&allowed_worker_ids),
            4027,
            "request-1",
            None,
            1000,
            2000,
        );

        assert!(
            !candidates.contains_key(&target),
            "remote G2 must not select a suffix already covered by target HostPinned"
        );
    }

    struct FakeSharedCache {
        hits: Option<dynamo_kv_router::protocols::SharedCacheHits>,
        should_error: bool,
    }

    #[async_trait]
    impl SharedKvCache for FakeSharedCache {
        async fn check_blocks(
            &self,
            _tokens: &[u32],
            _block_size: u32,
        ) -> Result<dynamo_kv_router::protocols::SharedCacheHits, KvRouterError> {
            if self.should_error {
                Err(KvRouterError::IndexerOffline)
            } else {
                Ok(self.hits.clone().unwrap_or_default())
            }
        }
    }

    struct InspectingSelector {
        expected_hits: Option<u32>,
        selected_worker: WorkerWithDpRank,
    }

    impl dynamo_kv_router::selector::WorkerSelector<ModelRuntimeConfig> for InspectingSelector {
        fn select_worker(
            &self,
            _workers: &HashMap<WorkerId, ModelRuntimeConfig>,
            request: &dynamo_kv_router::scheduling::SchedulingRequest,
            block_size: u32,
        ) -> Result<dynamo_kv_router::protocols::WorkerSelectionResult, KvSchedulerError> {
            let observed_hits = request
                .shared_cache_hits
                .as_ref()
                .map(|hits| hits.total_hits);
            assert_eq!(observed_hits, self.expected_hits);

            Ok(dynamo_kv_router::protocols::WorkerSelectionResult {
                worker: self.selected_worker,
                required_blocks: request.isl_tokens.div_ceil(block_size as usize) as u64,
                effective_overlap_blocks: 0.0,
                cached_tokens: 0,
            })
        }
    }

    async fn make_test_component(name: &str) -> dynamo_runtime::component::Component {
        let runtime = Runtime::from_current().unwrap();
        let drt = DistributedRuntime::new(runtime, DistributedConfig::process_local())
            .await
            .unwrap();
        let namespace = drt.namespace(format!("test-ns-{name}")).unwrap();
        namespace
            .component(format!("test-component-{name}"))
            .unwrap()
    }

    async fn make_test_router(
        selector: impl dynamo_kv_router::selector::WorkerSelector<ModelRuntimeConfig>
        + Send
        + Sync
        + 'static,
        shared_cache: Option<Box<dyn SharedKvCache>>,
    ) -> KvRouter<
        impl dynamo_kv_router::selector::WorkerSelector<ModelRuntimeConfig> + Send + Sync + 'static,
    > {
        let component = make_test_component("shared-cache-router").await;
        let endpoint = component.endpoint("backend");
        let client = endpoint.client().await.unwrap();

        let mut workers = HashMap::new();
        workers.insert(0, ModelRuntimeConfig::default());
        workers.insert(1, ModelRuntimeConfig::default());
        let (_tx, rx) = watch::channel(workers);

        let config = KvRouterConfig {
            overlap_score_weight: 0.0,
            router_temperature: 0.0,
            use_kv_events: false,
            router_track_active_blocks: false,
            shared_cache_multiplier: 0.5,
            skip_initial_worker_wait: true,
            ..Default::default()
        };

        KvRouter::new(
            endpoint,
            client,
            rx,
            2,
            selector,
            Some(config),
            None,
            "decode",
            None,
            false,
            shared_cache,
        )
        .await
        .unwrap()
    }

    #[tokio::test]
    async fn test_find_best_match_passes_shared_cache_hits_to_scheduler() {
        let router = make_test_router(
            InspectingSelector {
                expected_hits: Some(2),
                selected_worker: WorkerWithDpRank::from_worker_id(1),
            },
            Some(Box::new(FakeSharedCache {
                #[allow(clippy::single_range_in_vec_init)]
                hits: Some(dynamo_kv_router::protocols::SharedCacheHits::from_ranges(
                    vec![0..2],
                )),
                should_error: false,
            })),
        )
        .await;

        let (worker, overlap) = router
            .find_best_match(
                None,
                &[11, 12, 21, 22],
                None,
                None,
                false,
                None,
                0.0,
                None,
                None,
            )
            .await
            .unwrap();

        assert_eq!(worker, WorkerWithDpRank::from_worker_id(1));
        assert_eq!(overlap, 0);
    }

    #[tokio::test]
    async fn test_find_best_match_ignores_shared_cache_errors() {
        let router = make_test_router(
            InspectingSelector {
                expected_hits: None,
                selected_worker: WorkerWithDpRank::from_worker_id(0),
            },
            Some(Box::new(FakeSharedCache {
                hits: None,
                should_error: true,
            })),
        )
        .await;

        let (worker, overlap) = router
            .find_best_match(
                None,
                &[11, 12, 21, 22],
                None,
                None,
                false,
                None,
                0.0,
                None,
                None,
            )
            .await
            .unwrap();

        assert_eq!(worker, WorkerWithDpRank::from_worker_id(0));
        assert_eq!(overlap, 0);
    }

    fn remote_g2_test_request() -> PreprocessedRequest {
        PreprocessedRequest::builder()
            .model("test".to_string())
            .token_ids(vec![1, 2, 3, 4])
            .stop_conditions(Default::default())
            .sampling_options(Default::default())
            .output_options(Default::default())
            .build()
            .unwrap()
    }

    fn remote_g2_test_plan() -> RemoteKvReusePlan {
        RemoteKvReusePlan {
            plan_id: "plan-1".to_string(),
            request_id: "request-1".to_string(),
            x_request_id: Some("client-request-1".to_string()),
            target_worker_id: 42,
            target_dp_rank: 2,
            source_worker_id: 7,
            source_dp_rank: 0,
            source_tier: StorageTier::HostPinned,
            router_block_hashes: vec![LocalBlockHash(11), LocalBlockHash(22)],
            start_block_index: 0,
            planned_prefix_blocks: 2,
            block_size_tokens: 16,
            created_at_ms: 1000,
            expires_at_ms: 2000,
            plan_version: REMOTE_KV_REUSE_PLAN_VERSION,
            engine_block_hashes: vec![],
        }
    }

    fn remote_g2_test_source_route() -> RemoteKvReuseSourceRoute {
        RemoteKvReuseSourceRoute {
            source_worker_id: 7,
            source_host: "10.0.0.7".to_string(),
            source_bootstrap_port: 41000,
        }
    }

    #[test]
    fn router_attaches_remote_g2_plan_after_target_selection() {
        let mut request = remote_g2_test_request();
        let decision = RemoteKvReuseDecision::Plan {
            plan: remote_g2_test_plan(),
            source_route: Some(remote_g2_test_source_route()),
            stats: RemoteKvReuseSelectionStats {
                rejected_g1_candidates: 1,
            },
        };

        attach_remote_kv_reuse_decision(&mut request, &decision).unwrap();

        let extra_args = request.extra_args.unwrap();
        let plan = &extra_args[REMOTE_KV_REUSE_PLAN_EXTRA_ARGS_KEY];
        assert_eq!(plan["target_worker_id"], 42);
        assert_eq!(plan["target_dp_rank"], 2);
        assert_eq!(plan["source_worker_id"], 7);
        assert_eq!(plan["source_dp_rank"], 0);
        assert_eq!(plan["source_tier"], "host_pinned");
        assert!(plan.get("source_host").is_none());
        let source_route = &extra_args[REMOTE_KV_REUSE_SOURCE_ROUTE_EXTRA_ARGS_KEY];
        assert_eq!(source_route["source_worker_id"], 7);
        assert_eq!(source_route["source_host"], "10.0.0.7");
        assert_eq!(source_route["source_bootstrap_port"], 41000);
    }

    #[test]
    fn router_no_plan_has_reason_without_forbidden_fields() {
        let mut request = remote_g2_test_request();
        let decision = RemoteKvReuseDecision::NoPlan {
            reason: RemoteKvReuseNoPlanReason::NoRemoteG2Candidate,
            stats: RemoteKvReuseSelectionStats {
                rejected_g1_candidates: 1,
            },
        };

        attach_remote_kv_reuse_decision(&mut request, &decision).unwrap();

        let extra_args = request.extra_args.unwrap();
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
