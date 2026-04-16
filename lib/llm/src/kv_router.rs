// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
    time::Instant,
};

use anyhow::Result;
use dynamo_kv_router::{
    PrefillLoadEstimator,
    config::{KvRouterConfig, RouterConfigOverride, min_initial_workers_from_env},
    indexer::KvRouterError,
    protocols::KV_EVENT_SUBJECT,
    protocols::{
        BlockExtraInfo, BlockHashOptions, DpRank, LocalBlockHash, PrefillLoadHint, RouterEvent,
        RouterRequest, RouterResponse, TokensWithHashes, WorkerId, WorkerWithDpRank,
        compute_block_hash_for_seq,
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

#[derive(Debug, Clone, Copy)]
pub(crate) struct BestMatchDetails {
    pub worker: WorkerWithDpRank,
    pub cache_hit: WorkerCacheHitEstimate,
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

    if let Some(disk_matches) = tiered_matches
        .lower_tier
        .get(&dynamo_kv_router::protocols::StorageTier::Disk)
    {
        tier_overlap_blocks.disk.extend(
            disk_matches
                .hits
                .iter()
                .map(|(worker, hits)| (*worker, *hits)),
        );
    }

    tier_overlap_blocks
}

/// Generates a dp_rank-specific endpoint name for the worker KV indexer query service.
/// Each dp_rank has its own LocalKvIndexer and query endpoint to ensure per-dp_rank monotonicity.
pub fn worker_kv_indexer_query_endpoint(dp_rank: DpRank) -> String {
    format!("worker_kv_indexer_query_dp{dp_rank}")
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

        let tiered_matches = self
            .indexer
            .find_matches_by_tier(block_hashes)
            .instrument(tracing::info_span!("kv_router.find_matches"))
            .await?;
        let tier_overlap_blocks = tier_overlap_blocks_from_tiered_matches(&tiered_matches);
        let cache_hit_estimates = self.cache_hit_estimates_from_tiered_matches(&tiered_matches);
        let overlap_scores = tiered_matches.device.overlap_scores.clone();
        let find_matches_elapsed = start.elapsed();

        let response = self
            .scheduler
            .schedule(
                context_id.map(|s| s.to_string()),
                isl_tokens,
                maybe_seq_hashes,
                overlap_scores,
                tier_overlap_blocks,
                cache_hit_estimates.effective_overlap_blocks,
                cache_hit_estimates.cached_tokens,
                router_config_override,
                update_states,
                lora_name,
                priority_jump,
                expected_output_tokens,
                pinned_worker,
                allowed_worker_ids,
            )
            .instrument(tracing::info_span!("kv_router.schedule"))
            .await?;
        let total_elapsed = start.elapsed();

        if let Some(m) = metrics::RoutingOverheadMetrics::get() {
            m.observe(
                hash_elapsed,
                seq_hash_elapsed,
                find_matches_elapsed,
                total_elapsed,
            );
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
        let overlap_blocks = (cached_tokens / self.block_size as usize) as u32;
        let prefill_load_hint =
            self.prefill_load_hint_for(isl_tokens, overlap_blocks, track_prefill_tokens);

        if let Err(e) = self
            .scheduler
            .add_request(SequenceRequest {
                request_id: request_id.clone(),
                token_sequence: maybe_seq_hashes,
                isl: isl_tokens,
                cached_tokens,
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
        overlap_blocks: u32,
        track_prefill_tokens: bool,
    ) -> Option<PrefillLoadHint> {
        if !track_prefill_tokens {
            return None;
        }

        let prefix = (overlap_blocks as usize) * (self.block_size as usize);
        let effective_isl = isl_tokens.saturating_sub(prefix);
        if effective_isl == 0 {
            return None;
        }

        let Some(estimator) = &self.prefill_load_estimator else {
            return None;
        };

        match estimator.predict_prefill_duration(1, effective_isl, prefix) {
            Ok(expected_prefill_duration) => Some(PrefillLoadHint {
                initial_effective_prefill_tokens: effective_isl,
                expected_prefill_duration: Some(expected_prefill_duration),
            }),
            Err(error) => {
                tracing::warn!(
                    effective_isl,
                    prefix,
                    "failed to predict prefill duration for direct add_request path: {error}"
                );
                None
            }
        }
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
        let overlap_scores = tiered_matches.device.overlap_scores;

        Ok(self.scheduler.get_potential_loads(
            maybe_seq_hashes,
            isl_tokens,
            overlap_scores,
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
    use dynamo_kv_router::{
        indexer::{LowerTierMatchDetails, MatchDetails},
        protocols::{OverlapScores, StorageTier},
    };
    use std::collections::HashMap;

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
}
