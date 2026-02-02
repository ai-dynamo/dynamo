// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use derive_builder::Builder;
use dynamo_runtime::{
    component::{Client, Endpoint},
    discovery::{DiscoveryQuery, EventTransportKind},
    pipeline::{
        AsyncEngine, AsyncEngineContextProvider, Error, ManyOut, PushRouter, ResponseStream,
        SingleIn, async_trait,
    },
    protocols::EndpointId,
    protocols::annotated::Annotated,
    traits::DistributedRuntimeProvider,
};
use futures::stream::{self, StreamExt};
use rand::Rng;
use serde::{Deserialize, Serialize};
use serde_json::json;

// Re-export from dynamo-kv-router crate
pub use dynamo_kv_router::approx;
pub use dynamo_kv_router::indexer;
pub use dynamo_kv_router::protocols;

pub mod prefill_router;
pub mod publisher;
pub mod recorder;
pub mod scheduler;
pub mod sequence;
pub mod subscriber;
pub mod worker_query;

use indexer::WorkerKvQueryResponse;
pub use prefill_router::PrefillRouter;
use worker_query::WorkerQueryClient;

use crate::{
    discovery::RuntimeConfigs,
    kv_router::{
        approx::PruneConfig,
        indexer::{KvIndexer, KvIndexerInterface, KvRouterError},
        protocols::{
            LocalBlockHash, OverlapScores, RouterEvent, RouterRequest, RouterResponse,
            TokensWithHashes, WorkerId, WorkerSelectionResult, WorkerWithDpRank,
            compute_block_hash_for_seq, compute_seq_hash_for_block,
        },
        scheduler::{KvScheduler, KvSchedulerError, PotentialLoad, SchedulingRequest},
        sequence::SequenceError,
        subscriber::{start_kv_router_background, start_kv_router_background_event_plane},
    },
    local_model::runtime_config::ModelRuntimeConfig,
    preprocessor::PreprocessedRequest,
    protocols::common::llm_backend::LLMEngineOutput,
    protocols::common::timing::RequestPhase,
};

// [gluo TODO] shouldn't need to be public
// this should be discovered from the component

// for metric scraping (pull-based)
pub const KV_METRICS_ENDPOINT: &str = "load_metrics";

// for metric publishing (push-based)
pub const KV_EVENT_SUBJECT: &str = "kv-events";
pub const KV_HIT_RATE_SUBJECT: &str = "kv-hit-rate";
pub const KV_METRICS_SUBJECT: &str = "kv_metrics";

// for inter-router comms
pub const PREFILL_SUBJECT: &str = "prefill_events";
pub const ACTIVE_SEQUENCES_SUBJECT: &str = "active_sequences_events";

// for radix tree snapshot storage
pub const RADIX_STATE_BUCKET: &str = "radix-bucket";
pub const RADIX_STATE_FILE: &str = "radix-state";

// for worker-local kvindexer query
pub const WORKER_KV_INDEXER_QUERY_ENDPOINT: &str = "worker_kv_indexer_query";
pub const WORKER_KV_INDEXER_BUFFER_SIZE: usize = 1024; // store 1024 most recent events in worker buffer

// for router discovery registration
pub const KV_ROUTER_COMPONENT: &str = "kv-router";
pub const KV_ROUTER_ENDPOINT: &str = "generate";

/// Creates an EndpointId for the KV router in the given namespace.
pub fn router_endpoint_id(namespace: String) -> EndpointId {
    EndpointId {
        namespace,
        component: KV_ROUTER_COMPONENT.to_string(),
        name: KV_ROUTER_ENDPOINT.to_string(),
    }
}

/// Creates a DiscoveryQuery for the KV router in the given namespace.
pub fn router_discovery_query(namespace: String) -> DiscoveryQuery {
    DiscoveryQuery::Endpoint {
        namespace,
        component: KV_ROUTER_COMPONENT.to_string(),
        endpoint: KV_ROUTER_ENDPOINT.to_string(),
    }
}

/// A trait that users can implement to define custom selection logic
pub trait WorkerSelector {
    fn select_worker(
        &self,
        workers: &HashMap<protocols::WorkerId, Option<ModelRuntimeConfig>>,
        request: &SchedulingRequest,
        block_size: u32,
    ) -> Result<WorkerSelectionResult, KvSchedulerError>;
}

/// Override configuration for router settings that can be specified per-request
#[derive(Debug, Clone, Default, Builder, Serialize, Deserialize)]
pub struct RouterConfigOverride {
    #[builder(default)]
    pub overlap_score_weight: Option<f64>,

    #[builder(default)]
    pub router_temperature: Option<f64>,
}

/// KV Router configuration parameters
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct KvRouterConfig {
    pub overlap_score_weight: f64,

    pub router_temperature: f64,

    pub use_kv_events: bool,

    pub router_replica_sync: bool,

    /// Whether to track active blocks in the router (default: true)
    pub router_track_active_blocks: bool,

    /// Whether to track output blocks during generation (default: false)
    /// When enabled, the router adds placeholder blocks as tokens are generated
    /// and applies fractional decay based on progress toward expected_output_tokens.
    pub router_track_output_blocks: bool,

    /// Whether to assume KV cache reuse when tracking active blocks (default: true).
    /// When true, computes actual block hashes for sequence tracking.
    /// When false, generates random hashes (assuming no KV cache reuse).
    pub router_assume_kv_reuse: bool,

    /// Threshold for triggering snapshots. If None, no snapshots will be performed.
    pub router_snapshot_threshold: Option<u32>,

    /// Whether to reset the router state on startup (default: false)
    pub router_reset_states: bool,

    /// TTL for blocks in seconds (only used when use_kv_events is false, default: 120.0)
    pub router_ttl_secs: f64,

    /// Maximum tree size before pruning (only used when use_kv_events is false, default: 2^20 = 1048576)
    pub router_max_tree_size: usize,

    /// Target size ratio after pruning (only used when use_kv_events is false, default: 0.8)
    pub router_prune_target_ratio: f64,
}

impl Default for KvRouterConfig {
    fn default() -> Self {
        Self {
            overlap_score_weight: 1.0,
            router_temperature: 0.0,
            use_kv_events: true,
            router_replica_sync: false,
            router_track_active_blocks: true,
            router_track_output_blocks: false,
            router_assume_kv_reuse: true,
            router_snapshot_threshold: Some(1000000),
            router_reset_states: false,
            router_ttl_secs: 120.0,
            router_max_tree_size: 2usize.pow(20), // 2^20 = 1048576, matches PruneConfig::default()
            router_prune_target_ratio: 0.8,
        }
    }
}

impl KvRouterConfig {
    /// Create a new KvRouterConfig with optional weight values.
    /// If a weight is None, the default value will be used.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        overlap_score_weight: Option<f64>,
        temperature: Option<f64>,
        use_kv_events: Option<bool>,
        replica_sync: Option<bool>,
        track_active_blocks: Option<bool>,
        track_output_blocks: Option<bool>,
        assume_kv_reuse: Option<bool>,
        router_snapshot_threshold: Option<Option<u32>>,
        router_reset_states: Option<bool>,
        router_ttl_secs: Option<f64>,
        router_max_tree_size: Option<usize>,
        router_prune_target_ratio: Option<f64>,
    ) -> Self {
        let default = Self::default();
        Self {
            overlap_score_weight: overlap_score_weight.unwrap_or(default.overlap_score_weight),
            router_temperature: temperature.unwrap_or(default.router_temperature),
            use_kv_events: use_kv_events.unwrap_or(default.use_kv_events),
            router_replica_sync: replica_sync.unwrap_or(default.router_replica_sync),
            router_track_active_blocks: track_active_blocks
                .unwrap_or(default.router_track_active_blocks),
            router_track_output_blocks: track_output_blocks
                .unwrap_or(default.router_track_output_blocks),
            router_assume_kv_reuse: assume_kv_reuse.unwrap_or(default.router_assume_kv_reuse),
            router_snapshot_threshold: router_snapshot_threshold
                .unwrap_or(default.router_snapshot_threshold),
            router_reset_states: router_reset_states.unwrap_or(default.router_reset_states),
            router_ttl_secs: router_ttl_secs.unwrap_or(default.router_ttl_secs),
            router_max_tree_size: router_max_tree_size.unwrap_or(default.router_max_tree_size),
            router_prune_target_ratio: router_prune_target_ratio
                .unwrap_or(default.router_prune_target_ratio),
        }
    }

    /// Compute sequence hashes for active block tracking based on configuration.
    ///
    /// Returns:
    /// - `None` if `router_track_active_blocks` is false
    /// - Random hashes if `router_track_active_blocks` is true but `router_assume_kv_reuse` is false
    /// - Actual sequence hashes if both are true
    pub fn compute_seq_hashes_for_tracking(
        &self,
        tokens: &[u32],
        block_size: u32,
    ) -> Option<Vec<u64>> {
        if !self.router_track_active_blocks {
            return None;
        }

        let num_blocks = tokens.len() / block_size as usize;
        if num_blocks == 0 {
            return Some(Vec::new());
        }

        if self.router_assume_kv_reuse {
            // Compute actual block hashes and sequence hashes
            let block_hashes = compute_block_hash_for_seq(tokens, block_size, None);
            Some(compute_seq_hash_for_block(&block_hashes))
        } else {
            // Generate random hashes (no KV reuse assumed)
            let mut rng = rand::rng();
            Some((0..num_blocks).map(|_| rng.random::<u64>()).collect())
        }
    }
}

pub enum Indexer {
    /// Updates itself based on KV events emitted by backend workers or routing decisions.
    /// Supports TTL-based expiration and size-based pruning.
    /// Has the ability to persist and snapshot states.
    KvIndexer(KvIndexer),

    /// Used when we do not wish to use the indexer at all (e.g., when overlap_score_weight is 0).
    /// Note: This will cause KV events to accumulate in JetStream as we do not regularly purge them.
    None,
}

impl Indexer {
    async fn find_matches(
        &self,
        sequence: Vec<LocalBlockHash>,
    ) -> Result<OverlapScores, KvRouterError> {
        match self {
            Indexer::KvIndexer(indexer) => indexer.find_matches(sequence).await,
            Indexer::None => Ok(OverlapScores {
                scores: HashMap::new(),
                frequencies: Vec::new(),
                tree_sizes: HashMap::new(),
            }),
        }
    }

    async fn dump_events(&self) -> Result<Vec<RouterEvent>, KvRouterError> {
        match self {
            Indexer::KvIndexer(indexer) => indexer.dump_events().await,
            Indexer::None => {
                panic!(
                    "Cannot dump events: indexer does not exist (is overlap_score_weight set to 0?)"
                );
            }
        }
    }

    async fn process_routing_decision_for_request(
        &self,
        tokens_with_hashes: &mut TokensWithHashes,
        worker: WorkerWithDpRank,
    ) -> Result<(), KvRouterError> {
        match self {
            Indexer::KvIndexer(indexer) => {
                indexer
                    .process_routing_decision_for_request(tokens_with_hashes, worker)
                    .await
            }
            Indexer::None => Ok(()),
        }
    }
}

/// A KvRouter only decides which worker you should use. It doesn't send you there.
/// TODO: Rename this to indicate it only selects a worker, it does not route.
pub struct KvRouter {
    indexer: Indexer,

    // How about a Box<dyn KvIndexerInterface>
    scheduler: KvScheduler,

    block_size: u32,

    kv_router_config: KvRouterConfig,

    cancellation_token: tokio_util::sync::CancellationToken,

    client: Client,

    worker_query_client: Option<WorkerQueryClient>,
}

impl KvRouter {
    pub async fn new(
        endpoint: Endpoint,
        client: Client,
        workers_with_configs: Arc<RuntimeConfigs>,
        block_size: u32,
        selector: Option<Box<dyn WorkerSelector + Send + Sync>>,
        kv_router_config: Option<KvRouterConfig>,
        router_id: u64,
    ) -> Result<Self> {
        let kv_router_config = kv_router_config.unwrap_or_default();
        let component = endpoint.component();
        let cancellation_token = component.drt().primary_token();

        let indexer = if kv_router_config.overlap_score_weight == 0.0 {
            // When overlap_score_weight is zero, we don't need to track prefixes
            Indexer::None
        } else {
            let kv_indexer_metrics = indexer::KvIndexerMetrics::from_component(component);

            // If use_kv_events is false, enable TTL and pruning for approximate behavior
            let prune_config = if !kv_router_config.use_kv_events {
                Some(PruneConfig {
                    ttl: Duration::from_secs_f64(kv_router_config.router_ttl_secs),
                    max_tree_size: kv_router_config.router_max_tree_size,
                    prune_target_ratio: kv_router_config.router_prune_target_ratio,
                })
            } else {
                None
            };

            Indexer::KvIndexer(KvIndexer::new_with_frequency(
                cancellation_token.clone(),
                None, // expiration_duration for frequency tracking
                block_size,
                kv_indexer_metrics,
                prune_config,
            ))
        };

        // Wait for at least one worker with a known runtime config before starting scheduler
        workers_with_configs.subscribe().wait_for_some().await;

        let scheduler = KvScheduler::start(
            component.clone(),
            block_size,
            workers_with_configs.clone(),
            selector,
            kv_router_config.router_replica_sync,
            router_id,
        )
        .await?;

        // Initialize worker query client using namespace abstraction
        // (created before background task so we can use it for startup recovery)
        // Uses a subscriber from workers_with_configs
        let worker_query_client = worker_query::WorkerQueryClient::new(
            component.clone(),
            workers_with_configs.subscribe(),
        );
        tracing::info!("Worker query client initialized");

        // Start KV event subscriber background process (only when use_kv_events is enabled)
        if kv_router_config.use_kv_events
            && let Indexer::KvIndexer(ref kv_indexer) = indexer
        {
            let all_local_indexer = workers_with_configs
                .configs
                .iter()
                .filter_map(|r| r.value().as_ref().map(|c| c.enable_local_indexer))
                .all(|b| b);

            tracing::info!(
                "Found {} worker(s), starting KV event subscriber",
                workers_with_configs.num_workers()
            );

            let transport_kind = EventTransportKind::from_env_or_default();

            // Start subscriber - setup runs synchronously, then spawns background loop internally
            if all_local_indexer {
                if transport_kind == EventTransportKind::Zmq {
                    if kv_router_config.router_snapshot_threshold.is_some()
                        || kv_router_config.router_reset_states
                    {
                        tracing::warn!(
                            "ZMQ event plane does not support KV snapshots or state reset; ignoring snapshot/reset settings"
                        );
                    }
                } else {
                    tracing::info!(
                        "All {} workers have local_indexer enabled, using NATS Core subscription",
                        workers_with_configs.num_workers()
                    );
                }

                start_kv_router_background_event_plane(
                    component.clone(),
                    kv_indexer.event_sender(),
                    kv_indexer.remove_worker_sender(),
                    cancellation_token.clone(),
                    worker_query::WorkerQueryClient::new(
                        component.clone(),
                        workers_with_configs.subscribe(),
                    ),
                    transport_kind,
                )
                .await?;
            } else {
                if transport_kind == EventTransportKind::Zmq {
                    tracing::warn!(
                        "Not all workers have local_indexer enabled; falling back to JetStream for durability"
                    );
                }
                tracing::info!(
                    "Not all workers have local_indexer enabled, using JetStream subscription"
                );

                // Convert router_id to string for NATS consumer naming
                let consumer_id = router_id.to_string();
                start_kv_router_background(
                    component.clone(),
                    consumer_id,
                    kv_indexer.event_sender(),
                    kv_indexer.remove_worker_sender(),
                    kv_router_config
                        .router_snapshot_threshold
                        .map(|_| kv_indexer.get_workers_sender()),
                    kv_router_config
                        .router_snapshot_threshold
                        .map(|_| kv_indexer.snapshot_event_sender()),
                    cancellation_token.clone(),
                    kv_router_config.router_snapshot_threshold,
                    kv_router_config.router_reset_states,
                )
                .await?;
            }
        }

        tracing::info!("KV Routing initialized");
        Ok(Self {
            indexer,
            scheduler,
            block_size,
            kv_router_config,
            cancellation_token,
            client,
            worker_query_client: Some(worker_query_client),
        })
    }

    /// Get a reference to the client used by this KvRouter
    pub fn client(&self) -> &Client {
        &self.client
    }

    /// Give these tokens, find the worker with the best match in it's KV cache.
    /// Returns the best worker (with dp_rank) and overlap amount in number of blocks.
    /// Now also takes optional context_id for request tracking
    #[allow(clippy::too_many_arguments)]
    pub async fn find_best_match(
        &self,
        context_id: Option<&str>,
        tokens: &[u32],
        router_config_override: Option<&RouterConfigOverride>,
        update_states: bool,
        lora_name: Option<String>,
    ) -> anyhow::Result<(WorkerWithDpRank, u32)> {
        // Validate that context_id is provided when update_states is true
        if update_states && context_id.is_none() {
            panic!("context_id must be provided if update_states is true");
        }

        let isl_tokens = tokens.len();

        let block_hashes = compute_block_hash_for_seq(tokens, self.block_size, None);
        let overlap_scores = self.indexer.find_matches(block_hashes).await?;

        // Compute seq_hashes only if scheduler needs it for active blocks tracking
        let maybe_seq_hashes = self
            .kv_router_config
            .compute_seq_hashes_for_tracking(tokens, self.block_size);

        let best_worker = self
            .scheduler
            .schedule(
                context_id.map(|s| s.to_string()),
                isl_tokens,
                maybe_seq_hashes,
                overlap_scores.clone(),
                router_config_override,
                update_states,
                lora_name,
            )
            .await?;

        // Note: Routing decision recording (for approximate mode) is now handled
        // by KvPushRouter::generate after select_worker returns.

        let overlap_amount = overlap_scores
            .scores
            .get(&best_worker)
            .copied()
            .unwrap_or(0);
        Ok((best_worker, overlap_amount))
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn add_request(
        &self,
        request_id: String,
        tokens: &[u32],
        overlap_blocks: u32,
        expected_output_tokens: Option<u32>,
        worker: WorkerWithDpRank,
        lora_name: Option<String>,
    ) {
        let isl_tokens = tokens.len();

        let maybe_seq_hashes = self
            .kv_router_config
            .compute_seq_hashes_for_tracking(tokens, self.block_size);

        if let Err(e) = self
            .scheduler
            .add_request(
                request_id.clone(),
                maybe_seq_hashes,
                isl_tokens,
                overlap_blocks,
                expected_output_tokens,
                worker,
                lora_name,
            )
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

    pub async fn add_output_block(
        &self,
        request_id: &str,
        decay_fraction: Option<f64>,
    ) -> Result<(), SequenceError> {
        self.scheduler
            .add_output_block(request_id, decay_fraction)
            .await
    }

    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    /// Compute the overlap blocks for a given token sequence and worker.
    /// This queries the indexer to find how many blocks are already cached.
    pub async fn get_overlap_blocks(
        &self,
        tokens: &[u32],
        worker: WorkerWithDpRank,
    ) -> Result<u32, KvRouterError> {
        let block_hashes = compute_block_hash_for_seq(tokens, self.block_size, None);
        let overlap_scores = self.indexer.find_matches(block_hashes).await?;
        Ok(overlap_scores.scores.get(&worker).copied().unwrap_or(0))
    }

    /// Get potential prefill and decode loads for all workers
    pub async fn get_potential_loads(&self, tokens: &[u32]) -> Result<Vec<PotentialLoad>> {
        let isl_tokens = tokens.len();
        let block_hashes = compute_block_hash_for_seq(tokens, self.block_size, None);
        let overlap_scores = self.indexer.find_matches(block_hashes.clone()).await?;

        let maybe_seq_hashes = self
            .kv_router_config
            .compute_seq_hashes_for_tracking(tokens, self.block_size);

        Ok(self
            .scheduler
            .get_potential_loads(maybe_seq_hashes, isl_tokens, overlap_scores)
            .await)
    }

    /// Dump all events from the indexer
    pub async fn dump_events(&self) -> Result<Vec<RouterEvent>, KvRouterError> {
        self.indexer.dump_events().await
    }

    /// Query a specific worker's local KV indexer for its events
    /// (See docstring for `WorkerQueryClient.query_worker()`)
    pub async fn query_worker_local_kv(
        &self,
        worker_id: WorkerId,
        start_event_id: Option<u64>,
        end_event_id: Option<u64>,
    ) -> Result<WorkerKvQueryResponse> {
        let query_client = self
            .worker_query_client
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Worker query client not available (NATS required)"))?;

        query_client
            .query_worker(worker_id, start_event_id, end_event_id)
            .await
    }

    /// Recover missed KV events from a specific worker.
    ///
    /// Queries the worker's local KV indexer for events starting from
    /// `start_event_id` and applies them to the router's indexer.
    ///
    /// # Arguments
    ///
    /// * `worker_id` - The worker to recover from
    /// * `start_event_id` - First event ID to fetch (inclusive), or None to start from beginning
    /// * `end_event_id` - Last event ID to fetch (inclusive), or None for all
    pub async fn recover_from_worker(
        &self,
        worker_id: WorkerId,
        start_event_id: Option<u64>,
        end_event_id: Option<u64>,
    ) -> Result<usize> {
        let query_client = self
            .worker_query_client
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Worker query client not available"))?;

        let event_tx = match &self.indexer {
            Indexer::KvIndexer(kv_indexer) => kv_indexer.event_sender(),
            Indexer::None => {
                anyhow::bail!("Cannot recover: indexer is disabled (--overlap_score_weight is 0)")
            }
        };

        subscriber::recover_from_worker(
            query_client,
            worker_id,
            start_event_id,
            end_event_id,
            &event_tx,
        )
        .await
    }
}

// NOTE: KVRouter works like a PushRouter,
// but without the reverse proxy functionality, but based on contract of 3 request types
#[async_trait]
impl AsyncEngine<SingleIn<RouterRequest>, ManyOut<Annotated<RouterResponse>>, Error> for KvRouter {
    async fn generate(
        &self,
        request: SingleIn<RouterRequest>,
    ) -> Result<ManyOut<Annotated<RouterResponse>>> {
        let (request, ctx) = request.into_parts();
        let context_id = ctx.context().id().to_string();
        // Handle different request types
        let response = match request {
            RouterRequest::New { tokens } => {
                let (best_worker, overlap_blocks) = self
                    .find_best_match(Some(&context_id), &tokens, None, true, None)
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
            RouterRequest::MarkFree => RouterResponse::FreeMarked {
                success: self.free(&context_id).await.is_ok(),
            },
        };

        let response = Annotated::from_data(response);
        let stream = stream::iter(vec![response]);
        Ok(ResponseStream::new(Box::pin(stream), ctx.context()))
    }
}

pub struct KvPushRouter {
    inner: PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>,
    pub chooser: Arc<KvRouter>,
}

/// Result of worker selection containing instance ID, dp_rank, and overlap amount.
struct WorkerSelection {
    instance_id: u64,
    dp_rank: u32,
    overlap_amount: u32,
}

impl KvPushRouter {
    pub fn new(
        inner: PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>,
        chooser: Arc<KvRouter>,
    ) -> Self {
        KvPushRouter { inner, chooser }
    }

    /// Select a worker for the request, either using a preselected worker or finding the best match.
    ///
    /// When `is_query_only` is false and `handle_local_updates` is true, this also registers
    /// the request with the scheduler via `add_request`.
    async fn select_worker(
        &self,
        context_id: &str,
        request: &PreprocessedRequest,
        phase: RequestPhase,
        is_query_only: bool,
        handle_local_updates: bool,
    ) -> Result<WorkerSelection, Error> {
        let routing = request.routing.as_ref();

        // Extract LORA name from routing hints
        let lora_name = routing.and_then(|r| r.lora_name.clone());

        // Get pre-selected worker based on phase, with backend_instance_id as fallback
        let Some(id) = (match phase {
            RequestPhase::Prefill => {
                routing.and_then(|r| r.prefill_worker_id.or(r.backend_instance_id))
            }
            RequestPhase::Decode => {
                routing.and_then(|r| r.decode_worker_id.or(r.backend_instance_id))
            }
            RequestPhase::Aggregated => routing.and_then(|r| r.backend_instance_id),
        }) else {
            // No preselected worker - find the best match
            // Don't update states if this is a query-only request
            let (best_worker, overlap_amount) = self
                .chooser
                .find_best_match(
                    Some(context_id),
                    &request.token_ids,
                    request.router_config_override.as_ref(),
                    !is_query_only,
                    lora_name,
                )
                .await?;

            return Ok(WorkerSelection {
                instance_id: best_worker.worker_id,
                dp_rank: best_worker.dp_rank,
                overlap_amount,
            });
        };

        // Route to pre-selected or explicitly specified worker
        let dp_rank = routing.and_then(|r| r.dp_rank).unwrap_or(0);
        tracing::debug!(
            worker_id = id,
            dp_rank = dp_rank,
            ?phase,
            "Routing to specified worker"
        );

        // Compute actual overlap blocks by querying the indexer
        let worker = WorkerWithDpRank::new(id, dp_rank);
        let overlap_blocks = self
            .chooser
            .get_overlap_blocks(&request.token_ids, worker)
            .await?;

        // Extract expected_output_tokens from routing hints
        let expected_output_tokens = request
            .routing
            .as_ref()
            .and_then(|r| r.expected_output_tokens);

        // Perform add_request if this router handles local updates
        if !is_query_only && handle_local_updates {
            self.chooser
                .add_request(
                    context_id.to_string(),
                    &request.token_ids,
                    overlap_blocks,
                    expected_output_tokens,
                    worker,
                    lora_name,
                )
                .await;
        } else {
            tracing::debug!(
                request_id = %context_id,
                worker_id = id,
                dp_rank = dp_rank,
                "Skipping add_request - query or handled externally"
            );
        }

        Ok(WorkerSelection {
            instance_id: id,
            dp_rank,
            overlap_amount: overlap_blocks,
        })
    }
}

#[async_trait]
impl AsyncEngine<SingleIn<PreprocessedRequest>, ManyOut<Annotated<LLMEngineOutput>>, Error>
    for KvPushRouter
{
    /// Generate method that handles KV-aware routing with three distinct behaviors:
    ///
    /// 1. **If `query_instance_id` annotation is set**:
    ///    - Returns the best matching worker ID without routing the request
    ///    - Does NOT update any router local states
    ///    - Response includes worker_instance_id and token_data annotations
    ///
    /// 2. **If `backend_instance_id` is set in the request**:
    ///    - Routes directly to the specified backend instance
    ///    - DOES update router states to track this request (unless query_instance_id is also set)
    ///    - Bypasses the normal KV matching logic
    ///
    /// 3. **If neither are set (default behavior)**:
    ///    - Finds the best worker based on KV cache overlap
    ///    - Updates router states to track the request
    ///    - Routes to the selected worker
    ///
    /// The router state updates include tracking active sequences and managing
    /// prefill/completion lifecycle for proper KV cache management.
    async fn generate(
        &self,
        request: SingleIn<PreprocessedRequest>,
    ) -> Result<ManyOut<Annotated<LLMEngineOutput>>, Error> {
        // Extract context ID for request tracking
        let context_id = request.context().id().to_string();

        // Simple query-only detection: presence of query_instance_id annotation means query-only mode
        let is_query_only = request.get_annotation_value("query_instance_id").is_some();

        // Determine if this router should handle local state updates (add_request, free, etc.)
        // Default is true (router handles bookkeeping). Set to false for GAIE Stage 2 where
        // an external orchestrator (e.g., EPP sidecar) handles bookkeeping via C FFI.
        let handle_local_updates = request
            .routing
            .as_ref()
            .and_then(|r| r.enable_local_updates)
            .unwrap_or(true);

        // Get phase from tracker (defaults to Aggregated if no tracker or phase not set)
        let phase = request
            .tracker
            .as_ref()
            .map(|t| t.phase())
            .unwrap_or(RequestPhase::Aggregated);

        let block_size = self.chooser.block_size() as usize;
        let selection = self
            .select_worker(
                &context_id,
                &request,
                phase,
                is_query_only,
                handle_local_updates,
            )
            .await?;
        let WorkerSelection {
            instance_id,
            dp_rank,
            overlap_amount,
        } = selection;

        // In approximate mode (use_kv_events=false), record the routing decision
        // so the indexer can track cache state based on routing decisions.
        // This covers both pre-selected workers and find_best_match selections.
        if !is_query_only && !self.chooser.kv_router_config.use_kv_events {
            let worker = WorkerWithDpRank::new(instance_id, dp_rank);
            let mut tokens_with_hashes =
                TokensWithHashes::new(request.token_ids.clone(), self.chooser.block_size);
            if let Err(e) = self
                .chooser
                .indexer
                .process_routing_decision_for_request(&mut tokens_with_hashes, worker)
                .await
            {
                tracing::warn!(
                    request_id = %context_id,
                    worker_id = instance_id,
                    dp_rank = dp_rank,
                    error = %e,
                    "Failed to record routing decision in approximate mode"
                );
            }
        }

        // Record metrics in tracker: KV hit rate and worker ID based on phase
        if let Some(ref tracker) = request.tracker {
            let isl_blocks = request.token_ids.len().div_ceil(block_size);
            tracker.record_kv_hit(overlap_amount, isl_blocks);
            tracker.record_worker(instance_id);
        }

        // Handle query-only requests: early return with worker info
        if is_query_only {
            let stream_context = request.context().clone();
            // Tracker is always created for query-only requests (delta generator enables tracking
            // when query_instance_id annotation is present)
            let worker_id_info = request.tracker.as_ref().and_then(|t| t.get_worker_info());

            tracing::trace!(
                ?phase,
                worker_id = instance_id,
                ?worker_id_info,
                "Returning worker selection (query-only mode)"
            );

            let output = LLMEngineOutput {
                disaggregated_params: Some(json!({
                    "worker_id": worker_id_info,
                    "token_ids": request.token_ids
                })),
                ..Default::default()
            };
            let response = Annotated::from_data(output);
            let stream = stream::iter(vec![response]);
            return Ok(ResponseStream::new(Box::pin(stream), stream_context));
        }

        // Route to worker
        let isl_tokens = request.token_ids.len();
        let expected_output_tokens = request
            .routing
            .as_ref()
            .and_then(|r| r.expected_output_tokens);
        let track_output_blocks =
            self.chooser.kv_router_config.router_track_output_blocks && handle_local_updates;

        let (mut backend_input, context) = request.into_parts();
        backend_input.routing_mut().dp_rank = Some(dp_rank);
        let updated_request = context.map(|_| backend_input);

        let chooser = self.chooser.clone();
        let mut response_stream = self.inner.direct(updated_request, instance_id).await?;
        let stream_context = response_stream.context();
        let context_for_monitoring = stream_context.clone();

        // Wrap stream with lifecycle management (mark_prefill_completed, free)
        // Only perform these operations if handle_local_updates is true.
        // When false, an external caller (e.g., GAIE sidecar) handles bookkeeping via C FFI.
        let wrapped_stream = Box::pin(async_stream::stream! {
            let mut prefill_marked = false;

            // Output block tracking state
            let mut cumulative_osl: usize = 0;
            let mut current_total_blocks = isl_tokens.div_ceil(block_size);

            loop {
                tokio::select! {
                    biased;

                    _ = context_for_monitoring.stopped() => {
                        tracing::debug!("Request {context_id} cancelled, ending stream");
                        break;
                    }

                    item = response_stream.next() => {
                        let Some(item) = item else {
                            break;
                        };

                        if handle_local_updates && !prefill_marked {
                            // Only mark prefill completed when we receive actual tokens,
                            // not empty bootstrap info (token_ids: []) from disaggregated prefill
                            let has_tokens = item.data.as_ref()
                                .map(|d| !d.token_ids.is_empty())
                                .unwrap_or(false);
                            if has_tokens {
                                if let Err(e) = chooser.mark_prefill_completed(&context_id).await {
                                    tracing::warn!("Failed to mark prefill completed for request {context_id}: {e}");
                                }
                                prefill_marked = true;
                            }
                        }

                        // Track output blocks if enabled
                        if track_output_blocks {
                            let new_tokens = item.data.as_ref()
                                .map(|d| d.token_ids.len())
                                .unwrap_or(0);
                            cumulative_osl += new_tokens;

                            let new_total_blocks = (isl_tokens + cumulative_osl).div_ceil(block_size);
                            if new_total_blocks > current_total_blocks {
                                // New block boundary crossed - add output block with decay
                                // Clamp eot to min 1 to avoid division by zero, and result to min 0.0
                                let decay_fraction = expected_output_tokens.map(|eot| {
                                    (1.0 - (cumulative_osl as f64 / eot.max(1) as f64)).max(0.0)
                                });
                                if let Err(e) = chooser.add_output_block(&context_id, decay_fraction).await {
                                    tracing::warn!(
                                        "Failed to add output block for request {context_id}: {e}"
                                    );
                                }
                                current_total_blocks = new_total_blocks;
                            }
                        }

                        yield item;
                    }
                }
            }

            // Only call free() if we handle local updates.
            // When handle_local_updates=false, external caller handles cleanup via C FFI.
            if handle_local_updates
                && let Err(e) = chooser.free(&context_id).await
            {
                tracing::warn!("Failed to free request {context_id}: {e}");
            }
        });
        Ok(ResponseStream::new(wrapped_stream, stream_context))
    }
}

impl Drop for KvRouter {
    fn drop(&mut self) {
        tracing::info!("Dropping KvRouter - cancelling background tasks");
        self.cancellation_token.cancel();
    }
}
