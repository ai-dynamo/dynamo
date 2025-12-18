// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use derive_builder::Builder;
use dynamo_runtime::{
    component::{Client, Endpoint},
    discovery::{DiscoveryQuery, watch_and_extract_field},
    pipeline::{
        AsyncEngine, AsyncEngineContextProvider, Error, ManyOut, PushRouter, ResponseStream,
        SingleIn, async_trait,
    },
    protocols::EndpointId,
    protocols::annotated::Annotated,
    traits::DistributedRuntimeProvider,
};
use futures::stream::{self, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::protocols::openai::nvext::WorkerIdInfo;

pub mod approx;
pub mod indexer;
pub mod prefill_router;
pub mod protocols;
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
    kv_router::{
        approx::PruneConfig,
        indexer::{
            KvIndexer, KvIndexerInterface, KvRouterError, OverlapScores, RouterEvent,
            compute_block_hash_for_seq, compute_seq_hash_for_block,
        },
        protocols::{
            LocalBlockHash, RouterRequest, RouterResponse, WorkerId, WorkerSelectionResult,
            WorkerWithDpRank,
        },
        scheduler::{KvScheduler, KvSchedulerError, PotentialLoad, SchedulingRequest},
        sequence::SequenceError,
        subscriber::{start_kv_router_background, start_kv_router_background_nats_core},
    },
    local_model::runtime_config::ModelRuntimeConfig,
    model_card::ModelDeploymentCard,
    preprocessor::PreprocessedRequest,
    protocols::common::llm_backend::LLMEngineOutput,
    tokens::SequenceHash,
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
pub const WORKER_KV_INDEXER_QUERY_SUBJECT: &str = "worker_kv_indexer_query";
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

/// Specifies the type of worker being queried when using the `query_instance_id` annotation.
/// This tells the router which worker pool to select from and what type of operation is intended.
///
/// Query instance types for worker selection
/// - "prefill" → select a prefill worker (disaggregated serving)
/// - "decode" → select a decode worker (disaggregated serving)
///
/// Note: Empty value ("query_instance_id:") is handled by PrefillRouter for disagg orchestration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QueryInstanceType {
    /// Query for a prefill worker (disaggregated serving)
    Prefill,
    /// Query for a decode worker (disaggregated serving)
    Decode,
}

impl std::fmt::Display for QueryInstanceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QueryInstanceType::Prefill => write!(f, "prefill"),
            QueryInstanceType::Decode => write!(f, "decode"),
        }
    }
}

impl std::str::FromStr for QueryInstanceType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "prefill" => Ok(QueryInstanceType::Prefill),
            "decode" => Ok(QueryInstanceType::Decode),
            _ => Err(format!(
                "Invalid QueryInstanceType: '{s}'. Expected 'prefill' or 'decode'"
            )),
        }
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

    /// Threshold for triggering snapshots. If None, no snapshots will be performed.
    pub router_snapshot_threshold: Option<u32>,

    /// Whether to reset the router state on startup (default: false)
    pub router_reset_states: bool,

    /// TTL for blocks in seconds (only used when use_kv_events is false, default: 120.0)
    pub router_ttl_secs: f64,

    /// Maximum tree size before pruning (only used when use_kv_events is false, default: 1024)
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
            router_snapshot_threshold: Some(1000000),
            router_reset_states: false,
            router_ttl_secs: 120.0,
            router_max_tree_size: 1024,
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
            router_snapshot_threshold: router_snapshot_threshold
                .unwrap_or(default.router_snapshot_threshold),
            router_reset_states: router_reset_states.unwrap_or(default.router_reset_states),
            router_ttl_secs: router_ttl_secs.unwrap_or(default.router_ttl_secs),
            router_max_tree_size: router_max_tree_size.unwrap_or(default.router_max_tree_size),
            router_prune_target_ratio: router_prune_target_ratio
                .unwrap_or(default.router_prune_target_ratio),
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

    async fn process_routing_decision(
        &self,
        worker: WorkerWithDpRank,
        local_hashes: Vec<LocalBlockHash>,
        sequence_hashes: Vec<SequenceHash>,
    ) -> Result<(), KvRouterError> {
        match self {
            Indexer::KvIndexer(indexer) => {
                indexer
                    .process_routing_decision(worker, local_hashes, sequence_hashes)
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
        block_size: u32,
        selector: Option<Box<dyn WorkerSelector + Send + Sync>>,
        kv_router_config: Option<KvRouterConfig>,
        consumer_id: String,
    ) -> Result<Self> {
        let kv_router_config = kv_router_config.unwrap_or_default();
        let component = endpoint.component();
        let cancellation_token = component.drt().primary_token();

        let instance_ids_rx = client.instance_avail_watcher();

        // Watch for runtime config updates via discovery interface
        let discovery = component.drt().discovery();
        let endpoint_id = endpoint.id();
        let discovery_key = DiscoveryQuery::EndpointModels {
            namespace: endpoint_id.namespace.clone(),
            component: endpoint_id.component.clone(),
            endpoint: endpoint_id.name.clone(),
        };
        let discovery_stream = discovery
            .list_and_watch(discovery_key.clone(), Some(cancellation_token.clone()))
            .await?;
        let runtime_configs_rx =
            watch_and_extract_field(discovery_stream, |card: ModelDeploymentCard| {
                card.runtime_config
            });

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

        let scheduler = KvScheduler::start(
            component.clone(),
            block_size,
            instance_ids_rx,
            runtime_configs_rx.clone(),
            selector,
            kv_router_config.router_replica_sync,
            consumer_id.clone(),
        )
        .await?;

        // Initialize worker query client using namespace abstraction
        // (created before background task so we can use it for startup recovery)
        let worker_query_client =
            worker_query::WorkerQueryClient::new(component.clone(), runtime_configs_rx.clone());
        tracing::info!("Worker query client initialized");

        // Start KV event subscriber background process (only when use_kv_events is enabled)
        // This is spawned as a background task to avoid blocking router startup.
        // The task waits for runtime_configs to determine whether to use NATS Core or JetStream.
        if kv_router_config.use_kv_events
            && let Indexer::KvIndexer(ref kv_indexer) = indexer
        {
            // Clone everything needed for the background task
            let component_clone = component.clone();
            let kv_indexer_clone = kv_indexer.clone();
            let cancellation_token_clone = cancellation_token.clone();
            let mut runtime_configs_rx_clone = runtime_configs_rx.clone();
            let worker_query_client_clone =
                worker_query::WorkerQueryClient::new(component.clone(), runtime_configs_rx.clone());

            tokio::spawn(async move {
                // Wait for runtime_configs to have at least one entry
                let (all_local_indexer, count) = loop {
                    {
                        let configs = runtime_configs_rx_clone.borrow();
                        if !configs.is_empty() {
                            let all_local_indexer =
                                configs.values().all(|c| c.enable_local_indexer);
                            break (all_local_indexer, configs.len());
                        }
                    }

                    // Wait for changes to runtime_configs
                    tokio::select! {
                        _ = cancellation_token_clone.cancelled() => {
                            tracing::debug!("Subscriber selection task cancelled");
                            return;
                        }
                        result = runtime_configs_rx_clone.changed() => {
                            if result.is_err() {
                                tracing::debug!("Runtime configs channel closed");
                                return;
                            }
                        }
                    }
                };

                if all_local_indexer {
                    // All workers have local_indexer enabled - use NATS Core
                    tracing::info!(
                        "All {count} workers have local_indexer enabled, using NATS Core subscription"
                    );

                    if let Err(e) = start_kv_router_background_nats_core(
                        component_clone.clone(),
                        kv_indexer_clone.event_sender(),
                        kv_indexer_clone.remove_worker_sender(),
                        cancellation_token_clone.clone(),
                        worker_query_client_clone,
                    )
                    .await
                    {
                        tracing::error!("Failed to start NATS Core subscriber: {e}");
                    }
                } else {
                    // Not all workers have local_indexer - use JetStream
                    tracing::info!(
                        "Not all workers have local_indexer enabled, using JetStream subscription"
                    );

                    if let Err(e) = start_kv_router_background(
                        component_clone.clone(),
                        consumer_id,
                        kv_indexer_clone.event_sender(),
                        kv_indexer_clone.remove_worker_sender(),
                        kv_router_config
                            .router_snapshot_threshold
                            .map(|_| kv_indexer_clone.get_workers_sender()),
                        kv_router_config
                            .router_snapshot_threshold
                            .map(|_| kv_indexer_clone.snapshot_event_sender()),
                        cancellation_token_clone.clone(),
                        kv_router_config.router_snapshot_threshold,
                        kv_router_config.router_reset_states,
                    )
                    .await
                    {
                        tracing::error!("Failed to start JetStream subscriber: {e}");
                    }
                }
            });
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
    pub async fn find_best_match(
        &self,
        context_id: Option<&str>,
        tokens: &[u32],
        router_config_override: Option<&RouterConfigOverride>,
        update_states: bool,
    ) -> anyhow::Result<(WorkerWithDpRank, u32)> {
        // Validate that context_id is provided when update_states is true
        if update_states && context_id.is_none() {
            panic!("context_id must be provided if update_states is true");
        }

        let isl_tokens = tokens.len();

        let block_hashes = compute_block_hash_for_seq(tokens, self.block_size);
        let seq_hashes = compute_seq_hash_for_block(&block_hashes);

        let overlap_scores = self.indexer.find_matches(block_hashes.clone()).await?;

        // Determine who needs seq_hashes
        let needs_process_routing = !self.kv_router_config.use_kv_events;
        let scheduler_needs_it = self.kv_router_config.router_track_active_blocks;

        // Optimize cloning: only clone if both need it, otherwise move
        let (maybe_seq_hashes_1, maybe_seq_hashes_2) =
            match (needs_process_routing, scheduler_needs_it) {
                (true, true) => (Some(seq_hashes.clone()), Some(seq_hashes)),
                (true, false) => (Some(seq_hashes), None),
                (false, true) => (None, Some(seq_hashes)),
                (false, false) => (None, None),
            };

        let best_worker = self
            .scheduler
            .schedule(
                context_id.map(|s| s.to_string()),
                isl_tokens,
                maybe_seq_hashes_2,
                overlap_scores.clone(),
                router_config_override,
                update_states,
            )
            .await?;

        // Process routing decision when not using KV events (approximate mode with TTL/pruning)
        if needs_process_routing {
            self.indexer
                .process_routing_decision(best_worker, block_hashes, maybe_seq_hashes_1.unwrap())
                .await?;
        }

        let overlap_amount = overlap_scores
            .scores
            .get(&best_worker)
            .copied()
            .unwrap_or(0);
        Ok((best_worker, overlap_amount))
    }

    pub async fn add_request(
        &self,
        request_id: String,
        tokens: &[u32],
        overlap_blocks: u32,
        worker: WorkerWithDpRank,
    ) {
        let isl_tokens = tokens.len();

        let maybe_seq_hashes = self.kv_router_config.router_track_active_blocks.then(|| {
            let block_hashes = compute_block_hash_for_seq(tokens, self.block_size);
            compute_seq_hash_for_block(&block_hashes)
        });

        if let Err(e) = self
            .scheduler
            .add_request(
                request_id.clone(),
                maybe_seq_hashes,
                isl_tokens,
                overlap_blocks,
                worker,
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

    pub fn block_size(&self) -> u32 {
        self.block_size
    }

    /// Get the disaggregated endpoint for a worker, if available.
    /// Used to look up bootstrap host/port for prefill workers.
    pub async fn get_disaggregated_endpoint(
        &self,
        worker_id: u64,
    ) -> Option<crate::local_model::runtime_config::DisaggregatedEndpoint> {
        self.scheduler.get_disaggregated_endpoint(worker_id).await
    }

    /// Get potential prefill and decode loads for all workers
    pub async fn get_potential_loads(&self, tokens: &[u32]) -> Result<Vec<PotentialLoad>> {
        let isl_tokens = tokens.len();
        let block_hashes = compute_block_hash_for_seq(tokens, self.block_size);
        let overlap_scores = self.indexer.find_matches(block_hashes).await?;

        let maybe_seq_hashes = self.kv_router_config.router_track_active_blocks.then(|| {
            let block_hashes = compute_block_hash_for_seq(tokens, self.block_size);
            compute_seq_hash_for_block(&block_hashes)
        });

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
                    .find_best_match(Some(&context_id), &tokens, None, true)
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

impl KvPushRouter {
    pub fn new(
        inner: PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>,
        chooser: Arc<KvRouter>,
    ) -> Self {
        KvPushRouter { inner, chooser }
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

        // Check if this is a query_instance_id request and parse its type
        // Format: "query_instance_id:type" where type is "prefill", "decode", or "" (empty for aggregated)
        // Empty value ("query_instance_id:") means GAIE Aggregated mode - return same worker as both prefill and decode
        let query_instance_annotation = request.get_annotation_value("query_instance_id");
        let is_gaie_agg_query = query_instance_annotation
            .as_ref()
            .is_some_and(|s| s.is_empty());
        let query_instance_type: Option<QueryInstanceType> =
            if let Some(type_str) = &query_instance_annotation {
                match type_str.parse::<QueryInstanceType>() {
                    Ok(t) => Some(t),
                    Err(_) if type_str.is_empty() => {
                        // Empty value is valid for aggregated mode, not a warning
                        None
                    }
                    Err(e) => {
                        tracing::warn!("Invalid query_instance_id type '{type_str}': {e}");
                        None
                    }
                }
            } else {
                None
            };

        let (instance_id, dp_rank, overlap_amount) = if let Some(id) = request.backend_instance_id {
            // If instance_id is set, use it and compute actual overlap
            let dp_rank = request.dp_rank.unwrap_or(0);
            if query_instance_type.is_some() {
                tracing::debug!(
                    "backend_instance_id is set, routing to instance {id} with dp_rank {dp_rank} and ignoring query_instance_id annotation"
                );
            }

            // Compute actual overlap blocks by querying the indexer
            let block_hashes =
                compute_block_hash_for_seq(&request.token_ids, self.chooser.block_size());
            let overlap_scores = self.chooser.indexer.find_matches(block_hashes).await?;
            let worker = WorkerWithDpRank::new(id, dp_rank);
            let overlap_blocks = overlap_scores.scores.get(&worker).copied().unwrap_or(0);

            self.chooser
                .add_request(
                    context_id.clone(),
                    &request.token_ids,
                    overlap_blocks,
                    worker,
                )
                .await;
            (id, dp_rank, overlap_blocks)
        } else {
            // Otherwise, find the best match
            // Don't update states if this is a query-only request (any query_instance_id annotation)
            let should_update_states = query_instance_annotation.is_none();
            let (best_worker, overlap_amount) = self
                .chooser
                .find_best_match(
                    Some(&context_id),
                    &request.token_ids,
                    request.router_config_override.as_ref(),
                    should_update_states,
                )
                .await?;
            (best_worker.worker_id, best_worker.dp_rank, overlap_amount)
        };

        // If request has a query_instance_id annotation, return worker selection info
        // without routing to the actual worker. Returns LLMEngineOutput with disaggregated_params
        // containing worker_id info, same structure as normal execution for uniform extraction.
        let stream_context = request.context().clone();

        // Handle query-only requests (GAIE Stage 1)
        if query_instance_type.is_some() || is_gaie_agg_query {
            let worker_id_info = if is_gaie_agg_query {
                // GAIE Aggregated mode: same worker serves both prefill and decode
                tracing::trace!(
                    query_type = "aggregated",
                    worker_id = instance_id,
                    "Returning aggregated worker selection (same worker for prefill and decode)"
                );
                WorkerIdInfo {
                    prefill_worker_id: Some(instance_id),
                    decode_worker_id: Some(instance_id),
                }
            } else {
                match query_instance_type.unwrap() {
                    QueryInstanceType::Prefill => {
                        tracing::trace!(
                            query_type = "prefill",
                            prefill_worker_id = instance_id,
                            "Returning prefill worker selection"
                        );
                        WorkerIdInfo {
                            prefill_worker_id: Some(instance_id),
                            decode_worker_id: None,
                        }
                    }
                    QueryInstanceType::Decode => {
                        // Get prefill_worker_id from annotation (set by caller after prefill selection)
                        let prefill_worker_id = request
                            .get_annotation_value("prefill_worker_id")
                            .and_then(|s| s.parse::<u64>().ok());
                        tracing::trace!(
                            query_type = "decode",
                            prefill_worker_id = ?prefill_worker_id,
                            decode_worker_id = instance_id,
                            "Returning decode worker selection"
                        );
                        WorkerIdInfo {
                            prefill_worker_id,
                            decode_worker_id: Some(instance_id),
                        }
                    }
                }
            };

            // Return as LLMEngineOutput with disaggregated_params (same structure as normal execution)
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
        let (mut backend_input, context) = request.into_parts();
        backend_input.estimated_prefix_hit_num_blocks = Some(overlap_amount);
        backend_input.dp_rank = Some(dp_rank);

        // Get prefill worker ID from prefill_result if available
        // In aggregated mode, prefill_result is None, so we use decode_worker_id for both
        let decode_worker_id = instance_id;
        let prefill_worker_id = backend_input
            .prefill_result
            .as_ref()
            .and_then(|prefill_result| {
                prefill_result
                    .disaggregated_params
                    .get("worker_id")
                    .and_then(|v| serde_json::from_value::<WorkerIdInfo>(v.clone()).ok())
                    .and_then(|info| info.prefill_worker_id)
            })
            .or(Some(decode_worker_id)); // Use decode_worker_id if no separate prefill worker

        let updated_request = context.map(|_| backend_input);

        let mut response_stream = self.inner.direct(updated_request, instance_id).await?;
        let stream_context = response_stream.context();
        let chooser = self.chooser.clone();
        let context_for_monitoring = stream_context.clone();

        let wrapped_stream = Box::pin(async_stream::stream! {
            let mut prefill_marked = false;
            let mut first_item = true;

            loop {
                tokio::select! {
                    biased;

                    _ = context_for_monitoring.stopped() => {
                        tracing::debug!("Request {context_id} cancelled, ending stream");
                        break;
                    }

                    item = response_stream.next() => {
                        let Some(mut item) = item else {
                            break;
                        };

                        if !prefill_marked {
                            if let Err(e) = chooser.mark_prefill_completed(&context_id).await {
                                tracing::warn!("Failed to mark prefill completed for request {context_id}: {e}");
                            }
                            prefill_marked = true;
                        }

                        // Always inject worker_id in first item's disaggregated_params
                        // This is needed for:
                        // 1. PrefillRouter to know which prefill worker was chosen
                        // 2. Client response when extra_fields contains "worker_id"
                        if first_item {
                            first_item = false;

                            let Some(ref mut data) = item.data else {
                                yield item;
                                continue;
                            };

                            // prefill_worker_id comes from prefill_result.disaggregated_params or falls back to instance_id
                            // decode_worker_id is always the current instance_id
                            let worker_id_info = WorkerIdInfo {
                                prefill_worker_id,
                                decode_worker_id: Some(decode_worker_id),
                            };
                            let worker_id_json = serde_json::to_value(&worker_id_info)
                                .expect("WorkerIdInfo serialization should not fail");

                            if let Some(obj) = data.disaggregated_params.as_mut().and_then(|p| p.as_object_mut()) {
                                obj.insert("worker_id".to_string(), worker_id_json);
                            } else {
                                data.disaggregated_params = Some(json!({"worker_id": worker_id_json}));
                            }
                        }

                        yield item;
                    }
                }
            }

            if let Err(e) = chooser.free(&context_id).await {
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
