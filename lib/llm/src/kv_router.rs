// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use derive_builder::Builder;
use dynamo_runtime::{
    component::Component,
    discovery::{DiscoveryQuery, watch_and_extract_field},
    pipeline::{
        AsyncEngine, AsyncEngineContextProvider, Error, ManyOut, PushRouter, ResponseStream,
        SingleIn, async_trait,
    },
    protocols::annotated::Annotated,
    traits::DistributedRuntimeProvider,
};
use futures::stream::{self, StreamExt};
use serde::{Deserialize, Serialize};

pub mod approx;
pub mod indexer;
pub mod prefill_router;
pub mod protocols;
pub mod publisher;
pub mod recorder;
pub mod scheduler;
pub mod scoring;
pub mod sequence;
pub mod subscriber;

pub use prefill_router::PrefillRouter;

use crate::{
    kv_router::{
        approx::ApproxKvIndexer,
        approx::PruneConfig,
        indexer::{
            KvIndexer, KvIndexerInterface, KvRouterError, OverlapScores, RouterEvent,
            compute_block_hash_for_seq, compute_seq_hash_for_block,
        },
        protocols::{
            LocalBlockHash, RouterRequest, RouterResponse, WorkerSelectionResult, WorkerWithDpRank,
        },
        scheduler::{KvScheduler, KvSchedulerError, PotentialLoad, SchedulingRequest},
        subscriber::start_kv_router_background,
    },
    local_model::runtime_config::ModelRuntimeConfig,
    model_card::ModelDeploymentCard,
    preprocessor::PreprocessedRequest,
    protocols::common::llm_backend::LLMEngineOutput,
};

// [gluo TODO] shouldn't need to be public
// this should be discovered from the component

// for metric scraping (pull-based)
pub const KV_METRICS_ENDPOINT: &str = "load_metrics";

// for metric publishing (push-based)
pub const KV_EVENT_SUBJECT: &str = "kv_events";
pub const KV_HIT_RATE_SUBJECT: &str = "kv-hit-rate";
pub const KV_METRICS_SUBJECT: &str = "kv_metrics";

// for inter-router comms
pub const PREFILL_SUBJECT: &str = "prefill_events";
pub const ACTIVE_SEQUENCES_SUBJECT: &str = "active_sequences_events";

// for radix tree snapshot storage
pub const RADIX_STATE_BUCKET: &str = "radix-bucket";
pub const RADIX_STATE_FILE: &str = "radix-state";

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
        }
    }
}

// TODO: is there a way (macro) to auto-derive the KvIndexerInterface trait for this
// since both variants implement it
pub enum Indexer {
    /// Updates itself based on KV events emitted by backend workers.
    /// Has the ability to persist and snapshot states.
    KvIndexer(KvIndexer),

    /// Predicts the cached blocks based on requests on a TTL basis.
    /// Currently does not persist or snapshot states (WIP to enable that).
    ApproxKvIndexer(ApproxKvIndexer),

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
            Indexer::ApproxKvIndexer(indexer) => indexer.find_matches(sequence).await,
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
            Indexer::ApproxKvIndexer(indexer) => indexer.dump_events().await,
            Indexer::None => {
                panic!(
                    "Cannot dump events: indexer does not exist (is overlap_score_weight set to 0?)"
                );
            }
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
}

impl KvRouter {
    pub async fn new(
        component: Component,
        block_size: u32,
        selector: Option<Box<dyn WorkerSelector + Send + Sync>>,
        kv_router_config: Option<KvRouterConfig>,
        consumer_uuid: String,
    ) -> Result<Self> {
        let kv_router_config = kv_router_config.unwrap_or_default();
        let cancellation_token = component.drt().primary_token();
        let generate_endpoint = component.endpoint("generate");
        let client = generate_endpoint.client().await?;

        let instances_rx = client.instance_source.as_ref().clone();

        // Watch for runtime config updates via discovery interface
        let discovery = component.drt().discovery();
        let discovery_key = DiscoveryQuery::EndpointModels {
            namespace: component.namespace().name().to_string(),
            component: component.name().to_string(),
            endpoint: "generate".to_string(),
        };
        let discovery_stream = discovery
            .list_and_watch(discovery_key, Some(cancellation_token.clone()))
            .await?;
        let runtime_configs_rx =
            watch_and_extract_field(discovery_stream, |card: ModelDeploymentCard| {
                card.runtime_config
            });

        let indexer = if kv_router_config.overlap_score_weight == 0.0 {
            // When overlap_score_weight is zero, we don't need to track prefixes
            Indexer::None
        } else if kv_router_config.use_kv_events {
            let kv_indexer_metrics = indexer::KvIndexerMetrics::from_component(&component);
            Indexer::KvIndexer(KvIndexer::new(
                cancellation_token.clone(),
                block_size,
                kv_indexer_metrics,
            ))
        } else {
            // hard code 120 seconds for now
            Indexer::ApproxKvIndexer(ApproxKvIndexer::new(
                cancellation_token.clone(),
                block_size,
                Duration::from_secs(120),
                Some(PruneConfig {
                    max_tree_size: 2usize.pow(14), // 2** 14 = 16384
                    prune_target_ratio: 0.8,
                }),
            ))
        };

        let scheduler = KvScheduler::start(
            component.clone(),
            block_size,
            instances_rx,
            runtime_configs_rx,
            selector,
            kv_router_config.router_replica_sync,
            consumer_uuid.clone(),
        )
        .await?;

        // Start unified background process if using KvIndexer
        if let Indexer::KvIndexer(ref kv_indexer) = indexer {
            start_kv_router_background(
                component.clone(),
                consumer_uuid,
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

        tracing::info!("KV Routing initialized");
        Ok(Self {
            indexer,
            scheduler,
            block_size,
            kv_router_config,
            cancellation_token,
        })
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
        let approx_indexer_needs_it = matches!(self.indexer, Indexer::ApproxKvIndexer(_));
        let scheduler_needs_it = self.kv_router_config.router_track_active_blocks;

        // Optimize cloning: only clone if both need it, otherwise move
        let (maybe_seq_hashes_1, maybe_seq_hashes_2) =
            match (approx_indexer_needs_it, scheduler_needs_it) {
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

        if let Indexer::ApproxKvIndexer(ref indexer) = self.indexer {
            indexer
                .process_routing_decision(best_worker, block_hashes, maybe_seq_hashes_1.unwrap())
                .await
                .unwrap();
        };

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

        self.scheduler
            .add_request(
                request_id,
                maybe_seq_hashes,
                isl_tokens,
                overlap_blocks,
                worker,
            )
            .await;
    }

    pub async fn mark_prefill_completed(&self, request_id: &str) -> Result<()> {
        self.scheduler.mark_prefill_completed(request_id).await
    }

    pub async fn free(&self, request_id: &str) -> Result<()> {
        self.scheduler.free(request_id).await
    }

    pub fn block_size(&self) -> u32 {
        self.block_size
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

        // Check if this is a query_instance_id request first
        let query_instance_id = request.has_annotation("query_instance_id");

        let (instance_id, dp_rank, overlap_amount) = if let Some(id) = request.backend_instance_id {
            // If instance_id is set, use it and compute actual overlap
            let dp_rank = request.dp_rank.unwrap_or(0);
            if query_instance_id {
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
            let (best_worker, overlap_amount) = self
                .chooser
                .find_best_match(
                    Some(&context_id),
                    &request.token_ids,
                    request.router_config_override.as_ref(),
                    !query_instance_id, // Don't update states if query_instance_id
                )
                .await?;
            (best_worker.worker_id, best_worker.dp_rank, overlap_amount)
        };

        // if request has the annotation "query_instance_id",
        // then the request will not be routed to the worker,
        // and instead the worker_instance_id will be returned.
        let stream_context = request.context().clone();
        if query_instance_id {
            let instance_id_str = instance_id.to_string();
            let response = Annotated::from_annotation("worker_instance_id", &instance_id_str)?;

            // Return the tokens in nvext.token_data format
            let response_tokens = Annotated::from_annotation("token_data", &request.token_ids)?;
            tracing::trace!(
                "Tokens requested in the response through the query_instance_id annotation: {:?}",
                response_tokens
            );
            let stream = stream::iter(vec![response, response_tokens]);
            return Ok(ResponseStream::new(Box::pin(stream), stream_context));
        }
        let (mut backend_input, context) = request.into_parts();
        backend_input.estimated_prefix_hit_num_blocks = Some(overlap_amount);
        backend_input.dp_rank = Some(dp_rank);
        let updated_request = context.map(|_| backend_input);

        let mut response_stream = self.inner.direct(updated_request, instance_id).await?;
        let stream_context = response_stream.context();
        let chooser = self.chooser.clone();
        let context_for_monitoring = stream_context.clone();

        let wrapped_stream = Box::pin(async_stream::stream! {
            let mut prefill_marked = false;

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

                        if !prefill_marked {
                            if let Err(e) = chooser.mark_prefill_completed(&context_id).await {
                                tracing::warn!("Failed to mark prefill completed for request {context_id}: {e:?}");
                            }
                            prefill_marked = true;
                        }
                        yield item;
                    }
                }
            }

            if let Err(e) = chooser.free(&context_id).await {
                tracing::warn!("Failed to free request {context_id}: {e:?}");
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
