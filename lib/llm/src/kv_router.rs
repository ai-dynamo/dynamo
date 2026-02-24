// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use dynamo_kv_router::{ConcurrentRadixTree, ThreadPoolIndexer};
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
use tokio::sync::oneshot;
use tracing::Instrument;
use validator::Validate;

// Re-export from dynamo-kv-router crate
pub use dynamo_kv_router::approx;
pub use dynamo_kv_router::indexer;
pub use dynamo_kv_router::protocols;

pub mod config;
pub mod indexer_standalone;
pub mod metrics;
pub mod prefill_router;
pub mod publisher;
pub mod push_router;
pub mod queue;
pub mod recorder;
pub mod scheduler;
pub mod sequence;
pub mod subscriber;
pub mod worker_query;

pub use config::{KvRouterConfig, RouterConfigOverride};
pub use indexer_standalone::start_kv_block_indexer;
pub use prefill_router::PrefillRouter;
pub use push_router::{DirectRoutingRouter, KvPushRouter};

use crate::{
    discovery::RuntimeConfigWatch,
    kv_router::{
        approx::PruneConfig,
        indexer::{GetWorkersRequest, KvIndexer, KvIndexerInterface, KvRouterError},
        protocols::{
            BlockExtraInfo, DpRank, LocalBlockHash, OverlapScores, RouterEvent, RouterRequest,
            RouterResponse, TokensWithHashes, WorkerId, WorkerSelectionResult, WorkerWithDpRank,
            compute_block_hash_for_seq,
        },
        scheduler::{KvScheduler, KvSchedulerError, PotentialLoad, SchedulingRequest},
        sequence::{SequenceError, SequenceRequest},
    },
    local_model::runtime_config::ModelRuntimeConfig,
};

// [gluo TODO] shouldn't need to be public
// this should be discovered from the component

// for metric scraping (pull-based)
pub const KV_METRICS_ENDPOINT: &str = "load_metrics";

// for metric publishing (push-based)
pub const KV_EVENT_SUBJECT: &str = "kv-events";
pub const KV_METRICS_SUBJECT: &str = "kv_metrics";

// for inter-router comms
pub const PREFILL_SUBJECT: &str = "prefill_events";
pub const ACTIVE_SEQUENCES_SUBJECT: &str = "active_sequences_events";

// for radix tree snapshot storage
pub const RADIX_STATE_BUCKET: &str = "radix-bucket";
pub const RADIX_STATE_FILE: &str = "radix-state";

// for standalone indexer query
pub const KV_INDEXER_QUERY_ENDPOINT: &str = "kv_indexer_query";

// for worker-local kvindexer query
pub const WORKER_KV_INDEXER_BUFFER_SIZE: usize = 1024; // store 1024 most recent events in worker buffer

/// Generates a dp_rank-specific endpoint name for the worker KV indexer query service.
/// Each dp_rank has its own LocalKvIndexer and query endpoint to ensure per-dp_rank monotonicity.
pub fn worker_kv_indexer_query_endpoint(dp_rank: DpRank) -> String {
    format!("worker_kv_indexer_query_dp{dp_rank}")
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

/// A trait that users can implement to define custom selection logic
pub trait WorkerSelector {
    fn select_worker(
        &self,
        workers: &HashMap<protocols::WorkerId, ModelRuntimeConfig>,
        request: &SchedulingRequest,
        block_size: u32,
    ) -> Result<WorkerSelectionResult, KvSchedulerError>;
}

#[derive(Clone)]
pub enum Indexer {
    /// Single-threaded radix tree with channel-based event processing.
    /// Supports TTL-based expiration and size-based pruning.
    /// Has the ability to persist and snapshot states.
    KvIndexer(KvIndexer),

    /// Concurrent radix tree with a thread pool for event processing.
    /// Uses sticky worker routing for per-worker event serialization.
    /// Does not support TTL/pruning.
    Concurrent(Arc<ThreadPoolIndexer<ConcurrentRadixTree>>),

    /// Used when we do not wish to use the indexer at all (e.g., when overlap_score_weight is 0).
    /// Note: This will cause KV events to accumulate in JetStream as we do not regularly purge them.
    None,
}

impl Indexer {
    pub fn new(
        component: &dynamo_runtime::component::Component,
        kv_router_config: &KvRouterConfig,
        block_size: u32,
    ) -> Self {
        if kv_router_config.overlap_score_weight == 0.0 {
            return Indexer::None;
        }

        if kv_router_config.router_event_threads > 1 {
            return Indexer::Concurrent(Arc::new(ThreadPoolIndexer::new(
                ConcurrentRadixTree::new(),
                kv_router_config.router_event_threads as usize,
                block_size,
            )));
        }

        let kv_indexer_metrics = indexer::KvIndexerMetrics::from_component(component);
        let cancellation_token = component.drt().primary_token();

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
            cancellation_token,
            None, // expiration_duration for frequency tracking
            block_size,
            kv_indexer_metrics,
            prune_config,
        ))
    }

    pub(crate) async fn find_matches(
        &self,
        sequence: Vec<LocalBlockHash>,
    ) -> Result<OverlapScores, KvRouterError> {
        match self {
            Indexer::KvIndexer(indexer) => indexer.find_matches(sequence).await,
            Indexer::Concurrent(tpi) => tpi.find_matches(sequence).await,
            Indexer::None => Ok(OverlapScores::new()),
        }
    }

    pub(crate) async fn dump_events(&self) -> Result<Vec<RouterEvent>, KvRouterError> {
        match self {
            Indexer::KvIndexer(indexer) => indexer.dump_events().await,
            Indexer::Concurrent(tpi) => tpi.dump_events().await,
            Indexer::None => {
                panic!(
                    "Cannot dump events: indexer does not exist (is overlap_score_weight set to 0?)"
                );
            }
        }
    }

    pub(crate) async fn process_routing_decision_for_request(
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
            Indexer::Concurrent(tpi) => {
                tpi.process_routing_decision_for_request(tokens_with_hashes, worker)
                    .await
            }
            Indexer::None => Ok(()),
        }
    }

    pub(crate) async fn apply_event(&self, event: RouterEvent) {
        match self {
            Indexer::KvIndexer(indexer) => {
                if let Err(e) = indexer.event_sender().send(event).await {
                    tracing::warn!("Failed to send event to indexer: {e}");
                }
            }
            Indexer::Concurrent(tpi) => tpi.apply_event(event).await,
            Indexer::None => {}
        }
    }

    pub(crate) async fn remove_worker(&self, worker_id: WorkerId) {
        match self {
            Indexer::KvIndexer(indexer) => {
                if let Err(e) = indexer.remove_worker_sender().send(worker_id).await {
                    tracing::warn!("Failed to send worker removal for {worker_id}: {e}");
                }
            }
            Indexer::Concurrent(tpi) => {
                KvIndexerInterface::remove_worker(tpi.as_ref(), worker_id).await;
            }
            Indexer::None => {}
        }
    }

    pub(crate) async fn get_workers(&self) -> Vec<WorkerId> {
        match self {
            Indexer::KvIndexer(indexer) => {
                let (resp_tx, resp_rx) = oneshot::channel();
                let req = GetWorkersRequest { resp: resp_tx };
                if let Err(e) = indexer.get_workers_sender().send(req).await {
                    tracing::warn!("Failed to send get_workers request: {e}");
                    return Vec::new();
                }
                resp_rx.await.unwrap_or_default()
            }
            Indexer::Concurrent(tpi) => tpi.backend().get_workers(),
            Indexer::None => Vec::new(),
        }
    }
}

/// A KvRouter only decides which worker you should use. It doesn't send you there.
/// TODO: Rename this to indicate it only selects a worker, it does not route.
pub struct KvRouter {
    indexer: Indexer,
    scheduler: KvScheduler,
    block_size: u32,
    kv_router_config: KvRouterConfig,
    cancellation_token: tokio_util::sync::CancellationToken,
    client: Client,
}

impl KvRouter {
    pub async fn new(
        endpoint: Endpoint,
        client: Client,
        mut workers_with_configs: RuntimeConfigWatch,
        block_size: u32,
        selector: Option<Box<dyn WorkerSelector + Send + Sync>>,
        kv_router_config: Option<KvRouterConfig>,
        worker_type: &'static str,
    ) -> Result<Self> {
        let kv_router_config = kv_router_config.unwrap_or_default();
        kv_router_config.validate()?;
        let component = endpoint.component();
        let cancellation_token = component.drt().primary_token();

        let indexer = Indexer::new(component, &kv_router_config, block_size);

        // Wait for at least one worker with a known runtime config before starting scheduler
        let _ = workers_with_configs
            .wait_for(|m| !m.is_empty())
            .await
            .map_err(|_| {
                anyhow::anyhow!("runtime config watch closed before any workers appeared")
            })?;

        let scheduler = KvScheduler::start(
            component.clone(),
            block_size,
            workers_with_configs.clone(),
            selector,
            &kv_router_config,
            worker_type,
        )
        .await?;

        // Start KV event subscription if needed (use_kv_events=true and overlap_score_weight>0)
        if kv_router_config.should_subscribe_to_kv_events() {
            subscriber::start_subscriber(component.clone(), &kv_router_config, indexer.clone())
                .await?;
        } else {
            tracing::info!(
                "Skipping KV event subscription (use_kv_events={}, overlap_score_weight={})",
                kv_router_config.use_kv_events,
                kv_router_config.overlap_score_weight,
            );
        }

        tracing::info!("KV Routing initialized");
        Ok(Self {
            indexer,
            scheduler,
            block_size,
            kv_router_config,
            cancellation_token,
            client,
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

    /// Give these tokens, find the worker with the best match in it's KV cache.
    /// Returns the best worker (with dp_rank) and overlap amount in number of blocks.
    /// Now also takes optional context_id for request tracking
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
    ) -> anyhow::Result<(WorkerWithDpRank, u32)> {
        let start = Instant::now();

        if update_states && context_id.is_none() {
            anyhow::bail!("context_id must be provided when update_states is true");
        }

        let isl_tokens = tokens.len();

        let block_hashes = tracing::info_span!("kv_router.compute_block_hashes")
            .in_scope(|| compute_block_hash_for_seq(tokens, self.block_size, block_mm_infos));
        let hash_elapsed = start.elapsed();

        let overlap_scores = self
            .indexer
            .find_matches(block_hashes)
            .instrument(tracing::info_span!("kv_router.find_matches"))
            .await?;
        let find_matches_elapsed = start.elapsed();

        // Compute seq_hashes only if scheduler needs it for active blocks tracking
        let maybe_seq_hashes = tracing::info_span!("kv_router.compute_seq_hashes").in_scope(|| {
            self.kv_router_config.compute_seq_hashes_for_tracking(
                tokens,
                self.block_size,
                router_config_override,
            )
        });
        let seq_hash_elapsed = start.elapsed();

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
                priority_jump,
            )
            .instrument(tracing::info_span!("kv_router.schedule"))
            .await?;
        let total_elapsed = start.elapsed();

        metrics::ROUTING_OVERHEAD_METRICS.observe(
            hash_elapsed,
            find_matches_elapsed,
            seq_hash_elapsed,
            total_elapsed,
        );

        #[cfg(feature = "bench")]
        tracing::info!(
            isl_tokens,
            hash_us = hash_elapsed.as_micros() as u64,
            find_matches_us = (find_matches_elapsed - hash_elapsed).as_micros() as u64,
            seq_hash_us = (seq_hash_elapsed - find_matches_elapsed).as_micros() as u64,
            schedule_us = (total_elapsed - seq_hash_elapsed).as_micros() as u64,
            total_us = total_elapsed.as_micros() as u64,
            "find_best_match completed"
        );

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
        router_config_override: Option<&RouterConfigOverride>,
    ) {
        let isl_tokens = tokens.len();

        let maybe_seq_hashes = self.kv_router_config.compute_seq_hashes_for_tracking(
            tokens,
            self.block_size,
            router_config_override,
        );

        if let Err(e) = self
            .scheduler
            .add_request(SequenceRequest {
                request_id: request_id.clone(),
                token_sequence: maybe_seq_hashes,
                isl: isl_tokens,
                overlap: overlap_blocks,
                expected_output_tokens,
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

    /// Get the worker type for this router ("prefill" or "decode").
    /// Used for Prometheus metric labeling.
    pub fn worker_type(&self) -> &'static str {
        self.scheduler.worker_type()
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
    pub async fn get_potential_loads(
        &self,
        tokens: &[u32],
        router_config_override: Option<&RouterConfigOverride>,
    ) -> Result<Vec<PotentialLoad>> {
        let isl_tokens = tokens.len();
        let block_hashes = compute_block_hash_for_seq(tokens, self.block_size, None);
        let overlap_scores = self.indexer.find_matches(block_hashes.clone()).await?;

        let maybe_seq_hashes = self.kv_router_config.compute_seq_hashes_for_tracking(
            tokens,
            self.block_size,
            router_config_override,
        );

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
            RouterRequest::MarkFree => RouterResponse::FreeMarked {
                success: self.free(&context_id).await.is_ok(),
            },
        };

        let response = Annotated::from_data(response);
        let stream = stream::iter(vec![response]);
        Ok(ResponseStream::new(Box::pin(stream), ctx.context()))
    }
}

impl Drop for KvRouter {
    fn drop(&mut self) {
        tracing::info!("Dropping KvRouter - cancelling background tasks");
        self.cancellation_token.cancel();
    }
}
