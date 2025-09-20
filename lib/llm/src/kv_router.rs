// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use derive_builder::Builder;
use dynamo_runtime::{
    component::{Component, InstanceSource},
    pipeline::{
        AsyncEngine, AsyncEngineContextProvider, Error, ManyOut, PushRouter, ResponseStream,
        SingleIn, async_trait,
    },
    prelude::*,
    protocols::annotated::Annotated,
    traits::events::{EventPublisher, EventSubscriber},
    utils::typed_prefix_watcher::{key_extractors, watch_prefix_with_extraction},
};
use futures::stream::{self, StreamExt};
use serde::{Deserialize, Serialize};

pub mod approx;
pub mod indexer;
pub mod metrics_aggregator;
pub mod protocols;
pub mod publisher;
pub mod recorder;
pub mod scheduler;
pub mod scoring;
pub mod sequence;
pub mod subscriber;

use crate::{
    discovery::{MODEL_ROOT_PATH, ModelEntry},
    kv_router::{
        approx::ApproxKvIndexer,
        indexer::{
            KvIndexer, KvIndexerInterface, KvRouterError, OverlapScores, RouterEvent,
            compute_block_hash_for_seq, compute_seq_hash_for_block,
        },
        protocols::{LocalBlockHash, RouterRequest, RouterResponse, WorkerSelectionResult},
        scheduler::{KvScheduler, KvSchedulerError, PotentialLoad, SchedulingRequest},
        scoring::ProcessedEndpoints,
        subscriber::start_kv_router_background,
    },
    local_model::runtime_config::ModelRuntimeConfig,
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
pub const ROUTER_SNAPSHOT_LOCK: &str = "router-snapshot-lock";
pub const ROUTER_CLEANUP_LOCK: &str = "router-cleanup-lock";

/// A trait that users can implement to define custom selection logic
pub trait WorkerSelector {
    fn select_worker(
        &self,
        workers: &HashMap<i64, Option<ModelRuntimeConfig>>,
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

    // TODO: this is not actually used for now
    // Would need this (along with total kv blocks) to trigger AllWorkersBusy error for e.g. rate-limiting
    pub max_num_batched_tokens: u32,

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
            max_num_batched_tokens: 8192,
            router_snapshot_threshold: Some(10000),
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
        max_num_batched_tokens: Option<u32>,
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
            max_num_batched_tokens: max_num_batched_tokens
                .unwrap_or(default.max_num_batched_tokens),
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
    indexer: Arc<Indexer>,

    // How about a Box<dyn KvIndexerInterface>
    scheduler: KvScheduler,

    block_size: u32,

    kv_router_config: KvRouterConfig,
}

impl KvRouter {
    /// Core implementation shared by both the direct API and the NATS service.
    async fn select_prefill_worker_core(
        indexer: &Indexer,
        scheduler: &KvScheduler,
        block_size: u32,
        context_id: &str,
        tokens: &[u32],
        router_config_override: Option<&RouterConfigOverride>,
    ) -> anyhow::Result<(i64, u32)> {
        let isl_tokens = tokens.len();

        if tokens.is_empty() {
            anyhow::bail!("cannot select prefill worker for empty token set");
        }
        let block_hashes = compute_block_hash_for_seq(tokens, block_size);
        let seq_hashes = compute_seq_hash_for_block(&block_hashes);
        let overlap_scores = indexer.find_matches(block_hashes).await?;

        let best_prefill_worker_id = scheduler
            .schedule_prefill_worker(
                format!("prefill_selection_{}", context_id),
                isl_tokens,
                seq_hashes.clone(),
                overlap_scores.clone(),
                router_config_override,
            )
            .await?;

        let overlap_amount = overlap_scores
            .scores
            .get(&best_prefill_worker_id)
            .copied()
            .unwrap_or(0);

        Ok((best_prefill_worker_id, overlap_amount))
    }
    pub async fn new(
        component: Component,
        block_size: u32,
        selector: Option<Box<dyn WorkerSelector + Send + Sync>>,
        kv_router_config: Option<KvRouterConfig>,
        consumer_uuid: String,
    ) -> Result<Self> {
        let kv_router_config = kv_router_config.unwrap_or_default();

        let cancellation_token = component
            .drt()
            .primary_lease()
            .expect("Cannot KV route static workers")
            .primary_token();

        // CRITICAL ARCHITECTURAL CHANGE: Unified worker discovery for intelligent prefill routing
        //
        // PREVIOUS BEHAVIOR: Router only watched "backend" component (decode workers)
        // NEW BEHAVIOR: Router watches both "backend" (decode) and "prefill" components
        //
        // This change enables the router to:
        // 1. Discover prefill workers as available routing targets
        // 2. Track their runtime capabilities (memory, slots, load)
        // 3. Apply intelligent selection logic to prefill workers
        // 4. Provide unified routing decisions across both worker types

        // BACKEND COMPONENT DISCOVERY: Access backend workers from the same namespace
        let backend_component = component
            .drt()
            .namespace(component.namespace().name())
            .and_then(|ns| ns.component("backend"))
            .unwrap_or_else(|_| {
                tracing::warn!("Backend component not found, using current component as fallback");
                component.clone()
            });
        let backend_endpoint = backend_component.endpoint("generate");
        let backend_client = backend_endpoint.client().await?;

        let backend_instances_rx = match backend_client.instance_source.as_ref() {
            InstanceSource::Dynamic(rx) => rx.clone(),
            InstanceSource::Static => {
                panic!("Expected dynamic instance source for KV routing");
            }
        };

        // OPTIONAL PREFILL COMPONENT DISCOVERY: Gracefully handle prefill worker availability
        //
        // DESIGN DECISION: Prefill component discovery is optional to support different deployment patterns:
        // 1. Unified deployment: Single workers handle both prefill and decode (no prefill component)
        // 2. Disaggregated deployment: Separate prefill and decode workers (both components present)
        //
        // If prefill component exists, router gains visibility into dedicated prefill workers
        // If not present, router operates normally with only decode workers (backward compatibility)
        let prefill_instances_rx = match component
            .drt()
            .namespace(component.namespace().name())
            .and_then(|ns| ns.component("prefill"))
            .map(|prefill_comp| prefill_comp.endpoint("generate"))
        {
            Ok(prefill_endpoint) => match prefill_endpoint.client().await {
                Ok(prefill_client) => match prefill_client.instance_source.as_ref() {
                    InstanceSource::Dynamic(rx) => Some(rx.clone()),
                    InstanceSource::Static => None,
                },
                Err(_) => {
                    tracing::info!(
                        "Failed to create prefill client, router will only see backend workers"
                    );
                    None
                }
            },
            Err(_) => {
                tracing::info!(
                    "Prefill component not available, router will only see backend workers"
                );
                None
            }
        };

        // INSTANCE STREAM UNIFICATION: Merge backend and prefill worker instances
        //
        // IMPLEMENTATION: Creates a unified view of all workers (decode + prefill) for the scheduler.
        // This background task continuously merges instance updates from both components,
        // providing the scheduler with a complete worker inventory for intelligent routing.
        //
        // BENEFITS:
        // 1. Scheduler sees all available workers regardless of component type
        // 2. Dynamic updates when workers join/leave either component
        // 3. Consistent interface for existing scheduler logic
        // 4. Graceful handling when prefill component is unavailable
        let (unified_tx, instances_rx) = tokio::sync::watch::channel(Vec::new());

        // Background task: Continuously merge instances from both components
        let merge_token = cancellation_token.clone();
        tokio::spawn(async move {
            let mut backend_rx = backend_instances_rx;
            let mut prefill_rx_opt = prefill_instances_rx;

            // Initial snapshot
            {
                let mut all_instances = backend_rx.borrow().clone();
                if let Some(ref prefill_rx) = prefill_rx_opt {
                    all_instances.extend(prefill_rx.borrow().clone());
                }
                let _ = unified_tx.send(all_instances);
            }

            loop {
                tokio::select! {
                    _ = merge_token.cancelled() => {
                        tracing::debug!("Instance merger task cancelled");
                        break;
                    }
                    // Handle backend worker instance changes
                    result = backend_rx.changed() => {
                        if result.is_err() {
                            tracing::warn!("Backend instance watch ended");
                            break;
                        }

                        // Merge current instances from both components
                        let mut all_instances = backend_rx.borrow().clone();
                        tracing::debug!("Backend instances: {} workers", all_instances.len());
                        for instance in &all_instances {
                            tracing::debug!("Backend worker {} from component '{}'", instance.instance_id, instance.component);
                        }

                        if let Some(ref mut prefill_rx) = prefill_rx_opt {
                            let prefill_instances = prefill_rx.borrow().clone();
                            tracing::debug!("Prefill instances: {} workers", prefill_instances.len());
                            for instance in &prefill_instances {
                                tracing::debug!("Prefill worker {} from component '{}'", instance.instance_id, instance.component);
                            }
                            all_instances.extend(prefill_instances);
                        }

                        if unified_tx.send(all_instances).is_err() {
                            tracing::debug!("Unified instance receiver closed");
                            break;
                        }
                    }
                    // Handle prefill worker instance changes (if prefill component exists)
                    result = async {
                        if let Some(ref mut prefill_rx) = prefill_rx_opt {
                            prefill_rx.changed().await
                        } else {
                            std::future::pending().await
                        }
                    } => {
                        if result.is_err() {
                            tracing::warn!("Prefill instance watch ended");
                            prefill_rx_opt = None;
                            continue;
                        }

                        // Merge current instances from both components
                        let mut all_instances = backend_rx.borrow().clone();
                        tracing::debug!("Backend instances: {} workers", all_instances.len());
                        for instance in &all_instances {
                            tracing::debug!("Backend worker {} from component '{}'", instance.instance_id, instance.component);
                        }

                        if let Some(ref prefill_rx) = prefill_rx_opt {
                            let prefill_instances = prefill_rx.borrow().clone();
                            tracing::debug!("Prefill instances: {} workers", prefill_instances.len());
                            for instance in &prefill_instances {
                                tracing::debug!("Prefill worker {} from component '{}'", instance.instance_id, instance.component);
                            }
                            all_instances.extend(prefill_instances);
                        }

                        if unified_tx.send(all_instances).is_err() {
                            tracing::debug!("Unified instance receiver closed");
                            break;
                        }
                    }
                }
            }
        });

        // Create runtime config watcher using the generic etcd watcher
        // TODO: Migrate to discovery_client() once it exposes kv_get_and_watch_prefix functionality
        let etcd_client = component
            .drt()
            .etcd_client()
            .expect("Cannot KV route without etcd client");

        let runtime_configs_watcher = watch_prefix_with_extraction(
            etcd_client,
            MODEL_ROOT_PATH,
            key_extractors::lease_id,
            |model_entry: ModelEntry| model_entry.runtime_config,
            cancellation_token.clone(),
        )
        .await?;
        let runtime_configs_rx = runtime_configs_watcher.receiver();

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
            ))
        };
        let indexer = Arc::new(indexer);

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
        if let Indexer::KvIndexer(kv_indexer) = &*indexer {
            start_kv_router_background(
                component.clone(),
                consumer_uuid,
                kv_indexer.event_sender(),
                kv_indexer.remove_worker_sender(),
                kv_router_config
                    .router_snapshot_threshold
                    .map(|_| kv_indexer.snapshot_event_sender()),
                cancellation_token.clone(),
                kv_router_config.router_snapshot_threshold,
                kv_router_config.router_reset_states,
            )
            .await?;
        }

        // NEW NATS SERVICE: Enable decode workers to request intelligent prefill routing
        //
        // ARCHITECTURE DECISION: Start NATS service unconditionally to support multiple deployment patterns:
        // 1. Single-process: Decode workers and router in same process (direct NATS communication)
        // 2. Multi-process: Router as separate service (cross-process NATS communication)
        // 3. Distributed: Router and workers on different nodes (distributed NATS communication)
        //
        // This service listens for prefill selection requests and responds with intelligent
        // routing decisions based on KV cache overlap and load balancing analysis.
        Self::start_prefill_selection_service(
            component.clone(),
            indexer.clone(),
            scheduler.clone(),
            block_size,
            cancellation_token.clone(),
        );

        tracing::info!("KV Routing initialized with NATS prefill selection service");
        Ok(Self {
            indexer,
            scheduler,
            block_size,
            kv_router_config,
        })
    }

    /// Give these tokens, find the worker with the best match in it's KV cache.
    /// Returned overlap amount is in number of blocks.
    /// Now also takes context_id for request tracking
    async fn find_best_match(
        &self,
        context_id: &str,
        tokens: &[u32],
        router_config_override: Option<&RouterConfigOverride>,
        update_states: bool,
    ) -> anyhow::Result<(i64, u32)> {
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

        let best_worker_id = self
            .scheduler
            .schedule(
                context_id.to_string(),
                isl_tokens,
                maybe_seq_hashes_2,
                overlap_scores.clone(),
                router_config_override,
                update_states,
            )
            .await?;

        if let Indexer::ApproxKvIndexer(indexer) = &*self.indexer {
            indexer
                .process_routing_decision(best_worker_id, block_hashes, maybe_seq_hashes_1.unwrap())
                .await
                .unwrap();
        };

        let overlap_amount = overlap_scores
            .scores
            .get(&best_worker_id)
            .copied()
            .unwrap_or(0);
        Ok((best_worker_id, overlap_amount))
    }

    pub async fn add_request(
        &self,
        request_id: String,
        tokens: &[u32],
        overlap_blocks: u32,
        worker_id: i64,
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
                worker_id,
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

    /// Select the best prefill worker for the given tokens using the same intelligent
    /// routing logic as decode worker selection.
    ///
    /// This method applies KV cache overlap detection and load balancing to choose
    /// the optimal prefill worker for remote prefill requests. It reuses the same
    /// indexer and scheduler infrastructure as decode worker selection.
    ///
    /// # Purpose
    /// When a decode worker decides to use remote prefill (via disagg_router), it calls
    /// this method to get an intelligent prefill worker selection instead of using
    /// random NATS queue selection.
    ///
    /// # Arguments
    /// * `context_id` - Unique identifier for this request (for tracking and debugging)
    /// * `tokens` - Input tokens that need prefill processing
    /// * `router_config_override` - Optional per-request routing configuration
    ///
    /// # Returns
    /// * `Ok(worker_id)` - ID of the selected prefill worker
    /// * `Err(KvRouterError)` - If no prefill workers available or selection fails
    ///
    /// # Example Usage
    /// ```ignore
    /// // In decode worker handler, when remote prefill is needed:
    /// if disagg_router.prefill_remote(prefill_length, prefix_hit_length) {
    ///     let prefill_worker_id = kv_router.select_prefill_worker(
    ///         &request_id,
    ///         &request.token_ids,
    ///         None
    ///     ).await?;
    ///     send_to_prefill_worker(prefill_worker_id, request).await;
    /// }
    /// ```
    pub async fn select_prefill_worker(
        &self,
        context_id: &str,
        tokens: &[u32],
        router_config_override: Option<&RouterConfigOverride>,
    ) -> anyhow::Result<i64> {
        let (best_prefill_worker_id, overlap_amount) = Self::select_prefill_worker_core(
            &self.indexer,
            &self.scheduler,
            self.block_size,
            context_id,
            tokens,
            router_config_override,
        )
        .await?;

        tracing::info!(
            "Selected prefill worker {} for request {} with {} cached blocks out of {} total blocks ({}% cache hit)",
            best_prefill_worker_id,
            context_id,
            overlap_amount,
            tokens.len().div_ceil(self.block_size as usize),
            if !tokens.is_empty() {
                (overlap_amount as f64 * self.block_size as f64 / tokens.len() as f64 * 100.0)
                    as u32
            } else {
                0
            }
        );

        Ok(best_prefill_worker_id)
    }

    /// Dump all events from the indexer
    pub async fn dump_events(&self) -> Result<Vec<RouterEvent>, KvRouterError> {
        self.indexer.dump_events().await
    }

    /// Start NATS service for intelligent prefill worker selection requests.
    ///
    /// SERVICE ARCHITECTURE: This background service implements the server side of the
    /// NATS request-response pattern for prefill worker selection. It processes requests
    /// from decode workers and responds with intelligent routing decisions.
    ///
    /// REQUEST PROCESSING FLOW:
    /// 1. Listen on "kv_router.select_prefill_worker" for selection requests
    /// 2. Parse request containing token IDs and request metadata
    /// 3. Apply full KV cache overlap detection and load balancing logic
    /// 4. Respond on "kv_router.response.{request_id}" with selected worker ID
    ///
    /// DESIGN DECISION: Uses the same core selection logic as direct API calls
    /// (select_prefill_worker_core) to ensure consistency between NATS and direct routing.
    /// This provides full intelligent routing capabilities via NATS communication.
    ///
    /// DEPLOYMENT FLEXIBILITY: Service starts unconditionally to support various
    /// deployment patterns without requiring separate router component configuration.
    fn start_prefill_selection_service(
        component: Component,
        indexer: Arc<Indexer>,
        scheduler: KvScheduler,
        block_size: u32,
        cancellation_token: tokio_util::sync::CancellationToken,
    ) {
        tracing::debug!("Starting NATS prefill selection service task");
        tokio::spawn(async move {
            // Subscribe on the component-scoped bus to match publisher side
            let mut selection_rx = match component
                .subscribe("kv_router.select_prefill_worker")
                .await
            {
                Ok(rx) => {
                    tracing::debug!(
                        "Successfully subscribed to kv_router.select_prefill_worker (component bus)"
                    );
                    rx
                }
                Err(e) => {
                    tracing::error!("Failed to subscribe to prefill selection requests: {}", e);
                    return;
                }
            };

            tracing::info!(
                "KV router NATS prefill selection service started on subject 'kv_router.select_prefill_worker'"
            );

            // Process incoming prefill selection requests (similar to existing subscriber patterns)
            loop {
                tracing::debug!(
                    "NATS prefill selection service loop iteration - waiting for events"
                );
                tokio::select! {
                    _ = cancellation_token.cancelled() => {
                        tracing::info!("Prefill selection service shutting down");
                        break;
                    }
                    event_result = selection_rx.next() => {
                        tracing::debug!("NATS selection_rx.next() returned");
                        match event_result {
                            Some(event) => {
                                tracing::debug!("Received NATS prefill selection event, payload size: {} bytes", event.payload.len());
                                // Parse the selection request (similar to existing event parsing)
                                match serde_json::from_slice::<serde_json::Value>(&event.payload) {
                                    Ok(request_data) => {
                                        let request_id = request_data["request_id"].as_str().unwrap_or("unknown");
                                        let token_ids: Vec<u32> = match request_data.get("token_ids").and_then(|v| v.as_array()) {
                                            Some(arr) => arr.iter().filter_map(|v| v.as_u64().map(|n| n as u32)).collect(),
                                            None => {
                                                tracing::warn!("Prefill selection request missing token_ids; request_id={}", request_id);
                                                Vec::new()
                                            }
                                        };
                                        if token_ids.is_empty() {
                                            tracing::warn!("Empty token_ids in prefill selection request; skipping. request_id={}", request_id);
                                            continue;
                                        }

                                        tracing::debug!("Received NATS prefill selection request for {} tokens, request_id: {}", token_ids.len(), request_id);

                                        // CORE ROUTING LOGIC: Apply full intelligent selection algorithm
                                        // This uses the same logic as direct API calls to ensure consistency:
                                        // 1. Compute block hashes from tokens for cache overlap detection
                                        // 2. Query indexer for overlapping cached blocks across prefill workers
                                        // 3. Apply scheduler with prefill-specific cost function and filtering
                                        // 4. Return optimal worker ID based on cache overlap and load balancing
                                        match KvRouter::select_prefill_worker_core(
                                            &indexer,
                                            &scheduler,
                                            block_size,
                                            &format!("prefill_nats_{}", request_id),
                                            &token_ids,
                                            None,  // Use default router configuration
                                        ).await {
                                            Ok((prefill_worker_id, _overlap)) => {
                                                // Send response back via NATS (component bus)
                                                let response = serde_json::json!({
                                                    "prefill_worker_id": prefill_worker_id,
                                                    "request_id": request_id,
                                                    "timestamp": std::time::SystemTime::now()
                                                        .duration_since(std::time::UNIX_EPOCH)
                                                        .unwrap()
                                                        .as_secs()
                                                });

                                                let response_subject = format!("kv_router.response.{}", request_id);
                                                tracing::debug!("Sending prefill selection response to subject: {} with worker_id: {}", response_subject, prefill_worker_id);
                                                if let Err(e) = component.publish(&response_subject, &response).await {
                                                    tracing::error!("Failed to send prefill selection response: {}", e);
                                                } else {
                                                    tracing::debug!("Successfully sent prefill selection response");
                                                }
                                            }
                                            Err(e) => {
                                                tracing::error!("Prefill worker selection failed for request {}: {}", request_id, e);
                                                // Send error response (following existing error handling patterns)
                                                let error_response = serde_json::json!({
                                                    "error": format!("Selection failed: {}", e),
                                                    "request_id": request_id,
                                                    "timestamp": std::time::SystemTime::now()
                                                        .duration_since(std::time::UNIX_EPOCH)
                                                        .unwrap()
                                                        .as_secs()
                                                });
                                                let response_subject = format!("kv_router.response.{}", request_id);
                                                let _ = component.publish(&response_subject, &error_response).await;
                                            }
                                        }
                                    }
                                    Err(e) => {
                                        tracing::error!("Failed to parse prefill selection request: {}", e);
                                    }
                                }
                            }
                            None => {
                                tracing::warn!("NATS prefill selection subscription ended");
                                break;
                            }
                        }
                    }
                }
            }

            tracing::info!("Prefill selection NATS service stopped");
        });
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
                let (worker_id, overlap_blocks) = self
                    .find_best_match(&context_id, &tokens, None, true)
                    .await?;

                RouterResponse::New {
                    worker_id,
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
    chooser: Arc<KvRouter>,
}

impl KvPushRouter {
    pub fn new(
        inner: PushRouter<PreprocessedRequest, Annotated<LLMEngineOutput>>,
        chooser: Arc<KvRouter>,
    ) -> Self {
        KvPushRouter { inner, chooser }
    }

    /// Find the best matching worker for the given tokens without updating states
    pub async fn find_best_match(
        &self,
        context_id: &str,
        tokens: &[u32],
        router_config_override: Option<&RouterConfigOverride>,
    ) -> Result<(i64, u32)> {
        self.chooser
            .find_best_match(context_id, tokens, router_config_override, false)
            .await
    }

    /// Get potential prefill and decode loads for all workers
    pub async fn get_potential_loads(&self, tokens: &[u32]) -> Result<Vec<PotentialLoad>> {
        self.chooser.get_potential_loads(tokens).await
    }

    /// Select the best prefill worker for the given tokens using intelligent routing.
    ///
    /// This method delegates to the underlying KvRouter to apply KV cache overlap
    /// detection and load balancing specifically for prefill worker selection.
    /// It's used by decode workers when they need to route requests to prefill workers.
    ///
    /// # Arguments
    /// * `context_id` - Unique identifier for this request (for tracking)
    /// * `tokens` - Input tokens that need prefill processing
    /// * `router_config_override` - Optional per-request routing configuration
    ///
    /// # Returns
    /// * `Ok(worker_id)` - ID of the selected prefill worker
    /// * `Err(anyhow::Error)` - If no prefill workers available or selection fails
    pub async fn select_prefill_worker(
        &self,
        context_id: &str,
        tokens: &[u32],
        router_config_override: Option<&RouterConfigOverride>,
    ) -> Result<i64> {
        self.chooser
            .select_prefill_worker(context_id, tokens, router_config_override)
            .await
    }

    /// Dump all events from the KV router's indexer
    pub async fn dump_events(&self) -> Result<Vec<RouterEvent>, KvRouterError> {
        self.chooser.dump_events().await
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
        match self.inner.client.instance_source.as_ref() {
            InstanceSource::Static => self.inner.r#static(request).await,
            InstanceSource::Dynamic(_) => {
                // Extract context ID for request tracking
                let context_id = request.context().id().to_string();

                // Check if this is a query_instance_id request first
                let query_instance_id = request.has_annotation("query_instance_id");

                let (instance_id, overlap_amount) = if let Some(id) = request.backend_instance_id {
                    // If instance_id is set, use it and manually add the request to track it
                    if !query_instance_id {
                        self.chooser
                            .add_request(context_id.clone(), &request.token_ids, 0, id)
                            .await;
                    }
                    (id, 0)
                } else {
                    // Otherwise, find the best match
                    self.chooser
                        .find_best_match(
                            &context_id,
                            &request.token_ids,
                            request.router_config_override.as_ref(),
                            !query_instance_id, // Don't update states if query_instance_id
                        )
                        .await?
                };

                // if request has the annotation "query_instance_id",
                // then the request will not be routed to the worker,
                // and instead the worker_instance_id will be returned.
                let stream_context = request.context().clone();
                if query_instance_id {
                    let instance_id_str = instance_id.to_string();
                    let response =
                        Annotated::from_annotation("worker_instance_id", &instance_id_str)?;

                    // Return the tokens in nvext.token_data format
                    let response_tokens =
                        Annotated::from_annotation("token_data", &request.token_ids)?;
                    tracing::trace!(
                        "Tokens requested in the response through the query_instance_id annotation: {:?}",
                        response_tokens
                    );
                    let stream = stream::iter(vec![response, response_tokens]);
                    return Ok(ResponseStream::new(Box::pin(stream), stream_context));
                }
                let (mut backend_input, context) = request.into_parts();
                backend_input.estimated_prefix_hit_num_blocks = Some(overlap_amount);
                let updated_request = context.map(|_| backend_input);

                let mut response_stream = self.inner.direct(updated_request, instance_id).await?;
                let stream_context = response_stream.context();
                let chooser = self.chooser.clone();

                let wrapped_stream = Box::pin(async_stream::stream! {
                    if let Some(first_item) = response_stream.next().await {
                        if let Err(e) = chooser.mark_prefill_completed(&context_id).await {
                            tracing::warn!("Failed to mark prefill completed for request {context_id}: {e:?}");
                        }
                        yield first_item;
                    }

                    while let Some(item) = response_stream.next().await {
                        yield item;
                    }

                    if let Err(e) = chooser.free(&context_id).await {
                        tracing::warn!("Failed to free request {context_id}: {e:?}");
                    }
                });
                Ok(ResponseStream::new(wrapped_stream, stream_context))
            }
        }
    }
}
