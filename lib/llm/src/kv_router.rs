// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{collections::HashSet, sync::Arc, time::Instant};

use anyhow::Result;
use dynamo_kv_router::{
    KvSchedulerError, PrefillLoadEstimator, SharedKvCache,
    config::{KvRouterConfig, RouterConfigOverride, min_initial_workers_from_env},
    indexer::{KvRouterError, RoutingDecisionHashes},
    protocols::KV_EVENT_SUBJECT,
    protocols::{
        BlockExtraInfo, BlockHashOptions, LocalBlockHash, PrefillLoadHint, RouterEvent,
        RouterRequest, RouterResponse, RoutingConstraints, TokensWithHashes, WorkerConfigLike,
        WorkerId, WorkerWithDpRank, compute_block_hash_for_seq,
    },
    scheduling::{
        CacheHitEstimates, FencedWorkerProvider, OverlapAnalysis, OverloadedWorkerProvider,
        RequestLifecycleLease, RequestProgressUpdater, ScheduleMode, ScheduleRequest,
        TieredOverlapRefresher, effective_prefill_tokens,
        overlap::cache_hit_estimates_from_tiered_matches,
    },
};
use dynamo_runtime::{
    CancellationToken,
    component::{Client, Endpoint},
    discovery::DiscoveryQuery,
    error::{DynamoError, ErrorType},
    pipeline::{
        AsyncEngine, AsyncEngineContextProvider, Error, ManyOut, ResponseStream, SingleIn,
        async_trait, error::PipelineError,
    },
    protocols::EndpointId,
    protocols::annotated::Annotated,
    traits::DistributedRuntimeProvider,
};
use futures::stream;
use tracing::Instrument;

// Re-export from dynamo-kv-router crate
pub use dynamo_kv_router::approx;
pub use dynamo_kv_router::protocols;
pub use dynamo_kv_router::scheduling;
pub use dynamo_kv_router::selector;

pub mod encoder_router;
pub mod indexer;
pub mod metrics;
pub mod prefill_router;
pub mod publisher;
pub mod push_router;
mod route_lookup;
pub mod scheduler;
pub mod sequence;
pub mod shared_cache;

pub use dynamo_kv_router::scheduling::{
    OverlapScoresResponse, SharedCacheOverlapScore, WorkerOverlapScore,
};
pub use encoder_router::EncoderRouter;
pub use indexer::{Indexer, ServedIndexerHandle, ServedIndexerMode, ensure_served_indexer_service};
pub use prefill_router::PrefillRouter;
pub use push_router::{DirectRoutingRouter, KvPushRouter};

use crate::{
    discovery::{KvSourceMembershipWatch, RuntimeConfigWatch},
    kv_router::{
        scheduler::{DefaultWorkerSelector, KvScheduler, PotentialLoad},
        sequence::{SequenceError, SequenceRequest},
    },
    local_model::runtime_config::ModelRuntimeConfig,
};
use route_lookup::{TieredLookupResult, query_tiered_matches, split_retained_block_hashes};

pub enum FindBestMatchOutcome {
    Routed {
        worker: WorkerWithDpRank,
        overlap_blocks: u32,
        effective_overlap_blocks: f64,
        cached_tokens: usize,
        routing_hashes: Option<RoutingDecisionHashes>,
    },
    QueueRejected {
        rejection: scheduling::QueueRejection,
    },
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct WorkerCacheHitEstimate {
    pub effective_overlap_blocks: f64,
}

impl WorkerCacheHitEstimate {
    pub fn rounded_overlap_blocks(self) -> u32 {
        self.effective_overlap_blocks.round() as u32
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
    }
}

// [gluo TODO] shouldn't need to be public
// this should be discovered from the component

// for metric scraping (pull-based)
pub const KV_METRICS_ENDPOINT: &str = "load_metrics";

// for metric publishing (push-based)
pub const KV_METRICS_SUBJECT: &str = "kv_metrics";
pub const MULTIMODAL_EMBEDDING_CACHE_SUBJECT: &str = "multimodal_embedding_cache";

// for inter-router comms
pub const PREFILL_SUBJECT: &str = "prefill_events";
pub const ACTIVE_SEQUENCES_SUBJECT: &str = "active_sequences_events";

// for radix tree snapshot storage
pub const RADIX_STATE_BUCKET: &str = "radix-bucket";
pub const RADIX_STATE_FILE: &str = "radix-state";

// for worker-local kvindexer query
pub const WORKER_KV_INDEXER_BUFFER_SIZE: usize = 1024; // store 1024 most recent events in worker buffer

fn map_scheduler_error(error: scheduling::KvSchedulerError) -> anyhow::Error {
    if !error.is_overload() {
        return error.into();
    }

    let message = error.to_string();
    let cause = PipelineError::ServiceOverloaded(message.clone());
    DynamoError::builder()
        .error_type(ErrorType::ResourceExhausted)
        .message(message)
        .cause(cause)
        .build()
        .into()
}

fn cancelled_error(context_id: &str) -> anyhow::Error {
    DynamoError::builder()
        .error_type(ErrorType::Cancelled)
        .message(format!("Request {context_id} was cancelled"))
        .build()
        .into()
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
    scheduler: KvScheduler<Sel, TieredOverlapRefresher<Indexer>>,
    workers_with_configs: RuntimeConfigWatch,
    block_size: u32,
    kv_router_config: KvRouterConfig,
    prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
    cancellation_token: CancellationToken,
    client: Client,
    is_eagle: bool,
    _served_indexer_handle: Option<ServedIndexerHandle>,
    /// Optional external shared KV cache pool. When present, `find_best_match`
    /// queries it in parallel with the indexer and factors shared hits into scoring.
    shared_cache: Option<Box<dyn SharedKvCache>>,
    /// Optional LoRA filter. When present (LoRA serving enabled), candidate workers are
    /// narrowed to the LoRA's allocated/loaded replicas inside `find_best_match_details`,
    /// covering both the decode and prefill routers (both built via `kv_chooser_for`).
    lora_filter: Option<Arc<crate::lora::LoraFilter>>,
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
        kv_source_membership: Option<KvSourceMembershipWatch>,
        block_size: u32,
        selector: Sel,
        kv_router_config: Option<KvRouterConfig>,
        prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
        worker_type: &'static str,
        model_name: Option<String>,
        is_eagle: bool,
        shared_cache: Option<Box<dyn SharedKvCache>>,
        lora_filter: Option<Arc<crate::lora::LoraFilter>>,
    ) -> Result<Self> {
        let kv_router_config = kv_router_config.unwrap_or_default();
        kv_router_config.validate().map_err(anyhow::Error::msg)?;
        let component = endpoint.component();
        // Router-owned tasks derive from this token so a rebuild cannot cancel the runtime.
        let cancellation_token = component.drt().child_token();
        let cancellation_guard = cancellation_token.clone().drop_guard();
        let min_initial_workers = min_initial_workers_from_env()?;

        let indexer = Indexer::new(
            component,
            &kv_router_config,
            block_size,
            model_name.as_deref(),
            cancellation_token.child_token(),
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

        let overlap_scores_refresh = indexer.supports_overlap_refresh().then(|| {
            Arc::new(TieredOverlapRefresher::new(
                indexer.clone(),
                kv_router_config.clone(),
                block_size,
            ))
        });
        let client_for_overload = client.clone();
        let overloaded_worker_provider: OverloadedWorkerProvider =
            Arc::new(move || client_for_overload.overloaded_instance_ids());

        // DIS-2404: fenced (dead) workers, populated by the worker monitor on
        // discovery removal and read at selection to reject them everywhere.
        let client_for_fence = client.clone();
        let fenced_worker_provider: FencedWorkerProvider =
            Arc::new(move || client_for_fence.fenced_instance_ids());

        let scheduler = KvScheduler::start(
            endpoint.clone(),
            block_size,
            workers_with_configs.clone(),
            selector,
            &kv_router_config,
            prefill_load_estimator.clone(),
            overlap_scores_refresh,
            Some(overloaded_worker_provider),
            Some(fenced_worker_provider),
            model_name.as_deref(),
            worker_type,
            cancellation_token.child_token(),
            Default::default(),
        )
        .await?;

        // Start KV event subscription if needed — skip when using a remote indexer.
        if kv_router_config.use_remote_indexer {
            tracing::info!("Skipping KV event subscription (using remote indexer)");
        } else if kv_router_config.should_subscribe_to_kv_events() {
            let membership_watch = kv_source_membership.ok_or_else(|| {
                anyhow::anyhow!(
                    "KV source membership watch is required when local KV event subscription is enabled"
                )
            })?;
            indexer::start_subscriber(
                endpoint.clone(),
                indexer.clone(),
                membership_watch,
                model_name.clone().unwrap_or_else(|| "unknown".to_string()),
                worker_type,
                cancellation_token.child_token(),
            )
            .await?;
        } else {
            tracing::info!(
                "Skipping KV event subscription (use_kv_events={}, overlap_score_credit={})",
                kv_router_config.use_kv_events,
                kv_router_config.overlap_score_credit,
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
        let cancellation_token = cancellation_guard.disarm();
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
            lora_filter,
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

    pub(crate) async fn record_routing_decision_hashes(
        &self,
        hashes: RoutingDecisionHashes,
        worker: WorkerWithDpRank,
    ) -> Result<(), KvRouterError> {
        self.indexer
            .record_routing_decision_hashes(worker, hashes)
            .await
    }

    /// Narrow the candidate workers to this LoRA's allocated/loaded replicas, staying strictly
    /// within the existing candidate universe (never widening). Returns the (possibly narrowed)
    /// `allowed_worker_ids` to pass to the scheduler.
    ///
    /// - No filter (LoRA serving disabled) or base-model request (`lora_name` is `None`):
    ///   returns `allowed_worker_ids` unchanged.
    /// - Pinned worker: KV-cache correctness wins — it is always retained even if not in the
    ///   LoRA replica set (the worker lazy-loads the adapter).
    /// - If narrowing would exclude every candidate, falls back to the original set so the
    ///   request stays routable (lazy-load path) rather than failing.
    fn narrow_allowed_by_lora(
        &self,
        lora_name: Option<&str>,
        allowed_worker_ids: Option<HashSet<WorkerId>>,
        pinned_worker: Option<&WorkerWithDpRank>,
    ) -> Option<HashSet<WorkerId>> {
        let (Some(filter), Some(lora_name)) = (self.lora_filter.as_ref(), lora_name) else {
            return allowed_worker_ids;
        };
        // Base candidate universe: explicit allow-set if present, else all current workers.
        let base: Vec<WorkerId> = match &allowed_worker_ids {
            Some(allowed) => allowed.iter().copied().collect(),
            None => self.workers_with_configs.borrow().keys().copied().collect(),
        };
        if base.is_empty() {
            return allowed_worker_ids;
        }
        let mut narrowed: HashSet<WorkerId> = filter
            .filter_worker_ids_for_lora(Some(lora_name), &base)
            .into_iter()
            .collect();
        // Retain a pinned worker only if it is already within the candidate universe — never
        // widen the caller's `allowed_worker_ids` (KV-cache / EPP / migration invariants depend
        // on that set). If the filter excluded an in-universe pinned worker, re-add it so the
        // pin still wins for cache correctness; if the pin is outside the universe, honor the
        // caller's constraint and drop it.
        if let Some(p) = pinned_worker
            && base.contains(&p.worker_id)
        {
            narrowed.insert(p.worker_id);
        }
        if narrowed.is_empty() {
            return allowed_worker_ids;
        }
        Some(narrowed)
    }

    /// Give these tokens, find the worker with the best weighted cache hit.
    /// Returns the full match details for the selected worker.
    ///
    /// When `pinned_worker` is Some, scheduling and queueing are constrained to
    /// that exact worker/rank.
    ///
    /// When `allowed_worker_ids` is Some, only workers in that set are considered for selection.
    #[allow(clippy::too_many_arguments)]
    pub async fn find_best_match_details(
        &self,
        context_id: Option<&str>,
        tokens: &[u32],
        block_mm_infos: Option<&[Option<BlockExtraInfo>]>,
        router_config_override: Option<&RouterConfigOverride>,
        update_states: bool,
        return_routing_hashes: bool,
        lora_name: Option<String>,
        cache_namespace: Option<String>,
        priority_jump: f64,
        strict_priority: u32,
        expected_output_tokens: Option<u32>,
        pinned_worker: Option<WorkerWithDpRank>,
        allowed_worker_ids: Option<HashSet<WorkerId>>,
        routing_constraints: RoutingConstraints,
    ) -> anyhow::Result<FindBestMatchOutcome> {
        self.find_best_match_details_with_policy_class(
            context_id,
            tokens,
            block_mm_infos,
            router_config_override,
            update_states,
            return_routing_hashes,
            lora_name,
            cache_namespace,
            priority_jump,
            strict_priority,
            None,
            None,
            expected_output_tokens,
            pinned_worker,
            allowed_worker_ids,
            routing_constraints,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn find_best_match_details_with_policy_class(
        &self,
        context_id: Option<&str>,
        tokens: &[u32],
        block_mm_infos: Option<&[Option<BlockExtraInfo>]>,
        router_config_override: Option<&RouterConfigOverride>,
        update_states: bool,
        return_routing_hashes: bool,
        lora_name: Option<String>,
        cache_namespace: Option<String>,
        priority_jump: f64,
        strict_priority: u32,
        policy_class: Option<String>,
        session_id: Option<String>,
        expected_output_tokens: Option<u32>,
        pinned_worker: Option<WorkerWithDpRank>,
        allowed_worker_ids: Option<HashSet<WorkerId>>,
        routing_constraints: RoutingConstraints,
    ) -> anyhow::Result<FindBestMatchOutcome> {
        self.find_best_match_details_with_policy_class_inner(
            context_id,
            tokens,
            block_mm_infos,
            router_config_override,
            update_states,
            return_routing_hashes,
            lora_name,
            cache_namespace,
            priority_jump,
            strict_priority,
            policy_class,
            session_id,
            expected_output_tokens,
            pinned_worker,
            allowed_worker_ids,
            routing_constraints,
            false,
        )
        .await
        .map(|(outcome, _)| outcome)
    }

    #[allow(clippy::too_many_arguments)]
    async fn find_best_match_details_with_policy_class_inner(
        &self,
        context_id: Option<&str>,
        tokens: &[u32],
        block_mm_infos: Option<&[Option<BlockExtraInfo>]>,
        router_config_override: Option<&RouterConfigOverride>,
        update_states: bool,
        return_routing_hashes: bool,
        lora_name: Option<String>,
        cache_namespace: Option<String>,
        priority_jump: f64,
        strict_priority: u32,
        policy_class: Option<String>,
        session_id: Option<String>,
        expected_output_tokens: Option<u32>,
        pinned_worker: Option<WorkerWithDpRank>,
        allowed_worker_ids: Option<HashSet<WorkerId>>,
        routing_constraints: RoutingConstraints,
        track_lifecycle: bool,
    ) -> anyhow::Result<(
        FindBestMatchOutcome,
        Option<(RequestProgressUpdater, RequestLifecycleLease)>,
    )> {
        let start = Instant::now();

        if update_states && context_id.is_none() {
            anyhow::bail!("context_id must be provided if update_states is true");
        }
        let mode = if update_states && track_lifecycle {
            ScheduleMode::TrackedWithLifecycle {
                request_id: context_id.expect("validated above").to_string(),
            }
        } else if update_states {
            ScheduleMode::Tracked {
                request_id: context_id.expect("validated above").to_string(),
            }
        } else {
            ScheduleMode::QueryOnly {
                request_id: context_id.map(str::to_string),
            }
        };

        let isl_tokens = tokens.len();
        let hash_options = BlockHashOptions {
            block_mm_infos,
            lora_name: lora_name.as_deref(),
            cache_namespace: cache_namespace.as_deref(),
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

        let supports_overlap_refresh = self.scheduler.supports_overlap_refresh();
        let retain_block_hashes = supports_overlap_refresh || return_routing_hashes;

        let TieredLookupResult {
            tiered_matches,
            shared_cache_hits,
            indexer_duration,
            shared_cache_duration,
            retained_block_hashes,
        } = query_tiered_matches(
            &self.indexer,
            self.shared_cache.as_deref(),
            tokens,
            self.block_size,
            block_hashes,
            cache_namespace.as_deref(),
            retain_block_hashes,
        )
        .await?;

        let (block_hashes_for_refresh, routing_block_hashes) = retained_block_hashes
            .map(|block_hashes| {
                split_retained_block_hashes(
                    block_hashes,
                    supports_overlap_refresh,
                    return_routing_hashes,
                )
            })
            .unwrap_or((None, None));

        let overlap =
            OverlapAnalysis::new(&self.kv_router_config, self.block_size, &tiered_matches)
                .signals();
        drop(tiered_matches);
        let find_matches_elapsed = start.elapsed();

        // Capture shared cache info for metrics before moving into schedule().
        // Clone the hits so we can compute `hits_beyond(overlap_blocks)` after
        // scheduling returns, since `overlap_blocks` isn't known until then.
        let num_blocks = isl_tokens / self.block_size as usize;
        let sc_hits_for_metrics = shared_cache_hits.clone();

        // LoRA-aware candidate narrowing: restrict to this LoRA's allocated/loaded replicas,
        // strictly within the existing candidate universe (never widening). Covers both the
        // decode and prefill routers, since both flow through this method.
        let allowed_worker_ids = self.narrow_allowed_by_lora(
            lora_name.as_deref(),
            allowed_worker_ids,
            pinned_worker.as_ref(),
        );

        let response = match self
            .scheduler
            .schedule_request(ScheduleRequest {
                mode,
                token_seq: maybe_seq_hashes,
                block_hashes: block_hashes_for_refresh,
                isl_tokens,
                overlap,
                router_config_override: router_config_override.cloned(),
                lora_name,
                priority_jump,
                strict_priority,
                policy_class,
                session_id,
                expected_output_tokens,
                pinned_worker,
                allowed_worker_ids,
                routing_constraints,
                shared_cache_hits,
            })
            .instrument(tracing::info_span!("kv_router.schedule"))
            .await
        {
            Ok(response) => response,
            Err(KvSchedulerError::QueueRejected(rejection)) => {
                return Ok((FindBestMatchOutcome::QueueRejected { rejection }, None));
            }
            Err(error) => return Err(map_scheduler_error(error)),
        };
        let total_elapsed = start.elapsed();
        let routing_hashes = routing_block_hashes.map(RoutingDecisionHashes::from_local_hashes);

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

        debug_assert_eq!(
            response.request_progress.is_some(),
            response.lifecycle_lease.is_some()
        );
        let lifecycle = response.request_progress.zip(response.lifecycle_lease);
        Ok((
            FindBestMatchOutcome::Routed {
                worker: response.best_worker,
                overlap_blocks: response.effective_overlap_blocks.round() as u32,
                effective_overlap_blocks: response.effective_overlap_blocks,
                cached_tokens: response.cached_tokens,
                routing_hashes,
            },
            lifecycle,
        ))
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
        cache_namespace: Option<String>,
        priority_jump: f64,
        strict_priority: u32,
        expected_output_tokens: Option<u32>,
        allowed_worker_ids: Option<HashSet<WorkerId>>,
        routing_constraints: RoutingConstraints,
    ) -> anyhow::Result<(WorkerWithDpRank, u32)> {
        let result = self
            .find_best_match_details(
                context_id,
                tokens,
                block_mm_infos,
                router_config_override,
                update_states,
                false,
                lora_name,
                cache_namespace,
                priority_jump,
                strict_priority,
                expected_output_tokens,
                None,
                allowed_worker_ids,
                routing_constraints,
            )
            .await?;
        match result {
            FindBestMatchOutcome::Routed {
                worker,
                overlap_blocks,
                ..
            } => Ok((worker, overlap_blocks)),
            FindBestMatchOutcome::QueueRejected { rejection } => Err(rejection.into()),
        }
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
        cache_namespace: Option<String>,
        router_config_override: Option<&RouterConfigOverride>,
    ) {
        let isl_tokens = tokens.len();
        let hash_options = BlockHashOptions {
            block_mm_infos,
            lora_name: lora_name.as_deref(),
            cache_namespace: cache_namespace.as_deref(),
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

    /// Legacy slot cleanup. Lifecycle-tracked requests use their `RequestLifecycleLease`.
    pub async fn free(&self, request_id: &str) -> Result<(), SequenceError> {
        self.scheduler.free(request_id).await
    }

    /// Number of requests currently parked in the scheduler queue.
    pub fn pending_count(&self) -> usize {
        self.scheduler.pending_count()
    }

    /// Sum of ISL tokens for requests currently parked in the scheduler queue.
    pub fn pending_isl_tokens(&self) -> usize {
        self.scheduler.pending_isl_tokens()
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

        let effective_isl = effective_prefill_tokens(isl_tokens, cached_tokens);
        if effective_isl == 0 {
            return None;
        }
        let prefix = isl_tokens - effective_isl;

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
        cache_namespace: Option<&str>,
    ) -> Result<u32, KvRouterError> {
        Ok(self
            .get_cache_hit_estimate(tokens, block_mm_infos, worker, lora_name, cache_namespace)
            .await?
            .rounded_overlap_blocks())
    }

    pub(crate) async fn get_cache_hit_estimate(
        &self,
        tokens: &[u32],
        block_mm_infos: Option<&[Option<BlockExtraInfo>]>,
        worker: WorkerWithDpRank,
        lora_name: Option<&str>,
        cache_namespace: Option<&str>,
    ) -> Result<WorkerCacheHitEstimate, KvRouterError> {
        self.get_cache_hit_estimate_with_hashes(
            tokens,
            block_mm_infos,
            worker,
            lora_name,
            cache_namespace,
            false,
        )
        .await
        .map(|(estimate, _)| estimate)
    }

    pub(crate) async fn get_cache_hit_estimate_with_hashes(
        &self,
        tokens: &[u32],
        block_mm_infos: Option<&[Option<BlockExtraInfo>]>,
        worker: WorkerWithDpRank,
        lora_name: Option<&str>,
        cache_namespace: Option<&str>,
        return_routing_hashes: bool,
    ) -> Result<(WorkerCacheHitEstimate, Option<RoutingDecisionHashes>), KvRouterError> {
        let block_hashes = compute_block_hash_for_seq(
            tokens,
            self.block_size,
            BlockHashOptions {
                block_mm_infos,
                lora_name,
                cache_namespace,
                is_eagle: Some(self.is_eagle),
            },
        );
        let (tiered_matches, routing_hashes) = if return_routing_hashes {
            let tiered_matches = self.indexer.find_matches_by_tier_ref(&block_hashes).await?;
            (
                tiered_matches,
                Some(RoutingDecisionHashes::from_local_hashes(block_hashes)),
            )
        } else {
            (self.indexer.find_matches_by_tier(block_hashes).await?, None)
        };
        let cache_hit_estimates = self.cache_hit_estimates_from_tiered_matches(&tiered_matches);
        Ok((
            self.cache_hit_for_worker(&cache_hit_estimates, worker),
            routing_hashes,
        ))
    }

    /// Get potential prefill and decode loads for all workers
    pub async fn get_potential_loads(
        &self,
        tokens: &[u32],
        router_config_override: Option<&RouterConfigOverride>,
        block_mm_infos: Option<&[Option<BlockExtraInfo>]>,
        lora_name: Option<&str>,
        cache_namespace: Option<&str>,
    ) -> Result<Vec<PotentialLoad>> {
        let isl_tokens = tokens.len();
        let hash_options = BlockHashOptions {
            block_mm_infos,
            lora_name,
            cache_namespace,
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
            cache_hit_estimates.cached_tokens.into_iter().collect(),
            track_prefill_tokens,
        ))
    }

    /// Return per-worker KV overlap by storage tier.
    ///
    /// Device, host-pinned, and disk values are keyed by `(worker_id, dp_rank)`.
    /// Shared-cache hits are global to the request, so each worker row reports
    /// only the shared blocks beyond that rank's device-local prefix.
    pub async fn get_overlap_scores(
        &self,
        tokens: &[u32],
        router_config_override: Option<&RouterConfigOverride>,
        block_mm_infos: Option<&[Option<BlockExtraInfo>]>,
        lora_name: Option<&str>,
        cache_namespace: Option<&str>,
        include_shared: bool,
    ) -> Result<OverlapScoresResponse, KvRouterError> {
        let hash_options = BlockHashOptions {
            block_mm_infos,
            lora_name,
            cache_namespace,
            is_eagle: Some(self.is_eagle),
        };
        let block_hashes = compute_block_hash_for_seq(tokens, self.block_size, hash_options);
        let num_blocks = block_hashes.len();

        let tiered_matches = self.indexer.find_matches_by_tier(block_hashes).await?;

        let (shared_hits, shared_error) = if include_shared {
            if let Some(shared_cache) = self.shared_cache.as_ref() {
                match shared_cache
                    .check_blocks(tokens, self.block_size, cache_namespace)
                    .await
                {
                    Ok(hits) => (Some(hits), None),
                    Err(err) => {
                        tracing::warn!(error = %err, "Shared cache overlap query failed");
                        (None, Some(err.to_string()))
                    }
                }
            } else {
                (None, None)
            }
        } else {
            (None, None)
        };

        let shared_enabled = include_shared && self.shared_cache.is_some();
        let expected_workers = {
            let configs = self.workers_with_configs.borrow();
            configs
                .iter()
                .flat_map(|(&worker_id, config)| {
                    let start = config.data_parallel_start_rank();
                    let end = start.saturating_add(config.data_parallel_size());
                    (start..end).map(move |dp_rank| WorkerWithDpRank::new(worker_id, dp_rank))
                })
                .collect::<Vec<_>>()
        };
        Ok(
            OverlapAnalysis::new(&self.kv_router_config, self.block_size, &tiered_matches)
                .scores_response(
                    router_config_override,
                    num_blocks,
                    expected_workers,
                    shared_enabled,
                    shared_hits.as_ref(),
                    shared_error,
                ),
        )
    }

    /// Dump all events from the indexer
    pub async fn dump_events(&self) -> Result<Vec<RouterEvent>, KvRouterError> {
        self.indexer.dump_events().await
    }
}

// NOTE: KVRouter works like a PushRouter,
// but without the reverse proxy functionality, but based on the RouterRequest contract
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
        let policy_class = ctx.metadata().get("policy-class").cloned();
        // Handle different request types
        let response = match request {
            RouterRequest::New {
                tokens,
                block_mm_infos,
                routing_constraints,
                priority_jump,
                strict_priority,
                lora_name,
                cache_namespace,
            } => {
                let request_context = ctx.context();
                let mut schedule = Box::pin(self.find_best_match_details_with_policy_class(
                    Some(&context_id),
                    &tokens,
                    block_mm_infos.as_deref(),
                    None,
                    true,
                    false,
                    lora_name,
                    cache_namespace,
                    priority_jump,
                    strict_priority,
                    policy_class,
                    None,
                    None,
                    None,
                    None,
                    routing_constraints,
                ));
                let outcome = tokio::select! {
                    biased;

                    _ = request_context.stopped() => None,
                    outcome = &mut schedule => Some(outcome),
                };
                drop(schedule);

                let Some(outcome) = outcome else {
                    if let Err(error) = self.free(&context_id).await {
                        tracing::warn!(
                            request_id = %context_id,
                            %error,
                            "Failed to free scheduler state after RouterRequest::New cancellation"
                        );
                    }
                    return Err(cancelled_error(&context_id));
                };
                match outcome {
                    Ok(FindBestMatchOutcome::Routed {
                        worker,
                        overlap_blocks,
                        ..
                    }) => RouterResponse::New {
                        worker_id: worker.worker_id,
                        dp_rank: worker.dp_rank,
                        overlap_blocks,
                    },
                    Ok(FindBestMatchOutcome::QueueRejected { rejection }) => {
                        RouterResponse::QueueRejected { rejection }
                    }
                    Err(error) => return Err(error),
                }
            }
            RouterRequest::PotentialLoads {
                tokens,
                block_mm_infos,
                lora_name,
                cache_namespace,
            } => RouterResponse::PotentialLoads {
                loads: self
                    .get_potential_loads(
                        &tokens,
                        None,
                        block_mm_infos.as_deref(),
                        lora_name.as_deref(),
                        cache_namespace.as_deref(),
                    )
                    .await?,
                pending_count: self.pending_count(),
                pending_isl_tokens: self.pending_isl_tokens(),
            },
            RouterRequest::MarkPrefill { request_id } => {
                let request_id = match request_id.as_deref() {
                    Some(request_id) if !request_id.trim().is_empty() => request_id,
                    _ => &context_id,
                };
                RouterResponse::PrefillMarked {
                    success: self.mark_prefill_completed(request_id).await.is_ok(),
                }
            }
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
    use std::collections::HashMap;

    use async_trait::async_trait;
    use dynamo_kv_router::{
        indexer::{LowerTierMatchDetails, MatchDetails},
        protocols::{OverlapScores, StorageTier, compute_seq_hash_for_block},
    };
    use dynamo_runtime::{DistributedRuntime, Runtime, distributed::DistributedConfig};
    use tokio::sync::watch;

    use crate::kv_router::scheduler::KvSchedulerError;
    use crate::local_model::runtime_config::ModelRuntimeConfig;

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
            _cache_namespace: Option<&str>,
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
            _eligibility: dynamo_kv_router::scheduling::RoutingEligibility<'_>,
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

    struct OverloadedSelector;

    impl dynamo_kv_router::selector::WorkerSelector<ModelRuntimeConfig> for OverloadedSelector {
        fn select_worker(
            &self,
            _workers: &HashMap<WorkerId, ModelRuntimeConfig>,
            _request: &dynamo_kv_router::scheduling::SchedulingRequest,
            _eligibility: dynamo_kv_router::scheduling::RoutingEligibility<'_>,
            _block_size: u32,
        ) -> Result<dynamo_kv_router::protocols::WorkerSelectionResult, KvSchedulerError> {
            Err(KvSchedulerError::AllEligibleWorkersOverloaded)
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
            overlap_score_credit: 0.0,
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
            None,
            2,
            selector,
            Some(config),
            None,
            "decode",
            None,
            false,
            shared_cache,
            None,
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
                None,
                0.0,
                0,
                None,
                None,
                RoutingConstraints::default(),
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
                None,
                0.0,
                0,
                None,
                None,
                RoutingConstraints::default(),
            )
            .await
            .unwrap();

        assert_eq!(worker, WorkerWithDpRank::from_worker_id(0));
        assert_eq!(overlap, 0);
    }

    #[tokio::test]
    async fn test_find_best_match_maps_overload_to_resource_exhausted() {
        let router = make_test_router(OverloadedSelector, None).await;

        let err = router
            .find_best_match(
                None,
                &[11, 12],
                None,
                None,
                false,
                None,
                None,
                0.0,
                0,
                None,
                None,
                RoutingConstraints::default(),
            )
            .await
            .unwrap_err();

        assert!(dynamo_runtime::error::match_error_chain(
            err.as_ref(),
            &[dynamo_runtime::error::ErrorType::ResourceExhausted],
            &[]
        ));
        assert!(
            err.to_string()
                .contains("all eligible workers are overloaded")
        );
    }

    #[tokio::test]
    async fn test_find_best_match_details_returns_routing_hashes_when_requested() {
        let router = make_test_router(
            InspectingSelector {
                expected_hits: None,
                selected_worker: WorkerWithDpRank::from_worker_id(0),
            },
            None,
        )
        .await;
        let tokens = [11, 12, 21, 22];

        let outcome = router
            .find_best_match_details(
                None,
                &tokens,
                None,
                None,
                false,
                true,
                None,
                None,
                0.0,
                0,
                None,
                None,
                None,
                RoutingConstraints::default(),
            )
            .await
            .unwrap();

        let FindBestMatchOutcome::Routed {
            routing_hashes: Some(hashes),
            ..
        } = outcome
        else {
            panic!("expected routed outcome with routing hashes");
        };
        let expected_local = compute_block_hash_for_seq(
            &tokens,
            2,
            BlockHashOptions {
                block_mm_infos: None,
                lora_name: None,
                cache_namespace: None,
                is_eagle: Some(false),
            },
        );
        let expected_sequence = compute_seq_hash_for_block(&expected_local);

        assert_eq!(hashes.local_hashes, expected_local);
        assert_eq!(hashes.sequence_hashes, expected_sequence);
    }

    #[tokio::test]
    async fn test_find_best_match_details_omits_routing_hashes_when_not_requested() {
        let router = make_test_router(
            InspectingSelector {
                expected_hits: None,
                selected_worker: WorkerWithDpRank::from_worker_id(0),
            },
            None,
        )
        .await;

        let outcome = router
            .find_best_match_details(
                None,
                &[11, 12, 21, 22],
                None,
                None,
                false,
                false,
                None,
                None,
                0.0,
                0,
                None,
                None,
                None,
                RoutingConstraints::default(),
            )
            .await
            .unwrap();

        let FindBestMatchOutcome::Routed { routing_hashes, .. } = outcome else {
            panic!("expected routed outcome");
        };
        assert!(routing_hashes.is_none());
    }

    #[tokio::test]
    async fn test_get_overlap_scores_returns_tiered_rows_and_shared_hits() {
        let router = make_test_router(
            InspectingSelector {
                expected_hits: None,
                selected_worker: WorkerWithDpRank::from_worker_id(0),
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

        let scores = router
            .get_overlap_scores(&[11, 12, 21, 22], None, None, None, None, true)
            .await
            .unwrap();

        assert_eq!(scores.block_size, 2);
        assert_eq!(scores.num_blocks, 2);
        assert!(scores.shared_cache.enabled);
        assert_eq!(scores.shared_cache.total_hit_blocks, 2);
        assert_eq!(scores.shared_cache.ranges, vec![(0, 2)]);
        assert_eq!(scores.shared_cache.error, None);
        assert_eq!(scores.workers.len(), 2);

        for worker in scores.workers {
            assert_eq!(worker.device_blocks, 0);
            assert_eq!(worker.host_pinned_blocks, 0);
            assert_eq!(worker.disk_blocks, 0);
            assert_eq!(worker.host_pinned_extension_blocks, 0);
            assert_eq!(worker.disk_extension_blocks, 0);
            assert_eq!(worker.shared_beyond_device_blocks, Some(2));
            assert!((worker.router_credit_blocks - 1.0).abs() < f64::EPSILON);
        }
    }
}
