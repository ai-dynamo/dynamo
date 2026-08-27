// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::{HashMap, HashSet},
    fmt,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Instant,
};

use anyhow::Result;
use dynamo_kv_router::{
    DEFAULT_ROUTING_GROUP, KvSchedulerError, PrefillLoadEstimator, RoutingPartitionRef,
    SharedKvCache, TrackingHashAlgorithm, TrackingHashContext, TrackingHashScope,
    config::{KvRouterConfig, RouterConfigOverride, min_initial_workers_from_env},
    indexer::{
        ApproximateLruIncarnation, ApproximateLruRequestId, ApproximateLruStats, KvRouterError,
        RoutingDecisionHashes,
    },
    protocols::KV_EVENT_SUBJECT,
    protocols::{
        BlockExtraInfo, BlockHashOptions, LocalBlockHash, PrefillLoadHint, RouterEvent,
        RouterRequest, RouterResponse, RoutingConstraints, TokensWithHashes, WorkerConfigLike,
        WorkerId, WorkerWithDpRank, compute_block_hash_for_seq,
    },
    router_hint::{RouterHint, RouterHintCandidateSource, RouterHintRootCandidates},
    scheduling::{
        CacheHitEstimates, ClassifyEvent, ClassifyFuture, ClassifyInputs, ClassifyRequest,
        OverlapAnalysis, OverloadedWorkerProvider, RequestClassifier, ScheduleMode,
        ScheduleRequest, TieredOverlapRefresher, WorkerAvailabilityProvider,
        effective_prefill_tokens, overlap::cache_hit_estimates_from_tiered_matches,
    },
    selector::WorkerInputs,
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
mod route_lookup;
mod routing_host;
pub mod scheduler;
pub mod sequence;
pub mod shared_cache;

pub use dynamo_kv_router::scheduling::{
    OverlapScoresResponse, SharedCacheOverlapScore, WorkerOverlapScore,
};
pub use encoder_router::EncoderRouter;
pub use indexer::{Indexer, ServedIndexerHandle, ServedIndexerMode, ensure_served_indexer_service};
pub use prefill_router::PrefillRouter;
pub use routing_host::{KvPushRouter, RoutingHost};

use crate::{
    discovery::{KvSourceMembershipWatch, RuntimeConfigWatch},
    kv_router::{
        scheduler::{DefaultWorkerSelector, KvScheduler, PotentialLoad},
        sequence::{SequenceError, SequenceRequest},
    },
    local_model::runtime_config::ModelRuntimeConfig,
    worker_type::WorkerType,
};
use route_lookup::{
    TieredLookupOptions, TieredLookupResult, query_tiered_matches, split_retained_block_hashes,
};

pub(crate) type WorkerSelectorFactory<Sel> =
    Arc<dyn for<'a> Fn(&KvRouterConfig, WorkerType, RoutingPartitionRef<'a>) -> Sel + Send + Sync>;

#[derive(Clone, Copy)]
struct ApproximateLruRankRegistration {
    incarnation: ApproximateLruIncarnation,
    capacity: Option<usize>,
    reconciled: bool,
    retiring: bool,
}

#[derive(Default)]
struct ApproximateLruRankRegistry {
    ranks: HashMap<WorkerWithDpRank, ApproximateLruRankRegistration>,
    next_incarnation: ApproximateLruIncarnation,
}

impl ApproximateLruRankRegistry {
    fn register(
        &mut self,
        worker: WorkerWithDpRank,
        capacity: Option<usize>,
    ) -> ApproximateLruRankRegistration {
        self.next_incarnation = self.next_incarnation.wrapping_add(1).max(1);
        let registration = ApproximateLruRankRegistration {
            incarnation: self.next_incarnation,
            capacity,
            reconciled: false,
            retiring: false,
        };
        self.ranks.insert(worker, registration);
        registration
    }
}

type ApproximateLruRanks = Arc<parking_lot::Mutex<ApproximateLruRankRegistry>>;

async fn reconcile_approximate_lru_snapshot(
    indexer: &Indexer,
    snapshot: &HashMap<WorkerId, ModelRuntimeConfig>,
    registry: &ApproximateLruRanks,
) -> Result<(), KvRouterError> {
    let mut advertised = HashMap::new();
    for (&worker_id, config) in snapshot {
        let capacity = config
            .total_kv_blocks
            .and_then(|blocks| usize::try_from(blocks).ok())
            .filter(|blocks| *blocks > 0);
        let end_rank = config
            .data_parallel_start_rank
            .saturating_add(config.data_parallel_size);
        for dp_rank in config.data_parallel_start_rank..end_rank {
            advertised.insert(WorkerWithDpRank::new(worker_id, dp_rank), capacity);
        }
    }

    let retirements = {
        let mut registry = registry.lock();
        for (worker, registration) in &mut registry.ranks {
            if !advertised.contains_key(worker) {
                registration.retiring = true;
                registration.reconciled = false;
            }
        }
        let retirements = registry
            .ranks
            .iter()
            .filter(|(_, registration)| registration.retiring)
            .map(|(&worker, registration)| (worker, registration.incarnation))
            .collect::<Vec<_>>();

        for (worker, advertised_capacity) in advertised {
            let mut registration = match registry.ranks.get(&worker).copied() {
                Some(registration) if registration.retiring => continue,
                Some(mut registration) => {
                    // Missing capacity pins this worker incarnation to TTL until removal.
                    let effective_capacity = registration.capacity.and(advertised_capacity);
                    if registration.capacity == effective_capacity && registration.reconciled {
                        continue;
                    }
                    registration.capacity = effective_capacity;
                    registration
                }
                None => registry.register(worker, advertised_capacity),
            };
            if registration.capacity.is_none() {
                tracing::warn!(
                    worker_id = worker.worker_id,
                    dp_rank = worker.dp_rank,
                    "Approximate LRU requires a positive per-rank total_kv_blocks; clearing this rank and using TTL until it is removed and re-registered"
                );
            }
            registration.reconciled = indexer
                .set_approximate_lru_capacity_now(
                    worker,
                    registration.incarnation,
                    registration.capacity,
                )
                .is_ok();
            registry.ranks.insert(worker, registration);
        }
        retirements
    };

    for (worker, incarnation) in retirements {
        indexer
            .reset_worker_dp_rank_and_wait(worker.worker_id, worker.dp_rank)
            .await?;
        let mut registry = registry.lock();
        if registry.ranks.get(&worker).is_some_and(|registration| {
            registration.retiring && registration.incarnation == incarnation
        }) {
            registry.ranks.remove(&worker);
        }
    }

    Ok(())
}

fn start_approximate_lru_reconciler(
    indexer: Indexer,
    mut workers: RuntimeConfigWatch,
    registry: ApproximateLruRanks,
    cancellation: CancellationToken,
) {
    tokio::spawn(async move {
        loop {
            let changed = tokio::select! {
                _ = cancellation.cancelled() => break,
                changed = workers.changed() => changed,
            };
            if changed.is_err() {
                break;
            }
            let snapshot = workers.borrow_and_update().clone();
            if let Err(error) =
                reconcile_approximate_lru_snapshot(&indexer, &snapshot, &registry).await
            {
                tracing::error!(%error, "Failed to reconcile approximate LRU capacities");
            }
        }
    });
}

fn start_approximate_lru_metrics(
    indexer: Indexer,
    metrics: Arc<metrics::ApproximateLruMetrics>,
    cancellation: CancellationToken,
) {
    tokio::spawn(async move {
        let mut previous = ApproximateLruStats::default();
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(1));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            tokio::select! {
                _ = cancellation.cancelled() => break,
                _ = interval.tick() => {
                    match indexer.approximate_lru_stats().await {
                        Ok(stats) => metrics.observe(stats, &mut previous),
                        Err(error) => tracing::warn!(%error, "Failed to collect approximate LRU metrics"),
                    }
                }
            }
        }
    });
}

pub(crate) fn to_worker_selection_session_context(
    context: &crate::protocols::common::extensions::AgentContext,
) -> dynamo_kv_router::SessionContext {
    use crate::protocols::common::extensions::{AgentContext, InputTrigger, KvHints};
    use dynamo_kv_router::{SessionContext, WorkerSelectionInputTrigger, WorkerSelectionKvHints};

    // Keep this exhaustive so a new wire-level field must be handled here.
    let AgentContext {
        session_id,
        parent_session_id,
        session_final,
        compaction: _,
        kv_hints,
        input_trigger,
    } = context;
    let input_trigger = input_trigger.map(|trigger| match trigger {
        InputTrigger::UserMessage => WorkerSelectionInputTrigger::UserMessage,
        InputTrigger::ToolResult => WorkerSelectionInputTrigger::ToolResult,
        InputTrigger::Other => WorkerSelectionInputTrigger::Other,
    });
    SessionContext::new(
        session_id.clone(),
        parent_session_id.clone(),
        *session_final,
        kv_hints.as_ref().map(|hints| {
            let KvHints { evict_session } = hints;
            WorkerSelectionKvHints::new(*evict_session)
        }),
        input_trigger,
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum KvEventSourceRequirement {
    NotRequired,
    CacheAwareRouting,
    ConditionalDisaggDecodeCache,
    Unknown,
}

impl KvEventSourceRequirement {
    pub(crate) fn derive(worker_role: Option<WorkerType>, config: &KvRouterConfig) -> Self {
        let Some(worker_role) = worker_role else {
            return Self::Unknown;
        };
        if config.use_remote_indexer || !config.should_subscribe_to_kv_events() {
            return Self::NotRequired;
        }

        match worker_role {
            WorkerType::Prefill | WorkerType::Aggregated => Self::CacheAwareRouting,
            WorkerType::Decode if config.conditional_disagg_enabled => {
                Self::ConditionalDisaggDecodeCache
            }
            WorkerType::Decode | WorkerType::Encode => Self::NotRequired,
        }
    }

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::NotRequired => "not_required",
            Self::CacheAwareRouting => "cache_aware_routing",
            Self::ConditionalDisaggDecodeCache => "conditional_disagg_decode_cache",
            Self::Unknown => "unknown",
        }
    }

    pub(crate) fn requires_source(self) -> bool {
        matches!(
            self,
            Self::CacheAwareRouting | Self::ConditionalDisaggDecodeCache
        )
    }

    pub(crate) fn should_subscribe(self, config: &KvRouterConfig) -> bool {
        match self {
            Self::Unknown => !config.use_remote_indexer && config.should_subscribe_to_kv_events(),
            requirement => requirement.requires_source(),
        }
    }
}

impl fmt::Display for KvEventSourceRequirement {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

pub enum FindBestMatchOutcome {
    Routed {
        worker: WorkerWithDpRank,
        overlap_blocks: u32,
        effective_overlap_blocks: f64,
        cached_tokens: usize,
        potential_decode_blocks: u64,
        routing_hashes: Option<RoutingDecisionHashes>,
        router_hint: Option<RouterHint>,
    },
    QueueRejected {
        rejection: scheduling::QueueRejection,
    },
}

/// For probes that return best-match routing decisions plus selected-worker
/// scheduler-load snapshots, without admitting the request into scheduler state.
/// `FindBestMatchInnerOutcome` keeps this advisory shape internal so admitted
/// routing can keep using `FindBestMatchOutcome` unchanged.
pub enum FindBestMatchAdvisoryOutcome {
    Routed {
        worker: WorkerWithDpRank,
        overlap_blocks: u32,
        effective_overlap_blocks: f64,
        cached_tokens: usize,
        potential_decode_blocks: u64,
        selected_worker_load: scheduling::AdvisoryWorkerLoad,
        routing_hashes: Option<RoutingDecisionHashes>,
    },
    QueueRejected {
        rejection: scheduling::QueueRejection,
    },
}

#[derive(Debug, Clone, Copy)]
pub(super) enum FindBestMatchAdmission {
    WithAdmission { track_lifecycle: bool },
    WithoutAdmission,
}

pub(super) enum FindBestMatchInnerOutcome {
    WithAdmission(FindBestMatchOutcome),
    WithoutAdmission(FindBestMatchAdvisoryOutcome),
}

struct DeclaredInputsClassifier {
    classifier: Box<dyn RequestClassifier>,
    required_inputs: ClassifyInputs,
}

impl RequestClassifier for DeclaredInputsClassifier {
    fn required_inputs(&self) -> ClassifyInputs {
        self.required_inputs
    }

    fn classify(&self, request: ClassifyRequest) -> ClassifyFuture {
        self.classifier.classify(request)
    }

    fn on_event(&mut self, event: ClassifyEvent<'_>) {
        self.classifier.on_event(event);
    }
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
    // A single overloaded worker can be retried elsewhere; pool-wide and
    // admission-control overloads cannot. A filter rejection is unavailable, not
    // overload, and becomes HTTP 503.
    let (error_type, overloaded) = match error {
        scheduling::KvSchedulerError::PinnedWorkerOverloaded { .. } => {
            (ErrorType::WorkerOverloaded, true)
        }
        scheduling::KvSchedulerError::AllEligibleWorkersOverloaded => {
            (ErrorType::ResourceExhausted, true)
        }
        scheduling::KvSchedulerError::ClassifyPendingLimit { .. }
        | scheduling::KvSchedulerError::DueTimeExpired => (ErrorType::ResourceExhausted, true),
        scheduling::KvSchedulerError::AllEligibleWorkersFiltered => (ErrorType::Unavailable, false),
        _ => return error.into(),
    };

    let message = error.to_string();
    let error = DynamoError::builder()
        .error_type(error_type)
        .message(message.clone());
    if overloaded {
        error
            .cause(PipelineError::ServiceOverloaded(message))
            .build()
            .into()
    } else {
        error.build().into()
    }
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
    required_worker_inputs: dynamo_kv_router::selector::WorkerInputs,
    workers_with_configs: RuntimeConfigWatch,
    block_size: u32,
    kv_router_config: KvRouterConfig,
    prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
    cancellation_token: CancellationToken,
    client: Client,
    is_eagle: bool,
    kv_event_subscription: Option<indexer::KvEventSubscriptionHandle>,
    tracking_hash: TrackingHashContext,
    tracking_model_name: String,
    approximate_lru_ranks: ApproximateLruRanks,
    next_approximate_lru_request_id: AtomicU64,
    _served_indexer_handle: Option<ServedIndexerHandle>,
    /// Optional external shared KV cache pool. When present, `find_best_match`
    /// queries it in parallel with the indexer and factors shared hits into scoring.
    shared_cache: Option<Box<dyn SharedKvCache>>,
    /// Optional LoRA filter. When present (LoRA serving enabled), candidate workers are
    /// narrowed to the LoRA's allocated/loaded replicas inside `find_best_match_details`,
    /// covering both the decode and prefill routers (both built via `kv_chooser_for`).
    lora_filter: Option<Arc<crate::lora::LoraFilter>>,
    endpoint_registration: Option<dynamo_runtime::discovery::EndpointRegistrationLease>,
    teardown_task_guard: Option<dynamo_runtime::engine::EngineContextGuard>,
}

fn resolve_tracking_model_name(
    algorithm: TrackingHashAlgorithm,
    model_name: Option<&str>,
) -> Result<String> {
    if algorithm == TrackingHashAlgorithm::KeyedXxh3V1 {
        return model_name
            .filter(|value| !value.is_empty())
            .map(str::to_owned)
            .ok_or_else(|| {
                anyhow::anyhow!("model_name is required for keyed router tracking hashes")
            });
    }
    Ok(model_name.unwrap_or_default().to_owned())
}

impl<Sel> KvRouter<Sel>
where
    Sel: dynamo_kv_router::selector::WorkerSelector<ModelRuntimeConfig> + Send + 'static,
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
        metric_worker_type: &'static str,
        model_name: Option<String>,
        is_eagle: bool,
        shared_cache: Option<Box<dyn SharedKvCache>>,
        lora_filter: Option<Arc<crate::lora::LoraFilter>>,
    ) -> Result<Self> {
        Self::new_with_worker_role(
            endpoint,
            client,
            workers_with_configs,
            kv_source_membership,
            block_size,
            selector,
            kv_router_config,
            prefill_load_estimator,
            None,
            metric_worker_type,
            model_name,
            is_eagle,
            shared_cache,
            lora_filter,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn new_with_worker_role(
        endpoint: Endpoint,
        client: Client,
        workers_with_configs: RuntimeConfigWatch,
        kv_source_membership: Option<KvSourceMembershipWatch>,
        block_size: u32,
        selector: Sel,
        kv_router_config: Option<KvRouterConfig>,
        prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
        worker_role: Option<WorkerType>,
        metric_worker_type: &'static str,
        model_name: Option<String>,
        is_eagle: bool,
        shared_cache: Option<Box<dyn SharedKvCache>>,
        lora_filter: Option<Arc<crate::lora::LoraFilter>>,
    ) -> Result<Self> {
        Self::new_with_worker_role_inner(
            endpoint,
            client,
            workers_with_configs,
            kv_source_membership,
            block_size,
            selector,
            kv_router_config,
            prefill_load_estimator,
            worker_role,
            metric_worker_type,
            model_name,
            is_eagle,
            shared_cache,
            lora_filter,
            None,
        )
        .await
    }

    /// Construct a KV router with an optional classifier before policy queueing.
    ///
    /// Drive the returned router through [`RoutingHost`] so response lifecycle
    /// events reach the classifier.
    #[allow(clippy::too_many_arguments)]
    pub async fn new_with_request_classifier(
        endpoint: Endpoint,
        client: Client,
        workers_with_configs: RuntimeConfigWatch,
        kv_source_membership: Option<KvSourceMembershipWatch>,
        block_size: u32,
        selector: Sel,
        kv_router_config: Option<KvRouterConfig>,
        prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
        worker_role: Option<WorkerType>,
        metric_worker_type: &'static str,
        model_name: Option<String>,
        is_eagle: bool,
        shared_cache: Option<Box<dyn SharedKvCache>>,
        lora_filter: Option<Arc<crate::lora::LoraFilter>>,
        request_classifier: impl RequestClassifier,
    ) -> Result<Self> {
        Self::new_with_worker_role_inner(
            endpoint,
            client,
            workers_with_configs,
            kv_source_membership,
            block_size,
            selector,
            kv_router_config,
            prefill_load_estimator,
            worker_role,
            metric_worker_type,
            model_name,
            is_eagle,
            shared_cache,
            lora_filter,
            Some(Box::new(request_classifier)),
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn new_with_worker_role_inner(
        endpoint: Endpoint,
        client: Client,
        workers_with_configs: RuntimeConfigWatch,
        kv_source_membership: Option<KvSourceMembershipWatch>,
        block_size: u32,
        selector: Sel,
        kv_router_config: Option<KvRouterConfig>,
        prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
        worker_role: Option<WorkerType>,
        metric_worker_type: &'static str,
        model_name: Option<String>,
        is_eagle: bool,
        shared_cache: Option<Box<dyn SharedKvCache>>,
        lora_filter: Option<Arc<crate::lora::LoraFilter>>,
        request_classifier: Option<Box<dyn RequestClassifier>>,
    ) -> Result<Self> {
        let required_worker_inputs = selector.required_worker_inputs();
        let required_classify_inputs = request_classifier
            .as_ref()
            .map_or(ClassifyInputs::NONE, |classifier| {
                classifier.required_inputs()
            });
        let request_classifier = request_classifier.map(|classifier| {
            Box::new(DeclaredInputsClassifier {
                classifier,
                required_inputs: required_classify_inputs,
            }) as Box<dyn RequestClassifier>
        });
        let cache_inputs_required = required_worker_inputs.contains(WorkerInputs::CACHE)
            || required_classify_inputs.contains(ClassifyInputs::CACHE);
        // ModelManager gates client construction as well, but preserve the capability boundary for
        // direct KvRouter callers.
        let shared_cache = if cache_inputs_required {
            shared_cache
        } else {
            None
        };
        let kv_router_config = kv_router_config.unwrap_or_default();
        kv_router_config.validate().map_err(anyhow::Error::msg)?;
        let tracking_hash = TrackingHashContext::from_config(&kv_router_config)?;
        let tracking_model_name =
            resolve_tracking_model_name(tracking_hash.algorithm(), model_name.as_deref())?;
        let kv_event_source_requirement =
            KvEventSourceRequirement::derive(worker_role, &kv_router_config);
        let cache_required = cache_inputs_required
            || kv_router_config.serve_indexer
            || matches!(
                kv_event_source_requirement,
                KvEventSourceRequirement::ConditionalDisaggDecodeCache
                    | KvEventSourceRequirement::Unknown
            );
        let component = endpoint.component();
        // Router-owned tasks derive from this token so a rebuild cannot cancel the runtime.
        let cancellation_token = component.drt().child_token();
        let cancellation_guard = cancellation_token.clone().drop_guard();
        let min_initial_workers = min_initial_workers_from_env()?;

        let indexer = if cache_required {
            Indexer::new(
                component,
                &kv_router_config,
                block_size,
                model_name.as_deref(),
                cancellation_token.child_token(),
            )
            .await?
        } else {
            Indexer::None
        };
        let approximate_lru_metrics = metrics::ApproximateLruMetrics::from_component(component);
        let configured_policy = kv_router_config.router_approximate_cache_policy.to_string();
        let effective_policy = if kv_router_config.overlap_score_credit <= 0.0 {
            "disabled"
        } else if indexer.uses_approximate_lru() {
            "lru"
        } else {
            "ttl"
        };
        approximate_lru_metrics.set_policies(&configured_policy, effective_policy);

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

        let approximate_lru_ranks = Arc::new(parking_lot::Mutex::new(
            ApproximateLruRankRegistry::default(),
        ));
        if indexer.uses_approximate_lru() {
            let snapshot = workers_with_configs.borrow().clone();
            reconcile_approximate_lru_snapshot(&indexer, &snapshot, &approximate_lru_ranks).await?;
            start_approximate_lru_reconciler(
                indexer.clone(),
                workers_with_configs.clone(),
                Arc::clone(&approximate_lru_ranks),
                cancellation_token.child_token(),
            );
            start_approximate_lru_metrics(
                indexer.clone(),
                approximate_lru_metrics,
                cancellation_token.child_token(),
            );
        }

        let overlap_scores_refresh = indexer.supports_overlap_refresh().then(|| {
            Arc::new(TieredOverlapRefresher::new(
                indexer.clone(),
                kv_router_config.clone(),
                block_size,
            ))
        });
        let classification_overlap_scores_refresh =
            if request_classifier.is_some() && indexer.supports_classifier_overlap_refresh() {
                overlap_scores_refresh.clone().or_else(|| {
                    Some(Arc::new(TieredOverlapRefresher::new(
                        indexer.clone(),
                        kv_router_config.clone(),
                        block_size,
                    )))
                })
            } else {
                None
            };
        let client_for_overload = client.clone();
        let overloaded_worker_provider: OverloadedWorkerProvider =
            Arc::new(move || client_for_overload.overloaded_instance_ids());

        let client_for_availability = client.clone();
        let available_worker_provider: WorkerAvailabilityProvider =
            Arc::new(move || client_for_availability.available_instance_ids());

        let scheduler = match request_classifier {
            Some(request_classifier) => {
                KvScheduler::start_with_request_classifier(
                    endpoint.clone(),
                    block_size,
                    workers_with_configs.clone(),
                    selector,
                    &kv_router_config,
                    prefill_load_estimator.clone(),
                    overlap_scores_refresh,
                    classification_overlap_scores_refresh,
                    Some(overloaded_worker_provider),
                    Some(available_worker_provider),
                    model_name.as_deref(),
                    metric_worker_type,
                    cancellation_token.child_token(),
                    request_classifier,
                )
                .await?
            }
            None => {
                KvScheduler::start(
                    endpoint.clone(),
                    block_size,
                    workers_with_configs.clone(),
                    selector,
                    &kv_router_config,
                    prefill_load_estimator.clone(),
                    overlap_scores_refresh,
                    Some(overloaded_worker_provider),
                    Some(available_worker_provider),
                    model_name.as_deref(),
                    metric_worker_type,
                    cancellation_token.child_token(),
                )
                .await?
            }
        };
        // Start KV event subscription if needed — skip when using a remote indexer.
        let kv_event_subscription = if cache_required
            && kv_event_source_requirement.should_subscribe(&kv_router_config)
        {
            let membership_watch = kv_source_membership.ok_or_else(|| {
                anyhow::anyhow!(
                    "KV source membership watch is required when local KV event subscription is enabled"
                )
            })?;
            Some(
                indexer::start_subscriber(
                    endpoint.clone(),
                    indexer.clone(),
                    membership_watch,
                    block_size,
                    model_name.clone().unwrap_or_else(|| "unknown".to_string()),
                    worker_role,
                    kv_event_source_requirement,
                    metric_worker_type,
                    cancellation_token.child_token(),
                )
                .await?,
            )
        } else {
            tracing::info!(
                requirement = %kv_event_source_requirement,
                cache_required,
                "Skipping KV event subscription (use_kv_events={}, overlap_score_credit={}, use_remote_indexer={})",
                kv_router_config.use_kv_events,
                kv_router_config.overlap_score_credit,
                kv_router_config.use_remote_indexer,
            );
            None
        };

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
            required_worker_inputs,
            workers_with_configs,
            block_size,
            kv_router_config,
            prefill_load_estimator,
            cancellation_token,
            client,
            is_eagle,
            kv_event_subscription,
            tracking_hash,
            tracking_model_name,
            approximate_lru_ranks,
            next_approximate_lru_request_id: AtomicU64::new(0),
            _served_indexer_handle: served_indexer_handle,
            shared_cache,
            lora_filter,
            endpoint_registration: None,
            teardown_task_guard: None,
        })
    }

    pub(crate) fn set_endpoint_registration(
        &mut self,
        registration: dynamo_runtime::discovery::EndpointRegistrationLease,
    ) {
        self.endpoint_registration = Some(registration);
    }

    pub(crate) fn set_teardown_task_guard(
        &mut self,
        task_guard: dynamo_runtime::engine::EngineContextGuard,
    ) {
        if let Some(subscription) = self.kv_event_subscription.as_mut() {
            subscription.set_task_guard(task_guard.clone());
        }
        self.teardown_task_guard = Some(task_guard);
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

    pub fn required_worker_inputs(&self) -> dynamo_kv_router::selector::WorkerInputs {
        self.required_worker_inputs
    }

    pub(crate) fn begin_request_lifecycle(
        &self,
        request_id: &str,
    ) -> Result<Option<scheduling::RequestLifecycle>, KvSchedulerError> {
        self.scheduler.begin_request_lifecycle(request_id)
    }

    /// Cancel background work and wait for KV event ingestion to stop.
    pub async fn shutdown(mut self) {
        self.cancellation_token.cancel();
        if let Some(subscription) = self.kv_event_subscription.take() {
            subscription.shutdown().await;
        }
    }

    pub fn is_eagle(&self) -> bool {
        self.is_eagle
    }

    fn approximate_lru_rank_registration(
        &self,
        worker: WorkerWithDpRank,
    ) -> Option<ApproximateLruRankRegistration> {
        if !self.indexer.uses_approximate_lru() {
            return None;
        }
        // Serialize the authoritative MRC recheck with rank retirement. A request
        // that observed the prior snapshot cannot re-register a rank after its
        // reset has begun.
        let mut registry = self.approximate_lru_ranks.lock();
        if registry
            .ranks
            .get(&worker)
            .is_some_and(|registration| registration.retiring)
        {
            return None;
        }
        let configs = self.workers_with_configs.borrow();
        let config = configs.get(&worker.worker_id)?;
        let end_rank = config
            .data_parallel_start_rank
            .saturating_add(config.data_parallel_size);
        if !(config.data_parallel_start_rank..end_rank).contains(&worker.dp_rank) {
            return None;
        }
        let capacity = config
            .total_kv_blocks
            .and_then(|blocks| usize::try_from(blocks).ok())
            .filter(|blocks| *blocks > 0);
        drop(configs);

        let mut registration = match registry.ranks.get(&worker).copied() {
            Some(registration) => registration,
            None => registry.register(worker, capacity),
        };
        if registration.reconciled {
            return Some(registration);
        }
        if let Err(error) = self.indexer.set_approximate_lru_capacity_now(
            worker,
            registration.incarnation,
            registration.capacity,
        ) {
            tracing::warn!(
                worker_id = worker.worker_id,
                dp_rank = worker.dp_rank,
                %error,
                "Failed to register approximate LRU rank"
            );
            return None;
        }
        registration.reconciled = true;
        registry.ranks.insert(worker, registration);
        Some(registration)
    }

    fn next_approximate_lru_request_id(&self) -> ApproximateLruRequestId {
        ApproximateLruRequestId::new(
            self.next_approximate_lru_request_id
                .fetch_add(1, Ordering::Relaxed)
                .wrapping_add(1),
        )
    }

    fn tracking_hash_scope(&self) -> TrackingHashScope<'_> {
        TrackingHashScope {
            partition: RoutingPartitionRef::new(&self.tracking_model_name, DEFAULT_ROUTING_GROUP),
            block_size: self.block_size,
        }
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

    fn has_router_hint_capable_workers(&self) -> bool {
        // Router-hint capability is worker-level metadata. Check one
        // representative DP rank here so the coarse request-path gate does not
        // scale with data_parallel_size. Follow-up: cache this from the runtime
        // config watch if the per-worker scan shows up in large-fleet routing
        // benchmarks.
        self.workers_with_configs.borrow().values().any(|config| {
            config
                .router_hint_metadata_for_dp_rank(config.data_parallel_start_rank())
                .is_some()
        })
    }

    fn router_hint_for_selection(
        &self,
        target: WorkerWithDpRank,
        target_cached_prefix_blocks: u32,
        candidates: Option<&RouterHintRootCandidates>,
    ) -> Option<RouterHint> {
        let candidates = candidates?;

        let (block_hashes, source_control_endpoint) = {
            let configs = self.workers_with_configs.borrow();
            let target_config = configs.get(&target.worker_id)?;
            let target_metadata = target_config.router_hint_metadata_for_dp_rank(target.dp_rank)?;

            let prefix_blocks_to_beat =
                usize::try_from(target_cached_prefix_blocks).unwrap_or(usize::MAX);
            let (source, block_hashes) =
                candidates.best_source(prefix_blocks_to_beat, |source| match source {
                    RouterHintCandidateSource::Worker(worker) => {
                        worker != target
                            && configs.get(&worker.worker_id).is_some_and(|config| {
                                config.kv_event_source_mode.as_deref() != Some("state_agent_v2")
                                    && config
                                        .router_hint_metadata_for_dp_rank(worker.dp_rank)
                                        .is_some_and(|source_metadata| {
                                            source_metadata.worker_type
                                                == target_metadata.worker_type
                                                && source_metadata
                                                    .source_control_endpoint
                                                    .is_some_and(|endpoint| !endpoint.is_empty())
                                        })
                            })
                    }
                    RouterHintCandidateSource::CacheOwner(owner) => candidates
                        .routing_snapshot
                        .as_ref()
                        .and_then(|snapshot| snapshot.router_hint_source(owner))
                        .is_some_and(|source| {
                            source.attached_worker != Some(target)
                                && source.metadata.worker_type == target_metadata.worker_type
                                && !source.metadata.source_control_endpoint.is_empty()
                        }),
                })?;
            let source_control_endpoint = match source {
                RouterHintCandidateSource::Worker(worker) => configs
                    .get(&worker.worker_id)?
                    .router_hint_metadata_for_dp_rank(worker.dp_rank)?
                    .source_control_endpoint?
                    .to_string(),
                RouterHintCandidateSource::CacheOwner(owner) => candidates
                    .routing_snapshot
                    .as_ref()?
                    .router_hint_source(owner)?
                    .metadata
                    .source_control_endpoint
                    .clone(),
            };
            (block_hashes, source_control_endpoint)
        };

        if block_hashes.is_empty() {
            return None;
        }

        Some(RouterHint {
            source_control_endpoint,
            block_hashes,
        })
    }

    pub async fn record_routing_decision(
        &self,
        mut tokens_with_hashes: TokensWithHashes,
        worker: WorkerWithDpRank,
    ) -> Result<(), KvRouterError> {
        // Query-only, detached, and public admission paths do not own an LRU
        // request lifecycle. Keep them on the legacy TTL/no-op path.
        if self.indexer.uses_approximate_lru() {
            return Ok(());
        }
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
        session_context: Option<dynamo_kv_router::SessionContext>,
        expected_output_tokens: Option<u32>,
        pinned_worker: Option<WorkerWithDpRank>,
        allowed_worker_ids: Option<HashSet<WorkerId>>,
        routing_constraints: RoutingConstraints,
    ) -> anyhow::Result<FindBestMatchOutcome> {
        if self.scheduler.has_request_classifier() {
            anyhow::bail!("request-classifier routers must be driven through RoutingHost");
        }
        match self
            .find_best_match_details_with_policy_class_inner(
                context_id,
                Instant::now(),
                tokens,
                tokens.len(),
                block_mm_infos,
                router_config_override,
                update_states,
                return_routing_hashes,
                lora_name,
                cache_namespace,
                priority_jump,
                strict_priority,
                policy_class,
                session_context,
                expected_output_tokens,
                pinned_worker,
                allowed_worker_ids,
                routing_constraints,
                FindBestMatchAdmission::WithAdmission {
                    track_lifecycle: false,
                },
            )
            .await?
        {
            FindBestMatchInnerOutcome::WithAdmission(outcome) => Ok(outcome),
            FindBestMatchInnerOutcome::WithoutAdmission(_) => {
                unreachable!("with-admission routing returned advisory outcome")
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn find_best_match_details_without_admission(
        &self,
        context_id: Option<&str>,
        tokens: &[u32],
        block_mm_infos: Option<&[Option<BlockExtraInfo>]>,
        router_config_override: Option<&RouterConfigOverride>,
        return_routing_hashes: bool,
        lora_name: Option<String>,
        cache_namespace: Option<String>,
        priority_jump: f64,
        strict_priority: u32,
        policy_class: Option<String>,
        session_context: Option<dynamo_kv_router::SessionContext>,
        expected_output_tokens: Option<u32>,
        pinned_worker: Option<WorkerWithDpRank>,
        allowed_worker_ids: Option<HashSet<WorkerId>>,
        routing_constraints: RoutingConstraints,
    ) -> anyhow::Result<FindBestMatchAdvisoryOutcome> {
        match self
            .find_best_match_details_with_policy_class_inner(
                context_id,
                Instant::now(),
                tokens,
                tokens.len(),
                block_mm_infos,
                router_config_override,
                false,
                return_routing_hashes,
                lora_name,
                cache_namespace,
                priority_jump,
                strict_priority,
                policy_class,
                session_context,
                expected_output_tokens,
                pinned_worker,
                allowed_worker_ids,
                routing_constraints,
                FindBestMatchAdmission::WithoutAdmission,
            )
            .await?
        {
            FindBestMatchInnerOutcome::WithoutAdmission(outcome) => Ok(outcome),
            FindBestMatchInnerOutcome::WithAdmission(_) => {
                unreachable!("without-admission routing returned admitted outcome")
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn find_best_match_details_with_policy_class_inner(
        &self,
        context_id: Option<&str>,
        ingress_at: Instant,
        tokens: &[u32],
        input_tokens: usize,
        block_mm_infos: Option<&[Option<BlockExtraInfo>]>,
        router_config_override: Option<&RouterConfigOverride>,
        update_states: bool,
        return_routing_hashes: bool,
        lora_name: Option<String>,
        cache_namespace: Option<String>,
        priority_jump: f64,
        strict_priority: u32,
        policy_class: Option<String>,
        session_context: Option<dynamo_kv_router::SessionContext>,
        expected_output_tokens: Option<u32>,
        pinned_worker: Option<WorkerWithDpRank>,
        allowed_worker_ids: Option<HashSet<WorkerId>>,
        routing_constraints: RoutingConstraints,
        admission: FindBestMatchAdmission,
    ) -> anyhow::Result<FindBestMatchInnerOutcome> {
        let start = Instant::now();

        if update_states && context_id.is_none() {
            anyhow::bail!("context_id must be provided if update_states is true");
        }
        let mode = match admission {
            FindBestMatchAdmission::WithAdmission { track_lifecycle }
                if update_states && track_lifecycle =>
            {
                ScheduleMode::TrackedWithLifecycle {
                    request_id: context_id.expect("validated above").to_string(),
                }
            }
            FindBestMatchAdmission::WithAdmission { .. } if update_states => {
                ScheduleMode::Tracked {
                    request_id: context_id.expect("validated above").to_string(),
                }
            }
            FindBestMatchAdmission::WithAdmission { .. }
            | FindBestMatchAdmission::WithoutAdmission => ScheduleMode::QueryOnly {
                request_id: context_id.map(str::to_string),
            },
        };
        let isl_tokens = input_tokens;
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
            self.kv_router_config
                .compute_seq_hashes_for_tracking_with_context(
                    &self.tracking_hash,
                    self.tracking_hash_scope(),
                    tokens,
                    router_config_override,
                    hash_options,
                    Some(&block_hashes),
                )
        });
        let seq_hash_elapsed = start.elapsed();

        let is_admitted_routing = matches!(admission, FindBestMatchAdmission::WithAdmission { .. });
        let supports_overlap_refresh = self.scheduler.supports_overlap_refresh();
        let retain_block_hashes = supports_overlap_refresh || return_routing_hashes;
        let has_router_hint_capable_workers = self.has_router_hint_capable_workers();
        let should_prepare_router_hint = is_admitted_routing && has_router_hint_capable_workers;
        let retain_router_hint_chain =
            should_prepare_router_hint && self.indexer.supports_router_hint_chain_retention();
        if should_prepare_router_hint && !retain_router_hint_chain {
            static WARN_ONCE: std::sync::Once = std::sync::Once::new();
            WARN_ONCE.call_once(|| {
                tracing::warn!(
                    "router_hint chain retention requires a local event-driven indexer with no approximate side indexer and no remote-recorded routing decisions; proceeding without router hints"
                );
            });
        }

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
            TieredLookupOptions {
                cache_namespace: cache_namespace.as_deref(),
                retain_block_hashes,
                retain_router_hint_chain,
            },
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
        let router_hint_candidates = retain_router_hint_chain
            .then(|| tiered_matches.router_hint_root_candidates().cloned())
            .flatten();
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

        let schedule_request = ScheduleRequest {
            mode,
            token_seq: maybe_seq_hashes,
            block_hashes: block_hashes_for_refresh,
            isl_tokens,
            overlap,
            router_hint_candidates,
            retain_router_hint_chain,
            router_config_override: router_config_override.cloned(),
            lora_name,
            priority_jump,
            strict_priority,
            policy_class,
            session_context,
            expected_output_tokens,
            pinned_worker,
            allowed_worker_ids,
            routing_constraints,
            shared_cache_hits,
        };
        let (response, selected_worker_load) = match admission {
            FindBestMatchAdmission::WithAdmission { .. } => match self
                .scheduler
                .schedule_request_with_ingress_at(schedule_request, ingress_at)
                .instrument(tracing::info_span!("kv_router.schedule"))
                .await
            {
                Ok(response) => (response, None),
                Err(KvSchedulerError::QueueRejected(rejection)) => {
                    return Ok(FindBestMatchInnerOutcome::WithAdmission(
                        FindBestMatchOutcome::QueueRejected { rejection },
                    ));
                }
                Err(error) => return Err(map_scheduler_error(error)),
            },
            FindBestMatchAdmission::WithoutAdmission => match self
                .scheduler
                .select_without_admission(schedule_request)
                .instrument(tracing::info_span!("kv_router.select_without_admission"))
                .await
            {
                Ok(advisory) => (advisory.response, Some(advisory.selected_worker_load)),
                Err(KvSchedulerError::QueueRejected(rejection)) => {
                    return Ok(FindBestMatchInnerOutcome::WithoutAdmission(
                        FindBestMatchAdvisoryOutcome::QueueRejected { rejection },
                    ));
                }
                Err(error) => return Err(map_scheduler_error(error)),
            },
        };
        let router_hint = if is_admitted_routing {
            self.router_hint_for_selection(
                response.best_worker,
                response.target_cached_prefix_blocks,
                response.router_hint_candidates.as_ref(),
            )
        } else {
            None
        };

        let total_elapsed = start.elapsed();
        let routing_hashes = routing_block_hashes.map(RoutingDecisionHashes::from_local_hashes);

        // Keep existing routing metrics scoped to requests admitted into the scheduler by this call.
        if is_admitted_routing && let Some(m) = metrics::RoutingOverheadMetrics::get() {
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
        if is_admitted_routing
            && let Some(hits) = sc_hits_for_metrics
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

        match admission {
            FindBestMatchAdmission::WithAdmission { .. } => Ok(
                FindBestMatchInnerOutcome::WithAdmission(FindBestMatchOutcome::Routed {
                    worker: response.best_worker,
                    overlap_blocks: response.effective_overlap_blocks.round() as u32,
                    effective_overlap_blocks: response.effective_overlap_blocks,
                    cached_tokens: response.cached_tokens,
                    potential_decode_blocks: response.potential_decode_blocks as u64,
                    routing_hashes,
                    router_hint,
                }),
            ),
            FindBestMatchAdmission::WithoutAdmission => Ok(
                FindBestMatchInnerOutcome::WithoutAdmission(FindBestMatchAdvisoryOutcome::Routed {
                    worker: response.best_worker,
                    overlap_blocks: response.effective_overlap_blocks.round() as u32,
                    effective_overlap_blocks: response.effective_overlap_blocks,
                    cached_tokens: response.cached_tokens,
                    potential_decode_blocks: response.potential_decode_blocks as u64,
                    selected_worker_load: selected_worker_load
                        .expect("without-admission selection returns advisory load"),
                    routing_hashes,
                }),
            ),
        }
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

        let maybe_seq_hashes = self
            .kv_router_config
            .compute_seq_hashes_for_tracking_with_context(
                &self.tracking_hash,
                self.tracking_hash_scope(),
                tokens,
                router_config_override,
                hash_options,
                None,
            );
        let track_prefill_tokens = self
            .kv_router_config
            .track_prefill_tokens(router_config_override);
        let prefill_load_hint =
            self.prefill_load_hint_for(isl_tokens, cached_tokens, track_prefill_tokens);

        if let Err(error) = self
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
            tracing::warn!("Failed to add request {request_id}: {error}");
        }
    }

    pub async fn mark_prefill_completed(&self, request_id: &str) -> Result<(), SequenceError> {
        self.scheduler.mark_prefill_completed(request_id).await
    }

    pub async fn free(&self, request_id: &str) -> Result<(), SequenceError> {
        self.scheduler.free(request_id).await
    }

    /// Release a booking only if it still belongs to `worker`.
    ///
    /// An ownership mismatch is a harmless no-op, which makes this safe for
    /// delayed cleanup that captured the worker when it acquired the booking.
    pub async fn free_if_worker(
        &self,
        request_id: &str,
        worker: WorkerWithDpRank,
    ) -> Result<(), SequenceError> {
        self.scheduler.free_if_worker(request_id, worker).await
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

        let maybe_seq_hashes = self
            .kv_router_config
            .compute_seq_hashes_for_tracking_with_context(
                &self.tracking_hash,
                self.tracking_hash_scope(),
                tokens,
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
    Sel: dynamo_kv_router::selector::WorkerSelector<ModelRuntimeConfig> + Send + 'static,
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
    use std::collections::{HashMap, HashSet};

    use async_trait::async_trait;
    use dynamo_kv_router::{
        WorkerSelectionInput,
        identity::{
            CacheOwnerId, CacheSemanticsId, DcId, IdentitySource, IndexerDomainId, PoolId,
            RoutingScopeId, StableDpSlotId,
        },
        indexer::{LowerTierMatchDetails, MatchDetails},
        protocols::{
            ExternalSequenceBlockHash, OverlapScores, ResidencyOwner, ResidencyProjection,
            ResidencyRoutingSnapshot, RouterHintSourceMetadata, StorageTier,
            compute_seq_hash_for_block,
        },
    };
    use dynamo_runtime::{DistributedRuntime, Runtime, distributed::DistributedConfig};
    use tokio::sync::watch;

    use crate::kv_router::scheduler::KvSchedulerError;
    use crate::local_model::runtime_config::ModelRuntimeConfig;

    #[test]
    fn all_filtered_workers_map_to_unavailable() {
        let error = map_scheduler_error(KvSchedulerError::AllEligibleWorkersFiltered);
        let dynamo_error = error
            .downcast_ref::<DynamoError>()
            .expect("filtered workers should produce a DynamoError");

        assert_eq!(dynamo_error.error_type(), ErrorType::Unavailable);
    }

    #[test]
    fn classifier_rejections_map_to_resource_exhausted() {
        for error in [
            KvSchedulerError::ClassifyPendingLimit { limit: 8 },
            KvSchedulerError::DueTimeExpired,
        ] {
            let error = map_scheduler_error(error);
            let dynamo_error = error
                .downcast_ref::<DynamoError>()
                .expect("classifier rejection should produce a DynamoError");

            assert_eq!(dynamo_error.error_type(), ErrorType::ResourceExhausted);
        }
    }

    #[test]
    fn worker_selection_receives_complete_session_context() {
        use crate::protocols::common::extensions::{AgentContext, InputTrigger, KvHints};
        use dynamo_kv_router::WorkerSelectionInputTrigger;

        let context = AgentContext {
            session_id: "child-session".into(),
            parent_session_id: Some("root-session".into()),
            session_final: Some(true),
            compaction: None,
            kv_hints: Some(KvHints {
                evict_session: true,
            }),
            input_trigger: Some(InputTrigger::ToolResult),
        };

        let selection_context = to_worker_selection_session_context(&context);

        assert_eq!(selection_context.session_id(), "child-session");
        assert_eq!(selection_context.parent_session_id(), Some("root-session"));
        assert_eq!(selection_context.session_final(), Some(true));
        assert!(
            selection_context
                .kv_hints()
                .expect("KV hints")
                .evict_session()
        );
        assert_eq!(
            selection_context.input_trigger(),
            Some(WorkerSelectionInputTrigger::ToolResult)
        );
    }

    #[test]
    fn keyed_tracking_requires_nonempty_model_name() {
        assert!(resolve_tracking_model_name(TrackingHashAlgorithm::KeyedXxh3V1, None).is_err());
        assert!(resolve_tracking_model_name(TrackingHashAlgorithm::KeyedXxh3V1, Some("")).is_err());
        assert_eq!(
            resolve_tracking_model_name(TrackingHashAlgorithm::KeyedXxh3V1, Some("model-a"))
                .unwrap(),
            "model-a"
        );
    }

    #[test]
    fn public_tracking_preserves_optional_model_name() {
        assert_eq!(
            resolve_tracking_model_name(TrackingHashAlgorithm::PublicXxh3V1, None).unwrap(),
            ""
        );
        assert_eq!(
            resolve_tracking_model_name(TrackingHashAlgorithm::PublicXxh3V1, Some("")).unwrap(),
            ""
        );
    }

    #[test]
    fn kv_event_source_requirement_matrix() {
        let default = KvRouterConfig::default();
        let mut cases = vec![
            (
                Some(WorkerType::Prefill),
                default.clone(),
                KvEventSourceRequirement::CacheAwareRouting,
                true,
            ),
            (
                Some(WorkerType::Aggregated),
                default.clone(),
                KvEventSourceRequirement::CacheAwareRouting,
                true,
            ),
            (
                Some(WorkerType::Decode),
                default.clone(),
                KvEventSourceRequirement::NotRequired,
                false,
            ),
            (
                Some(WorkerType::Encode),
                default.clone(),
                KvEventSourceRequirement::NotRequired,
                false,
            ),
            (
                None,
                default.clone(),
                KvEventSourceRequirement::Unknown,
                true,
            ),
        ];
        for policy in [
            dynamo_kv_router::ConditionalDisaggPolicyKind::IslBounding,
            dynamo_kv_router::ConditionalDisaggPolicyKind::PrefillLoad,
            dynamo_kv_router::ConditionalDisaggPolicyKind::IslOrLoad,
        ] {
            cases.push((
                Some(WorkerType::Decode),
                KvRouterConfig {
                    conditional_disagg_enabled: true,
                    conditional_disagg_policy: policy,
                    ..default.clone()
                },
                KvEventSourceRequirement::ConditionalDisaggDecodeCache,
                true,
            ));
        }
        for config in [
            KvRouterConfig {
                use_remote_indexer: true,
                ..Default::default()
            },
            KvRouterConfig {
                use_kv_events: false,
                ..Default::default()
            },
            KvRouterConfig {
                overlap_score_credit: 0.0,
                ..Default::default()
            },
        ] {
            cases.extend([
                (
                    None,
                    config.clone(),
                    KvEventSourceRequirement::Unknown,
                    false,
                ),
                (
                    Some(WorkerType::Aggregated),
                    config.clone(),
                    KvEventSourceRequirement::NotRequired,
                    false,
                ),
                (
                    Some(WorkerType::Decode),
                    config,
                    KvEventSourceRequirement::NotRequired,
                    false,
                ),
            ]);
        }

        for (role, config, expected, should_subscribe) in cases {
            let requirement = KvEventSourceRequirement::derive(role, &config);
            assert_eq!(requirement, expected);
            assert_eq!(requirement.should_subscribe(&config), should_subscribe);
        }
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
        fn required_worker_inputs(&self) -> WorkerInputs {
            WorkerInputs::CACHE | WorkerInputs::LOAD
        }

        fn select_worker(
            &self,
            input: WorkerSelectionInput<'_, ModelRuntimeConfig>,
        ) -> Result<dynamo_kv_router::protocols::WorkerSelectionResult, KvSchedulerError> {
            let (_workers, request, _eligibility, block_size) = input.into_configured()?;
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
                potential_decode_blocks: request
                    .worker_load_for(self.selected_worker)
                    .potential_decode_blocks()
                    .saturating_add(request.isl_tokens.div_ceil(block_size as usize)),
            })
        }
    }

    struct OverloadedSelector;

    impl dynamo_kv_router::selector::WorkerSelector<ModelRuntimeConfig> for OverloadedSelector {
        fn required_worker_inputs(&self) -> WorkerInputs {
            WorkerInputs::NONE
        }

        fn select_worker(
            &self,
            _input: WorkerSelectionInput<'_, ModelRuntimeConfig>,
        ) -> Result<dynamo_kv_router::protocols::WorkerSelectionResult, KvSchedulerError> {
            Err(KvSchedulerError::AllEligibleWorkersOverloaded)
        }
    }

    struct LoadOnlySelector;

    impl dynamo_kv_router::selector::WorkerSelector<ModelRuntimeConfig> for LoadOnlySelector {
        fn required_worker_inputs(&self) -> WorkerInputs {
            WorkerInputs::LOAD
        }

        fn select_worker(
            &self,
            _input: WorkerSelectionInput<'_, ModelRuntimeConfig>,
        ) -> Result<dynamo_kv_router::protocols::WorkerSelectionResult, KvSchedulerError> {
            unreachable!("capability construction test does not select a worker")
        }
    }

    struct CacheClassifier;

    impl RequestClassifier for CacheClassifier {
        fn required_inputs(&self) -> ClassifyInputs {
            ClassifyInputs::CACHE
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

    async fn make_router_without_membership(worker_role: Option<WorkerType>) -> Result<KvRouter> {
        let component = make_test_component("role-aware-subscription").await;
        let endpoint = component.endpoint("backend");
        let client = endpoint.client().await?;
        let (_tx, workers) = watch::channel(HashMap::from([(7, ModelRuntimeConfig::default())]));
        let config = KvRouterConfig {
            skip_initial_worker_wait: true,
            router_event_threads: 1,
            ..Default::default()
        };

        KvRouter::new_with_worker_role(
            endpoint,
            client,
            workers,
            None,
            16,
            DefaultWorkerSelector::new(Some(config.clone()), "decode"),
            Some(config),
            None,
            worker_role,
            "decode",
            None,
            false,
            None,
            None,
        )
        .await
    }

    #[tokio::test]
    async fn constructor_skips_sources_for_decode_but_preserves_unknown_behavior() {
        let router = make_router_without_membership(Some(WorkerType::Decode))
            .await
            .expect("ordinary decode must not require KV source membership");
        assert!(router.kv_event_subscription.is_none());

        let error = make_router_without_membership(None)
            .await
            .err()
            .expect("unknown role must preserve config-driven subscription");
        assert!(
            error
                .to_string()
                .contains("KV source membership watch is required")
        );
    }

    #[tokio::test]
    async fn load_only_selector_skips_cache_inputs() {
        let component = make_test_component("load-only-capability").await;
        let endpoint = component.endpoint("backend");
        let client = endpoint.client().await.unwrap();
        let (_tx, workers) = watch::channel(HashMap::from([(7, ModelRuntimeConfig::default())]));
        let config = KvRouterConfig {
            skip_initial_worker_wait: true,
            router_event_threads: 1,
            ..Default::default()
        };

        let router = KvRouter::new_with_worker_role(
            endpoint,
            client,
            workers,
            None,
            16,
            LoadOnlySelector,
            Some(config),
            None,
            Some(WorkerType::Prefill),
            "prefill",
            None,
            false,
            Some(Box::new(FakeSharedCache {
                hits: None,
                should_error: false,
            })),
            None,
        )
        .await
        .unwrap();

        assert_eq!(router.required_worker_inputs(), WorkerInputs::LOAD);
        assert!(matches!(router.indexer, Indexer::None));
        assert!(router.kv_event_subscription.is_none());
        assert!(router.shared_cache.is_none());
        assert!(matches!(
            router.dump_events().await,
            Err(KvRouterError::Unsupported(message)) if message == "event dumping requires a KV indexer"
        ));
    }

    #[tokio::test]
    async fn classifier_cache_inputs_retain_cache_providers() {
        let component = make_test_component("classifier-cache-capability").await;
        let endpoint = component.endpoint("backend");
        let client = endpoint.client().await.unwrap();
        let (_tx, workers) = watch::channel(HashMap::from([(7, ModelRuntimeConfig::default())]));
        let config = KvRouterConfig {
            skip_initial_worker_wait: true,
            use_kv_events: false,
            router_track_active_blocks: false,
            router_event_threads: 1,
            ..Default::default()
        };

        let router = KvRouter::new_with_request_classifier(
            endpoint,
            client,
            workers,
            None,
            16,
            LoadOnlySelector,
            Some(config),
            None,
            Some(WorkerType::Prefill),
            "prefill",
            None,
            false,
            Some(Box::new(FakeSharedCache {
                hits: None,
                should_error: false,
            })),
            None,
            CacheClassifier,
        )
        .await
        .unwrap();

        assert_eq!(router.required_worker_inputs(), WorkerInputs::LOAD);
        assert!(!matches!(router.indexer, Indexer::None));
        assert!(router.shared_cache.is_some());
    }

    async fn make_test_router_with_workers(
        selector: impl dynamo_kv_router::selector::WorkerSelector<ModelRuntimeConfig>
        + Send
        + Sync
        + 'static,
        shared_cache: Option<Box<dyn SharedKvCache>>,
        workers: HashMap<WorkerId, ModelRuntimeConfig>,
    ) -> KvRouter<
        impl dynamo_kv_router::selector::WorkerSelector<ModelRuntimeConfig> + Send + Sync + 'static,
    > {
        let component = make_test_component("shared-cache-router").await;
        let endpoint = component.endpoint("backend");
        let client = endpoint.client().await.unwrap();
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

        KvRouter::new_with_worker_role(
            endpoint,
            client,
            rx,
            None,
            2,
            selector,
            Some(config),
            None,
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

    async fn make_test_router(
        selector: impl dynamo_kv_router::selector::WorkerSelector<ModelRuntimeConfig>
        + Send
        + Sync
        + 'static,
        shared_cache: Option<Box<dyn SharedKvCache>>,
    ) -> KvRouter<
        impl dynamo_kv_router::selector::WorkerSelector<ModelRuntimeConfig> + Send + Sync + 'static,
    > {
        let mut workers = HashMap::new();
        workers.insert(0, ModelRuntimeConfig::default());
        workers.insert(1, ModelRuntimeConfig::default());
        make_test_router_with_workers(selector, shared_cache, workers).await
    }

    fn router_hint_runtime_config(endpoint: Option<&str>) -> ModelRuntimeConfig {
        router_hint_runtime_config_with_worker_type(endpoint, "prefill")
    }

    fn router_hint_runtime_config_with_worker_type(
        endpoint: Option<&str>,
        worker_type: &str,
    ) -> ModelRuntimeConfig {
        let mut runtime_config = ModelRuntimeConfig::default();
        runtime_config.runtime_data.insert(
            dynamo_kv_router::router_hint::ROUTER_HINT_RUNTIME_CAPABILITY_KEY.to_string(),
            serde_json::Value::Bool(true),
        );
        runtime_config.runtime_data.insert(
            dynamo_kv_router::router_hint::ROUTER_HINT_WORKER_TYPE_RUNTIME_KEY.to_string(),
            serde_json::Value::String(worker_type.to_string()),
        );
        if let Some(endpoint) = endpoint {
            let mut endpoints = serde_json::Map::new();
            endpoints.insert(
                "0".to_string(),
                serde_json::Value::String(endpoint.to_string()),
            );
            runtime_config.runtime_data.insert(
                dynamo_kv_router::router_hint::ROUTER_HINT_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY
                    .to_string(),
                serde_json::Value::Object(endpoints),
            );
        }
        runtime_config
    }

    fn router_hint_runtime_config_with_dp_endpoints(
        endpoints: &[(u32, &str)],
    ) -> ModelRuntimeConfig {
        let mut runtime_config = router_hint_runtime_config(None);
        runtime_config.runtime_data.insert(
            dynamo_kv_router::router_hint::ROUTER_HINT_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY
                .to_string(),
            serde_json::Value::Object(
                endpoints
                    .iter()
                    .map(|(dp_rank, endpoint)| {
                        (
                            dp_rank.to_string(),
                            serde_json::Value::String(endpoint.to_string()),
                        )
                    })
                    .collect(),
            ),
        );
        runtime_config
    }

    fn router_hint_cache_owner() -> CacheOwnerId {
        CacheOwnerId::new(
            PoolId::new(
                IndexerDomainId::new(
                    CacheSemanticsId::new([1; 16], IdentitySource::Explicit),
                    RoutingScopeId::new([2; 16], IdentitySource::Explicit),
                ),
                DcId::new(3),
            ),
            StableDpSlotId::new([4; 16], IdentitySource::Explicit),
        )
    }

    #[tokio::test]
    async fn router_hint_allows_other_dp_ranks_of_selected_target_worker() {
        let mut workers = HashMap::new();
        workers.insert(
            7,
            router_hint_runtime_config_with_dp_endpoints(&[
                (0, "tcp://127.0.0.1:23280"),
                (1, "tcp://127.0.0.1:23281"),
            ]),
        );
        let router = make_test_router_with_workers(
            InspectingSelector {
                expected_hits: None,
                selected_worker: WorkerWithDpRank::new(7, 0),
            },
            None,
            workers,
        )
        .await;
        let candidates = RouterHintRootCandidates {
            block_hashes: vec![
                ExternalSequenceBlockHash(101),
                ExternalSequenceBlockHash(102),
            ],
            owner_prefix_blocks: vec![(WorkerWithDpRank::new(7, 1).into(), 2)],
            routing_snapshot: None,
        };

        let hint =
            router.router_hint_for_selection(WorkerWithDpRank::new(7, 0), 0, Some(&candidates));

        assert_eq!(
            hint,
            Some(RouterHint {
                source_control_endpoint: "tcp://127.0.0.1:23281".to_string(),
                block_hashes: vec![
                    ExternalSequenceBlockHash(101),
                    ExternalSequenceBlockHash(102)
                ],
            })
        );
    }

    #[tokio::test]
    async fn router_hint_skips_sources_without_usable_endpoint() {
        for source_endpoint in [None, Some("")] {
            let mut workers = HashMap::new();
            workers.insert(7, router_hint_runtime_config(Some("tcp://127.0.0.1:23280")));
            workers.insert(8, router_hint_runtime_config(source_endpoint));
            let router = make_test_router_with_workers(
                InspectingSelector {
                    expected_hits: None,
                    selected_worker: WorkerWithDpRank::new(7, 0),
                },
                None,
                workers,
            )
            .await;
            let candidates = RouterHintRootCandidates {
                block_hashes: vec![
                    ExternalSequenceBlockHash(101),
                    ExternalSequenceBlockHash(102),
                ],
                owner_prefix_blocks: vec![(WorkerWithDpRank::new(8, 0).into(), 2)],
                routing_snapshot: None,
            };

            let hint =
                router.router_hint_for_selection(WorkerWithDpRank::new(7, 0), 0, Some(&candidates));

            assert_eq!(hint, None);
        }
    }

    #[tokio::test]
    async fn router_hint_skips_sources_with_different_worker_type() {
        let mut workers = HashMap::new();
        workers.insert(
            7,
            router_hint_runtime_config_with_worker_type(Some("tcp://127.0.0.1:23280"), "prefill"),
        );
        workers.insert(
            8,
            router_hint_runtime_config_with_worker_type(Some("tcp://127.0.0.1:23281"), "decode"),
        );
        let router = make_test_router_with_workers(
            InspectingSelector {
                expected_hits: None,
                selected_worker: WorkerWithDpRank::new(7, 0),
            },
            None,
            workers,
        )
        .await;
        let candidates = RouterHintRootCandidates {
            block_hashes: vec![
                ExternalSequenceBlockHash(101),
                ExternalSequenceBlockHash(102),
            ],
            owner_prefix_blocks: vec![(WorkerWithDpRank::new(8, 0).into(), 2)],
            routing_snapshot: None,
        };

        let hint =
            router.router_hint_for_selection(WorkerWithDpRank::new(7, 0), 0, Some(&candidates));

        assert_eq!(hint, None);
    }

    #[tokio::test]
    async fn router_hint_selects_source_with_matching_worker_type() {
        let mut workers = HashMap::new();
        workers.insert(
            7,
            router_hint_runtime_config_with_worker_type(Some("tcp://127.0.0.1:23280"), "prefill"),
        );
        workers.insert(
            8,
            router_hint_runtime_config_with_worker_type(Some("tcp://127.0.0.1:23281"), "prefill"),
        );
        workers.insert(
            9,
            router_hint_runtime_config_with_worker_type(Some("tcp://127.0.0.1:23282"), "decode"),
        );
        workers.insert(
            10,
            router_hint_runtime_config_with_worker_type(Some("tcp://127.0.0.1:23283"), "decode"),
        );
        let router = make_test_router_with_workers(
            InspectingSelector {
                expected_hits: None,
                selected_worker: WorkerWithDpRank::new(7, 0),
            },
            None,
            workers,
        )
        .await;
        let candidates = RouterHintRootCandidates {
            block_hashes: vec![
                ExternalSequenceBlockHash(101),
                ExternalSequenceBlockHash(102),
                ExternalSequenceBlockHash(103),
            ],
            owner_prefix_blocks: vec![
                (WorkerWithDpRank::new(8, 0).into(), 2),
                (WorkerWithDpRank::new(9, 0).into(), 3),
            ],
            routing_snapshot: None,
        };

        let prefill_hint =
            router.router_hint_for_selection(WorkerWithDpRank::new(7, 0), 0, Some(&candidates));
        assert_eq!(
            prefill_hint,
            Some(RouterHint {
                source_control_endpoint: "tcp://127.0.0.1:23281".to_string(),
                block_hashes: vec![
                    ExternalSequenceBlockHash(101),
                    ExternalSequenceBlockHash(102),
                ],
            })
        );

        let decode_hint =
            router.router_hint_for_selection(WorkerWithDpRank::new(10, 0), 0, Some(&candidates));
        assert_eq!(
            decode_hint,
            Some(RouterHint {
                source_control_endpoint: "tcp://127.0.0.1:23282".to_string(),
                block_hashes: vec![
                    ExternalSequenceBlockHash(101),
                    ExternalSequenceBlockHash(102),
                    ExternalSequenceBlockHash(103),
                ],
            })
        );
    }

    #[tokio::test]
    async fn router_hint_resolves_persistent_owner_without_state_agent_fallback() {
        let target = WorkerWithDpRank::new(7, 0);
        let stale_source = WorkerWithDpRank::new(8, 0);
        let mut workers = HashMap::new();
        workers.insert(7, router_hint_runtime_config(None));
        let mut stale_source_config =
            router_hint_runtime_config(Some("tcp://stale-worker-endpoint:23280"));
        stale_source_config.kv_event_source_mode = Some("state_agent_v2".to_string());
        workers.insert(8, stale_source_config);
        let router = make_test_router_with_workers(
            InspectingSelector {
                expected_hits: None,
                selected_worker: target,
            },
            None,
            workers,
        )
        .await;
        let owner = router_hint_cache_owner();
        let owner_key = ResidencyOwner::cache_owner(owner).compact_key();
        let candidates = RouterHintRootCandidates {
            block_hashes: vec![
                ExternalSequenceBlockHash(101),
                ExternalSequenceBlockHash(102),
            ],
            owner_prefix_blocks: vec![
                (RouterHintCandidateSource::Worker(stale_source), 2),
                (RouterHintCandidateSource::CacheOwner(owner_key), 2),
            ],
            routing_snapshot: Some(Arc::new(ResidencyRoutingSnapshot::new(
                ResidencyProjection::default(),
                [(
                    owner,
                    RouterHintSourceMetadata {
                        source_control_endpoint: "tcp://persistent-owner:23280".to_string(),
                        worker_type: "prefill".to_string(),
                    },
                    None,
                )],
            ))),
        };

        assert_eq!(
            router.router_hint_for_selection(target, 0, Some(&candidates)),
            Some(RouterHint {
                source_control_endpoint: "tcp://persistent-owner:23280".to_string(),
                block_hashes: vec![
                    ExternalSequenceBlockHash(101),
                    ExternalSequenceBlockHash(102),
                ],
            })
        );
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

    #[tokio::test]
    async fn client_availability_distinguishes_startup_from_last_worker_removal() {
        use dynamo_kv_router::scheduling::{RoutingEligibility, WorkerEligibilityError};

        const DECODE_WORKER: u64 = 1;
        const PREFILL_WORKER: u64 = 2;
        const PREFILL_PEER: u64 = 3;

        let component = make_test_component("availability-lifecycle").await;
        let decode = component.endpoint("decode").client().await.unwrap();
        let prefill = component.endpoint("prefill").client().await.unwrap();

        assert!(
            prefill.available_instance_ids().is_none(),
            "startup without a discovered worker is uninitialized"
        );

        decode.override_discovered_instances(vec![DECODE_WORKER]);
        prefill.override_discovered_instances(vec![PREFILL_WORKER, PREFILL_PEER]);

        // Keep scheduler candidates stale so every transition below is decided
        // by the Client's hard-availability snapshot alone.
        let workers = HashMap::from([
            (DECODE_WORKER, ModelRuntimeConfig::default()),
            (PREFILL_WORKER, ModelRuntimeConfig::default()),
            (PREFILL_PEER, ModelRuntimeConfig::default()),
        ]);
        let constraints = RoutingConstraints::default();
        let validate = |available: &HashSet<u64>, worker: u64| {
            let pinned = WorkerWithDpRank::from_worker_id(worker);
            RoutingEligibility::new(None, None, Some(pinned), &constraints)
                .with_available_workers(Some(available))
                .validate_worker_rank(&workers, pinned)
                .map(|_| ())
        };

        let available = prefill.available_instance_ids().unwrap();
        assert!(validate(available.as_ref(), PREFILL_WORKER).is_ok());

        prefill.override_discovered_instances(vec![PREFILL_PEER]);
        let available = prefill.available_instance_ids().unwrap();
        assert_eq!(
            validate(available.as_ref(), PREFILL_WORKER).unwrap_err(),
            WorkerEligibilityError::WorkerNotRoutable {
                worker_id: PREFILL_WORKER
            }
        );
        assert!(
            decode
                .available_instance_ids()
                .unwrap()
                .contains(&DECODE_WORKER),
            "prefill removal must not alter decode availability"
        );

        prefill.override_discovered_instances(Vec::new());
        let available = prefill
            .available_instance_ids()
            .expect("last-worker removal is authoritative after discovery initialized");
        assert!(available.is_empty());
        assert_eq!(
            validate(available.as_ref(), PREFILL_PEER).unwrap_err(),
            WorkerEligibilityError::WorkerNotRoutable {
                worker_id: PREFILL_PEER
            }
        );

        prefill.override_discovered_instances(vec![PREFILL_WORKER, PREFILL_PEER]);
        let available = prefill.available_instance_ids().unwrap();
        assert!(validate(available.as_ref(), PREFILL_WORKER).is_ok());
    }
}
