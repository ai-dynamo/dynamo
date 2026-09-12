// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::{HashMap, HashSet},
    fmt,
    sync::Arc,
    time::Instant,
};

use anyhow::Result;
use dynamo_kv_router::{
    DEFAULT_ROUTING_GROUP, KvSchedulerError, PrefillLoadEstimator, RoutingPartitionRef,
    SharedKvCache, TrackingHashAlgorithm, TrackingHashContext, TrackingHashScope,
    WorkerSelectionPolicy, WorkerSelectionPolicyFactory,
    config::{KvRouterConfig, RouterConfigOverride, min_initial_workers_from_env},
    indexer::{
        ApproximateLruIncarnation, ApproximateLruStats, KvRouterError, RoutingDecisionHashes,
    },
    kv_hints::KvHint,
    protocols::KV_EVENT_SUBJECT,
    protocols::{
        BlockExtraInfo, BlockHashOptions, PrefillLoadHint, RouterEvent, RouterRequest,
        RouterResponse, RoutingConstraints, TokensWithHashes, WorkerConfigLike, WorkerId,
        WorkerWithDpRank, compute_block_hash_for_seq,
    },
    scheduling::{
        CacheHitEstimates, OverlapAnalysis, OverloadedWorkerProvider, PotentialLoad,
        WorkerAvailabilityProvider, effective_prefill_tokens,
        overlap::cache_hit_estimates_from_tiered_matches,
        queue::{BookingHandle, SchedulerBookingDescriptor},
    },
    selector::WorkerInputs,
    services::selection::{
        PromptView, Selected, SelectionAdmission, SelectionError, SelectionOperation,
        SelectionOutcome, SelectionRun, SessionBinding,
    },
};
use dynamo_runtime::{
    CancellationToken,
    component::{Client, Endpoint},
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

// Re-export from dynamo-kv-router crate
pub use dynamo_kv_router::protocols;
pub use dynamo_kv_router::scheduling;

pub(crate) mod embedded;
pub mod encoder_router;
pub mod indexer;
pub mod metrics;
pub(crate) mod metrics_subscriber;
pub mod prefill_router;
pub mod publisher;
mod request_lease;
mod routing_host;
pub(crate) mod routing_load;
pub mod sequence;
pub mod shared_cache;

pub use dynamo_kv_router::scheduling::OverlapScoresResponse;
pub use embedded::{install_worker_selection_policy_registry, worker_selection_policy_registry};
pub use encoder_router::EncoderRouter;
pub use indexer::Indexer;
pub use prefill_router::PrefillRouter;
pub use routing_host::{KvPushRouter, RoutingHost};
pub use routing_load::{
    ManagedKvRouter, RouterLoadSource, RoutingLoadContext, SchedulerLoadSender,
};

use crate::{
    discovery::{KvSourceMembershipWatch, RuntimeConfigWatch},
    kv_router::sequence::{SequenceError, SequenceRequest},
    local_model::runtime_config::ModelRuntimeConfig,
    worker_type::WorkerType,
};

/// Where a `KvRouter` gets the worker-selection policy its partition runs.
#[derive(Clone, Default)]
pub enum SelectionPolicySource {
    /// The linked policy `KvRouterConfig` names for the worker role, resolved
    /// from the installed [`worker_selection_policy_registry`]; Dynamo's
    /// default scorer and picker when it names none.
    #[default]
    Registry,
    /// This factory, called once per routing partition.
    Factory(WorkerSelectionPolicyFactory),
}

impl SelectionPolicySource {
    /// Resolve to the factory the partition will call. `label` is the worker
    /// pool name the default policy logs under.
    pub fn resolve(
        &self,
        config: &KvRouterConfig,
        worker_type: WorkerType,
        label: &'static str,
    ) -> Result<WorkerSelectionPolicyFactory> {
        match self {
            Self::Factory(factory) => Ok(factory.clone()),
            Self::Registry => Ok(
                match worker_selection_policy_registry()
                    .resolve_for_worker_type(config, worker_type)?
                {
                    Some(factory) => factory,
                    None => Arc::new(move |config: &KvRouterConfig, _worker_type, _partition| {
                        WorkerSelectionPolicy::default(config.clone(), label)
                    }),
                },
            ),
        }
    }
}

/// The optional worker inputs one instance of `factory`'s policy consumes.
pub(crate) fn policy_worker_inputs(
    factory: &WorkerSelectionPolicyFactory,
    config: &KvRouterConfig,
    worker_type: WorkerType,
    model_name: Option<&str>,
) -> WorkerInputs {
    let partition = RoutingPartitionRef::new(
        model_name.unwrap_or(dynamo_kv_router::services::selection::DEFAULT_MODEL_NAME),
        DEFAULT_ROUTING_GROUP,
    );
    dynamo_kv_router::selector::WorkerSelector::<ModelRuntimeConfig>::required_worker_inputs(
        &factory(config, worker_type, partition),
    )
}

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
    use crate::protocols::common::extensions::{AgentContext, InputTrigger};
    use dynamo_kv_router::{SessionContext, WorkerSelectionInputTrigger};

    // Keep this exhaustive so a new wire-level field must be handled here.
    let AgentContext {
        session_id,
        parent_session_id,
        session_final,
        compaction: _,
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
        kv_hint: Option<KvHint>,
    },
    QueueRejected {
        rejection: scheduling::QueueRejection,
    },
}

#[derive(Debug, Clone, Copy)]
pub(super) enum FindBestMatchAdmission {
    WithAdmission,
    WithoutAdmission,
}

/// A routed outcome with the booking's handle, when the request was booked.
/// Dropping the handle frees the booking; the caller commits it into its own
/// cleanup.
#[doc(hidden)]
pub struct AdmittedFindBestMatchOutcome {
    pub(super) outcome: FindBestMatchOutcome,
    pub(super) booking: Option<BookingHandle>,
    /// The selected worker's scheduler-load snapshot; set only by advisory
    /// probes (`FindBestMatchAdmission::WithoutAdmission`), which never book.
    pub(super) advisory_load: Option<scheduling::AdvisoryWorkerLoad>,
}

impl AdmittedFindBestMatchOutcome {
    #[doc(hidden)]
    pub fn into_parts(self) -> (FindBestMatchOutcome, Option<BookingHandle>) {
        (self.outcome, self.booking)
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

// for metric publishing (push-based)
pub const KV_METRICS_SUBJECT: &str = "kv_metrics";
pub const MULTIMODAL_EMBEDDING_CACHE_SUBJECT: &str = "multimodal_embedding_cache";

// for inter-router comms
pub const ACTIVE_SEQUENCES_SUBJECT: &str = "active_sequences_events";

// for worker-local kvindexer query
pub const WORKER_KV_INDEXER_BUFFER_SIZE: usize = 1024; // store 1024 most recent events in worker buffer

fn map_scheduler_error(error: scheduling::KvSchedulerError) -> anyhow::Error {
    // Keep the two overload cases apart. A single overloaded worker can be
    // retried elsewhere; a pool with no free worker cannot, and migrating it
    // would just bounce the request around. A filter rejection is unavailable,
    // not overload, and becomes HTTP 503.
    let (error_type, overloaded) = match error {
        scheduling::KvSchedulerError::PinnedWorkerOverloaded { .. } => {
            (ErrorType::WorkerOverloaded, true)
        }
        scheduling::KvSchedulerError::AllEligibleWorkersOverloaded => {
            (ErrorType::ResourceExhausted, true)
        }
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

/// A KvRouter only decides which worker you should use. It doesn't send you there.
/// TODO: Rename this to indicate it only selects a worker, it does not route.
pub struct KvRouter {
    indexer: Indexer,
    selection: embedded::EmbeddedSelection,
    required_worker_inputs: dynamo_kv_router::selector::WorkerInputs,
    workers_with_configs: RuntimeConfigWatch,
    block_size: u32,
    kv_router_config: KvRouterConfig,
    prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
    cancellation_token: CancellationToken,
    client: Client,
    is_eagle: bool,
    ingress: Arc<indexer::RuntimeIngress>,
    tracking_hash: TrackingHashContext,
    tracking_model_name: String,
    approximate_lru_ranks: ApproximateLruRanks,
    request_leases: request_lease::RequestLeaseManager,
    /// Optional external shared KV cache pool. When present, `find_best_match`
    /// queries it in parallel with the indexer and factors shared hits into scoring.
    shared_cache: Option<Arc<dyn SharedKvCache>>,
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

impl KvRouter {
    #[allow(clippy::too_many_arguments)]
    pub async fn new(
        endpoint: Endpoint,
        client: Client,
        workers_with_configs: RuntimeConfigWatch,
        kv_source_membership: Option<KvSourceMembershipWatch>,
        block_size: u32,
        policy: SelectionPolicySource,
        kv_router_config: Option<KvRouterConfig>,
        prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
        metric_worker_type: &'static str,
        model_name: Option<String>,
        is_eagle: bool,
        shared_cache: Option<Arc<dyn SharedKvCache>>,
        lora_filter: Option<Arc<crate::lora::LoraFilter>>,
    ) -> Result<Self> {
        Self::new_with_worker_role(
            endpoint,
            client,
            workers_with_configs,
            kv_source_membership,
            block_size,
            policy,
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
        policy: SelectionPolicySource,
        kv_router_config: Option<KvRouterConfig>,
        prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
        worker_role: Option<WorkerType>,
        metric_worker_type: &'static str,
        model_name: Option<String>,
        is_eagle: bool,
        shared_cache: Option<Arc<dyn SharedKvCache>>,
        lora_filter: Option<Arc<crate::lora::LoraFilter>>,
    ) -> Result<Self> {
        let parent_token = endpoint.component().drt().child_token();
        let scheduler_load = SchedulerLoadSender::disabled(parent_token.child_token());

        Self::new_with_worker_role_and_scheduler_load(
            endpoint,
            client,
            workers_with_configs,
            kv_source_membership,
            block_size,
            policy,
            kv_router_config,
            prefill_load_estimator,
            worker_role,
            metric_worker_type,
            model_name,
            is_eagle,
            shared_cache,
            lora_filter,
            scheduler_load,
            parent_token,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn new_with_worker_role_and_scheduler_load(
        endpoint: Endpoint,
        client: Client,
        workers_with_configs: RuntimeConfigWatch,
        kv_source_membership: Option<KvSourceMembershipWatch>,
        block_size: u32,
        policy: SelectionPolicySource,
        kv_router_config: Option<KvRouterConfig>,
        prefill_load_estimator: Option<Arc<dyn PrefillLoadEstimator>>,
        worker_role: Option<WorkerType>,
        metric_worker_type: &'static str,
        model_name: Option<String>,
        is_eagle: bool,
        shared_cache: Option<Arc<dyn SharedKvCache>>,
        lora_filter: Option<Arc<crate::lora::LoraFilter>>,
        scheduler_load: SchedulerLoadSender,
        parent_token: CancellationToken,
    ) -> Result<Self> {
        let kv_router_config = kv_router_config.unwrap_or_default();
        kv_router_config.validate().map_err(anyhow::Error::msg)?;
        let worker_type = worker_role.unwrap_or(WorkerType::Aggregated);
        let policy_factory = policy.resolve(&kv_router_config, worker_type, metric_worker_type)?;
        let required_worker_inputs = policy_worker_inputs(
            &policy_factory,
            &kv_router_config,
            worker_type,
            model_name.as_deref(),
        );
        // ModelManager gates client construction as well, but preserve the capability boundary for
        // direct KvRouter callers.
        let shared_cache = if required_worker_inputs.contains(WorkerInputs::CACHE) {
            shared_cache
        } else {
            None
        };
        let tracking_hash = TrackingHashContext::from_config(&kv_router_config)?;
        let tracking_model_name =
            resolve_tracking_model_name(tracking_hash.algorithm(), model_name.as_deref())?;
        let kv_event_source_requirement =
            KvEventSourceRequirement::derive(worker_role, &kv_router_config);
        let cache_required = required_worker_inputs.contains(WorkerInputs::CACHE)
            || kv_router_config.serve_indexer
            || matches!(
                kv_event_source_requirement,
                KvEventSourceRequirement::ConditionalDisaggDecodeCache
                    | KvEventSourceRequirement::Unknown
            );
        let component = endpoint.component();
        // All chooser tasks are children of the routing load context owner.
        let cancellation_token = parent_token.child_token();
        let cancellation_guard = cancellation_token.clone().drop_guard();
        let min_initial_workers = min_initial_workers_from_env()?;

        let ingress = indexer::RuntimeIngress::start(indexer::RuntimeIngressArgs {
            endpoint: &endpoint,
            kv_router_config: &kv_router_config,
            block_size,
            model_name: model_name.as_deref(),
            worker_role,
            metric_worker_type,
            cache_required,
            kv_event_source_requirement,
            kv_source_membership,
            cancellation_token: cancellation_token.clone(),
        })
        .await?;
        let indexer = ingress.indexer().clone();
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

        let client_for_overload = client.clone();
        let overloaded_worker_provider: OverloadedWorkerProvider =
            Arc::new(move || client_for_overload.overloaded_instance_ids());

        let client_for_availability = client.clone();
        let available_worker_provider: WorkerAvailabilityProvider =
            Arc::new(move || client_for_availability.available_instance_ids());

        let request_leases =
            request_lease::RequestLeaseManager::new(cancellation_token.child_token());
        let (selection, replica_ingress) = embedded::EmbeddedSelection::start(
            embedded::EmbeddedSelectionArgs {
                kv_router_config: kv_router_config.clone(),
                worker_role,
                metric_worker_type,
                model_name: model_name.clone(),
                block_size,
                is_eagle,
                prefill_load_estimator: prefill_load_estimator.clone(),
                overloaded_worker_provider,
                available_worker_provider,
                shared_cache: shared_cache.clone(),
                lora_worker_filter: lora_filter.map(|filter| {
                    filter as Arc<dyn dynamo_kv_router::scheduling::LoraWorkerFilter>
                }),
                ingress: Arc::clone(&ingress)
                    as Arc<dyn dynamo_kv_router::services::selection::KvEventIngress>,
                scheduler_load,
                endpoint: endpoint.clone(),
                router_id: endpoint.drt().discovery().instance_id(),
                policy_factory,
            },
            workers_with_configs.clone(),
            Some(Arc::new(request_leases.clone())),
            cancellation_token.child_token(),
        )
        .await?;
        request_leases.set_scheduler(selection.scheduler().booking_cleanup());
        // Inbound lifecycle events start only now that their consumer can
        // release bookings.
        replica_ingress.start().await;
        tracing::info!("KV Routing initialized");
        let cancellation_token = cancellation_guard.disarm();
        Ok(Self {
            indexer,
            selection,
            required_worker_inputs,
            workers_with_configs,
            block_size,
            kv_router_config,
            prefill_load_estimator,
            cancellation_token,
            client,
            is_eagle,
            ingress,
            tracking_hash,
            tracking_model_name,
            approximate_lru_ranks,
            request_leases,
            shared_cache,
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
        self.ingress.set_task_guard(task_guard.clone());
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

    pub async fn record_routing_decision(
        &self,
        mut tokens_with_hashes: TokensWithHashes,
        worker: WorkerWithDpRank,
    ) -> Result<(), KvRouterError> {
        // This public compatibility path has no admitted attempt identity. LRU
        // mutations require acquire/release fencing, so leave them unchanged.
        if self.indexer.uses_approximate_lru() {
            return Ok(());
        }
        self.indexer
            .process_routing_decision_for_request(&mut tokens_with_hashes, worker)
            .await
    }

    /// Record an update that has no admitted request attempt. Capacity-bounded
    /// LRU requires acquire/release lifecycle fencing, so query-only callers
    /// intentionally leave it unchanged. TTL recording retains its existing behavior.
    #[doc(hidden)]
    pub async fn record_query_only_routing_decision(
        &self,
        tokens_with_hashes: TokensWithHashes,
        worker: WorkerWithDpRank,
    ) -> Result<(), KvRouterError> {
        if self.indexer.uses_approximate_lru() {
            return Ok(());
        }
        self.record_routing_decision(tokens_with_hashes, worker)
            .await
    }

    /// Install the detached lifecycle used by public request-ID admissions.
    /// Registration precedes the fallible routing update, and its temporary
    /// owner releases the exact booking if this future is cancelled.
    #[doc(hidden)]
    pub async fn enroll_public_request_attempt(
        &self,
        booking: BookingHandle,
        routing_decision: Option<TokensWithHashes>,
    ) -> Result<(), KvRouterError> {
        // Nothing awaits between taking the booking over and registering it.
        let booking = booking.commit();
        let worker = booking.worker;
        let lru_registration = self.approximate_lru_rank_registration(worker);
        let approximate_lru = lru_registration.and_then(|registration| {
            self.indexer.begin_approximate_lru_request(
                worker,
                registration.incarnation,
                booking.attempt_id,
            )
        });
        let enrollment = self
            .request_leases
            .register_detached(booking, approximate_lru.clone());

        let Some(mut tokens_with_hashes) = routing_decision else {
            enrollment.commit();
            return Ok(());
        };
        if let Some(mut lease) = approximate_lru {
            let token_count = tokens_with_hashes.len();
            let local_hashes = tokens_with_hashes.get_or_compute_block_hashes().to_vec();
            let sequence_hashes = tokens_with_hashes.get_or_compute_seq_hashes().to_vec();
            let private_blocks = routing_host::prompt_private_blocks(
                token_count,
                local_hashes.len(),
                usize::try_from(self.block_size).unwrap_or(usize::MAX),
                self.is_eagle,
            );
            if let Err(error) = lease
                .acquire(
                    RoutingDecisionHashes {
                        local_hashes,
                        sequence_hashes,
                    },
                    private_blocks,
                )
                .await
            {
                enrollment.finish().await;
                return Err(error);
            }
            enrollment.commit();
            return Ok(());
        }
        if self.indexer.uses_approximate_lru() {
            enrollment.commit();
            return Ok(());
        }
        let result = self
            .record_routing_decision(tokens_with_hashes, worker)
            .await;
        match result {
            Ok(()) => {
                enrollment.commit();
                Ok(())
            }
            Err(error) => {
                enrollment.finish().await;
                Err(error)
            }
        }
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

    /// Give these tokens, find the worker with the best weighted cache hit.
    /// Returns the full match details for the selected worker.
    ///
    /// When `pinned_worker` is Some, scheduling and queueing are constrained to
    /// that exact worker/rank.
    ///
    /// When `allowed_worker_ids` is Some, only workers in that set are considered for selection.
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
        let admitted = self
            .find_best_match_details_with_policy_class_admitted(
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
                session_context,
                expected_output_tokens,
                pinned_worker,
                allowed_worker_ids,
                routing_constraints,
            )
            .await?;
        if let Some(booking) = admitted.booking {
            self.enroll_public_request_attempt(booking, None).await?;
        }
        Ok(admitted.outcome)
    }

    /// Return the admitted routing wrapper without enrolling its booking handle
    /// in a detached lease. Internal bindings use this to attach optional LRU state
    /// before installing the one shared request lease.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub async fn find_best_match_details_with_policy_class_admitted(
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
    ) -> anyhow::Result<AdmittedFindBestMatchOutcome> {
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
            session_context,
            expected_output_tokens,
            None,
            pinned_worker,
            allowed_worker_ids,
            routing_constraints,
            FindBestMatchAdmission::WithAdmission,
        )
        .await
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
        session_context: Option<dynamo_kv_router::SessionContext>,
        expected_output_tokens: Option<u32>,
        affinity_target: Option<dynamo_kv_router::protocols::WorkerAffinityTarget>,
        pinned_worker: Option<WorkerWithDpRank>,
        allowed_worker_ids: Option<HashSet<WorkerId>>,
        routing_constraints: RoutingConstraints,
        admission: FindBestMatchAdmission,
    ) -> anyhow::Result<AdmittedFindBestMatchOutcome> {
        let start = Instant::now();
        if update_states && context_id.is_none() {
            anyhow::bail!("context_id must be provided if update_states is true");
        }
        let is_admitted_routing = matches!(admission, FindBestMatchAdmission::WithAdmission);
        let core_admission = match admission {
            FindBestMatchAdmission::WithAdmission if update_states => SelectionAdmission::Lease {
                request_id: context_id.expect("validated above").to_string(),
            },
            FindBestMatchAdmission::WithAdmission => SelectionAdmission::Query {
                request_id: context_id.map(str::to_string),
            },
            FindBestMatchAdmission::WithoutAdmission => SelectionAdmission::Advisory {
                request_id: context_id.map(str::to_string),
            },
        };
        let SelectionRun {
            result: outcome,
            lookup,
        } = self
            .selection
            .run_selection(SelectionOperation {
                key: self.selection.partition_key().clone(),
                prompt: PromptView {
                    token_ids: Some(tokens),
                    mm_routing_info: None,
                    block_mm_infos,
                    block_hashes: None,
                    sequence_hashes: None,
                    isl_tokens: None,
                    lora_name: lora_name.as_deref(),
                    cache_namespace: cache_namespace.as_deref(),
                    is_eagle: Some(self.is_eagle),
                },
                router_config_override: router_config_override.cloned(),
                expected_output_tokens,
                priority_jump,
                strict_priority,
                policy_class,
                session_context,
                session: SessionBinding::None,
                affinity_target,
                pinned_worker,
                allowed_worker_ids,
                routing_constraints,
                admission: core_admission,
                track_active_blocks: self.kv_router_config.router_track_active_blocks,
                return_routing_hashes,
                replay_id: None,
            })
            .await;
        if lookup.is_some_and(|lookup| lookup.shared_cache_error)
            && let Some(m) = metrics::RoutingOverheadMetrics::get()
        {
            m.inc_shared_cache_errors();
        }
        let selected = match outcome {
            Ok(SelectionOutcome::Selected(selected)) => selected,
            Ok(SelectionOutcome::QueueRejected { rejection }) => {
                return Ok(AdmittedFindBestMatchOutcome {
                    outcome: FindBestMatchOutcome::QueueRejected { rejection },
                    booking: None,
                    advisory_load: None,
                });
            }
            Err(SelectionError::Scheduler(error)) => return Err(map_scheduler_error(error)),
            Err(SelectionError::Indexer(error)) => return Err(error.into()),
            // The partition has no schedulable worker: the scheduler's own answer.
            Err(SelectionError::NotReady(_)) => {
                return Err(map_scheduler_error(KvSchedulerError::NoEndpoints));
            }
            Err(error) => return Err(error.into()),
        };
        let Selected {
            response,
            advisory_load,
            isl_tokens,
            kv_hint,
            routing_hashes,
            shared_cache_hits,
            booking,
            ..
        } = selected;
        if update_states && is_admitted_routing && booking.is_none() {
            anyhow::bail!("booked selection returned no booking handle");
        }
        let routing_hashes = routing_hashes.map(RoutingDecisionHashes::from_local_hashes);
        let overlap_blocks = response.effective_overlap_blocks.round() as u32;

        // Routing metrics stay scoped to requests admitted by this call.
        if is_admitted_routing {
            let total_elapsed = start.elapsed();
            if let (Some(lookup), Some(m)) = (lookup, metrics::RoutingOverheadMetrics::get()) {
                let hash_elapsed = lookup.block_hashing;
                let seq_hash_elapsed = hash_elapsed + lookup.seq_hashing;
                let find_matches_elapsed = seq_hash_elapsed + lookup.lookups;
                m.observe(
                    hash_elapsed,
                    seq_hash_elapsed,
                    lookup.indexer,
                    lookup.shared_cache,
                    find_matches_elapsed,
                    total_elapsed,
                );
            }
            if let (Some(hits), Some(m)) = (shared_cache_hits, metrics::RouterRequestMetrics::get())
            {
                let num_blocks = isl_tokens / self.block_size as usize;
                if num_blocks > 0 {
                    m.shared_cache_hit_rate
                        .observe(hits.total_hits as f64 / num_blocks as f64);
                }
                m.shared_cache_beyond_blocks
                    .observe(hits.hits_beyond(overlap_blocks) as f64);
            }
        }

        // Advisory selections carry no booking or hint and always a load
        // snapshot; admitted ones the reverse (`SelectionCore::select_or_reject`).
        debug_assert_eq!(
            advisory_load.is_some(),
            !is_admitted_routing,
            "advisory load is set exactly for without-admission selection"
        );
        Ok(AdmittedFindBestMatchOutcome {
            outcome: FindBestMatchOutcome::Routed {
                worker: response.best_worker,
                overlap_blocks,
                effective_overlap_blocks: response.effective_overlap_blocks,
                cached_tokens: response.cached_tokens,
                potential_decode_blocks: response.potential_decode_blocks as u64,
                routing_hashes,
                kv_hint,
            },
            booking,
            advisory_load,
        })
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

        let admission = self
            .selection
            .scheduler()
            .add_request_admitted(SequenceRequest {
                request_id: request_id.clone(),
                token_sequence: maybe_seq_hashes,
                track_prefill_tokens,
                expected_output_tokens,
                prefill_load_hint,
                worker,
                lora_name,
            })
            .await;
        let attempt_id = match admission {
            Ok(attempt_id) => attempt_id,
            Err(error) => {
                tracing::warn!(%request_id, %error, "Failed to add request");
                return;
            }
        };
        self.request_leases
            .register_detached(
                SchedulerBookingDescriptor {
                    request_id,
                    worker,
                    attempt_id,
                },
                None,
            )
            .commit();
    }

    pub async fn mark_prefill_completed(&self, request_id: &str) -> Result<(), SequenceError> {
        self.selection
            .scheduler()
            .mark_prefill_completed(request_id)
            .await?;
        self.request_leases.touch_request(request_id);
        Ok(())
    }

    pub async fn free(&self, request_id: &str) -> Result<(), SequenceError> {
        if self.request_leases.finish_request(request_id).await {
            return Ok(());
        }
        self.selection.scheduler().free(request_id).await
    }

    pub(crate) fn affinity_coordinator(
        &self,
        ttl: std::time::Duration,
        mode: crate::session_affinity::SessionAffinityMode,
    ) -> anyhow::Result<crate::session_affinity::AffinityCoordinator> {
        self.selection.affinity_coordinator(ttl, mode)
    }

    pub(crate) fn request_lease_manager(&self) -> &request_lease::RequestLeaseManager {
        &self.request_leases
    }

    pub(crate) async fn mark_prefill_completed_if_booking(
        &self,
        booking: &SchedulerBookingDescriptor,
    ) -> Result<(), KvSchedulerError> {
        self.selection
            .scheduler()
            .mark_prefill_completed_if_booking(booking)
            .await
            .map(|_| ())
    }

    /// Number of requests currently parked in the scheduler queue.
    pub fn pending_count(&self) -> usize {
        self.selection.scheduler().pending_count()
    }

    /// Sum of ISL tokens for requests currently parked in the scheduler queue.
    pub fn pending_isl_tokens(&self) -> usize {
        self.selection.scheduler().pending_isl_tokens()
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
        self.selection.worker_type()
    }

    /// Return the worker's unique global DP rank when it owns exactly one rank.
    pub fn unique_dp_rank_for_worker(&self, worker_id: WorkerId) -> Option<u32> {
        let configs = self.workers_with_configs.borrow();
        let config = configs.get(&worker_id)?;
        (config.data_parallel_size == 1).then_some(config.data_parallel_start_rank)
    }

    pub(crate) async fn enqueue_output_block_if_booking(
        &self,
        booking: &SchedulerBookingDescriptor,
        decay_fraction: Option<f64>,
    ) -> Result<(), KvSchedulerError> {
        self.selection
            .scheduler()
            .enqueue_output_block_if_booking(booking, decay_fraction)
            .await
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

        Ok(self.selection.scheduler().get_potential_loads(
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
impl AsyncEngine<SingleIn<RouterRequest>, ManyOut<Annotated<RouterResponse>>, Error> for KvRouter {
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

impl Drop for KvRouter {
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
    use dynamo_kv_router::{WorkerSelectionPolicyError, protocols::compute_seq_hash_for_block};
    use dynamo_runtime::{DistributedRuntime, Runtime, distributed::DistributedConfig};
    use tokio::sync::watch;

    use crate::local_model::runtime_config::ModelRuntimeConfig;
    use dynamo_kv_router::selector::{
        WorkerCandidate, WorkerFilter, WorkerInputView, WorkerPicker, WorkerSelectionContext,
    };

    #[test]
    fn all_filtered_workers_map_to_unavailable() {
        let error = map_scheduler_error(KvSchedulerError::AllEligibleWorkersFiltered);
        let dynamo_error = error
            .downcast_ref::<DynamoError>()
            .expect("filtered workers should produce a DynamoError");

        assert_eq!(dynamo_error.error_type(), ErrorType::Unavailable);

        let error = map_scheduler_error(KvSchedulerError::AllEligibleWorkersOverloaded);
        let dynamo_error = error
            .downcast_ref::<DynamoError>()
            .expect("overloaded workers should produce a DynamoError");

        assert_eq!(dynamo_error.error_type(), ErrorType::ResourceExhausted);
    }

    #[test]
    fn worker_selection_receives_complete_session_context() {
        use crate::protocols::common::extensions::{AgentContext, InputTrigger};
        use dynamo_kv_router::WorkerSelectionInputTrigger;

        let context = AgentContext {
            session_id: "child-session".into(),
            parent_session_id: Some("root-session".into()),
            session_final: Some(true),
            compaction: None,
            input_trigger: Some(InputTrigger::ToolResult),
        };

        let selection_context = to_worker_selection_session_context(&context);

        assert_eq!(selection_context.session_id(), "child-session");
        assert_eq!(selection_context.parent_session_id(), Some("root-session"));
        assert_eq!(selection_context.session_final(), Some(true));
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

    /// Picks a fixed worker after checking the shared-cache hits the router
    /// projected into the candidate table.
    struct FixedPicker {
        expected_shared_blocks: Option<u32>,
        worker: WorkerWithDpRank,
    }

    impl WorkerPicker for FixedPicker {
        fn required_worker_inputs(&self) -> WorkerInputs {
            WorkerInputs::CACHE | WorkerInputs::LOAD
        }

        fn pick(
            &mut self,
            _context: &WorkerSelectionContext<'_>,
            input: WorkerInputView<'_>,
        ) -> Result<usize, WorkerSelectionPolicyError> {
            let row = input
                .candidates()
                .iter()
                .position(|candidate| candidate.worker() == self.worker)
                .ok_or_else(|| WorkerSelectionPolicyError::failed("fixed worker not eligible"))?;
            let shared = input
                .cache()
                .map(|cache| cache[row].shared_beyond_device_blocks())
                .filter(|blocks| *blocks > 0);
            assert_eq!(shared, self.expected_shared_blocks);
            Ok(row)
        }
    }

    struct RejectAll;

    impl WorkerFilter for RejectAll {
        fn keep(
            &mut self,
            _context: &WorkerSelectionContext<'_>,
            _candidate: &WorkerCandidate,
        ) -> Result<bool, WorkerSelectionPolicyError> {
            Ok(false)
        }
    }

    struct LoadOnlyPicker;

    impl WorkerPicker for LoadOnlyPicker {
        fn required_worker_inputs(&self) -> WorkerInputs {
            WorkerInputs::LOAD
        }

        fn pick(
            &mut self,
            _context: &WorkerSelectionContext<'_>,
            _input: WorkerInputView<'_>,
        ) -> Result<usize, WorkerSelectionPolicyError> {
            unreachable!("capability construction test does not select a worker")
        }
    }

    fn fixed_policy(
        expected_shared_blocks: Option<u32>,
        worker: WorkerWithDpRank,
    ) -> SelectionPolicySource {
        SelectionPolicySource::Factory(Arc::new(move |config: &KvRouterConfig, _, _| {
            WorkerSelectionPolicy::new(
                config.clone(),
                "decode",
                Vec::new(),
                Box::new(FixedPicker {
                    expected_shared_blocks,
                    worker,
                }),
            )
        }))
    }

    fn picker_policy(
        picker: impl Fn() -> Box<dyn WorkerPicker> + Send + Sync + 'static,
    ) -> SelectionPolicySource {
        SelectionPolicySource::Factory(Arc::new(move |config: &KvRouterConfig, _, _| {
            WorkerSelectionPolicy::new(config.clone(), "decode", Vec::new(), picker())
        }))
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
        make_router(
            "role-aware-subscription",
            HashMap::from([(7, ModelRuntimeConfig::default())]),
            16,
            SelectionPolicySource::Registry,
            None,
            worker_role,
            "decode",
            KvRouterConfig {
                skip_initial_worker_wait: true,
                router_event_threads: 1,
                ..Default::default()
            },
        )
        .await
    }

    #[tokio::test]
    async fn constructor_skips_sources_for_decode_but_preserves_unknown_behavior() {
        let router = make_router_without_membership(Some(WorkerType::Decode))
            .await
            .expect("ordinary decode must not require KV source membership");
        assert!(!router.ingress.has_subscription());

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
        let router = make_router(
            "load-only-capability",
            HashMap::from([(7, ModelRuntimeConfig::default())]),
            16,
            picker_policy(|| Box::new(LoadOnlyPicker)),
            Some(Arc::new(FakeSharedCache {
                hits: None,
                should_error: false,
            })),
            Some(WorkerType::Prefill),
            "prefill",
            KvRouterConfig {
                skip_initial_worker_wait: true,
                router_event_threads: 1,
                ..Default::default()
            },
        )
        .await
        .unwrap();

        assert_eq!(router.required_worker_inputs(), WorkerInputs::LOAD);
        assert!(matches!(router.indexer, Indexer::None));
        assert!(!router.ingress.has_subscription());
        assert!(router.shared_cache.is_none());
        assert!(matches!(
            router.dump_events().await,
            Err(KvRouterError::Unsupported(message)) if message == "event dumping requires a KV indexer"
        ));
    }

    /// One router over `workers`, no KV event subscription; `role` is the
    /// worker role and its metric label.
    #[allow(clippy::too_many_arguments)]
    async fn make_router(
        name: &str,
        workers: HashMap<WorkerId, ModelRuntimeConfig>,
        block_size: u32,
        policy: SelectionPolicySource,
        shared_cache: Option<Arc<dyn SharedKvCache>>,
        worker_role: Option<WorkerType>,
        metric_label: &'static str,
        config: KvRouterConfig,
    ) -> Result<KvRouter> {
        make_router_with_watch(
            name,
            workers,
            block_size,
            policy,
            shared_cache,
            worker_role,
            metric_label,
            config,
        )
        .await
        .map(|(router, _tx)| router)
    }

    /// [`make_router`] that also hands back the worker-config watch sender.
    #[allow(clippy::too_many_arguments)]
    async fn make_router_with_watch(
        name: &str,
        workers: HashMap<WorkerId, ModelRuntimeConfig>,
        block_size: u32,
        policy: SelectionPolicySource,
        shared_cache: Option<Arc<dyn SharedKvCache>>,
        worker_role: Option<WorkerType>,
        metric_label: &'static str,
        config: KvRouterConfig,
    ) -> Result<(
        KvRouter,
        watch::Sender<HashMap<WorkerId, ModelRuntimeConfig>>,
    )> {
        let component = make_test_component(name).await;
        let endpoint = component.endpoint("backend");
        let client = endpoint.client().await?;
        let (tx, rx) = watch::channel(workers);
        let router = KvRouter::new_with_worker_role(
            endpoint,
            client,
            rx,
            None,
            block_size,
            policy,
            Some(config),
            None,
            worker_role,
            metric_label,
            None,
            false,
            shared_cache,
            None,
        )
        .await?;
        Ok((router, tx))
    }

    /// Workers that appear on the config watch after construction become
    /// routable even when the router did not wait for an initial worker.
    #[tokio::test]
    async fn skip_initial_worker_wait_still_monitors_worker_config_updates() {
        let (router, tx) = make_router_with_watch(
            "skip-initial-worker-watch",
            HashMap::from([(0, ModelRuntimeConfig::default())]),
            2,
            SelectionPolicySource::Registry,
            None,
            Some(WorkerType::Decode),
            "decode",
            KvRouterConfig {
                skip_initial_worker_wait: true,
                use_kv_events: false,
                router_track_active_blocks: false,
                ..Default::default()
            },
        )
        .await
        .unwrap();
        // Catalog upserts reach the scheduler's slots asynchronously.
        async fn wait_for_worker(router: &KvRouter, worker_id: WorkerId) -> Vec<PotentialLoad> {
            tokio::time::timeout(std::time::Duration::from_secs(5), async {
                loop {
                    let loads = router
                        .get_potential_loads(&[1, 2, 3, 4], None, None, None, None)
                        .await
                        .unwrap();
                    if loads.iter().any(|load| load.worker_id == worker_id) {
                        return loads;
                    }
                    tokio::task::yield_now().await;
                }
            })
            .await
            .unwrap_or_else(|_| panic!("worker {worker_id} never became routable"))
        }
        assert_eq!(wait_for_worker(&router, 0).await.len(), 1);

        tx.send(HashMap::from([
            (0, ModelRuntimeConfig::default()),
            (1, ModelRuntimeConfig::default()),
        ]))
        .unwrap();
        assert_eq!(wait_for_worker(&router, 1).await.len(), 2);
    }

    /// Three default-config workers under the registry policy, with
    /// `router_track_active_blocks` on so bookings send tracking hashes.
    async fn tracked_router(name: &str) -> KvRouter {
        // Prefill load decays with wall time and would let near-ties flip
        // between two runs microseconds apart.
        let config = KvRouterConfig {
            use_kv_events: false,
            router_track_prefill_tokens: false,
            skip_initial_worker_wait: true,
            ..Default::default()
        };
        let workers = (0..3)
            .map(|worker_id| (worker_id, ModelRuntimeConfig::default()))
            .collect();
        make_router(
            name,
            workers,
            2,
            SelectionPolicySource::Registry,
            None,
            None,
            "decode",
            config,
        )
        .await
        .unwrap()
    }

    /// Advisory best-match query with default routing arguments.
    async fn find_best_match(
        router: &KvRouter,
        tokens: &[u32],
        return_routing_hashes: bool,
    ) -> anyhow::Result<FindBestMatchOutcome> {
        router
            .find_best_match_details_with_policy_class(
                None,
                tokens,
                None,
                None,
                false,
                return_routing_hashes,
                None,
                None,
                0.0,
                0,
                None,
                None,
                None,
                None,
                None,
                RoutingConstraints::default(),
            )
            .await
    }

    /// Worker 1 has served prefix A and worker 2 prefix B, so every corpus
    /// prompt has one unambiguous best worker.
    async fn seed_golden_prefixes(router: &KvRouter) {
        for (worker_id, tokens) in [(1u64, golden_prefix(1)), (2, golden_prefix(2))] {
            router
                .record_routing_decision(
                    TokensWithHashes::new(tokens.clone(), 2),
                    WorkerWithDpRank::from_worker_id(worker_id),
                )
                .await
                .unwrap();
            // The approximate index applies recordings asynchronously; wait
            // until a query sees the whole prefix.
            tokio::time::timeout(std::time::Duration::from_secs(5), async {
                loop {
                    let FindBestMatchOutcome::Routed {
                        worker,
                        overlap_blocks,
                        ..
                    } = find_best_match(router, &tokens, false).await.unwrap()
                    else {
                        panic!("seeding query must route");
                    };
                    if worker.worker_id == worker_id && overlap_blocks == 8 {
                        break;
                    }
                    tokio::task::yield_now().await;
                }
            })
            .await
            .expect("seeded prefix became visible");
        }
    }

    fn golden_prefix(seed: u32) -> Vec<u32> {
        (0..16).map(|i| seed * 100 + i).collect()
    }

    /// Prompts sharing 4, 2, 8 (then continuing past it), 8, and 8 blocks
    /// with a seeded prefix. Partial matches come first so they are scored
    /// against the seeded index alone, before bookings add decode load.
    fn golden_corpus() -> Vec<Vec<u32>> {
        let a = golden_prefix(1);
        let b = golden_prefix(2);
        vec![
            a[..8].to_vec(),
            b[..4].iter().copied().chain(900..912).collect(),
            a.iter().copied().chain(1_000..1_008).collect(),
            a,
            b,
        ]
    }

    #[derive(Debug, PartialEq, serde::Serialize, serde::Deserialize)]
    struct GoldenDecision {
        worker_id: WorkerId,
        dp_rank: u32,
        overlap_blocks: u32,
        cached_tokens: usize,
        potential_decode_blocks: u64,
    }

    impl GoldenDecision {
        fn from_outcome(outcome: &FindBestMatchOutcome) -> Self {
            match outcome {
                FindBestMatchOutcome::Routed {
                    worker,
                    overlap_blocks,
                    cached_tokens,
                    potential_decode_blocks,
                    ..
                } => Self {
                    worker_id: worker.worker_id,
                    dp_rank: worker.dp_rank,
                    overlap_blocks: *overlap_blocks,
                    cached_tokens: *cached_tokens,
                    potential_decode_blocks: *potential_decode_blocks,
                },
                FindBestMatchOutcome::QueueRejected { rejection } => {
                    panic!("golden corpus must not be queue rejected: {rejection:?}")
                }
            }
        }
    }

    const GOLDEN_PATH: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/selection_golden/aggregated_tracked.json"
    );

    /// The selection procedure, run over the corpus with tracked bookings, must
    /// match the frozen trace. Regenerate it with `SELECTION_GOLDEN_UPDATE=1`
    /// only for an intended behavior change.
    #[tokio::test]
    async fn selection_matches_frozen_frontend_trace() {
        let router = tracked_router("golden").await;
        seed_golden_prefixes(&router).await;
        let mut trace = Vec::new();
        for (index, tokens) in golden_corpus().iter().enumerate() {
            let context_id = format!("golden-{index}");
            let admitted = router
                .find_best_match_details_with_policy_class_inner(
                    Some(&context_id),
                    tokens,
                    None,
                    None,
                    true,
                    true,
                    None,
                    None,
                    0.0,
                    0,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    RoutingConstraints::default(),
                    FindBestMatchAdmission::WithAdmission,
                )
                .await
                .unwrap();
            let FindBestMatchOutcome::Routed { routing_hashes, .. } = &admitted.outcome else {
                panic!("golden corpus must route");
            };
            assert!(routing_hashes.is_some(), "routing hashes were requested");
            trace.push(GoldenDecision::from_outcome(&admitted.outcome));
            // Keep the booking: later rows are scored against its load.
            let booking = admitted
                .booking
                .map(BookingHandle::commit)
                .expect("tracked selection carries its booking handle");
            assert_eq!(booking.request_id, context_id);
        }

        if std::env::var_os("SELECTION_GOLDEN_UPDATE").is_some() {
            std::fs::write(
                GOLDEN_PATH,
                serde_json::to_string_pretty(&trace).unwrap() + "\n",
            )
            .unwrap();
        }
        let golden: Vec<GoldenDecision> =
            serde_json::from_str(&std::fs::read_to_string(GOLDEN_PATH).unwrap()).unwrap();
        assert_eq!(
            trace, golden,
            "selection drifted from the frozen frontend trace"
        );
    }

    /// A lease released before `set_scheduler` is a wiring bug: it logs and
    /// leaves the booking alone. After `set_scheduler`, the release reaches
    /// the scheduler's booking cleanup.
    #[tokio::test]
    async fn request_lease_manager_releases_bookings_only_once_scheduler_is_set() {
        use dynamo_kv_router::scheduling::queue::SchedulerBookingDescriptor;

        let router = tracked_router("lease-manager").await;
        let worker = WorkerWithDpRank::from_worker_id(1);
        let scheduler = router.selection.scheduler();
        let book = |request_id: &'static str| async move {
            let attempt_id = scheduler
                .add_request_admitted(SequenceRequest {
                    request_id: request_id.to_string(),
                    token_sequence: None,
                    track_prefill_tokens: false,
                    expected_output_tokens: None,
                    prefill_load_hint: None,
                    worker,
                    lora_name: None,
                })
                .await
                .expect("booking");
            SchedulerBookingDescriptor {
                request_id: request_id.to_string(),
                worker,
                attempt_id,
            }
        };
        let cancel = CancellationToken::new();
        let manager = request_lease::RequestLeaseManager::new(cancel.child_token());

        let before = manager.register_local(book("before").await, None);
        drop(before);
        assert!(
            router.selection.scheduler().has_request("before"),
            "no scheduler to release through yet"
        );

        manager.set_scheduler(router.selection.scheduler().booking_cleanup());
        let after = manager.register_local(book("after").await, None);
        after.finish().await;
        assert!(!router.selection.scheduler().has_request("after"));
        assert!(router.selection.scheduler().has_request("before"));
        cancel.cancel();
    }

    /// A public enrollment cancelled while its routing update is in flight
    /// frees the booking through the detached enrollment's drop.
    #[tokio::test]
    async fn enroll_public_request_attempt_cancelled_during_routing_update_frees_booking() {
        use std::sync::atomic::{AtomicBool, Ordering};

        use dynamo_kv_router::indexer::TieredMatchDetails;
        use dynamo_kv_router::protocols::LocalBlockHash;
        use dynamo_kv_router::services::indexer::backend::RemotePrimary;

        /// Records park until `release` is notified.
        struct PausedRecord {
            entered: AtomicBool,
            release: tokio::sync::Notify,
        }
        #[async_trait]
        impl RemotePrimary for PausedRecord {
            async fn find_matches_by_tier(
                &self,
                _: Vec<LocalBlockHash>,
                _: bool,
            ) -> anyhow::Result<TieredMatchDetails> {
                Ok(TieredMatchDetails::default())
            }
            async fn record_routing_decision(
                &self,
                _: WorkerWithDpRank,
                _: RoutingDecisionHashes,
            ) -> anyhow::Result<()> {
                self.entered.store(true, Ordering::Release);
                self.release.notified().await;
                Ok(())
            }
            fn use_kv_events(&self) -> bool {
                false
            }
        }

        let mut router = tracked_router("enroll").await;
        let record = Arc::new(PausedRecord {
            entered: AtomicBool::new(false),
            release: tokio::sync::Notify::new(),
        });
        router.indexer = Indexer::Remote {
            primary: record.clone(),
            approx: None,
            primary_records_routing_decisions: true,
        };
        let router = router;
        let prompt = golden_prefix(1);
        let outcome = router
            .find_best_match_details_with_policy_class_inner(
                Some("cancelled"),
                &prompt,
                None,
                None,
                true,
                false,
                None,
                None,
                0.0,
                0,
                None,
                None,
                None,
                None,
                None,
                None,
                RoutingConstraints::default(),
                FindBestMatchAdmission::WithAdmission,
            )
            .await
            .unwrap();
        let admitted = outcome;
        let booking = admitted
            .booking
            .expect("tracked selection carries its booking handle");

        // Park the routing update after `register_detached` has run.
        let mut enroll = Box::pin(
            router.enroll_public_request_attempt(booking, Some(TokensWithHashes::new(prompt, 2))),
        );
        assert!(futures::poll!(&mut enroll).is_pending());
        assert!(
            record.entered.load(Ordering::Acquire),
            "routing update is in flight"
        );
        assert!(router.selection.scheduler().has_request("cancelled"));
        drop(enroll);
        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            while router.selection.scheduler().has_request("cancelled") {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("cancelled enrollment released its booking");
    }

    /// Two default-config workers, deterministic scoring, untracked bookings.
    async fn make_test_router(
        policy: SelectionPolicySource,
        shared_cache: Option<Arc<dyn SharedKvCache>>,
    ) -> KvRouter {
        let config = KvRouterConfig {
            overlap_score_credit: 0.0,
            router_temperature: 0.0,
            use_kv_events: false,
            router_track_active_blocks: false,
            shared_cache_multiplier: 0.5,
            skip_initial_worker_wait: true,
            ..Default::default()
        };
        let workers = HashMap::from([
            (0, ModelRuntimeConfig::default()),
            (1, ModelRuntimeConfig::default()),
        ]);
        make_router(
            "shared-cache-router",
            workers,
            2,
            policy,
            shared_cache,
            None,
            "decode",
            config,
        )
        .await
        .unwrap()
    }

    #[tokio::test]
    async fn test_find_best_match_passes_shared_cache_hits_to_scheduler() {
        let router = make_test_router(
            fixed_policy(Some(2), WorkerWithDpRank::from_worker_id(1)),
            Some(Arc::new(FakeSharedCache {
                #[allow(clippy::single_range_in_vec_init)]
                hits: Some(dynamo_kv_router::protocols::SharedCacheHits::from_ranges(
                    vec![0..2],
                )),
                should_error: false,
            })),
        )
        .await;

        let FindBestMatchOutcome::Routed {
            worker,
            overlap_blocks,
            ..
        } = find_best_match(&router, &[11, 12, 21, 22], false)
            .await
            .unwrap()
        else {
            panic!("expected routed outcome");
        };

        assert_eq!(worker, WorkerWithDpRank::from_worker_id(1));
        assert_eq!(overlap_blocks, 0);
    }

    #[tokio::test]
    async fn test_find_best_match_ignores_shared_cache_errors() {
        let router = make_test_router(
            fixed_policy(None, WorkerWithDpRank::from_worker_id(0)),
            Some(Arc::new(FakeSharedCache {
                hits: None,
                should_error: true,
            })),
        )
        .await;

        let FindBestMatchOutcome::Routed {
            worker,
            overlap_blocks,
            ..
        } = find_best_match(&router, &[11, 12, 21, 22], false)
            .await
            .unwrap()
        else {
            panic!("expected routed outcome");
        };

        assert_eq!(worker, WorkerWithDpRank::from_worker_id(0));
        assert_eq!(overlap_blocks, 0);
    }

    #[tokio::test]
    async fn test_find_best_match_maps_filtered_workers_to_unavailable() {
        let policy = SelectionPolicySource::Factory(Arc::new(|config: &KvRouterConfig, _, _| {
            WorkerSelectionPolicy::new_with_filters(
                config.clone(),
                "decode",
                vec![Box::new(RejectAll)],
                Vec::new(),
                Box::new(LoadOnlyPicker),
            )
        }));
        let router = make_test_router(policy, None).await;

        let Err(err) = find_best_match(&router, &[11, 12], false).await else {
            panic!("filtered workers must not route");
        };

        assert!(dynamo_runtime::error::match_error_chain(
            err.as_ref(),
            &[dynamo_runtime::error::ErrorType::Unavailable],
            &[]
        ));
    }

    #[tokio::test]
    async fn test_find_best_match_details_returns_routing_hashes_when_requested() {
        let router = make_test_router(
            fixed_policy(None, WorkerWithDpRank::from_worker_id(0)),
            None,
        )
        .await;
        let tokens = [11, 12, 21, 22];

        let outcome = find_best_match(&router, &tokens, true).await.unwrap();

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
    async fn test_get_overlap_scores_returns_tiered_rows_and_shared_hits() {
        let router = make_test_router(
            fixed_policy(None, WorkerWithDpRank::from_worker_id(0)),
            Some(Arc::new(FakeSharedCache {
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
